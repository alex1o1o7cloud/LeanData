import Mathlib

namespace race_course_length_proof_l273_273241

def race_course_length (L : ℝ) (v_A v_B : ℝ) : Prop :=
  v_A = 4 * v_B ∧ (L / v_A = (L - 66) / v_B) → L = 88

theorem race_course_length_proof (v_A v_B : ℝ) : race_course_length 88 v_A v_B :=
by 
  intros
  sorry

end race_course_length_proof_l273_273241


namespace Andy_earnings_l273_273393

/-- Andy's total earnings during an 8-hour shift. --/
theorem Andy_earnings (hours : ℕ) (hourly_wage : ℕ) (num_racquets : ℕ) (pay_per_racquet : ℕ)
  (num_grommets : ℕ) (pay_per_grommet : ℕ) (num_stencils : ℕ) (pay_per_stencil : ℕ)
  (h_shift : hours = 8) (h_hourly : hourly_wage = 9) (h_racquets : num_racquets = 7)
  (h_pay_racquets : pay_per_racquet = 15) (h_grommets : num_grommets = 2)
  (h_pay_grommets : pay_per_grommet = 10) (h_stencils : num_stencils = 5)
  (h_pay_stencils : pay_per_stencil = 1) :
  (hours * hourly_wage + num_racquets * pay_per_racquet + num_grommets * pay_per_grommet +
  num_stencils * pay_per_stencil) = 202 :=
by
  sorry

end Andy_earnings_l273_273393


namespace x_squared_plus_y_squared_geq_five_l273_273124

theorem x_squared_plus_y_squared_geq_five (x y : ℝ) (h : abs (x - 2 * y) = 5) : x^2 + y^2 ≥ 5 := 
sorry

end x_squared_plus_y_squared_geq_five_l273_273124


namespace sum_cubes_mod_l273_273885

theorem sum_cubes_mod (n : ℕ) : (1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 + 7^3 + 8^3 + 9^3 + 10^3) % 7 = 1 := by
  sorry

end sum_cubes_mod_l273_273885


namespace polygon_sides_equation_l273_273039

theorem polygon_sides_equation (n : ℕ) 
  (h1 : (n-2) * 180 = 4 * 360) : n = 10 := 
by 
  sorry

end polygon_sides_equation_l273_273039


namespace tropical_island_parrots_l273_273799

theorem tropical_island_parrots :
  let total_parrots := 150
  let red_fraction := 4 / 5
  let yellow_fraction := 1 - red_fraction
  let yellow_parrots := yellow_fraction * total_parrots
  yellow_parrots = 30 := sorry

end tropical_island_parrots_l273_273799


namespace final_limes_count_l273_273404

def limes_initial : ℕ := 9
def limes_by_Sara : ℕ := 4
def limes_used_for_juice : ℕ := 5
def limes_given_to_neighbor : ℕ := 3

theorem final_limes_count :
  limes_initial + limes_by_Sara - limes_used_for_juice - limes_given_to_neighbor = 5 :=
by
  sorry

end final_limes_count_l273_273404


namespace books_per_shelf_l273_273483

theorem books_per_shelf (total_books : ℕ) (total_shelves : ℕ) (h_books : total_books = 14240) (h_shelves : total_shelves = 1780) :
    total_books / total_shelves = 8 :=
by
  rw [h_books, h_shelves]
  norm_num

end books_per_shelf_l273_273483


namespace money_saved_l273_273292

noncomputable def total_savings :=
  let fox_price := 15
  let pony_price := 18
  let num_fox_pairs := 3
  let num_pony_pairs := 2
  let total_discount_rate := 0.22
  let pony_discount_rate := 0.10999999999999996
  let fox_discount_rate := total_discount_rate - pony_discount_rate
  let fox_savings := fox_price * fox_discount_rate * num_fox_pairs
  let pony_savings := pony_price * pony_discount_rate * num_pony_pairs
  fox_savings + pony_savings

theorem money_saved :
  total_savings = 8.91 :=
by
  -- We assume the savings calculations are correct as per the problem statement
  sorry

end money_saved_l273_273292


namespace B_days_finish_work_l273_273615

theorem B_days_finish_work :
  ∀ (W : ℝ) (A_work B_work B_days : ℝ),
  (A_work = W / 9) → 
  (B_work = W / B_days) →
  (3 * (W / 9) + 10 * (W / B_days) = W) →
  B_days = 15 :=
by
  intros W A_work B_work B_days hA_work hB_work hTotal
  sorry

end B_days_finish_work_l273_273615


namespace cos_alpha_plus_beta_l273_273380

variable {α β : ℝ}
variable (sin_alpha : Real.sin α = 3/5)
variable (cos_beta : Real.cos β = 4/5)
variable (α_interval : α ∈ Set.Ioo (Real.pi / 2) Real.pi)
variable (β_interval : β ∈ Set.Ioo 0 (Real.pi / 2))

theorem cos_alpha_plus_beta: Real.cos (α + β) = -1 :=
by
  sorry

end cos_alpha_plus_beta_l273_273380


namespace find_r_condition_l273_273004

variable {x y z w r : ℝ}

axiom h1 : x ≠ 0
axiom h2 : y ≠ 0
axiom h3 : z ≠ 0
axiom h4 : w ≠ 0
axiom h5 : (x ≠ y) ∧ (x ≠ z) ∧ (x ≠ w) ∧ (y ≠ z) ∧ (y ≠ w) ∧ (z ≠ w)

noncomputable def is_geometric_progression (a b c d : ℝ) (r : ℝ) : Prop :=
  b = a * r ∧ c = a * r^2 ∧ d = a * r^3

theorem find_r_condition :
  is_geometric_progression (x * (y - z)) (y * (z - x)) (z * (x - y)) (w * (y - x)) r →
  r^3 + r^2 + r + 1 = 0 :=
by
  intros
  sorry

end find_r_condition_l273_273004


namespace tip_percentage_is_20_l273_273833

noncomputable def total_bill : ℕ := 16 + 14
noncomputable def james_share : ℕ := total_bill / 2
noncomputable def james_paid : ℕ := 21
noncomputable def tip_amount : ℕ := james_paid - james_share
noncomputable def tip_percentage : ℕ := (tip_amount * 100) / total_bill 

theorem tip_percentage_is_20 :
  tip_percentage = 20 :=
by
  sorry

end tip_percentage_is_20_l273_273833


namespace percentage_of_water_in_dried_grapes_l273_273293

theorem percentage_of_water_in_dried_grapes 
  (weight_fresh : ℝ) 
  (weight_dried : ℝ) 
  (percentage_water_fresh : ℝ) 
  (solid_weight : ℝ)
  (water_weight_dried : ℝ) 
  (percentage_water_dried : ℝ) 
  (H1 : weight_fresh = 30) 
  (H2 : weight_dried = 15) 
  (H3 : percentage_water_fresh = 0.60) 
  (H4 : solid_weight = weight_fresh * (1 - percentage_water_fresh)) 
  (H5 : water_weight_dried = weight_dried - solid_weight) 
  (H6 : percentage_water_dried = (water_weight_dried / weight_dried) * 100) 
  : percentage_water_dried = 20 := 
  by { sorry }

end percentage_of_water_in_dried_grapes_l273_273293


namespace fraction_of_decimal_l273_273021

theorem fraction_of_decimal (a b : ℕ) (h : 0.375 = (a : ℝ) / (b : ℝ)) (gcd_ab : Nat.gcd a b = 1) : a + b = 11 :=
  sorry

end fraction_of_decimal_l273_273021


namespace front_view_heights_l273_273792

-- Define conditions
def column1 := [4, 2]
def column2 := [3, 0, 3]
def column3 := [1, 5]

-- Define a function to get the max height in each column
def max_height (col : List Nat) : Nat :=
  col.foldr Nat.max 0

-- Define the statement to prove the frontal view heights
theorem front_view_heights : 
  max_height column1 = 4 ∧ 
  max_height column2 = 3 ∧ 
  max_height column3 = 5 :=
by 
  sorry

end front_view_heights_l273_273792


namespace total_kids_played_l273_273568

theorem total_kids_played (kids_monday : ℕ) (kids_tuesday : ℕ) (h_monday : kids_monday = 4) (h_tuesday : kids_tuesday = 14) : 
  kids_monday + kids_tuesday = 18 := 
by
  -- proof steps here (for now, use sorry to skip the proof)
  sorry

end total_kids_played_l273_273568


namespace apples_sold_by_noon_l273_273634

theorem apples_sold_by_noon 
  (k g c l : ℕ) 
  (hk : k = 23) 
  (hg : g = 37) 
  (hc : c = 14) 
  (hl : l = 38) :
  k + g + c - l = 36 := 
by
  -- k = 23
  -- g = 37
  -- c = 14
  -- l = 38
  -- k + g + c - l = 36

  sorry

end apples_sold_by_noon_l273_273634


namespace tangent_line_intercept_l273_273401

theorem tangent_line_intercept :
  ∃ (m : ℚ) (b : ℚ), m > 0 ∧ b = 740 / 171 ∧
    ∀ (x y : ℚ), ((x - 1)^2 + (y - 3)^2 = 9 ∨ (x - 15)^2 + (y - 8)^2 = 100) →
                 (y = m * x + b) ↔ False := 
sorry

end tangent_line_intercept_l273_273401


namespace list_price_of_article_l273_273223

theorem list_price_of_article
  (P : ℝ)
  (discount1 : ℝ)
  (discount2 : ℝ)
  (final_price : ℝ)
  (h1 : discount1 = 0.10)
  (h2 : discount2 = 0.01999999999999997)
  (h3 : final_price = 61.74) :
  P = 70 :=
by
  sorry

end list_price_of_article_l273_273223


namespace min_value_proof_l273_273801

noncomputable def min_value_expression (a b : ℝ) : ℝ :=
  (1 / (12 * a + 1)) + (1 / (8 * b + 1))

theorem min_value_proof (a b : ℝ) (h1 : 3 * a + 2 * b = 1) (h2 : a ≠ 0) (h3 : b ≠ 0) :
  min_value_expression a b = 2 / 3 :=
sorry

end min_value_proof_l273_273801


namespace number_of_numbers_l273_273723

theorem number_of_numbers (n S : ℕ) 
  (h1 : (S + 26) / n = 15)
  (h2 : (S + 36) / n = 16)
  : n = 10 :=
sorry

end number_of_numbers_l273_273723


namespace truck_license_combinations_l273_273910

theorem truck_license_combinations :
  let letter_choices := 3
  let digit_choices := 10
  let number_of_digits := 6
  letter_choices * (digit_choices ^ number_of_digits) = 3000000 :=
by
  sorry

end truck_license_combinations_l273_273910


namespace quadratic_roots_l273_273656

theorem quadratic_roots (a b c : ℝ) (h1 : a ≠ 0) (h2 : a - b + c = 0) (h3 : (b^2 - 4 * a * c) = 0) : 2 * a - b = 0 :=
by {
  sorry
}

end quadratic_roots_l273_273656


namespace total_seats_in_stadium_l273_273730

theorem total_seats_in_stadium (people_at_game : ℕ) (empty_seats : ℕ) (total_seats : ℕ)
  (h1 : people_at_game = 47) (h2 : empty_seats = 45) :
  total_seats = people_at_game + empty_seats :=
by
  rw [h1, h2]
  show total_seats = 47 + 45
  sorry

end total_seats_in_stadium_l273_273730


namespace initial_antifreeze_percentage_l273_273907

-- Definitions of conditions
def total_volume : ℚ := 10
def replaced_volume : ℚ := 2.85714285714
def final_percentage : ℚ := 50 / 100

-- Statement to prove
theorem initial_antifreeze_percentage (P : ℚ) :
  10 * P / 100 - P / 100 * 2.85714285714 + 2.85714285714 = 5 → 
  P = 30 :=
sorry

end initial_antifreeze_percentage_l273_273907


namespace Ivy_cupcakes_l273_273832

theorem Ivy_cupcakes (M : ℕ) (h1 : M + (M + 15) = 55) : M = 20 :=
by
  sorry

end Ivy_cupcakes_l273_273832


namespace product_of_points_is_correct_l273_273096

def f (n : ℕ) : ℕ :=
  if n % 3 = 0 then 8
  else if n % 2 = 0 then 4
  else 0

def totalPoints (rolls : List ℕ) : ℕ :=
  rolls.map f |> List.sum

def AlexRolls := [6, 4, 3, 2, 1]
def BobRolls := [5, 6, 2, 3, 3]

def AlexPoints := totalPoints AlexRolls
def BobPoints := totalPoints BobRolls

theorem product_of_points_is_correct : AlexPoints * BobPoints = 672 := by
  sorry

end product_of_points_is_correct_l273_273096


namespace probability_of_intersecting_diagonals_l273_273364

noncomputable def intersecting_diagonals_probability : ℚ :=
let total_vertices := 8 in
let total_pairs := Nat.choose total_vertices 2 in
let total_sides := 8 in
let total_diagonals := total_pairs - total_sides in
let total_pairs_diagonals := Nat.choose total_diagonals 2 in
let intersecting_diagonals := Nat.choose total_vertices 4 in
(intersecting_diagonals : ℚ) / (total_pairs_diagonals : ℚ)

theorem probability_of_intersecting_diagonals :
  intersecting_diagonals_probability = 7 / 19 :=
by
  sorry

end probability_of_intersecting_diagonals_l273_273364


namespace find_locus_of_M_l273_273418

variables {P : Type*} [MetricSpace P] 
variables (A B C M : P)

def on_perpendicular_bisector (A B M : P) : Prop := 
  dist A M = dist B M

def on_circle (center : P) (radius : ℝ) (M : P) : Prop := 
  dist center M = radius

def M_AB (A B M : P) : Prop :=
  (on_perpendicular_bisector A B M) ∨ (on_circle A (dist A B) M) ∨ (on_circle B (dist A B) M)

def M_BC (B C M : P) : Prop :=
  (on_perpendicular_bisector B C M) ∨ (on_circle B (dist B C) M) ∨ (on_circle C (dist B C) M)

theorem find_locus_of_M :
  {M : P | M_AB A B M} ∩ {M : P | M_BC B C M} = {M : P | M_AB A B M ∧ M_BC B C M} :=
by sorry

end find_locus_of_M_l273_273418


namespace root_interval_sum_l273_273431

def f (x : ℝ) : ℝ := x^3 - x + 1

theorem root_interval_sum (a b : ℤ) (h1 : b - a = 1) (h2 : ∃! x : ℝ, a < x ∧ x < b ∧ f x = 0) : a + b = -3 :=
by
  sorry

end root_interval_sum_l273_273431


namespace total_marbles_l273_273154

theorem total_marbles (r b g : ℕ) (total : ℕ) 
  (h_ratio : 2 * g = 4 * b) 
  (h_blue_marbles : b = 36) 
  (h_total_formula : total = r + b + g) 
  : total = 108 :=
by
  sorry

end total_marbles_l273_273154


namespace correct_equation_l273_273357

-- Define the conditions
variables {x : ℝ}

-- Condition 1: The unit price of a notebook is 2 yuan less than that of a water-based pen.
def notebook_price (water_pen_price : ℝ) : ℝ := water_pen_price - 2

-- Condition 2: Xiaogang bought 5 notebooks and 3 water-based pens for exactly 14 yuan.
def total_cost (notebook_price water_pen_price : ℝ) : ℝ :=
  5 * notebook_price + 3 * water_pen_price

-- Question restated as a theorem: Verify the given equation is correct
theorem correct_equation (water_pen_price : ℝ) (h : total_cost (notebook_price water_pen_price) water_pen_price = 14) :
  5 * (water_pen_price - 2) + 3 * water_pen_price = 14 :=
  by
    -- Introduce the assumption
    intros
    -- Sorry to skip the proof
    sorry

end correct_equation_l273_273357


namespace distance_to_axes_l273_273219

def point (P : ℝ × ℝ) : Prop :=
  P = (3, 5)

theorem distance_to_axes (P : ℝ × ℝ) (hx : P = (3, 5)) : 
  abs P.2 = 5 ∧ abs P.1 = 3 :=
by 
  sorry

end distance_to_axes_l273_273219


namespace domain_of_log_function_l273_273932

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x - 1)

theorem domain_of_log_function : {
  x : ℝ // ∃ y : ℝ, f y = x
} = { x : ℝ | x > 1 / 2 } := by
sorry

end domain_of_log_function_l273_273932


namespace expected_value_of_white_balls_l273_273386

-- Definitions for problem conditions
def totalBalls : ℕ := 6
def whiteBalls : ℕ := 2
def redBalls : ℕ := 4
def ballsDrawn : ℕ := 2

-- Probability calculations
def P_X_0 : ℚ := (Nat.choose 4 2) / (Nat.choose totalBalls ballsDrawn)
def P_X_1 : ℚ := ((Nat.choose whiteBalls 1) * (Nat.choose redBalls 1)) / (Nat.choose totalBalls ballsDrawn)
def P_X_2 : ℚ := (Nat.choose whiteBalls 2) / (Nat.choose totalBalls ballsDrawn)

-- Expected value calculation
def expectedValue : ℚ := (0 * P_X_0) + (1 * P_X_1) + (2 * P_X_2)

theorem expected_value_of_white_balls :
  expectedValue = 2 / 3 :=
by
  sorry

end expected_value_of_white_balls_l273_273386


namespace dice_probability_exactly_four_twos_l273_273530

theorem dice_probability_exactly_four_twos :
  let probability := (Nat.choose 8 4 : ℚ) * (1 / 8)^4 * (7 / 8)^4 
  probability = 168070 / 16777216 :=
by
  sorry

end dice_probability_exactly_four_twos_l273_273530


namespace arithmetic_sequence_sums_l273_273564

variable (a : ℕ → ℕ)

-- Conditions
def condition1 := a 1 + a 4 + a 7 = 39
def condition2 := a 2 + a 5 + a 8 = 33

-- Question and expected answer
def result := a 3 + a 6 + a 9 = 27

theorem arithmetic_sequence_sums (h1 : condition1 a) (h2 : condition2 a) : result a := 
sorry

end arithmetic_sequence_sums_l273_273564


namespace largest_number_of_hcf_lcm_l273_273862

theorem largest_number_of_hcf_lcm (a b c : ℕ) (h : Nat.gcd (Nat.gcd a b) c = 42)
  (factor1 : 10 ∣ Nat.lcm (Nat.lcm a b) c)
  (factor2 : 20 ∣ Nat.lcm (Nat.lcm a b) c)
  (factor3 : 25 ∣ Nat.lcm (Nat.lcm a b) c)
  (factor4 : 30 ∣ Nat.lcm (Nat.lcm a b) c) :
  max (max a b) c = 1260 := 
  sorry

end largest_number_of_hcf_lcm_l273_273862


namespace num_triangles_l273_273777

def vertices := 10
def chosen_vertices := 3

theorem num_triangles : (Nat.choose vertices chosen_vertices) = 120 := by
  sorry

end num_triangles_l273_273777


namespace geometric_sequence_terms_l273_273132

theorem geometric_sequence_terms
  (a : ℚ) (l : ℚ) (r : ℚ) (n : ℕ)
  (h_a : a = 9 / 8)
  (h_l : l = 1 / 3)
  (h_r : r = 2 / 3)
  (h_geo : l = a * r^(n - 1)) :
  n = 4 :=
by
  sorry

end geometric_sequence_terms_l273_273132


namespace smallest_number_divisible_by_20_and_36_is_180_l273_273740

theorem smallest_number_divisible_by_20_and_36_is_180 :
  ∃ x, (x % 20 = 0) ∧ (x % 36 = 0) ∧ (∀ y, (y % 20 = 0) ∧ (y % 36 = 0) → x ≤ y) ∧ x = 180 :=
by
  sorry

end smallest_number_divisible_by_20_and_36_is_180_l273_273740


namespace volume_ratio_l273_273052

theorem volume_ratio (V1 V2 M1 M2 : ℝ)
  (h1 : M1 / (V1 - M1) = 1 / 2)
  (h2 : M2 / (V2 - M2) = 3 / 2)
  (h3 : (M1 + M2) / (V1 - M1 + V2 - M2) = 1) :
  V1 / V2 = 9 / 5 :=
by
  sorry

end volume_ratio_l273_273052


namespace total_revenue_correct_l273_273936

-- Define the conditions
def original_price_sneakers : ℝ := 80
def discount_sneakers : ℝ := 0.25
def pairs_sold_sneakers : ℕ := 2

def original_price_sandals : ℝ := 60
def discount_sandals : ℝ := 0.35
def pairs_sold_sandals : ℕ := 4

def original_price_boots : ℝ := 120
def discount_boots : ℝ := 0.4
def pairs_sold_boots : ℕ := 11

-- Compute discounted prices
def discounted_price (original_price : ℝ) (discount : ℝ) : ℝ :=
  original_price - (original_price * discount)

-- Compute revenue from each type of shoe
def revenue (price : ℝ) (pairs_sold : ℕ) : ℝ :=
  price * (pairs_sold : ℝ)

open Real

-- Main statement to prove
theorem total_revenue_correct : 
  revenue (discounted_price original_price_sneakers discount_sneakers) pairs_sold_sneakers + 
  revenue (discounted_price original_price_sandals discount_sandals) pairs_sold_sandals + 
  revenue (discounted_price original_price_boots discount_boots) pairs_sold_boots = 1068 := 
by
  sorry

end total_revenue_correct_l273_273936


namespace height_of_cone_l273_273094

theorem height_of_cone (e : ℝ) (bA : ℝ) (v : ℝ) :
  e = 6 ∧ bA = 54 ∧ v = e^3 → ∃ h : ℝ, (1/3) * bA * h = v ∧ h = 12 := by
  sorry

end height_of_cone_l273_273094


namespace prob_class1_two_mcq_from_A_expected_value_best_of_five_l273_273383

-- Part 1
theorem prob_class1_two_mcq_from_A :
  let P_B1 := (5.choose 2) / (8.choose 2)
  let P_B2 := (5.choose 1 * 3.choose 1) / (8.choose 2)
  let P_B3 := (3.choose 2) / (8.choose 2)
  let P_A_given_B1 := 6 / 9
  let P_A_given_B2 := 5 / 9
  let P_A_given_B3 := 4 / 9
  let P_A := P_B1 * P_A_given_B1 + P_B2 * P_A_given_B2 + P_B3 * P_A_given_B3
  let P_B1_given_A := (P_B1 * P_A_given_B1) / P_A
  P_B1_given_A = 20 / 49 :=
by
  sorry

-- Part 2
theorem expected_value_best_of_five :
  let P_X3 := (3/5 * 2/5 * 2/5) + (2/5 * 2/5 * 2/5)
  let P_X4 := 3/5 * 3/5 * 3/5 * 2/5 + 2/5 * 3/5 * 3/5 * 2/5 + 2/5 * 2/5 * 3/5 * 3/5 + 2/5 * 3/5 * 2/5 * 2/5 + 3/5 * 3/5 * 3/5 * 3/5 + 3/5 * 2/5 * 3/5 * 3/5
  let P_X5 := 1 - P_X3 - P_X4
  let E_X := 3 * P_X3 + 4 * P_X4 + 5 * P_X5
  E_X = 537 / 125 :=
by
  sorry

end prob_class1_two_mcq_from_A_expected_value_best_of_five_l273_273383


namespace simplify_tan_product_l273_273212

noncomputable def tan_deg (d : ℝ) : ℝ := Real.tan (d * Real.pi / 180)

theorem simplify_tan_product :
  (1 + tan_deg 10) * (1 + tan_deg 35) = 2 := 
by
  -- Given conditions
  have h1 : Real.tan (Real.pi / 4) = 1 := Real.tan_pi_div_four
  have h2 : tan_deg 10 + tan_deg 35 = 1 - tan_deg 10 * tan_deg 35 :=
    by sorry -- Use tan addition formula here
  -- Proof of the theorem follows from here
  sorry

end simplify_tan_product_l273_273212


namespace value_of_expression_l273_273699

-- Define the quadratic equation
def quadratic_eq (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Let p and q be roots of the quadratic equation
noncomputable def p : ℝ := sorry
noncomputable def q : ℝ := sorry

-- Theorem to prove that (3p - 4)(6q - 8) = -22 given p and q are roots of 3x^2 + 9x - 21 = 0
theorem value_of_expression (h1 : quadratic_eq 3 9 -21 p) (h2 : quadratic_eq 3 9 -21 q) :
  (3 * p - 4) * (6 * q - 8) = -22 :=
by
  sorry

end value_of_expression_l273_273699


namespace great_circle_bisects_angle_l273_273374

noncomputable def north_pole : Point := sorry
noncomputable def equator_point (C : Point) : Prop := sorry
noncomputable def great_circle_through (P Q : Point) : Circle := sorry
noncomputable def equidistant_from_N (A B N : Point) : Prop := sorry
noncomputable def spherical_triangle (A B C : Point) : Triangle := sorry
noncomputable def bisects_angle (C N A B : Point) : Prop := sorry

theorem great_circle_bisects_angle
  (N A B C: Point)
  (hN: N = north_pole)
  (hA: equidistant_from_N A B N)
  (hC: equator_point C)
  (hTriangle: spherical_triangle A B C)
  : bisects_angle C N A B :=
sorry

end great_circle_bisects_angle_l273_273374


namespace find_p_q_l273_273277

theorem find_p_q (p q : ℤ)
  (h : (5 * d^2 - 4 * d + p) * (4 * d^2 + q * d - 5) = 20 * d^4 + 11 * d^3 - 45 * d^2 - 20 * d + 25) :
  p + q = 3 :=
sorry

end find_p_q_l273_273277


namespace total_experiments_non_adjacent_l273_273048

theorem total_experiments_non_adjacent (n_org n_inorg n_add : ℕ) 
  (h_org : n_org = 3) (h_inorg : n_inorg = 2) (h_add : n_add = 2) 
  (no_adjacent : True) : 
  (n_org + n_inorg + n_add).factorial / (n_inorg + n_add).factorial * 
  (n_inorg + n_add + 1).choose n_org = 1440 :=
by
  -- The actual proof will go here.
  sorry

end total_experiments_non_adjacent_l273_273048


namespace num_distinct_pos_factors_81_l273_273811

-- Define what it means to be a factor
def is_factor (n d : ℕ) : Prop := d ∣ n

-- 81 is 3^4 by definition
def eighty_one : ℕ := 3 ^ 4

-- Define the list of positive factors of 81
def pos_factors_81 : List ℕ := [1, 3, 9, 27, 81]

-- Formal statement that 81 has 5 distinct positive factors
theorem num_distinct_pos_factors_81 : (pos_factors_81.length = 5) :=
by sorry

end num_distinct_pos_factors_81_l273_273811


namespace proportional_parts_middle_l273_273974

theorem proportional_parts_middle (x : ℚ) (hx : x + (1/2) * x + (1/4) * x = 120) : (1/2) * x = 240 / 7 :=
by
  sorry

end proportional_parts_middle_l273_273974


namespace solve_equation_l273_273343

theorem solve_equation : 
  ∀ x : ℝ,
    (x + 5 ≠ 0) → 
    (x^2 + 3 * x + 4) / (x + 5) = x + 6 → 
    x = -13 / 4 :=
by 
  intro x
  intro hx
  intro h
  sorry

end solve_equation_l273_273343


namespace modulo_sum_of_99_plus_5_l273_273370

theorem modulo_sum_of_99_plus_5 : let s_n := (99 / 2) * (2 * 1 + (99 - 1) * 1)
                                 let sum_with_5 := s_n + 5
                                 sum_with_5 % 7 = 6 :=
by
  sorry

end modulo_sum_of_99_plus_5_l273_273370


namespace probability_of_four_odd_slips_l273_273115

-- Define the conditions
def number_of_slips : ℕ := 10
def odd_slips : ℕ := 5
def even_slips : ℕ := 5
def slips_drawn : ℕ := 4

-- Define the required probability calculation
def probability_four_odd_slips : ℚ := (5 / 10) * (4 / 9) * (3 / 8) * (2 / 7)

-- State the theorem we want to prove
theorem probability_of_four_odd_slips :
  probability_four_odd_slips = 1 / 42 :=
by
  sorry

end probability_of_four_odd_slips_l273_273115


namespace domain_of_function_l273_273533

def f (x : ℝ) : ℝ := x^5 - 5 * x^4 + 10 * x^3 - 10 * x^2 + 5 * x - 1
def g (x : ℝ) : ℝ := x^2 - 9

theorem domain_of_function :
  {x : ℝ | g x ≠ 0} = {x : ℝ | x < -3} ∪ {x : ℝ | -3 < x ∧ x < 3} ∪ {x : ℝ | x > 3} :=
by
  sorry

end domain_of_function_l273_273533


namespace weight_increase_percentage_l273_273735

theorem weight_increase_percentage :
  ∀ (x : ℝ), (2 * x * 1.1 + 5 * x * 1.17 = 82.8) →
    ((82.8 - (2 * x + 5 * x)) / (2 * x + 5 * x)) * 100 = 15.06 := 
by 
  intro x 
  intro h
  sorry

end weight_increase_percentage_l273_273735


namespace deepak_share_l273_273919

theorem deepak_share (investment_Anand investment_Deepak total_profit : ℕ)
  (h₁ : investment_Anand = 2250) (h₂ : investment_Deepak = 3200) (h₃ : total_profit = 1380) :
  ∃ share_Deepak, share_Deepak = 810 := sorry

end deepak_share_l273_273919


namespace number_of_planters_l273_273460

variable (a b : ℕ)

-- Conditions
def tree_planting_condition_1 : Prop := a * b = 2013
def tree_planting_condition_2 : Prop := (a - 5) * (b + 2) < 2013
def tree_planting_condition_3 : Prop := (a - 5) * (b + 3) > 2013

-- Theorem stating the number of people who participated in the planting is 61
theorem number_of_planters (h1 : tree_planting_condition_1 a b) 
                           (h2 : tree_planting_condition_2 a b) 
                           (h3 : tree_planting_condition_3 a b) : 
                           a = 61 := 
sorry

end number_of_planters_l273_273460


namespace num_distinct_paintings_l273_273322

theorem num_distinct_paintings :
  let disks := Finset.range 7
  let blue := 4
  let red := 2
  let green := 1
  let total_paintings := (disks.card.choose blue) * ((disks.card - blue).choose red) * ((disks.card - blue - red).choose green)
  let reflection_fixed := 3
  let distinct_paintings := (total_paintings + reflection_fixed) / 2
  distinct_paintings = 54 :=
by {
  let disks := Finset.range 7,
  let blue := 4,
  let red := 2,
  let green := 1,
  let total_paintings := (disks.card.choose blue) * ((disks.card - blue).choose red) * ((disks.card - blue - red).choose green),
  let reflection_fixed := 3,
  let distinct_paintings := (total_paintings + reflection_fixed) / 2,
  have h_disks : disks.card = 7 := sorry,
  have h_total : total_paintings = 105 := sorry,
  have h_reflection : reflection_fixed = 3 := sorry,
  have h_final : distinct_paintings = 54 := by {
    rw [h_total, h_reflection],
    norm_num,
  },
  exact h_final,
}

end num_distinct_paintings_l273_273322


namespace min_value_of_a_l273_273136

theorem min_value_of_a (a b c : ℝ) (h₁ : a > 0) (h₂ : ∃ p q : ℝ, 0 < p ∧ p < 2 ∧ 0 < q ∧ q < 2 ∧ 
  ∀ x, ax^2 + bx + c = a * (x - p) * (x - q)) (h₃ : 25 * a + 10 * b + 4 * c ≥ 4) (h₄ : c ≥ 1) : 
  a ≥ 16 / 25 :=
sorry

end min_value_of_a_l273_273136


namespace cost_of_carton_l273_273593

-- Definition of given conditions
def totalCost : ℝ := 4.88
def numberOfCartons : ℕ := 4
def costPerCarton : ℝ := 1.22

-- The proof statement
theorem cost_of_carton
  (h : totalCost = 4.88) 
  (n : numberOfCartons = 4) :
  totalCost / numberOfCartons = costPerCarton := 
sorry

end cost_of_carton_l273_273593


namespace walnut_trees_planted_l273_273233

theorem walnut_trees_planted (initial_trees : ℕ) (final_trees : ℕ) (num_trees_planted : ℕ) : initial_trees = 107 → final_trees = 211 → num_trees_planted = final_trees - initial_trees → num_trees_planted = 104 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end walnut_trees_planted_l273_273233


namespace determine_values_l273_273276

theorem determine_values (x y : ℝ) (h1 : x - y = 25) (h2 : x * y = 36) : (x^2 + y^2 = 697) ∧ (x + y = Real.sqrt 769) :=
by
  -- Proof goes here
  sorry

end determine_values_l273_273276


namespace height_eight_times_initial_maximum_growth_year_l273_273908

noncomputable def t : ℝ := 2^(-2/3 : ℝ)
noncomputable def f (n : ℕ) (A a b t : ℝ) : ℝ := 9 * A / (a + b * t^n)

theorem height_eight_times_initial (A : ℝ) : 
  ∀ n : ℕ, f n A 1 8 t = 8 * A ↔ n = 9 :=
sorry

theorem maximum_growth_year (A : ℝ) :
  ∃ n : ℕ, (∀ k : ℕ, (f n A 1 8 t - f (n-1) A 1 8 t) ≥ (f k A 1 8 t - f (k-1) A 1 8 t))
  ∧ n = 5 :=
sorry

end height_eight_times_initial_maximum_growth_year_l273_273908


namespace ali_spending_ratio_l273_273917

theorem ali_spending_ratio
  (initial_amount : ℝ := 480)
  (remaining_amount : ℝ := 160)
  (F : ℝ)
  (H1 : (initial_amount - F - (1/3) * (initial_amount - F) = remaining_amount))
  : (F / initial_amount) = 1 / 2 :=
by
  sorry

end ali_spending_ratio_l273_273917


namespace percentage_shaded_l273_273692

theorem percentage_shaded (total_squares shaded_squares : ℕ) (h1 : total_squares = 5 * 5) (h2 : shaded_squares = 9) :
  (shaded_squares:ℚ) / total_squares * 100 = 36 :=
by
  sorry

end percentage_shaded_l273_273692


namespace successive_percentage_reduction_l273_273066

theorem successive_percentage_reduction (a b : ℝ) (h₁ : a = 25) (h₂ : b = 20) :
  a + b - (a * b) / 100 = 40 := by
  sorry

end successive_percentage_reduction_l273_273066


namespace polygon_sides_equation_l273_273040

theorem polygon_sides_equation (n : ℕ) 
  (h1 : (n-2) * 180 = 4 * 360) : n = 10 := 
by 
  sorry

end polygon_sides_equation_l273_273040


namespace find_m_range_l273_273959

noncomputable def p (m : ℝ) : Prop :=
  m < 1 / 3

noncomputable def q (m : ℝ) : Prop :=
  0 < m ∧ m < 15

theorem find_m_range (m : ℝ) :
  (¬(p m ∧ q m) ∧ (p m ∨ q m)) ↔ (1 / 3 ≤ m ∧ m < 15) :=
by
  sorry

end find_m_range_l273_273959


namespace intersection_equivalence_l273_273553

open Set

noncomputable def U : Set ℤ := {-2, -1, 0, 1, 2}
noncomputable def M : Set ℤ := {-1, 0, 1}
noncomputable def N : Set ℤ := {x | x * x - x - 2 = 0}
noncomputable def complement_M_in_U : Set ℤ := U \ M

theorem intersection_equivalence : (complement_M_in_U ∩ N) = {2} := 
by
  sorry

end intersection_equivalence_l273_273553


namespace cost_of_staying_23_days_l273_273065

def hostel_cost (days: ℕ) : ℝ :=
  if days ≤ 7 then
    days * 18
  else
    7 * 18 + (days - 7) * 14

theorem cost_of_staying_23_days : hostel_cost 23 = 350 :=
by
  sorry

end cost_of_staying_23_days_l273_273065


namespace num_factors_of_81_l273_273810

theorem num_factors_of_81 : (Nat.factors 81).toFinset.card = 5 := 
begin
  -- We know that 81 = 3^4
  -- Therefore, its distinct positive factors are {1, 3, 9, 27, 81}
  -- Hence the number of distinct positive factors is 5
  sorry
end

end num_factors_of_81_l273_273810


namespace find_n_l273_273287

theorem find_n : ∃ n : ℤ, 3^3 - 5 = 2^5 + n ∧ n = -10 :=
by
  sorry

end find_n_l273_273287


namespace Rebecca_group_count_l273_273853

def groupEggs (total_eggs number_of_eggs_per_group total_groups : Nat) : Prop :=
  total_groups = total_eggs / number_of_eggs_per_group

theorem Rebecca_group_count :
  groupEggs 8 2 4 :=
by
  sorry

end Rebecca_group_count_l273_273853


namespace sum_of_a5_a6_l273_273591

variable (a : ℕ → ℕ)

def S (n : ℕ) : ℕ :=
  n ^ 2 + 2

theorem sum_of_a5_a6 :
  a 5 + a 6 = S 6 - S 4 := by
  sorry

end sum_of_a5_a6_l273_273591


namespace ternary_to_decimal_121_l273_273107

theorem ternary_to_decimal_121 : 
  let t : ℕ := 1 * 3^2 + 2 * 3^1 + 1 * 3^0 
  in t = 16 :=
by
  sorry

end ternary_to_decimal_121_l273_273107


namespace inequality_subtraction_real_l273_273805

theorem inequality_subtraction_real (a b c : ℝ) (h : a > b) : a - c > b - c :=
sorry

end inequality_subtraction_real_l273_273805


namespace ThreePowFifteenModFive_l273_273739

def rem_div_3_pow_15_by_5 : ℕ :=
  let base := 3
  let mod := 5
  let exp := 15
  
  base^exp % mod

theorem ThreePowFifteenModFive (h1: 3^4 ≡ 1 [MOD 5]) : rem_div_3_pow_15_by_5 = 2 := by
  sorry

end ThreePowFifteenModFive_l273_273739


namespace student_weight_l273_273503

theorem student_weight (S W : ℕ) (h1 : S - 5 = 2 * W) (h2 : S + W = 104) : S = 71 :=
by {
  sorry
}

end student_weight_l273_273503


namespace sum_of_digits_l273_273198

theorem sum_of_digits (d : ℕ) (h1 : d % 5 = 0) (h2 : 3 * d - 75 = d) : 
  (d / 10 + d % 10) = 11 :=
by {
  -- Placeholder for the proof
  sorry
}

end sum_of_digits_l273_273198


namespace salad_dressing_oil_percentage_l273_273855

theorem salad_dressing_oil_percentage 
  (vinegar_P : ℝ) (vinegar_Q : ℝ) (oil_Q : ℝ)
  (new_vinegar : ℝ) (proportion_P : ℝ) :
  vinegar_P = 0.30 ∧ vinegar_Q = 0.10 ∧ oil_Q = 0.90 ∧ new_vinegar = 0.12 ∧ proportion_P = 0.10 →
  (1 - vinegar_P) = 0.70 :=
by
  intro h
  sorry

end salad_dressing_oil_percentage_l273_273855


namespace ball_attendance_l273_273173

theorem ball_attendance
  (n m : ℕ)
  (h_total : n + m < 50)
  (h_ladies_dance : 3 * n = 4 * (Ladies who danced invited)
  (h_gentlemen_invited : 5 * m = 7 * (Gentlemen who invited)
  (h_dance_pairs : 3 * n = 20 * m) :
  n + m = 41 :=
by sorry

end ball_attendance_l273_273173


namespace a_received_share_l273_273064

def a_inv : ℕ := 7000
def b_inv : ℕ := 11000
def c_inv : ℕ := 18000

def b_share : ℕ := 2200

def total_profit : ℕ := (b_share / (b_inv / 1000)) * 36
def a_ratio : ℕ := a_inv / 1000
def total_ratio : ℕ := (a_inv / 1000) + (b_inv / 1000) + (c_inv / 1000)

def a_share : ℕ := (a_ratio / total_ratio) * total_profit

theorem a_received_share :
  a_share = 1400 := 
sorry

end a_received_share_l273_273064


namespace budget_for_supplies_l273_273384

-- Conditions as definitions
def percentage_transportation := 20
def percentage_research_development := 9
def percentage_utilities := 5
def percentage_equipment := 4
def degrees_salaries := 216
def total_degrees := 360
def total_percentage := 100

-- Mathematical problem: Prove the percentage spent on supplies
theorem budget_for_supplies :
  (total_percentage - (percentage_transportation +
                       percentage_research_development +
                       percentage_utilities +
                       percentage_equipment) - 
   ((degrees_salaries * total_percentage) / total_degrees)) = 2 := by
  sorry

end budget_for_supplies_l273_273384


namespace misha_class_predictions_probability_l273_273077

-- Definitions representing the conditions
def monday_classes : ℕ := 5
def tuesday_classes : ℕ := 6
def total_classes : ℕ := monday_classes + tuesday_classes
def total_flips : ℕ := total_classes
def correct_predictions : ℕ := 7
def correct_monday_predictions : ℕ := 3

-- Lean theorem representing the proof problem
theorem misha_class_predictions_probability :
  (probability_of_correct_predictions total_flips correct_predictions monday_classes correct_monday_predictions) =
    (5 / 11) :=
  sorry

end misha_class_predictions_probability_l273_273077


namespace pelicans_among_non_egrets_is_47_percent_l273_273560

-- Definitions for the percentage of each type of bird.
def pelican_percentage : ℝ := 0.4
def cormorant_percentage : ℝ := 0.2
def egret_percentage : ℝ := 0.15
def osprey_percentage : ℝ := 0.25

-- Calculate the percentage of pelicans among the non-egret birds.
theorem pelicans_among_non_egrets_is_47_percent :
  (pelican_percentage / (1 - egret_percentage)) * 100 = 47 :=
by
  -- Detailed proof goes here
  sorry

end pelicans_among_non_egrets_is_47_percent_l273_273560


namespace remainder_division_l273_273934

theorem remainder_division (β : ℂ) 
  (h1 : β^6 + β^5 + β^4 + β^3 + β^2 + β + 1 = 0) 
  (h2 : β^7 = 1) : (β^100 + β^75 + β^50 + β^25 + 1) % (β^6 + β^5 + β^4 + β^3 + β^2 + β + 1) = -1 :=
by
  sorry

end remainder_division_l273_273934


namespace percentage_difference_l273_273501

theorem percentage_difference:
  let x1 := 0.4 * 60
  let x2 := 0.8 * 25
  x1 - x2 = 4 :=
by
  sorry

end percentage_difference_l273_273501


namespace parabola_complementary_slope_l273_273957

theorem parabola_complementary_slope
  (p x0 y0 x1 y1 x2 y2 : ℝ)
  (hp : p > 0)
  (hy0 : y0 > 0)
  (hP : y0^2 = 2 * p * x0)
  (hA : y1^2 = 2 * p * x1)
  (hB : y2^2 = 2 * p * x2)
  (h_slopes : (y1 - y0) / (x1 - x0) = - (2 * p / (y2 + y0))) :
  (y1 + y2) / y0 = -2 :=
by
  sorry

end parabola_complementary_slope_l273_273957


namespace number_of_drawings_on_first_page_l273_273443

-- Let D be the number of drawings on the first page.
variable (D : ℕ)

-- Conditions:
-- 1. D is the number of drawings on the first page.
-- 2. The number of drawings increases by 5 after every page.
-- 3. The total number of drawings in the first five pages is 75.

theorem number_of_drawings_on_first_page (h : D + (D + 5) + (D + 10) + (D + 15) + (D + 20) = 75) :
    D = 5 :=
by
  sorry

end number_of_drawings_on_first_page_l273_273443


namespace ball_attendance_l273_273170

theorem ball_attendance
  (n m : ℕ)
  (h_total : n + m < 50)
  (h_ladies_dance : 3 * n = 4 * (Ladies who danced invited)
  (h_gentlemen_invited : 5 * m = 7 * (Gentlemen who invited)
  (h_dance_pairs : 3 * n = 20 * m) :
  n + m = 41 :=
by sorry

end ball_attendance_l273_273170


namespace binomial_parameters_l273_273839

noncomputable def binomial_distribution : Type :=
  { n p : ℝ // 0 ≤ p ∧ p ≤ 1 ∧ n ∈ ℕ }

def E (b : binomial_distribution) : ℝ :=
  let ⟨n, p, _⟩ := b in n * p

def D (b : binomial_distribution) : ℝ :=
  let ⟨n, p, _⟩ := b in n * p * (1 - p)

theorem binomial_parameters :
  ∀ (X : binomial_distribution), E(X) = 2 ∧ D(X) = 4 → 
  let ⟨n, p, _⟩ := X in (n = 18 ∧ p = 2 / 3) :=
by {
  rintros ⟨n, p, hp⟩ ⟨hE, hD⟩,
  sorry
}

end binomial_parameters_l273_273839


namespace discount_is_15_point_5_percent_l273_273256

noncomputable def wholesale_cost (W : ℝ) := W
noncomputable def retail_price (W : ℝ) := 1.5384615384615385 * W
noncomputable def selling_price (W : ℝ) := 1.3 * W
noncomputable def discount_percentage (W : ℝ) := 
  let D := retail_price W - selling_price W
  (D / retail_price W) * 100

theorem discount_is_15_point_5_percent (W : ℝ) (hW : W > 0) : 
  discount_percentage W = 15.5 := 
by 
  sorry

end discount_is_15_point_5_percent_l273_273256


namespace algebraic_expression_value_l273_273302

theorem algebraic_expression_value (x y : ℝ) (h : x^2 - 4 * x - 1 = 0) : 
  (2 * x - 3) ^ 2 - (x + y) * (x - y) - y ^ 2 = 12 := 
by {
  sorry
}

end algebraic_expression_value_l273_273302


namespace correct_regression_line_l273_273877

theorem correct_regression_line (h_neg_corr: ∀ x: ℝ, ∀ y: ℝ, y = -10*x + 200 ∨ y = 10*x + 200 ∨ y = -10*x - 200 ∨ y = 10*x - 200) 
(h_slope_neg : ∀ a b: ℝ, a < 0) 
(h_y_intercept: ∀ x: ℝ, x = 0 → 200 > 0 → y = 200) : 
∃ y: ℝ, y = -10*x + 200 :=
by
-- the proof will go here
sorry

end correct_regression_line_l273_273877


namespace certain_number_is_gcd_l273_273869

theorem certain_number_is_gcd (x : ℕ) (h1 : ∃ k : ℕ, 72 * 14 = k * x) (h2 : x = Nat.gcd 1008 72) : x = 72 :=
sorry

end certain_number_is_gcd_l273_273869


namespace root_in_interval_iff_a_range_l273_273550

def f (a x : ℝ) : ℝ := 2 * a * x ^ 2 + 2 * x - 3 - a

theorem root_in_interval_iff_a_range (a : ℝ) :
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ f a x = 0) ↔ (1 ≤ a ∨ a ≤ - (3 + Real.sqrt 7) / 2) :=
sorry

end root_in_interval_iff_a_range_l273_273550


namespace max_students_received_less_than_given_l273_273398

def max_students_received_less := 27
def max_possible_n := 13

theorem max_students_received_less_than_given (n : ℕ) :
  n <= max_students_received_less -> n = max_possible_n :=
sorry
 
end max_students_received_less_than_given_l273_273398


namespace greatest_of_5_consec_even_numbers_l273_273006

-- Definitions based on the conditions
def avg_of_5_consec_even_numbers (N : ℤ) : ℤ := (N - 4 + N - 2 + N + N + 2 + N + 4) / 5

-- Proof statement
theorem greatest_of_5_consec_even_numbers (N : ℤ) (h : avg_of_5_consec_even_numbers N = 35) : N + 4 = 39 :=
by
  sorry -- proof is omitted

end greatest_of_5_consec_even_numbers_l273_273006


namespace darrel_will_receive_l273_273927

noncomputable def darrel_coins_value : ℝ := 
  let quarters := 127 
  let dimes := 183 
  let nickels := 47 
  let pennies := 237 
  let half_dollars := 64 
  let euros := 32 
  let pounds := 55 
  let quarter_fee_rate := 0.12 
  let dime_fee_rate := 0.07 
  let nickel_fee_rate := 0.15 
  let penny_fee_rate := 0.10 
  let half_dollar_fee_rate := 0.05 
  let euro_exchange_rate := 1.18 
  let euro_fee_rate := 0.03 
  let pound_exchange_rate := 1.39 
  let pound_fee_rate := 0.04 
  let quarters_value := 127 * 0.25 
  let quarters_fee := quarters_value * 0.12 
  let quarters_after_fee := quarters_value - quarters_fee 
  let dimes_value := 183 * 0.10 
  let dimes_fee := dimes_value * 0.07 
  let dimes_after_fee := dimes_value - dimes_fee 
  let nickels_value := 47 * 0.05 
  let nickels_fee := nickels_value * 0.15 
  let nickels_after_fee := nickels_value - nickels_fee 
  let pennies_value := 237 * 0.01 
  let pennies_fee := pennies_value * 0.10 
  let pennies_after_fee := pennies_value - pennies_fee 
  let half_dollars_value := 64 * 0.50 
  let half_dollars_fee := half_dollars_value * 0.05 
  let half_dollars_after_fee := half_dollars_value - half_dollars_fee 
  let euros_value := 32 * 1.18 
  let euros_fee := euros_value * 0.03 
  let euros_after_fee := euros_value - euros_fee 
  let pounds_value := 55 * 1.39 
  let pounds_fee := pounds_value * 0.04 
  let pounds_after_fee := pounds_value - pounds_fee 
  quarters_after_fee + dimes_after_fee + nickels_after_fee + pennies_after_fee + half_dollars_after_fee + euros_after_fee + pounds_after_fee

theorem darrel_will_receive : darrel_coins_value = 189.51 := by
  unfold darrel_coins_value
  sorry

end darrel_will_receive_l273_273927


namespace rice_mixture_ratio_l273_273390

-- Definitions for the given conditions
def cost_per_kg_rice1 : ℝ := 5
def cost_per_kg_rice2 : ℝ := 8.75
def cost_per_kg_mixture : ℝ := 7.50

-- The problem: ratio of two quantities
theorem rice_mixture_ratio (x y : ℝ) (h : cost_per_kg_rice1 * x + cost_per_kg_rice2 * y = 
                                     cost_per_kg_mixture * (x + y)) :
  y / x = 2 := 
sorry

end rice_mixture_ratio_l273_273390


namespace find_x0_l273_273659

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 - 1
else if x < 0 then -x^2 + 1
else 0

theorem find_x0 :
  ∃ x0 : ℝ, f x0 = 1/2 ∧ x0 = -Real.sqrt 2 / 2 :=
by
  sorry

end find_x0_l273_273659


namespace union_of_A_and_B_l273_273311

def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | x < 4}

theorem union_of_A_and_B : A ∪ B = {x | x > 1} := 
by 
  sorry

end union_of_A_and_B_l273_273311


namespace calculate_total_students_l273_273525

-- Define the conditions and state the theorem
theorem calculate_total_students (perc_bio : ℝ) (num_not_bio : ℝ) (perc_not_bio : ℝ) (T : ℝ) :
  perc_bio = 0.475 →
  num_not_bio = 462 →
  perc_not_bio = 1 - perc_bio →
  perc_not_bio * T = num_not_bio →
  T = 880 :=
by
  intros
  -- proof will be here
  sorry

end calculate_total_students_l273_273525


namespace initial_number_is_11_l273_273410

theorem initial_number_is_11 :
  ∃ (N : ℤ), ∃ (k : ℤ), N - 11 = 17 * k ∧ N = 11 :=
by
  sorry

end initial_number_is_11_l273_273410


namespace total_training_hours_l273_273235

-- Define Thomas's training conditions
def hours_per_day := 5
def days_initial := 30
def days_additional := 12
def total_days := days_initial + days_additional

-- State the theorem to be proved
theorem total_training_hours : total_days * hours_per_day = 210 :=
by
  sorry

end total_training_hours_l273_273235


namespace number_of_integers_l273_273536

theorem number_of_integers (n : ℤ) : 
  (20 < n^2) ∧ (n^2 < 150) → 
  (∃ m : ℕ, m = 16) :=
by
  sorry

end number_of_integers_l273_273536


namespace encode_MATHEMATICS_correct_l273_273631

-- Definitions for conditions
def encoded_map : char → string
| 'R' := "31"
| 'O' := "12"
| 'B' := "13"
| 'T' := "33"
| 'C' := "X" -- X represents unknown mapping to be determined
| 'D' := "X"
| 'E' := "X"
| 'G' := "X"
| 'H' := "XX"
| 'I' := "X"
| 'K' := "X"
| 'L' := "X"
| 'M' := "X"
| 'P' := "X"
| 'S' := "X"
| 'U' := "X"
| 'A' := "X"

-- Given encoding to "РОБОТ"
def encoded_ROBOT := encoded_map 'R' ++ encoded_map 'O' ++ encoded_map 'B' ++ encoded_map 'O' ++ encoded_map 'T'

-- Same encoding for "CROCODILE" and "HIPPOPOTAMUS"
def encoded_CROCODILE_HIPPOPOTAMUS := "XXXXXXX" -- Placeholder for the actual identical sequence

-- Encoding for MATHEMATICS
def encoded_MATHEMATICS := 
  encoded_map 'M' ++ encoded_map 'A' ++ encoded_map 'T' ++ encoded_map 'H' ++ encoded_map 'E' ++ 
  encoded_map 'M' ++ encoded_map 'A' ++ encoded_map 'T' ++ encoded_map 'I' ++ encoded_map 'C' ++ 
  encoded_map 'S'

-- Theorem to prove equivalence
theorem encode_MATHEMATICS_correct :
  encoded_MATHEMATICS = "2232331122323323132" :=
sorry

end encode_MATHEMATICS_correct_l273_273631


namespace polygon_sides_l273_273037

theorem polygon_sides (n : ℕ) 
  (h1 : (n - 2) * 180 = 4 * 360) : 
  n = 6 :=
by sorry

end polygon_sides_l273_273037


namespace find_p_q_sum_l273_273698

variable (P Q x : ℝ)

theorem find_p_q_sum (h : (P / (x - 3)) +  Q * (x + 2) = (-5 * x^2 + 18 * x + 40) / (x - 3)) : P + Q = 20 :=
sorry

end find_p_q_sum_l273_273698


namespace find_constant_c_and_t_l273_273782

noncomputable def exists_constant_c_and_t (c : ℝ) (t : ℝ) : Prop :=
∀ (x1 x2 m : ℝ), (x1^2 - m * x1 - c = 0) ∧ (x2^2 - m * x2 - c = 0) →
  (t = 1 / ((1 + m^2) * x1^2) + 1 / ((1 + m^2) * x2^2))

theorem find_constant_c_and_t : ∃ (c t : ℝ), exists_constant_c_and_t c t ∧ c = 2 ∧ t = 3 / 2 :=
sorry

end find_constant_c_and_t_l273_273782


namespace find_x_l273_273085

-- Define the conditions as variables and the target equation
variable (x : ℝ)

theorem find_x : 67 * x - 59 * x = 4828 → x = 603.5 := by
  intro h
  sorry

end find_x_l273_273085


namespace sum_factors_24_l273_273493

theorem sum_factors_24 : (∑ d in (finset.filter (λ d, 24 % d = 0) (finset.range (25))), d) = 60 :=
by
  sorry

end sum_factors_24_l273_273493


namespace cube_points_l273_273395

theorem cube_points (A B C D E F : ℕ) 
  (h1 : A + B = 13)
  (h2 : C + D = 13)
  (h3 : E + F = 13)
  (h4 : A + C + E = 16)
  (h5 : B + D + E = 24) :
  F = 6 :=
by
  sorry  -- Proof to be filled in by the user

end cube_points_l273_273395


namespace ronald_next_roll_l273_273714

-- Definition of previous rolls
def previous_rolls : List ℕ := [1, 3, 2, 4, 3, 5, 3, 4, 4, 2]

-- Function to calculate the sum of a list of numbers
def sum (l : List ℕ) : ℕ := l.foldr (+) 0

-- Function to calculate the required next roll
def required_next_roll (rolls : List ℕ) (average : ℕ) : ℕ :=
  let n := rolls.length + 1
  let required_sum := n * average
  required_sum - sum rolls

-- Theorem stating Ronald needs to roll a 2 on his next roll to have an average of 3
theorem ronald_next_roll : required_next_roll previous_rolls 3 = 2 := by
  sorry

end ronald_next_roll_l273_273714


namespace initial_books_in_bin_l273_273246

theorem initial_books_in_bin
  (x : ℝ)
  (h : x + 33.0 + 2.0 = 76) :
  x = 41.0 :=
by 
  -- Proof goes here
  sorry

end initial_books_in_bin_l273_273246


namespace original_price_l273_273762

variable (P SP : ℝ)

axiom condition1 : SP = 0.8 * P
axiom condition2 : SP = 480

theorem original_price : P = 600 :=
by
  sorry

end original_price_l273_273762


namespace days_per_week_l273_273580

def threeChildren := 3
def schoolYearWeeks := 25
def totalJuiceBoxes := 375

theorem days_per_week (d : ℕ) :
  (threeChildren * d * schoolYearWeeks = totalJuiceBoxes) → d = 5 :=
by
  sorry

end days_per_week_l273_273580


namespace six_to_2049_not_square_l273_273744

theorem six_to_2049_not_square
  (h1: ∃ x: ℝ, 1^2048 = x^2)
  (h2: ∃ x: ℝ, 2^2050 = x^2)
  (h3: ¬∃ x: ℝ, 6^2049 = x^2)
  (h4: ∃ x: ℝ, 4^2051 = x^2)
  (h5: ∃ x: ℝ, 5^2052 = x^2):
  ¬∃ y: ℝ, y^2 = 6^2049 := 
by sorry

end six_to_2049_not_square_l273_273744


namespace exists_line_intersecting_circle_and_passing_origin_l273_273306

theorem exists_line_intersecting_circle_and_passing_origin :
  ∃ m : ℝ, (m = 1 ∨ m = -4) ∧ 
  ∃ (x y : ℝ), 
    ((x - 1) ^ 2 + (y + 2) ^ 2 = 9) ∧ 
    ((x - y + m = 0) ∧ 
     ∃ (x' y' : ℝ),
      ((x' - 1) ^ 2 + (y' + 2) ^ 2 = 9) ∧ 
      ((x' - y' + m = 0) ∧ ((x + x') / 2 = 0 ∧ (y + y') / 2 = 0))) :=
by 
  sorry

end exists_line_intersecting_circle_and_passing_origin_l273_273306


namespace coin_flip_sequences_count_l273_273742

noncomputable def num_sequences_with_given_occurrences : ℕ :=
  sorry

theorem coin_flip_sequences_count : num_sequences_with_given_occurrences = 560 :=
  sorry

end coin_flip_sequences_count_l273_273742


namespace min_value_of_function_l273_273372

noncomputable def f (a x : ℝ) : ℝ := (a^x - a)^2 + (a^(-x) - a)^2

theorem min_value_of_function (a : ℝ) (h : a > 0) : ∃ x : ℝ, f a x = 2 :=
by
  sorry

end min_value_of_function_l273_273372


namespace sum_of_three_distinct_integers_product_625_l273_273478

theorem sum_of_three_distinct_integers_product_625 :
  ∃ a b c : ℤ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = 5^4 ∧ a + b + c = 131 :=
by
  sorry

end sum_of_three_distinct_integers_product_625_l273_273478


namespace segment_length_at_1_point_5_l273_273548

-- Definitions for the conditions
def Point := ℝ × ℝ
def Triangle (A B C : Point) := ∃ a b c : ℝ, a = 4 ∧ b = 3 ∧ c = 5 ∧ (A = (0, 0)) ∧ (B = (4, 0)) ∧ (C = (0, 3)) ∧ (c^2 = a^2 + b^2)

noncomputable def length_l (x : ℝ) : ℝ := (4 * (abs ((3/4) * x + 3))) / 5

theorem segment_length_at_1_point_5 (A B C : Point) (h : Triangle A B C) : 
  length_l 1.5 = 3.3 := by 
  sorry

end segment_length_at_1_point_5_l273_273548


namespace certain_number_is_gcd_l273_273870

theorem certain_number_is_gcd (x : ℕ) (h1 : ∃ k : ℕ, 72 * 14 = k * x) (h2 : x = Nat.gcd 1008 72) : x = 72 :=
sorry

end certain_number_is_gcd_l273_273870


namespace total_number_of_fish_l273_273459

-- Define the number of each type of fish
def goldfish : ℕ := 23
def blue_fish : ℕ := 15
def angelfish : ℕ := 8
def neon_tetra : ℕ := 12

-- Theorem stating the total number of fish
theorem total_number_of_fish : goldfish + blue_fish + angelfish + neon_tetra = 58 := by
  sorry

end total_number_of_fish_l273_273459


namespace student_calls_out_2005th_l273_273822

theorem student_calls_out_2005th : 
  ∀ (n : ℕ), n = 2005 → ∃ k : ℕ, k ∈ [1, 2, 3, 4, 3, 2, 1] ∧ k = 1 := 
by
  sorry

end student_calls_out_2005th_l273_273822


namespace max_students_gave_away_balls_more_l273_273396

theorem max_students_gave_away_balls_more (N : ℕ) (hN : N ≤ 13) : 
  ∃(students : ℕ), students = 27 ∧ (students = 27 ∧ N ≤ students - N) :=
by
  sorry

end max_students_gave_away_balls_more_l273_273396


namespace irrationals_among_examples_l273_273496

theorem irrationals_among_examples :
  ¬ ∃ (r : ℚ), r = π ∧
  (∃ (a b : ℚ), a * a = 4) ∧
  (∃ (r : ℚ), r = 0) ∧
  (∃ (r : ℚ), r = -22 / 7) := 
sorry

end irrationals_among_examples_l273_273496


namespace work_rate_problem_l273_273895

theorem work_rate_problem (A B : ℚ) (h1 : A + B = 1/8) (h2 : A = 1/12) : B = 1/24 :=
sorry

end work_rate_problem_l273_273895


namespace total_trip_hours_l273_273381

-- Define the given conditions
def speed1 := 50 -- Speed in mph for the first 4 hours
def time1 := 4 -- First 4 hours
def distance1 := speed1 * time1 -- Distance covered in the first 4 hours

def speed2 := 80 -- Speed in mph for additional hours
def average_speed := 65 -- Average speed for the entire trip

-- Define the proof problem
theorem total_trip_hours (T : ℕ) (A : ℕ) :
  distance1 + (speed2 * A) = average_speed * T ∧ T = time1 + A → T = 8 :=
by
  sorry

end total_trip_hours_l273_273381


namespace annual_yield_range_l273_273697

-- Here we set up the conditions as definitions in Lean 4
def last_year_range : ℝ := 10000
def improvement_rate : ℝ := 0.15

-- Theorems that are based on the conditions and need proving
theorem annual_yield_range (last_year_range : ℝ) (improvement_rate : ℝ) : 
  last_year_range * (1 + improvement_rate) = 11500 := 
sorry

end annual_yield_range_l273_273697


namespace students_taking_neither_l273_273453

-- Definitions based on conditions
def total_students : ℕ := 60
def students_CS : ℕ := 40
def students_Elec : ℕ := 35
def students_both_CS_and_Elec : ℕ := 25

-- Lean statement to prove the number of students taking neither computer science nor electronics
theorem students_taking_neither : total_students - (students_CS + students_Elec - students_both_CS_and_Elec) = 10 :=
by
  sorry

end students_taking_neither_l273_273453


namespace worker_b_alone_time_l273_273240

theorem worker_b_alone_time (A B C : ℝ) (h1 : A + B = 1 / 8)
  (h2 : A = 1 / 12) (h3 : C = 1 / 18) :
  1 / B = 24 :=
sorry

end worker_b_alone_time_l273_273240


namespace exist_elements_inequality_l273_273993

open Set

theorem exist_elements_inequality (A : Set ℝ) (a_1 a_2 a_3 a_4 : ℝ)
(hA : A = {a_1, a_2, a_3, a_4})
(h_ineq1 : 0 < a_1 )
(h_ineq2 : a_1 < a_2 )
(h_ineq3 : a_2 < a_3 )
(h_ineq4 : a_3 < a_4 ) :
∃ (x y : ℝ), x ∈ A ∧ y ∈ A ∧ (2 + Real.sqrt 3) * |x - y| < (x + 1) * (y + 1) + x * y := 
sorry

end exist_elements_inequality_l273_273993


namespace line_tangent_to_ellipse_l273_273649

theorem line_tangent_to_ellipse (k : ℝ) :
  (∃ x : ℝ, 2 * x ^ 2 + 8 * (k * x + 2) ^ 2 = 8 ∧
             ∀ x1 x2 : ℝ, (2 + 8 * k ^ 2) * x1 ^ 2 + 32 * k * x1 + 24 = 0 →
             (2 + 8 * k ^ 2) * x2 ^ 2 + 32 * k * x2 + 24 = 0 → x1 = x2) →
  k^2 = 3 / 4 := by
  sorry

end line_tangent_to_ellipse_l273_273649


namespace parallel_line_through_A_is_2x_3y_minus_15_line_with_twice_slope_angle_l273_273670

open Real

-- Conditions:
def l1 (x y : ℝ) : Prop := x - 2 * y + 3 = 0
def l2 (x y : ℝ) : Prop := x + 2 * y - 9 = 0
def intersection_point (x y : ℝ) : Prop := l1 x y ∧ l2 x y

-- Point A is the intersection of l1 and l2
def A : ℝ × ℝ := ⟨3, 3⟩

-- Question 1
def line_parallel (x y : ℝ) (c : ℝ) : Prop := 2 * x + 3 * y + c = 0
def line_parallel_passing_through_A : Prop := line_parallel A.1 A.2 (-15)

theorem parallel_line_through_A_is_2x_3y_minus_15 : line_parallel_passing_through_A :=
sorry

-- Question 2
def slope_angle (tan_alpha : ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  ∃ y, ∃ x, y ≠ 0 ∧ l x y ∧ (tan_alpha = x / y)

def required_slope (tan_alpha : ℝ) : Prop :=
  tan_alpha = 4 / 3

def line_with_slope (x y slope : ℝ) : Prop :=
  y - A.2 = slope * (x - A.1)

def line_with_required_slope : Prop := 
  line_with_slope A.1 A.2 (4 / 3)

theorem line_with_twice_slope_angle : line_with_required_slope :=
sorry

end parallel_line_through_A_is_2x_3y_minus_15_line_with_twice_slope_angle_l273_273670


namespace polygon_sides_equation_l273_273041

theorem polygon_sides_equation (n : ℕ) 
  (h1 : (n-2) * 180 = 4 * 360) : n = 10 := 
by 
  sorry

end polygon_sides_equation_l273_273041


namespace average_monthly_balance_l273_273648

def january_balance : ℕ := 100
def february_balance : ℕ := 200
def march_balance : ℕ := 150
def april_balance : ℕ := 150
def may_balance : ℕ := 180
def number_of_months : ℕ := 5
def total_balance : ℕ := january_balance + february_balance + march_balance + april_balance + may_balance

theorem average_monthly_balance :
  (january_balance + february_balance + march_balance + april_balance + may_balance) / number_of_months = 156 := by
  sorry

end average_monthly_balance_l273_273648


namespace weaving_problem_l273_273318

theorem weaving_problem
  (a : ℕ → ℝ) -- the sequence
  (a_arith_seq : ∀ n, a n = a 0 + n * (a 1 - a 0)) -- arithmetic sequence condition
  (sum_seven_days : 7 * a 0 + 21 * (a 1 - a 0) = 21) -- sum in seven days
  (sum_days_2_5_8 : 3 * a 1 + 12 * (a 1 - a 0) = 15) -- sum on 2nd, 5th, and 8th days
  : a 10 = 15 := sorry

end weaving_problem_l273_273318


namespace max_divisor_f_l273_273950

def f (n : ℕ) : ℕ := (2 * n + 7) * 3^n + 9

theorem max_divisor_f (m : ℕ) : (∀ n : ℕ, m ∣ f n) → m = 36 :=
sorry

end max_divisor_f_l273_273950


namespace polygon_sides_l273_273034

theorem polygon_sides (n : ℕ) 
  (h_interior : (n - 2) * 180 = 4 * 360) : 
  n = 10 :=
by
  sorry

end polygon_sides_l273_273034


namespace mark_cans_l273_273624

theorem mark_cans (R J M : ℕ) (h1 : J = 2 * R + 5) (h2 : M = 4 * J) (h3 : R + J + M = 135) : M = 100 :=
by
  sorry

end mark_cans_l273_273624


namespace original_profit_percentage_l273_273087

noncomputable def originalCost : ℝ := 80
noncomputable def P := 30
noncomputable def profitPercentage : ℝ := ((100 - originalCost) / originalCost) * 100

theorem original_profit_percentage:
  ∀ (S C : ℝ),
  C = originalCost →
  ( ∀ (newCost : ℝ),
    newCost = 0.8 * C →
    ∀ (newSell : ℝ),
    newSell = S - 16.8 →
    newSell = 1.3 * newCost → P = 30 ) →
  profitPercentage = 25 := sorry

end original_profit_percentage_l273_273087


namespace max_chickens_ducks_l273_273570

theorem max_chickens_ducks (x y : ℕ) 
  (h1 : ∀ (k : ℕ), k = 6 → x + y - 6 ≥ 2) 
  (h2 : ∀ (k : ℕ), k = 9 → y ≥ 1) : 
  x + y ≤ 12 :=
sorry

end max_chickens_ducks_l273_273570


namespace find_k_l273_273326

theorem find_k (a : ℕ → ℕ) (h₀ : a 1 = 2) (h₁ : ∀ m n, a (m + n) = a m * a n) (hk : a (k + 1) = 1024) : k = 9 := 
sorry

end find_k_l273_273326


namespace main_theorem_l273_273143

variable (x : ℤ)

def H : ℤ := 12 - (3 + 7) + x
def T : ℤ := 12 - 3 + 7 + x

theorem main_theorem : H - T + x = -14 + x :=
by
  sorry

end main_theorem_l273_273143


namespace peanuts_difference_is_correct_l273_273166

-- Define the number of peanuts Jose has
def Jose_peanuts : ℕ := 85

-- Define the number of peanuts Kenya has
def Kenya_peanuts : ℕ := 133

-- Define the difference in the number of peanuts between Kenya and Jose
def peanuts_difference : ℕ := Kenya_peanuts - Jose_peanuts

-- Prove that the number of peanuts Kenya has minus the number of peanuts Jose has is equal to 48
theorem peanuts_difference_is_correct : peanuts_difference = 48 := by
  sorry

end peanuts_difference_is_correct_l273_273166


namespace units_digit_p_plus_one_l273_273131

theorem units_digit_p_plus_one (p : ℕ) (h1 : p % 2 = 0) (h2 : p % 10 ≠ 0)
  (h3 : (p ^ 3) % 10 = (p ^ 2) % 10) : (p + 1) % 10 = 7 :=
  sorry

end units_digit_p_plus_one_l273_273131


namespace PQRS_product_l273_273415

theorem PQRS_product :
  let P := (Real.sqrt 2012 + Real.sqrt 2013)
  let Q := (-Real.sqrt 2012 - Real.sqrt 2013)
  let R := (Real.sqrt 2012 - Real.sqrt 2013)
  let S := (Real.sqrt 2013 - Real.sqrt 2012)
  P * Q * R * S = 1 :=
by
  sorry

end PQRS_product_l273_273415


namespace triangle_least_perimeter_l273_273475

theorem triangle_least_perimeter (x : ℤ) (h1 : x + 27 > 34) (h2 : 34 + 27 > x) (h3 : x + 34 > 27) : 27 + 34 + x ≥ 69 :=
by
  have h1' : x > 7 := by linarith
  sorry

end triangle_least_perimeter_l273_273475


namespace not_sufficient_nor_necessary_l273_273572

theorem not_sufficient_nor_necessary (a b : ℝ) :
  ¬((a^2 > b^2) → (a > b)) ∧ ¬((a > b) → (a^2 > b^2)) :=
by
  sorry

end not_sufficient_nor_necessary_l273_273572


namespace solve_for_b_l273_273472

theorem solve_for_b (b : ℝ) : 
  let slope1 := -(3 / 4 : ℝ)
  let slope2 := -(b / 6 : ℝ)
  slope1 * slope2 = -1 → b = -8 :=
by
  intro h
  sorry

end solve_for_b_l273_273472


namespace admission_counts_l273_273321

-- Define the total number of ways to admit students under given conditions.
def ways_of_admission : Nat := 1518

-- Statement of the problem: given conditions, prove the result
theorem admission_counts (n_colleges : Nat) (n_students : Nat) (admitted_two_colleges : Bool) : 
  n_colleges = 23 → 
  n_students = 3 → 
  admitted_two_colleges = true →
  ways_of_admission = 1518 :=
by
  intros
  sorry

end admission_counts_l273_273321


namespace total_glasses_l273_273266

theorem total_glasses
  (x y : ℕ)
  (h1 : y = x + 16)
  (h2 : (12 * x + 16 * y) / (x + y) = 15) :
  12 * x + 16 * y = 480 :=
by
  sorry

end total_glasses_l273_273266


namespace g_value_at_6_l273_273084

noncomputable def g (v : ℝ) : ℝ :=
  let x := (v + 2) / 4
  x^2 - x + 2

theorem g_value_at_6 :
  g 6 = 4 := by
  sorry

end g_value_at_6_l273_273084


namespace expression_value_zero_l273_273983

theorem expression_value_zero (a b c : ℝ) (h : a^2 + b = b^2 + c ∧ b^2 + c = c^2 + a) : 
  a * (a^2 - b^2) + b * (b^2 - c^2) + c * (c^2 - a^2) = 0 := by
  sorry

end expression_value_zero_l273_273983


namespace marks_change_factor_l273_273585

def total_marks (n : ℕ) (avg : ℝ) : ℝ := n * avg

theorem marks_change_factor 
  (n : ℕ) (initial_avg new_avg : ℝ) 
  (initial_total := total_marks n initial_avg) 
  (new_total := total_marks n new_avg)
  (h1 : initial_avg = 36)
  (h2 : new_avg = 72)
  (h3 : n = 12):
  (new_total / initial_total) = 2 :=
by
  sorry

end marks_change_factor_l273_273585


namespace middle_integer_of_consecutive_odd_l273_273356

theorem middle_integer_of_consecutive_odd (n : ℕ)
  (h1 : n > 2)
  (h2 : n < 8)
  (h3 : (n-2) % 2 = 1)
  (h4 : n % 2 = 1)
  (h5 : (n+2) % 2 = 1)
  (h6 : (n-2) + n + (n+2) = (n-2) * n * (n+2) / 9) :
  n = 5 :=
by sorry

end middle_integer_of_consecutive_odd_l273_273356


namespace sqrt_calc1_sqrt_calc2_l273_273101

-- Problem 1 proof statement
theorem sqrt_calc1 : ( (Real.sqrt 2 + Real.sqrt 3) ^ 2 - (Real.sqrt 5 + 2) * (Real.sqrt 5 - 2) = 4 + 2 * Real.sqrt 6 ) :=
  sorry

-- Problem 2 proof statement
theorem sqrt_calc2 : ( (2 - Real.sqrt 3) ^ 2023 * (2 + Real.sqrt 3) ^ 2023 - 2 * abs (-Real.sqrt 3 / 2) - (-Real.sqrt 2) ^ 0 = -Real.sqrt 3 ) :=
  sorry

end sqrt_calc1_sqrt_calc2_l273_273101


namespace hannah_payment_l273_273142

def costWashingMachine : ℝ := 100
def costDryer : ℝ := costWashingMachine - 30
def totalCostBeforeDiscount : ℝ := costWashingMachine + costDryer
def discount : ℝ := totalCostBeforeDiscount * 0.1
def finalCost : ℝ := totalCostBeforeDiscount - discount

theorem hannah_payment : finalCost = 153 := by
  simp [costWashingMachine, costDryer, totalCostBeforeDiscount, discount, finalCost]
  sorry

end hannah_payment_l273_273142


namespace ball_attendance_l273_273185

noncomputable def num_people_attending_ball (n m : ℕ) : ℕ := n + m 

theorem ball_attendance:
  ∀ (n m : ℕ), 
  n + m < 50 ∧ 
  (n - n / 4) = (5 * m / 7) →
  num_people_attending_ball n m = 41 :=
by 
  intros n m h
  have h1 : n + m < 50 := h.1
  have h2 : n - n / 4 = 5 * m / 7 := h.2
  sorry

end ball_attendance_l273_273185


namespace complement_U_P_l273_273423

def U (y : ℝ) : Prop := y > 0
def P (y : ℝ) : Prop := 0 < y ∧ y < 1/3

theorem complement_U_P :
  {y : ℝ | U y} \ {y : ℝ | P y} = {y : ℝ | y ≥ 1/3} :=
by
  sorry

end complement_U_P_l273_273423


namespace quadratic_function_difference_zero_l273_273802

theorem quadratic_function_difference_zero
  (a b c x1 x2 x3 x4 x5 p q : ℝ)
  (h1 : a ≠ 0)
  (h2 : a * x1^2 + b * x1 + c = 5)
  (h3 : a * (x2 + x3 + x4 + x5)^2 + b * (x2 + x3 + x4 + x5) + c = 5)
  (h4 : x1 ≠ x2 + x3 + x4 + x5)
  (h5 : a * (x1 + x2)^2 + b * (x1 + x2) + c = p)
  (h6 : a * (x3 + x4 + x5)^2 + b * (x3 + x4 + x5) + c = q) :
  p - q = 0 := 
sorry

end quadratic_function_difference_zero_l273_273802


namespace isosceles_with_base_c_l273_273729

theorem isosceles_with_base_c (a b c: ℝ) (h: a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) (triangle_rel: 1/a - 1/b + 1/c = 1/(a - b + c)) : a = c ∨ b = c :=
sorry

end isosceles_with_base_c_l273_273729


namespace max_value_abs_cube_sum_l273_273549

theorem max_value_abs_cube_sum (x : Fin 5 → ℝ) (h : ∀ i, 0 ≤ x i ∧ x i ≤ 1) : 
  (|x 0 - x 1|^3 + |x 1 - x 2|^3 + |x 2 - x 3|^3 + |x 3 - x 4|^3 + |x 4 - x 0|^3) ≤ 4 :=
sorry

end max_value_abs_cube_sum_l273_273549


namespace necessary_but_not_sufficient_l273_273901

theorem necessary_but_not_sufficient (x : ℝ) : ( (x + 1) * (x + 2) > 0 → (x + 1) * (x^2 + 2) > 0 ) :=
by
  intro h
  -- insert steps urther here, if proof was required
  sorry

end necessary_but_not_sufficient_l273_273901


namespace cost_of_seven_books_l273_273359

theorem cost_of_seven_books (h : 3 * 12 = 36) : 7 * 12 = 84 :=
sorry

end cost_of_seven_books_l273_273359


namespace evaluate_expression_l273_273117

theorem evaluate_expression : 60 + (105 / 15) + (25 * 16) - 250 + (324 / 9) ^ 2 = 1513 := by
  sorry

end evaluate_expression_l273_273117


namespace num_integers_satisfying_ineq_count_integers_satisfying_ineq_l273_273538

theorem num_integers_satisfying_ineq (k : ℤ) :
  (20 < k^2 ∧ k^2 < 150) ↔ k ∈ ({-12, -11, -10, -9, -8, -7, -6, -5, 5, 6, 7, 8, 9, 10, 11, 12} : set ℤ) := by
  sorry

theorem count_integers_satisfying_ineq :
  {n : ℤ | 20 < n^2 ∧ n^2 < 150}.finite.to_finset.card = 16 := by
  sorry

end num_integers_satisfying_ineq_count_integers_satisfying_ineq_l273_273538


namespace mutually_exclusive_A_C_l273_273414

-- Definitions based on the given conditions
def all_not_defective (A : Prop) : Prop := A
def all_defective (B : Prop) : Prop := B
def at_least_one_defective (C : Prop) : Prop := C

-- Theorem to prove A and C are mutually exclusive
theorem mutually_exclusive_A_C (A B C : Prop) 
  (H1 : all_not_defective A) 
  (H2 : all_defective B) 
  (H3 : at_least_one_defective C) : 
  (A ∧ C) → False :=
sorry

end mutually_exclusive_A_C_l273_273414


namespace larger_box_can_carry_more_clay_l273_273614

variable {V₁ : ℝ} -- Volume of the first box
variable {V₂ : ℝ} -- Volume of the second box
variable {m₁ : ℝ} -- Mass the first box can carry
variable {m₂ : ℝ} -- Mass the second box can carry

-- Defining the dimensions of the first box.
def height₁ : ℝ := 1
def width₁ : ℝ := 2
def length₁ : ℝ := 4

-- Defining the dimensions of the second box.
def height₂ : ℝ := 3 * height₁
def width₂ : ℝ := 2 * width₁
def length₂ : ℝ := 2 * length₁

-- Volume calculation for the first box.
def volume₁ : ℝ := height₁ * width₁ * length₁

-- Volume calculation for the second box.
def volume₂ : ℝ := height₂ * width₂ * length₂

-- Condition: The first box can carry 30 grams of clay
def mass₁ : ℝ := 30

-- Given the above conditions, prove the second box can carry 360 grams of clay.
theorem larger_box_can_carry_more_clay (h₁ : volume₁ = height₁ * width₁ * length₁)
                                      (h₂ : volume₂ = height₂ * width₂ * length₂)
                                      (h₃ : mass₁ = 30)
                                      (h₄ : V₁ = volume₁)
                                      (h₅ : V₂ = volume₂) :
  m₂ = 12 * mass₁ := by
  -- Skipping the detailed proof.
  sorry

end larger_box_can_carry_more_clay_l273_273614


namespace neg_neg_two_neg_six_plus_six_neg_three_times_five_two_x_minus_three_x_l273_273098

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

end neg_neg_two_neg_six_plus_six_neg_three_times_five_two_x_minus_three_x_l273_273098


namespace midpoints_collinear_l273_273448

variables {E : Ellipse}
variables {r1 r2 r3 : Line}

-- Assume r1, r2, r3 are parallel
axiom parallel_lines : Parallel r1 r2 ∧ Parallel r2 r3

-- Assume r1, r2, r3 intersect ellipse E at points A1, B1, A2, B2, A3, B3 respectively
variables (A1 B1 A2 B2 A3 B3 : Point)

axiom intersections : 
  (A1, B1) ∈ (E ∩ r1) ∧ 
  (A2, B2) ∈ (E ∩ r2) ∧ 
  (A3, B3) ∈ (E ∩ r3)

-- Define midpoints of the segments (Ai, Bi)
def midpoint (P Q : Point) : Point := 
  ⟨(P.x + Q.x) / 2, (P.y + Q.y) / 2⟩

variable m1 := midpoint A1 B1
variable m2 := midpoint A2 B2
variable m3 := midpoint A3 B3

-- Statement to prove: the midpoints m1, m2, m3 are collinear
theorem midpoints_collinear : collinear {m1, m2, m3} :=
sorry

end midpoints_collinear_l273_273448


namespace roots_product_l273_273057

theorem roots_product : (27^(1/3) * 81^(1/4) * 64^(1/6)) = 18 := 
by
  sorry

end roots_product_l273_273057


namespace pencils_cost_proportion_l273_273267

/-- 
If a set of 15 pencils costs 9 dollars and the price of the set is directly 
proportional to the number of pencils it contains, then the cost of a set of 
35 pencils is 21 dollars.
--/
theorem pencils_cost_proportion :
  ∀ (p : ℕ), (∀ n : ℕ, n * 9 = p * 15) -> (35 * 9 = 21 * 15) :=
by
  intro p h1
  sorry

end pencils_cost_proportion_l273_273267


namespace retailer_marked_price_percentage_above_cost_l273_273514

noncomputable def cost_price : ℝ := 100
noncomputable def discount_rate : ℝ := 0.15
noncomputable def sales_profit_rate : ℝ := 0.275

theorem retailer_marked_price_percentage_above_cost :
  ∃ (MP : ℝ), ((MP - cost_price) / cost_price = 0.5) ∧ (((MP * (1 - discount_rate)) - cost_price) / cost_price = sales_profit_rate) :=
sorry

end retailer_marked_price_percentage_above_cost_l273_273514


namespace hotel_made_correct_revenue_l273_273265

noncomputable def hotelRevenue : ℕ :=
  let totalRooms := 260
  let doubleRooms := 196
  let singleRoomCost := 35
  let doubleRoomCost := 60
  let singleRooms := totalRooms - doubleRooms
  let revenueSingleRooms := singleRooms * singleRoomCost
  let revenueDoubleRooms := doubleRooms * doubleRoomCost
  revenueSingleRooms + revenueDoubleRooms

theorem hotel_made_correct_revenue :
  hotelRevenue = 14000 := by
  sorry

end hotel_made_correct_revenue_l273_273265


namespace range_of_a_l273_273426

def A (a : ℝ) := ({-1, 0, a} : Set ℝ)
def B := {x : ℝ | 0 < x ∧ x < 1}

theorem range_of_a (a : ℝ) (h : A a ∩ B ≠ ∅) : 0 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l273_273426


namespace mixtape_first_side_songs_l273_273638

theorem mixtape_first_side_songs (total_length : ℕ) (second_side_songs : ℕ) (song_length : ℕ) :
  total_length = 40 → second_side_songs = 4 → song_length = 4 → (total_length - second_side_songs * song_length) / song_length = 6 := 
by
  intros h1 h2 h3
  sorry

end mixtape_first_side_songs_l273_273638


namespace not_divisible_by_81_l273_273852

theorem not_divisible_by_81 (n : ℤ) : ¬ (81 ∣ n^3 - 9 * n + 27) :=
sorry

end not_divisible_by_81_l273_273852


namespace ratio_eq_one_l273_273502

variable {a b : ℝ}

theorem ratio_eq_one (h1 : 7 * a = 8 * b) (h2 : a * b ≠ 0) : (a / 8) / (b / 7) = 1 := 
by
  sorry

end ratio_eq_one_l273_273502


namespace initial_red_marbles_l273_273689

variable (r g : ℝ)

def red_green_ratio_initial (r g : ℝ) : Prop := r / g = 5 / 3
def red_green_ratio_new (r g : ℝ) : Prop := (r + 15) / (g - 9) = 3 / 1

theorem initial_red_marbles (r g : ℝ) (h₁ : red_green_ratio_initial r g) (h₂ : red_green_ratio_new r g) : r = 52.5 := sorry

end initial_red_marbles_l273_273689


namespace joanna_reading_rate_l273_273594

variable (P : ℝ)

theorem joanna_reading_rate (h : 3 * P + 6.5 * P + 6 * P = 248) : P = 16 := by
  sorry

end joanna_reading_rate_l273_273594


namespace quadratic_inequality_false_range_l273_273353

theorem quadratic_inequality_false_range (a : ℝ) :
  (¬ ∀ x : ℝ, a * x^2 - 2 * a * x + 3 > 0) ↔ (a < 0 ∨ a ≥ 3) :=
by
  sorry

end quadratic_inequality_false_range_l273_273353


namespace geometric_sequence_sixth_term_l273_273470

/-- 
The statement: 
The first term of a geometric sequence is 1000, and the 8th term is 125. Prove that the positive,
real value for the 6th term is 31.25.
-/
theorem geometric_sequence_sixth_term :
  ∀ (a1 a8 a6 : ℝ) (r : ℝ),
    a1 = 1000 →
    a8 = 125 →
    a8 = a1 * r^7 →
    a6 = a1 * r^5 →
    a6 = 31.25 :=
by
  intros a1 a8 a6 r h1 h2 h3 h4
  sorry

end geometric_sequence_sixth_term_l273_273470


namespace simplify_expression_l273_273001

variable (x : ℝ)

theorem simplify_expression (h : x ≠ 0) : x⁻¹ - 3 * x + 2 = - (3 * x^2 - 2 * x - 1) / x :=
by
  sorry

end simplify_expression_l273_273001


namespace nonnegative_integer_solutions_l273_273352

theorem nonnegative_integer_solutions (x : ℕ) :
  2 * x - 1 < 5 ↔ x = 0 ∨ x = 1 ∨ x = 2 := by
sorry

end nonnegative_integer_solutions_l273_273352


namespace num_ordered_pairs_l273_273675

open Real 

-- Define the conditions
def eq_condition (x y : ℕ) : Prop :=
  x * (sqrt y) + y * (sqrt x) + (sqrt (2006 * x * y)) - (sqrt (2006 * x)) - (sqrt (2006 * y)) - 2006 = 0

-- Define the main problem statement
theorem num_ordered_pairs : ∃ (n : ℕ), n = 8 ∧ (∀ (x y : ℕ), eq_condition x y → x * y = 2006) :=
by
  sorry

end num_ordered_pairs_l273_273675


namespace find_k_of_parallelepiped_volume_l273_273481

theorem find_k_of_parallelepiped_volume 
  (k : ℝ) 
  (h_pos : k > 0)
  (h_volume : Abs (3 * k^2 - 11 * k + 6) = 20) : 
  k = 14 / 3 := 
sorry

end find_k_of_parallelepiped_volume_l273_273481


namespace geometric_sequence_general_term_l273_273952

theorem geometric_sequence_general_term (a : ℕ → ℝ) (h1 : a 1 = 3) (h4 : a 4 = 81) :
  ∃ q : ℝ, (a n = 3 * q ^ (n - 1)) := by
  sorry

end geometric_sequence_general_term_l273_273952


namespace peaches_left_l273_273644

/-- Brenda picks 3600 peaches, 37.5% are fresh, and 250 are disposed of. Prove that Brenda has 1100 peaches left. -/
theorem peaches_left (total_peaches : ℕ) (percent_fresh : ℚ) (peaches_disposed : ℕ) (h1 : total_peaches = 3600) (h2 : percent_fresh = 3 / 8) (h3 : peaches_disposed = 250) : 
  total_peaches * percent_fresh - peaches_disposed = 1100 := 
by
  sorry

end peaches_left_l273_273644


namespace price_reduction_equation_l273_273616

variable (x : ℝ)

theorem price_reduction_equation :
    (58 * (1 - x)^2 = 43) :=
sorry

end price_reduction_equation_l273_273616


namespace total_yield_UncleLi_yield_difference_l273_273053

-- Define the conditions related to Uncle Li and Aunt Lin
def UncleLiAcres : ℕ := 12
def UncleLiYieldPerAcre : ℕ := 660
def AuntLinAcres : ℕ := UncleLiAcres - 2
def AuntLinTotalYield : ℕ := UncleLiYieldPerAcre * UncleLiAcres - 420

-- Prove the total yield of Uncle Li's rice
theorem total_yield_UncleLi : UncleLiYieldPerAcre * UncleLiAcres = 7920 := by
  sorry

-- Prove how much less the yield per acre of Uncle Li's rice is compared to Aunt Lin's
theorem yield_difference :
  UncleLiYieldPerAcre - AuntLinTotalYield / AuntLinAcres = 90 := by
  sorry

end total_yield_UncleLi_yield_difference_l273_273053


namespace polygon_sides_l273_273031

theorem polygon_sides (n : ℕ) 
  (h_interior : (n - 2) * 180 = 4 * 360) : 
  n = 10 :=
by
  sorry

end polygon_sides_l273_273031


namespace yuna_initial_pieces_l273_273452

variable (Y : ℕ)

theorem yuna_initial_pieces
  (namjoon_initial : ℕ := 250)
  (given_pieces : ℕ := 60)
  (namjoon_after : namjoon_initial - given_pieces = Y + given_pieces - 20) :
  Y = 150 :=
by
  sorry

end yuna_initial_pieces_l273_273452


namespace proportion1_proportion2_l273_273344

theorem proportion1 (x : ℚ) : (x / (5 / 9) = (1 / 20) / (1 / 3)) → x = 1 / 12 :=
sorry

theorem proportion2 (x : ℚ) : (x / 0.25 = 0.5 / 0.1) → x = 1.25 :=
sorry

end proportion1_proportion2_l273_273344


namespace coordinates_of_point_B_l273_273305

theorem coordinates_of_point_B (A B : ℝ × ℝ) (AB : ℝ) :
  A = (-1, 2) ∧ B.1 = -1 ∧ AB = 3 ∧ (B.2 = 5 ∨ B.2 = -1) :=
by
  sorry

end coordinates_of_point_B_l273_273305


namespace ratio_correct_l273_273228

theorem ratio_correct : 
    (2^17 * 3^19) / (6^18) = 3 / 2 :=
by sorry

end ratio_correct_l273_273228


namespace product_of_radii_l273_273755

-- Definitions based on the problem conditions
def passes_through (a : ℝ) (C : ℝ × ℝ) : Prop :=
  (C.1 - a)^2 + (C.2 - a)^2 = a^2

def tangent_to_axes (a : ℝ) : Prop :=
  a > 0

def circle_radii_roots (a b : ℝ) : Prop :=
  a^2 - 14 * a + 25 = 0 ∧ b^2 - 14 * b + 25 = 0

-- Theorem statement to prove the product of the radii
theorem product_of_radii (a r1 r2 : ℝ) (h1 : passes_through a (3, 4)) (h2 : tangent_to_axes a) (h3 : circle_radii_roots r1 r2) : r1 * r2 = 25 :=
by
  sorry

end product_of_radii_l273_273755


namespace translated_B_is_B_l273_273436

def point : Type := ℤ × ℤ

def A : point := (-4, -1)
def A' : point := (-2, 2)
def B : point := (1, 1)
def B' : point := (3, 4)

def translation_vector (p1 p2 : point) : point :=
  (p2.1 - p1.1, p2.2 - p1.2)

def translate_point (p : point) (v : point) : point :=
  (p.1 + v.1, p.2 + v.2)

theorem translated_B_is_B' : translate_point B (translation_vector A A') = B' :=
by
  sorry

end translated_B_is_B_l273_273436


namespace negation_universal_to_existential_l273_273149

-- Setup the necessary conditions and types
variable (a : ℝ) (ha : 0 < a ∧ a < 1)

-- Negate the universal quantifier
theorem negation_universal_to_existential :
  (¬ ∀ x < 0, a^x > 1) ↔ ∃ x_0 < 0, a^(x_0) ≤ 1 :=
by sorry

end negation_universal_to_existential_l273_273149


namespace seq_a_n_value_l273_273666

theorem seq_a_n_value (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (h1 : ∀ n, S n = n^2)
  (h2 : ∀ n ≥ 2, a n = S n - S (n-1)) :
  a 10 = 19 :=
sorry

end seq_a_n_value_l273_273666


namespace max_value_of_2a_plus_b_l273_273844

variable (a b : ℝ)

def cond1 := 4 * a + 3 * b ≤ 10
def cond2 := 3 * a + 5 * b ≤ 11

theorem max_value_of_2a_plus_b : 
  cond1 a b → 
  cond2 a b → 
  2 * a + b ≤ 48 / 11 := 
by 
  sorry

end max_value_of_2a_plus_b_l273_273844


namespace students_in_line_l273_273216

theorem students_in_line (between : ℕ) (Yoojung Eunji : ℕ) (h1 : Yoojung = 1) (h2 : Eunji = 1) : 
  between + Yoojung + Eunji = 16 :=
  sorry

end students_in_line_l273_273216


namespace equation1_solution_equation2_solution_equation3_solution_l273_273463

theorem equation1_solution :
  ∀ x : ℝ, x^2 + 4 * x = 0 ↔ x = 0 ∨ x = -4 :=
by
  sorry

theorem equation2_solution :
  ∀ x : ℝ, 2 * (x - 1) + x * (x - 1) = 0 ↔ x = 1 ∨ x = -2 :=
by
  sorry

theorem equation3_solution :
  ∀ x : ℝ, 3 * x^2 - 2 * x - 4 = 0 ↔ x = (1 + Real.sqrt 13) / 3 ∨ x = (1 - Real.sqrt 13) / 3 :=
by
  sorry

end equation1_solution_equation2_solution_equation3_solution_l273_273463


namespace find_larger_box_ounces_l273_273454

-- Define the conditions
def ounces_smaller_box : ℕ := 20
def cost_smaller_box : ℝ := 3.40
def cost_larger_box : ℝ := 4.80
def best_value_price_per_ounce : ℝ := 0.16

-- Define the question and its expected answer
def expected_ounces_larger_box : ℕ := 30

-- Proof statement
theorem find_larger_box_ounces :
  (cost_larger_box / best_value_price_per_ounce = expected_ounces_larger_box) :=
by
  sorry

end find_larger_box_ounces_l273_273454


namespace remainder_division_l273_273118

theorem remainder_division (G Q1 R1 Q2 : ℕ) (hG : G = 88)
  (h1 : 3815 = G * Q1 + R1) (h2 : 4521 = G * Q2 + 33) : R1 = 31 :=
sorry

end remainder_division_l273_273118


namespace union_of_S_and_T_l273_273129

-- Declare sets S and T
def S : Set ℕ := {3, 4, 5}
def T : Set ℕ := {4, 7, 8}

-- Statement about their union
theorem union_of_S_and_T : S ∪ T = {3, 4, 5, 7, 8} :=
sorry

end union_of_S_and_T_l273_273129


namespace find_angle_B_l273_273828

-- Definitions and conditions
variables (α β γ δ : ℝ) -- representing angles ∠A, ∠B, ∠C, and ∠D

-- Given Condition: it's a parallelogram and sum of angles A and C
def quadrilateral_parallelogram (A B C D : ℝ) : Prop :=
  A + C = 200 ∧ A = C ∧ A + B = 180

-- Theorem: Degree of angle B is 80°
theorem find_angle_B (A B C D : ℝ) (h : quadrilateral_parallelogram A B C D) : B = 80 := 
  by sorry

end find_angle_B_l273_273828


namespace simplify_tan_product_l273_273211

noncomputable def tan_deg (d : ℝ) : ℝ := Real.tan (d * Real.pi / 180)

theorem simplify_tan_product :
  (1 + tan_deg 10) * (1 + tan_deg 35) = 2 := 
by
  -- Given conditions
  have h1 : Real.tan (Real.pi / 4) = 1 := Real.tan_pi_div_four
  have h2 : tan_deg 10 + tan_deg 35 = 1 - tan_deg 10 * tan_deg 35 :=
    by sorry -- Use tan addition formula here
  -- Proof of the theorem follows from here
  sorry

end simplify_tan_product_l273_273211


namespace average_difference_correct_l273_273222

def daily_diff : List ℤ := [15, 0, -15, 25, 5, -5, 10]
def number_of_days : ℤ := 7

theorem average_difference_correct :
  (daily_diff.sum : ℤ) / number_of_days = 5 := by
  sorry

end average_difference_correct_l273_273222


namespace carrots_not_used_l273_273758

variable (totalCarrots : ℕ)
variable (ratioBeforeLunch : ℝ)
variable (ratioByEndOfDay : ℝ)

theorem carrots_not_used (h1 : totalCarrots = 300)
    (h2 : ratioBeforeLunch = 2 / 5)
    (h3 : ratioByEndOfDay = 3 / 5) :
    let carrotsUsedBeforeLunch := ratioBeforeLunch * totalCarrots
        remainingCarrotsAfterLunch := totalCarrots - carrotsUsedBeforeLunch
        carrotsUsedByEndOfDay := ratioByEndOfDay * remainingCarrotsAfterLunch
        carrotsNotUsed := remainingCarrotsAfterLunch - carrotsUsedByEndOfDay
    in carrotsNotUsed = 72 :=
by
  -- the detailed proof steps will go here
  sorry

end carrots_not_used_l273_273758


namespace smallest_base10_num_exists_l273_273888

noncomputable def A_digit_range := {n : ℕ // n < 6}
noncomputable def B_digit_range := {n : ℕ // n < 8}

theorem smallest_base10_num_exists :
  ∃ (A : A_digit_range) (B : B_digit_range), 7 * A.val = 9 * B.val ∧ 7 * A.val = 63 :=
by
  sorry

end smallest_base10_num_exists_l273_273888


namespace find_fraction_eq_l273_273898

theorem find_fraction_eq 
  {x : ℚ} 
  (h : x / (2 / 3) = (3 / 5) / (6 / 7)) : 
  x = 7 / 15 :=
by
  sorry

end find_fraction_eq_l273_273898


namespace negation_proposition_l273_273227

theorem negation_proposition :
  (¬ (∀ x : ℝ, ∃ n : ℕ, n > 0 ∧ n ≥ x)) ↔ (∃ x : ℝ, ∀ n : ℕ, n > 0 → n < x^2) := 
by
  sorry

end negation_proposition_l273_273227


namespace polygon_sides_l273_273028

theorem polygon_sides (n : ℕ) :
  ((n - 2) * 180 = 4 * 360) → n = 10 :=
by 
  sorry

end polygon_sides_l273_273028


namespace prime_sum_probability_l273_273650

open Finset

def first_ten_primes : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

def draw_two_without_replacement : Finset (ℕ × ℕ) := 
  (first_ten_primes.product first_ten_primes).filter (λ pair, pair.1 ≠ pair.2)

def is_prime_sum (pair : ℕ × ℕ) : Bool := Nat.prime (pair.1 + pair.2)

noncomputable def probability_prime_sum : ℚ :=
  (draw_two_without_replacement.filter (λ pair, is_prime_sum pair)).card.toRat / 
  draw_two_without_replacement.card.toRat

theorem prime_sum_probability : probability_prime_sum = 1 / 9 := by
  sorry

end prime_sum_probability_l273_273650


namespace solve_for_x_l273_273002

theorem solve_for_x : ∃ x : ℤ, 25 - 7 = 3 + x ∧ x = 15 := by
  sorry

end solve_for_x_l273_273002


namespace inequality_solution_l273_273949

variable {a b : ℝ}

theorem inequality_solution
  (h1 : a < 0)
  (h2 : -1 < b ∧ b < 0) :
  ab > ab^2 ∧ ab^2 > a := 
sorry

end inequality_solution_l273_273949


namespace tv_power_consumption_l273_273837

-- Let's define the problem conditions
def hours_per_day : ℕ := 4
def days_per_week : ℕ := 7
def weekly_cost : ℝ := 49              -- in cents
def cost_per_kwh : ℝ := 14             -- in cents

-- Define the theorem to prove the TV power consumption is 125 watts per hour
theorem tv_power_consumption : (weekly_cost / cost_per_kwh) / (hours_per_day * days_per_week) * 1000 = 125 :=
by
  sorry

end tv_power_consumption_l273_273837


namespace room_length_l273_273473

theorem room_length (length width rate cost : ℝ)
    (h_width : width = 3.75)
    (h_rate : rate = 1000)
    (h_cost : cost = 20625)
    (h_eq : cost = length * width * rate) :
    length = 5.5 :=
by
  -- the proof will go here
  sorry

end room_length_l273_273473


namespace fraction_meaningful_l273_273880

theorem fraction_meaningful (x : ℝ) : x - 5 ≠ 0 ↔ x ≠ 5 := 
by 
  sorry

end fraction_meaningful_l273_273880


namespace part1_part2_a_eq_1_part2_a_gt_1_part2_a_lt_1_l273_273245

section part1
variable (x : ℝ)

theorem part1 (a : ℝ) (h : a = 2) : (x + a) * (x - 2 * a + 1) < 0 ↔ -2 < x ∧ x < 3 :=
by
  sorry
end part1

section part2
variable (x a : ℝ)

-- Case: a = 1
theorem part2_a_eq_1 (h : a = 1) : (x - 1) * (x - 2 * a + 1) < 0 ↔ False :=
by
  sorry

-- Case: a > 1
theorem part2_a_gt_1 (h : a > 1) : (x - 1) * (x - 2 * a + 1) < 0 ↔ 1 < x ∧ x < 2 * a - 1 :=
by
  sorry

-- Case: a < 1
theorem part2_a_lt_1 (h : a < 1) : (x - 1) * (x - 2 * a + 1) < 0 ↔ 2 * a - 1 < x ∧ x < 1 :=
by
  sorry
end part2

end part1_part2_a_eq_1_part2_a_gt_1_part2_a_lt_1_l273_273245


namespace inequality_proof_l273_273804

variables (a b c : ℝ)

theorem inequality_proof (h : a > b) : a * c^2 ≥ b * c^2 :=
by sorry

end inequality_proof_l273_273804


namespace largest_multiple_of_11_less_than_100_l273_273882

theorem largest_multiple_of_11_less_than_100 : 
  ∀ n, n < 100 → (∃ k, n = k * 11) → n ≤ 99 :=
by
  intro n hn hmul
  sorry

end largest_multiple_of_11_less_than_100_l273_273882


namespace ternary_to_decimal_l273_273108

def to_decimal (ternary : Nat) : Nat :=
  match ternary with
  | 121 => 1 * 3^2 + 2 * 3^1 + 1 * 3^0
  | _ => 0

theorem ternary_to_decimal : to_decimal 121 = 16 := by
  sorry

end ternary_to_decimal_l273_273108


namespace tyler_puppies_l273_273738

theorem tyler_puppies (dogs : ℕ) (puppies_per_dog : ℕ) (total_puppies : ℕ) 
  (h1 : dogs = 15) (h2 : puppies_per_dog = 5) : total_puppies = 75 :=
by {
  sorry
}

end tyler_puppies_l273_273738


namespace attendance_ratio_3_to_1_l273_273250

variable (x y : ℕ)
variable (total_attendance : ℕ := 2700)
variable (second_day_attendance : ℕ := 300)

/-- 
Prove that the ratio of the number of people attending the third day to the number of people attending the first day is 3:1
-/
theorem attendance_ratio_3_to_1
  (h1 : total_attendance = 2700)
  (h2 : second_day_attendance = x / 2)
  (h3 : second_day_attendance = 300)
  (h4 : y = total_attendance - x - second_day_attendance) :
  y / x = 3 :=
by
  sorry

end attendance_ratio_3_to_1_l273_273250


namespace ternary_to_decimal_l273_273109

def to_decimal (ternary : Nat) : Nat :=
  match ternary with
  | 121 => 1 * 3^2 + 2 * 3^1 + 1 * 3^0
  | _ => 0

theorem ternary_to_decimal : to_decimal 121 = 16 := by
  sorry

end ternary_to_decimal_l273_273109


namespace polygon_sides_arithmetic_progression_l273_273017

theorem polygon_sides_arithmetic_progression
  (angles_in_arithmetic_progression : ∃ (a d : ℝ) (angles : ℕ → ℝ), ∀ (k : ℕ), angles k = a + k * d)
  (common_difference : ∃ (d : ℝ), d = 3)
  (largest_angle : ∃ (n : ℕ) (angles : ℕ → ℝ), angles n = 150) :
  ∃ (n : ℕ), n = 15 :=
sorry

end polygon_sides_arithmetic_progression_l273_273017


namespace range_of_a_l273_273965

noncomputable def f (a : ℝ) (x : ℝ) := a - x^2
def g (x : ℝ) := x + 2
def h (x : ℝ) := x^2 - x - 2

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ f a x = -g x) ↔ a ∈ set.Icc (-2 : ℝ) 0 := 
by
  sorry

end range_of_a_l273_273965


namespace females_in_orchestra_not_in_band_l273_273263

theorem females_in_orchestra_not_in_band 
  (females_in_band : ℤ) 
  (males_in_band : ℤ) 
  (females_in_orchestra : ℤ) 
  (males_in_orchestra : ℤ) 
  (females_in_both : ℤ) 
  (total_members : ℤ) 
  (h1 : females_in_band = 120) 
  (h2 : males_in_band = 100) 
  (h3 : females_in_orchestra = 100) 
  (h4 : males_in_orchestra = 120) 
  (h5 : females_in_both = 80) 
  (h6 : total_members = 260) : 
  (females_in_orchestra - females_in_both = 20) := 
  sorry

end females_in_orchestra_not_in_band_l273_273263


namespace complete_the_square_l273_273826

theorem complete_the_square (x : ℝ) : 
  ∃ (a h k : ℝ), a = 1 ∧ h = 7 / 2 ∧ k = -49 / 4 ∧ x^2 - 7 * x = a * (x - h) ^ 2 + k :=
by
  use 1, 7 / 2, -49 / 4
  sorry

end complete_the_square_l273_273826


namespace part_I_part_II_l273_273203

def f (x a : ℝ) := |x - a| + |x - 1|

theorem part_I {x : ℝ} : Set.Icc 0 4 = {y | f y 3 ≤ 4} := 
sorry

theorem part_II {a : ℝ} : (∀ x, ¬ (f x a < 2)) ↔ a ∈ Set.Ici 3 ∪ Set.Iic (-1) :=
sorry

end part_I_part_II_l273_273203


namespace factorial_300_zeros_l273_273875

theorem factorial_300_zeros : (∃ n, nat.factorial 300 % 10^(n+1) = 0 ∧ nat.factorial 300 % 10^n ≠ 0) ∧ ∀ n, nat.factorial 300 % 10^(74 + n) ≠ 10^74 + 1 :=
sorry

end factorial_300_zeros_l273_273875


namespace tangent_sufficient_but_not_necessary_condition_l273_273724

noncomputable def tangent_condition (m : ℝ) : Prop :=
  let line := fun (x y : ℝ) => x + y - m = 0
  let circle := fun (x y : ℝ) => (x - 1) ^ 2 + (y - 1) ^ 2 = 2
  ∃ (x y: ℝ), line x y ∧ circle x y -- A line and circle are tangent if they intersect exactly at one point

theorem tangent_sufficient_but_not_necessary_condition (m : ℝ) :
  (tangent_condition m) ↔ (m = 0 ∨ m = 4) := by
  sorry

end tangent_sufficient_but_not_necessary_condition_l273_273724


namespace max_elevation_l273_273385

def elevation (t : ℝ) : ℝ := 144 * t - 18 * t^2

theorem max_elevation : ∃ t : ℝ, elevation t = 288 :=
by
  use 4
  sorry

end max_elevation_l273_273385


namespace mrs_lee_earnings_percentage_l273_273153

noncomputable def percentage_earnings_june (T : ℝ) : ℝ :=
  let L := 0.5 * T
  let L_June := 1.2 * L
  let total_income_june := T
  (L_June / total_income_june) * 100

theorem mrs_lee_earnings_percentage (T : ℝ) (hT : T ≠ 0) : percentage_earnings_june T = 60 :=
by
  sorry

end mrs_lee_earnings_percentage_l273_273153


namespace algebraic_expression_value_l273_273545

-- Define the conditions 
variables (x y : ℝ)
def condition1 : Prop := x + y = 2
def condition2 : Prop := x - y = 4

-- State the main theorem
theorem algebraic_expression_value (h1 : condition1 x y) (h2 : condition2 x y) :
  1 + x^2 - y^2 = 9 :=
sorry

end algebraic_expression_value_l273_273545


namespace correct_number_of_sequences_l273_273657

noncomputable def athlete_sequences : Nat :=
  let total_permutations := 24
  let A_first_leg := 6
  let B_fourth_leg := 6
  let A_first_and_B_fourth := 2
  total_permutations - (A_first_leg + B_fourth_leg - A_first_and_B_fourth)

theorem correct_number_of_sequences : athlete_sequences = 14 := by
  sorry

end correct_number_of_sequences_l273_273657


namespace ball_attendance_l273_273175

variable (n m : ℕ)

def ball_conditions (n m : ℕ) := 
  n + m < 50 ∧ 
  4 ∣ 3 * n ∧ 
  7 ∣ 5 * m

theorem ball_attendance (n m : ℕ) (h : ball_conditions n m) : 
  n + m = 41 :=
sorry

end ball_attendance_l273_273175


namespace two_pi_irrational_l273_273495

-- Assuming \(\pi\) is irrational as is commonly accepted
def irrational (x : ℝ) : Prop := ¬ (∃ a b : ℤ, b ≠ 0 ∧ x = a / b)

theorem two_pi_irrational : irrational (2 * Real.pi) := 
by 
  sorry

end two_pi_irrational_l273_273495


namespace smoothie_ratios_l273_273752

variable (initial_p initial_v m_p m_ratio_p_v: ℕ) (y_p y_v : ℕ)

-- Given conditions
theorem smoothie_ratios (h_initial_p : initial_p = 24) (h_initial_v : initial_v = 25) 
                        (h_m_p : m_p = 20) (h_m_ratio_p_v : m_ratio_p_v = 4)
                        (h_y_p : y_p = initial_p - m_p) (h_y_v : y_v = initial_v - m_p / m_ratio_p_v) :
  (y_p / gcd y_p y_v) = 1 ∧ (y_v / gcd y_p y_v) = 5 :=
by
  sorry

end smoothie_ratios_l273_273752


namespace angle_CBE_minimal_l273_273987

theorem angle_CBE_minimal
    (ABC ABD DBE: ℝ)
    (h1: ABC = 40)
    (h2: ABD = 28)
    (h3: DBE = 10) : 
    CBE = 2 :=
by
  sorry

end angle_CBE_minimal_l273_273987


namespace directrix_of_parabola_l273_273220

theorem directrix_of_parabola :
  (∀ x y : ℝ, x^2 = 4 * y → ∃ y₀ : ℝ, y₀ = -1 ∧ ∀ y' : ℝ, y' = y₀) :=
by
  sorry

end directrix_of_parabola_l273_273220


namespace hannah_payment_l273_273141

def costWashingMachine : ℝ := 100
def costDryer : ℝ := costWashingMachine - 30
def totalCostBeforeDiscount : ℝ := costWashingMachine + costDryer
def discount : ℝ := totalCostBeforeDiscount * 0.1
def finalCost : ℝ := totalCostBeforeDiscount - discount

theorem hannah_payment : finalCost = 153 := by
  simp [costWashingMachine, costDryer, totalCostBeforeDiscount, discount, finalCost]
  sorry

end hannah_payment_l273_273141


namespace sum_of_divisors_24_l273_273492

theorem sum_of_divisors_24 : (∑ d in (finset.filter (λ n, 24 % n = 0) (finset.range 25)), d) = 60 :=
by
  sorry

end sum_of_divisors_24_l273_273492


namespace misha_class_predictions_probability_l273_273076

-- Definitions representing the conditions
def monday_classes : ℕ := 5
def tuesday_classes : ℕ := 6
def total_classes : ℕ := monday_classes + tuesday_classes
def total_flips : ℕ := total_classes
def correct_predictions : ℕ := 7
def correct_monday_predictions : ℕ := 3

-- Lean theorem representing the proof problem
theorem misha_class_predictions_probability :
  (probability_of_correct_predictions total_flips correct_predictions monday_classes correct_monday_predictions) =
    (5 / 11) :=
  sorry

end misha_class_predictions_probability_l273_273076


namespace not_sqrt2_rational_union_eq_intersection_implies_equal_intersection_eq_b_subset_a_element_in_both_implies_in_intersection_l273_273894

-- Definitions
def is_rational (x : ℝ) : Prop := ∃ a b : ℤ, b ≠ 0 ∧ (x = a / b)
def union (A B : Set α) : Set α := {x | x ∈ A ∨ x ∈ B}
def intersection (A B : Set α) : Set α := {x | x ∈ A ∧ x ∈ B}
def subset (A B : Set α) : Prop := ∀ x, x ∈ A → x ∈ B

-- Statement A
theorem not_sqrt2_rational : ¬ is_rational (Real.sqrt 2) :=
sorry

-- Statement B
theorem union_eq_intersection_implies_equal {α : Type*} {A B : Set α}
  (h : union A B = intersection A B) : A = B :=
sorry

-- Statement C
theorem intersection_eq_b_subset_a {α : Type*} {A B : Set α}
  (h : intersection A B = B) : subset B A :=
sorry

-- Statement D
theorem element_in_both_implies_in_intersection {α : Type*} {A B : Set α} {a : α}
  (haA : a ∈ A) (haB : a ∈ B) : a ∈ intersection A B :=
sorry

end not_sqrt2_rational_union_eq_intersection_implies_equal_intersection_eq_b_subset_a_element_in_both_implies_in_intersection_l273_273894


namespace algebraic_expression_evaluation_l273_273297

theorem algebraic_expression_evaluation (a b c : ℝ) 
  (h1 : a^2 + b * c = 14) 
  (h2 : b^2 - 2 * b * c = -6) : 
  3 * a^2 + 4 * b^2 - 5 * b * c = 18 :=
by 
  sorry

end algebraic_expression_evaluation_l273_273297


namespace ratio_SP_CP_l273_273354

variables (CP SP P : ℝ)
axiom ratio_profit_CP : P / CP = 2

theorem ratio_SP_CP : SP / CP = 3 :=
by
  -- Proof statement (not required as per the instruction)
  sorry

end ratio_SP_CP_l273_273354


namespace xu_jun_age_l273_273389

variable (x y : ℕ)

def condition1 : Prop := y - 2 = 3 * (x - 2)
def condition2 : Prop := y + 8 = 2 * (x + 8)

theorem xu_jun_age (h1 : condition1 x y) (h2 : condition2 x y) : x = 12 :=
by 
sorry

end xu_jun_age_l273_273389


namespace totalNumberOfPupils_l273_273092

-- Definitions of the conditions
def numberOfGirls : Nat := 232
def numberOfBoys : Nat := 253

-- Statement of the problem
theorem totalNumberOfPupils : numberOfGirls + numberOfBoys = 485 := by
  sorry

end totalNumberOfPupils_l273_273092


namespace sale_day_intersection_in_july_l273_273906

def is_multiple_of_five (d : ℕ) : Prop :=
  d % 5 = 0

def shoe_store_sale_days (d : ℕ) : Prop :=
  ∃ (k : ℕ), d = 3 + k * 6

theorem sale_day_intersection_in_july : 
  (∃ d, is_multiple_of_five d ∧ shoe_store_sale_days d ∧ 1 ≤ d ∧ d ≤ 31) = (1 = Nat.card {d | is_multiple_of_five d ∧ shoe_store_sale_days d ∧ 1 ≤ d ∧ d ≤ 31}) :=
by
  sorry

end sale_day_intersection_in_july_l273_273906


namespace math_problem_l273_273130

theorem math_problem (x : ℕ) (h : (2^x + 2^x + 2^x + 2^x + 2^x + 2^x + 2^x + 2^x = 512)) : (x + 2) * (x - 2) = 32 :=
sorry

end math_problem_l273_273130


namespace interest_rate_l273_273494

noncomputable def simple_interest (P r t : ℝ) : ℝ := P * r * t / 100

noncomputable def compound_interest (P r t : ℝ) : ℝ := P * (1 + r/100)^t - P

theorem interest_rate (P t : ℝ) (diff : ℝ) (r : ℝ) (h : P = 1000) (t_eq : t = 4) 
  (diff_eq : diff = 64.10) : 
  compound_interest P r t - simple_interest P r t = diff → r = 10 :=
by
  sorry

end interest_rate_l273_273494


namespace unknown_number_is_five_l273_273902

theorem unknown_number_is_five (x : ℕ) (h : 64 + x * 12 / (180 / 3) = 65) : x = 5 := 
by 
  sorry

end unknown_number_is_five_l273_273902


namespace solve_for_y_l273_273342

theorem solve_for_y (y : ℕ) : (1000^4 = 10^y) → y = 12 :=
by {
  sorry
}

end solve_for_y_l273_273342


namespace num_solution_pairs_l273_273972

theorem num_solution_pairs : 
  ∃! (n : ℕ), 
    n = 2 ∧ 
    ∃ x y : ℕ, 
      x > 0 ∧ y >0 ∧ 
      4^x = y^2 + 15 := 
by 
  sorry

end num_solution_pairs_l273_273972


namespace students_needed_to_fill_buses_l273_273610

theorem students_needed_to_fill_buses (n : ℕ) (c : ℕ) (h_n : n = 254) (h_c : c = 30) : 
  (c * ((n + c - 1) / c) - n) = 16 :=
by
  sorry

end students_needed_to_fill_buses_l273_273610


namespace Isabelle_ticket_cost_l273_273327

theorem Isabelle_ticket_cost :
  (∀ (week_salary : ℕ) (weeks_worked : ℕ) (brother_ticket_cost : ℕ) (brothers_saved : ℕ) (Isabelle_saved : ℕ),
  week_salary = 3 ∧ weeks_worked = 10 ∧ brother_ticket_cost = 10 ∧ brothers_saved = 5 ∧ Isabelle_saved = 5 →
  Isabelle_saved + (week_salary * weeks_worked) - ((brother_ticket_cost * 2) - brothers_saved) = 15) :=
by
  sorry

end Isabelle_ticket_cost_l273_273327


namespace time_to_pass_platform_is_correct_l273_273516

noncomputable def train_length : ℝ := 250 -- meters
noncomputable def time_to_pass_pole : ℝ := 10 -- seconds
noncomputable def time_to_pass_platform : ℝ := 60 -- seconds

-- Speed of the train
noncomputable def train_speed := train_length / time_to_pass_pole -- meters/second

-- Length of the platform
noncomputable def platform_length := train_speed * time_to_pass_platform - train_length -- meters

-- Proving the time to pass the platform is 50 seconds
theorem time_to_pass_platform_is_correct : 
  (platform_length / train_speed) = 50 :=
by
  sorry

end time_to_pass_platform_is_correct_l273_273516


namespace find_positive_integer_l273_273532

theorem find_positive_integer (n : ℕ) (hn_pos : n > 0) :
  (∃ a b : ℕ, n = a^2 ∧ n + 100 = b^2) → n = 576 :=
by sorry

end find_positive_integer_l273_273532


namespace segment_lengths_unique_l273_273128

def cutting_ratio (p q : ℕ) (x : ℝ) : ℝ → ℝ := 
  if p = q then 1
  else if p > q then Real.log x / Real.log (p / (p + q)) + 1
  else Real.log x / Real.log (q / (p + q)) + 1

noncomputable def a (p q : ℕ) (x : ℝ) : ℝ :=
  if p = q then 1 
  else if p > q then Nat.ceil (Real.log x / Real.log (p / (p + q))) + 1
  else Nat.ceil (Real.log x / Real.log (q / (p + q))) + 1

theorem segment_lengths_unique (p q : ℕ) (x : ℝ) : a(p, q, x) = 
  if p = q then 1
  else if p > q then Nat.ceil (Real.log x / Real.log (p / (p + q))) + 1
  else Nat.ceil (Real.log x / Real.log (q / (p + q))) + 1 := 
by 
  -- skipping the proof
  sorry

end segment_lengths_unique_l273_273128


namespace susan_backward_spaces_l273_273861

variable (spaces_to_win total_spaces : ℕ)
variables (first_turn second_turn_forward second_turn_back third_turn : ℕ)

theorem susan_backward_spaces :
  ∀ (total_spaces first_turn second_turn_forward second_turn_back third_turn win_left : ℕ),
  total_spaces = 48 →
  first_turn = 8 →
  second_turn_forward = 2 →
  third_turn = 6 →
  win_left = 37 →
  first_turn + second_turn_forward + third_turn - second_turn_back + win_left = total_spaces →
  second_turn_back = 6 :=
by
  intros total_spaces first_turn second_turn_forward second_turn_back third_turn win_left
  intros h_total h_first h_second_forward h_third h_win h_eq
  rw [h_total, h_first, h_second_forward, h_third, h_win] at h_eq
  sorry

end susan_backward_spaces_l273_273861


namespace quadratic_roots_expr_value_l273_273701

theorem quadratic_roots_expr_value :
  let p q : ℝ := roots_of_quadratic 3 9 (-21)
  (3 * p - 4) * (6 * q - 8) = 122 :=
by
  sorry

end quadratic_roots_expr_value_l273_273701


namespace number_of_factors_of_81_l273_273814

-- Define 81 as a power of 3
def n : ℕ := 3^4

-- Theorem stating the number of distinct positive factors of 81
theorem number_of_factors_of_81 : ∀ n = 81, nat.factors_count n = 5 := by
  sorry

end number_of_factors_of_81_l273_273814


namespace range_of_a_l273_273684

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 
if x < 1 then -x + 2 else a / x

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x < 1 ∧ (0 < -x + 2)) ∧ (∀ x : ℝ, x ≥ 1 → (0 < a / x)) → a ≥ 1 :=
by
  sorry

end range_of_a_l273_273684


namespace total_cards_in_stack_l273_273412

theorem total_cards_in_stack (n : ℕ) (H1: 252 ≤ 2 * n) (H2 : (2 * n) % 2 = 0)
                             (H3 : ∀ k : ℕ, k ≤ 2 * n → (if k % 2 = 0 then k / 2 else (k + 1) / 2) * 2 = k) :
  2 * n = 504 := sorry

end total_cards_in_stack_l273_273412


namespace proof_problem_binomial_variance_l273_273958

variable {X : ℝ}

def binomial_X (n : ℕ) (p : ℝ) := ∑ i in Finset.range (n + 1), 
  i * ((fin n).choose i) * (p^i) * ((1 - p)^ (n - i))

def var_binomial_X (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

theorem proof_problem_binomial_variance (h : binomial_X 4 p = 2) : var_binomial_X 4 p = 1 := by
  sorry

end proof_problem_binomial_variance_l273_273958


namespace polygon_sides_l273_273038

theorem polygon_sides (n : ℕ) 
  (h1 : (n - 2) * 180 = 4 * 360) : 
  n = 6 :=
by sorry

end polygon_sides_l273_273038


namespace power_product_rule_l273_273100

theorem power_product_rule (a : ℤ) : (-a^2)^3 = -a^6 := 
by 
  sorry

end power_product_rule_l273_273100


namespace john_extra_hours_l273_273165

theorem john_extra_hours (daily_earnings : ℕ) (hours_worked : ℕ) (bonus : ℕ) (hourly_wage : ℕ) (total_earnings_with_bonus : ℕ) (total_hours_with_bonus : ℕ) : 
  daily_earnings = 80 ∧ 
  hours_worked = 8 ∧ 
  bonus = 20 ∧ 
  hourly_wage = 10 ∧ 
  total_earnings_with_bonus = daily_earnings + bonus ∧
  total_hours_with_bonus = total_earnings_with_bonus / hourly_wage → 
  total_hours_with_bonus - hours_worked = 2 := 
by 
  sorry

end john_extra_hours_l273_273165


namespace polygon_sides_l273_273027

theorem polygon_sides (n : ℕ) :
  ((n - 2) * 180 = 4 * 360) → n = 10 :=
by 
  sorry

end polygon_sides_l273_273027


namespace largest_integer_x_l273_273405

theorem largest_integer_x (x : ℤ) : (x / 4 : ℚ) + (3 / 7 : ℚ) < (9 / 4 : ℚ) → x ≤ 7 ∧ (7 / 4 : ℚ) + (3 / 7 : ℚ) < (9 / 4 : ℚ) :=
by
  sorry

end largest_integer_x_l273_273405


namespace original_price_petrol_in_euros_l273_273388

theorem original_price_petrol_in_euros
  (P : ℝ) -- The original price of petrol in USD per gallon
  (h1 : 0.865 * P * 7.25 + 0.135 * 325 = 325) -- Condition derived from price reduction and additional gallons
  (h2 : P > 0) -- Ensure original price is positive
  (exchange_rate : ℝ) (h3 : exchange_rate = 1.15) : 
  P / exchange_rate = 38.98 :=
by 
  let price_in_euros := P / exchange_rate 
  have h4 : price_in_euros = 38.98 := sorry
  exact h4

end original_price_petrol_in_euros_l273_273388


namespace percentage_decrease_l273_273025

theorem percentage_decrease (a b p : ℝ) (a_pos : 0 < a) (b_pos : 0 < b)
    (h_ratio : a / b = 4 / 5) 
    (h_x : ∃ x, x = a * 1.25)
    (h_m : ∃ m, m = b * (1 - p / 100))
    (h_mx : ∃ m x, (m / x = 0.2)) :
        (p = 80) :=
by
  sorry

end percentage_decrease_l273_273025


namespace common_difference_of_arithmetic_sequence_l273_273803

theorem common_difference_of_arithmetic_sequence
  (a : ℕ → ℝ) (h_arith_seq : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (h1 : a 7 - 2 * a 4 = -1)
  (h2 : a 3 = 0) :
  (a 2 - a 1) = - 1 / 2 :=
by
  sorry

end common_difference_of_arithmetic_sequence_l273_273803


namespace counties_rained_on_monday_l273_273561

theorem counties_rained_on_monday : 
  ∀ (M T R_no_both R_both : ℝ),
    T = 0.55 → 
    R_no_both = 0.35 →
    R_both = 0.60 →
    (M + T - R_both = 1 - R_no_both) →
    M = 0.70 :=
by
  intros M T R_no_both R_both hT hR_no_both hR_both hInclusionExclusion
  sorry

end counties_rained_on_monday_l273_273561


namespace total_money_shared_l273_273167

-- Define the variables and conditions
def joshua_share : ℕ := 30
def justin_share : ℕ := joshua_share / 3
def total_shared_money : ℕ := joshua_share + justin_share

-- State the theorem to prove
theorem total_money_shared : total_shared_money = 40 :=
by
  -- proof will go here
  sorry

end total_money_shared_l273_273167


namespace find_p_l273_273784

theorem find_p (p : ℝ) (h : 0 < p ∧ p < 1) : 
  p + (1 - p) * p + (1 - p)^2 * p = 0.784 → p = 0.4 :=
by
  intros h_eq
  sorry

end find_p_l273_273784


namespace evaluate_composite_function_l273_273137

def f (x : ℝ) : ℝ := x^2 - 2 * x + 2
def g (x : ℝ) : ℝ := 3 * x + 2

theorem evaluate_composite_function :
  f (g (-2)) = 26 := by
  sorry

end evaluate_composite_function_l273_273137


namespace peach_tree_average_production_l273_273566

-- Definitions derived from the conditions
def num_apple_trees : ℕ := 30
def kg_per_apple_tree : ℕ := 150
def num_peach_trees : ℕ := 45
def total_mass_fruit : ℕ := 7425

-- Main Statement to be proven
theorem peach_tree_average_production : 
  (total_mass_fruit - (num_apple_trees * kg_per_apple_tree)) = (num_peach_trees * 65) :=
by
  sorry

end peach_tree_average_production_l273_273566


namespace locus_of_point_P_l273_273089

-- Definitions and conditions
def circle_M (x y : ℝ) : Prop := (x - 1) ^ 2 + y ^ 2 = 4
def A_point : ℝ × ℝ := (2, 1)
def chord_BC (x y x₀ y₀ : ℝ) : Prop := (x₀ - 1) * x + y₀ * y - x₀ - 3 = 0
def point_P_locus (x₀ y₀ : ℝ) : Prop := ∃ x y, (chord_BC x y x₀ y₀) ∧ x = 2 ∧ y = 1

-- Lean 4 statement to be proved
theorem locus_of_point_P (x₀ y₀ : ℝ) (h : point_P_locus x₀ y₀) : x₀ + y₀ - 5 = 0 :=
  by
  sorry

end locus_of_point_P_l273_273089


namespace find_b_when_a_is_negative12_l273_273476

theorem find_b_when_a_is_negative12 (a b : ℝ) (h1 : a + b = 60) (h2 : a = 3 * b) (h3 : ∃ k, a * b = k) : b = -56.25 :=
sorry

end find_b_when_a_is_negative12_l273_273476


namespace probability_of_intersecting_diagonals_l273_273365

noncomputable def intersecting_diagonals_probability : ℚ :=
let total_vertices := 8 in
let total_pairs := Nat.choose total_vertices 2 in
let total_sides := 8 in
let total_diagonals := total_pairs - total_sides in
let total_pairs_diagonals := Nat.choose total_diagonals 2 in
let intersecting_diagonals := Nat.choose total_vertices 4 in
(intersecting_diagonals : ℚ) / (total_pairs_diagonals : ℚ)

theorem probability_of_intersecting_diagonals :
  intersecting_diagonals_probability = 7 / 19 :=
by
  sorry

end probability_of_intersecting_diagonals_l273_273365


namespace man_l273_273745

noncomputable def man's_rate_in_still_water (downstream upstream : ℝ) : ℝ :=
  (downstream + upstream) / 2

theorem man's_rate_correct :
  let downstream := 6
  let upstream := 3
  man's_rate_in_still_water downstream upstream = 4.5 :=
by
  sorry

end man_l273_273745


namespace average_weight_of_removed_onions_l273_273611

theorem average_weight_of_removed_onions (total_weight_40_onions : ℝ := 7680)
    (average_weight_35_onions : ℝ := 190)
    (number_of_onions_removed : ℕ := 5)
    (total_onions_initial : ℕ := 40)
    (total_number_of_remaining_onions : ℕ := 35) :
    (total_weight_40_onions - total_number_of_remaining_onions * average_weight_35_onions) / number_of_onions_removed = 206 :=
by
    sorry

end average_weight_of_removed_onions_l273_273611


namespace variance_cows_l273_273088

-- Define the number of cows and incidence rate.
def n : ℕ := 10
def p : ℝ := 0.02

-- The variance of the binomial distribution, given n and p.
def variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

-- Statement to prove
theorem variance_cows : variance n p = 0.196 :=
by
  sorry

end variance_cows_l273_273088


namespace angle_of_inclination_l273_273409

theorem angle_of_inclination :
  ∀ (t : ℝ), (∃ (θ : ℝ), (θ = 230) ∧ (let x := 3 + t * Real.cos θ, y := -1 + t * Real.sin θ in
  ∀ θ', θ' = 50)) :=
by
suffices h : ∀ (θ : ℝ), θ = 230 → 50 = θ + 180 - 230 
exact h
sorry

end angle_of_inclination_l273_273409


namespace problem_statement_l273_273807

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x - 1

theorem problem_statement (x : ℝ) (h : x ≠ 0) : f x > 0 :=
by sorry

end problem_statement_l273_273807


namespace number_of_workers_l273_273757

theorem number_of_workers (supervisors team_leads_per_supervisor workers_per_team_lead : ℕ) 
    (h_supervisors : supervisors = 13)
    (h_team_leads_per_supervisor : team_leads_per_supervisor = 3)
    (h_workers_per_team_lead : workers_per_team_lead = 10):
    supervisors * team_leads_per_supervisor * workers_per_team_lead = 390 :=
by
  -- to avoid leaving the proof section empty and potentially creating an invalid Lean statement
  sorry

end number_of_workers_l273_273757


namespace quilt_shaded_fraction_l273_273231

theorem quilt_shaded_fraction (total_squares : ℕ) (fully_shaded : ℕ) (half_shaded_squares : ℕ) (half_shades_per_square: ℕ) : 
  (((fully_shaded) + (half_shaded_squares * half_shades_per_square / 2)) / total_squares) = (1 / 4) :=
by 
  let fully_shaded := 2
  let half_shaded_squares := 4
  let half_shades_per_square := 1
  let total_squares := 16
  sorry

end quilt_shaded_fraction_l273_273231


namespace probability_correct_predictions_monday_l273_273075

def number_of_classes_monday : ℕ := 5
def number_of_classes_tuesday : ℕ := 6
def total_classes : ℕ := number_of_classes_monday + number_of_classes_tuesday

def total_correct_predictions : ℕ := 7
def correct_predictions_monday : ℕ := 3
def correct_predictions_tuesday : ℕ := total_correct_predictions - correct_predictions_monday

noncomputable def binomial_coefficient (n k : ℕ) : ℝ := (nat.choose n k : ℝ)

noncomputable def probability_exact_correct_monday (n k : ℕ) : ℝ :=
  binomial_coefficient n k * (real.exp 11 (ln 2⁻¹ * total_classes))

theorem probability_correct_predictions_monday :
  probability_exact_correct_monday number_of_classes_monday correct_predictions_monday *
  probability_exact_correct_monday number_of_classes_tuesday correct_predictions_tuesday /
  probability_exact_correct_monday total_classes total_correct_predictions = 5 / 11 :=
sorry

end probability_correct_predictions_monday_l273_273075


namespace amount_tom_should_pay_l273_273049

theorem amount_tom_should_pay (original_price : ℝ) (multiplier : ℝ) 
  (h1 : original_price = 3) (h2 : multiplier = 3) : 
  original_price * multiplier = 9 :=
sorry

end amount_tom_should_pay_l273_273049


namespace sum_of_coefficients_l273_273429

theorem sum_of_coefficients (a_0 a_1 a_2 a_3 a_4 a_5 : ℕ) (h₁ : (1 + x)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) : 
  a_1 + a_2 + a_3 + a_4 + a_5 = 31 := by
  sorry

end sum_of_coefficients_l273_273429


namespace garden_area_l273_273642

theorem garden_area 
  (property_width : ℕ)
  (property_length : ℕ)
  (garden_width_ratio : ℚ)
  (garden_length_ratio : ℚ)
  (width_ratio_eq : garden_width_ratio = (1 : ℚ) / 8)
  (length_ratio_eq : garden_length_ratio = (1 : ℚ) / 10)
  (property_width_eq : property_width = 1000)
  (property_length_eq : property_length = 2250) :
  (property_width * garden_width_ratio * property_length * garden_length_ratio = 28125) :=
  sorry

end garden_area_l273_273642


namespace algebraic_expression_value_l273_273301

theorem algebraic_expression_value (x y : ℝ) (h : x^2 - 4 * x - 1 = 0) : 
  (2 * x - 3) ^ 2 - (x + y) * (x - y) - y ^ 2 = 12 := 
by {
  sorry
}

end algebraic_expression_value_l273_273301


namespace least_possible_square_area_l273_273606

theorem least_possible_square_area (s : ℝ) (h1 : 4.5 ≤ s) (h2 : s < 5.5) : s * s ≥ 20.25 := by
  sorry

end least_possible_square_area_l273_273606


namespace boxes_needed_l273_273449

-- Let's define the conditions
def total_paper_clips : ℕ := 81
def paper_clips_per_box : ℕ := 9

-- Define the target of our proof, which is that the number of boxes needed is 9
theorem boxes_needed : total_paper_clips / paper_clips_per_box = 9 := by
  sorry

end boxes_needed_l273_273449


namespace problem_f_2005_value_l273_273929

noncomputable def f : ℝ → ℝ := sorry

theorem problem_f_2005_value (h_even : ∀ x : ℝ, f (-x) = f x)
                            (h_periodic : ∀ x : ℝ, f (x + 8) = f x + f 4)
                            (h_initial : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 4 → f x = 4 - x) :
  f 2005 = 0 :=
sorry

end problem_f_2005_value_l273_273929


namespace find_coefficient_y_l273_273272

theorem find_coefficient_y (a b c : ℕ) (h1 : 100 * a + 10 * b + c - 7 * (a + b + c) = 100) (h2 : a + b + c ≠ 0) :
  100 * c + 10 * b + a = 43 * (a + b + c) :=
by
  sorry

end find_coefficient_y_l273_273272


namespace probability_diff_by_three_l273_273835

theorem probability_diff_by_three (r1 r2 : ℕ) (h1 : 1 ≤ r1 ∧ r1 ≤ 6) (h2 : 1 ≤ r2 ∧ r2 ≤ 6) :
  (∃ (rolls : List (ℕ × ℕ)), 
    rolls = [ (2, 5), (5, 2), (3, 6), (4, 1) ] ∧ 
    (r1, r2) ∈ rolls) →
  (4 : ℚ) / 36 = (1 / 9 : ℚ) :=
by sorry

end probability_diff_by_three_l273_273835


namespace Isabel_paper_used_l273_273982

theorem Isabel_paper_used
  (initial_pieces : ℕ)
  (remaining_pieces : ℕ)
  (initial_condition : initial_pieces = 900)
  (remaining_condition : remaining_pieces = 744) :
  initial_pieces - remaining_pieces = 156 :=
by 
  -- Admitting the proof for now
  sorry

end Isabel_paper_used_l273_273982


namespace simplify_expression_l273_273925

-- Define the variables a and b as real numbers
variables (a b : ℝ)

-- Define the expression and the simplified expression
def original_expr := -a^2 * (-2 * a * b) + 3 * a * (a^2 * b - 1)
def simplified_expr := 5 * a^3 * b - 3 * a

-- Statement that the original expression is equal to the simplified expression
theorem simplify_expression : original_expr a b = simplified_expr a b :=
by
  sorry

end simplify_expression_l273_273925


namespace find_smallest_a_l273_273540

noncomputable def smallest_triangle_length (a : ℝ) : Prop :=
  ∀ (A B C : ℝ × ℝ) (hA : A.1^2 + A.2^2 = 1) (hB : B.1^2 + B.2^2 = 1) (hC : C.1^2 + C.2^2 = 1),
    ∃ (P Q R : ℝ × ℝ) (hPQR : (P = Q) → false ∧ (Q = R) → false ∧ (R = P) → false) (eq_triangle : ∀ X Y, (X = P ∨ X = Q ∨ X = R) → (Y = P ∨ Y = Q ∨ Y = R) → dist X Y = a),
    (A = P ∨ A = Q ∨ A = R ∨ ∃x, (x ∈ [P,R] ∨ x ∈ [P,Q] ∨ x ∈ [Q,R]) ∧ dist A x = 0) ∧
    (B = P ∨ B = Q ∨ B = R ∨ ∃x, (x ∈ [P,R] ∨ x ∈ [P,Q] ∨ x ∈ [Q,R]) ∧ dist B x = 0) ∧
    (C = P ∨ C = Q ∨ C = R ∨ ∃x, (x ∈ [P,R] ∨ x ∈ [P,Q] ∨ x ∈ [Q,R]) ∧ dist C x = 0)

theorem find_smallest_a : smallest_triangle_length (\frac{4}{\sqrt{3}} * (Real.sin (80 * Real.pi / 180))^2) := 
sorry

end find_smallest_a_l273_273540


namespace eval_expression_l273_273787

theorem eval_expression : (2: ℤ)^2 - 3 * (2: ℤ) + 2 = 0 := by
  sorry

end eval_expression_l273_273787


namespace rons_height_l273_273916

variable (R : ℝ)

theorem rons_height
  (depth_eq_16_ron_height : 16 * R = 208) :
  R = 13 :=
by {
  sorry
}

end rons_height_l273_273916


namespace value_of_f_g_5_l273_273558

def g (x : ℕ) : ℕ := 4 * x - 5
def f (x : ℕ) : ℕ := 6 * x + 11

theorem value_of_f_g_5 : f (g 5) = 101 := by
  sorry

end value_of_f_g_5_l273_273558


namespace percentage_volume_taken_by_cubes_l273_273387

def volume_of_box (l w h : ℝ) : ℝ := l * w * h

def volume_of_cube (side : ℝ) : ℝ := side ^ 3

noncomputable def total_cubes_fit (l w h side : ℝ) : ℝ := 
  (l / side) * (w / side) * (h / side)

theorem percentage_volume_taken_by_cubes (l w h side : ℝ) (hl : l = 12) (hw : w = 6) (hh : h = 9) (hside : side = 3) :
  volume_of_box l w h ≠ 0 → 
  (total_cubes_fit l w h side * volume_of_cube side / volume_of_box l w h) * 100 = 100 :=
by
  intros
  rw [hl, hw, hh, hside]
  simp only [volume_of_box, volume_of_cube, total_cubes_fit]
  sorry

end percentage_volume_taken_by_cubes_l273_273387


namespace max_tetrahedron_volume_on_unit_sphere_l273_273127

open EuclideanGeometry

theorem max_tetrahedron_volume_on_unit_sphere
  (A B C D : EuclideanGeometry.Ch.Point 3)
  (unit_sphere : Metric.sphere (0 : EuclideanGeometry.Ch.Point 3) 1)
  (hA : A ∈ unit_sphere)
  (hB : B ∈ unit_sphere)
  (hC : C ∈ unit_sphere)
  (hD : D ∈ unit_sphere)
  (hAB : EuclideanGeometry.dist A B = EuclideanGeometry.dist A C)
  (hAC : EuclideanGeometry.dist A C = EuclideanGeometry.dist A D)
  (hBC : EuclideanGeometry.dist B C = EuclideanGeometry.dist C D)
  (hBD : EuclideanGeometry.dist B D = EuclideanGeometry.dist C D) :
  tetrahedron_volume A B C D ≤ (8 * Real.sqrt 3) / 27 := sorry

end max_tetrahedron_volume_on_unit_sphere_l273_273127


namespace find_k_l273_273824

def f (x : ℝ) := x^2 - 7 * x

theorem find_k : ∃ a h k : ℝ, f x = a * (x - h)^2 + k ∧ k = -49 / 4 := 
sorry

end find_k_l273_273824


namespace find_missing_dimension_l273_273086

def carton_volume (l w h : ℕ) : ℕ := l * w * h

def soapbox_base_area (l w : ℕ) : ℕ := l * w

def total_base_area (n l w : ℕ) : ℕ := n * soapbox_base_area l w

def missing_dimension (carton_volume total_base_area : ℕ) : ℕ := carton_volume / total_base_area

theorem find_missing_dimension 
  (carton_l carton_w carton_h : ℕ) 
  (soapbox_l soapbox_w : ℕ) 
  (n : ℕ) 
  (h_carton_l : carton_l = 25)
  (h_carton_w : carton_w = 48)
  (h_carton_h : carton_h = 60)
  (h_soapbox_l : soapbox_l = 8)
  (h_soapbox_w : soapbox_w = 6)
  (h_n : n = 300) :
  missing_dimension (carton_volume carton_l carton_w carton_h) (total_base_area n soapbox_l soapbox_w) = 5 := 
by 
  sorry

end find_missing_dimension_l273_273086


namespace sum_of_x_y_z_l273_273962

theorem sum_of_x_y_z (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = 2 * y) : x + y + z = 10 * x := by
  sorry

end sum_of_x_y_z_l273_273962


namespace length_after_5th_cut_l273_273095

theorem length_after_5th_cut (initial_length : ℝ) (n : ℕ) (h1 : initial_length = 1) (h2 : n = 5) :
  initial_length / 2^n = 1 / 2^5 := by
  sorry

end length_after_5th_cut_l273_273095


namespace algebraic_identity_l273_273298

theorem algebraic_identity (x : ℝ) (h : x = Real.sqrt 3 + 2) : x^2 - 4 * x + 3 = 2 := 
by
  -- proof steps here
  sorry

end algebraic_identity_l273_273298


namespace simplify_tan_expression_simplify_complex_expression_l273_273609

-- Problem 1
theorem simplify_tan_expression (α : ℝ) (hα : 0 < α ∧ α < 2 * π ∧ α > 3 / 2 * π) : 
  (Real.tan α + Real.sqrt ((1 / (Real.cos α)^2) - 1) + 2 * (Real.sin α)^2 + 2 * (Real.cos α)^2 = 2) :=
sorry

-- Problem 2
theorem simplify_complex_expression (α : ℝ) (hα : 0 < α ∧ α < 2 * π ∧ α > 3 / 2 * π) : 
  (Real.sin (α + π) * Real.tan (π - α) * Real.cos (2 * π - α) / (Real.sin (π - α) * Real.sin (π / 2 + α)) + Real.cos (5 * π / 2) = - Real.cos α) :=
sorry

end simplify_tan_expression_simplify_complex_expression_l273_273609


namespace prob_not_less_than_30_l273_273947

-- Define the conditions
def prob_less_than_30 : ℝ := 0.3
def prob_between_30_and_40 : ℝ := 0.5

-- State the theorem
theorem prob_not_less_than_30 (h1 : prob_less_than_30 = 0.3) : 1 - prob_less_than_30 = 0.7 :=
by
  sorry

end prob_not_less_than_30_l273_273947


namespace product_xy_min_value_x_plus_y_min_value_attained_l273_273314

theorem product_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 / x + 8 / y = 1) : x * y = 64 := 
sorry

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 / x + 8 / y = 1) : 
  x + y = 18 := 
sorry

-- Additional theorem to prove that the minimum value is attained when x = 6 and y = 12
theorem min_value_attained (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 / x + 8 / y = 1) :
  x = 6 ∧ y = 12 := 
sorry

end product_xy_min_value_x_plus_y_min_value_attained_l273_273314


namespace right_angled_triangle_not_axisymmetric_l273_273392

-- Define a type for geometric figures
inductive Figure
| Angle : Figure
| EquilateralTriangle : Figure
| LineSegment : Figure
| RightAngledTriangle : Figure

open Figure

-- Define a function to determine if a figure is axisymmetric
def is_axisymmetric: Figure -> Prop
| Angle => true
| EquilateralTriangle => true
| LineSegment => true
| RightAngledTriangle => false

-- Statement of the problem
theorem right_angled_triangle_not_axisymmetric : 
  is_axisymmetric RightAngledTriangle = false :=
by
  sorry

end right_angled_triangle_not_axisymmetric_l273_273392


namespace remainder_of_largest_divided_by_next_largest_l273_273234

/-
  Conditions:
  Let a = 10, b = 11, c = 12, d = 13.
  The largest number is d (13) and the next largest number is c (12).

  Question:
  What is the remainder when the largest number is divided by the next largest number?

  Answer:
  The remainder is 1.
-/

theorem remainder_of_largest_divided_by_next_largest :
  let a := 10 
  let b := 11
  let c := 12
  let d := 13
  d % c = 1 :=
by
  sorry

end remainder_of_largest_divided_by_next_largest_l273_273234


namespace four_ping_pong_four_shuttlecocks_cost_l273_273602

theorem four_ping_pong_four_shuttlecocks_cost
  (x y : ℝ)
  (h1 : 3 * x + 2 * y = 15.5)
  (h2 : 2 * x + 3 * y = 17) :
  4 * x + 4 * y = 26 :=
sorry

end four_ping_pong_four_shuttlecocks_cost_l273_273602


namespace sequence_k_value_l273_273324

theorem sequence_k_value
  (a : ℕ → ℕ)
  (h1 : a 1 = 2)
  (h2 : ∀ m n : ℕ, a (m + n) = a m * a n)
  (hk1 : ∀ k : ℕ, a (k + 1) = 1024) :
  ∃ k : ℕ, k = 9 :=
by {
  sorry
}

end sequence_k_value_l273_273324


namespace new_rectangle_area_l273_273547

theorem new_rectangle_area :
  let a := 3
  let b := 4
  let diagonal := Real.sqrt (a^2 + b^2)
  let sum_of_sides := a + b
  let area := diagonal * sum_of_sides
  area = 35 :=
by
  sorry

end new_rectangle_area_l273_273547


namespace chord_length_of_given_line_and_circle_l273_273270

noncomputable def chord_length_on_circle (r : ℝ) (line : ℝ → ℝ × ℝ) (circle : ℝ → ℝ × ℝ) : ℝ :=
let a := 1 in
let b := 1 in
let c := -2 in
let d := abs (a * 0 + b * 0 + c) / real.sqrt (a^2 + b^2) in
let half_chord_length := real.sqrt (r^2 - d^2) in
2 * half_chord_length

theorem chord_length_of_given_line_and_circle :
  chord_length_on_circle 3 (λ t : ℝ, (1 + 2*t, 1 - 2*t)) (λ α : ℝ, (3 * real.cos α, 3 * real.sin α)) = 2 * real.sqrt 7 :=
by
  sorry

end chord_length_of_given_line_and_circle_l273_273270


namespace solve_system_l273_273213

theorem solve_system :
  ∃ (x y : ℝ), (x^2 + y^2 ≤ 1) ∧ (x^4 - 18 * x^2 * y^2 + 81 * y^4 - 20 * x^2 - 180 * y^2 + 100 = 0) ∧
    ((x = -1 / Real.sqrt 10 ∧ y = 3 / Real.sqrt 10) ∨
    (x = -1 / Real.sqrt 10 ∧ y = -3 / Real.sqrt 10) ∨
    (x = 1 / Real.sqrt 10 ∧ y = 3 / Real.sqrt 10) ∨
    (x = 1 / Real.sqrt 10 ∧ y = -3 / Real.sqrt 10)) :=
  by
  sorry

end solve_system_l273_273213


namespace range_of_m_l273_273150

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x^2 - m * x + 1 > 0) ↔ (-2 < m ∧ m < 2) :=
  sorry

end range_of_m_l273_273150


namespace average_study_difference_is_6_l273_273320

def study_time_differences : List ℤ := [15, -5, 25, -10, 40, -30, 10]

def total_sum (lst : List ℤ) : ℤ := lst.foldr (· + ·) 0

def number_of_days : ℤ := 7

def average_difference : ℤ := total_sum study_time_differences / number_of_days

theorem average_study_difference_is_6 : average_difference = 6 :=
by
  unfold average_difference
  unfold total_sum 
  sorry

end average_study_difference_is_6_l273_273320


namespace cannot_achieve_61_cents_with_six_coins_l273_273718

theorem cannot_achieve_61_cents_with_six_coins :
  ¬ ∃ (p n d q : ℕ), 
      p + n + d + q = 6 ∧ 
      p + 5 * n + 10 * d + 25 * q = 61 :=
by
  sorry

end cannot_achieve_61_cents_with_six_coins_l273_273718


namespace eval_sin_570_l273_273788

theorem eval_sin_570:
  2 * Real.sin (570 * Real.pi / 180) = -1 := 
by sorry

end eval_sin_570_l273_273788


namespace intersection_M_N_l273_273668

def M : Set ℝ := { x | -1 < x ∧ x < 1 }
def N : Set ℝ := { x | x / (x - 1) ≤ 0 }

theorem intersection_M_N :
  M ∩ N = { x | 0 ≤ x ∧ x < 1 } :=
by
  sorry

end intersection_M_N_l273_273668


namespace coins_in_second_stack_l273_273112

theorem coins_in_second_stack (total_coins : ℕ) (stack1_coins : ℕ) (stack2_coins : ℕ) 
  (H1 : total_coins = 12) (H2 : stack1_coins = 4) : stack2_coins = 8 :=
by
  -- The proof is omitted.
  sorry

end coins_in_second_stack_l273_273112


namespace negation_of_proposition_l273_273225

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, 0 < x → (x^2 + x > 0)) ↔ ∃ x : ℝ, 0 < x ∧ (x^2 + x ≤ 0) :=
sorry

end negation_of_proposition_l273_273225


namespace rectangular_coordinates_2_pi_3_to_rectangular_l273_273733

noncomputable theory

def polar_to_rectangular (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

def tan_pos_quad3 (θ : ℝ) (x y : ℝ) : Prop :=
  ∃ k : ℤ, θ = Real.pi * (4 / 3) + 2 * Real.pi * k ∧ y / x = Real.tan (4 * Real.pi / 3)

theorem rectangular_coordinates_2_pi_3_to_rectangular :
  polar_to_rectangular 2 (4 * Real.pi / 3) = (-1, -Real.sqrt 3) :=
by
  sorry

end rectangular_coordinates_2_pi_3_to_rectangular_l273_273733


namespace min_value_expression_l273_273334

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c + 1) * (1 / (a + b + 1) + 1 / (b + c + 1) + 1 / (c + a + 1)) ≥ 9 / 2 :=
sorry

end min_value_expression_l273_273334


namespace compute_expression_l273_273573

-- Definition of the imaginary unit i
class ImaginaryUnit (i : ℂ) where
  I_square : i * i = -1

-- Definition of non-zero real number a
variable (a : ℝ) (h_a : a ≠ 0)

-- Theorem to prove the equivalence
theorem compute_expression (i : ℂ) [ImaginaryUnit i] :
  (a * i - i⁻¹)⁻¹ = -i / (a + 1) :=
by
  sorry

end compute_expression_l273_273573


namespace converse_statement_2_true_implies_option_A_l273_273146

theorem converse_statement_2_true_implies_option_A :
  (∀ x : ℕ, x = 1 ∨ x = 2 → (x^2 - 3 * x + 2 = 0)) →
  (x = 1 ∨ x = 2) :=
by
  intro h
  sorry

end converse_statement_2_true_implies_option_A_l273_273146


namespace smallest_base10_num_exists_l273_273889

noncomputable def A_digit_range := {n : ℕ // n < 6}
noncomputable def B_digit_range := {n : ℕ // n < 8}

theorem smallest_base10_num_exists :
  ∃ (A : A_digit_range) (B : B_digit_range), 7 * A.val = 9 * B.val ∧ 7 * A.val = 63 :=
by
  sorry

end smallest_base10_num_exists_l273_273889


namespace b_2016_result_l273_273159

theorem b_2016_result (b : ℕ → ℤ) (h₁ : b 1 = 1) (h₂ : b 2 = 5)
  (h₃ : ∀ n : ℕ, b (n + 2) = b (n + 1) - b n) : b 2016 = -4 := sorry

end b_2016_result_l273_273159


namespace max_grapes_discarded_l273_273612

theorem max_grapes_discarded (n : ℕ) : 
  ∃ k : ℕ, k ∣ n → 7 * k + 6 = n → ∃ m, m = 6 := by
  sorry

end max_grapes_discarded_l273_273612


namespace AM_GM_inequality_min_value_l273_273841

theorem AM_GM_inequality_min_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / b)^2 + (b / c)^2 + (c / a)^2 ≥ 3 :=
by
  sorry

end AM_GM_inequality_min_value_l273_273841


namespace sum_of_extreme_T_l273_273291

theorem sum_of_extreme_T (B M T : ℝ) 
  (h1 : B^2 + M^2 + T^2 = 2022)
  (h2 : B + M + T = 72) :
  ∃ Tmin Tmax, Tmin + Tmax = 48 ∧ Tmin ≤ T ∧ T ≤ Tmax :=
by
  sorry

end sum_of_extreme_T_l273_273291


namespace acute_angle_sum_eq_pi_div_two_l273_273661

open Real

theorem acute_angle_sum_eq_pi_div_two (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h_eq : sin α ^ 2 + sin β ^ 2 = sin (α + β)) : 
  α + β = π / 2 :=
sorry

end acute_angle_sum_eq_pi_div_two_l273_273661


namespace polygon_sides_l273_273030

theorem polygon_sides (n : ℕ) :
  ((n - 2) * 180 = 4 * 360) → n = 10 :=
by 
  sorry

end polygon_sides_l273_273030


namespace remainder_of_product_divided_by_10_l273_273061

theorem remainder_of_product_divided_by_10 :
  let a := 2457
  let b := 6273
  let c := 91409
  (a * b * c) % 10 = 9 :=
by
  sorry

end remainder_of_product_divided_by_10_l273_273061


namespace calculate_expression_l273_273269

theorem calculate_expression :
  (Int.floor ((15:ℚ)/8 * ((-34:ℚ)/4)) - Int.ceil ((15:ℚ)/8 * Int.floor ((-34:ℚ)/4))) = 0 := 
  by sorry

end calculate_expression_l273_273269


namespace train_length_calculation_l273_273518

noncomputable def length_of_train (time : ℝ) (speed_kmh : ℝ) : ℝ :=
  let speed_ms := speed_kmh * (1000 / 3600)
  speed_ms * time

theorem train_length_calculation : 
  length_of_train 4.99960003199744 72 = 99.9920006399488 :=
by 
  sorry  -- proof of the actual calculation

end train_length_calculation_l273_273518


namespace committee_count_l273_273619

theorem committee_count (club_members : Finset ℕ) (h_count : club_members.card = 30) :
  ∃ committee_count : ℕ, committee_count = 2850360 :=
by
  sorry

end committee_count_l273_273619


namespace angles_at_point_l273_273562

theorem angles_at_point (x y : ℝ) 
  (h1 : x + y + 120 = 360) 
  (h2 : x = 2 * y) : 
  x = 160 ∧ y = 80 :=
by
  sorry

end angles_at_point_l273_273562


namespace coffee_price_l273_273756

theorem coffee_price (qd : ℝ) (d : ℝ) (rp : ℝ) :
  qd = 4.5 ∧ d = 0.25 → rp = 12 :=
by 
  sorry

end coffee_price_l273_273756


namespace misha_class_predictions_probability_l273_273078

-- Definitions representing the conditions
def monday_classes : ℕ := 5
def tuesday_classes : ℕ := 6
def total_classes : ℕ := monday_classes + tuesday_classes
def total_flips : ℕ := total_classes
def correct_predictions : ℕ := 7
def correct_monday_predictions : ℕ := 3

-- Lean theorem representing the proof problem
theorem misha_class_predictions_probability :
  (probability_of_correct_predictions total_flips correct_predictions monday_classes correct_monday_predictions) =
    (5 / 11) :=
  sorry

end misha_class_predictions_probability_l273_273078


namespace solve_system_a_solve_system_b_l273_273859

-- For problem (a):
theorem solve_system_a (x y : ℝ) :
  (x + y + x * y = 5) ∧ (x * y * (x + y) = 6) → 
  (x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2) := 
by
  sorry

-- For problem (b):
theorem solve_system_b (x y : ℝ) :
  (x^3 + y^3 + 2 * x * y = 4) ∧ (x^2 - x * y + y^2 = 1) → 
  (x = 1 ∧ y = 1) := 
by
  sorry

end solve_system_a_solve_system_b_l273_273859


namespace MattRate_l273_273451

variable (M : ℝ) (t : ℝ)

def MattRateCondition : Prop := M * t = 220
def TomRateCondition : Prop := (M + 5) * t = 275

theorem MattRate (h1 : MattRateCondition M t) (h2 : TomRateCondition M t) : M = 20 := by
  sorry

end MattRate_l273_273451


namespace find_k_l273_273482

theorem find_k (k : ℝ) (h1 : k > 0) (h2 : |3 * (k^2 - 9) - 2 * (4 * k - 15) + 2 * (12 - 5 * k)| = 20) : k = 4 := by
  sorry

end find_k_l273_273482


namespace laura_change_l273_273331

theorem laura_change : 
  let pants_cost := 2 * 54
  let shirts_cost := 4 * 33
  let total_cost := pants_cost + shirts_cost
  let amount_given := 250
  (amount_given - total_cost) = 10 :=
by
  -- definitions from conditions
  let pants_cost := 2 * 54
  let shirts_cost := 4 * 33
  let total_cost := pants_cost + shirts_cost
  let amount_given := 250

  -- the statement we are proving
  show (amount_given - total_cost) = 10
  sorry

end laura_change_l273_273331


namespace calc_f_18_48_l273_273575

def f (x y : ℕ) : ℕ := sorry

axiom f_self (x : ℕ) : f x x = x
axiom f_symm (x y : ℕ) : f x y = f y x
axiom f_third_cond (x y : ℕ) : (x + y) * f x y = x * f x (x + y)

theorem calc_f_18_48 : f 18 48 = 48 := sorry

end calc_f_18_48_l273_273575


namespace bianca_total_books_l273_273607

theorem bianca_total_books (shelves_mystery shelves_picture books_per_shelf : ℕ) 
  (h1 : shelves_mystery = 5) 
  (h2 : shelves_picture = 4) 
  (h3 : books_per_shelf = 8) : 
  (shelves_mystery * books_per_shelf + shelves_picture * books_per_shelf) = 72 := 
by 
  sorry

end bianca_total_books_l273_273607


namespace odd_function_property_l273_273524

-- Define that f is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- The theorem statement
theorem odd_function_property (f : ℝ → ℝ) (h : is_odd_function f) : ∀ x : ℝ, f x * f (-x) ≤ 0 :=
by
  -- The proof is omitted as per the instruction
  sorry

end odd_function_property_l273_273524


namespace ways_to_go_from_first_to_fifth_l273_273232

theorem ways_to_go_from_first_to_fifth (floors : ℕ) (staircases_per_floor : ℕ) (total_ways : ℕ) 
    (h1 : floors = 5) (h2 : staircases_per_floor = 2) (h3 : total_ways = 2^4) : total_ways = 16 :=
by
  sorry

end ways_to_go_from_first_to_fifth_l273_273232


namespace transformed_sine_eqn_l273_273360

theorem transformed_sine_eqn (ω : ℝ) (φ : ℝ) : 
(ω > 0) ∧ (|φ| < (Real.pi / 2)) ∧ 
(∀ x, sin (ω * (2 * (x - Real.pi / 3)) + φ) = sin x) ↔ (ω = 1/2) ∧ (φ = Real.pi / 6) := 
by sorry

end transformed_sine_eqn_l273_273360


namespace ocean_depth_l273_273849

/-
  Problem:
  Determine the depth of the ocean at the current location of the ship.
  
  Given conditions:
  - The signal sent by the echo sounder was received after 5 seconds.
  - The speed of sound in water is 1.5 km/s.

  Correct answer to prove:
  - The depth of the ocean is 3750 meters.
-/

theorem ocean_depth
  (v : ℝ) (t : ℝ) (depth : ℝ) 
  (hv : v = 1500) 
  (ht : t = 5) 
  (hdepth : depth = 3750) :
  depth = (v * t) / 2 :=
sorry

end ocean_depth_l273_273849


namespace max_wickets_bowler_can_take_l273_273754

noncomputable def max_wickets_per_over : ℕ := 3
noncomputable def overs_bowled : ℕ := 6
noncomputable def max_possible_wickets := max_wickets_per_over * overs_bowled

theorem max_wickets_bowler_can_take : max_possible_wickets = 18 → max_possible_wickets == 10 :=
by
  sorry

end max_wickets_bowler_can_take_l273_273754


namespace loom_weaving_rate_l273_273635

noncomputable def total_cloth : ℝ := 27
noncomputable def total_time : ℝ := 210.9375

theorem loom_weaving_rate :
  (total_cloth / total_time) = 0.128 :=
by
  sorry

end loom_weaving_rate_l273_273635


namespace largest_n_binary_operation_l273_273315

-- Define the binary operation @
def binary_operation (n : ℤ) : ℤ := n - (n * 5)

-- Define the theorem stating the desired property
theorem largest_n_binary_operation (x : ℤ) (h : x > -8) :
  ∃ (n : ℤ), n = 2 ∧ binary_operation n < x :=
sorry

end largest_n_binary_operation_l273_273315


namespace max_students_received_less_than_given_l273_273399

def max_students_received_less := 27
def max_possible_n := 13

theorem max_students_received_less_than_given (n : ℕ) :
  n <= max_students_received_less -> n = max_possible_n :=
sorry
 
end max_students_received_less_than_given_l273_273399


namespace simplify_tan_product_l273_273210

noncomputable def tan_deg (d : ℝ) : ℝ := Real.tan (d * Real.pi / 180)

theorem simplify_tan_product :
  (1 + tan_deg 10) * (1 + tan_deg 35) = 2 := 
by
  -- Given conditions
  have h1 : Real.tan (Real.pi / 4) = 1 := Real.tan_pi_div_four
  have h2 : tan_deg 10 + tan_deg 35 = 1 - tan_deg 10 * tan_deg 35 :=
    by sorry -- Use tan addition formula here
  -- Proof of the theorem follows from here
  sorry

end simplify_tan_product_l273_273210


namespace garden_area_is_correct_l273_273641

def width_of_property : ℕ := 1000
def length_of_property : ℕ := 2250

def width_of_garden : ℕ := width_of_property / 8
def length_of_garden : ℕ := length_of_property / 10

def area_of_garden : ℕ := width_of_garden * length_of_garden

theorem garden_area_is_correct : area_of_garden = 28125 := by
  -- Skipping proof for the purpose of this example
  sorry

end garden_area_is_correct_l273_273641


namespace volume_of_tetrahedron_OABC_l273_273485

-- Definitions of side lengths and their squared values
def side_length_A_B := 7
def side_length_B_C := 8
def side_length_C_A := 9

-- Squared values of coordinates
def a_sq := 33
def b_sq := 16
def c_sq := 48

-- Main statement to prove the volume
theorem volume_of_tetrahedron_OABC :
  (1/6) * (Real.sqrt a_sq) * (Real.sqrt b_sq) * (Real.sqrt c_sq) = 2 * Real.sqrt 176 :=
by
  -- Proof steps would go here
  sorry

end volume_of_tetrahedron_OABC_l273_273485


namespace cost_of_two_sandwiches_l273_273834

theorem cost_of_two_sandwiches (J S : ℝ) 
  (h1 : 5 * J = 10) 
  (h2 : S + J = 5) :
  2 * S = 6 := 
sorry

end cost_of_two_sandwiches_l273_273834


namespace rachelle_gpa_l273_273769

noncomputable def points (grade : ℕ) : ℚ := 
  if grade = 1 then 4 else 
  if grade = 2 then 3 else 
  if grade = 3 then 2 else 1

noncomputable def gpa (total_points : ℚ) : ℚ := total_points / 4

def prob_of_grades_english : ℕ → ℚ
| 1 := 1/3
| 2 := 1/4
| 3 := 5/12
| _ := 0

def prob_of_grades_history : ℕ → ℚ
| 1 := 1/5
| 2 := 2/5
| 3 := 2/5
| _ := 0

noncomputable def final_probability : ℚ :=
  let pa_e := prob_of_grades_english 1
  let pb_e := prob_of_grades_english 2
  let pa_h := prob_of_grades_history 1
  let pb_h := prob_of_grades_history 2
  in pa_e * pa_h + pa_e * pb_h + pb_e * pa_h

theorem rachelle_gpa : final_probability = 1/4 :=
by sorry

end rachelle_gpa_l273_273769


namespace range_of_a_l273_273122

theorem range_of_a (a : ℝ) : (∀ x, 1 ≤ x ∧ x ≤ 3 → x^2 - a * x - 3 ≤ 0) ↔ (2 ≤ a) := by
  sorry

end range_of_a_l273_273122


namespace ball_attendance_l273_273177

variable (n m : ℕ)

def ball_conditions (n m : ℕ) := 
  n + m < 50 ∧ 
  4 ∣ 3 * n ∧ 
  7 ∣ 5 * m

theorem ball_attendance (n m : ℕ) (h : ball_conditions n m) : 
  n + m = 41 :=
sorry

end ball_attendance_l273_273177


namespace anita_smallest_number_of_candies_l273_273394

theorem anita_smallest_number_of_candies :
  ∃ x : ℕ, x ≡ 5 [MOD 6] ∧ x ≡ 3 [MOD 8] ∧ x ≡ 7 [MOD 9] ∧ ∀ y : ℕ,
  (y ≡ 5 [MOD 6] ∧ y ≡ 3 [MOD 8] ∧ y ≡ 7 [MOD 9]) → x ≤ y :=
  ⟨203, by sorry⟩

end anita_smallest_number_of_candies_l273_273394


namespace product_equals_9_l273_273646

theorem product_equals_9 :
  (1 + (1 / 1)) * (1 + (1 / 2)) * (1 + (1 / 3)) * (1 + (1 / 4)) * 
  (1 + (1 / 5)) * (1 + (1 / 6)) * (1 + (1 / 7)) * (1 + (1 / 8)) = 9 := 
by
  sorry

end product_equals_9_l273_273646


namespace slope_of_chord_l273_273831

theorem slope_of_chord (x1 x2 y1 y2 : ℝ) (P : ℝ × ℝ)
    (hp : P = (3, 2))
    (h1 : 4 * x1 ^ 2 + 9 * y1 ^ 2 = 144)
    (h2 : 4 * x2 ^ 2 + 9 * y2 ^ 2 = 144)
    (h3 : (x1 + x2) / 2 = 3)
    (h4 : (y1 + y2) / 2 = 2) : 
    (y1 - y2) / (x1 - x2) = -2 / 3 :=
by
  sorry

end slope_of_chord_l273_273831


namespace remainder_of_2n_div_9_l273_273688

theorem remainder_of_2n_div_9
  (n : ℤ) (h : ∃ k : ℤ, n = 18 * k + 10) : (2 * n) % 9 = 2 := 
by
  sorry

end remainder_of_2n_div_9_l273_273688


namespace f_monotonic_increasing_l273_273552

noncomputable def f (x : ℝ) : ℝ := 2 - 3 / x

theorem f_monotonic_increasing :
  ∀ (x1 x2 : ℝ), 0 < x1 → 0 < x2 → x1 > x2 → f x1 > f x2 :=
by
  intros x1 x2 hx1 hx2 h
  sorry

end f_monotonic_increasing_l273_273552


namespace train_speed_l273_273597

theorem train_speed (v : ℝ) (h1 : 60 * 6.5 + v * 6.5 = 910) : v = 80 := 
sorry

end train_speed_l273_273597


namespace value_of_t_l273_273555

def vec (x y : ℝ) := (x, y)

def p := vec 3 3
def q := vec (-1) 2
def r := vec 4 1

noncomputable def t := 3

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem value_of_t (t : ℝ) : (dot_product (vec (6 + 4 * t) (6 + t)) q = 0) ↔ t = 3 :=
by
  sorry

end value_of_t_l273_273555


namespace solution_of_inequality_l273_273026

theorem solution_of_inequality (x : ℝ) : x * (x - 1) < 2 ↔ -1 < x ∧ x < 2 :=
sorry

end solution_of_inequality_l273_273026


namespace simplify_tangent_expression_l273_273205

theorem simplify_tangent_expression :
  (1 + Real.tan (Real.pi / 18)) * (1 + Real.tan (35 * Real.pi / 180)) = 2 :=
by
  sorry

end simplify_tangent_expression_l273_273205


namespace divisibility_by_P_divisibility_by_P_squared_divisibility_by_P_cubed_l273_273900

noncomputable def Q (x : ℝ) (n : ℕ) : ℝ := (x + 1)^n - x^n - 1

def P (x : ℝ) : ℝ := x^2 + x + 1

-- Prove Q(x, n) is divisible by P(x) if and only if n ≡ 1 or 5 (mod 6)
theorem divisibility_by_P (x : ℝ) (n : ℕ) : 
  (Q x n) % (P x) = 0 ↔ (n % 6 = 1 ∨ n % 6 = 5) := 
sorry

-- Prove Q(x, n) is divisible by P(x)^2 if and only if n ≡ 1 (mod 6)
theorem divisibility_by_P_squared (x : ℝ) (n : ℕ) : 
  (Q x n) % (P x)^2 = 0 ↔ n % 6 = 1 := 
sorry

-- Prove Q(x, n) is divisible by P(x)^3 if and only if n = 1
theorem divisibility_by_P_cubed (x : ℝ) (n : ℕ) : 
  (Q x n) % (P x)^3 = 0 ↔ n = 1 := 
sorry

end divisibility_by_P_divisibility_by_P_squared_divisibility_by_P_cubed_l273_273900


namespace man_speed_proof_l273_273258

noncomputable def train_length : ℝ := 150 
noncomputable def crossing_time : ℝ := 6 
noncomputable def train_speed_kmph : ℝ := 84.99280057595394 
noncomputable def train_speed_mps : ℝ := train_speed_kmph * (1000 / 3600)

noncomputable def relative_speed_mps : ℝ := train_length / crossing_time
noncomputable def man_speed_mps : ℝ := relative_speed_mps - train_speed_mps
noncomputable def man_speed_kmph : ℝ := man_speed_mps * (3600 / 1000)

theorem man_speed_proof : man_speed_kmph = 5.007198224048459 := by 
  sorry

end man_speed_proof_l273_273258


namespace sufficient_but_not_necessary_condition_l273_273951

def p (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 3
def q (x : ℝ) : Prop := x ≠ 0

theorem sufficient_but_not_necessary_condition (h: ∀ x : ℝ, p x → q x) : (∀ x : ℝ, q x → p x) → false := sorry

end sufficient_but_not_necessary_condition_l273_273951


namespace sin_polar_circle_l273_273727

theorem sin_polar_circle (t : ℝ) (θ : ℝ) (r : ℝ) (h : ∀ θ, 0 ≤ θ ∧ θ ≤ t → r = Real.sin θ) :
  t = Real.pi := 
by
  sorry

end sin_polar_circle_l273_273727


namespace greatest_integer_difference_l273_273147

theorem greatest_integer_difference (x y : ℤ) (hx : 3 < x ∧ x < 6) (hy : 6 < y ∧ y < 7) : y - x = 3 :=
sorry

end greatest_integer_difference_l273_273147


namespace jennifer_book_fraction_l273_273567

theorem jennifer_book_fraction :
  (120 - (1/5 * 120 + 1/6 * 120 + 16)) / 120 = 1/2 :=
by
  sorry

end jennifer_book_fraction_l273_273567


namespace complement_intersection_l273_273312

-- Definitions of the sets as given in the problem
namespace ProofProblem

def U : Set ℤ := {-2, -1, 0, 1, 2}
def M : Set ℤ := {y | y > 0}
def N : Set ℤ := {x | x = -1 ∨ x = 2}

theorem complement_intersection :
  (U \ M) ∩ N = {-1} :=
by
  sorry

end ProofProblem

end complement_intersection_l273_273312


namespace encode_mathematics_l273_273630

def robotCipherMapping : String → String := sorry

theorem encode_mathematics :
  robotCipherMapping "MATHEMATICS" = "2232331122323323132" := sorry

end encode_mathematics_l273_273630


namespace regular_decagon_triangle_count_l273_273778

theorem regular_decagon_triangle_count :
  ∃ n, (n = 10) ∧ nat.choose 10 3 = 120 :=
by
  use 10
  split
  · rfl
  · exact nat.choose_succ_succ_succ 7 2

end regular_decagon_triangle_count_l273_273778


namespace james_hours_per_day_l273_273328

theorem james_hours_per_day (h : ℕ) (rental_rate : ℕ) (days_per_week : ℕ) (weekly_income : ℕ)
  (H1 : rental_rate = 20)
  (H2 : days_per_week = 4)
  (H3 : weekly_income = 640)
  (H4 : rental_rate * days_per_week * h = weekly_income) :
  h = 8 :=
sorry

end james_hours_per_day_l273_273328


namespace carla_glasses_lemonade_l273_273199

theorem carla_glasses_lemonade (time_total : ℕ) (rate : ℕ) (glasses : ℕ) 
  (h1 : time_total = 3 * 60 + 40) 
  (h2 : rate = 20) 
  (h3 : glasses = time_total / rate) : 
  glasses = 11 := 
by 
  -- We'll fill in the proof here in a real scenario
  sorry

end carla_glasses_lemonade_l273_273199


namespace solve_for_a_l273_273817

theorem solve_for_a (a : ℝ) (h : a / 0.3 = 0.6) : a = 0.18 :=
by sorry

end solve_for_a_l273_273817


namespace additional_money_needed_l273_273103

-- Define the initial conditions as assumptions
def initial_bales : ℕ := 15
def previous_cost_per_bale : ℕ := 20
def multiplier : ℕ := 3
def new_cost_per_bale : ℕ := 27

-- Define the problem statement
theorem additional_money_needed :
  let initial_cost := initial_bales * previous_cost_per_bale 
  let new_bales := initial_bales * multiplier
  let new_cost := new_bales * new_cost_per_bale
  new_cost - initial_cost = 915 :=
by
  sorry

end additional_money_needed_l273_273103


namespace intersection_y_condition_l273_273152

theorem intersection_y_condition (a : ℝ) :
  (∃ x y : ℝ, 2 * x - a * y + 2 = 0 ∧ x + y = 0 ∧ y < 0) → a < -2 :=
by
  sorry

end intersection_y_condition_l273_273152


namespace alice_probability_l273_273260

def alice_probability_starting_at_zero (n : ℕ) (p q : ℚ) : ℚ := 
  (p/q)

theorem alice_probability (a b : ℕ) (h : Nat.coprime a b) :
  alice_probability_starting_at_zero 10 15 64 = (15 / 64 : ℚ) ∧ a + b = 79 :=
by
  sorry

end alice_probability_l273_273260


namespace percentage_not_caught_l273_273434

theorem percentage_not_caught (x : ℝ) (h1 : 22 + x = 25.88235294117647) : x = 3.88235294117647 :=
sorry

end percentage_not_caught_l273_273434


namespace circle_center_coordinates_l273_273725

theorem circle_center_coordinates :
  ∃ (h k : ℝ), (∀ x y : ℝ, x^2 + y^2 - 4 * x + 6 * y = 0 ↔ (x - h)^2 + (y - k)^2 = 13) ∧ h = 2 ∧ k = -3 :=
sorry

end circle_center_coordinates_l273_273725


namespace renne_can_buy_vehicle_in_8_months_l273_273712

def monthly_earnings := 4000
def savings_rate := 0.5
def vehicle_cost := 16000
def monthly_savings := monthly_earnings * savings_rate
def months_to_save := vehicle_cost / monthly_savings

theorem renne_can_buy_vehicle_in_8_months : months_to_save = 8 := 
by 
  -- Proof is not required as per the task instruction
  sorry

end renne_can_buy_vehicle_in_8_months_l273_273712


namespace avg_salary_increases_by_150_l273_273467

def avg_salary_increase
  (emp_avg_salary : ℕ) (num_employees : ℕ) (mgr_salary : ℕ) : ℕ :=
  let total_salary_employees := emp_avg_salary * num_employees
  let total_salary_with_mgr := total_salary_employees + mgr_salary
  let new_avg_salary := total_salary_with_mgr / (num_employees + 1)
  new_avg_salary - emp_avg_salary

theorem avg_salary_increases_by_150 :
  avg_salary_increase 1800 15 4200 = 150 :=
by
  sorry

end avg_salary_increases_by_150_l273_273467


namespace seq_solution_l273_273967

theorem seq_solution {a b : ℝ} (h1 : a - b = 8) (h2 : a + b = 11) : 2 * a = 19 ∧ 2 * b = 3 := by
  sorry

end seq_solution_l273_273967


namespace sheep_count_l273_273605

theorem sheep_count (S H : ℕ) (h1 : S / H = 3 / 7) (h2 : H * 230 = 12880) : S = 24 :=
by
  sorry

end sheep_count_l273_273605


namespace number_of_employees_l273_273785

-- Definitions
def emily_original_salary : ℕ := 1000000
def emily_new_salary : ℕ := 850000
def employee_original_salary : ℕ := 20000
def employee_new_salary : ℕ := 35000
def salary_difference : ℕ := emily_original_salary - emily_new_salary
def salary_increase_per_employee : ℕ := employee_new_salary - employee_original_salary

-- Theorem: Prove Emily has n employees where n = 10
theorem number_of_employees : salary_difference / salary_increase_per_employee = 10 :=
by sorry

end number_of_employees_l273_273785


namespace sequence_k_value_l273_273323

theorem sequence_k_value
  (a : ℕ → ℕ)
  (h1 : a 1 = 2)
  (h2 : ∀ m n : ℕ, a (m + n) = a m * a n)
  (hk1 : ∀ k : ℕ, a (k + 1) = 1024) :
  ∃ k : ℕ, k = 9 :=
by {
  sorry
}

end sequence_k_value_l273_273323


namespace range_of_m_l273_273685

theorem range_of_m (m : ℝ) : (∀ x : ℝ, |x + 4| + |x + 8| ≥ m) → m ≤ 4 :=
by
  sorry

end range_of_m_l273_273685


namespace power_of_b_l273_273994

theorem power_of_b (b n : ℕ) (hb : b > 1) (hn : n > 1) (h : ∀ k > 1, ∃ a_k : ℤ, k ∣ (b - a_k ^ n)) :
  ∃ A : ℤ, b = A ^ n :=
by
  sorry

end power_of_b_l273_273994


namespace toothpicks_at_200th_stage_l273_273587

-- Define initial number of toothpicks at stage 1
def a_1 : ℕ := 4

-- Define the function to compute the number of toothpicks at stage n, taking into account the changing common difference
def a (n : ℕ) : ℕ :=
  if n = 1 then 4
  else if n <= 49 then 4 + 4 * (n - 1)
  else if n <= 99 then 200 + 5 * (n - 50)
  else if n <= 149 then 445 + 6 * (n - 100)
  else if n <= 199 then 739 + 7 * (n - 150)
  else 0  -- This covers cases not considered in the problem for clarity

-- State the theorem to check the number of toothpicks at stage 200
theorem toothpicks_at_200th_stage : a 200 = 1082 :=
  sorry

end toothpicks_at_200th_stage_l273_273587


namespace parabola_translation_l273_273237

theorem parabola_translation :
  ∀ x y, (y = -2 * x^2) →
    ∃ x' y', y' = -2 * (x' - 2)^2 + 1 ∧ x' = x ∧ y' = y + 1 :=
sorry

end parabola_translation_l273_273237


namespace part_one_part_two_l273_273938

def f (x : ℝ) (a : ℝ) := |2 * x - a| + a
def g (x : ℝ) := |2 * x - 1|

-- (1) Prove that if a = 2, then ∀ x, f(x, 2) ≤ 6 implies -1 ≤ x ≤ 3
theorem part_one (x : ℝ) : f x 2 ≤ 6 → -1 ≤ x ∧ x ≤ 3 :=
by sorry

-- (2) Prove that ∀ a ∈ ℝ, ∀ x ∈ ℝ, (f(x, a) + g(x) ≥ 3 → a ∈ [2, +∞))
theorem part_two (a x : ℝ) : f x a + g x ≥ 3 → 2 ≤ a :=
by sorry

end part_one_part_two_l273_273938


namespace multiples_of_4_between_50_and_300_l273_273968

theorem multiples_of_4_between_50_and_300 : 
  (∃ n : ℕ, 50 < n ∧ n < 300 ∧ n % 4 = 0) ∧ 
  (∃ k : ℕ, k = 62) :=
by
  sorry

end multiples_of_4_between_50_and_300_l273_273968


namespace children_on_bus_after_stops_l273_273921

-- Define the initial number of children and changes at each stop
def initial_children := 128
def first_stop_addition := 67
def second_stop_subtraction := 34
def third_stop_addition := 54

-- Prove that the number of children on the bus after all the stops is 215
theorem children_on_bus_after_stops :
  initial_children + first_stop_addition - second_stop_subtraction + third_stop_addition = 215 := by
  -- The proof is omitted
  sorry

end children_on_bus_after_stops_l273_273921


namespace quadratic_real_roots_iff_l273_273798

/-- For the quadratic equation x^2 + 3x + m = 0 to have two real roots,
    the value of m must satisfy m ≤ 9/4. -/
theorem quadratic_real_roots_iff (m : ℝ) : (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 * x2 = m ∧ x1 + x2 = -3) ↔ m ≤ 9 / 4 :=
by
  sorry

end quadratic_real_roots_iff_l273_273798


namespace hannah_total_payment_l273_273140

def washing_machine_cost : ℝ := 100
def dryer_cost : ℝ := washing_machine_cost - 30
def total_cost_before_discount : ℝ := washing_machine_cost + dryer_cost
def discount : ℝ := 0.10
def total_cost_after_discount : ℝ := total_cost_before_discount * (1 - discount)

theorem hannah_total_payment : total_cost_after_discount = 153 := by
  sorry

end hannah_total_payment_l273_273140


namespace difference_of_squares_l273_273891

def a : ℕ := 601
def b : ℕ := 597

theorem difference_of_squares : a^2 - b^2 = 4792 :=
by {
  sorry
}

end difference_of_squares_l273_273891


namespace mark_cans_count_l273_273626

-- Given definitions and conditions
def rachel_cans : Nat := x  -- Rachel's cans
def jaydon_cans (x : Nat) : Nat := 5 + 2 * x  -- Jaydon's cans (y)
def mark_cans (y : Nat) : Nat := 4 * y  -- Mark's cans (z)

-- Total cans equation
def total_cans (x y z : Nat) : Prop := x + y + z = 135

-- Main statement to prove
theorem mark_cans_count (x : Nat) (y := jaydon_cans x) (z := mark_cans y) (h : total_cans x y z) : z = 100 :=
sorry

end mark_cans_count_l273_273626


namespace number_of_elements_in_list_l273_273274

theorem number_of_elements_in_list :
  let a := 2.5
  let d := 5.0
  let l := 62.5
  ∃ n : ℕ, l = a + (n - 1) * d ∧ n = 13 :=
begin
  sorry
end

end number_of_elements_in_list_l273_273274


namespace general_term_formula_l273_273417

def sequence_sums (n : ℕ) : ℕ := 2 * n^2 + n

theorem general_term_formula (a : ℕ → ℕ) (S : ℕ → ℕ) (hS : S = sequence_sums) :
  (∀ n, a n = S n - S (n-1)) → ∀ n, a n = 4 * n - 1 :=
by
  sorry

end general_term_formula_l273_273417


namespace find_k_l273_273325

theorem find_k (a : ℕ → ℕ) (h₀ : a 1 = 2) (h₁ : ∀ m n, a (m + n) = a m * a n) (hk : a (k + 1) = 1024) : k = 9 := 
sorry

end find_k_l273_273325


namespace gwen_received_more_money_from_mom_l273_273943

theorem gwen_received_more_money_from_mom :
  let mom_money := 8
  let dad_money := 5
  mom_money - dad_money = 3 :=
by
  sorry

end gwen_received_more_money_from_mom_l273_273943


namespace profit_percentage_l273_273080

theorem profit_percentage (purchase_price sell_price : ℝ) (h1 : purchase_price = 600) (h2 : sell_price = 624) :
  ((sell_price - purchase_price) / purchase_price) * 100 = 4 := by
  sorry

end profit_percentage_l273_273080


namespace digits_in_equation_l273_273407

theorem digits_in_equation :
  ∃ (a b c d e f g h i : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
    f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
    g ≠ h ∧ g ≠ i ∧
    h ≠ i ∧
    {a, b, c, d, e, f, g, h, i} = {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    (a : ℚ) / b = (c : ℚ) / d ∧
    (a : ℚ) / b = (((e * 100) + (f * 10) + g) : ℚ) / (79 : ℚ) :=
begin
  -- given by the reference solution
  use 4, 2, 6, 3, 1, 5, 8, 7, 9,
  -- distinct digits from 1 to 9
  repeat { split },
  -- all elements are distinct
  all_goals { norm_num },
  -- set of elements is {1, 2, 3, 4, 5, 6, 7, 8, 9}
  refl,
  -- verify relationships
  all_goals { norm_num }
end


end digits_in_equation_l273_273407


namespace max_yellow_apples_max_total_apples_l273_273978

-- Definitions for the conditions
def num_green_apples : Nat := 10
def num_yellow_apples : Nat := 13
def num_red_apples : Nat := 18

-- Predicate for the stopping condition
def stop_condition (green yellow red : Nat) : Prop :=
  green < yellow ∧ yellow < red

-- Proof problem for maximum number of yellow apples
theorem max_yellow_apples (green yellow red : Nat) :
  num_green_apples = 10 →
  num_yellow_apples = 13 →
  num_red_apples = 18 →
  (∀ g y r, stop_condition g y r → y ≤ 13) →
  yellow ≤ 13 :=
sorry

-- Proof problem for maximum total number of apples
theorem max_total_apples (green yellow red : Nat) :
  num_green_apples = 10 →
  num_yellow_apples = 13 →
  num_red_apples = 18 →
  (∀ g y r, stop_condition g y r → g + y + r ≤ 39) →
  green + yellow + red ≤ 39 :=
sorry

end max_yellow_apples_max_total_apples_l273_273978


namespace sequence_a_1998_value_l273_273590

theorem sequence_a_1998_value :
  (∃ (a : ℕ → ℕ),
    (∀ n : ℕ, 0 <= a n) ∧
    (∀ n m : ℕ, n < m → a n < a m) ∧
    (∀ k : ℕ, ∃ i j t : ℕ, k = a i + 2 * a j + 4 * a t) ∧
    a 1998 = 1227096648) := sorry

end sequence_a_1998_value_l273_273590


namespace number_of_cars_l273_273464

theorem number_of_cars (people_per_car : ℝ) (total_people : ℝ) (h1 : people_per_car = 63.0) (h2 : total_people = 189) : total_people / people_per_car = 3 := by
  sorry

end number_of_cars_l273_273464


namespace ternary_to_decimal_l273_273110

theorem ternary_to_decimal (n : ℕ) (h : n = 121) : 
  (1 * 3^2 + 2 * 3^1 + 1 * 3^0) = 16 :=
by sorry

end ternary_to_decimal_l273_273110


namespace fraction_sum_is_11_l273_273018

theorem fraction_sum_is_11 (a b : ℕ) (h1 : 0.375 = (a : ℚ) / b) (h2 : Nat.coprime a b) : a + b = 11 := 
by sorry

end fraction_sum_is_11_l273_273018


namespace algebraic_expression_value_l273_273300

variables (x y : ℝ)

theorem algebraic_expression_value :
  x^2 - 4 * x - 1 = 0 →
  (2 * x - 3)^2 - (x + y) * (x - y) - y^2 = 12 :=
by
  intro h
  sorry

end algebraic_expression_value_l273_273300


namespace workshop_total_workers_l273_273348

noncomputable def total_workers (total_avg_salary technicians_avg_salary non_technicians_avg_salary : ℕ) (technicians_count : ℕ) : ℕ :=
  sorry

theorem workshop_total_workers (avg_salary : ℕ) (tech_avg_salary : ℕ) (non_tech_avg_salary : ℕ) (tech_count : ℕ) :
  total_workers avg_salary tech_avg_salary non_tech_avg_salary tech_count = 49 :=
by {
  -- Given conditions
  let avg_salary := 8000,
  let tech_avg_salary := 20000,
  let non_tech_avg_salary := 6000,
  let tech_count := 7,
  -- Assertions based on these conditions would follow
  sorry
}

end workshop_total_workers_l273_273348


namespace kaleb_saved_initial_amount_l273_273986

theorem kaleb_saved_initial_amount (allowance toys toy_price : ℕ) (total_savings : ℕ)
  (h1 : allowance = 15)
  (h2 : toys = 6)
  (h3 : toy_price = 6)
  (h4 : total_savings = toys * toy_price - allowance) :
  total_savings = 21 :=
  sorry

end kaleb_saved_initial_amount_l273_273986


namespace evaluate_expression_l273_273786

theorem evaluate_expression (x : ℝ) (h : x = 2) : x^2 - 3*x + 2 = 0 :=
by
  rw [h]
  norm_num
  sorry

end evaluate_expression_l273_273786


namespace quadratic_eq_has_two_distinct_real_roots_l273_273230

theorem quadratic_eq_has_two_distinct_real_roots (m : ℝ) : 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (∀ x : ℝ, x^2 - 2*m*x - m - 1 = 0 ↔ x = x1 ∨ x = x2) :=
by
  sorry

end quadratic_eq_has_two_distinct_real_roots_l273_273230


namespace fraction_evaluation_l273_273079

theorem fraction_evaluation :
  (11 - 10 + 9 - 8 + 7 - 6 + 5 - 4 + 3 - 2) / (0 - 1 + 2 - 3 + 4 - 5 + 6 - 7 + 8) = 5 / 4 :=
by
  sorry

end fraction_evaluation_l273_273079


namespace n_squared_plus_n_divisible_by_2_l273_273461

theorem n_squared_plus_n_divisible_by_2 (n : ℤ) : 2 ∣ (n^2 + n) :=
sorry

end n_squared_plus_n_divisible_by_2_l273_273461


namespace trailing_zeros_300_factorial_l273_273873

-- Definition to count the number of times a prime factor divides the factorial of n
def count_factors (n : ℕ) (p : ℕ) : ℕ :=
  Nat.div (n / p) 1 + Nat.div (n / p^2) 1 + Nat.div (n / p^3) 1 + Nat.div (n / p^4) 1

-- Theorem stating the number of trailing zeros in 300! is 74
theorem trailing_zeros_300_factorial : count_factors 300 5 = 74 := by
  sorry

end trailing_zeros_300_factorial_l273_273873


namespace inequality_example_l273_273193

variable (a b : ℝ)

theorem inequality_example (h1 : a > 1/2) (h2 : b > 1/2) : a + 2 * b - 5 * a * b < 1/4 :=
by
  sorry

end inequality_example_l273_273193


namespace ronald_next_roll_l273_273715

/-- Ronald's rolls -/
def rolls : List ℕ := [1, 3, 2, 4, 3, 5, 3, 4, 4, 2]

/-- Total number of rolls after the next roll -/
def total_rolls := rolls.length + 1

/-- The desired average of the rolls -/
def desired_average : ℕ := 3

/-- The sum Ronald needs to reach after the next roll to achieve the desired average -/
def required_sum : ℕ := desired_average * total_rolls

/-- Ronald's current sum of rolls -/
def current_sum : ℕ := List.sum rolls

/-- The next roll needed to achieve the desired average -/
def next_roll_needed : ℕ := required_sum - current_sum

theorem ronald_next_roll :
  next_roll_needed = 2 := by
  sorry

end ronald_next_roll_l273_273715


namespace work_completion_time_equal_l273_273500

/-- Define the individual work rates of a, b, c, and d --/
def work_rate_a : ℚ := 1 / 24
def work_rate_b : ℚ := 1 / 6
def work_rate_c : ℚ := 1 / 12
def work_rate_d : ℚ := 1 / 10

/-- Define the combined work rate when they work together --/
def combined_work_rate : ℚ := work_rate_a + work_rate_b + work_rate_c + work_rate_d

/-- Define total work as one unit divided by the combined work rate --/
def total_days_to_complete : ℚ := 1 / combined_work_rate

/-- Main theorem to prove: When a, b, c, and d work together, they complete the work in 120/47 days --/
theorem work_completion_time_equal : total_days_to_complete = 120 / 47 :=
by
  sorry

end work_completion_time_equal_l273_273500


namespace probability_of_events_l273_273051

def tetrahedron_outcomes : Set (ℕ × ℕ) :=
  { (i, j) | i ∈ {1, 2, 3, 4} ∧ j ∈ {1, 2, 3, 4} }

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1

def cond_A (outcome : ℕ × ℕ) : Prop := is_even outcome.1
def cond_B (outcome : ℕ × ℕ) : Prop := is_odd outcome.2
def cond_C (outcome : ℕ × ℕ) : Prop := (is_odd outcome.1 ∧ is_odd outcome.2) ∨ (is_even outcome.1 ∧ is_even outcome.2)

def event_A : Set (ℕ × ℕ) := {x ∈ tetrahedron_outcomes | cond_A x}
def event_B : Set (ℕ × ℕ) := {x ∈ tetrahedron_outcomes | cond_B x}
def event_C : Set (ℕ × ℕ) := {x ∈ tetrahedron_outcomes | cond_C x}
def event_AB : Set (ℕ × ℕ) := event_A ∩ event_B
def event_AC : Set (ℕ × ℕ) := event_A ∩ event_C
def event_BC : Set (ℕ × ℕ) := event_B ∩ event_C
def event_ABC : Set (ℕ × ℕ) := event_A ∩ event_B ∩ event_C

theorem probability_of_events :
  P(event_A) = 1 / 2 ∧
  P(event_B) = 1 / 2 ∧
  P(event_C) = 1 / 2 ∧
  P(event_AB) = 1 / 4 ∧
  P(event_AC) = 1 / 4 ∧
  P(event_BC) = 1 / 4 ∧
  P(event_ABC) = 0 ∧
  ¬ disjoint event_A event_B := by
  sorry

end probability_of_events_l273_273051


namespace find_incorrect_statements_l273_273600

-- Definitions of the statements based on their mathematical meanings
def is_regular_tetrahedron (shape : Type) : Prop := 
  -- assume some definition for regular tetrahedron
  sorry 

def is_cube (shape : Type) : Prop :=
  -- assume some definition for cube
  sorry

def is_generatrix_parallel (cylinder : Type) : Prop :=
  -- assume definition stating that generatrix of a cylinder is parallel to its axis
  sorry

def is_lateral_faces_isosceles (pyramid : Type) : Prop :=
  -- assume definition that in a regular pyramid, lateral faces are congruent isosceles triangles
  sorry

def forms_cone_on_rotation (triangle : Type) (axis : Type) : Prop :=
  -- assume definition that a right triangle forms a cone when rotated around one of its legs (other than hypotenuse)
  sorry

-- Given conditions as definitions
def statement_A : Prop := ∀ (shape : Type), is_regular_tetrahedron shape → is_cube shape = false
def statement_B : Prop := ∀ (cylinder : Type), is_generatrix_parallel cylinder = true
def statement_C : Prop := ∀ (pyramid : Type), is_lateral_faces_isosceles pyramid = true
def statement_D : Prop := ∀ (triangle : Type) (axis : Type), forms_cone_on_rotation triangle axis = false

-- The proof problem equivalent to incorrectness of statements A, B, and D
theorem find_incorrect_statements : 
  (statement_A = true) ∧ -- statement A is indeed incorrect
  (statement_B = true) ∧ -- statement B is indeed incorrect
  (statement_C = false) ∧ -- statement C is correct
  (statement_D = true)    -- statement D is indeed incorrect
:= 
sorry

end find_incorrect_statements_l273_273600


namespace lcm_1404_972_l273_273883

def num1 := 1404
def num2 := 972

theorem lcm_1404_972 : Nat.lcm num1 num2 = 88452 := 
by 
  sorry

end lcm_1404_972_l273_273883


namespace remainder_3203_4507_9929_mod_75_l273_273599

theorem remainder_3203_4507_9929_mod_75 :
  (3203 * 4507 * 9929) % 75 = 34 :=
by
  have h1 : 3203 % 75 = 53 := sorry
  have h2 : 4507 % 75 = 32 := sorry
  have h3 : 9929 % 75 = 29 := sorry
  -- complete the proof using modular arithmetic rules.
  sorry

end remainder_3203_4507_9929_mod_75_l273_273599


namespace x_intercept_of_perpendicular_line_is_16_over_3_l273_273369

theorem x_intercept_of_perpendicular_line_is_16_over_3 :
  (∃ x : ℚ, (∃ y : ℚ, (4 * x - 3 * y = 12))
    ∧ (∃ x y : ℚ, (y = - (3 / 4) * x + 4 ∧ y = 0) ∧ x = 16 / 3)) :=
by {
  sorry
}

end x_intercept_of_perpendicular_line_is_16_over_3_l273_273369


namespace nine_pow_n_sub_one_l273_273956

theorem nine_pow_n_sub_one (n : ℕ) (h1 : n % 2 = 1) (h2 : ∃ (p1 p2 p3 : ℕ), Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ (9^n - 1) = p1 * p2 * p3 ∧ (p1 = 61 ∨ p2 = 61 ∨ p3 = 61)) : 9^n - 1 = 59048 := 
sorry

end nine_pow_n_sub_one_l273_273956


namespace three_pow_1000_mod_seven_l273_273239

theorem three_pow_1000_mod_seven : (3 ^ 1000) % 7 = 4 := 
by 
  -- proof omitted
  sorry

end three_pow_1000_mod_seven_l273_273239


namespace factor_x4_plus_64_l273_273469

theorem factor_x4_plus_64 (x : ℝ) : 
  (x^4 + 64) = (x^2 - 4 * x + 8) * (x^2 + 4 * x + 8) :=
sorry

end factor_x4_plus_64_l273_273469


namespace Eddy_travel_time_l273_273937

theorem Eddy_travel_time (T V_e V_f : ℝ) 
  (dist_AB dist_AC : ℝ) 
  (time_Freddy : ℝ) 
  (speed_ratio : ℝ) 
  (h1 : dist_AB = 600) 
  (h2 : dist_AC = 300) 
  (h3 : time_Freddy = 3) 
  (h4 : speed_ratio = 2)
  (h5 : V_f = dist_AC / time_Freddy)
  (h6 : V_e = speed_ratio * V_f)
  (h7 : T = dist_AB / V_e) :
  T = 3 :=
by
  sorry

end Eddy_travel_time_l273_273937


namespace nine_pow_n_sub_one_l273_273955

theorem nine_pow_n_sub_one (n : ℕ) (h1 : n % 2 = 1) (h2 : ∃ (p1 p2 p3 : ℕ), Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ (9^n - 1) = p1 * p2 * p3 ∧ (p1 = 61 ∨ p2 = 61 ∨ p3 = 61)) : 9^n - 1 = 59048 := 
sorry

end nine_pow_n_sub_one_l273_273955


namespace find_a_l273_273878

theorem find_a (a x1 x2 : ℝ) (h1 : a > 0) (h2 : x^2 - 2*a*x - 8*(a^2) < 0) (h3 : x2 - x1 = 15) : a = 5 / 2 :=
by
  -- Sorry is used to skip the actual proof.
  sorry

end find_a_l273_273878


namespace number_of_x_for_P_eq_zero_l273_273413

noncomputable def P (x : ℝ) : ℂ :=
  1 + Complex.exp (Complex.I * x) - Complex.exp (2 * Complex.I * x) + Complex.exp (3 * Complex.I * x) - Complex.exp (4 * Complex.I * x)

theorem number_of_x_for_P_eq_zero : 
  ∃ (n : ℕ), n = 4 ∧ ∃ (xs : Fin n → ℝ), (∀ i, 0 ≤ xs i ∧ xs i < 2 * Real.pi ∧ P (xs i) = 0) ∧ Function.Injective xs := 
sorry

end number_of_x_for_P_eq_zero_l273_273413


namespace solve_n_l273_273541

/-
Define the condition for the problem.
Given condition: \(\frac{1}{n+1} + \frac{2}{n+1} + \frac{n}{n+1} = 4\)
-/

noncomputable def condition (n : ℚ) : Prop :=
  (1 / (n + 1) + 2 / (n + 1) + n / (n + 1)) = 4

/-
The theorem to prove: Value of \( n \) that satisfies the condition is \( n = -\frac{1}{3} \)
-/
theorem solve_n : ∃ n : ℚ, condition n ∧ n = -1 / 3 :=
by
  sorry

end solve_n_l273_273541


namespace no_n_repeats_stock_price_l273_273879

-- Problem statement translation
theorem no_n_repeats_stock_price (n : ℕ) (h1 : n < 100) : ¬ ∃ k l : ℕ, (100 + n) ^ k * (100 - n) ^ l = 100 ^ (k + l) :=
by
  sorry

end no_n_repeats_stock_price_l273_273879


namespace sum_and_product_of_roots_l273_273229

theorem sum_and_product_of_roots (m n : ℝ) (h1 : (m / 3) = 9) (h2 : (n / 3) = 20) : m + n = 87 :=
by
  sorry

end sum_and_product_of_roots_l273_273229


namespace people_joined_group_l273_273904

theorem people_joined_group (x y : ℕ) (h1 : 1430 = 22 * x) (h2 : 1430 = 13 * (x + y)) : y = 45 := 
by 
  -- This is just the statement, so we add sorry to skip the proof
  sorry

end people_joined_group_l273_273904


namespace range_of_g_l273_273942

noncomputable def g (x : ℝ) : ℝ := Real.arcsin x + Real.arccos x + Real.arctan (2 * x)

theorem range_of_g : 
  Set.range g = Set.Icc ((π / 2) - (π / 3)) ((π / 2) + (π / 3)) := by
  sorry

end range_of_g_l273_273942


namespace quadratic_expression_value_l273_273702

def roots (a b c : ℤ) : set ℝ := {x | a * x^2 + b * x + c = 0}

theorem quadratic_expression_value :
  let a : ℤ := 3
  let b : ℤ := 9
  let c : ℤ := -21
  let p q : ℝ := if p ∈ roots a b c ∧ q ∈ roots a b c ∧ p ≠ q then (p, q) else (0, 0)
  (3 * p - 4) * (6 * q - 8) = 14 :=
by 
  sorry

end quadratic_expression_value_l273_273702


namespace car_distance_covered_l273_273081

def distance_covered_by_car (time : ℝ) (speed : ℝ) : ℝ :=
  speed * time

theorem car_distance_covered :
  distance_covered_by_car (3 + 1/5 : ℝ) 195 = 624 :=
by
  sorry

end car_distance_covered_l273_273081


namespace min_max_f_l273_273544

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^2 - Real.cos x

theorem min_max_f :
  (∀ x, 2 * (Real.sin (x / 2))^2 = 1 - Real.cos x) →
  (∀ x, -1 ≤ f x ∧ f x ≤ 5 / 4) :=
by 
  intros h x
  sorry

end min_max_f_l273_273544


namespace ball_total_attendance_l273_273186

theorem ball_total_attendance :
    ∃ n m : ℕ, n + m = 41 ∧
    (n + m < 50) ∧
    (3 * n / 4 = (5 * m / 7)) :=
begin
  sorry
end

end ball_total_attendance_l273_273186


namespace find_first_m_gt_1959_l273_273930

theorem find_first_m_gt_1959 :
  ∃ m n : ℕ, 8 * m - 7 = n^2 ∧ m > 1959 ∧ m = 2017 :=
by
  sorry

end find_first_m_gt_1959_l273_273930


namespace angle_BDE_equilateral_l273_273319

theorem angle_BDE_equilateral
  (A B C D E : Type)
  [IsTriangle A B C]
  (angle_A : ∠ A = 65)
  (angle_C : ∠ C = 45)
  (midpoint_D : IsMidpoint D A B)
  (midpoint_E : IsMidpoint E B C)
  (equidistant : distance A D = distance D B ∧ distance B E = distance E C ∧ distance D B = distance B E)
  : ∠ BDE = 60 :=
begin
  sorry
end

end angle_BDE_equilateral_l273_273319


namespace simplify_fraction_l273_273000

theorem simplify_fraction (b : ℕ) (hb : b = 5) : (15 * b^4) / (90 * b^3 * b) = 1 / 6 := by
  sorry

end simplify_fraction_l273_273000


namespace garden_area_l273_273643

theorem garden_area 
  (property_width : ℕ)
  (property_length : ℕ)
  (garden_width_ratio : ℚ)
  (garden_length_ratio : ℚ)
  (width_ratio_eq : garden_width_ratio = (1 : ℚ) / 8)
  (length_ratio_eq : garden_length_ratio = (1 : ℚ) / 10)
  (property_width_eq : property_width = 1000)
  (property_length_eq : property_length = 2250) :
  (property_width * garden_width_ratio * property_length * garden_length_ratio = 28125) :=
  sorry

end garden_area_l273_273643


namespace basketball_lineup_count_l273_273905

theorem basketball_lineup_count :
  let total_players := 12 in
  let forwards := 6 in
  let guards := 4 in
  let players_a_b_play_both := true in
  let lineup_forwards := 3 in
  let lineup_guards := 2 in
  ∃ total_lineups,
    (let c6_k (k : ℕ) := Nat.choose 6 k in
     let c4_2 := Nat.choose 4 2 in
     let a_b_forward := 2 in
     let a_b_as_forward := 
       (c6_k 3 * (Nat.choose (6 + 2) 2)) +  -- Neither A nor B as forward
       ((c6_k 2) * a_b_forward * (Nat.choose (5 + 1) 2)) +  -- One of A or B as forward
       (Net 6 1 * 1 * c4_2)  -- Both A and B as forward
     in
     a_b_as_forward) = total_lineups
    in total_lineups = 636 :=
by
  sorry

end basketball_lineup_count_l273_273905


namespace polygon_sides_l273_273090

theorem polygon_sides (n : ℕ) (h1 : (n - 2) * 180 - 180 = 2190) : n = 15 :=
sorry

end polygon_sides_l273_273090


namespace largest_three_digit_sum_fifteen_l273_273489

theorem largest_three_digit_sum_fifteen : ∃ (a b c : ℕ), (a = 9 ∧ b = 6 ∧ c = 0 ∧ 100 * a + 10 * b + c = 960 ∧ a + b + c = 15 ∧ a < 10 ∧ b < 10 ∧ c < 10) := by
  sorry

end largest_three_digit_sum_fifteen_l273_273489


namespace slope_of_line_of_intersections_l273_273944

theorem slope_of_line_of_intersections : 
  ∀ s : ℝ, let x := (41 * s + 13) / 11
           let y := -((2 * s + 6) / 11)
           ∃ m : ℝ, m = -22 / 451 :=
sorry

end slope_of_line_of_intersections_l273_273944


namespace divisibility_of_n_pow_n_minus_1_l273_273851

theorem divisibility_of_n_pow_n_minus_1 (n : ℕ) (h : n > 1): (n^ (n - 1) - 1) % (n - 1)^2 = 0 := 
  sorry

end divisibility_of_n_pow_n_minus_1_l273_273851


namespace avg_adjacent_boy_girl_pairs_l273_273465

theorem avg_adjacent_boy_girl_pairs :
  let boys := 6
  let girls := 14
  let total_people := boys + girls
  let total_pairs := total_people - 1
  let prob_boy_girl := (boys / total_people) * (girls / (total_people - 1))
  let total_prob := 2 * prob_boy_girl
  let expected_pairs := total_pairs * total_prob
  ⌊expected_pairs⌉ = 8
  := by
  let boys := 6
  let girls := 14
  let total_people := boys + girls
  let total_pairs := total_people - 1
  let prob_boy_girl := (boys / total_people.toReal) * (girls / (total_people - 1).toReal)
  let total_prob := 2 * prob_boy_girl
  let expected_pairs := total_pairs.toReal * total_prob
  have h := Real.floor_eq 8 expected_pairs
  sorry

end avg_adjacent_boy_girl_pairs_l273_273465


namespace prism_sides_plus_two_l273_273737

theorem prism_sides_plus_two (E V S : ℕ) (h1 : E + V = 30) (h2 : E = 3 * S) (h3 : V = 2 * S) : S + 2 = 8 :=
by
  sorry

end prism_sides_plus_two_l273_273737


namespace count_multiples_of_4_in_range_l273_273970

-- Define the predicate for multiples of 4
def is_multiple_of_4 (n : ℕ) : Prop := n % 4 = 0

-- Define the range predicate
def in_range (n : ℕ) (a b : ℕ) : Prop := a ≤ n ∧ n ≤ b

-- Formulate the main theorem
theorem count_multiples_of_4_in_range (a b : ℕ) (a := 50) (b := 300) : 
  (∑ i in (Finset.filter (λ n, is_multiple_of_4 n ∧ in_range n a b) (Finset.range (b + 1))), 1) = 63 := 
by
  sorry

end count_multiples_of_4_in_range_l273_273970


namespace gallons_of_soup_l273_273280

def bowls_per_minute : ℕ := 5
def ounces_per_bowl : ℕ := 10
def serving_time_minutes : ℕ := 15
def ounces_per_gallon : ℕ := 128

theorem gallons_of_soup :
  (5 * 10 * 15 / 128) = 6 :=
by
  sorry

end gallons_of_soup_l273_273280


namespace solve_for_x_l273_273408

theorem solve_for_x :
  { x : Real | ⌊ 2 * x * ⌊ x ⌋ ⌋ = 58 } = {x : Real | 5.8 ≤ x ∧ x < 5.9} :=
sorry

end solve_for_x_l273_273408


namespace max_value_k_l273_273731

noncomputable def sqrt_minus (x : ℝ) : ℝ := Real.sqrt (x - 3)
noncomputable def sqrt_six_minus (x : ℝ) : ℝ := Real.sqrt (6 - x)

theorem max_value_k (k : ℝ) : (∃ x : ℝ, 3 ≤ x ∧ x ≤ 6 ∧ sqrt_minus x + sqrt_six_minus x ≥ k) ↔ k ≤ Real.sqrt 12 := by
  sorry

end max_value_k_l273_273731


namespace sum_of_sequence_l273_273310

noncomputable def sequence_sum (a : ℝ) (n : ℕ) : ℝ :=
if a = 1 then sorry else (5 * (1 - a ^ n) / (1 - a) ^ 2) - (4 + (5 * n - 4) * a ^ n) / (1 - a)

theorem sum_of_sequence (S : ℕ → ℝ) (a : ℝ) (h1 : S 1 = 1)
                       (h2 : ∀ n, S (n + 1) - S n = (5 * n + 1) * a ^ n) (h3 : |a| ≠ 1) :
  ∀ n, S n = sequence_sum a n :=
  sorry

end sum_of_sequence_l273_273310


namespace maximize_expression_l273_273941

theorem maximize_expression (x y : ℝ) :
  (2 * x + 3 * y + 4) / Real.sqrt (x^2 + y^2 + 4) ≤ Real.sqrt 29 :=
by
  sorry

end maximize_expression_l273_273941


namespace greatest_x_value_l273_273584

noncomputable def greatest_possible_value (x : ℕ) : ℕ :=
  if (x % 5 = 0) ∧ (x^3 < 3375) then x else 0

theorem greatest_x_value :
  ∃ x, greatest_possible_value x = 10 ∧ (∀ y, ((y % 5 = 0) ∧ (y^3 < 3375)) → y ≤ x) :=
by
  sorry

end greatest_x_value_l273_273584


namespace area_of_triangle_ABC_l273_273008

-- Given conditions
def circle_radius : ℝ := 4
def BD : ℝ := 5
def ED : ℝ := 6
def perpendicular_AD_ED : Prop := ED ⊥ ((2 * circle_radius + BD) : ℝ)

-- Definitions based on conditions
def AD : ℝ := (2 * circle_radius) + BD
def EA : ℝ := Real.sqrt (AD ^ 2 + ED ^ 2)

-- Correct Answer (to be proved)
theorem area_of_triangle_ABC : 
  let BC := Real.sqrt ((((2 * circle_radius)):ℝ) ^ 2 - (EA - 65 / EA) ^ 2) / Real.sqrt 205,
      AC := (EA - 65 / EA) / Real.sqrt 205
  in  1/2 * BC * AC = 140 * Real.sqrt 2360 / 205 :=
sorry

end area_of_triangle_ABC_l273_273008


namespace smallest_of_five_consecutive_even_sum_500_l273_273466

theorem smallest_of_five_consecutive_even_sum_500 : 
  ∃ (n : Int), (n - 4, n - 2, n, n + 2, n + 4).1 = 96 ∧ 
  ((n - 4) + (n - 2) + n + (n + 2) + (n + 4) = 500) :=
by
  sorry

end smallest_of_five_consecutive_even_sum_500_l273_273466


namespace problem_goal_l273_273158

-- Define the problem stating that there is a graph of points (x, y) satisfying the condition
def area_of_graph_satisfying_condition : Real :=
  let A := 2013
  -- Define the pairs (a, b) which are multiples of 2013
  let pairs := [(1, 2013), (3, 671), (11, 183), (33, 61)]
  -- Calculate the area of each region formed by pairs
  let area := pairs.length * 4
  area

-- Problem goal statement proving the area is equal to 16
theorem problem_goal : area_of_graph_satisfying_condition = 16 := by
  sorry

end problem_goal_l273_273158


namespace coefficient_x4_in_expansion_sum_l273_273468

theorem coefficient_x4_in_expansion_sum :
  (Nat.choose 5 4 + Nat.choose 6 4 + Nat.choose 7 4 = 55) :=
by
  sorry

end coefficient_x4_in_expansion_sum_l273_273468


namespace find_angle_y_l273_273980

theorem find_angle_y (ABC BAC BCA DCE CED y : ℝ)
  (h1 : ABC = 80) (h2 : BAC = 60)
  (h3 : ABC + BAC + BCA = 180)
  (h4 : CED = 90)
  (h5 : DCE = BCA)
  (h6 : DCE + CED + y = 180) :
  y = 50 :=
by
  sorry

end find_angle_y_l273_273980


namespace permutations_of_three_digit_numbers_from_set_l273_273022

theorem permutations_of_three_digit_numbers_from_set {digits : Finset ℕ} (h : digits = {1, 2, 3, 4, 5}) :
  ∃ n : ℕ, n = (Finset.card digits) * (Finset.card digits - 1) * (Finset.card digits - 2) ∧ n = 60 :=
by
  sorry

end permutations_of_three_digit_numbers_from_set_l273_273022


namespace suitable_sampling_method_l273_273247

theorem suitable_sampling_method 
  (seniorTeachers : ℕ)
  (intermediateTeachers : ℕ)
  (juniorTeachers : ℕ)
  (totalSample : ℕ)
  (totalTeachers : ℕ)
  (prob : ℚ)
  (seniorSample : ℕ)
  (intermediateSample : ℕ)
  (juniorSample : ℕ)
  (excludeOneSenior : ℕ) :
  seniorTeachers = 28 →
  intermediateTeachers = 54 →
  juniorTeachers = 81 →
  totalSample = 36 →
  excludeOneSenior = 27 →
  totalTeachers = excludeOneSenior + intermediateTeachers + juniorTeachers →
  prob = totalSample / totalTeachers →
  seniorSample = excludeOneSenior * prob →
  intermediateSample = intermediateTeachers * prob →
  juniorSample = juniorTeachers * prob →
  seniorSample + intermediateSample + juniorSample = totalSample :=
by
  intros hsenior hins hjunior htotal hexclude htotalTeachers hprob hseniorSample hintermediateSample hjuniorSample
  sorry

end suitable_sampling_method_l273_273247


namespace total_roses_planted_three_days_l273_273358

-- Definitions based on conditions
def susan_roses_two_days_ago : ℕ := 10
def maria_roses_two_days_ago : ℕ := 2 * susan_roses_two_days_ago
def john_roses_two_days_ago : ℕ := susan_roses_two_days_ago + 10
def roses_two_days_ago : ℕ := susan_roses_two_days_ago + maria_roses_two_days_ago + john_roses_two_days_ago

def roses_yesterday : ℕ := roses_two_days_ago + 20
def susan_roses_yesterday : ℕ := susan_roses_two_days_ago * roses_yesterday / roses_two_days_ago
def maria_roses_yesterday : ℕ := maria_roses_two_days_ago * roses_yesterday / roses_two_days_ago
def john_roses_yesterday : ℕ := john_roses_two_days_ago * roses_yesterday / roses_two_days_ago

def roses_today : ℕ := 2 * roses_two_days_ago
def susan_roses_today : ℕ := susan_roses_two_days_ago
def maria_roses_today : ℕ := maria_roses_two_days_ago + (maria_roses_two_days_ago * 25 / 100)
def john_roses_today : ℕ := john_roses_two_days_ago - (john_roses_two_days_ago * 10 / 100)

def total_roses_planted : ℕ := 
  (susan_roses_two_days_ago + maria_roses_two_days_ago + john_roses_two_days_ago) +
  (susan_roses_yesterday + maria_roses_yesterday + john_roses_yesterday) +
  (susan_roses_today + maria_roses_today + john_roses_today)

-- The statement that needs to be proved
theorem total_roses_planted_three_days : total_roses_planted = 173 := by 
  sorry

end total_roses_planted_three_days_l273_273358


namespace heather_walked_distance_l273_273748

theorem heather_walked_distance {H S : ℝ} (hH : H = 5) (hS : S = H + 1) (total_distance : ℝ) (time_delay_stacy : ℝ) (time_heather_meet : ℝ) :
  (total_distance = 30) → (time_delay_stacy = 0.4) → (time_heather_meet = (total_distance - S * time_delay_stacy) / (H + S)) →
  (H * time_heather_meet = 12.55) :=
by
  sorry

end heather_walked_distance_l273_273748


namespace calculate_width_of_vessel_base_l273_273760

noncomputable def cube_edge : ℝ := 17
noncomputable def base_length : ℝ := 20
noncomputable def water_rise : ℝ := 16.376666666666665
noncomputable def cube_volume : ℝ := cube_edge ^ 3
noncomputable def base_area (W : ℝ) : ℝ := base_length * W
noncomputable def displaced_volume (W : ℝ) : ℝ := base_area W * water_rise

theorem calculate_width_of_vessel_base :
  ∃ W : ℝ, displaced_volume W = cube_volume ∧ W = 15 := by
  sorry

end calculate_width_of_vessel_base_l273_273760


namespace robbers_divide_and_choose_l273_273144

/-- A model of dividing loot between two robbers who do not trust each other -/
def divide_and_choose (P1 P2 : ℕ) (A : P1 = P2) : Prop :=
  ∀ (B : ℕ → ℕ), B (max P1 P2) ≥ B P1 ∧ B (max P1 P2) ≥ B P2

theorem robbers_divide_and_choose (P1 P2 : ℕ) (A : P1 = P2) :
  divide_and_choose P1 P2 A :=
sorry

end robbers_divide_and_choose_l273_273144


namespace class1_draws_two_multiple_choice_questions_expected_value_of_X_l273_273382

-- Problem 1 in Lean 4 statement
theorem class1_draws_two_multiple_choice_questions (pB1 pB2 pB3 : ℚ)
    (pA_given_B1 pA_given_B2 pA_given_B3 pA : ℚ)
    (h1 : pB1 = 5 / 14) (h2 : pB2 = 15 / 28) (h3 : pB3 = 3 / 28)
    (h4 : pA_given_B1 = 6 / 9) (h5 : pA_given_B2 = 5 / 9) (h6 : pB3 = 4 / 9)
    (h7 : pA = (pB1 * 6 / 9) + (pB2 * 5 / 9) + (pB3 * 4 / 9)) :
  (pB1 * pA_given_B1 / pA) = 20 / 49 := by
  sorry

-- Problem 2 in Lean 4 statement
theorem expected_value_of_X (p3 p4 p5 eX : ℚ)
    (h1 : p3 = 4 / 25) (h2 : p4 = 48 / 125) (h3 : p5 = 57 / 125)
    (h4 : eX = 3 * p3 + 4 * p4 + 5 * p5) :
  eX = 537 / 125 := by
  sorry

end class1_draws_two_multiple_choice_questions_expected_value_of_X_l273_273382


namespace ratio_length_to_width_l273_273474

theorem ratio_length_to_width
  (w l : ℕ)
  (pond_length : ℕ)
  (field_length : ℕ)
  (pond_area : ℕ)
  (field_area : ℕ)
  (pond_to_field_area_ratio : ℚ)
  (field_length_given : field_length = 28)
  (pond_length_given : pond_length = 7)
  (pond_area_def : pond_area = pond_length * pond_length)
  (pond_to_field_area_ratio_def : pond_to_field_area_ratio = 1 / 8)
  (field_area_def : field_area = pond_area * 8)
  (field_area_calc : field_area = field_length * w) :
  (field_length / w) = 2 :=
by
  sorry

end ratio_length_to_width_l273_273474


namespace fraction_sum_is_11_l273_273019

theorem fraction_sum_is_11 (a b : ℕ) (h1 : 0.375 = (a : ℚ) / b) (h2 : Nat.coprime a b) : a + b = 11 := 
by sorry

end fraction_sum_is_11_l273_273019


namespace addition_value_l273_273618

def certain_number : ℝ := 5.46 - 3.97

theorem addition_value : 5.46 + certain_number = 6.95 := 
  by 
    -- The proof would go here, but is replaced with sorry.
    sorry

end addition_value_l273_273618


namespace elder_three_times_younger_l273_273721

-- Definitions based on conditions
def age_difference := 16
def elder_present_age := 30
def younger_present_age := elder_present_age - age_difference

-- The problem statement to prove the correct value of n (years ago)
theorem elder_three_times_younger (n : ℕ) 
  (h1 : elder_present_age = younger_present_age + age_difference)
  (h2 : elder_present_age - n = 3 * (younger_present_age - n)) : 
  n = 6 := 
sorry

end elder_three_times_younger_l273_273721


namespace games_played_l273_273577

def total_points : ℝ := 120.0
def points_per_game : ℝ := 12.0
def num_games : ℝ := 10.0

theorem games_played : (total_points / points_per_game) = num_games := 
by 
  sorry

end games_played_l273_273577


namespace Harriet_age_now_l273_273827

variable (P H: ℕ)

theorem Harriet_age_now (P : ℕ) (H : ℕ) (h1 : P + 4 = 2 * (H + 4)) (h2 : P = 60 / 2) : H = 13 := by
  sorry

end Harriet_age_now_l273_273827


namespace pow_1986_mod_7_l273_273490

theorem pow_1986_mod_7 : (5 ^ 1986) % 7 = 1 := by
  sorry

end pow_1986_mod_7_l273_273490


namespace Ashutosh_completion_time_l273_273345

def Suresh_work_rate := 1 / 15
def Ashutosh_work_rate := 1 / 25
def Suresh_work_time := 9

def job_completed_by_Suresh_in_9_hours := Suresh_work_rate * Suresh_work_time
def remaining_job := 1 - job_completed_by_Suresh_in_9_hours

theorem Ashutosh_completion_time : 
  Ashutosh_work_rate * t = remaining_job -> t = 10 :=
by
  sorry

end Ashutosh_completion_time_l273_273345


namespace math_problem_l273_273056

open Real

lemma radical_product :
  (3:ℝ) * (3:ℝ) * (2:ℝ) = 18 :=
by sorry

lemma cube_root_27 :
  real.cbrt 27 = 3 :=
by sorry

lemma fourth_root_81 :
  81 ^ (1 / 4:ℝ) = 3 :=
by sorry

lemma sixth_root_64 :
  64 ^ (1 / 6:ℝ) = 2 :=
by sorry

theorem math_problem :
  real.cbrt 27 * 81 ^ (1 / 4:ℝ) * 64 ^ (1 / 6:ℝ) = 18 :=
begin
  rw [cube_root_27, fourth_root_81, sixth_root_64],
  exact radical_product,
end

end math_problem_l273_273056


namespace distance_diff_is_0_point3_l273_273330

def john_walk_distance : ℝ := 0.7
def nina_walk_distance : ℝ := 0.4
def distance_difference_john_nina : ℝ := john_walk_distance - nina_walk_distance

theorem distance_diff_is_0_point3 : distance_difference_john_nina = 0.3 :=
by
  -- proof goes here
  sorry

end distance_diff_is_0_point3_l273_273330


namespace jack_age_difference_l273_273526

def beckett_age : ℕ := 12
def olaf_age : ℕ := beckett_age + 3
def shannen_age : ℕ := olaf_age - 2
def total_age : ℕ := 71
def jack_age : ℕ := total_age - (beckett_age + olaf_age + shannen_age)
def difference := jack_age - 2 * shannen_age

theorem jack_age_difference :
  difference = 5 :=
by
  -- Math proof goes here
  sorry

end jack_age_difference_l273_273526


namespace correct_choice_C_l273_273439

def geometric_sequence (n : ℕ) : ℕ := 
  2^(n - 1)

def sum_geometric_sequence (n : ℕ) : ℕ := 
  2^n - 1

theorem correct_choice_C (n : ℕ) (h : 0 < n) : sum_geometric_sequence n < geometric_sequence (n + 1) := by
  sorry

end correct_choice_C_l273_273439


namespace fraction_meaningful_iff_l273_273682

theorem fraction_meaningful_iff (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 1)) ↔ x ≠ 1 := sorry

end fraction_meaningful_iff_l273_273682


namespace sum_of_three_numbers_l273_273373

theorem sum_of_three_numbers (a b c : ℝ) :
  a + b = 35 → b + c = 47 → c + a = 58 → a + b + c = 70 :=
by
  intros h1 h2 h3
  sorry

end sum_of_three_numbers_l273_273373


namespace problem_statement_l273_273432

-- Defining the propositions p and q as Boolean variables
variables (p q : Prop)

-- Assume the given conditions
theorem problem_statement (hnp : ¬¬p) (hnpq : ¬(p ∧ q)) : p ∧ ¬q :=
by {
  -- Derived steps to satisfy the conditions are implicit within this scope
  sorry
}

end problem_statement_l273_273432


namespace original_profit_margin_l273_273687

theorem original_profit_margin (x : ℝ) (h1 : x - 0.9 / 0.9 = 12 / 100) : (x - 1) / 1 * 100 = 8 :=
by
  sorry

end original_profit_margin_l273_273687


namespace total_attended_ball_lt_fifty_l273_273181

-- Conditions from part a)
variable (n : ℕ) -- number of ladies
variable (m : ℕ) -- number of gentlemen

def total_people (n m : ℕ) : ℕ := n + m

-- Quarter of the ladies not invited means three-quarters danced
def ladies_danced (n : ℕ) : ℕ := 3 * n / 4

-- Two-sevenths of the gentlemen did not invite anyone
def gents_invited (m : ℕ) : ℕ := 5 * m / 7

-- From the condition we have these relations
axiom ratio_relation (h : n * 3 / 4 = m * 5 / 7) : True

-- Prove the total number of people attended the ball
theorem total_attended_ball_lt_fifty 
  (h : n * 3 / 4 = m * 5 / 7) 
  (hnm_lt_50 : total_people n m < 50) : total_people n m = 41 := 
by
  sorry

end total_attended_ball_lt_fifty_l273_273181


namespace find_b_l273_273728

open Real

theorem find_b (b : ℝ) : 
  (∀ x y : ℝ, 4 * y - 3 * x - 2 = 0 -> 6 * y + b * x + 1 = 0 -> 
   exists m₁ m₂ : ℝ, 
   ((y = m₁ * x + _1 / 2) -> m₁ = 3 / 4) ∧ ((y = m₂ * x - 1 / 6) -> m₂ = -b / 6)) -> 
  b = -4.5 :=
by
  sorry

end find_b_l273_273728


namespace double_angle_second_quadrant_l273_273678

theorem double_angle_second_quadrant (α : ℝ) (h : π/2 < α ∧ α < π) : 
  ¬((0 ≤ 2*α ∧ 2*α < π/2) ∨ (3*π/2 < 2*α ∧ 2*α < 2*π)) :=
sorry

end double_angle_second_quadrant_l273_273678


namespace distance_between_vertices_l273_273191

-- Define the equations of the parabolas
def C_eq (x : ℝ) : ℝ := x^2 + 6 * x + 13
def D_eq (x : ℝ) : ℝ := -x^2 + 2 * x + 8

-- Define the vertices of the parabolas
def vertex_C : (ℝ × ℝ) := (-3, 4)
def vertex_D : (ℝ × ℝ) := (1, 9)

-- Prove that the distance between the vertices is sqrt 41
theorem distance_between_vertices : 
  dist (vertex_C) (vertex_D) = Real.sqrt 41 := 
by
  sorry

end distance_between_vertices_l273_273191


namespace polygon_sides_l273_273035

theorem polygon_sides (n : ℕ) 
  (h1 : (n - 2) * 180 = 4 * 360) : 
  n = 6 :=
by sorry

end polygon_sides_l273_273035


namespace largest_invertible_interval_l273_273462

noncomputable def g (x : ℝ) : ℝ := 3 * x^2 + 6 * x - 4

def domain_includes_neg1 (d : Set ℝ) : Prop :=
  -1 ∈ d

def g_invertible_on (d : Set ℝ) : Prop :=
  ∀ (x y : ℝ), x ∈ d → y ∈ d → g x = g y → x = y

theorem largest_invertible_interval :
  ∃ (d : Set ℝ), domain_includes_neg1 d ∧ g_invertible_on d ∧ 
  ∀ (d' : Set ℝ), domain_includes_neg1 d' → g_invertible_on d' → d' ⊆ d :=
begin
  use {x : ℝ | x ≤ -1},
  split,
  { exact set.mem_set_of_eq.2 (le_refl (-1)) },
  split,
  { assume x y hx hy hxy,
    have := congr_arg (λ x, x + 7) hxy,
    simp [g, add_assoc, add_comm, add_left_comm] at this,
    rw [←sub_eq_zero, ←sub_sub, ←sqrt_eq_zero, ←mul_eq_zero] at this,
    rcases this,
    iterate 2 {simp [*] },
    exact this},
  {  -- Prove that this is the largest interval that includes -1 where g(x) is invertible
    -- Unfortunately this statement needs the complete formal proof involving properties of parabola and complete the square
    sorry } -- skip complete proof steps
end

end largest_invertible_interval_l273_273462


namespace fraction_exponent_evaluation_l273_273893

theorem fraction_exponent_evaluation : 
  (3 ^ 10 + 3 ^ 8) / (3 ^ 10 - 3 ^ 8) = 5 / 4 :=
by sorry

end fraction_exponent_evaluation_l273_273893


namespace domain_of_function_l273_273010

theorem domain_of_function (x : ℝ) (k : ℤ) :
  ∃ x, (2 * Real.sin x + 1 ≥ 0) ↔ (- (Real.pi / 6) + 2 * k * Real.pi ≤ x ∧ x ≤ (7 * Real.pi / 6) + 2 * k * Real.pi) :=
sorry

end domain_of_function_l273_273010


namespace june_earnings_l273_273695

theorem june_earnings 
    (total_clovers : ℕ := 300)
    (pct_3_petals : ℕ := 70)
    (pct_2_petals : ℕ := 20)
    (pct_4_petals : ℕ := 8)
    (pct_5_petals : ℕ := 2)
    (earn_3_petals : ℕ := 1)
    (earn_2_petals : ℕ := 2)
    (earn_4_petals : ℕ := 5)
    (earn_5_petals : ℕ := 10)
    (earn_total : ℕ := 510) : 
  (pct_3_petals * total_clovers) / 100 * earn_3_petals + 
  (pct_2_petals * total_clovers) / 100 * earn_2_petals + 
  (pct_4_petals * total_clovers) / 100 * earn_4_petals + 
  (pct_5_petals * total_clovers) / 100 * earn_5_petals = earn_total := 
by
  -- Proof of this theorem involves calculating each part and summing them. Skipping detailed steps with sorry.
  sorry

end june_earnings_l273_273695


namespace trig_identity_l273_273421

theorem trig_identity (α : ℝ) (h0 : Real.tan α = Real.sqrt 3) (h1 : π < α) (h2 : α < 3 * π / 2) :
  Real.cos (2 * α) - Real.sin (π / 2 + α) = 0 :=
sorry

end trig_identity_l273_273421


namespace convergent_inequalities_l273_273458

theorem convergent_inequalities (α : ℝ) (P Q : ℕ → ℤ) (h_convergent : ∀ n ≥ 1, abs (α - P n / Q n) < 1 / (2 * (Q n) ^ 2) ∨ abs (α - P (n - 1) / Q (n - 1)) < 1 / (2 * (Q (n - 1))^2))
  (h_continued_fraction : ∀ n ≥ 1, P (n-1) * Q n - P n * Q (n-1) = (-1)^(n-1)) :
  ∃ p q : ℕ, 0 < q ∧ abs (α - p / q) < 1 / (2 * q^2) :=
sorry

end convergent_inequalities_l273_273458


namespace percentage_of_literate_females_is_32_5_l273_273899

noncomputable def percentage_literate_females (inhabitants : ℕ) (percent_male : ℝ) (percent_literate_males : ℝ) (percent_literate_total : ℝ) : ℝ :=
  let males := (percent_male / 100) * inhabitants
  let females := inhabitants - males
  let literate_males := (percent_literate_males / 100) * males
  let literate_total := (percent_literate_total / 100) * inhabitants
  let literate_females := literate_total - literate_males
  (literate_females / females) * 100

theorem percentage_of_literate_females_is_32_5 :
  percentage_literate_females 1000 60 20 25 = 32.5 := 
by 
  unfold percentage_literate_females
  sorry

end percentage_of_literate_females_is_32_5_l273_273899


namespace last_digit_of_1_div_3_pow_9_is_7_l273_273058

noncomputable def decimal_expansion_last_digit (n d : ℕ) : ℕ :=
  (n / d) % 10

theorem last_digit_of_1_div_3_pow_9_is_7 :
  decimal_expansion_last_digit 1 (3^9) = 7 :=
by
  sorry

end last_digit_of_1_div_3_pow_9_is_7_l273_273058


namespace carrots_not_used_l273_273759

theorem carrots_not_used :
  let total_carrots := 300
  let carrots_before_lunch := (2 / 5) * total_carrots
  let remaining_after_lunch := total_carrots - carrots_before_lunch
  let carrots_by_end_of_day := (3 / 5) * remaining_after_lunch
  remaining_after_lunch - carrots_by_end_of_day = 72
:= by
  sorry

end carrots_not_used_l273_273759


namespace exists_positive_integer_divisible_by_14_with_sqrt_between_25_and_25_3_l273_273282

theorem exists_positive_integer_divisible_by_14_with_sqrt_between_25_and_25_3 :
  ∃ (x : ℕ), x % 14 = 0 ∧ 625 <= x ∧ x <= 640 ∧ x = 630 := 
by 
  sorry

end exists_positive_integer_divisible_by_14_with_sqrt_between_25_and_25_3_l273_273282


namespace no_information_loss_chart_is_stem_and_leaf_l273_273093

theorem no_information_loss_chart_is_stem_and_leaf :
  "The correct chart with no information loss" = "Stem-and-leaf plot" :=
sorry

end no_information_loss_chart_is_stem_and_leaf_l273_273093


namespace atLeastOneTrueRange_exactlyOneTrueRange_l273_273671

-- Definitions of Proposition A and B
def propA (a : ℝ) : Prop := ∀ x, x^2 + (a - 1) * x + a^2 ≤ 0 → false
def propB (a : ℝ) : Prop := ∀ x, (2 * a^2 - a)^x < (2 * a^2 - a)^(x + 1)

-- At least one of A or B is true
def atLeastOneTrue (a : ℝ) : Prop :=
  propA a ∨ propB a

-- Exactly one of A or B is true
def exactlyOneTrue (a : ℝ) : Prop := 
  (propA a ∧ ¬ propB a) ∨ (¬ propA a ∧ propB a)

-- Theorems to prove
theorem atLeastOneTrueRange :
  ∃ a : ℝ, atLeastOneTrue a ↔ (a < -1/2 ∨ a > 1/3) := 
sorry

theorem exactlyOneTrueRange :
  ∃ a : ℝ, exactlyOneTrue a ↔ ((1/3 < a ∧ a ≤ 1) ∨ (-1 ≤ a ∧ a < -1/2)) :=
sorry

end atLeastOneTrueRange_exactlyOneTrueRange_l273_273671


namespace union_of_sets_l273_273808

-- Defining the sets A and B
def A : Set ℕ := {1, 3, 6}
def B : Set ℕ := {1, 2}

-- The theorem we want to prove
theorem union_of_sets : A ∪ B = {1, 2, 3, 6} := by
  sorry

end union_of_sets_l273_273808


namespace polygon_sides_l273_273036

theorem polygon_sides (n : ℕ) 
  (h1 : (n - 2) * 180 = 4 * 360) : 
  n = 6 :=
by sorry

end polygon_sides_l273_273036


namespace smallest_integer_solution_l273_273062

theorem smallest_integer_solution : ∃ x : ℤ, (x^2 = 3 * x + 78) ∧ x = -6 :=
by {
  sorry
}

end smallest_integer_solution_l273_273062


namespace sum_of_double_factorials_l273_273928

-- Definition for double factorial
noncomputable def double_factorial : ℕ → ℕ
| 0       => 1
| 1       => 1
| n       => n * double_factorial (n - 2)

-- Binomial coefficient definition
def binom : ℕ → ℕ → ℕ
| n, 0     => 1
| 0, k + 1 => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

-- Main theorem statement
theorem sum_of_double_factorials : 
  let S := (Finset.range 12).sum (λ i => (binom (2 * (i + 1)) (i + 1)) / 4^(i + 1)) in
  ∃ c d : ℕ, c = 10 ∧ d = 1 ∧ (S.num * 10) = c * d :=
by
  sorry

end sum_of_double_factorials_l273_273928


namespace range_of_a_l273_273113

open Real

def otimes (x y : ℝ) : ℝ := x * (1 - y)

theorem range_of_a (a : ℝ) : (∀ x : ℝ, otimes x (x + a) < 1) ↔ -1 < a ∧ a < 3 :=
by
  sorry

end range_of_a_l273_273113


namespace equation_of_tangent_line_l273_273424

theorem equation_of_tangent_line (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 - 4 * x + a * y - 17 = 0) →
   (∃ m b : ℝ, ∀ x y : ℝ, y = m * x + b ↔ 4 * x - 3 * y + 11 = 0) :=
sorry

end equation_of_tangent_line_l273_273424


namespace sin_cos_sum_eq_l273_273608

theorem sin_cos_sum_eq :
  (Real.sin (20 * Real.pi / 180) * Real.cos (10 * Real.pi / 180) +
   Real.sin (70 * Real.pi / 180) * Real.sin (10 * Real.pi / 180)) = 1 / 2 :=
by 
  sorry

end sin_cos_sum_eq_l273_273608


namespace cone_cannot_have_rectangular_cross_section_l273_273442

noncomputable def solid := Type

def is_cylinder (s : solid) : Prop := sorry
def is_cone (s : solid) : Prop := sorry
def is_rectangular_prism (s : solid) : Prop := sorry
def is_cube (s : solid) : Prop := sorry

def has_rectangular_cross_section (s : solid) : Prop := sorry

axiom cylinder_has_rectangular_cross_section (s : solid) : is_cylinder s → has_rectangular_cross_section s
axiom rectangular_prism_has_rectangular_cross_section (s : solid) : is_rectangular_prism s → has_rectangular_cross_section s
axiom cube_has_rectangular_cross_section (s : solid) : is_cube s → has_rectangular_cross_section s

theorem cone_cannot_have_rectangular_cross_section (s : solid) : is_cone s → ¬has_rectangular_cross_section s := 
sorry

end cone_cannot_have_rectangular_cross_section_l273_273442


namespace tan_angle_equiv_tan_1230_l273_273534

theorem tan_angle_equiv_tan_1230 : ∃ n : ℤ, -90 < n ∧ n < 90 ∧ Real.tan (n * Real.pi / 180) = Real.tan (1230 * Real.pi / 180) :=
sorry

end tan_angle_equiv_tan_1230_l273_273534


namespace no_six_digit_number_meets_criteria_l273_273428

def valid_digit (n : ℕ) := 2 ≤ n ∧ n ≤ 8

theorem no_six_digit_number_meets_criteria :
  ¬ ∃ (digits : Finset ℕ), digits.card = 6 ∧ (∀ x ∈ digits, valid_digit x) ∧ (digits.sum id = 42) :=
by {
  sorry
}

end no_six_digit_number_meets_criteria_l273_273428


namespace max_value_a4b3c2_l273_273840

theorem max_value_a4b3c2 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) :
  a^4 * b^3 * c^2 ≤ 1 / 6561 :=
sorry

end max_value_a4b3c2_l273_273840


namespace clock_correction_calculation_l273_273248

noncomputable def clock_correction : ℝ :=
  let daily_gain := 5/4
  let hourly_gain := daily_gain / 24
  let total_hours := (9 * 24) + 9
  let total_gain := total_hours * hourly_gain
  total_gain

theorem clock_correction_calculation : clock_correction = 11.72 := by
  sorry

end clock_correction_calculation_l273_273248


namespace min_value_expression_71_l273_273652

noncomputable def min_value_expression (x y : ℝ) : ℝ :=
  4 * x + 9 * y + 1 / (x - 4) + 1 / (y - 5)

theorem min_value_expression_71 (x y : ℝ) (hx : x > 4) (hy : y > 5) : 
  min_value_expression x y ≥ 71 :=
by
  sorry

end min_value_expression_71_l273_273652


namespace probability_of_event_B_given_A_l273_273596

-- Definition of events and probability
noncomputable def prob_event_B_given_A : ℝ :=
  let total_outcomes := 36
  let outcomes_A := 30
  let outcomes_B_given_A := 10
  outcomes_B_given_A / outcomes_A

-- Theorem statement
theorem probability_of_event_B_given_A : prob_event_B_given_A = 1 / 3 := by
  sorry

end probability_of_event_B_given_A_l273_273596


namespace volume_of_region_l273_273121

theorem volume_of_region :
    ∀ (x y z : ℝ), 
    |x - y + z| + |x - y - z| ≤ 12 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 
    → true := by
    sorry

end volume_of_region_l273_273121


namespace ternary_to_decimal_121_l273_273106

theorem ternary_to_decimal_121 : 
  let t : ℕ := 1 * 3^2 + 2 * 3^1 + 1 * 3^0 
  in t = 16 :=
by
  sorry

end ternary_to_decimal_121_l273_273106


namespace students_wearing_specific_shirt_and_accessory_count_l273_273155

theorem students_wearing_specific_shirt_and_accessory_count :
  let total_students := 1000
  let blue_shirt_percent := 0.40
  let red_shirt_percent := 0.25
  let green_shirt_percent := 0.20
  let blue_shirt_students := blue_shirt_percent * total_students
  let red_shirt_students := red_shirt_percent * total_students
  let green_shirt_students := green_shirt_percent * total_students
  let blue_shirt_stripes_percent := 0.30
  let blue_shirt_polka_dots_percent := 0.35
  let red_shirt_stripes_percent := 0.20
  let red_shirt_polka_dots_percent := 0.40
  let green_shirt_stripes_percent := 0.25
  let green_shirt_polka_dots_percent := 0.25
  let accessory_hat_percent := 0.15
  let accessory_scarf_percent := 0.10
  let red_polka_dot_students := red_shirt_polka_dots_percent * red_shirt_students
  let red_polka_dot_hat_students := accessory_hat_percent * red_polka_dot_students
  let green_no_pattern_students := green_shirt_students - (green_shirt_stripes_percent * green_shirt_students + green_shirt_polka_dots_percent * green_shirt_students)
  let green_no_pattern_scarf_students := accessory_scarf_percent * green_no_pattern_students
  red_polka_dot_hat_students + green_no_pattern_scarf_students = 25 := by
    sorry

end students_wearing_specific_shirt_and_accessory_count_l273_273155


namespace minimum_value_of_a_l273_273151

theorem minimum_value_of_a :
  (∀ x : ℝ, x > 0 → (a : ℝ) * x * Real.exp x - x - Real.log x ≥ 0) → a ≥ 1 / Real.exp 1 :=
by
  sorry

end minimum_value_of_a_l273_273151


namespace y_share_l273_273255

theorem y_share (total_amount : ℝ) (x_share y_share z_share : ℝ)
  (hx : x_share = 1) (hy : y_share = 0.45) (hz : z_share = 0.30)
  (h_total : total_amount = 105) :
  (60 * y_share) = 27 :=
by
  have h_cycle : 1 + y_share + z_share = 1.75 := by sorry
  have h_num_cycles : total_amount / 1.75 = 60 := by sorry
  sorry

end y_share_l273_273255


namespace diophantine_soln_l273_273780

-- Define the Diophantine equation as a predicate
def diophantine_eq (x y : ℤ) : Prop := x^3 - y^3 = 2 * x * y + 8

-- Theorem stating that the only solutions are (0, -2) and (2, 0)
theorem diophantine_soln :
  ∀ x y : ℤ, diophantine_eq x y ↔ (x = 0 ∧ y = -2) ∨ (x = 2 ∧ y = 0) :=
by
  sorry

end diophantine_soln_l273_273780


namespace x_intercept_of_perpendicular_line_is_16_over_3_l273_273368

theorem x_intercept_of_perpendicular_line_is_16_over_3 :
  (∃ x : ℚ, (∃ y : ℚ, (4 * x - 3 * y = 12))
    ∧ (∃ x y : ℚ, (y = - (3 / 4) * x + 4 ∧ y = 0) ∧ x = 16 / 3)) :=
by {
  sorry
}

end x_intercept_of_perpendicular_line_is_16_over_3_l273_273368


namespace eq_solutions_count_l273_273677

def f (x a : ℝ) : ℝ := abs (abs (abs (x - a) - 1) - 1)

theorem eq_solutions_count (a b : ℝ) : 
  ∃ count : ℕ, (∀ x : ℝ, f x a = abs b → true) ∧ count = 4 :=
by
  sorry

end eq_solutions_count_l273_273677


namespace inequality_solution_set_l273_273529

theorem inequality_solution_set :
  ∀ x : ℝ, 8 * x^3 + 9 * x^2 + 7 * x - 6 < 0 ↔ (( -6 < x ∧ x < -1/8) ∨ (-1/8 < x ∧ x < 1)) :=
sorry

end inequality_solution_set_l273_273529


namespace ball_attendance_l273_273171

theorem ball_attendance
  (n m : ℕ)
  (h_total : n + m < 50)
  (h_ladies_dance : 3 * n = 4 * (Ladies who danced invited)
  (h_gentlemen_invited : 5 * m = 7 * (Gentlemen who invited)
  (h_dance_pairs : 3 * n = 20 * m) :
  n + m = 41 :=
by sorry

end ball_attendance_l273_273171


namespace sum_eq_expected_l273_273773

noncomputable def complex_sum : Complex :=
  12 * Complex.exp (Complex.I * 3 * Real.pi / 13) + 12 * Complex.exp (Complex.I * 6 * Real.pi / 13)

noncomputable def expected_value : Complex :=
  24 * Real.cos (Real.pi / 13) * Complex.exp (Complex.I * 9 * Real.pi / 26)

theorem sum_eq_expected :
  complex_sum = expected_value :=
by
  sorry

end sum_eq_expected_l273_273773


namespace no_valid_pairs_l273_273779

open Nat

theorem no_valid_pairs (l y : ℕ) (h1 : y % 30 = 0) (h2 : l > 1) :
  (∃ n m : ℕ, 180 - 360 / n = y ∧ 180 - 360 / m = l * y ∧ y * l ≤ 180) → False := 
by
  intro h
  sorry

end no_valid_pairs_l273_273779


namespace car_mpg_in_city_l273_273603

theorem car_mpg_in_city
  (H C T : ℕ)
  (h1 : H * T = 462)
  (h2 : C * T = 336)
  (h3 : C = H - 9) : C = 24 := by
  sorry

end car_mpg_in_city_l273_273603


namespace intersection_complement_l273_273332

def A : Set ℝ := { x | x^2 ≤ 4 * x }
def B : Set ℝ := { x | ∃ y, y = Real.sqrt (x - 3) }

theorem intersection_complement (x : ℝ) : 
  x ∈ A ∩ (Set.univ \ B) ↔ x ∈ Set.Ico 0 3 := 
sorry

end intersection_complement_l273_273332


namespace solve_problem_l273_273513

-- Define the polynomial g(x) as given in the problem
def g (p q r s t : ℝ) (x : ℝ) : ℝ := p * x^4 + q * x^3 + r * x^2 + s * x + t

-- Define the condition given in the problem
def condition (p q r s t : ℝ) : Prop := g p q r s t (-2) = -4

-- State the theorem to be proved
theorem solve_problem (p q r s t : ℝ) (h : condition p q r s t) :
  16 * p - 8 * q + 4 * r - 2 * s + t = 4 :=
by
  -- Proof is omitted
  sorry

end solve_problem_l273_273513


namespace sum_min_max_z_l273_273336

theorem sum_min_max_z (x y : ℝ) 
  (h1 : x - y - 2 ≥ 0) 
  (h2 : x - 5 ≤ 0) 
  (h3 : y + 2 ≥ 0) :
  ∃ (z_min z_max : ℝ), z_min = 2 ∧ z_max = 34 ∧ z_min + z_max = 36 :=
by
  sorry

end sum_min_max_z_l273_273336


namespace find_second_number_l273_273589

theorem find_second_number :
  ∃ (x y : ℕ), (y = x + 4) ∧ (x + y = 56) ∧ (y = 30) :=
by
  sorry

end find_second_number_l273_273589


namespace estate_value_l273_273999

theorem estate_value (E : ℝ) (x : ℝ) (hx : 5 * x = 0.6 * E) (charity_share : ℝ)
  (hcharity : charity_share = 800) (hwife : 3 * x * 4 = 12 * x)
  (htotal : E = 17 * x + charity_share) : E = 1923 :=
by
  sorry

end estate_value_l273_273999


namespace eggs_per_group_l273_273711

-- Define the conditions
def num_eggs : ℕ := 18
def num_groups : ℕ := 3

-- Theorem stating number of eggs per group
theorem eggs_per_group : num_eggs / num_groups = 6 :=
by
  sorry

end eggs_per_group_l273_273711


namespace set_intersection_complement_eq_l273_273809

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

theorem set_intersection_complement_eq {U : Set ℕ} {M : Set ℕ} {N : Set ℕ}
    (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2, 3}) (hN : N = {3, 4, 5}) :
    (U \ M) ∩ N = {4, 5} :=
by
  sorry

end set_intersection_complement_eq_l273_273809


namespace cannot_make_62_cents_with_five_coins_l273_273655

theorem cannot_make_62_cents_with_five_coins :
  ∀ (p n d q : ℕ), p + n + d + q = 5 ∧ q ≤ 1 →
  1 * p + 5 * n + 10 * d + 25 * q ≠ 62 := by
  intro p n d q h
  sorry

end cannot_make_62_cents_with_five_coins_l273_273655


namespace triangle_inequality_l273_273981

theorem triangle_inequality (A B C : ℝ) (k : ℝ) (hABC : A + B + C = π) (h1 : 1 ≤ k) (h2 : k ≤ 2) :
  (1 / (k - Real.cos A)) + (1 / (k - Real.cos B)) + (1 / (k - Real.cos C)) ≥ 6 / (2 * k - 1) := 
by
  sorry

end triangle_inequality_l273_273981


namespace find_prime_pair_l273_273531

-- Definition of the problem
def is_integral_expression (p q : ℕ) : Prop :=
  (p + q)^(p + q) * (p - q)^(p - q) - 1 ≠ 0 ∧
  (p + q)^(p - q) * (p - q)^(p + q) - 1 ≠ 0 ∧
  ((p + q)^(p + q) * (p - q)^(p - q) - 1) % ((p + q)^(p - q) * (p - q)^(p + q) - 1) = 0

-- Mathematical theorem to be proved
theorem find_prime_pair (p q : ℕ) (prime_p : Nat.Prime p) (prime_q : Nat.Prime q) (h : p > q) :
  is_integral_expression p q → (p, q) = (3, 2) :=
by 
  sorry

end find_prime_pair_l273_273531


namespace correct_product_of_a_and_b_l273_273691

theorem correct_product_of_a_and_b
    (a b : ℕ)
    (h_a_pos : 0 < a)
    (h_b_pos : 0 < b)
    (h_a_two_digits : 10 ≤ a ∧ a < 100)
    (a' : ℕ)
    (h_a' : a' = (a % 10) * 10 + (a / 10))
    (h_product_erroneous : a' * b = 198) :
  a * b = 198 :=
sorry

end correct_product_of_a_and_b_l273_273691


namespace gardener_area_l273_273595

-- The definition considers the placement of gardeners and the condition for attending flowers.
noncomputable def grid_assignment (gardener_position: (ℕ × ℕ)) (flower_position: (ℕ × ℕ)) : List (ℕ × ℕ) :=
  sorry

-- A theorem that states the equivalent proof.
theorem gardener_area (gardener_position: (ℕ × ℕ)) :
  ∀ flower_position: (ℕ × ℕ), (∃ g1 g2 g3, g1 ∈ grid_assignment gardener_position flower_position ∧
                                            g2 ∈ grid_assignment gardener_position flower_position ∧
                                            g3 ∈ grid_assignment gardener_position flower_position) →
  (gardener_position = g1 ∨ gardener_position = g2 ∨ gardener_position = g3) → true :=
by
  sorry

end gardener_area_l273_273595


namespace quadratic_sums_l273_273662

variables {α : Type} [CommRing α] {a b c : α}

theorem quadratic_sums 
  (h₁ : ∀ (a b c : α), a + b ≠ 0 ∧ b + c ≠ 0 ∧ c + a ≠ 0)
  (h₂ : ∀ (r₁ r₂ : α), 
    (r₁^2 + a * r₁ + b = 0 ∧ r₂^2 + b * r₂ + c = 0) → r₁ + r₂ = 0 ∧ r₁ - r₂ ≠ 0)
  (h₃ : ∀ (r₁ r₂ : α), 
    (r₁^2 + b * r₁ + c = 0 ∧ r₂^2 + c * r₂ + a = 0) → r₁ + r₂ = 0 ∧ r₁ - r₂ ≠ 0)
  (h₄ : ∀ (r₁ r₂ : α), 
    (r₁^2 + c * r₁ + a = 0 ∧ r₂^2 + a * r₂ + b = 0) → r₁ + r₂ = 0 ∧ r₁ - r₂ ≠ 0) :
  a^2 + b^2 + c^2 = 18 ∧
  a^2 * b + b^2 * c + c^2 * a = 27 ∧
  a^3 * b^2 + b^3 * c^2 + c^3 * a^2 = -162 :=
sorry

end quadratic_sums_l273_273662


namespace simplify_tangent_sum_l273_273208

theorem simplify_tangent_sum :
  (1 + Real.tan (10 * Real.pi / 180)) * (1 + Real.tan (35 * Real.pi / 180)) = 2 := by
  have h1 : Real.tan (45 * Real.pi / 180) = Real.tan ((10 + 35) * Real.pi / 180) := by sorry
  have h2 : Real.tan (45 * Real.pi / 180) = 1 := by sorry
  have h3 : ∀ x y, Real.tan (x + y) = (Real.tan x + Real.tan y) / (1 - Real.tan x * Real.tan y) := by sorry
  sorry

end simplify_tangent_sum_l273_273208


namespace rabbit_count_l273_273484

-- Define the conditions
def original_rabbits : ℕ := 8
def new_rabbits_born : ℕ := 5

-- Define the total rabbits based on the conditions
def total_rabbits : ℕ := original_rabbits + new_rabbits_born

-- The statement to prove that the total number of rabbits is 13
theorem rabbit_count : total_rabbits = 13 :=
by
  -- Proof not needed, hence using sorry
  sorry

end rabbit_count_l273_273484


namespace original_cost_l273_273259

theorem original_cost (P : ℝ) (h : 0.85 * 0.76 * P = 988) : P = 1529.41 := by
  sorry

end original_cost_l273_273259


namespace valid_param_a_valid_param_c_l273_273871

/-
The task is to prove that the goals provided are valid parameterizations of the given line.
-/

def line_eqn (x y : ℝ) : Prop := y = -7/4 * x + 21/4

def is_valid_param (p₀ : ℝ × ℝ) (d : ℝ × ℝ) : Prop :=
  ∀ t : ℝ, line_eqn ((p₀.1 + t * d.1) : ℝ) ((p₀.2 + t * d.2) : ℝ)

theorem valid_param_a : is_valid_param (7, 0) (4, -7) :=
by
  sorry

theorem valid_param_c : is_valid_param (0, 21/4) (-4, 7) :=
by
  sorry


end valid_param_a_valid_param_c_l273_273871


namespace calculate_expression_l273_273765

theorem calculate_expression : (7^2 - 5^2)^3 = 13824 := by
  sorry

end calculate_expression_l273_273765


namespace numberOfBoys_is_50_l273_273604

-- Define the number of boys and the conditions given.
def numberOfBoys (B G : ℕ) : Prop :=
  B / G = 5 / 13 ∧ G = B + 80

-- The theorem that we need to prove.
theorem numberOfBoys_is_50 (B G : ℕ) (h : numberOfBoys B G) : B = 50 :=
  sorry

end numberOfBoys_is_50_l273_273604


namespace find_x_l273_273954

theorem find_x (n : ℕ) (hn : n % 2 = 1) (hpf : ∃ p1 p2 p3 : ℕ, p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧ p1 * p2 * p3 = 9^n - 1 ∧ [p1, p2, p3].contains 61) :
  9^n - 1 = 59048 :=
by
  sorry

end find_x_l273_273954


namespace time_to_save_for_vehicle_l273_273713

def monthly_earnings : ℕ := 4000
def saving_factor : ℚ := 1 / 2
def vehicle_cost : ℕ := 16000

theorem time_to_save_for_vehicle : (vehicle_cost / (monthly_earnings * saving_factor)) = 8 := by
  sorry

end time_to_save_for_vehicle_l273_273713


namespace robot_min_steps_l273_273447

theorem robot_min_steps {a b : ℕ} (ha : 0 < a) (hb : 0 < b) : ∃ n, n = a + b - Nat.gcd a b :=
by
  sorry

end robot_min_steps_l273_273447


namespace f_analytical_expression_l273_273806

noncomputable def f (x : ℝ) : ℝ := (2^(x + 1) - 2^(-x)) / 3

theorem f_analytical_expression :
  ∀ x : ℝ, f (-x) + 2 * f x = 2^x :=
by
  sorry

end f_analytical_expression_l273_273806


namespace linear_inequality_m_eq_one_l273_273557

theorem linear_inequality_m_eq_one
  (m : ℤ)
  (h1 : |m| = 1)
  (h2 : m + 1 ≠ 0) :
  m = 1 :=
sorry

end linear_inequality_m_eq_one_l273_273557


namespace find_white_daisies_l273_273694

theorem find_white_daisies (W P R : ℕ) 
  (h1 : P = 9 * W) 
  (h2 : R = 4 * P - 3) 
  (h3 : W + P + R = 273) : 
  W = 6 :=
by
  sorry

end find_white_daisies_l273_273694


namespace shaded_to_white_area_ratio_l273_273884

-- Define the problem
theorem shaded_to_white_area_ratio :
  let total_triangles_shaded := 5
  let total_triangles_white := 3
  let ratio_shaded_to_white := total_triangles_shaded / total_triangles_white
  ratio_shaded_to_white = (5 : ℚ)/(3 : ℚ) := by
  -- Proof steps should be provided here, but "sorry" is used to skip the proof.
  sorry

end shaded_to_white_area_ratio_l273_273884


namespace exists_unique_solution_l273_273283

theorem exists_unique_solution : ∀ a b : ℝ, 2 * (a ^ 2 + 1) * (b ^ 2 + 1) = (a + 1) * (b + 1) * (a * b + 1) ↔ (a, b) = (1, 1) := by
  sorry

end exists_unique_solution_l273_273283


namespace inequality_positives_l273_273717

theorem inequality_positives (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a * (a + 1)) / (b + 1) + (b * (b + 1)) / (a + 1) ≥ a + b :=
sorry

end inequality_positives_l273_273717


namespace woman_complete_time_l273_273751

-- Define the work rate of one man
def man_rate := 1 / 100

-- Define the combined work rate equation for 10 men and 15 women completing work in 5 days
def combined_work_rate (W : ℝ) : Prop :=
  10 * man_rate + 15 * W = 1 / 5

-- Prove that given the combined work rate equation, one woman alone takes 150 days to complete the work
theorem woman_complete_time (W : ℝ) : combined_work_rate W → W = 1 / 150 :=
by
  intro h
  have h1 : 10 * man_rate + 15 * W = 1 / 5 := h
  rw [man_rate] at h1
  sorry -- Proof steps would go here

end woman_complete_time_l273_273751


namespace initial_girls_are_11_l273_273761

variable {n : ℕ}  -- Assume n (the total number of students initially) is a natural number

def initial_num_girls (n : ℕ) : ℕ := (n / 2)

def total_students_after_changes (n : ℕ) : ℕ := n - 2

def num_girls_after_changes (n : ℕ) : ℕ := (n / 2) - 3

def is_40_percent_girls (n : ℕ) : Prop := (num_girls_after_changes n) * 10 = 4 * (total_students_after_changes n)

theorem initial_girls_are_11 :
  is_40_percent_girls 22 → initial_num_girls 22 = 11 :=
by
  sorry

end initial_girls_are_11_l273_273761


namespace total_attended_ball_lt_fifty_l273_273180

-- Conditions from part a)
variable (n : ℕ) -- number of ladies
variable (m : ℕ) -- number of gentlemen

def total_people (n m : ℕ) : ℕ := n + m

-- Quarter of the ladies not invited means three-quarters danced
def ladies_danced (n : ℕ) : ℕ := 3 * n / 4

-- Two-sevenths of the gentlemen did not invite anyone
def gents_invited (m : ℕ) : ℕ := 5 * m / 7

-- From the condition we have these relations
axiom ratio_relation (h : n * 3 / 4 = m * 5 / 7) : True

-- Prove the total number of people attended the ball
theorem total_attended_ball_lt_fifty 
  (h : n * 3 / 4 = m * 5 / 7) 
  (hnm_lt_50 : total_people n m < 50) : total_people n m = 41 := 
by
  sorry

end total_attended_ball_lt_fifty_l273_273180


namespace part1_part2_l273_273133

noncomputable def a_n (n : ℕ) : ℕ :=
  2^(n - 1)

noncomputable def b_n (n : ℕ) : ℕ :=
  2 * n

noncomputable def S_n (n : ℕ) : ℕ :=
  n^2 + n

theorem part1 (n : ℕ) : 
  S_n n = n^2 + n := 
sorry

noncomputable def C_n (n : ℕ) : ℚ :=
  (n^2 + n) / 2^(n - 1)

theorem part2 (n : ℕ) (k : ℕ) (k_gt_0 : 0 < k) : 
  (∀ n, C_n n ≤ C_n k) ↔ (k = 2 ∨ k = 3) :=
sorry

end part1_part2_l273_273133


namespace dan_bought_18_stickers_l273_273104

variable (S D : ℕ)

-- Given conditions
def stickers_initially_same : Prop := S = S -- Cindy and Dan have the same number of stickers initially
def cindy_used_15_stickers : Prop := true -- Cindy used 15 of her stickers
def dan_bought_D_stickers : Prop := true -- Dan bought D stickers
def dan_has_33_more_stickers_than_cindy : Prop := (S + D) = (S - 15 + 33)

-- Question: Prove that the number of stickers Dan bought is 18
theorem dan_bought_18_stickers (h1 : stickers_initially_same S)
                               (h2 : cindy_used_15_stickers)
                               (h3 : dan_bought_D_stickers)
                               (h4 : dan_has_33_more_stickers_than_cindy S D) : D = 18 :=
sorry

end dan_bought_18_stickers_l273_273104


namespace find_salary_l273_273243

theorem find_salary (x y : ℝ) (h1 : x + y = 2000) (h2 : 0.05 * x = 0.15 * y) : x = 1500 :=
sorry

end find_salary_l273_273243


namespace polynomial_divisor_l273_273842

theorem polynomial_divisor (f : Polynomial ℂ) (n : ℕ) (h : (X - 1) ∣ (f.comp (X ^ n))) : (X ^ n - 1) ∣ (f.comp (X ^ n)) :=
sorry

end polynomial_divisor_l273_273842


namespace female_democrats_l273_273067

theorem female_democrats :
  ∀ (F M : ℕ),
  F + M = 720 →
  F/2 + M/4 = 240 →
  F / 2 = 120 :=
by
  intros F M h1 h2
  sorry

end female_democrats_l273_273067


namespace total_attended_ball_lt_fifty_l273_273178

-- Conditions from part a)
variable (n : ℕ) -- number of ladies
variable (m : ℕ) -- number of gentlemen

def total_people (n m : ℕ) : ℕ := n + m

-- Quarter of the ladies not invited means three-quarters danced
def ladies_danced (n : ℕ) : ℕ := 3 * n / 4

-- Two-sevenths of the gentlemen did not invite anyone
def gents_invited (m : ℕ) : ℕ := 5 * m / 7

-- From the condition we have these relations
axiom ratio_relation (h : n * 3 / 4 = m * 5 / 7) : True

-- Prove the total number of people attended the ball
theorem total_attended_ball_lt_fifty 
  (h : n * 3 / 4 = m * 5 / 7) 
  (hnm_lt_50 : total_people n m < 50) : total_people n m = 41 := 
by
  sorry

end total_attended_ball_lt_fifty_l273_273178


namespace remainder_cd_mod_40_l273_273450

theorem remainder_cd_mod_40 (c d : ℤ) (hc : c % 80 = 75) (hd : d % 120 = 117) : (c + d) % 40 = 32 :=
by
  sorry

end remainder_cd_mod_40_l273_273450


namespace johns_total_spending_l273_273637

theorem johns_total_spending:
  ∀ (X : ℝ), (3/7 * X + 2/5 * X + 1/4 * X + 1/14 * X + 12 = X) → X = 80 :=
by
  intro X h
  sorry

end johns_total_spending_l273_273637


namespace triangle_third_side_possibilities_l273_273818

theorem triangle_third_side_possibilities (x : ℕ) : 
  (6 + 8 > x) ∧ (x + 6 > 8) ∧ (x + 8 > 6) → 
  3 ≤ x ∧ x < 14 → 
  ∃ n, n = 11 :=
by
  sorry

end triangle_third_side_possibilities_l273_273818


namespace pentagonal_tiles_count_l273_273613

theorem pentagonal_tiles_count (t p : ℕ) (h1 : t + p = 30) (h2 : 3 * t + 5 * p = 100) : p = 5 :=
sorry

end pentagonal_tiles_count_l273_273613


namespace arithmetic_seq_8th_term_l273_273720

theorem arithmetic_seq_8th_term (a d : ℤ) 
  (h4 : a + 3 * d = 22) 
  (h6 : a + 5 * d = 46) : 
  a + 7 * d = 70 :=
by 
  sorry

end arithmetic_seq_8th_term_l273_273720


namespace students_on_field_trip_l273_273854

theorem students_on_field_trip 
    (vans : ℕ)
    (van_capacity : ℕ)
    (adults : ℕ)
    (students : ℕ)
    (H1 : vans = 3)
    (H2 : van_capacity = 8)
    (H3 : adults = 2)
    (H4 : students = vans * van_capacity - adults) :
    students = 22 := 
by 
  sorry

end students_on_field_trip_l273_273854


namespace original_apples_l273_273897

-- Define the conditions using the given data
def sells_fraction : ℝ := 0.40 -- Fraction of apples sold
def remaining_apples : ℝ := 420 -- Apples remaining after selling

-- Theorem statement for proving the original number of apples given the conditions
theorem original_apples (x : ℝ) (sells_fraction : ℝ := 0.40) (remaining_apples : ℝ := 420) : 
  420 / (1 - sells_fraction) = x :=
sorry

end original_apples_l273_273897


namespace right_isosceles_areas_l273_273911

theorem right_isosceles_areas (A B C : ℝ) (hA : A = 1 / 2 * 5 * 5) (hB : B = 1 / 2 * 12 * 12) (hC : C = 1 / 2 * 13 * 13) :
  A + B = C :=
by
  sorry

end right_isosceles_areas_l273_273911


namespace amount_c_gets_l273_273935

theorem amount_c_gets (total_amount : ℕ) (ratio_b ratio_c : ℕ) (h_total_amount : total_amount = 2000) (h_ratio : ratio_b = 4 ∧ ratio_c = 16) : ∃ (c_amount: ℕ), c_amount = 1600 :=
by
  sorry

end amount_c_gets_l273_273935


namespace bananas_per_box_l273_273997

theorem bananas_per_box (total_bananas : ℕ) (num_boxes : ℕ) (h1 : total_bananas = 40) (h2 : num_boxes = 8) :
  total_bananas / num_boxes = 5 := by
  sorry

end bananas_per_box_l273_273997


namespace abc_plus_ab_plus_a_div_4_l273_273582

noncomputable def prob_abc_div_4 (a b c : ℕ) (isPositive_a : 0 < a) (isPositive_b : 0 < b) (isPositive_c : 0 < c) (a_in_range : a ∈ {k | 1 ≤ k ∧ k ≤ 2009}) (b_in_range : b ∈ {k | 1 ≤ k ∧ k ≤ 2009}) (c_in_range : c ∈ {k | 1 ≤ k ∧ k ≤ 2009}) : ℚ :=
  let total_elements : ℚ := 2009
  let multiples_of_4 := 502
  let non_multiples_of_4 := total_elements - multiples_of_4
  let prob_a_div_4 : ℚ := multiples_of_4 / total_elements
  let prob_a_not_div_4 : ℚ := non_multiples_of_4 / total_elements
  sorry

theorem abc_plus_ab_plus_a_div_4 : ∃ P : ℚ, prob_abc_div_4 a b c isPositive_a isPositive_b isPositive_c a_in_range b_in_range c_in_range = P :=
by sorry

end abc_plus_ab_plus_a_div_4_l273_273582


namespace nested_radical_eq_5_l273_273119

noncomputable def nested_radical : ℝ :=
  Inf { x | x = Real.sqrt (20 + x)}

theorem nested_radical_eq_5 : ∃ x : ℝ, nested_radical = 5 :=
by
  sorry

end nested_radical_eq_5_l273_273119


namespace overall_class_average_proof_l273_273430

noncomputable def group_1_weighted_average := (0.40 * 80) + (0.60 * 80)
noncomputable def group_2_weighted_average := (0.30 * 60) + (0.70 * 60)
noncomputable def group_3_weighted_average := (0.50 * 40) + (0.50 * 40)
noncomputable def group_4_weighted_average := (0.20 * 50) + (0.80 * 50)

noncomputable def overall_class_average := (0.20 * group_1_weighted_average) + 
                                           (0.50 * group_2_weighted_average) + 
                                           (0.25 * group_3_weighted_average) + 
                                           (0.05 * group_4_weighted_average)

theorem overall_class_average_proof : overall_class_average = 58.5 :=
by 
  unfold overall_class_average
  unfold group_1_weighted_average
  unfold group_2_weighted_average
  unfold group_3_weighted_average
  unfold group_4_weighted_average
  -- now perform the arithmetic calculations
  sorry

end overall_class_average_proof_l273_273430


namespace total_problems_l273_273913

theorem total_problems (C : ℕ) (W : ℕ)
  (h1 : C = 20)
  (h2 : 3 * C + 5 * W = 110) : 
  C + W = 30 := by
  sorry

end total_problems_l273_273913


namespace count_multiples_of_4_in_range_l273_273971

-- Define the predicate for multiples of 4
def is_multiple_of_4 (n : ℕ) : Prop := n % 4 = 0

-- Define the range predicate
def in_range (n : ℕ) (a b : ℕ) : Prop := a ≤ n ∧ n ≤ b

-- Formulate the main theorem
theorem count_multiples_of_4_in_range (a b : ℕ) (a := 50) (b := 300) : 
  (∑ i in (Finset.filter (λ n, is_multiple_of_4 n ∧ in_range n a b) (Finset.range (b + 1))), 1) = 63 := 
by
  sorry

end count_multiples_of_4_in_range_l273_273971


namespace find_original_number_l273_273069

def original_four_digit_number (N : ℕ) : Prop :=
  N >= 1000 ∧ N < 10000 ∧ (70000 + N) - (10 * N + 7) = 53208

theorem find_original_number (N : ℕ) (h : original_four_digit_number N) : N = 1865 :=
by
  sorry

end find_original_number_l273_273069


namespace min_sum_sequence_n_l273_273441

theorem min_sum_sequence_n (S : ℕ → ℤ) (h : ∀ n, S n = n * n - 48 * n) : 
  ∃ n, n = 24 ∧ ∀ m, S n ≤ S m :=
by
  sorry

end min_sum_sequence_n_l273_273441


namespace bridge_length_is_100_l273_273633

noncomputable def length_of_bridge (train_length : ℝ) (train_speed_kmh : ℝ) (wind_speed_kmh : ℝ) (crossing_time_s : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let wind_speed_ms := wind_speed_kmh * 1000 / 3600
  let effective_speed_ms := train_speed_ms - wind_speed_ms
  let distance_covered := effective_speed_ms * crossing_time_s
  distance_covered - train_length

theorem bridge_length_is_100 :
  length_of_bridge 150 45 15 30 = 100 :=
by
  sorry

end bridge_length_is_100_l273_273633


namespace multiples_of_4_between_50_and_300_l273_273969

theorem multiples_of_4_between_50_and_300 : 
  (∃ n : ℕ, 50 < n ∧ n < 300 ∧ n % 4 = 0) ∧ 
  (∃ k : ℕ, k = 62) :=
by
  sorry

end multiples_of_4_between_50_and_300_l273_273969


namespace max_of_x_l273_273446

theorem max_of_x (x y z : ℝ) (h1 : x + y + z = 7) (h2 : xy + xz + yz = 10) : x ≤ 3 := by
  sorry

end max_of_x_l273_273446


namespace calculate_product_l273_273926

theorem calculate_product (a : ℝ) : 2 * a * (3 * a) = 6 * a^2 := by
  -- This will skip the proof, denoted by 'sorry'
  sorry

end calculate_product_l273_273926


namespace minimum_value_expression_l273_273651

theorem minimum_value_expression (a x1 x2 : ℝ) (h_pos : 0 < a)
  (h1 : x1 + x2 = 4 * a)
  (h2 : x1 * x2 = 3 * a^2)
  (h_ineq : ∀ x, x^2 - 4 * a * x + 3 * a^2 < 0 ↔ x1 < x ∧ x < x2) :
  x1 + x2 + a / (x1 * x2) = (4 * Real.sqrt 3) / 3 :=
by
  sorry

end minimum_value_expression_l273_273651


namespace inscribed_sphere_volume_l273_273254

theorem inscribed_sphere_volume (edge_length : ℝ) (h_edge : edge_length = 12) : 
  ∃ (V : ℝ), V = 288 * Real.pi :=
by
  sorry

end inscribed_sphere_volume_l273_273254


namespace race_results_l273_273979

-- Competitor times in seconds
def time_A : ℕ := 40
def time_B : ℕ := 50
def time_C : ℕ := 55

-- Time difference calculations
def time_diff_AB := time_B - time_A
def time_diff_AC := time_C - time_A
def time_diff_BC := time_C - time_B

theorem race_results :
  time_diff_AB = 10 ∧ time_diff_AC = 15 ∧ time_diff_BC = 5 :=
by
  -- Placeholder for proof
  sorry

end race_results_l273_273979


namespace exists_quadratic_sequence_l273_273620

theorem exists_quadratic_sequence (b c : ℤ) : ∃ n : ℕ, ∃ (a : ℕ → ℤ), (a 0 = b) ∧ (a n = c) ∧ ∀ i : ℕ, 1 ≤ i → i ≤ n → |a i - a (i - 1)| = i ^ 2 := 
sorry

end exists_quadratic_sequence_l273_273620


namespace f_at_pos_eq_l273_273350

noncomputable def f (x : ℝ) : ℝ :=
  if h : x < 0 then x * (x - 1)
  else if h : x > 0 then -x * (x + 1)
  else 0

theorem f_at_pos_eq (x : ℝ) (hx : 0 < x) : f x = -x * (x + 1) :=
by
  -- Assume f is an odd function
  have h_odd : ∀ x : ℝ, f (-x) = -f x := sorry
  
  -- Given for x in (-∞, 0), f(x) = x * (x - 1)
  have h_neg : ∀ x : ℝ, x < 0 → f x = x * (x - 1) := sorry
  
  -- Prove for x > 0, f(x) = -x * (x + 1)
  sorry

end f_at_pos_eq_l273_273350


namespace rice_weight_per_container_in_grams_l273_273377

-- Define the initial problem conditions
def total_weight_pounds : ℚ := 35 / 6
def number_of_containers : ℕ := 5
def pound_to_grams : ℚ := 453.592

-- Define the expected answer
def expected_answer : ℚ := 529.1907

-- The statement to prove
theorem rice_weight_per_container_in_grams :
  (total_weight_pounds / number_of_containers) * pound_to_grams = expected_answer :=
by
  sorry

end rice_weight_per_container_in_grams_l273_273377


namespace sam_total_cans_l273_273856

theorem sam_total_cans (bags_sat : ℕ) (bags_sun : ℕ) (cans_per_bag : ℕ) 
  (h_sat : bags_sat = 3) (h_sun : bags_sun = 4) (h_cans : cans_per_bag = 9) : 
  (bags_sat + bags_sun) * cans_per_bag = 63 := 
by
  sorry

end sam_total_cans_l273_273856


namespace matchstick_triangle_sides_l273_273486

theorem matchstick_triangle_sides (a b c : ℕ) :
  a + b + c = 100 ∧ max a (max b c) = 3 * min a (min b c) ∧
  (a < b ∧ b < c ∨ a < c ∧ c < b ∨ b < a ∧ a < c) →
  (a = 15 ∧ b = 40 ∧ c = 45 ∨ a = 16 ∧ b = 36 ∧ c = 48) :=
by
  sorry

end matchstick_triangle_sides_l273_273486


namespace problem_dorlir_ahmeti_equality_case_l273_273559

theorem problem_dorlir_ahmeti (x y z : ℝ)
  (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z)
  (h : x^2 + y^2 + z^2 = x + y + z) : 
  (x + 1) / Real.sqrt (x^5 + x + 1) + 
  (y + 1) / Real.sqrt (y^5 + y + 1) + 
  (z + 1) / Real.sqrt (z^5 + z + 1) ≥ 3 :=
sorry
  
theorem equality_case (x y z : ℝ)
  (hx : x = 0) (hy : y = 0) (hz : z = 0) : 
  (x + 1) / Real.sqrt (x^5 + x + 1) + 
  (y + 1) / Real.sqrt (y^5 + y + 1) + 
  (z + 1) / Real.sqrt (z^5 + z + 1) = 3 :=
sorry

end problem_dorlir_ahmeti_equality_case_l273_273559


namespace diane_15_cents_arrangement_l273_273278

def stamps : List (ℕ × ℕ) := 
  [(1, 1), 
   (2, 2), 
   (3, 3), 
   (4, 4), 
   (5, 5), 
   (6, 6), 
   (7, 7), 
   (8, 8), 
   (9, 9), 
   (10, 10), 
   (11, 11), 
   (12, 12)]

def number_of_arrangements (value : ℕ) (stamps : List (ℕ × ℕ)) : ℕ := sorry

theorem diane_15_cents_arrangement : number_of_arrangements 15 stamps = 32 := 
sorry

end diane_15_cents_arrangement_l273_273278


namespace corner_cell_revisit_l273_273253

theorem corner_cell_revisit
    (M N : ℕ)
    (hM : M = 101)
    (hN : N = 200)
    (initial_position : ℕ × ℕ)
    (h_initial : initial_position = (0, 0) ∨ initial_position = (0, 200) ∨ initial_position = (101, 0) ∨ initial_position = (101, 200)) :
    ∃ final_position : ℕ × ℕ, 
      final_position = initial_position ∧ (final_position = (0, 0) ∨ final_position = (0, 200) ∨ final_position = (101, 0) ∨ final_position = (101, 200)) :=
by
  sorry

end corner_cell_revisit_l273_273253


namespace exists_monotonicity_b_range_l273_273308

noncomputable def f (a x : ℝ) : ℝ := (1 / 2) * a * x ^ 2 - 2 * a * x + Real.log x

theorem exists_monotonicity_b_range :
  ∀ (a : ℝ) (b : ℝ), 1 < a ∧ a < 2 →
  (∀ (x0 : ℝ), x0 ∈ Set.Icc (1 + Real.sqrt 2 / 2) 2 →
   f a x0 + Real.log (a + 1) > b * (a^2 - 1) - (a + 1) + 2 * Real.log 2) →
   b ∈ Set.Iic (-1/4) :=
sorry

end exists_monotonicity_b_range_l273_273308


namespace soda_amount_l273_273341

theorem soda_amount (S : ℝ) (h1 : S / 2 + 2000 = (S - (S / 2 + 2000)) / 2 + 2000) : S = 12000 :=
by
  sorry

end soda_amount_l273_273341


namespace angle_A_condition_area_range_condition_l273_273829

/-- Given a triangle ABC with sides opposite to internal angles A, B, and C labeled as a, b, and c respectively. 
Given the condition a * cos C + sqrt 3 * a * sin C = b + c.
Prove that angle A = π / 3.
-/
theorem angle_A_condition
  (a b c : ℝ) (C : ℝ) (h : a * Real.cos C + Real.sqrt 3 * a * Real.sin C = b + c) :
  A = Real.pi / 3 := sorry
  
/-- Given an acute triangle ABC with b = 2 and angle A = π / 3,
find the range of possible values for the area of the triangle ABC.
-/
theorem area_range_condition
  (a c : ℝ) (A : ℝ) (b : ℝ) (C B : ℝ)
  (h1 : b = 2)
  (h2 : A = Real.pi / 3)
  (h3 : 0 < B) (h4 : B < Real.pi / 2)
  (h5 : 0 < C) (h6 : C < Real.pi / 2)
  (h7 : A + C = 2 * Real.pi / 3) :
  Real.sqrt 3 / 2 < (1 / 2) * a * b * Real.sin C ∧
  (1 / 2) * a * b * Real.sin C < 2 * Real.sqrt 3 := sorry

end angle_A_condition_area_range_condition_l273_273829


namespace cell_phone_plan_cost_equality_l273_273497

theorem cell_phone_plan_cost_equality (x : ℝ) :
  let cost_A := 0.25 * x + 9
  let cost_B := 0.40 * x
  cost_A = cost_B → x = 60 := 
by
  intros cost_A_eq_cost_B
  have : 0.25 * x + 9 = 0.40 * x := cost_A_eq_cost_B
  sorry

end cell_phone_plan_cost_equality_l273_273497


namespace find_f_value_l273_273658

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f (x : ℝ) : f (-x) = -f x
axiom even_f_shift (x : ℝ) : f (-x + 1) = f (x + 1)
axiom f_interval (x : ℝ) (h : 2 < x ∧ x < 4) : f x = |x - 3|

theorem find_f_value : f 1 + f 2 + f 3 + f 4 = 0 :=
by
  sorry

end find_f_value_l273_273658


namespace min_value_xyz_l273_273660

theorem min_value_xyz (x y z : ℝ) (h1 : xy + 2 * z = 1) (h2 : x^2 + y^2 + z^2 = 10 ) : xyz ≥ -28 :=
by
  sorry

end min_value_xyz_l273_273660


namespace number_of_males_in_village_l273_273023

-- Given the total population is 800 and it is divided into four equal groups.
def total_population : ℕ := 800
def num_groups : ℕ := 4

-- Proof statement
theorem number_of_males_in_village : (total_population / num_groups) = 200 := 
by sorry

end number_of_males_in_village_l273_273023


namespace correct_calculation_l273_273063

theorem correct_calculation :
  3 * Real.sqrt 2 - (Real.sqrt 2 / 2) = (5 / 2) * Real.sqrt 2 :=
by
  -- To proceed with the proof, we need to show:
  -- 3 * sqrt(2) - (sqrt(2) / 2) = (5 / 2) * sqrt(2)
  sorry

end correct_calculation_l273_273063


namespace hostel_cost_for_23_days_l273_273747

theorem hostel_cost_for_23_days :
  let first_week_days := 7
  let additional_days := 23 - first_week_days
  let cost_first_week := 18 * first_week_days
  let cost_additional_weeks := 11 * additional_days
  23 * ((cost_first_week + cost_additional_weeks) / 23) = 302 :=
by sorry

end hostel_cost_for_23_days_l273_273747


namespace maximize_profit_l273_273632

def cost_per_product : ℝ := 3
def management_fee_per_product : ℝ := 3
def sales_volume (x : ℝ) : ℝ := (12 - x)^2 * 10000
def annual_profit (x : ℝ) : ℝ := (x - cost_per_product - management_fee_per_product) * sales_volume x

theorem maximize_profit :
  (∀ x : ℝ, 9 ≤ x ∧ x ≤ 11 → annual_profit x = x^3 - 30*x^2 + 288*x - 864) ∧
  annual_profit 9 = 27 * 10000 ∧
  (∀ x : ℝ, 9 ≤ x ∧ x ≤ 11 → annual_profit x ≤ annual_profit 9) :=
by
  sorry

end maximize_profit_l273_273632


namespace tape_recorder_cost_l273_273857

theorem tape_recorder_cost (x y : ℕ) (h1 : 170 ≤ x * y) (h2 : x * y ≤ 195)
  (h3 : (y - 2) * (x + 1) = x * y) : x * y = 180 :=
by
  sorry

end tape_recorder_cost_l273_273857


namespace total_attended_ball_lt_fifty_l273_273179

-- Conditions from part a)
variable (n : ℕ) -- number of ladies
variable (m : ℕ) -- number of gentlemen

def total_people (n m : ℕ) : ℕ := n + m

-- Quarter of the ladies not invited means three-quarters danced
def ladies_danced (n : ℕ) : ℕ := 3 * n / 4

-- Two-sevenths of the gentlemen did not invite anyone
def gents_invited (m : ℕ) : ℕ := 5 * m / 7

-- From the condition we have these relations
axiom ratio_relation (h : n * 3 / 4 = m * 5 / 7) : True

-- Prove the total number of people attended the ball
theorem total_attended_ball_lt_fifty 
  (h : n * 3 / 4 = m * 5 / 7) 
  (hnm_lt_50 : total_people n m < 50) : total_people n m = 41 := 
by
  sorry

end total_attended_ball_lt_fifty_l273_273179


namespace value_of_nested_radical_l273_273120

def nested_radical : ℝ := 
  sorry -- Definition of the recurring expression is needed here, let's call it x
  
theorem value_of_nested_radical :
  (nested_radical = 5) :=
sorry -- The actual proof steps will be written here.

end value_of_nested_radical_l273_273120


namespace scientific_notation_of_12_06_million_l273_273455

theorem scientific_notation_of_12_06_million :
  12.06 * 10^6 = 1.206 * 10^7 :=
sorry

end scientific_notation_of_12_06_million_l273_273455


namespace find_largest_natural_number_l273_273794

theorem find_largest_natural_number :
  ∃ n : ℕ, (forall m : ℕ, m > n -> m ^ 300 ≥ 3 ^ 500) ∧ n ^ 300 < 3 ^ 500 :=
  sorry

end find_largest_natural_number_l273_273794


namespace expected_value_of_third_flip_l273_273753

-- Definitions for the conditions
def prob_heads : ℚ := 2/5
def prob_tails : ℚ := 3/5
def win_amount : ℚ := 4
def base_loss : ℚ := 3
def doubled_loss : ℚ := 2 * base_loss
def first_two_flips_were_tails : Prop := true 

-- The main statement: Proving the expected value of the third flip
theorem expected_value_of_third_flip (h : first_two_flips_were_tails) : 
  (prob_heads * win_amount + prob_tails * -doubled_loss) = -2 := by
  sorry

end expected_value_of_third_flip_l273_273753


namespace total_journey_distance_l273_273242

/-- 
A woman completes a journey in 5 hours. She travels the first half of the journey 
at 21 km/hr and the second half at 24 km/hr. Find the total journey in km.
-/
theorem total_journey_distance :
  ∃ D : ℝ, (D / 2) / 21 + (D / 2) / 24 = 5 ∧ D = 112 :=
by
  use 112
  -- Please prove the following statements
  sorry

end total_journey_distance_l273_273242


namespace text_message_cost_eq_l273_273498

theorem text_message_cost_eq (x : ℝ) (CA CB : ℝ) : 
  (CA = 0.25 * x + 9) → (CB = 0.40 * x) → CA = CB → x = 60 :=
by
  intros hCA hCB heq
  sorry

end text_message_cost_eq_l273_273498


namespace red_points_count_exists_exact_red_points_l273_273045

noncomputable def count_red_midpoints (points : Finset (ℝ × ℝ)) : ℕ :=
  (Finset.image (λ (p : (ℝ × ℝ) × (ℝ × ℝ)), ((p.1.1 + p.2.1) / 2, (p.1.2 + p.2.2) / 2))
    (points.product points)).card

theorem red_points_count (points : Finset (ℝ × ℝ)) (h : points.card = 997) :
  count_red_midpoints points ≥ 1991 :=
sorry

theorem exists_exact_red_points (h : ∃ (points : Finset (ℝ × ℝ)), points.card = 997 ∧ count_red_midpoints points = 1991) :
  ∃ (points : Finset (ℝ × ℝ)), points.card = 997 ∧ count_red_midpoints points = 1991 :=
h

end red_points_count_exists_exact_red_points_l273_273045


namespace largest_interior_angle_l273_273732

theorem largest_interior_angle (x : ℝ) (h_ratio : (5*x + 4*x + 3*x = 360)) :
  let e1 := 3 * x
  let e2 := 4 * x
  let e3 := 5 * x
  let i1 := 180 - e1
  let i2 := 180 - e2
  let i3 := 180 - e3
  max i1 (max i2 i3) = 90 :=
sorry

end largest_interior_angle_l273_273732


namespace probability_of_intersecting_diagonals_l273_273363

def num_vertices := 8
def total_diagonals := (num_vertices * (num_vertices - 3)) / 2
def total_ways_to_choose_two_diagonals := Nat.choose total_diagonals 2
def ways_to_choose_4_vertices := Nat.choose num_vertices 4
def number_of_intersecting_pairs := ways_to_choose_4_vertices
def probability_intersecting_diagonals := (number_of_intersecting_pairs : ℚ) / (total_ways_to_choose_two_diagonals : ℚ)

theorem probability_of_intersecting_diagonals :
  probability_intersecting_diagonals = 7 / 19 := by
  sorry

end probability_of_intersecting_diagonals_l273_273363


namespace smallest_positive_angle_l273_273479

theorem smallest_positive_angle (theta : ℝ) (h_theta : theta = -2002) :
  ∃ α : ℝ, 0 < α ∧ α < 360 ∧ ∃ k : ℤ, theta = α + k * 360 ∧ α = 158 := 
by
  sorry

end smallest_positive_angle_l273_273479


namespace inequality_solution_set_l273_273741

open Set

theorem inequality_solution_set (a : ℝ) (h : a < 0) :
  {x : ℝ | (x - 5 * a) * (x + a) > 0} = {x | x < 5 * a ∨ x > -a} :=
sorry

end inequality_solution_set_l273_273741


namespace ineq_pos_xy_l273_273457

theorem ineq_pos_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x + y) / Real.sqrt (x * y) ≤ x / y + y / x := 
sorry

end ineq_pos_xy_l273_273457


namespace quadratic_inequality_solution_l273_273819

theorem quadratic_inequality_solution (a : ℝ) :
  (∀ x : ℝ, x^2 - 2 * a * x + a > 0) ↔ (0 < a ∧ a < 1) :=
by
  sorry

end quadratic_inequality_solution_l273_273819


namespace frac_mul_sub_eq_l273_273060

/-
  Theorem:
  The result of multiplying 2/9 by 4/5 and then subtracting 1/45 is equal to 7/45.
-/
theorem frac_mul_sub_eq :
  (2/9 * 4/5 - 1/45) = 7/45 :=
by
  sorry

end frac_mul_sub_eq_l273_273060


namespace largest_three_digit_sum_l273_273556

open Nat

def isDigit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

def areDistinct (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem largest_three_digit_sum : 
  ∀ (X Y Z : ℕ), isDigit X → isDigit Y → isDigit Z → areDistinct X Y Z →
  100 ≤  (110 * X + 11 * Y + 2 * Z) → (110 * X + 11 * Y + 2 * Z) ≤ 999 → 
  110 * X + 11 * Y + 2 * Z ≤ 982 :=
by
  intros
  sorry

end largest_three_digit_sum_l273_273556


namespace binom_1293_1_eq_1293_l273_273105

theorem binom_1293_1_eq_1293 : (Nat.choose 1293 1) = 1293 := 
  sorry

end binom_1293_1_eq_1293_l273_273105


namespace min_value_x1_squared_plus_x2_squared_plus_x3_squared_l273_273192

theorem min_value_x1_squared_plus_x2_squared_plus_x3_squared
    (x1 x2 x3 : ℝ) 
    (h1 : 3 * x1 + 2 * x2 + x3 = 30) 
    (h2 : x1 > 0) 
    (h3 : x2 > 0) 
    (h4 : x3 > 0) : 
    x1^2 + x2^2 + x3^2 ≥ 125 := 
  by sorry

end min_value_x1_squared_plus_x2_squared_plus_x3_squared_l273_273192


namespace value_of_N_l273_273892

theorem value_of_N : ∃ N : ℕ, (32^5 * 16^4 / 8^7) = 2^N ∧ N = 20 := by
  use 20
  sorry

end value_of_N_l273_273892


namespace function_passes_through_fixed_point_l273_273351

theorem function_passes_through_fixed_point (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) : (2 - a^(0 : ℝ) = 1) :=
by
  sorry

end function_passes_through_fixed_point_l273_273351


namespace standard_eq_of_ellipse_value_of_k_l273_273126

-- Definitions and conditions
def is_ellipse (a b : ℝ) : Prop :=
  a > b ∧ b > 0

def eccentricity (a b : ℝ) (e : ℝ) : Prop :=
  e = (Real.sqrt 2) / 2 ∧ a^2 = b^2 + (a * e)^2

def minor_axis_length (b : ℝ) : Prop :=
  2 * b = 2

def is_tangency (k m : ℝ) : Prop := 
  m^2 = 1 + k^2

def line_intersect_ellipse (k m : ℝ) : Prop :=
  (4 * k * m)^2 - 4 * (1 + 2 * k^2) * (2 * m^2 - 2) > 0

def dot_product_condition (k m : ℝ) : Prop :=
  let x1 := -(4 * k * m) / (1 + 2 * k^2)
  let x2 := (2 * m^2 - 2) / (1 + 2 * k^2)
  let y1 := k * x1 + m
  let y2 := k * x2 + m
  x1 * x2 + y1 * y2 = 2 / 3

-- To prove the standard equation of the ellipse
theorem standard_eq_of_ellipse {a b : ℝ} (h_ellipse : is_ellipse a b)
  (h_eccentricity : eccentricity a b ((Real.sqrt 2) / 2)) 
  (h_minor_axis : minor_axis_length b) : 
  ∃ a, a = Real.sqrt 2 ∧ b = 1 ∧ (∀ x y, (x^2 / 2 + y^2 = 1)) := 
sorry

-- To prove the value of k
theorem value_of_k {k m : ℝ} (h_tangency : is_tangency k m) 
  (h_intersect : line_intersect_ellipse k m)
  (h_dot_product : dot_product_condition k m) :
  k = 1 ∨ k = -1 :=
sorry

end standard_eq_of_ellipse_value_of_k_l273_273126


namespace multiple_of_75_with_36_divisors_l273_273793

theorem multiple_of_75_with_36_divisors (n : ℕ) (h1 : n % 75 = 0) (h2 : ∃ (a b c : ℕ), a ≥ 1 ∧ b ≥ 2 ∧ n = 3^a * 5^b * (2^c) ∧ (a+1)*(b+1)*(c+1) = 36) : n / 75 = 24 := 
sorry

end multiple_of_75_with_36_divisors_l273_273793


namespace value_of_f_f_2_l273_273134

def f (x : ℤ) : ℤ := 4 * x^2 + 2 * x - 1

theorem value_of_f_f_2 : f (f 2) = 1481 := by
  sorry

end value_of_f_f_2_l273_273134


namespace product_divisible_by_eight_l273_273679

theorem product_divisible_by_eight (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 96) : 
  8 ∣ n * (n + 1) * (n + 2) := 
sorry

end product_divisible_by_eight_l273_273679


namespace probability_correct_l273_273339

-- Defining the values on the spinner
inductive SpinnerValue
| Bankrupt
| Thousand
| EightHundred
| FiveThousand
| Thousand'

open SpinnerValue

-- Function to get value in number from SpinnerValue
def value (v : SpinnerValue) : ℕ :=
  match v with
  | Bankrupt => 0
  | Thousand => 1000
  | EightHundred => 800
  | FiveThousand => 5000
  | Thousand' => 1000

-- Total number of spins
def total_spins : ℕ := 3

-- Total possible outcomes
def total_outcomes : ℕ := (5 : ℕ) ^ total_spins

-- Number of favorable outcomes (count of permutations summing to 5800)
def favorable_outcomes : ℕ :=
  12  -- This comes from solution steps

-- The probability as a ratio of favorable outcomes to total outcomes
def probability_of_5800_in_three_spins : ℚ :=
  favorable_outcomes / total_outcomes

theorem probability_correct :
  probability_of_5800_in_three_spins = 12 / 125 := by
  sorry

end probability_correct_l273_273339


namespace trapezoid_rectangle_ratio_l273_273767

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

end trapezoid_rectangle_ratio_l273_273767


namespace a2017_value_l273_273425

def seq (a : ℕ → ℝ) : Prop := ∀ n, a (n + 1) = a n / (a n + 1)

theorem a2017_value :
  ∃ (a : ℕ → ℝ),
  seq a ∧ a 1 = 1 / 2 ∧ a 2017 = 1 / 2018 :=
by
  sorry

end a2017_value_l273_273425


namespace triangle_inequality_l273_273200

theorem triangle_inequality (A B C : ℝ) (h : A + B + C = Real.pi) :
  (1 - Real.sqrt (Real.sqrt 3 * Real.tan (A / 2)) + Real.sqrt 3 * Real.tan (A / 2)) * 
  (1 - Real.sqrt (Real.sqrt 3 * Real.tan (B / 2)) + Real.sqrt 3 * Real.tan (B / 2)) +
  (1 - Real.sqrt (Real.sqrt 3 * Real.tan (B / 2)) + Real.sqrt 3 * Real.tan (B / 2)) * 
  (1 - Real.sqrt (Real.sqrt 3 * Real.tan (C / 2)) + Real.sqrt 3 * Real.tan (C / 2)) +
  (1 - Real.sqrt (Real.sqrt 3 * Real.tan (C / 2)) + Real.sqrt 3 * Real.tan (C / 2)) * 
  (1 - Real.sqrt (Real.sqrt 3 * Real.tan (A / 2)) + Real.sqrt 3 * Real.tan (A / 2)) ≥ 3 :=
by
  sorry

end triangle_inequality_l273_273200


namespace find_a_b_find_c_range_l273_273194

noncomputable def f (a b c x : ℝ) : ℝ := 2 * x^3 + 3 * a * x^2 + 3 * b * x + 8 * c

theorem find_a_b (a b c : ℝ) (extreme_x1 extreme_x2 : ℝ) (h1 : extreme_x1 = 1) (h2 : extreme_x2 = 2) 
  (h3 : (deriv (f a b c) 1) = 0) (h4 : (deriv (f a b c) 2) = 0) : 
  a = -3 ∧ b = 4 :=
by sorry

theorem find_c_range (c : ℝ) (h : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 3 → f (-3) 4 c x < c^2) : 
  c ∈ Set.Iio (-1) ∪ Set.Ioi 9 :=
by sorry

end find_a_b_find_c_range_l273_273194


namespace number_is_45_percent_of_27_l273_273361

theorem number_is_45_percent_of_27 (x : ℝ) (h : 27 / x = 45 / 100) : x = 60 := 
by
  sorry

end number_is_45_percent_of_27_l273_273361


namespace complex_sum_magnitude_eq_three_l273_273335

open Complex

theorem complex_sum_magnitude_eq_three (a b c : ℂ) 
    (h1 : abs a = 1) 
    (h2 : abs b = 1) 
    (h3 : abs c = 1) 
    (h4 : a^3 / (b * c) + b^3 / (a * c) + c^3 / (a * b) = -3) : 
    abs (a + b + c) = 3 := 
sorry

end complex_sum_magnitude_eq_three_l273_273335


namespace cost_of_each_art_book_l273_273918

-- Define the conditions
def total_cost : ℕ := 30
def cost_per_math_and_science_book : ℕ := 3
def num_math_books : ℕ := 2
def num_art_books : ℕ := 3
def num_science_books : ℕ := 6

-- The proof problem statement
theorem cost_of_each_art_book :
  (total_cost - (num_math_books * cost_per_math_and_science_book + num_science_books * cost_per_math_and_science_book)) / num_art_books = 2 :=
by
  sorry -- proof goes here,

end cost_of_each_art_book_l273_273918


namespace sin_eq_x_over_20_has_11_roots_l273_273676

open Real

theorem sin_eq_x_over_20_has_11_roots :
  ∃ n : ℕ, (20 * (n:ℝ) ∈ (11 : ℝ)) := sorry

end sin_eq_x_over_20_has_11_roots_l273_273676


namespace tortoise_age_l273_273257

-- Definitions based on the given problem conditions
variables (a b c : ℕ)

-- The conditions as provided in the problem
def condition1 (a b : ℕ) : Prop := a / 4 = 2 * a - b
def condition2 (b c : ℕ) : Prop := b / 7 = 2 * b - c
def condition3 (a b c : ℕ) : Prop := a + b + c = 264

-- The main theorem to prove
theorem tortoise_age (a b c : ℕ) (h1 : condition1 a b) (h2 : condition2 b c) (h3 : condition3 a b c) : b = 77 :=
sorry

end tortoise_age_l273_273257


namespace complex_numbers_with_condition_l273_273673

noncomputable def numOfComplexSatisfyingGivenCondition : ℕ :=
  8

theorem complex_numbers_with_condition (z : ℂ) (h1 : abs z < 20) (h2 : complex.exp z = (z - 1) / (z + 1)) :
  ∃ n : ℕ, n = numOfComplexSatisfyingGivenCondition := by
  use 8
  have h : n = 8 := by
    sorry
  exact h

end complex_numbers_with_condition_l273_273673


namespace linear_equation_solution_l273_273743

theorem linear_equation_solution (x : ℝ) (h : 1 - x = -3) : x = 4 :=
by
  sorry

end linear_equation_solution_l273_273743


namespace smallest_integer_represented_as_AA6_and_BB8_l273_273886

def valid_digit_in_base (d : ℕ) (b : ℕ) : Prop := d < b

theorem smallest_integer_represented_as_AA6_and_BB8 :
  ∃ (n : ℕ) (A B : ℕ),
  valid_digit_in_base A 6 ∧ valid_digit_in_base B 8 ∧ 
  n = 7 * A ∧ n = 9 * B ∧ n = 63 :=
by
  sorry

end smallest_integer_represented_as_AA6_and_BB8_l273_273886


namespace maximize_triangle_areas_l273_273998

theorem maximize_triangle_areas (L W : ℝ) (h1 : 2 * L + 2 * W = 80) (h2 : L ≤ 25) : W = 15 :=
by 
  sorry

end maximize_triangle_areas_l273_273998


namespace hannah_total_payment_l273_273139

def washing_machine_cost : ℝ := 100
def dryer_cost : ℝ := washing_machine_cost - 30
def total_cost_before_discount : ℝ := washing_machine_cost + dryer_cost
def discount : ℝ := 0.10
def total_cost_after_discount : ℝ := total_cost_before_discount * (1 - discount)

theorem hannah_total_payment : total_cost_after_discount = 153 := by
  sorry

end hannah_total_payment_l273_273139


namespace geometric_sequence_term_l273_273157

noncomputable def geometric_sequence (a : ℕ → ℤ) (q : ℤ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_term {a : ℕ → ℤ} {q : ℤ}
  (h1 : geometric_sequence a q)
  (h2 : a 7 = 10)
  (h3 : q = -2) :
  a 10 = -80 :=
by
  sorry

end geometric_sequence_term_l273_273157


namespace present_age_of_younger_l273_273863

-- Definition based on conditions
variable (y e : ℕ)
variable (h1 : e = y + 20)
variable (h2 : e - 8 = 5 * (y - 8))

-- Statement to be proven
theorem present_age_of_younger (y e: ℕ) (h1: e = y + 20) (h2: e - 8 = 5 * (y - 8)) : y = 13 := 
by 
  sorry

end present_age_of_younger_l273_273863


namespace probability_opposite_2_l273_273903

-- Define the faces of the dice
def die1_faces : Finset ℕ := {2, 2, 2, 2, 2, 2}
def die2_faces : Finset ℕ := {2, 2, 2, 4, 4, 4}

-- Define probabilities of picking each die
def p_die1 : ℚ := 1 / 2
def p_die2 : ℚ := 1 / 2

-- Probability of observing a 2 on a face of any of the dice
def p_observe_2 : ℚ := (3 / 6) * p_die2 + (6 / 6) * p_die1

-- Probability that the opposite face is 2 given observing 2
def p_opposite_2_given_observe_2 : ℚ := (6 / 9)

theorem probability_opposite_2:
  p_opposite_2_given_observe_2 = 2 / 3 :=
by
  -- Here would be the proof, but we use sorry to skip it.
  sorry

end probability_opposite_2_l273_273903


namespace find_f_2_l273_273123

open Real

noncomputable def f (x : ℝ) : ℝ := sorry

-- The statement to prove: if f is monotonically increasing and satisfies the functional equation
-- for all x, then f(2) = e^2 + 1.
theorem find_f_2
  (h_mono : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2)
  (h_eq : ∀ x : ℝ, f (f x - exp x) = exp 1 + 1) :
  f 2 = exp 2 + 1 := sorry

end find_f_2_l273_273123


namespace factorial_300_zeros_l273_273876

theorem factorial_300_zeros : (∃ n, nat.factorial 300 % 10^(n+1) = 0 ∧ nat.factorial 300 % 10^n ≠ 0) ∧ ∀ n, nat.factorial 300 % 10^(74 + n) ≠ 10^74 + 1 :=
sorry

end factorial_300_zeros_l273_273876


namespace find_k_l273_273376

theorem find_k (m n k : ℤ) (h1 : m = 2 * n + 5) (h2 : m + 2 = 2 * (n + k) + 5) : k = 0 := by
  sorry

end find_k_l273_273376


namespace cyclist_speed_l273_273510

theorem cyclist_speed 
  (v : ℝ) 
  (hiker1_speed : ℝ := 4)
  (hiker2_speed : ℝ := 5)
  (cyclist_overtakes_hiker2_after_hiker1 : ∃ t1 t2 : ℝ, 
      t1 = 8 / (v - hiker1_speed) ∧ 
      t2 = 5 / (v - hiker2_speed) ∧ 
      t2 - t1 = 1/6)
: (v = 20 ∨ v = 7 ∨ abs (v - 6.5) < 0.1) :=
sorry

end cyclist_speed_l273_273510


namespace garden_area_is_correct_l273_273640

def width_of_property : ℕ := 1000
def length_of_property : ℕ := 2250

def width_of_garden : ℕ := width_of_property / 8
def length_of_garden : ℕ := length_of_property / 10

def area_of_garden : ℕ := width_of_garden * length_of_garden

theorem garden_area_is_correct : area_of_garden = 28125 := by
  -- Skipping proof for the purpose of this example
  sorry

end garden_area_is_correct_l273_273640


namespace no_three_digit_number_such_that_sum_is_perfect_square_l273_273583

theorem no_three_digit_number_such_that_sum_is_perfect_square :
  ∀ (a b c : ℕ), a < 10 ∧ b < 10 ∧ c < 10 →
  ¬ (∃ m : ℕ, m * m = 100 * a + 10 * b + c + 100 * b + 10 * c + a + 100 * c + 10 * a + b) := by
  sorry

end no_three_digit_number_such_that_sum_is_perfect_square_l273_273583


namespace limit_example_l273_273456

theorem limit_example (ε : ℝ) (hε : 0 < ε) :
  ∃ δ : ℝ, 0 < δ ∧ 
  (∀ x : ℝ, 0 < |x - 1/2| ∧ |x - 1/2| < δ →
    |((2 * x^2 - 5 * x + 2) / (x - 1/2)) + 3| < ε) :=
sorry -- The proof is not provided

end limit_example_l273_273456


namespace sum_g_equals_half_l273_273838

noncomputable def g (n : ℕ) : ℝ :=
  ∑' k, if k ≥ 3 then 1 / k ^ n else 0

theorem sum_g_equals_half : ∑' n : ℕ, g n.succ = 1 / 2 := 
sorry

end sum_g_equals_half_l273_273838


namespace ball_attendance_l273_273184

noncomputable def num_people_attending_ball (n m : ℕ) : ℕ := n + m 

theorem ball_attendance:
  ∀ (n m : ℕ), 
  n + m < 50 ∧ 
  (n - n / 4) = (5 * m / 7) →
  num_people_attending_ball n m = 41 :=
by 
  intros n m h
  have h1 : n + m < 50 := h.1
  have h2 : n - n / 4 = 5 * m / 7 := h.2
  sorry

end ball_attendance_l273_273184


namespace max_num_pieces_l273_273068

-- Definition of areas
def largeCake_area : ℕ := 21 * 21
def smallPiece_area : ℕ := 3 * 3

-- Problem Statement
theorem max_num_pieces : largeCake_area / smallPiece_area = 49 := by
  sorry

end max_num_pieces_l273_273068


namespace slope_parallel_line_l273_273371

theorem slope_parallel_line (x y : ℝ) (a b c : ℝ) (h : 3 * x - 6 * y = 15) : 
  ∃ m : ℝ, m = 1 / 2 :=
by 
  sorry

end slope_parallel_line_l273_273371


namespace zero_in_set_l273_273667

theorem zero_in_set : 0 ∈ ({0, 1, 2} : Set Nat) := 
sorry

end zero_in_set_l273_273667


namespace num_administrative_personnel_l273_273507

noncomputable def total_employees : ℕ := 280
noncomputable def sample_size : ℕ := 56
noncomputable def ordinary_staff_sample : ℕ := 49

theorem num_administrative_personnel (n : ℕ) (h1 : total_employees = 280) 
(h2 : sample_size = 56) (h3 : ordinary_staff_sample = 49) : 
n = 35 := 
by
  have h_proportion : (sample_size - ordinary_staff_sample) / sample_size = n / total_employees := by sorry
  have h_sol : n = (sample_size - ordinary_staff_sample) * (total_employees / sample_size) := by sorry
  have h_n : n = 35 := by sorry
  exact h_n

end num_administrative_personnel_l273_273507


namespace min_value_x_plus_y_l273_273416

theorem min_value_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 9 / y = 2) : x + y = 8 :=
sorry

end min_value_x_plus_y_l273_273416


namespace function_decreasing_implies_a_range_l273_273471

open Real

-- Given conditions and the question to proof
theorem function_decreasing_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → ((log 1/2 a) ^ x > (log 1/2 a) ^ y)) →
  (1/2 < a ∧ a < 1) :=
begin
  sorry
end

end function_decreasing_implies_a_range_l273_273471


namespace number_of_integers_l273_273537

theorem number_of_integers (n : ℤ) : 
  (20 < n^2) ∧ (n^2 < 150) → 
  (∃ m : ℕ, m = 16) :=
by
  sorry

end number_of_integers_l273_273537


namespace calc_perm_product_l273_273527

-- Define the permutation function
def permutation (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

-- Lean statement to prove the given problem
theorem calc_perm_product : permutation 6 2 * permutation 4 2 = 360 := 
by
  -- Test the calculations if necessary, otherwise use sorry
  sorry

end calc_perm_product_l273_273527


namespace kelly_initial_games_l273_273169

theorem kelly_initial_games :
  ∃ g : ℕ, (g - 15 = 35) ↔ (g = 50) :=
begin
  sorry,
end

end kelly_initial_games_l273_273169


namespace decagon_number_of_triangles_l273_273775

theorem decagon_number_of_triangles : 
  let n := 10 in 
  ∃ k : ℕ, n = 10 ∧ k = nat.choose n 3 ∧ k = 120 :=
sorry

end decagon_number_of_triangles_l273_273775


namespace solve_equation_l273_273719

-- Define the equation as a Lean proposition
def equation (x : ℝ) : Prop :=
  (6 * x + 3) / (3 * x^2 + 6 * x - 9) = 3 * x / (3 * x - 3)

-- Define the solution set
def solution (x : ℝ) : Prop :=
  x = (3 + Real.sqrt 21) / 2 ∨ x = (3 - Real.sqrt 21) / 2

-- Define the condition to avoid division by zero
def valid (x : ℝ) : Prop := x ≠ 1

-- State the theorem
theorem solve_equation (x : ℝ) (h : equation x) (hv : valid x) : solution x :=
by
  sorry

end solve_equation_l273_273719


namespace certain_number_is_84_l273_273867

/-
The least number by which 72 must be multiplied in order to produce a multiple of a certain number is 14.
What is that certain number?
-/

theorem certain_number_is_84 (x : ℕ) (h: 72 * 14 % x = 0 ∧ ∀ y : ℕ, 1 ≤ y → y < 14 → 72 * y % x ≠ 0) : x = 84 :=
sorry

end certain_number_is_84_l273_273867


namespace hyperbola_solution_l273_273663

noncomputable def hyperbola_focus_parabola_equiv_hyperbola : Prop :=
  ∀ (a b c : ℝ),
    -- Condition 1: One focus of the hyperbola coincides with the focus of the parabola y^2 = 4sqrt(7)x
    (c^2 = a^2 + b^2) ∧ (c^2 = 7) →

    -- Condition 2: The hyperbola intersects the line y = x - 1 at points M and N
    (∃ M N : ℝ × ℝ, (M.2 = M.1 - 1) ∧ (N.2 = N.1 - 1) ∧ 
    ((M.1^2 / a^2) - (M.2^2 / b^2) = 1) ∧ ((N.1^2 / a^2) - (N.2^2 / b^2) = 1)) →

    -- Condition 3: The x-coordinate of the midpoint of MN is -2/3
    (∀ M N : ℝ × ℝ, 
    (M.2 = M.1 - 1) ∧ (N.2 = N.1 - 1) ∧ 
    ((M.1 + N.1) / 2 = -2/3)) →

    -- Conclusion: The standard equation of the hyperbola is x^2 / 2 - y^2 / 5 = 1
    a^2 = 2 ∧ b^2 = 5 ∧ (∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1 → (x^2 / 2) - (y^2 / 5) = 1)

-- Proof omitted
theorem hyperbola_solution : hyperbola_focus_parabola_equiv_hyperbola :=
by sorry

end hyperbola_solution_l273_273663


namespace seq_common_max_l273_273261

theorem seq_common_max : ∃ a, a ≤ 250 ∧ 1 ≤ a ∧ a % 8 = 1 ∧ a % 9 = 4 ∧ ∀ b, b ≤ 250 ∧ 1 ≤ b ∧ b % 8 = 1 ∧ b % 9 = 4 → b ≤ a :=
by 
  sorry

end seq_common_max_l273_273261


namespace domain_of_f_l273_273586

theorem domain_of_f :
  (∀ x : ℝ, (0 < 1 - x) ∧ (0 < 3 * x + 1) ↔ ( - (1 / 3 : ℝ) < x ∧ x < 1)) :=
by
  sorry

end domain_of_f_l273_273586


namespace min_value_expression_l273_273961

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) : 
  ∃ c, ∀ x y, 0 < x → 0 < y → x + y = 1 → c = 9 ∧ ((1 / x) + (4 / y)) ≥ 9 := 
sorry

end min_value_expression_l273_273961


namespace length_AB_eight_l273_273686

-- Define parabola and line
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def line (x y k : ℝ) : Prop := y = k * x - k

-- Define intersection points A and B
def intersects (p1 p2 : ℝ × ℝ) (k : ℝ) : Prop :=
  parabola p1.1 p1.2 ∧ line p1.1 p1.2 k ∧
  parabola p2.1 p2.2 ∧ line p2.1 p2.2 k

-- Define midpoint distance condition
def midpoint_condition (p1 p2 : ℝ × ℝ) : Prop :=
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  midpoint.1 = 3

-- The main theorem statement
theorem length_AB_eight (k : ℝ) (A B : ℝ × ℝ) (h1 : intersects A B k)
  (h2 : midpoint_condition A B) : abs ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 64 := 
sorry

end length_AB_eight_l273_273686


namespace time_per_window_l273_273251

-- Definitions of the given conditions
def total_windows : ℕ := 10
def installed_windows : ℕ := 6
def remaining_windows := total_windows - installed_windows
def total_hours : ℕ := 20
def hours_per_window := total_hours / remaining_windows

-- The theorem we need to prove
theorem time_per_window : hours_per_window = 5 := by
  -- This is where the proof would go
  sorry

end time_per_window_l273_273251


namespace point_not_in_third_quadrant_l273_273975

theorem point_not_in_third_quadrant (x y : ℝ) (h : y = -x + 1) : ¬(x < 0 ∧ y < 0) :=
by
  sorry

end point_not_in_third_quadrant_l273_273975


namespace min_moves_to_reassemble_l273_273521

theorem min_moves_to_reassemble (n : ℕ) (h : n > 0) : 
  ∃ m : ℕ, (∀ pieces, pieces = n - 1) ∧ pieces = 1 → move_count = n - 1 :=
by
  sorry

end min_moves_to_reassemble_l273_273521


namespace Weierstrass_theorem_l273_273710

variable {f : ℝ → ℝ} {a b : ℝ}

theorem Weierstrass_theorem (h_continuous : ContinuousOn f (set.Icc a b)) (h_ab : a ≤ b) :
  ∃ x₀ x₁ ∈ set.Icc a b, f x₀ = sup (set.range (λ x : ℝ, if x ∈ set.Icc a b then f x else 0)) ∧
                           f x₁ = inf (set.range (λ x : ℝ, if x ∈ set.Icc a b then f x else 0)) := 
sorry

end Weierstrass_theorem_l273_273710


namespace ball_attendance_l273_273176

variable (n m : ℕ)

def ball_conditions (n m : ℕ) := 
  n + m < 50 ∧ 
  4 ∣ 3 * n ∧ 
  7 ∣ 5 * m

theorem ball_attendance (n m : ℕ) (h : ball_conditions n m) : 
  n + m = 41 :=
sorry

end ball_attendance_l273_273176


namespace bin_rep_23_l273_273791

theorem bin_rep_23 : Nat.binary_repr 23 = "10111" :=
by
  sorry

end bin_rep_23_l273_273791


namespace polygon_sides_l273_273029

theorem polygon_sides (n : ℕ) :
  ((n - 2) * 180 = 4 * 360) → n = 10 :=
by 
  sorry

end polygon_sides_l273_273029


namespace ball_total_attendance_l273_273187

theorem ball_total_attendance :
    ∃ n m : ℕ, n + m = 41 ∧
    (n + m < 50) ∧
    (3 * n / 4 = (5 * m / 7)) :=
begin
  sorry
end

end ball_total_attendance_l273_273187


namespace larry_initial_money_l273_273639

theorem larry_initial_money
  (M : ℝ)
  (spent_maintenance : ℝ := 0.04 * M)
  (saved_for_emergencies : ℝ := 0.30 * M)
  (snack_cost : ℝ := 5)
  (souvenir_cost : ℝ := 25)
  (lunch_cost : ℝ := 12)
  (loan_cost : ℝ := 10)
  (remaining_money : ℝ := 368)
  (total_spent : ℝ := snack_cost + souvenir_cost + lunch_cost + loan_cost) :
  M - spent_maintenance - saved_for_emergencies - total_spent = remaining_money →
  M = 636.36 :=
by
  sorry

end larry_initial_money_l273_273639


namespace evaluate_fraction_l273_273939

theorem evaluate_fraction (a b : ℕ) (h₁ : a = 250) (h₂ : b = 240) :
  1800^2 / (a^2 - b^2) = 660 :=
by 
  sorry

end evaluate_fraction_l273_273939


namespace april_total_earned_l273_273768

variable (r_price t_price d_price : ℕ)
variable (r_sold t_sold d_sold : ℕ)
variable (r_total t_total d_total : ℕ)

-- Define prices
def rose_price : ℕ := 4
def tulip_price : ℕ := 3
def daisy_price : ℕ := 2

-- Define quantities sold
def roses_sold : ℕ := 9
def tulips_sold : ℕ := 6
def daisies_sold : ℕ := 12

-- Define total money earned for each type of flower
def rose_total := roses_sold * rose_price
def tulip_total := tulips_sold * tulip_price
def daisy_total := daisies_sold * daisy_price

-- Define total money earned
def total_earned := rose_total + tulip_total + daisy_total

-- Statement to prove
theorem april_total_earned : total_earned = 78 :=
by sorry

end april_total_earned_l273_273768


namespace desired_average_sale_l273_273509

def s1 := 2500
def s2 := 4000
def s3 := 3540
def s4 := 1520
def avg := 2890

theorem desired_average_sale : (s1 + s2 + s3 + s4) / 4 = avg := by
  sorry

end desired_average_sale_l273_273509


namespace ball_attendance_l273_273172

theorem ball_attendance
  (n m : ℕ)
  (h_total : n + m < 50)
  (h_ladies_dance : 3 * n = 4 * (Ladies who danced invited)
  (h_gentlemen_invited : 5 * m = 7 * (Gentlemen who invited)
  (h_dance_pairs : 3 * n = 20 * m) :
  n + m = 41 :=
by sorry

end ball_attendance_l273_273172


namespace mark_cans_l273_273622

variable (r j m : ℕ) -- r for Rachel, j for Jaydon, m for Mark

theorem mark_cans (r j m : ℕ) 
  (h1 : j = 5 + 2 * r)
  (h2 : m = 4 * j)
  (h3 : r + j + m = 135) : 
  m = 100 :=
by
  sorry

end mark_cans_l273_273622


namespace depth_of_grass_sheet_l273_273217

-- Given conditions
def playground_area : ℝ := 5900
def grass_cost_per_cubic_meter : ℝ := 2.80
def total_cost : ℝ := 165.2

-- Variable to solve for
variable (d : ℝ)

-- Theorem statement
theorem depth_of_grass_sheet
  (h : total_cost = (playground_area * d) * grass_cost_per_cubic_meter) :
  d = 0.01 :=
by
  sorry

end depth_of_grass_sheet_l273_273217


namespace Kelly_initial_games_l273_273168

-- Condition definitions
variable (give_away : ℕ) (left_over : ℕ)
variable (initial_games : ℕ)

-- Given conditions
axiom h1 : give_away = 15
axiom h2 : left_over = 35

-- Proof statement
theorem Kelly_initial_games : initial_games = give_away + left_over :=
sorry

end Kelly_initial_games_l273_273168


namespace pirates_total_coins_l273_273581

theorem pirates_total_coins (x : ℕ) (h : (x * (x + 1)) / 2 = 5 * x) : 6 * x = 54 := by
  -- The proof will go here, but it's currently omitted with 'sorry'
  sorry

end pirates_total_coins_l273_273581


namespace parabolic_arch_properties_l273_273629

noncomputable def parabolic_arch_height (x : ℝ) : ℝ :=
  let a : ℝ := -4 / 125
  let k : ℝ := 20
  a * x^2 + k

theorem parabolic_arch_properties :
  (parabolic_arch_height 10 = 16.8) ∧ (parabolic_arch_height 10 = parabolic_arch_height 10 → (10 = 10 ∨ 10 = -10)) :=
by
  have h1 : parabolic_arch_height 10 = 16.8 :=
    sorry
  have h2 : parabolic_arch_height 10 = parabolic_arch_height 10 → (10 = 10 ∨ 10 = -10) :=
    sorry
  exact ⟨h1, h2⟩

end parabolic_arch_properties_l273_273629


namespace miranda_pillows_l273_273704

theorem miranda_pillows (feathers_per_pound : ℕ) (total_feathers : ℕ) (pillows : ℕ)
  (h1 : feathers_per_pound = 300) (h2 : total_feathers = 3600) (h3 : pillows = 6) :
  (total_feathers / feathers_per_pound) / pillows = 2 := by
  sorry

end miranda_pillows_l273_273704


namespace fraction_meaningful_iff_l273_273683

theorem fraction_meaningful_iff (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 1)) ↔ x ≠ 1 := sorry

end fraction_meaningful_iff_l273_273683


namespace andrew_total_payment_l273_273097

-- Given conditions
def quantity_of_grapes := 14
def rate_per_kg_grapes := 54
def quantity_of_mangoes := 10
def rate_per_kg_mangoes := 62

-- Calculations
def cost_of_grapes := quantity_of_grapes * rate_per_kg_grapes
def cost_of_mangoes := quantity_of_mangoes * rate_per_kg_mangoes
def total_amount_paid := cost_of_grapes + cost_of_mangoes

-- Theorem to prove
theorem andrew_total_payment : total_amount_paid = 1376 := by
  sorry

end andrew_total_payment_l273_273097


namespace ball_attendance_l273_273174

variable (n m : ℕ)

def ball_conditions (n m : ℕ) := 
  n + m < 50 ∧ 
  4 ∣ 3 * n ∧ 
  7 ∣ 5 * m

theorem ball_attendance (n m : ℕ) (h : ball_conditions n m) : 
  n + m = 41 :=
sorry

end ball_attendance_l273_273174


namespace value_of_4_Y_3_eq_neg23_l273_273528

def my_operation (a b : ℝ) (c : ℝ) : ℝ := a^2 - 2 * a * b * c + b^2

theorem value_of_4_Y_3_eq_neg23 : my_operation 4 3 2 = -23 := by
  sorry

end value_of_4_Y_3_eq_neg23_l273_273528


namespace geometric_ratio_l273_273480

theorem geometric_ratio (a₁ q : ℝ) (h₀ : a₁ ≠ 0) (h₁ : a₁ + a₁ * q + a₁ * q^2 = 3 * a₁) : q = -2 ∨ q = 1 :=
by
  sorry

end geometric_ratio_l273_273480


namespace edward_mowed_lawns_l273_273116

theorem edward_mowed_lawns (L : ℕ) (h1 : 8 * L + 7 = 47) : L = 5 :=
by
  sorry

end edward_mowed_lawns_l273_273116


namespace Misha_probability_l273_273071

open Probability

-- Definitions
def classesMonday := 5
def classesTuesday := 6
def totalClasses := 11
def totalCorrect := 7
def mondayCorrect := 3

-- Calculating binomial probabilities
noncomputable def P_A1 := (binomial classesMonday mondayCorrect) * (1 / 2) ^ classesMonday
noncomputable def P_A2 := (binomial classesTuesday (totalCorrect - mondayCorrect)) * (1 / 2) ^ classesTuesday
noncomputable def P_B := (binomial totalClasses totalCorrect) * (1 / 2) ^ totalClasses

-- Theorem stating the required probability
theorem Misha_probability :
  P_A1 * P_A2 / P_B = 5 / 11 :=
by
  sorry

end Misha_probability_l273_273071


namespace Misha_probability_l273_273070

open Probability

-- Definitions
def classesMonday := 5
def classesTuesday := 6
def totalClasses := 11
def totalCorrect := 7
def mondayCorrect := 3

-- Calculating binomial probabilities
noncomputable def P_A1 := (binomial classesMonday mondayCorrect) * (1 / 2) ^ classesMonday
noncomputable def P_A2 := (binomial classesTuesday (totalCorrect - mondayCorrect)) * (1 / 2) ^ classesTuesday
noncomputable def P_B := (binomial totalClasses totalCorrect) * (1 / 2) ^ totalClasses

-- Theorem stating the required probability
theorem Misha_probability :
  P_A1 * P_A2 / P_B = 5 / 11 :=
by
  sorry

end Misha_probability_l273_273070


namespace number_of_crocodiles_l273_273563

theorem number_of_crocodiles
  (f : ℕ) -- number of frogs
  (c : ℕ) -- number of crocodiles
  (total_eyes : ℕ) -- total number of eyes
  (frog_eyes : ℕ) -- number of eyes per frog
  (croc_eyes : ℕ) -- number of eyes per crocodile
  (h_f : f = 20) -- condition: there are 20 frogs
  (h_total_eyes : total_eyes = 52) -- condition: total number of eyes is 52
  (h_frog_eyes : frog_eyes = 2) -- condition: each frog has 2 eyes
  (h_croc_eyes : croc_eyes = 2) -- condition: each crocodile has 2 eyes
  :
  c = 6 := -- proof goal: number of crocodiles is 6
by
  sorry

end number_of_crocodiles_l273_273563


namespace domain_of_function_l273_273275

def domain_condition_1 (x : ℝ) : Prop := 1 - x > 0
def domain_condition_2 (x : ℝ) : Prop := x + 3 ≥ 0

def in_domain (x : ℝ) : Prop := domain_condition_1 x ∧ domain_condition_2 x

theorem domain_of_function : ∀ x : ℝ, in_domain x ↔ (-3 : ℝ) ≤ x ∧ x < 1 := 
by sorry

end domain_of_function_l273_273275


namespace find_c_k_l273_273355

noncomputable def a_n (n d : ℕ) := 1 + (n - 1) * d
noncomputable def b_n (n r : ℕ) := r ^ (n - 1)
noncomputable def c_n (n d r : ℕ) := a_n n d + b_n n r

theorem find_c_k (d r k : ℕ) (hd1 : c_n (k - 1) d r = 200) (hd2 : c_n (k + 1) d r = 2000) :
  c_n k d r = 423 :=
sorry

end find_c_k_l273_273355


namespace john_unanswered_questions_l273_273164

theorem john_unanswered_questions :
  ∃ (c w u : ℕ), (30 + 4 * c - w = 84) ∧ (5 * c + 2 * u = 93) ∧ (c + w + u = 30) ∧ (u = 9) :=
by
  sorry

end john_unanswered_questions_l273_273164


namespace probability_entire_grid_black_l273_273977

-- Definitions of the problem in terms of conditions.
def grid_size : Nat := 4

def prob_black_initial : ℚ := 1 / 2

def middle_squares : List (Nat × Nat) := [(2, 2), (2, 3), (3, 2), (3, 3)]

def edge_squares : List (Nat × Nat) := 
  [ (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 0), (1, 3),
    (2, 0), (2, 3),
    (3, 0), (3, 1), (3, 2), (3, 3) ]

-- The probability that each of these squares is black independently.
def prob_all_middle_black : ℚ := (1 / 2) ^ 4

def prob_all_edge_black : ℚ := (1 / 2) ^ 12

-- The combined probability that the entire grid is black.
def prob_grid_black := prob_all_middle_black * prob_all_edge_black

-- Statement of the proof problem.
theorem probability_entire_grid_black :
  prob_grid_black = 1 / 65536 := by
  sorry

end probability_entire_grid_black_l273_273977


namespace dodecagon_diagonals_eq_54_dodecagon_triangles_eq_220_l273_273099

-- Define a regular dodecagon
def dodecagon_sides : ℕ := 12

-- Prove that the number of diagonals in a regular dodecagon is 54
theorem dodecagon_diagonals_eq_54 : (dodecagon_sides * (dodecagon_sides - 3)) / 2 = 54 :=
by sorry

-- Prove that the number of possible triangles formed from a regular dodecagon vertices is 220
theorem dodecagon_triangles_eq_220 : Nat.choose dodecagon_sides 3 = 220 :=
by sorry

end dodecagon_diagonals_eq_54_dodecagon_triangles_eq_220_l273_273099


namespace average_of_ABC_l273_273736

theorem average_of_ABC (A B C : ℤ)
  (h1 : 101 * C - 202 * A = 404)
  (h2 : 101 * B + 303 * A = 505)
  (h3 : 101 * A + 101 * B + 101 * C = 303) :
  (A + B + C) / 3 = 3 :=
by
  sorry

end average_of_ABC_l273_273736


namespace matrix_inverse_eq_l273_273015

theorem matrix_inverse_eq (d k : ℚ) (A : Matrix (Fin 2) (Fin 2) ℚ) 
  (hA : A = ![![1, 4], ![6, d]]) 
  (hA_inv : A⁻¹ = k • A) :
  (d, k) = (-1, 1/25) :=
  sorry

end matrix_inverse_eq_l273_273015


namespace fraction_of_decimal_l273_273020

theorem fraction_of_decimal (a b : ℕ) (h : 0.375 = (a : ℝ) / (b : ℝ)) (gcd_ab : Nat.gcd a b = 1) : a + b = 11 :=
  sorry

end fraction_of_decimal_l273_273020


namespace probability_coprime_l273_273543

open Nat

/-- Define the set of integers from 2 to 8 -/
def S : Finset ℕ := {2, 3, 4, 5, 6, 7, 8}

/-- Define the pairs of integers from the set S -/
def pairs := Finset.filter (λ p: ℕ × ℕ, p.1 < p.2) (S.product S)

/-- Define coprime pairs from the set pairs -/
def coprime_pairs := pairs.filter (λ p, gcd p.1 p.2 = 1)

/-- The probability that two randomly selected numbers from S are coprime is 2/3 -/
theorem probability_coprime : 
  (coprime_pairs.card : ℚ) / (pairs.card : ℚ) = 2 / 3 := 
sorry

end probability_coprime_l273_273543


namespace find_smallest_x_l273_273796

theorem find_smallest_x :
  ∃ x : ℕ, x > 0 ∧
  (45 * x + 9) % 25 = 3 ∧
  (2 * x) % 5 = 8 ∧
  x = 20 :=
by
  sorry

end find_smallest_x_l273_273796


namespace compound_interest_semiannual_l273_273504

theorem compound_interest_semiannual
  (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ)
  (initial_amount : P = 900)
  (interest_rate : r = 0.10)
  (compounding_periods : n = 2)
  (time_period : t = 1) :
  P * (1 + r / n) ^ (n * t) = 992.25 :=
by
  sorry

end compound_interest_semiannual_l273_273504


namespace remainder_when_expr_divided_by_9_l273_273337

theorem remainder_when_expr_divided_by_9 (n m p : ℤ)
  (h1 : n % 18 = 10)
  (h2 : m % 27 = 16)
  (h3 : p % 6 = 4) :
  (2 * n + 3 * m - p) % 9 = 1 := 
sorry

end remainder_when_expr_divided_by_9_l273_273337


namespace fraction_equality_solution_l273_273945

theorem fraction_equality_solution (x : ℝ) : (5 + x) / (7 + x) = (2 + x) / (3 + x) → x = 1 :=
by
  intro h
  sorry

end fraction_equality_solution_l273_273945


namespace find_angle_A_l273_273160

theorem find_angle_A
  (A B C : ℝ)
  (h1 : B = A + 10)
  (h2 : C = B + 10)
  (h3 : A + B + C = 180) : A = 50 :=
by 
  sorry

end find_angle_A_l273_273160


namespace john_cards_sum_l273_273329

theorem john_cards_sum :
  ∃ (g : ℕ → ℕ) (y : ℕ → ℕ),
    (∀ n, (g n) ∈ [1, 2, 3, 4, 5]) ∧
    (∀ n, (y n) ∈ [2, 3, 4, 5]) ∧
    (∀ n, (g n < g (n + 1))) ∧
    (∀ n, (y n < y (n + 1))) ∧
    (∀ n, (g n ∣ y (n + 1) ∨ y (n + 1) ∣ g n)) ∧
    (g 0 = 1 ∧ g 2 = 2 ∧ g 4 = 5) ∧
    ( y 1 = 2 ∧ y 3 = 3 ∧ y 5 = 4 ) →
  g 0 + g 2 + g 4 = 8 := by
sorry

end john_cards_sum_l273_273329


namespace exponent_property_l273_273403

theorem exponent_property (a b : ℕ) : (a * b^2)^3 = a^3 * b^6 :=
by sorry

end exponent_property_l273_273403


namespace interval_of_decrease_l273_273781

noncomputable def f (x : ℝ) := x * Real.exp x + 1

theorem interval_of_decrease : {x : ℝ | x < -1} = {x : ℝ | (x + 1) * Real.exp x < 0} :=
by
  sorry

end interval_of_decrease_l273_273781


namespace square_binomial_constant_l273_273013

theorem square_binomial_constant (y : ℝ) : ∃ b : ℝ, (y^2 + 12*y + 50 = (y + 6)^2 + b) ∧ b = 14 := 
by
  sorry

end square_binomial_constant_l273_273013


namespace sufficient_condition_hyperbola_l273_273551

theorem sufficient_condition_hyperbola (m : ℝ) (h : 5 < m) : 
  ∃ a b : ℝ, (a > 0) ∧ (b < 0) ∧ (∀ x y : ℝ, (x^2)/(a) + (y^2)/(b) = 1) := 
sorry

end sufficient_condition_hyperbola_l273_273551


namespace ball_attendance_l273_273182

noncomputable def num_people_attending_ball (n m : ℕ) : ℕ := n + m 

theorem ball_attendance:
  ∀ (n m : ℕ), 
  n + m < 50 ∧ 
  (n - n / 4) = (5 * m / 7) →
  num_people_attending_ball n m = 41 :=
by 
  intros n m h
  have h1 : n + m < 50 := h.1
  have h2 : n - n / 4 = 5 * m / 7 := h.2
  sorry

end ball_attendance_l273_273182


namespace Misha_probability_l273_273072

open Probability

-- Definitions
def classesMonday := 5
def classesTuesday := 6
def totalClasses := 11
def totalCorrect := 7
def mondayCorrect := 3

-- Calculating binomial probabilities
noncomputable def P_A1 := (binomial classesMonday mondayCorrect) * (1 / 2) ^ classesMonday
noncomputable def P_A2 := (binomial classesTuesday (totalCorrect - mondayCorrect)) * (1 / 2) ^ classesTuesday
noncomputable def P_B := (binomial totalClasses totalCorrect) * (1 / 2) ^ totalClasses

-- Theorem stating the required probability
theorem Misha_probability :
  P_A1 * P_A2 / P_B = 5 / 11 :=
by
  sorry

end Misha_probability_l273_273072


namespace find_time_l273_273734

theorem find_time (s z t : ℝ) (h : ∀ s, 0 ≤ s ∧ s ≤ 7 → z = s^2 + 2 * s) : 
  z = 35 ∧ z = t^2 + 2 * t + 20 → 0 ≤ t ∧ t ≤ 7 → t = 3 :=
by
  sorry

end find_time_l273_273734


namespace ball_total_attendance_l273_273189

theorem ball_total_attendance :
    ∃ n m : ℕ, n + m = 41 ∧
    (n + m < 50) ∧
    (3 * n / 4 = (5 * m / 7)) :=
begin
  sorry
end

end ball_total_attendance_l273_273189


namespace problem1_problem2_l273_273307

noncomputable def f (a x : ℝ) : ℝ :=
  if x < a then 2 * a - (x + 4 / x)
  else x - 4 / x

theorem problem1 (h : ∀ x : ℝ, f 1 x = 3 → x = 4) : ∃ x : ℝ, f 1 x = 3 ∧ x = 4 :=
sorry

theorem problem2 (h : ∀ x1 x2 x3 : ℝ, 
  (x1 < x2 ∧ x2 < x3 ∧ x2 - x1 = x3 - x2) →
  f a x1 = 3 ∧ f a x2 = 3 ∧ f a x3 = 3 ∧ a ≤ -1 → 
  a = -11 / 6) : ∃ a : ℝ, a ≤ -1 ∧ (∃ x1 x2 x3 : ℝ, 
  (x1 < x2 ∧ x2 < x3 ∧ x2 - x1 = x3 - x2) ∧ 
  f a x1 = 3 ∧ f a x2 = 3 ∧ f a x3 = 3 ∧ a = -11 / 6) :=
sorry

end problem1_problem2_l273_273307


namespace xiao_ming_cube_division_l273_273601

theorem xiao_ming_cube_division (large_edge small_cubes : ℕ)
  (large_edge_eq : large_edge = 4)
  (small_cubes_eq : small_cubes = 29)
  (total_volume : large_edge ^ 3 = 64) :
  ∃ (small_edge_1_cube : ℕ), small_edge_1_cube = 24 ∧ small_cubes = 29 ∧ 
  small_edge_1_cube + (small_cubes - small_edge_1_cube) * 8 = 64 := 
by
  -- We only need to assert the existence here as per the instruction.
  sorry

end xiao_ming_cube_division_l273_273601


namespace range_of_a_l273_273411

theorem range_of_a (a : ℝ) (h : a < 1) : ∀ x : ℝ, |x - 4| + |x - 5| > a :=
by
  sorry

end range_of_a_l273_273411


namespace range_of_a_l273_273294

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 - 4 * x + 3 < 0 ∧ 2^(1 - x) + a ≤ 0 ∧ x^2 - 2 * (a + 7) * x + 5 ≤ 0 ) ↔ (-4 ≤ a ∧ a ≤ -1) :=
by
  sorry

end range_of_a_l273_273294


namespace range_of_m_l273_273419

variable (m t : ℝ)

namespace proof_problem

def proposition_p : Prop :=
  ∀ x y : ℝ, (x^2 / (t + 2) + y^2 / (t - 10) = 1) → (t + 2) * (t - 10) < 0

def proposition_q (m : ℝ) : Prop :=
  -m < t ∧ t < m + 1 ∧ m > 0

theorem range_of_m :
  (∃ t, proposition_q m t) → proposition_p t → 0 < m ∧ m ≤ 2 := by
  sorry

end proof_problem

end range_of_m_l273_273419


namespace sphere_radius_l273_273964

theorem sphere_radius (R : ℝ) (h : 4 * Real.pi * R^2 = 4 * Real.pi) : R = 1 :=
by
  sorry

end sphere_radius_l273_273964


namespace find_number_l273_273783

theorem find_number (n : ℕ) (h : n / 3 = 10) : n = 30 := by
  sorry

end find_number_l273_273783


namespace negation_of_proposition_l273_273224

theorem negation_of_proposition:
  (¬ ∃ x : ℝ, x^2 - 2 * x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 - 2 * x + 1 > 0) :=
by
  sorry

end negation_of_proposition_l273_273224


namespace problem1_tangent_line_eq_problem2_range_of_a_l273_273309

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := (1 / x) - a

-- Define the specific case of the tangent line problem at a = 2 and point (1, f(1))
theorem problem1_tangent_line_eq (x y : ℝ) (a : ℝ) (h : a = 2) (hx : x = 1) :
  f 2 1 = -2 ∧ f' 2 1 = -1 → (x + y + 1 = 0) :=
begin
  sorry
end

-- Define the condition where f(x) < 0 for all x in (0, +∞),
-- and find the range of values for a
theorem problem2_range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x → x < +∞ → f a x < 0) → a > (1 / Real.exp 1) :=
begin
  sorry
end

end problem1_tangent_line_eq_problem2_range_of_a_l273_273309


namespace complex_solutions_count_eq_4_l273_273674

noncomputable def solution_count : ℕ :=
4

theorem complex_solutions_count_eq_4 :
  ∃ (z : ℂ), |z| < 20 ∧ ∀ (z : ℂ), (|z| < 20) → (exp z = (z - 1) / (z + 1)) → z ∈ ({z : ℂ | |z| < 20 ∧ exp z = (z - 1) / (z + 1)} : set ℂ) ∧ solution_count = 4 :=
by {
  sorry
}

end complex_solutions_count_eq_4_l273_273674


namespace mutually_exclusive_events_l273_273664

structure Event (α : Type) :=
  (elems : set α)

def mutually_exclusive {α : Type} (e1 e2 : Event α) : Prop :=
  e1.elems ∩ e2.elems = ∅

structure Shooting :=
  (hit_7 : Event ℕ)
  (hit_8 : Event ℕ)

structure PeopleShoots :=
  (hit_target : Event ℕ)
  (a_hits_b_misses : Event ℕ)

structure Balls :=
  (at_least_one_black : Event ℕ)
  (both_red : Event ℕ)
  (no_black : Event ℕ)
  (exactly_one_red : Event ℕ)

def problem_conditions : Prop :=
  ∃ (shooting : Shooting) (people_shoots : PeopleShoots) (balls : Balls),
    mutually_exclusive shooting.hit_7 shooting.hit_8 ∧
    ¬ mutually_exclusive people_shoots.hit_target people_shoots.a_hits_b_misses ∧
    mutually_exclusive balls.at_least_one_black balls.both_red ∧
    mutually_exclusive balls.no_black balls.exactly_one_red

theorem mutually_exclusive_events : problem_conditions :=
by sorry

end mutually_exclusive_events_l273_273664


namespace ternary_to_decimal_l273_273111

theorem ternary_to_decimal (n : ℕ) (h : n = 121) : 
  (1 * 3^2 + 2 * 3^1 + 1 * 3^0) = 16 :=
by sorry

end ternary_to_decimal_l273_273111


namespace range_of_a_l273_273866

noncomputable def f (a x : ℝ) : ℝ := (2 - a^2) * x + a

theorem range_of_a (a : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 1 → f a x > 0) ↔ (0 < a ∧ a < 2) :=
by
  sorry

end range_of_a_l273_273866


namespace tan_alpha_l273_273313

variable (α : ℝ)

theorem tan_alpha (h₁ : Real.sin α = -5/13) (h₂ : 0 < α ∧ α < 2 * Real.pi ∧ α > 3 * Real.pi / 2) :
  Real.tan α = -5/12 :=
sorry

end tan_alpha_l273_273313


namespace correct_statement_l273_273523

theorem correct_statement : ∀ (a b : ℝ), ((a ≠ b ∧ ¬(a < b ∧ ∀ x, (x = a ∨ x = b) → x = a ∨ x = b)) ∧
                                            ¬(∀ p q : ℝ, p = q → p = q) ∧
                                            ¬(∀ a : ℝ, |a| = -a → a < 0) ∧
                                            ¬(∀ a b : ℝ, (a ≠ 0 ∧ b ≠ 0 ∧ (a = -b)) → (a / b = -1))) :=
by sorry

-- Explanation of conditions:
-- a  ≠ b ensures two distinct points
-- ¬(a < b ∧ ∀ x, (x = a ∨ x = b) → x is between a and b) incorrectly rephrased as shortest distance as a line segment
-- ¬(∀ p q : ℝ, p = q → p = q) is not directly used, a minimum to refute the concept as required.
-- |a| = -a → a < 0 reinterpreted as a ≤ 0 but incorrectly stated as < 0 explicitly refuted
-- ¬(∀ a b : ℝ, a ≠ 0 and/or b ≠ 0 maintained where a / b not strictly required/misinterpreted)

end correct_statement_l273_273523


namespace largest_partner_share_l273_273542

-- Definitions for the conditions
def total_profit : ℕ := 48000
def ratio_parts : List ℕ := [2, 4, 5, 3, 6]
def total_ratio_parts : ℕ := ratio_parts.sum
def value_per_part : ℕ := total_profit / total_ratio_parts
def largest_share : ℕ := 6 * value_per_part

-- Statement of the proof problem
theorem largest_partner_share : largest_share = 14400 := by
  -- Insert proof here
  sorry

end largest_partner_share_l273_273542


namespace proof_second_number_is_30_l273_273592

noncomputable def second_number_is_30 : Prop :=
  ∃ (a b c : ℕ), 
    a + b + c = 98 ∧ 
    (a / (gcd a b) = 2) ∧ (b / (gcd a b) = 3) ∧
    (b / (gcd b c) = 5) ∧ (c / (gcd b c) = 8) ∧
    b = 30

theorem proof_second_number_is_30 : second_number_is_30 :=
  sorry

end proof_second_number_is_30_l273_273592


namespace letters_in_small_envelopes_l273_273262

theorem letters_in_small_envelopes (total_letters : ℕ) (large_envelopes : ℕ) (letters_per_large_envelope : ℕ) (letters_in_small_envelopes : ℕ) :
  total_letters = 80 →
  large_envelopes = 30 →
  letters_per_large_envelope = 2 →
  letters_in_small_envelopes = total_letters - (large_envelopes * letters_per_large_envelope) →
  letters_in_small_envelopes = 20 :=
by
  intros ht hl he hs
  rw [ht, hl, he] at hs
  exact hs

#check letters_in_small_envelopes

end letters_in_small_envelopes_l273_273262


namespace soft_drink_cost_l273_273922

/-- Benny bought 2 soft drinks for a certain price each and 5 candy bars.
    He spent a total of $28. Each candy bar cost $4. 
    Prove that the cost of each soft drink was $4.
--/
theorem soft_drink_cost (S : ℝ) (h1 : 2 * S + 5 * 4 = 28) : S = 4 := 
by
  sorry

end soft_drink_cost_l273_273922


namespace compute_expression_l273_273402

theorem compute_expression :
  (5 + 7)^2 + 5^2 + 7^2 = 218 :=
by
  sorry

end compute_expression_l273_273402


namespace remainder_when_divided_by_13_l273_273896

theorem remainder_when_divided_by_13 (N : ℤ) (k : ℤ) (h : N = 39 * k + 17) : 
  N % 13 = 4 :=
by
  sorry

end remainder_when_divided_by_13_l273_273896


namespace roots_expression_l273_273700

theorem roots_expression (p q : ℝ) (hpq : (∀ x, 3*x^2 + 9*x - 21 = 0 → x = p ∨ x = q)) 
  (sum_roots : p + q = -3) 
  (prod_roots : p * q = -7) : (3*p - 4) * (6*q - 8) = 122 :=
by
  sorry

end roots_expression_l273_273700


namespace first_place_points_l273_273435

-- Definitions for the conditions
def num_teams : Nat := 4
def points_win : Nat := 2
def points_draw : Nat := 1
def points_loss : Nat := 0

def games_played (n : Nat) : Nat :=
  let pairs := n * (n - 1) / 2  -- Binomial coefficient C(n, 2)
  2 * pairs  -- Each pair plays twice

def total_points_distributed (n : Nat) (points_per_game : Nat) : Nat :=
  (games_played n) * points_per_game

def last_place_points : Nat := 5

-- The theorem to prove
theorem first_place_points : ∃ a b c : Nat, a + b + c = total_points_distributed num_teams points_win - last_place_points ∧ (a = 7 ∨ b = 7 ∨ c = 7) :=
by
  sorry

end first_place_points_l273_273435


namespace instantaneous_velocity_at_t_5_l273_273872

noncomputable def s (t : ℝ) : ℝ := (1/4) * t^4 - 3

theorem instantaneous_velocity_at_t_5 : 
  (deriv s 5) = 125 :=
by
  sorry

end instantaneous_velocity_at_t_5_l273_273872


namespace top_and_bottom_area_each_l273_273196

def long_side_area : ℕ := 2 * 8 * 6
def short_side_area : ℕ := 2 * 5 * 6
def total_sides_area : ℕ := long_side_area + short_side_area
def total_needed_area : ℕ := 236
def top_and_bottom_area : ℕ := total_needed_area - total_sides_area

theorem top_and_bottom_area_each :
  top_and_bottom_area / 2 = 40 := by
  sorry

end top_and_bottom_area_each_l273_273196


namespace solve_system_l273_273214

theorem solve_system :
  ∃ (x y : ℝ), (x^2 + y^2 ≤ 1) ∧ (x^4 - 18 * x^2 * y^2 + 81 * y^4 - 20 * x^2 - 180 * y^2 + 100 = 0) ∧
    ((x = -1 / Real.sqrt 10 ∧ y = 3 / Real.sqrt 10) ∨
    (x = -1 / Real.sqrt 10 ∧ y = -3 / Real.sqrt 10) ∨
    (x = 1 / Real.sqrt 10 ∧ y = 3 / Real.sqrt 10) ∨
    (x = 1 / Real.sqrt 10 ∧ y = -3 / Real.sqrt 10)) :=
  by
  sorry

end solve_system_l273_273214


namespace walking_total_distance_l273_273628

theorem walking_total_distance :
  let t1 := 1    -- first hour on level ground
  let t2 := 0.5  -- next 0.5 hour on level ground
  let t3 := 0.75 -- 45 minutes uphill
  let t4 := 0.5  -- 30 minutes uphill
  let t5 := 0.5  -- 30 minutes downhill
  let t6 := 0.25 -- 15 minutes downhill
  let t7 := 1.5  -- 1.5 hours on level ground
  let t8 := 0.75 -- 45 minutes on level ground
  let s1 := 4    -- speed for t1 (4 km/hr)
  let s2 := 5    -- speed for t2 (5 km/hr)
  let s3 := 3    -- speed for t3 (3 km/hr)
  let s4 := 2    -- speed for t4 (2 km/hr)
  let s5 := 6    -- speed for t5 (6 km/hr)
  let s6 := 7    -- speed for t6 (7 km/hr)
  let s7 := 4    -- speed for t7 (4 km/hr)
  let s8 := 6    -- speed for t8 (6 km/hr)
  s1 * t1 + s2 * t2 + s3 * t3 + s4 * t4 + s5 * t5 + s6 * t6 + s7 * t7 + s8 * t8 = 25 :=
by sorry

end walking_total_distance_l273_273628


namespace find_a_and_b_nth_equation_conjecture_l273_273197

theorem find_a_and_b {a b : ℤ} (h1 : 1^2 + 2^2 - 3^2 = 1 * a - b)
                                        (h2 : 2^2 + 3^2 - 4^2 = 2 * 0 - b)
                                        (h3 : 3^2 + 4^2 - 5^2 = 3 * 1 - b)
                                        (h4 : 4^2 + 5^2 - 6^2 = 4 * 2 - b):
    a = -1 ∧ b = 3 :=
    sorry

theorem nth_equation_conjecture (n : ℤ) :
  n^2 + (n+1)^2 - (n+2)^2 = n * (n-2) - 3 :=
  sorry

end find_a_and_b_nth_equation_conjecture_l273_273197


namespace division_multiplication_result_l273_273645

theorem division_multiplication_result : (180 / 6) * 3 = 90 := by
  sorry

end division_multiplication_result_l273_273645


namespace product_of_undefined_roots_l273_273285

theorem product_of_undefined_roots :
  let f (x : ℝ) := (x^2 - 4*x + 4) / (x^2 - 5*x + 6)
  ∀ x : ℝ, (x^2 - 5*x + 6 = 0) → x = 2 ∨ x = 3 →
  (x = 2 ∨ x = 3 → x1 = 2 ∧ x2 = 3 → x1 * x2 = 6) :=
by
  sorry

end product_of_undefined_roots_l273_273285


namespace quadratic_real_roots_l273_273797

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, (k - 1) * x^2 + 2 * k * x + (k - 3) = 0) ↔ k ≥ 3 / 4 :=
by sorry

end quadratic_real_roots_l273_273797


namespace rectangle_area_l273_273091

theorem rectangle_area (a b c: ℝ) (h₁ : a = 7.1) (h₂ : b = 8.9) (h₃ : c = 10.0) (L W: ℝ)
  (h₄ : L = 2 * W) (h₅ : 2 * (L + W) = a + b + c) : L * W = 37.54 :=
by
  sorry

end rectangle_area_l273_273091


namespace carol_first_to_roll_six_l273_273522

def probability_roll (x : ℕ) (success : ℕ) : ℚ := success / x

def first_to_roll_six_probability (a b c : ℕ) : ℚ :=
  let p_six : ℚ := probability_roll 6 1
  let p_not_six : ℚ := 1 - p_six
  let cycle_prob : ℚ := p_not_six * p_not_six * p_six
  let continue_prob : ℚ := p_not_six * p_not_six * p_not_six
  let geometric_sum : ℚ := cycle_prob / (1 - continue_prob)
  geometric_sum

theorem carol_first_to_roll_six :
  first_to_roll_six_probability 1 1 1 = 25 / 91 := 
sorry

end carol_first_to_roll_six_l273_273522


namespace cagr_decline_l273_273708

theorem cagr_decline 
  (EV BV : ℝ) (n : ℕ) 
  (h_ev : EV = 52)
  (h_bv : BV = 89)
  (h_n : n = 3)
: ((EV / BV) ^ (1 / n) - 1) = -0.1678 := 
by
  rw [h_ev, h_bv, h_n]
  sorry

end cagr_decline_l273_273708


namespace total_volume_of_four_boxes_l273_273890

theorem total_volume_of_four_boxes :
  (∃ (V : ℕ), (∀ (edge_length : ℕ) (num_boxes : ℕ), edge_length = 6 → num_boxes = 4 → V = (edge_length ^ 3) * num_boxes)) :=
by
  let edge_length := 6
  let num_boxes := 4
  let volume := (edge_length ^ 3) * num_boxes
  use volume
  sorry

end total_volume_of_four_boxes_l273_273890


namespace lottery_probability_correct_l273_273515

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def combination (n k : Nat) : Nat := factorial n / (factorial k * factorial (n - k))

theorem lottery_probability_correct :
  let MegaBall_probability := 1 / 30
  let WinnerBalls_probability := 1 / (combination 50 6)
  MegaBall_probability * WinnerBalls_probability = 1 / 476721000 :=
by
  sorry

end lottery_probability_correct_l273_273515


namespace trapezoid_problem_case1_trapezoid_problem_case2_trapezoid_problem_case3_l273_273190

variable {a b : ℝ}
variable {M N : ℝ}

/-- Trapezoid problem statements -/
theorem trapezoid_problem_case1 (h : a < 2 * b) : M - N = a - 2 * b := 
sorry

theorem trapezoid_problem_case2 (h : a = 2 * b) : M - N = 0 := 
sorry

theorem trapezoid_problem_case3 (h : a > 2 * b) : M - N = 2 * b - a := 
sorry

end trapezoid_problem_case1_trapezoid_problem_case2_trapezoid_problem_case3_l273_273190


namespace distinct_factors_81_l273_273812

theorem distinct_factors_81 : nat.factors_count 81 = 5 :=
sorry

end distinct_factors_81_l273_273812


namespace min_minutes_to_make_B_cheaper_l273_273268

def costA (x : ℕ) : ℕ :=
  if x ≤ 300 then 8 * x else 2400 + 7 * (x - 300)

def costB (x : ℕ) : ℕ := 2500 + 4 * x

theorem min_minutes_to_make_B_cheaper : ∃ (x : ℕ), x ≥ 301 ∧ costB x < costA x :=
by
  use 301
  sorry

end min_minutes_to_make_B_cheaper_l273_273268


namespace projection_of_a_onto_b_l273_273138

namespace VectorProjection

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2)

noncomputable def scalar_projection (a b : ℝ × ℝ) : ℝ :=
  (dot_product a b) / (magnitude b)

theorem projection_of_a_onto_b :
  scalar_projection (1, -2) (3, 4) = -1 := by
    sorry

end VectorProjection

end projection_of_a_onto_b_l273_273138


namespace pints_in_5_liters_l273_273422

-- Define the condition based on the given conversion factor from liters to pints
def conversion_factor : ℝ := 2.1

-- The statement we need to prove
theorem pints_in_5_liters : 5 * conversion_factor = 10.5 :=
by sorry

end pints_in_5_liters_l273_273422


namespace product_complex_numbers_l273_273145

noncomputable def Q : ℂ := 3 + 4 * Complex.I
noncomputable def E : ℂ := 2 * Complex.I
noncomputable def D : ℂ := 3 - 4 * Complex.I
noncomputable def R : ℝ := 2

theorem product_complex_numbers : Q * E * D * (R : ℂ) = 100 * Complex.I := by
  sorry

end product_complex_numbers_l273_273145


namespace kite_area_l273_273948

theorem kite_area {length height : ℕ} (h_length : length = 8) (h_height : height = 10): 
  2 * (1/2 * (length * 2) * (height * 2 / 2)) = 160 :=
by
  rw [h_length, h_height]
  norm_num
  sorry

end kite_area_l273_273948


namespace solve_ineq_l273_273284

noncomputable def inequality (x : ℝ) : Prop :=
  (x^2 / (x+1)) ≥ (3 / (x+1) + 3)

theorem solve_ineq :
  { x : ℝ | inequality x } = { x : ℝ | x ≤ -6 ∨ (-1 < x ∧ x ≤ 3) } := sorry

end solve_ineq_l273_273284


namespace arithmetic_sequence_sum_l273_273438

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {d : ℝ}
variable {n : ℕ}

-- Conditions of the problem
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d
def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) := ∀ n : ℕ, S n = n * (a 1 + a n) / 2
def S9_is_90 (S : ℕ → ℝ) := S 9 = 90

-- The proof goal
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ)
  (h1 : is_arithmetic_sequence a d)
  (h2 : sum_first_n_terms a S)
  (h3 : S9_is_90 S) :
  a 3 + a 5 + a 7 = 30 :=
by
  sorry

end arithmetic_sequence_sum_l273_273438


namespace min_max_diff_val_l273_273574

def find_min_max_diff (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : ℝ :=
  let m := 0
  let M := 1
  M - m

theorem min_max_diff_val (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : find_min_max_diff x y hx hy = 1 :=
by sorry

end min_max_diff_val_l273_273574


namespace fraction_calculation_l273_273598

theorem fraction_calculation :
  (1 / 4) * (1 / 3) * (1 / 6) * 144 + (1 / 2) = (5 / 2) :=
by
  sorry

end fraction_calculation_l273_273598


namespace midpoint_integer_of_five_points_l273_273881

theorem midpoint_integer_of_five_points 
  (P : Fin 5 → ℤ × ℤ) 
  (distinct : Function.Injective P) :
  ∃ i j : Fin 5, i ≠ j ∧ (P i).1 + (P j).1 % 2 = 0 ∧ (P i).2 + (P j).2 % 2 = 0 :=
by
  sorry

end midpoint_integer_of_five_points_l273_273881


namespace Nedy_crackers_total_l273_273578

theorem Nedy_crackers_total :
  let packs_from_Mon_to_Thu := 8
  let packs_on_Fri := 2 * packs_from_Mon_to_Thu
  (packs_from_Mon_to_Thu + packs_on_Fri) = 24 :=
by
  let packs_from_Mon_to_Thu := 8
  let packs_on_Fri := 2 * packs_from_Mon_to_Thu
  show packs_from_Mon_to_Thu + packs_on_Fri = 24
  sorry

end Nedy_crackers_total_l273_273578


namespace ball_total_attendance_l273_273188

theorem ball_total_attendance :
    ∃ n m : ℕ, n + m = 41 ∧
    (n + m < 50) ∧
    (3 * n / 4 = (5 * m / 7)) :=
begin
  sorry
end

end ball_total_attendance_l273_273188


namespace jenna_interest_l273_273163

def compound_interest (P r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

def interest_earned (P r : ℝ) (n : ℕ) : ℝ :=
  compound_interest P r n - P

theorem jenna_interest :
  interest_earned 1500 0.05 5 = 414.42 :=
by
  sorry

end jenna_interest_l273_273163


namespace tangent_line_equation_l273_273909

noncomputable def equationOfTangentLine (P : ℝ × ℝ) (C : ℝ × ℝ → ℝ) :=
  ∃ (l : ℝ → ℝ), ∀ (x y : ℝ), y = l x ↔ x + 2*y - 6 = 0

theorem tangent_line_equation :
  let P := (2, 2)
  let C (p : ℝ × ℝ) := (p.1 - 1)^2 + p.2^2 = 5
  in equationOfTangentLine P (λ p, C p) := 
sorry

end tangent_line_equation_l273_273909


namespace triangles_from_decagon_l273_273776

theorem triangles_from_decagon : 
  ∃ (n : ℕ), n = 10 ∧ (nat.choose n 3 = 120) :=
by
  use 10,
  split,
  -- First condition: the decagon has 10 vertices
  rfl,
  -- Prove the number of distinct triangles
  sorry

end triangles_from_decagon_l273_273776


namespace roots_equal_and_real_l273_273931

theorem roots_equal_and_real:
  (∀ (x : ℝ), x ^ 2 - 3 * x * y + y ^ 2 + 2 * x - 9 * y + 1 = 0 → (y = 0 ∨ y = -24 / 5)) ∧
  (∀ (x : ℝ), x ^ 2 - 3 * x * y + y ^ 2 + 2 * x - 9 * y + 1 = 0 → (y ≥ 0 ∨ y ≤ -24 / 5)) :=
  by sorry

end roots_equal_and_real_l273_273931


namespace simplify_expression_l273_273858

theorem simplify_expression (x : ℝ) (hx : x ≠ 4):
  (x^2 - 4 * x) / (x^2 - 8 * x + 16) = x / (x - 4) :=
by sorry

end simplify_expression_l273_273858


namespace problem_BD_l273_273295

variable (a b c : ℝ)

theorem problem_BD (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) :
  (c - a < c - b) ∧ (a⁻¹ * c > b⁻¹ * c) :=
by
  sorry

end problem_BD_l273_273295


namespace sum_of_positive_integer_factors_of_24_l273_273491

-- Define the number 24
def n : ℕ := 24

-- Define the list of positive factors of 24
def pos_factors_of_24 : List ℕ := [1, 2, 4, 8, 3, 6, 12, 24]

-- Define the sum of the factors
def sum_of_factors : ℕ := pos_factors_of_24.sum

-- The theorem statement
theorem sum_of_positive_integer_factors_of_24 : sum_of_factors = 60 := by
  sorry

end sum_of_positive_integer_factors_of_24_l273_273491


namespace complete_the_square_l273_273825

theorem complete_the_square (x : ℝ) : 
  ∃ (a h k : ℝ), a = 1 ∧ h = 7 / 2 ∧ k = -49 / 4 ∧ x^2 - 7 * x = a * (x - h) ^ 2 + k :=
by
  use 1, 7 / 2, -49 / 4
  sorry

end complete_the_square_l273_273825


namespace polygon_sides_l273_273033

theorem polygon_sides (n : ℕ) 
  (h_interior : (n - 2) * 180 = 4 * 360) : 
  n = 10 :=
by
  sorry

end polygon_sides_l273_273033


namespace triangular_square_l273_273444

def triangular (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem triangular_square (m n : ℕ) (h1 : 1 ≤ m) (h2 : 1 ≤ n) (h3 : 2 * triangular m = triangular n) :
  ∃ k : ℕ, triangular (2 * m - n) = k * k :=
by
  sorry

end triangular_square_l273_273444


namespace total_rent_of_pasture_l273_273391

theorem total_rent_of_pasture 
  (oxen_A : ℕ) (months_A : ℕ) (oxen_B : ℕ) (months_B : ℕ)
  (oxen_C : ℕ) (months_C : ℕ) (share_C : ℕ) (total_rent : ℕ) :
  oxen_A = 10 →
  months_A = 7 →
  oxen_B = 12 →
  months_B = 5 →
  oxen_C = 15 →
  months_C = 3 →
  share_C = 72 →
  total_rent = 280 :=
by
  intros hA1 hA2 hB1 hB2 hC1 hC2 hC3
  sorry

end total_rent_of_pasture_l273_273391


namespace num_integers_satisfying_ineq_count_integers_satisfying_ineq_l273_273539

theorem num_integers_satisfying_ineq (k : ℤ) :
  (20 < k^2 ∧ k^2 < 150) ↔ k ∈ ({-12, -11, -10, -9, -8, -7, -6, -5, 5, 6, 7, 8, 9, 10, 11, 12} : set ℤ) := by
  sorry

theorem count_integers_satisfying_ineq :
  {n : ℤ | 20 < n^2 ∧ n^2 < 150}.finite.to_finset.card = 16 := by
  sorry

end num_integers_satisfying_ineq_count_integers_satisfying_ineq_l273_273539


namespace angle_BPE_l273_273007

-- Define the conditions given in the problem
def triangle_ABC (A B C : ℝ) : Prop := A = 60 ∧ 
  (∃ (B₁ B₂ B₃ : ℝ), B₁ = B / 3 ∧ B₂ = B / 3 ∧ B₃ = B / 3) ∧ 
  (∃ (C₁ C₂ C₃ : ℝ), C₁ = C / 3 ∧ C₂ = C / 3 ∧ C₃ = C / 3) ∧ 
  (B + C = 120)

-- State the theorem to proof
theorem angle_BPE (A B C x : ℝ) (h : triangle_ABC A B C) : x = 50 := by
  sorry

end angle_BPE_l273_273007


namespace solution_system_of_inequalities_l273_273215

theorem solution_system_of_inequalities (x : ℝ) : 
  (3 * x - 2) / (x - 6) ≤ 1 ∧ 2 * (x^2) - x - 1 > 0 ↔ (-2 ≤ x ∧ x < -1/2) ∨ (1 < x ∧ x < 6) :=
by {
  sorry
}

end solution_system_of_inequalities_l273_273215


namespace value_two_sd_below_mean_l273_273218

theorem value_two_sd_below_mean :
  let mean := 14.5
  let stdev := 1.7
  mean - 2 * stdev = 11.1 :=
by
  sorry

end value_two_sd_below_mean_l273_273218


namespace order_of_exponents_l273_273703

theorem order_of_exponents (p q r : ℕ) (hp : p = 2^3009) (hq : q = 3^2006) (hr : r = 5^1003) : r < p ∧ p < q :=
by {
  sorry -- Proof will go here
}

end order_of_exponents_l273_273703


namespace symphony_orchestra_has_260_members_l273_273043

def symphony_orchestra_member_count (n : ℕ) : Prop :=
  200 < n ∧ n < 300 ∧ n % 6 = 2 ∧ n % 8 = 3 ∧ n % 9 = 4

theorem symphony_orchestra_has_260_members : symphony_orchestra_member_count 260 :=
by {
  sorry
}

end symphony_orchestra_has_260_members_l273_273043


namespace initial_passengers_l273_273672

theorem initial_passengers (P : ℕ) (H1 : P - 263 + 419 = 725) : P = 569 :=
by
  sorry

end initial_passengers_l273_273672


namespace brad_more_pages_than_greg_l273_273427

def greg_pages_first_week : ℕ := 7 * 18
def greg_pages_next_two_weeks : ℕ := 14 * 22
def greg_total_pages : ℕ := greg_pages_first_week + greg_pages_next_two_weeks

def brad_pages_first_5_days : ℕ := 5 * 26
def brad_pages_remaining_12_days : ℕ := 12 * 20
def brad_total_pages : ℕ := brad_pages_first_5_days + brad_pages_remaining_12_days

def total_required_pages : ℕ := 800

theorem brad_more_pages_than_greg : brad_total_pages - greg_total_pages = 64 :=
by
  sorry

end brad_more_pages_than_greg_l273_273427


namespace simplify_tangent_expression_l273_273204

theorem simplify_tangent_expression :
  (1 + Real.tan (Real.pi / 18)) * (1 + Real.tan (35 * Real.pi / 180)) = 2 :=
by
  sorry

end simplify_tangent_expression_l273_273204


namespace value_of_expression_l273_273669

theorem value_of_expression (a b c : ℝ) (h1 : 4 * a = 5 * b) (h2 : 5 * b = 30) (h3 : a + b + c = 15) : 40 * a * b / c = 1200 :=
by
  sorry

end value_of_expression_l273_273669


namespace original_number_l273_273499

theorem original_number (x : ℝ) (h : x * 1.20 = 1080) : x = 900 :=
sorry

end original_number_l273_273499


namespace sum_of_interior_angles_of_polygon_l273_273252

theorem sum_of_interior_angles_of_polygon (n : ℕ) (h : n - 3 = 3) : (n - 2) * 180 = 720 :=
by
  sorry

end sum_of_interior_angles_of_polygon_l273_273252


namespace distinct_p_q_r_s_t_sum_l273_273576

theorem distinct_p_q_r_s_t_sum (p q r s t : ℤ) (h1 : (8 - p) * (8 - q) * (8 - r) * (8 - s) * (8 - t) = 120)
    (h2 : p ≠ q) (h3 : p ≠ r) (h4 : p ≠ s) (h5 : p ≠ t) 
    (h6 : q ≠ r) (h7 : q ≠ s) (h8 : q ≠ t)
    (h9 : r ≠ s) (h10 : r ≠ t)
    (h11 : s ≠ t) : p + q + r + s + t = 25 := by
  sorry

end distinct_p_q_r_s_t_sum_l273_273576


namespace remainder_of_difference_divided_by_prime_l273_273847

def largest_three_digit_number : ℕ := 999
def smallest_five_digit_number : ℕ := 10000
def smallest_prime_greater_than_1000 : ℕ := 1009

theorem remainder_of_difference_divided_by_prime :
  (smallest_five_digit_number - largest_three_digit_number) % smallest_prime_greater_than_1000 = 945 :=
by
  -- The proof will be filled in here
  sorry

end remainder_of_difference_divided_by_prime_l273_273847


namespace sqrt_last_digit_l273_273709

-- Definitions related to the problem
def is_p_adic_number (α : ℕ) (p : ℕ) := true -- assume this definition captures p-adic number system

-- Problem statement in Lean 4
theorem sqrt_last_digit (p α a1 b1 : ℕ) (hα : is_p_adic_number α p) (h_last_digit_α : α % p = a1)
  (h_sqrt : (b1 * b1) % p = α % p) :
  (b1 * b1) % p = a1 :=
by sorry

end sqrt_last_digit_l273_273709


namespace mark_cans_l273_273625

theorem mark_cans (R J M : ℕ) (h1 : J = 2 * R + 5) (h2 : M = 4 * J) (h3 : R + J + M = 135) : M = 100 :=
by
  sorry

end mark_cans_l273_273625


namespace passes_through_point_P_l273_273665

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 7 + a^(x - 1)

theorem passes_through_point_P
  (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) : f a 1 = 8 :=
by
  -- Proof omitted
  sorry

end passes_through_point_P_l273_273665


namespace fraction_meaningful_iff_l273_273680

theorem fraction_meaningful_iff (x : ℝ) : (∃ y, y = 1 / (x - 1)) ↔ x ≠ 1 :=
by 
  sorry

end fraction_meaningful_iff_l273_273680


namespace find_natural_number_n_l273_273535

theorem find_natural_number_n : 
  ∃ (n : ℕ), (∃ k : ℕ, n + 15 = k^2) ∧ (∃ m : ℕ, n - 14 = m^2) ∧ n = 210 :=
by
  sorry

end find_natural_number_n_l273_273535


namespace trailing_zeros_300_factorial_l273_273874

-- Definition to count the number of times a prime factor divides the factorial of n
def count_factors (n : ℕ) (p : ℕ) : ℕ :=
  Nat.div (n / p) 1 + Nat.div (n / p^2) 1 + Nat.div (n / p^3) 1 + Nat.div (n / p^4) 1

-- Theorem stating the number of trailing zeros in 300! is 74
theorem trailing_zeros_300_factorial : count_factors 300 5 = 74 := by
  sorry

end trailing_zeros_300_factorial_l273_273874


namespace football_team_total_progress_l273_273508

theorem football_team_total_progress :
  let play1 := -5
  let play2 := 13
  let play3 := -2 * play1
  let play4 := play3 / 2
  play1 + play2 + play3 + play4 = 3 :=
by
  sorry

end football_team_total_progress_l273_273508


namespace simplify_tangent_expression_l273_273206

theorem simplify_tangent_expression :
  (1 + Real.tan (Real.pi / 18)) * (1 + Real.tan (35 * Real.pi / 180)) = 2 :=
by
  sorry

end simplify_tangent_expression_l273_273206


namespace gcd_of_given_lcm_and_ratio_l273_273317

theorem gcd_of_given_lcm_and_ratio (C D : ℕ) (h1 : Nat.lcm C D = 200) (h2 : C * 5 = D * 2) : Nat.gcd C D = 5 :=
sorry

end gcd_of_given_lcm_and_ratio_l273_273317


namespace inequality_does_not_hold_l273_273083

noncomputable def f : ℝ → ℝ := sorry -- define f satisfying the conditions from a)

theorem inequality_does_not_hold :
  (∀ x, f (-x) = f x) ∧ -- f is even
  (∀ x, f x = f (x + 2)) ∧ -- f is periodic with period 2
  (∀ x, 3 ≤ x ∧ x ≤ 4 → f x = 2^x) → -- f(x) = 2^x when x is in [3, 4]
  ¬ (f (Real.sin 3) < f (Real.cos 3)) := by
  -- skipped proof
  sorry

end inequality_does_not_hold_l273_273083


namespace fraction_zero_implies_x_eq_neg3_l273_273821

theorem fraction_zero_implies_x_eq_neg3 (x : ℝ) (h1 : x ≠ 3) (h2 : (x^2 - 9) / (x - 3) = 0) : x = -3 :=
sorry

end fraction_zero_implies_x_eq_neg3_l273_273821


namespace median_score_interval_l273_273271

def intervals : List (Nat × Nat × Nat) :=
  [(80, 84, 20), (75, 79, 18), (70, 74, 15), (65, 69, 22), (60, 64, 14), (55, 59, 11)]

def total_students : Nat := 100

def median_interval : Nat × Nat :=
  (70, 74)

theorem median_score_interval :
  ∃ l u n, intervals = [(80, 84, 20), (75, 79, 18), (70, 74, 15), (65, 69, 22), (60, 64, 14), (55, 59, 11)]
  ∧ total_students = 100
  ∧ median_interval = (70, 74)
  ∧ ((l, u, n) ∈ intervals ∧ l ≤ 50 ∧ 50 ≤ u) :=
by
  sorry

end median_score_interval_l273_273271


namespace students_between_hoseok_and_minyoung_l273_273044

def num_students : Nat := 13
def hoseok_position_from_right : Nat := 9
def minyoung_position_from_left : Nat := 8

theorem students_between_hoseok_and_minyoung
    (n : Nat)
    (h : n = num_students)
    (p_h : n - hoseok_position_from_right + 1 = 5)
    (p_m : minyoung_position_from_left = 8):
    ∃ k : Nat, k = 2 :=
by
  sorry

end students_between_hoseok_and_minyoung_l273_273044


namespace complex_problem_l273_273816

theorem complex_problem (a b : ℝ) (i : ℂ) (hi : i^2 = -1) 
  (h : (a - 2 * i) * i = b - i) : a + b = 1 :=
by
  sorry

end complex_problem_l273_273816


namespace set_theorem_l273_273846

noncomputable def set_A : Set ℕ := {1, 2}
noncomputable def set_B : Set ℕ := {1, 2, 3}
noncomputable def set_C : Set ℕ := {2, 3, 4}

theorem set_theorem : (set_A ∩ set_B) ∪ set_C = {1, 2, 3, 4} := by
  sorry

end set_theorem_l273_273846


namespace binary_arithmetic_l273_273281

theorem binary_arithmetic :
  let a := 0b11101
  let b := 0b10011
  let c := 0b101
  (a * b) / c = 0b11101100 :=
by
  sorry

end binary_arithmetic_l273_273281


namespace infinite_solutions_if_one_exists_l273_273996

namespace RationalSolutions

def has_rational_solution (a b : ℚ) : Prop :=
  ∃ (x y : ℚ), a * x^2 + b * y^2 = 1

def infinite_rational_solutions (a b : ℚ) : Prop :=
  ∀ (x₀ y₀ : ℚ), (a * x₀^2 + b * y₀^2 = 1) → ∃ (f : ℕ → ℚ × ℚ), ∀ n : ℕ, a * (f n).1^2 + b * (f n).2^2 = 1 ∧ (f 0 = (x₀, y₀)) ∧ ∀ m n : ℕ, m ≠ n → (f m) ≠ (f n)

theorem infinite_solutions_if_one_exists (a b : ℚ) (h : has_rational_solution a b) : infinite_rational_solutions a b :=
  sorry

end RationalSolutions

end infinite_solutions_if_one_exists_l273_273996


namespace avg_of_x_y_is_41_l273_273289

theorem avg_of_x_y_is_41 
  (x y : ℝ) 
  (h : (4 + 6 + 8 + x + y) / 5 = 20) 
  : (x + y) / 2 = 41 := 
by 
  sorry

end avg_of_x_y_is_41_l273_273289


namespace sqrt_abc_sum_eq_54_sqrt_5_l273_273991

theorem sqrt_abc_sum_eq_54_sqrt_5 
  (a b c : ℝ) 
  (h1 : b + c = 17) 
  (h2 : c + a = 18) 
  (h3 : a + b = 19) : 
  Real.sqrt (a * b * c * (a + b + c)) = 54 * Real.sqrt 5 := 
by 
  sorry

end sqrt_abc_sum_eq_54_sqrt_5_l273_273991


namespace stratified_sampling_male_athletes_l273_273914

theorem stratified_sampling_male_athletes (total_males : ℕ) (total_females : ℕ) (sample_size : ℕ)
  (total_population : ℕ) (male_sample_fraction : ℚ) (n_sample_males : ℕ) :
  total_males = 56 →
  total_females = 42 →
  sample_size = 28 →
  total_population = total_males + total_females →
  male_sample_fraction = (sample_size : ℚ) / (total_population : ℚ) →
  n_sample_males = (total_males : ℚ) * male_sample_fraction →
  n_sample_males = 16 := by
  intros h_males h_females h_samples h_population h_fraction h_final
  sorry

end stratified_sampling_male_athletes_l273_273914


namespace sarah_daily_candy_consumption_l273_273290

def neighbors_candy : ℕ := 66
def sister_candy : ℕ := 15
def days : ℕ := 9

def total_candy : ℕ := neighbors_candy + sister_candy
def average_daily_consumption : ℕ := total_candy / days

theorem sarah_daily_candy_consumption : average_daily_consumption = 9 := by
  sorry

end sarah_daily_candy_consumption_l273_273290


namespace count_twelfth_power_l273_273815

-- Define the conditions under which a number must meet the criteria of being a square, a cube, and a fourth power
def is_square (n : ℕ) : Prop := ∃ m : ℕ, m^2 = n
def is_cube (n : ℕ) : Prop := ∃ m : ℕ, m^3 = n
def is_fourth_power (n : ℕ) : Prop := ∃ m : ℕ, m^4 = n

-- Define the main theorem, which proves the count of numbers less than 1000 meeting all criteria
theorem count_twelfth_power (h : ∀ n, is_square n → is_cube n → is_fourth_power n → n < 1000) :
  ∃! x : ℕ, x < 1000 ∧ ∃ k : ℕ, k^12 = x := 
sorry

end count_twelfth_power_l273_273815


namespace area_difference_depends_only_on_bw_l273_273845

variable (b w n : ℕ)
variable (hb : b ≥ 2)
variable (hw : w ≥ 2)
variable (hn : n = b + w)

/-- Given conditions: 
1. \(b \geq 2\) 
2. \(w \geq 2\) 
3. \(n = b + w\)
4. There are \(2b\) identical black rods and \(2w\) identical white rods, each of side length 1. 
5. These rods form a regular \(2n\)-gon with parallel sides of the same color.
6. A convex \(2b\)-gon \(B\) is formed by translating the black rods. 
7. A convex \(2w\) A convex \(2w\)-gon \(W\) is formed by translating the white rods. 
Prove that the difference of the areas of \(B\) and \(W\) depends only on the numbers \(b\) and \(w\). -/
theorem area_difference_depends_only_on_bw :
  ∀ (A B W : ℝ), A - B = 2 * (b - w) :=
sorry

end area_difference_depends_only_on_bw_l273_273845


namespace ball_attendance_l273_273183

noncomputable def num_people_attending_ball (n m : ℕ) : ℕ := n + m 

theorem ball_attendance:
  ∀ (n m : ℕ), 
  n + m < 50 ∧ 
  (n - n / 4) = (5 * m / 7) →
  num_people_attending_ball n m = 41 :=
by 
  intros n m h
  have h1 : n + m < 50 := h.1
  have h2 : n - n / 4 = 5 * m / 7 := h.2
  sorry

end ball_attendance_l273_273183


namespace man_twice_son_age_in_years_l273_273511

theorem man_twice_son_age_in_years :
  ∀ (S M Y : ℕ),
  (M = S + 26) →
  (S = 24) →
  (M + Y = 2 * (S + Y)) →
  Y = 2 :=
by
  intros S M Y h1 h2 h3
  sorry

end man_twice_son_age_in_years_l273_273511


namespace probability_of_intersecting_diagonals_l273_273362

def num_vertices := 8
def total_diagonals := (num_vertices * (num_vertices - 3)) / 2
def total_ways_to_choose_two_diagonals := Nat.choose total_diagonals 2
def ways_to_choose_4_vertices := Nat.choose num_vertices 4
def number_of_intersecting_pairs := ways_to_choose_4_vertices
def probability_intersecting_diagonals := (number_of_intersecting_pairs : ℚ) / (total_ways_to_choose_two_diagonals : ℚ)

theorem probability_of_intersecting_diagonals :
  probability_intersecting_diagonals = 7 / 19 := by
  sorry

end probability_of_intersecting_diagonals_l273_273362


namespace triangle_obtuse_at_15_l273_273988

-- Define the initial angles of the triangle
def x0 : ℝ := 59.999
def y0 : ℝ := 60
def z0 : ℝ := 60.001

-- Define the recurrence relations for the angles
def x (n : ℕ) : ℝ := (-2)^n * (x0 - 60) + 60
def y (n : ℕ) : ℝ := (-2)^n * (y0 - 60) + 60
def z (n : ℕ) : ℝ := (-2)^n * (z0 - 60) + 60

-- Define the obtuseness condition
def is_obtuse (a : ℝ) : Prop := a > 90

-- The main theorem stating the least positive integer n is 15 for which the triangle A_n B_n C_n is obtuse
theorem triangle_obtuse_at_15 : ∃ n : ℕ, n > 0 ∧ 
  (is_obtuse (x n) ∨ is_obtuse (y n) ∨ is_obtuse (z n)) ∧ n = 15 :=
sorry

end triangle_obtuse_at_15_l273_273988


namespace max_students_gave_away_balls_more_l273_273397

theorem max_students_gave_away_balls_more (N : ℕ) (hN : N ≤ 13) : 
  ∃(students : ℕ), students = 27 ∧ (students = 27 ∧ N ≤ students - N) :=
by
  sorry

end max_students_gave_away_balls_more_l273_273397


namespace find_k_l273_273973

theorem find_k (x k : ℝ) (h₁ : (x^2 - k) * (x - k) = x^3 - k * (x^2 + x + 3))
               (h₂ : k ≠ 0) : k = -3 :=
by
  sorry

end find_k_l273_273973


namespace number_of_valid_pairs_l273_273774

theorem number_of_valid_pairs :
  ∃ (n : ℕ), n = 4950 ∧ ∀ (x y : ℕ), 
  1 ≤ x ∧ x < y ∧ y ≤ 200 ∧ 
  (Complex.I ^ x + Complex.I ^ y).im = 0 → n = 4950 :=
sorry

end number_of_valid_pairs_l273_273774


namespace negation_all_nonzero_l273_273588

    theorem negation_all_nonzero (a b c : ℝ) : ¬ (¬ (a = 0 ∨ b = 0 ∨ c = 0)) → (a = 0 ∧ b = 0 ∧ c = 0) :=
    by
      sorry
    
end negation_all_nonzero_l273_273588


namespace certain_number_is_84_l273_273868

/-
The least number by which 72 must be multiplied in order to produce a multiple of a certain number is 14.
What is that certain number?
-/

theorem certain_number_is_84 (x : ℕ) (h: 72 * 14 % x = 0 ∧ ∀ y : ℕ, 1 ≤ y → y < 14 → 72 * y % x ≠ 0) : x = 84 :=
sorry

end certain_number_is_84_l273_273868


namespace simplify_tangent_sum_l273_273207

theorem simplify_tangent_sum :
  (1 + Real.tan (10 * Real.pi / 180)) * (1 + Real.tan (35 * Real.pi / 180)) = 2 := by
  have h1 : Real.tan (45 * Real.pi / 180) = Real.tan ((10 + 35) * Real.pi / 180) := by sorry
  have h2 : Real.tan (45 * Real.pi / 180) = 1 := by sorry
  have h3 : ∀ x y, Real.tan (x + y) = (Real.tan x + Real.tan y) / (1 - Real.tan x * Real.tan y) := by sorry
  sorry

end simplify_tangent_sum_l273_273207


namespace installation_time_l273_273378

-- Definitions (based on conditions)
def total_windows := 14
def installed_windows := 8
def hours_per_window := 8

-- Define what we need to prove
def remaining_windows := total_windows - installed_windows
def total_install_hours := remaining_windows * hours_per_window

theorem installation_time : total_install_hours = 48 := by
  sorry

end installation_time_l273_273378


namespace sqrt_expression_eq_two_l273_273924

theorem sqrt_expression_eq_two : 
  (Real.sqrt 3) * (Real.sqrt 3 - 1 / (Real.sqrt 3)) = 2 := 
  sorry

end sqrt_expression_eq_two_l273_273924


namespace john_games_l273_273836

variables (G_f G_g B G G_t : ℕ)

theorem john_games (h1: G_f = 21) (h2: B = 23) (h3: G = 6) 
(h4: G_t = G_f + G_g) (h5: G + B = G_t) : G_g = 8 :=
by sorry

end john_games_l273_273836


namespace probability_correct_predictions_monday_l273_273073

def number_of_classes_monday : ℕ := 5
def number_of_classes_tuesday : ℕ := 6
def total_classes : ℕ := number_of_classes_monday + number_of_classes_tuesday

def total_correct_predictions : ℕ := 7
def correct_predictions_monday : ℕ := 3
def correct_predictions_tuesday : ℕ := total_correct_predictions - correct_predictions_monday

noncomputable def binomial_coefficient (n k : ℕ) : ℝ := (nat.choose n k : ℝ)

noncomputable def probability_exact_correct_monday (n k : ℕ) : ℝ :=
  binomial_coefficient n k * (real.exp 11 (ln 2⁻¹ * total_classes))

theorem probability_correct_predictions_monday :
  probability_exact_correct_monday number_of_classes_monday correct_predictions_monday *
  probability_exact_correct_monday number_of_classes_tuesday correct_predictions_tuesday /
  probability_exact_correct_monday total_classes total_correct_predictions = 5 / 11 :=
sorry

end probability_correct_predictions_monday_l273_273073


namespace compare_negatives_l273_273647

theorem compare_negatives : -3.3 < -3.14 :=
sorry

end compare_negatives_l273_273647


namespace rabbit_roaming_area_l273_273162

noncomputable def rabbit_area_midpoint_long_side (r: ℝ) : ℝ :=
  (1/2) * Real.pi * r^2

noncomputable def rabbit_area_3_ft_from_corner (R r: ℝ) : ℝ :=
  (3/4) * Real.pi * R^2 - (1/4) * Real.pi * r^2

theorem rabbit_roaming_area (r R : ℝ) (h_r_pos: 0 < r) (h_R_pos: r < R) :
  rabbit_area_3_ft_from_corner R r - rabbit_area_midpoint_long_side R = 22.75 * Real.pi :=
by
  sorry

end rabbit_roaming_area_l273_273162


namespace smallest_integer_represented_as_AA6_and_BB8_l273_273887

def valid_digit_in_base (d : ℕ) (b : ℕ) : Prop := d < b

theorem smallest_integer_represented_as_AA6_and_BB8 :
  ∃ (n : ℕ) (A B : ℕ),
  valid_digit_in_base A 6 ∧ valid_digit_in_base B 8 ∧ 
  n = 7 * A ∧ n = 9 * B ∧ n = 63 :=
by
  sorry

end smallest_integer_represented_as_AA6_and_BB8_l273_273887


namespace train_speed_l273_273915

theorem train_speed (distance : ℝ) (time_minutes : ℝ) (speed : ℝ) (h_distance : distance = 7.5) (h_time : time_minutes = 5) :
  speed = 90 :=
by
  sorry

end train_speed_l273_273915


namespace train_length_correct_l273_273517

-- Define the given conditions as constants
def time_to_cross_pole : ℝ := 4.99960003199744
def speed_kmh : ℝ := 72

-- Convert the speed from km/hr to m/s
def speed_ms : ℝ := speed_kmh * (1000 / 3600)

-- Calculate the length of the train
def length_of_train : ℝ := speed_ms * time_to_cross_pole

-- The problem statement: prove that length_of_train is approximately 99.992 meters
theorem train_length_correct : abs (length_of_train - 99.992) < 0.001 := by
  sorry

end train_length_correct_l273_273517


namespace find_rs_l273_273850

theorem find_rs (r s : ℝ) (h1 : 0 < r) (h2 : 0 < s) (h3 : r^2 + s^2 = 1) (h4 : r^4 + s^4 = 7/8) : 
  r * s = 1/4 :=
sorry

end find_rs_l273_273850


namespace eval_expression_l273_273565

theorem eval_expression (x : ℝ) (h₀ : x = 3) :
  let initial_expr : ℝ := (2 * x + 2) / (x - 2)
  let replaced_expr : ℝ := (2 * initial_expr + 2) / (initial_expr - 2)
  replaced_expr = 8 :=
by
  sorry

end eval_expression_l273_273565


namespace no_real_solutions_to_equation_l273_273933

theorem no_real_solutions_to_equation :
  ¬ ∃ y : ℝ, (3 * y - 4)^2 + 4 = -(y + 3) :=
by
  sorry

end no_real_solutions_to_equation_l273_273933


namespace identify_solids_with_identical_views_l273_273912

def has_identical_views (s : Type) : Prop := sorry

def sphere : Type := sorry
def triangular_pyramid : Type := sorry
def cube : Type := sorry
def cylinder : Type := sorry

theorem identify_solids_with_identical_views :
  (has_identical_views sphere) ∧
  (¬ has_identical_views triangular_pyramid) ∧
  (has_identical_views cube) ∧
  (¬ has_identical_views cylinder) :=
sorry

end identify_solids_with_identical_views_l273_273912


namespace common_ratio_geometric_progression_l273_273976

theorem common_ratio_geometric_progression {x y z r : ℝ} (h_diff1 : x ≠ y) (h_diff2 : y ≠ z) (h_diff3 : z ≠ x)
  (hx_nonzero : x ≠ 0) (hy_nonzero : y ≠ 0) (hz_nonzero : z ≠ 0)
  (h_gm_progression : ∃ r : ℝ, x * (y - z) = x * (y - z) * r ∧ z * (x - y) = (y * (z - x)) * r) : r^2 + r + 1 = 0 :=
sorry

end common_ratio_geometric_progression_l273_273976


namespace max_value_of_f_l273_273016

noncomputable def f (x : ℝ) := 2 * Real.cos x + Real.sin x

theorem max_value_of_f : ∃ x, f x = Real.sqrt 5 := sorry

end max_value_of_f_l273_273016


namespace algebraic_expression_value_l273_273299

variables (x y : ℝ)

theorem algebraic_expression_value :
  x^2 - 4 * x - 1 = 0 →
  (2 * x - 3)^2 - (x + y) * (x - y) - y^2 = 12 :=
by
  intro h
  sorry

end algebraic_expression_value_l273_273299


namespace range_of_a_l273_273963

variable {a x : ℝ}

theorem range_of_a (h_eq : 2 * (x + a) = x + 3) (h_ineq : 2 * x - 10 > 8 * a) : a < -1 / 3 := 
sorry

end range_of_a_l273_273963


namespace train_speed_168_l273_273766

noncomputable def speed_of_train (L : ℕ) (V_man : ℕ) (T : ℕ) : ℚ :=
  let V_man_mps := (V_man * 5) / 18
  let relative_speed := L / T
  let V_train_mps := relative_speed - V_man_mps
  V_train_mps * (18 / 5)

theorem train_speed_168 :
  speed_of_train 500 12 10 = 168 :=
by
  sorry

end train_speed_168_l273_273766


namespace x_intercept_perpendicular_line_l273_273366

theorem x_intercept_perpendicular_line 
  (x y : ℝ)
  (h1 : 4 * x - 3 * y = 12)
  (h2 : y = - (3 / 4) * x + 4)
  : x = 16 / 3 := 
sorry

end x_intercept_perpendicular_line_l273_273366


namespace range_of_angle_of_inclination_l273_273349

theorem range_of_angle_of_inclination (α : ℝ) :
  ∃ θ : ℝ, θ ∈ (Set.Icc 0 (Real.pi / 4) ∪ Set.Ico (3 * Real.pi / 4) Real.pi) ∧
           ∀ x : ℝ, ∃ y : ℝ, y = x * Real.sin α + 1 := by
  sorry

end range_of_angle_of_inclination_l273_273349


namespace part1_part2_l273_273135

noncomputable def f (x a : ℝ) : ℝ := abs (2 * x - 5 * a) + abs (2 * x + 1)
noncomputable def g (x : ℝ) : ℝ := abs (x - 1) + 3

-- (1)
theorem part1 (x : ℝ) : abs (g x) < 8 → -4 < x ∧ x < 6 :=
by
  sorry

-- (2)
theorem part2 (a : ℝ) : (∀ x1 : ℝ, ∃ x2 : ℝ, f x1 a = g x2) → (a ≥ 0.4 ∨ a ≤ -0.8) :=
by
  sorry

end part1_part2_l273_273135


namespace probability_correct_predictions_monday_l273_273074

def number_of_classes_monday : ℕ := 5
def number_of_classes_tuesday : ℕ := 6
def total_classes : ℕ := number_of_classes_monday + number_of_classes_tuesday

def total_correct_predictions : ℕ := 7
def correct_predictions_monday : ℕ := 3
def correct_predictions_tuesday : ℕ := total_correct_predictions - correct_predictions_monday

noncomputable def binomial_coefficient (n k : ℕ) : ℝ := (nat.choose n k : ℝ)

noncomputable def probability_exact_correct_monday (n k : ℕ) : ℝ :=
  binomial_coefficient n k * (real.exp 11 (ln 2⁻¹ * total_classes))

theorem probability_correct_predictions_monday :
  probability_exact_correct_monday number_of_classes_monday correct_predictions_monday *
  probability_exact_correct_monday number_of_classes_tuesday correct_predictions_tuesday /
  probability_exact_correct_monday total_classes total_correct_predictions = 5 / 11 :=
sorry

end probability_correct_predictions_monday_l273_273074


namespace parameterized_line_equation_l273_273864

theorem parameterized_line_equation (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 * t + 6) 
  (h2 : y = 5 * t - 7) : 
  y = (5 / 3) * x - 17 :=
sorry

end parameterized_line_equation_l273_273864


namespace third_smallest_abc_sum_l273_273843

-- Define the necessary conditions and properties
def isIntegerRoots (a b c : ℕ) : Prop :=
  ∃ r1 r2 r3 r4 : ℤ, 
    a * r1^2 + b * r1 + c = 0 ∧ a * r2^2 + b * r2 - c = 0 ∧ 
    a * r3^2 - b * r3 + c = 0 ∧ a * r4^2 - b * r4 - c = 0

-- State the main theorem
theorem third_smallest_abc_sum : ∃ a b c : ℕ, 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ isIntegerRoots a b c ∧ 
  (a + b + c = 35 ∧ a = 1 ∧ b = 10 ∧ c = 24) :=
by sorry

end third_smallest_abc_sum_l273_273843


namespace football_team_practice_missed_days_l273_273621

theorem football_team_practice_missed_days 
(daily_practice_hours : ℕ) 
(total_practice_hours : ℕ) 
(days_in_week : ℕ) 
(h1 : daily_practice_hours = 5) 
(h2 : total_practice_hours = 30) 
(h3 : days_in_week = 7) : 
days_in_week - (total_practice_hours / daily_practice_hours) = 1 := 
by 
  sorry

end football_team_practice_missed_days_l273_273621


namespace emma_total_investment_l273_273279

theorem emma_total_investment (X : ℝ) (h : 0.09 * 6000 + 0.11 * (X - 6000) = 980) : X = 10000 :=
sorry

end emma_total_investment_l273_273279


namespace bob_distance_when_meet_l273_273707

def distance_xy : ℝ := 10
def yolanda_rate : ℝ := 3
def bob_rate : ℝ := 4
def time_start_diff : ℝ := 1

theorem bob_distance_when_meet : ∃ t : ℝ, yolanda_rate * t + bob_rate * (t - time_start_diff) = distance_xy ∧ bob_rate * (t - time_start_diff) = 4 :=
by
  sorry

end bob_distance_when_meet_l273_273707


namespace find_distance_between_stripes_l273_273764

-- Define the problem conditions
def parallel_curbs (a b : ℝ) := ∀ g : ℝ, g * a = b
def crosswalk_conditions (curb_distance curb_length stripe_length : ℝ) := 
  curb_distance = 60 ∧ curb_length = 22 ∧ stripe_length = 65

-- State the theorem
theorem find_distance_between_stripes (curb_distance curb_length stripe_length : ℝ) 
  (h : ℝ) (H : crosswalk_conditions curb_distance curb_length stripe_length) :
  h = 264 / 13 :=
sorry

end find_distance_between_stripes_l273_273764


namespace total_distance_traveled_l273_273050

theorem total_distance_traveled
  (r1 r2 : ℝ) (h1 : r1 = 15) (h2 : r2 = 25):
  let arc_outer := 1/4 * 2 * Real.pi * r2
  let radial := r2 - r1
  let circ_inner := 2 * Real.pi * r1
  let return_radial := radial
  let total_distance := arc_outer + radial + circ_inner + return_radial
  total_distance = 42.5 * Real.pi + 20 := 
by
  sorry

end total_distance_traveled_l273_273050


namespace calculate_full_recipes_needed_l273_273920

def initial_attendance : ℕ := 125
def attendance_drop_percentage : ℝ := 0.40
def cookies_per_student : ℕ := 2
def cookies_per_recipe : ℕ := 18

theorem calculate_full_recipes_needed :
  let final_attendance := initial_attendance * (1 - attendance_drop_percentage : ℝ)
  let total_cookies_needed := (final_attendance * (cookies_per_student : ℕ))
  let recipes_needed := total_cookies_needed / (cookies_per_recipe : ℕ)
  ⌈recipes_needed⌉ = 9 :=
  by
  sorry

end calculate_full_recipes_needed_l273_273920


namespace number_represented_by_B_l273_273009

theorem number_represented_by_B (b : ℤ) : 
  (abs (b - 3) = 5) -> (b = 8 ∨ b = -2) :=
by
  intro h
  sorry

end number_represented_by_B_l273_273009


namespace machine_working_time_l273_273636

def shirts_per_minute : ℕ := 3
def total_shirts_made : ℕ := 6

theorem machine_working_time : 
  (total_shirts_made / shirts_per_minute) = 2 :=
by
  sorry

end machine_working_time_l273_273636


namespace quadratic_term_elimination_l273_273236

theorem quadratic_term_elimination (m : ℝ) :
  (3 * (x : ℝ) ^ 2 - 10 - 2 * x - 4 * x ^ 2 + m * x ^ 2) = -(x : ℝ) * (2 * x + 10) ↔ m = 1 := 
by sorry

end quadratic_term_elimination_l273_273236


namespace volume_ratio_l273_273554

namespace Geometry

variables {Point : Type} [MetricSpace Point]

noncomputable def volume_pyramid (A B1 C1 D1 : Point) : ℝ := sorry

theorem volume_ratio 
  (A B1 B2 C1 C2 D1 D2 : Point) 
  (hA_B1: dist A B1 ≠ 0) (hA_B2: dist A B2 ≠ 0)
  (hA_C1: dist A C1 ≠ 0) (hA_C2: dist A C2 ≠ 0)
  (hA_D1: dist A D1 ≠ 0) (hA_D2: dist A D2 ≠ 0) :
  (volume_pyramid A B1 C1 D1 / volume_pyramid A B2 C2 D2) = 
    (dist A B1 * dist A C1 * dist A D1) / (dist A B2 * dist A C2 * dist A D2) := 
sorry

end Geometry

end volume_ratio_l273_273554


namespace clock_angle_7_35_l273_273059

noncomputable def hour_angle (hours : ℤ) (minutes : ℤ) : ℝ :=
  (hours * 30 + (minutes * 30) / 60 : ℝ)

noncomputable def minute_angle (minutes : ℤ) : ℝ :=
  (minutes * 360 / 60 : ℝ)

noncomputable def angle_between (angle1 angle2 : ℝ) : ℝ :=
  abs (angle1 - angle2)

theorem clock_angle_7_35 : angle_between (hour_angle 7 35) (minute_angle 35) = 17.5 :=
by
  sorry

end clock_angle_7_35_l273_273059


namespace cookies_number_l273_273273

-- Define all conditions in the problem
def number_of_chips_per_cookie := 7
def number_of_cookies_per_dozen := 12
def number_of_uneaten_chips := 168

-- Define D as the number of dozens of cookies
variable (D : ℕ)

-- Prove the Lean theorem
theorem cookies_number (h : 7 * 6 * D = 168) : D = 4 :=
by
  sorry

end cookies_number_l273_273273


namespace necessarily_positive_l273_273340

theorem necessarily_positive (a b c : ℝ) (ha : 0 < a ∧ a < 2) (hb : -2 < b ∧ b < 0) (hc : 0 < c ∧ c < 3) :
  (b + c) > 0 :=
sorry

end necessarily_positive_l273_273340


namespace mark_cans_l273_273623

variable (r j m : ℕ) -- r for Rachel, j for Jaydon, m for Mark

theorem mark_cans (r j m : ℕ) 
  (h1 : j = 5 + 2 * r)
  (h2 : m = 4 * j)
  (h3 : r + j + m = 135) : 
  m = 100 :=
by
  sorry

end mark_cans_l273_273623


namespace prob_A_prob_B_l273_273304

variable (a b : ℝ) -- Declare variables a and b as real numbers
variable (h_ab : a + b = 1) -- Declare the condition a + b = 1
variable (h_pos_a : 0 < a) -- Declare a is a positive real number
variable (h_pos_b : 0 < b) -- Declare b is a positive real number

-- Prove that 1/a + 1/b ≥ 4 under the given conditions
theorem prob_A (h_ab : a + b = 1) (h_pos_a : 0 < a) (h_pos_b : 0 < b) : 
  (1 / a) + (1 / b) ≥ 4 :=
by
  sorry

-- Prove that a^2 + b^2 ≥ 1/2 under the given conditions
theorem prob_B (h_ab : a + b = 1) (h_pos_a : 0 < a) (h_pos_b : 0 < b) : 
  a^2 + b^2 ≥ 1 / 2 :=
by
  sorry

end prob_A_prob_B_l273_273304


namespace pat_earns_per_photo_l273_273579

-- Defining conditions
def minutes_per_shark := 10
def fuel_cost_per_hour := 50
def hunting_hours := 5
def expected_profit := 200

-- Defining intermediate calculations based on the conditions
def sharks_per_hour := 60 / minutes_per_shark
def total_sharks := sharks_per_hour * hunting_hours
def total_fuel_cost := fuel_cost_per_hour * hunting_hours
def total_earnings := expected_profit + total_fuel_cost
def earnings_per_photo := total_earnings / total_sharks

-- Main theorem: Prove that Pat earns $15 for each photo
theorem pat_earns_per_photo : earnings_per_photo = 15 := by
  -- The proof would be here
  sorry

end pat_earns_per_photo_l273_273579


namespace find_equation_of_ellipse_C_l273_273749

def equation_of_ellipse_C (a b : ℝ) : Prop :=
  ∀ x y, (x^2 / a^2 + y^2 / b^2 = 1)

theorem find_equation_of_ellipse_C :
  ∀ (a b : ℝ), (a = 2) → (b = 1) →
  (equation_of_ellipse_C a b) →
  equation_of_ellipse_C 2 1 :=
by
  intros a b ha hb h
  sorry

end find_equation_of_ellipse_C_l273_273749


namespace triangles_congruent_alternative_condition_l273_273830

theorem triangles_congruent_alternative_condition
  (A B C A' B' C' : Type)
  (AB A'B' AC A'C' : ℝ)
  (angleA angleA' : ℝ)
  (h1 : AB = A'B')
  (h2 : angleA = angleA')
  (h3 : AC = A'C') :
  ∃ (triangleABC triangleA'B'C' : Type), (triangleABC = triangleA'B'C') :=
by sorry

end triangles_congruent_alternative_condition_l273_273830


namespace largest_integer_le_1_l273_273488

theorem largest_integer_le_1 (x : ℤ) (h : (2 * x : ℚ) / 7 + 3 / 4 < 8 / 7) : x ≤ 1 :=
sorry

end largest_integer_le_1_l273_273488


namespace betty_boxes_l273_273923

theorem betty_boxes (total_oranges boxes_capacity : ℕ) (h1 : total_oranges = 24) (h2 : boxes_capacity = 8) : total_oranges / boxes_capacity = 3 :=
by sorry

end betty_boxes_l273_273923


namespace vivi_total_yards_l273_273238

theorem vivi_total_yards (spent_checkered spent_plain cost_per_yard : ℝ)
  (h1 : spent_checkered = 75)
  (h2 : spent_plain = 45)
  (h3 : cost_per_yard = 7.50) :
  (spent_checkered / cost_per_yard + spent_plain / cost_per_yard) = 16 :=
by 
  sorry

end vivi_total_yards_l273_273238


namespace find_numbers_l273_273512

theorem find_numbers (N : ℕ) (a b : ℕ) :
  N = 5 * a →
  N = 7 * b →
  N = 35 ∨ N = 70 ∨ N = 105 :=
by
  sorry

end find_numbers_l273_273512


namespace actual_height_is_191_l273_273722

theorem actual_height_is_191 :
  ∀ (n incorrect_avg correct_avg incorrect_height x : ℝ),
  n = 20 ∧ incorrect_avg = 175 ∧ correct_avg = 173 ∧ incorrect_height = 151 ∧
  (n * incorrect_avg - n * correct_avg = x - incorrect_height) →
  x = 191 :=
by
  intros n incorrect_avg correct_avg incorrect_height x h
  -- skip the proof part
  sorry

end actual_height_is_191_l273_273722


namespace fraction_meaningful_iff_l273_273681

theorem fraction_meaningful_iff (x : ℝ) : (∃ y, y = 1 / (x - 1)) ↔ x ≠ 1 :=
by 
  sorry

end fraction_meaningful_iff_l273_273681


namespace screws_weight_l273_273055

theorem screws_weight (x y : ℕ) 
  (h1 : 3 * x + 2 * y = 319) 
  (h2 : 2 * x + 3 * y = 351) : 
  x = 51 ∧ y = 83 :=
by 
  sorry

end screws_weight_l273_273055


namespace calc_inverse_l273_273333

def N : Matrix (Fin 2) (Fin 2) ℚ := ![![3, 0], ![2, -4]]

def c : ℚ := 1 / 12
def d : ℚ := 1 / 12

theorem calc_inverse :
  N⁻¹ = c • N + d • (1 : Matrix (Fin 2) (Fin 2) ℚ) := by
  sorry

end calc_inverse_l273_273333


namespace total_fruits_l273_273716

def num_papaya_trees : ℕ := 2
def num_mango_trees : ℕ := 3
def papayas_per_tree : ℕ := 10
def mangos_per_tree : ℕ := 20

theorem total_fruits : (num_papaya_trees * papayas_per_tree) + (num_mango_trees * mangos_per_tree) = 80 := 
by
  sorry

end total_fruits_l273_273716


namespace f_eq_for_neg_l273_273960

-- Definition of an odd function
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

-- Given conditions
noncomputable def f (x : ℝ) : ℝ :=
  if h : x ≥ 0 then x * (2^(-x) + 1) else x * (2^x + 1)

-- Theorem to prove
theorem f_eq_for_neg (f : ℝ → ℝ) (h1 : is_odd f) (h2 : ∀ x : ℝ, 0 ≤ x → f x = x * (2^(-x) + 1)) :
  ∀ x : ℝ, x < 0 → f x = x * (2^x + 1) :=
by
  intro x hx
  sorry

end f_eq_for_neg_l273_273960


namespace quadratic_eq_proof_l273_273546

noncomputable def quadratic_eq := ∀ (a b : ℝ), 
  (a ≠ 0 → (∃ (x : ℝ), a * x^2 + b * x + 1/4 = 0) →
    (a = b^2 ∧ a = 1 ∧ b = 1) ∨ (a > 1 ∧ 0 < b ∧ b < 1 → ¬ ∃ (x : ℝ), a * x^2 + b * x + 1/4 = 0))

theorem quadratic_eq_proof : quadratic_eq := 
by
  sorry

end quadratic_eq_proof_l273_273546


namespace percentage_increase_twice_l273_273226

theorem percentage_increase_twice {P : ℝ} (x : ℝ) :
  (P * (1 + x)^2) = (P * (1 + 0.6900000000000001)) →
  x = 0.30 :=
by
  sorry

end percentage_increase_twice_l273_273226


namespace esperanza_savings_l273_273706

-- Define the conditions as constants
def rent := 600
def gross_salary := 4840
def food_cost := (3 / 5) * rent
def mortgage_bill := 3 * food_cost
def total_expenses := rent + food_cost + mortgage_bill
def savings := gross_salary - total_expenses
def taxes := (2 / 5) * savings
def actual_savings := savings - taxes

theorem esperanza_savings : actual_savings = 1680 := by
  sorry

end esperanza_savings_l273_273706


namespace James_will_take_7_weeks_l273_273984

def pages_per_hour : ℕ := 5
def hours_per_day : ℕ := 4 - 1
def pages_per_day : ℕ := hours_per_day * pages_per_hour
def total_pages : ℕ := 735
def days_to_finish : ℕ := total_pages / pages_per_day
def weeks_to_finish : ℕ := days_to_finish / 7

theorem James_will_take_7_weeks :
  weeks_to_finish = 7 :=
by
  -- You can add the necessary proof steps here
  sorry

end James_will_take_7_weeks_l273_273984


namespace additional_boxes_needed_l273_273705

theorem additional_boxes_needed
  (total_chocolates : ℕ)
  (chocolates_not_in_box : ℕ)
  (boxes_filled : ℕ)
  (friend_brought_chocolates : ℕ)
  (chocolates_per_box : ℕ)
  (h1 : total_chocolates = 50)
  (h2 : chocolates_not_in_box = 5)
  (h3 : boxes_filled = 3)
  (h4 : friend_brought_chocolates = 25)
  (h5 : chocolates_per_box = 15) :
  (chocolates_not_in_box + friend_brought_chocolates) / chocolates_per_box = 2 :=
by
  sorry
  
end additional_boxes_needed_l273_273705


namespace ellipse_h_k_a_c_sum_l273_273011

theorem ellipse_h_k_a_c_sum :
  let h := -3
  let k := 1
  let a := 4
  let c := 2
  h + k + a + c = 4 :=
by
  let h := -3
  let k := 1
  let a := 4
  let c := 2
  show h + k + a + c = 4
  sorry

end ellipse_h_k_a_c_sum_l273_273011


namespace center_of_circle_point_not_on_circle_l273_273940

-- Definitions and conditions
def circle_eq (x y : ℝ) := x^2 - 6 * x + y^2 + 2 * y - 11 = 0

-- The problem statement split into two separate theorems

-- Proving the center of the circle is (3, -1)
theorem center_of_circle : 
  ∃ h k : ℝ, (∀ x y, circle_eq x y ↔ (x - h)^2 + (y - k)^2 = 21) ∧ (h, k) = (3, -1) := sorry

-- Proving the point (5, -1) does not lie on the circle
theorem point_not_on_circle : ¬ circle_eq 5 (-1) := sorry

end center_of_circle_point_not_on_circle_l273_273940


namespace largest_common_value_less_1000_l273_273346

theorem largest_common_value_less_1000 : ∃ a < 1000, (∃ n m : ℕ, a = 5 + 4 * n ∧ a = 4 + 8 * m) ∧ a = 993 := 
  sorry

end largest_common_value_less_1000_l273_273346


namespace James_will_take_7_weeks_l273_273985

def pages_per_hour : ℕ := 5
def hours_per_day : ℕ := 4 - 1
def pages_per_day : ℕ := hours_per_day * pages_per_hour
def total_pages : ℕ := 735
def days_to_finish : ℕ := total_pages / pages_per_day
def weeks_to_finish : ℕ := days_to_finish / 7

theorem James_will_take_7_weeks :
  weeks_to_finish = 7 :=
by
  -- You can add the necessary proof steps here
  sorry

end James_will_take_7_weeks_l273_273985


namespace Linda_journey_length_l273_273195

theorem Linda_journey_length : 
  (∃ x : ℝ, x = 30 + x * 1/4 + x * 1/7) → x = 840 / 17 :=
by
  sorry

end Linda_journey_length_l273_273195


namespace factorial_divides_exponential_difference_l273_273992

theorem factorial_divides_exponential_difference (n : ℕ) : n! ∣ 2^(2 * n!) - 2^n! :=
by
  sorry

end factorial_divides_exponential_difference_l273_273992


namespace find_largest_natural_number_l273_273795

theorem find_largest_natural_number :
  ∃ n : ℕ, (forall m : ℕ, m > n -> m ^ 300 ≥ 3 ^ 500) ∧ n ^ 300 < 3 ^ 500 :=
  sorry

end find_largest_natural_number_l273_273795


namespace average_of_numbers_not_1380_l273_273487

def numbers : List ℤ := [1200, 1300, 1400, 1520, 1530, 1200]

theorem average_of_numbers_not_1380 :
  let s := numbers.sum
  let n := numbers.length
  n > 0 → (s / n : ℚ) ≠ 1380 := by
  sorry

end average_of_numbers_not_1380_l273_273487


namespace xy_sum_l273_273653

theorem xy_sum (x y : ℝ) (h1 : x^3 + 6 * x^2 + 16 * x = -15) (h2 : y^3 + 6 * y^2 + 16 * y = -17) : x + y = -4 :=
by
  -- The proof can be skipped with 'sorry'
  sorry

end xy_sum_l273_273653


namespace cosine_of_third_angle_l273_273690

theorem cosine_of_third_angle 
  (α β γ : ℝ) 
  (h1 : α < 40 * Real.pi / 180) 
  (h2 : β < 80 * Real.pi / 180) 
  (h3 : Real.sin γ = 5 / 8) :
  Real.cos γ = -Real.sqrt 39 / 8 := 
sorry

end cosine_of_third_angle_l273_273690


namespace distance_between_A_and_B_l273_273400

def average_speed : ℝ := 50  -- Speed in miles per hour

def travel_time : ℝ := 15.8  -- Time in hours

noncomputable def total_distance : ℝ := average_speed * travel_time  -- Distance in miles

theorem distance_between_A_and_B :
  total_distance = 790 :=
by
  sorry

end distance_between_A_and_B_l273_273400


namespace find_a_minus_inverse_l273_273303

-- Definition for the given condition
def condition (a : ℝ) : Prop := a + a⁻¹ = 6

-- Definition for the target value to be proven
def target_value (x : ℝ) : Prop := x = 4 * Real.sqrt 2 ∨ x = -4 * Real.sqrt 2

-- Theorem statement to be proved
theorem find_a_minus_inverse (a : ℝ) (ha : condition a) : target_value (a - a⁻¹) :=
by
  sorry

end find_a_minus_inverse_l273_273303


namespace simplify_fraction_l273_273014

variable {a b c : ℝ} -- assuming a, b, c are real numbers

theorem simplify_fraction (hc : a + b + c ≠ 0) :
  (a^2 + b^2 - c^2 + 2 * a * b) / (a^2 + c^2 - b^2 + 2 * a * c) = (a + b - c) / (a - b + c) :=
sorry

end simplify_fraction_l273_273014


namespace train_length_correct_l273_273519

noncomputable def train_length (v_kmph : ℝ) (t_sec : ℝ) (bridge_length : ℝ) : ℝ :=
  let v_mps := v_kmph / 3.6
  let total_distance := v_mps * t_sec
  total_distance - bridge_length

theorem train_length_correct : train_length 72 12.099 132 = 109.98 :=
by
  sorry

end train_length_correct_l273_273519


namespace eight_sided_dice_theorem_l273_273202
open Nat

noncomputable def eight_sided_dice_probability : ℚ :=
  let total_outcomes := 8^8
  let favorable_outcomes := 8!
  let probability_all_different := favorable_outcomes / total_outcomes
  let probability_at_least_two_same := 1 - probability_all_different
  probability_at_least_two_same

theorem eight_sided_dice_theorem :
  eight_sided_dice_probability = 16736996 / 16777216 := by
    sorry

end eight_sided_dice_theorem_l273_273202


namespace solve_tan_system_l273_273003

open Real

noncomputable def solve_system (a b : ℝ) : Prop :=
  ∃ x y k : ℤ, 
    (b ≠ 1 → 
      x = (a + 2 * (k : ℝ) * π + (arccos ((1 + b) / (1 - b) * cos a))) / 2 ∧
      y = (a - 2 * (k : ℝ) * π - (arccos ((1 + b) / (1 - b) * cos a))) / 2 ∧
      x + y = a ∧ tan x * tan y = b) ∧
    (b ≠ 1 → 
      x = (a + 2 * (k : ℝ) * π - (arccos ((1 + b) / (1 - b) * cos a))) / 2 ∧
      y = (a - 2 * (k : ℝ) * π + (arccos ((1 + b) / (1 - b) * cos a))) / 2 ∧
      x + y = a ∧ tan x * tan y = b) ∧    
    (b = 1 → 
      (∃ m : ℤ, a = (π / 2) + m * π ∧ y = (π / 2) + m * π - x ∧ x + y = a ∧ tan x * tan y = b))

-- Then we need to prove that for any a and b, the solutions satisfy the system:
theorem solve_tan_system (a b : ℝ) : solve_system a b := 
  sorry

end solve_tan_system_l273_273003


namespace small_triangle_area_ratio_l273_273047

theorem small_triangle_area_ratio (a b n : ℝ) 
  (h₀ : a > 0) 
  (h₁ : b > 0) 
  (h₂ : n > 0) 
  (h₃ : ∃ (r s : ℝ), r > 0 ∧ s > 0 ∧ (1/2) * a * r = n * a * b ∧ s = (b^2) / (2 * n * b)) :
  (b^2 / (4 * n)) / (a * b) = 1 / (4 * n) :=
by sorry

end small_triangle_area_ratio_l273_273047


namespace distinct_positive_factors_of_81_l273_273813

theorem distinct_positive_factors_of_81 : 
  let n := 81 in 
  let factors := {d | d > 0 ∧ d ∣ n} in
  n = 3^4 → factors.card = 5 :=
by
  sorry

end distinct_positive_factors_of_81_l273_273813


namespace rectangle_area_is_588_l273_273082

-- Definitions based on the conditions of the problem
def radius : ℝ := 7
def diameter : ℝ := 2 * radius
def width : ℝ := diameter
def length : ℝ := 3 * width

-- The statement to prove that the area of the rectangle is 588
theorem rectangle_area_is_588 : length * width = 588 :=
by
  -- Omitted proof
  sorry

end rectangle_area_is_588_l273_273082


namespace quadratic_has_two_distinct_real_roots_iff_l273_273966

theorem quadratic_has_two_distinct_real_roots_iff (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 * x1 - 2 * x1 + k - 1 = 0 ∧ x2 * x2 - 2 * x2 + k - 1 = 0) ↔ k < 2 :=
by
  sorry

end quadratic_has_two_distinct_real_roots_iff_l273_273966


namespace students_decrement_l273_273156

theorem students_decrement:
  ∃ d : ℕ, ∃ A : ℕ, 
  (∃ n1 n2 n3 n4 n5 : ℕ, n1 = A ∧ n2 = A - d ∧ n3 = A - 2 * d ∧ n4 = A - 3 * d ∧ n5 = A - 4 * d) ∧
  (5 = 5) ∧
  (n1 + n2 + n3 + n4 + n5 = 115) ∧
  (A = 27) → d = 2 :=
by {
  sorry
}

end students_decrement_l273_273156


namespace A_and_B_finish_together_in_20_days_l273_273249

noncomputable def W_B : ℝ := 1 / 30

noncomputable def W_A : ℝ := 1 / 2 * W_B

noncomputable def W_A_plus_B : ℝ := W_A + W_B

theorem A_and_B_finish_together_in_20_days :
  (1 / W_A_plus_B) = 20 :=
by
  sorry

end A_and_B_finish_together_in_20_days_l273_273249


namespace find_k_l273_273823

def f (x : ℝ) := x^2 - 7 * x

theorem find_k : ∃ a h k : ℝ, f x = a * (x - h)^2 + k ∧ k = -49 / 4 := 
sorry

end find_k_l273_273823


namespace price_reduction_l273_273617

variable (x : ℝ)

theorem price_reduction :
  28 * (1 - x) * (1 - x) = 16 :=
sorry

end price_reduction_l273_273617


namespace roots_polynomial_sum_pow_l273_273990

open Real

theorem roots_polynomial_sum_pow (a b : ℝ) (h : a^2 - 5 * a + 6 = 0) (h_b : b^2 - 5 * b + 6 = 0) :
  a^5 + a^4 * b + b^5 = -16674 := by
sorry

end roots_polynomial_sum_pow_l273_273990


namespace find_x_l273_273953

theorem find_x (n : ℕ) (hn : n % 2 = 1) (hpf : ∃ p1 p2 p3 : ℕ, p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧ p1 * p2 * p3 = 9^n - 1 ∧ [p1, p2, p3].contains 61) :
  9^n - 1 = 59048 :=
by
  sorry

end find_x_l273_273953


namespace equation_solution_l273_273024

open Real

theorem equation_solution (x : ℝ) : 
  (x = 4 ∨ x = -1 → 3 * (2 * x - 5) ≠ (2 * x - 5) ^ 2) ∧
  (3 * (2 * x - 5) = (2 * x - 5) ^ 2 → x = 5 / 2 ∨ x = 4) :=
by
  sorry

end equation_solution_l273_273024


namespace greatest_integer_less_than_M_over_100_l273_273420

theorem greatest_integer_less_than_M_over_100 :
  (1 / (Nat.factorial 3 * Nat.factorial 16) +
   1 / (Nat.factorial 4 * Nat.factorial 15) +
   1 / (Nat.factorial 5 * Nat.factorial 14) +
   1 / (Nat.factorial 6 * Nat.factorial 13) +
   1 / (Nat.factorial 7 * Nat.factorial 12) +
   1 / (Nat.factorial 8 * Nat.factorial 11) +
   1 / (Nat.factorial 9 * Nat.factorial 10) = M / (Nat.factorial 2 * Nat.factorial 17)) →
  (⌊(M : ℚ) / 100⌋ = 27) := 
sorry

end greatest_integer_less_than_M_over_100_l273_273420


namespace unique_pair_exists_for_each_n_l273_273201

theorem unique_pair_exists_for_each_n (n : ℕ) (h : n > 0) : 
  ∃! (a b : ℕ), a > 0 ∧ b > 0 ∧ n = (a + b - 1) * (a + b - 2) / 2 + a :=
sorry

end unique_pair_exists_for_each_n_l273_273201


namespace mark_cans_count_l273_273627

-- Given definitions and conditions
def rachel_cans : Nat := x  -- Rachel's cans
def jaydon_cans (x : Nat) : Nat := 5 + 2 * x  -- Jaydon's cans (y)
def mark_cans (y : Nat) : Nat := 4 * y  -- Mark's cans (z)

-- Total cans equation
def total_cans (x y z : Nat) : Prop := x + y + z = 135

-- Main statement to prove
theorem mark_cans_count (x : Nat) (y := jaydon_cans x) (z := mark_cans y) (h : total_cans x y z) : z = 100 :=
sorry

end mark_cans_count_l273_273627


namespace sum_of_possible_values_of_z_l273_273445

theorem sum_of_possible_values_of_z (x y z : ℂ) 
  (h₁ : z^2 + 5 * x = 10 * z)
  (h₂ : y^2 + 5 * z = 10 * y)
  (h₃ : x^2 + 5 * y = 10 * x) :
  z = 0 ∨ z = 9 / 5 := by
  sorry

end sum_of_possible_values_of_z_l273_273445


namespace total_workers_calculation_l273_273347

theorem total_workers_calculation :
  ∀ (N : ℕ), 
  (∀ (total_avg_salary : ℕ) (techs_salary : ℕ) (nontech_avg_salary : ℕ),
    total_avg_salary = 8000 → 
    techs_salary = 7 * 20000 → 
    nontech_avg_salary = 6000 →
    8000 * (7 + N) = 7 * 20000 + N * 6000 →
    (7 + N) = 49) :=
by
  intros
  sorry

end total_workers_calculation_l273_273347


namespace symmetrical_line_range_l273_273125

theorem symmetrical_line_range {k : ℝ} :
  (∀ x y : ℝ, (y = k * x - 1) ∧ (x + y - 1 = 0) → y ≠ -x + 1) → k > 1 ↔ k > 1 :=
by
  sorry

end symmetrical_line_range_l273_273125


namespace hypotenuse_length_l273_273477

-- Definitions derived from conditions
def is_isosceles_right_triangle (a b c : ℝ) : Prop :=
  a = b ∧ a^2 + b^2 = c^2

def perimeter (a b c : ℝ) : ℝ := a + b + c

-- Proposed theorem
theorem hypotenuse_length (a c : ℝ) 
  (h1 : is_isosceles_right_triangle a a c) 
  (h2 : perimeter a a c = 8 + 8 * Real.sqrt 2) :
  c = 4 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_length_l273_273477


namespace max_cake_boxes_in_carton_l273_273375

-- Define the dimensions of the carton as constants
def carton_length := 25
def carton_width := 42
def carton_height := 60

-- Define the dimensions of the cake box as constants
def box_length := 8
def box_width := 7
def box_height := 5

-- Define the volume of the carton and the volume of the cake box
def volume_carton := carton_length * carton_width * carton_height
def volume_box := box_length * box_width * box_height

-- Define the theorem statement
theorem max_cake_boxes_in_carton : 
  (volume_carton / volume_box) = 225 :=
by
  -- The proof is omitted.
  sorry

end max_cake_boxes_in_carton_l273_273375


namespace find_L_for_perfect_square_W_l273_273860

theorem find_L_for_perfect_square_W :
  ∃ L W : ℕ, 1000 < W ∧ W < 2000 ∧ L > 1 ∧ W = 2 * L^3 ∧ ∃ m : ℕ, W = m^2 ∧ L = 8 :=
by sorry

end find_L_for_perfect_square_W_l273_273860


namespace maximum_value_l273_273379

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x
noncomputable def g (x : ℝ) : ℝ := -Real.log x / x

theorem maximum_value (x1 x2 t : ℝ) (h1 : 0 < t) (h2 : f x1 = t) (h3 : g x2 = t) : 
  ∃ x1 x2, (t > 0) ∧ (f x1 = t) ∧ (g x2 = t) ∧ ((x1 / (x2 * Real.exp t)) = 1 / Real.exp 1) := 
sorry

end maximum_value_l273_273379


namespace clock_ticks_6_times_at_6_oclock_l273_273264

theorem clock_ticks_6_times_at_6_oclock
  (h6 : 5 * t = 25)
  (h12 : 11 * t = 55) :
  t = 5 ∧ 6 = 6 :=
by
  sorry

end clock_ticks_6_times_at_6_oclock_l273_273264


namespace problem_prob_l273_273989

definition bernoulli (p : ℝ) (b : Bool) : Prop :=
  if b then p else 1 - p

variable {n : ℕ}

-- Defining the Bernoulli random variables
def xi (k : ℕ) : Prop := bernoulli 0.5 (Bool.ofNat (k % 2))

-- Sums of the xi variables
def S (k : ℕ) : ℝ := if k = 0 then 0 else ∑ i in (finset.range k).filter (λ i, i > 0), (ite (even i) (xi i).to_real))

-- Definition of u
def u (k : ℕ) : ℝ := 2^(-2*k) * (nat.choose (2*k) k)

-- Definition of g
def g (n : ℕ) : ℕ :=
  (finset.range (n+1)).filter (λ k, (k > 0 ∧ even k) ∧ S k = 0).max' 0

-- Proving the required probability
theorem problem_prob (n k : ℕ) (h1 : 1 ≤ k) (h2 : k ≤ n) :
  (probability {g (2*n) = 2*k}) = (u (2*n)) * (u (2*(n-k))) :=
sorry

end problem_prob_l273_273989


namespace find_truck_weight_l273_273520

variable (T Tr : ℝ)

def weight_condition_1 : Prop := T + Tr = 7000
def weight_condition_2 : Prop := Tr = 0.5 * T - 200

theorem find_truck_weight (h1 : weight_condition_1 T Tr) 
                           (h2 : weight_condition_2 T Tr) : 
  T = 4800 :=
sorry

end find_truck_weight_l273_273520


namespace solve_quadratic_simplify_expression_l273_273750

-- 1. Solve the equation 2x^2 - 3x + 1 = 0
theorem solve_quadratic (x : ℝ) :
  2 * x^2 - 3 * x + 1 = 0 ↔ x = 1 / 2 ∨ x = 1 :=
sorry

-- 2. Simplify the given expression
theorem simplify_expression (a b : ℝ) :
  ( (a^2 - b^2) / (a^2 - 2*a*b + b^2) + a / (b - a) ) / (b^2 / (a^2 - a*b)) = a / b :=
sorry

end solve_quadratic_simplify_expression_l273_273750


namespace expression_equality_l273_273789

theorem expression_equality : 1 + 2 / (3 + 4 / 5) = 29 / 19 := by
  sorry

end expression_equality_l273_273789


namespace jackson_total_calories_l273_273161

def lettuce_calories : ℕ := 50
def carrots_calories : ℕ := 2 * lettuce_calories
def dressing_calories : ℕ := 210
def salad_calories : ℕ := lettuce_calories + carrots_calories + dressing_calories

def crust_calories : ℕ := 600
def pepperoni_calories : ℕ := crust_calories / 3
def cheese_calories : ℕ := 400
def pizza_calories : ℕ := crust_calories + pepperoni_calories + cheese_calories

def jackson_salad_fraction : ℚ := 1 / 4
def jackson_pizza_fraction : ℚ := 1 / 5

noncomputable def total_calories : ℚ := 
  jackson_salad_fraction * salad_calories + jackson_pizza_fraction * pizza_calories

theorem jackson_total_calories : total_calories = 330 := by
  sorry

end jackson_total_calories_l273_273161


namespace symmetric_line_equation_l273_273012

-- Define the original line as an equation in ℝ².
def original_line (x y : ℝ) : Prop := x - 2 * y + 1 = 0

-- Define the line of symmetry.
def line_of_symmetry (x : ℝ) : Prop := x = 1

-- The theorem stating the equation of the symmetric line.
theorem symmetric_line_equation (x y : ℝ) :
  original_line x y → line_of_symmetry x → (x + 2 * y - 3 = 0) :=
by
  intros h₁ h₂
  sorry

end symmetric_line_equation_l273_273012


namespace product_of_undefined_roots_l273_273286

theorem product_of_undefined_roots :
  let f (x : ℝ) := (x^2 - 4*x + 4) / (x^2 - 5*x + 6)
  ∀ x : ℝ, (x^2 - 5*x + 6 = 0) → x = 2 ∨ x = 3 →
  (x = 2 ∨ x = 3 → x1 = 2 ∧ x2 = 3 → x1 * x2 = 6) :=
by
  sorry

end product_of_undefined_roots_l273_273286


namespace soccer_team_physics_players_l273_273770

-- Define the number of players on the soccer team
def total_players := 15

-- Define the number of players taking mathematics
def math_players := 10

-- Define the number of players taking both mathematics and physics
def both_subjects_players := 4

-- Define the number of players taking physics
def physics_players := total_players - math_players + both_subjects_players

-- The theorem to prove
theorem soccer_team_physics_players : physics_players = 9 :=
by
  -- using the conditions defined above
  sorry

end soccer_team_physics_players_l273_273770


namespace compute_z_pow_8_l273_273571

noncomputable def z : ℂ := (1 - Real.sqrt 3 * Complex.I) / 2

theorem compute_z_pow_8 : z ^ 8 = -(1 + Real.sqrt 3 * Complex.I) / 2 :=
by
  sorry

end compute_z_pow_8_l273_273571


namespace arithmetic_sequence_max_sum_l273_273437

noncomputable def max_S_n (n : ℕ) (a : ℕ → ℝ) (d : ℝ) : ℝ :=
  n * a 1 + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_max_sum :
  ∃ d, ∃ a : ℕ → ℝ, 
  (a 1 = 1) ∧ (3 * (a 1 + 7 * d) = 5 * (a 1 + 12 * d)) ∧ 
  (∀ n, max_S_n n a d ≤ max_S_n 20 a d) := 
sorry

end arithmetic_sequence_max_sum_l273_273437


namespace line_intersects_parabola_once_l273_273114

theorem line_intersects_parabola_once (k : ℝ) :
  (x = k)
  ∧ (x = -3 * y^2 - 4 * y + 7)
  ∧ (3 * y^2 + 4 * y + (k - 7)) = 0
  ∧ ((4)^2 - 4 * 3 * (k - 7) = 0)
  → k = 25 / 3 := 
by
  sorry

end line_intersects_parabola_once_l273_273114


namespace arithmetic_sequence_general_term_l273_273865

theorem arithmetic_sequence_general_term (x : ℕ)
  (t1 t2 t3 : ℤ)
  (h1 : t1 = x - 1)
  (h2 : t2 = x + 1)
  (h3 : t3 = 2 * x + 3) :
  (∃ a : ℕ → ℤ, a 1 = t1 ∧ a 2 = t2 ∧ a 3 = t3 ∧ ∀ n, a n = 2 * n - 3) := 
sorry

end arithmetic_sequence_general_term_l273_273865


namespace smallest_x_such_that_sum_is_cubic_l273_273654

/-- 
  Given a positive integer x, the sum of the sequence x, x+3, x+6, x+9, and x+12 should be a perfect cube.
  Prove that the smallest such x is 19.
-/
theorem smallest_x_such_that_sum_is_cubic : 
  ∃ (x : ℕ), 0 < x ∧ (∃ k : ℕ, 5 * x + 30 = k^3) ∧ ∀ y : ℕ, 0 < y → (∃ m : ℕ, 5 * y + 30 = m^3) → y ≥ x :=
sorry

end smallest_x_such_that_sum_is_cubic_l273_273654


namespace ball_count_proof_l273_273505

noncomputable def valid_ball_count : ℕ :=
  150

def is_valid_ball_count (N : ℕ) : Prop :=
  80 < N ∧ N ≤ 200 ∧
  (∃ y b w r : ℕ,
    y = Nat.div (12 * N) 100 ∧
    b = Nat.div (20 * N) 100 ∧
    w = 2 * Nat.div N 3 ∧
    r = N - (y + b + w) ∧
    r.mod N = 0 )

theorem ball_count_proof : is_valid_ball_count valid_ball_count :=
by
  -- The proof would be inserted here.
  sorry

end ball_count_proof_l273_273505


namespace simplify_tangent_sum_l273_273209

theorem simplify_tangent_sum :
  (1 + Real.tan (10 * Real.pi / 180)) * (1 + Real.tan (35 * Real.pi / 180)) = 2 := by
  have h1 : Real.tan (45 * Real.pi / 180) = Real.tan ((10 + 35) * Real.pi / 180) := by sorry
  have h2 : Real.tan (45 * Real.pi / 180) = 1 := by sorry
  have h3 : ∀ x y, Real.tan (x + y) = (Real.tan x + Real.tan y) / (1 - Real.tan x * Real.tan y) := by sorry
  sorry

end simplify_tangent_sum_l273_273209


namespace initial_chips_in_bag_l273_273338

-- Definitions based on conditions
def chips_given_to_brother : ℕ := 7
def chips_given_to_sister : ℕ := 5
def chips_kept_by_nancy : ℕ := 10

-- Theorem statement
theorem initial_chips_in_bag (total_chips := chips_given_to_brother + chips_given_to_sister + chips_kept_by_nancy) : total_chips = 22 := 
by 
  -- we state the assertion
  sorry

end initial_chips_in_bag_l273_273338


namespace angle_ACE_is_38_l273_273693

noncomputable def measure_angle_ACE (A B C D E : Type) : Prop :=
  let angle_ABC := 55
  let angle_BCA := 38
  let angle_BAC := 87
  let angle_ABD := 125
  (angle_ABC + angle_ABD = 180) → -- supplementary condition
  (angle_BAC = 87) → -- given angle at BAC
  (let angle_ACB := 180 - angle_BAC - angle_ABC;
   angle_ACB = angle_BCA ∧  -- derived angle at BCA
   angle_ACB = 38) → -- target angle
  (angle_BCA = 38) -- final result that needs to be proven

theorem angle_ACE_is_38 {A B C D E : Type} :
  measure_angle_ACE A B C D E :=
by
  sorry

end angle_ACE_is_38_l273_273693


namespace binary_preceding_and_following_l273_273696

theorem binary_preceding_and_following :
  ∀ (n : ℕ), n = 0b1010100 → (Nat.pred n = 0b1010011 ∧ Nat.succ n = 0b1010101) := by
  intros
  sorry

end binary_preceding_and_following_l273_273696


namespace group_booking_cost_correct_l273_273316

-- Definitions based on the conditions of the problem
def weekday_rate_first_week : ℝ := 18.00
def weekend_rate_first_week : ℝ := 20.00
def weekday_rate_additional_weeks : ℝ := 11.00
def weekend_rate_additional_weeks : ℝ := 13.00
def security_deposit : ℝ := 50.00
def discount_rate : ℝ := 0.10
def group_size : ℝ := 5
def stay_duration : ℕ := 23

-- Computation of total cost
def total_cost (first_week_weekdays : ℕ) (first_week_weekends : ℕ) 
  (additional_week_weekdays : ℕ) (additional_week_weekends : ℕ) 
  (additional_days_weekdays : ℕ) : ℝ := 
  let cost_first_weekdays := first_week_weekdays * weekday_rate_first_week
  let cost_first_weekends := first_week_weekends * weekend_rate_first_week
  let cost_additional_weeks := 2 * (additional_week_weekdays * weekday_rate_additional_weeks + 
                                    additional_week_weekends * weekend_rate_additional_weeks)
  let cost_additional_days := additional_days_weekdays * weekday_rate_additional_weeks
  let total_before_deposit := cost_first_weekdays + cost_first_weekends + 
                              cost_additional_weeks + cost_additional_days
  let total_before_discount := total_before_deposit + security_deposit
  let total_discount := discount_rate * total_before_discount
  total_before_discount - total_discount

-- Proof setup
theorem group_booking_cost_correct :
  total_cost 5 2 5 2 2 = 327.60 :=
by
  -- Placeholder for the proof; steps not required for Lean statement
  sorry

end group_booking_cost_correct_l273_273316


namespace sarah_meets_vegetable_requirement_l273_273800

def daily_vegetable_requirement : ℝ := 2
def total_days : ℕ := 5
def weekly_requirement : ℝ := daily_vegetable_requirement * total_days

def sunday_consumption : ℝ := 3
def monday_consumption : ℝ := 1.5
def tuesday_consumption : ℝ := 1.5
def wednesday_consumption : ℝ := 1.5
def thursday_consumption : ℝ := 2.5

def total_consumption : ℝ := sunday_consumption + monday_consumption + tuesday_consumption + wednesday_consumption + thursday_consumption

theorem sarah_meets_vegetable_requirement : total_consumption = weekly_requirement :=
by
  sorry

end sarah_meets_vegetable_requirement_l273_273800


namespace ratio_of_votes_l273_273771

theorem ratio_of_votes (total_votes ben_votes : ℕ) (h_total : total_votes = 60) (h_ben : ben_votes = 24) :
  (ben_votes : ℚ) / (total_votes - ben_votes : ℚ) = 2 / 3 :=
by sorry

end ratio_of_votes_l273_273771


namespace x_intercept_perpendicular_line_l273_273367

theorem x_intercept_perpendicular_line 
  (x y : ℝ)
  (h1 : 4 * x - 3 * y = 12)
  (h2 : y = - (3 / 4) * x + 4)
  : x = 16 / 3 := 
sorry

end x_intercept_perpendicular_line_l273_273367


namespace optimal_price_l273_273506

def monthly_sales (p : ℝ) : ℝ := 150 - 6 * p
def break_even (p : ℝ) : Prop := 40 ≤ monthly_sales p
def revenue (p : ℝ) : ℝ := p * monthly_sales p

theorem optimal_price : ∃ p : ℝ, p = 13 ∧ p ≤ 30 ∧ break_even p ∧ ∀ q : ℝ, q ≤ 30 → break_even q → revenue p ≥ revenue q := 
by
  sorry

end optimal_price_l273_273506


namespace monotonic_function_a_range_l273_273726

theorem monotonic_function_a_range :
  ∀ (f : ℝ → ℝ) (a : ℝ), 
  (f x = x^2 + (2 * a + 1) * x + 1) →
  (∀ x y, 1 ≤ x → x ≤ 2 → 1 ≤ y → y ≤ 2 → (f x ≤ f y ∨ f x ≥ f y)) ↔ 
  (a ∈ Set.Ici (-3/2) ∪ Set.Iic (-5/2)) := 
sorry

end monotonic_function_a_range_l273_273726


namespace polygon_sides_equation_l273_273042

theorem polygon_sides_equation (n : ℕ) 
  (h1 : (n-2) * 180 = 4 * 360) : n = 10 := 
by 
  sorry

end polygon_sides_equation_l273_273042


namespace cuberoot_3375_sum_l273_273772

theorem cuberoot_3375_sum (a b : ℕ) (h : 3375 = 3^3 * 5^3) (h1 : a = 15) (h2 : b = 1) : a + b = 16 := by
  sorry

end cuberoot_3375_sum_l273_273772


namespace chess_tournament_participants_l273_273746

open Int

theorem chess_tournament_participants (n : ℕ) (h_games: n * (n - 1) / 2 = 190) : n = 20 :=
by
  sorry

end chess_tournament_participants_l273_273746


namespace range_of_a_l273_273820

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < 4 → |x - 1| < a) → a ≥ 3 :=
by {
  sorry
}

end range_of_a_l273_273820


namespace emilia_blueberries_l273_273406

def cartons_needed : Nat := 42
def cartons_strawberries : Nat := 2
def cartons_bought : Nat := 33

def cartons_blueberries (needed : Nat) (strawberries : Nat) (bought : Nat) : Nat :=
  needed - (strawberries + bought)

theorem emilia_blueberries : cartons_blueberries cartons_needed cartons_strawberries cartons_bought = 7 :=
by
  sorry

end emilia_blueberries_l273_273406


namespace max_street_lamps_proof_l273_273763

noncomputable def max_street_lamps_on_road : ℕ := 1998

theorem max_street_lamps_proof (L : ℕ) (l : ℕ)
    (illuminates : ∀ i, i ≤ max_street_lamps_on_road → 
                  (∃ unique_segment : ℕ, unique_segment ≤ L ∧ unique_segment > L - l )):
  max_street_lamps_on_road = 1998 := by
  sorry

end max_street_lamps_proof_l273_273763


namespace find_constant_a_l273_273288

theorem find_constant_a (x y a : ℝ) (h1 : (ax + 4 * y) / (x - 2 * y) = 13) (h2 : x / (2 * y) = 5 / 2) : a = 7 :=
sorry

end find_constant_a_l273_273288


namespace solve_inequality_l273_273244

variable (x : ℝ)

noncomputable def u := 1 + x^2
noncomputable def v := 1 - 3*x^2 + 36*(x^4)
noncomputable def w := 1 - 27*(x^5)

theorem solve_inequality :
  (Real.logBase (u x) (w x) + Real.logBase (v x) (u x) ≤ 1 + Real.logBase (v x) (w x)) ↔
  (x ∈ ({-1/3} ∪ Ioo (-1/(2*Real.sqrt 3)) 0 ∪ Ioo 0 (1/(2*Real.sqrt 3)) ∪ Icc (1/3) (1/Real.root 27 5))) :=
sorry

end solve_inequality_l273_273244


namespace find_width_of_jordan_rectangle_l273_273102

theorem find_width_of_jordan_rectangle (width : ℕ) (h1 : 12 * 15 = 9 * width) : width = 20 :=
by
  sorry

end find_width_of_jordan_rectangle_l273_273102


namespace infinite_n_exists_l273_273995

theorem infinite_n_exists (p : ℕ) (hp : Nat.Prime p) (hp_gt_7 : 7 < p) :
  ∃ᶠ n in at_top, (n ≡ 1 [MOD 2016]) ∧ (p ∣ 2^n + n) :=
sorry

end infinite_n_exists_l273_273995


namespace eldorado_license_plates_count_l273_273440

def is_vowel (c : Char) : Prop := c = 'A' ∨ c = 'E' ∨ c = 'I' ∨ c = 'O' ∨ c = 'U'

def valid_license_plates_count : Nat :=
  let num_vowels := 5
  let num_letters := 26
  let num_digits := 10
  num_vowels * num_letters * num_letters * num_digits * num_digits

theorem eldorado_license_plates_count : valid_license_plates_count = 338000 := by
  sorry

end eldorado_license_plates_count_l273_273440


namespace polygon_sides_l273_273032

theorem polygon_sides (n : ℕ) 
  (h_interior : (n - 2) * 180 = 4 * 360) : 
  n = 10 :=
by
  sorry

end polygon_sides_l273_273032


namespace polynomial_value_at_3_l273_273054

-- Definitions based on given conditions
def f (x : ℕ) : ℕ :=
  5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x

def x := 3

-- Proof statement
theorem polynomial_value_at_3 : f x = 1641 := by
  sorry

end polynomial_value_at_3_l273_273054


namespace g_increasing_in_interval_l273_273296

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1 / 3) * x^3 - a * x^2 + a * x + 2
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := x^2 - 2 * a * x + a
noncomputable def f'' (a : ℝ) (x : ℝ) : ℝ := 2 * x - 2 * a

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f'' a x / x
noncomputable def g' (a : ℝ) (x : ℝ) : ℝ := 1 - a / (x^2)

theorem g_increasing_in_interval (a : ℝ) (h : a < 1) :
  ∀ x : ℝ, 1 < x → 0 < g' a x := by
  sorry

end g_increasing_in_interval_l273_273296


namespace roots_opposite_signs_l273_273946

theorem roots_opposite_signs (p : ℝ) (hp : p > 0) :
  ( ∃ (x₁ x₂ : ℝ), (x₁ * x₂ < 0) ∧ (5 * x₁^2 - 4 * (p + 3) * x₁ + 4 = p^2) ∧  
      (5 * x₂^2 - 4 * (p + 3) * x₂ + 4 = p^2) ) ↔ p > 2 :=
by {
  sorry
}

end roots_opposite_signs_l273_273946


namespace f_of_5_l273_273221

/- The function f(x) is defined by f(x) = x^2 - x. Prove that f(5) = 20. -/
def f (x : ℤ) : ℤ := x^2 - x

theorem f_of_5 : f 5 = 20 := by
  sorry

end f_of_5_l273_273221


namespace laura_needs_to_buy_flour_l273_273569

/--
Laura is baking a cake and needs to buy ingredients.
Flour costs $4, sugar costs $2, butter costs $2.5, and eggs cost $0.5.
The cake is cut into 6 slices. Her mother ate 2 slices.
The dog ate the remaining cake, costing $6.
Prove that Laura needs to buy flour worth $4.
-/
theorem laura_needs_to_buy_flour
  (flour_cost sugar_cost butter_cost eggs_cost dog_ate_cost : ℝ)
  (cake_slices mother_ate_slices dog_ate_slices : ℕ)
  (H_flour : flour_cost = 4)
  (H_sugar : sugar_cost = 2)
  (H_butter : butter_cost = 2.5)
  (H_eggs : eggs_cost = 0.5)
  (H_dog_ate : dog_ate_cost = 6)
  (total_slices : cake_slices = 6)
  (mother_slices : mother_ate_slices = 2)
  (dog_slices : dog_ate_slices = 4) :
  flour_cost = 4 :=
by {
  sorry
}

end laura_needs_to_buy_flour_l273_273569


namespace binary_representation_of_23_l273_273790

theorem binary_representation_of_23 : 23 = 1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0 :=
by
  sorry

end binary_representation_of_23_l273_273790


namespace ratio_of_girls_to_boys_in_biology_class_l273_273046

-- Defining the conditions
def physicsClassStudents : Nat := 200
def biologyClassStudents := physicsClassStudents / 2
def boysInBiologyClass : Nat := 25
def girlsInBiologyClass := biologyClassStudents - boysInBiologyClass

-- Statement of the problem
theorem ratio_of_girls_to_boys_in_biology_class : girlsInBiologyClass / boysInBiologyClass = 3 :=
by
  sorry

end ratio_of_girls_to_boys_in_biology_class_l273_273046


namespace age_difference_l273_273433

theorem age_difference (x : ℕ) (older_age younger_age : ℕ) 
  (h1 : 3 * x = older_age)
  (h2 : 2 * x = younger_age)
  (h3 : older_age + younger_age = 60) : 
  older_age - younger_age = 12 := 
by
  sorry

end age_difference_l273_273433


namespace polynomial_coeff_diff_l273_273148

theorem polynomial_coeff_diff (a b c d e f : ℝ) :
  ((3*x + 1)^5 = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f) →
  (a - b + c - d + e - f = 32) :=
by
  sorry

end polynomial_coeff_diff_l273_273148


namespace students_in_first_class_l273_273005

variable (x : ℕ)
variable (avg_marks_first_class : ℕ := 40)
variable (num_students_second_class : ℕ := 28)
variable (avg_marks_second_class : ℕ := 60)
variable (avg_marks_all : ℕ := 54)

theorem students_in_first_class : (40 * x + 60 * 28) / (x + 28) = 54 → x = 12 := 
by 
  sorry

end students_in_first_class_l273_273005


namespace total_time_simultaneous_l273_273848

def total_time_bread1 : Nat := 30 + 120 + 20 + 120 + 10 + 30 + 30 + 15
def total_time_bread2 : Nat := 90 + 15 + 20 + 25 + 10
def total_time_bread3 : Nat := 40 + 100 + 5 + 110 + 15 + 5 + 25 + 20

theorem total_time_simultaneous :
  max (max total_time_bread1 total_time_bread2) total_time_bread3 = 375 :=
by
  sorry

end total_time_simultaneous_l273_273848
