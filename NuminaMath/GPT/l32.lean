import Mathlib

namespace star_polygon_n_value_l32_32650

theorem star_polygon_n_value (n : ℕ) (A B : ℕ → ℝ) (h1 : ∀ i, A i = B i - 20)
    (h2 : 360 = n * 20) : n = 18 :=
by {
  sorry
}

end star_polygon_n_value_l32_32650


namespace longest_side_of_rectangle_l32_32858

theorem longest_side_of_rectangle 
    (l w : ℝ) 
    (h1 : 2 * l + 2 * w = 240) 
    (h2 : l * w = 2400) : 
    max l w = 80 :=
by sorry

end longest_side_of_rectangle_l32_32858


namespace jars_of_plum_jelly_sold_l32_32199

theorem jars_of_plum_jelly_sold (P R G S : ℕ) (h1 : R = 2 * P) (h2 : G = 3 * R) (h3 : G = 2 * S) (h4 : S = 18) : P = 6 := by
  sorry

end jars_of_plum_jelly_sold_l32_32199


namespace circle_condition_l32_32347

theorem circle_condition (m : ℝ) : (∃ x y : ℝ, x^2 + y^2 + 4*x - 2*y + 5*m = 0) →
  (m < 1) :=
by
  sorry

end circle_condition_l32_32347


namespace cos_pi_minus_double_alpha_l32_32820

theorem cos_pi_minus_double_alpha (α : ℝ) (h : Real.sin α = 2 / 3) : Real.cos (π - 2 * α) = -1 / 9 :=
by
  sorry

end cos_pi_minus_double_alpha_l32_32820


namespace decorations_left_to_put_up_l32_32212

variable (S B W P C T : Nat)
variable (h₁ : S = 12)
variable (h₂ : B = 4)
variable (h₃ : W = 12)
variable (h₄ : P = 2 * W)
variable (h₅ : C = 1)
variable (h₆ : T = 83)

theorem decorations_left_to_put_up (h₁ : S = 12) (h₂ : B = 4) (h₃ : W = 12) (h₄ : P = 2 * W) (h₅ : C = 1) (h₆ : T = 83) :
  T - (S + B + W + P + C) = 30 := sorry

end decorations_left_to_put_up_l32_32212


namespace cube_difference_l32_32246

variable (x : ℝ)

theorem cube_difference (h : x - 1/x = 5) : x^3 - 1/x^3 = 120 := by
  sorry

end cube_difference_l32_32246


namespace function_monotonic_decreasing_l32_32824

open Real

/-- Given the function y = f(x) (x ∈ ℝ), the slope of the tangent line at any point (x₀, f(x₀))
    is k = (x₀ - 3) * (x₀ + 1)^2. Then, prove that the function is monotonically decreasing
    for x₀ ≤ 3. -/
theorem function_monotonic_decreasing (f : ℝ → ℝ) (x₀ : ℝ)
  (h_slope : deriv f x₀ = (x₀ - 3) * (x₀ + 1)^2) :
  ∀ x, (x ∈ Iic 3) → deriv f x ≤ 0 :=
by
  intros x hx
  rw [h_slope]
  sorry

end function_monotonic_decreasing_l32_32824


namespace nautical_mile_to_land_mile_l32_32804

theorem nautical_mile_to_land_mile 
    (speed_one_sail : ℕ := 25) 
    (speed_two_sails : ℕ := 50) 
    (travel_time_one_sail : ℕ := 4) 
    (travel_time_two_sails : ℕ := 4)
    (total_distance : ℕ := 345) : 
    ∃ (x : ℚ), x = 1.15 ∧ 
    total_distance = travel_time_one_sail * speed_one_sail * x +
                    travel_time_two_sails * speed_two_sails * x := 
by
  sorry

end nautical_mile_to_land_mile_l32_32804


namespace mean_height_basketball_team_l32_32510

def heights : List ℕ :=
  [58, 59, 60, 62, 63, 65, 65, 68, 70, 71, 71, 72, 76, 76, 78, 79, 79]

def mean_height (l : List ℕ) : ℕ :=
  l.sum / l.length

theorem mean_height_basketball_team :
  mean_height heights = 70 := by
  sorry

end mean_height_basketball_team_l32_32510


namespace isosceles_triangle_perimeter_l32_32517

theorem isosceles_triangle_perimeter (a b c : ℕ) (h1 : a = 6) (h2 : b = 8) (h3 : a = c ∨ b = c) :
  a + b + c = 22 :=
by
  -- This part of the proof is simplified using the conditions
  sorry

end isosceles_triangle_perimeter_l32_32517


namespace abs_lt_2_sufficient_not_necessary_l32_32907

theorem abs_lt_2_sufficient_not_necessary (x : ℝ) :
  (|x| < 2 → x^2 - x - 6 < 0) ∧ ¬ (x^2 - x - 6 < 0 → |x| < 2) :=
by {
  sorry
}

end abs_lt_2_sufficient_not_necessary_l32_32907


namespace cow_difference_l32_32539

variables (A M R : Nat)

def Aaron_has_four_times_as_many_cows_as_Matthews : Prop := A = 4 * M
def Matthews_has_cows : Prop := M = 60
def Total_cows_for_three := A + M + R = 570

theorem cow_difference (h1 : Aaron_has_four_times_as_many_cows_as_Matthews A M) 
                       (h2 : Matthews_has_cows M)
                       (h3 : Total_cows_for_three A M R) :
  (A + M) - R = 30 :=
by
  sorry

end cow_difference_l32_32539


namespace frac_difference_l32_32818

theorem frac_difference (m n : ℝ) (h : m^2 - n^2 = m * n) : (n / m) - (m / n) = -1 :=
sorry

end frac_difference_l32_32818


namespace pressure_increases_when_block_submerged_l32_32780

-- Definitions and conditions
variables (P_0 : ℝ) (ρ : ℝ) (g : ℝ) (h_0 : ℝ) (h_1 : ℝ)
hypothesis (h1_gt_h0 : h_1 > h_0)

-- The proof goal
theorem pressure_increases_when_block_submerged (P_0 ρ g h_0 h_1 : ℝ) (h1_gt_h0 : h_1 > h_0) : 
  let P := P_0 + ρ * g * h_0 in
  let P_1 := P_0 + ρ * g * h_1 in
  P_1 > P := sorry

end pressure_increases_when_block_submerged_l32_32780


namespace arithmetic_seq_contains_geometric_seq_l32_32056

theorem arithmetic_seq_contains_geometric_seq (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  (∃ (ns : ℕ → ℕ) (k : ℝ), k ≠ 1 ∧ (∀ n, a + b * (ns (n + 1)) = k * (a + b * (ns n)))) ↔ (∃ (q : ℚ), a = q * b) :=
sorry

end arithmetic_seq_contains_geometric_seq_l32_32056


namespace point_in_first_quadrant_l32_32142

-- Define the system of equations
def equations (x y : ℝ) : Prop :=
  x + y = 2 ∧ x - y = 1

-- State the theorem
theorem point_in_first_quadrant (x y : ℝ) (h : equations x y) : x > 0 ∧ y > 0 := by
  sorry

end point_in_first_quadrant_l32_32142


namespace arithmetic_geometric_seq_l32_32146

theorem arithmetic_geometric_seq (A B C D : ℤ) (h_arith_seq : A + C = 2 * B)
  (h_geom_seq : B * D = C * C) (h_frac : 7 * B = 3 * C) (h_posB : B > 0)
  (h_posC : C > 0) (h_posD : D > 0) : A + B + C + D = 76 :=
sorry

end arithmetic_geometric_seq_l32_32146


namespace acute_triangle_iff_sum_of_squares_l32_32474

theorem acute_triangle_iff_sum_of_squares (a b c R : ℝ) 
  (hRpos : R > 0) 
  (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) : 
  (∀ α β γ, (a = 2 * R * Real.sin α) ∧ (b = 2 * R * Real.sin β) ∧ (c = 2 * R * Real.sin γ) → 
   (α < Real.pi / 2 ∧ β < Real.pi / 2 ∧ γ < Real.pi / 2)) ↔ 
  (a^2 + b^2 + c^2 > 8 * R^2) :=
sorry

end acute_triangle_iff_sum_of_squares_l32_32474


namespace cannot_be_divided_into_parallelograms_l32_32450

noncomputable def isosceles_triangle (a : ℝ) (θ : ℝ) (hx : 0 < θ ∧ θ < π / 2) : Triangle :=
{ base := a,
  height := a * (cos θ) }

noncomputable def resulting_figure (a : ℝ) (θ : ℝ) 
  (hx : 0 < θ ∧ θ < π / 2) : Polygon :=
{ square := Square (side_length := a),
  triangles := λ s ∈ (Square.edges) => isosceles_triangle a θ hx }

noncomputable def can_be_split_into_parallelograms (fig : Polygon) : Prop :=
fig.decomposition.all (λ piece, is_parallelogram piece)

theorem cannot_be_divided_into_parallelograms 
  (a : ℝ) (θ : ℝ) (hx : 0 < θ ∧ θ < π / 2) : 
  ¬ can_be_split_into_parallelograms (resulting_figure a θ hx) :=
sorry

end cannot_be_divided_into_parallelograms_l32_32450


namespace prob_product_multiple_of_4_l32_32712

theorem prob_product_multiple_of_4 :
  (∑ i in ({1, 2, 3, 4, 5, 6, 7, 8} : Finset ℕ), ∑ j in ({1, 2, 3, 4, 5} : Finset ℕ), 
    if (i * j) % 4 = 0 then (1 / 8) * (1 / 5) else 0) = 2 / 5 := 
by 
  sorry

end prob_product_multiple_of_4_l32_32712


namespace correct_equation_l32_32756

theorem correct_equation (x : ℤ) : 232 + x = 3 * (146 - x) :=
sorry

end correct_equation_l32_32756


namespace sum_of_k_with_distinct_integer_roots_l32_32367

theorem sum_of_k_with_distinct_integer_roots :
  ∑ k in {k : ℤ | ∃ p q : ℤ, p ≠ q ∧ 3*p*q = 12 ∧ p + q = k/3}, k = 0 :=
by
  sorry

end sum_of_k_with_distinct_integer_roots_l32_32367


namespace value_fraction_l32_32737

variables {x y : ℝ}
variables (hx : x ≠ 0) (hy : y ≠ 0) (h : (4 * x + 2 * y) / (x - 4 * y) = 3)

theorem value_fraction : (x + 4 * y) / (4 * x - y) = 10 / 57 :=
by { sorry }

end value_fraction_l32_32737


namespace maximum_busses_l32_32099

open Finset

-- Definitions of conditions
def bus_stops : ℕ := 9
def stops_per_bus : ℕ := 3

-- Conditions stating that any two buses share at most one common stop
def no_two_buses_share_more_than_one_common_stop (buses : Finset (Finset (Fin (bus_stops)))) : Prop :=
  ∀ b1 b2 ∈ buses, b1 ≠ b2 → (b1 ∩ b2).card ≤ 1

-- Conditions stating that each bus stops at exactly three stops
def each_bus_stops_at_exactly_three_stops (buses : Finset (Finset (Fin (bus_stops)))) : Prop :=
  ∀ b ∈ buses, b.card = stops_per_bus

-- Theorems stating the maximum number of buses possible under given conditions
theorem maximum_busses (buses : Finset (Finset (Fin (bus_stops)))) :
  no_two_buses_share_more_than_one_common_stop buses →
  each_bus_stops_at_exactly_three_stops buses →
  buses.card ≤ 10 := by
  sorry

end maximum_busses_l32_32099


namespace solve_inequality_l32_32749

theorem solve_inequality (x : ℝ) : ((x + 3) ^ 2 < 1) ↔ (-4 < x ∧ x < -2) := by
  sorry

end solve_inequality_l32_32749


namespace journey_speed_l32_32536

theorem journey_speed (t_total : ℝ) (d_total : ℝ) (d_half : ℝ) (v_half2 : ℝ) (time_half2 : ℝ) (time_total : ℝ) (v_half1 : ℝ) :
  t_total = 5 ∧ d_total = 112 ∧ d_half = d_total / 2 ∧ v_half2 = 24 ∧ time_half2 = d_half / v_half2 ∧ time_total = t_total - time_half2 ∧ v_half1 = d_half / time_total → v_half1 = 21 :=
by
  intros h
  sorry

end journey_speed_l32_32536


namespace village_population_l32_32023

variable (Px : ℕ)
variable (py : ℕ := 42000)
variable (years : ℕ := 16)
variable (rate_decrease_x : ℕ := 1200)
variable (rate_increase_y : ℕ := 800)

theorem village_population (Px : ℕ) (py : ℕ := 42000)
  (years : ℕ := 16) (rate_decrease_x : ℕ := 1200)
  (rate_increase_y : ℕ := 800) :
  Px - rate_decrease_x * years = py + rate_increase_y * years → Px = 74000 := by
  sorry

end village_population_l32_32023


namespace cubic_identity_l32_32250

theorem cubic_identity (x : ℝ) (h : x - (1 / x) = 5) : x^3 - (1 / x^3) = 140 := 
  sorry

end cubic_identity_l32_32250


namespace volume_of_parallelepiped_l32_32073

theorem volume_of_parallelepiped (x y z : ℝ)
  (h1 : (x^2 + y^2) * z^2 = 13)
  (h2 : (y^2 + z^2) * x^2 = 40)
  (h3 : (x^2 + z^2) * y^2 = 45) :
  x * y * z = 6 :=
by 
  sorry

end volume_of_parallelepiped_l32_32073


namespace enclosed_area_four_circles_l32_32424

/-- 
Given four circles each of radius 7 cm such that each circle touches two other circles, 
prove that the area of the space enclosed by the four pieces is 196 cm² - 49π cm².
-/
theorem enclosed_area_four_circles :
  let r := 7 in
  let square_side := 2 * r in
  let square_area := square_side * square_side in
  let circle_area := π * r * r in
  square_area - circle_area = 196 - 49 * π :=
by
  let r := 7
  let square_side := 2 * r
  let square_area := square_side * square_side
  let circle_area := π * r * r
  show square_area - circle_area = 196 - 49 * π
  sorry

end enclosed_area_four_circles_l32_32424


namespace stops_time_proof_l32_32030

variable (departure_time arrival_time driving_time stop_time_in_minutes : ℕ)
variable (h_departure : departure_time = 7 * 60)
variable (h_arrival : arrival_time = 20 * 60)
variable (h_driving : driving_time = 12 * 60)
variable (total_minutes := arrival_time - departure_time)

theorem stops_time_proof :
  stop_time_in_minutes = (total_minutes - driving_time) := by
  sorry

end stops_time_proof_l32_32030


namespace min_total_fund_Required_l32_32516

noncomputable def sell_price_A (x : ℕ) : ℕ := x + 10
noncomputable def cost_A (x : ℕ) : ℕ := 600
noncomputable def cost_B (x : ℕ) : ℕ := 400

def num_barrels_A_B_purchased (x : ℕ) := cost_A x / (sell_price_A x) = cost_B x / x

noncomputable def total_cost (m : ℕ) : ℕ := 10 * m + 10000

theorem min_total_fund_Required (price_A price_B m total : ℕ) :
  price_B = 20 →
  price_A = 30 →
  price_A = price_B + 10 →
  (num_barrels_A_B_purchased price_B) →
  total = total_cost m →
  m = 250 →
  total = 12500 := 
by 
  intros h1 h2 h3 h4 h5 h6
  sorry

end min_total_fund_Required_l32_32516


namespace intersection_M_N_l32_32581

def M : Set ℝ := { x | |x - 2| ≤ 1 }
def N : Set ℝ := { x | x^2 - x - 6 ≥ 0 }

theorem intersection_M_N : M ∩ N = {3} := by
  sorry

end intersection_M_N_l32_32581


namespace units_digit_of_3_pow_y_l32_32477

theorem units_digit_of_3_pow_y
    (x : ℕ)
    (h1 : (2^3)^x = 4096)
    (y : ℕ)
    (h2 : y = x^3) :
    (3^y) % 10 = 1 :=
by
  sorry

end units_digit_of_3_pow_y_l32_32477


namespace total_dots_not_visible_l32_32227

-- Define the total dot sum for each die
def sum_of_dots_per_die : Nat := 1 + 2 + 3 + 4 + 5 + 6

-- Define the total number of dice
def number_of_dice : Nat := 4

-- Calculate the total dot sum for all dice
def total_dots_all_dice : Nat := sum_of_dots_per_die * number_of_dice

-- Sum of visible dots
def sum_of_visible_dots : Nat := 1 + 1 + 2 + 2 + 3 + 3 + 4 + 5 + 6 + 6

-- Prove the total dots not visible
theorem total_dots_not_visible : total_dots_all_dice - sum_of_visible_dots = 51 := by
  sorry

end total_dots_not_visible_l32_32227


namespace tree_height_end_of_2_years_l32_32203

theorem tree_height_end_of_2_years (h4 : ℕ → ℕ)
  (h_tripling : ∀ n, h4 (n + 1) = 3 * h4 n)
  (h4_at_4 : h4 4 = 81) :
  h4 2 = 9 :=
by
  sorry

end tree_height_end_of_2_years_l32_32203


namespace cos_555_value_l32_32412

noncomputable def cos_555_equals_neg_sqrt6_add_sqrt2_div4 : Prop :=
  (Real.cos 555 = -((Real.sqrt 6 + Real.sqrt 2) / 4))

theorem cos_555_value : cos_555_equals_neg_sqrt6_add_sqrt2_div4 :=
  by sorry

end cos_555_value_l32_32412


namespace seventh_term_arithmetic_sequence_l32_32752

theorem seventh_term_arithmetic_sequence (a d : ℚ)
  (h1 : a + (a + d) + (a + 2 * d) + (a + 3 * d) + (a + 4 * d) = 20)
  (h2 : a + 5 * d = 8) :
  a + 6 * d = 28 / 3 := 
sorry

end seventh_term_arithmetic_sequence_l32_32752


namespace a_2023_value_l32_32552

theorem a_2023_value :
  ∀ (a : ℕ → ℚ),
  a 1 = 5 ∧
  a 2 = 5 / 11 ∧
  (∀ n, 3 ≤ n → a n = (a (n - 2)) * (a (n - 1)) / (3 * (a (n - 2)) - (a (n - 1)))) →
  a 2023 = 5 / 10114 ∧ 5 + 10114 = 10119 :=
by
  sorry

end a_2023_value_l32_32552


namespace num_choices_l32_32788

theorem num_choices (classes scenic_spots : ℕ) (h_classes : classes = 4) (h_scenic_spots : scenic_spots = 3) :
  (scenic_spots ^ classes) = 81 :=
by
  -- The detailed proof goes here
  sorry

end num_choices_l32_32788


namespace angle_D_measure_l32_32621

theorem angle_D_measure (E D F : ℝ) (h1 : E + D + F = 180) (h2 : E = 30) (h3 : D = 2 * F) : D = 100 :=
by
  -- The proof is not required, only the statement
  sorry

end angle_D_measure_l32_32621


namespace sum_of_all_ks_l32_32371

theorem sum_of_all_ks (k : ℤ) :
  (∃ r s : ℤ, r ≠ s ∧ (3 * x^2 - k * x + 12 = 0) ∧ (r + s = k / 3) ∧ (r * s = 4)) → (k = 15 ∨ k = -15) → k + k = 0 :=
by
  sorry

end sum_of_all_ks_l32_32371


namespace smallest_scalene_triangle_perimeter_is_prime_l32_32651

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def consecutive_primes (p1 p2 p3 : ℕ) : Prop :=
  p1 < p2 ∧ p2 < p3 ∧ is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧
  (p2 = p1 + 2) ∧ (p3 = p1 + 6)

noncomputable def smallest_prime_perimeter : ℕ :=
  5 + 7 + 11

theorem smallest_scalene_triangle_perimeter_is_prime :
  ∃ (p1 p2 p3 : ℕ), p1 < p2 ∧ p2 < p3 ∧ consecutive_primes p1 p2 p3 ∧ is_prime (p1 + p2 + p3) ∧ (p1 + p2 + p3 = smallest_prime_perimeter) :=
by 
  sorry

end smallest_scalene_triangle_perimeter_is_prime_l32_32651


namespace tony_belinda_combined_age_l32_32017

/-- Tony and Belinda have a combined age. Belinda is 8 more than twice Tony's age. 
Tony is 16 years old and Belinda is 40 years old. What is their combined age? -/
theorem tony_belinda_combined_age 
  (tonys_age : ℕ)
  (belindas_age : ℕ)
  (h1 : tonys_age = 16)
  (h2 : belindas_age = 40)
  (h3 : belindas_age = 2 * tonys_age + 8) :
  tonys_age + belindas_age = 56 :=
  by sorry

end tony_belinda_combined_age_l32_32017


namespace coloring_even_conditional_l32_32319

-- Define the problem parameters and constraints
def number_of_colorings (n : Nat) (even_red : Bool) (even_yellow : Bool) : Nat :=
  sorry  -- This function would contain the detailed computational logic.

-- Define the main theorem statement
theorem coloring_even_conditional (n : ℕ) (h1 : n > 0) : ∃ C : Nat, number_of_colorings n true true = C := 
by
  sorry  -- The proof would go here.


end coloring_even_conditional_l32_32319


namespace part_I_part_II_l32_32857

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) - (a * x) / (x + 1)

theorem part_I (a : ℝ) : (∀ x, f a 0 ≤ f a x) → a = 1 := by
  sorry

theorem part_II (a : ℝ) : (∀ x > 0, f a x > 0) → a ≤ 1 := by
  sorry

end part_I_part_II_l32_32857


namespace num_kids_eq_3_l32_32314

def mom_eyes : ℕ := 1
def dad_eyes : ℕ := 3
def kid_eyes : ℕ := 4
def total_eyes : ℕ := 16

theorem num_kids_eq_3 : ∃ k : ℕ, 1 + 3 + 4 * k = 16 ∧ k = 3 := by
  sorry

end num_kids_eq_3_l32_32314


namespace AlbertTookAwayCandies_l32_32124

-- Define the parameters and conditions given in the problem
def PatriciaStartCandies : ℕ := 76
def PatriciaEndCandies : ℕ := 71

-- Define the statement that proves the number of candies Albert took away
theorem AlbertTookAwayCandies :
  PatriciaStartCandies - PatriciaEndCandies = 5 := by
  sorry

end AlbertTookAwayCandies_l32_32124


namespace cartesian_equation_of_c_min_distance_to_line_l32_32845

-- Problem 1: Cartesian equation of curve C'
theorem cartesian_equation_of_c' (x y x' y' : ℝ)
  (hC : x^2 + y^2 = 1)
  (htrans1 : x' = 2 * x)
  (htrans2 : y' = Real.sqrt 3 * y) :
  x'^2 / 4 + y'^2 / 3 = 1 := 
sorry

-- Problem 2: Minimum distance from point P to the line l
theorem min_distance_to_line (theta : ℝ) (x y x' y' : ℝ) (P : (ℝ × ℝ))
  (hC' : x'^2 / 4 + y'^2 / 3 = 1)
  (htrans1 : x' = 2 * (Real.cos theta))
  (htrans2 : y' = Real.sqrt 3 * (Real.sin theta))
  (hP : P = (2 * Real.cos theta, Real.sqrt 3 * Real.sin theta))
  (hline : P.1 ≠ 0) : 
  (Real.abs (2 * Real.sqrt 3 * Real.cos theta + Real.sqrt 3 * Real.sin theta - 6)) / 2 = (6 - Real.sqrt 15) / 2 ∧
  P = (4 * Real.sqrt 5 / 5, Real.sqrt 15 / 5) :=
sorry

end cartesian_equation_of_c_min_distance_to_line_l32_32845


namespace courtier_selection_l32_32489

theorem courtier_selection :
  ∀ (dogs : Finset ℕ) (courtiers : ℕ → Finset ℕ),
  dogs.card = 100 ∧ (∀ i, card (courtiers i) = 3) ∧ (∀ i j, i ≠ j → (courtiers i ∩ courtiers j).card = 2) →
  ∃ i j, i ≠ j ∧ courtiers i = courtiers j :=
by
  intros dogs courtiers h
  sorry

end courtier_selection_l32_32489


namespace space_shuttle_speed_l32_32398

-- Define the conditions in Lean
def speed_kmph : ℕ := 43200 -- Speed in kilometers per hour
def seconds_per_hour : ℕ := 60 * 60 -- Number of seconds in an hour

-- Define the proof problem
theorem space_shuttle_speed :
  speed_kmph / seconds_per_hour = 12 := by
  sorry

end space_shuttle_speed_l32_32398


namespace trigonometric_identity_l32_32688

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 3) : 
  (Real.sin (Real.pi - α) - Real.sin (Real.pi / 2 + α)) / 
  (Real.cos (3 * Real.pi / 2 - α) + 2 * Real.cos (-Real.pi + α)) = -2 / 5 := 
by
  sorry

end trigonometric_identity_l32_32688


namespace angle_between_NE_and_SW_l32_32391

theorem angle_between_NE_and_SW
  (n : ℕ) (hn : n = 12)
  (total_degrees : ℚ) (htotal : total_degrees = 360)
  (spaced_rays : ℚ) (hspaced : spaced_rays = total_degrees / n)
  (angles_between_NE_SW : ℕ) (hangles : angles_between_NE_SW = 4) :
  (angles_between_NE_SW * spaced_rays = 120) :=
by
  rw [htotal, hn] at hspaced
  rw [hangles]
  rw [hspaced]
  sorry

end angle_between_NE_and_SW_l32_32391


namespace log_base_change_l32_32379

theorem log_base_change (log_16_32 log_16_inv2: ℝ) : 
  (log_16_32 * log_16_inv2 = -5 / 16) :=
by
  sorry

end log_base_change_l32_32379


namespace condition_for_a_l32_32972

theorem condition_for_a (a : ℝ) :
  (∀ x : ℤ, (x < 0 → (x + a) / 2 ≥ 1) → (x = -1 ∨ x = -2)) ↔ 4 ≤ a ∧ a < 5 :=
by
  sorry

end condition_for_a_l32_32972


namespace minimum_value_x2_y2_l32_32684

variable {x y : ℝ}

theorem minimum_value_x2_y2 (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy : x * y = 1) : x^2 + y^2 = 2 :=
sorry

end minimum_value_x2_y2_l32_32684


namespace exist_two_courtiers_with_same_selection_l32_32488

noncomputable theory

-- Definitions
def num_dogs := 100
def num_courtiers := 100
def selected_dogs (courtier : Fin num_courtiers) : Finset (Fin num_dogs) := sorry
def num_selected_dogs := 3 

-- Conditions
axiom three_dogs_selected_each (c : Fin num_courtiers) : selected_dogs c.card = num_selected_dogs

axiom two_common_dogs (c₁ c₂ : Fin num_courtiers) (h : c₁ ≠ c₂) : (selected_dogs c₁ ∩ selected_dogs c₂).card ≥ 2

-- Proof goal
theorem exist_two_courtiers_with_same_selection : 
  ∃ (c₁ c₂ : Fin num_courtiers), c₁ ≠ c₂ ∧ selected_dogs c₁ = selected_dogs c₂ :=
sorry

end exist_two_courtiers_with_same_selection_l32_32488


namespace angle_between_planes_l32_32813

-- Definitions from problem conditions
def plane1 (r : ℝ) : ℝ × ℝ × ℝ → Prop := λ (x : ℝ × ℝ × ℝ), 4 * x.1 + 3 * x.3 - 2 = 0
def plane2 (r : ℝ) : ℝ × ℝ × ℝ → Prop := λ (x : ℝ × ℝ × ℝ), x.1 + 2 * x.2 + 2 * x.3 + 5 = 0

def n1 : ℝ × ℝ × ℝ := (4, 0, 3)
def n2 : ℝ × ℝ × ℝ := (1, 2, 2)

-- Proof goal:
theorem angle_between_planes : arccos ((n1.1 * n2.1 + n1.2 * n2.2 + n1.3 * n2.3) / 
  (real.sqrt (n1.1^2 + n1.2^2 + n1.3^2) * real.sqrt (n2.1^2 + n2.2^2 + n2.3^2))) 
  = real.to_real (48 + 11 / 60 + 23 / 3600) * real.pi / 180 :=
by sorry

end angle_between_planes_l32_32813


namespace total_travel_time_l32_32179

-- Defining the conditions
def car_travel_180_miles_in_4_hours : Prop :=
  180 / 4 = 45

def car_travel_135_miles_additional_time : Prop :=
  135 / 45 = 3

-- The main statement to be proved
theorem total_travel_time : car_travel_180_miles_in_4_hours ∧ car_travel_135_miles_additional_time → 4 + 3 = 7 := by
  sorry

end total_travel_time_l32_32179


namespace sum_of_all_ks_l32_32370

theorem sum_of_all_ks (k : ℤ) :
  (∃ r s : ℤ, r ≠ s ∧ (3 * x^2 - k * x + 12 = 0) ∧ (r + s = k / 3) ∧ (r * s = 4)) → (k = 15 ∨ k = -15) → k + k = 0 :=
by
  sorry

end sum_of_all_ks_l32_32370


namespace evaluate_power_l32_32677

theorem evaluate_power (a b : ℝ) (m n : ℝ) (h1 : a = b^m) (h2 : ∀ x y z : ℝ, (x^y)^z = x^(y*z)) : a^(n/m) = b^n :=
by
  sorry

example : 81^(5/4) = 243 := evaluate_power 81 3 4 5
  (by norm_num) -- Simplification for 81 = 3^4
  (by norm_num []) -- Using the power of a power rule

end evaluate_power_l32_32677


namespace count_FourDigitNumsWithThousandsDigitFive_is_1000_l32_32439

def count_FourDigitNumsWithThousandsDigitFive : Nat :=
  let minNum := 5000
  let maxNum := 5999
  maxNum - minNum + 1

theorem count_FourDigitNumsWithThousandsDigitFive_is_1000 :
  count_FourDigitNumsWithThousandsDigitFive = 1000 :=
by
  sorry

end count_FourDigitNumsWithThousandsDigitFive_is_1000_l32_32439


namespace perimeter_circumradius_ratio_neq_l32_32152

-- Define the properties for the equilateral triangle
def Triangle (A K R P : ℝ) : Prop :=
  P = 3 * A ∧ K = A^2 * Real.sqrt 3 / 4 ∧ R = A * Real.sqrt 3 / 3

-- Define the properties for the square
def Square (b k r p : ℝ) : Prop :=
  p = 4 * b ∧ k = b^2 ∧ r = b * Real.sqrt 2 / 2

-- Main statement to prove
theorem perimeter_circumradius_ratio_neq 
  (A b K R P k r p : ℝ)
  (hT : Triangle A K R P) 
  (hS : Square b k r p) :
  P / p ≠ R / r := 
by
  rcases hT with ⟨hP, hK, hR⟩
  rcases hS with ⟨hp, hk, hr⟩
  sorry

end perimeter_circumradius_ratio_neq_l32_32152


namespace box_third_dimension_length_l32_32649

noncomputable def box_height (num_cubes : ℕ) (cube_volume : ℝ) (length : ℝ) (width : ℝ) : ℝ :=
  let total_volume := num_cubes * cube_volume
  total_volume / (length * width)

theorem box_third_dimension_length (num_cubes : ℕ) (cube_volume : ℝ) (length : ℝ) (width : ℝ)
  (h_num_cubes : num_cubes = 24)
  (h_cube_volume : cube_volume = 27)
  (h_length : length = 8)
  (h_width : width = 12) :
  box_height num_cubes cube_volume length width = 6.75 :=
by {
  -- proof skipped
  sorry
}

end box_third_dimension_length_l32_32649


namespace G_at_16_l32_32718

noncomputable def G : ℝ → ℝ := sorry

-- Condition 1: G is a polynomial, implicitly stated
-- Condition 2: Given G(8) = 21
axiom G_at_8 : G 8 = 21

-- Condition 3: Given that
axiom G_fraction_condition : ∀ (x : ℝ), 
  (x^2 + 6*x + 8) ≠ 0 ∧ ((x+4)*(x+2)) ≠ 0 → 
  (G (2*x) / G (x+4) = 4 - (16*x + 32) / (x^2 + 6*x + 8))

-- The problem: Prove G(16) = 90
theorem G_at_16 : G 16 = 90 := 
sorry

end G_at_16_l32_32718


namespace ratio_pat_mark_l32_32342

theorem ratio_pat_mark (P K M : ℕ) (h1 : P + K + M = 180) 
  (h2 : P = 2 * K) (h3 : M = K + 100) : P / gcd P M = 1 ∧ M / gcd P M = 3 := by
  sorry

end ratio_pat_mark_l32_32342


namespace annette_miscalculation_l32_32803

theorem annette_miscalculation :
  let x := 6
  let y := 3
  let x' := 5
  let y' := 4
  x' - y' = 1 :=
by
  let x := 6
  let y := 3
  let x' := 5
  let y' := 4
  sorry

end annette_miscalculation_l32_32803


namespace cylinder_quadrilateral_intersection_l32_32770

-- Definitions for the intersection types for the geometric solids.
def can_intersect_plane_as_quadrilateral (solid : Type) : Prop :=
  solid = Cylinder

theorem cylinder_quadrilateral_intersection :
  (∀ solid, can_intersect_plane_as_quadrilateral solid ↔ solid = Cylinder) :=
begin
  intros solid,
  split,
  { -- Prove that if a solid can intersect plane as a quadrilateral, then it is a cylinder
    intro h,
    exact h
  },
  { -- Prove that a cylinder can intersect plane as a quadrilateral
    intro h,
    rw h,
    refl,
  }
end

end cylinder_quadrilateral_intersection_l32_32770


namespace result_more_than_half_l32_32975

theorem result_more_than_half (x : ℕ) (h : x = 4) : (2 * x + 5) - (x / 2) = 11 := by
  sorry

end result_more_than_half_l32_32975


namespace decaf_percentage_total_l32_32790

-- Defining the initial conditions
def initial_stock : ℝ := 400
def initial_decaf_percentage : ℝ := 0.30
def new_stock : ℝ := 100
def new_decaf_percentage : ℝ := 0.60

-- Given conditions
def amount_initial_decaf := initial_decaf_percentage * initial_stock
def amount_new_decaf := new_decaf_percentage * new_stock
def total_decaf := amount_initial_decaf + amount_new_decaf
def total_stock := initial_stock + new_stock

-- Prove the percentage of decaffeinated coffee in the total stock
theorem decaf_percentage_total : 
  (total_decaf / total_stock) * 100 = 36 := by
  sorry

end decaf_percentage_total_l32_32790


namespace fern_pays_228_11_usd_l32_32559

open Real

noncomputable def high_heels_price : ℝ := 66
noncomputable def ballet_slippers_price : ℝ := (2 / 3) * high_heels_price
noncomputable def purse_price : ℝ := 49.5
noncomputable def scarf_price : ℝ := 27.5
noncomputable def high_heels_discount : ℝ := 0.10 * high_heels_price
noncomputable def discounted_high_heels_price : ℝ := high_heels_price - high_heels_discount
noncomputable def total_cost_before_tax : ℝ := discounted_high_heels_price + ballet_slippers_price + purse_price + scarf_price
noncomputable def sales_tax : ℝ := 0.075 * total_cost_before_tax
noncomputable def total_cost_after_tax : ℝ := total_cost_before_tax + sales_tax
noncomputable def exchange_rate : ℝ := 1 / 0.85
noncomputable def total_cost_in_usd : ℝ := total_cost_after_tax * exchange_rate

theorem fern_pays_228_11_usd: total_cost_in_usd = 228.11 := by
  sorry

end fern_pays_228_11_usd_l32_32559


namespace multiplication_problems_l32_32069

theorem multiplication_problems :
  (30 * 30 = 900) ∧
  (30 * 40 = 1200) ∧
  (40 * 70 = 2800) ∧
  (50 * 70 = 3500) ∧
  (60 * 70 = 4200) ∧
  (4 * 90 = 360) :=
by sorry

end multiplication_problems_l32_32069


namespace largest_divisor_of_n_squared_divisible_by_72_l32_32279

theorem largest_divisor_of_n_squared_divisible_by_72
    (n : ℕ) (h1 : n > 0) (h2 : 72 ∣ n^2) : 12 ∣ n :=
by {
    sorry
}

end largest_divisor_of_n_squared_divisible_by_72_l32_32279


namespace abs_ineq_solution_set_l32_32509

theorem abs_ineq_solution_set {x : ℝ} : (|x - 1| ≤ 2) ↔ (-1 ≤ x ∧ x ≤ 3) :=
by
  sorry

end abs_ineq_solution_set_l32_32509


namespace not_divisible_1961_1963_divisible_1963_1965_l32_32603

def is_divisible_by_three (n : Nat) : Prop :=
  n % 3 = 0

theorem not_divisible_1961_1963 : ¬ is_divisible_by_three (1961 * 1963) :=
by
  sorry

theorem divisible_1963_1965 : is_divisible_by_three (1963 * 1965) :=
by
  sorry

end not_divisible_1961_1963_divisible_1963_1965_l32_32603


namespace cost_function_segments_l32_32602

def C (n : ℕ) : ℕ :=
  if h : 1 ≤ n ∧ n ≤ 10 then 10 * n
  else if h : 10 < n then 8 * n - 40
  else 0

theorem cost_function_segments :
  (∀ n, 1 ≤ n ∧ n ≤ 10 → C n = 10 * n) ∧
  (∀ n, 10 < n → C n = 8 * n - 40) ∧
  (∀ n, C n = if (1 ≤ n ∧ n ≤ 10) then 10 * n else if (10 < n) then 8 * n - 40 else 0) ∧
  ∃ n₁ n₂, (1 ≤ n₁ ∧ n₁ ≤ 10) ∧ (10 < n₂ ∧ n₂ ≤ 20) ∧ C n₁ = 10 * n₁ ∧ C n₂ = 8 * n₂ - 40 :=
by
  sorry

end cost_function_segments_l32_32602


namespace percentage_of_masters_l32_32921

-- Definition of given conditions
def average_points_juniors := 22
def average_points_masters := 47
def overall_average_points := 41

-- Problem statement
theorem percentage_of_masters (x y : ℕ) (hx : x ≥ 0) (hy : y ≥ 0) 
    (h_avg_juniors : 22 * x = average_points_juniors * x)
    (h_avg_masters : 47 * y = average_points_masters * y)
    (h_overall_average : 22 * x + 47 * y = overall_average_points * (x + y)) : 
    (y : ℚ) / (x + y) * 100 = 76 := 
sorry

end percentage_of_masters_l32_32921


namespace tony_water_drink_l32_32020

theorem tony_water_drink (W : ℝ) (h : W - 0.04 * W = 48) : W = 50 :=
sorry

end tony_water_drink_l32_32020


namespace find_divisor_l32_32170

theorem find_divisor 
  (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) (divisor : ℕ) :
  (dividend = 172) → (quotient = 10) → (remainder = 2) → (dividend = (divisor * quotient) + remainder) → divisor = 17 :=
by 
  sorry

end find_divisor_l32_32170


namespace Riverdale_High_students_l32_32058

theorem Riverdale_High_students
  (f j : ℕ)
  (h1 : (3 / 7) * f + (3 / 4) * j = 234)
  (h2 : f + j = 420) :
  f = 64 ∧ j = 356 := by
  sorry

end Riverdale_High_students_l32_32058


namespace bug_probability_at_A_after_8_meters_l32_32593

noncomputable def P : ℕ → ℚ 
| 0 => 1
| (n + 1) => (1 / 3) * (1 - P n)

theorem bug_probability_at_A_after_8_meters :
  P 8 = 547 / 2187 := 
sorry

end bug_probability_at_A_after_8_meters_l32_32593


namespace min_value_of_expression_l32_32324

open Real

theorem min_value_of_expression (α β : ℝ) :
  (3 * cos α + 4 * sin β - 7)^2 + (3 * sin α + 4 * cos β - 12)^2 ≥ 36 :=
by
  sorry

end min_value_of_expression_l32_32324


namespace find_x_l32_32051

theorem find_x (x : ℝ) (hx1 : x > 0) 
  (h1 : 0.20 * x + 14 = (1 / 3) * ((3 / 4) * x + 21)) : x = 140 :=
sorry

end find_x_l32_32051


namespace algebra_expression_evaluation_l32_32076

theorem algebra_expression_evaluation (a b : ℝ) (h : a + 3 * b = 4) : 2 * a + 6 * b - 1 = 7 := by
  sorry

end algebra_expression_evaluation_l32_32076


namespace find_N_l32_32287

theorem find_N : ∃ N : ℕ, (∃ x y : ℕ, 
(10 * x + y = N) ∧ 
(4 * x + 2 * y = N / 2) ∧ 
(1 ≤ x) ∧ (x ≤ 9) ∧ 
(0 ≤ y) ∧ (y ≤ 9)) ∧
(N = 32 ∨ N = 64 ∨ N = 96) :=
by {
    have h1 : ∀ x y : ℕ, 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 → 4 * x + 2 * y = (10 * x + y) / 2 → true, sorry,
    exact h1
}

end find_N_l32_32287


namespace part1_part2_l32_32083

-- Define the function f
def f (x m : ℝ) : ℝ := abs (x + m) + abs (2 * x - 1)

-- First part of the problem
theorem part1 (x : ℝ) : f x (-1) ≤ 2 ↔ 0 ≤ x ∧ x ≤ 4 / 3 := 
by sorry

-- Second part of the problem
theorem part2 (m : ℝ) : 
  (∀ x, 3 / 4 ≤ x → x ≤ 2 → f x m ≤ abs (2 * x + 1)) ↔ (-11 / 4) ≤ m ∧ m ≤ 0 := 
by sorry

end part1_part2_l32_32083


namespace plates_difference_l32_32129

noncomputable def num_pots_angela : ℕ := 20
noncomputable def num_plates_angela (P : ℕ) := P
noncomputable def num_cutlery_angela (P : ℕ) := P / 2
noncomputable def num_pots_sharon : ℕ := 10
noncomputable def num_plates_sharon (P : ℕ) := 3 * P - 20
noncomputable def num_cutlery_sharon (P : ℕ) := P
noncomputable def total_kitchen_supplies_sharon (P : ℕ) := 
  num_pots_sharon + num_plates_sharon P + num_cutlery_sharon P

theorem plates_difference (P : ℕ) 
  (hP: num_plates_angela P > 3 * num_pots_angela) 
  (h_supplies: total_kitchen_supplies_sharon P = 254) :
  P - 3 * num_pots_angela = 6 := 
sorry

end plates_difference_l32_32129


namespace boys_or_girls_rink_l32_32981

variables (Class : Type) (is_boy : Class → Prop) (is_girl : Class → Prop) (visited_rink : Class → Prop) (met_at_rink : Class → Class → Prop)

-- Every student in the class visited the rink at least once.
axiom all_students_visited : ∀ (s : Class), visited_rink s

-- Every boy met every girl at the rink.
axiom boys_meet_girls : ∀ (b g : Class), is_boy b → is_girl g → met_at_rink b g

-- Prove that there exists a time when all the boys, or all the girls were simultaneously on the rink.
theorem boys_or_girls_rink : ∃ (t : Prop), (∀ b, is_boy b → visited_rink b) ∨ (∀ g, is_girl g → visited_rink g) :=
sorry

end boys_or_girls_rink_l32_32981


namespace solution_set_of_quadratic_inequality_l32_32066

theorem solution_set_of_quadratic_inequality :
  {x : ℝ | x^2 + x - 2 ≥ 0} = {x : ℝ | x ≤ -2} ∪ {x : ℝ | x ≥ 1} :=
by
  sorry

end solution_set_of_quadratic_inequality_l32_32066


namespace find_N_l32_32285

theorem find_N : ∃ N : ℕ, (∃ x y : ℕ, 
(10 * x + y = N) ∧ 
(4 * x + 2 * y = N / 2) ∧ 
(1 ≤ x) ∧ (x ≤ 9) ∧ 
(0 ≤ y) ∧ (y ≤ 9)) ∧
(N = 32 ∨ N = 64 ∨ N = 96) :=
by {
    have h1 : ∀ x y : ℕ, 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 → 4 * x + 2 * y = (10 * x + y) / 2 → true, sorry,
    exact h1
}

end find_N_l32_32285


namespace value_of_x_plus_y_l32_32276

theorem value_of_x_plus_y (x y : ℝ) (h : (x + 1)^2 + |y - 6| = 0) : x + y = 5 :=
by
sorry

end value_of_x_plus_y_l32_32276


namespace positive_number_satisfying_condition_l32_32628

theorem positive_number_satisfying_condition :
  ∃ x : ℝ, x > 0 ∧ x^2 = 64 ∧ x = 8 := by sorry

end positive_number_satisfying_condition_l32_32628


namespace probability_at_least_two_heads_is_11_over_16_l32_32659

noncomputable def probability_of_heads : ℚ := 1 / 2

noncomputable def probability_at_least_two_heads : ℚ :=
  1 - (nat.choose 4 0 * probability_of_heads^4 + nat.choose 4 1 * probability_of_heads^4)

theorem probability_at_least_two_heads_is_11_over_16 :
  probability_at_least_two_heads = 11 / 16 := by
  sorry

end probability_at_least_two_heads_is_11_over_16_l32_32659


namespace family_percentage_eaten_after_dinner_l32_32112

theorem family_percentage_eaten_after_dinner
  (total_brownies : ℕ)
  (children_percentage : ℚ)
  (left_over_brownies : ℕ)
  (lorraine_extra_brownie : ℕ)
  (remaining_percentage : ℚ) :
  total_brownies = 16 →
  children_percentage = 0.25 →
  lorraine_extra_brownie = 1 →
  left_over_brownies = 5 →
  remaining_percentage = 50 := by
  sorry

end family_percentage_eaten_after_dinner_l32_32112


namespace average_donation_proof_l32_32588

noncomputable def average_donation (total_people : ℝ) (donated_200 : ℝ) (donated_100 : ℝ) (donated_50 : ℝ) : ℝ :=
  let proportion_200 := donated_200 / total_people
  let proportion_100 := donated_100 / total_people
  let proportion_50 := donated_50 / total_people
  let total_donation := (200 * proportion_200) + (100 * proportion_100) + (50 * proportion_50)
  total_donation

theorem average_donation_proof 
  (total_people : ℝ)
  (donated_200 donated_100 donated_50 : ℝ)
  (h1 : proportion_200 = 1 / 10)
  (h2 : proportion_100 = 3 / 4)
  (h3 : proportion_50 = 1 - proportion_200 - proportion_100) :
  average_donation total_people donated_200 donated_100 donated_50 = 102.5 :=
  by 
    sorry

end average_donation_proof_l32_32588


namespace max_buses_constraint_satisfied_l32_32098

def max_buses_9_bus_stops : ℕ := 10

theorem max_buses_constraint_satisfied :
  ∀ (buses : ℕ),
    (∀ (b : buses), b.stops.length = 3) ∧
    (∀ (b1 b2 : buses), b1 ≠ b2 → (b1.stops ∩ b2.stops).length ≤ 1) →
    buses ≤ max_buses_9_bus_stops :=
by
  sorry

end max_buses_constraint_satisfied_l32_32098


namespace bricks_required_l32_32530

theorem bricks_required (L_courtyard W_courtyard L_brick W_brick : Real)
  (hcourtyard : L_courtyard = 35) 
  (wcourtyard : W_courtyard = 24) 
  (hbrick_len : L_brick = 0.15) 
  (hbrick_wid : W_brick = 0.08) : 
  (L_courtyard * W_courtyard) / (L_brick * W_brick) = 70000 := 
by
  sorry

end bricks_required_l32_32530


namespace equilateral_triangle_l32_32475

theorem equilateral_triangle
  (a b c : ℝ) (α β γ : ℝ)
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b)
  (h7 : α + β + γ = π)
  (h8 : a = 2 * Real.sin α)
  (h9 : b = 2 * Real.sin β)
  (h10 : c = 2 * Real.sin γ)
  (h11 : a * (1 - 2 * Real.cos α) + b * (1 - 2 * Real.cos β) + c * (1 - 2 * Real.cos γ) = 0) :
  a = b ∧ b = c :=
sorry

end equilateral_triangle_l32_32475


namespace jake_peaches_l32_32709

noncomputable def steven_peaches : ℕ := 15
noncomputable def jake_fewer : ℕ := 7

theorem jake_peaches : steven_peaches - jake_fewer = 8 :=
by
  sorry

end jake_peaches_l32_32709


namespace max_area_of_triangle_l32_32723

noncomputable def area_of_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  1/2 * abs (A.1 * B.2 + B.1 * C.2 + C.1 * A.2 - A.2 * B.1 - B.2 * C.1 - C.2 * A.1)

theorem max_area_of_triangle : 
  let A := (1 : ℝ, 0 : ℝ)
  let B := (4 : ℝ, 3 : ℝ)
  ∃ p, 1 ≤ p ∧ p ≤ 4 ∧ 
  let C := (p, p^2 - 4 * p + 3) in 
  area_of_triangle A B C = 27 / 8 :=
by sorry

end max_area_of_triangle_l32_32723


namespace y_value_l32_32648

theorem y_value {y : ℝ} (h1 : (0, 2) = (0, 2))
                (h2 : (3, y) = (3, y))
                (h3 : dist (0, 2) (3, y) = 10)
                (h4 : y > 0) :
                y = 2 + Real.sqrt 91 := by
  sorry

end y_value_l32_32648


namespace number_of_valid_trapezoids_l32_32349

noncomputable def calculate_number_of_trapezoids : ℕ :=
  let rows_1 := 7
  let rows_2 := 9
  let unit_spacing := 1
  let height := 2
  -- Here, we should encode the actual combinatorial calculation as per the problem solution
  -- but for the Lean 4 statement, we will provide the correct answer directly.
  361

theorem number_of_valid_trapezoids :
  calculate_number_of_trapezoids = 361 :=
sorry

end number_of_valid_trapezoids_l32_32349


namespace distance_MF_l32_32261

-- Define the conditions for the problem
def parabola (x y : ℝ) : Prop :=
  y^2 = 8 * x

def focus : (ℝ × ℝ) := (2, 0)

def lies_on_parabola (M : ℝ × ℝ) : Prop :=
  parabola M.1 M.2

def distance_to_line (M : ℝ × ℝ) (line_x : ℝ) : ℝ :=
  abs (M.1 - line_x)

def point_M_conditions (M : ℝ × ℝ) : Prop :=
  distance_to_line M (-3) = 6 ∧ lies_on_parabola M

-- The final proof problem statement in Lean
theorem distance_MF (M : ℝ × ℝ) (h : point_M_conditions M) : dist M focus = 5 :=
by sorry

end distance_MF_l32_32261


namespace none_of_these_l32_32413

def y_values_match (f : ℕ → ℕ) : Prop :=
  f 0 = 200 ∧ f 1 = 140 ∧ f 2 = 80 ∧ f 3 = 20 ∧ f 4 = 0

theorem none_of_these :
  ¬ (∃ f : ℕ → ℕ, 
    (∀ x, f x = 200 - 15 * x ∨ 
    f x = 200 - 20 * x + 5 * x^2 ∨ 
    f x = 200 - 30 * x + 10 * x^2 ∨ 
    f x = 150 - 50 * x) ∧ 
    y_values_match f) :=
by sorry

end none_of_these_l32_32413


namespace payment_option1_payment_option2_cost_effective_option_most_cost_effective_plan_l32_32180

variable (x : ℕ)
variable (hx : x > 10)

noncomputable def option1_payment (x : ℕ) : ℕ := 200 * x + 8000
noncomputable def option2_payment (x : ℕ) : ℕ := 180 * x + 9000

theorem payment_option1 (x : ℕ) (hx : x > 10) : option1_payment x = 200 * x + 8000 :=
by sorry

theorem payment_option2 (x : ℕ) (hx : x > 10) : option2_payment x = 180 * x + 9000 :=
by sorry

theorem cost_effective_option (x : ℕ) (hx : x > 10) (h30 : x = 30) : option1_payment 30 < option2_payment 30 :=
by sorry

theorem most_cost_effective_plan (h30 : x = 30) : (10000 + 3600 = 13600) :=
by sorry

end payment_option1_payment_option2_cost_effective_option_most_cost_effective_plan_l32_32180


namespace basket_white_ball_probability_l32_32927

noncomputable def basket_problem_proof : Prop :=
  let P_A := 1 / 2
  let P_B := 1 / 2
  let P_W_given_A := 2 / 5
  let P_W_given_B := 1 / 4
  let P_W := P_A * P_W_given_A + P_B * P_W_given_B
  let P_A_given_W := (P_A * P_W_given_A) / P_W
  P_A_given_W = 8 / 13

theorem basket_white_ball_probability :
  basket_problem_proof :=
  sorry

end basket_white_ball_probability_l32_32927


namespace remainder_when_sum_divided_by_11_l32_32562

def sum_of_large_numbers : ℕ :=
  100001 + 100002 + 100003 + 100004 + 100005 + 100006 + 100007

theorem remainder_when_sum_divided_by_11 : sum_of_large_numbers % 11 = 2 := by
  sorry

end remainder_when_sum_divided_by_11_l32_32562


namespace problem1_problem2_problem3_problem4_l32_32550

theorem problem1 : 23 + (-16) - (-7) = 14 := by
  sorry

theorem problem2 : (3/4 - 7/8 - 5/12) * (-24) = 13 := by
  sorry

theorem problem3 : (7/4 - 7/8 - 7/12) / (-7/8) + (-7/8) / (7/4 - 7/8 - 7/12) = -(10/3) := by
  sorry

theorem problem4 : -1 ^ 4 - (1 - 0.5) * (1/3) * (2 - (-3) ^ 2) = 1/6 := by 
  sorry

end problem1_problem2_problem3_problem4_l32_32550


namespace original_purchase_price_first_commodity_l32_32348

theorem original_purchase_price_first_commodity (x y : ℝ) 
  (h1 : 1.07 * (x + y) = 827) 
  (h2 : x = y + 127) : 
  x = 450.415 :=
  sorry

end original_purchase_price_first_commodity_l32_32348


namespace geom_seq_sum_5_terms_l32_32843

theorem geom_seq_sum_5_terms (a : ℕ → ℝ) (q : ℝ) (h1 : a 4 = 8 * a 1) (h2 : 2 * (a 2 + 1) = a 1 + a 3) (h_q : q = 2) :
    a 1 * (1 - q^5) / (1 - q) = 62 :=
by
    sorry

end geom_seq_sum_5_terms_l32_32843


namespace find_n_l32_32292

theorem find_n (N : ℕ) (hN : 10 ≤ N ∧ N < 100)
  (h : ∃ a b : ℕ, N = 10 * a + b ∧ 4 * a + 2 * b = N / 2) : 
  N = 32 ∨ N = 64 ∨ N = 96 :=
sorry

end find_n_l32_32292


namespace equal_distribution_arithmetic_seq_l32_32640

theorem equal_distribution_arithmetic_seq :
  ∃ (a1 d : ℚ), (a1 + (a1 + d) = (a1 + 2 * d) + (a1 + 3 * d) + (a1 + 4 * d)) ∧ 
                (5 * a1 + 10 / 2 * d = 5) ∧ 
                (a1 = 4 / 3) :=
by
  sorry

end equal_distribution_arithmetic_seq_l32_32640


namespace factorize_difference_of_squares_l32_32219

theorem factorize_difference_of_squares (a : ℝ) : a^2 - 9 = (a + 3) * (a - 3) :=
sorry

end factorize_difference_of_squares_l32_32219


namespace intersection_of_sets_l32_32463

variable (A B : Set ℝ) (x : ℝ)

def setA : Set ℝ := { x | x > 0 }
def setB : Set ℝ := { x | -1 < x ∧ x ≤ 2 }

theorem intersection_of_sets : A ∩ B = { x | 0 < x ∧ x ≤ 2 } :=
by
  sorry

end intersection_of_sets_l32_32463


namespace sin_double_angle_of_tan_l32_32277

-- Given condition: tan(alpha) = 2
-- To prove: sin(2 * alpha) = 4/5
theorem sin_double_angle_of_tan (α : ℝ) (h : Real.tan α = 2) : Real.sin (2 * α) = 4 / 5 :=
  sorry

end sin_double_angle_of_tan_l32_32277


namespace rectangle_probability_l32_32643

theorem rectangle_probability (m n : ℕ) (h_m : m = 1003^2) (h_n : n = 1003 * 2005) :
  (1 - (m / n)) = 1002 / 2005 :=
by
  sorry

end rectangle_probability_l32_32643


namespace polynomial_unique_factorization_l32_32330

variables {k : Type*} [Field k]

theorem polynomial_unique_factorization (f : Polynomial k) :
  ∃! (l : Multiset (Polynomial k)), (∀ g ∈ l, Irreducible g) ∧ (l.Prod = f) :=
sorry

end polynomial_unique_factorization_l32_32330


namespace no_solution_system_l32_32553

noncomputable def system_inconsistent : Prop :=
  ∀ x y : ℝ, ¬ (3 * x - 4 * y = 8 ∧ 6 * x - 8 * y = 12)

theorem no_solution_system : system_inconsistent :=
by
  sorry

end no_solution_system_l32_32553


namespace determine_x_l32_32068

theorem determine_x (x : ℝ) (hx : 0 < x) (h : x * ⌊x⌋ = 72) : x = 9 :=
sorry

end determine_x_l32_32068


namespace p_sufficient_for_q_q_not_necessary_for_p_l32_32078

variable (x : ℝ)

def p := |x - 2| < 1
def q := 1 < x ∧ x < 5

theorem p_sufficient_for_q : p x → q x :=
by sorry

theorem q_not_necessary_for_p : ¬ (q x → p x) :=
by sorry

end p_sufficient_for_q_q_not_necessary_for_p_l32_32078


namespace fraction_spent_on_food_l32_32916

variable (salary : ℝ) (food_fraction rent_fraction clothes_fraction remaining_amount : ℝ)
variable (salary_condition : salary = 180000)
variable (rent_fraction_condition : rent_fraction = 1/10)
variable (clothes_fraction_condition : clothes_fraction = 3/5)
variable (remaining_amount_condition : remaining_amount = 18000)

theorem fraction_spent_on_food :
  rent_fraction * salary + clothes_fraction * salary + food_fraction * salary + remaining_amount = salary →
  food_fraction = 1/5 :=
by
  intros
  sorry

end fraction_spent_on_food_l32_32916


namespace measure_of_angle_C_l32_32305

-- Define the conditions using Lean 4 constructs
variable (a b c : ℝ)
variable (A B C : ℝ) -- Measures of angles in triangle ABC
variable (triangle_ABC : (a * a + b * b - c * c = a * b))

-- Statement of the proof problem
theorem measure_of_angle_C (h : a^2 + b^2 - c^2 = ab) (h2 : 0 < C ∧ C < π) : C = π / 3 :=
by
  -- Proof will go here but is omitted with sorry
  sorry

end measure_of_angle_C_l32_32305


namespace find_x_for_f_eq_f_inv_l32_32213

def f (x : ℝ) : ℝ := 3 * x - 8

noncomputable def f_inv (x : ℝ) : ℝ := (x + 8) / 3

theorem find_x_for_f_eq_f_inv : ∃ x : ℝ, f x = f_inv x ∧ x = 4 :=
by
  sorry

end find_x_for_f_eq_f_inv_l32_32213


namespace triangle_side_square_sum_eq_three_times_centroid_dist_square_sum_l32_32080

theorem triangle_side_square_sum_eq_three_times_centroid_dist_square_sum
  {A B C O : EuclideanSpace ℝ (Fin 2)}
  (h_centroid : O = (1/3 : ℝ) • (A + B + C)) :
  (dist A B)^2 + (dist B C)^2 + (dist C A)^2 =
  3 * ((dist O A)^2 + (dist O B)^2 + (dist O C)^2) :=
sorry

end triangle_side_square_sum_eq_three_times_centroid_dist_square_sum_l32_32080


namespace range_of_a_for_inequality_l32_32973

theorem range_of_a_for_inequality (a : ℝ) : (∀ x : ℝ, |x + 2| + |x - 1| ≥ a) → a ≤ 3 :=
by
  intro h
  sorry

end range_of_a_for_inequality_l32_32973


namespace calc_fraction_l32_32546

theorem calc_fraction : (36 + 12) / (6 - 3) = 16 :=
by
  sorry

end calc_fraction_l32_32546


namespace total_bottle_caps_l32_32216

-- Define the conditions
def bottle_caps_per_child : ℕ := 5
def number_of_children : ℕ := 9

-- Define the main statement to be proven
theorem total_bottle_caps : bottle_caps_per_child * number_of_children = 45 :=
by sorry

end total_bottle_caps_l32_32216


namespace y_intercept_of_line_l32_32771

theorem y_intercept_of_line : ∃ y : ℝ, 2 * 0 - 3 * y = 6 ∧ y = -2 :=
by
  exists (-2)
  split
  . simp
  . rfl

end y_intercept_of_line_l32_32771


namespace powerThreeExpression_l32_32243

theorem powerThreeExpression (x : ℝ) (h : x - (1 / x) = 5) : x^3 - (1 / (x^3)) = 140 := sorry

end powerThreeExpression_l32_32243


namespace collective_land_area_l32_32134

theorem collective_land_area 
  (C W : ℕ) 
  (h1 : 42 * C + 35 * W = 165200)
  (h2 : W = 3400)
  : C + W = 4500 :=
sorry

end collective_land_area_l32_32134


namespace number_of_non_empty_proper_subsets_of_A_range_of_m_for_A_superset_B_l32_32966

-- Definitions for the sets A and B
def A : Set Int := {x | x^2 - 3 * x - 10 <= 0}
def B (m : Int) : Set Int := {x | m - 1 <= x ∧ x <= 2 * m + 1}

-- Proof for the number of non-empty proper subsets of A
theorem number_of_non_empty_proper_subsets_of_A (x : Int) (h : x ∈ A) : 2^(8 : Nat) - 2 = 254 := by
  sorry

-- Proof for the range of m such that A ⊇ B
theorem range_of_m_for_A_superset_B (m : Int) : (∀ x, x ∈ B m → x ∈ A) ↔ (m < -2 ∨ (-1 ≤ m ∧ m ≤ 2)) := by
  sorry

end number_of_non_empty_proper_subsets_of_A_range_of_m_for_A_superset_B_l32_32966


namespace set_intersection_complement_l32_32827

theorem set_intersection_complement (U M N : Set ℤ)
  (hU : U = {0, -1, -2, -3, -4})
  (hM : M = {0, -1, -2})
  (hN : N = {0, -3, -4}) :
  (U \ M) ∩ N = {-3, -4} :=
by
  sorry

end set_intersection_complement_l32_32827


namespace smaller_square_area_percentage_l32_32654

noncomputable def percent_area_of_smaller_square (side_length_larger_square : ℝ) : ℝ :=
  let diagonal_larger_square := side_length_larger_square * Real.sqrt 2
  let radius_circle := diagonal_larger_square / 2
  let x := (2 + 4 * (side_length_larger_square / 2)) / ((side_length_larger_square / 2) * 2) -- Simplified quadratic solution
  let side_length_smaller_square := side_length_larger_square * x
  let area_smaller_square := side_length_smaller_square ^ 2
  let area_larger_square := side_length_larger_square ^ 2
  (area_smaller_square / area_larger_square) * 100

-- Statement to show that under given conditions, the area of the smaller square is 4% of the larger square's area
theorem smaller_square_area_percentage :
  percent_area_of_smaller_square 4 = 4 := 
sorry

end smaller_square_area_percentage_l32_32654


namespace vanessa_score_l32_32700

-- Define the total score of the team
def total_points : ℕ := 60

-- Define the score of the seven other players
def other_players_points : ℕ := 7 * 4

-- Mathematics statement for proof
theorem vanessa_score : total_points - other_players_points = 32 :=
by
    sorry

end vanessa_score_l32_32700


namespace determine_F_l32_32787

def f1 (x : ℝ) : ℝ := x^2 + x
def f2 (x : ℝ) : ℝ := 2 * x^2 - x
def f3 (x : ℝ) : ℝ := x^2 + x

def g1 (x : ℝ) : ℝ := x - 2
def g2 (x : ℝ) : ℝ := 2 * x
def g3 (x : ℝ) : ℝ := x + 2

def h (x : ℝ) : ℝ := x

theorem determine_F (F1 F2 F3 : ℕ) : 
  (F1 = 0 ∧ F2 = 0 ∧ F3 = 1) :=
by
  sorry

end determine_F_l32_32787


namespace decision_represented_by_D_l32_32205

-- Define the basic symbols in the flowchart
inductive BasicSymbol
| Start
| Process
| Decision
| End

open BasicSymbol

-- Define the meaning of each basic symbol
def meaning_of (sym : BasicSymbol) : String :=
  match sym with
  | Start => "start"
  | Process => "process"
  | Decision => "decision"
  | End => "end"

-- The theorem stating that the Decision symbol represents a decision
theorem decision_represented_by_D : meaning_of Decision = "decision" :=
by sorry

end decision_represented_by_D_l32_32205


namespace turnips_total_l32_32467

def melanie_turnips := 139
def benny_turnips := 113

def total_turnips (melanie_turnips benny_turnips : Nat) : Nat :=
  melanie_turnips + benny_turnips

theorem turnips_total :
  total_turnips melanie_turnips benny_turnips = 252 :=
by
  sorry

end turnips_total_l32_32467


namespace same_selection_exists_l32_32496

-- Define the set of dogs
def Dogs : Type := Fin 100

-- Define the courtiers
def Courtiers : Type := Fin 100

-- Each courtier selects exactly three dogs.
def selection (c : Courtiers) : Finset Dogs := sorry

-- Condition: For any two courtiers, there are at least two common dogs they both selected.
axiom common_dogs (c₁ c₂ : Courtiers) (h : c₁ ≠ c₂) :
  (selection c₁ ∩ selection c₂).card ≥ 2

-- Prove that there exist two courtiers who picked exactly the same three dogs.
theorem same_selection_exists :
  ∃ c₁ c₂ : Courtiers, c₁ ≠ c₂ ∧ selection c₁ = selection c₂ :=
sorry

end same_selection_exists_l32_32496


namespace train_crosses_platform_in_15_seconds_l32_32657

-- Definitions based on conditions
def length_of_train : ℝ := 330 -- in meters
def tunnel_length : ℝ := 1200 -- in meters
def time_to_cross_tunnel : ℝ := 45 -- in seconds
def platform_length : ℝ := 180 -- in meters

-- Definition based on the solution but directly asserting the correct answer.
def time_to_cross_platform : ℝ := 15 -- in seconds

-- Lean statement
theorem train_crosses_platform_in_15_seconds :
  (length_of_train + platform_length) / ((length_of_train + tunnel_length) / time_to_cross_tunnel) = time_to_cross_platform :=
by
  sorry

end train_crosses_platform_in_15_seconds_l32_32657


namespace number_of_duty_arrangements_l32_32133

noncomputable def dutyDays := {1, 2, 3, 4}

theorem number_of_duty_arrangements : 
  ∑ (A_day : dutyDays) in ({1, 4} : Finset ℕ), 
    ∑ (B_day : dutyDays) in ({1,2,3,4} \ ({A_day - 1, A_day + 1} : Finset ℕ)),
      2 = 8 := 
by
  sorry

end number_of_duty_arrangements_l32_32133


namespace sin_cos_eq_one_l32_32220

open Real

theorem sin_cos_eq_one (x : ℝ) (h0 : 0 ≤ x) (h2 : x < 2 * π) (h : sin x + cos x = 1) :
  x = 0 ∨ x = π / 2 := 
by
  sorry

end sin_cos_eq_one_l32_32220


namespace product_of_differing_inputs_equal_l32_32266

theorem product_of_differing_inputs_equal (a b : ℝ) (h₁ : a ≠ b)
(h₂ : |Real.log a - (1 / 2)| = |Real.log b - (1 / 2)|) : a * b = Real.exp 1 :=
sorry

end product_of_differing_inputs_equal_l32_32266


namespace blood_expiration_date_l32_32923

theorem blood_expiration_date :
  let expiry_seconds := (11.factorial : ℕ) in
  let seconds_in_a_day := 86400 in
  let days_until_expiry := expiry_seconds / seconds_in_a_day in
  let donation_date := date.mk 1 15 (year) in
  let expiration_date := days_after donation_date days_until_expiry in
  expiration_date = date.mk 3 8 (year + 1)
:= sorry

end blood_expiration_date_l32_32923


namespace bees_process_2_77_kg_nectar_l32_32928

noncomputable def nectar_to_honey : ℝ :=
  let percent_other_in_nectar : ℝ := 0.30
  let other_mass_in_honey : ℝ := 0.83
  other_mass_in_honey / percent_other_in_nectar

theorem bees_process_2_77_kg_nectar :
  nectar_to_honey = 2.77 :=
by
  sorry

end bees_process_2_77_kg_nectar_l32_32928


namespace purchase_gifts_and_have_money_left_l32_32520

/-
  We start with 5000 forints in our pocket to buy gifts, visiting three stores.
  In each store, we find a gift that we like and purchase it if we have enough money. 
  The prices in each store are independently 1000, 1500, or 2000 forints, each with a probability of 1/3. 
  What is the probability that we can purchase gifts from all three stores 
  and still have money left (i.e., the total expenditure is at most 4500 forints)?
-/

def giftProbability (totalForints : ℕ) (prices : List ℕ) : ℚ :=
  let outcomes := prices |>.product prices |>.product prices
  let favorable := outcomes.filter (λ ((p1, p2), p3) => p1 + p2 + p3 <= totalForints)
  favorable.length / outcomes.length

theorem purchase_gifts_and_have_money_left :
  giftProbability 4500 [1000, 1500, 2000] = 17 / 27 :=
sorry

end purchase_gifts_and_have_money_left_l32_32520


namespace equivalent_expression_l32_32048

-- Let a, b, c, d, e be real numbers
variables (a b c d e : ℝ)

-- Condition given in the problem
def condition : Prop := 81 * a - 27 * b + 9 * c - 3 * d + e = -5

-- Objective: Prove that 8 * a - 4 * b + 2 * c - d + e = -5 given the condition
theorem equivalent_expression (h : condition a b c d e) : 8 * a - 4 * b + 2 * c - d + e = -5 :=
sorry

end equivalent_expression_l32_32048


namespace abc_sum_square_identity_l32_32686

theorem abc_sum_square_identity (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 941) (h2 : a + b + c = 31) :
  ab + bc + ca = 10 :=
by
  sorry

end abc_sum_square_identity_l32_32686


namespace calc_fraction_l32_32547

theorem calc_fraction:
  (125: ℕ) = 5 ^ 3 →
  (25: ℕ) = 5 ^ 2 →
  (25 ^ 40) / (125 ^ 20) = 5 ^ 20 :=
by
  intros h1 h2
  sorry

end calc_fraction_l32_32547


namespace smallest_mn_sum_l32_32707

theorem smallest_mn_sum (m n : ℕ) (h : 3 * n ^ 3 = 5 * m ^ 2) : m + n = 60 :=
sorry

end smallest_mn_sum_l32_32707


namespace product_of_fractions_l32_32778

-- Definitions from the conditions
def a : ℚ := 2 / 3 
def b : ℚ := 3 / 5
def c : ℚ := 4 / 7
def d : ℚ := 5 / 9

-- Statement of the proof problem
theorem product_of_fractions : a * b * c * d = 8 / 63 := 
by
  sorry

end product_of_fractions_l32_32778


namespace geometric_series_first_term_l32_32354

theorem geometric_series_first_term
  (a r : ℚ)
  (h1 : a / (1 - r) = 30)
  (h2 : a^2 / (1 - r^2) = 150) :
  a = 60 / 7 :=
by
  sorry

end geometric_series_first_term_l32_32354


namespace three_digit_multiples_of_seven_count_l32_32272

theorem three_digit_multiples_of_seven_count :
  let smallest := 15
  let largest := 142
  largest - smallest + 1 = 128 :=
by
  let smallest := 15
  let largest := 142
  have h_smallest : 7 * smallest = 105 := rfl
  have h_largest : 7 * largest = 994 := rfl
  show largest - smallest + 1 = 128 from sorry

end three_digit_multiples_of_seven_count_l32_32272


namespace y_intercept_of_line_l32_32776

theorem y_intercept_of_line (x y : ℝ) (h : 2 * x - 3 * y = 6) (hx : x = 0) : y = -2 :=
by
  sorry

end y_intercept_of_line_l32_32776


namespace sum_of_number_and_radical_conjugate_l32_32666

theorem sum_of_number_and_radical_conjugate : 
  (10 - Real.sqrt 2018) + (10 + Real.sqrt 2018) = 20 := 
by 
  sorry

end sum_of_number_and_radical_conjugate_l32_32666


namespace triangle_angle_A_l32_32428

theorem triangle_angle_A (A B C : ℝ) (a b c : ℝ) (hC : C = Real.pi / 6) (hCos : c = 2 * a * Real.cos B) : A = (5 * Real.pi) / 12 :=
  sorry

end triangle_angle_A_l32_32428


namespace precision_mult_10_decreases_precision_mult_35_decreases_precision_div_10_increases_precision_div_35_increases_l32_32696

-- Given definitions for precision adjustment
def initial_precision := 3

def new_precision_mult (x : ℕ): ℕ :=
  initial_precision - 1   -- Example: Multiplying by 10 moves decimal point right decreasing precision by 1

def new_precision_mult_large (x : ℕ): ℕ := 
  initial_precision - 2   -- Example: Multiplying by 35 generally decreases precision by 2

def new_precision_div (x : ℕ): ℕ := 
  initial_precision + 1   -- Example: Dividing by 10 moves decimal point left increasing precision by 1

def new_precision_div_large (x : ℕ): ℕ := 
  initial_precision + 1   -- Example: Dividing by 35 generally increases precision by 1

-- Statements to prove
theorem precision_mult_10_decreases: 
  new_precision_mult 10 = 2 := 
by 
  sorry

theorem precision_mult_35_decreases: 
  new_precision_mult_large 35 = 1 := 
by 
  sorry

theorem precision_div_10_increases: 
  new_precision_div 10 = 4 := 
by 
  sorry

theorem precision_div_35_increases: 
  new_precision_div_large 35 = 4 := 
by 
  sorry

end precision_mult_10_decreases_precision_mult_35_decreases_precision_div_10_increases_precision_div_35_increases_l32_32696


namespace sum_of_values_k_l32_32372

theorem sum_of_values_k (k : ℕ) : 
  (∀ x y : ℤ, (3 * x * x - k * x + 12 = 0) ∧ (3 * y * y - k * y + 12 = 0) ∧ x ≠ y) → k = 0 :=
by
  sorry

end sum_of_values_k_l32_32372


namespace total_tickets_correct_l32_32655

-- Let's define the conditions given in the problem
def student_tickets (adult_tickets : ℕ) := 2 * adult_tickets
def adult_tickets := 122
def total_tickets := adult_tickets + student_tickets adult_tickets

-- We now state the theorem to be proved
theorem total_tickets_correct : total_tickets = 366 :=
by 
  sorry

end total_tickets_correct_l32_32655


namespace height_of_pole_l32_32610

noncomputable section
open Real

theorem height_of_pole (α β γ : ℝ) (h xA xB xC : ℝ) 
  (hA : tan α = h / xA) (hB : tan β = h / xB) (hC : tan γ = h / xC) 
  (sum_angles : α + β + γ = π / 2) : h = 10 :=
by
  sorry

end height_of_pole_l32_32610


namespace fruit_weights_l32_32116

theorem fruit_weights (orange banana mandarin peach apple : ℕ) (weights : Fin 5 → ℕ)
  (cond1 : peach < orange)
  (cond2 : apple < banana ∧ banana < peach)
  (cond3 : mandarin < banana)
  (cond4 : apple + banana > orange)
  (cond5 : multiset.coe {orange, banana, mandarin, peach, apple} = multiset.coe ({100, 150, 170, 200, 280} : multiset ℕ)) :
  orange = 280 ∧ banana = 170 ∧ mandarin = 100 ∧ peach = 200 ∧ apple = 150 :=
sorry

end fruit_weights_l32_32116


namespace derivative_at_2_l32_32965

noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

theorem derivative_at_2 : deriv f 2 = Real.sqrt 2 / 4 := by
  sorry

end derivative_at_2_l32_32965


namespace greatest_t_solution_l32_32943

theorem greatest_t_solution :
  ∀ t : ℝ, t ≠ 8 ∧ t ≠ -5 →
  (t^2 - t - 40) / (t - 8) = 5 / (t + 5) →
  t ≤ -2 :=
by
  sorry

end greatest_t_solution_l32_32943


namespace lcm_140_225_is_6300_l32_32027

def lcm_140_225 : ℕ := Nat.lcm 140 225

theorem lcm_140_225_is_6300 : lcm_140_225 = 6300 :=
by
  sorry

end lcm_140_225_is_6300_l32_32027


namespace supplement_of_complement_of_65_l32_32893

def complement (angle : ℝ) : ℝ := 90 - angle
def supplement (angle : ℝ) : ℝ := 180 - angle

theorem supplement_of_complement_of_65 : supplement (complement 65) = 155 :=
by
  -- provide the proof steps here
  sorry

end supplement_of_complement_of_65_l32_32893


namespace original_cost_proof_l32_32881

/-!
# Prove that the original cost of the yearly subscription to professional magazines is $940.
# Given conditions:
# 1. The company must make a 20% cut in the magazine budget.
# 2. After the cut, the company will spend $752.
-/

theorem original_cost_proof (x : ℝ)
  (h1 : 0.80 * x = 752) :
  x = 940 :=
by
  sorry

end original_cost_proof_l32_32881


namespace odd_divisors_l32_32595

-- Define p_1, p_2, p_3 as distinct prime numbers greater than 3
variables {p_1 p_2 p_3 : ℕ}
-- Define k, a, b, c as positive integers
variables {n k a b c : ℕ}

-- The conditions
def distinct_primes (p_1 p_2 p_3 : ℕ) : Prop :=
  p_1 > 3 ∧ p_2 > 3 ∧ p_3 > 3 ∧ p_1 ≠ p_2 ∧ p_1 ≠ p_3 ∧ p_2 ≠ p_3

def is_n (n k p_1 p_2 p_3 a b c : ℕ) : Prop :=
  n = 2^k * p_1^a * p_2^b * p_3^c

def conditions (a b c : ℕ) : Prop :=
  a + b > c ∧ 1 ≤ b ∧ b ≤ c

-- The main statement
theorem odd_divisors
  (h_prime : distinct_primes p_1 p_2 p_3)
  (h_n : is_n n k p_1 p_2 p_3 a b c)
  (h_cond : conditions a b c) : 
  ∃ d : ℕ, d = (a + 1) * (b + 1) * (c + 1) :=
by sorry

end odd_divisors_l32_32595


namespace surface_area_ratio_l32_32301

-- Definitions for side lengths in terms of common multiplier x
def side_length_a (x : ℝ) := 2 * x
def side_length_b (x : ℝ) := 1 * x
def side_length_c (x : ℝ) := 3 * x
def side_length_d (x : ℝ) := 4 * x
def side_length_e (x : ℝ) := 6 * x

-- Definitions for surface areas using the given formula
def surface_area (side_length : ℝ) := 6 * side_length^2

def surface_area_a (x : ℝ) := surface_area (side_length_a x)
def surface_area_b (x : ℝ) := surface_area (side_length_b x)
def surface_area_c (x : ℝ) := surface_area (side_length_c x)
def surface_area_d (x : ℝ) := surface_area (side_length_d x)
def surface_area_e (x : ℝ) := surface_area (side_length_e x)

-- Proof statement for the ratio of total surface areas
theorem surface_area_ratio (x : ℝ) (hx : x ≠ 0) :
  (surface_area_a x) / (surface_area_b x) = 4 ∧
  (surface_area_c x) / (surface_area_b x) = 9 ∧
  (surface_area_d x) / (surface_area_b x) = 16 ∧
  (surface_area_e x) / (surface_area_b x) = 36 :=
by {
  sorry
}

end surface_area_ratio_l32_32301


namespace crayons_per_child_l32_32935

theorem crayons_per_child (total_crayons children : ℕ) (h_total : total_crayons = 56) (h_children : children = 7) : (total_crayons / children) = 8 := by
  -- proof will go here
  sorry

end crayons_per_child_l32_32935


namespace compare_expression_solve_inequality_l32_32641

-- Part (1) Problem Statement in Lean 4
theorem compare_expression (x : ℝ) (h : x ≥ -1) : 
  x^3 + 1 ≥ x^2 + x ∧ (x^3 + 1 = x^2 + x ↔ x = 1 ∨ x = -1) :=
by sorry

-- Part (2) Problem Statement in Lean 4
theorem solve_inequality (x a : ℝ) (ha : a < 0) : 
  (x^2 - a * x - 6 * a^2 > 0) ↔ (x < 3 * a ∨ x > -2 * a) :=
by sorry

end compare_expression_solve_inequality_l32_32641


namespace symmetric_points_l32_32280

theorem symmetric_points (m n : ℤ) (h1 : m - 1 = -3) (h2 : 1 = n - 1) : m + n = 0 := by
  sorry

end symmetric_points_l32_32280


namespace smallest_even_number_l32_32012

theorem smallest_even_number (x : ℤ) (h : (x + (x + 2) + (x + 4) + (x + 6)) = 140) : x = 32 :=
by
  sorry

end smallest_even_number_l32_32012


namespace min_value_expression_l32_32323

theorem min_value_expression (α β : ℝ) : 
  (3 * Real.cos α + 4 * Real.sin β - 7) ^ 2 + (3 * Real.sin α + 4 * Real.cos β - 12) ^ 2 ≥ 36 :=
sorry

end min_value_expression_l32_32323


namespace functional_eq_uniq_l32_32855

noncomputable def f (x : ℝ) : ℝ := sorry

theorem functional_eq_uniq (f : ℝ → ℝ) (h : ∀ x y : ℝ, (f x * f y - f (x * y)) / 4 = x^2 + y^2 + 2) : 
  ∀ x : ℝ, f x = x^2 + 3 :=
by 
  sorry

end functional_eq_uniq_l32_32855


namespace inequality_solution_range_l32_32300

theorem inequality_solution_range (a : ℝ) : (∃ x : ℝ, |x + 1| + |x - 3| ≤ a) ↔ a ≥ 4 :=
sorry

end inequality_solution_range_l32_32300


namespace supplementary_angle_l32_32957

theorem supplementary_angle {α β : ℝ} (angle_supplementary : α + β = 180) (angle_1_eq : α = 80) : β = 100 :=
by
  sorry

end supplementary_angle_l32_32957


namespace pythagorean_triple_345_l32_32799

theorem pythagorean_triple_345 : (3^2 + 4^2 = 5^2) := 
by 
  -- Here, the proof will be filled in, but we use 'sorry' for now.
  sorry

end pythagorean_triple_345_l32_32799


namespace a_n_sequence_term2015_l32_32748

theorem a_n_sequence_term2015 :
  ∃ (a : ℕ → ℚ), a 1 = 1 ∧ a 2 = 1/2 ∧ (∀ n ≥ 2, a n * (a (n-1) + a (n+1)) = 2 * a (n+1) * a (n-1)) ∧ a 2015 = 1/2015 :=
sorry

end a_n_sequence_term2015_l32_32748


namespace sum_of_k_with_distinct_integer_solutions_l32_32364

-- Definitions from conditions
def quadratic_eq (k : ℤ) (x : ℤ) : Prop := 3 * x^2 - k * x + 12 = 0

def distinct_integer_solutions (k : ℤ) : Prop := ∃ p q : ℤ, p ≠ q ∧ quadratic_eq k p ∧ quadratic_eq k q

-- Main statement to prove
theorem sum_of_k_with_distinct_integer_solutions : 
  ∑ k in {k : ℤ | distinct_integer_solutions k}, k = 0 := 
sorry

end sum_of_k_with_distinct_integer_solutions_l32_32364


namespace dice_product_probability_l32_32523

def is_valid_die_value (n : ℕ) : Prop := n ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ)

theorem dice_product_probability :
  ∃ (a b c : ℕ), is_valid_die_value a ∧ is_valid_die_value b ∧ is_valid_die_value c ∧ 
  a * b * c = 8 ∧ 
  (1 / 6 : ℝ) * (1 / 6) * (1 / 6) * (6 + 1) = (7 / 216 : ℝ) :=
sorry

end dice_product_probability_l32_32523


namespace principal_amount_correct_l32_32679

-- Define the given conditions and quantities
def P : ℝ := 1054.76
def final_amount : ℝ := 1232.0
def rate1 : ℝ := 0.05
def rate2 : ℝ := 0.07
def rate3 : ℝ := 0.04

-- Define the statement we want to prove
theorem principal_amount_correct :
  final_amount = P * (1 + rate1) * (1 + rate2) * (1 + rate3) :=
sorry

end principal_amount_correct_l32_32679


namespace min_value_A_mul_abs_x1_minus_x2_l32_32682

noncomputable def f (x : ℝ) : ℝ :=
  Real.sin (2017 * x + Real.pi / 6) + Real.cos (2017 * x - Real.pi / 3)

theorem min_value_A_mul_abs_x1_minus_x2 :
  ∃ x1 x2 : ℝ, (∀ x : ℝ, f x1 ≤ f x ∧ f x ≤ f x2) →
  2 * |x1 - x2| = (2 * Real.pi) / 2017 :=
sorry

end min_value_A_mul_abs_x1_minus_x2_l32_32682


namespace maximum_number_of_buses_l32_32102

-- Definitions of the conditions
def bus_stops : ℕ := 9 -- There are 9 bus stops in the city.
def max_common_stops : ℕ := 1 -- Any two buses have at most one common stop.
def stops_per_bus : ℕ := 3 -- Each bus stops at exactly three stops.

-- The main theorem statement
theorem maximum_number_of_buses (n : ℕ) :
  (bus_stops = 9) →
  (max_common_stops = 1) →
  (stops_per_bus = 3) →
  n ≤ 12 :=
sorry

end maximum_number_of_buses_l32_32102


namespace trigonometric_identity_proof_l32_32175

variable (α : ℝ)

theorem trigonometric_identity_proof :
  3 + 4 * (Real.sin (4 * α + (3 / 2) * Real.pi)) +
  Real.sin (8 * α + (5 / 2) * Real.pi) = 
  8 * (Real.sin (2 * α))^4 :=
sorry

end trigonometric_identity_proof_l32_32175


namespace lesser_solution_is_minus_15_l32_32028

noncomputable def lesser_solution : ℤ := -15

theorem lesser_solution_is_minus_15 :
  ∃ x y : ℤ, x^2 + 10 * x - 75 = 0 ∧ y^2 + 10 * y - 75 = 0 ∧ x < y ∧ x = lesser_solution :=
by 
  sorry

end lesser_solution_is_minus_15_l32_32028


namespace mike_initial_marbles_l32_32468

-- Defining the conditions
def gave_marble (initial_marbles : ℕ) (given_marbles : ℕ) : ℕ := initial_marbles - given_marbles
def marbles_left (initial_marbles : ℕ) (given_marbles : ℕ) : ℕ := initial_marbles - given_marbles

-- Using the given conditions
def initial_mike_marbles : ℕ := 8
def given_marbles : ℕ := 4
def remaining_marbles : ℕ := 4

-- Proving the statement
theorem mike_initial_marbles :
  initial_mike_marbles - given_marbles = remaining_marbles :=
by
  -- The proof
  sorry

end mike_initial_marbles_l32_32468


namespace unique_10_tuple_solution_l32_32422

noncomputable def condition (x : Fin 10 → ℝ) : Prop :=
  (1 - x 0)^2 +
  (x 0 - x 1)^2 + 
  (x 1 - x 2)^2 + 
  (x 2 - x 3)^2 + 
  (x 3 - x 4)^2 + 
  (x 4 - x 5)^2 + 
  (x 5 - x 6)^2 + 
  (x 6 - x 7)^2 + 
  (x 7 - x 8)^2 + 
  (x 8 - x 9)^2 + 
  x 9^2 + 
  (1/2) * (x 9 - x 0)^2 = 1/10

theorem unique_10_tuple_solution : 
  ∃! (x : Fin 10 → ℝ), condition x := 
sorry

end unique_10_tuple_solution_l32_32422


namespace two_courtiers_have_same_selection_l32_32494

-- Definitions based on the given conditions
def dogs : Type := Fin 100
def courtiers : Type := Fin 100
def select_three_dogs (c : courtiers) : set dogs := sorry

-- Main theorem statement
theorem two_courtiers_have_same_selection :
  (∀ (c1 c2 : courtiers), c1 ≠ c2 → (select_three_dogs c1 ∩ select_three_dogs c2).card = 2) →
  ∃ (c1 c2 : courtiers), c1 ≠ c2 ∧ select_three_dogs c1 = select_three_dogs c2 :=
sorry

end two_courtiers_have_same_selection_l32_32494


namespace courtiers_dog_selection_l32_32504

theorem courtiers_dog_selection :
  ∃ (c1 c2 : ℕ), c1 ≠ c2 ∧ ∀ d1 d2 d3 : ℕ, 
  (dog_selected_by c1 = {d1, d2, d3} ∧ dog_selected_by c2 = {d1, d2, d3}) :=
begin
  -- Parameters
  let num_courtiers : ℕ := 100,
  let num_dogs : ℕ := 100,
  let picks_per_courtier : finset (fin 100) := {a b c | a, b, c ∈ fin 100},
  
  -- Conditions
  have h1 : ∀ (i j : ℕ) (hic : i ≠ j), 
  (picks_per_courtier i ∩ picks_per_courtier j).card ≥ 2 := sorry,

  -- Proof to show that there exist two courtiers who pick the exact same three dogs
  sorry
end

end courtiers_dog_selection_l32_32504


namespace probability_A_seven_rolls_l32_32235

noncomputable def probability_A_after_n_rolls (n : ℕ) : ℚ :=
  if n = 0 then 1 else 1/3 * (1 - (-1/2)^(n-1))

theorem probability_A_seven_rolls : probability_A_after_n_rolls 7 = 21 / 64 :=
by sorry

end probability_A_seven_rolls_l32_32235


namespace parabola_focus_l32_32825

theorem parabola_focus (p : ℝ) (hp : ∃ (p : ℝ), ∀ x y : ℝ, x^2 = 2 * p * y) : (∀ (hf : (0, 2) = (0, p / 2)), p = 4) :=
sorry

end parabola_focus_l32_32825


namespace a_plus_b_l32_32278

theorem a_plus_b (a b : ℝ) (h : a * (a - 4) = 12) (h2 : b * (b - 4) = 12) (h3 : a ≠ b) : a + b = 4 :=
sorry

end a_plus_b_l32_32278


namespace num_stripes_on_us_flag_l32_32739

-- Definitions based on conditions in the problem
def num_stars : ℕ := 50

def num_circles : ℕ := (num_stars / 2) - 3

def num_squares (S : ℕ) : ℕ := 2 * S + 6

def total_shapes (num_squares : ℕ) : ℕ := num_circles + num_squares

-- The theorem stating the number of stripes
theorem num_stripes_on_us_flag (S : ℕ) (h1 : num_circles = 22) (h2 : total_shapes (num_squares S) = 54) : S = 13 := by
  sorry

end num_stripes_on_us_flag_l32_32739


namespace problem1_problem2_l32_32267

-- Define the function f
def f (x a : ℝ) : ℝ := |x + a| + |3 * x - 1|

-- Part (1) statement
theorem problem1 (x : ℝ) : f x (-1) ≤ 1 ↔ (1/4 ≤ x ∧ x ≤ 1/2) :=
by
    sorry

-- Part (2) statement
theorem problem2 (x a : ℝ) (h : 1/4 ≤ x ∧ x ≤ 1) : f x a ≤ |3 * x + 1| ↔ -7/3 ≤ a ∧ a ≤ 1 :=
by
    sorry

end problem1_problem2_l32_32267


namespace julia_height_is_172_7_cm_l32_32109

def julia_height_in_cm (height_in_inches : ℝ) (conversion_factor : ℝ) : ℝ :=
  height_in_inches * conversion_factor

theorem julia_height_is_172_7_cm :
  julia_height_in_cm 68 2.54 = 172.7 :=
by
  sorry

end julia_height_is_172_7_cm_l32_32109


namespace number_of_valid_permutations_l32_32592

noncomputable def valid_permutations := 
  let S := Finset.range 14 + 1
  let descending_part := ((S.erase 1).subsets 6).filter (fun s => s.to_list.sorted.reverse = s.to_list)
  let ascending_part := ((S.erase 1).subsets 7).filter (fun s => s.to_list.sorted = s.to_list)
  descending_part.card * ascending_part.card

theorem number_of_valid_permutations : valid_permutations = nat.choose 13 6 := by
  sorry

end number_of_valid_permutations_l32_32592


namespace solve_for_asterisk_l32_32161

theorem solve_for_asterisk (asterisk : ℝ) : 
  ((60 / 20) * (60 / asterisk) = 1) → asterisk = 180 :=
by
  sorry

end solve_for_asterisk_l32_32161


namespace three_digit_multiples_of_seven_l32_32271

theorem three_digit_multiples_of_seven : 
  ∃ k, (k = {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ n % 7 = 0}.card) ∧ k = 128 :=
by {
  sorry
}

end three_digit_multiples_of_seven_l32_32271


namespace discount_price_l32_32616

theorem discount_price (a : ℝ) (original_price : ℝ) (sold_price : ℝ) :
  original_price = 200 ∧ sold_price = 148 → (original_price * (1 - a/100) * (1 - a/100) = sold_price) :=
by
  sorry

end discount_price_l32_32616


namespace white_tulips_multiple_of_seven_l32_32513

/-- Let R be the number of red tulips, which is given as 91. 
    We also know that the greatest number of identical bouquets that can be made without 
    leaving any flowers out is 7.
    Prove that the number of white tulips W is a multiple of 7. -/
theorem white_tulips_multiple_of_seven (R : ℕ) (g : ℕ) (W : ℕ) (hR : R = 91) (hg : g = 7) :
  ∃ w : ℕ, W = 7 * w :=
by
  sorry

end white_tulips_multiple_of_seven_l32_32513


namespace part_a_part_b_part_c_l32_32384

-- Define initial setup and conditions
def average (scores: List ℚ) : ℚ :=
  scores.sum / scores.length

-- Part (a)
theorem part_a (A B : List ℚ) (a b : ℚ) (A' : List ℚ) (B' : List ℚ) :
  average A = a ∧ average B = b ∧ average A' = a ∧ average B' = b ∧
  average A' > a ∧ average B' > b :=
sorry

-- Part (b)
theorem part_b (A B : List ℚ) : 
  ∀ a b : ℚ, (average A = a ∧ average B = b ∧ ∀ A' : List ℚ, average A' > a ∧ ∀ B' : List ℚ, average B' > b) :=
sorry

-- Part (c)
theorem part_c (A B C : List ℚ) (a b c : ℚ) (A' B' C' A'' B'' C'' : List ℚ) :
  average A = a ∧ average B = b ∧ average C = c ∧
  average A' = a ∧ average B' = b ∧ average C' = c ∧
  average A'' = a ∧ average B'' = b ∧ average C'' = c ∧
  average A' > a ∧ average B' > b ∧ average C' > c ∧
  average A'' > average A' ∧ average B'' > average B' ∧ average C'' > average C' :=
sorry

end part_a_part_b_part_c_l32_32384


namespace find_a100_l32_32262

-- Define the arithmetic sequence with the given conditions
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

-- Given conditions
variables {a d : ℤ}
variables (S_9 : ℤ) (a_10 : ℤ)

-- Conditions in Lean definition
def conditions (a d : ℤ) : Prop :=
  (9 / 2 * (2 * a + 8 * d) = 27) ∧ (a + 9 * d = 8)

-- Prove the final statement
theorem find_a100 : ∃ a d : ℤ, conditions a d → arithmetic_sequence a d 100 = 98 := 
by {
    sorry
}

end find_a100_l32_32262


namespace bacteria_population_at_2_15_l32_32045

noncomputable def bacteria_at_time (initial_pop : ℕ) (start_time end_time : ℕ) (interval : ℕ) : ℕ :=
  initial_pop * 2 ^ ((end_time - start_time) / interval)

theorem bacteria_population_at_2_15 :
  let initial_pop := 50
  let start_time := 0  -- 2:00 p.m.
  let end_time := 15   -- 2:15 p.m.
  let interval := 4
  bacteria_at_time initial_pop start_time end_time interval = 400 := sorry

end bacteria_population_at_2_15_l32_32045


namespace basil_plants_count_l32_32060

-- Define the number of basil plants and the number of oregano plants
variables (B O : ℕ)

-- Define the conditions
def condition1 : Prop := O = 2 * B + 2
def condition2 : Prop := B + O = 17

-- The proof statement
theorem basil_plants_count (h1 : condition1 B O) (h2 : condition2 B O) : B = 5 := by
  sorry

end basil_plants_count_l32_32060


namespace find_N_l32_32286

theorem find_N : ∃ N : ℕ, (∃ x y : ℕ, 
(10 * x + y = N) ∧ 
(4 * x + 2 * y = N / 2) ∧ 
(1 ≤ x) ∧ (x ≤ 9) ∧ 
(0 ≤ y) ∧ (y ≤ 9)) ∧
(N = 32 ∨ N = 64 ∨ N = 96) :=
by {
    have h1 : ∀ x y : ℕ, 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 → 4 * x + 2 * y = (10 * x + y) / 2 → true, sorry,
    exact h1
}

end find_N_l32_32286


namespace lee_charged_per_action_figure_l32_32716

theorem lee_charged_per_action_figure :
  ∀ (sneakers_cost savings action_figures leftovers price_per_fig),
    sneakers_cost = 90 →
    savings = 15 →
    action_figures = 10 →
    leftovers = 25 →
    price_per_fig = 10 →
    (savings + action_figures * price_per_fig) - sneakers_cost = leftovers → price_per_fig = 10 :=
by
  intros sneakers_cost savings action_figures leftovers price_per_fig
  intros h_sneakers_cost h_savings h_action_figures h_leftovers h_price_per_fig
  intros h_total
  sorry

end lee_charged_per_action_figure_l32_32716


namespace seconds_in_8_point_5_minutes_l32_32967

def minutesToSeconds (minutes : ℝ) : ℝ := minutes * 60

theorem seconds_in_8_point_5_minutes : minutesToSeconds 8.5 = 510 := 
by
  sorry

end seconds_in_8_point_5_minutes_l32_32967


namespace y_intercept_of_line_l32_32772

theorem y_intercept_of_line : ∃ y : ℝ, 2 * 0 - 3 * y = 6 ∧ y = -2 :=
by
  exists (-2)
  split
  . simp
  . rfl

end y_intercept_of_line_l32_32772


namespace non_obtuse_triangle_medians_ge_4R_l32_32996

theorem non_obtuse_triangle_medians_ge_4R
  (A B C : Type*)
  (triangle_non_obtuse : ∀ (α β γ : ℝ), α ≤ 90 ∧ β ≤ 90 ∧ γ ≤ 90)
  (m_a m_b m_c : ℝ)
  (R : ℝ)
  (h1 : AO + BO ≤ AM + BM)
  (h2 : AM = 2 * m_a / 3 ∧ BM = 2 * m_b / 3)
  (h3 : AO + BO = 2 * R)
  (h4 : m_c ≥ R) : 
  m_a + m_b + m_c ≥ 4 * R :=
by
  sorry

end non_obtuse_triangle_medians_ge_4R_l32_32996


namespace number_of_people_in_tour_l32_32673

theorem number_of_people_in_tour (x : ℕ) : 
  (x ≤ 25 ∧ 100 * x = 2700 ∨ 
  (x > 25 ∧ 
   (100 - 2 * (x - 25)) * x = 2700 ∧ 
   70 ≤ 100 - 2 * (x - 25))) → 
  x = 30 := 
by
  sorry

end number_of_people_in_tour_l32_32673


namespace program_output_l32_32925

-- Define the initial conditions
def initial_a := 1
def initial_b := 3

-- Define the program transformations
def a_step1 (a b : ℕ) := a + b
def b_step2 (a b : ℕ) := a - b

-- Define the final values after program execution
def final_a := a_step1 initial_a initial_b
def final_b := b_step2 final_a initial_b

-- Statement to prove
theorem program_output :
  final_a = 4 ∧ final_b = 1 :=
by
  -- Placeholder for the actual proof
  sorry

end program_output_l32_32925


namespace probability_of_same_color_l32_32840

noncomputable def prob_same_color (P_A P_B : ℚ) : ℚ :=
  P_A + P_B

theorem probability_of_same_color :
  let P_A := (1 : ℚ) / 7
  let P_B := (12 : ℚ) / 35
  prob_same_color P_A P_B = 17 / 35 := 
by 
  -- Definition of P_A and P_B
  let P_A := (1 : ℚ) / 7
  let P_B := (12 : ℚ) / 35
  -- Use the definition of prob_same_color
  let result := prob_same_color P_A P_B
  -- Now we are supposed to prove that result = 17 / 35
  have : result = (5 : ℚ) / 35 + (12 : ℚ) / 35 := by
    -- Simplifying the fractions individually can be done at this intermediate step
    sorry
  sorry

end probability_of_same_color_l32_32840


namespace smallest_norwegian_is_1344_l32_32791

def is_norwegian (n : ℕ) : Prop :=
  ∃ d1 d2 d3 : ℕ, n > 0 ∧ d1 < d2 ∧ d2 < d3 ∧ d1 * d2 * d3 = n ∧ d1 + d2 + d3 = 2022

theorem smallest_norwegian_is_1344 : ∀ m : ℕ, (is_norwegian m) → m ≥ 1344 :=
by
  sorry

end smallest_norwegian_is_1344_l32_32791


namespace alice_bob_meet_same_point_in_5_turns_l32_32795

theorem alice_bob_meet_same_point_in_5_turns :
  ∃ k : ℕ, k = 5 ∧ 
  (∀ n, (1 + 7 * n) % 24 = 12 ↔ (n = k)) :=
by
  sorry

end alice_bob_meet_same_point_in_5_turns_l32_32795


namespace prob_ξ_greater_than_7_l32_32961

noncomputable def ξ : ℝ → ℝ := sorry -- Scratches actual random variable behaviour setup.

open ProbabilityTheory MeasureTheory

variable {P : Measure ℝ}
variable {σ : ℝ}
variable (h₁ : gaussian P ξ 5 σ)
variable (h₂ : P {ω | 3 ≤ ξ ω ∧ ξ ω ≤ 7} = 0.4)

theorem prob_ξ_greater_than_7 : P {ω | ξ ω > 7} = 0.3 := 
sorry

end prob_ξ_greater_than_7_l32_32961


namespace lcm_hcf_relationship_l32_32304

theorem lcm_hcf_relationship (a b : ℕ) (h_prod : a * b = 84942) (h_hcf : Nat.gcd a b = 33) : Nat.lcm a b = 2574 :=
by
  sorry

end lcm_hcf_relationship_l32_32304


namespace largest_n_value_l32_32026

theorem largest_n_value (n : ℕ) (h1: n < 100000) (h2: (9 * (n - 3)^6 - n^3 + 16 * n - 27) % 7 = 0) : n = 99996 := 
sorry

end largest_n_value_l32_32026


namespace average_income_correct_l32_32043

def incomes : List ℕ := [250, 400, 750, 400, 500]

noncomputable def average : ℕ := (incomes.sum) / incomes.length

theorem average_income_correct : average = 460 :=
by 
  sorry

end average_income_correct_l32_32043


namespace contradiction_problem_l32_32153

theorem contradiction_problem (a b c d : ℝ) (h1 : a + b = 1) (h2 : c + d = 1) (h3 : ac + bd > 1) : 
  (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) → False := 
by
  sorry

end contradiction_problem_l32_32153


namespace general_formulas_values_of_n_for_c_n_gt_one_no_three_terms_arithmetic_seq_l32_32224
noncomputable def a_n (n : ℕ) : ℕ := 2^(n-1)
noncomputable def b_n (n : ℕ) : ℕ := 3*n - 1
noncomputable def c_n (n : ℕ) : ℚ := (3*n - 1) / 2^(n-1)

-- 1. Prove that the sequence {a_n} is given by a_n = 2^(n-1) and {b_n} is given by b_n = 3n - 1
theorem general_formulas :
  (∀ n : ℕ, n > 0 → a_n n = 2^(n-1)) ∧
  (∀ n : ℕ, n > 0 → b_n n = 3*n - 1) :=
sorry

-- 2. Prove that the values of n for which c_n > 1 are n = 1, 2, 3, 4
theorem values_of_n_for_c_n_gt_one :
  { n : ℕ | n > 0 ∧ c_n n > 1 } = {1, 2, 3, 4} :=
sorry

-- 3. Prove that no three terms from {a_n} can form an arithmetic sequence
theorem no_three_terms_arithmetic_seq :
  ∀ p q r : ℕ, p < q ∧ q < r ∧ p > 0 ∧ q > 0 ∧ r > 0 →
  ¬ (2 * a_n q = a_n p + a_n r) :=
sorry

end general_formulas_values_of_n_for_c_n_gt_one_no_three_terms_arithmetic_seq_l32_32224


namespace frank_cookies_l32_32228

theorem frank_cookies :
  ∀ (F M M_i L : ℕ),
    (F = M / 2 - 3) →
    (M = 3 * M_i) →
    (M_i = 2 * L) →
    (L = 5) →
    F = 12 :=
by
  intros F M M_i L h1 h2 h3 h4
  rw [h4] at h3
  rw [h3] at h2
  rw [h2] at h1
  sorry

end frank_cookies_l32_32228


namespace trapezoid_possible_and_area_sum_l32_32233

theorem trapezoid_possible_and_area_sum (a b c d : ℕ) (h1 : a = 4) (h2 : b = 6) (h3 : c = 8) (h4 : d = 12) :
  ∃ (S : ℚ), S = 72 := 
by
  -- conditions ensure one pair of sides is parallel
  -- area calculation based on trapezoid properties
  sorry

end trapezoid_possible_and_area_sum_l32_32233


namespace girls_in_art_class_l32_32009

theorem girls_in_art_class (g b : ℕ) (h_ratio : 4 * b = 3 * g) (h_total : g + b = 70) : g = 40 :=
by {
  sorry
}

end girls_in_art_class_l32_32009


namespace Shara_will_owe_money_l32_32607

theorem Shara_will_owe_money
    (B : ℕ)
    (h1 : 6 * 10 = 60)
    (h2 : B / 2 = 60)
    (h3 : 4 * 10 = 40)
    (h4 : 60 + 40 = 100) :
  B - 100 = 20 :=
sorry

end Shara_will_owe_money_l32_32607


namespace range_of_inclination_angle_l32_32357

theorem range_of_inclination_angle (α : ℝ) (A : ℝ × ℝ) (hA : A = (-2, 0))
  (ellipse_eq : ∀ (x y : ℝ), (x^2 / 2) + y^2 = 1) :
    (0 ≤ α ∧ α < Real.arcsin (Real.sqrt 3 / 3)) ∨ 
    (π - Real.arcsin (Real.sqrt 3 / 3) < α ∧ α < π) :=
sorry

end range_of_inclination_angle_l32_32357


namespace number_exceeds_percent_l32_32035

theorem number_exceeds_percent (x : ℝ) (h : x = 0.12 * x + 52.8) : x = 60 :=
by {
  sorry
}

end number_exceeds_percent_l32_32035


namespace probability_of_sum_24_l32_32912

def die_a_faces : Finset ℕ := (Finset.range 20).erase 0
def die_b_faces : Finset ℕ := (Finset.range 21).erase_all [0, 9]

theorem probability_of_sum_24 :
  let favorable_outcomes := { (a, b) | a ∈ die_a_faces ∧ b ∈ die_b_faces ∧ a + b = 24 }.card
  let total_possible_outcomes := 20 * 20 
  let probability := (favorable_outcomes : ℚ) / total_possible_outcomes 
  probability = 3 / 80 := by
  sorry

end probability_of_sum_24_l32_32912


namespace horner_evaluation_at_2_l32_32360

def f (x : ℤ) : ℤ := 3 * x^5 - 2 * x^4 + 2 * x^3 - 4 * x^2 - 7

theorem horner_evaluation_at_2 : f 2 = 16 :=
by {
  sorry
}

end horner_evaluation_at_2_l32_32360


namespace range_of_a_l32_32446

theorem range_of_a :
  (∀ x : ℝ, abs (x - a) < 1 ↔ (1 / 2 < x ∧ x < 3 / 2)) → (1 / 2 ≤ a ∧ a ≤ 3 / 2) :=
by sorry

end range_of_a_l32_32446


namespace jayda_spending_l32_32661

theorem jayda_spending
  (J A : ℝ)
  (h1 : A = J + (2/5) * J)
  (h2 : J + A = 960) :
  J = 400 :=
by
  sorry

end jayda_spending_l32_32661


namespace area_of_formed_triangle_l32_32318

def triangle_area (S R d : ℝ) (S₁ : ℝ) : Prop :=
  S₁ = (S / 4) * |1 - (d^2 / R^2)|

variable (S R d : ℝ)

theorem area_of_formed_triangle (h : S₁ = (S / 4) * |1 - (d^2 / R^2)|) : triangle_area S R d S₁ :=
by
  sorry

end area_of_formed_triangle_l32_32318


namespace simplify_complex_squaring_l32_32130

theorem simplify_complex_squaring :
  (4 - 3 * Complex.i) ^ 2 = 7 - 24 * Complex.i :=
by
  intro
  sorry

end simplify_complex_squaring_l32_32130


namespace min_value_of_a_l32_32974

theorem min_value_of_a (a : ℝ) :
  (¬ ∃ x0 : ℝ, -1 < x0 ∧ x0 ≤ 2 ∧ x0 - a > 0) → a = 2 :=
by
  sorry

end min_value_of_a_l32_32974


namespace Bella_catch_correct_l32_32113

def Martha_catch : ℕ := 3 + 7
def Cara_catch : ℕ := 5 * Martha_catch - 3
def T : ℕ := Martha_catch + Cara_catch
def Andrew_catch : ℕ := T^2 + 2
def F : ℕ := Martha_catch + Cara_catch + Andrew_catch
def Bella_catch : ℕ := 2 ^ (F / 3)

theorem Bella_catch_correct : Bella_catch = 2 ^ 1102 := by
  sorry

end Bella_catch_correct_l32_32113


namespace steps_to_get_down_empire_state_building_l32_32270

theorem steps_to_get_down_empire_state_building (total_steps : ℕ) (steps_building_to_garden : ℕ) (steps_to_madison_square : ℕ) :
  total_steps = 991 -> steps_building_to_garden = 315 -> steps_to_madison_square = total_steps - steps_building_to_garden -> steps_to_madison_square = 676 :=
by
  intros
  subst_vars
  sorry

end steps_to_get_down_empire_state_building_l32_32270


namespace fruit_weights_correct_l32_32118

def problem_statement :=
  let weights := [100, 150, 170, 200, 280] in
  ∃ (apple mandarin banana peach orange : ℕ),
    apple ∈ weights ∧ mandarin ∈ weights ∧ banana ∈ weights ∧ peach ∈ weights ∧ orange ∈ weights ∧
    apple ≠ mandarin ∧ apple ≠ banana ∧ apple ≠ peach ∧ apple ≠ orange ∧ 
    mandarin ≠ banana ∧ mandarin ≠ peach ∧ mandarin ≠ orange ∧ 
    banana ≠ peach ∧ banana ≠ orange ∧
    peach ≠ orange ∧
    peach < orange ∧
    apple < banana ∧ banana < peach ∧ 
    mandarin < banana ∧
    apple + banana > orange ∧
    apple = 150 ∧ mandarin = 100 ∧ banana = 170 ∧ peach = 200 ∧ orange = 280

theorem fruit_weights_correct : problem_statement := 
by 
  sorry

end fruit_weights_correct_l32_32118


namespace triangle_area_l32_32658

theorem triangle_area (r : ℝ) (a b c : ℝ) (h : a^2 + b^2 = c^2) (hc : c = 2 * r) (r_val : r = 5) (ratio : a / b = 3 / 4 ∧ b / c = 4 / 5) :
  (1 / 2) * a * b = 24 :=
by
  -- We assume statements are given
  sorry

end triangle_area_l32_32658


namespace find_n_l32_32294

theorem find_n (N : ℕ) (hN : 10 ≤ N ∧ N < 100)
  (h : ∃ a b : ℕ, N = 10 * a + b ∧ 4 * a + 2 * b = N / 2) : 
  N = 32 ∨ N = 64 ∨ N = 96 :=
sorry

end find_n_l32_32294


namespace inequality_holds_for_all_x_l32_32567

theorem inequality_holds_for_all_x (a : ℝ) (h : -1 < a ∧ a < 2) :
  ∀ x : ℝ, -3 < (x^2 + a * x - 2) / (x^2 - x + 1) ∧ (x^2 + a * x - 2) / (x^2 - x + 1) < 2 :=
by
  intro x
  sorry

end inequality_holds_for_all_x_l32_32567


namespace bus_trip_times_l32_32816

/-- Given two buses traveling towards each other from points A and B which are 120 km apart.
The first bus stops for 10 minutes and the second bus stops for 5 minutes. The first bus reaches 
its destination 25 minutes before the second bus. The first bus travels 20 km/h faster than the 
second bus. Prove that the travel times for the buses are 
1 hour 40 minutes and 2 hours 5 minutes respectively. -/
theorem bus_trip_times (d : ℕ) (v1 v2 : ℝ) (t1 t2 t : ℝ) (h1 : d = 120) (h2 : v1 = v2 + 20) 
(h3 : t1 = d / v1 + 10) (h4 : t2 = d / v2 + 5) (h5 : t2 - t1 = 25) :
t1 = 100 ∧ t2 = 125 := 
by 
  sorry

end bus_trip_times_l32_32816


namespace sqrt_2023_irrational_l32_32029

theorem sqrt_2023_irrational : ¬ ∃ (r : ℚ), r^2 = 2023 := by
  sorry

end sqrt_2023_irrational_l32_32029


namespace cubic_polynomial_roots_l32_32994

theorem cubic_polynomial_roots (a : ℚ) :
  (x^3 - 6*x^2 + a*x - 6 = 0) ∧ (x = 3) → (x = 1 ∨ x = 2 ∨ x = 3) :=
by
  sorry

end cubic_polynomial_roots_l32_32994


namespace three_digit_number_parity_count_equal_l32_32630

/-- Prove the number of three-digit numbers with all digits having the same parity is equal to the number of three-digit numbers where adjacent digits have different parity. -/
theorem three_digit_number_parity_count_equal :
  ∃ (same_parity_count alternating_parity_count : ℕ),
    same_parity_count = alternating_parity_count ∧
    -- Condition for digits of the same parity
    same_parity_count = (4 * 5 * 5) + (5 * 5 * 5) ∧
    -- Condition for alternating parity digits (patterns EOE and OEO)
    alternating_parity_count = (4 * 5 * 5) + (5 * 5 * 5) := by
  sorry

end three_digit_number_parity_count_equal_l32_32630


namespace negation_of_existence_l32_32008

variable (Triangle : Type) (has_circumcircle : Triangle → Prop)

theorem negation_of_existence :
  ¬ (∃ t : Triangle, ¬ has_circumcircle t) ↔ ∀ t : Triangle, has_circumcircle t :=
by sorry

end negation_of_existence_l32_32008


namespace intersection_A_B_l32_32685

open Set

noncomputable def A : Set ℝ := {x | log 2 x < 1}
noncomputable def B : Set ℝ := {x | x^2 + x - 2 ≤ 0}

theorem intersection_A_B : A ∩ B = {x | 0 < x ∧ x ≤ 1} :=
by
  sorry

end intersection_A_B_l32_32685


namespace solve_for_x_l32_32159

theorem solve_for_x (x : ℝ) (y : ℝ) (z : ℝ) (h1 : y = 1) (h2 : z = 3) (h3 : x^2 * y * z - x * y * z^2 = 6) :
  x = -2 / 3 ∨ x = 3 :=
by sorry

end solve_for_x_l32_32159


namespace relationship_y1_y2_l32_32692

theorem relationship_y1_y2 :
  ∀ (b y1 y2 : ℝ), 
  (∃ b y1 y2, y1 = -2023 * (-2) + b ∧ y2 = -2023 * (-1) + b) → y1 > y2 :=
by
  intro b y1 y2 h
  sorry

end relationship_y1_y2_l32_32692


namespace fruit_shop_apples_l32_32614

-- Given conditions
def morning_fraction : ℚ := 3 / 10
def afternoon_fraction : ℚ := 4 / 10
def total_sold : ℕ := 140

-- Define the total number of apples and the resulting condition
def total_fraction_sold : ℚ := morning_fraction + afternoon_fraction

theorem fruit_shop_apples (A : ℕ) (h : total_fraction_sold * A = total_sold) : A = 200 := 
by sorry

end fruit_shop_apples_l32_32614


namespace find_sales_discount_l32_32392

noncomputable def salesDiscountPercentage (P N : ℝ) (D : ℝ): Prop :=
  let originalGrossIncome := P * N
  let newPrice := P * (1 - D / 100)
  let newNumberOfItems := N * 1.20
  let newGrossIncome := newPrice * newNumberOfItems
  newGrossIncome = originalGrossIncome * 1.08

theorem find_sales_discount (P N : ℝ) (hP : P > 0) (hN : N > 0) (h: ∃ D, salesDiscountPercentage P N D) :
  ∃ D, D = 10 :=
sorry

end find_sales_discount_l32_32392


namespace sandy_remaining_puppies_l32_32128

-- Definitions from the problem
def initial_puppies : ℕ := 8
def given_away_puppies : ℕ := 4

-- Theorem statement
theorem sandy_remaining_puppies : initial_puppies - given_away_puppies = 4 := by
  sorry

end sandy_remaining_puppies_l32_32128


namespace percentage_less_than_y_l32_32094

variable (w x y z : ℝ)

-- Given conditions
variable (h1 : w = 0.60 * x)
variable (h2 : x = 0.60 * y)
variable (h3 : z = 1.50 * w)

theorem percentage_less_than_y : ( (y - z) / y) * 100 = 46 := by
  sorry

end percentage_less_than_y_l32_32094


namespace probability_of_at_least_40_cents_l32_32345

-- Definitions for each type of coin and their individual values in cents.
def penny := 1
def nickel := 5
def dime := 10
def quarter := 25
def half_dollar := 50

-- The total value needed for a successful outcome
def minimum_success_value := 40

-- Total number of possible outcomes from flipping 5 coins independently
def total_outcomes := 2^5

-- Count the successful outcomes that result in at least 40 cents
-- This is a placeholder for the actual successful counting method
noncomputable def successful_outcomes := 18

-- Calculate the probability of successful outcomes
noncomputable def probability := (successful_outcomes : ℚ) / total_outcomes

-- Proof statement to show the probability is 9/16
theorem probability_of_at_least_40_cents : probability = 9 / 16 := 
by
  sorry

end probability_of_at_least_40_cents_l32_32345


namespace complex_ratio_identity_l32_32724

variable {x y : ℂ}

theorem complex_ratio_identity :
  ( (x + y) / (x - y) - (x - y) / (x + y) = 3 ) →
  ( (x^4 + y^4) / (x^4 - y^4) - (x^4 - y^4) / (x^4 + y^4) = 49 / 600) :=
by
  sorry

end complex_ratio_identity_l32_32724


namespace min_value_of_expression_l32_32326

open Real

theorem min_value_of_expression (α β : ℝ) :
  (3 * cos α + 4 * sin β - 7)^2 + (3 * sin α + 4 * cos β - 12)^2 ≥ 36 :=
by
  sorry

end min_value_of_expression_l32_32326


namespace expression_evaluation_l32_32909

theorem expression_evaluation :
  (0.86^3) - ((0.1^3) / (0.86^2)) + 0.086 + (0.1^2) = 0.730704 := 
by 
  sorry

end expression_evaluation_l32_32909


namespace correction_amount_l32_32809

variable (x : ℕ)

def half_dollar := 50
def quarter := 25
def nickel := 5
def dime := 10

theorem correction_amount : 
  ∀ x, (x * (half_dollar - quarter)) - (x * (dime - nickel)) = 20 * x := by
  intros x 
  sorry

end correction_amount_l32_32809


namespace range_of_a_l32_32954

open Real

noncomputable def p (a : ℝ) := ∀ (x : ℝ), x ≥ 1 → (2 * x - 3 * a) ≥ 0
noncomputable def q (a : ℝ) := (0 < 2 * a - 1) ∧ (2 * a - 1 < 1)

theorem range_of_a (a : ℝ) : p a ∧ q a ↔ (1/2 < a ∧ a ≤ 2/3) := by
  sorry

end range_of_a_l32_32954


namespace Liam_chapters_in_fourth_week_l32_32333

noncomputable def chapters_in_first_week (x : ℕ) : ℕ := x
noncomputable def chapters_in_second_week (x : ℕ) : ℕ := x + 3
noncomputable def chapters_in_third_week (x : ℕ) : ℕ := x + 6
noncomputable def chapters_in_fourth_week (x : ℕ) : ℕ := x + 9
noncomputable def total_chapters (x : ℕ) : ℕ := x + (x + 3) + (x + 6) + (x + 9)

theorem Liam_chapters_in_fourth_week : ∃ x : ℕ, total_chapters x = 50 → chapters_in_fourth_week x = 17 :=
by
  sorry

end Liam_chapters_in_fourth_week_l32_32333


namespace pieces_length_l32_32697

theorem pieces_length :
  let total_length_meters := 29.75
  let number_of_pieces := 35
  let length_per_piece_meters := total_length_meters / number_of_pieces
  let length_per_piece_centimeters := length_per_piece_meters * 100
  length_per_piece_centimeters = 85 :=
by
  sorry

end pieces_length_l32_32697


namespace hall_ratio_l32_32619

variable (w l : ℝ)

theorem hall_ratio
  (h1 : w * l = 200)
  (h2 : l - w = 10) :
  w / l = 1 / 2 := 
by
  sorry

end hall_ratio_l32_32619


namespace probability_correct_l32_32933

/-
  Problem statement:
  Consider a modified city map where a student walks from intersection A to intersection B, passing through C and D.
  The student always walks east or south and at each intersection, decides the direction to go with a probability of 1/2.
  The map requires 4 eastward and 3 southward moves to reach B from A. C is 2 east, 1 south move from A. D is 3 east, 2 south moves from A.
  Prove that the probability the student goes through both C and D is 12/35.
-/

noncomputable def probability_passing_C_and_D : ℚ :=
  let total_paths_A_to_B := Nat.choose 7 4
  let paths_A_to_C := Nat.choose 3 2
  let paths_C_to_D := Nat.choose 2 1
  let paths_D_to_B := Nat.choose 2 1
  (paths_A_to_C * paths_C_to_D * paths_D_to_B) / total_paths_A_to_B

theorem probability_correct :
  probability_passing_C_and_D = 12 / 35 :=
by
  sorry

end probability_correct_l32_32933


namespace simplify_expression_l32_32343

theorem simplify_expression : 4 * (14 / 5) * (20 / -42) = -4 / 15 := 
by sorry

end simplify_expression_l32_32343


namespace cats_not_eating_either_l32_32841

theorem cats_not_eating_either (total_cats : ℕ) (cats_like_apples : ℕ) (cats_like_chicken : ℕ) (cats_like_both : ℕ) 
  (h1 : total_cats = 80)
  (h2 : cats_like_apples = 15)
  (h3 : cats_like_chicken = 60)
  (h4 : cats_like_both = 10) : 
  total_cats - (cats_like_apples + cats_like_chicken - cats_like_both) = 15 :=
by sorry

end cats_not_eating_either_l32_32841


namespace no_valid_n_for_conditions_l32_32566

theorem no_valid_n_for_conditions :
  ¬∃ n : ℕ, 1000 ≤ n / 4 ∧ n / 4 ≤ 9999 ∧ 1000 ≤ 4 * n ∧ 4 * n ≤ 9999 := by
  sorry

end no_valid_n_for_conditions_l32_32566


namespace complete_the_square_sum_l32_32464

theorem complete_the_square_sum :
  ∃ p q : ℝ, (∀ x : ℝ, 15 * x^2 - 30 * x - 60 = 0 → (x + p)^2 = q) ∧ p + q = 1 :=
by 
  sorry

end complete_the_square_sum_l32_32464


namespace paint_coverage_is_10_l32_32971

noncomputable def paintCoverage (cost_per_quart : ℝ) (cube_edge_length : ℝ) (total_cost : ℝ) : ℝ :=
  let total_surface_area := 6 * (cube_edge_length ^ 2)
  let number_of_quarts := total_cost / cost_per_quart
  total_surface_area / number_of_quarts

theorem paint_coverage_is_10 :
  paintCoverage 3.2 10 192 = 10 :=
by
  sorry

end paint_coverage_is_10_l32_32971


namespace circle_tangent_line_standard_equation_l32_32564

-- Problem Statement:
-- Prove that the standard equation of the circle with center at (1,1)
-- and tangent to the line x + y = 4 is (x - 1)^2 + (y - 1)^2 = 2
theorem circle_tangent_line_standard_equation :
  (forall (x y : ℝ), (x + y = 4) -> (x - 1)^2 + (y - 1)^2 = 2) := by
  sorry

end circle_tangent_line_standard_equation_l32_32564


namespace sum_of_k_with_distinct_integer_solutions_l32_32365

-- Definitions from conditions
def quadratic_eq (k : ℤ) (x : ℤ) : Prop := 3 * x^2 - k * x + 12 = 0

def distinct_integer_solutions (k : ℤ) : Prop := ∃ p q : ℤ, p ≠ q ∧ quadratic_eq k p ∧ quadratic_eq k q

-- Main statement to prove
theorem sum_of_k_with_distinct_integer_solutions : 
  ∑ k in {k : ℤ | distinct_integer_solutions k}, k = 0 := 
sorry

end sum_of_k_with_distinct_integer_solutions_l32_32365


namespace largest_integer_solution_l32_32877

theorem largest_integer_solution :
  ∀ (x : ℤ), x - 5 > 3 * x - 1 → x ≤ -3 := by
  sorry

end largest_integer_solution_l32_32877


namespace find_ratio_l32_32425

theorem find_ratio (a b : ℝ) (h : a / b = 3 / 2) : (a + b) / a = 5 / 3 :=
sorry

end find_ratio_l32_32425


namespace find_m_l32_32137

theorem find_m {x1 x2 m : ℝ} 
  (h_eqn : ∀ x, x^2 - (m+3)*x + (m+2) = 0) 
  (h_cond : x1 / (x1 + 1) + x2 / (x2 + 1) = 13 / 10) : 
  m = 2 := 
sorry

end find_m_l32_32137


namespace find_y_l32_32132

theorem find_y (t : ℝ) (x y : ℝ) (h1 : x = 3 - 2 * t) (h2 : y = 5 * t + 3) (h3 : x = -7) : y = 28 :=
by {
  sorry
}

end find_y_l32_32132


namespace last_digit_one_over_three_pow_neg_ten_l32_32777

theorem last_digit_one_over_three_pow_neg_ten : (3^10) % 10 = 9 := by
  sorry

end last_digit_one_over_three_pow_neg_ten_l32_32777


namespace largest_six_digit_number_l32_32050

/-- The largest six-digit number \( A \) that is divisible by 19, 
  the number obtained by removing its last digit is divisible by 17, 
  and the number obtained by removing the last two digits in \( A \) is divisible by 13 
  is \( 998412 \). -/
theorem largest_six_digit_number (A : ℕ) (h1 : A % 19 = 0) 
  (h2 : (A / 10) % 17 = 0) 
  (h3 : (A / 100) % 13 = 0) : 
  A = 998412 :=
sorry

end largest_six_digit_number_l32_32050


namespace problem_statement_l32_32842

def P (m n : ℕ) : ℕ :=
  let coeff_x := Nat.choose 4 m
  let coeff_y := Nat.choose 6 n
  coeff_x * coeff_y

theorem problem_statement : P 2 1 + P 1 2 = 96 :=
by
  sorry

end problem_statement_l32_32842


namespace max_x_y_given_condition_l32_32574

theorem max_x_y_given_condition (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y + 1/x + 1/y = 5) : x + y ≤ 4 :=
sorry

end max_x_y_given_condition_l32_32574


namespace solve_for_k_l32_32807

theorem solve_for_k : {k : ℕ | ∀ x : ℝ, (x^2 - 1)^(2*k) + (x^2 + 2*x)^(2*k) + (2*x + 1)^(2*k) = 2*(1 + x + x^2)^(2*k)} = {1, 2} :=
sorry

end solve_for_k_l32_32807


namespace volume_of_cube_with_surface_area_l32_32193

theorem volume_of_cube_with_surface_area (S : ℝ) (hS : S = 294) : 
  ∃ V : ℝ, V = 343 :=
by
  let s := (S / 6).sqrt
  have hs : s = 7 := by sorry
  use s ^ 3
  simp [hs]
  exact sorry

end volume_of_cube_with_surface_area_l32_32193


namespace arithmetic_geometric_sequences_l32_32144

theorem arithmetic_geometric_sequences :
  ∃ (A B C D : ℤ), A < B ∧ B > 0 ∧ C > 0 ∧ -- Ensure A, B, C are positive
  (B - A) = (C - B) ∧  -- Arithmetic sequence condition
  B * (49 : ℚ) = C * (49 / 9 : ℚ) ∧ -- Geometric sequence condition written using fractional equality
  A + B + C + D = 76 := 
by {
  sorry -- Placeholder for actual proof
}

end arithmetic_geometric_sequences_l32_32144


namespace series_sum_eq_50_l32_32065

noncomputable def series_sum (x : ℝ) : ℝ :=
  2 + 6 * x + 10 * x^2 + 14 * x^3 -- This represents the series

theorem series_sum_eq_50 : 
  ∃ x : ℝ, series_sum x = 50 ∧ x = 0.59 :=
by
  sorry

end series_sum_eq_50_l32_32065


namespace union_complement_l32_32110

open Set

variable (U A B : Set ℕ)
variable (u_spec : U = {1, 2, 3, 4, 5})
variable (a_spec : A = {1, 2, 3})
variable (b_spec : B = {2, 4})

theorem union_complement (U A B : Set ℕ)
  (u_spec : U = {1, 2, 3, 4, 5})
  (a_spec : A = {1, 2, 3})
  (b_spec : B = {2, 4}) :
  A ∪ (U \ B) = {1, 2, 3, 5} := by
  sorry

end union_complement_l32_32110


namespace total_bees_is_25_l32_32755

def initial_bees : ℕ := 16
def additional_bees : ℕ := 9

theorem total_bees_is_25 : initial_bees + additional_bees = 25 := by
  sorry

end total_bees_is_25_l32_32755


namespace height_of_pyramid_equal_to_cube_volume_l32_32197

theorem height_of_pyramid_equal_to_cube_volume :
  (∃ h : ℝ, (5:ℝ)^3 = (1/3:ℝ) * (10:ℝ)^2 * h) ↔ h = 3.75 :=
by
  sorry

end height_of_pyramid_equal_to_cube_volume_l32_32197


namespace hunting_dogs_theorem_l32_32500

noncomputable def hunting_dogs_problem : Prop :=
  ∃ (courtiers : Finset (Finset (Fin 100))) (h1 : courtiers.card = 100),
  ∀ (c1 c2 : Finset (Fin 100)), c1 ∈ courtiers → c2 ∈ courtiers → c1 ≠ c2 → (c1 ∩ c2).card ≥ 2 → 
  ∃ (c₁ c₂ : Finset (Fin 100)), c₁ ∈ courtiers ∧ c₂ ∈ courtiers ∧ c₁ = c₂

theorem hunting_dogs_theorem : hunting_dogs_problem :=
sorry

end hunting_dogs_theorem_l32_32500


namespace value_of_x_l32_32948

theorem value_of_x (x : ℝ) (a : ℝ) (h1 : x ^ 2 * 8 ^ 3 / 256 = a) (h2 : a = 450) : x = 15 ∨ x = -15 := by
  sorry

end value_of_x_l32_32948


namespace simplify_fraction_l32_32784

theorem simplify_fraction (c : ℚ) : (6 + 5 * c) / 9 + 3 = (33 + 5 * c) / 9 := 
sorry

end simplify_fraction_l32_32784


namespace powerThreeExpression_l32_32239

theorem powerThreeExpression (x : ℝ) (h : x - (1 / x) = 5) : x^3 - (1 / (x^3)) = 140 := sorry

end powerThreeExpression_l32_32239


namespace handrail_length_is_17_point_3_l32_32653

noncomputable def length_of_handrail (turn : ℝ) (rise : ℝ) (radius : ℝ) : ℝ :=
  let circumference := 2 * Real.pi * radius
  let arc_length := (turn / 360) * circumference
  Real.sqrt (rise^2 + arc_length^2)

theorem handrail_length_is_17_point_3 : length_of_handrail 270 10 3 = 17.3 :=
by 
  sorry

end handrail_length_is_17_point_3_l32_32653


namespace students_same_group_in_all_lessons_l32_32638

theorem students_same_group_in_all_lessons (students : Fin 28 → Fin 3 × Fin 3 × Fin 3) :
  ∃ (i j : Fin 28), i ≠ j ∧ students i = students j :=
by
  sorry

end students_same_group_in_all_lessons_l32_32638


namespace rectangle_width_is_16_l32_32726

-- Definitions based on the conditions
def length : ℝ := 24
def ratio := 6 / 5
def perimeter := 80

-- The proposition to prove
theorem rectangle_width_is_16 (W : ℝ) (h1 : length = 24) (h2 : length = ratio * W) (h3 : 2 * length + 2 * W = perimeter) :
  W = 16 :=
by
  sorry

end rectangle_width_is_16_l32_32726


namespace possible_values_a_possible_values_m_l32_32852

noncomputable def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | (x - 1) * (x - a + 1) = 0}
noncomputable def C (m : ℝ) : Set ℝ := {x | x^2 - m * x + 2 = 0}

theorem possible_values_a (a : ℝ) : 
  (A ∪ B a = A) → a = 2 ∨ a = 3 := sorry

theorem possible_values_m (m : ℝ) : 
  (A ∩ C m = C m) → (m = 3 ∨ (-2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2)) := sorry

end possible_values_a_possible_values_m_l32_32852


namespace distance_travelled_is_960_l32_32383

-- Definitions based on conditions
def speed_slower := 60 -- Speed of slower bike in km/h
def speed_faster := 64 -- Speed of faster bike in km/h
def time_diff := 1 -- Time difference in hours

-- Problem statement: Prove that the distance covered by both bikes is 960 km.
theorem distance_travelled_is_960 (T : ℝ) (D : ℝ) 
  (h1 : D = speed_slower * T)
  (h2 : D = speed_faster * (T - time_diff)) :
  D = 960 := 
sorry

end distance_travelled_is_960_l32_32383


namespace courtiers_selection_l32_32502

theorem courtiers_selection (dogs : Finset ℕ) (courtiers : Finset (Finset ℕ))
  (h1 : dogs.card = 100)
  (h2 : courtiers.card = 100)
  (h3 : ∀ c ∈ courtiers, c.card = 3)
  (h4 : ∀ {c1 c2 : Finset ℕ}, c1 ∈ courtiers → c2 ∈ courtiers → c1 ≠ c2 → (c1 ∩ c2).card = 2) :
  ∃ c1 c2 ∈ courtiers, c1 = c2 :=
sorry

end courtiers_selection_l32_32502


namespace camels_horses_oxen_elephants_l32_32041

theorem camels_horses_oxen_elephants :
  ∀ (C H O E : ℝ),
  10 * C = 24 * H →
  H = 4 * O →
  6 * O = 4 * E →
  10 * E = 170000 →
  C = 4184.615384615385 →
  (4 * O) / H = 1 :=
by
  intros C H O E h1 h2 h3 h4 h5
  sorry

end camels_horses_oxen_elephants_l32_32041


namespace sum_first_five_terms_arithmetic_seq_l32_32702

theorem sum_first_five_terms_arithmetic_seq
  (a : ℕ → ℕ)
  (h_arith_seq : ∀ n, a n = a 0 + n * (a 1 - a 0))
  (h_a2 : a 2 = 5)
  (h_a4 : a 4 = 9)
  : (Finset.range 5).sum a = 35 := by
  sorry

end sum_first_five_terms_arithmetic_seq_l32_32702


namespace solve_inequalities_l32_32671

-- Define the interval [-1, 1]
def interval := {x : ℝ | -1 ≤ x ∧ x ≤ 1}

-- State the problem
theorem solve_inequalities :
  {x : ℝ | 3 * x^2 + 2 * x - 9 ≤ 0 ∧ x ≥ -1} = interval := 
sorry

end solve_inequalities_l32_32671


namespace emissions_from_tap_water_l32_32351

def carbon_dioxide_emission (x : ℕ) : ℕ := 9 / 10 * x  -- Note: using 9/10 instead of 0.9 to maintain integer type

theorem emissions_from_tap_water : carbon_dioxide_emission 10 = 9 :=
by
  sorry

end emissions_from_tap_water_l32_32351


namespace find_a_if_perpendicular_l32_32445

theorem find_a_if_perpendicular (a : ℝ) :
  (∀ x y : ℝ, x + a * y + 2 = 0 → 2 * x + 3 * y + 1 = 0 → False) →
  a = -2 / 3 :=
by
  sorry

end find_a_if_perpendicular_l32_32445


namespace vector_subtraction_l32_32950

-- Definitions of given conditions
def OA : ℝ × ℝ := (2, 1)
def OB : ℝ × ℝ := (-3, 4)

-- Definition of vector subtraction
def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 - v2.1, v1.2 - v2.2)

theorem vector_subtraction : vector_sub OB OA = (-5, 3) :=
by 
  -- The proof would go here.
  sorry

end vector_subtraction_l32_32950


namespace cuboid_second_edge_l32_32875

variable (x : ℝ)

theorem cuboid_second_edge (h1 : 4 * x * 6 = 96) : x = 4 := by
  sorry

end cuboid_second_edge_l32_32875


namespace sum_of_x_y_is_13_l32_32832

theorem sum_of_x_y_is_13 (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h : x^4 + y^4 = 4721) : x + y = 13 :=
sorry

end sum_of_x_y_is_13_l32_32832


namespace calc1_calc2_calc3_calc4_l32_32549

theorem calc1 : 23 + (-16) - (-7) = 14 :=
by
  sorry

theorem calc2 : (3/4 - 7/8 - 5/12) * (-24) = 13 :=
by
  sorry

theorem calc3 : ((7/4 - 7/8 - 7/12) / (-7/8)) + ((-7/8) / (7/4 - 7/8 - 7/12)) = -10/3 :=
by
  sorry

theorem calc4 : -1^4 - (1 - 0.5) * (1/3) * (2 - (-3)^2) = 1/6 :=
by
  sorry

end calc1_calc2_calc3_calc4_l32_32549


namespace cubic_identity_l32_32258

theorem cubic_identity (x : ℝ) (h : x - 1/x = 5) : x^3 - 1/x^3 = 140 :=
by
  sorry

end cubic_identity_l32_32258


namespace multiplication_of_monomials_l32_32545

-- Define the constants and assumptions
def a : ℝ := -2
def b : ℝ := 4
def e1 : ℤ := 4
def e2 : ℤ := 5
def result : ℝ := -8
def result_exp : ℤ := 9

-- State the theorem to be proven
theorem multiplication_of_monomials :
  (a * 10^e1) * (b * 10^e2) = result * 10^result_exp := 
by
  sorry

end multiplication_of_monomials_l32_32545


namespace tournament_rounds_l32_32605

/-- 
Given a tournament where each participant plays several games with every other participant
and a total of 224 games were played, prove that the number of rounds in the competition is 8.
-/
theorem tournament_rounds (x y : ℕ) (hx : x > 1) (hy : y > 0) (h : x * (x - 1) * y = 448) : y = 8 :=
sorry

end tournament_rounds_l32_32605


namespace point_Q_in_third_quadrant_l32_32281

-- Define point P in the fourth quadrant with coordinates a and b.
variable (a b : ℝ)
variable (h1 : a > 0)  -- Condition for the x-coordinate of P in fourth quadrant
variable (h2 : b < 0)  -- Condition for the y-coordinate of P in fourth quadrant

-- Point Q is defined by the coordinates (-a, b-1). We need to show it lies in the third quadrant.
theorem point_Q_in_third_quadrant : (-a < 0) ∧ (b - 1 < 0) :=
  by
    sorry

end point_Q_in_third_quadrant_l32_32281


namespace sum_max_min_f_l32_32148

noncomputable def f (x : ℝ) : ℝ :=
  1 + (Real.sin x / (2 + Real.cos x))

theorem sum_max_min_f {a b : ℝ} (ha : ∀ x, f x ≤ a) (hb : ∀ x, b ≤ f x) (h_max : ∃ x, f x = a) (h_min : ∃ x, f x = b) :
  a + b = 2 :=
sorry

end sum_max_min_f_l32_32148


namespace fruit_weights_l32_32117

theorem fruit_weights
  (Mandarin Apple Banana Peach Orange : ℕ)
  (h1 : Peach < Orange)
  (h2 : Apple < Banana ∧ Banana < Peach)
  (h3 : Mandarin < Banana)
  (h4 : Apple + Banana > Orange) :
  ({Mandarin, Apple, Banana, Peach, Orange} = {100, 150, 170, 200, 280}) :=
by
  sorry

end fruit_weights_l32_32117


namespace vector_minimization_and_angle_condition_l32_32238

noncomputable def find_OC_condition (C_op C_oa C_ob : ℝ × ℝ) 
  (C : ℝ × ℝ) : Prop := 
  let CA := (C_oa.1 - C.1, C_oa.2 - C.2)
  let CB := (C_ob.1 - C.1, C_ob.2 - C.2)
  (CA.1 * CB.1 + CA.2 * CB.2) ≤ (C_op.1 * CB.1 + C_op.2 * CB.2)

theorem vector_minimization_and_angle_condition (C : ℝ × ℝ) 
  (C_op := (2, 1)) (C_oa := (1, 7)) (C_ob := (5, 1)) :
  (C = (4, 2)) → 
  find_OC_condition C_op C_oa C_ob C →
  let CA := (C_oa.1 - C.1, C_oa.2 - C.2)
  let CB := (C_ob.1 - C.1, C_ob.2 - C.2)
  let cos_ACB := (CA.1 * CB.1 + CA.2 * CB.2) / 
                 (Real.sqrt (CA.1^2 + CA.2^2) * Real.sqrt (CB.1^2 + CB.2^2))
  cos_ACB = -4 * Real.sqrt (17) / 17 :=
  by 
    intro h1 find
    let CA := (C_oa.1 - C.1, C_oa.2 - C.2)
    let CB := (C_ob.1 - C.1, C_ob.2 - C.2)
    let cos_ACB := (CA.1 * CB.1 + CA.2 * CB.2) / 
                   (Real.sqrt (CA.1^2 + CA.2^2) * Real.sqrt (CB.1^2 + CB.2^2))
    exact sorry

end vector_minimization_and_angle_condition_l32_32238


namespace N_even_for_all_permutations_l32_32958

noncomputable def N (a b : Fin 2013 → ℕ) : ℕ :=
  Finset.prod (Finset.univ : Finset (Fin 2013)) (λ i => a i - b i)

theorem N_even_for_all_permutations {a : Fin 2013 → ℕ}
  (h_distinct : Function.Injective a) :
  ∀ b : Fin 2013 → ℕ,
  (∀ i, b i ∈ Finset.univ.image a) →
  ∃ n, n = N a b ∧ Even n :=
by
  -- This is where the proof would go, using the given conditions.
  sorry

end N_even_for_all_permutations_l32_32958


namespace f_2009_l32_32431

noncomputable def f : ℝ → ℝ := sorry -- This will be defined by the conditions.

axiom even_f (x : ℝ) : f x = f (-x)
axiom periodic_f (x : ℝ) : f (x + 6) = f x + f 3
axiom f_one : f 1 = 2

theorem f_2009 : f 2009 = 2 :=
by {
  -- The proof would go here, summarizing the logical steps derived in the previous sections.
  sorry
}

end f_2009_l32_32431


namespace find_q_l32_32969

theorem find_q (p q : ℚ) (h1 : 5 * p + 7 * q = 20) (h2 : 7 * p + 5 * q = 26) : q = 5 / 12 := by
  sorry

end find_q_l32_32969


namespace function_characterization_l32_32669

def isRelativelyPrime (x y : ℕ) : Prop := Nat.gcd x y = 1

theorem function_characterization (f : ℕ → ℤ) (hyp : ∀ x y, isRelativelyPrime x y → f (x + y) = f (x + 1) + f (y + 1)) :
  ∃ a b : ℤ, ∀ n : ℕ, f (2 * n) = (n - 1) * b ∧ f (2 * n + 1) = (n - 1) * b + a :=
by
  sorry

end function_characterization_l32_32669


namespace antonio_weight_l32_32542

-- Let A be the weight of Antonio
variable (A : ℕ)

-- Conditions:
-- 1. Antonio's sister weighs A - 12 kilograms.
-- 2. The total weight of Antonio and his sister is 88 kilograms.

theorem antonio_weight (A: ℕ) (h1: A - 12 >= 0) (h2: A + (A - 12) = 88) : A = 50 := by
  sorry

end antonio_weight_l32_32542


namespace process_terminates_with_one_element_in_each_list_final_elements_are_different_l32_32156

-- Define the initial lists
def List1 := [1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 81, 86, 91, 96]
def List2 := [4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84, 89, 94, 99]

-- Predicate to state the termination of the process with exactly one element in each list
theorem process_terminates_with_one_element_in_each_list (List1 List2 : List ℕ):
  ∃ n m, List.length List1 = n ∧ List.length List2 = m ∧ (n = 1 ∧ m = 1) :=
sorry

-- Predicate to state that the final elements in the lists are different
theorem final_elements_are_different (List1 List2 : List ℕ) :
  ∀ a b, a ∈ List1 → b ∈ List2 → (a % 5 = 1 ∧ b % 5 = 4) → a ≠ b :=
sorry

end process_terminates_with_one_element_in_each_list_final_elements_are_different_l32_32156


namespace find_other_number_l32_32479

theorem find_other_number (HCF LCM a b : ℕ) (h1 : HCF = 108) (h2 : LCM = 27720) (h3 : a = 216) (h4 : HCF * LCM = a * b) : b = 64 :=
  sorry

end find_other_number_l32_32479


namespace find_angle_between_vectors_l32_32720

open Real

noncomputable def vector_norm (v : ℝ × ℝ) : ℝ := sqrt (v.1 ^ 2 + v.2 ^ 2)
noncomputable def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem find_angle_between_vectors 
  (a b : ℝ × ℝ)
  (a_nonzero : a ≠ (0, 0))
  (b_nonzero : b ≠ (0, 0))
  (ha : vector_norm a = 2)
  (hb : vector_norm b = 3)
  (h_sum : vector_norm (a.1 + b.1, a.2 + b.2) = 1)
  : arccos (dot_product a b / (vector_norm a * vector_norm b)) = π :=
sorry

end find_angle_between_vectors_l32_32720


namespace union_A_B_inter_A_compl_B_range_of_a_l32_32597

-- Define the sets A, B, and C
def A := {x : ℝ | -1 ≤ x ∧ x < 3}
def B := {x : ℝ | 2 * x - 4 ≥ x - 2}
def C (a : ℝ) := {x : ℝ | x ≥ a - 1}

-- Prove A ∪ B = {x | -1 ≤ x}
theorem union_A_B : A ∪ B = {x : ℝ | -1 ≤ x} :=
by sorry

-- Prove A ∩ (complement B) = {x | -1 ≤ x < 2}
theorem inter_A_compl_B : A ∩ (compl B) = {x : ℝ | -1 ≤ x ∧ x < 2} :=
by sorry

-- Prove the range of a given B ∪ C = C
theorem range_of_a (a : ℝ) (h : B ∪ C a = C a) : a ≤ 3 :=
by sorry

end union_A_B_inter_A_compl_B_range_of_a_l32_32597


namespace min_value_expression_l32_32329

open Real

theorem min_value_expression (α β : ℝ) :
  (3 * cos α + 4 * sin β - 7)^2 + (3 * sin α + 4 * cos β - 12)^2 ≥ 36 := by
  sorry

end min_value_expression_l32_32329


namespace percentage_of_masters_l32_32920

theorem percentage_of_masters (x y : ℕ) (avg_juniors avg_masters avg_team : ℚ) 
  (h1 : avg_juniors = 22)
  (h2 : avg_masters = 47)
  (h3 : avg_team = 41)
  (h4 : 22 * x + 47 * y = 41 * (x + y)) : 
  76% of the team are masters := by sorry

end percentage_of_masters_l32_32920


namespace initial_rate_of_commission_is_4_l32_32485

noncomputable def initial_commission_rate (B : ℝ) (x : ℝ) : Prop :=
  B * (x / 100) = 0.8 * B * (5 / 100)

theorem initial_rate_of_commission_is_4 (B : ℝ) (hB : B > 0) :
  initial_commission_rate B 4 :=
by
  unfold initial_commission_rate
  sorry

end initial_rate_of_commission_is_4_l32_32485


namespace false_statement_d_l32_32828

-- Define lines and planes
variables (l m : Type*) (α β : Type*)

-- Define parallel relation
def parallel (l m : Type*) : Prop := sorry

-- Define subset relation
def in_plane (l : Type*) (α : Type*) : Prop := sorry

-- Define the given conditions
axiom l_parallel_alpha : parallel l α
axiom m_in_alpha : in_plane m α

-- Main theorem statement: prove \( l \parallel m \) is false given the conditions.
theorem false_statement_d : ¬ parallel l m :=
sorry

end false_statement_d_l32_32828


namespace required_words_to_learn_l32_32089

def total_words : ℕ := 500
def required_percentage : ℕ := 85

theorem required_words_to_learn (x : ℕ) :
  (x : ℚ) / total_words ≥ (required_percentage : ℚ) / 100 ↔ x ≥ 425 := 
sorry

end required_words_to_learn_l32_32089


namespace exist_two_courtiers_with_same_selection_l32_32487

noncomputable theory

-- Definitions
def num_dogs := 100
def num_courtiers := 100
def selected_dogs (courtier : Fin num_courtiers) : Finset (Fin num_dogs) := sorry
def num_selected_dogs := 3 

-- Conditions
axiom three_dogs_selected_each (c : Fin num_courtiers) : selected_dogs c.card = num_selected_dogs

axiom two_common_dogs (c₁ c₂ : Fin num_courtiers) (h : c₁ ≠ c₂) : (selected_dogs c₁ ∩ selected_dogs c₂).card ≥ 2

-- Proof goal
theorem exist_two_courtiers_with_same_selection : 
  ∃ (c₁ c₂ : Fin num_courtiers), c₁ ≠ c₂ ∧ selected_dogs c₁ = selected_dogs c₂ :=
sorry

end exist_two_courtiers_with_same_selection_l32_32487


namespace BobsFruitDrinkCost_l32_32802

theorem BobsFruitDrinkCost 
  (AndySpent : ℕ)
  (BobSpent : ℕ)
  (AndySodaCost : ℕ)
  (AndyHamburgerCost : ℕ)
  (BobSandwichCost : ℕ)
  (FruitDrinkCost : ℕ) :
  AndySpent = 5 ∧ AndySodaCost = 1 ∧ AndyHamburgerCost = 2 ∧ 
  AndySpent = BobSpent ∧ 
  BobSandwichCost = 3 ∧ 
  FruitDrinkCost = BobSpent - BobSandwichCost →
  FruitDrinkCost = 2 := by
  sorry

end BobsFruitDrinkCost_l32_32802


namespace cube_difference_l32_32247

variable (x : ℝ)

theorem cube_difference (h : x - 1/x = 5) : x^3 - 1/x^3 = 120 := by
  sorry

end cube_difference_l32_32247


namespace total_students_in_high_school_l32_32600

theorem total_students_in_high_school (selected_first: ℕ) (selected_second: ℕ) (students_third: ℕ) (total_selected: ℕ) (p: ℚ) :
  selected_first = 15 →
  selected_second = 12 →
  students_third = 900 →
  total_selected = 36 →
  p = 1 / 100 →
  ∃ n: ℕ, (total_selected : ℚ) / n = p ∧ n = 3600 :=
by 
  intros h1 h2 h3 h4 h5
  use 3600
  split
  · sorry -- omit the proof for successful compilation.
  · exact rfl

end total_students_in_high_school_l32_32600


namespace proof_l32_32953

-- Define proposition p as negated form: ∀ x < 1, log_3 x ≤ 0
def p : Prop := ∀ x : ℝ, x < 1 → Real.log x / Real.log 3 ≤ 0

-- Define proposition q: ∃ x_0 ∈ ℝ, x_0^2 ≥ 2^x_0
def q : Prop := ∃ x_0 : ℝ, x_0^2 ≥ Real.exp (x_0 * Real.log 2)

-- State we need to prove: p ∨ q
theorem proof : p ∨ q := sorry

end proof_l32_32953


namespace car_return_speed_l32_32389

theorem car_return_speed (d : ℕ) (speed_CD : ℕ) (avg_speed_round_trip : ℕ) 
  (round_trip_distance : ℕ) (time_CD : ℕ) (time_round_trip : ℕ) (r: ℕ) 
  (h1 : d = 150) (h2 : speed_CD = 75) (h3 : avg_speed_round_trip = 60)
  (h4 : d * 2 = round_trip_distance) 
  (h5 : time_CD = d / speed_CD) 
  (h6 : time_round_trip = time_CD + d / r) 
  (h7 : avg_speed_round_trip = round_trip_distance / time_round_trip) :
  r = 50 :=
by {
  -- proof steps will go here
  sorry
}

end car_return_speed_l32_32389


namespace cube_volume_from_surface_area_l32_32190

theorem cube_volume_from_surface_area (A : ℝ) (hA : A = 294) : ∃ V : ℝ, V = 343 :=
by
  let s := real.sqrt (A / 6)
  have h_s : s = 7 := sorry
  let V := s^3
  have h_V : V = 343 := sorry
  exact ⟨V, h_V⟩

end cube_volume_from_surface_area_l32_32190


namespace abs_inequality_solution_l32_32618

theorem abs_inequality_solution (x : ℝ) : |2 * x - 5| > 1 ↔ x < 2 ∨ x > 3 := sorry

end abs_inequality_solution_l32_32618


namespace find_original_price_l32_32402

theorem find_original_price (a b x : ℝ) (h : x * (1 - 0.1) - a = b) : 
  x = (a + b) / (1 - 0.1) :=
sorry

end find_original_price_l32_32402


namespace parker_daily_earning_l32_32995

-- Definition of conditions
def total_earned : ℕ := 2646
def weeks_worked : ℕ := 6
def days_per_week : ℕ := 7
def total_days (weeks : ℕ) (days_in_week : ℕ) : ℕ := weeks * days_in_week

-- Proof statement
theorem parker_daily_earning (h : total_days weeks_worked days_per_week = 42) : (total_earned / 42) = 63 :=
by
  sorry

end parker_daily_earning_l32_32995


namespace min_value_of_reciprocal_sum_l32_32302

-- Define the problem
theorem min_value_of_reciprocal_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : ∃ x y : ℝ, (x^2 + y^2 + 2 * x - 4 * y + 1 = 0) ∧ (2 * a * x - b * y + 2 = 0)):
  ∃ (m : ℝ), m = 4 ∧ (1 / a + 1 / b) ≥ m :=
by
  sorry

end min_value_of_reciprocal_sum_l32_32302


namespace quadratic_one_solution_m_value_l32_32667

theorem quadratic_one_solution_m_value (m : ℝ) : (∃ x : ℝ, 3 * x^2 - 6 * x + m = 0) → (b^2 - 4 * a * m = 0) → m = 3 :=
by
  sorry

end quadratic_one_solution_m_value_l32_32667


namespace overlapping_squares_area_l32_32515

theorem overlapping_squares_area :
  let s : ℝ := 5
  let total_area := 3 * s^2
  let redundant_area := s^2 / 8 * 4
  total_area - redundant_area = 62.5 := by
  sorry

end overlapping_squares_area_l32_32515


namespace lemonade_quart_calculation_l32_32831

-- Define the conditions
def water_parts := 5
def lemon_juice_parts := 3
def total_parts := water_parts + lemon_juice_parts

def gallons := 2
def quarts_per_gallon := 4
def total_quarts := gallons * quarts_per_gallon
def quarts_per_part := total_quarts / total_parts

-- Proof problem
theorem lemonade_quart_calculation :
  let water_quarts := water_parts * quarts_per_part
  let lemon_juice_quarts := lemon_juice_parts * quarts_per_part
  water_quarts = 5 ∧ lemon_juice_quarts = 3 :=
by
  let water_quarts := water_parts * quarts_per_part
  let lemon_juice_quarts := lemon_juice_parts * quarts_per_part
  have h_w : water_quarts = 5 := sorry
  have h_l : lemon_juice_quarts = 3 := sorry
  exact ⟨h_w, h_l⟩

end lemonade_quart_calculation_l32_32831


namespace inequality_problem_l32_32081

theorem inequality_problem
  (a b c d : ℝ)
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
  (h_sum : a + b + c + d = 1) :
  (a^2 / (1 + a)) + (b^2 / (1 + b)) + (c^2 / (1 + c)) + (d^2 / (1 + d)) ≥ 1 / 5 :=
by
  sorry

end inequality_problem_l32_32081


namespace pow_evaluation_l32_32678

theorem pow_evaluation : 81^(5/4) = 243 := 
by sorry

end pow_evaluation_l32_32678


namespace tim_pays_300_l32_32760

def mri_cost : ℕ := 1200
def doctor_rate_per_hour : ℕ := 300
def examination_time_in_hours : ℕ := 1 / 2
def consultation_fee : ℕ := 150
def insurance_coverage : ℚ := 0.8

def examination_cost : ℕ := doctor_rate_per_hour * examination_time_in_hours
def total_cost_before_insurance : ℕ := mri_cost + examination_cost + consultation_fee
def insurance_coverage_amount : ℚ := total_cost_before_insurance * insurance_coverage
def amount_tim_pays : ℚ := total_cost_before_insurance - insurance_coverage_amount

theorem tim_pays_300 : amount_tim_pays = 300 := 
by
  -- proof goes here
  sorry

end tim_pays_300_l32_32760


namespace calculation1_calculation2_calculation3_calculation4_l32_32404

-- Proving the first calculation: 3 * 232 + 456 = 1152
theorem calculation1 : 3 * 232 + 456 = 1152 := 
by 
  sorry

-- Proving the second calculation: 760 * 5 - 2880 = 920
theorem calculation2 : 760 * 5 - 2880 = 920 :=
by 
  sorry

-- Proving the third calculation: 805 / 7 = 115 (integer division)
theorem calculation3 : 805 / 7 = 115 :=
by 
  sorry

-- Proving the fourth calculation: 45 + 255 / 5 = 96
theorem calculation4 : 45 + 255 / 5 = 96 :=
by 
  sorry

end calculation1_calculation2_calculation3_calculation4_l32_32404


namespace expand_expression_l32_32416

variable (x y z : ℕ)

theorem expand_expression (x y z: ℕ) : 
  (x + 10) * (3 * y + 5 * z + 15) = 3 * x * y + 5 * x * z + 15 * x + 30 * y + 50 * z + 150 :=
by
  sorry

end expand_expression_l32_32416


namespace convert_mixed_decimals_to_fractions_l32_32806

theorem convert_mixed_decimals_to_fractions :
  (4.26 = 4 + 13/50) ∧
  (1.15 = 1 + 3/20) ∧
  (3.08 = 3 + 2/25) ∧
  (2.37 = 2 + 37/100) :=
by
  -- Proof omitted
  sorry

end convert_mixed_decimals_to_fractions_l32_32806


namespace batsman_average_runs_l32_32003

theorem batsman_average_runs
  (average_20_matches : ℕ → ℕ)
  (average_10_matches : ℕ → ℕ)
  (h1 : average_20_matches = 20 * 40)
  (h2 : average_10_matches = 10 * 13) :
  (average_20_matches + average_10_matches) / 30 = 31 := 
by 
  sorry

end batsman_average_runs_l32_32003


namespace cubic_identity_l32_32254

theorem cubic_identity (x : ℝ) (h : x - 1/x = 5) : x^3 - 1/x^3 = 140 :=
by
  sorry

end cubic_identity_l32_32254


namespace goose_eggs_count_l32_32339

theorem goose_eggs_count (E : ℕ)
  (hatch_ratio : ℚ := 2 / 3)
  (survive_first_month_ratio : ℚ := 3 / 4)
  (survive_first_year_ratio : ℚ := 2 / 5)
  (survived_first_year : ℕ := 130) :
  (survive_first_year_ratio * survive_first_month_ratio * hatch_ratio * (E : ℚ) = survived_first_year) →
  E = 1300 := by
  sorry

end goose_eggs_count_l32_32339


namespace find_angle_A_l32_32448

theorem find_angle_A (a b : ℝ) (B A : ℝ) (h1 : a = Real.sqrt 3) (h2 : b = Real.sqrt 2) 
  (h3 : B = Real.pi / 4) : A = Real.pi / 3 ∨ A = 2 * Real.pi / 3 :=
sorry

end find_angle_A_l32_32448


namespace number_of_girls_in_school_l32_32884

/-- Statement: There are 408 boys and some girls in a school which are to be divided into equal sections
of either boys or girls alone. The total number of sections thus formed is 26. Prove that the number 
of girls is 216. -/
theorem number_of_girls_in_school (n : ℕ) (n_boys : ℕ := 408) (total_sections : ℕ := 26)
  (h1 : n_boys = 408)
  (h2 : ∃ b g : ℕ, b + g = total_sections ∧ 408 / b = n / g ∧ b ∣ 408 ∧ g ∣ n) :
  n = 216 :=
by
  -- Proof would go here
  sorry

end number_of_girls_in_school_l32_32884


namespace relationship_y1_y2_y3_l32_32955

def on_hyperbola (x y k : ℝ) : Prop := y = k / x

theorem relationship_y1_y2_y3 (y1 y2 y3 k : ℝ) (h1 : on_hyperbola (-5) y1 k) (h2 : on_hyperbola (-1) y2 k) (h3 : on_hyperbola 2 y3 k) (hk : k > 0) :
  y2 < y1 ∧ y1 < y3 :=
sorry

end relationship_y1_y2_y3_l32_32955


namespace triangle_existence_condition_l32_32062

theorem triangle_existence_condition 
  (a b f_c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : f_c > 0) : 
  (2 * a * b / (a + b)) > f_c :=
sorry

end triangle_existence_condition_l32_32062


namespace difference_sixth_seventh_l32_32135

theorem difference_sixth_seventh
  (A1 A2 A3 A4 A5 A6 A7 A8 : ℕ)
  (h_avg_8 : (A1 + A2 + A3 + A4 + A5 + A6 + A7 + A8) / 8 = 25)
  (h_avg_2 : (A1 + A2) / 2 = 20)
  (h_avg_3 : (A3 + A4 + A5) / 3 = 26)
  (h_A8 : A8 = 30)
  (h_A6_A8 : A6 = A8 - 6) :
  A7 - A6 = 4 :=
by
  sorry

end difference_sixth_seventh_l32_32135


namespace two_digit_number_solution_l32_32290

theorem two_digit_number_solution (N : ℕ) (x y : ℕ) :
  (10 * x + y = N) ∧ (4 * x + 2 * y = (10 * x + y) / 2) →
  N = 32 ∨ N = 64 ∨ N = 96 := 
sorry

end two_digit_number_solution_l32_32290


namespace two_digit_numbers_solution_l32_32297

theorem two_digit_numbers_solution :
  ∀ (N : ℕ), (∃ (x y : ℕ), (N = 10 * x + y) ∧ (x < 10) ∧ (y < 10) ∧ 4 * x + 2 * y = N / 2) →
    (N = 32 ∨ N = 64 ∨ N = 96) := 
by
  sorry

end two_digit_numbers_solution_l32_32297


namespace lucy_fish_count_l32_32334

theorem lucy_fish_count (initial_fish : ℕ) (additional_fish : ℕ) (final_fish : ℕ) : 
  initial_fish = 212 ∧ additional_fish = 68 → final_fish = 280 :=
by
  sorry

end lucy_fish_count_l32_32334


namespace probability_no_adjacent_birch_trees_l32_32789

open Nat

theorem probability_no_adjacent_birch_trees : 
    let m := 7
    let n := 990
    m + n = 106 := 
by
  sorry

end probability_no_adjacent_birch_trees_l32_32789


namespace mary_animals_count_l32_32114

def initial_lambs := 18
def initial_alpacas := 5
def initial_baby_lambs := 7 * 4
def traded_lambs := 8
def traded_alpacas := 2
def received_goats := 3
def received_chickens := 10
def chickens_traded_for_alpacas := received_chickens / 2
def additional_lambs := 20
def additional_alpacas := 6

noncomputable def final_lambs := initial_lambs + initial_baby_lambs - traded_lambs + additional_lambs
noncomputable def final_alpacas := initial_alpacas - traded_alpacas + 2 + additional_alpacas
noncomputable def final_goats := received_goats
noncomputable def final_chickens := received_chickens - chickens_traded_for_alpacas

theorem mary_animals_count :
  final_lambs = 58 ∧ 
  final_alpacas = 11 ∧ 
  final_goats = 3 ∧ 
  final_chickens = 5 :=
by 
  sorry

end mary_animals_count_l32_32114


namespace count_multiples_of_15_l32_32670

theorem count_multiples_of_15 : ∃ n : ℕ, ∀ k, 12 < k ∧ k < 202 ∧ k % 15 = 0 ↔ k = 15 * n ∧ n = 13 := sorry

end count_multiples_of_15_l32_32670


namespace exist_two_courtiers_with_same_selection_l32_32486

noncomputable theory

-- Definitions
def num_dogs := 100
def num_courtiers := 100
def selected_dogs (courtier : Fin num_courtiers) : Finset (Fin num_dogs) := sorry
def num_selected_dogs := 3 

-- Conditions
axiom three_dogs_selected_each (c : Fin num_courtiers) : selected_dogs c.card = num_selected_dogs

axiom two_common_dogs (c₁ c₂ : Fin num_courtiers) (h : c₁ ≠ c₂) : (selected_dogs c₁ ∩ selected_dogs c₂).card ≥ 2

-- Proof goal
theorem exist_two_courtiers_with_same_selection : 
  ∃ (c₁ c₂ : Fin num_courtiers), c₁ ≠ c₂ ∧ selected_dogs c₁ = selected_dogs c₂ :=
sorry

end exist_two_courtiers_with_same_selection_l32_32486


namespace system_solution_exists_l32_32940

theorem system_solution_exists (a : ℝ) : 
  ∃ (x y : ℝ), (2 * y - 2 = a * (x - 2)) ∧ (4 * y / (|x| + x) = Real.sqrt y) := sorry

end system_solution_exists_l32_32940


namespace inlet_pipe_rate_16_liters_per_minute_l32_32207

noncomputable def rate_of_inlet_pipe : ℝ :=
  let capacity := 21600 -- litres
  let outlet_time_alone := 10 -- hours
  let outlet_time_with_inlet := 18 -- hours
  let outlet_rate := capacity / outlet_time_alone
  let combined_rate := capacity / outlet_time_with_inlet
  let inlet_rate := outlet_rate - combined_rate
  inlet_rate / 60 -- converting litres/hour to litres/min

theorem inlet_pipe_rate_16_liters_per_minute : rate_of_inlet_pipe = 16 :=
by
  sorry

end inlet_pipe_rate_16_liters_per_minute_l32_32207


namespace distance_AB_bounds_l32_32087

noncomputable def distance_AC : ℕ := 10
noncomputable def distance_AD : ℕ := 10
noncomputable def distance_BE : ℕ := 10
noncomputable def distance_BF : ℕ := 10
noncomputable def distance_AE : ℕ := 12
noncomputable def distance_AF : ℕ := 12
noncomputable def distance_BC : ℕ := 12
noncomputable def distance_BD : ℕ := 12
noncomputable def distance_CD : ℕ := 11
noncomputable def distance_EF : ℕ := 11
noncomputable def distance_CE : ℕ := 5
noncomputable def distance_DF : ℕ := 5

theorem distance_AB_bounds (AB : ℝ) :
  8.8 < AB ∧ AB < 19.2 :=
sorry

end distance_AB_bounds_l32_32087


namespace price_per_pie_l32_32059

-- Define the relevant variables and conditions
def cost_pumpkin_pie : ℕ := 3
def num_pumpkin_pies : ℕ := 10
def cost_cherry_pie : ℕ := 5
def num_cherry_pies : ℕ := 12
def desired_profit : ℕ := 20

-- Total production and profit calculation
def total_cost : ℕ := (cost_pumpkin_pie * num_pumpkin_pies) + (cost_cherry_pie * num_cherry_pies)
def total_earnings_needed : ℕ := total_cost + desired_profit
def total_pies : ℕ := num_pumpkin_pies + num_cherry_pies

-- Proposition to prove that the price per pie should be $5
theorem price_per_pie : (total_earnings_needed / total_pies) = 5 := by
  sorry

end price_per_pie_l32_32059


namespace powerThreeExpression_l32_32240

theorem powerThreeExpression (x : ℝ) (h : x - (1 / x) = 5) : x^3 - (1 / (x^3)) = 140 := sorry

end powerThreeExpression_l32_32240


namespace min_unplowed_cells_l32_32851

theorem min_unplowed_cells (n k : ℕ) (hn : n > 0) (hk : k > 0) (hnk : n > k) :
  ∃ M : ℕ, M = (n - k)^2 := by
  sorry

end min_unplowed_cells_l32_32851


namespace find_x_find_y_find_p_q_r_l32_32036

-- Condition: The number on the line connecting two circles is the sum of the two numbers in the circles.

-- For part (a):
theorem find_x (a b : ℝ) (x : ℝ) (h1 : a + 4 = 13) (h2 : a + b = 10) (h3 : b + 4 = x) : x = 5 :=
by {
  -- Proof can be filled in here to show x = 5 by solving the equations.
  sorry
}

-- For part (b):
theorem find_y (w y : ℝ) (h1 : 3 * w + w = y) (h2 : 6 * w = 48) : y = 32 := 
by {
  -- Proof can be filled in here to show y = 32 by solving the equations.
  sorry
}

-- For part (c):
theorem find_p_q_r (p q r : ℝ) (h1 : p + r = 3) (h2 : p + q = 18) (h3 : q + r = 13) : p = 4 ∧ q = 14 ∧ r = -1 :=
by {
  -- Proof can be filled in here to show p = 4, q = 14, r = -1 by solving the equations.
  sorry
}

end find_x_find_y_find_p_q_r_l32_32036


namespace inequality_proof_l32_32989

theorem inequality_proof (a b c : ℝ) (k : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : k ≥ 1) : 
  (a^(k + 1) / b^k + b^(k + 1) / c^k + c^(k + 1) / a^k) ≥ (a^k / b^(k - 1) + b^k / c^(k - 1) + c^k / a^(k - 1)) :=
by
  sorry

end inequality_proof_l32_32989


namespace pyramid_angle_l32_32796

theorem pyramid_angle (k : ℝ) (hk : k > 5) : 
  ∃ α : ℝ, α = Real.arccos (4 / (k - 1)) ∧ -1 ≤ 4 / (k - 1) ∧ 4 / (k - 1) ≤ 1 :=
by 
  -- We set up the requirement that k > 5
  have h_alpha_valid : 4 / (k - 1) ∈ Icc (-1 : ℝ) 1 :=
  begin
    -- We'll derive these assumptions.
    sorry,
  end,
  use Real.arccos (4 / (k - 1)),
  refine ⟨rfl, h_alpha_valid.left, h_alpha_valid.right⟩

end pyramid_angle_l32_32796


namespace manager_salary_correct_l32_32612

-- Define the conditions of the problem
def total_salary_of_24_employees : ℕ := 24 * 2400
def new_average_salary_with_manager : ℕ := 2500
def number_of_people_with_manager : ℕ := 25

-- Define the manager's salary to be proved
def managers_salary : ℕ := 4900

-- Statement of the theorem to prove that the manager's salary is Rs. 4900
theorem manager_salary_correct :
  (number_of_people_with_manager * new_average_salary_with_manager) - total_salary_of_24_employees = managers_salary :=
by
  -- Proof to be filled
  sorry

end manager_salary_correct_l32_32612


namespace geometric_sequence_304th_term_l32_32310

theorem geometric_sequence_304th_term (a r : ℤ) (n : ℕ) (h_a : a = 8) (h_ar : a * r = -8) (h_n : n = 304) :
  ∃ t : ℤ, t = -8 :=
by
  sorry

end geometric_sequence_304th_term_l32_32310


namespace decreasing_line_implies_m_half_l32_32138

theorem decreasing_line_implies_m_half (m b : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → (2 * m - 1) * x₁ + b > (2 * m - 1) * x₂ + b) → m < 1 / 2 :=
by
  intro h
  sorry

end decreasing_line_implies_m_half_l32_32138


namespace each_person_has_5_bags_l32_32885

def people := 6
def weight_per_bag := 50
def max_plane_weight := 6000
def additional_capacity := 90

theorem each_person_has_5_bags :
  (max_plane_weight / weight_per_bag - additional_capacity) / people = 5 :=
by
  sorry

end each_person_has_5_bags_l32_32885


namespace min_value_expression_l32_32327

open Real

theorem min_value_expression (α β : ℝ) :
  (3 * cos α + 4 * sin β - 7)^2 + (3 * sin α + 4 * cos β - 12)^2 ≥ 36 := by
  sorry

end min_value_expression_l32_32327


namespace swimming_pool_paint_area_l32_32136

theorem swimming_pool_paint_area :
  let length := 20 -- The pool is 20 meters long
  let width := 12  -- The pool is 12 meters wide
  let depth := 2   -- The pool is 2 meters deep
  let area_longer_walls := 2 * length * depth
  let area_shorter_walls := 2 * width * depth
  let total_side_wall_area := area_longer_walls + area_shorter_walls
  let floor_area := length * width
  let total_area_to_paint := total_side_wall_area + floor_area
  total_area_to_paint = 368 :=
by
  sorry

end swimming_pool_paint_area_l32_32136


namespace least_distance_fly_crawled_l32_32049

noncomputable def leastDistance (baseRadius height startDist endDist : ℝ) : ℝ :=
  let C := 2 * Real.pi * baseRadius
  let slantHeight := Real.sqrt (baseRadius ^ 2 + height ^ 2)
  let theta := C / slantHeight
  let x1 := startDist * Real.cos 0
  let y1 := startDist * Real.sin 0
  let x2 := endDist * Real.cos (theta / 2)
  let y2 := endDist * Real.sin (theta / 2)
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem least_distance_fly_crawled (baseRadius height startDist endDist : ℝ) (h1 : baseRadius = 500) (h2 : height = 150 * Real.sqrt 7) (h3 : startDist = 150) (h4 : endDist = 300 * Real.sqrt 2) :
  leastDistance baseRadius height startDist endDist = 150 * Real.sqrt 13 := by
  sorry

end least_distance_fly_crawled_l32_32049


namespace divisor_and_remainder_correct_l32_32414

theorem divisor_and_remainder_correct:
  ∃ d r : ℕ, d ≠ 0 ∧ 1270 = 74 * d + r ∧ r = 12 ∧ d = 17 :=
by
  sorry

end divisor_and_remainder_correct_l32_32414


namespace sufficient_but_not_necessary_l32_32079

-- Define what it means for a line to be perpendicular to a plane
def line_perpendicular_to_plane (l : Type) (alpha : Type) : Prop := 
  sorry -- Definition not provided

-- Define what it means for a line to be perpendicular to countless lines in a plane
def line_perpendicular_to_countless_lines_in_plane (l : Type) (alpha : Type) : Prop := 
  sorry -- Definition not provided

-- The formal statement
theorem sufficient_but_not_necessary (l : Type) (alpha : Type) :
  (line_perpendicular_to_plane l alpha) → (line_perpendicular_to_countless_lines_in_plane l alpha) ∧ 
  ¬ ((line_perpendicular_to_countless_lines_in_plane l alpha) → (line_perpendicular_to_plane l alpha)) :=
by sorry

end sufficient_but_not_necessary_l32_32079


namespace two_digit_numbers_solution_l32_32298

theorem two_digit_numbers_solution :
  ∀ (N : ℕ), (∃ (x y : ℕ), (N = 10 * x + y) ∧ (x < 10) ∧ (y < 10) ∧ 4 * x + 2 * y = N / 2) →
    (N = 32 ∨ N = 64 ∨ N = 96) := 
by
  sorry

end two_digit_numbers_solution_l32_32298


namespace systematic_sampling_method_l32_32532

theorem systematic_sampling_method :
  ∀ (num_classes num_students_per_class selected_student : ℕ),
    num_classes = 12 →
    num_students_per_class = 50 →
    selected_student = 40 →
    (∃ (start_interval: ℕ) (interval: ℕ) (total_population: ℕ), 
      total_population > 100 ∧ start_interval < interval ∧ interval * num_classes = total_population ∧
      ∀ (c : ℕ), c < num_classes → (start_interval + c * interval) % num_students_per_class = selected_student - 1) →
    "Systematic Sampling" = "Systematic Sampling" :=
by
  intros num_classes num_students_per_class selected_student h_classes h_students h_selected h_conditions
  sorry

end systematic_sampling_method_l32_32532


namespace general_term_correct_l32_32010

-- Define the sequence a_n
def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = 2 * a n + 2^n

-- Define the general term formula for the sequence a_n
def general_term (a : ℕ → ℕ) : Prop :=
  ∀ n, a n = n * 2^(n - 1)

-- Theorem statement: the general term formula holds for the sequence a_n
theorem general_term_correct (a : ℕ → ℕ) (h_seq : seq a) : general_term a :=
by
  sorry

end general_term_correct_l32_32010


namespace jack_marbles_l32_32313

theorem jack_marbles (initial_marbles share_marbles : ℕ) (h_initial : initial_marbles = 62) (h_share : share_marbles = 33) : 
  initial_marbles - share_marbles = 29 :=
by 
  sorry

end jack_marbles_l32_32313


namespace fibonacci_money_problem_l32_32309

variable (x : ℕ)

theorem fibonacci_money_problem (h : 0 < x - 6) (eq_amounts : 90 / (x - 6) = 120 / x) : 
    90 / (x - 6) = 120 / x :=
sorry

end fibonacci_money_problem_l32_32309


namespace mehki_age_l32_32991

theorem mehki_age (Z J M : ℕ) (h1 : Z = 6) (h2 : J = Z - 4) (h3 : M = 2 * (J + Z)) : M = 16 := by
  sorry

end mehki_age_l32_32991


namespace diagonal_length_of_rhombus_l32_32000

-- Definitions for the conditions
def side_length_of_square : ℝ := 8
def area_of_square : ℝ := side_length_of_square ^ 2
def area_of_rhombus : ℝ := 64
def d2 : ℝ := 8
-- Question
theorem diagonal_length_of_rhombus (d1 : ℝ) : (d1 * d2) / 2 = area_of_rhombus ↔ d1 = 16 := by
  sorry

end diagonal_length_of_rhombus_l32_32000


namespace largest_divisor_of_expression_l32_32420

theorem largest_divisor_of_expression (n : ℤ) : 6 ∣ (n^4 - n) := 
sorry

end largest_divisor_of_expression_l32_32420


namespace fraction_problem_l32_32557

theorem fraction_problem :
  (3 / 7 + 5 / 8) / (5 / 12 + 2 / 9) = 531 / 322 :=
by sorry

end fraction_problem_l32_32557


namespace half_way_fraction_l32_32521

def half_way_between (a b : ℚ) : ℚ := (a + b) / 2

theorem half_way_fraction : 
  half_way_between (1/3) (3/4) = 13/24 :=
by 
  -- Proof follows from the calculation steps, but we leave it unproved.
  sorry

end half_way_fraction_l32_32521


namespace sum_of_cubes_of_roots_eq_1_l32_32905

theorem sum_of_cubes_of_roots_eq_1 (a : ℝ) (x1 x2 : ℝ) :
  (x1^2 + a * x1 + a + 1 = 0) → 
  (x2^2 + a * x2 + a + 1 = 0) → 
  (x1 + x2 = -a) → 
  (x1 * x2 = a + 1) → 
  (x1^3 + x2^3 = 1) → 
  a = -1 :=
sorry

end sum_of_cubes_of_roots_eq_1_l32_32905


namespace quadratic_y_axis_intersection_l32_32741

theorem quadratic_y_axis_intersection :
  (∃ y, (y = (0 - 1) ^ 2 + 2) ∧ (0, y) = (0, 3)) :=
sorry

end quadratic_y_axis_intersection_l32_32741


namespace answer_to_rarely_infrequently_word_l32_32538

-- Declare variables and definitions based on given conditions
-- In this context, we'll introduce a basic definition for the word "seldom".

noncomputable def is_word_meaning_rarely (w : String) : Prop :=
  w = "seldom"

-- Now state the problem in the form of a Lean theorem
theorem answer_to_rarely_infrequently_word : ∃ w, is_word_meaning_rarely w :=
by
  use "seldom"
  unfold is_word_meaning_rarely
  rfl

end answer_to_rarely_infrequently_word_l32_32538


namespace two_digit_number_solution_l32_32289

theorem two_digit_number_solution (N : ℕ) (x y : ℕ) :
  (10 * x + y = N) ∧ (4 * x + 2 * y = (10 * x + y) / 2) →
  N = 32 ∨ N = 64 ∨ N = 96 := 
sorry

end two_digit_number_solution_l32_32289


namespace martha_total_payment_l32_32151

noncomputable def cheese_kg : ℝ := 1.5
noncomputable def meat_kg : ℝ := 0.55
noncomputable def pasta_kg : ℝ := 0.28
noncomputable def tomatoes_kg : ℝ := 2.2

noncomputable def cheese_price_per_kg : ℝ := 6.30
noncomputable def meat_price_per_kg : ℝ := 8.55
noncomputable def pasta_price_per_kg : ℝ := 2.40
noncomputable def tomatoes_price_per_kg : ℝ := 1.79

noncomputable def total_cost :=
  cheese_kg * cheese_price_per_kg +
  meat_kg * meat_price_per_kg +
  pasta_kg * pasta_price_per_kg +
  tomatoes_kg * tomatoes_price_per_kg

theorem martha_total_payment : total_cost = 18.76 := by
  sorry

end martha_total_payment_l32_32151


namespace sophist_statements_correct_l32_32123

-- Definitions based on conditions
def num_knights : ℕ := 40
def num_liars : ℕ := 25

-- Statements made by the sophist
def sophist_statement1 : Prop := ∃ (sophist : Prop), sophist ∧ sophist → num_knights = 40
def sophist_statement2 : Prop := ∃ (sophist : Prop), sophist ∧ sophist → num_liars + 1 = 26

-- Theorem to be proved
theorem sophist_statements_correct :
  sophist_statement1 ∧ sophist_statement2 :=
by
  -- Placeholder for the actual proof
  sorry

end sophist_statements_correct_l32_32123


namespace words_per_page_l32_32177

theorem words_per_page (p : ℕ) :
  (136 * p) % 203 = 184 % 203 ∧ p ≤ 100 → p = 73 :=
sorry

end words_per_page_l32_32177


namespace nancy_ate_3_apples_l32_32469

theorem nancy_ate_3_apples
  (mike_apples : ℝ)
  (keith_apples : ℝ)
  (apples_left : ℝ)
  (mike_apples_eq : mike_apples = 7.0)
  (keith_apples_eq : keith_apples = 6.0)
  (apples_left_eq : apples_left = 10.0) :
  mike_apples + keith_apples - apples_left = 3.0 := 
by
  rw [mike_apples_eq, keith_apples_eq, apples_left_eq]
  norm_num

end nancy_ate_3_apples_l32_32469


namespace find_b_l32_32681

-- Definitions for the conditions
variables (a b c d : ℝ)
def four_segments_proportional := a / b = c / d

theorem find_b (h1: a = 3) (h2: d = 4) (h3: c = 6) (h4: four_segments_proportional a b c d) : b = 2 :=
by
  sorry

end find_b_l32_32681


namespace part1_l32_32573

def U : Set ℝ := Set.univ
def P (a : ℝ) : Set ℝ := {x | 4 ≤ x ∧ x ≤ 7}
def Q : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

theorem part1 (a : ℝ) (P_def : P 3 = {x | 4 ≤ x ∧ x ≤ 7}) :
  ((U \ P a) ∩ Q = {x | -2 ≤ x ∧ x < 4}) := by
  sorry

end part1_l32_32573


namespace courtier_selection_l32_32490

theorem courtier_selection :
  ∀ (dogs : Finset ℕ) (courtiers : ℕ → Finset ℕ),
  dogs.card = 100 ∧ (∀ i, card (courtiers i) = 3) ∧ (∀ i j, i ≠ j → (courtiers i ∩ courtiers j).card = 2) →
  ∃ i j, i ≠ j ∧ courtiers i = courtiers j :=
by
  intros dogs courtiers h
  sorry

end courtier_selection_l32_32490


namespace problem_statement_l32_32317

variable {x : ℝ}
noncomputable def A : ℝ := 39
noncomputable def B : ℝ := -5

theorem problem_statement (h : ∀ x ≠ 3, (A / (x - 3) + B * (x + 2)) = (-5 * x ^ 2 + 18 * x + 30) / (x - 3)) : A + B = 34 := 
sorry

end problem_statement_l32_32317


namespace shopkeeper_intended_profit_l32_32917

noncomputable def intended_profit_percentage (C L S : ℝ) : ℝ :=
  (L / C) - 1

theorem shopkeeper_intended_profit (C L S : ℝ) (h1 : L = C * (1 + intended_profit_percentage C L S))
  (h2 : S = 0.90 * L) (h3 : S = 1.35 * C) : intended_profit_percentage C L S = 0.5 :=
by
  -- We indicate that the proof is skipped
  sorry

end shopkeeper_intended_profit_l32_32917


namespace cube_volume_l32_32186

theorem cube_volume (A : ℝ) (hA : A = 294) : ∃ (V : ℝ), V = 343 :=
by
  sorry

end cube_volume_l32_32186


namespace volume_of_cube_with_surface_area_l32_32192

theorem volume_of_cube_with_surface_area (S : ℝ) (hS : S = 294) : 
  ∃ V : ℝ, V = 343 :=
by
  let s := (S / 6).sqrt
  have hs : s = 7 := by sorry
  use s ^ 3
  simp [hs]
  exact sorry

end volume_of_cube_with_surface_area_l32_32192


namespace fruit_weights_l32_32120

theorem fruit_weights:
  ∃ (mandarin apple banana peach orange : ℕ),
    {mandarin, apple, banana, peach, orange} = {100, 150, 170, 200, 280} ∧
    peach < orange ∧
    apple < banana ∧ banana < peach ∧
    mandarin < banana ∧
    apple + banana > orange :=
begin
  -- Assign weights corresponding to each fruit
  use [100, 150, 170, 200, 280],
  split,
  {
    -- Check if all weights are used
    dsimp,
    exact rfl,
  },
  split,
  {
    -- peach < orange
    exact nat.lt_succ_self _,
  },
  split,
  {
    -- apple < banana < peach
    split,
    { exact nat.lt_succ_self _ },
    { exact nat.lt_succ_self _ }
  },
  split,
  {
    -- mandarin < banana
    exact nat.lt_succ_self _,
  },
  {
    -- apple + banana > orange
    exact nat.lt_succ_self _,
  }
end

end fruit_weights_l32_32120


namespace arithmetic_sequence__geometric_sequence__l32_32176

-- Part 1: Arithmetic Sequence
theorem arithmetic_sequence_
  (d : ℤ) (n : ℤ) (a_n : ℤ) (a_1 : ℤ) (S_n : ℤ)
  (h_d : d = 2) (h_n : n = 15) (h_a_n : a_n = -10)
  (h_a_1 : a_1 = -38) (h_S_n : S_n = -360) :
  a_n = a_1 + (n - 1) * d ∧ S_n = n * (a_1 + a_n) / 2 :=
by
  sorry

-- Part 2: Geometric Sequence
theorem geometric_sequence_
  (a_1 : ℝ) (q : ℝ) (S_10 : ℝ)
  (a_2 : ℝ) (a_3 : ℝ) (a_4 : ℝ)
  (h_a_2_3 : a_2 + a_3 = 6) (h_a_3_4 : a_3 + a_4 = 12)
  (h_a_1 : a_1 = 1) (h_q : q = 2) (h_S_10 : S_10 = 1023) :
  a_2 = a_1 * q ∧ a_3 = a_1 * q^2 ∧ a_4 = a_1 * q^3 ∧ S_10 = a_1 * (1 - q^10) / (1 - q) :=
by
  sorry

end arithmetic_sequence__geometric_sequence__l32_32176


namespace remainder_when_divided_by_x_minus_2_l32_32072

def polynomial (x : ℝ) := x^5 + 2 * x^3 - x + 4

theorem remainder_when_divided_by_x_minus_2 :
  polynomial 2 = 50 :=
by
  sorry

end remainder_when_divided_by_x_minus_2_l32_32072


namespace find_ab_exponent_l32_32427

theorem find_ab_exponent (a b : ℝ) 
  (h : |a - 2| + (b + 1 / 2)^2 = 0) : 
  a^2022 * b^2023 = -1 / 2 := 
sorry

end find_ab_exponent_l32_32427


namespace domain_of_function_l32_32480

theorem domain_of_function :
  {x : ℝ | 2 - x ≥ 0} = {x : ℝ | x ≤ 2} :=
by
  sorry

end domain_of_function_l32_32480


namespace min_value_fraction_l32_32578

theorem min_value_fraction (a b c : ℝ) (h1 : a > 0) (h2 : a < (2 / 3) * b) (h3 : c ≥ b^2 / (3 * a)) : 
  ∃ x : ℝ, (∀ y : ℝ, y ≥ x → y ≥ 1) ∧ (x = 1) :=
by
  sorry

end min_value_fraction_l32_32578


namespace number_of_friends_l32_32211

def total_envelopes : ℕ := 37
def envelopes_per_friend : ℕ := 3
def envelopes_left : ℕ := 22

theorem number_of_friends :
  ((total_envelopes - envelopes_left) / envelopes_per_friend) = 5 := by
  sorry

end number_of_friends_l32_32211


namespace percentage_tax_proof_l32_32454

theorem percentage_tax_proof (total_worth tax_free cost taxable tax_rate tax_value percentage_sales_tax : ℝ)
  (h1 : total_worth = 40)
  (h2 : tax_free = 34.7)
  (h3 : tax_rate = 0.06)
  (h4 : total_worth = taxable + tax_rate * taxable + tax_free)
  (h5 : tax_value = tax_rate * taxable)
  (h6 : percentage_sales_tax = (tax_value / total_worth) * 100) :
  percentage_sales_tax = 0.75 :=
by
  sorry

end percentage_tax_proof_l32_32454


namespace starting_number_of_range_divisible_by_11_l32_32757

theorem starting_number_of_range_divisible_by_11 (a : ℕ) : 
  a ≤ 79 ∧ (a + 22 = 77) ∧ ((a + 11) + 11 = 77) → a = 55 := 
by
  sorry

end starting_number_of_range_divisible_by_11_l32_32757


namespace cost_per_pound_mixed_feed_correct_l32_32358

noncomputable def total_weight_of_feed : ℝ := 17
noncomputable def cost_per_pound_cheaper_feed : ℝ := 0.11
noncomputable def cost_per_pound_expensive_feed : ℝ := 0.50
noncomputable def weight_cheaper_feed : ℝ := 12.2051282051

noncomputable def total_cost_of_feed : ℝ :=
  (cost_per_pound_cheaper_feed * weight_cheaper_feed) + 
  (cost_per_pound_expensive_feed * (total_weight_of_feed - weight_cheaper_feed))

noncomputable def cost_per_pound_mixed_feed : ℝ :=
  total_cost_of_feed / total_weight_of_feed

theorem cost_per_pound_mixed_feed_correct : 
  cost_per_pound_mixed_feed = 0.22 :=
  by
    sorry

end cost_per_pound_mixed_feed_correct_l32_32358


namespace two_courtiers_have_same_selection_l32_32492

-- Definitions based on the given conditions
def dogs : Type := Fin 100
def courtiers : Type := Fin 100
def select_three_dogs (c : courtiers) : set dogs := sorry

-- Main theorem statement
theorem two_courtiers_have_same_selection :
  (∀ (c1 c2 : courtiers), c1 ≠ c2 → (select_three_dogs c1 ∩ select_three_dogs c2).card = 2) →
  ∃ (c1 c2 : courtiers), c1 ≠ c2 ∧ select_three_dogs c1 = select_three_dogs c2 :=
sorry

end two_courtiers_have_same_selection_l32_32492


namespace V_not_measurable_l32_32407

open Set
open Classical

noncomputable theory

def equivalence_on_unit_interval (x y : ℝ) : Prop :=
  ∃ q : ℚ, x - y = (q : ℝ)

axiom axiom_of_choice (S : Set (Set α)) : ∃ (f : (Set α → α)), ∀ (B : Set α), B ∈ S → f B ∈ B

noncomputable def select_one_per_class :
  {V : Set ℝ // V ⊆ Icc 0 1 ∧ ∀ (x ∈ V) (y ∈ V), equivalence_on_unit_interval x y → x = y} :=
by
  choose v hv using axiom_of_choice (equivalence_on_unit_interval '' (Icc 0 1))
  use {x | ∃ y ∈ Icc 0 1, v (equivalence_on_unit_interval y) = x}
  split
  · intros x hx
    rcases hx with ⟨y, hy1, hy2⟩
    exact hy1
  · intros x hx y hy hxy
    rcases hx with ⟨x', hx', rfl⟩
    rcases hy with ⟨y', hy', rfl⟩
    by_contra h
    exact h (hv (equivalence_on_unit_interval y x') hxy)

theorem V_not_measurable : ¬ (measurable_set select_one_per_class.val) :=
sorry

end V_not_measurable_l32_32407


namespace max_buses_in_city_l32_32095

theorem max_buses_in_city (num_stops stops_per_bus shared_stops : ℕ) (h_stops : num_stops = 9) (h_stops_per_bus : stops_per_bus = 3) (h_shared_stops : shared_stops = 1) : 
  ∃ (max_buses : ℕ), max_buses = 12 :=
by
  sorry

end max_buses_in_city_l32_32095


namespace hunting_dogs_theorem_l32_32499

noncomputable def hunting_dogs_problem : Prop :=
  ∃ (courtiers : Finset (Finset (Fin 100))) (h1 : courtiers.card = 100),
  ∀ (c1 c2 : Finset (Fin 100)), c1 ∈ courtiers → c2 ∈ courtiers → c1 ≠ c2 → (c1 ∩ c2).card ≥ 2 → 
  ∃ (c₁ c₂ : Finset (Fin 100)), c₁ ∈ courtiers ∧ c₂ ∈ courtiers ∧ c₁ = c₂

theorem hunting_dogs_theorem : hunting_dogs_problem :=
sorry

end hunting_dogs_theorem_l32_32499


namespace gcd_of_items_l32_32406

theorem gcd_of_items :
  ∀ (plates spoons glasses bowls : ℕ),
  plates = 3219 →
  spoons = 5641 →
  glasses = 1509 →
  bowls = 2387 →
  Nat.gcd (Nat.gcd (Nat.gcd plates spoons) glasses) bowls = 1 :=
by
  intros plates spoons glasses bowls
  intros Hplates Hspoons Hglasses Hbowls
  rw [Hplates, Hspoons, Hglasses, Hbowls]
  sorry

end gcd_of_items_l32_32406


namespace total_blocks_fallen_l32_32590

def stack_height (n : Nat) : Nat :=
  if n = 1 then 7
  else if n = 2 then 7 + 5
  else if n = 3 then 7 + 5 + 7
  else 0

def blocks_standing (n : Nat) : Nat :=
  if n = 1 then 0
  else if n = 2 then 2
  else if n = 3 then 3
  else 0

def blocks_fallen (n : Nat) : Nat :=
  stack_height n - blocks_standing n

theorem total_blocks_fallen : blocks_fallen 1 + blocks_fallen 2 + blocks_fallen 3 = 33 :=
  by
    sorry

end total_blocks_fallen_l32_32590


namespace complement_union_example_l32_32038

open Set

universe u

variable (U : Set ℕ) (A B : Set ℕ)

def U_def : Set ℕ := {0, 1, 2, 3, 4}
def A_def : Set ℕ := {0, 1, 2}
def B_def : Set ℕ := {2, 3}

theorem complement_union_example :
  (U \ A) ∪ B = {2, 3, 4} := 
by
  -- Proving the theorem considering
  -- complement and union operations on sets
  sorry

end complement_union_example_l32_32038


namespace donation_fifth_sixth_l32_32478

-- Conditions definitions
def total_donation := 10000
def first_home := 2750
def second_home := 1945
def third_home := 1275
def fourth_home := 1890

-- Proof statement
theorem donation_fifth_sixth : 
  (total_donation - (first_home + second_home + third_home + fourth_home)) = 2140 := by
  sorry

end donation_fifth_sixth_l32_32478


namespace lines_intersect_at_single_point_l32_32067

theorem lines_intersect_at_single_point (m : ℚ)
    (h1 : ∃ x y : ℚ, y = 4 * x - 8 ∧ y = -3 * x + 9)
    (h2 : ∀ x y : ℚ, (y = 4 * x - 8 ∧ y = -3 * x + 9) → (y = 2 * x + m)) :
    m = -22/7 := by
  sorry

end lines_intersect_at_single_point_l32_32067


namespace product_value_l32_32555

theorem product_value (x : ℝ) (h : (Real.sqrt (6 + x) + Real.sqrt (21 - x) = 8)) : (6 + x) * (21 - x) = 1369 / 4 :=
by
  sorry

end product_value_l32_32555


namespace problem1_l32_32040

theorem problem1 (a b : ℝ) (i : ℝ) (h : (a-2*i)*i = b-i) : a^2 + b^2 = 5 := by
  sorry

end problem1_l32_32040


namespace part1_real_roots_part2_integer_roots_l32_32691

-- Equation structure
variable (k : ℝ) (k_ne_zero : k ≠ 0)

-- Define the quadratic equation kx^2 + (k-2)x - 2 = 0
noncomputable def quadratic_eq (x : ℝ) : ℝ :=
  k * x^2 + (k - 2) * x - 2

-- Part (1): Proving the equation always has real roots
theorem part1_real_roots : ∃ x₁ x₂ : ℝ, quadratic_eq k_ne_zero x₁ = 0 ∧ quadratic_eq k_ne_zero x₂ = 0 :=
by sorry

-- Part (2): Finding values of k for which the equation has two distinct integer roots
theorem part2_integer_roots (k_int : k ∈ Int) : k = 1 ∨ k = -1 ∨ k = 2 :=
by sorry

end part1_real_roots_part2_integer_roots_l32_32691


namespace set_equality_l32_32444

theorem set_equality (A : Set ℕ) (h : {1} ∪ A = {1, 3, 5}) : 
  A = {1, 3, 5} ∨ A = {3, 5} :=
  sorry

end set_equality_l32_32444


namespace range_of_a_l32_32430

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x > 3 → x > a)) ↔ (a ≤ 3) :=
sorry

end range_of_a_l32_32430


namespace count_coin_distributions_l32_32886

-- Mathematical conditions
def coin_denominations : Finset ℕ := {1, 2, 3, 5}
def number_of_boys : ℕ := 6

-- Theorem statement
theorem count_coin_distributions : (coin_denominations.card ^ number_of_boys) = 4096 :=
by
  sorry

end count_coin_distributions_l32_32886


namespace length_of_platform_l32_32922

theorem length_of_platform (length_of_train speed_of_train time_to_cross : ℕ) 
    (h1 : length_of_train = 450) (h2 : speed_of_train = 126) (h3 : time_to_cross = 20) :
    ∃ length_of_platform : ℕ, length_of_platform = 250 := 
by 
  sorry

end length_of_platform_l32_32922


namespace candy_initial_count_l32_32453

theorem candy_initial_count (candy_given_first candy_given_second candy_given_third candy_bought candy_eaten candy_left initial_candy : ℕ) 
    (h1 : candy_given_first = 18) 
    (h2 : candy_given_second = 12)
    (h3 : candy_given_third = 25)
    (h4 : candy_bought = 10)
    (h5 : candy_eaten = 7)
    (h6 : candy_left = 16)
    (h_initial : candy_left + candy_eaten = initial_candy - candy_bought - candy_given_first - candy_given_second - candy_given_third):
    initial_candy = 68 := 
by 
  sorry

end candy_initial_count_l32_32453


namespace intersecting_to_quadrilateral_l32_32769

-- Define the geometric solids
inductive GeometricSolid
| cone : GeometricSolid
| sphere : GeometricSolid
| cylinder : GeometricSolid

-- Define a function that checks if intersecting a given solid with a plane can produce a quadrilateral
def can_intersect_to_quadrilateral (solid : GeometricSolid) : Prop :=
  match solid with
  | GeometricSolid.cone => false
  | GeometricSolid.sphere => false
  | GeometricSolid.cylinder => true

-- State the theorem
theorem intersecting_to_quadrilateral (solid : GeometricSolid) :
  can_intersect_to_quadrilateral solid ↔ solid = GeometricSolid.cylinder :=
sorry

end intersecting_to_quadrilateral_l32_32769


namespace faster_train_speed_correct_l32_32624

noncomputable def speed_of_faster_train (V_s_kmph : ℝ) (length_faster_train_m : ℝ) (time_s : ℝ) : ℝ :=
  let V_s_mps := V_s_kmph * (1000 / 3600)
  let V_r_mps := length_faster_train_m / time_s
  let V_f_mps := V_r_mps - V_s_mps
  V_f_mps * (3600 / 1000)

theorem faster_train_speed_correct : 
  speed_of_faster_train 36 90.0072 4 = 45.00648 := 
by
  sorry

end faster_train_speed_correct_l32_32624


namespace toms_dog_age_is_twelve_l32_32761

-- Definitions based on given conditions
def toms_cat_age : ℕ := 8
def toms_rabbit_age : ℕ := toms_cat_age / 2
def toms_dog_age : ℕ := 3 * toms_rabbit_age

-- The statement to be proved
theorem toms_dog_age_is_twelve : toms_dog_age = 12 := by
  sorry

end toms_dog_age_is_twelve_l32_32761


namespace classroom_problem_l32_32034

noncomputable def classroom_problem_statement : Prop :=
  ∀ (B G : ℕ) (b g : ℝ),
    b > 0 →
    g > 0 →
    B > 0 →
    G > 0 →
    ¬ ((B * g + G * b) / (B + G) = b + g ∧ b > 0 ∧ g > 0)

theorem classroom_problem : classroom_problem_statement :=
  by
    intros B G b g hb_gt0 hg_gt0 hB_gt0 hG_gt0
    sorry

end classroom_problem_l32_32034


namespace center_of_circle_in_second_quadrant_l32_32481

theorem center_of_circle_in_second_quadrant (a : ℝ) (h : a > 12) :
  ∃ x y : ℝ, x^2 + y^2 + a * x - 2 * a * y + a^2 + 3 * a = 0 ∧ (-a / 2, a).2 > 0 ∧ (-a / 2, a).1 < 0 :=
by
  sorry

end center_of_circle_in_second_quadrant_l32_32481


namespace remainder_mod_5_is_0_l32_32929

theorem remainder_mod_5_is_0 :
  (88144 * 88145 + 88146 + 88147 + 88148 + 88149 + 88150) % 5 = 0 := by
  sorry

end remainder_mod_5_is_0_l32_32929


namespace cube_volume_from_surface_area_l32_32188

theorem cube_volume_from_surface_area (A : ℝ) (hA : A = 294) : ∃ V : ℝ, V = 343 :=
by
  let s := real.sqrt (A / 6)
  have h_s : s = 7 := sorry
  let V := s^3
  have h_V : V = 343 := sorry
  exact ⟨V, h_V⟩

end cube_volume_from_surface_area_l32_32188


namespace necessary_condition_l32_32534

theorem necessary_condition (m : ℝ) (h : ∀ x : ℝ, x^2 - x + m > 0) : m > 0 := 
sorry

end necessary_condition_l32_32534


namespace max_buses_l32_32104

theorem max_buses (bus_stops : ℕ) (each_bus_stops : ℕ) (no_shared_stops : ∀ b₁ b₂ : Finset ℕ, b₁ ≠ b₂ → (bus_stops \ (b₁ ∪ b₂)).card ≤ bus_stops - 6) : 
  bus_stops = 9 ∧ each_bus_stops = 3 → ∃ max_buses, max_buses = 12 :=
by
  sorry

end max_buses_l32_32104


namespace problem1_problem2_l32_32462

noncomputable def interval1 (a : ℝ) : Set ℝ := {x | 2 * a < x ∧ x < a + 1}
noncomputable def interval2 : Set ℝ := {x | x < -1 ∨ x > 3}

theorem problem1 (a : ℝ) : (interval1 a ∩ interval2 = interval1 a) ↔ a ∈ {x | x ≤ -2} ∪ {x | 1 ≤ x} := by sorry

theorem problem2 (a : ℝ) : (interval1 a ∩ interval2 ≠ ∅) ↔ a < -1 / 2 := by sorry

end problem1_problem2_l32_32462


namespace inequality_proof_l32_32569

theorem inequality_proof (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h_sum : x + y + z = 1) :
  ((1 / x^2 - x) * (1 / y^2 - y) * (1 / z^2 - z) ≥ (26 / 3)^3) :=
by sorry

end inequality_proof_l32_32569


namespace price_of_toy_organizers_is_78_l32_32455

variable (P : ℝ) -- Price per set of toy organizers

-- Conditions
def total_cost_of_toy_organizers (P : ℝ) : ℝ := 3 * P
def total_cost_of_gaming_chairs : ℝ := 2 * 83
def total_sales (P : ℝ) : ℝ := total_cost_of_toy_organizers P + total_cost_of_gaming_chairs
def delivery_fee (P : ℝ) : ℝ := 0.05 * total_sales P
def total_amount_paid (P : ℝ) : ℝ := total_sales P + delivery_fee P

-- Proof statement
theorem price_of_toy_organizers_is_78 (h : total_amount_paid P = 420) : P = 78 :=
by
  sorry

end price_of_toy_organizers_is_78_l32_32455


namespace sum_products_roots_l32_32856

theorem sum_products_roots :
  (∃ p q r : ℂ, (3 * p^3 - 5 * p^2 + 12 * p - 10 = 0) ∧
                  (3 * q^3 - 5 * q^2 + 12 * q - 10 = 0) ∧
                  (3 * r^3 - 5 * r^2 + 12 * r - 10 = 0) ∧
                  (p ≠ q) ∧ (q ≠ r) ∧ (p ≠ r)) →
  ∀ p q r : ℂ, (3 * p) * (q * r) + (3 * q) * (r * p) + (3 * r) * (p * q) =
    (3 * p * q * r) :=
sorry

end sum_products_roots_l32_32856


namespace cube_difference_l32_32244

variable (x : ℝ)

theorem cube_difference (h : x - 1/x = 5) : x^3 - 1/x^3 = 120 := by
  sorry

end cube_difference_l32_32244


namespace part_a_roots_part_b_sum_l32_32039

theorem part_a_roots : ∀ x : ℝ, 2^x = x + 1 ↔ x = 0 ∨ x = 1 :=
by 
  intros x
  sorry

theorem part_b_sum (f : ℝ → ℝ) (h : ∀ x : ℝ, (f ∘ f) x = 2^x - 1) : f 0 + f 1 = 1 :=
by 
  sorry

end part_a_roots_part_b_sum_l32_32039


namespace intersecting_lines_l32_32007

theorem intersecting_lines (a b : ℝ) (h1 : 1 = 1 / 4 * 2 + a) (h2 : 2 = 1 / 4 * 1 + b) : 
  a + b = 9 / 4 := 
sorry

end intersecting_lines_l32_32007


namespace two_digit_number_solution_l32_32288

theorem two_digit_number_solution (N : ℕ) (x y : ℕ) :
  (10 * x + y = N) ∧ (4 * x + 2 * y = (10 * x + y) / 2) →
  N = 32 ∨ N = 64 ∨ N = 96 := 
sorry

end two_digit_number_solution_l32_32288


namespace difference_of_profit_share_l32_32168

theorem difference_of_profit_share (a b c : ℕ) (pa pb pc : ℕ) (profit_b : ℕ) 
  (a_capital : a = 8000) (b_capital : b = 10000) (c_capital : c = 12000) 
  (b_profit_share : profit_b = 1600)
  (investment_ratio : pa / 4 = pb / 5 ∧ pb / 5 = pc / 6) :
  pa - pc = 640 := 
sorry

end difference_of_profit_share_l32_32168


namespace roots_of_polynomial_l32_32680

theorem roots_of_polynomial :
  {x | x * (2 * x - 5) ^ 2 * (x + 3) * (7 - x) = 0} = {0, 2.5, -3, 7} :=
by {
  sorry
}

end roots_of_polynomial_l32_32680


namespace yan_ratio_distance_l32_32901

theorem yan_ratio_distance (w x y : ℕ) (h : w > 0) (h_eq_time : (y / w) = (x / w) + ((x + y) / (6 * w))) :
  x / y = 5 / 7 :=
by
  sorry

end yan_ratio_distance_l32_32901


namespace same_selection_exists_l32_32495

-- Define the set of dogs
def Dogs : Type := Fin 100

-- Define the courtiers
def Courtiers : Type := Fin 100

-- Each courtier selects exactly three dogs.
def selection (c : Courtiers) : Finset Dogs := sorry

-- Condition: For any two courtiers, there are at least two common dogs they both selected.
axiom common_dogs (c₁ c₂ : Courtiers) (h : c₁ ≠ c₂) :
  (selection c₁ ∩ selection c₂).card ≥ 2

-- Prove that there exist two courtiers who picked exactly the same three dogs.
theorem same_selection_exists :
  ∃ c₁ c₂ : Courtiers, c₁ ≠ c₂ ∧ selection c₁ = selection c₂ :=
sorry

end same_selection_exists_l32_32495


namespace sum_of_k_values_with_distinct_integer_solutions_l32_32369

theorem sum_of_k_values_with_distinct_integer_solutions : 
  ∑ k in {k : ℤ | ∃ p q : ℤ, p ≠ q ∧ 3 * p * q + (-k) * (p + q) + 12 = 0}, k = 0 :=
by
  sorry

end sum_of_k_values_with_distinct_integer_solutions_l32_32369


namespace abs_diff_eq_two_l32_32862

def equation (x y : ℝ) : Prop := y^2 + x^4 = 2 * x^2 * y + 1

theorem abs_diff_eq_two (a b e : ℝ) (ha : equation e a) (hb : equation e b) (hab : a ≠ b) :
  |a - b| = 2 :=
sorry

end abs_diff_eq_two_l32_32862


namespace value_of_x_l32_32635

theorem value_of_x (n x : ℝ) (h1: x = 3 * n) (h2: 2 * n + 3 = 0.2 * 25) : x = 3 :=
by
  sorry

end value_of_x_l32_32635


namespace find_circle_area_l32_32979

-- Definitions for the conditions
variables {r : ℝ} -- the radius of the circle
variables {O F : Type*} -- center O and point F
variables (DF FG : ℝ) (x : ℝ) -- lengths on the circle diameter

-- Given conditions
def diameter_condition (DF FG : ℝ) : Prop :=
  DF = 8 ∧ FG = 4

def distances_condition (DF FG : ℝ) : ℝ :=
  let DG := DF + FG in DG

-- Hypothesis Statements
lemma geometry_hypothesis (DF FG x : ℝ) (hD : diameter_condition DF FG) : r^2 = x^2 + 64 :=
begin
  have h1 : DF = 8 := hD.left,
  have h2 : FG = 4 := hD.right,
  sorry
end

lemma power_point_condition (DF FG x : ℝ) (hD : diameter_condition DF FG) : r^2 - x^2 = 32 :=
begin
  have h1 : DF = 8 := hD.left,
  have h2 : FG = 4 := hD.right,
  sorry
end

-- The proper proof statement to show r^2 = 32 given the conditions
theorem find_circle_area (DF FG x : ℝ) 
  (hD : diameter_condition DF FG)
  (h_geom : r^2 = x^2 + 64)
  (h_pow : r^2 - x^2 = 32) : ∃ r, r^2 = 32 :=
begin
  sorry
end

-- Concluding the problem by defining the area
lemma area_of_circle : Real := π * 32

end find_circle_area_l32_32979


namespace emma_age_when_sister_is_56_l32_32675

theorem emma_age_when_sister_is_56 (e s : ℕ) (he : e = 7) (hs : s = e + 9) : 
  (s + (56 - s) - 9 = 47) :=
by {
  sorry
}

end emma_age_when_sister_is_56_l32_32675


namespace larger_number_is_sixty_three_l32_32011

theorem larger_number_is_sixty_three (x y : ℕ) (h1 : x + y = 84) (h2 : y = 3 * x) : y = 63 :=
  sorry

end larger_number_is_sixty_three_l32_32011


namespace ferry_time_increases_l32_32394

noncomputable def ferryRoundTrip (S V x : ℝ) : ℝ :=
  (S / (V + x)) + (S / (V - x))

theorem ferry_time_increases (S V x : ℝ) (h_V_pos : 0 < V) (h_x_lt_V : x < V) :
  ferryRoundTrip S V (x + 1) > ferryRoundTrip S V x :=
by
  sorry

end ferry_time_increases_l32_32394


namespace problem_statement_l32_32457

theorem problem_statement (n : ℕ) (h1 : 0 < n) (h2 : ∃ k : ℤ, (1/2 + 1/3 + 1/11 + 1/n : ℚ) = k) : ¬ (n > 66) := 
sorry

end problem_statement_l32_32457


namespace smallest_number_divisible_l32_32897

theorem smallest_number_divisible (n : ℕ) : 
  ( ∀ m ∈ {12, 24, 36, 48, 56}, (n - 12) % m = 0) → n = 1020 :=
by
  intro h
  have lcm_val: Nat.lcm 12 (Nat.lcm 24 (Nat.lcm 36 (Nat.lcm 48 56))) = 1008 := by sorry
  have n_minus_12: n - 12 = 1008 := by sorry
  exact Nat.add_eq_of_eq_sub' n_minus_12 12 1020

end smallest_number_divisible_l32_32897


namespace minimum_value_expression_l32_32456

theorem minimum_value_expression (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  4 * a ^ 3 + 8 * b ^ 3 + 27 * c ^ 3 + 64 * d ^ 3 + 2 / (a * b * c * d) ≥ 16 * Real.sqrt 3 :=
by
  sorry

end minimum_value_expression_l32_32456


namespace evaluate_polynomial_at_4_l32_32768

noncomputable def polynomial_horner (x : ℤ) : ℤ :=
  (((((3 * x + 6) * x - 20) * x - 8) * x + 15) * x + 9)

theorem evaluate_polynomial_at_4 :
  polynomial_horner 4 = 3269 :=
by
  sorry

end evaluate_polynomial_at_4_l32_32768


namespace spending_after_drink_l32_32315

variable (X : ℝ)
variable (Y : ℝ)

theorem spending_after_drink (h : X - 1.75 - Y = 6) : Y = X - 7.75 :=
by sorry

end spending_after_drink_l32_32315


namespace sum_of_k_with_distinct_integer_roots_l32_32366

theorem sum_of_k_with_distinct_integer_roots :
  ∑ k in {k : ℤ | ∃ p q : ℤ, p ≠ q ∧ 3*p*q = 12 ∧ p + q = k/3}, k = 0 :=
by
  sorry

end sum_of_k_with_distinct_integer_roots_l32_32366


namespace f_gt_e_plus_2_l32_32821

noncomputable def f (x : ℝ) : ℝ := ( (Real.exp x) / x ) - ( (8 * Real.log (x / 2)) / (x^2) ) + x

lemma slope_at_2 : HasDerivAt f (Real.exp 2 / 4) 2 := 
by 
  sorry

theorem f_gt_e_plus_2 (x : ℝ) (hx : 0 < x) : f x > Real.exp 1 + 2 :=
by
  sorry

end f_gt_e_plus_2_l32_32821


namespace xyz_unique_solution_l32_32906

theorem xyz_unique_solution (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_eq : x + y^2 + z^3 = x * y * z)
  (h_gcd : z = Nat.gcd x y) : x = 5 ∧ y = 1 ∧ z = 1 :=
by
  sorry

end xyz_unique_solution_l32_32906


namespace math_city_police_officers_needed_l32_32466

def number_of_streets : Nat := 10
def initial_intersections : Nat := Nat.choose number_of_streets 2
def non_intersections : Nat := 2
def effective_intersections : Nat := initial_intersections - non_intersections

theorem math_city_police_officers_needed :
  effective_intersections = 43 := by
  sorry

end math_city_police_officers_needed_l32_32466


namespace problem_l32_32594

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
if h : (-1 : ℝ) ≤ x ∧ x < 0 then a*x + 1
else if h : (0 : ℝ) ≤ x ∧ x ≤ 1 then (b*x + 2) / (x + 1)
else 0 -- This should not matter as we only care about the given ranges

theorem problem (a b : ℝ) (h₁ : f 0.5 a b = f 1.5 a b) : a + 3 * b = -10 :=
by
  -- We'll derive equations from given conditions and prove the result.
  sorry

end problem_l32_32594


namespace y_intercept_of_line_l32_32773

theorem y_intercept_of_line (x y : ℝ) (eq : 2 * x - 3 * y = 6) (hx : x = 0) : y = -2 := 
by
  sorry

end y_intercept_of_line_l32_32773


namespace total_pieces_equiv_231_l32_32405

-- Define the arithmetic progression for rods.
def rods_arithmetic_sequence : ℕ → ℕ
| 0 => 0
| n + 1 => 3 * (n + 1)

-- Define the sum of the first 10 terms of the sequence.
def rods_total (n : ℕ) : ℕ :=
  let a := 3
  let d := 3
  n / 2 * (2 * a + (n - 1) * d)

def rods_count : ℕ :=
  rods_total 10

-- Define the 11th triangular number for connectors.
def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def connectors_count : ℕ :=
  triangular_number 11

-- Define the total number of pieces.
def total_pieces : ℕ :=
  rods_count + connectors_count

-- The theorem we aim to prove.
theorem total_pieces_equiv_231 : total_pieces = 231 := by
  sorry

end total_pieces_equiv_231_l32_32405


namespace annual_interest_rate_is_6_percent_l32_32711

-- Definitions from the conditions
def principal : ℕ := 150
def total_amount_paid : ℕ := 159
def interest := total_amount_paid - principal
def interest_rate := (interest * 100) / principal

-- The theorem to prove
theorem annual_interest_rate_is_6_percent :
  interest_rate = 6 := by sorry

end annual_interest_rate_is_6_percent_l32_32711


namespace cannot_determine_right_triangle_l32_32577

def is_right_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180 ∧ (A = 90 ∨ B = 90 ∨ C = 90)

theorem cannot_determine_right_triangle :
  ∀ A B C : ℝ, 
    (A = 2 * B ∧ A = 3 * C) →
    ¬ is_right_triangle A B C :=
by
  intro A B C h
  have h1 : A = 2 * B := h.1
  have h2 : A = 3 * C := h.2
  sorry

end cannot_determine_right_triangle_l32_32577


namespace correct_operation_l32_32524

theorem correct_operation (a : ℝ) : a^8 / a^2 = a^6 :=
by
  -- proof will go here, let's use sorry to indicate it's unfinished
  sorry

end correct_operation_l32_32524


namespace pentagon_area_l32_32952

open Real

/-- The area of a pentagon with sides 18, 25, 30, 28, and 25 units is 950 square units -/
theorem pentagon_area (a b c d e : ℝ) (h₁ : a = 18) (h₂ : b = 25) (h₃ : c = 30) (h₄ : d = 28) (h₅ : e = 25) : 
  ∃ (area : ℝ), area = 950 :=
by {
  sorry
}

end pentagon_area_l32_32952


namespace pure_imaginary_m_eq_zero_l32_32874

noncomputable def z (m : ℝ) : ℂ := (m * (m - 1) : ℂ) + (m - 1) * Complex.I

theorem pure_imaginary_m_eq_zero (m : ℝ) (h : z m = (m - 1) * Complex.I) : m = 0 :=
by
  sorry

end pure_imaginary_m_eq_zero_l32_32874


namespace initial_bottles_count_l32_32734

theorem initial_bottles_count
  (players : ℕ)
  (bottles_per_player_first_break : ℕ)
  (bottles_per_player_end_game : ℕ)
  (remaining_bottles : ℕ)
  (total_bottles_taken_first_break : bottles_per_player_first_break * players = 22)
  (total_bottles_taken_end_game : bottles_per_player_end_game * players = 11)
  (total_remaining_bottles : remaining_bottles = 15) :
  players * bottles_per_player_first_break + players * bottles_per_player_end_game + remaining_bottles = 48 :=
by 
  -- skipping the proof
  sorry

end initial_bottles_count_l32_32734


namespace general_solution_of_diff_eq_l32_32263

theorem general_solution_of_diff_eq
  (f : ℝ → ℝ → ℝ)
  (D : Set (ℝ × ℝ))
  (hf : ∀ x y, f x y = x)
  (hD : D = Set.univ) :
  ∃ C : ℝ, ∀ x : ℝ, ∃ y : ℝ, y = (x^2) / 2 + C :=
by
  sorry

end general_solution_of_diff_eq_l32_32263


namespace charlie_paints_60_sqft_l32_32204

theorem charlie_paints_60_sqft (A B C : ℕ) (total_sqft : ℕ) (h_ratio : A = 3 ∧ B = 5 ∧ C = 2) (h_total : total_sqft = 300) : 
  C * (total_sqft / (A + B + C)) = 60 :=
by
  rcases h_ratio with ⟨rfl, rfl, rfl⟩
  rcases h_total with rfl
  sorry

end charlie_paints_60_sqft_l32_32204


namespace number_of_yellow_balloons_l32_32887

-- Define the problem
theorem number_of_yellow_balloons :
  ∃ (Y B : ℕ), 
  B = Y + 1762 ∧ 
  Y + B = 10 * 859 ∧ 
  Y = 3414 :=
by
  -- Proof is skipped, so we use sorry
  sorry

end number_of_yellow_balloons_l32_32887


namespace find_p_for_quadratic_l32_32811

theorem find_p_for_quadratic (p : ℝ) (h : p ≠ 0) 
  (h_eq : ∀ x : ℝ, p * x^2 - 10 * x + 2 = 0 → x = 5 / p) : p = 12.5 :=
sorry

end find_p_for_quadratic_l32_32811


namespace integer_value_of_expression_l32_32441

theorem integer_value_of_expression (m n p : ℕ) (h1 : 2 ≤ m) (h2 : m ≤ 9)
  (h3 : 2 ≤ n) (h4 : n ≤ 9) (h5 : 2 ≤ p) (h6 : p ≤ 9)
  (h7 : m ≠ n ∧ n ≠ p ∧ m ≠ p) :
  (m + n + p) / (m + n) = 1 :=
sorry

end integer_value_of_expression_l32_32441


namespace total_initial_candles_l32_32054

-- Define the conditions
def used_candles : ℕ := 32
def leftover_candles : ℕ := 12

-- State the theorem
theorem total_initial_candles : used_candles + leftover_candles = 44 := by
  sorry

end total_initial_candles_l32_32054


namespace problem_l32_32721

-- Define first terms
def a_1 : ℕ := 12
def b_1 : ℕ := 48

-- Define the 100th term condition
def a_100 (d_a : ℚ) := 12 + 99 * d_a
def b_100 (d_b : ℚ) := 48 + 99 * d_b

-- Condition that the sum of the 100th terms is 200
def condition (d_a d_b : ℚ) := a_100 d_a + b_100 d_b = 200

-- Define the value of the sum of the first 100 terms
def sequence_sum (d_a d_b : ℚ) := 100 * 60 + (140 / 99) * ((99 * 100) / 2)

-- The proof theorem
theorem problem : ∀ d_a d_b : ℚ, condition d_a d_b → sequence_sum d_a d_b = 13000 :=
by
  intros d_a d_b h_cond
  sorry

end problem_l32_32721


namespace compute_fraction_sum_l32_32735

variable (a b c : ℝ)
open Real

theorem compute_fraction_sum (h1 : (a * c) / (a + b) + (b * a) / (b + c) + (c * b) / (c + a) = -15)
                            (h2 : (b * c) / (a + b) + (c * a) / (b + c) + (a * b) / (c + a) = 6) :
  (b / (a + b) + c / (b + c) + a / (c + a)) = 12 := 
sorry

end compute_fraction_sum_l32_32735


namespace Shara_will_owe_money_l32_32606

theorem Shara_will_owe_money
    (B : ℕ)
    (h1 : 6 * 10 = 60)
    (h2 : B / 2 = 60)
    (h3 : 4 * 10 = 40)
    (h4 : 60 + 40 = 100) :
  B - 100 = 20 :=
sorry

end Shara_will_owe_money_l32_32606


namespace pentagon_area_l32_32544

noncomputable def square_area (side_length : ℤ) : ℤ :=
  side_length * side_length

theorem pentagon_area (CF : ℤ) (a b : ℤ) (CE : ℤ) (ED : ℤ) (EF : ℤ) :
  (CF = 5) →
  (a = CE + ED) →
  (b = EF) →
  (CE < ED) →
  CF * CF = CE * CE + EF * EF →
  square_area a + square_area b - (CE * EF / 2) = 71 :=
by
  intros hCF ha hb hCE_lt_ED hPythagorean
  sorry

end pentagon_area_l32_32544


namespace fencing_required_l32_32913

def width : ℝ := 25
def area : ℝ := 260
def height_difference : ℝ := 15
def extra_fencing_per_5ft_height : ℝ := 2

noncomputable def length : ℝ := area / width

noncomputable def expected_fencing : ℝ := 2 * length + width + (height_difference / 5) * extra_fencing_per_5ft_height

-- Theorem stating the problem's conclusion
theorem fencing_required : expected_fencing = 51.8 := by
  sorry -- Proof will go here

end fencing_required_l32_32913


namespace candy_pieces_per_package_l32_32866

theorem candy_pieces_per_package (packages_gum : ℕ) (packages_candy : ℕ) (total_candies : ℕ) :
  packages_gum = 21 →
  packages_candy = 45 →
  total_candies = 405 →
  total_candies / packages_candy = 9 := by
  intros h1 h2 h3
  sorry

end candy_pieces_per_package_l32_32866


namespace dice_probability_sum_is_ten_l32_32890

noncomputable def probability_sum_is_ten : ℚ := 27 / 216

theorem dice_probability_sum_is_ten : 
  let faces := Fin 6
  ( {outcomes : Multiset (faces + 1) // outcomes.sum = 10} ).card / (faces.card * faces.card * faces.card) = probability_sum_is_ten :=
by
  sorry

end dice_probability_sum_is_ten_l32_32890


namespace Kelly_remaining_games_l32_32713

-- Definitions according to the conditions provided
def initial_games : ℝ := 121.0
def given_away : ℝ := 99.0
def remaining_games : ℝ := initial_games - given_away

-- The proof problem statement
theorem Kelly_remaining_games : remaining_games = 22.0 :=
by
  -- sorry is used here to skip the proof
  sorry

end Kelly_remaining_games_l32_32713


namespace limit_cos_x_limit_sin_3x_limit_cos_pi_x_limit_sin_pi_x_inv_l32_32174

-- Statement a
theorem limit_cos_x (x : ℝ) :
  (∃ ε > 0, ∀ x, abs(x) < ε → (∃ L, L = 1/2 ∧ tendsto (λ x, (1 - cos x) / x^2) (nhds 0) (𝓝 L))) :=
sorry

-- Statement b
theorem limit_sin_3x (x : ℝ) :
  (∃ δ > 0, ∀ x, abs(x - π) < δ → (∃ L, L = -3 ∧ tendsto (λ x, (sin (3 * x)) / (x - π)) (nhds π) (𝓝 L))) :=
sorry

-- Statement c
theorem limit_cos_pi_x (x : ℝ) :
  (∃ δ > 0, ∀ x, abs(x - 1) < δ → (∃ L, L = -π / 2 ∧ tendsto (λ x, (cos (π * x / 2)) / (x - 1)) (nhds 1) (𝓝 L))) :=
sorry

-- Statement d
theorem limit_sin_pi_x_inv (x : ℝ) :
  (∃ ε > 0, ∀ x, abs(x) < ε → (∃ L, L = π^2 ∧ tendsto (λ x, (1 / x) * sin (π / (1 + π * x))) (nhds 0) (𝓝 L))) :=
sorry

end limit_cos_x_limit_sin_3x_limit_cos_pi_x_limit_sin_pi_x_inv_l32_32174


namespace coordinates_of_C_are_correct_l32_32914

noncomputable section 

def Point := (ℝ × ℝ)

def A : Point := (1, 3)
def B : Point := (13, 9)

def vector_AB (A B : Point) : Point :=
  (B.1 - A.1, B.2 - A.2)

def scalar_mult (s : ℝ) (v : Point) : Point :=
  (s * v.1, s * v.2)

def add_vectors (v1 v2 : Point) : Point :=
  (v1.1 + v2.1, v1.2 + v2.2)

def C : Point :=
  let AB := vector_AB A B
  add_vectors B (scalar_mult (1 / 2) AB)

theorem coordinates_of_C_are_correct : C = (19, 12) := by sorry

end coordinates_of_C_are_correct_l32_32914


namespace volunteers_selection_l32_32568

theorem volunteers_selection :
  let total_volunteers := 6
  let boys := 4
  let girls := 2
  let chosen_volunteers := 4
  (boys > 0 ∧ girls > 0) →
  ∃ (ways : ℕ), ways = (Nat.choose 4 3 * Nat.choose 2 1) + (Nat.choose 4 2 * Nat.choose 2 2) ∧ ways = 14 :=  
by
  intros _ _ _ _ _ _
  use (Nat.choose 4 3 * Nat.choose 2 1) + (Nat.choose 4 2 * Nat.choose 2 2)
  split
  · rfl
  · exact Eq.refl (8 + 6)
  sorry

end volunteers_selection_l32_32568


namespace ratio_of_areas_l32_32846

structure Triangle :=
  (AB BC AC AD AE : ℝ)
  (AB_pos : 0 < AB)
  (BC_pos : 0 < BC)
  (AC_pos : 0 < AC)
  (AD_pos : 0 < AD)
  (AE_pos : 0 < AE)

theorem ratio_of_areas (t : Triangle)
  (hAB : t.AB = 30)
  (hBC : t.BC = 45)
  (hAC : t.AC = 54)
  (hAD : t.AD = 24)
  (hAE : t.AE = 18) :
  (t.AD / t.AB) * (t.AE / t.AC) / (1 - (t.AD / t.AB) * (t.AE / t.AC)) = 4 / 11 :=
by
  sorry

end ratio_of_areas_l32_32846


namespace average_age_of_two_new_men_l32_32002

theorem average_age_of_two_new_men :
  ∀ (A N : ℕ), 
    (∀ n : ℕ, n = 12) → 
    (N = 21 + 23 + 12) → 
    (A = N / 2) → 
    A = 28 :=
by
  intros A N twelve men_replace_eq_avg men_avg_eq
  sorry

end average_age_of_two_new_men_l32_32002


namespace packs_of_green_bouncy_balls_l32_32729

/-- Maggie bought 10 bouncy balls in each pack of red, yellow, and green bouncy balls.
    She bought 4 packs of red bouncy balls, 8 packs of yellow bouncy balls, and some 
    packs of green bouncy balls. In total, she bought 160 bouncy balls. This theorem 
    aims to prove how many packs of green bouncy balls Maggie bought. 
 -/
theorem packs_of_green_bouncy_balls (red_packs : ℕ) (yellow_packs : ℕ) (total_balls : ℕ) (balls_per_pack : ℕ) 
(pack : ℕ) :
  red_packs = 4 →
  yellow_packs = 8 →
  balls_per_pack = 10 →
  total_balls = 160 →
  red_packs * balls_per_pack + yellow_packs * balls_per_pack + pack * balls_per_pack = total_balls →
  pack = 4 :=
by
  intros h_red h_yellow h_balls_per_pack h_total_balls h_eq
  sorry

end packs_of_green_bouncy_balls_l32_32729


namespace limit_exists_implies_d_eq_zero_l32_32988

variable (a₁ d : ℝ) (S : ℕ → ℝ)

noncomputable def limExists := ∃ L : ℝ, ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, abs (S n - L) < ε

def is_sum_of_arithmetic_sequence (S : ℕ → ℝ) (a₁ d : ℝ) :=
  ∀ n : ℕ, S n = (a₁ * n + d * (n * (n - 1) / 2))

theorem limit_exists_implies_d_eq_zero (h₁ : ∀ n : ℕ, n > 0 → S n = (a₁ * n + d * (n * (n - 1) / 2))) :
  limExists S → d = 0 :=
by sorry

end limit_exists_implies_d_eq_zero_l32_32988


namespace find_d_l32_32140

theorem find_d (d : ℕ) : (1059 % d = 1417 % d) ∧ (1059 % d = 2312 % d) ∧ (1417 % d = 2312 % d) ∧ (d > 1) → d = 179 :=
by
  sorry

end find_d_l32_32140


namespace speeds_of_bus_and_car_l32_32622

theorem speeds_of_bus_and_car
  (d t : ℝ) (v1 v2 : ℝ)
  (h1 : 1.5 * v1 + 1.5 * v2 = d)
  (h2 : 2.5 * v1 + 1 * v2 = d) :
  v1 = 40 ∧ v2 = 80 :=
by sorry

end speeds_of_bus_and_car_l32_32622


namespace fred_now_has_l32_32815

-- Definitions based on conditions
def original_cards : ℕ := 40
def purchased_cards : ℕ := 22

-- Theorem to prove the number of cards Fred has now
theorem fred_now_has (original_cards : ℕ) (purchased_cards : ℕ) : original_cards - purchased_cards = 18 :=
by
  sorry

end fred_now_has_l32_32815


namespace quadratic_roots_bounds_l32_32746

theorem quadratic_roots_bounds (a b c : ℤ) (p1 p2 : ℝ) (h_a_pos : a > 0) 
  (h_int_coeff : ∀ x : ℤ, x = a ∨ x = b ∨ x = c) 
  (h_distinct_roots : p1 ≠ p2) 
  (h_roots : a * p1^2 + b * p1 + c = 0 ∧ a * p2^2 + b * p2 + c = 0) 
  (h_roots_bounds : 0 < p1 ∧ p1 < 1 ∧ 0 < p2 ∧ p2 < 1) : 
     a ≥ 5 := 
sorry

end quadratic_roots_bounds_l32_32746


namespace equal_circles_common_point_l32_32387

theorem equal_circles_common_point (n : ℕ) (r : ℝ) 
  (centers : Fin n → ℝ × ℝ)
  (h : ∀ i j k : Fin n, i ≠ j → j ≠ k → i ≠ k →
    ∃ (p : ℝ × ℝ),
      dist p (centers i) = r ∧
      dist p (centers j) = r ∧
      dist p (centers k) = r) :
  ∃ O : ℝ × ℝ, ∀ i : Fin n, dist O (centers i) = r := sorry

end equal_circles_common_point_l32_32387


namespace correct_calculation_l32_32167

def calculation_is_correct (a b x y : ℝ) : Prop :=
  (3 * x^2 * y - 2 * y * x^2 = x^2 * y)

theorem correct_calculation :
  ∀ (a b x y : ℝ), calculation_is_correct a b x y :=
by
  intros a b x y
  sorry

end correct_calculation_l32_32167


namespace cubic_identity_l32_32249

theorem cubic_identity (x : ℝ) (h : x - (1 / x) = 5) : x^3 - (1 / x^3) = 140 := 
  sorry

end cubic_identity_l32_32249


namespace inequality_condition_l32_32533

theorem inequality_condition (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*a*x + a > 0) ↔ (0 < a ∧ a < 1) ∨ (False) := 
sorry

end inequality_condition_l32_32533


namespace tuna_per_customer_l32_32471

noncomputable def total_customers := 100
noncomputable def total_tuna := 10
noncomputable def weight_per_tuna := 200
noncomputable def customers_without_fish := 20

theorem tuna_per_customer : (total_tuna * weight_per_tuna) / (total_customers - customers_without_fish) = 25 := by
  sorry

end tuna_per_customer_l32_32471


namespace log_equation_solution_l32_32631

theorem log_equation_solution (x : ℝ) (h₁ : x > 0) (h₂ : x ≠ 1) (h₃ : x ≠ 1/16) (h₄ : x ≠ 1/2) 
    (h_eq : (Real.log 2 / Real.log (4 * Real.sqrt x)) / (Real.log 2 / Real.log (2 * x)) 
            + (Real.log 2 / Real.log (2 * x)) * (Real.log (2 * x) / Real.log (1 / 2)) = 0) 
    : x = 4 := 
sorry

end log_equation_solution_l32_32631


namespace courtiers_selection_l32_32501

theorem courtiers_selection (dogs : Finset ℕ) (courtiers : Finset (Finset ℕ))
  (h1 : dogs.card = 100)
  (h2 : courtiers.card = 100)
  (h3 : ∀ c ∈ courtiers, c.card = 3)
  (h4 : ∀ {c1 c2 : Finset ℕ}, c1 ∈ courtiers → c2 ∈ courtiers → c1 ≠ c2 → (c1 ∩ c2).card = 2) :
  ∃ c1 c2 ∈ courtiers, c1 = c2 :=
sorry

end courtiers_selection_l32_32501


namespace option1_payment_correct_option2_payment_correct_most_cost_effective_for_30_l32_32182

variables (x : ℕ) (hx : x > 10)

def suit_price : ℕ := 1000
def tie_price : ℕ := 200
def num_suits : ℕ := 10

-- Option 1: Buy one suit, get one tie for free
def option1_payment : ℕ := 200 * x + 8000

-- Option 2: All items sold at a 10% discount
def option2_payment : ℕ := (10 * 1000 + x * 200) * 9 / 10

-- For x = 30, which option is more cost-effective
def x_value := 30
def option1_payment_30 : ℕ := 200 * x_value + 8000
def option2_payment_30 : ℕ := (10 * 1000 + x_value * 200) * 9 / 10
def more_cost_effective_option_30 : ℕ := if option1_payment_30 < option2_payment_30 then option1_payment_30 else option2_payment_30

-- Most cost-effective option for x = 30 with new combination plan
def combination_payment_30 : ℕ := 10000 + 20 * 200 * 9 / 10

-- Statements to be proved
theorem option1_payment_correct : option1_payment x = 200 * x + 8000 := sorry

theorem option2_payment_correct : option2_payment x = (10 * 1000 + x * 200) * 9 / 10 := sorry

theorem most_cost_effective_for_30 :
  option1_payment_30 = 14000 ∧ 
  option2_payment_30 = 14400 ∧ 
  more_cost_effective_option_30 = 14000 ∧
  combination_payment_30 = 13600 := sorry

end option1_payment_correct_option2_payment_correct_most_cost_effective_for_30_l32_32182


namespace number_2018_location_l32_32057

-- Define the odd square pattern as starting positions of rows
def odd_square (k : ℕ) : ℕ := (2 * k - 1) ^ 2

-- Define the conditions in terms of numbers in each row
def start_of_row (n : ℕ) : ℕ := (2 * n - 1) ^ 2 + 1

def number_at_row_column (n m : ℕ) :=
  start_of_row n + (m - 1)

theorem number_2018_location :
  number_at_row_column 44 82 = 2018 :=
by
  sorry

end number_2018_location_l32_32057


namespace find_k_value_l32_32854

theorem find_k_value (x k : ℝ) (hx : Real.logb 9 3 = x) (hk : Real.logb 3 81 = k * x) : k = 8 :=
by sorry

end find_k_value_l32_32854


namespace chickens_count_l32_32934

-- Define conditions
def cows : Nat := 4
def sheep : Nat := 3
def bushels_per_cow : Nat := 2
def bushels_per_sheep : Nat := 2
def bushels_per_chicken : Nat := 3
def total_bushels_needed : Nat := 35

-- The main theorem to be proven
theorem chickens_count : 
  (total_bushels_needed - ((cows * bushels_per_cow) + (sheep * bushels_per_sheep))) / bushels_per_chicken = 7 :=
by
  sorry

end chickens_count_l32_32934


namespace remainder_9053_div_98_l32_32896

theorem remainder_9053_div_98 : 9053 % 98 = 37 :=
by sorry

end remainder_9053_div_98_l32_32896


namespace least_number_divisible_l32_32878

theorem least_number_divisible (x : ℕ) (h1 : x = 857) 
  (h2 : (x + 7) % 24 = 0) 
  (h3 : (x + 7) % 36 = 0) 
  (h4 : (x + 7) % 54 = 0) :
  (x + 7) % 32 = 0 := 
sorry

end least_number_divisible_l32_32878


namespace zero_point_six_one_eight_method_l32_32386

theorem zero_point_six_one_eight_method (a b : ℝ) (h : a = 2 ∧ b = 4) : 
  ∃ x₁ x₂, x₁ = a + 0.618 * (b - a) ∧ x₂ = a + b - x₁ ∧ (x₁ = 3.236 ∨ x₂ = 2.764) := by
  sorry

end zero_point_six_one_eight_method_l32_32386


namespace range_of_k_smallest_m_l32_32265

-- Part (1)
theorem range_of_k (f : ℝ → ℝ) (k : ℝ) (h : ∀ x > 0, f x > k * x - 1 / 2) :
  k < 1 - log 2 :=
sorry

-- Part (2)
theorem smallest_m (f : ℝ → ℝ) (m : ℕ) (h : ∀ x > 0, f (m + x) < f m * exp x) :
  m = 3 :=
sorry

end range_of_k_smallest_m_l32_32265


namespace vinny_final_weight_l32_32155

theorem vinny_final_weight :
  let initial_weight := 300
  let first_month_loss := 20
  let second_month_loss := first_month_loss / 2
  let third_month_loss := second_month_loss / 2
  let fourth_month_loss := third_month_loss / 2
  let fifth_month_loss := 12
  let total_loss := first_month_loss + second_month_loss + third_month_loss + fourth_month_loss + fifth_month_loss
  let final_weight := initial_weight - total_loss
  final_weight = 250.5 :=
by
  sorry

end vinny_final_weight_l32_32155


namespace correct_calculation_l32_32166

def calculation_is_correct (a b x y : ℝ) : Prop :=
  (3 * x^2 * y - 2 * y * x^2 = x^2 * y)

theorem correct_calculation :
  ∀ (a b x y : ℝ), calculation_is_correct a b x y :=
by
  intros a b x y
  sorry

end correct_calculation_l32_32166


namespace part_a_part_b_part_c_l32_32786

-- Part (a)
theorem part_a (x y : ℕ) (h : (2 * x + 11 * y) = 3 * x + 4 * y) : x = 7 * y := by
  sorry

-- Part (b)
theorem part_b (u v : ℚ) : ∃ (x y : ℚ), (x + y) / 2 = (u.num * v.den + v.num * u.den) / (2 * u.den * v.den) := by
  sorry

-- Part (c)
theorem part_c (u v : ℚ) (h : u < v) : ∀ (m : ℚ), (m.num = u.num + v.num) ∧ (m.den = u.den + v.den) → u < m ∧ m < v := by
  sorry

end part_a_part_b_part_c_l32_32786


namespace volume_of_cube_with_surface_area_l32_32191

theorem volume_of_cube_with_surface_area (S : ℝ) (hS : S = 294) : 
  ∃ V : ℝ, V = 343 :=
by
  let s := (S / 6).sqrt
  have hs : s = 7 := by sorry
  use s ^ 3
  simp [hs]
  exact sorry

end volume_of_cube_with_surface_area_l32_32191


namespace set_diff_M_N_l32_32853

def set_diff {α : Type} (A B : Set α) : Set α := {x | x ∈ A ∧ x ∉ B}

def M : Set ℝ := {x | |x + 1| ≤ 2}

def N : Set ℝ := {x | ∃ α : ℝ, x = |Real.sin α| }

theorem set_diff_M_N :
  set_diff M N = {x | -3 ≤ x ∧ x < 0} :=
by
  sorry

end set_diff_M_N_l32_32853


namespace circle_equation_through_points_l32_32563

theorem circle_equation_through_points (A B: ℝ × ℝ) (C : ℝ × ℝ)
  (hA : A = (1, -1)) (hB : B = (-1, 1)) (hC : C.1 + C.2 = 2)
  (hAC : dist A C = dist B C) :
  (x - C.1) ^ 2 + (y - C.2) ^ 2 = 4 :=
by
  sorry

end circle_equation_through_points_l32_32563


namespace min_cost_to_win_l32_32540

theorem min_cost_to_win (n : ℕ) : 
  (∀ m : ℕ, m = 0 →
  (∀ cents : ℕ, 
  (n = 5 * m ∨ n = m + 1) ∧ n > 2008 ∧ n % 100 = 42 → 
  cents = 35)) :=
sorry

end min_cost_to_win_l32_32540


namespace triangle_angle_property_l32_32706

variables {a b c : ℝ}
variables {A B C : ℝ} -- angles in triangle ABC

-- definition of a triangle side condition
def triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a

-- condition given in the problem
def satisfies_condition (a b c : ℝ) : Prop := b^2 = a^2 + c^2

-- angle property based on given problem
def angle_B_is_right (A B C : ℝ) : Prop := B = 90

theorem triangle_angle_property (a b c : ℝ) (A B C : ℝ)
  (ht : triangle a b c) 
  (hc : satisfies_condition a b c) : 
  angle_B_is_right A B C :=
sorry

end triangle_angle_property_l32_32706


namespace contradiction_assumption_l32_32107

theorem contradiction_assumption (a : ℝ) (h : a < |a|) : ¬(a ≥ 0) :=
by 
  sorry

end contradiction_assumption_l32_32107


namespace slope_angle_tangent_line_at_zero_l32_32222

noncomputable def curve (x : ℝ) : ℝ := 2 * x - Real.exp x

noncomputable def slope_at (x : ℝ) : ℝ := 
  (deriv curve) x

theorem slope_angle_tangent_line_at_zero : 
  Real.arctan (slope_at 0) = Real.pi / 4 :=
by
  sorry

end slope_angle_tangent_line_at_zero_l32_32222


namespace complex_addition_l32_32626

def c : ℂ := 3 - 2 * Complex.I
def d : ℂ := 1 + 3 * Complex.I

theorem complex_addition : 3 * c + 4 * d = 13 + 6 * Complex.I := by
  -- proof goes here
  sorry

end complex_addition_l32_32626


namespace cubic_identity_l32_32257

theorem cubic_identity (x : ℝ) (h : x - 1/x = 5) : x^3 - 1/x^3 = 140 :=
by
  sorry

end cubic_identity_l32_32257


namespace cistern_total_wet_surface_area_l32_32645

-- Define the length, width, and depth of water in the cistern
def length : ℝ := 9
def width : ℝ := 4
def depth : ℝ := 1.25

-- Define the bottom surface area
def bottom_surface_area : ℝ := length * width

-- Define the longer side surface area
def longer_side_surface_area_each : ℝ := depth * length

-- Define the shorter end surface area
def shorter_end_surface_area_each : ℝ := depth * width

-- Calculate the total wet surface area
def total_wet_surface_area : ℝ := bottom_surface_area + 2 * longer_side_surface_area_each + 2 * shorter_end_surface_area_each

-- The theorem to be proved
theorem cistern_total_wet_surface_area :
  total_wet_surface_area = 68.5 :=
by
  -- since bottom_surface_area = 36,
  -- 2 * longer_side_surface_area_each = 22.5, and
  -- 2 * shorter_end_surface_area_each = 10
  -- the total will be equal to 68.5
  sorry

end cistern_total_wet_surface_area_l32_32645


namespace range_of_inclination_angle_l32_32356

theorem range_of_inclination_angle (α : ℝ) :
  let A := (-2 : ℝ, 0 : ℝ)
  let ellipse := ∀ x y : ℝ, x^2 / 2 + y^2 = 1
  ∃ B C : ℝ × ℝ, (∃ l : ℝ → ℝ × ℝ, ∀ t : ℝ, l t = (-2 + t * Real.cos α, t * Real.sin α) ∧ ellipse (fst (l t)) (snd (l t))) ∧ B ≠ C ↔ (0 ≤ α ∧ α < Real.arcsin (Real.sqrt 3 / 3) ∨ π - Real.arcsin (Real.sqrt 3 / 3) < α ∧ α < π) :=
begin
  sorry
end

end range_of_inclination_angle_l32_32356


namespace find_constants_l32_32850

noncomputable def f (a b x : ℝ) : ℝ :=
(a * x + b) / (x + 1)

theorem find_constants (a b : ℝ) (x : ℝ) (h : x ≠ -1) : 
  (f a b (f a b x) = x) → (a = -1 ∧ ∀ b, ∃ c : ℝ, b = c) :=
by 
  sorry

end find_constants_l32_32850


namespace laura_owes_amount_l32_32849

def principal : ℝ := 35
def rate : ℝ := 0.04
def time : ℝ := 1
def interest : ℝ := principal * rate * time
def total_amount : ℝ := principal + interest

theorem laura_owes_amount :
  total_amount = 36.40 := by
  sorry

end laura_owes_amount_l32_32849


namespace number_of_rocks_chosen_l32_32882

open Classical

theorem number_of_rocks_chosen
  (total_rocks : ℕ)
  (slate_rocks : ℕ)
  (pumice_rocks : ℕ)
  (granite_rocks : ℕ)
  (probability_both_slate : ℚ) :
  total_rocks = 44 →
  slate_rocks = 14 →
  pumice_rocks = 20 →
  granite_rocks = 10 →
  probability_both_slate = (14 / 44) * (13 / 43) →
  2 = 2 := 
by {
  sorry
}

end number_of_rocks_chosen_l32_32882


namespace balance_expenses_l32_32541

-- Define the basic amounts paid by Alice, Bob, and Carol
def alicePaid : ℕ := 120
def bobPaid : ℕ := 150
def carolPaid : ℕ := 210

-- The total expenditure
def totalPaid : ℕ := alicePaid + bobPaid + carolPaid

-- Each person's share of the total expenses
def eachShare : ℕ := totalPaid / 3

-- Amount Alice should give to balance the expenses
def a : ℕ := eachShare - alicePaid

-- Amount Bob should give to balance the expenses
def b : ℕ := eachShare - bobPaid

-- The statement to be proven
theorem balance_expenses : a - b = 30 :=
by
  sorry

end balance_expenses_l32_32541


namespace michael_lap_time_l32_32808

theorem michael_lap_time (T : ℝ) :
  (∀ (lap_time_donovan : ℝ), lap_time_donovan = 45 → (9 * T) / lap_time_donovan + 1 = 9 → T = 40) :=
by
  intro lap_time_donovan
  intro h1
  intro h2
  sorry

end michael_lap_time_l32_32808


namespace sufficient_condition_for_equation_l32_32672

theorem sufficient_condition_for_equation (x y z : ℤ) (h1 : x = y + 1) (h2 : z = y) :
    x * (x - y) + y * (y - z) + z * (z - x) = 1 :=
by
  -- Proof omitted
  sorry

end sufficient_condition_for_equation_l32_32672


namespace negation_of_universal_statement_l32_32507

open Real

theorem negation_of_universal_statement :
  ¬(∀ x : ℝ, x^3 > x^2) ↔ ∃ x : ℝ, x^3 ≤ x^2 :=
by
  sorry

end negation_of_universal_statement_l32_32507


namespace quadratic_expression_positive_intervals_l32_32410

noncomputable def quadratic_expression (x : ℝ) : ℝ := (x + 3) * (x - 1)
def interval_1 (x : ℝ) : Prop := x < (1 - Real.sqrt 13) / 2
def interval_2 (x : ℝ) : Prop := x > (1 + Real.sqrt 13) / 2

theorem quadratic_expression_positive_intervals (x : ℝ) :
  quadratic_expression x > 0 ↔ interval_1 x ∨ interval_2 x :=
by {
  sorry
}

end quadratic_expression_positive_intervals_l32_32410


namespace baker_new_cakes_bought_l32_32208

variable (total_cakes initial_sold sold_more_than_bought : ℕ)

def new_cakes_bought (total_cakes initial_sold sold_more_than_bought : ℕ) : ℕ :=
  total_cakes - (initial_sold + sold_more_than_bought)

theorem baker_new_cakes_bought (total_cakes initial_sold sold_more_than_bought : ℕ) 
  (h1 : total_cakes = 170)
  (h2 : initial_sold = 78)
  (h3 : sold_more_than_bought = 47) :
  new_cakes_bought total_cakes initial_sold sold_more_than_bought = 78 :=
  sorry

end baker_new_cakes_bought_l32_32208


namespace geometric_sequence_product_geometric_sequence_sum_not_definitely_l32_32687

theorem geometric_sequence_product (a b : ℕ → ℝ) (r1 r2 : ℝ) 
  (ha : ∀ n, a (n+1) = r1 * a n)
  (hb : ∀ n, b (n+1) = r2 * b n) :
  ∃ r3, ∀ n, (a n * b n) = r3 * (a (n-1) * b (n-1)) :=
sorry

theorem geometric_sequence_sum_not_definitely (a b : ℕ → ℝ) (r1 r2 : ℝ) 
  (ha : ∀ n, a (n+1) = r1 * a n)
  (hb : ∀ n, b (n+1) = r2 * b n) :
  ¬ ∀ r3, ∃ N, ∀ n ≥ N, (a n + b n) = r3 * (a (n-1) + b (n-1)) :=
sorry

end geometric_sequence_product_geometric_sequence_sum_not_definitely_l32_32687


namespace canal_depth_l32_32636

-- Define the problem parameters
def top_width : ℝ := 6
def bottom_width : ℝ := 4
def cross_section_area : ℝ := 10290

-- Define the theorem to prove the depth of the canal
theorem canal_depth :
  (1 / 2) * (top_width + bottom_width) * h = cross_section_area → h = 2058 :=
by sorry

end canal_depth_l32_32636


namespace polygon_diagonals_integer_l32_32744

theorem polygon_diagonals_integer (n : ℤ) : ∃ k : ℤ, 2 * k = n * (n - 3) := by
sorry

end polygon_diagonals_integer_l32_32744


namespace time_to_cover_escalator_l32_32800

variable (v_e v_p L : ℝ)

theorem time_to_cover_escalator
  (h_v_e : v_e = 15)
  (h_v_p : v_p = 5)
  (h_L : L = 180) :
  (L / (v_e + v_p) = 9) :=
by
  -- Set up the given conditions
  rw [h_v_e, h_v_p, h_L]
  -- This will now reduce to proving 180 / (15 + 5) = 9
  sorry

end time_to_cover_escalator_l32_32800


namespace maximum_distance_l32_32634

-- Defining the conditions
def highway_mileage : ℝ := 12.2
def city_mileage : ℝ := 7.6
def gasoline_amount : ℝ := 22

-- Mathematical equivalent proof statement
theorem maximum_distance (h_mileage : ℝ) (g_amount : ℝ) : h_mileage = 12.2 ∧ g_amount = 22 → g_amount * h_mileage = 268.4 :=
by
  intro h
  sorry

end maximum_distance_l32_32634


namespace find_N_l32_32284

theorem find_N : ∃ N : ℕ, (∃ x y : ℕ, 
(10 * x + y = N) ∧ 
(4 * x + 2 * y = N / 2) ∧ 
(1 ≤ x) ∧ (x ≤ 9) ∧ 
(0 ≤ y) ∧ (y ≤ 9)) ∧
(N = 32 ∨ N = 64 ∨ N = 96) :=
by {
    have h1 : ∀ x y : ℕ, 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 → 4 * x + 2 * y = (10 * x + y) / 2 → true, sorry,
    exact h1
}

end find_N_l32_32284


namespace largest_int_square_3_digits_base_7_l32_32986

theorem largest_int_square_3_digits_base_7 :
  ∃ (N : ℕ), (7^2 ≤ N^2) ∧ (N^2 < 7^3) ∧ 
  ∃ k : ℕ, N = k ∧ k^2 ≥ 7^2 ∧ k^2 < 7^3 ∧
  N = 45 := sorry

end largest_int_square_3_digits_base_7_l32_32986


namespace safe_dishes_count_l32_32440

theorem safe_dishes_count (total_dishes vegan_dishes vegan_with_nuts : ℕ) 
  (h1 : vegan_dishes = total_dishes / 3) 
  (h2 : vegan_with_nuts = 4) 
  (h3 : vegan_dishes = 6) : vegan_dishes - vegan_with_nuts = 2 :=
by
  sorry

end safe_dishes_count_l32_32440


namespace height_percentage_difference_l32_32022

theorem height_percentage_difference 
  (r1 h1 r2 h2 : ℝ) 
  (V1_eq_V2 : π * r1^2 * h1 = π * r2^2 * h2)
  (r2_eq_1_2_r1 : r2 = (6 / 5) * r1) :
  h1 = (36 / 25) * h2 :=
by
  sorry

end height_percentage_difference_l32_32022


namespace syllogism_sequence_l32_32962

theorem syllogism_sequence (P Q R : Prop)
  (h1 : R)
  (h2 : Q)
  (h3 : P) : 
  (Q ∧ R → P) → (R → P) ∧ (Q → (P ∧ R)) := 
by
  sorry

end syllogism_sequence_l32_32962


namespace find_borrowed_interest_rate_l32_32792

theorem find_borrowed_interest_rate :
  ∀ (principal : ℝ) (time : ℝ) (lend_rate : ℝ) (gain_per_year : ℝ) (borrow_rate : ℝ),
  principal = 5000 →
  time = 1 → -- Considering per year
  lend_rate = 0.06 →
  gain_per_year = 100 →
  (principal * lend_rate - gain_per_year = principal * borrow_rate * time) →
  borrow_rate * 100 = 4 :=
by
  intros principal time lend_rate gain_per_year borrow_rate h_principal h_time h_lend_rate h_gain h_equation
  rw [h_principal, h_time, h_lend_rate] at h_equation
  have h_borrow_rate := h_equation
  sorry

end find_borrowed_interest_rate_l32_32792


namespace perpendicular_line_sum_l32_32960

theorem perpendicular_line_sum (a b c : ℝ) 
  (h1 : -a / 4 * 2 / 5 = -1)
  (h2 : 10 * 1 + 4 * c - 2 = 0)
  (h3 : 2 * 1 - 5 * c + b = 0) : 
  a + b + c = -4 :=
sorry

end perpendicular_line_sum_l32_32960


namespace average_weight_estimation_exclude_friend_l32_32543

theorem average_weight_estimation_exclude_friend
    (w : ℝ)
    (H1 : 62.4 < w ∧ w < 72.1)
    (H2 : 60.3 < w ∧ w < 70.6)
    (H3 : w ≤ 65.9)
    (H4 : 63.7 < w ∧ w < 66.3)
    (H5 : 75.0 ≤ w ∧ w ≤ 78.5) :
    False ∧ ((63.7 < w ∧ w ≤ 65.9) → (w = 64.8)) :=
by
  sorry

end average_weight_estimation_exclude_friend_l32_32543


namespace king_gvidon_descendants_l32_32848

def number_of_sons : ℕ := 5
def number_of_descendants_with_sons : ℕ := 100
def number_of_sons_each : ℕ := 3
def number_of_grandsons : ℕ := number_of_descendants_with_sons * number_of_sons_each

def total_descendants : ℕ := number_of_sons + number_of_grandsons

theorem king_gvidon_descendants : total_descendants = 305 :=
by
  sorry

end king_gvidon_descendants_l32_32848


namespace pressure_increases_when_block_submerged_l32_32781

theorem pressure_increases_when_block_submerged 
  (P0 : ℝ) (ρ : ℝ) (g : ℝ) (h0 h1 : ℝ) :
  h1 > h0 → 
  (P0 + ρ * g * h1) > (P0 + ρ * g * h0) :=
by
  intros h1_gt_h0
  sorry

end pressure_increases_when_block_submerged_l32_32781


namespace correctness_check_l32_32164

noncomputable def questionD (x y : ℝ) : Prop := 
  3 * x^2 * y - 2 * y * x^2 = x^2 * y

theorem correctness_check (x y : ℝ) : questionD x y :=
by 
  sorry

end correctness_check_l32_32164


namespace two_digit_number_solution_l32_32291

theorem two_digit_number_solution (N : ℕ) (x y : ℕ) :
  (10 * x + y = N) ∧ (4 * x + 2 * y = (10 * x + y) / 2) →
  N = 32 ∨ N = 64 ∨ N = 96 := 
sorry

end two_digit_number_solution_l32_32291


namespace number_of_boxes_on_pallet_l32_32047

-- Define the total weight of the pallet.
def total_weight_of_pallet : ℤ := 267

-- Define the weight of each box.
def weight_of_each_box : ℤ := 89

-- The theorem states that given the total weight of the pallet and the weight of each box,
-- the number of boxes on the pallet is 3.
theorem number_of_boxes_on_pallet : total_weight_of_pallet / weight_of_each_box = 3 :=
by sorry

end number_of_boxes_on_pallet_l32_32047


namespace cost_per_mile_first_plan_l32_32694

theorem cost_per_mile_first_plan 
  (initial_fee : ℝ) (cost_per_mile_first : ℝ) (cost_per_mile_second : ℝ) (miles : ℝ)
  (h_first : initial_fee = 65)
  (h_cost_second : cost_per_mile_second = 0.60)
  (h_miles : miles = 325)
  (h_equal_cost : initial_fee + miles * cost_per_mile_first = miles * cost_per_mile_second) :
  cost_per_mile_first = 0.40 :=
by
  sorry

end cost_per_mile_first_plan_l32_32694


namespace even_func_decreasing_on_neg_interval_l32_32283

variable {f : ℝ → ℝ}

theorem even_func_decreasing_on_neg_interval
  (h_even : ∀ x, f x = f (-x))
  (h_increasing : ∀ (a b : ℝ), 3 ≤ a → a < b → b ≤ 7 → f a < f b)
  (h_min_val : ∀ x, 3 ≤ x → x ≤ 7 → f x ≥ 2) :
  (∀ (a b : ℝ), -7 ≤ a → a < b → b ≤ -3 → f a > f b) ∧ (∀ x, -7 ≤ x → x ≤ -3 → f x ≤ 2) :=
by
  sorry

end even_func_decreasing_on_neg_interval_l32_32283


namespace M_lies_in_third_quadrant_l32_32260

noncomputable def harmonious_point (a b : ℝ) : Prop :=
  3 * a = 2 * b + 5

noncomputable def point_M_harmonious (m : ℝ) : Prop :=
  harmonious_point (m - 1) (3 * m + 2)

theorem M_lies_in_third_quadrant (m : ℝ) (hM : point_M_harmonious m) : 
  (m - 1 < 0 ∧ 3 * m + 2 < 0) :=
by {
  sorry
}

end M_lies_in_third_quadrant_l32_32260


namespace method_of_moments_estimation_l32_32880

noncomputable def gamma_density (α β x : ℝ) (hα : α > -1) (hβ : β > 0) :=
  (1 / (β ^ (α + 1) * Mathlib.Function.Other.gamma (α + 1))) * (x^α) * (Real.exp (-x/ β)) 

def data :=
  [37.5, 62.5, 87.5, 112.5, 137.5, 162.5, 187.5, 250, 350] 

def frequency :=
  [1, 3, 6, 7, 7, 5, 4, 8, 4]

def sample_mean (xs : List ℝ) (ns : List ℝ) :=
  (List.sum (List.zipWith (*) xs ns)) / List.sum ns

def sample_variance (xs : List ℝ) (ns : List ℝ) (mean : ℝ) :=
  (List.sum (List.zipWith (λ x n => n * (x - mean)^2) xs ns)) / (List.sum ns - 1)

theorem method_of_moments_estimation (α β : ℝ) (hα : α > -1) (hβ : β > 0) :
  let mean := sample_mean data frequency
  let variance := sample_variance data frequency mean
  let α_est := mean^2 / variance - 1
  let β_est := variance / mean
  α_est ≈ 3.06 ∧ β_est ≈ 40.86 := by
  let mean := sample_mean data frequency
  let variance := sample_variance data frequency mean
  let α_est := mean^2 / variance - 1
  let β_est := variance / mean
  sorry

end method_of_moments_estimation_l32_32880


namespace physics_marks_l32_32919

theorem physics_marks (P C M : ℕ) 
  (h1 : P + C + M = 180) 
  (h2 : P + M = 180) 
  (h3 : P + C = 140) : 
  P = 140 := 
by 
  sorry

end physics_marks_l32_32919


namespace problem_equivalence_l32_32939

theorem problem_equivalence (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (⌊(a^2 : ℚ) / b⌋ + ⌊(b^2 : ℚ) / a⌋ = ⌊(a^2 + b^2 : ℚ) / (a * b)⌋ + a * b) ↔
  (∃ k : ℕ, k > 0 ∧ ((a = k ∧ b = k^2 + 1) ∨ (a = k^2 + 1 ∧ b = k))) :=
sorry

end problem_equivalence_l32_32939


namespace mark_bought_5_pounds_of_apples_l32_32598

noncomputable def cost_of_tomatoes (pounds_tomatoes : ℕ) (cost_per_pound_tomato : ℝ) : ℝ :=
  pounds_tomatoes * cost_per_pound_tomato

noncomputable def cost_of_apples (total_spent : ℝ) (cost_of_tomatoes : ℝ) : ℝ :=
  total_spent - cost_of_tomatoes

noncomputable def pounds_of_apples (cost_of_apples : ℝ) (cost_per_pound_apples : ℝ) : ℝ :=
  cost_of_apples / cost_per_pound_apples

theorem mark_bought_5_pounds_of_apples (pounds_tomatoes : ℕ) (cost_per_pound_tomato : ℝ) 
  (total_spent : ℝ) (cost_per_pound_apples : ℝ) :
  pounds_tomatoes = 2 →
  cost_per_pound_tomato = 5 →
  total_spent = 40 →
  cost_per_pound_apples = 6 →
  pounds_of_apples (cost_of_apples total_spent (cost_of_tomatoes pounds_tomatoes cost_per_pound_tomato)) cost_per_pound_apples = 5 := by
  intros h1 h2 h3 h4
  sorry

end mark_bought_5_pounds_of_apples_l32_32598


namespace a_range_l32_32572

open Set

variable (A B : Set Real) (a : Real)

def A_def : Set Real := {x | 3 * x + 1 < 4}
def B_def : Set Real := {x | x - a < 0}
def intersection_eq : A ∩ B = A := sorry

theorem a_range : a ≥ 1 :=
  by
  have hA : A = {x | x < 1} := sorry
  have hB : B = {x | x < a} := sorry
  have h_intersection : (A ∩ B) = A := sorry
  sorry

end a_range_l32_32572


namespace courtier_selection_l32_32491

theorem courtier_selection :
  ∀ (dogs : Finset ℕ) (courtiers : ℕ → Finset ℕ),
  dogs.card = 100 ∧ (∀ i, card (courtiers i) = 3) ∧ (∀ i j, i ≠ j → (courtiers i ∩ courtiers j).card = 2) →
  ∃ i j, i ≠ j ∧ courtiers i = courtiers j :=
by
  intros dogs courtiers h
  sorry

end courtier_selection_l32_32491


namespace hunting_dogs_theorem_l32_32498

noncomputable def hunting_dogs_problem : Prop :=
  ∃ (courtiers : Finset (Finset (Fin 100))) (h1 : courtiers.card = 100),
  ∀ (c1 c2 : Finset (Fin 100)), c1 ∈ courtiers → c2 ∈ courtiers → c1 ≠ c2 → (c1 ∩ c2).card ≥ 2 → 
  ∃ (c₁ c₂ : Finset (Fin 100)), c₁ ∈ courtiers ∧ c₂ ∈ courtiers ∧ c₁ = c₂

theorem hunting_dogs_theorem : hunting_dogs_problem :=
sorry

end hunting_dogs_theorem_l32_32498


namespace number_of_markings_l32_32362

def markings (L : ℕ → ℕ) := ∀ n, (n > 0) → L n = L (n - 1) + 1

theorem number_of_markings : ∃ L : ℕ → ℕ, (∀ n, n = 1 → L n = 2) ∧ markings L ∧ L 200 = 201 := 
sorry

end number_of_markings_l32_32362


namespace pieces_eaten_first_night_l32_32223

def initial_candy_debby : ℕ := 32
def initial_candy_sister : ℕ := 42
def candy_after_first_night : ℕ := 39

theorem pieces_eaten_first_night :
  (initial_candy_debby + initial_candy_sister) - candy_after_first_night = 35 := by
  sorry

end pieces_eaten_first_night_l32_32223


namespace price_per_half_pound_of_basil_l32_32867

theorem price_per_half_pound_of_basil
    (cost_per_pound_eggplant : ℝ)
    (pounds_eggplant : ℝ)
    (cost_per_pound_zucchini : ℝ)
    (pounds_zucchini : ℝ)
    (cost_per_pound_tomato : ℝ)
    (pounds_tomato : ℝ)
    (cost_per_pound_onion : ℝ)
    (pounds_onion : ℝ)
    (quarts_ratatouille : ℝ)
    (cost_per_quart : ℝ) :
    pounds_eggplant = 5 → cost_per_pound_eggplant = 2 →
    pounds_zucchini = 4 → cost_per_pound_zucchini = 2 →
    pounds_tomato = 4 → cost_per_pound_tomato = 3.5 →
    pounds_onion = 3 → cost_per_pound_onion = 1 →
    quarts_ratatouille = 4 → cost_per_quart = 10 →
    (cost_per_quart * quarts_ratatouille - 
    (cost_per_pound_eggplant * pounds_eggplant + 
    cost_per_pound_zucchini * pounds_zucchini + 
    cost_per_pound_tomato * pounds_tomato + 
    cost_per_pound_onion * pounds_onion)) / 2 = 2.5 :=
by
    intros h₁ h₂ h₃ h₄ h₅ h₆ h₇ h₈ h₉ h₀
    rw [h₁, h₂, h₃, h₄, h₅, h₆, h₇, h₈, h₉, h₀]
    sorry

end price_per_half_pound_of_basil_l32_32867


namespace answer_l32_32331

def p : Prop := ∃ x > Real.exp 1, (1 / 2)^x > Real.log x
def q : Prop := ∀ a b : Real, a > 1 → b > 1 → Real.log a / Real.log b + 2 * (Real.log b / Real.log a) ≥ 2 * Real.sqrt 2

theorem answer : ¬ p ∧ q :=
by
  have h1 : ¬ p := sorry
  have h2 : q := sorry
  exact ⟨h1, h2⟩

end answer_l32_32331


namespace problem_statement_l32_32163

variables {a b y x : ℝ}

theorem problem_statement :
  (3 * a + 2 * b ≠ 5 * a * b) ∧
  (5 * y - 3 * y ≠ 2) ∧
  (7 * a + a ≠ 7 * a ^ 2) ∧
  (3 * x^2 * y - 2 * y * x^2 = x^2 * y) :=
by
  split
  · intro h
    have h₁ : 3 * a + 2 * b = 3 * a + 2 * b := rfl
    rw h₁ at h
    sorry
  split
  · intro h
    have h₁ : 5 * y - 3 * y = 2 * y := by ring
    rw h₁ at h
    sorry
  split
  · intro h
    have h₁ : 7 * a + a = 8 * a := by ring
    rw h₁ at h
    sorry
  · ring

end problem_statement_l32_32163


namespace initial_bales_l32_32889

theorem initial_bales (B : ℕ) (cond1 : B + 35 = 82) : B = 47 :=
by
  sorry

end initial_bales_l32_32889


namespace a_eq_one_sufficient_not_necessary_P_subset_M_iff_l32_32461

open Set

-- Define sets P and M based on conditions
def P : Set ℝ := {x | x > 2}
def M (a : ℝ) : Set ℝ := {x | x > a}

-- State the theorem
theorem a_eq_one_sufficient_not_necessary (a : ℝ) : (a = 1) → (P ⊆ M a) := 
by
  sorry

theorem P_subset_M_iff (a : ℝ) : (P ⊆ M a) ↔ (a < 2) :=
by
  sorry

end a_eq_one_sufficient_not_necessary_P_subset_M_iff_l32_32461


namespace acute_angle_of_parallelogram_l32_32200

theorem acute_angle_of_parallelogram
  (a b : ℝ) (h : a < b)
  (parallelogram_division : ∀ x y : ℝ, x + y = a → b = x + 2 * Real.sqrt (x * y) + y) :
  ∃ α : ℝ, α = Real.arcsin ((b / a) - 1) :=
sorry

end acute_angle_of_parallelogram_l32_32200


namespace wash_and_dry_time_l32_32798

theorem wash_and_dry_time :
  let whites_wash := 72
  let whites_dry := 50
  let darks_wash := 58
  let darks_dry := 65
  let colors_wash := 45
  let colors_dry := 54
  let total_whites := whites_wash + whites_dry
  let total_darks := darks_wash + darks_dry
  let total_colors := colors_wash + colors_dry
  let total_time := total_whites + total_darks + total_colors
  total_time = 344 :=
by
  unfold total_time
  unfold total_whites
  unfold total_darks
  unfold total_colors
  unfold whites_wash whites_dry darks_wash darks_dry colors_wash colors_dry
  sorry

end wash_and_dry_time_l32_32798


namespace seventy_five_percent_of_number_l32_32860

variable (N : ℝ)

theorem seventy_five_percent_of_number :
  (1 / 8) * (3 / 5) * (4 / 7) * (5 / 11) * N - (1 / 9) * (2 / 3) * (3 / 4) * (5 / 8) * N = 30 →
  0.75 * N = -1476 :=
by
  sorry

end seventy_five_percent_of_number_l32_32860


namespace whale_length_l32_32764

theorem whale_length
  (velocity_fast : ℕ)
  (velocity_slow : ℕ)
  (time : ℕ)
  (h1 : velocity_fast = 18)
  (h2 : velocity_slow = 15)
  (h3 : time = 15) :
  (velocity_fast - velocity_slow) * time = 45 := 
by
  sorry

end whale_length_l32_32764


namespace supplement_of_complement_of_65_l32_32894

def complement (angle : ℝ) : ℝ := 90 - angle
def supplement (angle : ℝ) : ℝ := 180 - angle

theorem supplement_of_complement_of_65 : supplement (complement 65) = 155 :=
by
  -- provide the proof steps here
  sorry

end supplement_of_complement_of_65_l32_32894


namespace two_digit_numbers_solution_l32_32296

theorem two_digit_numbers_solution :
  ∀ (N : ℕ), (∃ (x y : ℕ), (N = 10 * x + y) ∧ (x < 10) ∧ (y < 10) ∧ 4 * x + 2 * y = N / 2) →
    (N = 32 ∨ N = 64 ∨ N = 96) := 
by
  sorry

end two_digit_numbers_solution_l32_32296


namespace sum_f_1_to_2010_eq_zero_l32_32434

theorem sum_f_1_to_2010_eq_zero :
  let f (x : ℕ) := Real.sin (x * Real.pi / 3)
  in (Finset.range 2010).sum (λ x, f (x + 1)) = 0 :=
by
  let f : ℕ → ℝ := λ x, Real.sin (x * Real.pi / 3)
  have periodicity : ∀ n, f (n + 6) = f n := by
    intro n
    simp [f, Real.sin_add, Real.sin_two_pi, Real.cos_two_pi]
  sorry

end sum_f_1_to_2010_eq_zero_l32_32434


namespace remainder_67_pow_67_plus_67_mod_68_l32_32378

theorem remainder_67_pow_67_plus_67_mod_68 :
  (67 ^ 67 + 67) % 68 = 66 :=
by
  -- Skip the proof for now
  sorry

end remainder_67_pow_67_plus_67_mod_68_l32_32378


namespace refrigerator_cost_l32_32731

theorem refrigerator_cost
  (R : ℝ)
  (mobile_phone_cost : ℝ := 8000)
  (loss_percent_refrigerator : ℝ := 0.04)
  (profit_percent_mobile_phone : ℝ := 0.09)
  (overall_profit : ℝ := 120)
  (selling_price_refrigerator : ℝ := 0.96 * R)
  (selling_price_mobile_phone : ℝ := 8720)
  (total_selling_price : ℝ := selling_price_refrigerator + selling_price_mobile_phone)
  (total_cost_price : ℝ := R + mobile_phone_cost)
  (balance_profit_eq : total_selling_price = total_cost_price + overall_profit):
  R = 15000 :=
by
  sorry

end refrigerator_cost_l32_32731


namespace real_roots_quadratic_range_l32_32303

theorem real_roots_quadratic_range (k : ℝ) :
  (∃ x : ℝ, x^2 + 2 * x - k = 0) ↔ k ≥ -1 :=
by
  sorry

end real_roots_quadratic_range_l32_32303


namespace courtiers_dog_selection_l32_32505

theorem courtiers_dog_selection :
  ∃ (c1 c2 : ℕ), c1 ≠ c2 ∧ ∀ d1 d2 d3 : ℕ, 
  (dog_selected_by c1 = {d1, d2, d3} ∧ dog_selected_by c2 = {d1, d2, d3}) :=
begin
  -- Parameters
  let num_courtiers : ℕ := 100,
  let num_dogs : ℕ := 100,
  let picks_per_courtier : finset (fin 100) := {a b c | a, b, c ∈ fin 100},
  
  -- Conditions
  have h1 : ∀ (i j : ℕ) (hic : i ≠ j), 
  (picks_per_courtier i ∩ picks_per_courtier j).card ≥ 2 := sorry,

  -- Proof to show that there exist two courtiers who pick the exact same three dogs
  sorry
end

end courtiers_dog_selection_l32_32505


namespace perfect_square_pairs_l32_32560

theorem perfect_square_pairs (x y : ℕ) (a b : ℤ) :
  (x^2 + 8 * ↑y = a^2 ∧ y^2 - 8 * ↑x = b^2) →
  (∃ n : ℕ, x = n ∧ y = n + 2) ∨ (x = 7 ∧ y = 15) ∨ (x = 33 ∧ y = 17) ∨ (x = 45 ∧ y = 23) :=
by
  sorry

end perfect_square_pairs_l32_32560


namespace exposed_surface_area_equals_42_l32_32662

-- Define the structure and exposed surface area calculations.
def surface_area_of_sculpture (layers : List Nat) : Nat :=
  (layers.headD 0 * 5) +  -- Top layer (5 faces exposed)
  (layers.getD 1 0 * 3 + layers.getD 1 0) +  -- Second layer
  (layers.getD 2 0 * 1 + layers.getD 2 0) +  -- Third layer
  (layers.getD 3 0 * 1) -- Bottom layer

-- Define the conditions
def number_of_layers : List Nat := [1, 4, 9, 6]

-- State the theorem
theorem exposed_surface_area_equals_42 :
  surface_area_of_sculpture number_of_layers = 42 :=
by
  sorry

end exposed_surface_area_equals_42_l32_32662


namespace maximum_number_of_buses_l32_32101

-- Definitions of the conditions
def bus_stops : ℕ := 9 -- There are 9 bus stops in the city.
def max_common_stops : ℕ := 1 -- Any two buses have at most one common stop.
def stops_per_bus : ℕ := 3 -- Each bus stops at exactly three stops.

-- The main theorem statement
theorem maximum_number_of_buses (n : ℕ) :
  (bus_stops = 9) →
  (max_common_stops = 1) →
  (stops_per_bus = 3) →
  n ≤ 12 :=
sorry

end maximum_number_of_buses_l32_32101


namespace sufficient_but_not_necessary_condition_l32_32570

theorem sufficient_but_not_necessary_condition (x : ℝ) (p : Prop) (q : Prop)
  (h₁ : p ↔ (x^2 - 1 > 0)) (h₂ : q ↔ (x < -2)) :
  (¬p → ¬q) ∧ ¬(¬q → ¬p) := 
by
  sorry

end sufficient_but_not_necessary_condition_l32_32570


namespace find_number_of_cows_l32_32307

-- Definitions for the problem
def number_of_legs (cows chickens : ℕ) := 4 * cows + 2 * chickens
def twice_the_heads_plus_12 (cows chickens : ℕ) := 2 * (cows + chickens) + 12

-- Main statement to prove
theorem find_number_of_cows (h : ℕ) : ∃ c : ℕ, number_of_legs c h = twice_the_heads_plus_12 c h ∧ c = 6 := 
by
  -- Sorry is used as a placeholder for the proof
  sorry

end find_number_of_cows_l32_32307


namespace value_of_k_l32_32092

theorem value_of_k (x y k : ℝ) (h1 : 4 * x - 3 * y = k) (h2 : 2 * x + 3 * y = 5) (h3 : x = y) : k = 1 :=
sorry

end value_of_k_l32_32092


namespace lines_are_perpendicular_l32_32617

-- Define the first line equation
def line1 (x y : ℝ) : Prop := x + y - 2 = 0

-- Define the second line equation
def line2 (x y : ℝ) : Prop := x - y + 3 = 0

-- Definition to determine the perpendicularity of two lines
def are_perpendicular (k1 k2 : ℝ) : Prop := k1 * k2 = -1

theorem lines_are_perpendicular :
  are_perpendicular (-1) (1) := 
by
  sorry

end lines_are_perpendicular_l32_32617


namespace value_of_f_at_2_l32_32522

def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem value_of_f_at_2 : f 2 = 3 := by
  -- Definition of the function f.
  -- The goal is to prove that f(2) = 3.
  sorry

end value_of_f_at_2_l32_32522


namespace discount_equivalence_l32_32046

theorem discount_equivalence :
  ∀ (p d1 d2 : ℝ) (d : ℝ),
    p = 800 →
    d1 = 0.15 →
    d2 = 0.10 →
    p * (1 - d1) * (1 - d2) = p * (1 - d) →
    d = 0.235 := by
  intros p d1 d2 d hp hd1 hd2 heq
  sorry

end discount_equivalence_l32_32046


namespace inequality_holds_l32_32864

variable (a b c : ℝ)

theorem inequality_holds : 
  (a * b + b * c + c * a - 1)^2 ≤ (a^2 + 1) * (b^2 + 1) * (c^2 + 1) := 
by 
  sorry

end inequality_holds_l32_32864


namespace pressure_increases_when_block_submerged_l32_32782

theorem pressure_increases_when_block_submerged 
  (P0 : ℝ) (ρ : ℝ) (g : ℝ) (h0 h1 : ℝ) :
  h1 > h0 → 
  (P0 + ρ * g * h1) > (P0 + ρ * g * h0) :=
by
  intros h1_gt_h0
  sorry

end pressure_increases_when_block_submerged_l32_32782


namespace zero_point_in_12_l32_32585

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 3

theorem zero_point_in_12 : ∃ c ∈ Ioo 1 2, f c = 0 :=
by {
  sorry
}

end zero_point_in_12_l32_32585


namespace remainder_of_3_pow_2023_mod_7_l32_32363

theorem remainder_of_3_pow_2023_mod_7 : (3^2023) % 7 = 3 :=
by
  sorry

end remainder_of_3_pow_2023_mod_7_l32_32363


namespace transform_equation_to_square_form_l32_32359

theorem transform_equation_to_square_form : 
  ∀ x : ℝ, (x^2 - 6 * x = 0) → ∃ m n : ℝ, (x + m) ^ 2 = n ∧ m = -3 ∧ n = 9 := 
sorry

end transform_equation_to_square_form_l32_32359


namespace square_roots_sum_eq_zero_l32_32835

theorem square_roots_sum_eq_zero (x y : ℝ) (h1 : x^2 = 2011) (h2 : y^2 = 2011) : x + y = 0 :=
by sorry

end square_roots_sum_eq_zero_l32_32835


namespace problem_solution_l32_32944

def f (x y : ℝ) : ℝ :=
  (x - y) * x * y * (x + y) * (2 * x^2 - 5 * x * y + 2 * y^2)

theorem problem_solution :
  (∀ x y : ℝ, f x y + f y x = 0) ∧
  (∀ x y : ℝ, f x (x + y) + f y (x + y) = 0) :=
by
  sorry

end problem_solution_l32_32944


namespace problem_statement_l32_32951

noncomputable def f (x : ℝ) := Real.log 9 * (Real.log x / Real.log 3)

theorem problem_statement : deriv f 2 + deriv f 2 = 1 := sorry

end problem_statement_l32_32951


namespace base8_to_base10_conversion_l32_32710

theorem base8_to_base10_conversion : 
  (6 * 8^3 + 3 * 8^2 + 7 * 8^1 + 5 * 8^0) = 3325 := 
by 
  sorry

end base8_to_base10_conversion_l32_32710


namespace max_value_7a_9b_l32_32460

theorem max_value_7a_9b 
    (r_1 r_2 r_3 a b : ℝ) 
    (h_eq : ∀ x, x^3 - x^2 + a * x - b = 0 → (x = r_1 ∨ x = r_2 ∨ x = r_3))
    (h_root_sum : r_1 + r_2 + r_3 = 1)
    (h_root_prod : r_1 * r_2 * r_3 = b)
    (h_root_sumprod : r_1 * r_2 + r_2 * r_3 + r_3 * r_1 = a)
    (h_bounds : ∀ i, i = r_1 ∨ i = r_2 ∨ i = r_3 → 0 < i ∧ i < 1) :
        7 * a - 9 * b ≤ 2 := 
sorry

end max_value_7a_9b_l32_32460


namespace power_sum_tenth_l32_32992

theorem power_sum_tenth (a b : ℝ) (h1 : a + b = 1)
    (h2 : a^2 + b^2 = 3)
    (h3 : a^3 + b^3 = 4)
    (h4 : a^4 + b^4 = 7)
    (h5 : a^5 + b^5 = 11) : 
    a^10 + b^10 = 123 := 
sorry

end power_sum_tenth_l32_32992


namespace amount_spent_on_raw_materials_l32_32733

-- Given conditions
def spending_on_machinery : ℝ := 125
def spending_as_cash (total_amount : ℝ) : ℝ := 0.10 * total_amount
def total_amount : ℝ := 250

-- Mathematically equivalent problem
theorem amount_spent_on_raw_materials :
  (X : ℝ) → X + spending_on_machinery + spending_as_cash total_amount = total_amount →
    X = 100 :=
by
  (intro X h)
  sorry

end amount_spent_on_raw_materials_l32_32733


namespace find_other_endpoint_l32_32352

def other_endpoint (midpoint endpoint: ℝ × ℝ) : ℝ × ℝ :=
  let (mx, my) := midpoint
  let (ex, ey) := endpoint
  (2 * mx - ex, 2 * my - ey)

theorem find_other_endpoint :
  other_endpoint (3, 1) (7, -4) = (-1, 6) :=
by
  -- Midpoint formula to find other endpoint
  sorry

end find_other_endpoint_l32_32352


namespace soccer_season_length_l32_32758

def total_games : ℕ := 27
def games_per_month : ℕ := 9
def months_in_season : ℕ := total_games / games_per_month

theorem soccer_season_length : months_in_season = 3 := by
  unfold months_in_season
  unfold total_games
  unfold games_per_month
  sorry

end soccer_season_length_l32_32758


namespace part_one_part_two_l32_32435

noncomputable def f (a x : ℝ) := a * Real.log x - x + 1

theorem part_one (a : ℝ) (h : ∀ x > 0, f a x ≤ 0) : a = 1 := 
sorry

theorem part_two (h₁ : ∀ x > 0, f 1 x ≤ 0) (x : ℝ) (h₂ : 0 < x) (h₃ : x < Real.pi / 2) :
  Real.exp x * Real.sin x - x > f 1 x :=
sorry

end part_one_part_two_l32_32435


namespace seq_eighth_term_l32_32888

-- Define the sequence recursively
def seq : ℕ → ℕ
| 0     => 1  -- Base case, since 1 is the first term of the sequence
| (n+1) => seq n + (n + 1)  -- Recursive case, each term is the previous term plus the index number (which is n + 1) minus 1

-- Define the statement to prove 
theorem seq_eighth_term : seq 7 = 29 :=  -- Note: index 7 corresponds to the 8th term since indexing is 0-based
  by
  sorry

end seq_eighth_term_l32_32888


namespace a3_minus_a2_plus_a1_l32_32230

theorem a3_minus_a2_plus_a1 (a_4 a_3 a_2 a_1 a : ℕ) :
  (a_4 * (1 : ℕ + 1)^4 + a_3 * (1 + 1)^3 + a_2 * (1 + 1)^2 + a_1 * (1 + 1) + a = 1^4) → 
  a_3 - a_2 + a_1 = -14 :=
by
  -- Definitions using provided binomial coefficients
  let a_4 := nat.choose 4 0 -- equal to 1
  let a_3 := -(nat.choose 4 1) -- equal to -4
  let a_2 := nat.choose 4 2 -- equal to 6
  let a_1 := -(nat.choose 4 3) -- equal to -4
  
  -- State the main goal using sorry to serve as the placeholder for the proof
  sorry

end a3_minus_a2_plus_a1_l32_32230


namespace three_digit_multiples_of_seven_l32_32274

theorem three_digit_multiples_of_seven : 
  let min_bound := 100
  let max_bound := 999
  let first_multiple := Nat.ceil (min_bound / 7)
  let last_multiple := Nat.floor (max_bound / 7)
  (last_multiple - first_multiple + 1) = 128 :=
by 
  let min_bound := 100
  let max_bound := 999
  let first_multiple := Nat.ceil (min_bound / 7)
  let last_multiple := Nat.floor (max_bound / 7)
  have calc_first_multiple : first_multiple = 15 := sorry
  have calc_last_multiple : last_multiple = 142 := sorry
  have count_multiples : (last_multiple - first_multiple + 1) = 128 := sorry
  exact count_multiples

end three_digit_multiples_of_seven_l32_32274


namespace toms_dog_age_is_twelve_l32_32762

-- Definitions based on given conditions
def toms_cat_age : ℕ := 8
def toms_rabbit_age : ℕ := toms_cat_age / 2
def toms_dog_age : ℕ := 3 * toms_rabbit_age

-- The statement to be proved
theorem toms_dog_age_is_twelve : toms_dog_age = 12 := by
  sorry

end toms_dog_age_is_twelve_l32_32762


namespace probability_correct_l32_32127

noncomputable def probability_all_players_have_5_after_2023_rings 
    (initial_money : ℕ)
    (num_rings : ℕ) 
    (target_money : ℕ)
    : ℝ := 
    if initial_money = 5 ∧ num_rings = 2023 ∧ target_money = 5 
    then 1 / 4 
    else 0

theorem probability_correct : 
        probability_all_players_have_5_after_2023_rings 5 2023 5 = 1 / 4 := 
by 
    sorry

end probability_correct_l32_32127


namespace compute_expression_l32_32805

theorem compute_expression : (5 + 9)^2 + Real.sqrt (5^2 + 9^2) = 196 + Real.sqrt 106 := 
by sorry

end compute_expression_l32_32805


namespace probability_diff_color_balls_l32_32259

theorem probability_diff_color_balls 
  (Box_A_red : ℕ) (Box_A_black : ℕ) (Box_A_white : ℕ) 
  (Box_B_yellow : ℕ) (Box_B_black : ℕ) (Box_B_white : ℕ) 
  (hA : Box_A_red = 3 ∧ Box_A_black = 3 ∧ Box_A_white = 3)
  (hB : Box_B_yellow = 2 ∧ Box_B_black = 2 ∧ Box_B_white = 2) :
  ((Box_A_red * (Box_B_black + Box_B_white + Box_B_yellow))
  + (Box_A_black * (Box_B_yellow + Box_B_white))
  + (Box_A_white * (Box_B_black + Box_B_yellow))) / 
  ((Box_A_red + Box_A_black + Box_A_white) * 
  (Box_B_yellow + Box_B_black + Box_B_white)) = 7 / 9 := 
by
  sorry

end probability_diff_color_balls_l32_32259


namespace total_spider_legs_l32_32785

-- Definition of the number of spiders
def number_of_spiders : ℕ := 5

-- Definition of the number of legs per spider
def legs_per_spider : ℕ := 8

-- Theorem statement to prove the total number of spider legs
theorem total_spider_legs : number_of_spiders * legs_per_spider = 40 :=
by 
  -- We've planned to use 'sorry' to skip the proof
  sorry

end total_spider_legs_l32_32785


namespace fish_per_bowl_l32_32883

theorem fish_per_bowl : 6003 / 261 = 23 := by
  sorry

end fish_per_bowl_l32_32883


namespace infinite_series_value_l32_32931

noncomputable def infinite_series : ℝ :=
  ∑' n, if n ≥ 2 then (n^4 + 5 * n^2 + 8 * n + 8) / (2^(n + 1) * (n^4 + 4)) else 0

theorem infinite_series_value :
  infinite_series = 3 / 10 :=
by
  sorry

end infinite_series_value_l32_32931


namespace T1_T2_l32_32924

variables {Grip No : Type} [Fintype Grip] [Fintype No]

-- Definitions according to conditions
def P1 (g : Grip) : Finset No := sorry  -- Each grip is a set of nos
def P2 (g1 g2 g3 : Grip) : g1 ≠ g2 → g2 ≠ g3 → g1 ≠ g3 → Finset.inter (Finset.inter (P1 g1) (P1 g2)) (P1 g3) = sorry -- One common no for any three distinct grips
def P3 (n : No) : Fintype {g : Grip // n ∈ P1 g} := sorry  -- Each no belongs to at least two grips
def P4 : Fintype.card Grip = 5 := sorry  -- There is a total of five grips
def P5 (g : Grip) : Fintype.card (P1 g) = 3 := sorry  -- There are exactly three nos in each grip

-- Theorems to be proved
theorem T1 : Fintype.card No = 10 :=
  by sorry

theorem T2 (n : No) : Fintype.card {g : Grip // n ∈ P1 g} = 3 :=
  by sorry

end T1_T2_l32_32924


namespace cylinder_height_l32_32201

theorem cylinder_height (OA OB : ℝ) (h_OA : OA = 7) (h_OB : OB = 2) :
  ∃ (h_cylinder : ℝ), h_cylinder = 3 * Real.sqrt 5 :=
by
  use (Real.sqrt (OA^2 - OB^2))
  rw [h_OA, h_OB]
  norm_num
  sorry

end cylinder_height_l32_32201


namespace pam_bags_equiv_gerald_bags_l32_32341

theorem pam_bags_equiv_gerald_bags :
  ∀ (total_apples pam_bags apples_per_gerald_bag : ℕ), 
    total_apples = 1200 ∧ pam_bags = 10 ∧ apples_per_gerald_bag = 40 → 
    (total_apples / pam_bags) / apples_per_gerald_bag = 3 :=
by
  intros total_apples pam_bags apples_per_gerald_bag h
  obtain ⟨ht, hp, hg⟩ : total_apples = 1200 ∧ pam_bags = 10 ∧ apples_per_gerald_bag = 40 := h
  sorry

end pam_bags_equiv_gerald_bags_l32_32341


namespace middle_card_is_five_l32_32150

section card_numbers

variables {a b c : ℕ}

-- Conditions
def distinct (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c
def sum_fifteen (a b c : ℕ) : Prop := a + b + c = 15
def sum_two_smallest_less_than_ten (a b : ℕ) : Prop := a + b < 10
def ascending_order (a b c : ℕ) : Prop := a < b ∧ b < c 

-- Main theorem statement
theorem middle_card_is_five 
  (h1 : distinct a b c)
  (h2 : sum_fifteen a b c)
  (h3 : sum_two_smallest_less_than_ten a b) 
  (h4 : ascending_order a b c)
  (h5 : ∀ x, (x = a → (∃ b c, sum_fifteen x b c ∧ distinct x b c ∧ sum_two_smallest_less_than_ten x b ∧ ascending_order x b c ∧ ¬ (b = 5 ∧ c = 10))) →
           (∃ b c, sum_fifteen x b c ∧ distinct x b c ∧ sum_two_smallest_less_than_ten b c ∧ ascending_order x b c ∧ ¬ (b = 2 ∧ c = 7)))
  (h6 : ∀ x, (x = c → (∃ a b, sum_fifteen a b x ∧ distinct a b x ∧ sum_two_smallest_less_than_ten a b ∧ ascending_order a b x ∧ ¬ (a = 1 ∧ b = 4))) →
           (∃ a b, sum_fifteen a b x ∧ distinct a b x ∧ sum_two_smallest_less_than_ten a b ∧ ascending_order a b x ∧ ¬ (a = 2 ∧ b = 6)))
  (h7 : ∀ x, (x = b → (∃ a c, sum_fifteen a x c ∧ distinct a x c ∧ sum_two_smallest_less_than_ten a c ∧ ascending_order a x c ∧ ¬ (a = 1 ∧ c = 9 ∨ a = 2 ∧ c = 8))) →
           (∃ a c, sum_fifteen a x c ∧ distinct a x c ∧ sum_two_smallest_less_than_ten a c ∧ ascending_order a x c ∧ ¬ (a = 1 ∧ c = 6 ∨ a = 2 ∧ c = 5)))
  : b = 5 := sorry

end card_numbers

end middle_card_is_five_l32_32150


namespace sum_of_real_solutions_l32_32945

theorem sum_of_real_solutions :
  let eq := ∀ (x : ℝ), (x-3)/(x^2 + 5*x + 2) = (x-6)/(x^2 - 11*x)
  ∑ (r : ℝ) in { x : ℝ | eq x }, r = 62/13 :=
by { sorry }

end sum_of_real_solutions_l32_32945


namespace largest_of_four_l32_32206

theorem largest_of_four : 
  let a := 1 
  let b := 0 
  let c := |(-2)| 
  let d := -3 
  max (max (max a b) c) d = c := by
  sorry

end largest_of_four_l32_32206


namespace cube_volume_l32_32194

-- Define the given condition: The surface area of the cube
def surface_area (A : ℕ) := A = 294

-- The key proposition we need to prove using the given condition
theorem cube_volume (s : ℕ) (A : ℕ) (V : ℕ) 
  (area_condition : surface_area A)
  (side_length_condition : s ^ 2 = A / 6) 
  (volume_condition : V = s ^ 3) : 
  V = 343 := by
  sorry

end cube_volume_l32_32194


namespace percent_of_a_is_4b_l32_32344

theorem percent_of_a_is_4b (a b : ℝ) (h : a = 1.2 * b) : (4 * b / a) * 100 = 333.33 := by
  sorry

end percent_of_a_is_4b_l32_32344


namespace option_a_option_b_option_c_option_d_l32_32900

open Real

theorem option_a (x : ℝ) (h1 : 0 < x) (h2 : x < π) : x > sin x :=
sorry

theorem option_b (x : ℝ) (h : 0 < x) : ¬ (1 - (1 / x) > log x) :=
sorry

theorem option_c (x : ℝ) : (x + 1) * exp x >= -1 / (exp 2) :=
sorry

theorem option_d : ¬ (∀ x : ℝ, x^2 > - (1 / x)) :=
sorry

end option_a_option_b_option_c_option_d_l32_32900


namespace floor_width_is_120_l32_32126

def tile_length := 25 -- cm
def tile_width := 16 -- cm
def floor_length := 180 -- cm
def max_tiles := 54

theorem floor_width_is_120 :
  ∃ (W : ℝ), W = 120 ∧ (floor_length / tile_width) * W = max_tiles * (tile_length * tile_width) := 
sorry

end floor_width_is_120_l32_32126


namespace fraction_to_terminanting_decimal_l32_32936

theorem fraction_to_terminanting_decimal : (47 / (5^4 * 2) : ℚ) = 0.0376 := 
by 
  sorry

end fraction_to_terminanting_decimal_l32_32936


namespace expected_value_is_6_5_l32_32895

noncomputable def expected_value_12_sided_die : ℚ :=
  (1 / 12) * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12)

theorem expected_value_is_6_5 : expected_value_12_sided_die = 6.5 := 
by
  sorry

end expected_value_is_6_5_l32_32895


namespace minimum_value_a_plus_4b_l32_32268

theorem minimum_value_a_plus_4b (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : (1 / a) + (1 / b) = 1) : a + 4 * b ≥ 9 :=
sorry

end minimum_value_a_plus_4b_l32_32268


namespace find_a_value_l32_32014

noncomputable def collinear (points : List (ℚ × ℚ)) := 
  ∃ a b c, ∀ (x y : ℚ), (x, y) ∈ points → a * x + b * y + c = 0

theorem find_a_value (a : ℚ) :
  collinear [(3, -5), (-a + 2, 3), (2*a + 3, 2)] → a = -7 / 23 :=
by
  sorry

end find_a_value_l32_32014


namespace product_of_consecutive_integers_l32_32620

theorem product_of_consecutive_integers (n : ℤ) :
  n * (n + 1) * (n + 2) = (n + 1)^3 - (n + 1) :=
by
  sorry

end product_of_consecutive_integers_l32_32620


namespace value_to_add_l32_32743

theorem value_to_add (a b c d : ℕ) (x : ℕ) 
  (h1 : a = 24) (h2 : b = 32) (h3 : c = 36) (h4 : d = 54)
  (h_lcm : Nat.lcm (Nat.lcm a b) (Nat.lcm c d) = 864) :
  x + 857 = 864 → x = 7 := 
by
  intros h
  rw [h]
  exact rfl

end value_to_add_l32_32743


namespace container_dimensions_l32_32587

theorem container_dimensions (a b c : ℝ) 
  (h1 : a * b * 16 = 2400)
  (h2 : a * c * 10 = 2400)
  (h3 : b * c * 9.6 = 2400) :
  a = 12 ∧ b = 12.5 ∧ c = 20 :=
by
  sorry

end container_dimensions_l32_32587


namespace negation_of_p_l32_32579

-- Define the proposition p: ∀ x ∈ ℝ, sin x ≤ 1
def proposition_p : Prop := ∀ x : ℝ, Real.sin x ≤ 1

-- The statement to prove the negation of proposition p
theorem negation_of_p : ¬proposition_p ↔ ∃ x : ℝ, Real.sin x > 1 := by
  sorry

end negation_of_p_l32_32579


namespace emma_age_when_sister_is_56_l32_32676

theorem emma_age_when_sister_is_56 :
  ∀ (emma_age sister_age_difference sister_future_age : ℕ),
  emma_age = 7 →
  sister_age_difference = 9 →
  sister_future_age = 56 →
  emma_age + (sister_future_age - (emma_age + sister_age_difference)) = 47 :=
by
  intros emma_age sister_age_difference sister_future_age hEmma hSisterDiff hSisterFuture
  rw [hEmma, hSisterDiff, hSisterFuture]
  norm_num
  sorry

end emma_age_when_sister_is_56_l32_32676


namespace find_b_value_l32_32947

theorem find_b_value (x y b : ℝ) (h1 : (7 * x + b * y) / (x - 2 * y) = 29) (h2 : x / (2 * y) = 3 / 2) : b = 8 :=
by
  sorry

end find_b_value_l32_32947


namespace supplement_of_complement_65_degrees_l32_32892

def complement (α : ℝ) : ℝ := 90 - α
def supplement (α : ℝ) : ℝ := 180 - α

theorem supplement_of_complement_65_degrees : 
  supplement (complement 65) = 155 :=
by
  sorry

end supplement_of_complement_65_degrees_l32_32892


namespace stratified_sampling_l32_32053

-- Definition of conditions as hypothesis
def total_employees : ℕ := 100
def under_35 : ℕ := 45
def between_35_49 : ℕ := 25
def over_50 : ℕ := total_employees - under_35 - between_35_49
def sample_size : ℕ := 20
def sampling_ratio : ℚ := sample_size / total_employees

-- The target number of people from each group
def under_35_sample : ℚ := sampling_ratio * under_35
def between_35_49_sample : ℚ := sampling_ratio * between_35_49
def over_50_sample : ℚ := sampling_ratio * over_50

-- Problem statement
theorem stratified_sampling : 
  under_35_sample = 9 ∧ 
  between_35_49_sample = 5 ∧ 
  over_50_sample = 6 :=
  by
  sorry

end stratified_sampling_l32_32053


namespace isosceles_triangle_perimeter_l32_32236

-- Define a structure to represent a triangle with sides a, b, c
structure Triangle :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (triangle_ineq_1 : a + b > c)
  (triangle_ineq_2 : a + c > b)
  (triangle_ineq_3 : b + c > a)

-- Define the specific triangle given the condition
def isosceles_triangle_with_sides (s1 s2 : ℝ) (h_iso : s1 = 3 ∨ s2 = 3) (h_ineq : s1 = 6 ∨ s2 = 6) : Triangle :=
  if h_iso then
    { a := 3, b := 3, c := 6,
      triangle_ineq_1 := by linarith,
      triangle_ineq_2 := by linarith,
      triangle_ineq_3 := by linarith }
  else 
    sorry -- We cover the second case directly with the checked option

-- Prove that the perimeter of the given isosceles triangle is as expected
theorem isosceles_triangle_perimeter :
  let t := isosceles_triangle_with_sides 3 6 (or.inl rfl) (or.inr rfl) in
  t.a + t.b + t.c = 15 :=
by simp [isosceles_triangle_with_sides, add_assoc]

end isosceles_triangle_perimeter_l32_32236


namespace apples_weight_l32_32793

theorem apples_weight (x : ℝ) (price1 : ℝ) (price2 : ℝ) (new_price_diff : ℝ) (total_revenue : ℝ)
  (h1 : price1 * x = 228)
  (h2 : price2 * (x + 5) = 180)
  (h3 : ∀ kg: ℝ, kg * (price1 - new_price_diff) = total_revenue)
  (h4 : new_price_diff = 0.9)
  (h5 : total_revenue = 408) :
  2 * x + 5 = 85 :=
by
  sorry

end apples_weight_l32_32793


namespace f_x_neg_l32_32232

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 + 1 else -x^2 - 1

theorem f_x_neg (x : ℝ) (h : x < 0) : f x = -x^2 - 1 :=
by
  sorry

end f_x_neg_l32_32232


namespace find_sums_of_integers_l32_32879

theorem find_sums_of_integers (x y : ℤ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_prod_sum : x * y + x + y = 125) (h_rel_prime : Int.gcd x y = 1) (h_lt_x : x < 30) (h_lt_y : y < 30) : 
  (x + y = 25) ∨ (x + y = 23) ∨ (x + y = 21) := 
by 
  sorry

end find_sums_of_integers_l32_32879


namespace smallest_possible_b_l32_32459

theorem smallest_possible_b (a b c : ℚ) (h1 : a < b) (h2 : b < c)
    (arithmetic_seq : 2 * b = a + c) (geometric_seq : c^2 = a * b) :
    b = 1 / 2 :=
by
  let a := 4 * b
  let c := 2 * b - a
  -- rewrite and derived equations will be done in the proof
  sorry

end smallest_possible_b_l32_32459


namespace tan_beta_is_neg3_l32_32959

theorem tan_beta_is_neg3 (α β : ℝ) (h1 : Real.tan α = -2) (h2 : Real.tan (α + β) = 1) : Real.tan β = -3 := 
sorry

end tan_beta_is_neg3_l32_32959


namespace intersection_of_M_and_N_is_12_l32_32580

def M : Set ℤ := {x | -1 ≤ x ∧ x < 3}
def N : Set ℤ := {1, 2, 3}

theorem intersection_of_M_and_N_is_12 : M ∩ N = {1, 2} :=
by
  sorry

end intersection_of_M_and_N_is_12_l32_32580


namespace sin_1320_eq_neg_sqrt_3_div_2_l32_32754

theorem sin_1320_eq_neg_sqrt_3_div_2 : Real.sin (1320 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_1320_eq_neg_sqrt_3_div_2_l32_32754


namespace central_angle_of_region_l32_32390

theorem central_angle_of_region (A : ℝ) (θ : ℝ) (h : (1:ℝ) / 8 = (θ / 360) * A / A) : θ = 45 :=
by
  sorry

end central_angle_of_region_l32_32390


namespace find_angle4_l32_32527

noncomputable def angle_1 := 70
noncomputable def angle_2 := 110
noncomputable def angle_3 := 35
noncomputable def angle_4 := 35

theorem find_angle4 (h1 : angle_1 + angle_2 = 180) (h2 : angle_3 = angle_4) :
  angle_4 = 35 :=
by
  have h3: angle_1 + 70 + 40 = 180 := by sorry
  have h4: angle_2 + angle_3 + angle_4 = 180 := by sorry
  sorry

end find_angle4_l32_32527


namespace cubic_identity_l32_32253

theorem cubic_identity (x : ℝ) (h : x - (1 / x) = 5) : x^3 - (1 / x^3) = 140 := 
  sorry

end cubic_identity_l32_32253


namespace max_min_x1_x2_squared_l32_32433

theorem max_min_x1_x2_squared (k : ℝ) (x1 x2 : ℝ) 
  (h1 : x1^2 - (k-2)*x1 + (k^2 + 3*k + 5) = 0)
  (h2 : x2^2 - (k-2)*x2 + (k^2 + 3*k + 5) = 0)
  (h3 : -4 ≤ k ∧ k ≤ -4/3) : 
  (∃ (k_max k_min : ℝ), 
    k = -4 → x1^2 + x2^2 = 18 ∧ k = -4/3 → x1^2 + x2^2 = 50/9) :=
sorry

end max_min_x1_x2_squared_l32_32433


namespace halfway_between_fractions_l32_32070

theorem halfway_between_fractions : ( (1/8 : ℚ) + (1/3 : ℚ) ) / 2 = 11 / 48 :=
by
  sorry

end halfway_between_fractions_l32_32070


namespace courtiers_dog_selection_l32_32506

theorem courtiers_dog_selection :
  ∃ (c1 c2 : ℕ), c1 ≠ c2 ∧ ∀ d1 d2 d3 : ℕ, 
  (dog_selected_by c1 = {d1, d2, d3} ∧ dog_selected_by c2 = {d1, d2, d3}) :=
begin
  -- Parameters
  let num_courtiers : ℕ := 100,
  let num_dogs : ℕ := 100,
  let picks_per_courtier : finset (fin 100) := {a b c | a, b, c ∈ fin 100},
  
  -- Conditions
  have h1 : ∀ (i j : ℕ) (hic : i ≠ j), 
  (picks_per_courtier i ∩ picks_per_courtier j).card ≥ 2 := sorry,

  -- Proof to show that there exist two courtiers who pick the exact same three dogs
  sorry
end

end courtiers_dog_selection_l32_32506


namespace common_chord_length_l32_32690

-- Circles: x^2 + y^2 - 2x + 10y - 24 = 0, and x^2 + y^2 + 2x + 2y - 8 = 0
noncomputable def circle1 : set (ℝ × ℝ) :=
  { p | ∃ (x y : ℝ), x^2 + y^2 - 2*x + 10*y - 24 = 0 }

noncomputable def circle2 : set (ℝ × ℝ) :=
  { p | ∃ (x y : ℝ), x^2 + y^2 + 2*x + 2*y - 8 = 0 }

-- Prove the length of the common chord
theorem common_chord_length : 
  ∃ l : ℝ, l = 2*Real.sqrt(5) ∧
  ∃ p1 p2, p1 ∈ circle1 ∧ p1 ∈ circle2 ∧ 
           p2 ∈ circle1 ∧ p2 ∈ circle2 ∧ 
           dist p1 p2 = l :=
begin
  sorry
end

end common_chord_length_l32_32690


namespace sector_area_l32_32837

theorem sector_area (θ r a : ℝ) (hθ : θ = 2) (haarclength : r * θ = 4) : 
  (1/2) * r * r * θ = 4 :=
by {
  -- Proof goes here
  sorry
}

end sector_area_l32_32837


namespace angles_sum_l32_32863

def points_on_circle (A B C R S O : Type) : Prop := sorry

def arc_measure (B R S : Type) (m1 m2 : ℝ) : Prop := sorry

def angle_T (A C B S : Type) (T : ℝ) : Prop := sorry

def angle_U (O C B S : Type) (U : ℝ) : Prop := sorry

theorem angles_sum
  (A B C R S O : Type)
  (h1 : points_on_circle A B C R S O)
  (h2 : arc_measure B R S 48 54)
  (h3 : angle_T A C B S 78)
  (h4 : angle_U O C B S 27) :
  78 + 27 = 105 :=
by sorry

end angles_sum_l32_32863


namespace min_value_expression_l32_32328

open Real

theorem min_value_expression (α β : ℝ) :
  (3 * cos α + 4 * sin β - 7)^2 + (3 * sin α + 4 * cos β - 12)^2 ≥ 36 := by
  sorry

end min_value_expression_l32_32328


namespace correct_average_after_error_l32_32873

theorem correct_average_after_error (n : ℕ) (a m_wrong m_correct : ℤ) 
  (h_n : n = 30) (h_a : a = 60) (h_m_wrong : m_wrong = 90) (h_m_correct : m_correct = 15) : 
  ((n * a + (m_correct - m_wrong)) / n : ℤ) = 57 := 
by
  sorry

end correct_average_after_error_l32_32873


namespace find_n_l32_32293

theorem find_n (N : ℕ) (hN : 10 ≤ N ∧ N < 100)
  (h : ∃ a b : ℕ, N = 10 * a + b ∧ 4 * a + 2 * b = N / 2) : 
  N = 32 ∨ N = 64 ∨ N = 96 :=
sorry

end find_n_l32_32293


namespace circle_equation_has_valid_k_l32_32968

theorem circle_equation_has_valid_k (k : ℝ) : (∃ a b r : ℝ, r > 0 ∧ ∀ x y : ℝ, (x - a) ^ 2 + (y - b) ^ 2 = r ^ 2) ↔ k < 5 / 4 := by
  sorry

end circle_equation_has_valid_k_l32_32968


namespace max_car_passing_400_l32_32993

noncomputable def max_cars_passing (speed : ℕ) (car_length : ℤ) (hour : ℕ) : ℕ :=
  20000 * speed / (5 * (speed + 1))

theorem max_car_passing_400 :
  max_cars_passing 20 5 1 / 10 = 400 := by
  sorry

end max_car_passing_400_l32_32993


namespace probability_person_A_three_consecutive_days_l32_32075

noncomputable def charity_event_probability : ℚ :=
  let total_scenarios :=
    Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 3)
  let favorable_scenarios := 4
  favorable_scenarios / total_scenarios

theorem probability_person_A_three_consecutive_days :
  charity_event_probability = 1/5 :=
by
  sorry

end probability_person_A_three_consecutive_days_l32_32075


namespace grandfather_age_5_years_back_l32_32393

variable (F S G : ℕ)

-- Conditions
def father_age : Prop := F = 58
def son_current_age : Prop := S = 58 - S
def son_grandfather_age_relation : Prop := S - 5 = 1 / 2 * (G - 5)

-- Theorem: Prove the grandfather's age 5 years back given the conditions.
theorem grandfather_age_5_years_back (h1 : father_age F) (h2 : son_current_age S) (h3 : son_grandfather_age_relation S G) : G = 2 * S - 5 :=
sorry

end grandfather_age_5_years_back_l32_32393


namespace average_of_t_b_c_29_l32_32381
-- Importing the entire Mathlib library

theorem average_of_t_b_c_29 (t b c : ℝ) 
  (h : (t + b + c + 14 + 15) / 5 = 12) : 
  (t + b + c + 29) / 4 = 15 :=
by 
  sorry

end average_of_t_b_c_29_l32_32381


namespace fruit_weights_assigned_l32_32121

def weights := {100, 150, 170, 200, 280}

variables (mandarin apple banana peach orange : ℕ)

axiom peach_lighter_than_orange : peach < orange
axiom banana_between_apple_and_peach : apple < banana ∧ banana < peach
axiom mandarin_lighter_than_banana : mandarin < banana
axiom apple_banana_heavier_than_orange : apple + banana > orange
axiom weights_assignment : {mandarin, apple, banana, peach, orange} = weights

theorem fruit_weights_assigned :
  (mandarin = 100 ∧ 
   apple = 150 ∧ 
   banana = 170 ∧ 
   peach = 200 ∧ 
   orange = 280) :=
sorry

end fruit_weights_assigned_l32_32121


namespace find_a_l32_32596

def A : Set ℝ := {-1, 0, 1}
noncomputable def B (a : ℝ) : Set ℝ := {a + 1, 2 * a}

theorem find_a (a : ℝ) : (A ∩ B a = {0}) → a = -1 := by
  sorry

end find_a_l32_32596


namespace arrange_numbers_l32_32403

theorem arrange_numbers :
  (2 : ℝ) ^ 1000 < (5 : ℝ) ^ 500 ∧ (5 : ℝ) ^ 500 < (3 : ℝ) ^ 750 :=
by
  sorry

end arrange_numbers_l32_32403


namespace empty_subset_singleton_zero_l32_32783

theorem empty_subset_singleton_zero : ∅ ⊆ ({0} : Set ℕ) :=
by
  sorry

end empty_subset_singleton_zero_l32_32783


namespace spinner_prob_l32_32169

theorem spinner_prob:
  let sections := 4
  let prob := 1 / sections
  let prob_not_e := 1 - prob
  (prob_not_e * prob_not_e) = 9 / 16 :=
by
  sorry

end spinner_prob_l32_32169


namespace attention_index_proof_l32_32918

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 
  if 0 ≤ x ∧ x ≤ 10 then 100 * a ^ (x / 10) - 60
  else if 10 < x ∧ x ≤ 20 then 340
  else if 20 < x ∧ x ≤ 40 then 640 - 15 * x
  else 0

theorem attention_index_proof (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f 5 a = 140) :
  a = 4 ∧ f 5 4 > f 35 4 ∧ (5 ≤ (x : ℝ) ∧ x ≤ 100 / 3 → f x 4 ≥ 140) :=
by
  sorry

end attention_index_proof_l32_32918


namespace isosceles_triangle_perimeter_l32_32237

-- Defining the given conditions
def is_isosceles (a b c : ℕ) : Prop := a = b ∨ b = c ∨ c = a
def triangle (a b c : ℕ) : Prop := a + b > c ∧ b + c > a ∧ c + a > b

-- Stating the problem and goal
theorem isosceles_triangle_perimeter (a b c : ℕ) 
  (h_iso: is_isosceles a b c)
  (h_len1: a = 3 ∨ a = 6)
  (h_len2: b = 3 ∨ b = 6)
  (h_triangle: triangle a b c): a + b + c = 15 :=
sorry

end isosceles_triangle_perimeter_l32_32237


namespace root_line_discriminant_curve_intersection_l32_32998

theorem root_line_discriminant_curve_intersection (a p q : ℝ) :
  (4 * p^3 + 27 * q^2 = 0) ∧ (ap + q + a^3 = 0) →
  (a = 0 ∧ ∀ p q, 4 * p^3 + 27 * q^2 = 0 → ap + q + a^3 = 0 → (p = 0 ∧ q = 0)) ∨
  (a ≠ 0 ∧ (∃ p1 q1 p2 q2, 
             4 * p1^3 + 27 * q1^2 = 0 ∧ ap + q1 + a^3 = 0 ∧ 
             4 * p2^3 + 27 * q2^2 = 0 ∧ ap + q2 + a^3 = 0 ∧ 
             (p1, q1) ≠ (p2, q2))) := 
sorry

end root_line_discriminant_curve_intersection_l32_32998


namespace count_three_digit_multiples_of_seven_l32_32273

theorem count_three_digit_multiples_of_seven :
  let a := 100 in
  let b := 999 in
  let smallest := (Nat.ceil (a.toRat / 7)).natAbs * 7 in
  let largest := (b / 7) * 7 in
  (largest / 7) - ((smallest - 1) / 7) = 128 := sorry

end count_three_digit_multiples_of_seven_l32_32273


namespace compute_b_l32_32956

noncomputable def polynomial := Polynomial ℚ

theorem compute_b (a b : ℚ) : 
  (polynomial.X^3 + a * polynomial.X^2 + b * polynomial.X + 6).isRoot (1 + Real.sqrt 2) → 
  b = 11 := 
by
  sorry

end compute_b_l32_32956


namespace solve_for_n_l32_32131

theorem solve_for_n (n : ℕ) (h : (8 ^ n) * (8 ^ n) * (8 ^ n) = 64 ^ 3) : n = 2 :=
by sorry

end solve_for_n_l32_32131


namespace ordering_of_variables_l32_32426

noncomputable def f (x : ℝ) : ℝ := x - Real.log x

theorem ordering_of_variables 
  (a b c : ℝ)
  (ha : a - 2 = Real.log (a / 2))
  (hb : b - 3 = Real.log (b / 3))
  (hc : c - 3 = Real.log (c / 2))
  (ha_pos : 0 < a) (ha_lt_one : a < 1)
  (hb_pos : 0 < b) (hb_lt_one : b < 1)
  (hc_pos : 0 < c) (hc_lt_one : c < 1) :
  c < b ∧ b < a :=
sorry

end ordering_of_variables_l32_32426


namespace binom_expansion_l32_32264

/-- Given the binomial expansion of (sqrt(x) + 3x)^n for n < 15, 
    with the binomial coefficients of the 9th, 10th, and 11th terms forming an arithmetic sequence,
    we conclude that n must be 14 and describe all the rational terms in the expansion.
-/
theorem binom_expansion (n : ℕ) (h : n < 15)
  (h_seq : Nat.choose n 8 + Nat.choose n 10 = 2 * Nat.choose n 9) :
  n = 14 ∧
  (∃ (t1 t2 t3 : ℕ), 
    (t1 = 1 ∧ (Nat.choose 14 0 : ℕ) * (x ^ 7 : ℤ) = x ^ 7) ∧
    (t2 = 164 ∧ (Nat.choose 14 6 : ℕ) * (x ^ 6 : ℤ) = 164 * x ^ 6) ∧
    (t3 = 91 ∧ (Nat.choose 14 12 : ℕ) * (x ^ 5 : ℤ) = 91 * x ^ 5)) := 
  sorry

end binom_expansion_l32_32264


namespace ratio_of_arithmetic_seqs_l32_32987

noncomputable def arithmetic_seq_sum (a_1 a_n : ℕ) (n : ℕ) : ℝ := (n * (a_1 + a_n)) / 2

theorem ratio_of_arithmetic_seqs (a_1 a_6 a_11 b_1 b_6 b_11 : ℕ) :
  (∀ n : ℕ, (arithmetic_seq_sum a_1 a_n n) / (arithmetic_seq_sum b_1 b_n n) = n / (2 * n + 1))
  → (a_1 + a_6) / (b_1 + b_6) = 6 / 13
  → (a_1 + a_11) / (b_1 + b_11) = 11 / 23
  → (a_6 : ℝ) / (b_6 : ℝ) = 11 / 23 :=
  by
    intros h₁₁ h₆ h₁₁b
    sorry

end ratio_of_arithmetic_seqs_l32_32987


namespace J_3_3_4_l32_32225

def J (a b c : ℚ) : ℚ := a / b + b / c + c / a

theorem J_3_3_4 : J 3 (3 / 4) 4 = 259 / 48 := 
by {
    -- We would normally include proof steps here, but according to the instruction, we use 'sorry'.
    sorry
}

end J_3_3_4_l32_32225


namespace drawing_time_total_l32_32209

theorem drawing_time_total
  (bianca_school : ℕ)
  (bianca_home : ℕ)
  (lucas_school : ℕ)
  (lucas_home : ℕ)
  (h_bianca_school : bianca_school = 22)
  (h_bianca_home : bianca_home = 19)
  (h_lucas_school : lucas_school = 10)
  (h_lucas_home : lucas_home = 35) :
  bianca_school + bianca_home + lucas_school + lucas_home = 86 := 
by
  -- Proof would go here
  sorry

end drawing_time_total_l32_32209


namespace back_seat_tickets_sold_l32_32184

variable (M B : ℕ)

theorem back_seat_tickets_sold:
  M + B = 20000 ∧ 55 * M + 45 * B = 955000 → B = 14500 :=
by
  sorry

end back_seat_tickets_sold_l32_32184


namespace train_length_l32_32794

noncomputable def speed_kmph := 90
noncomputable def time_sec := 5
noncomputable def speed_mps := speed_kmph * 1000 / 3600

theorem train_length : (speed_mps * time_sec) = 125 := by
  -- We need to assert and prove this theorem
  sorry

end train_length_l32_32794


namespace unique_real_value_for_equal_roots_l32_32411

-- Definitions of conditions
def quadratic_eq (p : ℝ) : Prop := 
  ∀ x : ℝ, x^2 - (p + 1) * x + p = 0

-- Statement of the problem
theorem unique_real_value_for_equal_roots :
  ∃! p : ℝ, ∀ x y : ℝ, (x^2 - (p+1)*x + p = 0) ∧ (y^2 - (p+1)*y + p = 0) → x = y := 
sorry

end unique_real_value_for_equal_roots_l32_32411


namespace cube_difference_l32_32245

variable (x : ℝ)

theorem cube_difference (h : x - 1/x = 5) : x^3 - 1/x^3 = 120 := by
  sorry

end cube_difference_l32_32245


namespace total_packs_sold_l32_32031

def packs_sold_village_1 : ℕ := 23
def packs_sold_village_2 : ℕ := 28

theorem total_packs_sold : packs_sold_village_1 + packs_sold_village_2 = 51 :=
by
  -- We acknowledge the correctness of the calculation.
  sorry

end total_packs_sold_l32_32031


namespace problem_statement_l32_32442

-- Define complex number i
noncomputable def i : ℂ := Complex.I

-- Define x as per the problem statement
noncomputable def x : ℂ := (1 + i * Real.sqrt 3) / 2

-- The main proposition to prove
theorem problem_statement : (1 / (x^2 - x)) = -1 :=
  sorry

end problem_statement_l32_32442


namespace angle_A_and_shape_of_triangle_l32_32312

theorem angle_A_and_shape_of_triangle 
  (a b c : ℝ)
  (h1 : a^2 - c^2 = a * c - b * c)
  (h2 : ∃ r : ℝ, a = b * r ∧ c = b / r)
  (h3 : ∃ B C : Type, B = A ∧ C ≠ A ) :
  ∃ (A : ℝ), A = 60 ∧ a = b ∧ b = c := 
sorry

end angle_A_and_shape_of_triangle_l32_32312


namespace marbles_total_l32_32985

def marbles_initial := 22
def marbles_given := 20

theorem marbles_total : marbles_initial + marbles_given = 42 := by
  sorry

end marbles_total_l32_32985


namespace therapist_charge_difference_l32_32044

theorem therapist_charge_difference :
  ∃ F A : ℝ, F + 4 * A = 350 ∧ F + A = 161 ∧ F - A = 35 :=
by {
  -- Placeholder for the actual proof.
  sorry
}

end therapist_charge_difference_l32_32044


namespace problem_solution_l32_32085

open Real

theorem problem_solution :
  (∃ x₀ : ℝ, log x₀ ≥ x₀ - 1) ∧ (¬ ∀ θ : ℝ, sin θ + cos θ < 1) :=
by
  sorry

end problem_solution_l32_32085


namespace bad_carrots_count_l32_32639

def total_carrots (vanessa_carrots : ℕ) (mother_carrots : ℕ) : ℕ := 
vanessa_carrots + mother_carrots

def bad_carrots (total_carrots : ℕ) (good_carrots : ℕ) : ℕ := 
total_carrots - good_carrots

theorem bad_carrots_count : 
  ∀ (vanessa_carrots mother_carrots good_carrots : ℕ), 
  vanessa_carrots = 17 → 
  mother_carrots = 14 → 
  good_carrots = 24 → 
  bad_carrots (total_carrots vanessa_carrots mother_carrots) good_carrots = 7 := 
by 
  intros; 
  sorry

end bad_carrots_count_l32_32639


namespace inverse_proposition_l32_32139

theorem inverse_proposition (a b : ℝ) (h : ab = 0) : (a = 0 → ab = 0) :=
by
  sorry

end inverse_proposition_l32_32139


namespace intersection_of_sets_l32_32693

def setA : Set ℝ := {x : ℝ | -3 ≤ x ∧ x < 4}
def setB : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 5}

theorem intersection_of_sets :
  setA ∩ setB = {x : ℝ | -2 ≤ x ∧ x < 4} :=
by
  sorry

end intersection_of_sets_l32_32693


namespace scale_reading_l32_32999

theorem scale_reading (x : ℝ) (h₁ : 3.25 < x) (h₂ : x < 3.5) : x = 3.3 :=
sorry

end scale_reading_l32_32999


namespace five_digit_palindromes_count_l32_32396

def num_five_digit_palindromes : ℕ :=
  let choices_for_A := 9
  let choices_for_B := 10
  let choices_for_C := 10
  choices_for_A * choices_for_B * choices_for_C

theorem five_digit_palindromes_count : num_five_digit_palindromes = 900 :=
by
  unfold num_five_digit_palindromes
  sorry

end five_digit_palindromes_count_l32_32396


namespace rebecca_bought_2_more_bottles_of_water_l32_32732

noncomputable def number_of_more_bottles_of_water_than_tent_stakes
  (T D W : ℕ) 
  (hT : T = 4) 
  (hD : D = 3 * T) 
  (hTotal : T + D + W = 22) : Prop :=
  W - T = 2

theorem rebecca_bought_2_more_bottles_of_water
  (T D W : ℕ) 
  (hT : T = 4) 
  (hD : D = 3 * T) 
  (hTotal : T + D + W = 22) : 
  number_of_more_bottles_of_water_than_tent_stakes T D W hT hD hTotal :=
by 
  sorry

end rebecca_bought_2_more_bottles_of_water_l32_32732


namespace tangent_ellipse_hyperbola_l32_32004

theorem tangent_ellipse_hyperbola (m : ℝ) :
  (∀ x y : ℝ, x^2 + 9 * y^2 = 9 → x^2 - m * (y + 3)^2 = 4) → m = 5 / 9 :=
by
  sorry

end tangent_ellipse_hyperbola_l32_32004


namespace trig_proof_l32_32082

variable {α a : ℝ}

theorem trig_proof (h₁ : (∃ a : ℝ, a < 0 ∧ (4 * a, -3 * a) = (4 * a, -3 * a)))
                    (h₂ : a < 0) :
  2 * Real.sin α + Real.cos α = 2 / 5 := 
sorry

end trig_proof_l32_32082


namespace fruit_weights_determined_l32_32119

noncomputable def fruit_weight_configuration : Prop :=
  let weights : List ℕ := [100, 150, 170, 200, 280]
  let mandarin := 100
  let apple := 150
  let banana := 170
  let peach := 200
  let orange := 280
  (peach < orange) ∧
  (apple < banana ∧ banana < peach) ∧
  (mandarin < banana) ∧
  (apple + banana > orange)
  
theorem fruit_weights_determined :
  fruit_weight_configuration :=
by
  sorry

end fruit_weights_determined_l32_32119


namespace tony_water_intake_l32_32019

theorem tony_water_intake (yesterday water_two_days_ago : ℝ) 
    (h1 : yesterday = 48) (h2 : yesterday = 0.96 * water_two_days_ago) :
    water_two_days_ago = 50 :=
by
  sorry

end tony_water_intake_l32_32019


namespace value_of_a_l32_32338

theorem value_of_a (a : ℝ) (h1 : a < 0) (h2 : |a| = 3) : a = -3 := 
by
  sorry

end value_of_a_l32_32338


namespace books_remaining_after_second_day_l32_32647

theorem books_remaining_after_second_day :
  let initial_books := 100
  let first_day_borrowed := 5 * 2
  let second_day_borrowed := 20
  let total_borrowed := first_day_borrowed + second_day_borrowed
  let remaining_books := initial_books - total_borrowed
  remaining_books = 70 :=
by
  sorry

end books_remaining_after_second_day_l32_32647


namespace natural_numbers_satisfying_condition_l32_32077

open Nat

theorem natural_numbers_satisfying_condition (r : ℕ) :
  ∃ k : Set ℕ, k = { k | ∃ s t : ℕ, k = 2^(r + s) * t ∧ 2 ∣ t ∧ 2 ∣ s } :=
by
  sorry

end natural_numbers_satisfying_condition_l32_32077


namespace determine_a_b_l32_32615

-- Define the function f
def f (x a b : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

-- Define the first derivative of the function f
def f' (x a b : ℝ) : ℝ := 3*x^2 + 2*a*x + b

-- Define the conditions given in the problem
def conditions (a b : ℝ) : Prop :=
  (f' 1 a b = 0) ∧ (f 1 a b = 10)

-- Provide the main theorem stating the required proof
theorem determine_a_b (a b : ℝ) (h : conditions a b) : a = 4 ∧ b = -11 :=
by {
  sorry
}

end determine_a_b_l32_32615


namespace train_lengths_l32_32763

variable (P L_A L_B : ℝ)

noncomputable def speedA := 180 * 1000 / 3600
noncomputable def speedB := 240 * 1000 / 3600

-- Train A crosses platform P in one minute
axiom hA : speedA * 60 = L_A + P

-- Train B crosses platform P in 45 seconds
axiom hB : speedB * 45 = L_B + P

-- Sum of the lengths of Train A and platform P is twice the length of Train B
axiom hSum : L_A + P = 2 * L_B

theorem train_lengths : L_A = 1500 ∧ L_B = 1500 :=
by
  sorry

end train_lengths_l32_32763


namespace aisha_probability_four_tosses_l32_32660

noncomputable def probability_at_least_two_heads (tosses : ℕ) (heads_needed : ℕ) : ℚ :=
  1 - (nat.choose tosses 1 * (1/2)^tosses + (1/2)^tosses)

theorem aisha_probability_four_tosses :
  probability_at_least_two_heads 4 2 = 11/16 := by
  sorry 

end aisha_probability_four_tosses_l32_32660


namespace unique_solution_c_eq_one_l32_32421

theorem unique_solution_c_eq_one (b c : ℝ) (hb : b > 0) 
  (h_unique_solution : ∃ x : ℝ, x^2 + (b + 1/b) * x + c = 0 ∧ 
  ∀ y : ℝ, y^2 + (b + 1/b) * y + c = 0 → y = x) : c = 1 :=
by
  sorry

end unique_solution_c_eq_one_l32_32421


namespace fraction_work_completed_by_third_group_l32_32701

def working_speeds (name : String) : ℚ :=
  match name with
  | "A"  => 1
  | "B"  => 2
  | "C"  => 1.5
  | "D"  => 2.5
  | "E"  => 3
  | "F"  => 2
  | "W1" => 1
  | "W2" => 1.5
  | "W3" => 1
  | "W4" => 1
  | "W5" => 0.5
  | "W6" => 1
  | "W7" => 1.5
  | "W8" => 1
  | _    => 0

def work_done_per_hour (workers : List String) : ℚ :=
  workers.map working_speeds |>.sum

def first_group : List String := ["A", "B", "C", "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8"]
def second_group : List String := ["A", "B", "C", "D", "E", "F", "W1", "W2"]
def third_group : List String := ["A", "B", "C", "D", "E", "W1", "W2"]

theorem fraction_work_completed_by_third_group :
  (work_done_per_hour third_group) / (work_done_per_hour second_group) = 25 / 29 :=
by
  sorry

end fraction_work_completed_by_third_group_l32_32701


namespace donna_has_40_bananas_l32_32668

-- Define the number of bananas each person has
variables (dawn lydia donna total : ℕ)

-- State the conditions
axiom h1 : dawn + lydia + donna = total
axiom h2 : dawn = lydia + 40
axiom h3 : lydia = 60
axiom h4 : total = 200

-- State the theorem to be proved
theorem donna_has_40_bananas : donna = 40 :=
by {
  sorry -- Placeholder for the proof
}

end donna_has_40_bananas_l32_32668


namespace minimize_integral_l32_32423

open Real
open interval_integral

theorem minimize_integral (a : ℝ) (k : ℝ) (h1 : 0 < a) (h2 : a < (π / 2)) (h3 : cos a = k * a) :
  k = (2 * sqrt 2 / π) * cos (π / (2 * sqrt 2)) :=
sorry

end minimize_integral_l32_32423


namespace powerThreeExpression_l32_32242

theorem powerThreeExpression (x : ℝ) (h : x - (1 / x) = 5) : x^3 - (1 / (x^3)) = 140 := sorry

end powerThreeExpression_l32_32242


namespace number_of_terms_in_simplified_expression_l32_32476

theorem number_of_terms_in_simplified_expression (x y z w : ℕ) :
  let count_even (n : ℕ) := 1 + n / 2 in
  let count_combinations (a : ℕ) := (2024 - a).choose 2 in
  ∑ a in finset.range 1013, count_combinations (2 * a) = "Number of terms in the simplified expression".
sorry

end number_of_terms_in_simplified_expression_l32_32476


namespace matches_in_each_box_l32_32452

noncomputable def matches_per_box (dozens_boxes : ℕ) (total_matches : ℕ) : ℕ :=
  total_matches / (dozens_boxes * 12)

theorem matches_in_each_box :
  matches_per_box 5 1200 = 20 :=
by
  sorry

end matches_in_each_box_l32_32452


namespace find_roses_last_year_l32_32565

-- Definitions based on conditions
def roses_last_year : ℕ := sorry
def roses_this_year := roses_last_year / 2
def roses_needed := 2 * roses_last_year
def rose_cost := 3 -- cost per rose in dollars
def total_spent := 54 -- total spent in dollars

-- Formulate the problem
theorem find_roses_last_year (h : 2 * roses_last_year - roses_this_year = 18)
  (cost_eq : total_spent / rose_cost = 18) :
  roses_last_year = 12 :=
by
  sorry

end find_roses_last_year_l32_32565


namespace spend_on_video_games_l32_32275

/-- Given the total allowance and the fractions of spending on various categories,
prove the amount spent on video games. -/
theorem spend_on_video_games (total_allowance : ℝ)
  (fraction_books fraction_snacks fraction_crafts : ℝ)
  (h_total : total_allowance = 50)
  (h_fraction_books : fraction_books = 1 / 4)
  (h_fraction_snacks : fraction_snacks = 1 / 5)
  (h_fraction_crafts : fraction_crafts = 3 / 10) :
  total_allowance - (fraction_books * total_allowance + fraction_snacks * total_allowance + fraction_crafts * total_allowance) = 12.5 :=
by
  sorry

end spend_on_video_games_l32_32275


namespace highest_slope_product_l32_32623

theorem highest_slope_product (m1 m2 : ℝ) (h1 : m1 = 5 * m2) 
    (h2 : abs ((m2 - m1) / (1 + m1 * m2)) = 1) : (m1 * m2) ≤ 1.8 :=
by
  sorry

end highest_slope_product_l32_32623


namespace truck_loading_time_l32_32646

theorem truck_loading_time (h1_rate h2_rate h3_rate : ℝ)
  (h1 : h1_rate = 1 / 5) (h2 : h2_rate = 1 / 4) (h3 : h3_rate = 1 / 6) :
  (1 / (h1_rate + h2_rate + h3_rate)) = 60 / 37 :=
by simp [h1, h2, h3]; sorry

end truck_loading_time_l32_32646


namespace taco_castle_num_dodge_trucks_l32_32449

theorem taco_castle_num_dodge_trucks
  (D F T V H C : ℕ)
  (hV : V = 5)
  (h1 : F = D / 3)
  (h2 : F = 2 * T)
  (h3 : V = T / 2)
  (h4 : H = 3 * F / 4)
  (h5 : C = 2 * H / 3) :
  D = 60 :=
by
  sorry

end taco_castle_num_dodge_trucks_l32_32449


namespace souvenir_prices_total_profit_l32_32202

variables (x y m n : ℝ)

-- Conditions for the first part
def conditions_part1 : Prop :=
  7 * x + 8 * y = 380 ∧
  10 * x + 6 * y = 380

-- Result for the first part
def result_part1 : Prop :=
  x = 20 ∧ y = 30

-- Conditions for the second part
def conditions_part2 : Prop :=
  m + n = 40 ∧
  20 * m + 30 * n = 900 

-- Result for the second part
def result_part2 : Prop :=
  30 * 5 + 10 * 7 = 220

theorem souvenir_prices (x y : ℝ) (h : conditions_part1 x y) : result_part1 x y :=
by { sorry }

theorem total_profit (m n : ℝ) (h : conditions_part2 m n) : result_part2 :=
by { sorry }

end souvenir_prices_total_profit_l32_32202


namespace polynomial_at_neg_one_eq_neg_two_l32_32829

-- Define the polynomial f(x)
def polynomial (x : ℝ) : ℝ := 1 + x + 2 * x^2 + 3 * x^3 + 4 * x^4 + 5 * x^5

-- Define Horner's method process
def horner_method (x : ℝ) : ℝ :=
  let a5 := 5
  let a4 := 4
  let a3 := 3
  let a2 := 2
  let a1 := 1
  let a  := 1
  let u4 := a5 * x + a4
  let u3 := u4 * x + a3
  let u2 := u3 * x + a2
  let u1 := u2 * x + a1
  let u0 := u1 * x + a
  u0

-- Prove that the polynomial evaluated using Horner's method at x := -1 is equal to -2
theorem polynomial_at_neg_one_eq_neg_two : horner_method (-1) = -2 := by
  sorry

end polynomial_at_neg_one_eq_neg_two_l32_32829


namespace find_2a_plus_b_l32_32736

noncomputable def f (a b x : ℝ) : ℝ := a * x - b
noncomputable def g (x : ℝ) : ℝ := -4 * x + 6
noncomputable def h (a b x : ℝ) : ℝ := f a b (g x)
noncomputable def h_inv (x : ℝ) : ℝ := x + 9

theorem find_2a_plus_b (a b : ℝ) (h_inv_eq: ∀ x : ℝ, h a b (h_inv x) = x) : 2 * a + b = 7 :=
sorry

end find_2a_plus_b_l32_32736


namespace max_sum_xy_l32_32418

theorem max_sum_xy (x y : ℤ) (h1 : x^2 + y^2 = 64) (h2 : x ≥ 0) (h3 : y ≥ 0) : x + y ≤ 8 :=
by sorry

end max_sum_xy_l32_32418


namespace max_quartets_in_5x5_max_quartets_in_mxn_l32_32397

def quartet (c : Nat) : Bool := 
  c > 0

theorem max_quartets_in_5x5 : ∃ q, q = 5 ∧ 
  quartet q := by
  sorry

theorem max_quartets_in_mxn 
  (m n : Nat) (Hmn : m > 0 ∧ n > 0) :
  (∃ q, q = (m * (n - 1)) / 4 ∧ quartet q) ∨ 
  (∃ q, q = (m * (n - 1) - 2) / 4 ∧ quartet q) := by
  sorry

end max_quartets_in_5x5_max_quartets_in_mxn_l32_32397


namespace option1_payment_correct_option2_payment_correct_most_cost_effective_for_30_l32_32183

variables (x : ℕ) (hx : x > 10)

def suit_price : ℕ := 1000
def tie_price : ℕ := 200
def num_suits : ℕ := 10

-- Option 1: Buy one suit, get one tie for free
def option1_payment : ℕ := 200 * x + 8000

-- Option 2: All items sold at a 10% discount
def option2_payment : ℕ := (10 * 1000 + x * 200) * 9 / 10

-- For x = 30, which option is more cost-effective
def x_value := 30
def option1_payment_30 : ℕ := 200 * x_value + 8000
def option2_payment_30 : ℕ := (10 * 1000 + x_value * 200) * 9 / 10
def more_cost_effective_option_30 : ℕ := if option1_payment_30 < option2_payment_30 then option1_payment_30 else option2_payment_30

-- Most cost-effective option for x = 30 with new combination plan
def combination_payment_30 : ℕ := 10000 + 20 * 200 * 9 / 10

-- Statements to be proved
theorem option1_payment_correct : option1_payment x = 200 * x + 8000 := sorry

theorem option2_payment_correct : option2_payment x = (10 * 1000 + x * 200) * 9 / 10 := sorry

theorem most_cost_effective_for_30 :
  option1_payment_30 = 14000 ∧ 
  option2_payment_30 = 14400 ∧ 
  more_cost_effective_option_30 = 14000 ∧
  combination_payment_30 = 13600 := sorry

end option1_payment_correct_option2_payment_correct_most_cost_effective_for_30_l32_32183


namespace find_base_numerica_l32_32844

theorem find_base_numerica (r : ℕ) (h_gadget_cost : 5*r^2 + 3*r = 530) (h_payment : r^3 + r^2 = 1100) (h_change : 4*r^2 + 6*r = 460) :
  r = 9 :=
sorry

end find_base_numerica_l32_32844


namespace cube_surface_area_726_l32_32904

noncomputable def cubeSurfaceArea (volume : ℝ) : ℝ :=
  let side := volume^(1 / 3)
  6 * (side ^ 2)

theorem cube_surface_area_726 (h : cubeSurfaceArea 1331 = 726) : cubeSurfaceArea 1331 = 726 :=
by
  sorry

end cube_surface_area_726_l32_32904


namespace num_pairs_eq_12_l32_32409

theorem num_pairs_eq_12 :
  ∃ (n : ℕ), (∀ (a b : ℕ), 0 < a ∧ 0 < b ∧ a + b ≤ 100 ∧
    (a + 1/b : ℚ) / (1/a + b : ℚ) = 7 ↔ (7 * b = a)) ∧ n = 12 :=
sorry

end num_pairs_eq_12_l32_32409


namespace probability_of_two_hearts_and_three_diff_suits_l32_32738

def prob_two_hearts_and_three_diff_suits (n : ℕ) : ℚ :=
  if n = 5 then 135 / 1024 else 0

theorem probability_of_two_hearts_and_three_diff_suits :
  prob_two_hearts_and_three_diff_suits 5 = 135 / 1024 :=
by
  sorry

end probability_of_two_hearts_and_three_diff_suits_l32_32738


namespace find_n_l32_32295

theorem find_n (N : ℕ) (hN : 10 ≤ N ∧ N < 100)
  (h : ∃ a b : ℕ, N = 10 * a + b ∧ 4 * a + 2 * b = N / 2) : 
  N = 32 ∨ N = 64 ∨ N = 96 :=
sorry

end find_n_l32_32295


namespace NaOH_combined_l32_32561

theorem NaOH_combined (n : ℕ) (h : n = 54) : 
  (2 * n) / 2 = 54 :=
by
  sorry

end NaOH_combined_l32_32561


namespace max_value_of_f_symmetric_about_point_concave_inequality_l32_32964

noncomputable def f (x : ℝ) : ℝ := x^2 / (1 - x)

theorem max_value_of_f : ∃ x, f x = -4 :=
by
  sorry

theorem symmetric_about_point : ∀ x, f (1 - x) + f (1 + x) = -4 :=
by
  sorry

theorem concave_inequality (x1 x2 : ℝ) (h1 : x1 > 1) (h2 : x2 > 1) : 
  f ((x1 + x2) / 2) ≥ (f x1 + f x2) / 2 :=
by
  sorry

end max_value_of_f_symmetric_about_point_concave_inequality_l32_32964


namespace find_V_l32_32526

theorem find_V 
  (c : ℝ)
  (R₁ V₁ W₁ R₂ W₂ V₂ : ℝ)
  (h1 : R₁ = c * (V₁ / W₁))
  (h2 : R₁ = 6)
  (h3 : V₁ = 2)
  (h4 : W₁ = 3)
  (h5 : R₂ = 25)
  (h6 : W₂ = 5)
  (h7 : V₂ = R₂ * W₂ / 9) :
  V₂ = 125 / 9 :=
by sorry

end find_V_l32_32526


namespace point_P_x_coordinate_l32_32705

variable {P : Type} [LinearOrderedField P]

-- Definitions from the conditions
def line_equation (x : P) : P := 0.8 * x
def y_coordinate_P : P := 6
def x_coordinate_P : P := 7.5

-- Theorems to prove that the x-coordinate of P is 7.5.
theorem point_P_x_coordinate (x : P) :
  line_equation x = y_coordinate_P → x = x_coordinate_P :=
by
  intro h
  sorry

end point_P_x_coordinate_l32_32705


namespace value_of_collection_l32_32401

theorem value_of_collection (n : ℕ) (v : ℕ → ℕ) (h1 : n = 20) 
    (h2 : v 5 = 20) (h3 : ∀ k1 k2, v k1 = v k2) : v n = 80 :=
by
  sorry

end value_of_collection_l32_32401


namespace sqrt_addition_l32_32908

theorem sqrt_addition : Real.sqrt 8 + Real.sqrt 2 = 3 * Real.sqrt 2 :=
by
  sorry

end sqrt_addition_l32_32908


namespace arithmetic_geometric_sequences_l32_32143

theorem arithmetic_geometric_sequences :
  ∃ (A B C D : ℤ), A < B ∧ B > 0 ∧ C > 0 ∧ -- Ensure A, B, C are positive
  (B - A) = (C - B) ∧  -- Arithmetic sequence condition
  B * (49 : ℚ) = C * (49 / 9 : ℚ) ∧ -- Geometric sequence condition written using fractional equality
  A + B + C + D = 76 := 
by {
  sorry -- Placeholder for actual proof
}

end arithmetic_geometric_sequences_l32_32143


namespace distinct_ordered_pairs_l32_32830

theorem distinct_ordered_pairs (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (h : 1/m + 1/n = 1/5) : 
  ∃! (m n : ℕ), m > 0 ∧ n > 0 ∧ (1 / m + 1 / n = 1 / 5) :=
sorry

end distinct_ordered_pairs_l32_32830


namespace charge_per_action_figure_l32_32715

-- Definitions according to given conditions
def cost_of_sneakers : ℕ := 90
def saved_amount : ℕ := 15
def num_action_figures : ℕ := 10
def left_after_purchase : ℕ := 25

-- Theorem to prove the charge per action figure
theorem charge_per_action_figure : 
  (cost_of_sneakers - saved_amount + left_after_purchase) / num_action_figures = 10 :=
by 
  sorry

end charge_per_action_figure_l32_32715


namespace sin_double_angle_l32_32229

theorem sin_double_angle (A : ℝ) (h1 : π / 2 < A) (h2 : A < π) (h3 : Real.sin A = 4 / 5) : Real.sin (2 * A) = -24 / 25 := 
by 
  sorry

end sin_double_angle_l32_32229


namespace farm_area_l32_32514

theorem farm_area (length width area : ℝ) 
  (h1 : length = 0.6) 
  (h2 : width = 3 * length) 
  (h3 : area = length * width) : 
  area = 1.08 := 
by 
  sorry

end farm_area_l32_32514


namespace square_field_area_l32_32872

theorem square_field_area (x : ℕ) 
    (hx : 4 * x - 2 = 666) : x^2 = 27889 := by
  -- We would solve for x using the given equation.
  sorry

end square_field_area_l32_32872


namespace good_numbers_identification_l32_32093

def is_good_number (n : ℕ) : Prop :=
  ∃ (a : Fin n → Fin n), 
    (∀ k : Fin n, ∃ m : ℕ, k.val + a k = m * m)

theorem good_numbers_identification : 
  { n : ℕ | ¬is_good_number n } = {1, 2, 4, 6, 7, 9, 11} :=
  sorry

end good_numbers_identification_l32_32093


namespace original_price_of_article_l32_32915

theorem original_price_of_article 
  (S : ℝ) (gain_percent : ℝ) (P : ℝ)
  (h1 : S = 25)
  (h2 : gain_percent = 1.5)
  (h3 : S = P + P * gain_percent) : 
  P = 10 :=
by 
  sorry

end original_price_of_article_l32_32915


namespace y_intercept_of_line_l32_32775

theorem y_intercept_of_line (x y : ℝ) (h : 2 * x - 3 * y = 6) (hx : x = 0) : y = -2 :=
by
  sorry

end y_intercept_of_line_l32_32775


namespace composite_has_at_least_three_factors_l32_32911

-- Definition of composite number in terms of its factors
def is_composite (n : ℕ) : Prop :=
  ∃ (d : ℕ), d ∣ n ∧ d ≠ 1 ∧ d ≠ n

-- Theorem stating that a composite number has at least 3 factors
theorem composite_has_at_least_three_factors (n : ℕ) (h : is_composite n) : 
  (∃ f1 f2 f3, f1 ∣ n ∧ f2 ∣ n ∧ f3 ∣ n ∧ f1 ≠ 1 ∧ f1 ≠ n ∧ f2 ≠ 1 ∧ f2 ≠ n ∧ f3 ≠ 1 ∧ f3 ≠ n ∧ f1 ≠ f2 ∧ f2 ≠ f3) := 
sorry

end composite_has_at_least_three_factors_l32_32911


namespace problem_l32_32125

theorem problem (x : ℝ) (h : x^2 - 1 = 0) : x = -1 ∨ x = 1 :=
sorry

end problem_l32_32125


namespace cost_price_percentage_l32_32997

theorem cost_price_percentage (MP CP : ℝ) 
  (h1 : MP * 0.9 = CP * (72 / 70))
  (h2 : CP / MP * 100 = 87.5) :
  CP / MP = 0.875 :=
by {
  sorry
}

end cost_price_percentage_l32_32997


namespace arithmetic_geometric_inequality_l32_32910

theorem arithmetic_geometric_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (a + b) / 2 ≥ Real.sqrt (a * b) := 
sorry

end arithmetic_geometric_inequality_l32_32910


namespace pressure_increases_when_block_submerged_l32_32779

-- Definitions and conditions
variables (P_0 : ℝ) (ρ : ℝ) (g : ℝ) (h_0 : ℝ) (h_1 : ℝ)
hypothesis (h1_gt_h0 : h_1 > h_0)

-- The proof goal
theorem pressure_increases_when_block_submerged (P_0 ρ g h_0 h_1 : ℝ) (h1_gt_h0 : h_1 > h_0) : 
  let P := P_0 + ρ * g * h_0 in
  let P_1 := P_0 + ρ * g * h_1 in
  P_1 > P := sorry

end pressure_increases_when_block_submerged_l32_32779


namespace cubic_identity_l32_32252

theorem cubic_identity (x : ℝ) (h : x - (1 / x) = 5) : x^3 - (1 / x^3) = 140 := 
  sorry

end cubic_identity_l32_32252


namespace min_groups_with_conditions_l32_32511

theorem min_groups_with_conditions (n a b m : ℕ) (h_n : n = 8) (h_a : a = 4) (h_b : b = 1) :
  m ≥ 2 :=
sorry

end min_groups_with_conditions_l32_32511


namespace function_decreases_l32_32554

def op (m n : ℝ) : ℝ := - (m * n) + n

def f (x : ℝ) : ℝ := op x 2

theorem function_decreases (x1 x2 : ℝ) (h : x1 < x2) : f x1 > f x2 :=
by sorry

end function_decreases_l32_32554


namespace solve_quadratic_for_negative_integer_l32_32013

theorem solve_quadratic_for_negative_integer (N : ℤ) (h_neg : N < 0) (h_eq : 2 * N^2 + N = 20) : N = -4 :=
sorry

end solve_quadratic_for_negative_integer_l32_32013


namespace graphs_symmetric_about_a_axis_of_symmetry_l32_32528

def graph_symmetric_about_line (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a - x) = f (x - a)

theorem graphs_symmetric_about_a (f : ℝ → ℝ) (a : ℝ) :
  ∀ x, f (x - a) = f (a - (x - a)) :=
sorry

theorem axis_of_symmetry (f : ℝ → ℝ) :
  (∀ x : ℝ, f (1 + 2 * x) = f (1 - 2 * x)) →
  ∀ x, f x = f (2 - x) := 
sorry

end graphs_symmetric_about_a_axis_of_symmetry_l32_32528


namespace relationship_M_N_l32_32086

def M : Set Int := {-1, 0, 1}
def N : Set Int := {x | ∃ a b : Int, a ∈ M ∧ b ∈ M ∧ a ≠ b ∧ x = a * b}

theorem relationship_M_N : N ⊆ M ∧ N ≠ M := by
  sorry

end relationship_M_N_l32_32086


namespace lee_charged_per_action_figure_l32_32717

theorem lee_charged_per_action_figure :
  ∀ (sneakers_cost savings action_figures leftovers price_per_fig),
    sneakers_cost = 90 →
    savings = 15 →
    action_figures = 10 →
    leftovers = 25 →
    price_per_fig = 10 →
    (savings + action_figures * price_per_fig) - sneakers_cost = leftovers → price_per_fig = 10 :=
by
  intros sneakers_cost savings action_figures leftovers price_per_fig
  intros h_sneakers_cost h_savings h_action_figures h_leftovers h_price_per_fig
  intros h_total
  sorry

end lee_charged_per_action_figure_l32_32717


namespace supplement_of_complement_65_degrees_l32_32891

def complement (α : ℝ) : ℝ := 90 - α
def supplement (α : ℝ) : ℝ := 180 - α

theorem supplement_of_complement_65_degrees : 
  supplement (complement 65) = 155 :=
by
  sorry

end supplement_of_complement_65_degrees_l32_32891


namespace number_of_students_l32_32601

/-- 
We are given that 36 students are selected from three grades: 
15 from the first grade, 12 from the second grade, and the rest from the third grade. 
Additionally, there are 900 students in the third grade.
We need to prove: the total number of students in the high school is 3600
-/
theorem number_of_students (x y z : ℕ) (s_total : ℕ) (x_sel : ℕ) (y_sel : ℕ) (z_students : ℕ) 
  (h1 : x_sel = 15) 
  (h2 : y_sel = 12) 
  (h3 : x_sel + y_sel + (s_total - (x_sel + y_sel)) = s_total) 
  (h4 : s_total = 36) 
  (h5 : z_students = 900) 
  (h6 : (s_total - (x_sel + y_sel)) = 9) 
  (h7 : 9 / 900 = 1 / 100) : 
  (36 * 100 = 3600) :=
by sorry

end number_of_students_l32_32601


namespace sin_alpha_plus_3pi_div_2_l32_32231

theorem sin_alpha_plus_3pi_div_2 (α : ℝ) (h : Real.cos α = 1 / 3) : 
  Real.sin (α + 3 * Real.pi / 2) = -1 / 3 :=
by
  sorry

end sin_alpha_plus_3pi_div_2_l32_32231


namespace evaluate_polynomial_at_4_l32_32767

noncomputable def polynomial_horner (x : ℤ) : ℤ :=
  (((((3 * x + 6) * x - 20) * x - 8) * x + 15) * x + 9)

theorem evaluate_polynomial_at_4 :
  polynomial_horner 4 = 3269 :=
by
  sorry

end evaluate_polynomial_at_4_l32_32767


namespace abs_inequality_solution_l32_32147

theorem abs_inequality_solution :
  {x : ℝ | |2 * x + 1| > 3} = {x : ℝ | x < -2} ∪ {x : ℝ | x > 1} :=
by
  sorry

end abs_inequality_solution_l32_32147


namespace friends_count_l32_32400

noncomputable def university_students := 1995

theorem friends_count (students : ℕ)
  (knows_each_other : (ℕ → ℕ → Prop))
  (acquaintances : ℕ → ℕ)
  (h_university_students : students = university_students)
  (h_knows_iff_same_acq : ∀ a b, knows_each_other a b ↔ acquaintances a = acquaintances b)
  (h_not_knows_iff_diff_acq : ∀ a b, ¬ knows_each_other a b ↔ acquaintances a ≠ acquaintances b) :
  ∃ a, acquaintances a ≥ 62 ∧ ¬ ∃ a, acquaintances a ≥ 63 :=
sorry

end friends_count_l32_32400


namespace cylinder_lateral_surface_area_l32_32531

theorem cylinder_lateral_surface_area 
  (r h : ℝ) 
  (radius_eq : r = 2) 
  (height_eq : h = 5) : 
  2 * Real.pi * r * h = 62.8 :=
by
  -- Proof steps go here
  sorry

end cylinder_lateral_surface_area_l32_32531


namespace birds_initially_sitting_l32_32608

theorem birds_initially_sitting (initial_birds birds_joined total_birds : ℕ) 
  (h1 : birds_joined = 4) (h2 : total_birds = 6) (h3 : total_birds = initial_birds + birds_joined) : 
  initial_birds = 2 :=
by
  sorry

end birds_initially_sitting_l32_32608


namespace max_buses_constraint_satisfied_l32_32097

def max_buses_9_bus_stops : ℕ := 10

theorem max_buses_constraint_satisfied :
  ∀ (buses : ℕ),
    (∀ (b : buses), b.stops.length = 3) ∧
    (∀ (b1 b2 : buses), b1 ≠ b2 → (b1.stops ∩ b2.stops).length ≤ 1) →
    buses ≤ max_buses_9_bus_stops :=
by
  sorry

end max_buses_constraint_satisfied_l32_32097


namespace number_of_chickens_l32_32730

def cost_per_chicken := 3
def total_cost := 15
def potato_cost := 6
def remaining_amount := total_cost - potato_cost

theorem number_of_chickens : (total_cost - potato_cost) / cost_per_chicken = 3 := by
  sorry

end number_of_chickens_l32_32730


namespace range_of_a_l32_32447

theorem range_of_a (a : ℝ) (h : ∃ x : ℝ, |x - a| + |x - 1| ≤ 3) : -2 ≤ a ∧ a ≤ 4 :=
sorry

end range_of_a_l32_32447


namespace op_value_l32_32834

def op (x y : ℕ) : ℕ := x^3 - 3*x*y^2 + y^3

theorem op_value :
  op 2 1 = 3 := by sorry

end op_value_l32_32834


namespace find_pairs_l32_32215

theorem find_pairs (m n : ℕ) (h : m > 0 ∧ n > 0 ∧ m^2 = n^2 + m + n + 2018) :
  (m, n) = (1010, 1008) ∨ (m, n) = (506, 503) :=
by sorry

end find_pairs_l32_32215


namespace total_weight_of_10_moles_l32_32158

theorem total_weight_of_10_moles
  (molecular_weight : ℕ)
  (moles : ℕ)
  (h_molecular_weight : molecular_weight = 2670)
  (h_moles : moles = 10) :
  moles * molecular_weight = 26700 := by
  -- By substituting the values from the hypotheses:
  -- We will get:
  -- 10 * 2670 = 26700
  sorry

end total_weight_of_10_moles_l32_32158


namespace Wendy_runs_farther_l32_32625

-- Define the distances Wendy ran and walked
def distance_ran : ℝ := 19.83
def distance_walked : ℝ := 9.17

-- Define the difference in distances
def difference : ℝ := distance_ran - distance_walked

-- The theorem to prove
theorem Wendy_runs_farther : difference = 10.66 := by
  sorry

end Wendy_runs_farther_l32_32625


namespace min_value_expression_l32_32321

theorem min_value_expression (α β : ℝ) : 
  (3 * Real.cos α + 4 * Real.sin β - 7) ^ 2 + (3 * Real.sin α + 4 * Real.cos β - 12) ^ 2 ≥ 36 :=
sorry

end min_value_expression_l32_32321


namespace graph_passes_fixed_point_l32_32484

-- Mathematical conditions
variables (a : ℝ)

-- Real numbers and conditions
def is_fixed_point (a : ℝ) : Prop :=
  a > 0 ∧ a ≠ 1 ∧ ∃ x y, (x, y) = (2, 2) ∧ y = a^(x-2) + 1

-- Lean statement for the problem
theorem graph_passes_fixed_point : is_fixed_point a :=
  sorry

end graph_passes_fixed_point_l32_32484


namespace percentage_multiplication_l32_32160

theorem percentage_multiplication :
  (0.15 * 0.20 * 0.25) * 100 = 0.75 := 
by
  sorry

end percentage_multiplication_l32_32160


namespace complex_root_of_unity_prod_l32_32111

theorem complex_root_of_unity_prod (r : ℂ) (h₁ : r^6 = 1) (h₂ : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) = 6 :=
by
  sorry

end complex_root_of_unity_prod_l32_32111


namespace semesters_per_year_l32_32984

-- Definitions of conditions
def cost_per_semester : ℕ := 20000
def total_cost_13_years : ℕ := 520000
def years : ℕ := 13

-- Main theorem to prove
theorem semesters_per_year (S : ℕ) (h1 : total_cost_13_years = years * (S * cost_per_semester)) : S = 2 := by
  sorry

end semesters_per_year_l32_32984


namespace range_of_m_l32_32429

noncomputable def p (m : ℝ) : Prop := ∀ x : ℝ, -m * x ^ 2 + 2 * x - m > 0
noncomputable def q (m : ℝ) : Prop := ∀ x > 0, (4 / x + x - m + 1) > 2

theorem range_of_m : 
  (∃ (m : ℝ), (p m ∨ q m) ∧ ¬(p m ∧ q m)) → (∃ (m : ℝ), -1 ≤ m ∧ m < 3) :=
by
  intros h
  sorry

end range_of_m_l32_32429


namespace value_of_a_l32_32725

theorem value_of_a (a : ℝ) (A : Set ℝ) (B : Set ℝ) 
  (hA : A = {-1, 1, 2}) 
  (hB : B = {a + 1, a ^ 2 + 3}) 
  (h_inter : A ∩ B = {2}) : 
  a = 1 := 
by sorry

end value_of_a_l32_32725


namespace boat_speed_in_still_water_l32_32750

variable (b r d v t : ℝ)

theorem boat_speed_in_still_water (hr : r = 3) 
                                 (hd : d = 3.6) 
                                 (ht : t = 1/5) 
                                 (hv : v = b + r) 
                                 (dist_eq : d = v * t) : 
  b = 15 := 
by
  sorry

end boat_speed_in_still_water_l32_32750


namespace complement_of_supplement_of_30_degrees_l32_32025

def supplementary_angle (x : ℕ) : ℕ := 180 - x
def complementary_angle (x : ℕ) : ℕ := if x > 90 then x - 90 else 90 - x

theorem complement_of_supplement_of_30_degrees : complementary_angle (supplementary_angle 30) = 60 := by
  sorry

end complement_of_supplement_of_30_degrees_l32_32025


namespace equation_holds_if_a_eq_neg_b_c_l32_32226

-- Define the conditions and equation
variables {a b c : ℝ} (h1 : a ≠ 0) (h2 : a + b ≠ 0)

-- Statement to be proved
theorem equation_holds_if_a_eq_neg_b_c : 
  (a = -(b + c)) ↔ (a + b + c) / a = (b + c) / (a + b) := 
sorry

end equation_holds_if_a_eq_neg_b_c_l32_32226


namespace hamburgers_second_day_l32_32663

theorem hamburgers_second_day (x H D : ℕ) (h1 : 3 * H + 4 * D = 10) (h2 : x * H + 3 * D = 7) (h3 : D = 1) (h4 : H = 2) :
  x = 2 :=
by
  sorry

end hamburgers_second_day_l32_32663


namespace find_divisor_l32_32377

theorem find_divisor (d : ℕ) (h1 : d ∣ (9671 - 1)) : d = 9670 :=
by
  sorry

end find_divisor_l32_32377


namespace sum_groups_eq_250_l32_32665

-- Definitions for each sum
def sum1 : ℕ := 3 + 13 + 23 + 33 + 43
def sum2 : ℕ := 7 + 17 + 27 + 37 + 47

-- Theorem statement that the sum of these groups is 250
theorem sum_groups_eq_250 : sum1 + sum2 = 250 :=
by sorry

end sum_groups_eq_250_l32_32665


namespace a_pow_11_b_pow_11_l32_32599

-- Define the conditions a + b = 1, a^2 + b^2 = 3, a^3 + b^3 = 4, a^4 + b^4 = 7, and a^5 + b^5 = 11
def a : ℝ := sorry
def b : ℝ := sorry

axiom h1 : a + b = 1
axiom h2 : a^2 + b^2 = 3
axiom h3 : a^3 + b^3 = 4
axiom h4 : a^4 + b^4 = 7
axiom h5 : a^5 + b^5 = 11

-- Define the recursion pattern for n ≥ 3
axiom h6 (n : ℕ) (hn : n ≥ 3) : a^n + b^n = a^(n-1) + b^(n-1) + a^(n-2) + b^(n-2)

-- Prove that a^11 + b^11 = 199
theorem a_pow_11_b_pow_11 : a^11 + b^11 = 199 :=
by sorry

end a_pow_11_b_pow_11_l32_32599


namespace problem_statement_l32_32719

variable (a b c p q r α β γ : ℝ)

-- Given conditions
def plane_condition : Prop := (a / α) + (b / β) + (c / γ) = 1
def sphere_conditions : Prop := p^3 = α ∧ q^3 = β ∧ r^3 = γ

-- The statement to prove
theorem problem_statement (h_plane : plane_condition a b c α β γ) (h_sphere : sphere_conditions p q r α β γ) :
  (a / p^3) + (b / q^3) + (c / r^3) = 1 := sorry

end problem_statement_l32_32719


namespace correctness_check_l32_32165

noncomputable def questionD (x y : ℝ) : Prop := 
  3 * x^2 * y - 2 * y * x^2 = x^2 * y

theorem correctness_check (x y : ℝ) : questionD x y :=
by 
  sorry

end correctness_check_l32_32165


namespace investment_ratio_l32_32346

theorem investment_ratio (A B : ℕ) (hA : A = 12000) (hB : B = 12000) 
  (interest_A : ℕ := 11 * A / 100) (interest_B : ℕ := 9 * B / 100) 
  (total_interest : interest_A + interest_B = 2400) :
  A / B = 1 :=
by
  sorry

end investment_ratio_l32_32346


namespace total_laundry_time_correct_l32_32797

-- Define the washing and drying times for each load
def whites_washing_time : Nat := 72
def whites_drying_time : Nat := 50
def darks_washing_time : Nat := 58
def darks_drying_time : Nat := 65
def colors_washing_time : Nat := 45
def colors_drying_time : Nat := 54

-- Define total times for each load
def whites_total_time : Nat := whites_washing_time + whites_drying_time
def darks_total_time : Nat := darks_washing_time + darks_drying_time
def colors_total_time : Nat := colors_washing_time + colors_drying_time

-- Define the total time for all three loads
def total_laundry_time : Nat := whites_total_time + darks_total_time + colors_total_time

-- The proof statement
theorem total_laundry_time_correct : total_laundry_time = 344 := by
  unfold total_laundry_time
  unfold whites_total_time darks_total_time colors_total_time
  unfold whites_washing_time whites_drying_time
  unfold darks_washing_time darks_drying_time
  unfold colors_washing_time colors_drying_time
  sorry

end total_laundry_time_correct_l32_32797


namespace calc1_calc2_calc3_calc4_l32_32548

theorem calc1 : 23 + (-16) - (-7) = 14 :=
by
  sorry

theorem calc2 : (3/4 - 7/8 - 5/12) * (-24) = 13 :=
by
  sorry

theorem calc3 : ((7/4 - 7/8 - 7/12) / (-7/8)) + ((-7/8) / (7/4 - 7/8 - 7/12)) = -10/3 :=
by
  sorry

theorem calc4 : -1^4 - (1 - 0.5) * (1/3) * (2 - (-3)^2) = 1/6 :=
by
  sorry

end calc1_calc2_calc3_calc4_l32_32548


namespace two_courtiers_have_same_selection_l32_32493

-- Definitions based on the given conditions
def dogs : Type := Fin 100
def courtiers : Type := Fin 100
def select_three_dogs (c : courtiers) : set dogs := sorry

-- Main theorem statement
theorem two_courtiers_have_same_selection :
  (∀ (c1 c2 : courtiers), c1 ≠ c2 → (select_three_dogs c1 ∩ select_three_dogs c2).card = 2) →
  ∃ (c1 c2 : courtiers), c1 ≠ c2 ∧ select_three_dogs c1 = select_three_dogs c2 :=
sorry

end two_courtiers_have_same_selection_l32_32493


namespace simplify_complex_fraction_l32_32869

noncomputable def simplify_fraction (a b c d : ℂ) : ℂ := sorry

theorem simplify_complex_fraction : 
  let i := Complex.I in
  i^2 = -1 → (3 - 2 * i) / (1 + 4 * i) = (-5/17 - (14/17) * i) := 
by 
  intro h
  sorry

end simplify_complex_fraction_l32_32869


namespace girls_more_than_boys_by_155_l32_32105

def number_of_girls : Real := 542.0
def number_of_boys : Real := 387.0
def difference : Real := number_of_girls - number_of_boys

theorem girls_more_than_boys_by_155 :
  difference = 155.0 := 
by
  sorry

end girls_more_than_boys_by_155_l32_32105


namespace determine_fruit_weights_l32_32122

def Fruit : Type := string

variables (weight : Fruit → ℕ) (orange banana mandarin peach apple : Fruit)

axiom orange_eq : weight orange = 280
axiom peach_lt_orange : weight peach < weight orange
axiom banana_between_apple_and_peach : weight apple < weight banana ∧ weight banana < weight peach
axiom mandarin_lt_banana : weight mandarin < weight banana
axiom apple_banana_gt_orange : weight apple + weight banana > weight orange

-- Lean execution environment requiring proof of the fruit weights
theorem determine_fruit_weights :
  weight mandarin = 100 ∧
  weight apple = 150 ∧
  weight banana = 170 ∧
  weight peach = 200 ∧
  weight orange = 280 :=
  by sorry

end determine_fruit_weights_l32_32122


namespace sum_geometric_series_l32_32930

theorem sum_geometric_series :
  let a := (1 : ℚ) / 5
  let r := (1 : ℚ) / 5
  let n := 8
  let S_n := a * (1 - r^n) / (1 - r)
  S_n = 195312 / 781250 := by
    sorry

end sum_geometric_series_l32_32930


namespace student_total_marks_l32_32978

theorem student_total_marks (total_questions correct_answers incorrect_mark correct_mark : ℕ) 
                             (H1 : total_questions = 60) 
                             (H2 : correct_answers = 34)
                             (H3 : incorrect_mark = 1)
                             (H4 : correct_mark = 4) :
  ((correct_answers * correct_mark) - ((total_questions - correct_answers) * incorrect_mark)) = 110 := 
by {
  -- The proof goes here.
  sorry
}

end student_total_marks_l32_32978


namespace fraction_equivalent_to_decimal_l32_32024

theorem fraction_equivalent_to_decimal : 
  ∃ (x : ℚ), x = 0.6 + 0.0037 * (1 / (1 - 0.01)) ∧ x = 631 / 990 :=
by
  sorry

end fraction_equivalent_to_decimal_l32_32024


namespace cube_volume_l32_32187

theorem cube_volume (A : ℝ) (hA : A = 294) : ∃ (V : ℝ), V = 343 :=
by
  sorry

end cube_volume_l32_32187


namespace find_q_l32_32178

theorem find_q (p : ℝ) (q : ℝ) (h1 : p ≠ 0) (h2 : p = 4) (h3 : q ≠ 0) (avg_speed_eq : (2 * p * 3) / (p + 3) = 24 / q) : q = 7 := 
 by
  sorry

end find_q_l32_32178


namespace maximum_busses_l32_32100

open Finset

-- Definitions of conditions
def bus_stops : ℕ := 9
def stops_per_bus : ℕ := 3

-- Conditions stating that any two buses share at most one common stop
def no_two_buses_share_more_than_one_common_stop (buses : Finset (Finset (Fin (bus_stops)))) : Prop :=
  ∀ b1 b2 ∈ buses, b1 ≠ b2 → (b1 ∩ b2).card ≤ 1

-- Conditions stating that each bus stops at exactly three stops
def each_bus_stops_at_exactly_three_stops (buses : Finset (Finset (Fin (bus_stops)))) : Prop :=
  ∀ b ∈ buses, b.card = stops_per_bus

-- Theorems stating the maximum number of buses possible under given conditions
theorem maximum_busses (buses : Finset (Finset (Fin (bus_stops)))) :
  no_two_buses_share_more_than_one_common_stop buses →
  each_bus_stops_at_exactly_three_stops buses →
  buses.card ≤ 10 := by
  sorry

end maximum_busses_l32_32100


namespace find_vertex_A_l32_32589

variables (B C: ℝ × ℝ × ℝ)

-- Defining midpoints conditions
def midpoint_BC : ℝ × ℝ × ℝ := (1, 5, -1)
def midpoint_AC : ℝ × ℝ × ℝ := (0, 4, -2)
def midpoint_AB : ℝ × ℝ × ℝ := (2, 3, 4)

-- The coordinates of point A we need to prove
def target_A : ℝ × ℝ × ℝ := (1, 2, 3)

-- Lean statement proving the coordinates of A
theorem find_vertex_A (A B C : ℝ × ℝ × ℝ)
  (hBC : midpoint_BC = (1, 5, -1))
  (hAC : midpoint_AC = (0, 4, -2))
  (hAB : midpoint_AB = (2, 3, 4)) :
  A = (1, 2, 3) := 
sorry

end find_vertex_A_l32_32589


namespace largest_integer_solution_of_inequality_l32_32005

theorem largest_integer_solution_of_inequality :
  ∃ x : ℤ, x < 2 ∧ (∀ y : ℤ, y < 2 → y ≤ x) ∧ -x + 3 > 1 :=
sorry

end largest_integer_solution_of_inequality_l32_32005


namespace same_selection_exists_l32_32497

-- Define the set of dogs
def Dogs : Type := Fin 100

-- Define the courtiers
def Courtiers : Type := Fin 100

-- Each courtier selects exactly three dogs.
def selection (c : Courtiers) : Finset Dogs := sorry

-- Condition: For any two courtiers, there are at least two common dogs they both selected.
axiom common_dogs (c₁ c₂ : Courtiers) (h : c₁ ≠ c₂) :
  (selection c₁ ∩ selection c₂).card ≥ 2

-- Prove that there exist two courtiers who picked exactly the same three dogs.
theorem same_selection_exists :
  ∃ c₁ c₂ : Courtiers, c₁ ≠ c₂ ∧ selection c₁ = selection c₂ :=
sorry

end same_selection_exists_l32_32497


namespace anne_equals_bob_l32_32747

-- Define the conditions as constants and functions
def original_price : ℝ := 120.00
def tax_rate : ℝ := 0.06
def discount_rate : ℝ := 0.25

-- Calculation models for Anne and Bob
def anne_total (price : ℝ) (tax : ℝ) (discount : ℝ) : ℝ :=
  (price * (1 + tax)) * (1 - discount)

def bob_total (price : ℝ) (tax : ℝ) (discount : ℝ) : ℝ :=
  (price * (1 - discount)) * (1 + tax)

-- The theorem that states what we need to prove
theorem anne_equals_bob : anne_total original_price tax_rate discount_rate = bob_total original_price tax_rate discount_rate :=
by
  sorry

end anne_equals_bob_l32_32747


namespace min_distance_to_water_all_trees_l32_32015

/-- Proof that the minimum distance Xiao Zhang must walk to water all 10 trees is 410 meters -/
def minimum_distance_to_water_trees (num_trees : ℕ) (distance_between_trees : ℕ) : ℕ := 
  (sorry) -- implementation to calculate the minimum distance

theorem min_distance_to_water_all_trees (num_trees distance_between_trees : ℕ) :
  num_trees = 10 → 
  distance_between_trees = 10 →
  minimum_distance_to_water_trees num_trees distance_between_trees = 410 :=
by
  intros h_num_trees h_distance_between_trees
  rw [h_num_trees, h_distance_between_trees]
  -- Add proof here that the distance is 410
  sorry

end min_distance_to_water_all_trees_l32_32015


namespace remainder_of_sum_l32_32071

theorem remainder_of_sum :
  ((88134 + 88135 + 88136 + 88137 + 88138 + 88139) % 9) = 6 :=
by
  sorry

end remainder_of_sum_l32_32071


namespace sum_a4_a5_a6_l32_32703

variable (a : ℕ → ℤ)
variable (d : ℤ)
variable (h_arith_seq : ∀ n, a (n + 1) = a n + d)
variable (h_a5 : a 5 = 21)

theorem sum_a4_a5_a6 : a 4 + a 5 + a 6 = 63 := by
  sorry

end sum_a4_a5_a6_l32_32703


namespace all_numbers_equal_l32_32642

theorem all_numbers_equal
  (n : ℕ)
  (h n_eq_20 : n = 20)
  (a : ℕ → ℝ)
  (h_avg : ∀ i : ℕ, i < n → a i = (a ((i+n-1) % n) + a ((i+1) % n)) / 2) :
  ∀ i j : ℕ, i < n → j < n → a i = a j :=
by {
  -- Proof steps go here.
  sorry
}

end all_numbers_equal_l32_32642


namespace cube_volume_given_face_perimeter_l32_32141

-- Define the perimeter condition
def is_face_perimeter (perimeter : ℝ) (side_length : ℝ) : Prop :=
  4 * side_length = perimeter

-- Define volume computation
def cube_volume (side_length : ℝ) : ℝ :=
  side_length^3

-- Theorem stating the relationship between face perimeter and cube volume
theorem cube_volume_given_face_perimeter : 
  ∀ (side_length perimeter : ℝ), is_face_perimeter 40 side_length → cube_volume side_length = 1000 :=
by
  intros side_length perimeter h
  sorry

end cube_volume_given_face_perimeter_l32_32141


namespace baked_goods_not_eaten_l32_32108

theorem baked_goods_not_eaten : 
  let cookies_initial := 200
  let brownies_initial := 150
  let cupcakes_initial := 100
  
  let cookies_after_wife := cookies_initial - 0.30 * cookies_initial
  let brownies_after_wife := brownies_initial - 0.20 * brownies_initial
  let cupcakes_after_wife := cupcakes_initial / 2
  
  let cookies_after_daughter := cookies_after_wife - 40
  let brownies_after_daughter := brownies_after_wife - 0.15 * brownies_after_wife
  
  let cookies_after_friend := cookies_after_daughter - (cookies_after_daughter / 4)
  let brownies_after_friend := brownies_after_daughter - 0.10 * brownies_after_daughter
  let cupcakes_after_friend := cupcakes_after_wife - 10
  
  let cookies_after_other_friend := cookies_after_friend - 0.05 * cookies_after_friend
  let brownies_after_other_friend := brownies_after_friend - 0.05 * brownies_after_friend
  let cupcakes_after_other_friend := cupcakes_after_friend - 5
  
  let cookies_after_javier := cookies_after_other_friend / 2
  let brownies_after_javier := brownies_after_other_friend / 2
  let cupcakes_after_javier := cupcakes_after_other_friend / 2
  
  let total_remaining := cookies_after_javier + brownies_after_javier + cupcakes_after_javier
  total_remaining = 98 := by
{
  sorry
}

end baked_goods_not_eaten_l32_32108


namespace jake_peaches_calculation_l32_32847

variable (S_p : ℕ) (J_p : ℕ)

-- Given that Steven has 19 peaches
def steven_peaches : ℕ := 19

-- Jake has 12 fewer peaches than Steven
def jake_peaches : ℕ := S_p - 12

theorem jake_peaches_calculation (h1 : S_p = steven_peaches) (h2 : S_p = 19) :
  J_p = jake_peaches := 
by
  sorry

end jake_peaches_calculation_l32_32847


namespace time_ratio_upstream_downstream_l32_32337

theorem time_ratio_upstream_downstream (S_boat S_stream D : ℝ) (h1 : S_boat = 72) (h2 : S_stream = 24) :
  let time_upstream := D / (S_boat - S_stream)
  let time_downstream := D / (S_boat + S_stream)
  (time_upstream / time_downstream) = 2 :=
by
  sorry

end time_ratio_upstream_downstream_l32_32337


namespace sum_of_number_and_conjugate_l32_32061

theorem sum_of_number_and_conjugate :
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end sum_of_number_and_conjugate_l32_32061


namespace max_buses_in_city_l32_32096

theorem max_buses_in_city (num_stops stops_per_bus shared_stops : ℕ) (h_stops : num_stops = 9) (h_stops_per_bus : stops_per_bus = 3) (h_shared_stops : shared_stops = 1) : 
  ∃ (max_buses : ℕ), max_buses = 12 :=
by
  sorry

end max_buses_in_city_l32_32096


namespace arithmetic_sequence_fifth_term_l32_32483

noncomputable def fifth_term (x y : ℝ) : ℝ :=
  let a1 := x^2 + y^2
  let a2 := x^2 - y^2
  let a3 := x^2 * y^2
  let a4 := x^2 / y^2
  let d := -2 * y^2
  a4 + d

theorem arithmetic_sequence_fifth_term (x y : ℝ) (hy : y ≠ 0) (hx2 : x ^ 2 = 3 * y ^ 2 / (y ^ 2 - 1)) :
  fifth_term x y = 3 / (y ^ 2 - 1) - 2 * y ^ 2 :=
by
  sorry

end arithmetic_sequence_fifth_term_l32_32483


namespace powerThreeExpression_l32_32241

theorem powerThreeExpression (x : ℝ) (h : x - (1 / x) = 5) : x^3 - (1 / (x^3)) = 140 := sorry

end powerThreeExpression_l32_32241


namespace function_inverse_overlap_form_l32_32436

theorem function_inverse_overlap_form (a b c d : ℝ) (h : ¬(a = 0 ∧ c = 0)) : 
  (∀ x, (c * x + d) * (dx - b) = (a * x + b) * (-c * x + a)) → 
  (∃ f : ℝ → ℝ, (∀ x, f x = x ∨ f x = (a * x + b) / (c * x - a))) :=
by 
  sorry

end function_inverse_overlap_form_l32_32436


namespace range_of_k_l32_32584

theorem range_of_k (a k : ℝ) : 
  (∀ x y : ℝ, y^2 - x * y + 2 * x + k = 0 → (x = a ∧ y = -a)) →
  k ≤ 1/2 :=
by sorry

end range_of_k_l32_32584


namespace total_guppies_l32_32695

-- Define conditions
def Haylee_guppies : ℕ := 3 * 12
def Jose_guppies : ℕ := Haylee_guppies / 2
def Charliz_guppies : ℕ := Jose_guppies / 3
def Nicolai_guppies : ℕ := Charliz_guppies * 4

-- Theorem statement: total number of guppies is 84
theorem total_guppies : Haylee_guppies + Jose_guppies + Charliz_guppies + Nicolai_guppies = 84 := 
by 
  sorry

end total_guppies_l32_32695


namespace intersection_of_sets_l32_32571

open Set

def A : Set ℝ := { x | -1 < x ∧ x < 2 }
def B : Set ℝ := { x | 0 < x }

theorem intersection_of_sets : A ∩ B = { x | 0 < x ∧ x < 2 } :=
by sorry

end intersection_of_sets_l32_32571


namespace certain_number_is_11_l32_32042

theorem certain_number_is_11 (x : ℝ) (h : 15 * x = 165) : x = 11 :=
by {
  sorry
}

end certain_number_is_11_l32_32042


namespace min_value_of_expression_l32_32325

open Real

theorem min_value_of_expression (α β : ℝ) :
  (3 * cos α + 4 * sin β - 7)^2 + (3 * sin α + 4 * cos β - 12)^2 ≥ 36 :=
by
  sorry

end min_value_of_expression_l32_32325


namespace simplify_and_find_ratio_l32_32698

theorem simplify_and_find_ratio (m : ℤ) (c d : ℤ) (h : (5 * m + 15) / 5 = c * m + d) : d / c = 3 := by
  sorry

end simplify_and_find_ratio_l32_32698


namespace perpendicular_bisector_l32_32613

theorem perpendicular_bisector (x y : ℝ) :
  (x - 2 * y + 1 = 0 ∧ -1 ≤ x ∧ x ≤ 3) → (2 * x + y - 3 = 0) :=
by
  sorry

end perpendicular_bisector_l32_32613


namespace solve_for_n_l32_32833

theorem solve_for_n (n : ℝ) (h : 1 / (2 * n) + 1 / (4 * n) = 3 / 12) : n = 3 :=
sorry

end solve_for_n_l32_32833


namespace polygon_expected_value_l32_32977

def polygon_expected_sides (area_square : ℝ) (flower_prob : ℝ) (area_flower : ℝ) (hex_sides : ℝ) (pent_sides : ℝ) : ℝ :=
  hex_sides * flower_prob + pent_sides * (area_square - flower_prob)

theorem polygon_expected_value :
  polygon_expected_sides 1 (π - 1) (π - 1) 6 5 = π + 4 :=
by
  -- Proof is skipped
  sorry

end polygon_expected_value_l32_32977


namespace Maryann_frees_all_friends_in_42_minutes_l32_32115

-- Definitions for the problem conditions
def time_to_pick_cheap_handcuffs := 6
def time_to_pick_expensive_handcuffs := 8
def number_of_friends := 3

-- Define the statement we need to prove
theorem Maryann_frees_all_friends_in_42_minutes :
  (time_to_pick_cheap_handcuffs + time_to_pick_expensive_handcuffs) * number_of_friends = 42 :=
by
  sorry

end Maryann_frees_all_friends_in_42_minutes_l32_32115


namespace direct_proportion_increases_inverse_proportion_increases_l32_32361

-- Question 1: Prove y=2x increases as x increases.
theorem direct_proportion_increases (x1 x2 : ℝ) (h : x1 < x2) : 
  2 * x1 < 2 * x2 := by sorry

-- Question 2: Prove y=-2/x increases as x increases when x > 0.
theorem inverse_proportion_increases (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < x2) : 
  - (2 / x1) < - (2 / x2) := by sorry

end direct_proportion_increases_inverse_proportion_increases_l32_32361


namespace calc_one_calc_two_calc_three_l32_32861

theorem calc_one : (54 + 38) * 15 = 1380 := by
  sorry

theorem calc_two : 1500 - 32 * 45 = 60 := by
  sorry

theorem calc_three : 157 * (70 / 35) = 314 := by
  sorry

end calc_one_calc_two_calc_three_l32_32861


namespace cube_volume_l32_32195

-- Define the given condition: The surface area of the cube
def surface_area (A : ℕ) := A = 294

-- The key proposition we need to prove using the given condition
theorem cube_volume (s : ℕ) (A : ℕ) (V : ℕ) 
  (area_condition : surface_area A)
  (side_length_condition : s ^ 2 = A / 6) 
  (volume_condition : V = s ^ 3) : 
  V = 343 := by
  sorry

end cube_volume_l32_32195


namespace weighted_average_inequality_l32_32399

variable (x y z : ℝ)
variable (h1 : x < y) (h2 : y < z)

theorem weighted_average_inequality :
  (4 * z + x + y) / 6 > (x + y + 2 * z) / 4 :=
by
  sorry

end weighted_average_inequality_l32_32399


namespace total_seniors_is_161_l32_32470

def total_students : ℕ := 240

def percentage_statistics : ℚ := 0.45
def percentage_geometry : ℚ := 0.35
def percentage_calculus : ℚ := 0.20

def percentage_stats_and_calc : ℚ := 0.10
def percentage_geom_and_calc : ℚ := 0.05

def percentage_seniors_statistics : ℚ := 0.90
def percentage_seniors_geometry : ℚ := 0.60
def percentage_seniors_calculus : ℚ := 0.80

def students_in_statistics : ℚ := percentage_statistics * total_students
def students_in_geometry : ℚ := percentage_geometry * total_students
def students_in_calculus : ℚ := percentage_calculus * total_students

def students_in_stats_and_calc : ℚ := percentage_stats_and_calc * students_in_statistics
def students_in_geom_and_calc : ℚ := percentage_geom_and_calc * students_in_geometry

def unique_students_in_statistics : ℚ := students_in_statistics - students_in_stats_and_calc
def unique_students_in_geometry : ℚ := students_in_geometry - students_in_geom_and_calc
def unique_students_in_calculus : ℚ := students_in_calculus - students_in_stats_and_calc - students_in_geom_and_calc

def seniors_in_statistics : ℚ := percentage_seniors_statistics * unique_students_in_statistics
def seniors_in_geometry : ℚ := percentage_seniors_geometry * unique_students_in_geometry
def seniors_in_calculus : ℚ := percentage_seniors_calculus * unique_students_in_calculus

def total_seniors : ℚ := seniors_in_statistics + seniors_in_geometry + seniors_in_calculus

theorem total_seniors_is_161 : total_seniors = 161 :=
by
  sorry

end total_seniors_is_161_l32_32470


namespace cauchy_functional_eq_l32_32937

theorem cauchy_functional_eq
  (f : ℚ → ℚ)
  (h : ∀ x y : ℚ, f (x + y) = f x + f y) :
  ∃ a : ℚ, ∀ x : ℚ, f x = a * x :=
sorry

end cauchy_functional_eq_l32_32937


namespace area_of_quadrilateral_l32_32903

theorem area_of_quadrilateral (d h1 h2 : ℝ) (hd : d = 20) (hh1 : h1 = 9) (hh2 : h2 = 6) :
  (1 / 2) * d * (h1 + h2) = 150 :=
by
  rw [hd, hh1, hh2]
  norm_num

end area_of_quadrilateral_l32_32903


namespace evaluate_expression_l32_32218

theorem evaluate_expression : 
  (4 * 6) / (12 * 14) * (8 * 12 * 14) / (4 * 6 * 8) = 1 := 
by
  sorry

end evaluate_expression_l32_32218


namespace num_males_selected_l32_32052

theorem num_males_selected (total_male total_female total_selected : ℕ)
                           (h_male : total_male = 56)
                           (h_female : total_female = 42)
                           (h_selected : total_selected = 28) :
  (total_male * total_selected) / (total_male + total_female) = 16 := 
by {
  sorry
}

end num_males_selected_l32_32052


namespace integer_solutions_conditions_even_l32_32074

theorem integer_solutions_conditions_even (n : ℕ) (x : ℕ → ℤ) :
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → 
    x i ^ 2 + x ((i % n) + 1) ^ 2 + 50 = 16 * x i + 12 * x ((i % n) + 1) ) → 
  n % 2 = 0 :=
by 
sorry

end integer_solutions_conditions_even_l32_32074


namespace computer_price_ratio_l32_32838

theorem computer_price_ratio (d : ℝ) (h1 : d + 0.30 * d = 377) :
  ((d + 377) / d) = 2.3 := by
  sorry

end computer_price_ratio_l32_32838


namespace inequality_order_l32_32871

theorem inequality_order (a b c : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0)
  (h : (a^2 / (b^2 + c^2)) < (b^2 / (c^2 + a^2)) ∧ (b^2 / (c^2 + a^2)) < (c^2 / (a^2 + b^2))) :
  |a| < |b| ∧ |b| < |c| := 
sorry

end inequality_order_l32_32871


namespace speed_of_A_is_24_speed_of_A_is_18_l32_32106

-- Definitions for part 1
def speed_of_B (x : ℝ) := x
def speed_of_A_1 (x : ℝ) := 1.2 * x
def distance_AB := 30 -- kilometers
def distance_B_rides_first := 2 -- kilometers
def time_A_catches_up := 0.5 -- hours

theorem speed_of_A_is_24 (x : ℝ) (h1 : 0.6 * x = 2 + 0.5 * x) : speed_of_A_1 x = 24 := by
  sorry

-- Definitions for part 2
def speed_of_A_2 (y : ℝ) := 1.2 * y
def time_B_rides_first := 1/3 -- hours
def time_difference := 1/3 -- hours

theorem speed_of_A_is_18 (y : ℝ) (h2 : (30 / y) - (30 / (1.2 * y)) = 1/3) : speed_of_A_2 y = 18 := by
  sorry

end speed_of_A_is_24_speed_of_A_is_18_l32_32106


namespace triangle_area_given_conditions_l32_32306

theorem triangle_area_given_conditions (a b c A B S : ℝ) (h₁ : (2 * c - b) * Real.cos A = a * Real.cos B) (h₂ : b = 1) (h₃ : c = 2) :
  S = (1 / 2) * b * c * Real.sin A → S = Real.sqrt 3 / 2 := 
by
  intros
  sorry

end triangle_area_given_conditions_l32_32306


namespace cubic_identity_l32_32256

theorem cubic_identity (x : ℝ) (h : x - 1/x = 5) : x^3 - 1/x^3 = 140 :=
by
  sorry

end cubic_identity_l32_32256


namespace isosceles_triangle_perimeter_l32_32753

theorem isosceles_triangle_perimeter (a b : ℝ) (h₁ : a = 4) (h₂ : b = 9) (h_iso : ¬(4 + 4 > 9 ∧ 4 + 9 > 4 ∧ 9 + 4 > 4))
  (h_ineq : (9 + 9 > 4) ∧ (9 + 4 > 9) ∧ (4 + 9 > 9)) : 2 * b + a = 22 :=
by sorry

end isosceles_triangle_perimeter_l32_32753


namespace sum_of_k_distinct_integer_roots_l32_32374

theorem sum_of_k_distinct_integer_roots : 
  (∑ k in {k : ℤ | ∃ p q : ℤ, p ≠ q ∧ p * q = 4 ∧ k = 3 * (p + q)}, k) = 0 := 
by
  sorry

end sum_of_k_distinct_integer_roots_l32_32374


namespace sum_first_8_geometric_l32_32980

theorem sum_first_8_geometric :
  let a₁ := 1 / 15
  let r := 2
  let S₄ := a₁ * (1 - r^4) / (1 - r)
  let S₈ := a₁ * (1 - r^8) / (1 - r)
  S₄ = 1 → S₈ = 17 := 
by
  intros a₁ r S₄ S₈ h
  sorry

end sum_first_8_geometric_l32_32980


namespace smallest_e_value_l32_32269

noncomputable def poly := (1, -3, 7, -2/5)

theorem smallest_e_value (a b c d e : ℤ) 
  (h_poly_eq : a * (1)^4 + b * (1)^3 + c * (1)^2 + d * (1) + e = 0)
  (h_poly_eq_2 : a * (-3)^4 + b * (-3)^3 + c * (-3)^2 + d * (-3) + e = 0)
  (h_poly_eq_3 : a * 7^4 + b * 7^3 + c * 7^2 + d * 7 + e = 0)
  (h_poly_eq_4 : a * (-2/5)^4 + b * (-2/5)^3 + c * (-2/5)^2 + d * (-2/5) + e = 0)
  (h_e_positive : e > 0) :
  e = 42 :=
sorry

end smallest_e_value_l32_32269


namespace ladder_slip_l32_32388

theorem ladder_slip 
  (ladder_length : ℝ) 
  (initial_base : ℝ) 
  (slip_height : ℝ) 
  (h_length : ladder_length = 30) 
  (h_base : initial_base = 11) 
  (h_slip : slip_height = 6) 
  : ∃ (slide_distance : ℝ), abs (slide_distance - 9.49) < 0.01 :=
by
  let initial_height := Real.sqrt (ladder_length^2 - initial_base^2)
  let new_height := initial_height - slip_height
  let new_base := Real.sqrt (ladder_length^2 - new_height^2)
  let slide_distance := new_base - initial_base
  use slide_distance
  have h_approx : abs (slide_distance - 9.49) < 0.01 := sorry
  exact h_approx

end ladder_slip_l32_32388


namespace count_four_digit_numbers_with_thousands_digit_five_l32_32438

theorem count_four_digit_numbers_with_thousands_digit_five :
  ∃ n : ℕ, n = 1000 ∧ ∀ x : ℕ, 5000 ≤ x ∧ x ≤ 5999 ↔ (x - 4999) ∈ (finset.range 1000 + 1) :=
by
  sorry

end count_four_digit_numbers_with_thousands_digit_five_l32_32438


namespace seashells_collected_l32_32473

theorem seashells_collected (x y z : ℕ) (hyp : x + y / 2 + z + 5 = 76) : x + y + z = 71 := 
by {
  sorry
}

end seashells_collected_l32_32473


namespace sum_of_prime_factors_eq_22_l32_32355

-- Conditions: n is defined as 3^6 - 1
def n : ℕ := 3^6 - 1

-- Statement: The sum of the prime factors of n is 22
theorem sum_of_prime_factors_eq_22 : 
  (∀ p : ℕ, p ∣ n → Prime p → p = 2 ∨ p = 7 ∨ p = 13) → 
  (2 + 7 + 13 = 22) :=
by sorry

end sum_of_prime_factors_eq_22_l32_32355


namespace problem_statement_l32_32162

variables {a b y x : ℝ}

theorem problem_statement :
  (3 * a + 2 * b ≠ 5 * a * b) ∧
  (5 * y - 3 * y ≠ 2) ∧
  (7 * a + a ≠ 7 * a ^ 2) ∧
  (3 * x^2 * y - 2 * y * x^2 = x^2 * y) :=
by
  split
  · intro h
    have h₁ : 3 * a + 2 * b = 3 * a + 2 * b := rfl
    rw h₁ at h
    sorry
  split
  · intro h
    have h₁ : 5 * y - 3 * y = 2 * y := by ring
    rw h₁ at h
    sorry
  split
  · intro h
    have h₁ : 7 * a + a = 8 * a := by ring
    rw h₁ at h
    sorry
  · ring

end problem_statement_l32_32162


namespace second_derivative_of_y_l32_32814

noncomputable def y (x : ℝ) : ℝ := x^2 * Real.log (1 + Real.sin x)

theorem second_derivative_of_y :
  (deriv^[2] y) x = 
  2 * Real.log (1 + Real.sin x) + (4 * x * Real.cos x - x ^ 2) / (1 + Real.sin x) :=
sorry

end second_derivative_of_y_l32_32814


namespace average_speed_of_car_l32_32380

theorem average_speed_of_car (time : ℝ) (distance : ℝ) (h_time : time = 4.5) (h_distance : distance = 360) : 
  distance / time = 80 :=
by
  sorry

end average_speed_of_car_l32_32380


namespace simplify_fraction_l32_32519

theorem simplify_fraction : 1 / (Real.sqrt 3 + 1) = (Real.sqrt 3 - 1) / 2 :=
by
sorry

end simplify_fraction_l32_32519


namespace tangent_line_equation_l32_32963

theorem tangent_line_equation
  (x y : ℝ)
  (h₁ : x^2 + y^2 = 5)
  (hM : x = -1 ∧ y = 2) :
  x - 2 * y + 5 = 0 :=
by
  sorry

end tangent_line_equation_l32_32963


namespace two_pow_58_plus_one_factored_l32_32604

theorem two_pow_58_plus_one_factored :
  ∃ (a b c : ℕ), 2 < a ∧ 2 < b ∧ 2 < c ∧ 2 ^ 58 + 1 = a * b * c :=
sorry

end two_pow_58_plus_one_factored_l32_32604


namespace chairs_to_remove_is_33_l32_32535

-- Definitions for the conditions
def chairs_per_row : ℕ := 11
def total_chairs : ℕ := 110
def students : ℕ := 70

-- Required statement
theorem chairs_to_remove_is_33 
  (h_divisible_by_chairs_per_row : ∀ n, n = total_chairs - students → ∃ k, n = chairs_per_row * k) :
  ∃ rem_chairs : ℕ, rem_chairs = total_chairs - 77 ∧ rem_chairs = 33 := sorry

end chairs_to_remove_is_33_l32_32535


namespace courtiers_selection_l32_32503

theorem courtiers_selection (dogs : Finset ℕ) (courtiers : Finset (Finset ℕ))
  (h1 : dogs.card = 100)
  (h2 : courtiers.card = 100)
  (h3 : ∀ c ∈ courtiers, c.card = 3)
  (h4 : ∀ {c1 c2 : Finset ℕ}, c1 ∈ courtiers → c2 ∈ courtiers → c1 ≠ c2 → (c1 ∩ c2).card = 2) :
  ∃ c1 c2 ∈ courtiers, c1 = c2 :=
sorry

end courtiers_selection_l32_32503


namespace distance_between_home_and_school_l32_32632

variable (D T : ℝ)

def boy_travel_5kmhr : Prop :=
  5 * (T + 7 / 60) = D

def boy_travel_10kmhr : Prop :=
  10 * (T - 8 / 60) = D

theorem distance_between_home_and_school :
  (boy_travel_5kmhr D T) ∧ (boy_travel_10kmhr D T) → D = 2.5 :=
by
  intro h
  sorry

end distance_between_home_and_school_l32_32632


namespace consecutive_even_sum_l32_32751

theorem consecutive_even_sum (N S : ℤ) (m : ℤ) 
  (hk : 2 * m + 1 > 0) -- k is the number of consecutive even numbers, which is odd
  (h_sum : (2 * m + 1) * N = S) -- The condition of the sum
  (h_even : N % 2 = 0) -- The middle number is even
  : (∃ k : ℤ, k = 2 * m + 1 ∧ k > 0 ∧ (k * N / 2) = S/2 ) := 
  sorry

end consecutive_even_sum_l32_32751


namespace pasture_feeding_l32_32508

-- The definitions corresponding to the given conditions
def portion_per_cow_per_day := 1

def food_needed (cows : ℕ) (days : ℕ) : ℕ := cows * days

def growth_rate (food10for20 : ℕ) (food15for10 : ℕ) (days10_20 : ℕ) : ℕ :=
  (food10for20 - food15for10) / days10_20

def food_growth_rate := growth_rate (food_needed 10 20) (food_needed 15 10) 10

def new_grass_feed_cows_per_day := food_growth_rate / portion_per_cow_per_day

def original_grass := (food_needed 10 20) - (food_growth_rate * 20)

def days_to_feed_30_cows := original_grass / (30 - new_grass_feed_cows_per_day)

-- The statement we want to prove
theorem pasture_feeding :
  new_grass_feed_cows_per_day = 5 ∧ days_to_feed_30_cows = 4 := by
  sorry

end pasture_feeding_l32_32508


namespace arithmetic_sequence_25th_term_l32_32898

theorem arithmetic_sequence_25th_term (a1 a2 : ℕ) (d : ℕ) (n : ℕ) (h1 : a1 = 2) (h2 : a2 = 5) (h3 : d = a2 - a1) (h4 : n = 25) :
  a1 + (n - 1) * d = 74 :=
by
  sorry

end arithmetic_sequence_25th_term_l32_32898


namespace max_buses_l32_32103

theorem max_buses (bus_stops : ℕ) (each_bus_stops : ℕ) (no_shared_stops : ∀ b₁ b₂ : Finset ℕ, b₁ ≠ b₂ → (bus_stops \ (b₁ ∪ b₂)).card ≤ bus_stops - 6) : 
  bus_stops = 9 ∧ each_bus_stops = 3 → ∃ max_buses, max_buses = 12 :=
by
  sorry

end max_buses_l32_32103


namespace find_m_l32_32437

-- Definition of vector
def vector (α : Type*) := α × α

-- Two vectors are collinear and have the same direction
def collinear_and_same_direction (a b : vector ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ a = (k * b.1, k * b.2)

-- The vectors a and b
def a (m : ℝ) : vector ℝ := (m, 1)
def b (m : ℝ) : vector ℝ := (4, m)

-- The theorem we want to prove
theorem find_m (m : ℝ) (h1 : collinear_and_same_direction (a m) (b m)) : m = 2 :=
  sorry

end find_m_l32_32437


namespace find_numbers_l32_32899

theorem find_numbers :
  ∃ (x y z : ℕ), x = y + 75 ∧ 
                 (x * y = z + 1000) ∧
                 (z = 227 * y + 113) ∧
                 (x = 234) ∧ 
                 (y = 159) := by
  sorry

end find_numbers_l32_32899


namespace set_expression_l32_32558

def is_natural_number (x : ℚ) : Prop :=
  ∃ n : ℕ, x = n

theorem set_expression :
  {x : ℕ | is_natural_number (6 / (5 - x) : ℚ)} = {2, 3, 4} :=
sorry

end set_expression_l32_32558


namespace unique_solution_l32_32870

theorem unique_solution (x : ℝ) (h : (1 / (x - 1)) = (3 / (2 * x - 3))) : x = 0 := 
sorry

end unique_solution_l32_32870


namespace count_primes_5p2p1_minus_1_perfect_square_l32_32088

-- Define the predicate for a prime number
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Predicate for perfect square
def is_perfect_square (n : ℕ) : Prop := 
  ∃ m : ℕ, m * m = n

-- The main theorem statement
theorem count_primes_5p2p1_minus_1_perfect_square :
  (∀ p : ℕ, is_prime p → is_perfect_square (5 * p * (2^(p + 1) - 1))) → ∃! p : ℕ, is_prime p ∧ is_perfect_square (5 * p * (2^(p + 1) - 1)) :=
sorry

end count_primes_5p2p1_minus_1_perfect_square_l32_32088


namespace probability_two_from_same_province_l32_32395

theorem probability_two_from_same_province :
  let total_singers := 12
  let selected_singers := 4
  let num_provinces := 6
  let singers_per_province := 2
  let total_ways := Nat.choose total_singers selected_singers
  let favorable_ways := num_provinces * Nat.choose singers_per_province singers_per_province *
                        Nat.choose (total_singers - singers_per_province) (selected_singers - singers_per_province) *
                        (total_singers - singers_per_province - (selected_singers - singers_per_province + 1))
  ∃ (p : ℚ), p = (favorable_ways : ℚ) / (total_ways : ℚ) ∧ p = 16 / 33 := 
by by sorry

end probability_two_from_same_province_l32_32395


namespace profit_calculation_l32_32525

-- Define conditions based on investments
def JohnInvestment := 700
def MikeInvestment := 300

-- Define the equality condition where John received $800 more than Mike
theorem profit_calculation (P : ℝ) 
  (h1 : (P / 6 + (7 / 10) * (2 * P / 3)) - (P / 6 + (3 / 10) * (2 * P / 3)) = 800) : 
  P = 3000 := 
sorry

end profit_calculation_l32_32525


namespace sum_of_values_k_l32_32373

theorem sum_of_values_k (k : ℕ) : 
  (∀ x y : ℤ, (3 * x * x - k * x + 12 = 0) ∧ (3 * y * y - k * y + 12 = 0) ∧ x ≠ y) → k = 0 :=
by
  sorry

end sum_of_values_k_l32_32373


namespace shyne_total_plants_l32_32868

/-- Shyne's seed packets -/
def eggplants_per_packet : ℕ := 14
def sunflowers_per_packet : ℕ := 10

/-- Seed packets purchased by Shyne -/
def eggplant_packets : ℕ := 4
def sunflower_packets : ℕ := 6

/-- Total number of plants grown by Shyne -/
def total_plants : ℕ := 116

theorem shyne_total_plants :
  eggplants_per_packet * eggplant_packets + sunflowers_per_packet * sunflower_packets = total_plants :=
by
  sorry

end shyne_total_plants_l32_32868


namespace avg_remaining_two_l32_32171

theorem avg_remaining_two (a b c d e : ℝ) (h1 : (a + b + c + d + e) / 5 = 8) (h2 : (a + b + c) / 3 = 4) :
  (d + e) / 2 = 14 := by
  sorry

end avg_remaining_two_l32_32171


namespace verify_differential_eq_l32_32154

noncomputable def function_z (x y : ℝ) : ℝ := 2 * Real.cos (y - x / 2) ^ 2

theorem verify_differential_eq (x y : ℝ) :
  2 * (deriv (λ x, deriv (λ x, function_z x y) x) x) +
    (deriv (λ y, deriv (λ x, function_z x y) x) y) = 0 :=
by
  sorry

end verify_differential_eq_l32_32154


namespace cubic_identity_l32_32255

theorem cubic_identity (x : ℝ) (h : x - 1/x = 5) : x^3 - 1/x^3 = 140 :=
by
  sorry

end cubic_identity_l32_32255


namespace tan_alpha_eq_one_l32_32443

theorem tan_alpha_eq_one (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h : Real.cos (α + β) = Real.sin (α - β)) : Real.tan α = 1 :=
sorry

end tan_alpha_eq_one_l32_32443


namespace john_splits_profit_correctly_l32_32983

-- Conditions
def total_cookies : ℕ := 6 * 12
def revenue_per_cookie : ℝ := 1.5
def cost_per_cookie : ℝ := 0.25
def amount_per_charity : ℝ := 45

-- Computations based on conditions
def total_revenue : ℝ := total_cookies * revenue_per_cookie
def total_cost : ℝ := total_cookies * cost_per_cookie
def total_profit : ℝ := total_revenue - total_cost

-- Proof statement
theorem john_splits_profit_correctly : total_profit / amount_per_charity = 2 := by
  sorry

end john_splits_profit_correctly_l32_32983


namespace product_value_l32_32949

noncomputable def product_expression : ℝ :=
  ∏ k in (Finset.range 9).map (Nat.castAdd 2), (1 - 1/(k^2))

theorem product_value : product_expression = 0.55 :=
by
  sorry

end product_value_l32_32949


namespace range_of_numbers_l32_32652

theorem range_of_numbers (a b c : ℕ) (h_mean : (a + b + c) / 3 = 4) (h_median : b = 4) (h_smallest : a = 1) :
  c - a = 6 :=
sorry

end range_of_numbers_l32_32652


namespace largest_divisor_of_n_given_n_squared_divisible_by_72_l32_32970

theorem largest_divisor_of_n_given_n_squared_divisible_by_72 (n : ℕ) (h1 : 0 < n) (h2 : 72 ∣ n^2) :
  ∃ q, q = 12 ∧ q ∣ n :=
by
  sorry

end largest_divisor_of_n_given_n_squared_divisible_by_72_l32_32970


namespace marla_drive_time_l32_32335

theorem marla_drive_time (x : ℕ) (h_total : x + 70 + x = 110) : x = 20 :=
sorry

end marla_drive_time_l32_32335


namespace max_sum_is_38_l32_32727

-- Definition of the problem variables and conditions
def number_set : Set ℤ := {2, 3, 8, 9, 14, 15}
variable (a b c d e : ℤ)

-- Conditions translated to Lean
def condition1 : Prop := b = c
def condition2 : Prop := a = d

-- Sum condition to find maximum sum
def max_combined_sum : ℤ := a + b + e

theorem max_sum_is_38 : 
  ∃ a b c d e, 
    {a, b, c, d, e} ⊆ number_set ∧
    b = c ∧ 
    a = d ∧ 
    a + b + e = 38 :=
sorry

end max_sum_is_38_l32_32727


namespace tim_out_of_pocket_expense_l32_32759

noncomputable def total_cost (mri_cost doc_cost seen_fee : ℝ) := mri_cost + doc_cost + seen_fee
noncomputable def insurance_covered (total insurance_percentage : ℝ) := total * insurance_percentage
noncomputable def out_of_pocket (total insurance_covered : ℝ) := total - insurance_covered

theorem tim_out_of_pocket_expense :
  let mri_cost := 1200 in
  let doc_rate := 300 in
  let exam_duration_hr := 0.5 in
  let seen_fee := 150 in
  let insurance_percentage := 0.8 in
  let doc_cost := doc_rate * exam_duration_hr in
  let total := total_cost mri_cost doc_cost seen_fee in
  let covered := insurance_covered total insurance_percentage in
  let out_of_pocket_expense := out_of_pocket total covered in
  out_of_pocket_expense = 300 :=
by
  sorry

end tim_out_of_pocket_expense_l32_32759


namespace prime_solution_exists_l32_32812

theorem prime_solution_exists (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) :
  p^2 + 1 = 74 * (q^2 + r^2) → (p = 31 ∧ q = 2 ∧ r = 3) :=
by
  sorry

end prime_solution_exists_l32_32812


namespace probability_none_solve_l32_32637

theorem probability_none_solve (a b c : ℕ) (ha : 0 < a ∧ a < 10)
                               (hb : 0 < b ∧ b < 10)
                               (hc : 0 < c ∧ c < 10)
                               (P_A : ℚ := 1 / a)
                               (P_B : ℚ := 1 / b)
                               (P_C : ℚ := 1 / c)
                               (H : (1 - (1 / a)) * (1 - (1 / b)) * (1 - (1 / c)) = 8 / 15) :
                               -- Conclusion: The probability that none of them solve the problem is 8/15
                               (1 - (1 / a)) * (1 - (1 / b)) * (1 - (1 / c)) = 8 / 15 :=
sorry

end probability_none_solve_l32_32637


namespace two_digit_numbers_solution_l32_32299

theorem two_digit_numbers_solution :
  ∀ (N : ℕ), (∃ (x y : ℕ), (N = 10 * x + y) ∧ (x < 10) ∧ (y < 10) ∧ 4 * x + 2 * y = N / 2) →
    (N = 32 ∨ N = 64 ∨ N = 96) := 
by
  sorry

end two_digit_numbers_solution_l32_32299


namespace cube_volume_l32_32185

theorem cube_volume (A : ℝ) (hA : A = 294) : ∃ (V : ℝ), V = 343 :=
by
  sorry

end cube_volume_l32_32185


namespace max_value_f_l32_32221

noncomputable def f (x : ℝ) : ℝ := (4 * x - 4 * x^3) / (1 + 2 * x^2 + x^4)

theorem max_value_f : ∃ x : ℝ, (f x = 1) ∧ (∀ y : ℝ, f y ≤ 1) :=
sorry

end max_value_f_l32_32221


namespace no_solution_implies_b_positive_l32_32451

theorem no_solution_implies_b_positive (a b : ℝ) :
  (¬ ∃ x y : ℝ, y = x^2 + a * x + b ∧ x = y^2 + a * y + b) → b > 0 :=
by
  sorry

end no_solution_implies_b_positive_l32_32451


namespace expandProduct_l32_32556

theorem expandProduct (x : ℝ) : 4 * (x - 5) * (x + 8) = 4 * x^2 + 12 * x - 160 := 
by 
  sorry

end expandProduct_l32_32556


namespace find_b_in_expression_l32_32090

theorem find_b_in_expression
  (a b : ℚ)
  (h : (1 + Real.sqrt 3)^5 = a + b * Real.sqrt 3) :
  b = 44 :=
sorry

end find_b_in_expression_l32_32090


namespace quadratic_polynomial_coefficients_l32_32689

theorem quadratic_polynomial_coefficients (a b : ℝ)
  (h1 : 2 * a - 1 - b = 0)
  (h2 : 5 * a + b - 13 = 0) :
  a^2 + b^2 = 13 := 
by 
  sorry

end quadratic_polynomial_coefficients_l32_32689


namespace fill_half_jar_in_18_days_l32_32001

-- Define the doubling condition and the days required to fill half the jar
variable (area : ℕ → ℕ)
variable (doubling : ∀ t, area (t + 1) = 2 * area t)
variable (full_jar : area 19 = 2^19)
variable (half_jar : area 18 = 2^18)

theorem fill_half_jar_in_18_days :
  ∃ n, n = 18 ∧ area n = 2^18 :=
by {
  -- The proof is omitted, but we state the goal
  sorry
}

end fill_half_jar_in_18_days_l32_32001


namespace roots_polynomial_sum_l32_32458

theorem roots_polynomial_sum (p q r s : ℂ)
  (h_roots : (p, q, r, s) ∈ { (p, q, r, s) | (Polynomial.eval p (Polynomial.C 1 * Polynomial.X ^ 4 + Polynomial.C 10 * Polynomial.X ^ 3 + Polynomial.C 20 * Polynomial.X ^ 2 + Polynomial.C 15 * Polynomial.X + Polynomial.C 6) = 0) ∧
                                      (Polynomial.eval q (Polynomial.C 1 * Polynomial.X ^ 4 + Polynomial.C 10 * Polynomial.X ^ 3 + Polynomial.C 20 * Polynomial.X ^ 2 + Polynomial.C 15 * Polynomial.X + Polynomial.C 6) = 0) ∧
                                      (Polynomial.eval r (Polynomial.C 1 * Polynomial.X ^ 4 + Polynomial.C 10 * Polynomial.X ^ 3 + Polynomial.C 20 * Polynomial.X ^ 2 + Polynomial.C 15 * Polynomial.X + Polynomial.C 6) = 0) ∧
                                      (Polynomial.eval s (Polynomial.C 1 * Polynomial.X ^ 4 + Polynomial.C 10 * Polynomial.X ^ 3 + Polynomial.C 20 * Polynomial.X ^ 2 + Polynomial.C 15 * Polynomial.X + Polynomial.C 6) = 0) })
  (h_sum_two_at_a_time : p*q + p*r + p*s + q*r + q*s + r*s = 20)
  (h_product : p*q*r*s = 6) :
  1 / (p * q) + 1 / (p * r) + 1 / (p * s) + 1 / (q * r) + 1 / (q * s) + 1 / (r * s) = 10 / 3 := by
  sorry

end roots_polynomial_sum_l32_32458


namespace number_of_dimes_l32_32591

theorem number_of_dimes (d q : ℕ) (h₁ : 10 * d + 25 * q = 580) (h₂ : d = q + 10) : d = 23 := 
by 
  sorry

end number_of_dimes_l32_32591


namespace max_sinA_sinB_l32_32308

-- Definition and theorem statement
theorem max_sinA_sinB (A B C : ℝ) (hABC : A + B + C = π) (hC : C = π / 2) :
  ∃ m, m = 1 / 2 ∧ (∀ A B, A + B = π / 2 → sin A * sin B ≤ m) :=
by
  sorry

end max_sinA_sinB_l32_32308


namespace total_ladybugs_correct_l32_32609

noncomputable def total_ladybugs (with_spots : ℕ) (without_spots : ℕ) : ℕ :=
  with_spots + without_spots

theorem total_ladybugs_correct :
  total_ladybugs 12170 54912 = 67082 :=
by
  unfold total_ladybugs
  rfl

end total_ladybugs_correct_l32_32609


namespace petrol_price_increase_l32_32745

variable (P C : ℝ)

/- The original price of petrol is P per unit, and the user consumes C units of petrol.
   The new consumption after a 28.57142857142857% reduction is (5/7) * C units.
   The expenditure remains constant, i.e., P * C = P' * (5/7) * C.
-/
theorem petrol_price_increase (h : P * C = (P * (7/5)) * (5/7) * C) :
  (P * (7/5) - P) / P * 100 = 40 :=
by
  sorry

end petrol_price_increase_l32_32745


namespace ratio_elyse_to_rick_l32_32415

-- Define the conditions
def Elyse_initial_gum : ℕ := 100
def Shane_leftover_gum : ℕ := 14
def Shane_chewed_gum : ℕ := 11

-- Theorem stating the ratio of pieces Elyse gave to Rick to the total number of pieces Elyse had
theorem ratio_elyse_to_rick :
  let total_gum := Elyse_initial_gum
  let Shane_initial_gum := Shane_leftover_gum + Shane_chewed_gum
  let Rick_initial_gum := 2 * Shane_initial_gum
  let Elyse_given_to_Rick := Rick_initial_gum
  (Elyse_given_to_Rick : ℚ) / total_gum = 1 / 2 :=
by
  sorry

end ratio_elyse_to_rick_l32_32415


namespace find_even_increasing_l32_32055

theorem find_even_increasing (f : ℝ → ℝ) :
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ x y : ℝ, 0 < x → x < y → 0 < y → f x < f y) ↔
  f = (fun x => 3 * x^2 - 1) ∨ f = (fun x => 2^|x|) :=
by
  sorry

end find_even_increasing_l32_32055


namespace polynomial_value_at_4_l32_32766

def f (x : ℝ) : ℝ := 9 + 15 * x - 8 * x^2 - 20 * x^3 + 6 * x^4 + 3 * x^5

theorem polynomial_value_at_4 :
  f 4 = 3269 :=
by
  sorry

end polynomial_value_at_4_l32_32766


namespace parallel_lines_a_eq_neg1_l32_32819

theorem parallel_lines_a_eq_neg1 (a : ℝ) :
  ∀ (x y : ℝ), 
    (x + a * y + 6 = 0) ∧ ((a - 2) * x + 3 * y + 2 * a = 0) →
    (-1 / a = - (a - 2) / 3) → 
    a = -1 :=
by
  sorry

end parallel_lines_a_eq_neg1_l32_32819


namespace inverse_proportion_l32_32332

theorem inverse_proportion (x y : ℝ) (k : ℝ) (h1 : x^2 * y^4 = k)
  (h2 : 6^2 * 2^4 = k) (hy : y = 4) : x^2 = 2.25 :=
by
  sorry

end inverse_proportion_l32_32332


namespace tony_water_intake_l32_32018

theorem tony_water_intake (yesterday water_two_days_ago : ℝ) 
    (h1 : yesterday = 48) (h2 : yesterday = 0.96 * water_two_days_ago) :
    water_two_days_ago = 50 :=
by
  sorry

end tony_water_intake_l32_32018


namespace smallest_possible_n_l32_32198

theorem smallest_possible_n
  (n : ℕ)
  (d : ℕ)
  (h_d_pos : d > 0)
  (h_profit : 10 * n - 30 = 100)
  (h_cost_multiple : ∃ k, d = 2 * n * k) :
  n = 13 :=
by {
  sorry
}

end smallest_possible_n_l32_32198


namespace line_equation_problem_l32_32941

theorem line_equation_problem
  (P : ℝ × ℝ)
  (h1 : (P.1 + P.2 - 2 = 0) ∧ (P.1 - P.2 + 4 = 0))
  (l : ℝ × ℝ → Prop)
  (h2 : ∀ A B : ℝ × ℝ, l A → l B → (∃ k, B.2 - A.2 = k * (B.1 - A.1)))
  (h3 : ∀ Q : ℝ × ℝ, l Q → (3 * Q.1 - 2 * Q.2 + 4 = 0)) :
  l P ↔ 3 * P.1 - 2 * P.2 + 9 = 0 := 
sorry

end line_equation_problem_l32_32941


namespace cube_difference_l32_32248

variable (x : ℝ)

theorem cube_difference (h : x - 1/x = 5) : x^3 - 1/x^3 = 120 := by
  sorry

end cube_difference_l32_32248


namespace sum_of_k_values_with_distinct_integer_solutions_l32_32368

theorem sum_of_k_values_with_distinct_integer_solutions : 
  ∑ k in {k : ℤ | ∃ p q : ℤ, p ≠ q ∧ 3 * p * q + (-k) * (p + q) + 12 = 0}, k = 0 :=
by
  sorry

end sum_of_k_values_with_distinct_integer_solutions_l32_32368


namespace gcd_of_expression_l32_32942

noncomputable def gcd_expression (a b c d : ℤ) : ℤ :=
  (a - b) * (b - c) * (c - d) * (d - a) * (b - d) * (a - c)

theorem gcd_of_expression : 
  ∀ (a b c d : ℤ), ∃ (k : ℤ), gcd_expression a b c d = 12 * k :=
sorry

end gcd_of_expression_l32_32942


namespace sin_product_exact_value_l32_32157

open Real

theorem sin_product_exact_value :
  (sin (10 * π / 180) * sin (30 * π / 180) * sin (50 * π / 180) * sin (70 * π / 180)) = 1 / 16 := 
by
  sorry

end sin_product_exact_value_l32_32157


namespace factorization_of_z6_minus_64_l32_32417

theorem factorization_of_z6_minus_64 :
  ∀ (z : ℝ), (z^6 - 64) = (z - 2) * (z^2 + 2*z + 4) * (z + 2) * (z^2 - 2*z + 4) := 
by
  intros z
  sorry

end factorization_of_z6_minus_64_l32_32417


namespace positive_rationals_in_S_l32_32316

variable (S : Set ℚ)

-- Conditions
axiom closed_under_addition (a b : ℚ) (ha : a ∈ S) (hb : b ∈ S) : a + b ∈ S
axiom closed_under_multiplication (a b : ℚ) (ha : a ∈ S) (hb : b ∈ S) : a * b ∈ S
axiom zero_rule : ∀ r : ℚ, r ∈ S ∨ -r ∈ S ∨ r = 0

-- Prove that S is the set of positive rational numbers
theorem positive_rationals_in_S : S = {r : ℚ | 0 < r} :=
by
  sorry

end positive_rationals_in_S_l32_32316


namespace seashells_remainder_l32_32408

theorem seashells_remainder :
  let derek := 58
  let emily := 73
  let fiona := 31 
  let total_seashells := derek + emily + fiona
  total_seashells % 10 = 2 :=
by
  sorry

end seashells_remainder_l32_32408


namespace sum_of_remainders_mod_53_l32_32629

theorem sum_of_remainders_mod_53 (x y z : ℕ) (hx : x % 53 = 36) (hy : y % 53 = 15) (hz : z % 53 = 7) : 
  (x + y + z) % 53 = 5 :=
by
  sorry

end sum_of_remainders_mod_53_l32_32629


namespace bridge_length_correct_l32_32656

noncomputable def length_of_bridge : ℝ :=
  let train_length := 110 -- in meters
  let train_speed_kmh := 72 -- in km/hr
  let crossing_time := 14.248860091192705 -- in seconds
  let speed_in_mps := train_speed_kmh * (1000 / 3600)
  let distance := speed_in_mps * crossing_time
  distance - train_length

theorem bridge_length_correct :
  length_of_bridge = 174.9772018238541 := by
  sorry

end bridge_length_correct_l32_32656


namespace find_s_range_l32_32575

variables {a b c s t y1 y2 : ℝ}

-- Conditions
def is_vertex (a b c s t : ℝ) : Prop := ∀ x : ℝ, (a * x^2 + b * x + c = a * (x - s)^2 + t)

def passes_points (a b c y1 y2 : ℝ) : Prop := 
  (a * (-2)^2 + b * (-2) + c = y1) ∧ (a * 4^2 + b * 4 + c = y2)

def valid_constants (a y1 y2 t : ℝ) : Prop := 
  (a ≠ 0) ∧ (y1 > y2) ∧ (y2 > t)

-- Theorem
theorem find_s_range {a b c s t y1 y2 : ℝ}
  (hv : is_vertex a b c s t)
  (hp : passes_points a b c y1 y2)
  (vc : valid_constants a y1 y2 t) : 
  s > 1 ∧ s ≠ 4 :=
sorry -- Proof skipped

end find_s_range_l32_32575


namespace max_c_val_l32_32282

theorem max_c_val (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h1 : 2 * a * b = 2 * a + b) 
  (h2 : a * b * c = 2 * a + b + c) :
  c ≤ 4 :=
sorry

end max_c_val_l32_32282


namespace avg_class_weight_is_46_67_l32_32173

-- Define the total number of students in section A
def num_students_a : ℕ := 40

-- Define the average weight of students in section A
def avg_weight_a : ℚ := 50

-- Define the total number of students in section B
def num_students_b : ℕ := 20

-- Define the average weight of students in section B
def avg_weight_b : ℚ := 40

-- Calculate the total weight of section A
def total_weight_a : ℚ := num_students_a * avg_weight_a

-- Calculate the total weight of section B
def total_weight_b : ℚ := num_students_b * avg_weight_b

-- Calculate the total weight of the entire class
def total_weight_class : ℚ := total_weight_a + total_weight_b

-- Calculate the total number of students in the entire class
def total_students_class : ℕ := num_students_a + num_students_b

-- Calculate the average weight of the entire class
def avg_weight_class : ℚ := total_weight_class / total_students_class

-- Theorem to prove
theorem avg_class_weight_is_46_67 :
  avg_weight_class = 46.67 := sorry

end avg_class_weight_is_46_67_l32_32173


namespace calc_154_1836_minus_54_1836_l32_32932

-- Statement of the problem in Lean 4
theorem calc_154_1836_minus_54_1836 : 154 * 1836 - 54 * 1836 = 183600 :=
by
  sorry

end calc_154_1836_minus_54_1836_l32_32932


namespace distance_from_left_focal_to_line_l32_32836

noncomputable def ellipse_eq_line_dist : Prop :=
  let a := 2
  let b := Real.sqrt 3
  let c := 1
  let x₀ := -1
  let y₀ := 0
  let x₁ := 0
  let y₁ := Real.sqrt 3
  let x₂ := 1
  let y₂ := 0
  
  -- Equation of the line derived from the upper vertex and right focal point
  let m := -(y₁ - y₂) / (x₁ - x₂)
  let line_eq (x y : ℝ) := (Real.sqrt 3 * x + y - Real.sqrt 3 = 0)
  
  -- Distance formula from point to line
  let d := abs (Real.sqrt 3 * x₀ + y₀ - Real.sqrt 3) / Real.sqrt ((Real.sqrt 3)^2 + 1^2)

  -- The assertion that the distance is √3
  d = Real.sqrt 3

theorem distance_from_left_focal_to_line : ellipse_eq_line_dist := 
  sorry  -- Proof is omitted as per the instruction

end distance_from_left_focal_to_line_l32_32836


namespace value_of_a2022_l32_32353

theorem value_of_a2022 (a : ℕ → ℤ) (h : ∀ (n k : ℕ), 1 ≤ n ∧ n ≤ 2022 ∧ 1 ≤ k ∧ k ≤ 2022 → a n - a k ≥ (n^3 : ℤ) - (k^3 : ℤ)) (ha1011 : a 1011 = 0) : 
  a 2022 = 7246031367 := 
by
  sorry

end value_of_a2022_l32_32353


namespace gumball_difference_l32_32210

theorem gumball_difference :
  ∀ C : ℕ, 19 ≤ (29 + C) / 3 ∧ (29 + C) / 3 ≤ 25 →
  (46 - 28) = 18 :=
by
  intros C h
  sorry

end gumball_difference_l32_32210


namespace total_cost_of_vacation_l32_32172

variable (C : ℚ)

def cost_per_person_divided_among_3 := C / 3
def cost_per_person_divided_among_4 := C / 4
def per_person_difference := 40

theorem total_cost_of_vacation
  (h : cost_per_person_divided_among_3 C - cost_per_person_divided_among_4 C = per_person_difference) :
  C = 480 := by
  sorry

end total_cost_of_vacation_l32_32172


namespace pairs_of_different_positives_l32_32938

def W (x : ℕ) : ℕ := x^4 - 3 * x^3 + 5 * x^2 - 9 * x

theorem pairs_of_different_positives (a b : ℕ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b)
  (hW : W a = W b) : (a, b) = (1, 2) ∨ (a, b) = (2, 1) := 
sorry

end pairs_of_different_positives_l32_32938


namespace commute_time_l32_32537

theorem commute_time (d s1 s2 : ℝ) (h1 : s1 = 45) (h2 : s2 = 30) (h3 : d = 18) : (d / s1 + d / s2 = 1) :=
by
  -- Definitions and assumptions
  rw [h1, h2, h3]
  -- Total time calculation
  exact sorry

end commute_time_l32_32537


namespace alpha_plus_beta_l32_32214

noncomputable def alpha_beta (α β : ℝ) : Prop :=
  ∀ x : ℝ, ((x - α) / (x + β)) = ((x^2 - 54 * x + 621) / (x^2 + 42 * x - 1764))

theorem alpha_plus_beta : ∃ α β : ℝ, α + β = 86 ∧ alpha_beta α β :=
by
  sorry

end alpha_plus_beta_l32_32214


namespace problem1_problem2_problem3_problem4_l32_32551

theorem problem1 : 23 + (-16) - (-7) = 14 := by
  sorry

theorem problem2 : (3/4 - 7/8 - 5/12) * (-24) = 13 := by
  sorry

theorem problem3 : (7/4 - 7/8 - 7/12) / (-7/8) + (-7/8) / (7/4 - 7/8 - 7/12) = -(10/3) := by
  sorry

theorem problem4 : -1 ^ 4 - (1 - 0.5) * (1/3) * (2 - (-3) ^ 2) = 1/6 := by 
  sorry

end problem1_problem2_problem3_problem4_l32_32551


namespace xiao_ming_excellent_score_probability_l32_32699

theorem xiao_ming_excellent_score_probability :
  let P_M : ℝ := 0.5
  let P_L : ℝ := 0.3
  let P_E := 1 - P_M - P_L
  P_E = 0.2 :=
by
  let P_M : ℝ := 0.5
  let P_L : ℝ := 0.3
  let P_E := 1 - P_M - P_L
  sorry

end xiao_ming_excellent_score_probability_l32_32699


namespace gcd_36_60_l32_32982

theorem gcd_36_60 : Nat.gcd 36 60 = 12 := by
  sorry

end gcd_36_60_l32_32982


namespace tony_water_drink_l32_32021

theorem tony_water_drink (W : ℝ) (h : W - 0.04 * W = 48) : W = 50 :=
sorry

end tony_water_drink_l32_32021


namespace square_is_six_l32_32704

def represents_digit (square triangle circle : ℕ) : Prop :=
  square < 10 ∧ triangle < 10 ∧ circle < 10 ∧
  square ≠ triangle ∧ square ≠ circle ∧ triangle ≠ circle

theorem square_is_six :
  ∃ (square triangle circle : ℕ), represents_digit square triangle circle ∧ triangle = 1 ∧ circle = 9 ∧ (square + triangle + 100 * 1 + 10 * 9) = 117 ∧ square = 6 :=
by {
  sorry
}

end square_is_six_l32_32704


namespace arrangement_count_5x5_chessboard_pieces_l32_32839

theorem arrangement_count_5x5_chessboard_pieces : 
  ∃ (f : Fin 5 → Fin 5), 
    ∀ x y : Fin 5, x ≠ y → f x ≠ f y ∧ (∃ (g : Fin 5 → Fin 5),
    ∀ u v : Fin 5, u ≠ v → g u ≠ g v ∧ ∃ (arrangements : Finset ((Fin 5) → (Fin 5))), arrangements.card = 1200) :=
begin
  sorry
end

end arrangement_count_5x5_chessboard_pieces_l32_32839


namespace count_squares_below_graph_l32_32876

theorem count_squares_below_graph : 
  (count_squares_below (12 * x + 240 * y = 2880) (first_quadrant) = 1315) :=
sorry

end count_squares_below_graph_l32_32876


namespace loaf_slices_l32_32217

theorem loaf_slices (S : ℕ) (T : ℕ) : 
  (S - 7 = 2 * T + 3) ∧ (S ≥ 20) → S = 20 :=
by
  sorry

end loaf_slices_l32_32217


namespace charge_per_action_figure_l32_32714

-- Definitions according to given conditions
def cost_of_sneakers : ℕ := 90
def saved_amount : ℕ := 15
def num_action_figures : ℕ := 10
def left_after_purchase : ℕ := 25

-- Theorem to prove the charge per action figure
theorem charge_per_action_figure : 
  (cost_of_sneakers - saved_amount + left_after_purchase) / num_action_figures = 10 :=
by 
  sorry

end charge_per_action_figure_l32_32714


namespace similar_triangle_perimeter_l32_32801

theorem similar_triangle_perimeter
  (a b c : ℕ)
  (h1 : a = 7)
  (h2 : b = 7)
  (h3 : c = 12)
  (similar_triangle_longest_side : ℕ)
  (h4 : similar_triangle_longest_side = 36)
  (h5 : c * similar_triangle_longest_side = 12 * 36) :
  ∃ P : ℕ, P = 78 := by
  sorry

end similar_triangle_perimeter_l32_32801


namespace number_of_dogs_is_correct_l32_32976

variable (D C B : ℕ)
variable (k : ℕ)

def validRatio (D C B : ℕ) : Prop := D = 7 * k ∧ C = 7 * k ∧ B = 8 * k
def totalDogsAndBunnies (D B : ℕ) : Prop := D + B = 330
def correctNumberOfDogs (D : ℕ) : Prop := D = 154

theorem number_of_dogs_is_correct (D C B k : ℕ) 
  (hRatio : validRatio D C B k)
  (hTotal : totalDogsAndBunnies D B) :
  correctNumberOfDogs D :=
by
  sorry

end number_of_dogs_is_correct_l32_32976


namespace problem_statement_l32_32822

noncomputable def f (a x : ℝ) : ℝ := a^x + a^(-x)

theorem problem_statement (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) (h₂ : f a 1 = 3) :
  f a 0 + f a 1 + f a 2 = 12 :=
sorry

end problem_statement_l32_32822


namespace cubic_identity_l32_32251

theorem cubic_identity (x : ℝ) (h : x - (1 / x) = 5) : x^3 - (1 / x^3) = 140 := 
  sorry

end cubic_identity_l32_32251


namespace part_I_part_II_l32_32823

noncomputable def f (x a : ℝ) : ℝ := |2 * x + 1| - |x - a|

-- Problem (I)
theorem part_I (x : ℝ) : 
  (f x 4) > 2 ↔ (x < -7 ∨ x > 5 / 3) :=
sorry

-- Problem (II)
theorem part_II (a : ℝ) :
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 3 → f x a ≥ |x - 4|) ↔ -1 ≤ a ∧ a ≤ 5 :=
sorry

end part_I_part_II_l32_32823


namespace boat_speed_ratio_l32_32633

variable (B S : ℝ)

theorem boat_speed_ratio (h : 1 / (B - S) = 2 * (1 / (B + S))) : B / S = 3 := 
by
  sorry

end boat_speed_ratio_l32_32633


namespace Luke_spent_per_week_l32_32728

-- Definitions based on the conditions
def money_from_mowing := 9
def money_from_weeding := 18
def total_money := money_from_mowing + money_from_weeding
def weeks := 9
def amount_spent_per_week := total_money / weeks

-- The proof statement
theorem Luke_spent_per_week :
  amount_spent_per_week = 3 := 
  sorry

end Luke_spent_per_week_l32_32728


namespace number_of_slices_left_l32_32472

-- Conditions
def total_slices : ℕ := 8
def slices_given_to_joe_and_darcy : ℕ := total_slices / 2
def slices_given_to_carl : ℕ := total_slices / 4

-- Question: How many slices were left?
def slices_left : ℕ := total_slices - (slices_given_to_joe_and_darcy + slices_given_to_carl)

-- Proof statement to demonstrate that slices_left == 2
theorem number_of_slices_left : slices_left = 2 := by
  sorry

end number_of_slices_left_l32_32472


namespace divisor_of_p_l32_32320

theorem divisor_of_p (p q r s : ℕ) (h₁ : Nat.gcd p q = 30) (h₂ : Nat.gcd q r = 45) (h₃ : Nat.gcd r s = 75) (h₄ : 120 < Nat.gcd s p) (h₅ : Nat.gcd s p < 180) : 5 ∣ p := 
sorry

end divisor_of_p_l32_32320


namespace value_of_g_neg2_l32_32627

-- Define the function g as given in the conditions
def g (x : ℝ) : ℝ := x^2 - 3 * x + 1

-- Statement of the problem: Prove that g(-2) = 11
theorem value_of_g_neg2 : g (-2) = 11 := by
  sorry

end value_of_g_neg2_l32_32627


namespace henry_time_proof_l32_32063

-- Define the time Dawson took to run the first leg of the course
def dawson_time : ℝ := 38

-- Define the average time they took to run a leg of the course
def average_time : ℝ := 22.5

-- Define the time Henry took to run the second leg of the course
def henry_time : ℝ := 7

-- Prove that Henry took 7 seconds to run the second leg
theorem henry_time_proof : 
  (dawson_time + henry_time) / 2 = average_time :=
by
  -- This is where the proof would go
  sorry

end henry_time_proof_l32_32063


namespace general_term_formula_l32_32234

theorem general_term_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (hS : ∀ n, S n = 2 * a n - 1) : 
  ∀ n, a n = 2^(n-1) := 
by
  sorry

end general_term_formula_l32_32234


namespace women_more_than_men_l32_32512

def men (W : ℕ) : ℕ := (5 * W) / 11

theorem women_more_than_men (M W : ℕ) (h1 : M + W = 16) (h2 : M = (5 * W) / 11) : W - M = 6 :=
by
  sorry

end women_more_than_men_l32_32512


namespace cube_volume_from_surface_area_l32_32189

theorem cube_volume_from_surface_area (A : ℝ) (hA : A = 294) : ∃ V : ℝ, V = 343 :=
by
  let s := real.sqrt (A / 6)
  have h_s : s = 7 := sorry
  let V := s^3
  have h_V : V = 343 := sorry
  exact ⟨V, h_V⟩

end cube_volume_from_surface_area_l32_32189


namespace sequence_solution_l32_32826

theorem sequence_solution (a : ℕ → ℝ) (h1 : a 1 = 1/2)
    (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 1 / (n^2 + n)) : ∀ n : ℕ, n ≥ 1 → a n = 3/2 - 1/n :=
by
  intros n hn
  sorry

end sequence_solution_l32_32826


namespace age_of_15th_student_l32_32611

theorem age_of_15th_student (avg_age_all : ℝ) (avg_age_4 : ℝ) (avg_age_10 : ℝ) 
  (total_students : ℕ) (group_4_students : ℕ) (group_10_students : ℕ) 
  (h1 : avg_age_all = 15) (h2 : avg_age_4 = 14) (h3 : avg_age_10 = 16) 
  (h4 : total_students = 15) (h5 : group_4_students = 4) (h6 : group_10_students = 10) : 
  ∃ x : ℝ, x = 9 := 
by 
  sorry

end age_of_15th_student_l32_32611


namespace units_digit_of_2_to_the_10_l32_32376

theorem units_digit_of_2_to_the_10 : ∃ d : ℕ, (d < 10) ∧ (2^10 % 10 = d) ∧ (d == 4) :=
by {
  -- sorry to skip the proof
  sorry
}

end units_digit_of_2_to_the_10_l32_32376


namespace william_farm_tax_l32_32810

theorem william_farm_tax :
  let total_tax_collected := 3840
  let william_land_percentage := 0.25
  william_land_percentage * total_tax_collected = 960 :=
by sorry

end william_farm_tax_l32_32810


namespace sixth_inequality_l32_32859

theorem sixth_inequality :
  (1 + 1/2^2 + 1/3^2 + 1/4^2 + 1/5^2 + 1/6^2 + 1/7^2) < 13/7 :=
  sorry

end sixth_inequality_l32_32859


namespace y_intercept_of_line_l32_32774

theorem y_intercept_of_line (x y : ℝ) (eq : 2 * x - 3 * y = 6) (hx : x = 0) : y = -2 := 
by
  sorry

end y_intercept_of_line_l32_32774


namespace score_on_fourth_board_l32_32064

theorem score_on_fourth_board 
  (score1 score2 score3 score4 : ℕ)
  (h1 : score1 = 30)
  (h2 : score2 = 38)
  (h3 : score3 = 41)
  (total_score : score1 + score2 = 2 * score4) :
  score4 = 34 := by
  sorry

end score_on_fourth_board_l32_32064


namespace how_many_bottles_did_maria_drink_l32_32990

-- Define the conditions as variables and constants.
variable (x : ℕ)
def initial_bottles : ℕ := 14
def bought_bottles : ℕ := 45
def total_bottles_after_drinking_and_buying : ℕ := 51

-- The goal is to prove that Maria drank 8 bottles of water.
theorem how_many_bottles_did_maria_drink (h : initial_bottles - x + bought_bottles = total_bottles_after_drinking_and_buying) : x = 8 :=
by
  sorry

end how_many_bottles_did_maria_drink_l32_32990


namespace solve_for_x_l32_32742

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ -1 then x + 2 
  else if x < 2 then x^2 
  else 2 * x

theorem solve_for_x (x : ℝ) : f x = 3 ↔ x = Real.sqrt 3 := by
  sorry

end solve_for_x_l32_32742


namespace final_sum_l32_32722

def Q (x : ℝ) : ℝ := 2 * x^2 - 4 * x - 4

noncomputable def probability_condition_holds : ℝ :=
  by sorry

theorem final_sum :
  let m := 1
  let n := 1
  let o := 1
  let p := 0
  let q := 8
  (m + n + o + p + q) = 11 :=
  by
    sorry

end final_sum_l32_32722


namespace transformed_ellipse_equation_l32_32576

namespace EllipseTransformation

open Real

def original_ellipse (x y : ℝ) : Prop :=
  x^2 / 6 + y^2 = 1

def transformation (x' y' x y : ℝ) : Prop :=
  x' = 1 / 2 * x ∧ y' = 2 * y

theorem transformed_ellipse_equation (x y x' y' : ℝ) 
  (h : original_ellipse x y) (tr : transformation x' y' x y) :
  2 * x'^2 / 3 + y'^2 / 4 = 1 :=
by 
  sorry

end EllipseTransformation

end transformed_ellipse_equation_l32_32576


namespace evaluate_f_at_3_l32_32683

theorem evaluate_f_at_3 (f : ℤ → ℤ) (h : ∀ x : ℤ, f (x + 1) = 2 * x + 3) : f 3 = 7 :=
by
  -- proof goes here
  sorry

end evaluate_f_at_3_l32_32683


namespace nth_equation_l32_32149

theorem nth_equation (n : ℕ) : (2 * n + 1)^2 - (2 * n - 1)^2 = 8 * n := by
  sorry

end nth_equation_l32_32149


namespace hyperbola_eccentricity_l32_32084

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 + (a^2) / (b^2))

theorem hyperbola_eccentricity {b : ℝ} (hb_pos : b > 0)
  (h_area : b = 1) :
  eccentricity 1 b = Real.sqrt 3 :=
by
  sorry

end hyperbola_eccentricity_l32_32084


namespace min_value_expression_l32_32322

theorem min_value_expression (α β : ℝ) : 
  (3 * Real.cos α + 4 * Real.sin β - 7) ^ 2 + (3 * Real.sin α + 4 * Real.cos β - 12) ^ 2 ≥ 36 :=
sorry

end min_value_expression_l32_32322


namespace mary_initial_stickers_l32_32465

theorem mary_initial_stickers (stickers_remaining : ℕ) 
  (front_page_stickers : ℕ) (other_page_stickers : ℕ) 
  (num_other_pages : ℕ) 
  (h1 : front_page_stickers = 3)
  (h2 : other_page_stickers = 7 * num_other_pages)
  (h3 : num_other_pages = 6)
  (h4 : stickers_remaining = 44) :
  ∃ initial_stickers : ℕ, initial_stickers = front_page_stickers + other_page_stickers + stickers_remaining ∧ initial_stickers = 89 :=
by
  sorry

end mary_initial_stickers_l32_32465


namespace polynomial_value_at_4_l32_32765

def f (x : ℝ) : ℝ := 9 + 15 * x - 8 * x^2 - 20 * x^3 + 6 * x^4 + 3 * x^5

theorem polynomial_value_at_4 :
  f 4 = 3269 :=
by
  sorry

end polynomial_value_at_4_l32_32765


namespace original_height_of_ball_l32_32340

theorem original_height_of_ball (h : ℝ) : 
  (h + 2 * (0.5 * h) + 2 * ((0.5)^2 * h) = 200) -> 
  h = 800 / 9 := 
by
  sorry

end original_height_of_ball_l32_32340


namespace both_solve_correctly_l32_32091

-- Define the probabilities of making an error for individuals A and B
variables (a b : ℝ)

-- Assuming a and b are probabilities, they must lie in the interval [0, 1]
axiom a_prob : 0 ≤ a ∧ a ≤ 1
axiom b_prob : 0 ≤ b ∧ b ≤ 1

-- Define the event that both individuals solve the problem correctly
theorem both_solve_correctly : (1 - a) * (1 - b) = (1 - a) * (1 - b) :=
by
  sorry

end both_solve_correctly_l32_32091


namespace quadratic_odd_coefficients_l32_32865

theorem quadratic_odd_coefficients (a b c : ℕ) (a_nonzero : a ≠ 0) (h : ∃ r : ℚ, r^2 * a + r * b + c = 0) :
  ¬ (odd a ∧ odd b ∧ odd c) :=
by
  sorry

end quadratic_odd_coefficients_l32_32865


namespace number_of_girls_l32_32586

-- Definitions from the problem conditions
def ratio_girls_boys (g b : ℕ) : Prop := 4 * b = 3 * g
def total_students (g b : ℕ) : Prop := g + b = 56

-- The proof statement
theorem number_of_girls (g b k : ℕ) (hg : 4 * k = g) (hb : 3 * k = b) (hr : ratio_girls_boys g b) (ht : total_students g b) : g = 32 :=
by sorry

end number_of_girls_l32_32586


namespace sum_of_k_distinct_integer_roots_l32_32375

theorem sum_of_k_distinct_integer_roots : 
  (∑ k in {k : ℤ | ∃ p q : ℤ, p ≠ q ∧ p * q = 4 ∧ k = 3 * (p + q)}, k) = 0 := 
by
  sorry

end sum_of_k_distinct_integer_roots_l32_32375


namespace arithmetic_geometric_seq_l32_32145

theorem arithmetic_geometric_seq (A B C D : ℤ) (h_arith_seq : A + C = 2 * B)
  (h_geom_seq : B * D = C * C) (h_frac : 7 * B = 3 * C) (h_posB : B > 0)
  (h_posC : C > 0) (h_posD : D > 0) : A + B + C + D = 76 :=
sorry

end arithmetic_geometric_seq_l32_32145


namespace joe_first_lift_weight_l32_32382

theorem joe_first_lift_weight (x y : ℕ) (h1 : x + y = 1500) (h2 : 2 * x = y + 300) : x = 600 :=
by
  sorry

end joe_first_lift_weight_l32_32382


namespace abs_sum_eq_3_given_condition_l32_32432

theorem abs_sum_eq_3_given_condition (m n p : ℤ)
  (h : |m - n|^3 + |p - m|^5 = 1) :
  |p - m| + |m - n| + 2 * |n - p| = 3 :=
sorry

end abs_sum_eq_3_given_condition_l32_32432


namespace baby_guppies_calculation_l32_32926

-- Define the problem in Lean
theorem baby_guppies_calculation :
  ∀ (initial_guppies first_sighting two_days_gups total_guppies_after_two_days : ℕ), 
  initial_guppies = 7 →
  first_sighting = 36 →
  total_guppies_after_two_days = 52 →
  total_guppies_after_two_days = initial_guppies + first_sighting + two_days_gups →
  two_days_gups = 9 :=
by
  intros initial_guppies first_sighting two_days_gups total_guppies_after_two_days
  intros h_initial h_first h_total h_eq
  sorry

end baby_guppies_calculation_l32_32926


namespace scalene_triangle_third_side_l32_32006

theorem scalene_triangle_third_side (a b c : ℕ) (h : (a - 3)^2 + (b - 2)^2 = 0) : 
  a = 3 ∧ b = 2 → c = 2 ∨ c = 3 ∨ c = 4 := 
by {
  sorry
}

end scalene_triangle_third_side_l32_32006


namespace even_product_when_eight_cards_drawn_l32_32674

theorem even_product_when_eight_cards_drawn :
  ∀ (s : Finset ℕ), (∀ n ∈ s, n ∈ Finset.range 15) →
  s.card ≥ 8 →
  (∃ m ∈ s, Even m) :=
by
  sorry

end even_product_when_eight_cards_drawn_l32_32674


namespace balloons_popped_on_ground_l32_32336

def max_rate : Nat := 2
def max_time : Nat := 30
def zach_rate : Nat := 3
def zach_time : Nat := 40
def total_filled_balloons : Nat := 170

theorem balloons_popped_on_ground :
  (max_rate * max_time + zach_rate * zach_time) - total_filled_balloons = 10 :=
by
  sorry

end balloons_popped_on_ground_l32_32336


namespace two_person_subcommittees_from_six_l32_32583

theorem two_person_subcommittees_from_six :
  (Nat.choose 6 2) = 15 := by
  sorry

end two_person_subcommittees_from_six_l32_32583


namespace evaluate_expression_l32_32037

theorem evaluate_expression : 2009 * (2007 / 2008) + (1 / 2008) = 2008 := 
by 
  sorry

end evaluate_expression_l32_32037


namespace smallest_mn_sum_l32_32708

theorem smallest_mn_sum (m n : ℕ) (h : 3 * n ^ 3 = 5 * m ^ 2) : m + n = 60 :=
sorry

end smallest_mn_sum_l32_32708


namespace payment_option1_payment_option2_cost_effective_option_most_cost_effective_plan_l32_32181

variable (x : ℕ)
variable (hx : x > 10)

noncomputable def option1_payment (x : ℕ) : ℕ := 200 * x + 8000
noncomputable def option2_payment (x : ℕ) : ℕ := 180 * x + 9000

theorem payment_option1 (x : ℕ) (hx : x > 10) : option1_payment x = 200 * x + 8000 :=
by sorry

theorem payment_option2 (x : ℕ) (hx : x > 10) : option2_payment x = 180 * x + 9000 :=
by sorry

theorem cost_effective_option (x : ℕ) (hx : x > 10) (h30 : x = 30) : option1_payment 30 < option2_payment 30 :=
by sorry

theorem most_cost_effective_plan (h30 : x = 30) : (10000 + 3600 = 13600) :=
by sorry

end payment_option1_payment_option2_cost_effective_option_most_cost_effective_plan_l32_32181


namespace find_n_l32_32419

theorem find_n (n : ℤ) : -180 ≤ n ∧ n ≤ 180 ∧ (Real.sin (n * Real.pi / 180) = Real.cos (690 * Real.pi / 180)) → n = 60 :=
by
  intro h
  sorry

end find_n_l32_32419


namespace mary_cut_roses_l32_32016

theorem mary_cut_roses (initial_roses add_roses total_roses : ℕ) (h1 : initial_roses = 6) (h2 : total_roses = 16) (h3 : total_roses = initial_roses + add_roses) : add_roses = 10 :=
by
  sorry

end mary_cut_roses_l32_32016


namespace correct_calculation_result_l32_32902

theorem correct_calculation_result (x : ℤ) (h : 4 * x + 16 = 32) : (x / 4) + 16 = 17 := by
  sorry

end correct_calculation_result_l32_32902


namespace first_vessel_milk_water_l32_32518

variable (V : ℝ)

def vessel_ratio (v1 v2 : ℝ) : Prop := 
  v1 / v2 = 3 / 5

def vessel1_milk_water_ratio (milk water : ℝ) : Prop :=
  milk / water = 1 / 2

def vessel2_milk_water_ratio (milk water : ℝ) : Prop :=
  milk / water = 3 / 2

def mix_ratio (milk1 water1 milk2 water2 : ℝ) : Prop :=
  (milk1 + milk2) / (water1 + water2) = 1

theorem first_vessel_milk_water (V : ℝ) (v1 v2 : ℝ) (m1 w1 m2 w2 : ℝ)
  (hv : vessel_ratio v1 v2)
  (hv1 : vessel1_milk_water_ratio m1 w1)
  (hv2 : vessel2_milk_water_ratio m2 w2)
  (hmix : mix_ratio m1 w1 m2 w2) :
  vessel1_milk_water_ratio m1 w1 :=
  sorry

end first_vessel_milk_water_l32_32518


namespace contrapositive_l32_32740

variable {α : Type} (M : α → Prop) (a b : α)

theorem contrapositive (h : (M a → ¬ M b)) : (M b → ¬ M a) := 
by
  sorry

end contrapositive_l32_32740


namespace average_age_constant_l32_32529

theorem average_age_constant 
  (average_age_3_years_ago : ℕ) 
  (number_of_members_3_years_ago : ℕ) 
  (baby_age_today : ℕ) 
  (number_of_members_today : ℕ) 
  (H1 : average_age_3_years_ago = 17) 
  (H2 : number_of_members_3_years_ago = 5) 
  (H3 : baby_age_today = 2) 
  (H4 : number_of_members_today = 6) : 
  average_age_3_years_ago = (average_age_3_years_ago * number_of_members_3_years_ago + baby_age_today + 3 * number_of_members_3_years_ago) / number_of_members_today := 
by sorry

end average_age_constant_l32_32529


namespace amy_total_tickets_l32_32664

theorem amy_total_tickets (initial_tickets additional_tickets : ℕ) (h_initial : initial_tickets = 33) (h_additional : additional_tickets = 21) : 
  initial_tickets + additional_tickets = 54 := 
by 
  sorry

end amy_total_tickets_l32_32664


namespace general_term_l32_32311

noncomputable def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, n > 0 → a (n + 1) = 2 * a n / (2 + a n)

theorem general_term (a : ℕ → ℝ) (h : seq a) : ∀ n : ℕ, n > 0 → a n = 2 / (n + 1) :=
by
sorry

end general_term_l32_32311


namespace no_solutions_l32_32385

theorem no_solutions (x y : ℕ) (hx : x ≥ 1) (hy : y ≥ 1) : ¬ (x^5 = y^2 + 4) :=
by sorry

end no_solutions_l32_32385


namespace saltwater_solution_l32_32033

theorem saltwater_solution (x : ℝ) (h1 : ∃ v : ℝ, v = x ∧ v * 0.2 = 0.20 * x)
(h2 : 3 / 4 * x = 3 / 4 * x)
(h3 : ∃ v' : ℝ, v' = 3 / 4 * x + 6 + 12)
(h4 : (0.20 * x + 12) / (3 / 4 * x + 18) = 1 / 3) : x = 120 :=
by 
  sorry

end saltwater_solution_l32_32033


namespace rectangle_probability_no_shaded_square_l32_32644

theorem rectangle_probability_no_shaded_square :
  let n := 1003 * 2005 in
  let m := 1003 ^ 2 in
  2 by 2005 rectangle
  ∧ middle unit square of each row is shaded
  ∧ rect chosen at random
  → 1 - m / n = 1002 / 2005 :=
by
  sorry

end rectangle_probability_no_shaded_square_l32_32644


namespace root_analysis_l32_32350

noncomputable def root1 (a : ℝ) : ℝ :=
2 * a + 2 * Real.sqrt (a^2 - 3 * a + 2)

noncomputable def root2 (a : ℝ) : ℝ :=
2 * a - 2 * Real.sqrt (a^2 - 3 * a + 2)

noncomputable def derivedRoot (a : ℝ) : ℝ :=
(3 * a - 2) / a

theorem root_analysis (a : ℝ) (ha : a > 0) :
( (2/3 ≤ a ∧ a < 1) ∨ (2 < a) → (root1 a ≥ 0 ∧ root2 a ≥ 0)) ∧
( 0 < a ∧ a < 2/3 → (derivedRoot a < 0 ∧ root1 a ≥ 0)) :=
sorry

end root_analysis_l32_32350


namespace parabola_directrix_l32_32482

noncomputable def directrix_value (a : ℝ) : ℝ := -1 / (4 * a)

theorem parabola_directrix (a : ℝ) (h : directrix_value a = 2) : a = -1 / 8 :=
by
  sorry

end parabola_directrix_l32_32482


namespace magnitude_range_l32_32582

noncomputable def vector_a (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)
noncomputable def vector_b : ℝ × ℝ := (Real.sqrt 3, -1)

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem magnitude_range (θ : ℝ) : 
  0 ≤ (vector_magnitude (2 • vector_a θ - vector_b)) ∧ (vector_magnitude (2 • vector_a θ - vector_b)) ≤ 4 := 
sorry

end magnitude_range_l32_32582


namespace cube_volume_l32_32196

-- Define the given condition: The surface area of the cube
def surface_area (A : ℕ) := A = 294

-- The key proposition we need to prove using the given condition
theorem cube_volume (s : ℕ) (A : ℕ) (V : ℕ) 
  (area_condition : surface_area A)
  (side_length_condition : s ^ 2 = A / 6) 
  (volume_condition : V = s ^ 3) : 
  V = 343 := by
  sorry

end cube_volume_l32_32196


namespace estimated_total_fish_l32_32032

-- Let's define the conditions first
def total_fish_marked := 100
def second_catch_total := 200
def marked_in_second_catch := 5

-- The variable representing the total number of fish in the pond
variable (x : ℕ)

-- The theorem stating that given the conditions, the total number of fish is 4000
theorem estimated_total_fish
  (h1 : total_fish_marked = 100)
  (h2 : second_catch_total = 200)
  (h3 : marked_in_second_catch = 5)
  (h4 : (marked_in_second_catch : ℝ) / second_catch_total = (total_fish_marked : ℝ) / x) :
  x = 4000 := 
sorry

end estimated_total_fish_l32_32032


namespace chosen_numbers_divisibility_l32_32817

theorem chosen_numbers_divisibility (n : ℕ) (S : Finset ℕ) (hS : S.card > (n + 1) / 2) :
  ∃ a ∈ S, ∃ b ∈ S, a ≠ b ∧ a ∣ b :=
by sorry

end chosen_numbers_divisibility_l32_32817


namespace sum_of_real_solutions_l32_32946

theorem sum_of_real_solutions :
  ∀ x : ℝ, (x - 3) / (x^2 + 5 * x + 2) = (x - 6) / (x^2 - 11 * x) →
  (∃ r1 r2 : ℝ, r1 + r2 = 46 / 13) :=
by
  sorry

end sum_of_real_solutions_l32_32946
