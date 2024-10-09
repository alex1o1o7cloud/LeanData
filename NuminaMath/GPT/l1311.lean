import Mathlib

namespace imaginary_unit_power_l1311_131145

-- Definition of the imaginary unit i
def imaginary_unit_i : ℂ := Complex.I

theorem imaginary_unit_power :
  (imaginary_unit_i ^ 2015) = -imaginary_unit_i := by
  sorry

end imaginary_unit_power_l1311_131145


namespace total_wheels_at_park_l1311_131186

-- Conditions as definitions
def number_of_adults := 6
def number_of_children := 15
def wheels_per_bicycle := 2
def wheels_per_tricycle := 3

-- To prove: total number of wheels = 57
theorem total_wheels_at_park : 
  (number_of_adults * wheels_per_bicycle) + (number_of_children * wheels_per_tricycle) = 57 :=
by
  sorry

end total_wheels_at_park_l1311_131186


namespace total_distance_is_correct_l1311_131171

def Jonathan_d : Real := 7.5

def Mercedes_d (J : Real) : Real := 2 * J

def Davonte_d (M : Real) : Real := M + 2

theorem total_distance_is_correct : 
  let J := Jonathan_d
  let M := Mercedes_d J
  let D := Davonte_d M
  M + D = 32 :=
by
  sorry

end total_distance_is_correct_l1311_131171


namespace time_to_cross_pole_is_2_5_l1311_131198

noncomputable def time_to_cross_pole : ℝ :=
  let length_of_train := 100 -- meters
  let speed_km_per_hr := 144 -- km/hr
  let speed_m_per_s := speed_km_per_hr * 1000 / 3600 -- converting speed to m/s
  length_of_train / speed_m_per_s

theorem time_to_cross_pole_is_2_5 :
  time_to_cross_pole = 2.5 :=
by
  -- The Lean proof will be written here.
  -- Placeholder for the formal proof.
  sorry

end time_to_cross_pole_is_2_5_l1311_131198


namespace martin_distance_l1311_131178

-- Define the given conditions
def speed : ℝ := 12.0
def time : ℝ := 6.0

-- State the theorem we want to prove
theorem martin_distance : speed * time = 72.0 := by
  sorry

end martin_distance_l1311_131178


namespace same_days_to_dig_scenario_l1311_131155

def volume (depth length breadth : ℝ) : ℝ :=
  depth * length * breadth

def days_to_dig (depth length breadth days : ℝ) : Prop :=
  ∃ (labors : ℝ), 
    (volume depth length breadth) * days = (volume 100 25 30) * 12

theorem same_days_to_dig_scenario :
  days_to_dig 75 20 50 12 :=
sorry

end same_days_to_dig_scenario_l1311_131155


namespace Cinderella_solves_l1311_131133

/--
There are three bags labeled as "Poppy", "Millet", and "Mixture". Each label is incorrect.
By inspecting one grain from the bag labeled as "Mixture", Cinderella can determine the exact contents of all three bags.
-/
theorem Cinderella_solves (bag_contents : String → String) (examined_grain : String) :
  (bag_contents "Mixture" = "Poppy" ∨ bag_contents "Mixture" = "Millet") →
  (∀ l, bag_contents l ≠ l) →
  (examined_grain = "Poppy" ∨ examined_grain = "Millet") →
  examined_grain = bag_contents "Mixture" →
  ∃ poppy_bag millet_bag mixture_bag : String,
    poppy_bag ≠ "Poppy" ∧ millet_bag ≠ "Millet" ∧ mixture_bag ≠ "Mixture" ∧
    bag_contents poppy_bag = "Poppy" ∧
    bag_contents millet_bag = "Millet" ∧
    bag_contents mixture_bag = "Mixture" :=
sorry

end Cinderella_solves_l1311_131133


namespace winning_votes_cast_l1311_131190

variable (V : ℝ) -- Total number of votes (real number)
variable (winner_votes_ratio : ℝ) -- Ratio for winner's votes
variable (votes_difference : ℝ) -- Vote difference due to winning

-- Conditions given
def election_conditions (V : ℝ) (winner_votes_ratio : ℝ) (votes_difference : ℝ) : Prop :=
  winner_votes_ratio = 0.54 ∧
  votes_difference = 288

-- Proof problem: Proving the number of votes cast to the winning candidate is 1944
theorem winning_votes_cast (V : ℝ) (winner_votes_ratio : ℝ) (votes_difference : ℝ) 
  (h : election_conditions V winner_votes_ratio votes_difference) :
  winner_votes_ratio * V = 1944 :=
by
  sorry

end winning_votes_cast_l1311_131190


namespace perfect_squares_less_than_500_ending_in_4_l1311_131117

theorem perfect_squares_less_than_500_ending_in_4 : 
  (∃ (squares : Finset ℕ), (∀ n ∈ squares, n < 500 ∧ (n % 10 = 4)) ∧ squares.card = 5) :=
by
  sorry

end perfect_squares_less_than_500_ending_in_4_l1311_131117


namespace function_bounds_l1311_131101

theorem function_bounds {a : ℝ} :
  (∀ x : ℝ, x > 0 → 4 - x^2 + a * Real.log x ≤ 3) → a = 2 :=
by
  sorry

end function_bounds_l1311_131101


namespace new_mixture_alcohol_percentage_l1311_131173

/-- 
Given: 
  - a solution with 15 liters containing 26% alcohol
  - 5 liters of water added to the solution
Prove:
  The percentage of alcohol in the new mixture is 19.5%
-/
theorem new_mixture_alcohol_percentage 
  (original_volume : ℝ) (original_percent_alcohol : ℝ) (added_water_volume : ℝ) :
  original_volume = 15 → 
  original_percent_alcohol = 26 →
  added_water_volume = 5 →
  (original_volume * (original_percent_alcohol / 100) / (original_volume + added_water_volume)) * 100 = 19.5 :=
by 
  intros h1 h2 h3
  sorry

end new_mixture_alcohol_percentage_l1311_131173


namespace sequence_geometric_l1311_131125

noncomputable def S (n : ℕ) (a : ℕ → ℝ) : ℝ :=
  3 * a n - 2

theorem sequence_geometric (a : ℕ → ℝ) (h : ∀ n, S n a = 3 * a n - 2) :
  ∀ n, a n = (3/2)^(n-1) :=
by
  intro n
  sorry

end sequence_geometric_l1311_131125


namespace sufficient_but_not_necessary_l1311_131165

theorem sufficient_but_not_necessary (x : ℝ) (h1 : x > 1 → x > 0) (h2 : ¬ (x > 0 → x > 1)) : 
  (x > 1 → x > 0) ∧ ¬ (x > 0 → x > 1) := 
by 
  sorry

end sufficient_but_not_necessary_l1311_131165


namespace average_marks_physics_chemistry_l1311_131166

theorem average_marks_physics_chemistry
  (P C M : ℕ)
  (h1 : (P + C + M) / 3 = 60)
  (h2 : (P + M) / 2 = 90)
  (h3 : P = 140) :
  (P + C) / 2 = 70 :=
by
  sorry

end average_marks_physics_chemistry_l1311_131166


namespace score_entered_twice_l1311_131164

theorem score_entered_twice (scores : List ℕ) (h : scores = [68, 74, 77, 82, 85, 90]) :
  ∃ (s : ℕ), s = 82 ∧ ∀ (entered : List ℕ), entered.length = 7 ∧ (∀ i, (List.take (i + 1) entered).sum % (i + 1) = 0) →
  (List.count (List.insertNth i 82 scores)) = 2 ∧ (∀ x, x ∈ scores.remove 82 → x ≠ s) :=
by
  sorry

end score_entered_twice_l1311_131164


namespace division_and_multiplication_result_l1311_131143

theorem division_and_multiplication_result :
  let num : ℝ := 6.5
  let divisor : ℝ := 6
  let multiplier : ℝ := 12
  num / divisor * multiplier = 13 :=
by
  sorry

end division_and_multiplication_result_l1311_131143


namespace hat_value_in_rice_l1311_131196

variables (f l r h : ℚ)

theorem hat_value_in_rice :
  (4 * f = 3 * l) →
  (l = 5 * r) →
  (5 * f = 7 * h) →
  h = (75 / 28) * r :=
by
  intros h1 h2 h3
  -- proof goes here
  sorry

end hat_value_in_rice_l1311_131196


namespace sequence_sum_l1311_131182

theorem sequence_sum (r x y : ℝ) (h1 : r = 1/4) 
  (h2 : x = 256 * r)
  (h3 : y = x * r) : x + y = 80 :=
by
  sorry

end sequence_sum_l1311_131182


namespace remainder_of_6_pow_50_mod_215_l1311_131134

theorem remainder_of_6_pow_50_mod_215 :
  (6 ^ 50) % 215 = 36 := 
sorry

end remainder_of_6_pow_50_mod_215_l1311_131134


namespace quadratic_inequality_solution_l1311_131109

-- Definition of the given conditions and the theorem to prove
theorem quadratic_inequality_solution 
  (a b c : ℝ)
  (h1 : a < 0)
  (h2 : ∀ x, ax^2 + bx + c < 0 ↔ x < -2 ∨ x > -1/2) :
  ∀ x, ax^2 - bx + c > 0 ↔ 1/2 < x ∧ x < 2 :=
by
  sorry

end quadratic_inequality_solution_l1311_131109


namespace solve_inequality_1_range_of_m_l1311_131181

noncomputable def f (x : ℝ) : ℝ := abs (x - 1)
noncomputable def g (x m : ℝ) : ℝ := -abs (x + 3) + m

theorem solve_inequality_1 : {x : ℝ | f x + x^2 - 1 > 0} = {x : ℝ | x > 1 ∨ x < 0} := sorry

theorem range_of_m (m : ℝ) (h : m > 4) : ∃ x : ℝ, f x < g x m := sorry

end solve_inequality_1_range_of_m_l1311_131181


namespace seq_arithmetic_l1311_131136

theorem seq_arithmetic (a : ℕ → ℕ) (h : ∀ p q : ℕ, a p + a q = a (p + q)) (h1 : a 1 = 2) :
  ∀ n : ℕ, a n = 2 * n :=
by
  sorry

end seq_arithmetic_l1311_131136


namespace like_terms_exponents_l1311_131144

theorem like_terms_exponents (m n : ℤ) 
  (h1 : m - 1 = 1) 
  (h2 : m + n = 3) : 
  m = 2 ∧ n = 1 :=
by 
  sorry

end like_terms_exponents_l1311_131144


namespace solution_l1311_131141

theorem solution (y : ℝ) (h : 6 * y^2 + 7 = 2 * y + 12) : (12 * y - 4)^2 = 128 :=
sorry

end solution_l1311_131141


namespace term_15_of_sequence_l1311_131111

theorem term_15_of_sequence : 
  ∃ (a : ℕ → ℝ), a 1 = 3 ∧ a 2 = 7 ∧ (∀ n, a (n + 1) = 21 / a n) ∧ a 15 = 3 :=
sorry

end term_15_of_sequence_l1311_131111


namespace xyz_value_l1311_131120

noncomputable def find_xyz (x y z : ℝ) 
  (h₁ : (x + y + z) * (x * y + x * z + y * z) = 45)
  (h₂ : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 14)
  (h₃ : (x + y + z)^2 = 25) : ℝ :=
  if (x * y * z = 31 / 3) then 31 / 3 else 0  -- This should hold with the given conditions

theorem xyz_value (x y z : ℝ)
  (h₁ : (x + y + z) * (x * y + x * z + y * z) = 45)
  (h₂ : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 14)
  (h₃ : (x + y + z)^2 = 25) :
  find_xyz x y z h₁ h₂ h₃ = 31 / 3 :=
by 
  sorry  -- The proof should demonstrate that find_xyz equals 31 / 3 given the conditions

end xyz_value_l1311_131120


namespace domain_of_fx_l1311_131119

theorem domain_of_fx :
  {x : ℝ | x ≥ 1 ∧ x^2 < 2} = {x : ℝ | 1 ≤ x ∧ x < Real.sqrt 2} := by
sorry

end domain_of_fx_l1311_131119


namespace max_sides_of_convex_polygon_l1311_131199

theorem max_sides_of_convex_polygon (n : ℕ) 
  (h_convex : n ≥ 3) 
  (h_angles: ∀ (a : Fin 4), (100 : ℝ) ≤ a.val) 
  : n ≤ 8 :=
sorry

end max_sides_of_convex_polygon_l1311_131199


namespace Dima_impossible_cut_l1311_131121

theorem Dima_impossible_cut (n : ℕ) 
  (h1 : n % 5 = 0) 
  (h2 : n % 7 = 0) 
  (h3 : n ≤ 200) : ¬(n % 6 = 0) :=
sorry

end Dima_impossible_cut_l1311_131121


namespace sum_of_decimals_l1311_131159

theorem sum_of_decimals : 5.47 + 2.58 + 1.95 = 10.00 := by
  sorry

end sum_of_decimals_l1311_131159


namespace pyramid_volume_l1311_131105

noncomputable def volume_of_pyramid (EF FG QE : ℝ) : ℝ :=
  (1 / 3) * (EF * FG) * QE

theorem pyramid_volume
  (EF FG QE : ℝ)
  (h1 : EF = 10)
  (h2 : FG = 5)
  (h3 : QE = 9) :
  volume_of_pyramid EF FG QE = 150 :=
by
  simp [volume_of_pyramid, h1, h2, h3]
  sorry

end pyramid_volume_l1311_131105


namespace polynomial_constant_l1311_131123

theorem polynomial_constant
  (P : Polynomial ℤ)
  (h : ∀ Q F G : Polynomial ℤ, P.comp Q = F * G → F.degree = 0 ∨ G.degree = 0) :
  P.degree = 0 :=
by sorry

end polynomial_constant_l1311_131123


namespace positive_number_is_25_l1311_131183

theorem positive_number_is_25 {a x : ℝ}
(h1 : x = (3 * a + 1)^2)
(h2 : x = (-a - 3)^2)
(h_sum : 3 * a + 1 + (-a - 3) = 0) :
x = 25 :=
sorry

end positive_number_is_25_l1311_131183


namespace division_to_fraction_fraction_to_division_mixed_to_improper_fraction_whole_to_fraction_l1311_131160

theorem division_to_fraction : (7 / 9) = 7 / 9 := by
  sorry

theorem fraction_to_division : 12 / 7 = 12 / 7 := by
  sorry

theorem mixed_to_improper_fraction : (3 + 5 / 8) = 29 / 8 := by
  sorry

theorem whole_to_fraction : 6 = 66 / 11 := by
  sorry

end division_to_fraction_fraction_to_division_mixed_to_improper_fraction_whole_to_fraction_l1311_131160


namespace initial_stops_eq_l1311_131148

-- Define the total number of stops S
def total_stops : ℕ := 7

-- Define the number of stops made after the initial deliveries
def additional_stops : ℕ := 4

-- Define the number of initial stops as a proof problem
theorem initial_stops_eq : total_stops - additional_stops = 3 :=
by
sorry

end initial_stops_eq_l1311_131148


namespace abs_neg_six_l1311_131118

theorem abs_neg_six : abs (-6) = 6 := by
  sorry

end abs_neg_six_l1311_131118


namespace minimum_value_x_plus_four_over_x_minimum_value_occurs_at_x_eq_2_l1311_131189

theorem minimum_value_x_plus_four_over_x (x : ℝ) (h : x ≥ 2) : 
  x + 4 / x ≥ 4 :=
by sorry

theorem minimum_value_occurs_at_x_eq_2 : ∀ (x : ℝ), x ≥ 2 → (x + 4 / x = 4 ↔ x = 2) :=
by sorry

end minimum_value_x_plus_four_over_x_minimum_value_occurs_at_x_eq_2_l1311_131189


namespace problem1_problem2_l1311_131185

-- Problem 1: Prove the simplification of an expression
theorem problem1 (x : ℝ) : (2*x + 1)^2 + x*(x-4) = 5*x^2 + 1 := 
by sorry

-- Problem 2: Prove the solution set for the system of inequalities
theorem problem2 (x : ℝ) (h1 : 3*x - 6 > 0) (h2 : (5 - x) / 2 < 1) : x > 3 := 
by sorry

end problem1_problem2_l1311_131185


namespace parabola_cubic_intersection_points_l1311_131167

def parabola (x : ℝ) : ℝ := 3 * x^2 - 12 * x - 15

def cubic (x : ℝ) : ℝ := x^3 - 6 * x^2 + 11 * x - 6

theorem parabola_cubic_intersection_points :
  ∃ (p1 p2 p3 : ℝ × ℝ),
    p1 = (-1, 0) ∧ p2 = (1, -24) ∧ p3 = (9, 162) ∧
    parabola p1.1 = p1.2 ∧ cubic p1.1 = p1.2 ∧
    parabola p2.1 = p2.2 ∧ cubic p2.1 = p2.2 ∧
    parabola p3.1 = p3.2 ∧ cubic p3.1 = p3.2 :=
by {
  -- This is the statement
  sorry
}

end parabola_cubic_intersection_points_l1311_131167


namespace mirror_area_l1311_131128

theorem mirror_area (frame_length frame_width frame_border_length : ℕ) (mirror_area : ℕ)
  (h_frame_length : frame_length = 100)
  (h_frame_width : frame_width = 130)
  (h_frame_border_length : frame_border_length = 15)
  (h_mirror_area : mirror_area = (frame_length - 2 * frame_border_length) * (frame_width - 2 * frame_border_length)) :
  mirror_area = 7000 := by 
    sorry

end mirror_area_l1311_131128


namespace longer_subsegment_length_l1311_131191

-- Define the given conditions and proof goal in Lean 4
theorem longer_subsegment_length {DE EF DF DG GF : ℝ} (h1 : 3 * EF < 4 * EF) (h2 : 4 * EF < 5 * EF)
  (ratio_condition : DE / EF = 4 / 5) (DF_length : DF = 12) :
  DG + GF = DF ∧ DE / EF = DG / GF ∧ GF = (5 * 12 / 9) :=
by
  sorry

end longer_subsegment_length_l1311_131191


namespace robert_elizabeth_age_difference_l1311_131195

theorem robert_elizabeth_age_difference 
  (patrick_age_1_5_times_robert : ∀ (robert_age : ℝ), ∃ (patrick_age : ℝ), patrick_age = 1.5 * robert_age)
  (elizabeth_born_after_richard : ∀ (richard_age : ℝ), ∃ (elizabeth_age : ℝ), elizabeth_age = richard_age - 7 / 12)
  (elizabeth_younger_by_4_5_years : ∀ (patrick_age : ℝ), ∃ (elizabeth_age : ℝ), elizabeth_age = patrick_age - 4.5)
  (robert_will_be_30_3_after_2_5_years : ∃ (robert_age_current : ℝ), robert_age_current = 30.3 - 2.5) :
  ∃ (years : ℤ) (months : ℤ), years = 9 ∧ months = 4 := by
  sorry

end robert_elizabeth_age_difference_l1311_131195


namespace right_pyramid_sum_edges_l1311_131102

theorem right_pyramid_sum_edges (a h : ℝ) (base_side slant_height : ℝ) :
  base_side = 12 ∧ slant_height = 15 ∧ ∀ x : ℝ, a = 117 :=
by
  sorry

end right_pyramid_sum_edges_l1311_131102


namespace rate_of_second_batch_l1311_131170

-- Define the problem statement
theorem rate_of_second_batch
  (rate_first : ℝ)
  (weight_first weight_second weight_total : ℝ)
  (rate_mixture : ℝ)
  (profit_multiplier : ℝ) 
  (total_selling_price : ℝ) :
  rate_first = 11.5 →
  weight_first = 30 →
  weight_second = 20 →
  weight_total = weight_first + weight_second →
  rate_mixture = 15.12 →
  profit_multiplier = 1.20 →
  total_selling_price = weight_total * rate_mixture →
  (rate_first * weight_first + (weight_second * x) * profit_multiplier = total_selling_price) →
  x = 14.25 :=
by
  intros
  sorry

end rate_of_second_batch_l1311_131170


namespace water_tank_capacity_l1311_131175

theorem water_tank_capacity (x : ℝ)
  (h1 : (2 / 3) * x - (1 / 3) * x = 20) : x = 60 := 
  sorry

end water_tank_capacity_l1311_131175


namespace isosceles_triangle_and_sin_cos_range_l1311_131174

theorem isosceles_triangle_and_sin_cos_range 
  (A B C : ℝ) (a b c : ℝ) 
  (hA_pos : 0 < A) (hA_lt_pi_div_2 : A < π / 2) (h_triangle : a * Real.cos B = b * Real.cos A) :
  (A = B ∧
  ∃ x, x = Real.sin B + Real.cos (A + π / 6) ∧ (1 / 2 < x ∧ x ≤ 1)) :=
by
  sorry

end isosceles_triangle_and_sin_cos_range_l1311_131174


namespace pies_count_l1311_131142

-- Definitions based on the conditions given in the problem
def strawberries_per_pie := 3
def christine_strawberries := 10
def rachel_strawberries := 2 * christine_strawberries

-- The theorem to prove
theorem pies_count : (christine_strawberries + rachel_strawberries) / strawberries_per_pie = 10 := by
  sorry

end pies_count_l1311_131142


namespace least_number_to_add_l1311_131176

theorem least_number_to_add (n : ℕ) (d : ℕ) (r : ℕ) : n = 1100 → d = 23 → r = n % d → (r ≠ 0) → (d - r) = 4 :=
by
  intros h₀ h₁ h₂ h₃
  simp [h₀, h₁] at h₂
  sorry

end least_number_to_add_l1311_131176


namespace quad_completion_l1311_131151

theorem quad_completion (a b c : ℤ) 
    (h : ∀ x : ℤ, 8 * x^2 - 48 * x - 128 = a * (x + b)^2 + c) : 
    a + b + c = -195 := 
by
  sorry

end quad_completion_l1311_131151


namespace adam_change_l1311_131188

theorem adam_change : 
  let amount : ℝ := 5.00
  let cost : ℝ := 4.28
  amount - cost = 0.72 :=
by
  -- proof goes here
  sorry

end adam_change_l1311_131188


namespace sum_of_d_and_e_l1311_131154

-- Define the original numbers and their sum
def original_first := 3742586
def original_second := 4829430
def correct_sum := 8572016

-- The given incorrect addition result
def given_sum := 72120116

-- Define the digits d and e
def d := 2
def e := 8

-- Define the correct adjusted sum if we replace d with e
def adjusted_first := 3782586
def adjusted_second := 4889430
def adjusted_sum := 8672016

-- State the final theorem
theorem sum_of_d_and_e : 
  (given_sum != correct_sum) → 
  (original_first + original_second = correct_sum) → 
  (adjusted_first + adjusted_second = adjusted_sum) → 
  (d + e = 10) :=
by
  sorry

end sum_of_d_and_e_l1311_131154


namespace measurable_length_l1311_131139

-- Definitions of lines, rays, and line segments

-- A line is infinitely long with no endpoints.
def isLine (l : Type) : Prop := ∀ x y : l, (x ≠ y)

-- A line segment has two endpoints and a finite length.
def isLineSegment (ls : Type) : Prop := ∃ a b : ls, a ≠ b ∧ ∃ d : ℝ, d > 0

-- A ray has one endpoint and is infinitely long.
def isRay (r : Type) : Prop := ∃ e : r, ∀ x : r, x ≠ e

-- Problem statement
theorem measurable_length (x : Type) : isLineSegment x → (∃ d : ℝ, d > 0) :=
by
  -- Proof is not required
  sorry

end measurable_length_l1311_131139


namespace third_box_nuts_l1311_131177

theorem third_box_nuts
  (A B C : ℕ)
  (h1 : A = B + C - 6)
  (h2 : B = A + C - 10) :
  C = 8 :=
by
  sorry

end third_box_nuts_l1311_131177


namespace parabola_equation_line_tangent_to_fixed_circle_l1311_131197

open Real

def parabola_vertex_origin_directrix (p : ℝ) : Prop :=
  ∀ x y : ℝ, y^2 = 2 * p * x ↔ x = -2

def point_on_directrix (l: ℝ) (t : ℝ) : Prop :=
  t ≠ 0 ∧ l = 3 * t - 1 / t

def point_on_y_axis (q : ℝ) (t : ℝ) : Prop :=
  q = 2 * t

theorem parabola_equation (p : ℝ) : 
  parabola_vertex_origin_directrix 4 →
  y^2 = 8 * x :=
by
  sorry

theorem line_tangent_to_fixed_circle (t : ℝ) (x0 : ℝ) (r : ℝ) :
  t ≠ 0 →
  point_on_directrix (-2) t →
  point_on_y_axis (2 * t) t →
  (x0 = 2 ∧ r = 2) →
  ∀ x y : ℝ, (x - 2)^2 + y^2 = 4 :=
by
  sorry

end parabola_equation_line_tangent_to_fixed_circle_l1311_131197


namespace find_r_in_geometric_sum_l1311_131150

theorem find_r_in_geometric_sum (S_n : ℕ → ℕ) (r : ℤ)
  (hSn : ∀ n : ℕ, S_n n = 2 * 3^n + r)
  (hgeo : ∀ n : ℕ, n ≥ 2 → S_n n - S_n (n - 1) = 4 * 3^(n - 1))
  (hn1 : S_n 1 = 6 + r) :
  r = -2 :=
by
  sorry

end find_r_in_geometric_sum_l1311_131150


namespace abs_eq_solution_l1311_131138

theorem abs_eq_solution (x : ℝ) (h : abs (x - 3) = abs (x + 2)) : x = 1 / 2 :=
sorry

end abs_eq_solution_l1311_131138


namespace line_of_intersection_l1311_131127

theorem line_of_intersection :
  ∀ (x y z : ℝ),
    (3 * x + 4 * y - 2 * z + 1 = 0) ∧ (2 * x - 4 * y + 3 * z + 4 = 0) →
    (∃ t : ℝ, x = -1 + 4 * t ∧ y = 1 / 2 - 13 * t ∧ z = -20 * t) :=
by
  intro x y z
  intro h
  cases h
  sorry

end line_of_intersection_l1311_131127


namespace salary_for_may_l1311_131157

theorem salary_for_may
  (J F M A May : ℝ)
  (h1 : J + F + M + A = 32000)
  (h2 : F + M + A + May = 34400)
  (h3 : J = 4100) :
  May = 6500 := 
by 
  sorry

end salary_for_may_l1311_131157


namespace pressure_relation_l1311_131172

-- Definitions from the problem statement
variables (Q Δu A k x P S ΔV V R T T₀ c_v n P₀ V₀ : ℝ)
noncomputable def first_law := Q = Δu + A
noncomputable def Δu_def := Δu = c_v * (T - T₀)
noncomputable def A_def := A = (k * x^2) / 2
noncomputable def spring_relation := k * x = P * S
noncomputable def volume_change := ΔV = S * x
noncomputable def volume_after_expansion := V = (n / (n - 1)) * (S * x)
noncomputable def ideal_gas_law := P * V = R * T
noncomputable def initial_state := P₀ * V₀ = R * T₀
noncomputable def expanded_state := P * (n * V₀) = R * T

-- Theorem to prove the final relation
theorem pressure_relation
  (h1: first_law Q Δu A)
  (h2: Δu_def Δu c_v T T₀)
  (h3: A_def A k x)
  (h4: spring_relation k x P S)
  (h5: volume_change ΔV S x)
  (h6: volume_after_expansion V S x n)
  (h7: ideal_gas_law P V R T)
  (h8: initial_state P₀ V₀ R T₀)
  (h9: expanded_state P R T n V₀)
  : P / P₀ = 1 / (n * (1 + ((n - 1) * R) / (2 * n * c_v))) :=
  sorry

end pressure_relation_l1311_131172


namespace non_congruent_triangles_with_perimeter_11_l1311_131114

theorem non_congruent_triangles_with_perimeter_11 : 
  ∀ (a b c : ℕ), a + b + c = 11 → a < b + c → b < a + c → c < a + b → 
  ∃! (a b c : ℕ), (a, b, c) = (2, 4, 5) ∨ (a, b, c) = (3, 4, 4) := 
by sorry

end non_congruent_triangles_with_perimeter_11_l1311_131114


namespace smallest_positive_integer_satisfying_condition_l1311_131131

-- Define the condition
def isConditionSatisfied (n : ℕ) : Prop :=
  (Real.sqrt n - Real.sqrt (n - 1) < 0.01) ∧ n > 0

-- State the theorem
theorem smallest_positive_integer_satisfying_condition :
  ∃ n : ℕ, isConditionSatisfied n ∧ (∀ m : ℕ, isConditionSatisfied m → n ≤ m) ∧ n = 2501 :=
by
  sorry

end smallest_positive_integer_satisfying_condition_l1311_131131


namespace verify_first_rope_length_l1311_131122

def length_first_rope : ℝ :=
  let rope1_len := 20
  let rope2_len := 2
  let rope3_len := 2
  let rope4_len := 2
  let rope5_len := 7
  let knots := 4
  let knot_loss := 1.2
  let total_len := 35
  rope1_len

theorem verify_first_rope_length : length_first_rope = 20 := by
  sorry

end verify_first_rope_length_l1311_131122


namespace jakes_weight_l1311_131110

theorem jakes_weight
  (J K : ℝ)
  (h1 : J - 8 = 2 * K)
  (h2 : J + K = 290) :
  J = 196 :=
by
  sorry

end jakes_weight_l1311_131110


namespace general_admission_tickets_l1311_131107

variable (x y : ℕ)

theorem general_admission_tickets (h1 : x + y = 525) (h2 : 4 * x + 6 * y = 2876) : y = 388 := by
  sorry

end general_admission_tickets_l1311_131107


namespace quadratic_has_real_root_l1311_131106

theorem quadratic_has_real_root (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ≤ -10 ∨ b ≥ 10) := sorry

end quadratic_has_real_root_l1311_131106


namespace food_left_after_bbqs_l1311_131126

noncomputable def mushrooms_bought : ℕ := 15
noncomputable def chicken_bought : ℕ := 20
noncomputable def beef_bought : ℕ := 10

noncomputable def mushrooms_consumed : ℕ := 5 * 3
noncomputable def chicken_consumed : ℕ := 4 * 2
noncomputable def beef_consumed : ℕ := 2 * 1

noncomputable def mushrooms_left : ℕ := mushrooms_bought - mushrooms_consumed
noncomputable def chicken_left : ℕ := chicken_bought - chicken_consumed
noncomputable def beef_left : ℕ := beef_bought - beef_consumed

noncomputable def total_food_left : ℕ := mushrooms_left + chicken_left + beef_left

theorem food_left_after_bbqs : total_food_left = 20 :=
  by
    unfold total_food_left mushrooms_left chicken_left beef_left
    unfold mushrooms_consumed chicken_consumed beef_consumed
    unfold mushrooms_bought chicken_bought beef_bought
    sorry

end food_left_after_bbqs_l1311_131126


namespace inequality_proof_l1311_131163

theorem inequality_proof (a b c d : ℕ) (h1 : a > b) (h2 : b > c) (h3 : c > d) (h4 : d > 0) (h5 : a * d = b * c) :
  (a - d) ^ 2 ≥ 4 * d + 8 := 
sorry

end inequality_proof_l1311_131163


namespace income_difference_l1311_131140

theorem income_difference
  (D W : ℝ)
  (hD : 0.08 * D = 800)
  (hW : 0.08 * W = 840) :
  (W + 840) - (D + 800) = 540 := 
  sorry

end income_difference_l1311_131140


namespace coin_flip_difference_l1311_131103

/-- The positive difference between the probability of a fair coin landing heads up
exactly 4 times out of 5 flips and the probability of a fair coin landing heads up
5 times out of 5 flips is 1/8. -/
theorem coin_flip_difference :
  (5 * (1 / 2) ^ 5) - ((1 / 2) ^ 5) = (1 / 8) :=
by
  sorry

end coin_flip_difference_l1311_131103


namespace otimes_property_l1311_131146

def otimes (a b : ℚ) : ℚ := (a^3) / b

theorem otimes_property : otimes (otimes 2 3) 4 - otimes 2 (otimes 3 4) = 80 / 27 := by
  sorry

end otimes_property_l1311_131146


namespace beef_weight_after_processing_l1311_131162

theorem beef_weight_after_processing
  (initial_weight : ℝ)
  (weight_loss_percentage : ℝ)
  (processed_weight : ℝ)
  (h1 : initial_weight = 892.31)
  (h2 : weight_loss_percentage = 0.35)
  (h3 : processed_weight = initial_weight * (1 - weight_loss_percentage)) :
  processed_weight = 579.5015 :=
by
  sorry

end beef_weight_after_processing_l1311_131162


namespace opposite_of_lime_is_black_l1311_131104

-- Given colors of the six faces
inductive Color
| Purple | Cyan | Magenta | Silver | Lime | Black

-- Hinged squares forming a cube
structure Cube :=
(top : Color) (bottom : Color) (front : Color) (back : Color) (left : Color) (right : Color)

-- Condition: Magenta is on the top
def magenta_top (c : Cube) : Prop := c.top = Color.Magenta

-- Problem statement: Prove the color opposite to Lime is Black
theorem opposite_of_lime_is_black (c : Cube) (HM : magenta_top c) (HL : c.front = Color.Lime)
    (HBackFace : c.back = Color.Black) : c.back = Color.Black := 
sorry

end opposite_of_lime_is_black_l1311_131104


namespace zero_points_C_exist_l1311_131153

theorem zero_points_C_exist (A B C : ℝ × ℝ) (hAB_dist : dist A B = 12) (h_perimeter : dist A B + dist A C + dist B C = 52)
    (h_area : abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2 = 100) : 
    false :=
by
  sorry

end zero_points_C_exist_l1311_131153


namespace max_dist_AC_l1311_131193

open Real EuclideanGeometry

variables (P A B C : ℝ × ℝ)
  (hPA : dist P A = 1)
  (hPB : dist P B = 1)
  (hPA_PB : dot_product (A.1 - P.1, A.2 - P.2) (B.1 - P.1, B.2 - P.2) = - 1 / 2)
  (hBC : dist B C = 1)

theorem max_dist_AC : ∃ C : ℝ × ℝ, dist A C ≤ dist A B + dist B C ∧ dist A C = sqrt 3 + 1 :=
by
  sorry

end max_dist_AC_l1311_131193


namespace max_value_ratio_l1311_131147

/-- Define the conditions on function f and variables x and y. -/
def conditions (f : ℝ → ℝ) (x y : ℝ) :=
  (∀ x, f (-x) + f x = 0) ∧
  (∀ x1 x2, x1 < x2 → f x1 < f x2) ∧
  f (x^2 - 6 * x) + f (y^2 - 4 * y + 12) ≤ 0

/-- The maximum value of (y - 2) / x under the given conditions. -/
theorem max_value_ratio (f : ℝ → ℝ) (x y : ℝ) (cond : conditions f x y) :
  (y - 2) / x ≤ (Real.sqrt 2) / 4 :=
sorry

end max_value_ratio_l1311_131147


namespace ratio_of_boys_l1311_131169

theorem ratio_of_boys (p : ℝ) (h : p = (3 / 5) * (1 - p)) 
  : p = 3 / 8 := 
by
  sorry

end ratio_of_boys_l1311_131169


namespace least_number_subtracted_l1311_131156

theorem least_number_subtracted {
  x : ℕ
} : 
  (∀ (m : ℕ), m ∈ [5, 9, 11] → (997 - x) % m = 3) → x = 4 :=
by
  sorry

end least_number_subtracted_l1311_131156


namespace find_m_for_positive_integer_x_l1311_131194

theorem find_m_for_positive_integer_x :
  ∃ (m : ℤ), (2 * m * x - 8 = (m + 2) * x) → ∀ (x : ℤ), x > 0 → m = 3 ∨ m = 4 ∨ m = 6 ∨ m = 10 :=
sorry

end find_m_for_positive_integer_x_l1311_131194


namespace storks_equal_other_birds_l1311_131135

-- Definitions of initial numbers of birds
def initial_sparrows := 2
def initial_crows := 1
def initial_storks := 3
def initial_egrets := 0

-- Birds arriving initially
def sparrows_arrived := 1
def crows_arrived := 3
def storks_arrived := 6
def egrets_arrived := 4

-- Birds leaving after 15 minutes
def sparrows_left := 2
def crows_left := 0
def storks_left := 0
def egrets_left := 1

-- Additional birds arriving after 30 minutes
def additional_sparrows := 0
def additional_crows := 4
def additional_storks := 3
def additional_egrets := 0

-- Final counts
def final_sparrows := initial_sparrows + sparrows_arrived - sparrows_left + additional_sparrows
def final_crows := initial_crows + crows_arrived - crows_left + additional_crows
def final_storks := initial_storks + storks_arrived - storks_left + additional_storks
def final_egrets := initial_egrets + egrets_arrived - egrets_left + additional_egrets

def total_other_birds := final_sparrows + final_crows + final_egrets

-- Theorem statement
theorem storks_equal_other_birds : final_storks - total_other_birds = 0 := by
  sorry

end storks_equal_other_birds_l1311_131135


namespace speed_of_boat_in_still_water_l1311_131115

theorem speed_of_boat_in_still_water
    (speed_stream : ℝ)
    (distance_downstream : ℝ)
    (distance_upstream : ℝ)
    (t : ℝ)
    (x : ℝ)
    (h1 : speed_stream = 10)
    (h2 : distance_downstream = 80)
    (h3 : distance_upstream = 40)
    (h4 : t = distance_downstream / (x + speed_stream))
    (h5 : t = distance_upstream / (x - speed_stream)) :
  x = 30 :=
by sorry

end speed_of_boat_in_still_water_l1311_131115


namespace barbara_total_candies_l1311_131116

-- Condition: Barbara originally has 9 candies.
def C1 := 9

-- Condition: Barbara buys 18 more candies.
def C2 := 18

-- Question (proof problem): Prove that the total number of candies Barbara has is 27.
theorem barbara_total_candies : C1 + C2 = 27 := by
  -- Proof steps are not required, hence using sorry.
  sorry

end barbara_total_candies_l1311_131116


namespace friends_in_group_l1311_131180

theorem friends_in_group : 
  ∀ (total_chicken_wings cooked_wings additional_wings chicken_wings_per_person : ℕ), 
    cooked_wings = 8 →
    additional_wings = 10 →
    chicken_wings_per_person = 6 →
    total_chicken_wings = cooked_wings + additional_wings →
    total_chicken_wings / chicken_wings_per_person = 3 :=
by
  intros total_chicken_wings cooked_wings additional_wings chicken_wings_per_person hcooked hadditional hperson htotal
  sorry

end friends_in_group_l1311_131180


namespace numbers_sum_and_difference_l1311_131192

variables (a b : ℝ)

theorem numbers_sum_and_difference (h : a / b = -1) : a + b = 0 ∧ (a - b = 2 * b ∨ a - b = -2 * b) :=
by {
  sorry
}

end numbers_sum_and_difference_l1311_131192


namespace length_of_field_l1311_131149

-- Define the conditions and given facts.
def double_length (w l : ℝ) : Prop := l = 2 * w
def pond_area (l w : ℝ) : Prop := 49 = 1/8 * (l * w)

-- Define the main statement that incorporates the given conditions and expected result.
theorem length_of_field (w l : ℝ) (h1 : double_length w l) (h2 : pond_area l w) : l = 28 := by
  sorry

end length_of_field_l1311_131149


namespace verify_graphical_method_l1311_131137

variable {R : Type} [LinearOrderedField R]

/-- Statement of the mentioned conditions -/
def poly (a b c d x : R) : R := a * x^3 + b * x^2 + c * x + d

/-- The main theorem stating the graphical method validity -/
theorem verify_graphical_method (a b c d x0 EJ : R) (h : 0 < a) (h1 : 0 < b) (h2 : 0 < c) (h3 : 0 < d) (h4 : 0 < x0) (h5 : x0 < 1)
: EJ = poly a b c d x0 := by sorry

end verify_graphical_method_l1311_131137


namespace shortest_distance_l1311_131184

-- The initial position of the cowboy.
def initial_position : ℝ × ℝ := (-2, -6)

-- The position of the cabin relative to the cowboy's initial position.
def cabin_position : ℝ × ℝ := (10, -15)

-- The equation of the stream flowing due northeast.
def stream_equation : ℝ → ℝ := id  -- y = x

-- Function to calculate the distance between two points (x1, y1) and (x2, y2).
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

-- Calculate the reflection point of C over y = x.
def reflection_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

-- Main proof statement: shortest distance the cowboy can travel.
theorem shortest_distance : distance initial_position (reflection_point initial_position) +
                            distance (reflection_point initial_position) cabin_position = 8 +
                            Real.sqrt 545 :=
by
  sorry

end shortest_distance_l1311_131184


namespace evaluate_fraction_l1311_131161

theorem evaluate_fraction : (8 / 29) - (5 / 87) = (19 / 87) := sorry

end evaluate_fraction_l1311_131161


namespace bacteria_colony_growth_l1311_131130

theorem bacteria_colony_growth (n : ℕ) : 
  (∀ m: ℕ, 4 * 3^m ≤ 500 → m < n) → n = 5 :=
by
  sorry

end bacteria_colony_growth_l1311_131130


namespace molecular_weight_6_moles_C4H8O2_is_528_624_l1311_131132

-- Define the atomic weights of Carbon, Hydrogen, and Oxygen.
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

-- Define the molecular formula of C4H8O2.
def num_C_atoms : ℕ := 4
def num_H_atoms : ℕ := 8
def num_O_atoms : ℕ := 2

-- Define the number of moles of C4H8O2.
def num_moles_C4H8O2 : ℝ := 6

-- Define the molecular weight of one mole of C4H8O2.
def molecular_weight_C4H8O2 : ℝ :=
  (num_C_atoms * atomic_weight_C) +
  (num_H_atoms * atomic_weight_H) +
  (num_O_atoms * atomic_weight_O)

-- The total weight of 6 moles of C4H8O2.
def total_weight_6_moles_C4H8O2 : ℝ :=
  num_moles_C4H8O2 * molecular_weight_C4H8O2

-- Theorem stating that the molecular weight of 6 moles of C4H8O2 is 528.624 grams.
theorem molecular_weight_6_moles_C4H8O2_is_528_624 :
  total_weight_6_moles_C4H8O2 = 528.624 :=
by
  -- Proof is omitted.
  sorry

end molecular_weight_6_moles_C4H8O2_is_528_624_l1311_131132


namespace total_loaves_served_l1311_131112

-- Definitions based on the conditions provided
def wheat_bread_loaf : ℝ := 0.2
def white_bread_loaf : ℝ := 0.4

-- Statement that needs to be proven
theorem total_loaves_served : wheat_bread_loaf + white_bread_loaf = 0.6 := 
by
  sorry

end total_loaves_served_l1311_131112


namespace directrix_of_parabola_l1311_131113

theorem directrix_of_parabola (x y : ℝ) : 
  (x^2 = - (1/8) * y) → (y = 1/32) :=
sorry

end directrix_of_parabola_l1311_131113


namespace Asya_Petya_l1311_131108

theorem Asya_Petya (a b : ℕ) (ha : 100 ≤ a ∧ a < 1000) (hb : 100 ≤ b ∧ b < 1000) 
  (h : 1000 * a + b = 7 * a * b) : a = 143 ∧ b = 143 :=
by
  sorry

end Asya_Petya_l1311_131108


namespace quadratic_parabola_equation_l1311_131152

theorem quadratic_parabola_equation :
  ∃ (a b c : ℝ), 
    (∀ x y, y = 3 * x^2 - 6 * x + 5 → (x - 1)*(x - 1) = (x - 1)^2) ∧ -- Original vertex condition and standard form
    (∀ x y, y = -x - 2 → a = 2) ∧ -- Given intersection point condition
    (∀ x y, y = -3 * (x - 1)^2 + 2 → y = -3 * (x - 1)^2 + b ∧ y = -4) → -- Vertex unchanged and direction reversed
    (a, b, c) = (-3, 6, -4) := -- Resulting equation coefficients
sorry

end quadratic_parabola_equation_l1311_131152


namespace octal_to_decimal_l1311_131168

theorem octal_to_decimal : (1 * 8^3 + 7 * 8^2 + 4 * 8^1 + 3 * 8^0) = 995 :=
by
  sorry

end octal_to_decimal_l1311_131168


namespace new_rate_of_commission_l1311_131179

theorem new_rate_of_commission 
  (R1 : ℝ) (R1_eq : R1 = 0.04) 
  (slump_percentage : ℝ) (slump_percentage_eq : slump_percentage = 0.20000000000000007)
  (income_unchanged : ∀ (B B_new : ℝ) (R2 : ℝ),
    B_new = B * (1 - slump_percentage) →
    B * R1 = B_new * R2 → 
    R2 = 0.05) : 
  true := 
by 
  sorry

end new_rate_of_commission_l1311_131179


namespace a_plus_b_l1311_131129

theorem a_plus_b (a b : ℝ) (h : a * (a - 4) = 12) (h2 : b * (b - 4) = 12) (h3 : a ≠ b) : a + b = 4 :=
sorry

end a_plus_b_l1311_131129


namespace p_necessary_for_q_l1311_131187

def p (x : ℝ) := x ≠ 1
def q (x : ℝ) := x ≥ 2

theorem p_necessary_for_q : ∀ x, q x → p x :=
by
  intro x
  intro hqx
  rw [q] at hqx
  rw [p]
  sorry

end p_necessary_for_q_l1311_131187


namespace measure_of_angle_x_l1311_131100

-- Defining the conditions
def angle_ABC : ℝ := 108
def angle_ABD : ℝ := 180 - angle_ABC
def angle_in_triangle_ABD_1 : ℝ := 26
def sum_of_angles_in_triangle (a b c : ℝ) : Prop := a + b + c = 180

-- The theorem to prove
theorem measure_of_angle_x (h1 : angle_ABD = 72)
                           (h2 : angle_in_triangle_ABD_1 = 26)
                           (h3 : sum_of_angles_in_triangle angle_ABD angle_in_triangle_ABD_1 x) :
  x = 82 :=
by {
  -- Since this is a formal statement, we leave the proof as an exercise 
  sorry
}

end measure_of_angle_x_l1311_131100


namespace tangent_condition_l1311_131124

theorem tangent_condition (a b : ℝ) : 
    a = b → 
    (∀ x y : ℝ, (y = x + 2 → (x - a)^2 + (y - b)^2 = 2 → y = x + 2)) :=
by
  sorry

end tangent_condition_l1311_131124


namespace probability_red_blue_l1311_131158

-- Declare the conditions (probabilities for white, green and yellow marbles).
variables (total_marbles : ℕ) (P_white P_green P_yellow P_red_blue : ℚ)
-- implicitly P_white, P_green, P_yellow, P_red_blue are probabilities, therefore between 0 and 1

-- Assume the conditions given in the problem
axiom total_marbles_condition : total_marbles = 250
axiom P_white_condition : P_white = 2 / 5
axiom P_green_condition : P_green = 1 / 4
axiom P_yellow_condition : P_yellow = 1 / 10

-- Proving the required probability of red or blue marbles
theorem probability_red_blue :
  P_red_blue = 1 - (P_white + P_green + P_yellow) :=
sorry

end probability_red_blue_l1311_131158
