import Mathlib

namespace positive_n_for_one_solution_l238_238471

theorem positive_n_for_one_solution :
  ∀ (n : ℝ), (4 * (0 : ℝ)) ^ 2 + n * (0) + 16 = 0 → (n^2 - 256 = 0) → n = 16 :=
by
  intro n
  intro h
  intro discriminant_eq_zero
  sorry

end positive_n_for_one_solution_l238_238471


namespace find_coefficients_l238_238462

variable (P Q x : ℝ)

theorem find_coefficients :
  (∀ x, x^2 - 8 * x - 20 = (x - 10) * (x + 2))
  → (∀ x, 6 * x - 4 = P * (x + 2) + Q * (x - 10))
  → P = 14 / 3 ∧ Q = 4 / 3 :=
by
  intros h1 h2
  sorry

end find_coefficients_l238_238462


namespace fractional_to_decimal_l238_238061

theorem fractional_to_decimal : (7 / 16 : ℚ) = 0.4375 := by
  sorry

end fractional_to_decimal_l238_238061


namespace Ryan_spits_percentage_shorter_l238_238443

theorem Ryan_spits_percentage_shorter (Billy_dist Madison_dist Ryan_dist : ℝ) (h1 : Billy_dist = 30) (h2 : Madison_dist = 1.20 * Billy_dist) (h3 : Ryan_dist = 18) :
  ((Madison_dist - Ryan_dist) / Madison_dist) * 100 = 50 :=
by
  sorry

end Ryan_spits_percentage_shorter_l238_238443


namespace stratified_sampling_groupD_l238_238094

-- Definitions for the conditions
def totalDistrictCount : ℕ := 38
def groupADistrictCount : ℕ := 4
def groupBDistrictCount : ℕ := 10
def groupCDistrictCount : ℕ := 16
def groupDDistrictCount : ℕ := 8
def numberOfCitiesToSelect : ℕ := 9

-- Define stratified sampling calculation with a floor function or rounding
noncomputable def numberSelectedFromGroupD : ℕ := (groupDDistrictCount * numberOfCitiesToSelect) / totalDistrictCount

-- The theorem to prove 
theorem stratified_sampling_groupD : numberSelectedFromGroupD = 2 := by
  sorry -- This is where the proof would go

end stratified_sampling_groupD_l238_238094


namespace cos_300_eq_half_l238_238827

-- Define the point on the unit circle at 300 degrees
def point_on_unit_circle_angle_300 : ℝ × ℝ :=
  let θ := 300 * (Real.pi / 180) in
  (Real.cos θ, Real.sin θ)

-- Define the cosine of the angle 300 degrees
def cos_300 : ℝ :=
  Real.cos (300 * (Real.pi / 180))

theorem cos_300_eq_half : cos_300 = 1 / 2 :=
by sorry

end cos_300_eq_half_l238_238827


namespace cos_300_eq_half_l238_238831

-- Define the point on the unit circle at 300 degrees
def point_on_unit_circle_angle_300 : ℝ × ℝ :=
  let θ := 300 * (Real.pi / 180) in
  (Real.cos θ, Real.sin θ)

-- Define the cosine of the angle 300 degrees
def cos_300 : ℝ :=
  Real.cos (300 * (Real.pi / 180))

theorem cos_300_eq_half : cos_300 = 1 / 2 :=
by sorry

end cos_300_eq_half_l238_238831


namespace total_steps_l238_238370

theorem total_steps (up_steps down_steps : ℕ) (h1 : up_steps = 567) (h2 : down_steps = 325) : up_steps + down_steps = 892 := by
  sorry

end total_steps_l238_238370


namespace fraction_to_decimal_l238_238055

theorem fraction_to_decimal :
  (7 : ℝ) / 16 = 0.4375 := 
by sorry

end fraction_to_decimal_l238_238055


namespace greatest_sum_of_consecutive_integers_product_lt_500_l238_238739

theorem greatest_sum_of_consecutive_integers_product_lt_500 : 
  ∃ (n : ℤ), n * (n + 1) < 500 ∧ (∀ m : ℤ, m * (m + 1) < 500 → m + (m + 1) ≤ n + (n + 1)) ∧ n + (n + 1) = 43 :=
begin
  sorry
end

end greatest_sum_of_consecutive_integers_product_lt_500_l238_238739


namespace fraction_to_decimal_l238_238076

theorem fraction_to_decimal : (7 : ℚ) / 16 = 4375 / 10000 := by
  sorry

end fraction_to_decimal_l238_238076


namespace reflection_across_x_axis_l238_238216

def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

theorem reflection_across_x_axis :
  reflect_x_axis (-2, -3) = (-2, 3) :=
by
  sorry

end reflection_across_x_axis_l238_238216


namespace sum_first_5n_eq_630_l238_238956

theorem sum_first_5n_eq_630 (n : ℕ)
  (h : (4 * n * (4 * n + 1)) / 2 = (2 * n * (2 * n + 1)) / 2 + 300) :
  (5 * n * (5 * n + 1)) / 2 = 630 :=
sorry

end sum_first_5n_eq_630_l238_238956


namespace shaded_region_probability_eq_half_l238_238441

open Real -- Use the Real namespace
open Probability -- Use the Probability namespace

noncomputable def probability_shaded_region (A B C D : Real) : Real :=
    if h : (A = (0, 0) ∧ B = (2, 0) ∧ C = (1, 4) ∧ D = (1, 0))
    then 1 / 2
    else 0

theorem shaded_region_probability_eq_half :
  let A := (0 : Real, 0 : Real)
  let B := (2 : Real, 0 : Real)
  let C := (1 : Real, 4 : Real)
  let D := (1 : Real, 0 : Real)
  probability_shaded_region A B C D = 1 / 2 :=
by
  sorry

end shaded_region_probability_eq_half_l238_238441


namespace find_y_l238_238239

theorem find_y (AB BC : ℕ) (y x : ℕ) 
  (h1 : AB = 3 * y)
  (h2 : BC = 2 * x)
  (h3 : AB * BC = 2400) 
  (h4 : AB * BC = 6 * x * y) :
  y = 20 := by
  sorry

end find_y_l238_238239


namespace digit_difference_one_l238_238531

variable (d C D : ℕ)

-- Assumptions
variables (h1 : d > 8)
variables (h2 : d * d * d + C * d + D + d * d * d + C * d + C = 2 * d * d + 5 * d + 3)

theorem digit_difference_one (h1 : d > 8) (h2 : d * d * d + C * d + D + d * d * d + C * d + C = 2 * d * d + 5 * d + 3) :
  C - D = 1 :=
by
  sorry

end digit_difference_one_l238_238531


namespace part1_solution_set_part2_range_of_a_l238_238908

-- Define the function f for a general a
def f (x a : ℝ) : ℝ := |x - a^2| + |x - (2 * a) + 1|

-- Part 1: When a = 2
theorem part1_solution_set (x : ℝ) (h : f x 2 ≥ 4) : x ≤ 3/2 ∨ x ≥ 11/2 :=
  sorry

-- Part 2: Range of values for a
theorem part2_range_of_a (a : ℝ) (h : ∀ x, f x a ≥ 4) : a ≤ -1 ∨ a ≥ 3 :=
  sorry

end part1_solution_set_part2_range_of_a_l238_238908


namespace part1_solution_set_part2_range_of_a_l238_238941

-- Defining the function f(x) under given conditions
def f (x a : ℝ) : ℝ := |x - a^2| + |x - (2 * a + 1)|

-- Part 1: Determine the solution set for the inequality when a = 2
theorem part1_solution_set (x : ℝ) : (f x 2) ≥ 4 ↔ x ≤ 3/2 ∨ x ≥ 11/2 := by
  sorry

-- Part 2: Determine the range of values for a given the inequality
theorem part2_range_of_a (a : ℝ) : (∀ x : ℝ, f x a ≥ 4) ↔ a ≤ -1 ∨ a ≥ 3 := by
  sorry

end part1_solution_set_part2_range_of_a_l238_238941


namespace monotonic_intervals_find_f_max_l238_238340

noncomputable def f (x : ℝ) : ℝ := Real.log x / x

theorem monotonic_intervals :
  (∀ x, 0 < x → x < Real.exp 1 → 0 < (1 - Real.log x) / x^2) ∧
  (∀ x, x > Real.exp 1 → (1 - Real.log x) / x^2 < 0) :=
sorry

theorem find_f_max (m : ℝ) (h : m > 0) :
  if 0 < 2 * m ∧ 2 * m ≤ Real.exp 1 then f (2 * m) = Real.log (2 * m) / (2 * m)
  else if m ≥ Real.exp 1 then f m = Real.log m / m
  else f (Real.exp 1) = 1 / Real.exp 1 :=
sorry

end monotonic_intervals_find_f_max_l238_238340


namespace cos_300_eq_cos_300_l238_238822

open Real

theorem cos_300_eq (angle_1 angle_2 : ℝ) (h1 : angle_1 = 300 * π / 180) (h2 : angle_2 = 60 * π / 180) :
  cos angle_1 = cos angle_2 :=
by
  rw [←h1, ←h2, cos_sub, cos_pi_div_six]
  norm_num
  sorry

theorem cos_300 : cos (300 * π / 180) = 1 / 2 :=
by
  apply cos_300_eq (300 * π / 180) (60 * π / 180)
  norm_num
  norm_num
  sorry

end cos_300_eq_cos_300_l238_238822


namespace probability_sum_is_10_l238_238208

theorem probability_sum_is_10 : 
  let outcomes := [(a, b, c) | a ∈ [1, 2, 3, 4, 5, 6], b ∈ [1, 2, 3, 4, 5, 6], c ∈ [1, 2, 3, 4, 5, 6]],
      favorable_outcomes := [(a, b, c) | (a, b, c) ∈ outcomes, a + b + c = 10] in
  (favorable_outcomes.length : ℚ) / outcomes.length = 1 / 9 := by
  sorry

end probability_sum_is_10_l238_238208


namespace taxi_fare_max_distance_l238_238993

-- Setting up the conditions
def starting_price : ℝ := 7
def additional_fare_per_km : ℝ := 2.4
def max_base_distance_km : ℝ := 3
def total_fare : ℝ := 19

-- Defining the maximum distance based on the given conditions
def max_distance : ℝ := 8

-- The theorem is to prove that the maximum distance is indeed 8 kilometers
theorem taxi_fare_max_distance :
  ∀ (x : ℝ), total_fare = starting_price + additional_fare_per_km * (x - max_base_distance_km) → x ≤ max_distance :=
by
  intros x h
  sorry

end taxi_fare_max_distance_l238_238993


namespace part_a_proof_part_b_proof_l238_238420

-- Definitions for Part (a)

variables {A B C M C_1 B_1 : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space M] [metric_space C_1] [metric_space B_1]
  
-- Assume properties of right triangles constructed externally on the sides of triangle ABC
variables (ABC1 : triangle B C A C1)
variables (AB1C : triangle B1 A C B)

-- External right triangles with specified angles
axiom ex_triangle_ABC1 : ∀ {ABC : triangle A B C}, ∀ {ϕ : angle}, right_triangle ABC1 ∧ angle.BAC = ϕ ∧ angle.C1 = pi/2
axiom ex_triangle_AB1C : ∀ {ABC : triangle A B C}, ∀ {ϕ : angle}, right_triangle AB1C ∧ angle.BCA = ϕ ∧ angle.B1 = pi/2

-- Midpoint M of BC
axiom M_midpoint : ∀ {A B C : point}, is_midpoint (M) B C 

-- Required to prove MB1 = MC1 and the specified angle relationship
theorem part_a_proof : ∀ {A B C M C1 B1 : point}, (is_midpoint M B C) → right_triangle ABC1 → (angle.BAC = ϕ) → right_triangle AB1C → (angle.BCA = ϕ) →  (dist M B1 = dist M C1) ∧ (angle B1 M C1 = 2 * ϕ) := 
sorry

-- Definitions for Part (b)

variables {A B C G : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space G]

-- External equilateral triangles on the sides of triangle ABC
variables (E1 E2 E3 : Type) [metric_space E1] [metric_space E2] [metric_space E3]

-- Centers of externally constructed equilateral triangles on the sides
axiom centers_equi_triangles : ∀ {A B C : point}, ∀ {E1 E2 E3 : point}, equilateral_triangle E1 E2 E3 ∧ is_center_tri E1 B C ∧ is_center_tri E2 A C ∧ is_center_tri E3 A B

-- Required to prove that the centers form an equilateral triangle coinciding with centroid
theorem part_b_proof : ∀ {A B C G : point}, equilateral_triangle E1 E2 E3 → (is_center_tri E1 B C) → (is_center_tri E2 A C) → (is_center_tri E3 A B) → equilateral_triangle (E1 E2 E3) ∧ centroid (triangle A B C) (E1 E2 E3) :=
sorry

end part_a_proof_part_b_proof_l238_238420


namespace fraction_to_decimal_l238_238039

theorem fraction_to_decimal :
  (7 / 16 : ℝ) = 0.4375 := 
begin
  -- placeholder for the proof
  sorry
end

end fraction_to_decimal_l238_238039


namespace selling_price_l238_238106

-- Definitions
def price_coffee_A : ℝ := 10
def price_coffee_B : ℝ := 12
def weight_coffee_A : ℝ := 240
def weight_coffee_B : ℝ := 240
def total_weight : ℝ := 480
def total_cost : ℝ := (weight_coffee_A * price_coffee_A) + (weight_coffee_B * price_coffee_B)

-- Theorem
theorem selling_price (h_total_weight : total_weight = weight_coffee_A + weight_coffee_B) :
  total_cost / total_weight = 11 :=
by
  sorry

end selling_price_l238_238106


namespace cos_300_eq_half_l238_238783

theorem cos_300_eq_half :
  ∀ (deg : ℝ), deg = 300 → cos (deg * π / 180) = 1 / 2 :=
by
  intros deg h_deg
  rw [h_deg]
  have h1 : 300 = 360 - 60, by norm_num
  rw [h1]
  -- Convert angles to radians for usage with cos function
  have h2 : 360 - 60 = (2 * π - (π / 3)) * 180 / π, by norm_num
  rw [h2]
  have h3 : cos ((2 * π - π / 3) * π / 180) = cos (π / 3), by rw [real.cos_sub_pi_div_two, real.cos_pi_div_three]
  rw [h3]
  norm_num

end cos_300_eq_half_l238_238783


namespace find_m_range_l238_238647

theorem find_m_range (m : ℝ) (x : ℝ) (h : ∃ c d : ℝ, (c ≠ 0) ∧ (∀ x, (c * x + d)^2 = x^2 + (12 / 5) * x + (2 * m / 5))) : 3.5 ≤ m ∧ m ≤ 3.7 :=
by
  sorry

end find_m_range_l238_238647


namespace f_monotonic_intervals_f_maximum_value_on_interval_l238_238341

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem f_monotonic_intervals :
  (∀ x : ℝ, 0 < x → x < Real.e → f x < f Real.e) ∧
  (∀ x : ℝ, x > Real.e → f x < f Real.e) :=
sorry

theorem f_maximum_value_on_interval (m : ℝ) (hm : m > 0) :
  ∃ x_max : ℝ, x_max ∈ (Set.Icc m (2 * m)) ∧
  (∀ x ∈ (Set.Icc m (2 * m)), f x ≤ f x_max) ∧
  ((m₁ : 0 < m ∧ m ≤ Real.e / 2 → x_max = 2 * m ∧ f x_max = (Real.log (2 * m)) / (2 * m)) ∨
   (m₂ : m ≥ Real.e → x_max = m ∧ f x_max = (Real.log m) / m) ∨
   (m₃ : Real.e / 2 < m ∧ m < Real.e → x_max = Real.e ∧ f x_max = 1 / Real.e)) :=
sorry

end f_monotonic_intervals_f_maximum_value_on_interval_l238_238341


namespace disputed_piece_weight_l238_238657

theorem disputed_piece_weight (x d : ℝ) (h1 : x - d = 300) (h2 : x + d = 500) : x = 400 := by
  sorry

end disputed_piece_weight_l238_238657


namespace mary_average_speed_l238_238387

noncomputable def trip_distance : ℝ := 1.5 + 1.5
noncomputable def trip_time_minutes : ℝ := 45 + 15
noncomputable def trip_time_hours : ℝ := trip_time_minutes / 60

theorem mary_average_speed :
  (trip_distance / trip_time_hours) = 3 := by
  sorry

end mary_average_speed_l238_238387


namespace g_x_squared_plus_2_l238_238513

namespace PolynomialProof

open Polynomial

noncomputable def g (x : ℚ) : ℚ := sorry

theorem g_x_squared_plus_2 (x : ℚ) (h : g (x^2 - 2) = x^4 - 6*x^2 + 8) :
  g (x^2 + 2) = x^4 + 2*x^2 + 2 :=
sorry

end PolynomialProof

end g_x_squared_plus_2_l238_238513


namespace find_x_of_product_eq_72_l238_238138

theorem find_x_of_product_eq_72 (x : ℝ) (h : 0 < x) (hx : x * ⌊x⌋₊ = 72) : x = 9 :=
sorry

end find_x_of_product_eq_72_l238_238138


namespace parabola_vertex_relationship_l238_238985

theorem parabola_vertex_relationship (m x y : ℝ) :
  (y = x^2 - 2*m*x + 2*m^2 - 3*m + 1) → (y = x^2 - 3*x + 1) :=
by
  intro h
  sorry

end parabola_vertex_relationship_l238_238985


namespace simplify_and_evaluate_expression_l238_238987

theorem simplify_and_evaluate_expression (m : ℝ) (h : m = 2):
  ( ( (2 * m + 1) / m - 1 ) / ( (m^2 - 1) / m ) ) = 1 :=
by
  rw [h] -- Replace m by 2
  sorry

end simplify_and_evaluate_expression_l238_238987


namespace fraction_to_decimal_l238_238022

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l238_238022


namespace find_x_l238_238166

theorem find_x (x : ℝ) (h : 1 - 1 / (1 - x) = 1 / (1 - x)) : x = -1 :=
by
  sorry

end find_x_l238_238166


namespace boys_count_l238_238104

def total_pupils : ℕ := 485
def number_of_girls : ℕ := 232
def number_of_boys : ℕ := total_pupils - number_of_girls

theorem boys_count : number_of_boys = 253 := by
  -- The proof is omitted according to instruction
  sorry

end boys_count_l238_238104


namespace sum_first_9_terms_l238_238854

variable (a b : ℕ → ℝ)
variable (S : ℕ → ℝ)

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop := ∀ m n k l, m + n = k + l → a m * a n = a k * a l
def geometric_prop (a : ℕ → ℝ) : Prop := a 3 * a 7 = 2 * a 5
def arithmetic_b5_eq_a5 (a b : ℕ → ℝ) : Prop := b 5 = a 5

-- The Sum Sn of an arithmetic sequence up to the nth terms
def arithmetic_sum (b : ℕ → ℝ) (S : ℕ → ℝ) : Prop := ∀ n, S n = (n / 2) * (b 1 + b n)

-- Question statement: proving the required sum
theorem sum_first_9_terms (a b : ℕ → ℝ) (S : ℕ → ℝ) 
  (hg : is_geometric_sequence a) 
  (hp : geometric_prop a) 
  (hb : arithmetic_b5_eq_a5 a b) 
  (arith_sum: arithmetic_sum b S) :
  S 9 = 18 :=
  sorry

end sum_first_9_terms_l238_238854


namespace area_of_AFCH_l238_238235

-- Define the sides of the rectangles ABCD and EFGH
def AB : ℝ := 9
def BC : ℝ := 5
def EF : ℝ := 3
def FG : ℝ := 10

-- Define the area of quadrilateral AFCH
def area_AFCH : ℝ := 52.5

-- The theorem we want to prove
theorem area_of_AFCH :
  AB = 9 ∧ BC = 5 ∧ EF = 3 ∧ FG = 10 → (area_AFCH = 52.5) :=
by
  sorry

end area_of_AFCH_l238_238235


namespace bananas_indeterminate_l238_238390

namespace RubyBananaProblem

variables (number_of_candies : ℕ) (number_of_friends : ℕ) (candies_per_friend : ℕ)
           (number_of_bananas : Option ℕ)

-- Given conditions
def Ruby_has_36_candies := number_of_candies = 36
def Ruby_has_9_friends := number_of_friends = 9
def Each_friend_gets_4_candies := candies_per_friend = 4
def Can_distribute_candies := number_of_candies = number_of_friends * candies_per_friend

-- Mathematical statement
theorem bananas_indeterminate (h1 : Ruby_has_36_candies number_of_candies)
                              (h2 : Ruby_has_9_friends number_of_friends)
                              (h3 : Each_friend_gets_4_candies candies_per_friend)
                              (h4 : Can_distribute_candies number_of_candies number_of_friends candies_per_friend) :
  number_of_bananas = none :=
by
  sorry

end RubyBananaProblem

end bananas_indeterminate_l238_238390


namespace cos_300_eq_half_l238_238782

theorem cos_300_eq_half :
  ∀ (deg : ℝ), deg = 300 → cos (deg * π / 180) = 1 / 2 :=
by
  intros deg h_deg
  rw [h_deg]
  have h1 : 300 = 360 - 60, by norm_num
  rw [h1]
  -- Convert angles to radians for usage with cos function
  have h2 : 360 - 60 = (2 * π - (π / 3)) * 180 / π, by norm_num
  rw [h2]
  have h3 : cos ((2 * π - π / 3) * π / 180) = cos (π / 3), by rw [real.cos_sub_pi_div_two, real.cos_pi_div_three]
  rw [h3]
  norm_num

end cos_300_eq_half_l238_238782


namespace problem_statement_l238_238323

theorem problem_statement (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) 
    (h3 : m + 5 < n) 
    (h4 : (m + 3 + m + 7 + m + 13 + n + 4 + n + 5 + 2 * n + 3) / 6 = n + 3)
    (h5 : (↑((m + 13) + (n + 4)) / 2 : ℤ) = n + 3) : 
  m + n = 37 :=
by
  sorry

end problem_statement_l238_238323


namespace abs_diff_squares_105_95_l238_238688

theorem abs_diff_squares_105_95 : abs ((105 : ℤ)^2 - (95 : ℤ)^2) = 2000 := 
by
  sorry

end abs_diff_squares_105_95_l238_238688


namespace sum_first_15_odd_integers_from_5_l238_238556

theorem sum_first_15_odd_integers_from_5 :
  let a := 5
  let n := 15
  let d := 2
  let last_term := a + (n - 1) * d
  let S := n * a + (n * (n - 1) * d) / 2
  last_term = 37 ∧ S = 315 := by
  sorry

end sum_first_15_odd_integers_from_5_l238_238556


namespace gcd_of_78_and_36_l238_238253

theorem gcd_of_78_and_36 : Int.gcd 78 36 = 6 := by
  sorry

end gcd_of_78_and_36_l238_238253


namespace dice_sum_probability_l238_238202

theorem dice_sum_probability : 
  let die_faces := {1, 2, 3, 4, 5, 6};
      successful_outcomes := { (x, y, z) | x ∈ die_faces ∧ y ∈ die_faces ∧ z ∈ die_faces ∧ x + y + z = 10 }.toFinset.card;
      total_outcomes := (finset.range 6).card ^ 3
  in  successful_outcomes / total_outcomes = 1 / 9 :=
by
  let die_faces := {1, 2, 3, 4, 5, 6};
  let successful_outcomes := { (x, y, z) | x ∈ die_faces ∧ y ∈ die_faces ∧ z ∈ die_faces ∧ x + y + z = 10 }.toFinset.card;
  let total_outcomes := (finset.range 6).card ^ 3;
  sorry

end dice_sum_probability_l238_238202


namespace initial_students_per_group_l238_238678

-- Define the conditions
variables {x : ℕ} (h : 3 * x - 2 = 22)

-- Lean 4 statement of the proof problem
theorem initial_students_per_group (x : ℕ) (h : 3 * x - 2 = 22) : x = 8 :=
sorry

end initial_students_per_group_l238_238678


namespace part1_solution_set_part2_range_of_a_l238_238914

-- Define the function f for a general a
def f (x a : ℝ) : ℝ := |x - a^2| + |x - (2 * a) + 1|

-- Part 1: When a = 2
theorem part1_solution_set (x : ℝ) (h : f x 2 ≥ 4) : x ≤ 3/2 ∨ x ≥ 11/2 :=
  sorry

-- Part 2: Range of values for a
theorem part2_range_of_a (a : ℝ) (h : ∀ x, f x a ≥ 4) : a ≤ -1 ∨ a ≥ 3 :=
  sorry

end part1_solution_set_part2_range_of_a_l238_238914


namespace halloween_candy_l238_238322

theorem halloween_candy (katie_candy : ℕ) (sister_candy : ℕ) (remaining_candy : ℕ) (total_candy : ℕ) (eaten_candy : ℕ)
  (h1 : katie_candy = 10) 
  (h2 : sister_candy = 6) 
  (h3 : remaining_candy = 7) 
  (h4 : total_candy = katie_candy + sister_candy) 
  (h5 : eaten_candy = total_candy - remaining_candy) : 
  eaten_candy = 9 :=
by sorry

end halloween_candy_l238_238322


namespace cubing_identity_l238_238160

theorem cubing_identity (x : ℝ) (hx : x + 1/x = -3) : x^3 + 1/x^3 = -18 := by
  sorry

end cubing_identity_l238_238160


namespace youngest_brother_age_l238_238523

theorem youngest_brother_age 
  (Rick_age : ℕ)
  (oldest_brother_age : ℕ)
  (middle_brother_age : ℕ)
  (smallest_brother_age : ℕ)
  (youngest_brother_age : ℕ)
  (h1 : Rick_age = 15)
  (h2 : oldest_brother_age = 2 * Rick_age)
  (h3 : middle_brother_age = oldest_brother_age / 3)
  (h4 : smallest_brother_age = middle_brother_age / 2)
  (h5 : youngest_brother_age = smallest_brother_age - 2) :
  youngest_brother_age = 3 := 
sorry

end youngest_brother_age_l238_238523


namespace probability_sum_is_10_l238_238209

theorem probability_sum_is_10 : 
  let outcomes := [(a, b, c) | a ∈ [1, 2, 3, 4, 5, 6], b ∈ [1, 2, 3, 4, 5, 6], c ∈ [1, 2, 3, 4, 5, 6]],
      favorable_outcomes := [(a, b, c) | (a, b, c) ∈ outcomes, a + b + c = 10] in
  (favorable_outcomes.length : ℚ) / outcomes.length = 1 / 9 := by
  sorry

end probability_sum_is_10_l238_238209


namespace find_k_l238_238149

-- Define the vectors
def e1 : ℝ × ℝ := (1, 0)
def e2 : ℝ × ℝ := (0, 1)

def a : ℝ × ℝ := (e1.1 - 2 * e2.1, e1.2 - 2 * e2.2)
def b (k : ℝ) : ℝ × ℝ := (k * e1.1 + e2.1, k * e1.2 + e2.2)

-- Define the parallel condition
def parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

-- Statement of the problem translated to a Lean theorem
theorem find_k (k : ℝ) : 
  parallel a (b k) -> k = -1 / 2 := by
  sorry

end find_k_l238_238149


namespace cos_300_is_half_l238_238791

noncomputable def cos_300_degrees : ℝ :=
  real.cos (300 * real.pi / 180)

theorem cos_300_is_half :
  cos_300_degrees = 1 / 2 :=
sorry

end cos_300_is_half_l238_238791


namespace fraction_calls_processed_by_team_B_l238_238276

variable (A B C_A C_B : ℕ)

theorem fraction_calls_processed_by_team_B 
  (h1 : A = (5 / 8) * B)
  (h2 : C_A = (2 / 5) * C_B) :
  (B * C_B) / ((A * C_A) + (B * C_B)) = 8 / 9 := by
  sorry

end fraction_calls_processed_by_team_B_l238_238276


namespace fraction_decimal_equivalent_l238_238005

theorem fraction_decimal_equivalent : (7 / 16 : ℝ) = 0.4375 :=
by
  sorry

end fraction_decimal_equivalent_l238_238005


namespace part1_solution_set_part2_range_of_a_l238_238937

-- Defining the function f(x) under given conditions
def f (x a : ℝ) : ℝ := |x - a^2| + |x - (2 * a + 1)|

-- Part 1: Determine the solution set for the inequality when a = 2
theorem part1_solution_set (x : ℝ) : (f x 2) ≥ 4 ↔ x ≤ 3/2 ∨ x ≥ 11/2 := by
  sorry

-- Part 2: Determine the range of values for a given the inequality
theorem part2_range_of_a (a : ℝ) : (∀ x : ℝ, f x a ≥ 4) ↔ a ≤ -1 ∨ a ≥ 3 := by
  sorry

end part1_solution_set_part2_range_of_a_l238_238937


namespace greatest_sum_of_consecutive_integers_product_lt_500_l238_238740

theorem greatest_sum_of_consecutive_integers_product_lt_500 : 
  ∃ (n : ℤ), n * (n + 1) < 500 ∧ (∀ m : ℤ, m * (m + 1) < 500 → m + (m + 1) ≤ n + (n + 1)) ∧ n + (n + 1) = 43 :=
begin
  sorry
end

end greatest_sum_of_consecutive_integers_product_lt_500_l238_238740


namespace longer_strap_length_l238_238547

theorem longer_strap_length (S L : ℕ) 
  (h1 : L = S + 72) 
  (h2 : S + L = 348) : 
  L = 210 := 
sorry

end longer_strap_length_l238_238547


namespace fractional_to_decimal_l238_238058

theorem fractional_to_decimal : (7 / 16 : ℚ) = 0.4375 := by
  sorry

end fractional_to_decimal_l238_238058


namespace fraction_to_decimal_l238_238025

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l238_238025


namespace theater_ticket_sales_l238_238108

theorem theater_ticket_sales
  (A C : ℕ)
  (h₁ : 8 * A + 5 * C = 236)
  (h₂ : A + C = 34) : A = 22 :=
by
  sorry

end theater_ticket_sales_l238_238108


namespace proposition_C_is_correct_l238_238754

theorem proposition_C_is_correct :
  ∃ a b : ℝ, (a > 2 ∧ b > 2) → (a * b > 4) :=
by
  sorry

end proposition_C_is_correct_l238_238754


namespace john_vegetables_used_l238_238968

noncomputable def pounds_of_beef_bought : ℕ := 4
noncomputable def pounds_of_beef_used : ℕ := pounds_of_beef_bought - 1
noncomputable def pounds_of_vegetables_used : ℕ := 2 * pounds_of_beef_used

theorem john_vegetables_used : pounds_of_vegetables_used = 6 :=
by
  -- the proof can be provided here later
  sorry

end john_vegetables_used_l238_238968


namespace difference_of_roots_l238_238445

theorem difference_of_roots 
  (a b c : ℝ)
  (h : ∀ x, x^2 - 2 * (a^2 + b^2 + c^2 - 2 * a * c) * x + (b^2 - a^2 - c^2 + 2 * a * c)^2 = 0) :
  ∃ (x1 x2 : ℝ), (x1 - x2 = 4 * b * (a - c)) ∨ (x1 - x2 = -4 * b * (a - c)) :=
sorry

end difference_of_roots_l238_238445


namespace probability_sum_10_l238_238205

def is_valid_roll (d1 d2 d3 : ℕ) : Prop :=
  1 ≤ d1 ∧ d1 ≤ 6 ∧ 1 ≤ d2 ∧ d2 ≤ 6 ∧ 1 ≤ d3 ∧ d3 ≤ 6

def sum_is_10 (d1 d2 d3 : ℕ) : Prop :=
  d1 + d2 + d3 = 10

def valid_rolls_count : ℕ :=
  216 -- 6^3 distinct rolls of three 6-sided dice

def successful_rolls_count : ℕ :=
  24 -- number of valid rolls that sum to 10

theorem probability_sum_10 :
  (successful_rolls_count : ℚ) / valid_rolls_count = 1 / 9 := by
  sorry

end probability_sum_10_l238_238205


namespace birds_joined_l238_238563

-- Definitions based on the identified conditions
def initial_birds : ℕ := 3
def initial_storks : ℕ := 2
def total_after_joining : ℕ := 10

-- Theorem statement that follows from the problem setup
theorem birds_joined :
  total_after_joining - (initial_birds + initial_storks) = 5 := by
  sorry

end birds_joined_l238_238563


namespace charge_difference_l238_238424

theorem charge_difference (cost_x cost_y : ℝ) (num_copies : ℕ) (hx : cost_x = 1.25) (hy : cost_y = 2.75) (hn : num_copies = 40) : 
  num_copies * cost_y - num_copies * cost_x = 60 := by
  sorry

end charge_difference_l238_238424


namespace fraction_to_decimal_l238_238071

theorem fraction_to_decimal : (7 : ℝ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l238_238071


namespace gas_usage_l238_238640

def distance_dermatologist : ℕ := 30
def distance_gynecologist : ℕ := 50
def car_efficiency : ℕ := 20

theorem gas_usage (d_1 d_2 e : ℕ) (H1 : d_1 = distance_dermatologist) (H2 : d_2 = distance_gynecologist) (H3 : e = car_efficiency) :
  (2 * d_1 + 2 * d_2) / e = 8 :=
by
  rw [H1, H2, H3]
  norm_num
  sorry

end gas_usage_l238_238640


namespace total_distance_travelled_eight_boys_on_circle_l238_238309

noncomputable def distance_travelled_by_boys (radius : ℝ) : ℝ :=
  let n := 8
  let angle := 2 * Real.pi / n
  let distance_to_non_adjacent := 2 * radius * Real.sin (2 * angle / 2)
  n * (100 + 3 * distance_to_non_adjacent)

theorem total_distance_travelled_eight_boys_on_circle :
  distance_travelled_by_boys 50 = 800 + 1200 * Real.sqrt 2 :=
  by
    sorry

end total_distance_travelled_eight_boys_on_circle_l238_238309


namespace cos_sum_seventh_root_of_unity_l238_238613

theorem cos_sum_seventh_root_of_unity (z : ℂ) (α : ℝ) 
  (h1 : z ^ 7 = 1) (h2 : z ≠ 1) (h3 : ∃ k : ℤ, α = (2 * k * π) / 7 ) :
  (Real.cos α + Real.cos (2 * α) + Real.cos (4 * α)) = -1 / 2 :=
by 
  sorry

end cos_sum_seventh_root_of_unity_l238_238613


namespace solution_set_l238_238384

variable {f : ℝ → ℝ}
variable {f' : ℝ → ℝ}

-- Given conditions
axiom differentiable_f : Differentiable ℝ f
axiom f_deriv : ∀ x, deriv f x = f' x
axiom f_at_3 : f 3 = 1
axiom inequality : ∀ x, 3 * f x + x * f' x > 1

-- Goal to prove
theorem solution_set :
  {x : ℝ | (x - 2017) ^ 3 * f (x - 2017) - 27 > 0} = {x | 2020 < x} :=
  sorry

end solution_set_l238_238384


namespace abscissa_midpoint_range_l238_238850

-- Definitions based on the given conditions.
def circle_eq (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 6
def on_circle (x y : ℝ) : Prop := circle_eq x y
def chord_length (A B : ℝ × ℝ) : Prop := (A.1 - B.1)^2 + (A.2 - B.2)^2 = (2 * Real.sqrt 2)^2
def line_eq (x y : ℝ) : Prop := x - y - 2 = 0
def on_line (x y : ℝ) : Prop := line_eq x y
def segment_length (P Q : ℝ × ℝ) : Prop := (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 4
def acute_angle (P Q G : ℝ × ℝ) : Prop := -- definition of acute angle condition
  sorry -- placeholder for the actual definition

-- The proof statement.
theorem abscissa_midpoint_range {A B P Q G M : ℝ × ℝ}
  (h_A_on_circle : on_circle A.1 A.2)
  (h_B_on_circle : on_circle B.1 B.2)
  (h_AB_length : chord_length A B)
  (h_P_on_line : on_line P.1 P.2)
  (h_Q_on_line : on_line Q.1 Q.2)
  (h_PQ_length : segment_length P Q)
  (h_angle_acute : acute_angle P Q G)
  (h_G_mid : G = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (h_M_mid : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) :
  (M.1 < 0) ∨ (M.1 > 3) :=
sorry

end abscissa_midpoint_range_l238_238850


namespace probability_heads_l238_238089

theorem probability_heads (p : ℝ) (q : ℝ) (C_10_5 C_10_6 : ℝ)
  (h1 : q = 1 - p)
  (h2 : C_10_5 = 252)
  (h3 : C_10_6 = 210)
  (eqn : C_10_5 * p^5 * q^5 = C_10_6 * p^6 * q^4) :
  p = 6 / 11 := 
by
  sorry

end probability_heads_l238_238089


namespace probability_sum_10_l238_238206

def is_valid_roll (d1 d2 d3 : ℕ) : Prop :=
  1 ≤ d1 ∧ d1 ≤ 6 ∧ 1 ≤ d2 ∧ d2 ≤ 6 ∧ 1 ≤ d3 ∧ d3 ≤ 6

def sum_is_10 (d1 d2 d3 : ℕ) : Prop :=
  d1 + d2 + d3 = 10

def valid_rolls_count : ℕ :=
  216 -- 6^3 distinct rolls of three 6-sided dice

def successful_rolls_count : ℕ :=
  24 -- number of valid rolls that sum to 10

theorem probability_sum_10 :
  (successful_rolls_count : ℚ) / valid_rolls_count = 1 / 9 := by
  sorry

end probability_sum_10_l238_238206


namespace sum_first_15_odd_from_5_l238_238552

theorem sum_first_15_odd_from_5 : 
  let a₁ := 5 
  let d := 2 
  let n := 15 
  let a₁₅ := a₁ + (n - 1) * d 
  let S := n * (a₁ + a₁₅) / 2 
  S = 285 := by 
  sorry

end sum_first_15_odd_from_5_l238_238552


namespace garden_length_l238_238777

theorem garden_length 
  (W : ℕ) (small_gate_width : ℕ) (large_gate_width : ℕ) (P : ℕ)
  (hW : W = 125)
  (h_small_gate : small_gate_width = 3)
  (h_large_gate : large_gate_width = 10)
  (hP : P = 687) :
  ∃ (L : ℕ), P = 2 * L + 2 * W - (small_gate_width + large_gate_width) ∧ L = 225 := by
  sorry

end garden_length_l238_238777


namespace cos_300_eq_half_l238_238830

-- Define the point on the unit circle at 300 degrees
def point_on_unit_circle_angle_300 : ℝ × ℝ :=
  let θ := 300 * (Real.pi / 180) in
  (Real.cos θ, Real.sin θ)

-- Define the cosine of the angle 300 degrees
def cos_300 : ℝ :=
  Real.cos (300 * (Real.pi / 180))

theorem cos_300_eq_half : cos_300 = 1 / 2 :=
by sorry

end cos_300_eq_half_l238_238830


namespace max_mow_time_l238_238228

-- Define the conditions
def timeToMow (x : ℕ) : Prop := 
  let timeToFertilize := 2 * x
  x + timeToFertilize = 120

-- State the theorem
theorem max_mow_time (x : ℕ) (h : timeToMow x) : x = 40 := by
  sorry

end max_mow_time_l238_238228


namespace probability_is_correct_l238_238403

namespace ProbabilityProof

-- Define the labels on the balls
def balls : List ℕ := [1, 2, 2, 3, 4, 5]

-- Noncomputable because we use real number arithmetic
noncomputable def probability_sum_greater_than_7 : ℚ :=
  let total_combinations := (balls.choose 3).length
  let non_exceeding_combinations := (balls.choose 3).count (fun l => l.sum ≤ 7)
  1 - (non_exceeding_combinations / total_combinations)

theorem probability_is_correct :
  probability_sum_greater_than_7 = 7 / 10 :=
by
  simp [probability_sum_greater_than_7]
  sorry -- Proof omitted

end ProbabilityProof

end probability_is_correct_l238_238403


namespace combined_weight_is_correct_l238_238605

def EvanDogWeight := 63
def IvanDogWeight := EvanDogWeight / 7
def CombinedWeight := EvanDogWeight + IvanDogWeight

theorem combined_weight_is_correct 
: CombinedWeight = 72 :=
by 
  sorry

end combined_weight_is_correct_l238_238605


namespace fraction_to_decimal_l238_238068

theorem fraction_to_decimal : (7 : ℝ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l238_238068


namespace mutually_exclusive_not_opposite_l238_238596

universe u

-- Define the colors and people involved
inductive Color
| black
| red
| white

inductive Person 
| A
| B
| C

-- Define a function that distributes the cards amongst the people
def distributes (cards : List Color) (people : List Person) : People -> Color :=
  sorry

-- Define events as propositions
def A_gets_red (d : Person -> Color) : Prop :=
  d Person.A = Color.red

def B_gets_red (d : Person -> Color) : Prop :=
  d Person.B = Color.red

-- The main theorem stating the problem
theorem mutually_exclusive_not_opposite 
  (d : Person -> Color)
  (h : A_gets_red d → ¬ B_gets_red d) : 
  ¬ ( ∀ (p : Prop), A_gets_red d ↔ p ) → B_gets_red d :=
sorry

end mutually_exclusive_not_opposite_l238_238596


namespace factor_2310_two_digit_numbers_l238_238348

theorem factor_2310_two_digit_numbers :
  ∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 2310 ∧ ∀ (c d : ℕ), 10 ≤ c ∧ c < 100 ∧ 10 ≤ d ∧ d < 100 ∧ c * d = 2310 → (c = a ∧ d = b) ∨ (c = b ∧ d = a) :=
by {
  sorry
}

end factor_2310_two_digit_numbers_l238_238348


namespace greatest_sum_consecutive_integers_lt_500_l238_238727

theorem greatest_sum_consecutive_integers_lt_500 : 
  ∃ n : ℤ, (n * (n + 1) < 500) ∧ n + (n + 1) = 43 := 
by {
  sorry -- Proof needed
}

end greatest_sum_consecutive_integers_lt_500_l238_238727


namespace cos_300_is_half_l238_238816

noncomputable def cos_300_eq_half : Prop :=
  real.cos (2 * real.pi * 5 / 6) = 1 / 2

theorem cos_300_is_half : cos_300_eq_half :=
sorry

end cos_300_is_half_l238_238816


namespace percentage_of_white_chips_l238_238405

theorem percentage_of_white_chips (T : ℕ) (h1 : 3 = 10 * T / 100) (h2 : 12 = 12): (15 / T * 100) = 50 := by
  sorry

end percentage_of_white_chips_l238_238405


namespace inequality_smallest_integer_solution_l238_238394

theorem inequality_smallest_integer_solution (x : ℤ) :
    (9 * x + 8) / 6 - x / 3 ≥ -1 → x ≥ -2 := sorry

end inequality_smallest_integer_solution_l238_238394


namespace solution_l238_238651

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def is_even (g : ℝ → ℝ) : Prop :=
  ∀ y : ℝ, g (-y) = g y

def problem (f g : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * g y) ∧
  (f 0 = 0) ∧
  (∃ x : ℝ, f x ≠ 0)

theorem solution (f g : ℝ → ℝ) (h : problem f g) : is_odd f ∧ is_even g :=
sorry

end solution_l238_238651


namespace find_k_value_l238_238622

theorem find_k_value (x y k : ℝ) 
  (h1 : 2 * x + y = 1) 
  (h2 : x + 2 * y = k - 2) 
  (h3 : x - y = 2) : 
  k = 1 := 
by
  sorry

end find_k_value_l238_238622


namespace lemon_cookies_amount_l238_238966

def cookies_problem 
  (jenny_pb_cookies : ℕ) (jenny_cc_cookies : ℕ) (marcus_pb_cookies : ℕ) (marcus_lemon_cookies : ℕ)
  (total_pb_cookies : ℕ) (total_non_pb_cookies : ℕ) : Prop :=
  jenny_pb_cookies = 40 ∧
  jenny_cc_cookies = 50 ∧
  marcus_pb_cookies = 30 ∧
  total_pb_cookies = jenny_pb_cookies + marcus_pb_cookies ∧
  total_pb_cookies = 70 ∧
  total_non_pb_cookies = jenny_cc_cookies + marcus_lemon_cookies ∧
  total_pb_cookies = total_non_pb_cookies

theorem lemon_cookies_amount
  (jenny_pb_cookies : ℕ) (jenny_cc_cookies : ℕ) (marcus_pb_cookies : ℕ) (marcus_lemon_cookies : ℕ)
  (total_pb_cookies : ℕ) (total_non_pb_cookies : ℕ) :
  cookies_problem jenny_pb_cookies jenny_cc_cookies marcus_pb_cookies marcus_lemon_cookies total_pb_cookies total_non_pb_cookies →
  marcus_lemon_cookies = 20 :=
by
  sorry

end lemon_cookies_amount_l238_238966


namespace jasmine_percentage_is_approx_l238_238440

noncomputable def initial_solution_volume : ℝ := 80
noncomputable def initial_jasmine_percent : ℝ := 0.10
noncomputable def initial_lemon_percent : ℝ := 0.05
noncomputable def initial_orange_percent : ℝ := 0.03
noncomputable def added_jasmine_volume : ℝ := 8
noncomputable def added_water_volume : ℝ := 12
noncomputable def added_lemon_volume : ℝ := 6
noncomputable def added_orange_volume : ℝ := 7

noncomputable def initial_jasmine_volume := initial_solution_volume * initial_jasmine_percent
noncomputable def initial_lemon_volume := initial_solution_volume * initial_lemon_percent
noncomputable def initial_orange_volume := initial_solution_volume * initial_orange_percent
noncomputable def initial_water_volume := initial_solution_volume - (initial_jasmine_volume + initial_lemon_volume + initial_orange_volume)

noncomputable def new_jasmine_volume := initial_jasmine_volume + added_jasmine_volume
noncomputable def new_water_volume := initial_water_volume + added_water_volume
noncomputable def new_lemon_volume := initial_lemon_volume + added_lemon_volume
noncomputable def new_orange_volume := initial_orange_volume + added_orange_volume
noncomputable def new_total_volume := new_jasmine_volume + new_water_volume + new_lemon_volume + new_orange_volume

noncomputable def new_jasmine_percent := (new_jasmine_volume / new_total_volume) * 100

theorem jasmine_percentage_is_approx :
  abs (new_jasmine_percent - 14.16) < 0.01 := sorry

end jasmine_percentage_is_approx_l238_238440


namespace maximum_tangency_circles_l238_238084

/-- Points \( P_1, P_2, \ldots, P_n \) are in the plane
    Real numbers \( r_1, r_2, \ldots, r_n \) are such that the distance between \( P_i \) and \( P_j \) is \( r_i + r_j \) for \( i \ne j \).
    -/
theorem maximum_tangency_circles (n : ℕ) (P : Fin n → ℝ × ℝ) (r : Fin n → ℝ)
  (h : ∀ i j : Fin n, i ≠ j → dist (P i) (P j) = r i + r j) : n ≤ 4 :=
sorry

end maximum_tangency_circles_l238_238084


namespace find_max_problems_l238_238458

def max_problems_in_7_days (P : ℕ → ℕ) : Prop :=
  (∀ i : ℕ, i ∈ Finset.range 7 → P i ≤ 10) ∧
  (∀ i : ℕ, i ∈ Finset.range 5 → (P i > 7) → (P (i + 1) ≤ 5 ∧ P (i + 2) ≤ 5))

theorem find_max_problems : ∃ P : ℕ → ℕ, max_problems_in_7_days P ∧ (Finset.range 7).sum P = 50 :=
by
  sorry

end find_max_problems_l238_238458


namespace brian_gallons_usage_l238_238589

/-
Brian’s car gets 20 miles per gallon. 
On his last trip, he traveled 60 miles. 
How many gallons of gas did he use?
-/

theorem brian_gallons_usage (miles_per_gallon : ℝ) (total_miles : ℝ) (gallons_used : ℝ) 
    (h1 : miles_per_gallon = 20) 
    (h2 : total_miles = 60) 
    (h3 : gallons_used = total_miles / miles_per_gallon) : 
    gallons_used = 3 := 
by
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end brian_gallons_usage_l238_238589


namespace find_ordered_pairs_l238_238132

theorem find_ordered_pairs (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a > b) (h4 : (a - b) ^ (a * b) = a ^ b * b ^ a) :
  (a, b) = (4, 2) := by
  sorry

end find_ordered_pairs_l238_238132


namespace sum_of_roots_l238_238747

theorem sum_of_roots (a b c : ℝ) (h : 6 * a^3 + 7 * a^2 - 12 * a = 0) : 
  - (7 / 6 : ℝ) = -1.17 := 
sorry

end sum_of_roots_l238_238747


namespace mechanism_parts_l238_238429

theorem mechanism_parts (L S : ℕ) (h1 : L + S = 30) (h2 : L ≤ 11) (h3 : S ≤ 19) :
  L = 11 ∧ S = 19 :=
by
  sorry

end mechanism_parts_l238_238429


namespace dice_sum_probability_l238_238194

theorem dice_sum_probability : (∃ s, (∃ x y z : ℕ, 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 ∧ 1 ≤ z ∧ z ≤ 6 ∧ x + y + z = 10)
  ∧ s = {n : ℕ // ∃ x y z : ℕ, 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 ∧ 1 ≤ z ∧ z ≤ 6 ∧ x + y + z = n} ∧ s.card = 24)
  → 24 / 216 = 1 / 9 := 
sorry

end dice_sum_probability_l238_238194


namespace quadratic_point_inequality_l238_238237

theorem quadratic_point_inequality 
  (m y1 y2 : ℝ)
  (hA : y1 = (m - 1)^2)
  (hB : y2 = (m + 1 - 1)^2)
  (hy1_lt_y2 : y1 < y2) :
  m > 1 / 2 :=
by 
  sorry

end quadratic_point_inequality_l238_238237


namespace expected_babies_is_1008_l238_238287

noncomputable def babies_expected_after_loss
  (num_kettles : ℕ)
  (pregnancies_per_kettle : ℕ)
  (babies_per_pregnancy : ℕ)
  (loss_percentage : ℤ) : ℤ :=
  let total_babies := num_kettles * pregnancies_per_kettle * babies_per_pregnancy
  let survival_rate := (100 - loss_percentage) / 100
  total_babies * survival_rate

theorem expected_babies_is_1008 :
  babies_expected_after_loss 12 20 6 30 = 1008 :=
by
  sorry

end expected_babies_is_1008_l238_238287


namespace cos_300_eq_half_l238_238819

theorem cos_300_eq_half : Real.cos (300 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end cos_300_eq_half_l238_238819


namespace product_of_last_two_digits_l238_238354

theorem product_of_last_two_digits (A B : ℕ) (h1 : A + B = 14) (h2 : B = 0 ∨ B = 5) : A * B = 45 :=
sorry

end product_of_last_two_digits_l238_238354


namespace inverse_function_problem_l238_238385

theorem inverse_function_problem
  (f : ℝ → ℝ)
  (f_inv : ℝ → ℝ)
  (h₁ : ∀ x, f (f_inv x) = x)
  (h₂ : ∀ x, f_inv (f x) = x)
  (a b : ℝ)
  (h₃ : f_inv (a - 1) + f_inv (b - 1) = 1) :
  f (a * b) = 3 :=
by
  sorry

end inverse_function_problem_l238_238385


namespace division_of_sums_and_products_l238_238408

theorem division_of_sums_and_products (a b c : ℕ) (h_a : a = 7) (h_b : b = 5) (h_c : c = 3) :
  (a^3 + b^3 + c^3) / (a^2 - a * b + b^2 - b * c + c^2) = 15 := by
  -- proofs go here
  sorry

end division_of_sums_and_products_l238_238408


namespace part1_solution_set_part2_range_of_a_l238_238921

def f (x a : ℝ) := |x - a^2| + |x - (2*a + 1)|

-- Part (1)
theorem part1_solution_set (x : ℝ) :
  ∀ x, f x 2 ≥ 4 ↔ x ≤ 3 / 2 ∨ x ≥ 11 / 2 := sorry

-- Part (2)
theorem part2_range_of_a (a : ℝ) :
  (∀ x, f x a ≥ 4) ↔ (a ≤ -1 ∨ a ≥ 3) := sorry

end part1_solution_set_part2_range_of_a_l238_238921


namespace line_intersects_y_axis_at_eight_l238_238761

theorem line_intersects_y_axis_at_eight :
  ∃ b : ℝ, ∃ f : ℝ → ℝ, (∀ x, f x = 2 * x + b) ∧ f 1 = 10 ∧ f (-9) = -10 ∧ f 0 = 8 :=
by
  -- Definitions and calculations leading to verify the theorem
  sorry

end line_intersects_y_axis_at_eight_l238_238761


namespace sample_size_calculation_l238_238567

theorem sample_size_calculation (n : ℕ) (ratio_A_B_C q_A q_B q_C : ℕ) 
  (ratio_condition : ratio_A_B_C = 2 ∧ ratio_A_B_C * q_A = 2 ∧ ratio_A_B_C * q_B = 3 ∧ ratio_A_B_C * q_C = 5)
  (sample_A_units : q_A = 16) : n = 80 :=
sorry

end sample_size_calculation_l238_238567


namespace difference_between_perfect_and_cracked_l238_238230

def total_eggs : ℕ := 24
def broken_eggs : ℕ := 3
def cracked_eggs : ℕ := 2 * broken_eggs

def perfect_eggs : ℕ := total_eggs - broken_eggs - cracked_eggs
def difference : ℕ := perfect_eggs - cracked_eggs

theorem difference_between_perfect_and_cracked :
  difference = 9 := by
  sorry

end difference_between_perfect_and_cracked_l238_238230


namespace lcm_proof_l238_238142

theorem lcm_proof (a b c : ℕ) (h1 : Nat.lcm a b = 60) (h2 : Nat.lcm a c = 270) : Nat.lcm b c = 540 :=
sorry

end lcm_proof_l238_238142


namespace range_of_a_l238_238326

def proposition_p (a : ℝ) : Prop := a > 1
def proposition_q (a : ℝ) : Prop := 0 < a ∧ a < 4

theorem range_of_a
(a : ℝ)
(h1 : a > 0)
(h2 : ¬ proposition_p a)
(h3 : ¬ proposition_q a)
(h4 : proposition_p a ∨ proposition_q a) :
  (0 < a ∧ a ≤ 1) ∨ (4 ≤ a) :=
by sorry

end range_of_a_l238_238326


namespace fraction_to_decimal_l238_238049

theorem fraction_to_decimal :
  (7 : ℝ) / 16 = 0.4375 := 
by sorry

end fraction_to_decimal_l238_238049


namespace value_of_S_l238_238311

def pseudocode_value : ℕ := 1
def increment (S I : ℕ) : ℕ := S + I

def loop_steps : ℕ :=
  let S := pseudocode_value
  let S := increment S 1
  let S := increment S 3
  let S := increment S 5
  let S := increment S 7
  S

theorem value_of_S : loop_steps = 17 :=
  by sorry

end value_of_S_l238_238311


namespace cos_300_eq_half_l238_238798

theorem cos_300_eq_half : Real.cos (300 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end cos_300_eq_half_l238_238798


namespace valid_integer_values_of_x_l238_238277

theorem valid_integer_values_of_x (x : ℤ) 
  (h1 : 3 < x) (h2 : x < 10)
  (h3 : 5 < x) (h4 : x < 18)
  (h5 : -2 < x) (h6 : x < 9)
  (h7 : 0 < x) (h8 : x < 8) 
  (h9 : x + 1 < 9) : x = 6 ∨ x = 7 :=
by
  sorry

end valid_integer_values_of_x_l238_238277


namespace tan_positive_implies_sin_cos_positive_l238_238496

variables {α : ℝ}

theorem tan_positive_implies_sin_cos_positive (h : Real.tan α > 0) : Real.sin α * Real.cos α > 0 :=
sorry

end tan_positive_implies_sin_cos_positive_l238_238496


namespace fraction_equality_l238_238514

def f (x : ℤ) : ℤ := 3 * x + 2
def g (x : ℤ) : ℤ := 2 * x - 3

theorem fraction_equality :
  (f (g (f 3))) / (g (f (g 3))) = 59 / 35 := 
by
  sorry

end fraction_equality_l238_238514


namespace price_per_maple_tree_l238_238946

theorem price_per_maple_tree 
  (cabin_price : ℕ) (initial_cash : ℕ) (remaining_cash : ℕ)
  (num_cypress : ℕ) (price_cypress : ℕ)
  (num_pine : ℕ) (price_pine : ℕ)
  (num_maple : ℕ) 
  (total_raised_from_trees : ℕ) :
  cabin_price = 129000 ∧ 
  initial_cash = 150 ∧ 
  remaining_cash = 350 ∧ 
  num_cypress = 20 ∧ 
  price_cypress = 100 ∧ 
  num_pine = 600 ∧ 
  price_pine = 200 ∧ 
  num_maple = 24 ∧ 
  total_raised_from_trees = 129350 - initial_cash → 
  (price_maple : ℕ) = 300 :=
by 
  sorry

end price_per_maple_tree_l238_238946


namespace cole_drive_time_l238_238301

noncomputable def time_to_drive_to_work (D : ℝ) : ℝ :=
  D / 50

theorem cole_drive_time (D : ℝ) (h₁ : time_to_drive_to_work D + (D / 110) = 2) : time_to_drive_to_work D * 60 = 82.5 :=
by
  sorry

end cole_drive_time_l238_238301


namespace cos_300_eq_cos_300_l238_238825

open Real

theorem cos_300_eq (angle_1 angle_2 : ℝ) (h1 : angle_1 = 300 * π / 180) (h2 : angle_2 = 60 * π / 180) :
  cos angle_1 = cos angle_2 :=
by
  rw [←h1, ←h2, cos_sub, cos_pi_div_six]
  norm_num
  sorry

theorem cos_300 : cos (300 * π / 180) = 1 / 2 :=
by
  apply cos_300_eq (300 * π / 180) (60 * π / 180)
  norm_num
  norm_num
  sorry

end cos_300_eq_cos_300_l238_238825


namespace central_angle_of_sector_l238_238151

variable (r θ : ℝ)
variable (r_pos : 0 < r) (θ_pos : 0 < θ)

def perimeter_eq : Prop := 2 * r + r * θ = 5
def area_eq : Prop := (1 / 2) * r^2 * θ = 1

theorem central_angle_of_sector :
  perimeter_eq r θ ∧ area_eq r θ → θ = 1 / 2 :=
sorry

end central_angle_of_sector_l238_238151


namespace intersection_of_A_and_B_l238_238971

-- Define the sets A and B
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {x | ∃ a ∈ A, x = 2 * a - 1}

-- The main statement to prove
theorem intersection_of_A_and_B : A ∩ B = {1, 3} :=
by {
  sorry
}

end intersection_of_A_and_B_l238_238971


namespace percentage_decrease_increase_l238_238524

theorem percentage_decrease_increase (S : ℝ) (x : ℝ) (h : S > 0) :
  S * (1 - x / 100) * (1 + x / 100) = S * (64 / 100) → x = 6 :=
by
  sorry

end percentage_decrease_increase_l238_238524


namespace part1_solution_set_part2_range_of_a_l238_238873

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1_solution_set (x : ℝ) : 
  (f x 2 ≥ 4) ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
sorry

theorem part2_range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 4) → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end part1_solution_set_part2_range_of_a_l238_238873


namespace seats_needed_on_bus_l238_238999

variable (f t tr dr c h : ℕ)

def flute_players := 5
def trumpet_players := 3 * flute_players
def trombone_players := trumpet_players - 8
def drummers := trombone_players + 11
def clarinet_players := 2 * flute_players
def french_horn_players := trombone_players + 3

theorem seats_needed_on_bus :
  f = 5 →
  t = 3 * f →
  tr = t - 8 →
  dr = tr + 11 →
  c = 2 * f →
  h = tr + 3 →
  f + t + tr + dr + c + h = 65 :=
by
  sorry

end seats_needed_on_bus_l238_238999


namespace fraction_to_decimal_l238_238051

theorem fraction_to_decimal :
  (7 : ℝ) / 16 = 0.4375 := 
by sorry

end fraction_to_decimal_l238_238051


namespace find_j_l238_238759

theorem find_j (n j : ℕ) (h_n_pos : n > 0) (h_j_pos : j > 0) (h_rem : n % j = 28) (h_div : n / j = 142 ∧ (↑n / ↑j : ℝ) = 142.07) : j = 400 :=
by {
  sorry
}

end find_j_l238_238759


namespace jake_steps_per_second_l238_238113

/-
Conditions:
1. Austin and Jake start descending from the 9th floor at the same time.
2. The stairs have 30 steps across each floor.
3. The elevator takes 1 minute (60 seconds) to reach the ground floor.
4. Jake reaches the ground floor 30 seconds after Austin.
5. Jake descends 8 floors to reach the ground floor.
-/

def floors : ℕ := 8
def steps_per_floor : ℕ := 30
def time_elevator : ℕ := 60 -- in seconds
def additional_time_jake : ℕ := 30 -- in seconds

def total_time_jake := time_elevator + additional_time_jake -- in seconds
def total_steps := floors * steps_per_floor

def steps_per_second_jake := (total_steps : ℚ) / (total_time_jake : ℚ)

theorem jake_steps_per_second :
  steps_per_second_jake = 2.67 := by
  sorry

end jake_steps_per_second_l238_238113


namespace hall_area_l238_238434

theorem hall_area {L W : ℝ} (h₁ : W = 0.5 * L) (h₂ : L - W = 20) : L * W = 800 := by
  sorry

end hall_area_l238_238434


namespace dice_sum_probability_l238_238201

theorem dice_sum_probability : 
  let die_faces := {1, 2, 3, 4, 5, 6};
      successful_outcomes := { (x, y, z) | x ∈ die_faces ∧ y ∈ die_faces ∧ z ∈ die_faces ∧ x + y + z = 10 }.toFinset.card;
      total_outcomes := (finset.range 6).card ^ 3
  in  successful_outcomes / total_outcomes = 1 / 9 :=
by
  let die_faces := {1, 2, 3, 4, 5, 6};
  let successful_outcomes := { (x, y, z) | x ∈ die_faces ∧ y ∈ die_faces ∧ z ∈ die_faces ∧ x + y + z = 10 }.toFinset.card;
  let total_outcomes := (finset.range 6).card ^ 3;
  sorry

end dice_sum_probability_l238_238201


namespace minimum_number_of_participants_l238_238261

theorem minimum_number_of_participants (a1 a2 a3 a4 : ℕ) (h1 : a1 = 90) (h2 : a2 = 50) (h3 : a3 = 40) (h4 : a4 = 20) 
  (h5 : ∀ (n : ℕ), n * 2 ≥ a1 + a2 + a3 + a4) : ∃ n, (n ≥ 100) :=
by 
  use 100
  sorry

end minimum_number_of_participants_l238_238261


namespace binomial_standard_deviation_l238_238842

noncomputable def standard_deviation_binomial (n : ℕ) (p : ℝ) : ℝ :=
  Real.sqrt (n * p * (1 - p))

theorem binomial_standard_deviation (n : ℕ) (p : ℝ) (hn : 0 ≤ n) (hp : 0 ≤ p) (hp1: p ≤ 1) :
  standard_deviation_binomial n p = Real.sqrt (n * p * (1 - p)) :=
by
  sorry

end binomial_standard_deviation_l238_238842


namespace arith_seq_problem_l238_238218

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
  ∀ n, a n = a1 + (n - 1) * d

theorem arith_seq_problem 
  (a : ℕ → ℝ) (a1 d : ℝ)
  (h1 : arithmetic_sequence a a1 d)
  (h2 : a 1 + 3 * a 8 + a 15 = 120) : 
  3 * a 9 - a 11 = 48 :=
by 
  sorry

end arith_seq_problem_l238_238218


namespace light_intensity_at_10_m_l238_238254

theorem light_intensity_at_10_m (k : ℝ) (d1 d2 : ℝ) (I1 I2 : ℝ)
  (h1: I1 = k / d1^2) (h2: I1 = 200) (h3: d1 = 5) (h4: d2 = 10) :
  I2 = k / d2^2 → I2 = 50 :=
sorry

end light_intensity_at_10_m_l238_238254


namespace probability_sum_10_l238_238204

def is_valid_roll (d1 d2 d3 : ℕ) : Prop :=
  1 ≤ d1 ∧ d1 ≤ 6 ∧ 1 ≤ d2 ∧ d2 ≤ 6 ∧ 1 ≤ d3 ∧ d3 ≤ 6

def sum_is_10 (d1 d2 d3 : ℕ) : Prop :=
  d1 + d2 + d3 = 10

def valid_rolls_count : ℕ :=
  216 -- 6^3 distinct rolls of three 6-sided dice

def successful_rolls_count : ℕ :=
  24 -- number of valid rolls that sum to 10

theorem probability_sum_10 :
  (successful_rolls_count : ℚ) / valid_rolls_count = 1 / 9 := by
  sorry

end probability_sum_10_l238_238204


namespace mass_percentage_Al_in_mixture_l238_238846

/-- Define molar masses for the respective compounds -/
def molar_mass_AlCl3 : ℝ := 133.33
def molar_mass_Al2SO4_3 : ℝ := 342.17
def molar_mass_AlOH3 : ℝ := 78.01

/-- Define masses of respective compounds given in grams -/
def mass_AlCl3 : ℝ := 50
def mass_Al2SO4_3 : ℝ := 70
def mass_AlOH3 : ℝ := 40

/-- Define molar mass of Al -/
def molar_mass_Al : ℝ := 26.98

theorem mass_percentage_Al_in_mixture :
  (mass_AlCl3 / molar_mass_AlCl3 * molar_mass_Al +
   mass_Al2SO4_3 / molar_mass_Al2SO4_3 * (2 * molar_mass_Al) +
   mass_AlOH3 / molar_mass_AlOH3 * molar_mass_Al) / 
  (mass_AlCl3 + mass_Al2SO4_3 + mass_AlOH3) * 100 
  = 21.87 := by
  sorry

end mass_percentage_Al_in_mixture_l238_238846


namespace maximal_integer_set_size_l238_238833

-- Definitions of relevant sets and properties
def valid_integers (s : Finset (Finset ℕ)) : Prop :=
  ∀ x ∈ s, ∀ y ∈ s, x ≠ y → (x ∩ y ≠ ∅)

def maximal_valid_size : ℕ :=
  2^5

-- Prove the maximal size of such a set is 32
theorem maximal_integer_set_size :
  ∃ (s : Finset (Finset ℕ)), 
    s.card = maximal_valid_size ∧
    valid_integers s ∧
    (∀ x ∈ s, x ⊆ {1, 2, 3, 4, 5, 6}) ∧
    (∀ x ∈ s, ∀ i j ∈ x, i < j) ∧
    (∀ i ∈ {1, 2, 3, 4, 5, 6}, ∃ y ∈ s, i ∉ y) :=
begin
  sorry
end

end maximal_integer_set_size_l238_238833


namespace ratio_of_eggs_used_l238_238770

theorem ratio_of_eggs_used (total_eggs : ℕ) (eggs_left : ℕ) (eggs_broken : ℕ) (eggs_bought : ℕ) :
  total_eggs = 72 →
  eggs_left = 21 →
  eggs_broken = 15 →
  eggs_bought = total_eggs - (eggs_left + eggs_broken) →
  (eggs_bought / total_eggs) = 1 / 2 :=
by
  intros h1 h2 h3 h4
  sorry

end ratio_of_eggs_used_l238_238770


namespace percentage_increase_from_second_to_third_building_l238_238455

theorem percentage_increase_from_second_to_third_building :
  let first_building_units := 4000
  let second_building_units := (2 / 5 : ℝ) * first_building_units
  let total_units := 7520
  let third_building_units := total_units - (first_building_units + second_building_units)
  (third_building_units - second_building_units) / second_building_units * 100 = 20 := by
  let first_building_units := 4000
  let second_building_units : ℝ := (2 / 5 : ℝ) * first_building_units
  let total_units := 7520
  let third_building_units := total_units - (first_building_units + second_building_units)
  have H : (third_building_units - second_building_units) / second_building_units * 100 = 20 := sorry
  exact H

end percentage_increase_from_second_to_third_building_l238_238455


namespace share_equally_l238_238597

variable (Emani Howard : ℕ)
axiom h1 : Emani = 150
axiom h2 : Emani = Howard + 30

theorem share_equally : (Emani + Howard) / 2 = 135 :=
by sorry

end share_equally_l238_238597


namespace part1_solution_set_part2_range_of_a_l238_238871

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1_solution_set (x : ℝ) : 
  (f x 2 ≥ 4) ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
sorry

theorem part2_range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 4) → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end part1_solution_set_part2_range_of_a_l238_238871


namespace jessica_walks_distance_l238_238511

theorem jessica_walks_distance (rate time : ℝ) (h_rate : rate = 4) (h_time : time = 2) :
  rate * time = 8 :=
by 
  rw [h_rate, h_time]
  norm_num

end jessica_walks_distance_l238_238511


namespace prism_edges_l238_238775

theorem prism_edges (n : ℕ) (h1 : n > 310) (h2 : n < 320) (h3 : n % 2 = 1) : n = 315 := by
  sorry

end prism_edges_l238_238775


namespace part1_solution_set_part2_values_of_a_l238_238867

-- Parameters and definitions for the function and conditions
def f (x a : ℝ) := |x - a^2| + |x - (2*a + 1)|

-- Part (1) When a = 2, the inequality's solution set
theorem part1_solution_set (x : ℝ) : f x 2 ≥ 4 ↔ (x ≤ 3/2 ∨ x ≥ 11/2) := 
sorry

-- Part (2) Range of values for a given f(x, a) ≥ 4
theorem part2_values_of_a (a : ℝ) : 
∀ x, f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end part1_solution_set_part2_values_of_a_l238_238867


namespace cos_300_is_half_l238_238789

noncomputable def cos_300_degrees : ℝ :=
  real.cos (300 * real.pi / 180)

theorem cos_300_is_half :
  cos_300_degrees = 1 / 2 :=
sorry

end cos_300_is_half_l238_238789


namespace cos_300_is_half_l238_238790

noncomputable def cos_300_degrees : ℝ :=
  real.cos (300 * real.pi / 180)

theorem cos_300_is_half :
  cos_300_degrees = 1 / 2 :=
sorry

end cos_300_is_half_l238_238790


namespace greatest_possible_sum_consecutive_product_lt_500_l238_238737

noncomputable def largest_sum_consecutive_product_lt_500 : ℕ :=
  let n := nat.sub ((nat.sqrt 500) + 1) 1 in
  n + (n + 1)

theorem greatest_possible_sum_consecutive_product_lt_500 :
  (∃ (n : ℕ), n * (n + 1) < 500 ∧ largest_sum_consecutive_product_lt_500 = (n + (n + 1))) →
  largest_sum_consecutive_product_lt_500 = 43 := by
  sorry

end greatest_possible_sum_consecutive_product_lt_500_l238_238737


namespace find_b_l238_238495

theorem find_b (c b : ℤ) (h : ∃ k : ℤ, (x^2 - x - 1) * (c * x - 3) = c * x^3 + b * x^2 + 3) : b = -6 :=
by
  sorry

end find_b_l238_238495


namespace focus_of_parabola_l238_238665

theorem focus_of_parabola (x y : ℝ) (h : y = 2 * x^2) : 
  ∃ f : ℝ × ℝ, f = (0, 1 / 8) :=
by
  sorry

end focus_of_parabola_l238_238665


namespace fraction_to_decimal_l238_238046

theorem fraction_to_decimal :
  (7 / 16 : ℝ) = 0.4375 := 
begin
  -- placeholder for the proof
  sorry
end

end fraction_to_decimal_l238_238046


namespace smallest_k_with_properties_l238_238841

noncomputable def exists_coloring_and_function (k : ℕ) : Prop :=
  ∃ (colors : ℤ → Fin k) (f : ℤ → ℤ),
    (∀ m n : ℤ, colors m = colors n → f (m + n) = f m + f n) ∧
    (∃ m n : ℤ, f (m + n) ≠ f m + f n)

theorem smallest_k_with_properties : ∃ (k : ℕ), k > 0 ∧ exists_coloring_and_function k ∧
                                         (∀ k' : ℕ, k' > 0 ∧ k' < k → ¬ exists_coloring_and_function k') :=
by
  sorry

end smallest_k_with_properties_l238_238841


namespace problem_statement_l238_238494

def f (x : ℝ) : ℝ := 3 * x^2 - 2
def k (x : ℝ) : ℝ := -2 * x^3 + 2

theorem problem_statement : f (k 2) = 586 := by
  sorry

end problem_statement_l238_238494


namespace total_students_surveyed_l238_238570

variable (T : ℕ)
variable (F : ℕ)

theorem total_students_surveyed :
  (F = 20 + 60) → (F = 40 * (T / 100)) → (T = 200) :=
by
  intros h1 h2
  sorry

end total_students_surveyed_l238_238570


namespace a_is_minus_one_l238_238333

theorem a_is_minus_one (a : ℤ) (h1 : 2 * a + 1 < 0) (h2 : 2 + a > 0) : a = -1 := 
by
  sorry

end a_is_minus_one_l238_238333


namespace fractional_to_decimal_l238_238057

theorem fractional_to_decimal : (7 / 16 : ℚ) = 0.4375 := by
  sorry

end fractional_to_decimal_l238_238057


namespace abs_diff_squares_105_95_l238_238689

theorem abs_diff_squares_105_95 : abs ((105 : ℤ)^2 - (95 : ℤ)^2) = 2000 := 
by
  sorry

end abs_diff_squares_105_95_l238_238689


namespace sum_of_distinct_integers_eq_36_l238_238975

theorem sum_of_distinct_integers_eq_36
  (p q r s t : ℤ)
  (hpq : p ≠ q) (hpr : p ≠ r) (hps : p ≠ s) (hpt : p ≠ t)
  (hqr : q ≠ r) (hqs : q ≠ s) (hqt : q ≠ t)
  (hrs : r ≠ s) (hrt : r ≠ t)
  (hst : s ≠ t)
  (h : (8 - p) * (8 - q) * (8 - r) * (8 - s) * (8 - t) = 80) :
  p + q + r + s + t = 36 :=
by
  sorry

end sum_of_distinct_integers_eq_36_l238_238975


namespace unique_solution_positive_n_l238_238469

theorem unique_solution_positive_n (n : ℝ) : 
  ( ∃ x : ℝ, 4 * x^2 + n * x + 16 = 0 ∧ ∀ y : ℝ, 4 * y^2 + n * y + 16 = 0 → y = x ) → n = 16 := 
by {
  sorry
}

end unique_solution_positive_n_l238_238469


namespace find_highest_m_l238_238101

def reverse_num (m : ℕ) : ℕ :=
  let digits := m.toString.data.toList.reverse
  digits.foldl (λ acc c => acc * 10 + c.toNat - '0'.toNat) 0

theorem find_highest_m (m : ℕ) :
  (1000 ≤ m ∧ m ≤ 9999) ∧
  (1000 ≤ reverse_num m ∧ reverse_num m ≤ 9999) ∧
  (m % 36 = 0 ∧ reverse_num m % 36 = 0) ∧
  (m % 7 = 0) →
  m = 5796 :=
by
  sorry

end find_highest_m_l238_238101


namespace average_speed_is_70_l238_238428

noncomputable def average_speed (d₁ d₂ d₃ d₄ t₁ t₂ t₃ t₄ : ℝ) : ℝ :=
  (d₁ + d₂ + d₃ + d₄) / (t₁ + t₂ + t₃ + t₄)

theorem average_speed_is_70 :
  let d₁ := 30
  let s₁ := 60
  let t₁ := d₁ / s₁
  let d₂ := 35
  let s₂ := 70
  let t₂ := d₂ / s₂
  let d₃ := 80
  let t₃ := 1
  let s₃ := d₃ / t₃
  let s₄ := 55
  let t₄ := 20/60.0
  let d₄ := s₄ * t₄
  average_speed d₁ d₂ d₃ d₄ t₁ t₂ t₃ t₄ = 70 :=
by
  sorry

end average_speed_is_70_l238_238428


namespace randy_fifth_quiz_score_l238_238521

def scores : List ℕ := [90, 98, 92, 94]

def goal_average : ℕ := 94

def total_points (n : ℕ) (avg : ℕ) : ℕ := n * avg

def current_points (l : List ℕ) : ℕ := l.sum

def needed_score (total current : ℕ) : ℕ := total - current

theorem randy_fifth_quiz_score :
  needed_score (total_points 5 goal_average) (current_points scores) = 96 :=
by 
  sorry

end randy_fifth_quiz_score_l238_238521


namespace fraction_to_decimal_l238_238012

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l238_238012


namespace total_cost_of_shirts_is_24_l238_238376

-- Definitions based on conditions
def cost_first_shirt : ℕ := 15
def cost_difference : ℕ := 6
def cost_second_shirt : ℕ := cost_first_shirt - cost_difference

-- The proof problem statement: Calculate total cost given the conditions
theorem total_cost_of_shirts_is_24 : cost_first_shirt + cost_second_shshirt = 24 :=
by
  sorry

end total_cost_of_shirts_is_24_l238_238376


namespace greatest_sum_of_consecutive_integers_l238_238706

def consecutiveSumCondition (n : ℤ) : Prop :=
  n * (n + 1) < 500 

theorem greatest_sum_of_consecutive_integers : 
  ∃ n : ℤ, consecutiveSumCondition n ∧ ∀ m : ℤ, consecutiveSumCondition m → n + (n + 1) ≥ m + (m + 1) :=
sorry

end greatest_sum_of_consecutive_integers_l238_238706


namespace max_n_for_factorization_l238_238840

theorem max_n_for_factorization (A B n : ℤ) (AB_cond : A * B = 48) (n_cond : n = 5 * B + A) :
  n ≤ 241 :=
by
  sorry

end max_n_for_factorization_l238_238840


namespace correct_operation_A_l238_238559

-- Definitions for the problem
def division_rule (a : ℝ) (m n : ℕ) : Prop := a^m / a^n = a^(m - n)
def multiplication_rule (a : ℝ) (m n : ℕ) : Prop := a^m * a^n = a^(m + n)
def power_rule (a : ℝ) (m n : ℕ) : Prop := (a^m)^n = a^(m * n)
def addition_like_terms_rule (a : ℝ) (m : ℕ) : Prop := a^m + a^m = 2 * a^m

-- The theorem to prove
theorem correct_operation_A (a : ℝ) : division_rule a 4 2 :=
by {
  sorry
}

end correct_operation_A_l238_238559


namespace find_compound_interest_principal_l238_238676

noncomputable def SI (P R T: ℝ) := (P * R * T) / 100
noncomputable def CI (P R T: ℝ) := P * (1 + R / 100)^T - P

theorem find_compound_interest_principal :
  let SI_amount := 3500.000000000004
  let SI_years := 2
  let SI_rate := 6
  let CI_years := 2
  let CI_rate := 10
  let SI_value := SI SI_amount SI_rate SI_years
  let P := 4000
  (SI_value = (CI P CI_rate CI_years) / 2) →
  P = 4000 :=
by
  intros
  sorry

end find_compound_interest_principal_l238_238676


namespace milk_leftover_l238_238302

theorem milk_leftover 
  (total_milk : ℕ := 24)
  (kids_percent : ℝ := 0.80)
  (cooking_percent : ℝ := 0.60)
  (neighbor_percent : ℝ := 0.25)
  (husband_percent : ℝ := 0.06) :
  let milk_after_kids := total_milk * (1 - kids_percent)
  let milk_after_cooking := milk_after_kids * (1 - cooking_percent)
  let milk_after_neighbor := milk_after_cooking * (1 - neighbor_percent)
  let milk_after_husband := milk_after_neighbor * (1 - husband_percent)
  milk_after_husband = 1.3536 :=
by 
  -- skip the proof for simplicity
  sorry

end milk_leftover_l238_238302


namespace percentage_error_in_area_l238_238579

-- Definitions based on conditions
def actual_side (s : ℝ) := s
def measured_side (s : ℝ) := s * 1.01
def actual_area (s : ℝ) := s^2
def calculated_area (s : ℝ) := (measured_side s)^2

-- Theorem statement of the proof problem
theorem percentage_error_in_area (s : ℝ) : 
  (calculated_area s - actual_area s) / actual_area s * 100 = 2.01 := 
by 
  -- Proof is omitted
  sorry

end percentage_error_in_area_l238_238579


namespace count_sums_of_two_cubes_lt_400_l238_238948

theorem count_sums_of_two_cubes_lt_400 : 
  ∃ (s : Finset ℕ), 
    (∀ n ∈ s, ∃ a b, 1 ≤ a ∧ a ≤ 7 ∧ 1 ≤ b ∧ b ≤ 7 ∧ n = a^3 + b^3 ∧ (Odd a ∨ Odd b) ∧ n < 400) ∧
    s.card = 15 :=
by 
  sorry

end count_sums_of_two_cubes_lt_400_l238_238948


namespace fraction_to_decimal_l238_238018

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l238_238018


namespace dice_sum_10_probability_l238_238168

open Probability

noncomputable def probability_sum_10 : ℚ := 
  let total_outcomes := 216
  let favorable_outcomes := 21
  favorable_outcomes / total_outcomes

theorem dice_sum_10_probability :
  probability (λ (x : Fin 6 × Fin 6 × Fin 6), x.1.val + x.2.val + x.3.val + 3 = 10) = probability_sum_10 := 
sorry

end dice_sum_10_probability_l238_238168


namespace cos_300_eq_half_l238_238821

theorem cos_300_eq_half : Real.cos (300 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end cos_300_eq_half_l238_238821


namespace fraction_to_decimal_l238_238067

theorem fraction_to_decimal : (7 : ℝ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l238_238067


namespace fraction_to_decimal_l238_238024

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l238_238024


namespace greatest_sum_of_consecutive_integers_product_less_500_l238_238715

theorem greatest_sum_of_consecutive_integers_product_less_500 :
  ∃ n : ℤ, n * (n + 1) < 500 ∧ (n + (n + 1)) = 43 :=
by
  sorry

end greatest_sum_of_consecutive_integers_product_less_500_l238_238715


namespace correct_sample_size_l238_238566

variable {StudentScore : Type} {scores : Finset StudentScore} (extract_sample : Finset StudentScore → Finset StudentScore)

noncomputable def is_correct_statement : Prop :=
  ∀ (total_scores : Finset StudentScore) (sample_scores : Finset StudentScore),
  (total_scores.card = 1000) →
  (extract_sample total_scores = sample_scores) →
  (sample_scores.card = 100) →
  sample_scores.card = 100

theorem correct_sample_size (total_scores sample_scores : Finset StudentScore)
  (H_total : total_scores.card = 1000)
  (H_sample : extract_sample total_scores = sample_scores)
  (H_card : sample_scores.card = 100) :
  sample_scores.card = 100 :=
sorry

end correct_sample_size_l238_238566


namespace max_value_of_quadratic_on_interval_l238_238346

theorem max_value_of_quadratic_on_interval : 
  ∃ (x : ℝ), -2 ≤ x ∧ x ≤ 2 ∧ (∀ y, (∃ x, -2 ≤ x ∧ x ≤ 2 ∧ y = (x + 1)^2 - 4) → y ≤ 5) :=
sorry

end max_value_of_quadratic_on_interval_l238_238346


namespace total_distance_l238_238512

/--
John's journey is from point (-3, 4) to (2, 2) to (6, -3).
Prove that the total distance John travels is the sum of distances
from (-3, 4) to (2, 2) and from (2, 2) to (6, -3).
-/
theorem total_distance : 
  let d1 := Real.sqrt ((-3 - 2)^2 + (4 - 2)^2)
  let d2 := Real.sqrt ((6 - 2)^2 + (-3 - 2)^2)
  d1 + d2 = Real.sqrt 29 + Real.sqrt 41 :=
by
  sorry

end total_distance_l238_238512


namespace find_xy_integers_l238_238313

theorem find_xy_integers (x y : ℤ) (h : x^3 + 2 * x * y = 7) :
  (x, y) = (-7, -25) ∨ (x, y) = (-1, -4) ∨ (x, y) = (1, 3) ∨ (x, y) = (7, -24) :=
sorry

end find_xy_integers_l238_238313


namespace goshawk_eurasian_reserve_hawks_l238_238634

variable (H P : ℝ)

theorem goshawk_eurasian_reserve_hawks :
  P = 100 ∧
  (35 / 100) * P = P - (H + (40 / 100) * (P - H) + (25 / 100) * (40 / 100) * (P - H))
    → H = 25 :=
by sorry

end goshawk_eurasian_reserve_hawks_l238_238634


namespace cos_300_eq_half_l238_238807

theorem cos_300_eq_half : Real.cos (300 * Real.pi / 180) = 1 / 2 := 
by sorry

end cos_300_eq_half_l238_238807


namespace abs_diff_squares_l238_238691

theorem abs_diff_squares (a b : ℤ) (h_a : a = 105) (h_b : b = 95):
  |a^2 - b^2| = 2000 := by
  sorry

end abs_diff_squares_l238_238691


namespace rectangle_perimeter_l238_238535

theorem rectangle_perimeter {y x : ℝ} (hxy : x < y) : 
  2 * (y - x) + 2 * x = 2 * y :=
by
  sorry

end rectangle_perimeter_l238_238535


namespace cos_300_eq_half_l238_238805

theorem cos_300_eq_half :
  let θ := 300
  let ref_θ := θ - 360 * (θ // 360) -- Reference angle in the proper range
  θ = 300 ∧ ref_θ = 60 ∧ cos 60 = 1/2 ∧ 300 > 270 → cos 300 = 1/2 :=
by
  intros
  sorry

end cos_300_eq_half_l238_238805


namespace cos_300_eq_half_l238_238829

-- Define the point on the unit circle at 300 degrees
def point_on_unit_circle_angle_300 : ℝ × ℝ :=
  let θ := 300 * (Real.pi / 180) in
  (Real.cos θ, Real.sin θ)

-- Define the cosine of the angle 300 degrees
def cos_300 : ℝ :=
  Real.cos (300 * (Real.pi / 180))

theorem cos_300_eq_half : cos_300 = 1 / 2 :=
by sorry

end cos_300_eq_half_l238_238829


namespace total_trees_in_gray_regions_l238_238431

theorem total_trees_in_gray_regions (trees_rectangle1 trees_rectangle2 trees_rectangle3 trees_gray1 trees_gray2 trees_total : ℕ)
  (h1 : trees_rectangle1 = 100)
  (h2 : trees_rectangle2 = 90)
  (h3 : trees_rectangle3 = 82)
  (h4 : trees_total = 82)
  (h_gray1 : trees_gray1 = trees_rectangle1 - trees_total)
  (h_gray2 : trees_gray2 = trees_rectangle2 - trees_total)
  : trees_gray1 + trees_gray2 = 26 := 
sorry

end total_trees_in_gray_regions_l238_238431


namespace trim_hedges_purpose_l238_238536

-- Given possible answers
inductive Answer
| A : Answer
| B : Answer
| C : Answer
| D : Answer

-- Define the purpose of trimming hedges
def trimmingHedges : Answer :=
  Answer.B

-- Formal problem statement
theorem trim_hedges_purpose : trimmingHedges = Answer.B :=
  sorry

end trim_hedges_purpose_l238_238536


namespace qr_length_is_correct_l238_238505

/-- Define points and segments in the triangle. -/
structure Point :=
(x : ℝ)
(y : ℝ)

structure Triangle :=
(P Q R : Point)

def PQ_length (T : Triangle) : ℝ :=
(T.Q.x - T.P.x) * (T.Q.x - T.P.x) + (T.Q.y - T.P.y) * (T.Q.y - T.P.y)

def PR_length (T : Triangle) : ℝ :=
(T.R.x - T.P.x) * (T.R.x - T.P.x) + (T.R.y - T.P.y) * (T.R.y - T.P.y)

def QR_length (T : Triangle) : ℝ :=
(T.R.x - T.Q.x) * (T.R.x - T.Q.x) + (T.R.y - T.Q.y) * (T.R.y - T.Q.y)

noncomputable def XZ_length (T : Triangle) (X Y Z : Point) : ℝ :=
(PQ_length T)^(1/2) -- Assume the least length of XZ that follows the given conditions

theorem qr_length_is_correct (T : Triangle) :
  PQ_length T = 4*4 → 
  XZ_length T T.P T.Q T.R = 3.2 →
  QR_length T = 4*4 :=
sorry

end qr_length_is_correct_l238_238505


namespace sum_of_factors_30_l238_238271

def sum_of_factors (n : Nat) : Nat :=
  (List.range (n + 1)).filter (λ x => n % x = 0) |>.sum

theorem sum_of_factors_30 : sum_of_factors 30 = 72 := by
  sorry

end sum_of_factors_30_l238_238271


namespace gcd_b2_add_11b_add_28_b_add_6_eq_2_l238_238335

theorem gcd_b2_add_11b_add_28_b_add_6_eq_2 {b : ℤ} (h : ∃ k : ℤ, b = 1836 * k) : 
  Int.gcd (b^2 + 11 * b + 28) (b + 6) = 2 := 
by
  sorry

end gcd_b2_add_11b_add_28_b_add_6_eq_2_l238_238335


namespace arithmetic_sequence_ratio_l238_238347

-- Define conditions
def sum_ratios (A_n B_n : ℕ → ℚ) (n : ℕ) : Prop := (A_n n) / (B_n n) = (4 * n + 2) / (5 * n - 5)
def arithmetic_sequences (a_n b_n : ℕ → ℚ) : Prop :=
  ∃ A_n B_n : ℕ → ℚ,
    (∀ n, A_n n = n * (a_n 1) + (n * (n - 1) / 2) * (a_n 2 - a_n 1)) ∧
    (∀ n, B_n n = n * (b_n 1) + (n * (n - 1) / 2) * (b_n 2 - b_n 1)) ∧
    ∀ n, sum_ratios A_n B_n n

-- Theorem to be proven
theorem arithmetic_sequence_ratio
  (a_n b_n : ℕ → ℚ)
  (h : arithmetic_sequences a_n b_n) :
  (a_n 5 + a_n 13) / (b_n 5 + b_n 13) = 7 / 8 :=
sorry

end arithmetic_sequence_ratio_l238_238347


namespace construct_80_construct_160_construct_20_l238_238681

-- Define the notion of constructibility from an angle
inductive Constructible : ℝ → Prop
| base (a : ℝ) : a = 40 → Constructible a
| add (a b : ℝ) : Constructible a → Constructible b → Constructible (a + b)
| sub (a b : ℝ) : Constructible a → Constructible b → Constructible (a - b)

-- Lean statements for proving the constructibility
theorem construct_80 : Constructible 80 :=
sorry

theorem construct_160 : Constructible 160 :=
sorry

theorem construct_20 : Constructible 20 :=
sorry

end construct_80_construct_160_construct_20_l238_238681


namespace problem_inequality_l238_238143

variable (a b : ℝ)

theorem problem_inequality (h1 : a < 0) (h2 : -1 < b) (h3 : b < 0) : ab > ab^2 ∧ ab^2 > a :=
by
  sorry

end problem_inequality_l238_238143


namespace glen_animals_total_impossible_l238_238402

theorem glen_animals_total_impossible (t : ℕ) :
  ¬ (∃ t : ℕ, 41 * t = 108) := sorry

end glen_animals_total_impossible_l238_238402


namespace average_of_first_12_is_14_l238_238250

-- Definitions based on given conditions
def average_of_25 := 19
def sum_of_25 := average_of_25 * 25

def average_of_last_12 := 17
def sum_of_last_12 := average_of_last_12 * 12

def result_13 := 103

-- Main proof statement to be checked
theorem average_of_first_12_is_14 (A : ℝ) (h1 : sum_of_25 = sum_of_25) (h2 : sum_of_last_12 = sum_of_last_12) (h3 : result_13 = 103) :
  (A * 12 + result_13 + sum_of_last_12 = sum_of_25) → (A = 14) :=
by
  sorry

end average_of_first_12_is_14_l238_238250


namespace cos_300_eq_one_half_l238_238795

theorem cos_300_eq_one_half :
  Real.cos (300 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end cos_300_eq_one_half_l238_238795


namespace fraction_to_decimal_l238_238079

theorem fraction_to_decimal : (7 : ℚ) / 16 = 4375 / 10000 := by
  sorry

end fraction_to_decimal_l238_238079


namespace candy_mixture_solution_l238_238107

theorem candy_mixture_solution :
  ∃ x y : ℝ, 18 * x + 10 * y = 1500 ∧ x + y = 100 ∧ x = 62.5 ∧ y = 37.5 := by
  sorry

end candy_mixture_solution_l238_238107


namespace maximum_visibility_sum_l238_238238

theorem maximum_visibility_sum (X Y : ℕ) (h : X + 2 * Y = 30) :
  X * Y ≤ 112 :=
by
  sorry

end maximum_visibility_sum_l238_238238


namespace part1_solution_set_part2_range_a_l238_238904

def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

theorem part1_solution_set (x : ℝ) (h : f x 2 ≥ 4) : 
  x ≤ 3/2 ∨ x ≥ 11/2 :=
sorry

theorem part2_range_a (a : ℝ) (h : ∃ x, f x a ≥ 4) : 
  a ≤ -1 ∨ a ≥ 3 :=
sorry

end part1_solution_set_part2_range_a_l238_238904


namespace fraction_to_decimal_l238_238030

theorem fraction_to_decimal : (7 : ℝ) / 16 = 0.4375 := 
by
  sorry

end fraction_to_decimal_l238_238030


namespace part1_solution_set_part2_range_of_a_l238_238935

-- Defining the function f(x) under given conditions
def f (x a : ℝ) : ℝ := |x - a^2| + |x - (2 * a + 1)|

-- Part 1: Determine the solution set for the inequality when a = 2
theorem part1_solution_set (x : ℝ) : (f x 2) ≥ 4 ↔ x ≤ 3/2 ∨ x ≥ 11/2 := by
  sorry

-- Part 2: Determine the range of values for a given the inequality
theorem part2_range_of_a (a : ℝ) : (∀ x : ℝ, f x a ≥ 4) ↔ a ≤ -1 ∨ a ≥ 3 := by
  sorry

end part1_solution_set_part2_range_of_a_l238_238935


namespace fraction_to_decimal_l238_238015

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l238_238015


namespace tangent_line_at_M_find_f_one_plus_f_prime_one_l238_238677

noncomputable def f (x : ℝ) : ℝ := sorry

theorem tangent_line_at_M :
  tangent_line f 1 = λ x, 3 * x - 2 := sorry

theorem find_f_one_plus_f_prime_one :
  f 1 + (deriv f 1) = 4 := 
by
  have h_tangent : tangent_line f 1 = λ x, 3 * x - 2 := sorry,
  have f_prime_1 : deriv f 1 = 3 := sorry,
  have f_1 : f 1 = 1 := sorry,
  show 1 + 3 = 4,
  linarith

end tangent_line_at_M_find_f_one_plus_f_prime_one_l238_238677


namespace fraction_to_decimal_l238_238001

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l238_238001


namespace greatest_sum_consecutive_integers_product_less_than_500_l238_238701

theorem greatest_sum_consecutive_integers_product_less_than_500 :
  ∃ n : ℕ, n * (n + 1) < 500 ∧ (n + (n + 1) = 43) := sorry

end greatest_sum_consecutive_integers_product_less_than_500_l238_238701


namespace number_of_hard_drives_sold_l238_238288

theorem number_of_hard_drives_sold 
    (H : ℕ)
    (price_per_graphics_card : ℕ := 600)
    (price_per_hard_drive : ℕ := 80)
    (price_per_cpu : ℕ := 200)
    (price_per_ram_pair : ℕ := 60)
    (graphics_cards_sold : ℕ := 10)
    (cpus_sold : ℕ := 8)
    (ram_pairs_sold : ℕ := 4)
    (total_earnings : ℕ := 8960)
    (earnings_from_graphics_cards : graphics_cards_sold * price_per_graphics_card = 6000)
    (earnings_from_cpus : cpus_sold * price_per_cpu = 1600)
    (earnings_from_ram : ram_pairs_sold * price_per_ram_pair = 240)
    (earnings_from_hard_drives : H * price_per_hard_drive = 80 * H) :
  graphics_cards_sold * price_per_graphics_card +
  cpus_sold * price_per_cpu +
  ram_pairs_sold * price_per_ram_pair +
  H * price_per_hard_drive = total_earnings → H = 14 :=
by
  intros h
  sorry

end number_of_hard_drives_sold_l238_238288


namespace sequence_term_20_l238_238369

theorem sequence_term_20 :
  ∀ (a : ℕ → ℕ), (a 1 = 1) → (∀ n, a (n+1) = a n + 2) → (a 20 = 39) := by
  intros a h1 h2
  sorry

end sequence_term_20_l238_238369


namespace part1_solution_set_part2_values_of_a_l238_238866

-- Parameters and definitions for the function and conditions
def f (x a : ℝ) := |x - a^2| + |x - (2*a + 1)|

-- Part (1) When a = 2, the inequality's solution set
theorem part1_solution_set (x : ℝ) : f x 2 ≥ 4 ↔ (x ≤ 3/2 ∨ x ≥ 11/2) := 
sorry

-- Part (2) Range of values for a given f(x, a) ≥ 4
theorem part2_values_of_a (a : ℝ) : 
∀ x, f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end part1_solution_set_part2_values_of_a_l238_238866


namespace cubing_identity_l238_238161

theorem cubing_identity (x : ℝ) (hx : x + 1/x = -3) : x^3 + 1/x^3 = -18 := by
  sorry

end cubing_identity_l238_238161


namespace cos_300_eq_cos_300_l238_238826

open Real

theorem cos_300_eq (angle_1 angle_2 : ℝ) (h1 : angle_1 = 300 * π / 180) (h2 : angle_2 = 60 * π / 180) :
  cos angle_1 = cos angle_2 :=
by
  rw [←h1, ←h2, cos_sub, cos_pi_div_six]
  norm_num
  sorry

theorem cos_300 : cos (300 * π / 180) = 1 / 2 :=
by
  apply cos_300_eq (300 * π / 180) (60 * π / 180)
  norm_num
  norm_num
  sorry

end cos_300_eq_cos_300_l238_238826


namespace trapezoids_not_necessarily_congruent_l238_238683

-- Define trapezoid structure
structure Trapezoid (α : Type) [LinearOrderedField α] :=
(base1 base2 side1 side2 diag1 diag2 : α) -- sides and diagonals
(angle1 angle2 angle3 angle4 : α)        -- internal angles

-- Conditions about given trapezoids
variables {α : Type} [LinearOrderedField α]
variables (T1 T2 : Trapezoid α)

-- The condition that corresponding angles of the trapezoids are equal
def equal_angles := 
  T1.angle1 = T2.angle1 ∧ T1.angle2 = T2.angle2 ∧ 
  T1.angle3 = T2.angle3 ∧ T1.angle4 = T2.angle4

-- The condition that diagonals of the trapezoids are equal
def equal_diagonals := 
  T1.diag1 = T2.diag1 ∧ T1.diag2 = T2.diag2

-- The statement to prove
theorem trapezoids_not_necessarily_congruent :
  equal_angles T1 T2 ∧ equal_diagonals T1 T2 → ¬ (T1 = T2) := by
  sorry

end trapezoids_not_necessarily_congruent_l238_238683


namespace part1_solution_set_part2_range_of_a_l238_238929

noncomputable def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1_solution_set (a : ℝ) (a_eq_2 : a = 2) : 
  {x : ℝ | f x a ≥ 4} = {x | x ≤ 3 / 2} ∪ {x | x ≥ 11 / 2} :=
by
  sorry

theorem part2_range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 4) → (a ∈ set.Iic (-1) ∪ set.Ici 3) :=
by
  sorry

end part1_solution_set_part2_range_of_a_l238_238929


namespace total_pies_bigger_event_l238_238653

def pies_last_week := 16.5
def apple_pies_last_week := 14.25
def cherry_pies_last_week := 12.75

def pecan_multiplier := 4.3
def apple_multiplier := 3.5
def cherry_multiplier := 5.7

theorem total_pies_bigger_event :
  (pies_last_week * pecan_multiplier) + 
  (apple_pies_last_week * apple_multiplier) + 
  (cherry_pies_last_week * cherry_multiplier) = 193.5 :=
by
  sorry

end total_pies_bigger_event_l238_238653


namespace percent_defective_units_l238_238962

-- Definition of the given problem conditions
variable (D : ℝ) -- D represents the percentage of defective units

-- The main statement we want to prove
theorem percent_defective_units (h1 : 0.04 * D = 0.36) : D = 9 := by
  sorry

end percent_defective_units_l238_238962


namespace find_minimum_n_l238_238146

variable {a_1 d : ℝ}
variable {S : ℕ → ℝ}

def is_arithmetic_sequence (a_1 d : ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n / 2 : ℝ) * (2 * a_1 + (n - 1) * d)

def condition1 (a_1 : ℝ) : Prop := a_1 < 0

def condition2 (S : ℕ → ℝ) : Prop := S 7 = S 13

theorem find_minimum_n (a_1 d : ℝ) (S : ℕ → ℝ)
  (h_arith_seq : is_arithmetic_sequence a_1 d S)
  (h_a1_neg : condition1 a_1)
  (h_s7_eq_s13 : condition2 S) :
  ∃ n : ℕ, n = 10 ∧ ∀ m : ℕ, S n ≤ S m := 
sorry

end find_minimum_n_l238_238146


namespace proof_problems_l238_238592

def otimes (a b : ℝ) : ℝ :=
  a * (1 - b)

theorem proof_problems :
  (otimes 2 (-2) = 6) ∧
  ¬ (∀ (a b : ℝ), otimes a b = otimes b a) ∧
  (∀ (a b : ℝ), a + b = 0 → otimes a a + otimes b b = 2 * a * b) ∧
  ¬ (∀ (a b : ℝ), otimes a b = 0 → a = 0) :=
by
  sorry
 
end proof_problems_l238_238592


namespace part1_part2_l238_238887

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1 (x : ℝ) : 
  ∀ x, f x 2 ≥ 4 ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
by sorry

theorem part2 (x a : ℝ) :
  f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
by sorry

end part1_part2_l238_238887


namespace total_distance_of_drive_l238_238449

theorem total_distance_of_drive :
  let christina_speed := 30
  let christina_time_minutes := 180
  let christina_time_hours := christina_time_minutes / 60
  let friend_speed := 40
  let friend_time := 3
  let distance_christina := christina_speed * christina_time_hours
  let distance_friend := friend_speed * friend_time
  let total_distance := distance_christina + distance_friend
  total_distance = 210 :=
by
  sorry

end total_distance_of_drive_l238_238449


namespace fraction_to_decimal_l238_238050

theorem fraction_to_decimal :
  (7 : ℝ) / 16 = 0.4375 := 
by sorry

end fraction_to_decimal_l238_238050


namespace prop3_prop4_l238_238223

-- Definitions to represent planes and lines
variable (Plane Line : Type)

-- Predicate representing parallel planes or lines
variable (parallel : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Predicate representing perpendicular planes or lines
variable (perpendicular : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)

-- Distinct planes and a line
variables (α β γ : Plane) (l : Line)

-- Proposition 3: If l ⊥ α and l ∥ β, then α ⊥ β
theorem prop3 : perpendicular_line_plane l α ∧ parallel_line_plane l β → perpendicular α β :=
sorry

-- Proposition 4: If α ∥ β and α ⊥ γ, then β ⊥ γ
theorem prop4 : parallel α β ∧ perpendicular α γ → perpendicular β γ :=
sorry

end prop3_prop4_l238_238223


namespace cos_alpha_sub_beta_cos_alpha_l238_238473

section

variables (α β : ℝ)
variables (cos_α : ℝ) (sin_α : ℝ) (cos_β : ℝ) (sin_β : ℝ)

-- The given conditions as premises
variable (h1: cos_α = Real.cos α)
variable (h2: sin_α = Real.sin α)
variable (h3: cos_β = Real.cos β)
variable (h4: sin_β = Real.sin β)
variable (h5: 0 < α ∧ α < π / 2)
variable (h6: -π / 2 < β ∧ β < 0)
variable (h7: (cos_α - cos_β)^2 + (sin_α - sin_β)^2 = 4 / 5)

-- Part I: Prove that cos(α - β) = 3/5
theorem cos_alpha_sub_beta : Real.cos (α - β) = 3 / 5 :=
by
  sorry

-- Additional condition for Part II
variable (h8: cos_β = 12 / 13)

-- Part II: Prove that cos α = 56 / 65
theorem cos_alpha : Real.cos α = 56 / 65 :=
by
  sorry

end

end cos_alpha_sub_beta_cos_alpha_l238_238473


namespace cos_300_eq_half_l238_238828

-- Define the point on the unit circle at 300 degrees
def point_on_unit_circle_angle_300 : ℝ × ℝ :=
  let θ := 300 * (Real.pi / 180) in
  (Real.cos θ, Real.sin θ)

-- Define the cosine of the angle 300 degrees
def cos_300 : ℝ :=
  Real.cos (300 * (Real.pi / 180))

theorem cos_300_eq_half : cos_300 = 1 / 2 :=
by sorry

end cos_300_eq_half_l238_238828


namespace fraction_to_decimal_l238_238048

theorem fraction_to_decimal :
  (7 : ℝ) / 16 = 0.4375 := 
by sorry

end fraction_to_decimal_l238_238048


namespace find_a_value_l238_238995

theorem find_a_value 
  (f : ℝ → ℝ)
  (a : ℝ)
  (h : ∀ x : ℝ, f x = x^3 + a*x^2 + 3*x - 9)
  (extreme_at_minus_3 : ∀ f' : ℝ → ℝ, (∀ x, f' x = 3*x^2 + 2*a*x + 3) → f' (-3) = 0) :
  a = 5 := 
sorry

end find_a_value_l238_238995


namespace Ellen_won_17_legos_l238_238461

theorem Ellen_won_17_legos (initial_legos : ℕ) (current_legos : ℕ) (h₁ : initial_legos = 2080) (h₂ : current_legos = 2097) : 
  current_legos - initial_legos = 17 := 
  by 
    sorry

end Ellen_won_17_legos_l238_238461


namespace quadratic_inequality_solution_l238_238140

-- Definition of the given conditions and the theorem to prove
theorem quadratic_inequality_solution 
  (a b c : ℝ)
  (h1 : a < 0)
  (h2 : ∀ x, ax^2 + bx + c < 0 ↔ x < -2 ∨ x > -1/2) :
  ∀ x, ax^2 - bx + c > 0 ↔ 1/2 < x ∧ x < 2 :=
by
  sorry

end quadratic_inequality_solution_l238_238140


namespace smallest_five_digit_number_divisibility_l238_238413

-- Define the smallest 5-digit number satisfying the conditions
def isDivisibleBy (n m : ℕ) : Prop := m ∣ n

theorem smallest_five_digit_number_divisibility :
  ∃ (n : ℕ), isDivisibleBy n 15
          ∧ isDivisibleBy n (2^8)
          ∧ isDivisibleBy n 45
          ∧ isDivisibleBy n 54
          ∧ n >= 10000
          ∧ n < 100000
          ∧ n = 69120 :=
sorry

end smallest_five_digit_number_divisibility_l238_238413


namespace smallest_rational_number_l238_238772

theorem smallest_rational_number : ∀ (a b c d : ℚ), (a = -3) → (b = -1) → (c = 0) → (d = 1) → (a < b ∧ a < c ∧ a < d) :=
by
  intros a b c d h₁ h₂ h₃ h₄
  have h₅ : a = -3 := h₁
  have h₆ : b = -1 := h₂
  have h₇ : c = 0 := h₃
  have h₈ : d = 1 := h₄
  sorry

end smallest_rational_number_l238_238772


namespace abs_diff_squares_l238_238697

-- Definitions for the numbers 105 and 95
def a : ℕ := 105
def b : ℕ := 95

-- Statement to prove: The absolute value of the difference between the squares of 105 and 95 is 2000.
theorem abs_diff_squares : |a^2 - b^2| = 2000 := by
  sorry

end abs_diff_squares_l238_238697


namespace x_plus_y_plus_z_equals_4_l238_238350

theorem x_plus_y_plus_z_equals_4 (x y z : ℝ) 
  (h1 : 2 * x + 3 * y + 4 * z = 10) 
  (h2 : y + 2 * z = 2) : 
  x + y + z = 4 :=
by
  sorry

end x_plus_y_plus_z_equals_4_l238_238350


namespace chord_PQ_eqn_l238_238164

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 9
def midpoint_PQ (M : ℝ × ℝ) : Prop := M = (1, 2)
def line_PQ_eq (x y : ℝ) : Prop := x + 2 * y - 5 = 0

theorem chord_PQ_eqn : 
  (∃ P Q : ℝ × ℝ, circle_eq P.1 P.2 ∧ circle_eq Q.1 Q.2 ∧ midpoint_PQ ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) →
  ∃ x y : ℝ, line_PQ_eq x y := 
sorry

end chord_PQ_eqn_l238_238164


namespace no_prime_roots_l238_238444

noncomputable def roots_are_prime (p q : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q

theorem no_prime_roots : 
  ∀ k : ℕ, ¬ (∃ p q : ℕ, roots_are_prime p q ∧ p + q = 65 ∧ p * q = k) := 
sorry

end no_prime_roots_l238_238444


namespace reduced_price_per_kg_l238_238574

-- Assume the constants in the conditions
variables (P R : ℝ)
variables (h1 : R = P - 0.40 * P) -- R = 0.60P
variables (h2 : 2000 / P + 10 = 2000 / R) -- extra 10 kg for the same 2000 rs

-- State the target we want to prove
theorem reduced_price_per_kg : R = 80 :=
by
  -- The steps and details of the proof
  sorry

end reduced_price_per_kg_l238_238574


namespace fraction_to_decimal_l238_238037

theorem fraction_to_decimal : (7 : ℝ) / 16 = 0.4375 := 
by
  sorry

end fraction_to_decimal_l238_238037


namespace part_1_solution_set_part_2_a_range_l238_238893

-- Define the function f
def f (x a : ℝ) := |x - a^2| + |x - 2 * a + 1|

-- Part (1)
theorem part_1_solution_set (x : ℝ) : {x | x ≤ 3 / 2 ∨ x ≥ 11 / 2} = 
  {x | f x 2 ≥ 4} :=
sorry

-- Part (2)
theorem part_2_a_range (a : ℝ) : 
  {a | (a - 1)^2 ≥ 4} = {a | a ≤ -1 ∨ a ≥ 3} :=
sorry

end part_1_solution_set_part_2_a_range_l238_238893


namespace total_cost_of_shirts_is_24_l238_238375

-- Definitions based on conditions
def cost_first_shirt : ℕ := 15
def cost_difference : ℕ := 6
def cost_second_shirt : ℕ := cost_first_shirt - cost_difference

-- The proof problem statement: Calculate total cost given the conditions
theorem total_cost_of_shirts_is_24 : cost_first_shirt + cost_second_shshirt = 24 :=
by
  sorry

end total_cost_of_shirts_is_24_l238_238375


namespace time_in_future_is_4_l238_238679

def current_time := 5
def future_hours := 1007
def modulo := 12
def future_time := (current_time + future_hours) % modulo

theorem time_in_future_is_4 : future_time = 4 := by
  sorry

end time_in_future_is_4_l238_238679


namespace average_honey_per_bee_per_day_l238_238363

-- Definitions based on conditions
def num_honey_bees : ℕ := 50
def honey_bee_days : ℕ := 35
def total_honey_produced : ℕ := 75
def expected_avg_honey_per_bee_per_day : ℝ := 2.14

-- Statement of the proof problem
theorem average_honey_per_bee_per_day :
  ((total_honey_produced : ℝ) / (num_honey_bees * honey_bee_days)) = expected_avg_honey_per_bee_per_day := by
  sorry

end average_honey_per_bee_per_day_l238_238363


namespace fraction_decimal_equivalent_l238_238004

theorem fraction_decimal_equivalent : (7 / 16 : ℝ) = 0.4375 :=
by
  sorry

end fraction_decimal_equivalent_l238_238004


namespace hyperbola_asymptote_l238_238466

theorem hyperbola_asymptote (m : ℝ) : 
  (∀ x y : ℝ, (y^2 / 16 - x^2 / 9 = 1) ↔ (y = m * x ∨ y = -m * x)) → 
  (m = 4 / 3) :=
by
  sorry

end hyperbola_asymptote_l238_238466


namespace product_of_real_values_l238_238307

theorem product_of_real_values (r : ℝ) :
  (∀ x : ℝ, x ≠ 0 → (1 / (3 * x)) = (r - x) / 8 → (3 * x * x - 3 * r * x + 8 = 0)) →
  r = 4 * Real.sqrt 6 / 3 ∨ r = -(4 * Real.sqrt 6 / 3) →
  r * -r = -32 / 3 :=
by
  intro h_x
  intro h_r
  sorry

end product_of_real_values_l238_238307


namespace cost_of_limes_after_30_days_l238_238643

def lime_juice_per_mocktail : ℝ := 1  -- tablespoons per mocktail
def days : ℕ := 30  -- number of days
def lime_juice_per_lime : ℝ := 2  -- tablespoons per lime
def limes_per_dollar : ℝ := 3  -- limes per dollar

theorem cost_of_limes_after_30_days : 
  let total_lime_juice := (lime_juice_per_mocktail * days),
      number_of_limes  := (total_lime_juice / lime_juice_per_lime),
      total_cost       := (number_of_limes / limes_per_dollar)
  in total_cost = 5 :=
by
  sorry

end cost_of_limes_after_30_days_l238_238643


namespace mod_product_prob_l238_238549

def prob_mod_product (a b : ℕ) : ℚ :=
  let quotient := a * b % 4
  if quotient = 0 then 1/2
  else if quotient = 1 then 1/8
  else if quotient = 2 then 1/4
  else if quotient = 3 then 1/8
  else 0

theorem mod_product_prob (a b : ℕ) :
  (∃ n : ℚ, n = prob_mod_product a b) :=
by
  sorry

end mod_product_prob_l238_238549


namespace cos_300_eq_half_l238_238808

theorem cos_300_eq_half : Real.cos (300 * Real.pi / 180) = 1 / 2 := 
by sorry

end cos_300_eq_half_l238_238808


namespace greatest_sum_of_consecutive_integers_product_less_500_l238_238717

theorem greatest_sum_of_consecutive_integers_product_less_500 :
  ∃ n : ℤ, n * (n + 1) < 500 ∧ (n + (n + 1)) = 43 :=
by
  sorry

end greatest_sum_of_consecutive_integers_product_less_500_l238_238717


namespace nested_radical_solution_l238_238600

theorem nested_radical_solution :
  (∃ x : ℝ, x = Real.sqrt (18 + x) ∧ x ≥ 0) ∧ ∀ x : ℝ, x = Real.sqrt (18 + x) → x ≥ 0 → x = 6 :=
by
  sorry

end nested_radical_solution_l238_238600


namespace find_total_amount_before_brokerage_l238_238992

noncomputable def total_amount_before_brokerage (realized_amount : ℝ) (brokerage_rate : ℝ) : ℝ :=
  realized_amount / (1 - brokerage_rate / 100)

theorem find_total_amount_before_brokerage :
  total_amount_before_brokerage 107.25 (1 / 4) = 107.25 * 400 / 399 := by
sorry

end find_total_amount_before_brokerage_l238_238992


namespace incorrect_option_B_l238_238294

noncomputable def Sn : ℕ → ℝ := sorry
-- S_n is the sum of the first n terms of the arithmetic sequence

axiom S5_S6 : Sn 5 < Sn 6
axiom S6_eq_S_gt_S8 : Sn 6 = Sn 7 ∧ Sn 7 > Sn 8

theorem incorrect_option_B : ¬ (Sn 9 < Sn 5) := sorry

end incorrect_option_B_l238_238294


namespace number_of_female_only_child_students_l238_238246

def students : Finset ℕ := Finset.range 21 -- Set of students with attendance numbers from 1 to 20

def female_students : Finset ℕ := {1, 3, 4, 6, 7, 10, 11, 13, 16, 17, 18, 20}

def only_child_students : Finset ℕ := {1, 4, 5, 8, 11, 14, 17, 20}

def common_students : Finset ℕ := female_students ∩ only_child_students

theorem number_of_female_only_child_students :
  common_students.card = 5 :=
by
  sorry

end number_of_female_only_child_students_l238_238246


namespace dice_sum_probability_l238_238183

/-- The probability that the sum of the numbers on three six-faced dice equals 10 is 1/8. -/
theorem dice_sum_probability : 
  (1 / 8 : ℚ) = (∑ x in (finset.univ : finset (fin 6)), 
                ∑ y in (finset.univ : finset (fin 6)), 
                ∑ z in (finset.univ : finset (fin 6)), 
                if x.1 + y.1 + z.1 = 10 then 1 else 0) / 6^3 :=
begin
  sorry
end

end dice_sum_probability_l238_238183


namespace box_weight_no_apples_l238_238565

variable (initialWeight : ℕ) (halfWeight : ℕ) (totalWeight : ℕ)
variable (boxWeight : ℕ)

-- Given conditions
axiom initialWeight_def : initialWeight = 9
axiom halfWeight_def : halfWeight = 5
axiom appleWeight_consistent : ∃ w : ℕ, ∀ n : ℕ, n * w = totalWeight

-- Question: How many kilograms does the empty box weigh?
theorem box_weight_no_apples : (initialWeight - totalWeight) = boxWeight :=
by
  -- The proof steps are omitted as indicated by the 'sorry' placeholder.
  sorry

end box_weight_no_apples_l238_238565


namespace greatest_sum_consecutive_integers_lt_500_l238_238726

theorem greatest_sum_consecutive_integers_lt_500 : 
  ∃ n : ℤ, (n * (n + 1) < 500) ∧ n + (n + 1) = 43 := 
by {
  sorry -- Proof needed
}

end greatest_sum_consecutive_integers_lt_500_l238_238726


namespace cos_300_eq_half_l238_238802

theorem cos_300_eq_half :
  let θ := 300
  let ref_θ := θ - 360 * (θ // 360) -- Reference angle in the proper range
  θ = 300 ∧ ref_θ = 60 ∧ cos 60 = 1/2 ∧ 300 > 270 → cos 300 = 1/2 :=
by
  intros
  sorry

end cos_300_eq_half_l238_238802


namespace part1_solution_set_part2_range_of_a_l238_238918

def f (x a : ℝ) := |x - a^2| + |x - (2*a + 1)|

-- Part (1)
theorem part1_solution_set (x : ℝ) :
  ∀ x, f x 2 ≥ 4 ↔ x ≤ 3 / 2 ∨ x ≥ 11 / 2 := sorry

-- Part (2)
theorem part2_range_of_a (a : ℝ) :
  (∀ x, f x a ≥ 4) ↔ (a ≤ -1 ∨ a ≥ 3) := sorry

end part1_solution_set_part2_range_of_a_l238_238918


namespace number_of_seedlings_l238_238300

theorem number_of_seedlings (packets : ℕ) (seeds_per_packet : ℕ) (h1 : packets = 60) (h2 : seeds_per_packet = 7) : packets * seeds_per_packet = 420 :=
by
  sorry

end number_of_seedlings_l238_238300


namespace part1_part2_l238_238859

-- Define the function f
def f (a x: ℝ) : ℝ := Real.exp x - a * x^2 - x

-- Define the first derivative of f
def f_prime (a x: ℝ) : ℝ := Real.exp x - 2 * a * x - 1

-- Define the conditions for part (1)
def monotonic_increasing (f_prime : ℝ → ℝ) : Prop :=
  ∀ x, f_prime x ≥ 0

-- Prove the condition for part (1)
theorem part1 (h : monotonic_increasing (f_prime (1/2))) :
  ∀ a, a = 1/2 := by
  sorry

-- Define the conditions for part (2)
def two_extreme_points (f_prime : ℝ → ℝ) (a : ℝ) : Prop :=
  ∃ x1 x2, x1 < x2 ∧ f_prime x1 = 0 ∧ f_prime x2 = 0

def f_x2_bound (f : ℝ → ℝ) (x2 : ℝ) : Prop :=
  f x2 < 1 + ((Real.sin x2) - x2) / 2

-- Prove the condition for part (2)
theorem part2 (a : ℝ) (ha : a > 1/2) :
  two_extreme_points (f_prime a) a ∧ f_x2_bound (f a) (no_proof_x2) := by
  sorry

end part1_part2_l238_238859


namespace fraction_to_decimal_l238_238081

theorem fraction_to_decimal : (7 : ℚ) / 16 = 4375 / 10000 := by
  sorry

end fraction_to_decimal_l238_238081


namespace percentage_of_y_l238_238262

theorem percentage_of_y (x y : ℝ) (h1 : x = 4 * y) (h2 : 0.80 * x = (P / 100) * y) : P = 320 :=
by
  -- Proof goes here
  sorry

end percentage_of_y_l238_238262


namespace unique_real_root_iff_a_eq_3_l238_238139

noncomputable def f (x a : ℝ) : ℝ := x^2 + a * abs x + a^2 - 9

theorem unique_real_root_iff_a_eq_3 {a : ℝ} (hu : ∃! x : ℝ, f x a = 0) : a = 3 :=
sorry

end unique_real_root_iff_a_eq_3_l238_238139


namespace ratio_of_heights_l238_238358

theorem ratio_of_heights (a b : ℝ) (area_ratio_is_9_4 : a / b = 9 / 4) :
  ∃ h₁ h₂ : ℝ, h₁ / h₂ = 3 / 2 :=
by
  sorry

end ratio_of_heights_l238_238358


namespace part1_solution_set_part2_range_of_a_l238_238907

-- Define the function f for a general a
def f (x a : ℝ) : ℝ := |x - a^2| + |x - (2 * a) + 1|

-- Part 1: When a = 2
theorem part1_solution_set (x : ℝ) (h : f x 2 ≥ 4) : x ≤ 3/2 ∨ x ≥ 11/2 :=
  sorry

-- Part 2: Range of values for a
theorem part2_range_of_a (a : ℝ) (h : ∀ x, f x a ≥ 4) : a ≤ -1 ∨ a ≥ 3 :=
  sorry

end part1_solution_set_part2_range_of_a_l238_238907


namespace solution_interval_l238_238541

noncomputable def f (x : ℝ) := 2^x + x - 2

theorem solution_interval : ∃ x ∈ Ioo 0 1, f x = 0 :=
begin
  sorry
end

end solution_interval_l238_238541


namespace c_share_of_profit_l238_238422

theorem c_share_of_profit (a b c total_profit : ℕ) 
  (h₁ : a = 5000) (h₂ : b = 8000) (h₃ : c = 9000) (h₄ : total_profit = 88000) :
  c * total_profit / (a + b + c) = 36000 :=
by
  sorry

end c_share_of_profit_l238_238422


namespace distributive_laws_none_hold_l238_238222

def star (a b : ℝ) : ℝ := a + b + a * b

theorem distributive_laws_none_hold (x y z : ℝ) :
  ¬ (x * (y + z) = (x * y) + (x * z)) ∧
  ¬ (x + (y * z) = (x + y) * (x + z)) ∧
  ¬ (x * (y * z) = (x * y) * (x * z)) :=
by
  sorry

end distributive_laws_none_hold_l238_238222


namespace part1_solution_set_part2_range_of_a_l238_238917

def f (x a : ℝ) := |x - a^2| + |x - (2*a + 1)|

-- Part (1)
theorem part1_solution_set (x : ℝ) :
  ∀ x, f x 2 ≥ 4 ↔ x ≤ 3 / 2 ∨ x ≥ 11 / 2 := sorry

-- Part (2)
theorem part2_range_of_a (a : ℝ) :
  (∀ x, f x a ≥ 4) ↔ (a ≤ -1 ∨ a ≥ 3) := sorry

end part1_solution_set_part2_range_of_a_l238_238917


namespace molecular_weight_of_compound_l238_238410

def atomic_weight (count : ℕ) (atomic_mass : ℝ) : ℝ :=
  count * atomic_mass

def molecular_weight (C_atom_count H_atom_count O_atom_count : ℕ)
  (C_atomic_weight H_atomic_weight O_atomic_weight : ℝ) : ℝ :=
  (atomic_weight C_atom_count C_atomic_weight) +
  (atomic_weight H_atom_count H_atomic_weight) +
  (atomic_weight O_atom_count O_atomic_weight)

theorem molecular_weight_of_compound :
  molecular_weight 3 6 1 12.01 1.008 16.00 = 58.078 :=
by
  sorry

end molecular_weight_of_compound_l238_238410


namespace part1_solution_set_part2_range_of_a_l238_238919

def f (x a : ℝ) := |x - a^2| + |x - (2*a + 1)|

-- Part (1)
theorem part1_solution_set (x : ℝ) :
  ∀ x, f x 2 ≥ 4 ↔ x ≤ 3 / 2 ∨ x ≥ 11 / 2 := sorry

-- Part (2)
theorem part2_range_of_a (a : ℝ) :
  (∀ x, f x a ≥ 4) ↔ (a ≤ -1 ∨ a ≥ 3) := sorry

end part1_solution_set_part2_range_of_a_l238_238919


namespace average_sitting_time_per_student_l238_238274

def total_travel_time_in_minutes : ℕ := 152
def number_of_seats : ℕ := 5
def number_of_students : ℕ := 8

theorem average_sitting_time_per_student :
  (total_travel_time_in_minutes * number_of_seats) / number_of_students = 95 := 
by
  sorry

end average_sitting_time_per_student_l238_238274


namespace part1_solution_set_part2_range_a_l238_238899

def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

theorem part1_solution_set (x : ℝ) (h : f x 2 ≥ 4) : 
  x ≤ 3/2 ∨ x ≥ 11/2 :=
sorry

theorem part2_range_a (a : ℝ) (h : ∃ x, f x a ≥ 4) : 
  a ≤ -1 ∨ a ≥ 3 :=
sorry

end part1_solution_set_part2_range_a_l238_238899


namespace probability_sum_is_ten_l238_238182

structure Die :=
(value : ℕ)
(h_value : 1 ≤ value ∧ value ≤ 6)

def possible_rolls : List (ℕ × ℕ × ℕ) :=
List.product 
  (List.product [1, 2, 3, 4, 5, 6] [1, 2, 3, 4, 5, 6]) [1, 2, 3, 4, 5, 6]
|>.map (λ ((a, b), c) => (a, b, c))

def favorable_outcomes : List (ℕ × ℕ × ℕ) :=
possible_rolls.filter (λ (x, y, z) => x + y + z = 10)

def probability (n favorable : ℕ) : ℚ := favourable / n

theorem probability_sum_is_ten :
  probability possible_rolls.length favourable_outcomes.length = 1 / 9 :=
by
  sorry

end probability_sum_is_ten_l238_238182


namespace evaluate_expression_l238_238601

theorem evaluate_expression : ∃ x : ℝ, (x = Real.sqrt (18 + x)) ∧ (x = (1 + Real.sqrt 73) / 2) := by
  sorry

end evaluate_expression_l238_238601


namespace number_of_shirts_is_39_l238_238270

-- Define the conditions as Lean definitions.
def washing_machine_capacity : ℕ := 8
def number_of_sweaters : ℕ := 33
def number_of_loads : ℕ := 9

-- Define the total number of pieces of clothing based on the conditions.
def total_pieces_of_clothing : ℕ :=
  number_of_loads * washing_machine_capacity

-- Define the number of shirts.
noncomputable def number_of_shirts : ℕ :=
  total_pieces_of_clothing - number_of_sweaters

-- The actual proof problem statement.
theorem number_of_shirts_is_39 :
  number_of_shirts = 39 := by
  sorry

end number_of_shirts_is_39_l238_238270


namespace ken_kept_pencils_l238_238380

def ken_total_pencils := 50
def pencils_given_to_manny := 10
def pencils_given_to_nilo := pencils_given_to_manny + 10
def pencils_given_away := pencils_given_to_manny + pencils_given_to_nilo

theorem ken_kept_pencils : ken_total_pencils - pencils_given_away = 20 := by
  sorry

end ken_kept_pencils_l238_238380


namespace chord_length_l238_238672

theorem chord_length (t : ℝ) :
  (∃ x y, x = 1 + 2 * t ∧ y = 2 + t ∧ x ^ 2 + y ^ 2 = 9) →
  ((1.8 - (-3)) ^ 2 + (2.4 - 0) ^ 2 = (12 / 5 * Real.sqrt 5) ^ 2) :=
by
  sorry

end chord_length_l238_238672


namespace combined_area_is_256_l238_238575

-- Define the conditions
def side_length : ℝ := 16
def area_square : ℝ := side_length ^ 2

-- Define the property of the sides r and s
def r_s_property (r s : ℝ) : Prop :=
  (r + s)^2 + (r - s)^2 = side_length^2

-- The combined area of the four triangles
def combined_area_of_triangles (r s : ℝ) : ℝ :=
  2 * (r ^ 2 + s ^ 2)

-- Prove the final statement
theorem combined_area_is_256 (r s : ℝ) (h : r_s_property r s) :
  combined_area_of_triangles r s = 256 := by
  sorry

end combined_area_is_256_l238_238575


namespace part1_solution_set_part2_range_a_l238_238898

def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

theorem part1_solution_set (x : ℝ) (h : f x 2 ≥ 4) : 
  x ≤ 3/2 ∨ x ≥ 11/2 :=
sorry

theorem part2_range_a (a : ℝ) (h : ∃ x, f x a ≥ 4) : 
  a ≤ -1 ∨ a ≥ 3 :=
sorry

end part1_solution_set_part2_range_a_l238_238898


namespace james_sheets_of_paper_l238_238508

noncomputable def sheets_of_paper (books : ℕ) (pages_per_book : ℕ) (pages_per_side : ℕ) (sides_per_sheet : ℕ) : ℕ :=
  (books * pages_per_book) / (pages_per_side * sides_per_sheet)

theorem james_sheets_of_paper :
  sheets_of_paper 2 600 4 2 = 150 :=
by
  sorry

end james_sheets_of_paper_l238_238508


namespace probability_sum_is_10_l238_238190

theorem probability_sum_is_10 (d1 d2 d3 : ℕ) : 
  (1 ≤ d1 ∧ d1 ≤ 6) ∧ (1 ≤ d2 ∧ d2 ≤ 6) ∧ (1 ≤ d3 ∧ d3 ≤ 6) ∧ (d1 + d2 + d3 = 10) → 
  (((d1 = 1 ∧ d2 = 3 ∧ d3 = 6) ∨ 
    (d1 = 1 ∧ d2 = 4 ∧ d3 = 5) ∨ 
    (d1 = 2 ∧ d2 = 3 ∧ d3 = 5) ∨ 
    (d1 = 2 ∧ d2 = 4 ∧ d3 = 4) ∨ 
    (d1 = 3 ∧ d2 = 3 ∧ d3 = 4) ∨ 
    (d1 = 4 ∧ d2 = 4 ∧ d3 = 2)) 
    ↔ nat.cast (1 / 9)) :=
begin
  sorry,
end

end probability_sum_is_10_l238_238190


namespace smallest_k_l238_238414

theorem smallest_k (k: ℕ) : k > 1 ∧ (k % 23 = 1) ∧ (k % 7 = 1) ∧ (k % 3 = 1) → k = 484 :=
sorry

end smallest_k_l238_238414


namespace curve_M_cartesian_eq_proof_curve_N_cartesian_eq_proof_min_distance_between_AB_l238_238942

noncomputable def curve_M_parametric_eq (α : ℝ) : ℝ × ℝ :=
  (2 * Real.cos α, 2 + 2 * Real.sin α)

noncomputable def curve_M_cartesian_eq (x y : ℝ) : Prop :=
  x^2 + (y - 2)^2 = 4

noncomputable def curve_N_polar_eq (ρ θ : ℝ) : Prop :=
  ρ * Real.sin (θ + Real.pi / 3) = 8

noncomputable def curve_N_cartesian_eq (x y : ℝ) : Prop :=
  sqrt 3 * x + y - 16 = 0

theorem curve_M_cartesian_eq_proof (α : ℝ) :
  ∃ (x y : ℝ), curve_M_parametric_eq α = (x, y) ∧ curve_M_cartesian_eq x y :=
sorry

theorem curve_N_cartesian_eq_proof (ρ θ : ℝ) :
  ∃ (x y : ℝ), curve_N_polar_eq ρ θ ∧ curve_N_cartesian_eq x y :=
sorry

noncomputable def distance_from_center_to_line (x₀ y₀ : ℝ) (a b c : ℝ) : ℝ :=
  Real.abs (a * x₀ + b * y₀ + c) / Real.sqrt (a^2 + b^2)

noncomputable def radius_M : ℝ := 2

noncomputable def center_M : ℝ × ℝ := (0, 2)

theorem min_distance_between_AB : 
  let (x₀, y₀) := center_M in 
  let d := distance_from_center_to_line x₀ y₀ (sqrt 3) 1 (-16) in
  d - radius_M = 5 :=
sorry

end curve_M_cartesian_eq_proof_curve_N_cartesian_eq_proof_min_distance_between_AB_l238_238942


namespace not_necessarily_divisor_sixty_four_l238_238395

theorem not_necessarily_divisor_sixty_four (k : ℤ) (h : (k * (k + 1) * (k + 2)) % 8 = 0) :
  ¬ ((k * (k + 1) * (k + 2)) % 64 = 0) := 
sorry

end not_necessarily_divisor_sixty_four_l238_238395


namespace fraction_to_decimal_l238_238035

theorem fraction_to_decimal : (7 : ℝ) / 16 = 0.4375 := 
by
  sorry

end fraction_to_decimal_l238_238035


namespace cos_300_is_half_l238_238814

noncomputable def cos_300_eq_half : Prop :=
  real.cos (2 * real.pi * 5 / 6) = 1 / 2

theorem cos_300_is_half : cos_300_eq_half :=
sorry

end cos_300_is_half_l238_238814


namespace probability_of_yellow_face_l238_238243

theorem probability_of_yellow_face :
  let total_faces : ℕ := 10
  let yellow_faces : ℕ := 4
  (yellow_faces : ℚ) / (total_faces : ℚ) = 2 / 5 :=
by
  sorry

end probability_of_yellow_face_l238_238243


namespace socks_count_l238_238582

theorem socks_count
  (black_socks : ℕ) 
  (white_socks : ℕ)
  (H1 : white_socks = 4 * black_socks)
  (H2 : black_socks = 6)
  (H3 : white_socks / 2 = white_socks - (white_socks / 2)) :
  (white_socks / 2) - black_socks = 6 := by
  sorry

end socks_count_l238_238582


namespace youngest_brother_is_3_l238_238522

def rick_age : ℕ := 15
def oldest_brother_age := 2 * rick_age
def middle_brother_age := oldest_brother_age / 3
def smallest_brother_age := middle_brother_age / 2
def youngest_brother_age := smallest_brother_age - 2

theorem youngest_brother_is_3 : youngest_brother_age = 3 := 
by simp [rick_age, oldest_brother_age, middle_brother_age, smallest_brother_age, youngest_brother_age]; sorry

end youngest_brother_is_3_l238_238522


namespace abs_diff_squares_105_95_l238_238694

theorem abs_diff_squares_105_95 : abs ((105:ℤ)^2 - (95:ℤ)^2) = 2000 := by
  sorry

end abs_diff_squares_105_95_l238_238694


namespace find_k_value_l238_238623

theorem find_k_value (x y k : ℝ) 
  (h1 : 2 * x + y = 1) 
  (h2 : x + 2 * y = k - 2) 
  (h3 : x - y = 2) : 
  k = 1 := 
by
  sorry

end find_k_value_l238_238623


namespace part_1_solution_set_part_2_a_range_l238_238895

-- Define the function f
def f (x a : ℝ) := |x - a^2| + |x - 2 * a + 1|

-- Part (1)
theorem part_1_solution_set (x : ℝ) : {x | x ≤ 3 / 2 ∨ x ≥ 11 / 2} = 
  {x | f x 2 ≥ 4} :=
sorry

-- Part (2)
theorem part_2_a_range (a : ℝ) : 
  {a | (a - 1)^2 ≥ 4} = {a | a ≤ -1 ∨ a ≥ 3} :=
sorry

end part_1_solution_set_part_2_a_range_l238_238895


namespace pyramid_volume_l238_238766

theorem pyramid_volume (S A : ℝ)
  (h_surface : 3 * S = 432)
  (h_half_triangular : A = 0.5 * S) :
  (1 / 3) * S * (12 * Real.sqrt 3) = 288 * Real.sqrt 3 :=
by
  sorry

end pyramid_volume_l238_238766


namespace number_of_faces_l238_238546

-- Define the given conditions
def ways_to_paint_faces (n : ℕ) := Nat.factorial n

-- State the problem: Given ways_to_paint_faces n = 720, prove n = 6
theorem number_of_faces (n : ℕ) (h : ways_to_paint_faces n = 720) : n = 6 :=
sorry

end number_of_faces_l238_238546


namespace part1_solution_set_part2_range_of_a_l238_238938

-- Defining the function f(x) under given conditions
def f (x a : ℝ) : ℝ := |x - a^2| + |x - (2 * a + 1)|

-- Part 1: Determine the solution set for the inequality when a = 2
theorem part1_solution_set (x : ℝ) : (f x 2) ≥ 4 ↔ x ≤ 3/2 ∨ x ≥ 11/2 := by
  sorry

-- Part 2: Determine the range of values for a given the inequality
theorem part2_range_of_a (a : ℝ) : (∀ x : ℝ, f x a ≥ 4) ↔ a ≤ -1 ∨ a ≥ 3 := by
  sorry

end part1_solution_set_part2_range_of_a_l238_238938


namespace cos_300_eq_half_l238_238797

theorem cos_300_eq_half : Real.cos (300 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end cos_300_eq_half_l238_238797


namespace sine_ratio_triangle_area_l238_238500

variables {A B C : ℝ}
variables {a b c : ℝ}
variables {area : ℝ}

-- Main statement for part 1
theorem sine_ratio 
  (h1 : a * c * Real.cos B - b * c * Real.cos A = 3 * b^2) :
  (Real.sin A / Real.sin B) = Real.sqrt 7 := 
sorry

-- Main statement for part 2
theorem triangle_area 
  (h1 : a * c * Real.cos B - b * c * Real.cos A = 3 * b^2)
  (h2 : c = Real.sqrt 11)
  (h3 : Real.sin C = (2 * Real.sqrt 2)/3)
  (h4 : C < π / 2) :
  area = Real.sqrt 14 :=
sorry

end sine_ratio_triangle_area_l238_238500


namespace necessary_but_not_sufficient_condition_l238_238331
open Locale

variables {l m : Line} {α β : Plane}

def perp (l : Line) (p : Plane) : Prop := sorry
def subset (l : Line) (p : Plane) : Prop := sorry
def parallel (p₁ p₂ : Plane) : Prop := sorry

theorem necessary_but_not_sufficient_condition (h1 : perp l α) (h2 : subset m β) (h3 : perp l m) :
  ∃ (α : Plane) (β : Plane), parallel α β ∧ (perp l α → perp l β) ∧ (parallel α β → perp l β)  :=
sorry

end necessary_but_not_sufficient_condition_l238_238331


namespace max_n_for_factored_poly_l238_238320

theorem max_n_for_factored_poly : 
  ∃ (n : ℤ), (∀ (A B : ℤ), 2 * B + A = n → A * B = 50) ∧ 
            (∀ (m : ℤ), (∀ (A B : ℤ), 2 * B + A = m → A * B = 50) → m ≤ 101) ∧ 
            n = 101 :=
by
  sorry

end max_n_for_factored_poly_l238_238320


namespace egg_condition_difference_l238_238232

theorem egg_condition_difference :
  let initial_eggs := 2 * 12
  let broken_eggs := 3
  let cracked_eggs := 2 * broken_eggs
  let not_perfect_condition := broken_eggs + cracked_eggs
  let perfect_condition := initial_eggs - not_perfect_condition
  in perfect_condition - cracked_eggs = 9 :=
by
  let initial_eggs := 2 * 12
  let broken_eggs := 3
  let cracked_eggs := 2 * broken_eggs
  let not_perfect_condition := broken_eggs + cracked_eggs
  let perfect_condition := initial_eggs - not_perfect_condition
  show perfect_condition - cracked_eggs = 9, by sorry

end egg_condition_difference_l238_238232


namespace cos_300_is_half_l238_238815

noncomputable def cos_300_eq_half : Prop :=
  real.cos (2 * real.pi * 5 / 6) = 1 / 2

theorem cos_300_is_half : cos_300_eq_half :=
sorry

end cos_300_is_half_l238_238815


namespace greatest_sum_of_consecutive_integers_product_less_500_l238_238716

theorem greatest_sum_of_consecutive_integers_product_less_500 :
  ∃ n : ℤ, n * (n + 1) < 500 ∧ (n + (n + 1)) = 43 :=
by
  sorry

end greatest_sum_of_consecutive_integers_product_less_500_l238_238716


namespace integral_of_x_squared_l238_238497

-- Define the conditions
noncomputable def constant_term : ℝ := 3

-- Define the main theorem we want to prove
theorem integral_of_x_squared : ∫ (x : ℝ) in (1 : ℝ)..constant_term, x^2 = 26 / 3 := 
by 
  sorry

end integral_of_x_squared_l238_238497


namespace problem1_problem2_problem2_equality_l238_238221

variable {a b c d : ℝ}

-- Problem 1
theorem problem1 (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d) (h4 : (a - b) * (b - c) * (c - d) * (d - a) = -3) (h5 : a + b + c + d = 6) : d < 0.36 :=
sorry

-- Problem 2
theorem problem2 (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d) (h4 : (a - b) * (b - c) * (c - d) * (d - a) = -3) (h5 : a^2 + b^2 + c^2 + d^2 = 14) : (a + c) * (b + d) ≤ 8 :=
sorry

theorem problem2_equality (h1 : a = 3) (h2 : b = 2) (h3 : c = 1) (h4 : d = 0) : (a + c) * (b + d) = 8 :=
sorry

end problem1_problem2_problem2_equality_l238_238221


namespace pos_real_unique_solution_l238_238314

theorem pos_real_unique_solution (x : ℝ) (hx_pos : 0 < x) (h : (x - 3) / 8 = 5 / (x - 8)) : x = 16 :=
sorry

end pos_real_unique_solution_l238_238314


namespace spadesuit_value_l238_238456

def spadesuit (a b : ℤ) : ℤ :=
  |a^2 - b^2|

theorem spadesuit_value :
  spadesuit 3 (spadesuit 5 2) = 432 :=
by
  sorry

end spadesuit_value_l238_238456


namespace cube_union_volume_is_correct_cube_union_surface_area_is_correct_l238_238550

noncomputable def cubeUnionVolume : ℝ :=
  let cubeVolume := 1
  let intersectionVolume := 1 / 4
  cubeVolume * 2 - intersectionVolume

theorem cube_union_volume_is_correct :
  cubeUnionVolume = 5 / 4 := sorry

noncomputable def cubeUnionSurfaceArea : ℝ :=
  2 * (6 * (1 / 4) + 6 * (1 / 4 / 4))

theorem cube_union_surface_area_is_correct :
  cubeUnionSurfaceArea = 15 / 2 := sorry

end cube_union_volume_is_correct_cube_union_surface_area_is_correct_l238_238550


namespace find_a_if_odd_f_monotonically_increasing_on_pos_l238_238844

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x^2 - 1) / (x + a)

-- Part 1: Proving that a = 0
theorem find_a_if_odd : (∀ x : ℝ, f x a = -f (-x) a) → a = 0 := by sorry

-- Part 2: Proving that f(x) is monotonically increasing on (0, +∞) given a = 0
theorem f_monotonically_increasing_on_pos : (∀ x : ℝ, x > 0 → 
  ∃ y : ℝ, y > 0 ∧ f x 0 < f y 0) := by sorry

end find_a_if_odd_f_monotonically_increasing_on_pos_l238_238844


namespace cos_300_eq_half_l238_238784

theorem cos_300_eq_half :
  ∀ (deg : ℝ), deg = 300 → cos (deg * π / 180) = 1 / 2 :=
by
  intros deg h_deg
  rw [h_deg]
  have h1 : 300 = 360 - 60, by norm_num
  rw [h1]
  -- Convert angles to radians for usage with cos function
  have h2 : 360 - 60 = (2 * π - (π / 3)) * 180 / π, by norm_num
  rw [h2]
  have h3 : cos ((2 * π - π / 3) * π / 180) = cos (π / 3), by rw [real.cos_sub_pi_div_two, real.cos_pi_div_three]
  rw [h3]
  norm_num

end cos_300_eq_half_l238_238784


namespace seventh_graders_problems_l238_238242

theorem seventh_graders_problems (n : ℕ) (S : ℕ) (a : ℕ) (h1 : a > (S - a) / 5) (h2 : a < (S - a) / 3) : n = 5 :=
  sorry

end seventh_graders_problems_l238_238242


namespace greatest_sum_of_consecutive_integers_product_lt_500_l238_238733

theorem greatest_sum_of_consecutive_integers_product_lt_500 : 
  ∃ (n : ℕ), n * (n + 1) < 500 ∧ ∀ (m : ℕ), m * (m + 1) < 500 → m ≤ n → n + (n + 1) = 43 := 
by
  use 21
  split
  {
    norm_num
    linarith
  }
  {
    intros m h_hint h_ineq
    have : m ≤ 21, sorry
    linarith
  }
  sorry

end greatest_sum_of_consecutive_integers_product_lt_500_l238_238733


namespace dice_sum_probability_l238_238186

/-- The probability that the sum of the numbers on three six-faced dice equals 10 is 1/8. -/
theorem dice_sum_probability : 
  (1 / 8 : ℚ) = (∑ x in (finset.univ : finset (fin 6)), 
                ∑ y in (finset.univ : finset (fin 6)), 
                ∑ z in (finset.univ : finset (fin 6)), 
                if x.1 + y.1 + z.1 = 10 then 1 else 0) / 6^3 :=
begin
  sorry
end

end dice_sum_probability_l238_238186


namespace probability_sector_F_l238_238096

theorem probability_sector_F (prob_D prob_E prob_F : ℚ)
    (hD : prob_D = 1/4) 
    (hE : prob_E = 1/3) 
    (hSum : prob_D + prob_E + prob_F = 1) :
    prob_F = 5/12 := by
  sorry

end probability_sector_F_l238_238096


namespace inequality_proof_l238_238646

variable {a b c : ℝ}

theorem inequality_proof (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_mul : a * b * c = 1) : 
  (a - 1) / c + (c - 1) / b + (b - 1) / a ≥ 0 :=
sorry

end inequality_proof_l238_238646


namespace cos_300_eq_one_half_l238_238794

theorem cos_300_eq_one_half :
  Real.cos (300 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end cos_300_eq_one_half_l238_238794


namespace Mickey_horses_per_week_l238_238123

-- Definitions based on the conditions
def days_in_week : Nat := 7
def Minnie_mounts_per_day : Nat := days_in_week + 3 
def Mickey_mounts_per_day : Nat := 2 * Minnie_mounts_per_day - 6
def Mickey_mounts_per_week : Nat := Mickey_mounts_per_day * days_in_week

-- Theorem statement
theorem Mickey_horses_per_week : Mickey_mounts_per_week = 98 :=
by
  sorry

end Mickey_horses_per_week_l238_238123


namespace high_speed_train_equation_l238_238780

theorem high_speed_train_equation (x : ℝ) (h1 : x > 0) : 
  700 / x - 700 / (2.8 * x) = 3.6 :=
by
  sorry

end high_speed_train_equation_l238_238780


namespace unique_a_for_system_solution_l238_238317

-- Define the variables
variables (a b x y : ℝ)

-- Define the system of equations
def system_has_solution (a b : ℝ) : Prop :=
  ∃ x y : ℝ, 2^(b * x) + (a + 1) * b * y^2 = a^2 ∧ (a-1) * x^3 + y^3 = 1

-- Main theorem statement
theorem unique_a_for_system_solution :
  a = -1 ↔ ∀ b : ℝ, system_has_solution a b :=
sorry

end unique_a_for_system_solution_l238_238317


namespace sequence_expression_l238_238338

-- Given conditions
variable (S : ℕ → ℝ)
variable (a : ℕ → ℝ)
variable (h1 : ∀ n, S n = (1/4) * (a n + 1)^2)

-- Theorem statement
theorem sequence_expression (n : ℕ) : a n = 2 * n - 1 :=
sorry

end sequence_expression_l238_238338


namespace water_usage_difference_l238_238991

theorem water_usage_difference (C X : ℕ)
    (h1 : C = 111000)
    (h2 : C = 3 * X)
    (days : ℕ) (h3 : days = 365) :
    (C * days - X * days) = 26910000 := by
  sorry

end water_usage_difference_l238_238991


namespace at_most_n_pairs_with_distance_d_l238_238635

theorem at_most_n_pairs_with_distance_d
  (n : ℕ) (hn : n ≥ 3)
  (points : Fin n → ℝ × ℝ)
  (d : ℝ)
  (hd : ∀ i j, i ≠ j → dist (points i) (points j) ≤ d)
  (dmax : ∃ i j, i ≠ j ∧ dist (points i) (points j) = d) :
  ∃ (pairs : Finset (Fin n × Fin n)), ∀ p ∈ pairs, dist (points p.1) (points p.2) = d ∧ pairs.card ≤ n := 
sorry

end at_most_n_pairs_with_distance_d_l238_238635


namespace proof_problem_l238_238976

open Set

variable {U : Set ℕ} {A : Set ℕ} {B : Set ℕ}

def problem_statement (U A B : Set ℕ) : Prop :=
  ((U \ A) ∪ B) = {2, 3}

theorem proof_problem :
  problem_statement {0, 1, 2, 3} {0, 1, 2} {2, 3} :=
by
  unfold problem_statement
  simp
  sorry

end proof_problem_l238_238976


namespace angle_BAC_eq_angle_DAE_l238_238984

-- Define types and points A, B, C, D, E
variables (A B C D E : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E]
variables (P Q R S T : Point)

-- Define angles
variable {α β γ δ θ ω : Angle}

-- Establish the conditions
axiom angle_ABC_eq_angle_ADE : α = θ
axiom angle_AEC_eq_angle_ADB : β = ω

-- State the theorem
theorem angle_BAC_eq_angle_DAE
  (h1 : α = θ) -- Given \(\angle ABC = \angle ADE\)
  (h2 : β = ω) -- Given \(\angle AEC = \angle ADB\)
  : γ = δ := sorry

end angle_BAC_eq_angle_DAE_l238_238984


namespace greatest_sum_consecutive_integers_lt_500_l238_238725

theorem greatest_sum_consecutive_integers_lt_500 : 
  ∃ n : ℤ, (n * (n + 1) < 500) ∧ n + (n + 1) = 43 := 
by {
  sorry -- Proof needed
}

end greatest_sum_consecutive_integers_lt_500_l238_238725


namespace solve_for_x_opposites_l238_238349

theorem solve_for_x_opposites (x : ℝ) (h : -2 * x = -(3 * x - 1)) : x = 1 :=
by {
  sorry
}

end solve_for_x_opposites_l238_238349


namespace batsman_average_increase_l238_238093

theorem batsman_average_increase
  (A : ℤ)
  (h1 : (16 * A + 85) / 17 = 37) :
  37 - A = 3 :=
by
  sorry

end batsman_average_increase_l238_238093


namespace part1_solution_set_part2_range_of_a_l238_238924

noncomputable def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1_solution_set (a : ℝ) (a_eq_2 : a = 2) : 
  {x : ℝ | f x a ≥ 4} = {x | x ≤ 3 / 2} ∪ {x | x ≥ 11 / 2} :=
by
  sorry

theorem part2_range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 4) → (a ∈ set.Iic (-1) ∪ set.Ici 3) :=
by
  sorry

end part1_solution_set_part2_range_of_a_l238_238924


namespace greatest_sum_of_consecutive_integers_l238_238705

def consecutiveSumCondition (n : ℤ) : Prop :=
  n * (n + 1) < 500 

theorem greatest_sum_of_consecutive_integers : 
  ∃ n : ℤ, consecutiveSumCondition n ∧ ∀ m : ℤ, consecutiveSumCondition m → n + (n + 1) ≥ m + (m + 1) :=
sorry

end greatest_sum_of_consecutive_integers_l238_238705


namespace cubing_identity_l238_238163

theorem cubing_identity (x : ℝ) (hx : x + 1/x = -3) : x^3 + 1/x^3 = -18 := by
  sorry

end cubing_identity_l238_238163


namespace inequality_abc_squared_l238_238518

theorem inequality_abc_squared (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : 2 * (a + b + c + d) ≥ a * b * c * d) : 
  a^2 + b^2 + c^2 + d^2 ≥ a * b * c * d := 
sorry

end inequality_abc_squared_l238_238518


namespace fraction_to_decimal_l238_238053

theorem fraction_to_decimal :
  (7 : ℝ) / 16 = 0.4375 := 
by sorry

end fraction_to_decimal_l238_238053


namespace sqrt_20_19_18_17_plus_1_eq_341_l238_238832

theorem sqrt_20_19_18_17_plus_1_eq_341 :
  Real.sqrt ((20: ℝ) * 19 * 18 * 17 + 1) = 341 := by
sorry

end sqrt_20_19_18_17_plus_1_eq_341_l238_238832


namespace inequality_add_one_l238_238493

variable {α : Type*} [LinearOrderedField α]

theorem inequality_add_one {a b : α} (h : a > b) : a + 1 > b + 1 :=
sorry

end inequality_add_one_l238_238493


namespace consecutive_triples_with_product_divisible_by_1001_l238_238464

theorem consecutive_triples_with_product_divisible_by_1001 :
  ∃ (a b c : ℕ), 
    (a = 76 ∧ b = 77 ∧ c = 78) ∨ 
    (a = 77 ∧ b = 78 ∧ c = 79) ∧ 
    (a ≤ 100 ∧ b ≤ 100 ∧ c ≤ 100) ∧ 
    (b = a + 1 ∧ c = b + 1) ∧ 
    (1001 ∣ (a * b * c)) :=
by sorry

end consecutive_triples_with_product_divisible_by_1001_l238_238464


namespace tiffany_lives_l238_238269

theorem tiffany_lives (initial_lives lives_lost lives_after_next_level lives_gained : ℕ)
  (h1 : initial_lives = 43)
  (h2 : lives_lost = 14)
  (h3 : lives_after_next_level = 56)
  (h4 : lives_gained = lives_after_next_level - (initial_lives - lives_lost)) :
  lives_gained = 27 :=
by {
  sorry
}

end tiffany_lives_l238_238269


namespace sum_of_roots_cubic_equation_l238_238749

theorem sum_of_roots_cubic_equation :
  let roots := multiset.to_finset (multiset.filter (λ r, r ≠ 0) (RootSet (6 * (X ^ 3) + 7 * (X ^ 2) + (-12) * X) ℤ))
  (roots.sum : ℤ) / (roots.card : ℤ) = -117 / 100 := sorry

end sum_of_roots_cubic_equation_l238_238749


namespace weight_of_piece_l238_238660

theorem weight_of_piece (x d : ℝ) (h1 : x - d = 300) (h2 : x + d = 500) : x = 400 := 
by
  sorry

end weight_of_piece_l238_238660


namespace greatest_possible_sum_of_consecutive_integers_product_less_500_l238_238722

theorem greatest_possible_sum_of_consecutive_integers_product_less_500 :
  ∃ (n : ℤ), (n * (n + 1) < 500) ∧ (∀ (m : ℤ), (m * (m + 1) < 500) → (n + (n + 1) ≥ m + (m + 1))) :=
begin
  sorry
end

end greatest_possible_sum_of_consecutive_integers_product_less_500_l238_238722


namespace example_function_indeterminate_unbounded_l238_238520

theorem example_function_indeterminate_unbounded:
  (∀ x, ∃ f : ℝ → ℝ, (f x = (x^2 + x - 2) / (x^3 + 2 * x + 1)) ∧ 
                      (f 1 = (0 / (1^3 + 2 * 1 + 1))) ∧
                      (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 1) < δ → abs (f x) > ε)) :=
by
  sorry

end example_function_indeterminate_unbounded_l238_238520


namespace find_AC_l238_238295

noncomputable def find_AC' (BC' : ℝ) : ℝ := 
  real.cbrt 2

theorem find_AC'_correct (BC' : ℝ) (hBC' : BC' = 1) : find_AC' BC' = real.cbrt 2 := 
  by
    rw [hBC']
    sorry

end find_AC_l238_238295


namespace part1_solution_set_part2_range_a_l238_238900

def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

theorem part1_solution_set (x : ℝ) (h : f x 2 ≥ 4) : 
  x ≤ 3/2 ∨ x ≥ 11/2 :=
sorry

theorem part2_range_a (a : ℝ) (h : ∃ x, f x a ≥ 4) : 
  a ≤ -1 ∨ a ≥ 3 :=
sorry

end part1_solution_set_part2_range_a_l238_238900


namespace angles_between_plane_and_catheti_l238_238502

theorem angles_between_plane_and_catheti
  (α β : ℝ)
  (h_alpha : 0 < α ∧ α < π / 2)
  (h_beta : 0 < β ∧ β < π / 2) :
  ∃ γ θ : ℝ,
    γ = Real.arcsin (Real.sin β * Real.cos α) ∧
    θ = Real.arcsin (Real.sin β * Real.sin α) :=
by
  sorry

end angles_between_plane_and_catheti_l238_238502


namespace cos_300_eq_half_l238_238818

theorem cos_300_eq_half : Real.cos (300 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end cos_300_eq_half_l238_238818


namespace solution_set_empty_iff_l238_238263

def quadratic_no_solution (a b c : ℝ) : Prop :=
  ∀ x : ℝ, ¬ (a * x^2 + b * x + c < 0)

theorem solution_set_empty_iff (a b c : ℝ) (h : quadratic_no_solution a b c) : a > 0 ∧ (b^2 - 4 * a * c ≤ 0) :=
sorry

end solution_set_empty_iff_l238_238263


namespace direction_vector_of_l_l238_238539

open Matrix

noncomputable def projection_matrix : Matrix (Fin 3) (Fin 3) ℚ :=
  ![
    ![3 / 19, -2 / 19, -1 / 2],
    ![-2 / 19, 1 / 19, 1 / 4],
    ![-1 / 2, 1 / 4, 9 / 10]
  ]

def standard_basis_vector : Fin 3 → ℚ :=
  λ i, if i = 0 then 1 else 0

def projected_vector : Fin 3 → ℚ :=
  Matrix.mulVec projection_matrix standard_basis_vector

theorem direction_vector_of_l := 
  let direction_vector := (6, -4, -19) in
  projected_vector = direction_vector :=
  sorry

end direction_vector_of_l_l238_238539


namespace total_poles_needed_l238_238098

theorem total_poles_needed (longer_side_poles : ℕ) (shorter_side_poles : ℕ) (internal_fence_poles : ℕ) :
  longer_side_poles = 35 → 
  shorter_side_poles = 27 → 
  internal_fence_poles = (shorter_side_poles - 1) → 
  ((longer_side_poles * 2) + (shorter_side_poles * 2) - 4 + internal_fence_poles) = 146 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end total_poles_needed_l238_238098


namespace tamika_vs_carlos_l238_238245

theorem tamika_vs_carlos :
  let tamika_sums := {21, 22, 23}
  let carlos_products := {24, 28, 42}
  ∀ t ∈ tamika_sums, ∀ c ∈ carlos_products, t ≤ c →
  (∃ p : ℚ, p = (0 : ℚ) / 9) :=
by
  let tamika_sums := {21, 22, 23}
  let carlos_products := {24, 28, 42}
  intros t ht c hc htc
  use 0 / 9
  sorry

end tamika_vs_carlos_l238_238245


namespace length_HD_is_3_l238_238637

noncomputable def square_side : ℝ := 8

noncomputable def midpoint_AD : ℝ := square_side / 2

noncomputable def length_FD : ℝ := midpoint_AD

theorem length_HD_is_3 :
  ∃ (x : ℝ), 0 < x ∧ x < square_side ∧ (8 - x) ^ 2 = x ^ 2 + length_FD ^ 2 ∧ x = 3 :=
by
  sorry

end length_HD_is_3_l238_238637


namespace Mickey_horses_per_week_l238_238125

-- Definitions based on the conditions
def days_in_week : Nat := 7
def Minnie_mounts_per_day : Nat := days_in_week + 3 
def Mickey_mounts_per_day : Nat := 2 * Minnie_mounts_per_day - 6
def Mickey_mounts_per_week : Nat := Mickey_mounts_per_day * days_in_week

-- Theorem statement
theorem Mickey_horses_per_week : Mickey_mounts_per_week = 98 :=
by
  sorry

end Mickey_horses_per_week_l238_238125


namespace marbles_initial_count_l238_238517

theorem marbles_initial_count :
  let total_customers := 20
  let marbles_per_customer := 15
  let marbles_remaining := 100
  ∃ initial_marbles, initial_marbles = total_customers * marbles_per_customer + marbles_remaining :=
by
  let total_customers := 20
  let marbles_per_customer := 15
  let marbles_remaining := 100
  existsi (total_customers * marbles_per_customer + marbles_remaining)
  rfl

end marbles_initial_count_l238_238517


namespace range_of_a_for_perpendicular_tangent_line_l238_238631

theorem range_of_a_for_perpendicular_tangent_line (a : ℝ) :
  (∃ x > 0, ∃ y : ℝ, (f : ℝ → ℝ) = (λ x, a*x^3 + Real.log x) ∧ (f' : ℝ → ℝ) = (λ x, 3*a*x^2 + 1/x) ∧ (f'' : ℝ → ℝ) = (λ x, 6*a*x - 1/x^2) ∧ (∀ x, f'' x ≠ 0) ∧ (∀ x > 0, f' x → ∞)) → a < 0 := 
begin
  sorry
end

end range_of_a_for_perpendicular_tangent_line_l238_238631


namespace determine_g_l238_238224

noncomputable def g : ℝ → ℝ := sorry 

lemma g_functional_equation (x y : ℝ) : g (x * y) = g ((x^2 + y^2 + 1) / 3) + (x - y)^2 :=
sorry

lemma g_at_zero : g 0 = 1 :=
sorry

theorem determine_g (x : ℝ) : g x = 2 - 2 * x :=
sorry

end determine_g_l238_238224


namespace abs_diff_squares_l238_238696

-- Definitions for the numbers 105 and 95
def a : ℕ := 105
def b : ℕ := 95

-- Statement to prove: The absolute value of the difference between the squares of 105 and 95 is 2000.
theorem abs_diff_squares : |a^2 - b^2| = 2000 := by
  sorry

end abs_diff_squares_l238_238696


namespace fraction_to_decimal_l238_238020

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l238_238020


namespace apple_tree_production_l238_238293

def first_year_production : ℕ := 40
def second_year_production (first_year_production : ℕ) : ℕ := 2 * first_year_production + 8
def third_year_production (second_year_production : ℕ) : ℕ := second_year_production - (second_year_production / 4)
def total_production (first_year_production second_year_production third_year_production : ℕ) : ℕ :=
    first_year_production + second_year_production + third_year_production

-- Proof statement
theorem apple_tree_production : total_production 40 88 66 = 194 := by
  sorry

end apple_tree_production_l238_238293


namespace odd_factors_of_420_l238_238627

/-- 
Proof problem: Given that 420 can be factorized as \( 2^2 \times 3 \times 5 \times 7 \), 
we need to prove that the number of odd factors of 420 is 8.
-/
def number_of_odd_factors (n : ℕ) : ℕ :=
  let odd_factors := n.factors.filter (λ x, x % 2 ≠ 0)
  in odd_factors.length

theorem odd_factors_of_420 : number_of_odd_factors 420 = 8 := 
sorry

end odd_factors_of_420_l238_238627


namespace disputed_piece_weight_l238_238658

theorem disputed_piece_weight (x d : ℝ) (h1 : x - d = 300) (h2 : x + d = 500) : x = 400 := by
  sorry

end disputed_piece_weight_l238_238658


namespace cos_300_eq_half_l238_238801

theorem cos_300_eq_half : Real.cos (300 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end cos_300_eq_half_l238_238801


namespace fraction_decimal_equivalent_l238_238009

theorem fraction_decimal_equivalent : (7 / 16 : ℝ) = 0.4375 :=
by
  sorry

end fraction_decimal_equivalent_l238_238009


namespace part1_solution_set_part2_values_of_a_l238_238868

-- Parameters and definitions for the function and conditions
def f (x a : ℝ) := |x - a^2| + |x - (2*a + 1)|

-- Part (1) When a = 2, the inequality's solution set
theorem part1_solution_set (x : ℝ) : f x 2 ≥ 4 ↔ (x ≤ 3/2 ∨ x ≥ 11/2) := 
sorry

-- Part (2) Range of values for a given f(x, a) ≥ 4
theorem part2_values_of_a (a : ℝ) : 
∀ x, f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end part1_solution_set_part2_values_of_a_l238_238868


namespace savings_percentage_is_correct_l238_238571

-- Definitions for given conditions
def jacket_original_price : ℕ := 100
def shirt_original_price : ℕ := 50
def shoes_original_price : ℕ := 60

def jacket_discount : ℝ := 0.30
def shirt_discount : ℝ := 0.40
def shoes_discount : ℝ := 0.25

-- Definitions for savings
def jacket_savings : ℝ := jacket_original_price * jacket_discount
def shirt_savings : ℝ := shirt_original_price * shirt_discount
def shoes_savings : ℝ := shoes_original_price * shoes_discount

-- Definition for total savings and total original cost
def total_savings : ℝ := jacket_savings + shirt_savings + shoes_savings
def total_original_cost : ℕ := jacket_original_price + shirt_original_price + shoes_original_price

-- The theorems to be proven
theorem savings_percentage_is_correct : (total_savings / total_original_cost * 100) = 30.95 := by
  sorry

end savings_percentage_is_correct_l238_238571


namespace probability_heads_equals_l238_238087

theorem probability_heads_equals (p q: ℚ) (h1 : q = 1 - p) (h2 : (binomial 10 5) * p^5 * q^5 = (binomial 10 6) * p^6 * q^4) : p = 6 / 11 :=
by {
  sorry
}

end probability_heads_equals_l238_238087


namespace equation_one_solution_equation_two_solution_l238_238988

theorem equation_one_solution (x : ℝ) : (6 * x - 7 = 4 * x - 5) ↔ (x = 1) := by
  sorry

theorem equation_two_solution (x : ℝ) : ((x + 1) / 2 - 1 = 2 + (2 - x) / 4) ↔ (x = 4) := by
  sorry

end equation_one_solution_equation_two_solution_l238_238988


namespace quotient_of_1575_210_l238_238994

theorem quotient_of_1575_210 (a b q : ℕ) (h1 : a = 1575) (h2 : b = a - 1365) (h3 : a % b = 15) : q = 7 :=
by {
  sorry
}

end quotient_of_1575_210_l238_238994


namespace nested_radical_eq_6_l238_238603

theorem nested_radical_eq_6 (x : ℝ) (h : x = Real.sqrt (18 + x)) : x = 6 :=
by 
  have h_eq : x^2 = 18 + x,
  { rw h, exact pow_two (Real.sqrt (18 + x)) },
  have quad_eq : x^2 - x - 18 = 0,
  { linarith [h_eq] },
  have factored : (x - 6) * (x + 3) = x^2 - x - 18,
  { ring },
  rw [←quad_eq, factored] at h,
  sorry

end nested_radical_eq_6_l238_238603


namespace range_of_a_l238_238856

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 2 then (a - 5) * x + 8 else 2 * a / x

theorem range_of_a (a : ℝ) : 
  (∀ x y, x < y → f a x ≥ f a y) → (2 ≤ a ∧ a < 5) :=
sorry

end range_of_a_l238_238856


namespace number_of_ways_to_write_2024_l238_238492

theorem number_of_ways_to_write_2024 :
  (∃ a b c : ℕ, 2 * a + 3 * b + 4 * c = 2024) -> 
  (∃ n m p : ℕ, a = 3 * n + 2 * m + p ∧ n + m + p = 337) ->
  (∃ n m p : ℕ, n + m + p = 337 ∧ 2 * n * 3 + m * 2 + p * 6 = 2 * (57231 + 498)) :=
sorry

end number_of_ways_to_write_2024_l238_238492


namespace gloria_coins_l238_238945

theorem gloria_coins (qd qda qdc : ℕ) (h1 : qdc = 350) (h2 : qda = qdc / 5) (h3 : qd = qda - (2 * qda / 5)) :
  qd + qdc = 392 :=
by sorry

end gloria_coins_l238_238945


namespace savings_after_increase_l238_238433

theorem savings_after_increase (salary savings_rate increase_rate : ℝ) (old_savings old_expenses new_expenses new_savings : ℝ)
  (h_salary : salary = 6000)
  (h_savings_rate : savings_rate = 0.2)
  (h_increase_rate : increase_rate = 0.2)
  (h_old_savings : old_savings = savings_rate * salary)
  (h_old_expenses : old_expenses = salary - old_savings)
  (h_new_expenses : new_expenses = old_expenses * (1 + increase_rate))
  (h_new_savings : new_savings = salary - new_expenses) :
  new_savings = 240 :=
by sorry

end savings_after_increase_l238_238433


namespace find_fourth_power_sum_l238_238950

theorem find_fourth_power_sum (a b c : ℝ) 
    (h1 : a + b + c = 2) 
    (h2 : a^2 + b^2 + c^2 = 3) 
    (h3 : a^3 + b^3 + c^3 = 4) : 
    a^4 + b^4 + c^4 = 7.833 :=
sorry

end find_fourth_power_sum_l238_238950


namespace expression_value_l238_238324

theorem expression_value (a b c : ℕ) (h1 : 25^a * 5^(2*b) = 5^6) (h2 : 4^b / 4^c = 4) : a^2 + a * b + 3 * c = 6 := by
  sorry

end expression_value_l238_238324


namespace pencil_distribution_l238_238404

theorem pencil_distribution (total_pens : ℕ) (total_pencils : ℕ) (max_students : ℕ) 
  (h1 : total_pens = 1001) (h2 : total_pencils = 910) (h3 : max_students = 91) : 
  total_pencils / max_students = 10 :=
by
  sorry

end pencil_distribution_l238_238404


namespace find_a_find_min_difference_l238_238860

noncomputable def f (a x : ℝ) : ℝ := x + a * Real.log x
noncomputable def g (a b x : ℝ) : ℝ := f a x + (1 / 2) * x ^ 2 - b * x

theorem find_a (a : ℝ) (h_perpendicular : (1 : ℝ) + a = 2) : a = 1 := 
sorry

theorem find_min_difference (a b x1 x2 : ℝ) (h_b : b ≥ (7 / 2)) 
    (hx1_lt_hx2 : x1 < x2) (hx_sum : x1 + x2 = b - 1)
    (hx_prod : x1 * x2 = 1) :
    g a b x1 - g a b x2 = (15 / 8) - 2 * Real.log 2 :=
sorry

end find_a_find_min_difference_l238_238860


namespace determine_value_of_x_l238_238352

theorem determine_value_of_x {b x : ℝ} (hb : 1 < b) (hx : 0 < x) 
  (h_eq : (4 * x)^(Real.logb b 2) = (5 * x)^(Real.logb b 3)) : 
  x = (4 / 5)^(Real.logb (3 / 2) b) :=
by
  sorry

end determine_value_of_x_l238_238352


namespace fraction_to_decimal_l238_238044

theorem fraction_to_decimal :
  (7 / 16 : ℝ) = 0.4375 := 
begin
  -- placeholder for the proof
  sorry
end

end fraction_to_decimal_l238_238044


namespace gigi_ate_33_bananas_l238_238472

def gigi_bananas (total_bananas : ℕ) (days : ℕ) (diff : ℕ) (bananas_day_7 : ℕ) : Prop :=
  ∃ b, (days * b + diff * ((days * (days - 1)) / 2)) = total_bananas ∧ 
       (b + 6 * diff) = bananas_day_7

theorem gigi_ate_33_bananas :
  gigi_bananas 150 7 4 33 :=
by {
  sorry
}

end gigi_ate_33_bananas_l238_238472


namespace arc_length_l238_238481

-- Define the conditions
def radius (r : ℝ) := 2 * r + 2 * r = 8
def central_angle (θ : ℝ) := θ = 2 -- Given the central angle

-- Define the length of the arc
def length_of_arc (l r : ℝ) := l = r * 2

-- Theorem stating that given the sector conditions, the length of the arc is 4 cm
theorem arc_length (r l : ℝ) (h1 : central_angle 2) (h2 : radius r) (h3 : length_of_arc l r) : l = 4 :=
by
  sorry

end arc_length_l238_238481


namespace unique_solution_of_quadratic_l238_238990

theorem unique_solution_of_quadratic :
  ∀ (b : ℝ), b ≠ 0 → (∃ x : ℝ, 3 * x^2 + b * x + 12 = 0 ∧ ∀ y : ℝ, 3 * y^2 + b * y + 12 = 0 → y = x) → 
  (b = 12 ∧ ∃ x : ℝ, x = -2 ∧ 3 * x^2 + 12 * x + 12 = 0 ∧ (∀ y : ℝ, 3 * y^2 + 12 * y + 12 = 0 → y = x)) ∨ 
  (b = -12 ∧ ∃ x : ℝ, x = 2 ∧ 3 * x^2 - 12 * x + 12 = 0 ∧ (∀ y : ℝ, 3 * y^2 - 12 * y + 12 = 0 → y = x)) :=
by 
  sorry

end unique_solution_of_quadratic_l238_238990


namespace euler_totient_inequality_l238_238644

open Int

def is_power_of_prime (m : ℕ) : Prop :=
  ∃ p k : ℕ, (Nat.Prime p) ∧ (k ≥ 1) ∧ (m = p^k)

def φ (n m : ℕ) (h : m ≠ 1) : ℕ := -- This is a placeholder, you would need an actual implementation for φ
  sorry

theorem euler_totient_inequality (m : ℕ) (h : m ≠ 1) :
  (is_power_of_prime m) ↔ (∀ n > 0, (φ n m h) / n ≥ (φ m m h) / m) :=
sorry

end euler_totient_inequality_l238_238644


namespace part1_solution_set_part2_range_of_a_l238_238875

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1_solution_set (x : ℝ) : 
  (f x 2 ≥ 4) ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
sorry

theorem part2_range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 4) → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end part1_solution_set_part2_range_of_a_l238_238875


namespace greatest_possible_sum_of_consecutive_integers_product_less_500_l238_238720

theorem greatest_possible_sum_of_consecutive_integers_product_less_500 :
  ∃ (n : ℤ), (n * (n + 1) < 500) ∧ (∀ (m : ℤ), (m * (m + 1) < 500) → (n + (n + 1) ≥ m + (m + 1))) :=
begin
  sorry
end

end greatest_possible_sum_of_consecutive_integers_product_less_500_l238_238720


namespace part1_solution_set_part2_values_of_a_l238_238863

-- Parameters and definitions for the function and conditions
def f (x a : ℝ) := |x - a^2| + |x - (2*a + 1)|

-- Part (1) When a = 2, the inequality's solution set
theorem part1_solution_set (x : ℝ) : f x 2 ≥ 4 ↔ (x ≤ 3/2 ∨ x ≥ 11/2) := 
sorry

-- Part (2) Range of values for a given f(x, a) ≥ 4
theorem part2_values_of_a (a : ℝ) : 
∀ x, f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end part1_solution_set_part2_values_of_a_l238_238863


namespace sum_divides_exp_sum_l238_238614

theorem sum_divides_exp_sum (p a b c d : ℕ) [Fact (Nat.Prime p)] 
  (h1 : 0 < a) (h2 : a < b) (h3 : b < c) (h4 : c < d) (h5 : d < p)
  (h6 : a^4 % p = b^4 % p) (h7 : b^4 % p = c^4 % p) (h8 : c^4 % p = d^4 % p) :
  (a + b + c + d) ∣ (a^2013 + b^2013 + c^2013 + d^2013) :=
sorry

end sum_divides_exp_sum_l238_238614


namespace greatest_possible_sum_of_consecutive_integers_product_less_500_l238_238721

theorem greatest_possible_sum_of_consecutive_integers_product_less_500 :
  ∃ (n : ℤ), (n * (n + 1) < 500) ∧ (∀ (m : ℤ), (m * (m + 1) < 500) → (n + (n + 1) ≥ m + (m + 1))) :=
begin
  sorry
end

end greatest_possible_sum_of_consecutive_integers_product_less_500_l238_238721


namespace triangle_side_lengths_log_l238_238538

theorem triangle_side_lengths_log (m : ℕ) (log15 log81 logm : ℝ)
  (h1 : log15 = Real.log 15 / Real.log 10)
  (h2 : log81 = Real.log 81 / Real.log 10)
  (h3 : logm = Real.log m / Real.log 10)
  (h4 : 0 < log15 ∧ 0 < log81 ∧ 0 < logm)
  (h5 : log15 + log81 > logm)
  (h6 : log15 + logm > log81)
  (h7 : log81 + logm > log15)
  (h8 : m > 0) :
  6 ≤ m ∧ m < 1215 → 
  ∃ n : ℕ, n = 1215 - 6 ∧ n = 1209 :=
by
  sorry

end triangle_side_lengths_log_l238_238538


namespace probability_sum_is_ten_l238_238180

structure Die :=
(value : ℕ)
(h_value : 1 ≤ value ∧ value ≤ 6)

def possible_rolls : List (ℕ × ℕ × ℕ) :=
List.product 
  (List.product [1, 2, 3, 4, 5, 6] [1, 2, 3, 4, 5, 6]) [1, 2, 3, 4, 5, 6]
|>.map (λ ((a, b), c) => (a, b, c))

def favorable_outcomes : List (ℕ × ℕ × ℕ) :=
possible_rolls.filter (λ (x, y, z) => x + y + z = 10)

def probability (n favorable : ℕ) : ℚ := favourable / n

theorem probability_sum_is_ten :
  probability possible_rolls.length favourable_outcomes.length = 1 / 9 :=
by
  sorry

end probability_sum_is_ten_l238_238180


namespace car_clock_correctness_l238_238663

variables {t_watch t_car : ℕ} 
--  Variable declarations for time on watch (accurate) and time on car clock.

-- Define the initial times at 8:00 AM
def initial_time_watch : ℕ := 8 * 60 -- 8:00 AM in minutes
def initial_time_car : ℕ := 8 * 60 -- also 8:00 AM in minutes

-- Define the known times in the afternoon
def afternoon_time_watch : ℕ := 14 * 60 -- 2:00 PM in minutes
def afternoon_time_car : ℕ := 14 * 60 + 10 -- 2:10 PM in minutes

-- Car clock runs 37 minutes in the time the watch runs 36 minutes
def car_clock_rate : ℕ × ℕ := (37, 36)

-- Check the car clock time when the accurate watch shows 10:00 PM
def car_time_at_10pm_watch : ℕ := 22 * 60 -- 10:00 PM in minutes

-- Define the actual time that we need to prove
def actual_time_at_10pm_car : ℕ := 21 * 60 + 47 -- 9:47 PM in minutes

theorem car_clock_correctness : 
  (t_watch = actual_time_at_10pm_car) ↔ 
  (t_car = car_time_at_10pm_watch) ∧ 
  (initial_time_watch = initial_time_car) ∧ 
  (afternoon_time_watch = 14 * 60) ∧ 
  (afternoon_time_car = 14 * 60 + 10) ∧ 
  (car_clock_rate = (37, 36)) :=
sorry

end car_clock_correctness_l238_238663


namespace part1_solution_set_part2_range_of_a_l238_238922

def f (x a : ℝ) := |x - a^2| + |x - (2*a + 1)|

-- Part (1)
theorem part1_solution_set (x : ℝ) :
  ∀ x, f x 2 ≥ 4 ↔ x ≤ 3 / 2 ∨ x ≥ 11 / 2 := sorry

-- Part (2)
theorem part2_range_of_a (a : ℝ) :
  (∀ x, f x a ≥ 4) ↔ (a ≤ -1 ∨ a ≥ 3) := sorry

end part1_solution_set_part2_range_of_a_l238_238922


namespace smallest_positive_period_of_f_intervals_where_f_is_monotonically_increasing_l238_238944

def vector_dot (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

noncomputable def f (x : ℝ) : ℝ :=
  vector_dot (2 * Real.cos x, Real.sin x) (Real.sqrt 3 * Real.cos x, 2 * Real.cos x) - Real.sqrt 3

theorem smallest_positive_period_of_f :
  ∃ T > 0, (∀ x : ℝ, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x : ℝ, f (x + T') = f x) → T ≤ T')
:= sorry

theorem intervals_where_f_is_monotonically_increasing :
  ∀ k : ℤ, ∀ x : ℝ, - (5 * Real.pi / 12) + (↑k * Real.pi) ≤ x ∧ x ≤ (↑k * Real.pi) + (Real.pi / 12) → 
            f' (x) > 0
:= sorry

end smallest_positive_period_of_f_intervals_where_f_is_monotonically_increasing_l238_238944


namespace AC_plus_third_BA_l238_238325

def point := (ℝ × ℝ)

def A : point := (2, 4)
def B : point := (-1, -5)
def C : point := (3, -2)

noncomputable def vec (p₁ p₂ : point) : point :=
  (p₂.1 - p₁.1, p₂.2 - p₁.2)

noncomputable def scal_mult (scalar : ℝ) (v : point) : point :=
  (scalar * v.1, scalar * v.2)

noncomputable def vec_add (v₁ v₂ : point) : point :=
  (v₁.1 + v₂.1, v₁.2 + v₂.2)

theorem AC_plus_third_BA : 
  vec_add (vec A C) (scal_mult (1 / 3) (vec B A)) = (2, -3) :=
by
  sorry

end AC_plus_third_BA_l238_238325


namespace arithmetic_sequence_15th_term_is_171_l238_238251

theorem arithmetic_sequence_15th_term_is_171 :
  ∀ (a d : ℕ), a = 3 → d = 15 - a → a + 14 * d = 171 :=
by
  intros a d h_a h_d
  rw [h_a, h_d]
  -- The proof would follow with the arithmetic calculation to determine the 15th term
  sorry

end arithmetic_sequence_15th_term_is_171_l238_238251


namespace shirts_total_cost_l238_238378

def shirt_cost_problem : Prop :=
  ∃ (first_shirt_cost second_shirt_cost total_cost : ℕ),
    first_shirt_cost = 15 ∧
    first_shirt_cost = second_shirt_cost + 6 ∧
    total_cost = first_shirt_cost + second_shirt_cost ∧
    total_cost = 24

theorem shirts_total_cost : shirt_cost_problem := by
  sorry

end shirts_total_cost_l238_238378


namespace square_circle_area_ratio_l238_238767

theorem square_circle_area_ratio {r : ℝ} (h : ∀ s : ℝ, 2 * r = s * Real.sqrt 2) :
  (2 * r ^ 2) / (Real.pi * r ^ 2) = 2 / Real.pi :=
by
  sorry

end square_circle_area_ratio_l238_238767


namespace savings_amount_l238_238448

-- Define the conditions for Celia's spending
def food_spending_per_week : ℝ := 100
def weeks : ℕ := 4
def rent_spending : ℝ := 1500
def video_streaming_services_spending : ℝ := 30
def cell_phone_usage_spending : ℝ := 50
def savings_rate : ℝ := 0.10

-- Define the total spending calculation
def total_spending : ℝ :=
  food_spending_per_week * weeks + rent_spending + video_streaming_services_spending + cell_phone_usage_spending

-- Define the savings calculation
def savings : ℝ :=
  savings_rate * total_spending

-- Prove the amount of savings
theorem savings_amount : savings = 198 :=
by
  -- This is the statement that needs to be proven, hence adding a placeholder proof.
  sorry

end savings_amount_l238_238448


namespace part1_solution_set_part2_values_of_a_l238_238869

-- Parameters and definitions for the function and conditions
def f (x a : ℝ) := |x - a^2| + |x - (2*a + 1)|

-- Part (1) When a = 2, the inequality's solution set
theorem part1_solution_set (x : ℝ) : f x 2 ≥ 4 ↔ (x ≤ 3/2 ∨ x ≥ 11/2) := 
sorry

-- Part (2) Range of values for a given f(x, a) ≥ 4
theorem part2_values_of_a (a : ℝ) : 
∀ x, f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end part1_solution_set_part2_values_of_a_l238_238869


namespace cos_300_eq_one_half_l238_238792

theorem cos_300_eq_one_half :
  Real.cos (300 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end cos_300_eq_one_half_l238_238792


namespace total_cost_of_shirts_l238_238373

theorem total_cost_of_shirts 
    (first_shirt_cost : ℤ)
    (second_shirt_cost : ℤ)
    (h1 : first_shirt_cost = 15)
    (h2 : first_shirt_cost = second_shirt_cost + 6) : 
    first_shirt_cost + second_shirt_cost = 24 := 
by
  sorry

end total_cost_of_shirts_l238_238373


namespace fraction_to_decimal_l238_238056

theorem fraction_to_decimal :
  (7 : ℝ) / 16 = 0.4375 := 
by sorry

end fraction_to_decimal_l238_238056


namespace polynomial_at_3_l238_238344

def f (x : ℝ) : ℝ := 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x

theorem polynomial_at_3 : f 3 = 1641 := 
by
  -- proof would go here
  sorry

end polynomial_at_3_l238_238344


namespace number_of_ways_to_construct_cube_l238_238097

theorem number_of_ways_to_construct_cube :
  let num_white_cubes := 5
  let num_blue_cubes := 3
  let cube_size := (2, 2, 2)
  let num_rotations := 24
  let num_constructions := 4
  ∃ (num_constructions : ℕ), num_constructions = 4 :=
sorry

end number_of_ways_to_construct_cube_l238_238097


namespace asymptotic_minimal_eccentricity_l238_238318

noncomputable def hyperbola (m : ℝ) (x y : ℝ) : Prop :=
  x^2 / m - y^2 / (m^2 + 4) = 1

noncomputable def eccentricity (m : ℝ) : ℝ :=
  Real.sqrt (m + 4 / m + 1)

theorem asymptotic_minimal_eccentricity :
  ∃ (m : ℝ), m = 2 ∧ hyperbola m x y → ∀ x y, y = 2 * x ∨ y = -2 * x :=
by
  sorry

end asymptotic_minimal_eccentricity_l238_238318


namespace proof_F_2_f_3_l238_238383

def f (a : ℕ) : ℕ := a ^ 2 - 1

def F (a : ℕ) (b : ℕ) : ℕ := 3 * b ^ 2 + 2 * a

theorem proof_F_2_f_3 : F 2 (f 3) = 196 := by
  have h1 : f 3 = 3 ^ 2 - 1 := rfl
  rw [h1]
  have h2 : 3 ^ 2 - 1 = 8 := by norm_num
  rw [h2]
  exact rfl

end proof_F_2_f_3_l238_238383


namespace dice_sum_probability_l238_238198

theorem dice_sum_probability : 
  let die_faces := {1, 2, 3, 4, 5, 6};
      successful_outcomes := { (x, y, z) | x ∈ die_faces ∧ y ∈ die_faces ∧ z ∈ die_faces ∧ x + y + z = 10 }.toFinset.card;
      total_outcomes := (finset.range 6).card ^ 3
  in  successful_outcomes / total_outcomes = 1 / 9 :=
by
  let die_faces := {1, 2, 3, 4, 5, 6};
  let successful_outcomes := { (x, y, z) | x ∈ die_faces ∧ y ∈ die_faces ∧ z ∈ die_faces ∧ x + y + z = 10 }.toFinset.card;
  let total_outcomes := (finset.range 6).card ^ 3;
  sorry

end dice_sum_probability_l238_238198


namespace probability_heads_l238_238088

theorem probability_heads (p : ℝ) (q : ℝ) (C_10_5 C_10_6 : ℝ)
  (h1 : q = 1 - p)
  (h2 : C_10_5 = 252)
  (h3 : C_10_6 = 210)
  (eqn : C_10_5 * p^5 * q^5 = C_10_6 * p^6 * q^4) :
  p = 6 / 11 := 
by
  sorry

end probability_heads_l238_238088


namespace houses_count_l238_238978

theorem houses_count (n : ℕ) 
  (h1 : ∃ k : ℕ, k + 7 = 12)
  (h2 : ∃ m : ℕ, m + 25 = 30) :
  n = 32 :=
sorry

end houses_count_l238_238978


namespace find_a_l238_238951

theorem find_a (a b c d : ℕ) (h1 : a + b = d) (h2 : b + c = 6) (h3 : c + d = 7) : a = 1 :=
by
  sorry

end find_a_l238_238951


namespace abs_diff_squares_l238_238698

-- Definitions for the numbers 105 and 95
def a : ℕ := 105
def b : ℕ := 95

-- Statement to prove: The absolute value of the difference between the squares of 105 and 95 is 2000.
theorem abs_diff_squares : |a^2 - b^2| = 2000 := by
  sorry

end abs_diff_squares_l238_238698


namespace fraction_to_decimal_l238_238075

theorem fraction_to_decimal : (7 : ℚ) / 16 = 4375 / 10000 := by
  sorry

end fraction_to_decimal_l238_238075


namespace greatest_possible_sum_of_two_consecutive_integers_lt_500_l238_238713

theorem greatest_possible_sum_of_two_consecutive_integers_lt_500 (n : ℕ) (h : n * (n + 1) < 500) : n + (n + 1) ≤ 43 := by
  sorry

end greatest_possible_sum_of_two_consecutive_integers_lt_500_l238_238713


namespace first_term_geometric_series_l238_238849

theorem first_term_geometric_series (a1 q : ℝ) (h1 : a1 / (1 - q) = 1)
  (h2 : |a1| / (1 - |q|) = 2) (h3 : -1 < q) (h4 : q < 1) (h5 : q ≠ 0) :
  a1 = 4 / 3 :=
by {
  sorry
}

end first_term_geometric_series_l238_238849


namespace positive_n_for_one_solution_l238_238470

theorem positive_n_for_one_solution :
  ∀ (n : ℝ), (4 * (0 : ℝ)) ^ 2 + n * (0) + 16 = 0 → (n^2 - 256 = 0) → n = 16 :=
by
  intro n
  intro h
  intro discriminant_eq_zero
  sorry

end positive_n_for_one_solution_l238_238470


namespace fraction_to_decimal_l238_238066

theorem fraction_to_decimal : (7 : ℝ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l238_238066


namespace greatest_possible_sum_consecutive_product_lt_500_l238_238735

noncomputable def largest_sum_consecutive_product_lt_500 : ℕ :=
  let n := nat.sub ((nat.sqrt 500) + 1) 1 in
  n + (n + 1)

theorem greatest_possible_sum_consecutive_product_lt_500 :
  (∃ (n : ℕ), n * (n + 1) < 500 ∧ largest_sum_consecutive_product_lt_500 = (n + (n + 1))) →
  largest_sum_consecutive_product_lt_500 = 43 := by
  sorry

end greatest_possible_sum_consecutive_product_lt_500_l238_238735


namespace greatest_possible_sum_consecutive_product_lt_500_l238_238738

noncomputable def largest_sum_consecutive_product_lt_500 : ℕ :=
  let n := nat.sub ((nat.sqrt 500) + 1) 1 in
  n + (n + 1)

theorem greatest_possible_sum_consecutive_product_lt_500 :
  (∃ (n : ℕ), n * (n + 1) < 500 ∧ largest_sum_consecutive_product_lt_500 = (n + (n + 1))) →
  largest_sum_consecutive_product_lt_500 = 43 := by
  sorry

end greatest_possible_sum_consecutive_product_lt_500_l238_238738


namespace probability_p_eq_l238_238091

theorem probability_p_eq (p q : ℝ) (h_q : q = 1 - p)
  (h_eq : (Nat.choose 10 5) * p^5 * q^5 = (Nat.choose 10 6) * p^6 * q^4) : 
  p = 6 / 11 :=
by
  sorry

end probability_p_eq_l238_238091


namespace best_fitting_model_l238_238215

/-- A type representing the coefficient of determination of different models -/
def r_squared (m : ℕ) : ℝ :=
  match m with
  | 1 => 0.98
  | 2 => 0.80
  | 3 => 0.50
  | 4 => 0.25
  | _ => 0 -- An auxiliary value for invalid model numbers

/-- The best fitting model is the one with the highest r_squared value --/
theorem best_fitting_model : r_squared 1 = max (r_squared 1) (max (r_squared 2) (max (r_squared 3) (r_squared 4))) :=
by
  sorry

end best_fitting_model_l238_238215


namespace fraction_to_decimal_l238_238013

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l238_238013


namespace egg_difference_l238_238231

theorem egg_difference
    (total_eggs : ℕ := 24)
    (broken_eggs : ℕ := 3)
    (cracked_eggs : ℕ := 6)
    (perfect_eggs : ℕ := total_eggs - broken_eggs - cracked_eggs)
    : (perfect_eggs - cracked_eggs = 9) :=
begin
  sorry
end

end egg_difference_l238_238231


namespace percentage_increase_in_expenditure_l238_238418

/-- Given conditions:
- The price of sugar increased by 32%
- The family's original monthly sugar consumption was 30 kg
- The family's new monthly sugar consumption is 25 kg
- The family's expenditure on sugar increased by 10%

Prove that the percentage increase in the family's expenditure on sugar is 10%. -/
theorem percentage_increase_in_expenditure (P : ℝ) :
  let initial_consumption := 30
  let new_consumption := 25
  let price_increase := 0.32
  let original_price := P
  let new_price := (1 + price_increase) * original_price
  let original_expenditure := initial_consumption * original_price
  let new_expenditure := new_consumption * new_price
  let expenditure_increase := new_expenditure - original_expenditure
  let percentage_increase := (expenditure_increase / original_expenditure) * 100
  percentage_increase = 10 := sorry

end percentage_increase_in_expenditure_l238_238418


namespace find_n_l238_238479

theorem find_n (a b c : ℝ) (h : a^2 + b^2 = c^2) (n : ℕ) (hn : n > 2) : 
  (a^n + b^n + c^n)^2 = 2 * (a^(2*n) + b^(2*n) + c^(2*n)) → n = 4 :=
by
  sorry

end find_n_l238_238479


namespace european_confidence_95_european_teams_not_face_l238_238247

-- Definitions for the conditions
def european_teams_round_of_16 := 44
def european_teams_not_round_of_16 := 22
def other_regions_round_of_16 := 36
def other_regions_not_round_of_16 := 58
def total_teams := 160

-- Formula for K^2 calculation
def k_value : ℚ := 3.841
def k_squared (n a_d_diff b_c_diff a b c d : ℚ) : ℚ :=
  n * ((a_d_diff - b_c_diff)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Definitions and calculation of K^2
def n1 := (european_teams_round_of_16 + other_regions_round_of_16 : ℚ)
def a_d_diff1 := (european_teams_round_of_16 * other_regions_not_round_of_16 : ℚ)
def b_c_diff1 := (european_teams_not_round_of_16 * other_regions_round_of_16 : ℚ)
def k_squared_result := k_squared n1 a_d_diff1 b_c_diff1
                                 (european_teams_round_of_16 + european_teams_not_round_of_16)
                                 (other_regions_round_of_16 + other_regions_not_round_of_16)
                                 total_teams total_teams

-- Theorem for 95% confidence derived
theorem european_confidence_95 :
  k_squared_result > k_value := sorry

-- Probability calculation setup
def total_ways_to_pair_teams : ℚ := 15
def ways_european_teams_not_face : ℚ := 6
def probability_european_teams_not_face := ways_european_teams_not_face / total_ways_to_pair_teams

-- Theorem for probability
theorem european_teams_not_face :
  probability_european_teams_not_face = 2 / 5 := sorry

end european_confidence_95_european_teams_not_face_l238_238247


namespace part1_solution_set_part2_range_of_a_l238_238913

-- Define the function f for a general a
def f (x a : ℝ) : ℝ := |x - a^2| + |x - (2 * a) + 1|

-- Part 1: When a = 2
theorem part1_solution_set (x : ℝ) (h : f x 2 ≥ 4) : x ≤ 3/2 ∨ x ≥ 11/2 :=
  sorry

-- Part 2: Range of values for a
theorem part2_range_of_a (a : ℝ) (h : ∀ x, f x a ≥ 4) : a ≤ -1 ∨ a ≥ 3 :=
  sorry

end part1_solution_set_part2_range_of_a_l238_238913


namespace surface_area_correct_l238_238608

def w := 3 -- width in cm
def l := 4 -- length in cm
def h := 5 -- height in cm

def surface_area : Nat := 
  2 * (h * w) + 2 * (l * w) + 2 * (l * h)

theorem surface_area_correct : surface_area = 94 := 
  by
    sorry

end surface_area_correct_l238_238608


namespace mul_three_point_six_and_zero_point_twenty_five_l238_238298

theorem mul_three_point_six_and_zero_point_twenty_five : 3.6 * 0.25 = 0.9 := by 
  sorry

end mul_three_point_six_and_zero_point_twenty_five_l238_238298


namespace ordering_of_exponentials_l238_238306

theorem ordering_of_exponentials :
  let A := 3^20
  let B := 6^10
  let C := 2^30
  B < A ∧ A < C :=
by
  -- Definitions and conditions
  have h1 : 6^10 = 3^10 * 2^10 := by sorry
  have h2 : 3^10 = 59049 := by sorry
  have h3 : 2^10 = 1024 := by sorry
  have h4 : 2^30 = (2^10)^3 := by sorry
  
  -- We know 3^20, 6^10, 2^30 by definition and conditions
  -- Comparison
  have h5 : 3^20 = (3^10)^2 := by sorry
  have h6 : 2^30 = 1024^3 := by sorry
  
  -- Combine to get results
  have h7 : (3^10)^2 > 6^10 := by sorry
  have h8 : 1024^3 > 6^10 := by sorry
  have h9 : 1024^3 > (3^10)^2 := by sorry

  exact ⟨h7, h9⟩

end ordering_of_exponentials_l238_238306


namespace binary_representation_253_l238_238453

-- Define the decimal number
def decimal := 253

-- Define the number of zeros (x) and ones (y) in the binary representation of 253
def num_zeros := 1
def num_ones := 7

-- Prove that 2y - x = 13 given these conditions
theorem binary_representation_253 : (2 * num_ones - num_zeros) = 13 :=
by
  sorry

end binary_representation_253_l238_238453


namespace equal_tuesdays_and_fridays_l238_238763

theorem equal_tuesdays_and_fridays (days_in_month : ℕ) (days_of_week : ℕ) (extra_days : ℕ) (starting_days : Finset ℕ) :
  days_in_month = 30 → days_of_week = 7 → extra_days = 2 →
  starting_days = {0, 3, 6} →
  ∃ n : ℕ, n = 3 :=
by
  sorry

end equal_tuesdays_and_fridays_l238_238763


namespace fractional_to_decimal_l238_238060

theorem fractional_to_decimal : (7 / 16 : ℚ) = 0.4375 := by
  sorry

end fractional_to_decimal_l238_238060


namespace fraction_is_correct_l238_238620

def f (x : ℕ) : ℕ := 3 * x + 2
def g (x : ℕ) : ℕ := 2 * x - 3

theorem fraction_is_correct : (f (g (f 3))) / (g (f (g 3))) = 59 / 19 :=
by
  sorry

end fraction_is_correct_l238_238620


namespace find_interest_rate_l238_238655

-- Define the conditions
def total_amount : ℝ := 2500
def second_part_rate : ℝ := 0.06
def annual_income : ℝ := 145
def first_part_amount : ℝ := 500.0000000000002
noncomputable def interest_rate (r : ℝ) : Prop :=
  first_part_amount * r + (total_amount - first_part_amount) * second_part_rate = annual_income

-- State the theorem
theorem find_interest_rate : interest_rate 0.05 :=
by
  sorry

end find_interest_rate_l238_238655


namespace square_value_is_10000_l238_238416
noncomputable def squareValue : Real := 6400000 / 400 / 1.6

theorem square_value_is_10000 : squareValue = 10000 :=
  by
  -- The proof is based on the provided steps, which will be omitted here.
  sorry

end square_value_is_10000_l238_238416


namespace total_clouds_counted_l238_238117

def clouds_counted (carson_clouds : ℕ) (brother_factor : ℕ) : ℕ :=
  carson_clouds + (carson_clouds * brother_factor)

theorem total_clouds_counted (carson_clouds brother_factor total_clouds : ℕ) 
  (h₁ : carson_clouds = 6) (h₂ : brother_factor = 3) (h₃ : total_clouds = 24) :
  clouds_counted carson_clouds brother_factor = total_clouds :=
by
  sorry

end total_clouds_counted_l238_238117


namespace dice_sum_probability_l238_238185

/-- The probability that the sum of the numbers on three six-faced dice equals 10 is 1/8. -/
theorem dice_sum_probability : 
  (1 / 8 : ℚ) = (∑ x in (finset.univ : finset (fin 6)), 
                ∑ y in (finset.univ : finset (fin 6)), 
                ∑ z in (finset.univ : finset (fin 6)), 
                if x.1 + y.1 + z.1 = 10 then 1 else 0) / 6^3 :=
begin
  sorry
end

end dice_sum_probability_l238_238185


namespace min_value_sum_inverse_squares_l238_238382

theorem min_value_sum_inverse_squares (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_sum : a + b + c = 3) :
    (1 / (a + b)^2) + (1 / (a + c)^2) + (1 / (b + c)^2) >= 3 / 2 :=
sorry

end min_value_sum_inverse_squares_l238_238382


namespace incorrect_expression_l238_238953

theorem incorrect_expression (x y : ℝ) (h : x > y) : ¬ (1 - 3*x > 1 - 3*y) :=
sorry

end incorrect_expression_l238_238953


namespace cos_300_eq_half_l238_238786

theorem cos_300_eq_half :
  ∀ (deg : ℝ), deg = 300 → cos (deg * π / 180) = 1 / 2 :=
by
  intros deg h_deg
  rw [h_deg]
  have h1 : 300 = 360 - 60, by norm_num
  rw [h1]
  -- Convert angles to radians for usage with cos function
  have h2 : 360 - 60 = (2 * π - (π / 3)) * 180 / π, by norm_num
  rw [h2]
  have h3 : cos ((2 * π - π / 3) * π / 180) = cos (π / 3), by rw [real.cos_sub_pi_div_two, real.cos_pi_div_three]
  rw [h3]
  norm_num

end cos_300_eq_half_l238_238786


namespace greatest_sum_of_consecutive_integers_product_less_500_l238_238718

theorem greatest_sum_of_consecutive_integers_product_less_500 :
  ∃ n : ℤ, n * (n + 1) < 500 ∧ (n + (n + 1)) = 43 :=
by
  sorry

end greatest_sum_of_consecutive_integers_product_less_500_l238_238718


namespace greatest_possible_sum_consecutive_product_lt_500_l238_238736

noncomputable def largest_sum_consecutive_product_lt_500 : ℕ :=
  let n := nat.sub ((nat.sqrt 500) + 1) 1 in
  n + (n + 1)

theorem greatest_possible_sum_consecutive_product_lt_500 :
  (∃ (n : ℕ), n * (n + 1) < 500 ∧ largest_sum_consecutive_product_lt_500 = (n + (n + 1))) →
  largest_sum_consecutive_product_lt_500 = 43 := by
  sorry

end greatest_possible_sum_consecutive_product_lt_500_l238_238736


namespace polynomial_root_condition_l238_238316

noncomputable def polynomial_q (q x : ℝ) : ℝ :=
  x^6 + 3 * q * x^4 + 3 * x^4 + 3 * q * x^2 + x^2 + 3 * q + 1

theorem polynomial_root_condition (q : ℝ) :
  (∃ x > 0, polynomial_q q x = 0) ↔ (q ≥ 3 / 2) :=
sorry

end polynomial_root_condition_l238_238316


namespace parallel_segments_k_value_l238_238308

open Real

theorem parallel_segments_k_value :
  let A' := (-6, 0)
  let B' := (0, -6)
  let X' := (0, 12)
  ∃ k : ℝ,
  let Y' := (18, k)
  let m_ab := (B'.2 - A'.2) / (B'.1 - A'.1)
  let m_xy := (Y'.2 - X'.2) / (Y'.1 - X'.1)
  m_ab = m_xy → k = -6 :=
by
  sorry

end parallel_segments_k_value_l238_238308


namespace dice_sum_prob_10_l238_238177

/-- A standard six-faced die has faces numbered 1 through 6 --/
def is_standard_die_face (n : ℕ) : Prop := n ∈ {1, 2, 3, 4, 5, 6}

/-- The probability of the sum of the numbers rolled on three standard six-faced dice being 10 --/
theorem dice_sum_prob_10 : 
  ∃ (count_10 : ℕ), ∃ (total : ℕ), 
  (total = 6^3) ∧ 
  (count_10 = 27) ∧ 
  (count_10.toRat / total.toRat = 27 / 216) := 
by 
  sorry

end dice_sum_prob_10_l238_238177


namespace correct_statements_count_l238_238610

noncomputable def f (x : ℝ) := (1 / 2) * Real.sin (2 * x)
noncomputable def g (x : ℝ) := (1 / 2) * Real.sin (2 * x + Real.pi / 4)

theorem correct_statements_count :
  let complexity := True
    -- Condition 1: Smallest positive period of f(x) is 2π
    (∀ x : ℝ, f (x + 2 * Real.pi) = f x) = False ∧
    -- Condition 2: f(x) is monotonically increasing on [-π/4, π/4]
    (∀ x y : ℝ, -Real.pi / 4 ≤ x ∧ x ≤ y ∧ y ≤ Real.pi / 4 → f x ≤ f y) = True ∧
    -- Condition 3: Range of f(x) when x ∈ [-π/6, π/3]
    (∀ y : ℝ, ∀ x : ℝ, -Real.pi / 6 ≤ x ∧ x ≤ Real.pi / 3 → f x = y ↔ -Real.sqrt(3) / 4 ≤ y ∧ y ≤ Real.sqrt(3) / 4) = False ∧
    -- Condition 4: Graph of f(x) can be obtained by shifting g(x) to the left by π/8
    (∀ x : ℝ, f x = g (x + Real.pi / 8)) = False →
  true := sorry

end correct_statements_count_l238_238610


namespace block3_reaches_target_l238_238426

-- Type representing the position of a block on a 3x7 grid
structure Position where
  row : Nat
  col : Nat
  deriving DecidableEq, Repr

-- Defining the initial positions of blocks
def Block1Start : Position := ⟨2, 2⟩
def Block2Start : Position := ⟨3, 5⟩
def Block3Start : Position := ⟨1, 4⟩

-- The target position in the center of the board
def TargetPosition : Position := ⟨3, 5⟩

-- A function to represent if blocks collide or not
def canMove (current : Position) (target : Position) (blocks : List Position) : Prop :=
  target.row < 3 ∧ target.col < 7 ∧ ¬(target ∈ blocks)

-- Main theorem stating the goal
theorem block3_reaches_target : ∃ (steps : Nat → Position), steps 0 = Block3Start ∧ steps 7 = TargetPosition :=
  sorry

end block3_reaches_target_l238_238426


namespace fraction_to_decimal_l238_238074

theorem fraction_to_decimal : (7 : ℝ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l238_238074


namespace solve_for_k_l238_238625

variable {x y k : ℝ}

theorem solve_for_k (h1 : 2 * x + y = 1) (h2 : x + 2 * y = k - 2) (h3 : x - y = 2) : k = 1 :=
by
  sorry

end solve_for_k_l238_238625


namespace sum_of_midpoints_of_triangle_l238_238545

theorem sum_of_midpoints_of_triangle (a b c : ℝ) (h : a + b + c = 15) :
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 :=
by
  sorry

end sum_of_midpoints_of_triangle_l238_238545


namespace traveling_zoo_l238_238960

theorem traveling_zoo (x y : ℕ) (h1 : x + y = 36) (h2 : 4 * x + 6 * y = 100) : x = 14 ∧ y = 22 :=
by {
  sorry
}

end traveling_zoo_l238_238960


namespace solve_x_for_collinear_and_same_direction_l238_238156

-- Define vectors a and b
def vector_a (x : ℝ) : ℝ × ℝ := (-1, x)
def vector_b (x : ℝ) : ℝ × ℝ := (-x, 2)

-- Define the conditions for collinearity and same direction
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k • b.1, k • b.2)

def same_direction (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ a = (k • b.1, k • b.2)

theorem solve_x_for_collinear_and_same_direction
  (x : ℝ)
  (h_collinear : collinear (vector_a x) (vector_b x))
  (h_same_direction : same_direction (vector_a x) (vector_b x)) :
  x = Real.sqrt 2 :=
sorry

end solve_x_for_collinear_and_same_direction_l238_238156


namespace dice_sum_probability_l238_238199

theorem dice_sum_probability : 
  let die_faces := {1, 2, 3, 4, 5, 6};
      successful_outcomes := { (x, y, z) | x ∈ die_faces ∧ y ∈ die_faces ∧ z ∈ die_faces ∧ x + y + z = 10 }.toFinset.card;
      total_outcomes := (finset.range 6).card ^ 3
  in  successful_outcomes / total_outcomes = 1 / 9 :=
by
  let die_faces := {1, 2, 3, 4, 5, 6};
  let successful_outcomes := { (x, y, z) | x ∈ die_faces ∧ y ∈ die_faces ∧ z ∈ die_faces ∧ x + y + z = 10 }.toFinset.card;
  let total_outcomes := (finset.range 6).card ^ 3;
  sorry

end dice_sum_probability_l238_238199


namespace captain_age_l238_238214

theorem captain_age (C : ℕ) (h1 : ∀ W : ℕ, W = C + 3) 
                    (h2 : 21 * 11 = 231) 
                    (h3 : 21 - 1 = 20) 
                    (h4 : 20 * 9 = 180)
                    (h5 : 231 - 180 = 51) :
  C = 24 :=
by
  sorry

end captain_age_l238_238214


namespace find_z_l238_238282

-- Given conditions as Lean definitions
def consecutive (x y z : ℕ) : Prop := x = z + 2 ∧ y = z + 1 ∧ x > y ∧ y > z
def equation (x y z : ℕ) : Prop := 2 * x + 3 * y + 3 * z = 5 * y + 11

-- The statement to be proven
theorem find_z (x y z : ℕ) (h1 : consecutive x y z) (h2 : equation x y z) : z = 3 :=
sorry

end find_z_l238_238282


namespace required_cement_l238_238103

def total_material : ℝ := 0.67
def sand : ℝ := 0.17
def dirt : ℝ := 0.33
def cement : ℝ := 0.17

theorem required_cement : cement = total_material - (sand + dirt) := 
by
  sorry

end required_cement_l238_238103


namespace mutually_exclusive_one_two_odd_l238_238273

-- Define the event that describes rolling a fair die
def is_odd (n : ℕ) : Prop := n % 2 = 1

/-- Event: Exactly one die shows an odd number -/
def exactly_one_odd (d1 d2 : ℕ) : Prop :=
  (is_odd d1 ∧ ¬ is_odd d2) ∨ (¬ is_odd d1 ∧ is_odd d2)

/-- Event: Exactly two dice show odd numbers -/
def exactly_two_odd (d1 d2 : ℕ) : Prop :=
  is_odd d1 ∧ is_odd d2

/-- Main theorem: Exactly one odd number and exactly two odd numbers are mutually exclusive but not converse-/
theorem mutually_exclusive_one_two_odd (d1 d2 : ℕ) :
  (exactly_one_odd d1 d2 ∧ ¬ exactly_two_odd d1 d2) ∧
  (¬ exactly_one_odd d1 d2 ∧ exactly_two_odd d1 d2) ∧
  (exactly_one_odd d1 d2 ∨ exactly_two_odd d1 d2) :=
by
  sorry

end mutually_exclusive_one_two_odd_l238_238273


namespace fraction_to_decimal_l238_238077

theorem fraction_to_decimal : (7 : ℚ) / 16 = 4375 / 10000 := by
  sorry

end fraction_to_decimal_l238_238077


namespace least_number_divisor_l238_238744

theorem least_number_divisor (d : ℕ) (n m : ℕ) 
  (h1 : d = 1081)
  (h2 : m = 1077)
  (h3 : n = 4)
  (h4 : ∃ k, m + n = k * d) :
  d = 1081 :=
by
  sorry

end least_number_divisor_l238_238744


namespace class_overall_score_l238_238286

def max_score : ℝ := 100
def percentage_study : ℝ := 0.4
def percentage_hygiene : ℝ := 0.25
def percentage_discipline : ℝ := 0.25
def percentage_activity : ℝ := 0.1

def score_study : ℝ := 85
def score_hygiene : ℝ := 90
def score_discipline : ℝ := 80
def score_activity : ℝ := 75

theorem class_overall_score :
  (score_study * percentage_study) +
  (score_hygiene * percentage_hygiene) +
  (score_discipline * percentage_discipline) +
  (score_activity * percentage_activity) = 84 :=
  by sorry

end class_overall_score_l238_238286


namespace socks_count_l238_238583

theorem socks_count
  (black_socks : ℕ) 
  (white_socks : ℕ)
  (H1 : white_socks = 4 * black_socks)
  (H2 : black_socks = 6)
  (H3 : white_socks / 2 = white_socks - (white_socks / 2)) :
  (white_socks / 2) - black_socks = 6 := by
  sorry

end socks_count_l238_238583


namespace fraction_to_decimal_l238_238016

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l238_238016


namespace abs_diff_squares_105_95_l238_238686

def abs_diff_squares (a b : ℕ) : ℕ :=
  abs ((a ^ 2) - (b ^ 2))

theorem abs_diff_squares_105_95 : abs_diff_squares 105 95 = 2000 :=
by {
  let a := 105;
  let b := 95;
  have h1 : abs ((a ^ 2) - (b ^ 2)) = abs_diff_squares a b,
  simp [abs_diff_squares],
  sorry
}

end abs_diff_squares_105_95_l238_238686


namespace greatest_sum_consecutive_integers_product_less_than_500_l238_238702

theorem greatest_sum_consecutive_integers_product_less_than_500 :
  ∃ n : ℕ, n * (n + 1) < 500 ∧ (n + (n + 1) = 43) := sorry

end greatest_sum_consecutive_integers_product_less_than_500_l238_238702


namespace ny_mets_fans_count_l238_238561

theorem ny_mets_fans_count (Y M R : ℕ) (h1 : 3 * M = 2 * Y) (h2 : 4 * R = 5 * M) (h3 : Y + M + R = 390) : M = 104 := 
by
  sorry

end ny_mets_fans_count_l238_238561


namespace multiple_of_rohan_age_l238_238427

theorem multiple_of_rohan_age (x : ℝ) (h1 : 25 - 15 = 10) (h2 : 25 + 15 = 40) (h3 : 40 = x * 10) : x = 4 := 
by 
  sorry

end multiple_of_rohan_age_l238_238427


namespace part1_solution_set_part2_range_of_a_l238_238920

def f (x a : ℝ) := |x - a^2| + |x - (2*a + 1)|

-- Part (1)
theorem part1_solution_set (x : ℝ) :
  ∀ x, f x 2 ≥ 4 ↔ x ≤ 3 / 2 ∨ x ≥ 11 / 2 := sorry

-- Part (2)
theorem part2_range_of_a (a : ℝ) :
  (∀ x, f x a ≥ 4) ↔ (a ≤ -1 ∨ a ≥ 3) := sorry

end part1_solution_set_part2_range_of_a_l238_238920


namespace complement_of_A_in_U_l238_238943

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 4}
def complement_set (U A : Set ℕ) : Set ℕ := U \ A

theorem complement_of_A_in_U :
  complement_set U A = {2, 3, 5} :=
by
  apply Set.ext
  intro x
  simp [complement_set, U, A]
  sorry

end complement_of_A_in_U_l238_238943


namespace greatest_sum_consecutive_integers_lt_500_l238_238724

theorem greatest_sum_consecutive_integers_lt_500 : 
  ∃ n : ℤ, (n * (n + 1) < 500) ∧ n + (n + 1) = 43 := 
by {
  sorry -- Proof needed
}

end greatest_sum_consecutive_integers_lt_500_l238_238724


namespace no_solution_exists_l238_238607

   theorem no_solution_exists (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
     ¬ (3 / a + 4 / b = 12 / (a + b)) := 
   sorry
   
end no_solution_exists_l238_238607


namespace trap_speed_independent_of_location_l238_238397

theorem trap_speed_independent_of_location 
  (h b a : ℝ) (v_mouse : ℝ) 
  (path_length : ℝ := Real.sqrt (a^2 + (3*h)^2)) 
  (T : ℝ := path_length / v_mouse) 
  (step_height : ℝ := h) 
  (v_trap : ℝ := step_height / T) 
  (h_val : h = 3) 
  (b_val : b = 1) 
  (a_val : a = 8) 
  (v_mouse_val : v_mouse = 17) : 
  v_trap = 8 := by
  sorry

end trap_speed_independent_of_location_l238_238397


namespace arithmetic_sequence_problem_l238_238145

theorem arithmetic_sequence_problem (a₁ d S₁₀ : ℝ) (h1 : d < 0) (h2 : (a₁ + d) * (a₁ + 3 * d) = 12) 
  (h3 : (a₁ + d) + (a₁ + 3 * d) = 8) (h4 : S₁₀ = 10 * a₁ + 10 * (10 - 1) / 2 * d) : 
  true := sorry

end arithmetic_sequence_problem_l238_238145


namespace evaluate_expression_l238_238834

theorem evaluate_expression (a : ℕ) (h : a = 4) : (a ^ a - a * (a - 2) ^ a) ^ (a + 1) = 14889702426 :=
by
  rw [h]
  sorry

end evaluate_expression_l238_238834


namespace three_f_l238_238952

noncomputable def f (x : ℝ) : ℝ := sorry

theorem three_f (x : ℝ) (hx : 0 < x) (h : ∀ y > 0, f (3 * y) = 5 / (3 + y)) :
  3 * f x = 45 / (9 + x) :=
by
  sorry

end three_f_l238_238952


namespace harmonic_mean_of_x_and_y_l238_238227

noncomputable def x : ℝ := 88 + (40 / 100) * 88
noncomputable def y : ℝ := x - (25 / 100) * x
noncomputable def harmonic_mean (a b : ℝ) : ℝ := 2 / ((1 / a) + (1 / b))

theorem harmonic_mean_of_x_and_y :
  harmonic_mean x y = 105.6 :=
by
  sorry

end harmonic_mean_of_x_and_y_l238_238227


namespace part1_solution_set_part2_range_of_a_l238_238927

noncomputable def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1_solution_set (a : ℝ) (a_eq_2 : a = 2) : 
  {x : ℝ | f x a ≥ 4} = {x | x ≤ 3 / 2} ∪ {x | x ≥ 11 / 2} :=
by
  sorry

theorem part2_range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 4) → (a ∈ set.Iic (-1) ∪ set.Ici 3) :=
by
  sorry

end part1_solution_set_part2_range_of_a_l238_238927


namespace largest_value_expression_l238_238636

theorem largest_value_expression (a b c : ℝ) (ha : a ∈ ({1, 2, 4} : Set ℝ)) (hb : b ∈ ({1, 2, 4} : Set ℝ)) (hc : c ∈ ({1, 2, 4} : Set ℝ)) (habc_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  (a / 2) / (b / c) ≤ 4 :=
sorry

end largest_value_expression_l238_238636


namespace part1_solution_set_part2_range_of_a_l238_238934

-- Defining the function f(x) under given conditions
def f (x a : ℝ) : ℝ := |x - a^2| + |x - (2 * a + 1)|

-- Part 1: Determine the solution set for the inequality when a = 2
theorem part1_solution_set (x : ℝ) : (f x 2) ≥ 4 ↔ x ≤ 3/2 ∨ x ≥ 11/2 := by
  sorry

-- Part 2: Determine the range of values for a given the inequality
theorem part2_range_of_a (a : ℝ) : (∀ x : ℝ, f x a ≥ 4) ↔ a ≤ -1 ∨ a ≥ 3 := by
  sorry

end part1_solution_set_part2_range_of_a_l238_238934


namespace part_1_solution_set_part_2_a_range_l238_238892

-- Define the function f
def f (x a : ℝ) := |x - a^2| + |x - 2 * a + 1|

-- Part (1)
theorem part_1_solution_set (x : ℝ) : {x | x ≤ 3 / 2 ∨ x ≥ 11 / 2} = 
  {x | f x 2 ≥ 4} :=
sorry

-- Part (2)
theorem part_2_a_range (a : ℝ) : 
  {a | (a - 1)^2 ≥ 4} = {a | a ≤ -1 ∨ a ≥ 3} :=
sorry

end part_1_solution_set_part_2_a_range_l238_238892


namespace fractional_to_decimal_l238_238059

theorem fractional_to_decimal : (7 / 16 : ℚ) = 0.4375 := by
  sorry

end fractional_to_decimal_l238_238059


namespace total_clouds_l238_238115

theorem total_clouds (C B : ℕ) (h1 : C = 6) (h2 : B = 3 * C) : C + B = 24 := by
  sorry

end total_clouds_l238_238115


namespace f_for_negative_x_l238_238150

noncomputable def f (x : ℝ) : ℝ :=
if h : x > 0 then x * abs (x - 2) else 0  -- only assume the given case for x > 0

theorem f_for_negative_x (x : ℝ) (h : x < 0) : 
  f x = x * abs (x + 2) :=
by
  -- Sorry block to bypass the proof
  sorry

end f_for_negative_x_l238_238150


namespace age_sum_l238_238112

variables (A B C : ℕ)

theorem age_sum (h1 : A = 20 + B + C) (h2 : A^2 = 2000 + (B + C)^2) : A + B + C = 100 :=
by
  -- Assume the subsequent proof follows here
  sorry

end age_sum_l238_238112


namespace abs_diff_squares_105_95_l238_238687

theorem abs_diff_squares_105_95 : abs ((105 : ℤ)^2 - (95 : ℤ)^2) = 2000 := 
by
  sorry

end abs_diff_squares_105_95_l238_238687


namespace income_calculation_l238_238996

theorem income_calculation
  (x : ℕ)
  (income : ℕ := 5 * x)
  (expenditure : ℕ := 4 * x)
  (savings : ℕ := income - expenditure)
  (savings_eq : savings = 3000) :
  income = 15000 :=
sorry

end income_calculation_l238_238996


namespace greatest_sum_of_consecutive_integers_l238_238708

def consecutiveSumCondition (n : ℤ) : Prop :=
  n * (n + 1) < 500 

theorem greatest_sum_of_consecutive_integers : 
  ∃ n : ℤ, consecutiveSumCondition n ∧ ∀ m : ℤ, consecutiveSumCondition m → n + (n + 1) ≥ m + (m + 1) :=
sorry

end greatest_sum_of_consecutive_integers_l238_238708


namespace abs_diff_squares_l238_238692

theorem abs_diff_squares (a b : ℤ) (h_a : a = 105) (h_b : b = 95):
  |a^2 - b^2| = 2000 := by
  sorry

end abs_diff_squares_l238_238692


namespace cost_of_50_roses_l238_238586

def cost_of_dozen_roses : ℝ := 24

def is_proportional (n : ℕ) (cost : ℝ) : Prop :=
  cost = (cost_of_dozen_roses / 12) * n

def has_discount (n : ℕ) : Prop :=
  n ≥ 45

theorem cost_of_50_roses :
  ∃ (cost : ℝ), is_proportional 50 cost ∧ has_discount 50 ∧ cost * 0.9 = 90 :=
by
  sorry

end cost_of_50_roses_l238_238586


namespace cos_300_eq_half_l238_238785

theorem cos_300_eq_half :
  ∀ (deg : ℝ), deg = 300 → cos (deg * π / 180) = 1 / 2 :=
by
  intros deg h_deg
  rw [h_deg]
  have h1 : 300 = 360 - 60, by norm_num
  rw [h1]
  -- Convert angles to radians for usage with cos function
  have h2 : 360 - 60 = (2 * π - (π / 3)) * 180 / π, by norm_num
  rw [h2]
  have h3 : cos ((2 * π - π / 3) * π / 180) = cos (π / 3), by rw [real.cos_sub_pi_div_two, real.cos_pi_div_three]
  rw [h3]
  norm_num

end cos_300_eq_half_l238_238785


namespace probability_sum_is_ten_l238_238178

structure Die :=
(value : ℕ)
(h_value : 1 ≤ value ∧ value ≤ 6)

def possible_rolls : List (ℕ × ℕ × ℕ) :=
List.product 
  (List.product [1, 2, 3, 4, 5, 6] [1, 2, 3, 4, 5, 6]) [1, 2, 3, 4, 5, 6]
|>.map (λ ((a, b), c) => (a, b, c))

def favorable_outcomes : List (ℕ × ℕ × ℕ) :=
possible_rolls.filter (λ (x, y, z) => x + y + z = 10)

def probability (n favorable : ℕ) : ℚ := favourable / n

theorem probability_sum_is_ten :
  probability possible_rolls.length favourable_outcomes.length = 1 / 9 :=
by
  sorry

end probability_sum_is_ten_l238_238178


namespace dice_sum_probability_l238_238184

/-- The probability that the sum of the numbers on three six-faced dice equals 10 is 1/8. -/
theorem dice_sum_probability : 
  (1 / 8 : ℚ) = (∑ x in (finset.univ : finset (fin 6)), 
                ∑ y in (finset.univ : finset (fin 6)), 
                ∑ z in (finset.univ : finset (fin 6)), 
                if x.1 + y.1 + z.1 = 10 then 1 else 0) / 6^3 :=
begin
  sorry
end

end dice_sum_probability_l238_238184


namespace fraction_to_decimal_l238_238033

theorem fraction_to_decimal : (7 : ℝ) / 16 = 0.4375 := 
by
  sorry

end fraction_to_decimal_l238_238033


namespace abs_diff_squares_105_95_l238_238684

def abs_diff_squares (a b : ℕ) : ℕ :=
  abs ((a ^ 2) - (b ^ 2))

theorem abs_diff_squares_105_95 : abs_diff_squares 105 95 = 2000 :=
by {
  let a := 105;
  let b := 95;
  have h1 : abs ((a ^ 2) - (b ^ 2)) = abs_diff_squares a b,
  simp [abs_diff_squares],
  sorry
}

end abs_diff_squares_105_95_l238_238684


namespace infinite_geometric_series_sum_l238_238310

noncomputable def a : ℚ := 5 / 3
noncomputable def r : ℚ := -1 / 2

theorem infinite_geometric_series_sum : 
  ∑' (n : ℕ), a * r^n = 10 / 9 := 
by sorry

end infinite_geometric_series_sum_l238_238310


namespace sin_3x_over_4_period_l238_238411

noncomputable def sine_period (b : ℝ) : ℝ :=
  (2 * Real.pi) / b

theorem sin_3x_over_4_period :
  sine_period (3/4) = (8 * Real.pi) / 3 :=
by
  sorry

end sin_3x_over_4_period_l238_238411


namespace fraction_to_decimal_l238_238073

theorem fraction_to_decimal : (7 : ℝ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l238_238073


namespace prob_B_given_A_l238_238965

theorem prob_B_given_A (P_A P_B P_A_and_B : ℝ) (h1 : P_A = 0.06) (h2 : P_B = 0.08) (h3 : P_A_and_B = 0.02) :
  (P_A_and_B / P_A) = (1 / 3) :=
by
  -- substitute values
  sorry

end prob_B_given_A_l238_238965


namespace midpoints_on_straight_line_l238_238389

-- Define the standard form of a hyperbola
def hyperbola (a b x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

-- Define the equation of a chord, where m is the slope and c is the y-intercept
def chord (m c x y : ℝ) : Prop := y = m * x + c

-- Given that the chords are parallel, they have the same slope m.
-- We need to prove that the midpoints of these chords lie on a straight line.
theorem midpoints_on_straight_line (a b m : ℝ) (hm : m ≠ 0) :
  ∃ k k' : ℝ, ∀ c : ℝ, ∀ x1 x2 y1 y2 : ℝ,
  hyperbola a b x1 y1 →
  hyperbola a b x2 y2 →
  chord m c x1 y1 →
  chord m c x2 y2 →
  let xm := (x1 + x2) / 2,
  xm = k * c + k' :=
sorry

end midpoints_on_straight_line_l238_238389


namespace college_girls_count_l238_238213

/-- Given conditions:
 1. The ratio of the numbers of boys to girls is 8:5.
 2. The total number of students in the college is 416.
 
 Prove: The number of girls in the college is 160.
 -/
theorem college_girls_count (B G : ℕ) (h1 : B = (8 * G) / 5) (h2 : B + G = 416) : G = 160 :=
by
  sorry

end college_girls_count_l238_238213


namespace thor_jumps_to_exceed_29000_l238_238532

theorem thor_jumps_to_exceed_29000 :
  ∃ (n : ℕ), (3 ^ n) > 29000 ∧ n = 10 := sorry

end thor_jumps_to_exceed_29000_l238_238532


namespace six_digit_numbers_with_and_without_one_difference_l238_238291

theorem six_digit_numbers_with_and_without_one_difference :
  let total_numbers := Nat.choose 9 6 in
  let numbers_with_one := Nat.choose 8 5 in
  let numbers_without_one := total_numbers - numbers_with_one in
  numbers_with_one - numbers_without_one = 28 :=
by
  let total_numbers := Nat.choose 9 6
  let numbers_with_one := Nat.choose 8 5
  let numbers_without_one := total_numbers - numbers_with_one
  exact (numbers_with_one - numbers_without_one)
  sorry

end six_digit_numbers_with_and_without_one_difference_l238_238291


namespace abs_diff_squares_105_95_l238_238695

theorem abs_diff_squares_105_95 : abs ((105:ℤ)^2 - (95:ℤ)^2) = 2000 := by
  sorry

end abs_diff_squares_105_95_l238_238695


namespace sum_and_product_of_roots_l238_238240

-- Define the equation in terms of |x|
def equation (x : ℝ) : ℝ := |x|^3 - |x|^2 - 6 * |x| + 8

-- Lean statement to prove the sum and product of the roots
theorem sum_and_product_of_roots :
  (∀ x, equation x = 0 → (∃ L : List ℝ, L.sum = 0 ∧ L.prod = 16 ∧ ∀ y ∈ L, equation y = 0)) := 
sorry

end sum_and_product_of_roots_l238_238240


namespace dice_sum_probability_l238_238195

theorem dice_sum_probability : (∃ s, (∃ x y z : ℕ, 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 ∧ 1 ≤ z ∧ z ≤ 6 ∧ x + y + z = 10)
  ∧ s = {n : ℕ // ∃ x y z : ℕ, 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 ∧ 1 ≤ z ∧ z ≤ 6 ∧ x + y + z = n} ∧ s.card = 24)
  → 24 / 216 = 1 / 9 := 
sorry

end dice_sum_probability_l238_238195


namespace part_1_solution_set_part_2_a_range_l238_238888

-- Define the function f
def f (x a : ℝ) := |x - a^2| + |x - 2 * a + 1|

-- Part (1)
theorem part_1_solution_set (x : ℝ) : {x | x ≤ 3 / 2 ∨ x ≥ 11 / 2} = 
  {x | f x 2 ≥ 4} :=
sorry

-- Part (2)
theorem part_2_a_range (a : ℝ) : 
  {a | (a - 1)^2 ≥ 4} = {a | a ≤ -1 ∨ a ≥ 3} :=
sorry

end part_1_solution_set_part_2_a_range_l238_238888


namespace total_price_is_correct_l238_238407

def total_price_of_hats (total_hats : ℕ) (blue_hat_cost green_hat_cost : ℕ) (num_green_hats : ℕ) : ℕ :=
  let num_blue_hats := total_hats - num_green_hats
  let cost_green_hats := num_green_hats * green_hat_cost
  let cost_blue_hats := num_blue_hats * blue_hat_cost
  cost_green_hats + cost_blue_hats

theorem total_price_is_correct : total_price_of_hats 85 6 7 40 = 550 := 
  sorry

end total_price_is_correct_l238_238407


namespace fraction_of_phone_numbers_l238_238773

-- Define the total number of valid 7-digit phone numbers
def totalValidPhoneNumbers : Nat := 7 * 10^6

-- Define the number of valid phone numbers that begin with 3 and end with 5
def validPhoneNumbersBeginWith3EndWith5 : Nat := 10^5

-- Prove the fraction of phone numbers that begin with 3 and end with 5 is 1/70
theorem fraction_of_phone_numbers (h : validPhoneNumbersBeginWith3EndWith5 = 10^5) 
(h2 : totalValidPhoneNumbers = 7 * 10^6) : 
validPhoneNumbersBeginWith3EndWith5 / totalValidPhoneNumbers = 1 / 70 := 
sorry

end fraction_of_phone_numbers_l238_238773


namespace last_digit_of_large_prime_l238_238110

theorem last_digit_of_large_prime : 
  (859433 = 214858 * 4 + 1) → 
  (∃ d, (2 ^ 859433 - 1) % 10 = d ∧ d = 1) :=
by
  intro h
  sorry

end last_digit_of_large_prime_l238_238110


namespace fraction_to_decimal_l238_238072

theorem fraction_to_decimal : (7 : ℝ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l238_238072


namespace distance_on_map_is_correct_l238_238776

-- Define the parameters
def time_hours : ℝ := 1.5
def speed_mph : ℝ := 60
def map_scale_inches_per_mile : ℝ := 0.05555555555555555

-- Define the computation of actual distance and distance on the map
def actual_distance_miles : ℝ := speed_mph * time_hours
def distance_on_map_inches : ℝ := actual_distance_miles * map_scale_inches_per_mile

-- Theorem statement
theorem distance_on_map_is_correct :
  distance_on_map_inches = 5 :=
by 
  sorry

end distance_on_map_is_correct_l238_238776


namespace min_area_of_B_l238_238615

noncomputable def setA := { p : ℝ × ℝ | abs (p.1 - 2) + abs (p.2 - 3) ≤ 1 }

noncomputable def setB (D E F : ℝ) := { p : ℝ × ℝ | p.1^2 + p.2^2 + D * p.1 + E * p.2 + F ≤ 0 ∧ D^2 + E^2 - 4 * F > 0 }

theorem min_area_of_B (D E F : ℝ) (h : setA ⊆ setB D E F) : 
  ∃ r : ℝ, (∀ p ∈ setB D E F, p.1^2 + p.2^2 ≤ r^2) ∧ (π * r^2 = 2 * π) :=
sorry

end min_area_of_B_l238_238615


namespace no_valid_pairs_l238_238837

/-- 
Statement: There are no pairs of positive integers (a, b) such that
a * b + 100 = 25 * lcm(a, b) + 15 * gcd(a, b).
-/
theorem no_valid_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 
  a * b + 100 ≠ 25 * Nat.lcm a b + 15 * Nat.gcd a b :=
sorry

end no_valid_pairs_l238_238837


namespace bridge_length_l238_238997

def train_length : ℕ := 170 -- Train length in meters
def train_speed : ℕ := 45 -- Train speed in kilometers per hour
def crossing_time : ℕ := 30 -- Time to cross the bridge in seconds

noncomputable def speed_m_per_s : ℚ := (train_speed * 1000) / 3600

noncomputable def total_distance : ℚ := speed_m_per_s * crossing_time

theorem bridge_length : total_distance - train_length = 205 :=
by
  sorry

end bridge_length_l238_238997


namespace fraction_to_decimal_l238_238042

theorem fraction_to_decimal :
  (7 / 16 : ℝ) = 0.4375 := 
begin
  -- placeholder for the proof
  sorry
end

end fraction_to_decimal_l238_238042


namespace greatest_sum_consecutive_integers_lt_500_l238_238728

theorem greatest_sum_consecutive_integers_lt_500 : 
  ∃ n : ℤ, (n * (n + 1) < 500) ∧ n + (n + 1) = 43 := 
by {
  sorry -- Proof needed
}

end greatest_sum_consecutive_integers_lt_500_l238_238728


namespace fraction_to_decimal_l238_238027

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l238_238027


namespace fractional_to_decimal_l238_238062

theorem fractional_to_decimal : (7 / 16 : ℚ) = 0.4375 := by
  sorry

end fractional_to_decimal_l238_238062


namespace fraction_to_decimal_l238_238070

theorem fraction_to_decimal : (7 : ℝ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l238_238070


namespace evaluate_expression_l238_238412

theorem evaluate_expression : 
  (1 / 2 + ((2 / 3 * (3 / 8)) + 4) - (8 / 16)) = (17 / 4) :=
by
  sorry

end evaluate_expression_l238_238412


namespace power_of_square_l238_238590

variable {R : Type*} [CommRing R] (a : R)

theorem power_of_square (a : R) : (3 * a^2)^2 = 9 * a^4 :=
by sorry

end power_of_square_l238_238590


namespace abs_diff_squares_105_95_l238_238685

def abs_diff_squares (a b : ℕ) : ℕ :=
  abs ((a ^ 2) - (b ^ 2))

theorem abs_diff_squares_105_95 : abs_diff_squares 105 95 = 2000 :=
by {
  let a := 105;
  let b := 95;
  have h1 : abs ((a ^ 2) - (b ^ 2)) = abs_diff_squares a b,
  simp [abs_diff_squares],
  sorry
}

end abs_diff_squares_105_95_l238_238685


namespace total_cost_of_shirts_l238_238372

theorem total_cost_of_shirts 
    (first_shirt_cost : ℤ)
    (second_shirt_cost : ℤ)
    (h1 : first_shirt_cost = 15)
    (h2 : first_shirt_cost = second_shirt_cost + 6) : 
    first_shirt_cost + second_shirt_cost = 24 := 
by
  sorry

end total_cost_of_shirts_l238_238372


namespace part1_solution_set_part2_values_of_a_l238_238861

-- Parameters and definitions for the function and conditions
def f (x a : ℝ) := |x - a^2| + |x - (2*a + 1)|

-- Part (1) When a = 2, the inequality's solution set
theorem part1_solution_set (x : ℝ) : f x 2 ≥ 4 ↔ (x ≤ 3/2 ∨ x ≥ 11/2) := 
sorry

-- Part (2) Range of values for a given f(x, a) ≥ 4
theorem part2_values_of_a (a : ℝ) : 
∀ x, f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end part1_solution_set_part2_values_of_a_l238_238861


namespace expand_polynomial_eq_l238_238835

theorem expand_polynomial_eq :
  (3 * t^3 - 2 * t^2 + 4 * t - 1) * (2 * t^2 - 5 * t + 3) = 6 * t^5 - 19 * t^4 + 27 * t^3 - 28 * t^2 + 17 * t - 3 :=
by
  sorry

end expand_polynomial_eq_l238_238835


namespace part1_part2_l238_238884

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1 (x : ℝ) : 
  ∀ x, f x 2 ≥ 4 ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
by sorry

theorem part2 (x a : ℝ) :
  f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
by sorry

end part1_part2_l238_238884


namespace arithmetic_seq_sum_l238_238629

theorem arithmetic_seq_sum (a : ℕ → ℝ) (h_arith : ∀ n, a (n + 1) = a n + (a 2 - a 1)) 
(h_given : a 2 + a 8 = 10) : 
a 3 + a 7 = 10 :=
sorry

end arithmetic_seq_sum_l238_238629


namespace probability_p_eq_l238_238090

theorem probability_p_eq (p q : ℝ) (h_q : q = 1 - p)
  (h_eq : (Nat.choose 10 5) * p^5 * q^5 = (Nat.choose 10 6) * p^6 * q^4) : 
  p = 6 / 11 :=
by
  sorry

end probability_p_eq_l238_238090


namespace interval_of_monotonic_increase_l238_238667

noncomputable def y (x : ℝ) : ℝ := x^2 * Real.exp x
noncomputable def y' (x : ℝ) : ℝ := 2 * x * Real.exp x + x^2 * Real.exp x

theorem interval_of_monotonic_increase :
  ∀ x : ℝ, (y' x ≥ 0 ↔ (x ∈ Set.Ici 0 ∨ x ∈ Set.Iic (-2))) :=
by
  sorry

end interval_of_monotonic_increase_l238_238667


namespace arithmetic_mean_25_41_50_l238_238409

theorem arithmetic_mean_25_41_50 :
  (25 + 41 + 50) / 3 = 116 / 3 := by
  sorry

end arithmetic_mean_25_41_50_l238_238409


namespace number_of_solutions_l238_238628

open Real

theorem number_of_solutions :
  ∀ x : ℝ, (0 < x ∧ x < 3 * π) → (3 * cos x ^ 2 + 2 * sin x ^ 2 = 2) → 
  ∃ (L : Finset ℝ), L.card = 3 ∧ ∀ y ∈ L, 0 < y ∧ y < 3 * π ∧ 3 * cos y ^ 2 + 2 * sin y ^ 2 = 2 :=
by 
  sorry

end number_of_solutions_l238_238628


namespace greatest_sum_of_consecutive_integers_product_lt_500_l238_238732

theorem greatest_sum_of_consecutive_integers_product_lt_500 : 
  ∃ (n : ℕ), n * (n + 1) < 500 ∧ ∀ (m : ℕ), m * (m + 1) < 500 → m ≤ n → n + (n + 1) = 43 := 
by
  use 21
  split
  {
    norm_num
    linarith
  }
  {
    intros m h_hint h_ineq
    have : m ≤ 21, sorry
    linarith
  }
  sorry

end greatest_sum_of_consecutive_integers_product_lt_500_l238_238732


namespace second_smallest_odd_number_l238_238265

-- Define the conditions
def four_consecutive_odd_numbers_sum (n : ℕ) : Prop := 
  n % 2 = 1 ∧ (n + (n + 2) + (n + 4) + (n + 6) = 112)

-- State the theorem
theorem second_smallest_odd_number (n : ℕ) (h : four_consecutive_odd_numbers_sum n) : n + 2 = 27 :=
sorry

end second_smallest_odd_number_l238_238265


namespace three_bodies_with_triangle_front_view_l238_238569

def has_triangle_front_view (b : Type) : Prop :=
  -- Placeholder definition for example purposes
  sorry

theorem three_bodies_with_triangle_front_view :
  ∃ (body1 body2 body3 : Type),
  has_triangle_front_view body1 ∧
  has_triangle_front_view body2 ∧
  has_triangle_front_view body3 :=
sorry

end three_bodies_with_triangle_front_view_l238_238569


namespace sum_of_fractions_eq_13_5_l238_238447

noncomputable def sumOfFractions : ℚ :=
  (1/10 + 2/10 + 3/10 + 4/10 + 5/10 + 6/10 + 7/10 + 8/10 + 9/10 + 90/10)

theorem sum_of_fractions_eq_13_5 :
  sumOfFractions = 13.5 := by
  sorry

end sum_of_fractions_eq_13_5_l238_238447


namespace max_rows_l238_238319

theorem max_rows (m : ℕ) : (∀ T : Matrix (Fin m) (Fin 8) (Fin 4), 
  ∀ i j : Fin m, ∀ k l : Fin 8, i ≠ j ∧ T i k = T j k ∧ T i l = T j l → k ≠ l) → m ≤ 28 :=
sorry

end max_rows_l238_238319


namespace probability_sum_10_l238_238207

def is_valid_roll (d1 d2 d3 : ℕ) : Prop :=
  1 ≤ d1 ∧ d1 ≤ 6 ∧ 1 ≤ d2 ∧ d2 ≤ 6 ∧ 1 ≤ d3 ∧ d3 ≤ 6

def sum_is_10 (d1 d2 d3 : ℕ) : Prop :=
  d1 + d2 + d3 = 10

def valid_rolls_count : ℕ :=
  216 -- 6^3 distinct rolls of three 6-sided dice

def successful_rolls_count : ℕ :=
  24 -- number of valid rolls that sum to 10

theorem probability_sum_10 :
  (successful_rolls_count : ℚ) / valid_rolls_count = 1 / 9 := by
  sorry

end probability_sum_10_l238_238207


namespace fractional_to_decimal_l238_238065

theorem fractional_to_decimal : (7 / 16 : ℚ) = 0.4375 := by
  sorry

end fractional_to_decimal_l238_238065


namespace cos_300_eq_half_l238_238799

theorem cos_300_eq_half : Real.cos (300 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end cos_300_eq_half_l238_238799


namespace cos_300_eq_half_l238_238806

theorem cos_300_eq_half :
  let θ := 300
  let ref_θ := θ - 360 * (θ // 360) -- Reference angle in the proper range
  θ = 300 ∧ ref_θ = 60 ∧ cos 60 = 1/2 ∧ 300 > 270 → cos 300 = 1/2 :=
by
  intros
  sorry

end cos_300_eq_half_l238_238806


namespace skateboard_total_distance_l238_238436

theorem skateboard_total_distance :
  let a_1 := 8
  let d := 6
  let n := 40
  let distance (m : ℕ) := a_1 + (m - 1) * d
  let S_n := n / 2 * (distance 1 + distance n)
  S_n = 5000 := by
  sorry

end skateboard_total_distance_l238_238436


namespace alcohol_percentage_solution_x_l238_238392

theorem alcohol_percentage_solution_x :
  ∃ (P : ℝ), 
  (∀ (vol_x vol_y : ℝ), vol_x = 50 → vol_y = 150 →
    ∀ (percent_y percent_new : ℝ), percent_y = 30 → percent_new = 25 →
      ((P / 100) * vol_x + (percent_y / 100) * vol_y) / (vol_x + vol_y) = percent_new) → P = 10 :=
by
  -- Given conditions
  let vol_x := 50
  let vol_y := 150
  let percent_y := 30
  let percent_new := 25

  -- The proof body should be here
  sorry

end alcohol_percentage_solution_x_l238_238392


namespace cheaper_joint_work_l238_238233

theorem cheaper_joint_work (r L P : ℝ) (hr_pos : 0 < r) (hL_pos : 0 < L) (hP_pos : 0 < P) : 
  (2 * P * L) / (3 * r) < (3 * P * L) / (4 * r) :=
by
  sorry

end cheaper_joint_work_l238_238233


namespace greatest_possible_sum_of_consecutive_integers_product_less_500_l238_238719

theorem greatest_possible_sum_of_consecutive_integers_product_less_500 :
  ∃ (n : ℤ), (n * (n + 1) < 500) ∧ (∀ (m : ℤ), (m * (m + 1) < 500) → (n + (n + 1) ≥ m + (m + 1))) :=
begin
  sorry
end

end greatest_possible_sum_of_consecutive_integers_product_less_500_l238_238719


namespace greatest_sum_of_consecutive_integers_product_lt_500_l238_238743

theorem greatest_sum_of_consecutive_integers_product_lt_500 : 
  ∃ (n : ℤ), n * (n + 1) < 500 ∧ (∀ m : ℤ, m * (m + 1) < 500 → m + (m + 1) ≤ n + (n + 1)) ∧ n + (n + 1) = 43 :=
begin
  sorry
end

end greatest_sum_of_consecutive_integers_product_lt_500_l238_238743


namespace cos_300_is_half_l238_238787

noncomputable def cos_300_degrees : ℝ :=
  real.cos (300 * real.pi / 180)

theorem cos_300_is_half :
  cos_300_degrees = 1 / 2 :=
sorry

end cos_300_is_half_l238_238787


namespace expression_is_correct_l238_238131

theorem expression_is_correct (a : ℝ) : 2 * (a + 1) = 2 * a + 1 := 
sorry

end expression_is_correct_l238_238131


namespace probability_sum_is_10_l238_238192

theorem probability_sum_is_10 (d1 d2 d3 : ℕ) : 
  (1 ≤ d1 ∧ d1 ≤ 6) ∧ (1 ≤ d2 ∧ d2 ≤ 6) ∧ (1 ≤ d3 ∧ d3 ≤ 6) ∧ (d1 + d2 + d3 = 10) → 
  (((d1 = 1 ∧ d2 = 3 ∧ d3 = 6) ∨ 
    (d1 = 1 ∧ d2 = 4 ∧ d3 = 5) ∨ 
    (d1 = 2 ∧ d2 = 3 ∧ d3 = 5) ∨ 
    (d1 = 2 ∧ d2 = 4 ∧ d3 = 4) ∨ 
    (d1 = 3 ∧ d2 = 3 ∧ d3 = 4) ∨ 
    (d1 = 4 ∧ d2 = 4 ∧ d3 = 2)) 
    ↔ nat.cast (1 / 9)) :=
begin
  sorry,
end

end probability_sum_is_10_l238_238192


namespace fraction_to_decimal_l238_238052

theorem fraction_to_decimal :
  (7 : ℝ) / 16 = 0.4375 := 
by sorry

end fraction_to_decimal_l238_238052


namespace fraction_to_decimal_l238_238028

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l238_238028


namespace repeating_decimal_as_fraction_l238_238165

theorem repeating_decimal_as_fraction :
  ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ Int.natAbs (Int.gcd a b) = 1 ∧ a + b = 15 ∧ (a : ℚ) / b = 0.3636363636363636 :=
by
  sorry

end repeating_decimal_as_fraction_l238_238165


namespace simplify_expression_l238_238525

theorem simplify_expression :
  let a := 7
  let b := 2
  (a^5 + b^8) * (b^3 - (-b)^3)^7 = 0 := by
  let a := 7
  let b := 2
  sorry

end simplify_expression_l238_238525


namespace part1_solution_set_part2_range_of_a_l238_238926

noncomputable def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1_solution_set (a : ℝ) (a_eq_2 : a = 2) : 
  {x : ℝ | f x a ≥ 4} = {x | x ≤ 3 / 2} ∪ {x | x ≥ 11 / 2} :=
by
  sorry

theorem part2_range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 4) → (a ∈ set.Iic (-1) ∪ set.Ici 3) :=
by
  sorry

end part1_solution_set_part2_range_of_a_l238_238926


namespace min_path_length_l238_238361

noncomputable def problem_statement : Prop :=
  let XY := 12
  let XZ := 8
  let angle_XYZ := 30
  let YP_PQ_QZ := by {
    -- Reflect Z across XY to get Z' and Y across XZ to get Y'.
    -- Use the Law of cosines in triangle XY'Z'.
    let cos_150 := -Real.sqrt 3 / 2
    let Y_prime_Z_prime := Real.sqrt (8^2 + 12^2 + 2 * 8 * 12 * cos_150)
    exact Y_prime_Z_prime
  }
  ∃ (P Q : Type), (YP_PQ_QZ = Real.sqrt (208 + 96 * Real.sqrt 3))

-- Goal is to prove the problem statement
theorem min_path_length : problem_statement := sorry

end min_path_length_l238_238361


namespace complement_of_P_in_U_l238_238359

def U : Set ℤ := {-1, 0, 1, 2}
def P : Set ℤ := {x | -Real.sqrt 2 < x ∧ x < Real.sqrt 2}
def compl_U (P : Set ℤ) : Set ℤ := {x ∈ U | x ∉ P}

theorem complement_of_P_in_U : compl_U P = {2} :=
by
  sorry

end complement_of_P_in_U_l238_238359


namespace find_first_term_geom_seq_l238_238399

noncomputable def first_term (a r : ℝ) := a

theorem find_first_term_geom_seq 
  (a r : ℝ) 
  (h1 : a * r ^ 3 = 720) 
  (h2 : a * r ^ 6 = 5040) : 
  first_term a r = 720 / 7 := 
sorry

end find_first_term_geom_seq_l238_238399


namespace cos_300_is_half_l238_238812

noncomputable def cos_300_eq_half : Prop :=
  real.cos (2 * real.pi * 5 / 6) = 1 / 2

theorem cos_300_is_half : cos_300_eq_half :=
sorry

end cos_300_is_half_l238_238812


namespace probability_sum_is_10_l238_238188

theorem probability_sum_is_10 (d1 d2 d3 : ℕ) : 
  (1 ≤ d1 ∧ d1 ≤ 6) ∧ (1 ≤ d2 ∧ d2 ≤ 6) ∧ (1 ≤ d3 ∧ d3 ≤ 6) ∧ (d1 + d2 + d3 = 10) → 
  (((d1 = 1 ∧ d2 = 3 ∧ d3 = 6) ∨ 
    (d1 = 1 ∧ d2 = 4 ∧ d3 = 5) ∨ 
    (d1 = 2 ∧ d2 = 3 ∧ d3 = 5) ∨ 
    (d1 = 2 ∧ d2 = 4 ∧ d3 = 4) ∨ 
    (d1 = 3 ∧ d2 = 3 ∧ d3 = 4) ∨ 
    (d1 = 4 ∧ d2 = 4 ∧ d3 = 2)) 
    ↔ nat.cast (1 / 9)) :=
begin
  sorry,
end

end probability_sum_is_10_l238_238188


namespace scientific_notation_of_8_36_billion_l238_238980

theorem scientific_notation_of_8_36_billion : 
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 8.36 * 10^9 = a * 10^n := 
by
  use 8.36
  use 9
  simp
  sorry

end scientific_notation_of_8_36_billion_l238_238980


namespace strictly_positive_integer_le_36_l238_238133

theorem strictly_positive_integer_le_36 (n : ℕ) (h_pos : n > 0) :
  (∀ a : ℤ, (a % 2 = 1) → (a * a ≤ n) → (a ∣ n)) → n ≤ 36 := by
  sorry

end strictly_positive_integer_le_36_l238_238133


namespace weight_of_replaced_person_l238_238533

theorem weight_of_replaced_person :
  (∃ (W : ℝ), 
    let avg_increase := 1.5 
    let num_persons := 5 
    let new_person_weight := 72.5 
    (avg_increase * num_persons = new_person_weight - W)
  ) → 
  ∃ (W : ℝ), W = 65 :=
by
  sorry

end weight_of_replaced_person_l238_238533


namespace correct_equation_l238_238781

def distance : ℝ := 700
def speed_ratio : ℝ := 2.8
def time_difference : ℝ := 3.6

def express_train_time (x : ℝ) : ℝ := distance / x
def high_speed_train_time (x : ℝ) : ℝ := distance / (speed_ratio * x)

theorem correct_equation (x : ℝ) (hx : x ≠ 0) : 
  express_train_time x - high_speed_train_time x = time_difference :=
by
  unfold express_train_time high_speed_train_time
  sorry

end correct_equation_l238_238781


namespace abs_c_eq_116_l238_238530

theorem abs_c_eq_116 (a b c : ℤ) (h : Int.gcd a (Int.gcd b c) = 1) 
  (h_eq : a * (Complex.ofReal 3 + Complex.I) ^ 4 + 
          b * (Complex.ofReal 3 + Complex.I) ^ 3 + 
          c * (Complex.ofReal 3 + Complex.I) ^ 2 + 
          b * (Complex.ofReal 3 + Complex.I) + 
          a = 0) : 
  |c| = 116 :=
sorry

end abs_c_eq_116_l238_238530


namespace sum_of_even_conditions_l238_238989

theorem sum_of_even_conditions (m n : ℤ) :
  ((∃ k l : ℤ, m = 2 * k ∧ n = 2 * l) → ∃ p : ℤ, m + n = 2 * p) ∧
  (∃ q : ℤ, m + n = 2 * q → (∃ k l : ℤ, m = 2 * k ∧ n = 2 * l) → False) :=
by
  sorry

end sum_of_even_conditions_l238_238989


namespace dice_sum_10_probability_l238_238171

open Probability

noncomputable def probability_sum_10 : ℚ := 
  let total_outcomes := 216
  let favorable_outcomes := 21
  favorable_outcomes / total_outcomes

theorem dice_sum_10_probability :
  probability (λ (x : Fin 6 × Fin 6 × Fin 6), x.1.val + x.2.val + x.3.val + 3 = 10) = probability_sum_10 := 
sorry

end dice_sum_10_probability_l238_238171


namespace find_x_value_l238_238137

theorem find_x_value (x : ℝ) (h_pos : 0 < x) (h_eq : x * ⌊x⌋ = 72) : x = 9 :=
begin
  sorry
end

end find_x_value_l238_238137


namespace part1_solution_set_part2_range_of_a_l238_238878

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1_solution_set (x : ℝ) : 
  (f x 2 ≥ 4) ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
sorry

theorem part2_range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 4) → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end part1_solution_set_part2_range_of_a_l238_238878


namespace circle_radius_is_3_l238_238618

theorem circle_radius_is_3 (m : ℝ) (r : ℝ) :
  (∀ (M N : ℝ × ℝ), (M ∈ {p : ℝ × ℝ | p.1^2 + p.2^2 - 2 * p.1 + m * p.2 - 4 = 0} ∧
                     N ∈ {p : ℝ × ℝ | p.1^2 + p.2^2 - 2 * p.1 + m * p.2 - 4 = 0} ∧
                     M + N = (-(M.1 + N.1), -(M.2 + N.2))) →
  r = 3) :=
sorry

end circle_radius_is_3_l238_238618


namespace plan_y_more_cost_effective_l238_238111

theorem plan_y_more_cost_effective (m : Nat) : 2500 + 7 * m < 15 * m → 313 ≤ m :=
by
  intro h
  sorry

end plan_y_more_cost_effective_l238_238111


namespace part1_solution_set_part2_range_of_a_l238_238906

-- Define the function f for a general a
def f (x a : ℝ) : ℝ := |x - a^2| + |x - (2 * a) + 1|

-- Part 1: When a = 2
theorem part1_solution_set (x : ℝ) (h : f x 2 ≥ 4) : x ≤ 3/2 ∨ x ≥ 11/2 :=
  sorry

-- Part 2: Range of values for a
theorem part2_range_of_a (a : ℝ) (h : ∀ x, f x a ≥ 4) : a ≤ -1 ∨ a ≥ 3 :=
  sorry

end part1_solution_set_part2_range_of_a_l238_238906


namespace smallest_x_for_quadratic_l238_238415

theorem smallest_x_for_quadratic :
  ∃ x, 8 * x^2 - 38 * x + 35 = 0 ∧ (∀ y, 8 * y^2 - 38 * y + 35 = 0 → x ≤ y) ∧ x = 1.25 :=
by
  sorry

end smallest_x_for_quadratic_l238_238415


namespace probability_sum_is_10_l238_238211

theorem probability_sum_is_10 : 
  let outcomes := [(a, b, c) | a ∈ [1, 2, 3, 4, 5, 6], b ∈ [1, 2, 3, 4, 5, 6], c ∈ [1, 2, 3, 4, 5, 6]],
      favorable_outcomes := [(a, b, c) | (a, b, c) ∈ outcomes, a + b + c = 10] in
  (favorable_outcomes.length : ℚ) / outcomes.length = 1 / 9 := by
  sorry

end probability_sum_is_10_l238_238211


namespace fraction_to_decimal_l238_238000

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l238_238000


namespace total_campers_rowing_and_hiking_l238_238284

def campers_morning_rowing : ℕ := 41
def campers_morning_hiking : ℕ := 4
def campers_afternoon_rowing : ℕ := 26

theorem total_campers_rowing_and_hiking :
  campers_morning_rowing + campers_morning_hiking + campers_afternoon_rowing = 71 :=
by
  -- We are skipping the proof since instructions specify only the statement is needed
  sorry

end total_campers_rowing_and_hiking_l238_238284


namespace solve_fraction_eq_zero_l238_238751

theorem solve_fraction_eq_zero (x : ℝ) (h : x ≠ 0) : 
  (x^2 - 4*x + 3) / (5*x) = 0 ↔ (x = 1 ∨ x = 3) :=
by
  sorry

end solve_fraction_eq_zero_l238_238751


namespace percentage_of_60_l238_238439

theorem percentage_of_60 (x : ℝ) : 
  (0.2 * 40) + (x / 100) * 60 = 23 → x = 25 :=
by
  sorry

end percentage_of_60_l238_238439


namespace part1_part2_l238_238886

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1 (x : ℝ) : 
  ∀ x, f x 2 ≥ 4 ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
by sorry

theorem part2 (x a : ℝ) :
  f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
by sorry

end part1_part2_l238_238886


namespace Deepak_age_l238_238296

-- Define the current ages of Arun and Deepak
variable (A D : ℕ)

-- Define the conditions
def ratio_condition := A / D = 4 / 3
def future_age_condition := A + 6 = 26

-- Define the proof statement
theorem Deepak_age (h1 : ratio_condition A D) (h2 : future_age_condition A) : D = 15 :=
  sorry

end Deepak_age_l238_238296


namespace part1_solution_set_part2_range_of_a_l238_238876

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1_solution_set (x : ℝ) : 
  (f x 2 ≥ 4) ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
sorry

theorem part2_range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 4) → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end part1_solution_set_part2_range_of_a_l238_238876


namespace part_1_solution_set_part_2_a_range_l238_238889

-- Define the function f
def f (x a : ℝ) := |x - a^2| + |x - 2 * a + 1|

-- Part (1)
theorem part_1_solution_set (x : ℝ) : {x | x ≤ 3 / 2 ∨ x ≥ 11 / 2} = 
  {x | f x 2 ≥ 4} :=
sorry

-- Part (2)
theorem part_2_a_range (a : ℝ) : 
  {a | (a - 1)^2 ≥ 4} = {a | a ≤ -1 ∨ a ≥ 3} :=
sorry

end part_1_solution_set_part_2_a_range_l238_238889


namespace num_divisors_1215_l238_238129

theorem num_divisors_1215 : (Finset.filter (λ d => 1215 % d = 0) (Finset.range (1215 + 1))).card = 12 :=
by
  sorry

end num_divisors_1215_l238_238129


namespace part1_solution_set_part2_range_a_l238_238897

def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

theorem part1_solution_set (x : ℝ) (h : f x 2 ≥ 4) : 
  x ≤ 3/2 ∨ x ≥ 11/2 :=
sorry

theorem part2_range_a (a : ℝ) (h : ∃ x, f x a ≥ 4) : 
  a ≤ -1 ∨ a ≥ 3 :=
sorry

end part1_solution_set_part2_range_a_l238_238897


namespace intercept_form_impossible_values_l238_238673

-- Define the problem statement
theorem intercept_form_impossible_values (m : ℝ) :
  (¬ (∃ a b c : ℝ, m ≠ 0 ∧ a * m = 0 ∧ b * m = 0 ∧ c * m = 1) ↔ (m = 4 ∨ m = -3 ∨ m = 5)) :=
sorry

end intercept_form_impossible_values_l238_238673


namespace seventh_triangular_number_is_28_l238_238258

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem seventh_triangular_number_is_28 : triangular_number 7 = 28 :=
by
  /- proof goes here -/
  sorry

end seventh_triangular_number_is_28_l238_238258


namespace greatest_possible_sum_of_consecutive_integers_product_less_500_l238_238723

theorem greatest_possible_sum_of_consecutive_integers_product_less_500 :
  ∃ (n : ℤ), (n * (n + 1) < 500) ∧ (∀ (m : ℤ), (m * (m + 1) < 500) → (n + (n + 1) ≥ m + (m + 1))) :=
begin
  sorry
end

end greatest_possible_sum_of_consecutive_integers_product_less_500_l238_238723


namespace rick_bought_30_guppies_l238_238654

theorem rick_bought_30_guppies (G : ℕ) (T C : ℕ) 
  (h1 : T = 4 * C) 
  (h2 : C = 2 * G) 
  (h3 : G + C + T = 330) : 
  G = 30 := 
by 
  sorry

end rick_bought_30_guppies_l238_238654


namespace part1_part2_l238_238885

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1 (x : ℝ) : 
  ∀ x, f x 2 ≥ 4 ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
by sorry

theorem part2 (x a : ℝ) :
  f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
by sorry

end part1_part2_l238_238885


namespace intersection_of_lines_l238_238135

theorem intersection_of_lines :
  ∃ (x y : ℚ), (6 * x - 5 * y = 15) ∧ (8 * x + 3 * y = 1) ∧ x = 25 / 29 ∧ y = -57 / 29 :=
by
  sorry

end intersection_of_lines_l238_238135


namespace value_of_x_l238_238595

/-
Given the following conditions:
  x = a + 7,
  a = b + 9,
  b = c + 15,
  c = d + 25,
  d = 60,
Prove that x = 116.
-/

theorem value_of_x (a b c d x : ℤ) 
    (h1 : x = a + 7)
    (h2 : a = b + 9)
    (h3 : b = c + 15)
    (h4 : c = d + 25)
    (h5 : d = 60) : x = 116 := 
  sorry

end value_of_x_l238_238595


namespace part1_solution_set_part2_range_of_a_l238_238933

-- Defining the function f(x) under given conditions
def f (x a : ℝ) : ℝ := |x - a^2| + |x - (2 * a + 1)|

-- Part 1: Determine the solution set for the inequality when a = 2
theorem part1_solution_set (x : ℝ) : (f x 2) ≥ 4 ↔ x ≤ 3/2 ∨ x ≥ 11/2 := by
  sorry

-- Part 2: Determine the range of values for a given the inequality
theorem part2_range_of_a (a : ℝ) : (∀ x : ℝ, f x a ≥ 4) ↔ a ≤ -1 ∨ a ≥ 3 := by
  sorry

end part1_solution_set_part2_range_of_a_l238_238933


namespace min_value_abs_function_l238_238255

theorem min_value_abs_function : ∀ (x : ℝ), (|x + 1| + |2 - x|) ≥ 3 :=
by
  sorry

end min_value_abs_function_l238_238255


namespace part1_solution_set_part2_range_of_a_l238_238912

-- Define the function f for a general a
def f (x a : ℝ) : ℝ := |x - a^2| + |x - (2 * a) + 1|

-- Part 1: When a = 2
theorem part1_solution_set (x : ℝ) (h : f x 2 ≥ 4) : x ≤ 3/2 ∨ x ≥ 11/2 :=
  sorry

-- Part 2: Range of values for a
theorem part2_range_of_a (a : ℝ) (h : ∀ x, f x a ≥ 4) : a ≤ -1 ∨ a ≥ 3 :=
  sorry

end part1_solution_set_part2_range_of_a_l238_238912


namespace part1_solution_set_part2_values_of_a_l238_238862

-- Parameters and definitions for the function and conditions
def f (x a : ℝ) := |x - a^2| + |x - (2*a + 1)|

-- Part (1) When a = 2, the inequality's solution set
theorem part1_solution_set (x : ℝ) : f x 2 ≥ 4 ↔ (x ≤ 3/2 ∨ x ≥ 11/2) := 
sorry

-- Part (2) Range of values for a given f(x, a) ≥ 4
theorem part2_values_of_a (a : ℝ) : 
∀ x, f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end part1_solution_set_part2_values_of_a_l238_238862


namespace roster_representation_of_M_l238_238386

def M : Set ℚ := {x | ∃ m n : ℤ, x = m / n ∧ |m| < 2 ∧ 1 ≤ n ∧ n ≤ 3}

theorem roster_representation_of_M :
  M = {-1, -1/2, -1/3, 0, 1/2, 1/3} :=
by sorry

end roster_representation_of_M_l238_238386


namespace greatest_sum_of_consecutive_integers_product_lt_500_l238_238731

theorem greatest_sum_of_consecutive_integers_product_lt_500 : 
  ∃ (n : ℕ), n * (n + 1) < 500 ∧ ∀ (m : ℕ), m * (m + 1) < 500 → m ≤ n → n + (n + 1) = 43 := 
by
  use 21
  split
  {
    norm_num
    linarith
  }
  {
    intros m h_hint h_ineq
    have : m ≤ 21, sorry
    linarith
  }
  sorry

end greatest_sum_of_consecutive_integers_product_lt_500_l238_238731


namespace sum_of_ages_l238_238979

theorem sum_of_ages (M C : ℝ) (h1 : M = C + 12) (h2 : M + 10 = 3 * (C - 6)) : M + C = 52 :=
by
  sorry

end sum_of_ages_l238_238979


namespace find_xy_l238_238303

noncomputable def star (a b c d : ℝ) : ℝ × ℝ :=
  (a * c + b * d, a * d + b * c)

theorem find_xy (a b x y : ℝ) (h : star a b x y = (a, b)) (h' : a^2 ≠ b^2) : (x, y) = (1, 0) :=
  sorry

end find_xy_l238_238303


namespace original_number_is_twenty_l238_238771

theorem original_number_is_twenty (x : ℕ) (h : 100 * x = x + 1980) : x = 20 :=
sorry

end original_number_is_twenty_l238_238771


namespace dice_sum_prob_10_l238_238173

/-- A standard six-faced die has faces numbered 1 through 6 --/
def is_standard_die_face (n : ℕ) : Prop := n ∈ {1, 2, 3, 4, 5, 6}

/-- The probability of the sum of the numbers rolled on three standard six-faced dice being 10 --/
theorem dice_sum_prob_10 : 
  ∃ (count_10 : ℕ), ∃ (total : ℕ), 
  (total = 6^3) ∧ 
  (count_10 = 27) ∧ 
  (count_10.toRat / total.toRat = 27 / 216) := 
by 
  sorry

end dice_sum_prob_10_l238_238173


namespace distance_A_to_plane_BCD_l238_238764

noncomputable theory
open Real

/-- Define the points A, B, C, D in Euclidean space -/
def A := (0, 0, 0) : ℝ × ℝ × ℝ
def B := (1, 0, 0) : ℝ × ℝ × ℝ
def C := (0, 2, 0) : ℝ × ℝ × ℝ
def D := (0, 0, 3) : ℝ × ℝ × ℝ

/-- Define a helper function to compute the distance from a point to a plane defined by three points -/
def point_to_plane_distance (p : ℝ × ℝ × ℝ) (a b c : ℝ × ℝ × ℝ) : ℝ :=
  let n := ((b.1 - a.1) * (c.2 - a.2) - (c.1 - a.1) * (b.2 - a.2), 
            (b.2 - a.2) * (c.3 - a.3) - (c.2 - a.2) * (b.3 - a.3), 
            (b.3 - a.3) * (c.1 - a.1) - (c.3 - a.3) * (b.1 - a.1)) in
  abs ((p.1 - a.1) * n.1 + (p.2 - a.2) * n.2 + (p.3 - a.3) * n.3) / sqrt (n.1^2 + n.2^2 + n.3^2)

/-- formal statement to be proved: distance from A to the plane containing B, C, D is 6/7 -/
theorem distance_A_to_plane_BCD : 
  point_to_plane_distance A B C D = 6 / 7 :=
by sorry

end distance_A_to_plane_BCD_l238_238764


namespace part1_solution_set_part2_range_of_a_l238_238928

noncomputable def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1_solution_set (a : ℝ) (a_eq_2 : a = 2) : 
  {x : ℝ | f x a ≥ 4} = {x | x ≤ 3 / 2} ∪ {x | x ≥ 11 / 2} :=
by
  sorry

theorem part2_range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 4) → (a ∈ set.Iic (-1) ∪ set.Ici 3) :=
by
  sorry

end part1_solution_set_part2_range_of_a_l238_238928


namespace mickey_horses_per_week_l238_238120

def days_in_week : ℕ := 7

def horses_minnie_per_day : ℕ := days_in_week + 3

def horses_twice_minnie_per_day : ℕ := 2 * horses_minnie_per_day

def horses_mickey_per_day : ℕ := horses_twice_minnie_per_day - 6

def horses_mickey_per_week : ℕ := days_in_week * horses_mickey_per_day

theorem mickey_horses_per_week : horses_mickey_per_week = 98 := sorry

end mickey_horses_per_week_l238_238120


namespace functional_equation_implies_identity_l238_238388

theorem functional_equation_implies_identity 
  (f : ℝ → ℝ) 
  (hf : ∀ x y : ℝ, 0 < x → 0 < y → 
    f ((x + y) / 2) + f ((2 * x * y) / (x + y)) = f x + f y) 
  (x y : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  : 2 * f (Real.sqrt (x * y)) = f x + f y := sorry

end functional_equation_implies_identity_l238_238388


namespace complex_coordinate_l238_238476

theorem complex_coordinate (i : ℂ) (h : i * i = -1) : i * (1 - i) = 1 + i :=
by sorry

end complex_coordinate_l238_238476


namespace focus_of_parabola_x_squared_eq_4y_is_0_1_l238_238664

theorem focus_of_parabola_x_squared_eq_4y_is_0_1 :
  ∃ (x y : ℝ), (0, 1) = (x, y) ∧ (∀ a b : ℝ, a^2 = 4 * b → (x, y) = (0, 1)) :=
sorry

end focus_of_parabola_x_squared_eq_4y_is_0_1_l238_238664


namespace intersection_of_A_and_complement_B_l238_238485

def A : Set ℝ := {1, 2, 3, 4, 5}
def B : Set ℝ := {x | x < 3}
def complement_B : Set ℝ := {x | x ≥ 3}

theorem intersection_of_A_and_complement_B : A ∩ complement_B = {3, 4, 5} :=
by
  sorry

end intersection_of_A_and_complement_B_l238_238485


namespace solve_eq_l238_238633

theorem solve_eq (x : ℝ) (h : 2 - 1 / (2 - x) = 1 / (2 - x)) : x = 1 := 
sorry

end solve_eq_l238_238633


namespace max_problems_solved_l238_238459

theorem max_problems_solved (D : Fin 7 -> ℕ) :
  (∀ i, D i ≤ 10) →
  (∀ i, i < 5 → D i > 7 → D (i + 1) ≤ 5 ∧ D (i + 2) ≤ 5) →
  (∑ i, D i <= 50) →
  ∀ D, ∑ i, D i ≤ 50 :=
by
  intros h1 h2
  sorry

end max_problems_solved_l238_238459


namespace radius_of_base_of_cone_correct_l238_238958

noncomputable def radius_of_base_of_cone (n : ℕ) (r α : ℝ) : ℝ :=
  r * (1 / Real.sin (Real.pi / n) - 1 / Real.tan (Real.pi / 4 + α / 2))

theorem radius_of_base_of_cone_correct :
  radius_of_base_of_cone 11 3 (Real.pi / 6) = 3 / Real.sin (Real.pi / 11) - Real.sqrt 3 :=
by
  sorry

end radius_of_base_of_cone_correct_l238_238958


namespace Julio_limes_expense_l238_238642

/-- Julio's expense on limes after 30 days --/
theorem Julio_limes_expense :
  ((30 * (1 / 2)) / 3) * 1 = 5 := 
by
  sorry

end Julio_limes_expense_l238_238642


namespace angle_bisector_eqn_l238_238768

-- Define the vertices A, B, and C
def A : (ℝ × ℝ) := (4, 3)
def B : (ℝ × ℝ) := (-4, -1)
def C : (ℝ × ℝ) := (9, -7)

-- State the theorem with conditions and the given answer
theorem angle_bisector_eqn (A B C : (ℝ × ℝ)) (hA : A = (4, 3)) (hB : B = (-4, -1)) (hC : C = (9, -7)) :
  ∃ b c, (3:ℝ) * (3:ℝ) - b * (3:ℝ) + c = 0 ∧ b + c = -6 := 
by 
  use -1, -5
  simp
  sorry

end angle_bisector_eqn_l238_238768


namespace resulting_expression_l238_238226

def x : ℕ := 1000
def y : ℕ := 10

theorem resulting_expression : 
  (x + 2 * y) + x + 3 * y + x + 4 * y + x + y = 4 * x + 10 * y :=
by
  sorry

end resulting_expression_l238_238226


namespace gcd_lcm_product_l238_238446

theorem gcd_lcm_product (a b : ℕ) (h₁ : a = 24) (h₂ : b = 36) :
  Nat.gcd a b * Nat.lcm a b = 864 := 
by
  rw [h₁, h₂]
  -- You can include specific calculation just to express the idea
  -- rw [Nat.gcd_comm, Nat.gcd_rec]
  -- rw [Nat.lcm_def]
  -- rw [Nat.mul_subst]
  sorry

end gcd_lcm_product_l238_238446


namespace sin_x_lt_a_l238_238845

theorem sin_x_lt_a (a θ : ℝ) (h1 : -1 < a) (h2 : a < 0) (hθ : θ = Real.arcsin a) :
  {x : ℝ | ∃ n : ℤ, (2 * n - 1) * Real.pi - θ < x ∧ x < 2 * n * Real.pi + θ} = {x : ℝ | Real.sin x < a} :=
sorry

end sin_x_lt_a_l238_238845


namespace harper_water_intake_l238_238490

theorem harper_water_intake
  (cases_cost : ℕ := 12)
  (cases_count : ℕ := 24)
  (total_spent : ℕ)
  (days : ℕ)
  (total_days_spent : ℕ := 240)
  (total_money_spent: ℕ := 60)
  (total_water: ℕ := 5 * 24)
  (water_per_day : ℝ := 0.5):
  total_spent = total_money_spent ->
  days = total_days_spent ->
  water_per_day = (total_water : ℝ) / total_days_spent :=
by
  sorry

end harper_water_intake_l238_238490


namespace fraction_to_decimal_l238_238019

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l238_238019


namespace part1_solution_set_part2_range_a_l238_238901

def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

theorem part1_solution_set (x : ℝ) (h : f x 2 ≥ 4) : 
  x ≤ 3/2 ∨ x ≥ 11/2 :=
sorry

theorem part2_range_a (a : ℝ) (h : ∃ x, f x a ≥ 4) : 
  a ≤ -1 ∨ a ≥ 3 :=
sorry

end part1_solution_set_part2_range_a_l238_238901


namespace profit_percentage_l238_238572

theorem profit_percentage (CP SP : ℝ) (h1 : CP = 500) (h2 : SP = 650) : 
  (SP - CP) / CP * 100 = 30 :=
by
  sorry

end profit_percentage_l238_238572


namespace fraction_to_decimal_l238_238038

theorem fraction_to_decimal : (7 : ℝ) / 16 = 0.4375 := 
by
  sorry

end fraction_to_decimal_l238_238038


namespace Andy_more_white_socks_than_black_l238_238580

def num_black_socks : ℕ := 6
def initial_num_white_socks : ℕ := 4 * num_black_socks
def final_num_white_socks : ℕ := initial_num_white_socks / 2
def more_white_than_black : ℕ := final_num_white_socks - num_black_socks

theorem Andy_more_white_socks_than_black :
  more_white_than_black = 6 :=
sorry

end Andy_more_white_socks_than_black_l238_238580


namespace shirts_total_cost_l238_238379

def shirt_cost_problem : Prop :=
  ∃ (first_shirt_cost second_shirt_cost total_cost : ℕ),
    first_shirt_cost = 15 ∧
    first_shirt_cost = second_shirt_cost + 6 ∧
    total_cost = first_shirt_cost + second_shirt_cost ∧
    total_cost = 24

theorem shirts_total_cost : shirt_cost_problem := by
  sorry

end shirts_total_cost_l238_238379


namespace inequality_of_pos_real_product_l238_238853

theorem inequality_of_pos_real_product
  (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x * y * z = 1) :
  (x^3 / ((1 + y) * (1 + z)) + y^3 / ((1 + z) * (1 + x)) + z^3 / ((1 + x) * (1 + y))) ≥ (3 / 4) :=
sorry

end inequality_of_pos_real_product_l238_238853


namespace fraction_to_decimal_l238_238031

theorem fraction_to_decimal : (7 : ℝ) / 16 = 0.4375 := 
by
  sorry

end fraction_to_decimal_l238_238031


namespace part1_solution_set_part2_range_of_a_l238_238923

def f (x a : ℝ) := |x - a^2| + |x - (2*a + 1)|

-- Part (1)
theorem part1_solution_set (x : ℝ) :
  ∀ x, f x 2 ≥ 4 ↔ x ≤ 3 / 2 ∨ x ≥ 11 / 2 := sorry

-- Part (2)
theorem part2_range_of_a (a : ℝ) :
  (∀ x, f x a ≥ 4) ↔ (a ≤ -1 ∨ a ≥ 3) := sorry

end part1_solution_set_part2_range_of_a_l238_238923


namespace sum_of_squares_divisibility_l238_238649

theorem sum_of_squares_divisibility
  (p : ℕ) (hp : Nat.Prime p)
  (x y z : ℕ)
  (hx : 0 < x) (hxy : x < y) (hyz : y < z) (hzp : z < p)
  (hmod_eq : ∀ a b c : ℕ, a^3 % p = b^3 % p → b^3 % p = c^3 % p → a^3 % p = c^3 % p) :
  (x^2 + y^2 + z^2) % (x + y + z) = 0 := by
  sorry

end sum_of_squares_divisibility_l238_238649


namespace fraction_to_decimal_l238_238017

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l238_238017


namespace fraction_to_decimal_l238_238023

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l238_238023


namespace no_real_solutions_l238_238305

theorem no_real_solutions : ∀ (x y : ℝ), ¬ (3 * x^2 + y^2 - 9 * x - 6 * y + 23 = 0) :=
by sorry

end no_real_solutions_l238_238305


namespace quotient_in_first_division_l238_238236

theorem quotient_in_first_division (N Q Q' : ℕ) (h₁ : N = 68 * Q) (h₂ : N % 67 = 1) : Q = 1 :=
by
  -- rest of the proof goes here
  sorry

end quotient_in_first_division_l238_238236


namespace Andy_more_white_socks_than_black_l238_238581

def num_black_socks : ℕ := 6
def initial_num_white_socks : ℕ := 4 * num_black_socks
def final_num_white_socks : ℕ := initial_num_white_socks / 2
def more_white_than_black : ℕ := final_num_white_socks - num_black_socks

theorem Andy_more_white_socks_than_black :
  more_white_than_black = 6 :=
sorry

end Andy_more_white_socks_than_black_l238_238581


namespace find_constant_k_l238_238964

theorem find_constant_k (S : ℕ → ℝ) (a : ℕ → ℝ) (k : ℝ)
  (h₁ : ∀ n, S n = 3 * 2^n + k)
  (h₂ : ∀ n, 1 ≤ n → a n = S n - S (n - 1))
  (h₃ : ∃ q, ∀ n, 1 ≤ n → a (n + 1) = a n * q ) :
  k = -3 := 
sorry

end find_constant_k_l238_238964


namespace fraction_to_decimal_l238_238078

theorem fraction_to_decimal : (7 : ℚ) / 16 = 4375 / 10000 := by
  sorry

end fraction_to_decimal_l238_238078


namespace baez_marble_loss_l238_238114

theorem baez_marble_loss :
  ∃ p : ℚ, (p > 0 ∧ (p / 100) * 25 * 2 = 60) ∧ p = 20 :=
by
  sorry

end baez_marble_loss_l238_238114


namespace mickey_horses_per_week_l238_238126

variable (days_in_week : ℕ := 7)
variable (minnie_horses_per_day : ℕ := days_in_week + 3)
variable (mickey_horses_per_day : ℕ := 2 * minnie_horses_per_day - 6)

theorem mickey_horses_per_week : mickey_horses_per_day * days_in_week = 98 := by
  sorry

end mickey_horses_per_week_l238_238126


namespace correct_propositions_identification_l238_238345

theorem correct_propositions_identification (x y : ℝ) (h1 : x ≥ 0 ∧ y ≥ 0 → x * y ≥ 0)
    (h2 : ¬(x * y ≥ 0 → x ≥ 0 ∧ y ≥ 0))
    (h3 : ¬(¬(x ≥ 0 ∧ y ≥ 0 → x * y ≥ 0)))
    (h4 : (¬(x * y ≥ 0) → ¬(x ≥ 0) ∨ ¬(y ≥ 0))) :
  true :=
by
  -- Proof skipped
  sorry

end correct_propositions_identification_l238_238345


namespace Jeffs_donuts_l238_238510

theorem Jeffs_donuts (D : ℕ) (h1 : ∀ n, n = 12 * D - 20) (h2 : n = 100) : D = 10 :=
by
  sorry

end Jeffs_donuts_l238_238510


namespace expression_simplification_l238_238779

theorem expression_simplification : (2^2020 + 2^2018) / (2^2020 - 2^2018) = 5 / 3 := 
by 
    sorry

end expression_simplification_l238_238779


namespace common_difference_is_1_l238_238961

variable (a_2 a_5 : ℕ) (d : ℤ)

def arithmetic_sequence (n a_1 : ℤ) (d : ℤ) : ℤ := a_1 + (n - 1) * d

theorem common_difference_is_1 
  (h1 : arithmetic_sequence 2 a_1 d = 3) 
  (h2 : arithmetic_sequence 5 a_1 d = 6) : 
  d = 1 := 
sorry

end common_difference_is_1_l238_238961


namespace cost_of_scooter_l238_238229

-- Given conditions
variables (M T : ℕ)
axiom h1 : T = M + 4
axiom h2 : T = 15

-- Proof goal: The cost of the scooter is $26
theorem cost_of_scooter : M + T = 26 :=
by sorry

end cost_of_scooter_l238_238229


namespace circle_diameter_l238_238249

theorem circle_diameter (A : ℝ) (π : ℝ) (r : ℝ) (d : ℝ) (h1 : A = 64 * π) (h2 : A = π * r^2) (h3 : d = 2 * r) :
  d = 16 :=
by
  sorry

end circle_diameter_l238_238249


namespace all_numbers_equal_l238_238612

theorem all_numbers_equal 
  (x : Fin 2007 → ℝ)
  (h : ∀ (I : Finset (Fin 2007)), I.card = 7 → ∃ (J : Finset (Fin 2007)), J.card = 11 ∧ 
  (1 / 7 : ℝ) * I.sum x = (1 / 11 : ℝ) * J.sum x) :
  ∃ c : ℝ, ∀ i : Fin 2007, x i = c :=
by sorry

end all_numbers_equal_l238_238612


namespace integers_a_b_c_d_arbitrarily_large_l238_238986

theorem integers_a_b_c_d_arbitrarily_large (n : ℤ) : 
  ∃ (a b c d : ℤ), (a^2 + b^2 + c^2 + d^2 = a * b * c + a * b * d + a * c * d + b * c * d) ∧ 
    min (min a b) (min c d) ≥ n := 
by sorry

end integers_a_b_c_d_arbitrarily_large_l238_238986


namespace Q_contribution_l238_238981

def P_contribution : ℕ := 4000
def P_months : ℕ := 12
def Q_months : ℕ := 8
def profit_ratio_PQ : ℚ := 2 / 3

theorem Q_contribution :
  ∃ X : ℕ, (P_contribution * P_months) / (X * Q_months) = profit_ratio_PQ → X = 9000 := 
by sorry

end Q_contribution_l238_238981


namespace probability_sum_is_10_l238_238191

theorem probability_sum_is_10 (d1 d2 d3 : ℕ) : 
  (1 ≤ d1 ∧ d1 ≤ 6) ∧ (1 ≤ d2 ∧ d2 ≤ 6) ∧ (1 ≤ d3 ∧ d3 ≤ 6) ∧ (d1 + d2 + d3 = 10) → 
  (((d1 = 1 ∧ d2 = 3 ∧ d3 = 6) ∨ 
    (d1 = 1 ∧ d2 = 4 ∧ d3 = 5) ∨ 
    (d1 = 2 ∧ d2 = 3 ∧ d3 = 5) ∨ 
    (d1 = 2 ∧ d2 = 4 ∧ d3 = 4) ∨ 
    (d1 = 3 ∧ d2 = 3 ∧ d3 = 4) ∨ 
    (d1 = 4 ∧ d2 = 4 ∧ d3 = 2)) 
    ↔ nat.cast (1 / 9)) :=
begin
  sorry,
end

end probability_sum_is_10_l238_238191


namespace cube_property_l238_238417

theorem cube_property (x : ℝ) (s : ℝ) 
  (h1 : s^3 = 8 * x)
  (h2 : 6 * s^2 = 4 * x) :
  x = 5400 :=
by
  sorry

end cube_property_l238_238417


namespace fraction_to_decimal_l238_238040

theorem fraction_to_decimal :
  (7 / 16 : ℝ) = 0.4375 := 
begin
  -- placeholder for the proof
  sorry
end

end fraction_to_decimal_l238_238040


namespace triangle_side_lengths_l238_238947

theorem triangle_side_lengths {x : ℤ} (h₁ : x + 4 > 10) (h₂ : x + 10 > 4) (h₃ : 10 + 4 > x) :
  ∃ (n : ℕ), n = 7 :=
by
  sorry

end triangle_side_lengths_l238_238947


namespace fraction_to_decimal_l238_238043

theorem fraction_to_decimal :
  (7 / 16 : ℝ) = 0.4375 := 
begin
  -- placeholder for the proof
  sorry
end

end fraction_to_decimal_l238_238043


namespace friend_jogging_time_l238_238949

theorem friend_jogging_time (D : ℝ) (my_time : ℝ) (friend_speed : ℝ) :
  my_time = 3 * 60 →
  friend_speed = 2 * (D / my_time) →
  (D / friend_speed) = 90 :=
by
  sorry

end friend_jogging_time_l238_238949


namespace solve_inequality_smallest_integer_solution_l238_238393

theorem solve_inequality (x : ℝ) : 
    (9 * x + 8) / 6 - x / 3 ≥ -1 ↔ x ≥ -2 := 
sorry

theorem smallest_integer_solution :
    ∃ (x : ℤ), (∃ (y : ℝ) (h₁ : y = x), 
    (9 * y + 8) / 6 - y / 3 ≥ -1) ∧ 
    ∀ (z : ℤ), ((∃ (w : ℝ) (h₂ : w = z), 
    (9 * w + 8) / 6 - w / 3 ≥ -1) → -2 ≤ z) :=
    ⟨-2, __, sorry⟩

end solve_inequality_smallest_integer_solution_l238_238393


namespace students_taking_geometry_or_science_but_not_both_l238_238312

def students_taking_both : ℕ := 15
def students_taking_geometry : ℕ := 30
def students_taking_science_only : ℕ := 18

theorem students_taking_geometry_or_science_but_not_both : students_taking_geometry - students_taking_both + students_taking_science_only = 33 := by
  sorry

end students_taking_geometry_or_science_but_not_both_l238_238312


namespace find_q_l238_238963

variable {m n q : ℝ}

theorem find_q (h1 : m = 3 * n + 5) (h2 : m + 2 = 3 * (n + q) + 5) : q = 2 / 3 := by
  sorry

end find_q_l238_238963


namespace fraction_decimal_equivalent_l238_238007

theorem fraction_decimal_equivalent : (7 / 16 : ℝ) = 0.4375 :=
by
  sorry

end fraction_decimal_equivalent_l238_238007


namespace zara_goats_l238_238756

noncomputable def total_animals_per_group := 48
noncomputable def total_groups := 3
noncomputable def total_cows := 24
noncomputable def total_sheep := 7

theorem zara_goats : 
  (total_groups * total_animals_per_group = 144) ∧ 
  (144 = total_cows + total_sheep + 113) →
  113 = 144 - total_cows - total_sheep := 
by sorry

end zara_goats_l238_238756


namespace number_of_birds_l238_238548

-- Conditions
def geese : ℕ := 58
def ducks : ℕ := 37

-- Proof problem statement
theorem number_of_birds : geese + ducks = 95 :=
by
  -- The actual proof is to be provided
  sorry

end number_of_birds_l238_238548


namespace time_to_cross_first_platform_l238_238437

noncomputable def train_length : ℝ := 30
noncomputable def first_platform_length : ℝ := 180
noncomputable def second_platform_length : ℝ := 250
noncomputable def time_second_platform : ℝ := 20

noncomputable def train_speed : ℝ :=
(train_length + second_platform_length) / time_second_platform

noncomputable def time_first_platform : ℝ :=
(train_length + first_platform_length) / train_speed

theorem time_to_cross_first_platform :
  time_first_platform = 15 :=
by
  sorry

end time_to_cross_first_platform_l238_238437


namespace shorter_leg_right_triangle_l238_238503

theorem shorter_leg_right_triangle (a b c : ℕ) (h0 : a^2 + b^2 = c^2) (h1 : c = 39) (h2 : a < b) : a = 15 :=
by {
  sorry
}

end shorter_leg_right_triangle_l238_238503


namespace number_of_correct_statements_l238_238611

noncomputable def f (x : ℝ) : ℝ := 1 / 2 * Real.sin (2 * x)

def statement_1 : Prop := ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = 2 * Real.pi
def statement_2 : Prop := ∀ x y, -Real.pi / 4 ≤ x ∧ x ≤ y ∧ y ≤ Real.pi / 4 → f x ≤ f y
def statement_3 : Prop := ∀ y, -Real.pi / 6 ≤ y ∧ y ≤ Real.pi / 3 → -Real.sqrt 3 / 4 ≤ f y ∧ f y ≤ Real.sqrt 3 / 4
def statement_4 : Prop := ∀ x, f x = (1 / 2 * Real.sin (2 * x + Real.pi / 4) - Real.pi / 8)

theorem number_of_correct_statements : 
  (¬ statement_1 ∧ statement_2 ∧ ¬ statement_3 ∧ ¬ statement_4) = true :=
sorry

end number_of_correct_statements_l238_238611


namespace greatest_sum_of_consecutive_integers_product_lt_500_l238_238729

theorem greatest_sum_of_consecutive_integers_product_lt_500 : 
  ∃ (n : ℕ), n * (n + 1) < 500 ∧ ∀ (m : ℕ), m * (m + 1) < 500 → m ≤ n → n + (n + 1) = 43 := 
by
  use 21
  split
  {
    norm_num
    linarith
  }
  {
    intros m h_hint h_ineq
    have : m ≤ 21, sorry
    linarith
  }
  sorry

end greatest_sum_of_consecutive_integers_product_lt_500_l238_238729


namespace bobby_gasoline_left_l238_238778

theorem bobby_gasoline_left
  (initial_gasoline : ℕ) (supermarket_distance : ℕ) 
  (travel_distance : ℕ) (turn_back_distance : ℕ)
  (trip_fuel_efficiency : ℕ) : 
  initial_gasoline = 12 →
  supermarket_distance = 5 →
  travel_distance = 6 →
  turn_back_distance = 2 →
  trip_fuel_efficiency = 2 →
  ∃ remaining_gasoline,
    remaining_gasoline = initial_gasoline - 
    ((supermarket_distance * 2 + 
    turn_back_distance * 2 + 
    travel_distance) / trip_fuel_efficiency) ∧ 
    remaining_gasoline = 2 :=
by sorry

end bobby_gasoline_left_l238_238778


namespace find_a_l238_238482

theorem find_a (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 2*x - 4*y = 0 → 3*x + y + a = 0) →
  a = 1 :=
by
  sorry

end find_a_l238_238482


namespace part1_part2_l238_238879

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1 (x : ℝ) : 
  ∀ x, f x 2 ≥ 4 ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
by sorry

theorem part2 (x a : ℝ) :
  f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
by sorry

end part1_part2_l238_238879


namespace find_sum_of_squares_l238_238480

-- Define the variables x and y as real numbers
variables (x y : ℝ)

-- Given conditions
def condition1 : Prop := x + y = 12
def condition2 : Prop := x * y = 50

-- The statement we need to prove
theorem find_sum_of_squares (h1 : condition1 x y) (h2 : condition2 x y) : x^2 + y^2 = 44 := by
  sorry

end find_sum_of_squares_l238_238480


namespace union_M_N_l238_238515

def M := {x : ℝ | -2 < x ∧ x < -1}
def N := {x : ℝ | (1 / 2 : ℝ)^x ≤ 4}

theorem union_M_N :
  M ∪ N = {x : ℝ | x ≥ -2} :=
sorry

end union_M_N_l238_238515


namespace greatest_sum_consecutive_integers_product_less_than_500_l238_238700

theorem greatest_sum_consecutive_integers_product_less_than_500 :
  ∃ n : ℕ, n * (n + 1) < 500 ∧ (n + (n + 1) = 43) := sorry

end greatest_sum_consecutive_integers_product_less_than_500_l238_238700


namespace part1_solution_set_part2_range_a_l238_238902

def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

theorem part1_solution_set (x : ℝ) (h : f x 2 ≥ 4) : 
  x ≤ 3/2 ∨ x ≥ 11/2 :=
sorry

theorem part2_range_a (a : ℝ) (h : ∃ x, f x a ≥ 4) : 
  a ≤ -1 ∨ a ≥ 3 :=
sorry

end part1_solution_set_part2_range_a_l238_238902


namespace sum_of_factors_of_30_l238_238272

open Nat

theorem sum_of_factors_of_30 : 
  ∑ d in (Finset.filter (λ n, 30 % n = 0) (Finset.range (30 + 1))), d = 72 := by
  sorry

end sum_of_factors_of_30_l238_238272


namespace gcd_polynomials_l238_238396

-- Given condition: a is an even multiple of 1009
def is_even_multiple_of_1009 (a : ℤ) : Prop :=
  ∃ k : ℤ, a = 2 * 1009 * k

-- Statement: gcd(2a^2 + 31a + 58, a + 15) = 1
theorem gcd_polynomials (a : ℤ) (ha : is_even_multiple_of_1009 a) :
  gcd (2 * a^2 + 31 * a + 58) (a + 15) = 1 := 
sorry

end gcd_polynomials_l238_238396


namespace count_difference_l238_238292

-- Given definitions
def count_six_digit_numbers_in_ascending_order_by_digits : ℕ := by
  -- Calculation using binomial coefficient
  exact Nat.choose 9 6

def count_six_digit_numbers_with_one : ℕ := by
  -- Calculation using binomial coefficient with fixed '1' in one position
  exact Nat.choose 8 5

def count_six_digit_numbers_without_one : ℕ := by
  -- Calculation subtracting with and without 1
  exact count_six_digit_numbers_in_ascending_order_by_digits - count_six_digit_numbers_with_one

-- Theorem to prove
theorem count_difference : 
  count_six_digit_numbers_with_one - count_six_digit_numbers_without_one = 28 :=
by
  sorry

end count_difference_l238_238292


namespace padic_zeros_l238_238973

variable {p : ℕ} (hp : p > 1)
variable {a : ℕ} (hnz : a % p ≠ 0)

theorem padic_zeros (k : ℕ) (hk : k ≥ 1) :
  (a^(p^(k-1)*(p-1)) - 1) % (p^k) = 0 :=
sorry

end padic_zeros_l238_238973


namespace percentage_needed_to_pass_l238_238576

-- Definitions for conditions
def obtained_marks : ℕ := 125
def failed_by : ℕ := 40
def total_marks : ℕ := 500
def passing_marks := obtained_marks + failed_by

-- Assertion to prove
theorem percentage_needed_to_pass : (passing_marks : ℕ) * 100 / total_marks = 33 := by
  sorry

end percentage_needed_to_pass_l238_238576


namespace percentage_markup_l238_238259

theorem percentage_markup (sell_price : ℝ) (cost_price : ℝ)
  (h_sell : sell_price = 8450) (h_cost : cost_price = 6500) : 
  (sell_price - cost_price) / cost_price * 100 = 30 :=
by
  sorry

end percentage_markup_l238_238259


namespace cos_300_eq_half_l238_238820

theorem cos_300_eq_half : Real.cos (300 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end cos_300_eq_half_l238_238820


namespace quadratic_unique_solution_k_neg_l238_238609

theorem quadratic_unique_solution_k_neg (k : ℝ) :
  (∃ x : ℝ, 9 * x^2 + k * x + 36 = 0 ∧ ∀ y : ℝ, 9 * y^2 + k * y + 36 = 0 → y = x) →
  k = -36 :=
by
  sorry

end quadratic_unique_solution_k_neg_l238_238609


namespace greatest_sum_of_consecutive_integers_product_lt_500_l238_238742

theorem greatest_sum_of_consecutive_integers_product_lt_500 : 
  ∃ (n : ℤ), n * (n + 1) < 500 ∧ (∀ m : ℤ, m * (m + 1) < 500 → m + (m + 1) ≤ n + (n + 1)) ∧ n + (n + 1) = 43 :=
begin
  sorry
end

end greatest_sum_of_consecutive_integers_product_lt_500_l238_238742


namespace part1_solution_set_part2_range_of_a_l238_238916

def f (x a : ℝ) := |x - a^2| + |x - (2*a + 1)|

-- Part (1)
theorem part1_solution_set (x : ℝ) :
  ∀ x, f x 2 ≥ 4 ↔ x ≤ 3 / 2 ∨ x ≥ 11 / 2 := sorry

-- Part (2)
theorem part2_range_of_a (a : ℝ) :
  (∀ x, f x a ≥ 4) ↔ (a ≤ -1 ∨ a ≥ 3) := sorry

end part1_solution_set_part2_range_of_a_l238_238916


namespace lawnmower_percentage_drop_l238_238297

theorem lawnmower_percentage_drop :
  ∀ (initial_value value_after_one_year value_after_six_months : ℝ)
    (percentage_drop_in_year : ℝ),
  initial_value = 100 →
  value_after_one_year = 60 →
  percentage_drop_in_year = 20 →
  value_after_one_year = (1 - percentage_drop_in_year / 100) * value_after_six_months →
  (initial_value - value_after_six_months) / initial_value * 100 = 25 :=
by
  intros initial_value value_after_one_year value_after_six_months percentage_drop_in_year
  intros h_initial h_value_after_one_year h_percentage_drop_in_year h_value_equation
  sorry

end lawnmower_percentage_drop_l238_238297


namespace faye_pencils_allocation_l238_238606

theorem faye_pencils_allocation (pencils total_pencils rows : ℕ) (h_pencils : total_pencils = 6) (h_rows : rows = 2) (h_allocation : pencils = total_pencils / rows) : pencils = 3 := by
  sorry

end faye_pencils_allocation_l238_238606


namespace cos_300_eq_half_l238_238804

theorem cos_300_eq_half :
  let θ := 300
  let ref_θ := θ - 360 * (θ // 360) -- Reference angle in the proper range
  θ = 300 ∧ ref_θ = 60 ∧ cos 60 = 1/2 ∧ 300 > 270 → cos 300 = 1/2 :=
by
  intros
  sorry

end cos_300_eq_half_l238_238804


namespace fraction_to_decimal_l238_238083

theorem fraction_to_decimal : (7 : ℚ) / 16 = 4375 / 10000 := by
  sorry

end fraction_to_decimal_l238_238083


namespace fraction_to_decimal_l238_238029

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l238_238029


namespace ways_to_distribute_books_into_bags_l238_238587

theorem ways_to_distribute_books_into_bags : 
  let books := 5
  let bags := 4
  ∃ (ways : ℕ), ways = 41 := 
sorry

end ways_to_distribute_books_into_bags_l238_238587


namespace fraction_to_decimal_l238_238080

theorem fraction_to_decimal : (7 : ℚ) / 16 = 4375 / 10000 := by
  sorry

end fraction_to_decimal_l238_238080


namespace min_value_expr_l238_238650

theorem min_value_expr (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0) (h_xyz : x * y * z = 1) : 
  x^2 + 4 * x * y + 9 * y^2 + 8 * y * z + 3 * z^2 ≥ 9^(10/9) :=
sorry

end min_value_expr_l238_238650


namespace cos_300_eq_half_l238_238803

theorem cos_300_eq_half :
  let θ := 300
  let ref_θ := θ - 360 * (θ // 360) -- Reference angle in the proper range
  θ = 300 ∧ ref_θ = 60 ∧ cos 60 = 1/2 ∧ 300 > 270 → cos 300 = 1/2 :=
by
  intros
  sorry

end cos_300_eq_half_l238_238803


namespace ratio_boysGradeA_girlsGradeB_l238_238362

variable (S G B : ℕ)

-- Given conditions
axiom h1 : (1 / 3 : ℚ) * G = (1 / 4 : ℚ) * S
axiom h2 : S = B + G

-- Definitions based on conditions
def boys_in_GradeA (B : ℕ) := (2 / 5 : ℚ) * B
def girls_in_GradeB (G : ℕ) := (3 / 5 : ℚ) * G

-- The proof goal
theorem ratio_boysGradeA_girlsGradeB (S G B : ℕ) (h1 : (1 / 3 : ℚ) * G = (1 / 4 : ℚ) * S) (h2 : S = B + G) :
    boys_in_GradeA B / girls_in_GradeB G = 2 / 9 :=
by
  sorry

end ratio_boysGradeA_girlsGradeB_l238_238362


namespace solve_color_problem_l238_238638

variables (R B G C : Prop)

def color_problem (R B G C : Prop) : Prop :=
  (C → (R ∨ B)) ∧ (¬C → (¬R ∧ ¬G)) ∧ ((B ∨ G) → C) → C ∧ (R ∨ B)

theorem solve_color_problem (R B G C : Prop) (h : (C → (R ∨ B)) ∧ (¬C → (¬R ∧ ¬G)) ∧ ((B ∨ G) → C)) : C ∧ (R ∨ B) :=
  by {
    sorry
  }

end solve_color_problem_l238_238638


namespace dihedral_angle_between_planes_l238_238256

noncomputable def normal_vector_plane_alpha : EuclideanSpace ℝ (Fin 3) := ![1,0,-1]
noncomputable def normal_vector_plane_beta : EuclideanSpace ℝ (Fin 3) := ![0,-1,1]

def cosine_of_angle (m n : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  (inner_product_space.inner m n) / (norm m * norm n)

theorem dihedral_angle_between_planes :
  let m := normal_vector_plane_alpha
      n := normal_vector_plane_beta in
  ∃ θ : ℝ, (θ = real.acos (cosine_of_angle m n) ∨ θ = π - real.acos (cosine_of_angle m n))
          → θ = π / 3 ∨ θ = 2 * π / 3 :=
sorry

end dihedral_angle_between_planes_l238_238256


namespace arithmetic_sequence_middle_term_l238_238217

theorem arithmetic_sequence_middle_term 
  (a b c d e : ℕ) 
  (h_seq : a = 23 ∧ e = 53 ∧ (b - a = c - b) ∧ (c - b = d - c) ∧ (d - c = e - d)) :
  c = 38 :=
by
  sorry

end arithmetic_sequence_middle_term_l238_238217


namespace determine_values_of_a_and_b_l238_238478

def ab_product_eq_one (a b : ℝ) : Prop := a * b = 1

def given_equation (a b : ℝ) : Prop :=
  (a + b + 2) / 4 = (1 / (a + 1)) + (1 / (b + 1))

theorem determine_values_of_a_and_b (a b : ℝ) (h1 : ab_product_eq_one a b) (h2 : given_equation a b) :
  a = 1 ∧ b = 1 :=
by
  sorry

end determine_values_of_a_and_b_l238_238478


namespace greatest_sum_of_consecutive_integers_l238_238707

def consecutiveSumCondition (n : ℤ) : Prop :=
  n * (n + 1) < 500 

theorem greatest_sum_of_consecutive_integers : 
  ∃ n : ℤ, consecutiveSumCondition n ∧ ∀ m : ℤ, consecutiveSumCondition m → n + (n + 1) ≥ m + (m + 1) :=
sorry

end greatest_sum_of_consecutive_integers_l238_238707


namespace minimum_participants_l238_238260

theorem minimum_participants
  (correct_first : ℕ)
  (correct_second : ℕ)
  (correct_third : ℕ)
  (correct_fourth : ℕ)
  (H_first : correct_first = 90)
  (H_second : correct_second = 50)
  (H_third : correct_third = 40)
  (H_fourth : correct_fourth = 20)
  (H_max_two : ∀ p : ℕ, 1 ≤ p ∧ p ≤ correct_first + correct_second + correct_third + correct_fourth → p ≤ 2 * (correct_first + correct_second + correct_third + correct_fourth))
  : ∃ n : ℕ, (correct_first + correct_second + correct_third + correct_fourth) / 2 = 100 :=
by
  sorry

end minimum_participants_l238_238260


namespace smallest_value_of_3b_plus_2_l238_238351

theorem smallest_value_of_3b_plus_2 (b : ℝ) (h : 8 * b^2 + 7 * b + 6 = 5) : (∃ t : ℝ, t = 3 * b + 2 ∧ (∀ x : ℝ, 8 * x^2 + 7 * x + 6 = 5 → x = b → t ≤ 3 * x + 2)) :=
sorry

end smallest_value_of_3b_plus_2_l238_238351


namespace part_1_solution_set_part_2_a_range_l238_238896

-- Define the function f
def f (x a : ℝ) := |x - a^2| + |x - 2 * a + 1|

-- Part (1)
theorem part_1_solution_set (x : ℝ) : {x | x ≤ 3 / 2 ∨ x ≥ 11 / 2} = 
  {x | f x 2 ≥ 4} :=
sorry

-- Part (2)
theorem part_2_a_range (a : ℝ) : 
  {a | (a - 1)^2 ≥ 4} = {a | a ≤ -1 ∨ a ≥ 3} :=
sorry

end part_1_solution_set_part_2_a_range_l238_238896


namespace find_Y_l238_238266

theorem find_Y 
  (a b c d X Y : ℕ)
  (h1 : a + b + c + d = 40)
  (h2 : X + Y + c + b = 40)
  (h3 : a + b + X = 30)
  (h4 : c + d + Y = 30)
  (h5 : X = 9) 
  : Y = 11 := 
by 
  sorry

end find_Y_l238_238266


namespace dice_sum_probability_l238_238196

theorem dice_sum_probability : (∃ s, (∃ x y z : ℕ, 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 ∧ 1 ≤ z ∧ z ≤ 6 ∧ x + y + z = 10)
  ∧ s = {n : ℕ // ∃ x y z : ℕ, 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 ∧ 1 ≤ z ∧ z ≤ 6 ∧ x + y + z = n} ∧ s.card = 24)
  → 24 / 216 = 1 / 9 := 
sorry

end dice_sum_probability_l238_238196


namespace eq_relation_q_r_l238_238368

-- Define the angles in the context of the problem
variables {A B C D E F : Type}
variables {angle_BAC angle_BFD angle_ADE angle_FEC : ℝ}
variables (right_triangle_ABC : A → B → C → angle_BAC = 90)

-- Equilateral triangle DEF inscribed in ABC
variables (inscribed_equilateral_DEF : D → E → F)
variables (angle_BFD_eq_p : ∀ p : ℝ, angle_BFD = p)
variables (angle_ADE_eq_q : ∀ q : ℝ, angle_ADE = q)
variables (angle_FEC_eq_r : ∀ r : ℝ, angle_FEC = r)

-- Main statement to be proved
theorem eq_relation_q_r {p q r : ℝ} 
  (right_triangle_ABC : angle_BAC = 90)
  (angle_BFD : angle_BFD = 30 + q)
  (angle_FEC : angle_FEC = 120 - r) :
  q + r = 60 :=
sorry

end eq_relation_q_r_l238_238368


namespace horizontal_distance_travelled_l238_238109

theorem horizontal_distance_travelled (r : ℝ) (θ : ℝ) (d : ℝ)
  (h_r : r = 2) (h_θ : θ = Real.pi / 6) :
  d = 2 * Real.sqrt 3 * Real.pi := sorry

end horizontal_distance_travelled_l238_238109


namespace sum_of_roots_l238_238746

theorem sum_of_roots (a b c : ℝ) (x1 x2 x3 : ℝ) (h_eq: 6*x1^3 + 7*x2^2 - 12*x3 = 0) :
  (x1 + x2 + x3) = -1.17 :=
sorry

end sum_of_roots_l238_238746


namespace julios_spending_on_limes_l238_238641

theorem julios_spending_on_limes 
    (days : ℕ) (lime_juice_per_day : ℕ) (lime_juice_per_lime : ℕ) (limes_per_dollar : ℕ) 
    (total_spending : ℝ) 
    (h1 : days = 30) 
    (h2 : lime_juice_per_day = 1) 
    (h3 : lime_juice_per_lime = 2) 
    (h4 : limes_per_dollar = 3) 
    (h5 : total_spending = 5) :
    let lime_juice_needed := days * lime_juice_per_day,
        total_limes := lime_juice_needed / lime_juice_per_lime,
        cost := (total_limes / limes_per_dollar : ℕ) in
    (cost : ℝ) = total_spending := 
by 
    sorry

end julios_spending_on_limes_l238_238641


namespace Mickey_horses_per_week_l238_238124

-- Definitions based on the conditions
def days_in_week : Nat := 7
def Minnie_mounts_per_day : Nat := days_in_week + 3 
def Mickey_mounts_per_day : Nat := 2 * Minnie_mounts_per_day - 6
def Mickey_mounts_per_week : Nat := Mickey_mounts_per_day * days_in_week

-- Theorem statement
theorem Mickey_horses_per_week : Mickey_mounts_per_week = 98 :=
by
  sorry

end Mickey_horses_per_week_l238_238124


namespace Mike_onions_grew_l238_238234

-- Define the data:
variables (nancy_onions dan_onions total_onions mike_onions : ℕ)

-- Conditions:
axiom Nancy_onions_grew : nancy_onions = 2
axiom Dan_onions_grew : dan_onions = 9
axiom Total_onions_grew : total_onions = 15

-- Theorem to prove:
theorem Mike_onions_grew (h : total_onions = nancy_onions + dan_onions + mike_onions) : mike_onions = 4 :=
by
  -- The proof is not provided, so we use sorry:
  sorry

end Mike_onions_grew_l238_238234


namespace initial_pipes_num_l238_238241

variable {n : ℕ}

theorem initial_pipes_num (h1 : ∀ t : ℕ, (n * t = 8) → n = 3) (h2 : ∀ t : ℕ, (2 * t = 12) → n = 3) : n = 3 := 
by 
  sorry

end initial_pipes_num_l238_238241


namespace condition_A_sufficient_not_necessary_condition_B_l238_238330

theorem condition_A_sufficient_not_necessary_condition_B {a b : ℝ} (hA : a > 1 ∧ b > 1) : 
  (a + b > 2 ∧ ab > 1) ∧ ¬∀ a b, (a + b > 2 ∧ ab > 1) → (a > 1 ∧ b > 1) :=
by
  sorry

end condition_A_sufficient_not_necessary_condition_B_l238_238330


namespace product_of_ratios_l238_238680

theorem product_of_ratios:
  ∀ (x1 y1 x2 y2 x3 y3 : ℝ),
    (x1^3 - 3 * x1 * y1^2 = 2023) ∧ (y1^3 - 3 * x1^2 * y1 = 2022) →
    (x2^3 - 3 * x2 * y2^2 = 2023) ∧ (y2^3 - 3 * x2^2 * y2 = 2022) →
    (x3^3 - 3 * x3 * y3^2 = 2023) ∧ (y3^3 - 3 * x3^2 * y3 = 2022) →
    (1 - x1/y1) * (1 - x2/y2) * (1 - x3/y3) = 1 / 2023 :=
by
  intros x1 y1 x2 y2 x3 y3
  sorry

end product_of_ratios_l238_238680


namespace dice_sum_probability_l238_238200

theorem dice_sum_probability : 
  let die_faces := {1, 2, 3, 4, 5, 6};
      successful_outcomes := { (x, y, z) | x ∈ die_faces ∧ y ∈ die_faces ∧ z ∈ die_faces ∧ x + y + z = 10 }.toFinset.card;
      total_outcomes := (finset.range 6).card ^ 3
  in  successful_outcomes / total_outcomes = 1 / 9 :=
by
  let die_faces := {1, 2, 3, 4, 5, 6};
  let successful_outcomes := { (x, y, z) | x ∈ die_faces ∧ y ∈ die_faces ∧ z ∈ die_faces ∧ x + y + z = 10 }.toFinset.card;
  let total_outcomes := (finset.range 6).card ^ 3;
  sorry

end dice_sum_probability_l238_238200


namespace chord_length_l238_238669

theorem chord_length (x y t : ℝ) (h₁ : x = 1 + 2 * t) (h₂ : y = 2 + t) (h_circle : x^2 + y^2 = 9) : 
  ∃ l, l = 12 / 5 * Real.sqrt 5 := 
sorry

end chord_length_l238_238669


namespace part1_solution_set_part2_range_of_a_l238_238877

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1_solution_set (x : ℝ) : 
  (f x 2 ≥ 4) ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
sorry

theorem part2_range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 4) → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end part1_solution_set_part2_range_of_a_l238_238877


namespace part1_part2_l238_238882

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1 (x : ℝ) : 
  ∀ x, f x 2 ≥ 4 ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
by sorry

theorem part2 (x a : ℝ) :
  f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
by sorry

end part1_part2_l238_238882


namespace part1_solution_set_part2_range_of_a_l238_238939

-- Defining the function f(x) under given conditions
def f (x a : ℝ) : ℝ := |x - a^2| + |x - (2 * a + 1)|

-- Part 1: Determine the solution set for the inequality when a = 2
theorem part1_solution_set (x : ℝ) : (f x 2) ≥ 4 ↔ x ≤ 3/2 ∨ x ≥ 11/2 := by
  sorry

-- Part 2: Determine the range of values for a given the inequality
theorem part2_range_of_a (a : ℝ) : (∀ x : ℝ, f x a ≥ 4) ↔ a ≤ -1 ∨ a ≥ 3 := by
  sorry

end part1_solution_set_part2_range_of_a_l238_238939


namespace math_problem_l238_238353

theorem math_problem (x y : Int)
  (hx : x = 2 - 4 + 6)
  (hy : y = 1 - 3 + 5) :
  x - y = 1 :=
by
  sorry

end math_problem_l238_238353


namespace Emily_total_cost_l238_238598

theorem Emily_total_cost :
  let cost_curtains := 2 * 30
  let cost_prints := 9 * 15
  let installation_cost := 50
  let total_cost := cost_curtains + cost_prints + installation_cost
  total_cost = 245 := by
{
 sorry
}

end Emily_total_cost_l238_238598


namespace probability_heads_equals_l238_238086

theorem probability_heads_equals (p q: ℚ) (h1 : q = 1 - p) (h2 : (binomial 10 5) * p^5 * q^5 = (binomial 10 6) * p^6 * q^4) : p = 6 / 11 :=
by {
  sorry
}

end probability_heads_equals_l238_238086


namespace sin_cos_sixth_l238_238972

theorem sin_cos_sixth (θ : ℝ) (h : Real.sin (2 * θ) = 1 / 3) :
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 11 / 12 :=
sorry

end sin_cos_sixth_l238_238972


namespace sufficient_but_not_necessary_condition_l238_238332

variable {α : Type*} (A B : Set α)

theorem sufficient_but_not_necessary_condition (h₁ : A ∩ B = A) (h₂ : A ≠ B) :
  (∀ x, x ∈ A → x ∈ B) ∧ ¬(∀ x, x ∈ B → x ∈ A) :=
by 
  sorry

end sufficient_but_not_necessary_condition_l238_238332


namespace probability_sum_is_ten_l238_238179

structure Die :=
(value : ℕ)
(h_value : 1 ≤ value ∧ value ≤ 6)

def possible_rolls : List (ℕ × ℕ × ℕ) :=
List.product 
  (List.product [1, 2, 3, 4, 5, 6] [1, 2, 3, 4, 5, 6]) [1, 2, 3, 4, 5, 6]
|>.map (λ ((a, b), c) => (a, b, c))

def favorable_outcomes : List (ℕ × ℕ × ℕ) :=
possible_rolls.filter (λ (x, y, z) => x + y + z = 10)

def probability (n favorable : ℕ) : ℚ := favourable / n

theorem probability_sum_is_ten :
  probability possible_rolls.length favourable_outcomes.length = 1 / 9 :=
by
  sorry

end probability_sum_is_ten_l238_238179


namespace nested_radical_eq_6_l238_238604

theorem nested_radical_eq_6 (x : ℝ) (h : x = Real.sqrt (18 + x)) : x = 6 :=
by 
  have h_eq : x^2 = 18 + x,
  { rw h, exact pow_two (Real.sqrt (18 + x)) },
  have quad_eq : x^2 - x - 18 = 0,
  { linarith [h_eq] },
  have factored : (x - 6) * (x + 3) = x^2 - x - 18,
  { ring },
  rw [←quad_eq, factored] at h,
  sorry

end nested_radical_eq_6_l238_238604


namespace midpoint_coordinates_l238_238147

theorem midpoint_coordinates (xM yM xN yN : ℝ) (hM : xM = 3) (hM' : yM = -2) (hN : xN = -1) (hN' : yN = 0) :
  (xM + xN) / 2 = 1 ∧ (yM + yN) / 2 = -1 :=
by
  simp [hM, hM', hN, hN']
  sorry

end midpoint_coordinates_l238_238147


namespace find_x_l238_238337

-- Define the operation "※" as given
def star (a b : ℕ) : ℚ := (a + 2 * b) / 3

-- Given that 6 ※ x = 22 / 3, prove that x = 8
theorem find_x : ∃ x : ℕ, star 6 x = 22 / 3 ↔ x = 8 :=
by
  sorry -- Proof not required

end find_x_l238_238337


namespace total_time_correct_l238_238279

-- Define the base speeds and distance
def speed_boat : ℕ := 8
def speed_stream : ℕ := 6
def distance : ℕ := 210

-- Define the speeds downstream and upstream
def speed_downstream : ℕ := speed_boat + speed_stream
def speed_upstream : ℕ := speed_boat - speed_stream

-- Define the time taken for downstream and upstream
def time_downstream : ℕ := distance / speed_downstream
def time_upstream : ℕ := distance / speed_upstream

-- Define the total time taken
def total_time : ℕ := time_downstream + time_upstream

-- The theorem to be proven
theorem total_time_correct : total_time = 120 := by
  sorry

end total_time_correct_l238_238279


namespace two_trains_cross_time_l238_238281

/-- Definition for the two trains' parameters -/
structure Train :=
  (length : ℝ)  -- length in meters
  (speed : ℝ)  -- speed in km/hr

/-- The parameters of Train 1 and Train 2 -/
def train1 : Train := { length := 140, speed := 60 }
def train2 : Train := { length := 160, speed := 40 }

noncomputable def relative_speed_mps (t1 t2 : Train) : ℝ :=
  (t1.speed + t2.speed) * (5 / 18)

noncomputable def total_length (t1 t2 : Train) : ℝ :=
  t1.length + t2.length

noncomputable def time_to_cross (t1 t2 : Train) : ℝ :=
  total_length t1 t2 / relative_speed_mps t1 t2

theorem two_trains_cross_time :
  time_to_cross train1 train2 = 10.8 := by
  sorry

end two_trains_cross_time_l238_238281


namespace part1_solution_set_part2_range_of_a_l238_238872

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1_solution_set (x : ℝ) : 
  (f x 2 ≥ 4) ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
sorry

theorem part2_range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 4) → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end part1_solution_set_part2_range_of_a_l238_238872


namespace lilly_fish_l238_238652

-- Define the conditions
def total_fish : ℕ := 18
def rosy_fish : ℕ := 8

-- Statement: Prove that Lilly has 10 fish
theorem lilly_fish (h1 : total_fish = 18) (h2 : rosy_fish = 8) :
  total_fish - rosy_fish = 10 :=
by sorry

end lilly_fish_l238_238652


namespace part1_solution_set_part2_range_a_l238_238905

def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

theorem part1_solution_set (x : ℝ) (h : f x 2 ≥ 4) : 
  x ≤ 3/2 ∨ x ≥ 11/2 :=
sorry

theorem part2_range_a (a : ℝ) (h : ∃ x, f x a ≥ 4) : 
  a ≤ -1 ∨ a ≥ 3 :=
sorry

end part1_solution_set_part2_range_a_l238_238905


namespace find_m_l238_238477

theorem find_m (m : ℝ) (x : ℝ) (h : x = 1) (h_eq : (m / (2 - x)) - (1 / (x - 2)) = 3) : m = 2 :=
sorry

end find_m_l238_238477


namespace range_a_l238_238630

-- Conditions
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + Real.log x
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 1 / x

-- Theorem to prove the range of a
theorem range_a (a : ℝ) : (∀ x > 0, f' a x → ℝ → ∃ c : ℝ, a < 0) :=
sorry

end range_a_l238_238630


namespace solve_for_k_l238_238624

variable {x y k : ℝ}

theorem solve_for_k (h1 : 2 * x + y = 1) (h2 : x + 2 * y = k - 2) (h3 : x - y = 2) : k = 1 :=
by
  sorry

end solve_for_k_l238_238624


namespace first_term_geometric_sequence_l238_238593

theorem first_term_geometric_sequence (a r : ℕ) (h₁ : a * r^5 = 32) (h₂ : r = 2) : a = 1 := by
  sorry

end first_term_geometric_sequence_l238_238593


namespace total_clouds_counted_l238_238118

def clouds_counted (carson_clouds : ℕ) (brother_factor : ℕ) : ℕ :=
  carson_clouds + (carson_clouds * brother_factor)

theorem total_clouds_counted (carson_clouds brother_factor total_clouds : ℕ) 
  (h₁ : carson_clouds = 6) (h₂ : brother_factor = 3) (h₃ : total_clouds = 24) :
  clouds_counted carson_clouds brother_factor = total_clouds :=
by
  sorry

end total_clouds_counted_l238_238118


namespace new_solution_percentage_l238_238391

theorem new_solution_percentage 
  (initial_weight : ℝ) (evaporated_water : ℝ) (added_solution_weight : ℝ) 
  (percentage_X : ℝ) (percentage_water : ℝ)
  (total_initial_X : ℝ := initial_weight * percentage_X)
  (initial_water : ℝ := initial_weight * percentage_water)
  (post_evaporation_weight : ℝ := initial_weight - evaporated_water)
  (post_evaporation_X : ℝ := total_initial_X)
  (post_evaporation_water : ℝ := post_evaporation_weight - total_initial_X)
  (added_X : ℝ := added_solution_weight * percentage_X)
  (added_water : ℝ := added_solution_weight * percentage_water)
  (total_X : ℝ := post_evaporation_X + added_X)
  (total_water : ℝ := post_evaporation_water + added_water)
  (new_total_weight : ℝ := post_evaporation_weight + added_solution_weight) :
  (total_X / new_total_weight) * 100 = 41.25 := 
by {
  sorry
}

end new_solution_percentage_l238_238391


namespace solve_equations_l238_238527

theorem solve_equations :
  (∀ x, x^2 - 4 = 0 ↔ x = 2 ∨ x = -2) ∧
  (∀ x, x^2 - 6 * x + 9 = 0 ↔ x = 3) ∧
  (∀ x, x^2 - 7 * x + 12 = 0 ↔ x = 3 ∨ x = 4) ∧
  (∀ x, 2 * x^2 - 3 * x - 5 = 0 ↔ x = 5 / 2 ∨ x = -1) :=
by
  -- Proof goes here
  sorry

end solve_equations_l238_238527


namespace probability_sum_is_10_l238_238212

theorem probability_sum_is_10 : 
  let outcomes := [(a, b, c) | a ∈ [1, 2, 3, 4, 5, 6], b ∈ [1, 2, 3, 4, 5, 6], c ∈ [1, 2, 3, 4, 5, 6]],
      favorable_outcomes := [(a, b, c) | (a, b, c) ∈ outcomes, a + b + c = 10] in
  (favorable_outcomes.length : ℚ) / outcomes.length = 1 / 9 := by
  sorry

end probability_sum_is_10_l238_238212


namespace area_not_covered_by_smaller_squares_l238_238105

-- Define the conditions given in the problem
def side_length_larger_square : ℕ := 10
def side_length_smaller_square : ℕ := 4
def area_of_larger_square : ℕ := side_length_larger_square * side_length_larger_square
def area_of_each_smaller_square : ℕ := side_length_smaller_square * side_length_smaller_square

-- Define the total area of the two smaller squares
def total_area_smaller_squares : ℕ := area_of_each_smaller_square * 2

-- Define the uncovered area
def uncovered_area : ℕ := area_of_larger_square - total_area_smaller_squares

-- State the theorem to prove
theorem area_not_covered_by_smaller_squares :
  uncovered_area = 68 := by
  -- Placeholder for the actual proof
  sorry

end area_not_covered_by_smaller_squares_l238_238105


namespace exists_unequal_m_n_l238_238154

theorem exists_unequal_m_n (a b c : ℕ → ℕ) :
  ∃ (m n : ℕ), m ≠ n ∧ a m ≥ a n ∧ b m ≥ b n ∧ c m ≥ c n :=
sorry

end exists_unequal_m_n_l238_238154


namespace probability_sum_10_l238_238203

def is_valid_roll (d1 d2 d3 : ℕ) : Prop :=
  1 ≤ d1 ∧ d1 ≤ 6 ∧ 1 ≤ d2 ∧ d2 ≤ 6 ∧ 1 ≤ d3 ∧ d3 ≤ 6

def sum_is_10 (d1 d2 d3 : ℕ) : Prop :=
  d1 + d2 + d3 = 10

def valid_rolls_count : ℕ :=
  216 -- 6^3 distinct rolls of three 6-sided dice

def successful_rolls_count : ℕ :=
  24 -- number of valid rolls that sum to 10

theorem probability_sum_10 :
  (successful_rolls_count : ℚ) / valid_rolls_count = 1 / 9 := by
  sorry

end probability_sum_10_l238_238203


namespace fraction_to_decimal_l238_238047

theorem fraction_to_decimal :
  (7 / 16 : ℝ) = 0.4375 := 
begin
  -- placeholder for the proof
  sorry
end

end fraction_to_decimal_l238_238047


namespace find_x_l238_238489

noncomputable def vector_a : ℝ × ℝ × ℝ := (-3, 2, 5)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ × ℝ := (1, x, -1)

theorem find_x (x : ℝ) (h : (-3) * 1 + 2 * x + 5 * (-1) = 2) : x = 5 :=
by 
  sorry

end find_x_l238_238489


namespace number_appears_n_times_in_grid_l238_238848

theorem number_appears_n_times_in_grid (n : ℕ) (G : Fin n → Fin n → ℤ) :
  (∀ i j : Fin n, abs (G i j - G i.succ j) ≤ 1 ∧ abs (G i j - G i j.succ) ≤ 1) →
  ∃ x : ℤ, (∃ count, count ≥ n ∧ (∀ i j : Fin n, G i j = x → count = count + 1)) :=
by sorry

end number_appears_n_times_in_grid_l238_238848


namespace problem_I_II_l238_238857

noncomputable def f (x : ℝ) (φ : ℝ) := Real.sin (2 * x + φ)

theorem problem_I_II
  (φ : ℝ) (h1 : 0 < φ) (h2 : φ < Real.pi / 2)
  (h3 : f 0 φ = 1 / 2) :
  ∃ T : ℝ, T = Real.pi ∧ φ = Real.pi / 6 ∧ 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 → (f x φ) ≥ -1 / 2) :=
by
  sorry

end problem_I_II_l238_238857


namespace eggs_leftover_l238_238578

theorem eggs_leftover (eggs_abigail eggs_beatrice eggs_carson cartons : ℕ)
  (h_abigail : eggs_abigail = 37)
  (h_beatrice : eggs_beatrice = 49)
  (h_carson : eggs_carson = 14)
  (h_cartons : cartons = 12) :
  ((eggs_abigail + eggs_beatrice + eggs_carson) % cartons) = 4 :=
by
  sorry

end eggs_leftover_l238_238578


namespace part1_part2_l238_238343

def f (x : ℝ) : ℝ := abs (x - 5) + abs (x + 4)

theorem part1 (x : ℝ) : f x ≥ 12 ↔ x ≥ 13 / 2 ∨ x ≤ -11 / 2 :=
by
    sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x - 2 ^ (1 - 3 * a) - 1 ≥ 0) ↔ -2 / 3 ≤ a :=
by
    sorry

end part1_part2_l238_238343


namespace exactly_one_box_empty_count_l238_238141

-- Define the setting with four different balls and four boxes.
def numberOfWaysExactlyOneBoxEmpty (balls : Finset ℕ) (boxes : Finset ℕ) : ℕ :=
  if (balls.card = 4 ∧ boxes.card = 4) then
     Nat.choose 4 2 * Nat.factorial 3
  else 0

theorem exactly_one_box_empty_count :
  numberOfWaysExactlyOneBoxEmpty {1, 2, 3, 4} {1, 2, 3, 4} = 144 :=
by
  -- The proof is omitted
  sorry

end exactly_one_box_empty_count_l238_238141


namespace frac_f_ratio_interval_l238_238432

theorem frac_f_ratio_interval (f : ℝ → ℝ) (h1 : ∀ x > 0, f x > 0)
  (h2 : ∀ x > 0, 2 * f x < x * deriv f x ∧ x * deriv f x < 3 * f x) :
  1 / 8 < f 1 / f 2 ∧ f 1 / f 2 < 1 / 4 :=
sorry

end frac_f_ratio_interval_l238_238432


namespace perpendicular_lines_a_value_l238_238360

theorem perpendicular_lines_a_value :
  ∀ (a : ℝ), (∀ x y : ℝ, 2 * x - y = 0) -> (∀ x y : ℝ, a * x - 2 * y - 1 = 0) ->    
  (∀ m1 m2 : ℝ, m1 = 2 -> m2 = a / 2 -> m1 * m2 = -1) -> a = -1 :=
sorry

end perpendicular_lines_a_value_l238_238360


namespace value_of_a5_max_sum_first_n_value_l238_238329

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

def sum_first_n (a d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

theorem value_of_a5 (a d a5 : ℤ) :
  a5 = 4 ↔ (2 * a + 4 * d) + (a + 4 * d) + (a + 8 * d) = (a + 5 * d) + 8 :=
  sorry

theorem max_sum_first_n_value (a d : ℤ) (n : ℕ) (max_n : ℕ) :
  a = 16 →
  d = -3 →
  (∀ i, sum_first_n a d i ≤ sum_first_n a d max_n) →
  max_n = 6 :=
  sorry

end value_of_a5_max_sum_first_n_value_l238_238329


namespace sum_of_roots_eq_neg_five_l238_238852

theorem sum_of_roots_eq_neg_five (x₁ x₂ : ℝ) (h₁ : x₁^2 + 5 * x₁ - 2 = 0) (h₂ : x₂^2 + 5 * x₂ - 2 = 0) (h_distinct : x₁ ≠ x₂) :
  x₁ + x₂ = -5 := sorry

end sum_of_roots_eq_neg_five_l238_238852


namespace greatest_sum_of_consecutive_integers_product_lt_500_l238_238741

theorem greatest_sum_of_consecutive_integers_product_lt_500 : 
  ∃ (n : ℤ), n * (n + 1) < 500 ∧ (∀ m : ℤ, m * (m + 1) < 500 → m + (m + 1) ≤ n + (n + 1)) ∧ n + (n + 1) = 43 :=
begin
  sorry
end

end greatest_sum_of_consecutive_integers_product_lt_500_l238_238741


namespace dice_sum_prob_10_l238_238175

/-- A standard six-faced die has faces numbered 1 through 6 --/
def is_standard_die_face (n : ℕ) : Prop := n ∈ {1, 2, 3, 4, 5, 6}

/-- The probability of the sum of the numbers rolled on three standard six-faced dice being 10 --/
theorem dice_sum_prob_10 : 
  ∃ (count_10 : ℕ), ∃ (total : ℕ), 
  (total = 6^3) ∧ 
  (count_10 = 27) ∧ 
  (count_10.toRat / total.toRat = 27 / 216) := 
by 
  sorry

end dice_sum_prob_10_l238_238175


namespace smallest_number_of_students_l238_238585

-- Define the conditions as given in the problem
def eight_to_six_ratio : ℕ × ℕ := (5, 3) -- ratio of 8th-graders to 6th-graders
def eight_to_nine_ratio : ℕ × ℕ := (7, 4) -- ratio of 8th-graders to 9th-graders

theorem smallest_number_of_students (a b c : ℕ)
  (h1 : a = 5 * b) (h2 : b = 3 * c) (h3 : a = 7 * c) : a + b + c = 76 := 
sorry

end smallest_number_of_students_l238_238585


namespace largest_angle_of_consecutive_interior_angles_pentagon_l238_238998

theorem largest_angle_of_consecutive_interior_angles_pentagon (x : ℕ)
  (h1 : (x - 3) + (x - 2) + (x - 1) + x + (x + 1) = 540) :
  x + 1 = 110 := sorry

end largest_angle_of_consecutive_interior_angles_pentagon_l238_238998


namespace find_principal_sum_l238_238758

noncomputable def principal_sum (P R : ℝ) : ℝ := P * (R + 6) / 100 - P * R / 100

theorem find_principal_sum (P R : ℝ) (h : P * (R + 6) / 100 - P * R / 100 = 30) : P = 500 :=
by sorry

end find_principal_sum_l238_238758


namespace cubing_identity_l238_238162

theorem cubing_identity (x : ℝ) (hx : x + 1/x = -3) : x^3 + 1/x^3 = -18 := by
  sorry

end cubing_identity_l238_238162


namespace abs_diff_squares_l238_238690

theorem abs_diff_squares (a b : ℤ) (h_a : a = 105) (h_b : b = 95):
  |a^2 - b^2| = 2000 := by
  sorry

end abs_diff_squares_l238_238690


namespace total_weight_of_courtney_marble_collection_l238_238454

def marble_weight_first_jar : ℝ := 80 * 0.35
def marble_weight_second_jar : ℝ := 160 * 0.45
def marble_weight_third_jar : ℝ := 20 * 0.25

/-- The total weight of Courtney's marble collection -/
theorem total_weight_of_courtney_marble_collection :
    marble_weight_first_jar + marble_weight_second_jar + marble_weight_third_jar = 105 := by
  sorry

end total_weight_of_courtney_marble_collection_l238_238454


namespace coefficient_x5_in_expansion_l238_238838

noncomputable def binom : ℕ → ℕ → ℕ := λ n k, Nat.choose n k

theorem coefficient_x5_in_expansion :
  let f1 := (1 + x + x^2)
  let f2 := (1 - x)^(10)
  let expansion := f1 * f2 in
  (expansion.coeff 5) = -162 :=
by
  sorry

end coefficient_x5_in_expansion_l238_238838


namespace smallest_tangent_circle_eqn_l238_238465

noncomputable def center_curve (x : ℝ) : ℝ := -(3 / x)

noncomputable def point_to_line_distance (a b c x y : ℝ) : ℝ :=
  abs (a * x + b * y + c) / (real.sqrt (a^2 + b^2))

theorem smallest_tangent_circle_eqn :
  ∃ x y r : ℝ, 
    (y = center_curve x) ∧
    (3 * x - 4 * y + 3 = 0) ∧
    r = 3 ∧
    ((x - 2) ^ 2 + (y + 3 / 2) ^ 2 = r ^ 2) :=
by
  sorry

end smallest_tangent_circle_eqn_l238_238465


namespace dice_sum_10_probability_l238_238169

open Probability

noncomputable def probability_sum_10 : ℚ := 
  let total_outcomes := 216
  let favorable_outcomes := 21
  favorable_outcomes / total_outcomes

theorem dice_sum_10_probability :
  probability (λ (x : Fin 6 × Fin 6 × Fin 6), x.1.val + x.2.val + x.3.val + 3 = 10) = probability_sum_10 := 
sorry

end dice_sum_10_probability_l238_238169


namespace sol_sells_more_candy_each_day_l238_238661

variable {x : ℕ}

-- Definition of the conditions
def sells_candy (first_day : ℕ) (rate : ℕ) (days : ℕ) : ℕ :=
  first_day + rate * (days - 1) * days / 2

def earns (bars_sold : ℕ) (price_cents : ℕ) : ℕ :=
  bars_sold * price_cents

-- Problem statement in Lean:
theorem sol_sells_more_candy_each_day
  (first_day_sales : ℕ := 10)
  (days : ℕ := 6)
  (price_cents : ℕ := 10)
  (total_earnings : ℕ := 1200) :
  earns (sells_candy first_day_sales x days) price_cents = total_earnings → x = 76 :=
sorry

end sol_sells_more_candy_each_day_l238_238661


namespace part_1_solution_set_part_2_a_range_l238_238890

-- Define the function f
def f (x a : ℝ) := |x - a^2| + |x - 2 * a + 1|

-- Part (1)
theorem part_1_solution_set (x : ℝ) : {x | x ≤ 3 / 2 ∨ x ≥ 11 / 2} = 
  {x | f x 2 ≥ 4} :=
sorry

-- Part (2)
theorem part_2_a_range (a : ℝ) : 
  {a | (a - 1)^2 ≥ 4} = {a | a ≤ -1 ∨ a ≥ 3} :=
sorry

end part_1_solution_set_part_2_a_range_l238_238890


namespace part1_solution_set_part2_range_of_a_l238_238874

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1_solution_set (x : ℝ) : 
  (f x 2 ≥ 4) ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
sorry

theorem part2_range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 4) → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end part1_solution_set_part2_range_of_a_l238_238874


namespace find_value_of_f3_l238_238537

variable {R : Type} [LinearOrderedField R]

/-- f is an odd function -/
def is_odd_function (f : R → R) : Prop := ∀ x : R, f (-x) = -f x

/-- f is symmetric about the line x = 1 -/
def is_symmetric_about (f : R → R) (a : R) : Prop := ∀ x : R, f (a + x) = f (a - x)

variable (f : R → R)
variable (Hodd : is_odd_function f)
variable (Hsymmetric : is_symmetric_about f 1)
variable (Hf1 : f 1 = 2)

theorem find_value_of_f3 : f 3 = -2 :=
by
  sorry

end find_value_of_f3_l238_238537


namespace average_price_of_tshirts_l238_238656

theorem average_price_of_tshirts
  (A : ℝ)
  (total_cost_seven_remaining : ℝ := 7 * 505)
  (total_cost_three_returned : ℝ := 3 * 673)
  (total_cost_eight : ℝ := total_cost_seven_remaining + 673) -- since (1 t-shirt with price is included in the total)
  (total_cost_eight_eq : total_cost_eight = 8 * A) :
  A = 526 :=
by sorry

end average_price_of_tshirts_l238_238656


namespace part1_part2_l238_238881

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1 (x : ℝ) : 
  ∀ x, f x 2 ≥ 4 ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
by sorry

theorem part2 (x a : ℝ) :
  f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
by sorry

end part1_part2_l238_238881


namespace min_eq_floor_sqrt_l238_238983

theorem min_eq_floor_sqrt (n : ℕ) (h : n > 0) : 
  (∀ k : ℕ, k > 0 → (k + n / k) ≥ ⌊(Real.sqrt (4 * n + 1))⌋) := 
sorry

end min_eq_floor_sqrt_l238_238983


namespace hotdogs_sold_correct_l238_238099

def initial_hotdogs : ℕ := 99
def remaining_hotdogs : ℕ := 97
def sold_hotdogs : ℕ := initial_hotdogs - remaining_hotdogs

theorem hotdogs_sold_correct : sold_hotdogs = 2 := by
  sorry

end hotdogs_sold_correct_l238_238099


namespace dice_sum_probability_l238_238187

/-- The probability that the sum of the numbers on three six-faced dice equals 10 is 1/8. -/
theorem dice_sum_probability : 
  (1 / 8 : ℚ) = (∑ x in (finset.univ : finset (fin 6)), 
                ∑ y in (finset.univ : finset (fin 6)), 
                ∑ z in (finset.univ : finset (fin 6)), 
                if x.1 + y.1 + z.1 = 10 then 1 else 0) / 6^3 :=
begin
  sorry
end

end dice_sum_probability_l238_238187


namespace no_coprime_xy_multiple_l238_238357

theorem no_coprime_xy_multiple (n : ℕ) (hn : ∀ d : ℕ, d ∣ n → d^2 ∣ n → d = 1)
  (x y : ℕ) (hx_pos : x > 0) (hy_pos : y > 0) (h_coprime : Nat.gcd x y = 1) :
  ¬ ((x^n + y^n) % ((x + y)^3) = 0) :=
by
  sorry

end no_coprime_xy_multiple_l238_238357


namespace slope_of_line_l238_238621

theorem slope_of_line
  (k : ℝ) 
  (hk : 0 < k) 
  (h1 : ¬ (2 / Real.sqrt (k^2 + 1) = 3 * 2 * Real.sqrt (1 - 8 * k^2) / Real.sqrt (k^2 + 1))) 
  : k = 1 / 3 :=
sorry

end slope_of_line_l238_238621


namespace sum_of_roots_l238_238748

theorem sum_of_roots (a b c : ℝ) (h : 6 * a^3 + 7 * a^2 - 12 * a = 0) : 
  - (7 / 6 : ℝ) = -1.17 := 
sorry

end sum_of_roots_l238_238748


namespace cos_300_is_half_l238_238788

noncomputable def cos_300_degrees : ℝ :=
  real.cos (300 * real.pi / 180)

theorem cos_300_is_half :
  cos_300_degrees = 1 / 2 :=
sorry

end cos_300_is_half_l238_238788


namespace triangle_problem_l238_238957

theorem triangle_problem
  (A B C : ℝ)
  (a b c : ℝ)
  (hb : 0 < B ∧ B < Real.pi)
  (hc : 0 < C ∧ C < Real.pi)
  (ha : 0 < A ∧ A < Real.pi)
  (h_sum_angles : A + B + C = Real.pi)
  (h_sides : a > b)
  (h_perimeter : a + b + c = 20)
  (h_area : (1/2) * a * b * Real.sin C = 10 * Real.sqrt 3)
  (h_eq : a * (Real.sqrt 3 * Real.tan B - 1) = (b * Real.cos A / Real.cos B) + (c * Real.cos A / Real.cos C)) :
  C = Real.pi / 3 ∧ a = 8 ∧ b = 5 ∧ c = 7 := sorry

end triangle_problem_l238_238957


namespace statement_A_statement_C_statement_D_l238_238755

theorem statement_A (x : ℝ) :
  (¬ (∀ x ≥ 3, 2 * x - 10 ≥ 0)) ↔ (∃ x0 ≥ 3, 2 * x0 - 10 < 0) := 
sorry

theorem statement_C {a b c : ℝ} (h1 : c > a) (h2 : a > b) (h3 : b > 0) :
  (a / (c - a)) > (b / (c - b)) := 
sorry

theorem statement_D {a b m : ℝ} (h1 : a > b) (h2 : b > 0) (h3 : m > 0) :
  (a / b) > ((a + m) / (b + m)) := 
sorry

end statement_A_statement_C_statement_D_l238_238755


namespace evaluate_expression_l238_238602

theorem evaluate_expression : ∃ x : ℝ, (x = Real.sqrt (18 + x)) ∧ (x = (1 + Real.sqrt 73) / 2) := by
  sorry

end evaluate_expression_l238_238602


namespace sum_of_midpoints_l238_238543

theorem sum_of_midpoints {a b c : ℝ} (h : a + b + c = 15) : 
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 := 
by
  sorry

end sum_of_midpoints_l238_238543


namespace greatest_possible_sum_of_two_consecutive_integers_lt_500_l238_238712

theorem greatest_possible_sum_of_two_consecutive_integers_lt_500 (n : ℕ) (h : n * (n + 1) < 500) : n + (n + 1) ≤ 43 := by
  sorry

end greatest_possible_sum_of_two_consecutive_integers_lt_500_l238_238712


namespace greatest_sum_of_consecutive_integers_product_lt_500_l238_238730

theorem greatest_sum_of_consecutive_integers_product_lt_500 : 
  ∃ (n : ℕ), n * (n + 1) < 500 ∧ ∀ (m : ℕ), m * (m + 1) < 500 → m ≤ n → n + (n + 1) = 43 := 
by
  use 21
  split
  {
    norm_num
    linarith
  }
  {
    intros m h_hint h_ineq
    have : m ≤ 21, sorry
    linarith
  }
  sorry

end greatest_sum_of_consecutive_integers_product_lt_500_l238_238730


namespace marcia_banana_count_l238_238516

variable (B : ℕ)

-- Conditions
def appleCost := 2
def bananaCost := 1
def orangeCost := 3
def numApples := 12
def numOranges := 4
def avgCost := 2

-- Prove that given the conditions, B equals 4
theorem marcia_banana_count : 
  (24 + 12 + B) / (16 + B) = avgCost → B = 4 :=
by sorry

end marcia_banana_count_l238_238516


namespace percentage_increase_variable_cost_l238_238760

noncomputable def variable_cost_first_year : ℝ := 26000
noncomputable def fixed_cost : ℝ := 40000
noncomputable def total_breeding_cost_third_year : ℝ := 71460

theorem percentage_increase_variable_cost (x : ℝ) 
  (h : 40000 + 26000 * (1 + x) ^ 2 = 71460) : 
  x = 0.1 := 
by sorry

end percentage_increase_variable_cost_l238_238760


namespace cos_300_eq_half_l238_238810

theorem cos_300_eq_half : Real.cos (300 * Real.pi / 180) = 1 / 2 := 
by sorry

end cos_300_eq_half_l238_238810


namespace cos_300_is_half_l238_238813

noncomputable def cos_300_eq_half : Prop :=
  real.cos (2 * real.pi * 5 / 6) = 1 / 2

theorem cos_300_is_half : cos_300_eq_half :=
sorry

end cos_300_is_half_l238_238813


namespace part1_solution_set_part2_range_of_a_l238_238910

-- Define the function f for a general a
def f (x a : ℝ) : ℝ := |x - a^2| + |x - (2 * a) + 1|

-- Part 1: When a = 2
theorem part1_solution_set (x : ℝ) (h : f x 2 ≥ 4) : x ≤ 3/2 ∨ x ≥ 11/2 :=
  sorry

-- Part 2: Range of values for a
theorem part2_range_of_a (a : ℝ) (h : ∀ x, f x a ≥ 4) : a ≤ -1 ∨ a ≥ 3 :=
  sorry

end part1_solution_set_part2_range_of_a_l238_238910


namespace evaluate_expression_l238_238130

theorem evaluate_expression : 
  (3 / 20 - 5 / 200 + 7 / 2000 : ℚ) = 0.1285 :=
by
  sorry

end evaluate_expression_l238_238130


namespace length_error_probability_l238_238336

noncomputable def normal_dist (μ σ : ℝ) : MeasureTheory.ProbMeasure ℝ :=
  MeasureTheory.ProbMeasure.fromDensity ((Real.gaussian μ σ).toDensity)

theorem length_error_probability :
  let μ := 0
  let σ := 3
  let P := normal_dist μ σ
  P.measure {x | 3 < x ∧ x < 6} = 0.1359 :=
by
  sorry

end length_error_probability_l238_238336


namespace cos_300_eq_cos_300_l238_238824

open Real

theorem cos_300_eq (angle_1 angle_2 : ℝ) (h1 : angle_1 = 300 * π / 180) (h2 : angle_2 = 60 * π / 180) :
  cos angle_1 = cos angle_2 :=
by
  rw [←h1, ←h2, cos_sub, cos_pi_div_six]
  norm_num
  sorry

theorem cos_300 : cos (300 * π / 180) = 1 / 2 :=
by
  apply cos_300_eq (300 * π / 180) (60 * π / 180)
  norm_num
  norm_num
  sorry

end cos_300_eq_cos_300_l238_238824


namespace parabola_directrix_l238_238666

theorem parabola_directrix {x y : ℝ} (h : y^2 = 6 * x) : x = -3 / 2 := 
sorry

end parabola_directrix_l238_238666


namespace equivalent_statement_l238_238152

theorem equivalent_statement (x y z w : ℝ)
  (h : (2 * x + y) / (y + z) = (z + w) / (w + 2 * x)) :
  (x = z / 2 ∨ 2 * x + y + z + w = 0) :=
sorry

end equivalent_statement_l238_238152


namespace smallest_prime_divides_l238_238321

theorem smallest_prime_divides (p : ℕ) (a : ℕ) 
  (h1 : Prime p) (h2 : p > 100) (h3 : a > 1) (h4 : p ∣ (a^89 - 1) / (a - 1)) :
  p = 179 := 
sorry

end smallest_prime_divides_l238_238321


namespace sum_of_roots_l238_238745

theorem sum_of_roots (a b c : ℝ) (x1 x2 x3 : ℝ) (h_eq: 6*x1^3 + 7*x2^2 - 12*x3 = 0) :
  (x1 + x2 + x3) = -1.17 :=
sorry

end sum_of_roots_l238_238745


namespace part1_solution_set_part2_range_of_a_l238_238915

def f (x a : ℝ) := |x - a^2| + |x - (2*a + 1)|

-- Part (1)
theorem part1_solution_set (x : ℝ) :
  ∀ x, f x 2 ≥ 4 ↔ x ≤ 3 / 2 ∨ x ≥ 11 / 2 := sorry

-- Part (2)
theorem part2_range_of_a (a : ℝ) :
  (∀ x, f x a ≥ 4) ↔ (a ≤ -1 ∨ a ≥ 3) := sorry

end part1_solution_set_part2_range_of_a_l238_238915


namespace greatest_sum_of_consecutive_integers_product_less_500_l238_238714

theorem greatest_sum_of_consecutive_integers_product_less_500 :
  ∃ n : ℤ, n * (n + 1) < 500 ∧ (n + (n + 1)) = 43 :=
by
  sorry

end greatest_sum_of_consecutive_integers_product_less_500_l238_238714


namespace lindy_total_distance_l238_238278

theorem lindy_total_distance (distance_jc : ℝ) (speed_j : ℝ) (speed_c : ℝ) (speed_l : ℝ)
  (h1 : distance_jc = 270) (h2 : speed_j = 4) (h3 : speed_c = 5) (h4 : speed_l = 8) : 
  ∃ time : ℝ, time = distance_jc / (speed_j + speed_c) ∧ speed_l * time = 240 :=
by
  sorry

end lindy_total_distance_l238_238278


namespace find_m_l238_238487

theorem find_m (A B : Set ℝ) (m : ℝ) (hA: A = {2, m}) (hB: B = {1, m^2}) (hU: A ∪ B = {1, 2, 3, 9}) : m = 3 :=
by 
  sorry

end find_m_l238_238487


namespace dice_sum_prob_10_l238_238176

/-- A standard six-faced die has faces numbered 1 through 6 --/
def is_standard_die_face (n : ℕ) : Prop := n ∈ {1, 2, 3, 4, 5, 6}

/-- The probability of the sum of the numbers rolled on three standard six-faced dice being 10 --/
theorem dice_sum_prob_10 : 
  ∃ (count_10 : ℕ), ∃ (total : ℕ), 
  (total = 6^3) ∧ 
  (count_10 = 27) ∧ 
  (count_10.toRat / total.toRat = 27 / 216) := 
by 
  sorry

end dice_sum_prob_10_l238_238176


namespace sum_of_midpoints_of_triangle_l238_238544

theorem sum_of_midpoints_of_triangle (a b c : ℝ) (h : a + b + c = 15) :
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 :=
by
  sorry

end sum_of_midpoints_of_triangle_l238_238544


namespace find_value_of_4_minus_2a_l238_238157

theorem find_value_of_4_minus_2a (a b : ℚ) (h1 : 4 + 2 * a = 5 - b) (h2 : 5 + b = 9 + 3 * a) : 4 - 2 * a = 26 / 5 := 
by
  sorry

end find_value_of_4_minus_2a_l238_238157


namespace fraction_to_decimal_l238_238002

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l238_238002


namespace lorry_empty_weight_l238_238289

-- Define variables for the weights involved
variable (lw : ℕ)  -- weight of the lorry when empty
variable (bl : ℕ)  -- number of bags of apples
variable (bw : ℕ)  -- weight of each bag of apples
variable (total_weight : ℕ)  -- total loaded weight of the lorry

-- Given conditions
axiom lorry_loaded_weight : bl = 20 ∧ bw = 60 ∧ total_weight = 1700

-- The theorem we want to prove
theorem lorry_empty_weight : (∀ lw bw, total_weight - bl * bw = lw) → lw = 500 :=
by
  intro h
  rw [←h lw bw]
  sorry

end lorry_empty_weight_l238_238289


namespace necessary_for_A_l238_238475

-- Define the sets A, B, C as non-empty sets
variables {α : Type*} (A B C : Set α)
-- Non-empty sets
axiom non_empty_A : ∃ x, x ∈ A
axiom non_empty_B : ∃ x, x ∈ B
axiom non_empty_C : ∃ x, x ∈ C

-- Conditions
axiom union_condition : A ∪ B = C
axiom subset_condition : ¬ (B ⊆ A)

-- Statement to prove
theorem necessary_for_A (x : α) : (x ∈ C → x ∈ A) ∧ ¬(x ∈ C ↔ x ∈ A) :=
sorry

end necessary_for_A_l238_238475


namespace arithmetic_sequence_sum_l238_238381

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) (d : ℤ) :
  (∀ n, a n = a 1 + (n - 1) * d) → 
  (∀ n, S n = n * (a 1 + a n) / 2) → 
  (a 3 + 4 = a 2 + a 7) → 
  S 11 = 44 :=
by 
  sorry

end arithmetic_sequence_sum_l238_238381


namespace janet_pills_monthly_l238_238509

def daily_intake_first_two_weeks := 2 + 3 -- 2 multivitamins + 3 calcium supplements
def daily_intake_last_two_weeks := 2 + 1 -- 2 multivitamins + 1 calcium supplement
def days_in_two_weeks := 2 * 7

theorem janet_pills_monthly :
  (daily_intake_first_two_weeks * days_in_two_weeks) + (daily_intake_last_two_weeks * days_in_two_weeks) = 112 :=
by
  sorry

end janet_pills_monthly_l238_238509


namespace part1_solution_set_part2_range_of_a_l238_238932

noncomputable def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1_solution_set (a : ℝ) (a_eq_2 : a = 2) : 
  {x : ℝ | f x a ≥ 4} = {x | x ≤ 3 / 2} ∪ {x | x ≥ 11 / 2} :=
by
  sorry

theorem part2_range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 4) → (a ∈ set.Iic (-1) ∪ set.Ici 3) :=
by
  sorry

end part1_solution_set_part2_range_of_a_l238_238932


namespace part1_solution_set_part2_values_of_a_l238_238864

-- Parameters and definitions for the function and conditions
def f (x a : ℝ) := |x - a^2| + |x - (2*a + 1)|

-- Part (1) When a = 2, the inequality's solution set
theorem part1_solution_set (x : ℝ) : f x 2 ≥ 4 ↔ (x ≤ 3/2 ∨ x ≥ 11/2) := 
sorry

-- Part (2) Range of values for a given f(x, a) ≥ 4
theorem part2_values_of_a (a : ℝ) : 
∀ x, f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end part1_solution_set_part2_values_of_a_l238_238864


namespace fountains_for_m_4_fountains_for_m_3_l238_238119

noncomputable def ceil_div (a b : ℕ) : ℕ :=
  (a + b - 1) / b

-- Problem for m = 4
theorem fountains_for_m_4 (n : ℕ) : ∃ f : ℕ, f = 2 * ceil_div n 3 := 
sorry

-- Problem for m = 3
theorem fountains_for_m_3 (n : ℕ) : ∃ f : ℕ, f = 3 * ceil_div n 3 :=
sorry

end fountains_for_m_4_fountains_for_m_3_l238_238119


namespace dice_sum_10_probability_l238_238170

open Probability

noncomputable def probability_sum_10 : ℚ := 
  let total_outcomes := 216
  let favorable_outcomes := 21
  favorable_outcomes / total_outcomes

theorem dice_sum_10_probability :
  probability (λ (x : Fin 6 × Fin 6 × Fin 6), x.1.val + x.2.val + x.3.val + 3 = 10) = probability_sum_10 := 
sorry

end dice_sum_10_probability_l238_238170


namespace total_alphabets_written_l238_238584

-- Define the number of vowels and the number of times each is written
def num_vowels : ℕ := 5
def repetitions : ℕ := 4

-- The theorem stating the total number of alphabets written on the board
theorem total_alphabets_written : num_vowels * repetitions = 20 := by
  sorry

end total_alphabets_written_l238_238584


namespace fraction_decimal_equivalent_l238_238010

theorem fraction_decimal_equivalent : (7 / 16 : ℝ) = 0.4375 :=
by
  sorry

end fraction_decimal_equivalent_l238_238010


namespace A_share_of_gain_l238_238275

-- Definitions of conditions
variables 
  (x : ℕ) -- Initial investment by A
  (annual_gain : ℕ := 24000) -- Total annual gain
  (A_investment_period : ℕ := 12) -- Months A invested
  (B_investment_period : ℕ := 6) -- Months B invested after 6 months
  (C_investment_period : ℕ := 4) -- Months C invested after 8 months

-- Investment ratios
def A_ratio := x * A_investment_period
def B_ratio := (2 * x) * B_investment_period
def C_ratio := (3 * x) * C_investment_period

-- Proof statement
theorem A_share_of_gain : 
  A_ratio = 12 * x ∧ B_ratio = 12 * x ∧ C_ratio = 12 * x ∧ annual_gain = 24000 →
  annual_gain / 3 = 8000 :=
by
  sorry

end A_share_of_gain_l238_238275


namespace correct_relation_l238_238488

open Set

def U : Set ℝ := univ

def A : Set ℝ := { x | x^2 < 4 }

def B : Set ℝ := { x | x > 2 }

def comp_of_B : Set ℝ := U \ B

theorem correct_relation : A ∩ comp_of_B = A := by
  sorry

end correct_relation_l238_238488


namespace area_of_border_correct_l238_238765

def height_of_photograph : ℕ := 12
def width_of_photograph : ℕ := 16
def border_width : ℕ := 3
def lining_width : ℕ := 1

def area_of_photograph : ℕ := height_of_photograph * width_of_photograph

def total_height : ℕ := height_of_photograph + 2 * (lining_width + border_width)
def total_width : ℕ := width_of_photograph + 2 * (lining_width + border_width)

def area_of_framed_area : ℕ := total_height * total_width

def area_of_border_including_lining : ℕ := area_of_framed_area - area_of_photograph

theorem area_of_border_correct : area_of_border_including_lining = 288 := by
  sorry

end area_of_border_correct_l238_238765


namespace sin_15_cos_15_l238_238136

theorem sin_15_cos_15 : (Real.sin (15 * Real.pi / 180)) * (Real.cos (15 * Real.pi / 180)) = 1 / 4 := by
  sorry

end sin_15_cos_15_l238_238136


namespace fraction_to_decimal_l238_238032

theorem fraction_to_decimal : (7 : ℝ) / 16 = 0.4375 := 
by
  sorry

end fraction_to_decimal_l238_238032


namespace intersection_nonempty_implies_a_gt_neg1_l238_238486

def A := {x : ℝ | -1 ≤ x ∧ x < 2}
def B (a : ℝ) := {x : ℝ | x < a}

theorem intersection_nonempty_implies_a_gt_neg1 (a : ℝ) : (A ∩ B a).Nonempty → a > -1 :=
by
  sorry

end intersection_nonempty_implies_a_gt_neg1_l238_238486


namespace find_integer_pairs_l238_238836

theorem find_integer_pairs (a b : ℤ) (h₁ : 1 < a) (h₂ : 1 < b) 
    (h₃ : a ∣ (b + 1)) (h₄ : b ∣ (a^3 - 1)) : 
    ∃ (s : ℤ), (s ≥ 2 ∧ (a, b) = (s, s^3 - 1)) ∨ (s ≥ 3 ∧ (a, b) = (s, s - 1)) :=
  sorry

end find_integer_pairs_l238_238836


namespace train_crosses_pole_in_12_seconds_l238_238577

noncomputable def time_to_cross_pole (speed train_length : ℕ) : ℕ := 
  train_length / speed

theorem train_crosses_pole_in_12_seconds 
  (speed : ℕ) (platform_length : ℕ) (time_to_cross_platform : ℕ) (train_crossing_time : ℕ)
  (h_speed : speed = 10) 
  (h_platform_length : platform_length = 320) 
  (h_time_to_cross_platform : time_to_cross_platform = 44) 
  (h_train_crossing_time : train_crossing_time = 12) :
  time_to_cross_pole speed 120 = train_crossing_time := 
by 
  sorry

end train_crosses_pole_in_12_seconds_l238_238577


namespace max_value_product_focal_distances_l238_238982

theorem max_value_product_focal_distances {a b c : ℝ} 
  (h1 : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1) 
  (h2 : ∀ x : ℝ, -a ≤ x ∧ x ≤ a) 
  (e : ℝ) :
  (∀ x : ℝ, (a - e * x) * (a + e * x) ≤ a^2) :=
sorry

end max_value_product_focal_distances_l238_238982


namespace greatest_possible_sum_of_two_consecutive_integers_lt_500_l238_238710

theorem greatest_possible_sum_of_two_consecutive_integers_lt_500 (n : ℕ) (h : n * (n + 1) < 500) : n + (n + 1) ≤ 43 := by
  sorry

end greatest_possible_sum_of_two_consecutive_integers_lt_500_l238_238710


namespace karting_number_of_routes_l238_238367

theorem karting_number_of_routes :
  let M : ℕ → ℕ := λ n, Nat.fib (n + 1)
  in M 10 = 34 :=
by
  let M : ℕ → ℕ := λ n, Nat.fib (n + 1)
  show M 10 = 34
  exact Nat.fib_succ_succ 9

end karting_number_of_routes_l238_238367


namespace evaluate_at_2_l238_238299

-- Define the polynomial function using Lean
def f (x : ℝ) : ℝ := 3 * x^3 + 2 * x^2 + x + 1

-- State the theorem that f(2) evaluates to 35 using Horner's method
theorem evaluate_at_2 : f 2 = 35 := by
  sorry

end evaluate_at_2_l238_238299


namespace point_in_fourth_quadrant_l238_238355

theorem point_in_fourth_quadrant (m : ℝ) : (m-1 > 0 ∧ 2-m < 0) ↔ m > 2 :=
by
  sorry

end point_in_fourth_quadrant_l238_238355


namespace pyramid_volume_is_one_sixth_l238_238451

noncomputable def volume_of_pyramid_in_cube : ℝ :=
  let edge_length := 1
  let base_area := (1 / 2) * edge_length * edge_length
  let height := edge_length
  (1 / 3) * base_area * height

theorem pyramid_volume_is_one_sixth : volume_of_pyramid_in_cube = 1 / 6 :=
by
  -- Let edge_length = 1, base_area = 1 / 2 * edge_length * edge_length = 1 / 2, 
  -- height = edge_length = 1. Then volume = 1 / 3 * base_area * height = 1 / 6.
  sorry

end pyramid_volume_is_one_sixth_l238_238451


namespace inequality_correct_l238_238334

variable {a b c : ℝ}

theorem inequality_correct (h : a * b < 0) : |a - c| ≤ |a - b| + |b - c| :=
sorry

end inequality_correct_l238_238334


namespace connected_graphs_bound_l238_238304

noncomputable def num_connected_graphs (n : ℕ) : ℕ := sorry
  
theorem connected_graphs_bound (n : ℕ) : 
  num_connected_graphs n ≥ (1/2) * 2^(n*(n-1)/2) := 
sorry

end connected_graphs_bound_l238_238304


namespace james_calories_per_minute_l238_238220

-- Define the conditions
def bags : Nat := 3
def ounces_per_bag : Nat := 2
def calories_per_ounce : Nat := 150
def excess_calories : Nat := 420
def run_minutes : Nat := 40

-- Calculate the total consumed calories
def consumed_calories : Nat := (bags * ounces_per_bag) * calories_per_ounce

-- Calculate the calories burned during the run
def run_calories : Nat := consumed_calories - excess_calories

-- Calculate the calories burned per minute
def calories_per_minute : Nat := run_calories / run_minutes

-- The proof problem statement
theorem james_calories_per_minute : calories_per_minute = 12 := by
  -- Due to the proof not required, we use sorry to skip it.
  sorry

end james_calories_per_minute_l238_238220


namespace dice_sum_probability_l238_238197

theorem dice_sum_probability : (∃ s, (∃ x y z : ℕ, 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 ∧ 1 ≤ z ∧ z ≤ 6 ∧ x + y + z = 10)
  ∧ s = {n : ℕ // ∃ x y z : ℕ, 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 ∧ 1 ≤ z ∧ z ≤ 6 ∧ x + y + z = n} ∧ s.card = 24)
  → 24 / 216 = 1 / 9 := 
sorry

end dice_sum_probability_l238_238197


namespace ken_paid_20_l238_238442

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

end ken_paid_20_l238_238442


namespace problem_proof_l238_238225

variables (a b : ℝ) (n : ℕ)

theorem problem_proof (h1: a > 0) (h2: b > 0) (h3: a + b = 1) (h4: n >= 2) :
  3/2 < 1/(a^n + 1) + 1/(b^n + 1) ∧ 1/(a^n + 1) + 1/(b^n + 1) ≤ (2^(n+1))/(2^n + 1) := sorry

end problem_proof_l238_238225


namespace fraction_decimal_equivalent_l238_238008

theorem fraction_decimal_equivalent : (7 / 16 : ℝ) = 0.4375 :=
by
  sorry

end fraction_decimal_equivalent_l238_238008


namespace dice_sum_probability_l238_238193

theorem dice_sum_probability : (∃ s, (∃ x y z : ℕ, 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 ∧ 1 ≤ z ∧ z ≤ 6 ∧ x + y + z = 10)
  ∧ s = {n : ℕ // ∃ x y z : ℕ, 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 ∧ 1 ≤ z ∧ z ≤ 6 ∧ x + y + z = n} ∧ s.card = 24)
  → 24 / 216 = 1 / 9 := 
sorry

end dice_sum_probability_l238_238193


namespace statues_ratio_l238_238626

theorem statues_ratio :
  let y1 := 4                  -- Number of statues after first year.
  let y2 := 4 * y1             -- Number of statues after second year.
  let y3 := (y2 + 12) - 3      -- Number of statues after third year.
  let y4 := 31                 -- Number of statues after fourth year.
  let added_fourth_year := y4 - y3  -- Statues added in the fourth year.
  let broken_third_year := 3        -- Statues broken in the third year.
  added_fourth_year / broken_third_year = 2 :=
by
  sorry

end statues_ratio_l238_238626


namespace value_of_x_l238_238753

theorem value_of_x (x : ℝ) :
  (4 / x) * 12 = 8 ↔ x = 6 :=
by
  sorry

end value_of_x_l238_238753


namespace fraction_to_decimal_l238_238021

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l238_238021


namespace fraction_to_decimal_l238_238054

theorem fraction_to_decimal :
  (7 : ℝ) / 16 = 0.4375 := 
by sorry

end fraction_to_decimal_l238_238054


namespace range_of_a_l238_238752

noncomputable def isIncreasing (f : ℝ → ℝ) := ∀ ⦃x y : ℝ⦄, x < y → f x < f y

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x < 1 → e ^ x - 1 ≥ x ^ 2 - a * x) → a ≥ 2 - Real.exp 1 :=
by
  sorry

end range_of_a_l238_238752


namespace part1_solution_set_part2_range_of_a_l238_238925

noncomputable def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1_solution_set (a : ℝ) (a_eq_2 : a = 2) : 
  {x : ℝ | f x a ≥ 4} = {x | x ≤ 3 / 2} ∪ {x | x ≥ 11 / 2} :=
by
  sorry

theorem part2_range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 4) → (a ∈ set.Iic (-1) ∪ set.Ici 3) :=
by
  sorry

end part1_solution_set_part2_range_of_a_l238_238925


namespace trajectory_of_P_equation_of_line_l238_238474

-- Given conditions
def P (x y : ℝ) : Prop := True
def F : ℝ × ℝ := (3 * real.sqrt 3, 0)
def l : ℝ → ℝ := λ x, 4 * real.sqrt 3
def ratio : ℝ := real.sqrt 3 / 2
def M : ℝ × ℝ := (4, 2)

-- Questions translated to Lean statements
theorem trajectory_of_P (x y : ℝ) :
  (real.sqrt ((x - 3 * real.sqrt 3)^2 + y^2) / |x - 4 * real.sqrt 3| = real.sqrt 3 / 2) →
  (x^2 / 36 + y^2 / 9 = 1) :=
sorry

theorem equation_of_line (k : ℝ) :
  let line := λ x, k * (x - 4) + 2 in
  (∃ B C : ℝ × ℝ, B ≠ C ∧ M = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) ∧
   (B.2 = line B.1 ∧ C.2 = line C.1)) →
  k = -1/2 ∧ (∀ x y, y = k * (x - 4) + 2 ↔ x + 2 * y - 8 = 0) :=
sorry

end trajectory_of_P_equation_of_line_l238_238474


namespace data_variance_l238_238398

def data : List ℝ := [9.8, 9.9, 10.1, 10, 10.2]

noncomputable def mean (data : List ℝ) : ℝ :=
  data.sum / data.length

noncomputable def variance (data : List ℝ) : ℝ :=
  (data.map (λ x => (x - mean data) ^ 2)).sum / data.length

theorem data_variance : variance data = 0.02 := by
  sorry

end data_variance_l238_238398


namespace part1_solution_set_part2_range_a_l238_238903

def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

theorem part1_solution_set (x : ℝ) (h : f x 2 ≥ 4) : 
  x ≤ 3/2 ∨ x ≥ 11/2 :=
sorry

theorem part2_range_a (a : ℝ) (h : ∃ x, f x a ≥ 4) : 
  a ≤ -1 ∨ a ≥ 3 :=
sorry

end part1_solution_set_part2_range_a_l238_238903


namespace center_of_circle_in_second_or_fourth_quadrant_l238_238632

theorem center_of_circle_in_second_or_fourth_quadrant
  (α : ℝ) 
  (hyp1 : ∀ x y : ℝ, x^2 * Real.cos α - y^2 * Real.sin α + 2 = 0 → Real.cos α * Real.sin α > 0)
  (circle_eq : ∀ x y : ℝ, x^2 + y^2 + 2*x*Real.cos α - 2*y*Real.sin α = 0) :
  (-Real.cos α > 0 ∧ Real.sin α > 0) ∨ (-Real.cos α < 0 ∧ Real.sin α < 0) :=
sorry

end center_of_circle_in_second_or_fourth_quadrant_l238_238632


namespace sanity_proof_l238_238283

-- Define the characters and their sanity status as propositions
variables (Griffin QuasiTurtle Lobster : Prop)

-- Conditions
axiom Lobster_thinks : (Griffin ∧ ¬QuasiTurtle ∧ ¬Lobster) ∨ (¬Griffin ∧ QuasiTurtle ∧ ¬Lobster) ∨ (¬Griffin ∧ ¬QuasiTurtle ∧ Lobster)
axiom QuasiTurtle_thinks : Griffin

-- Statement to prove
theorem sanity_proof : ¬Griffin ∧ ¬QuasiTurtle ∧ ¬Lobster :=
by {
  sorry
}

end sanity_proof_l238_238283


namespace only_prime_satisfying_condition_l238_238315

theorem only_prime_satisfying_condition (p : ℕ) (h_prime : Prime p) : (Prime (p^2 + 14) ↔ p = 3) := 
by
  sorry

end only_prime_satisfying_condition_l238_238315


namespace A_investment_l238_238769

-- Conditions as definitions
def B_investment := 72000
def C_investment := 81000
def C_profit := 36000
def Total_profit := 80000

-- Statement to prove
theorem A_investment : 
  ∃ (x : ℕ), x = 27000 ∧
  (C_profit / Total_profit = (9 : ℕ) / 20) ∧
  (C_investment / (x + B_investment + C_investment) = (9 : ℕ) / 20) :=
by sorry

end A_investment_l238_238769


namespace inequality_proof_l238_238974

noncomputable def inequality (a b c : ℝ) (ha: a > 1) (hb: b > 1) (hc: c > 1) : Prop :=
  (a * b) / (c - 1) + (b * c) / (a - 1) + (c * a) / (b - 1) >= 12

theorem inequality_proof (a b c : ℝ) (ha: a > 1) (hb: b > 1) (hc: c > 1) : inequality a b c ha hb hc :=
by
  sorry

end inequality_proof_l238_238974


namespace determine_k_for_linear_dependence_l238_238457

theorem determine_k_for_linear_dependence :
  ∃ k : ℝ, (∀ (a1 a2 : ℝ), a1 ≠ 0 ∧ a2 ≠ 0 → 
  a1 • (⟨1, 2, 3⟩ : ℝ × ℝ × ℝ) + a2 • (⟨4, k, 6⟩ : ℝ × ℝ × ℝ) = (⟨0, 0, 0⟩ : ℝ × ℝ × ℝ)) → k = 8 :=
by
  sorry

end determine_k_for_linear_dependence_l238_238457


namespace percentage_increase_proof_l238_238969

def breakfast_calories : ℕ := 500
def shakes_total_calories : ℕ := 3 * 300
def total_daily_calories : ℕ := 3275

noncomputable def percentage_increase_in_calories (P : ℝ) : Prop :=
  let lunch_calories := breakfast_calories * (1 + P / 100)
  let dinner_calories := 2 * lunch_calories
  breakfast_calories + lunch_calories + dinner_calories + shakes_total_calories = total_daily_calories

theorem percentage_increase_proof : percentage_increase_in_calories 125 :=
by
  sorry

end percentage_increase_proof_l238_238969


namespace problem1_problem2_l238_238153

-- Proof problem (1)
theorem problem1 (x : ℝ) : 
  A = {x : ℝ | -3 ≤ x ∧ x ≤ 4} ∧ B = {x : ℝ | 1 < x ∧ x < 2} ∧ m = 1 →
  (A ∩ B) = {x : ℝ | 1 < x ∧ x < 2} := 
by 
  sorry

-- Proof problem (2)
theorem problem2 (x : ℝ) : 
  A = {x : ℝ | -3 ≤ x ∧ x ≤ 4} ∧ B = {x : ℝ | 2 * m - 1 < x ∧ x < m + 1} →
  (B ⊆ A ↔ (m ≥ 2 ∨ (-1 ≤ m ∧ m < 2))) := 
by 
  sorry

end problem1_problem2_l238_238153


namespace area_of_region_l238_238551

theorem area_of_region : 
  (∃ (x y : ℝ), x^2 + y^2 + 4*x - 6*y = 1) → (∃ (A : ℝ), A = 14 * Real.pi) := 
by
  sorry

end area_of_region_l238_238551


namespace sheets_of_paper_used_l238_238506

-- Define the conditions
def pages_per_book := 600
def number_of_books := 2
def pages_per_side := 4
def sides_per_sheet := 2

-- Calculate the total number of pages
def total_pages := pages_per_book * number_of_books

-- Calculate the number of pages per sheet of paper
def pages_per_sheet := pages_per_side * sides_per_sheet

-- Define the proof problem
theorem sheets_of_paper_used : total_pages / pages_per_sheet = 150 :=
by
  have h1 : total_pages = 1200 := by simp [total_pages, pages_per_book, number_of_books]
  have h2 : pages_per_sheet = 8 := by simp [pages_per_sheet, pages_per_side, sides_per_sheet]
  rw [h1, h2]
  norm_num
  done

end sheets_of_paper_used_l238_238506


namespace hexagon_shaded_area_correct_l238_238365

theorem hexagon_shaded_area_correct :
  let side_length := 3
  let semicircle_radius := side_length / 2
  let central_circle_radius := 1
  let hexagon_area := (3 * Real.sqrt 3 / 2) * side_length ^ 2
  let semicircle_area := (π * (semicircle_radius ^ 2)) / 2
  let total_semicircle_area := 6 * semicircle_area
  let central_circle_area := π * (central_circle_radius ^ 2)
  let shaded_area := hexagon_area - (total_semicircle_area + central_circle_area)
  shaded_area = 13.5 * Real.sqrt 3 - 7.75 * π := by
  sorry

end hexagon_shaded_area_correct_l238_238365


namespace cos_300_eq_cos_300_l238_238823

open Real

theorem cos_300_eq (angle_1 angle_2 : ℝ) (h1 : angle_1 = 300 * π / 180) (h2 : angle_2 = 60 * π / 180) :
  cos angle_1 = cos angle_2 :=
by
  rw [←h1, ←h2, cos_sub, cos_pi_div_six]
  norm_num
  sorry

theorem cos_300 : cos (300 * π / 180) = 1 / 2 :=
by
  apply cos_300_eq (300 * π / 180) (60 * π / 180)
  norm_num
  norm_num
  sorry

end cos_300_eq_cos_300_l238_238823


namespace abs_diff_squares_105_95_l238_238693

theorem abs_diff_squares_105_95 : abs ((105:ℤ)^2 - (95:ℤ)^2) = 2000 := by
  sorry

end abs_diff_squares_105_95_l238_238693


namespace not_prime_for_some_n_l238_238970

theorem not_prime_for_some_n (a : ℕ) (h : 1 < a) : ∃ n : ℕ, ¬ Nat.Prime (2^(2^n) + a) := 
sorry

end not_prime_for_some_n_l238_238970


namespace present_age_of_father_l238_238425

/-- The present age of the father is 3 years more than 3 times the age of his son, 
    and 3 years hence, the father's age will be 8 years more than twice the age of the son. 
    Prove that the present age of the father is 27 years. -/
theorem present_age_of_father (F S : ℕ) (h1 : F = 3 * S + 3) (h2 : F + 3 = 2 * (S + 3) + 8) : F = 27 :=
by
  sorry

end present_age_of_father_l238_238425


namespace cell_count_at_end_of_twelvth_day_l238_238430

def initial_cells : ℕ := 5
def days_per_cycle : ℕ := 3
def total_days : ℕ := 12
def dead_cells_on_ninth_day : ℕ := 3
noncomputable def cells_after_twelvth_day : ℕ :=
  let cycles := total_days / days_per_cycle
  let cells_before_death := initial_cells * 2^cycles
  cells_before_death - dead_cells_on_ninth_day

theorem cell_count_at_end_of_twelvth_day : cells_after_twelvth_day = 77 :=
by sorry

end cell_count_at_end_of_twelvth_day_l238_238430


namespace alex_loan_difference_l238_238290

theorem alex_loan_difference :
  let P := (15000 : ℝ)
  let r1 := (0.08 : ℝ)
  let n := (2 : ℕ)
  let t := (12 : ℕ)
  let r2 := (0.09 : ℝ)
  
  -- Calculate the amount owed after 6 years with compound interest (first option)
  let A1_half := P * (1 + r1 / n)^(n * t / 2)
  let half_payment := A1_half / 2
  let remaining_balance := A1_half / 2
  let A1_final := remaining_balance * (1 + r1 / n)^(n * t / 2)
  
  -- Total payment for the first option
  let total1 := half_payment + A1_final
  
  -- Total payment for the second option (simple interest)
  let simple_interest := P * r2 * t
  let total2 := P + simple_interest
  
  -- Compute the positive difference
  let difference := abs (total1 - total2)
  
  difference = 24.59 :=
  by
  sorry

end alex_loan_difference_l238_238290


namespace comic_book_issue_pages_l238_238406

theorem comic_book_issue_pages (total_pages: ℕ) 
  (speed_month1 speed_month2 speed_month3: ℕ) 
  (bonus_pages: ℕ) (issue1_2_pages: ℕ) 
  (issue3_pages: ℕ)
  (h1: total_pages = 220)
  (h2: speed_month1 = 5)
  (h3: speed_month2 = 4)
  (h4: speed_month3 = 4)
  (h5: issue3_pages = issue1_2_pages + 4)
  (h6: bonus_pages = 3)
  (h7: (issue1_2_pages + bonus_pages) + 
       (issue1_2_pages + bonus_pages) + 
       (issue3_pages + bonus_pages) = total_pages) : 
  issue1_2_pages = 69 := 
by 
  sorry

end comic_book_issue_pages_l238_238406


namespace alice_score_l238_238364

variables (correct_answers wrong_answers unanswered_questions : ℕ)
variables (points_correct points_incorrect : ℚ)

def compute_score (correct_answers wrong_answers : ℕ) (points_correct points_incorrect : ℚ) : ℚ :=
    (correct_answers : ℚ) * points_correct + (wrong_answers : ℚ) * points_incorrect

theorem alice_score : 
    correct_answers = 15 → 
    wrong_answers = 5 → 
    unanswered_questions = 10 → 
    points_correct = 1 → 
    points_incorrect = -0.25 → 
    compute_score 15 5 1 (-0.25) = 13.75 := 
by intros; sorry

end alice_score_l238_238364


namespace functional_equation_solution_l238_238594

theorem functional_equation_solution (f : ℤ → ℝ) (hf : ∀ x y : ℤ, f (↑((x + y) / 3)) = (f x + f y) / 2) :
    ∃ c : ℝ, ∀ x : ℤ, x ≠ 0 → f x = c :=
sorry

end functional_equation_solution_l238_238594


namespace cos_300_eq_one_half_l238_238793

theorem cos_300_eq_one_half :
  Real.cos (300 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end cos_300_eq_one_half_l238_238793


namespace mickey_horses_per_week_l238_238121

def days_in_week : ℕ := 7

def horses_minnie_per_day : ℕ := days_in_week + 3

def horses_twice_minnie_per_day : ℕ := 2 * horses_minnie_per_day

def horses_mickey_per_day : ℕ := horses_twice_minnie_per_day - 6

def horses_mickey_per_week : ℕ := days_in_week * horses_mickey_per_day

theorem mickey_horses_per_week : horses_mickey_per_week = 98 := sorry

end mickey_horses_per_week_l238_238121


namespace probability_sum_is_ten_l238_238181

structure Die :=
(value : ℕ)
(h_value : 1 ≤ value ∧ value ≤ 6)

def possible_rolls : List (ℕ × ℕ × ℕ) :=
List.product 
  (List.product [1, 2, 3, 4, 5, 6] [1, 2, 3, 4, 5, 6]) [1, 2, 3, 4, 5, 6]
|>.map (λ ((a, b), c) => (a, b, c))

def favorable_outcomes : List (ℕ × ℕ × ℕ) :=
possible_rolls.filter (λ (x, y, z) => x + y + z = 10)

def probability (n favorable : ℕ) : ℚ := favourable / n

theorem probability_sum_is_ten :
  probability possible_rolls.length favourable_outcomes.length = 1 / 9 :=
by
  sorry

end probability_sum_is_ten_l238_238181


namespace fractional_to_decimal_l238_238063

theorem fractional_to_decimal : (7 / 16 : ℚ) = 0.4375 := by
  sorry

end fractional_to_decimal_l238_238063


namespace dice_sum_prob_10_l238_238174

/-- A standard six-faced die has faces numbered 1 through 6 --/
def is_standard_die_face (n : ℕ) : Prop := n ∈ {1, 2, 3, 4, 5, 6}

/-- The probability of the sum of the numbers rolled on three standard six-faced dice being 10 --/
theorem dice_sum_prob_10 : 
  ∃ (count_10 : ℕ), ∃ (total : ℕ), 
  (total = 6^3) ∧ 
  (count_10 = 27) ∧ 
  (count_10.toRat / total.toRat = 27 / 216) := 
by 
  sorry

end dice_sum_prob_10_l238_238174


namespace sum_of_roots_cubic_equation_l238_238750

theorem sum_of_roots_cubic_equation :
  let roots := multiset.to_finset (multiset.filter (λ r, r ≠ 0) (RootSet (6 * (X ^ 3) + 7 * (X ^ 2) + (-12) * X) ℤ))
  (roots.sum : ℤ) / (roots.card : ℤ) = -117 / 100 := sorry

end sum_of_roots_cubic_equation_l238_238750


namespace probability_sum_is_10_l238_238189

theorem probability_sum_is_10 (d1 d2 d3 : ℕ) : 
  (1 ≤ d1 ∧ d1 ≤ 6) ∧ (1 ≤ d2 ∧ d2 ≤ 6) ∧ (1 ≤ d3 ∧ d3 ≤ 6) ∧ (d1 + d2 + d3 = 10) → 
  (((d1 = 1 ∧ d2 = 3 ∧ d3 = 6) ∨ 
    (d1 = 1 ∧ d2 = 4 ∧ d3 = 5) ∨ 
    (d1 = 2 ∧ d2 = 3 ∧ d3 = 5) ∨ 
    (d1 = 2 ∧ d2 = 4 ∧ d3 = 4) ∨ 
    (d1 = 3 ∧ d2 = 3 ∧ d3 = 4) ∨ 
    (d1 = 4 ∧ d2 = 4 ∧ d3 = 2)) 
    ↔ nat.cast (1 / 9)) :=
begin
  sorry,
end

end probability_sum_is_10_l238_238189


namespace relationship_between_f_x1_and_f_x2_l238_238648

variable (f : ℝ → ℝ)
variable (x1 x2 : ℝ)

-- Conditions:
variable (h_even : ∀ x, f x = f (-x))          -- f is even
variable (h_increasing : ∀ a b, 0 < a → a < b → f a < f b)  -- f is increasing on (0, +∞)
variable (h_x1_neg : x1 < 0)                   -- x1 < 0
variable (h_x2_pos : 0 < x2)                   -- x2 > 0
variable (h_abs : |x1| > |x2|)                 -- |x1| > |x2|

-- Goal:
theorem relationship_between_f_x1_and_f_x2 : f x1 > f x2 :=
by
  sorry

end relationship_between_f_x1_and_f_x2_l238_238648


namespace reduced_price_is_3_84_l238_238573

noncomputable def reduced_price_per_dozen (original_price : ℝ) (bananas_for_40 : ℕ) : ℝ := 
  let reduced_price := 0.6 * original_price
  let total_bananas := bananas_for_40 + 50
  let price_per_banana := 40 / total_bananas
  12 * price_per_banana

theorem reduced_price_is_3_84 
  (original_price : ℝ) 
  (bananas_for_40 : ℕ) 
  (h₁ : 40 = bananas_for_40 * original_price) 
  (h₂ : bananas_for_40 = 75) 
    : reduced_price_per_dozen original_price bananas_for_40 = 3.84 :=
sorry

end reduced_price_is_3_84_l238_238573


namespace value_of_f_at_2_l238_238252

theorem value_of_f_at_2 (a b : ℝ) (h : (a + -b + 8) = (9 * a + 3 * b + 8)) :
  (a * 2 ^ 2 + b * 2 + 8) = 8 := 
by
  sorry

end value_of_f_at_2_l238_238252


namespace total_cost_of_shirts_is_24_l238_238374

-- Definitions based on conditions
def cost_first_shirt : ℕ := 15
def cost_difference : ℕ := 6
def cost_second_shirt : ℕ := cost_first_shirt - cost_difference

-- The proof problem statement: Calculate total cost given the conditions
theorem total_cost_of_shirts_is_24 : cost_first_shirt + cost_second_shshirt = 24 :=
by
  sorry

end total_cost_of_shirts_is_24_l238_238374


namespace minimum_value_a_2b_3c_l238_238617

theorem minimum_value_a_2b_3c (a b c : ℝ)
  (h : ∀ x y : ℝ, x + 2*y - 3 ≤ a*x + b*y + c ∧ a*x + b*y + c ≤ x + 2*y + 3) :
  (a + 2*b - 3*c) = -4 :=
sorry

end minimum_value_a_2b_3c_l238_238617


namespace combination_add_l238_238257

def combination (n m : ℕ) : ℕ := n.choose m

theorem combination_add {n : ℕ} (h1 : 4 ≤ 9) (h2 : 5 ≤ 9) :
  combination 9 4 + combination 9 5 = combination 10 5 := by
  sorry

end combination_add_l238_238257


namespace fencing_cost_l238_238423

noncomputable def pi_approx : ℝ := 3.14159

theorem fencing_cost 
  (d : ℝ) (r : ℝ)
  (h_d : d = 20) 
  (h_r : r = 1.50) :
  abs (r * pi_approx * d - 94.25) < 1 :=
by
  -- Proof omitted
  sorry

end fencing_cost_l238_238423


namespace part1_solution_set_part2_range_of_a_l238_238940

-- Defining the function f(x) under given conditions
def f (x a : ℝ) : ℝ := |x - a^2| + |x - (2 * a + 1)|

-- Part 1: Determine the solution set for the inequality when a = 2
theorem part1_solution_set (x : ℝ) : (f x 2) ≥ 4 ↔ x ≤ 3/2 ∨ x ≥ 11/2 := by
  sorry

-- Part 2: Determine the range of values for a given the inequality
theorem part2_range_of_a (a : ℝ) : (∀ x : ℝ, f x a ≥ 4) ↔ a ≤ -1 ∨ a ≥ 3 := by
  sorry

end part1_solution_set_part2_range_of_a_l238_238940


namespace cos_300_eq_half_l238_238811

theorem cos_300_eq_half : Real.cos (300 * Real.pi / 180) = 1 / 2 := 
by sorry

end cos_300_eq_half_l238_238811


namespace best_fitting_model_l238_238959

theorem best_fitting_model (R2_1 R2_2 R2_3 R2_4 : ℝ)
  (h1 : R2_1 = 0.25) 
  (h2 : R2_2 = 0.50) 
  (h3 : R2_3 = 0.80) 
  (h4 : R2_4 = 0.98) : 
  (R2_4 = max (max R2_1 (max R2_2 R2_3)) R2_4) :=
by
  sorry

end best_fitting_model_l238_238959


namespace farmer_planting_problem_l238_238568

theorem farmer_planting_problem (total_acres : ℕ) (flax_acres : ℕ) (sunflower_acres : ℕ)
  (h1 : total_acres = 240)
  (h2 : flax_acres = 80)
  (h3 : sunflower_acres = total_acres - flax_acres) :
  sunflower_acres - flax_acres = 80 := by
  sorry

end farmer_planting_problem_l238_238568


namespace minimize_total_resistance_l238_238148

variable (a1 a2 a3 a4 a5 a6 : ℝ)
variable (h : a1 > a2 ∧ a2 > a3 ∧ a3 > a4 ∧ a4 > a5 ∧ a5 > a6)

/-- Theorem: Given resistances a1, a2, a3, a4, a5, a6 such that a1 > a2 > a3 > a4 > a5 > a6, 
arranging them in the sequence a1 > a2 > a3 > a4 > a5 > a6 minimizes the total resistance
for the assembled component. -/
theorem minimize_total_resistance : 
  True := 
sorry

end minimize_total_resistance_l238_238148


namespace valid_start_days_for_equal_tuesdays_fridays_l238_238762

-- Define the structure of weekdays
inductive Weekday : Type
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday
deriving DecidableEq, Repr

open Weekday

-- The structure representing the problem conditions
structure Month30Days where
  days : Fin 30 → Weekday

-- A helper function that calculates the weekdays count in a range of days
def count_days (f : Weekday → Bool) (month : Month30Days) : Nat :=
  (Finset.univ.filter (λ i, f (month.days i))).card

def equal_tuesdays_fridays (month : Month30Days) : Prop :=
  count_days (λ d, d = Tuesday) month = count_days (λ d, d = Friday) month

-- Defining the theorem that states the number of valid starting days
theorem valid_start_days_for_equal_tuesdays_fridays : 
  {d : Weekday // ∃ (month : Month30Days), month.days ⟨0⟩ = d ∧ equal_tuesdays_fridays month}.card = 2 := 
sorry

end valid_start_days_for_equal_tuesdays_fridays_l238_238762


namespace correct_equation_l238_238558

theorem correct_equation (x : ℝ) : (-x^2)^2 = x^4 := by sorry

end correct_equation_l238_238558


namespace chord_length_l238_238671

theorem chord_length (t : ℝ) :
  (∃ x y, x = 1 + 2 * t ∧ y = 2 + t ∧ x ^ 2 + y ^ 2 = 9) →
  ((1.8 - (-3)) ^ 2 + (2.4 - 0) ^ 2 = (12 / 5 * Real.sqrt 5) ^ 2) :=
by
  sorry

end chord_length_l238_238671


namespace fractional_to_decimal_l238_238064

theorem fractional_to_decimal : (7 / 16 : ℚ) = 0.4375 := by
  sorry

end fractional_to_decimal_l238_238064


namespace increase_to_restore_l238_238401

noncomputable def percentage_increase_to_restore (P : ℝ) : ℝ :=
  let reduced_price := 0.9 * P
  let restore_factor := P / reduced_price
  (restore_factor - 1) * 100

theorem increase_to_restore :
  percentage_increase_to_restore 100 = 100 / 9 :=
by
  sorry

end increase_to_restore_l238_238401


namespace ratio_third_to_first_l238_238267

theorem ratio_third_to_first (F S T : ℕ) (h1 : F = 33) (h2 : S = 4 * F) (h3 : (F + S + T) / 3 = 77) :
  T / F = 2 :=
by
  sorry

end ratio_third_to_first_l238_238267


namespace range_of_a_l238_238327

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then a + |x - 2|
  else x^2 - 2 * a * x + 2 * a

theorem range_of_a (a : ℝ) (x : ℝ) (h : ∀ x : ℝ, f a x ≥ 0) : -1 ≤ a ∧ a ≤ 2 :=
sorry

end range_of_a_l238_238327


namespace sum_first_15_odd_starting_from_5_l238_238555

-- Definitions based on conditions in the problem.
def a : ℕ := 5    -- First term of the sequence is 5
def n : ℕ := 15   -- Number of terms is 15

-- Define the sequence of odd numbers starting from 5
def oddSeq (i : ℕ) : ℕ := a + 2 * i

-- Define the sum of the first n terms of this sequence
def sumOddSeq : ℕ := ∑ i in Finset.range n, oddSeq i

-- Key statement to prove that the sum of the sequence is 255
theorem sum_first_15_odd_starting_from_5 : sumOddSeq = 255 := by
  sorry

end sum_first_15_odd_starting_from_5_l238_238555


namespace part1_solution_set_part2_range_of_a_l238_238931

noncomputable def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1_solution_set (a : ℝ) (a_eq_2 : a = 2) : 
  {x : ℝ | f x a ≥ 4} = {x | x ≤ 3 / 2} ∪ {x | x ≥ 11 / 2} :=
by
  sorry

theorem part2_range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 4) → (a ∈ set.Iic (-1) ∪ set.Ici 3) :=
by
  sorry

end part1_solution_set_part2_range_of_a_l238_238931


namespace max_d_n_l238_238675

def sequence_a (n : ℕ) : ℤ := 100 + n^2

def d_n (n : ℕ) : ℤ := Int.gcd (sequence_a n) (sequence_a (n + 1))

theorem max_d_n : ∃ n, d_n n = 401 :=
by
  -- Placeholder for the actual proof
  sorry

end max_d_n_l238_238675


namespace part1_solution_set_part2_range_of_a_l238_238870

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1_solution_set (x : ℝ) : 
  (f x 2 ≥ 4) ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
sorry

theorem part2_range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 4) → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end part1_solution_set_part2_range_of_a_l238_238870


namespace maximum_value_of_sum_l238_238954

variables (x y : ℝ)

def s : ℝ := x + y

theorem maximum_value_of_sum (h : s ≤ 9) : s = 9 :=
sorry

end maximum_value_of_sum_l238_238954


namespace greatest_m_div_36_and_7_l238_238100

def reverse_digits (m : ℕ) : ℕ :=
  let d1 := (m / 1000) % 10
  let d2 := (m / 100) % 10
  let d3 := (m / 10) % 10
  let d4 := m % 10
  d4 * 1000 + d3 * 100 + d2 * 10 + d1

theorem greatest_m_div_36_and_7
  (m : ℕ) (n : ℕ := reverse_digits m)
  (h1 : 1000 ≤ m ∧ m < 10000)
  (h2 : 1000 ≤ n ∧ n < 10000)
  (h3 : 36 ∣ m ∧ 36 ∣ n)
  (h4 : 7 ∣ m) :
  m = 9828 := 
sorry

end greatest_m_div_36_and_7_l238_238100


namespace calculate_expression_l238_238591

theorem calculate_expression : -1^4 * 8 - 2^3 / (-4) * (-7 + 5) = -12 := 
by 
  /-
  In Lean, we typically perform arithmetic simplifications step by step;
  however, for the purpose of this example, only stating the goal:
  -/
  sorry

end calculate_expression_l238_238591


namespace determine_c_absolute_value_l238_238529

theorem determine_c_absolute_value (a b c : ℤ) 
  (h_gcd : Int.gcd (Int.gcd a b) c = 1)
  (h_eq : a * (3 + 1*Complex.i)^4 + b * (3 + 1*Complex.i)^3 + c * (3 + 1*Complex.i)^2 + b * (3 + 1*Complex.i) + a = 0) :
  |c| = 111 :=
sorry

end determine_c_absolute_value_l238_238529


namespace exists_triangle_with_edges_l238_238328

variable {A B C D: Type}
variables (AB AC AD BC BD CD : ℝ)
variables (tetrahedron : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)

def x := AB * CD
def y := AC * BD
def z := AD * BC

theorem exists_triangle_with_edges :
  ∃ (x y z : ℝ), 
  ∃ (A B C D: Type),
  ∃ (AB AC AD BC BD CD : ℝ) (tetrahedron : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D),
  x = AB * CD ∧ y = AC * BD ∧ z = AD * BC → 
  (x + y > z ∧ y + z > x ∧ z + x > y) :=
by
  sorry

end exists_triangle_with_edges_l238_238328


namespace mickey_horses_per_week_l238_238128

variable (days_in_week : ℕ := 7)
variable (minnie_horses_per_day : ℕ := days_in_week + 3)
variable (mickey_horses_per_day : ℕ := 2 * minnie_horses_per_day - 6)

theorem mickey_horses_per_week : mickey_horses_per_day * days_in_week = 98 := by
  sorry

end mickey_horses_per_week_l238_238128


namespace together_complete_days_l238_238562

-- Define the work rates of x and y
def work_rate_x := (1 : ℚ) / 30
def work_rate_y := (1 : ℚ) / 45

-- Define the combined work rate when x and y work together
def combined_work_rate := work_rate_x + work_rate_y

-- Define the number of days to complete the work together
def days_to_complete_work := 1 / combined_work_rate

-- The theorem we want to prove
theorem together_complete_days : days_to_complete_work = 18 := by
  sorry

end together_complete_days_l238_238562


namespace sheets_of_paper_needed_l238_238507

theorem sheets_of_paper_needed
  (books : ℕ)
  (pages_per_book : ℕ)
  (double_sided : Bool)
  (pages_per_side : ℕ)
  (total_sheets : ℕ) :
  books = 2 ∧
  pages_per_book = 600 ∧
  double_sided = true ∧
  pages_per_side = 4 →
  total_sheets = 150 :=
begin
  -- Define the total number of pages
  let total_pages := books * pages_per_book,
  -- Define the pages per sheet
  let pages_per_sheet := pages_per_side * 2,
  -- Calculate total sheets
  let required_sheets := total_pages / pages_per_sheet,
  -- Show equivalence to the desired total sheets
  assume h,
  have : total_sheets = required_sheets,
  { sorry }, -- Proof to be provided
  rw this,
  rw ←h.right.right.right
end

end sheets_of_paper_needed_l238_238507


namespace equalize_rice_move_amount_l238_238435

open Real

noncomputable def containerA_kg : Real := 12
noncomputable def containerA_g : Real := 400
noncomputable def containerB_g : Real := 7600

noncomputable def total_rice_in_A_g : Real := containerA_kg * 1000 + containerA_g
noncomputable def total_rice_in_A_and_B_g : Real := total_rice_in_A_g + containerB_g
noncomputable def equalized_rice_per_container_g : Real := total_rice_in_A_and_B_g / 2

noncomputable def amount_to_move_g : Real := total_rice_in_A_g - equalized_rice_per_container_g
noncomputable def amount_to_move_kg : Real := amount_to_move_g / 1000

theorem equalize_rice_move_amount :
  amount_to_move_kg = 2.4 :=
by
  sorry

end equalize_rice_move_amount_l238_238435


namespace mickey_horses_per_week_l238_238122

def days_in_week : ℕ := 7

def horses_minnie_per_day : ℕ := days_in_week + 3

def horses_twice_minnie_per_day : ℕ := 2 * horses_minnie_per_day

def horses_mickey_per_day : ℕ := horses_twice_minnie_per_day - 6

def horses_mickey_per_week : ℕ := days_in_week * horses_mickey_per_day

theorem mickey_horses_per_week : horses_mickey_per_week = 98 := sorry

end mickey_horses_per_week_l238_238122


namespace part1_part2_l238_238616

open Real

variables (α : ℝ) (A : (ℝ × ℝ)) (B : (ℝ × ℝ)) (C : (ℝ × ℝ))

def points_coordinates : Prop :=
A = (3, 0) ∧ B = (0, 3) ∧ C = (cos α, sin α) ∧ π / 2 < α ∧ α < 3 * π / 2

theorem part1 (h : points_coordinates α A B C) (h1 : dist (3, 0) (cos α, sin α) = dist (0, 3) (cos α, sin α)) : 
  α = 5 * π / 4 :=
sorry

theorem part2 (h : points_coordinates α A B C) (h2 : ((cos α - 3) * cos α + (sin α) * (sin α - 3)) = -1) : 
  (2 * sin α * sin α + sin (2 * α)) / (1 + tan α) = -5 / 9 :=
sorry

end part1_part2_l238_238616


namespace find_x_y_z_sum_l238_238540

theorem find_x_y_z_sum :
  ∃ (x y z : ℝ), 
    x^2 + 27 = -8 * y + 10 * z ∧
    y^2 + 196 = 18 * z + 13 * x ∧
    z^2 + 119 = -3 * x + 30 * y ∧
    x + 3 * y + 5 * z = 127.5 :=
sorry

end find_x_y_z_sum_l238_238540


namespace intersection_M_N_l238_238499

-- Define the set M and N
def M : Set ℝ := { x | x^2 ≤ 1 }
def N : Set ℝ := {-2, 0, 1}

-- Theorem stating that the intersection of M and N is {0, 1}
theorem intersection_M_N : M ∩ N = {0, 1} :=
by
  sorry

end intersection_M_N_l238_238499


namespace part1_part2_l238_238883

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1 (x : ℝ) : 
  ∀ x, f x 2 ≥ 4 ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
by sorry

theorem part2 (x a : ℝ) :
  f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
by sorry

end part1_part2_l238_238883


namespace greatest_possible_sum_of_two_consecutive_integers_lt_500_l238_238711

theorem greatest_possible_sum_of_two_consecutive_integers_lt_500 (n : ℕ) (h : n * (n + 1) < 500) : n + (n + 1) ≤ 43 := by
  sorry

end greatest_possible_sum_of_two_consecutive_integers_lt_500_l238_238711


namespace fraction_to_decimal_l238_238026

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l238_238026


namespace circle_S_radius_properties_l238_238504

theorem circle_S_radius_properties :
  let DE := 120
  let DF := 120
  let EF := 68
  let R_radius := 20
  let S_radius := 52 - 6 * Real.sqrt 35
  let m := 52
  let n := 6
  let k := 35
  m + n * k = 262 := by
  sorry

end circle_S_radius_properties_l238_238504


namespace least_multiple_72_112_199_is_310_l238_238280

theorem least_multiple_72_112_199_is_310 :
  ∃ k : ℕ, (112 ∣ k * 72) ∧ (199 ∣ k * 72) ∧ k = 310 := 
by
  sorry

end least_multiple_72_112_199_is_310_l238_238280


namespace part1_part2_l238_238858

noncomputable def f (a x : ℝ) : ℝ := (Real.exp x) - a * x ^ 2 - x

theorem part1 {a : ℝ} : (∀ x y: ℝ, x < y → f a x ≤ f a y) ↔ (a = 1 / 2) :=
sorry

theorem part2 {a : ℝ} (h1 : a > 1 / 2):
  ∃ (x1 x2 : ℝ), (x1 < x2) ∧ (f a x2 < 1 + (Real.sin x2 - x2) / 2) :=
sorry

end part1_part2_l238_238858


namespace symmetric_points_origin_l238_238167

theorem symmetric_points_origin {a b : ℝ} (h₁ : a = -(-4)) (h₂ : b = -(3)) : a - b = 7 :=
by 
  -- since this is a statement template, the proof is omitted
  sorry

end symmetric_points_origin_l238_238167


namespace mickey_horses_per_week_l238_238127

variable (days_in_week : ℕ := 7)
variable (minnie_horses_per_day : ℕ := days_in_week + 3)
variable (mickey_horses_per_day : ℕ := 2 * minnie_horses_per_day - 6)

theorem mickey_horses_per_week : mickey_horses_per_day * days_in_week = 98 := by
  sorry

end mickey_horses_per_week_l238_238127


namespace sum_of_midpoints_l238_238542

theorem sum_of_midpoints {a b c : ℝ} (h : a + b + c = 15) : 
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 := 
by
  sorry

end sum_of_midpoints_l238_238542


namespace part1_part2_l238_238619

def f (x : ℝ) : ℝ := abs (x - 2)

theorem part1 (x : ℝ) : f x > 4 - abs (x + 1) ↔ x < -3 / 2 ∨ x > 5 / 2 := 
sorry

theorem part2 (a b : ℝ) (ha : 0 < a ∧ a < 1/2) (hb : 0 < b ∧ b < 1/2)
  (h : f (1 / a) + f (2 / b) = 10) : a + b / 2 ≥ 2 / 7 := 
sorry

end part1_part2_l238_238619


namespace find_t_l238_238662

-- Define the roots and basic properties
variables (a b c : ℝ)
variables (r s t : ℝ)

-- Define conditions from the first cubic equation
def first_eq_roots : Prop :=
  a + b + c = -5 ∧ a * b * c = 13

-- Define conditions from the second cubic equation with shifted roots
def second_eq_roots : Prop :=
  t = -(a * b * c + a * b + a * c + b * c + a + b + c + 1)

-- The theorem stating the value of t
theorem find_t (h₁ : first_eq_roots a b c) (h₂ : second_eq_roots a b c t) : t = -15 :=
sorry

end find_t_l238_238662


namespace fraction_decimal_equivalent_l238_238006

theorem fraction_decimal_equivalent : (7 / 16 : ℝ) = 0.4375 :=
by
  sorry

end fraction_decimal_equivalent_l238_238006


namespace harly_initial_dogs_l238_238155

theorem harly_initial_dogs (x : ℝ) 
  (h1 : 0.40 * x + 0.60 * x + 5 = 53) : 
  x = 80 := 
by 
  sorry

end harly_initial_dogs_l238_238155


namespace proposition_check_l238_238102

variable (P : ℕ → Prop)

theorem proposition_check 
  (h : ∀ k : ℕ, ¬ P (k + 1) → ¬ P k)
  (h2012 : P 2012) : P 2013 :=
by
  sorry

end proposition_check_l238_238102


namespace minimum_apples_to_guarantee_18_one_color_l238_238564

theorem minimum_apples_to_guarantee_18_one_color :
  let red := 32
  let green := 24
  let yellow := 22
  let blue := 15
  let orange := 14
  ∀ n, (n >= 81) →
  (∃ red_picked green_picked yellow_picked blue_picked orange_picked : ℕ,
    red_picked + green_picked + yellow_picked + blue_picked + orange_picked = n
    ∧ red_picked ≤ red ∧ green_picked ≤ green ∧ yellow_picked ≤ yellow ∧ blue_picked ≤ blue ∧ orange_picked ≤ orange
    ∧ (red_picked = 18 ∨ green_picked = 18 ∨ yellow_picked = 18 ∨ blue_picked = 18 ∨ orange_picked = 18)) :=
by {
  -- The proof is omitted for now.
  sorry
}

end minimum_apples_to_guarantee_18_one_color_l238_238564


namespace pounds_of_fudge_sold_l238_238285

variable (F : ℝ)
variable (price_fudge price_truffles price_pretzels total_revenue : ℝ)

def conditions := 
  price_fudge = 2.50 ∧
  price_truffles = 60 * 1.50 ∧
  price_pretzels = 36 * 2.00 ∧
  total_revenue = 212 ∧
  total_revenue = (price_fudge * F) + price_truffles + price_pretzels

theorem pounds_of_fudge_sold (F : ℝ) (price_fudge price_truffles price_pretzels total_revenue : ℝ) 
  (h : conditions F price_fudge price_truffles price_pretzels total_revenue ) :
  F = 20 :=
by
  sorry

end pounds_of_fudge_sold_l238_238285


namespace part1_solution_set_part2_range_of_a_l238_238930

noncomputable def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1_solution_set (a : ℝ) (a_eq_2 : a = 2) : 
  {x : ℝ | f x a ≥ 4} = {x | x ≤ 3 / 2} ∪ {x | x ≥ 11 / 2} :=
by
  sorry

theorem part2_range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 4) → (a ∈ set.Iic (-1) ∪ set.Ici 3) :=
by
  sorry

end part1_solution_set_part2_range_of_a_l238_238930


namespace fraction_decimal_equivalent_l238_238003

theorem fraction_decimal_equivalent : (7 / 16 : ℝ) = 0.4375 :=
by
  sorry

end fraction_decimal_equivalent_l238_238003


namespace hyperbola_standard_form_l238_238264

noncomputable def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 8 - y^2 / 2 = 1

theorem hyperbola_standard_form :
  (foci_of_ellipse : Set (ℝ × ℝ)) 
  (asymptotes_hyperbola : Set ℝ) 
  (3 * x^2 + 13 * y^2 = 39) 
  (foci_of_ellipse = {(sqrt 10, 0), (-sqrt 10, 0)}) 
  (asymptotes_hyperbola = {y = x / 2, y = - x / 2}) :
  ∀ x y, hyperbola_equation x y := 
-- Proof omitted
  sorry

end hyperbola_standard_form_l238_238264


namespace part1_solution_set_part2_values_of_a_l238_238865

-- Parameters and definitions for the function and conditions
def f (x a : ℝ) := |x - a^2| + |x - (2*a + 1)|

-- Part (1) When a = 2, the inequality's solution set
theorem part1_solution_set (x : ℝ) : f x 2 ≥ 4 ↔ (x ≤ 3/2 ∨ x ≥ 11/2) := 
sorry

-- Part (2) Range of values for a given f(x, a) ≥ 4
theorem part2_values_of_a (a : ℝ) : 
∀ x, f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end part1_solution_set_part2_values_of_a_l238_238865


namespace probability_sum_is_10_l238_238210

theorem probability_sum_is_10 : 
  let outcomes := [(a, b, c) | a ∈ [1, 2, 3, 4, 5, 6], b ∈ [1, 2, 3, 4, 5, 6], c ∈ [1, 2, 3, 4, 5, 6]],
      favorable_outcomes := [(a, b, c) | (a, b, c) ∈ outcomes, a + b + c = 10] in
  (favorable_outcomes.length : ℚ) / outcomes.length = 1 / 9 := by
  sorry

end probability_sum_is_10_l238_238210


namespace fraction_to_decimal_l238_238014

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l238_238014


namespace joe_two_different_fruits_l238_238967

-- Define the set of fruits
inductive Fruit
| apple | orange | banana | grape

open Fruit

-- Define the probability space for one selection
def fruit_prob_space : MeasureSpace Fruit :=
  probability_space {Finset.univ}

-- Define the event Joe eats the same fruit for all meals
def same_fruit_for_all_meals : set (Fruit × Fruit × Fruit × Fruit) :=
  {x | x.1 = x.2 ∧ x.2 = x.3 ∧ x.3 = x.4}

-- Calculate the probability of Joe eating the same fruit for all meals
noncomputable def prob_same_fruit : ℚ :=
  (1 / 4) ^ 4 * 4

-- Define the probability of eating at least two different kinds of fruit
noncomputable def prob_at_least_two_different_fruits : ℚ :=
  1 - prob_same_fruit

theorem joe_two_different_fruits :
  prob_at_least_two_different_fruits = 63 / 64 :=
by
  unfold prob_at_least_two_different_fruits
  unfold prob_same_fruit
  sorry

end joe_two_different_fruits_l238_238967


namespace part_1_solution_set_part_2_a_range_l238_238891

-- Define the function f
def f (x a : ℝ) := |x - a^2| + |x - 2 * a + 1|

-- Part (1)
theorem part_1_solution_set (x : ℝ) : {x | x ≤ 3 / 2 ∨ x ≥ 11 / 2} = 
  {x | f x 2 ≥ 4} :=
sorry

-- Part (2)
theorem part_2_a_range (a : ℝ) : 
  {a | (a - 1)^2 ≥ 4} = {a | a ≤ -1 ∨ a ≥ 3} :=
sorry

end part_1_solution_set_part_2_a_range_l238_238891


namespace prob1_prob2_max_area_prob3_circle_diameter_l238_238144

-- Definitions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0
def line_through_center (x y : ℝ) : Prop := x - y - 3 = 0
def line_eq (m : ℝ) (x y : ℝ) : Prop := y = x + m

-- Problem 1: Line passes through the center of the circle
theorem prob1 (x y : ℝ) : line_through_center x y ↔ circle_eq x y :=
sorry

-- Problem 2: Maximum area of triangle CAB
theorem prob2_max_area (x y : ℝ) (m : ℝ) : line_eq m x y → (m = 0 ∨ m = -6) :=
sorry

-- Problem 3: Circle with diameter AB passes through origin
theorem prob3_circle_diameter (x y : ℝ) (m : ℝ) : line_eq m x y → (m = 1 ∨ m = -4) :=
sorry

end prob1_prob2_max_area_prob3_circle_diameter_l238_238144


namespace unique_seating_arrangements_l238_238774

/--
There are five couples including Charlie and his wife. The five men sit on the 
inner circle and each man's wife sits directly opposite him on the outer circle.
Prove that the number of unique seating arrangements where each man has another 
man seated directly to his right on the inner circle, counting all seat 
rotations as the same but not considering inner to outer flips as different, is 30.
-/
theorem unique_seating_arrangements : 
  ∃ (n : ℕ), n = 30 := 
sorry

end unique_seating_arrangements_l238_238774


namespace chord_length_l238_238670

theorem chord_length (x y t : ℝ) (h₁ : x = 1 + 2 * t) (h₂ : y = 2 + t) (h_circle : x^2 + y^2 = 9) : 
  ∃ l, l = 12 / 5 * Real.sqrt 5 := 
sorry

end chord_length_l238_238670


namespace infinite_solutions_ax2_by2_eq_z3_l238_238645

theorem infinite_solutions_ax2_by2_eq_z3 
  (a b : ℤ) 
  (coprime_ab : Int.gcd a b = 1) :
  ∃ (x y z : ℤ), (∀ n : ℤ, ∃ (x y z : ℤ), a * x^2 + b * y^2 = z^3 
  ∧ Int.gcd x y = 1) := 
sorry

end infinite_solutions_ax2_by2_eq_z3_l238_238645


namespace part1_part2_l238_238880

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1 (x : ℝ) : 
  ∀ x, f x 2 ≥ 4 ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
by sorry

theorem part2 (x a : ℝ) :
  f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
by sorry

end part1_part2_l238_238880


namespace minimum_value_of_a_plus_b_l238_238955

noncomputable def f (x : ℝ) := Real.log x - (1 / x)
noncomputable def f' (x : ℝ) := 1 / x + 1 / (x^2)

theorem minimum_value_of_a_plus_b (a b m : ℝ) (h1 : a = 1 / m + 1 / (m^2)) 
  (h2 : b = Real.log m - 2 / m - 1) : a + b = -1 :=
by
  sorry

end minimum_value_of_a_plus_b_l238_238955


namespace pairs_with_green_shirts_l238_238366

theorem pairs_with_green_shirts (r g t p rr_pairs gg_pairs : ℕ)
  (h1 : r = 60)
  (h2 : g = 90)
  (h3 : t = 150)
  (h4 : p = 75)
  (h5 : rr_pairs = 28)
  : gg_pairs = 43 := 
sorry

end pairs_with_green_shirts_l238_238366


namespace probability_of_sum_greater_than_15_l238_238268

-- Definition of the dice and outcomes
def total_outcomes : ℕ := 6 * 6 * 6
def favorable_outcomes : ℕ := 10

-- Probability calculation
def probability_sum_gt_15 : ℚ := favorable_outcomes / total_outcomes

-- Theorem to be proven
theorem probability_of_sum_greater_than_15 : probability_sum_gt_15 = 5 / 108 := by
  sorry

end probability_of_sum_greater_than_15_l238_238268


namespace dice_sum_10_probability_l238_238172

open Probability

noncomputable def probability_sum_10 : ℚ := 
  let total_outcomes := 216
  let favorable_outcomes := 21
  favorable_outcomes / total_outcomes

theorem dice_sum_10_probability :
  probability (λ (x : Fin 6 × Fin 6 × Fin 6), x.1.val + x.2.val + x.3.val + 3 = 10) = probability_sum_10 := 
sorry

end dice_sum_10_probability_l238_238172


namespace find_f_2016_minus_f_2015_l238_238484

-- Definitions for the given conditions

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def periodic_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 2) = -f x

def specific_values (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, (0 < x ∧ x ≤ 1) → f x = 2^x

-- Main theorem statement
theorem find_f_2016_minus_f_2015 {f : ℝ → ℝ} 
    (H1 : odd_function f) 
    (H2 : periodic_function f)
    (H3 : specific_values f)
    : f 2016 - f 2015 = 2 := 
sorry

end find_f_2016_minus_f_2015_l238_238484


namespace greatest_sum_consecutive_integers_product_less_than_500_l238_238699

theorem greatest_sum_consecutive_integers_product_less_than_500 :
  ∃ n : ℕ, n * (n + 1) < 500 ∧ (n + (n + 1) = 43) := sorry

end greatest_sum_consecutive_integers_product_less_than_500_l238_238699


namespace part1_solution_set_part2_range_of_a_l238_238911

-- Define the function f for a general a
def f (x a : ℝ) : ℝ := |x - a^2| + |x - (2 * a) + 1|

-- Part 1: When a = 2
theorem part1_solution_set (x : ℝ) (h : f x 2 ≥ 4) : x ≤ 3/2 ∨ x ≥ 11/2 :=
  sorry

-- Part 2: Range of values for a
theorem part2_range_of_a (a : ℝ) (h : ∀ x, f x a ≥ 4) : a ≤ -1 ∨ a ≥ 3 :=
  sorry

end part1_solution_set_part2_range_of_a_l238_238911


namespace all_values_equal_l238_238588

noncomputable def f : ℤ × ℤ → ℕ :=
sorry

theorem all_values_equal (f : ℤ × ℤ → ℕ)
  (h_pos : ∀ p, 0 < f p)
  (h_mean : ∀ x y, f (x, y) = 1/4 * (f (x+1, y) + f (x-1, y) + f (x, y+1) + f (x, y-1))) :
  ∀ (x1 y1 x2 y2 : ℤ), f (x1, y1) = f (x2, y2) := 
sorry

end all_values_equal_l238_238588


namespace part1_solution_set_part2_range_of_a_l238_238909

-- Define the function f for a general a
def f (x a : ℝ) : ℝ := |x - a^2| + |x - (2 * a) + 1|

-- Part 1: When a = 2
theorem part1_solution_set (x : ℝ) (h : f x 2 ≥ 4) : x ≤ 3/2 ∨ x ≥ 11/2 :=
  sorry

-- Part 2: Range of values for a
theorem part2_range_of_a (a : ℝ) (h : ∀ x, f x a ≥ 4) : a ≤ -1 ∨ a ≥ 3 :=
  sorry

end part1_solution_set_part2_range_of_a_l238_238909


namespace jaclyn_constant_term_l238_238977

variable {R : Type*} [CommRing R] (P Q : Polynomial R)

theorem jaclyn_constant_term (hP : P.leadingCoeff = 1) (hQ : Q.leadingCoeff = 1)
  (deg_P : P.degree = 4) (deg_Q : Q.degree = 4)
  (constant_terms_eq : P.coeff 0 = Q.coeff 0)
  (coeff_z_eq : P.coeff 1 = Q.coeff 1)
  (product_eq : P * Q = Polynomial.C 1 * 
    Polynomial.C 1 * Polynomial.C 1 * Polynomial.C (-1) *
    Polynomial.C 1) :
  Jaclyn's_constant_term = 3 :=
sorry

end jaclyn_constant_term_l238_238977


namespace total_cost_of_shirts_l238_238371

theorem total_cost_of_shirts 
    (first_shirt_cost : ℤ)
    (second_shirt_cost : ℤ)
    (h1 : first_shirt_cost = 15)
    (h2 : first_shirt_cost = second_shirt_cost + 6) : 
    first_shirt_cost + second_shirt_cost = 24 := 
by
  sorry

end total_cost_of_shirts_l238_238371


namespace fraction_to_decimal_l238_238069

theorem fraction_to_decimal : (7 : ℝ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l238_238069


namespace fraction_to_decimal_l238_238034

theorem fraction_to_decimal : (7 : ℝ) / 16 = 0.4375 := 
by
  sorry

end fraction_to_decimal_l238_238034


namespace greatest_possible_sum_of_two_consecutive_integers_lt_500_l238_238709

theorem greatest_possible_sum_of_two_consecutive_integers_lt_500 (n : ℕ) (h : n * (n + 1) < 500) : n + (n + 1) ≤ 43 := by
  sorry

end greatest_possible_sum_of_two_consecutive_integers_lt_500_l238_238709


namespace sum_first_15_odd_from_5_l238_238553

theorem sum_first_15_odd_from_5 : 
  let a₁ := 5 
  let d := 2 
  let n := 15 
  let a₁₅ := a₁ + (n - 1) * d 
  let S := n * (a₁ + a₁₅) / 2 
  S = 285 := by 
  sorry

end sum_first_15_odd_from_5_l238_238553


namespace weight_of_piece_l238_238659

theorem weight_of_piece (x d : ℝ) (h1 : x - d = 300) (h2 : x + d = 500) : x = 400 := 
by
  sorry

end weight_of_piece_l238_238659


namespace find_2023rd_digit_of_11_div_13_l238_238134

noncomputable def decimal_expansion_repeating (n d : Nat) : List Nat := sorry

noncomputable def decimal_expansion_digit (n d pos : Nat) : Nat :=
  let repeating_block := decimal_expansion_repeating n d
  repeating_block.get! ((pos - 1) % repeating_block.length)

theorem find_2023rd_digit_of_11_div_13 :
  decimal_expansion_digit 11 13 2023 = 8 := by
  sorry

end find_2023rd_digit_of_11_div_13_l238_238134


namespace compute_value_l238_238452

-- Definitions based on problem conditions
def x : ℤ := (150 - 100 + 1) * (100 + 150) / 2  -- Sum of integers from 100 to 150

def y : ℤ := (150 - 100) / 2 + 1  -- Number of even integers from 100 to 150

def z : ℤ := 0  -- Product of odd integers from 100 to 150 (including even numbers makes the product 0)

-- The theorem to prove
theorem compute_value : x + y - z = 6401 :=
by
  sorry

end compute_value_l238_238452


namespace correct_pair_has_integer_distance_l238_238483

-- Define the pairs of (x, y)
def pairs : List (ℕ × ℕ) :=
  [(88209, 90288), (82098, 89028), (28098, 89082), (90882, 28809)]

-- Define the property: a pair (x, y) has the distance √(x^2 + y^2) as an integer
def is_integer_distance_pair (x y : ℕ) : Prop :=
  ∃ (n : ℕ), n * n = x * x + y * y

-- Translate the problem to the proof: Prove (88209, 90288) satisfies the given property
theorem correct_pair_has_integer_distance :
  is_integer_distance_pair 88209 90288 :=
by
  sorry

end correct_pair_has_integer_distance_l238_238483


namespace minimum_toothpicks_to_remove_l238_238526

-- Conditions
def number_of_toothpicks : ℕ := 60
def largest_triangle_side : ℕ := 3
def smallest_triangle_side : ℕ := 1

-- Problem Statement
theorem minimum_toothpicks_to_remove (toothpicks_total : ℕ) (largest_side : ℕ) (smallest_side : ℕ) 
  (h1 : toothpicks_total = 60) 
  (h2 : largest_side = 3) 
  (h3 : smallest_side = 1) : 
  ∃ n : ℕ, n = 20 := by
  sorry

end minimum_toothpicks_to_remove_l238_238526


namespace spider_legs_total_l238_238639

-- Definitions based on given conditions
def spiders : ℕ := 4
def legs_per_spider : ℕ := 8

-- Theorem statement
theorem spider_legs_total : (spiders * legs_per_spider) = 32 := by
  sorry

end spider_legs_total_l238_238639


namespace extreme_value_at_one_is_minimum_l238_238342

noncomputable def f (x a : ℝ) : ℝ := (x^2 + 1) / (x + a)

theorem extreme_value_at_one_is_minimum (a : ℝ) (h_extreme : ∀ x ≠ (-a), deriv (λ x, (x^2 + 1) / (x + a)) x = 0 → x = 1) :
  ∃ y, f y a ≤ f 1 a ∧ ∀ z, z ≠ y → f y a < f z a :=
begin
  sorry
end

end extreme_value_at_one_is_minimum_l238_238342


namespace greatest_sum_consecutive_integers_product_less_than_500_l238_238703

theorem greatest_sum_consecutive_integers_product_less_than_500 :
  ∃ n : ℕ, n * (n + 1) < 500 ∧ (n + (n + 1) = 43) := sorry

end greatest_sum_consecutive_integers_product_less_than_500_l238_238703


namespace max_coins_as_pleases_max_coins_equally_distributed_l238_238682

-- Part a
theorem max_coins_as_pleases {N : ℕ} (N_warriors : N = 33) (total_coins : ℕ := 240) : 
  ∃ k : ℕ, k ≤ N ∧ (∃ remaining_coins : ℕ, remaining_coins ≤ total_coins ∧ remaining_coins = 31) := 
by
  sorry

-- Part b
theorem max_coins_equally_distributed {N : ℕ} (N_warriors : N = 33) (total_coins : ℕ := 240) : 
  ∃ k : ℕ, k ≤ N ∧ (∃ remaining_coins : ℕ, remaining_coins ≤ total_coins ∧ remaining_coins = 30) := 
by
  sorry

end max_coins_as_pleases_max_coins_equally_distributed_l238_238682


namespace unique_solution_positive_n_l238_238468

theorem unique_solution_positive_n (n : ℝ) : 
  ( ∃ x : ℝ, 4 * x^2 + n * x + 16 = 0 ∧ ∀ y : ℝ, 4 * y^2 + n * y + 16 = 0 → y = x ) → n = 16 := 
by {
  sorry
}

end unique_solution_positive_n_l238_238468


namespace other_factor_computation_l238_238095

theorem other_factor_computation (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ) (e : ℕ) :
  a = 11 → b = 43 → c = 2 → d = 31 → e = 1311 → 33 ∣ 363 →
  a * b * c * d * e = 38428986 :=
by
  intros ha hb hc hd he hdiv
  rw [ha, hb, hc, hd, he]
  -- proof steps go here if required
  sorry

end other_factor_computation_l238_238095


namespace cos_300_eq_half_l238_238817

theorem cos_300_eq_half : Real.cos (300 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end cos_300_eq_half_l238_238817


namespace max_problems_in_7_days_l238_238460

/-- 
  Pasha can solve at most 10 problems in a day. 
  If he solves more than 7 problems on any given day, then for the next two days, he can solve no more than 5 problems each day.
  Prove that the maximum number of problems Pasha can solve in 7 consecutive days is 50.
--/
theorem max_problems_in_7_days :
  ∃ (D : Fin 7 → ℕ), 
    (∀ i, D i ≤ 10) ∧
    (∀ i, D i > 7 → (i + 1 < 7 → D (i + 1) ≤ 5) ∧ (i + 2 < 7 → D (i + 2) ≤ 5)) ∧
    (∑ i in Finset.range 7, D i) = 50 :=
by
  sorry

end max_problems_in_7_days_l238_238460


namespace total_profit_l238_238421

-- Definitions
def investment_a : ℝ := 45000
def investment_b : ℝ := 63000
def investment_c : ℝ := 72000
def c_share : ℝ := 24000

-- Theorem statement
theorem total_profit : (investment_a + investment_b + investment_c) * (c_share / investment_c) = 60000 := by
  sorry

end total_profit_l238_238421


namespace cos_300_eq_half_l238_238800

theorem cos_300_eq_half : Real.cos (300 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end cos_300_eq_half_l238_238800


namespace incorrect_correlation_statement_l238_238419

/--
  The correlation coefficient measures the degree of linear correlation between two variables. 
  The linear correlation coefficient is a quantity whose absolute value is less than 1. 
  Furthermore, the larger its absolute value, the greater the degree of correlation.

  Let r be the sample correlation coefficient.

  We want to prove that the statement "D: |r| ≥ 1, and the closer |r| is to 1, the greater the degree of correlation" 
  is incorrect.
-/
theorem incorrect_correlation_statement (r : ℝ) (h1 : |r| ≤ 1) : ¬ (|r| ≥ 1) :=
by
  -- Proof steps go here
  sorry

end incorrect_correlation_statement_l238_238419


namespace max_kopeyka_coins_l238_238560

def coins (n : Nat) (k : Nat) : Prop :=
  k ≤ n / 4 + 1

theorem max_kopeyka_coins : coins 2001 501 :=
by
  sorry

end max_kopeyka_coins_l238_238560


namespace log_base_3_domain_is_minus_infinity_to_3_l238_238839

noncomputable def log_base_3_domain (x : ℝ) : Prop :=
  3 - x > 0

theorem log_base_3_domain_is_minus_infinity_to_3 :
  ∀ x : ℝ, log_base_3_domain x ↔ x < 3 :=
by
  sorry

end log_base_3_domain_is_minus_infinity_to_3_l238_238839


namespace polynomial_divisibility_l238_238519

open Polynomial

noncomputable def f (n : ℕ) : ℤ[X] :=
  (X + 1) ^ (2 * n + 1) + X ^ (n + 2)

noncomputable def p : ℤ[X] :=
  X ^ 2 + X + 1

theorem polynomial_divisibility (n : ℕ) : p ∣ f n :=
  sorry

end polynomial_divisibility_l238_238519


namespace fraction_to_decimal_l238_238045

theorem fraction_to_decimal :
  (7 / 16 : ℝ) = 0.4375 := 
begin
  -- placeholder for the proof
  sorry
end

end fraction_to_decimal_l238_238045


namespace part_1_solution_set_part_2_a_range_l238_238894

-- Define the function f
def f (x a : ℝ) := |x - a^2| + |x - 2 * a + 1|

-- Part (1)
theorem part_1_solution_set (x : ℝ) : {x | x ≤ 3 / 2 ∨ x ≥ 11 / 2} = 
  {x | f x 2 ≥ 4} :=
sorry

-- Part (2)
theorem part_2_a_range (a : ℝ) : 
  {a | (a - 1)^2 ≥ 4} = {a | a ≤ -1 ∨ a ≥ 3} :=
sorry

end part_1_solution_set_part_2_a_range_l238_238894


namespace cos_300_eq_one_half_l238_238796

theorem cos_300_eq_one_half :
  Real.cos (300 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end cos_300_eq_one_half_l238_238796


namespace greatest_possible_sum_consecutive_product_lt_500_l238_238734

noncomputable def largest_sum_consecutive_product_lt_500 : ℕ :=
  let n := nat.sub ((nat.sqrt 500) + 1) 1 in
  n + (n + 1)

theorem greatest_possible_sum_consecutive_product_lt_500 :
  (∃ (n : ℕ), n * (n + 1) < 500 ∧ largest_sum_consecutive_product_lt_500 = (n + (n + 1))) →
  largest_sum_consecutive_product_lt_500 = 43 := by
  sorry

end greatest_possible_sum_consecutive_product_lt_500_l238_238734


namespace area_of_convex_quadrilateral_l238_238843

open Complex Polynomial

noncomputable def P (z : ℂ) : Polynomial ℂ :=
  Polynomial.Coeff 4 1 -
  (6 * Complex.I + 6) * Polynomial.Coeff 3 1 +
  24 * Complex.I * Polynomial.Coeff 2 1 -
  (18 * Complex.I - 18) * Polynomial.Coeff 1 1 -
  13

theorem area_of_convex_quadrilateral :
  let z1 := (1 : ℂ),
      z2 := Complex.I,
      z3 := (3 + 2 * Complex.I),
      z4 := (2 + 3 * Complex.I)
  in abs (z1 - z3) * abs (z1 - z4) * Complex.cos (Complex.arg (z3 - z1) - Complex.arg (z4 - z1)) = 2 :=
sorry

end area_of_convex_quadrilateral_l238_238843


namespace sum_first_15_odd_integers_from_5_l238_238557

theorem sum_first_15_odd_integers_from_5 :
  let a := 5
  let n := 15
  let d := 2
  let last_term := a + (n - 1) * d
  let S := n * a + (n * (n - 1) * d) / 2
  last_term = 37 ∧ S = 315 := by
  sorry

end sum_first_15_odd_integers_from_5_l238_238557


namespace equation_root_a_plus_b_l238_238534

theorem equation_root_a_plus_b (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b ≥ 0) 
(h_root : (∃ x : ℝ, x > 0 ∧ x^3 - x^2 + 18 * x - 320 = 0 ∧ x = Real.sqrt a - ↑b)) : 
a + b = 25 := by
  sorry

end equation_root_a_plus_b_l238_238534


namespace triangle_problem_l238_238219

/--
Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C, respectively, 
where:
1. b * (sin B - sin C) = a * sin A - c * sin C
2. a = 2 * sqrt 3
3. the area of triangle ABC is 2 * sqrt 3

Prove:
1. A = π / 3
2. The perimeter of triangle ABC is 2 * sqrt 3 + 6
-/
theorem triangle_problem 
  (A B C : ℝ) (a b c : ℝ)
  (h1 : b * (Real.sin B - Real.sin C) = a * Real.sin A - c * Real.sin C)
  (h2 : a = 2 * Real.sqrt 3)
  (h3 : 0.5 * b * c * Real.sin A = 2 * Real.sqrt 3) :
  A = Real.pi / 3 ∧ a + b + c = 2 * Real.sqrt 3 + 6 := 
sorry

end triangle_problem_l238_238219


namespace sum_of_coefficients_l238_238092

theorem sum_of_coefficients (a a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 : ℝ) :
  (∀ x, (x^2 + 1) * (x - 2)^9 = a + a1 * (x - 1) + a2 * (x - 1)^2 + a3 * (x - 1)^3 + a4 * (x - 1)^4 +
        a5 * (x - 1)^5 + a6 * (x - 1)^6 + a7 * (x - 1)^7 + a8 * (x - 1)^8 + a9 * (x - 1)^9 + a10 * (x - 1)^10 + a11 * (x - 1)^11) →
  a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 = 2 := 
sorry

end sum_of_coefficients_l238_238092


namespace quadratic_eq_positive_integer_roots_l238_238159

theorem quadratic_eq_positive_integer_roots (k p : ℕ) 
  (h1 : k > 0)
  (h2 : ∃ x1 x2 : ℕ, x1 > 0 ∧ x2 > 0 ∧ (k-1) * x1^2 - p * x1 + k = 0 ∧ (k-1) * x2^2 - p * x2 + k = 0) :
  k ^ (k * p) * (p ^ p + k ^ k) + (p + k) = 1989 :=
by
  sorry

end quadratic_eq_positive_integer_roots_l238_238159


namespace fraction_decimal_equivalent_l238_238011

theorem fraction_decimal_equivalent : (7 / 16 : ℝ) = 0.4375 :=
by
  sorry

end fraction_decimal_equivalent_l238_238011


namespace imaginary_part_of_z_is_sqrt2_div2_l238_238356

open Complex

noncomputable def z : ℂ := abs (1 - I) / (1 - I)

theorem imaginary_part_of_z_is_sqrt2_div2 : z.im = Real.sqrt 2 / 2 := by
  sorry

end imaginary_part_of_z_is_sqrt2_div2_l238_238356


namespace distinct_integers_sum_of_three_elems_l238_238491

-- Define the set S and the property of its elements
def S : Set ℕ := {1, 4, 7, 10, 13, 16, 19}

-- Define the property that each element in S is of the form 3k + 1
def is_form_3k_plus_1 (x : ℕ) : Prop := ∃ k : ℤ, x = 3 * k + 1

theorem distinct_integers_sum_of_three_elems (h₁ : ∀ x ∈ S, is_form_3k_plus_1 x) :
  (∃! n, n = 13) :=
by
  sorry

end distinct_integers_sum_of_three_elems_l238_238491


namespace angle_AOD_128_57_l238_238501

-- Define angles as real numbers
variables {α β : ℝ}

-- Define the conditions
def perp (v1 v2 : ℝ) := v1 = 90 - v2

theorem angle_AOD_128_57 
  (h1 : perp α 90)
  (h2 : perp β 90)
  (h3 : α = 2.5 * β) :
  α = 128.57 :=
by
  -- Proof would go here
  sorry

end angle_AOD_128_57_l238_238501


namespace integer_roots_l238_238463

-- Define the polynomial
def poly (x : ℤ) : ℤ := x^3 - 4 * x^2 - 11 * x + 24

-- State the theorem
theorem integer_roots : {x : ℤ | poly x = 0} = {-1, 2, 3} := 
  sorry

end integer_roots_l238_238463


namespace tamika_carlos_probability_l238_238244

theorem tamika_carlos_probability :
  let tamika_results := [10 + 11, 10 + 12, 11 + 12],
      carlos_results := [4 * 6, 4 * 7, 6 * 7] in
  (∃ t ∈ tamika_results, ∃ c ∈ carlos_results, t > c) = false :=
by sorry

end tamika_carlos_probability_l238_238244


namespace fraction_to_decimal_l238_238041

theorem fraction_to_decimal :
  (7 / 16 : ℝ) = 0.4375 := 
begin
  -- placeholder for the proof
  sorry
end

end fraction_to_decimal_l238_238041


namespace fraction_to_decimal_l238_238082

theorem fraction_to_decimal : (7 : ℚ) / 16 = 4375 / 10000 := by
  sorry

end fraction_to_decimal_l238_238082


namespace cos_300_eq_half_l238_238809

theorem cos_300_eq_half : Real.cos (300 * Real.pi / 180) = 1 / 2 := 
by sorry

end cos_300_eq_half_l238_238809


namespace total_clouds_l238_238116

theorem total_clouds (C B : ℕ) (h1 : C = 6) (h2 : B = 3 * C) : C + B = 24 := by
  sorry

end total_clouds_l238_238116


namespace greatest_sum_of_consecutive_integers_l238_238704

def consecutiveSumCondition (n : ℤ) : Prop :=
  n * (n + 1) < 500 

theorem greatest_sum_of_consecutive_integers : 
  ∃ n : ℤ, consecutiveSumCondition n ∧ ∀ m : ℤ, consecutiveSumCondition m → n + (n + 1) ≥ m + (m + 1) :=
sorry

end greatest_sum_of_consecutive_integers_l238_238704


namespace clara_meeting_time_l238_238450

theorem clara_meeting_time (d T : ℝ) :
  (d / 20 = T - 0.5) →
  (d / 12 = T + 0.5) →
  (d / T = 15) :=
by
  intros h1 h2
  sorry

end clara_meeting_time_l238_238450


namespace sum_first_15_odd_starting_from_5_l238_238554

-- Definitions based on conditions in the problem.
def a : ℕ := 5    -- First term of the sequence is 5
def n : ℕ := 15   -- Number of terms is 15

-- Define the sequence of odd numbers starting from 5
def oddSeq (i : ℕ) : ℕ := a + 2 * i

-- Define the sum of the first n terms of this sequence
def sumOddSeq : ℕ := ∑ i in Finset.range n, oddSeq i

-- Key statement to prove that the sum of the sequence is 255
theorem sum_first_15_odd_starting_from_5 : sumOddSeq = 255 := by
  sorry

end sum_first_15_odd_starting_from_5_l238_238554


namespace num_of_friends_donated_same_l238_238438

def total_clothing_donated_by_adam (pants jumpers pajama_sets t_shirts : ℕ) : ℕ :=
  pants + jumpers + 2 * pajama_sets + t_shirts

def clothing_kept_by_adam (initial_donation : ℕ) : ℕ :=
  initial_donation / 2

def clothing_donated_by_friends (total_donated keeping friends_donation : ℕ) : ℕ :=
  total_donated - keeping

def num_friends (friends_donation adam_initial_donation : ℕ) : ℕ :=
  friends_donation / adam_initial_donation

theorem num_of_friends_donated_same (pants jumpers pajama_sets t_shirts total_donated : ℕ)
  (initial_donation := total_clothing_donated_by_adam pants jumpers pajama_sets t_shirts)
  (keeping := clothing_kept_by_adam initial_donation)
  (friends_donation := clothing_donated_by_friends total_donated keeping initial_donation)
  (friends := num_friends friends_donation initial_donation)
  (hp : pants = 4)
  (hj : jumpers = 4)
  (hps : pajama_sets = 4)
  (ht : t_shirts = 20)
  (htotal : total_donated = 126) :
  friends = 3 :=
by
  sorry

end num_of_friends_donated_same_l238_238438


namespace nested_radical_solution_l238_238599

theorem nested_radical_solution :
  (∃ x : ℝ, x = Real.sqrt (18 + x) ∧ x ≥ 0) ∧ ∀ x : ℝ, x = Real.sqrt (18 + x) → x ≥ 0 → x = 6 :=
by
  sorry

end nested_radical_solution_l238_238599


namespace find_r_amount_l238_238085

theorem find_r_amount (p q r : ℝ) (h_total : p + q + r = 8000) (h_r_fraction : r = 2 / 3 * (p + q)) : r = 3200 :=
by 
  -- Proof is not required, hence we use sorry
  sorry

end find_r_amount_l238_238085


namespace value_of_y_l238_238855

theorem value_of_y (y m : ℕ) (h1 : ((1 ^ m) / (y ^ m)) * (1 ^ 16 / 4 ^ 16) = 1 / (2 * 10 ^ 31)) (h2 : m = 31) : 
  y = 5 := 
sorry

end value_of_y_l238_238855


namespace number_of_correct_propositions_l238_238467

def double_factorial : ℕ → ℕ
| 0       := 1
| 1       := 1
| (n + 2) := (n + 2) * double_factorial n

lemma prop1 : double_factorial 2011 * double_factorial 2010 = 2011! :=
sorry

lemma prop2 : double_factorial 2010 = 2^1005 * 1005! :=
sorry

lemma prop3 : (double_factorial 2010) % 10 = 0 :=
sorry

lemma prop4 : (double_factorial 2011) % 10 = 5 :=
sorry

theorem number_of_correct_propositions : (num_correct : ℕ) :=
  let prop1_correct := prop1,
      prop2_correct := prop2,
      prop3_correct := prop3,
      prop4_correct := prop4 in
  nat.succ (nat.succ (nat.succ nat.zero)) -- equivalent to 4

end number_of_correct_propositions_l238_238467


namespace zara_owns_113_goats_l238_238757

-- Defining the conditions
def cows : Nat := 24
def sheep : Nat := 7
def groups : Nat := 3
def animals_per_group : Nat := 48

-- Stating the problem, with conditions as definitions
theorem zara_owns_113_goats : 
  let total_animals := groups * animals_per_group in
  let cows_and_sheep := cows + sheep in
  let goats := total_animals - cows_and_sheep in
  goats = 113 := by
  sorry

end zara_owns_113_goats_l238_238757


namespace solve_a_l238_238851

theorem solve_a (a x : ℤ) (h₀ : x = 5) (h₁ : a * x - 8 = 20 + a) : a = 7 :=
by
  sorry

end solve_a_l238_238851


namespace not_jog_probability_eq_l238_238674

def P_jog : ℚ := 5 / 8

theorem not_jog_probability_eq :
  1 - P_jog = 3 / 8 :=
by
  sorry

end not_jog_probability_eq_l238_238674


namespace tire_swap_distance_l238_238668

theorem tire_swap_distance : ∃ x : ℕ, 
  (1 - x / 11000) * 9000 = (1 - x / 9000) * 11000 ∧ x = 4950 := 
by
  sorry

end tire_swap_distance_l238_238668


namespace fraction_to_decimal_l238_238036

theorem fraction_to_decimal : (7 : ℝ) / 16 = 0.4375 := 
by
  sorry

end fraction_to_decimal_l238_238036


namespace algebraic_expression_value_l238_238498

-- Define the problem conditions and the final proof statement.
theorem algebraic_expression_value : 
  (∀ m n : ℚ, (2 * m - 1 = 0) → (1 / 2 * n - 2 * m = 0) → m ^ 2023 * n ^ 2022 = 1 / 2) :=
by
  sorry

end algebraic_expression_value_l238_238498


namespace simplify_expression_l238_238339

theorem simplify_expression (b : ℝ) (h1 : b ≠ 1) (h2 : b ≠ 1 / 2) :
  (1 / 2 - 1 / (1 + b / (1 - 2 * b))) = (3 * b - 1) / (2 * (1 - b)) :=
sorry

end simplify_expression_l238_238339


namespace pentomino_symmetry_count_l238_238847

noncomputable def num_symmetric_pentominoes : Nat :=
  15 -- This represents the given set of 15 different pentominoes

noncomputable def symmetric_pentomino_count : Nat :=
  -- Here we are asserting that the count of pentominoes with at least one vertical symmetry is 8
  8

theorem pentomino_symmetry_count :
  symmetric_pentomino_count = 8 :=
sorry

end pentomino_symmetry_count_l238_238847


namespace shirts_total_cost_l238_238377

def shirt_cost_problem : Prop :=
  ∃ (first_shirt_cost second_shirt_cost total_cost : ℕ),
    first_shirt_cost = 15 ∧
    first_shirt_cost = second_shirt_cost + 6 ∧
    total_cost = first_shirt_cost + second_shirt_cost ∧
    total_cost = 24

theorem shirts_total_cost : shirt_cost_problem := by
  sorry

end shirts_total_cost_l238_238377


namespace diff_of_squares_l238_238158

theorem diff_of_squares (a b : ℝ) (h1 : a + b = 8) (h2 : a - b = 4) : a^2 - b^2 = 32 :=
by
  sorry

end diff_of_squares_l238_238158


namespace students_in_class_l238_238528

theorem students_in_class (n : ℕ) 
  (h1 : 15 = 15)
  (h2 : ∃ m, n = m + 20 - 1)
  (h3 : ∃ x : ℕ, x = 3) :
  n = 38 :=
by
  sorry

end students_in_class_l238_238528


namespace part1_solution_set_part2_range_of_a_l238_238936

-- Defining the function f(x) under given conditions
def f (x a : ℝ) : ℝ := |x - a^2| + |x - (2 * a + 1)|

-- Part 1: Determine the solution set for the inequality when a = 2
theorem part1_solution_set (x : ℝ) : (f x 2) ≥ 4 ↔ x ≤ 3/2 ∨ x ≥ 11/2 := by
  sorry

-- Part 2: Determine the range of values for a given the inequality
theorem part2_range_of_a (a : ℝ) : (∀ x : ℝ, f x a ≥ 4) ↔ a ≤ -1 ∨ a ≥ 3 := by
  sorry

end part1_solution_set_part2_range_of_a_l238_238936


namespace negation_of_universal_statement_l238_238400

theorem negation_of_universal_statement :
  ¬ (∀ x : ℝ, x^2 ≤ 1) ↔ ∃ x : ℝ, x^2 > 1 :=
by
  sorry

end negation_of_universal_statement_l238_238400


namespace t_shirts_sold_l238_238248

theorem t_shirts_sold (total_money : ℕ) (money_per_tshirt : ℕ) (n : ℕ) 
  (h1 : total_money = 2205) (h2 : money_per_tshirt = 9) (h3 : total_money = n * money_per_tshirt) : 
  n = 245 :=
by
  sorry

end t_shirts_sold_l238_238248
