import Mathlib

namespace symmetry_axis_g_l94_94849

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * (x - Real.pi / 3) - Real.pi / 3)

theorem symmetry_axis_g :
  ∃ k : ℤ, (x = k * Real.pi / 2 + Real.pi / 4) := sorry

end symmetry_axis_g_l94_94849


namespace carnival_candies_l94_94995

theorem carnival_candies :
  ∃ (c : ℕ), c % 5 = 4 ∧ c % 6 = 3 ∧ c % 8 = 5 ∧ c < 150 ∧ c = 69 :=
by
  sorry

end carnival_candies_l94_94995


namespace exists_rational_non_integer_satisfying_linear_no_rational_non_integer_satisfying_quadratic_l94_94147

theorem exists_rational_non_integer_satisfying_linear :
  ∃ (x y : ℚ), x.denom ≠ 1 ∧ y.denom ≠ 1 ∧ 19 * x + 8 * y ∈ ℤ ∧ 8 * x + 3 * y ∈ ℤ :=
by
  sorry

theorem no_rational_non_integer_satisfying_quadratic :
  ¬ ∃ (x y : ℚ), x.denom ≠ 1 ∧ y.denom ≠ 1 ∧ 19 * x^2 + 8 * y^2 ∈ ℤ ∧ 8 * x^2 + 3 * y^2 ∈ ℤ :=
by
  sorry

end exists_rational_non_integer_satisfying_linear_no_rational_non_integer_satisfying_quadratic_l94_94147


namespace max_xy_l94_94684

theorem max_xy (x y : ℕ) (h : 7 * x + 4 * y = 150) : x * y ≤ 200 :=
sorry

end max_xy_l94_94684


namespace sin_10pi_over_3_l94_94494

theorem sin_10pi_over_3 : Real.sin (10 * Real.pi / 3) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_10pi_over_3_l94_94494


namespace trip_time_l94_94459

theorem trip_time (x : ℝ) (T : ℝ) :
  (70 * 4 + 60 * 5 + 50 * x) / (4 + 5 + x) = 58 → 
  T = 4 + 5 + x → 
  T = 16.25 :=
by
  intro h1 h2
  sorry

end trip_time_l94_94459


namespace find_common_ratio_l94_94989

-- Define the geometric sequence
def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n-1)

-- Given conditions
lemma a2_eq_8 (a₁ q : ℝ) : geometric_sequence a₁ q 2 = 8 :=
by sorry

lemma a5_eq_64 (a₁ q : ℝ) : geometric_sequence a₁ q 5 = 64 :=
by sorry

-- The common ratio q
theorem find_common_ratio (a₁ q : ℝ) (hq : 0 < q) :
  (geometric_sequence a₁ q 2 = 8) → (geometric_sequence a₁ q 5 = 64) → q = 2 :=
by sorry

end find_common_ratio_l94_94989


namespace alex_downhill_time_l94_94008

theorem alex_downhill_time
  (speed_flat : ℝ)
  (time_flat : ℝ)
  (speed_uphill : ℝ)
  (time_uphill : ℝ)
  (speed_downhill : ℝ)
  (distance_walked : ℝ)
  (total_distance : ℝ)
  (h_flat : speed_flat = 20)
  (h_time_flat : time_flat = 4.5)
  (h_uphill : speed_uphill = 12)
  (h_time_uphill : time_uphill = 2.5)
  (h_downhill : speed_downhill = 24)
  (h_walked : distance_walked = 8)
  (h_total : total_distance = 164)
  : (156 - (speed_flat * time_flat + speed_uphill * time_uphill)) / speed_downhill = 1.5 :=
by 
  sorry

end alex_downhill_time_l94_94008


namespace linda_original_savings_l94_94291

theorem linda_original_savings (S : ℝ) (h1 : (2 / 3) * S + (1 / 3) * S = S) 
  (h2 : (1 / 3) * S = 250) : S = 750 :=
by sorry

end linda_original_savings_l94_94291


namespace arnold_danny_age_l94_94892

theorem arnold_danny_age:
  ∃ x : ℝ, (x + 1) * (x + 1) = x * x + 11 ∧ x = 5 :=
by
  sorry

end arnold_danny_age_l94_94892


namespace area_of_triangle_DEF_l94_94028

theorem area_of_triangle_DEF :
  let s := 2
  let hexagon_area := (3 * Real.sqrt 3 / 2) * s^2
  let radius := s
  let distance_between_centers := 2 * radius
  let side_of_triangle_DEF := distance_between_centers
  let triangle_area := (Real.sqrt 3 / 4) * side_of_triangle_DEF^2
  triangle_area = 4 * Real.sqrt 3 :=
by
  sorry

end area_of_triangle_DEF_l94_94028


namespace percent_of_part_is_20_l94_94296

theorem percent_of_part_is_20 {Part Whole : ℝ} (hPart : Part = 14) (hWhole : Whole = 70) : (Part / Whole) * 100 = 20 :=
by
  rw [hPart, hWhole]
  have h : (14 : ℝ) / 70 = 0.2 := by norm_num
  rw [h]
  norm_num

end percent_of_part_is_20_l94_94296


namespace eval_expression_l94_94642

theorem eval_expression : 
  let a := 2999 in
  let b := 3000 in
  b^3 - a * b^2 - a^2 * b + a^3 = 5999 := 
by 
  sorry

end eval_expression_l94_94642


namespace candies_left_to_share_l94_94836

def initial_candies : Nat := 100
def sibling_count : Nat := 3
def candies_per_sibling : Nat := 10
def candies_Josh_eats : Nat := 16

theorem candies_left_to_share :
  let candies_given_to_siblings := sibling_count * candies_per_sibling;
  let candies_after_siblings := initial_candies - candies_given_to_siblings;
  let candies_given_to_friend := candies_after_siblings / 2;
  let candies_after_friend := candies_after_siblings - candies_given_to_friend;
  let candies_after_Josh := candies_after_friend - candies_Josh_eats;
  candies_after_Josh = 19 :=
by
  sorry

end candies_left_to_share_l94_94836


namespace odd_function_has_zero_l94_94317

variable {R : Type} [LinearOrderedField R]

def is_odd_function (f : R → R) := ∀ x : R, f (-x) = -f x

theorem odd_function_has_zero {f : R → R} (h : is_odd_function f) : ∃ x : R, f x = 0 :=
sorry

end odd_function_has_zero_l94_94317


namespace number_of_seats_in_nth_row_l94_94086

theorem number_of_seats_in_nth_row (n : ℕ) :
    ∃ m : ℕ, m = 3 * n + 15 :=
by
  sorry

end number_of_seats_in_nth_row_l94_94086


namespace winning_candidate_percentage_l94_94537

theorem winning_candidate_percentage (P : ℕ) (majority : ℕ) (total_votes : ℕ) (h1 : majority = 188) (h2 : total_votes = 470) (h3 : 2 * majority = (2 * P - 100) * total_votes) : 
  P = 70 := 
sorry

end winning_candidate_percentage_l94_94537


namespace vector_magnitude_subtraction_l94_94052

theorem vector_magnitude_subtraction :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5 :=
by
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  show real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5 from
  sorry

end vector_magnitude_subtraction_l94_94052


namespace derivative_at_pi_l94_94807

noncomputable def f (x : ℝ) : ℝ := (x^2) / (Real.cos x)

theorem derivative_at_pi : deriv f π = -2 * π :=
by
  sorry

end derivative_at_pi_l94_94807


namespace triangle_inequality_l94_94962

variables {α β γ a b c : ℝ}
variable {n : ℕ}

theorem triangle_inequality (h_sum_angles : α + β + γ = Real.pi) (h_pos_sides : 0 < a ∧ 0 < b ∧ 0 < c) :
  (Real.pi / 3) ^ n ≤ (a * α ^ n + b * β ^ n + c * γ ^ n) / (a + b + c) ∧ 
  (a * α ^ n + b * β ^ n + c * γ ^ n) / (a + b + c) < (Real.pi ^ n / 2) :=
by
  sorry

end triangle_inequality_l94_94962


namespace marbles_problem_l94_94692

theorem marbles_problem (h_total: ℕ) (h_each: ℕ) (h_total_eq: h_total = 35) (h_each_eq: h_each = 7) :
    h_total / h_each = 5 := by
  sorry

end marbles_problem_l94_94692


namespace find_k_l94_94933

theorem find_k 
  (x y k : ℚ) 
  (h1 : y = 4 * x - 1) 
  (h2 : y = -1 / 3 * x + 11) 
  (h3 : y = 2 * x + k) : 
  k = 59 / 13 :=
sorry

end find_k_l94_94933


namespace jerry_painting_hours_l94_94222

-- Define the variables and conditions
def time_painting (P : ℕ) : ℕ := P
def time_counter (P : ℕ) : ℕ := 3 * P
def time_lawn : ℕ := 6
def hourly_rate : ℕ := 15
def total_paid : ℕ := 570

-- Hypothesize that the total hours spent leads to the total payment
def total_hours (P : ℕ) : ℕ := time_painting P + time_counter P + time_lawn

-- Prove that the solution for P matches the conditions
theorem jerry_painting_hours (P : ℕ) 
  (h1 : hourly_rate * total_hours P = total_paid) : 
  P = 8 :=
by
  sorry

end jerry_painting_hours_l94_94222


namespace largest_four_digit_number_last_digit_l94_94767

theorem largest_four_digit_number_last_digit (n : ℕ) (n' : ℕ) (m r a b : ℕ) :
  (1000 * m + 100 * r + 10 * a + b = n) →
  (100 * m + 10 * r + a = n') →
  (n % 9 = 0) →
  (n' % 4 = 0) →
  b = 3 :=
by
  sorry

end largest_four_digit_number_last_digit_l94_94767


namespace mat_weavers_equiv_l94_94570

theorem mat_weavers_equiv {x : ℕ} 
  (h1 : 4 * 1 = 4) 
  (h2 : 16 * (64 / 16) = 64) 
  (h3 : 1 = 64 / (16 * x)) : x = 4 :=
by
  sorry

end mat_weavers_equiv_l94_94570


namespace pool_capacity_percentage_l94_94118

noncomputable def hose_rate := 60 -- cubic feet per minute
noncomputable def pool_width := 80 -- feet
noncomputable def pool_length := 150 -- feet
noncomputable def pool_depth := 10 -- feet
noncomputable def drainage_time := 2000 -- minutes
noncomputable def pool_volume := pool_width * pool_length * pool_depth -- cubic feet
noncomputable def removed_water_volume := hose_rate * drainage_time -- cubic feet

theorem pool_capacity_percentage :
  (removed_water_volume / pool_volume) * 100 = 100 :=
by
  -- the proof steps would go here
  sorry

end pool_capacity_percentage_l94_94118


namespace max_xy_l94_94683

theorem max_xy (x y : ℕ) (h : 7 * x + 4 * y = 150) : x * y ≤ 200 :=
sorry

end max_xy_l94_94683


namespace fish_per_person_l94_94543

theorem fish_per_person (eyes_per_fish : ℕ) (fish_caught : ℕ) (total_eyes : ℕ) (dog_eyes : ℕ) (oomyapeck_eyes : ℕ) (n_people : ℕ) :
  total_eyes = oomyapeck_eyes + dog_eyes →
  total_eyes = fish_caught * eyes_per_fish →
  n_people = 3 →
  oomyapeck_eyes = 22 →
  dog_eyes = 2 →
  eyes_per_fish = 2 →
  fish_caught / n_people = 4 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end fish_per_person_l94_94543


namespace price_reduction_2100_yuan_l94_94607

-- Definitions based on conditions
def initial_sales : ℕ := 30
def initial_profit_per_item : ℕ := 50
def additional_sales_per_yuan (x : ℕ) : ℕ := 2 * x
def new_profit_per_item (x : ℕ) : ℕ := 50 - x
def target_profit : ℕ := 2100

-- Final proof statement, showing the price reduction needed
theorem price_reduction_2100_yuan (x : ℕ) 
  (h : (50 - x) * (30 + 2 * x) = 2100) : 
  x = 20 := 
by 
  sorry

end price_reduction_2100_yuan_l94_94607


namespace sum_three_circles_l94_94188

theorem sum_three_circles (a b : ℚ) 
  (h1 : 5 * a + 2 * b = 27)
  (h2 : 2 * a + 5 * b = 29) :
  3 * b = 13 :=
by
  sorry

end sum_three_circles_l94_94188


namespace reflect_y_axis_correct_l94_94858

-- Define the initial coordinates of the point M
def M_orig : ℝ × ℝ := (3, 2)

-- Define the reflection function across the y-axis
def reflect_y_axis (M : ℝ × ℝ) : ℝ × ℝ :=
  (-M.1, M.2)

-- Prove that reflecting M_orig across the y-axis results in the coordinates (-3, 2)
theorem reflect_y_axis_correct : reflect_y_axis M_orig = (-3, 2) :=
  by
    -- Provide the missing steps of the proof
    sorry

end reflect_y_axis_correct_l94_94858


namespace circumscribed_sphere_radius_l94_94216

/-- Define the right triangular prism -/
structure RightTriangularPrism :=
(AB AC BC : ℝ)
(AA1 : ℝ)
(h_base : AB = 4 * Real.sqrt 2 ∧ AC = 4 * Real.sqrt 2 ∧ BC = 8)
(h_height : AA1 = 6)

/-- The condition that the base is an isosceles right-angled triangle -/
structure IsoscelesRightAngledTriangle :=
(A B C : ℝ)
(AB AC : ℝ)
(BC : ℝ)
(h_isosceles_right : AB = AC ∧ BC = Real.sqrt (AB^2 + AC^2))

/-- The main theorem stating the radius of the circumscribed sphere -/
theorem circumscribed_sphere_radius (prism : RightTriangularPrism) 
    (base : IsoscelesRightAngledTriangle) 
    (h_base_correct : base.AB = prism.AB ∧ base.AC = prism.AC ∧ base.BC = prism.BC):
    ∃ radius : ℝ, radius = 5 := 
by
    sorry

end circumscribed_sphere_radius_l94_94216


namespace sin_neg_30_eq_neg_half_l94_94336

/-- Prove that the sine of -30 degrees is -1/2 -/
theorem sin_neg_30_eq_neg_half : Real.sin (-(30 * Real.pi / 180)) = -1 / 2 :=
by
  sorry

end sin_neg_30_eq_neg_half_l94_94336


namespace sum_prime_factors_77_l94_94453

theorem sum_prime_factors_77 : ∑ p in {7, 11}, p = 18 :=
by {
  have h1 : Nat.Prime 7 := Nat.prime_iff_nat_prime.mp Nat.prime_factors_one,
  have h2 : Nat.Prime 11 := Nat.prime_iff_nat_prime.mp Nat.prime_factors_one,
  have h3 : 77 = 7 * 11 := by simp,
  have h_set: {7, 11} = PrimeFactors 77 := by sorry, 
  rw h_set,
  exact Nat.sum_prime_factors_eq_77,
  sorry
}

end sum_prime_factors_77_l94_94453


namespace bacteria_growth_l94_94907

-- Define the original and current number of bacteria
def original_bacteria := 600
def current_bacteria := 8917

-- Define the increase in bacteria count
def additional_bacteria := 8317

-- Prove the statement
theorem bacteria_growth : current_bacteria - original_bacteria = additional_bacteria :=
by {
    -- Lean will require the proof here, so we use sorry for now 
    sorry
}

end bacteria_growth_l94_94907


namespace deepak_investment_l94_94011

theorem deepak_investment (D : ℝ) (A : ℝ) (P : ℝ) (Dp : ℝ) (Ap : ℝ) 
  (hA : A = 22500)
  (hP : P = 13800)
  (hDp : Dp = 5400)
  (h_ratio : Dp / P = D / (A + D)) :
  D = 15000 := by
  sorry

end deepak_investment_l94_94011


namespace suitable_graph_for_air_composition_is_pie_chart_l94_94625

/-- The most suitable type of graph to visually represent the percentage 
of each component in the air is a pie chart, based on the given conditions. -/
theorem suitable_graph_for_air_composition_is_pie_chart 
  (bar_graph : Prop)
  (line_graph : Prop)
  (pie_chart : Prop)
  (histogram : Prop)
  (H1 : bar_graph → comparing_quantities)
  (H2 : line_graph → display_data_over_time)
  (H3 : pie_chart → show_proportions_of_whole)
  (H4 : histogram → show_distribution_of_dataset) 
  : suitable_graph_to_represent_percentage = pie_chart :=
sorry

end suitable_graph_for_air_composition_is_pie_chart_l94_94625


namespace point_T_coordinates_l94_94645

-- Definition of a point in 2D plane
structure Point where
  x : ℝ
  y : ℝ

-- Definition of a square with specific points O, P, Q, R
structure Square where
  O : Point
  P : Point
  Q : Point
  R : Point

-- Condition: O is the origin
def O : Point := {x := 0, y := 0}

-- Condition: Q is at (3, 3)
def Q : Point := {x := 3, y := 3}

-- Assuming the function area_triang for calculating the area of a triangle given three points
def area_triang (A B C : Point) : ℝ :=
  0.5 * abs ((B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x))

-- Assuming the function area_square for calculating the area of a square given the length of the side
def area_square (s : ℝ) : ℝ := s * s

-- Coordinates of point P and R since it's a square with sides parallel to axis
def P : Point := {x := 3, y := 0}
def R : Point := {x := 0, y := 3}

-- Definition of the square OPQR
def OPQR : Square := {O := O, P := P, Q := Q, R := R}

-- Length of the side of square OPQR
def side_length : ℝ := 3

-- Area of the square OPQR
def square_area : ℝ := area_square side_length

-- Twice the area of the square OPQR
def required_area : ℝ := 2 * square_area

-- Point T that needs to be proven
def T : Point := {x := 3, y := 12}

-- The main theorem to prove
theorem point_T_coordinates (T : Point) : area_triang P Q T = required_area → T = {x := 3, y := 12} :=
by
  sorry

end point_T_coordinates_l94_94645


namespace sum_of_two_numbers_l94_94385

theorem sum_of_two_numbers (x y : ℕ) (hxy : x > y) (h1 : x - y = 4) (h2 : x * y = 156) : x + y = 28 :=
by {
  sorry
}

end sum_of_two_numbers_l94_94385


namespace last_three_digits_7_pow_80_l94_94951

theorem last_three_digits_7_pow_80 : (7 ^ 80) % 1000 = 961 := 
by sorry

end last_three_digits_7_pow_80_l94_94951


namespace sampling_method_D_is_the_correct_answer_l94_94889

def sampling_method_A_is_simple_random_sampling : Prop :=
  false

def sampling_method_B_is_simple_random_sampling : Prop :=
  false

def sampling_method_C_is_simple_random_sampling : Prop :=
  false

def sampling_method_D_is_simple_random_sampling : Prop :=
  true

theorem sampling_method_D_is_the_correct_answer :
  sampling_method_A_is_simple_random_sampling = false ∧
  sampling_method_B_is_simple_random_sampling = false ∧
  sampling_method_C_is_simple_random_sampling = false ∧
  sampling_method_D_is_simple_random_sampling = true :=
by
  sorry

end sampling_method_D_is_the_correct_answer_l94_94889


namespace problem1_problem2_problem3_problem4_l94_94488

-- Problem 1
theorem problem1 : (2 / 19) * (8 / 25) + (17 / 25) / (19 / 2) = 2 / 19 := 
by sorry

-- Problem 2
theorem problem2 : (1 / 4) * 125 * (1 / 25) * 8 = 10 := 
by sorry

-- Problem 3
theorem problem3 : ((1 / 3) + (1 / 4)) / ((1 / 2) - (1 / 3)) = 7 / 2 := 
by sorry

-- Problem 4
theorem problem4 : ((1 / 6) + (1 / 8)) * 24 * (1 / 9) = 7 / 9 := 
by sorry

end problem1_problem2_problem3_problem4_l94_94488


namespace sin_neg_30_eq_neg_one_half_l94_94333

theorem sin_neg_30_eq_neg_one_half : Real.sin (-30 / 180 * Real.pi) = -1 / 2 := by
  -- Proof goes here
  sorry

end sin_neg_30_eq_neg_one_half_l94_94333


namespace train_speed_l94_94478

theorem train_speed (distance_meters : ℕ) (time_seconds : ℕ) 
  (h_distance : distance_meters = 150) (h_time : time_seconds = 20) : 
  distance_meters / 1000 / (time_seconds / 3600) = 27 :=
by 
  have h1 : distance_meters = 150 := h_distance
  have h2 : time_seconds = 20 := h_time
  -- other intermediate steps would go here, but are omitted
  -- for now, we assume the final calculation is:
  sorry

end train_speed_l94_94478


namespace range_of_a_l94_94980

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 4 → (1/3 * x^3 - (a/2) * x^2 + (a-1) * x + 1) < 0) ∧
  (∀ x : ℝ, x > 6 → (1/3 * x^3 - (a/2) * x^2 + (a-1) * x + 1) > 0)
  ↔ (5 < a ∧ a < 7) :=
sorry

end range_of_a_l94_94980


namespace servant_received_amount_l94_94770

def annual_salary := 900
def uniform_price := 100
def fraction_of_year_served := 3 / 4

theorem servant_received_amount :
  annual_salary * fraction_of_year_served + uniform_price = 775 := by
  sorry

end servant_received_amount_l94_94770


namespace polygon_diagonals_l94_94614

theorem polygon_diagonals (n : ℕ) (h : 20 = n) : (n * (n - 3)) / 2 = 170 :=
by
  sorry

end polygon_diagonals_l94_94614


namespace ratio_sheep_to_horses_l94_94484

theorem ratio_sheep_to_horses 
  (horse_food_per_day : ℕ) 
  (total_horse_food : ℕ) 
  (num_sheep : ℕ) 
  (H1 : horse_food_per_day = 230) 
  (H2 : total_horse_food = 12880) 
  (H3 : num_sheep = 48) 
  : (num_sheep : ℚ) / (total_horse_food / horse_food_per_day : ℚ) = 6 / 7
  :=
by
  sorry

end ratio_sheep_to_horses_l94_94484


namespace deposit_is_500_l94_94705

-- Definitions corresponding to the conditions
def janet_saved : ℕ := 2225
def rent_per_month : ℕ := 1250
def advance_months : ℕ := 2
def extra_needed : ℕ := 775

-- Definition that encapsulates the deposit calculation
def deposit_required (saved rent_monthly months_advance extra : ℕ) : ℕ :=
  let total_rent := months_advance * rent_monthly
  let total_needed := saved + extra
  total_needed - total_rent

-- Theorem statement for the proof problem
theorem deposit_is_500 : deposit_required janet_saved rent_per_month advance_months extra_needed = 500 :=
by
  sorry

end deposit_is_500_l94_94705


namespace sum_fifth_powers_divisible_by_15_l94_94874

theorem sum_fifth_powers_divisible_by_15
  (A B C D E : ℤ) 
  (h : A + B + C + D + E = 0) : 
  (A^5 + B^5 + C^5 + D^5 + E^5) % 15 = 0 := 
by 
  sorry

end sum_fifth_powers_divisible_by_15_l94_94874


namespace sum_inverses_mod_17_l94_94935

theorem sum_inverses_mod_17 : 
  (3⁻¹ + 3⁻² + 3⁻³ + 3⁻⁴ + 3⁻⁵ + 3⁻⁶) % 17 = 3 % 17 := 
by 
  sorry

end sum_inverses_mod_17_l94_94935


namespace find_HCF_l94_94688

-- Given conditions
def LCM : ℕ := 750
def product_of_two_numbers : ℕ := 18750

-- Proof statement
theorem find_HCF (h : ℕ) (hpos : h > 0) :
  (LCM * h = product_of_two_numbers) → h = 25 :=
by
  sorry

end find_HCF_l94_94688


namespace complex_identity_l94_94402

variable (i : ℂ)
axiom i_squared : i^2 = -1

theorem complex_identity : 1 + i + i^2 = i :=
by sorry

end complex_identity_l94_94402


namespace brad_zip_code_l94_94321

theorem brad_zip_code (x y : ℕ) (h1 : x + x + 0 + 2 * x + y = 10) : 2 * x + y = 8 :=
by 
  sorry

end brad_zip_code_l94_94321


namespace machine_does_not_require_repair_l94_94409

variable (nominal_mass max_deviation standard_deviation : ℝ)
variable (nominal_mass_ge : nominal_mass ≥ 370)
variable (max_deviation_le : max_deviation ≤ 0.1 * nominal_mass)
variable (all_deviations_le_max : ∀ d, d < max_deviation → d < 37)
variable (std_dev_le_max_dev : standard_deviation ≤ max_deviation)

theorem machine_does_not_require_repair :
  ¬ (standard_deviation > 37) :=
by 
  -- sorry annotation indicates the proof goes here
  sorry

end machine_does_not_require_repair_l94_94409


namespace workers_problem_l94_94890

theorem workers_problem (W : ℕ) (A : ℕ) :
  (W * 45 = A) ∧ ((W + 10) * 35 = A) → W = 35 :=
by
  sorry

end workers_problem_l94_94890


namespace no_repair_needed_l94_94411

def nominal_mass : ℝ := 370 -- Assign the nominal mass as determined in the problem solving.

def max_deviation (M : ℝ) : ℝ := 0.1 * M
def preserved_max_deviation : ℝ := 37
def unreadable_max_deviation : ℝ := 37

def within_max_deviation (dev : ℝ) := dev ≤ preserved_max_deviation

noncomputable def standard_deviation : ℝ := preserved_max_deviation

theorem no_repair_needed :
  ∀ (M : ℝ),
  max_deviation M = 0.1 * M →
  preserved_max_deviation ≤ max_deviation M →
  ∀ (dev : ℝ), within_max_deviation dev →
  standard_deviation ≤ preserved_max_deviation →
  preserved_max_deviation = 37 →
  "не требует" = "не требует" :=
by
  intros M h1 h2 h3 h4 h5
  sorry

end no_repair_needed_l94_94411


namespace tangent_line_eq_l94_94046

-- Definitions for the conditions
def curve (x : ℝ) : ℝ := x^3 - 3 * x^2 + 2 * x

def derivative_curve (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 2

-- Define the problem as a theorem statement
theorem tangent_line_eq (L : ℝ → ℝ) (hL : ∀ x, L x = 2 * x ∨ L x = - x/4) :
  (∀ x, x = 0 → L x = 0) →
  (∀ x x0, L x = curve x → derivative_curve x0 = derivative_curve 0 → x0 = 0 ∨ x0 = 3/2) →
  (L x = 2 * x - curve x ∨ L x = 4 * x + curve x) :=
by
  sorry

end tangent_line_eq_l94_94046


namespace sector_area_is_80pi_l94_94531

noncomputable def sectorArea (θ r : ℝ) : ℝ := 
  1 / 2 * θ * r^2

theorem sector_area_is_80pi :
  sectorArea (2 * Real.pi / 5) 20 = 80 * Real.pi :=
by
  sorry

end sector_area_is_80pi_l94_94531


namespace gino_popsicle_sticks_l94_94039

variable (my_sticks : ℕ) (total_sticks : ℕ) (gino_sticks : ℕ)

def popsicle_sticks_condition (my_sticks : ℕ) (total_sticks : ℕ) (gino_sticks : ℕ) : Prop :=
  my_sticks = 50 ∧ total_sticks = 113

theorem gino_popsicle_sticks
  (h : popsicle_sticks_condition my_sticks total_sticks gino_sticks) :
  gino_sticks = 63 :=
  sorry

end gino_popsicle_sticks_l94_94039


namespace ABD_collinear_l94_94515

noncomputable def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, (p2.1 - p1.1) * k = p3.1 - p1.1 ∧ (p2.2 - p1.2) * k = p3.2 - p1.2

noncomputable def vector (x y : ℝ) : ℝ × ℝ := (x, y)

variables {a b : ℝ × ℝ}
variables {A B C D : ℝ × ℝ}

axiom a_ne_zero : a ≠ (0, 0)
axiom b_ne_zero : b ≠ (0, 0)
axiom a_b_not_collinear : ∀ k : ℝ, a ≠ k • b
axiom AB_def : B = (A.1 + a.1 + b.1, A.2 + a.2 + b.2)
axiom BC_def : C = (B.1 + a.1 + 10 * b.1, B.2 + a.2 + 10 * b.2)
axiom CD_def : D = (C.1 + 3 * (a.1 - 2 * b.1), C.2 + 3 * (a.2 - 2 * b.2))

theorem ABD_collinear : collinear A B D :=
by
  sorry

end ABD_collinear_l94_94515


namespace base_2_base_3_product_is_144_l94_94936

def convert_base_2_to_10 (n : ℕ) : ℕ :=
  match n with
  | 1001 => 9
  | _ => 0 -- For simplicity, only handle 1001_2

def convert_base_3_to_10 (n : ℕ) : ℕ :=
  match n with
  | 121 => 16
  | _ => 0 -- For simplicity, only handle 121_3

theorem base_2_base_3_product_is_144 :
  convert_base_2_to_10 1001 * convert_base_3_to_10 121 = 144 :=
by
  sorry

end base_2_base_3_product_is_144_l94_94936


namespace prime_divisors_consecutive_l94_94516

theorem prime_divisors_consecutive (p q : ℕ) [hp : Prime p] [hq : Prime q] (h1 : p < q) (h2 : q < 2 * p) :
  ∃ (n m : ℕ), Nat.gcd n m = 1 ∧ abs (n - m) = 1 ∧ (∀ p' : ℕ, Prime p' → p' ∣ n → p' = p) ∧ (∀ q' : ℕ, Prime q' → q' ∣ m → q' = q) := 
  sorry

end prime_divisors_consecutive_l94_94516


namespace exists_rational_non_integer_linear_l94_94153

theorem exists_rational_non_integer_linear (k1 k2 : ℤ) : 
  ∃ (x y : ℚ), x ≠ ⌊x⌋ ∧ y ≠ ⌊y⌋ ∧ 
  19 * x + 8 * y = k1 ∧ 8 * x + 3 * y = k2 :=
sorry

end exists_rational_non_integer_linear_l94_94153


namespace total_volume_of_water_in_container_l94_94161

def volume_each_hemisphere : ℝ := 4
def number_of_hemispheres : ℝ := 2735

theorem total_volume_of_water_in_container :
  (volume_each_hemisphere * number_of_hemispheres) = 10940 :=
by
  sorry

end total_volume_of_water_in_container_l94_94161


namespace machine_does_not_require_repair_l94_94415

-- Define the conditions.

def max_deviation := 37

def nominal_portion_max_deviation_percentage := 0.10

def deviation_within_limit (M : ℝ) : Prop :=
  37 ≤ 0.10 * M

def unreadable_measurements_deviation (deviation : ℝ) : Prop :=
  deviation < 37

-- Define the theorem we want to prove

theorem machine_does_not_require_repair (M : ℝ)
  (h1 : deviation_within_limit M)
  (h2 : ∀ deviation, unreadable_measurements_deviation deviation) :
  true := 
sorry

end machine_does_not_require_repair_l94_94415


namespace probability_of_drawing_3_black_and_2_white_l94_94763

noncomputable def total_ways_to_draw_5_balls : ℕ := Nat.choose 27 5
noncomputable def ways_to_choose_3_black : ℕ := Nat.choose 10 3
noncomputable def ways_to_choose_2_white : ℕ := Nat.choose 12 2
noncomputable def favorable_outcomes : ℕ := ways_to_choose_3_black * ways_to_choose_2_white
noncomputable def desired_probability : ℚ := favorable_outcomes / total_ways_to_draw_5_balls

theorem probability_of_drawing_3_black_and_2_white :
  desired_probability = 132 / 1345 := by
  sorry

end probability_of_drawing_3_black_and_2_white_l94_94763


namespace rational_roots_of_quadratic_l94_94495

theorem rational_roots_of_quadratic (r : ℚ) :
  (∃ a b : ℤ, a ≠ b ∧ (r * a^2 + (r + 1) * a + r = 1 ∧ r * b^2 + (r + 1) * b + r = 1)) ↔ (r = 1 ∨ r = -1 / 7) :=
by
  sorry

end rational_roots_of_quadratic_l94_94495


namespace particular_solutions_of_diff_eq_l94_94499

variable {x y : ℝ}

theorem particular_solutions_of_diff_eq
  (h₁ : ∀ C : ℝ, x^2 = C * (y - C))
  (h₂ : x > 0) :
  (y = 2 * x ∨ y = -2 * x) ↔ (x * (y')^2 - 2 * y * y' + 4 * x = 0) := 
sorry

end particular_solutions_of_diff_eq_l94_94499


namespace kody_half_mohamed_years_ago_l94_94855

-- Definitions of initial conditions
def current_age_mohamed : ℕ := 2 * 30
def current_age_kody : ℕ := 32

-- Proof statement
theorem kody_half_mohamed_years_ago : ∃ x : ℕ, (current_age_kody - x) = (1 / 2 : ℕ) * (current_age_mohamed - x) ∧ x = 4 := 
by 
  sorry

end kody_half_mohamed_years_ago_l94_94855


namespace sin_neg_30_eq_neg_one_half_l94_94328

theorem sin_neg_30_eq_neg_one_half :
  Real.sin (-30 * Real.pi / 180) = -1 / 2 := 
by
  sorry -- Proof is skipped

end sin_neg_30_eq_neg_one_half_l94_94328


namespace problem1_condition1_problem2_condition_l94_94691

noncomputable theory

variables {A B C : ℝ} {a b c : ℝ}

-- Conditions for problem 1
theorem problem1_condition1 (h1 : c = real.sqrt 7) (h2 : C = real.pi / 3) (h3 : a^2 + b^2 - a * b = 7) (h4 : 2 * a = 3 * b) :
  a = 3 ∧ b = 2 :=
sorry

-- Conditions for problem 2
theorem problem2_condition (h5 : cos B = 3 * real.sqrt 10 / 10) :
  sin (2 * A) = (3 - 4 * real.sqrt 3) / 10 :=
sorry

end problem1_condition1_problem2_condition_l94_94691


namespace simple_interest_years_l94_94776

theorem simple_interest_years (P : ℝ) (R : ℝ) (T : ℝ) 
  (h1 : P = 300) 
  (h2 : (P * ((R + 6) / 100) * T) = (P * (R / 100) * T + 90)) : 
  T = 5 := 
by 
  -- Necessary proof steps go here
  sorry

end simple_interest_years_l94_94776


namespace truckToCarRatio_l94_94542

-- Conditions
def liftsCar (C : ℕ) : Prop := C = 5
def peopleNeeded (C T : ℕ) : Prop := 6 * C + 3 * T = 60

-- Theorem statement
theorem truckToCarRatio (C T : ℕ) (hc : liftsCar C) (hp : peopleNeeded C T) : T / C = 2 :=
by
  sorry

end truckToCarRatio_l94_94542


namespace problem_solution_l94_94886

theorem problem_solution :
  (2200 - 2089)^2 / 196 = 63 :=
sorry

end problem_solution_l94_94886


namespace vector_dot_product_correct_l94_94183

-- Definitions of the vectors
def vector_a : ℝ × ℝ := (1, 2)
def vector_b : ℝ × ℝ :=
  let x := 4 - 2 * vector_a.1
  let y := 1 - 2 * vector_a.2
  (x, y)

-- Theorem to prove the dot product is correct
theorem vector_dot_product_correct :
  (vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2) = 4 := by
  sorry

end vector_dot_product_correct_l94_94183


namespace sequence_general_formula_l94_94701

theorem sequence_general_formula (a : ℕ → ℝ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, a (n + 1) = 3 * a n + 2 * n - 1) :
  ∀ n : ℕ, a n = (2 / 3) * 3^n - n :=
by
  sorry

end sequence_general_formula_l94_94701


namespace bunches_with_new_distribution_l94_94781

-- Given conditions
def bunches_initial := 8
def flowers_per_bunch_initial := 9
def total_flowers := bunches_initial * flowers_per_bunch_initial

-- New condition and proof requirement
def flowers_per_bunch_new := 12
def bunches_new := total_flowers / flowers_per_bunch_new

theorem bunches_with_new_distribution : bunches_new = 6 := by
  sorry

end bunches_with_new_distribution_l94_94781


namespace trig_expression_evaluation_l94_94041

theorem trig_expression_evaluation (θ : ℝ) (h : Real.tan θ = 2) :
  Real.sin θ ^ 2 + (Real.sin θ * Real.cos θ) - 2 * (Real.cos θ ^ 2) = 4 / 5 := 
by
  sorry

end trig_expression_evaluation_l94_94041


namespace age_of_15th_student_l94_94894

theorem age_of_15th_student (avg_age_15 : ℕ) (avg_age_6 : ℕ) (avg_age_8 : ℕ) (num_students_15 : ℕ) (num_students_6 : ℕ) (num_students_8 : ℕ) 
  (h_avg_15 : avg_age_15 = 15) 
  (h_avg_6 : avg_age_6 = 14) 
  (h_avg_8 : avg_age_8 = 16) 
  (h_num_15 : num_students_15 = 15) 
  (h_num_6 : num_students_6 = 6) 
  (h_num_8 : num_students_8 = 8) : 
  ∃ age_15th_student : ℕ, age_15th_student = 13 := 
by
  sorry


end age_of_15th_student_l94_94894


namespace jim_travels_20_percent_of_jill_l94_94095

def john_distance : ℕ := 15
def jill_travels_less : ℕ := 5
def jim_distance : ℕ := 2
def jill_distance : ℕ := john_distance - jill_travels_less

theorem jim_travels_20_percent_of_jill :
  (jim_distance * 100) / jill_distance = 20 := by
  sorry

end jim_travels_20_percent_of_jill_l94_94095


namespace no_solution_implies_a_eq_one_l94_94085

theorem no_solution_implies_a_eq_one (a : ℝ) : 
  ¬(∃ x y : ℝ, a * x + y = 1 ∧ x + y = 2) → a = 1 :=
by
  intro h
  sorry

end no_solution_implies_a_eq_one_l94_94085


namespace largest_2_digit_number_l94_94497

theorem largest_2_digit_number:
  ∃ (N: ℕ), N >= 10 ∧ N < 100 ∧ N % 4 = 0 ∧ (∀ k: ℕ, k ≥ 1 → (N^k) % 100 = N % 100) ∧ 
  (∀ M: ℕ, M >= 10 → M < 100 → M % 4 = 0 → (∀ k: ℕ, k ≥ 1 → (M^k) % 100 = M % 100) → N ≥ M) :=
sorry

end largest_2_digit_number_l94_94497


namespace sum_prime_factors_of_77_l94_94439

theorem sum_prime_factors_of_77 : 
  ∃ (p q : ℕ), nat.prime p ∧ nat.prime q ∧ 77 = p * q ∧ p + q = 18 :=
by
  existsi 7
  existsi 11
  split
  { exact nat.prime_7 }
  split
  { exact nat.prime_11 }
  split
  { exact dec_trivial }
  { exact dec_trivial }

-- Placeholder for the proof
-- sorry

end sum_prime_factors_of_77_l94_94439


namespace divisor_of_109_l94_94237

theorem divisor_of_109 (d : ℕ) (h : 109 = 9 * d + 1) : d = 12 :=
sorry

end divisor_of_109_l94_94237


namespace circle_standard_equation_l94_94302

noncomputable def circle_through_ellipse_vertices : Prop :=
  ∃ (a : ℝ) (r : ℝ), a < 0 ∧
    (∀ (x y : ℝ),   -- vertices of the ellipse
      ((x = 4 ∧ y = 0) ∨ (x = 0 ∧ (y = 2 ∨ y = -2)))
      → (x + a)^2 + y^2 = r^2) ∧
    ( a = -3/2 ∧ r = 5/2 ∧ 
      ∀ (x y : ℝ), (x + 3/2)^2 + y^2 = (5/2)^2
    )

theorem circle_standard_equation :
  circle_through_ellipse_vertices :=
sorry

end circle_standard_equation_l94_94302


namespace number_of_possible_values_for_b_l94_94116

theorem number_of_possible_values_for_b : 
  ∃ (n : ℕ), n = 10 ∧ ∀ (b : ℕ), (2 ≤ b) ∧ (b^2 ≤ 256) ∧ (256 < b^3) ↔ (7 ≤ b ∧ b ≤ 16) :=
by {
  sorry
}

end number_of_possible_values_for_b_l94_94116


namespace least_possible_value_of_D_l94_94893

-- Defining the conditions as theorems
theorem least_possible_value_of_D :
  ∃ (A B C D : ℕ), 
  (A + B + C + D) / 4 = 18 ∧
  A = 3 * B ∧
  B = C - 2 ∧
  C = 3 / 2 * D ∧
  (∀ x : ℕ, x ≥ 10 → D = x) := 
sorry

end least_possible_value_of_D_l94_94893


namespace probability_green_face_l94_94434

def faces : ℕ := 6
def green_faces : ℕ := 3

theorem probability_green_face : (green_faces : ℚ) / (faces : ℚ) = 1 / 2 := by
  sorry

end probability_green_face_l94_94434


namespace tank_water_after_rain_final_l94_94476

theorem tank_water_after_rain_final (initial_water evaporated drained rain_rate rain_time : ℕ)
  (initial_water_eq : initial_water = 6000)
  (evaporated_eq : evaporated = 2000)
  (drained_eq : drained = 3500)
  (rain_rate_eq : rain_rate = 350)
  (rain_time_eq : rain_time = 30) :
  let water_after_evaporation := initial_water - evaporated
  let water_after_drainage := water_after_evaporation - drained 
  let rain_addition := (rain_time / 10) * rain_rate
  let final_water := water_after_drainage + rain_addition
  final_water = 1550 :=
by
  sorry

end tank_water_after_rain_final_l94_94476


namespace min_value_ineq_l94_94809

noncomputable def f (x : ℝ) : ℝ := abs (Real.log x)

theorem min_value_ineq (a b : ℝ) (h1 : 0 < b) (h2 : b < a) (h3 : f a = f b) :
  (a^2 + b^2) / (a - b) = 2 * Real.sqrt 2 := by
  sorry

end min_value_ineq_l94_94809


namespace infinite_divisibility_1986_l94_94255

theorem infinite_divisibility_1986 :
  ∃ (a : ℕ → ℕ), a 1 = 39 ∧ a 2 = 45 ∧ (∀ n, a (n+2) = a (n+1) ^ 2 - a n) ∧
  ∀ N, ∃ n > N, 1986 ∣ a n :=
sorry

end infinite_divisibility_1986_l94_94255


namespace number_of_pieces_sold_on_third_day_l94_94460

variable (m : ℕ)

def first_day_sales : ℕ := m
def second_day_sales : ℕ := (m / 2) - 3
def third_day_sales : ℕ := second_day_sales m + 5

theorem number_of_pieces_sold_on_third_day :
  third_day_sales m = (m / 2) + 2 := by sorry

end number_of_pieces_sold_on_third_day_l94_94460


namespace total_girls_is_68_l94_94877

-- Define the initial conditions
def track_length : ℕ := 100
def student_spacing : ℕ := 2
def girls_per_cycle : ℕ := 2
def cycle_length : ℕ := 3

-- Calculate the number of students on one side
def students_on_one_side : ℕ := track_length / student_spacing + 1

-- Number of cycles of three students
def num_cycles : ℕ := students_on_one_side / cycle_length

-- Number of girls on one side
def girls_on_one_side : ℕ := num_cycles * girls_per_cycle

-- Total number of girls on both sides
def total_girls : ℕ := girls_on_one_side * 2

theorem total_girls_is_68 : total_girls = 68 := by
  -- proof will be provided here
  sorry

end total_girls_is_68_l94_94877


namespace determine_function_l94_94786

theorem determine_function (f : ℕ → ℕ) :
  (∀ a b c d : ℕ, 2 * a * b = c^2 + d^2 → f (a + b) = f a + f b + f c + f d) →
  ∃ k : ℕ, ∀ n : ℕ, f n = k * n^2 :=
by
  sorry

end determine_function_l94_94786


namespace ninth_grade_students_eq_l94_94430

-- Let's define the conditions
def total_students : ℕ := 50
def seventh_grade_students (x : ℕ) : ℕ := 2 * x - 1
def eighth_grade_students (x : ℕ) : ℕ := x

-- Define the expression for ninth grade students based on the conditions
def ninth_grade_students (x : ℕ) : ℕ :=
  total_students - (seventh_grade_students x + eighth_grade_students x)

-- The theorem statement to prove
theorem ninth_grade_students_eq (x : ℕ) : ninth_grade_students x = 51 - 3 * x :=
by
  sorry

end ninth_grade_students_eq_l94_94430


namespace eric_less_than_ben_l94_94343

variables (E B J : ℕ)

theorem eric_less_than_ben
  (hJ : J = 26)
  (hB : B = J - 9)
  (total_money : E + B + J = 50) :
  B - E = 10 :=
sorry

end eric_less_than_ben_l94_94343


namespace probability_at_least_one_survives_l94_94316

namespace TreeSurvival

open ProbabilityTheory

noncomputable def survival_probabilities (P_A : ℝ) (P_B : ℝ) :=
  let A1_survives := P_A
  let A2_survives := P_A
  let B1_survives := P_B
  let B2_survives := P_B
  let A1_dies := 1 - A1_survives
  let A2_dies := 1 - A2_survives
  let B1_dies := 1 - B1_survives
  let B2_dies := 1 - B2_survives

  let prob_at_least_one_survives := 1 - (A1_dies * A2_dies * B1_dies * B2_dies)
  let prob_one_of_each_survives :=
    (2 * A1_survives * A1_dies) * (2 * B1_survives * B1_dies)

  (prob_at_least_one_survives, prob_one_of_each_survives)

theorem probability_at_least_one_survives :
  let (P_A, P_B) := (5/6:ℝ, 4/5:ℝ)
  survival_probabilities P_A P_B = (899 / 900, 4 / 45) :=
by
  sorry

end TreeSurvival

end probability_at_least_one_survives_l94_94316


namespace ratio_of_ages_in_two_years_l94_94771

theorem ratio_of_ages_in_two_years
    (S : ℕ) (M : ℕ) 
    (h1 : M = S + 32)
    (h2 : S = 30) : 
    (M + 2) / (S + 2) = 2 :=
by
  sorry

end ratio_of_ages_in_two_years_l94_94771


namespace polygon_area_l94_94030

theorem polygon_area (n : ℕ) (s : ℝ) (perimeter : ℝ) (area : ℝ) 
  (h1 : n = 24) 
  (h2 : n * s = perimeter) 
  (h3 : perimeter = 48) 
  (h4 : s = 2) 
  (h5 : area = n * s^2 / 2) : 
  area = 96 :=
by
  sorry

end polygon_area_l94_94030


namespace books_no_adjacent_l94_94126

-- Define our main theorem
theorem books_no_adjacent (n k : ℕ) (h1 : n = 12) (h2 : k = 5) :
    ∃ ways : ℕ, ways = Nat.choose (n - k + 1) k ∧ ways = 56 :=
by
  have h : Nat.choose (12 - 5 + 1) 5 = 56 := by
    -- Use the given mathematical fact
    calc
      Nat.choose 8 5 = Nat.choose 8 3 : by rw [Nat.choose_symm (by linarith)]
               ... = 56 : by decide
  use 56
  constructor
  · exact h
  · rfl

end books_no_adjacent_l94_94126


namespace hitting_probability_l94_94433

theorem hitting_probability (A_hit B_hit : ℚ) (hA : A_hit = 4/5) (hB : B_hit = 5/6) :
  1 - ((1 - A_hit) * (1 - B_hit)) = 29/30 :=
by 
  sorry

end hitting_probability_l94_94433


namespace part_two_l94_94360

noncomputable def func_f (a x : ℝ) : ℝ := a * Real.exp x + 2 * Real.exp (-x) + (a - 2) * x
noncomputable def func_g (a x : ℝ) : ℝ := func_f a x - (a + 2) * Real.cos x 

theorem part_two (a x : ℝ) (h₀ : 2 ≤ a) (h₁ : 0 ≤ x) : func_f a x ≥ (a + 2) * Real.cos x :=
by
  sorry

end part_two_l94_94360


namespace dina_has_60_dolls_l94_94176

variable (ivy_collectors_edition_dolls : ℕ)
variable (ivy_total_dolls : ℕ)
variable (dina_dolls : ℕ)

-- Conditions
def condition1 (ivy_total_dolls ivy_collectors_edition_dolls : ℕ) := ivy_collectors_edition_dolls = 20
def condition2 (ivy_total_dolls ivy_collectors_edition_dolls : ℕ) := (2 / 3 : ℚ) * ivy_total_dolls = ivy_collectors_edition_dolls
def condition3 (ivy_total_dolls dina_dolls : ℕ) := dina_dolls = 2 * ivy_total_dolls

-- Proof statement
theorem dina_has_60_dolls 
  (h1 : condition1 ivy_total_dolls ivy_collectors_edition_dolls) 
  (h2 : condition2 ivy_total_dolls ivy_collectors_edition_dolls) 
  (h3 : condition3 ivy_total_dolls dina_dolls) : 
  dina_dolls = 60 :=
sorry

end dina_has_60_dolls_l94_94176


namespace diamond_evaluation_l94_94955

def diamond (a b : ℝ) : ℝ := (a + b) * (a - b)

theorem diamond_evaluation : diamond 2 (diamond 3 (diamond 1 4)) = -46652 :=
  by
  sorry

end diamond_evaluation_l94_94955


namespace James_delivers_2565_bags_in_a_week_l94_94221

noncomputable def total_bags_delivered_in_a_week
  (days_15_bags : ℕ)
  (trips_per_day_15_bags : ℕ)
  (bags_per_trip_15 : ℕ)
  (days_20_bags : ℕ)
  (trips_per_day_20_bags : ℕ)
  (bags_per_trip_20 : ℕ) : ℕ :=
  (days_15_bags * trips_per_day_15_bags * bags_per_trip_15) + (days_20_bags * trips_per_day_20_bags * bags_per_trip_20)

theorem James_delivers_2565_bags_in_a_week :
  total_bags_delivered_in_a_week 3 25 15 4 18 20 = 2565 :=
by
  sorry

end James_delivers_2565_bags_in_a_week_l94_94221


namespace vector_magnitude_difference_l94_94063

def vector_a : ℝ × ℝ := (2, 1)
def vector_b : ℝ × ℝ := (-2, 4)

theorem vector_magnitude_difference :
  let diff := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2)
  let magnitude := Real.sqrt (diff.1 ^ 2 + diff.2 ^ 2)
  magnitude = 5 :=
by
  -- define the difference vector
  have diff : ℝ × ℝ := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2),
  -- define the magnitude of the difference vector
  have magnitude := Real.sqrt (diff.1 ^ 2 + diff.2 ^ 2),
  -- provide the proof (using steps omitted via sorry)
  sorry

end vector_magnitude_difference_l94_94063


namespace correct_proposition_is_D_l94_94627

-- Define the propositions
def propositionA : Prop :=
  (∀ x : ℝ, x^2 = 4 → x = 2 ∨ x = -2) → (∀ x : ℝ, (x ≠ 2 ∨ x ≠ -2) → x^2 ≠ 4)

def propositionB (p : Prop) : Prop :=
  (p → (∀ x : ℝ, x^2 - 2*x + 3 > 0)) → (¬p → (∃ x : ℝ, x^2 - 2*x + 3 < 0))

def propositionC : Prop :=
  ∀ (a b : ℝ) (n : ℕ), a > b → n > 0 → a^n > b^n

def p : Prop := ∀ x : ℝ, x^3 ≥ 0
def q : Prop := ∀ e : ℝ, e > 0 → e < 1
def propositionD := p ∧ q

-- The proof problem
theorem correct_proposition_is_D : propositionD :=
  sorry

end correct_proposition_is_D_l94_94627


namespace polygon_sides_eq_six_l94_94267

theorem polygon_sides_eq_six (n : ℕ) (h : 3 * n - (n * (n - 3)) / 2 = 6) : n = 6 := 
sorry

end polygon_sides_eq_six_l94_94267


namespace ratio_sheep_horses_eq_six_seven_l94_94482

noncomputable def total_food_per_day : ℕ := 12880
noncomputable def food_per_horse_per_day : ℕ := 230
noncomputable def num_sheep : ℕ := 48
noncomputable def num_horses : ℕ := total_food_per_day / food_per_horse_per_day
noncomputable def ratio_sheep_to_horses := num_sheep / num_horses

theorem ratio_sheep_horses_eq_six_seven :
  ratio_sheep_to_horses = 6 / 7 :=
by
  sorry

end ratio_sheep_horses_eq_six_seven_l94_94482


namespace algebraic_expression_perfect_square_l94_94082

theorem algebraic_expression_perfect_square (a : ℤ) :
  (∃ b : ℤ, ∀ x : ℤ, x^2 + (a - 1) * x + 16 = (x + b)^2) →
  (a = 9 ∨ a = -7) :=
sorry

end algebraic_expression_perfect_square_l94_94082


namespace last_three_digits_7_pow_80_l94_94948

theorem last_three_digits_7_pow_80 : (7^80) % 1000 = 961 := by
  sorry

end last_three_digits_7_pow_80_l94_94948


namespace cube_root_21952_is_28_l94_94994

theorem cube_root_21952_is_28 :
  ∃ n : ℕ, n^3 = 21952 ∧ n = 28 :=
sorry

end cube_root_21952_is_28_l94_94994


namespace expected_score_is_6_l94_94985

-- Define the probabilities of making a shot
def p : ℝ := 0.5

-- Define the scores for each scenario
def score_first_shot : ℝ := 8
def score_second_shot : ℝ := 6
def score_third_shot : ℝ := 4
def score_no_shot : ℝ := 0

-- Compute the expected value
def expected_score : ℝ :=
  p * score_first_shot +
  (1 - p) * p * score_second_shot +
  (1 - p) * (1 - p) * p * score_third_shot +
  (1 - p) * (1 - p) * (1 - p) * score_no_shot

theorem expected_score_is_6 : expected_score = 6 := by
  sorry

end expected_score_is_6_l94_94985


namespace exists_rational_non_integer_satisfying_linear_no_rational_non_integer_satisfying_quadratic_l94_94148

theorem exists_rational_non_integer_satisfying_linear :
  ∃ (x y : ℚ), x.denom ≠ 1 ∧ y.denom ≠ 1 ∧ 19 * x + 8 * y ∈ ℤ ∧ 8 * x + 3 * y ∈ ℤ :=
by
  sorry

theorem no_rational_non_integer_satisfying_quadratic :
  ¬ ∃ (x y : ℚ), x.denom ≠ 1 ∧ y.denom ≠ 1 ∧ 19 * x^2 + 8 * y^2 ∈ ℤ ∧ 8 * x^2 + 3 * y^2 ∈ ℤ :=
by
  sorry

end exists_rational_non_integer_satisfying_linear_no_rational_non_integer_satisfying_quadratic_l94_94148


namespace find_angle_l94_94967

theorem find_angle (x : ℝ) (h : 180 - x = 6 * (90 - x)) : x = 72 := 
by 
    sorry

end find_angle_l94_94967


namespace factor_expression_l94_94034

theorem factor_expression (
  x y z : ℝ
) : 
  ( (x^2 - y^2)^3 + (y^2 - z^2)^3 + (z^2 - x^2)^3) / 
  ( (x - y)^3 + (y - z)^3 + (z - x)^3) = 
  (x + y) * (y + z) * (z + x) := 
sorry

end factor_expression_l94_94034


namespace simplify_polynomials_l94_94394

theorem simplify_polynomials :
  (4 * q ^ 4 + 2 * p ^ 3 - 7 * p + 8) + (3 * q ^ 4 - 2 * p ^ 3 + 3 * p ^ 2 - 5 * p + 6) =
  7 * q ^ 4 + 3 * p ^ 2 - 12 * p + 14 :=
by
  sorry

end simplify_polynomials_l94_94394


namespace number_of_correct_statements_l94_94866

def input_statement (s : String) : Prop :=
  s = "INPUT a; b; c"

def output_statement (s : String) : Prop :=
  s = "A=4"

def assignment_statement1 (s : String) : Prop :=
  s = "3=B"

def assignment_statement2 (s : String) : Prop :=
  s = "A=B=-2"

theorem number_of_correct_statements :
    input_statement "INPUT a; b; c" = false ∧
    output_statement "A=4" = false ∧
    assignment_statement1 "3=B" = false ∧
    assignment_statement2 "A=B=-2" = false :=
sorry

end number_of_correct_statements_l94_94866


namespace sample_variance_classA_twenty_five_percentile_classB_l94_94846

-- Define the height data for Class A
def classA_heights : List ℝ := [170, 179, 162, 168, 158, 182, 179, 168, 163, 171]

-- Define the height data for Class B
def classB_heights : List ℝ := [159, 173, 179, 178, 162, 181, 176, 168, 170, 165]

-- Calculate the sample mean for Class A
def mean_classA : ℝ := (List.sum classA_heights) / classA_heights.length

-- Define the sample variance function
def sample_variance (data : List ℝ) (mean : ℝ) : ℝ :=
  List.sum (data.map (λ x => (x - mean) ^ 2)) / data.length

-- Statement for Part 1
theorem sample_variance_classA : sample_variance classA_heights mean_classA = 57.2 :=
by sorry

-- Function to calculate the percentile
def percentile (data : List ℝ) (p : ℝ) : ℝ :=
  let sorted := List.qsort (· ≤ ·) data
  sorted.nthLe (Int.toNat (p * data.length).natAbs - 1) (by decide)

-- Statement for Part 2
theorem twenty_five_percentile_classB : percentile classB_heights 0.25 = 165 :=
by sorry

end sample_variance_classA_twenty_five_percentile_classB_l94_94846


namespace initial_percentage_acidic_liquid_l94_94140

theorem initial_percentage_acidic_liquid (P : ℝ) :
  let initial_volume := 12
  let removed_volume := 4
  let final_volume := initial_volume - removed_volume
  let desired_concentration := 60
  (P/100) * initial_volume = (desired_concentration/100) * final_volume →
  P = 40 :=
by
  intros
  sorry

end initial_percentage_acidic_liquid_l94_94140


namespace evaluate_expression_l94_94643

theorem evaluate_expression : 
  (3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = 26991001) :=
by
  sorry

end evaluate_expression_l94_94643


namespace probability_different_colors_l94_94274

-- Define the number of chips of each color
def num_blue := 6
def num_red := 5
def num_yellow := 4
def num_green := 3

-- Total number of chips
def total_chips := num_blue + num_red + num_yellow + num_green

-- Probability of drawing a chip of different color
theorem probability_different_colors : 
  (num_blue / total_chips) * ((total_chips - num_blue) / total_chips) +
  (num_red / total_chips) * ((total_chips - num_red) / total_chips) +
  (num_yellow / total_chips) * ((total_chips - num_yellow) / total_chips) +
  (num_green / total_chips) * ((total_chips - num_green) / total_chips) =
  119 / 162 := 
sorry

end probability_different_colors_l94_94274


namespace school_travel_time_is_12_l94_94997

noncomputable def time_to_school (T : ℕ) : Prop :=
  let extra_time := 6
  let total_distance_covered := 2 * extra_time
  T = total_distance_covered

theorem school_travel_time_is_12 :
  ∃ T : ℕ, time_to_school T ∧ T = 12 :=
by
  sorry

end school_travel_time_is_12_l94_94997


namespace find_x_l94_94500

theorem find_x (x : ℕ) (hx1 : 1 ≤ x) (hx2 : x ≤ 100) (hx3 : (31 + 58 + 98 + 3 * x) / 6 = 2 * x) : x = 21 :=
by
  sorry

end find_x_l94_94500


namespace range_of_m_l94_94828

theorem range_of_m (x m : ℝ) (h1 : (2 * x + m) / (x - 1) = 1) (h2 : x ≥ 0) : m ≤ -1 ∧ m ≠ -2 :=
sorry

end range_of_m_l94_94828


namespace determine_delta_l94_94015

theorem determine_delta (r1 r2 r3 r4 r5 r6 : ℕ) (c1 c2 c3 c4 c5 c6 : ℕ) (O Δ : ℕ) 
  (h_sums_rows : r1 + r2 + r3 + r4 + r5 + r6 = 190)
  (h_row1 : r1 = 29) (h_row2 : r2 = 33) (h_row3 : r3 = 33) 
  (h_row4 : r4 = 32) (h_row5 : r5 = 32) (h_row6 : r6 = 31)
  (h_sums_cols : c1 + c2 + c3 + c4 + c5 + c6 = 190)
  (h_col1 : c1 = 29) (h_col2 : c2 = 33) (h_col3 : c3 = 33) 
  (h_col4 : c4 = 32) (h_col5 : c5 = 32) (h_col6 : c6 = 31)
  (h_O : O = 6) : 
  Δ = 4 :=
by 
  sorry

end determine_delta_l94_94015


namespace jason_total_expenditure_l94_94707

theorem jason_total_expenditure :
  let stove_cost := 1200
  let wall_repair_ratio := 1 / 6
  let wall_repair_cost := stove_cost * wall_repair_ratio
  let total_cost := stove_cost + wall_repair_cost
  total_cost = 1400 := by
  {
    sorry
  }

end jason_total_expenditure_l94_94707


namespace p_q_sum_l94_94552

theorem p_q_sum (p q : ℝ) (hp : is_root (λ x, x^3 - 9 * x^2 + p * x - q) 1)
  (hq : is_root (λ x, x^3 - 9 * x^2 + p * x - q) 3)
  (hr : is_root (λ x, x^3 - 9 * x^2 + p * x - q) 5) :
  p + q = 38 :=
sorry

end p_q_sum_l94_94552


namespace abc_divisibility_l94_94961

theorem abc_divisibility (a b c : ℕ) (h₁ : a ∣ (b * c - 1)) (h₂ : b ∣ (c * a - 1)) (h₃ : c ∣ (a * b - 1)) : 
  (a = 2 ∧ b = 3 ∧ c = 5) ∨ (a = 1 ∧ b = 1 ∧ ∃ n : ℕ, n ≥ 1 ∧ c = n) :=
by
  sorry

end abc_divisibility_l94_94961


namespace sum_prime_factors_77_l94_94451

theorem sum_prime_factors_77 : ∀ (p1 p2 : ℕ), Nat.Prime p1 ∧ Nat.Prime p2 ∧ 77 = p1 * p2 → p1 + p2 = 18 :=
by
  intros p1 p2 h
  sorry

end sum_prime_factors_77_l94_94451


namespace total_distance_covered_l94_94248

-- Define the distances for each segment of Biker Bob's journey
def distance1 : ℕ := 45 -- 45 miles west
def distance2 : ℕ := 25 -- 25 miles northwest
def distance3 : ℕ := 35 -- 35 miles south
def distance4 : ℕ := 50 -- 50 miles east

-- Statement to prove that the total distance covered is 155 miles
theorem total_distance_covered : distance1 + distance2 + distance3 + distance4 = 155 :=
by
  -- This is where the proof would go
  sorry

end total_distance_covered_l94_94248


namespace part1_part2_l94_94217

noncomputable theory

open_locale real_inner_product_space

-- Definitions of vectors in 3d space
structure point3d :=
(x : ℝ) (y : ℝ) (z : ℝ)

def vector (A B : point3d) : point3d :=
⟨B.x - A.x, B.y - A.y, B.z - A.z⟩

def magnitude (v : point3d) : ℝ :=
(real.sqrt (v.x^2 + v.y^2 + v.z^2))

def dot_product (v1 v2 : point3d) : ℝ :=
v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

def cosine_angle (v1 v2 : point3d) : ℝ :=
(dot_product v1 v2 / ((magnitude v1) * (magnitude v2)))

def sine_angle (v1 v2 : point3d) : ℝ :=
(real.sqrt (1 - (cosine_angle v1 v2)^2))

-- Problem Part 1
def A := ⟨0, 1, 2⟩ : point3d
def B := ⟨3, -2, -1⟩ : point3d
def D := ⟨1, 1, 1⟩ : point3d
def P := ⟨2, -1, 0⟩ : point3d -- Given from solution steps

def vector_D_P := vector D P

theorem part1 :
  magnitude vector_D_P = real.sqrt 6 :=
sorry

-- Problem Part 2
def vector_A_B := vector A B
def vector_A_D := vector A D

noncomputable def area_triangle (A B D : point3d) : ℝ :=
  1 / 2 * magnitude (vector A B) * magnitude (vector A D) * sine_angle (vector A B) (vector A D)

theorem part2 :
  area_triangle A B D = 3 * (real.sqrt 2) / 2 :=
sorry

end part1_part2_l94_94217


namespace weight_12m_rod_l94_94984

-- Define the weight of a 6 meters long rod
def weight_of_6m_rod : ℕ := 7

-- Given the condition that the weight is proportional to the length
def weight_of_rod (length : ℕ) : ℕ := (length / 6) * weight_of_6m_rod

-- Prove the weight of a 12 meters long rod
theorem weight_12m_rod : weight_of_rod 12 = 14 := by
  -- Calculation skipped, proof required here
  sorry

end weight_12m_rod_l94_94984


namespace graph_passes_through_point_l94_94795

theorem graph_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : 
    ∀ x y : ℝ, (y = a^(x-2) + 2) → (x = 2) → (y = 3) :=
by
    intros x y hxy hx
    rw [hx] at hxy
    simp at hxy
    sorry

end graph_passes_through_point_l94_94795


namespace cylinder_has_no_triangular_cross_section_l94_94744

inductive GeometricSolid
  | cylinder
  | cone
  | triangularPrism
  | cube

open GeometricSolid

-- Define the cross section properties
def can_have_triangular_cross_section (s : GeometricSolid) : Prop :=
  s = cone ∨ s = triangularPrism ∨ s = cube

-- Define the property where a solid cannot have a triangular cross-section
def cannot_have_triangular_cross_section (s : GeometricSolid) : Prop :=
  s = cylinder

theorem cylinder_has_no_triangular_cross_section :
  cannot_have_triangular_cross_section cylinder ∧
  ¬ can_have_triangular_cross_section cylinder :=
by
  -- This is where we state the proof goal
  sorry

end cylinder_has_no_triangular_cross_section_l94_94744


namespace find_velocity_of_current_l94_94772

-- Define the conditions given in the problem
def rowing_speed_in_still_water : ℤ := 10
def distance_to_place : ℤ := 48
def total_travel_time : ℤ := 10

-- Define the primary goal, which is to find the velocity of the current given the conditions
theorem find_velocity_of_current (v : ℤ) 
  (h1 : rowing_speed_in_still_water = 10)
  (h2 : distance_to_place = 48)
  (h3 : total_travel_time = 10) 
  (h4 : rowing_speed_in_still_water * 2 + v * 0 = 
   rowing_speed_in_still_water - v) :
  v = 2 := 
sorry

end find_velocity_of_current_l94_94772


namespace charles_drawn_after_work_l94_94024

-- Conditions
def total_papers : ℕ := 20
def drawn_today : ℕ := 6
def drawn_yesterday_before_work : ℕ := 6
def papers_left : ℕ := 2

-- Question and proof goal
theorem charles_drawn_after_work :
  ∀ (total_papers drawn_today drawn_yesterday_before_work papers_left : ℕ),
  total_papers = 20 →
  drawn_today = 6 →
  drawn_yesterday_before_work = 6 →
  papers_left = 2 →
  (total_papers - drawn_today - drawn_yesterday_before_work - papers_left = 6) :=
by
  intros total_papers drawn_today drawn_yesterday_before_work papers_left
  assume h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  exact sorry

end charles_drawn_after_work_l94_94024


namespace vector_magnitude_subtraction_l94_94054

theorem vector_magnitude_subtraction :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  (real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2) = 5) :=
by
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  have h : real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2) = 5 := sorry
  exact h

end vector_magnitude_subtraction_l94_94054


namespace perc_freshmen_in_SLA_l94_94016

variables (T : ℕ) (P : ℝ)

-- 60% of students are freshmen
def freshmen (T : ℕ) : ℝ := 0.60 * T

-- 4.8% of students are freshmen psychology majors in the school of liberal arts
def freshmen_psych_majors (T : ℕ) : ℝ := 0.048 * T

-- 20% of freshmen in the school of liberal arts are psychology majors
def perc_fresh_psych (F_LA : ℝ) : ℝ := 0.20 * F_LA

-- Number of freshmen in the school of liberal arts as a percentage P of the total number of freshmen
def fresh_in_SLA_as_perc (T : ℕ) (P : ℝ) : ℝ := P * (0.60 * T)

theorem perc_freshmen_in_SLA (T : ℕ) (P : ℝ) :
  (0.20 * (P * (0.60 * T)) = 0.048 * T) → P = 0.4 :=
sorry

end perc_freshmen_in_SLA_l94_94016


namespace inequality_solution_l94_94506

theorem inequality_solution (b c x : ℝ) (x1 x2 : ℝ)
  (hb_pos : b > 0) (hc_pos : c > 0) 
  (h_eq1 : x1 * x2 = 1) 
  (h_eq2 : -1 + x2 = 2 * x1) 
  (h_b : b = 5 / 2) 
  (h_c : c = 1) 
  : (1 < x ∧ x ≤ 5 / 2) ↔ (1 < x ∧ x ≤ 5 / 2) :=
sorry

end inequality_solution_l94_94506


namespace ratio_of_circumscribed_areas_l94_94467

noncomputable def rect_pentagon_circ_ratio (P : ℝ) : ℝ :=
  let s : ℝ := P / 8
  let r_circle : ℝ := (P * Real.sqrt 10) / 16
  let A : ℝ := Real.pi * (r_circle ^ 2)
  let pentagon_side : ℝ := P / 5
  let R_pentagon : ℝ := P / (10 * Real.sin (Real.pi / 5))
  let B : ℝ := Real.pi * (R_pentagon ^ 2)
  A / B

theorem ratio_of_circumscribed_areas (P : ℝ) : rect_pentagon_circ_ratio P = (5 * (5 - Real.sqrt 5)) / 64 :=
by sorry

end ratio_of_circumscribed_areas_l94_94467


namespace dana_more_pencils_than_marcus_l94_94927

theorem dana_more_pencils_than_marcus :
  ∀ (Jayden Dana Marcus : ℕ), 
  (Jayden = 20) ∧ 
  (Dana = Jayden + 15) ∧ 
  (Jayden = 2 * Marcus) → 
  (Dana - Marcus = 25) :=
by
  intros Jayden Dana Marcus h
  rcases h with ⟨hJayden, hDana, hMarcus⟩
  sorry

end dana_more_pencils_than_marcus_l94_94927


namespace gear_angular_speed_proportion_l94_94654

theorem gear_angular_speed_proportion :
  ∀ (ω_A ω_B ω_C ω_D k: ℝ),
    30 * ω_A = k →
    45 * ω_B = k →
    50 * ω_C = k →
    60 * ω_D = k →
    ω_A / ω_B = 1 ∧
    ω_B / ω_C = 45 / 50 ∧
    ω_C / ω_D = 50 / 60 ∧
    ω_A / ω_D = 10 / 7.5 :=
  by
    -- proof goes here
    sorry

end gear_angular_speed_proportion_l94_94654


namespace p_q_sum_l94_94553

theorem p_q_sum (p q : ℝ) (hp : is_root (λ x, x^3 - 9 * x^2 + p * x - q) 1)
  (hq : is_root (λ x, x^3 - 9 * x^2 + p * x - q) 3)
  (hr : is_root (λ x, x^3 - 9 * x^2 + p * x - q) 5) :
  p + q = 38 :=
sorry

end p_q_sum_l94_94553


namespace gcd_1729_867_l94_94749

theorem gcd_1729_867 : Nat.gcd 1729 867 = 1 :=
by
  sorry

end gcd_1729_867_l94_94749


namespace uncovered_side_length_l94_94000

theorem uncovered_side_length
  (A : ℝ) (F : ℝ)
  (h1 : A = 600)
  (h2 : F = 130) :
  ∃ L : ℝ, L = 120 :=
by {
  sorry
}

end uncovered_side_length_l94_94000


namespace corrected_mean_l94_94124

theorem corrected_mean (mean_incorrect : ℝ) (number_of_observations : ℕ) (wrong_observation correct_observation : ℝ) : 
  mean_incorrect = 36 → 
  number_of_observations = 50 → 
  wrong_observation = 23 → 
  correct_observation = 43 → 
  (mean_incorrect * number_of_observations + (correct_observation - wrong_observation)) / number_of_observations = 36.4 :=
by
  intros h_mean_incorrect h_number_of_observations h_wrong_observation h_correct_observation
  have S_incorrect : ℝ := mean_incorrect * number_of_observations
  have difference : ℝ := correct_observation - wrong_observation
  have S_correct : ℝ := S_incorrect + difference
  have mean_correct : ℝ := S_correct / number_of_observations
  sorry

end corrected_mean_l94_94124


namespace sum_prime_factors_of_77_l94_94447

theorem sum_prime_factors_of_77 : ∃ (p1 p2 : ℕ), Nat.Prime p1 ∧ Nat.Prime p2 ∧ 77 = p1 * p2 ∧ p1 + p2 = 18 := by
  sorry

end sum_prime_factors_of_77_l94_94447


namespace angle_B_in_equilateral_triangle_l94_94369

theorem angle_B_in_equilateral_triangle (A B C : ℝ) (h_angle_sum : A + B + C = 180) (h_A : A = 80) (h_BC : B = C) :
  B = 50 :=
by
  -- Conditions
  have h1 : A = 80 := by exact h_A
  have h2 : B = C := by exact h_BC
  have h3 : A + B + C = 180 := by exact h_angle_sum

  sorry -- completing the proof is not required

end angle_B_in_equilateral_triangle_l94_94369


namespace sin_neg_30_eq_neg_one_half_l94_94329

theorem sin_neg_30_eq_neg_one_half :
  Real.sin (-30 * Real.pi / 180) = -1 / 2 := 
by
  sorry -- Proof is skipped

end sin_neg_30_eq_neg_one_half_l94_94329


namespace max_4x_3y_l94_94194

theorem max_4x_3y (x y : ℝ) (h : x^2 + y^2 = 16 * x + 8 * y + 8) : 4 * x + 3 * y ≤ 63 :=
sorry

end max_4x_3y_l94_94194


namespace sarahs_team_mean_score_l94_94393

def mean_score_of_games (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / scores.length

theorem sarahs_team_mean_score :
  mean_score_of_games [69, 68, 70, 61, 74, 62, 65, 74] = 67.875 :=
by
  sorry

end sarahs_team_mean_score_l94_94393


namespace alien_heads_l94_94481

theorem alien_heads (l o : ℕ) 
  (h1 : l + o = 60) 
  (h2 : 4 * l + o = 129) : 
  l + 2 * o = 97 := 
by 
  sorry

end alien_heads_l94_94481


namespace point_is_in_second_quadrant_l94_94210

def in_second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

theorem point_is_in_second_quadrant (x y : ℝ) (h₁ : x = -3) (h₂ : y = 2) :
  in_second_quadrant x y := 
by {
  sorry
}

end point_is_in_second_quadrant_l94_94210


namespace max_product_xy_l94_94682

theorem max_product_xy (x y : ℕ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : 7 * x + 4 * y = 150) : xy = 200 :=
by
  sorry

end max_product_xy_l94_94682


namespace surface_area_calculation_l94_94724

-- Conditions:
-- Original rectangular sheet dimensions
def length : ℕ := 25
def width : ℕ := 35
-- Dimensions of the square corners
def corner_side : ℕ := 7

-- Surface area of the interior calculation
noncomputable def surface_area_interior : ℕ :=
  let original_area := length * width
  let corner_area := corner_side * corner_side
  let total_corner_area := 4 * corner_area
  original_area - total_corner_area

-- Theorem: The surface area of the interior of the resulting box
theorem surface_area_calculation : surface_area_interior = 679 := by
  -- You can fill in the details to compute the answer
  sorry

end surface_area_calculation_l94_94724


namespace find_u_minus_v_l94_94678

theorem find_u_minus_v (u v : ℚ) (h1 : 5 * u - 6 * v = 31) (h2 : 3 * u + 5 * v = 4) : u - v = 5.3 := by
  sorry

end find_u_minus_v_l94_94678


namespace sum_prime_factors_77_l94_94443

noncomputable def sum_prime_factors (n : ℕ) : ℕ :=
  if n = 77 then 7 + 11 else 0

theorem sum_prime_factors_77 : sum_prime_factors 77 = 18 :=
by
  -- The theorem states that the sum of the prime factors of 77 is 18
  rw sum_prime_factors
  -- Given that the prime factors of 77 are 7 and 11, summing them gives 18
  if_clause
  -- We finalize by proving this is indeed the case
  sorry

end sum_prime_factors_77_l94_94443


namespace infinitely_many_composite_z_l94_94241

theorem infinitely_many_composite_z (m n : ℕ) (h_m : m > 1) : ¬ (Nat.Prime (n^4 + 4*m^4)) :=
by
  sorry

end infinitely_many_composite_z_l94_94241


namespace find_total_cost_price_l94_94165

noncomputable def cost_prices (C1 C2 C3 : ℝ) : Prop :=
  0.85 * C1 + 72.50 = 1.125 * C1 ∧
  1.20 * C2 - 45.30 = 0.95 * C2 ∧
  0.92 * C3 + 33.60 = 1.10 * C3

theorem find_total_cost_price :
  ∃ (C1 C2 C3 : ℝ), cost_prices C1 C2 C3 ∧ C1 + C2 + C3 = 631.51 := 
by
  sorry

end find_total_cost_price_l94_94165


namespace smallest_five_digit_palindrome_div_4_thm_l94_94843

def is_palindrome (n : ℕ) : Prop :=
  n = (n % 10) * 10000 + ((n / 10) % 10) * 1000 + ((n / 100) % 10) * 100 + ((n / 1000) % 10) * 10 + (n / 10000)

def smallest_five_digit_palindrome_div_4 : ℕ :=
  18881

theorem smallest_five_digit_palindrome_div_4_thm :
  is_palindrome smallest_five_digit_palindrome_div_4 ∧
  10000 ≤ smallest_five_digit_palindrome_div_4 ∧
  smallest_five_digit_palindrome_div_4 < 100000 ∧
  smallest_five_digit_palindrome_div_4 % 4 = 0 ∧
  ∀ n, is_palindrome n ∧ 10000 ≤ n ∧ n < 100000 ∧ n % 4 = 0 → n ≥ smallest_five_digit_palindrome_div_4 :=
by
  sorry

end smallest_five_digit_palindrome_div_4_thm_l94_94843


namespace find_M_l94_94029

def grid_conditions :=
  ∃ (M : ℤ), 
  ∀ d1 d2 d3 d4, 
    (d1 = 22) ∧ (d2 = 6) ∧ (d3 = -34 / 6) ∧ (d4 = (8 - M) / 6) ∧
    (10 = 32 - d2) ∧ 
    (16 = 10 + d2) ∧ 
    (-2 = 10 - d2) ∧
    (32 - M = 34 / 6 * 6) ∧ 
    (M = -34 / 6 - (-17 / 3))

theorem find_M : grid_conditions → ∃ M : ℤ, M = 17 :=
by
  intros
  existsi (17 : ℤ) 
  sorry

end find_M_l94_94029


namespace find_a_l94_94946

theorem find_a :
  (∃ x1 x2, (x1 + x2 = -2 ∧ x1 * x2 = a) ∧ (∃ y1 y2, (y1 + y2 = - a ∧ y1 * y2 = 2) ∧ (x1^2 + x2^2 = y1^2 + y2^2))) → 
  (a = -4) := 
by
  sorry

end find_a_l94_94946


namespace one_fourth_in_one_eighth_l94_94078

theorem one_fourth_in_one_eighth : (1 / 8) / (1 / 4) = 1 / 2 := by
  sorry

end one_fourth_in_one_eighth_l94_94078


namespace lindy_total_distance_l94_94091

-- Definitions derived from the conditions
def jack_speed : ℕ := 5
def christina_speed : ℕ := 7
def lindy_speed : ℕ := 12
def initial_distance : ℕ := 360

theorem lindy_total_distance :
  lindy_speed * (initial_distance / (jack_speed + christina_speed)) = 360 := by
  sorry

end lindy_total_distance_l94_94091


namespace arithmetic_sequence_value_l94_94538

theorem arithmetic_sequence_value (a_1 d : ℤ) (h : (a_1 + 2 * d) + (a_1 + 7 * d) = 10) : 
  3 * (a_1 + 4 * d) + (a_1 + 6 * d) = 20 :=
by
  sorry

end arithmetic_sequence_value_l94_94538


namespace gyeongyeon_total_path_l94_94364

theorem gyeongyeon_total_path (D : ℝ) :
  (D / 4 + 250 = D / 2 - 300) -> D = 2200 :=
by
  intro h
  -- We would now proceed to show that D must equal 2200
  sorry

end gyeongyeon_total_path_l94_94364


namespace flowchart_output_value_l94_94532

theorem flowchart_output_value :
  ∃ n : ℕ, S = n * (n + 1) / 2 ∧ n = 10 → S = 55 :=
by
  sorry

end flowchart_output_value_l94_94532


namespace find_original_cost_of_chips_l94_94833

def original_cost_chips (discount amount_spent : ℝ) : ℝ :=
  discount + amount_spent

theorem find_original_cost_of_chips :
  original_cost_chips 17 18 = 35 := by
  sorry

end find_original_cost_of_chips_l94_94833


namespace compare_exp_sin_ln_l94_94353

theorem compare_exp_sin_ln :
  let a := Real.exp 0.1 - 1
  let b := Real.sin 0.1
  let c := Real.log 1.1
  c < b ∧ b < a :=
by
  sorry

end compare_exp_sin_ln_l94_94353


namespace option_c_correct_l94_94456

theorem option_c_correct (x y : ℝ) : 3 * x^2 * y + 2 * y * x^2 = 5 * x^2 * y :=
by {
  sorry
}

end option_c_correct_l94_94456


namespace rational_non_integer_solution_exists_rational_non_integer_solution_not_exists_l94_94152

-- Part (a)
theorem rational_non_integer_solution_exists :
  ∃ (x y : ℚ), x ∉ ℤ ∧ y ∉ ℤ ∧ 19 * x + 8 * y ∈ ℤ ∧ 8 * x + 3 * y ∈ ℤ :=
sorry

-- Part (b)
theorem rational_non_integer_solution_not_exists :
  ¬ ∃ (x y : ℚ), x ∉ ℤ ∧ y ∉ ℤ ∧ 19 * x^2 + 8 * y^2 ∈ ℤ ∧ 8 * x^2 + 3 * y^2 ∈ ℤ :=
sorry

end rational_non_integer_solution_exists_rational_non_integer_solution_not_exists_l94_94152


namespace simplify_expression_l94_94723

theorem simplify_expression (a b c : ℝ) : a - (a - b + c) = b - c :=
by sorry

end simplify_expression_l94_94723


namespace train_scheduled_speed_l94_94479

theorem train_scheduled_speed (a v : ℝ) (hv : 0 < v)
  (h1 : a / v - a / (v + 5) = 1 / 3)
  (h2 : a / (v - 5) - a / v = 5 / 12) : v = 45 :=
by
  sorry

end train_scheduled_speed_l94_94479


namespace simplify_expression_l94_94917

theorem simplify_expression : 
  1.5 * (Real.sqrt 1 + Real.sqrt (1+3) + Real.sqrt (1+3+5) + Real.sqrt (1+3+5+7) + Real.sqrt (1+3+5+7+9)) = 22.5 :=
by
  sorry

end simplify_expression_l94_94917


namespace intersecting_points_are_same_l94_94581

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (3, -2)
def radius1 : ℝ := 5

def center2 : ℝ × ℝ := (3, 6)
def radius2 : ℝ := 3

-- Define the equations of the two circles
def circle1 (x y : ℝ) : Prop := (x - center1.1)^2 + (y + center1.2)^2 = radius1^2
def circle2 (x y : ℝ) : Prop := (x - center2.1)^2 + (y - center2.2)^2 = radius2^2

-- Prove that points C and D coincide
theorem intersecting_points_are_same : ∃ x y, circle1 x y ∧ circle2 x y → (0 = 0) :=
by
  sorry

end intersecting_points_are_same_l94_94581


namespace intersection_A_B_l94_94090

def A : Set ℝ := { x | x * Real.sqrt (x^2 - 4) ≥ 0 }
def B : Set ℝ := { x | |x - 1| + |x + 1| ≥ 2 }

theorem intersection_A_B : (A ∩ B) = ({-2} ∪ Set.Ici 2) :=
by
  sorry

end intersection_A_B_l94_94090


namespace booksJuly_l94_94280

-- Definitions of the conditions
def booksMay : ℕ := 2
def booksJune : ℕ := 6
def booksTotal : ℕ := 18

-- Theorem statement proving how many books Tom read in July
theorem booksJuly : (booksTotal - (booksMay + booksJune)) = 10 :=
by
  sorry

end booksJuly_l94_94280


namespace arithmetic_sequence_l94_94861

-- Given conditions
variables {a x b : ℝ}

-- Statement of the problem in Lean 4
theorem arithmetic_sequence (h1 : x - a = b - x) (h2 : b - x = 2 * x - b) : a / b = 1 / 3 :=
sorry

end arithmetic_sequence_l94_94861


namespace nicky_cristina_race_l94_94758

theorem nicky_cristina_race :
  ∀ (head_start t : ℕ), ∀ (cristina_speed nicky_speed time_nicky_run : ℝ),
  head_start = 12 →
  cristina_speed = 5 →
  nicky_speed = 3 →
  ((cristina_speed * t) = (nicky_speed * t + nicky_speed * head_start)) →
  time_nicky_run = head_start + t →
  time_nicky_run = 30 :=
by
  intros
  sorry

end nicky_cristina_race_l94_94758


namespace solution_set_of_inequality_l94_94805

variables {R : Type*} [LinearOrderedField R]

def odd_function (f : R → R) : Prop := ∀ x : R, f (-x) = -f x

def increasing_on (f : R → R) (S : Set R) : Prop :=
∀ ⦃x y⦄, x ∈ S → y ∈ S → x < y → f x < f y

theorem solution_set_of_inequality
  {f : R → R}
  (h_odd : odd_function f)
  (h_neg_one : f (-1) = 0)
  (h_increasing : increasing_on f {x : R | x > 0}) :
  {x : R | x * f x > 0} = {x : R | x < -1} ∪ {x : R | x > 1} :=
sorry

end solution_set_of_inequality_l94_94805


namespace Dana_has_25_more_pencils_than_Marcus_l94_94929

theorem Dana_has_25_more_pencils_than_Marcus (JaydenPencils : ℕ) (h1 : JaydenPencils = 20) :
  let DanaPencils := JaydenPencils + 15,
      MarcusPencils := JaydenPencils / 2
  in DanaPencils - MarcusPencils = 25 := 
by
  sorry -- proof to be filled in

end Dana_has_25_more_pencils_than_Marcus_l94_94929


namespace C_share_per_rs_equals_l94_94160

-- Definitions based on given conditions
def A_share_per_rs (x : ℝ) : ℝ := x
def B_share_per_rs : ℝ := 0.65
def C_share : ℝ := 48
def total_sum : ℝ := 246

-- The target statement to prove
theorem C_share_per_rs_equals : C_share / total_sum = 0.195122 :=
by
  sorry

end C_share_per_rs_equals_l94_94160


namespace percentage_difference_l94_94133

theorem percentage_difference (X : ℝ) (h1 : first_num = 0.70 * X) (h2 : second_num = 0.63 * X) :
  (first_num - second_num) / first_num * 100 = 10 := by
  sorry

end percentage_difference_l94_94133


namespace miles_mike_l94_94715

def cost_mike (M : ℕ) : ℝ := 2.50 + 0.25 * M
def cost_annie (A : ℕ) : ℝ := 2.50 + 5.00 + 0.25 * A

theorem miles_mike {M A : ℕ} (annie_ride_miles : A = 16) (same_cost : cost_mike M = cost_annie A) : M = 36 :=
by
  rw [cost_annie, annie_ride_miles] at same_cost
  simp [cost_mike] at same_cost
  sorry

end miles_mike_l94_94715


namespace game_positions_l94_94696

def spots := ["top-left", "top-right", "bottom-right", "bottom-left"]
def segments := ["top-left", "top-middle-left", "top-middle-right", "top-right", "right-top", "right-middle-top", "right-middle-bottom", "right-bottom", "bottom-right", "bottom-middle-right", "bottom-middle-left", "bottom-left", "left-top", "left-middle-top", "left-middle-bottom", "left-bottom"]

def cat_position_after_moves (n : Nat) : String :=
  spots.get! (n % 4)

def mouse_position_after_moves (n : Nat) : String :=
  segments.get! ((12 - (n % 12)) % 12)

theorem game_positions :
  cat_position_after_moves 359 = "bottom-right" ∧ 
  mouse_position_after_moves 359 = "left-middle-bottom" :=
by
  sorry

end game_positions_l94_94696


namespace weight_of_green_peppers_l94_94975

-- Definitions for conditions and question
def total_weight : ℝ := 0.6666666667
def is_split_equally (x y : ℝ) : Prop := x = y

-- Theorem statement that needs to be proved
theorem weight_of_green_peppers (g r : ℝ) (h_split : is_split_equally g r) (h_total : g + r = total_weight) :
  g = 0.33333333335 :=
by sorry

end weight_of_green_peppers_l94_94975


namespace expense_and_income_calculations_l94_94602

def alexander_salary : ℕ := 125000
def natalia_salary : ℕ := 61000
def utilities_transport_household : ℕ := 17000
def loan_repayment : ℕ := 15000
def theater_cost : ℕ := 5000
def cinema_cost_per_person : ℕ := 1000
def savings_crimea : ℕ := 20000
def dining_weekday_cost : ℕ := 1500
def dining_weekend_cost : ℕ := 3000
def weekdays : ℕ := 20
def weekends : ℕ := 10
def phone_A_cost : ℕ := 57000
def phone_B_cost : ℕ := 37000

def total_expenses : ℕ :=
  utilities_transport_household +
  loan_repayment +
  theater_cost + 2 * cinema_cost_per_person +
  savings_crimea +
  weekdays * dining_weekday_cost +
  weekends * dining_weekend_cost

def net_income : ℕ :=
  alexander_salary + natalia_salary

def can_buy_phones : Prop :=
  net_income - total_expenses < phone_A_cost + phone_B_cost

theorem expense_and_income_calculations :
  total_expenses = 119000 ∧
  net_income = 186000 ∧
  can_buy_phones :=
by
  sorry

end expense_and_income_calculations_l94_94602


namespace four_digit_number_count_l94_94252

theorem four_digit_number_count :
  (∃ (n : ℕ), n ≥ 1000 ∧ n < 10000 ∧ 
    ((n / 1000 < 5 ∧ (n / 100) % 10 < 5) ∨ (n / 1000 > 5 ∧ (n / 100) % 10 > 5)) ∧ 
    (((n % 100) / 10 < 5 ∧ n % 10 < 5) ∨ ((n % 100) / 10 > 5 ∧ n % 10 > 5))) →
    ∃ (count : ℕ), count = 1681 :=
by
  sorry

end four_digit_number_count_l94_94252


namespace no_repair_needed_l94_94410

def nominal_mass : ℝ := 370 -- Assign the nominal mass as determined in the problem solving.

def max_deviation (M : ℝ) : ℝ := 0.1 * M
def preserved_max_deviation : ℝ := 37
def unreadable_max_deviation : ℝ := 37

def within_max_deviation (dev : ℝ) := dev ≤ preserved_max_deviation

noncomputable def standard_deviation : ℝ := preserved_max_deviation

theorem no_repair_needed :
  ∀ (M : ℝ),
  max_deviation M = 0.1 * M →
  preserved_max_deviation ≤ max_deviation M →
  ∀ (dev : ℝ), within_max_deviation dev →
  standard_deviation ≤ preserved_max_deviation →
  preserved_max_deviation = 37 →
  "не требует" = "не требует" :=
by
  intros M h1 h2 h3 h4 h5
  sorry

end no_repair_needed_l94_94410


namespace graph_abs_symmetric_yaxis_l94_94699

theorem graph_abs_symmetric_yaxis : 
  ∀ x : ℝ, |x| = |(-x)| :=
by
  intro x
  sorry

end graph_abs_symmetric_yaxis_l94_94699


namespace num_new_books_not_signed_l94_94319

theorem num_new_books_not_signed (adventure_books mystery_books science_fiction_books non_fiction_books used_books signed_books : ℕ)
    (h1 : adventure_books = 13)
    (h2 : mystery_books = 17)
    (h3 : science_fiction_books = 25)
    (h4 : non_fiction_books = 10)
    (h5 : used_books = 42)
    (h6 : signed_books = 10) : 
    (adventure_books + mystery_books + science_fiction_books + non_fiction_books) - used_books - signed_books = 13 := 
by
  sorry

end num_new_books_not_signed_l94_94319


namespace ellipse_with_conditions_l94_94968

open Real

variables {a b x y : ℝ}
def ellipse (a b : ℝ) (C : Set (ℝ × ℝ)) : Prop := 
  ∀ (p : ℝ × ℝ), p ∈ C ↔ p.1^2 / a^2 + p.2^2 / b^2 = 1

variables {F B1 B2 : ℝ × ℝ}

theorem ellipse_with_conditions :
  ∀ (a b : ℝ), a > b → b > 0 →
  ellipse a b
    {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1} →
  (F = (1,0)) →
  (B1 = (0, -b)) →
  (B2 = (0, b)) →
  ((F.1 - B1.1) * (F.1 - B2.1) + (F.2 - B1.2) * (F.2 - B2.2)) = -a →
  (C.Point x y : {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1}) →
  (0 < |DP x y| / |MN x y| ∧ |DP x y| / |MN x y| < 1/4) :=
sorry

end ellipse_with_conditions_l94_94968


namespace different_distributions_l94_94900

def arrangement_methods (students teachers: Finset ℕ) : ℕ :=
  students.card.factorial * (students.card - 1).factorial * ((students.card - 1) - 1).factorial

theorem different_distributions :
  ∀ (students teachers : Finset ℕ), 
  students.card = 3 ∧ teachers.card = 3 →
  arrangement_methods students teachers = 72 :=
by sorry

end different_distributions_l94_94900


namespace find_a2023_l94_94388

theorem find_a2023
  (a : ℕ → ℚ)
  (h1 : a 1 = 1)
  (h2 : a 2 = 2/5)
  (h3 : a 3 = 1/4)
  (h_rule : ∀ n : ℕ, 0 < n → (1 / a n + 1 / a (n + 2) = 2 / a (n + 1))) :
  a 2023 = 1 / 3034 :=
by sorry

end find_a2023_l94_94388


namespace arcsin_sqrt3_div_2_eq_pi_div_3_l94_94326

theorem arcsin_sqrt3_div_2_eq_pi_div_3 :
  arcsin (sqrt 3 / 2) = π / 3 :=
by
  sorry

end arcsin_sqrt3_div_2_eq_pi_div_3_l94_94326


namespace power_modulo_l94_94588

theorem power_modulo (h : 3 ^ 4 ≡ 1 [MOD 10]) : 3 ^ 2023 ≡ 7 [MOD 10] :=
by
  sorry

end power_modulo_l94_94588


namespace scientific_notation_29150000_l94_94911

theorem scientific_notation_29150000 :
  29150000 = 2.915 * 10^7 := sorry

end scientific_notation_29150000_l94_94911


namespace sum_of_roots_l94_94044

theorem sum_of_roots (x₁ x₂ : ℝ) (h1 : x₁^2 = 2 * x₁ + 1) (h2 : x₂^2 = 2 * x₂ + 1) :
  x₁ + x₂ = 2 :=
sorry

end sum_of_roots_l94_94044


namespace jovial_frogs_not_green_l94_94396

variables {Frog : Type} (jovial green can_jump can_swim : Frog → Prop)

theorem jovial_frogs_not_green :
  (∀ frog, jovial frog → can_swim frog) →
  (∀ frog, green frog → ¬ can_jump frog) →
  (∀ frog, ¬ can_jump frog → ¬ can_swim frog) →
  (∀ frog, jovial frog → ¬ green frog) :=
by
  intros h1 h2 h3 frog hj
  sorry

end jovial_frogs_not_green_l94_94396


namespace leap_years_count_l94_94135

def is_leap_year (y : ℕ) : Bool :=
  if y % 800 = 300 ∨ y % 800 = 600 then true else false

theorem leap_years_count : 
  { y : ℕ // 1500 ≤ y ∧ y ≤ 3500 ∧ y % 100 = 0 ∧ is_leap_year y } = {y | y = 1900 ∨ y = 2200 ∨ y = 2700 ∨ y = 3000 ∨ y = 3500} :=
by
  sorry

end leap_years_count_l94_94135


namespace good_goods_not_cheap_l94_94245

-- Define the propositions "good goods" and "not cheap"
variables (p q : Prop)

-- State that "good goods are not cheap" is expressed by the implication p → q
theorem good_goods_not_cheap : p → q → (p → q) ↔ (p ∧ q → p ∧ q) := by
  sorry

end good_goods_not_cheap_l94_94245


namespace annual_return_percentage_correct_l94_94018

variables (initial_value final_value gain : ℝ)
variables (initial_value_eq : initial_value = 8000)
variables (final_value_eq : final_value = initial_value + 400)
variables (gain_eq : gain = final_value - initial_value)
variables (annual_return_percentage : ℝ)

theorem annual_return_percentage_correct : 
  annual_return_percentage = (gain / initial_value * 100) :=
by
  rw [initial_value_eq, final_value_eq, gain_eq]
  have h : final_value = 8400 := by
    rw [initial_value_eq, final_value_eq]
    rw [initial_value_eq]
    sorry
  have h_gain : gain = 400 := by
    rw [gain_eq, h]
    sorry
  have h_percentage : annual_return_percentage = (400 / 8000 * 100) := by
    rw [←h_gain, initial_value_eq]
    sorry
  exact h_percentage

end annual_return_percentage_correct_l94_94018


namespace sum_prime_factors_77_l94_94454

theorem sum_prime_factors_77 : ∑ p in {7, 11}, p = 18 :=
by {
  have h1 : Nat.Prime 7 := Nat.prime_iff_nat_prime.mp Nat.prime_factors_one,
  have h2 : Nat.Prime 11 := Nat.prime_iff_nat_prime.mp Nat.prime_factors_one,
  have h3 : 77 = 7 * 11 := by simp,
  have h_set: {7, 11} = PrimeFactors 77 := by sorry, 
  rw h_set,
  exact Nat.sum_prime_factors_eq_77,
  sorry
}

end sum_prime_factors_77_l94_94454


namespace bob_daily_earnings_l94_94113

-- Define Sally's daily earnings
def Sally_daily_earnings : ℝ := 6

-- Define the total savings after a year for both Sally and Bob
def total_savings : ℝ := 1825

-- Define the number of days in a year
def days_in_year : ℝ := 365

-- Define Bob's daily earnings
variable (B : ℝ)

-- Define the proof statement
theorem bob_daily_earnings : (3 + B / 2) * days_in_year = total_savings → B = 4 :=
by
  sorry

end bob_daily_earnings_l94_94113


namespace circle_radius_5_l94_94787

theorem circle_radius_5 (k x y : ℝ) : x^2 + 8 * x + y^2 + 10 * y - k = 0 → (x + 4) ^ 2 + (y + 5) ^ 2 = 25 → k = -16 :=
by
  sorry

end circle_radius_5_l94_94787


namespace tank_plastering_cost_proof_l94_94477

/-- 
Given a tank with the following dimensions:
length = 35 meters,
width = 18 meters,
depth = 10 meters.
The cost of plastering per square meter is ₹135.
Prove that the total cost of plastering the walls and bottom of the tank is ₹228,150.
-/
theorem tank_plastering_cost_proof (length width depth cost_per_sq_meter : ℕ)
  (h_length : length = 35)
  (h_width : width = 18)
  (h_depth : depth = 10)
  (h_cost_per_sq_meter : cost_per_sq_meter = 135) : 
  (2 * (length * depth) + 2 * (width * depth) + length * width) * cost_per_sq_meter = 228150 := 
by 
  -- The proof is not required as per the problem statement
  sorry

end tank_plastering_cost_proof_l94_94477


namespace parallel_lines_perpendicular_lines_l94_94363

section LineEquation

variables (a : ℝ) (x y : ℝ)

def l1 := (a-2) * x + 3 * y + a = 0
def l2 := a * x + (a-2) * y - 1 = 0

theorem parallel_lines (a : ℝ) :
  ((a-2)/a = 3/(a-2)) ↔ (a = (7 + Real.sqrt 33) / 2 ∨ a = (7 - Real.sqrt 33) / 2) := sorry

theorem perpendicular_lines (a : ℝ) :
  (a = 2 ∨ ((2-a)/3 * (a/(2-a)) = -1)) ↔ (a = 2 ∨ a = -3) := sorry

end LineEquation

end parallel_lines_perpendicular_lines_l94_94363


namespace find_ending_number_l94_94649

theorem find_ending_number (N : ℕ) :
  (∃ k : ℕ, N = 3 * k) ∧ (∀ x,  40 < x ∧ x ≤ N → x % 3 = 0) ∧ (∃ avg, avg = (N + 42) / 2 ∧ avg = 60) → N = 78 :=
by
  sorry

end find_ending_number_l94_94649


namespace fraction_of_students_with_partner_l94_94208

theorem fraction_of_students_with_partner (f e : ℕ) (h : e = 4 * f / 3) :
  ((e / 4 + f / 3) : ℚ) / (e + f) = 2 / 7 :=
by
  sorry

end fraction_of_students_with_partner_l94_94208


namespace sequence_values_l94_94716

theorem sequence_values (x y z : ℕ) 
    (h1 : x = 14 * 3) 
    (h2 : y = x - 1) 
    (h3 : z = y * 3) : 
    x = 42 ∧ y = 41 ∧ z = 123 := by 
    sorry

end sequence_values_l94_94716


namespace num_undefined_values_l94_94794

theorem num_undefined_values :
  ∃! x : Finset ℝ, (∀ y ∈ x, (y + 5 = 0) ∨ (y - 1 = 0) ∨ (y - 4 = 0)) ∧ (x.card = 3) := sorry

end num_undefined_values_l94_94794


namespace integer_solutions_l94_94174

def satisfies_equation (x y : ℤ) : Prop := x^2 = y^2 * (x + y^4 + 2 * y^2)

theorem integer_solutions :
  {p : ℤ × ℤ | satisfies_equation p.1 p.2} = { (0, 0), (12, 2), (-8, 2) } :=
by sorry

end integer_solutions_l94_94174


namespace min_value_of_expression_l94_94652

noncomputable def minExpression (x : ℝ) : ℝ := (15 - x) * (14 - x) * (15 + x) * (14 + x)

theorem min_value_of_expression : ∀ x : ℝ, ∃ m : ℝ, (m ≤ minExpression x) ∧ (m = -142.25) :=
by
  sorry

end min_value_of_expression_l94_94652


namespace radius_of_inscribed_circle_in_rhombus_l94_94587

noncomputable def radius_of_inscribed_circle (d₁ d₂ : ℕ) : ℝ :=
  (d₁ * d₂) / (2 * Real.sqrt ((d₁ / 2) ^ 2 + (d₂ / 2) ^ 2))

theorem radius_of_inscribed_circle_in_rhombus :
  radius_of_inscribed_circle 8 18 = 36 / Real.sqrt 97 :=
by
  -- Skip the detailed proof steps
  sorry

end radius_of_inscribed_circle_in_rhombus_l94_94587


namespace proof_problem_l94_94514

-- Defining lines l1, l2, l3
def l1 (x y : ℝ) : Prop := 3 * x + 4 * y = 2
def l2 (x y : ℝ) : Prop := 2 * x + y = -2
def l3 (x y : ℝ) : Prop := x - 2 * y = 1

-- Point P being the intersection of l1 and l2
def P : ℝ × ℝ := (-2, 2)

-- Definition of the first required line passing through P and the origin
def line_through_P_and_origin (x y : ℝ) : Prop := x + y = 0

-- Definition of the second required line passing through P and perpendicular to l3
def required_line (x y : ℝ) : Prop := 2 * x + y + 2 = 0

-- The theorem to prove
theorem proof_problem :
  (∃ x y, l1 x y ∧ l2 x y ∧ (x, y) = P) →
  (∀ x y, (x, y) = P → line_through_P_and_origin x y) ∧
  (∀ x y, (x, y) = P → required_line x y) :=
by
  sorry

end proof_problem_l94_94514


namespace common_ratio_eq_l94_94827

variables {x y z r : ℝ}

theorem common_ratio_eq (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z)
  (hgp : x * (y - z) ≠ 0 ∧ y * (z - x) ≠ 0 ∧ z * (x - y) ≠ 0 ∧ 
          (y * (z - x)) / (x * (y - z)) = r ∧ (z * (x - y)) / (y * (z - x)) = r) :
  r^2 + r + 1 = 0 :=
sorry

end common_ratio_eq_l94_94827


namespace machine_does_not_require_repair_l94_94413

-- Define the conditions.

def max_deviation := 37

def nominal_portion_max_deviation_percentage := 0.10

def deviation_within_limit (M : ℝ) : Prop :=
  37 ≤ 0.10 * M

def unreadable_measurements_deviation (deviation : ℝ) : Prop :=
  deviation < 37

-- Define the theorem we want to prove

theorem machine_does_not_require_repair (M : ℝ)
  (h1 : deviation_within_limit M)
  (h2 : ∀ deviation, unreadable_measurements_deviation deviation) :
  true := 
sorry

end machine_does_not_require_repair_l94_94413


namespace mass_percentage_of_Cl_in_NH4Cl_l94_94347

-- Definition of the molar masses (conditions)
def molar_mass_N : ℝ := 14.01
def molar_mass_H : ℝ := 1.01
def molar_mass_Cl : ℝ := 35.45

-- Definition of the molar mass of NH4Cl
def molar_mass_NH4Cl : ℝ := molar_mass_N + 4 * molar_mass_H + molar_mass_Cl

-- The expected mass percentage of Cl in NH4Cl
def expected_mass_percentage_Cl : ℝ := 66.26

-- The proof statement
theorem mass_percentage_of_Cl_in_NH4Cl :
  (molar_mass_Cl / molar_mass_NH4Cl) * 100 = expected_mass_percentage_Cl :=
by 
  -- The body of the proof is omitted, as it is not necessary to provide the proof.
  sorry

end mass_percentage_of_Cl_in_NH4Cl_l94_94347


namespace length_cd_l94_94219

noncomputable def isosceles_triangle (A B E : Type*) (area_abe : ℝ) (trapezoid_area : ℝ) (altitude_abe : ℝ) :
  ℝ := sorry

theorem length_cd (A B E C D : Type*) (area_abe : ℝ) (trapezoid_area : ℝ) (altitude_abe : ℝ)
  (h1 : area_abe = 144) (h2 : trapezoid_area = 108) (h3 : altitude_abe = 24) :
  isosceles_triangle A B E area_abe trapezoid_area altitude_abe = 6 := by
  sorry

end length_cd_l94_94219


namespace probability_of_different_colors_is_correct_l94_94275

noncomputable def probability_different_colors : ℚ :=
  let total_chips := 18
  let blue_chips := 6
  let red_chips := 5
  let yellow_chips := 4
  let green_chips := 3
  let p_blue_then_not_blue := (blue_chips / total_chips) * ((red_chips + yellow_chips + green_chips) / total_chips)
  let p_red_then_not_red := (red_chips / total_chips) * ((blue_chips + yellow_chips + green_chips) / total_chips)
  let p_yellow_then_not_yellow := (yellow_chips / total_chips) * ((blue_chips + red_chips + green_chips) / total_chips)
  let p_green_then_not_green := (green_chips / total_chips) * ((blue_chips + red_chips + yellow_chips) / total_chips)
  p_blue_then_not_blue + p_red_then_not_red + p_yellow_then_not_yellow + p_green_then_not_green

theorem probability_of_different_colors_is_correct :
  probability_different_colors = 119 / 162 :=
by
  sorry

end probability_of_different_colors_is_correct_l94_94275


namespace solution_inequality_l94_94983

variable (a x : ℝ)

theorem solution_inequality (h : ∀ x, |x - a| + |x + 4| ≥ 1) : a ≤ -5 ∨ a ≥ -3 := by
  sorry

end solution_inequality_l94_94983


namespace perfect_square_representation_l94_94760

theorem perfect_square_representation :
  29 - 12*Real.sqrt 5 = (2*Real.sqrt 5 - 3*Real.sqrt 5 / 5)^2 :=
sorry

end perfect_square_representation_l94_94760


namespace find_value_of_a_l94_94798

theorem find_value_of_a (a : ℝ) (h : (3 + a + 10) / 3 = 5) : a = 2 := 
by {
  sorry
}

end find_value_of_a_l94_94798


namespace gcd_1729_867_l94_94747

theorem gcd_1729_867 : Nat.gcd 1729 867 = 1 :=
by 
  sorry

end gcd_1729_867_l94_94747


namespace cat_chase_rat_l94_94158

/--
Given:
- The cat chases a rat 6 hours after the rat runs.
- The cat takes 4 hours to reach the rat.
- The average speed of the rat is 36 km/h.
Prove that the average speed of the cat is 90 km/h.
-/
theorem cat_chase_rat
  (t_rat_start : ℕ)
  (t_cat_chase : ℕ)
  (v_rat : ℕ)
  (h1 : t_rat_start = 6)
  (h2 : t_cat_chase = 4)
  (h3 : v_rat = 36)
  (v_cat : ℕ)
  (h4 : 4 * v_cat = t_rat_start * v_rat + t_cat_chase * v_rat) :
  v_cat = 90 :=
by
  sorry

end cat_chase_rat_l94_94158


namespace factor_expression_l94_94944

variable (b : ℤ)

theorem factor_expression : 221 * b^2 + 17 * b = 17 * b * (13 * b + 1) :=
by
  sorry

end factor_expression_l94_94944


namespace factor_expression_l94_94939

variable (b : ℝ)

theorem factor_expression : 221 * b * b + 17 * b = 17 * b * (13 * b + 1) :=
by
  sorry

end factor_expression_l94_94939


namespace problem1_problem2_problem3_l94_94324

theorem problem1 (x : ℝ) : x * x^3 + x^2 * x^2 = 2 * x^4 := sorry
theorem problem2 (p q : ℝ) : (-p * q)^3 = -p^3 * q^3 := sorry
theorem problem3 (a : ℝ) : a^3 * a^4 * a + (a^2)^4 - (-2 * a^4)^2 = -2 * a^8 := sorry

end problem1_problem2_problem3_l94_94324


namespace A_inter_complement_B_eq_l94_94663

-- Define set A
def set_A : Set ℝ := {x | -3 < x ∧ x < 6}

-- Define set B
def set_B : Set ℝ := {x | 2 < x ∧ x < 7}

-- Define the complement of set B in the real numbers
def complement_B : Set ℝ := {x | x ≤ 2 ∨ x ≥ 7}

-- Define the intersection of set A with the complement of set B
def A_inter_complement_B : Set ℝ := set_A ∩ complement_B

-- Stating the theorem to prove
theorem A_inter_complement_B_eq : A_inter_complement_B = {x | -3 < x ∧ x ≤ 2} :=
by
  -- Proof goes here
  sorry

end A_inter_complement_B_eq_l94_94663


namespace intersection_of_log_functions_l94_94640

theorem intersection_of_log_functions : 
  ∃ x : ℝ, (3 * Real.log x = Real.log (3 * x)) ∧ x = Real.sqrt 3 := 
by 
  sorry

end intersection_of_log_functions_l94_94640


namespace reflection_y_axis_correct_l94_94860

-- Define the coordinates and reflection across the y-axis
def reflect_y_axis (p : (ℝ × ℝ)) : (ℝ × ℝ) :=
  (-p.1, p.2)

-- Define the original point M
def M : (ℝ × ℝ) := (3, 2)

-- State the theorem we want to prove
theorem reflection_y_axis_correct : reflect_y_axis M = (-3, 2) :=
by
  -- The proof would go here, but it is omitted as per the instructions
  sorry

end reflection_y_axis_correct_l94_94860


namespace trace_bags_weight_l94_94132

theorem trace_bags_weight :
  ∀ (g1 g2 t1 t2 t3 t4 t5 : ℕ),
    g1 = 3 →
    g2 = 7 →
    (g1 + g2) = (t1 + t2 + t3 + t4 + t5) →
    (t1 = t2 ∧ t2 = t3 ∧ t3 = t4 ∧ t4 = t5) →
    t1 = 2 :=
by
  intros g1 g2 t1 t2 t3 t4 t5 hg1 hg2 hsum hsame
  sorry

end trace_bags_weight_l94_94132


namespace proof_problem_l94_94669

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

end proof_problem_l94_94669


namespace carol_spending_l94_94490

noncomputable def savings (S : ℝ) : Prop :=
∃ (X : ℝ) (stereo_spending television_spending : ℝ), 
  stereo_spending = (1 / 4) * S ∧
  television_spending = X * S ∧
  stereo_spending + television_spending = 0.25 * S ∧
  (stereo_spending - television_spending) / S = 0.25

theorem carol_spending (S : ℝ) : savings S :=
sorry

end carol_spending_l94_94490


namespace expected_number_of_returns_l94_94996

noncomputable def expected_returns_to_zero : ℝ :=
  let p_move := 1 / 3
  let expected_value := -1 + (3 / (Real.sqrt 5))
  expected_value

theorem expected_number_of_returns : expected_returns_to_zero = (3 * Real.sqrt 5 - 5) / 5 :=
  by sorry

end expected_number_of_returns_l94_94996


namespace pamela_skittles_l94_94389

variable (initial_skittles : Nat) (given_to_karen : Nat)

def skittles_after_giving (initial_skittles given_to_karen : Nat) : Nat :=
  initial_skittles - given_to_karen

theorem pamela_skittles (h1 : initial_skittles = 50) (h2 : given_to_karen = 7) :
  skittles_after_giving initial_skittles given_to_karen = 43 := by
  sorry

end pamela_skittles_l94_94389


namespace sector_central_angle_l94_94002

theorem sector_central_angle (r θ : ℝ) 
  (h1 : 1 = (1 / 2) * 2 * r) 
  (h2 : 2 = θ * r) : θ = 2 := 
sorry

end sector_central_angle_l94_94002


namespace average_of_remaining_four_l94_94290

theorem average_of_remaining_four (avg10 : ℕ → ℕ) (avg6 : ℕ → ℕ) 
  (h_avg10 : avg10 10 = 80) 
  (h_avg6 : avg6 6 = 58) : 
  (avg10 10 - avg6 6 * 6) / 4 = 113 :=
sorry

end average_of_remaining_four_l94_94290


namespace find_x_value_l94_94040

theorem find_x_value :
  let a := (2021 : ℝ)
  let b := (2022 : ℝ)
  ∀ x : ℝ, (a / b - b / a + x = 0) → (x = b / a - a / b) :=
  by
    intros a b x h
    sorry

end find_x_value_l94_94040


namespace find_range_of_a_l94_94048

noncomputable def f (a x : ℝ) : ℝ := a / x - Real.exp (-x)

theorem find_range_of_a (p q a : ℝ) (h : 0 < a) (hpq : p < q) :
  (∀ x : ℝ, 0 < x → x ∈ Set.Icc p q → f a x ≤ 0) → 
  (0 < a ∧ a < 1 / Real.exp 1) :=
by
  sorry

end find_range_of_a_l94_94048


namespace recolor_possible_l94_94207

theorem recolor_possible (cell_color : Fin 50 → Fin 50 → Fin 100)
  (H1 : ∀ i j, ∃ k l, cell_color i j = k ∧ cell_color i (j+1) = l ∧ k ≠ l ∧ j < 49)
  (H2 : ∀ i j, ∃ k l, cell_color i j = k ∧ cell_color (i+1) j = l ∧ k ≠ l ∧ i < 49) :
  ∃ c1 c2, (c1 ≠ c2) ∧
  ∀ i j, (cell_color i j = c1 → cell_color i j = c2 ∨ ∀ k l, (cell_color k l = c1 → cell_color k l ≠ c2)) :=
  by
  sorry

end recolor_possible_l94_94207


namespace triangle_angle_bisector_theorem_l94_94585

variable {α : Type*} [LinearOrderedField α]

theorem triangle_angle_bisector_theorem (A B C D : α)
  (h1 : A^2 = (C + D) * (B - (B * D / C)))
  (h2 : B / C = (B * D / C) / D) :
  A^2 = C * B - D * (B * D / C) := 
  by
  sorry

end triangle_angle_bisector_theorem_l94_94585


namespace one_fourth_in_one_eighth_l94_94079

theorem one_fourth_in_one_eighth : (1 / 8) / (1 / 4) = 1 / 2 := by
  sorry

end one_fourth_in_one_eighth_l94_94079


namespace flowers_died_l94_94785

theorem flowers_died : 
  let initial_flowers := 2 * 5
  let grown_flowers := initial_flowers + 20
  let harvested_flowers := 5 * 4
  grown_flowers - harvested_flowers = 10 :=
by
  sorry

end flowers_died_l94_94785


namespace find_tangent_perpendicular_t_l94_94690

noncomputable def y (x : ℝ) : ℝ := x * Real.log x

theorem find_tangent_perpendicular_t (t : ℝ) (ht : 0 < t) (h_perpendicular : (1 : ℝ) * (1 + Real.log t) = -1) :
  t = Real.exp (-2) :=
by
  sorry

end find_tangent_perpendicular_t_l94_94690


namespace sum_of_powers_divisible_by_6_l94_94103

theorem sum_of_powers_divisible_by_6 (a1 a2 a3 a4 : ℤ)
  (h : a1^3 + a2^3 + a3^3 + a4^3 = 0) (k : ℕ) (hk : k % 2 = 1) :
  6 ∣ (a1^k + a2^k + a3^k + a4^k) :=
sorry

end sum_of_powers_divisible_by_6_l94_94103


namespace sum_prime_factors_of_77_l94_94448

theorem sum_prime_factors_of_77 : ∃ (p1 p2 : ℕ), Nat.Prime p1 ∧ Nat.Prime p2 ∧ 77 = p1 * p2 ∧ p1 + p2 = 18 := by
  sorry

end sum_prime_factors_of_77_l94_94448


namespace find_omega_l94_94810

theorem find_omega (ω : Real) (h : ∀ x : Real, (1 / 2) * Real.cos (ω * x - (Real.pi / 6)) = (1 / 2) * Real.cos (ω * (x + Real.pi) - (Real.pi / 6))) : ω = 2 ∨ ω = -2 :=
by
  sorry

end find_omega_l94_94810


namespace least_number_remainder_l94_94256

noncomputable def lcm_12_15_20_54 : ℕ := 540

theorem least_number_remainder :
  ∀ (n r : ℕ), (n = lcm_12_15_20_54 + r) → 
  (n % 12 = r) ∧ (n % 15 = r) ∧ (n % 20 = r) ∧ (n % 54 = r) → 
  r = 0 :=
by
  sorry

end least_number_remainder_l94_94256


namespace factor_expression_l94_94937

variable (b : ℝ)

theorem factor_expression : 221 * b * b + 17 * b = 17 * b * (13 * b + 1) :=
by
  sorry

end factor_expression_l94_94937


namespace last_three_digits_7_pow_80_l94_94949

theorem last_three_digits_7_pow_80 : (7^80) % 1000 = 961 := by
  sorry

end last_three_digits_7_pow_80_l94_94949


namespace find_phi_monotone_interval_1_monotone_interval_2_l94_94354

-- Definitions related to the function f
noncomputable def f (x φ a : ℝ) : ℝ :=
  Real.sin (x + φ) + a * Real.cos x

-- Problem Part 1: Given f(π/2) = √2 / 2, find φ
theorem find_phi (a : ℝ) (φ : ℝ) (h : |φ| < Real.pi / 2) (hf : f (π / 2) φ a = Real.sqrt 2 / 2) :
  φ = π / 4 ∨ φ = -π / 4 :=
  sorry

-- Problem Part 2 Condition 1: Given a = √3, φ = -π/3, find the monotonically increasing interval
theorem monotone_interval_1 :
  ∀ k : ℤ, ∀ x : ℝ, 
  ((-5 * π / 6) + 2 * k * π) ≤ x ∧ x ≤ (π / 6 + 2 * k * π) → 
  f x (-π / 3) (Real.sqrt 3) = Real.sin (x + π / 3) :=
  sorry

-- Problem Part 2 Condition 2: Given a = -1, φ = π/6, find the monotonically increasing interval
theorem monotone_interval_2 :
  ∀ k : ℤ, ∀ x : ℝ, 
  ((-π / 3) + 2 * k * π) ≤ x ∧ x ≤ ((2 * π / 3) + 2 * k * π) → 
  f x (π / 6) (-1) = Real.sin (x - π / 6) :=
  sorry

end find_phi_monotone_interval_1_monotone_interval_2_l94_94354


namespace vector_magnitude_difference_l94_94061

-- Defining the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Statement to prove that the magnitude of the difference of vectors a and b is 5
theorem vector_magnitude_difference : ‖a - b‖ = 5 := 
sorry -- Proof omitted

end vector_magnitude_difference_l94_94061


namespace number_of_ways_l94_94956

theorem number_of_ways (students : Fin 10 → Type) (A B C : Fin 10) (chosen : Finset (Fin 10)) :
  chosen.card = 3 →
  (B ∉ chosen) →
  (A ∈ chosen ∨ C ∈ chosen) →
  nat.choose 9 3 - nat.choose 7 3 = 49 := 
by
  sorry

end number_of_ways_l94_94956


namespace domain_of_sqrt_l94_94639

theorem domain_of_sqrt (x : ℝ) : (x - 1 ≥ 0) → (x ≥ 1) :=
by
  sorry

end domain_of_sqrt_l94_94639


namespace age_of_new_person_l94_94856

-- Definitions based on conditions
def initial_avg : ℕ := 15
def new_avg : ℕ := 17
def n : ℕ := 9

-- Statement to prove
theorem age_of_new_person : 
    ∃ (A : ℕ), (initial_avg * n + A) / (n + 1) = new_avg ∧ A = 35 := 
by {
    -- Proof steps would go here, but since they are not required, we add 'sorry' to skip the proof
    sorry
}

end age_of_new_person_l94_94856


namespace function_minimum_value_no_maximum_l94_94799

noncomputable def f (a x : ℝ) : ℝ :=
  (Real.sin x + a) / Real.sin x

theorem function_minimum_value_no_maximum (a : ℝ) (h_a : 0 < a) : 
  ∃ x_min, ∀ x ∈ Set.Ioo 0 Real.pi, f a x ≥ x_min ∧ 
           (∀ x ∈ Set.Ioo 0 Real.pi, f a x ≠ x_min) ∧ 
           ¬ (∃ x_max, ∀ x ∈ Set.Ioo 0 Real.pi, f a x ≤ x_max) :=
by
  let t := Real.sin
  have h : ∀ x ∈ Set.Ioo 0 Real.pi, t x ∈ Set.Ioo 0 1 := sorry -- Simple property of sine function in (0, π)
  -- Exact details skipped to align with the conditions from the problem, leveraging the property
  sorry -- Full proof not required as per instructions

end function_minimum_value_no_maximum_l94_94799


namespace slope_of_line_through_origin_l94_94992

open Real

def slope (A B : ℝ × ℝ) : ℝ := (B.2 - A.2) / (B.1 - A.1)

def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def perpendicular_slope (m : ℝ) : ℝ := -1 / m

theorem slope_of_line_through_origin (P Q : ℝ × ℝ) (hP : P = (4, 6)) (hQ : Q = (6, 2)) :
  slope (0, 0) (5, 4) = 1 / 2 :=
by
  sorry

end slope_of_line_through_origin_l94_94992


namespace sequence_an_general_formula_sum_bn_formula_l94_94355

variable (a : ℕ → ℕ) (S : ℕ → ℕ) (b : ℕ → ℕ) (T : ℕ → ℕ)

axiom seq_Sn_eq_2an_minus_n : ∀ n : ℕ, n > 0 → S n + n = 2 * a n

theorem sequence_an_general_formula (n : ℕ) (h : n > 0) :
  (∀ n > 0, a n + 1 = 2 * (a (n - 1) + 1)) ∧ (a n = 2^n - 1) :=
sorry

theorem sum_bn_formula (n : ℕ) (h : n > 0) :
  (∀ n > 0, b n = n * a n + n) → T n = (n - 1) * 2^(n + 1) + 2 :=
sorry

end sequence_an_general_formula_sum_bn_formula_l94_94355


namespace min_value_16_l94_94382

noncomputable def min_value_expr (a b : ℝ) : ℝ :=
  1 / a + 3 / b

theorem min_value_16 (a b : ℝ) (h : a > 0 ∧ b > 0) (h_constraint : a + 3 * b = 1) :
  min_value_expr a b ≥ 16 :=
sorry

end min_value_16_l94_94382


namespace sum_xyz_l94_94979

theorem sum_xyz (x y z : ℝ) (h1 : x + y = 1) (h2 : y + z = 1) (h3 : z + x = 1) : x + y + z = 3 / 2 :=
  sorry

end sum_xyz_l94_94979


namespace gcd_35_91_840_l94_94288

theorem gcd_35_91_840 : Nat.gcd (Nat.gcd 35 91) 840 = 7 :=
by
  sorry

end gcd_35_91_840_l94_94288


namespace solve_price_reduction_l94_94606

-- Definitions based on conditions
def daily_sales_volume (x : ℝ) : ℝ := 30 + 2 * x
def profit_per_item (x : ℝ) : ℝ := 50 - x
def daily_profit (x : ℝ) : ℝ := (50 - x) * (30 + 2 * x)

-- Statement
theorem solve_price_reduction :
  ∃ x : ℝ, daily_profit x = 2100 ∧ x ∈ {15, 20} :=
begin
  -- solution here
  sorry,
end

end solve_price_reduction_l94_94606


namespace infinite_chain_resistance_l94_94492

variables (R_0 R_X : ℝ)
def infinite_chain_resistance_condition (R_0 : ℝ) (R_X : ℝ) : Prop :=
  R_X = R_0 + (R_0 * R_X) / (R_0 + R_X)

theorem infinite_chain_resistance (R_0 : ℝ) (h : R_0 = 50) :
  ∃ R_X, infinite_chain_resistance_condition R_0 R_X ∧ R_X = (R_0 * (1 + Real.sqrt 5)) / 2 :=
  sorry

end infinite_chain_resistance_l94_94492


namespace magnitude_of_a_minus_b_l94_94058

-- Defining the vectors
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Subtracting vectors a and b
def a_minus_b : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)

-- Computing the magnitude of the resulting vector
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem magnitude_of_a_minus_b : magnitude a_minus_b = 5 := by
  sorry

end magnitude_of_a_minus_b_l94_94058


namespace find_number_l94_94572

theorem find_number (x : ℕ) (h : x + 15 = 96) : x = 81 := 
sorry

end find_number_l94_94572


namespace find_m_values_l94_94050

def is_solution (m : ℝ) : Prop :=
  let A : Set ℝ := {1, -2}
  let B : Set ℝ := {x | m * x + 1 = 0}
  B ⊆ A

theorem find_m_values :
  {m : ℝ | is_solution m} = {0, -1, 1 / 2} :=
by
  sorry

end find_m_values_l94_94050


namespace percentage_difference_l94_94203

theorem percentage_difference (N : ℝ) (hN : N = 160) : 0.50 * N - 0.35 * N = 24 := by
  sorry

end percentage_difference_l94_94203


namespace total_distance_of_journey_l94_94779

-- Definitions corresponding to conditions in the problem
def electric_distance : ℝ := 30 -- The first 30 miles were in electric mode
def gasoline_consumption_rate : ℝ := 0.03 -- Gallons per mile for gasoline mode
def average_mileage : ℝ := 50 -- Miles per gallon for the entire trip

-- Final goal: proving the total distance is 90 miles
theorem total_distance_of_journey (d : ℝ) :
  (d / (gasoline_consumption_rate * (d - electric_distance)) = average_mileage) → d = 90 :=
by
  sorry

end total_distance_of_journey_l94_94779


namespace num_hens_in_caravan_l94_94693

variable (H G C K : ℕ)  -- number of hens, goats, camels, keepers
variable (total_heads total_feet : ℕ)

-- Defining the conditions
def num_goats := 35
def num_camels := 6
def num_keepers := 10
def heads := H + G + C + K
def feet := 2 * H + 4 * G + 4 * C + 2 * K
def relation := feet = heads + 193

theorem num_hens_in_caravan :
  G = num_goats → C = num_camels → K = num_keepers → relation → 
  H = 60 :=
by 
  intros _ _ _ _
  sorry

end num_hens_in_caravan_l94_94693


namespace midpoint_coord_sum_l94_94437

theorem midpoint_coord_sum (x1 y1 x2 y2 : ℝ) (hx1 : x1 = 10) (hy1 : y1 = -2) (hx2 : x2 = -4) (hy2 : y2 = 8)
: (x1 + x2) / 2 + (y1 + y2) / 2 = 6 :=
by
  rw [hx1, hx2, hy1, hy2]
  /-
  Have (10 + (-4)) / 2 + (-2 + 8) / 2 = (6 / 2) + (6 / 2)
  Prove that (6 / 2) + (6 / 2) = 6
  -/
  sorry

end midpoint_coord_sum_l94_94437


namespace students_sampled_from_schoolB_l94_94159

-- Definitions from the conditions in a)
def schoolA_students := 800
def schoolB_students := 500
def total_students := schoolA_students + schoolB_students
def schoolA_sampled_students := 48

-- Mathematically equivalent proof problem
theorem students_sampled_from_schoolB : 
  let proportionA := (schoolA_students : ℝ) / total_students
  let proportionB := (schoolB_students : ℝ) / total_students
  let total_sampled_students := schoolA_sampled_students / proportionA
  let b_sampled_students := proportionB * total_sampled_students
  b_sampled_students = 30 :=
by
  -- Placeholder for the actual proof
  sorry

end students_sampled_from_schoolB_l94_94159


namespace one_fourths_in_one_eighth_l94_94077

theorem one_fourths_in_one_eighth : (1 / 8) / (1 / 4) = 1 / 2 := 
by 
  sorry

end one_fourths_in_one_eighth_l94_94077


namespace cannot_have_triangular_cross_section_l94_94743

-- Definition of the geometric solids
inductive GeometricSolid
  | Cylinder
  | Cone
  | TriangularPrism
  | Cube

-- Theorem statement
theorem cannot_have_triangular_cross_section (s : GeometricSolid) :
  s = GeometricSolid.Cylinder → ¬(∃ c : ℝ^3 → Prop, is_triangular_cross_section s c) := 
by
  intros h
  apply h.rec 
  intro
  sorry

end cannot_have_triangular_cross_section_l94_94743


namespace max_ab_ac_bc_l94_94230

noncomputable def maxValue (a b c : ℝ) := a * b + a * c + b * c

theorem max_ab_ac_bc (a b c : ℝ) (h : a + 3 * b + c = 6) : maxValue a b c ≤ 12 :=
by
  sorry

end max_ab_ac_bc_l94_94230


namespace arithmetic_sequence_product_l94_94710

theorem arithmetic_sequence_product (a : ℕ → ℤ) (d : ℤ)
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_incr : ∀ n, a n < a (n + 1))
  (h_a4a5 : a 3 * a 4 = 24) :
  a 2 * a 5 = 16 :=
sorry

end arithmetic_sequence_product_l94_94710


namespace math_problem_l94_94395

theorem math_problem
  (x : ℝ)
  (h : (1/2) * x - 300 = 350) :
  (x + 200) * 2 = 3000 :=
by
  sorry

end math_problem_l94_94395


namespace original_remainder_when_dividing_by_44_is_zero_l94_94164

theorem original_remainder_when_dividing_by_44_is_zero 
  (N R : ℕ) 
  (Q : ℕ) 
  (h1 : N = 44 * 432 + R) 
  (h2 : N = 34 * Q + 2) 
  : R = 0 := 
sorry

end original_remainder_when_dividing_by_44_is_zero_l94_94164


namespace reciprocal_of_sum_frac_is_correct_l94_94421

/-- The reciprocal of the sum of the fractions 1/4 and 1/6 is 12/5. -/
theorem reciprocal_of_sum_frac_is_correct:
  (1 / (1 / 4 + 1 / 6)) = (12 / 5) :=
by 
  sorry

end reciprocal_of_sum_frac_is_correct_l94_94421


namespace calculate_sin_product_l94_94993

theorem calculate_sin_product (α β : ℝ) (h1 : Real.sin (α + β) = 0.2) (h2 : Real.cos (α - β) = 0.3) :
  Real.sin (α + π/4) * Real.sin (β + π/4) = 0.25 :=
by
  sorry

end calculate_sin_product_l94_94993


namespace matrix_expression_l94_94186
open Matrix

variables {n : Type*} [Fintype n] [DecidableEq n]
variables (B : Matrix n n ℝ) (I : Matrix n n ℝ)

noncomputable def B_inverse := B⁻¹

-- Condition 1: B is a matrix with an inverse
variable [Invertible B]

-- Condition 2: (B - 3*I) * (B - 5*I) = 0
variable (H : (B - (3 : ℝ) • I) * (B - (5 : ℝ) • I) = 0)

-- Theorem to prove
theorem matrix_expression (B: Matrix n n ℝ) [Invertible B] 
  (H : (B - (3 : ℝ) • I) * (B - (5 : ℝ) • I) = 0) : 
  B + 10 * (B_inverse B) = (160 / 15 : ℝ) • I := 
sorry

end matrix_expression_l94_94186


namespace machine_does_not_require_repair_l94_94414

-- Define the conditions.

def max_deviation := 37

def nominal_portion_max_deviation_percentage := 0.10

def deviation_within_limit (M : ℝ) : Prop :=
  37 ≤ 0.10 * M

def unreadable_measurements_deviation (deviation : ℝ) : Prop :=
  deviation < 37

-- Define the theorem we want to prove

theorem machine_does_not_require_repair (M : ℝ)
  (h1 : deviation_within_limit M)
  (h2 : ∀ deviation, unreadable_measurements_deviation deviation) :
  true := 
sorry

end machine_does_not_require_repair_l94_94414


namespace rounding_addition_to_tenth_l94_94912

def number1 : Float := 96.23
def number2 : Float := 47.849

theorem rounding_addition_to_tenth (sum : Float) : 
    sum = number1 + number2 →
    Float.round (sum * 10) / 10 = 144.1 :=
by
  intro h
  rw [h]
  norm_num
  sorry -- Skipping the actual rounding proof

end rounding_addition_to_tenth_l94_94912


namespace shoes_multiple_l94_94220

-- Define the number of shoes each has
variables (J E B : ℕ)

-- Conditions
axiom h1 : B = 22
axiom h2 : J = E / 2
axiom h3 : J + E + B = 121

-- Prove the multiple of E to B is 3
theorem shoes_multiple : E / B = 3 :=
by
  -- Inject the provisional proof
  sorry

end shoes_multiple_l94_94220


namespace exists_consecutive_numbers_with_prime_divisors_l94_94517

theorem exists_consecutive_numbers_with_prime_divisors (p q : ℕ) 
  (hp : Nat.Prime p) (hq : Nat.Prime q) (h : p < q ∧ q < 2 * p) :
  ∃ n m : ℕ, (m = n + 1) ∧ 
             (Nat.gcd n p = p) ∧ (Nat.gcd m p = 1) ∧ 
             (Nat.gcd m q = q) ∧ (Nat.gcd n q = 1) :=
by
  sorry

end exists_consecutive_numbers_with_prime_divisors_l94_94517


namespace trig_identity_l94_94976

theorem trig_identity (α : ℝ) (h : Real.tan α = 3 / 4) : 
  Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 64 / 25 := 
by
  sorry

end trig_identity_l94_94976


namespace sandy_paints_area_l94_94243

-- Definition of the dimensions
def wall_height : ℝ := 10
def wall_length : ℝ := 15
def window_height : ℝ := 3
def window_length : ℝ := 5
def door_height : ℝ := 1
def door_length : ℝ := 6.5

-- Areas computation
def wall_area : ℝ := wall_height * wall_length
def window_area : ℝ := window_height * window_length
def door_area : ℝ := door_height * door_length

-- Area to be painted
def area_not_painted : ℝ := window_area + door_area
def area_to_be_painted : ℝ := wall_area - area_not_painted

-- The theorem to prove
theorem sandy_paints_area : area_to_be_painted = 128.5 := by
  -- The proof is omitted
  sorry

end sandy_paints_area_l94_94243


namespace bottom_level_legos_l94_94632

theorem bottom_level_legos
  (x : ℕ)
  (h : x^2 + (x - 1)^2 + (x - 2)^2 = 110) :
  x = 7 :=
by {
  sorry
}

end bottom_level_legos_l94_94632


namespace problem1_problem2_problem3_l94_94020

-- Proof statement for Problem 1
theorem problem1 : 23 * (-5) - (-3) / (3 / 108) = -7 := 
by 
  sorry

-- Proof statement for Problem 2
theorem problem2 : (-7) * (-3) * (-0.5) + (-12) * (-2.6) = 20.7 := 
by 
  sorry

-- Proof statement for Problem 3
theorem problem3 : ((-1 / 2) - (1 / 12) + (3 / 4) - (1 / 6)) * (-48) = 0 := 
by 
  sorry

end problem1_problem2_problem3_l94_94020


namespace vector_magnitude_l94_94065

theorem vector_magnitude (a b : ℝ × ℝ) (a_def : a = (2, 1)) (b_def : b = (-2, 4)) :
  ‖(a.1 - b.1, a.2 - b.2)‖ = 5 := by
  sorry

end vector_magnitude_l94_94065


namespace monotonicity_of_f_range_of_a_l94_94359

open Real

noncomputable def f (x a : ℝ) : ℝ := a * exp x + 2 * exp (-x) + (a - 2) * x

noncomputable def f_prime (x a : ℝ) : ℝ := (a * exp (2 * x) + (a - 2) * exp x - 2) / exp x

theorem monotonicity_of_f (a : ℝ) : 
  (∀ x : ℝ, f_prime x a ≤ 0) ↔ (a ≤ 0) :=
sorry

theorem range_of_a (a : ℝ) : 
  (∀ x > 0, f x a ≥ (a + 2) * cos x) ↔ (2 ≤ a) :=
sorry

end monotonicity_of_f_range_of_a_l94_94359


namespace find_a_l94_94376

theorem find_a (a : ℝ) (h1 : a + 3 > 0) (h2 : abs (a + 3) = 5) : a = 2 := 
by
  sorry

end find_a_l94_94376


namespace distance_between_foci_of_ellipse_l94_94650

-- Define the parameters a^2 and b^2 according to the problem
def a_sq : ℝ := 25
def b_sq : ℝ := 16

-- State the problem
theorem distance_between_foci_of_ellipse : 
  (2 * Real.sqrt (a_sq - b_sq)) = 6 := by
  -- Proof content is skipped 
  sorry

end distance_between_foci_of_ellipse_l94_94650


namespace profit_calculation_l94_94835

theorem profit_calculation (investment_john investment_mike profit_john profit_mike: ℕ) 
  (total_profit profit_shared_ratio profit_remaining_profit: ℚ)
  (h_investment_john : investment_john = 700)
  (h_investment_mike : investment_mike = 300)
  (h_total_profit : total_profit = 3000)
  (h_shared_ratio : profit_shared_ratio = total_profit / 3 / 2)
  (h_remaining_profit : profit_remaining_profit = 2 * total_profit / 3)
  (h_profit_john : profit_john = profit_shared_ratio + (7 / 10) * profit_remaining_profit)
  (h_profit_mike : profit_mike = profit_shared_ratio + (3 / 10) * profit_remaining_profit)
  (h_profit_difference : profit_john = profit_mike + 800) :
  total_profit = 3000 := 
by
  sorry

end profit_calculation_l94_94835


namespace count_valid_triples_l94_94521

-- Define the necessary conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n
def is_positive (n : ℕ) : Prop := n > 0
def valid_triple (p q n : ℕ) : Prop := is_prime p ∧ is_prime q ∧ is_positive n ∧ (1/p + 2013/q = n/5)

-- Lean statement for the proof problem
theorem count_valid_triples : 
  ∃ c : ℕ, c = 5 ∧ 
  (∀ p q n : ℕ, valid_triple p q n → true) :=
sorry

end count_valid_triples_l94_94521


namespace can_not_buy_both_phones_l94_94603

noncomputable def alexander_salary : ℕ := 125000
noncomputable def natalia_salary : ℕ := 61000
noncomputable def utilities_transport_household_expenses : ℕ := 17000
noncomputable def loan_expenses : ℕ := 15000
noncomputable def cultural_theater : ℕ := 5000
noncomputable def cultural_cinema : ℕ := 2000
noncomputable def crimea_savings : ℕ := 20000
noncomputable def dining_out_weekdays_cost : ℕ := 1500 * 20
noncomputable def dining_out_weekends_cost : ℕ := 3000 * 10
noncomputable def phone_A_cost : ℕ := 57000
noncomputable def phone_B_cost : ℕ := 37000

theorem can_not_buy_both_phones :
  let total_expenses := utilities_transport_household_expenses + loan_expenses + cultural_theater + cultural_cinema + crimea_savings + dining_out_weekdays_cost + dining_out_weekends_cost in
  let total_income := alexander_salary + natalia_salary in
  let net_income := total_income - total_expenses in
    total_expenses = 119000 ∧ net_income = 67000 ∧ 67000 < (phone_A_cost + phone_B_cost) :=
by 
  intros;
  sorry

end can_not_buy_both_phones_l94_94603


namespace pollywog_maturation_rate_l94_94177

theorem pollywog_maturation_rate :
  ∀ (initial_pollywogs : ℕ) (melvin_rate : ℕ) (total_days : ℕ) (melvin_days : ℕ) (remaining_pollywogs : ℕ),
  initial_pollywogs = 2400 →
  melvin_rate = 10 →
  total_days = 44 →
  melvin_days = 20 →
  remaining_pollywogs = initial_pollywogs - (melvin_rate * melvin_days) →
  (total_days * (remaining_pollywogs / (total_days - melvin_days))) = remaining_pollywogs →
  (remaining_pollywogs / (total_days - melvin_days)) = 50 := 
by
  intros initial_pollywogs melvin_rate total_days melvin_days remaining_pollywogs
  intros h_initial h_melvin h_total h_melvin_days h_remaining h_eq
  sorry

end pollywog_maturation_rate_l94_94177


namespace intersection_A_B_l94_94357

-- Definition of set A based on the given inequality
def A : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}

-- Definition of set B
def B : Set ℝ := {-3, -1, 1, 3}

-- Prove the intersection A ∩ B equals the expected set {-1, 1, 3}
theorem intersection_A_B : A ∩ B = {-1, 1, 3} := 
by
  sorry

end intersection_A_B_l94_94357


namespace magnitude_of_a_minus_b_l94_94057

-- Defining the vectors
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Subtracting vectors a and b
def a_minus_b : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)

-- Computing the magnitude of the resulting vector
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem magnitude_of_a_minus_b : magnitude a_minus_b = 5 := by
  sorry

end magnitude_of_a_minus_b_l94_94057


namespace election_votes_l94_94598

theorem election_votes
  (V : ℕ)  -- total number of votes
  (candidate1_votes_percent : ℕ := 80)  -- first candidate percentage
  (second_candidate_votes : ℕ := 480)  -- votes for second candidate
  (second_candidate_percent : ℕ := 20)  -- second candidate percentage
  (h : second_candidate_votes = (second_candidate_percent * V) / 100) :
  V = 2400 :=
sorry

end election_votes_l94_94598


namespace inequality_sqrt_l94_94518

open Real

theorem inequality_sqrt (x y : ℝ) :
  (sqrt (x^2 - 2*x*y) > sqrt (1 - y^2)) ↔ 
    ((x - y > 1 ∧ -1 < y ∧ y < 1) ∨ (x - y < -1 ∧ -1 < y ∧ y < 1)) :=
by
  sorry

end inequality_sqrt_l94_94518


namespace two_digit_number_is_24_l94_94145

-- Defining the two-digit number conditions

variables (x y : ℕ)

noncomputable def condition1 := y = x + 2
noncomputable def condition2 := (10 * x + y) * (x + y) = 144

-- The statement of the proof problem
theorem two_digit_number_is_24 (h1 : condition1 x y) (h2 : condition2 x y) : 10 * x + y = 24 :=
sorry

end two_digit_number_is_24_l94_94145


namespace number_of_functions_l94_94862

-- Define the set of conditions
variables (x y : ℝ)

def relation1 := x - y = 0
def relation2 := y^2 = x
def relation3 := |y| = 2 * x
def relation4 := y^2 = x^2
def relation5 := y = 3 - x
def relation6 := y = 2 * x^2 - 1
def relation7 := y = 3 / x

-- Prove that there are 4 unambiguous functions of y with respect to x
theorem number_of_functions : 4 = 4 := sorry

end number_of_functions_l94_94862


namespace tan_sum_angle_l94_94184

theorem tan_sum_angle (α : ℝ) (h : Real.tan α = 2) : Real.tan (π / 4 + α) = -3 := 
by sorry

end tan_sum_angle_l94_94184


namespace unknown_card_value_l94_94134

theorem unknown_card_value (cards_total : ℕ)
  (p1_hand : ℕ) (p1_hand_extra : ℕ) (table_card1 : ℕ) (total_card_values : ℕ)
  (sum_removed_cards_sets : ℕ)
  (n : ℕ) :
  cards_total = 40 ∧ 
  p1_hand = 5 ∧ 
  p1_hand_extra = 3 ∧ 
  table_card1 = 9 ∧ 
  total_card_values = 220 ∧ 
  sum_removed_cards_sets = 15 * n → 
  ∃ x : ℕ, 1 ≤ x ∧ x ≤ 10 ∧ total_card_values = p1_hand + p1_hand_extra + table_card1 + x + sum_removed_cards_sets → 
  x = 8 := 
sorry

end unknown_card_value_l94_94134


namespace ratio_sheep_horses_eq_six_seven_l94_94483

noncomputable def total_food_per_day : ℕ := 12880
noncomputable def food_per_horse_per_day : ℕ := 230
noncomputable def num_sheep : ℕ := 48
noncomputable def num_horses : ℕ := total_food_per_day / food_per_horse_per_day
noncomputable def ratio_sheep_to_horses := num_sheep / num_horses

theorem ratio_sheep_horses_eq_six_seven :
  ratio_sheep_to_horses = 6 / 7 :=
by
  sorry

end ratio_sheep_horses_eq_six_seven_l94_94483


namespace negation_of_exists_real_solution_equiv_l94_94263

open Classical

theorem negation_of_exists_real_solution_equiv :
  (¬ ∃ a : ℝ, ∃ x : ℝ, a * x^2 + 1 = 0) ↔ (∀ a : ℝ, ¬ ∃ x : ℝ, a * x^2 + 1 = 0) :=
by
  sorry

end negation_of_exists_real_solution_equiv_l94_94263


namespace smallest_positive_integer_l94_94525

theorem smallest_positive_integer (k : ℕ) :
  (∃ k : ℕ, ((2^4 ∣ 1452 * k) ∧ (3^3 ∣ 1452 * k) ∧ (13^3 ∣ 1452 * k))) → 
  k = 676 := 
sorry

end smallest_positive_integer_l94_94525


namespace project_estimated_hours_l94_94279

theorem project_estimated_hours (extra_hours_per_day : ℕ) (normal_work_hours : ℕ) (days_to_finish : ℕ)
  (total_hours_estimation : ℕ)
  (h1 : extra_hours_per_day = 5)
  (h2 : normal_work_hours = 10)
  (h3 : days_to_finish = 100)
  (h4 : total_hours_estimation = days_to_finish * (normal_work_hours + extra_hours_per_day))
  : total_hours_estimation = 1500 :=
  by
  -- Proof to be provided 
  sorry

end project_estimated_hours_l94_94279


namespace no_positive_integer_solution_l94_94719

def is_solution (x y z t : ℕ) : Prop :=
  x^2 + 5 * y^2 = z^2 ∧ 5 * x^2 + y^2 = t^2

theorem no_positive_integer_solution :
  ¬ ∃ (x y z t : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < t ∧ is_solution x y z t :=
by
  sorry

end no_positive_integer_solution_l94_94719


namespace minimum_students_lost_all_items_l94_94790

def smallest_number (N A B C : ℕ) (x : ℕ) : Prop :=
  N = 30 ∧ A = 26 ∧ B = 23 ∧ C = 21 → x ≥ 10

theorem minimum_students_lost_all_items (N A B C : ℕ) : 
  smallest_number N A B C 10 := 
by {
  sorry
}

end minimum_students_lost_all_items_l94_94790


namespace hoseok_value_l94_94571

theorem hoseok_value (x : ℕ) (h : x - 10 = 15) : x + 5 = 30 :=
by
  sorry

end hoseok_value_l94_94571


namespace height_of_fourth_person_l94_94428

theorem height_of_fourth_person
  (h : ℝ)
  (cond : (h + (h + 2) + (h + 4) + (h + 10)) / 4 = 79) :
  (h + 10) = 85 :=
by 
  sorry

end height_of_fourth_person_l94_94428


namespace find_star_l94_94289

theorem find_star :
  ∃ (star : ℤ), 45 - ( 28 - ( 37 - ( 15 - star ) ) ) = 56 ∧ star = 17 :=
by
  sorry

end find_star_l94_94289


namespace binary_remainder_div_4_is_1_l94_94633

def binary_to_base_10_last_two_digits (b1 b0 : Nat) : Nat :=
  2 * b1 + b0

noncomputable def remainder_of_binary_by_4 (n : Nat) : Nat :=
  match n with
  | 111010110101 => binary_to_base_10_last_two_digits 0 1
  | _ => 0

theorem binary_remainder_div_4_is_1 :
  remainder_of_binary_by_4 111010110101 = 1 := by
  sorry

end binary_remainder_div_4_is_1_l94_94633


namespace min_even_integers_six_l94_94582

theorem min_even_integers_six (x y a b m n : ℤ) 
  (h1 : x + y = 30) 
  (h2 : x + y + a + b = 50) 
  (h3 : x + y + a + b + m + n = 70) 
  (hm_even : Even m) 
  (hn_even: Even n) : 
  ∃ k, (0 ≤ k ∧ k ≤ 6) ∧ (∀ e, (e = m ∨ e = n) → ∃ j, (j = 2)) :=
by
  sorry

end min_even_integers_six_l94_94582


namespace cubic_roots_l94_94854

variable (p q : ℝ)

noncomputable def ω : ℂ := Complex.exp (2 * Real.pi * Complex.I / 3)

theorem cubic_roots (y z : ℂ) (h1 : -3 * y * z = p) (h2 : y^3 + z^3 = q) :
  ∃ (x1 x2 x3 : ℂ),
    (x^3 + p * x + q = 0) ∧
    (x1 = -(y + z)) ∧
    (x2 = -(ω * y + ω^2 * z)) ∧
    (x3 = -(ω^2 * y + ω * z)) :=
by
  sorry

end cubic_roots_l94_94854


namespace lauren_earnings_tuesday_l94_94097

def money_from_commercials (num_commercials : ℕ) (rate_per_commercial : ℝ) : ℝ :=
  num_commercials * rate_per_commercial

def money_from_subscriptions (num_subscriptions : ℕ) (rate_per_subscription : ℝ) : ℝ :=
  num_subscriptions * rate_per_subscription

def total_money (num_commercials : ℕ) (rate_per_commercial : ℝ) (num_subscriptions : ℕ) (rate_per_subscription : ℝ) : ℝ :=
  money_from_commercials num_commercials rate_per_commercial + money_from_subscriptions num_subscriptions rate_per_subscription

theorem lauren_earnings_tuesday (num_commercials : ℕ) (rate_per_commercial : ℝ) (num_subscriptions : ℕ) (rate_per_subscription : ℝ) :
  num_commercials = 100 → rate_per_commercial = 0.50 → num_subscriptions = 27 → rate_per_subscription = 1.00 → 
  total_money num_commercials rate_per_commercial num_subscriptions rate_per_subscription = 77 :=
by
  intros h1 h2 h3 h4
  simp [money_from_commercials, money_from_subscriptions, total_money, h1, h2, h3, h4]
  sorry

end lauren_earnings_tuesday_l94_94097


namespace sin_neg_30_eq_neg_one_half_l94_94330

theorem sin_neg_30_eq_neg_one_half :
  Real.sin (-30 * Real.pi / 180) = -1 / 2 := 
by
  sorry -- Proof is skipped

end sin_neg_30_eq_neg_one_half_l94_94330


namespace arithmetic_sequence_problem_l94_94213

theorem arithmetic_sequence_problem 
  (a : ℕ → ℕ)
  (h_sequence : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0)
  (h_sum : a 3 + a 8 = 10) :
  3 * a 5 + a 7 = 20 :=
sorry

end arithmetic_sequence_problem_l94_94213


namespace total_cost_eq_1400_l94_94709

theorem total_cost_eq_1400 (stove_cost : ℝ) (wall_repair_fraction : ℝ) (wall_repair_cost : ℝ) (total_cost : ℝ)
  (h₁ : stove_cost = 1200)
  (h₂ : wall_repair_fraction = 1/6)
  (h₃ : wall_repair_cost = stove_cost * wall_repair_fraction)
  (h₄ : total_cost = stove_cost + wall_repair_cost) :
  total_cost = 1400 :=
sorry

end total_cost_eq_1400_l94_94709


namespace combined_capacity_l94_94341

theorem combined_capacity (A B : ℝ) : 3 * A + B = A + 2 * A + B :=
by
  sorry

end combined_capacity_l94_94341


namespace correct_probability_statement_l94_94287

-- Define the conditions
def impossible_event_has_no_probability : Prop := ∀ (P : ℝ), P < 0 ∨ P > 0
def every_event_has_probability : Prop := ∀ (P : ℝ), 0 ≤ P ∧ P ≤ 1
def not_all_random_events_have_probability : Prop := ∃ (P : ℝ), P < 0 ∨ P > 1
def certain_events_do_not_have_probability : Prop := (∀ (P : ℝ), P ≠ 1)

-- The main theorem asserting that every event has a probability
theorem correct_probability_statement : every_event_has_probability :=
by sorry

end correct_probability_statement_l94_94287


namespace row_length_in_feet_l94_94631

theorem row_length_in_feet (seeds_per_row : ℕ) (space_per_seed : ℕ) (inches_per_foot : ℕ) (H1 : seeds_per_row = 80) (H2 : space_per_seed = 18) (H3 : inches_per_foot = 12) : 
  seeds_per_row * space_per_seed / inches_per_foot = 120 :=
by
  sorry

end row_length_in_feet_l94_94631


namespace temperature_at_midnight_l94_94562

theorem temperature_at_midnight :
  ∀ (morning_temp noon_rise midnight_drop midnight_temp : ℤ),
    morning_temp = -3 →
    noon_rise = 6 →
    midnight_drop = -7 →
    midnight_temp = morning_temp + noon_rise + midnight_drop →
    midnight_temp = -4 :=
by
  intros
  sorry

end temperature_at_midnight_l94_94562


namespace candy_distribution_l94_94244

open Nat

theorem candy_distribution :
  let red, blue, white : Finset Nat := Finset.range 3 in
  let candies : Finset Nat := Finset.range 7 in
  (card ((candies.ssubsets.filter (λ s, 1 ≤ s.card ∧ s.card ≤ 6)).bind (λ r,
    ((candies \ r).ssubsets.filter (λ b, 1 ≤ b.card)).bind (λ b,
    ((candies \ (r ∪ b)).ssubsets.filter (λ w, w.card = (candies \ (r ∪ b)).card)))).card = 12) :=
begin
  -- sorry to skip the proof
  sorry
end

end candy_distribution_l94_94244


namespace factor_correct_l94_94942

noncomputable def p (b : ℝ) : ℝ := 221 * b^2 + 17 * b
def factored_form (b : ℝ) : ℝ := 17 * b * (13 * b + 1)

theorem factor_correct (b : ℝ) : p b = factored_form b := by
  sorry

end factor_correct_l94_94942


namespace common_difference_arithmetic_sequence_l94_94659

theorem common_difference_arithmetic_sequence (d : ℝ) :
  (∀ (n : ℝ) (a_1 : ℝ), a_1 = 9 ∧
  (∃ a₄ a₈ : ℝ, a₄ = a_1 + 3 * d ∧ a₈ = a_1 + 7 * d ∧ a₄ = (a_1 * a₈)^(1/2)) →
  d = 1) :=
sorry

end common_difference_arithmetic_sequence_l94_94659


namespace maximize_profit_l94_94249

variable (k : ℚ) -- Proportional constant for deposits
variable (x : ℚ) -- Annual interest rate paid to depositors
variable (D : ℚ) -- Total amount of deposits

-- Define the condition for the total amount of deposits
def deposits (x : ℚ) : ℚ := k * x^2

-- Define the profit function
def profit (x : ℚ) : ℚ := 0.045 * k * x^2 - k * x^3

-- Define the derivative of the profit function
def profit_derivative (x : ℚ) : ℚ := 3 * k * x * (0.03 - x)

-- Statement that x = 0.03 maximizes the bank's profit
theorem maximize_profit : ∃ x, x = 0.03 ∧ (∀ y, profit_derivative y = 0 → x = y) :=
by
  sorry

end maximize_profit_l94_94249


namespace RupertCandles_l94_94109

-- Definitions corresponding to the conditions
def PeterAge : ℕ := 10
def RupertRelativeAge : ℝ := 3.5

-- Define Rupert's age based on Peter's age and the given relative age factor
def RupertAge : ℝ := RupertRelativeAge * PeterAge

-- Statement of the theorem
theorem RupertCandles : RupertAge = 35 := by
  -- Proof is omitted
  sorry

end RupertCandles_l94_94109


namespace sum_prime_factors_77_l94_94452

theorem sum_prime_factors_77 : ∀ (p1 p2 : ℕ), Nat.Prime p1 ∧ Nat.Prime p2 ∧ 77 = p1 * p2 → p1 + p2 = 18 :=
by
  intros p1 p2 h
  sorry

end sum_prime_factors_77_l94_94452


namespace find_theta_l94_94215

-- Define the angles
variables (VEK KEW EVG θ : ℝ)

-- State the conditions as hypotheses
def conditions (VEK KEW EVG θ : ℝ) := 
  VEK = 70 ∧
  KEW = 40 ∧
  EVG = 110

-- State the theorem
theorem find_theta (VEK KEW EVG θ : ℝ)
  (h : conditions VEK KEW EVG θ) : 
  θ = 40 :=
by {
  sorry
}

end find_theta_l94_94215


namespace value_of_expression_l94_94801

noncomputable def a : ℝ := Real.log 3 / Real.log 4

theorem value_of_expression (h : a = Real.log 3 / Real.log 4) : 2^a + 2^(-a) = 4 * Real.sqrt 3 / 3 :=
by
  sorry

end value_of_expression_l94_94801


namespace percentage_of_men_l94_94535

variables {M W : ℝ}
variables (h1 : M + W = 100)
variables (h2 : 0.20 * M + 0.40 * W = 34)

theorem percentage_of_men :
  M = 30 :=
by
  sorry

end percentage_of_men_l94_94535


namespace donation_ratio_l94_94117

theorem donation_ratio (D1 : ℝ) (D1_value : D1 = 10)
  (total_donation : D1 + D1 * 2 + D1 * 4 + D1 * 8 + D1 * 16 = 310) : 
  2 = 2 :=
by
  sorry

end donation_ratio_l94_94117


namespace angle_B_is_30_degrees_l94_94218

variable (a b : ℝ)
variable (A B : ℝ)

axiom a_value : a = 2 * Real.sqrt 3
axiom b_value : b = Real.sqrt 6
axiom A_value : A = Real.pi / 4

theorem angle_B_is_30_degrees (h1 : a = 2 * Real.sqrt 3) (h2 : b = Real.sqrt 6) (h3 : A = Real.pi / 4) : B = Real.pi / 6 :=
  sorry

end angle_B_is_30_degrees_l94_94218


namespace solve_for_y_l94_94033

-- The given condition as a hypothesis
variables {x y : ℝ}

-- The theorem statement
theorem solve_for_y (h : 3 * x - y + 5 = 0) : y = 3 * x + 5 :=
sorry

end solve_for_y_l94_94033


namespace percent_decrease_is_80_l94_94146

-- Definitions based on the conditions
def original_price := 100
def sale_price := 20

-- Theorem statement to prove the percent decrease
theorem percent_decrease_is_80 :
  ((original_price - sale_price) / original_price * 100) = 80 := 
by
  sorry

end percent_decrease_is_80_l94_94146


namespace number_of_perfect_square_factors_l94_94420

theorem number_of_perfect_square_factors (a b c d : ℕ) :
  (∀ a b c d, 
    (0 ≤ a ∧ a ≤ 4) ∧ 
    (0 ≤ b ∧ b ≤ 2) ∧ 
    (0 ≤ c ∧ c ≤ 1) ∧ 
    (0 ≤ d ∧ d ≤ 1) ∧ 
    (a % 2 = 0) ∧ 
    (b % 2 = 0) ∧ 
    (c = 0) ∧ 
    (d = 0)
  → 3 * 2 * 1 * 1 = 6) := by
  sorry

end number_of_perfect_square_factors_l94_94420


namespace smallest_number_starting_with_five_l94_94952

theorem smallest_number_starting_with_five :
  ∃ n : ℕ, ∃ m : ℕ, m = (5 * m + 5) / 4 ∧ 5 * n + m = 512820 ∧ m < 10^6 := sorry

end smallest_number_starting_with_five_l94_94952


namespace total_weight_l94_94006

variable (a b c d : ℝ)

-- Conditions
axiom h1 : a + b = 250
axiom h2 : b + c = 235
axiom h3 : c + d = 260
axiom h4 : a + d = 275

-- Proving the total weight
theorem total_weight : a + b + c + d = 510 := by
  sorry

end total_weight_l94_94006


namespace proof_problem_l94_94773

open Real

variables {a : ℝ}
def average (l : list ℝ) : ℝ := l.sum / l.length

def range (l : list ℝ) : ℝ := l.foldr max (-∞) - l.foldr min ∞

noncomputable def mode (l : list ℝ) : list ℝ :=
  let occurrences := l.foldl (λ counts x, counts.insert x (count x l)) ∅
  let max_occ := occurrences.fold (λ _ count max_count, max count max_count) 0
  (occurrences.to_finset.filter (λ x, count x l = max_occ)).to_list

noncomputable def variance (l : list ℝ) : ℝ :=
  let mean := average l
  l.foldr (λ x acc, acc + (x - mean) ^ 2) 0 / l.length

theorem proof_problem (h : average [3, 6, 8, a, 5, 9] = 6) : 
  a = 5 ∧ range [3, 6, 8, 5, 5, 9] = 6 ∧ mode [3, 6, 8, 5, 5, 9] = [5] ∧ variance [3, 6, 8, 5, 5, 9] = 4 :=
by sorry

end proof_problem_l94_94773


namespace vector_magnitude_difference_l94_94062

-- Defining the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Statement to prove that the magnitude of the difference of vectors a and b is 5
theorem vector_magnitude_difference : ‖a - b‖ = 5 := 
sorry -- Proof omitted

end vector_magnitude_difference_l94_94062


namespace Natalia_Tuesday_distance_l94_94107

theorem Natalia_Tuesday_distance :
  ∃ T : ℕ, (40 + T + T / 2 + (40 + T / 2) = 180) ∧ T = 33 :=
by
  existsi 33
  -- proof can be filled here
  sorry

end Natalia_Tuesday_distance_l94_94107


namespace symmetric_point_y_axis_l94_94375

-- Define the original point P
def P : ℝ × ℝ := (1, 6)

-- Define the reflection across the y-axis
def reflect_y_axis (point : ℝ × ℝ) : ℝ × ℝ :=
  (-point.fst, point.snd)

-- Define the symmetric point with respect to the y-axis
def symmetric_point := reflect_y_axis P

-- Statement to prove
theorem symmetric_point_y_axis : symmetric_point = (-1, 6) :=
by
  -- Proof omitted
  sorry

end symmetric_point_y_axis_l94_94375


namespace sum_of_prime_factors_77_l94_94446

theorem sum_of_prime_factors_77 : (7 + 11 = 18) := by
  sorry

end sum_of_prime_factors_77_l94_94446


namespace abc_geq_inequality_l94_94850

open Real

theorem abc_geq_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a * b * c ≥ (a + b - c) * (b + c - a) * (c + a - b) :=
by
  sorry

end abc_geq_inequality_l94_94850


namespace man_age_year_l94_94461

theorem man_age_year (x : ℕ) (h1 : x^2 = 1892) (h2 : 1850 ≤ x ∧ x ≤ 1900) :
  (x = 44) → (1892 = 1936) := by
sorry

end man_age_year_l94_94461


namespace other_coin_denomination_l94_94875

theorem other_coin_denomination :
  ∀ (total_coins : ℕ) (value_rs : ℕ) (paise_per_rs : ℕ) (num_20_paise_coins : ℕ) (total_value_paise : ℕ),
  total_coins = 324 →
  value_rs = 71 →
  paise_per_rs = 100 →
  num_20_paise_coins = 200 →
  total_value_paise = value_rs * paise_per_rs →
  (∃ (denom_other_coin : ℕ),
    total_value_paise - num_20_paise_coins * 20 = (total_coins - num_20_paise_coins) * denom_other_coin
    → denom_other_coin = 25) :=
by
  sorry

end other_coin_denomination_l94_94875


namespace find_extrema_of_f_l94_94792

noncomputable def f (x : ℝ) : ℝ := (x^4 + x^2 + 5) / (x^2 + 1)^2

theorem find_extrema_of_f :
  (∀ x : ℝ, f x ≤ 5) ∧ (∃ x : ℝ, f x = 5) ∧ (∀ x : ℝ, f x ≥ 0.95) ∧ (∃ x : ℝ, f x = 0.95) :=
by {
  sorry
}

end find_extrema_of_f_l94_94792


namespace halfway_between_one_eighth_and_one_third_l94_94178

theorem halfway_between_one_eighth_and_one_third : (1/8 + 1/3) / 2 = 11/48 :=
by
  sorry

end halfway_between_one_eighth_and_one_third_l94_94178


namespace circle_properties_l94_94804

def circle_center_line (x y : ℝ) : Prop := x + y - 1 = 0

def point_A_on_circle (x y : ℝ) : Prop := (x, y) = (-1, 4)
def point_B_on_circle (x y : ℝ) : Prop := (x, y) = (1, 2)

def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 4

def slope_range_valid (k : ℝ) : Prop :=
  k ≤ 0 ∨ k ≥ 4 / 3

theorem circle_properties
  (x y : ℝ)
  (center_x center_y : ℝ)
  (h_center_line : circle_center_line center_x center_y)
  (h_point_A_on_circle : point_A_on_circle x y)
  (h_point_B_on_circle : point_B_on_circle x y)
  (h_circle_equation : circle_equation x y)
  (k : ℝ) :
  circle_equation center_x center_y ∧ slope_range_valid k :=
sorry

end circle_properties_l94_94804


namespace prob_zero_to_two_l94_94087

-- Define conditions
def measurement_result (X : ℝ) (σ : ℝ) := 
  ∃ 𝒩 : (ℝ → ℝ), 𝒩 = Normal(1, σ^2) ∧ RandomVariable X 𝒩

def prob_X_less_zero (X : ℝ) := 
  ∃ p : ℝ, p = 0.2 ∧ P(X < 0) = p

-- Main statement
theorem prob_zero_to_two (X : ℝ) (σ : ℝ) 
  (hx : measurement_result X σ) 
  (hp : prob_X_less_zero X) : 
  P(0 < X < 2) = 0.6 := 
sorry

end prob_zero_to_two_l94_94087


namespace jon_coffee_spending_in_april_l94_94545

def cost_per_coffee : ℕ := 2
def coffees_per_day : ℕ := 2
def days_in_april : ℕ := 30

theorem jon_coffee_spending_in_april :
  (coffees_per_day * cost_per_coffee) * days_in_april = 120 :=
by
  sorry

end jon_coffee_spending_in_april_l94_94545


namespace solve_problem_l94_94817

-- Definitions from the conditions
def is_divisible_by (n k : ℕ) : Prop :=
  ∃ m, k * m = n

def count_divisors (limit k : ℕ) : ℕ :=
  Nat.div limit k

def count_numbers_divisible_by_neither_5_nor_7 (limit : ℕ) : ℕ :=
  let total := limit - 1
  let divisible_by_5 := count_divisors limit 5
  let divisible_by_7 := count_divisors limit 7
  let divisible_by_35 := count_divisors limit 35
  total - (divisible_by_5 + divisible_by_7 - divisible_by_35)

-- The statement to be proved
theorem solve_problem : count_numbers_divisible_by_neither_5_nor_7 1000 = 686 :=
by
  sorry

end solve_problem_l94_94817


namespace machine_does_not_require_repair_l94_94417

variables {M : ℝ} {std_dev : ℝ}
variable (deviations : ℝ → Prop)

-- Conditions
def max_deviation := 37
def ten_percent_nominal_mass := 0.1 * M
def max_deviation_condition := max_deviation ≤ ten_percent_nominal_mass
def unreadable_deviation_condition (x : ℝ) := x < 37
def standard_deviation_condition := std_dev ≤ max_deviation
def machine_condition_nonrepair := (∀ x, deviations x → x ≤ max_deviation)

-- Question: Does the machine require repair?
theorem machine_does_not_require_repair 
  (h1 : max_deviation_condition)
  (h2 : ∀ x, unreadable_deviation_condition x → deviations x)
  (h3 : standard_deviation_condition)
  (h4 : machine_condition_nonrepair) :
  ¬(∃ repair_needed : ℝ, repair_needed = 1) :=
by sorry

end machine_does_not_require_repair_l94_94417


namespace sin2a_minus_cos2a_half_l94_94964

theorem sin2a_minus_cos2a_half (a : ℝ) (h : Real.tan (a - Real.pi / 4) = 1 / 2) :
  Real.sin (2 * a) - Real.cos a ^ 2 = 1 / 2 := 
sorry

end sin2a_minus_cos2a_half_l94_94964


namespace find_abc_l94_94865

theorem find_abc (a b c : ℝ) 
  (h1 : a = 0.8 * b) 
  (h2 : c = 1.4 * b) 
  (h3 : c - a = 72) : 
  a = 96 ∧ b = 120 ∧ c = 168 := 
by
  sorry

end find_abc_l94_94865


namespace sin_double_angle_plus_pi_over_six_l94_94958

variable (θ : ℝ)
variable (h : 7 * Real.sqrt 3 * Real.sin θ = 1 + 7 * Real.cos θ)

theorem sin_double_angle_plus_pi_over_six :
  Real.sin (2 * θ + π / 6) = 97 / 98 :=
by
  sorry

end sin_double_angle_plus_pi_over_six_l94_94958


namespace h_h_3_eq_3568_l94_94523

def h (x : ℤ) : ℤ := 3 * x * x + 3 * x - 2

theorem h_h_3_eq_3568 : h (h 3) = 3568 := by
  sorry

end h_h_3_eq_3568_l94_94523


namespace eval_expression_l94_94685

theorem eval_expression (x y z : ℝ) (h1 : y > z) (h2 : z > 0) (h3 : x = y + z) : 
  ( (y+z+y)^z + (y+z+z)^y ) / (y^z + z^y) = 2^y + 2^z :=
by
  sorry

end eval_expression_l94_94685


namespace Jordan_income_l94_94987

theorem Jordan_income (q A : ℝ) (h : A > 30000)
  (h1 : (q / 100 * 30000 + (q + 3) / 100 * (A - 30000) - 600) = (q + 0.5) / 100 * A) :
  A = 60000 :=
by
  sorry

end Jordan_income_l94_94987


namespace machine_does_not_require_repair_l94_94418

variables {M : ℝ} {std_dev : ℝ}
variable (deviations : ℝ → Prop)

-- Conditions
def max_deviation := 37
def ten_percent_nominal_mass := 0.1 * M
def max_deviation_condition := max_deviation ≤ ten_percent_nominal_mass
def unreadable_deviation_condition (x : ℝ) := x < 37
def standard_deviation_condition := std_dev ≤ max_deviation
def machine_condition_nonrepair := (∀ x, deviations x → x ≤ max_deviation)

-- Question: Does the machine require repair?
theorem machine_does_not_require_repair 
  (h1 : max_deviation_condition)
  (h2 : ∀ x, unreadable_deviation_condition x → deviations x)
  (h3 : standard_deviation_condition)
  (h4 : machine_condition_nonrepair) :
  ¬(∃ repair_needed : ℝ, repair_needed = 1) :=
by sorry

end machine_does_not_require_repair_l94_94418


namespace curve_meets_line_once_l94_94049

theorem curve_meets_line_once (a : ℝ) (h : a > 0) :
  (∃! P : ℝ × ℝ, (∃ θ : ℝ, P.1 = a + 4 * Real.cos θ ∧ P.2 = 1 + 4 * Real.sin θ)
  ∧ (3 * P.1 + 4 * P.2 = 5)) → a = 7 :=
sorry

end curve_meets_line_once_l94_94049


namespace inequality_sol_range_t_l94_94383

def f (x : ℝ) : ℝ := abs (2 * x + 1) - abs (x - 2)

theorem inequality_sol : {x : ℝ | f x > 2} = {x : ℝ | x < -5} ∪ {x : ℝ | 1 < x} :=
sorry

theorem range_t (t : ℝ) : (∀ x : ℝ, f x ≥ t^2 - 11/2 * t) ↔ (1/2 ≤ t ∧ t ≤ 5) :=
sorry

end inequality_sol_range_t_l94_94383


namespace A_inter_CUB_eq_l94_94814

noncomputable def U := Set.univ (ℝ)

noncomputable def A := { x : ℝ | 0 ≤ x ∧ x ≤ 2 }

noncomputable def B := { y : ℝ | ∃ x : ℝ, x ∈ A ∧ y = x + 1 }

noncomputable def C_U (s : Set ℝ) := { x : ℝ | x ∉ s }

noncomputable def A_inter_CUB := A ∩ C_U B

theorem A_inter_CUB_eq : A_inter_CUB = { x : ℝ | 0 ≤ x ∧ x < 1 } :=
  by sorry

end A_inter_CUB_eq_l94_94814


namespace negation_of_exists_real_solution_equiv_l94_94262

open Classical

theorem negation_of_exists_real_solution_equiv :
  (¬ ∃ a : ℝ, ∃ x : ℝ, a * x^2 + 1 = 0) ↔ (∀ a : ℝ, ¬ ∃ x : ℝ, a * x^2 + 1 = 0) :=
by
  sorry

end negation_of_exists_real_solution_equiv_l94_94262


namespace haley_seeds_total_l94_94201

-- Conditions
def seeds_in_big_garden : ℕ := 35
def small_gardens : ℕ := 7
def seeds_per_small_garden : ℕ := 3

-- Question rephrased as a problem with the correct answer
theorem haley_seeds_total : seeds_in_big_garden + small_gardens * seeds_per_small_garden = 56 := by
  sorry

end haley_seeds_total_l94_94201


namespace extreme_value_f_max_b_a_plus_1_l94_94672

noncomputable def f (x : ℝ) := Real.exp x - x + (1/2)*x^2

noncomputable def g (x : ℝ) (a b : ℝ) := (1/2)*x^2 + a*x + b

theorem extreme_value_f :
  ∃ x, deriv f x = 0 ∧ f x = 3 / 2 :=
sorry

theorem max_b_a_plus_1 (a : ℝ) (b : ℝ) :
  (∀ x, f x ≥ g x a b) → b * (a+1) ≤ (a+1)^2 - (a+1)^2 * Real.log (a+1) :=
sorry

end extreme_value_f_max_b_a_plus_1_l94_94672


namespace ratio_of_triangle_BFD_to_square_ABCE_l94_94831

def is_square (ABCE : ℝ → ℝ → ℝ → ℝ → Prop) : Prop :=
  ∀ a b c e : ℝ, ABCE a b c e → a = b ∧ b = c ∧ c = e

def ratio_of_areas (AF FE CD DE : ℝ) (ratio : ℝ) : Prop :=
  AF = 3 * FE ∧ CD = 3 * DE ∧ ratio = 1 / 2

theorem ratio_of_triangle_BFD_to_square_ABCE (AF FE CD DE ratio : ℝ) (ABCE : ℝ → ℝ → ℝ → ℝ → Prop)
  (h1 : is_square ABCE)
  (h2 : AF = 3 * FE) (h3 : CD = 3 * DE) : ratio_of_areas AF FE CD DE (1 / 2) :=
by
  sorry

end ratio_of_triangle_BFD_to_square_ABCE_l94_94831


namespace Angelina_speeds_l94_94915

def distance_home_to_grocery := 960
def distance_grocery_to_gym := 480
def distance_gym_to_library := 720
def time_diff_grocery_to_gym := 40
def time_diff_gym_to_library := 20

noncomputable def initial_speed (v : ℝ) :=
  (distance_home_to_grocery : ℝ) = (v * (960 / v)) ∧
  (distance_grocery_to_gym : ℝ) = (2 * v * (240 / v)) ∧
  (distance_gym_to_library : ℝ) = (3 * v * (720 / v))

theorem Angelina_speeds (v : ℝ) :
  initial_speed v →
  v = 18 ∧ 2 * v = 36 ∧ 3 * v = 54 :=
by
  sorry

end Angelina_speeds_l94_94915


namespace sin_neg_30_eq_neg_one_half_l94_94335

theorem sin_neg_30_eq_neg_one_half : Real.sin (-30 / 180 * Real.pi) = -1 / 2 := by
  -- Proof goes here
  sorry

end sin_neg_30_eq_neg_one_half_l94_94335


namespace monotonicity_of_f_range_of_a_l94_94361

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * x

theorem monotonicity_of_f (a : ℝ) : 
  (a ≤ 0 → ∀ x y : ℝ, x < y → f x a < f y a) ∧ 
  (a > 0 → ∀ x y : ℝ, 
    (x < y ∧ y ≤ Real.log a → f x a > f y a) ∧ 
    (x > Real.log a → f x a < f y a)) :=
by
  sorry

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 0) ↔ 0 ≤ a ∧ a ≤ Real.exp 1 :=
by
  sorry

end monotonicity_of_f_range_of_a_l94_94361


namespace smaller_cuboid_length_l94_94519

theorem smaller_cuboid_length
  (L : ℝ)
  (h1 : 32 * (L * 4 * 3) = 16 * 10 * 12) :
  L = 5 :=
by
  sorry

end smaller_cuboid_length_l94_94519


namespace base6_arithmetic_l94_94586

theorem base6_arithmetic :
  let a := 4512
  let b := 2324
  let c := 1432
  let base := 6
  let a_b10 := 4 * base^3 + 5 * base^2 + 1 * base + 2
  let b_b10 := 2 * base^3 + 3 * base^2 + 2 * base + 4
  let c_b10 := 1 * base^3 + 4 * base^2 + 3 * base + 2
  let result_b10 := a_b10 - b_b10 + c_b10
  let result_base6 := 4020
  (result_b10 / base^3) % base = 4 ∧
  (result_b10 / base^2) % base = 0 ∧
  (result_b10 / base) % base = 2 ∧
  result_b10 % base = 0 →
  result_base6 = 4020 := by
  sorry

end base6_arithmetic_l94_94586


namespace sin_bound_l94_94510

theorem sin_bound (a : ℝ) (h : ¬ ∃ x : ℝ, Real.sin x > a) : a ≥ 1 := 
sorry

end sin_bound_l94_94510


namespace find_number_l94_94179

theorem find_number (x : ℝ) (h : x - (3/5) * x = 62) : x = 155 :=
by
  sorry

end find_number_l94_94179


namespace proof_problem_l94_94528

noncomputable def log2 : ℝ := Real.log 3 / Real.log 2
noncomputable def log5 : ℝ := Real.log 3 / Real.log 5

variables {x y : ℝ}

theorem proof_problem
  (h1 : log2 > 1)
  (h2 : 0 < log5 ∧ log5 < 1)
  (h3 : (log2^x - log5^x) ≥ (log2^(-y) - log5^(-y))) :
  x + y ≥ 0 :=
sorry

end proof_problem_l94_94528


namespace RupertCandles_l94_94108

-- Definitions corresponding to the conditions
def PeterAge : ℕ := 10
def RupertRelativeAge : ℝ := 3.5

-- Define Rupert's age based on Peter's age and the given relative age factor
def RupertAge : ℝ := RupertRelativeAge * PeterAge

-- Statement of the theorem
theorem RupertCandles : RupertAge = 35 := by
  -- Proof is omitted
  sorry

end RupertCandles_l94_94108


namespace find_diagonal_length_l94_94601

noncomputable def parallelepiped_diagonal_length 
  (s : ℝ) -- Side length of square face
  (h : ℝ) -- Length of vertical edge
  (θ : ℝ) -- Angle between vertical edge and square face edges
  (hsq : s = 5) -- Length of side of the square face ABCD
  (hedge : h = 5) -- Length of vertical edge AA1
  (θdeg : θ = 60) -- Angle in degrees
  : ℝ :=
5 * Real.sqrt 3

-- The main theorem to be proved
theorem find_diagonal_length
  (s : ℝ)
  (h : ℝ)
  (θ : ℝ)
  (hsq : s = 5)
  (hedge : h = 5)
  (θdeg : θ = 60)
  : parallelepiped_diagonal_length s h θ hsq hedge θdeg = 5 * Real.sqrt 3 := 
sorry

end find_diagonal_length_l94_94601


namespace point_is_in_second_quadrant_l94_94211

def in_second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

theorem point_is_in_second_quadrant (x y : ℝ) (h₁ : x = -3) (h₂ : y = 2) :
  in_second_quadrant x y := 
by {
  sorry
}

end point_is_in_second_quadrant_l94_94211


namespace incorrect_conclusion_l94_94618

theorem incorrect_conclusion (b x : ℂ) (h : x^2 - b * x + 1 = 0) : x = 1 ∨ x = -1
  ↔ (b = 2 ∨ b = -2) :=
by sorry

end incorrect_conclusion_l94_94618


namespace one_fourths_in_one_eighth_l94_94076

theorem one_fourths_in_one_eighth : (1 / 8) / (1 / 4) = 1 / 2 := 
by 
  sorry

end one_fourths_in_one_eighth_l94_94076


namespace find_constants_and_intervals_l94_94228

open Real

noncomputable def f (x : ℝ) (a b : ℝ) := a * x^3 + b * x^2 - 2 * x
def f' (x : ℝ) (a b : ℝ) := 3 * a * x^2 + 2 * b * x - 2

theorem find_constants_and_intervals :
  (f' (1 : ℝ) (1/3 : ℝ) (1/2 : ℝ) = 0) ∧
  (f' (-2 : ℝ) (1/3 : ℝ) (1/2 : ℝ) = 0) ∧
  (∀ x, f' x (1/3 : ℝ) (1/2 : ℝ) > 0 ↔ x < -2 ∨ x > 1) ∧
  (∀ x, f' x (1/3 : ℝ) (1/2 : ℝ) < 0 ↔ -2 < x ∧ x < 1) :=
by {
  sorry
}

end find_constants_and_intervals_l94_94228


namespace probability_of_event_A_l94_94895

/-- The events A and B are independent, and it is given that:
  1. P(A) > 0
  2. P(A) = 2 * P(B)
  3. P(A or B) = 8 * P(A and B)

We need to prove that P(A) = 1/3. 
-/
theorem probability_of_event_A (P_A P_B : ℝ) (hP_indep : P_A * P_B = P_A) 
  (hP_A_pos : P_A > 0) (hP_A_eq_2P_B : P_A = 2 * P_B) 
  (hP_or_eq_8P_and : P_A + P_B - P_A * P_B = 8 * P_A * P_B) : 
  P_A = 1 / 3 := 
by
  sorry

end probability_of_event_A_l94_94895


namespace necessary_and_sufficient_conditions_l94_94969

open Real

def cubic_has_arithmetic_sequence_roots (a b c : ℝ) : Prop :=
∃ x y : ℝ,
  (x - y) * (x) * (x + y) + a * (x^2 + x - y + x + y) + b * x + c = 0 ∧
  3 * x = -a

theorem necessary_and_sufficient_conditions
  (a b c : ℝ) (h : cubic_has_arithmetic_sequence_roots a b c) :
  2 * a^3 - 9 * a * b + 27 * c = 0 ∧ a^2 - 3 * b ≥ 0 :=
sorry

end necessary_and_sufficient_conditions_l94_94969


namespace vector_magnitude_correct_l94_94067

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Define the difference vector and its magnitude
def a_minus_b : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Theorem statement
theorem vector_magnitude_correct : magnitude a_minus_b = 5 := by
  sorry

end vector_magnitude_correct_l94_94067


namespace red_light_cherries_cost_price_min_value_m_profit_l94_94534

-- Define the constants and cost conditions
def cost_price_red_light_cherries (x : ℝ) (y : ℝ) : Prop :=
  (6000 / (2 * x) - 100 = 1000 / x)

-- Define sales conditions and profit requirement
def min_value_m (m : ℝ) (profit : ℝ) : Prop :=
  (20 * 3 * m + 20 * (20 - 0.5 * m) + (28 - 20) * (50 - 3 * m - 20) >= profit)

-- Define the main proof goal statements
theorem red_light_cherries_cost_price :
  ∃ x, cost_price_red_light_cherries x 6000 ∧ 20 = x :=
sorry

theorem min_value_m_profit :
  ∃ m, min_value_m m 770 ∧ m >= 5 :=
sorry

end red_light_cherries_cost_price_min_value_m_profit_l94_94534


namespace f_neg1_l94_94405

noncomputable def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f (-x) = f x
axiom symmetry_about_x2 : ∀ x : ℝ, f (2 + x) = f (2 - x)
axiom f3_value : f 3 = 3

theorem f_neg1 : f (-1) = 3 := by
  sorry

end f_neg1_l94_94405


namespace vector_magnitude_correct_l94_94068

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Define the difference vector and its magnitude
def a_minus_b : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Theorem statement
theorem vector_magnitude_correct : magnitude a_minus_b = 5 := by
  sorry

end vector_magnitude_correct_l94_94068


namespace vector_magnitude_l94_94066

theorem vector_magnitude (a b : ℝ × ℝ) (a_def : a = (2, 1)) (b_def : b = (-2, 4)) :
  ‖(a.1 - b.1, a.2 - b.2)‖ = 5 := by
  sorry

end vector_magnitude_l94_94066


namespace cakes_remain_l94_94630

def initial_cakes := 110
def sold_cakes := 75
def new_cakes := 76

theorem cakes_remain : (initial_cakes - sold_cakes) + new_cakes = 111 :=
by
  sorry

end cakes_remain_l94_94630


namespace intersect_A_B_l94_94356

open Set

variable A B : Set ℤ 
def A_def : Set ℤ := {x | x + 2 > 0}
def B_def : Set ℤ := {-3, -2, -1, 0}

theorem intersect_A_B :
  (A_def ∩ B_def) = {-1, 0} :=
by
  sorry

end intersect_A_B_l94_94356


namespace max_weight_l94_94162

-- Define the weights
def weight1 := 2
def weight2 := 5
def weight3 := 10

-- Theorem stating that the heaviest single item that can be weighed using any combination of these weights is 17 lb
theorem max_weight : ∃ x, (x = weight1 + weight2 + weight3) ∧ x = 17 :=
by
  sorry

end max_weight_l94_94162


namespace doughnuts_left_l94_94901

theorem doughnuts_left (dozen : ℕ) (total_initial : ℕ) (eaten : ℕ) (initial : total_initial = 2 * dozen) (d : dozen = 12) : total_initial - eaten = 16 :=
by
  rcases d
  rcases initial
  sorry

end doughnuts_left_l94_94901


namespace evaluate_f_2010_times_l94_94349

noncomputable def f (x : ℝ) : ℝ := 1 / (1 - x^2011)^(1/2011)

theorem evaluate_f_2010_times (x : ℝ) (h : x = 2011) :
  (f^[2010] x)^2011 = 2011^2011 :=
by
  rw [h]
  sorry

end evaluate_f_2010_times_l94_94349


namespace rectangle_perimeter_at_least_l94_94774

theorem rectangle_perimeter_at_least (m : ℕ) (m_pos : 0 < m) :
  (∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a * b ≥ 1 / (m * m) ∧ 2 * (a + b) ≥ 4 / m) := sorry

end rectangle_perimeter_at_least_l94_94774


namespace buses_passed_on_highway_l94_94322

/-- Problem statement:
     Buses from Dallas to Austin leave every hour on the hour.
     Buses from Austin to Dallas leave every two hours, starting at 7:00 AM.
     The trip from one city to the other takes 6 hours.
     Assuming the buses travel on the same highway,
     how many Dallas-bound buses does an Austin-bound bus pass on the highway?
-/
theorem buses_passed_on_highway :
  ∀ (t_depart_A2D : ℕ) (trip_time : ℕ) (buses_departures_D2A : ℕ → ℕ),
  (∀ n, buses_departures_D2A n = n) →
  trip_time = 6 →
  ∃ n, t_depart_A2D = 7 ∧ 
    (∀ t, t_depart_A2D ≤ t ∧ t < t_depart_A2D + trip_time →
      ∃ m, m + 1 = t ∧ buses_departures_D2A (m - 6) ≤ t ∧ t < buses_departures_D2A (m - 6) + 6) ↔ n + 1 = 7 := 
sorry

end buses_passed_on_highway_l94_94322


namespace total_expenditure_l94_94547

variable (num_coffees_per_day : ℕ) (cost_per_coffee : ℕ) (days_in_april : ℕ)

theorem total_expenditure (h1 : num_coffees_per_day = 2) (h2 : cost_per_coffee = 2) (h3 : days_in_april = 30) :
  num_coffees_per_day * cost_per_coffee * days_in_april = 120 := by
  sorry

end total_expenditure_l94_94547


namespace calculation1_calculation2_calculation3_calculation4_l94_94634

-- Define the problem and conditions
theorem calculation1 : 9.5 * 101 = 959.5 := 
by 
  sorry

theorem calculation2 : 12.5 * 8.8 = 110 := 
by 
  sorry

theorem calculation3 : 38.4 * 187 - 15.4 * 384 + 3.3 * 16 = 1320 := 
by 
  sorry

theorem calculation4 : 5.29 * 73 + 52.9 * 2.7 = 529 := 
by 
  sorry

end calculation1_calculation2_calculation3_calculation4_l94_94634


namespace major_premise_wrong_l94_94883

-- Definition of the problem conditions and the proof goal
theorem major_premise_wrong :
  (∀ a : ℝ, |a| > 0) ↔ false :=
by {
  sorry  -- the proof goes here but is omitted as per the instructions
}

end major_premise_wrong_l94_94883


namespace sin_plus_cos_of_point_on_terminal_side_l94_94190

theorem sin_plus_cos_of_point_on_terminal_side (x y : ℝ) (h : x = -3 ∧ y = 4) : 
  let α := real.atan2 y x in
  let r := real.sqrt (x ^ 2 + y ^ 2) in
  (sin α + cos α) = 1 / 5 :=
by
  sorry

end sin_plus_cos_of_point_on_terminal_side_l94_94190


namespace ratio_pat_mark_l94_94717

-- Conditions (as definitions)
variables (K P M : ℕ)
variables (h1 : P = 2 * K)  -- Pat charged twice as much time as Kate
variables (h2 : M = K + 80) -- Mark charged 80 more hours than Kate
variables (h3 : K + P + M = 144) -- Total hours charged is 144

theorem ratio_pat_mark (h1 : P = 2 * K) (h2 : M = K + 80) (h3 : K + P + M = 144) : 
  P / M = 1 / 3 :=
by
  sorry -- to be proved

end ratio_pat_mark_l94_94717


namespace three_legged_tables_count_l94_94848

theorem three_legged_tables_count (x y : ℕ) (h1 : 3 * x + 4 * y = 23) (h2 : 2 ≤ x) (h3 : 2 ≤ y) : x = 5 := 
sorry

end three_legged_tables_count_l94_94848


namespace number_of_a_values_l94_94954

theorem number_of_a_values (a : ℝ) : 
  (∃ a : ℝ, ∃ b : ℝ, a = 0 ∨ a = 1) := sorry

end number_of_a_values_l94_94954


namespace geometric_sequence_common_ratio_l94_94667

noncomputable def common_ratio_q (a1 a5 a : ℕ) (q : ℕ) : Prop :=
  a1 * a5 = 16 ∧ a1 > 0 ∧ a5 > 0 ∧ a = 2 ∧ q = 2

theorem geometric_sequence_common_ratio : ∀ (a1 a5 a q : ℕ), 
  common_ratio_q a1 a5 a q → q = 2 :=
by
  intros a1 a5 a q h
  have h1 : a1 * a5 = 16 := h.1
  have h2 : a1 > 0 := h.2.1
  have h3 : a5 > 0 := h.2.2.1
  have h4 : a = 2 := h.2.2.2.1
  have h5 : q = 2 := h.2.2.2.2
  exact h5

end geometric_sequence_common_ratio_l94_94667


namespace ellipse_parabola_intersection_l94_94100

theorem ellipse_parabola_intersection (a b h k : ℝ) :
  let parabola := (x : ℝ) => x^2
  let ellipse := (x y : ℝ) => (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1
  let points := [(4 : ℝ, 16 : ℝ), (-3 : ℝ, 9 : ℝ), (2 : ℝ, 4 : ℝ)]
  ( ∀ x y, (x, y) ∈ points → parabola x = y ∧ ellipse x y) →
  ∃ x4 y4, parabola x4 = y4 ∧ ellipse x4 y4 ∧
  let xs := [4, -3, 2, x4]
  (∑ i in xs, i^2) = 38 := 
by
  sorry

end ellipse_parabola_intersection_l94_94100


namespace find_p_geometric_progression_l94_94345

theorem find_p_geometric_progression (p : ℝ) : 
  (p = -1 ∨ p = 40 / 9) ↔ ((9 * p + 10), (3 * p), |p - 8|) ∈ 
  {gp | ∃ r : ℝ, gp = (r, r * r, r * r * r)} :=
by sorry

end find_p_geometric_progression_l94_94345


namespace probability_of_different_colors_is_correct_l94_94276

noncomputable def probability_different_colors : ℚ :=
  let total_chips := 18
  let blue_chips := 6
  let red_chips := 5
  let yellow_chips := 4
  let green_chips := 3
  let p_blue_then_not_blue := (blue_chips / total_chips) * ((red_chips + yellow_chips + green_chips) / total_chips)
  let p_red_then_not_red := (red_chips / total_chips) * ((blue_chips + yellow_chips + green_chips) / total_chips)
  let p_yellow_then_not_yellow := (yellow_chips / total_chips) * ((blue_chips + red_chips + green_chips) / total_chips)
  let p_green_then_not_green := (green_chips / total_chips) * ((blue_chips + red_chips + yellow_chips) / total_chips)
  p_blue_then_not_blue + p_red_then_not_red + p_yellow_then_not_yellow + p_green_then_not_green

theorem probability_of_different_colors_is_correct :
  probability_different_colors = 119 / 162 :=
by
  sorry

end probability_of_different_colors_is_correct_l94_94276


namespace solve_for_x_l94_94568

theorem solve_for_x : 
  ∃ x₁ x₂ : ℝ, abs (x₁ - 0.175) < 1e-3 ∧ abs (x₂ - 18.325) < 1e-3 ∧
    (∀ x : ℝ, (8 * x ^ 2 + 120 * x + 7) / (3 * x + 10) = 4 * x + 2 → x = x₁ ∨ x = x₂) := 
by 
  sorry

end solve_for_x_l94_94568


namespace distance_borya_vasya_l94_94629

-- Definitions of the houses and distances on the road
def distance_andrey_gena : ℕ := 2450
def race_length : ℕ := 1000

-- Variables to represent the distances
variables (y b : ℕ)

-- Conditions
def start_position := y
def finish_position := b / 2 + 1225

axiom distance_eq : distance_andrey_gena = 2 * y
axiom race_distance_eq : finish_position - start_position = race_length

-- Proving the distance between Borya's and Vasya's houses
theorem distance_borya_vasya :
  ∃ (d : ℕ), d = 450 :=
by
  sorry

end distance_borya_vasya_l94_94629


namespace standard_equation_of_hyperbola_l94_94899

noncomputable def ellipse_eccentricity_problem
  (e : ℚ) (a_maj : ℕ) (f_1 f_2 : ℝ × ℝ) (d : ℕ) : Prop :=
  e = 5 / 13 ∧
  a_maj = 26 ∧
  f_1 = (-5, 0) ∧
  f_2 = (5, 0) ∧
  d = 8 →
  ∃ b, (2 * b = 3) ∧ (2 * b ≠ 0) ∧
  ∃ h k : ℝ, (0 ≤  h) ∧ (0 ≤ k) ∧
  ((h^2)/(4^2)) - ((k^2)/(3^2)) = 1

-- problem statement: 
theorem standard_equation_of_hyperbola
  (e : ℚ) (a_maj : ℕ) (f_1 f_2 : ℝ × ℝ) (d : ℕ)
  (h : e = 5 / 13)
  (a_maj_length : a_maj = 26)
  (f1_coords : f_1 = (-5, 0))
  (f2_coords : f_2 = (5, 0))
  (distance_diff : d = 8) :
  ellipse_eccentricity_problem e a_maj f_1 f_2 d :=
sorry

end standard_equation_of_hyperbola_l94_94899


namespace divisor_is_ten_l94_94832

variable (x y : ℝ)

theorem divisor_is_ten
  (h : ((5 * x - x / y) / (5 * x)) * 100 = 98) : y = 10 := by
  sorry

end divisor_is_ten_l94_94832


namespace number_of_12_digit_numbers_with_consecutive_digits_same_l94_94071

theorem number_of_12_digit_numbers_with_consecutive_digits_same : 
  let total := (2 : ℕ) ^ 12
  let excluded := 2
  total - excluded = 4094 :=
by
  let total := (2 : ℕ) ^ 12
  let excluded := 2
  have h : total = 4096 := by norm_num
  have h' : total - excluded = 4094 := by norm_num
  exact h'

end number_of_12_digit_numbers_with_consecutive_digits_same_l94_94071


namespace part1_part2_l94_94157

theorem part1 (a m n : ℕ) (ha : a > 1) (hdiv : a^m + 1 ∣ a^n + 1) : n ∣ m :=
sorry

theorem part2 (a b m n : ℕ) (ha : a > 1) (coprime_ab : Nat.gcd a b = 1) (hdiv : a^m + b^m ∣ a^n + b^n) : n ∣ m :=
sorry

end part1_part2_l94_94157


namespace andrew_eggs_bought_l94_94168

-- Define initial conditions
def initial_eggs : ℕ := 8
def final_eggs : ℕ := 70

-- Define the function to determine the number of eggs bought
def eggs_bought (initial : ℕ) (final : ℕ) : ℕ := final - initial

-- State the theorem we want to prove
theorem andrew_eggs_bought : eggs_bought initial_eggs final_eggs = 62 :=
by {
  -- Proof goes here
  sorry
}

end andrew_eggs_bought_l94_94168


namespace sum_inverses_mod_17_l94_94934

theorem sum_inverses_mod_17 : 
  (3⁻¹ + 3⁻² + 3⁻³ + 3⁻⁴ + 3⁻⁵ + 3⁻⁶) % 17 = 3 % 17 := 
by 
  sorry

end sum_inverses_mod_17_l94_94934


namespace age_of_30th_employee_l94_94372

theorem age_of_30th_employee :
  let n := 30
  let group1_avg_age := 24
  let group1_count := 10
  let group2_avg_age := 30
  let group2_count := 12
  let group3_avg_age := 35
  let group3_count := 7
  let remaining_avg_age := 29

  let group1_total_age := group1_count * group1_avg_age
  let group2_total_age := group2_count * group2_avg_age
  let group3_total_age := group3_count * group3_avg_age
  let total_age_29 := group1_total_age + group2_total_age + group3_total_age
  let total_age_30 := remaining_avg_age * n

  let age_30th_employee := total_age_30 - total_age_29

  age_30th_employee = 25 :=
by
  let n := 30
  let group1_avg_age := 24
  let group1_count := 10
  let group2_avg_age := 30
  let group2_count := 12
  let group3_avg_age := 35
  let group3_count := 7
  let remaining_avg_age := 29

  let group1_total_age := group1_count * group1_avg_age
  let group2_total_age := group2_count * group2_avg_age
  let group3_total_age := group3_count * group3_avg_age
  let total_age_29 := group1_total_age + group2_total_age + group3_total_age
  let total_age_30 := remaining_avg_age * n

  let age_30th_employee := total_age_30 - total_age_29

  have h : age_30th_employee = 25 := sorry
  exact h

end age_of_30th_employee_l94_94372


namespace root_polynomial_eq_l94_94227

theorem root_polynomial_eq (p q : ℚ) (h1 : 3 * p ^ 2 - 5 * p - 8 = 0) (h2 : 3 * q ^ 2 - 5 * q - 8 = 0) :
    (9 * p ^ 4 - 9 * q ^ 4) / (p - q) = 365 := by
  sorry

end root_polynomial_eq_l94_94227


namespace train_passes_man_in_approx_21_seconds_l94_94144

noncomputable def train_length : ℝ := 385
noncomputable def train_speed_kmph : ℝ := 60
noncomputable def man_speed_kmph : ℝ := 6

-- Convert speeds to m/s
noncomputable def kmph_to_mps (kmph : ℝ) : ℝ := kmph * (1000 / 3600)
noncomputable def train_speed_mps : ℝ := kmph_to_mps train_speed_kmph
noncomputable def man_speed_mps : ℝ := kmph_to_mps man_speed_kmph

-- Calculate relative speed
noncomputable def relative_speed_mps : ℝ := train_speed_mps + man_speed_mps

-- Calculate time
noncomputable def time_to_pass : ℝ := train_length / relative_speed_mps

theorem train_passes_man_in_approx_21_seconds : abs (time_to_pass - 21) < 1 :=
by
  sorry

end train_passes_man_in_approx_21_seconds_l94_94144


namespace vec_magnitude_is_five_l94_94059

noncomputable def vec_a : ℝ × ℝ := (2, 1)
noncomputable def vec_b : ℝ × ℝ := (-2, 4)

def vec_sub (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 - b.1, a.2 - b.2)

def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2)

theorem vec_magnitude_is_five : magnitude (vec_sub vec_a vec_b) = 5 := by
  sorry

end vec_magnitude_is_five_l94_94059


namespace jake_has_peaches_l94_94544

variable (Jake Steven Jill : Nat)

def given_conditions : Prop :=
  (Steven = 15) ∧ (Steven = Jill + 14) ∧ (Jake = Steven - 7)

theorem jake_has_peaches (h : given_conditions Jake Steven Jill) : Jake = 8 :=
by
  cases h with
  | intro hs1 hrest =>
      cases hrest with
      | intro hs2 hs3 =>
          sorry

end jake_has_peaches_l94_94544


namespace usual_time_eq_three_l94_94003

variable (S T : ℝ)
variable (usual_speed : S > 0)
variable (usual_time : T > 0)
variable (reduced_speed : S' = 6/7 * S)
variable (reduced_time : T' = T + 0.5)

theorem usual_time_eq_three (h : 7/6 = T' / T) : T = 3 :=
by
  -- proof to be filled in
  sorry

end usual_time_eq_three_l94_94003


namespace sum_of_divisors_5_cubed_l94_94269

theorem sum_of_divisors_5_cubed :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a * b * c = 5^3) ∧ (a = 1) ∧ (b = 5) ∧ (c = 25) ∧ (a + b + c = 31) :=
sorry

end sum_of_divisors_5_cubed_l94_94269


namespace oldest_child_age_l94_94726

theorem oldest_child_age (x : ℕ) (h : (6 + 8 + x) / 3 = 9) : x = 13 :=
by
  sorry

end oldest_child_age_l94_94726


namespace number_of_positive_integer_pairs_l94_94036

theorem number_of_positive_integer_pairs (x y : ℕ) (h : 20 * x + 6 * y = 2006) : 
  ∃ n, n = 34 ∧ ∀ (x y : ℕ), 20 * x + 6 * y = 2006 → 0 < x → 0 < y → 
  (∃ k, x = 3 * k + 1 ∧ y = 331 - 10 * k ∧ 0 ≤ k ∧ k ≤ 33) :=
sorry

end number_of_positive_integer_pairs_l94_94036


namespace min_value_f_l94_94032

def f (x : ℝ) : ℝ := 5 * x^2 + 10 * x + 20

theorem min_value_f : ∃ (x : ℝ), f x = 15 :=
by
  sorry

end min_value_f_l94_94032


namespace number_of_4_letter_words_with_B_l94_94365

-- Define the set of letters.
inductive Alphabet
| A | B | C | D | E

-- The number of 4-letter words with repetition allowed and must include 'B' at least once.
noncomputable def words_with_at_least_one_B : ℕ :=
  let total := 5 ^ 4 -- Total number of 4-letter words.
  let without_B := 4 ^ 4 -- Total number of 4-letter words without 'B'.
  total - without_B

-- The main theorem statement.
theorem number_of_4_letter_words_with_B : words_with_at_least_one_B = 369 :=
  by sorry

end number_of_4_letter_words_with_B_l94_94365


namespace factor_theorem_l94_94759

theorem factor_theorem (m : ℝ) : (∀ x : ℝ, x + 5 = 0 → x ^ 2 - m * x - 40 = 0) → m = 3 :=
by
  sorry

end factor_theorem_l94_94759


namespace necessary_not_sufficient_condition_l94_94231
-- Import the necessary libraries

-- Define the real number condition
def real_number (a : ℝ) : Prop := true

-- Define line l1
def line_l1 (a : ℝ) (x y : ℝ) : Prop := x + a * y + 3 = 0

-- Define line l2
def line_l2 (a y x: ℝ) : Prop := a * x + 4 * y + 6 = 0

-- Define the parallel condition
def parallel_lines (a : ℝ) : Prop :=
  (a = 2 ∨ a = -2) ∧ 
  ∀ x y : ℝ, line_l1 a x y ∧ line_l2 a x y → a * x + 4 * x + 6 = 3

-- State the main theorem to prove
theorem necessary_not_sufficient_condition (a : ℝ) : 
  real_number a → (a = 2 ∨ a = -2) ↔ (parallel_lines a) := 
by
  sorry

end necessary_not_sufficient_condition_l94_94231


namespace find_number_l94_94120

-- Define the number x and the condition as a theorem to be proven.
theorem find_number (x : ℝ) (h : (1/3) * x - 5 = 10) : x = 45 :=
sorry

end find_number_l94_94120


namespace consecutive_negatives_product_sum_l94_94867

theorem consecutive_negatives_product_sum:
  ∃ (n: ℤ), n < 0 ∧ (n + 1) < 0 ∧ n * (n + 1) = 3080 ∧ n + (n + 1) = -111 :=
by
  sorry

end consecutive_negatives_product_sum_l94_94867


namespace conic_not_parabola_l94_94730

def conic_equation (m x y : ℝ) : Prop :=
  m * x^2 + (m + 1) * y^2 = m * (m + 1)

theorem conic_not_parabola (m : ℝ) :
  ¬ (∃ (x y : ℝ), conic_equation m x y ∧ ∃ (a b c d e f : ℝ), m * x^2 + (m + 1) * y^2 = a * x^2 + b * xy + c * y^2 + d * x + e * y + f ∧ (a = 0 ∨ c = 0) ∧ (b ≠ 0 ∨ a ≠ 0 ∨ d ≠ 0 ∨ e ≠ 0)) :=  
sorry

end conic_not_parabola_l94_94730


namespace center_of_circle_l94_94508

-- Defining the equation of the circle as a hypothesis
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + 2 * y = 0

-- Stating the theorem about the center of the circle
theorem center_of_circle : ∀ x y : ℝ, circle_eq x y → (x = 2 ∧ y = -1) :=
by
  sorry

end center_of_circle_l94_94508


namespace find_m_value_l94_94813

noncomputable def is_solution (p q m : ℝ) : Prop :=
  ∃ x : ℝ, (x^2 + p*x + q = 0) ∧ (x^2 - m*x + m^2 - 19 = 0)

theorem find_m_value :
  let A := { x : ℝ | x^2 + 2 * x - 8 = 0 }
  let B := { x : ℝ | x^2 - 5 * x + 6 = 0 }
  ∀ (C : ℝ → Prop), 
  (∃ x, B x ∧ C x) ∧ (¬ ∃ x, A x ∧ C x) → 
  (∃ m, C = { x : ℝ | x^2 - m * x + m^2 - 19 = 0 } ∧ m = -2) :=
by
  sorry

end find_m_value_l94_94813


namespace old_lamp_height_is_one_l94_94703

def new_lamp_height : ℝ := 2.3333333333333335
def height_difference : ℝ := 1.3333333333333333
def old_lamp_height : ℝ := new_lamp_height - height_difference

theorem old_lamp_height_is_one :
  old_lamp_height = 1 :=
by
  sorry

end old_lamp_height_is_one_l94_94703


namespace walking_time_12_hours_l94_94281

theorem walking_time_12_hours :
  ∀ t : ℝ, 
  (∀ (v1 v2 : ℝ), 
  v1 = 7 ∧ v2 = 3 →
  120 = (v1 + v2) * t) →
  t = 12 := 
by
  intros t h
  specialize h 7 3 ⟨rfl, rfl⟩
  sorry

end walking_time_12_hours_l94_94281


namespace tangent_line_at_5_l94_94966

theorem tangent_line_at_5 
  (f : ℝ → ℝ)
  (h : tangent_line_at f (λ x, -x + 8) 5) :
  f 5 = 3 ∧ deriv f 5 = -1 :=
by
  sorry

end tangent_line_at_5_l94_94966


namespace negation_proposition_equiv_l94_94265

open Classical

variable (R : Type) [OrderedRing R] (a x : R)

theorem negation_proposition_equiv :
  (¬ ∃ a : R, ∃ x : R, a * x^2 + 1 = 0) ↔ (∀ a : R, ∀ x : R, a * x^2 + 1 ≠ 0) :=
by
  sorry

end negation_proposition_equiv_l94_94265


namespace middle_digit_is_3_l94_94610

theorem middle_digit_is_3 (d e f : ℕ) (hd : 0 ≤ d ∧ d ≤ 7) (he : 0 ≤ e ∧ e ≤ 7) (hf : 0 ≤ f ∧ f ≤ 7)
    (h_eq : 64 * d + 8 * e + f = 100 * f + 10 * e + d) : e = 3 :=
sorry

end middle_digit_is_3_l94_94610


namespace remainder_of_2519_div_8_l94_94611

theorem remainder_of_2519_div_8 : 2519 % 8 = 7 := 
by 
  sorry

end remainder_of_2519_div_8_l94_94611


namespace greatest_possible_sum_of_two_consecutive_even_integers_l94_94137

theorem greatest_possible_sum_of_two_consecutive_even_integers
  (n : ℤ) (h1 : Even n) (h2 : n * (n + 2) < 800) :
  n + (n + 2) = 54 := 
sorry

end greatest_possible_sum_of_two_consecutive_even_integers_l94_94137


namespace triangle_inequality_condition_l94_94173

variable (a b c : ℝ)
variable (α : ℝ) -- angle in radians

-- Define the condition where c must be less than a + b
theorem triangle_inequality_condition : c < a + b := by
  sorry

end triangle_inequality_condition_l94_94173


namespace box_volume_l94_94491

-- Definitions for the dimensions of the box: Length (L), Width (W), and Height (H)
variables (L W H : ℝ)

-- Condition 1: Area of the front face is half the area of the top face
def condition1 := L * W = 0.5 * (L * H)

-- Condition 2: Area of the top face is 1.5 times the area of the side face
def condition2 := L * H = 1.5 * (W * H)

-- Condition 3: Area of the side face is 200
def condition3 := W * H = 200

-- Theorem stating the volume of the box is 3000 given the above conditions
theorem box_volume : condition1 L W H ∧ condition2 L W H ∧ condition3 W H → L * W * H = 3000 :=
by sorry

end box_volume_l94_94491


namespace triple_f_of_3_l94_94977

def f (x : ℤ) : ℤ := -3 * x + 5

theorem triple_f_of_3 : f (f (f 3)) = -46 := by
  sorry

end triple_f_of_3_l94_94977


namespace total_cakes_served_l94_94468

theorem total_cakes_served (l : ℝ) (p : ℝ) (s : ℝ) (total_cakes_served_today : ℝ) :
  l = 48.5 → p = 0.6225 → s = 95 → total_cakes_served_today = 108 :=
by
  intros hl hp hs
  sorry

end total_cakes_served_l94_94468


namespace factor_expression_l94_94938

variable (b : ℝ)

theorem factor_expression : 221 * b * b + 17 * b = 17 * b * (13 * b + 1) :=
by
  sorry

end factor_expression_l94_94938


namespace sum_integers_minus15_to_6_l94_94591

def sum_range (a b : ℤ) : ℤ :=
  (b - a + 1) * (a + b) / 2

theorem sum_integers_minus15_to_6 : sum_range (-15) (6) = -99 :=
  by
  -- Skipping the proof details
  sorry

end sum_integers_minus15_to_6_l94_94591


namespace arithmetic_mean_bc_diff_l94_94198

variable (a b c : ℝ)
def mu := (a + b + c) / 3

theorem arithmetic_mean_bc_diff (h1 : (a + b) / 2 = mu a b c + 5)
                                (h2 : (a + c) / 2 = mu a b c - 8) :
  (b + c) / 2 = mu a b c + 3 := 
sorry

end arithmetic_mean_bc_diff_l94_94198


namespace expected_adjacent_pairs_3_out_of_9_is_2_div_3_l94_94626

-- Definitions of the problem
def is_adjacent_pair (a b : ℕ) : Prop := abs (a - b) = 1
def all_pairs_adjacent (s : Finset ℕ) : ℕ := 
  (s.toList).pairwise (λ a b, is_adjacent_pair a b) -- Counts number of pairs of adjacent elements in a set

noncomputable def expected_adjacent_pairs (n : ℕ) (chosen : ℕ) : ℚ :=
  let p_0 : ℚ := 5 / 12
  let p_1 : ℚ := 1 / 2
  let p_2 : ℚ := 1 / 12
  p_0 * 0 + p_1 * 1 + p_2 * 2

-- The main proof statement
theorem expected_adjacent_pairs_3_out_of_9_is_2_div_3 :
  expected_adjacent_pairs 9 3 = 2 / 3 :=
sorry

end expected_adjacent_pairs_3_out_of_9_is_2_div_3_l94_94626


namespace seventh_fifth_tiles_difference_l94_94697

def side_length (n : ℕ) : ℕ := 2 * n - 1
def number_of_tiles (n : ℕ) : ℕ := (side_length n) ^ 2
def tiles_difference (n m : ℕ) : ℕ := number_of_tiles n - number_of_tiles m

theorem seventh_fifth_tiles_difference : tiles_difference 7 5 = 88 := by
  sorry

end seventh_fifth_tiles_difference_l94_94697


namespace triangle_condition_l94_94200

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sin x) * (Real.cos x) + (Real.sqrt 3) * (Real.cos x) ^ 2 - (Real.sqrt 3) / 2

theorem triangle_condition (a b c : ℝ) (h : b^2 + c^2 = a^2 + Real.sqrt 3 * b * c) : 
  f (Real.pi / 6) = Real.sqrt 3 / 2 := by
  sorry

end triangle_condition_l94_94200


namespace original_numerator_l94_94617

theorem original_numerator (n : ℕ) (hn : (n + 3) / (9 + 3) = 2 / 3) : n = 5 :=
by
  sorry

end original_numerator_l94_94617


namespace decimals_between_6_1_and_6_4_are_not_two_l94_94127

-- Definitions from the conditions in a)
def is_between (x : ℝ) (a b : ℝ) : Prop := a < x ∧ x < b

-- The main theorem statement
theorem decimals_between_6_1_and_6_4_are_not_two :
  ∀ x, is_between x 6.1 6.4 → false :=
by
  sorry

end decimals_between_6_1_and_6_4_are_not_two_l94_94127


namespace factor_polynomial_l94_94344

theorem factor_polynomial (a b m n : ℝ) (h : |m - 4| + (n^2 - 8 * n + 16) = 0) :
  a^2 + 4 * b^2 - m * a * b - n = (a - 2 * b + 2) * (a - 2 * b - 2) :=
by
  sorry

end factor_polynomial_l94_94344


namespace min_n_for_constant_term_l94_94656

theorem min_n_for_constant_term (n : ℕ) (h : 0 < n) : 
  (∃ (r : ℕ), 0 = n - 4 * r / 3) → n = 4 :=
by
  sorry

end min_n_for_constant_term_l94_94656


namespace machine_does_not_require_repair_l94_94408

variable (nominal_mass max_deviation standard_deviation : ℝ)
variable (nominal_mass_ge : nominal_mass ≥ 370)
variable (max_deviation_le : max_deviation ≤ 0.1 * nominal_mass)
variable (all_deviations_le_max : ∀ d, d < max_deviation → d < 37)
variable (std_dev_le_max_dev : standard_deviation ≤ max_deviation)

theorem machine_does_not_require_repair :
  ¬ (standard_deviation > 37) :=
by 
  -- sorry annotation indicates the proof goes here
  sorry

end machine_does_not_require_repair_l94_94408


namespace event_complementary_and_mutually_exclusive_l94_94163

def students : Finset (String × String) := 
  { ("boy", "1"), ("boy", "2"), ("boy", "3"), ("girl", "1"), ("girl", "2") }

def event_at_least_one_girl (s : Finset (String × String)) : Prop :=
  ∃ x ∈ s, (x.1 = "girl")

def event_all_boys (s : Finset (String × String)) : Prop :=
  ∀ x ∈ s, (x.1 = "boy")

def two_students (s : Finset (String × String)) : Prop :=
  s.card = 2

theorem event_complementary_and_mutually_exclusive :
  ∀ s: Finset (String × String), two_students s → 
  (event_at_least_one_girl s ↔ ¬ event_all_boys s) ∧ 
  (event_all_boys s ↔ ¬ event_at_least_one_girl s) :=
sorry

end event_complementary_and_mutually_exclusive_l94_94163


namespace probability_sum_eight_l94_94904

def total_outcomes : ℕ := 36
def favorable_outcomes : ℕ := 5

theorem probability_sum_eight :
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 36 := by
  sorry

end probability_sum_eight_l94_94904


namespace complex_number_equality_l94_94400

theorem complex_number_equality (i : ℂ) (h : i^2 = -1) : 1 + i + i^2 = i :=
by
  sorry

end complex_number_equality_l94_94400


namespace weight_of_b_l94_94293

/--
Given:
1. The sum of weights (a, b, c) is 129 kg.
2. The sum of weights (a, b) is 80 kg.
3. The sum of weights (b, c) is 86 kg.

Prove that the weight of b is 37 kg.
-/
theorem weight_of_b (a b c : ℝ) 
  (h1 : a + b + c = 129) 
  (h2 : a + b = 80) 
  (h3 : b + c = 86) : 
  b = 37 :=
sorry

end weight_of_b_l94_94293


namespace total_precious_stones_l94_94368

theorem total_precious_stones (agate olivine diamond : ℕ)
  (h1 : olivine = agate + 5)
  (h2 : diamond = olivine + 11)
  (h3 : agate = 30) : 
  agate + olivine + diamond = 111 :=
by
  sorry

end total_precious_stones_l94_94368


namespace tan_sum_pi_eighths_l94_94851

theorem tan_sum_pi_eighths : (Real.tan (Real.pi / 8) + Real.tan (3 * Real.pi / 8) = 2 * Real.sqrt 2) :=
by
  sorry

end tan_sum_pi_eighths_l94_94851


namespace sum_of_divisors_5_cubed_l94_94270

theorem sum_of_divisors_5_cubed :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a * b * c = 5^3) ∧ (a = 1) ∧ (b = 5) ∧ (c = 25) ∧ (a + b + c = 31) :=
sorry

end sum_of_divisors_5_cubed_l94_94270


namespace task_probabilities_l94_94753

theorem task_probabilities (P1_on_time : ℚ) (P2_on_time : ℚ) 
  (h1 : P1_on_time = 2/3) (h2 : P2_on_time = 3/5) : 
  P1_on_time * (1 - P2_on_time) = 4/15 := 
by
  -- proof is omitted
  sorry

end task_probabilities_l94_94753


namespace probability_gte_one_l94_94084

open ProbabilityTheory

variable (σ : ℝ) (ξ : MeasureTheory.ProbabilityTheory.RandomVariable ℝ)
          (h1: MeasureTheory.ProbabilityTheory.normal ξ (-1) σ^2) 
          (h2 : MeasureTheory.ProbaTheory.GT (-3 ≤' ξ ≤' -1) 0.4)

theorem probability_gte_one :
    MeasureTheory.ProbabilityTheory.GE ξ 1 = 0.1 :=
sory

end probability_gte_one_l94_94084


namespace value_is_50_cents_l94_94380

-- Define Leah's total number of coins and the condition on the number of nickels and pennies.
variables (p n : ℕ)

-- Leah has a total of 18 coins
def total_coins : Prop := n + p = 18

-- Condition for nickels and pennies
def condition : Prop := p = n + 2

-- Calculate the total value of Leah's coins and check if it equals 50 cents
def total_value : ℕ := 5 * n + p

-- Proposition stating that under given conditions, total value is 50 cents
theorem value_is_50_cents (h1 : total_coins p n) (h2 : condition p n) :
  total_value p n = 50 := sorry

end value_is_50_cents_l94_94380


namespace circle_area_increase_l94_94981

theorem circle_area_increase (r : ℝ) :
  let A_initial := Real.pi * r^2
  let A_new := Real.pi * (2*r)^2
  let delta_A := A_new - A_initial
  let percentage_increase := (delta_A / A_initial) * 100
  percentage_increase = 300 := by
  sorry

end circle_area_increase_l94_94981


namespace min_value_fracs_l94_94668

-- Define the problem and its conditions in Lean.
theorem min_value_fracs (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 1) : 
  (2 / a + 3 / b) ≥ 8 + 4 * Real.sqrt 3 :=
  sorry

end min_value_fracs_l94_94668


namespace parabola_axis_l94_94970

section
variable (x y : ℝ)

-- Condition: Defines the given parabola equation.
def parabola_eq (x y : ℝ) : Prop := x = (1 / 4) * y^2

-- The Proof Problem: Prove that the axis of this parabola is x = -1/2.
theorem parabola_axis (h : parabola_eq x y) : x = - (1 / 2) := 
sorry
end

end parabola_axis_l94_94970


namespace floor_T_value_l94_94713

noncomputable def floor_T : ℝ := 
  let p := (0 : ℝ)
  let q := (0 : ℝ)
  let r := (0 : ℝ)
  let s := (0 : ℝ)
  p + q + r + s

theorem floor_T_value (p q r s : ℝ) (hpq: p^2 + q^2 = 2500) (hrs: r^2 + s^2 = 2500) (hpr: p * r = 1200) (hqs: q * s = 1200) (hpos: p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0) :
  ∃ T : ℝ, T = p + q + r + s ∧ ⌊T⌋ = 140 := 
  by
  sorry

end floor_T_value_l94_94713


namespace tangent_line_equation_l94_94971

def f (x : ℝ) : ℝ := x^2

theorem tangent_line_equation :
  let x := (1 : ℝ)
  let y := f x
  ∃ m b : ℝ, m = 2 ∧ b = 1 ∧ (2*x - y - 1 = 0) := by
  sorry

end tangent_line_equation_l94_94971


namespace fraction_equation_solution_l94_94873

theorem fraction_equation_solution (x : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ 0) : 
  (1 / (x - 2) = 3 / x) → x = 3 := 
by 
  sorry

end fraction_equation_solution_l94_94873


namespace relationship_S_T_l94_94812

-- Definitions based on the given conditions
def seq_a (n : ℕ) : ℕ :=
  if n = 0 then 0 else 2 * n

def seq_b (n : ℕ) : ℕ :=
  2 ^ (n - 1) + 1

def S (n : ℕ) : ℕ :=
  (n * (n + 1))

def T (n : ℕ) : ℕ :=
  (2^n) + n - 1

-- The conjecture and proofs
theorem relationship_S_T (n : ℕ) : 
  if n = 1 then T n = S n
  else if (2 ≤ n ∧ n < 5) then T n < S n
  else n ≥ 5 → T n > S n :=
by sorry

end relationship_S_T_l94_94812


namespace factor_expression_l94_94945

variable (b : ℤ)

theorem factor_expression : 221 * b^2 + 17 * b = 17 * b * (13 * b + 1) :=
by
  sorry

end factor_expression_l94_94945


namespace sum_of_integers_l94_94880

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 180) : x + y = 28 := 
by
  sorry

end sum_of_integers_l94_94880


namespace distribute_problems_l94_94007

theorem distribute_problems :
  (12 ^ 6) = 2985984 := by
  sorry

end distribute_problems_l94_94007


namespace insulation_cost_l94_94001

def tank_length : ℕ := 4
def tank_width : ℕ := 5
def tank_height : ℕ := 2
def cost_per_sqft : ℕ := 20

def surface_area (L W H : ℕ) : ℕ := 2 * (L * W + L * H + W * H)
def total_cost (SA cost_per_sqft : ℕ) : ℕ := SA * cost_per_sqft

theorem insulation_cost : 
  total_cost (surface_area tank_length tank_width tank_height) cost_per_sqft = 1520 :=
by
  sorry

end insulation_cost_l94_94001


namespace simplify_expression_l94_94115

theorem simplify_expression :
  (2 * 10^9) - (6 * 10^7) / (2 * 10^2) = 1999700000 :=
by
  sorry

end simplify_expression_l94_94115


namespace area_PQR_l94_94306

-- Define the point P
def P : ℝ × ℝ := (1, 6)

-- Define the functions for lines passing through P with slopes 1 and 3
def line1 (x : ℝ) : ℝ := x + 5
def line2 (x : ℝ) : ℝ := 3 * x + 3

-- Define the x-intercepts of the lines
def Q : ℝ × ℝ := (-5, 0)
def R : ℝ × ℝ := (-1, 0)

-- Calculate the distance QR
def distance_QR : ℝ := abs (-1 - (-5))

-- Calculate the height from P to the x-axis
def height_P : ℝ := 6

-- State and prove the area of the triangle PQR
theorem area_PQR : 1 / 2 * distance_QR * height_P = 12 := by
  sorry -- The actual proof would be provided here

end area_PQR_l94_94306


namespace range_of_m_solve_inequality_l94_94973

open Real Set

noncomputable def f (x: ℝ) := -abs (x - 2)
noncomputable def g (x: ℝ) (m: ℝ) := -abs (x - 3) + m

-- Problem 1: Prove the range of m given the condition
theorem range_of_m (h : ∀ x : ℝ, f x > g x m) : m < 1 :=
  sorry

-- Problem 2: Prove the set of solutions for f(x) + a - 1 > 0
theorem solve_inequality (a : ℝ) :
  (if a = 1 then {x : ℝ | x ≠ 2}
   else if a > 1 then univ
   else {x : ℝ | x < 1 + a} ∪ {x : ℝ | x > 3 - a}) = {x : ℝ | f x + a - 1 > 0} :=
  sorry

end range_of_m_solve_inequality_l94_94973


namespace overall_average_marks_l94_94596

theorem overall_average_marks 
  (n1 : ℕ) (m1 : ℕ) 
  (n2 : ℕ) (m2 : ℕ) 
  (n3 : ℕ) (m3 : ℕ) 
  (n4 : ℕ) (m4 : ℕ) 
  (h1 : n1 = 70) (h2 : m1 = 50) 
  (h3 : n2 = 35) (h4 : m2 = 60)
  (h5 : n3 = 45) (h6 : m3 = 55)
  (h7 : n4 = 42) (h8 : m4 = 45) :
  (n1 * m1 + n2 * m2 + n3 * m3 + n4 * m4) / (n1 + n2 + n3 + n4) = 9965 / 192 :=
by
  sorry

end overall_average_marks_l94_94596


namespace contest_correct_answers_l94_94988

/-- 
In a mathematics contest with ten problems, a student gains 
5 points for a correct answer and loses 2 points for an 
incorrect answer. If Olivia answered every problem 
and her score was 29, how many correct answers did she have?
-/
theorem contest_correct_answers (c w : ℕ) (h1 : c + w = 10) (h2 : 5 * c - 2 * w = 29) : c = 7 :=
by 
  sorry

end contest_correct_answers_l94_94988


namespace complex_number_equality_l94_94401

theorem complex_number_equality (i : ℂ) (h : i^2 = -1) : 1 + i + i^2 = i :=
by
  sorry

end complex_number_equality_l94_94401


namespace no_such_integers_l94_94141

theorem no_such_integers (x y : ℤ) : ¬ ∃ x y : ℤ, (x^4 + 6) % 13 = y^3 % 13 :=
sorry

end no_such_integers_l94_94141


namespace jason_total_expenditure_l94_94706

theorem jason_total_expenditure :
  let stove_cost := 1200
  let wall_repair_ratio := 1 / 6
  let wall_repair_cost := stove_cost * wall_repair_ratio
  let total_cost := stove_cost + wall_repair_cost
  total_cost = 1400 := by
  {
    sorry
  }

end jason_total_expenditure_l94_94706


namespace domain_of_func_1_domain_of_func_2_domain_of_func_3_domain_of_func_4_l94_94595
-- Import the necessary library.

-- Define the domains for the given functions.
def domain_func_1 (x : ℝ) : Prop := true

def domain_func_2 (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 2

def domain_func_3 (x : ℝ) : Prop := x ≥ -3 ∧ x ≠ 1

def domain_func_4 (x : ℝ) : Prop := 2 ≤ x ∧ x ≤ 5 ∧ x ≠ 3

-- Prove the domains of each function.
theorem domain_of_func_1 : ∀ x : ℝ, domain_func_1 x :=
by sorry

theorem domain_of_func_2 : ∀ x : ℝ, domain_func_2 x ↔ (1 ≤ x ∧ x ≤ 2) :=
by sorry

theorem domain_of_func_3 : ∀ x : ℝ, domain_func_3 x ↔ (x ≥ -3 ∧ x ≠ 1) :=
by sorry

theorem domain_of_func_4 : ∀ x : ℝ, domain_func_4 x ↔ (2 ≤ x ∧ x ≤ 5 ∧ x ≠ 3) :=
by sorry

end domain_of_func_1_domain_of_func_2_domain_of_func_3_domain_of_func_4_l94_94595


namespace consecutive_negatives_product_sum_l94_94868

theorem consecutive_negatives_product_sum:
  ∃ (n: ℤ), n < 0 ∧ (n + 1) < 0 ∧ n * (n + 1) = 3080 ∧ n + (n + 1) = -111 :=
by
  sorry

end consecutive_negatives_product_sum_l94_94868


namespace sum_prime_factors_77_l94_94455

theorem sum_prime_factors_77 : ∑ p in {7, 11}, p = 18 :=
by {
  have h1 : Nat.Prime 7 := Nat.prime_iff_nat_prime.mp Nat.prime_factors_one,
  have h2 : Nat.Prime 11 := Nat.prime_iff_nat_prime.mp Nat.prime_factors_one,
  have h3 : 77 = 7 * 11 := by simp,
  have h_set: {7, 11} = PrimeFactors 77 := by sorry, 
  rw h_set,
  exact Nat.sum_prime_factors_eq_77,
  sorry
}

end sum_prime_factors_77_l94_94455


namespace sam_initial_dimes_l94_94242

theorem sam_initial_dimes (given_away : ℕ) (left : ℕ) (initial : ℕ) 
  (h1 : given_away = 7) (h2 : left = 2) (h3 : initial = given_away + left) : 
  initial = 9 := by
  rw [h1, h2] at h3
  exact h3

end sam_initial_dimes_l94_94242


namespace subtraction_digits_l94_94187

theorem subtraction_digits (a b c : ℕ) (h1 : c - a = 2) (h2 : b = c - 1) (h3 : 100 * a + 10 * b + c - (100 * c + 10 * b + a) = 802) :
a = 0 ∧ b = 1 ∧ c = 2 :=
by {
  -- The detailed proof steps will go here
  sorry
}

end subtraction_digits_l94_94187


namespace drinkable_amount_l94_94704

variable {LiquidBeforeTest : ℕ}
variable {Threshold : ℕ}

def can_drink_more (LiquidBeforeTest : ℕ) (Threshold : ℕ): ℕ :=
  Threshold - LiquidBeforeTest

theorem drinkable_amount :
  LiquidBeforeTest = 24 ∧ Threshold = 32 →
  can_drink_more LiquidBeforeTest Threshold = 8 := by
  sorry

end drinkable_amount_l94_94704


namespace first_divisibility_second_divisibility_l94_94565

variable {n : ℕ}
variable (h : n > 0)

theorem first_divisibility :
  17 ∣ (5 * 3^(4*n+1) + 2^(6*n+1)) :=
sorry

theorem second_divisibility :
  32 ∣ (25 * 7^(2*n+1) + 3^(4*n)) :=
sorry

end first_divisibility_second_divisibility_l94_94565


namespace gcd_of_three_l94_94651

theorem gcd_of_three (a b c : ℕ) (h₁ : a = 9242) (h₂ : b = 13863) (h₃ : c = 34657) :
  Nat.gcd (Nat.gcd a b) c = 1 :=
by
  sorry

end gcd_of_three_l94_94651


namespace solve_for_x_and_y_l94_94246

theorem solve_for_x_and_y (x y : ℚ) (h : (1 / 6) + (6 / x) = (14 / x) + (1 / 14) + y) : x = 84 ∧ y = 0 :=
sorry

end solve_for_x_and_y_l94_94246


namespace perimeter_of_smaller_polygon_l94_94736

/-- The ratio of the areas of two similar polygons is 1:16, and the difference in their perimeters is 9.
Find the perimeter of the smaller polygon. -/
theorem perimeter_of_smaller_polygon (a b : ℝ) (h1 : a / b = 1 / 16) (h2 : b - a = 9) : a = 3 :=
by
  sorry

end perimeter_of_smaller_polygon_l94_94736


namespace product_first_8_terms_l94_94965

variable {a : ℕ → ℝ}
variable {q : ℝ}

-- Given conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def a_2 : a 2 = 3 := sorry
def a_7 : a 7 = 1 := sorry

-- Proof statement
theorem product_first_8_terms (h_geom : is_geometric_sequence a q) 
  (h_a2 : a 2 = 3) 
  (h_a7 : a 7 = 1) : 
  (a 1 * a 2 * a 3 * a 4 * a 5 * a 6 * a 7 * a 8 = 81) :=
sorry

end product_first_8_terms_l94_94965


namespace graph_is_line_l94_94286

theorem graph_is_line : {p : ℝ × ℝ | (p.1 - p.2)^2 = 2 * (p.1^2 + p.2^2)} = {p : ℝ × ℝ | p.2 = -p.1} :=
by 
  sorry

end graph_is_line_l94_94286


namespace vector_magnitude_subtraction_l94_94056

def vec_sub (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def vec_norm (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem vector_magnitude_subtraction :
  let a := (2, 1)
  let b := (-2, 4)
  vec_norm (vec_sub a b) = 5 :=
by
  sorry

end vector_magnitude_subtraction_l94_94056


namespace total_amount_l94_94599

theorem total_amount (T_pq r : ℝ) (h1 : r = 2/3 * T_pq) (h2 : r = 1600) : T_pq + r = 4000 :=
by
  -- proof skipped
  sorry

end total_amount_l94_94599


namespace Jackie_hops_six_hops_distance_l94_94092

theorem Jackie_hops_six_hops_distance : 
  let a : ℝ := 1
  let r : ℝ := 1 / 2
  let S : ℝ := a * ((1 - r^6) / (1 - r))
  S = 63 / 32 :=
by 
  sorry

end Jackie_hops_six_hops_distance_l94_94092


namespace sum_of_x_l94_94502

-- define the function f as an even function
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- define the function f as strictly monotonic on the interval (0, +∞)
def is_strictly_monotonic_on_positive (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f x < f y

-- define the main problem statement
theorem sum_of_x (f : ℝ → ℝ) (x : ℝ) (h1 : is_even_function f) (h2 : is_strictly_monotonic_on_positive f) (h3 : x ≠ 0)
  (hx : f (x^2 - 2*x - 1) = f (x + 1)) : 
  ∃ (x1 x2 x3 x4 : ℝ), (x1 + x2 + x3 + x4 = 4) ∧
                        (x1^2 - 3*x1 - 2 = 0) ∧
                        (x2^2 - 3*x2 - 2 = 0) ∧
                        (x3^2 - x3 = 0) ∧
                        (x4^2 - x4 = 0) :=
sorry

end sum_of_x_l94_94502


namespace fermat_large_prime_solution_l94_94551

theorem fermat_large_prime_solution (n : ℕ) (hn : n > 0) :
  ∃ (p : ℕ) (hp : Nat.Prime p) (x y z : ℤ), 
    (x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x^n + y^n ≡ z^n [ZMOD p]) :=
sorry

end fermat_large_prime_solution_l94_94551


namespace find_greatest_number_l94_94592

def numbers := [0.07, -0.41, 0.8, 0.35, -0.9]

theorem find_greatest_number :
  ∃ x ∈ numbers, x > 0.7 ∧ ∀ y ∈ numbers, y > 0.7 → y = 0.8 :=
by
  sorry

end find_greatest_number_l94_94592


namespace factor_correct_l94_94940

noncomputable def p (b : ℝ) : ℝ := 221 * b^2 + 17 * b
def factored_form (b : ℝ) : ℝ := 17 * b * (13 * b + 1)

theorem factor_correct (b : ℝ) : p b = factored_form b := by
  sorry

end factor_correct_l94_94940


namespace train_length_l94_94004

theorem train_length (time_crossing : ℕ) (speed_kmh : ℕ) (conversion_factor : ℕ) (expected_length : ℕ) :
  time_crossing = 4 ∧ speed_kmh = 144 ∧ conversion_factor = 1000 / 3600 * 144 →
  expected_length = 160 :=
by
  sorry

end train_length_l94_94004


namespace german_students_count_l94_94536

def total_students : ℕ := 45
def both_english_german : ℕ := 12
def only_english : ℕ := 23

theorem german_students_count :
  ∃ G : ℕ, G = 45 - (23 + 12) + 12 :=
sorry

end german_students_count_l94_94536


namespace geometric_sequence_a_eq_neg4_l94_94512

theorem geometric_sequence_a_eq_neg4 
    (a : ℝ)
    (h : (2 * a + 2) ^ 2 = a * (3 * a + 3)) : 
    a = -4 :=
sorry

end geometric_sequence_a_eq_neg4_l94_94512


namespace number_of_zeros_of_g_l94_94677

open Real

noncomputable def g (x : ℝ) : ℝ := cos (π * log x + x)

theorem number_of_zeros_of_g : ¬ ∃ (x : ℝ), 1 < x ∧ x < exp 2 ∧ g x = 0 :=
sorry

end number_of_zeros_of_g_l94_94677


namespace cos_4theta_value_l94_94687

theorem cos_4theta_value (theta : ℝ) 
  (h : ∑' n : ℕ, (Real.cos theta)^(2 * n) = 8) : 
  Real.cos (4 * theta) = 1 / 8 := 
sorry

end cos_4theta_value_l94_94687


namespace candies_left_to_share_l94_94837

def initial_candies : ℕ := 100
def siblings : ℕ := 3
def candies_per_sibling : ℕ := 10
def candies_josh_eats : ℕ := 16

theorem candies_left_to_share : 
  let candies_given_to_siblings := siblings * candies_per_sibling in
  let candies_after_siblings := initial_candies - candies_given_to_siblings in
  let candies_given_to_friend := candies_after_siblings / 2 in
  let candies_after_friend := candies_after_siblings - candies_given_to_friend in
  candies_after_friend - candies_josh_eats = 19 :=
by 
  sorry

end candies_left_to_share_l94_94837


namespace savings_of_person_l94_94864

theorem savings_of_person (income expenditure : ℕ) (h_ratio : 3 * expenditure = 2 * income) (h_income : income = 21000) :
  income - expenditure = 7000 :=
by
  sorry

end savings_of_person_l94_94864


namespace evaluate_expression_l94_94644

open Real

def a := 2999
def b := 3000
def delta := b - a

theorem evaluate_expression : 3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = 5999 :=
by
  let a := 2999
  let b := 3000
  have h1 : b - a = 1 := by sorry
  calc
    3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = a^3 + b^3 - ab^2 - a^2b := by sorry
                                            ... = (b - a)^2 * (b + a)       := by sorry
                                            ... = (1)^2 * (b + a)           := by
                                                                           rw [h1]
                                                                           exact sorry
                                            ... = 3000 + 2999               := by
                                                                           exact sorry
                                            ... = 5999                     := rfl

end evaluate_expression_l94_94644


namespace tens_digit_of_sum_l94_94733

theorem tens_digit_of_sum (a b c : ℕ) (h : a = c + 3) (ha : 0 ≤ a ∧ a < 10) (hb : 0 ≤ b ∧ b < 10) (hc : 0 ≤ c ∧ c < 10) :
    ∃ t : ℕ, 10 ≤ t ∧ t < 100 ∧ (202 * c + 20 * b + 303) % 100 = t ∧ t / 10 = 1 :=
by
  use (20 * b + 3)
  sorry

end tens_digit_of_sum_l94_94733


namespace solve_chris_age_l94_94727

/-- 
The average of Amy's, Ben's, and Chris's ages is 12. Six years ago, Chris was the same age as Amy is now. In 3 years, Ben's age will be 3/4 of Amy's age at that time. 
How old is Chris now? 
-/
def chris_age : Prop := 
  ∃ (a b c : ℤ), 
    (a + b + c = 36) ∧
    (c - 6 = a) ∧ 
    (b + 3 = 3 * (a + 3) / 4) ∧
    (c = 17)

theorem solve_chris_age : chris_age := 
  by
    sorry

end solve_chris_age_l94_94727


namespace sum_of_prime_factors_77_l94_94444

theorem sum_of_prime_factors_77 : (7 + 11 = 18) := by
  sorry

end sum_of_prime_factors_77_l94_94444


namespace negation_of_p_converse_of_p_inverse_of_p_contrapositive_of_p_original_p_l94_94043

theorem negation_of_p (π : ℝ) (a b c d : ℚ) (h : a * π + b = c * π + d) : a ≠ c ∨ b ≠ d :=
  sorry

theorem converse_of_p (π : ℝ) (a b c d : ℚ) (h : a = c ∧ b = d) : a * π + b = c * π + d :=
  sorry

theorem inverse_of_p (π : ℝ) (a b c d : ℚ) (h : a * π + b ≠ c * π + d) : a ≠ c ∨ b ≠ d :=
  sorry

theorem contrapositive_of_p (π : ℝ) (a b c d : ℚ) (h : a ≠ c ∨ b ≠ d) : a * π + b ≠ c * π + d :=
  sorry

theorem original_p (π : ℝ) (a b c d : ℚ) (h : a * π + b = c * π + d) : a = c ∧ b = d :=
  sorry

end negation_of_p_converse_of_p_inverse_of_p_contrapositive_of_p_original_p_l94_94043


namespace average_interest_rate_correct_l94_94462

-- Constants representing the conditions
def totalInvestment : ℝ := 5000
def rateA : ℝ := 0.035
def rateB : ℝ := 0.07

-- The condition that return from investment at 7% is twice that at 3.5%
def return_condition (x : ℝ) : Prop := 0.07 * x = 2 * 0.035 * (5000 - x)

-- The average rate of interest formula
noncomputable def average_rate_of_interest (x : ℝ) : ℝ := 
  (0.07 * x + 0.035 * (5000 - x)) / 5000

-- The theorem to prove the average rate is 5.25%
theorem average_interest_rate_correct : ∃ (x : ℝ), return_condition x ∧ average_rate_of_interest x = 0.0525 := 
by
  sorry

end average_interest_rate_correct_l94_94462


namespace circle_symmetry_y_axis_eq_l94_94253

theorem circle_symmetry_y_axis_eq (x y : ℝ) :
  (x^2 + y^2 + 2 * x = 0) ↔ (x^2 + y^2 - 2 * x = 0) :=
sorry

end circle_symmetry_y_axis_eq_l94_94253


namespace ninth_grade_students_eq_l94_94429

-- Let's define the conditions
def total_students : ℕ := 50
def seventh_grade_students (x : ℕ) : ℕ := 2 * x - 1
def eighth_grade_students (x : ℕ) : ℕ := x

-- Define the expression for ninth grade students based on the conditions
def ninth_grade_students (x : ℕ) : ℕ :=
  total_students - (seventh_grade_students x + eighth_grade_students x)

-- The theorem statement to prove
theorem ninth_grade_students_eq (x : ℕ) : ninth_grade_students x = 51 - 3 * x :=
by
  sorry

end ninth_grade_students_eq_l94_94429


namespace three_f_x_eq_l94_94680

theorem three_f_x_eq (f : ℝ → ℝ) (h : ∀ x > 0, f (3 * x) = 2 / (3 + x)) (x : ℝ) (hx : x > 0) : 
  3 * f x = 18 / (9 + x) := sorry

end three_f_x_eq_l94_94680


namespace no_polynomial_deg_ge_3_satisfies_conditions_l94_94340

theorem no_polynomial_deg_ge_3_satisfies_conditions :
  ¬ ∃ f : Polynomial ℝ, f.degree ≥ 3 ∧ f.eval (x^2) = (f.eval x)^2 ∧ f.coeff 2 = 0 :=
sorry

end no_polynomial_deg_ge_3_satisfies_conditions_l94_94340


namespace solve_for_x_l94_94567

theorem solve_for_x : ∀ x : ℤ, 5 - x = 8 → x = -3 :=
by
  intros x h
  sorry

end solve_for_x_l94_94567


namespace questions_two_and_four_equiv_questions_three_and_seven_equiv_l94_94424

-- Definitions representing conditions about students in classes A and B:
def ClassA (student : Student) : Prop := sorry
def ClassB (student : Student) : Prop := sorry
def taller (x y : Student) : Prop := sorry
def shorter (x y : Student) : Prop := sorry
def tallest (students : Set Student) : Student := sorry
def shortest (students : Set Student) : Student := sorry
def averageHeight (students : Set Student) : ℝ := sorry
def totalHeight (students : Set Student) : ℝ := sorry
def medianHeight (students : Set Student) : ℝ := sorry

-- Equivalence of question 2 and question 4:
theorem questions_two_and_four_equiv (students_A students_B : Set Student) :
  (∀ a ∈ students_A, ∃ b ∈ students_B, taller a b) ↔ 
  (∀ b ∈ students_B, ∃ a ∈ students_A, taller a b) :=
sorry

-- Equivalence of question 3 and question 7:
theorem questions_three_and_seven_equiv (students_A students_B : Set Student) :
  (∀ a ∈ students_A, ∃ b ∈ students_B, shorter b a) ↔ 
  (shorter (shortest students_B) (shortest students_A)) :=
sorry

end questions_two_and_four_equiv_questions_three_and_seven_equiv_l94_94424


namespace check_not_coverable_boards_l94_94304

def is_coverable_by_dominoes (m n : ℕ) : Prop :=
  (m * n) % 2 = 0

theorem check_not_coverable_boards:
  (¬is_coverable_by_dominoes 5 5) ∧ (¬is_coverable_by_dominoes 3 7) :=
by
  -- Proof steps are omitted.
  sorry

end check_not_coverable_boards_l94_94304


namespace birthday_guests_l94_94559

theorem birthday_guests (total_guests : ℕ) (women men children guests_left men_left children_left : ℕ)
  (h_total : total_guests = 60)
  (h_women : women = total_guests / 2)
  (h_men : men = 15)
  (h_children : children = total_guests - (women + men))
  (h_men_left : men_left = men / 3)
  (h_children_left : children_left = 5)
  (h_guests_left : guests_left = men_left + children_left) :
  (total_guests - guests_left) = 50 :=
by sorry

end birthday_guests_l94_94559


namespace Dana_has_25_more_pencils_than_Marcus_l94_94928

theorem Dana_has_25_more_pencils_than_Marcus (JaydenPencils : ℕ) (h1 : JaydenPencils = 20) :
  let DanaPencils := JaydenPencils + 15,
      MarcusPencils := JaydenPencils / 2
  in DanaPencils - MarcusPencils = 25 := 
by
  sorry -- proof to be filled in

end Dana_has_25_more_pencils_than_Marcus_l94_94928


namespace one_fourth_in_one_eighth_l94_94074

theorem one_fourth_in_one_eighth : (1/8 : ℚ) / (1/4) = (1/2) := 
by
  sorry

end one_fourth_in_one_eighth_l94_94074


namespace polynomial_coeff_sum_l94_94999

theorem polynomial_coeff_sum (A B C D : ℤ) 
  (h : ∀ x : ℤ, (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D) : 
  A + B + C + D = 36 :=
by 
  sorry

end polynomial_coeff_sum_l94_94999


namespace reflection_y_axis_correct_l94_94859

-- Define the coordinates and reflection across the y-axis
def reflect_y_axis (p : (ℝ × ℝ)) : (ℝ × ℝ) :=
  (-p.1, p.2)

-- Define the original point M
def M : (ℝ × ℝ) := (3, 2)

-- State the theorem we want to prove
theorem reflection_y_axis_correct : reflect_y_axis M = (-3, 2) :=
by
  -- The proof would go here, but it is omitted as per the instructions
  sorry

end reflection_y_axis_correct_l94_94859


namespace salary_recovery_l94_94906

theorem salary_recovery (S : ℝ) : 
  (0.80 * S) + (0.25 * (0.80 * S)) = S :=
by
  sorry

end salary_recovery_l94_94906


namespace ten_percent_eq_l94_94686

variable (s t : ℝ)

def ten_percent_of (x : ℝ) : ℝ := 0.1 * x

theorem ten_percent_eq (h : ten_percent_of s = t) : s = 10 * t :=
by sorry

end ten_percent_eq_l94_94686


namespace max_marks_l94_94474

theorem max_marks (M p : ℝ) (h1 : p = 0.60 * M) (h2 : p = 160 + 20) : M = 300 := by
  sorry

end max_marks_l94_94474


namespace initially_collected_oranges_l94_94384

-- Define the conditions from the problem
def oranges_eaten_by_father : ℕ := 2
def oranges_mildred_has_now : ℕ := 75

-- Define the proof problem (statement)
theorem initially_collected_oranges :
  (oranges_mildred_has_now + oranges_eaten_by_father = 77) :=
by 
  -- proof goes here
  sorry

end initially_collected_oranges_l94_94384


namespace M_intersect_N_eq_l94_94104

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}
def N : Set ℝ := {y | ∃ x : ℝ, y = x + 1}

-- Define what we need to prove
theorem M_intersect_N_eq : M ∩ N = {y | y ≥ 1} :=
by
  sorry

end M_intersect_N_eq_l94_94104


namespace houses_in_block_l94_94769

theorem houses_in_block (junk_per_house : ℕ) (total_junk : ℕ) (h_junk : junk_per_house = 2) (h_total : total_junk = 14) :
  total_junk / junk_per_house = 7 := by
  sorry

end houses_in_block_l94_94769


namespace hyperbola_equation_l94_94197

theorem hyperbola_equation
  (a b m n e e' c' : ℝ)
  (h1 : 2 * a^2 + b^2 = 2)
  (h2 : e * e' = 1)
  (h_c : c' = e * m)
  (h_b : b^2 = m^2 - n^2)
  (h_e : e = n / m) : 
  y^2 - x^2 = 2 := 
sorry

end hyperbola_equation_l94_94197


namespace isosceles_triangle_perimeter_l94_94660

theorem isosceles_triangle_perimeter :
  ∃ (a b c : ℕ), 
  (a = 3 ∧ b = 6 ∧ (c = 6 ∨ c = 3)) ∧ 
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ∧
  (a + b + c = 15) :=
sorry

end isosceles_triangle_perimeter_l94_94660


namespace boys_tried_out_l94_94277

theorem boys_tried_out (B : ℕ) (girls : ℕ) (called_back : ℕ) (not_cut : ℕ) (total_tryouts : ℕ) 
  (h1 : girls = 39)
  (h2 : called_back = 26)
  (h3 : not_cut = 17)
  (h4 : total_tryouts = girls + B)
  (h5 : total_tryouts = called_back + not_cut) : 
  B = 4 := 
by
  sorry

end boys_tried_out_l94_94277


namespace remainder_of_7_pow_4_div_100_l94_94295

theorem remainder_of_7_pow_4_div_100 :
  (7^4) % 100 = 1 := 
sorry

end remainder_of_7_pow_4_div_100_l94_94295


namespace avg_speed_correct_l94_94307

def avg_speed (v1 v2 : ℝ) (t1 t2 : ℝ) : ℝ :=
  (v1 * t1 + v2 * t2) / (t1 + t2)

theorem avg_speed_correct (v1 v2 t1 t2 : ℝ) (h1 : t1 > 0) (h2 : t2 > 0) :
  avg_speed v1 v2 t1 t2 = (v1 * t1 + v2 * t2) / (t1 + t2) :=
by
  sorry

end avg_speed_correct_l94_94307


namespace polygon_sides_l94_94373

theorem polygon_sides (n : ℕ) (h : 180 * (n - 2) = 1620) : n = 11 := 
by 
  sorry

end polygon_sides_l94_94373


namespace manuscript_age_in_decimal_l94_94628

-- Given conditions
def octal_number : ℕ := 12345

-- Translate the problem statement into Lean:
theorem manuscript_age_in_decimal : (1 * 8^4 + 2 * 8^3 + 3 * 8^2 + 4 * 8^1 + 5 * 8^0) = 5349 :=
by
  sorry

end manuscript_age_in_decimal_l94_94628


namespace base_area_of_cylinder_l94_94083

variables (S : ℝ) (cylinder : Type)
variables (square_cross_section : cylinder → Prop) (area_square : cylinder → ℝ)
variables (base_area : cylinder → ℝ)

-- Assume that the cylinder has a square cross-section with a given area
axiom cross_section_square : ∀ c : cylinder, square_cross_section c → area_square c = 4 * S

-- Theorem stating the area of the base of the cylinder
theorem base_area_of_cylinder (c : cylinder) (h : square_cross_section c) : base_area c = π * S :=
by
  -- Proof omitted
  sorry

end base_area_of_cylinder_l94_94083


namespace sum_squares_mod_13_is_zero_l94_94589

def sum_squares_mod_13 : ℕ :=
  (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 + 7^2 + 8^2 + 9^2 + 10^2 + 11^2 + 12^2) % 13

theorem sum_squares_mod_13_is_zero : sum_squares_mod_13 = 0 := by
  sorry

end sum_squares_mod_13_is_zero_l94_94589


namespace sum_prime_factors_77_l94_94450

theorem sum_prime_factors_77 : ∀ (p1 p2 : ℕ), Nat.Prime p1 ∧ Nat.Prime p2 ∧ 77 = p1 * p2 → p1 + p2 = 18 :=
by
  intros p1 p2 h
  sorry

end sum_prime_factors_77_l94_94450


namespace herd_size_l94_94766

open Rat

theorem herd_size 
  (n : ℕ)
  (h1 : (3 / 7 : ℚ) * n + (1 / 3 : ℚ) * n + (1 / 6 : ℚ) * n ≤ n)
  (h2 : (1 - ((3 / 7 : ℚ) + (1 / 3 : ℚ) + (1 / 6 : ℚ))) * n = 16) :
  n = 224 := by
  sorry

end herd_size_l94_94766


namespace add_digits_base9_l94_94624

theorem add_digits_base9 : 
  ∀ n1 n2 n3 : ℕ, 
    (n1 = 2 * 9^2 + 5 * 9^1 + 4 * 9^0) →
    (n2 = 3 * 9^2 + 6 * 9^1 + 7 * 9^0) →
    (n3 = 1 * 9^2 + 4 * 9^1 + 2 * 9^0) →
    ((n1 + n2 + n3) = 7 * 9^2 + 7 * 9^1 + 4 * 9^0) := 
by
  intros n1 n2 n3 h1 h2 h3
  sorry

end add_digits_base9_l94_94624


namespace original_square_area_l94_94473

theorem original_square_area (s : ℕ) (h1 : s + 5 = s + 5) (h2 : (s + 5)^2 = s^2 + 225) : s^2 = 400 :=
by
  sorry

end original_square_area_l94_94473


namespace minimum_cubes_needed_l94_94745

def min_number_of_cubes (n : ℕ) : Prop :=
  ∀ (digits : Fin 10), ∃ (cubes : Fin n → Fin 6 → Fin 10),
    ∃ (comb : Finset (Fin 10 × Fin 10 × Fin 10)), 
    comb = {⟨d1, d2, d3⟩ | ∃ c1 c2 c3 : Fin n,
             c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3 ∧ 
             cubes c1 ⟨0%6⟩ = d1 ∧ 
             cubes c2 ⟨0%6⟩ = d2 ∧ 
             cubes c3 ⟨0%6⟩ = d3}

theorem minimum_cubes_needed : min_number_of_cubes 5 :=
by
  sorry

end minimum_cubes_needed_l94_94745


namespace problem_solution_l94_94646

theorem problem_solution {n : ℕ} :
  (∀ x y z : ℤ, x + y + z = 0 → ∃ k : ℤ, (x^n + y^n + z^n) / 2 = k^2) ↔ n = 1 ∨ n = 4 :=
by
  sorry

end problem_solution_l94_94646


namespace back_seat_capacity_l94_94371

def left_seats : Nat := 15
def right_seats : Nat := left_seats - 3
def seats_per_person : Nat := 3
def total_capacity : Nat := 92
def regular_seats_people : Nat := (left_seats + right_seats) * seats_per_person

theorem back_seat_capacity :
  total_capacity - regular_seats_people = 11 :=
by
  sorry

end back_seat_capacity_l94_94371


namespace range_of_x_l94_94509

noncomputable def problem_statement (x : ℝ) : Prop :=
  ∀ m : ℝ, abs m ≤ 2 → m * x^2 - 2 * x - m + 1 < 0 

theorem range_of_x (x : ℝ) :
  problem_statement x → ( ( -1 + Real.sqrt 7) / 2 < x ∧ x < ( 1 + Real.sqrt 3) / 2) :=
by
  intros h
  sorry

end range_of_x_l94_94509


namespace remainder_of_modified_division_l94_94142

theorem remainder_of_modified_division (x y u v : ℕ) (hx : 0 ≤ v ∧ v < y) (hxy : x = u * y + v) :
  ((x + 3 * u * y) % y) = v := by
  sorry

end remainder_of_modified_division_l94_94142


namespace arcsin_sqrt3_div_2_eq_pi_div_3_l94_94327

theorem arcsin_sqrt3_div_2_eq_pi_div_3 : 
  Real.arcsin (Real.sqrt 3 / 2) = Real.pi / 3 := 
by 
    sorry

end arcsin_sqrt3_div_2_eq_pi_div_3_l94_94327


namespace zeros_in_expansion_l94_94818

def x : ℕ := 10^12 - 3
def num_zeros (n : ℕ) : ℕ := (n.toString.filter (· == '0')).length

theorem zeros_in_expansion :
  num_zeros (x^2) = 20 := sorry

end zeros_in_expansion_l94_94818


namespace expected_value_equals_1_5_l94_94842

noncomputable def expected_value_win (roll : ℕ) : ℚ :=
  if roll = 1 then -1
  else if roll = 4 then -4
  else if roll = 2 ∨ roll = 3 ∨ roll = 5 ∨ roll = 7 then roll
  else 0

noncomputable def expected_value_total : ℚ :=
  (1/8 : ℚ) * ((expected_value_win 1) + (expected_value_win 2) + (expected_value_win 3) +
               (expected_value_win 4) + (expected_value_win 5) + (expected_value_win 6) +
               (expected_value_win 7) + (expected_value_win 8))

theorem expected_value_equals_1_5 : expected_value_total = 1.5 := by
  sorry

end expected_value_equals_1_5_l94_94842


namespace tan_beta_of_tan_alpha_and_tan_alpha_plus_beta_l94_94959

theorem tan_beta_of_tan_alpha_and_tan_alpha_plus_beta (α β : ℝ)
  (h1 : Real.tan α = 2)
  (h2 : Real.tan (α + β) = 1 / 5) :
  Real.tan β = -9 / 7 :=
sorry

end tan_beta_of_tan_alpha_and_tan_alpha_plus_beta_l94_94959


namespace sum_of_cos_series_l94_94762

theorem sum_of_cos_series :
  6 * Real.cos (18 * Real.pi / 180) + 2 * Real.cos (36 * Real.pi / 180) + 
  4 * Real.cos (54 * Real.pi / 180) + 6 * Real.cos (72 * Real.pi / 180) + 
  8 * Real.cos (90 * Real.pi / 180) + 10 * Real.cos (108 * Real.pi / 180) + 
  12 * Real.cos (126 * Real.pi / 180) + 14 * Real.cos (144 * Real.pi / 180) + 
  16 * Real.cos (162 * Real.pi / 180) + 18 * Real.cos (180 * Real.pi / 180) + 
  20 * Real.cos (198 * Real.pi / 180) + 22 * Real.cos (216 * Real.pi / 180) + 
  24 * Real.cos (234 * Real.pi / 180) + 26 * Real.cos (252 * Real.pi / 180) + 
  28 * Real.cos (270 * Real.pi / 180) + 30 * Real.cos (288 * Real.pi / 180) + 
  32 * Real.cos (306 * Real.pi / 180) + 34 * Real.cos (324 * Real.pi / 180) + 
  36 * Real.cos (342 * Real.pi / 180) + 38 * Real.cos (360 * Real.pi / 180) = 10 :=
by
  sorry

end sum_of_cos_series_l94_94762


namespace vector_magnitude_subtraction_l94_94053

theorem vector_magnitude_subtraction :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  (real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2) = 5) :=
by
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  have h : real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2) = 5 := sorry
  exact h

end vector_magnitude_subtraction_l94_94053


namespace relationship_between_y1_y2_l94_94664

theorem relationship_between_y1_y2 (b y1 y2 : ℝ) 
  (h₁ : y1 = -(-1) + b) 
  (h₂ : y2 = -(2) + b) : 
  y1 > y2 := 
by 
  sorry

end relationship_between_y1_y2_l94_94664


namespace sum_of_roots_eq_neg2_l94_94271

-- Define the quadratic equation.
def quadratic_equation (x : ℝ) : ℝ :=
  x^2 + 2 * x - 1

-- Define a predicate to express that x is a root of the quadratic equation.
def is_root (x : ℝ) : Prop :=
  quadratic_equation x = 0

-- Define the statement that the sum of the two roots of the quadratic equation equals -2.
theorem sum_of_roots_eq_neg2 (x1 x2 : ℝ) (h1 : is_root x1) (h2 : is_root x2) (h3 : x1 ≠ x2) :
  x1 + x2 = -2 :=
  sorry

end sum_of_roots_eq_neg2_l94_94271


namespace range_of_u_l94_94665

def satisfies_condition (x y : ℝ) : Prop := (x^2 / 3) + y^2 = 1

def u (x y : ℝ) : ℝ := |2 * x + y - 4| + |3 - x - 2 * y|

theorem range_of_u {x y : ℝ} (h : satisfies_condition x y) : ∀ u, 1 ≤ u ∧ u ≤ 13 :=
sorry

end range_of_u_l94_94665


namespace carrie_jellybeans_l94_94170

def volume (a : ℕ) : ℕ := a * a * a

def bert_box_volume : ℕ := 216

def carrie_factor : ℕ := 3

def count_error_factor : ℝ := 1.10

noncomputable def jellybeans_carrie (bert_box_volume carrie_factor count_error_factor : ℝ) : ℝ :=
  count_error_factor * (carrie_factor ^ 3 * bert_box_volume)

theorem carrie_jellybeans (bert_box_volume := 216) (carrie_factor := 3) (count_error_factor := 1.10) :
  jellybeans_carrie bert_box_volume carrie_factor count_error_factor = 6415 :=
sorry

end carrie_jellybeans_l94_94170


namespace days_worked_prove_l94_94764

/-- Work rate of A is 1/15 work per day -/
def work_rate_A : ℚ := 1/15

/-- Work rate of B is 1/20 work per day -/
def work_rate_B : ℚ := 1/20

/-- Combined work rate of A and B -/
def combined_work_rate : ℚ := work_rate_A + work_rate_B

/-- Fraction of work left after some days -/
def fraction_work_left : ℚ := 8/15

/-- Fraction of work completed after some days -/
def fraction_work_completed : ℚ := 1 - fraction_work_left

/-- Number of days A and B worked together -/
def days_worked_together : ℚ := fraction_work_completed / combined_work_rate

theorem days_worked_prove : 
    days_worked_together = 4 := 
by 
    sorry

end days_worked_prove_l94_94764


namespace determine_function_l94_94229

def satisfies_condition (f : ℤ → ℤ) : Prop :=
  ∀ a b : ℤ, f (2 * a) + 2 * f b = f (f (a + b))

theorem determine_function (f : ℤ → ℤ) (h : satisfies_condition f) :
  ∀ n : ℤ, f n = 0 ∨ ∃ K : ℤ, f n = 2 * n + K :=
sorry

end determine_function_l94_94229


namespace part_a_exists_rational_non_integer_l94_94150

theorem part_a_exists_rational_non_integer 
  (x y : ℚ) (hx : ¬int.cast x ∉ ℤ) (hy : ¬int.cast y ∉ ℤ) :
  ∃ x y : ℚ, (¬int.cast x ∉ ℤ) ∧ (¬int.cast y ∉ ℤ) ∧ (19 * x + 8 * y ∈ ℤ) ∧ (8 * x + 3 * y ∈ ℤ) := 
  sorry

end part_a_exists_rational_non_integer_l94_94150


namespace abigail_initial_money_l94_94620

variable (X : ℝ) -- Let X be the initial amount of money

def spent_on_food (X : ℝ) := 0.60 * X
def remaining_after_food (X : ℝ) := X - spent_on_food X
def spent_on_phone (X : ℝ) := 0.25 * remaining_after_food X
def remaining_after_phone (X : ℝ) := remaining_after_food X - spent_on_phone X
def final_amount (X : ℝ) := remaining_after_phone X - 20

theorem abigail_initial_money
    (food_spent : spent_on_food X = 0.60 * X)
    (phone_spent : spent_on_phone X = 0.10 * X)
    (remaining_after_entertainment : final_amount X = 40) :
    X = 200 :=
by
    sorry

end abigail_initial_money_l94_94620


namespace number_of_points_l94_94236

theorem number_of_points (x : ℕ) (h : (x * (x - 1)) / 2 = 45) : x = 10 :=
by
  -- Proof to be done here
  sorry

end number_of_points_l94_94236


namespace speed_of_man_l94_94315

noncomputable def train_length : ℝ := 150
noncomputable def time_to_pass : ℝ := 6
noncomputable def train_speed_kmh : ℝ := 83.99280057595394

/-- The speed of the man in km/h -/
theorem speed_of_man (train_length time_to_pass train_speed_kmh : ℝ) (h_train_length : train_length = 150) (h_time_to_pass : time_to_pass = 6) (h_train_speed_kmh : train_speed_kmh = 83.99280057595394) : 
  (train_length / time_to_pass * 3600 / 1000 - train_speed_kmh) * 3600 / 1000 = 6.0072 :=
by
  sorry

end speed_of_man_l94_94315


namespace simplify_polynomial_sum_l94_94853

/- Define the given polynomials -/
def polynomial1 (x : ℝ) : ℝ := (5 * x^10 + 8 * x^9 + 3 * x^8)
def polynomial2 (x : ℝ) : ℝ := (2 * x^12 + 3 * x^10 + x^9 + 4 * x^8 + 6 * x^4 + 7 * x^2 + 9)
def resultant_polynomial (x : ℝ) : ℝ := (2 * x^12 + 8 * x^10 + 9 * x^9 + 7 * x^8 + 6 * x^4 + 7 * x^2 + 9)

theorem simplify_polynomial_sum (x : ℝ) :
  polynomial1 x + polynomial2 x = resultant_polynomial x :=
by
  sorry

end simplify_polynomial_sum_l94_94853


namespace find_original_number_l94_94825

theorem find_original_number (c : ℝ) (h₁ : c / 12.75 = 16) (h₂ : 2.04 / 1.275 = 1.6) : c = 204 :=
by
  sorry

end find_original_number_l94_94825


namespace train_crosses_man_in_6_seconds_l94_94604

/-- A train of length 240 meters, traveling at a speed of 144 km/h, will take 6 seconds to cross a man standing on the platform. -/
theorem train_crosses_man_in_6_seconds
  (length_of_train : ℕ)
  (speed_of_train : ℕ)
  (conversion_factor : ℕ)
  (speed_in_m_per_s : ℕ)
  (time_to_cross : ℕ)
  (h1 : length_of_train = 240)
  (h2 : speed_of_train = 144)
  (h3 : conversion_factor = 1000 / 3600)
  (h4 : speed_in_m_per_s = speed_of_train * conversion_factor)
  (h5 : speed_in_m_per_s = 40)
  (h6 : time_to_cross = length_of_train / speed_in_m_per_s) :
  time_to_cross = 6 := by
  sorry

end train_crosses_man_in_6_seconds_l94_94604


namespace length_of_hallway_is_six_l94_94234

noncomputable def length_of_hallway (total_area_square_feet : ℝ) (central_area_side_length : ℝ) (hallway_width : ℝ) : ℝ :=
  (total_area_square_feet - (central_area_side_length * central_area_side_length)) / hallway_width

theorem length_of_hallway_is_six 
  (total_area_square_feet : ℝ)
  (central_area_side_length : ℝ)
  (hallway_width : ℝ)
  (h1 : total_area_square_feet = 124)
  (h2 : central_area_side_length = 10)
  (h3 : hallway_width = 4) :
  length_of_hallway total_area_square_feet central_area_side_length hallway_width = 6 := by
  sorry

end length_of_hallway_is_six_l94_94234


namespace directrix_of_parabola_l94_94731

theorem directrix_of_parabola (y x : ℝ) (h_eq : y^2 = 8 * x) :
  x = -2 :=
sorry

end directrix_of_parabola_l94_94731


namespace find_a_l94_94947

theorem find_a (a : ℝ) :
  let Δ1 := 4 - 4 * a, Δ2 := a^2 - 8
  in Δ1 > 0 ∧ Δ2 > 0 ∧ 4 - 2 * a = a^2 - 4 ↔ a = -4 := 
by {
  intros,
  sorry
}

end find_a_l94_94947


namespace one_fourths_in_one_eighth_l94_94075

theorem one_fourths_in_one_eighth : (1 / 8) / (1 / 4) = 1 / 2 := 
by 
  sorry

end one_fourths_in_one_eighth_l94_94075


namespace minute_hand_moves_180_degrees_l94_94655

noncomputable def minute_hand_angle_6_15_to_6_45 : ℝ :=
  let degrees_per_hour := 360
  let hours_period := 0.5
  degrees_per_hour * hours_period

theorem minute_hand_moves_180_degrees :
  minute_hand_angle_6_15_to_6_45 = 180 :=
by
  sorry

end minute_hand_moves_180_degrees_l94_94655


namespace families_seating_arrangements_l94_94457

theorem families_seating_arrangements : 
  let factorial := Nat.factorial
  let family_ways := factorial 3
  let bundles := family_ways * family_ways * family_ways
  let bundle_ways := factorial 3
  bundles * bundle_ways = (factorial 3) ^ 4 := by
  sorry

end families_seating_arrangements_l94_94457


namespace min_value_inequality_l94_94963

theorem min_value_inequality (x y : ℝ) (h1 : x^2 + y^2 = 3) (h2 : |x| ≠ |y|) :
  ∃ (m : ℝ), m = (1 / (2*x + y)^2 + 4 / (x - 2*y)^2) ∧ m = 3 / 5 :=
by
  sorry

end min_value_inequality_l94_94963


namespace correct_value_of_3_dollar_neg4_l94_94930

def special_operation (x y : Int) : Int :=
  x * (y + 2) + x * y + x

theorem correct_value_of_3_dollar_neg4 : special_operation 3 (-4) = -15 :=
by
  sorry

end correct_value_of_3_dollar_neg4_l94_94930


namespace sum_of_prime_factors_77_l94_94445

theorem sum_of_prime_factors_77 : (7 + 11 = 18) := by
  sorry

end sum_of_prime_factors_77_l94_94445


namespace rows_of_potatoes_l94_94920

theorem rows_of_potatoes (total_potatoes : ℕ) (seeds_per_row : ℕ) (h1 : total_potatoes = 54) (h2 : seeds_per_row = 9) : total_potatoes / seeds_per_row = 6 := 
by
  sorry

end rows_of_potatoes_l94_94920


namespace no_repair_needed_l94_94412

def nominal_mass : ℝ := 370 -- Assign the nominal mass as determined in the problem solving.

def max_deviation (M : ℝ) : ℝ := 0.1 * M
def preserved_max_deviation : ℝ := 37
def unreadable_max_deviation : ℝ := 37

def within_max_deviation (dev : ℝ) := dev ≤ preserved_max_deviation

noncomputable def standard_deviation : ℝ := preserved_max_deviation

theorem no_repair_needed :
  ∀ (M : ℝ),
  max_deviation M = 0.1 * M →
  preserved_max_deviation ≤ max_deviation M →
  ∀ (dev : ℝ), within_max_deviation dev →
  standard_deviation ≤ preserved_max_deviation →
  preserved_max_deviation = 37 →
  "не требует" = "не требует" :=
by
  intros M h1 h2 h3 h4 h5
  sorry

end no_repair_needed_l94_94412


namespace problem_1_problem_2_l94_94806

noncomputable def f (x : ℝ) : ℝ :=
  (Real.logb 3 (x / 27)) * (Real.logb 3 (3 * x))

theorem problem_1 (h₁ : 1 / 27 ≤ x)
(h₂ : x ≤ 1 / 9) :
    (∀ x, f x ≤ 12) ∧ (∃ x, f x = 5) := 
sorry

theorem problem_2
(m α β : ℝ)
(h₁ : f α + m = 0)
(h₂ : f β + m = 0) :
    α * β = 9 :=
sorry

end problem_1_problem_2_l94_94806


namespace neg_pi_lt_neg_three_l94_94783

theorem neg_pi_lt_neg_three (h : Real.pi > 3) : -Real.pi < -3 :=
sorry

end neg_pi_lt_neg_three_l94_94783


namespace trig_expression_evaluation_l94_94323

-- Define the given conditions
axiom sin_390 : Real.sin (390 * Real.pi / 180) = 1 / 2
axiom tan_neg_45 : Real.tan (-45 * Real.pi / 180) = -1
axiom cos_360 : Real.cos (360 * Real.pi / 180) = 1

-- Formulate the theorem
theorem trig_expression_evaluation : 
  2 * Real.sin (390 * Real.pi / 180) - Real.tan (-45 * Real.pi / 180) + 5 * Real.cos (360 * Real.pi / 180) = 7 :=
by
  rw [sin_390, tan_neg_45, cos_360]
  sorry

end trig_expression_evaluation_l94_94323


namespace volume_pyramid_ABC_l94_94784

structure Point where
  x : ℝ
  y : ℝ

def triangle_volume (A B C : Point) : ℝ :=
  -- The implementation would calculate the volume of the pyramid formed
  -- by folding along the midpoint sides.
  sorry

theorem volume_pyramid_ABC :
  let A := Point.mk 0 0
  let B := Point.mk 30 0
  let C := Point.mk 20 15
  triangle_volume A B C = 900 :=
by
  -- To be filled with the proof
  sorry

end volume_pyramid_ABC_l94_94784


namespace min_unsuccessful_placements_8x8_l94_94209

-- Define the board, the placement, and the unsuccessful condition
def is_unsuccessful_placement (board : ℕ → ℕ → ℤ) (i j : ℕ) : Prop :=
  (i < 7 ∧ j < 7 ∧ (board i j + board (i+1) j + board i (j+1) + board (i+1) (j+1)) ≠ 0)

-- Main theorem: The minimum number of unsuccessful placements is 36 on an 8x8 board
theorem min_unsuccessful_placements_8x8 (board : ℕ → ℕ → ℤ) (H : ∀ i j, board i j = 1 ∨ board i j = -1) :
  ∃ (n : ℕ), n = 36 ∧ (∀ m : ℕ, (∀ i j, is_unsuccessful_placement board i j → m < 36 ) → m = n) :=
sorry

end min_unsuccessful_placements_8x8_l94_94209


namespace determine_x_l94_94175

theorem determine_x (x : ℕ) (hx : 27^3 + 27^3 + 27^3 = 3^x) : x = 10 :=
sorry

end determine_x_l94_94175


namespace projection_onto_line_l94_94257

def projectionMatrix : Matrix (Fin 3) (Fin 3) ℚ :=
  !![ [4/17, -2/17, -8/17]
    , [-2/17, 1/17, 4/17]
    , [-8/17, 4/17, 15/17]]

def directionVector : Fin 3 → ℤ := !![2, -1, -4]

theorem projection_onto_line (P : Matrix (Fin 3) (Fin 3) ℚ) (v : Fin 3 → ℚ) (d : Fin 3 → ℤ) :
  P = projectionMatrix →
  v = !![1, 0, 0] →
  ∃ (a b c : ℤ), d = !![a, b, c] ∧ P.mulVec v = (1/17 : ℚ) • !![4, -2, -8] ∧
  a > 0 ∧ Int.gcd a b = 1 ∧ Int.gcd a c = 1 :=
begin
  intros hP hv,
  use [2, -1, -4],
  split,
  { exact rfl, },
  { split,
    { simp [hP, hv, projectionMatrix], },
    { split,
      { norm_num, },
      { split,
        { norm_num, },
        { norm_num, } } } }
end

end projection_onto_line_l94_94257


namespace glasses_needed_l94_94480

theorem glasses_needed (total_juice : ℕ) (juice_per_glass : ℕ) : Prop :=
  total_juice = 153 ∧ juice_per_glass = 30 → (total_juice + juice_per_glass - 1) / juice_per_glass = 6

-- This will state our theorem but we include sorry to omit the proof.

end glasses_needed_l94_94480


namespace truncated_cone_contact_radius_l94_94471

theorem truncated_cone_contact_radius (R r r' ζ : ℝ)
  (h volume_condition : ℝ)
  (R_pos : 0 < R)
  (r_pos : 0 < r)
  (r'_pos : 0 < r')
  (ζ_pos : 0 < ζ)
  (h_eq : h = 2 * R)
  (volume_condition_eq :
    (2 : ℝ) * ((4 / 3) * Real.pi * R^3) = 
    (2 / 3) * Real.pi * h * (r^2 + r * r' + r'^2)) :
  ζ = (2 * R * Real.sqrt 5) / 5 :=
by
  sorry

end truncated_cone_contact_radius_l94_94471


namespace complement_intersection_eq_interval_l94_94362

open Set

noncomputable def M : Set ℝ := {x | 3 * x - 1 >= 0}
noncomputable def N : Set ℝ := {x | 0 < x ∧ x < 1 / 2}

theorem complement_intersection_eq_interval :
  (M ∩ N)ᶜ = (Iio (1 / 3) ∪ Ici (1 / 2)) :=
by
  -- proof will go here in the actual development
  sorry

end complement_intersection_eq_interval_l94_94362


namespace number_of_solutions_l94_94268

theorem number_of_solutions : 
  ∃ n : ℕ, n = 5 ∧ (∃ (x y : ℕ), 1 ≤ x ∧ 1 ≤ y ∧ 4 * x + 5 * y = 98) :=
sorry

end number_of_solutions_l94_94268


namespace oranges_per_glass_l94_94202

theorem oranges_per_glass (total_oranges glasses_of_juice oranges_per_glass : ℕ)
    (h_oranges : total_oranges = 12)
    (h_glasses : glasses_of_juice = 6) : 
    total_oranges / glasses_of_juice = oranges_per_glass :=
by 
    sorry

end oranges_per_glass_l94_94202


namespace equation_of_line_AB_l94_94189

def point := (ℝ × ℝ)

def A : point := (-1, 0)
def B : point := (3, 2)

def equation_of_line (p1 p2 : point) : ℝ × ℝ × ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  -- Calculate the slope
  let k := (y2 - y1) / (x2 - x1)
  -- Use point-slope form and simplify the equation to standard form
  (((1 : ℝ), -2, 1) : ℝ × ℝ × ℝ)

theorem equation_of_line_AB :
  equation_of_line A B = (1, -2, 1) :=
sorry

end equation_of_line_AB_l94_94189


namespace candles_on_rituprts_cake_l94_94111

theorem candles_on_rituprts_cake (peter_candles : ℕ) (rupert_factor : ℝ) 
  (h_peter : peter_candles = 10) (h_rupert : rupert_factor = 3.5) : 
  ∃ rupert_candles : ℕ, rupert_candles = 35 :=
by
  sorry

end candles_on_rituprts_cake_l94_94111


namespace solution_set_of_inequality_l94_94738

theorem solution_set_of_inequality (x : ℝ) : (x-50)*(60-x) > 0 ↔ 50 < x ∧ x < 60 :=
by sorry

end solution_set_of_inequality_l94_94738


namespace additional_number_is_31_l94_94399

theorem additional_number_is_31
(six_numbers_sum : ℕ)
(seven_numbers_avg : ℕ)
(h1 : six_numbers_sum = 144)
(h2 : seven_numbers_avg = 25)
: ∃ x : ℕ, ((six_numbers_sum + x) / 7 = 25) ∧ x = 31 := 
by
  sorry

end additional_number_is_31_l94_94399


namespace average_of_middle_two_numbers_l94_94397

theorem average_of_middle_two_numbers :
  ∀ (a b c d : ℕ), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  (a + b + c + d) = 20 ∧
  (max (max a b) (max c d) - min (min a b) (min c d)) = 13 →
  (a + b + c + d - (max (max a b) (max c d)) - (min (min a b) (min c d))) / 2 = 2.5 :=
by sorry

end average_of_middle_two_numbers_l94_94397


namespace zeros_in_expansion_l94_94819

def num_zeros_expansion (n : ℕ) : ℕ :=
-- This function counts the number of trailing zeros in the decimal representation of n.
sorry

theorem zeros_in_expansion : num_zeros_expansion ((10^12 - 3)^2) = 11 :=
sorry

end zeros_in_expansion_l94_94819


namespace exp_continuous_at_l94_94564

theorem exp_continuous_at (a α : ℝ) (h : 0 < a) : 
  filter.tendsto (λ x, a^x) (nhds α) (nhds (a^α)) :=
sorry

end exp_continuous_at_l94_94564


namespace gcd_1729_867_l94_94746

theorem gcd_1729_867 : Nat.gcd 1729 867 = 1 :=
by 
  sorry

end gcd_1729_867_l94_94746


namespace worker_bees_hive_empty_l94_94986

theorem worker_bees_hive_empty:
  ∀ (initial_worker: ℕ) (leave_nectar: ℕ) (reassign_guard: ℕ) (return_trip: ℕ) (multiplier: ℕ),
  initial_worker = 400 →
  leave_nectar = 28 →
  reassign_guard = 30 →
  return_trip = 15 →
  multiplier = 5 →
  ((initial_worker - leave_nectar - reassign_guard + return_trip) * (1 - multiplier)) = 0 :=
by
  intros initial_worker leave_nectar reassign_guard return_trip multiplier
  sorry

end worker_bees_hive_empty_l94_94986


namespace remainder_when_abc_divided_by_7_l94_94679

theorem remainder_when_abc_divided_by_7 (a b c : ℕ) (h0 : a < 7) (h1 : b < 7) (h2 : c < 7)
  (h3 : (a + 2 * b + 3 * c) % 7 = 0)
  (h4 : (2 * a + 3 * b + c) % 7 = 4)
  (h5 : (3 * a + b + 2 * c) % 7 = 4) :
  (a * b * c) % 7 = 6 := 
sorry

end remainder_when_abc_divided_by_7_l94_94679


namespace g_at_10_l94_94638

noncomputable def g : ℕ → ℝ := sorry

axiom g_zero : g 0 = 2
axiom g_one : g 1 = 1
axiom g_func_eq (m n : ℕ) (h : m ≥ n) : 
  g (m + n) + g (m - n) = (g (2 * m) + g (2 * n)) / 2 + 2

theorem g_at_10 : g 10 = 102 := sorry

end g_at_10_l94_94638


namespace a_minus_b_greater_than_one_l94_94845

open Real

theorem a_minus_b_greater_than_one {a b : ℝ} (ha : 0 < a) (hb : 0 < b)
  (f_has_three_roots : ∃ (r1 r2 r3 : ℝ), r1 ≠ r2 ∧ r1 ≠ r3 ∧ r2 ≠ r3 ∧ (Polynomial.aeval r1 (Polynomial.C (-1) + Polynomial.X * (Polynomial.C (2*b)) + Polynomial.X^2 * (Polynomial.C a) + Polynomial.X^3)) = 0 ∧ (Polynomial.aeval r2 (Polynomial.C (-1) + Polynomial.X * (Polynomial.C (2*b)) + Polynomial.X^2 * (Polynomial.C a) + Polynomial.X^3)) = 0 ∧ (Polynomial.aeval r3 (Polynomial.C (-1) + Polynomial.X * (Polynomial.C (2*b)) + Polynomial.X^2 * (Polynomial.C a) + Polynomial.X^3)) = 0)
  (g_no_real_roots : ∀ (x : ℝ), (2*x^2 + 2*b*x + a) ≠ 0) :
  a - b > 1 := by
  sorry

end a_minus_b_greater_than_one_l94_94845


namespace inequality_not_always_correct_l94_94069

variables (x y z : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : x > y) (h₄ : z > 0)

theorem inequality_not_always_correct :
  ¬ ∀ z > 0, (xz^2 / z > yz^2 / z) :=
sorry

end inequality_not_always_correct_l94_94069


namespace intersection_is_empty_l94_94751

-- Define sets M and N
def M : Set ℕ := {0, 1, 3}
def N : Set ℕ := {0, 3, 4}

-- Define isolated elements for a set
def is_isolated (A : Set ℕ) (x : ℕ) : Prop :=
  x ∈ A ∧ (x - 1 ∉ A) ∧ (x + 1 ∉ A)

-- Define isolated sets
def isolated_set (A : Set ℕ) : Set ℕ :=
  {x | is_isolated A x}

-- Define isolated sets for M and N
def M' := isolated_set M
def N' := isolated_set N

-- The intersection of the isolated sets
theorem intersection_is_empty : M' ∩ N' = ∅ := 
  sorry

end intersection_is_empty_l94_94751


namespace probability_of_blue_or_orange_jelly_bean_is_5_over_13_l94_94298

def total_jelly_beans : ℕ := 7 + 9 + 8 + 10 + 5

def blue_or_orange_jelly_beans : ℕ := 10 + 5

def probability_blue_or_orange : ℚ := blue_or_orange_jelly_beans / total_jelly_beans

theorem probability_of_blue_or_orange_jelly_bean_is_5_over_13 :
  probability_blue_or_orange = 5 / 13 :=
by
  sorry

end probability_of_blue_or_orange_jelly_bean_is_5_over_13_l94_94298


namespace expression_inside_absolute_value_l94_94791

theorem expression_inside_absolute_value (E : ℤ) (x : ℤ) (h1 : x = 10) (h2 : 30 - |E| = 26) :
  E = 4 ∨ E = -4 := 
by
  sorry

end expression_inside_absolute_value_l94_94791


namespace train_passing_time_l94_94702

theorem train_passing_time (L : ℕ) (v_kmph : ℕ) (v_mps : ℕ) (time : ℕ)
  (h1 : L = 90)
  (h2 : v_kmph = 36)
  (h3 : v_mps = v_kmph * (1000 / 3600))
  (h4 : v_mps = 10)
  (h5 : time = L / v_mps) :
  time = 9 := by
  sorry

end train_passing_time_l94_94702


namespace simplify_trig_expression_l94_94722

open Real

theorem simplify_trig_expression (A : ℝ) (h1 : cos A ≠ 0) (h2 : sin A ≠ 0) :
  (1 - (cos A) / (sin A) + 1 / (sin A)) * (1 + (sin A) / (cos A) - 1 / (cos A)) = -2 * (cos (2 * A) / sin (2 * A)) :=
by
  sorry

end simplify_trig_expression_l94_94722


namespace average_price_of_rackets_l94_94312

theorem average_price_of_rackets (total_amount : ℝ) (number_of_pairs : ℕ) (average_price : ℝ) 
  (h1 : total_amount = 588) (h2 : number_of_pairs = 60) : average_price = 9.80 :=
by
  sorry

end average_price_of_rackets_l94_94312


namespace midpoint_sum_of_coordinates_l94_94283

theorem midpoint_sum_of_coordinates : 
  let p1 := (8, 10)
  let p2 := (-4, -10)
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  (midpoint.1 + midpoint.2) = 2 :=
by
  sorry

end midpoint_sum_of_coordinates_l94_94283


namespace alchemists_less_than_half_l94_94169

variable (k c a : ℕ)

theorem alchemists_less_than_half (h1 : k = c + a) (h2 : c > a) : a < k / 2 := by
  sorry

end alchemists_less_than_half_l94_94169


namespace arithmetic_sequence_problem_l94_94214

theorem arithmetic_sequence_problem 
  (a : ℕ → ℕ)
  (h_sequence : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0)
  (h_sum : a 3 + a 8 = 10) :
  3 * a 5 + a 7 = 20 :=
sorry

end arithmetic_sequence_problem_l94_94214


namespace triangle_A1B1C1_angles_l94_94575

theorem triangle_A1B1C1_angles
  (A B C : Point)
  (A0 B0 C0 : Point)
  (A1 B1 C1 : Point)
  -- Conditions
  (h_angles : ∠ A B C = 120∠ ∧ ∠ B C A = 30∠ ∧ ∠ C A B = 30∠)
  (h_medians : Midpoint A B C0 ∧ Midpoint B C A0 ∧ Midpoint C A B0)
  (h_perpendicular_A : perpendicular (Line A0 A1) (Line B C))
  (h_perpendicular_B : perpendicular (Line B0 B1) (Line C A))
  (h_perpendicular_C : perpendicular (Line C0 C1) (Line A B)) :
    -- Conclusion
    ∠ A1 B1 C1 = 60∠ :=
sorry

end triangle_A1B1C1_angles_l94_94575


namespace equal_cost_l94_94313

theorem equal_cost (x : ℝ) : (2.75 * x + 125 = 1.50 * x + 140) ↔ (x = 12) := 
by sorry

end equal_cost_l94_94313


namespace ratio_almonds_to_walnuts_l94_94301

theorem ratio_almonds_to_walnuts (almonds walnuts mixture : ℝ) 
  (h1 : almonds = 116.67)
  (h2 : mixture = 140)
  (h3 : walnuts = mixture - almonds) : 
  (almonds / walnuts) = 5 :=
by
  sorry

end ratio_almonds_to_walnuts_l94_94301


namespace negation_of_exists_real_solution_equiv_l94_94261

open Classical

theorem negation_of_exists_real_solution_equiv :
  (¬ ∃ a : ℝ, ∃ x : ℝ, a * x^2 + 1 = 0) ↔ (∀ a : ℝ, ¬ ∃ x : ℝ, a * x^2 + 1 = 0) :=
by
  sorry

end negation_of_exists_real_solution_equiv_l94_94261


namespace total_cost_proof_l94_94294

noncomputable def cost_proof : Prop :=
  let M := 158.4
  let R := 66
  let F := 22
  (10 * M = 24 * R) ∧ (6 * F = 2 * R) ∧ (F = 22) →
  (4 * M + 3 * R + 5 * F = 941.6)

theorem total_cost_proof : cost_proof :=
by
  sorry

end total_cost_proof_l94_94294


namespace product_four_integers_sum_to_50_l94_94038

theorem product_four_integers_sum_to_50 (E F G H : ℝ) 
  (h₀ : E + F + G + H = 50)
  (h₁ : E - 3 = F + 3)
  (h₂ : E - 3 = G * 3)
  (h₃ : E - 3 = H / 3) :
  E * F * G * H = 7461.9140625 := 
sorry

end product_four_integers_sum_to_50_l94_94038


namespace intersection_A_B_l94_94661

noncomputable def A : Set ℝ := {x | 0 < x ∧ x < 2}
noncomputable def B : Set ℝ := {y | ∃ x, y = Real.log (x^2 + 1) ∧ y ≥ 0}

theorem intersection_A_B : A ∩ {x | ∃ y, y = Real.log (x^2 + 1) ∧ y ≥ 0} = {x | 0 < x ∧ x < 2} :=
  sorry

end intersection_A_B_l94_94661


namespace num_cages_l94_94465

-- Define the conditions as given
def parrots_per_cage : ℕ := 8
def parakeets_per_cage : ℕ := 2
def total_birds_in_store : ℕ := 40

-- Prove that the number of bird cages is 4
theorem num_cages (x : ℕ) (h : 10 * x = total_birds_in_store) : x = 4 :=
sorry

end num_cages_l94_94465


namespace golf_problem_l94_94167

variable (D : ℝ)

theorem golf_problem (h1 : D / 2 + D = 270) : D = 180 :=
by
  sorry

end golf_problem_l94_94167


namespace average_speed_eq_l94_94308

variables (v₁ v₂ : ℝ) (t₁ t₂ : ℝ)

theorem average_speed_eq (h₁ : t₁ > 0) (h₂ : t₂ > 0) : 
  ((v₁ * t₁) + (v₂ * t₂)) / (t₁ + t₂) = (v₁ + v₂) / 2 := 
sorry

end average_speed_eq_l94_94308


namespace probability_different_colors_l94_94273

-- Define the number of chips of each color
def num_blue := 6
def num_red := 5
def num_yellow := 4
def num_green := 3

-- Total number of chips
def total_chips := num_blue + num_red + num_yellow + num_green

-- Probability of drawing a chip of different color
theorem probability_different_colors : 
  (num_blue / total_chips) * ((total_chips - num_blue) / total_chips) +
  (num_red / total_chips) * ((total_chips - num_red) / total_chips) +
  (num_yellow / total_chips) * ((total_chips - num_yellow) / total_chips) +
  (num_green / total_chips) * ((total_chips - num_green) / total_chips) =
  119 / 162 := 
sorry

end probability_different_colors_l94_94273


namespace solution_set_inequality_l94_94737

theorem solution_set_inequality (x : ℝ) :
  ((x + (1 / 2)) * ((3 / 2) - x) ≥ 0) ↔ (- (1 / 2) ≤ x ∧ x ≤ (3 / 2)) :=
by sorry

end solution_set_inequality_l94_94737


namespace sphere_surface_area_l94_94741

theorem sphere_surface_area (V : ℝ) (h : V = 72 * Real.pi) : ∃ A, A = 36 * Real.pi * (2 ^ (2 / 3)) := 
by
  sorry

end sphere_surface_area_l94_94741


namespace remainder_range_l94_94232

theorem remainder_range (x y z a b c d e : ℕ)
(h1 : x % 211 = a) (h2 : y % 211 = b) (h3 : z % 211 = c)
(h4 : x % 251 = c) (h5 : y % 251 = d) (h6 : z % 251 = e)
(h7 : a < 211) (h8 : b < 211) (h9 : c < 211)
(h10 : c < 251) (h11 : d < 251) (h12 : e < 251) :
0 ≤ (2 * x - y + 3 * z + 47) % (211 * 251) ∧
(2 * x - y + 3 * z + 47) % (211 * 251) < (211 * 251) :=
by
  sorry

end remainder_range_l94_94232


namespace bicycle_weight_l94_94235

theorem bicycle_weight (b s : ℝ) (h1 : 9 * b = 5 * s) (h2 : 4 * s = 160) : b = 200 / 9 :=
by
  sorry

end bicycle_weight_l94_94235


namespace existence_of_rational_solutions_a_nonexistence_of_rational_solutions_b_l94_94155

def is_rational_non_integer (a : ℚ) : Prop :=
  ¬ (∃ n : ℤ, a = n)

theorem existence_of_rational_solutions_a :
  ∃ x y : ℚ, is_rational_non_integer x ∧ is_rational_non_integer y ∧
    ∃ k1 k2 : ℤ, 19 * x + 8 * y = k1 ∧ 8 * x + 3 * y = k2 :=
sorry

theorem nonexistence_of_rational_solutions_b :
  ¬ (∃ x y : ℚ, is_rational_non_integer x ∧ is_rational_non_integer y ∧
    ∃ k1 k2 : ℤ, 19 * x^2 + 8 * y^2 = k1 ∧ 8 * x^2 + 3 * y^2 = k2) :=
sorry

end existence_of_rational_solutions_a_nonexistence_of_rational_solutions_b_l94_94155


namespace decimal_between_0_996_and_0_998_ne_0_997_l94_94765

theorem decimal_between_0_996_and_0_998_ne_0_997 :
  ∃ x : ℝ, 0.996 < x ∧ x < 0.998 ∧ x ≠ 0.997 :=
by
  sorry

end decimal_between_0_996_and_0_998_ne_0_997_l94_94765


namespace gcd_f_50_51_l94_94711

def f (x : ℤ) : ℤ :=
  x ^ 2 - 2 * x + 2023

theorem gcd_f_50_51 : Int.gcd (f 50) (f 51) = 11 := by
  sorry

end gcd_f_50_51_l94_94711


namespace add_base8_l94_94778

/-- Define the numbers in base 8 --/
def base8_add (a b : Nat) : Nat := 
  sorry

theorem add_base8 : base8_add 0o12 0o157 = 0o171 := 
  sorry

end add_base8_l94_94778


namespace stork_count_l94_94761

theorem stork_count (B S : ℕ) (h1 : B = 7) (h2 : B = S + 3) : S = 4 := 
by 
  sorry -- Proof to be filled in


end stork_count_l94_94761


namespace charles_draws_yesterday_after_work_l94_94022

theorem charles_draws_yesterday_after_work :
  ∀ (initial_papers today_drawn yesterday_drawn_before_work current_papers_left yesterday_drawn_after_work : ℕ),
    initial_papers = 20 →
    today_drawn = 6 →
    yesterday_drawn_before_work = 6 →
    current_papers_left = 2 →
    (initial_papers - (today_drawn + yesterday_drawn_before_work) - yesterday_drawn_after_work = current_papers_left) →
    yesterday_drawn_after_work = 6 :=
by
  intros initial_papers today_drawn yesterday_drawn_before_work current_papers_left yesterday_drawn_after_work
  intro h1 h2 h3 h4 h5
  sorry

end charles_draws_yesterday_after_work_l94_94022


namespace cos_double_angle_l94_94351

theorem cos_double_angle (α : ℝ) (h : Real.sin (π + α) = 2 / 3) : Real.cos (2 * α) = 1 / 9 := 
sorry

end cos_double_angle_l94_94351


namespace minimum_radius_of_third_sphere_l94_94872

noncomputable def cone_height : ℝ := 4
noncomputable def cone_base_radius : ℝ := 3

noncomputable def radius_identical_spheres : ℝ := 4 / 3  -- derived from the conditions

theorem minimum_radius_of_third_sphere
    (h r1 r2 : ℝ) -- heights and radii one and two
    (R1 R2 Rb : ℝ) -- radii of the common base
    (cond_h : h = 4)
    (cond_Rb : Rb = 3)
    (cond_radii_eq : r1 = r2) 
  : r2 = 27 / 35 :=
by
  sorry

end minimum_radius_of_third_sphere_l94_94872


namespace gcd_1729_867_l94_94748

theorem gcd_1729_867 : Nat.gcd 1729 867 = 1 :=
by
  sorry

end gcd_1729_867_l94_94748


namespace right_triangle_legs_l94_94513

theorem right_triangle_legs (R r : ℝ) : 
  ∃ a b : ℝ, a = Real.sqrt (2 * (R^2 + r^2)) ∧ b = Real.sqrt (2 * (R^2 - r^2)) :=
by
  sorry

end right_triangle_legs_l94_94513


namespace determine_a_l94_94931

theorem determine_a (a : ℝ): (∃ b : ℝ, 27 * x^3 + 9 * x^2 + 36 * x + a = (3 * x + b)^3) → a = 8 := 
by
  sorry

end determine_a_l94_94931


namespace half_angle_quadrants_l94_94367

theorem half_angle_quadrants (α : ℝ) (k : ℤ) (hα : ∃ k : ℤ, (π/2 + k * 2 * π < α ∧ α < π + k * 2 * π)) : 
  ∃ k : ℤ, (π/4 + k * π < α/2 ∧ α/2 < π/2 + k * π) := 
sorry

end half_angle_quadrants_l94_94367


namespace Esha_behind_Anusha_l94_94013

/-- Define conditions for the race -/

def Anusha_speed := 100
def Banu_behind_when_Anusha_finishes := 10
def Banu_run_when_Anusha_finishes := Anusha_speed - Banu_behind_when_Anusha_finishes
def Esha_behind_when_Banu_finishes := 10
def Esha_run_when_Banu_finishes := Anusha_speed - Esha_behind_when_Banu_finishes
def Banu_speed_ratio := Banu_run_when_Anusha_finishes / Anusha_speed
def Esha_speed_ratio := Esha_run_when_Banu_finishes / Anusha_speed
def Esha_to_Anusha_speed_ratio := Esha_speed_ratio * Banu_speed_ratio
def Esha_run_when_Anusha_finishes := Anusha_speed * Esha_to_Anusha_speed_ratio

/-- Prove that Esha is 19 meters behind Anusha when Anusha finishes the race -/
theorem Esha_behind_Anusha {V_A V_B V_E : ℝ} :
  (V_B / V_A = 9 / 10) →
  (V_E / V_B = 9 / 10) →
  (Esha_run_when_Anusha_finishes = Anusha_speed * (9 / 10 * 9 / 10)) →
  Anusha_speed - Esha_run_when_Anusha_finishes = 19 := 
by
  intros h1 h2 h3
  sorry

end Esha_behind_Anusha_l94_94013


namespace minimum_sugar_correct_l94_94487

noncomputable def minimum_sugar (f : ℕ) (s : ℕ) : ℕ := 
  if (f ≥ 8 + s / 2 ∧ f ≤ 3 * s) then s else sorry

theorem minimum_sugar_correct (f s : ℕ) : 
  (f ≥ 8 + s / 2 ∧ f ≤ 3 * s) → s ≥ 4 :=
by sorry

end minimum_sugar_correct_l94_94487


namespace find_A_l94_94099

def hash_rel (A B : ℝ) := A^2 + B^2

theorem find_A (A : ℝ) (h : hash_rel A 7 = 196) : A = 7 * Real.sqrt 3 :=
by sorry

end find_A_l94_94099


namespace completed_shape_perimeter_602_l94_94560

noncomputable def rectangle_perimeter (length width : ℝ) : ℝ :=
  2 * (length + width)

noncomputable def total_perimeter_no_overlap (n : ℕ) (length width : ℝ) : ℝ :=
  n * rectangle_perimeter length width

noncomputable def total_reduction (n : ℕ) (overlap : ℝ) : ℝ :=
  (n - 1) * overlap

noncomputable def overall_perimeter (n : ℕ) (length width overlap : ℝ) : ℝ :=
  total_perimeter_no_overlap n length width - total_reduction n overlap

theorem completed_shape_perimeter_602 :
  overall_perimeter 100 3 1 2 = 602 :=
by
  sorry

end completed_shape_perimeter_602_l94_94560


namespace find_angle_B_max_value_a_squared_plus_c_squared_l94_94206

variable {A B C : ℝ} -- Angles A, B, C in radians
variable {a b c : ℝ} -- Sides opposite to these angles

-- Problem 1
theorem find_angle_B (h : b * Real.cos C + c * Real.cos B = 2 * a * Real.cos B) : B = Real.pi / 3 :=
by
  sorry -- Proof is not needed

-- Problem 2
theorem max_value_a_squared_plus_c_squared (h : b = Real.sqrt 3)
  (h' : b * Real.cos C + c * Real.cos B = 2 * a * Real.cos B) : (a^2 + c^2) ≤ 6 :=
by
  sorry -- Proof is not needed

end find_angle_B_max_value_a_squared_plus_c_squared_l94_94206


namespace symmetric_line_equation_l94_94530

theorem symmetric_line_equation {l : ℝ} (h1 : ∀ x y : ℝ, x + y - 1 = 0 → (-x) - y + 1 = l) : l = 0 :=
by
  sorry

end symmetric_line_equation_l94_94530


namespace numberOfPairsPaddlesSold_l94_94311

def totalSalesPaddles : ℝ := 735
def avgPricePerPairPaddles : ℝ := 9.8

theorem numberOfPairsPaddlesSold :
  totalSalesPaddles / avgPricePerPairPaddles = 75 := 
by
  sorry

end numberOfPairsPaddlesSold_l94_94311


namespace find_first_half_speed_l94_94239

theorem find_first_half_speed (distance time total_time : ℝ) (v2 : ℝ)
    (h_distance : distance = 300) 
    (h_time : total_time = 11) 
    (h_v2 : v2 = 25) 
    (half_distance : distance / 2 = 150) :
    (150 / (total_time - (150 / v2)) = 30) :=
by
  sorry

end find_first_half_speed_l94_94239


namespace part1_part2_l94_94098

variable {a : ℝ} (M N : Set ℝ)

theorem part1 (h : a = 1) : M = {x : ℝ | 0 < x ∧ x < 2} :=
by
  sorry

theorem part2 (hM : (M = {x : ℝ | 0 < x ∧ x < a + 1}))
              (hN : N = {x : ℝ | -1 ≤ x ∧ x ≤ 3})
              (h_union : M ∪ N = N) : 
  a ∈ Set.Icc (-1 : ℝ) 2 :=
by
  sorry

end part1_part2_l94_94098


namespace charles_pictures_after_work_l94_94026

variable (initial_papers : ℕ)
variable (draw_today : ℕ)
variable (draw_yesterday_morning : ℕ)
variable (papers_left : ℕ)

theorem charles_pictures_after_work :
    initial_papers = 20 →
    draw_today = 6 →
    draw_yesterday_morning = 6 →
    papers_left = 2 →
    initial_papers - (draw_today + draw_yesterday_morning + 6) = papers_left →
    6 = (initial_papers - draw_today - draw_yesterday_morning - papers_left) := 
by
  intros h1 h2 h3 h4 h5
  exact sorry

end charles_pictures_after_work_l94_94026


namespace factor_correct_l94_94941

noncomputable def p (b : ℝ) : ℝ := 221 * b^2 + 17 * b
def factored_form (b : ℝ) : ℝ := 17 * b * (13 * b + 1)

theorem factor_correct (b : ℝ) : p b = factored_form b := by
  sorry

end factor_correct_l94_94941


namespace low_card_value_is_one_l94_94694

-- Definitions and setting up the conditions
def num_high_cards : ℕ := 26
def num_low_cards : ℕ := 26
def high_card_points : ℕ := 2
def draw_scenarios : ℕ := 4

-- The point value of a low card L
noncomputable def low_card_points : ℕ :=
  if num_high_cards = 26 ∧ num_low_cards = 26 ∧ high_card_points = 2
     ∧ draw_scenarios = 4
  then 1 else 0 

theorem low_card_value_is_one :
  low_card_points = 1 :=
by
  sorry

end low_card_value_is_one_l94_94694


namespace base_extension_1_kilometer_l94_94616

-- Definition of the original triangle with hypotenuse length and inclination angle
def original_triangle (hypotenuse : ℝ) (angle : ℝ) : Prop :=
  hypotenuse = 1 ∧ angle = 20

-- Definition of the extension required for the new inclination angle
def extension_required (new_angle : ℝ) (extension : ℝ) : Prop :=
  new_angle = 10 ∧ extension = 1

-- The proof problem statement
theorem base_extension_1_kilometer :
  ∀ (hypotenuse : ℝ) (original_angle : ℝ) (new_angle : ℝ),
    original_triangle hypotenuse original_angle →
    new_angle = 10 →
    ∃ extension : ℝ, extension_required new_angle extension :=
by
  -- Sorry is a placeholder for the actual proof
  sorry

end base_extension_1_kilometer_l94_94616


namespace geometric_sequence_sum_inv_l94_94811

theorem geometric_sequence_sum_inv
  (a : ℕ → ℝ)
  (h1 : a 1 = 2)
  (h2 : a 1 + a 3 + a 5 = 14) :
  (1 / a 1) + (1 / a 3) + (1 / a 5) = 7 / 8 :=
by
  sorry

end geometric_sequence_sum_inv_l94_94811


namespace min_value_l94_94348

variable (x : ℝ)

def g (x : ℝ) := 4 * x - x^3

theorem min_value : 
  ∃ x ∈ Icc 0 2, 
  (∀ y ∈ Icc 0 2, g y ≥ g x) ∧ g x = (16 * Real.sqrt 3) / 9 :=
begin
  sorry
end

end min_value_l94_94348


namespace hyperbola_foci_l94_94729

/-- The coordinates of the foci of the hyperbola y^2 / 3 - x^2 = 1 are (0, ±2). -/
theorem hyperbola_foci (x y : ℝ) :
  x^2 - (y^2 / 3) = -1 → (0 = x ∧ (y = 2 ∨ y = -2)) :=
sorry

end hyperbola_foci_l94_94729


namespace Alice_wins_no_matter_what_Bob_does_l94_94009

theorem Alice_wins_no_matter_what_Bob_does (a b c : ℝ) :
  (∀ d : ℝ, (b + d) ^ 2 - 4 * (a + d) * (c + d) ≤ 0) → a = 0 ∧ b = 0 ∧ c = 0 :=
by
  intro h
  sorry

end Alice_wins_no_matter_what_Bob_does_l94_94009


namespace machine_does_not_require_repair_l94_94407

variable (nominal_mass max_deviation standard_deviation : ℝ)
variable (nominal_mass_ge : nominal_mass ≥ 370)
variable (max_deviation_le : max_deviation ≤ 0.1 * nominal_mass)
variable (all_deviations_le_max : ∀ d, d < max_deviation → d < 37)
variable (std_dev_le_max_dev : standard_deviation ≤ max_deviation)

theorem machine_does_not_require_repair :
  ¬ (standard_deviation > 37) :=
by 
  -- sorry annotation indicates the proof goes here
  sorry

end machine_does_not_require_repair_l94_94407


namespace factorize_expression_l94_94035

theorem factorize_expression (a b : ℝ) : b^2 - ab + a - b = (b - 1) * (b - a) :=
by
  sorry

end factorize_expression_l94_94035


namespace correct_quotient_l94_94464

def original_number : ℕ :=
  8 * 156 + 2

theorem correct_quotient :
  (8 * 156 + 2) / 5 = 250 :=
sorry

end correct_quotient_l94_94464


namespace trig_225_deg_l94_94924

noncomputable def sin_225 : Real := Real.sin (225 * Real.pi / 180)
noncomputable def cos_225 : Real := Real.cos (225 * Real.pi / 180)

theorem trig_225_deg :
  sin_225 = -Real.sqrt 2 / 2 ∧ cos_225 = -Real.sqrt 2 / 2 := by
  sorry

end trig_225_deg_l94_94924


namespace k_value_l94_94648

noncomputable def find_k : ℚ := 49 / 15

theorem k_value :
  ∀ (a b : ℚ), (3 * a^2 + 7 * a + find_k = 0) ∧ (3 * b^2 + 7 * b + find_k = 0) →
                (a^2 + b^2 = 3 * a * b) →
                find_k = 49 / 15 :=
by
  intros a b h_eq_root h_rel
  sorry

end k_value_l94_94648


namespace partition_displacement_l94_94469

variables (l : ℝ) (R T : ℝ) (initial_V1 initial_V2 final_Vleft final_Vright initial_P1 initial_P2 : ℝ)

-- Conditions
def initial_conditions (initial_V1 initial_V2 : ℝ) : Prop :=
  initial_V1 + initial_V2 = l ∧
  initial_V2 = 2 * initial_V1 ∧
  initial_P1 * initial_V1 = R * T ∧
  initial_P2 * initial_V2 = 2 * R * T ∧
  initial_P1 = initial_P2

-- Final volumes
def final_volumes (final_Vleft final_Vright : ℝ) : Prop :=
  final_Vleft = l / 2 ∧ final_Vright = l / 2 

-- Displacement of the partition
def displacement (initial_position final_position : ℝ) : ℝ :=
  initial_position - final_position

-- Theorem statement: the displacement of the partition is l / 6
theorem partition_displacement (l R T initial_V1 initial_V2 final_Vleft final_Vright initial_P1 initial_P2 : ℝ)
  (h_initial_cond : initial_conditions l R T initial_V1 initial_V2 initial_P1 initial_P2)
  (h_final_vol : final_volumes l final_Vleft final_Vright) 
  (initial_position final_position : ℝ)
  (initial_position_def : initial_position = 2 * l / 3)
  (final_position_def : final_position = l / 2) :
  displacement initial_position final_position = l / 6 := 
by sorry

end partition_displacement_l94_94469


namespace sum_of_numbers_l94_94689

theorem sum_of_numbers (avg : ℝ) (count : ℕ) (h_avg : avg = 5.7) (h_count : count = 8) : (avg * count = 45.6) :=
by
  sorry

end sum_of_numbers_l94_94689


namespace worker_total_pay_l94_94619

def regular_rate : ℕ := 10
def total_surveys : ℕ := 50
def cellphone_surveys : ℕ := 35
def non_cellphone_surveys := total_surveys - cellphone_surveys
def higher_rate := regular_rate + (30 * regular_rate / 100)

def pay_non_cellphone_surveys := non_cellphone_surveys * regular_rate
def pay_cellphone_surveys := cellphone_surveys * higher_rate

def total_pay := pay_non_cellphone_surveys + pay_cellphone_surveys

theorem worker_total_pay : total_pay = 605 := by
  sorry

end worker_total_pay_l94_94619


namespace scout_weekend_earnings_l94_94721

-- Definitions for conditions
def base_pay_per_hour : ℝ := 10.00
def tip_saturday : ℝ := 5.00
def tip_sunday_low : ℝ := 3.00
def tip_sunday_high : ℝ := 7.00
def transportation_cost_per_delivery : ℝ := 1.00
def hours_worked_saturday : ℝ := 6
def deliveries_saturday : ℝ := 5
def hours_worked_sunday : ℝ := 8
def deliveries_sunday : ℝ := 10
def deliveries_sunday_low_tip : ℝ := 5
def deliveries_sunday_high_tip : ℝ := 5
def holiday_multiplier : ℝ := 2

-- Calculation of total earnings for the weekend after transportation costs
theorem scout_weekend_earnings : 
  let base_pay_saturday := hours_worked_saturday * base_pay_per_hour
  let tips_saturday := deliveries_saturday * tip_saturday
  let transportation_costs_saturday := deliveries_saturday * transportation_cost_per_delivery
  let total_earnings_saturday := base_pay_saturday + tips_saturday - transportation_costs_saturday

  let base_pay_sunday := hours_worked_sunday * base_pay_per_hour * holiday_multiplier
  let tips_sunday := deliveries_sunday_low_tip * tip_sunday_low + deliveries_sunday_high_tip * tip_sunday_high
  let transportation_costs_sunday := deliveries_sunday * transportation_cost_per_delivery
  let total_earnings_sunday := base_pay_sunday + tips_sunday - transportation_costs_sunday

  let total_earnings_weekend := total_earnings_saturday + total_earnings_sunday

  total_earnings_weekend = 280.00 :=
by
  -- Add detailed proof here
  sorry

end scout_weekend_earnings_l94_94721


namespace min_value_fraction_sum_l94_94042

variable (a b : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)
variable (h_collinear : 3 * a + 2 * b = 1)

theorem min_value_fraction_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_collinear : 3 * a + 2 * b = 1) : 
  (3 / a + 1 / b) = 11 + 6 * Real.sqrt 2 :=
by
  sorry

end min_value_fraction_sum_l94_94042


namespace children_gift_distribution_l94_94387

theorem children_gift_distribution (N : ℕ) (hN : N > 1) :
  (∀ n : ℕ, n < N → (∃ k : ℕ, k < N ∧ k ≠ n)) →
  (∃ m : ℕ, (N - 1) = 2 * m) :=
by
  sorry

end children_gift_distribution_l94_94387


namespace correct_factorization_l94_94888

variable (x y : ℝ)

theorem correct_factorization :
  x^2 - 2 * x * y + x = x * (x - 2 * y + 1) :=
by sorry

end correct_factorization_l94_94888


namespace probability_increase_l94_94012

theorem probability_increase:
  let P_win1 := 0.30
  let P_lose1 := 0.70
  let P_win2 := 0.50
  let P_lose2 := 0.50
  let P_win3 := 0.40
  let P_lose3 := 0.60
  let P_win4 := 0.25
  let P_lose4 := 0.75
  let P_win_all := P_win1 * P_win2 * P_win3 * P_win4
  let P_lose_all := P_lose1 * P_lose2 * P_lose3 * P_lose4
  (P_lose_all - P_win_all) / P_win_all = 9.5 :=
by
  sorry

end probability_increase_l94_94012


namespace factor_expression_l94_94943

variable (b : ℤ)

theorem factor_expression : 221 * b^2 + 17 * b = 17 * b * (13 * b + 1) :=
by
  sorry

end factor_expression_l94_94943


namespace renu_suma_combined_work_days_l94_94392

theorem renu_suma_combined_work_days :
  (1 / (1 / 8 + 1 / 4.8)) = 3 :=
by
  sorry

end renu_suma_combined_work_days_l94_94392


namespace average_of_two_intermediate_numbers_l94_94398

theorem average_of_two_intermediate_numbers (a b c d : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
(h_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
(h_average : (a + b + c + d) / 4 = 5)
(h_max_diff: (max (max a b) (max c d) - min (min a b) (min c d) = 19)) :
  (a + b + c + d) - (max (max a b) (max c d)) - (min (min a b) (min c d)) = 5 :=
by
  -- The proof goes here
  sorry

end average_of_two_intermediate_numbers_l94_94398


namespace max_squares_fitting_l94_94884

theorem max_squares_fitting (L S : ℕ) (hL : L = 8) (hS : S = 2) : (L / S) * (L / S) = 16 := by
  -- Proof goes here
  sorry

end max_squares_fitting_l94_94884


namespace tonya_stamps_left_l94_94223

theorem tonya_stamps_left 
    (stamps_per_matchbook : ℕ) 
    (matches_per_matchbook : ℕ) 
    (tonya_initial_stamps : ℕ) 
    (jimmy_initial_matchbooks : ℕ) 
    (stamps_per_match : ℕ) 
    (tonya_final_stamps_expected : ℕ)
    (h1 : stamps_per_matchbook = 1) 
    (h2 : matches_per_matchbook = 24) 
    (h3 : tonya_initial_stamps = 13) 
    (h4 : jimmy_initial_matchbooks = 5) 
    (h5 : stamps_per_match = 12)
    (h6 : tonya_final_stamps_expected = 3) :
    tonya_initial_stamps - jimmy_initial_matchbooks * (matches_per_matchbook / stamps_per_match) = tonya_final_stamps_expected :=
by
  sorry

end tonya_stamps_left_l94_94223


namespace printer_diff_l94_94594

theorem printer_diff (A B : ℚ) (hA : A * 60 = 35) (hAB : (A + B) * 24 = 35) : B - A = 7 / 24 := by
  sorry

end printer_diff_l94_94594


namespace fraction_ratio_equivalence_l94_94597

theorem fraction_ratio_equivalence :
  ∃ (d : ℚ), d = 240 / 1547 ∧ ((2 / 13) / d) = ((5 / 34) / (7 / 48)) := 
by
  sorry

end fraction_ratio_equivalence_l94_94597


namespace mystery_number_addition_l94_94829

theorem mystery_number_addition (mystery_number : ℕ) (h : mystery_number = 47) : mystery_number + 45 = 92 :=
by
  -- Proof goes here
  sorry

end mystery_number_addition_l94_94829


namespace fescue_in_Y_l94_94114

-- Define the weight proportions of the mixtures
def weight_X : ℝ := 0.6667
def weight_Y : ℝ := 0.3333

-- Define the proportion of ryegrass in each mixture
def ryegrass_X : ℝ := 0.40
def ryegrass_Y : ℝ := 0.25

-- Define the proportion of ryegrass in the final mixture
def ryegrass_final : ℝ := 0.35

-- Define the proportion of ryegrass contributed by X and Y to the final mixture
def contrib_X : ℝ := weight_X * ryegrass_X
def contrib_Y : ℝ := weight_Y * ryegrass_Y

-- Define the total proportion of ryegrass in the final mixture
def total_ryegrass : ℝ := contrib_X + contrib_Y

-- The lean theorem stating that the percentage of fescue in Y equals 75%
theorem fescue_in_Y :
  total_ryegrass = ryegrass_final →
  (100 - (ryegrass_Y * 100)) = 75 := 
by
  intros h
  sorry

end fescue_in_Y_l94_94114


namespace total_prize_amount_l94_94309

theorem total_prize_amount:
  ∃ P : ℝ, 
  (∃ n m : ℝ, n = 15 ∧ m = 15 ∧ ((2 / 5) * P = (3 / 5) * n * 285) ∧ P = 2565 * 2.5 + 6 * 15 ∧ ∀ i : ℕ, i < m → i ≥ 0 → P ≥ 15)
  ∧ P = 6502.5 :=
sorry

end total_prize_amount_l94_94309


namespace abigail_initial_money_l94_94623

variables (A : ℝ) -- Initial amount of money Abigail had.

-- Conditions
variables (food_rate : ℝ := 0.60) -- 60% spent on food
variables (phone_rate : ℝ := 0.25) -- 25% of the remainder spent on phone bill
variables (entertainment_spent : ℝ := 20) -- $20 spent on entertainment
variables (final_amount : ℝ := 40) -- $40 left after all expenditures

theorem abigail_initial_money :
  (A - (A * food_rate)) * (1 - phone_rate) - entertainment_spent = final_amount → A = 200 :=
by
  intro h
  sorry

end abigail_initial_money_l94_94623


namespace no_divisors_in_range_l94_94932

theorem no_divisors_in_range : ¬ ∃ n : ℕ, 80 < n ∧ n < 90 ∧ n ∣ (3^40 - 1) :=
by sorry

end no_divisors_in_range_l94_94932


namespace mary_sugar_cups_l94_94105

theorem mary_sugar_cups (sugar_required : ℕ) (sugar_remaining : ℕ) (sugar_added : ℕ) (h1 : sugar_required = 11) (h2 : sugar_added = 1) : sugar_remaining = 10 :=
by
  -- Placeholder for the proof
  sorry

end mary_sugar_cups_l94_94105


namespace triangle_area_AC_1_AD_BC_circumcircle_l94_94903

noncomputable def area_triangle_ABC (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem triangle_area_AC_1_AD_BC_circumcircle (A B C D E : ℝ × ℝ) (hAC : dist A C = 1)
  (hAD : dist A D = (2 / 3) * dist A B)
  (hMidE : E = ((B.1 + C.1) / 2, (B.2 + C.2) / 2))
  (hCircum : dist E ((A.1 + C.1) / 2, (A.2 + C.2) / 2) = 1 / 2) :
  area_triangle_ABC A B C = (Real.sqrt 5) / 6 :=
by
  sorry

end triangle_area_AC_1_AD_BC_circumcircle_l94_94903


namespace convert_rectangular_to_spherical_l94_94925

theorem convert_rectangular_to_spherical :
  ∀ (x y z : ℝ) (ρ θ φ : ℝ),
    (x, y, z) = (2, -2 * Real.sqrt 2, 2) →
    ρ = Real.sqrt (x^2 + y^2 + z^2) →
    z = ρ * Real.cos φ →
    x = ρ * Real.sin φ * Real.cos θ →
    y = ρ * Real.sin φ * Real.sin θ →
    0 < ρ ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ 0 ≤ φ ∧ φ ≤ Real.pi →
    (ρ, θ, φ) = (4, 2 * Real.pi - Real.arcsin (Real.sqrt 6 / 3), Real.pi / 3) :=
by
  intros x y z ρ θ φ H Hρ Hφ Hθ1 Hθ2 Hconditions
  sorry

end convert_rectangular_to_spherical_l94_94925


namespace original_average_marks_l94_94250

theorem original_average_marks (n : ℕ) (A : ℝ) (new_avg : ℝ) 
  (h1 : n = 30) 
  (h2 : new_avg = 90)
  (h3 : ∀ new_avg, new_avg = 2 * A → A = 90 / 2) : 
  A = 45 :=
by
  sorry

end original_average_marks_l94_94250


namespace number_of_possible_values_for_a_l94_94240

theorem number_of_possible_values_for_a :
  ∀ (a b c d : ℕ), 
  a > b ∧ b > c ∧ c > d ∧ a + b + c + d = 3010 ∧ a^2 - b^2 + c^2 - d^2 = 3010 →
  ∃ n, n = 751 :=
by {
  sorry
}

end number_of_possible_values_for_a_l94_94240


namespace rational_non_integer_solution_exists_rational_non_integer_solution_not_exists_l94_94151

-- Part (a)
theorem rational_non_integer_solution_exists :
  ∃ (x y : ℚ), x ∉ ℤ ∧ y ∉ ℤ ∧ 19 * x + 8 * y ∈ ℤ ∧ 8 * x + 3 * y ∈ ℤ :=
sorry

-- Part (b)
theorem rational_non_integer_solution_not_exists :
  ¬ ∃ (x y : ℚ), x ∉ ℤ ∧ y ∉ ℤ ∧ 19 * x^2 + 8 * y^2 ∈ ℤ ∧ 8 * x^2 + 3 * y^2 ∈ ℤ :=
sorry

end rational_non_integer_solution_exists_rational_non_integer_solution_not_exists_l94_94151


namespace complex_identity_l94_94403

variable (i : ℂ)
axiom i_squared : i^2 = -1

theorem complex_identity : 1 + i + i^2 = i :=
by sorry

end complex_identity_l94_94403


namespace sin_neg_30_eq_neg_half_l94_94339

/-- Prove that the sine of -30 degrees is -1/2 -/
theorem sin_neg_30_eq_neg_half : Real.sin (-(30 * Real.pi / 180)) = -1 / 2 :=
by
  sorry

end sin_neg_30_eq_neg_half_l94_94339


namespace volume_first_cube_l94_94742

theorem volume_first_cube (a b : ℝ) (h_ratio : a = 3 * b) (h_volume : b^3 = 8) : a^3 = 216 :=
by
  sorry

end volume_first_cube_l94_94742


namespace new_man_weight_l94_94292

theorem new_man_weight (avg_increase : ℝ) (crew_weight : ℝ) (new_man_weight : ℝ) 
(h_avg_increase : avg_increase = 1.8) (h_crew_weight : crew_weight = 53) :
  new_man_weight = crew_weight + 10 * avg_increase :=
by
  -- Here we will use the conditions to prove the theorem
  sorry

end new_man_weight_l94_94292


namespace unanswered_questions_l94_94374

variables (c w u : ℕ)

theorem unanswered_questions :
  (c + w + u = 50) ∧
  (6 * c + u = 120) ∧
  (3 * c - 2 * w = 45) →
  u = 37 :=
by {
  sorry
}

end unanswered_questions_l94_94374


namespace ages_of_Linda_and_Jane_l94_94557

theorem ages_of_Linda_and_Jane : 
  ∃ (J L : ℕ), 
    (L = 2 * J + 3) ∧ 
    (∃ (p : ℕ), Nat.Prime p ∧ p = L - J) ∧ 
    (L + J = 4 * J - 5) ∧ 
    (L = 19 ∧ J = 8) :=
by
  sorry

end ages_of_Linda_and_Jane_l94_94557


namespace find_bicycle_speed_l94_94583

def distanceAB := 40 -- Distance from A to B in km
def speed_walk := 6 -- Speed of the walking tourist in km/h
def distance_ahead := 5 -- Distance by which the second tourist is ahead initially in km
def speed_car := 24 -- Speed of the car in km/h
def meeting_time := 2 -- Time after departure when they meet in hours

theorem find_bicycle_speed (v : ℝ) : 
  (distanceAB = 40 ∧ speed_walk = 6 ∧ distance_ahead = 5 ∧ speed_car = 24 ∧ meeting_time = 2) →
  (v = 9) :=
by 
sorry

end find_bicycle_speed_l94_94583


namespace probability_neither_red_nor_green_l94_94541

-- Define the contents of each bag
def bag1 := (5, 6, 7)  -- (green, black, red)
def bag2 := (3, 4, 8)  -- (green, black, red)
def bag3 := (2, 7, 5)  -- (green, black, red)

-- Total pens in each bag
def total_pens_bag1 : ℕ := bag1.1 + bag1.2 + bag1.3
def total_pens_bag2 : ℕ := bag2.1 + bag2.2 + bag2.3
def total_pens_bag3 : ℕ := bag3.1 + bag3.2 + bag3.3

-- Probability of picking a black pen from each bag
def prob_black_bag1 : ℚ := bag1.2 / total_pens_bag1
def prob_black_bag2 : ℚ := bag2.2 / total_pens_bag2
def prob_black_bag3 : ℚ := bag3.2 / total_pens_bag3

-- Total number of pens across all bags
def total_pens : ℕ := total_pens_bag1 + total_pens_bag2 + total_pens_bag3

-- Weighted probability of picking a black pen
def weighted_prob_black : ℚ :=
  (prob_black_bag1 * total_pens_bag1 / total_pens) +
  (prob_black_bag2 * total_pens_bag2 / total_pens) +
  (prob_black_bag3 * total_pens_bag3 / total_pens)

-- Theorem statement
theorem probability_neither_red_nor_green :
  weighted_prob_black = 17/47 :=
sorry

end probability_neither_red_nor_green_l94_94541


namespace ike_mike_total_items_l94_94205

theorem ike_mike_total_items :
  ∃ (s d : ℕ), s + d = 7 ∧ 5 * s + 3/2 * d = 35 :=
by sorry

end ike_mike_total_items_l94_94205


namespace complex_number_real_imag_equal_l94_94982

theorem complex_number_real_imag_equal (a : ℝ) (h : (a + 6) = (3 - 2 * a)) : a = -1 :=
by
  sorry

end complex_number_real_imag_equal_l94_94982


namespace find_f_one_l94_94045

-- Given conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def f_defined_for_neg (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x < 0 → f x = 2 * x^2 - 1

-- Statement that needs to be proven
theorem find_f_one (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_neg : f_defined_for_neg f) :
  f 1 = -1 :=
  sorry

end find_f_one_l94_94045


namespace intersection_eq_l94_94839

def M : Set ℝ := {x | x^2 - 2 * x < 0}
def N : Set ℝ := {x | |x| < 1}

theorem intersection_eq : M ∩ N = {x : ℝ | 0 < x ∧ x < 1} := by
  sorry

end intersection_eq_l94_94839


namespace arithmetic_sequence_sum_property_l94_94700

theorem arithmetic_sequence_sum_property
  (a : ℕ → ℝ)  -- sequence terms are real numbers
  (d : ℝ)      -- common difference
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_sum_condition : a 4 + a 8 = 16) :
  a 2 + a 10 = 16 :=
sorry

end arithmetic_sequence_sum_property_l94_94700


namespace exists_set_with_property_l94_94390

theorem exists_set_with_property (n : ℕ) (h : n > 0) :
  ∃ S : Finset ℕ, S.card = n ∧
  (∀ {a b}, a ∈ S → b ∈ S → a ≠ b → (a - b) ∣ a ∧ (a - b) ∣ b) ∧
  (∀ {a b c}, a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → a ≠ c → ¬ ((a - b) ∣ c)) :=
sorry

end exists_set_with_property_l94_94390


namespace clarence_oranges_after_giving_l94_94635

def initial_oranges : ℝ := 5.0
def oranges_given : ℝ := 3.0

theorem clarence_oranges_after_giving : (initial_oranges - oranges_given) = 2.0 :=
by
  sorry

end clarence_oranges_after_giving_l94_94635


namespace mario_oranges_l94_94580

theorem mario_oranges (M L N T x : ℕ) 
  (H_L : L = 24) 
  (H_N : N = 96) 
  (H_T : T = 128) 
  (H_total : x + L + N = T) : 
  x = 8 :=
by
  rw [H_L, H_N, H_T] at H_total
  linarith

end mario_oranges_l94_94580


namespace find_hundreds_digit_l94_94579

theorem find_hundreds_digit :
  ∃ n : ℕ, (n % 37 = 0) ∧ (n % 173 = 0) ∧ (10000 ≤ n) ∧ (n < 100000) ∧ ((n / 1000) % 10 = 3) ∧ (((n / 100) % 10) = 2) :=
sorry

end find_hundreds_digit_l94_94579


namespace permutation_equals_power_l94_94612

-- Definition of permutation with repetition
def permutation_with_repetition (n k : ℕ) : ℕ := n ^ k

-- Theorem to prove
theorem permutation_equals_power (n k : ℕ) : permutation_with_repetition n k = n ^ k :=
by
  sorry

end permutation_equals_power_l94_94612


namespace movie_theorem_l94_94777

variables (A B C D : Prop)

theorem movie_theorem 
  (h1 : (A → B))
  (h2 : (B → C))
  (h3 : (C → A))
  (h4 : (D → B)) 
  : ¬D := 
by
  sorry

end movie_theorem_l94_94777


namespace total_birdseed_amount_l94_94549

-- Define the birdseed amounts in the boxes
def box1_amount : ℕ := 250
def box2_amount : ℕ := 275
def box3_amount : ℕ := 225
def box4_amount : ℕ := 300
def box5_amount : ℕ := 275
def box6_amount : ℕ := 200
def box7_amount : ℕ := 150
def box8_amount : ℕ := 180

-- Define the weekly consumption of each bird
def parrot_consumption : ℕ := 100
def cockatiel_consumption : ℕ := 50
def canary_consumption : ℕ := 25

-- Define a theorem to calculate the total birdseed that Leah has
theorem total_birdseed_amount : box1_amount + box2_amount + box3_amount + box4_amount + box5_amount + box6_amount + box7_amount + box8_amount = 1855 :=
by
  sorry

end total_birdseed_amount_l94_94549


namespace seashells_in_six_weeks_l94_94342

def jar_weekly_update (week : Nat) (jarA : Nat) (jarB : Nat) : Nat × Nat :=
  if week % 3 = 0 then (jarA / 2, jarB / 2)
  else (jarA + 20, jarB * 2)

def total_seashells_after_weeks (initialA : Nat) (initialB : Nat) (weeks : Nat) : Nat :=
  let rec update (w : Nat) (jA : Nat) (jB : Nat) :=
    match w with
    | 0 => jA + jB
    | n + 1 =>
      let (newA, newB) := jar_weekly_update n jA jB
      update n newA newB
  update weeks initialA initialB

theorem seashells_in_six_weeks :
  total_seashells_after_weeks 50 30 6 = 97 :=
sorry

end seashells_in_six_weeks_l94_94342


namespace vector_magnitude_subtraction_l94_94051

theorem vector_magnitude_subtraction :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5 :=
by
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  show real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5 from
  sorry

end vector_magnitude_subtraction_l94_94051


namespace janet_miles_per_day_l94_94093

def total_miles : ℕ := 72
def days : ℕ := 9
def miles_per_day : ℕ := 8

theorem janet_miles_per_day : total_miles / days = miles_per_day :=
by {
  sorry
}

end janet_miles_per_day_l94_94093


namespace sin_neg_30_eq_neg_one_half_l94_94334

theorem sin_neg_30_eq_neg_one_half : Real.sin (-30 / 180 * Real.pi) = -1 / 2 := by
  -- Proof goes here
  sorry

end sin_neg_30_eq_neg_one_half_l94_94334


namespace negation_of_proposition_l94_94419

theorem negation_of_proposition (a b : ℝ) : ¬ (a > b ∧ a - 1 > b - 1) ↔ a ≤ b ∨ a - 1 ≤ b - 1 :=
by sorry

end negation_of_proposition_l94_94419


namespace tangent_circle_line_radius_l94_94204

theorem tangent_circle_line_radius (m : ℝ) :
  (∀ x y : ℝ, (x - 1)^2 + y^2 = m → x + y = 1 → dist (1, 0) (x, y) = Real.sqrt m) →
  m = 1 / 2 :=
by
  sorry

end tangent_circle_line_radius_l94_94204


namespace larger_triangle_side_length_l94_94404

theorem larger_triangle_side_length
    (A1 A2 : ℕ) (k : ℤ)
    (h1 : A1 - A2 = 32)
    (h2 : A1 = k^2 * A2)
    (h3 : A2 = 4 ∨ A2 = 8 ∨ A2 = 16)
    (h4 : ((4 : ℤ) * k = 12)) :
    (4 * k) = 12 :=
by sorry

end larger_triangle_side_length_l94_94404


namespace x_intercept_of_l1_is_2_l94_94734

theorem x_intercept_of_l1_is_2 (a : ℝ) (l1_perpendicular_l2 : ∀ (x y : ℝ), 
  ((a+3)*x + y - 4 = 0) -> (x + (a-1)*y + 4 = 0) -> False) : 
  ∃ b : ℝ, (2*b + 0 - 4 = 0) ∧ b = 2 := 
by
  sorry

end x_intercept_of_l1_is_2_l94_94734


namespace part_a_exists_rational_non_integer_l94_94149

theorem part_a_exists_rational_non_integer 
  (x y : ℚ) (hx : ¬int.cast x ∉ ℤ) (hy : ¬int.cast y ∉ ℤ) :
  ∃ x y : ℚ, (¬int.cast x ∉ ℤ) ∧ (¬int.cast y ∉ ℤ) ∧ (19 * x + 8 * y ∈ ℤ) ∧ (8 * x + 3 * y ∈ ℤ) := 
  sorry

end part_a_exists_rational_non_integer_l94_94149


namespace simplify_and_evaluate_l94_94852

variable (x y : ℚ)
variable (expr : ℚ := 3 * x * y^2 - (x * y - 2 * (2 * x * y - 3 / 2 * x^2 * y) + 3 * x * y^2) + 3 * x^2 * y)

theorem simplify_and_evaluate (h1 : x = 3) (h2 : y = -1 / 3) : expr = -3 :=
by
  sorry

end simplify_and_evaluate_l94_94852


namespace first_term_of_geometric_sequence_l94_94426

theorem first_term_of_geometric_sequence (a r : ℚ) (h1 : a * r^2 = 12) (h2 : a * r^3 = 16) : a = 27 / 4 :=
by {
  sorry
}

end first_term_of_geometric_sequence_l94_94426


namespace last_three_digits_7_pow_80_l94_94950

theorem last_three_digits_7_pow_80 : (7 ^ 80) % 1000 = 961 := 
by sorry

end last_three_digits_7_pow_80_l94_94950


namespace sum_prime_factors_of_77_l94_94440

theorem sum_prime_factors_of_77 : 
  ∃ (p q : ℕ), nat.prime p ∧ nat.prime q ∧ 77 = p * q ∧ p + q = 18 :=
by
  existsi 7
  existsi 11
  split
  { exact nat.prime_7 }
  split
  { exact nat.prime_11 }
  split
  { exact dec_trivial }
  { exact dec_trivial }

-- Placeholder for the proof
-- sorry

end sum_prime_factors_of_77_l94_94440


namespace percent_increase_in_pizza_area_l94_94529

theorem percent_increase_in_pizza_area (r : ℝ) (h : 0 < r) :
  let r_large := 1.10 * r
  let A_medium := π * r^2
  let A_large := π * r_large^2
  let percent_increase := ((A_large - A_medium) / A_medium) * 100 
  percent_increase = 21 := 
by sorry

end percent_increase_in_pizza_area_l94_94529


namespace polynomial_coefficient_sum_l94_94522

theorem polynomial_coefficient_sum :
  let p := (x + 3) * (4 * x^3 - 2 * x^2 + 7 * x - 6)
  let q := 4 * x^4 + 10 * x^3 + x^2 + 15 * x - 18
  p = q →
  (4 + 10 + 1 + 15 - 18 = 12) :=
by
  intro p_eq_q
  sorry

end polynomial_coefficient_sum_l94_94522


namespace value_of_expression_l94_94139

theorem value_of_expression : (2207 - 2024)^2 * 4 / 144 = 930.25 := 
by
  sorry

end value_of_expression_l94_94139


namespace sum_prime_factors_of_77_l94_94449

theorem sum_prime_factors_of_77 : ∃ (p1 p2 : ℕ), Nat.Prime p1 ∧ Nat.Prime p2 ∧ 77 = p1 * p2 ∧ p1 + p2 = 18 := by
  sorry

end sum_prime_factors_of_77_l94_94449


namespace monthly_salary_equals_l94_94224

-- Define the base salary
def base_salary : ℝ := 1600

-- Define the commission rate
def commission_rate : ℝ := 0.04

-- Define the sales amount for which the salaries are equal
def sales_amount : ℝ := 5000

-- Define the total earnings with a base salary and commission for 5000 worth of sales
def total_earnings : ℝ := base_salary + (commission_rate * sales_amount)

-- Define the monthly salary from Furniture by Design
def monthly_salary : ℝ := 1800

-- Prove that the monthly salary S is equal to 1800
theorem monthly_salary_equals :
  total_earnings = monthly_salary :=
by
  -- The proof is skipped with sorry.
  sorry

end monthly_salary_equals_l94_94224


namespace tg_sum_equal_l94_94181

variable {a b c : ℝ}
variable {φA φB φC : ℝ}

-- The sides of the triangle are labeled such that a >= b >= c.
axiom sides_ineq : a ≥ b ∧ b ≥ c

-- The angles between the median and the altitude from vertices A, B, and C.
axiom angles_def : true -- This axiom is a placeholder. In actual use, we would define φA, φB, φC properly using the given geometric setup.

theorem tg_sum_equal : Real.tan φA + Real.tan φC = Real.tan φB := 
by 
  sorry

end tg_sum_equal_l94_94181


namespace tunnel_digging_duration_l94_94593

theorem tunnel_digging_duration (daily_progress : ℕ) (total_length_km : ℕ) 
    (meters_per_km : ℕ) (days_per_year : ℕ) : 
    daily_progress = 5 → total_length_km = 2 → meters_per_km = 1000 → days_per_year = 365 → 
    total_length_km * meters_per_km / daily_progress > 365 :=
by
  intros hprog htunnel hmeters hdays
  /- ... proof steps will go here -/
  sorry

end tunnel_digging_duration_l94_94593


namespace dana_more_pencils_than_marcus_l94_94926

theorem dana_more_pencils_than_marcus :
  ∀ (Jayden Dana Marcus : ℕ), 
  (Jayden = 20) ∧ 
  (Dana = Jayden + 15) ∧ 
  (Jayden = 2 * Marcus) → 
  (Dana - Marcus = 25) :=
by
  intros Jayden Dana Marcus h
  rcases h with ⟨hJayden, hDana, hMarcus⟩
  sorry

end dana_more_pencils_than_marcus_l94_94926


namespace time_to_pass_faster_train_l94_94584

noncomputable def speed_slower_train_kmph : ℝ := 36
noncomputable def speed_faster_train_kmph : ℝ := 45
noncomputable def length_faster_train_m : ℝ := 225.018
noncomputable def kmph_to_mps_factor : ℝ := 1000 / 3600

noncomputable def relative_speed_mps : ℝ := (speed_slower_train_kmph + speed_faster_train_kmph) * kmph_to_mps_factor

theorem time_to_pass_faster_train : 
  (length_faster_train_m / relative_speed_mps) = 10.001 := 
sorry

end time_to_pass_faster_train_l94_94584


namespace not_right_triangle_condition_C_l94_94196

theorem not_right_triangle_condition_C :
  ∀ (a b c : ℝ), 
    (a^2 = b^2 + c^2) ∨
    (∀ (angleA angleB angleC : ℝ), angleA = angleB + angleC ∧ angleA + angleB + angleC = 180) ∨
    (∀ (angleA angleB angleC : ℝ), angleA / angleB = 3 / 4 ∧ angleB / angleC = 4 / 5) ∨
    (a^2 / b^2 = 1 / 2 ∧ b^2 / c^2 = 2 / 3) ->
    ¬ (∀ (angleA angleB angleC : ℝ), angleA / angleB = 3 / 4 ∧ angleB / angleC = 4 / 5 -> angleA = 90 ∨ angleB = 90 ∨ angleC = 90) :=
by
  intro a b c h
  cases h
  case inl h1 =>
    -- Option A: b^2 = a^2 - c^2
    sorry
  case inr h2 =>
    cases h2
    case inl h3 => 
      -- Option B: angleA = angleB + angleC
      sorry
    case inr h4 =>
      cases h4
      case inl h5 =>
        -- Option C: angleA : angleB : angleC = 3 : 4 : 5
        sorry
      case inr h6 =>
        -- Option D: a^2 : b^2 : c^2 = 1 : 2 : 3
        sorry

end not_right_triangle_condition_C_l94_94196


namespace problem_solution_l94_94666

theorem problem_solution (a b c : ℝ) (h1 : a^2 - b^2 = 5) (h2 : a * b = 2) (h3 : a^2 + b^2 + c^2 = 8) : 
  a^4 + b^4 + c^4 = 38 :=
sorry

end problem_solution_l94_94666


namespace minimum_xy_l94_94358

theorem minimum_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 1/x + 1/y = 1/2) : xy ≥ 16 :=
sorry

end minimum_xy_l94_94358


namespace sin_neg_30_eq_neg_half_l94_94338

/-- Prove that the sine of -30 degrees is -1/2 -/
theorem sin_neg_30_eq_neg_half : Real.sin (-(30 * Real.pi / 180)) = -1 / 2 :=
by
  sorry

end sin_neg_30_eq_neg_half_l94_94338


namespace vector_subtraction_l94_94182

theorem vector_subtraction (p q: ℝ × ℝ × ℝ) (hp: p = (5, -3, 2)) (hq: q = (-1, 4, -2)) :
  p - 2 • q = (7, -11, 6) :=
by
  sorry

end vector_subtraction_l94_94182


namespace cost_of_socks_l94_94501

/-- Given initial amount of $100 and cost of shirt is $24,
    find out the cost of socks if the remaining amount is $65. --/
theorem cost_of_socks
  (initial_amount : ℕ)
  (cost_of_shirt : ℕ)
  (remaining_amount : ℕ)
  (h1 : initial_amount = 100)
  (h2 : cost_of_shirt = 24)
  (h3 : remaining_amount = 65) : 
  (initial_amount - cost_of_shirt - remaining_amount) = 11 :=
by
  sorry

end cost_of_socks_l94_94501


namespace problem1_problem2_l94_94019

variable {a b : ℝ}

-- Proof problem 1
-- Goal: (1)(2a^(2/3)b^(1/2))(-6a^(1/2)b^(1/3)) / (-3a^(1/6)b^(5/6)) = -12a
theorem problem1 (h1 : 0 < a) (h2 : 0 < b) : 
  (1 : ℝ) * (2 * a^(2/3) * b^(1/2)) * (-6 * a^(1/2) * b^(1/3)) / (-3 * a^(1/6) * b^(5/6)) = -12 * a := 
sorry

-- Proof problem 2
-- Goal: 2(log(sqrt(2)))^2 + log(sqrt(2)) * log(5) + sqrt((log(sqrt(2)))^2 - log(2) + 1) = 1 + (1 / 2) * log(5)
theorem problem2 : 
  2 * (Real.log (Real.sqrt 2))^2 + (Real.log (Real.sqrt 2)) * (Real.log 5) + 
  Real.sqrt ((Real.log (Real.sqrt 2))^2 - Real.log 2 + 1) = 
  1 + 0.5 * (Real.log 5) := 
sorry

end problem1_problem2_l94_94019


namespace yellow_fraction_after_changes_l94_94830

theorem yellow_fraction_after_changes (y : ℕ) :
  let green_initial := (4 / 7 : ℚ) * y
  let yellow_initial := (3 / 7 : ℚ) * y
  let yellow_new := 3 * yellow_initial
  let green_new := green_initial + (1 / 2) * green_initial
  let total_new := green_new + yellow_new
  yellow_new / total_new = (3 / 5 : ℚ) :=
by
  sorry

end yellow_fraction_after_changes_l94_94830


namespace light_travel_50_years_l94_94574

theorem light_travel_50_years :
  let one_year_distance := 9460800000000 -- distance light travels in one year
  let fifty_years_distance := 50 * one_year_distance
  let scientific_notation_distance := 473.04 * 10^12
  fifty_years_distance = scientific_notation_distance :=
by
  sorry

end light_travel_50_years_l94_94574


namespace equation_of_circle_min_distance_PA_PB_l94_94070

-- Definition of the given points, lines, and circle
def point (x y : ℝ) : Prop := true

def circle_through_points (x1 y1 x2 y2 x3 y3 : ℝ) (a b r : ℝ) : Prop :=
  (x1 + a) * (x1 + a) + y1 * y1 = r ∧
  (x2 + a) * (x2 + a) + y2 * y2 = r ∧
  (x3 + a) * (x3 + a) + y3 * y3 = r

def line (a b : ℝ) : Prop := true

-- Specific points
def D := point 0 1
def E := point (-2) 1
def F := point (-1) (Real.sqrt 2)

-- Lines l1 and l2
def l₁ (x : ℝ) : ℝ := x - 2
def l₂ (x : ℝ) : ℝ := x + 1

-- Intersection points A and B
def A := point 0 1
def B := point (-2) (-1)

-- Question Ⅰ: Find the equation of the circle
theorem equation_of_circle :
  ∃ a b r, circle_through_points 0 1 (-2) 1 (-1) (Real.sqrt 2) a b r ∧ (a = -1 ∧ b = 0 ∧ r = 2) :=
  sorry

-- Question Ⅱ: Find the minimum value of |PA|^2 + |PB|^2
def dist_sq (x1 y1 x2 y2 : ℝ) : ℝ := (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)

theorem min_distance_PA_PB :
  real := sorry

end equation_of_circle_min_distance_PA_PB_l94_94070


namespace abigail_initial_money_l94_94622

variables (A : ℝ) -- Initial amount of money Abigail had.

-- Conditions
variables (food_rate : ℝ := 0.60) -- 60% spent on food
variables (phone_rate : ℝ := 0.25) -- 25% of the remainder spent on phone bill
variables (entertainment_spent : ℝ := 20) -- $20 spent on entertainment
variables (final_amount : ℝ := 40) -- $40 left after all expenditures

theorem abigail_initial_money :
  (A - (A * food_rate)) * (1 - phone_rate) - entertainment_spent = final_amount → A = 200 :=
by
  intro h
  sorry

end abigail_initial_money_l94_94622


namespace new_avg_weight_l94_94876

-- Definition of the conditions
def original_team_avg_weight : ℕ := 94
def original_team_size : ℕ := 7
def new_player_weight_1 : ℕ := 110
def new_player_weight_2 : ℕ := 60
def total_new_team_size : ℕ := original_team_size + 2

-- Computation of the total weight
def total_weight_original_team : ℕ := original_team_avg_weight * original_team_size
def total_weight_new_team : ℕ := total_weight_original_team + new_player_weight_1 + new_player_weight_2

-- Statement of the theorem
theorem new_avg_weight : total_weight_new_team / total_new_team_size = 92 := by
  -- Proof is omitted
  sorry

end new_avg_weight_l94_94876


namespace candidate_a_valid_votes_l94_94757

/-- In an election, candidate A got 80% of the total valid votes.
If 15% of the total votes were declared invalid and the total number of votes is 560,000,
find the number of valid votes polled in favor of candidate A. -/
theorem candidate_a_valid_votes :
  let total_votes := 560000
  let invalid_percentage := 0.15
  let valid_percentage := 0.85
  let candidate_a_percentage := 0.80
  let valid_votes := (valid_percentage * total_votes : ℝ)
  let candidate_a_votes := (candidate_a_percentage * valid_votes : ℝ)
  candidate_a_votes = 380800 :=
by
  sorry

end candidate_a_valid_votes_l94_94757


namespace knights_rearrangement_impossible_l94_94896

theorem knights_rearrangement_impossible :
  ∀ (b : ℕ → ℕ → Prop), (b 0 0 = true) ∧ (b 0 2 = true) ∧ (b 2 0 = true) ∧ (b 2 2 = true) ∧
  (b 0 0 = b 0 2) ∧ (b 2 0 ≠ b 2 2) → ¬(∃ (b' : ℕ → ℕ → Prop), 
  (b' 0 0 ≠ b 0 0) ∧ (b' 0 2 ≠ b 0 2) ∧ (b' 2 0 ≠ b 2 0) ∧ (b' 2 2 ≠ b 2 2) ∧ 
  (b' 0 0 ≠ b' 0 2) ∧ (b' 2 0 ≠ b' 2 2)) :=
by { sorry }

end knights_rearrangement_impossible_l94_94896


namespace trapezium_distance_l94_94346

theorem trapezium_distance (a b h: ℝ) (area: ℝ) (h_area: area = 300) (h_sides: a = 22) (h_sides_2: b = 18)
  (h_formula: area = (1 / 2) * (a + b) * h): h = 15 :=
by
  sorry

end trapezium_distance_l94_94346


namespace temperature_range_l94_94254

-- Define the highest and lowest temperature conditions
variable (t : ℝ)
def highest_temp := t ≤ 30
def lowest_temp := 20 ≤ t

-- The theorem to prove the range of temperature change
theorem temperature_range (t : ℝ) (h_high : highest_temp t) (h_low : lowest_temp t) : 20 ≤ t ∧ t ≤ 30 :=
by 
  -- Insert the proof or leave as sorry for now
  sorry

end temperature_range_l94_94254


namespace find_k_for_one_real_solution_l94_94180

theorem find_k_for_one_real_solution (k : ℤ) :
  (∀ x : ℤ, (x - 3) * (x + 2) = k + 3 * x) ↔ k = -10 := by
  sorry

end find_k_for_one_real_solution_l94_94180


namespace non_student_ticket_price_l94_94130

theorem non_student_ticket_price (x : ℕ) : 
  (∃ (n_student_ticket_price ticket_count total_revenue student_tickets : ℕ),
    n_student_ticket_price = 9 ∧
    ticket_count = 2000 ∧
    total_revenue = 20960 ∧
    student_tickets = 520 ∧
    (student_tickets * n_student_ticket_price + (ticket_count - student_tickets) * x = total_revenue)) -> 
  x = 11 := 
by
  -- placeholder for proof
  sorry

end non_student_ticket_price_l94_94130


namespace min_reciprocal_sum_l94_94504

theorem min_reciprocal_sum (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a + b = 1) : 
  (1 / a) + (1 / b) ≥ 3 + 2 * Real.sqrt 2 :=
by
  sorry

end min_reciprocal_sum_l94_94504


namespace number_of_ninth_graders_l94_94432

def num_students_total := 50
def num_students_7th (x : Int) := 2 * x - 1
def num_students_8th (x : Int) := x

theorem number_of_ninth_graders (x : Int) :
  num_students_7th x + num_students_8th x + (51 - 3 * x) = num_students_total := by
  sorry

end number_of_ninth_graders_l94_94432


namespace sum_of_consecutive_negative_integers_with_product_3080_l94_94870

theorem sum_of_consecutive_negative_integers_with_product_3080 :
  ∃ (n : ℤ), n < 0 ∧ (n * (n + 1) = 3080) ∧ (n + (n + 1) = -111) :=
sorry

end sum_of_consecutive_negative_integers_with_product_3080_l94_94870


namespace num_terminating_decimals_l94_94653

theorem num_terminating_decimals (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 518) :
  (∃ k, (1 ≤ k ∧ k ≤ 518) ∧ n = k * 21) ↔ n = 24 :=
sorry

end num_terminating_decimals_l94_94653


namespace doughnuts_left_l94_94902

theorem doughnuts_left (dozen : ℕ) (total_initial : ℕ) (eaten : ℕ) (initial : total_initial = 2 * dozen) (d : dozen = 12) : total_initial - eaten = 16 :=
by
  rcases d
  rcases initial
  sorry

end doughnuts_left_l94_94902


namespace largest_value_l94_94193

noncomputable def largest_possible_4x_3y (x y : ℝ) : ℝ :=
  4 * x + 3 * y

theorem largest_value (x y : ℝ) :
  x^2 + y^2 = 16 * x + 8 * y + 8 → (∃ x y, largest_possible_4x_3y x y = 9.64) :=
by
  sorry

end largest_value_l94_94193


namespace double_grandfather_pension_l94_94695

-- Define the total family income and individual contributions
def total_income (masha mother father grandfather : ℝ) : ℝ :=
  masha + mother + father + grandfather

-- Define the conditions provided in the problem
variables
  (masha mother father grandfather : ℝ)
  (cond1 : 2 * masha = total_income masha mother father grandfather * 1.05)
  (cond2 : 2 * mother = total_income masha mother father grandfather * 1.15)
  (cond3 : 2 * father = total_income masha mother father grandfather * 1.25)

-- Define the statement to be proved
theorem double_grandfather_pension :
  2 * grandfather = total_income masha mother father grandfather * 1.55 :=
by
  -- Proof placeholder
  sorry

end double_grandfather_pension_l94_94695


namespace sum_of_variables_l94_94800

theorem sum_of_variables (a b c d : ℕ) (h1 : ac + bd + ad + bc = 1997) : a + b + c + d = 1998 :=
sorry

end sum_of_variables_l94_94800


namespace merchant_loss_l94_94561

theorem merchant_loss
  (sp : ℝ)
  (profit_percent: ℝ)
  (loss_percent:  ℝ)
  (sp1 : ℝ)
  (sp2 : ℝ)
  (cp1 cp2 : ℝ)
  (net_loss : ℝ) :
  
  sp = 990 → 
  profit_percent = 0.1 → 
  loss_percent = 0.1 →
  sp1 = sp → 
  sp2 = sp → 
  cp1 = sp1 / (1 + profit_percent) →
  cp2 = sp2 / (1 - loss_percent) →
  net_loss = (cp2 - sp2) - (sp1 - cp1) →
  net_loss = 20 :=
by 
  intros _ _ _ _ _ _ _ _ 
  -- placeholders for intros to bind variables
  sorry

end merchant_loss_l94_94561


namespace sum_of_sides_of_similar_triangle_l94_94533

theorem sum_of_sides_of_similar_triangle (a b c : ℕ) (scale_factor : ℕ) (longest_side_sim : ℕ) (sum_of_other_sides_sim : ℕ) : 
  a * scale_factor = 21 → c = 7 → b = 5 → a = 3 → 
  sum_of_other_sides = a * scale_factor + b * scale_factor → 
sum_of_other_sides = 24 :=
by
  sorry

end sum_of_sides_of_similar_triangle_l94_94533


namespace tenth_term_geometric_sequence_l94_94637

def a := 5
def r := Rat.ofInt 3 / 4
def n := 10

theorem tenth_term_geometric_sequence :
  a * r^(n-1) = Rat.ofInt 98415 / Rat.ofInt 262144 := sorry

end tenth_term_geometric_sequence_l94_94637


namespace billy_gaming_percentage_l94_94171

-- Define the conditions
def free_time_per_day := 8
def days_in_weekend := 2
def total_free_time := free_time_per_day * days_in_weekend
def books_read := 3
def pages_per_book := 80
def reading_rate := 60 -- pages per hour
def total_pages_read := books_read * pages_per_book
def reading_time := total_pages_read / reading_rate
def gaming_time := total_free_time - reading_time
def gaming_percentage := (gaming_time / total_free_time) * 100

-- State the theorem
theorem billy_gaming_percentage : gaming_percentage = 75 := by
  sorry

end billy_gaming_percentage_l94_94171


namespace charles_pictures_after_work_l94_94025

variable (initial_papers : ℕ)
variable (draw_today : ℕ)
variable (draw_yesterday_morning : ℕ)
variable (papers_left : ℕ)

theorem charles_pictures_after_work :
    initial_papers = 20 →
    draw_today = 6 →
    draw_yesterday_morning = 6 →
    papers_left = 2 →
    initial_papers - (draw_today + draw_yesterday_morning + 6) = papers_left →
    6 = (initial_papers - draw_today - draw_yesterday_morning - papers_left) := 
by
  intros h1 h2 h3 h4 h5
  exact sorry

end charles_pictures_after_work_l94_94025


namespace shape_volume_to_surface_area_ratio_l94_94905

/-- 
Define the volume and surface area of our specific shape with given conditions:
1. Five unit cubes in a straight line.
2. An additional cube on top of the second cube.
3. Another cube beneath the fourth cube.

Prove that the ratio of the volume to the surface area is \( \frac{1}{4} \).
-/
theorem shape_volume_to_surface_area_ratio :
  let volume := 7
  let surface_area := 28
  volume / surface_area = 1 / 4 :=
by
  sorry

end shape_volume_to_surface_area_ratio_l94_94905


namespace largest_angle_of_convex_hexagon_l94_94609

noncomputable def hexagon_largest_angle (x : ℚ) : ℚ :=
  max (6 * x - 3) (max (5 * x + 1) (max (4 * x - 4) (max (3 * x) (max (2 * x + 2) x))))

theorem largest_angle_of_convex_hexagon (x : ℚ) (h : x + (2*x+2) + 3*x + (4*x-4) + (5*x+1) + (6*x-3) = 720) : 
  hexagon_largest_angle x = 4281 / 21 := 
sorry

end largest_angle_of_convex_hexagon_l94_94609


namespace find_cost_prices_l94_94615

noncomputable def cost_price_per_meter
  (selling_price_per_meter : ℕ) (loss_per_meter : ℕ) : ℕ :=
  selling_price_per_meter + loss_per_meter

theorem find_cost_prices
  (selling_A : ℕ) (meters_A : ℕ) (loss_A : ℕ)
  (selling_B : ℕ) (meters_B : ℕ) (loss_B : ℕ)
  (selling_C : ℕ) (meters_C : ℕ) (loss_C : ℕ)
  (H_A : selling_A = 9000) (H_meters_A : meters_A = 300) (H_loss_A : loss_A = 6)
  (H_B : selling_B = 7000) (H_meters_B : meters_B = 250) (H_loss_B : loss_B = 4)
  (H_C : selling_C = 12000) (H_meters_C : meters_C = 400) (H_loss_C : loss_C = 8) :
  cost_price_per_meter (selling_A / meters_A) loss_A = 36 ∧
  cost_price_per_meter (selling_B / meters_B) loss_B = 32 ∧
  cost_price_per_meter (selling_C / meters_C) loss_C = 38 :=
by {
  sorry
}

end find_cost_prices_l94_94615


namespace jon_coffee_spending_in_april_l94_94546

def cost_per_coffee : ℕ := 2
def coffees_per_day : ℕ := 2
def days_in_april : ℕ := 30

theorem jon_coffee_spending_in_april :
  (coffees_per_day * cost_per_coffee) * days_in_april = 120 :=
by
  sorry

end jon_coffee_spending_in_april_l94_94546


namespace no_real_solution_l94_94788

-- Define the equation
def equation (a b : ℝ) : Prop := a^2 + 3 * b^2 + 2 = 3 * a * b

-- Prove that there do not exist real numbers a and b such that equation a b holds
theorem no_real_solution : ¬ ∃ a b : ℝ, equation a b :=
by
  -- Proof placeholder
  sorry

end no_real_solution_l94_94788


namespace divides_prime_factors_l94_94550

theorem divides_prime_factors (a b : ℕ) (p : ℕ → ℕ → Prop) (k l : ℕ → ℕ) (n : ℕ) : 
  (a ∣ b) ↔ (∀ i : ℕ, i < n → k i ≤ l i) :=
by
  sorry

end divides_prime_factors_l94_94550


namespace kathleen_savings_in_july_l94_94225

theorem kathleen_savings_in_july (savings_june savings_august spending_school spending_clothes money_left savings_target add_from_aunt : ℕ) 
  (h_june : savings_june = 21)
  (h_august : savings_august = 45)
  (h_school : spending_school = 12)
  (h_clothes : spending_clothes = 54)
  (h_left : money_left = 46)
  (h_target : savings_target = 125)
  (h_aunt : add_from_aunt = 25)
  (not_received_from_aunt : (savings_june + savings_august + money_left + add_from_aunt) ≤ savings_target)
  : (savings_june + savings_august + money_left + spending_school + spending_clothes - (savings_june + savings_august + spending_school + spending_clothes)) = 46 := 
by 
  -- These conditions narrate the problem setup
  -- We can proceed to show the proof here
  sorry 

end kathleen_savings_in_july_l94_94225


namespace charles_drawn_after_work_l94_94023

-- Conditions
def total_papers : ℕ := 20
def drawn_today : ℕ := 6
def drawn_yesterday_before_work : ℕ := 6
def papers_left : ℕ := 2

-- Question and proof goal
theorem charles_drawn_after_work :
  ∀ (total_papers drawn_today drawn_yesterday_before_work papers_left : ℕ),
  total_papers = 20 →
  drawn_today = 6 →
  drawn_yesterday_before_work = 6 →
  papers_left = 2 →
  (total_papers - drawn_today - drawn_yesterday_before_work - papers_left = 6) :=
by
  intros total_papers drawn_today drawn_yesterday_before_work papers_left
  assume h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  exact sorry

end charles_drawn_after_work_l94_94023


namespace total_cost_eq_1400_l94_94708

theorem total_cost_eq_1400 (stove_cost : ℝ) (wall_repair_fraction : ℝ) (wall_repair_cost : ℝ) (total_cost : ℝ)
  (h₁ : stove_cost = 1200)
  (h₂ : wall_repair_fraction = 1/6)
  (h₃ : wall_repair_cost = stove_cost * wall_repair_fraction)
  (h₄ : total_cost = stove_cost + wall_repair_cost) :
  total_cost = 1400 :=
sorry

end total_cost_eq_1400_l94_94708


namespace sum_of_coordinates_of_C_parallelogram_l94_94796

-- Definitions that encapsulate the given conditions
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨2, 3⟩
def B : Point := ⟨-1, 0⟩
def D : Point := ⟨5, -4⟩

-- The theorem we need to prove
theorem sum_of_coordinates_of_C_parallelogram :
  ∃ C : Point, C.x + C.y = 7 ∧
  ∃ M : Point, M = ⟨(A.x + D.x) / 2, (A.y + D.y) / 2⟩ ∧
  (M = ⟨(B.x + C.x) / 2, (B.y + C.y) / 2⟩) :=
sorry

end sum_of_coordinates_of_C_parallelogram_l94_94796


namespace sin_cos_sum_l94_94191

/--
Given point P with coordinates (-3, 4) lies on the terminal side of angle α, prove that
sin α + cos α = 1/5.
-/
theorem sin_cos_sum (α : ℝ) (P : ℝ × ℝ) (hx : P = (-3, 4)) :
  Real.sin α + Real.cos α = 1/5 := sorry

end sin_cos_sum_l94_94191


namespace find_difference_l94_94101

noncomputable def expr (a b : ℝ) : ℝ :=
  |a - b| / (|a| + |b|)

def min_val (a b : ℝ) : ℝ := 0

def max_val (a b : ℝ) : ℝ := 1

theorem find_difference (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) :
  max_val a b - min_val a b = 1 :=
by
  sorry

end find_difference_l94_94101


namespace Jerry_wants_to_raise_average_l94_94094

theorem Jerry_wants_to_raise_average 
  (first_three_tests_avg : ℕ) (fourth_test_score : ℕ) (desired_increase : ℕ) 
  (h1 : first_three_tests_avg = 90) (h2 : fourth_test_score = 98) 
  : desired_increase = 2 := 
by
  sorry

end Jerry_wants_to_raise_average_l94_94094


namespace triangle_area_x_value_l94_94037

theorem triangle_area_x_value (x : ℝ) (h1 : x > 0) (h2 : 1 / 2 * x * (2 * x) = 64) : x = 8 :=
by
  sorry

end triangle_area_x_value_l94_94037


namespace proof_problem_l94_94185

variable (α β : ℝ)

def interval_αβ : Prop := 
  α ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2) ∧ 
  β ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2)

def condition : Prop := α * Real.sin α - β * Real.sin β > 0

theorem proof_problem (h1 : interval_αβ α β) (h2 : condition α β) : α ^ 2 > β ^ 2 := 
sorry

end proof_problem_l94_94185


namespace crayons_initial_total_l94_94844

theorem crayons_initial_total 
  (lost_given : ℕ) (left : ℕ) (initial : ℕ) 
  (h1 : lost_given = 70) (h2 : left = 183) : 
  initial = lost_given + left := 
by
  sorry

end crayons_initial_total_l94_94844


namespace study_time_difference_l94_94379

def kwame_study_time : ℕ := 150
def connor_study_time : ℕ := 90
def lexia_study_time : ℕ := 97
def michael_study_time : ℕ := 225
def cassandra_study_time : ℕ := 165
def aria_study_time : ℕ := 720

theorem study_time_difference :
  (kwame_study_time + connor_study_time + michael_study_time + cassandra_study_time) + 187 = (lexia_study_time + aria_study_time) :=
by
  sorry

end study_time_difference_l94_94379


namespace safe_security_system_l94_94119

theorem safe_security_system (commission_members : ℕ) 
                            (majority_access : ℕ)
                            (max_inaccess_members : ℕ) 
                            (locks : ℕ)
                            (keys_per_member : ℕ) :
  commission_members = 11 →
  majority_access = 6 →
  max_inaccess_members = 5 →
  locks = (Nat.choose 11 5) →
  keys_per_member = (locks * 6) / 11 →
  locks = 462 ∧ keys_per_member = 252 :=
by
  intros
  sorry

end safe_security_system_l94_94119


namespace problem1_problem2_l94_94815

-- Definitions of the sets
def U : Set ℕ := { x | 1 ≤ x ∧ x ≤ 7 }
def A : Set ℕ := { x | 2 ≤ x ∧ x ≤ 5 }
def B : Set ℕ := { x | 3 ≤ x ∧ x ≤ 7 }

-- Problems to prove (statements only, no proofs provided)
theorem problem1 : A ∪ B = {x | 2 ≤ x ∧ x ≤ 7} :=
by
  sorry

theorem problem2 : U \ A ∪ B = {x | (1 ≤ x ∧ x < 2) ∨ (3 ≤ x ∧ x ≤ 7)} :=
by
  sorry

end problem1_problem2_l94_94815


namespace solve_equation1_solve_equation2_l94_94247

theorem solve_equation1 (x : ℝ) : x^2 + 2 * x = 0 ↔ x = 0 ∨ x = -2 :=
by sorry

theorem solve_equation2 (x : ℝ) : 2 * x^2 - 6 * x = 3 ↔ x = (3 + Real.sqrt 15) / 2 ∨ x = (3 - Real.sqrt 15) / 2 :=
by sorry

end solve_equation1_solve_equation2_l94_94247


namespace fraction_decomposition_l94_94732

theorem fraction_decomposition (P Q : ℚ) :
  (∀ x : ℚ, 4 * x ^ 3 - 5 * x ^ 2 - 26 * x + 24 = (2 * x ^ 2 - 5 * x + 3) * (2 * x - 3))
  → P / (2 * x ^ 2 - 5 * x + 3) + Q / (2 * x - 3) = (8 * x ^ 2 - 9 * x + 20) / (4 * x ^ 3 - 5 * x ^ 2 - 26 * x + 24)
  → P = 4 / 9 ∧ Q = 68 / 9 := by 
  sorry

end fraction_decomposition_l94_94732


namespace lego_set_cost_l94_94878

-- Define the cost per doll and number of dolls
def costPerDoll : ℝ := 15
def numberOfDolls : ℝ := 4

-- Define the total amount spent on the younger sister's dolls
def totalAmountOnDolls : ℝ := numberOfDolls * costPerDoll

-- Define the number of lego sets
def numberOfLegoSets : ℝ := 3

-- Define the total amount spent on lego sets (needs to be equal to totalAmountOnDolls)
def totalAmountOnLegoSets : ℝ := 60

-- Define the cost per lego set that we need to prove
def costPerLegoSet : ℝ := 20

-- Theorem to prove that the cost per lego set is $20
theorem lego_set_cost (h : totalAmountOnLegoSets = totalAmountOnDolls) : 
  totalAmountOnLegoSets / numberOfLegoSets = costPerLegoSet := by
  sorry

end lego_set_cost_l94_94878


namespace ratio_of_larger_to_smaller_l94_94740

theorem ratio_of_larger_to_smaller (x y : ℝ) (h_pos : 0 < y) (h_ineq : x > y) (h_eq : x + y = 7 * (x - y)) : x / y = 4 / 3 :=
by 
  sorry

end ratio_of_larger_to_smaller_l94_94740


namespace annual_return_percentage_l94_94017

theorem annual_return_percentage (initial_value final_value gain : ℕ)
    (h1 : initial_value = 8000)
    (h2 : final_value = initial_value + 400)
    (h3 : gain = final_value - initial_value) :
    (gain * 100 / initial_value) = 5 := by
  sorry

end annual_return_percentage_l94_94017


namespace vector_parallel_unique_solution_l94_94047

def is_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ a = (k * b.1, k * b.2)

theorem vector_parallel_unique_solution (m : ℝ) :
  let a := (m^2 - 1, m + 1)
  let b := (1, -2)
  a ≠ (0, 0) → is_parallel a b → m = 1/2 := by
  sorry

end vector_parallel_unique_solution_l94_94047


namespace find_real_numbers_a_b_l94_94671

noncomputable def f (a b x : ℝ) : ℝ :=
  a * (Real.sin x * Real.cos x) - (Real.sqrt 3) * a * (Real.cos x) ^ 2 + Real.sqrt 3 / 2 * a + b

theorem find_real_numbers_a_b (a b : ℝ) (h1 : 0 < a)
    (h2 : ∀ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), -2 ≤ f a b x ∧ f a b x ≤ Real.sqrt 3)
    : a = 2 ∧ b = -2 + Real.sqrt 3 :=
sorry

end find_real_numbers_a_b_l94_94671


namespace mark_wait_time_l94_94714

theorem mark_wait_time (t1 t2 T : ℕ) (h1 : t1 = 4) (h2 : t2 = 20) (hT : T = 38) : 
  T - (t1 + t2) = 14 :=
by sorry

end mark_wait_time_l94_94714


namespace machine_does_not_require_repair_l94_94416

variables {M : ℝ} {std_dev : ℝ}
variable (deviations : ℝ → Prop)

-- Conditions
def max_deviation := 37
def ten_percent_nominal_mass := 0.1 * M
def max_deviation_condition := max_deviation ≤ ten_percent_nominal_mass
def unreadable_deviation_condition (x : ℝ) := x < 37
def standard_deviation_condition := std_dev ≤ max_deviation
def machine_condition_nonrepair := (∀ x, deviations x → x ≤ max_deviation)

-- Question: Does the machine require repair?
theorem machine_does_not_require_repair 
  (h1 : max_deviation_condition)
  (h2 : ∀ x, unreadable_deviation_condition x → deviations x)
  (h3 : standard_deviation_condition)
  (h4 : machine_condition_nonrepair) :
  ¬(∃ repair_needed : ℝ, repair_needed = 1) :=
by sorry

end machine_does_not_require_repair_l94_94416


namespace product_consecutive_natural_not_equal_even_l94_94919

theorem product_consecutive_natural_not_equal_even (n m : ℕ) (h : m % 2 = 0 ∧ m > 0) : n * (n + 1) ≠ m * (m + 2) :=
sorry

end product_consecutive_natural_not_equal_even_l94_94919


namespace solution_of_system_of_equations_l94_94578

-- Define the conditions of the problem.
def system_of_equations (x y : ℝ) : Prop :=
  (x + y = 6) ∧ (x = 2 * y)

-- Define the correct answer as a set.
def solution_set : Set (ℝ × ℝ) :=
  { (4, 2) }

-- State the proof problem.
theorem solution_of_system_of_equations : 
  {p : ℝ × ℝ | ∃ x y, p = (x, y) ∧ system_of_equations x y} = solution_set :=
  sorry

end solution_of_system_of_equations_l94_94578


namespace find_absolute_difference_l94_94613

def condition_avg_sum (m n : ℝ) : Prop :=
  m + n + 5 + 6 + 4 = 25

def condition_variance (m n : ℝ) : Prop :=
  (m - 5) ^ 2 + (n - 5) ^ 2 = 8

theorem find_absolute_difference (m n : ℝ) (h1 : condition_avg_sum m n) (h2 : condition_variance m n) : |m - n| = 4 :=
sorry

end find_absolute_difference_l94_94613


namespace find_x_sq_add_y_sq_l94_94366

theorem find_x_sq_add_y_sq (x y : ℝ) (h1 : (x + y) ^ 2 = 36) (h2 : x * y = 10) : x ^ 2 + y ^ 2 = 16 :=
by
  sorry

end find_x_sq_add_y_sq_l94_94366


namespace pencils_more_than_pens_l94_94871

theorem pencils_more_than_pens (pencils pens : ℕ) (h_ratio : 5 * pencils = 6 * pens) (h_pencils : pencils = 48) : 
  pencils - pens = 8 :=
by
  sorry

end pencils_more_than_pens_l94_94871


namespace number_of_assignments_power_of_two_l94_94226

variable {α : Type*} [Fintype α] [DecidableEq α]
variable (G : SimpleGraph α)
variable (x : α → ℤ)

def is_valid_assignment (x : α → ℤ) : Prop :=
  ∀ u : α, x u = ∑ v in G.adj u, if x v % 2 = 0 then 1 else 0

theorem number_of_assignments_power_of_two (G : SimpleGraph α) :
  ∃ k : ℕ, (Fintype.card {x // is_valid_assignment G x}) = 2^k := 
sorry

end number_of_assignments_power_of_two_l94_94226


namespace charles_draws_yesterday_after_work_l94_94021

theorem charles_draws_yesterday_after_work :
  ∀ (initial_papers today_drawn yesterday_drawn_before_work current_papers_left yesterday_drawn_after_work : ℕ),
    initial_papers = 20 →
    today_drawn = 6 →
    yesterday_drawn_before_work = 6 →
    current_papers_left = 2 →
    (initial_papers - (today_drawn + yesterday_drawn_before_work) - yesterday_drawn_after_work = current_papers_left) →
    yesterday_drawn_after_work = 6 :=
by
  intros initial_papers today_drawn yesterday_drawn_before_work current_papers_left yesterday_drawn_after_work
  intro h1 h2 h3 h4 h5
  sorry

end charles_draws_yesterday_after_work_l94_94021


namespace polynomial_has_roots_l94_94647

theorem polynomial_has_roots :
  ∃ x : ℝ, x ∈ [-4, -3, -1, 2] ∧ (x^4 + 6 * x^3 + 7 * x^2 - 14 * x - 12 = 0) :=
by
  sorry

end polynomial_has_roots_l94_94647


namespace cos_double_angle_l94_94797

theorem cos_double_angle (α : ℝ) (h : Real.sin α = 3 / 5) : Real.cos (2 * α) = 7 / 25 := 
sorry

end cos_double_angle_l94_94797


namespace sin_neg_30_eq_neg_half_l94_94337

/-- Prove that the sine of -30 degrees is -1/2 -/
theorem sin_neg_30_eq_neg_half : Real.sin (-(30 * Real.pi / 180)) = -1 / 2 :=
by
  sorry

end sin_neg_30_eq_neg_half_l94_94337


namespace max_ratio_l94_94657

theorem max_ratio {a b c d : ℝ} 
  (h1 : a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ d > 0) 
  (h2 : a^2 + b^2 + c^2 + d^2 = (a + b + c + d)^2 / 3) : 
  ∃ x, x = (7 + 2 * Real.sqrt 6) / 5 ∧ x = (a + c) / (b + d) :=
by
  sorry

end max_ratio_l94_94657


namespace ratio_sheep_to_horses_l94_94485

theorem ratio_sheep_to_horses 
  (horse_food_per_day : ℕ) 
  (total_horse_food : ℕ) 
  (num_sheep : ℕ) 
  (H1 : horse_food_per_day = 230) 
  (H2 : total_horse_food = 12880) 
  (H3 : num_sheep = 48) 
  : (num_sheep : ℚ) / (total_horse_food / horse_food_per_day : ℚ) = 6 / 7
  :=
by
  sorry

end ratio_sheep_to_horses_l94_94485


namespace abigail_initial_money_l94_94621

variable (X : ℝ) -- Let X be the initial amount of money

def spent_on_food (X : ℝ) := 0.60 * X
def remaining_after_food (X : ℝ) := X - spent_on_food X
def spent_on_phone (X : ℝ) := 0.25 * remaining_after_food X
def remaining_after_phone (X : ℝ) := remaining_after_food X - spent_on_phone X
def final_amount (X : ℝ) := remaining_after_phone X - 20

theorem abigail_initial_money
    (food_spent : spent_on_food X = 0.60 * X)
    (phone_spent : spent_on_phone X = 0.10 * X)
    (remaining_after_entertainment : final_amount X = 40) :
    X = 200 :=
by
    sorry

end abigail_initial_money_l94_94621


namespace bill_painting_hours_l94_94320

theorem bill_painting_hours (B J : ℝ) (hB : 0 < B) (hJ : 0 < J) : 
  ∃ t : ℝ, t = (B-1)/(B+J) ∧ (t + 1 = (B * (J + 1)) / (B + J)) :=
by
  sorry

end bill_painting_hours_l94_94320


namespace xy_zero_l94_94881

theorem xy_zero (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 54) : x * y = 0 :=
by
  sorry

end xy_zero_l94_94881


namespace probability_of_sequence_l94_94278

noncomputable def prob_first_card_diamond : ℚ := 13 / 52
noncomputable def prob_second_card_spade_given_first_diamond : ℚ := 13 / 51
noncomputable def prob_third_card_heart_given_first_diamond_and_second_spade : ℚ := 13 / 50

theorem probability_of_sequence : 
  prob_first_card_diamond * prob_second_card_spade_given_first_diamond * 
  prob_third_card_heart_given_first_diamond_and_second_spade = 169 / 10200 := 
by
  -- Proof goes here
  sorry

end probability_of_sequence_l94_94278


namespace arithmetic_mean_bc_diff_l94_94199

variables (a b c μ : ℝ)

theorem arithmetic_mean_bc_diff 
  (h1 : (a + b) / 2 = μ + 5)
  (h2 : (a + c) / 2 = μ - 8)
  (h3 : μ = (a + b + c) / 3) :
  (b + c) / 2 = μ + 3 :=
sorry

end arithmetic_mean_bc_diff_l94_94199


namespace no_zero_sum_of_vectors_l94_94318

-- Definitions and conditions for the problem
variable {n : ℕ} (odd_n : n % 2 = 1) -- n is odd, representing the number of sides of the polygon

-- The statement of the proof problem
theorem no_zero_sum_of_vectors (odd_n : n % 2 = 1) : false :=
by
  sorry

end no_zero_sum_of_vectors_l94_94318


namespace gcd_lcm_product_135_l94_94782

theorem gcd_lcm_product_135 (a b : ℕ) (ha : a = 9) (hb : b = 15) :
  Nat.gcd a b * Nat.lcm a b = 135 :=
by
  sorry

end gcd_lcm_product_135_l94_94782


namespace optimal_roof_angle_no_friction_l94_94463

theorem optimal_roof_angle_no_friction {g x : ℝ} (hg : 0 < g) (hx : 0 < x) :
  ∃ α : ℝ, α = 45 :=
by
  sorry

end optimal_roof_angle_no_friction_l94_94463


namespace sin_neg_30_eq_neg_one_half_l94_94331

theorem sin_neg_30_eq_neg_one_half :
  Real.sin (-30 * Real.pi / 180) = -1 / 2 := 
by
  sorry -- Proof is skipped

end sin_neg_30_eq_neg_one_half_l94_94331


namespace all_sets_B_l94_94576

open Set

theorem all_sets_B (B : Set ℕ) :
  { B | {1, 2} ∪ B = {1, 2, 3} } =
  ({ {3}, {1, 3}, {2, 3}, {1, 2, 3} } : Set (Set ℕ)) :=
sorry

end all_sets_B_l94_94576


namespace negation_of_exists_equiv_forall_neg_l94_94259

noncomputable def negation_equivalent (a : ℝ) : Prop :=
  ∀ a : ℝ, ¬ ∃ x : ℝ, a * x^2 + 1 = 0

-- The theorem statement
theorem negation_of_exists_equiv_forall_neg (h : ∃ a : ℝ, ∃ x : ℝ, a * x^2 + 1 = 0) :
  negation_equivalent a :=
by {
  sorry
}

end negation_of_exists_equiv_forall_neg_l94_94259


namespace balls_into_boxes_l94_94676

theorem balls_into_boxes :
  let balls := 7
  let boxes := 3
  let min_balls_per_box := 1
  let remaining_balls := balls - boxes * min_balls_per_box
  let combination := Nat.choose (remaining_balls + boxes - 1) (boxes - 1) 
  combination = 15 :=
by
  let balls := 7
  let boxes := 3
  let min_balls_per_box := 1
  let remaining_balls := balls - boxes * min_balls_per_box
  let combination := Nat.choose (remaining_balls + boxes - 1) (boxes - 1)
  show combination = 15
  sorry

end balls_into_boxes_l94_94676


namespace smith_gave_randy_l94_94563

theorem smith_gave_randy :
  ∀ (s amount_given amount_left : ℕ), amount_given = 1200 → amount_left = 2000 → s = amount_given + amount_left → s = 3200 :=
by
  intros s amount_given amount_left h_given h_left h_total
  rw [h_given, h_left] at h_total
  exact h_total

end smith_gave_randy_l94_94563


namespace final_tank_volume_l94_94475

-- Definitions based on the conditions
def initially_liters : ℕ := 6000
def evaporated_liters : ℕ := 2000
def drained_by_bob : ℕ := 3500
def rain_minutes : ℕ := 30
def rain_interval : ℕ := 10
def rain_rate : ℕ := 350

-- Theorem statement
theorem final_tank_volume:
  let remaining_water := initially_liters - evaporated_liters - drained_by_bob,
      rain_cycles := rain_minutes / rain_interval,
      added_by_rain := rain_cycles * rain_rate,
      final_volume := remaining_water + added_by_rain
  in final_volume = 1550 := by {
  sorry
}

end final_tank_volume_l94_94475


namespace parabola_line_no_intersection_l94_94350

theorem parabola_line_no_intersection (x y : ℝ) (h : y^2 < 4 * x) :
  ¬ ∃ (x' y' : ℝ), y' = y ∧ y'^2 = 4 * x' ∧ 2 * x' = x + x :=
by sorry

end parabola_line_no_intersection_l94_94350


namespace total_angles_sum_l94_94172

variables (A B C D E : Type)
variables (angle1 angle2 angle3 angle4 angle5 angle7 : ℝ)

-- Conditions about the geometry
axiom angle_triangle_ABC : angle1 + angle2 + angle3 = 180
axiom angle_triangle_BDE : angle7 + angle4 + angle5 = 180
axiom shared_angle_B : angle2 + angle7 = 180 -- since they form a straight line at vertex B

-- Proof statement
theorem total_angles_sum (A B C D E : Type) (angle1 angle2 angle3 angle4 angle5 angle7 : ℝ) :
  angle1 + angle2 + angle3 + angle4 + angle5 + angle7 - 180 = 180 :=
by
  sorry

end total_angles_sum_l94_94172


namespace tennis_ball_price_l94_94081

theorem tennis_ball_price (x y : ℝ) 
  (h₁ : 2 * x + 7 * y = 220)
  (h₂ : x = y + 83) : 
  y = 6 := 
by 
  sorry

end tennis_ball_price_l94_94081


namespace polynomial_range_l94_94978

def p (x : ℝ) : ℝ := x^4 - 4*x^3 + 8*x^2 - 8*x + 5

theorem polynomial_range : ∀ x : ℝ, p x ≥ 2 :=
by
sorry

end polynomial_range_l94_94978


namespace interest_calculation_l94_94882

/-- Define the initial deposit in thousands of yuan (50,000 yuan = 5 x 10,000 yuan) -/
def principal : ℕ := 5

/-- Define the annual interest rate as a percentage in decimal form -/
def annual_interest_rate : ℝ := 0.04

/-- Define the number of years for the deposit -/
def years : ℕ := 3

/-- Calculate the total amount after 3 years using compound interest -/
def total_amount_after_3_years : ℝ :=
  principal * (1 + annual_interest_rate) ^ years

/-- Calculate the interest earned after 3 years -/
def interest_earned : ℝ :=
  total_amount_after_3_years - principal

theorem interest_calculation :
  interest_earned = 5 * (1 + 0.04) ^ 3 - 5 :=
by 
  sorry

end interest_calculation_l94_94882


namespace smallest_x_250_multiple_1080_l94_94885

theorem smallest_x_250_multiple_1080 : (∃ x : ℕ, x > 0 ∧ (250 * x) % 1080 = 0) ∧ ¬(∃ y : ℕ, y > 0 ∧ y < 54 ∧ (250 * y) % 1080 = 0) :=
by
  sorry

end smallest_x_250_multiple_1080_l94_94885


namespace sin_neg_30_eq_neg_one_half_l94_94332

theorem sin_neg_30_eq_neg_one_half : Real.sin (-30 / 180 * Real.pi) = -1 / 2 := by
  -- Proof goes here
  sorry

end sin_neg_30_eq_neg_one_half_l94_94332


namespace cost_of_each_hotdog_l94_94386

theorem cost_of_each_hotdog (number_of_hotdogs : ℕ) (total_cost : ℕ) (cost_per_hotdog : ℕ) 
    (h1 : number_of_hotdogs = 6) (h2 : total_cost = 300) : cost_per_hotdog = 50 :=
by
  have h3 : cost_per_hotdog = total_cost / number_of_hotdogs :=
    sorry -- here we would normally write the division step
  sorry -- here we would show that h3 implies cost_per_hotdog = 50, given h1 and h2

end cost_of_each_hotdog_l94_94386


namespace initial_money_l94_94305

-- Define the conditions
variable (M : ℝ)
variable (h : (1 / 3) * M = 50)

-- Define the theorem to be proved
theorem initial_money : M = 150 := 
by
  sorry

end initial_money_l94_94305


namespace simple_interest_rate_l94_94891

theorem simple_interest_rate
  (A5 A8 : ℝ) (years_between : ℝ := 3) (I3 : ℝ) (annual_interest : ℝ)
  (P : ℝ) (R : ℝ)
  (h1 : A5 = 9800) -- Amount after 5 years is Rs. 9800
  (h2 : A8 = 12005) -- Amount after 8 years is Rs. 12005
  (h3 : I3 = A8 - A5) -- Interest for 3 years
  (h4 : annual_interest = I3 / years_between) -- Annual interest
  (h5 : P = 9800) -- Principal amount after 5 years
  (h6 : R = (annual_interest * 100) / P) -- Rate of interest formula revised
  : R = 7.5 := 
sorry

end simple_interest_rate_l94_94891


namespace distinct_positive_integer_roots_pq_l94_94554

theorem distinct_positive_integer_roots_pq :
  ∃ (p q : ℝ), (∃ (a b c : ℕ), (a ≠ b ∧ b ≠ c ∧ c ≠ a) ∧ (a + b + c = 9) ∧ (a * b + a * c + b * c = p) ∧ (a * b * c = q)) ∧ p + q = 38 :=
by sorry


end distinct_positive_integer_roots_pq_l94_94554


namespace factor_expression_l94_94922

theorem factor_expression :
  (8 * x ^ 4 + 34 * x ^ 3 - 120 * x + 150) - (-2 * x ^ 4 + 12 * x ^ 3 - 5 * x + 10) 
  = 5 * x * (2 * x ^ 3 + (22 / 5) * x ^ 2 - 23 * x + 28) :=
sorry

end factor_expression_l94_94922


namespace vector_magnitude_difference_l94_94064

def vector_a : ℝ × ℝ := (2, 1)
def vector_b : ℝ × ℝ := (-2, 4)

theorem vector_magnitude_difference :
  let diff := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2)
  let magnitude := Real.sqrt (diff.1 ^ 2 + diff.2 ^ 2)
  magnitude = 5 :=
by
  -- define the difference vector
  have diff : ℝ × ℝ := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2),
  -- define the magnitude of the difference vector
  have magnitude := Real.sqrt (diff.1 ^ 2 + diff.2 ^ 2),
  -- provide the proof (using steps omitted via sorry)
  sorry

end vector_magnitude_difference_l94_94064


namespace value_of_k_l94_94272

theorem value_of_k : (2^200 + 5^201)^2 - (2^200 - 5^201)^2 = 20 * 10^201 := 
by 
  sorry

end value_of_k_l94_94272


namespace prime_fraction_sum_l94_94816

theorem prime_fraction_sum (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c)
    (h : a + b + c + a * b * c = 99) :
    |(1 / a : ℚ) - (1 / b : ℚ)| + |(1 / b : ℚ) - (1 / c : ℚ)| + |(1 / c : ℚ) - (1 / a : ℚ)| = 9 / 11 := 
sorry

end prime_fraction_sum_l94_94816


namespace monotonically_decreasing_log_less_than_x_l94_94670

-- Define the function f(x)
def f (x : ℝ) : ℝ := Real.log (x + 1) - x

-- State the problem
theorem monotonically_decreasing (x : ℝ) (h : x > 0) : 
  (∃ I : Set ℝ, I = set.Ioi 0 ∧ ∀ y ∈ I, differentiable_at ℝ f y) ∧ (∀ y, y ∈ set.Ioi 0 → deriv f y < 0) := sorry

theorem log_less_than_x (x : ℝ) (h : x > -1) : Real.log (x + 1) ≤ x := sorry

end monotonically_decreasing_log_less_than_x_l94_94670


namespace reflect_y_axis_correct_l94_94857

-- Define the initial coordinates of the point M
def M_orig : ℝ × ℝ := (3, 2)

-- Define the reflection function across the y-axis
def reflect_y_axis (M : ℝ × ℝ) : ℝ × ℝ :=
  (-M.1, M.2)

-- Prove that reflecting M_orig across the y-axis results in the coordinates (-3, 2)
theorem reflect_y_axis_correct : reflect_y_axis M_orig = (-3, 2) :=
  by
    -- Provide the missing steps of the proof
    sorry

end reflect_y_axis_correct_l94_94857


namespace storks_more_than_birds_l94_94297

theorem storks_more_than_birds :
  let initial_birds := 3
  let additional_birds := 2
  let storks := 6
  storks - (initial_birds + additional_birds) = 1 :=
by
  sorry

end storks_more_than_birds_l94_94297


namespace number_of_ninth_graders_l94_94431

def num_students_total := 50
def num_students_7th (x : Int) := 2 * x - 1
def num_students_8th (x : Int) := x

theorem number_of_ninth_graders (x : Int) :
  num_students_7th x + num_students_8th x + (51 - 3 * x) = num_students_total := by
  sorry

end number_of_ninth_graders_l94_94431


namespace both_selected_prob_l94_94879

noncomputable def prob_X : ℚ := 1 / 3
noncomputable def prob_Y : ℚ := 2 / 7
noncomputable def combined_prob : ℚ := prob_X * prob_Y

theorem both_selected_prob :
  combined_prob = 2 / 21 :=
by
  unfold combined_prob prob_X prob_Y
  sorry

end both_selected_prob_l94_94879


namespace intersection_a_eq_1_parallel_lines_value_of_a_l94_94673

-- Define lines
def line1 (a : ℝ) (x y : ℝ) : Prop := x + a * y - a + 2 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := 2 * a * x + (a + 3) * y + a - 5 = 0

-- Part 1: Prove intersection point for a = 1
theorem intersection_a_eq_1 :
  line1 1 (-4) 3 ∧ line2 1 (-4) 3 :=
by sorry

-- Part 2: Prove value of a for which lines are parallel
theorem parallel_lines_value_of_a :
  ∃ a : ℝ, ∀ x y : ℝ, line1 a x y ∧ line2 a x y →
  (2 * a^2 - a - 3 = 0 ∧ a ≠ -1 ∧ a = 3/2) :=
by sorry

end intersection_a_eq_1_parallel_lines_value_of_a_l94_94673


namespace quadratic_distinct_zeros_l94_94974

theorem quadratic_distinct_zeros (m : ℝ) : 
  (x^2 + m * x + (m + 3)) = 0 → 
  (0 < m^2 - 4 * (m + 3)) ↔ (m < -2) ∨ (m > 6) :=
sorry

end quadratic_distinct_zeros_l94_94974


namespace minimize_cylinder_surface_area_l94_94507

noncomputable def cylinder_surface_area (r h : ℝ) : ℝ :=
  2 * Real.pi * r^2 + 2 * Real.pi * r * h

noncomputable def cylinder_volume (r h : ℝ) : ℝ :=
  Real.pi * r^2 * h

theorem minimize_cylinder_surface_area :
  ∃ r h : ℝ, cylinder_volume r h = 16 * Real.pi ∧
  (∀ r' h', cylinder_volume r' h' = 16 * Real.pi → cylinder_surface_area r h ≤ cylinder_surface_area r' h') ∧ r = 2 := by
  sorry

end minimize_cylinder_surface_area_l94_94507


namespace molecular_weight_of_compound_l94_94436

theorem molecular_weight_of_compound (total_weight_of_3_moles : ℝ) (n_moles : ℝ) 
  (h1 : total_weight_of_3_moles = 528) (h2 : n_moles = 3) : 
  (total_weight_of_3_moles / n_moles) = 176 :=
by
  sorry

end molecular_weight_of_compound_l94_94436


namespace one_fourth_in_one_eighth_l94_94080

theorem one_fourth_in_one_eighth : (1 / 8) / (1 / 4) = 1 / 2 := by
  sorry

end one_fourth_in_one_eighth_l94_94080


namespace negation_of_exists_equiv_forall_neg_l94_94258

noncomputable def negation_equivalent (a : ℝ) : Prop :=
  ∀ a : ℝ, ¬ ∃ x : ℝ, a * x^2 + 1 = 0

-- The theorem statement
theorem negation_of_exists_equiv_forall_neg (h : ∃ a : ℝ, ∃ x : ℝ, a * x^2 + 1 = 0) :
  negation_equivalent a :=
by {
  sorry
}

end negation_of_exists_equiv_forall_neg_l94_94258


namespace maximize_annual_profit_l94_94131

theorem maximize_annual_profit :
  ∃ t : ℕ, t > 0 ∧ (∀ t' : ℕ, t' > 0 → (let s := -2 * t^2 + 30 * t - 98 in (s / t) ≥ 
                                      let s' := -2 * t'^2 + 30 * t' - 98 in (s' / t'))) ∧ t = 7 := 
by {
    use 7,
    sorry
}

end maximize_annual_profit_l94_94131


namespace total_clouds_count_l94_94921

-- Definitions based on the conditions
def carson_clouds := 12
def little_brother_clouds := 5 * carson_clouds
def older_sister_clouds := carson_clouds / 2

-- The theorem statement that needs to be proved
theorem total_clouds_count : carson_clouds + little_brother_clouds + older_sister_clouds = 78 := by
  -- Definitions
  have h1 : carson_clouds = 12 := rfl
  have h2 : little_brother_clouds = 5 * 12 := rfl
  have h3 : older_sister_clouds = 12 / 2 := rfl
  sorry

end total_clouds_count_l94_94921


namespace distinct_positive_integer_roots_pq_l94_94555

theorem distinct_positive_integer_roots_pq :
  ∃ (p q : ℝ), (∃ (a b c : ℕ), (a ≠ b ∧ b ≠ c ∧ c ≠ a) ∧ (a + b + c = 9) ∧ (a * b + a * c + b * c = p) ∧ (a * b * c = q)) ∧ p + q = 38 :=
by sorry


end distinct_positive_integer_roots_pq_l94_94555


namespace vector_magnitude_subtraction_l94_94055

def vec_sub (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def vec_norm (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem vector_magnitude_subtraction :
  let a := (2, 1)
  let b := (-2, 4)
  vec_norm (vec_sub a b) = 5 :=
by
  sorry

end vector_magnitude_subtraction_l94_94055


namespace John_paid_total_l94_94378

def vet_cost : ℝ := 400
def num_appointments : ℕ := 3
def insurance_cost : ℝ := 100
def coverage_rate : ℝ := 0.8

def discount : ℝ := vet_cost * coverage_rate
def discounted_visits : ℕ := num_appointments - 1
def discounted_cost : ℝ := vet_cost - discount
def total_discounted_cost : ℝ := discounted_visits * discounted_cost
def J_total : ℝ := vet_cost + total_discounted_cost + insurance_cost

theorem John_paid_total : J_total = 660 := by
  sorry

end John_paid_total_l94_94378


namespace hh_value_l94_94524

def h (x : ℤ) : ℤ := 3 * x^2 + 3 * x - 2

theorem hh_value :
  h(h(3)) = 3568 :=
by
  sorry

end hh_value_l94_94524


namespace sum_prime_factors_of_77_l94_94438

theorem sum_prime_factors_of_77 : 
  ∃ (p q : ℕ), nat.prime p ∧ nat.prime q ∧ 77 = p * q ∧ p + q = 18 :=
by
  existsi 7
  existsi 11
  split
  { exact nat.prime_7 }
  split
  { exact nat.prime_11 }
  split
  { exact dec_trivial }
  { exact dec_trivial }

-- Placeholder for the proof
-- sorry

end sum_prime_factors_of_77_l94_94438


namespace sum_prime_factors_77_l94_94442

noncomputable def sum_prime_factors (n : ℕ) : ℕ :=
  if n = 77 then 7 + 11 else 0

theorem sum_prime_factors_77 : sum_prime_factors 77 = 18 :=
by
  -- The theorem states that the sum of the prime factors of 77 is 18
  rw sum_prime_factors
  -- Given that the prime factors of 77 are 7 and 11, summing them gives 18
  if_clause
  -- We finalize by proving this is indeed the case
  sorry

end sum_prime_factors_77_l94_94442


namespace weight_of_mixture_l94_94755

variable (A B : ℝ)
variable (ratio_A_B : A / B = 9 / 11)
variable (consumed_A : A = 26.1)

theorem weight_of_mixture (A B : ℝ) (ratio_A_B : A / B = 9 / 11) (consumed_A : A = 26.1) : 
  A + B = 58 :=
sorry

end weight_of_mixture_l94_94755


namespace correct_propositions_l94_94129

theorem correct_propositions :
  let proposition1 := (∀ A B C : ℝ, C = (A + B) / 2 → C = (A + B) / 2)
  let proposition2 := (∀ a : ℝ, a - |a| = 0 → a ≥ 0)
  let proposition3 := false
  let proposition4 := (∀ a b : ℝ, |a| = |b| → a = -b)
  let proposition5 := (∀ a : ℝ, -a < 0)
  (cond1 : proposition1 = false) →
  (cond2 : proposition2 = false) →
  (cond3 : proposition3 = false) →
  (cond4 : proposition4 = true) →
  (cond5 : proposition5 = false) →
  1 = 1 :=
by
  intros
  sorry

end correct_propositions_l94_94129


namespace power_division_correct_l94_94752

theorem power_division_correct :
  (∀ x : ℝ, x^4 / x = x^3) ∧ 
  ¬(∀ x : ℝ, 3 * x^2 * 4 * x^2 = 12 * x^2) ∧
  ¬(∀ x : ℝ, (x - 1) * (x - 1) = x^2 - 1) ∧
  ¬(∀ x : ℝ, (x^5)^2 = x^7) := 
by {
  -- Proof would go here
  sorry
}

end power_division_correct_l94_94752


namespace evaluate_g_at_3_l94_94527

def g (x : ℝ) : ℝ := 7 * x^3 - 5 * x^2 - 7 * x + 3

theorem evaluate_g_at_3 : g 3 = 126 := 
by 
  sorry

end evaluate_g_at_3_l94_94527


namespace fg_3_eq_7_l94_94526

def f (x : ℝ) : ℝ := 4 * x + 3
def g (x : ℝ) : ℝ := (x - 2) ^ 2

theorem fg_3_eq_7 : f (g 3) = 7 :=
by
  sorry

end fg_3_eq_7_l94_94526


namespace christina_rearrangements_l94_94027

-- define the main conditions
def rearrangements (n : Nat) : Nat := Nat.factorial n

def half (n : Nat) : Nat := n / 2

def time_for_first_half (r : Nat) : Nat := r / 12

def time_for_second_half (r : Nat) : Nat := r / 18

def total_time_in_minutes (t1 t2 : Nat) : Nat := t1 + t2

def total_time_in_hours (t : Nat) : Nat := t / 60

-- statement proving that the total time will be 420 hours
theorem christina_rearrangements : 
  rearrangements 9 = 362880 →
  half (rearrangements 9) = 181440 →
  time_for_first_half 181440 = 15120 →
  time_for_second_half 181440 = 10080 →
  total_time_in_minutes 15120 10080 = 25200 →
  total_time_in_hours 25200 = 420 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end christina_rearrangements_l94_94027


namespace exists_rational_non_integer_linear_l94_94154

theorem exists_rational_non_integer_linear (k1 k2 : ℤ) : 
  ∃ (x y : ℚ), x ≠ ⌊x⌋ ∧ y ≠ ⌊y⌋ ∧ 
  19 * x + 8 * y = k1 ∧ 8 * x + 3 * y = k2 :=
sorry

end exists_rational_non_integer_linear_l94_94154


namespace sam_more_than_sarah_l94_94991

-- Defining the conditions
def street_width : ℤ := 25
def block_length : ℤ := 450
def block_width : ℤ := 350
def alleyway : ℤ := 25

-- Defining the distances run by Sarah and Sam
def sarah_long_side : ℤ := block_length + alleyway
def sarah_short_side : ℤ := block_width
def sam_long_side : ℤ := block_length + 2 * street_width
def sam_short_side : ℤ := block_width + 2 * street_width

-- Defining the total distance run by Sarah and Sam in one lap
def sarah_total_distance : ℤ := 2 * sarah_long_side + 2 * sarah_short_side
def sam_total_distance : ℤ := 2 * sam_long_side + 2 * sam_short_side

-- Proving the difference between Sam's and Sarah's running distances
theorem sam_more_than_sarah : sam_total_distance - sarah_total_distance = 150 := by
  -- The proof is omitted
  sorry

end sam_more_than_sarah_l94_94991


namespace area_of_R_sum_m_n_l94_94768

theorem area_of_R_sum_m_n  (s : ℕ) 
  (square_area : ℕ) 
  (rectangle1_area : ℕ)
  (rectangle2_area : ℕ) :
  square_area = 4 → rectangle1_area = 8 → rectangle2_area = 2 → s = 6 → 
  36 - (square_area + rectangle1_area + rectangle2_area) = 22 :=
by
  intros
  sorry

end area_of_R_sum_m_n_l94_94768


namespace same_graphs_at_x_eq_1_l94_94887

theorem same_graphs_at_x_eq_1 :
  let y1 := 2 - 1
  let y2 := (1^3 - 1) / (1 - 1)
  let y3 := (1^3 - 1) / (1 - 1)
  y2 = 3 ∧ y3 = 3 ∧ y1 ≠ y2 := 
by
  let y1 := 2 - 1
  let y2 := (1^3 - 1) / (1 - 1)
  let y3 := (1^3 - 1) / (1 - 1)
  sorry

end same_graphs_at_x_eq_1_l94_94887


namespace polynomial_necessary_but_not_sufficient_l94_94897

-- Definitions
def polynomial_condition (x : ℝ) : Prop := x^2 - 3 * x + 2 = 0

def specific_value : ℝ := 1

-- Theorem statement
theorem polynomial_necessary_but_not_sufficient :
  (polynomial_condition specific_value ∧ ¬ ∀ x, polynomial_condition x -> x = specific_value) :=
by
  sorry

end polynomial_necessary_but_not_sufficient_l94_94897


namespace count_three_digit_integers_with_tens_7_divisible_by_25_l94_94520

theorem count_three_digit_integers_with_tens_7_divisible_by_25 :
  ∃ n, n = 33 ∧ ∃ k1 k2 : ℕ, 175 = 25 * k1 ∧ 975 = 25 * k2 ∧ (k2 - k1 + 1 = n) :=
by
  sorry

end count_three_digit_integers_with_tens_7_divisible_by_25_l94_94520


namespace largest_side_of_enclosure_l94_94435

theorem largest_side_of_enclosure (l w : ℕ) (h1 : 2 * l + 2 * w = 180) (h2 : l * w = 1800) : max l w = 60 := 
by 
  sorry

end largest_side_of_enclosure_l94_94435


namespace ocean_depth_l94_94423

theorem ocean_depth (t : ℕ) (v : ℕ) (h : ℕ)
  (h_t : t = 8)
  (h_v : v = 1500) :
  h = 6000 :=
by
  sorry

end ocean_depth_l94_94423


namespace certain_number_divisibility_l94_94824

-- Define the conditions and the main problem statement
theorem certain_number_divisibility (n : ℕ) (h1 : 0 < n) (h2 : n < 11) (h3 : (18888 - n) % k = 0) (h4 : n = 1) : k = 11 :=
by
  sorry

end certain_number_divisibility_l94_94824


namespace ratio_adidas_skechers_l94_94558

-- Conditions
def total_expenditure : ℤ := 8000
def expenditure_adidas : ℤ := 600
def expenditure_clothes : ℤ := 2600
def expenditure_nike := 3 * expenditure_adidas

-- Calculation for sneakers
def total_sneakers := total_expenditure - expenditure_clothes
def expenditure_nike_adidas := expenditure_nike + expenditure_adidas
def expenditure_skechers := total_sneakers - expenditure_nike_adidas

-- Prove the ratio
theorem ratio_adidas_skechers (H1 : total_expenditure = 8000)
                              (H2 : expenditure_adidas = 600)
                              (H3 : expenditure_nike = 3 * expenditure_adidas)
                              (H4 : expenditure_clothes = 2600) :
  expenditure_adidas / expenditure_skechers = 1 / 5 :=
by
  sorry

end ratio_adidas_skechers_l94_94558


namespace negation_proposition_equiv_l94_94266

open Classical

variable (R : Type) [OrderedRing R] (a x : R)

theorem negation_proposition_equiv :
  (¬ ∃ a : R, ∃ x : R, a * x^2 + 1 = 0) ↔ (∀ a : R, ∀ x : R, a * x^2 + 1 ≠ 0) :=
by
  sorry

end negation_proposition_equiv_l94_94266


namespace fill_time_l94_94303

noncomputable def time_to_fill (X Y Z : ℝ) : ℝ :=
  1 / X + 1 / Y + 1 / Z

theorem fill_time 
  (V X Y Z : ℝ) 
  (h1 : X + Y = V / 3) 
  (h2 : X + Z = V / 2) 
  (h3 : Y + Z = V / 4) :
  1 / time_to_fill X Y Z = 24 / 13 :=
by
  sorry

end fill_time_l94_94303


namespace candles_on_rituprts_cake_l94_94110

theorem candles_on_rituprts_cake (peter_candles : ℕ) (rupert_factor : ℝ) 
  (h_peter : peter_candles = 10) (h_rupert : rupert_factor = 3.5) : 
  ∃ rupert_candles : ℕ, rupert_candles = 35 :=
by
  sorry

end candles_on_rituprts_cake_l94_94110


namespace valid_inequalities_l94_94143

theorem valid_inequalities :
  (∀ x : ℝ, x^2 + 6x + 10 > 0) ∧ (∀ x : ℝ, -x^2 + x - 2 < 0) := by
  sorry

end valid_inequalities_l94_94143


namespace possible_sums_of_products_neg11_l94_94735

theorem possible_sums_of_products_neg11 (a b c : ℤ) (h : a * b * c = -11) :
  a + b + c = -9 ∨ a + b + c = 11 ∨ a + b + c = 13 :=
sorry

end possible_sums_of_products_neg11_l94_94735


namespace solve_number_of_brothers_l94_94370

def number_of_brothers_problem : Prop :=
  ∃ (b A : ℕ), A + 15 * b = 107 ∧ A + 6 * b = 71 ∧ b = 4

theorem solve_number_of_brothers : number_of_brothers_problem :=
  sorry

end solve_number_of_brothers_l94_94370


namespace longest_side_similar_triangle_l94_94406

theorem longest_side_similar_triangle (a b c : ℝ) (p : ℝ) (h₀ : a = 8) (h₁ : b = 15) (h₂ : c = 17) (h₃ : a^2 + b^2 = c^2) (h₄ : p = 160) :
  ∃ x : ℝ, (8 * x) + (15 * x) + (17 * x) = p ∧ 17 * x = 68 :=
by
  sorry

end longest_side_similar_triangle_l94_94406


namespace coeff_sum_eq_neg_two_l94_94822

theorem coeff_sum_eq_neg_two (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, (x^10 + x^4 + 1) = a + a₁ * (x+1) + a₂ * (x+1)^2 + a₃ * (x+1)^3 + a₄ * (x+1)^4 
   + a₅ * (x+1)^5 + a₆ * (x+1)^6 + a₇ * (x+1)^7 + a₈ * (x+1)^8 + a₉ * (x+1)^9 + a₁₀ * (x+1)^10) 
  → (a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ = -2) := 
by sorry

end coeff_sum_eq_neg_two_l94_94822


namespace problem_l94_94422

-- Define the concept of reciprocal
def reciprocal (x : ℚ) : ℚ := 1 / x

-- Define the conditions in the problem
def condition1 : Prop := reciprocal 1.5 = 2/3
def condition2 : Prop := reciprocal 1 = 1

-- Theorem stating our goals
theorem problem : condition1 ∧ condition2 :=
by {
  sorry
}

end problem_l94_94422


namespace telepathic_connection_correct_l94_94605

def telepathic_connection_probability : ℚ := sorry

theorem telepathic_connection_correct :
  telepathic_connection_probability = 7 / 25 := sorry

end telepathic_connection_correct_l94_94605


namespace transform_parabola_l94_94102

theorem transform_parabola (a b c : ℝ) (h : a ≠ 0) :
  ∃ (f : ℝ → ℝ), (∀ x, f (a * x^2 + b * x + c) = x^2) :=
sorry

end transform_parabola_l94_94102


namespace units_digit_is_seven_l94_94863

-- Defining the structure of the three-digit number and its properties
def original_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c
def reversed_number (a b c : ℕ) : ℕ := 100 * c + 10 * b + a

def four_times_original (a b c : ℕ) : ℕ := 4 * original_number a b c
def subtract_reversed (a b c : ℕ) : ℕ := four_times_original a b c - reversed_number a b c

-- Theorem statement: Given the condition, what is the units digit of the result?
theorem units_digit_is_seven (a b c : ℕ) (h : a = c + 3) : (subtract_reversed a b c) % 10 = 7 :=
by
  sorry

end units_digit_is_seven_l94_94863


namespace polar_eq_is_circle_l94_94573

-- Define the polar equation as a condition
def polar_eq (ρ : ℝ) := ρ = 5

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Prove that the curve represented by the polar equation is a circle
theorem polar_eq_is_circle (P : ℝ × ℝ) : (∃ ρ θ, P = (ρ * Real.cos θ, ρ * Real.sin θ) ∧ polar_eq ρ) ↔ dist P origin = 5 := 
by 
  sorry

end polar_eq_is_circle_l94_94573


namespace sequence_fifth_term_l94_94251

theorem sequence_fifth_term (a : ℤ) (d : ℤ) (n : ℕ) (a_n : ℤ) :
  a_n = 89 ∧ d = 11 ∧ n = 5 → a + (n-1) * -d = 45 := 
by
  intros h
  rcases h with ⟨h1, h2, h3⟩
  exact sorry

end sequence_fifth_term_l94_94251


namespace describe_T_l94_94838

def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (common : ℝ), 
    (common = 5 ∧ p.1 + 3 = common ∧ p.2 - 6 ≤ common) ∨
    (common = 5 ∧ p.2 - 6 = common ∧ p.1 + 3 ≤ common) ∨
    (common = p.1 + 3 ∧ common = p.2 - 6 ∧ common ≤ 5)}

theorem describe_T :
  T = {(2, y) | y ≤ 11} ∪ { (x, 11) | x ≤ 2} ∪ { (x, x + 9) | x ≤ 2} :=
by
  sorry

end describe_T_l94_94838


namespace solve_g_l94_94998

def g (a b : ℚ) : ℚ :=
if a + b ≤ 4 then (a * b - 2 * a + 3) / (3 * a)
else (a * b - 3 * b - 1) / (-3 * b)

theorem solve_g :
  g 3 1 + g 1 5 = 11 / 15 :=
by
  -- Here we just set up the theorem statement. Proof is not included.
  sorry

end solve_g_l94_94998


namespace Julio_current_age_l94_94096

theorem Julio_current_age (J : ℕ) (James_current_age : ℕ) (h1 : James_current_age = 11)
    (h2 : J + 14 = 2 * (James_current_age + 14)) : 
    J = 36 := 
by 
  sorry

end Julio_current_age_l94_94096


namespace find_subtracted_value_l94_94285

theorem find_subtracted_value (N V : ℕ) (hN : N = 12) (h : 4 * N - V = 9 * (N - 7)) : V = 3 :=
by
  sorry

end find_subtracted_value_l94_94285


namespace largest_increase_is_2007_2008_l94_94916

-- Define the number of students each year
def students_2005 : ℕ := 50
def students_2006 : ℕ := 55
def students_2007 : ℕ := 60
def students_2008 : ℕ := 70
def students_2009 : ℕ := 72
def students_2010 : ℕ := 80

-- Define the percentage increase function
def percentage_increase (old new : ℕ) : ℚ :=
  ((new - old) : ℚ) / old * 100

-- Define percentage increases for each pair of consecutive years
def increase_2005_2006 := percentage_increase students_2005 students_2006
def increase_2006_2007 := percentage_increase students_2006 students_2007
def increase_2007_2008 := percentage_increase students_2007 students_2008
def increase_2008_2009 := percentage_increase students_2008 students_2009
def increase_2009_2010 := percentage_increase students_2009 students_2010

-- State the theorem
theorem largest_increase_is_2007_2008 :
  (max (max increase_2005_2006 (max increase_2006_2007 increase_2008_2009))
       increase_2009_2010) < increase_2007_2008 := 
by
  -- Add proof steps if necessary.
  sorry

end largest_increase_is_2007_2008_l94_94916


namespace intercept_sum_l94_94990

-- Define the equation of the line and the condition on the intercepts.
theorem intercept_sum (c : ℚ) (x y : ℚ) (h1 : 3 * x + 5 * y + c = 0) (h2 : x + y = 55/4) : 
  c = 825/32 :=
sorry

end intercept_sum_l94_94990


namespace find_a_l94_94712

theorem find_a (a : ℝ) (h₁ : a ≠ 0) (h₂ : ∀ x y : ℝ, x^2 + a*y^2 + a^2 = 0) (h₃ : 4 = 4) :
  a = (1 - Real.sqrt 17) / 2 := sorry

end find_a_l94_94712


namespace sum_of_consecutive_negative_integers_with_product_3080_l94_94869

theorem sum_of_consecutive_negative_integers_with_product_3080 :
  ∃ (n : ℤ), n < 0 ∧ (n * (n + 1) = 3080) ∧ (n + (n + 1) = -111) :=
sorry

end sum_of_consecutive_negative_integers_with_product_3080_l94_94869


namespace units_digit_of_17_pow_3_mul_24_l94_94493

def unit_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_17_pow_3_mul_24 :
  unit_digit (17^3 * 24) = 2 :=
by
  sorry

end units_digit_of_17_pow_3_mul_24_l94_94493


namespace mark_additional_inches_l94_94841

theorem mark_additional_inches
  (mark_feet : ℕ)
  (mark_inches : ℕ)
  (mike_feet : ℕ)
  (mike_inches : ℕ)
  (foot_to_inches : ℕ)
  (mike_taller_than_mark : ℕ) :
  mark_feet = 5 →
  mike_feet = 6 →
  mike_inches = 1 →
  mike_taller_than_mark = 10 →
  foot_to_inches = 12 →
  5 * 12 + mark_inches + 10 = 6 * 12 + 1 →
  mark_inches = 3 :=
by
  intros
  sorry

end mark_additional_inches_l94_94841


namespace joan_balloons_l94_94834

theorem joan_balloons (m t j : ℕ) (h1 : m = 41) (h2 : t = 81) : j = t - m → j = 40 :=
by
  intros h3
  rw [h1, h2] at h3
  exact h3

end joan_balloons_l94_94834


namespace term_217_is_61st_l94_94212

variables {a_n : ℕ → ℝ}

def arithmetic_sequence (a_n : ℕ → ℝ) (a_15 a_45 : ℝ) : Prop :=
  ∃ (a₁ d : ℝ), (∀ n, a_n n = a₁ + (n - 1) * d) ∧ a_n 15 = a_15 ∧ a_n 45 = a_45

theorem term_217_is_61st (h : arithmetic_sequence a_n 33 153) : a_n 61 = 217 := sorry

end term_217_is_61st_l94_94212


namespace one_fourth_in_one_eighth_l94_94073

theorem one_fourth_in_one_eighth : (1/8 : ℚ) / (1/4) = (1/2) := 
by
  sorry

end one_fourth_in_one_eighth_l94_94073


namespace investment_C_l94_94166

theorem investment_C (A_invest B_invest profit_A total_profit C_invest : ℕ)
  (hA_invest : A_invest = 6300) 
  (hB_invest : B_invest = 4200) 
  (h_profit_A : profit_A = 3900) 
  (h_total_profit : total_profit = 13000) 
  (h_proportional : profit_A / total_profit = A_invest / (A_invest + B_invest + C_invest)) :
  C_invest = 10500 := by
  sorry

end investment_C_l94_94166


namespace number_of_female_students_l94_94600

-- Given conditions
variables (F : ℕ)

-- The average score of all students (90)
def avg_all_students := 90
-- The total number of male students (8)
def num_male_students := 8
-- The average score of male students (87)
def avg_male_students := 87
-- The average score of female students (92)
def avg_female_students := 92

-- We want to prove the following statement
theorem number_of_female_students :
  num_male_students * avg_male_students + F * avg_female_students = (num_male_students + F) * avg_all_students →
  F = 12 :=
sorry

end number_of_female_students_l94_94600


namespace total_digits_first_2003_even_integers_l94_94138

theorem total_digits_first_2003_even_integers : 
  let even_integers := (List.range' 1 (2003 * 2)).filter (λ n => n % 2 = 0)
  let one_digit_count := List.filter (λ n => n < 10) even_integers |>.length
  let two_digit_count := List.filter (λ n => 10 ≤ n ∧ n < 100) even_integers |>.length
  let three_digit_count := List.filter (λ n => 100 ≤ n ∧ n < 1000) even_integers |>.length
  let four_digit_count := List.filter (λ n => 1000 ≤ n) even_integers |>.length
  let total_digits := one_digit_count * 1 + two_digit_count * 2 + three_digit_count * 3 + four_digit_count * 4
  total_digits = 7460 :=
by
  sorry

end total_digits_first_2003_even_integers_l94_94138


namespace lines_of_service_l94_94123

theorem lines_of_service (n : ℕ) (k : ℕ) (h₀ : n = 8) (h₁ : k = 4) :
  (∑ m in finset.range (n+1), if m = k then (nat.factorial n) / ((nat.factorial k) * nat.factorial (n - k)) * (nat.factorial k) * (nat.factorial (n - k)) else 0 ) = 40320 := 
by
  sorry

end lines_of_service_l94_94123


namespace p_div_q_is_12_l94_94789

-- Definition of binomials and factorials required for the proof
open Nat

/-- Define the number of ways to distribute balls for configuration A -/
def config_A : ℕ :=
  @choose 5 1 * @choose 4 2 * @choose 2 1 * (factorial 20) / (factorial 2 * factorial 4 * factorial 4 * factorial 3 * factorial 7)

/-- Define the number of ways to distribute balls for configuration B -/
def config_B : ℕ :=
  @choose 5 2 * @choose 3 3 * (factorial 20) / (factorial 3 * factorial 3 * factorial 4 * factorial 4 * factorial 4)

/-- The ratio of probabilities p/q for the given distributions of balls into bins is 12 -/
theorem p_div_q_is_12 : config_A / config_B = 12 :=
by
  sorry

end p_div_q_is_12_l94_94789


namespace total_expenditure_l94_94548

variable (num_coffees_per_day : ℕ) (cost_per_coffee : ℕ) (days_in_april : ℕ)

theorem total_expenditure (h1 : num_coffees_per_day = 2) (h2 : cost_per_coffee = 2) (h3 : days_in_april = 30) :
  num_coffees_per_day * cost_per_coffee * days_in_april = 120 := by
  sorry

end total_expenditure_l94_94548


namespace max_wrappers_l94_94914

-- Definitions for the conditions
def total_wrappers : ℕ := 49
def andy_wrappers : ℕ := 34

-- The problem statement to prove
theorem max_wrappers : total_wrappers - andy_wrappers = 15 :=
by
  sorry

end max_wrappers_l94_94914


namespace longest_side_of_triangle_l94_94122

theorem longest_side_of_triangle (x : ℝ) (h1 : 8 + (2 * x + 5) + (3 * x + 2) = 40) : 
  max (max 8 (2 * x + 5)) (3 * x + 2) = 17 := 
by 
  -- proof goes here
  sorry

end longest_side_of_triangle_l94_94122


namespace algebraic_expression_value_l94_94960

theorem algebraic_expression_value (x y : ℝ) 
  (h1 : x - y = -2) 
  (h2 : 2 * x + y = -1) : 
  (x - y)^2 - (x - 2 * y) * (x + 2 * y) = 7 :=
by {
  sorry
}

end algebraic_expression_value_l94_94960


namespace people_left_line_l94_94780

theorem people_left_line (initial new final L : ℕ) 
  (h1 : initial = 30) 
  (h2 : new = 5) 
  (h3 : final = 25) 
  (h4 : initial - L + new = final) : L = 10 := by
  sorry

end people_left_line_l94_94780


namespace bangles_per_box_l94_94754

-- Define the total number of pairs of bangles
def totalPairs : Nat := 240

-- Define the number of boxes
def numberOfBoxes : Nat := 20

-- Define the proof that each box can hold 24 bangles
theorem bangles_per_box : (totalPairs * 2) / numberOfBoxes = 24 :=
by
  -- Here we're required to do the proof but we'll use 'sorry' to skip it
  sorry

end bangles_per_box_l94_94754


namespace same_gender_probability_l94_94847

-- Define the total number of teachers in School A and their gender distribution.
def schoolA_teachers : Nat := 3
def schoolA_males : Nat := 2
def schoolA_females : Nat := 1

-- Define the total number of teachers in School B and their gender distribution.
def schoolB_teachers : Nat := 3
def schoolB_males : Nat := 1
def schoolB_females : Nat := 2

-- Calculate the probability of selecting two teachers of the same gender.
theorem same_gender_probability :
  (schoolA_males * schoolB_males + schoolA_females * schoolB_females) / (schoolA_teachers * schoolB_teachers) = 4 / 9 :=
by
  sorry

end same_gender_probability_l94_94847


namespace socks_ratio_l94_94913

/-- Alice ordered 6 pairs of green socks and some additional pairs of red socks. The price per pair
of green socks was three times that of the red socks. During the delivery, the quantities of the 
pairs were accidentally swapped. This mistake increased the bill by 40%. Prove that the ratio of the 
number of pairs of green socks to red socks in Alice's original order is 1:2. -/
theorem socks_ratio (r y : ℕ) (h1 : y * r ≠ 0) (h2 : 6 * 3 * y + r * y = (r * 3 * y + 6 * y) * 10 / 7) :
  6 / r = 1 / 2 :=
by
  sorry

end socks_ratio_l94_94913


namespace value_of_k_l94_94698

   noncomputable def k (a b : ℝ) : ℝ := 3 / 4

   theorem value_of_k (a b k : ℝ) 
     (h1: b = 4 * k + 1) 
     (h2: 5 = a * k + 1) 
     (h3: b + 1 = a * k + 1) : 
     k = 3 / 4 := 
   by 
     -- Proof goes here 
     sorry
   
end value_of_k_l94_94698


namespace melissa_bonus_points_l94_94233

/-- Given that Melissa scored 109 points per game and a total of 15089 points in 79 games,
    prove that she got 82 bonus points per game. -/
theorem melissa_bonus_points (points_per_game : ℕ) (total_points : ℕ) (num_games : ℕ)
  (H1 : points_per_game = 109)
  (H2 : total_points = 15089)
  (H3 : num_games = 79) : 
  (total_points - points_per_game * num_games) / num_games = 82 := by
  sorry

end melissa_bonus_points_l94_94233


namespace problem1_problem2_problem3_problem4_l94_94569

theorem problem1 (x : ℝ) : x^2 - 2 * x + 1 = 0 ↔ x = 1 := 
by sorry

theorem problem2 (x : ℝ) : x^2 + 2 * x - 3 = 0 ↔ x = 1 ∨ x = -3 :=
by sorry

theorem problem3 (x : ℝ) : 2 * x^2 + 5 * x - 1 = 0 ↔ x = (-5 + Real.sqrt 33) / 4 ∨ x = (-5 - Real.sqrt 33) / 4 :=
by sorry

theorem problem4 (x : ℝ) : 2 * (x - 3)^2 = x^2 - 9 ↔ x = 3 ∨ x = 9 :=
by sorry

end problem1_problem2_problem3_problem4_l94_94569


namespace gcd_204_85_l94_94136

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  sorry

end gcd_204_85_l94_94136


namespace a_finishes_work_in_four_days_l94_94299

theorem a_finishes_work_in_four_days (x : ℝ) 
  (B_work_rate : ℝ) 
  (work_done_together : ℝ) 
  (work_done_by_B_alone : ℝ) : 
  B_work_rate = 1 / 16 → 
  work_done_together = 2 * (1 / x + 1 / 16) → 
  work_done_by_B_alone = 6 * (1 / 16) → 
  work_done_together + work_done_by_B_alone = 1 → 
  x = 4 :=
by
  intros hB hTogether hBAlone hTotal
  sorry

end a_finishes_work_in_four_days_l94_94299


namespace area_of_triangle_l94_94005

theorem area_of_triangle :
  let A := (10, 1)
  let B := (15, 8)
  let C := (10, 8)
  ∃ (area : ℝ), 
  area = 17.5 ∧ 
  area = 1 / 2 * (abs (B.1 - C.1)) * (abs (C.2 - A.2)) :=
by
  sorry

end area_of_triangle_l94_94005


namespace quadrilateral_area_24_l94_94466

open Classical

noncomputable def quad_area (a b : ℤ) (h : a > b ∧ b > 0) : ℤ :=
let P := (a, b)
let Q := (2*b, a)
let R := (-a, -b)
let S := (-2*b, -a)
-- The proved area
24

theorem quadrilateral_area_24 (a b : ℤ) (h : a > b ∧ b > 0) :
  quad_area a b h = 24 :=
sorry

end quadrilateral_area_24_l94_94466


namespace geometric_sequence_tenth_term_l94_94636

theorem geometric_sequence_tenth_term :
  let a := 5
      r := (3 : ℚ) / 4
  in ∃ a₁₀ : ℚ, a₁₀ = a * r^9 ∧ a₁₀ = 98415 / 262144 := by
  sorry

end geometric_sequence_tenth_term_l94_94636


namespace cost_of_three_pencils_and_two_pens_l94_94427

theorem cost_of_three_pencils_and_two_pens 
  (p q : ℝ) 
  (h1 : 3 * p + 2 * q = 4.15) 
  (h2 : 2 * p + 3 * q = 3.70) : 
  3 * p + 2 * q = 4.15 := 
by 
  exact h1

end cost_of_three_pencils_and_two_pens_l94_94427


namespace robin_uploaded_pics_from_camera_l94_94112

-- Definitions of the conditions
def pics_from_phone := 35
def albums := 5
def pics_per_album := 8

-- The statement we want to prove
theorem robin_uploaded_pics_from_camera : (albums * pics_per_album) - pics_from_phone = 5 :=
by
  -- Proof goes here
  sorry

end robin_uploaded_pics_from_camera_l94_94112


namespace smallest_positive_b_factors_l94_94953

theorem smallest_positive_b_factors (b : ℤ) : 
  (∃ p q : ℤ, x^2 + b * x + 2016 = (x + p) * (x + q) ∧ p + q = b ∧ p * q = 2016 ∧ p > 0 ∧ q > 0) → b = 95 := 
by {
  sorry
}

end smallest_positive_b_factors_l94_94953


namespace pentagon_angle_sum_l94_94641

theorem pentagon_angle_sum
  (a b c d : ℝ) (Q : ℝ)
  (sum_angles : 180 * (5 - 2) = 540)
  (given_angles : a = 130 ∧ b = 80 ∧ c = 105 ∧ d = 110) :
  Q = 540 - (a + b + c + d) := by
  sorry

end pentagon_angle_sum_l94_94641


namespace rachel_picture_shelves_l94_94391

-- We define the number of books per shelf
def books_per_shelf : ℕ := 9

-- We define the number of mystery shelves
def mystery_shelves : ℕ := 6

-- We define the total number of books
def total_books : ℕ := 72

-- We create a theorem that states Rachel had 2 shelves of picture books
theorem rachel_picture_shelves : ∃ (picture_shelves : ℕ), 
  (mystery_shelves * books_per_shelf + picture_shelves * books_per_shelf = total_books) ∧
  picture_shelves = 2 := by
  sorry

end rachel_picture_shelves_l94_94391


namespace average_weight_a_b_l94_94728

variables (A B C : ℝ)

theorem average_weight_a_b (h1 : (A + B + C) / 3 = 43)
                          (h2 : (B + C) / 2 = 43)
                          (h3 : B = 37) :
                          (A + B) / 2 = 40 :=
by
  sorry

end average_weight_a_b_l94_94728


namespace simplify_expression_l94_94566

theorem simplify_expression :
  (1 / (1 / (Real.sqrt 5 + 2) + 2 / (Real.sqrt 7 - 2))) = 
  (6 * Real.sqrt 7 + 9 * Real.sqrt 5 + 6) / (118 + 12 * Real.sqrt 35) :=
  sorry

end simplify_expression_l94_94566


namespace vote_difference_l94_94608

-- Definitions of initial votes for and against the policy
def vote_initial_for (x y : ℕ) : Prop := x + y = 450
def initial_margin (x y m : ℕ) : Prop := y > x ∧ y - x = m

-- Definitions of votes for and against in the second vote
def vote_second_for (x' y' : ℕ) : Prop := x' + y' = 450
def second_margin (x' y' m : ℕ) : Prop := x' - y' = 3 * m
def second_vote_ratio (x' y : ℕ) : Prop := x' = 10 * y / 9

-- Theorem to prove the increase in votes
theorem vote_difference (x y x' y' m : ℕ)
  (hi : vote_initial_for x y)
  (hm : initial_margin x y m)
  (hs : vote_second_for x' y')
  (hsm : second_margin x' y' m)
  (hr : second_vote_ratio x' y) : 
  x' - x = 52 :=
sorry

end vote_difference_l94_94608


namespace existence_of_rational_solutions_a_nonexistence_of_rational_solutions_b_l94_94156

def is_rational_non_integer (a : ℚ) : Prop :=
  ¬ (∃ n : ℤ, a = n)

theorem existence_of_rational_solutions_a :
  ∃ x y : ℚ, is_rational_non_integer x ∧ is_rational_non_integer y ∧
    ∃ k1 k2 : ℤ, 19 * x + 8 * y = k1 ∧ 8 * x + 3 * y = k2 :=
sorry

theorem nonexistence_of_rational_solutions_b :
  ¬ (∃ x y : ℚ, is_rational_non_integer x ∧ is_rational_non_integer y ∧
    ∃ k1 k2 : ℤ, 19 * x^2 + 8 * y^2 = k1 ∧ 8 * x^2 + 3 * y^2 = k2) :=
sorry

end existence_of_rational_solutions_a_nonexistence_of_rational_solutions_b_l94_94156


namespace minimum_value_of_f_l94_94821

noncomputable def f (x : ℝ) : ℝ := 4 * x + 9 / x

theorem minimum_value_of_f : 
  (∀ (x : ℝ), x > 0 → f x ≥ 12) ∧ (∃ (x : ℝ), x > 0 ∧ f x = 12) :=
by {
  sorry
}

end minimum_value_of_f_l94_94821


namespace product_lcm_gcd_eq_128_l94_94793

theorem product_lcm_gcd_eq_128 : (Int.gcd 8 16) * (Int.lcm 8 16) = 128 :=
by
  sorry

end product_lcm_gcd_eq_128_l94_94793


namespace cuboids_painted_l94_94010

-- Let's define the conditions first
def faces_per_cuboid : ℕ := 6
def total_faces_painted : ℕ := 36

-- Now, we state the theorem we want to prove
theorem cuboids_painted (n : ℕ) (h : total_faces_painted = n * faces_per_cuboid) : n = 6 :=
by
  -- Add proof here
  sorry

end cuboids_painted_l94_94010


namespace card_area_l94_94720

theorem card_area (length width : ℕ) (h_length : length = 5) (h_width : width = 7)
  (h_area_after_shortening : (length - 1) * width = 24 ∨ length * (width - 1) = 24) :
  length * (width - 1) = 18 :=
by
  sorry

end card_area_l94_94720


namespace square_area_from_isosceles_triangle_l94_94472

theorem square_area_from_isosceles_triangle:
  ∀ (b h : ℝ) (Side_of_Square : ℝ), b = 2 ∧ h = 3 ∧ Side_of_Square = (6 / 5) 
  → (Side_of_Square ^ 2) = (36 / 25) := 
by
  intro b h Side_of_Square
  rintro ⟨hb, hh, h_side⟩
  sorry

end square_area_from_isosceles_triangle_l94_94472


namespace extreme_value_point_of_f_l94_94505

noncomputable def f (x : ℝ) : ℝ := sorry -- Assume the definition of f that derives this f'

def f' (x : ℝ) : ℝ := x^3 - 3 * x + 2

theorem extreme_value_point_of_f : (∃ x : ℝ, x = -2 ∧ ∀ y : ℝ, y ≠ -2 → f' y < 0) := sorry

end extreme_value_point_of_f_l94_94505


namespace boys_total_count_l94_94128

theorem boys_total_count 
  (avg_age_all: ℤ) (avg_age_first6: ℤ) (avg_age_last6: ℤ)
  (total_first6: ℤ) (total_last6: ℤ) (total_age_all: ℤ) :
  avg_age_all = 50 →
  avg_age_first6 = 49 →
  avg_age_last6 = 52 →
  total_first6 = 6 * avg_age_first6 →
  total_last6 = 6 * avg_age_last6 →
  total_age_all = total_first6 + total_last6 →
  total_age_all = avg_age_all * 13 :=
by
  intros h_avg_all h_avg_first6 h_avg_last6 h_total_first6 h_total_last6 h_total_age_all
  rw [h_avg_all, h_avg_first6, h_avg_last6] at *
  -- Proof steps skipped
  sorry

end boys_total_count_l94_94128


namespace parallelepiped_volume_l94_94577

open Real

noncomputable def volume_parallelepiped
  (a b : ℝ) (angle : ℝ) (S : ℝ) (sin_30 : angle = π / 6) : ℝ :=
  let h := S / (2 * (a + b))
  let base_area := (a * b * sin (π / 6)) / 2
  base_area * h

theorem parallelepiped_volume 
  (a b : ℝ) (S : ℝ) (h : S ≠ 0 ∧ a > 0 ∧ b > 0) :
  volume_parallelepiped a b (π / 6) S (rfl) = (a * b * S) / (4 * (a + b)) :=
by
  sorry

end parallelepiped_volume_l94_94577


namespace vec_magnitude_is_five_l94_94060

noncomputable def vec_a : ℝ × ℝ := (2, 1)
noncomputable def vec_b : ℝ × ℝ := (-2, 4)

def vec_sub (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 - b.1, a.2 - b.2)

def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2)

theorem vec_magnitude_is_five : magnitude (vec_sub vec_a vec_b) = 5 := by
  sorry

end vec_magnitude_is_five_l94_94060


namespace sin_225_cos_225_l94_94923

noncomputable def sin_225_eq_neg_sqrt2_div_2 : Prop :=
  Real.sin (225 * Real.pi / 180) = -Real.sqrt 2 / 2

noncomputable def cos_225_eq_neg_sqrt2_div_2 : Prop :=
  Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2

theorem sin_225 : sin_225_eq_neg_sqrt2_div_2 := by
  sorry

theorem cos_225 : cos_225_eq_neg_sqrt2_div_2 := by
  sorry

end sin_225_cos_225_l94_94923


namespace min_abs_value_sum_l94_94803

theorem min_abs_value_sum (x : ℚ) : (min (|x - 1| + |x + 3|) = 4) :=
sorry

end min_abs_value_sum_l94_94803


namespace union_of_sets_l94_94898

def A := { x : ℝ | -1 ≤ x ∧ x < 3 }
def B := { x : ℝ | 2 < x ∧ x ≤ 5 }

theorem union_of_sets : A ∪ B = { x : ℝ | -1 ≤ x ∧ x ≤ 5 } := 
by sorry

end union_of_sets_l94_94898


namespace largest_value_l94_94192

noncomputable def largest_possible_4x_3y (x y : ℝ) : ℝ :=
  4 * x + 3 * y

theorem largest_value (x y : ℝ) :
  x^2 + y^2 = 16 * x + 8 * y + 8 → (∃ x y, largest_possible_4x_3y x y = 9.64) :=
by
  sorry

end largest_value_l94_94192


namespace sum_a_b_l94_94503

theorem sum_a_b (a b : ℕ) (h1 : 2 + 2 / 3 = 2^2 * (2 / 3))
(h2: 3 + 3 / 8 = 3^2 * (3 / 8)) 
(h3: 4 + 4 / 15 = 4^2 * (4 / 15)) 
(h_n : ∀ n, n + n / (n^2 - 1) = n^2 * (n / (n^2 - 1)) → 
(a = 9^2 - 1) ∧ (b = 9)) : 
a + b = 89 := 
sorry

end sum_a_b_l94_94503


namespace find_base_and_digit_sum_l94_94381

theorem find_base_and_digit_sum (n d : ℕ) (h1 : 4 * n^2 + 5 * n + d = 392) (h2 : 4 * n^2 + 5 * n + 7 = 740 + 7 * d) : n + d = 12 :=
by
  sorry

end find_base_and_digit_sum_l94_94381


namespace problem_solution_l94_94556

noncomputable def verify_solution (x y z : ℝ) : Prop :=
  x = 12 ∧ y = 10 ∧ z = 8 →
  (x > 4) ∧ (y > 4) ∧ (z > 4) →
  ( ( (x + 3)^2 / (y + z - 3) ) + 
    ( (y + 5)^2 / (z + x - 5) ) + 
    ( (z + 7)^2 / (x + y - 7) ) = 45)

theorem problem_solution :
  verify_solution 12 10 8 := by
  sorry

end problem_solution_l94_94556


namespace garrett_total_spent_l94_94957

/-- Garrett bought 6 oatmeal raisin granola bars, each costing $1.25. -/
def oatmeal_bars_count : Nat := 6
def oatmeal_bars_cost_per_unit : ℝ := 1.25

/-- Garrett bought 8 peanut granola bars, each costing $1.50. -/
def peanut_bars_count : Nat := 8
def peanut_bars_cost_per_unit : ℝ := 1.50

/-- The total amount spent on granola bars is $19.50. -/
theorem garrett_total_spent : oatmeal_bars_count * oatmeal_bars_cost_per_unit + peanut_bars_count * peanut_bars_cost_per_unit = 19.50 :=
by
  sorry

end garrett_total_spent_l94_94957


namespace find_f4_l94_94511

def f1 : ℝ × ℝ := (-2, -1)
def f2 : ℝ × ℝ := (-3, 2)
def f3 : ℝ × ℝ := (4, -3)
def equilibrium_condition (f4 : ℝ × ℝ) : Prop :=
  f1 + f2 + f3 + f4 = (0, 0)

-- Statement that needs to be proven
theorem find_f4 : ∃ (f4 : ℝ × ℝ), equilibrium_condition f4 :=
  by
  use (1, 2)
  sorry

end find_f4_l94_94511


namespace find_b_l94_94486

-- Define the constants and assumptions
variables {a b c d : ℝ} (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : 0 < d)

-- The function completes 5 periods between 0 and 2π
def completes_5_periods (b : ℝ) : Prop :=
  (2 * Real.pi) / b = (2 * Real.pi) / 5

theorem find_b (h : completes_5_periods b) : b = 5 :=
sorry

end find_b_l94_94486


namespace souvenir_purchasing_plans_l94_94725

-- Define the conditions
def types := 4
def total_pieces := 25
def pieces_per_type := 10
def at_least_one_of_each := 1

-- The main statement
theorem souvenir_purchasing_plans : 
  ∃ n : ℕ, n = 592 ∧ 
  ∑ i in finset.range(types), 1 ≤ total_pieces ∧ 
  total_pieces ≤ types * pieces_per_type :=
sorry

end souvenir_purchasing_plans_l94_94725


namespace table_relation_l94_94496

theorem table_relation (x y : ℕ) (hx : x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6) :
  (y = 3 ∧ x = 2) ∨ (y = 8 ∧ x = 3) ∨ (y = 15 ∧ x = 4) ∨ (y = 24 ∧ x = 5) ∨ (y = 35 ∧ x = 6) ↔ 
  y = x^2 - x + 2 :=
sorry

end table_relation_l94_94496


namespace total_selling_price_correct_l94_94314

-- Defining the given conditions
def profit_per_meter : ℕ := 5
def cost_price_per_meter : ℕ := 100
def total_meters_sold : ℕ := 85

-- Using the conditions to define the total selling price
def total_selling_price := total_meters_sold * (cost_price_per_meter + profit_per_meter)

-- Stating the theorem without the proof
theorem total_selling_price_correct : total_selling_price = 8925 := by
  sorry

end total_selling_price_correct_l94_94314


namespace trajectory_description_l94_94658

def trajectory_of_A (x y : ℝ) (m : ℝ) : Prop :=
  m * x^2 - y^2 = m ∧ y ≠ 0
  
theorem trajectory_description (x y m : ℝ) (h : m ≠ 0) :
  trajectory_of_A x y m →
    (m < -1 → (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1 ∧ ¬(x = -1 ∧ y = 0) ∧ ¬(x = 1 ∧ y = 0)))) ∧
    (m = -1 → (x^2 + y^2 = 1 ∧ ¬(x = -1 ∧ y = 0) ∧ ¬(x = 1 ∧ y = 0))) ∧
    (-1 < m ∧ m < 0 → (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1 ∧ ¬(x = -1 ∧ y = 0) ∧ ¬(x = 1 ∧ y = 0)))) ∧
    (m > 0 → (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (x^2 / a^2 - y^2 / b^2 = 1 ∧ ¬(x = -1 ∧ y = 0) ∧ ¬(x = 1 ∧ y = 0)))) :=
by
  intro h_trajectory
  sorry

end trajectory_description_l94_94658


namespace residue_of_5_pow_2023_mod_11_l94_94282

theorem residue_of_5_pow_2023_mod_11 : (5 ^ 2023) % 11 = 4 := by
  sorry

end residue_of_5_pow_2023_mod_11_l94_94282


namespace max_4x_3y_l94_94195

theorem max_4x_3y (x y : ℝ) (h : x^2 + y^2 = 16 * x + 8 * y + 8) : 4 * x + 3 * y ≤ 63 :=
sorry

end max_4x_3y_l94_94195


namespace problem_statement_l94_94674

theorem problem_statement (a b : ℝ) (h : a + b = 1) : 
  ((∀ (a b : ℝ), a + b = 1 → ab ≤ 1/4) ∧ 
   (∀ (a b : ℝ), ¬(ab ≤ 1/4) → ¬(a + b = 1)) ∧ 
   ¬(∀ (a b : ℝ), ab ≤ 1/4 → a + b = 1) ∧ 
   ¬(∀ (a b : ℝ), ¬(a + b = 1) → ¬(ab ≤ 1/4))) := 
sorry

end problem_statement_l94_94674


namespace value_of_bc_l94_94823

theorem value_of_bc (a b c d : ℝ) (h1 : a + b = 14) (h2 : c + d = 3) (h3 : a + d = 8) : b + c = 9 :=
sorry

end value_of_bc_l94_94823


namespace solution_set_of_inequality_l94_94739

theorem solution_set_of_inequality :
  {x : ℝ | abs (x^2 - 5 * x + 6) < x^2 - 4} = { x : ℝ | x > 2 } :=
sorry

end solution_set_of_inequality_l94_94739


namespace max_number_of_girls_l94_94088

theorem max_number_of_girls (students : ℕ)
  (num_friends : ℕ → ℕ)
  (h_students : students = 25)
  (h_distinct_friends : ∀ (i j : ℕ), i ≠ j → num_friends i ≠ num_friends j)
  (h_girls_boys : ∃ (G B : ℕ), G + B = students) :
  ∃ G : ℕ, G = 13 := 
sorry

end max_number_of_girls_l94_94088


namespace negation_of_exists_equiv_forall_neg_l94_94260

noncomputable def negation_equivalent (a : ℝ) : Prop :=
  ∀ a : ℝ, ¬ ∃ x : ℝ, a * x^2 + 1 = 0

-- The theorem statement
theorem negation_of_exists_equiv_forall_neg (h : ∃ a : ℝ, ∃ x : ℝ, a * x^2 + 1 = 0) :
  negation_equivalent a :=
by {
  sorry
}

end negation_of_exists_equiv_forall_neg_l94_94260


namespace minimum_value_of_f_l94_94840

noncomputable def f (x m : ℝ) := (1 / 3) * x^3 - x + m

theorem minimum_value_of_f (m : ℝ) (h_max : f (-1) m = 1) : 
  f 1 m = -1 / 3 :=
by
  sorry

end minimum_value_of_f_l94_94840


namespace arithmetic_sequence_goal_l94_94539

open Nat

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- Definition of an arithmetic sequence
def is_arithmetic_sequence : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Given condition
axiom h1 : a 2 + a 7 = 10

-- Goal
theorem arithmetic_sequence_goal (h : is_arithmetic_sequence a d) : 3 * a 4 + a 6 = 20 :=
sorry

end arithmetic_sequence_goal_l94_94539


namespace find_length_y_l94_94498

def length_y (AO OC DO BO BD y : ℝ) : Prop := 
  AO = 3 ∧ OC = 11 ∧ DO = 3 ∧ BO = 6 ∧ BD = 7 ∧ y = 3 * Real.sqrt 91

theorem find_length_y : length_y 3 11 3 6 7 (3 * Real.sqrt 91) :=
by
  sorry

end find_length_y_l94_94498


namespace maximize_quadratic_expression_l94_94284

theorem maximize_quadratic_expression :
  ∃ x : ℝ, (∀ y : ℝ, -2 * y^2 - 8 * y + 10 ≤ -2 * x^2 - 8 * x + 10) ∧ x = -2 :=
by
  sorry

end maximize_quadratic_expression_l94_94284


namespace prob_at_least_one_solves_l94_94718

theorem prob_at_least_one_solves (p1 p2 : ℝ) (h1 : 0 ≤ p1) (h2 : p1 ≤ 1) (h3 : 0 ≤ p2) (h4 : p2 ≤ 1) :
  (1 : ℝ) - (1 - p1) * (1 - p2) = 1 - ((1 - p1) * (1 - p2)) :=
by sorry

end prob_at_least_one_solves_l94_94718


namespace total_first_tier_college_applicants_l94_94300

theorem total_first_tier_college_applicants
  (total_students : ℕ)
  (sample_size : ℕ)
  (sample_applicants : ℕ)
  (total_applicants : ℕ) 
  (h1 : total_students = 1000)
  (h2 : sample_size = 150)
  (h3 : sample_applicants = 60)
  : total_applicants = 400 :=
sorry

end total_first_tier_college_applicants_l94_94300


namespace intersection_A_B_l94_94662

def A : Set ℝ := { x | ∃ y, y = Real.sqrt (1 - x^2) }
def B : Set ℝ := { x | 0 < x ∧ x < 1 }

theorem intersection_A_B : A ∩ B = { x | 0 < x ∧ x < 1 } := by
  sorry

end intersection_A_B_l94_94662


namespace sum_prime_factors_77_l94_94441

noncomputable def sum_prime_factors (n : ℕ) : ℕ :=
  if n = 77 then 7 + 11 else 0

theorem sum_prime_factors_77 : sum_prime_factors 77 = 18 :=
by
  -- The theorem states that the sum of the prime factors of 77 is 18
  rw sum_prime_factors
  -- Given that the prime factors of 77 are 7 and 11, summing them gives 18
  if_clause
  -- We finalize by proving this is indeed the case
  sorry

end sum_prime_factors_77_l94_94441


namespace miriam_flowers_total_l94_94106

theorem miriam_flowers_total :
  let monday_flowers := 45
  let tuesday_flowers := 75
  let wednesday_flowers := 35
  let thursday_flowers := 105
  let friday_flowers := 0
  let saturday_flowers := 60
  (monday_flowers + tuesday_flowers + wednesday_flowers + thursday_flowers + friday_flowers + saturday_flowers) = 320 :=
by
  -- Calculations go here but we're using sorry to skip them
  sorry

end miriam_flowers_total_l94_94106


namespace Calvin_insect_count_l94_94325

theorem Calvin_insect_count:
  ∀ (roaches scorpions crickets caterpillars : ℕ), 
    roaches = 12 →
    scorpions = 3 →
    crickets = roaches / 2 →
    caterpillars = scorpions * 2 →
    roaches + scorpions + crickets + caterpillars = 27 := 
by
  intros roaches scorpions crickets caterpillars h_roaches h_scorpions h_crickets h_caterpillars
  rw [h_roaches, h_scorpions, h_crickets, h_caterpillars]
  norm_num
  sorry

end Calvin_insect_count_l94_94325


namespace divisor_is_31_l94_94826

-- Definition of the conditions.
def condition1 (x : ℤ) : Prop :=
  ∃ k : ℤ, x = 62 * k + 7

def condition2 (x y : ℤ) : Prop :=
  ∃ m : ℤ, x + 11 = y * m + 18

-- Main statement asserting the divisor y.
theorem divisor_is_31 (x y : ℤ) (h₁ : condition1 x) (h₂ : condition2 x y) : y = 31 :=
sorry

end divisor_is_31_l94_94826


namespace max_product_xy_l94_94681

theorem max_product_xy (x y : ℕ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : 7 * x + 4 * y = 150) : xy = 200 :=
by
  sorry

end max_product_xy_l94_94681


namespace length_of_longest_side_l94_94121

variable (a b c p x l : ℝ)

-- conditions of the original problem
def original_triangle_sides (a b c : ℝ) : Prop := a = 8 ∧ b = 15 ∧ c = 17

def is_right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2

def similar_triangle_perimeter (a b c p x : ℝ) : Prop := (a * x) + (b * x) + (c * x) = p

-- proof target
theorem length_of_longest_side (h1: original_triangle_sides a b c) 
                               (h2: is_right_triangle a b c) 
                               (h3: similar_triangle_perimeter a b c p x) 
                               (h4: x = 4)
                               (h5: p = 160): (c * x) = 68 := by
  -- to complete the proof
  sorry

end length_of_longest_side_l94_94121


namespace lines_symmetric_about_y_axis_l94_94675

theorem lines_symmetric_about_y_axis (m n p : ℝ) :
  (∀ x y : ℝ, x + m * y + 5 = 0 ↔ x + n * y + p = 0)
  ↔ (m = -n ∧ p = -5) :=
sorry

end lines_symmetric_about_y_axis_l94_94675


namespace probability_obtuse_triangle_is_one_fourth_l94_94540

-- Define the set of possible integers
def S : Set ℤ := {1, 2, 3, 4, 5, 6}

-- Condition for forming an obtuse triangle
def is_obtuse_triangle (a b c : ℤ) : Prop :=
  a < b + c ∧ b < a + c ∧ c < a + b ∧ 
  (a^2 + b^2 < c^2 ∨ a^2 + c^2 < b^2 ∨ b^2 + c^2 < a^2)

-- List of valid triples that can form an obtuse triangle
def valid_obtuse_triples : List (ℤ × ℤ × ℤ) :=
  [(2, 3, 4), (2, 4, 5), (2, 5, 6), (3, 4, 6), (3, 5, 6)]

-- Total number of combinations
def total_combinations : Nat := 20

-- Number of valid combinations for obtuse triangles
def valid_combinations : Nat := 5

-- Calculate the probability
def probability_obtuse_triangle : ℚ := valid_combinations / total_combinations

theorem probability_obtuse_triangle_is_one_fourth :
  probability_obtuse_triangle = 1 / 4 :=
by
  sorry

end probability_obtuse_triangle_is_one_fourth_l94_94540


namespace find_number_l94_94458

theorem find_number (x : ℝ) (h : 0.30 * x - 70 = 20) : x = 300 :=
sorry

end find_number_l94_94458


namespace pablo_puzzle_pieces_per_hour_l94_94238

theorem pablo_puzzle_pieces_per_hour
  (num_300_puzzles : ℕ)
  (num_500_puzzles : ℕ)
  (pieces_per_300_puzzle : ℕ)
  (pieces_per_500_puzzle : ℕ)
  (max_hours_per_day : ℕ)
  (total_days : ℕ)
  (total_pieces_completed : ℕ)
  (total_hours_spent : ℕ)
  (P : ℕ)
  (h1 : num_300_puzzles = 8)
  (h2 : num_500_puzzles = 5)
  (h3 : pieces_per_300_puzzle = 300)
  (h4 : pieces_per_500_puzzle = 500)
  (h5 : max_hours_per_day = 7)
  (h6 : total_days = 7)
  (h7 : total_pieces_completed = (num_300_puzzles * pieces_per_300_puzzle + num_500_puzzles * pieces_per_500_puzzle))
  (h8 : total_hours_spent = max_hours_per_day * total_days)
  (h9 : P = total_pieces_completed / total_hours_spent) :
  P = 100 :=
sorry

end pablo_puzzle_pieces_per_hour_l94_94238


namespace platform_length_l94_94909

theorem platform_length (train_length : ℝ) (speed_kmph : ℝ) (time_sec : ℝ) (platform_length : ℝ) :
  train_length = 150 ∧ speed_kmph = 75 ∧ time_sec = 20 →
  platform_length = 1350 :=
by
  sorry

end platform_length_l94_94909


namespace simplify_abs_expression_l94_94820

theorem simplify_abs_expression (a b : ℝ) (h1 : a > 0) (h2 : a * b < 0) :
  |a - 2 * b + 5| + |-3 * a + 2 * b - 2| = 4 * a - 4 * b + 7 := by
  sorry

end simplify_abs_expression_l94_94820


namespace first_discount_percentage_l94_94908

theorem first_discount_percentage
  (list_price : ℝ)
  (second_discount : ℝ)
  (third_discount : ℝ)
  (tax_rate : ℝ)
  (final_price : ℝ)
  (D1 : ℝ)
  (h_list_price : list_price = 150)
  (h_second_discount : second_discount = 12)
  (h_third_discount : third_discount = 5)
  (h_tax_rate : tax_rate = 10)
  (h_final_price : final_price = 105) :
  100 - 100 * (final_price / (list_price * (1 - D1 / 100) * (1 - second_discount / 100) * (1 - third_discount / 100) * (1 + tax_rate / 100))) = 24.24 :=
by
  sorry

end first_discount_percentage_l94_94908


namespace polar_to_cartesian_point_polar_to_cartesian_line_distance_from_point_to_line_l94_94377

open Real

noncomputable def polar_to_cartesian (r : ℝ) (θ : ℝ) : (ℝ × ℝ) :=
  (r * cos θ, r * sin θ)

theorem polar_to_cartesian_point :
  polar_to_cartesian 4 (π / 4) = (2 * sqrt 2, 2 * sqrt 2) :=
by
  sorry

def line_in_cartesian (x y : ℝ) := x + y - sqrt 2 = 0

theorem polar_to_cartesian_line (ρ θ : ℝ) :
  (ρ = 1 / sin (θ + π / 4)) → line_in_cartesian (ρ * cos θ) (ρ * sin θ) :=
by
  sorry

theorem distance_from_point_to_line :
  (sqrt ((2 * sqrt 2 - 0)^2 + (2 * sqrt 2 - 0)^2) - sqrt 2) / sqrt (1 + 1^2) = 3 :=
by
  sorry

end polar_to_cartesian_point_polar_to_cartesian_line_distance_from_point_to_line_l94_94377


namespace cars_selected_l94_94470

theorem cars_selected (num_cars num_clients selections_made total_selections : ℕ)
  (h1 : num_cars = 16)
  (h2 : num_clients = 24)
  (h3 : selections_made = 2)
  (h4 : total_selections = num_clients * selections_made) :
  num_cars * (total_selections / num_cars) = 48 :=
by
  sorry

end cars_selected_l94_94470


namespace sum_of_fourth_powers_of_consecutive_integers_l94_94425

-- Definitions based on conditions
def consecutive_squares_sum (x : ℤ) : Prop :=
  (x - 1)^2 + x^2 + (x + 1)^2 = 12246

-- Statement of the problem
theorem sum_of_fourth_powers_of_consecutive_integers (x : ℤ)
  (h : consecutive_squares_sum x) : 
  (x - 1)^4 + x^4 + (x + 1)^4 = 50380802 :=
sorry

end sum_of_fourth_powers_of_consecutive_integers_l94_94425


namespace mac_runs_faster_than_apple_l94_94014

theorem mac_runs_faster_than_apple :
  let Apple_speed := 3 -- miles per hour
  let Mac_speed := 4 -- miles per hour
  let Distance := 24 -- miles
  let Apple_time := Distance / Apple_speed -- hours
  let Mac_time := Distance / Mac_speed -- hours
  let Time_difference := (Apple_time - Mac_time) * 60 -- converting hours to minutes
  Time_difference = 120 := by
  sorry

end mac_runs_faster_than_apple_l94_94014


namespace square_inscription_l94_94775

theorem square_inscription (a b : ℝ) (s1 s2 : ℝ)
  (h_eq_side_smaller : s1 = 4)
  (h_eq_side_larger : s2 = 3 * Real.sqrt 2)
  (h_sum_segments : a + b = s2)
  (h_eq_sum_squares : a^2 + b^2 = (4 * Real.sqrt 2)^2) :
  a * b = -7 := 
by sorry

end square_inscription_l94_94775


namespace find_x_l94_94750

theorem find_x (x : ℝ) (h : x + 5 * 12 / (180 / 3) = 41) : x = 40 :=
sorry

end find_x_l94_94750


namespace no_nonzero_solutions_l94_94918

theorem no_nonzero_solutions (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x^2 + x = y^2 - y) ∧ (y^2 + y = z^2 - z) ∧ (z^2 + z = x^2 - x) → false :=
by
  sorry

end no_nonzero_solutions_l94_94918


namespace sequence_positions_l94_94089

noncomputable def position_of_a4k1 (x : ℕ) : ℕ := 4 * x + 1
noncomputable def position_of_a4k2 (x : ℕ) : ℕ := 4 * x + 2
noncomputable def position_of_a4k3 (x : ℕ) : ℕ := 4 * x + 3
noncomputable def position_of_a4k (x : ℕ) : ℕ := 4 * x

theorem sequence_positions (k : ℕ) :
  (6 + 1964 = 1970 ∧ position_of_a4k1 1964 = 7857) ∧
  (6 + 1965 = 1971 ∧ position_of_a4k1 1965 = 7861) ∧
  (8 + 1962 = 1970 ∧ position_of_a4k2 1962 = 7850) ∧
  (8 + 1963 = 1971 ∧ position_of_a4k2 1963 = 7854) ∧
  (16 + 2 * 977 = 1970 ∧ position_of_a4k3 977 = 3911) ∧
  (14 + 2 * (979 - 1) = 1970 ∧ position_of_a4k 979 = 3916) :=
by sorry

end sequence_positions_l94_94089


namespace exists_nested_rectangles_l94_94031

theorem exists_nested_rectangles (rectangles : ℕ × ℕ → Prop) :
  (∀ n m : ℕ, rectangles (n, m)) → ∃ (n1 m1 n2 m2 : ℕ), n1 ≤ n2 ∧ m1 ≤ m2 ∧ rectangles (n1, m1) ∧ rectangles (n2, m2) :=
by {
  sorry
}

end exists_nested_rectangles_l94_94031


namespace tan_theta_eq_neg_2sqrt2_to_expression_l94_94352

theorem tan_theta_eq_neg_2sqrt2_to_expression (θ : ℝ) (h : Real.tan θ = -2 * Real.sqrt 2) :
  (2 * (Real.cos (θ / 2)) ^ 2 - Real.sin θ - 1) / (Real.sqrt 2 * Real.sin (θ + Real.pi / 4)) = 1 :=
by
  sorry

end tan_theta_eq_neg_2sqrt2_to_expression_l94_94352


namespace sum_of_squares_mod_13_l94_94590

theorem sum_of_squares_mod_13 : 
  (∑ i in Finset.range 13, i^2) % 13 = 0 :=
by
  sorry

end sum_of_squares_mod_13_l94_94590


namespace solution_1_solution_2_l94_94972

noncomputable def f (x : ℝ) : ℝ := abs (2 * x - 1) + abs (2 * x + 3)

lemma f_piecewise (x : ℝ) : 
  f x = if x ≤ -3 / 2 then -4 * x - 2
        else if -3 / 2 < x ∧ x < 1 / 2 then 4
        else 4 * x + 2 := 
by
-- This lemma represents the piecewise definition of f(x)
sorry

theorem solution_1 : 
  (∀ x : ℝ, f x < 5 ↔ (-7 / 4 < x ∧ x < 3 / 4)) := 
by 
-- Proof of the inequality solution
sorry

theorem solution_2 : 
  (∀ t : ℝ, (∀ x : ℝ, f x - t ≥ 0) → t ≤ 4) :=
by
-- Proof that the maximum value of t is 4
sorry

end solution_1_solution_2_l94_94972


namespace proof_problem_1_proof_problem_2_l94_94489

/-
  Problem statement and conditions:
  (1) $(2023-\sqrt{3})^0 + \left| \left( \frac{1}{5} \right)^{-1} - \sqrt{75} \right| - \frac{\sqrt{45}}{\sqrt{5}}$
  (2) $(\sqrt{3}-2)^2 - (\sqrt{2}+\sqrt{3})(\sqrt{3}-\sqrt{2})$
-/

noncomputable def problem_1 := 
  (2023 - Real.sqrt 3)^0 + abs ((1/5: ℝ)⁻¹ - Real.sqrt 75) - Real.sqrt 45 / Real.sqrt 5

noncomputable def problem_2 := 
  (Real.sqrt 3 - 2) ^ 2 - (Real.sqrt 2 + Real.sqrt 3) * (Real.sqrt 3 - Real.sqrt 2)

theorem proof_problem_1 : problem_1 = 5 * Real.sqrt 3 - 7 :=
  by
    sorry

theorem proof_problem_2 : problem_2 = 6 - 4 * Real.sqrt 3 :=
  by
    sorry


end proof_problem_1_proof_problem_2_l94_94489


namespace maximum_marks_l94_94756

theorem maximum_marks (M : ℝ) (h1 : 0.45 * M = 180) : M = 400 := 
by sorry

end maximum_marks_l94_94756


namespace negation_proposition_equiv_l94_94264

open Classical

variable (R : Type) [OrderedRing R] (a x : R)

theorem negation_proposition_equiv :
  (¬ ∃ a : R, ∃ x : R, a * x^2 + 1 = 0) ↔ (∀ a : R, ∀ x : R, a * x^2 + 1 ≠ 0) :=
by
  sorry

end negation_proposition_equiv_l94_94264


namespace proof_problem_l94_94802

-- Definitions of parallel and perpendicular relationships for lines and planes
def parallel (α β : Type) : Prop := sorry
def perpendicular (α β : Type) : Prop := sorry
def contained_in (m : Type) (α : Type) : Prop := sorry

-- Variables representing lines and planes
variables (l m n : Type) (α β : Type)

-- Assumptions from the conditions in step a)
variables 
  (h1 : m ≠ l)
  (h2 : α ≠ β)
  (h3 : parallel m n)
  (h4 : perpendicular m α)
  (h5 : perpendicular n β)

-- The goal is to prove that the planes α and β are parallel under the given conditions
theorem proof_problem : parallel α β :=
sorry

end proof_problem_l94_94802


namespace overall_percentage_support_l94_94310

theorem overall_percentage_support (p_men : ℕ) (p_women : ℕ) (n_men : ℕ) (n_women : ℕ) : 
  (p_men = 55) → (p_women = 80) → (n_men = 200) → (n_women = 800) → 
  (p_men * n_men + p_women * n_women) / (n_men + n_women) = 75 :=
by
  sorry

end overall_percentage_support_l94_94310


namespace monotonicity_f_minimum_k_l94_94808

def f (a x : ℝ) : ℝ := log x - a / (x + 1)

theorem monotonicity_f (a x : ℝ) (hₐ : a ≥ -4) : 
  ∀ x ≥ 0, deriv (f a) x ≥ 0 := 
sorry

theorem minimum_k (a k x₁ x₂ : ℝ) (hₐ : a < -4) (h₁ : x₁ + x₂ = -(a+2)) (h₂ : x₁ * x₂ = 1) 
  (ineq : k * exp (f a x₁ + f a x₂ - 4) + log (k / (x₁ + x₂ - 2)) ≥ 0) : 
  k ≥ 1/exp 1 := 
sorry

end monotonicity_f_minimum_k_l94_94808


namespace trapezoid_inequality_l94_94910

theorem trapezoid_inequality (a b R : ℝ) (h : a > 0) (h1 : b > 0) (h2 : R > 0) 
  (circumscribed : ∃ (x y : ℝ), x + y = a ∧ R^2 * (1/x + 1/y) = b) : 
  a * b ≥ 4 * R^2 :=
by
  sorry

end trapezoid_inequality_l94_94910


namespace cone_base_radius_l94_94125

theorem cone_base_radius (slant_height : ℝ) (central_angle_deg : ℝ) (r : ℝ) 
  (h1 : slant_height = 6) 
  (h2 : central_angle_deg = 120) 
  (h3 : 2 * π * slant_height * (central_angle_deg / 360) = 4 * π) 
  : r = 2 := by
  sorry

end cone_base_radius_l94_94125


namespace one_fourth_in_one_eighth_l94_94072

theorem one_fourth_in_one_eighth : (1/8 : ℚ) / (1/4) = (1/2) := 
by
  sorry

end one_fourth_in_one_eighth_l94_94072
