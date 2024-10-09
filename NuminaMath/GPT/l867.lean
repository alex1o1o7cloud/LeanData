import Mathlib

namespace pounds_in_one_ton_is_2600_l867_86776

variable (pounds_in_one_ton : ℕ)
variable (ounces_in_one_pound : ℕ := 16)
variable (packets : ℕ := 2080)
variable (weight_per_packet_pounds : ℕ := 16)
variable (weight_per_packet_ounces : ℕ := 4)
variable (gunny_bag_capacity_tons : ℕ := 13)

theorem pounds_in_one_ton_is_2600 :
  (packets * (weight_per_packet_pounds + weight_per_packet_ounces / ounces_in_one_pound)) = (gunny_bag_capacity_tons * pounds_in_one_ton) →
  pounds_in_one_ton = 2600 :=
sorry

end pounds_in_one_ton_is_2600_l867_86776


namespace train_passing_time_l867_86752

theorem train_passing_time :
  ∀ (length : ℕ) (speed_kmph : ℕ),
    length = 120 →
    speed_kmph = 72 →
    ∃ (time : ℕ), time = 6 :=
by
  intro length speed_kmph hlength hspeed_kmph
  sorry

end train_passing_time_l867_86752


namespace max_largest_integer_of_five_l867_86729

theorem max_largest_integer_of_five (a b c d e : ℕ) (h1 : (a + b + c + d + e) = 500)
    (h2 : e > c ∧ c > d ∧ d > b ∧ b > a)
    (h3 : (a + b + d + e) / 4 = 105)
    (h4 : b + e = 150) : d ≤ 269 := 
sorry

end max_largest_integer_of_five_l867_86729


namespace measure_of_angle_A_l867_86784

-- Define the conditions as assumptions
variable (B : Real) (angle1 angle2 A : Real)
-- Angle B is 120 degrees
axiom h1 : B = 120
-- One of the angles formed by the dividing line is 50 degrees
axiom h2 : angle1 = 50
-- Angles formed sum up to 180 degrees as they are supplementary
axiom h3 : angle2 = 180 - angle1
-- Vertical angles are equal
axiom h4 : A = angle2

theorem measure_of_angle_A (B angle1 angle2 A : Real) 
    (h1 : B = 120) (h2 : angle1 = 50) (h3 : angle2 = 180 - angle1) (h4 : A = angle2) : A = 130 := 
by
    sorry

end measure_of_angle_A_l867_86784


namespace croissant_price_l867_86787

theorem croissant_price (price_almond: ℝ) (total_expenditure: ℝ) (weeks: ℕ) (price_regular: ℝ) 
  (h1: price_almond = 5.50) (h2: total_expenditure = 468) (h3: weeks = 52) 
  (h4: weeks * price_regular + weeks * price_almond = total_expenditure) : price_regular = 3.50 :=
by 
  sorry

end croissant_price_l867_86787


namespace initial_boys_l867_86737

-- Define the initial conditions
def initial_girls := 4
def final_children := 8
def boys_left := 3
def girls_entered := 2

-- Define the statement to be proved
theorem initial_boys : 
  ∃ (B : ℕ), (B - boys_left) + (initial_girls + girls_entered) = final_children ∧ B = 5 :=
by
  -- Placeholder for the proof
  sorry

end initial_boys_l867_86737


namespace taylor_family_reunion_l867_86719

theorem taylor_family_reunion :
  let number_of_kids := 45
  let number_of_adults := 123
  let number_of_tables := 14
  (number_of_kids + number_of_adults) / number_of_tables = 12 := by sorry

end taylor_family_reunion_l867_86719


namespace square_side_length_l867_86788

theorem square_side_length (s : ℝ) (h1 : 4 * s = 12) (h2 : s^2 = 9) : s = 3 :=
sorry

end square_side_length_l867_86788


namespace min_S_in_grid_l867_86725

def valid_grid (grid : Fin 10 × Fin 10 → Fin 100) (S : ℕ) : Prop :=
  ∀ i j, 
    (i < 9 → grid (i, j) + grid (i + 1, j) ≤ S) ∧
    (j < 9 → grid (i, j) + grid (i, j + 1) ≤ S)

theorem min_S_in_grid : ∃ grid : Fin 10 × Fin 10 → Fin 100, ∃ S : ℕ, valid_grid grid S ∧ 
  (∀ (other_S : ℕ), valid_grid grid other_S → S ≤ other_S) ∧ S = 106 :=
sorry

end min_S_in_grid_l867_86725


namespace intersections_vary_with_A_l867_86786

theorem intersections_vary_with_A (A : ℝ) (hA : A > 0) :
  ∃ x y : ℝ, (y = A * x^2) ∧ (y^2 + 2 = x^2 + 6 * y) ∧ (y = 2 * x - 1) :=
sorry

end intersections_vary_with_A_l867_86786


namespace motorcycles_count_l867_86723

/-- In a parking lot, there are cars and motorcycles. 
    Each car has 5 wheels (including one spare) and each motorcycle has 2 wheels. 
    There are 19 cars in the parking lot. 
    Altogether all vehicles have 117 wheels. 
    Prove that there are 11 motorcycles in the parking lot. -/
theorem motorcycles_count 
  (C M : ℕ)
  (hc : C = 19)
  (total_wheels : ℕ)
  (total_wheels_eq : total_wheels = 117)
  (car_wheels : ℕ)
  (car_wheels_eq : car_wheels = 5 * C)
  (bike_wheels : ℕ)
  (bike_wheels_eq : bike_wheels = total_wheels - car_wheels)
  (wheels_per_bike : ℕ)
  (wheels_per_bike_eq : wheels_per_bike = 2):
  M = bike_wheels / wheels_per_bike :=
by
  sorry

end motorcycles_count_l867_86723


namespace part_a_l867_86704

theorem part_a (n : ℕ) (h_n : n ≥ 3) (x : Fin n → ℝ) (hx : ∀ i j : Fin n, i ≠ j → x i ≠ x j) (hx_pos : ∀ i : Fin n, 0 < x i) :
  ∃ (i j : Fin n), i ≠ j ∧ 0 < (x i - x j) / (1 + (x i) * (x j)) ∧ (x i - x j) / (1 + (x i) * (x j)) < Real.tan (π / (2 * (n - 1))) :=
by
  sorry

end part_a_l867_86704


namespace teal_sales_revenue_l867_86795

theorem teal_sales_revenue :
  let pumpkin_pie_slices := 8
  let pumpkin_pie_price := 5
  let pumpkin_pies_sold := 4
  let custard_pie_slices := 6
  let custard_pie_price := 6
  let custard_pies_sold := 5
  let apple_pie_slices := 10
  let apple_pie_price := 4
  let apple_pies_sold := 3
  let pecan_pie_slices := 12
  let pecan_pie_price := 7
  let pecan_pies_sold := 2
  (pumpkin_pie_slices * pumpkin_pie_price * pumpkin_pies_sold) +
  (custard_pie_slices * custard_pie_price * custard_pies_sold) +
  (apple_pie_slices * apple_pie_price * apple_pies_sold) +
  (pecan_pie_slices * pecan_pie_price * pecan_pies_sold) = 
  628 := by
  sorry

end teal_sales_revenue_l867_86795


namespace simplification_evaluation_l867_86771

-- Define the variables x and y
def x : ℕ := 2
def y : ℕ := 3

-- Define the expression
def expr := 5 * (3 * x^2 * y - x * y^2) - (x * y^2 + 3 * x^2 * y)

-- Lean 4 statement to prove the equivalence
theorem simplification_evaluation : expr = 36 :=
by
  -- Place the proof here when needed
  sorry

end simplification_evaluation_l867_86771


namespace average_payment_l867_86757

theorem average_payment (n m : ℕ) (p1 p2 : ℕ) (h1 : n = 20) (h2 : m = 45) (h3 : p1 = 410) (h4 : p2 = 475) :
  (20 * p1 + 45 * p2) / 65 = 455 :=
by
  sorry

end average_payment_l867_86757


namespace range_of_a_l867_86706

-- Given definition of the function
def f (x a : ℝ) := abs (x - a)

-- Statement of the problem
theorem range_of_a (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → x₁ < -1 → x₂ < -1 → f x₁ a ≤ f x₂ a) → a ≥ -1 :=
by
  sorry

end range_of_a_l867_86706


namespace line_passes_through_circle_center_l867_86794

theorem line_passes_through_circle_center (a : ℝ) : 
  ∀ x y : ℝ, (x, y) = (a, 2*a) → (x - a)^2 + (y - 2*a)^2 = 1 → 2*x - y = 0 :=
by
  sorry

end line_passes_through_circle_center_l867_86794


namespace half_radius_of_circle_y_l867_86728

theorem half_radius_of_circle_y (Cx Cy : ℝ) (r_x r_y : ℝ) 
  (h1 : Cx = 10 * π) 
  (h2 : Cx = 2 * π * r_x) 
  (h3 : π * r_x ^ 2 = π * r_y ^ 2) :
  (1 / 2) * r_y = 2.5 := 
by
-- sorry skips the proof
sorry

end half_radius_of_circle_y_l867_86728


namespace xy_is_necessary_but_not_sufficient_l867_86775

theorem xy_is_necessary_but_not_sufficient (x y : ℝ) :
  (x^2 + y^2 = 0 → xy = 0) ∧ (xy = 0 → ¬(x^2 + y^2 ≠ 0)) := by
  sorry

end xy_is_necessary_but_not_sufficient_l867_86775


namespace cos_double_angle_l867_86701

theorem cos_double_angle (α : ℝ) (h : Real.sin (π/6 - α) = 1/3) :
  Real.cos (2 * (π/3 + α)) = -7/9 :=
by
  sorry

end cos_double_angle_l867_86701


namespace union_complement_A_B_eq_l867_86747

-- Define the universal set U, set A, and set B
def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {2, 3, 4}

-- Define the complement of A with respect to U
def complement_U_A : Set ℕ := {x | x ∈ U ∧ x ∉ A}

-- The statement to be proved
theorem union_complement_A_B_eq {U A B : Set ℕ} (hU : U = {0, 1, 2, 3, 4}) 
  (hA : A = {0, 1, 2, 3}) (hB : B = {2, 3, 4}) :
  (complement_U_A) ∪ B = {2, 3, 4} := 
by
  sorry

end union_complement_A_B_eq_l867_86747


namespace flowers_bloom_l867_86722

theorem flowers_bloom (num_unicorns : ℕ) (flowers_per_step : ℕ) (distance_km : ℕ) (step_length_m : ℕ) 
  (h1 : num_unicorns = 6) (h2 : flowers_per_step = 4) (h3 : distance_km = 9) (h4 : step_length_m = 3) : 
  num_unicorns * (distance_km * 1000 / step_length_m) * flowers_per_step = 72000 :=
by
  sorry

end flowers_bloom_l867_86722


namespace pascal_row_10_sum_l867_86726

-- Define the function that represents the sum of Row n in Pascal's Triangle
def pascal_row_sum (n : ℕ) : ℕ := 2^n

-- State the theorem to be proven
theorem pascal_row_10_sum : pascal_row_sum 10 = 1024 :=
by
  -- Proof is omitted
  sorry

end pascal_row_10_sum_l867_86726


namespace quadratic_point_inequality_l867_86762

theorem quadratic_point_inequality 
  (m y1 y2 : ℝ)
  (hA : y1 = (m - 1)^2)
  (hB : y2 = (m + 1 - 1)^2)
  (hy1_lt_y2 : y1 < y2) :
  m > 1 / 2 :=
by 
  sorry

end quadratic_point_inequality_l867_86762


namespace inequality_and_equality_conditions_l867_86759

theorem inequality_and_equality_conditions
  (x y a b : ℝ)
  (h1 : a + b = 1)
  (h2 : a ≥ 0)
  (h3 : b ≥ 0) :
  (a * x + b * y)^2 ≤ a * x^2 + b * y^2 ∧ ((a * b = 0) ∨ (x = y)) :=
by
  sorry

end inequality_and_equality_conditions_l867_86759


namespace greg_total_earnings_l867_86793

-- Define the charges and walking times as given
def charge_per_dog : ℕ := 20
def charge_per_minute : ℕ := 1
def one_dog_minutes : ℕ := 10
def two_dogs_minutes : ℕ := 7
def three_dogs_minutes : ℕ := 9
def total_dogs_one : ℕ := 1
def total_dogs_two : ℕ := 2
def total_dogs_three : ℕ := 3

-- Total earnings computation
def earnings_one_dog : ℕ := charge_per_dog + charge_per_minute * one_dog_minutes
def earnings_two_dogs : ℕ := total_dogs_two * charge_per_dog + total_dogs_two * charge_per_minute * two_dogs_minutes
def earnings_three_dogs : ℕ := total_dogs_three * charge_per_dog + total_dogs_three * charge_per_minute * three_dogs_minutes
def total_earnings : ℕ := earnings_one_dog + earnings_two_dogs + earnings_three_dogs

-- The proof: Greg's total earnings should be $171
theorem greg_total_earnings : total_earnings = 171 := by
  -- Placeholder for the proof (not required as per the instructions)
  sorry

end greg_total_earnings_l867_86793


namespace tshirt_cost_correct_l867_86724

   -- Definitions of the conditions
   def initial_amount : ℕ := 91
   def cost_of_sweater : ℕ := 24
   def cost_of_shoes : ℕ := 11
   def remaining_amount : ℕ := 50

   -- Define the total cost of the T-shirt purchase
   noncomputable def cost_of_tshirt := 
     initial_amount - remaining_amount - cost_of_sweater - cost_of_shoes

   -- Proof statement that cost_of_tshirt = 6
   theorem tshirt_cost_correct : cost_of_tshirt = 6 := 
   by
     sorry
   
end tshirt_cost_correct_l867_86724


namespace fill_cistern_time_l867_86780

-- Define the rates of the taps
def rateA := (1 : ℚ) / 3  -- Tap A fills 1 cistern in 3 hours (rate is 1/3 per hour)
def rateB := -(1 : ℚ) / 6  -- Tap B empties 1 cistern in 6 hours (rate is -1/6 per hour)
def rateC := (1 : ℚ) / 2  -- Tap C fills 1 cistern in 2 hours (rate is 1/2 per hour)

-- Define the combined rate
def combinedRate := rateA + rateB + rateC

-- The time to fill the cistern when all taps are opened simultaneously
def timeToFill := 1 / combinedRate

-- The theorem stating that the time to fill the cistern is 1.5 hours
theorem fill_cistern_time : timeToFill = (3 : ℚ) / 2 := by
  sorry  -- The proof is omitted as per the instructions

end fill_cistern_time_l867_86780


namespace solve_for_x_l867_86717

-- Definitions of conditions
def sqrt_81_as_3sq : ℝ := (81 : ℝ)^(1/2)  -- sqrt(81)
def sqrt_81_as_3sq_simplified : ℝ := (3^4 : ℝ)^(1/2)  -- equivalent to (3^2) since 81 = 3^4

-- Theorem and goal statement
theorem solve_for_x :
  sqrt_81_as_3sq = sqrt_81_as_3sq_simplified →
  (3 : ℝ)^(3 * (2/3)) = sqrt_81_as_3sq :=
by
  -- Placeholder for the proof
  sorry

end solve_for_x_l867_86717


namespace remainder_of_2n_div4_l867_86727

theorem remainder_of_2n_div4 (n : ℕ) (h : ∃ k : ℕ, n = 4 * k + 3) : (2 * n) % 4 = 2 := 
by
  sorry

end remainder_of_2n_div4_l867_86727


namespace a_cubed_divisible_l867_86739

theorem a_cubed_divisible {a : ℤ} (h1 : 60 ≤ a) (h2 : a^3 ∣ 216000) : a = 60 :=
by {
   sorry
}

end a_cubed_divisible_l867_86739


namespace beef_not_used_l867_86790

-- Define the context and necessary variables
variable (totalBeef : ℕ) (usedVegetables : ℕ)
variable (beefUsed : ℕ)

-- The conditions given in the problem
def initial_beef : Prop := totalBeef = 4
def used_vegetables : Prop := usedVegetables = 6
def relation_vegetables_beef : Prop := usedVegetables = 2 * beefUsed

-- The statement we need to prove
theorem beef_not_used
  (h1 : initial_beef totalBeef)
  (h2 : used_vegetables usedVegetables)
  (h3 : relation_vegetables_beef usedVegetables beefUsed) :
  (totalBeef - beefUsed) = 1 := by
  sorry

end beef_not_used_l867_86790


namespace money_equation_l867_86772

variables (a b: ℝ)

theorem money_equation (h1: 8 * a + b > 160) (h2: 4 * a + b = 120) : a > 10 ∧ ∀ (a1 a2 : ℝ), a1 > a2 → b = 120 - 4 * a → b = 120 - 4 * a1 ∧ 120 - 4 * a1 < 120 - 4 * a2 :=
by 
  sorry

end money_equation_l867_86772


namespace find_a_l867_86763

theorem find_a (x : ℝ) (n : ℕ) (hx : x > 0) (hn : n > 0) :
  x + n^n * (1 / (x^n)) ≥ n + 1 :=
sorry

end find_a_l867_86763


namespace quadratic_intersects_at_3_points_l867_86778

theorem quadratic_intersects_at_3_points (m : ℝ) : 
  (exists x : ℝ, x^2 + 2*x + m = 0) ∧ (m ≠ 0) → m < 1 :=
by
  sorry

end quadratic_intersects_at_3_points_l867_86778


namespace by_how_much_were_the_numerator_and_denominator_increased_l867_86773

noncomputable def original_fraction_is_six_over_eleven (n : ℕ) : Prop :=
  n / (n + 5) = 6 / 11

noncomputable def resulting_fraction_is_seven_over_twelve (n x : ℕ) : Prop :=
  (n + x) / (n + 5 + x) = 7 / 12

theorem by_how_much_were_the_numerator_and_denominator_increased :
  ∃ (n x : ℕ), original_fraction_is_six_over_eleven n ∧ resulting_fraction_is_seven_over_twelve n x ∧ x = 1 :=
by
  sorry

end by_how_much_were_the_numerator_and_denominator_increased_l867_86773


namespace sum_of_4_corners_is_200_l867_86743

-- Define the conditions: 9x9 grid, numbers start from 10, and filled sequentially from left to right and top to bottom.
def topLeftCorner : ℕ := 10
def topRightCorner : ℕ := 18
def bottomLeftCorner : ℕ := 82
def bottomRightCorner : ℕ := 90

-- The main theorem stating that the sum of the numbers in the four corners is 200.
theorem sum_of_4_corners_is_200 :
  topLeftCorner + topRightCorner + bottomLeftCorner + bottomRightCorner = 200 :=
by
  -- Placeholder for proof
  sorry

end sum_of_4_corners_is_200_l867_86743


namespace boat_upstream_time_l867_86713

theorem boat_upstream_time (v t : ℝ) (d c : ℝ) 
  (h1 : d = 24) (h2 : c = 1) (h3 : 4 * (v + c) = d) 
  (h4 : d / (v - c) = t) : t = 6 :=
by
  sorry

end boat_upstream_time_l867_86713


namespace maximum_value_ratio_l867_86731

theorem maximum_value_ratio (a b : ℝ) (h1 : a + b - 2 ≥ 0) (h2 : b - a - 1 ≤ 0) (h3 : a ≤ 1) :
  ∃ x, x = (a + 2 * b) / (2 * a + b) ∧ x ≤ 7/5 := sorry

end maximum_value_ratio_l867_86731


namespace length_of_ae_l867_86783

-- Definition of points and lengths between them
variables (a b c d e : Type)
variables (bc cd de ab ac : ℝ)

-- Given conditions
axiom H1 : bc = 3 * cd
axiom H2 : de = 8
axiom H3 : ab = 5
axiom H4 : ac = 11
axiom H5 : bc = ac - ab
axiom H6 : cd = bc / 3

-- Theorem to prove
theorem length_of_ae : ∀ ab bc cd de : ℝ, ae = ab + bc + cd + de := by
  sorry

end length_of_ae_l867_86783


namespace distance_circumcenter_centroid_inequality_l867_86744

variable {R r d : ℝ}

theorem distance_circumcenter_centroid_inequality 
  (h1 : d = distance_circumcenter_to_centroid)
  (h2 : R = circumradius)
  (h3 : r = inradius) : d^2 ≤ R * (R - 2 * r) := 
sorry

end distance_circumcenter_centroid_inequality_l867_86744


namespace james_remaining_balance_l867_86709

theorem james_remaining_balance 
  (initial_balance : ℕ := 500) 
  (ticket_1_2_cost : ℕ := 150)
  (ticket_3_cost : ℕ := ticket_1_2_cost / 3)
  (total_cost : ℕ := 2 * ticket_1_2_cost + ticket_3_cost)
  (roommate_share : ℕ := total_cost / 2) :
  initial_balance - roommate_share = 325 := 
by 
  -- By not considering the solution steps, we skip to the proof.
  sorry

end james_remaining_balance_l867_86709


namespace min_value_x_squared_plus_10x_l867_86774

theorem min_value_x_squared_plus_10x : ∃ x : ℝ, (x^2 + 10 * x) = -25 :=
by {
  sorry
}

end min_value_x_squared_plus_10x_l867_86774


namespace area_of_triangle_l867_86710

theorem area_of_triangle (A B C : ℝ) (a c : ℝ) (d B_value: ℝ) (h1 : A + B + C = 180) 
                         (h2 : A = B - d) (h3 : C = B + d) (h4 : a = 4) (h5 : c = 3)
                         (h6 : B = 60) :
  (1 / 2) * a * c * Real.sin (B * Real.pi / 180) = 3 * Real.sqrt 3 :=
by
  sorry

end area_of_triangle_l867_86710


namespace no_real_value_x_l867_86738

theorem no_real_value_x (R H : ℝ) (π : ℝ := Real.pi) :
  R = 10 → H = 5 →
  ¬∃ x : ℝ,  π * (R + x)^2 * H = π * R^2 * (H + x) ∧ x ≠ 0 :=
by
  intros hR hH; sorry

end no_real_value_x_l867_86738


namespace initial_innings_l867_86785

/-- The number of innings a player played initially given the conditions described in the problem. -/
theorem initial_innings (n : ℕ)
  (average_runs : ℕ)
  (additional_runs : ℕ)
  (new_average_increase : ℕ)
  (h1 : average_runs = 42)
  (h2 : additional_runs = 86)
  (h3 : new_average_increase = 4) :
  42 * n + 86 = 46 * (n + 1) → n = 10 :=
by
  intros h
  linarith

end initial_innings_l867_86785


namespace marathon_end_time_l867_86749

open Nat

def marathonStart := 15 * 60  -- 3:00 p.m. in minutes (15 hours * 60 minutes)
def marathonDuration := 780    -- Duration in minutes

theorem marathon_end_time : marathonStart + marathonDuration = 28 * 60 := -- 4:00 a.m. in minutes (28 hours * 60 minutes)
  sorry

end marathon_end_time_l867_86749


namespace fourth_term_of_geometric_sequence_l867_86768

theorem fourth_term_of_geometric_sequence (a₁ : ℝ) (a₆ : ℝ) (a₄ : ℝ) (r : ℝ)
  (h₁ : a₁ = 1000)
  (h₂ : a₆ = a₁ * r^5)
  (h₃ : a₆ = 125)
  (h₄ : a₄ = a₁ * r^3) : 
  a₄ = 125 :=
sorry

end fourth_term_of_geometric_sequence_l867_86768


namespace batsman_avg_after_17th_inning_l867_86750

def batsman_average : Prop :=
  ∃ (A : ℕ), 
    (A + 3 = (16 * A + 92) / 17) → 
    (A + 3 = 44)

theorem batsman_avg_after_17th_inning : batsman_average :=
by
  sorry

end batsman_avg_after_17th_inning_l867_86750


namespace problem_21_divisor_l867_86734

theorem problem_21_divisor 
    (k : ℕ) 
    (h1 : ∃ k, 21^k ∣ 435961) 
    (h2 : 21^k ∣ 435961) 
    : 7^k - k^7 = 1 := 
sorry

end problem_21_divisor_l867_86734


namespace large_monkey_doll_cost_l867_86702

theorem large_monkey_doll_cost :
  ∃ (L : ℝ), (300 / L - 300 / (L - 2) = 25) ∧ L > 0 := by
  sorry

end large_monkey_doll_cost_l867_86702


namespace find_a10_l867_86740

variable {a : ℕ → ℝ}
variable (h1 : ∀ n m, a (n + 1) = a n + a m)
variable (h2 : a 6 + a 8 = 16)
variable (h3 : a 4 = 1)

theorem find_a10 : a 10 = 15 := by
  sorry

end find_a10_l867_86740


namespace chloe_first_round_points_l867_86796

variable (P : ℤ)
variable (totalPoints : ℤ := 86)
variable (secondRoundPoints : ℤ := 50)
variable (lastRoundLoss : ℤ := 4)

theorem chloe_first_round_points 
  (h : P + secondRoundPoints - lastRoundLoss = totalPoints) : 
  P = 40 := by
  sorry

end chloe_first_round_points_l867_86796


namespace find_a_b_l867_86742

noncomputable def f (a b x: ℝ) : ℝ := x / (a * x + b)

theorem find_a_b (a b : ℝ) (h₁ : a ≠ 0) (h₂ : f a b (-4) = 4) (h₃ : ∀ x, f a b x = f b a x) :
  a + b = 3 / 2 :=
sorry

end find_a_b_l867_86742


namespace units_digit_4659_pow_157_l867_86753

theorem units_digit_4659_pow_157 : 
  (4659^157) % 10 = 9 := 
by 
  sorry

end units_digit_4659_pow_157_l867_86753


namespace range_of_a_l867_86760

noncomputable def f (a x : ℝ) := a / x - 1 + Real.log x

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, 0 < x ∧ f a x ≤ 0) → a ≤ 1 := 
sorry

end range_of_a_l867_86760


namespace erik_ate_more_pie_l867_86751

theorem erik_ate_more_pie :
  let erik_pies := 0.67
  let frank_pies := 0.33
  erik_pies - frank_pies = 0.34 :=
by
  sorry

end erik_ate_more_pie_l867_86751


namespace find_m_in_function_l867_86700

noncomputable def f (m : ℝ) (x : ℝ) := (1 / 3) * x^3 - x^2 - x + m

theorem find_m_in_function {m : ℝ} (h : ∀ x ∈ Set.Icc (0:ℝ) (1:ℝ), f m x ≥ (1/3)) :
  m = 2 :=
sorry

end find_m_in_function_l867_86700


namespace plantingMethodsCalculation_l867_86766

noncomputable def numPlantingMethods : Nat :=
  let totalSeeds := 5
  let endChoices := 3 * 2 -- Choosing 2 seeds for the ends from 3 remaining types
  let middleChoices := 6 -- Permutations of (A, B, another type) = 3! = 6
  endChoices * middleChoices

theorem plantingMethodsCalculation : numPlantingMethods = 24 := by
  sorry

end plantingMethodsCalculation_l867_86766


namespace polynomial_equation_example_l867_86764

theorem polynomial_equation_example (a0 a1 a2 a3 a4 a5 a6 a7 a8 : ℤ)
  (h : x^5 * (x + 3)^3 = a8 * (x + 1)^8 + a7 * (x + 1)^7 + a6 * (x + 1)^6 + a5 * (x + 1)^5 + a4 * (x + 1)^4 + a3 * (x + 1)^3 + a2 * (x + 1)^2 + a1 * (x + 1) + a0) :
  7 * a7 + 5 * a5 + 3 * a3 + a1 = -8 :=
sorry

end polynomial_equation_example_l867_86764


namespace problem_statement_l867_86754

-- Define f(x) and g(x)
def f (x : ℝ) : ℝ := x^2 + 2 * x + 5
def g (x : ℝ) : ℝ := 2 * x + 3

-- Statement to prove: f(g(3)) - g(f(3)) = 61
theorem problem_statement : f (g 3) - g (f 3) = 61 := by
  sorry

end problem_statement_l867_86754


namespace two_pow_gt_n_square_plus_one_l867_86716

theorem two_pow_gt_n_square_plus_one (n : ℕ) (h : n ≥ 5) : 2^n > n^2 + 1 := 
by {
  sorry
}

end two_pow_gt_n_square_plus_one_l867_86716


namespace parabola_focus_coordinates_l867_86735

theorem parabola_focus_coordinates : 
  ∀ (x y : ℝ), y = 4 * x^2 → (0, y / 16) = (0, 1 / 16) :=
by
  intros x y h
  sorry

end parabola_focus_coordinates_l867_86735


namespace oilProductionPerPerson_west_oilProductionPerPerson_nonwest_oilProductionPerPerson_russia_l867_86789

-- Definitions for oil consumption per person
def oilConsumptionWest : ℝ := 55.084
def oilConsumptionNonWest : ℝ := 214.59
def oilConsumptionRussia : ℝ := 1038.33

-- Lean statements
theorem oilProductionPerPerson_west : oilConsumptionWest = 55.084 := by
  sorry

theorem oilProductionPerPerson_nonwest : oilConsumptionNonWest = 214.59 := by
  sorry

theorem oilProductionPerPerson_russia : oilConsumptionRussia = 1038.33 := by
  sorry

end oilProductionPerPerson_west_oilProductionPerPerson_nonwest_oilProductionPerPerson_russia_l867_86789


namespace set_of_possible_values_l867_86798

-- Define the variables and the conditions as a Lean definition
noncomputable def problem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum_eq_one : a + b + c = 1) : Set ℝ :=
  {x : ℝ | x = (1 / a + 1 / b + 1 / c)}

-- Define the theorem to state that the set of all possible values is [9, ∞)
theorem set_of_possible_values (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum_eq_one : a + b + c = 1) :
  problem a b c ha hb hc sum_eq_one = {x : ℝ | 9 ≤ x} :=
sorry

end set_of_possible_values_l867_86798


namespace probability_intersection_interval_l867_86758

theorem probability_intersection_interval (PA PB p : ℝ) (hPA : PA = 5 / 6) (hPB : PB = 3 / 4) :
  0 ≤ p ∧ p ≤ 3 / 4 :=
sorry

end probability_intersection_interval_l867_86758


namespace bacteria_colony_exceeds_500_l867_86736

theorem bacteria_colony_exceeds_500 :
  ∃ (n : ℕ), (∀ m : ℕ, m < n → 4 * 3^m ≤ 500) ∧ 4 * 3^n > 500 :=
sorry

end bacteria_colony_exceeds_500_l867_86736


namespace chord_angle_measure_l867_86756

theorem chord_angle_measure (AB_ratio : ℕ) (circ : ℝ) (h : AB_ratio = 1 + 5) : 
  ∃ θ : ℝ, θ = (1 / 6) * circ ∧ θ = 60 :=
by
  sorry

end chord_angle_measure_l867_86756


namespace completion_days_l867_86732

theorem completion_days (D : ℝ) :
  (1 / D + 1 / 9 = 1 / 3.2142857142857144) → D = 5 := by
  sorry

end completion_days_l867_86732


namespace average_weight_increase_l867_86799

-- Define the initial conditions as given in the problem
def W_old : ℕ := 53
def W_new : ℕ := 71
def N : ℕ := 10

-- Average weight increase after replacing one oarsman
theorem average_weight_increase : (W_new - W_old : ℝ) / N = 1.8 := by
  sorry

end average_weight_increase_l867_86799


namespace Billys_age_l867_86767

variable (B J : ℕ)

theorem Billys_age :
  B = 2 * J ∧ B + J = 45 → B = 30 :=
by
  sorry

end Billys_age_l867_86767


namespace constant_term_binomial_expansion_n_6_middle_term_coefficient_l867_86707

open Nat

-- Define the binomial expansion term
def binomial_term (n : ℕ) (r : ℕ) (x : ℝ) : ℝ :=
  (Nat.choose n r) * (2 ^ r) * x^(2 * (n-r) - r)

-- (I) Prove the constant term of the binomial expansion when n = 6
theorem constant_term_binomial_expansion_n_6 :
  binomial_term 6 4 (1 : ℝ) = 240 := 
sorry

-- (II) Prove the coefficient of the middle term under given conditions
theorem middle_term_coefficient (n : ℕ) :
  (Nat.choose 8 2 = Nat.choose 8 6) →
  binomial_term 8 4 (1 : ℝ) = 1120 := 
sorry

end constant_term_binomial_expansion_n_6_middle_term_coefficient_l867_86707


namespace find_original_wage_l867_86714

-- Defining the conditions
variables (W : ℝ) (W_new : ℝ) (h : W_new = 35) (h2 : W_new = 1.40 * W)

-- Statement that needs to be proved
theorem find_original_wage : W = 25 :=
by
  -- proof omitted
  sorry

end find_original_wage_l867_86714


namespace ajay_total_gain_l867_86770

theorem ajay_total_gain:
  let dal_A_kg := 15
  let dal_B_kg := 10
  let dal_C_kg := 12
  let dal_D_kg := 8
  let rate_A := 14.50
  let rate_B := 13
  let rate_C := 16
  let rate_D := 18
  let selling_rate := 17.50
  let cost_A := dal_A_kg * rate_A
  let cost_B := dal_B_kg * rate_B
  let cost_C := dal_C_kg * rate_C
  let cost_D := dal_D_kg * rate_D
  let total_cost := cost_A + cost_B + cost_C + cost_D
  let total_weight := dal_A_kg + dal_B_kg + dal_C_kg + dal_D_kg
  let total_selling_price := total_weight * selling_rate
  let gain := total_selling_price - total_cost
  gain = 104 := by
    sorry

end ajay_total_gain_l867_86770


namespace circles_tangent_dist_l867_86765

theorem circles_tangent_dist (t : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 4) ∧ 
  (∀ x y : ℝ, (x - t)^2 + y^2 = 1) ∧ 
  (∀ x1 y1 x2 y2 : ℝ, x1^2 + y1^2 = 4 → (x2 - t)^2 + y2^2 = 1 → 
    dist (x1, y1) (x2, y2) = 3) → 
  t = 3 ∨ t = -3 :=
by 
  sorry

end circles_tangent_dist_l867_86765


namespace free_space_on_new_drive_l867_86711

theorem free_space_on_new_drive
  (initial_free : ℝ) (initial_used : ℝ) (delete_size : ℝ) (new_files_size : ℝ) (new_drive_size : ℝ) :
  initial_free = 2.4 → initial_used = 12.6 → delete_size = 4.6 → new_files_size = 2 → new_drive_size = 20 →
  (new_drive_size - ((initial_used - delete_size) + new_files_size)) = 10 :=
by simp; sorry

end free_space_on_new_drive_l867_86711


namespace ages_of_children_l867_86715

theorem ages_of_children : ∃ (a1 a2 a3 a4 : ℕ),
  a1 + a2 + a3 + a4 = 33 ∧
  (a1 - 3) + (a2 - 3) + (a3 - 3) + (a4 - 3) = 22 ∧
  (a1 - 7) + (a2 - 7) + (a3 - 7) + (a4 - 7) = 11 ∧
  (a1 - 13) + (a2 - 13) + (a3 - 13) + (a4 - 13) = 1 ∧
  a1 = 14 ∧ a2 = 11 ∧ a3 = 6 ∧ a4 = 2 :=
by
  sorry

end ages_of_children_l867_86715


namespace largest_even_integer_sum_l867_86791

theorem largest_even_integer_sum (x : ℤ) (h : (20 * (x + x + 38) / 2) = 6400) : 
  x + 38 = 339 :=
sorry

end largest_even_integer_sum_l867_86791


namespace commission_percentage_proof_l867_86777

-- Let's define the problem conditions in Lean

-- Condition 1: Commission on first Rs. 10,000
def commission_first_10000 (sales : ℕ) : ℕ :=
  if sales ≤ 10000 then
    5 * sales / 100
  else
    500

-- Condition 2: Amount remitted to company after commission
def amount_remitted (total_sales : ℕ) (commission : ℕ) : ℕ :=
  total_sales - commission

-- Condition 3: Function to calculate commission on exceeding amount
def commission_exceeding (sales : ℕ) (x : ℕ) : ℕ :=
  x * sales / 100

-- The main hypothesis as per the given problem
def correct_commission_percentage (total_sales : ℕ) (remitted : ℕ) (x : ℕ) :=
  commission_first_10000 10000 + commission_exceeding (total_sales - 10000) x
  = total_sales - remitted

-- Problem statement to prove the percentage of commission on exceeding Rs. 10,000 is 4%
theorem commission_percentage_proof : correct_commission_percentage 32500 31100 4 := 
  by sorry

end commission_percentage_proof_l867_86777


namespace num_square_tiles_l867_86730

theorem num_square_tiles (a b c : ℕ) (h1 : a + b + c = 30) (h2 : 3 * a + 4 * b + 5 * c = 100) : b = 10 :=
  sorry

end num_square_tiles_l867_86730


namespace total_cows_in_herd_l867_86745

theorem total_cows_in_herd {n : ℚ} (h1 : 1/3 + 1/6 + 1/9 = 11/18) 
                           (h2 : (1 - 11/18) = 7/18) 
                           (h3 : 8 = (7/18) * n) : 
                           n = 144/7 :=
by sorry

end total_cows_in_herd_l867_86745


namespace value_of_each_gift_card_l867_86748

theorem value_of_each_gift_card (students total_thank_you_cards with_gift_cards total_value : ℕ) 
  (h1 : students = 50)
  (h2 : total_thank_you_cards = 30 * students / 100)
  (h3 : with_gift_cards = total_thank_you_cards / 3)
  (h4 : total_value = 50) :
  total_value / with_gift_cards = 10 := by
  sorry

end value_of_each_gift_card_l867_86748


namespace gain_percent_l867_86746

theorem gain_percent (cp sp : ℝ) (h_cp : cp = 900) (h_sp : sp = 1080) :
    ((sp - cp) / cp) * 100 = 20 :=
by
    sorry

end gain_percent_l867_86746


namespace polygon_diagonals_15_sides_l867_86708

/-- Given a convex polygon with 15 sides, the number of diagonals is 90. -/
theorem polygon_diagonals_15_sides (n : ℕ) (h : n = 15) (convex : Prop) : 
  ∃ d : ℕ, d = 90 :=
by
    sorry

end polygon_diagonals_15_sides_l867_86708


namespace simplify_vector_expression_l867_86755

-- Definitions for vectors
variables {A B C D : Type} [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D]

-- Defining the vectors
variables (AB CA BD CD : A)

-- A definition using the head-to-tail addition of vectors.
def vector_add (v1 v2 : A) : A := v1 + v2

-- Statement to prove
theorem simplify_vector_expression :
  vector_add (vector_add AB CA) BD = CD :=
sorry

end simplify_vector_expression_l867_86755


namespace prime_square_mod_24_l867_86769

theorem prime_square_mod_24 (p q : ℕ) (k : ℤ) 
  (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (hp_gt_5 : p > 5) (hq_gt_5 : q > 5) 
  (h_diff : p ≠ q)
  (h_eq : p^2 - q^2 = 6 * k) : (p^2 - q^2) % 24 = 0 := by
sorry

end prime_square_mod_24_l867_86769


namespace parabola_intersection_ratios_l867_86703

noncomputable def parabola_vertex_x1 (a b c : ℝ) := -b / (2 * a)
noncomputable def parabola_vertex_y1 (a b c : ℝ) := (4 * a * c - b^2) / (4 * a)
noncomputable def parabola_vertex_x2 (a d e : ℝ) := d / (2 * a)
noncomputable def parabola_vertex_y2 (a d e : ℝ) := (4 * a * e + d^2) / (4 * a)

theorem parabola_intersection_ratios
  (a b c d e : ℝ)
  (h1 : 144 * a + 12 * b + c = 21)
  (h2 : 784 * a + 28 * b + c = 3)
  (h3 : -144 * a + 12 * d + e = 21)
  (h4 : -784 * a + 28 * d + e = 3) :
  (parabola_vertex_x1 a b c + parabola_vertex_x2 a d e) / 
  (parabola_vertex_y1 a b c + parabola_vertex_y2 a d e) = 5 / 3 := by
  sorry

end parabola_intersection_ratios_l867_86703


namespace arithmetic_geometric_sequence_ratio_l867_86797

theorem arithmetic_geometric_sequence_ratio (a : ℕ → ℕ) (d : ℕ)
  (h_arithmetic : ∀ n, a (n + 1) = a n + d)
  (h_positive_d : d > 0)
  (h_geometric : a 6 ^ 2 = a 2 * a 12) :
  (a 12) / (a 2) = 9 / 4 :=
sorry

end arithmetic_geometric_sequence_ratio_l867_86797


namespace sam_and_david_licks_l867_86733

theorem sam_and_david_licks :
  let Dan_licks := 58
  let Michael_licks := 63
  let Lance_licks := 39
  let avg_licks := 60
  let total_people := 5
  let total_licks := avg_licks * total_people
  let total_licks_Dan_Michael_Lance := Dan_licks + Michael_licks + Lance_licks
  total_licks - total_licks_Dan_Michael_Lance = 140 := by
  sorry

end sam_and_david_licks_l867_86733


namespace problem1_problem2_l867_86761

-- Problem 1: Prove the expression evaluates to 8
theorem problem1 : (1:ℝ) * (- (1 / 2)⁻¹) + (3 - Real.pi)^0 + (-3)^2 = 8 := 
by
  sorry

-- Problem 2: Prove the expression simplifies to 9a^6 - 2a^2
theorem problem2 (a : ℝ) : a^2 * a^4 - (-2 * a^2)^3 - 3 * a^2 + a^2 = 9 * a^6 - 2 * a^2 := 
by
  sorry

end problem1_problem2_l867_86761


namespace magic_square_S_divisible_by_3_l867_86781

-- Definitions of the 3x3 magic square conditions
def is_magic_square (a : ℕ → ℕ → ℤ) (S : ℤ) : Prop :=
  (a 0 0 + a 0 1 + a 0 2 = S) ∧
  (a 1 0 + a 1 1 + a 1 2 = S) ∧
  (a 2 0 + a 2 1 + a 2 2 = S) ∧
  (a 0 0 + a 1 0 + a 2 0 = S) ∧
  (a 0 1 + a 1 1 + a 2 1 = S) ∧
  (a 0 2 + a 1 2 + a 2 2 = S) ∧
  (a 0 0 + a 1 1 + a 2 2 = S) ∧
  (a 0 2 + a 1 1 + a 2 0 = S)

-- Main theorem statement
theorem magic_square_S_divisible_by_3 :
  ∀ (a : ℕ → ℕ → ℤ) (S : ℤ),
    is_magic_square a S →
    S % 3 = 0 :=
by
  -- Here we assume the existence of the proof
  sorry

end magic_square_S_divisible_by_3_l867_86781


namespace notebooks_difference_l867_86712

theorem notebooks_difference 
  (cost_mika : ℝ) (cost_leo : ℝ) (notebook_price : ℝ)
  (h_cost_mika : cost_mika = 2.40)
  (h_cost_leo : cost_leo = 3.20)
  (h_notebook_price : notebook_price > 0.10)
  (h_mika : ∃ m : ℕ, cost_mika = m * notebook_price)
  (h_leo : ∃ l : ℕ, cost_leo = l * notebook_price)
  : ∃ n : ℕ, (l - m = 4) :=
by
  sorry

end notebooks_difference_l867_86712


namespace rectangle_width_l867_86741

theorem rectangle_width (r l w : ℝ) (h_r : r = Real.sqrt 12) (h_l : l = 3 * Real.sqrt 2)
  (h_area_eq: Real.pi * r^2 = l * w) : w = 2 * Real.sqrt 2 * Real.pi :=
by
  sorry

end rectangle_width_l867_86741


namespace quadratic_eq_one_solution_has_ordered_pair_l867_86721

theorem quadratic_eq_one_solution_has_ordered_pair (a c : ℝ) 
  (h1 : a * c = 25) 
  (h2 : a + c = 17) 
  (h3 : a > c) : 
  (a, c) = (15.375, 1.625) :=
sorry

end quadratic_eq_one_solution_has_ordered_pair_l867_86721


namespace root_of_quadratic_l867_86720

theorem root_of_quadratic (a : ℝ) (h : ∃ (x : ℝ), x = 0 ∧ x^2 + x + 2 * a - 1 = 0) : a = 1 / 2 := by
  sorry

end root_of_quadratic_l867_86720


namespace find_m_l867_86705

/-- Given vectors \(\overrightarrow{OA} = (1, m)\) and \(\overrightarrow{OB} = (m-1, 2)\), if 
\(\overrightarrow{OA} \perp \overrightarrow{AB}\), then \(m = \frac{1}{3}\). -/
theorem find_m (m : ℝ) (h : (1, m).1 * (m - 1 - 1, 2 - m).1 + (1, m).2 * (m - 1 - 1, 2 - m).2 = 0) :
  m = 1 / 3 :=
sorry

end find_m_l867_86705


namespace intersection_AB_l867_86792

/-- Define the set A based on the given condition -/
def setA : Set ℝ := {x | 2 * x ^ 2 + x > 0}

/-- Define the set B based on the given condition -/
def setB : Set ℝ := {x | 2 * x + 1 > 0}

/-- Prove that A ∩ B = {x | x > 0} -/
theorem intersection_AB : (setA ∩ setB) = {x | x > 0} :=
sorry

end intersection_AB_l867_86792


namespace min_ab_min_a_plus_b_l867_86779

theorem min_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 1) : ab >= 8 :=
sorry

theorem min_a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 1) : a + b >= 3 + 2 * Real.sqrt 2 :=
sorry

end min_ab_min_a_plus_b_l867_86779


namespace distinct_prime_factors_330_l867_86782

def num_prime_factors (n : ℕ) : ℕ :=
  if n = 330 then 4 else 0

theorem distinct_prime_factors_330 : num_prime_factors 330 = 4 :=
sorry

end distinct_prime_factors_330_l867_86782


namespace three_digit_number_count_correct_l867_86718

def number_of_three_digit_numbers_with_repetition (digit_count : ℕ) (positions : ℕ) : ℕ :=
  let choices_for_repeated_digit := 5  -- 5 choices for repeated digit
  let ways_to_place_repeated_digit := 3 -- 3 ways to choose positions
  let choices_for_remaining_digit := 4 -- 4 choices for the remaining digit
  choices_for_repeated_digit * ways_to_place_repeated_digit * choices_for_remaining_digit

theorem three_digit_number_count_correct :
  number_of_three_digit_numbers_with_repetition 5 3 = 60 := 
sorry

end three_digit_number_count_correct_l867_86718
