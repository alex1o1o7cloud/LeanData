import Mathlib

namespace find_x_l17_1748

theorem find_x (x y : ℤ) (hx : x > 0) (hy : y > 0) (hxy : x > y) (h : x + y + x * y = 152) : x = 16 := 
by
  sorry

end find_x_l17_1748


namespace find_a_b_sum_pos_solution_l17_1740

theorem find_a_b_sum_pos_solution :
  ∃ (a b : ℕ), (∃ (x : ℝ), x^2 + 16 * x = 100 ∧ x = Real.sqrt a - b) ∧ a + b = 172 :=
by
  sorry

end find_a_b_sum_pos_solution_l17_1740


namespace max_dominoes_in_grid_l17_1732

-- Definitions representing the conditions
def total_squares (rows cols : ℕ) : ℕ := rows * cols
def domino_squares : ℕ := 3
def max_dominoes (total domino : ℕ) : ℕ := total / domino

-- Statement of the problem
theorem max_dominoes_in_grid : max_dominoes (total_squares 20 19) domino_squares = 126 :=
by
  -- placeholders for the actual proof
  sorry

end max_dominoes_in_grid_l17_1732


namespace inequality_holds_equality_condition_l17_1745

variables {x y z : ℝ}
-- Assuming positive real numbers and the given condition
axiom pos_x : 0 < x
axiom pos_y : 0 < y
axiom pos_z : 0 < z
axiom h : x * y + y * z + z * x = x + y + z

theorem inequality_holds : 
  (1 / (x^2 + y + 1)) + (1 / (y^2 + z + 1)) + (1 / (z^2 + x + 1)) ≤ 1 :=
by
  sorry

theorem equality_condition : 
  (1 / (x^2 + y + 1)) + (1 / (y^2 + z + 1)) + (1 / (z^2 + x + 1)) = 1 ↔ x = 1 ∧ y = 1 ∧ z = 1 :=
by
  sorry

end inequality_holds_equality_condition_l17_1745


namespace total_amount_invested_l17_1760

-- Define the problem details: given conditions
def interest_rate_share1 : ℚ := 9 / 100
def interest_rate_share2 : ℚ := 11 / 100
def total_interest_rate : ℚ := 39 / 400
def amount_invested_share2 : ℚ := 3750

-- Define the total amount invested (A), the amount invested at the 9% share (x)
variable (A x : ℚ)

-- Conditions
axiom condition1 : x + amount_invested_share2 = A
axiom condition2 : interest_rate_share1 * x + interest_rate_share2 * amount_invested_share2 = total_interest_rate * A

-- Prove that the total amount invested in both types of shares is Rs. 10,000
theorem total_amount_invested : A = 10000 :=
by {
  -- proof goes here
  sorry
}

end total_amount_invested_l17_1760


namespace no_prime_divisor_of_form_8k_minus_1_l17_1736

theorem no_prime_divisor_of_form_8k_minus_1 (n : ℕ) (h : 0 < n) :
  ¬ ∃ p k : ℕ, Nat.Prime p ∧ p = 8 * k - 1 ∧ p ∣ (2^n + 1) :=
by
  sorry

end no_prime_divisor_of_form_8k_minus_1_l17_1736


namespace solve_equation_l17_1752

-- Define the equation to be proven
def equation (x : ℚ) : Prop :=
  (x + 4) / (x - 3) = (x - 2) / (x + 2)

-- State the theorem
theorem solve_equation : equation (-2 / 11) :=
by
  -- Introduce the equation and the solution to be proven
  unfold equation

  -- Simplify the equation to verify the solution
  sorry


end solve_equation_l17_1752


namespace graph1_higher_than_graph2_l17_1754

theorem graph1_higher_than_graph2 :
  ∀ (x : ℝ), (-x^2 + 2 * x + 3) ≥ (x^2 - 2 * x + 3) :=
by
  intros x
  sorry

end graph1_higher_than_graph2_l17_1754


namespace sixth_number_is_811_l17_1731

noncomputable def sixth_number_in_21st_row : ℕ := 
  let n := 21 
  let k := 6
  let total_numbers_up_to_previous_row := n * n
  let position_in_row := total_numbers_up_to_previous_row + k
  2 * position_in_row - 1

theorem sixth_number_is_811 : sixth_number_in_21st_row = 811 := by
  sorry

end sixth_number_is_811_l17_1731


namespace subset_definition_l17_1781

variable {α : Type} {A B : Set α}

theorem subset_definition :
  A ⊆ B ↔ ∀ a ∈ A, a ∈ B :=
by sorry

end subset_definition_l17_1781


namespace prime_numbers_count_and_sum_l17_1715

-- Definition of prime numbers less than or equal to 20
def prime_numbers_leq_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Proposition stating the number of prime numbers and their sum within 20
theorem prime_numbers_count_and_sum :
  (prime_numbers_leq_20.length = 8) ∧ (prime_numbers_leq_20.sum = 77) := by
  sorry

end prime_numbers_count_and_sum_l17_1715


namespace best_store_is_A_l17_1773

/-- Problem conditions -/
def price_per_ball : Nat := 25
def balls_to_buy : Nat := 58

/-- Store A conditions -/
def balls_bought_per_offer_A : Nat := 10
def balls_free_per_offer_A : Nat := 3

/-- Store B conditions -/
def discount_per_ball_B : Nat := 5

/-- Store C conditions -/
def cashback_rate_C : Nat := 40
def cashback_threshold_C : Nat := 200

/-- Cost calculations -/
def cost_store_A (total_balls : Nat) (price : Nat) : Nat :=
  let full_offers := total_balls / balls_bought_per_offer_A
  let remaining_balls := total_balls % balls_bought_per_offer_A
  let balls_paid_for := full_offers * (balls_bought_per_offer_A - balls_free_per_offer_A) + remaining_balls
  balls_paid_for * price

def cost_store_B (total_balls : Nat) (price : Nat) (discount : Nat) : Nat :=
  total_balls * (price - discount)

def cost_store_C (total_balls : Nat) (price : Nat) (cashback_rate : Nat) (threshold : Nat) : Nat :=
  let cost_before_cashback := total_balls * price
  let full_cashbacks := cost_before_cashback / threshold
  let cashback_amount := full_cashbacks * cashback_rate
  cost_before_cashback - cashback_amount

theorem best_store_is_A :
  cost_store_A balls_to_buy price_per_ball = 1075 ∧
  cost_store_B balls_to_buy price_per_ball discount_per_ball_B = 1160 ∧
  cost_store_C balls_to_buy price_per_ball cashback_rate_C cashback_threshold_C = 1170 ∧
  cost_store_A balls_to_buy price_per_ball < cost_store_B balls_to_buy price_per_ball discount_per_ball_B ∧
  cost_store_A balls_to_buy price_per_ball < cost_store_C balls_to_buy price_per_ball cashback_rate_C cashback_threshold_C :=
by {
  -- placeholder for the proof
  sorry
}

end best_store_is_A_l17_1773


namespace total_animals_count_l17_1778

theorem total_animals_count (a m : ℕ) (h1 : a = 35) (h2 : a + 7 = m) : a + m = 77 :=
by
  sorry

end total_animals_count_l17_1778


namespace liquid_levels_proof_l17_1762

noncomputable def liquid_levels (H : ℝ) : ℝ × ℝ :=
  let ρ_water := 1000
  let ρ_gasoline := 600
  -- x = level drop in the left vessel
  let x := (3 / 14) * H
  let h_left := 0.9 * H - x
  let h_right := H
  (h_left, h_right)

theorem liquid_levels_proof (H : ℝ) (h : ℝ) :
  H > 0 →
  h = 0.9 * H →
  liquid_levels H = (0.69 * H, H) :=
by
  intros
  sorry

end liquid_levels_proof_l17_1762


namespace ratio_apps_optimal_l17_1775

theorem ratio_apps_optimal (max_apps : ℕ) (recommended_apps : ℕ) (apps_to_delete : ℕ) (current_apps : ℕ)
  (h_max_apps : max_apps = 50)
  (h_recommended_apps : recommended_apps = 35)
  (h_apps_to_delete : apps_to_delete = 20)
  (h_current_apps : current_apps = max_apps + apps_to_delete) :
  current_apps / recommended_apps = 2 :=
by {
  sorry
}

end ratio_apps_optimal_l17_1775


namespace max_area_rectangle_l17_1723

theorem max_area_rectangle (x y : ℝ) (h : 2 * x + 2 * y = 40) : x * y ≤ 100 :=
by
  sorry

end max_area_rectangle_l17_1723


namespace ironed_clothing_l17_1749

theorem ironed_clothing (shirts_rate pants_rate shirts_hours pants_hours : ℕ)
    (h1 : shirts_rate = 4)
    (h2 : pants_rate = 3)
    (h3 : shirts_hours = 3)
    (h4 : pants_hours = 5) :
    shirts_rate * shirts_hours + pants_rate * pants_hours = 27 := by
  sorry

end ironed_clothing_l17_1749


namespace rate_percent_correct_l17_1726

noncomputable def findRatePercent (P A T : ℕ) : ℚ :=
  let SI := A - P
  (SI * 100 : ℚ) / (P * T)

theorem rate_percent_correct :
  findRatePercent 12000 19500 7 = 8.93 := by
  sorry

end rate_percent_correct_l17_1726


namespace train_speed_correct_l17_1799

/-- Define the length of the train in meters -/
def length_train : ℝ := 120

/-- Define the length of the bridge in meters -/
def length_bridge : ℝ := 160

/-- Define the time taken to pass the bridge in seconds -/
def time_taken : ℝ := 25.2

/-- Define the expected speed of the train in meters per second -/
def expected_speed : ℝ := 11.1111

/-- Prove that the speed of the train is 11.1111 meters per second given conditions -/
theorem train_speed_correct :
  (length_train + length_bridge) / time_taken = expected_speed :=
by
  sorry

end train_speed_correct_l17_1799


namespace evaluate_f_of_f_of_3_l17_1789

def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 2

theorem evaluate_f_of_f_of_3 :
  f (f 3) = 2943 :=
by
  sorry

end evaluate_f_of_f_of_3_l17_1789


namespace grains_in_batch_l17_1733

-- Define the given constants from the problem
def total_rice_shi : ℕ := 1680
def sample_total_grains : ℕ := 250
def sample_containing_grains : ℕ := 25

-- Define the statement to be proven
theorem grains_in_batch : (total_rice_shi * (sample_containing_grains / sample_total_grains)) = 168 := by
  -- Proof steps will go here
  sorry

end grains_in_batch_l17_1733


namespace truck_distance_l17_1791

theorem truck_distance :
  let a1 := 8
  let d := 9
  let n := 40
  let an := a1 + (n - 1) * d
  let S_n := n / 2 * (a1 + an)
  S_n = 7340 :=
by
  sorry

end truck_distance_l17_1791


namespace calculate_sum_l17_1779

theorem calculate_sum : (2 / 20) + (3 / 50 * 5 / 100) + (4 / 1000) + (6 / 10000) = 0.1076 := 
by
  sorry

end calculate_sum_l17_1779


namespace luggage_max_length_l17_1785

theorem luggage_max_length
  (l w h : ℕ)
  (h_eq : h = 30)
  (ratio_l_w : l = 3 * w / 2)
  (sum_leq : l + w + h ≤ 160) :
  l ≤ 78 := sorry

end luggage_max_length_l17_1785


namespace perpendicular_bisector_l17_1746

theorem perpendicular_bisector (x y : ℝ) (h : -1 ≤ x ∧ x ≤ 3) (h_line : x - 2 * y + 1 = 0) : 
  2 * x - y - 1 = 0 :=
sorry

end perpendicular_bisector_l17_1746


namespace pond_eye_count_l17_1783

def total_animal_eyes (snakes alligators spiders snails : ℕ) 
    (snake_eyes alligator_eyes spider_eyes snail_eyes: ℕ) : ℕ :=
  snakes * snake_eyes + alligators * alligator_eyes + spiders * spider_eyes + snails * snail_eyes

theorem pond_eye_count : total_animal_eyes 18 10 5 15 2 2 8 2 = 126 := 
by
  sorry

end pond_eye_count_l17_1783


namespace bogatyrs_truthful_count_l17_1717

noncomputable def number_of_truthful_warriors (total_warriors: ℕ) (sword_yes: ℕ) (spear_yes: ℕ) (axe_yes: ℕ) (bow_yes: ℕ) : ℕ :=
  let total_yes := sword_yes + spear_yes + axe_yes + bow_yes
  let lying_warriors := (total_yes - total_warriors) / 2
  total_warriors - lying_warriors

theorem bogatyrs_truthful_count :
  number_of_truthful_warriors 33 13 15 20 27 = 12 := by
  sorry

end bogatyrs_truthful_count_l17_1717


namespace lowest_temperature_l17_1716

theorem lowest_temperature 
  (temps : Fin 5 → ℝ) 
  (avg_temp : (temps 0 + temps 1 + temps 2 + temps 3 + temps 4) / 5 = 60)
  (max_range : ∀ i j, temps i - temps j ≤ 75) : 
  ∃ L : ℝ, L = 0 ∧ ∃ i, temps i = L :=
by 
  sorry

end lowest_temperature_l17_1716


namespace avg_one_fourth_class_l17_1728

variable (N : ℕ) -- Total number of students

-- Define the average grade for the entire class
def avg_entire_class : ℝ := 84

-- Define the average grade of three fourths of the class
def avg_three_fourths_class : ℝ := 80

-- Statement to prove
theorem avg_one_fourth_class (A : ℝ) (h1 : 1/4 * A + 3/4 * avg_three_fourths_class = avg_entire_class) : 
  A = 96 := 
sorry

end avg_one_fourth_class_l17_1728


namespace line_equation_l17_1725

theorem line_equation (x y : ℝ) : 
  ((y = 1 → x = 2) ∧ ((x,y) = (1,1) ∨ (x,y) = (3,5)))
  → (2 * x - y - 3 = 0) ∨ (x = 2) :=
sorry

end line_equation_l17_1725


namespace solve_inequality_l17_1780

-- Define the inequality
def inequality (x : ℝ) : Prop :=
  -3 * (x^2 - 4 * x + 16) * (x^2 + 6 * x + 8) / ((x^3 + 64) * (Real.sqrt (x^2 + 4 * x + 4))) ≤ x^2 + x - 3

-- Define the solution set
def solution_set (x : ℝ) : Prop :=
  x ∈ Set.Iic (-4) ∪ {x : ℝ | -4 < x ∧ x ≤ -3} ∪ {x : ℝ | -2 < x ∧ x ≤ -1} ∪ Set.Ici 0

-- The theorem statement, which we need to prove
theorem solve_inequality : ∀ x : ℝ, inequality x ↔ solution_set x :=
by
  intro x
  sorry

end solve_inequality_l17_1780


namespace minimum_value_of_f_range_of_t_l17_1729

def f (x : ℝ) : ℝ := abs (x + 2) + abs (x - 4)

theorem minimum_value_of_f : ∀ x, f x ≥ 6 ∧ ∃ x0 : ℝ, f x0 = 6 := 
by sorry

theorem range_of_t (t : ℝ) : (t ≤ -2 ∨ t ≥ 3) ↔ ∃ x : ℝ, -3 ≤ x ∧ x ≤ 5 ∧ f x ≤ t^2 - t :=
by sorry

end minimum_value_of_f_range_of_t_l17_1729


namespace triangles_with_positive_area_l17_1777

-- Define the set of points in the coordinate grid
def points := { p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 ≤ 4 ∧ 1 ≤ p.2 ∧ p.2 ≤ 4 }

-- Number of ways to choose 3 points from the grid
def total_triples := Nat.choose 16 3

-- Number of collinear triples
def collinear_triples := 32 + 8 + 4

-- Number of triangles with positive area
theorem triangles_with_positive_area :
  (total_triples - collinear_triples) = 516 :=
by
  -- Definitions for total_triples and collinear_triples.
  -- Proof steps would go here.
  sorry

end triangles_with_positive_area_l17_1777


namespace cylinder_volume_ratio_l17_1750

theorem cylinder_volume_ratio (s : ℝ) :
  let r := s / 2
  let h := s
  let V_cylinder := π * r^2 * h
  let V_cube := s^3
  V_cylinder / V_cube = π / 4 :=
by
  sorry

end cylinder_volume_ratio_l17_1750


namespace cost_of_black_and_white_drawing_l17_1705

-- Given the cost of the color drawing is 1.5 times the cost of the black and white drawing
-- and John paid $240 for the color drawing, we need to prove the cost of the black and white drawing is $160.

theorem cost_of_black_and_white_drawing (C : ℝ) (h : 1.5 * C = 240) : C = 160 :=
by
  sorry

end cost_of_black_and_white_drawing_l17_1705


namespace solve_inequalities_l17_1739

theorem solve_inequalities (x : ℝ) (h₁ : 5 * x - 8 > 12 - 2 * x) (h₂ : |x - 1| ≤ 3) : 
  (20 / 7) < x ∧ x ≤ 4 :=
by
  sorry

end solve_inequalities_l17_1739


namespace gas_usage_correct_l17_1737

def starting_gas : ℝ := 0.5
def ending_gas : ℝ := 0.16666666666666666

theorem gas_usage_correct : starting_gas - ending_gas = 0.33333333333333334 := by
  sorry

end gas_usage_correct_l17_1737


namespace find_sale4_l17_1793

variable (sale1 sale2 sale3 sale5 sale6 avg : ℕ)
variable (total_sales : ℕ := 6 * avg)
variable (known_sales : ℕ := sale1 + sale2 + sale3 + sale5 + sale6)
variable (sale4 : ℕ := total_sales - known_sales)

theorem find_sale4 (h1 : sale1 = 6235) (h2 : sale2 = 6927) (h3 : sale3 = 6855)
                   (h5 : sale5 = 6562) (h6 : sale6 = 5191) (h_avg : avg = 6500) :
  sale4 = 7225 :=
by 
  sorry

end find_sale4_l17_1793


namespace range_of_a_l17_1792

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 + a * x + 4 < 0) ↔ (a < -4 ∨ a > 4) := by
  sorry

end range_of_a_l17_1792


namespace largest_odd_integer_satisfying_inequality_l17_1700

theorem largest_odd_integer_satisfying_inequality : 
  ∃ (x : ℤ), (x % 2 = 1) ∧ (1 / 4 < x / 6) ∧ (x / 6 < 7 / 9) ∧ (∀ y : ℤ, (y % 2 = 1) ∧ (1 / 4 < y / 6) ∧ (y / 6 < 7 / 9) → y ≤ x) :=
sorry

end largest_odd_integer_satisfying_inequality_l17_1700


namespace journey_total_distance_l17_1753

-- Define the conditions
def miles_already_driven : ℕ := 642
def miles_to_drive : ℕ := 558

-- The total distance of the journey
def total_distance : ℕ := miles_already_driven + miles_to_drive

-- Prove that the total distance of the journey equals 1200 miles
theorem journey_total_distance : total_distance = 1200 := 
by
  -- here the proof would go
  sorry

end journey_total_distance_l17_1753


namespace area_of_PQRS_l17_1786

noncomputable def length_EF := 6
noncomputable def width_EF := 4

noncomputable def area_PQRS := (length_EF + 6 * Real.sqrt 3) * (width_EF + 4 * Real.sqrt 3)

theorem area_of_PQRS :
  area_PQRS = 60 + 48 * Real.sqrt 3 := by
  sorry

end area_of_PQRS_l17_1786


namespace solve_system_eqns_l17_1769

theorem solve_system_eqns (x y z a : ℝ)
  (h1 : x + y + z = a)
  (h2 : x^2 + y^2 + z^2 = a^2)
  (h3 : x^3 + y^3 + z^3 = a^3) :
  (x = a ∧ y = 0 ∧ z = 0) ∨
  (x = 0 ∧ y = a ∧ z = 0) ∨
  (x = 0 ∧ y = 0 ∧ z = a) := 
by
  sorry

end solve_system_eqns_l17_1769


namespace square_of_other_leg_l17_1714

-- Conditions
variable (a b c : ℝ)
variable (h₁ : c = a + 2)
variable (h₂ : a^2 + b^2 = c^2)

-- The theorem statement
theorem square_of_other_leg (a b c : ℝ) (h₁ : c = a + 2) (h₂ : a^2 + b^2 = c^2) : b^2 = 4 * a + 4 :=
by
  sorry

end square_of_other_leg_l17_1714


namespace flagpole_height_in_inches_l17_1718

theorem flagpole_height_in_inches
  (height_lamppost shadow_lamppost : ℚ)
  (height_flagpole shadow_flagpole : ℚ)
  (h₁ : height_lamppost = 50)
  (h₂ : shadow_lamppost = 12)
  (h₃ : shadow_flagpole = 18 / 12) :
  height_flagpole * 12 = 75 :=
by
  -- Note: To keep the theorem concise, proof steps are omitted
  sorry

end flagpole_height_in_inches_l17_1718


namespace axis_of_symmetry_parabola_l17_1772

theorem axis_of_symmetry_parabola (a b : ℝ) (h₁ : a = -3) (h₂ : b = 6) :
  -b / (2 * a) = 1 :=
by
  sorry

end axis_of_symmetry_parabola_l17_1772


namespace stella_profit_l17_1761

def price_of_doll := 5
def price_of_clock := 15
def price_of_glass := 4

def number_of_dolls := 3
def number_of_clocks := 2
def number_of_glasses := 5

def cost := 40

def dolls_sales := number_of_dolls * price_of_doll
def clocks_sales := number_of_clocks * price_of_clock
def glasses_sales := number_of_glasses * price_of_glass

def total_sales := dolls_sales + clocks_sales + glasses_sales

def profit := total_sales - cost

theorem stella_profit : profit = 25 :=
by 
  sorry

end stella_profit_l17_1761


namespace time_spent_moving_l17_1755

noncomputable def time_per_trip_filling : ℝ := 15
noncomputable def time_per_trip_driving : ℝ := 30
noncomputable def time_per_trip_unloading : ℝ := 20
noncomputable def number_of_trips : ℕ := 10

theorem time_spent_moving :
  10.83 = (time_per_trip_filling + time_per_trip_driving + time_per_trip_unloading) * number_of_trips / 60 :=
by
  sorry

end time_spent_moving_l17_1755


namespace paper_cups_pallets_l17_1724

theorem paper_cups_pallets (total_pallets : ℕ) (paper_towels_fraction tissues_fraction paper_plates_fraction : ℚ) :
  total_pallets = 20 → paper_towels_fraction = 1 / 2 → tissues_fraction = 1 / 4 → paper_plates_fraction = 1 / 5 →
  total_pallets - (total_pallets * paper_towels_fraction + total_pallets * tissues_fraction + total_pallets * paper_plates_fraction) = 1 :=
by sorry

end paper_cups_pallets_l17_1724


namespace rectangle_perimeter_l17_1701

theorem rectangle_perimeter (s : ℝ) (h1 : 4 * s = 180) :
    let length := s
    let width := s / 3
    2 * (length + width) = 120 := 
by
  sorry

end rectangle_perimeter_l17_1701


namespace upstream_distance_is_48_l17_1720

variables (distance_downstream time_downstream time_upstream speed_stream : ℝ)
variables (speed_boat distance_upstream : ℝ)

-- Given conditions
axiom h1 : distance_downstream = 84
axiom h2 : time_downstream = 2
axiom h3 : time_upstream = 2
axiom h4 : speed_stream = 9

-- Define the effective speeds
def speed_downstream (speed_boat speed_stream : ℝ) := speed_boat + speed_stream
def speed_upstream (speed_boat speed_stream : ℝ) := speed_boat - speed_stream

-- Equations based on travel times and distances
axiom eq1 : distance_downstream = (speed_downstream speed_boat speed_stream) * time_downstream
axiom eq2 : distance_upstream = (speed_upstream speed_boat speed_stream) * time_upstream

-- Theorem to prove the distance rowed upstream is 48 km
theorem upstream_distance_is_48 :
  distance_upstream = 48 :=
by
  sorry

end upstream_distance_is_48_l17_1720


namespace arithmetic_evaluation_l17_1798

theorem arithmetic_evaluation : 6 * 2 - 3 = 9 := by
  sorry

end arithmetic_evaluation_l17_1798


namespace smallest_number_condition_l17_1784

def smallest_number := 1621432330
def primes := [29, 53, 37, 41, 47, 61]
def lcm_of_primes := primes.prod

theorem smallest_number_condition :
  ∃ k : ℕ, 5 * (smallest_number + 11) = k * lcm_of_primes ∧
          (∀ y, (∃ m : ℕ, 5 * (y + 11) = m * lcm_of_primes) → smallest_number ≤ y) :=
by
  -- The proof goes here
  sorry

#print smallest_number_condition

end smallest_number_condition_l17_1784


namespace greatest_possible_x_max_possible_x_l17_1787

theorem greatest_possible_x (x : ℕ) (h : Nat.lcm x (Nat.lcm 15 21) = 105) : x ≤ 105 :=
by
  -- Proof goes here
  sorry

-- As a corollary, we can state the maximum value of x
theorem max_possible_x : 105 ≤ 105 :=
by
  -- Proof goes here
  exact le_refl 105

end greatest_possible_x_max_possible_x_l17_1787


namespace smallest_n_positive_odd_integer_l17_1730

theorem smallest_n_positive_odd_integer (n : ℕ) (h1 : n % 2 = 1) (h2 : 3 ^ ((n + 1)^2 / 5) > 500) : n = 6 := sorry

end smallest_n_positive_odd_integer_l17_1730


namespace noRepeatedDigitsFourDigit_noRepeatedDigitsFiveDigitDiv5_noRepeatedDigitsFourDigitGreaterThan1325_l17_1711

-- Problem 1: Four-digit numbers with no repeated digits
theorem noRepeatedDigitsFourDigit :
  ∃ (n : ℕ), (n = 120) := sorry

-- Problem 2: Five-digit numbers with no repeated digits and divisible by 5
theorem noRepeatedDigitsFiveDigitDiv5 :
  ∃ (n : ℕ), (n = 216) := sorry

-- Problem 3: Four-digit numbers with no repeated digits and greater than 1325
theorem noRepeatedDigitsFourDigitGreaterThan1325 :
  ∃ (n : ℕ), (n = 181) := sorry

end noRepeatedDigitsFourDigit_noRepeatedDigitsFiveDigitDiv5_noRepeatedDigitsFourDigitGreaterThan1325_l17_1711


namespace veronica_flashlight_distance_l17_1706

theorem veronica_flashlight_distance (V F Vel : ℕ) 
  (h1 : F = 3 * V)
  (h2 : Vel = 5 * F - 2000)
  (h3 : Vel = V + 12000) : 
  V = 1000 := 
by {
  sorry 
}

end veronica_flashlight_distance_l17_1706


namespace longer_train_length_l17_1710

def length_of_longer_train
  (speed_train1 : ℝ) (speed_train2 : ℝ)
  (length_shorter_train : ℝ) (time_to_clear : ℝ)
  (relative_speed : ℝ := (speed_train1 + speed_train2) * 1000 / 3600)
  (total_distance : ℝ := relative_speed * time_to_clear) : ℝ :=
  total_distance - length_shorter_train

theorem longer_train_length :
  length_of_longer_train 80 55 121 7.626056582140095 = 164.9771230827526 :=
by
  unfold length_of_longer_train
  norm_num
  sorry  -- This placeholder is used to avoid writing out the full proof.

end longer_train_length_l17_1710


namespace sum_sequence_eq_l17_1759

noncomputable def S (n : ℕ) : ℝ := Real.log (1 + n) / Real.log 0.1

theorem sum_sequence_eq :
  (S 99 - S 9) = -1 := by
  sorry

end sum_sequence_eq_l17_1759


namespace y_intercept_of_line_l17_1766

theorem y_intercept_of_line : 
  (∃ t : ℝ, 4 - 4 * t = 0) → (∃ y : ℝ, y = -2 + 3 * 1) := 
by
  sorry

end y_intercept_of_line_l17_1766


namespace lydia_candy_problem_l17_1704

theorem lydia_candy_problem :
  ∃ m: ℕ, (∀ k: ℕ, (k * 24 = Nat.lcm (Nat.lcm 16 18) 20) → k ≥ m) ∧ 24 * m = Nat.lcm (Nat.lcm 16 18) 20 ∧ m = 30 :=
by
  sorry

end lydia_candy_problem_l17_1704


namespace no_digit_c_make_2C4_multiple_of_5_l17_1721

theorem no_digit_c_make_2C4_multiple_of_5 : ∀ C, ¬ (C ≥ 0 ∧ C ≤ 9 ∧ (20 * 10 + C * 10 + 4) % 5 = 0) :=
by intros C hC; sorry

end no_digit_c_make_2C4_multiple_of_5_l17_1721


namespace expected_profit_l17_1707

namespace DailyLottery

/-- Definitions for the problem -/

def ticket_cost : ℝ := 2
def first_prize : ℝ := 100
def second_prize : ℝ := 10
def prob_first_prize : ℝ := 0.001
def prob_second_prize : ℝ := 0.1
def prob_no_prize : ℝ := 1 - prob_first_prize - prob_second_prize

/-- Expected profit calculation as a theorem -/

theorem expected_profit :
  (first_prize * prob_first_prize + second_prize * prob_second_prize + 0 * prob_no_prize) - ticket_cost = -0.9 :=
by
  sorry

end DailyLottery

end expected_profit_l17_1707


namespace general_term_arithmetic_sequence_sum_terms_sequence_l17_1770

noncomputable def a_n (n : ℕ) : ℤ := 
  2 * (n : ℤ) - 1

theorem general_term_arithmetic_sequence :
  ∀ n : ℕ, a_n n = 2 * (n : ℤ) - 1 :=
by sorry

noncomputable def c (n : ℕ) : ℚ := 
  1 / ((2 * (n : ℤ) - 1) * (2 * (n + 1) - 1))

noncomputable def T_n (n : ℕ) : ℚ :=
  (1 / 2 : ℚ) * (1 - (1 / (2 * (n : ℤ) + 1)))

theorem sum_terms_sequence :
  ∀ n : ℕ, T_n n = (n : ℚ) / (2 * (n : ℤ) + 1) :=
by sorry

end general_term_arithmetic_sequence_sum_terms_sequence_l17_1770


namespace cyclic_inequality_l17_1771

theorem cyclic_inequality
    (x1 x2 x3 x4 x5 : ℝ)
    (h1 : 0 < x1)
    (h2 : 0 < x2)
    (h3 : 0 < x3)
    (h4 : 0 < x4)
    (h5 : 0 < x5) :
    (x1 + x2 + x3 + x4 + x5)^2 > 4 * (x1 * x2 + x2 * x3 + x3 * x4 + x4 * x5 + x5 * x1) :=
by
  sorry

end cyclic_inequality_l17_1771


namespace sugar_cups_l17_1796

theorem sugar_cups (S : ℕ) (h1 : 21 = S + 8) : S = 13 := 
by { sorry }

end sugar_cups_l17_1796


namespace contrapositive_proof_l17_1741

theorem contrapositive_proof (a b : ℝ) : 
  (a > b → a - 1 > b - 1) ↔ (a - 1 ≤ b - 1 → a ≤ b) :=
sorry

end contrapositive_proof_l17_1741


namespace incorrect_statement_B_is_wrong_l17_1767

variable (number_of_students : ℕ) (sample_size : ℕ) (population : Set ℕ) (sample : Set ℕ)

-- Conditions
def school_population_is_4000 := number_of_students = 4000
def sample_selected_is_400 := sample_size = 400
def valid_population := population = { x | x < 4000 }
def valid_sample := sample = { x | x < 400 }

-- Incorrect statement (as per given solution)
def incorrect_statement_B := ¬(∀ student ∈ population, true)

theorem incorrect_statement_B_is_wrong 
  (h1 : school_population_is_4000 number_of_students)
  (h2 : sample_selected_is_400 sample_size)
  (h3 : valid_population population)
  (h4 : valid_sample sample)
  : incorrect_statement_B population :=
sorry

end incorrect_statement_B_is_wrong_l17_1767


namespace quadratic_eq_coeff_m_l17_1735

theorem quadratic_eq_coeff_m (m : ℤ) : 
  (|m| = 2 ∧ m + 2 ≠ 0) → m = 2 := 
by
  intro h
  sorry

end quadratic_eq_coeff_m_l17_1735


namespace sam_spent_136_96_l17_1764

def glove_original : Real := 35
def glove_discount : Real := 0.20
def baseball_price : Real := 15
def bat_original : Real := 50
def bat_discount : Real := 0.10
def cleats_price : Real := 30
def cap_price : Real := 10
def tax_rate : Real := 0.07

def total_spent (glove_original : Real) (glove_discount : Real) (baseball_price : Real) (bat_original : Real) (bat_discount : Real) (cleats_price : Real) (cap_price : Real) (tax_rate : Real) : Real :=
  let glove_price := glove_original - (glove_discount * glove_original)
  let bat_price := bat_original - (bat_discount * bat_original)
  let total_before_tax := glove_price + baseball_price + bat_price + cleats_price + cap_price
  let tax_amount := total_before_tax * tax_rate
  total_before_tax + tax_amount

theorem sam_spent_136_96 :
  total_spent glove_original glove_discount baseball_price bat_original bat_discount cleats_price cap_price tax_rate = 136.96 :=
sorry

end sam_spent_136_96_l17_1764


namespace function_property_l17_1703

theorem function_property 
  (f : ℝ → ℝ) 
  (hf : ∀ x, x ≠ 0 → f (1 - 2 * x) = (1 - x^2) / (x^2)) 
  : 
  (f (1 / 2) = 15) ∧
  (∀ x, x ≠ 1 → f (x) = 4 / (x - 1)^2 - 1) ∧
  (∀ x, x ≠ 0 → x ≠ 1 → f (1 / x) = 4 * x^2 / (x - 1)^2 - 1) :=
by {
  sorry
}

end function_property_l17_1703


namespace remainder_when_sum_divided_by_7_l17_1747

theorem remainder_when_sum_divided_by_7 (a b c : ℕ) (ha : a < 7) (hb : b < 7) (hc : c < 7)
  (h1 : a * b * c ≡ 1 [MOD 7])
  (h2 : 4 * c ≡ 3 [MOD 7])
  (h3 : 5 * b ≡ 4 + b [MOD 7]) :
  (a + b + c) % 7 = 6 := by
  sorry

end remainder_when_sum_divided_by_7_l17_1747


namespace time_to_walk_against_walkway_150_l17_1756

def v_p := 4 / 3
def v_w := 2 - v_p
def distance := 100
def time_against_walkway := distance / (v_p - v_w)

theorem time_to_walk_against_walkway_150 :
  time_against_walkway = 150 := by
  -- Note: Proof goes here (not required)
  sorry

end time_to_walk_against_walkway_150_l17_1756


namespace length_of_platform_is_correct_l17_1727

-- Given conditions:
def length_of_train : ℕ := 250
def speed_of_train_kmph : ℕ := 72
def time_to_cross_platform : ℕ := 20

-- Convert speed from kmph to m/s
def speed_of_train_mps : ℕ := speed_of_train_kmph * 1000 / 3600

-- Distance covered in 20 seconds
def distance_covered : ℕ := speed_of_train_mps * time_to_cross_platform

-- Length of the platform
def length_of_platform : ℕ := distance_covered - length_of_train

-- The proof statement
theorem length_of_platform_is_correct :
  length_of_platform = 150 := by
  -- This proof would involve the detailed calculations and verifications as laid out in the solution steps.
  sorry

end length_of_platform_is_correct_l17_1727


namespace train_john_arrival_probability_l17_1768

-- Define the probability of independent uniform distributions on the interval [0, 120]
noncomputable def probability_train_present_when_john_arrives : ℝ :=
  let total_square_area := (120 : ℝ) * 120
  let triangle_area := (1 / 2) * 90 * 30
  let trapezoid_area := (1 / 2) * (30 + 0) * 30
  let total_shaded_area := triangle_area + trapezoid_area
  total_shaded_area / total_square_area

theorem train_john_arrival_probability :
  probability_train_present_when_john_arrives = 1 / 8 :=
by {
  sorry
}

end train_john_arrival_probability_l17_1768


namespace graduating_class_total_l17_1797

theorem graduating_class_total (boys girls : ℕ) 
  (h_boys : boys = 138)
  (h_more_girls : girls = boys + 69) :
  boys + girls = 345 :=
sorry

end graduating_class_total_l17_1797


namespace max_value_f_min_value_a_l17_1751

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - 4 * Real.pi / 3) + 2 * (Real.cos x)^2

theorem max_value_f :
  ∀ x, f x ≤ 2 ∧ (∃ k : ℤ, x = k * Real.pi - Real.pi / 6) → f x = 2 :=
by { sorry }

variables {A B C a b c : ℝ}

noncomputable def f' (x : ℝ) : ℝ := Real.cos (2 * x +  Real.pi / 3) + 1

theorem min_value_a
  (h1 : f' (B + C) = 3/2)
  (h2 : b + c = 2)
  (h3 : A + B + C = Real.pi)
  (h4 : Real.cos A = 1/2) :
  ∃ a, ∀ b c, a^2 = b^2 + c^2 - 2 * b * c * Real.cos A ∧ a ≥ 1 :=
by { sorry }

end max_value_f_min_value_a_l17_1751


namespace ellipse_area_l17_1765

/-- 
In a certain ellipse, the endpoints of the major axis are (1, 6) and (21, 6). 
Also, the ellipse passes through the point (19, 9). Prove that the area of the ellipse is 50π. 
-/
theorem ellipse_area : 
  let a := 10
  let b := 5 
  let center := (11, 6)
  let endpoints_major := [(1, 6), (21, 6)]
  let point_on_ellipse := (19, 9)
  ∀ x y, ((x - 11)^2 / a^2) + ((y - 6)^2 / b^2) = 1 → 
    (x, y) = (19, 9) →  -- given point on the ellipse
    (endpoints_major = [(1, 6), (21, 6)]) →  -- given endpoints of the major axis
    50 * Real.pi = π * a * b := 
by
  sorry

end ellipse_area_l17_1765


namespace find_m_value_l17_1774

theorem find_m_value :
  let x_values := [8, 9.5, m, 10.5, 12]
  let y_values := [16, 10, 8, 6, 5]
  let regression_eq (x : ℝ) := -3.5 * x + 44
  let avg (l : List ℝ) := l.sum / l.length
  avg y_values = 9 →
  avg x_values = (40 + m) / 5 →
  9 = regression_eq (avg x_values) →
  m = 10 :=
by
  sorry

end find_m_value_l17_1774


namespace side_length_of_square_base_l17_1763

theorem side_length_of_square_base (area slant_height : ℝ) (h_area : area = 120) (h_slant_height : slant_height = 40) :
  ∃ s : ℝ, area = 0.5 * s * slant_height ∧ s = 6 :=
by
  sorry

end side_length_of_square_base_l17_1763


namespace benny_money_l17_1742

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

end benny_money_l17_1742


namespace solve_for_y_l17_1743

-- Define the given condition as a Lean definition
def equation (y : ℝ) : Prop :=
  (2 / y) + ((3 / y) / (6 / y)) = 1.2

-- Theorem statement proving the solution given the condition
theorem solve_for_y (y : ℝ) (h : equation y) : y = 20 / 7 := by
  sorry

-- Example usage to instantiate and make use of the definition
example : equation (20 / 7) := by
  unfold equation
  sorry

end solve_for_y_l17_1743


namespace percent_sold_second_day_l17_1757

-- Defining the problem conditions
def initial_pears (x : ℕ) : ℕ := x
def pears_sold_first_day (x : ℕ) : ℕ := (20 * x) / 100
def pears_remaining_after_first_sale (x : ℕ) : ℕ := x - pears_sold_first_day x
def pears_thrown_away_first_day (x : ℕ) : ℕ := (50 * pears_remaining_after_first_sale x) / 100
def pears_remaining_after_first_day (x : ℕ) : ℕ := pears_remaining_after_first_sale x - pears_thrown_away_first_day x
def total_pears_thrown_away (x : ℕ) : ℕ := (72 * x) / 100
def pears_thrown_away_second_day (x : ℕ) : ℕ := total_pears_thrown_away x - pears_thrown_away_first_day x
def pears_remaining_after_second_day (x : ℕ) : ℕ := pears_remaining_after_first_day x - pears_thrown_away_second_day x

-- Prove that the vendor sold 20% of the remaining pears on the second day
theorem percent_sold_second_day (x : ℕ) (h : x > 0) :
  ((pears_remaining_after_second_day x * 100) / pears_remaining_after_first_day x) = 20 :=
by 
  sorry

end percent_sold_second_day_l17_1757


namespace total_cost_is_15_l17_1722

def toast_cost : ℕ := 1
def egg_cost : ℕ := 3

def dale_toast : ℕ := 2
def dale_eggs : ℕ := 2

def andrew_toast : ℕ := 1
def andrew_eggs : ℕ := 2

def dale_breakfast_cost := dale_toast * toast_cost + dale_eggs * egg_cost
def andrew_breakfast_cost := andrew_toast * toast_cost + andrew_eggs * egg_cost

def total_breakfast_cost := dale_breakfast_cost + andrew_breakfast_cost

theorem total_cost_is_15 : total_breakfast_cost = 15 := by
  sorry

end total_cost_is_15_l17_1722


namespace calculate_expression_l17_1702

theorem calculate_expression : 14 - (-12) + (-25) - 17 = -16 := by
  -- definitions from conditions are understood and used here implicitly
  sorry

end calculate_expression_l17_1702


namespace lisa_matching_pair_probability_l17_1712

theorem lisa_matching_pair_probability :
  let total_socks := 22
  let gray_socks := 12
  let white_socks := 10
  let total_pairs := total_socks * (total_socks - 1) / 2
  let gray_pairs := gray_socks * (gray_socks - 1) / 2
  let white_pairs := white_socks * (white_socks - 1) / 2
  let matching_pairs := gray_pairs + white_pairs
  let probability := matching_pairs / total_pairs
  probability = (111 / 231) :=
by
  sorry

end lisa_matching_pair_probability_l17_1712


namespace gel_pen_ratio_l17_1719

-- Definitions corresponding to the conditions in the problem
variables (x y : ℕ) (b g : ℝ)

-- The total amount paid 
def total_amount := x * b + y * g

-- Condition given in the problem
def condition1 := (x + y) * g = 4 * total_amount x y b g
def condition2 := (x + y) * b = (1/2) * total_amount x y b g

-- The theorem to prove the ratio of the price of a gel pen to a ballpoint pen is 8
theorem gel_pen_ratio (x y : ℕ) (b g : ℝ) (h1 : condition1 x y b g) (h2 : condition2 x y b g) : 
  g = 8 * b := by
  sorry

end gel_pen_ratio_l17_1719


namespace base_b_three_digit_count_l17_1758

-- Define the condition that counts the valid three-digit numbers in base b
def num_three_digit_numbers (b : ℕ) : ℕ :=
  (b - 1) ^ 2 * b

-- Define the specific problem statement
theorem base_b_three_digit_count :
  num_three_digit_numbers 4 = 72 :=
by
  -- Proof skipped as per the instruction
  sorry

end base_b_three_digit_count_l17_1758


namespace range_of_m_l17_1738

theorem range_of_m (a m x : ℝ) (p q : Prop) :
  (p ↔ ∃ (a : ℝ) (m : ℝ), ∀ (x : ℝ), 4 * x^2 - 2 * a * x + 2 * a + 5 = 0) →
  (q ↔ 1 - m ≤ x ∧ x ≤ 1 + m ∧ m > 0) →
  (¬ p → ¬ q) →
  (∀ a, -2 ≤ a ∧ a ≤ 10) →
  (1 - m ≤ -2) ∧ (1 + m ≥ 10) →
  m ≥ 9 :=
by
  intros hp hq npnq ha hm
  sorry  -- Proof omitted

end range_of_m_l17_1738


namespace factorization_of_cubic_polynomial_l17_1782

theorem factorization_of_cubic_polynomial (x y z : ℝ) : 
  x^3 + y^3 + z^3 - 3 * x * y * z = (x + y + z) * (x^2 + y^2 + z^2 - x * y - y * z - z * x) := 
by sorry

end factorization_of_cubic_polynomial_l17_1782


namespace tyler_saltwater_animals_l17_1744

/-- Tyler had 56 aquariums for saltwater animals and each aquarium has 39 animals in it. 
    We need to prove that the total number of saltwater animals Tyler has is 2184. --/
theorem tyler_saltwater_animals : (56 * 39) = 2184 := by
  sorry

end tyler_saltwater_animals_l17_1744


namespace x_intercept_of_line_l17_1713

variables (x₁ y₁ x₂ y₂ : ℝ) (m : ℝ)

/-- The line passing through the points (-1, 1) and (3, 9) has an x-intercept of -3/2. -/
theorem x_intercept_of_line : 
  let x₁ := -1
  let y₁ := 1
  let x₂ := 3
  let y₂ := 9
  let m := (y₂ - y₁) / (x₂ - x₁)
  let b := y₁ - m * x₁
  (0 : ℝ) = m * (x : ℝ) + b → x = (-3 / 2) := 
by 
  sorry

end x_intercept_of_line_l17_1713


namespace average_interest_rate_l17_1709

theorem average_interest_rate
  (x : ℝ)
  (h₀ : 0 ≤ x)
  (h₁ : x ≤ 5000)
  (h₂ : 0.05 * x = 0.03 * (5000 - x)) :
  (0.05 * x + 0.03 * (5000 - x)) / 5000 = 0.0375 :=
by
  sorry

end average_interest_rate_l17_1709


namespace problem_1_problem_2_l17_1734

noncomputable def f (x a : ℝ) : ℝ := |2 * x - a| + |2 * x - 1|

-- Problem 1: When a = -1, prove the solution set for f(x) ≤ 2 is [-1/2, 1/2].
theorem problem_1 (x : ℝ) : (f x (-1) ≤ 2) ↔ (-1/2 ≤ x ∧ x ≤ 1/2) := 
sorry

-- Problem 2: If the solution set of f(x) ≤ |2x + 1| contains the interval [1/2, 1], find the range of a.
theorem problem_2 (a : ℝ) : (∀ x, (1/2 ≤ x ∧ x ≤ 1) → f x a ≤ |2 * x + 1|) ↔ (0 ≤ a ∧ a ≤ 3) :=
sorry

end problem_1_problem_2_l17_1734


namespace who_threw_at_third_child_l17_1788

-- Definitions based on conditions
def children_count : ℕ := 43

def threw_snowball (i j : ℕ) : Prop :=
∃ k, i = (k % children_count).succ ∧ j = ((k + 1) % children_count).succ

-- Conditions
axiom cond_1 : threw_snowball 1 (1 + 1) -- child 1 threw a snowball at the child who threw a snowball at child 2
axiom cond_2 : threw_snowball 2 (2 + 1) -- child 2 threw a snowball at the child who threw a snowball at child 3
axiom cond_3 : threw_snowball 43 1 -- child 43 threw a snowball at the child who threw a snowball at the first child

-- Question to prove
theorem who_threw_at_third_child : threw_snowball 24 3 :=
sorry

end who_threw_at_third_child_l17_1788


namespace total_fare_for_20km_l17_1776

def base_fare : ℝ := 8
def fare_per_km_from_3_to_10 : ℝ := 1.5
def fare_per_km_beyond_10 : ℝ := 0.8

def fare_for_first_3km : ℝ := base_fare
def fare_for_3_to_10_km : ℝ := 7 * fare_per_km_from_3_to_10
def fare_for_beyond_10_km : ℝ := 10 * fare_per_km_beyond_10

theorem total_fare_for_20km : fare_for_first_3km + fare_for_3_to_10_km + fare_for_beyond_10_km = 26.5 :=
by
  sorry

end total_fare_for_20km_l17_1776


namespace fraction_power_seven_l17_1795

theorem fraction_power_seven : (5 / 3 : ℚ) ^ 7 = 78125 / 2187 := 
by
  sorry

end fraction_power_seven_l17_1795


namespace find_a_given_integer_roots_l17_1790

-- Given polynomial equation and the condition of integer roots
theorem find_a_given_integer_roots (a : ℤ) :
    (∃ x y : ℤ, x ≠ y ∧ (x^2 - (a+8)*x + 8*a - 1 = 0) ∧ (y^2 - (a+8)*y + 8*a - 1 = 0)) → 
    a = 8 := 
by
  sorry

end find_a_given_integer_roots_l17_1790


namespace binom_60_3_l17_1708

theorem binom_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binom_60_3_l17_1708


namespace arrange_abc_l17_1794

theorem arrange_abc (a b c : ℝ) (h1 : a = Real.log 2 / Real.log 2)
                               (h2 : b = Real.sqrt 2)
                               (h3 : c = Real.cos ((3 / 4) * Real.pi)) :
  c < a ∧ a < b :=
by
  sorry

end arrange_abc_l17_1794
