import Mathlib

namespace derivative_at_0_eq_6_l967_96760

-- Definition of the function
def f (x : ℝ) : ℝ := (2 * x + 1)^3

-- Theorem statement indicating the derivative at x = 0 is 6
theorem derivative_at_0_eq_6 : (deriv f 0) = 6 := 
by 
  sorry -- The proof is omitted as per the instructions

end derivative_at_0_eq_6_l967_96760


namespace find_a8_l967_96732

theorem find_a8 (a : ℕ → ℝ) 
  (h_arith_seq : ∀ n : ℕ, (1 / (a n + 1)) = (1 / (a 0 + 1)) + n * ((1 / (a 1 + 1 - 1)) / 3)) 
  (h2 : a 2 = 3) 
  (h5 : a 5 = 1) : 
  a 8 = 1 / 3 :=
by
  sorry

end find_a8_l967_96732


namespace bathroom_area_is_50_square_feet_l967_96712

/-- A bathroom has 10 6-inch tiles along its width and 20 6-inch tiles along its length. --/
def bathroom_width_inches := 10 * 6
def bathroom_length_inches := 20 * 6

/-- Convert width and length from inches to feet. --/
def bathroom_width_feet := bathroom_width_inches / 12
def bathroom_length_feet := bathroom_length_inches / 12

/-- Calculate the square footage of the bathroom. --/
def bathroom_square_footage := bathroom_width_feet * bathroom_length_feet

/-- The square footage of the bathroom is 50 square feet. --/
theorem bathroom_area_is_50_square_feet : bathroom_square_footage = 50 := by
  sorry

end bathroom_area_is_50_square_feet_l967_96712


namespace find_counterfeit_10_l967_96772

theorem find_counterfeit_10 (coins : Fin 10 → ℕ) (h_counterfeit : ∃ k, ∀ i, i ≠ k → coins i < coins k) : 
  ∃ w : ℕ → ℕ → Prop, (∀ g1 g2, g1 ≠ g2 → w g1 g2 ∨ w g2 g1) → 
  ∃ k, ∀ i, i ≠ k → coins i < coins k :=
sorry

end find_counterfeit_10_l967_96772


namespace find_x_l967_96767

-- Definitions based on provided conditions

def rectangle_length (x : ℝ) : ℝ := 4 * x
def rectangle_width (x : ℝ) : ℝ := x + 7
def rectangle_area (x : ℝ) : ℝ := rectangle_length x * rectangle_width x
def rectangle_perimeter (x : ℝ) : ℝ := 2 * rectangle_length x + 2 * rectangle_width x

-- Theorem statement
theorem find_x (x : ℝ) (h : rectangle_area x = 2 * rectangle_perimeter x) : x = 1 := 
sorry

end find_x_l967_96767


namespace smallest_pos_int_mult_4410_sq_l967_96721

noncomputable def smallest_y : ℤ := 10

theorem smallest_pos_int_mult_4410_sq (y : ℕ) (hy : y > 0) :
  (∃ z : ℕ, 4410 * y = z^2) ↔ y = smallest_y :=
sorry

end smallest_pos_int_mult_4410_sq_l967_96721


namespace cycling_journey_l967_96797

theorem cycling_journey :
  ∃ y : ℚ, 0 < y ∧ y <= 12 ∧ (15 * y + 10 * (12 - y) = 150) ∧ y = 6 :=
by
  sorry

end cycling_journey_l967_96797


namespace abs_neg_2023_l967_96755

theorem abs_neg_2023 : abs (-2023) = 2023 := by
  sorry

end abs_neg_2023_l967_96755


namespace ink_percentage_left_l967_96786

def area_of_square (side: ℕ) := side * side
def area_of_rectangle (length: ℕ) (width: ℕ) := length * width
def total_area_marker_can_paint (num_squares: ℕ) (square_side: ℕ) :=
  num_squares * area_of_square square_side
def total_area_colored (num_rectangles: ℕ) (rect_length: ℕ) (rect_width: ℕ) :=
  num_rectangles * area_of_rectangle rect_length rect_width

def fraction_of_ink_used (num_rectangles: ℕ) (rect_length: ℕ) (rect_width: ℕ)
  (num_squares: ℕ) (square_side: ℕ) :=
  (total_area_colored num_rectangles rect_length rect_width : ℚ)
    / (total_area_marker_can_paint num_squares square_side : ℚ)

def percentage_ink_left (num_rectangles: ℕ) (rect_length: ℕ) (rect_width: ℕ)
  (num_squares: ℕ) (square_side: ℕ) :=
  100 * (1 - fraction_of_ink_used num_rectangles rect_length rect_width num_squares square_side)

theorem ink_percentage_left :
  percentage_ink_left 2 6 2 3 4 = 50 := by
  sorry

end ink_percentage_left_l967_96786


namespace slower_pump_time_l967_96702

def pool_problem (R : ℝ) :=
  (∀ t : ℝ, (2.5 * R * t = 1) → (t = 5))
  ∧ (∀ R1 R2 : ℝ, (R1 = 1.5 * R) → (R1 + R = 2.5 * R))
  ∧ (∀ t : ℝ, (R * t = 1) → (t = 12.5))

theorem slower_pump_time (R : ℝ) : pool_problem R :=
by
  -- Assume that the combined rates take 5 hours to fill the pool
  sorry

end slower_pump_time_l967_96702


namespace frankie_pets_total_l967_96790

theorem frankie_pets_total
  (C S P D : ℕ)
  (h_snakes : S = C + 6)
  (h_parrots : P = C - 1)
  (h_dogs : D = 2)
  (h_total : C + S + P + D = 19) :
  C + (C + 6) + (C - 1) + 2 = 19 := by
  sorry

end frankie_pets_total_l967_96790


namespace sequence_sum_l967_96785

theorem sequence_sum (r z w : ℝ) (h1 : 4 * r = 1) (h2 : 256 * r = z) (h3 : z * r = w) : z + w = 80 :=
by
  -- Proceed with your proof here.
  -- sorry for skipping the proof part.
  sorry

end sequence_sum_l967_96785


namespace pieces_per_pizza_is_five_l967_96771

-- Definitions based on the conditions
def cost_per_pizza (total_cost : ℕ) (number_of_pizzas : ℕ) : ℕ :=
  total_cost / number_of_pizzas

def number_of_pieces_per_pizza (cost_per_pizza : ℕ) (cost_per_piece : ℕ) : ℕ :=
  cost_per_pizza / cost_per_piece

-- Given conditions
def total_cost : ℕ := 80
def number_of_pizzas : ℕ := 4
def cost_per_piece : ℕ := 4

-- Prove
theorem pieces_per_pizza_is_five : number_of_pieces_per_pizza (cost_per_pizza total_cost number_of_pizzas) cost_per_piece = 5 :=
by sorry

end pieces_per_pizza_is_five_l967_96771


namespace meaningful_fraction_l967_96764

theorem meaningful_fraction (x : ℝ) : (x ≠ -2) ↔ (∃ y : ℝ, y = 1 / (x + 2)) :=
by sorry

end meaningful_fraction_l967_96764


namespace stones_required_to_pave_hall_l967_96737

theorem stones_required_to_pave_hall :
  ∀ (hall_length_m hall_breadth_m stone_length_dm stone_breadth_dm: ℕ),
  hall_length_m = 72 →
  hall_breadth_m = 30 →
  stone_length_dm = 6 →
  stone_breadth_dm = 8 →
  (hall_length_m * 10 * hall_breadth_m * 10) / (stone_length_dm * stone_breadth_dm) = 4500 := by
  intros _ _ _ _ h_length h_breadth h_slength h_sbreadth
  sorry

end stones_required_to_pave_hall_l967_96737


namespace problem_I_problem_II_l967_96799

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := |x - 2| + |2 * x + a|

-- Problem (I): Inequality solution when a = 1
theorem problem_I (x : ℝ) : f x 1 ≥ 5 ↔ x ∈ (Set.Iic (-4 / 3) ∪ Set.Ici 2) :=
sorry

-- Problem (II): Range of a given the conditions
theorem problem_II (x₀ : ℝ) (a : ℝ) (h : f x₀ a + |x₀ - 2| < 3) : -7 < a ∧ a < -1 :=
sorry

end problem_I_problem_II_l967_96799


namespace probability_entire_grid_black_l967_96717

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

end probability_entire_grid_black_l967_96717


namespace smaller_number_is_22_l967_96709

noncomputable def smaller_number (x y : ℕ) : ℕ := 
x

theorem smaller_number_is_22 (x y : ℕ) (h1 : x + y = 56) (h2 : y = x + 12) : x = 22 :=
by
  sorry

end smaller_number_is_22_l967_96709


namespace a37_b37_sum_l967_96722

-- Declare the sequences as functions from natural numbers to real numbers
variables {a b : ℕ → ℝ}

-- State the hypotheses based on the conditions
variables (h1 : ∀ n, a (n + 1) = a n + a 2 - a 1)
variables (h2 : ∀ n, b (n + 1) = b n + b 2 - b 1)
variables (h3 : a 1 = 25)
variables (h4 : b 1 = 75)
variables (h5 : a 2 + b 2 = 100)

-- State the theorem to be proved
theorem a37_b37_sum : a 37 + b 37 = 100 := 
by 
  sorry

end a37_b37_sum_l967_96722


namespace bill_has_6_less_pieces_than_mary_l967_96776

-- Definitions based on the conditions
def total_candy : ℕ := 20
def candy_kate : ℕ := 4
def candy_robert : ℕ := candy_kate + 2
def candy_mary : ℕ := candy_robert + 2
def candy_bill : ℕ := candy_kate - 2

-- Statement of the theorem
theorem bill_has_6_less_pieces_than_mary :
  candy_mary - candy_bill = 6 :=
sorry

end bill_has_6_less_pieces_than_mary_l967_96776


namespace least_positive_integer_l967_96796

theorem least_positive_integer (n : ℕ) : 
  (n % 4 = 1) ∧ (n % 5 = 2) ∧ (n % 6 = 3) → n = 57 := by
sorry

end least_positive_integer_l967_96796


namespace possible_k_values_l967_96734

def triangle_right_k_values (AB AC : ℝ × ℝ) (k : ℝ) : Prop :=
  let BC := (AC.1 - AB.1, AC.2 - AB.2)
  let angle_A := AB.1 * AC.1 + AB.2 * AC.2 = 0   -- Condition for ∠A = 90°
  let angle_B := AB.1 * BC.1 + AB.2 * BC.2 = 0   -- Condition for ∠B = 90°
  let angle_C := BC.1 * AC.1 + BC.2 * AC.2 = 0   -- Condition for ∠C = 90°
  (angle_A ∨ angle_B ∨ angle_C)

theorem possible_k_values (k : ℝ) :
  triangle_right_k_values (2, 3) (1, k) k ↔
  k = -2/3 ∨ k = 11/3 ∨ k = (3 + Real.sqrt 13) / 2 ∨ k = (3 - Real.sqrt 13) / 2 :=
by
  sorry

end possible_k_values_l967_96734


namespace sprinkler_system_days_l967_96765

theorem sprinkler_system_days 
  (morning_water : ℕ) (evening_water : ℕ) (total_water : ℕ) 
  (h_morning : morning_water = 4) 
  (h_evening : evening_water = 6) 
  (h_total : total_water = 50) :
  total_water / (morning_water + evening_water) = 5 := 
by 
  sorry

end sprinkler_system_days_l967_96765


namespace movie_theater_charge_l967_96752

theorem movie_theater_charge 
    (charge_adult : ℝ) 
    (children : ℕ) 
    (adults : ℕ) 
    (total_receipts : ℝ) 
    (charge_child : ℝ) 
    (condition1 : charge_adult = 6.75) 
    (condition2 : children = adults + 20) 
    (condition3 : total_receipts = 405) 
    (condition4 : children = 48) 
    : charge_child = 4.5 :=
sorry

end movie_theater_charge_l967_96752


namespace interior_angle_regular_octagon_exterior_angle_regular_octagon_l967_96729

-- Definitions
def sumInteriorAngles (n : ℕ) : ℕ := 180 * (n - 2)
def oneInteriorAngle (n : ℕ) (sumInterior : ℕ) : ℕ := sumInterior / n
def sumExteriorAngles : ℕ := 360
def oneExteriorAngle (n : ℕ) (sumExterior : ℕ) : ℕ := sumExterior / n

-- Theorem statements
theorem interior_angle_regular_octagon : oneInteriorAngle 8 (sumInteriorAngles 8) = 135 := by sorry

theorem exterior_angle_regular_octagon : oneExteriorAngle 8 sumExteriorAngles = 45 := by sorry

end interior_angle_regular_octagon_exterior_angle_regular_octagon_l967_96729


namespace one_person_remains_dry_l967_96746

theorem one_person_remains_dry (n : ℕ) :
  ∃ (person_dry : ℕ -> Bool), (∀ i : ℕ, i < 2 * n + 1 -> person_dry i = tt) := 
sorry

end one_person_remains_dry_l967_96746


namespace number_of_solutions_l967_96754

noncomputable def system_of_equations (a b c : ℕ) : Prop :=
  a * b + b * c = 44 ∧ a * c + b * c = 23

theorem number_of_solutions : ∃! (a b c : ℕ), system_of_equations a b c :=
by
  sorry

end number_of_solutions_l967_96754


namespace james_faster_than_john_l967_96761

theorem james_faster_than_john :
  let john_time := 13
  let john_distance := 100
  let john_first_second := 4
  let john_remaining_seconds := john_time - 1
  let john_remaining_distance := john_distance - john_first_second
  let john_top_speed := john_remaining_distance / john_remaining_seconds

  let james_time := 11
  let james_first_two_seconds := 10
  let james_remaining_seconds := james_time - 2
  let james_remaining_distance := john_distance - james_first_two_seconds
  let james_top_speed := james_remaining_distance / james_remaining_seconds
  
  james_top_speed - john_top_speed = 2 :=
by
  let john_time := 13
  let john_distance := 100
  let john_first_second := 4
  let john_remaining_seconds := john_time - 1
  let john_remaining_distance := john_distance - john_first_second
  let john_top_speed := john_remaining_distance / john_remaining_seconds

  let james_time := 11
  let james_first_two_seconds := 10
  let james_remaining_seconds := james_time - 2
  let james_remaining_distance := john_distance - james_first_two_seconds
  let james_top_speed := james_remaining_distance / james_remaining_seconds

  sorry

end james_faster_than_john_l967_96761


namespace seventh_observation_is_4_l967_96795

def avg_six := 11 -- Average of the first six observations
def sum_six := 6 * avg_six -- Total sum of the first six observations
def new_avg := avg_six - 1 -- New average after including the new observation
def new_sum := 7 * new_avg -- Total sum after including the new observation

theorem seventh_observation_is_4 : 
  (new_sum - sum_six) = 4 :=
by
  sorry

end seventh_observation_is_4_l967_96795


namespace plants_given_away_l967_96779

-- Define the conditions as constants
def initial_plants : ℕ := 3
def final_plants : ℕ := 20
def months : ℕ := 3

-- Function to calculate the number of plants after n months
def plants_after_months (initial: ℕ) (months: ℕ) : ℕ := initial * (2 ^ months)

-- The proof problem statement
theorem plants_given_away : (plants_after_months initial_plants months - final_plants) = 4 :=
by
  sorry

end plants_given_away_l967_96779


namespace alcohol_percentage_first_solution_l967_96780

theorem alcohol_percentage_first_solution
  (x : ℝ)
  (h1 : 0 ≤ x ∧ x ≤ 1) -- since percentage in decimal form is between 0 and 1
  (h2 : 75 * x + 0.12 * 125 = 0.15 * 200) :
  x = 0.20 :=
by
  sorry

end alcohol_percentage_first_solution_l967_96780


namespace a2_value_is_42_l967_96791

noncomputable def a₂_value (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 : ℝ) :=
  a_2

theorem a2_value_is_42 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 : ℝ) :
  (x^3 + x^10 = a_0 + a_1 * (x + 1) + a_2 * (x + 1)^2 + a_3 * (x + 1)^3 + a_4 * (x + 1)^4 +
                a_5 * (x + 1)^5 + a_6 * (x + 1)^6 + a_7 * (x + 1)^7 + a_8 * (x + 1)^8 + 
                a_9 * (x + 1)^9 + a_10 * (x + 1)^10) →
  a₂_value a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 = 42 :=
by
  sorry

end a2_value_is_42_l967_96791


namespace discrim_of_quad_l967_96762

-- Definition of the quadratic equation coefficients
def a : ℤ := 5
def b : ℤ := -9
def c : ℤ := 4

-- Definition of the discriminant formula which needs to be proved as 1
def discriminant (a b c : ℤ) : ℤ :=
  b^2 - 4 * a * c

-- The proof problem statement
theorem discrim_of_quad : discriminant a b c = 1 := by
  sorry

end discrim_of_quad_l967_96762


namespace simplify_expression_calculate_difference_of_squares_l967_96711

section Problem1
variable (a b : ℝ)
variable (ha : a ≠ 0) (hb : b ≠ 0)

theorem simplify_expression : ((-2 * a^2) ^ 2 * (-b^2)) / (4 * a^3 * b^2) = -a :=
by sorry
end Problem1

section Problem2

theorem calculate_difference_of_squares : 2023^2 - 2021 * 2025 = 4 :=
by sorry
end Problem2

end simplify_expression_calculate_difference_of_squares_l967_96711


namespace side_length_of_square_IJKL_l967_96706

theorem side_length_of_square_IJKL 
  (x y : ℝ) (hypotenuse : ℝ) 
  (h1 : x - y = 3) 
  (h2 : x + y = 9) 
  (h3 : hypotenuse = Real.sqrt (x^2 + y^2)) : 
  hypotenuse = 3 * Real.sqrt 5 :=
by
  sorry

end side_length_of_square_IJKL_l967_96706


namespace copper_sheet_area_l967_96719

noncomputable def area_of_copper_sheet (l w h : ℝ) (thickness_mm : ℝ) : ℝ :=
  let volume := l * w * h
  let thickness_cm := thickness_mm / 10
  (volume / thickness_cm) / 10000

theorem copper_sheet_area :
  ∀ (l w h thickness_mm : ℝ), 
  l = 80 → w = 20 → h = 5 → thickness_mm = 1 → 
  area_of_copper_sheet l w h thickness_mm = 8 := 
by
  intros l w h thickness_mm hl hw hh hthickness_mm
  rw [hl, hw, hh, hthickness_mm]
  simp [area_of_copper_sheet]
  sorry

end copper_sheet_area_l967_96719


namespace aubree_animals_total_l967_96773

theorem aubree_animals_total (b_go c_go b_return c_return : ℕ) 
    (h1 : b_go = 20) (h2 : c_go = 40) 
    (h3 : b_return = b_go * 2) 
    (h4 : c_return = c_go - 10) : 
    b_go + c_go + b_return + c_return = 130 := by 
  sorry

end aubree_animals_total_l967_96773


namespace inequality_solutions_l967_96768

theorem inequality_solutions (y : ℝ) :
  (2 / (y + 2) + 4 / (y + 8) ≥ 1 ↔ (y > -8 ∧ y ≤ -4) ∨ (y ≥ -2 ∧ y ≤ 2)) :=
by
  sorry

end inequality_solutions_l967_96768


namespace part_a_part_b_l967_96723

namespace TrihedralAngle

-- Part (a)
theorem part_a (α β γ : ℝ) (h1 : β = 70) (h2 : γ = 100) (h3 : α < β + γ) (h4 : β < α + γ) (h5 : γ < α + β) (h6 : α + β + γ < 360) : 
    30 < α ∧ α < 170 := 
sorry

-- Part (b)
theorem part_b (α β γ : ℝ) (h1 : β = 130) (h2 : γ = 150) (h3 : α < β + γ) (h4 : β < α + γ) (h5 : γ < α + β) (h6 : α + β + γ < 360) : 
    20 < α ∧ α < 80 := 
sorry

end TrihedralAngle

end part_a_part_b_l967_96723


namespace consecutive_odd_integers_sum_l967_96725

theorem consecutive_odd_integers_sum (n : ℤ) (h : (n - 2) + (n + 2) = 150) : n = 75 := 
by
  sorry

end consecutive_odd_integers_sum_l967_96725


namespace cannot_use_diff_of_squares_l967_96700

def diff_of_squares (a b : ℤ) : ℤ := a^2 - b^2

theorem cannot_use_diff_of_squares (x y : ℤ) : 
  ¬ ( ((-x + y) * (x - y)) = diff_of_squares (x - y) (0) ) :=
by {
  sorry
}

end cannot_use_diff_of_squares_l967_96700


namespace find_g_of_3_l967_96774

noncomputable def g : ℝ → ℝ := sorry

theorem find_g_of_3 (h : ∀ x : ℝ, g (3 ^ x) + x * g (3 ^ (-x)) = 3) : g 3 = 0 :=
by sorry

end find_g_of_3_l967_96774


namespace Linda_needs_15_hours_to_cover_fees_l967_96710

def wage : ℝ := 10
def fee_per_college : ℝ := 25
def number_of_colleges : ℝ := 6

theorem Linda_needs_15_hours_to_cover_fees :
  (number_of_colleges * fee_per_college) / wage = 15 := by
  sorry

end Linda_needs_15_hours_to_cover_fees_l967_96710


namespace jeff_makes_donuts_for_days_l967_96704

variable (d : ℕ) (boxes donuts_per_box : ℕ) (donuts_per_day eaten_per_day : ℕ) (chris_eaten total_donuts : ℕ)

theorem jeff_makes_donuts_for_days :
  (donuts_per_day = 10) →
  (eaten_per_day = 1) →
  (chris_eaten = 8) →
  (boxes = 10) →
  (donuts_per_box = 10) →
  (total_donuts = boxes * donuts_per_box) →
  (9 * d - chris_eaten = total_donuts) →
  d = 12 :=
  by
    intros h1 h2 h3 h4 h5 h6 h7
    sorry

end jeff_makes_donuts_for_days_l967_96704


namespace highest_throw_christine_janice_l967_96714

theorem highest_throw_christine_janice
  (c1 : ℕ) -- Christine's first throw
  (j1 : ℕ) -- Janice's first throw
  (c2 : ℕ) -- Christine's second throw
  (j2 : ℕ) -- Janice's second throw
  (c3 : ℕ) -- Christine's third throw
  (j3 : ℕ) -- Janice's third throw
  (h1 : c1 = 20)
  (h2 : j1 = c1 - 4)
  (h3 : c2 = c1 + 10)
  (h4 : j2 = j1 * 2)
  (h5 : c3 = c2 + 4)
  (h6 : j3 = c1 + 17) :
  max c1 (max c2 (max c3 (max j1 (max j2 j3)))) = 37 := by
  sorry

end highest_throw_christine_janice_l967_96714


namespace probability_not_cash_l967_96766

theorem probability_not_cash (h₁ : 0.45 + 0.15 + pnc = 1) : pnc = 0.4 :=
by
  sorry

end probability_not_cash_l967_96766


namespace total_number_of_workers_l967_96742

theorem total_number_of_workers 
    (W N : ℕ) 
    (h1 : 8000 * W = 12000 * 8 + 6000 * N) 
    (h2 : W = 8 + N) : 
    W = 24 :=
by
  sorry

end total_number_of_workers_l967_96742


namespace fruit_count_l967_96775

theorem fruit_count :
  let limes_mike : ℝ := 32.5
  let limes_alyssa : ℝ := 8.25
  let limes_jenny_picked : ℝ := 10.8
  let limes_jenny_ate := limes_jenny_picked / 2
  let limes_jenny := limes_jenny_picked - limes_jenny_ate
  let plums_tom : ℝ := 14.5
  let plums_tom_ate : ℝ := 2.5
  let X := (limes_mike - limes_alyssa) + limes_jenny
  let Y := plums_tom - plums_tom_ate
  X = 29.65 ∧ Y = 12 :=
by {
  sorry
}

end fruit_count_l967_96775


namespace problem_solution_l967_96739

variables (x y : ℝ)

def cond1 : Prop := 4 * x + y = 12
def cond2 : Prop := x + 4 * y = 18

theorem problem_solution (h1 : cond1 x y) (h2 : cond2 x y) : 20 * x^2 + 24 * x * y + 20 * y^2 = 468 :=
by
  -- Proof would go here
  sorry

end problem_solution_l967_96739


namespace symmetrical_circle_equation_l967_96798

theorem symmetrical_circle_equation :
  ∀ (x y : ℝ), (x^2 + y^2 - 2 * x - 1 = 0) ∧ (2 * x - y + 1 = 0) →
  ((x + 7/5)^2 + (y - 6/5)^2 = 2) :=
sorry

end symmetrical_circle_equation_l967_96798


namespace new_person_weight_l967_96781

theorem new_person_weight (avg_increase : Real) (n : Nat) (old_weight : Real) (W_new : Real) :
  avg_increase = 2.5 → n = 8 → old_weight = 67 → W_new = old_weight + n * avg_increase → W_new = 87 :=
by
  intros avg_increase_eq n_eq old_weight_eq calc_eq
  sorry

end new_person_weight_l967_96781


namespace eddy_time_to_B_l967_96763

-- Definitions
def distance_A_to_B : ℝ := 570
def distance_A_to_C : ℝ := 300
def time_C : ℝ := 4
def speed_ratio : ℝ := 2.5333333333333333

-- Theorem Statement
theorem eddy_time_to_B : 
  (distance_A_to_B / (distance_A_to_C / time_C * speed_ratio)) = 3 := 
by
  sorry

end eddy_time_to_B_l967_96763


namespace scientists_speculation_reasonable_uranus_will_not_affect_earth_next_observation_l967_96787

-- Define the given conditions as constants and theorems in Lean
theorem scientists_speculation_reasonable : 
  ∃ (a b c : ℝ), 
  (64*a - 8*b + c = 32) ∧ 
  (36*a - 6*b + c = 28.5) ∧ 
  (16*a - 4*b + c = 26) ∧ 
  (∀ (x y : ℝ), (y = a*x^2 + b*x + c) → (x = 0) → y < 24.5) :=
by -- sorry is a placeholder for the proof
sorry

theorem uranus_will_not_affect_earth_next_observation : 
  ∃ (a b c : ℝ), 
  (64*a - 8*b + c = 32) ∧ 
  (36*a - 6*b + c = 28.5) ∧ 
  (16*a - 4*b + c = 26) ∧ 
  (∀ (x y : ℝ), (y = a*x^2 + b*x + c) → (x = 2) → y ≥ 24.5) :=
by -- sorry is a placeholder for the proof
sorry

end scientists_speculation_reasonable_uranus_will_not_affect_earth_next_observation_l967_96787


namespace dot_product_result_l967_96744

def a : ℝ × ℝ := (-1, 2)
def b (m : ℝ) : ℝ × ℝ := (2, m)
def c : ℝ × ℝ := (7, 1)

def are_parallel (a b : ℝ × ℝ) : Prop := 
  a.1 * b.2 = a.2 * b.1

def dot_product (u v : ℝ × ℝ) : ℝ := 
  u.1 * v.1 + u.2 * v.2

theorem dot_product_result : 
  ∀ m : ℝ, are_parallel a (b m) → dot_product (b m) c = 10 := 
by
  sorry

end dot_product_result_l967_96744


namespace total_money_from_selling_watermelons_l967_96788

-- Given conditions
def weight_of_one_watermelon : ℝ := 23
def price_per_pound : ℝ := 2
def number_of_watermelons : ℝ := 18

-- Statement to be proved
theorem total_money_from_selling_watermelons : 
  (weight_of_one_watermelon * price_per_pound) * number_of_watermelons = 828 := 
by 
  sorry

end total_money_from_selling_watermelons_l967_96788


namespace transformation_l967_96770

noncomputable def Q (a b c x y : ℝ) : ℝ := a * x^2 + 2 * b * x * y + c * y^2

theorem transformation 
  (a b c d e f x y x₀ y₀ x' y' : ℝ)
  (h : a * c - b^2 ≠ 0)
  (hQ : Q a b c x y + 2 * d * x + 2 * e * y = f)
  (hx : x' = x + x₀)
  (hy : y' = y + y₀) :
  ∃ f' : ℝ, (a * x'^2 + 2 * b * x' * y' + c * y'^2 = f' ∧ 
             f' = f - Q a b c x₀ y₀ + 2 * (d * x₀ + e * y₀)) :=
sorry

end transformation_l967_96770


namespace time_spent_cleaning_bathroom_l967_96716

-- Define the times spent on each task
def laundry_time : ℕ := 30
def room_cleaning_time : ℕ := 35
def homework_time : ℕ := 40
def total_time : ℕ := 120

-- Let b be the time spent cleaning the bathroom
variable (b : ℕ)

-- Total time spent on all tasks is the sum of individual times
def total_task_time := laundry_time + b + room_cleaning_time + homework_time

-- Proof that b = 15 given the total time
theorem time_spent_cleaning_bathroom (h : total_task_time = total_time) : b = 15 :=
by
  sorry

end time_spent_cleaning_bathroom_l967_96716


namespace find_number_l967_96705

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def XiaoQian_statements (n : ℕ) : Prop :=
  is_perfect_square n ∧ n < 5

def XiaoLu_statements (n : ℕ) : Prop :=
  n < 7 ∧ 10 ≤ n ∧ n < 100

def XiaoDai_statements (n : ℕ) : Prop :=
  is_perfect_square n ∧ ¬ (n < 5)

theorem find_number :
  ∃! n : ℕ, 1 ≤ n ∧ n ≤ 99 ∧ 
    ( (XiaoQian_statements n ∧ ¬XiaoLu_statements n ∧ ¬XiaoDai_statements n) ∨
      (¬XiaoQian_statements n ∧ XiaoLu_statements n ∧ ¬XiaoDai_statements n) ∨
      (¬XiaoQian_statements n ∧ ¬XiaoLu_statements n ∧ XiaoDai_statements n) ) ∧
    n = 9 :=
sorry

end find_number_l967_96705


namespace average_of_w_and_x_is_one_half_l967_96759

noncomputable def average_of_w_and_x (w x y : ℝ) : ℝ :=
  (w + x) / 2

theorem average_of_w_and_x_is_one_half (w x y : ℝ)
  (h1 : 2 / w + 2 / x = 2 / y)
  (h2 : w * x = y) : average_of_w_and_x w x y = 1 / 2 :=
by
  sorry

end average_of_w_and_x_is_one_half_l967_96759


namespace max_value_2cosx_3sinx_l967_96789

open Real 

theorem max_value_2cosx_3sinx : ∀ x : ℝ, 2 * cos x + 3 * sin x ≤ sqrt 13 :=
by sorry

end max_value_2cosx_3sinx_l967_96789


namespace total_balls_l967_96735

theorem total_balls (colors : ℕ) (balls_per_color : ℕ) (h_colors : colors = 10) (h_balls_per_color : balls_per_color = 35) : 
    colors * balls_per_color = 350 :=
by
  -- Import necessary libraries
  sorry

end total_balls_l967_96735


namespace find_triples_l967_96784

theorem find_triples (a m n : ℕ) (k : ℕ):
  a ≥ 2 ∧ m ≥ 2 ∧ a^n + 203 ≡ 0 [MOD a^m + 1] ↔ 
  (a = 2 ∧ ((n = 4 * k + 1 ∧ m = 2) ∨ (n = 6 * k + 2 ∧ m = 3) ∨ (n = 8 * k + 8 ∧ m = 4) ∨ (n = 12 * k + 9 ∧ m = 6))) ∨
  (a = 3 ∧ n = 4 * k + 3 ∧ m = 2) ∨
  (a = 4 ∧ n = 4 * k + 4 ∧ m = 2) ∨
  (a = 5 ∧ n = 4 * k + 1 ∧ m = 2) ∨
  (a = 8 ∧ n = 4 * k + 3 ∧ m = 2) ∨
  (a = 10 ∧ n = 4 * k + 2 ∧ m = 2) ∨
  (a = 203 ∧ n = (2 * k + 1) * m + 1 ∧ m ≥ 2) := by sorry

end find_triples_l967_96784


namespace avg_reading_time_l967_96748

theorem avg_reading_time (emery_book_time serena_book_time emery_article_time serena_article_time : ℕ)
    (h1 : emery_book_time = 20)
    (h2 : emery_article_time = 2)
    (h3 : emery_book_time * 5 = serena_book_time)
    (h4 : emery_article_time * 3 = serena_article_time) :
    (emery_book_time + emery_article_time + serena_book_time + serena_article_time) / 2 = 64 := by
  sorry

end avg_reading_time_l967_96748


namespace range_of_x_inequality_l967_96724

theorem range_of_x_inequality (x : ℝ) (h : |2 * x - 1| + x + 3 ≤ 5) : -1 ≤ x ∧ x ≤ 1 :=
by
  sorry

end range_of_x_inequality_l967_96724


namespace solution_x_y_l967_96749

noncomputable def eq_values (x y : ℝ) := (
  x ≠ 0 ∧ x ≠ 1 ∧ y ≠ 0 ∧ y ≠ 3 ∧ (3/x + 2/y = 1/3)
)

theorem solution_x_y (x y : ℝ) (h : eq_values x y) : x = 9 * y / (y - 6) :=
sorry

end solution_x_y_l967_96749


namespace rectangle_area_in_ellipse_l967_96753

theorem rectangle_area_in_ellipse :
  ∃ a b : ℝ, 2 * a = b ∧ (a^2 / 4 + b^2 / 8 = 1) ∧ 2 * a * b = 16 :=
by
  sorry

end rectangle_area_in_ellipse_l967_96753


namespace inequality_solution_l967_96741

theorem inequality_solution (x : ℝ) :
  (x - 3) / (x^2 + 4 * x + 13) ≥ 0 ↔ x ∈ Set.Ici 3 :=
by
  sorry

end inequality_solution_l967_96741


namespace prob_a_prob_b_l967_96793

def A (a : ℝ) := {x : ℝ | 0 < x + a ∧ x + a ≤ 5}
def B := {x : ℝ | -1/2 ≤ x ∧ x < 6}

theorem prob_a (a : ℝ) : (A a ⊆ B) → (-1 < a ∧ a ≤ 1/2) :=
sorry

theorem prob_b (a : ℝ) : (∃ x, A a ∩ B = {x}) → a = 11/2 :=
sorry

end prob_a_prob_b_l967_96793


namespace esther_commute_distance_l967_96751

theorem esther_commute_distance (D : ℕ) :
  (D / 45 + D / 30 = 1) → D = 18 :=
by
  sorry

end esther_commute_distance_l967_96751


namespace max_value_of_f_l967_96718

-- Define the quadratic function
def f (x : ℝ) : ℝ := 9 * x - 4 * x^2

-- Define a proof problem to show that the maximum value of f(x) is 81/16
theorem max_value_of_f : ∃ x : ℝ, f x = 81 / 16 :=
by
  -- The vertex of the quadratic function gives the maximum value since the parabola opens downward
  let x := 9 / (2 * 4)
  use x
  -- sorry to skip the proof steps
  sorry

end max_value_of_f_l967_96718


namespace meet_at_35_l967_96703

def walking_distance_A (t : ℕ) := 5 * t

def walking_distance_B (t : ℕ) := (t * (7 + t)) / 2

def total_distance (t : ℕ) := walking_distance_A t + walking_distance_B t

theorem meet_at_35 : ∃ (t : ℕ), total_distance t = 100 ∧ walking_distance_A t - walking_distance_B t = 35 := by
  sorry

end meet_at_35_l967_96703


namespace four_leaf_area_l967_96713

theorem four_leaf_area (a : ℝ) : 
  let radius := a / 2
  let semicircle_area := (π * radius ^ 2) / 2
  let triangle_area := (a / 2) * (a / 2) / 2
  let half_leaf_area := semicircle_area - triangle_area
  let leaf_area := 2 * half_leaf_area
  let total_area := 4 * leaf_area
  total_area = a ^ 2 / 2 * (π - 2) := 
by
  sorry

end four_leaf_area_l967_96713


namespace bridge_length_calculation_l967_96758

def length_of_bridge (train_length : ℕ) (train_speed_kmph : ℕ) (time_seconds : ℕ) : ℕ :=
  let speed_mps := (train_speed_kmph * 1000) / 3600
  let distance_covered := speed_mps * time_seconds
  distance_covered - train_length

theorem bridge_length_calculation :
  length_of_bridge 140 45 30 = 235 :=
by
  unfold length_of_bridge
  norm_num
  sorry

end bridge_length_calculation_l967_96758


namespace calculate_expression_l967_96733

theorem calculate_expression : 200 * 39.96 * 3.996 * 500 = (3996)^2 :=
by
  sorry

end calculate_expression_l967_96733


namespace sum_of_powers_seven_l967_96794

theorem sum_of_powers_seven (α1 α2 α3 : ℂ)
  (h1 : α1 + α2 + α3 = 2)
  (h2 : α1^2 + α2^2 + α3^2 = 6)
  (h3 : α1^3 + α2^3 + α3^3 = 14) :
  α1^7 + α2^7 + α3^7 = 478 := by
  sorry

end sum_of_powers_seven_l967_96794


namespace area_enclosed_by_curves_l967_96740

noncomputable def areaBetweenCurves : ℝ :=
  ∫ x in (0 : ℝ)..(4 : ℝ), (x - (x^2 - 3*x))

theorem area_enclosed_by_curves :
  (∫ x in (0 : ℝ)..(4 : ℝ), (x - (x^2 - 3*x))) = (32 / 3 : ℝ) :=
by
  sorry

end area_enclosed_by_curves_l967_96740


namespace complement_set_U_A_l967_96727

-- Definitions of U and A
def U : Set ℝ := { x : ℝ | x^2 ≤ 4 }
def A : Set ℝ := { x : ℝ | |x - 1| ≤ 1 }

-- Theorem statement
theorem complement_set_U_A : (U \ A) = { x : ℝ | -2 ≤ x ∧ x < 0 } := 
by
  sorry

end complement_set_U_A_l967_96727


namespace find_k_l967_96738

theorem find_k (k : ℝ) (d : ℝ) (h : d = 4) :
  -x^2 - (k + 10) * x - 8 = -(x - 2) * (x - d) → k = -16 :=
by
  intros
  rw [h] at *
  sorry

end find_k_l967_96738


namespace rhombus_perimeter_and_radius_l967_96715

-- Define the rhombus with given diagonals
structure Rhombus where
  d1 : ℝ -- diagonal 1
  d2 : ℝ -- diagonal 2
  h : d1 = 20 ∧ d2 = 16

-- Define the proof problem
theorem rhombus_perimeter_and_radius (r : Rhombus) : 
  let side_length := Real.sqrt ((r.d1 / 2) ^ 2 + (r.d2 / 2) ^ 2)
  let perimeter := 4 * side_length
  let radius := r.d1 / 2
  perimeter = 16 * Real.sqrt 41 ∧ radius = 10 :=
by
  sorry

end rhombus_perimeter_and_radius_l967_96715


namespace inequalities_sufficient_but_not_necessary_l967_96778

theorem inequalities_sufficient_but_not_necessary (a b c d : ℝ) :
  (a > b ∧ c > d) → (a + c > b + d) ∧ ¬((a + c > b + d) → (a > b ∧ c > d)) :=
by
  sorry

end inequalities_sufficient_but_not_necessary_l967_96778


namespace pipes_fill_tank_l967_96736

theorem pipes_fill_tank (T : ℝ) (h1 : T > 0)
  (h2 : (1/4 : ℝ) + 1/T - 1/20 = 1/2.5) : T = 5 := by
  sorry

end pipes_fill_tank_l967_96736


namespace rational_powers_imply_integers_l967_96792

theorem rational_powers_imply_integers (a b : ℚ) (h_distinct : a ≠ b)
  (h_infinitely_many_n : ∃ᶠ (n : ℕ) in Filter.atTop, (n * (a^n - b^n) : ℚ).den = 1) :
  ∃ (a_int b_int : ℤ), a = a_int ∧ b = b_int := 
sorry

end rational_powers_imply_integers_l967_96792


namespace ferry_time_difference_l967_96701

-- Definitions for the given conditions
def speed_p := 8
def time_p := 3
def distance_p := speed_p * time_p
def distance_q := 3 * distance_p
def speed_q := speed_p + 1
def time_q := distance_q / speed_q

-- Theorem to be proven
theorem ferry_time_difference : (time_q - time_p) = 5 := 
by
  let speed_p := 8
  let time_p := 3
  let distance_p := speed_p * time_p
  let distance_q := 3 * distance_p
  let speed_q := speed_p + 1
  let time_q := distance_q / speed_q
  sorry

end ferry_time_difference_l967_96701


namespace percentage_deficit_for_second_side_l967_96708

-- Defining the given conditions and the problem statement
def side1_excess : ℚ := 0.14
def area_error : ℚ := 0.083
def original_length (L : ℚ) := L
def original_width (W : ℚ) := W
def measured_length_side1 (L : ℚ) := (1 + side1_excess) * L
def measured_width_side2 (W : ℚ) (x : ℚ) := W * (1 - 0.01 * x)
def original_area (L W : ℚ) := L * W
def calculated_area (L W x : ℚ) := 
  measured_length_side1 L * measured_width_side2 W x

theorem percentage_deficit_for_second_side (L W : ℚ) :
  (calculated_area L W 5) / (original_area L W) = 1 + area_error :=
by
  sorry

end percentage_deficit_for_second_side_l967_96708


namespace total_marbles_l967_96750

variable (r b g : ℝ)
variable (h1 : r = 1.3 * b)
variable (h2 : g = 1.7 * r)

theorem total_marbles (r b g : ℝ) (h1 : r = 1.3 * b) (h2 : g = 1.7 * r) :
  r + b + g = 3.469 * r :=
by
  sorry

end total_marbles_l967_96750


namespace inverse_variation_with_constant_l967_96745

theorem inverse_variation_with_constant
  (k : ℝ)
  (x y : ℝ)
  (h1 : y = (3 * k) / x)
  (h2 : x = 4)
  (h3 : y = 8) :
  (y = (3 * (32 / 3)) / -16) := by
sorry

end inverse_variation_with_constant_l967_96745


namespace seating_arrangement_correct_l967_96707

-- Define the number of seating arrangements based on the given conditions

def seatingArrangements : Nat := 
  2 * 4 * 6

theorem seating_arrangement_correct :
  seatingArrangements = 48 := by
  sorry

end seating_arrangement_correct_l967_96707


namespace isosceles_triangle_perimeter_l967_96743

theorem isosceles_triangle_perimeter (a b : ℕ) (h₁ : a = 5) (h₂ : b = 10) :
  (a + b + b = 25) ∧ (a + a + b ≤ b → False) :=
by
  sorry

end isosceles_triangle_perimeter_l967_96743


namespace sum_of_exponents_l967_96782

theorem sum_of_exponents : 
  (-1)^(2010) + (-1)^(2013) + 1^(2014) + (-1)^(2016) = 0 := 
by
  sorry

end sum_of_exponents_l967_96782


namespace remainder_when_divided_by_11_l967_96756

theorem remainder_when_divided_by_11 :
  (7 * 10^20 + 2^20) % 11 = 8 := by
sorry

end remainder_when_divided_by_11_l967_96756


namespace four_thirds_of_twelve_fifths_l967_96783

theorem four_thirds_of_twelve_fifths : (4 / 3) * (12 / 5) = 16 / 5 := 
by sorry

end four_thirds_of_twelve_fifths_l967_96783


namespace supplement_of_complement_65_degrees_l967_96777

def complement (α : ℝ) : ℝ := 90 - α
def supplement (α : ℝ) : ℝ := 180 - α

theorem supplement_of_complement_65_degrees : 
  supplement (complement 65) = 155 :=
by
  sorry

end supplement_of_complement_65_degrees_l967_96777


namespace range_of_m_l967_96720

noncomputable def set_A (x : ℝ) : ℝ := x^2 - (3 / 2) * x + 1

def A : Set ℝ := {y | ∃ (x : ℝ), x ∈ (Set.Icc (-1/2 : ℝ) 2) ∧ y = set_A x}
def B (m : ℝ) : Set ℝ := {x : ℝ | x ≥ m + 1 ∨ x ≤ m - 1}

def sufficient_condition (m : ℝ) : Prop := A ⊆ B m

theorem range_of_m :
  {m : ℝ | sufficient_condition m} = {m | m ≤ -(9 / 16) ∨ m ≥ 3} :=
sorry

end range_of_m_l967_96720


namespace problem_l967_96730

variable (x : ℝ) (Q : ℝ)

theorem problem (h : 2 * (5 * x + 3 * Real.pi) = Q) : 4 * (10 * x + 6 * Real.pi + 2) = 4 * Q + 8 :=
by
  sorry

end problem_l967_96730


namespace find_real_numbers_l967_96731

theorem find_real_numbers (x1 x2 x3 x4 : ℝ) :
  x1 + x2 * x3 * x4 = 2 →
  x2 + x1 * x3 * x4 = 2 →
  x3 + x1 * x2 * x4 = 2 →
  x4 + x1 * x2 * x3 = 2 →
  (x1 = 1 ∧ x2 = 1 ∧ x3 = 1 ∧ x4 = 1) ∨ 
  (x1 = -1 ∧ x2 = -1 ∧ x3 = -1 ∧ x4 = 3) ∨
  (x1 = -1 ∧ x2 = -1 ∧ x3 = 3 ∧ x4 = -1) ∨
  (x1 = -1 ∧ x2 = 3 ∧ x3 = -1 ∧ x4 = -1) ∨
  (x1 = 3 ∧ x2 = -1 ∧ x3 = -1 ∧ x4 = -1) :=
by sorry

end find_real_numbers_l967_96731


namespace blocks_left_l967_96769

theorem blocks_left (initial_blocks used_blocks : ℕ) (h_initial : initial_blocks = 59) (h_used : used_blocks = 36) : initial_blocks - used_blocks = 23 :=
by
  -- proof here
  sorry

end blocks_left_l967_96769


namespace half_angle_in_first_or_third_quadrant_l967_96726

noncomputable 
def angle_in_first_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, 2 * k * Real.pi < α ∧ α < (2 * k + 1) * Real.pi / 2

noncomputable 
def angle_in_first_or_third_quadrant (β : ℝ) : Prop :=
  ∃ k : ℤ, k * Real.pi < β ∧ β < (k + 1/4) * Real.pi ∨
  ∃ i : ℤ, (2 * i + 1) * Real.pi < β ∧ β < (2 * i + 5/4) * Real.pi 

theorem half_angle_in_first_or_third_quadrant (α : ℝ) (h : angle_in_first_quadrant α) :
  angle_in_first_or_third_quadrant (α / 2) :=
  sorry

end half_angle_in_first_or_third_quadrant_l967_96726


namespace mean_study_hours_l967_96747

theorem mean_study_hours :
  let students := [3, 6, 8, 5, 4, 2, 2]
  let hours := [0, 2, 4, 6, 8, 10, 12]
  (0 * 3 + 2 * 6 + 4 * 8 + 6 * 5 + 8 * 4 + 10 * 2 + 12 * 2) / (3 + 6 + 8 + 5 + 4 + 2 + 2) = 5 :=
by
  sorry

end mean_study_hours_l967_96747


namespace John_cycles_distance_l967_96757

-- Define the rate and time as per the conditions in the problem
def rate : ℝ := 8 -- miles per hour
def time : ℝ := 2.25 -- hours

-- The mathematical statement to prove: distance = rate * time
theorem John_cycles_distance : rate * time = 18 := by
  sorry

end John_cycles_distance_l967_96757


namespace original_number_of_cards_l967_96728

-- Declare variables r and b as naturals representing the number of red and black cards, respectively.
variable (r b : ℕ)

-- Assume the probabilities given in the problem.
axiom prob_red : (r : ℝ) / (r + b) = 1 / 3
axiom prob_red_after_add : (r : ℝ) / (r + b + 4) = 1 / 4

-- Define the statement we need to prove.
theorem original_number_of_cards : r + b = 12 :=
by
  -- The proof steps would be here, but we'll use sorry to avoid implementing them.
  sorry

end original_number_of_cards_l967_96728
