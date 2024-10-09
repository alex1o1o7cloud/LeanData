import Mathlib

namespace solution_set_eq_l1586_158676

theorem solution_set_eq : { x : ℝ | |x| * (x - 2) ≥ 0 } = { x : ℝ | x ≥ 2 ∨ x = 0 } := by
  sorry

end solution_set_eq_l1586_158676


namespace fraction_division_l1586_158692

variable {x : ℝ}
variable (hx : x ≠ 0)

theorem fraction_division (hx : x ≠ 0) : (3 / 8) / (5 * x / 12) = 9 / (10 * x) := 
by
  sorry

end fraction_division_l1586_158692


namespace find_S6_l1586_158660

variable (a : ℕ → ℝ) (S_n : ℕ → ℝ)

-- The sequence {a_n} is given as a geometric sequence
-- Partial sums are given as S_2 = 1 and S_4 = 3

-- Conditions
axiom geom_sequence : ∀ n : ℕ, a (n + 1) / a n = a 1 / a 0
axiom S2 : S_n 2 = 1
axiom S4 : S_n 4 = 3

-- Theorem statement
theorem find_S6 : S_n 6 = 7 :=
sorry

end find_S6_l1586_158660


namespace shortest_distance_to_circle_l1586_158646

def center : ℝ × ℝ := (8, 7)
def radius : ℝ := 5
def point : ℝ × ℝ := (1, -2)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  (Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2))

theorem shortest_distance_to_circle :
  distance point center - radius = Real.sqrt 130 - 5 :=
by
  sorry

end shortest_distance_to_circle_l1586_158646


namespace total_interest_correct_l1586_158698

-- Initial conditions
def initial_investment : ℝ := 1200
def annual_interest_rate : ℝ := 0.08
def additional_deposit : ℝ := 500
def first_period : ℕ := 2
def second_period : ℕ := 2

-- Calculate the accumulated value after the first period
def first_accumulated_value : ℝ := initial_investment * (1 + annual_interest_rate)^first_period

-- Calculate the new principal after additional deposit
def new_principal := first_accumulated_value + additional_deposit

-- Calculate the accumulated value after the second period
def final_value := new_principal * (1 + annual_interest_rate)^second_period

-- Calculate the total interest earned after 4 years
def total_interest_earned := final_value - initial_investment - additional_deposit

-- Final theorem statement to be proven
theorem total_interest_correct : total_interest_earned = 515.26 :=
by sorry

end total_interest_correct_l1586_158698


namespace lengthDE_is_correct_l1586_158626

noncomputable def triangleBase : ℝ := 12

noncomputable def triangleArea (h : ℝ) : ℝ := (1 / 2) * triangleBase * h

noncomputable def projectedArea (h : ℝ) : ℝ := 0.16 * triangleArea h

noncomputable def lengthDE (h : ℝ) : ℝ := 0.4 * triangleBase

theorem lengthDE_is_correct (h : ℝ) :
  lengthDE h = 4.8 :=
by
  simp [lengthDE, triangleBase, triangleArea, projectedArea]
  sorry

end lengthDE_is_correct_l1586_158626


namespace segment_length_in_meters_l1586_158625

-- Conditions
def inch_to_meters : ℝ := 500
def segment_length_in_inches : ℝ := 7.25

-- Theorem to prove
theorem segment_length_in_meters : segment_length_in_inches * inch_to_meters = 3625 := by
  sorry

end segment_length_in_meters_l1586_158625


namespace find_pairs_l1586_158671

theorem find_pairs (m n : ℕ) (h1 : m > 0) (h2 : n > 0)
  (cond1 : (m^2 - n) ∣ (m + n^2))
  (cond2 : (n^2 - m) ∣ (n + m^2)) :
  (m = 2 ∧ n = 1) ∨ (m = 3 ∧ n = 2) := 
sorry

end find_pairs_l1586_158671


namespace not_suitable_for_storing_l1586_158650

-- Define the acceptable temperature range conditions for storing dumplings
def acceptable_range (t : ℤ) : Prop :=
  -20 ≤ t ∧ t ≤ -16

-- Define the specific temperatures under consideration
def temp_A : ℤ := -17
def temp_B : ℤ := -18
def temp_C : ℤ := -19
def temp_D : ℤ := -22

-- Define a theorem stating that temp_D is not in the acceptable range
theorem not_suitable_for_storing (t : ℤ) (h : t = temp_D) : ¬ acceptable_range t :=
by {
  sorry
}

end not_suitable_for_storing_l1586_158650


namespace distances_perimeter_inequality_l1586_158689

variable {Point Polygon : Type}

-- Definitions for the conditions
variables (O : Point) (M : Polygon)
variable (ρ : ℝ) -- perimeter of M
variable (d : ℝ) -- sum of distances to each vertex of M from O
variable (h : ℝ) -- sum of distances to each side of M from O

-- The theorem statement
theorem distances_perimeter_inequality :
  d^2 - h^2 ≥ ρ^2 / 4 :=
by
  sorry

end distances_perimeter_inequality_l1586_158689


namespace gcd_2024_2048_l1586_158612

theorem gcd_2024_2048 : Nat.gcd 2024 2048 = 8 := by
  sorry

end gcd_2024_2048_l1586_158612


namespace pq_problem_l1586_158655

theorem pq_problem
  (p q : ℝ)
  (h1 : ∀ x : ℝ, (x - 7) * (2 * x + 11) = x^2 - 19 * x +  60)
  (h2 : p * q = 7 * (-9))
  (h3 : 7 + (-9) = -16):
  (p - 2) * (q - 2) = -55 :=
by
  sorry

end pq_problem_l1586_158655


namespace clock_hands_angle_120_l1586_158667

-- We are only defining the problem statement and conditions. No need for proof steps or calculations.

def angle_between_clock_hands (hour minute : ℚ) : ℚ :=
  abs ((30 * hour + minute / 2) - (6 * minute))

-- Given conditions
def time_in_range (hour : ℚ) (minute : ℚ) := 7 ≤ hour ∧ hour < 8

-- Problem statement to be proved
theorem clock_hands_angle_120 (hour minute : ℚ) :
  time_in_range hour minute → angle_between_clock_hands hour minute = 120 :=
sorry

end clock_hands_angle_120_l1586_158667


namespace germination_percentage_in_second_plot_l1586_158643

theorem germination_percentage_in_second_plot
     (seeds_first_plot : ℕ := 300)
     (seeds_second_plot : ℕ := 200)
     (germination_first_plot : ℕ := 75)
     (total_seeds : ℕ := 500)
     (germination_total : ℕ := 155)
     (x : ℕ := 40) :
  (x : ℕ) = (80 / 2) := by
  -- Provided conditions, skipping the proof part with sorry
  have h1 : 75 = 0.25 * 300 := sorry
  have h2 : 500 = 300 + 200 := sorry
  have h3 : 155 = 0.31 * 500 := sorry
  have h4 : 80 = 155 - 75 := sorry
  have h5 : x = (80 / 2) := sorry
  exact h5

end germination_percentage_in_second_plot_l1586_158643


namespace min_value_a_plus_b_l1586_158633

theorem min_value_a_plus_b (a b : ℕ) (h₁ : 79 ∣ (a + 77 * b)) (h₂ : 77 ∣ (a + 79 * b)) : a + b = 193 :=
by
  sorry

end min_value_a_plus_b_l1586_158633


namespace ellipse_standard_form_l1586_158636

theorem ellipse_standard_form (α : ℝ) 
  (x y : ℝ) 
  (hx : x = 5 * Real.cos α) 
  (hy : y = 3 * Real.sin α) : 
  (x^2 / 25) + (y^2 / 9) = 1 := 
by 
  sorry

end ellipse_standard_form_l1586_158636


namespace units_digit_of_expression_l1586_158639

def units_digit (n : ℕ) : ℕ := n % 10

noncomputable def expression := (20 * 21 * 22 * 23 * 24 * 25) / 1000

theorem units_digit_of_expression : units_digit (expression) = 2 :=
by
  sorry

end units_digit_of_expression_l1586_158639


namespace geometric_sequence_sum_l1586_158632

theorem geometric_sequence_sum (n : ℕ) (S : ℕ → ℚ) (a : ℚ) :
  (∀ n, S n = (1 / 2) * 3^(n + 1) - a) →
  S 1 - (S 2 - S 1)^2 = (S 2 - S 1) * (S 3 - S 2) →
  a = 3 / 2 :=
by
  intros hSn hgeo
  sorry

end geometric_sequence_sum_l1586_158632


namespace curve_is_circle_l1586_158688

theorem curve_is_circle : ∀ (θ : ℝ), ∃ r : ℝ, r = 3 * Real.cos θ → ∃ (x y : ℝ), x^2 + y^2 = (3/2)^2 :=
by
  intro θ
  use 3 * Real.cos θ
  sorry

end curve_is_circle_l1586_158688


namespace number_of_factors_in_224_l1586_158653

def smallest_is_half_largest (n1 n2 : ℕ) : Prop :=
  n1 * 2 = n2

theorem number_of_factors_in_224 :
  ∃ n1 n2 n3 : ℕ, n1 * n2 * n3 = 224 ∧ smallest_is_half_largest (min n1 (min n2 n3)) (max n1 (max n2 n3)) ∧
    (if h : n1 < n2 ∧ n1 < n3 then
      if h2 : n2 < n3 then 
        smallest_is_half_largest n1 n3 
        else 
        smallest_is_half_largest n1 n2 
    else if h : n2 < n1 ∧ n2 < n3 then 
      if h2 : n1 < n3 then 
        smallest_is_half_largest n2 n3 
        else 
        smallest_is_half_largest n2 n1 
    else 
      if h2 : n1 < n2 then 
        smallest_is_half_largest n3 n2 
        else 
        smallest_is_half_largest n3 n1) = true ∧ 
    (if h : n1 < n2 ∧ n1 < n3 then
       if h2 : n2 < n3 then 
         n1 * n2 * n3 
         else 
         n1 * n3 * n2 
     else if h : n2 < n1 ∧ n2 < n3 then 
       if h2 : n1 < n3 then 
         n2 * n1 * n3
         else 
         n2 * n3 * n1 
     else 
       if h2 : n1 < n2 then 
         n3 * n1 * n2 
         else 
         n3 * n2 * n1) = 224 := sorry

end number_of_factors_in_224_l1586_158653


namespace perpendicular_line_slope_l1586_158613

theorem perpendicular_line_slope (a : ℝ) :
  let M := (0, -1)
  let N := (2, 3)
  let k_MN := (N.2 - M.2) / (N.1 - M.1)
  k_MN * (-a / 2) = -1 → a = 1 :=
by
  intros M N k_MN H
  let M := (0, -1)
  let N := (2, 3)
  let k_MN := (N.2 - M.2) / (N.1 - M.1)
  sorry

end perpendicular_line_slope_l1586_158613


namespace max_sum_cos_l1586_158682

theorem max_sum_cos (a b c : ℝ) (h : ∀ x : ℝ, a * Real.cos x + b * Real.cos (2 * x) + c * Real.cos (3 * x) ≥ -1) : a + b + c ≤ 3 := by
  sorry

end max_sum_cos_l1586_158682


namespace proof_problem_l1586_158674

def f (x : ℝ) : ℝ := 4 * x + 3
def g (x : ℝ) : ℝ := (x - 1)^2

theorem proof_problem : f (g (-3)) = 67 := 
by 
  sorry

end proof_problem_l1586_158674


namespace distance_PF_l1586_158614

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8 * x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define the point P on the parabola with x-coordinate 4
def point_on_parabola (y : ℝ) : ℝ × ℝ := (4, y)

-- Prove the distance |PF| for given conditions
theorem distance_PF
  (hP : ∃ y : ℝ, parabola 4 y)
  (hF : focus = (2, 0)) :
  ∃ y : ℝ, y^2 = 8 * 4 ∧ abs (4 - 2) + abs y = 6 := 
by
  sorry

end distance_PF_l1586_158614


namespace dealer_sold_BMWs_l1586_158697

theorem dealer_sold_BMWs (total_cars : ℕ) (ford_pct toyota_pct nissan_pct bmw_pct : ℝ)
  (h_total_cars : total_cars = 300)
  (h_ford_pct : ford_pct = 0.1)
  (h_toyota_pct : toyota_pct = 0.2)
  (h_nissan_pct : nissan_pct = 0.3)
  (h_bmw_pct : bmw_pct = 1 - (ford_pct + toyota_pct + nissan_pct)) :
  total_cars * bmw_pct = 120 := by
  sorry

end dealer_sold_BMWs_l1586_158697


namespace solution_l1586_158696

def f (x a b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

theorem solution (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 :=
by 
  -- Here we will skip the actual proof by using sorry
  sorry

end solution_l1586_158696


namespace product_of_solutions_of_abs_eq_l1586_158687

theorem product_of_solutions_of_abs_eq (x : ℝ) (h : |x - 5| - 4 = 3) : x * (if x = 12 then -2 else if x = -2 then 12 else 1) = -24 :=
by
  sorry

end product_of_solutions_of_abs_eq_l1586_158687


namespace fair_bets_allocation_l1586_158608

theorem fair_bets_allocation (p_a : ℚ) (p_b : ℚ) (coins : ℚ) 
  (h_prob : p_a = 3 / 4 ∧ p_b = 1 / 4) (h_coins : coins = 96) : 
  (coins * p_a = 72) ∧ (coins * p_b = 24) :=
by 
  sorry

end fair_bets_allocation_l1586_158608


namespace car_distance_kilometers_l1586_158634

theorem car_distance_kilometers (d_amar : ℝ) (d_car : ℝ) (ratio : ℝ) (total_d_amar : ℝ) :
  d_amar = 24 ->
  d_car = 60 ->
  ratio = 2 / 5 ->
  total_d_amar = 880 ->
  (d_car / d_amar) = 5 / 2 ->
  (total_d_amar * 5 / 2) / 1000 = 2.2 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end car_distance_kilometers_l1586_158634


namespace triangle_equilateral_l1586_158651

variable {a b c : ℝ}

theorem triangle_equilateral (h : a^2 + 2 * b^2 = 2 * b * (a + c) - c^2) : a = b ∧ b = c := by
  sorry

end triangle_equilateral_l1586_158651


namespace regular_polygon_sides_l1586_158611

theorem regular_polygon_sides 
  (A B C : ℝ)
  (h₁ : A + B + C = 180)
  (h₂ : B = 3 * A)
  (h₃ : C = 6 * A) :
  ∃ (n : ℕ), n = 5 :=
by
  sorry

end regular_polygon_sides_l1586_158611


namespace Beast_of_War_running_time_l1586_158665

theorem Beast_of_War_running_time 
  (M : ℕ) 
  (AE : ℕ) 
  (BoWAC : ℕ)
  (h1 : M = 120)
  (h2 : AE = M - 30)
  (h3 : BoWAC = AE + 10) : 
  BoWAC = 100 
  := 
sorry

end Beast_of_War_running_time_l1586_158665


namespace solve_exp_equation_l1586_158683

theorem solve_exp_equation (e : ℝ) (x : ℝ) (h_e : e = Real.exp 1) :
  e^x + 2 * e^(-x) = 3 ↔ x = 0 ∨ x = Real.log 2 :=
sorry

end solve_exp_equation_l1586_158683


namespace platform_length_l1586_158610

theorem platform_length (train_length : ℕ) (tree_cross_time : ℕ) (platform_cross_time : ℕ) (platform_length : ℕ)
  (h_train_length : train_length = 1200)
  (h_tree_cross_time : tree_cross_time = 120)
  (h_platform_cross_time : platform_cross_time = 160)
  (h_speed_calculation : (train_length / tree_cross_time = 10))
  : (train_length + platform_length) / 10 = platform_cross_time → platform_length = 400 :=
sorry

end platform_length_l1586_158610


namespace Dana_pencils_equals_combined_l1586_158607

-- Definitions based on given conditions
def pencils_Jayden : ℕ := 20
def pencils_Marcus (pencils_Jayden : ℕ) : ℕ := pencils_Jayden / 2
def pencils_Dana (pencils_Jayden : ℕ) : ℕ := pencils_Jayden + 15
def pencils_Ella (pencils_Marcus : ℕ) : ℕ := 3 * pencils_Marcus - 5
def combined_pencils (pencils_Marcus : ℕ) (pencils_Ella : ℕ) : ℕ := pencils_Marcus + pencils_Ella

-- Theorem to prove:
theorem Dana_pencils_equals_combined (pencils_Jayden : ℕ := 20) : 
  pencils_Dana pencils_Jayden = combined_pencils (pencils_Marcus pencils_Jayden) (pencils_Ella (pencils_Marcus pencils_Jayden)) := by
  sorry

end Dana_pencils_equals_combined_l1586_158607


namespace inscribed_circle_radii_rel_l1586_158618

theorem inscribed_circle_radii_rel {a b c r r1 r2 : ℝ} :
  (a^2 + b^2 = c^2) ∧
  (r1 = (a / c) * r) ∧
  (r2 = (b / c) * r) →
  r^2 = r1^2 + r2^2 :=
by 
  sorry

end inscribed_circle_radii_rel_l1586_158618


namespace expression_value_correct_l1586_158677

theorem expression_value_correct (a b : ℤ) (h1 : a = -3) (h2 : b = 2) : -a - b^3 + a * b = -11 := by
  sorry

end expression_value_correct_l1586_158677


namespace total_cost_is_correct_l1586_158619

-- Definitions based on conditions
def bedroomDoorCount : ℕ := 3
def outsideDoorCount : ℕ := 2
def outsideDoorCost : ℕ := 20
def bedroomDoorCost : ℕ := outsideDoorCost / 2

-- Total costs calculations
def totalBedroomCost : ℕ := bedroomDoorCount * bedroomDoorCost
def totalOutsideCost : ℕ := outsideDoorCount * outsideDoorCost
def totalCost : ℕ := totalBedroomCost + totalOutsideCost

-- Proof statement
theorem total_cost_is_correct : totalCost = 70 := 
by
  sorry

end total_cost_is_correct_l1586_158619


namespace find_sum_invested_l1586_158645

noncomputable def sum_invested (interest_difference: ℝ) (rate1: ℝ) (rate2: ℝ) (time: ℝ): ℝ := 
  interest_difference * 100 / (time * (rate1 - rate2))

theorem find_sum_invested :
  let interest_difference := 600
  let rate1 := 18 / 100
  let rate2 := 12 / 100
  let time := 2
  sum_invested interest_difference rate1 rate2 time = 5000 :=
by
  sorry

end find_sum_invested_l1586_158645


namespace abc_sum_seven_l1586_158664

theorem abc_sum_seven (a b c : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : (a + b + c)^3 - a^3 - b^3 - c^3 = 210) : a + b + c = 7 :=
sorry

end abc_sum_seven_l1586_158664


namespace boat_width_l1586_158640

-- Definitions: river width, number of boats, and space between/banks
def river_width : ℝ := 42
def num_boats : ℕ := 8
def space_between : ℝ := 2

-- Prove the width of each boat given the conditions
theorem boat_width : 
  ∃ w : ℝ, 
    8 * w + 7 * space_between + 2 * space_between = river_width ∧
    w = 3 :=
by
  sorry

end boat_width_l1586_158640


namespace dogs_eat_each_day_l1586_158699

theorem dogs_eat_each_day (h1 : 0.125 + 0.125 = 0.25) : true := by
  sorry

end dogs_eat_each_day_l1586_158699


namespace disk_difference_l1586_158679

/-- Given the following conditions:
    1. Every disk is either blue, yellow, green, or red.
    2. The ratio of blue disks to yellow disks to green disks to red disks is 3 : 7 : 8 : 4.
    3. The total number of disks in the bag is 176.
    Prove that the number of green disks minus the number of blue disks is 40.
-/
theorem disk_difference (b y g r : ℕ) (h_ratio : b * 7 = y * 3 ∧ b * 8 = g * 3 ∧ b * 4 = r * 3) (h_total : b + y + g + r = 176) : g - b = 40 :=
by
  sorry

end disk_difference_l1586_158679


namespace difference_of_squares_l1586_158604

theorem difference_of_squares : (23 + 15)^2 - (23 - 15)^2 = 1380 := by
  sorry

end difference_of_squares_l1586_158604


namespace sum_x_coords_Q3_is_132_l1586_158663

noncomputable def sum_x_coords_Q3 (x_coords: Fin 44 → ℝ) (sum_x1: ℝ) : ℝ :=
  sum_x1 -- given sum_x1 is the sum of x-coordinates of Q1 i.e., 132

theorem sum_x_coords_Q3_is_132 (x_coords: Fin 44 → ℝ) (sum_x1: ℝ) (h: sum_x1 = 132) :
  sum_x_coords_Q3 x_coords sum_x1 = 132 :=
by
  sorry

end sum_x_coords_Q3_is_132_l1586_158663


namespace fraction_simplify_l1586_158694

theorem fraction_simplify : (7 + 21) / (14 + 42) = 1 / 2 := by
  sorry

end fraction_simplify_l1586_158694


namespace find_a_l1586_158659

-- Define the given context (condition)
def condition (a : ℝ) : Prop := 0.5 / 100 * a = 75 / 100 -- since 1 paise = 1/100 rupee

-- Define the statement to prove
theorem find_a (a : ℝ) (h : condition a) : a = 150 := 
sorry

end find_a_l1586_158659


namespace vasya_wins_l1586_158680

-- Definition of the game and players
inductive Player
| Vasya : Player
| Petya : Player

-- Define the problem conditions
structure Game where
  initial_piles : ℕ := 1      -- Initially, there is one pile
  players_take_turns : Bool := true
  take_or_divide : Bool := true
  remove_last_wins : Bool := true
  vasya_first_but_cannot_take_initially : Bool := true

-- Define the function to determine the winner
def winner_of_game (g : Game) : Player :=
  if g.initial_piles = 1 ∧ g.vasya_first_but_cannot_take_initially then Player.Vasya else Player.Petya

-- Define the theorem stating Vasya will win given the game conditions
theorem vasya_wins : ∀ (g : Game), g = {
    initial_piles := 1,
    players_take_turns := true,
    take_or_divide := true,
    remove_last_wins := true,
    vasya_first_but_cannot_take_initially := true
} → winner_of_game g = Player.Vasya := by
  -- Insert proof here
  sorry

end vasya_wins_l1586_158680


namespace calc_f_at_3_l1586_158693

def f (x : ℝ) : ℝ := x^5 + 2*x^3 + 3*x^2 + x + 1

theorem calc_f_at_3 : f 3 = 328 := 
sorry

end calc_f_at_3_l1586_158693


namespace neg_prop_p_l1586_158662

-- Define the function f as a real-valued function
variable (f : ℝ → ℝ)

-- Definitions for the conditions in the problem
def prop_p := ∀ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) ≥ 0

-- Theorem stating the negation of proposition p
theorem neg_prop_p : ¬prop_p f ↔ ∃ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) < 0 :=
by 
  sorry

end neg_prop_p_l1586_158662


namespace segment_length_R_R_l1586_158644

theorem segment_length_R_R' :
  let R := (-4, 1)
  let R' := (-4, -1)
  let distance : ℝ := Real.sqrt ((R'.1 - R.1)^2 + (R'.2 - R.2)^2)
  distance = 2 :=
by
  sorry

end segment_length_R_R_l1586_158644


namespace periodic_function_l1586_158648

open Real

theorem periodic_function (f : ℝ → ℝ) 
  (h_bound : ∀ x : ℝ, |f x| ≤ 1)
  (h_func_eq : ∀ x : ℝ, f (x + 13 / 42) + f x = f (x + 1 / 6) + f (x + 1 / 7)) : 
  ∀ x : ℝ, f (x + 1) = f x := 
  sorry

end periodic_function_l1586_158648


namespace complex_sum_l1586_158681

noncomputable def omega : ℂ := sorry
axiom h1 : omega^11 = 1
axiom h2 : omega ≠ 1

theorem complex_sum 
: omega^10 + omega^14 + omega^18 + omega^22 + omega^26 + omega^30 + omega^34 + omega^38 + omega^42 + omega^46 + omega^50 + omega^54 + omega^58 
= -omega^10 :=
sorry

end complex_sum_l1586_158681


namespace arithmetic_sequence_150th_term_l1586_158615

theorem arithmetic_sequence_150th_term :
  let a₁ := 3
  let d := 4
  a₁ + (150 - 1) * d = 599 :=
by
  sorry

end arithmetic_sequence_150th_term_l1586_158615


namespace initial_books_calculation_l1586_158685

-- Definitions based on conditions
def total_books : ℕ := 77
def additional_books : ℕ := 23

-- Statement of the problem
theorem initial_books_calculation : total_books - additional_books = 54 :=
by
  sorry

end initial_books_calculation_l1586_158685


namespace printer_time_l1586_158638

theorem printer_time (Tx : ℝ) 
  (h1 : ∀ (Ty Tz : ℝ), Ty = 10 → Tz = 20 → 1 / Ty + 1 / Tz = 3 / 20) 
  (h2 : ∀ (T_combined : ℝ), T_combined = 20 / 3 → Tx / T_combined = 2.4) :
  Tx = 16 := 
by 
  sorry

end printer_time_l1586_158638


namespace max_d_77733e_divisible_by_33_l1586_158609

open Int

theorem max_d_77733e_divisible_by_33 : ∃ d e : ℕ, 
  (7 * 100000 + d * 10000 + 7 * 1000 + 3 * 100 + 3 * 10 + e) % 33 = 0 ∧ 
  (d ≤ 9) ∧ (e ≤ 9) ∧ 
  (∀ d' e', ((7 * 100000 + d' * 10000 + 7 * 1000 + 3 * 100 + 3 * 10 + e') % 33 = 0 ∧ d' ≤ 9 ∧ e' ≤ 9 → d' ≤ d)) 
  := ⟨6, 0, by sorry⟩

end max_d_77733e_divisible_by_33_l1586_158609


namespace smallest_possible_value_l1586_158656

theorem smallest_possible_value (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y) (h : (1 / (x : ℝ)) + (1 / (y : ℝ)) = 1 / 15) : x + y = 64 :=
sorry

end smallest_possible_value_l1586_158656


namespace find_pairs_s_t_l1586_158603

theorem find_pairs_s_t (n : ℤ) (hn : n > 1) : 
  ∃ s t : ℤ, (
    (∀ x : ℝ, x ^ n + s * x = 2007 ∧ x ^ n + t * x = 2008 → 
     (s, t) = (2006, 2007) ∨ (s, t) = (-2008, -2009) ∨ (s, t) = (-2006, -2007))
  ) :=
sorry

end find_pairs_s_t_l1586_158603


namespace binomial_prime_div_l1586_158628

theorem binomial_prime_div {p : ℕ} {m : ℕ} (hp : Nat.Prime p) (hm : 0 < m) : (Nat.choose (p ^ m) p - p ^ (m - 1)) % p ^ m = 0 := 
  sorry

end binomial_prime_div_l1586_158628


namespace kaleb_lives_left_l1586_158629

theorem kaleb_lives_left (initial_lives : ℕ) (lives_lost : ℕ) (remaining_lives : ℕ) :
  initial_lives = 98 → lives_lost = 25 → remaining_lives = initial_lives - lives_lost → remaining_lives = 73 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end kaleb_lives_left_l1586_158629


namespace area_of_union_of_rectangle_and_circle_l1586_158617

theorem area_of_union_of_rectangle_and_circle :
  let width := 8
  let length := 12
  let radius := 12
  let A_rectangle := length * width
  let A_circle := Real.pi * radius ^ 2
  let A_overlap := (1 / 4) * A_circle
  A_rectangle + A_circle - A_overlap = 96 + 108 * Real.pi :=
by
  sorry

end area_of_union_of_rectangle_and_circle_l1586_158617


namespace flagpole_height_l1586_158602

theorem flagpole_height (x : ℝ) (h1 : (x + 2)^2 = x^2 + 6^2) : x = 8 := 
by 
  sorry

end flagpole_height_l1586_158602


namespace subsets_containing_six_l1586_158668

theorem subsets_containing_six :
  ∃ s : Finset (Fin 6), s = {1, 2, 3, 4, 5, 6} ∧ (∃ n : ℕ, n = 32 ∧ n = 2 ^ 5) := by
  sorry

end subsets_containing_six_l1586_158668


namespace correct_substitution_l1586_158641

theorem correct_substitution (x y : ℤ) (h1 : x = 3 * y - 1) (h2 : x - 2 * y = 4) :
  3 * y - 1 - 2 * y = 4 :=
by
  sorry

end correct_substitution_l1586_158641


namespace conic_section_is_ellipse_l1586_158616

theorem conic_section_is_ellipse :
  (∃ (x y : ℝ), 3 * x^2 + y^2 - 12 * x - 4 * y + 36 = 0) ∧
  ∀ (x y : ℝ), 3 * x^2 + y^2 - 12 * x - 4 * y + 36 = 0 →
    ((x - 2)^2 / (20 / 3) + (y - 2)^2 / 20 = 1) :=
sorry

end conic_section_is_ellipse_l1586_158616


namespace valid_license_plates_count_l1586_158630

/--
The problem is to prove that the total number of valid license plates under the given format is equal to 45,697,600.
The given conditions are:
1. A valid license plate in Xanadu consists of three letters followed by two digits, and then one more letter at the end.
2. There are 26 choices of letters for each letter spot.
3. There are 10 choices of digits for each digit spot.

We need to conclude that the number of possible license plates is:
26^4 * 10^2 = 45,697,600.
-/

def num_valid_license_plates : Nat :=
  let letter_choices := 26
  let digit_choices := 10
  let total_choices := letter_choices ^ 3 * digit_choices ^ 2 * letter_choices
  total_choices

theorem valid_license_plates_count : num_valid_license_plates = 45697600 := by
  sorry

end valid_license_plates_count_l1586_158630


namespace combinations_of_4_blocks_no_same_row_col_in_6x6_is_5400_l1586_158669

noncomputable def num_combinations_4_blocks_no_same_row_col :=
  (Nat.choose 6 4) * (Nat.choose 6 4) * (Nat.factorial 4)

theorem combinations_of_4_blocks_no_same_row_col_in_6x6_is_5400 :
  num_combinations_4_blocks_no_same_row_col = 5400 := 
by
  sorry

end combinations_of_4_blocks_no_same_row_col_in_6x6_is_5400_l1586_158669


namespace problem1_problem2_l1586_158647

-- Problem (1)
theorem problem1 (a : ℕ → ℤ) (h1 : a 1 = 4) (h2 : ∀ n, a n = a (n + 1) + 3) : a 10 = -23 :=
by {
  sorry
}

-- Problem (2)
theorem problem2 (a : ℕ → ℚ) (h1 : a 6 = (1 / 4)) (h2 : ∃ d : ℚ, ∀ n, 1 / a n = 1 / a 1 + (n - 1) * d) : 
  ∀ n, a n = (4 / (3 * n - 2)) :=
by {
  sorry
}

end problem1_problem2_l1586_158647


namespace smallest_integer_sum_to_2020_l1586_158658

theorem smallest_integer_sum_to_2020 :
  ∃ B : ℤ, (∃ (n : ℤ), (B * (B + 1) / 2) + ((n * (n + 1)) / 2) = 2020) ∧ (∀ C : ℤ, (∃ (m : ℤ), (C * (C + 1) / 2) + ((m * (m + 1)) / 2) = 2020) → B ≤ C) ∧ B = -2019 :=
by
  sorry

end smallest_integer_sum_to_2020_l1586_158658


namespace intersection_solution_l1586_158690

theorem intersection_solution (x : ℝ) (y : ℝ) (h₁ : y = 12 / (x^2 + 6)) (h₂ : x + y = 4) : x = 2 :=
by
  sorry

end intersection_solution_l1586_158690


namespace bc_lt_3ad_l1586_158673

theorem bc_lt_3ad {a b c d x1 x2 x3 : ℝ}
    (h1 : a ≠ 0)
    (h2 : x1 > 0 ∧ x2 > 0 ∧ x3 > 0)
    (h3 : x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3)
    (h4 : x1 + x2 + x3 = -b / a)
    (h5 : x1 * x2 + x2 * x3 + x1 * x3 = c / a)
    (h6 : x1 * x2 * x3 = -d / a) : 
    b * c < 3 * a * d := 
sorry

end bc_lt_3ad_l1586_158673


namespace total_accidents_l1586_158657

theorem total_accidents :
  let accidentsA := (75 / 100) * 2500
  let accidentsB := (50 / 80) * 1600
  let accidentsC := (90 / 200) * 1900
  accidentsA + accidentsB + accidentsC = 3730 :=
by
  let accidentsA := (75 / 100) * 2500
  let accidentsB := (50 / 80) * 1600
  let accidentsC := (90 / 200) * 1900
  sorry

end total_accidents_l1586_158657


namespace range_of_a_l1586_158624

variables {x a : ℝ}

def p (x : ℝ) : Prop := (x - 5) / (x - 3) ≥ 2
def q (x a : ℝ) : Prop := x ^ 2 - a * x ≤ x - a

theorem range_of_a (h : ¬(∃ x, p x) → ¬(∃ x, q x a)) :
  1 ≤ a ∧ a < 3 :=
by 
  sorry

end range_of_a_l1586_158624


namespace values_of_k_real_equal_roots_l1586_158623

theorem values_of_k_real_equal_roots (k : ℝ) :
  (∀ x : ℝ, 3 * x^2 - (k + 2) * x + 12 = 0 → x * x = 0) ↔ (k = 10 ∨ k = -14) :=
by
  sorry

end values_of_k_real_equal_roots_l1586_158623


namespace circle_tangent_to_line_at_parabola_focus_l1586_158649

noncomputable def parabola_focus : (ℝ × ℝ) := (2, 0)

def line_eq (p : ℝ × ℝ) : Prop := p.2 = p.1

def circle_eq (center radius : ℝ) (p : ℝ × ℝ) : Prop := 
  (p.1 - center)^2 + p.2^2 = radius

theorem circle_tangent_to_line_at_parabola_focus : 
  ∀ p : ℝ × ℝ, (circle_eq 2 2 p ↔ (line_eq p ∧ p = parabola_focus)) := by
  sorry

end circle_tangent_to_line_at_parabola_focus_l1586_158649


namespace number_of_matches_in_first_set_l1586_158654

theorem number_of_matches_in_first_set
  (avg_next_13_matches : ℕ := 15)
  (total_matches : ℕ := 35)
  (avg_all_matches : ℚ := 23.17142857142857)
  (x : ℕ := total_matches - 13) :
  x = 22 := by
  sorry

end number_of_matches_in_first_set_l1586_158654


namespace symmetric_circle_eq_l1586_158666

-- Define the original circle equation
def originalCircle (x y : ℝ) : Prop :=
  (x - 1) ^ 2 + (y - 2) ^ 2 = 4

-- Define the equation of the circle symmetric to the original with respect to the y-axis
def symmetricCircle (x y : ℝ) : Prop :=
  (x + 1) ^ 2 + (y - 2) ^ 2 = 4

-- Theorem to prove that the symmetric circle equation is correct
theorem symmetric_circle_eq :
  ∀ x y : ℝ, originalCircle x y → symmetricCircle (-x) y := 
by
  sorry

end symmetric_circle_eq_l1586_158666


namespace total_interest_paid_l1586_158670

-- Define the problem as a theorem in Lean 4
theorem total_interest_paid
  (initial_investment : ℝ)
  (interest_6_months : ℝ)
  (interest_10_months : ℝ)
  (interest_18_months : ℝ)
  (total_interest : ℝ) :
  initial_investment = 10000 ∧ 
  interest_6_months = 0.02 * initial_investment ∧
  interest_10_months = 0.03 * (initial_investment + interest_6_months) ∧
  interest_18_months = 0.04 * (initial_investment + interest_6_months + interest_10_months) ∧
  total_interest = interest_6_months + interest_10_months + interest_18_months →
  total_interest = 926.24 :=
by
  sorry

end total_interest_paid_l1586_158670


namespace not_product_24_pair_not_24_l1586_158672

theorem not_product_24 (a b : ℤ) : 
  (a, b) = (-4, -6) ∨ (a, b) = (-2, -12) ∨ (a, b) = (2, 12) ∨ (a, b) = (3/4, 32) → a * b = 24 :=
sorry

theorem pair_not_24 :
  ¬(1/3 * -72 = 24) :=
sorry

end not_product_24_pair_not_24_l1586_158672


namespace koala_fiber_intake_l1586_158620

theorem koala_fiber_intake (x : ℝ) (h : 0.30 * x = 12) : x = 40 := 
sorry

end koala_fiber_intake_l1586_158620


namespace largest_possible_rational_root_l1586_158642

noncomputable def rational_root_problem : Prop :=
  ∃ (a b c : ℕ), (a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≤ 100 ∧ b ≤ 100 ∧ c ≤ 100) ∧
  ∀ p q : ℤ, (q ≠ 0) → (a * p^2 + b * p + c * q = 0) → 
  (p / q) ≤ -1 / 99

theorem largest_possible_rational_root : rational_root_problem :=
sorry

end largest_possible_rational_root_l1586_158642


namespace area_of_PQRS_l1586_158600

noncomputable def length_square_EFGH := 6
noncomputable def height_equilateral_triangle := 3 * Real.sqrt 3
noncomputable def diagonal_PQRS := length_square_EFGH + 2 * height_equilateral_triangle
noncomputable def area_PQRS := (1 / 2) * (diagonal_PQRS * diagonal_PQRS)

theorem area_of_PQRS :
  (area_PQRS = 72 + 36 * Real.sqrt 3) :=
sorry

end area_of_PQRS_l1586_158600


namespace largest_difference_l1586_158637

def A : ℕ := 3 * 2005^2006
def B : ℕ := 2005^2006
def C : ℕ := 2004 * 2005^2005
def D : ℕ := 3 * 2005^2005
def E : ℕ := 2005^2005
def F : ℕ := 2005^2004

theorem largest_difference : (A - B > B - C) ∧ (A - B > C - D) ∧ (A - B > D - E) ∧ (A - B > E - F) :=
by
  sorry  -- Proof is omitted as per instructions.

end largest_difference_l1586_158637


namespace polynomial_rewrite_l1586_158652

theorem polynomial_rewrite :
  ∃ (a b c d e f : ℤ), 
  (2401 * x^4 + 16 = (a * x + b) * (c * x^3 + d * x^2 + e * x + f)) ∧
  (a + b + c + d + e + f = 274) :=
sorry

end polynomial_rewrite_l1586_158652


namespace friends_count_is_four_l1586_158605

def number_of_friends (Melanie Benny Sally Jessica : ℕ) (total_cards : ℕ) : ℕ :=
  4

theorem friends_count_is_four (Melanie Benny Sally Jessica : ℕ) (total_cards : ℕ) (h1 : total_cards = 12) :
  number_of_friends Melanie Benny Sally Jessica total_cards = 4 :=
by
  sorry

end friends_count_is_four_l1586_158605


namespace sam_has_12_nickels_l1586_158661

theorem sam_has_12_nickels (n d : ℕ) (h1 : n + d = 30) (h2 : 5 * n + 10 * d = 240) : n = 12 :=
sorry

end sam_has_12_nickels_l1586_158661


namespace evaluate_expression_l1586_158678

theorem evaluate_expression :
  (1 / (-8^2)^4) * (-8)^9 = -8 := by
  sorry

end evaluate_expression_l1586_158678


namespace average_age_group_l1586_158622

theorem average_age_group (n : ℕ) (T : ℕ) (h1 : T = n * 14) (h2 : T + 32 = (n + 1) * 15) : n = 17 :=
by
  sorry

end average_age_group_l1586_158622


namespace slips_numbers_exist_l1586_158635

theorem slips_numbers_exist (x y z : ℕ) (h₁ : x + y + z = 20) (h₂ : 5 * x + 3 * y = 46) : 
  (x = 4) ∧ (y = 10) ∧ (z = 6) :=
by {
  -- Technically, the actual proving steps should go here, but skipped due to 'sorry'
  sorry
}

end slips_numbers_exist_l1586_158635


namespace total_pencils_correct_l1586_158621

def pencils_in_drawer : ℕ := 43
def pencils_on_desk_originally : ℕ := 19
def pencils_added_by_dan : ℕ := 16
def total_pencils : ℕ := pencils_in_drawer + pencils_on_desk_originally + pencils_added_by_dan

theorem total_pencils_correct : total_pencils = 78 := by
  sorry

end total_pencils_correct_l1586_158621


namespace intersection_complement_l1586_158675

-- Definitions of the sets
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}

-- The complement of B in U
def complement_U (U B : Set ℕ) : Set ℕ := U \ B

-- Statement to prove
theorem intersection_complement : A ∩ (complement_U U B) = {1} := 
by 
  sorry

end intersection_complement_l1586_158675


namespace rock_height_at_30_l1586_158695

theorem rock_height_at_30 (t : ℝ) (h : ℝ) 
  (h_eq : h = 80 - 9 * t - 5 * t^2) 
  (h_30 : h = 30) : 
  t = 2.3874 :=
by
  -- Proof omitted
  sorry

end rock_height_at_30_l1586_158695


namespace smallest_prime_12_less_than_square_l1586_158627

def is_perfect_square (n : ℕ) := ∃ k : ℕ, k * k = n

def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem smallest_prime_12_less_than_square : ∃ n : ℕ, is_prime n ∧ (∃ k : ℕ, k * k = n + 12) ∧ n = 13 :=
by
  sorry

end smallest_prime_12_less_than_square_l1586_158627


namespace original_number_increased_by_45_percent_is_870_l1586_158601

theorem original_number_increased_by_45_percent_is_870 (x : ℝ) (h : x * 1.45 = 870) : x = 870 / 1.45 :=
by sorry

end original_number_increased_by_45_percent_is_870_l1586_158601


namespace Lesha_received_11_gifts_l1586_158606

theorem Lesha_received_11_gifts (x : ℕ) 
    (h1 : x < 100) 
    (h2 : x % 2 = 0) 
    (h3 : x % 5 = 0) 
    (h4 : x % 7 = 0) :
    x - (x / 2 + x / 5 + x / 7) = 11 :=
by {
    sorry
}

end Lesha_received_11_gifts_l1586_158606


namespace arithmetic_mean_three_fractions_l1586_158684

theorem arithmetic_mean_three_fractions :
  let a := (5 : ℚ) / 8
  let b := (7 : ℚ) / 8
  let c := (3 : ℚ) / 4
  (a + b) / 2 = c :=
by
  sorry

end arithmetic_mean_three_fractions_l1586_158684


namespace planting_cost_l1586_158631

-- Define the costs of the individual items
def cost_of_flowers : ℝ := 9
def cost_of_clay_pot : ℝ := cost_of_flowers + 20
def cost_of_soil : ℝ := cost_of_flowers - 2
def cost_of_fertilizer : ℝ := cost_of_flowers + (0.5 * cost_of_flowers)
def cost_of_tools : ℝ := cost_of_clay_pot - (0.25 * cost_of_clay_pot)

-- Define the total cost
def total_cost : ℝ :=
  cost_of_flowers + cost_of_clay_pot + cost_of_soil + cost_of_fertilizer + cost_of_tools

-- The statement to prove
theorem planting_cost : total_cost = 80.25 :=
by
  sorry

end planting_cost_l1586_158631


namespace stamps_count_l1586_158686

theorem stamps_count {x : ℕ} (h1 : x % 3 = 1) (h2 : x % 5 = 3) (h3 : x % 7 = 5) (h4 : 150 < x ∧ x ≤ 300) :
  x = 208 :=
sorry

end stamps_count_l1586_158686


namespace math_problem_l1586_158691

   theorem math_problem :
     6 * (-1 / 2) + Real.sqrt 3 * Real.sqrt 8 + (-15 : ℝ)^0 = 2 * Real.sqrt 6 - 2 :=
   by
     sorry
   
end math_problem_l1586_158691
