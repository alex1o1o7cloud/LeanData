import Mathlib

namespace range_of_a_l581_58160

variable (a x y : ℝ)

theorem range_of_a (h1 : 2 * x + y = 1 + 4 * a) (h2 : x + 2 * y = 2 - a) (h3 : x + y > 0) : a > -1 :=
sorry

end range_of_a_l581_58160


namespace length_of_the_bridge_l581_58141

theorem length_of_the_bridge
  (train_length : ℕ)
  (train_speed_kmh : ℕ)
  (cross_time_s : ℕ)
  (h_train_length : train_length = 120)
  (h_train_speed_kmh : train_speed_kmh = 45)
  (h_cross_time_s : cross_time_s = 30) :
  ∃ bridge_length : ℕ, bridge_length = 255 := 
by 
  sorry

end length_of_the_bridge_l581_58141


namespace determine_value_of_x_l581_58108

theorem determine_value_of_x (x y z : ℤ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxy : x ≥ y) (hyz : y ≥ z)
  (h1 : x^2 - y^2 - z^2 + x * y = 4033) 
  (h2 : x^2 + 4 * y^2 + 4 * z^2 - 4 * x * y - 3 * x * z - 3 * y * z = -3995) : 
  x = 69 := sorry

end determine_value_of_x_l581_58108


namespace minimum_cost_to_store_food_l581_58143

-- Define the problem setting
def total_volume : ℕ := 15
def capacity_A : ℕ := 2
def capacity_B : ℕ := 3
def price_A : ℕ := 13
def price_B : ℕ := 15
def cashback_threshold : ℕ := 3
def cashback : ℕ := 10

-- The mathematical theorem statement for the proof problem
theorem minimum_cost_to_store_food : 
  ∃ (x y : ℕ), 
    capacity_A * x + capacity_B * y = total_volume ∧ 
    (y = 5 ∧ price_B * y = 75) ∨ 
    (x = 3 ∧ y = 3 ∧ price_A * x + price_B * y - cashback = 74) :=
sorry

end minimum_cost_to_store_food_l581_58143


namespace sum_of_intercepts_l581_58127

theorem sum_of_intercepts (x y : ℝ) (h : y + 3 = -2 * (x + 5)) : 
  (- (13 / 2) : ℝ) + (- 13 : ℝ) = - (39 / 2) :=
by sorry

end sum_of_intercepts_l581_58127


namespace wine_age_problem_l581_58150

theorem wine_age_problem
  (carlo_rosi : ℕ)
  (franzia : ℕ)
  (twin_valley : ℕ)
  (h1 : franzia = 3 * carlo_rosi)
  (h2 : carlo_rosi = 4 * twin_valley)
  (h3 : carlo_rosi = 40) :
  franzia + carlo_rosi + twin_valley = 170 :=
by
  sorry

end wine_age_problem_l581_58150


namespace increasing_a_n_not_increasing_n_a_n_increasing_a_n_over_n_not_increasing_a_n_sq_l581_58135

def a_n (n : ℕ) : ℤ := 2 * n - 8

theorem increasing_a_n : ∀ n : ℕ, a_n (n + 1) > a_n n := 
by 
-- Assuming n >= 0
intro n
dsimp [a_n]
sorry

def n_a_n (n : ℕ) : ℤ := n * (2 * n - 8)

theorem not_increasing_n_a_n : ∀ n : ℕ, n > 0 → n_a_n (n + 1) ≤ n_a_n n :=
by
-- Assuming n > 0
intro n hn
dsimp [n_a_n]
sorry

def a_n_over_n (n : ℕ) : ℚ := (2 * n - 8 : ℚ) / n

theorem increasing_a_n_over_n : ∀ n > 0, a_n_over_n (n + 1) > a_n_over_n n :=
by 
-- Assuming n > 0
intro n hn
dsimp [a_n_over_n]
sorry

def a_n_sq (n : ℕ) : ℤ := (2 * n - 8) * (2 * n - 8)

theorem not_increasing_a_n_sq : ∀ n : ℕ, a_n_sq (n + 1) ≤ a_n_sq n :=
by
-- Assuming n >= 0
intro n
dsimp [a_n_sq]
sorry

end increasing_a_n_not_increasing_n_a_n_increasing_a_n_over_n_not_increasing_a_n_sq_l581_58135


namespace problem_solution_l581_58121

noncomputable def sqrt_3_simplest : Prop :=
  let A := Real.sqrt 3
  let B := Real.sqrt 0.5
  let C := Real.sqrt 8
  let D := Real.sqrt (1 / 3)
  ∀ (x : ℝ), x = A ∨ x = B ∨ x = C ∨ x = D → x = A → 
    (x = Real.sqrt 0.5 ∨ x = Real.sqrt 8 ∨ x = Real.sqrt (1 / 3)) ∧ 
    ¬(x = Real.sqrt 0.5 ∨ x = 2 * Real.sqrt 2 ∨ x = Real.sqrt (1 / 3))

theorem problem_solution : sqrt_3_simplest :=
by
  sorry

end problem_solution_l581_58121


namespace fishing_problem_l581_58147

theorem fishing_problem
  (P : ℕ) -- weight of the fish Peter caught
  (H1 : Ali_weight = 2 * P) -- Ali caught twice as much as Peter
  (H2 : Joey_weight = P + 1) -- Joey caught 1 kg more than Peter
  (H3 : P + 2 * P + (P + 1) = 25) -- Together they caught 25 kg
  : Ali_weight = 12 :=
by
  sorry

end fishing_problem_l581_58147


namespace bahs_from_yahs_l581_58100

theorem bahs_from_yahs (b r y : ℝ) 
  (h1 : 18 * b = 30 * r) 
  (h2 : 10 * r = 25 * y) : 
  1250 * y = 300 * b := 
by
  sorry

end bahs_from_yahs_l581_58100


namespace silk_dyed_amount_l581_58120

-- Define the conditions
def yards_green : ℕ := 61921
def yards_pink : ℕ := 49500

-- Define the total calculation
def total_yards : ℕ := yards_green + yards_pink

-- State what needs to be proven: that the total yards is 111421
theorem silk_dyed_amount : total_yards = 111421 := by
  sorry

end silk_dyed_amount_l581_58120


namespace find_a_l581_58168

-- Define the function f
def f (a x : ℝ) := a * x^3 - 2 * x

-- State the theorem, asserting that if f passes through the point (-1, 4) then a = -2.
theorem find_a (a : ℝ) (h : f a (-1) = 4) : a = -2 :=
by {
    sorry
}

end find_a_l581_58168


namespace john_trip_time_l581_58188

theorem john_trip_time (normal_distance : ℕ) (normal_time : ℕ) (extra_distance : ℕ) 
  (double_extra_distance : ℕ) (same_speed : ℕ) 
  (h1: normal_distance = 150) 
  (h2: normal_time = 3) 
  (h3: extra_distance = 50)
  (h4: double_extra_distance = 2 * extra_distance)
  (h5: same_speed = normal_distance / normal_time) : 
  normal_time + double_extra_distance / same_speed = 5 :=
by 
  sorry

end john_trip_time_l581_58188


namespace prime_number_conditions_l581_58115

theorem prime_number_conditions :
  ∃ p n : ℕ, Prime p ∧ p = n^2 + 9 ∧ p = (n+1)^2 - 8 :=
by
  sorry

end prime_number_conditions_l581_58115


namespace relationship_between_a_b_c_l581_58177

theorem relationship_between_a_b_c :
  let m := 2
  let n := 3
  let f (x : ℝ) := x^3
  let a := f (Real.sqrt 3 / 3)
  let b := f (Real.log Real.pi)
  let c := f (Real.sqrt 2 / 2)
  a < c ∧ c < b :=
by
  sorry

end relationship_between_a_b_c_l581_58177


namespace max_passengers_l581_58101

theorem max_passengers (total_stops : ℕ) (bus_capacity : ℕ)
  (h_total_stops : total_stops = 12) 
  (h_bus_capacity : bus_capacity = 20) 
  (h_no_same_stop : ∀ (a b : ℕ), a ≠ b → (a < total_stops) → (b < total_stops) → 
    ∃ x y : ℕ, x ≠ y ∧ x < total_stops ∧ y < total_stops ∧ 
    ((x = a ∧ y ≠ a) ∨ (x ≠ b ∧ y = b))) :
  ∃ max_passengers : ℕ, max_passengers = 50 :=
  sorry

end max_passengers_l581_58101


namespace intersection_M_N_l581_58198

-- Definitions of sets M and N
def M : Set ℝ := {x | x < 2}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

-- The statement to prove
theorem intersection_M_N : M ∩ N = {x | 1 ≤ x ∧ x < 2} := 
by 
  sorry

end intersection_M_N_l581_58198


namespace neg_p_l581_58166

open Real

variable {f : ℝ → ℝ}

theorem neg_p :
  (∀ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) ≥ 0) →
  ∃ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) < 0 :=
sorry

end neg_p_l581_58166


namespace gcd_f100_f101_l581_58195

-- Define the function f
def f (x : ℤ) : ℤ := x^2 - x + 2010

-- A statement asserting the greatest common divisor of f(100) and f(101) is 10
theorem gcd_f100_f101 : Int.gcd (f 100) (f 101) = 10 := by
  sorry

end gcd_f100_f101_l581_58195


namespace surface_dots_sum_l581_58102

-- Define the sum of dots on opposite faces of a standard die
axiom sum_opposite_faces (x y : ℕ) : x + y = 7

-- Define the large cube dimensions
def large_cube_dimension : ℕ := 3

-- Define the total number of small cubes
def num_small_cubes : ℕ := large_cube_dimension ^ 3

-- Calculate the number of faces on the surface of the large cube
def num_surface_faces : ℕ := 6 * large_cube_dimension ^ 2

-- Given the sum of opposite faces, compute the total number of dots on the surface
theorem surface_dots_sum : num_surface_faces / 2 * 7 = 189 := by
  sorry

end surface_dots_sum_l581_58102


namespace find_x_l581_58148

theorem find_x : ∃ x, (2015 + x)^2 = x^2 ∧ x = -2015 / 2 := 
by
  sorry

end find_x_l581_58148


namespace find_b_in_triangle_l581_58159

-- Given conditions
variable {A B C : ℝ}
variable {a b c : ℝ}
variable (h1 : a = 3)
variable (h2 : c = 2 * Real.sqrt 3)
variable (h3 : b * Real.sin A = a * Real.cos (B + Real.pi / 6))

-- The proof goal
theorem find_b_in_triangle (h1 : a = 3) (h2 : c = 2 * Real.sqrt 3) (h3 : b * Real.sin A = a * Real.cos (B + Real.pi / 6)) : b = Real.sqrt 3 :=
sorry

end find_b_in_triangle_l581_58159


namespace greatest_integer_gcd_30_is_125_l581_58117

theorem greatest_integer_gcd_30_is_125 : ∃ n : ℕ, n < 150 ∧ Nat.gcd n 30 = 5 ∧ ∀ k : ℕ, k < 150 ∧ Nat.gcd k 30 = 5 → k ≤ n := 
sorry

end greatest_integer_gcd_30_is_125_l581_58117


namespace apples_remaining_l581_58105

-- Define the initial conditions
def number_of_trees := 52
def apples_on_tree_before := 9
def apples_picked := 2

-- Define the target proof: the number of apples remaining on the tree
def apples_on_tree_after := apples_on_tree_before - apples_picked

-- The statement we aim to prove
theorem apples_remaining : apples_on_tree_after = 7 := sorry

end apples_remaining_l581_58105


namespace max_books_l581_58151

theorem max_books (cost_per_book : ℝ) (total_money : ℝ) (h_cost : cost_per_book = 8.75) (h_money : total_money = 250.0) :
  ∃ n : ℕ, n = 28 ∧ cost_per_book * n ≤ total_money ∧ ∀ m : ℕ, cost_per_book * m ≤ total_money → m ≤ 28 :=
by
  sorry

end max_books_l581_58151


namespace total_volume_correct_l581_58186

-- Definitions based on the conditions
def box_length := 30 -- in cm
def box_width := 1 -- in cm
def box_height := 1 -- in cm
def horizontal_rows := 7
def vertical_rows := 5
def floors := 3

-- The volume of a single box
def box_volume : Int := box_length * box_width * box_height

-- The total number of boxes is the product of rows and floors
def total_boxes : Int := horizontal_rows * vertical_rows * floors

-- The total volume of all the boxes
def total_volume : Int := box_volume * total_boxes

-- The statement to prove
theorem total_volume_correct : total_volume = 3150 := 
by 
  simp [box_volume, total_boxes, total_volume]
  sorry

end total_volume_correct_l581_58186


namespace percentage_books_not_sold_is_60_percent_l581_58185

def initial_stock : ℕ := 700
def sold_monday : ℕ := 50
def sold_tuesday : ℕ := 82
def sold_wednesday : ℕ := 60
def sold_thursday : ℕ := 48
def sold_friday : ℕ := 40

def total_sold : ℕ := sold_monday + sold_tuesday + sold_wednesday + sold_thursday + sold_friday
def books_not_sold : ℕ := initial_stock - total_sold
def percentage_not_sold : ℚ := (books_not_sold * 100) / initial_stock

theorem percentage_books_not_sold_is_60_percent : percentage_not_sold = 60 := by
  sorry

end percentage_books_not_sold_is_60_percent_l581_58185


namespace square_perimeter_calculation_l581_58137

noncomputable def perimeter_of_square (radius: ℝ) : ℝ := 
  if radius = 4 then 64 * Real.sqrt 2 else 0

theorem square_perimeter_calculation :
  perimeter_of_square 4 = 64 * Real.sqrt 2 :=
by
  sorry

end square_perimeter_calculation_l581_58137


namespace length_of_fourth_side_in_cyclic_quadrilateral_l581_58192

theorem length_of_fourth_side_in_cyclic_quadrilateral :
  ∀ (r a b c : ℝ), r = 300 ∧ a = 300 ∧ b = 300 ∧ c = 150 * Real.sqrt 2 →
  ∃ d : ℝ, d = 450 :=
by
  sorry

end length_of_fourth_side_in_cyclic_quadrilateral_l581_58192


namespace remainder_of_6_pow_1234_mod_13_l581_58172

theorem remainder_of_6_pow_1234_mod_13 : 6 ^ 1234 % 13 = 10 := 
by 
  sorry

end remainder_of_6_pow_1234_mod_13_l581_58172


namespace curve_has_axis_of_symmetry_l581_58178

theorem curve_has_axis_of_symmetry (x y : ℝ) :
  (x^2 - x * y + y^2 + x - y - 1 = 0) ↔ (x+y = 0) :=
sorry

end curve_has_axis_of_symmetry_l581_58178


namespace correct_statements_l581_58169

def f (x : ℝ) (b : ℝ) (c : ℝ) (d : ℝ) : ℝ := x^3 + b*x^2 + c*x + d
def f_prime (x : ℝ) (b : ℝ) (c : ℝ) : ℝ := 3*x^2 + 2*b*x + c

theorem correct_statements (b c d : ℝ) :
  (∃ x : ℝ, f x b c d = 4 ∧ f_prime x b c = 0) ∧
  (∃ x : ℝ, f x b c d = 0 ∧ f_prime x b c = 0) :=
by
  sorry

end correct_statements_l581_58169


namespace integers_abs_le_3_l581_58106

theorem integers_abs_le_3 :
  {x : ℤ | |x| ≤ 3} = { -3, -2, -1, 0, 1, 2, 3 } :=
by
  sorry

end integers_abs_le_3_l581_58106


namespace find_mode_l581_58122

def scores : List ℕ :=
  [105, 107, 111, 111, 112, 112, 115, 118, 123, 124, 124, 126, 127, 129, 129, 129, 130, 130, 130, 130, 131, 140, 140, 140, 140]

def mode (ls : List ℕ) : ℕ :=
  ls.foldl (λmodeScore score => if ls.count score > ls.count modeScore then score else modeScore) 0

theorem find_mode :
  mode scores = 130 :=
by
  sorry

end find_mode_l581_58122


namespace isosceles_triangle_sides_l581_58158

theorem isosceles_triangle_sides (r R : ℝ) (a b c : ℝ) (h1 : r = 3 / 2) (h2 : R = 25 / 8)
  (h3 : a = c) (h4 : 5 = a) (h5 : 6 = b) : 
  ∃ a b c, a = 5 ∧ c = 5 ∧ b = 6 := by 
  sorry

end isosceles_triangle_sides_l581_58158


namespace sufficient_condition_l581_58180

theorem sufficient_condition (x y : ℤ) (h : x + y ≠ 2) : x ≠ 1 ∧ y ≠ 1 := 
sorry

end sufficient_condition_l581_58180


namespace running_speed_proof_l581_58116

-- Definitions used in the conditions
def num_people : ℕ := 4
def stretch_km : ℕ := 300
def bike_speed_kmph : ℕ := 50
def total_time_hours : ℚ := 19 + (1/3)

-- The running speed to be proven
def running_speed_kmph : ℚ := 15.52

-- The main statement
theorem running_speed_proof
  (num_people_eq : num_people = 4)
  (stretch_eq : stretch_km = 300)
  (bike_speed_eq : bike_speed_kmph = 50)
  (total_time_eq : total_time_hours = 19.333333333333332) :
  running_speed_kmph = 15.52 :=
sorry

end running_speed_proof_l581_58116


namespace photographer_max_photos_l581_58142

-- The initial number of birds of each species
def total_birds : ℕ := 20
def starlings : ℕ := 8
def wagtails : ℕ := 7
def woodpeckers : ℕ := 5

-- Define a function to count the remaining birds of each species after n photos
def remaining_birds (n : ℕ) (species : ℕ) : ℕ := species - (if species ≤ n then species else n)

-- Define the main theorem we want to prove
theorem photographer_max_photos (n : ℕ) (h1 : remaining_birds n starlings ≥ 4) (h2 : remaining_birds n wagtails ≥ 3) : 
  n ≤ 7 :=
by
  sorry

end photographer_max_photos_l581_58142


namespace total_volume_of_mixed_solutions_l581_58123

theorem total_volume_of_mixed_solutions :
  let v1 := 3.6
  let v2 := 1.4
  v1 + v2 = 5.0 := by
  sorry

end total_volume_of_mixed_solutions_l581_58123


namespace james_net_profit_l581_58191

def totalCandyBarsSold (boxes : Nat) (candyBarsPerBox : Nat) : Nat :=
  boxes * candyBarsPerBox

def revenue30CandyBars (pricePerCandyBar : Real) : Real :=
  30 * pricePerCandyBar

def revenue20CandyBars (pricePerCandyBar : Real) : Real :=
  20 * pricePerCandyBar

def totalRevenue (revenue1 : Real) (revenue2 : Real) : Real :=
  revenue1 + revenue2

def costNonDiscountedBoxes (candyBars : Nat) (pricePerCandyBar : Real) : Real :=
  candyBars * pricePerCandyBar

def costDiscountedBoxes (candyBars : Nat) (pricePerCandyBar : Real) : Real :=
  candyBars * pricePerCandyBar

def totalCost (cost1 : Real) (cost2 : Real) : Real :=
  cost1 + cost2

def salesTax (totalRevenue : Real) (taxRate : Real) : Real :=
  totalRevenue * taxRate

def totalExpenses (cost : Real) (salesTax : Real) (fixedExpense : Real) : Real :=
  cost + salesTax + fixedExpense

def netProfit (totalRevenue : Real) (totalExpenses : Real) : Real :=
  totalRevenue - totalExpenses

theorem james_net_profit :
  let boxes := 5
  let candyBarsPerBox := 10
  let totalCandyBars := totalCandyBarsSold boxes candyBarsPerBox

  let priceFirst30 := 1.50
  let priceNext20 := 1.30
  let priceSubsequent := 1.10

  let revenueFirst30 := revenue30CandyBars priceFirst30
  let revenueNext20 := revenue20CandyBars priceNext20
  let totalRevenue := totalRevenue revenueFirst30 revenueNext20

  let priceNonDiscounted := 1.00
  let candyBarsNonDiscounted := 20
  let costNonDiscounted := costNonDiscountedBoxes candyBarsNonDiscounted priceNonDiscounted

  let priceDiscounted := 0.80
  let candyBarsDiscounted := 30
  let costDiscounted := costDiscountedBoxes candyBarsDiscounted priceDiscounted

  let totalCost := totalCost costNonDiscounted costDiscounted

  let taxRate := 0.07
  let salesTax := salesTax totalRevenue taxRate

  let fixedExpense := 15.0
  let totalExpenses := totalExpenses totalCost salesTax fixedExpense

  netProfit totalRevenue totalExpenses = 7.03 :=
by
  sorry

end james_net_profit_l581_58191


namespace yellow_balloons_ratio_l581_58109

theorem yellow_balloons_ratio 
  (total_balloons : ℕ) 
  (colors : ℕ) 
  (yellow_balloons_taken : ℕ) 
  (h_total_balloons : total_balloons = 672)
  (h_colors : colors = 4)
  (h_yellow_balloons_taken : yellow_balloons_taken = 84) :
  yellow_balloons_taken / (total_balloons / colors) = 1 / 2 :=
sorry

end yellow_balloons_ratio_l581_58109


namespace complement_of_M_is_correct_l581_58112

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 3}

-- Define the complement of M in U
def complement_M_in_U : Set ℝ := {x : ℝ | x < -1 ∨ x > 3}

-- State the theorem
theorem complement_of_M_is_correct : (U \ M) = complement_M_in_U := by sorry

end complement_of_M_is_correct_l581_58112


namespace find_line_equation_l581_58113
noncomputable def line_equation (l : ℝ → ℝ → Prop) : Prop :=
    (∀ x y : ℝ, l x y ↔ (2 * x + y - 4 = 0) ∨ (x + y - 3 = 0))

theorem find_line_equation (l : ℝ → ℝ → Prop) :
  (l 1 2) →
  (∃ x1 : ℝ, x1 > 0 ∧ ∃ y1 : ℝ, y1 > 0 ∧ l x1 0 ∧ l 0 y1) ∧
  (∃ x2 : ℝ, x2 < 0 ∧ ∃ y2 : ℝ, y2 > 0 ∧ l x2 0 ∧ l 0 y2) ∧
  (∃ x4 : ℝ, x4 > 0 ∧ ∃ y4 : ℝ, y4 < 0 ∧ l x4 0 ∧ l 0 y4) ∧
  (∃ x_int y_int : ℝ, l x_int 0 ∧ l 0 y_int ∧ x_int + y_int = 6) →
  (line_equation l) :=
by
  sorry

end find_line_equation_l581_58113


namespace root_of_function_is_four_l581_58182

noncomputable def f (x : ℝ) : ℝ := 2 - Real.log x / Real.log 2

theorem root_of_function_is_four (a : ℝ) (h : f a = 0) : a = 4 :=
by
  sorry

end root_of_function_is_four_l581_58182


namespace bottles_in_cups_l581_58119

-- Defining the given conditions
variables (BOTTLE GLASS CUP JUG : ℕ)

axiom h1 : JUG = BOTTLE + GLASS
axiom h2 : 2 * JUG = 7 * GLASS
axiom h3 : BOTTLE = CUP + 2 * GLASS

theorem bottles_in_cups : BOTTLE = 5 * CUP :=
sorry

end bottles_in_cups_l581_58119


namespace bob_average_speed_l581_58157

theorem bob_average_speed
  (lap_distance : ℕ) (lap1_time lap2_time lap3_time total_laps : ℕ)
  (h_lap_distance : lap_distance = 400)
  (h_lap1_time : lap1_time = 70)
  (h_lap2_time : lap2_time = 85)
  (h_lap3_time : lap3_time = 85)
  (h_total_laps : total_laps = 3) : 
  (lap_distance * total_laps) / (lap1_time + lap2_time + lap3_time) = 5 := by
    sorry

end bob_average_speed_l581_58157


namespace solve_for_b_l581_58189

theorem solve_for_b (b : ℚ) (h : b - b / 4 = 5 / 2) : b = 10 / 3 :=
by 
  sorry

end solve_for_b_l581_58189


namespace amy_total_soups_l581_58187

def chicken_soup := 6
def tomato_soup := 3
def vegetable_soup := 4
def clam_chowder := 2
def french_onion_soup := 1
def minestrone_soup := 5

theorem amy_total_soups : (chicken_soup + tomato_soup + vegetable_soup + clam_chowder + french_onion_soup + minestrone_soup) = 21 := by
  sorry

end amy_total_soups_l581_58187


namespace min_a_for_increasing_interval_l581_58149

def f (x a : ℝ) : ℝ := x^2 + (a - 2) * x - 1

theorem min_a_for_increasing_interval (a : ℝ) : (∀ x : ℝ, x ≥ 2 → f x a ≤ f (x + 1) a) ↔ a ≥ -2 :=
sorry

end min_a_for_increasing_interval_l581_58149


namespace circle_equation_l581_58193

theorem circle_equation : ∃ (x y : ℝ), (x - 2)^2 + y^2 = 2 :=
by
  sorry

end circle_equation_l581_58193


namespace parabola_vertex_l581_58164

theorem parabola_vertex (a b c : ℝ) :
  (∀ x, y = ax^2 + bx + c ↔ 
   y = a*((x+3)^2) + 4) ∧
   (∀ x y, (x, y) = ((1:ℝ), (2:ℝ))) →
   a + b + c = 3 := by
  sorry

end parabola_vertex_l581_58164


namespace cube_traversal_count_l581_58194

-- Defining the cube traversal problem
def cube_traversal (num_faces : ℕ) (adj_faces : ℕ) (visits : ℕ) : ℕ :=
  if (num_faces = 6 ∧ adj_faces = 4) then
    4 * 2
  else
    0

-- Theorem statement
theorem cube_traversal_count : 
  cube_traversal 6 4 1 = 8 :=
by
  -- Skipping the proof with sorry for now
  sorry

end cube_traversal_count_l581_58194


namespace minimum_value_a_plus_3b_plus_9c_l581_58176

open Real

theorem minimum_value_a_plus_3b_plus_9c (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 27) :
  a + 3 * b + 9 * c ≥ 27 :=
sorry

end minimum_value_a_plus_3b_plus_9c_l581_58176


namespace problem_l581_58152

theorem problem (a : ℤ) (ha : 0 ≤ a ∧ a < 13) (hdiv : (51 ^ 2016 + a) % 13 = 0) : a = 12 :=
sorry

end problem_l581_58152


namespace largest_both_writers_editors_l581_58144

-- Define the conditions
def writers : ℕ := 45
def editors_gt : ℕ := 38
def total_attendees : ℕ := 90
def both_writers_editors (x : ℕ) : ℕ := x
def neither_writers_editors (x : ℕ) : ℕ := x / 2

-- Define the main proof statement
theorem largest_both_writers_editors :
  ∃ x : ℕ, x ≤ 4 ∧
  (writers + (editors_gt + (0 : ℕ)) + neither_writers_editors x + both_writers_editors x = total_attendees) :=
sorry

end largest_both_writers_editors_l581_58144


namespace ratio_of_lost_diaries_to_total_diaries_l581_58175

theorem ratio_of_lost_diaries_to_total_diaries 
  (original_diaries : ℕ)
  (bought_diaries : ℕ)
  (current_diaries : ℕ)
  (h1 : original_diaries = 8)
  (h2 : bought_diaries = 2 * original_diaries)
  (h3 : current_diaries = 18) :
  (original_diaries + bought_diaries - current_diaries) / gcd (original_diaries + bought_diaries - current_diaries) (original_diaries + bought_diaries) 
  = 1 / 4 :=
by
  sorry

end ratio_of_lost_diaries_to_total_diaries_l581_58175


namespace total_brownies_l581_58103

theorem total_brownies (brought_to_school left_at_home : ℕ) (h1 : brought_to_school = 16) (h2 : left_at_home = 24) : 
  brought_to_school + left_at_home = 40 := 
by 
  sorry

end total_brownies_l581_58103


namespace car_late_speed_l581_58136

theorem car_late_speed :
  ∀ (d : ℝ) (t_on_time : ℝ) (t_late : ℝ) (v_on_time : ℝ) (v_late : ℝ),
  d = 225 →
  v_on_time = 60 →
  t_on_time = d / v_on_time →
  t_late = t_on_time + 0.75 →
  v_late = d / t_late →
  v_late = 50 :=
by
  intros d t_on_time t_late v_on_time v_late hd hv_on_time ht_on_time ht_late hv_late
  sorry

end car_late_speed_l581_58136


namespace union_of_sets_l581_58110

-- Define the sets and conditions
variables (a b : ℝ)
variables (A : Set ℝ) (B : Set ℝ)
variables (log2 : ℝ → ℝ)

-- State the assumptions and final proof goal
theorem union_of_sets (h_inter : A ∩ B = {2}) 
                      (h_A : A = {3, log2 a}) 
                      (h_B : B = {a, b}) 
                      (h_log2 : log2 4 = 2) :
  A ∪ B = {2, 3, 4} :=
by {
    sorry
}

end union_of_sets_l581_58110


namespace birds_initially_sitting_l581_58165

theorem birds_initially_sitting (initial_birds birds_joined total_birds : ℕ) 
  (h1 : birds_joined = 4) (h2 : total_birds = 6) (h3 : total_birds = initial_birds + birds_joined) : 
  initial_birds = 2 :=
by
  sorry

end birds_initially_sitting_l581_58165


namespace inequality_solution_l581_58190

theorem inequality_solution (x : ℝ) (h1 : 2 * x + 1 > x + 3) (h2 : 2 * x - 4 < x) : 2 < x ∧ x < 4 := sorry

end inequality_solution_l581_58190


namespace diagonals_diff_heptagon_octagon_l581_58138

-- Define the function to calculate the number of diagonals in a polygon with n sides
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem diagonals_diff_heptagon_octagon : 
  let A := num_diagonals 7
  let B := num_diagonals 8
  B - A = 6 :=
by
  sorry

end diagonals_diff_heptagon_octagon_l581_58138


namespace express_in_scientific_notation_l581_58104

theorem express_in_scientific_notation :
  ∃ (a : ℝ) (b : ℤ), 159600 = a * 10 ^ b ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.596 ∧ b = 5 :=
by
  sorry

end express_in_scientific_notation_l581_58104


namespace last_three_digits_7_pow_123_l581_58129

theorem last_three_digits_7_pow_123 : (7^123 % 1000) = 717 := sorry

end last_three_digits_7_pow_123_l581_58129


namespace find_x_l581_58171

variable {a b x : ℝ}

-- Defining the given conditions
def is_linear_and_unique_solution (a b : ℝ) : Prop :=
  3 * a + 2 * b = 0 ∧ a ≠ 0

-- The proof problem: prove that x = 1.5, given the conditions.
theorem find_x (ha : is_linear_and_unique_solution a b) : x = 1.5 :=
  sorry

end find_x_l581_58171


namespace pentagon_area_l581_58183

theorem pentagon_area (a b c d e : ℤ) (O : 31 * 25 = 775) (H : 12^2 + 5^2 = 13^2) 
  (rect_side_lengths : (a, b, c, d, e) = (13, 19, 20, 25, 31)) :
  775 - 1/2 * 12 * 5 = 745 := 
by
  sorry

end pentagon_area_l581_58183


namespace orchid_bushes_total_l581_58163

def current_bushes : ℕ := 47
def bushes_today : ℕ := 37
def bushes_tomorrow : ℕ := 25

theorem orchid_bushes_total : current_bushes + bushes_today + bushes_tomorrow = 109 := 
by sorry

end orchid_bushes_total_l581_58163


namespace find_fraction_l581_58170

variables (a b c : ℝ)
variables (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
variable (h : a + b + c = 1)

theorem find_fraction :
  (a^3 + b^3 + c^3) / (a * b * c) = (1 + 3 * (a - b)^2) / (a * b * (1 - a - b)) :=
by
  sorry

end find_fraction_l581_58170


namespace raisin_addition_l581_58199

theorem raisin_addition : 
  let yellow_raisins := 0.3
  let black_raisins := 0.4
  yellow_raisins + black_raisins = 0.7 := 
by
  sorry

end raisin_addition_l581_58199


namespace larger_value_3a_plus_1_l581_58184

theorem larger_value_3a_plus_1 {a : ℝ} (h : 8 * a^2 + 6 * a + 2 = 0) : 3 * a + 1 ≤ 3 * (-1/4 : ℝ) + 1 := 
sorry

end larger_value_3a_plus_1_l581_58184


namespace spider_paths_l581_58126

-- Define the grid points and the binomial coefficient calculation.
def grid_paths (n m : ℕ) : ℕ := Nat.choose (n + m) n

-- The problem statement
theorem spider_paths : grid_paths 4 3 = 35 := by
  sorry

end spider_paths_l581_58126


namespace number_of_integers_l581_58118

theorem number_of_integers (n : ℤ) : (200 < n ∧ n < 300 ∧ ∃ r : ℤ, n % 7 = r ∧ n % 9 = r) ↔ 
  n = 252 ∨ n = 253 ∨ n = 254 ∨ n = 255 ∨ n = 256 ∨ n = 257 ∨ n = 258 :=
by {
  sorry
}

end number_of_integers_l581_58118


namespace sum_series_eq_l581_58107

open BigOperators

theorem sum_series_eq : 
  ∑ n in Finset.range 256, (1 : ℝ) / ((2 * (n + 1 : ℕ) - 3) * (2 * (n + 1 : ℕ) + 1)) = -257 / 513 := 
by 
  sorry

end sum_series_eq_l581_58107


namespace oranges_for_profit_l581_58155

theorem oranges_for_profit (cost_buy: ℚ) (number_buy: ℚ) (cost_sell: ℚ) (number_sell: ℚ)
  (desired_profit: ℚ) (h₁: cost_buy / number_buy = 3.75) (h₂: cost_sell / number_sell = 4.5)
  (h₃: desired_profit = 120) :
  ∃ (oranges_to_sell: ℚ), oranges_to_sell = 160 ∧ (desired_profit / ((cost_sell / number_sell) - (cost_buy / number_buy))) = oranges_to_sell :=
by
  sorry

end oranges_for_profit_l581_58155


namespace arccos_neg_one_eq_pi_l581_58196

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi :=
  sorry

end arccos_neg_one_eq_pi_l581_58196


namespace journey_distance_l581_58197

theorem journey_distance :
  ∃ D : ℝ, (D / 42 + D / 48 = 10) ∧ D = 224 :=
by
  sorry

end journey_distance_l581_58197


namespace tenth_term_of_geometric_sequence_l581_58145

theorem tenth_term_of_geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) (tenth_term : ℚ) :
  a = 5 →
  r = 4 / 3 →
  n = 10 →
  tenth_term = a * r ^ (n - 1) →
  tenth_term = 1310720 / 19683 :=
by sorry

end tenth_term_of_geometric_sequence_l581_58145


namespace primer_cost_before_discount_l581_58174

theorem primer_cost_before_discount (primer_cost_after_discount : ℝ) (paint_cost : ℝ) (total_cost : ℝ) 
  (rooms : ℕ) (primer_discount : ℝ) (paint_cost_per_gallon : ℝ) :
  (primer_cost_after_discount = total_cost - (rooms * paint_cost_per_gallon)) →
  (rooms * (primer_cost - primer_discount * primer_cost) = primer_cost_after_discount) →
  primer_cost = 30 := by
  sorry

end primer_cost_before_discount_l581_58174


namespace square_root_25_pm5_l581_58132

-- Define that a number x satisfies the equation x^2 = 25
def square_root_of_25 (x : ℝ) : Prop := x * x = 25

-- The theorem states that the square root of 25 is ±5
theorem square_root_25_pm5 : ∀ x : ℝ, square_root_of_25 x ↔ x = 5 ∨ x = -5 :=
by
  intros x
  sorry

end square_root_25_pm5_l581_58132


namespace mark_notebooks_at_126_percent_l581_58162

variable (L : ℝ) (C : ℝ) (M : ℝ) (S : ℝ)

def merchant_condition1 := C = 0.85 * L
def merchant_condition2 := C = 0.75 * S
def merchant_condition3 := S = 0.9 * M

theorem mark_notebooks_at_126_percent :
    merchant_condition1 L C →
    merchant_condition2 C S →
    merchant_condition3 S M →
    M = 1.259 * L := by
  intros h1 h2 h3
  sorry

end mark_notebooks_at_126_percent_l581_58162


namespace no_nat_k_divides_7_l581_58134

theorem no_nat_k_divides_7 (k : ℕ) : ¬ 7 ∣ (2^(2*k - 1) + 2^k + 1) := 
sorry

end no_nat_k_divides_7_l581_58134


namespace sum_area_of_R_eq_20_l581_58181

noncomputable def sum_m_n : ℝ := 
  let s := 4 + 2 * Real.sqrt 2
  let total_area := s ^ 2
  let small_square_area := 4
  let given_rectangle_area := 4 * Real.sqrt 2
  let area_R := total_area - (small_square_area + given_rectangle_area)
  let m := 20
  let n := 12 * Real.sqrt 2
  m + n

theorem sum_area_of_R_eq_20 :
  let s := 4 + 2 * Real.sqrt 2
  let total_area := s ^ 2
  let small_square_area := 4
  let given_rectangle_area := 4 * Real.sqrt 2
  let area_R := total_area - (small_square_area + given_rectangle_area)
  area_R = 20 + 12 * Real.sqrt 2 :=
by
  sorry

end sum_area_of_R_eq_20_l581_58181


namespace billy_apples_ratio_l581_58125

theorem billy_apples_ratio :
  let monday := 2
  let tuesday := 2 * monday
  let wednesday := 9
  let friday := monday / 2
  let total_apples := 20
  let thursday := total_apples - (monday + tuesday + wednesday + friday)
  thursday / friday = 4 := 
by
  let monday := 2
  let tuesday := 2 * monday
  let wednesday := 9
  let friday := monday / 2
  let total_apples := 20
  let thursday := total_apples - (monday + tuesday + wednesday + friday)
  sorry

end billy_apples_ratio_l581_58125


namespace f_of_f_of_f_of_3_l581_58114

def f (x : ℕ) : ℕ := 
  if x > 9 then x - 1 
  else x ^ 3

theorem f_of_f_of_f_of_3 : f (f (f 3)) = 25 :=
by sorry

end f_of_f_of_f_of_3_l581_58114


namespace alan_has_5_20_cent_coins_l581_58179

theorem alan_has_5_20_cent_coins
  (a b c : ℕ)
  (h1 : a + b + c = 20)
  (h2 : ((400 - 15 * a - 10 * b) / 5) + 1 = 24) :
  c = 5 :=
by
  sorry

end alan_has_5_20_cent_coins_l581_58179


namespace dodgeball_tournament_l581_58156

theorem dodgeball_tournament (N : ℕ) (points : ℕ) :
  points = 1151 →
  (∀ {G : ℕ}, G = N * (N - 1) / 2 →
    (∃ (win_points loss_points tie_points : ℕ), 
      win_points = 15 * (N * (N - 1) / 2 - tie_points) ∧ 
      tie_points = 11 * tie_points ∧ 
      points = win_points + tie_points + loss_points)) → 
  N = 12 :=
by
  intro h_points h_games
  sorry

end dodgeball_tournament_l581_58156


namespace pqrs_product_l581_58173

noncomputable def P := (Real.sqrt 2007 + Real.sqrt 2008)
noncomputable def Q := (-Real.sqrt 2007 - Real.sqrt 2008)
noncomputable def R := (Real.sqrt 2007 - Real.sqrt 2008)
noncomputable def S := (-Real.sqrt 2008 + Real.sqrt 2007)

theorem pqrs_product : P * Q * R * S = -1 := by
  sorry

end pqrs_product_l581_58173


namespace option_C_is_quadratic_l581_58130

theorem option_C_is_quadratic : ∀ (x : ℝ), (x = x^2) ↔ (∃ (a b c : ℝ), a ≠ 0 ∧ a*x^2 + b*x + c = 0) := 
by
  sorry

end option_C_is_quadratic_l581_58130


namespace triangle_side_cube_l581_58124

theorem triangle_side_cube 
  (a b c : ℕ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_gcd : Nat.gcd a (Nat.gcd b c) = 1)
  (angle_condition : ∃ A B : ℝ, A = 3 * B) 
  : ∃ n m : ℕ, (a = n ^ 3 ∨ b = n ^ 3 ∨ c = n ^ 3) :=
sorry

end triangle_side_cube_l581_58124


namespace find_m_l581_58146

theorem find_m (m : ℕ) (h₁ : 0 < m) : 
  144^5 + 91^5 + 56^5 + 19^5 = m^5 → m = 147 := by
  -- Mathematically, we know the sum of powers equals a fifth power of 147
  -- 144^5 = 61917364224
  -- 91^5 = 6240321451
  -- 56^5 = 550731776
  -- 19^5 = 2476099
  -- => 61917364224 + 6240321451 + 550731776 + 2476099 = 68897423550
  -- Find the nearest  m such that m^5 = 68897423550
  sorry

end find_m_l581_58146


namespace anne_speed_l581_58131

-- Conditions
def time_hours : ℝ := 3
def distance_miles : ℝ := 6

-- Question with correct answer
theorem anne_speed : distance_miles / time_hours = 2 := by 
  sorry

end anne_speed_l581_58131


namespace plum_purchase_l581_58133

theorem plum_purchase
    (x : ℕ)
    (h1 : ∃ x, 5 * (6 * (4 * x) / 5) - 6 * ((5 * x) / 6) = -30) :
    2 * x = 60 := sorry

end plum_purchase_l581_58133


namespace shifted_quadratic_roots_l581_58167

theorem shifted_quadratic_roots {a h k : ℝ} (h_root_neg3 : a * (-3 + h) ^ 2 + k = 0)
                                 (h_root_2 : a * (2 + h) ^ 2 + k = 0) :
  (a * (-2 + h) ^ 2 + k = 0) ∧ (a * (3 + h) ^ 2 + k = 0) := by
  sorry

end shifted_quadratic_roots_l581_58167


namespace horner_evaluation_of_f_at_5_l581_58128

def f (x : ℝ) : ℝ := x^5 - 2*x^4 + x^3 + x^2 - x - 5

theorem horner_evaluation_of_f_at_5 : f 5 = 2015 :=
by sorry

end horner_evaluation_of_f_at_5_l581_58128


namespace linda_savings_fraction_l581_58111

theorem linda_savings_fraction (savings tv_cost : ℝ) (h1 : savings = 960) (h2 : tv_cost = 240) : (savings - tv_cost) / savings = 3 / 4 :=
by
  intros
  sorry

end linda_savings_fraction_l581_58111


namespace f_monotone_decreasing_without_min_value_l581_58139

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 + 1)

theorem f_monotone_decreasing_without_min_value :
  (∀ x y : ℝ, x < y → f y < f x) ∧ (∃ b : ℝ, ∀ x : ℝ, f x > b) :=
by
  sorry

end f_monotone_decreasing_without_min_value_l581_58139


namespace divided_scale_length_l581_58154

/-
  The problem definition states that we have a scale that is 6 feet 8 inches long, 
  and we need to prove that when the scale is divided into two equal parts, 
  each part is 3 feet 4 inches long.
-/

/-- Given length conditions in feet and inches --/
def total_length_feet : ℕ := 6
def total_length_inches : ℕ := 8

/-- Convert total length to inches --/
def total_length_in_inches := total_length_feet * 12 + total_length_inches

/-- Proof that if a scale is 6 feet 8 inches long and divided into 2 parts, each part is 3 feet 4 inches --/
theorem divided_scale_length :
  (total_length_in_inches / 2) = 40 ∧ (40 / 12 = 3 ∧ 40 % 12 = 4) :=
by
  sorry

end divided_scale_length_l581_58154


namespace charlies_mother_cookies_l581_58140

theorem charlies_mother_cookies 
    (charlie_cookies : ℕ) 
    (father_cookies : ℕ) 
    (total_cookies : ℕ)
    (h_charlie : charlie_cookies = 15)
    (h_father : father_cookies = 10)
    (h_total : total_cookies = 30) : 
    (total_cookies - charlie_cookies - father_cookies = 5) :=
by {
    sorry
}

end charlies_mother_cookies_l581_58140


namespace product_mod_32_l581_58161

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem product_mod_32 :
  M % 32 = 17 := by
  sorry

end product_mod_32_l581_58161


namespace workbook_arrangement_l581_58153

-- Define the condition of having different Korean and English workbooks
variables (K1 K2 : Type) (E1 E2 : Type)

-- The main theorem statement
theorem workbook_arrangement :
  ∃ (koreanWorkbooks englishWorkbooks : List (Type)), 
  (koreanWorkbooks.length = 2) ∧
  (englishWorkbooks.length = 2) ∧
  (∀ wb ∈ (koreanWorkbooks ++ englishWorkbooks), wb ≠ wb) ∧
  (∃ arrangements : Nat,
    arrangements = 12) :=
  sorry

end workbook_arrangement_l581_58153
