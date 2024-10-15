import Mathlib

namespace NUMINAMATH_GPT_john_paint_area_l1067_106788

noncomputable def area_to_paint (length width height openings : ℝ) : ℝ :=
  let wall_area := 2 * (length * height) + 2 * (width * height)
  let ceiling_area := length * width
  let total_area := wall_area + ceiling_area
  total_area - openings

theorem john_paint_area :
  let length := 15
  let width := 12
  let height := 10
  let openings := 70
  let bedrooms := 2
  2 * (area_to_paint length width height openings) = 1300 :=
by
  let length := 15
  let width := 12
  let height := 10
  let openings := 70
  let bedrooms := 2
  sorry

end NUMINAMATH_GPT_john_paint_area_l1067_106788


namespace NUMINAMATH_GPT_greatest_number_of_consecutive_integers_whose_sum_is_36_l1067_106701

/-- 
Given that the sum of N consecutive integers starting from a is 36, 
prove that the greatest possible value of N is 72.
-/
theorem greatest_number_of_consecutive_integers_whose_sum_is_36 :
  ∀ (N a : ℤ), (N > 0) → (N * (2 * a + N - 1)) = 72 → N ≤ 72 := 
by
  intros N a hN h
  sorry

end NUMINAMATH_GPT_greatest_number_of_consecutive_integers_whose_sum_is_36_l1067_106701


namespace NUMINAMATH_GPT_find_m_l1067_106709

def vector_parallel (a b : ℝ × ℝ) : Prop := 
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem find_m
  (m : ℝ)
  (a : ℝ × ℝ := (m, 1))
  (b : ℝ × ℝ := (2, -1))
  (h : vector_parallel a (b.1 - a.1, b.2 - a.2)) :
  m = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1067_106709


namespace NUMINAMATH_GPT_calc_result_l1067_106796

theorem calc_result : 
  let a := 82 + 3/5
  let b := 1/15
  let c := 3
  let d := 42 + 7/10
  (a / b) * c - d = 3674.3 :=
by
  sorry

end NUMINAMATH_GPT_calc_result_l1067_106796


namespace NUMINAMATH_GPT_cages_used_l1067_106792

theorem cages_used (total_puppies sold_puppies puppies_per_cage remaining_puppies needed_cages additional_cage total_cages: ℕ) 
  (h1 : total_puppies = 36) 
  (h2 : sold_puppies = 7) 
  (h3 : puppies_per_cage = 4) 
  (h4 : remaining_puppies = total_puppies - sold_puppies) 
  (h5 : needed_cages = remaining_puppies / puppies_per_cage) 
  (h6 : additional_cage = if (remaining_puppies % puppies_per_cage = 0) then 0 else 1) 
  (h7 : total_cages = needed_cages + additional_cage) : 
  total_cages = 8 := 
by 
  sorry

end NUMINAMATH_GPT_cages_used_l1067_106792


namespace NUMINAMATH_GPT_total_travel_time_l1067_106733

-- Defining the conditions
def car_travel_180_miles_in_4_hours : Prop :=
  180 / 4 = 45

def car_travel_135_miles_additional_time : Prop :=
  135 / 45 = 3

-- The main statement to be proved
theorem total_travel_time : car_travel_180_miles_in_4_hours ∧ car_travel_135_miles_additional_time → 4 + 3 = 7 := by
  sorry

end NUMINAMATH_GPT_total_travel_time_l1067_106733


namespace NUMINAMATH_GPT_general_term_formula_sum_of_sequence_l1067_106767

-- Define the arithmetic sequence {a_n}
def a (n : ℕ) : ℤ := n - 1

-- Conditions: a_5 = 4, a_3 + a_8 = 9
def cond1 : Prop := a 5 = 4
def cond2 : Prop := a 3 + a 8 = 9

theorem general_term_formula (h1 : cond1) (h2 : cond2) : ∀ n : ℕ, a n = n - 1 :=
by
  -- Place holder for proof
  sorry

-- Define the sequence {b_n}
def b (n : ℕ) : ℤ := 2 * a n - 1

-- Sum of the first n terms of b_n
def S (n : ℕ) : ℤ := n * (n - 2)

theorem sum_of_sequence (h1 : cond1) (h2 : cond2) : ∀ n : ℕ, (Finset.range (n + 1)).sum b = S n :=
by
  -- Place holder for proof
  sorry

end NUMINAMATH_GPT_general_term_formula_sum_of_sequence_l1067_106767


namespace NUMINAMATH_GPT_sum_of_interior_angles_octagon_l1067_106731

theorem sum_of_interior_angles_octagon : (8 - 2) * 180 = 1080 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_interior_angles_octagon_l1067_106731


namespace NUMINAMATH_GPT_apples_picked_l1067_106743

theorem apples_picked (n_a : ℕ) (k_a : ℕ) (total : ℕ) (m_a : ℕ) (h_n : n_a = 3) (h_k : k_a = 6) (h_t : total = 16) :
  m_a = total - (n_a + k_a) →
  m_a = 7 :=
by
  sorry

end NUMINAMATH_GPT_apples_picked_l1067_106743


namespace NUMINAMATH_GPT_maximum_additional_payment_expected_value_difference_l1067_106768

-- Add the conditions as definitions
def a1 : ℕ := 1298
def a2 : ℕ := 1347
def a3 : ℕ := 1337
def b1 : ℕ := 1402
def b2 : ℕ := 1310
def b3 : ℕ := 1298

-- Prices in rubles per kilowatt-hour
def peak_price : ℝ := 4.03
def night_price : ℝ := 1.01
def semi_peak_price : ℝ := 3.39

-- Actual consumptions in kilowatt-hour
def ΔP : ℝ := 104
def ΔN : ℝ := 37
def ΔSP : ℝ := 39

-- Correct payment calculated by the company
def correct_payment : ℝ := 660.72

-- Statements to prove
theorem maximum_additional_payment : 397.34 = (104 * 4.03 + 39 * 3.39 + 37 * 1.01 - 660.72) :=
by
  sorry

theorem expected_value_difference : 19.3 = ((5 * 1402 + 3 * 1347 + 1337 - 1298 - 3 * 1270 - 5 * 1214) / 15 * 8.43 - 660.72) :=
by
  sorry

end NUMINAMATH_GPT_maximum_additional_payment_expected_value_difference_l1067_106768


namespace NUMINAMATH_GPT_female_lion_weight_l1067_106708

theorem female_lion_weight (male_weight : ℚ) (weight_difference : ℚ) (female_weight : ℚ) : 
  male_weight = 145/4 → 
  weight_difference = 47/10 → 
  male_weight = female_weight + weight_difference → 
  female_weight = 631/20 :=
by
  intros h₁ h₂ h₃
  sorry

end NUMINAMATH_GPT_female_lion_weight_l1067_106708


namespace NUMINAMATH_GPT_regular_hexagon_interior_angle_measure_l1067_106711

theorem regular_hexagon_interior_angle_measure :
  let n := 6
  let sum_of_angles := (n - 2) * 180
  let measure_of_each_angle := sum_of_angles / n
  measure_of_each_angle = 120 :=
by
  sorry

end NUMINAMATH_GPT_regular_hexagon_interior_angle_measure_l1067_106711


namespace NUMINAMATH_GPT_dice_sum_is_4_l1067_106757

-- Defining the sum of points obtained from two dice rolls
def sum_of_dice (a b : ℕ) : ℕ := a + b

-- The main theorem stating the condition we need to prove
theorem dice_sum_is_4 (a b : ℕ) (h : sum_of_dice a b = 4) :
  (a = 3 ∧ b = 1) ∨ (a = 1 ∧ b = 3) ∨ (a = 2 ∧ b = 2) :=
sorry

end NUMINAMATH_GPT_dice_sum_is_4_l1067_106757


namespace NUMINAMATH_GPT_find_expression_value_l1067_106707

-- We declare our variables x and y
variables (x y : ℝ)

-- We state our conditions as hypotheses
def h1 : 3 * x + y = 5 := sorry
def h2 : x + 3 * y = 8 := sorry

-- We prove the given mathematical expression
theorem find_expression_value (h1 : 3 * x + y = 5) (h2 : x + 3 * y = 8) : 10 * x^2 + 19 * x * y + 10 * y^2 = 153 := 
by
  -- We intentionally skip the proof
  sorry

end NUMINAMATH_GPT_find_expression_value_l1067_106707


namespace NUMINAMATH_GPT_maxwell_distance_when_meeting_l1067_106794

variable (total_distance : ℝ := 50)
variable (maxwell_speed : ℝ := 4)
variable (brad_speed : ℝ := 6)
variable (t : ℝ := total_distance / (maxwell_speed + brad_speed))

theorem maxwell_distance_when_meeting :
  (maxwell_speed * t = 20) :=
by
  sorry

end NUMINAMATH_GPT_maxwell_distance_when_meeting_l1067_106794


namespace NUMINAMATH_GPT_arithmetic_sequence_solution_l1067_106752

variable {a : ℕ → ℤ}  -- assuming our sequence is integer-valued for simplicity

-- a is an arithmetic sequence if there exists a common difference d such that 
-- ∀ n, a_{n+1} = a_n + d
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- sum of the terms from a₁ to a₁₀₁₇ is equal to zero
def sum_condition (a : ℕ → ℤ) : Prop :=
  (Finset.range 2017).sum a = 0

theorem arithmetic_sequence_solution (a : ℕ → ℤ) (h_arith : is_arithmetic_sequence a) (h_sum : sum_condition a) :
  a 3 + a 2013 = 0 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_solution_l1067_106752


namespace NUMINAMATH_GPT_JordanRectangleWidth_l1067_106745

/-- Given that Carol's rectangle measures 15 inches by 24 inches,
and Jordan's rectangle is 8 inches long with equal area as Carol's rectangle,
prove that Jordan's rectangle is 45 inches wide. -/
theorem JordanRectangleWidth :
  ∃ W : ℝ, (15 * 24 = 8 * W) → W = 45 := by
  sorry

end NUMINAMATH_GPT_JordanRectangleWidth_l1067_106745


namespace NUMINAMATH_GPT_bathroom_area_l1067_106704

-- Definitions based on conditions
def totalHouseArea : ℝ := 1110
def numBedrooms : ℕ := 4
def bedroomArea : ℝ := 11 * 11
def kitchenArea : ℝ := 265
def numBathrooms : ℕ := 2

-- Mathematically equivalent proof problem
theorem bathroom_area :
  let livingArea := kitchenArea  -- living area is equal to kitchen area
  let totalRoomArea := numBedrooms * bedroomArea + kitchenArea + livingArea
  let remainingArea := totalHouseArea - totalRoomArea
  let bathroomArea := remainingArea / numBathrooms
  bathroomArea = 48 :=
by
  repeat { sorry }

end NUMINAMATH_GPT_bathroom_area_l1067_106704


namespace NUMINAMATH_GPT_min_value_frac_l1067_106782

theorem min_value_frac (x y : ℝ) (h1 : x > -1) (h2 : y > 0) (h3 : x + 2 * y = 2) : 
  ∃ L, (L = 3) ∧ (∀ (x y : ℝ), x > -1 → y > 0 → x + 2*y = 2 → 
  (∃ L, (L = 3) ∧ (∀ (x y : ℝ), x > -1 → y > 0 → x + 2*y = 2 → 
  ∀ (f : ℝ), f = (1 / (x + 1) + 2 / y) → f ≥ L))) :=
sorry

end NUMINAMATH_GPT_min_value_frac_l1067_106782


namespace NUMINAMATH_GPT_largest_root_of_equation_l1067_106703

theorem largest_root_of_equation : ∃ (x : ℝ), (x - 37)^2 - 169 = 0 ∧ ∀ y, (y - 37)^2 - 169 = 0 → y ≤ x :=
by
  sorry

end NUMINAMATH_GPT_largest_root_of_equation_l1067_106703


namespace NUMINAMATH_GPT_count_dracula_is_alive_l1067_106715

variable (P Q : Prop)
variable (h1 : P)          -- I am human
variable (h2 : P → Q)      -- If I am human, then Count Dracula is alive

theorem count_dracula_is_alive : Q :=
by
  sorry

end NUMINAMATH_GPT_count_dracula_is_alive_l1067_106715


namespace NUMINAMATH_GPT_find_a_l1067_106732

theorem find_a (a b c : ℂ) (ha : a.im = 0)
  (h1 : a + b + c = 5)
  (h2 : a * b + b * c + c * a = 8)
  (h3 : a * b * c = 4) :
  a = 1 ∨ a = 2 :=
sorry

end NUMINAMATH_GPT_find_a_l1067_106732


namespace NUMINAMATH_GPT_remaining_area_is_344_l1067_106738

def garden_length : ℕ := 20
def garden_width : ℕ := 18
def shed_side : ℕ := 4

def area_rectangle : ℕ := garden_length * garden_width
def area_shed : ℕ := shed_side * shed_side

def remaining_garden_area : ℕ := area_rectangle - area_shed

theorem remaining_area_is_344 : remaining_garden_area = 344 := by
  sorry

end NUMINAMATH_GPT_remaining_area_is_344_l1067_106738


namespace NUMINAMATH_GPT_hotel_R_greater_than_G_l1067_106781

variables (R G P : ℝ)

def hotel_charges_conditions :=
  P = 0.50 * R ∧ P = 0.80 * G

theorem hotel_R_greater_than_G :
  hotel_charges_conditions R G P → R = 1.60 * G :=
by
  sorry

end NUMINAMATH_GPT_hotel_R_greater_than_G_l1067_106781


namespace NUMINAMATH_GPT_sets_equal_l1067_106747

theorem sets_equal :
  let M := {x | x^2 - 2 * x + 1 = 0}
  let N := {1}
  M = N :=
by
  sorry

end NUMINAMATH_GPT_sets_equal_l1067_106747


namespace NUMINAMATH_GPT_smallest_positive_x_l1067_106750

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

theorem smallest_positive_x : ∃ x : ℕ, x > 0 ∧ is_palindrome (x + 6789) ∧ x = 218 := by
  sorry

end NUMINAMATH_GPT_smallest_positive_x_l1067_106750


namespace NUMINAMATH_GPT_gcd_three_numbers_l1067_106754

theorem gcd_three_numbers (a b c : ℕ) (h1 : a = 72) (h2 : b = 120) (h3 : c = 168) :
  Nat.gcd (Nat.gcd a b) c = 24 :=
by
  rw [h1, h2, h3]
  exact sorry

end NUMINAMATH_GPT_gcd_three_numbers_l1067_106754


namespace NUMINAMATH_GPT_ellipsoid_volume_div_pi_l1067_106734

noncomputable def ellipsoid_projection_min_area : ℝ := 9 * Real.pi
noncomputable def ellipsoid_projection_max_area : ℝ := 25 * Real.pi
noncomputable def ellipsoid_circle_projection_area : ℝ := 16 * Real.pi
noncomputable def ellipsoid_volume (a b c : ℝ) : ℝ := (4/3) * Real.pi * a * b * c

theorem ellipsoid_volume_div_pi (a b c : ℝ)
  (h_min : (a * b = 9))
  (h_max : (b * c = 25))
  (h_circle : (b = 4)) :
  ellipsoid_volume a b c / Real.pi = 75 := 
  by
    sorry

end NUMINAMATH_GPT_ellipsoid_volume_div_pi_l1067_106734


namespace NUMINAMATH_GPT_rachel_assembly_time_l1067_106751

theorem rachel_assembly_time :
  let chairs := 20
  let tables := 8
  let bookshelves := 5
  let time_per_chair := 6
  let time_per_table := 8
  let time_per_bookshelf := 12
  let total_chairs_time := chairs * time_per_chair
  let total_tables_time := tables * time_per_table
  let total_bookshelves_time := bookshelves * time_per_bookshelf
  total_chairs_time + total_tables_time + total_bookshelves_time = 244 := by
  sorry

end NUMINAMATH_GPT_rachel_assembly_time_l1067_106751


namespace NUMINAMATH_GPT_remainder_n_sq_plus_3n_5_mod_25_l1067_106749

theorem remainder_n_sq_plus_3n_5_mod_25 (k : ℤ) (n : ℤ) (h : n = 25 * k - 1) : 
  (n^2 + 3 * n + 5) % 25 = 3 := 
by
  sorry

end NUMINAMATH_GPT_remainder_n_sq_plus_3n_5_mod_25_l1067_106749


namespace NUMINAMATH_GPT_students_in_section_B_l1067_106741

variable (x : ℕ)

/-- There are 30 students in section A and the number of students in section B is x. The 
    average weight of section A is 40 kg, and the average weight of section B is 35 kg. 
    The average weight of the whole class is 38 kg. Prove that the number of students in
    section B is 20. -/
theorem students_in_section_B (h : 30 * 40 + x * 35 = 38 * (30 + x)) : x = 20 :=
  sorry

end NUMINAMATH_GPT_students_in_section_B_l1067_106741


namespace NUMINAMATH_GPT_total_cost_of_backpack_and_pencil_case_l1067_106735

-- Definitions based on the given conditions
def pencil_case_price : ℕ := 8
def backpack_price : ℕ := 5 * pencil_case_price

-- Statement of the proof problem
theorem total_cost_of_backpack_and_pencil_case : 
  pencil_case_price + backpack_price = 48 :=
by
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_total_cost_of_backpack_and_pencil_case_l1067_106735


namespace NUMINAMATH_GPT_fraction_simplification_l1067_106702

theorem fraction_simplification :
  8 * (15 / 11) * (-25 / 40) = -15 / 11 :=
by
  sorry

end NUMINAMATH_GPT_fraction_simplification_l1067_106702


namespace NUMINAMATH_GPT_tank_capacity_l1067_106713

theorem tank_capacity (x : ℝ) (h : 0.50 * x = 75) : x = 150 :=
by sorry

end NUMINAMATH_GPT_tank_capacity_l1067_106713


namespace NUMINAMATH_GPT_am_gm_inequality_l1067_106777

theorem am_gm_inequality (a1 a2 a3 : ℝ) (h₀ : 0 < a1) (h₁ : 0 < a2) (h₂ : 0 < a3) (h₃ : a1 + a2 + a3 = 1) : 
  1 / a1 + 1 / a2 + 1 / a3 ≥ 9 :=
by
  sorry

end NUMINAMATH_GPT_am_gm_inequality_l1067_106777


namespace NUMINAMATH_GPT_trapezoid_area_is_correct_l1067_106785

noncomputable def trapezoid_area (base_short : ℝ) (angle_adj : ℝ) (angle_diag : ℝ) : ℝ :=
  let width := 2 * base_short -- calculated width from angle_adj
  let height := base_short / Real.tan (angle_adj / 2 * Real.pi / 180)
  (base_short + base_short + width) * height / 2

theorem trapezoid_area_is_correct :
  trapezoid_area 2 135 150 = 2 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_area_is_correct_l1067_106785


namespace NUMINAMATH_GPT_simplify_powers_l1067_106710

-- Defining the multiplicative rule for powers
def power_mul (x : ℕ) (a b : ℕ) : ℕ := x^(a+b)

-- Proving that x^5 * x^6 = x^11
theorem simplify_powers (x : ℕ) : x^5 * x^6 = x^11 :=
by
  change x^5 * x^6 = x^(5 + 6)
  sorry

end NUMINAMATH_GPT_simplify_powers_l1067_106710


namespace NUMINAMATH_GPT_games_attended_l1067_106722

theorem games_attended (games_this_month games_last_month games_next_month total_games : ℕ) 
  (h1 : games_this_month = 11) 
  (h2 : games_last_month = 17) 
  (h3 : games_next_month = 16) : 
  total_games = games_this_month + games_last_month + games_next_month → 
  total_games = 44 :=
by
  sorry

end NUMINAMATH_GPT_games_attended_l1067_106722


namespace NUMINAMATH_GPT_unique_ordered_pairs_satisfying_equation_l1067_106762

theorem unique_ordered_pairs_satisfying_equation :
  ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x^6 * y^6 - 19 * x^3 * y^3 + 18 = 0 ↔ (x, y) = (1, 1) ∧
  (∀ x y : ℕ, 0 < x ∧ 0 < y ∧ x^6 * y^6 - 19 * x^3 * y^3 + 18 = 0 → (x, y) = (1, 1)) :=
by
  sorry

end NUMINAMATH_GPT_unique_ordered_pairs_satisfying_equation_l1067_106762


namespace NUMINAMATH_GPT_shaded_region_area_l1067_106789

/-- A rectangle measuring 12cm by 8cm has four semicircles drawn with their diameters as the sides
of the rectangle. Prove that the area of the shaded region inside the rectangle but outside
the semicircles is equal to 96 - 52π (cm²). --/
theorem shaded_region_area (A : ℝ) (π : ℝ) (hA : A = 96 - 52 * π) : 
  ∀ (length width r1 r2 : ℝ) (hl : length = 12) (hw : width = 8) 
  (hr1 : r1 = length / 2) (hr2 : r2 = width / 2),
  (length * width) - (2 * (1/2 * π * r1^2 + 1/2 * π * r2^2)) = A := 
by 
  sorry

end NUMINAMATH_GPT_shaded_region_area_l1067_106789


namespace NUMINAMATH_GPT_calculate_expression_l1067_106730

variable (f g : ℝ → ℝ)

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)
def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g x = - g (-x)

theorem calculate_expression 
  (hf : is_even_function f)
  (hg : is_odd_function g)
  (hfg : ∀ x : ℝ, f x - g x = x ^ 3 + x ^ 2 + 1) :
  f 1 + g 1 = 1 :=
  sorry

end NUMINAMATH_GPT_calculate_expression_l1067_106730


namespace NUMINAMATH_GPT_first_player_winning_strategy_l1067_106728

-- Defining the type for the positions on the chessboard
structure Position where
  x : Nat
  y : Nat
  deriving DecidableEq

-- Initial position C1
def C1 : Position := ⟨3, 1⟩

-- Winning position H8
def H8 : Position := ⟨8, 8⟩

-- Function to check if a position is a winning position
-- the target winning position is H8
def isWinningPosition (p : Position) : Bool :=
  p = H8

-- Function to determine the next possible positions
-- from the current position based on the allowed moves
def nextPositions (p : Position) : List Position :=
  (List.range (8 - p.x)).map (λ dx => ⟨p.x + dx + 1, p.y⟩) ++
  (List.range (8 - p.y)).map (λ dy => ⟨p.x, p.y + dy + 1⟩) ++
  (List.range (min (8 - p.x) (8 - p.y))).map (λ d => ⟨p.x + d + 1, p.y + d + 1⟩)

-- Statement of the problem: First player has a winning strategy from C1
theorem first_player_winning_strategy : 
  ∃ move : Position, move ∈ nextPositions C1 ∧
  ∀ next_move : Position, next_move ∈ nextPositions move → isWinningPosition next_move :=
sorry

end NUMINAMATH_GPT_first_player_winning_strategy_l1067_106728


namespace NUMINAMATH_GPT_duration_of_each_movie_l1067_106720

-- define the conditions
def num_screens : ℕ := 6
def hours_open : ℕ := 8
def num_movies : ℕ := 24

-- define the total screening time
def total_screening_time : ℕ := num_screens * hours_open

-- define the expected duration of each movie
def movie_duration : ℕ := total_screening_time / num_movies

-- state the theorem
theorem duration_of_each_movie : movie_duration = 2 := by sorry

end NUMINAMATH_GPT_duration_of_each_movie_l1067_106720


namespace NUMINAMATH_GPT_triangle_condition_l1067_106759

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sin x) * (Real.cos x) + (Real.sqrt 3) * (Real.cos x) ^ 2 - (Real.sqrt 3) / 2

theorem triangle_condition (a b c : ℝ) (h : b^2 + c^2 = a^2 + Real.sqrt 3 * b * c) : 
  f (Real.pi / 6) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_GPT_triangle_condition_l1067_106759


namespace NUMINAMATH_GPT_cubic_roots_geometric_progression_l1067_106739

theorem cubic_roots_geometric_progression 
  (a r : ℝ)
  (h_roots: 27 * a^3 * r^3 - 81 * a^2 * r^2 + 63 * a * r - 14 = 0)
  (h_sum: a + a * r + a * r^2 = 3)
  (h_product: a^3 * r^3 = 14 / 27)
  : (max (a^2) ((a * r^2)^2) - min (a^2) ((a * r^2)^2) = 5 / 3) := 
sorry

end NUMINAMATH_GPT_cubic_roots_geometric_progression_l1067_106739


namespace NUMINAMATH_GPT_correct_operation_l1067_106776

theorem correct_operation (x y : ℝ) : (x^3 * y^2 - y^2 * x^3 = 0) :=
by sorry

end NUMINAMATH_GPT_correct_operation_l1067_106776


namespace NUMINAMATH_GPT_arithmetic_sequence_a4_l1067_106723

theorem arithmetic_sequence_a4 (a1 : ℤ) (S3 : ℤ) (h1 : a1 = 3) (h2 : S3 = 15) : 
  ∃ (a4 : ℤ), a4 = 9 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a4_l1067_106723


namespace NUMINAMATH_GPT_action_figure_total_l1067_106742

variable (initial_figures : ℕ) (added_figures : ℕ)

theorem action_figure_total (h₁ : initial_figures = 8) (h₂ : added_figures = 2) : (initial_figures + added_figures) = 10 := by
  sorry

end NUMINAMATH_GPT_action_figure_total_l1067_106742


namespace NUMINAMATH_GPT_intersection_condition_sufficient_but_not_necessary_l1067_106765

theorem intersection_condition_sufficient_but_not_necessary (k : ℝ) :
  (-Real.sqrt 3 / 3 < k ∧ k < Real.sqrt 3 / 3) →
  ((∃ x : ℝ, (k^2 + 1) * x^2 + (2 * k^2 - 2) * x + k^2 = 0) ∧ 
   ¬ (∃ k, (∃ x : ℝ, (k^2 + 1) * x^2 + (2 * k^2 - 2) * x + k^2 = 0) → 
   (¬ (-Real.sqrt 3 / 3 < k ∧ k < Real.sqrt 3 / 3)))) :=
sorry

end NUMINAMATH_GPT_intersection_condition_sufficient_but_not_necessary_l1067_106765


namespace NUMINAMATH_GPT_f_odd_f_periodic_f_def_on_interval_problem_solution_l1067_106744

noncomputable def f : ℝ → ℝ := 
sorry

theorem f_odd (x : ℝ) : f (-x) = -f x := 
sorry

theorem f_periodic (x : ℝ) : f (x + 4) = f x := 
sorry

theorem f_def_on_interval (x : ℝ) (h : -2 < x ∧ x < 0) : f x = 2 ^ x :=
sorry

theorem problem_solution : f 2015 - f 2014 = 1 / 2 :=
sorry

end NUMINAMATH_GPT_f_odd_f_periodic_f_def_on_interval_problem_solution_l1067_106744


namespace NUMINAMATH_GPT_product_of_four_consecutive_integers_divisible_by_12_l1067_106716

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end NUMINAMATH_GPT_product_of_four_consecutive_integers_divisible_by_12_l1067_106716


namespace NUMINAMATH_GPT_cooking_time_remaining_l1067_106727

def time_to_cook_remaining (n_total n_cooked t_per : ℕ) : ℕ := (n_total - n_cooked) * t_per

theorem cooking_time_remaining :
  ∀ (n_total n_cooked t_per : ℕ), n_total = 13 → n_cooked = 5 → t_per = 6 → time_to_cook_remaining n_total n_cooked t_per = 48 :=
by
  intros n_total n_cooked t_per h1 h2 h3
  rw [h1, h2, h3]
  exact rfl

end NUMINAMATH_GPT_cooking_time_remaining_l1067_106727


namespace NUMINAMATH_GPT_find_number_l1067_106717

theorem find_number {x : ℤ} (h : x + 5 = 6) : x = 1 :=
sorry

end NUMINAMATH_GPT_find_number_l1067_106717


namespace NUMINAMATH_GPT_green_function_solution_l1067_106798

noncomputable def G (x ξ : ℝ) (α : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ ξ then α + Real.log ξ else if ξ ≤ x ∧ x ≤ 1 then α + Real.log x else 0

theorem green_function_solution (x ξ α : ℝ) (hα : α ≠ 0) (hx_bound : 0 < x ∧ x ≤ 1) :
  ( G x ξ α = if 0 < x ∧ x ≤ ξ then α + Real.log ξ else if ξ ≤ x ∧ x ≤ 1 then α + Real.log x else 0 ) :=
sorry

end NUMINAMATH_GPT_green_function_solution_l1067_106798


namespace NUMINAMATH_GPT_find_cost_of_jersey_l1067_106761

def cost_of_jersey (J : ℝ) : Prop := 
  let shorts_cost := 15.20
  let socks_cost := 6.80
  let total_players := 16
  let total_cost := 752
  total_players * (J + shorts_cost + socks_cost) = total_cost

theorem find_cost_of_jersey : cost_of_jersey 25 :=
  sorry

end NUMINAMATH_GPT_find_cost_of_jersey_l1067_106761


namespace NUMINAMATH_GPT_present_age_of_eldest_is_45_l1067_106763

theorem present_age_of_eldest_is_45 (x : ℕ) 
  (h1 : (5 * x - 10) + (7 * x - 10) + (8 * x - 10) + (9 * x - 10) = 107) :
  9 * x = 45 :=
sorry

end NUMINAMATH_GPT_present_age_of_eldest_is_45_l1067_106763


namespace NUMINAMATH_GPT_find_c_l1067_106779

noncomputable def cubic_function (x : ℝ) (c : ℝ) : ℝ :=
  x^3 - 3 * x + c

theorem find_c (c : ℝ) :
  (∃ x₁ x₂ : ℝ, cubic_function x₁ c = 0 ∧ cubic_function x₂ c = 0 ∧ x₁ ≠ x₂) →
  (c = -2 ∨ c = 2) :=
by
  sorry

end NUMINAMATH_GPT_find_c_l1067_106779


namespace NUMINAMATH_GPT_students_selected_milk_l1067_106736

noncomputable def selected_soda_percent : ℚ := 50 / 100
noncomputable def selected_milk_percent : ℚ := 30 / 100
noncomputable def selected_soda_count : ℕ := 90
noncomputable def selected_milk_count := selected_milk_percent / selected_soda_percent * selected_soda_count

theorem students_selected_milk :
    selected_milk_count = 54 :=
by
  sorry

end NUMINAMATH_GPT_students_selected_milk_l1067_106736


namespace NUMINAMATH_GPT_centroids_coincide_l1067_106726

noncomputable def centroid (A B C : ℂ) : ℂ :=
  (A + B + C) / 3

theorem centroids_coincide (A B C : ℂ) (k : ℝ) (C1 A1 B1 : ℂ)
  (h1 : C1 = k * (B - A) + A)
  (h2 : A1 = k * (C - B) + B)
  (h3 : B1 = k * (A - C) + C) :
  centroid A1 B1 C1 = centroid A B C := by
  sorry

end NUMINAMATH_GPT_centroids_coincide_l1067_106726


namespace NUMINAMATH_GPT_A_and_B_together_finish_in_ten_days_l1067_106724

-- Definitions of conditions
def B_daily_work := 1 / 15
def A_daily_work := B_daily_work / 2
def combined_daily_work := A_daily_work + B_daily_work

-- The theorem to be proved
theorem A_and_B_together_finish_in_ten_days : 1 / combined_daily_work = 10 := 
  by 
    sorry

end NUMINAMATH_GPT_A_and_B_together_finish_in_ten_days_l1067_106724


namespace NUMINAMATH_GPT_trapezoid_area_l1067_106770

theorem trapezoid_area (EF GH EG FH : ℝ) (h : ℝ)
  (h_EF : EF = 60) (h_GH : GH = 30) (h_EG : EG = 25) (h_FH : FH = 18) (h_alt : h = 15) :
  (1 / 2 * (EF + GH) * h) = 675 :=
by
  rw [h_EF, h_GH, h_alt]
  sorry

end NUMINAMATH_GPT_trapezoid_area_l1067_106770


namespace NUMINAMATH_GPT_machines_used_l1067_106799

variable (R S : ℕ)

/-- 
  A company has two types of machines, type R and type S. 
  Operating at a constant rate, a machine of type R does a certain job in 36 hours, 
  and a machine of type S does the job in 9 hours. 
  If the company used the same number of each type of machine to do the job in 12 hours, 
  then the company used 15 machines of type R.
-/
theorem machines_used (hR : ∀ ⦃n⦄, n * (1 / 36) + n * (1 / 9) = (1 / 12)) :
  R = 15 := 
by 
  sorry

end NUMINAMATH_GPT_machines_used_l1067_106799


namespace NUMINAMATH_GPT_odd_base_divisibility_by_2_base_divisibility_by_m_l1067_106719

-- Part (a)
theorem odd_base_divisibility_by_2 (q : ℕ) :
  (∀ a : ℕ, (a * q) % 2 = 0 ↔ a % 2 = 0) → q % 2 = 1 := 
sorry

-- Part (b)
theorem base_divisibility_by_m (q m : ℕ) (h1 : m > 1) :
  (∀ a : ℕ, (a * q) % m = 0 ↔ a % m = 0) → ∃ k : ℕ, q = 1 + m * k ∧ k ≥ 1 :=
sorry

end NUMINAMATH_GPT_odd_base_divisibility_by_2_base_divisibility_by_m_l1067_106719


namespace NUMINAMATH_GPT_olivia_bags_count_l1067_106769

def cans_per_bag : ℕ := 5
def total_cans : ℕ := 20

theorem olivia_bags_count : total_cans / cans_per_bag = 4 := by
  sorry

end NUMINAMATH_GPT_olivia_bags_count_l1067_106769


namespace NUMINAMATH_GPT_find_f_zero_l1067_106714

variable (f : ℝ → ℝ)

def odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (x + 1) = -g (-x + 1)

def even_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (x - 1) = g (-x - 1)

theorem find_f_zero
  (H1 : odd_function f)
  (H2 : even_function f)
  (H3 : f 4 = 6) :
  f 0 = -6 := by
  sorry

end NUMINAMATH_GPT_find_f_zero_l1067_106714


namespace NUMINAMATH_GPT_largest_c_such_that_neg5_in_range_l1067_106784

theorem largest_c_such_that_neg5_in_range :
  ∃ (c : ℝ), (∀ x : ℝ, x^2 + 5 * x + c = -5) → c = 5 / 4 :=
sorry

end NUMINAMATH_GPT_largest_c_such_that_neg5_in_range_l1067_106784


namespace NUMINAMATH_GPT_unique_function_property_l1067_106756

def f (n : Nat) : Nat := sorry

theorem unique_function_property :
  (∀ x y : ℕ+, x < y → f x < f y) ∧
  (∀ y x : ℕ+, f (y * f x) = x^2 * f (x * y)) →
  ∀ n : ℕ+, f n = n^2 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_unique_function_property_l1067_106756


namespace NUMINAMATH_GPT_madeline_biked_more_l1067_106725

def madeline_speed : ℕ := 12
def madeline_time : ℕ := 3
def max_speed : ℕ := 15
def max_time : ℕ := 2

theorem madeline_biked_more : (madeline_speed * madeline_time) - (max_speed * max_time) = 6 := 
by 
  sorry

end NUMINAMATH_GPT_madeline_biked_more_l1067_106725


namespace NUMINAMATH_GPT_find_g_of_2_l1067_106760

open Real

noncomputable def g : ℝ → ℝ := sorry

theorem find_g_of_2
  (H: ∀ x : ℝ, g (2 ^ x) + x * g (2 ^ (-x)) + x = 1) : g 2 = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_g_of_2_l1067_106760


namespace NUMINAMATH_GPT_solve_for_y_l1067_106721

theorem solve_for_y : ∀ (y : ℚ), 2 * y + 3 * y = 500 - (4 * y + 6 * y) → y = 100 / 3 := by
  intros y h
  sorry

end NUMINAMATH_GPT_solve_for_y_l1067_106721


namespace NUMINAMATH_GPT_sellingPrice_is_459_l1067_106790

-- Definitions based on conditions
def costPrice : ℝ := 540
def markupPercentage : ℝ := 0.15
def discountPercentage : ℝ := 0.2608695652173913

-- Calculating the marked price based on the given conditions
def markedPrice (cp : ℝ) (markup : ℝ) : ℝ := cp + (markup * cp)

-- Calculating the discount amount based on the marked price and the discount percentage
def discount (mp : ℝ) (discountPct : ℝ) : ℝ := discountPct * mp

-- Calculating the selling price
def sellingPrice (mp : ℝ) (discountAmt : ℝ) : ℝ := mp - discountAmt

-- Stating the final proof problem
theorem sellingPrice_is_459 :
  sellingPrice (markedPrice costPrice markupPercentage) (discount (markedPrice costPrice markupPercentage) discountPercentage) = 459 :=
by
  sorry

end NUMINAMATH_GPT_sellingPrice_is_459_l1067_106790


namespace NUMINAMATH_GPT_gcd_lcm_product_24_36_l1067_106778

-- Definitions for gcd, lcm, and product for given numbers, skipping proof with sorry
theorem gcd_lcm_product_24_36 : Nat.gcd 24 36 * Nat.lcm 24 36 = 864 := by
  -- Sorry used to skip proof
  sorry

end NUMINAMATH_GPT_gcd_lcm_product_24_36_l1067_106778


namespace NUMINAMATH_GPT_julia_more_kids_on_monday_l1067_106786

-- Definition of the problem statement
def playedWithOnMonday : ℕ := 6
def playedWithOnTuesday : ℕ := 5
def difference := playedWithOnMonday - playedWithOnTuesday

theorem julia_more_kids_on_monday : difference = 1 :=
by
  -- Proof can be filled out here.
  sorry

end NUMINAMATH_GPT_julia_more_kids_on_monday_l1067_106786


namespace NUMINAMATH_GPT_no_adjacent_same_roll_probability_l1067_106706

noncomputable def probability_no_adjacent_same_roll : ℚ :=
  (1331 / 1728)

theorem no_adjacent_same_roll_probability :
  (probability_no_adjacent_same_roll = (1331 / 1728)) :=
by
  sorry

end NUMINAMATH_GPT_no_adjacent_same_roll_probability_l1067_106706


namespace NUMINAMATH_GPT_negation_of_proposition_p_l1067_106771

def has_real_root (m : ℝ) : Prop := ∃ x : ℝ, x^2 + m * x + 1 = 0

def negation_of_p : Prop := ∀ m : ℝ, ¬ has_real_root m

theorem negation_of_proposition_p : negation_of_p :=
by sorry

end NUMINAMATH_GPT_negation_of_proposition_p_l1067_106771


namespace NUMINAMATH_GPT_minimum_value_quadratic_l1067_106755

noncomputable def quadratic (x : ℝ) : ℝ := x^2 - 4 * x + 5

theorem minimum_value_quadratic :
  ∀ x : ℝ, quadratic x ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_quadratic_l1067_106755


namespace NUMINAMATH_GPT_product_of_two_digit_numbers_5488_has_smaller_number_56_l1067_106712

theorem product_of_two_digit_numbers_5488_has_smaller_number_56 (a b : ℕ) (h_a2 : 10 ≤ a) (h_a3 : a < 100) (h_b2 : 10 ≤ b) (h_b3 : b < 100) (h_prod : a * b = 5488) : a = 56 ∨ b = 56 :=
by {
  sorry
}

end NUMINAMATH_GPT_product_of_two_digit_numbers_5488_has_smaller_number_56_l1067_106712


namespace NUMINAMATH_GPT_horse_problem_l1067_106729

-- Definitions based on conditions:
def total_horses : ℕ := 100
def tiles_pulled_by_big_horse (x : ℕ) : ℕ := 3 * x
def tiles_pulled_by_small_horses (x : ℕ) : ℕ := (100 - x) / 3

-- The statement to prove:
theorem horse_problem (x : ℕ) : 
    tiles_pulled_by_big_horse x + tiles_pulled_by_small_horses x = 100 :=
sorry

end NUMINAMATH_GPT_horse_problem_l1067_106729


namespace NUMINAMATH_GPT_function_even_l1067_106774

theorem function_even (n : ℤ) (h : 30 ∣ n)
    (h_prop: (1 : ℝ)^n^2 + (-1: ℝ)^n^2 = 2 * ((1: ℝ)^n + (-1: ℝ)^n - 1)) :
    ∀ x : ℝ, (x^n = (-x)^n) :=
by
    sorry

end NUMINAMATH_GPT_function_even_l1067_106774


namespace NUMINAMATH_GPT_min_C2_minus_D2_is_36_l1067_106791

noncomputable def find_min_C2_minus_D2 (x y z : ℝ) : ℝ :=
  (Real.sqrt (x + 3) + Real.sqrt (y + 6) + Real.sqrt (z + 11))^2 -
  (Real.sqrt (x + 2) + Real.sqrt (y + 2) + Real.sqrt (z + 2))^2

theorem min_C2_minus_D2_is_36 : ∀ (x y z : ℝ), 0 ≤ x → 0 ≤ y → 0 ≤ z → 
  find_min_C2_minus_D2 x y z ≥ 36 :=
by
  intros x y z hx hy hz
  sorry

end NUMINAMATH_GPT_min_C2_minus_D2_is_36_l1067_106791


namespace NUMINAMATH_GPT_cos_double_angle_l1067_106772

theorem cos_double_angle 
  (α β : ℝ) 
  (h1 : Real.sin (α - β) = 1/3) 
  (h2 : Real.cos α * Real.sin β = 1/6) :
  Real.cos (2 * α + 2 * β) = 1/9 :=
sorry

end NUMINAMATH_GPT_cos_double_angle_l1067_106772


namespace NUMINAMATH_GPT_ratio_volume_sphere_to_hemisphere_l1067_106780

noncomputable def volume_sphere (r : ℝ) : ℝ :=
  (4/3) * Real.pi * r^3

noncomputable def volume_hemisphere (r : ℝ) : ℝ :=
  (1/2) * volume_sphere r

theorem ratio_volume_sphere_to_hemisphere (p : ℝ) (hp : 0 < p) :
  (volume_sphere p) / (volume_hemisphere (2 * p)) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_volume_sphere_to_hemisphere_l1067_106780


namespace NUMINAMATH_GPT_rake_yard_alone_time_l1067_106773

-- Definitions for the conditions
def brother_time := 45 -- Brother takes 45 minutes
def together_time := 18 -- Together it takes 18 minutes

-- Define and prove the time it takes you to rake the yard alone based on given conditions
theorem rake_yard_alone_time : 
  ∃ (x : ℕ), (1 / (x : ℚ) + 1 / (brother_time : ℚ) = 1 / (together_time : ℚ)) ∧ x = 30 :=
by
  sorry

end NUMINAMATH_GPT_rake_yard_alone_time_l1067_106773


namespace NUMINAMATH_GPT_heather_bicycling_time_l1067_106775

theorem heather_bicycling_time (distance speed : ℕ) (h1 : distance = 96) (h2 : speed = 6) : 
(distance / speed) = 16 := by
  sorry

end NUMINAMATH_GPT_heather_bicycling_time_l1067_106775


namespace NUMINAMATH_GPT_infinite_primes_dividing_expression_l1067_106740

theorem infinite_primes_dividing_expression (k : ℕ) (hk : k > 0) : 
  ∃ᶠ p in Filter.atTop, ∃ n : ℕ, p ∣ (2017^n + k) :=
sorry

end NUMINAMATH_GPT_infinite_primes_dividing_expression_l1067_106740


namespace NUMINAMATH_GPT_arithmetic_value_l1067_106787

theorem arithmetic_value : (8 * 4) + 3 = 35 := by
  sorry

end NUMINAMATH_GPT_arithmetic_value_l1067_106787


namespace NUMINAMATH_GPT_trent_walks_to_bus_stop_l1067_106783

theorem trent_walks_to_bus_stop (x : ℕ) (h1 : 2 * (x + 7) = 22) : x = 4 :=
sorry

end NUMINAMATH_GPT_trent_walks_to_bus_stop_l1067_106783


namespace NUMINAMATH_GPT_smallest_of_seven_even_numbers_l1067_106793

theorem smallest_of_seven_even_numbers (a b c d e f g : ℕ) 
  (h1 : a % 2 = 0) 
  (h2 : b = a + 2) 
  (h3 : c = a + 4) 
  (h4 : d = a + 6) 
  (h5 : e = a + 8) 
  (h6 : f = a + 10) 
  (h7 : g = a + 12) 
  (h_sum : a + b + c + d + e + f + g = 700) : 
  a = 94 :=
by sorry

end NUMINAMATH_GPT_smallest_of_seven_even_numbers_l1067_106793


namespace NUMINAMATH_GPT_intersection_A_B_l1067_106753

-- Define the sets A and B
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {x | x > 0}

-- The theorem we want to prove
theorem intersection_A_B : A ∩ B = {1} := 
by {
  sorry
}

end NUMINAMATH_GPT_intersection_A_B_l1067_106753


namespace NUMINAMATH_GPT_beginning_of_spring_period_and_day_l1067_106748

noncomputable def daysBetween : Nat := 46 -- Total days: Dec 21, 2004 to Feb 4, 2005

theorem beginning_of_spring_period_and_day :
  let total_days := daysBetween
  let segment := total_days / 9
  let day_within_segment := total_days % 9
  segment = 5 ∧ day_within_segment = 1 := by
sorry

end NUMINAMATH_GPT_beginning_of_spring_period_and_day_l1067_106748


namespace NUMINAMATH_GPT_max_non_overlapping_squares_l1067_106758

theorem max_non_overlapping_squares (m n : ℕ) : 
  ∃ max_squares : ℕ, max_squares = m :=
by
  sorry

end NUMINAMATH_GPT_max_non_overlapping_squares_l1067_106758


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l1067_106764

variable (x y : ℝ)
variable (h1 : x = 1)
variable (h2 : y = Real.sqrt 2)

theorem simplify_and_evaluate_expression : 
  (x + 2 * y) ^ 2 - x * (x + 4 * y) + (1 - y) * (1 + y) = 7 := by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l1067_106764


namespace NUMINAMATH_GPT_area_of_triangle_MAB_l1067_106766

noncomputable def triangle_area (A B M : ℝ × ℝ) : ℝ :=
  0.5 * ((B.1 - A.1) * (M.2 - A.2) - (M.1 - A.1) * (B.2 - A.2))

theorem area_of_triangle_MAB :
  let C1 (p : ℝ × ℝ) := p.1^2 - p.2^2 = 2
  let C2 (p : ℝ × ℝ) := ∃ θ, p.1 = 2 + 2 * Real.cos θ ∧ p.2 = 2 * Real.sin θ
  let M := (3.0, 0.0)
  let A := (2, 2 * Real.sin (Real.pi / 6))
  let B := (2 * Real.sqrt 3, 2 * Real.sin (Real.pi / 6))
  triangle_area A B M = (3 * Real.sqrt 3 - 3) / 2 :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_MAB_l1067_106766


namespace NUMINAMATH_GPT_inverse_proportion_points_l1067_106737

theorem inverse_proportion_points (x1 x2 x3 : ℝ) :
  (10 / x1 = -5) →
  (10 / x2 = 2) →
  (10 / x3 = 5) →
  x1 < x3 ∧ x3 < x2 :=
by sorry

end NUMINAMATH_GPT_inverse_proportion_points_l1067_106737


namespace NUMINAMATH_GPT_sufficient_and_necessary_condition_l1067_106746

theorem sufficient_and_necessary_condition (x : ℝ) :
  (x - 2) * (x + 2) > 0 ↔ x > 2 ∨ x < -2 :=
by sorry

end NUMINAMATH_GPT_sufficient_and_necessary_condition_l1067_106746


namespace NUMINAMATH_GPT_bigger_number_l1067_106797

theorem bigger_number (yoongi : ℕ) (jungkook : ℕ) (h1 : yoongi = 4) (h2 : jungkook = 6 + 3) : jungkook > yoongi :=
by
  sorry

end NUMINAMATH_GPT_bigger_number_l1067_106797


namespace NUMINAMATH_GPT_asymptotes_of_hyperbola_l1067_106718

theorem asymptotes_of_hyperbola : 
  ∀ (x y : ℝ), 9 * y^2 - 25 * x^2 = 169 → (y = (5/3) * x ∨ y = -(5/3) * x) :=
by 
  sorry

end NUMINAMATH_GPT_asymptotes_of_hyperbola_l1067_106718


namespace NUMINAMATH_GPT_find_width_l1067_106700

variable (L W : ℕ)

def perimeter (L W : ℕ) : ℕ := 2 * L + 2 * W

theorem find_width (h1 : perimeter L W = 46) (h2 : W = L + 7) : W = 15 :=
sorry

end NUMINAMATH_GPT_find_width_l1067_106700


namespace NUMINAMATH_GPT_inequality_subtract_l1067_106795

-- Definitions of the main variables and conditions
variables {a b : ℝ}
-- Condition that should hold
axiom h : a > b

-- Expected conclusion
theorem inequality_subtract : a - 1 > b - 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_subtract_l1067_106795


namespace NUMINAMATH_GPT_simplify_and_evaluate_evaluate_at_zero_l1067_106705

noncomputable def simplified_expression (x : ℤ) : ℚ :=
  (1 - 1/(x-1)) / ((x^2 - 4*x + 4) / (x^2 - 1))

theorem simplify_and_evaluate (x : ℤ) (h : x ≠ 1 ∧ x ≠ 2 ∧ x ≠ -1) : 
  simplified_expression x = (x+1)/(x-2) :=
by
  sorry

theorem evaluate_at_zero : simplified_expression 0 = -1/2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_evaluate_at_zero_l1067_106705
