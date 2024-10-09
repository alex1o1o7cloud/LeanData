import Mathlib

namespace solve_x_l0_86

theorem solve_x (x : ℝ) (h : 2 - 2 / (1 - x) = 2 / (1 - x)) : x = -2 := 
by
  sorry

end solve_x_l0_86


namespace max_area_triangle_l0_72

noncomputable def max_area (QA QB QC BC : ℝ) : ℝ :=
  1 / 2 * ((QA^2 + QB^2 - QC^2) / (2 * BC) + 3) * BC

theorem max_area_triangle (QA QB QC BC : ℝ) (hQA : QA = 3) (hQB : QB = 4) (hQC : QC = 5) (hBC : BC = 6) :
  max_area QA QB QC BC = 19 := by
  sorry

end max_area_triangle_l0_72


namespace inverse_of_parallel_lines_l0_49

theorem inverse_of_parallel_lines 
  (P Q : Prop) 
  (parallel_impl_alt_angles : P → Q) :
  (Q → P) := 
by
  sorry

end inverse_of_parallel_lines_l0_49


namespace triangular_square_is_triangular_l0_50

-- Definition of a triangular number
def is_triang_number (n : ℕ) : Prop :=
  ∃ x : ℕ, n = x * (x + 1) / 2

-- The main theorem statement
theorem triangular_square_is_triangular :
  ∃ x : ℕ, 
    is_triang_number x ∧ 
    is_triang_number (x * x) :=
sorry

end triangular_square_is_triangular_l0_50


namespace four_leaved_clovers_percentage_l0_82

noncomputable def percentage_of_four_leaved_clovers (clovers total_clovers purple_four_leaved_clovers : ℕ ) : ℝ := 
  (purple_four_leaved_clovers * 4 * 100) / total_clovers 

theorem four_leaved_clovers_percentage :
  percentage_of_four_leaved_clovers 500 500 25 = 20 := 
by
  -- application of conditions and arithmetic simplification.
  sorry

end four_leaved_clovers_percentage_l0_82


namespace generate_13121_not_generate_12131_l0_20

theorem generate_13121 : ∃ n m : ℕ, 13121 + 1 = 2^n * 3^m := by
  sorry

theorem not_generate_12131 : ¬∃ n m : ℕ, 12131 + 1 = 2^n * 3^m := by
  sorry

end generate_13121_not_generate_12131_l0_20


namespace contrapositive_proposition_l0_41

theorem contrapositive_proposition (x : ℝ) : (x ≤ -1 ∨ x ≥ 1) → (x^2 ≥ 1) :=
by
  sorry

end contrapositive_proposition_l0_41


namespace parabola_intersects_x_axis_l0_23

-- Define the quadratic equation
def quadratic (x m : ℝ) : ℝ := x^2 + 2 * x + m - 1

-- Define the discriminant of the quadratic equation
def discriminant (m : ℝ) : ℝ := 4 - 4 * (m - 1)

-- Lean statement to prove the range of m
theorem parabola_intersects_x_axis (m : ℝ) : (∃ x : ℝ, quadratic x m = 0) ↔ m ≤ 2 := by
  sorry

end parabola_intersects_x_axis_l0_23


namespace num_candidates_l0_48

theorem num_candidates (n : ℕ) (h : n * (n - 1) = 30) : n = 6 :=
sorry

end num_candidates_l0_48


namespace cord_lengths_l0_37

noncomputable def cordLengthFirstDog (distance : ℝ) : ℝ :=
  distance / 2

noncomputable def cordLengthSecondDog (distance : ℝ) : ℝ :=
  distance / 2

noncomputable def cordLengthThirdDog (radius : ℝ) : ℝ :=
  radius

theorem cord_lengths (d1 d2 r : ℝ) (h1 : d1 = 30) (h2 : d2 = 40) (h3 : r = 20) :
  cordLengthFirstDog d1 = 15 ∧ cordLengthSecondDog d2 = 20 ∧ cordLengthThirdDog r = 20 := by
  sorry

end cord_lengths_l0_37


namespace Jasmine_shoe_size_l0_32

theorem Jasmine_shoe_size (J A : ℕ) (h1 : A = 2 * J) (h2 : J + A = 21) : J = 7 :=
by 
  sorry

end Jasmine_shoe_size_l0_32


namespace geometric_sequence_sum_l0_73

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ)
  (h1 : a 3 + a 5 = 20)
  (h2 : a 4 = 8)
  (h3 : ∀ n, a (n + 1) = a n * q) :
  a 2 + a 6 = 34 := 
sorry

end geometric_sequence_sum_l0_73


namespace goldfish_graph_finite_set_of_points_l0_27

-- Define the cost function for goldfish including the setup fee
def cost (n : ℕ) : ℝ := 20 * n + 5

-- Define the condition
def n_values := {n : ℕ | 1 ≤ n ∧ n ≤ 12}

-- The Lean statement to prove the nature of the graph
theorem goldfish_graph_finite_set_of_points :
  ∀ n ∈ n_values, ∃ k : ℝ, (k = cost n) :=
by
  sorry

end goldfish_graph_finite_set_of_points_l0_27


namespace solve_system_of_equations_l0_96

theorem solve_system_of_equations :
  ∃ x y : ℝ, 
  (4 * x - 3 * y = -0.5) ∧ 
  (5 * x + 7 * y = 10.3) ∧ 
  (|x - 0.6372| < 1e-4) ∧ 
  (|y - 1.0163| < 1e-4) :=
by
  sorry

end solve_system_of_equations_l0_96


namespace necessary_but_not_sufficient_condition_for_geometric_sequence_l0_7

theorem necessary_but_not_sufficient_condition_for_geometric_sequence
  (a b c : ℝ) :
  (∃ (r : ℝ), a = r * b ∧ b = r * c) → (b^2 = a * c) ∧ ¬((b^2 = a * c) → (∃ (r : ℝ), a = r * b ∧ b = r * c)) := 
by
  sorry

end necessary_but_not_sufficient_condition_for_geometric_sequence_l0_7


namespace scientific_notation_400000000_l0_13

theorem scientific_notation_400000000 : 400000000 = 4 * 10^8 :=
by
  sorry

end scientific_notation_400000000_l0_13


namespace problem1_problem2_problem3_l0_18

-- Define the functions f and g
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c
def g (a b : ℝ) (x : ℝ) : ℝ := a * x + b

-- Problem statements in Lean
theorem problem1 (a b c : ℝ) (h : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |f a b c x| ≤ 1) : |c| ≤ 1 :=
sorry

theorem problem2 (a b c : ℝ) (h₁ : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |f a b c x| ≤ 1) : 
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |g a b x| ≤ 2 :=
sorry

theorem problem3 (a b c : ℝ) (ha : a > 0) (hx : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → g a b x ≤ 2) (hf : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |f a b c x| ≤ 1) :
  ∃ x, -1 ≤ x ∧ x ≤ 1 ∧ g a b x = 2 :=
sorry

end problem1_problem2_problem3_l0_18


namespace sectionBSeats_l0_83

-- Definitions from the conditions
def seatsIn60SeatSubsectionA : Nat := 60
def subsectionsIn80SeatA : Nat := 3
def seatsPer80SeatSubsectionA : Nat := 80
def extraSeatsInSectionB : Nat := 20

-- Total seats in 80-seat subsections of Section A
def totalSeatsIn80SeatSubsections : Nat := subsectionsIn80SeatA * seatsPer80SeatSubsectionA

-- Total seats in Section A
def totalSeatsInSectionA : Nat := totalSeatsIn80SeatSubsections + seatsIn60SeatSubsectionA

-- Total seats in Section B
def totalSeatsInSectionB : Nat := 3 * totalSeatsInSectionA + extraSeatsInSectionB

-- The statement to prove
theorem sectionBSeats : totalSeatsInSectionB = 920 := by
  sorry

end sectionBSeats_l0_83


namespace find_natural_numbers_with_integer_roots_l0_9

theorem find_natural_numbers_with_integer_roots :
  ∃ (p q : ℕ), 
    (∀ x : ℤ, x * x - (p * q) * x + (p + q) = 0 → ∃ (x1 x2 : ℤ), x = x1 ∧ x = x2 ∧ x1 + x2 = p * q ∧ x1 * x2 = p + q) ↔
    ((p = 1 ∧ q = 5) ∨ (p = 5 ∧ q = 1) ∨ (p = 2 ∧ q = 2) ∨ (p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) :=
by
-- proof skipped
sorry

end find_natural_numbers_with_integer_roots_l0_9


namespace not_equal_factorial_l0_44

noncomputable def permutations (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem not_equal_factorial (n : ℕ) :
  permutations (n + 1) n ≠ (by apply Nat.factorial n) := by
  sorry

end not_equal_factorial_l0_44


namespace part1_part2_part3_l0_94

noncomputable def a (n : ℕ) : ℝ := 
if n = 1 then 1 else 
if n = 2 then 3/2 else 
if n = 3 then 5/4 else 
sorry

noncomputable def S (n : ℕ) : ℝ := sorry

axiom recurrence {n : ℕ} (h : n ≥ 2) : 4 * S (n + 2) + 5 * S n = 8 * S (n + 1) + S (n - 1)

-- Part 1
theorem part1 : a 4 = 7 / 8 :=
sorry

-- Part 2
theorem part2 : ∃ (r : ℝ) (b : ℕ → ℝ), (r = 1/2) ∧ (∀ n ≥ 1, a (n + 1) - r * a n = b n) :=
sorry

-- Part 3
theorem part3 : ∀ n, a n = (2 * n - 1) / 2^(n - 1) :=
sorry

end part1_part2_part3_l0_94


namespace wings_count_total_l0_4

def number_of_wings (num_planes : Nat) (wings_per_plane : Nat) : Nat :=
  num_planes * wings_per_plane

theorem wings_count_total :
  number_of_wings 45 2 = 90 :=
  by
    sorry

end wings_count_total_l0_4


namespace converse_of_posImpPosSquare_l0_34

-- Let's define the condition proposition first
def posImpPosSquare (x : ℝ) : Prop := x > 0 → x^2 > 0

-- Now, we state the converse we need to prove
theorem converse_of_posImpPosSquare (x : ℝ) (h : posImpPosSquare x) : x^2 > 0 → x > 0 := sorry

end converse_of_posImpPosSquare_l0_34


namespace maximal_q_for_broken_line_l0_99

theorem maximal_q_for_broken_line :
  ∃ q : ℝ, (∀ i : ℕ, 0 ≤ i → i < 5 → ∀ A_i : ℝ, (A_i = q ^ i)) ∧ 
  (q = (1 + Real.sqrt 5) / 2) := sorry

end maximal_q_for_broken_line_l0_99


namespace fraction_problem_l0_36

theorem fraction_problem (a : ℕ) (h1 : (a:ℚ)/(a + 27) = 865/1000) : a = 173 := 
by
  sorry

end fraction_problem_l0_36


namespace positive_difference_for_6_points_l0_1

def combinations (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def positiveDifferenceTrianglesAndQuadrilaterals (n : ℕ) : ℕ :=
  combinations n 3 - combinations n 4

theorem positive_difference_for_6_points : positiveDifferenceTrianglesAndQuadrilaterals 6 = 5 :=
by
  sorry

end positive_difference_for_6_points_l0_1


namespace charlie_coins_l0_26

variables (a c : ℕ)

axiom condition1 : c + 2 = 5 * (a - 2)
axiom condition2 : c - 2 = 4 * (a + 2)

theorem charlie_coins : c = 98 :=
by {
    sorry
}

end charlie_coins_l0_26


namespace color_crafter_secret_codes_l0_76

theorem color_crafter_secret_codes :
  8^5 = 32768 := by
  sorry

end color_crafter_secret_codes_l0_76


namespace sinC_calculation_maxArea_calculation_l0_71

noncomputable def sinC_given_sides_and_angles (A B C a b c : ℝ) (h1 : 2 * Real.sin A = a * Real.cos B) (h2 : b = Real.sqrt 5) (h3 : c = 2) : ℝ :=
  Real.sin C

theorem sinC_calculation 
  (A B C a b c : ℝ) 
  (h1 : 2 * Real.sin A = a * Real.cos B)
  (h2 : b = Real.sqrt 5)
  (h3 : c = 2) 
  (h4 : Real.sin B = Real.sqrt 5 / 3) : 
  sinC_given_sides_and_angles A B C a b c h1 h2 h3 = 2 / 3 := by sorry

noncomputable def maxArea_given_sides_and_angles (A B C a b c : ℝ) (h1 : 2 * Real.sin A = a * Real.cos B) (h2 : b = Real.sqrt 5) (h3 : c = 2) : ℝ :=
  1 / 2 * a * c * Real.sin B

theorem maxArea_calculation 
  (A B C a b c : ℝ) 
  (h1 : 2 * Real.sin A = a * Real.cos B)
  (h2 : b = Real.sqrt 5)
  (h3 : c = 2)
  (h4 : Real.sin B = Real.sqrt 5 / 3) 
  (h5 : a * c ≤ 15 / 2) : 
  maxArea_given_sides_and_angles A B C a b c h1 h2 h3 = 5 * Real.sqrt 5 / 4 := by sorry

end sinC_calculation_maxArea_calculation_l0_71


namespace catFinishesOnMondayNextWeek_l0_29

def morningConsumptionDaily (day : String) : ℚ := if day = "Wednesday" then 1 / 3 else 1 / 4
def eveningConsumptionDaily : ℚ := 1 / 6

def totalDailyConsumption (day : String) : ℚ :=
  morningConsumptionDaily day + eveningConsumptionDaily

-- List of days in order
def week : List String := ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

-- Total food available initially
def totalInitialFood : ℚ := 8

-- Function to calculate total food consumed until a given day
def foodConsumedUntil (day : String) : ℚ :=
  week.takeWhile (· != day) |>.foldl (λ acc d => acc + totalDailyConsumption d) 0

-- Function to determine the day when 8 cans are completely consumed
def finishingDay : String :=
  match week.find? (λ day => foodConsumedUntil day + totalDailyConsumption day = totalInitialFood) with
  | some day => day
  | none => "Monday"  -- If no exact match is found in the first week, it is Monday of the next week

theorem catFinishesOnMondayNextWeek :
  finishingDay = "Monday" := by
  sorry

end catFinishesOnMondayNextWeek_l0_29


namespace truck_travel_distance_l0_53

theorem truck_travel_distance (miles_per_5gallons miles distance gallons rate : ℕ)
  (h1 : miles_per_5gallons = 150) 
  (h2 : gallons = 5) 
  (h3 : rate = miles_per_5gallons / gallons) 
  (h4 : gallons = 7) 
  (h5 : distance = rate * gallons) : 
  distance = 210 := 
by sorry

end truck_travel_distance_l0_53


namespace sin_120_eq_sqrt3_div_2_l0_10

theorem sin_120_eq_sqrt3_div_2 : Real.sin (120 * Real.pi / 180) = Real.sqrt 3 / 2 :=
sorry

end sin_120_eq_sqrt3_div_2_l0_10


namespace phone_plan_cost_equal_at_2500_l0_11

-- We define the costs C1 and C2 as described in the problem conditions.
def C1 (x : ℕ) : ℝ :=
  if x <= 500 then 50 else 50 + 0.35 * (x - 500)

def C2 (x : ℕ) : ℝ :=
  if x <= 1000 then 75 else 75 + 0.45 * (x - 1000)

-- We need to prove that the costs are equal when x = 2500.
theorem phone_plan_cost_equal_at_2500 : C1 2500 = C2 2500 := by
  sorry

end phone_plan_cost_equal_at_2500_l0_11


namespace min_value_of_2x_plus_y_l0_6

theorem min_value_of_2x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x^2 + 2*x*y - 3 = 0) :
  2*x + y ≥ 3 :=
sorry

end min_value_of_2x_plus_y_l0_6


namespace frustum_lateral_surface_area_l0_3

theorem frustum_lateral_surface_area (r1 r2 h : ℝ) (r1_eq : r1 = 10) (r2_eq : r2 = 4) (h_eq : h = 6) :
  let s := Real.sqrt (h^2 + (r1 - r2)^2)
  let A := Real.pi * (r1 + r2) * s
  A = 84 * Real.pi * Real.sqrt 2 :=
by
  sorry

end frustum_lateral_surface_area_l0_3


namespace initial_number_of_men_l0_19

theorem initial_number_of_men (W : ℝ) (M : ℝ) (h1 : (M * 15) = W / 2) (h2 : ((M - 2) * 25) = W / 2) : M = 5 :=
sorry

end initial_number_of_men_l0_19


namespace cuboid_area_correct_l0_15

def cuboid_surface_area (length breadth height : ℕ) :=
  2 * (length * height) + 2 * (breadth * height) + 2 * (length * breadth)

theorem cuboid_area_correct : cuboid_surface_area 4 6 5 = 148 := by
  sorry

end cuboid_area_correct_l0_15


namespace total_heads_eq_fifteen_l0_93

-- Definitions for types of passengers and their attributes
def cats_heads : Nat := 7
def cats_legs : Nat := 7 * 4
def total_legs : Nat := 43
def captain_heads : Nat := 1
def captain_legs : Nat := 1

noncomputable def crew_heads (C : Nat) : Nat := C
noncomputable def crew_legs (C : Nat) : Nat := 2 * C

theorem total_heads_eq_fifteen : 
  ∃ (C : Nat),
    cats_legs + crew_legs C + captain_legs = total_legs ∧
    cats_heads + crew_heads C + captain_heads = 15 :=
by
  sorry

end total_heads_eq_fifteen_l0_93


namespace none_of_these_valid_l0_5

variables {x y z w u v : ℝ}

def statement_1 (x y z w : ℝ) := x > y → z < w
def statement_2 (z w u v : ℝ) := z > w → u < v

theorem none_of_these_valid (h₁ : statement_1 x y z w) (h₂ : statement_2 z w u v) :
  ¬ ( (x < y → u < v) ∨ (u < v → x < y) ∨ (u > v → x > y) ∨ (x > y → u > v) ) :=
by {
  sorry
}

end none_of_these_valid_l0_5


namespace largest_angle_of_pentagon_l0_67

theorem largest_angle_of_pentagon (a d : ℝ) (h1 : a = 100) (h2 : d = 2) :
  let angle1 := a
  let angle2 := a + d
  let angle3 := a + 2 * d
  let angle4 := a + 3 * d
  let angle5 := a + 4 * d
  angle1 + angle2 + angle3 + angle4 + angle5 = 540 ∧ angle5 = 116 :=
by
  sorry

end largest_angle_of_pentagon_l0_67


namespace find_h_l0_89

theorem find_h (j k h : ℕ) (h₁ : 2013 = 3 * h^2 + j) (h₂ : 2014 = 2 * h^2 + k)
  (pos_int_x_intercepts_1 : ∃ x1 x2 : ℕ, x1 ≠ x2 ∧ x1 > 0 ∧ x2 > 0 ∧ (3 * (x1 - h)^2 + j = 0 ∧ 3 * (x2 - h)^2 + j = 0))
  (pos_int_x_intercepts_2 : ∃ y1 y2 : ℕ, y1 ≠ y2 ∧ y1 > 0 ∧ y2 > 0 ∧ (2 * (y1 - h)^2 + k = 0 ∧ 2 * (y2 - h)^2 + k = 0)):
  h = 36 :=
by
  sorry

end find_h_l0_89


namespace problem_statement_l0_80

noncomputable def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
noncomputable def B (m : ℝ) : Set ℝ := {x | x^2 - m*x + 2 = 0}

theorem problem_statement (m : ℝ) : (A ∩ (B m) = B m) → (m = 3 ∨ (-2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2)) :=
by
  sorry

end problem_statement_l0_80


namespace shape_of_r_eq_c_in_cylindrical_coords_l0_69

variable {c : ℝ}

theorem shape_of_r_eq_c_in_cylindrical_coords (h : c > 0) :
  ∀ (r θ z : ℝ), (r = c) ↔ ∃ (cylinder : ℝ), cylinder = r ∧ cylinder = c :=
by
  sorry

end shape_of_r_eq_c_in_cylindrical_coords_l0_69


namespace geoff_election_l0_74

theorem geoff_election (Votes: ℝ) (Percent: ℝ) (ExtraVotes: ℝ) (x: ℝ) 
  (h1 : Votes = 6000) 
  (h2 : Percent = 1) 
  (h3 : ExtraVotes = 3000) 
  (h4 : ReceivedVotes = (Percent / 100) * Votes) 
  (h5 : TotalVotesNeeded = ReceivedVotes + ExtraVotes) 
  (h6 : x = (TotalVotesNeeded / Votes) * 100) :
  x = 51 := 
  by 
    sorry

end geoff_election_l0_74


namespace games_needed_in_single_elimination_l0_92

theorem games_needed_in_single_elimination (teams : ℕ) (h : teams = 23) : 
  ∃ games : ℕ, games = teams - 1 ∧ games = 22 :=
by
  existsi (teams - 1)
  sorry

end games_needed_in_single_elimination_l0_92


namespace four_digit_palindrome_perfect_squares_l0_90

theorem four_digit_palindrome_perfect_squares : 
  ∃ (count : ℕ), count = 2 ∧ 
  (∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 → 
            ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 
            n = 1001 * a + 110 * b ∧ 
            ∃ k : ℕ, k * k = n) → count = 2 := by
  sorry

end four_digit_palindrome_perfect_squares_l0_90


namespace equilateral_triangle_black_area_l0_62

theorem equilateral_triangle_black_area :
  let initial_black_area := 1
  let change_fraction := 5/6 * 9/10
  let area_after_n_changes (n : Nat) : ℚ := initial_black_area * (change_fraction ^ n)
  area_after_n_changes 3 = 27/64 := 
by
  sorry

end equilateral_triangle_black_area_l0_62


namespace least_number_of_cans_l0_54

theorem least_number_of_cans 
  (Maaza_volume : ℕ) (Pepsi_volume : ℕ) (Sprite_volume : ℕ) 
  (h1 : Maaza_volume = 80) (h2 : Pepsi_volume = 144) (h3 : Sprite_volume = 368) :
  (Maaza_volume / Nat.gcd (Nat.gcd Maaza_volume Pepsi_volume) Sprite_volume) +
  (Pepsi_volume / Nat.gcd (Nat.gcd Maaza_volume Pepsi_volume) Sprite_volume) +
  (Sprite_volume / Nat.gcd (Nat.gcd Maaza_volume Pepsi_volume) Sprite_volume) = 37 := by
  sorry

end least_number_of_cans_l0_54


namespace solution_set_inequality_l0_77

theorem solution_set_inequality (x : ℝ) : (x^2-2*x-3)*(x^2+1) < 0 ↔ -1 < x ∧ x < 3 :=
by
  sorry

end solution_set_inequality_l0_77


namespace exists_plane_intersecting_in_parallel_lines_l0_55

variables {Point Line Plane : Type}
variables (a : Line) (S₁ S₂ : Plane)

-- Definitions and assumptions
def intersects_in (a : Line) (P : Plane) : Prop := sorry
def parallel_lines (l₁ l₂ : Line) : Prop := sorry

-- Proof problem statement
theorem exists_plane_intersecting_in_parallel_lines :
  ∃ P : Plane, intersects_in a P ∧
    (∀ l₁ l₂ : Line, (intersects_in l₁ S₁ ∧ intersects_in l₂ S₂ ∧ l₁ = l₂)
                     → parallel_lines l₁ l₂) :=
sorry

end exists_plane_intersecting_in_parallel_lines_l0_55


namespace absolute_value_of_neg_eight_l0_25

/-- Absolute value of a number is the distance from 0 on the number line. -/
def absolute_value (x : ℤ) : ℤ :=
  if x >= 0 then x else -x

theorem absolute_value_of_neg_eight : absolute_value (-8) = 8 := by
  -- Proof is omitted
  sorry

end absolute_value_of_neg_eight_l0_25


namespace number_of_human_family_members_l0_33

-- Definitions for the problem
def num_birds := 4
def num_dogs := 3
def num_cats := 18
def bird_feet := 2
def dog_feet := 4
def cat_feet := 4
def human_feet := 2
def human_heads := 1

def animal_feet := (num_birds * bird_feet) + (num_dogs * dog_feet) + (num_cats * cat_feet)
def animal_heads := num_birds + num_dogs + num_cats

def total_feet (H : Nat) := animal_feet + (H * human_feet)
def total_heads (H : Nat) := animal_heads + (H * human_heads)

-- The problem statement translated to Lean
theorem number_of_human_family_members (H : Nat) : (total_feet H) = (total_heads H) + 74 → H = 7 :=
by
  sorry

end number_of_human_family_members_l0_33


namespace product_of_first_five_terms_l0_52

variable {a : ℕ → ℝ}

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ m n p q : ℕ, m > 0 ∧ n > 0 ∧ p > 0 ∧ q > 0 ∧ m + n = p + q → a m * a n = a p * a q

theorem product_of_first_five_terms 
  (h : geometric_sequence a) 
  (h3 : a 3 = 2) : 
  a 1 * a 2 * a 3 * a 4 * a 5 = 32 :=
sorry

end product_of_first_five_terms_l0_52


namespace emily_and_berengere_contribution_l0_61

noncomputable def euro_to_usd : ℝ := 1.20
noncomputable def euro_to_gbp : ℝ := 0.85

noncomputable def cake_cost_euros : ℝ := 12
noncomputable def cookies_cost_euros : ℝ := 5
noncomputable def total_cost_euros : ℝ := cake_cost_euros + cookies_cost_euros

noncomputable def emily_usd : ℝ := 10
noncomputable def liam_gbp : ℝ := 10

noncomputable def emily_euros : ℝ := emily_usd / euro_to_usd
noncomputable def liam_euros : ℝ := liam_gbp / euro_to_gbp

noncomputable def total_available_euros : ℝ := emily_euros + liam_euros

theorem emily_and_berengere_contribution : total_available_euros >= total_cost_euros := by
  sorry

end emily_and_berengere_contribution_l0_61


namespace number_of_boxes_l0_65

-- Definitions based on conditions
def bottles_per_box := 50
def bottle_capacity := 12
def fill_fraction := 3 / 4
def total_water := 4500

-- Question rephrased as a proof problem
theorem number_of_boxes (h1 : bottles_per_box = 50)
                        (h2 : bottle_capacity = 12)
                        (h3 : fill_fraction = 3 / 4)
                        (h4 : total_water = 4500) :
  4500 / ((12 : ℝ) * (3 / 4)) / 50 = 10 := 
by {
  sorry
}

end number_of_boxes_l0_65


namespace train_stoppages_l0_78

variables (sA sA' sB sB' sC sC' : ℝ)
variables (x y z : ℝ)

-- Conditions
def conditions : Prop :=
  sA = 80 ∧ sA' = 60 ∧
  sB = 100 ∧ sB' = 75 ∧
  sC = 120 ∧ sC' = 90

-- Goal that we need to prove
def goal : Prop :=
  x = 15 ∧ y = 15 ∧ z = 15

-- Main statement
theorem train_stoppages : conditions sA sA' sB sB' sC sC' → goal x y z :=
by
  sorry

end train_stoppages_l0_78


namespace number_of_valid_lines_l0_63

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

def lines_passing_through_point (x_int : ℕ) (y_int : ℕ) (p : ℕ × ℕ) : Prop :=
  p.1 * y_int + p.2 * x_int = x_int * y_int

theorem number_of_valid_lines (p : ℕ × ℕ) : 
  ∃! l : ℕ × ℕ, is_prime (l.1) ∧ is_power_of_two (l.2) ∧ lines_passing_through_point l.1 l.2 p :=
sorry

end number_of_valid_lines_l0_63


namespace john_speed_when_runs_alone_l0_17

theorem john_speed_when_runs_alone (x : ℝ) : 
  (6 * (1/2) + x * (1/2) = 5) → x = 4 :=
by
  intro h
  linarith

end john_speed_when_runs_alone_l0_17


namespace Tom_needs_11_25_hours_per_week_l0_59

theorem Tom_needs_11_25_hours_per_week
  (summer_weeks: ℕ) (summer_weeks_val: summer_weeks = 8)
  (summer_hours_per_week: ℕ) (summer_hours_per_week_val: summer_hours_per_week = 45)
  (summer_earnings: ℝ) (summer_earnings_val: summer_earnings = 3600)
  (rest_weeks: ℕ) (rest_weeks_val: rest_weeks = 40)
  (rest_earnings_goal: ℝ) (rest_earnings_goal_val: rest_earnings_goal = 4500) :
  (rest_earnings_goal / (summer_earnings / (summer_hours_per_week * summer_weeks))) / rest_weeks = 11.25 :=
by
  simp [summer_earnings_val, rest_earnings_goal_val, summer_hours_per_week_val, summer_weeks_val]
  sorry

end Tom_needs_11_25_hours_per_week_l0_59


namespace razorback_tshirt_money_l0_35

noncomputable def money_made_from_texas_tech_game (tshirt_price : ℕ) (total_sold : ℕ) (arkansas_sold : ℕ) : ℕ :=
  tshirt_price * (total_sold - arkansas_sold)

theorem razorback_tshirt_money :
  money_made_from_texas_tech_game 78 186 172 = 1092 := by
  sorry

end razorback_tshirt_money_l0_35


namespace books_loaned_out_l0_88

theorem books_loaned_out (initial_books loaned_books returned_percentage end_books missing_books : ℕ)
  (h1 : initial_books = 75)
  (h2 : end_books = 66)
  (h3 : returned_percentage = 70)
  (h4 : initial_books - end_books = missing_books)
  (h5 : missing_books = (loaned_books * (100 - returned_percentage)) / 100):
  loaned_books = 30 :=
by
  sorry

end books_loaned_out_l0_88


namespace compare_a_b_c_l0_70

noncomputable def a := 0.1 * Real.exp 0.1
def b := 1 / 9
noncomputable def c := -Real.log 0.9

theorem compare_a_b_c : c < a ∧ a < b := by
  have h_c_b : c < b := sorry
  have h_a_b : a < b := sorry
  have h_c_a : c < a := sorry
  exact ⟨h_c_a, h_a_b⟩

end compare_a_b_c_l0_70


namespace perfect_square_trinomial_l0_85

theorem perfect_square_trinomial (k : ℝ) : 
  ∃ a : ℝ, (x^2 - k*x + 1 = (x + a)^2) → (k = 2 ∨ k = -2) :=
by
  sorry

end perfect_square_trinomial_l0_85


namespace mabel_initial_daisies_l0_81

theorem mabel_initial_daisies (D: ℕ) (h1: 8 * (D - 2) = 24) : D = 5 :=
by
  sorry

end mabel_initial_daisies_l0_81


namespace area_of_efgh_l0_60

def small_rectangle_shorter_side : ℝ := 7
def small_rectangle_longer_side : ℝ := 3 * small_rectangle_shorter_side
def larger_rectangle_width : ℝ := small_rectangle_longer_side
def larger_rectangle_length : ℝ := small_rectangle_longer_side + small_rectangle_shorter_side

theorem area_of_efgh :
  larger_rectangle_length * larger_rectangle_width = 588 := by
  sorry

end area_of_efgh_l0_60


namespace find_number_of_elements_l0_8

theorem find_number_of_elements (n S : ℕ) (h1 : S + 26 = 19 * n) (h2 : S + 76 = 24 * n) : n = 10 := 
sorry

end find_number_of_elements_l0_8


namespace restaurant_dinners_sold_on_Monday_l0_75

theorem restaurant_dinners_sold_on_Monday (M : ℕ) 
  (h1 : ∀ tues_dinners, tues_dinners = M + 40) 
  (h2 : ∀ wed_dinners, wed_dinners = (M + 40) / 2)
  (h3 : ∀ thurs_dinners, thurs_dinners = ((M + 40) / 2) + 3)
  (h4 : M + (M + 40) + ((M + 40) / 2) + (((M + 40) / 2) + 3) = 203) : 
  M = 40 := 
sorry

end restaurant_dinners_sold_on_Monday_l0_75


namespace typing_difference_l0_66

theorem typing_difference (initial_speed after_speed : ℕ) (time_interval : ℕ) (h_initial : initial_speed = 10) 
  (h_after : after_speed = 8) (h_time : time_interval = 5) : 
  (initial_speed * time_interval) - (after_speed * time_interval) = 10 := 
by 
  sorry

end typing_difference_l0_66


namespace min_people_in_group_l0_24

theorem min_people_in_group (B G : ℕ) (h : B / (B + G : ℝ) > 0.94) : B + G ≥ 17 :=
sorry

end min_people_in_group_l0_24


namespace cat_food_insufficient_for_six_days_l0_14

theorem cat_food_insufficient_for_six_days
  (B S : ℕ)
  (h1 : B > S)
  (h2 : B < 2 * S)
  (h3 : B + 2 * S = 2 * ((B + 2 * S) / 2)) :
  4 * B + 4 * S < 3 * (B + 2 * S) :=
by sorry

end cat_food_insufficient_for_six_days_l0_14


namespace covered_digits_l0_12

def four_digit_int (n : ℕ) : Prop := n ≥ 1000 ∧ n < 10000

theorem covered_digits (a b c : ℕ) (n1 n2 n3 : ℕ) :
  four_digit_int n1 → four_digit_int n2 → four_digit_int n3 →
  n1 + n2 + n3 = 10126 →
  (n1 % 10 = 3 ∧ n2 % 10 = 7 ∧ n3 % 10 = 6) →
  (n1 / 10 % 10 = 4 ∧ n2 / 10 % 10 = a ∧ n3 / 10 % 10 = 2) →
  (n1 / 100 % 10 = 2 ∧ n2 / 100 % 10 = 1 ∧ n3 / 100 % 10 = c) →
  (n1 / 1000 = 1 ∧ n2 / 1000 = 2 ∧ n3 / 1000 = b) →
  (a = 5 ∧ b = 6 ∧ c = 7) := 
sorry

end covered_digits_l0_12


namespace problem_l0_43

def f (a b c x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 8

theorem problem 
  (a b c : ℝ) 
  (h : f a b c (-2) = 10) 
  : f a b c 2 = 6 :=
by
  sorry

end problem_l0_43


namespace pencils_to_make_profit_l0_79

theorem pencils_to_make_profit
  (total_pencils : ℕ)
  (cost_per_pencil : ℝ)
  (selling_price_per_pencil : ℝ)
  (desired_profit : ℝ)
  (pencils_to_be_sold : ℕ) :
  total_pencils = 2000 →
  cost_per_pencil = 0.08 →
  selling_price_per_pencil = 0.20 →
  desired_profit = 160 →
  pencils_to_be_sold = 1600 :=
sorry

end pencils_to_make_profit_l0_79


namespace probability_of_drawing_red_ball_l0_31

/-- Define the colors of the balls in the bag -/
def yellow_balls : ℕ := 2
def red_balls : ℕ := 3
def white_balls : ℕ := 5

/-- Define the total number of balls in the bag -/
def total_balls : ℕ := yellow_balls + red_balls + white_balls

/-- Define the probability of drawing exactly one red ball -/
def probability_of_red_ball : ℚ := red_balls / total_balls

/-- The main theorem to prove the given problem -/
theorem probability_of_drawing_red_ball :
  probability_of_red_ball = 3 / 10 :=
by
  -- Calculation steps would go here, but are omitted
  sorry

end probability_of_drawing_red_ball_l0_31


namespace weight_of_new_person_l0_21

theorem weight_of_new_person 
  (avg_increase : Real)
  (num_persons : Nat)
  (old_weight : Real)
  (new_avg_increase : avg_increase = 2.2)
  (number_of_persons : num_persons = 15)
  (weight_of_old_person : old_weight = 75)
  : (new_weight : Real) = old_weight + avg_increase * num_persons := 
  by sorry

end weight_of_new_person_l0_21


namespace tan_eq_tan_of_period_for_405_l0_0

theorem tan_eq_tan_of_period_for_405 (m : ℤ) (h : -180 < m ∧ m < 180) :
  (Real.tan (m * (Real.pi / 180))) = (Real.tan (405 * (Real.pi / 180))) ↔ m = 45 ∨ m = -135 :=
by sorry

end tan_eq_tan_of_period_for_405_l0_0


namespace surface_area_of_given_cube_l0_64

-- Define the edge length condition
def edge_length_of_cube (sum_edge_lengths : ℕ) :=
  sum_edge_lengths / 12

-- Define the surface area of a cube given an edge length
def surface_area_of_cube (edge_length : ℕ) :=
  6 * (edge_length * edge_length)

-- State the theorem
theorem surface_area_of_given_cube : 
  edge_length_of_cube 36 = 3 ∧ surface_area_of_cube 3 = 54 :=
by
  -- We leave the proof as an exercise.
  sorry

end surface_area_of_given_cube_l0_64


namespace students_at_end_l0_47

def initial_students := 11
def students_left := 6
def new_students := 42

theorem students_at_end (init : ℕ := initial_students) (left : ℕ := students_left) (new : ℕ := new_students) :
    (init - left + new) = 47 := 
by
  sorry

end students_at_end_l0_47


namespace trigonometric_identity_l0_68

open Real

theorem trigonometric_identity
  (α : ℝ)
  (h₁ : tan α + 1 / tan α = 10 / 3)
  (h₂ : π / 4 < α ∧ α < π / 2) :
  sin (2 * α + π / 4) + 2 * cos (π / 4) * sin α ^ 2 = 4 * sqrt 2 / 5 :=
by
  sorry

end trigonometric_identity_l0_68


namespace fraction_of_income_from_tips_l0_16

theorem fraction_of_income_from_tips (S T : ℚ) (h : T = (11/4) * S) : (T / (S + T)) = (11/15) :=
by sorry

end fraction_of_income_from_tips_l0_16


namespace sqrt_360000_eq_600_l0_95

theorem sqrt_360000_eq_600 : Real.sqrt 360000 = 600 := by
  sorry

end sqrt_360000_eq_600_l0_95


namespace new_mixture_concentration_l0_57

def vessel1_capacity : ℝ := 2
def vessel1_concentration : ℝ := 0.30
def vessel2_capacity : ℝ := 6
def vessel2_concentration : ℝ := 0.40
def total_volume : ℝ := 8
def expected_concentration : ℝ := 37.5

theorem new_mixture_concentration :
  ((vessel1_capacity * vessel1_concentration + vessel2_capacity * vessel2_concentration) / total_volume) * 100 = expected_concentration :=
by
  sorry

end new_mixture_concentration_l0_57


namespace possible_values_of_x_l0_22

theorem possible_values_of_x (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 10) (h2 : y + 1 / x = 5 / 12) : x = 4 ∨ x = 6 :=
by
  sorry

end possible_values_of_x_l0_22


namespace probability_heads_all_three_tosses_l0_38

theorem probability_heads_all_three_tosses :
  (1 / 2) * (1 / 2) * (1 / 2) = 1 / 8 := 
sorry

end probability_heads_all_three_tosses_l0_38


namespace compound_interest_amount_l0_98

theorem compound_interest_amount 
  (P_si : ℝ := 3225) 
  (R_si : ℝ := 8) 
  (T_si : ℝ := 5) 
  (R_ci : ℝ := 15) 
  (T_ci : ℝ := 2) 
  (SI : ℝ := P_si * R_si * T_si / 100) 
  (CI : ℝ := 2 * SI) 
  (CI_formula : ℝ := P_ci * ((1 + R_ci / 100)^T_ci - 1))
  (P_ci := 516 / 0.3225) :
  P_ci = 1600 := 
by
  sorry

end compound_interest_amount_l0_98


namespace temp_fri_l0_58

-- Define the temperatures on Monday, Tuesday, Wednesday, Thursday, and Friday
variables (M T W Th F : ℝ)

-- Define the conditions as given in the problem
axiom avg_mon_thurs : (M + T + W + Th) / 4 = 48
axiom avg_tues_fri : (T + W + Th + F) / 4 = 46
axiom temp_mon : M = 39

-- The theorem to prove that the temperature on Friday is 31 degrees
theorem temp_fri : F = 31 :=
by
  -- placeholder for proof
  sorry

end temp_fri_l0_58


namespace algebra_ineq_l0_84

theorem algebra_ineq (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (h : a * b + b * c + c * a = 1) : a + b + c ≥ 2 := 
by sorry

end algebra_ineq_l0_84


namespace train_length_l0_40

theorem train_length (speed_kmph : ℝ) (time_sec : ℝ) (length : ℝ) : 
  speed_kmph = 60 → time_sec = 12 → 
  length = speed_kmph * (1000 / 3600) * time_sec → 
  length = 200.04 :=
by
  intros h_speed h_time h_length
  sorry

end train_length_l0_40


namespace smallest_geometric_third_term_l0_51

theorem smallest_geometric_third_term (d : ℝ) (a₁ a₂ a₃ g₁ g₂ g₃ : ℝ) 
  (h_AP : a₁ = 5 ∧ a₂ = 5 + d ∧ a₃ = 5 + 2 * d)
  (h_GP : g₁ = a₁ ∧ g₂ = a₂ + 3 ∧ g₃ = a₃ + 15)
  (h_geom : (g₂)^2 = g₁ * g₃) : g₃ = -4 := 
by
  -- We would provide the proof here.
  sorry

end smallest_geometric_third_term_l0_51


namespace monotonicity_of_even_function_l0_39

-- Define the function and its properties
def f (m : ℝ) (x : ℝ) : ℝ := (m-1)*x^2 + 2*m*x + 3

-- A function is even if f(x) = f(-x) for all x
def is_even (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g x = g (-x)

-- The main theorem statement
theorem monotonicity_of_even_function :
  ∀ (m : ℝ), is_even (f m) → (f 0 = 3) ∧ (∀ x : ℝ, f 0 x = - x^2 + 3) →
  (∀ a b, -3 < a ∧ a < b ∧ b < 1 → f 0 a < f 0 b → f 0 b > f 0 a) :=
by
  intro m
  intro h
  intro H
  sorry

end monotonicity_of_even_function_l0_39


namespace no_solutions_to_equation_l0_56

theorem no_solutions_to_equation :
  ¬∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x ^ 2 - 2 * y ^ 2 = 5 := by
  sorry

end no_solutions_to_equation_l0_56


namespace ophelia_age_l0_46

/-- 
If Lennon is currently 8 years old, 
and in two years Ophelia will be four times as old as Lennon,
then Ophelia is currently 38 years old 
-/
theorem ophelia_age 
  (lennon_age : ℕ) 
  (ophelia_age_in_two_years : ℕ) 
  (h1 : lennon_age = 8)
  (h2 : ophelia_age_in_two_years = 4 * (lennon_age + 2)) : 
  ophelia_age_in_two_years - 2 = 38 :=
by
  sorry

end ophelia_age_l0_46


namespace purple_coincide_pairs_l0_97

theorem purple_coincide_pairs
    (yellow_triangles_upper : ℕ)
    (yellow_triangles_lower : ℕ)
    (green_triangles_upper : ℕ)
    (green_triangles_lower : ℕ)
    (purple_triangles_upper : ℕ)
    (purple_triangles_lower : ℕ)
    (yellow_coincide_pairs : ℕ)
    (green_coincide_pairs : ℕ)
    (yellow_purple_pairs : ℕ) :
    yellow_triangles_upper = 4 →
    yellow_triangles_lower = 4 →
    green_triangles_upper = 6 →
    green_triangles_lower = 6 →
    purple_triangles_upper = 10 →
    purple_triangles_lower = 10 →
    yellow_coincide_pairs = 3 →
    green_coincide_pairs = 4 →
    yellow_purple_pairs = 3 →
    (∃ purple_coincide_pairs : ℕ, purple_coincide_pairs = 5) :=
by sorry

end purple_coincide_pairs_l0_97


namespace max_value_real_roots_l0_30

theorem max_value_real_roots (k x1 x2 : ℝ) :
  (∀ k, k^2 + 3 * k + 5 ≥ 0) →
  (x1 + x2 = k - 2) →
  (x1 * x2 = k^2 + 3 * k + 5) →
  (x1^2 + x2^2 ≤ 18) :=
by
  intro h1 h2 h3
  sorry

end max_value_real_roots_l0_30


namespace final_student_count_is_correct_l0_2

-- Define the initial conditions
def initial_students : ℕ := 11
def students_left_first_semester : ℕ := 6
def students_joined_first_semester : ℕ := 25
def additional_students_second_semester : ℕ := 15
def students_transferred_second_semester : ℕ := 3
def students_switched_class_second_semester : ℕ := 2

-- Define the final number of students to be proven
def final_number_of_students : ℕ := 
  initial_students - students_left_first_semester + students_joined_first_semester + 
  additional_students_second_semester - students_transferred_second_semester - students_switched_class_second_semester

-- The theorem we need to prove
theorem final_student_count_is_correct : final_number_of_students = 40 := by
  sorry

end final_student_count_is_correct_l0_2


namespace find_m_l0_91

theorem find_m (m : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = (m^2 - 5*m + 7)*x^(m-2)) 
  (h2 : ∀ x, f (-x) = - f x) : 
  m = 3 :=
by
  sorry

end find_m_l0_91


namespace minimum_distance_l0_45

-- Define conditions and problem

def lies_on_line (P : ℝ × ℝ) : Prop :=
  P.1 + P.2 - 4 = 0

theorem minimum_distance (P : ℝ × ℝ) (h : lies_on_line P) : P.1^2 + P.2^2 ≥ 8 :=
sorry

end minimum_distance_l0_45


namespace max_side_of_triangle_exists_max_side_of_elevent_l0_28

noncomputable def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem max_side_of_triangle (a b c : ℕ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c)
  (h₄ : a + b + c = 24) (h_triangle : is_valid_triangle a b c) :
  max a (max b c) ≤ 11 :=
sorry

theorem exists_max_side_of_elevent (h₄ : ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c) :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c 
  ∧ max a (max b c) = 11 :=
sorry

end max_side_of_triangle_exists_max_side_of_elevent_l0_28


namespace trains_meet_after_time_l0_42

/-- Given the lengths of two trains, the initial distance between them, and their speeds,
prove that they will meet after approximately 2.576 seconds. --/
theorem trains_meet_after_time 
  (length_train1 : ℝ) (length_train2 : ℝ) (initial_distance : ℝ)
  (speed_train1_kmph : ℝ) (speed_train2_mps : ℝ) :
  length_train1 = 87.5 →
  length_train2 = 94.3 →
  initial_distance = 273.2 →
  speed_train1_kmph = 65 →
  speed_train2_mps = 88 →
  abs ((initial_distance / ((speed_train1_kmph * 1000 / 3600) + speed_train2_mps)) - 2.576) < 0.001 := by
  sorry

end trains_meet_after_time_l0_42


namespace binomial_coefficient_10_3_l0_87

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_coefficient_10_3_l0_87
