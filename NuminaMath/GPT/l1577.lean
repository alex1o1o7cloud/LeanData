import Mathlib

namespace fish_caught_l1577_157768

theorem fish_caught (x y : ℕ) 
  (h1 : y - 2 = 4 * (x + 2))
  (h2 : y - 6 = 2 * (x + 6)) :
  x = 4 ∧ y = 26 :=
by
  sorry

end fish_caught_l1577_157768


namespace average_visitors_per_day_l1577_157776

theorem average_visitors_per_day (average_sunday : ℕ) (average_other : ℕ) (days_in_month : ℕ) (begins_with_sunday : Bool) :
  average_sunday = 600 → average_other = 240 → days_in_month = 30 → begins_with_sunday = true → (8640 / 30 = 288) :=
by
  intros h1 h2 h3 h4
  sorry

end average_visitors_per_day_l1577_157776


namespace lowest_position_l1577_157708

theorem lowest_position (num_cyclists : ℕ) (num_stages : ℕ) (vasya_position : ℕ) :
  num_cyclists = 500 →
  num_stages = 15 →
  vasya_position = 7 →
  ∃ n, n = 91 :=
by
  intros
  sorry

end lowest_position_l1577_157708


namespace maxAdditionalTiles_l1577_157797

-- Board definition
structure Board where
  width : Nat
  height : Nat
  cells : List (Nat × Nat) -- List of cells occupied by tiles

def initialBoard : Board := 
  ⟨10, 9, [(1,1), (1,2), (2,1), (2,2), (3,1), (3,2), (4,1), (4,2), (5,1), (5,2),
            (6,1), (6,2), (7,1), (7,2)]⟩

-- Function to count cells occupied
def occupiedCells (b : Board) : Nat :=
  b.cells.length

-- Function to calculate total cells in a board
def totalCells (b : Board) : Nat :=
  b.width * b.height

-- Function to calculate additional 2x1 tiles that can be placed
def additionalTiles (board : Board) : Nat :=
  (totalCells board - occupiedCells board) / 2

theorem maxAdditionalTiles : additionalTiles initialBoard = 36 := by
  sorry

end maxAdditionalTiles_l1577_157797


namespace people_per_column_in_second_arrangement_l1577_157795
-- Import the necessary libraries

-- Define the conditions as given in the problem
def number_of_people_first_arrangement : ℕ := 30 * 16
def number_of_columns_second_arrangement : ℕ := 8

-- Define the problem statement with proof
theorem people_per_column_in_second_arrangement :
  (number_of_people_first_arrangement / number_of_columns_second_arrangement) = 60 :=
by
  -- Skip the proof here
  sorry

end people_per_column_in_second_arrangement_l1577_157795


namespace sum_of_reciprocals_of_shifted_roots_l1577_157757

theorem sum_of_reciprocals_of_shifted_roots (p q r : ℝ)
  (h1 : p^3 - 2 * p^2 - p + 3 = 0)
  (h2 : q^3 - 2 * q^2 - q + 3 = 0)
  (h3 : r^3 - 2 * r^2 - r + 3 = 0) :
  (1 / (p - 2)) + (1 / (q - 2)) + (1 / (r - 2)) = -3 :=
by
  sorry

end sum_of_reciprocals_of_shifted_roots_l1577_157757


namespace edward_earnings_l1577_157753

theorem edward_earnings
    (total_lawns : ℕ := 17)
    (forgotten_lawns : ℕ := 9)
    (total_earnings : ℕ := 32) :
    (total_earnings / (total_lawns - forgotten_lawns) = 4) :=
by
  sorry

end edward_earnings_l1577_157753


namespace regular_21_gon_symmetry_calculation_l1577_157760

theorem regular_21_gon_symmetry_calculation:
  let L := 21
  let R := 360 / 21
  L + R = 38 :=
by
  sorry

end regular_21_gon_symmetry_calculation_l1577_157760


namespace Union_A_B_eq_l1577_157742

noncomputable def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}
noncomputable def B : Set ℝ := {x | -2 < x ∧ x < 2}

theorem Union_A_B_eq : A ∪ B = {x | -2 < x ∧ x ≤ 4} :=
by
  sorry

end Union_A_B_eq_l1577_157742


namespace daily_coffee_machine_cost_l1577_157727

def coffee_machine_cost := 200 -- $200
def discount := 20 -- $20
def daily_coffee_cost := 2 * 4 -- $8/day
def days_to_pay_off := 36 -- 36 days

theorem daily_coffee_machine_cost :
  (days_to_pay_off * daily_coffee_cost - (coffee_machine_cost - discount)) / days_to_pay_off = 3 := 
by
  -- Using the given conditions: 
  -- coffee_machine_cost = 200
  -- discount = 20
  -- daily_coffee_cost = 8
  -- days_to_pay_off = 36
  sorry

end daily_coffee_machine_cost_l1577_157727


namespace base_problem_l1577_157748

theorem base_problem (c d : Nat) (pos_c : c > 0) (pos_d : d > 0) (h : 5 * c + 8 = 8 * d + 5) : c + d = 15 :=
sorry

end base_problem_l1577_157748


namespace entire_show_length_l1577_157789

def first_segment (S T : ℕ) : ℕ := 2 * (S + T)
def second_segment (T : ℕ) : ℕ := 2 * T
def third_segment : ℕ := 10

theorem entire_show_length : 
  first_segment (second_segment third_segment) third_segment + 
  second_segment third_segment + 
  third_segment = 90 :=
by
  sorry

end entire_show_length_l1577_157789


namespace winning_jackpot_is_event_l1577_157784

-- Definitions based on the conditions
def has_conditions (experiment : String) : Prop :=
  experiment = "A" ∨ experiment = "B" ∨ experiment = "C" ∨ experiment = "D"

def has_outcomes (experiment : String) : Prop :=
  experiment = "D"

def is_event (experiment : String) : Prop :=
  has_conditions experiment ∧ has_outcomes experiment

-- Statement to prove
theorem winning_jackpot_is_event : is_event "D" :=
by
  -- Trivial step to show that D meets both conditions and outcomes
  exact sorry

end winning_jackpot_is_event_l1577_157784


namespace page_added_twice_is_33_l1577_157738

noncomputable def sum_first_n (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem page_added_twice_is_33 :
  ∃ n : ℕ, ∃ m : ℕ, sum_first_n n + m = 1986 ∧ 1 ≤ m ∧ m ≤ n → m = 33 := 
by {
  sorry
}

end page_added_twice_is_33_l1577_157738


namespace marathon_yards_l1577_157755

theorem marathon_yards (miles_per_marathon : ℕ) (extra_yards_per_marathon : ℕ) (yards_per_mile : ℕ) (num_marathons : ℕ)
  (total_miles : ℕ) (total_yards : ℕ) 
  (H1 : miles_per_marathon = 26) 
  (H2 : extra_yards_per_marathon = 395) 
  (H3 : yards_per_mile = 1760) 
  (H4 : num_marathons = 15) 
  (H5 : total_miles = num_marathons * miles_per_marathon + (num_marathons * extra_yards_per_marathon) / yards_per_mile)
  (H6 : total_yards = (num_marathons * extra_yards_per_marathon) % yards_per_mile)
  (H7 : 0 ≤ total_yards ∧ total_yards < yards_per_mile) 
  : total_yards = 645 :=
sorry

end marathon_yards_l1577_157755


namespace original_combined_price_l1577_157794

theorem original_combined_price (C S : ℝ)
  (hC_new : (C + 0.25 * C) = 12.5)
  (hS_new : (S + 0.50 * S) = 13.5) :
  (C + S) = 19 := by
  -- sorry makes sure to skip the proof
  sorry

end original_combined_price_l1577_157794


namespace suitable_for_lottery_method_B_l1577_157766

def total_items_A : Nat := 3000
def samples_A : Nat := 600

def total_items_B (n: Nat) : Nat := 2 * 15
def samples_B : Nat := 6

def total_items_C : Nat := 2 * 15
def samples_C : Nat := 6

def total_items_D : Nat := 3000
def samples_D : Nat := 10

def is_lottery_suitable (total_items : Nat) (samples : Nat) (different_factories : Bool) : Bool :=
  total_items <= 30 && samples <= total_items && !different_factories

theorem suitable_for_lottery_method_B : 
  is_lottery_suitable (total_items_B 2) samples_B false = true :=
  sorry

end suitable_for_lottery_method_B_l1577_157766


namespace expand_polynomials_l1577_157793

def p (z : ℤ) := 3 * z^3 + 4 * z^2 - 2 * z + 1
def q (z : ℤ) := 2 * z^2 - 3 * z + 5
def r (z : ℤ) := 10 * z^5 - 8 * z^4 + 11 * z^3 + 5 * z^2 - 10 * z + 5

theorem expand_polynomials (z : ℤ) : (p z) * (q z) = r z :=
by sorry

end expand_polynomials_l1577_157793


namespace exists_line_l_l1577_157777

-- Define the parabola and line l1
def parabola (P : ℝ × ℝ) : Prop := P.2^2 = 8 * P.1
def line_l1 (P : ℝ × ℝ) : Prop := P.1 + 5 * P.2 - 5 = 0

-- Define the problem statement
theorem exists_line_l :
  ∃ l : ℝ × ℝ → Prop, 
    ((∃ A B : ℝ × ℝ, parabola A ∧ parabola B ∧ A ≠ B ∧ l A ∧ l B) ∧
    (∃ M : ℝ × ℝ, M = (1, 4/5) ∧ line_l1 M) ∧
    (∀ A B : ℝ × ℝ, l A ∧ l B → (A.2 - B.2) / (A.1 - B.1) = 5)) ∧
    (∀ P : ℝ × ℝ, l P ↔ 25 * P.1 - 5 * P.2 - 21 = 0) :=
sorry

end exists_line_l_l1577_157777


namespace radio_price_position_l1577_157723

def price_positions (n : ℕ) (total_items : ℕ) (rank_lowest : ℕ) : Prop :=
  rank_lowest = total_items - n + 1

theorem radio_price_position :
  ∀ (n total_items rank_lowest : ℕ),
    total_items = 34 →
    rank_lowest = 21 →
    price_positions n total_items rank_lowest →
    n = 14 :=
by
  intros n total_items rank_lowest h_total h_rank h_pos
  rw [h_total, h_rank] at h_pos
  sorry

end radio_price_position_l1577_157723


namespace train_length_l1577_157711

theorem train_length (speed : ℝ) (time_seconds : ℝ) (time_hours : ℝ) (distance_km : ℝ) (distance_m : ℝ) 
  (h1 : speed = 60) 
  (h2 : time_seconds = 42) 
  (h3 : time_hours = time_seconds / 3600)
  (h4 : distance_km = speed * time_hours) 
  (h5 : distance_m = distance_km * 1000) :
  distance_m = 700 :=
by 
  sorry

end train_length_l1577_157711


namespace max_cables_cut_l1577_157759

/-- 
Prove that given 200 computers connected by 345 cables initially forming a single cluster, after 
cutting cables to form 8 clusters, the maximum possible number of cables that could have been 
cut is 153.
--/
theorem max_cables_cut (computers : ℕ) (initial_cables : ℕ) (final_clusters : ℕ) (initial_clusters : ℕ) 
  (minimal_cables : ℕ) (cuts : ℕ) : 
  computers = 200 ∧ initial_cables = 345 ∧ final_clusters = 8 ∧ initial_clusters = 1 ∧ 
  minimal_cables = computers - final_clusters ∧ 
  cuts = initial_cables - minimal_cables →
  cuts = 153 := 
sorry

end max_cables_cut_l1577_157759


namespace intersection_A_B_l1577_157791

section
  def A : Set ℤ := {-2, 0, 1}
  def B : Set ℤ := {x | x^2 > 1}
  theorem intersection_A_B : A ∩ B = {-2} := 
  by
    sorry
end

end intersection_A_B_l1577_157791


namespace boat_travel_distance_downstream_l1577_157750

-- Definitions of the given conditions
def boatSpeedStillWater : ℕ := 10 -- km/hr
def streamSpeed : ℕ := 8 -- km/hr
def timeDownstream : ℕ := 3 -- hours

-- Effective speed downstream
def effectiveSpeedDownstream : ℕ := boatSpeedStillWater + streamSpeed

-- Goal: Distance traveled downstream equals 54 km
theorem boat_travel_distance_downstream :
  effectiveSpeedDownstream * timeDownstream = 54 := 
by
  -- Since only the statement is needed, we use sorry to indicate the proof is skipped
  sorry

end boat_travel_distance_downstream_l1577_157750


namespace complex_mul_eq_l1577_157731

/-- Proof that the product of two complex numbers (1 + i) and (2 + i) is equal to (1 + 3i) -/
theorem complex_mul_eq (i : ℂ) (h_i_squared : i^2 = -1) : (1 + i) * (2 + i) = 1 + 3 * i :=
by
  -- The actual proof logic goes here.
  sorry

end complex_mul_eq_l1577_157731


namespace problem_statement_l1577_157788

-- Definitions
def MagnitudeEqual : Prop := (2.4 : ℝ) = (2.40 : ℝ)
def CountUnit2_4 : Prop := (0.1 : ℝ) = 2.4 / 24
def CountUnit2_40 : Prop := (0.01 : ℝ) = 2.40 / 240

-- Theorem statement
theorem problem_statement : MagnitudeEqual ∧ CountUnit2_4 ∧ CountUnit2_40 → True := by
  intros
  sorry

end problem_statement_l1577_157788


namespace cans_purchased_l1577_157763

theorem cans_purchased (S Q E : ℝ) (h1 : Q ≠ 0) (h2 : S > 0) :
  (10 * E * S) / Q = (10 * (E : ℝ) * (S : ℝ)) / (Q : ℝ) := by 
  sorry

end cans_purchased_l1577_157763


namespace ellipse_product_l1577_157737

noncomputable def a (b : ℝ) := b + 4
noncomputable def AB (a: ℝ) := 2 * a
noncomputable def CD (b: ℝ) := 2 * b

theorem ellipse_product:
  (∀ (a b : ℝ), a = b + 4 → a^2 - b^2 = 64) →
  (∃ (a b : ℝ), (AB a) * (CD b) = 240) :=
by
  intros h
  use 10, 6
  simp [AB, CD]
  sorry

end ellipse_product_l1577_157737


namespace total_value_of_treats_l1577_157786

def hotel_cost_per_night : ℕ := 4000
def number_of_nights : ℕ := 2
def car_cost : ℕ := 30000
def house_multiplier : ℕ := 4

theorem total_value_of_treats : 
  (number_of_nights * hotel_cost_per_night) + car_cost + (house_multiplier * car_cost) = 158000 := 
by
  sorry

end total_value_of_treats_l1577_157786


namespace diagonals_in_nine_sided_polygon_l1577_157749

-- Given a regular polygon with 9 sides
def regular_polygon_sides : ℕ := 9

-- To find the number of diagonals in a polygon, we use the formula
noncomputable def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- We need to prove this particular instance where the number of sides is 9
theorem diagonals_in_nine_sided_polygon : number_of_diagonals regular_polygon_sides = 27 := 
by sorry

end diagonals_in_nine_sided_polygon_l1577_157749


namespace gold_coins_count_l1577_157761

theorem gold_coins_count (n c : ℕ) (h1 : n = 8 * (c - 3))
                                     (h2 : n = 5 * c + 4)
                                     (h3 : c ≥ 10) : n = 54 :=
by
  sorry

end gold_coins_count_l1577_157761


namespace find_last_number_l1577_157739

theorem find_last_number (A B C D : ℝ) (h1 : A + B + C = 18) (h2 : B + C + D = 9) (h3 : A + D = 13) : D = 2 :=
by
sorry

end find_last_number_l1577_157739


namespace combined_stickers_l1577_157718

theorem combined_stickers (k j a : ℕ) (h : 7 * j + 5 * a = 54) (hk : k = 42) (hk_ratio : k = 7 * 6) :
  j + a = 54 :=
by
  sorry

end combined_stickers_l1577_157718


namespace quadratic_inequality_l1577_157796

-- Define the quadratic function and conditions
variables {a b c x0 y1 y2 y3 : ℝ}
variables (A : (a * x0^2 + b * x0 + c = 0))
variables (B : (a * (-2)^2 + b * (-2) + c = 0))
variables (C : (a + b + c) * (4 * a + 2 * b + c) < 0)
variables (D : a > 0)
variables (E1 : y1 = a * (-1)^2 + b * (-1) + c)
variables (E2 : y2 = a * (- (sqrt 2) / 2)^2 + b * (- (sqrt 2) / 2) + c)
variables (E3 : y3 = a * 1^2 + b * 1 + c)

-- Prove that y3 > y1 > y2
theorem quadratic_inequality : y3 > y1 ∧ y1 > y2 := by 
  sorry

end quadratic_inequality_l1577_157796


namespace min_seats_to_occupy_l1577_157725

theorem min_seats_to_occupy (n : ℕ) (h_n : n = 150) : 
  ∃ (k : ℕ), k = 90 ∧ ∀ m : ℕ, m ≥ k → ∀ i : ℕ, i < n → ∃ j : ℕ, (j < n) ∧ ((j = i + 1) ∨ (j = i - 1)) :=
sorry

end min_seats_to_occupy_l1577_157725


namespace bob_wins_even_n_l1577_157710

def game_of_islands (n : ℕ) (even_n : n % 2 = 0) : Prop :=
  ∃ strategy : (ℕ → ℕ), -- strategy is a function representing each player's move
    ∀ A B : ℕ → ℕ, -- A and B represent the moves of Alice and Bob respectively
    (A 0 + B 1) = n → (A (A 0 + 1) ≠ B (A 0 + 1)) -- Bob can always mirror Alice’s move.

theorem bob_wins_even_n (n : ℕ) (h : n % 2 = 0) : game_of_islands n h :=
sorry

end bob_wins_even_n_l1577_157710


namespace problem_statement_l1577_157724

theorem problem_statement (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (b + 2 * c) + b / (c + 2 * a) + c / (a + 2 * b) > 1 / 2) :=
by
  sorry

end problem_statement_l1577_157724


namespace evaluate_expression_l1577_157765

theorem evaluate_expression : (↑7 ^ (1/4) / ↑7 ^ (1/6)) = (↑7 ^ (1/12)) :=
by
  sorry

end evaluate_expression_l1577_157765


namespace fairfield_middle_school_geography_players_l1577_157721

/-- At Fairfield Middle School, there are 24 players on the football team.
All players are enrolled in at least one of the subjects: history or geography.
There are 10 players taking history and 6 players taking both subjects.
We need to prove that the number of players taking geography is 20. -/
theorem fairfield_middle_school_geography_players
  (total_players : ℕ)
  (history_players : ℕ)
  (both_subjects_players : ℕ)
  (h1 : total_players = 24)
  (h2 : history_players = 10)
  (h3 : both_subjects_players = 6) :
  total_players - (history_players - both_subjects_players) = 20 :=
by {
  sorry
}

end fairfield_middle_school_geography_players_l1577_157721


namespace intersection_eq_l1577_157782

def A : Set ℤ := {-2, -1, 3, 4}
def B : Set ℤ := {-1, 2, 3}

theorem intersection_eq : A ∩ B = {-1, 3} := 
by
  sorry

end intersection_eq_l1577_157782


namespace parallel_lines_m_eq_neg2_l1577_157780

def l1_equation (m : ℝ) (x y: ℝ) : Prop :=
  (m+1) * x + y - 1 = 0

def l2_equation (m : ℝ) (x y: ℝ) : Prop :=
  2 * x + m * y - 1 = 0

theorem parallel_lines_m_eq_neg2 (m : ℝ) :
  (∀ x y : ℝ, l1_equation m x y) →
  (∀ x y : ℝ, l2_equation m x y) →
  (m ≠ 1) →
  (m = -2) :=
sorry

end parallel_lines_m_eq_neg2_l1577_157780


namespace rectangle_width_is_3_l1577_157746

-- Define the given conditions
def length_square : ℝ := 9
def length_rectangle : ℝ := 27

-- Calculate the area based on the given conditions
def area_square : ℝ := length_square * length_square

-- Define the area equality condition
def area_equality (width_rectangle : ℝ) : Prop :=
  area_square = length_rectangle * width_rectangle

-- The theorem stating the width of the rectangle
theorem rectangle_width_is_3 (width_rectangle: ℝ) :
  area_equality width_rectangle → width_rectangle = 3 :=
by
  -- Skipping the proof itself as instructed
  intro h
  sorry

end rectangle_width_is_3_l1577_157746


namespace solve_problem_l1577_157775

theorem solve_problem :
  ∃ a b c d e f : ℤ,
  (208208 = 8^5 * a + 8^4 * b + 8^3 * c + 8^2 * d + 8 * e + f) ∧
  (0 ≤ a ∧ a ≤ 7) ∧ (0 ≤ b ∧ b ≤ 7) ∧ (0 ≤ c ∧ c ≤ 7) ∧
  (0 ≤ d ∧ d ≤ 7) ∧ (0 ≤ e ∧ e ≤ 7) ∧ (0 ≤ f ∧ f ≤ 7) ∧
  (a * b * c + d * e * f = 72) :=
by
  sorry

end solve_problem_l1577_157775


namespace number_of_ordered_pairs_l1577_157744

theorem number_of_ordered_pairs :
  ∃ n : ℕ, n = 89 ∧ (∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ x < y ∧ 2 * x * y = 8 ^ 30 * (x + y)) := sorry

end number_of_ordered_pairs_l1577_157744


namespace purely_imaginary_solution_l1577_157778

noncomputable def complex_number_is_purely_imaginary (m : ℝ) : Prop :=
  (m^2 - 2 * m - 3 = 0) ∧ (m + 1 ≠ 0)

theorem purely_imaginary_solution (m : ℝ) (h : complex_number_is_purely_imaginary m) : m = 3 := by
  sorry

end purely_imaginary_solution_l1577_157778


namespace find_m_value_l1577_157702

theorem find_m_value (m : ℝ) (h : (m - 4)^2 + 1^2 + 2^2 = 30) : m = 9 ∨ m = -1 :=
by {
  sorry
}

end find_m_value_l1577_157702


namespace max_product_of_functions_l1577_157741

theorem max_product_of_functions (f h : ℝ → ℝ) (hf : ∀ x, -5 ≤ f x ∧ f x ≤ 3) (hh : ∀ x, -3 ≤ h x ∧ h x ≤ 4) :
  ∃ x, f x * h x = 20 :=
by {
  sorry
}

end max_product_of_functions_l1577_157741


namespace at_least_240_students_l1577_157733

-- Define the total number of students
def total_students : ℕ := 1200

-- Define the 80th percentile score
def percentile_80_score : ℕ := 103

-- Define the number of students below the 80th percentile
def students_below_80th_percentile : ℕ := total_students * 80 / 100

-- Define the number of students with at least the 80th percentile score
def students_at_least_80th_percentile : ℕ := total_students - students_below_80th_percentile

-- The theorem to prove
theorem at_least_240_students : students_at_least_80th_percentile ≥ 240 :=
by
  -- Placeholder proof, to be filled in as the actual proof
  sorry

end at_least_240_students_l1577_157733


namespace pizza_slices_left_l1577_157770

theorem pizza_slices_left (total_slices john_ate : ℕ) 
  (initial_slices : total_slices = 12) 
  (john_slices : john_ate = 3) 
  (sam_ate : ¬¬(2 * john_ate = 6)) : 
  ∃ slices_left, slices_left = 3 :=
by
  sorry

end pizza_slices_left_l1577_157770


namespace family_of_four_children_has_at_least_one_boy_and_one_girl_l1577_157783

noncomputable section

def probability_at_least_one_boy_one_girl : ℚ :=
  1 - (1 / 16 + 1 / 16)

theorem family_of_four_children_has_at_least_one_boy_and_one_girl :
  probability_at_least_one_boy_one_girl = 7 / 8 := by
  sorry

end family_of_four_children_has_at_least_one_boy_and_one_girl_l1577_157783


namespace total_votes_election_l1577_157754

theorem total_votes_election
  (pct_candidate1 pct_candidate2 pct_candidate3 pct_candidate4 : ℝ)
  (votes_candidate4 total_votes : ℝ)
  (h1 : pct_candidate1 = 0.42)
  (h2 : pct_candidate2 = 0.30)
  (h3 : pct_candidate3 = 0.20)
  (h4 : pct_candidate4 = 0.08)
  (h5 : votes_candidate4 = 720)
  (h6 : votes_candidate4 = pct_candidate4 * total_votes) :
  total_votes = 9000 :=
sorry

end total_votes_election_l1577_157754


namespace distance_point_C_to_line_is_2_inch_l1577_157706

/-- 
Four 2-inch squares are aligned in a straight line. The second square from the left is rotated 90 degrees, 
and then shifted vertically downward until it touches the adjacent squares. Prove that the distance from 
point C, the top vertex of the rotated square, to the original line on which the bases of the squares were 
placed is 2 inches.
-/
theorem distance_point_C_to_line_is_2_inch :
  ∀ (squares : Fin 4 → ℝ) (rotation : ℝ) (vertical_shift : ℝ) (C_position : ℝ),
  (∀ n : Fin 4, squares n = 2) →
  rotation = 90 →
  vertical_shift = 0 →
  C_position = 2 →
  C_position = 2 :=
by
  intros squares rotation vertical_shift C_position
  sorry

end distance_point_C_to_line_is_2_inch_l1577_157706


namespace regular_polygon_sides_l1577_157717

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (x : ℕ), x > 2 → n = x)
  (h2 : ∀ (θ : ℕ), θ = 18 → 360 / n = θ) : n = 20 := by
  sorry

end regular_polygon_sides_l1577_157717


namespace factorize_expression_l1577_157774

theorem factorize_expression (x : ℝ) :
  (x + 1)^4 + (x + 3)^4 - 272 = 2 * (x^2 + 4*x + 19) * (x + 5) * (x - 1) :=
  sorry

end factorize_expression_l1577_157774


namespace coupon_percentage_l1577_157781

theorem coupon_percentage (P i d final_price total_price discount_amount percentage: ℝ)
  (h1 : P = 54) (h2 : i = 20) (h3 : d = 0.20 * i) 
  (h4 : total_price = P - d) (h5 : final_price = 45) 
  (h6 : discount_amount = total_price - final_price) 
  (h7 : percentage = (discount_amount / total_price) * 100) : 
  percentage = 10 := 
by
  sorry

end coupon_percentage_l1577_157781


namespace championship_outcomes_l1577_157704

theorem championship_outcomes (students events : ℕ) (h_students : students = 3) (h_events : events = 2) : 
  students ^ events = 9 :=
by
  rw [h_students, h_events]
  have h : 3 ^ 2 = 9 := by norm_num
  exact h

end championship_outcomes_l1577_157704


namespace perfect_cube_factors_count_l1577_157729

-- Define the given prime factorization
def prime_factorization_8820 : Prop :=
  ∃ a b c d : ℕ, 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 1 ∧ 0 ≤ d ∧ d ≤ 2 ∧
  (2 ^ a) * (3 ^ b) * (5 ^ c) * (7 ^ d) = 8820

-- Prove the statement about positive integer factors that are perfect cubes
theorem perfect_cube_factors_count : prime_factorization_8820 → (∃ n : ℕ, n = 1) :=
by
  sorry

end perfect_cube_factors_count_l1577_157729


namespace problem_statement_l1577_157705

theorem problem_statement (x : ℚ) (h : 8 * x = 3) : 200 * (1 / x) = 1600 / 3 :=
by
  sorry

end problem_statement_l1577_157705


namespace greatest_divisor_consistent_remainder_l1577_157726

noncomputable def gcd_of_differences : ℕ :=
  Nat.gcd (Nat.gcd 1050 28770) 71670

theorem greatest_divisor_consistent_remainder :
  gcd_of_differences = 30 :=
by
  -- The proof can be filled in here...
  sorry

end greatest_divisor_consistent_remainder_l1577_157726


namespace sum_first_15_odd_integers_l1577_157785

theorem sum_first_15_odd_integers : 
  let a := 1
  let n := 15
  let d := 2
  let l := a + (n-1) * d
  let S := n / 2 * (a + l)
  S = 225 :=
by
  sorry

end sum_first_15_odd_integers_l1577_157785


namespace primes_diff_power_of_two_divisible_by_three_l1577_157709

theorem primes_diff_power_of_two_divisible_by_three
  (p q : ℕ) (m n : ℕ)
  (hp : Prime p) (hq : Prime q) (hp_gt : p > 3) (hq_gt : q > 3)
  (diff : q - p = 2^n ∨ p - q = 2^n) :
  3 ∣ (p^(2*m+1) + q^(2*m+1)) := by
  sorry

end primes_diff_power_of_two_divisible_by_three_l1577_157709


namespace three_alpha_four_plus_eight_beta_three_eq_876_l1577_157707

variable (α β : ℝ)

-- Condition 1: α and β are roots of the equation x^2 - 3x - 4 = 0
def roots_of_quadratic : Prop := α^2 - 3 * α - 4 = 0 ∧ β^2 - 3 * β - 4 = 0

-- Question: 3α^4 + 8β^3 = ?
theorem three_alpha_four_plus_eight_beta_three_eq_876 
  (h : roots_of_quadratic α β) : (3 * α^4 + 8 * β^3 = 876) := sorry

end three_alpha_four_plus_eight_beta_three_eq_876_l1577_157707


namespace problem_statement_l1577_157772

def U := Set ℝ
def M := { x : ℝ | x^2 - 4 * x - 5 < 0 }
def N := { x : ℝ | 1 ≤ x }
def comp_U_N := { x : ℝ | x < 1 }
def intersection := { x : ℝ | -1 < x ∧ x < 1 }

theorem problem_statement : M ∩ comp_U_N = intersection := sorry

end problem_statement_l1577_157772


namespace factorial_multiple_of_3_l1577_157713

theorem factorial_multiple_of_3 (n : ℤ) (h : n ≥ 9) : 3 ∣ (n+1) * (n+3) :=
sorry

end factorial_multiple_of_3_l1577_157713


namespace sea_creatures_lost_l1577_157751

theorem sea_creatures_lost (sea_stars seashells snails items_left : ℕ) 
  (h1 : sea_stars = 34) 
  (h2 : seashells = 21) 
  (h3 : snails = 29) 
  (h4 : items_left = 59) : 
  sea_stars + seashells + snails - items_left = 25 :=
by
  sorry

end sea_creatures_lost_l1577_157751


namespace shallow_depth_of_pool_l1577_157745

theorem shallow_depth_of_pool (w l D V : ℝ) (h₀ : w = 9) (h₁ : l = 12) (h₂ : D = 4) (h₃ : V = 270) :
  (0.5 * (d + D) * w * l = V) → d = 1 :=
by
  intros h_equiv
  sorry

end shallow_depth_of_pool_l1577_157745


namespace func1_max_min_func2_max_min_l1577_157743

noncomputable def func1 (x : ℝ) : ℝ := 2 * Real.sin x - 3
noncomputable def func2 (x : ℝ) : ℝ := (7/4 : ℝ) + Real.sin x - (Real.sin x) ^ 2

theorem func1_max_min : (∀ x : ℝ, func1 x ≤ -1) ∧ (∃ x : ℝ, func1 x = -1) ∧ (∀ x : ℝ, func1 x ≥ -5) ∧ (∃ x : ℝ, func1 x = -5)  :=
by
  sorry

theorem func2_max_min : (∀ x : ℝ, func2 x ≤ 2) ∧ (∃ x : ℝ, func2 x = 2) ∧ (∀ x : ℝ, func2 x ≥ 7 / 4) ∧ (∃ x : ℝ, func2 x = 7 / 4) :=
by
  sorry

end func1_max_min_func2_max_min_l1577_157743


namespace mean_of_remaining_four_numbers_l1577_157715

theorem mean_of_remaining_four_numbers (a b c d : ℝ) 
  (h_mean_five : (a + b + c + d + 120) / 5 = 100) : 
  (a + b + c + d) / 4 = 95 :=
by
  sorry

end mean_of_remaining_four_numbers_l1577_157715


namespace find_linear_function_l1577_157728

theorem find_linear_function (α : ℝ) (hα : α > 0)
  (f : ℕ+ → ℝ)
  (h : ∀ (k m : ℕ+), α * (m : ℝ) ≤ (k : ℝ) ∧ (k : ℝ) < (α + 1) * (m : ℝ) → f (k + m) = f k + f m)
: ∃ (b : ℝ), ∀ (n : ℕ+), f n = b * (n : ℝ) :=
sorry

end find_linear_function_l1577_157728


namespace best_k_k_l1577_157790

theorem best_k_k' (v w x y z : ℝ) (hv : 0 < v) (hw : 0 < w) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  1 < (v / (v + w) + w / (w + x) + x / (x + y) + y / (y + z) + z / (z + v)) ∧ 
  (v / (v + w) + w / (w + x) + x / (x + y) + y / (y + z) + z / (z + v)) < 4 :=
sorry

end best_k_k_l1577_157790


namespace sum_of_corners_is_164_l1577_157792

section CheckerboardSum

-- Define the total number of elements in the 9x9 grid
def num_elements := 81

-- Define the positions of the corners
def top_left : ℕ := 1
def top_right : ℕ := 9
def bottom_left : ℕ := 73
def bottom_right : ℕ := 81

-- Define the sum of the corners
def corner_sum : ℕ := top_left + top_right + bottom_left + bottom_right

-- State the theorem
theorem sum_of_corners_is_164 : corner_sum = 164 :=
by
  exact sorry

end CheckerboardSum

end sum_of_corners_is_164_l1577_157792


namespace division_problem_l1577_157716

theorem division_problem (n : ℕ) (h : n / 4 = 12) : n / 3 = 16 := by
  sorry

end division_problem_l1577_157716


namespace expression_not_computable_by_square_difference_l1577_157735

theorem expression_not_computable_by_square_difference (x : ℝ) :
  ¬ ((x + 1) * (1 + x) = (x + 1) * (x - 1) ∨
     (x + 1) * (1 + x) = (-x + 1) * (-x - 1) ∨
     (x + 1) * (1 + x) = (x + 1) * (-x + 1)) :=
by
  sorry

end expression_not_computable_by_square_difference_l1577_157735


namespace max_a_plus_2b_plus_c_l1577_157732

open Real

theorem max_a_plus_2b_plus_c
  (A : Set ℝ := {x | |x + 1| ≤ 4})
  (T : ℝ := 3)
  (a b c : ℝ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_T : a^2 + b^2 + c^2 = T) :
  a + 2 * b + c ≤ 3 * sqrt 2 :=
by
  -- Proof is omitted
  sorry

end max_a_plus_2b_plus_c_l1577_157732


namespace mila_social_media_hours_l1577_157779

/-- 
Mila spends 6 hours on his phone every day. 
Half of this time is spent on social media. 
Prove that Mila spends 21 hours on social media in a week.
-/
theorem mila_social_media_hours 
  (hours_per_day : ℕ)
  (phone_time_per_day : hours_per_day = 6)
  (daily_social_media_fraction : ℕ)
  (fractional_time : daily_social_media_fraction = hours_per_day / 2)
  (days_per_week : ℕ)
  (days_in_week : days_per_week = 7) :
  (daily_social_media_fraction * days_per_week = 21) :=
sorry

end mila_social_media_hours_l1577_157779


namespace tasty_residue_count_2016_l1577_157719

def tasty_residue (n : ℕ) (a : ℕ) : Prop :=
  1 < a ∧ a < n ∧ ∃ m : ℕ, m > 1 ∧ a ^ m ≡ a [MOD n]

theorem tasty_residue_count_2016 : 
  (∃ count : ℕ, count = 831 ∧ ∀ a : ℕ, 1 < a ∧ a < 2016 ↔ tasty_residue 2016 a) :=
sorry

end tasty_residue_count_2016_l1577_157719


namespace gcd_1722_966_l1577_157773

theorem gcd_1722_966 : Nat.gcd 1722 966 = 42 :=
  sorry

end gcd_1722_966_l1577_157773


namespace angle_proof_l1577_157701

-- Variables and assumptions
variable {α : Type} [LinearOrderedField α]    -- using a general type for angles
variable {A B C D E : α}                       -- points of the triangle and extended segment

-- Given conditions
variable (angle_ACB angle_ABC : α)
variable (H1 : angle_ACB = 2 * angle_ABC)      -- angle condition
variable (CD BD AD DE : α)
variable (H2 : CD = 2 * BD)                    -- segment length condition
variable (H3 : AD = DE)                        -- extended segment condition

-- The proof goal in Lean format
theorem angle_proof (H1 : angle_ACB = 2 * angle_ABC) 
  (H2 : CD = 2 * BD) 
  (H3 : AD = DE) :
  angle_ECB + 180 = 2 * angle_EBC := 
sorry  -- proof to be filled in

end angle_proof_l1577_157701


namespace length_of_plot_l1577_157736

open Real

variable (breadth : ℝ) (length : ℝ)
variable (b : ℝ)

axiom H1 : length = b + 40
axiom H2 : 26.5 * (4 * b + 80) = 5300

theorem length_of_plot : length = 70 :=
by
  -- To prove: The length of the plot is 70 meters.
  exact sorry

end length_of_plot_l1577_157736


namespace union_complement_subset_range_l1577_157740

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 0 < 2 * x + a ∧ 2 * x + a ≤ 3}
def B : Set ℝ := {x | 2 * x ^ 2 - 3 * x - 2 < 0}

-- Define the complement of B
def complement_R (s : Set ℝ) : Set ℝ := {x | x ∉ s}

-- 1. The proof problem for A ∪ (complement of B) when a = 1
theorem union_complement (a : ℝ) (h : a = 1) :
  { x : ℝ | (-1/2 < x ∧ x ≤ 1) ∨ (x ≥ 2 ∨ x ≤ -1/2) } = 
  { x : ℝ | x ≤ 1 ∨ x ≥ 2 } :=
by
  sorry

-- 2. The proof problem for A ⊆ B to find the range of a
theorem subset_range (a : ℝ) :
  (∀ x, A a x → B x) ↔ -1 < a ∧ a ≤ 1 :=
by
  sorry

end union_complement_subset_range_l1577_157740


namespace simplify_expression_l1577_157703

theorem simplify_expression (x : ℝ) : 2 * x + 1 - (x + 1) = x := 
by 
sorry

end simplify_expression_l1577_157703


namespace gcd_of_powers_of_three_l1577_157734

theorem gcd_of_powers_of_three :
  let a := 3^1001 - 1
  let b := 3^1012 - 1
  gcd a b = 177146 := by
  sorry

end gcd_of_powers_of_three_l1577_157734


namespace calls_on_friday_l1577_157764

noncomputable def total_calls_monday := 35
noncomputable def total_calls_tuesday := 46
noncomputable def total_calls_wednesday := 27
noncomputable def total_calls_thursday := 61
noncomputable def average_calls_per_day := 40
noncomputable def number_of_days := 5
noncomputable def total_calls_week := average_calls_per_day * number_of_days

theorem calls_on_friday : 
  total_calls_week - (total_calls_monday + total_calls_tuesday + total_calls_wednesday + total_calls_thursday) = 31 :=
by
  sorry

end calls_on_friday_l1577_157764


namespace age_difference_l1577_157787

-- Defining the necessary variables and their types
variables (A B : ℕ)

-- Given conditions: 
axiom B_current_age : B = 38
axiom future_age_relationship : A + 10 = 2 * (B - 10)

-- Proof goal statement
theorem age_difference : A - B = 8 :=
by
  sorry

end age_difference_l1577_157787


namespace complement_A_in_U_range_of_a_l1577_157712

open Set Real

noncomputable def U : Set ℝ := univ
noncomputable def f (x : ℝ) : ℝ := (1 / (sqrt (x + 2))) + log (3 - x)
noncomputable def A : Set ℝ := {x | -2 < x ∧ x < 3}
noncomputable def B (a : ℝ) : Set ℝ := {x | a < x ∧ x < (2 * a - 1)}

theorem complement_A_in_U : compl A = {x | x ≤ -2 ∨ 3 ≤ x} :=
by {
  sorry
}

theorem range_of_a (a : ℝ) (h : A ∪ B a = A) : a ∈ Iic 2 :=
by {
  sorry
}

end complement_A_in_U_range_of_a_l1577_157712


namespace polynomial_factorization_l1577_157722

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 1) * (x^2 + 6*x + 37) :=
  sorry

end polynomial_factorization_l1577_157722


namespace tangerine_count_l1577_157756

def initial_tangerines : ℕ := 10
def added_tangerines : ℕ := 6

theorem tangerine_count : initial_tangerines + added_tangerines = 16 :=
by
  sorry

end tangerine_count_l1577_157756


namespace two_a7_minus_a8_l1577_157747

variable (a : ℕ → ℝ) -- Assuming the arithmetic sequence {a_n} is a sequence of real numbers

-- Definitions and conditions of the problem
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

axiom a1_plus_3a6_plus_a11 : a 1 + 3 * (a 6) + a 11 = 120

-- The theorem to be proved
theorem two_a7_minus_a8 (h : is_arithmetic_sequence a) : 2 * a 7 - a 8 = 24 := 
sorry

end two_a7_minus_a8_l1577_157747


namespace total_spokes_in_garage_l1577_157762

-- Definitions based on the problem conditions
def num_bicycles : ℕ := 4
def spokes_per_wheel : ℕ := 10
def wheels_per_bicycle : ℕ := 2

-- The goal is to prove the total number of spokes
theorem total_spokes_in_garage : (num_bicycles * wheels_per_bicycle * spokes_per_wheel) = 80 :=
by
    sorry

end total_spokes_in_garage_l1577_157762


namespace eval_expression_correct_l1577_157798

noncomputable def eval_expression : ℝ := (-64)^(4/3)

theorem eval_expression_correct : eval_expression = 256 := by
  sorry

end eval_expression_correct_l1577_157798


namespace translation_correct_l1577_157799

-- Define the first line l1
def l1 (x : ℝ) : ℝ := 2 * x - 2

-- Define the second line l2
def l2 (x : ℝ) : ℝ := 2 * x

-- State the theorem
theorem translation_correct :
  ∀ x : ℝ, l2 x = l1 x + 2 :=
by
  intro x
  unfold l1 l2
  sorry

end translation_correct_l1577_157799


namespace dark_squares_exceed_light_squares_by_one_l1577_157752

theorem dark_squares_exceed_light_squares_by_one 
  (m n : ℕ) (h_m : m = 9) (h_n : n = 9) (h_total_squares : m * n = 81) :
  let dark_squares := 5 * 5 + 4 * 4
  let light_squares := 5 * 4 + 4 * 5
  dark_squares - light_squares = 1 :=
by {
  sorry
}

end dark_squares_exceed_light_squares_by_one_l1577_157752


namespace problem1_l1577_157771

theorem problem1 (a : ℝ) (h : Real.sqrt a + 1 / Real.sqrt a = 3) :
  (a ^ 2 + 1 / a ^ 2 + 3) / (4 * a + 1 / (4 * a)) = 10 * Real.sqrt 5 := sorry

end problem1_l1577_157771


namespace largest_number_Ahn_can_get_l1577_157767

theorem largest_number_Ahn_can_get :
  ∃ (n : ℕ), (100 ≤ n ∧ n ≤ 999) ∧ (∀ m, (100 ≤ m ∧ m ≤ 999) → 3 * (500 - m) ≤ 1200) := sorry

end largest_number_Ahn_can_get_l1577_157767


namespace lucky_lucy_l1577_157730

theorem lucky_lucy (a b c d e : ℤ)
  (ha : a = 2)
  (hb : b = 4)
  (hc : c = 6)
  (hd : d = 8)
  (he : a + b - c + d - e = a + (b - (c + (d - e)))) :
  e = 8 :=
by
  rw [ha, hb, hc, hd] at he
  exact eq_of_sub_eq_zero (by linarith)

end lucky_lucy_l1577_157730


namespace volleyball_practice_start_time_l1577_157720

def homework_time := 1 * 60 + 59  -- convert 1:59 p.m. to minutes since 12:00 p.m.
def homework_duration := 96        -- duration in minutes
def buffer_time := 25              -- time between finishing homework and practice
def practice_start_time := 4 * 60  -- convert 4:00 p.m. to minutes since 12:00 p.m.

theorem volleyball_practice_start_time :
  homework_time + homework_duration + buffer_time = practice_start_time := 
by
  sorry

end volleyball_practice_start_time_l1577_157720


namespace books_in_special_collection_at_beginning_of_month_l1577_157714

theorem books_in_special_collection_at_beginning_of_month
  (loaned_out_real : Real)
  (loaned_out_books : Int)
  (returned_ratio : Real)
  (books_at_end : Int)
  (B : Int)
  (h1 : loaned_out_real = 49.99999999999999)
  (h2 : loaned_out_books = 50)
  (h3 : returned_ratio = 0.70)
  (h4 : books_at_end = 60)
  (h5 : loaned_out_books = Int.floor loaned_out_real)
  (h6 : ∀ (loaned_books : Int), loaned_books ≤ loaned_out_books → returned_ratio * loaned_books + (loaned_books - returned_ratio * loaned_books) = loaned_books)
  : B = 75 :=
by
  sorry

end books_in_special_collection_at_beginning_of_month_l1577_157714


namespace find_a_plus_2b_l1577_157758

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 + 3 * x^2 - 6 * a * x + b

noncomputable def f' (a b x : ℝ) : ℝ := 3 * a * x^2 + 6 * x - 6 * a

theorem find_a_plus_2b (a b : ℝ) 
  (h1 : f' a b 2 = 0)
  (h2 : f a b 2 = 9) : a + 2 * b = -24 := 
by sorry

end find_a_plus_2b_l1577_157758


namespace find_function_expression_l1577_157769

noncomputable def f (x : ℝ) : ℝ := x^2 - 5*x + 7

theorem find_function_expression (x : ℝ) :
  (∀ x : ℝ, f (x + 2) = x^2 - x + 1) →
  f x = x^2 - 5*x + 7 :=
by
  intro h
  sorry

end find_function_expression_l1577_157769


namespace find_BE_l1577_157700

-- Definitions from the conditions
variable {A B C D E : Point}
variable (AB BC CA BD BE CE : ℝ)
variable (angleBAE angleCAD : Real.Angle)

-- Given conditions
axiom h1 : AB = 12
axiom h2 : BC = 17
axiom h3 : CA = 15
axiom h4 : BD = 7
axiom h5 : angleBAE = angleCAD

-- Required proof statement
theorem find_BE :
  BE = 1632 / 201 := by
  sorry

end find_BE_l1577_157700
