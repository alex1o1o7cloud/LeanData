import Mathlib

namespace part1_part2_l1656_165662

noncomputable def f (x : ℝ) : ℝ := abs (x + 1) - abs (x - 2)

theorem part1 : 
  {x : ℝ | f x ≥ 1} = {x : ℝ | x ≥ 1} :=
sorry 

noncomputable def g (x : ℝ) : ℝ := f x - x^2 + x

theorem part2 (m : ℝ) : 
  (∃ x : ℝ, f x ≥ x^2 - x + m) → m ≤ 5/4 :=
sorry 

end part1_part2_l1656_165662


namespace exists_acute_triangle_l1656_165617

theorem exists_acute_triangle (a b c d e : ℝ) 
  (h1 : a ≤ b) (h2 : b ≤ c) (h3 : c ≤ d) (h4 : d ≤ e)
  (h_triangle_abc : a + b > c) (h_triangle_abd : a + b > d) (h_triangle_abe : a + b > e)
  (h_triangle_bcd : b + c > d) (h_triangle_bce : b + c > e) (h_triangle_cde : c + d > e)
  (h_triangle_abc2 : a + c > b) (h_triangle_abd2 : a + d > b) (h_triangle_abe2 : a + e > b)
  (h_triangle_bcd2 : b + d > c) (h_triangle_bce2 : b + e > c) (h_triangle_cde2 : c + e > d)
  (h_triangle_abc3 : b + c > a) (h_triangle_abd3 : b + d > a) (h_triangle_abe3 : b + e > a)
  (h_triangle_bcd3 : b + d > a) (h_triangle_bce3 : c + e > a) (h_triangle_cde3 : d + e > c) :
  ∃ x y z : ℝ, (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) ∧ 
              (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e) ∧ 
              (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e) ∧ 
              (x ≠ y) ∧ (y ≠ z) ∧ (x ≠ z) ∧
              x + y > z ∧ 
              ¬ (x^2 + y^2 ≤ z^2) :=
by
  sorry

end exists_acute_triangle_l1656_165617


namespace smallest_b_greater_than_1_l1656_165627

def g (x : ℕ) : ℕ :=
  if x % 35 = 0 then x / 35
  else if x % 7 = 0 then 5 * x
  else if x % 5 = 0 then 7 * x
  else x + 5

def g_iter (n : ℕ) (x : ℕ) : ℕ := Nat.iterate g n x

theorem smallest_b_greater_than_1 (b : ℕ) :
  (b > 1) → 
  g_iter 1 3 = 8 ∧ g_iter b 3 = 8 →
  b = 21 := by
  sorry

end smallest_b_greater_than_1_l1656_165627


namespace lines_parallel_if_perpendicular_to_same_plane_l1656_165607

-- Definitions and conditions
variables {Point : Type*} [MetricSpace Point]
variables {Line Plane : Type*}

def is_parallel (l₁ l₂ : Line) : Prop := sorry
def is_perpendicular (l : Line) (p : Plane) : Prop := sorry

variables (m n : Line) (α : Plane)

-- Theorem statement
theorem lines_parallel_if_perpendicular_to_same_plane :
  is_perpendicular m α → is_perpendicular n α → is_parallel m n :=
sorry

end lines_parallel_if_perpendicular_to_same_plane_l1656_165607


namespace james_hears_beats_per_week_l1656_165636

theorem james_hears_beats_per_week
  (beats_per_minute : ℕ)
  (hours_per_day : ℕ)
  (days_per_week : ℕ)
  (H1 : beats_per_minute = 200)
  (H2 : hours_per_day = 2)
  (H3 : days_per_week = 7) :
  beats_per_minute * hours_per_day * 60 * days_per_week = 168000 := 
by
  -- sorry proof step placeholder
  sorry

end james_hears_beats_per_week_l1656_165636


namespace spaceship_speed_conversion_l1656_165666

theorem spaceship_speed_conversion (speed_km_per_sec : ℕ) (seconds_in_hour : ℕ) (correct_speed_km_per_hour : ℕ) :
  speed_km_per_sec = 12 →
  seconds_in_hour = 3600 →
  correct_speed_km_per_hour = 43200 →
  speed_km_per_sec * seconds_in_hour = correct_speed_km_per_hour := by
  sorry

end spaceship_speed_conversion_l1656_165666


namespace seats_taken_correct_l1656_165678

-- Define the conditions
def rows := 40
def chairs_per_row := 20
def unoccupied_seats := 10

-- Define the total number of seats
def total_seats := rows * chairs_per_row

-- Define the number of seats taken
def seats_taken := total_seats - unoccupied_seats

-- Statement of our math proof problem
theorem seats_taken_correct : seats_taken = 790 := by
  sorry

end seats_taken_correct_l1656_165678


namespace geom_seq_general_term_arith_seq_sum_l1656_165653

theorem geom_seq_general_term (q : ℕ → ℕ) (a_1 a_2 a_3 : ℕ) (h1 : a_1 = 2)
  (h2 : (a_1 + a_3) / 2 = a_2 + 1) (h3 : a_2 = q 2) (h4 : a_3 = q 3)
  (g : ℕ → ℕ) (Sn : ℕ → ℕ) (gen_term : ∀ n, q n = 2^n) (sum_term : ∀ n, Sn n = 2^(n+1) - 2) :
  q n = g n :=
sorry

theorem arith_seq_sum (a_1 a_2 a_4 : ℕ) (b : ℕ → ℕ) (Tn : ℕ → ℕ) (h1 : a_1 = 2)
  (h2 : a_2 = 4) (h3 : a_4 = 16) (h4 : b 2 = a_1) (h5 : b 8 = a_2 + a_4)
  (gen_term : ∀ n, b n = 1 + 3 * (n - 1)) (sum_term : ∀ n, Tn n = (3 * n^2 - n) / 2) :
  Tn n = (3 * n^2 - 1) / 2 :=
sorry

end geom_seq_general_term_arith_seq_sum_l1656_165653


namespace percentage_pine_cones_on_roof_l1656_165682

theorem percentage_pine_cones_on_roof 
  (num_trees : Nat) 
  (pine_cones_per_tree : Nat) 
  (pine_cone_weight_oz : Nat) 
  (total_pine_cone_weight_on_roof_oz : Nat) 
  : num_trees = 8 ∧ pine_cones_per_tree = 200 ∧ pine_cone_weight_oz = 4 ∧ total_pine_cone_weight_on_roof_oz = 1920 →
    (total_pine_cone_weight_on_roof_oz / pine_cone_weight_oz) / (num_trees * pine_cones_per_tree) * 100 = 30 := 
by
  sorry

end percentage_pine_cones_on_roof_l1656_165682


namespace ratio_AR_AU_l1656_165616

-- Define the conditions in the problem as variables and constraints
variables (A B C P Q U R : Type)
variables (AP PB AQ QC : ℝ)
variables (angle_bisector_AU : A -> U)
variables (intersect_AU_PQ_at_R : A -> U -> P -> Q -> R)

-- Assuming the given distances
def conditions (AP PB AQ QC : ℝ) : Prop :=
  AP = 2 ∧ PB = 6 ∧ AQ = 4 ∧ QC = 5

-- The statement to prove
theorem ratio_AR_AU (h : conditions AP PB AQ QC) : 
  (AR / AU) = 108 / 289 :=
sorry

end ratio_AR_AU_l1656_165616


namespace TriangleInscribedAngle_l1656_165665

theorem TriangleInscribedAngle
  (x : ℝ)
  (arc_PQ : ℝ := x + 100)
  (arc_QR : ℝ := 2 * x + 50)
  (arc_RP : ℝ := 3 * x - 40)
  (angle_sum_eq_360 : arc_PQ + arc_QR + arc_RP = 360) :
  ∃ angle_PQR : ℝ, angle_PQR = 70.84 := 
sorry

end TriangleInscribedAngle_l1656_165665


namespace sequence_term_l1656_165628

noncomputable def geometric_sum (n : ℕ) : ℝ :=
  2 * (1 - (1 / 2) ^ n) / (1 - 1 / 2)

theorem sequence_term (m n : ℕ) (h : n < m) : 
  let Sn := geometric_sum n
  let Sn_plus_1 := geometric_sum (n + 1)
  Sn - Sn_plus_1 = -(1 / 2 ^ (n - 1)) := sorry

end sequence_term_l1656_165628


namespace valid_range_of_x_l1656_165651

theorem valid_range_of_x (x : ℝ) : 3 * x + 5 ≥ 0 → x ≥ -5 / 3 := 
by
  sorry

end valid_range_of_x_l1656_165651


namespace child_ticket_cost_l1656_165621

noncomputable def cost_of_child_ticket : ℝ := 3.50

theorem child_ticket_cost
  (adult_ticket_price : ℝ)
  (total_tickets : ℕ)
  (total_cost : ℝ)
  (adult_tickets_bought : ℕ)
  (adult_ticket_price_eq : adult_ticket_price = 5.50)
  (total_tickets_bought_eq : total_tickets = 21)
  (total_cost_eq : total_cost = 83.50)
  (adult_tickets_count : adult_tickets_bought = 5) :
  cost_of_child_ticket = 3.50 :=
by
  sorry

end child_ticket_cost_l1656_165621


namespace number_of_proper_subsets_of_P_l1656_165630

theorem number_of_proper_subsets_of_P (P : Set ℝ) (hP : P = {x | x^2 = 1}) : 
  (∃ n, n = 2 ∧ ∃ k, k = 2 ^ n - 1 ∧ k = 3) :=
by
  sorry

end number_of_proper_subsets_of_P_l1656_165630


namespace pyramid_certain_height_l1656_165698

noncomputable def certain_height (h : ℝ) : Prop :=
  let height := h + 20
  let width := height + 234
  (height + width = 1274) → h = 1000 / 3

theorem pyramid_certain_height (h : ℝ) : certain_height h :=
by
  let height := h + 20
  let width := height + 234
  have h_eq : (height + width = 1274) → h = 1000 / 3 := sorry
  exact h_eq

end pyramid_certain_height_l1656_165698


namespace age_problem_l1656_165670

variables (a b c : ℕ)

theorem age_problem (h₁ : a = b + 2) (h₂ : b = 2 * c) (h₃ : a + b + c = 27) : b = 10 :=
by {
  -- Interactive proof steps can go here.
  sorry
}

end age_problem_l1656_165670


namespace Maria_height_in_meters_l1656_165692

theorem Maria_height_in_meters :
  let inch_to_cm := 2.54
  let cm_to_m := 0.01
  let height_in_inch := 54
  let height_in_cm := height_in_inch * inch_to_cm
  let height_in_m := height_in_cm * cm_to_m
  let rounded_height_in_m := Float.round (height_in_m * 1000) / 1000
  rounded_height_in_m = 1.372 := 
by
  sorry

end Maria_height_in_meters_l1656_165692


namespace worker_assignment_l1656_165676

theorem worker_assignment (x : ℕ) (y : ℕ) 
  (h1 : x + y = 90)
  (h2 : 2 * 15 * x = 3 * 8 * y) : 
  (x = 40 ∧ y = 50) := by
  sorry

end worker_assignment_l1656_165676


namespace baseEight_conversion_l1656_165611

-- Base-eight number is given as 1563
def baseEight : Nat := 1563

-- Function to convert a base-eight number to base-ten
noncomputable def baseEightToBaseTen (n : Nat) : Nat :=
  let digit3 := (n / 1000) % 10
  let digit2 := (n / 100) % 10
  let digit1 := (n / 10) % 10
  let digit0 := n % 10
  digit3 * 8^3 + digit2 * 8^2 + digit1 * 8^1 + digit0 * 8^0

theorem baseEight_conversion :
  baseEightToBaseTen baseEight = 883 := by
  sorry

end baseEight_conversion_l1656_165611


namespace johns_total_cost_l1656_165652

variable (C_s C_d : ℝ)

theorem johns_total_cost (h_s : C_s = 20) (h_d : C_d = 0.5 * C_s) : C_s + C_d = 30 := by
  sorry

end johns_total_cost_l1656_165652


namespace no_factors_multiple_of_210_l1656_165620

theorem no_factors_multiple_of_210 (n : ℕ) (h : n = 2^12 * 3^18 * 5^10) : ∀ d : ℕ, d ∣ n → ¬ (210 ∣ d) :=
by
  sorry

end no_factors_multiple_of_210_l1656_165620


namespace abs_x_lt_2_sufficient_not_necessary_for_x_sq_minus_x_minus_6_lt_0_l1656_165609

theorem abs_x_lt_2_sufficient_not_necessary_for_x_sq_minus_x_minus_6_lt_0 :
  (∀ x : ℝ, |x| < 2 → x^2 - x - 6 < 0) ∧ (¬ ∀ x : ℝ, x^2 - x - 6 < 0 → |x| < 2) :=
by
  sorry

end abs_x_lt_2_sufficient_not_necessary_for_x_sq_minus_x_minus_6_lt_0_l1656_165609


namespace smallest_possible_product_l1656_165686

def digits : Set ℕ := {2, 4, 5, 8}

def is_valid_pair (a b : ℤ) : Prop :=
  let (d1, d2, d3, d4) := (a / 10, a % 10, b / 10, b % 10)
  {d1.toNat, d2.toNat, d3.toNat, d4.toNat} ⊆ digits ∧ {d1.toNat, d2.toNat, d3.toNat, d4.toNat} = digits

def smallest_product : ℤ :=
  1200

theorem smallest_possible_product :
  ∀ (a b : ℤ), is_valid_pair a b → a * b ≥ smallest_product :=
by
  intro a b h
  sorry

end smallest_possible_product_l1656_165686


namespace same_solution_sets_l1656_165680

theorem same_solution_sets (a : ℝ) :
  (∀ x : ℝ, 3 * x - 5 < a ↔ 2 * x < 4) → a = 1 := 
by
  sorry

end same_solution_sets_l1656_165680


namespace mom_has_enough_money_l1656_165693

def original_price : ℝ := 268
def discount_rate : ℝ := 0.2
def money_brought : ℝ := 230
def discounted_price := original_price * (1 - discount_rate)

theorem mom_has_enough_money : money_brought ≥ discounted_price := by
  sorry

end mom_has_enough_money_l1656_165693


namespace remainder_196c_2008_mod_97_l1656_165646

theorem remainder_196c_2008_mod_97 (c : ℤ) : ((196 * c) ^ 2008) % 97 = 44 := by
  sorry

end remainder_196c_2008_mod_97_l1656_165646


namespace households_selected_l1656_165645

theorem households_selected (H : ℕ) (M L S n h : ℕ)
  (h1 : H = 480)
  (h2 : M = 200)
  (h3 : L = 160)
  (h4 : H = M + L + S)
  (h5 : h = 6)
  (h6 : (h : ℚ) / n = (S : ℚ) / H) : n = 24 :=
by
  sorry

end households_selected_l1656_165645


namespace student_chose_number_l1656_165625

theorem student_chose_number (x : ℤ) (h : 2 * x - 138 = 104) : x = 121 := by
  sorry

end student_chose_number_l1656_165625


namespace acute_triangle_condition_l1656_165673

theorem acute_triangle_condition (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A > 0) (h3 : B > 0) (h4 : C > 0)
    (h5 : A + B > 90) (h6 : B + C > 90) (h7 : C + A > 90) : A < 90 ∧ B < 90 ∧ C < 90 :=
sorry

end acute_triangle_condition_l1656_165673


namespace fish_speed_in_still_water_l1656_165614

theorem fish_speed_in_still_water (u d : ℕ) (v : ℕ) : 
  u = 35 → d = 55 → 2 * v = u + d → v = 45 := 
by 
  intros h1 h2 h3
  rw [h1, h2] at h3
  linarith

end fish_speed_in_still_water_l1656_165614


namespace truck_loading_time_l1656_165654

theorem truck_loading_time (h1_rate h2_rate h3_rate : ℝ)
  (h1 : h1_rate = 1 / 5) (h2 : h2_rate = 1 / 4) (h3 : h3_rate = 1 / 6) :
  (1 / (h1_rate + h2_rate + h3_rate)) = 60 / 37 :=
by simp [h1, h2, h3]; sorry

end truck_loading_time_l1656_165654


namespace bobby_initial_candy_count_l1656_165633

theorem bobby_initial_candy_count (x : ℕ) (h : x + 17 = 43) : x = 26 :=
by
  sorry

end bobby_initial_candy_count_l1656_165633


namespace speed_of_journey_l1656_165658

-- Define the conditions
def journey_time : ℕ := 10
def journey_distance : ℕ := 200
def half_journey_distance : ℕ := journey_distance / 2

-- Define the hypothesis that the journey is split into two equal parts, each traveled at the same speed
def equal_speed (v : ℕ) : Prop :=
  (half_journey_distance / v) + (half_journey_distance / v) = journey_time

-- Prove the speed v is 20 km/hr given the conditions
theorem speed_of_journey : ∃ v : ℕ, equal_speed v ∧ v = 20 :=
by
  have h : equal_speed 20 := sorry
  exact ⟨20, h, rfl⟩

end speed_of_journey_l1656_165658


namespace lisa_total_miles_flown_l1656_165641

-- Definitions based on given conditions
def distance_per_trip : ℝ := 256.0
def number_of_trips : ℝ := 32.0
def total_miles_flown : ℝ := 8192.0

-- Lean statement asserting the equivalence
theorem lisa_total_miles_flown : 
    (distance_per_trip * number_of_trips = total_miles_flown) :=
by 
    sorry

end lisa_total_miles_flown_l1656_165641


namespace average_speed_of_participant_l1656_165604

noncomputable def average_speed (d : ℝ) : ℝ :=
  let total_distance := 4 * d
  let total_time := (d / 6) + (d / 12) + (d / 18) + (d / 24)
  total_distance / total_time

theorem average_speed_of_participant :
  ∀ (d : ℝ), d > 0 → average_speed d = 11.52 :=
by
  intros d hd
  unfold average_speed
  sorry

end average_speed_of_participant_l1656_165604


namespace sub_neg_seven_eq_neg_fourteen_l1656_165606

theorem sub_neg_seven_eq_neg_fourteen : (-7) - 7 = -14 := 
  by
  sorry

end sub_neg_seven_eq_neg_fourteen_l1656_165606


namespace not_possible_coloring_l1656_165610

def color : Nat → Option ℕ := sorry

def all_colors_used (f : Nat → Option ℕ) : Prop := 
  (∃ n, f n = some 0) ∧ (∃ n, f n = some 1) ∧ (∃ n, f n = some 2)

def valid_coloring (f : Nat → Option ℕ) : Prop :=
  ∀ (a b : Nat), 1 < a → 1 < b → f a ≠ f b → f (a * b) ≠ f a ∧ f (a * b) ≠ f b

theorem not_possible_coloring : ¬ (∃ f : Nat → Option ℕ, all_colors_used f ∧ valid_coloring f) := 
sorry

end not_possible_coloring_l1656_165610


namespace boat_speed_l1656_165696

theorem boat_speed (b s : ℝ) (h1 : b + s = 11) (h2 : b - s = 7) : b = 9 :=
by
  sorry

end boat_speed_l1656_165696


namespace find_greater_number_l1656_165615

-- Define the two numbers x and y
variables (x y : ℕ)

-- Conditions
theorem find_greater_number (h1 : x + y = 36) (h2 : x - y = 12) : x = 24 := 
by
  sorry

end find_greater_number_l1656_165615


namespace gcd_relatively_prime_l1656_165672

theorem gcd_relatively_prime (a : ℤ) (m n : ℕ) (h_odd : a % 2 = 1) (h_pos_m : m > 0) (h_pos_n : n > 0) (h_diff : n ≠ m) :
  Int.gcd (a ^ 2^m + 2 ^ 2^m) (a ^ 2^n + 2 ^ 2^n) = 1 :=
by
  sorry

end gcd_relatively_prime_l1656_165672


namespace cost_price_of_a_ball_l1656_165605

variables (C : ℝ) (selling_price : ℝ) (cost_price_20_balls : ℝ) (loss_on_20_balls : ℝ)

def cost_price_per_ball (C : ℝ) := (20 * C - 720 = 5 * C)

theorem cost_price_of_a_ball :
  (∃ C : ℝ, 20 * C - 720 = 5 * C) -> (C = 48) := 
by
  sorry

end cost_price_of_a_ball_l1656_165605


namespace proof1_l1656_165694

def prob1 : Prop :=
  (1 : ℝ) * (Real.sqrt 45 + Real.sqrt 18) - (Real.sqrt 8 - Real.sqrt 125) = 8 * Real.sqrt 5 + Real.sqrt 2

theorem proof1 : prob1 :=
by
  sorry

end proof1_l1656_165694


namespace loss_percentage_l1656_165657

theorem loss_percentage (CP SP SP_new : ℝ) (L : ℝ) 
  (h1 : CP = 1428.57)
  (h2 : SP = CP - (L / 100 * CP))
  (h3 : SP_new = CP + 0.04 * CP)
  (h4 : SP_new = SP + 200) :
  L = 10 := by
    sorry

end loss_percentage_l1656_165657


namespace inequality_bound_l1656_165600

theorem inequality_bound 
  (a b c d : ℝ) 
  (ha : 0 ≤ a) (hb : a ≤ 1)
  (hb : 0 ≤ b) (hc : b ≤ 1)
  (hc : 0 ≤ c) (hd : c ≤ 1)
  (hd : 0 ≤ d) (ha2 : d ≤ 1) : 
  ab * (a - b) + bc * (b - c) + cd * (c - d) + da * (d - a) ≤ 8/27 := 
by
  sorry

end inequality_bound_l1656_165600


namespace rectangle_enclosing_ways_l1656_165659

/-- Given five horizontal lines and five vertical lines, the total number of ways to choose four lines (two horizontal, two vertical) such that they form a rectangle is 100 --/
theorem rectangle_enclosing_ways : 
  let horizontal_lines := [1, 2, 3, 4, 5]
  let vertical_lines := [1, 2, 3, 4, 5]
  let ways_horizontal := Nat.choose 5 2
  let ways_vertical := Nat.choose 5 2
  ways_horizontal * ways_vertical = 100 := 
by
  sorry

end rectangle_enclosing_ways_l1656_165659


namespace ball_attendance_l1656_165622

noncomputable def num_people_attending_ball (n m : ℕ) : ℕ := n + m 

theorem ball_attendance:
  ∀ (n m : ℕ), 
  n + m < 50 ∧ 
  (n - n / 4) = (5 * m / 7) →
  num_people_attending_ball n m = 41 :=
by 
  intros n m h
  have h1 : n + m < 50 := h.1
  have h2 : n - n / 4 = 5 * m / 7 := h.2
  sorry

end ball_attendance_l1656_165622


namespace shopkeeper_marked_price_l1656_165699

theorem shopkeeper_marked_price 
  (L C M S : ℝ)
  (h1 : C = 0.75 * L)
  (h2 : C = 0.75 * S)
  (h3 : S = 0.85 * M) :
  M = 1.17647 * L :=
sorry

end shopkeeper_marked_price_l1656_165699


namespace farm_transaction_difference_l1656_165656

theorem farm_transaction_difference
  (x : ℕ)
  (h_initial : 6 * x - 15 > 0) -- Ensure initial horses are enough to sell 15
  (h_ratio_initial : 6 * x = x * 6)
  (h_ratio_final : (6 * x - 15) = 3 * (x + 15)) :
  (6 * x - 15) - (x + 15) = 70 :=
by
  sorry

end farm_transaction_difference_l1656_165656


namespace mangoes_ratio_l1656_165685

theorem mangoes_ratio (a d_a : ℕ)
  (h1 : a = 60)
  (h2 : a + d_a = 75) : a / (75 - a) = 4 := by
  sorry

end mangoes_ratio_l1656_165685


namespace division_problem_l1656_165626

theorem division_problem (A : ℕ) (h : 23 = (A * 3) + 2) : A = 7 :=
sorry

end division_problem_l1656_165626


namespace sector_area_15deg_radius_6cm_l1656_165612

noncomputable def sector_area (r : ℝ) (theta : ℝ) : ℝ :=
  0.5 * theta * r^2

theorem sector_area_15deg_radius_6cm :
  sector_area 6 (15 * Real.pi / 180) = 3 * Real.pi / 2 := by
  sorry

end sector_area_15deg_radius_6cm_l1656_165612


namespace quadratic_eq_distinct_solutions_l1656_165637

theorem quadratic_eq_distinct_solutions (b : ℤ) (k : ℤ) (h1 : 1 ≤ b ∧ b ≤ 100) :
  ∃ n : ℕ, n = 27 ∧ (x^2 + (2 * b + 3) * x + b^2 = 0 →
    12 * b + 9 = k^2 → 
    (∃ m n : ℤ, x = m ∧ x = n ∧ m ≠ n)) :=
sorry

end quadratic_eq_distinct_solutions_l1656_165637


namespace rate_per_kg_for_fruits_l1656_165690

-- Definitions and conditions
def total_cost (rate_per_kg : ℝ) : ℝ := 8 * rate_per_kg + 9 * rate_per_kg

def total_paid : ℝ := 1190

theorem rate_per_kg_for_fruits : ∃ R : ℝ, total_cost R = total_paid ∧ R = 70 :=
by
  sorry

end rate_per_kg_for_fruits_l1656_165690


namespace speed_of_boat_in_still_water_l1656_165634

variable (b s : ℝ) -- Speed of the boat in still water and speed of the stream

-- Condition 1: The boat goes 9 km along the stream in 1 hour
def boat_along_stream := b + s = 9

-- Condition 2: The boat goes 5 km against the stream in 1 hour
def boat_against_stream := b - s = 5

-- Theorem to prove: The speed of the boat in still water is 7 km/hr
theorem speed_of_boat_in_still_water : boat_along_stream b s → boat_against_stream b s → b = 7 := 
by
  sorry

end speed_of_boat_in_still_water_l1656_165634


namespace regular_18gon_lines_rotational_symmetry_sum_l1656_165671

def L : ℕ := 18
def R : ℕ := 20

theorem regular_18gon_lines_rotational_symmetry_sum : L + R = 38 :=
by 
  sorry

end regular_18gon_lines_rotational_symmetry_sum_l1656_165671


namespace bar_weight_calc_l1656_165619

variable (blue_weight green_weight num_blue_weights num_green_weights bar_weight total_weight : ℕ)

theorem bar_weight_calc
  (h1 : blue_weight = 2)
  (h2 : green_weight = 3)
  (h3 : num_blue_weights = 4)
  (h4 : num_green_weights = 5)
  (h5 : total_weight = 25)
  (weights_total := num_blue_weights * blue_weight + num_green_weights * green_weight)
  : bar_weight = total_weight - weights_total :=
by
  sorry

end bar_weight_calc_l1656_165619


namespace example_theorem_l1656_165640

def not_a_term : Prop := ∀ n : ℕ, ¬ (24 - 2 * n = 3)

theorem example_theorem : not_a_term :=
  by sorry

end example_theorem_l1656_165640


namespace compare_exponents_product_of_roots_l1656_165649

noncomputable def f (x : ℝ) (a : ℝ) := (Real.log x) / (x + a)

theorem compare_exponents : (2016 : ℝ) ^ 2017 > (2017 : ℝ) ^ 2016 :=
sorry

theorem product_of_roots (x1 x2 : ℝ) (h1 : x1 ≠ x2) (h2 : f x1 0 = k) (h3 : f x2 0 = k) : 
  x1 * x2 > Real.exp 2 :=
sorry

end compare_exponents_product_of_roots_l1656_165649


namespace tangent_line_eq_extreme_values_range_of_a_l1656_165613

noncomputable def f (x : ℝ) (a: ℝ) : ℝ := x^2 - a * Real.log x

-- (I) Proving the tangent line equation is y = x for a = 1 at x = 1.
theorem tangent_line_eq (h : ∀ x, f x 1 = x^2 - Real.log x) :
  ∃ y : (ℝ → ℝ), y = id ∧ y 1 = x :=
sorry

-- (II) Proving extreme values of the function f(x).
theorem extreme_values (a: ℝ) :
  (∃ x_min : ℝ, f x_min a = (a/2) - (a/2) * Real.log (a/2)) ∧ 
  (∀ x, ¬∃ x_max : ℝ, f x_max a > f x a) :=
sorry

-- (III) Proving the range of values for a.
theorem range_of_a :
  (∀ x, 2*x - (a/x) ≥ 0 → 2 < x) → a ≤ 8 :=
sorry

end tangent_line_eq_extreme_values_range_of_a_l1656_165613


namespace residue_of_neg_1235_mod_29_l1656_165679

theorem residue_of_neg_1235_mod_29 : 
  ∃ r, 0 ≤ r ∧ r < 29 ∧ (-1235) % 29 = r ∧ r = 12 :=
by
  sorry

end residue_of_neg_1235_mod_29_l1656_165679


namespace roger_steps_time_l1656_165603

theorem roger_steps_time (steps_per_30_min : ℕ := 2000) (time_for_2000_steps : ℕ := 30) (goal_steps : ℕ := 10000) : 
  (goal_steps * time_for_2000_steps) / steps_per_30_min = 150 :=
by 
  -- This is the statement. Proof is omitted as per instruction.
  sorry

end roger_steps_time_l1656_165603


namespace dot_product_ABC_l1656_165663

-- Defining vectors as pairs of real numbers
def vector := (ℝ × ℝ)

-- Defining the vectors AB and AC
def AB : vector := (1, 0)
def AC : vector := (-2, 3)

-- Definition of vector subtraction
def vector_sub (v1 v2 : vector) : vector := (v1.1 - v2.1, v1.2 - v2.2)

-- Definition of dot product
def dot_product (v1 v2 : vector) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Define vector BC using the given vectors AB and AC
def BC : vector := vector_sub AC AB

-- The theorem stating the desired dot product result
theorem dot_product_ABC : dot_product AB BC = -3 := by
  sorry

end dot_product_ABC_l1656_165663


namespace minimum_value_of_function_l1656_165623

theorem minimum_value_of_function (x : ℝ) (h : x * Real.log 2 / Real.log 3 ≥ 1) : 
  ∃ t : ℝ, t = 2^x ∧ t ≥ 3 ∧ ∀ y : ℝ, y = t^2 - 2*t - 3 → y = (t-1)^2 - 4 := 
sorry

end minimum_value_of_function_l1656_165623


namespace trig_identity_l1656_165655

theorem trig_identity (α : ℝ) (h : Real.tan α = 1/3) :
  Real.cos α ^ 2 + Real.cos (Real.pi / 2 + 2 * α) = 3 / 10 := 
sorry

end trig_identity_l1656_165655


namespace expectation_of_binomial_l1656_165674

noncomputable def binomial_expectation (n : ℕ) (p : ℝ) : ℝ := n * p

theorem expectation_of_binomial :
  binomial_expectation 6 (1/3) = 2 :=
by
  sorry

end expectation_of_binomial_l1656_165674


namespace solve_for_x_l1656_165602

noncomputable def proof (x : ℚ) : Prop :=
  (x + 6) / (x - 4) = (x - 7) / (x + 2)

theorem solve_for_x (x : ℚ) (h : proof x) : x = 16 / 19 :=
by
  sorry

end solve_for_x_l1656_165602


namespace number_of_ways_to_tile_dominos_l1656_165684

-- Define the dimensions of the shapes and the criteria for the tiling problem
def L_shaped_area := 24
def size_of_square := 4
def size_of_rectangles := 2 * 10
def number_of_ways_to_tile := 208

-- Theorem statement
theorem number_of_ways_to_tile_dominos :
  (L_shaped_area = size_of_square + size_of_rectangles) →
  number_of_ways_to_tile = 208 :=
by
  intros h
  sorry

end number_of_ways_to_tile_dominos_l1656_165684


namespace no_base_makes_131b_square_l1656_165601

theorem no_base_makes_131b_square : ∀ (b : ℤ), b > 3 → ∀ (n : ℤ), n * n ≠ b^2 + 3 * b + 1 :=
by
  intros b h_gt_3 n
  sorry

end no_base_makes_131b_square_l1656_165601


namespace parabola_focus_distance_x_l1656_165667

theorem parabola_focus_distance_x (x y : ℝ) :
  y^2 = 4 * x ∧ y^2 = 4 * (x^2 + 5^2) → x = 4 :=
by
  sorry

end parabola_focus_distance_x_l1656_165667


namespace total_profit_correct_l1656_165687

noncomputable def total_profit (Cp Cq Cr Tp : ℝ) (h1 : 4 * Cp = 6 * Cq) (h2 : 6 * Cq = 10 * Cr) (hR : 900 = (6 / (15 + 10 + 6)) * Tp) : ℝ := Tp

theorem total_profit_correct (Cp Cq Cr Tp : ℝ) (h1 : 4 * Cp = 6 * Cq) (h2 : 6 * Cq = 10 * Cr) (hR : 900 = (6 / (15 + 10 + 6)) * Tp) : 
  total_profit Cp Cq Cr Tp h1 h2 hR = 4650 :=
sorry

end total_profit_correct_l1656_165687


namespace numberOfKidsInOtherClass_l1656_165675

-- Defining the conditions as given in the problem
def kidsInSwansonClass := 25
def averageZitsSwansonClass := 5
def averageZitsOtherClass := 6
def additionalZitsInOtherClass := 67

-- Total number of zits in Ms. Swanson's class
def totalZitsSwansonClass := kidsInSwansonClass * averageZitsSwansonClass

-- Total number of zits in the other class
def totalZitsOtherClass := totalZitsSwansonClass + additionalZitsInOtherClass

-- Proof that the number of kids in the other class is 32
theorem numberOfKidsInOtherClass : 
  (totalZitsOtherClass / averageZitsOtherClass = 32) :=
by
  -- Proof is left as an exercise.
  sorry

end numberOfKidsInOtherClass_l1656_165675


namespace second_solution_percentage_l1656_165697

theorem second_solution_percentage (P : ℝ) : 
  (28 * 0.30 + 12 * P = 40 * 0.45) → P = 0.8 :=
by
  intros h
  sorry

end second_solution_percentage_l1656_165697


namespace find_inverse_value_l1656_165650

noncomputable def f (x : ℝ) : ℝ := sorry

axiom even_function (x : ℝ) : f x = f (-x)
axiom periodic (x : ℝ) : f (x - 1) = f (x + 3)
axiom defined_interval (x : ℝ) (h : 4 ≤ x ∧ x ≤ 5) : f x = 2 ^ x + 1

noncomputable def f_inv : ℝ → ℝ := sorry
axiom inverse_defined (x : ℝ) (h : -2 ≤ x ∧ x ≤ 0) : f (f_inv x) = x

theorem find_inverse_value : f_inv 19 = 3 - 2 * (Real.log 3 / Real.log 2) := by
  sorry

end find_inverse_value_l1656_165650


namespace range_of_a_l1656_165635

theorem range_of_a (a : ℝ) : |a - 1| + |a - 4| = 3 ↔ 1 ≤ a ∧ a ≤ 4 :=
sorry

end range_of_a_l1656_165635


namespace cross_section_area_correct_l1656_165624

noncomputable def area_of_cross_section (a : ℝ) : ℝ :=
  (3 * a^2 * Real.sqrt 11) / 16

theorem cross_section_area_correct (a : ℝ) (h : 0 < a) :
  area_of_cross_section a = (3 * a^2 * Real.sqrt 11) / 16 := by
  sorry

end cross_section_area_correct_l1656_165624


namespace simplification_and_evaluation_l1656_165632

theorem simplification_and_evaluation (a : ℚ) (h : a = -1 / 2) :
  (3 * a + 2) * (a - 1) - 4 * a * (a + 1) = 1 / 4 := 
by
  sorry

end simplification_and_evaluation_l1656_165632


namespace angle_part_a_angle_part_b_l1656_165689

noncomputable def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

noncomputable def magnitude (a : ℝ × ℝ) : ℝ :=
  Real.sqrt (a.1^2 + a.2^2)

noncomputable def angle_between_vectors (a b : ℝ × ℝ) : ℝ :=
  Real.arccos ((dot_product a b) / (magnitude a * magnitude b))

theorem angle_part_a :
  angle_between_vectors (4, 0) (2, -2) = Real.arccos (Real.sqrt 2 / 2) :=
by
  sorry

theorem angle_part_b :
  angle_between_vectors (5, -3) (3, 5) = Real.pi / 2 :=
by
  sorry

end angle_part_a_angle_part_b_l1656_165689


namespace cost_of_article_l1656_165695

theorem cost_of_article 
    (C G : ℝ) 
    (h1 : 340 = C + G) 
    (h2 : 350 = C + G + 0.05 * G) 
    : C = 140 :=
by
    -- We do not need to provide the proof; 'sorry' is sufficient.
    sorry

end cost_of_article_l1656_165695


namespace part_a_l1656_165647

theorem part_a (x : ℝ) : 1 + (1 / (2 + 1 / ((4 * x + 1) / (2 * x + 1) - 1 / (2 + 1 / x)))) = 19 / 14 ↔ x = 1 / 2 := sorry

end part_a_l1656_165647


namespace sequence_general_term_l1656_165639

theorem sequence_general_term :
  ∀ n : ℕ, n > 0 → (∀ a: ℕ → ℝ,  a 1 = 4 ∧ (∀ n: ℕ, n > 0 → a (n + 1) = (3 * a n + 2) / (a n + 4))
  → a n = (2 ^ (n - 1) + 5 ^ (n - 1)) / (5 ^ (n - 1) - 2 ^ (n - 1))) :=
by
  sorry

end sequence_general_term_l1656_165639


namespace pencils_added_l1656_165688

theorem pencils_added (initial_pencils total_pencils Mike_pencils : ℕ) 
    (h1 : initial_pencils = 41) 
    (h2 : total_pencils = 71) 
    (h3 : total_pencils = initial_pencils + Mike_pencils) :
    Mike_pencils = 30 := by
  sorry

end pencils_added_l1656_165688


namespace proposition_holds_for_odd_numbers_l1656_165677

variable (P : ℕ → Prop)

theorem proposition_holds_for_odd_numbers 
  (h1 : P 1)
  (h_ind : ∀ k : ℕ, k ≥ 1 → P k → P (k + 2)) :
  ∀ n : ℕ, n % 2 = 1 → P n :=
by
  sorry

end proposition_holds_for_odd_numbers_l1656_165677


namespace solve_eq_f_x_plus_3_l1656_165681

-- Define the function f with its piecewise definition based on the conditions
noncomputable def f (x : ℝ) : ℝ :=
  if h : x ≥ 0 then x^2 - 3 * x
  else -(x^2 - 3 * (-x))

-- Define the main theorem to find the solution set
theorem solve_eq_f_x_plus_3 (x : ℝ) :
  f x = x + 3 ↔ x = 2 + Real.sqrt 7 ∨ x = -1 ∨ x = -3 :=
by sorry

end solve_eq_f_x_plus_3_l1656_165681


namespace greatest_integer_y_l1656_165648

-- Define the fraction and inequality condition
def inequality_condition (y : ℤ) : Prop := 8 * 17 > 11 * y

-- Prove the greatest integer y satisfying the condition is 12
theorem greatest_integer_y : ∃ y : ℤ, inequality_condition y ∧ (∀ z : ℤ, inequality_condition z → z ≤ y) ∧ y = 12 :=
by
  exists 12
  sorry

end greatest_integer_y_l1656_165648


namespace total_number_of_people_l1656_165618

-- Definitions corresponding to conditions
variables (A C : ℕ)
variables (cost_adult cost_child total_revenue : ℝ)
variables (ratio_child_adult : ℝ)

-- Assumptions given in the problem
axiom cost_adult_def : cost_adult = 7
axiom cost_child_def : cost_child = 3
axiom total_revenue_def : total_revenue = 6000
axiom ratio_def : C = 3 * A
axiom revenue_eq : total_revenue = cost_adult * A + cost_child * C

-- The main statement to prove
theorem total_number_of_people : A + C = 1500 :=
by
  sorry  -- Proof of the theorem

end total_number_of_people_l1656_165618


namespace production_today_l1656_165664

-- Conditions
def average_daily_production_past_n_days (P : ℕ) (n : ℕ) := P = n * 50
def new_average_daily_production (P : ℕ) (T : ℕ) (new_n : ℕ) := (P + T) / new_n = 55

-- Values from conditions
def n := 11
def P := 11 * 50

-- Mathematically equivalent proof problem
theorem production_today :
  ∃ (T : ℕ), average_daily_production_past_n_days P n ∧ new_average_daily_production P T 12 → T = 110 :=
by
  sorry

end production_today_l1656_165664


namespace find_k_l1656_165683

theorem find_k (angle_BAC : ℝ) (angle_D : ℝ)
  (h1 : 0 < angle_BAC ∧ angle_BAC < π)
  (h2 : 0 < angle_D ∧ angle_D < π)
  (h3 : (π - angle_BAC) / 2 = 3 * angle_D) :
  angle_BAC = (5 / 11) * π :=
by sorry

end find_k_l1656_165683


namespace p_satisfies_conditions_l1656_165691

noncomputable def p (x : ℕ) : ℕ := sorry

theorem p_satisfies_conditions (h_monic : p 1 = 1 ∧ p 2 = 2 ∧ p 3 = 3 ∧ p 4 = 4 ∧ p 5 = 5) : 
  p 6 = 126 := sorry

end p_satisfies_conditions_l1656_165691


namespace find_q_l1656_165660

noncomputable def Q (x p q d : ℝ) : ℝ := x^3 + p * x^2 + q * x + d

theorem find_q (p q d : ℝ) (h₁ : -p / 3 = q) (h₂ : q = 1 + p + q + 5) (h₃ : d = 5) : q = 2 :=
by
  sorry

end find_q_l1656_165660


namespace perimeter_of_AF1B_l1656_165669

noncomputable def ellipse_perimeter (a b x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  (2 * a)

theorem perimeter_of_AF1B (h : (6:ℝ) = 6) :
  ellipse_perimeter 6 4 0 0 6 0 = 24 :=
by
  sorry

end perimeter_of_AF1B_l1656_165669


namespace rent_percentage_l1656_165643

variable (E : ℝ)
variable (last_year_rent : ℝ := 0.20 * E)
variable (this_year_earnings : ℝ := 1.20 * E)
variable (this_year_rent : ℝ := 0.30 * this_year_earnings)

theorem rent_percentage (E : ℝ) (h_last_year_rent : last_year_rent = 0.20 * E)
  (h_this_year_earnings : this_year_earnings = 1.20 * E)
  (h_this_year_rent : this_year_rent = 0.30 * this_year_earnings) : 
  this_year_rent / last_year_rent * 100 = 180 := by
  sorry

end rent_percentage_l1656_165643


namespace linear_function_behavior_l1656_165642

theorem linear_function_behavior (x y : ℝ) (h : y = -3 * x + 6) :
  ∀ x1 x2 : ℝ, x1 < x2 → (y = -3 * x1 + 6) → (y = -3 * x2 + 6) → -3 * (x1 - x2) > 0 :=
by
  sorry

end linear_function_behavior_l1656_165642


namespace positive_difference_enrollment_l1656_165638

theorem positive_difference_enrollment 
  (highest_enrollment : ℕ)
  (lowest_enrollment : ℕ)
  (h_highest : highest_enrollment = 2150)
  (h_lowest : lowest_enrollment = 980) :
  highest_enrollment - lowest_enrollment = 1170 :=
by {
  -- Proof to be added here
  sorry
}

end positive_difference_enrollment_l1656_165638


namespace inequality_solution_set_l1656_165644

theorem inequality_solution_set :
  { x : ℝ | (3 * x + 1) / (x - 2) ≤ 0 } = { x : ℝ | -1/3 ≤ x ∧ x < 2 } :=
sorry

end inequality_solution_set_l1656_165644


namespace marians_groceries_l1656_165631

variables (G : ℝ)

theorem marians_groceries :
  let initial_balance := 126
  let returned_amount := 45
  let new_balance := 171
  let gas_expense := G / 2
  initial_balance + G + gas_expense - returned_amount = new_balance → G = 60 :=
sorry

end marians_groceries_l1656_165631


namespace problem_1_problem_2_l1656_165661

variable {c : ℝ}

def p (c : ℝ) : Prop := ∀ x₁ x₂ : ℝ, x₁ < x₂ → c ^ x₁ > c ^ x₂

def q (c : ℝ) : Prop := ∀ x₁ x₂ : ℝ, (1 / 2) < x₁ ∧ x₁ < x₂ → (x₁ ^ 2 - 2 * c * x₁ + 1) < (x₂ ^ 2 - 2 * c * x₂ + 1)

theorem problem_1 (hc : 0 < c) (hcn1 : c ≠ 1) (hp : p c) (hnq_false : ¬ ¬ q c) : 0 < c ∧ c ≤ 1 / 2 :=
by
  sorry

theorem problem_2 (hc : 0 < c) (hcn1 : c ≠ 1) (hpq_false : ¬ (p c ∧ q c)) (hp_or_q : p c ∨ q c) : 1 / 2 < c ∧ c < 1 :=
by
  sorry

end problem_1_problem_2_l1656_165661


namespace min_value_condition_l1656_165608

theorem min_value_condition (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  3 * m + n = 1 → (1 / m + 2 / n) ≥ 5 + 2 * Real.sqrt 6 :=
by
  sorry

end min_value_condition_l1656_165608


namespace square_side_length_l1656_165629

theorem square_side_length (x : ℝ) (h : 4 * x = x^2) : x = 4 := 
by
  sorry

end square_side_length_l1656_165629


namespace triangle_AD_eq_8sqrt2_l1656_165668

/-- Given a triangle ABC where AB = 13, AC = 20, and
    D is the foot of the perpendicular from A to BC,
    with the ratio BD : CD = 3 : 4, prove that AD = 8√2. -/
theorem triangle_AD_eq_8sqrt2 
  (AB AC : ℝ) (BD CD AD : ℝ) 
  (h₁ : AB = 13)
  (h₂ : AC = 20)
  (h₃ : BD / CD = 3 / 4)
  (h₄ : BD^2 = AB^2 - AD^2)
  (h₅ : CD^2 = AC^2 - AD^2) :
  AD = 8 * Real.sqrt 2 :=
by
  sorry

end triangle_AD_eq_8sqrt2_l1656_165668
