import Mathlib

namespace sector_area_half_triangle_area_l2004_200416

theorem sector_area_half_triangle_area (θ : Real) (r : Real) (hθ1 : 0 < θ) (hθ2 : θ < π / 3) :
    2 * θ = Real.tan θ := by
  sorry

end sector_area_half_triangle_area_l2004_200416


namespace percent_not_covering_politics_l2004_200424

-- Definitions based on the conditions
def total_reporters : ℕ := 100
def local_politics_reporters : ℕ := 28
def percent_cover_local_politics : ℚ := 0.7

-- To be proved
theorem percent_not_covering_politics :
  let politics_reporters := local_politics_reporters / percent_cover_local_politics 
  (total_reporters - politics_reporters) / total_reporters = 0.6 := 
by
  sorry

end percent_not_covering_politics_l2004_200424


namespace sequence_ratio_proof_l2004_200491

variable {a : ℕ → ℤ}

-- Sequence definition
axiom a₁ : a 1 = 3
axiom a_recurrence : ∀ n : ℕ, a (n + 1) = 4 * a n + 3

-- The theorem to be proved
theorem sequence_ratio_proof (n : ℕ) : (a (n + 1) + 1) / (a n + 1) = 4 := by
  sorry

end sequence_ratio_proof_l2004_200491


namespace max_grapes_in_bag_l2004_200423

theorem max_grapes_in_bag : ∃ (x : ℕ), x > 100 ∧ x % 3 = 1 ∧ x % 5 = 2 ∧ x % 7 = 4 ∧ x = 172 := by
  sorry

end max_grapes_in_bag_l2004_200423


namespace find_x_l2004_200492

def x_y_conditions (x y : ℝ) : Prop :=
  x > y ∧
  x^2 * y^2 + x^2 + y^2 + 2 * x * y = 40 ∧
  x * y + x + y = 8

theorem find_x (x y : ℝ) (h : x_y_conditions x y) : x = 3 + Real.sqrt 7 :=
by
  sorry

end find_x_l2004_200492


namespace no_solution_system_of_inequalities_l2004_200410

theorem no_solution_system_of_inequalities :
  ¬ ∃ (x y : ℝ),
    11 * x^2 - 10 * x * y + 3 * y^2 ≤ 3 ∧
    5 * x + y ≤ -10 :=
by
  sorry

end no_solution_system_of_inequalities_l2004_200410


namespace find_c_l2004_200464

noncomputable def func_condition (f : ℝ → ℝ) (c : ℝ) :=
  ∀ x y : ℝ, (f x + 1) * (f y + 1) = f (x + y) + f (x * y + c)

theorem find_c :
  ∃ c : ℝ, ∀ (f : ℝ → ℝ), func_condition f c → (c = 1 ∨ c = -1) :=
sorry

end find_c_l2004_200464


namespace C_D_meeting_time_l2004_200481

-- Defining the conditions.
variables (A B C D : Type) [LinearOrderedField A] (V_A V_B V_C V_D : A)
variables (startTime meet_AC meet_BD meet_AB meet_CD : A)

-- Cars' initial meeting conditions
axiom init_cond : startTime = 0
axiom meet_cond_AC : meet_AC = 7
axiom meet_cond_BD : meet_BD = 7
axiom meet_cond_AB : meet_AB = 53
axiom speed_relation : V_A + V_C = V_B + V_D ∧ V_A - V_B = V_D - V_C

-- The problem asks for the meeting time of C and D
theorem C_D_meeting_time : meet_CD = 53 :=
by sorry

end C_D_meeting_time_l2004_200481


namespace snail_kite_first_day_snails_l2004_200402

theorem snail_kite_first_day_snails (x : ℕ) 
  (h : x + (x + 2) + (x + 4) + (x + 6) + (x + 8) = 35) : 
  x = 3 :=
sorry

end snail_kite_first_day_snails_l2004_200402


namespace select_k_plus_1_nums_divisible_by_n_l2004_200476

theorem select_k_plus_1_nums_divisible_by_n (n k : ℕ) (hn : n > 0) (hk : k > 0) (nums : Fin (n + k) → ℕ) :
  ∃ (indices : Finset (Fin (n + k))), indices.card ≥ k + 1 ∧ (indices.sum (nums ∘ id)) % n = 0 :=
sorry

end select_k_plus_1_nums_divisible_by_n_l2004_200476


namespace rabbit_is_hit_l2004_200425

noncomputable def P_A : ℝ := 0.6
noncomputable def P_B : ℝ := 0.5
noncomputable def P_C : ℝ := 0.4

noncomputable def P_none_hit : ℝ := (1 - P_A) * (1 - P_B) * (1 - P_C)
noncomputable def P_rabbit_hit : ℝ := 1 - P_none_hit

theorem rabbit_is_hit :
  P_rabbit_hit = 0.88 :=
by
  -- Proof is omitted
  sorry

end rabbit_is_hit_l2004_200425


namespace coordinates_of_P_l2004_200496

-- Define a structure for a 2D point
structure Point where
  x : ℝ
  y : ℝ

-- Define what it means for a point to be in the third quadrant
def in_third_quadrant (P : Point) : Prop :=
  P.x < 0 ∧ P.y < 0

-- Define the distance from a point to the x-axis
def distance_to_x_axis (P : Point) : ℝ :=
  |P.y|

-- Define the distance from a point to the y-axis
def distance_to_y_axis (P : Point) : ℝ :=
  |P.x|

-- The main proof statement
theorem coordinates_of_P (P : Point) :
  in_third_quadrant P →
  distance_to_x_axis P = 2 →
  distance_to_y_axis P = 5 →
  P = { x := -5, y := -2 } :=
by
  intros h1 h2 h3
  sorry

end coordinates_of_P_l2004_200496


namespace geom_seq_log_eqn_l2004_200413

theorem geom_seq_log_eqn {a : ℕ → ℝ} {b : ℕ → ℝ}
    (geom_seq : ∃ (r : ℝ) (a1 : ℝ), ∀ n : ℕ, a (n + 1) = a1 * r^n)
    (log_seq : ∀ n : ℕ, b n = Real.log (a (n + 1)) / Real.log 2)
    (b_eqn : b 1 + b 3 = 4) : a 2 = 4 :=
by
  sorry

end geom_seq_log_eqn_l2004_200413


namespace cosine_of_angle_in_convex_quadrilateral_l2004_200477

theorem cosine_of_angle_in_convex_quadrilateral
    (A C : ℝ)
    (AB CD AD BC : ℝ)
    (h1 : A = C)
    (h2 : AB = 150)
    (h3 : CD = 150)
    (h4 : AD = BC)
    (h5 : AB + BC + CD + AD = 580) :
    Real.cos A = 7 / 15 := 
  sorry

end cosine_of_angle_in_convex_quadrilateral_l2004_200477


namespace sports_club_problem_l2004_200433

theorem sports_club_problem (N B T Neither X : ℕ) (hN : N = 42) (hB : B = 20) (hT : T = 23) (hNeither : Neither = 6) :
  (B + T - X + Neither = N) → X = 7 :=
by
  intro h
  sorry

end sports_club_problem_l2004_200433


namespace value_of_knife_l2004_200467

/-- Two siblings sold their flock of sheep. Each sheep was sold for as many florins as 
the number of sheep originally in the flock. They divided the revenue by giving out 
10 florins at a time. First, the elder brother took 10 florins, then the younger brother, 
then the elder again, and so on. In the end, the younger brother received less than 10 florins, 
so the elder brother gave him his knife, making their earnings equal. 
Prove that the value of the knife in florins is 2. -/
theorem value_of_knife (n : ℕ) (k m : ℕ) (h1 : n^2 = 20 * k + 10 + m) (h2 : 1 ≤ m ∧ m ≤ 9) : 
  (∃ b : ℕ, 10 - b = m + b ∧ b = 2) :=
by
  sorry

end value_of_knife_l2004_200467


namespace cost_of_adult_ticket_l2004_200434

-- Conditions provided in the original problem.
def total_people : ℕ := 23
def child_tickets_cost : ℕ := 10
def total_money_collected : ℕ := 246
def children_attended : ℕ := 7

-- Define some unknown amount A for the adult tickets cost to be solved.
variable (A : ℕ)

-- Define the Lean statement for the proof problem.
theorem cost_of_adult_ticket :
  16 * A = 176 →
  A = 11 :=
by
  -- Start the proof (this part will be filled out during the proof process).
  sorry

#check cost_of_adult_ticket  -- To ensure it type-checks

end cost_of_adult_ticket_l2004_200434


namespace imaginary_part_of_z_l2004_200439

noncomputable def i : ℂ := Complex.I
noncomputable def z : ℂ := i / (i - 1)

theorem imaginary_part_of_z : z.im = -1 / 2 := by
  sorry

end imaginary_part_of_z_l2004_200439


namespace unique_x_value_l2004_200461

theorem unique_x_value (x : ℝ) (h : x ≠ 0) (h_sqrt : Real.sqrt (5 * x / 7) = x) : x = 5 / 7 :=
by
  sorry

end unique_x_value_l2004_200461


namespace geometric_sequence_common_ratio_l2004_200465

theorem geometric_sequence_common_ratio (a_1 a_4 q : ℕ) (h1 : a_1 = 8) (h2 : a_4 = 64) (h3 : a_4 = a_1 * q^3) : q = 2 :=
by {
  -- Given: a_1 = 8
  --        a_4 = 64
  --        a_4 = a_1 * q^3
  -- Prove: q = 2
  sorry
}

end geometric_sequence_common_ratio_l2004_200465


namespace total_trout_caught_l2004_200407

theorem total_trout_caught (n_share j_share total_caught : ℕ) (h1 : n_share = 9) (h2 : j_share = 9) (h3 : total_caught = n_share + j_share) :
  total_caught = 18 :=
by
  sorry

end total_trout_caught_l2004_200407


namespace simplify_expr_l2004_200404

theorem simplify_expr (a : ℝ) (h_a : a = (8:ℝ)^(1/2) * (1/2) - (3:ℝ)^(1/2)^(0) ) : 
  a = (2:ℝ)^(1/2) - 1 := 
by
  sorry

end simplify_expr_l2004_200404


namespace savings_percentage_l2004_200448

theorem savings_percentage
  (S : ℝ)
  (last_year_saved : ℝ := 0.06 * S)
  (this_year_salary : ℝ := 1.10 * S)
  (this_year_saved : ℝ := 0.10 * this_year_salary)
  (ratio := this_year_saved / last_year_saved * 100):
  ratio = 183.33 := 
sorry

end savings_percentage_l2004_200448


namespace maximum_sum_of_triplets_l2004_200482

-- Define a list representing a 9-digit number consisting of digits 1 to 9 in some order
def valid_digits (digits : List ℕ) : Prop :=
  digits.length = 9 ∧ ∀ n, n ∈ digits → n ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9]
  
def sum_of_triplets (digits : List ℕ) : ℕ :=
  100 * digits[0]! + 10 * digits[1]! + digits[2]! +
  100 * digits[1]! + 10 * digits[2]! + digits[3]! +
  100 * digits[2]! + 10 * digits[3]! + digits[4]! +
  100 * digits[3]! + 10 * digits[4]! + digits[5]! +
  100 * digits[4]! + 10 * digits[5]! + digits[6]! +
  100 * digits[5]! + 10 * digits[6]! + digits[7]! +
  100 * digits[6]! + 10 * digits[7]! + digits[8]!

theorem maximum_sum_of_triplets :
  ∃ digits : List ℕ, valid_digits digits ∧ sum_of_triplets digits = 4648 :=
  sorry

end maximum_sum_of_triplets_l2004_200482


namespace train_length_problem_l2004_200488

noncomputable def train_length (v : ℝ) (t : ℝ) (L : ℝ) : Prop :=
v = 90 / 3.6 ∧ t = 60 ∧ 2 * L = v * t

theorem train_length_problem : train_length 90 1 750 :=
by
  -- Define speed in m/s
  let v_m_s := 90 * (1000 / 3600)
  -- Calculate distance = speed * time
  let distance := 25 * 60
  -- Since distance = 2 * Length
  have h : 2 * 750 = 1500 := sorry
  show train_length 90 1 750
  simp [train_length, h]
  sorry

end train_length_problem_l2004_200488


namespace correct_triangle_set_l2004_200494

/-- Definition of triangle inequality -/
def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- Sets of lengths for checking the triangle inequality -/
def Set1 : ℝ × ℝ × ℝ := (5, 8, 2)
def Set2 : ℝ × ℝ × ℝ := (5, 8, 13)
def Set3 : ℝ × ℝ × ℝ := (5, 8, 5)
def Set4 : ℝ × ℝ × ℝ := (2, 7, 5)

/-- The correct set of lengths that can form a triangle according to the triangle inequality -/
theorem correct_triangle_set : satisfies_triangle_inequality 5 8 5 :=
by
  -- Proof would be here
  sorry

end correct_triangle_set_l2004_200494


namespace number_of_girls_calculation_l2004_200431

theorem number_of_girls_calculation : 
  ∀ (number_of_boys number_of_girls total_children : ℕ), 
  number_of_boys = 27 → total_children = 62 → number_of_girls = total_children - number_of_boys → number_of_girls = 35 :=
by
  intros number_of_boys number_of_girls total_children 
  intros h_boys h_total h_calc
  rw [h_boys, h_total] at h_calc
  simp at h_calc
  exact h_calc

end number_of_girls_calculation_l2004_200431


namespace allocation_schemes_l2004_200455

theorem allocation_schemes (students factories: ℕ) (has_factory_a: Prop) (A_must_have_students: has_factory_a): students = 3 → factories = 4 → has_factory_a → (∃ n: ℕ, n = 4^3 - 3^3 ∧ n = 37) :=
by try { sorry }

end allocation_schemes_l2004_200455


namespace chime_2203_occurs_on_March_19_l2004_200473

-- Define the initial conditions: chime patterns
def chimes_at_half_hour : Nat := 1
def chimes_at_hour (h : Nat) : Nat := if h = 12 then 12 else h % 12

-- Define the start time and the question parameters
def start_time_hours : Nat := 10
def start_time_minutes : Nat := 45
def start_day : Nat := 26 -- Assume February 26 as starting point, to facilitate day count accurately
def target_chime : Nat := 2203

-- Define the date calculation function (based on given solution steps)
noncomputable def calculate_chime_date (start_day : Nat) : Nat := sorry

-- The goal is to prove calculate_chime_date with given start conditions equals 19 (March 19th is the 19th day after the base day assumption of March 0)
theorem chime_2203_occurs_on_March_19 :
  calculate_chime_date start_day = 19 :=
sorry

end chime_2203_occurs_on_March_19_l2004_200473


namespace unique_triangle_solution_l2004_200474

noncomputable def triangle_solutions (a b A : ℝ) : ℕ :=
sorry -- Placeholder for actual function calculating number of solutions

theorem unique_triangle_solution : triangle_solutions 30 25 150 = 1 :=
sorry -- Proof goes here

end unique_triangle_solution_l2004_200474


namespace first_discount_percentage_l2004_200441

theorem first_discount_percentage (original_price final_price : ℝ)
  (first_discount second_discount : ℝ) (h_orig : original_price = 200)
  (h_final : final_price = 144) (h_second_disc : second_discount = 0.20) :
  first_discount = 0.10 :=
by
  sorry

end first_discount_percentage_l2004_200441


namespace alice_has_largest_result_l2004_200484

def initial_number : ℕ := 15

def alice_transformation (x : ℕ) : ℕ := (x * 3 - 2 + 4)
def bob_transformation (x : ℕ) : ℕ := (x * 2 + 3 - 5)
def charlie_transformation (x : ℕ) : ℕ := (x + 5) / 2 * 4

def alice_final := alice_transformation initial_number
def bob_final := bob_transformation initial_number
def charlie_final := charlie_transformation initial_number

theorem alice_has_largest_result :
  alice_final > bob_final ∧ alice_final > charlie_final := by
  sorry

end alice_has_largest_result_l2004_200484


namespace cafeteria_apples_l2004_200495

theorem cafeteria_apples (handed_out: ℕ) (pies: ℕ) (apples_per_pie: ℕ) 
(h1: handed_out = 27) (h2: pies = 5) (h3: apples_per_pie = 4) : handed_out + pies * apples_per_pie = 47 :=
by
  -- The proof will be provided here if needed
  sorry

end cafeteria_apples_l2004_200495


namespace soccer_league_total_games_l2004_200470

theorem soccer_league_total_games :
  let teams := 20
  let regular_games_per_team := 19 * 3
  let total_regular_games := (regular_games_per_team * teams) / 2
  let promotional_games_per_team := 3
  let total_promotional_games := promotional_games_per_team * teams
  let total_games := total_regular_games + total_promotional_games
  total_games = 1200 :=
by
  sorry

end soccer_league_total_games_l2004_200470


namespace history_percentage_l2004_200489

theorem history_percentage (H : ℕ) (math_percentage : ℕ := 72) (third_subject_percentage : ℕ := 69) (overall_average : ℕ := 75) :
  (math_percentage + H + third_subject_percentage) / 3 = overall_average → H = 84 :=
by
  intro h
  sorry

end history_percentage_l2004_200489


namespace problem1_problem2_l2004_200420

-- First proof problem
theorem problem1 : - (2^2 : ℚ) + (2/3) * ((1 - 1/3) ^ 2) = -100/27 :=
by sorry

-- Second proof problem
theorem problem2 : (8 : ℚ) ^ (1 / 3) - |2 - (3 : ℚ) ^ (1 / 2)| - (3 : ℚ) ^ (1 / 2) = 0 :=
by sorry

end problem1_problem2_l2004_200420


namespace find_usual_time_l2004_200421

variables (P D T : ℝ)
variable (h1 : P = D / T)
variable (h2 : 3 / 4 * P = D / (T + 20))

theorem find_usual_time (h1 : P = D / T) (h2 : 3 / 4 * P = D / (T + 20)) : T = 80 := 
  sorry

end find_usual_time_l2004_200421


namespace smallest_nonfactor_product_of_48_l2004_200438

noncomputable def is_factor_of (a b : ℕ) : Prop :=
  b % a = 0

theorem smallest_nonfactor_product_of_48
  (m n : ℕ)
  (h1 : m ≠ n)
  (h2 : is_factor_of m 48)
  (h3 : is_factor_of n 48)
  (h4 : ¬is_factor_of (m * n) 48) :
  m * n = 18 :=
sorry

end smallest_nonfactor_product_of_48_l2004_200438


namespace find_k_l2004_200417

-- Define the variables and conditions
variables (x y k : ℤ)

-- State the theorem
theorem find_k (h1 : x = 2) (h2 : y = 1) (h3 : k * x - y = 3) : k = 2 :=
sorry

end find_k_l2004_200417


namespace field_trip_students_l2004_200440

theorem field_trip_students (bus_cost admission_per_student budget : ℕ) (students : ℕ)
  (h1 : bus_cost = 100)
  (h2 : admission_per_student = 10)
  (h3 : budget = 350)
  (total_cost : students * admission_per_student + bus_cost ≤ budget) : 
  students = 25 :=
by
  sorry

end field_trip_students_l2004_200440


namespace grace_wait_time_l2004_200450

variable (hose1_rate : ℕ) (hose2_rate : ℕ) (pool_capacity : ℕ) (time_after_second_hose : ℕ)
variable (h : ℕ)

theorem grace_wait_time 
  (h1 : hose1_rate = 50)
  (h2 : hose2_rate = 70)
  (h3 : pool_capacity = 390)
  (h4 : time_after_second_hose = 2) : 
  50 * h + (50 + 70) * 2 = 390 → h = 3 :=
by
  sorry

end grace_wait_time_l2004_200450


namespace soccer_tournament_matches_l2004_200443

theorem soccer_tournament_matches (x : ℕ) (h : 1 ≤ x) : (1 / 2 : ℝ) * x * (x - 1) = 45 := sorry

end soccer_tournament_matches_l2004_200443


namespace find_y_l2004_200400

def G (a b c d : ℕ) : ℕ := a^b + c * d

theorem find_y (y : ℕ) (h : G 3 y 5 18 = 500) : y = 6 :=
sorry

end find_y_l2004_200400


namespace patio_tiles_l2004_200472

theorem patio_tiles (r c : ℕ) (h1 : r * c = 48) (h2 : (r + 4) * (c - 2) = 48) : r = 6 :=
sorry

end patio_tiles_l2004_200472


namespace f_6_plus_f_neg3_l2004_200408

noncomputable def f : ℝ → ℝ := sorry

-- f is an odd function
def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- f is increasing in the interval [3,6]
def is_increasing_interval (f : ℝ → ℝ) (a b : ℝ) := a ≤ b → ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

-- Define the given conditions
axiom h1 : is_odd_function f
axiom h2 : is_increasing_interval f 3 6
axiom h3 : f 6 = 8
axiom h4 : f 3 = -1

-- The statement to be proved
theorem f_6_plus_f_neg3 : f 6 + f (-3) = 9 :=
by
  sorry

end f_6_plus_f_neg3_l2004_200408


namespace smallest_four_digit_in_pascals_triangle_l2004_200479

-- Define Pascal's triangle
def pascals_triangle : ℕ → ℕ → ℕ
| 0, _ => 1
| _, 0 => 1
| n+1, k+1 => pascals_triangle n k + pascals_triangle n (k+1)

-- State the theorem
theorem smallest_four_digit_in_pascals_triangle : ∃ (n k : ℕ), pascals_triangle n k = 1000 ∧ ∀ m, m < 1000 → ∀ r s, pascals_triangle r s ≠ m :=
sorry

end smallest_four_digit_in_pascals_triangle_l2004_200479


namespace scorpion_segments_daily_total_l2004_200453

theorem scorpion_segments_daily_total (seg1 : ℕ) (seg2 : ℕ) (additional : ℕ) (total_daily : ℕ) :
  (seg1 = 60) →
  (seg2 = 2 * seg1 * 2) →
  (additional = 10 * 50) →
  (total_daily = seg1 + seg2 + additional) →
  total_daily = 800 :=
by
  intros h1 h2 h3 h4
  sorry

end scorpion_segments_daily_total_l2004_200453


namespace anne_distance_l2004_200405

theorem anne_distance (speed time : ℕ) (h_speed : speed = 2) (h_time : time = 3) : 
  (speed * time) = 6 := by
  sorry

end anne_distance_l2004_200405


namespace discount_on_shoes_l2004_200493

theorem discount_on_shoes (x : ℝ) :
  let shoe_price := 200
  let shirt_price := 80
  let total_spent := 285
  let total_shirt_price := 2 * shirt_price
  let initial_total := shoe_price + total_shirt_price
  let disc_shoe_price := shoe_price - (shoe_price * x / 100)
  let pre_final_total := disc_shoe_price + total_shirt_price
  let final_total := pre_final_total * (1 - 0.05)
  final_total = total_spent → x = 30 :=
by
  intros shoe_price shirt_price total_spent total_shirt_price initial_total disc_shoe_price pre_final_total final_total h
  dsimp [shoe_price, shirt_price, total_spent, total_shirt_price, initial_total, disc_shoe_price, pre_final_total, final_total] at h
  -- Here, we would normally continue the proof, but we'll insert 'sorry' for now as instructed.
  sorry

end discount_on_shoes_l2004_200493


namespace math_problem_l2004_200403

theorem math_problem :
  8 / 4 - 3^2 + 4 * 2 + (Nat.factorial 5) = 121 :=
by
  sorry

end math_problem_l2004_200403


namespace point_A_lies_on_plane_l2004_200445

-- Define the plane equation
def plane (x y z : ℝ) : Prop := 2 * x - y + 2 * z = 7

-- Define the specific point
def point_A : Prop := plane 2 3 3

-- The theorem stating that point A lies on the plane
theorem point_A_lies_on_plane : point_A :=
by
  -- Proof skipped
  sorry

end point_A_lies_on_plane_l2004_200445


namespace solution_to_prime_equation_l2004_200442

theorem solution_to_prime_equation (x y : ℕ) (p : ℕ) (h1 : Nat.Prime p) (hx : 0 < x) (hy : 0 < y) :
  x^3 + y^3 = p * (xy + p) ↔ (x = 8 ∧ y = 1 ∧ p = 19) ∨ (x = 1 ∧ y = 8 ∧ p = 19) ∨ 
              (x = 7 ∧ y = 2 ∧ p = 13) ∨ (x = 2 ∧ y = 7 ∧ p = 13) ∨ 
              (x = 5 ∧ y = 4 ∧ p = 7) ∨ (x = 4 ∧ y = 5 ∧ p = 7) := sorry

end solution_to_prime_equation_l2004_200442


namespace ratio_of_diamonds_to_spades_l2004_200487

-- Given conditions
variable (total_cards : Nat := 13)
variable (black_cards : Nat := 7)
variable (red_cards : Nat := 6)
variable (clubs : Nat := 6)
variable (diamonds : Nat)
variable (spades : Nat)
variable (hearts : Nat := 2 * diamonds)
variable (cards_distribution : clubs + diamonds + hearts + spades = total_cards)
variable (black_distribution : clubs + spades = black_cards)

-- Define the proof theorem
theorem ratio_of_diamonds_to_spades : (diamonds / spades : ℝ) = 2 :=
 by
  -- temporarily we insert sorry to skip the proof
  sorry

end ratio_of_diamonds_to_spades_l2004_200487


namespace arithmetic_square_root_of_9_l2004_200414

theorem arithmetic_square_root_of_9 : Real.sqrt 9 = 3 := by
  sorry

end arithmetic_square_root_of_9_l2004_200414


namespace least_perimeter_of_triangle_l2004_200446

theorem least_perimeter_of_triangle (a b : ℕ) (a_eq : a = 33) (b_eq : b = 42) (c : ℕ) (h1 : c + a > b) (h2 : c + b > a) (h3 : a + b > c) : a + b + c = 85 :=
sorry

end least_perimeter_of_triangle_l2004_200446


namespace xyz_squared_sum_l2004_200475

theorem xyz_squared_sum (x y z : ℝ)
  (h1 : x^2 + 6 * y = -17)
  (h2 : y^2 + 4 * z = 1)
  (h3 : z^2 + 2 * x = 2) :
  x^2 + y^2 + z^2 = 14 := 
sorry

end xyz_squared_sum_l2004_200475


namespace odd_primes_pq_division_l2004_200430

theorem odd_primes_pq_division (p q : ℕ) (m : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
(hp_odd : ¬Even p) (hq_odd : ¬Even q) (hp_gt_hq : p > q) (hm_pos : 0 < m) : ¬(p * q ∣ m ^ (p - q) + 1) :=
by 
  sorry

end odd_primes_pq_division_l2004_200430


namespace mary_animals_count_l2004_200447

def initial_lambs := 18
def initial_alpacas := 5
def initial_baby_lambs := 7 * 4
def traded_lambs := 8
def traded_alpacas := 2
def received_goats := 3
def received_chickens := 10
def chickens_traded_for_alpacas := received_chickens / 2
def additional_lambs := 20
def additional_alpacas := 6

noncomputable def final_lambs := initial_lambs + initial_baby_lambs - traded_lambs + additional_lambs
noncomputable def final_alpacas := initial_alpacas - traded_alpacas + 2 + additional_alpacas
noncomputable def final_goats := received_goats
noncomputable def final_chickens := received_chickens - chickens_traded_for_alpacas

theorem mary_animals_count :
  final_lambs = 58 ∧ 
  final_alpacas = 11 ∧ 
  final_goats = 3 ∧ 
  final_chickens = 5 :=
by 
  sorry

end mary_animals_count_l2004_200447


namespace find_f1_l2004_200444

def periodic (f : ℝ → ℝ) (p : ℝ) := ∀ x, f (x + p) = f x
def odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem find_f1 (f : ℝ → ℝ)
  (h_periodic : periodic f 2)
  (h_odd : odd f) :
  f 1 = 0 :=
sorry

end find_f1_l2004_200444


namespace sum_of_digits_triangular_array_l2004_200437

theorem sum_of_digits_triangular_array (N : ℕ) (h : N * (N + 1) / 2 = 5050) : 
  Nat.digits 10 N = [1, 0, 0] := by
  sorry

end sum_of_digits_triangular_array_l2004_200437


namespace C_eq_D_at_n_l2004_200415

noncomputable def C_n (n : ℕ) : ℝ := 768 * (1 - (1 / (3^n)))
noncomputable def D_n (n : ℕ) : ℝ := (4096 / 5) * (1 - ((-1)^n / (4^n)))
noncomputable def n_ge_1 : ℕ := 4

theorem C_eq_D_at_n : ∀ n ≥ 1, C_n n = D_n n → n = n_ge_1 :=
by
  intro n hn heq
  sorry

end C_eq_D_at_n_l2004_200415


namespace billing_error_l2004_200452

theorem billing_error (x y : ℕ) (hx : 10 ≤ x ∧ x ≤ 99) (hy : 10 ≤ y ∧ y ≤ 99) 
    (h : 100 * y + x - (100 * x + y) = 2970) : y - x = 30 ∧ 10 ≤ x ∧ x ≤ 69 ∧ 40 ≤ y ∧ y ≤ 99 := 
by
  sorry

end billing_error_l2004_200452


namespace speed_of_mrs_a_l2004_200454

theorem speed_of_mrs_a
  (distance_between : ℝ)
  (speed_mr_a : ℝ)
  (speed_bee : ℝ)
  (distance_bee_travelled : ℝ)
  (time_bee : ℝ)
  (remaining_distance : ℝ)
  (speed_mrs_a : ℝ) :
  distance_between = 120 ∧
  speed_mr_a = 30 ∧
  speed_bee = 60 ∧
  distance_bee_travelled = 180 ∧
  time_bee = distance_bee_travelled / speed_bee ∧
  remaining_distance = distance_between - (speed_mr_a * time_bee) ∧
  speed_mrs_a = remaining_distance / time_bee →
  speed_mrs_a = 10 := by
  sorry

end speed_of_mrs_a_l2004_200454


namespace value_of_m_div_x_l2004_200418

noncomputable def ratio_of_a_to_b (a b : ℝ) : Prop := a / b = 4 / 5
noncomputable def x_value (a : ℝ) : ℝ := a * 1.75
noncomputable def m_value (b : ℝ) : ℝ := b * 0.20

theorem value_of_m_div_x (a b : ℝ) (h1 : ratio_of_a_to_b a b) (h2 : 0 < a) (h3 : 0 < b) :
  (m_value b) / (x_value a) = 1 / 7 :=
by
  sorry

end value_of_m_div_x_l2004_200418


namespace problem1_problem2_l2004_200419

variable (a b c : ℝ)

-- Condition: a, b, c are positive numbers and a + b + c = 1
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a + b + c = 1)

-- Problem 1: Prove that a^2 / b + b^2 / c + c^2 / a ≥ 1
theorem problem1 : a^2 / b + b^2 / c + c^2 / a ≥ 1 :=
by sorry

-- Problem 2: Prove that ab + bc + ca ≤ 1 / 3
theorem problem2 : ab + bc + ca ≤ 1 / 3 :=
by sorry

end problem1_problem2_l2004_200419


namespace remainder_of_3_pow_2023_mod_7_l2004_200451

theorem remainder_of_3_pow_2023_mod_7 :
  (3^2023) % 7 = 3 := 
by
  sorry

end remainder_of_3_pow_2023_mod_7_l2004_200451


namespace equation_of_line_is_correct_l2004_200499

/-! Given the circle x^2 + y^2 + 2x - 4y + a = 0 with a < 3 and the midpoint of the chord AB as C(-2, 3), prove that the equation of the line l that intersects the circle at points A and B is x - y + 5 = 0. -/

theorem equation_of_line_is_correct (a : ℝ) (h : a < 3) :
  ∃ l : ℝ × ℝ × ℝ, (l = (1, -1, 5)) ∧ 
  (∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y + a = 0 → 
    (x - y + 5 = 0)) :=
sorry

end equation_of_line_is_correct_l2004_200499


namespace point_in_second_quadrant_l2004_200459

def point (x : ℤ) (y : ℤ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant : point (-1) 3 = true := by
  sorry

end point_in_second_quadrant_l2004_200459


namespace relation_between_3a5_3b5_l2004_200409

theorem relation_between_3a5_3b5 (a b : ℝ) (h : a > b) : 3 * a + 5 > 3 * b + 5 := by
  sorry

end relation_between_3a5_3b5_l2004_200409


namespace divides_8x_7y_l2004_200486

theorem divides_8x_7y (x y : ℤ) (h : 5 ∣ (x + 9 * y)) : 5 ∣ (8 * x + 7 * y) :=
sorry

end divides_8x_7y_l2004_200486


namespace mark_bought_5_pounds_of_apples_l2004_200429

noncomputable def cost_of_tomatoes (pounds_tomatoes : ℕ) (cost_per_pound_tomato : ℝ) : ℝ :=
  pounds_tomatoes * cost_per_pound_tomato

noncomputable def cost_of_apples (total_spent : ℝ) (cost_of_tomatoes : ℝ) : ℝ :=
  total_spent - cost_of_tomatoes

noncomputable def pounds_of_apples (cost_of_apples : ℝ) (cost_per_pound_apples : ℝ) : ℝ :=
  cost_of_apples / cost_per_pound_apples

theorem mark_bought_5_pounds_of_apples (pounds_tomatoes : ℕ) (cost_per_pound_tomato : ℝ) 
  (total_spent : ℝ) (cost_per_pound_apples : ℝ) :
  pounds_tomatoes = 2 →
  cost_per_pound_tomato = 5 →
  total_spent = 40 →
  cost_per_pound_apples = 6 →
  pounds_of_apples (cost_of_apples total_spent (cost_of_tomatoes pounds_tomatoes cost_per_pound_tomato)) cost_per_pound_apples = 5 := by
  intros h1 h2 h3 h4
  sorry

end mark_bought_5_pounds_of_apples_l2004_200429


namespace average_output_l2004_200449

theorem average_output (t1 t2: ℝ) (cogs1 cogs2 : ℕ) (h1 : t1 = cogs1 / 36) (h2 : t2 = cogs2 / 60) (h_sum_cogs : cogs1 = 60) (h_sum_more_cogs : cogs2 = 60) (h_sum_time : t1 + t2 = 60 / 36 + 60 / 60) : 
  (cogs1 + cogs2) / (t1 + t2) = 45 := by
  sorry

end average_output_l2004_200449


namespace sqrt_difference_eq_neg_six_sqrt_two_l2004_200483

theorem sqrt_difference_eq_neg_six_sqrt_two :
  (Real.sqrt ((5 - 3 * Real.sqrt 2)^2)) - (Real.sqrt ((5 + 3 * Real.sqrt 2)^2)) = -6 * Real.sqrt 2 := 
sorry

end sqrt_difference_eq_neg_six_sqrt_two_l2004_200483


namespace pair_a_n_uniq_l2004_200460

theorem pair_a_n_uniq (a n : ℕ) (h_pos_a : 0 < a) (h_pos_n : 0 < n) (h_eq : 3^n = a^2 - 16) : a = 5 ∧ n = 2 := 
by 
  sorry

end pair_a_n_uniq_l2004_200460


namespace number_of_t_in_T_such_that_f_t_mod_8_eq_0_l2004_200468

def f (x : ℤ) : ℤ := x^3 + 2 * x^2 + 3 * x + 4

def T := { n : ℤ | 0 ≤ n ∧ n ≤ 50 }

theorem number_of_t_in_T_such_that_f_t_mod_8_eq_0 : 
  (∃ t ∈ T, f t % 8 = 0) = false := sorry

end number_of_t_in_T_such_that_f_t_mod_8_eq_0_l2004_200468


namespace range_of_a_l2004_200456

theorem range_of_a (b c a : ℝ) (h_intersect : ∀ x : ℝ, 
  (x ^ 2 - 2 * b * x + b ^ 2 + c = 1 - x → x = b )) 
  (h_vertex : c = a * b ^ 2) :
  a ≥ (-1 / 5) ∧ a ≠ 0 := 
by 
-- Proof skipped
sorry

end range_of_a_l2004_200456


namespace perimeter_C_correct_l2004_200457

variables (x y : ℕ)

def perimeter_A (x y : ℕ) := 6 * x + 2 * y
def perimeter_B (x y : ℕ) := 4 * x + 6 * y
def perimeter_C (x y : ℕ) := 2 * x + 6 * y

theorem perimeter_C_correct (x y : ℕ) (h1 : 6 * x + 2 * y = 56) (h2 : 4 * x + 6 * y = 56) :
  2 * x + 6 * y = 40 :=
sorry

end perimeter_C_correct_l2004_200457


namespace problem_statement_l2004_200471

theorem problem_statement :
  let a := -12
  let b := 45
  let c := -45
  let d := 54
  8 * a + 4 * b + 2 * c + d = 48 :=
by
  sorry

end problem_statement_l2004_200471


namespace janelle_initial_green_marbles_l2004_200427

def initial_green_marbles (blue_bags : ℕ) (marbles_per_bag : ℕ) (gift_green : ℕ) (gift_blue : ℕ) (remaining_marbles : ℕ) : ℕ :=
  let blue_marbles := blue_bags * marbles_per_bag
  let remaining_blue_marbles := blue_marbles - gift_blue
  let remaining_green_marbles := remaining_marbles - remaining_blue_marbles
  remaining_green_marbles + gift_green

theorem janelle_initial_green_marbles :
  initial_green_marbles 6 10 6 8 72 = 26 :=
by
  rfl

end janelle_initial_green_marbles_l2004_200427


namespace solve_equation_l2004_200401

theorem solve_equation (x: ℝ) (h : (5 - x)^(1/3) = -5/2) : x = 165/8 :=
by
  -- proof skipped
  sorry

end solve_equation_l2004_200401


namespace sqrt3_mul_sqrt12_eq_6_l2004_200462

noncomputable def sqrt3 := Real.sqrt 3
noncomputable def sqrt12 := Real.sqrt 12

theorem sqrt3_mul_sqrt12_eq_6 : sqrt3 * sqrt12 = 6 :=
by
  sorry

end sqrt3_mul_sqrt12_eq_6_l2004_200462


namespace least_three_digit_product_12_l2004_200436

-- Problem statement: Find the least three-digit number whose digits multiply to 12
theorem least_three_digit_product_12 : ∃ (n : ℕ), n ≥ 100 ∧ n < 1000 ∧ (∃ a b c : ℕ, n = 100 * a + 10 * b + c ∧ a * b * c = 12 ∧ n = 126) :=
by
  sorry

end least_three_digit_product_12_l2004_200436


namespace tangent_lines_ln_e_proof_l2004_200422

noncomputable def tangent_tangent_ln_e : Prop :=
  ∀ (x₁ x₂ : ℝ) 
  (h₁ : x₁ > 0) 
  (h₁_eq : x₂ = -Real.log x₁)
  (h₂_eq : Real.log x₁ - 1 = (Real.exp x₂) * (1 - x₂)),
  (2 / (x₁ - 1) + x₂ = -1)

theorem tangent_lines_ln_e_proof : tangent_tangent_ln_e :=
  sorry

end tangent_lines_ln_e_proof_l2004_200422


namespace factor_values_l2004_200411

theorem factor_values (a b : ℤ) :
  (∀ s : ℂ, s^2 - s - 1 = 0 → a * s^15 + b * s^14 + 1 = 0) ∧
  (∀ t : ℂ, t^2 - t - 1 = 0 → a * t^15 + b * t^14 + 1 = 0) →
  a = 377 ∧ b = -610 :=
by
  sorry

end factor_values_l2004_200411


namespace find_real_pairs_l2004_200497

theorem find_real_pairs (x y : ℝ) (h1 : x^5 + y^5 = 33) (h2 : x + y = 3) :
  (x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2) :=
by
  sorry

end find_real_pairs_l2004_200497


namespace quadrilateral_is_trapezium_l2004_200412

-- Define the angles of the quadrilateral and the sum of the angles condition
variables {x : ℝ}
def sum_of_angles (x : ℝ) : Prop := x + 5 * x + 2 * x + 4 * x = 360

-- State the theorem
theorem quadrilateral_is_trapezium (x : ℝ) (h : sum_of_angles x) : 
  30 + 150 = 180 ∧ 60 + 120 = 180 → is_trapezium :=
sorry

end quadrilateral_is_trapezium_l2004_200412


namespace scientific_notation_of_300670_l2004_200458

theorem scientific_notation_of_300670 : ∃ a : ℝ, ∃ n : ℤ, (1 ≤ |a| ∧ |a| < 10) ∧ 300670 = a * 10^n ∧ a = 3.0067 ∧ n = 5 :=
  by
    sorry

end scientific_notation_of_300670_l2004_200458


namespace range_of_a_l2004_200498

-- Given conditions
def condition1 (x : ℝ) := (4 + x) / 3 > (x + 2) / 2
def condition2 (x : ℝ) (a : ℝ) := (x + a) / 2 < 0

-- The statement to prove
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, condition1 x → condition2 x a → x < 2) → a ≤ -2 :=
sorry

end range_of_a_l2004_200498


namespace hyperbola_asymptotes_l2004_200463

-- Define the given hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop := (y^2 / 4) - (x^2 / 9) = 1

-- Define the standard form of hyperbola asymptotes equations
def asymptotes_eq (x y : ℝ) : Prop := 2 * x + 3 * y = 0 ∨ 2 * x - 3 * y = 0

-- The final proof statement
theorem hyperbola_asymptotes (x y : ℝ) (h : hyperbola_eq x y) : asymptotes_eq x y :=
    sorry

end hyperbola_asymptotes_l2004_200463


namespace social_gathering_married_men_fraction_l2004_200428

theorem social_gathering_married_men_fraction {W : ℝ} {MW : ℝ} {MM : ℝ} 
  (hW_pos : 0 < W)
  (hMW_def : MW = W * (3/7))
  (hMM_def : MM = W - MW)
  (h_total_people : 2 * MM + MW = 11) :
  (MM / 11) = 4/11 :=
by {
  sorry
}

end social_gathering_married_men_fraction_l2004_200428


namespace grant_room_proof_l2004_200478

/-- Danielle's apartment has 6 rooms -/
def danielle_rooms : ℕ := 6

/-- Heidi's apartment has 3 times as many rooms as Danielle's apartment -/
def heidi_rooms : ℕ := 3 * danielle_rooms

/-- Jenny's apartment has 5 more rooms than Danielle's apartment -/
def jenny_rooms : ℕ := danielle_rooms + 5

/-- Lina's apartment has 7 rooms -/
def lina_rooms : ℕ := 7

/-- The total number of rooms from Danielle, Heidi, Jenny,
    and Lina's apartments -/
def total_rooms : ℕ := danielle_rooms + heidi_rooms + jenny_rooms + lina_rooms

/-- Grant's apartment has 1/3 less rooms than 1/9 of the
    combined total of rooms from Danielle's, Heidi's, Jenny's, and Lina's apartments -/
def grant_rooms : ℕ := (total_rooms / 9) - (total_rooms / 9) / 3

/-- Prove that Grant's apartment has 3 rooms -/
theorem grant_room_proof : grant_rooms = 3 :=
by
  sorry

end grant_room_proof_l2004_200478


namespace initial_average_age_is_16_l2004_200435

-- Given conditions
variable (N : ℕ) (newPersons : ℕ) (avgNewPersonsAge : ℝ) (totalPersonsAfter : ℕ) (avgAgeAfter : ℝ)
variable (initial_avg_age : ℝ) -- This represents the initial average age (A) we need to prove

-- The specific values from the problem
def N_value : ℕ := 20
def newPersons_value : ℕ := 20
def avgNewPersonsAge_value : ℝ := 15
def totalPersonsAfter_value : ℕ := 40
def avgAgeAfter_value : ℝ := 15.5

-- Theorem statement to prove that the initial average age is 16 years
theorem initial_average_age_is_16 (h1 : N = N_value) (h2 : newPersons = newPersons_value) 
  (h3 : avgNewPersonsAge = avgNewPersonsAge_value) (h4 : totalPersonsAfter = totalPersonsAfter_value) 
  (h5 : avgAgeAfter = avgAgeAfter_value) : initial_avg_age = 16 := by
  sorry

end initial_average_age_is_16_l2004_200435


namespace computation_result_l2004_200466

theorem computation_result :
  2 + 8 * 3 - 4 + 7 * 2 / 2 * 3 = 43 :=
by
  sorry

end computation_result_l2004_200466


namespace complete_the_square_l2004_200490

theorem complete_the_square (x : ℝ) : 
  (∃ a b : ℝ, (x^2 + 10 * x - 3 = 0) → ((x + a)^2 = b) ∧ b = 28) :=
sorry

end complete_the_square_l2004_200490


namespace find_coordinates_of_point_M_l2004_200480

theorem find_coordinates_of_point_M :
  ∃ (M : ℝ × ℝ), 
    (M.1 > 0) ∧ (M.2 < 0) ∧ 
    abs M.2 = 12 ∧ 
    abs M.1 = 4 ∧ 
    M = (4, -12) :=
by
  sorry

end find_coordinates_of_point_M_l2004_200480


namespace download_time_l2004_200406

def first_segment_size : ℝ := 30
def first_segment_rate : ℝ := 5
def second_segment_size : ℝ := 40
def second_segment_rate1 : ℝ := 10
def second_segment_rate2 : ℝ := 2
def third_segment_size : ℝ := 20
def third_segment_rate1 : ℝ := 8
def third_segment_rate2 : ℝ := 4

theorem download_time :
  let time_first := first_segment_size / first_segment_rate
  let time_second := (10 / second_segment_rate1) + (10 / second_segment_rate2) + (10 / second_segment_rate1) + (10 / second_segment_rate2)
  let time_third := (10 / third_segment_rate1) + (10 / third_segment_rate2)
  time_first + time_second + time_third = 21.75 :=
by
  sorry

end download_time_l2004_200406


namespace find_number_l2004_200426

theorem find_number : ∃ x : ℝ, x^2 + 100 = (x - 20)^2 ∧ x = 7.5 :=
by {
  sorry
}

end find_number_l2004_200426


namespace simon_students_l2004_200469

theorem simon_students (S L : ℕ) (h1 : S = 4 * L) (h2 : S + L = 2500) : S = 2000 :=
by {
  sorry
}

end simon_students_l2004_200469


namespace max_stickers_l2004_200485

theorem max_stickers (n_players : ℕ) (avg_stickers : ℕ) (min_stickers : ℕ) 
  (total_players : n_players = 22) 
  (average : avg_stickers = 4) 
  (minimum : ∀ i, i < n_players → min_stickers = 1) :
  ∃ max_sticker : ℕ, max_sticker = 67 :=
by
  sorry

end max_stickers_l2004_200485


namespace proposition_contrapositive_same_truth_value_l2004_200432

variable {P : Prop}

theorem proposition_contrapositive_same_truth_value (P : Prop) :
  (P → P) = (¬P → ¬P) := 
sorry

end proposition_contrapositive_same_truth_value_l2004_200432
