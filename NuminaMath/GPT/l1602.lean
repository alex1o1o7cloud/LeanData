import Mathlib

namespace gcd_360_504_l1602_160296

theorem gcd_360_504 : Int.gcd 360 504 = 72 := by
  sorry

end gcd_360_504_l1602_160296


namespace evaluate_expression_l1602_160250

theorem evaluate_expression (a : ℝ) (h : a = 2) : 
    (a / (a^2 - 1) - 1 / (a^2 - 1)) = 1 / 3 := by
  sorry

end evaluate_expression_l1602_160250


namespace chandu_work_days_l1602_160257

theorem chandu_work_days (W : ℝ) (c : ℝ) 
  (anand_rate : ℝ := W / 7) 
  (bittu_rate : ℝ := W / 8) 
  (chandu_rate : ℝ := W / c) 
  (completed_in_7_days : 3 * anand_rate + 2 * bittu_rate + 2 * chandu_rate = W) : 
  c = 7 :=
by
  sorry

end chandu_work_days_l1602_160257


namespace sum_of_roots_cubic_l1602_160271

theorem sum_of_roots_cubic :
  let a := 3
  let b := 7
  let c := -12
  let d := -4
  let roots_sum := -(b / a)
  roots_sum = -2.33 :=
by
  sorry

end sum_of_roots_cubic_l1602_160271


namespace consequence_of_implication_l1602_160251

-- Define the conditions
variable (A B : Prop)

-- State the theorem to prove
theorem consequence_of_implication (h : B → A) : A → B := 
  sorry

end consequence_of_implication_l1602_160251


namespace total_spent_on_computer_l1602_160235

def initial_cost_of_pc : ℕ := 1200
def sale_price_old_card : ℕ := 300
def cost_new_card : ℕ := 500

theorem total_spent_on_computer : 
  (initial_cost_of_pc + (cost_new_card - sale_price_old_card)) = 1400 :=
by
  sorry

end total_spent_on_computer_l1602_160235


namespace fraction_of_janes_age_is_five_eighths_l1602_160299

/-- Jane's current age -/
def jane_current_age : ℕ := 34

/-- Number of years ago Jane stopped babysitting -/
def years_since_stopped_babysitting : ℕ := 10

/-- Current age of the oldest child Jane could have babysat -/
def oldest_child_current_age : ℕ := 25

/-- Calculate Jane's age when she stopped babysitting -/
def jane_age_when_stopped_babysitting : ℕ := jane_current_age - years_since_stopped_babysitting

/-- Calculate the child's age when Jane stopped babysitting -/
def oldest_child_age_when_jane_stopped : ℕ := oldest_child_current_age - years_since_stopped_babysitting 

/-- Calculate the fraction of Jane's age that the child could be at most -/
def babysitting_age_fraction : ℚ := (oldest_child_age_when_jane_stopped : ℚ) / (jane_age_when_stopped_babysitting : ℚ)

theorem fraction_of_janes_age_is_five_eighths :
  babysitting_age_fraction = 5 / 8 :=
by 
  -- Declare the proof steps (this part is the placeholder as proof is not required)
  sorry

end fraction_of_janes_age_is_five_eighths_l1602_160299


namespace projected_increase_is_25_l1602_160268

variable (R P : ℝ) -- variables for last year's revenue and projected increase in percentage

-- Conditions
axiom h1 : ∀ (R : ℝ), R > 0
axiom h2 : ∀ (P : ℝ), P/100 ≥ 0
axiom h3 : ∀ (R : ℝ), 0.75 * R = 0.60 * (R + (P/100) * R)

-- Goal
theorem projected_increase_is_25 (R : ℝ) : P = 25 :=
by {
    -- import the required axioms and provide the necessary proof
    apply sorry
}

end projected_increase_is_25_l1602_160268


namespace function_satisfies_equation_l1602_160276

theorem function_satisfies_equation (y : ℝ → ℝ) (h : ∀ x : ℝ, y x = Real.exp (x + x^2) + 2 * Real.exp x) :
  ∀ x : ℝ, deriv y x - y x = 2 * x * Real.exp (x + x^2) :=
by {
  sorry
}

end function_satisfies_equation_l1602_160276


namespace raft_drift_time_l1602_160245

theorem raft_drift_time (s : ℝ) (v_down v_up v_c : ℝ) 
  (h1 : v_down = s / 3) 
  (h2 : v_up = s / 4) 
  (h3 : v_down = v_c + v_c)
  (h4 : v_up = v_c - v_c) :
  v_c = s / 24 → (s / v_c) = 24 := 
by
  sorry

end raft_drift_time_l1602_160245


namespace x_cubed_plus_y_cubed_l1602_160293

theorem x_cubed_plus_y_cubed (x y : ℝ) (h1 : x + y = 8) (h2 : x * y = 14) : x^3 + y^3 = 176 :=
sorry

end x_cubed_plus_y_cubed_l1602_160293


namespace parabola_intersects_x_axis_l1602_160249

theorem parabola_intersects_x_axis :
  ∀ m : ℝ, (m^2 - m - 1 = 0) → (-2 * m^2 + 2 * m + 2023 = 2021) :=
by 
intros m hm
/-
  Given condition: m^2 - m - 1 = 0
  We need to show: -2 * m^2 + 2 * m + 2023 = 2021
-/
sorry

end parabola_intersects_x_axis_l1602_160249


namespace total_money_received_a_l1602_160231

-- Define the partners and their capitals
structure Partner :=
  (name : String)
  (capital : ℕ)
  (isWorking : Bool)

def a : Partner := { name := "a", capital := 3500, isWorking := true }
def b : Partner := { name := "b", capital := 2500, isWorking := false }

-- Define the total profit
def totalProfit : ℕ := 9600

-- Define the managing fee as 10% of total profit
def managingFee (total : ℕ) : ℕ := (10 * total) / 100

-- Define the remaining profit after deducting the managing fee
def remainingProfit (total : ℕ) (fee : ℕ) : ℕ := total - fee

-- Calculate the share of remaining profit based on capital contribution
def share (capital totalCapital remaining : ℕ) : ℕ := (capital * remaining) / totalCapital

-- Theorem to prove the total money received by partner a
theorem total_money_received_a :
  let totalCapitals := a.capital + b.capital
  let fee := managingFee totalProfit
  let remaining := remainingProfit totalProfit fee
  let aShare := share a.capital totalCapitals remaining
  (fee + aShare) = 6000 :=
by
  sorry

end total_money_received_a_l1602_160231


namespace hyperbola_equation_l1602_160275

theorem hyperbola_equation:
  let F1 := (-Real.sqrt 10, 0)
  let F2 := (Real.sqrt 10, 0)
  ∃ P : ℝ × ℝ, 
    (let PF1 := (P.1 - F1.1, P.2 - F1.2);
     let PF2 := (P.1 - F2.1, P.2 - F2.2);
     (PF1.1 * PF2.1 + PF1.2 * PF2.2 = 0) ∧ 
     ((Real.sqrt (PF1.1^2 + PF1.2^2)) * (Real.sqrt (PF2.1^2 + PF2.2^2)) = 2)) →
    (∃ a b : ℝ, (a^2 = 9 ∧ b^2 = 1) ∧ 
                (∀ x y : ℝ, 
                 (a ≠ 0 ∧ (x^2 / a^2) - (y^2 / b^2) = 1 ↔ 
                  ∃ P : ℝ × ℝ, 
                    let PF1 := (P.1 - F1.1, P.2 - F1.2);
                    let PF2 := (P.1 - F2.1, P.2 - F2.2);
                    PF1.1 * PF2.1 + PF1.2 * PF2.2 = 0 ∧ 
                    (Real.sqrt (PF1.1^2 + PF1.2^2)) * (Real.sqrt (PF2.1^2 + PF2.2^2)) = 2)))
:= by
sorry

end hyperbola_equation_l1602_160275


namespace trapezoid_area_l1602_160244

theorem trapezoid_area (a b H : ℝ) (h_lat1 : a = 10) (h_lat2 : b = 8) (h_height : H = b) : 
∃ S : ℝ, S = 104 :=
by sorry

end trapezoid_area_l1602_160244


namespace range_of_a_l1602_160206

theorem range_of_a (a x y : ℝ)
  (h1 : x + y = 3 * a + 4)
  (h2 : x - y = 7 * a - 4)
  (h3 : 3 * x - 2 * y < 11) : a < 1 :=
sorry

end range_of_a_l1602_160206


namespace algebraic_expression_value_l1602_160262

theorem algebraic_expression_value (m : ℝ) (h : m^2 - m - 2 = 0) : 2 * m^2 - 2 * m = 4 := by
  sorry

end algebraic_expression_value_l1602_160262


namespace calculate_visits_to_water_fountain_l1602_160239

-- Define the distance from the desk to the fountain
def distance_desk_to_fountain : ℕ := 30

-- Define the total distance Mrs. Hilt walked
def total_distance_walked : ℕ := 120

-- Define the distance of a round trip (desk to fountain and back)
def round_trip_distance : ℕ := 2 * distance_desk_to_fountain

-- Define the number of round trips and hence the number of times to water fountain
def number_of_visits : ℕ := total_distance_walked / round_trip_distance

theorem calculate_visits_to_water_fountain:
    number_of_visits = 2 := 
by
    sorry

end calculate_visits_to_water_fountain_l1602_160239


namespace quadratic_union_nonempty_l1602_160242

theorem quadratic_union_nonempty (a : ℝ) :
  (∃ x : ℝ, x^2 - (a-2)*x - 2*a + 4 = 0) ∨ (∃ y : ℝ, y^2 + (2*a-3)*y + 2*a^2 - a - 3 = 0) ↔
    a ≤ -6 ∨ (-7/2) ≤ a ∧ a ≤ (3/2) ∨ a ≥ 2 :=
sorry

end quadratic_union_nonempty_l1602_160242


namespace correct_relationship_5_25_l1602_160292

theorem correct_relationship_5_25 : 5^2 = 25 :=
by
  sorry

end correct_relationship_5_25_l1602_160292


namespace prob1_prob2_l1602_160254

variables (x y a b c : ℝ)

-- Proof for the first problem
theorem prob1 :
  3 * x^2 * (-3 * x * y)^2 - x^2 * (x^2 * y^2 - 2 * x) = 26 * x^4 * y^2 + 2 * x^3 := 
sorry

-- Proof for the second problem
theorem prob2 :
  -2 * (-a^2 * b * c)^2 * (1 / 2) * a * (b * c)^3 - (-a * b * c)^3 * (-a * b * c)^2 = 0 :=
sorry

end prob1_prob2_l1602_160254


namespace decimal_to_fraction_l1602_160207

theorem decimal_to_fraction (x : ℚ) (h : x = 3.68) : x = 92 / 25 := by
  sorry

end decimal_to_fraction_l1602_160207


namespace sequence_property_l1602_160277

theorem sequence_property (a : ℕ+ → ℤ) (h_add : ∀ p q : ℕ+, a (p + q) = a p + a q) (h_a2 : a 2 = -6) :
  a 10 = -30 := 
sorry

end sequence_property_l1602_160277


namespace carrie_pants_l1602_160225

theorem carrie_pants (P : ℕ) (shirts := 4) (pants := P) (jackets := 2)
  (shirt_cost := 8) (pant_cost := 18) (jacket_cost := 60)
  (total_cost := shirts * shirt_cost + jackets * jacket_cost + pants * pant_cost)
  (total_cost_half := 94) :
  total_cost = 188 → total_cost_half = 94 → total_cost = 2 * total_cost_half → P = 2 :=
by
  intros h_total h_half h_relation
  sorry

end carrie_pants_l1602_160225


namespace union_of_A_and_B_l1602_160232

def set_A : Set Int := {0, 1}
def set_B : Set Int := {0, -1}

theorem union_of_A_and_B : set_A ∪ set_B = {-1, 0, 1} := by
  sorry

end union_of_A_and_B_l1602_160232


namespace greatest_three_digit_multiple_of_17_l1602_160246

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, (n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ (∀ m : ℕ, (m % 17 = 0 ∧ 100 ≤ m ∧ m ≤ 999) → n ≥ m)) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l1602_160246


namespace employee_salary_l1602_160224

theorem employee_salary (X Y : ℝ) (h1 : X = 1.2 * Y) (h2 : X + Y = 528) : Y = 240 :=
by
  sorry

end employee_salary_l1602_160224


namespace complement_of_A_l1602_160222

-- Definition of the universal set U and the set A
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | (x - 1) * (x + 2) > 0}

-- Theorem statement for the complement of A in U
theorem complement_of_A:
  (U \ A) = {x | -2 ≤ x ∧ x ≤ 1} :=
  sorry

end complement_of_A_l1602_160222


namespace isosceles_triangle_vertex_angle_l1602_160270

theorem isosceles_triangle_vertex_angle (θ : ℝ) (h₀ : θ = 80) (h₁ : ∃ (x y z : ℝ), (x = y ∨ y = z ∨ z = x) ∧ x + y + z = 180) : θ = 80 ∨ θ = 20 := 
sorry

end isosceles_triangle_vertex_angle_l1602_160270


namespace central_angle_of_sector_l1602_160256

theorem central_angle_of_sector (r S : ℝ) (h_r : r = 2) (h_S : S = 4) : 
  ∃ α : ℝ, α = 2 ∧ S = (1/2) * α * r^2 := 
by 
  sorry

end central_angle_of_sector_l1602_160256


namespace michael_has_more_flying_robots_l1602_160204

theorem michael_has_more_flying_robots (tom_robots michael_robots : ℕ) (h_tom : tom_robots = 3) (h_michael : michael_robots = 12) :
  michael_robots / tom_robots = 4 :=
by
  sorry

end michael_has_more_flying_robots_l1602_160204


namespace number_of_three_digit_integers_l1602_160220

-- Defining the set of available digits
def digits : List ℕ := [3, 5, 8, 9]

-- Defining the property for selecting a digit without repetition
def no_repetition (l : List ℕ) : Prop :=
  ∀ (d : ℕ), d ∈ l → l.filter (fun x => x = d) = [d]

-- The main theorem stating the number of three-digit integers that can be formed
theorem number_of_three_digit_integers (h : no_repetition digits) : 
  ∃ n : ℕ, n = 24 :=
by
  sorry

end number_of_three_digit_integers_l1602_160220


namespace discount_problem_l1602_160229

theorem discount_problem (m : ℝ) (h : (200 * (1 - m / 100)^2 = 162)) : m = 10 :=
sorry

end discount_problem_l1602_160229


namespace original_price_hat_l1602_160219

theorem original_price_hat 
  (x : ℝ)
  (discounted_price := x / 5)
  (final_price := discounted_price * 1.2)
  (h : final_price = 8) :
  x = 100 / 3 :=
by
  sorry

end original_price_hat_l1602_160219


namespace abs_sum_inequality_l1602_160215

theorem abs_sum_inequality (k : ℝ) : (∀ x : ℝ, |x + 2| + |x + 1| > k) → k < 1 := 
sorry

end abs_sum_inequality_l1602_160215


namespace collinear_a_b_l1602_160211

-- Define the vectors a and b
def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (1, -2)

-- Definition of collinearity of vectors
def collinear (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u = (k * v.1, k * v.2) ∨ v = (k * u.1, k * u.2)

-- Statement to prove
theorem collinear_a_b : collinear a b :=
by
  sorry

end collinear_a_b_l1602_160211


namespace initial_deadline_l1602_160228

theorem initial_deadline (D : ℝ) :
  (∀ (n : ℝ), (10 * 20) / 4 = n / 1) → 
  (∀ (m : ℝ), 8 * 75 = m * 3) →
  (∀ (d1 d2 : ℝ), d1 = 20 ∧ d2 = 93.75 → D = d1 + d2) →
  D = 113.75 :=
by {
  sorry
}

end initial_deadline_l1602_160228


namespace impossible_to_fill_grid_l1602_160214

def is_impossible : Prop :=
  ∀ (grid : Fin 3 → Fin 3 → ℕ), 
  (∀ i j, grid i j ≠ grid i (j + 1) ∧ grid i j ≠ grid (i + 1) j) →
  (∀ i, (grid i 0) * (grid i 1) * (grid i 2) = 2005) →
  (∀ j, (grid 0 j) * (grid 1 j) * (grid 2 j) = 2005) →
  (grid 0 0) * (grid 1 1) * (grid 2 2) = 2005 →
  (grid 0 2) * (grid 1 1) * (grid 2 0) = 2005 →
  False

theorem impossible_to_fill_grid : is_impossible :=
  sorry

end impossible_to_fill_grid_l1602_160214


namespace students_apply_colleges_l1602_160218

    -- Define that there are 5 students
    def students : Nat := 5

    -- Each student has 3 choices of colleges
    def choices_per_student : Nat := 3

    -- The number of different ways the students can apply
    def number_of_ways : Nat := choices_per_student ^ students

    theorem students_apply_colleges : number_of_ways = 3 ^ 5 :=
    by
        -- Proof will be done here
        sorry
    
end students_apply_colleges_l1602_160218


namespace books_on_shelf_l1602_160273

-- Step definitions based on the conditions
def initial_books := 38
def marta_books_removed := 10
def tom_books_removed := 5
def tom_books_added := 12

-- Final number of books on the shelf
def final_books : ℕ := initial_books - marta_books_removed - tom_books_removed + tom_books_added

-- Theorem statement to prove the final number of books
theorem books_on_shelf : final_books = 35 :=
by 
  -- Proof for the statement goes here
  sorry

end books_on_shelf_l1602_160273


namespace real_possible_b_values_quadratic_non_real_roots_l1602_160200

theorem real_possible_b_values_quadratic_non_real_roots :
  {b : ℝ | ∃ (a c : ℝ), a = 1 ∧ c = 16 ∧ (b^2 - 4*a*c < 0)} = {b : ℝ | -8 < b ∧ b < 8} :=
by 
  sorry

end real_possible_b_values_quadratic_non_real_roots_l1602_160200


namespace conference_games_l1602_160263

theorem conference_games (teams_per_division : ℕ) (divisions : ℕ) 
  (intradivision_games_per_team : ℕ) (interdivision_games_per_team : ℕ) 
  (total_teams : ℕ) (total_games : ℕ) : 
  total_teams = teams_per_division * divisions →
  intradivision_games_per_team = (teams_per_division - 1) * 2 →
  interdivision_games_per_team = teams_per_division →
  total_games = (total_teams * (intradivision_games_per_team + interdivision_games_per_team)) / 2 →
  total_games = 133 :=
by
  intros
  sorry

end conference_games_l1602_160263


namespace quadratic_roots_real_or_imaginary_l1602_160212

theorem quadratic_roots_real_or_imaginary (a b c d: ℝ) (h1: a > 0) (h2: b > 0) (h3: c > 0) (h4: d > 0) 
(h_distinct: a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :
∃ (A B C: ℝ), (A = a ∨ A = b ∨ A = c ∨ A = d) ∧ (B = a ∨ B = b ∨ B = c ∨ B = d) ∧ (C = a ∨ C = b ∨ C = c ∨ C = d) ∧ 
(A ≠ B) ∧ (A ≠ C) ∧ (B ≠ C) ∧ 
((1 - 4*B*C ≥ 0 ∧ 1 - 4*C*A ≥ 0 ∧ 1 - 4*A*B ≥ 0) ∨ (1 - 4*B*C < 0 ∧ 1 - 4*C*A < 0 ∧ 1 - 4*A*B < 0)) :=
by
  sorry

end quadratic_roots_real_or_imaginary_l1602_160212


namespace number_in_tenth_group_l1602_160286

-- Number of students
def students : ℕ := 1000

-- Number of groups
def groups : ℕ := 100

-- Interval between groups
def interval : ℕ := students / groups

-- First number drawn
def first_number : ℕ := 6

-- Number drawn from n-th group given first_number and interval
def number_in_group (n : ℕ) : ℕ := first_number + interval * (n - 1)

-- Statement to prove
theorem number_in_tenth_group :
  number_in_group 10 = 96 :=
by
  sorry

end number_in_tenth_group_l1602_160286


namespace second_group_num_persons_l1602_160243

def man_hours (num_persons : ℕ) (days : ℕ) (hours_per_day : ℕ) : ℕ :=
  num_persons * days * hours_per_day

theorem second_group_num_persons :
  ∀ (x : ℕ),
    let first_group_man_hours := man_hours 36 12 5
    let second_group_days := 12
    let second_group_hours_per_day := 6
    (first_group_man_hours = man_hours x second_group_days second_group_hours_per_day) →
    x = 30 :=
by
  intros x first_group_man_hours second_group_days second_group_hours_per_day h
  sorry

end second_group_num_persons_l1602_160243


namespace weight_of_gravel_l1602_160267

theorem weight_of_gravel (total_weight : ℝ) (weight_sand : ℝ) (weight_water : ℝ) (weight_gravel : ℝ) 
  (h1 : total_weight = 48)
  (h2 : weight_sand = (1/3) * total_weight)
  (h3 : weight_water = (1/2) * total_weight)
  (h4 : weight_gravel = total_weight - (weight_sand + weight_water)) :
  weight_gravel = 8 :=
sorry

end weight_of_gravel_l1602_160267


namespace girl_speed_l1602_160210

theorem girl_speed (distance time : ℝ) (h_distance : distance = 96) (h_time : time = 16) : distance / time = 6 :=
by
  sorry

end girl_speed_l1602_160210


namespace remainder_3_pow_405_mod_13_l1602_160236

theorem remainder_3_pow_405_mod_13 : (3^405) % 13 = 1 :=
by
  sorry

end remainder_3_pow_405_mod_13_l1602_160236


namespace players_count_l1602_160217

theorem players_count (total_socks : ℕ) (socks_per_player : ℕ) (h1 : total_socks = 16) (h2 : socks_per_player = 2) :
  total_socks / socks_per_player = 8 :=
by
  sorry

end players_count_l1602_160217


namespace b_arithmetic_sequence_max_S_n_l1602_160294

variables {a : ℕ → ℝ} {b : ℕ → ℝ} {S : ℕ → ℝ}

-- Conditions
noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n m, a m ≠ 0 → a n = a (n + 1) * a (m-1) / (a m)

axiom a_pos_terms : ∀ n, 0 < a n
axiom a11_eight : a 11 = 8
axiom b_log : ∀ n, b n = Real.log (a n) / Real.log 2
axiom b4_seventeen : b 4 = 17

-- Question I: Prove b_n is an arithmetic sequence with common difference -2
theorem b_arithmetic_sequence (d : ℝ) (h_d : d = (-2)) :
  ∃ d, ∀ n, b (n + 1) - b n = d :=
sorry

-- Question II: Find the maximum value of S_n
theorem max_S_n : ∃ n, S n = 144 :=
sorry

end b_arithmetic_sequence_max_S_n_l1602_160294


namespace who_wears_which_dress_l1602_160297

-- Define the possible girls
inductive Girl
| Katya | Olya | Liza | Rita
deriving DecidableEq

-- Define the possible dresses
inductive Dress
| Pink | Green | Yellow | Blue
deriving DecidableEq

-- Define the fact that each girl is wearing a dress
structure Wearing (girl : Girl) (dress : Dress) : Prop

-- Define the conditions
theorem who_wears_which_dress :
  (¬ Wearing Girl.Katya Dress.Pink ∧ ¬ Wearing Girl.Katya Dress.Blue) ∧
  (∀ g1 g2 g3, Wearing g1 Dress.Green → (Wearing g2 Dress.Pink ∧ Wearing g3 Dress.Yellow → (g2 = Girl.Liza ∧ (g3 = Girl.Rita)) ∨ (g3 = Girl.Liza ∧ g2 = Girl.Rita))) ∧
  (¬ Wearing Girl.Rita Dress.Green ∧ ¬ Wearing Girl.Rita Dress.Blue) ∧
  (∀ g1 g2, (Wearing g1 Dress.Pink ∧ Wearing g2 Dress.Yellow) → Girl.Olya = g2 ∧ Girl.Rita = g1) →
  (Wearing Girl.Katya Dress.Green ∧ Wearing Girl.Olya Dress.Blue ∧ Wearing Girl.Liza Dress.Pink ∧ Wearing Girl.Rita Dress.Yellow) :=
by
  sorry

end who_wears_which_dress_l1602_160297


namespace xiao_liang_correct_l1602_160287

theorem xiao_liang_correct :
  ∀ (x : ℕ), (0 ≤ x ∧ x ≤ 26 ∧ 30 - x ≤ 24 ∧ 26 - x ≤ 20) →
  let boys_A := x
  let girls_A := 30 - x
  let boys_B := 26 - x
  let girls_B := 24 - girls_A
  ∃ k : ℤ, boys_A - girls_B = 6 := 
by 
  sorry

end xiao_liang_correct_l1602_160287


namespace male_athletes_sampled_l1602_160284

-- Define the total number of athletes
def total_athletes : Nat := 98

-- Define the number of female athletes
def female_athletes : Nat := 42

-- Define the probability of being selected
def selection_probability : ℚ := 2 / 7

-- Calculate the number of male athletes
def male_athletes : Nat := total_athletes - female_athletes

-- State the theorem about the number of male athletes sampled
theorem male_athletes_sampled : male_athletes * selection_probability = 16 :=
by
  sorry

end male_athletes_sampled_l1602_160284


namespace volume_set_points_sum_l1602_160205

-- Defining the problem conditions
def rectangular_parallelepiped_length : ℝ := 5
def rectangular_parallelepiped_width : ℝ := 6
def rectangular_parallelepiped_height : ℝ := 7
def unit_extension : ℝ := 1

-- Defining what we need to prove
theorem volume_set_points_sum :
  let V_box : ℝ := rectangular_parallelepiped_length * rectangular_parallelepiped_width * rectangular_parallelepiped_height
  let V_ext : ℝ := 2 * (unit_extension * rectangular_parallelepiped_length * rectangular_parallelepiped_width 
                  + unit_extension * rectangular_parallelepiped_length * rectangular_parallelepiped_height 
                  + unit_extension * rectangular_parallelepiped_width * rectangular_parallelepiped_height)
  let V_cyl : ℝ := 18 * π
  let V_sph : ℝ := (4 / 3) * π
  let V_total : ℝ := V_box + V_ext + V_cyl + V_sph
  let m : ℕ := 1272
  let n : ℕ := 58
  let p : ℕ := 3
  V_total = (m : ℝ) + (n : ℝ) * π / (p : ℝ) ∧ (m + n + p = 1333)
  := by
  sorry

end volume_set_points_sum_l1602_160205


namespace center_of_large_hexagon_within_small_hexagon_l1602_160265

-- Define a structure for a regular hexagon with the necessary properties
structure RegularHexagon (α : Type) [LinearOrderedField α] :=
  (center : α × α)      -- Coordinates of the center
  (side_length : α)      -- Length of the side

-- Define the conditions: two regular hexagons with specific side length relationship
variables {α : Type} [LinearOrderedField α]
def hexagon_large : RegularHexagon α := 
  {center := (0, 0), side_length := 2}

def hexagon_small : RegularHexagon α := 
  {center := (0, 0), side_length := 1}

-- The theorem to prove
theorem center_of_large_hexagon_within_small_hexagon (hl : RegularHexagon α) (hs : RegularHexagon α) 
  (hc : hs.side_length = hl.side_length / 2) : (hl.center = hs.center) → 
  (∀ (x y : α × α), x = hs.center → (∃ r, y = hl.center → (y.1 - x.1) ^ 2 + (y.2 - x.2) ^ 2 < r ^ 2)) :=
by sorry

end center_of_large_hexagon_within_small_hexagon_l1602_160265


namespace solve_for_x_l1602_160252

theorem solve_for_x : ∀ (x : ℤ), (5 * x - 2) * 4 = (3 * (6 * x - 6)) → x = -5 :=
by
  intro x
  intro h
  sorry

end solve_for_x_l1602_160252


namespace no_natural_m_n_exists_l1602_160255

theorem no_natural_m_n_exists (m n : ℕ) : 
  (0.07 = (1 : ℝ) / m + (1 : ℝ) / n) → False :=
by
  -- Normally, the proof would go here, but it's not required by the prompt
  sorry

end no_natural_m_n_exists_l1602_160255


namespace running_time_15mph_l1602_160279

theorem running_time_15mph (x y z : ℝ) (h1 : x + y + z = 14) (h2 : 15 * x + 10 * y + 8 * z = 164) :
  x = 3 :=
sorry

end running_time_15mph_l1602_160279


namespace obtuse_angle_probability_l1602_160260

-- Defining the vertices of the pentagon
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨0, 3⟩
def B : Point := ⟨5, 0⟩
def C : Point := ⟨8, 0⟩
def D : Point := ⟨8, 5⟩
def E : Point := ⟨0, 5⟩

def is_interior (P : Point) : Prop :=
  -- A condition to define if a point is inside the pentagon
  sorry

def is_obtuse_angle (A B P : Point) : Prop :=
  -- Condition for angle APB to be obtuse
  sorry

noncomputable def probability_obtuse_angle :=
  -- Probability calculation
  let area_pentagon := 40
  let area_circle := (34 * Real.pi) / 4
  let area_outside_circle := area_pentagon - area_circle
  area_outside_circle / area_pentagon

theorem obtuse_angle_probability :
  ∀ P : Point, is_interior P → ∃! p : ℝ, p = (160 - 34 * Real.pi) / 160 :=
sorry

end obtuse_angle_probability_l1602_160260


namespace jam_jars_weight_l1602_160209

noncomputable def jars_weight 
    (initial_suitcase_weight : ℝ) 
    (perfume_weight_oz : ℝ) (num_perfume : ℕ)
    (chocolate_weight_lb : ℝ)
    (soap_weight_oz : ℝ) (num_soap : ℕ)
    (total_return_weight : ℝ)
    (oz_to_lb : ℝ) : ℝ :=
  initial_suitcase_weight 
  + (num_perfume * perfume_weight_oz) / oz_to_lb 
  + chocolate_weight_lb 
  + (num_soap * soap_weight_oz) / oz_to_lb

theorem jam_jars_weight
    (initial_suitcase_weight : ℝ := 5)
    (perfume_weight_oz : ℝ := 1.2) (num_perfume : ℕ := 5)
    (chocolate_weight_lb : ℝ := 4)
    (soap_weight_oz : ℝ := 5) (num_soap : ℕ := 2)
    (total_return_weight : ℝ := 11)
    (oz_to_lb : ℝ := 16) :
    jars_weight initial_suitcase_weight perfume_weight_oz num_perfume
      chocolate_weight_lb soap_weight_oz num_soap total_return_weight oz_to_lb + (jars_weight initial_suitcase_weight perfume_weight_oz num_perfume
      chocolate_weight_lb soap_weight_oz num_soap total_return_weight oz_to_lb) = 1 :=
by
  sorry

end jam_jars_weight_l1602_160209


namespace trigonometric_expression_value_l1602_160264

theorem trigonometric_expression_value :
  (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180)) = 4 :=
by
  sorry

end trigonometric_expression_value_l1602_160264


namespace arrange_descending_order_l1602_160216

noncomputable def a := 8 ^ 0.7
noncomputable def b := 8 ^ 0.9
noncomputable def c := 2 ^ 0.8

theorem arrange_descending_order :
    b > a ∧ a > c := by
  sorry

end arrange_descending_order_l1602_160216


namespace birds_landing_l1602_160281

theorem birds_landing (initial_birds total_birds birds_landed : ℤ) 
  (h_initial : initial_birds = 12) 
  (h_total : total_birds = 20) :
  birds_landed = total_birds - initial_birds :=
by
  sorry

end birds_landing_l1602_160281


namespace social_studies_score_l1602_160291

-- Step d): Translate to Lean 4
theorem social_studies_score 
  (K E S SS : ℝ)
  (h1 : (K + E + S) / 3 = 89)
  (h2 : (K + E + S + SS) / 4 = 90) :
  SS = 93 :=
by
  -- We'll leave the mathematics formal proof details to Lean.
  sorry

end social_studies_score_l1602_160291


namespace walking_rate_ratio_l1602_160201

theorem walking_rate_ratio :
  let T := 16
  let T' := 12
  (T : ℚ) / (T' : ℚ) = (4 : ℚ) / (3 : ℚ) := 
by
  sorry

end walking_rate_ratio_l1602_160201


namespace domain_ln_2_minus_x_is_interval_l1602_160272

noncomputable def domain_ln_2_minus_x : Set Real := { x : Real | 2 - x > 0 }

theorem domain_ln_2_minus_x_is_interval : domain_ln_2_minus_x = Set.Iio 2 :=
by
  sorry

end domain_ln_2_minus_x_is_interval_l1602_160272


namespace simplify_frac_and_find_cd_l1602_160259

theorem simplify_frac_and_find_cd :
  ∀ (m : ℤ), ∃ (c d : ℤ), 
    (c * m + d = (6 * m + 12) / 3) ∧ (c = 2) ∧ (d = 4) ∧ (c / d = 1 / 2) :=
by
  sorry

end simplify_frac_and_find_cd_l1602_160259


namespace operations_correctness_l1602_160288

theorem operations_correctness (a b : ℝ) : 
  ((-ab)^2 ≠ -a^2 * b^2)
  ∧ (a^3 * a^2 ≠ a^6)
  ∧ ((a^3)^4 ≠ a^7)
  ∧ (b^2 + b^2 = 2 * b^2) :=
by
  sorry

end operations_correctness_l1602_160288


namespace locus_of_P_l1602_160221

variables {x y : ℝ}
variables {x0 y0 : ℝ}

-- The initial ellipse equation
def ellipse (x y : ℝ) : Prop :=
  x^2 / 20 + y^2 / 16 = 1

-- Point M is on the ellipse
def point_M (x0 y0 : ℝ) : Prop :=
  ellipse x0 y0

-- The equation of P, symmetric to transformations applied to point Q derived from M
theorem locus_of_P 
  (hx0 : x0^2 / 20 + y0^2 / 16 = 1) :
  ∃ x y, (x^2 / 20 + y^2 / 36 = 1) ∧ y ≠ 0 :=
sorry

end locus_of_P_l1602_160221


namespace column_1000_is_B_l1602_160233

-- Definition of the column pattern
def columnPattern : List String := ["B", "C", "D", "E", "F", "E", "D", "C", "B", "A"]

-- Function to determine the column for a given integer
def columnOf (n : Nat) : String :=
  columnPattern.get! ((n - 2) % 10)

-- The theorem we want to prove
theorem column_1000_is_B : columnOf 1000 = "B" :=
by
  sorry

end column_1000_is_B_l1602_160233


namespace quadratic_smaller_solution_l1602_160237

theorem quadratic_smaller_solution : ∀ (x : ℝ), x^2 - 9 * x + 20 = 0 → x = 4 ∨ x = 5 :=
by
  sorry

end quadratic_smaller_solution_l1602_160237


namespace min_workers_for_profit_l1602_160253

def revenue (n : ℕ) : ℕ := 240 * n
def cost (n : ℕ) : ℕ := 600 + 200 * n

theorem min_workers_for_profit (n : ℕ) (h : 240 * n > 600 + 200 * n) : n >= 16 :=
by {
  -- Placeholder for the proof steps (which are not required per instructions)
  sorry
}

end min_workers_for_profit_l1602_160253


namespace a7_plus_a11_l1602_160269

variable {a : ℕ → ℤ} (d : ℤ) (a₁ : ℤ)

-- Definitions based on given conditions
def S_n (n : ℕ) := (n * (2 * a₁ + (n - 1) * d)) / 2
def a_n (n : ℕ) := a₁ + (n - 1) * d

-- Condition: S_17 = 51
axiom h : S_n 17 = 51

-- Theorem to prove the question is equivalent to the answer
theorem a7_plus_a11 (h : S_n 17 = 51) : a_n 7 + a_n 11 = 6 :=
by
  -- This is where you'd fill in the actual proof, but we'll use sorry for now
  sorry

end a7_plus_a11_l1602_160269


namespace solve_inequality_l1602_160285

theorem solve_inequality (x : ℝ) : 2 * x + 4 > 0 ↔ x > -2 := sorry

end solve_inequality_l1602_160285


namespace perimeter_of_ABFCDE_l1602_160203

theorem perimeter_of_ABFCDE {side : ℝ} (h : side = 12) : 
  ∃ perimeter : ℝ, perimeter = 84 :=
by
  sorry

end perimeter_of_ABFCDE_l1602_160203


namespace system_of_inequalities_l1602_160234

theorem system_of_inequalities (p : ℝ) (h1 : 18 * p < 10) (h2 : p > 0.5) : (0.5 < p ∧ p < 5 / 9) :=
by sorry

end system_of_inequalities_l1602_160234


namespace find_equation_of_line_l1602_160241

theorem find_equation_of_line 
  (l : ℝ → ℝ → Prop)
  (h_intersect : ∃ x y : ℝ, 2 * x + y - 8 = 0 ∧ x - 2 * y + 1 = 0 ∧ l x y)
  (h_parallel : ∀ x y : ℝ, l x y → 4 * x - 3 * y - 6 = 0) :
  ∀ x y : ℝ, l x y ↔ 4 * x - 3 * y - 6 = 0 :=
by
  sorry

end find_equation_of_line_l1602_160241


namespace angle_between_a_and_b_is_2pi_over_3_l1602_160280

open Real

variables (a b c : ℝ × ℝ)

-- Given conditions
def condition1 := a.1^2 + a.2^2 = 2  -- |a| = sqrt(2)
def condition2 := b = (-1, 1)        -- b = (-1, 1)
def condition3 := c = (2, -2)        -- c = (2, -2)
def condition4 := a.1 * (b.1 + c.1) + a.2 * (b.2 + c.2) = 1  -- a · (b + c) = 1

-- Prove the angle θ between a and b is 2π/3
theorem angle_between_a_and_b_is_2pi_over_3 :
  condition1 a → condition2 b → condition3 c → condition4 a b c →
  ∃ θ, 0 ≤ θ ∧ θ ≤ π ∧ cos θ = -(1/2) ∧ θ = 2 * π / 3 :=
by
  sorry

end angle_between_a_and_b_is_2pi_over_3_l1602_160280


namespace volume_of_water_cylinder_l1602_160258

theorem volume_of_water_cylinder :
  let r := 5
  let h := 10
  let depth := 3
  let θ := Real.arccos (3 / 5)
  let sector_area := (2 * θ) / (2 * Real.pi) * Real.pi * r^2
  let triangle_area := r * (2 * r * Real.sin θ)
  let water_surface_area := sector_area - triangle_area
  let volume := h * water_surface_area
  volume = 232.6 * Real.pi - 160 :=
by
  sorry

end volume_of_water_cylinder_l1602_160258


namespace maximum_cookies_by_andy_l1602_160227

-- Define the conditions
def total_cookies := 36
def cookies_by_andry (a : ℕ) := a
def cookies_by_alexa (a : ℕ) := 3 * a
def cookies_by_alice (a : ℕ) := 2 * a
def sum_cookies (a : ℕ) := cookies_by_andry a + cookies_by_alexa a + cookies_by_alice a

-- The theorem stating the problem and solution
theorem maximum_cookies_by_andy :
  ∃ a : ℕ, sum_cookies a = total_cookies ∧ a = 6 :=
by
  sorry

end maximum_cookies_by_andy_l1602_160227


namespace JungMinBoughtWire_l1602_160248

theorem JungMinBoughtWire
  (side_length : ℕ)
  (number_of_sides : ℕ)
  (remaining_wire : ℕ)
  (total_wire_bought : ℕ)
  (h1 : side_length = 13)
  (h2 : number_of_sides = 5)
  (h3 : remaining_wire = 8)
  (h4 : total_wire_bought = side_length * number_of_sides + remaining_wire) :
    total_wire_bought = 73 :=
by {
  sorry
}

end JungMinBoughtWire_l1602_160248


namespace tail_growth_problem_l1602_160247

def initial_tail_length : ℕ := 1
def final_tail_length : ℕ := 864
def transformations (ordinary_count cowardly_count : ℕ) : ℕ := initial_tail_length * 2^ordinary_count * 3^cowardly_count

theorem tail_growth_problem (ordinary_count cowardly_count : ℕ) :
  transformations ordinary_count cowardly_count = final_tail_length ↔ ordinary_count = 5 ∧ cowardly_count = 3 :=
by
  sorry

end tail_growth_problem_l1602_160247


namespace find_possible_f_one_l1602_160278

noncomputable def f : ℝ → ℝ := sorry

theorem find_possible_f_one (h : ∀ x y : ℝ, f (x + y) = 2 * f x * f y) :
  f 1 = 0 ∨ (∃ c : ℝ, f 0 = 1/2 ∧ f 1 = c) :=
sorry

end find_possible_f_one_l1602_160278


namespace part1_part2_l1602_160240

theorem part1 (p : ℝ) (h : p = 2 / 5) : 
  (p^2 + 2 * (3 / 5) * p^2) = 0.352 :=
by 
  rw [h]
  sorry

theorem part2 (p : ℝ) (h : p = 2 / 5) : 
  (4 * (1 / (11.32 * p^4)) + 5 * (2.4 / (11.32 * p^4)) + 6 * (3.6 / (11.32 * p^4)) + 7 * (2.16 / (11.32 * p^4))) = 4.834 :=
by 
  rw [h]
  sorry

end part1_part2_l1602_160240


namespace examination_students_total_l1602_160230

/-
  Problem Statement:
  Given:
  - 35% of the students passed the examination.
  - 546 students failed the examination.

  Prove:
  - The total number of students who appeared for the examination is 840.
-/

theorem examination_students_total (T : ℝ) (h1 : 0.35 * T + 0.65 * T = T) (h2 : 0.65 * T = 546) : T = 840 :=
by
  -- skipped proof part
  sorry

end examination_students_total_l1602_160230


namespace roots_square_sum_l1602_160282

theorem roots_square_sum (r s p q : ℝ) 
  (root_cond : ∀ x : ℝ, x^2 - 2 * p * x + 3 * q = 0 → (x = r ∨ x = s)) :
  r^2 + s^2 = 4 * p^2 - 6 * q :=
by
  sorry

end roots_square_sum_l1602_160282


namespace abs_ineq_solution_range_l1602_160261

theorem abs_ineq_solution_range (a : ℝ) :
  (∃ x : ℝ, |x + 1| - |x - 2| > a) → a < 3 :=
by
  sorry

end abs_ineq_solution_range_l1602_160261


namespace min_max_values_l1602_160226

theorem min_max_values (x1 x2 x3 : ℝ) (h1 : x1 ≥ 0) (h2 : x2 ≥ 0) (h3 : x3 ≥ 0) (h_sum : x1 + x2 + x3 = 1) :
  1 ≤ (x1 + 3*x2 + 5*x3) * (x1 + x2 / 3 + x3 / 5) ∧ (x1 + 3*x2 + 5*x3) * (x1 + x2 / 3 + x3 / 5) ≤ 9/5 :=
by sorry

end min_max_values_l1602_160226


namespace point_c_in_second_quadrant_l1602_160274

-- Definitions for the points
def PointA : ℝ × ℝ := (1, 2)
def PointB : ℝ × ℝ := (-1, -2)
def PointC : ℝ × ℝ := (-1, 2)
def PointD : ℝ × ℝ := (1, -2)

-- Definition of the second quadrant condition
def in_second_quadrant (p : ℝ × ℝ) : Prop :=
p.1 < 0 ∧ p.2 > 0

theorem point_c_in_second_quadrant : in_second_quadrant PointC :=
sorry

end point_c_in_second_quadrant_l1602_160274


namespace quadrant_of_half_angle_in_second_quadrant_l1602_160295

theorem quadrant_of_half_angle_in_second_quadrant (θ : ℝ) (h : π / 2 < θ ∧ θ < π) :
  (0 < θ / 2 ∧ θ / 2 < π / 2) ∨ (π < θ / 2 ∧ θ / 2 < 3 * π / 2) :=
by
  sorry

end quadrant_of_half_angle_in_second_quadrant_l1602_160295


namespace triangle_side_lengths_l1602_160289

theorem triangle_side_lengths (A B C : ℝ) (a b c : ℝ) 
  (hcosA : Real.cos A = 1/4)
  (ha : a = 4)
  (hbc_sum : b + c = 6)
  (hbc_order : b < c) :
  b = 2 ∧ c = 4 := by
  sorry

end triangle_side_lengths_l1602_160289


namespace number_of_rectangles_is_24_l1602_160266

-- Define the rectangles on a 1x5 stripe
def rectangles_1x5 : ℕ := 1 + 2 + 3 + 4 + 5

-- Define the rectangles on a 1x4 stripe
def rectangles_1x4 : ℕ := 1 + 2 + 3 + 4

-- Define the overlap (intersection) adjustment
def overlap_adjustment : ℕ := 1

-- Total number of rectangles calculation
def total_rectangles : ℕ := rectangles_1x5 + rectangles_1x4 - overlap_adjustment

theorem number_of_rectangles_is_24 : total_rectangles = 24 := by
  sorry

end number_of_rectangles_is_24_l1602_160266


namespace regular_hexagon_interior_angle_deg_l1602_160213

theorem regular_hexagon_interior_angle_deg (n : ℕ) (h1 : n = 6) :
  let sum_of_interior_angles : ℕ := (n - 2) * 180
  let each_angle : ℕ := sum_of_interior_angles / n
  each_angle = 120 := by
  sorry

end regular_hexagon_interior_angle_deg_l1602_160213


namespace luca_lost_more_weight_l1602_160283

theorem luca_lost_more_weight (barbi_kg_month : ℝ) (luca_kg_year : ℝ) (months_in_year : ℕ) (years : ℕ) 
(h_barbi : barbi_kg_month = 1.5) (h_luca : luca_kg_year = 9) (h_months_in_year : months_in_year = 12) (h_years : years = 11) : 
  (luca_kg_year * years) - (barbi_kg_month * months_in_year * (years / 11)) = 81 := 
by 
  sorry

end luca_lost_more_weight_l1602_160283


namespace jessica_age_l1602_160208

theorem jessica_age 
  (j g : ℚ)
  (h1 : g = 15 * j) 
  (h2 : g - j = 60) : 
  j = 30 / 7 :=
by
  sorry

end jessica_age_l1602_160208


namespace part1_part2_l1602_160238

theorem part1 (a b : ℝ) (h1 : ∀ x, a * x^2 - 3 * x + 2 > 0 ↔ x < 1 ∨ x > b) (hb : b > 1) : a = 1 ∧ b = 2 :=
sorry

theorem part2 (k : ℝ) (x y : ℝ) (hx : x > 0) (hy : y > 0) (a b : ℝ) 
  (ha : a = 1) (hb : b = 2) 
  (h2 : a / x + b / y = 1)
  (h3 : 2 * x + y ≥ k^2 + k + 2) : -3 ≤ k ∧ k ≤ 2 :=
sorry

end part1_part2_l1602_160238


namespace max_value_of_expression_l1602_160223

theorem max_value_of_expression 
  (a b c : ℝ)
  (h : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) :
  6 * a + 3 * b + 10 * c ≤ 3.2 :=
sorry

end max_value_of_expression_l1602_160223


namespace maria_bottles_proof_l1602_160290

theorem maria_bottles_proof 
    (initial_bottles : ℕ)
    (drank_bottles : ℕ)
    (current_bottles : ℕ)
    (bought_bottles : ℕ) 
    (h1 : initial_bottles = 14)
    (h2 : drank_bottles = 8)
    (h3 : current_bottles = 51)
    (h4 : current_bottles = initial_bottles - drank_bottles + bought_bottles) :
  bought_bottles = 45 :=
by
  sorry

end maria_bottles_proof_l1602_160290


namespace maximum_volume_prism_l1602_160298

-- Define the conditions
variables {l w h : ℝ}
axiom area_sum_eq : 2 * h * l + l * w = 30

-- Define the volume of the prism
def volume (l w h : ℝ) : ℝ := l * w * h

-- Statement to be proved
theorem maximum_volume_prism : 
  (∃ l w h : ℝ, 2 * h * l + l * w = 30 ∧ 
  ∀ u v t : ℝ, 2 * t * u + u * v = 30 → l * w * h ≥ u * v * t) → volume l w h = 112.5 :=
by
  sorry

end maximum_volume_prism_l1602_160298


namespace T_100_gt_T_99_l1602_160202

-- Definition: T(n) denotes the number of ways to place n objects of weights 1, 2, ..., n on a balance such that the sum of the weights in each pan is the same.
def T (n : ℕ) : ℕ := sorry

-- Theorem we need to prove
theorem T_100_gt_T_99 : T 100 > T 99 := 
sorry

end T_100_gt_T_99_l1602_160202
