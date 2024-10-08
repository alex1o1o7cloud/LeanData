import Mathlib

namespace quadratic_positive_difference_l2_2525

theorem quadratic_positive_difference :
  ∀ x : ℝ, x^2 - 5 * x + 15 = x + 55 → x = 10 ∨ x = -4 →
  |10 - (-4)| = 14 :=
by
  intro x h1 h2
  have h3 : x = 10 ∨ x = -4 := h2
  have h4 : |10 - (-4)| = 14 := by norm_num
  exact h4

end quadratic_positive_difference_l2_2525


namespace problem_1_problem_2_l2_2305

def simplify_calc : Prop :=
  125 * 3.2 * 25 = 10000

def solve_equation : Prop :=
  ∀ x: ℝ, 24 * (x - 12) = 16 * (x - 4) → x = 28

theorem problem_1 : simplify_calc :=
by
  sorry

theorem problem_2 : solve_equation :=
by
  sorry

end problem_1_problem_2_l2_2305


namespace last_person_is_knight_l2_2689

-- Definitions for the conditions:
def first_whispered_number := 7
def last_announced_number_first_game := 3
def last_whispered_number_second_game := 5
def first_announced_number_second_game := 2

-- Definitions to represent the roles:
inductive Role
| knight
| liar

-- Definition of the last person in the first game being a knight:
def last_person_first_game_role := Role.knight

theorem last_person_is_knight 
  (h1 : Role.liar = Role.liar)
  (h2 : last_announced_number_first_game = 3)
  (h3 : first_whispered_number = 7)
  (h4 : first_announced_number_second_game = 2)
  (h5 : last_whispered_number_second_game = 5) :
  last_person_first_game_role = Role.knight :=
sorry

end last_person_is_knight_l2_2689


namespace triangle_inequality_l2_2269

variable {α β γ a b c : ℝ}

theorem triangle_inequality (h1: α ≥ β) (h2: β ≥ γ) (h3: a ≥ b) (h4: b ≥ c) (h5: α ≥ γ) (h6: a ≥ c) :
  a * α + b * β + c * γ ≥ a * β + b * γ + c * α :=
by
  sorry

end triangle_inequality_l2_2269


namespace part_I_extreme_values_part_II_three_distinct_real_roots_part_III_compare_sizes_l2_2459

noncomputable def f (x : ℝ) (p q : ℝ) : ℝ := (1 / 3) * x^3 + (1 / 2) * (p - 1) * x^2 + q * x

theorem part_I_extreme_values : 
  (∀ x, f x (-3) 3 = (1 / 3) * x^3 - 2 * x^2 + 3 * x) → 
  (f 1 (-3) 3 = f 3 (-3) 3) := 
sorry

theorem part_II_three_distinct_real_roots : 
  (∀ x, f x (-3) 3 = (1 / 3) * x^3 - 2 * x^2 + 3 * x) → 
  (∀ g : ℝ → ℝ, g x = f x (-3) 3 - 1 → 
  (∀ x, g x ≠ 0) → 
  ∃ a b c, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ g a = 0 ∧ g b = 0 ∧ g c = 0) :=
sorry

theorem part_III_compare_sizes (x1 x2 p a l q: ℝ) :
  f (x : ℝ) (-3) 3 = (1 / 3) * x^3 - 2 * x^2 + 3 * x → 
  x1 < x2 → 
  x2 - x1 > l → 
  x1 > a → 
  (a^2 + p * a + q) > x1 := 
sorry

end part_I_extreme_values_part_II_three_distinct_real_roots_part_III_compare_sizes_l2_2459


namespace triangle_angle_C_l2_2367

open Real

theorem triangle_angle_C (b c : ℝ) (B C : ℝ) (hb : b = sqrt 2) (hc : c = 1) (hB : B = 45) : C = 30 :=
sorry

end triangle_angle_C_l2_2367


namespace find_rs_l2_2950

-- Define a structure to hold the conditions
structure Conditions (r s : ℝ) : Prop :=
  (positive_r : 0 < r)
  (positive_s : 0 < s)
  (eq1 : r^3 + s^3 = 1)
  (eq2 : r^6 + s^6 = (15 / 16))

-- State the theorem
theorem find_rs (r s : ℝ) (h : Conditions r s) : rs = 1 / (48 : ℝ)^(1/3) :=
by
  sorry

end find_rs_l2_2950


namespace problem1_problem2_problem3_problem4_l2_2699

theorem problem1 : (-20 + (-14) - (-18) - 13) = -29 := by
  sorry

theorem problem2 : (-6 * (-2) / (1 / 8)) = 96 := by
  sorry

theorem problem3 : (-24 * (-3 / 4 - 5 / 6 + 7 / 8)) = 17 := by
  sorry

theorem problem4 : (-1^4 - (1 - 0.5) * (1 / 3) * (-3)^2) = -5 / 2 := by
  sorry

end problem1_problem2_problem3_problem4_l2_2699


namespace coin_difference_l2_2897

noncomputable def max_value (p n d : ℕ) : ℕ := p + 5 * n + 10 * d
noncomputable def min_value (p n d : ℕ) : ℕ := p + 5 * n + 10 * d

theorem coin_difference (p n d : ℕ) (h₁ : p + n + d = 3030) (h₂ : 10 ≤ p) (h₃ : 10 ≤ n) (h₄ : 10 ≤ d) :
  max_value 10 10 3010 - min_value 3010 10 10 = 27000 := by
  sorry

end coin_difference_l2_2897


namespace concession_stand_total_revenue_l2_2168

theorem concession_stand_total_revenue :
  let hot_dog_price : ℝ := 1.50
  let soda_price : ℝ := 0.50
  let total_items_sold : ℕ := 87
  let hot_dogs_sold : ℕ := 35
  let sodas_sold := total_items_sold - hot_dogs_sold
  let revenue_from_hot_dogs := hot_dogs_sold * hot_dog_price
  let revenue_from_sodas := sodas_sold * soda_price
  revenue_from_hot_dogs + revenue_from_sodas = 78.50 :=
by {
  -- Proof will go here
  sorry
}

end concession_stand_total_revenue_l2_2168


namespace not_enough_evidence_to_show_relationship_l2_2899

noncomputable def isEvidenceToShowRelationship (table : Array (Array Nat)) : Prop :=
  ∃ evidence : Bool, ¬evidence

theorem not_enough_evidence_to_show_relationship :
  isEvidenceToShowRelationship #[#[5, 15, 20], #[40, 10, 50], #[45, 25, 70]] :=
sorry 

end not_enough_evidence_to_show_relationship_l2_2899


namespace find_x0_l2_2479

noncomputable def f (x : ℝ) : ℝ := 13 - 8 * x + x^2

theorem find_x0 :
  (∃ x0 : ℝ, deriv f x0 = 4) → ∃ x0 : ℝ, x0 = 6 :=
by
  sorry

end find_x0_l2_2479


namespace find_length_of_second_train_l2_2090

def length_of_second_train (L : ℚ) : Prop :=
  let length_first_train : ℚ := 300
  let speed_first_train : ℚ := 120 * 1000 / 3600
  let speed_second_train : ℚ := 80 * 1000 / 3600
  let crossing_time : ℚ := 9
  let relative_speed : ℚ := speed_first_train + speed_second_train
  let total_distance : ℚ := relative_speed * crossing_time
  total_distance = length_first_train + L

theorem find_length_of_second_train :
  ∃ (L : ℚ), length_of_second_train L ∧ L = 199.95 := 
by
  sorry

end find_length_of_second_train_l2_2090


namespace total_income_by_nth_year_max_m_and_k_range_l2_2861

noncomputable def total_income (a : ℝ) (k : ℝ) (n : ℕ) : ℝ :=
  (6 - (n + 6) * 0.1 ^ n) * a

theorem total_income_by_nth_year (a : ℝ) (n : ℕ) :
  total_income a 0.1 n = (6 - (n + 6) * 0.1 ^ n) * a :=
sorry

theorem max_m_and_k_range (a : ℝ) (m : ℕ) :
  (m = 4 ∧ 1 ≤ 1) ∧ (∀ k, k ≥ 1 → m = 4) :=
sorry

end total_income_by_nth_year_max_m_and_k_range_l2_2861


namespace find_k_l2_2545

theorem find_k (k : ℝ) (x₁ x₂ y₁ y₂ : ℝ) 
  (h_parabola_A : y₁ = x₁^2)
  (h_parabola_B : y₂ = x₂^2)
  (h_line_A : y₁ = x₁ - k)
  (h_line_B : y₂ = x₂ - k)
  (h_midpoint : (y₁ + y₂) / 2 = 1) 
  (h_sum_x : x₁ + x₂ = 1) :
  k = -1 / 2 :=
by sorry

end find_k_l2_2545


namespace proof_problem_l2_2638

variable {a b : ℝ}
variable (cond : sqrt a > sqrt b)

theorem proof_problem (h1 : a > b) (h2 : 0 ≤ a) (h3 : 0 ≤ b) :
  (a^2 > b^2) ∧
  ((b + 1) / (a + 1) > b / a) ∧
  (b + 1 / (b + 1) ≥ 1) :=
by
  sorry

end proof_problem_l2_2638


namespace at_least_two_even_l2_2223

theorem at_least_two_even (x y z : ℤ) (u : ℤ)
  (h : x^2 + y^2 + z^2 = u^2) : (↑x % 2 = 0) ∨ (↑y % 2 = 0) → (↑x % 2 = 0) ∨ (↑z % 2 = 0) ∨ (↑y % 2 = 0) := 
by
  sorry

end at_least_two_even_l2_2223


namespace quadratic_expression_l2_2203

theorem quadratic_expression (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 6) 
  (h2 : 2 * x + 3 * y = 8) : 
  13 * x^2 + 22 * x * y + 13 * y^2 = 98.08 := 
by sorry

end quadratic_expression_l2_2203


namespace fermat_1000_units_digit_l2_2769

-- Define Fermat numbers
def FermatNumber (n : ℕ) : ℕ := 2 ^ (2 ^ n) + 1

-- Define a function to extract the units digit
def units_digit (n : ℕ) : ℕ := n % 10

-- The theorem to be proven
theorem fermat_1000_units_digit : units_digit (FermatNumber 1000) = 7 := 
by sorry

end fermat_1000_units_digit_l2_2769


namespace candidate1_fails_by_l2_2635

-- Define the total marks (T), passing marks (P), percentage marks (perc1 and perc2), and the extra marks.
def T : ℝ := 600
def P : ℝ := 160
def perc1 : ℝ := 0.20
def perc2 : ℝ := 0.30
def extra_marks : ℝ := 20

-- Define the marks obtained by the candidates.
def marks_candidate1 : ℝ := perc1 * T
def marks_candidate2 : ℝ := perc2 * T

-- The theorem stating the number of marks by which the first candidate fails.
theorem candidate1_fails_by (h_pass: perc2 * T = P + extra_marks) : P - marks_candidate1 = 40 :=
by
  -- The proof would go here.
  sorry

end candidate1_fails_by_l2_2635


namespace runner_injury_point_l2_2905

-- Define the initial setup conditions
def total_distance := 40
def second_half_time := 10
def first_half_additional_time := 5

-- Prove that given the conditions, the runner injured her foot at 20 miles.
theorem runner_injury_point : 
  ∃ (d v : ℝ), (d = 5 * v) ∧ (total_distance - d = 5 * v) ∧ (10 = second_half_time) ∧ (first_half_additional_time = 5) ∧ (d = 20) :=
by
  sorry

end runner_injury_point_l2_2905


namespace turn_all_black_l2_2944

def invertColor (v : Vertex) (G : Graph) : Graph := sorry

theorem turn_all_black (G : Graph) (n : ℕ) (whiteBlack : Vertex → Bool) :
  (∀ v : Vertex, whiteBlack v = false) :=
by
 -- Providing the base case for induction
  induction n with 
  | zero => sorry -- The base case for graphs with one vertex
  | succ n ih =>
    -- Inductive step: assume true for graph with n vertices and prove for graph with n+1 vertices
    sorry

end turn_all_black_l2_2944


namespace items_left_in_store_l2_2125

def restocked : ℕ := 4458
def sold : ℕ := 1561
def storeroom : ℕ := 575

theorem items_left_in_store : restocked - sold + storeroom = 3472 := by
  sorry

end items_left_in_store_l2_2125


namespace building_height_l2_2975

theorem building_height
    (flagpole_height : ℝ)
    (flagpole_shadow_length : ℝ)
    (building_shadow_length : ℝ)
    (h : ℝ)
    (h_eq : flagpole_height / flagpole_shadow_length = h / building_shadow_length)
    (flagpole_height_eq : flagpole_height = 18)
    (flagpole_shadow_length_eq : flagpole_shadow_length = 45)
    (building_shadow_length_eq : building_shadow_length = 65) :
  h = 26 := by
  sorry

end building_height_l2_2975


namespace books_printed_l2_2730

-- Definitions of the conditions
def book_length := 600
def pages_per_sheet := 8
def total_sheets := 150

-- The theorem to prove
theorem books_printed : (total_sheets * pages_per_sheet / book_length) = 2 := by
  sorry

end books_printed_l2_2730


namespace tie_rate_correct_l2_2904

-- Define the fractions indicating win rates for Amy, Lily, and John
def AmyWinRate : ℚ := 4 / 9
def LilyWinRate : ℚ := 1 / 3
def JohnWinRate : ℚ := 1 / 6

-- Define the fraction they tie
def TieRate : ℚ := 1 / 18

-- The theorem for proving the tie rate
theorem tie_rate_correct : AmyWinRate + LilyWinRate + JohnWinRate = 17 / 18 → (1 : ℚ) - (17 / 18) = TieRate :=
by
  sorry -- Proof is omitted

-- Define the win rate sums and tie rate equivalence
example : (AmyWinRate + LilyWinRate + JohnWinRate = 17 / 18) ∧ (TieRate = 1 - 17 / 18) :=
by
  sorry -- Proof is omitted

end tie_rate_correct_l2_2904


namespace add_fractions_l2_2252

theorem add_fractions : (2 / 3 : ℚ) + (7 / 8) = 37 / 24 := 
by sorry

end add_fractions_l2_2252


namespace pond_length_l2_2616

-- Define the dimensions and volume of the pond
def pond_width : ℝ := 15
def pond_depth : ℝ := 5
def pond_volume : ℝ := 1500

-- Define the length variable
variable (L : ℝ)

-- State that the volume relationship holds and L is the length we're solving for
theorem pond_length :
  pond_volume = L * pond_width * pond_depth → L = 20 :=
by
  sorry

end pond_length_l2_2616


namespace find_value_l2_2510

theorem find_value 
  (x1 x2 x3 x4 x5 : ℝ)
  (condition1 : x1 + 4 * x2 + 9 * x3 + 16 * x4 + 25 * x5 = 2)
  (condition2 : 4 * x1 + 9 * x2 + 16 * x3 + 25 * x4 + 36 * x5 = 15)
  (condition3 : 9 * x1 + 16 * x2 + 25 * x3 + 36 * x4 + 49 * x5 = 130) :
  16 * x1 + 25 * x2 + 36 * x3 + 49 * x4 + 64 * x5 = 347 :=
by
  sorry

end find_value_l2_2510


namespace factor_expression_l2_2361

variables (a : ℝ)

theorem factor_expression : (45 * a^2 + 135 * a + 90 * a^3) = 45 * a * (90 * a^2 + a + 3) :=
by sorry

end factor_expression_l2_2361


namespace triangle_altitude_length_l2_2359

-- Define the problem
theorem triangle_altitude_length (l w h : ℝ) (hl : l = 2 * w) 
  (h_triangle_area : 0.5 * l * h = 0.5 * (l * w)) : h = w := 
by 
  -- Use the provided conditions and the equation setup to continue the proof
  sorry

end triangle_altitude_length_l2_2359


namespace remainder_98_pow_50_mod_100_l2_2833

/-- 
Theorem: The remainder when \(98^{50}\) is divided by 100 is 24.
-/
theorem remainder_98_pow_50_mod_100 : (98^50 % 100) = 24 := by
  sorry

end remainder_98_pow_50_mod_100_l2_2833


namespace find_y_given_conditions_l2_2175

theorem find_y_given_conditions (x : ℤ) (y : ℤ) (h1 : x^2 - 3 * x + 7 = y + 2) (h2 : x = -5) : y = 45 :=
by
  sorry

end find_y_given_conditions_l2_2175


namespace calculate_weight_of_6_moles_HClO2_l2_2782

noncomputable def weight_of_6_moles_HClO2 := 
  let molar_mass_H := 1.01
  let molar_mass_Cl := 35.45
  let molar_mass_O := 16.00
  let molar_mass_HClO2 := molar_mass_H + molar_mass_Cl + 2 * molar_mass_O
  let moles_HClO2 := 6
  moles_HClO2 * molar_mass_HClO2

theorem calculate_weight_of_6_moles_HClO2 : weight_of_6_moles_HClO2 = 410.76 :=
by
  sorry

end calculate_weight_of_6_moles_HClO2_l2_2782


namespace circle_diameter_and_circumference_l2_2462

theorem circle_diameter_and_circumference (A : ℝ) (hA : A = 225 * π) : 
  ∃ r d C, r = 15 ∧ d = 2 * r ∧ C = 2 * π * r ∧ d = 30 ∧ C = 30 * π :=
by
  sorry

end circle_diameter_and_circumference_l2_2462


namespace simplify_fraction_l2_2628

theorem simplify_fraction :
  (4^5 + 4^3) / (4^4 - 4^2 + 2) = 544 / 121 := 
by sorry

end simplify_fraction_l2_2628


namespace range_of_a_l2_2502

theorem range_of_a (a : ℝ) : (∀ x > 0, a - x - |Real.log x| ≤ 0) → a ≤ 1 := by
  sorry

end range_of_a_l2_2502


namespace mildred_heavier_than_carol_l2_2582

def mildred_weight : ℕ := 59
def carol_weight : ℕ := 9

theorem mildred_heavier_than_carol : mildred_weight - carol_weight = 50 := 
by
  sorry

end mildred_heavier_than_carol_l2_2582


namespace unoccupied_seats_l2_2813

theorem unoccupied_seats (rows chairs_per_row seats_taken : Nat) (h1 : rows = 40)
  (h2 : chairs_per_row = 20) (h3 : seats_taken = 790) :
  rows * chairs_per_row - seats_taken = 10 :=
by
  sorry

end unoccupied_seats_l2_2813


namespace gnomes_telling_the_truth_l2_2855

-- Conditions
def gnome_height_claims : List ℕ := [60, 61, 62, 63, 64, 65, 66]

-- Question and Proof Problem in Lean 4
theorem gnomes_telling_the_truth :
  (∀ (actual_heights : List ℕ), 
    actual_heights.length = 7 →
    (∀ i, i < 7 → i > 0 → 
    actual_heights.get! i > actual_heights.get! (i - 1) → gnome_height_claims.get! i ≠ actual_heights.get! i)) →
  -- conclusion
  (∃ count, count = 1) :=
by
  sorry

end gnomes_telling_the_truth_l2_2855


namespace probability_of_purple_l2_2910

def total_faces := 10
def purple_faces := 3

theorem probability_of_purple : (purple_faces : ℚ) / (total_faces : ℚ) = 3 / 10 := 
by 
  sorry

end probability_of_purple_l2_2910


namespace poles_intersection_l2_2384

-- Define the known heights and distances
def heightOfIntersection (d h1 h2 x : ℝ) : ℝ := sorry

theorem poles_intersection :
  heightOfIntersection 120 30 60 40 = 20 := by
  sorry

end poles_intersection_l2_2384


namespace restaurant_donates_24_l2_2515

def restaurant_donation (customer_donation_per_person : ℕ) (num_customers : ℕ) (restaurant_donation_per_ten_dollars : ℕ) : ℕ :=
  let total_customer_donation := customer_donation_per_person * num_customers
  let increments_of_ten := total_customer_donation / 10
  increments_of_ten * restaurant_donation_per_ten_dollars

theorem restaurant_donates_24 :
  restaurant_donation 3 40 2 = 24 :=
by
  sorry

end restaurant_donates_24_l2_2515


namespace total_supervisors_l2_2511

theorem total_supervisors (buses : ℕ) (supervisors_per_bus : ℕ) (h1 : buses = 7) (h2 : supervisors_per_bus = 3) :
  buses * supervisors_per_bus = 21 :=
by
  sorry

end total_supervisors_l2_2511


namespace find_largest_divisor_l2_2764

def f (n : ℕ) : ℕ := (2 * n + 7) * 3 ^ n + 9

theorem find_largest_divisor :
  ∃ m : ℕ, (∀ n : ℕ, f n % m = 0) ∧ m = 36 :=
sorry

end find_largest_divisor_l2_2764


namespace students_attending_swimming_class_l2_2989

theorem students_attending_swimming_class 
  (total_students : ℕ) 
  (chess_percentage : ℕ) 
  (swimming_percentage : ℕ) 
  (number_of_students : ℕ)
  (chess_students := chess_percentage * total_students / 100)
  (swimming_students := swimming_percentage * chess_students / 100) 
  (condition1 : total_students = 2000)
  (condition2 : chess_percentage = 10)
  (condition3 : swimming_percentage = 50)
  (condition4 : number_of_students = chess_students) :
  swimming_students = 100 := 
by 
  sorry

end students_attending_swimming_class_l2_2989


namespace total_cards_l2_2302

def basketball_boxes : ℕ := 12
def cards_per_basketball_box : ℕ := 20
def football_boxes : ℕ := basketball_boxes - 5
def cards_per_football_box : ℕ := 25

theorem total_cards : basketball_boxes * cards_per_basketball_box + football_boxes * cards_per_football_box = 415 := by
  sorry

end total_cards_l2_2302


namespace f_increasing_f_at_2_solve_inequality_l2_2966

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_add (a b : ℝ) : f (a + b) = f a + f b - 1
axiom f_pos (x : ℝ) (h : x > 0) : f x > 1
axiom f_at_4 : f 4 = 5

theorem f_increasing : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2 :=
sorry

theorem f_at_2 : f 2 = 3 :=
sorry

theorem solve_inequality (m : ℝ) : f (3 * m^2 - m - 2) < 3 ↔ -1 < m ∧ m < 4 / 3 :=
sorry

end f_increasing_f_at_2_solve_inequality_l2_2966


namespace highest_score_l2_2788

-- Definitions based on conditions
variable (H L : ℕ)

-- Condition (1): H - L = 150
def condition1 : Prop := H - L = 150

-- Condition (2): H + L = 208
def condition2 : Prop := H + L = 208

-- Condition (3): Total runs in 46 innings at an average of 60, excluding two innings averages to 58
def total_runs := 60 * 46
def excluded_runs := total_runs - 2552

theorem highest_score
  (cond1 : condition1 H L)
  (cond2 : condition2 H L)
  : H = 179 :=
by sorry

end highest_score_l2_2788


namespace probability_of_chosen_figure_is_circle_l2_2457

-- Define the total number of figures and number of circles.
def total_figures : ℕ := 12
def number_of_circles : ℕ := 5

-- Define the probability calculation.
def probability_of_circle (total : ℕ) (circles : ℕ) : ℚ := circles / total

-- State the theorem using the defined conditions.
theorem probability_of_chosen_figure_is_circle : 
  probability_of_circle total_figures number_of_circles = 5 / 12 :=
by
  sorry  -- Placeholder for the actual proof.

end probability_of_chosen_figure_is_circle_l2_2457


namespace average_speed_trip_l2_2249

-- Conditions: Definitions
def distance_north_feet : ℝ := 5280
def speed_north_mpm : ℝ := 2
def speed_south_mpm : ℝ := 1

-- Question and Equivalent Proof Problem
theorem average_speed_trip :
  let distance_north_miles := distance_north_feet / 5280
  let distance_south_miles := 2 * distance_north_miles
  let total_distance_miles := distance_north_miles + distance_south_miles + distance_south_miles
  let time_north_hours := distance_north_miles / speed_north_mpm / 60
  let time_south_hours := distance_south_miles / speed_south_mpm / 60
  let time_return_hours := distance_south_miles / speed_south_mpm / 60
  let total_time_hours := time_north_hours + time_south_hours + time_return_hours
  let average_speed_mph := total_distance_miles / total_time_hours
  average_speed_mph = 76.4 := by
    sorry

end average_speed_trip_l2_2249


namespace pyramid_inscribed_sphere_radius_l2_2319

noncomputable def inscribed_sphere_radius (a : ℝ) : ℝ := 
a * Real.sqrt 2 / (2 * (2 + Real.sqrt 3))

theorem pyramid_inscribed_sphere_radius (a : ℝ) (h1 : a > 0) : 
  inscribed_sphere_radius a = a * Real.sqrt 2 / (2 * (2 + Real.sqrt 3)) :=
by
  sorry

end pyramid_inscribed_sphere_radius_l2_2319


namespace sector_max_area_l2_2675

theorem sector_max_area (r : ℝ) (α : ℝ) (S : ℝ) :
  (0 < r ∧ r < 10) ∧ (2 * r + r * α = 20) ∧ (S = (1 / 2) * r * (r * α)) →
  (α = 2 ∧ S = 25) :=
by
  sorry

end sector_max_area_l2_2675


namespace rower_rate_in_still_water_l2_2396

theorem rower_rate_in_still_water (V_m V_s : ℝ) (h1 : V_m + V_s = 16) (h2 : V_m - V_s = 12) : V_m = 14 := 
sorry

end rower_rate_in_still_water_l2_2396


namespace largest_number_in_box_l2_2095

theorem largest_number_in_box
  (a : ℕ)
  (sum_eq_480 : a + (a + 1) + (a + 2) + (a + 10) + (a + 11) + (a + 12) = 480) :
  a + 12 = 86 :=
by
  sorry

end largest_number_in_box_l2_2095


namespace distance_to_conference_l2_2270

theorem distance_to_conference (t d : ℝ) 
  (h1 : d = 40 * (t + 0.75))
  (h2 : d - 40 = 60 * (t - 1.25)) :
  d = 160 :=
by
  sorry

end distance_to_conference_l2_2270


namespace solve_for_x_l2_2929

theorem solve_for_x (x : ℝ) (h : 40 / x - 1 = 19) : x = 2 :=
by {
  sorry
}

end solve_for_x_l2_2929


namespace minimize_sum_pos_maximize_product_pos_l2_2151

def N : ℕ := 10^1001 - 1

noncomputable def find_min_sum_position : ℕ := 996

noncomputable def find_max_product_position : ℕ := 995

theorem minimize_sum_pos :
  ∀ m : ℕ, (m ≠ find_min_sum_position) → 
      (2 * 10^m + 10^(1001-m) - 10) ≥ (2 * 10^find_min_sum_position + 10^(1001-find_min_sum_position) - 10) := 
sorry

theorem maximize_product_pos :
  ∀ m : ℕ, (m ≠ find_max_product_position) → 
      ((2 * 10^m - 1) * (10^(1001 - m) - 9)) ≤ ((2 * 10^find_max_product_position - 1) * (10^(1001 - find_max_product_position) - 9)) :=
sorry

end minimize_sum_pos_maximize_product_pos_l2_2151


namespace power_mod_equivalence_l2_2566

theorem power_mod_equivalence : (7^700) % 100 = 1 := 
by 
  -- Given that (7^4) % 100 = 1
  have h : 7^4 % 100 = 1 := by sorry
  -- Use this equivalence to prove the statement
  sorry

end power_mod_equivalence_l2_2566


namespace no_solutions_cryptarithm_l2_2423

theorem no_solutions_cryptarithm : 
  ∀ (K O P H A B U y C : ℕ), 
  K ≠ O ∧ K ≠ P ∧ K ≠ H ∧ K ≠ A ∧ K ≠ B ∧ K ≠ U ∧ K ≠ y ∧ K ≠ C ∧ 
  O ≠ P ∧ O ≠ H ∧ O ≠ A ∧ O ≠ B ∧ O ≠ U ∧ O ≠ y ∧ O ≠ C ∧ 
  P ≠ H ∧ P ≠ A ∧ P ≠ B ∧ P ≠ U ∧ P ≠ y ∧ P ≠ C ∧ 
  H ≠ A ∧ H ≠ B ∧ H ≠ U ∧ H ≠ y ∧ H ≠ C ∧ 
  A ≠ B ∧ A ≠ U ∧ A ≠ y ∧ A ≠ C ∧ 
  B ≠ U ∧ B ≠ y ∧ B ≠ C ∧ 
  U ≠ y ∧ U ≠ C ∧ 
  y ≠ C ∧
  K < O ∧ O < P ∧ P > O ∧ O > H ∧ H > A ∧ A > B ∧ B > U ∧ U > P ∧ P > y ∧ y > C → 
  false :=
sorry

end no_solutions_cryptarithm_l2_2423


namespace area_of_45_45_90_triangle_l2_2518

theorem area_of_45_45_90_triangle (h : ℝ) (h_eq : h = 8 * Real.sqrt 2) : 
  ∃ (A : ℝ), A = 32 := 
by
  sorry

end area_of_45_45_90_triangle_l2_2518


namespace sum_of_numbers_l2_2664

def contains_digit (n : Nat) (d : Nat) : Prop := 
  (n / 100 = d) ∨ (n % 100 / 10 = d) ∨ (n % 10 = d)

variables {A B C : Nat}

-- Given conditions
axiom three_digit_number : A ≥ 100 ∧ A < 1000
axiom two_digit_numbers : B ≥ 10 ∧ B < 100 ∧ C ≥ 10 ∧ C < 100
axiom sum_with_sevens : contains_digit A 7 ∨ contains_digit B 7 ∨ contains_digit C 7 → A + B + C = 208
axiom sum_with_threes : contains_digit B 3 ∧ contains_digit C 3 ∧ B + C = 76

-- Main theorem to be proved
theorem sum_of_numbers : A + B + C = 247 :=
sorry

end sum_of_numbers_l2_2664


namespace more_customers_left_than_stayed_l2_2388

-- Define the initial number of customers.
def initial_customers : ℕ := 11

-- Define the number of customers who stayed behind.
def customers_stayed : ℕ := 3

-- Define the number of customers who left.
def customers_left : ℕ := initial_customers - customers_stayed

-- Prove that the number of customers who left is 5 more than those who stayed behind.
theorem more_customers_left_than_stayed : customers_left - customers_stayed = 5 := by
  -- Sorry to skip the proof 
  sorry

end more_customers_left_than_stayed_l2_2388


namespace determine_Y_in_arithmetic_sequence_matrix_l2_2274

theorem determine_Y_in_arithmetic_sequence_matrix :
  (exists a₁ a₂ a₃ a₄ a₅ : ℕ, 
    -- Conditions for the first row (arithmetic sequence with first term 3 and fifth term 15)
    a₁ = 3 ∧ a₅ = 15 ∧ 
    (∃ d₁ : ℕ, a₂ = a₁ + d₁ ∧ a₃ = a₂ + d₁ ∧ a₄ = a₃ + d₁ ∧ a₅ = a₄ + d₁) ∧

    -- Conditions for the fifth row (arithmetic sequence with first term 25 and fifth term 65)
    a₁ = 25 ∧ a₅ = 65 ∧ 
    (∃ d₅ : ℕ, a₂ = a₁ + d₅ ∧ a₃ = a₂ + d₅ ∧ a₄ = a₃ + d₅ ∧ a₅ = a₄ + d₅) ∧

    -- Middle element Y
    a₃ = 27) :=
sorry

end determine_Y_in_arithmetic_sequence_matrix_l2_2274


namespace quadratic_j_value_l2_2468

theorem quadratic_j_value (a b c : ℝ) (h : a * (0 : ℝ)^2 + b * (0 : ℝ) + c = 5 * ((0 : ℝ) - 3)^2 + 15) :
  ∃ m j n, 4 * a * (0 : ℝ)^2 + 4 * b * (0 : ℝ) + 4 * c = m * ((0 : ℝ) - j)^2 + n ∧ j = 3 :=
by
  sorry

end quadratic_j_value_l2_2468


namespace train_length_l2_2586

theorem train_length (speed_kmh : ℕ) (cross_time : ℕ) (h_speed : speed_kmh = 54) (h_time : cross_time = 9) :
  let speed_ms := speed_kmh * (1000 / 3600)
  let length_m := speed_ms * cross_time
  length_m = 135 := by
  sorry

end train_length_l2_2586


namespace taller_tree_height_l2_2443

-- Definitions and Variables
variables (h : ℝ)

-- Conditions as Definitions
def top_difference_condition := (h - 20) / h = 5 / 7

-- Proof Statement
theorem taller_tree_height (h : ℝ) (H : top_difference_condition h) : h = 70 := 
by {
  sorry
}

end taller_tree_height_l2_2443


namespace log35_28_l2_2362

variable (a b : ℝ)
variable (log : ℝ → ℝ → ℝ)

-- Conditions
axiom log14_7_eq_a : log 14 7 = a
axiom log14_5_eq_b : log 14 5 = b

-- Theorem to prove
theorem log35_28 (h1 : log 14 7 = a) (h2 : log 14 5 = b) : log 35 28 = (2 - a) / (a + b) :=
sorry

end log35_28_l2_2362


namespace value_of_a_plus_b_2023_l2_2406

theorem value_of_a_plus_b_2023 
    (x y a b : ℤ)
    (h1 : 4*x + 3*y = 11)
    (h2 : 2*x - y = 3)
    (h3 : a*x + b*y = -2)
    (h4 : b*x - a*y = 6)
    (hx : x = 2)
    (hy : y = 1) :
    (a + b) ^ 2023 = 0 := 
sorry

end value_of_a_plus_b_2023_l2_2406


namespace binomial_expansion_evaluation_l2_2537

theorem binomial_expansion_evaluation : 
  (8 ^ 4 + 4 * (8 ^ 3) * 2 + 6 * (8 ^ 2) * (2 ^ 2) + 4 * 8 * (2 ^ 3) + 2 ^ 4) = 10000 := 
by 
  sorry

end binomial_expansion_evaluation_l2_2537


namespace complement_union_l2_2257

open Set

def I : Set ℕ := {0, 1, 2, 3}
def M : Set ℕ := {0, 2}
def N : Set ℕ := {0, 2, 3}

theorem complement_union :
  compl (M ∪ N) = {1} :=
by
  sorry

end complement_union_l2_2257


namespace maximum_a_for_monotonically_increasing_interval_l2_2572

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x)
noncomputable def g (x : ℝ) : ℝ := f (x - (Real.pi / 4))

theorem maximum_a_for_monotonically_increasing_interval :
  ∀ a : ℝ, (∀ x y : ℝ, 0 ≤ x ∧ x ≤ a ∧ 0 ≤ y ∧ y ≤ a ∧ x < y → g x < g y) → a ≤ Real.pi / 4 := 
by
  sorry

end maximum_a_for_monotonically_increasing_interval_l2_2572


namespace fewer_bees_than_flowers_l2_2123

theorem fewer_bees_than_flowers :
  (5 - 3 = 2) :=
by
  sorry

end fewer_bees_than_flowers_l2_2123


namespace area_of_sector_l2_2107

theorem area_of_sector (r : ℝ) (θ : ℝ) (h1 : r = 10) (h2 : θ = π / 5) : 
  (1 / 2) * r * r * θ = 10 * π :=
by
  rw [h1, h2]
  sorry

end area_of_sector_l2_2107


namespace mindy_messages_total_l2_2608

theorem mindy_messages_total (P : ℕ) (h1 : 83 = 9 * P - 7) : 83 + P = 93 :=
  by
    sorry

end mindy_messages_total_l2_2608


namespace B_gives_C_100_meters_start_l2_2209

-- Definitions based on given conditions
variables (Va Vb Vc : ℝ) (T : ℝ)

-- Assume the conditions based on the problem statement
def race_condition_1 := Va = 1000 / T
def race_condition_2 := Vb = 900 / T
def race_condition_3 := Vc = 850 / T

-- Theorem stating that B can give C a 100 meter start
theorem B_gives_C_100_meters_start
  (h1 : race_condition_1 Va T)
  (h2 : race_condition_2 Vb T)
  (h3 : race_condition_3 Vc T) :
  (Vb = (1000 - 100) / T) :=
by
  -- Utilize conditions h1, h2, and h3
  sorry

end B_gives_C_100_meters_start_l2_2209


namespace find_value_of_m_l2_2114

theorem find_value_of_m (m : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 - 4*x + 2*y + m = 0 ∧ (x - 2)^2 + (y + 1)^2 = 4) →
  m = 1 :=
sorry

end find_value_of_m_l2_2114


namespace mean_proportional_AC_is_correct_l2_2557

-- Definitions based on conditions
def AB := 4
def BC (AC : ℝ) := AB - AC

-- Lean theorem
theorem mean_proportional_AC_is_correct (AC : ℝ) :
  AC > 0 ∧ AC^2 = AB * BC AC ↔ AC = 2 * Real.sqrt 5 - 2 := 
sorry

end mean_proportional_AC_is_correct_l2_2557


namespace taxi_ride_cost_l2_2822

theorem taxi_ride_cost :
  let base_fare : ℝ := 2.00
  let cost_per_mile_first_3 : ℝ := 0.30
  let cost_per_mile_additional : ℝ := 0.40
  let total_distance : ℕ := 8
  let first_3_miles_cost : ℝ := base_fare + 3 * cost_per_mile_first_3
  let additional_miles_cost : ℝ := (total_distance - 3) * cost_per_mile_additional
  let total_cost : ℝ := first_3_miles_cost + additional_miles_cost
  total_cost = 4.90 :=
by
  sorry

end taxi_ride_cost_l2_2822


namespace hexagon_points_fourth_layer_l2_2914

theorem hexagon_points_fourth_layer :
  ∃ (h : ℕ → ℕ), h 1 = 1 ∧ (∀ n ≥ 2, h n = h (n - 1) + 6 * (n - 1)) ∧ h 4 = 37 :=
by
  sorry

end hexagon_points_fourth_layer_l2_2914


namespace cosine_identity_l2_2099

theorem cosine_identity (α : ℝ) (h : Real.sin (Real.pi / 6 - α) = 1 / 4) : 
  Real.cos (2 * α - Real.pi / 3) = 7 / 8 := by 
  sorry

end cosine_identity_l2_2099


namespace systematic_sampling_fourth_group_l2_2643

theorem systematic_sampling_fourth_group (n m k g2 g4 : ℕ) (h_class_size : n = 72)
  (h_sample_size : m = 6) (h_k : k = n / m) (h_group2 : g2 = 16) (h_group4 : g4 = g2 + 2 * k) :
  g4 = 40 := by
  sorry

end systematic_sampling_fourth_group_l2_2643


namespace red_balls_in_bag_l2_2924

theorem red_balls_in_bag (total_balls : ℕ) (white_balls : ℕ) (green_balls : ℕ) (yellow_balls : ℕ) (purple_balls : ℕ) (prob_neither_red_nor_purple : ℝ) :
  total_balls = 60 → 
  white_balls = 22 → 
  green_balls = 18 → 
  yellow_balls = 8 → 
  purple_balls = 7 → 
  prob_neither_red_nor_purple = 0.8 → 
  ( ∃ (red_balls : ℕ), red_balls = 5 ) :=
by
  intros h₁ h₂ h₃ h₄ h₅ h₆
  sorry

end red_balls_in_bag_l2_2924


namespace find_a6_of_arithmetic_seq_l2_2298

noncomputable def arithmetic_sequence (n : ℕ) (a1 d : ℝ) : ℝ :=
  a1 + (n - 1) * d

noncomputable def sum_of_arithmetic_sequence (n : ℕ) (a1 d : ℝ) : ℝ :=
  n / 2 * (2 * a1 + (n - 1) * d)

theorem find_a6_of_arithmetic_seq 
  (a1 d : ℝ) 
  (S3 : ℝ) 
  (h_a1 : a1 = 2) 
  (h_S3 : S3 = 12) 
  (h_sum : S3 = sum_of_arithmetic_sequence 3 a1 d) :
  arithmetic_sequence 6 a1 d = 12 := 
sorry

end find_a6_of_arithmetic_seq_l2_2298


namespace find_a1_l2_2217

variable {a : ℕ → ℝ}
variable {q : ℝ}

-- The sequence {a_n} is a geometric sequence with a common ratio q > 0
axiom geom_seq : (∀ n, a (n + 1) = a n * q)

-- Given conditions of the problem
def condition1 : q > 0 := sorry
def condition2 : a 5 * a 7 = 4 * (a 4) ^ 2 := sorry
def condition3 : a 2 = 1 := sorry

-- Prove that a_1 = sqrt 2 / 2
theorem find_a1 : a 1 = (Real.sqrt 2) / 2 := sorry

end find_a1_l2_2217


namespace eval_f_at_neg_twenty_three_sixth_pi_l2_2596

noncomputable def f (α : ℝ) : ℝ := 
    (2 * (Real.sin (2 * Real.pi - α)) * (Real.cos (2 * Real.pi + α)) - Real.cos (-α)) / 
    (1 + Real.sin α ^ 2 + Real.sin (2 * Real.pi + α) - Real.cos (4 * Real.pi - α) ^ 2)

theorem eval_f_at_neg_twenty_three_sixth_pi : 
  f (-23 / 6 * Real.pi) = -Real.sqrt 3 :=
  sorry

end eval_f_at_neg_twenty_three_sixth_pi_l2_2596


namespace greatest_common_factor_36_45_l2_2993

theorem greatest_common_factor_36_45 : 
  ∃ g, g = (gcd 36 45) ∧ g = 9 :=
by {
  sorry
}

end greatest_common_factor_36_45_l2_2993


namespace no_int_solutions_except_zero_l2_2161

theorem no_int_solutions_except_zero 
  (a b c n : ℤ) :
  6 * (6 * a^2 + 3 * b^2 + c^2) = 5 * n^2 → 
  a = 0 ∧ b = 0 ∧ c = 0 ∧ n = 0 :=
by
  sorry

end no_int_solutions_except_zero_l2_2161


namespace parabola_coefficients_l2_2106

theorem parabola_coefficients 
  (a b c : ℝ) 
  (h_vertex : ∀ x : ℝ, (2 - (-2))^2 * a + (-2 * 2 * a + b) * (2 - (-2)) + (c - 5) = 0)
  (h_point : 9 = a * (2:ℝ)^2 + b * (2:ℝ) + c) : 
  a = 1 / 4 ∧ b = 1 ∧ c = 6 := 
by 
  sorry

end parabola_coefficients_l2_2106


namespace expression_simplified_l2_2326

noncomputable def expression : ℚ := 1 + 3 / (4 + 5 / 6)

theorem expression_simplified : expression = 47 / 29 :=
by
  sorry

end expression_simplified_l2_2326


namespace abs_gt_1_not_sufficient_nor_necessary_l2_2201

theorem abs_gt_1_not_sufficient_nor_necessary (a : ℝ) :
  ¬((|a| > 1) → (a > 0)) ∧ ¬((a > 0) → (|a| > 1)) :=
by
  sorry

end abs_gt_1_not_sufficient_nor_necessary_l2_2201


namespace exponentiation_81_5_4_eq_243_l2_2035

theorem exponentiation_81_5_4_eq_243 : 81^(5/4) = 243 := by
  sorry

end exponentiation_81_5_4_eq_243_l2_2035


namespace remainder_when_divided_by_23_l2_2983

theorem remainder_when_divided_by_23 (y : ℕ) (h : y % 276 = 42) : y % 23 = 19 := by
  sorry

end remainder_when_divided_by_23_l2_2983


namespace largest_divisor_of_expression_l2_2440

theorem largest_divisor_of_expression :
  ∃ x : ℕ, (∀ y : ℕ, x ∣ (7^y + 12*y - 1)) ∧ (∀ z : ℕ, (∀ y : ℕ, z ∣ (7^y + 12*y - 1)) → z ≤ x) :=
sorry

end largest_divisor_of_expression_l2_2440


namespace sqrt_37_between_6_and_7_l2_2981

theorem sqrt_37_between_6_and_7 : 6 < Real.sqrt 37 ∧ Real.sqrt 37 < 7 := 
by 
  have h₁ : Real.sqrt 36 = 6 := by sorry
  have h₂ : Real.sqrt 49 = 7 := by sorry
  sorry

end sqrt_37_between_6_and_7_l2_2981


namespace jill_draws_spade_probability_l2_2260

noncomputable def probability_jill_draws_spade : ℚ :=
  ∑' (k : ℕ), ((3 / 4) * (3 / 4))^k * ((3 / 4) * (1 / 4))

theorem jill_draws_spade_probability : probability_jill_draws_spade = 3 / 7 :=
sorry

end jill_draws_spade_probability_l2_2260


namespace simplify_expr1_simplify_expr2_l2_2509

-- (1) Simplify the expression: 3a(a+1) - (3+a)(3-a) - (2a-1)^2 == 7a - 10
theorem simplify_expr1 (a : ℝ) : 
  3 * a * (a + 1) - (3 + a) * (3 - a) - (2 * a - 1) ^ 2 = 7 * a - 10 :=
sorry

-- (2) Simplify the expression: ((x^2 - 2x + 4) / (x - 1) + 2 - x) / (x^2 + 4x + 4) / (1 - x) == -2 / (x + 2)^2
theorem simplify_expr2 (x : ℝ) (h : x ≠ 1) (h1 : x ≠ 0) : 
  (((x^2 - 2 * x + 4) / (x - 1) + 2 - x) / ((x^2 + 4 * x + 4) / (1 - x))) = -2 / (x + 2)^2 :=
sorry

end simplify_expr1_simplify_expr2_l2_2509


namespace totalFourOfAKindCombinations_l2_2329

noncomputable def numberOfFourOfAKindCombinations : Nat :=
  13 * 48

theorem totalFourOfAKindCombinations : numberOfFourOfAKindCombinations = 624 := by
  sorry

end totalFourOfAKindCombinations_l2_2329


namespace points_with_tangent_length_six_l2_2273

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 2 * x - 8 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4 * x - 6 * y + 4 = 0

-- Define the property of a point having a tangent of length 6 to the circle
def tangent_length_six (h k cx cy r : ℝ) : Prop :=
  (cx - h)^2 + (cy - k)^2 - r^2 = 36

-- Main theorem statement
theorem points_with_tangent_length_six : 
  (∀ x1 y1 : ℝ, (x1 = -4 ∧ y1 = 6) ∨ (x1 = 5 ∧ y1 = -3) → 
    (∃ r1 : ℝ, tangent_length_six x1 y1 (-1) 0 3) ∧ 
    (∃ r2 : ℝ, tangent_length_six x1 y1 2 3 3)) :=
  by 
  sorry

end points_with_tangent_length_six_l2_2273


namespace farmer_land_l2_2199

noncomputable def farmer_land_example (A : ℝ) : Prop :=
  let cleared_land := 0.90 * A
  let barley_land := 0.70 * cleared_land
  let potatoes_land := 0.10 * cleared_land
  let corn_land := 0.10 * cleared_land
  let tomatoes_bell_peppers_land := 0.10 * cleared_land
  tomatoes_bell_peppers_land = 90 → A = 1000

theorem farmer_land (A : ℝ) (h_cleared_land : 0.90 * A = cleared_land)
  (h_barley_land : 0.70 * cleared_land = barley_land)
  (h_potatoes_land : 0.10 * cleared_land = potatoes_land)
  (h_corn_land : 0.10 * cleared_land = corn_land)
  (h_tomatoes_bell_peppers_land : 0.10 * cleared_land = 90) :
  A = 1000 :=
by
  sorry

end farmer_land_l2_2199


namespace value_of_f_37_5_l2_2779

-- Mathematical definitions and conditions
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)
def periodic_function (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f (x)
def satisfies_condition (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = -f (x)
def interval_condition (f : ℝ → ℝ) : Prop := ∀ x, 0 ≤ x ∧ x ≤ 1 → f (x) = x

-- Main theorem to be proved
theorem value_of_f_37_5 (f : ℝ → ℝ) 
  (h_odd : odd_function f) 
  (h_periodic : satisfies_condition f) 
  (h_interval : interval_condition f) : 
  f 37.5 = 0.5 := 
sorry

end value_of_f_37_5_l2_2779


namespace geo_seq_sum_monotone_l2_2376

theorem geo_seq_sum_monotone (q a1 : ℝ) (n : ℕ) (S : ℕ → ℝ) :
  (∀ n, S (n + 1) > S n) ↔ (a1 > 0 ∧ q > 0) :=
sorry -- Proof of the theorem (omitted)

end geo_seq_sum_monotone_l2_2376


namespace integer_exponentiation_l2_2847

theorem integer_exponentiation
  (a b x y : ℕ)
  (h_gcd : a.gcd b = 1)
  (h_pos_a : 1 < a)
  (h_pos_b : 1 < b)
  (h_pos_x : 1 < x)
  (h_pos_y : 1 < y)
  (h_eq : x^a = y^b) :
  ∃ n : ℕ, 1 < n ∧ x = n^b ∧ y = n^a :=
by sorry

end integer_exponentiation_l2_2847


namespace joan_change_received_l2_2711

theorem joan_change_received :
  let cat_toy_cost := 8.77
  let cage_cost := 10.97
  let payment := 20.00
  let total_cost := cat_toy_cost + cage_cost
  let change_received := payment - total_cost
  change_received = 0.26 :=
by
  sorry

end joan_change_received_l2_2711


namespace exists_polynomial_p_l2_2815

theorem exists_polynomial_p (x : ℝ) (h : x ∈ Set.Icc (1 / 10 : ℝ) (9 / 10 : ℝ)) :
  ∃ (P : ℝ → ℝ), (∀ (k : ℤ), P k = P k) ∧ (∀ (x : ℝ), x ∈ Set.Icc (1 / 10 : ℝ) (9 / 10 : ℝ) → 
  abs (P x - 1 / 2) < 1 / 1000) :=
by
  sorry

end exists_polynomial_p_l2_2815


namespace negation_of_p_is_neg_p_l2_2213

-- Define the proposition p
def p : Prop :=
  ∀ x > 0, (x + 1) * Real.exp x > 1

-- Define the negation of the proposition p
def neg_p : Prop :=
  ∃ x > 0, (x + 1) * Real.exp x ≤ 1

-- State the proof problem: negation of p is neg_p
theorem negation_of_p_is_neg_p : ¬p ↔ neg_p :=
by
  -- Stating that ¬p is equivalent to neg_p
  sorry

end negation_of_p_is_neg_p_l2_2213


namespace probability_exactly_two_singers_same_province_l2_2128

-- Defining the number of provinces and number of singers per province
def num_provinces : ℕ := 6
def singers_per_province : ℕ := 2

-- Total number of singers
def num_singers : ℕ := num_provinces * singers_per_province

-- Define the total number of ways to choose 4 winners from 12 contestants
def total_combinations : ℕ := Nat.choose num_singers 4

-- Define the number of favorable ways to select exactly two singers from the same province and two from two other provinces
def favorable_combinations : ℕ := 
  (Nat.choose num_provinces 1) *  -- Choose one province for the pair
  (Nat.choose (num_provinces - 1) 2) *  -- Choose two remaining provinces
  (Nat.choose singers_per_province 1) *
  (Nat.choose singers_per_province 1)

-- Calculate the probability
def probability : ℚ := favorable_combinations / total_combinations

-- Stating the theorem to be proved
theorem probability_exactly_two_singers_same_province : probability = 16 / 33 :=
by
  sorry

end probability_exactly_two_singers_same_province_l2_2128


namespace sum_geometric_terms_l2_2055

noncomputable def geometric_sequence (a q : ℝ) (n : ℕ) : ℝ := a * q^n

theorem sum_geometric_terms (a q : ℝ) :
  a * (1 + q) = 3 → a * (1 + q) * q^2 = 6 → 
  a * (1 + q) * q^6 = 24 :=
by
  intros h1 h2
  -- Proof would go here
  sorry

end sum_geometric_terms_l2_2055


namespace intersection_M_N_l2_2447

def M : Set ℝ := { x | -2 ≤ x ∧ x < 2 }
def N : Set ℝ := { x | x ≥ -2 }

theorem intersection_M_N : M ∩ N = { x | -2 ≤ x ∧ x < 2 } := by
  sorry

end intersection_M_N_l2_2447


namespace parallel_vectors_l2_2229

theorem parallel_vectors (m : ℝ) :
  let a : (ℝ × ℝ × ℝ) := (2, -1, 2)
  let b : (ℝ × ℝ × ℝ) := (-4, 2, m)
  (∀ k : ℝ, a = (k * -4, k * 2, k * m)) →
  m = -4 :=
by
  sorry

end parallel_vectors_l2_2229


namespace valid_ATM_passwords_l2_2876

theorem valid_ATM_passwords : 
  let total_passwords := 10^4
  let restricted_passwords := 10
  total_passwords - restricted_passwords = 9990 :=
by
  sorry

end valid_ATM_passwords_l2_2876


namespace minimum_for_specific_values_proof_minimum_for_arbitrary_values_proof_l2_2755

noncomputable def minimum_for_specific_values : ℝ :=
  let m := 2 
  let n := 2 
  let p := 2 
  let xyz := 8 
  let x := 2
  let y := 2
  let z := 2
  x^2 + y^2 + z^2 + m * x * y + n * x * z + p * y * z

theorem minimum_for_specific_values_proof : minimum_for_specific_values = 36 := by
  sorry

noncomputable def minimum_for_arbitrary_values (m n p : ℝ) (h : m * n * p = 8) : ℝ :=
  let x := 2
  let y := 2
  let z := 2
  x^2 + y^2 + z^2 + m * x * y + n * x * z + p * y * z

theorem minimum_for_arbitrary_values_proof (m n p : ℝ) (h : m * n * p = 8) : minimum_for_arbitrary_values m n p h = 12 + 4 * (m + n + p) := by
  sorry

end minimum_for_specific_values_proof_minimum_for_arbitrary_values_proof_l2_2755


namespace general_term_arithmetic_sequence_l2_2054

-- Define an arithmetic sequence with first term a1 and common ratio q
def arithmetic_sequence (a1 : ℤ) (q : ℤ) (n : ℕ) : ℤ :=
  a1 * q ^ (n - 1)

-- Theorem: given the conditions, prove that the general term is a1 * q^(n-1)
theorem general_term_arithmetic_sequence (a1 q : ℤ) (n : ℕ) :
  arithmetic_sequence a1 q n = a1 * q ^ (n - 1) :=
by
  sorry

end general_term_arithmetic_sequence_l2_2054


namespace identity_equality_l2_2681

theorem identity_equality (a b m n x y : ℝ) :
  ((a^2 + b^2) * (m^2 + n^2) * (x^2 + y^2)) =
  ((a * n * y - a * m * x - b * m * y + b * n * x)^2 + (a * m * y + a * n * x + b * m * x - b * n * y)^2) :=
by
  sorry

end identity_equality_l2_2681


namespace rectangle_properties_l2_2435

theorem rectangle_properties (w l : ℝ) (h₁ : l = 4 * w) (h₂ : 2 * l + 2 * w = 200) :
  ∃ A d, A = 1600 ∧ d = 82.46 := 
by {
  sorry
}

end rectangle_properties_l2_2435


namespace final_price_relative_l2_2402

-- Definitions of the conditions
variable (x : ℝ)
#check x * 1.30  -- original price increased by 30%
#check x * 1.30 * 0.85  -- after 15% discount on increased price
#check x * 1.30 * 0.85 * 1.05  -- after applying 5% tax on discounted price

-- Theorem to prove the final price relative to the original price
theorem final_price_relative (x : ℝ) : 
  (x * 1.30 * 0.85 * 1.05) = (1.16025 * x) :=
by
  sorry

end final_price_relative_l2_2402


namespace scatter_plot_exists_l2_2380

theorem scatter_plot_exists (sample_data : List (ℝ × ℝ)) :
  ∃ plot : List (ℝ × ℝ), plot = sample_data :=
by
  sorry

end scatter_plot_exists_l2_2380


namespace unique_solution_to_equation_l2_2676

theorem unique_solution_to_equation (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 2 * x^y - y = 2005) : x = 1003 ∧ y = 1 :=
by
  sorry

end unique_solution_to_equation_l2_2676


namespace real_y_iff_x_interval_l2_2366

theorem real_y_iff_x_interval (x : ℝ) :
  (∃ y : ℝ, 3*y^2 + 2*x*y + x + 5 = 0) ↔ (x ≤ -3 ∨ x ≥ 5) :=
by
  sorry

end real_y_iff_x_interval_l2_2366


namespace xyz_sum_divisible_l2_2953

-- Define variables and conditions
variable (p x y z : ℕ) [Fact (Prime p)]
variable (h1 : 0 < x) (h2 : x < y) (h3 : y < z) (h4 : z < p)
variable (h_eq1 : x^3 % p = y^3 % p)
variable (h_eq2 : y^3 % p = z^3 % p)

-- Theorem statement
theorem xyz_sum_divisible (p x y z : ℕ) [Fact (Prime p)]
  (h1 : 0 < x) (h2 : x < y) (h3 : y < z) (h4 : z < p)
  (h_eq1 : x^3 % p = y^3 % p)
  (h_eq2 : y^3 % p = z^3 % p) :
  (x^2 + y^2 + z^2) % (x + y + z) = 0 := 
  sorry

end xyz_sum_divisible_l2_2953


namespace base_b_cube_l2_2022

theorem base_b_cube (b : ℕ) : (b > 4) → (∃ n : ℕ, (b^2 + 4 * b + 4 = n^3)) ↔ (b = 5 ∨ b = 6) :=
by
  sorry

end base_b_cube_l2_2022


namespace periodic_modulo_h_l2_2032

open Nat

-- Defining the binomial coefficient
def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Defining the sequence as per the problem
def x_seq (n : ℕ) : ℕ :=
  binom (2 * n) n

-- The main theorem stating the required condition
theorem periodic_modulo_h (h : ℕ) (h_gt_one : h > 1) :
  (∃ N, ∀ n ≥ N, x_seq n % h = x_seq (n + 1) % h) ↔ h = 2 :=
by
  sorry

end periodic_modulo_h_l2_2032


namespace least_m_value_l2_2171

def recursive_sequence (x : ℕ → ℚ) : Prop :=
  x 0 = 3 ∧ ∀ n, x (n + 1) = (x n ^ 2 + 9 * x n + 20) / (x n + 8)

theorem least_m_value (x : ℕ → ℚ) (h : recursive_sequence x) : ∃ m, m > 0 ∧ x m ≤ 3 + 1 / 2^10 ∧ ∀ k, k > 0 → k < m → x k > 3 + 1 / 2^10 :=
sorry

end least_m_value_l2_2171


namespace students_not_enrolled_in_either_course_l2_2896

theorem students_not_enrolled_in_either_course 
  (total : ℕ) (french : ℕ) (german : ℕ) (both : ℕ) 
  (h_total : total = 87) (h_french : french = 41) (h_german : german = 22) (h_both : both = 9) : 
  ∃ (not_enrolled : ℕ), not_enrolled = (total - (french + german - both)) ∧ not_enrolled = 33 := by
  have h_french_or_german : ℕ := french + german - both
  have h_not_enrolled : ℕ := total - h_french_or_german
  use h_not_enrolled
  sorry

end students_not_enrolled_in_either_course_l2_2896


namespace sum_same_probability_l2_2555

-- Definition for standard dice probability problem
def dice_problem (n : ℕ) (target_sum : ℕ) (target_sum_of_faces : ℕ) : Prop :=
  let faces := [1, 2, 3, 4, 5, 6]
  let min_sum := n * 1
  let max_sum := n * 6
  let average_sum := (min_sum + max_sum) / 2
  let symmetric_sum := 2 * average_sum - target_sum
  symmetric_sum = target_sum_of_faces

-- The proof statement (no proof included, just the declaration)
theorem sum_same_probability : dice_problem 8 12 44 :=
by sorry

end sum_same_probability_l2_2555


namespace telephone_number_A_value_l2_2696

theorem telephone_number_A_value :
  ∃ A B C D E F G H I J : ℕ,
    A > B ∧ B > C ∧
    D > E ∧ E > F ∧
    G > H ∧ H > I ∧ I > J ∧
    (D = E + 1) ∧ (E = F + 1) ∧
    G + H + I + J = 20 ∧
    A + B + C = 15 ∧
    A = 8 := sorry

end telephone_number_A_value_l2_2696


namespace captain_smollett_problem_l2_2526

/-- 
Given the captain's age, the number of children he has, and the length of his schooner, 
prove that the unique solution to the product condition is age = 53 years, children = 6, 
and length = 101 feet, under the given constraints.
-/
theorem captain_smollett_problem
  (age children length : ℕ)
  (h1 : age < 100)
  (h2 : children > 3)
  (h3 : age * children * length = 32118) : age = 53 ∧ children = 6 ∧ length = 101 :=
by {
  -- Proof will be filled in later
  sorry
}

end captain_smollett_problem_l2_2526


namespace man_l2_2163

-- Define all given conditions using Lean definitions
def speed_with_current_wind : ℝ := 22
def speed_of_current : ℝ := 5
def wind_resistance_factor : ℝ := 0.15
def current_increase_factor : ℝ := 0.10

-- Define the key quantities (man's speed in still water, effective speed in still water, new current speed against)
def speed_in_still_water : ℝ := speed_with_current_wind - speed_of_current
def effective_speed_in_still_water : ℝ := speed_in_still_water - (wind_resistance_factor * speed_in_still_water)
def new_speed_of_current_against : ℝ := speed_of_current + (current_increase_factor * speed_of_current)

-- Proof goal: Prove that the man's speed against the current is 8.95 km/hr considering all the conditions
theorem man's_speed_against_current_is_correct : 
  (effective_speed_in_still_water - new_speed_of_current_against) = 8.95 := 
by
  sorry

end man_l2_2163


namespace intersection_point_of_curve_and_line_l2_2418

theorem intersection_point_of_curve_and_line : 
  ∃ (e : ℝ), (0 < e) ∧ (e = Real.exp 1) ∧ ((e, e) ∈ { p : ℝ × ℝ | ∃ (x y : ℝ), x ^ y = y ^ x ∧ 0 ≤ x ∧ 0 ≤ y}) :=
by {
  sorry
}

end intersection_point_of_curve_and_line_l2_2418


namespace mary_hourly_wage_l2_2029

-- Defining the conditions as given in the problem
def hours_per_day_MWF : ℕ := 9
def hours_per_day_TTh : ℕ := 5
def days_MWF : ℕ := 3
def days_TTh : ℕ := 2
def weekly_earnings : ℕ := 407

-- Total hours worked in a week by Mary
def total_hours_worked : ℕ := (days_MWF * hours_per_day_MWF) + (days_TTh * hours_per_day_TTh)

-- The hourly wage calculation
def hourly_wage : ℕ := weekly_earnings / total_hours_worked

-- The statement to prove
theorem mary_hourly_wage : hourly_wage = 11 := by
  sorry

end mary_hourly_wage_l2_2029


namespace kids_stay_home_lawrence_county_l2_2978

def total_kids_lawrence_county : ℕ := 1201565
def kids_camp_lawrence_county : ℕ := 610769

theorem kids_stay_home_lawrence_county : total_kids_lawrence_county - kids_camp_lawrence_county = 590796 := by
  sorry

end kids_stay_home_lawrence_county_l2_2978


namespace find_original_expenditure_l2_2569

def original_expenditure (x : ℝ) := 35 * x
def new_expenditure (x : ℝ) := 42 * (x - 1)

theorem find_original_expenditure :
  ∃ x, 35 * x + 42 = 42 * (x - 1) ∧ original_expenditure x = 420 :=
by
  sorry

end find_original_expenditure_l2_2569


namespace evaluateExpression_correct_l2_2103

open Real

noncomputable def evaluateExpression : ℝ :=
  (-2)^2 + 2 * sin (π / 3) - tan (π / 3)

theorem evaluateExpression_correct : evaluateExpression = 4 :=
  sorry

end evaluateExpression_correct_l2_2103


namespace inequality_satisfaction_l2_2072

theorem inequality_satisfaction (a b : ℝ) (h : a < 0) : (a < b) ∧ (a^2 + b^2 > 2) :=
by
  sorry

end inequality_satisfaction_l2_2072


namespace ninth_term_arithmetic_sequence_l2_2414

theorem ninth_term_arithmetic_sequence :
  ∃ (a d : ℤ), (a + 2 * d = 5 ∧ a + 5 * d = 17) ∧ (a + 8 * d = 29) := 
by
  sorry

end ninth_term_arithmetic_sequence_l2_2414


namespace mean_weight_players_l2_2713

/-- Definitions for the weights of the players and proving the mean weight. -/
def weights : List ℕ := [62, 65, 70, 73, 73, 76, 78, 79, 81, 81, 82, 84, 87, 89, 89, 89, 90, 93, 95]

def mean (lst : List ℕ) : ℚ := (lst.sum : ℚ) / lst.length

theorem mean_weight_players : mean weights = 80.84 := by
  sorry

end mean_weight_players_l2_2713


namespace four_digit_numbers_divisible_by_5_l2_2176

theorem four_digit_numbers_divisible_by_5 : 
  let smallest_4_digit := 1000
  let largest_4_digit := 9999
  let divisible_by_5 (n : ℕ) := ∃ k : ℕ, n = 5 * k
  ∃ n : ℕ, ( ∀ x : ℕ, smallest_4_digit ≤ x ∧ x ≤ largest_4_digit ∧ divisible_by_5 x ↔ (smallest_4_digit + (n-1) * 5 = x) ) ∧ n = 1800 :=
by
  sorry

end four_digit_numbers_divisible_by_5_l2_2176


namespace fifth_month_sale_correct_l2_2042

noncomputable def fifth_month_sale
  (sales : Fin 4 → ℕ)
  (sixth_month_sale : ℕ)
  (average_sale : ℕ) : ℕ :=
  let total_sales := average_sale * 6
  let known_sales := sales 0 + sales 1 + sales 2 + sales 3 + sixth_month_sale
  total_sales - known_sales

theorem fifth_month_sale_correct :
  ∀ (sales : Fin 4 → ℕ) (sixth_month_sale : ℕ) (average_sale : ℕ),
    sales 0 = 6435 →
    sales 1 = 6927 →
    sales 2 = 6855 →
    sales 3 = 7230 →
    sixth_month_sale = 5591 →
    average_sale = 6600 →
    fifth_month_sale sales sixth_month_sale average_sale = 13562 :=
by
  intros sales sixth_month_sale average_sale h0 h1 h2 h3 h4 h5
  unfold fifth_month_sale
  sorry

end fifth_month_sale_correct_l2_2042


namespace problem_l2_2216

theorem problem
  (x y : ℝ)
  (h1 : x + 3 * y = 9)
  (h2 : x * y = -27) :
  x^2 + 9 * y^2 = 243 :=
sorry

end problem_l2_2216


namespace determine_continuous_function_l2_2892

open Real

theorem determine_continuous_function (f : ℝ → ℝ) 
  (h_continuous : Continuous f)
  (h_initial : f 0 = 1)
  (h_inequality : ∀ x y : ℝ, f (x + y) ≥ f x * f y) : 
  ∃ k : ℝ, ∀ x : ℝ, f x = exp (k * x) :=
sorry

end determine_continuous_function_l2_2892


namespace conversion_base_10_to_5_l2_2392

theorem conversion_base_10_to_5 : 
  (425 : ℕ) = 3 * 5^3 + 2 * 5^2 + 0 * 5^1 + 0 * 5^0 :=
by sorry

end conversion_base_10_to_5_l2_2392


namespace min_sum_a_b_l2_2733

theorem min_sum_a_b (a b : ℝ) (h_cond: 1/a + 4/b = 1) (a_pos : 0 < a) (b_pos : 0 < b) : 
  a + b ≥ 9 :=
sorry

end min_sum_a_b_l2_2733


namespace extremum_range_l2_2455

noncomputable def f (a x : ℝ) : ℝ := x^3 + 2 * x^2 - a * x + 1

noncomputable def f_prime (a x : ℝ) : ℝ := 3 * x^2 + 4 * x - a

theorem extremum_range 
  (h : ∀ a : ℝ, (∃ (x : ℝ) (hx : -1 < x ∧ x < 1), f_prime a x = 0) → 
                (∀ x : ℝ, -1 < x ∧ x < 1 → f_prime a x ≠ 0)):
  ∀ a : ℝ, -1 < a ∧ a < 7 :=
sorry

end extremum_range_l2_2455


namespace non_obtuse_triangle_range_l2_2726

noncomputable def range_of_2a_over_c (a b c A C : ℝ) (h1 : B = π / 3) (h2 : A + C = 2 * π / 3) (h3 : π / 6 < C ∧ C ≤ π / 2) : Set ℝ :=
  {x | ∃ (a b c A : ℝ), x = (2 * a) / c ∧ 1 < x ∧ x ≤ 4}

theorem non_obtuse_triangle_range (a b c A C : ℝ) (h1 : B = π / 3) (h2 : A + C = 2 * π / 3) (h3 : π / 6 < C ∧ C ≤ π / 2) :
  (2 * a) / c ∈ range_of_2a_over_c a b c A C h1 h2 h3 := 
sorry

end non_obtuse_triangle_range_l2_2726


namespace max_median_cans_per_customer_l2_2584

theorem max_median_cans_per_customer : 
    ∀ (total_cans : ℕ) (total_customers : ℕ), 
    total_cans = 252 → total_customers = 100 →
    (∀ (cans_per_customer : ℕ),
    1 ≤ cans_per_customer) →
    (∃ (max_median : ℝ),
    max_median = 3.5) :=
by
  sorry

end max_median_cans_per_customer_l2_2584


namespace quotient_remainder_div_by_18_l2_2499

theorem quotient_remainder_div_by_18 (M q : ℕ) (h : M = 54 * q + 37) : 
  ∃ k r, M = 18 * k + r ∧ r < 18 ∧ k = 3 * q + 2 ∧ r = 1 :=
by sorry

end quotient_remainder_div_by_18_l2_2499


namespace h_two_n_mul_h_2024_l2_2068

variable {h : ℕ → ℝ}
variable {k : ℝ}
variable (n : ℕ) (k_ne_zero : k ≠ 0)

-- Condition 1: h(m + n) = h(m) * h(n)
axiom h_add_mul (m n : ℕ) : h (m + n) = h m * h n

-- Condition 2: h(2) = k
axiom h_two : h 2 = k

theorem h_two_n_mul_h_2024 : h (2 * n) * h 2024 = k^(n + 1012) := 
  sorry

end h_two_n_mul_h_2024_l2_2068


namespace max_incircle_circumcircle_ratio_l2_2694

theorem max_incircle_circumcircle_ratio (c : ℝ) (α : ℝ) 
  (hα : 0 < α ∧ α < π / 2) :
  let a := c * Real.cos α
  let b := c * Real.sin α
  let R := c / 2
  let r := (a + b - c) / 2
  (r / R <= Real.sqrt 2 - 1) :=
by
  sorry

end max_incircle_circumcircle_ratio_l2_2694


namespace equation_of_chord_l2_2601

open Real

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 6 * x = 0

def is_midpoint_of_chord (P M N : ℝ × ℝ) : Prop :=
  ∃ (C : ℝ × ℝ), circle_eq (C.1) (C.2) ∧ (P.1, P.2) = ((M.1 + N.1) / 2, (M.2 + N.2) / 2)

theorem equation_of_chord (P : ℝ × ℝ) (M N : ℝ × ℝ) (h : P = (4, 2)) (h_mid : is_midpoint_of_chord P M N) :
  ∀ (x y : ℝ), (2 * y) - (8 : ℝ) = (-(1 / 2) * (x - 4)) →
  x + 2 * y - 8 = 0 :=
by
  intro x y H
  sorry

end equation_of_chord_l2_2601


namespace number_value_l2_2976

theorem number_value (N : ℝ) (h : 0.40 * N = 180) : 
  (1/4) * (1/3) * (2/5) * N = 15 :=
by
  -- assume the conditions have been stated correctly
  sorry

end number_value_l2_2976


namespace max_bench_weight_support_l2_2985

/-- Definitions for the given problem conditions -/
def john_weight : ℝ := 250
def bar_weight : ℝ := 550
def total_weight : ℝ := john_weight + bar_weight
def safety_percentage : ℝ := 0.80

/-- Theorem stating the maximum weight the bench can support given the conditions -/
theorem max_bench_weight_support :
  ∀ (W : ℝ), safety_percentage * W = total_weight → W = 1000 :=
by
  sorry

end max_bench_weight_support_l2_2985


namespace students_with_all_three_pets_l2_2338

theorem students_with_all_three_pets :
  ∀ (total_students : ℕ)
    (dog_fraction cat_fraction : ℚ)
    (other_pets students_no_pets dogs_only cats_only other_pets_only x y z w : ℕ),
    total_students = 40 →
    dog_fraction = 5 / 8 →
    cat_fraction = 1 / 4 →
    other_pets = 8 →
    students_no_pets = 4 →
    dogs_only = 15 →
    cats_only = 3 →
    other_pets_only = 2 →
    dogs_only + x + z + w = total_students * dog_fraction →
    cats_only + x + y + w = total_students * cat_fraction →
    other_pets_only + y + z + w = other_pets →
    dogs_only + cats_only + other_pets_only + x + y + z + w = total_students - students_no_pets →
    w = 4  := 
by
  sorry

end students_with_all_three_pets_l2_2338


namespace syllogism_error_l2_2948

-- Definitions based on conditions from a)
def major_premise (a: ℝ) : Prop := a^2 > 0

def minor_premise (a: ℝ) : Prop := true

-- Theorem stating that the conclusion does not necessarily follow
theorem syllogism_error (a : ℝ) (h_minor : minor_premise a) : ¬major_premise 0 :=
by
  sorry

end syllogism_error_l2_2948


namespace johns_groceries_cost_l2_2007

noncomputable def calculate_total_cost : ℝ := 
  let bananas_cost := 6 * 2
  let bread_cost := 2 * 3
  let butter_cost := 3 * 5
  let cereal_cost := 4 * (6 - 0.25 * 6)
  let subtotal := bananas_cost + bread_cost + butter_cost + cereal_cost
  if subtotal >= 50 then
    subtotal - 10
  else
    subtotal

-- The statement to prove
theorem johns_groceries_cost : calculate_total_cost = 41 := by
  sorry

end johns_groceries_cost_l2_2007


namespace find_coordinates_of_C_l2_2650

structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def A : Point := { x := 4, y := -1, z := 2 }
def B : Point := { x := 2, y := -3, z := 0 }

def satisfies_condition (C : Point) : Prop :=
  (C.x - B.x, C.y - B.y, C.z - B.z) = (2 * (A.x - C.x), 2 * (A.y - C.y), 2 * (A.z - C.z))

theorem find_coordinates_of_C (C : Point) (h : satisfies_condition C) : C = { x := 10/3, y := -5/3, z := 4/3 } :=
  sorry -- Proof is omitted as requested

end find_coordinates_of_C_l2_2650


namespace monotonic_intervals_and_extreme_points_l2_2632

noncomputable def f (x a : ℝ) : ℝ := (1 / 2) * x^2 - (a + 1) * x + a * Real.log x

theorem monotonic_intervals_and_extreme_points (a : ℝ) (h : 1 < a) :
  ∃ x1 x2, x1 = 1 ∧ x2 = a ∧ x1 < x2 ∧ f x2 a < - (3 / 2) * x1 :=
by
  sorry

end monotonic_intervals_and_extreme_points_l2_2632


namespace hundred_squared_plus_two_hundred_one_is_composite_l2_2940

theorem hundred_squared_plus_two_hundred_one_is_composite : 
    ¬ Prime (100^2 + 201) :=
by {
  sorry
}

end hundred_squared_plus_two_hundred_one_is_composite_l2_2940


namespace original_profit_margin_theorem_l2_2781

noncomputable def original_profit_margin (a : ℝ) (x : ℝ) (h : a > 0) : Prop := 
  (a * (1 + x) - a * (1 - 0.064)) / (a * (1 - 0.064)) = x + 0.08

theorem original_profit_margin_theorem (a : ℝ) (x : ℝ) (h : a > 0) :
  original_profit_margin a x h → x = 0.17 :=
sorry

end original_profit_margin_theorem_l2_2781


namespace gabrielle_peaches_l2_2609

theorem gabrielle_peaches (B G : ℕ) 
  (h1 : 16 = 2 * B + 6)
  (h2 : B = G / 3) :
  G = 15 :=
by
  sorry

end gabrielle_peaches_l2_2609


namespace cost_of_apples_and_oranges_correct_l2_2321

-- Define the initial money jasmine had
def initial_money : ℝ := 100.00

-- Define the remaining money after purchase
def remaining_money : ℝ := 85.00

-- Define the cost of apples and oranges
def cost_of_apples_and_oranges : ℝ := initial_money - remaining_money

-- This is our theorem statement that needs to be proven
theorem cost_of_apples_and_oranges_correct :
  cost_of_apples_and_oranges = 15.00 :=
by
  sorry

end cost_of_apples_and_oranges_correct_l2_2321


namespace boat_equation_l2_2740

-- Define the conditions given in the problem
def total_boats : ℕ := 8
def large_boat_capacity : ℕ := 6
def small_boat_capacity : ℕ := 4
def total_students : ℕ := 38

-- Define the theorem to be proven
theorem boat_equation (x : ℕ) (h0 : x ≤ total_boats) : 
  large_boat_capacity * (total_boats - x) + small_boat_capacity * x = total_students := by
  sorry

end boat_equation_l2_2740


namespace abs_eq_condition_l2_2073

theorem abs_eq_condition (x : ℝ) (h : |x - 1| + x = 1) : x ≤ 1 :=
by sorry

end abs_eq_condition_l2_2073


namespace rectangle_in_triangle_area_l2_2969

theorem rectangle_in_triangle_area (b h : ℕ) (hb : b = 12) (hh : h = 8)
  (x : ℕ) (hx : x = h / 2) : (b * x / 2) = 48 := 
by
  sorry

end rectangle_in_triangle_area_l2_2969


namespace factor_expression_equals_one_l2_2684

theorem factor_expression_equals_one (a b c : ℝ) :
  ((a^2 - b^2)^2 + (b^2 - c^2)^2 + (c^2 - a^2)^2) / ((a - b)^2 + (b - c)^2 + (c - a)^2) = 1 :=
by
  sorry

end factor_expression_equals_one_l2_2684


namespace max_abs_z_2_2i_l2_2890

open Complex

theorem max_abs_z_2_2i (z : ℂ) (h : abs (z + 2 - 2 * I) = 1) : 
  ∃ w : ℂ, abs (w - 2 - 2 * I) = 5 :=
sorry

end max_abs_z_2_2i_l2_2890


namespace diagonal_length_not_possible_l2_2893

-- Define the side lengths of the parallelogram
def sides_of_parallelogram : ℕ × ℕ := (6, 8)

-- Define the length of a diagonal that cannot exist
def invalid_diagonal_length : ℕ := 15

-- Statement: Prove that a diagonal of length 15 cannot exist for such a parallelogram.
theorem diagonal_length_not_possible (a b d : ℕ) 
  (h₁ : sides_of_parallelogram = (a, b)) 
  (h₂ : d = invalid_diagonal_length) 
  : d ≥ a + b := 
sorry

end diagonal_length_not_possible_l2_2893


namespace solution_set_l2_2413

noncomputable def f : ℝ → ℝ := sorry
def dom := {x : ℝ | x < 0 ∨ x > 0 } -- Definition of the function domain

-- Assumptions and conditions as definitions in Lean
axiom f_odd : ∀ x ∈ dom, f (-x) = -f x
axiom f_at_1 : f 1 = 1
axiom symmetric_f : ∀ x ∈ dom, (f (x + 1)) = -f (-x + 1)
axiom inequality_condition : ∀ (x1 x2 : ℝ), x1 ∈ dom → x2 ∈ dom → x1 ≠ x2 → (x1^3 * f x1 - x2^3 * f x2) / (x1 - x2) > 0

-- The main statement to be proved
theorem solution_set :
  {x ∈ dom | f x ≤ 1 / x^3} = {x ∈ dom | x ≤ -1} ∪ {x ∈ dom | 0 < x ∧ x ≤ 1} :=
sorry

end solution_set_l2_2413


namespace find_f_7_l2_2745

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_7 (h_odd : ∀ x, f (-x) = -f x)
                 (h_periodic : ∀ x, f (x + 4) = f x)
                 (h_interval : ∀ x, 0 < x ∧ x < 2 → f x = 2 * x ^ 2) :
  f 7 = -2 := 
sorry

end find_f_7_l2_2745


namespace red_shells_correct_l2_2808

-- Define the conditions
def total_shells : Nat := 291
def green_shells : Nat := 49
def non_red_green_shells : Nat := 166

-- Define the number of red shells as per the given conditions
def red_shells : Nat :=
  total_shells - green_shells - non_red_green_shells

-- State the theorem
theorem red_shells_correct : red_shells = 76 :=
by
  sorry

end red_shells_correct_l2_2808


namespace tetrahedron_circumsphere_radius_l2_2909

theorem tetrahedron_circumsphere_radius :
  ∃ (r : ℝ), 
    (∀ (A B C P : ℝ × ℝ × ℝ),
      (dist A B = 5) ∧
      (dist A C = 5) ∧
      (dist A P = 5) ∧
      (dist B C = 5) ∧
      (dist B P = 5) ∧
      (dist C P = 6) →
      r = (20 * Real.sqrt 39) / 39) :=
sorry

end tetrahedron_circumsphere_radius_l2_2909


namespace sqrt_20_minus_1_range_l2_2783

theorem sqrt_20_minus_1_range : 
  16 < 20 ∧ 20 < 25 ∧ Real.sqrt 16 = 4 ∧ Real.sqrt 25 = 5 → (3 < Real.sqrt 20 - 1 ∧ Real.sqrt 20 - 1 < 4) :=
by
  intro h
  sorry

end sqrt_20_minus_1_range_l2_2783


namespace find_k_for_perfect_square_trinomial_l2_2293

noncomputable def perfect_square_trinomial (k : ℝ) : Prop :=
∀ x : ℝ, (x^2 - 8*x + k) = (x - 4)^2

theorem find_k_for_perfect_square_trinomial :
  ∃ k : ℝ, perfect_square_trinomial k ∧ k = 16 :=
by
  use 16
  sorry

end find_k_for_perfect_square_trinomial_l2_2293


namespace number_of_arrangements_l2_2084

theorem number_of_arrangements (P : Fin 5 → Type) (youngest : Fin 5) 
  (h_in_not_first_last : ∀ (i : Fin 5), i ≠ 0 → i ≠ 4 → i ≠ youngest) : 
  ∃ n, n = 72 := 
by
  sorry

end number_of_arrangements_l2_2084


namespace tom_final_payment_l2_2722

noncomputable def cost_of_fruit (kg: ℝ) (rate_per_kg: ℝ) := kg * rate_per_kg

noncomputable def total_bill := 
  cost_of_fruit 15.3 1.85 + cost_of_fruit 12.7 2.45 + cost_of_fruit 10.5 3.20 + cost_of_fruit 6.2 4.50

noncomputable def discount (bill: ℝ) := 0.10 * bill

noncomputable def discounted_total (bill: ℝ) := bill - discount bill

noncomputable def sales_tax (amount: ℝ) := 0.06 * amount

noncomputable def final_amount (bill: ℝ) := discounted_total bill + sales_tax (discounted_total bill)

theorem tom_final_payment : final_amount total_bill = 115.36 :=
  sorry

end tom_final_payment_l2_2722


namespace hypotenuse_of_right_angle_triangle_l2_2071

theorem hypotenuse_of_right_angle_triangle {a b c : ℕ} (h1 : a^2 + b^2 = c^2) 
  (h2 : a > 0) (h3 : b > 0) 
  (h4 : a + b + c = (a * b) / 2): 
  c = 10 ∨ c = 13 :=
sorry

end hypotenuse_of_right_angle_triangle_l2_2071


namespace gas_volumes_correct_l2_2002

noncomputable def west_gas_vol_per_capita : ℝ := 21428
noncomputable def non_west_gas_vol : ℝ := 185255
noncomputable def non_west_population : ℝ := 6.9
noncomputable def non_west_gas_vol_per_capita : ℝ := non_west_gas_vol / non_west_population

noncomputable def russia_gas_vol_68_percent : ℝ := 30266.9
noncomputable def russia_gas_vol : ℝ := russia_gas_vol_68_percent * 100 / 68
noncomputable def russia_population : ℝ := 0.147
noncomputable def russia_gas_vol_per_capita : ℝ := russia_gas_vol / russia_population

theorem gas_volumes_correct :
  west_gas_vol_per_capita = 21428 ∧
  non_west_gas_vol_per_capita = 26848.55 ∧
  russia_gas_vol_per_capita = 302790.13 := by
    sorry

end gas_volumes_correct_l2_2002


namespace extreme_value_at_x_eq_one_l2_2288

noncomputable def f (x a b: ℝ) : ℝ := x^3 - a * x^2 + b * x + a^2
noncomputable def f_prime (x a b: ℝ) : ℝ := 3 * x^2 - 2 * a * x + b

theorem extreme_value_at_x_eq_one (a b : ℝ) (h_prime : f_prime 1 a b = 0) (h_value : f 1 a b = 10) : a = -4 :=
by 
  sorry -- proof goes here

end extreme_value_at_x_eq_one_l2_2288


namespace units_digit_quotient_l2_2750

theorem units_digit_quotient (n : ℕ) (h1 : n % 2 = 1): 
  (4^n + 6^n) / 10 % 10 = 1 :=
by 
  -- Given the cyclical behavior of 4^n % 10 and 6^n % 10
  -- 4^n % 10 cycles between 4 and 6, 6^n % 10 is always 6
  -- Since n is odd, 4^n % 10 = 4 and 6^n % 10 = 6
  -- Adding them gives us 4 + 6 = 10, and thus a quotient of 1
  sorry

end units_digit_quotient_l2_2750


namespace find_coeff_and_root_range_l2_2477

def f (x : ℝ) (a b : ℝ) : ℝ := a * x^3 - b * x + 4

theorem find_coeff_and_root_range (a b : ℝ)
  (h1 : f 2 a b = - (4/3))
  (h2 : deriv (λ x => f x a b) 2 = 0) :
  a = 1 / 3 ∧ b = 4 ∧ 
  (∀ k : ℝ, (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 (1/3) 4 = k ∧ f x2 (1/3) 4 = k ∧ f x3 (1/3) 4 = k) ↔ - (4/3) < k ∧ k < 28/3) :=
sorry

end find_coeff_and_root_range_l2_2477


namespace jon_percentage_increase_l2_2723

def initial_speed : ℝ := 80
def trainings : ℕ := 4
def weeks_per_training : ℕ := 4
def speed_increase_per_week : ℝ := 1

theorem jon_percentage_increase :
  let total_weeks := trainings * weeks_per_training
  let total_increase := total_weeks * speed_increase_per_week
  let final_speed := initial_speed + total_increase
  let percentage_increase := (total_increase / initial_speed) * 100
  percentage_increase = 20 :=
by
  sorry

end jon_percentage_increase_l2_2723


namespace descending_order_of_numbers_l2_2225

theorem descending_order_of_numbers :
  let a := 62
  let b := 78
  let c := 64
  let d := 59
  b > c ∧ c > a ∧ a > d :=
by
  let a := 62
  let b := 78
  let c := 64
  let d := 59
  sorry

end descending_order_of_numbers_l2_2225


namespace f_1_eq_2_f_6_plus_f_7_eq_15_f_2012_eq_3849_l2_2488

noncomputable def f : ℕ+ → ℕ+ := sorry

axiom f_properties (n : ℕ+) : f (f n) = 3 * n

axiom f_increasing (n : ℕ+) : f (n + 1) > f n

-- Proof for f(1)
theorem f_1_eq_2 : f 1 = 2 := 
by
sorry

-- Proof for f(6) + f(7)
theorem f_6_plus_f_7_eq_15 : f 6 + f 7 = 15 := 
by
sorry

-- Proof for f(2012)
theorem f_2012_eq_3849 : f 2012 = 3849 := 
by
sorry

end f_1_eq_2_f_6_plus_f_7_eq_15_f_2012_eq_3849_l2_2488


namespace inverse_value_exists_l2_2115

noncomputable def f (a x : ℝ) := a^x - 1

theorem inverse_value_exists (a : ℝ) (h : f a 1 = 1) : (f a)⁻¹ 3 = 2 :=
by
  sorry

end inverse_value_exists_l2_2115


namespace inequality_pow4_geq_sum_l2_2595

theorem inequality_pow4_geq_sum (a b c d e : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0) :
  (a / b) ^ 4 + (b / c) ^ 4 + (c / d) ^ 4 + (d / e) ^ 4 + (e / a) ^ 4 ≥ 
  (a / b) + (b / c) + (c / d) + (d / e) + (e / a) :=
by
  sorry

end inequality_pow4_geq_sum_l2_2595


namespace florist_total_roses_l2_2980

-- Define the known quantities
def originalRoses : ℝ := 37.0
def firstPick : ℝ := 16.0
def secondPick : ℝ := 19.0

-- The theorem stating the total number of roses
theorem florist_total_roses : originalRoses + firstPick + secondPick = 72.0 :=
  sorry

end florist_total_roses_l2_2980


namespace probability_of_defective_on_second_draw_l2_2901

-- Define the conditions
variable (batch_size : ℕ) (defective_items : ℕ) (good_items : ℕ)
variable (first_draw_good : Prop)
variable (without_replacement : Prop)

-- Given conditions
def batch_conditions : Prop :=
  batch_size = 10 ∧ defective_items = 3 ∧ good_items = 7 ∧ first_draw_good ∧ without_replacement

-- The desired probability as a proof
theorem probability_of_defective_on_second_draw
  (h : batch_conditions batch_size defective_items good_items first_draw_good without_replacement) : 
  (3 / 9 : ℝ) = 1 / 3 :=
sorry

end probability_of_defective_on_second_draw_l2_2901


namespace soda_relationship_l2_2846

theorem soda_relationship (J : ℝ) (L : ℝ) (A : ℝ) (hL : L = 1.75 * J) (hA : A = 1.20 * J) : 
  (L - A) / A = 0.46 := 
by
  sorry

end soda_relationship_l2_2846


namespace average_candies_correct_l2_2568

noncomputable def Eunji_candies : ℕ := 35
noncomputable def Jimin_candies : ℕ := Eunji_candies + 6
noncomputable def Jihyun_candies : ℕ := Eunji_candies - 3
noncomputable def Total_candies : ℕ := Eunji_candies + Jimin_candies + Jihyun_candies
noncomputable def Average_candies : ℚ := Total_candies / 3

theorem average_candies_correct :
  Average_candies = 36 := by
  sorry

end average_candies_correct_l2_2568


namespace exactly_one_absent_l2_2600

variables (B K Z : Prop)

theorem exactly_one_absent (h1 : B ∨ K) (h2 : K ∨ Z) (h3 : Z ∨ B)
    (h4 : ¬B ∨ ¬K ∨ ¬Z) : (¬B ∧ K ∧ Z) ∨ (B ∧ ¬K ∧ Z) ∨ (B ∧ K ∧ ¬Z) :=
by
  sorry

end exactly_one_absent_l2_2600


namespace expression_value_l2_2337

theorem expression_value :
  (100 - (3000 - 300) + (3000 - (300 - 100)) = 200) := by
  sorry

end expression_value_l2_2337


namespace no_solution_m_l2_2878

theorem no_solution_m {
  m : ℚ
  } (h : ∀ x : ℚ, x ≠ 3 → (3 - 2 * x) / (x - 3) - (m * x - 2) / (3 - x) ≠ -1) : 
  m = 1 ∨ m = 5 / 3 :=
sorry

end no_solution_m_l2_2878


namespace total_students_in_class_l2_2328

def students_chorus := 18
def students_band := 26
def students_both := 2
def students_neither := 8

theorem total_students_in_class : 
  (students_chorus + students_band - students_both) + students_neither = 50 := by
  sorry

end total_students_in_class_l2_2328


namespace sock_pairs_l2_2856

open Nat

theorem sock_pairs (r g y : ℕ) (hr : r = 5) (hg : g = 6) (hy : y = 4) :
  (choose r 2) + (choose g 2) + (choose y 2) = 31 :=
by
  rw [hr, hg, hy]
  norm_num
  sorry

end sock_pairs_l2_2856


namespace order_scores_l2_2702

theorem order_scores
  (J K M Q S : ℕ)
  (h1 : J ≥ Q) (h2 : J ≥ M) (h3 : J ≥ S) (h4 : J ≥ K)
  (h5 : M > Q ∨ M > S ∨ M > K)
  (h6 : K < S) (h7 : S < J) :
  K < S ∧ S < M ∧ M < Q :=
by
  sorry

end order_scores_l2_2702


namespace algebraic_expression_standard_l2_2994

theorem algebraic_expression_standard :
  (∃ (expr : String), expr = "-(1/3)m" ∧
    expr ≠ "1(2/5)a" ∧
    expr ≠ "m / n" ∧
    expr ≠ "t × 3") :=
  sorry

end algebraic_expression_standard_l2_2994


namespace inequality_of_power_sums_l2_2682

variable (a b c : ℝ)

theorem inequality_of_power_sums (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a < b + c) (h5 : b < c + a) (h6 : c < a + b) :
  a^4 + b^4 + c^4 < 2 * (a^2 * b^2 + b^2 * c^2 + c^2 * a^2) := sorry

end inequality_of_power_sums_l2_2682


namespace divides_of_exponentiation_l2_2607

theorem divides_of_exponentiation (n : ℕ) : 7 ∣ 3^(12 * n + 1) + 2^(6 * n + 2) := 
  sorry

end divides_of_exponentiation_l2_2607


namespace composite_for_all_n_greater_than_one_l2_2478

theorem composite_for_all_n_greater_than_one (n : ℕ) (h : n > 1) : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n^4 + 4^n = a * b :=
by
  sorry

end composite_for_all_n_greater_than_one_l2_2478


namespace train_crossing_time_l2_2602

theorem train_crossing_time
    (length_of_train : ℕ)
    (speed_of_train_kmph : ℕ)
    (length_of_bridge : ℕ)
    (h_train_length : length_of_train = 160)
    (h_speed_kmph : speed_of_train_kmph = 45)
    (h_bridge_length : length_of_bridge = 215)
  : length_of_train + length_of_bridge / ((speed_of_train_kmph * 1000) / 3600) = 30 :=
by
  rw [h_train_length, h_speed_kmph, h_bridge_length]
  norm_num
  sorry

end train_crossing_time_l2_2602


namespace probability_calculation_l2_2368

noncomputable def probability_floor_sqrt_x_eq_17_given_floor_sqrt_2x_eq_25 : ℝ :=
  let total_interval_length := 100
  let intersection_interval_length := 324 - 312.5
  intersection_interval_length / total_interval_length

theorem probability_calculation : probability_floor_sqrt_x_eq_17_given_floor_sqrt_2x_eq_25 = 23 / 200 := by
  sorry

end probability_calculation_l2_2368


namespace expand_product_l2_2287

noncomputable def expand_poly (x : ℝ) : ℝ := (x + 3) * (x^2 + 2 * x + 4)

theorem expand_product (x : ℝ) : expand_poly x = x^3 + 5 * x^2 + 10 * x + 12 := 
by 
  -- This will be filled with the proof steps, but for now we use sorry.
  sorry

end expand_product_l2_2287


namespace area_of_pentagon_AEDCB_l2_2137

structure Rectangle (A B C D : Type) :=
  (AB BC AD CD : ℕ)

def is_perpendicular (A E E' D : Type) : Prop := sorry

def area_of_triangle (AE DE : ℕ) : ℕ :=
  (AE * DE) / 2

def area_of_rectangle (length width : ℕ) : ℕ :=
  length * width

def area_of_pentagon (area_rect area_triangle : ℕ) : ℕ :=
  area_rect - area_triangle

theorem area_of_pentagon_AEDCB
  (A B C D E : Type)
  (h_rectangle : Rectangle A B C D)
  (h_perpendicular : is_perpendicular A E E D)
  (AE DE : ℕ)
  (h_ae : AE = 9)
  (h_de : DE = 12)
  : area_of_pentagon (area_of_rectangle 15 12) (area_of_triangle AE DE) = 126 := 
  sorry

end area_of_pentagon_AEDCB_l2_2137


namespace sum_of_first_8_terms_l2_2918

theorem sum_of_first_8_terms (seq : ℕ → ℝ) (q : ℝ) (h_q : q = 2) 
  (h_sum_first_4 : seq 0 + seq 1 + seq 2 + seq 3 = 1) 
  (h_geom : ∀ n, seq (n + 1) = q * seq n) : 
  seq 0 + seq 1 + seq 2 + seq 3 + seq 4 + seq 5 + seq 6 + seq 7 = 17 := 
sorry

end sum_of_first_8_terms_l2_2918


namespace find_integer_x_l2_2771

theorem find_integer_x (x : ℤ) :
  1 < x ∧ x < 9 ∧ 
  2 < x ∧ x < 15 ∧ 
  0 < x ∧ x < 7 ∧ 
  0 < x ∧ x < 4 ∧ 
  x + 1 < 5 
  → x = 3 :=
by
  intros h
  sorry

end find_integer_x_l2_2771


namespace find_f_prime_at_one_l2_2289

theorem find_f_prime_at_one (a b : ℝ)
  (h1 : ∀ x, f x = a * Real.exp x + b * x) 
  (h2 : f 0 = 1)
  (h3 : ∀ x, deriv f x = a * Real.exp x + b)
  (h4 : deriv f 0 = 0) :
  deriv f 1 = Real.exp 1 - 1 :=
by {
  sorry
}

end find_f_prime_at_one_l2_2289


namespace average_height_of_four_people_l2_2190

theorem average_height_of_four_people (
  h1 h2 h3 h4 : ℕ
) (diff12 : h2 = h1 + 2)
  (diff23 : h3 = h2 + 2)
  (diff34 : h4 = h3 + 6)
  (h4_eq : h4 = 83) :
  (h1 + h2 + h3 + h4) / 4 = 77 :=
by sorry

end average_height_of_four_people_l2_2190


namespace max_min_x_plus_y_on_circle_l2_2996

-- Define the conditions
def polar_eq (ρ θ : Real) : Prop :=
  ρ^2 - 4 * Real.sqrt 2 * ρ * Real.cos (θ - Real.pi / 4) + 6 = 0

-- Define the standard form of the circle
def circle_eq (x y : Real) : Prop :=
  (x - 2)^2 + (y - 2)^2 = 2

-- Define the parametric equations of the circle
def parametric_eq (α : Real) (x y : Real) : Prop :=
  x = 2 + Real.sqrt 2 * Real.cos α ∧ y = 2 + Real.sqrt 2 * Real.sin α

-- Define the problem in Lean
theorem max_min_x_plus_y_on_circle :
  (∀ (ρ θ : Real), polar_eq ρ θ → circle_eq (ρ * Real.cos θ) (ρ * Real.sin θ)) →
  (∀ (α : Real), parametric_eq α (2 + Real.sqrt 2 * Real.cos α) (2 + Real.sqrt 2 * Real.sin α)) →
  (∀ (P : Real × Real), circle_eq P.1 P.2 → 2 ≤ P.1 + P.2 ∧ P.1 + P.2 ≤ 6) :=
by
  intros hpolar hparam P hcircle
  sorry

end max_min_x_plus_y_on_circle_l2_2996


namespace problem_A_plus_B_l2_2340

variable {A B : ℝ} (h1 : A ≠ B) (h2 : ∀ x : ℝ, (A * (B * x + A) + B) - (B * (A * x + B) + A) = 2 * (B - A))

theorem problem_A_plus_B : A + B = -2 :=
by
  sorry

end problem_A_plus_B_l2_2340


namespace total_questions_correct_total_answers_correct_l2_2824

namespace ForumCalculation

def members : ℕ := 200
def questions_per_hour_per_user : ℕ := 3
def hours_in_day : ℕ := 24
def answers_multiplier : ℕ := 3

def total_questions_per_user_per_day : ℕ :=
  questions_per_hour_per_user * hours_in_day

def total_questions_in_a_day : ℕ :=
  members * total_questions_per_user_per_day

def total_answers_per_user_per_day : ℕ :=
  answers_multiplier * total_questions_per_user_per_day

def total_answers_in_a_day : ℕ :=
  members * total_answers_per_user_per_day

theorem total_questions_correct :
  total_questions_in_a_day = 14400 :=
by
  sorry

theorem total_answers_correct :
  total_answers_in_a_day = 43200 :=
by
  sorry

end ForumCalculation

end total_questions_correct_total_answers_correct_l2_2824


namespace dart_board_probability_l2_2945

variable {s : ℝ} (hexagon_area : ℝ := (3 * Real.sqrt 3) / 2 * s^2) (center_hexagon_area : ℝ := (3 * Real.sqrt 3) / 8 * s^2)

theorem dart_board_probability (s : ℝ) (P : ℝ) (h : P = center_hexagon_area / hexagon_area) :
  P = 1 / 4 :=
by
  sorry

end dart_board_probability_l2_2945


namespace slope_of_tangent_line_at_A_l2_2984

noncomputable def f (x : ℝ) := x^2 + 3 * x

def derivative_at (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  (sorry : ℝ)  -- Placeholder for the definition of the derivative

theorem slope_of_tangent_line_at_A : 
  derivative_at f 1 = 5 := 
sorry

end slope_of_tangent_line_at_A_l2_2984


namespace linear_system_solution_l2_2988

theorem linear_system_solution (k x y : ℝ) (h₁ : x + y = 5 * k) (h₂ : x - y = 9 * k) (h₃ : 2 * x + 3 * y = 6) :
  k = 3 / 4 :=
by
  sorry

end linear_system_solution_l2_2988


namespace positive_integer_is_48_l2_2320

theorem positive_integer_is_48 (n p : ℕ) (h_prime : Prime p) (h_eq : n = 24 * p) (h_min : n ≥ 48) : n = 48 :=
by
  sorry

end positive_integer_is_48_l2_2320


namespace find_g_two_l2_2256

variable (g : ℝ → ℝ)

-- Condition 1: Functional equation
axiom g_eq : ∀ x y : ℝ, g (x - y) = g x * g y

-- Condition 2: Non-zero property
axiom g_ne_zero : ∀ x : ℝ, g x ≠ 0

-- Proof statement
theorem find_g_two : g 2 = 1 := 
by sorry

end find_g_two_l2_2256


namespace three_monotonic_intervals_l2_2352

open Real

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := - (4 / 3) * x ^ 3 + (b - 1) * x

noncomputable def f' (b : ℝ) (x : ℝ) : ℝ := -4 * x ^ 2 + (b - 1)

theorem three_monotonic_intervals (b : ℝ) (h : (b - 1) > 0) : b > 1 := 
by
  have discriminant : 16 * (b - 1) > 0 := sorry
  sorry

end three_monotonic_intervals_l2_2352


namespace distance_between_meeting_points_is_48_l2_2540

noncomputable def distance_between_meeting_points 
    (d : ℝ) -- total distance between points A and B
    (first_meeting_from_B : ℝ)   -- distance of the first meeting point from B
    (second_meeting_from_A : ℝ) -- distance of the second meeting point from A
    (second_meeting_from_B : ℝ) : ℝ :=
    (second_meeting_from_B - first_meeting_from_B)

theorem distance_between_meeting_points_is_48 
    (d : ℝ)
    (hm1 : first_meeting_from_B = 108)
    (hm2 : second_meeting_from_A = 84) 
    (hm3 : second_meeting_from_B = d - 24) :
    distance_between_meeting_points d first_meeting_from_B second_meeting_from_A second_meeting_from_B = 48 := by
  sorry

end distance_between_meeting_points_is_48_l2_2540


namespace find_ordered_pair_l2_2679

theorem find_ordered_pair : 
  ∃ (x y : ℚ), 7 * x = -5 - 3 * y ∧ 4 * x = 5 * y - 34 ∧
  x = -127 / 47 ∧ y = 218 / 47 :=
by
  sorry

end find_ordered_pair_l2_2679


namespace percentage_cut_l2_2986

theorem percentage_cut (S C : ℝ) (hS : S = 940) (hC : C = 611) :
  (C / S) * 100 = 65 := 
by
  rw [hS, hC]
  norm_num

end percentage_cut_l2_2986


namespace machines_job_completion_time_l2_2858

theorem machines_job_completion_time (t : ℕ) 
  (hR_rate : ∀ t, 1 / t = 1 / 216) 
  (hS_rate : ∀ t, 1 / t = 1 / 216) 
  (same_num_machines : ∀ R S, R = 9 ∧ S = 9) 
  (total_time : 12 = 12) 
  (jobs_completed : 1 = (18 / t) * 12) : 
  t = 216 := 
sorry

end machines_job_completion_time_l2_2858


namespace problem1_problem2_l2_2930

-- Definitions for first problem
def increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f x ≤ f y

-- Theorem for first problem
theorem problem1 (f : ℝ → ℝ) (h1 : increasing_function f) (h2 : ∀ x, -3 ≤ x → x ≤ 3) (h : f (m + 1) > f (2 * m - 1)) :
  -1 ≤ m ∧ m < 2 :=
sorry

-- Definitions for second problem
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Theorem for second problem
theorem problem2 (f : ℝ → ℝ) (h1 : increasing_function f) (h2 : odd_function f) (h3 : f 2 = 1) (h4 : ∀ x, -3 ≤ x → x ≤ 3) :
  ∀ x, f (x + 1) + 1 > 0 ↔ -3 < x ∧ x ≤ 2 :=
sorry

end problem1_problem2_l2_2930


namespace minimum_value_expression_l2_2122

theorem minimum_value_expression {x1 x2 x3 x4 : ℝ} (hx1 : 0 < x1) (hx2 : 0 < x2) (hx3 : 0 < x3) (hx4 : 0 < x4) (h_sum : x1 + x2 + x3 + x4 = Real.pi) :
  (2 * (Real.sin x1)^2 + 1 / (Real.sin x1)^2) * (2 * (Real.sin x2)^2 + 1 / (Real.sin x2)^2) * (2 * (Real.sin x3)^2 + 1 / (Real.sin x3)^2) * (2 * (Real.sin x4)^2 + 1 / (Real.sin x4)^2) ≥ 81 :=
by {
  sorry
}

end minimum_value_expression_l2_2122


namespace average_rst_l2_2013

theorem average_rst (r s t : ℝ) (h : (5 / 4) * (r + s + t - 2) = 15) : (r + s + t) / 3 = 14 / 3 :=
by
  sorry

end average_rst_l2_2013


namespace principal_invested_years_l2_2599

-- Define the given conditions
def principal : ℕ := 9200
def rate : ℕ := 12
def interest_deficit : ℤ := 5888

-- Define the time to be proved
def time_invested : ℤ := 3

-- Define the simple interest formula
def simple_interest (P R t : ℕ) : ℕ :=
  (P * R * t) / 100

-- Define the problem statement
theorem principal_invested_years :
  ∃ t : ℕ, principal - interest_deficit = simple_interest principal rate t ∧ t = time_invested := 
by
  sorry

end principal_invested_years_l2_2599


namespace kristen_turtles_l2_2015

variable (K : ℕ)
variable (T : ℕ)
variable (R : ℕ)

-- Conditions
def kris_turtles (K : ℕ) : ℕ := K / 4
def trey_turtles (R : ℕ) : ℕ := 7 * R
def trey_more_than_kristen (T K : ℕ) : Prop := T = K + 9

-- Theorem to prove 
theorem kristen_turtles (K : ℕ) (R : ℕ) (T : ℕ) (h1 : R = kris_turtles K) (h2 : T = trey_turtles R) (h3 : trey_more_than_kristen T K) : K = 12 :=
by
  sorry

end kristen_turtles_l2_2015


namespace incircle_radius_of_right_triangle_l2_2344

noncomputable def radius_of_incircle (a b c : ℝ) : ℝ := (a + b - c) / 2

theorem incircle_radius_of_right_triangle
  (a : ℝ) (b_proj_hypotenuse : ℝ) (r : ℝ) :
  a = 15 ∧ b_proj_hypotenuse = 16 ∧ r = 5 :=
by
  sorry

end incircle_radius_of_right_triangle_l2_2344


namespace intersection_of_sets_l2_2529

open Set

theorem intersection_of_sets (p q : ℝ) :
  (M = {x : ℝ | x^2 - 5 * x < 0}) →
  (M = {x : ℝ | 0 < x ∧ x < 5}) →
  (N = {x : ℝ | p < x ∧ x < 6}) →
  (M ∩ N = {x : ℝ | 2 < x ∧ x < q}) →
  p + q = 7 :=
by
  intros h1 h2 h3 h4
  sorry

end intersection_of_sets_l2_2529


namespace part_I_part_II_l2_2698

noncomputable def f (x m : ℝ) : ℝ := |3 * x + m|
noncomputable def g (x m : ℝ) : ℝ := f x m - 2 * |x - 1|

theorem part_I (m : ℝ) : (∀ x : ℝ, (f x m - m ≤ 9) ↔ (-1 ≤ x ∧ x ≤ 3)) → m = -3 :=
by
  sorry

theorem part_II (m : ℝ) (h : m > 0) : (∃ A B C : ℝ × ℝ, 
  let A := (-m-2, 0)
  let B := ((2-m)/5, 0)
  let C := (-m/3, -2*m/3-2)
  let Area : ℝ := 1/2 * |(B.1 - A.1) * (C.2 - 0) - (B.2 - A.2) * (C.1 - A.1)|
  Area > 60 ) → m > 12 :=
by
  sorry

end part_I_part_II_l2_2698


namespace num_distinct_orders_of_targets_l2_2152

theorem num_distinct_orders_of_targets : 
  let total_targets := 10
  let column_A_targets := 4
  let column_B_targets := 4
  let column_C_targets := 2
  (Nat.factorial total_targets) / 
  ((Nat.factorial column_A_targets) * (Nat.factorial column_B_targets) * (Nat.factorial column_C_targets)) = 5040 := 
by
  sorry

end num_distinct_orders_of_targets_l2_2152


namespace sum_first_20_integers_l2_2749

def sum_first_n_integers (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem sum_first_20_integers : sum_first_n_integers 20 = 210 :=
by
  -- Provided proof omitted
  sorry

end sum_first_20_integers_l2_2749


namespace find_number_l2_2088

theorem find_number (n : ℕ) (h1 : n % 5 = 0) (h2 : 70 ≤ n ∧ n ≤ 90) (h3 : Nat.Prime n) : n = 85 := 
sorry

end find_number_l2_2088


namespace problem_solution_l2_2805

theorem problem_solution (m n : ℕ) (h1 : m + 7 < n + 3) 
  (h2 : (m + (m+3) + (m+7) + (n+3) + (n+6) + 2 * n) / 6 = n + 3) 
  (h3 : (m + 7 + n + 3) / 2 = n + 3) : m + n = 12 := 
  sorry

end problem_solution_l2_2805


namespace royal_children_count_l2_2954

theorem royal_children_count :
  ∀ (d n : ℕ), 
    d ≥ 1 → 
    n = 35 / (d + 1) →
    (d + 3) ≤ 20 →
    (d + 3 = 7 ∨ d + 3 = 9) :=
by
  intros d n H1 H2 H3
  sorry

end royal_children_count_l2_2954


namespace directrix_of_parabola_l2_2721

theorem directrix_of_parabola :
  ∀ (x y : ℝ), (y = (x^2 - 4 * x + 4) / 8) → y = -2 :=
sorry

end directrix_of_parabola_l2_2721


namespace line_intersects_y_axis_at_0_2_l2_2623

theorem line_intersects_y_axis_at_0_2 (P1 P2 : ℝ × ℝ) (h1 : P1 = (2, 8)) (h2 : P2 = (6, 20)) :
  ∃ y : ℝ, (0, y) = (0, 2) :=
by {
  sorry
}

end line_intersects_y_axis_at_0_2_l2_2623


namespace vector_dot_product_value_l2_2219

def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ := (-3, 4)
def c : ℝ × ℝ := (3, 2)

def scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem vector_dot_product_value : dot_product (add (scalar_mul 2 a) b) c = -3 := by
  sorry

end vector_dot_product_value_l2_2219


namespace david_marks_in_english_l2_2052

variable (E : ℕ)
variable (marks_in_math : ℕ := 98)
variable (marks_in_physics : ℕ := 99)
variable (marks_in_chemistry : ℕ := 100)
variable (marks_in_biology : ℕ := 98)
variable (average_marks : ℚ := 98.2)
variable (num_subjects : ℕ := 5)

theorem david_marks_in_english 
  (H1 : average_marks = (E + marks_in_math + marks_in_physics + marks_in_chemistry + marks_in_biology) / num_subjects) :
  E = 96 :=
sorry

end david_marks_in_english_l2_2052


namespace find_a_value_l2_2691

theorem find_a_value
  (a : ℕ)
  (x y : ℝ)
  (h1 : a * x + y = -4)
  (h2 : 2 * x + y = -2)
  (hx_neg : x < 0)
  (hy_pos : y > 0) :
  a = 3 :=
by
  sorry

end find_a_value_l2_2691


namespace find_d_l2_2301

-- Definitions based on conditions
def f (x : ℝ) (c : ℝ) := 5 * x + c
def g (x : ℝ) (c : ℝ) := c * x + 3

-- The theorem statement
theorem find_d (c d : ℝ) (h₁ : f (g x c) c = 15 * x + d) : d = 18 :=
by
  sorry -- Proof is omitted as per the instructions

end find_d_l2_2301


namespace smallest_positive_integer_y_l2_2048

theorem smallest_positive_integer_y
  (y : ℕ)
  (h_pos : 0 < y)
  (h_ineq : y^3 > 80) :
  y = 5 :=
sorry

end smallest_positive_integer_y_l2_2048


namespace overall_ranking_l2_2630

-- Define the given conditions
def total_participants := 99
def rank_number_theory := 16
def rank_combinatorics := 30
def rank_geometry := 23
def exams := ["geometry", "number_theory", "combinatorics"]
def final_ranking_strategy := "sum_of_scores"

-- Given: best possible rank and worst possible rank should be the same in this specific problem (from solution steps).
def best_possible_rank := 67
def worst_possible_rank := 67

-- Mathematically prove that 100 * best possible rank + worst possible rank = 167
theorem overall_ranking :
  100 * best_possible_rank + worst_possible_rank = 167 :=
by {
  -- Add the "sorry" here to skip the proof, as required:
  sorry
}

end overall_ranking_l2_2630


namespace find_angle_D_l2_2717

noncomputable def calculate_angle (A B C D : ℝ) : ℝ :=
  if (A + B = 180) ∧ (C = D) ∧ (A = 2 * D - 10) then D else 0

theorem find_angle_D (A B C D : ℝ) (h1: A + B = 180) (h2: C = D) (h3: A = 2 * D - 10) : D = 70 :=
by
  sorry

end find_angle_D_l2_2717


namespace total_cases_sold_is_correct_l2_2182

-- Define the customer groups and their respective number of cases bought
def n1 : ℕ := 8
def k1 : ℕ := 3
def n2 : ℕ := 4
def k2 : ℕ := 2
def n3 : ℕ := 8
def k3 : ℕ := 1

-- Define the total number of cases sold
def total_cases_sold : ℕ := n1 * k1 + n2 * k2 + n3 * k3

-- The proof statement that the total cases sold is 40
theorem total_cases_sold_is_correct : total_cases_sold = 40 := by
  -- Proof content will be provided here.
  sorry

end total_cases_sold_is_correct_l2_2182


namespace find_n_l2_2476

theorem find_n (n k : ℕ) (h_pos : k > 0) (h_calls : ∀ (s : Finset (Fin n)), s.card = n-2 → (∃ (f : Finset (Fin n × Fin n)), f.card = 3^k ∧ ∀ (x y : Fin n), (x, y) ∈ f → x ≠ y)) : n = 5 := 
sorry

end find_n_l2_2476


namespace number_of_students_joined_l2_2377

theorem number_of_students_joined
  (A : ℝ)
  (x : ℕ)
  (h1 : A = 50)
  (h2 : (100 + x) * (A - 10) = 5400) 
  (h3 : 100 * A + 400 = 5400) :
  x = 35 := 
by 
  -- all conditions in a) are used as definitions in Lean 4 statement
  sorry

end number_of_students_joined_l2_2377


namespace two_trains_cross_time_l2_2486

/-- Definition for the two trains' parameters -/
structure Train :=
  (length : ℝ)  -- length in meters
  (speed : ℝ)  -- speed in km/hr

/-- The parameters of Train 1 and Train 2 -/
def train1 : Train := { length := 140, speed := 60 }
def train2 : Train := { length := 160, speed := 40 }

noncomputable def relative_speed_mps (t1 t2 : Train) : ℝ :=
  (t1.speed + t2.speed) * (5 / 18)

noncomputable def total_length (t1 t2 : Train) : ℝ :=
  t1.length + t2.length

noncomputable def time_to_cross (t1 t2 : Train) : ℝ :=
  total_length t1 t2 / relative_speed_mps t1 t2

theorem two_trains_cross_time :
  time_to_cross train1 train2 = 10.8 := by
  sorry

end two_trains_cross_time_l2_2486


namespace suji_present_age_l2_2297

/-- Present ages of Abi and Suji are in the ratio of 5:4. --/
def abi_suji_ratio (abi_age suji_age : ℕ) : Prop := abi_age = 5 * (suji_age / 4)

/-- 3 years hence, the ratio of their ages will be 11:9. --/
def abi_suji_ratio_future (abi_age suji_age : ℕ) : Prop :=
  ((abi_age + 3).toFloat / (suji_age + 3).toFloat) = 11 / 9

theorem suji_present_age (suji_age : ℕ) (abi_age : ℕ) (x : ℕ) 
  (h1 : abi_age = 5 * x) (h2 : suji_age = 4 * x)
  (h3 : abi_suji_ratio_future abi_age suji_age) :
  suji_age = 24 := 
sorry

end suji_present_age_l2_2297


namespace domain_of_function_l2_2028

theorem domain_of_function (x : ℝ) : (|x - 2| + |x + 2| ≠ 0) := 
sorry

end domain_of_function_l2_2028


namespace tiles_painted_in_15_minutes_l2_2974

open Nat

theorem tiles_painted_in_15_minutes:
  let don_rate := 3
  let ken_rate := don_rate + 2
  let laura_rate := 2 * ken_rate
  let kim_rate := laura_rate - 3
  don_rate + ken_rate + laura_rate + kim_rate == 25 → 
  15 * (don_rate + ken_rate + laura_rate + kim_rate) = 375 :=
by
  intros
  sorry

end tiles_painted_in_15_minutes_l2_2974


namespace amount_after_two_years_l2_2049

theorem amount_after_two_years (P : ℝ) (r1 r2 : ℝ) : 
  P = 64000 → 
  r1 = 0.12 → 
  r2 = 0.15 → 
  (P + P * r1) + (P + P * r1) * r2 = 82432 := by
  sorry

end amount_after_two_years_l2_2049


namespace simplify_and_evaluate_expression_l2_2706

theorem simplify_and_evaluate_expression : 
  ∀ (x y : ℤ), x = -1 → y = 2 → -2 * x^2 * y - 3 * (2 * x * y - x^2 * y) + 4 * x * y = 6 :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end simplify_and_evaluate_expression_l2_2706


namespace integral_sqrt_1_minus_x_sq_plus_2x_l2_2246

theorem integral_sqrt_1_minus_x_sq_plus_2x :
  ∫ x in (0 : Real)..1, (Real.sqrt (1 - x^2) + 2 * x) = (Real.pi + 4) / 4 := by
  sorry

end integral_sqrt_1_minus_x_sq_plus_2x_l2_2246


namespace inverse_proportion_comparison_l2_2937

theorem inverse_proportion_comparison (y1 y2 : ℝ) 
  (h1 : y1 = - 6 / 2)
  (h2 : y2 = - 6 / -1) : 
  y1 < y2 :=
by
  sorry

end inverse_proportion_comparison_l2_2937


namespace quadratic_has_distinct_real_roots_l2_2189

theorem quadratic_has_distinct_real_roots :
  ∀ (x : ℝ), x^2 - 2 * x - 1 = 0 → (∃ Δ > 0, Δ = ((-2)^2 - 4 * 1 * (-1))) := by
  sorry

end quadratic_has_distinct_real_roots_l2_2189


namespace inequality_for_pos_reals_equality_condition_l2_2138

open Real

theorem inequality_for_pos_reals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / c + c / b ≥ 4 * a / (a + b) :=
by
  -- Theorem Statement Proof Skeleton
  sorry

theorem equality_condition (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / c + c / b = 4 * a / (a + b)) ↔ (a = b ∧ b = c) :=
by
  -- Theorem Statement Proof Skeleton
  sorry

end inequality_for_pos_reals_equality_condition_l2_2138


namespace problem_correct_statements_l2_2585

def T (a b x y : ℚ) : ℚ := a * x * y + b * x - 4

theorem problem_correct_statements (a b : ℚ) (h₁ : T a b 2 1 = 2) (h₂ : T a b (-1) 2 = -8) :
  (a = 1 ∧ b = 2) ∧
  (∀ m n : ℚ, T 1 2 m n = 0 ∧ n ≠ -2 → m = 4 / (n + 2)) ∧
  ¬ (∃ m n : ℤ, T 1 2 m n = 0 ∧ n ≠ -2 ∧ m + n = 3) ∧
  (∀ k x y : ℚ, T 1 2 (k * x) y = T 1 2 (k * x) y → y = -2) ∧
  (∀ k x y : ℚ, x ≠ y → T 1 2 (k * x) y = T 1 2 (k * y) x → k = 0) :=
by
  sorry

end problem_correct_statements_l2_2585


namespace find_angle_x_l2_2939

theorem find_angle_x (angle_ABC angle_BAC angle_BCA angle_DCE angle_CED x : ℝ)
  (h1 : angle_ABC + angle_BAC + angle_BCA = 180)
  (h2 : angle_ABC = 70) 
  (h3 : angle_BAC = 50)
  (h4 : angle_DCE + angle_CED = 90)
  (h5 : angle_DCE = angle_BCA) :
  x = 30 :=
by
  sorry

end find_angle_x_l2_2939


namespace find_lesser_fraction_l2_2842

theorem find_lesser_fraction (x y : ℚ) (h₁ : x + y = 3 / 4) (h₂ : x * y = 1 / 8) : min x y = 1 / 4 := 
by 
  sorry

end find_lesser_fraction_l2_2842


namespace identify_radioactive_balls_l2_2710

theorem identify_radioactive_balls (balls : Fin 11 → Bool) (measure : (Finset (Fin 11)) → Bool) :
  (∃ (t1 t2 : Fin 11), ¬ t1 = t2 ∧ balls t1 = true ∧ balls t2 = true) →
  (∃ (pairs : List (Finset (Fin 11))), pairs.length ≤ 7 ∧
    ∀ t1 t2, t1 ≠ t2 ∧ balls t1 = true ∧ balls t2 = true →
      ∃ pair ∈ pairs, measure pair = true ∧ (t1 ∈ pair ∨ t2 ∈ pair)) :=
by
  sorry

end identify_radioactive_balls_l2_2710


namespace min_value_of_fraction_l2_2003

theorem min_value_of_fraction (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2 * x + 3 * y = 3) : 
  (3 / x + 2 / y) = 8 :=
sorry

end min_value_of_fraction_l2_2003


namespace inscribed_rectangle_area_l2_2206

variables (b h x : ℝ)
variables (h_isosceles_triangle : b > 0 ∧ h > 0 ∧ x > 0 ∧ x < h)

noncomputable def rectangle_area (b h x : ℝ) : ℝ :=
  (b * x / h) * (h - x)

theorem inscribed_rectangle_area :
  rectangle_area b h x = (b * x / h) * (h - x) :=
by
  unfold rectangle_area
  sorry

end inscribed_rectangle_area_l2_2206


namespace totalSandwiches_l2_2108

def numberOfPeople : ℝ := 219.0
def sandwichesPerPerson : ℝ := 3.0

theorem totalSandwiches : numberOfPeople * sandwichesPerPerson = 657.0 := by
  -- Proof goes here
  sorry

end totalSandwiches_l2_2108


namespace max_erasers_l2_2493

theorem max_erasers (p n e : ℕ) (h₁ : p ≥ 1) (h₂ : n ≥ 1) (h₃ : e ≥ 1) (h₄ : 3 * p + 4 * n + 8 * e = 60) :
  e ≤ 5 :=
sorry

end max_erasers_l2_2493


namespace problem_1_problem_2_l2_2648

def f (x : ℝ) (a : ℝ) : ℝ := |x + 2| - |x + a|

theorem problem_1 (a : ℝ) (h : a = 3) :
  ∀ x, f x a ≤ 1/2 → x ≥ -11/4 := sorry

theorem problem_2 (a : ℝ) :
  (∀ x, f x a ≤ a) → a ≥ 1 := sorry

end problem_1_problem_2_l2_2648


namespace simplify_expression_l2_2775

theorem simplify_expression : 
  (1 / ((1 / (1 / 3)^1) + (1 / (1 / 3)^2) + (1 / (1 / 3)^3))) = 1 / 39 :=
by
  sorry

end simplify_expression_l2_2775


namespace sum_of_digits_of_N_is_19_l2_2424

-- Given facts about N
variables (N : ℕ) (h1 : 100 ≤ N ∧ N < 1000) 
           (h2 : N % 10 = 7) 
           (h3 : N % 11 = 7) 
           (h4 : N % 12 = 7)

-- Main theorem statement
theorem sum_of_digits_of_N_is_19 : 
  ((N / 100) + ((N % 100) / 10) + (N % 10) = 19) := sorry

end sum_of_digits_of_N_is_19_l2_2424


namespace john_must_work_10_more_days_l2_2936

-- Define the conditions as hypotheses
def total_days_worked := 10
def total_earnings := 250
def desired_total_earnings := total_earnings * 2
def daily_earnings := total_earnings / total_days_worked

-- Theorem that needs to be proved
theorem john_must_work_10_more_days:
  (desired_total_earnings / daily_earnings) - total_days_worked = 10 := by
  sorry

end john_must_work_10_more_days_l2_2936


namespace inequality_no_solution_l2_2564

-- Define the quadratic inequality.
def quadratic_ineq (m x : ℝ) : Prop :=
  (m + 1) * x^2 - m * x + (m - 1) > 0

-- Define the condition for m.
def range_of_m (m : ℝ) : Prop :=
  m ≤ - (2 * Real.sqrt 3) / 3

-- Theorem stating that if the inequality has no solution, m gets restricted.
theorem inequality_no_solution (m : ℝ) :
  (∀ x : ℝ, ¬ quadratic_ineq m x) ↔ range_of_m m :=
by sorry

end inequality_no_solution_l2_2564


namespace solution_l2_2886

noncomputable def determine_numbers (x y : ℚ) : Prop :=
  x^2 + y^2 = 45 / 4 ∧ x - y = x * y

theorem solution (x y : ℚ) :
  determine_numbers x y → (x = -3 ∧ y = 3/2) ∨ (x = -3/2 ∧ y = 3) :=
-- We state the main theorem that relates the determine_numbers predicate to the specific pairs of numbers
sorry

end solution_l2_2886


namespace technicans_permanent_50pct_l2_2470

noncomputable def percentage_technicians_permanent (p : ℝ) : Prop :=
  let technicians := 0.5
  let non_technicians := 0.5
  let temporary := 0.5
  (0.5 * (1 - 0.5)) + (technicians * p) = 0.5 ->
  p = 0.5

theorem technicans_permanent_50pct (p : ℝ) :
  percentage_technicians_permanent p :=
sorry

end technicans_permanent_50pct_l2_2470


namespace multiply_469160_999999_l2_2281

theorem multiply_469160_999999 :
  469160 * 999999 = 469159530840 :=
by
  sorry

end multiply_469160_999999_l2_2281


namespace smallest_consecutive_divisible_by_17_l2_2061

def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_consecutive_divisible_by_17 :
  ∃ (n m : ℕ), 
    (m = n + 1) ∧
    sum_digits n % 17 = 0 ∧ 
    sum_digits m % 17 = 0 ∧ 
    n = 8899 ∧ 
    m = 8900 := 
by
  sorry

end smallest_consecutive_divisible_by_17_l2_2061


namespace sin_double_angle_identity_l2_2292

noncomputable def given_tan_alpha (α : ℝ) : Prop := 
  Real.tan α = 1/2

theorem sin_double_angle_identity (α : ℝ) (h : given_tan_alpha α) : 
  Real.sin (2 * α) = 4 / 5 := 
sorry

end sin_double_angle_identity_l2_2292


namespace prob_both_primes_l2_2393

-- Define the set of integers from 1 through 30
def int_set : Set ℕ := {n | 1 ≤ n ∧ n ≤ 30}

-- Define the set of prime numbers between 1 and 30
def primes_between_1_and_30 : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

-- Calculate the number of ways to choose two distinct elements from a set
noncomputable def combination (n k : ℕ) : ℕ := if k > n then 0 else n.choose k

-- Define the probabilities
noncomputable def prob_primes : ℚ :=
  (combination 10 2) / (combination 30 2)

-- State the theorem to prove
theorem prob_both_primes : prob_primes = 10 / 87 := by
  sorry

end prob_both_primes_l2_2393


namespace repeating_decimal_denominators_l2_2516

theorem repeating_decimal_denominators (a b c : ℕ) (ha : 0 ≤ a ∧ a < 10) (hb : 0 ≤ b ∧ b < 10) (hc : 0 ≤ c ∧ c < 10) (h_not_all_nine : ¬(a = 9 ∧ b = 9 ∧ c = 9)) : 
  ∃ denominators : Finset ℕ, denominators.card = 7 ∧ (∀ d ∈ denominators, d ∣ 999) ∧ ¬ 1 ∈ denominators :=
sorry

end repeating_decimal_denominators_l2_2516


namespace smaller_of_two_digit_numbers_with_product_2210_l2_2851

theorem smaller_of_two_digit_numbers_with_product_2210 :
  ∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 2210 ∧ a ≤ b ∧ a = 26 :=
by
  sorry

end smaller_of_two_digit_numbers_with_product_2210_l2_2851


namespace range_of_a_l2_2317

-- Definitions
def domain_f : Set ℝ := {x : ℝ | x ≤ -4 ∨ x ≥ 4}
def range_g (a : ℝ) : Set ℝ := {y : ℝ | ∃ x : ℝ, 0 ≤ x ∧ x ≤ 4 ∧ y = x^2 - 2*x + a}

-- Theorem to prove the range of values for a
theorem range_of_a :
  (∀ x : ℝ, x ∈ domain_f ∨ (∃ y : ℝ, ∃ a : ℝ, y ∈ range_g a ∧ x = y)) ↔ (-4 ≤ a ∧ a ≤ -3) :=
sorry

end range_of_a_l2_2317


namespace F_double_reflection_l2_2304

structure Point where
  x : ℝ
  y : ℝ

def reflect_y (p : Point) : Point :=
  { x := -p.x, y := p.y }

def reflect_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

def F : Point := { x := -1, y := -1 }

theorem F_double_reflection :
  reflect_x (reflect_y F) = { x := 1, y := 1 } :=
  sorry

end F_double_reflection_l2_2304


namespace complex_fraction_simplification_l2_2198

theorem complex_fraction_simplification (i : ℂ) (h : i^2 = -1) : (2 : ℂ) / (1 + i)^2 = i :=
by 
-- this will be filled when proving the theorem in Lean
sorry

end complex_fraction_simplification_l2_2198


namespace exponentiation_rule_l2_2336

theorem exponentiation_rule (m n : ℤ) : (-2 * m^3 * n^2)^2 = 4 * m^6 * n^4 :=
by
  sorry

end exponentiation_rule_l2_2336


namespace minimum_value_of_fraction_l2_2964

theorem minimum_value_of_fraction (a b : ℝ) (h1 : a > 2 * b) (h2 : 2 * b > 0) :
  (a^4 + 1) / (b * (a - 2 * b)) >= 16 :=
sorry

end minimum_value_of_fraction_l2_2964


namespace value_of_a_sq_sub_b_sq_l2_2365

theorem value_of_a_sq_sub_b_sq (a b : ℝ) (h1 : a + b = 20) (h2 : a - b = 4) : a^2 - b^2 = 80 :=
by
  sorry

end value_of_a_sq_sub_b_sq_l2_2365


namespace description_of_T_l2_2332

def T (x y : ℝ) : Prop :=
  (5 = x+3 ∧ y-6 ≤ 5) ∨
  (5 = y-6 ∧ x+3 ≤ 5) ∨
  ((x+3 = y-6) ∧ 5 ≤ x+3)

theorem description_of_T :
  ∀ (x y : ℝ), T x y ↔ (x = 2 ∧ y ≤ 11) ∨ (y = 11 ∧ x ≤ 2) ∨ (y = x + 9 ∧ x ≥ 2) :=
sorry

end description_of_T_l2_2332


namespace pipe_fill_time_with_leak_l2_2220

theorem pipe_fill_time_with_leak (A L : ℝ) (hA : A = 1 / 2) (hL : L = 1 / 6) :
  (1 / (A - L)) = 3 :=
by
  sorry

end pipe_fill_time_with_leak_l2_2220


namespace find_YW_in_triangle_l2_2995

theorem find_YW_in_triangle
  (X Y Z W : Type)
  (d_XZ d_YZ d_XW d_CW : ℝ)
  (h_XZ : d_XZ = 10)
  (h_YZ : d_YZ = 10)
  (h_XW : d_XW = 12)
  (h_CW : d_CW = 5) : 
  YW = 29 / 12 :=
sorry

end find_YW_in_triangle_l2_2995


namespace smallest_prime_less_than_square_l2_2173

theorem smallest_prime_less_than_square : ∃ p n : ℕ, Prime p ∧ p = n^2 - 20 ∧ p = 5 :=
by 
  sorry

end smallest_prime_less_than_square_l2_2173


namespace paul_crayons_l2_2379

def initial_crayons : ℝ := 479.0
def additional_crayons : ℝ := 134.0
def total_crayons : ℝ := initial_crayons + additional_crayons

theorem paul_crayons : total_crayons = 613.0 :=
by
  sorry

end paul_crayons_l2_2379


namespace cos_A_eq_neg_quarter_l2_2759

-- Definitions of angles and sides in the triangle
variables (A B C : ℝ)
variables (a b c : ℝ)

-- Conditions from the math problem
axiom sin_arithmetic_sequence : 2 * Real.sin B = Real.sin A + Real.sin C
axiom side_relation : a = 2 * c

-- Question to be proved as Lean 4 statement
theorem cos_A_eq_neg_quarter (h1 : ∀ {x y z : ℝ}, 2 * y = x + z) 
                              (h2 : ∀ {a b c : ℝ}, a = 2 * c) : 
                              Real.cos A = -1/4 := 
sorry

end cos_A_eq_neg_quarter_l2_2759


namespace art_collection_area_l2_2441

theorem art_collection_area :
  let square_paintings := 3 * (6 * 6)
  let small_paintings := 4 * (2 * 3)
  let large_painting := 1 * (10 * 15)
  square_paintings + small_paintings + large_painting = 282 := by
  sorry

end art_collection_area_l2_2441


namespace racers_meet_at_start_again_l2_2982

-- We define the conditions as given
def RacingMagic_time := 60
def ChargingBull_time := 60 * 60 / 40 -- 90 seconds
def SwiftShadow_time := 80
def SpeedyStorm_time := 100

-- Prove the LCM of their lap times is 3600 seconds,
-- which is equivalent to 60 minutes.
theorem racers_meet_at_start_again :
  Nat.lcm (Nat.lcm (Nat.lcm RacingMagic_time ChargingBull_time) SwiftShadow_time) SpeedyStorm_time = 3600 ∧
  3600 / 60 = 60 := by
  sorry

end racers_meet_at_start_again_l2_2982


namespace product_of_first_four_consecutive_primes_l2_2823

theorem product_of_first_four_consecutive_primes : 
  (2 * 3 * 5 * 7) = 210 :=
by
  sorry

end product_of_first_four_consecutive_primes_l2_2823


namespace minimize_a_plus_b_l2_2931

theorem minimize_a_plus_b (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : 4 * a + b = 30) :
  a + b = 9 → (a, b) = (7, 2) := sorry

end minimize_a_plus_b_l2_2931


namespace min_value_of_polynomial_l2_2819

theorem min_value_of_polynomial :
  ∃ x : ℝ, ∀ y, y = (x - 16) * (x - 14) * (x + 14) * (x + 16) → y ≥ -900 :=
by
  sorry

end min_value_of_polynomial_l2_2819


namespace simplify_frac_l2_2046

theorem simplify_frac (b : ℤ) (hb : b = 2) : (15 * b^4) / (45 * b^3) = 2 / 3 :=
by {
  sorry
}

end simplify_frac_l2_2046


namespace set_intersections_l2_2565

open Set Nat

def I : Set ℕ := univ

def A : Set ℕ := { x | ∃ n, x = 3 * n ∧ ∃ k, n = 2 * k }

def B : Set ℕ := { y | ∃ m, y = m ∧ 24 % m = 0 }

theorem set_intersections :
  A ∩ B = {6, 12, 24} ∧ (I \ A) ∩ B = {1, 2, 3, 4, 8} :=
by
  sorry

end set_intersections_l2_2565


namespace count_divisible_neither_5_nor_7_below_500_l2_2882

def count_divisible_by (n k : ℕ) : ℕ := (n - 1) / k

def count_divisible_by_5_or_7_below (n : ℕ) : ℕ :=
  let count_5 := count_divisible_by n 5
  let count_7 := count_divisible_by n 7
  let count_35 := count_divisible_by n 35
  count_5 + count_7 - count_35

def count_divisible_neither_5_nor_7_below (n : ℕ) : ℕ :=
  n - 1 - count_divisible_by_5_or_7_below n

theorem count_divisible_neither_5_nor_7_below_500 : count_divisible_neither_5_nor_7_below 500 = 343 :=
by
  sorry

end count_divisible_neither_5_nor_7_below_500_l2_2882


namespace spencer_total_distance_l2_2409

def distances : ℝ := 0.3 + 0.1 + 0.4

theorem spencer_total_distance :
  distances = 0.8 :=
sorry

end spencer_total_distance_l2_2409


namespace fractions_problem_l2_2438

theorem fractions_problem (x y : ℚ) (hx : x = 2 / 3) (hy : y = 3 / 2) :
  (1 / 3) * x^5 * y^6 = 3 / 2 := by
  sorry

end fractions_problem_l2_2438


namespace complement_union_eq_l2_2460

variable (U : Set ℝ) (M N : Set ℝ)

noncomputable def complement_union (U M N : Set ℝ) : Set ℝ :=
  U \ (M ∪ N)

theorem complement_union_eq :
  U = Set.univ → 
  M = {x | |x| < 1} → 
  N = {y | ∃ x, y = 2^x} → 
  complement_union U M N = {x | x ≤ -1} :=
by
  intros hU hM hN
  unfold complement_union
  sorry

end complement_union_eq_l2_2460


namespace min_value_of_vector_sum_l2_2957

noncomputable def min_vector_sum_magnitude (P Q: (ℝ×ℝ)) : ℝ :=
  let x := P.1
  let y := P.2
  let a := Q.1
  let b := Q.2
  Real.sqrt ((x + a)^2 + (y + b)^2)

theorem min_value_of_vector_sum :
  ∃ P Q, 
  (P.1 - 2)^2 + (P.2 - 2)^2 = 1 ∧ 
  Q.1 + Q.2 = 1 ∧ 
  min_vector_sum_magnitude P Q = (5 * Real.sqrt 2 - 2) / 2 :=
by
  sorry

end min_value_of_vector_sum_l2_2957


namespace least_y_value_l2_2531

theorem least_y_value (y : ℝ) : 2 * y ^ 2 + 7 * y + 3 = 5 → y ≥ -2 :=
by
  intro h
  sorry

end least_y_value_l2_2531


namespace sin_sum_of_roots_l2_2059

theorem sin_sum_of_roots (x1 x2 m : ℝ) (hx1 : 0 ≤ x1 ∧ x1 ≤ π) (hx2 : 0 ≤ x2 ∧ x2 ≤ π)
    (hroot1 : 2 * Real.sin x1 + Real.cos x1 = m) (hroot2 : 2 * Real.sin x2 + Real.cos x2 = m) :
    Real.sin (x1 + x2) = 4 / 5 := 
sorry

end sin_sum_of_roots_l2_2059


namespace find_digit_A_l2_2357

theorem find_digit_A (A M C : ℕ) (h1 : A < 10) (h2 : M < 10) (h3 : C < 10) (h4 : (100 * A + 10 * M + C) * (A + M + C) = 2008) : 
  A = 2 :=
sorry

end find_digit_A_l2_2357


namespace evaluate_expression_l2_2166

theorem evaluate_expression (a b : ℤ) (h_a : a = 1) (h_b : b = -2) : 
  2 * (a^2 - 3 * a * b + 1) - (2 * a^2 - b^2) + 5 * a * b = 8 :=
by
  sorry

end evaluate_expression_l2_2166


namespace abc_value_l2_2725

theorem abc_value 
  (a b c : ℝ)
  (a_pos : 0 < a) 
  (b_pos : 0 < b) 
  (c_pos : 0 < c) 
  (hab : a * b = 24) 
  (hac : a * c = 40) 
  (hbc : b * c = 60) : 
  a * b * c = 240 := 
by sorry

end abc_value_l2_2725


namespace sum_of_squares_of_roots_l2_2887

theorem sum_of_squares_of_roots :
  ∀ r1 r2 : ℝ, (r1 + r2 = 14) ∧ (r1 * r2 = 8) → (r1^2 + r2^2 = 180) := by
  sorry

end sum_of_squares_of_roots_l2_2887


namespace diagonal_of_rectangle_l2_2386

noncomputable def L : ℝ := 40 * Real.sqrt 3
noncomputable def W : ℝ := 30 * Real.sqrt 3
noncomputable def d : ℝ := Real.sqrt (L^2 + W^2)

theorem diagonal_of_rectangle :
  d = 50 * Real.sqrt 3 :=
by sorry

end diagonal_of_rectangle_l2_2386


namespace xy_product_range_l2_2124

theorem xy_product_range (x y : ℝ) (h : x^2 * y^2 + x^2 - 10 * x * y - 8 * x + 16 = 0) :
  0 ≤ x * y ∧ x * y ≤ 10 := 
sorry

end xy_product_range_l2_2124


namespace correct_system_of_equations_l2_2673

noncomputable def system_of_equations (x y : ℝ) : Prop :=
x + y = 150 ∧ 3 * x + (1 / 3) * y = 210

theorem correct_system_of_equations : ∃ x y : ℝ, system_of_equations x y :=
sorry

end correct_system_of_equations_l2_2673


namespace symmetric_y_axis_l2_2778

-- Definition of a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Definition of point symmetry with respect to the y-axis
def symmetric_about_y_axis (M : Point3D) : Point3D := 
  { x := -M.x, y := M.y, z := -M.z }

-- Theorem statement: proving the symmetry
theorem symmetric_y_axis (M : Point3D) : 
  symmetric_about_y_axis M = { x := -M.x, y := M.y, z := -M.z } := by
  sorry  -- Proof is left out as per instruction.

end symmetric_y_axis_l2_2778


namespace simplify_expression_l2_2358

theorem simplify_expression (a : ℝ) : 2 * (a + 2) - 2 * a = 4 :=
by
  sorry

end simplify_expression_l2_2358


namespace determine_constants_l2_2421

theorem determine_constants (k a b : ℝ) :
  (3*x^2 - 4*x + 5)*(5*x^2 + k*x + 8) = 15*x^4 - 47*x^3 + a*x^2 - b*x + 40 →
  k = -9 ∧ a = 15 ∧ b = 72 :=
by
  sorry

end determine_constants_l2_2421


namespace number_of_girls_in_class_l2_2322

theorem number_of_girls_in_class (B S G : ℕ)
  (h1 : 3 * B = 4 * 18)  -- 3/4 * B = 18
  (h2 : 2 * S = 3 * B)  -- 2/3 * S = B
  (h3 : G = S - B) : G = 12 :=
by
  sorry

end number_of_girls_in_class_l2_2322


namespace only_selected_A_is_20_l2_2524

def cardinality_A (x : ℕ) : ℕ := x
def cardinality_B (x : ℕ) : ℕ := x + 8
def cardinality_union (x : ℕ) : ℕ := 54
def cardinality_intersection (x : ℕ) : ℕ := 6

theorem only_selected_A_is_20 (x : ℕ) (h_total : cardinality_union x = 54) 
  (h_inter : cardinality_intersection x = 6) (h_B : cardinality_B x = x + 8) :
  cardinality_A x - cardinality_intersection x = 20 :=
by
  sorry

end only_selected_A_is_20_l2_2524


namespace find_x_for_sin_cos_l2_2863

theorem find_x_for_sin_cos (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 2 * Real.pi) (h3 : Real.sin x + Real.cos x = Real.sqrt 2) : x = Real.pi / 4 :=
sorry

end find_x_for_sin_cos_l2_2863


namespace geometric_sequence_product_l2_2562

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∃ r : ℝ, ∀ n, a (n + 1) = a n * r

noncomputable def quadratic_roots (a1 a10 : ℝ) : Prop :=
3 * a1^2 - 2 * a1 - 6 = 0 ∧ 3 * a10^2 - 2 * a10 - 6 = 0

theorem geometric_sequence_product {a : ℕ → ℝ}
  (h_geom : geometric_sequence a)
  (h_roots : quadratic_roots (a 1) (a 10)) :
  a 4 * a 7 = -2 :=
sorry

end geometric_sequence_product_l2_2562


namespace find_counterfeit_coin_l2_2065

-- Define the context of the problem
variables (coins : Fin 6 → ℝ) -- six coins represented as a function from Fin 6 to their weights
          (is_counterfeit : Fin 6 → Prop) -- a predicate indicating if the coin is counterfeit
          (real_weight : ℝ) -- the unknown weight of a real coin

-- Existence assertion for the counterfeit coin
axiom exists_counterfeit : ∃ x, is_counterfeit x

-- Define the total weights of coins 1&2 and 3&4
def weight_1_2 := coins 0 + coins 1
def weight_3_4 := coins 2 + coins 3

-- Statement of the problem
theorem find_counterfeit_coin :
  (weight_1_2 = weight_3_4 → (is_counterfeit 4 ∨ is_counterfeit 5)) ∧ 
  (weight_1_2 ≠ weight_3_4 → (is_counterfeit 0 ∨ is_counterfeit 1 ∨ is_counterfeit 2 ∨ is_counterfeit 3)) :=
sorry

end find_counterfeit_coin_l2_2065


namespace arithmetic_sequence_S2008_l2_2318

theorem arithmetic_sequence_S2008 (a1 : ℤ) (S : ℕ → ℤ) (d : ℤ)
  (h1 : a1 = -2008)
  (h2 : ∀ n, S n = n * a1 + n * (n - 1) / 2 * d)
  (h3 : (S 12 / 12) - (S 10 / 10) = 2) :
  S 2008 = -2008 := 
sorry

end arithmetic_sequence_S2008_l2_2318


namespace cats_on_edges_l2_2907

variables {W1 W2 B1 B2 : ℕ}  -- representing positions of cats on a line

def distance_from_white_to_black_sum_1 (a1 a2 : ℕ) : Prop := a1 + a2 = 4
def distance_from_white_to_black_sum_2 (b1 b2 : ℕ) : Prop := b1 + b2 = 8
def distance_from_black_to_white_sum_1 (b1 a1 : ℕ) : Prop := b1 + a1 = 9
def distance_from_black_to_white_sum_2 (b2 a2 : ℕ) : Prop := b2 + a2 = 3

theorem cats_on_edges
  (a1 a2 b1 b2 : ℕ)
  (h1 : distance_from_white_to_black_sum_1 a1 a2)
  (h2 : distance_from_white_to_black_sum_2 b1 b2)
  (h3 : distance_from_black_to_white_sum_1 b1 a1)
  (h4 : distance_from_black_to_white_sum_2 b2 a2) :
  (a1 = 2) ∧ (a2 = 2) ∧ (b1 = 7) ∧ (b2 = 1) ∧ (W1 = min W1 W2) ∧ (B2 = max B1 B2) :=
sorry

end cats_on_edges_l2_2907


namespace laura_house_distance_l2_2916

-- Definitions based on conditions
def x : Real := 10  -- Distance from Laura's house to her school in miles

def distance_to_school_per_day := 2 * x
def school_days_per_week := 5
def distance_to_school_per_week := school_days_per_week * distance_to_school_per_day

def distance_to_supermarket := x + 10
def supermarket_trips_per_week := 2
def distance_to_supermarket_per_trip := 2 * distance_to_supermarket
def distance_to_supermarket_per_week := supermarket_trips_per_week * distance_to_supermarket_per_trip

def total_distance_per_week := 220

-- The proof statement
theorem laura_house_distance :
  distance_to_school_per_week + distance_to_supermarket_per_week = total_distance_per_week ∧ x = 10 := by
  sorry

end laura_house_distance_l2_2916


namespace middle_group_frequency_l2_2627

theorem middle_group_frequency (f : ℕ) (A : ℕ) (h_total : A + f = 100) (h_middle : f = A) : f = 50 :=
by
  sorry

end middle_group_frequency_l2_2627


namespace books_sold_to_used_bookstore_l2_2865

-- Conditions
def initial_books := 72
def books_from_club := 1 * 12
def books_from_bookstore := 5
def books_from_yardsales := 2
def books_from_daughter := 1
def books_from_mother := 4
def books_donated := 12
def books_end_of_year := 81

-- Proof problem
theorem books_sold_to_used_bookstore :
  initial_books
  + books_from_club
  + books_from_bookstore
  + books_from_yardsales
  + books_from_daughter
  + books_from_mother
  - books_donated
  - books_end_of_year
  = 3 := by
  -- calculation omitted
  sorry

end books_sold_to_used_bookstore_l2_2865


namespace find_a_l2_2801

theorem find_a (S_n : ℕ → ℝ) (a_n : ℕ → ℝ) (a : ℝ) (h1 : ∀ n, S_n n = 3^(n+1) + a)
  (h2 : ∀ n, a_n (n+1) = S_n (n+1) - S_n n)
  (h3 : ∀ n m k, a_n m * a_n k = (a_n n)^2 → n = m + k) : 
  a = -3 := 
sorry

end find_a_l2_2801


namespace curve_is_parabola_l2_2553

-- Define the condition: the curve is defined by the given polar equation
def polar_eq (r θ : ℝ) : Prop :=
  r = 1 / (1 - Real.sin θ)

-- The main theorem statement: Prove that the curve defined by the equation is a parabola
theorem curve_is_parabola (r θ : ℝ) (h : polar_eq r θ) : ∃ x y : ℝ, x = 1 + 2 * y :=
sorry

end curve_is_parabola_l2_2553


namespace original_cost_of_tomatoes_correct_l2_2642

noncomputable def original_cost_of_tomatoes := 
  let original_order := 25
  let new_tomatoes := 2.20
  let new_lettuce := 1.75
  let old_lettuce := 1.00
  let new_celery := 2.00
  let old_celery := 1.96
  let delivery_tip := 8
  let new_total_bill := 35
  let new_groceries := new_total_bill - delivery_tip
  let increase_in_cost := (new_lettuce - old_lettuce) + (new_celery - old_celery)
  let difference_due_to_substitutions := new_groceries - original_order
  let x := new_tomatoes + (difference_due_to_substitutions - increase_in_cost)
  x

theorem original_cost_of_tomatoes_correct :
  original_cost_of_tomatoes = 3.41 := by
  sorry

end original_cost_of_tomatoes_correct_l2_2642


namespace fraction_inequality_fraction_inequality_equality_case_l2_2724

variables {α β a b : ℝ}

theorem fraction_inequality 
  (h_alpha_beta_pos : 0 < α ∧ 0 < β)
  (h_bounds_a : α ≤ a ∧ a ≤ β)
  (h_bounds_b : α ≤ b ∧ b ≤ β) :
  (b / a + a / b) ≤ (β / α + α / β) :=
sorry

-- Additional equality statement
theorem fraction_inequality_equality_case
  (h_alpha_beta_pos : 0 < α ∧ 0 < β)
  (h_bounds_a : α ≤ a ∧ a ≤ β)
  (h_bounds_b : α ≤ b ∧ b ≤ β) :
  (b / a + a / b = β / α + α / β) ↔ (a = α ∧ b = β ∨ a = β ∧ b = α) :=
sorry

end fraction_inequality_fraction_inequality_equality_case_l2_2724


namespace quadratic_inequality_solution_set_l2_2404

variables {a b c : ℝ}

theorem quadratic_inequality_solution_set (h : ∀ x, x > -1 ∧ x < 2 → ax^2 - bx + c > 0) :
  a + b + c = 0 :=
sorry

end quadratic_inequality_solution_set_l2_2404


namespace common_ratio_of_geometric_series_l2_2915

theorem common_ratio_of_geometric_series (a S r : ℝ) (h₁ : a = 400) (h₂ : S = 2500) :
  S = a / (1 - r) → r = 21 / 25 :=
by
  intros h₃
  rw [h₁, h₂] at h₃
  sorry

end common_ratio_of_geometric_series_l2_2915


namespace situps_ratio_l2_2845

theorem situps_ratio (ken_situps : ℕ) (nathan_situps : ℕ) (bob_situps : ℕ) :
  ken_situps = 20 →
  nathan_situps = 2 * ken_situps →
  bob_situps = ken_situps + 10 →
  (bob_situps : ℚ) / (ken_situps + nathan_situps : ℚ) = 1 / 2 :=
by
  sorry

end situps_ratio_l2_2845


namespace total_cost_is_correct_l2_2141

def cost_per_pound : ℝ := 0.45
def weight_sugar : ℝ := 40
def weight_flour : ℝ := 16

theorem total_cost_is_correct :
  weight_sugar * cost_per_pound + weight_flour * cost_per_pound = 25.20 :=
by
  sorry

end total_cost_is_correct_l2_2141


namespace coefficient_A_l2_2744

-- Definitions from the conditions
variable (A c₀ d : ℝ)
variable (h₁ : c₀ = 47)
variable (h₂ : A * c₀ + (d - 12) ^ 2 = 235)

-- The theorem to prove
theorem coefficient_A (h₁ : c₀ = 47) (h₂ : A * c₀ + (d - 12) ^ 2 = 235) : A = 5 :=
by sorry

end coefficient_A_l2_2744


namespace room_volume_correct_l2_2644

variable (Length Width Height : ℕ) (Volume : ℕ)

-- Define the dimensions of the room
def roomLength := 100
def roomWidth := 10
def roomHeight := 10

-- Define the volume function
def roomVolume (l w h : ℕ) : ℕ := l * w * h

-- Theorem to prove the volume of the room
theorem room_volume_correct : roomVolume roomLength roomWidth roomHeight = 10000 := 
by
  -- roomVolume 100 10 10 = 10000
  sorry

end room_volume_correct_l2_2644


namespace value_of_b_l2_2821

theorem value_of_b (f : ℝ → ℝ) (a b : ℝ) (h1 : ∀ x ≠ 0, f x = -1 / x) (h2 : f a = -1 / 3) (h3 : f (a * b) = 1 / 6) : b = -2 :=
sorry

end value_of_b_l2_2821


namespace work_together_days_l2_2505

theorem work_together_days (A B : ℝ) (h1 : A = 1/2 * B) (h2 : B = 1/48) :
  1 / (A + B) = 32 :=
by
  sorry

end work_together_days_l2_2505


namespace roots_quadratic_expression_l2_2877

theorem roots_quadratic_expression :
  ∀ (a b : ℝ), (a^2 - 5 * a + 6 = 0) ∧ (b^2 - 5 * b + 6 = 0) → 
  a^3 + a^4 * b^2 + a^2 * b^4 + b^3 + a * b * (a + b) = 533 :=
by
  intros a b h
  sorry

end roots_quadratic_expression_l2_2877


namespace pens_sold_during_promotion_l2_2999

theorem pens_sold_during_promotion (x y n : ℕ) 
  (h_profit: 12 * x + 7 * y = 2011)
  (h_n: n = 2 * x + y) : 
  n = 335 := by
  sorry

end pens_sold_during_promotion_l2_2999


namespace Tonya_buys_3_lego_sets_l2_2935

-- Definitions based on conditions
def num_sisters : Nat := 2
def num_dolls : Nat := 4
def price_per_doll : Nat := 15
def price_per_lego_set : Nat := 20

-- The amount of money spent on each sister should be the same
def amount_spent_on_younger_sister := num_dolls * price_per_doll
def amount_spent_on_older_sister := (amount_spent_on_younger_sister / price_per_lego_set)

-- Proof statement
theorem Tonya_buys_3_lego_sets : amount_spent_on_older_sister = 3 :=
by
  sorry

end Tonya_buys_3_lego_sets_l2_2935


namespace beka_flew_more_l2_2242

def bekaMiles := 873
def jacksonMiles := 563

theorem beka_flew_more : bekaMiles - jacksonMiles = 310 := by
  -- proof here
  sorry

end beka_flew_more_l2_2242


namespace find_cost_per_sq_foot_l2_2492

noncomputable def monthly_rent := 2800 / 2
noncomputable def old_annual_rent (C : ℝ) := 750 * C * 12
noncomputable def new_annual_rent := monthly_rent * 12
noncomputable def annual_savings := old_annual_rent - new_annual_rent

theorem find_cost_per_sq_foot (C : ℝ):
    (750 * C * 12 - 2800 / 2 * 12 = 1200) ↔ (C = 2) :=
sorry

end find_cost_per_sq_foot_l2_2492


namespace leaves_fall_total_l2_2803

theorem leaves_fall_total : 
  let planned_cherry_trees := 7 
  let planned_maple_trees := 5 
  let actual_cherry_trees := 2 * planned_cherry_trees
  let actual_maple_trees := 3 * planned_maple_trees
  let leaves_per_cherry_tree := 100
  let leaves_per_maple_tree := 150
  actual_cherry_trees * leaves_per_cherry_tree + actual_maple_trees * leaves_per_maple_tree = 3650 :=
by
  let planned_cherry_trees := 7 
  let planned_maple_trees := 5 
  let actual_cherry_trees := 2 * planned_cherry_trees
  let actual_maple_trees := 3 * planned_maple_trees
  let leaves_per_cherry_tree := 100
  let leaves_per_maple_tree := 150
  sorry

end leaves_fall_total_l2_2803


namespace min_value_fraction_l2_2754

variable (a b : ℝ)
variable (h1 : 2 * a - 2 * b + 2 = 0) -- This corresponds to a + b = 1 based on the given center (-1, 2)
variable (ha : a > 0)
variable (hb : b > 0)

theorem min_value_fraction (h1 : a + b = 1) (ha : a > 0) (hb : b > 0) : 
  (4 / a) + (1 / b) ≥ 9 :=
  sorry

end min_value_fraction_l2_2754


namespace pumpkins_eaten_l2_2817

-- Definitions for the conditions
def originalPumpkins : ℕ := 43
def leftPumpkins : ℕ := 20

-- Theorem statement
theorem pumpkins_eaten : originalPumpkins - leftPumpkins = 23 :=
  by
    -- Proof steps are omitted
    sorry

end pumpkins_eaten_l2_2817


namespace intersection_A_B_l2_2157

-- Definitions for sets A and B based on the problem conditions
def A : Set ℝ := { x | x^2 - 2 * x - 3 < 0 }
def B : Set ℝ := { x | ∃ y : ℝ, y = Real.log (2 - x) }

-- Proof problem statement
theorem intersection_A_B : (A ∩ B) = { x : ℝ | -1 < x ∧ x < 2 } :=
by
  sorry

end intersection_A_B_l2_2157


namespace negation_proposition_l2_2381

theorem negation_proposition {x : ℝ} : ¬ (x^2 - x + 3 > 0) ↔ x^2 - x + 3 ≤ 0 := sorry

end negation_proposition_l2_2381


namespace students_per_row_first_scenario_l2_2507

theorem students_per_row_first_scenario 
  (S R x : ℕ)
  (h1 : S = x * R + 6)
  (h2 : S = 12 * (R - 3))
  (h3 : S = 6 * R) :
  x = 5 :=
by
  sorry

end students_per_row_first_scenario_l2_2507


namespace zac_strawberries_l2_2660

theorem zac_strawberries (J M Z : ℕ) 
  (h1 : J + M + Z = 550) 
  (h2 : J + M = 350) 
  (h3 : M + Z = 250) : 
  Z = 200 :=
sorry

end zac_strawberries_l2_2660


namespace value_of_A_l2_2446

theorem value_of_A (A C : ℤ) (h₁ : 2 * A - C + 4 = 26) (h₂ : C = 6) : A = 14 :=
by sorry

end value_of_A_l2_2446


namespace sequences_converge_and_find_limits_l2_2656

theorem sequences_converge_and_find_limits (x y : ℕ → ℝ)
  (h1 : x 1 = 1)
  (h2 : y 1 = Real.sqrt 3)
  (h3 : ∀ n : ℕ, x (n + 1) * y (n + 1) = x n)
  (h4 : ∀ n : ℕ, x (n + 1)^2 + y n = 2) :
  ∃ (Lx Ly : ℝ), (∀ ε : ℝ, ε > 0 → ∃ N : ℕ, ∀ n ≥ N, |x n - Lx| < ε) ∧ 
                  (∀ ε : ℝ, ε > 0 → ∃ N : ℕ, ∀ n ≥ N, |y n - Ly| < ε) ∧ 
                  Lx = 0 ∧ 
                  Ly = 2 := 
sorry

end sequences_converge_and_find_limits_l2_2656


namespace curve_is_hyperbola_l2_2498

theorem curve_is_hyperbola (u : ℝ) (x y : ℝ) 
  (h1 : x = Real.cos u ^ 2)
  (h2 : y = Real.sin u ^ 4) : 
  ∃ (a b : ℝ), a ≠ 0 ∧  b ≠ 0 ∧ x / a ^ 2 - y / b ^ 2 = 1 := 
sorry

end curve_is_hyperbola_l2_2498


namespace strokes_over_par_l2_2408

theorem strokes_over_par (n s p : ℕ) (t : ℕ) (par : ℕ )
  (h1 : n = 9)
  (h2 : s = 4)
  (h3 : p = 3)
  (h4: t = n * s)
  (h5: par = n * p) :
  t - par = 9 :=
by 
  sorry

end strokes_over_par_l2_2408


namespace mitchell_more_than_antonio_l2_2461

-- Definitions based on conditions
def mitchell_pencils : ℕ := 30
def total_pencils : ℕ := 54

-- Definition of the main question
def antonio_pencils : ℕ := total_pencils - mitchell_pencils

-- The theorem to be proved
theorem mitchell_more_than_antonio : mitchell_pencils - antonio_pencils = 6 :=
by
-- Proof is omitted
sorry

end mitchell_more_than_antonio_l2_2461


namespace product_between_21st_and_24th_multiple_of_3_l2_2030

theorem product_between_21st_and_24th_multiple_of_3 : 
  (66 * 69 = 4554) :=
by
  sorry

end product_between_21st_and_24th_multiple_of_3_l2_2030


namespace number_of_fours_is_even_l2_2952

theorem number_of_fours_is_even 
  (x y z : ℕ) 
  (h1 : x + y + z = 80) 
  (h2 : 3 * x + 4 * y + 5 * z = 276) : 
  Even y :=
by
  sorry

end number_of_fours_is_even_l2_2952


namespace integer_values_sides_triangle_l2_2894

theorem integer_values_sides_triangle (x : ℝ) (hx_pos : x > 0) (hx1 : x + 15 > 40) (hx2 : x + 40 > 15) (hx3 : 15 + 40 > x) : 
    (∃ (n : ℤ), ∃ (hn : 0 < n) (hn1 : (n : ℝ) = x) (hn2 : 26 ≤ n) (hn3 : n ≤ 54), 
    ∀ (y : ℤ), (26 ≤ y ∧ y ≤ 54) → (∃ (m : ℤ), y = 26 + m ∧ m < 29 ∧ m ≥ 0)) := 
sorry

end integer_values_sides_triangle_l2_2894


namespace david_tips_l2_2612

noncomputable def avg_tips_resort (tips_other_months : ℝ) (months : ℕ) := tips_other_months / months

theorem david_tips 
  (tips_march_to_july_september : ℝ)
  (tips_august_resort : ℝ)
  (total_tips_delivery_driver : ℝ)
  (total_tips_resort : ℝ)
  (total_tips : ℝ)
  (fraction_august : ℝ)
  (avg_tips := avg_tips_resort tips_march_to_july_september 6):
  tips_august_resort = 4 * avg_tips →
  total_tips_delivery_driver = 2 * avg_tips →
  total_tips_resort = tips_march_to_july_september + tips_august_resort →
  total_tips = total_tips_resort + total_tips_delivery_driver →
  fraction_august = tips_august_resort / total_tips →
  fraction_august = 1 / 2 :=
by
  sorry

end david_tips_l2_2612


namespace solve_for_x_l2_2678

theorem solve_for_x : (∃ x : ℝ, (1/2 - 1/3 = 1/x)) ↔ (x = 6) := sorry

end solve_for_x_l2_2678


namespace intersection_of_A_and_B_l2_2382

open Set

-- Definitions of sets A and B as per conditions in the problem
def A := {x : ℝ | -1 < x ∧ x < 2}
def B := {x : ℝ | -3 < x ∧ x ≤ 1}

-- The proof statement that A ∩ B = {x | -1 < x ∧ x ≤ 1}
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -1 < x ∧ x ≤ 1} := by
  sorry

end intersection_of_A_and_B_l2_2382


namespace sum_of_real_y_values_l2_2649

theorem sum_of_real_y_values :
  (∀ (x y : ℝ), x^2 + x^2 * y^2 + x^2 * y^4 = 525 ∧ x + x * y + x * y^2 = 35 → y = 1 / 2 ∨ y = 2) →
    (1 / 2 + 2 = 5 / 2) :=
by
  intro h
  have := h (1 / 2)
  have := h 2
  sorry  -- Proof steps showing 1/2 and 2 are the solutions, leading to the sum 5/2

end sum_of_real_y_values_l2_2649


namespace not_invited_students_l2_2170

-- Definition of the problem conditions
def students := 15
def direct_friends_of_mia := 4
def unique_friends_of_each_friend := 2

-- Problem statement
theorem not_invited_students : (students - (1 + direct_friends_of_mia + direct_friends_of_mia * unique_friends_of_each_friend) = 2) :=
by
  sorry

end not_invited_students_l2_2170


namespace remainder_when_98_mul_102_divided_by_11_l2_2712

theorem remainder_when_98_mul_102_divided_by_11 :
  (98 * 102) % 11 = 1 :=
by
  sorry

end remainder_when_98_mul_102_divided_by_11_l2_2712


namespace condo_cats_l2_2767

theorem condo_cats (x y : ℕ) (h1 : 2 * x + y = 29) : 6 * x + 3 * y = 87 := by
  sorry

end condo_cats_l2_2767


namespace problem_conditions_imply_options_l2_2541

theorem problem_conditions_imply_options (a b : ℝ) 
  (h1 : a + 1 > b) 
  (h2 : b > 2 / a) 
  (h3 : 2 / a > 0) : 
  (a = 2 ∧ a + 1 > 2 / a ∧ b > 2 / 2) ∨
  (a = 1 → a + 1 ≤ 2 / a) ∨
  (b = 1 → ∃ a, a > 1 ∧ a + 1 > 1 ∧ 1 > 2 / a) ∨
  (a * b = 1 → ab ≤ 2) := 
sorry

end problem_conditions_imply_options_l2_2541


namespace area_of_right_triangle_l2_2605

variables {x y : ℝ} (r : ℝ)

theorem area_of_right_triangle (hx : ∀ r, r * (x + y + r) = x * y) :
  1 / 2 * (x + r) * (y + r) = x * y :=
by sorry

end area_of_right_triangle_l2_2605


namespace squirrel_nuts_l2_2037

theorem squirrel_nuts :
  ∃ (a b c d : ℕ), 103 ≤ a ∧ 103 ≤ b ∧ 103 ≤ c ∧ 103 ≤ d ∧
                   a ≥ b ∧ a ≥ c ∧ a ≥ d ∧
                   a + b + c + d = 2020 ∧
                   b + c = 1277 ∧
                   a = 640 :=
by {
  -- proof goes here
  sorry
}

end squirrel_nuts_l2_2037


namespace silver_coins_change_l2_2746

-- Define the conditions
def condition1 : ℕ × ℕ := (20, 4) -- (20 silver coins, 4 gold coins change)
def condition2 : ℕ × ℕ := (15, 1) -- (15 silver coins, 1 gold coin change)
def cost_of_cloak_in_gold_coins : ℕ := 14

-- Define the theorem to be proven
theorem silver_coins_change (s1 g1 s2 g2 cloak_g : ℕ) (h1 : (s1, g1) = condition1) (h2 : (s2, g2) = condition2) :
  ∃ silver : ℕ, (silver = 10) :=
by {
  sorry
}

end silver_coins_change_l2_2746


namespace decreasing_intervals_tangent_line_eq_l2_2230

-- Define the function f and its derivative.
def f (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + 1
def f' (x : ℝ) : ℝ := -3*x^2 + 6*x + 9

-- Part 1: Prove intervals of monotonic decreasing.
theorem decreasing_intervals :
  (∀ x, f' x < 0 → x < -1 ∨ x > 3) := 
sorry

-- Part 2: Prove the tangent line equation.
theorem tangent_line_eq :
  15 * (-2) + (-13) + 27 = 0 :=
sorry

end decreasing_intervals_tangent_line_eq_l2_2230


namespace James_balloons_correct_l2_2854

def Amy_balloons : ℕ := 101
def diff_balloons : ℕ := 131
def James_balloons (a : ℕ) (d : ℕ) : ℕ := a + d

theorem James_balloons_correct : James_balloons Amy_balloons diff_balloons = 232 :=
by
  sorry

end James_balloons_correct_l2_2854


namespace fourth_house_number_l2_2490

theorem fourth_house_number (sum: ℕ) (k x: ℕ) (h1: sum = 78) (h2: k ≥ 4)
  (h3: (k+1) * (x + k) = 78) : x + 6 = 14 :=
by
  sorry

end fourth_house_number_l2_2490


namespace trainB_reaches_in_3_hours_l2_2295

variable (trainA_speed trainB_speed : ℕ) (x t : ℝ)

-- Given conditions
axiom h1 : trainA_speed = 70
axiom h2 : trainB_speed = 105
axiom h3 : ∀ x t, 70 * x + 70 * 9 = 105 * x + 105 * t

-- Prove that train B takes 3 hours to reach destination after meeting
theorem trainB_reaches_in_3_hours : t = 3 :=
by
  sorry

end trainB_reaches_in_3_hours_l2_2295


namespace wedding_reception_friends_l2_2195

theorem wedding_reception_friends (total_guests bride_couples groom_couples bride_coworkers groom_coworkers bride_relatives groom_relatives: ℕ)
  (h1: total_guests = 400)
  (h2: bride_couples = 40) 
  (h3: groom_couples = 40)
  (h4: bride_coworkers = 10) 
  (h5: groom_coworkers = 10)
  (h6: bride_relatives = 20)
  (h7: groom_relatives = 20)
  : (total_guests - ((bride_couples + groom_couples) * 2 + (bride_coworkers + groom_coworkers) + (bride_relatives + groom_relatives))) = 180 := 
by 
  sorry

end wedding_reception_friends_l2_2195


namespace find_c_l2_2347

theorem find_c (a c : ℤ) (h1 : 3 * a + 2 = 2) (h2 : c - a = 3) : c = 3 := by
  sorry

end find_c_l2_2347


namespace unique_solution_l2_2399

noncomputable def uniquely_solvable (a : ℝ) : Prop :=
  ∀ x : ℝ, a > 0 ∧ a ≠ 1 → ∃! x, a^x = (Real.log x / Real.log (1/4))

theorem unique_solution (a : ℝ) : a > 0 ∧ a ≠ 1 → uniquely_solvable a :=
by sorry

end unique_solution_l2_2399


namespace number_of_students_preferring_dogs_l2_2631

-- Define the conditions
def total_students : ℕ := 30
def dogs_video_games_chocolate_percentage : ℚ := 0.50
def dogs_movies_vanilla_percentage : ℚ := 0.10
def cats_video_games_chocolate_percentage : ℚ := 0.20
def cats_movies_vanilla_percentage : ℚ := 0.15

-- Define the target statement to prove
theorem number_of_students_preferring_dogs : 
  (dogs_video_games_chocolate_percentage + dogs_movies_vanilla_percentage) * total_students = 18 :=
by
  sorry

end number_of_students_preferring_dogs_l2_2631


namespace avg_abc_l2_2884

variable (A B C : ℕ)

-- Conditions
def avg_ac : Prop := (A + C) / 2 = 29
def age_b : Prop := B = 26

-- Theorem stating the average age of a, b, and c
theorem avg_abc (h1 : avg_ac A C) (h2 : age_b B) : (A + B + C) / 3 = 28 := by
  sorry

end avg_abc_l2_2884


namespace product_is_2008th_power_l2_2844

theorem product_is_2008th_power (a b c : ℕ) (h1 : a = (b + c) / 2) (h2 : b ≠ c) (h3 : c ≠ a) (h4 : a ≠ b) :
  ∃ k : ℕ, (a * b * c) = k^2008 :=
by
  sorry

end product_is_2008th_power_l2_2844


namespace b_can_finish_work_in_15_days_l2_2955

theorem b_can_finish_work_in_15_days (W : ℕ) (r_A : ℕ) (r_B : ℕ) (h1 : r_A = W / 21) (h2 : 10 * r_B + 7 * r_A / 21 = W) : r_B = W / 15 :=
by sorry

end b_can_finish_work_in_15_days_l2_2955


namespace find_percentage_of_other_investment_l2_2579

theorem find_percentage_of_other_investment
  (total_investment : ℝ) (specific_investment : ℝ) (specific_rate : ℝ) (total_interest : ℝ) 
  (other_investment : ℝ) (other_interest : ℝ) (P : ℝ) :
  total_investment = 17000 ∧
  specific_investment = 12000 ∧
  specific_rate = 0.04 ∧
  total_interest = 1380 ∧
  other_investment = total_investment - specific_investment ∧
  other_interest = total_interest - specific_rate * specific_investment ∧ 
  other_interest = (P / 100) * other_investment
  → P = 18 :=
by
  intros
  sorry

end find_percentage_of_other_investment_l2_2579


namespace consecutive_integers_avg_l2_2255

theorem consecutive_integers_avg (n x : ℤ) (h_avg : (2*x + n - 1 : ℝ)/2 = 20.5) (h_10th : x + 9 = 25) :
  n = 10 :=
by
  sorry

end consecutive_integers_avg_l2_2255


namespace min_value_reciprocal_sum_l2_2850

theorem min_value_reciprocal_sum 
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_sum : a + b = 2) : 
  ∃ x, x = 2 ∧ (∀ y, y = (1 / a) + (1 / b) → x ≤ y) := 
sorry

end min_value_reciprocal_sum_l2_2850


namespace maximize_revenue_l2_2474

-- Define the conditions
def price (p : ℝ) := p ≤ 30
def toys_sold (p : ℝ) : ℝ := 150 - 4 * p
def revenue (p : ℝ) := p * (toys_sold p)

-- State the theorem to solve the problem
theorem maximize_revenue : ∃ p : ℝ, price p ∧ 
  (∀ q : ℝ, price q → revenue q ≤ revenue p) ∧ p = 18.75 :=
by {
  sorry
}

end maximize_revenue_l2_2474


namespace fraction_in_pairing_l2_2550

open Function

theorem fraction_in_pairing (s t : ℕ) (h : (t : ℚ) / 4 = s / 3) : 
  ((t / 4 : ℚ) + (s / 3)) / (t + s) = 2 / 7 :=
by sorry

end fraction_in_pairing_l2_2550


namespace trigonometric_bound_l2_2116

open Real

theorem trigonometric_bound (x y : ℝ) : 
  -1/2 ≤ (x + y) * (1 - x * y) / ((1 + x^2) * (1 + y^2)) ∧ 
  (x + y) * (1 - x * y) / ((1 + x^2) * (1 + y^2)) ≤ 1/2 :=
by 
  sorry

end trigonometric_bound_l2_2116


namespace smallest_b_for_factorization_l2_2977

theorem smallest_b_for_factorization :
  ∃ b : ℕ, (∀ r s : ℤ, r * s = 2016 → r + s = b) ∧ b = 90 :=
sorry

end smallest_b_for_factorization_l2_2977


namespace line_a_minus_b_l2_2704

theorem line_a_minus_b (a b : ℝ)
  (h1 : (2 : ℝ) = a * (3 : ℝ) + b)
  (h2 : (26 : ℝ) = a * (7 : ℝ) + b) :
  a - b = 22 :=
by
  sorry

end line_a_minus_b_l2_2704


namespace triangle_with_angle_ratio_is_right_triangle_l2_2780

theorem triangle_with_angle_ratio_is_right_triangle (x : ℝ) (h1 : 1 * x + 2 * x + 3 * x = 180) : 
  ∃ A B C : ℝ, A = x ∧ B = 2 * x ∧ C = 3 * x ∧ (A = 90 ∨ B = 90 ∨ C = 90) := 
by
  sorry

end triangle_with_angle_ratio_is_right_triangle_l2_2780


namespace application_methods_l2_2139

variables (students : Fin 6) (colleges : Fin 3)

def total_applications_without_restriction : ℕ := 3^6
def applications_missing_one_college : ℕ := 2^6
def overcounted_applications_missing_two_college : ℕ := 1

theorem application_methods (h1 : total_applications_without_restriction = 729)
    (h2 : applications_missing_one_college = 64)
    (h3 : overcounted_applications_missing_two_college = 1) :
    ∀ (students : Fin 6), ∀ (colleges : Fin 3),
      (total_applications_without_restriction - 3 * applications_missing_one_college + 3 * overcounted_applications_missing_two_college = 540) :=
by {
  sorry
}

end application_methods_l2_2139


namespace fish_population_estimation_l2_2280

def tagged_fish_day1 := (30, 25, 25) -- (Species A, Species B, Species C)
def tagged_fish_day2 := (40, 35, 25) -- (Species A, Species B, Species C)
def caught_fish_day3 := (60, 50, 30) -- (Species A, Species B, Species C)
def tagged_fish_day3 := (4, 6, 2)    -- (Species A, Species B, Species C)
def caught_fish_day4 := (70, 40, 50) -- (Species A, Species B, Species C)
def tagged_fish_day4 := (5, 7, 3)    -- (Species A, Species B, Species C)

def total_tagged_fish (day1 : (ℕ × ℕ × ℕ)) (day2 : (ℕ × ℕ × ℕ)) :=
  let (a1, b1, c1) := day1
  let (a2, b2, c2) := day2
  (a1 + a2, b1 + b2, c1 + c2)

def average_proportion_tagged (caught3 tagged3 caught4 tagged4 : (ℕ × ℕ × ℕ)) :=
  let (c3a, c3b, c3c) := caught3
  let (t3a, t3b, t3c) := tagged3
  let (c4a, c4b, c4c) := caught4
  let (t4a, t4b, t4c) := tagged4
  ((t3a / c3a + t4a / c4a) / 2,
   (t3b / c3b + t4b / c4b) / 2,
   (t3c / c3c + t4c / c4c) / 2)

def estimate_population (total_tagged average_proportion : (ℕ × ℕ × ℕ)) :=
  let (ta, tb, tc) := total_tagged
  let (pa, pb, pc) := average_proportion
  (ta / pa, tb / pb, tc / pc)

theorem fish_population_estimation :
  let total_tagged := total_tagged_fish tagged_fish_day1 tagged_fish_day2
  let avg_prop := average_proportion_tagged caught_fish_day3 tagged_fish_day3 caught_fish_day4 tagged_fish_day4
  estimate_population total_tagged avg_prop = (1014, 407, 790) :=
by
  sorry

end fish_population_estimation_l2_2280


namespace complex_modulus_to_real_l2_2092

theorem complex_modulus_to_real (a : ℝ) (h : (a + 1)^2 + (1 - a)^2 = 10) : a = 2 ∨ a = -2 :=
sorry

end complex_modulus_to_real_l2_2092


namespace remainder_8_pow_215_mod_9_l2_2661

theorem remainder_8_pow_215_mod_9 : (8 ^ 215) % 9 = 8 := by
  -- condition
  have pattern : ∀ n, (8 ^ (2 * n + 1)) % 9 = 8 := by sorry
  -- final proof
  exact pattern 107

end remainder_8_pow_215_mod_9_l2_2661


namespace train_speed_is_correct_l2_2039

noncomputable def train_length : ℕ := 900
noncomputable def platform_length : ℕ := train_length
noncomputable def time_in_minutes : ℕ := 1
noncomputable def distance_covered : ℕ := train_length + platform_length
noncomputable def speed_m_per_minute : ℕ := distance_covered / time_in_minutes
noncomputable def speed_km_per_hr : ℕ := (speed_m_per_minute * 60) / 1000

theorem train_speed_is_correct :
  speed_km_per_hr = 108 :=
by
  sorry

end train_speed_is_correct_l2_2039


namespace set_difference_P_M_l2_2508

open Set

noncomputable def M : Set ℕ := {x | 1 ≤ x ∧ x ≤ 2009}
noncomputable def P : Set ℕ := {y | 2 ≤ y ∧ y ≤ 2010}

theorem set_difference_P_M : P \ M = {2010} :=
by
  sorry

end set_difference_P_M_l2_2508


namespace incorrect_connection_probability_l2_2143

noncomputable def probability_of_incorrect_connection (p : ℝ) : ℝ :=
  let r2 := 1 / 9
  let r3 := (8 / 9) * (1 / 9)
  (3 * p^2 * (1 - p) * r2) + (1 * p^3 * r3)

theorem incorrect_connection_probability : probability_of_incorrect_connection 0.02 = 0.000131 :=
by
  sorry

end incorrect_connection_probability_l2_2143


namespace whisker_relationship_l2_2756

theorem whisker_relationship :
  let P_whiskers := 14
  let C_whiskers := 22
  (C_whiskers - P_whiskers = 8) ∧ (C_whiskers / P_whiskers = 11 / 7) :=
by
  let P_whiskers := 14
  let C_whiskers := 22
  have h1 : C_whiskers - P_whiskers = 8 := by sorry
  have h2 : C_whiskers / P_whiskers = 11 / 7 := by sorry
  exact And.intro h1 h2

end whisker_relationship_l2_2756


namespace gear_ratio_proportion_l2_2350

variables {x y z w : ℕ} {ω_A ω_B ω_C ω_D : ℝ}

theorem gear_ratio_proportion 
  (h1: x * ω_A = y * ω_B) 
  (h2: y * ω_B = z * ω_C) 
  (h3: z * ω_C = w * ω_D):
  ω_A / ω_B = y * z * w / (x * z * w) ∧ 
  ω_B / ω_C = x * z * w / (y * x * w) ∧ 
  ω_C / ω_D = x * y * w / (z * y * w) ∧ 
  ω_D / ω_A = x * y * z / (w * z * y) :=
sorry  -- Proof is not included

end gear_ratio_proportion_l2_2350


namespace michael_twenty_dollar_bills_l2_2867

theorem michael_twenty_dollar_bills (total_amount : ℕ) (denomination : ℕ) 
  (h_total : total_amount = 280) (h_denom : denomination = 20) : 
  total_amount / denomination = 14 := by
  sorry

end michael_twenty_dollar_bills_l2_2867


namespace sqrt_multiplication_division_l2_2802

theorem sqrt_multiplication_division :
  Real.sqrt 27 * Real.sqrt (8 / 3) / Real.sqrt (1 / 2) = 18 :=
by
  sorry

end sqrt_multiplication_division_l2_2802


namespace segment_measure_l2_2412

theorem segment_measure (a b : ℝ) (m : ℝ) (h : a = m * b) : (1 / m) * a = b :=
by sorry

end segment_measure_l2_2412


namespace find_x_squared_minus_y_squared_l2_2113

theorem find_x_squared_minus_y_squared 
  (x y : ℝ)
  (h1 : x + y = 5)
  (h2 : x - y = 1) :
  x^2 - y^2 = 5 := 
by
  sorry

end find_x_squared_minus_y_squared_l2_2113


namespace det_of_matrix_l2_2491

variable {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ)

theorem det_of_matrix (h1 : 1 ≤ n)
  (h2 : A ^ 7 + A ^ 5 + A ^ 3 + A - 1 = 0) :
  0 < Matrix.det A :=
sorry

end det_of_matrix_l2_2491


namespace johns_total_cost_after_discount_l2_2798

/-- Price of different utensils for John's purchase --/
def forks_cost : ℕ := 25
def knives_cost : ℕ := 30
def spoons_cost : ℕ := 20
def dinner_plate_cost (silverware_cost : ℕ) : ℚ := 0.5 * silverware_cost

/-- Calculating the total cost of silverware --/
def total_silverware_cost : ℕ := forks_cost + knives_cost + spoons_cost

/-- Calculating the total cost before discount --/
def total_cost_before_discount : ℚ := total_silverware_cost + dinner_plate_cost total_silverware_cost

/-- Discount rate --/
def discount_rate : ℚ := 0.10

/-- Discount amount --/
def discount_amount (total_cost : ℚ) : ℚ := discount_rate * total_cost

/-- Total cost after applying discount --/
def total_cost_after_discount : ℚ := total_cost_before_discount - discount_amount total_cost_before_discount

/-- John's total cost after the discount should be $101.25 --/
theorem johns_total_cost_after_discount : total_cost_after_discount = 101.25 := by
  sorry

end johns_total_cost_after_discount_l2_2798


namespace price_of_stock_l2_2006

-- Defining the conditions
def income : ℚ := 650
def dividend_rate : ℚ := 10
def investment : ℚ := 6240

-- Defining the face value calculation from income and dividend rate
def face_value (i : ℚ) (d_rate : ℚ) : ℚ := (i * 100) / d_rate

-- Calculating the price of the stock
def stock_price (inv : ℚ) (fv : ℚ) : ℚ := (inv / fv) * 100

-- Main theorem to be proved
theorem price_of_stock : stock_price investment (face_value income dividend_rate) = 96 := by
  sorry

end price_of_stock_l2_2006


namespace total_money_l2_2117

theorem total_money 
  (n_pennies n_nickels n_dimes n_quarters n_half_dollars : ℝ) 
  (h_pennies : n_pennies = 9) 
  (h_nickels : n_nickels = 4) 
  (h_dimes : n_dimes = 3) 
  (h_quarters : n_quarters = 7) 
  (h_half_dollars : n_half_dollars = 5) : 
  0.01 * n_pennies + 0.05 * n_nickels + 0.10 * n_dimes + 0.25 * n_quarters + 0.50 * n_half_dollars = 4.84 :=
by 
  sorry

end total_money_l2_2117


namespace cosine_theorem_a_cosine_theorem_b_cosine_theorem_c_l2_2776

theorem cosine_theorem_a (a b c A : ℝ) :
  a^2 = b^2 + c^2 - 2 * b * c * Real.cos A := sorry

theorem cosine_theorem_b (a b c B : ℝ) :
  b^2 = a^2 + c^2 - 2 * a * c * Real.cos B := sorry

theorem cosine_theorem_c (a b c C : ℝ) :
  c^2 = a^2 + b^2 - 2 * a * b * Real.cos C := sorry

end cosine_theorem_a_cosine_theorem_b_cosine_theorem_c_l2_2776


namespace exists_valid_numbers_l2_2736

noncomputable def sum_of_numbers_is_2012_using_two_digits : Prop :=
  ∃ (a b c d : ℕ), (a < 1000) ∧ (b < 1000) ∧ (c < 1000) ∧ (d < 1000) ∧ 
                    (∀ n ∈ [a, b, c, d], ∃ x y, (x ≠ y) ∧ ((∀ d ∈ [n / 100 % 10, n / 10 % 10, n % 10], d = x ∨ d = y))) ∧
                    (a + b + c + d = 2012)

theorem exists_valid_numbers : sum_of_numbers_is_2012_using_two_digits :=
  sorry

end exists_valid_numbers_l2_2736


namespace min_value_of_inverse_proportional_function_l2_2021

theorem min_value_of_inverse_proportional_function 
  (x y : ℝ) (k : ℝ) 
  (h1 : y = k / x) 
  (h2 : ∀ x, -2 ≤ x ∧ x ≤ -1 → y ≤ 4) :
  (∀ x, x ≥ 8 → y = -1 / 2) :=
by
  sorry

end min_value_of_inverse_proportional_function_l2_2021


namespace problem1_problem2_l2_2056

theorem problem1 : -24 - (-15) + (-1) + (-15) = -25 := 
by 
  sorry

theorem problem2 : -27 / (3 / 2) * (2 / 3) = -12 := 
by 
  sorry

end problem1_problem2_l2_2056


namespace circle_area_circle_circumference_l2_2968

section CircleProperties

variable (r : ℝ) -- Define the radius of the circle as a real number

-- State the theorem for the area of the circle
theorem circle_area (A : ℝ) : A = π * r^2 :=
sorry

-- State the theorem for the circumference of the circle
theorem circle_circumference (C : ℝ) : C = 2 * π * r :=
sorry

end CircleProperties

end circle_area_circle_circumference_l2_2968


namespace deepak_present_age_l2_2645

theorem deepak_present_age (x : ℕ) (h1 : ∀ current_age_rahul current_age_deepak, 
  4 * x = current_age_rahul ∧ 3 * x = current_age_deepak)
  (h2 : ∀ current_age_rahul, current_age_rahul + 6 = 22) :
  3 * x = 12 :=
by
  have h3 : 4 * x + 6 = 22 := h2 (4 * x)
  linarith

end deepak_present_age_l2_2645


namespace SarahsScoreIs135_l2_2581

variable (SarahsScore GregsScore : ℕ)

-- Conditions
def ScoreDifference (SarahsScore GregsScore : ℕ) : Prop := SarahsScore = GregsScore + 50
def AverageScore (SarahsScore GregsScore : ℕ) : Prop := (SarahsScore + GregsScore) / 2 = 110

-- Theorem statement
theorem SarahsScoreIs135 (h1 : ScoreDifference SarahsScore GregsScore) (h2 : AverageScore SarahsScore GregsScore) : SarahsScore = 135 :=
sorry

end SarahsScoreIs135_l2_2581


namespace problem_statement_l2_2622

open Function

theorem problem_statement :
  ∃ g : ℝ → ℝ, 
    (g 1 = 2) ∧ 
    (∀ (x y : ℝ), g (x^2 - y^2) = (x - y) * (g x + g y)) ∧ 
    (g 3 = 6) := 
by
  sorry

end problem_statement_l2_2622


namespace max_value_l2_2663

open Real

theorem max_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + 5 * y < 75) : 
  xy * (75 - 2 * x - 5 * y) ≤ 1562.5 := 
sorry

end max_value_l2_2663


namespace multiply_square_expression_l2_2949

theorem multiply_square_expression (x : ℝ) : ((-3 * x) ^ 2) * (2 * x) = 18 * x ^ 3 := by
  sorry

end multiply_square_expression_l2_2949


namespace average_of_integers_is_ten_l2_2434

theorem average_of_integers_is_ten (k m r s t : ℕ) 
  (h1 : k < m) (h2 : m < r) (h3 : r < s) (h4 : s < t)
  (h5 : k > 0) (h6 : m > 0)
  (h7 : t = 20) (h8 : r = 13)
  (h9 : k = 1) (h10 : m = 2) (h11 : s = 14) :
  (k + m + r + s + t) / 5 = 10 := by
  sorry

end average_of_integers_is_ten_l2_2434


namespace percentage_cats_less_dogs_l2_2500

theorem percentage_cats_less_dogs (C D F : ℕ) (h1 : C < D) (h2 : F = 2 * D) (h3 : C + D + F = 304) (h4 : F = 160) :
  ((D - C : ℕ) * 100 / D : ℕ) = 20 := 
sorry

end percentage_cats_less_dogs_l2_2500


namespace number_of_terms_arithmetic_sequence_l2_2814

-- Definitions for the arithmetic sequence conditions
open Nat

noncomputable def S4 := 26
noncomputable def Sn := 187
noncomputable def last4_sum (n : ℕ) (a d : ℕ) := 
  (n - 3) * a + 3 * (n - 2) * d + 3 * (n - 1) * d + n * d

-- Statement for the problem
theorem number_of_terms_arithmetic_sequence 
  (a d n : ℕ) (h1 : 4 * a + 6 * d = S4) (h2 : n * (2 * a + (n - 1) * d) / 2 = Sn) 
  (h3 : last4_sum n a d = 110) : 
  n = 11 :=
sorry

end number_of_terms_arithmetic_sequence_l2_2814


namespace earbuds_cost_before_tax_l2_2839

-- Define the conditions
variable (C : ℝ) -- The cost before tax
variable (taxRate : ℝ := 0.15)
variable (totalPaid : ℝ := 230)

-- Define the main question in Lean
theorem earbuds_cost_before_tax : C + taxRate * C = totalPaid → C = 200 :=
by
  sorry

end earbuds_cost_before_tax_l2_2839


namespace laser_beam_total_distance_l2_2852

theorem laser_beam_total_distance :
  let A := (4, 7)
  let B := (-4, 7)
  let C := (-4, -7)
  let D := (4, -7)
  let E := (9, 7)
  let dist (p1 p2 : (ℤ × ℤ)) : ℝ := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  dist A B + dist B C + dist C D + dist D E = 30 + Real.sqrt 221 :=
by
  sorry

end laser_beam_total_distance_l2_2852


namespace jogger_distance_ahead_l2_2958

theorem jogger_distance_ahead
  (train_speed_km_hr : ℝ) (jogger_speed_km_hr : ℝ)
  (train_length_m : ℝ) (time_seconds : ℝ)
  (relative_speed_m_s : ℝ) (distance_covered_m : ℝ)
  (D : ℝ)
  (h1 : train_speed_km_hr = 45)
  (h2 : jogger_speed_km_hr = 9)
  (h3 : train_length_m = 100)
  (h4 : time_seconds = 25)
  (h5 : relative_speed_m_s = 36 * (5/18))
  (h6 : distance_covered_m = 10 * 25)
  (h7 : D + train_length_m = distance_covered_m) :
  D = 150 :=
by sorry

end jogger_distance_ahead_l2_2958


namespace circle_placement_possible_l2_2142

theorem circle_placement_possible
  (length : ℕ)
  (width : ℕ)
  (n : ℕ)
  (area_ci : ℕ)
  (ne_int_lt : length = 20)
  (ne_wid_lt : width = 25)
  (ne_squares : n = 120)
  (sm_area_lt : area_ci = 456) :
  120 * (1 + (Real.pi / 4)) < area_ci :=
by sorry

end circle_placement_possible_l2_2142


namespace final_number_not_perfect_square_l2_2010

theorem final_number_not_perfect_square :
  (∃ final_number : ℕ, 
    ∀ a b : ℕ, a ∈ Finset.range 101 ∧ b ∈ Finset.range 101 ∧ a ≠ b → 
    gcd (a^2 + b^2 + 2) (a^2 * b^2 + 3) = final_number) →
  ∀ final_number : ℕ, ¬ ∃ k : ℕ, final_number = k ^ 2 :=
sorry

end final_number_not_perfect_square_l2_2010


namespace vanya_faster_speed_l2_2873

def vanya_speed (v : ℝ) : Prop :=
  (v + 2) / v = 2.5

theorem vanya_faster_speed (v : ℝ) (h : vanya_speed v) : (v + 4) / v = 4 :=
by
  sorry

end vanya_faster_speed_l2_2873


namespace points_on_hyperbola_order_l2_2742

theorem points_on_hyperbola_order (k a b c : ℝ) (hk : k > 0)
  (h₁ : a = k / -2)
  (h₂ : b = k / 2)
  (h₃ : c = k / 3) :
  a < c ∧ c < b := 
sorry

end points_on_hyperbola_order_l2_2742


namespace crates_needed_l2_2615

-- Conditions as definitions
def novels : ℕ := 145
def comics : ℕ := 271
def documentaries : ℕ := 419
def albums : ℕ := 209
def crate_capacity : ℕ := 9

-- Total items calculation
def total_items : ℕ := novels + comics + documentaries + albums

-- Proof statement
theorem crates_needed : (total_items / crate_capacity) = 116 := by
  sorry

end crates_needed_l2_2615


namespace toby_money_share_l2_2009

theorem toby_money_share (initial_money : ℕ) (fraction : ℚ) (brothers : ℕ) (money_per_brother : ℚ)
  (total_shared : ℕ) (remaining_money : ℕ) :
  initial_money = 343 →
  fraction = 1/7 →
  brothers = 2 →
  money_per_brother = fraction * initial_money →
  total_shared = brothers * money_per_brother →
  remaining_money = initial_money - total_shared →
  remaining_money = 245 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end toby_money_share_l2_2009


namespace arithmetic_sequence_common_difference_l2_2772

theorem arithmetic_sequence_common_difference 
  (a1 a2 a3 a4 d : ℕ)
  (S : ℕ → ℕ)
  (h1 : S 2 = a1 + a2)
  (h2 : S 4 = a1 + a2 + a3 + a4)
  (h3 : S 2 = 4)
  (h4 : S 4 = 20)
  (h5 : a2 = a1 + d)
  (h6 : a3 = a2 + d)
  (h7 : a4 = a3 + d) :
  d = 3 :=
by
  sorry

end arithmetic_sequence_common_difference_l2_2772


namespace questionnaires_drawn_from_unit_D_l2_2908

theorem questionnaires_drawn_from_unit_D 
  (arith_seq_collected : ∃ a1 d : ℕ, [a1, a1 + d, a1 + 2 * d, a1 + 3 * d] = [aA, aB, aC, aD] ∧ aA + aB + aC + aD = 1000)
  (stratified_sample : [30 - d, 30, 30 + d, 30 + 2 * d] = [sA, sB, sC, sD] ∧ sA + sB + sC + sD = 150)
  (B_drawn : 30 = sB) :
  sD = 60 := 
by {
  sorry
}

end questionnaires_drawn_from_unit_D_l2_2908


namespace probability_no_adjacent_standing_l2_2748

-- Define the problem conditions in Lean 4.
def total_outcomes := 2^10
def favorable_outcomes := 123

-- The probability is given by favorable outcomes over total outcomes.
def probability : ℚ := favorable_outcomes / total_outcomes

-- Now state the theorem regarding the probability.
theorem probability_no_adjacent_standing : 
  probability = 123 / 1024 :=
by {
  sorry
}

end probability_no_adjacent_standing_l2_2748


namespace sin_double_angle_l2_2310

theorem sin_double_angle (θ : ℝ) (h : Real.sin (π / 4 + θ) = 1 / 3) : Real.sin (2 * θ) = -7 / 9 :=
by
  sorry

end sin_double_angle_l2_2310


namespace election_votes_l2_2753

theorem election_votes
  (V : ℕ)  -- total number of votes
  (candidate1_votes_percent : ℕ := 80)  -- first candidate percentage
  (second_candidate_votes : ℕ := 480)  -- votes for second candidate
  (second_candidate_percent : ℕ := 20)  -- second candidate percentage
  (h : second_candidate_votes = (second_candidate_percent * V) / 100) :
  V = 2400 :=
sorry

end election_votes_l2_2753


namespace negation_of_proposition_l2_2714

theorem negation_of_proposition (x : ℝ) : 
  ¬ (|x| < 2 → x < 2) ↔ (|x| ≥ 2 → x ≥ 2) :=
sorry

end negation_of_proposition_l2_2714


namespace sum_of_fractions_le_half_l2_2917

theorem sum_of_fractions_le_half {a b c : ℝ} (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : a * b * c = 1) :
  1 / (a^2 + 2 * b^2 + 3) + 1 / (b^2 + 2 * c^2 + 3) + 1 / (c^2 + 2 * a^2 + 3) ≤ 1 / 2 :=
by
  sorry

end sum_of_fractions_le_half_l2_2917


namespace rational_roots_of_quadratic_l2_2589

theorem rational_roots_of_quadratic (k : ℤ) (h : k > 0) :
  (∃ x : ℚ, k * x^2 + 12 * x + k = 0) ↔ (k = 3 ∨ k = 6) :=
by
  sorry

end rational_roots_of_quadratic_l2_2589


namespace radius_of_2007_l2_2637

-- Define the conditions
def given_condition (n : ℕ) (r : ℕ → ℝ) : Prop :=
  r 1 = 1 ∧ (∀ i, 1 ≤ i ∧ i < n → r (i + 1) = 3 * r i)

-- State the theorem we want to prove
theorem radius_of_2007 (r : ℕ → ℝ) : given_condition 2007 r → r 2007 = 3^2006 :=
by
  sorry -- Proof placeholder

end radius_of_2007_l2_2637


namespace noah_small_paintings_sold_last_month_l2_2245

theorem noah_small_paintings_sold_last_month
  (large_painting_price small_painting_price : ℕ)
  (large_paintings_sold_last_month : ℕ)
  (total_sales_this_month : ℕ)
  (sale_multiplier : ℕ)
  (x : ℕ)
  (h1 : large_painting_price = 60)
  (h2 : small_painting_price = 30)
  (h3 : large_paintings_sold_last_month = 8)
  (h4 : total_sales_this_month = 1200)
  (h5 : sale_multiplier = 2) :
  (2 * ((large_paintings_sold_last_month * large_painting_price) + (x * small_painting_price)) = total_sales_this_month) → x = 4 :=
by
  sorry

end noah_small_paintings_sold_last_month_l2_2245


namespace seven_large_power_mod_seventeen_l2_2688

theorem seven_large_power_mod_seventeen :
  (7 : ℤ)^1985 % 17 = 7 :=
by
  have h1 : (7 : ℤ)^2 % 17 = 15 := sorry
  have h2 : (7 : ℤ)^4 % 17 = 16 := sorry
  have h3 : (7 : ℤ)^8 % 17 = 1 := sorry
  have h4 : 1985 = 8 * 248 + 1 := sorry
  sorry

end seven_large_power_mod_seventeen_l2_2688


namespace remainder_div_8_l2_2580

theorem remainder_div_8 (x : ℤ) (h : ∃ k : ℤ, x = 63 * k + 27) : x % 8 = 3 :=
by
  sorry

end remainder_div_8_l2_2580


namespace truck_distance_l2_2927

theorem truck_distance (d: ℕ) (g: ℕ) (eff: ℕ) (new_g: ℕ) (total_distance: ℕ)
  (h1: d = 300) (h2: g = 10) (h3: eff = d / g) (h4: new_g = 15) (h5: total_distance = eff * new_g):
  total_distance = 450 :=
sorry

end truck_distance_l2_2927


namespace value_of_a_l2_2192

theorem value_of_a (a : ℚ) (h : a + a / 4 = 6 / 2) : a = 12 / 5 := by
  sorry

end value_of_a_l2_2192


namespace mouse_jump_vs_grasshopper_l2_2165

-- Definitions for jumps
def grasshopper_jump : ℕ := 14
def frog_jump : ℕ := grasshopper_jump + 37
def mouse_jump : ℕ := frog_jump - 16

-- Theorem stating the result
theorem mouse_jump_vs_grasshopper : mouse_jump - grasshopper_jump = 21 :=
by
  -- Skip the proof
  sorry

end mouse_jump_vs_grasshopper_l2_2165


namespace average_age_of_dance_group_l2_2119

theorem average_age_of_dance_group
  (avg_age_children : ℕ)
  (avg_age_adults : ℕ)
  (num_children : ℕ)
  (num_adults : ℕ)
  (total_num_members : ℕ)
  (total_sum_ages : ℕ)
  (average_age : ℚ)
  (h_children : avg_age_children = 12)
  (h_adults : avg_age_adults = 40)
  (h_num_children : num_children = 8)
  (h_num_adults : num_adults = 12)
  (h_total_members : total_num_members = 20)
  (h_total_ages : total_sum_ages = 576)
  (h_average_age : average_age = 28.8) :
  average_age = (total_sum_ages : ℚ) / total_num_members :=
by
  sorry

end average_age_of_dance_group_l2_2119


namespace difference_of_numbers_l2_2469

variables (x y : ℝ)

-- Definitions corresponding to the conditions
def sum_of_numbers (x y : ℝ) : Prop := x + y = 30
def product_of_numbers (x y : ℝ) : Prop := x * y = 200

-- The proof statement in Lean
theorem difference_of_numbers (x y : ℝ) 
  (h1: sum_of_numbers x y) 
  (h2: product_of_numbers x y) : x - y = 10 ∨ y - x = 10 :=
by
  sorry

end difference_of_numbers_l2_2469


namespace squares_difference_l2_2687

theorem squares_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x - y = 4) : x^2 - y^2 = 40 :=
by sorry

end squares_difference_l2_2687


namespace part1_part2_l2_2370

variable (a m : ℝ)

def f (x : ℝ) : ℝ := 2 * |x - 1| - a

theorem part1 (h : ∃ x, f a x - 2 * |x - 7| ≤ 0) : a ≥ -12 :=
sorry

theorem part2 (h : ∀ x, f 1 x + |x + 7| ≥ m) : m ≤ 7 :=
sorry

end part1_part2_l2_2370


namespace binary_arithmetic_l2_2513

def a : ℕ := 0b10110  -- 10110_2
def b : ℕ := 0b1101   -- 1101_2
def c : ℕ := 0b11100  -- 11100_2
def d : ℕ := 0b11101  -- 11101_2
def e : ℕ := 0b101    -- 101_2

theorem binary_arithmetic :
  (a + b - c + d + e) = 0b101101 := by
  sorry

end binary_arithmetic_l2_2513


namespace find_p_l2_2067

theorem find_p 
  (a : ℝ) (p : ℕ) 
  (h1 : 12345 * 6789 = a * 10^p)
  (h2 : 1 ≤ a) (h3 : a < 10) (h4 : 0 < p) 
  : p = 7 := 
sorry

end find_p_l2_2067


namespace initial_books_count_l2_2928

-- Definitions in conditions
def books_sold : ℕ := 42
def books_left : ℕ := 66

-- The theorem to prove the initial books count
theorem initial_books_count (initial_books : ℕ) : initial_books = books_sold + books_left :=
  by sorry

end initial_books_count_l2_2928


namespace charlotte_flour_cost_l2_2211

noncomputable def flour_cost 
  (flour_sugar_eggs_butter_cost blueberry_cost cherry_cost total_cost : ℝ)
  (blueberry_weight oz_per_lb blueberry_cost_per_container cherry_weight cherry_cost_per_bag : ℝ)
  (additional_cost : ℝ) : ℝ :=
  total_cost - (blueberry_cost + additional_cost)

theorem charlotte_flour_cost :
  flour_cost 2.5 13.5 14 18 3 16 2.25 4 14 2.5 = 2 :=
by
  unfold flour_cost
  sorry

end charlotte_flour_cost_l2_2211


namespace line_through_A_area_1_l2_2132

def line_equation : Prop :=
  ∃ k : ℚ, ∀ x y : ℚ, (y = k * (x + 2) + 2) ↔ 
    (x + 2 * y - 2 = 0 ∨ 2 * x + y + 2 = 0) ∧ 
    (2 * (k * 0 + 2) * (-2 - 2 / k) = 2)

theorem line_through_A_area_1 : line_equation :=
by
  sorry

end line_through_A_area_1_l2_2132


namespace sum_of_squares_first_20_l2_2797

-- Define the sum of squares function
def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

-- Specific problem instance
theorem sum_of_squares_first_20 : sum_of_squares 20 = 5740 :=
  by
  -- Proof skipping placeholder
  sorry

end sum_of_squares_first_20_l2_2797


namespace matrix_determinant_eq_16_l2_2277

theorem matrix_determinant_eq_16 (x : ℝ) :
  (3 * x) * (4 * x) - (2 * x) = 16 ↔ x = 4 / 3 ∨ x = -1 :=
by sorry

end matrix_determinant_eq_16_l2_2277


namespace total_revenue_correct_l2_2633

-- Definitions based on the problem conditions
def price_per_kg_first_week : ℝ := 10
def quantity_sold_first_week : ℝ := 50
def discount_percentage : ℝ := 0.25
def multiplier_next_week : ℝ := 3

-- Derived definitions
def revenue_first_week := quantity_sold_first_week * price_per_kg_first_week
def quantity_sold_second_week := multiplier_next_week * quantity_sold_first_week
def discounted_price_per_kg := price_per_kg_first_week * (1 - discount_percentage)
def revenue_second_week := quantity_sold_second_week * discounted_price_per_kg
def total_revenue := revenue_first_week + revenue_second_week

-- The theorem that needs to be proven
theorem total_revenue_correct : total_revenue = 1625 := 
by
  sorry

end total_revenue_correct_l2_2633


namespace no_integer_solution_l2_2397

theorem no_integer_solution (a b : ℤ) : ¬ (4 ∣ a^2 + b^2 + 1) :=
by
  -- Prevent use of the solution steps and add proof obligations
  sorry

end no_integer_solution_l2_2397


namespace simplify_and_evaluate_l2_2018

theorem simplify_and_evaluate (m n : ℤ) (h1 : m = 1) (h2 : n = -2) :
  -2 * (m * n - 3 * m^2) - (2 * m * n - 5 * (m * n - m^2)) = -1 :=
by
  sorry

end simplify_and_evaluate_l2_2018


namespace smallest_d_for_inverse_l2_2889

def g (x : ℝ) : ℝ := (x - 3) ^ 2 - 1

theorem smallest_d_for_inverse :
  ∃ d : ℝ, (∀ x1 x2 : ℝ, x1 ≠ x2 → (d ≤ x1) → (d ≤ x2) → g x1 ≠ g x2) ∧ d = 3 :=
by
  sorry

end smallest_d_for_inverse_l2_2889


namespace bank_card_payment_technology_order_l2_2014

-- Conditions as definitions
def action_tap := 1
def action_pay_online := 2
def action_swipe := 3
def action_insert_into_terminal := 4

-- Corresponding proof problem statement
theorem bank_card_payment_technology_order :
  [action_insert_into_terminal, action_swipe, action_tap, action_pay_online] = [4, 3, 1, 2] := by
  sorry

end bank_card_payment_technology_order_l2_2014


namespace plan1_has_higher_expected_loss_l2_2683

noncomputable def prob_minor_flooding : ℝ := 0.2
noncomputable def prob_major_flooding : ℝ := 0.05
noncomputable def cost_plan1 : ℝ := 4000
noncomputable def loss_major_plan1 : ℝ := 30000
noncomputable def loss_minor_plan2 : ℝ := 15000
noncomputable def loss_major_plan2 : ℝ := 30000

noncomputable def expected_loss_plan1 : ℝ :=
  (loss_major_plan1 * prob_major_flooding) + (cost_plan1 * prob_minor_flooding) + cost_plan1

noncomputable def expected_loss_plan2 : ℝ :=
  (loss_major_plan2 * prob_major_flooding) + (loss_minor_plan2 * prob_minor_flooding)

theorem plan1_has_higher_expected_loss : expected_loss_plan1 > expected_loss_plan2 :=
by
  sorry

end plan1_has_higher_expected_loss_l2_2683


namespace sum_of_sides_l2_2757

-- Definitions: Given conditions
def ratio (a b c : ℕ) : Prop := 
a * 5 = b * 3 ∧ b * 7 = c * 5

-- Given that the longest side is 21 cm and the ratio of the sides is 3:5:7
def similar_triangle (x y : ℕ) : Prop :=
ratio x y 21

-- Proof statement: The sum of the lengths of the other two sides is 24 cm
theorem sum_of_sides (x y : ℕ) (h : similar_triangle x y) : x + y = 24 :=
sorry

end sum_of_sides_l2_2757


namespace Sara_house_size_l2_2118

theorem Sara_house_size (nada_size : ℕ) (h1 : nada_size = 450) (h2 : Sara_size = 2 * nada_size + 100) : Sara_size = 1000 :=
by sorry

end Sara_house_size_l2_2118


namespace intersection_eq_l2_2741

def set1 : Set ℝ := {x | 1 ≤ x ∧ x < 4}
def set2 : Set ℝ := {x | -2 ≤ x ∧ x < 2}

theorem intersection_eq : (set1 ∩ set2) = {x | 1 ≤ x ∧ x < 2} :=
by
  sorry

end intersection_eq_l2_2741


namespace production_steps_use_process_flowchart_l2_2843

def describe_production_steps (task : String) : Prop :=
  task = "describe production steps of a certain product in a factory"

def correct_diagram (diagram : String) : Prop :=
  diagram = "Process Flowchart"

theorem production_steps_use_process_flowchart (task : String) (diagram : String) :
  describe_production_steps task → correct_diagram diagram :=
sorry

end production_steps_use_process_flowchart_l2_2843


namespace olivia_probability_l2_2883

noncomputable def total_outcomes (n m : ℕ) : ℕ := Nat.choose n m

noncomputable def favorable_outcomes : ℕ :=
  let choose_three_colors := total_outcomes 4 3
  let choose_one_for_pair := total_outcomes 3 1
  let choose_socks :=
    (total_outcomes 3 2) * (total_outcomes 3 1) * (total_outcomes 3 1)
  choose_three_colors * choose_one_for_pair * choose_socks

def probability (n m : ℕ) : ℚ := n / m

theorem olivia_probability :
  probability favorable_outcomes (total_outcomes 12 5) = 9 / 22 :=
by
  sorry

end olivia_probability_l2_2883


namespace problem_solution_l2_2259

theorem problem_solution (x y : ℝ) (h₁ : (4 * y^2 + 1) * (x^4 + 2 * x^2 + 2) = 8 * |y| * (x^2 + 1))
  (h₂ : y ≠ 0) :
  (x = 0 ∧ (y = 1/2 ∨ y = -1/2)) :=
by {
  sorry -- Proof required
}

end problem_solution_l2_2259


namespace evaluate_expression_l2_2665

theorem evaluate_expression : (3^2)^4 * 2^3 = 52488 := by
  sorry

end evaluate_expression_l2_2665


namespace smallest_side_of_triangle_l2_2282

theorem smallest_side_of_triangle (A B C : ℝ) (a b c : ℝ) 
  (hA : A = 60) (hC : C = 45) (hb : b = 4) (h_sum : A + B + C = 180) : 
  c = 4 * Real.sqrt 3 - 4 := 
sorry

end smallest_side_of_triangle_l2_2282


namespace number_of_parallel_lines_l2_2789

/-- 
Given 10 parallel lines in the first set and the fact that the intersection 
of two sets of parallel lines forms 1260 parallelograms, 
prove that the second set contains 141 parallel lines.
-/
theorem number_of_parallel_lines (n : ℕ) (h₁ : 10 - 1 = 9) (h₂ : 9 * (n - 1) = 1260) : n = 141 :=
sorry

end number_of_parallel_lines_l2_2789


namespace total_weight_gain_l2_2391

def orlando_gained : ℕ := 5

def jose_gained (orlando : ℕ) : ℕ :=
  2 * orlando + 2

def fernando_gained (jose : ℕ) : ℕ :=
  jose / 2 - 3

theorem total_weight_gain (O J F : ℕ) 
  (ho : O = orlando_gained) 
  (hj : J = jose_gained O) 
  (hf : F = fernando_gained J) :
  O + J + F = 20 :=
by
  sorry

end total_weight_gain_l2_2391


namespace quotient_three_l2_2625

theorem quotient_three (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : a * b ∣ a^2 + b^2 + 1) :
  (a^2 + b^2 + 1) / (a * b) = 3 :=
sorry

end quotient_three_l2_2625


namespace sum_m_n_zero_l2_2036

theorem sum_m_n_zero (m n p : ℝ) (h1 : mn + p^2 + 4 = 0) (h2 : m - n = 4) : m + n = 0 :=
sorry

end sum_m_n_zero_l2_2036


namespace number_of_members_in_league_l2_2449

-- Define the costs of the items considering the conditions
def sock_cost : ℕ := 6
def tshirt_cost : ℕ := sock_cost + 3
def shorts_cost : ℕ := sock_cost + 2

-- Define the total cost for one member
def total_cost_one_member : ℕ := 
  2 * (sock_cost + tshirt_cost + shorts_cost)

-- Given total expenditure
def total_expenditure : ℕ := 4860

-- Define the theorem to be proved
theorem number_of_members_in_league :
  total_expenditure / total_cost_one_member = 106 :=
by 
  sorry

end number_of_members_in_league_l2_2449


namespace speed_against_current_l2_2657

theorem speed_against_current (V_m V_c : ℕ) (h1 : V_m + V_c = 20) (h2 : V_c = 3) : V_m - V_c = 14 :=
by 
  sorry

end speed_against_current_l2_2657


namespace smallest_value_of_x_l2_2311

theorem smallest_value_of_x (x : ℝ) (h : 6 * x ^ 2 - 37 * x + 48 = 0) : x = 13 / 6 :=
sorry

end smallest_value_of_x_l2_2311


namespace k_value_l2_2879

noncomputable def find_k : ℚ := 49 / 15

theorem k_value :
  ∀ (a b : ℚ), (3 * a^2 + 7 * a + find_k = 0) ∧ (3 * b^2 + 7 * b + find_k = 0) →
                (a^2 + b^2 = 3 * a * b) →
                find_k = 49 / 15 :=
by
  intros a b h_eq_root h_rel
  sorry

end k_value_l2_2879


namespace even_not_div_by_4_not_sum_consecutive_odds_l2_2654

theorem even_not_div_by_4_not_sum_consecutive_odds
  (e : ℤ) (h_even: e % 2 = 0) (h_nondiv4: ¬ (e % 4 = 0)) :
  ∀ n : ℤ, e ≠ n + (n + 2) :=
by
  sorry

end even_not_div_by_4_not_sum_consecutive_odds_l2_2654


namespace smallest_square_side_length_l2_2373

theorem smallest_square_side_length (s : ℕ) :
  (∃ s, s > 3 ∧ s ≤ 4 ∧ (s - 1) * (s - 1) = 5) ↔ s = 4 := by
  sorry

end smallest_square_side_length_l2_2373


namespace car_production_total_l2_2743

theorem car_production_total (northAmericaCars europeCars : ℕ) (h1 : northAmericaCars = 3884) (h2 : europeCars = 2871) : northAmericaCars + europeCars = 6755 := by
  sorry

end car_production_total_l2_2743


namespace geometric_sequence_ratio_l2_2445

-- Definitions and conditions from part a)
def q : ℚ := 1 / 2

def sum_of_first_n (a1 : ℚ) (n : ℕ) : ℚ :=
  a1 * (1 - q ^ n) / (1 - q)

def a_n (a1 : ℚ) (n : ℕ) : ℚ :=
  a1 * q ^ (n - 1)

-- Theorem representing the proof problem from part c)
theorem geometric_sequence_ratio (a1 : ℚ) : 
  (sum_of_first_n a1 4) / (a_n a1 3) = 15 / 2 := 
sorry

end geometric_sequence_ratio_l2_2445


namespace min_value_l2_2481

theorem min_value (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_ab : a * b = 1) (h_a_2b : a = 2 * b) :
  a + 2 * b = 2 * Real.sqrt 2 := by
  sorry

end min_value_l2_2481


namespace ellipse_sum_l2_2731

noncomputable def h : ℝ := 3
noncomputable def k : ℝ := 0
noncomputable def a : ℝ := 5
noncomputable def b : ℝ := Real.sqrt 21
noncomputable def F_1 : (ℝ × ℝ) := (1, 0)
noncomputable def F_2 : (ℝ × ℝ) := (5, 0)

theorem ellipse_sum :
  (F_1 = (1, 0)) → 
  (F_2 = (5, 0)) →
  (∀ P : (ℝ × ℝ), (Real.sqrt ((P.1 - F_1.1)^2 + (P.2 - F_1.2)^2) + Real.sqrt ((P.1 - F_2.1)^2 + (P.2 - F_2.2)^2) = 10)) →
  (h + k + a + b = 8 + Real.sqrt 21) :=
by
  intros
  sorry

end ellipse_sum_l2_2731


namespace moon_temp_difference_l2_2705

def temp_difference (T_day T_night : ℤ) : ℤ := T_day - T_night

theorem moon_temp_difference :
  temp_difference 127 (-183) = 310 :=
by
  sorry

end moon_temp_difference_l2_2705


namespace multiplication_result_l2_2442

theorem multiplication_result : 
  (500 * 2468 * 0.2468 * 100) = 30485120 :=
by
  sorry

end multiplication_result_l2_2442


namespace prove_minimality_of_smallest_three_digit_multiple_of_17_l2_2921

def smallest_three_digit_multiple_of_17: ℕ :=
  102

theorem prove_minimality_of_smallest_three_digit_multiple_of_17 (n : ℕ) : 
  (∃ k : ℕ, n = 17 * k ∧ 100 ≤ 17 * k ∧ 17 * k < 1000) 
  → n = 102 :=
    by
      sorry

end prove_minimality_of_smallest_three_digit_multiple_of_17_l2_2921


namespace initial_customers_count_l2_2375

theorem initial_customers_count (left_count remaining_people_per_table tables remaining_customers : ℕ) 
  (h1 : left_count = 14) 
  (h2 : remaining_people_per_table = 4) 
  (h3 : tables = 2) 
  (h4 : remaining_customers = tables * remaining_people_per_table) 
  : n = 22 :=
  sorry

end initial_customers_count_l2_2375


namespace smallest_multiple_1_through_10_l2_2004

theorem smallest_multiple_1_through_10 : ∃ n : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ n) ∧ (∀ m : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ m) → n ≤ m) ∧ n = 27720 :=
by sorry

end smallest_multiple_1_through_10_l2_2004


namespace pure_imaginary_complex_l2_2023

theorem pure_imaginary_complex (a : ℝ) (i : ℂ) (h : i * i = -1) (p : (1 + a * i) / (1 - i) = (0 : ℂ) + b * i) :
  a = 1 := 
sorry

end pure_imaginary_complex_l2_2023


namespace point_equal_distances_l2_2463

theorem point_equal_distances (x y : ℝ) (hx : y = x) (hxy : y - 4 = -x) (hline : x + y = 4) : x = 2 :=
by sorry

end point_equal_distances_l2_2463


namespace find_total_grade10_students_l2_2614

/-
Conditions:
1. The school has a total of 1800 students in grades 10 and 11.
2. 90 students are selected as a sample for a survey.
3. The sample contains 42 grade 10 students.
-/

variables (total_students sample_size sample_grade10 total_grade10 : ℕ)

axiom total_students_def : total_students = 1800
axiom sample_size_def : sample_size = 90
axiom sample_grade10_def : sample_grade10 = 42

theorem find_total_grade10_students : total_grade10 = 840 :=
by
  have h : (sample_size : ℚ) / (total_students : ℚ) = (sample_grade10 : ℚ) / (total_grade10 : ℚ) :=
    sorry
  sorry

end find_total_grade10_students_l2_2614


namespace largest_trifecta_sum_l2_2428

def trifecta (a b c : ℕ) : Prop :=
  a < b ∧ b < c ∧ a ∣ b ∧ b ∣ c ∧ c ∣ (a * b) ∧ (100 ≤ a) ∧ (a < 1000) ∧ (100 ≤ b) ∧ (b < 1000) ∧ (100 ≤ c) ∧ (c < 1000)

theorem largest_trifecta_sum : ∃ (a b c : ℕ), trifecta a b c ∧ a + b + c = 700 :=
sorry

end largest_trifecta_sum_l2_2428


namespace perfect_play_winner_l2_2903

theorem perfect_play_winner (A B : ℕ) :
    (A = B → (∃ f : ℕ → ℕ, ∀ n, 0 < f n ∧ f n ≤ B ∧ f n = B - A → false)) ∧
    (A ≠ B → (∃ g : ℕ → ℕ, ∀ n, 0 < g n ∧ g n ≤ B ∧ g n = A - B → false)) :=
sorry

end perfect_play_winner_l2_2903


namespace chips_calories_l2_2647

-- Define the conditions
def calories_from_breakfast : ℕ := 560
def calories_from_lunch : ℕ := 780
def calories_from_cake : ℕ := 110
def calories_from_coke : ℕ := 215
def daily_calorie_limit : ℕ := 2500
def remaining_calories : ℕ := 525

-- Define the total calories consumed so far
def total_consumed : ℕ := calories_from_breakfast + calories_from_lunch + calories_from_cake + calories_from_coke

-- Define the total allowable calories without exceeding the limit
def total_allowed : ℕ := daily_calorie_limit - remaining_calories

-- Define the calories in the chips
def calories_in_chips : ℕ := total_allowed - total_consumed

-- Prove that the number of calories in the chips is 310
theorem chips_calories :
  calories_in_chips = 310 :=
by
  sorry

end chips_calories_l2_2647


namespace mrs_hilt_apple_pies_l2_2970

-- Given definitions
def total_pies := 30 * 5
def pecan_pies := 16

-- The number of apple pies
def apple_pies := total_pies - pecan_pies

-- The proof statement
theorem mrs_hilt_apple_pies : apple_pies = 134 :=
by
  sorry -- Proof step to be filled

end mrs_hilt_apple_pies_l2_2970


namespace r_needs_35_days_l2_2300

def work_rate (P Q R: ℚ) : Prop :=
  (P = Q + R) ∧ (P + Q = 1/10) ∧ (Q = 1/28)

theorem r_needs_35_days (P Q R: ℚ) (h: work_rate P Q R) : 1 / R = 35 :=
by 
  sorry

end r_needs_35_days_l2_2300


namespace curve_representation_l2_2941

def curve_set (x y : Real) : Prop := 
  ((x + y - 1) * Real.sqrt (x^2 + y^2 - 4) = 0)

def line_set (x y : Real) : Prop :=
  (x + y - 1 = 0) ∧ (x^2 + y^2 ≥ 4)

def circle_set (x y : Real) : Prop :=
  (x^2 + y^2 = 4)

theorem curve_representation (x y : Real) :
  curve_set x y ↔ (line_set x y ∨ circle_set x y) :=
sorry

end curve_representation_l2_2941


namespace find_x_pow_y_l2_2866

theorem find_x_pow_y (x y : ℝ) : |x + 2| + (y - 3)^2 = 0 → x ^ y = -8 :=
by
  sorry

end find_x_pow_y_l2_2866


namespace find_m_l2_2483

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x + m
noncomputable def g (x : ℝ) : ℝ := 2 * x - 2

theorem find_m : 
  ∃ m : ℝ, ∀ x : ℝ, f m x = g x → m = -2 := by
  sorry

end find_m_l2_2483


namespace joining_fee_per_person_l2_2250

variables (F : ℝ)
variables (family_members : ℕ) (monthly_cost_per_person : ℝ) (john_yearly_payment : ℝ)

def total_cost (F : ℝ) (family_members : ℕ) (monthly_cost_per_person : ℝ) : ℝ :=
  family_members * (F + 12 * monthly_cost_per_person)

theorem joining_fee_per_person :
  (family_members = 4) →
  (monthly_cost_per_person = 1000) →
  (john_yearly_payment = 32000) →
  john_yearly_payment = 0.5 * total_cost F family_members monthly_cost_per_person →
  F = 4000 :=
by
  intros h_family h_monthly_cost h_yearly_payment h_eq
  sorry

end joining_fee_per_person_l2_2250


namespace debate_students_handshake_l2_2425

theorem debate_students_handshake 
    (S1 S2 S3 : ℕ)
    (h1 : S1 = 2 * S2)
    (h2 : S2 = S3 + 40)
    (h3 : S3 = 200) :
    S1 + S2 + S3 = 920 :=
by
  sorry

end debate_students_handshake_l2_2425


namespace part1_distance_part2_equation_l2_2485

noncomputable section

-- Define the conditions for Part 1
def hyperbola_C1 (x y : ℝ) : Prop := (x^2 / 4) - (y^2 / 12) = 1

-- Define the point M(3, t) existing on hyperbola C₁
def point_on_hyperbola_C1 (t : ℝ) : Prop := hyperbola_C1 3 t

-- Define the right focus of hyperbola C1
def right_focus_C1 : ℝ × ℝ := (4, 0)

-- Part 1: Distance from point M to the right focus
theorem part1_distance (t : ℝ) (h : point_on_hyperbola_C1 t) :  
  let distance := Real.sqrt ((3 - 4)^2 + (t - 0)^2)
  distance = 4 := sorry

-- Define the conditions for Part 2
def hyperbola_C2 (x y : ℝ) (m : ℝ) : Prop := (x^2 / 4) - (y^2 / 12) = m

-- Define the point (-3, 2√6) existing on hyperbola C₂
def point_on_hyperbola_C2 (m : ℝ) : Prop := hyperbola_C2 (-3) (2 * Real.sqrt 6) m

-- Part 2: The standard equation of hyperbola C₂
theorem part2_equation (h : point_on_hyperbola_C2 (1/4)) : 
  ∀ (x y : ℝ), hyperbola_C2 x y (1/4) ↔ (x^2 - (y^2 / 3) = 1) := sorry

end part1_distance_part2_equation_l2_2485


namespace value_of_x2y_plus_xy2_l2_2558

-- Define variables x and y as real numbers
variables (x y : ℝ)

-- Define the conditions
def condition1 : Prop := x + y = -2
def condition2 : Prop := x * y = -3

-- Define the proof problem
theorem value_of_x2y_plus_xy2 (h1 : condition1 x y) (h2 : condition2 x y) : x^2 * y + x * y^2 = 6 := by
  sorry

end value_of_x2y_plus_xy2_l2_2558


namespace minimum_value_of_sum_l2_2613

theorem minimum_value_of_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) : 
    1 / (2 * a + b) + 1 / (2 * b + c) + 1 / (2 * c + a) >= 3 :=
by
  sorry

end minimum_value_of_sum_l2_2613


namespace convex_polygon_from_non_overlapping_rectangles_is_rectangle_l2_2934

def isConvexPolygon (P : Set Point) : Prop := sorry
def canBeFormedByNonOverlappingRectangles (P : Set Point) (rects: List (Set Point)) : Prop := sorry
def isRectangle (P : Set Point) : Prop := sorry

theorem convex_polygon_from_non_overlapping_rectangles_is_rectangle
  (P : Set Point)
  (rects : List (Set Point))
  (h_convex : isConvexPolygon P)
  (h_form : canBeFormedByNonOverlappingRectangles P rects) :
  isRectangle P :=
sorry

end convex_polygon_from_non_overlapping_rectangles_is_rectangle_l2_2934


namespace equivalent_form_l2_2020

theorem equivalent_form (p q : ℝ) (hp₁ : p ≠ 0) (hp₂ : p ≠ 5) (hq₁ : q ≠ 0) (hq₂ : q ≠ 7) :
  (3/p + 4/q = 1/3) ↔ (p = 9*q/(q - 12)) :=
by
  sorry

end equivalent_form_l2_2020


namespace jerrie_minutes_l2_2144

-- Define the conditions
def barney_situps_per_minute := 45
def carrie_situps_per_minute := 2 * barney_situps_per_minute
def jerrie_situps_per_minute := carrie_situps_per_minute + 5
def barney_total_situps := 1 * barney_situps_per_minute
def carrie_total_situps := 2 * carrie_situps_per_minute
def combined_total_situps := 510

-- Define the question and required proof
theorem jerrie_minutes :
  ∃ J : ℕ, barney_total_situps + carrie_total_situps + J * jerrie_situps_per_minute = combined_total_situps ∧ J = 3 :=
  by
  sorry

end jerrie_minutes_l2_2144


namespace solve_rebus_l2_2539

-- Definitions for the conditions
def is_digit (n : Nat) : Prop := 1 ≤ n ∧ n ≤ 9

def distinct_digits (A B C D : Nat) : Prop := 
  is_digit A ∧ is_digit B ∧ is_digit C ∧ is_digit D ∧ 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D

-- Main Statement
theorem solve_rebus (A B C D : Nat) (h_distinct : distinct_digits A B C D) 
(h_eq : 1001 * A + 100 * B + 10 * C + A = 182 * (10 * C + D)) :
  1000 * A + 100 * B + 10 * C + D = 2916 :=
by
  sorry

end solve_rebus_l2_2539


namespace average_visitors_remaining_days_l2_2261

-- Definitions
def visitors_monday := 50
def visitors_tuesday := 2 * visitors_monday
def total_week_visitors := 250
def days_remaining := 5
def remaining_visitors := total_week_visitors - (visitors_monday + visitors_tuesday)
def average_remaining_visitors_per_day := remaining_visitors / days_remaining

-- Theorem statement
theorem average_visitors_remaining_days : average_remaining_visitors_per_day = 20 :=
by
  -- Proof is skipped
  sorry

end average_visitors_remaining_days_l2_2261


namespace trip_time_is_correct_l2_2177

noncomputable def total_trip_time : ℝ :=
  let wrong_direction_time := 100 / 60
  let return_time := 100 / 45
  let detour_time := 30 / 45
  let normal_trip_time := 300 / 60
  let stop_time := 2 * (15 / 60)
  wrong_direction_time + return_time + detour_time + normal_trip_time + stop_time

theorem trip_time_is_correct : total_trip_time = 10.06 :=
  by
    -- Proof steps are omitted
    sorry

end trip_time_is_correct_l2_2177


namespace Rachel_homework_difference_l2_2655

theorem Rachel_homework_difference (m r : ℕ) (hm : m = 8) (hr : r = 14) : r - m = 6 := 
by 
  sorry

end Rachel_homework_difference_l2_2655


namespace tan_x_eq_sqrt3_l2_2000

theorem tan_x_eq_sqrt3 (x : Real) (h : Real.sin (x + 20 * Real.pi / 180) = Real.cos (x + 10 * Real.pi / 180) + Real.cos (x - 10 * Real.pi / 180)) : Real.tan x = Real.sqrt 3 := 
by
  sorry

end tan_x_eq_sqrt3_l2_2000


namespace nina_running_distance_l2_2450

theorem nina_running_distance (total_distance : ℝ) (initial_run : ℝ) (num_initial_runs : ℕ) :
  total_distance = 0.8333333333333334 →
  initial_run = 0.08333333333333333 →
  num_initial_runs = 2 →
  (total_distance - initial_run * num_initial_runs = 0.6666666666666667) :=
by
  intros h_total h_initial h_num
  sorry

end nina_running_distance_l2_2450


namespace total_toothpicks_480_l2_2076

/- Define the number of toothpicks per side -/
def toothpicks_per_side : ℕ := 15

/- Define the number of horizontal lines in the grid -/
def horizontal_lines (sides : ℕ) : ℕ := sides + 1

/- Define the number of vertical lines in the grid -/
def vertical_lines (sides : ℕ) : ℕ := sides + 1

/- Define the total number of toothpicks used -/
def total_toothpicks (sides : ℕ) : ℕ :=
  (horizontal_lines sides * toothpicks_per_side) + (vertical_lines sides * toothpicks_per_side)

/- Theorem statement: Prove that for a grid with 15 toothpicks per side, the total number of toothpicks is 480 -/
theorem total_toothpicks_480 : total_toothpicks 15 = 480 :=
  sorry

end total_toothpicks_480_l2_2076


namespace number_of_items_l2_2444

variable (s d : ℕ)
variable (total_money cost_sandwich cost_drink discount : ℝ)
variable (s_purchase_criterion : s > 5)
variable (total_money_value : total_money = 50.00)
variable (cost_sandwich_value : cost_sandwich = 6.00)
variable (cost_drink_value : cost_drink = 1.50)
variable (discount_value : discount = 5.00)

theorem number_of_items (h1 : total_money = 50.00)
(h2 : cost_sandwich = 6.00)
(h3 : cost_drink = 1.50)
(h4 : discount = 5.00)
(h5 : s > 5) :
  s + d = 9 :=
by
  sorry

end number_of_items_l2_2444


namespace manuscript_typing_total_cost_is_1400_l2_2718

-- Defining the variables and constants based on given conditions
def cost_first_time_per_page := 10
def cost_revision_per_page := 5
def total_pages := 100
def pages_revised_once := 20
def pages_revised_twice := 30
def pages_no_revision := total_pages - pages_revised_once - pages_revised_twice

-- Calculations based on the given conditions
def cost_first_time :=
  total_pages * cost_first_time_per_page

def cost_revised_once :=
  pages_revised_once * cost_revision_per_page

def cost_revised_twice :=
  pages_revised_twice * cost_revision_per_page * 2

def total_cost :=
  cost_first_time + cost_revised_once + cost_revised_twice

-- Prove that the total cost equals the calculated value
theorem manuscript_typing_total_cost_is_1400 :
  total_cost = 1400 := by
  sorry

end manuscript_typing_total_cost_is_1400_l2_2718


namespace age_of_oldest_child_l2_2472

theorem age_of_oldest_child (a1 a2 a3 x : ℕ) (h1 : a1 = 5) (h2 : a2 = 7) (h3 : a3 = 10) (h_avg : (a1 + a2 + a3 + x) / 4 = 8) : x = 10 :=
by
  sorry

end age_of_oldest_child_l2_2472


namespace find_w_squared_l2_2667

theorem find_w_squared (w : ℝ) :
  (w + 15)^2 = (4 * w + 9) * (3 * w + 6) →
  w^2 = ((-21 + Real.sqrt 7965) / 22)^2 ∨ 
        w^2 = ((-21 - Real.sqrt 7965) / 22)^2 :=
by sorry

end find_w_squared_l2_2667


namespace total_votes_cast_l2_2653

theorem total_votes_cast (S : ℝ) (x : ℝ) (h1 : S = 120) (h2 : S = 0.72 * x - 0.28 * x) : x = 273 := by
  sorry

end total_votes_cast_l2_2653


namespace segment_parallel_to_x_axis_l2_2286

theorem segment_parallel_to_x_axis 
  (f : ℤ → ℤ) 
  (hf : ∀ n, ∃ m, f n = m) 
  (a b : ℤ) 
  (h_dist : ∃ d : ℤ, d * d = (b - a) * (b - a) + (f b - f a) * (f b - f a)) : 
  f a = f b :=
sorry

end segment_parallel_to_x_axis_l2_2286


namespace percent_markdown_l2_2701

theorem percent_markdown (P S : ℝ) (h : S * 1.25 = P) : (P - S) / P * 100 = 20 := by
  sorry

end percent_markdown_l2_2701


namespace min_value_expression_l2_2577

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  8 * x^3 + 27 * y^3 + 64 * z^3 + (1 / (8 * x * y * z)) ≥ 4 :=
by
  sorry

end min_value_expression_l2_2577


namespace carpet_dimensions_l2_2278

-- Define the problem parameters
def width_a : ℕ := 50
def width_b : ℕ := 38

-- The dimensions x and y are integral numbers of feet
variables (x y : ℕ)

-- The same length L for both rooms that touches all four walls
noncomputable def length (x y : ℕ) : ℚ := (22 * (x^2 + y^2)) / (x * y)

-- The final theorem to be proven
theorem carpet_dimensions (x y : ℕ) (h : (x^2 + y^2) * 1056 = (x * y) * 48 * (length x y)) : (x = 50) ∧ (y = 25) :=
by
  sorry -- Proof is omitted

end carpet_dimensions_l2_2278


namespace length_of_BC_l2_2188

theorem length_of_BC (AB AC : ℕ) (hAB : AB = 86) (hAC : AC = 97)
    (BX CX : ℕ) (h_pow : CX * (BX + CX) = 2013) : 
    BX + CX = 61 :=
  sorry

end length_of_BC_l2_2188


namespace banana_cantaloupe_cost_l2_2266

theorem banana_cantaloupe_cost {a b c d : ℕ} 
  (h1 : a + b + c + d = 20) 
  (h2 : d = 2 * a)
  (h3 : c = a - b) : b + c = 5 :=
sorry

end banana_cantaloupe_cost_l2_2266


namespace initial_oranges_in_box_l2_2728

theorem initial_oranges_in_box (o_taken_out o_left_in_box : ℕ) (h1 : o_taken_out = 35) (h2 : o_left_in_box = 20) :
  o_taken_out + o_left_in_box = 55 := 
by
  sorry

end initial_oranges_in_box_l2_2728


namespace original_price_per_tire_l2_2341

-- Definitions derived from the problem
def number_of_tires : ℕ := 4
def sale_price_per_tire : ℝ := 75
def total_savings : ℝ := 36

-- Goal to prove the original price of each tire
theorem original_price_per_tire :
  (sale_price_per_tire + total_savings / number_of_tires) = 84 :=
by sorry

end original_price_per_tire_l2_2341


namespace simplify_complex_expr_l2_2466

theorem simplify_complex_expr : ∀ i : ℂ, i^2 = -1 → 3 * (4 - 2 * i) + 2 * i * (3 - i) = 14 :=
by 
  intro i 
  intro h
  sorry

end simplify_complex_expr_l2_2466


namespace symmetric_point_correct_l2_2690

-- Define the point and the symmetry operation
structure Point :=
  (x : ℝ)
  (y : ℝ)

def symmetric_with_respect_to_x_axis (p : Point) : Point :=
  {x := p.x, y := -p.y}

-- Define the specific point M
def M : Point := {x := 1, y := 2}

-- Define the expected answer point M'
def M' : Point := {x := 1, y := -2}

-- Prove that the symmetric point with respect to the x-axis is as expected
theorem symmetric_point_correct :
  symmetric_with_respect_to_x_axis M = M' :=
by sorry

end symmetric_point_correct_l2_2690


namespace verify_other_root_l2_2456

variable {a b c x : ℝ}

-- Given conditions
axiom distinct_non_zero_constants : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a

axiom root_two : a * 2^2 - (a + b + c) * 2 + (b + c) = 0

-- Function under test
noncomputable def other_root (a b c : ℝ) : ℝ :=
  (b + c - a) / a

-- The goal statement
theorem verify_other_root :
  ∀ (a b c : ℝ), (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a) → (a * 2^2 - (a + b + c) * 2 + (b + c) = 0) → 
  (∀ x, (a * x^2 - (a + b + c) * x + (b + c) = 0) → (x = 2 ∨ x = (b + c - a) / a)) :=
by
  intros a b c h1 h2 x h3
  sorry

end verify_other_root_l2_2456


namespace recurring_decimal_mul_seven_l2_2101

-- Declare the repeating decimal as a definition
def recurring_decimal_0_3 : ℚ := 1 / 3

-- Theorem stating that the product of 0.333... and 7 is 7/3
theorem recurring_decimal_mul_seven : recurring_decimal_0_3 * 7 = 7 / 3 :=
by
  -- Insert proof here
  sorry

end recurring_decimal_mul_seven_l2_2101


namespace crayons_birthday_l2_2426

theorem crayons_birthday (C E : ℕ) (hC : C = 523) (hE : E = 457) (hDiff : C = E + 66) : C = 523 := 
by {
  -- proof would go here
  sorry
}

end crayons_birthday_l2_2426


namespace perimeter_of_semi_circle_region_l2_2234

theorem perimeter_of_semi_circle_region (side_length : ℝ) (h : side_length = 1/π) : 
  let radius := side_length / 2
  let circumference_of_half_circle := (1 / 2) * π * side_length
  3 * circumference_of_half_circle = 3 / 2
  := by
  sorry

end perimeter_of_semi_circle_region_l2_2234


namespace point_symmetric_about_y_axis_l2_2501

theorem point_symmetric_about_y_axis (A B : ℝ × ℝ) 
  (hA : A = (1, -2)) 
  (hSym : B = (-A.1, A.2)) :
  B = (-1, -2) := 
by 
  sorry

end point_symmetric_about_y_axis_l2_2501


namespace max_dot_product_of_points_on_ellipses_l2_2611

theorem max_dot_product_of_points_on_ellipses :
  let C1 (M : ℝ × ℝ) := M.1^2 / 25 + M.2^2 / 9 = 1
  let C2 (N : ℝ × ℝ) := N.1^2 / 9 + N.2^2 / 25 = 1
  ∃ M N : ℝ × ℝ,
    C1 M ∧ C2 N ∧
    (∀ M N, C1 M ∧ C2 N → M.1 * N.1 + M.2 * N.2 ≤ 15 ∧ 
      (∃ θ φ, M = (5 * Real.cos θ, 3 * Real.sin θ) ∧ N = (3 * Real.cos φ, 5 * Real.sin φ) ∧ (M.1 * N.1 + M.2 * N.2 = 15))) :=
by
  sorry

end max_dot_product_of_points_on_ellipses_l2_2611


namespace fraction_simplification_l2_2236

theorem fraction_simplification :
  (3100 - 3037)^2 / 81 = 49 := by
  sorry

end fraction_simplification_l2_2236


namespace find_m_l2_2074

theorem find_m (m : ℝ) (h1 : m > 0) (h2 : (4 - m) / (m - 2) = 2 * m) : 
  m = (3 + Real.sqrt 41) / 4 := by
  sorry

end find_m_l2_2074


namespace valid_relationship_l2_2334

noncomputable def proof_statement (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^2 + c^2 = 2 * b * c) : Prop :=
  b > a ∧ a > c

theorem valid_relationship (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^2 + c^2 = 2 * b * c) : proof_statement a b c h_distinct h_pos h_eq :=
  sorry

end valid_relationship_l2_2334


namespace number_of_gigs_played_l2_2967

-- Definitions based on given conditions
def earnings_per_member : ℕ := 20
def number_of_members : ℕ := 4
def total_earnings : ℕ := 400

-- Proof statement in Lean 4
theorem number_of_gigs_played : (total_earnings / (earnings_per_member * number_of_members)) = 5 :=
by
  sorry

end number_of_gigs_played_l2_2967


namespace largest_possible_b_l2_2081

theorem largest_possible_b (a b c : ℕ) (h1 : 1 < c) (h2 : c < b) (h3 : b < a) (h4 : a * b * c = 360) : b = 12 :=
sorry

end largest_possible_b_l2_2081


namespace total_amount_collected_l2_2389

theorem total_amount_collected (h1 : ∀ (P_I P_II : ℕ), P_I * 50 = P_II) 
                               (h2 : ∀ (F_I F_II : ℕ), F_I = 3 * F_II) 
                               (h3 : ∀ (P_II F_II : ℕ), P_II * F_II = 1250) : 
                               ∃ (Total : ℕ), Total = 1325 :=
by
  sorry

end total_amount_collected_l2_2389


namespace geometric_sum_S5_l2_2324

variable (a_n : ℕ → ℝ)
variable (S : ℕ → ℝ)

def geometric_sequence (a_n : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a_n (n+1) = a_n n * q

theorem geometric_sum_S5 (a_n : ℕ → ℝ) (S : ℕ → ℝ)
  (h_geom : geometric_sequence a_n)
  (h_cond1 : a_n 2 * a_n 3 = 8 * a_n 1)
  (h_cond2 : (a_n 4 + 2 * a_n 5) / 2 = 20) :
  S 5 = 31 :=
sorry

end geometric_sum_S5_l2_2324


namespace total_weight_correct_l2_2530

-- Definitions for the problem conditions
def jug1_capacity : ℝ := 2
def jug2_capacity : ℝ := 3
def jug3_capacity : ℝ := 4

def fill1 : ℝ := 0.7
def fill2 : ℝ := 0.6
def fill3 : ℝ := 0.5

def density1 : ℝ := 5
def density2 : ℝ := 4
def density3 : ℝ := 3

-- The weights of the sand in each jug
def weight1 : ℝ := fill1 * jug1_capacity * density1
def weight2 : ℝ := fill2 * jug2_capacity * density2
def weight3 : ℝ := fill3 * jug3_capacity * density3

-- The total weight of the sand in all jugs
def total_weight : ℝ := weight1 + weight2 + weight3

-- The proof statement
theorem total_weight_correct : total_weight = 20.2 := by
  sorry

end total_weight_correct_l2_2530


namespace coefficients_sum_correct_l2_2697

noncomputable def poly_expr (x : ℝ) : ℝ := (x + 2)^4

def coefficients_sum (a a_1 a_2 a_3 a_4 : ℝ) : ℝ :=
  a_1 + a_2 + a_3 + a_4

theorem coefficients_sum_correct (a a_1 a_2 a_3 a_4 : ℝ) :
  poly_expr 1 = a_4 * 1 ^ 4 + a_3 * 1 ^ 3 + a_2 * 1 ^ 2 + a_1 * 1 + a →
  a = 16 → coefficients_sum a a_1 a_2 a_3 a_4 = 65 :=
by
  intro h₁ h₂
  sorry

end coefficients_sum_correct_l2_2697


namespace sin_cos_difference_l2_2738

theorem sin_cos_difference
  (θ : ℝ)
  (h1 : θ ∈ Set.Ioo 0 Real.pi)
  (h2 : Real.sin θ + Real.cos θ = 1 / 5) :
  Real.sin θ - Real.cos θ = 7 / 5 :=
sorry

end sin_cos_difference_l2_2738


namespace gcd_sum_abcde_edcba_l2_2496

-- Definition to check if digits are consecutive
def consecutive_digits (a b c d e : ℤ) : Prop :=
  b = a + 1 ∧ c = a + 2 ∧ d = a + 3 ∧ e = a + 4

-- Definition of the five-digit number in the form abcde
def abcde (a b c d e : ℤ) : ℤ :=
  10000 * a + 1000 * b + 100 * c + 10 * d + e

-- Definition of the five-digit number in the form edcba
def edcba (a b c d e : ℤ) : ℤ :=
  10000 * e + 1000 * d + 100 * c + 10 * b + a

-- Definition which sums both abcde and edcba
def sum_abcde_edcba (a b c d e : ℤ) : ℤ :=
  abcde a b c d e + edcba a b c d e

-- Lean theorem statement for the problem
theorem gcd_sum_abcde_edcba (a b c d e : ℤ) (h : consecutive_digits a b c d e) :
  Int.gcd (sum_abcde_edcba a b c d e) 11211 = 11211 :=
by
  sorry

end gcd_sum_abcde_edcba_l2_2496


namespace inequality_proof_l2_2058

theorem inequality_proof (x y : ℝ) (h : 2 * y + 5 * x = 10) : (3 * x * y - x^2 - y^2 < 7) :=
sorry

end inequality_proof_l2_2058


namespace x_in_A_neither_sufficient_nor_necessary_for_x_in_B_l2_2158

def A : Set ℝ := {x | 0 < x ∧ x ≤ 1}
def B : Set ℝ := {x | 0 ≤ x ∧ x < 1}

theorem x_in_A_neither_sufficient_nor_necessary_for_x_in_B : ¬ ((∀ x, x ∈ A → x ∈ B) ∧ (∀ x, x ∈ B → x ∈ A)) := by
  sorry

end x_in_A_neither_sufficient_nor_necessary_for_x_in_B_l2_2158


namespace mod_congruence_l2_2695

theorem mod_congruence (N : ℕ) (hN : N > 1) (h1 : 69 % N = 90 % N) (h2 : 90 % N = 125 % N) : 81 % N = 4 := 
by {
    sorry
}

end mod_congruence_l2_2695


namespace inequality_a2b3c_l2_2183

theorem inequality_a2b3c {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : 1 / a + 1 / (2 * b) + 1 / (3 * c) = 1) : 
  a + 2 * b + 3 * c ≥ 9 :=
sorry

end inequality_a2b3c_l2_2183


namespace percent_decrease_l2_2112

theorem percent_decrease (p_original p_sale : ℝ) (h₁ : p_original = 100) (h₂ : p_sale = 50) :
  ((p_original - p_sale) / p_original * 100) = 50 := by
  sorry

end percent_decrease_l2_2112


namespace find_avg_mpg_first_car_l2_2790

def avg_mpg_first_car (x : ℝ) : Prop :=
  let miles_per_month := 450 / 3
  let gallons_first_car := miles_per_month / x
  let gallons_second_car := miles_per_month / 10
  let gallons_third_car := miles_per_month / 15
  let total_gallons := 56 / 2
  gallons_first_car + gallons_second_car + gallons_third_car = total_gallons

theorem find_avg_mpg_first_car : avg_mpg_first_car 50 :=
  sorry

end find_avg_mpg_first_car_l2_2790


namespace parallel_vectors_m_eq_neg3_l2_2313

theorem parallel_vectors_m_eq_neg3 {m : ℝ} :
  let a := (1, -2)
  let b := (1 + m, 1 - m)
  (a.1 * b.2 =  a.2 * b.1) → m = -3 :=
by 
  let a := (1, -2)
  let b := (1 + m, 1 - m)
  intro h
  sorry

end parallel_vectors_m_eq_neg3_l2_2313


namespace solve_for_x_l2_2140

theorem solve_for_x (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 :=
by
  sorry

end solve_for_x_l2_2140


namespace find_alpha_plus_beta_l2_2991

variable (α β : ℝ)

def condition_1 : Prop := α^3 - 3*α^2 + 5*α = 1
def condition_2 : Prop := β^3 - 3*β^2 + 5*β = 5

theorem find_alpha_plus_beta (h1 : condition_1 α) (h2 : condition_2 β) : α + β = 2 := 
  sorry

end find_alpha_plus_beta_l2_2991


namespace sample_capacity_is_480_l2_2658

-- Problem conditions
def total_people : ℕ := 500 + 400 + 300
def selection_probability : ℝ := 0.4

-- Statement: Prove that sample capacity n equals 480
theorem sample_capacity_is_480 (n : ℕ) (h : n / total_people = selection_probability) : n = 480 := by
  sorry

end sample_capacity_is_480_l2_2658


namespace money_brought_to_store_l2_2570

theorem money_brought_to_store : 
  let sheet_cost := 42
  let rope_cost := 18
  let propane_and_burner_cost := 14
  let helium_cost_per_ounce := 1.5
  let height_per_ounce := 113
  let max_height := 9492
  let total_item_cost := sheet_cost + rope_cost + propane_and_burner_cost
  let helium_needed := max_height / height_per_ounce
  let helium_total_cost := helium_needed * helium_cost_per_ounce
  total_item_cost + helium_total_cost = 200 :=
by
  sorry

end money_brought_to_store_l2_2570


namespace find_negative_integer_l2_2716

theorem find_negative_integer (N : ℤ) (h : N^2 + N = -12) : N = -4 := 
by sorry

end find_negative_integer_l2_2716


namespace cubic_polynomial_solution_l2_2831

theorem cubic_polynomial_solution (x : ℝ) :
  x^3 + 6*x^2 + 11*x + 6 = 12 ↔ x = -1 ∨ x = -2 ∨ x = -3 := by
  sorry

end cubic_polynomial_solution_l2_2831


namespace find_y_coordinate_l2_2512

theorem find_y_coordinate (y : ℝ) (h : y > 0) (dist_eq : (10 - 2)^2 + (y - 5)^2 = 13^2) : y = 16 :=
by
  sorry

end find_y_coordinate_l2_2512


namespace compute_expression_l2_2038

theorem compute_expression :
  21 * 47 + 21 * 53 = 2100 := 
by
  sorry

end compute_expression_l2_2038


namespace find_k_value_l2_2162

variable (x y z k : ℝ)

theorem find_k_value (h : 7 / (x + y) = k / (x + z) ∧ k / (x + z) = 11 / (z - y)) :
  k = 18 :=
sorry

end find_k_value_l2_2162


namespace age_ratio_in_two_years_l2_2542

variable (S M : ℕ)

-- Conditions
def sonCurrentAge : Prop := S = 18
def manCurrentAge : Prop := M = S + 20
def multipleCondition : Prop := ∃ k : ℕ, M + 2 = k * (S + 2)

-- Statement to prove
theorem age_ratio_in_two_years (h1 : sonCurrentAge S) (h2 : manCurrentAge S M) (h3 : multipleCondition S M) : 
  (M + 2) / (S + 2) = 2 := 
by
  sorry

end age_ratio_in_two_years_l2_2542


namespace find_second_term_of_ratio_l2_2127

theorem find_second_term_of_ratio
  (a b c d : ℕ)
  (h1 : a = 6)
  (h2 : b = 7)
  (h3 : c = 3)
  (h4 : (a - c) * 4 < a * d) :
  d = 5 :=
by
  sorry

end find_second_term_of_ratio_l2_2127


namespace tina_coins_after_five_hours_l2_2063

theorem tina_coins_after_five_hours :
  let coins_in_first_hour := 20
  let coins_in_second_hour := 30
  let coins_in_third_hour := 30
  let coins_in_fourth_hour := 40
  let coins_taken_out_in_fifth_hour := 20
  let total_coins_after_five_hours := coins_in_first_hour + coins_in_second_hour + coins_in_third_hour + coins_in_fourth_hour - coins_taken_out_in_fifth_hour
  total_coins_after_five_hours = 100 :=
by {
  sorry
}

end tina_coins_after_five_hours_l2_2063


namespace exists_unique_pair_l2_2829

theorem exists_unique_pair (X : Set ℤ) :
  (∀ n : ℤ, ∃! (a b : ℤ), a ∈ X ∧ b ∈ X ∧ a + 2 * b = n) :=
sorry

end exists_unique_pair_l2_2829


namespace largest_a_pow_b_l2_2279

theorem largest_a_pow_b (a b : ℕ) (h_pos_a : 1 < a) (h_pos_b : 1 < b) (h_eq : a^b * b^a + a^b + b^a = 5329) : 
  a^b = 64 :=
by
  sorry

end largest_a_pow_b_l2_2279


namespace roberto_current_salary_l2_2040

theorem roberto_current_salary (starting_salary current_salary : ℝ) (h₀ : starting_salary = 80000)
(h₁ : current_salary = (starting_salary * 1.4) * 1.2) : 
current_salary = 134400 := by
  sorry

end roberto_current_salary_l2_2040


namespace science_books_initially_l2_2760

def initial_number_of_books (borrowed left : ℕ) : ℕ := 
borrowed + left

theorem science_books_initially (borrowed left : ℕ) (h1 : borrowed = 18) (h2 : left = 57) :
initial_number_of_books borrowed left = 75 := by
sorry

end science_books_initially_l2_2760


namespace average_earnings_per_minute_l2_2422

theorem average_earnings_per_minute 
  (laps : ℕ) (meters_per_lap : ℕ) (dollars_per_100_meters : ℝ) (total_minutes : ℕ) (total_laps : ℕ)
  (h_laps : total_laps = 24)
  (h_meters_per_lap : meters_per_lap = 100)
  (h_dollars_per_100_meters : dollars_per_100_meters = 3.5)
  (h_total_minutes : total_minutes = 12)
  : (total_laps * meters_per_lap / 100 * dollars_per_100_meters / total_minutes) = 7 := 
by
  sorry

end average_earnings_per_minute_l2_2422


namespace angle_quadrant_l2_2050

theorem angle_quadrant 
  (θ : Real) 
  (h1 : Real.cos θ > 0) 
  (h2 : Real.sin (2 * θ) < 0) : 
  3 * π / 2 < θ ∧ θ < 2 * π := 
by
  sorry

end angle_quadrant_l2_2050


namespace basketball_free_throws_l2_2785

theorem basketball_free_throws (a b x : ℕ) 
  (h1 : 3 * b = 4 * a) 
  (h2 : x = 2 * a) 
  (h3 : 2 * a + 3 * b + x = 72) : 
  x = 18 := 
sorry

end basketball_free_throws_l2_2785


namespace segment_AB_length_l2_2306

-- Defining the conditions
def area_ratio (AB CD : ℝ) : Prop := AB / CD = 5 / 2
def length_sum (AB CD : ℝ) : Prop := AB + CD = 280

-- The theorem stating the problem
theorem segment_AB_length (AB CD : ℝ) (h₁ : area_ratio AB CD) (h₂ : length_sum AB CD) : AB = 200 :=
by {
  -- Proof step would be inserted here, but it is omitted as per instructions
  sorry
}

end segment_AB_length_l2_2306


namespace count_integer_radii_l2_2093

theorem count_integer_radii (r : ℕ) (h : r < 150) :
  (∃ n : ℕ, n = 11 ∧ (∀ r, 0 < r ∧ r < 150 → (150 % r = 0)) ∧ (r ≠ 150)) := sorry

end count_integer_radii_l2_2093


namespace repeating_decimal_to_fraction_l2_2685

/--
Express \(2.\overline{06}\) as a reduced fraction, given that \(0.\overline{01} = \frac{1}{99}\)
-/
theorem repeating_decimal_to_fraction : 
  (0.01:ℚ) = 1 / 99 → (2.06:ℚ) = 68 / 33 := 
by 
  sorry 

end repeating_decimal_to_fraction_l2_2685


namespace no_solutions_l2_2047

theorem no_solutions (x y : ℕ) (hx : x ≥ 1) (hy : y ≥ 1) : ¬ (x^5 = y^2 + 4) :=
by sorry

end no_solutions_l2_2047


namespace concert_songs_l2_2452

def total_songs (g : ℕ) : ℕ := (9 + 3 + 9 + g) / 3

theorem concert_songs 
  (g : ℕ) 
  (h1 : 9 + 3 + 9 + g = 3 * total_songs g) 
  (h2 : 3 + g % 4 = 0) 
  (h3 : 4 ≤ g ∧ g ≤ 9) 
  : total_songs g = 9 ∨ total_songs g = 10 := 
sorry

end concert_songs_l2_2452


namespace triangle_shape_l2_2258

theorem triangle_shape (a b : ℝ) (A B : ℝ) (hA : 0 < A) (hB : A < π) (h : a * Real.cos A = b * Real.cos B) :
  (A = B ∨ A + B = π / 2 ∨ a = b) :=
by
  sorry

end triangle_shape_l2_2258


namespace calculate_f_g_l2_2385

noncomputable def f (x : ℕ) : ℕ := 4 * x + 3
noncomputable def g (x : ℕ) : ℕ := (x + 2) ^ 2

theorem calculate_f_g : f (g 3) = 103 :=
by 
  -- Proof omitted.
  sorry

end calculate_f_g_l2_2385


namespace simple_interest_rate_l2_2641

theorem simple_interest_rate (P R : ℝ) (T : ℕ) (hT : T = 10) (h_double : P * 2 = P + P * R * T / 100) : R = 10 :=
by
  sorry

end simple_interest_rate_l2_2641


namespace union_of_A_and_B_l2_2763

def A : Set ℤ := {0, 1}
def B : Set ℤ := {-1, 1}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1} := by
  sorry

end union_of_A_and_B_l2_2763


namespace shape_formed_is_line_segment_l2_2315

def point := (ℝ × ℝ)

noncomputable def A : point := (0, 0)
noncomputable def B : point := (0, 4)
noncomputable def C : point := (6, 4)
noncomputable def D : point := (6, 0)

noncomputable def line_eq (p1 p2 : point) : ℝ × ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  (x2 - x1, y2 - y1)

theorem shape_formed_is_line_segment :
  let l1 := line_eq A (1, 1)  -- Line from A at 45°
  let l2 := line_eq B (-1, -1) -- Line from B at -45°
  let l3 := line_eq D (1, -1) -- Line from D at 45°
  let l4 := line_eq C (-1, 5) -- Line from C at -45°
  let intersection1 := (5, 5)  -- Intersection of l1 and l4: solve x = 10 - x
  let intersection2 := (5, -1)  -- Intersection of l2 and l3: solve 4 - x = x - 6
  intersection1.1 = intersection2.1 := 
by
  sorry

end shape_formed_is_line_segment_l2_2315


namespace find_num_apples_l2_2034

def num_apples (A P : ℕ) : Prop :=
  P = (3 * A) / 5 ∧ A + P = 240

theorem find_num_apples (A : ℕ) (P : ℕ) :
  num_apples A P → A = 150 :=
by
  intros h
  -- sorry for proof
  sorry

end find_num_apples_l2_2034


namespace parallelogram_area_l2_2885

open Matrix

noncomputable def u : Fin 2 → ℝ := ![7, -4]
noncomputable def z : Fin 2 → ℝ := ![8, -1]

theorem parallelogram_area :
  let matrix := ![u, z]
  |det (of fun (i j : Fin 2) => (matrix i) j)| = 25 :=
by
  sorry

end parallelogram_area_l2_2885


namespace yoki_cans_correct_l2_2546

def total_cans := 85
def ladonna_cans := 25
def prikya_cans := 2 * ladonna_cans
def yoki_cans := total_cans - ladonna_cans - prikya_cans

theorem yoki_cans_correct : yoki_cans = 10 :=
by
  sorry

end yoki_cans_correct_l2_2546


namespace jordan_more_novels_than_maxime_l2_2215

def jordan_french_novels : ℕ := 130
def jordan_spanish_novels : ℕ := 20

def alexandre_french_novels : ℕ := jordan_french_novels / 10
def alexandre_spanish_novels : ℕ := 3 * jordan_spanish_novels

def camille_french_novels : ℕ := 2 * alexandre_french_novels
def camille_spanish_novels : ℕ := jordan_spanish_novels / 2

def total_french_novels : ℕ := jordan_french_novels + alexandre_french_novels + camille_french_novels

def maxime_french_novels : ℕ := total_french_novels / 2 - 5
def maxime_spanish_novels : ℕ := 2 * camille_spanish_novels

def jordan_total_novels : ℕ := jordan_french_novels + jordan_spanish_novels
def maxime_total_novels : ℕ := maxime_french_novels + maxime_spanish_novels

def novels_difference : ℕ := jordan_total_novels - maxime_total_novels

theorem jordan_more_novels_than_maxime : novels_difference = 51 :=
sorry

end jordan_more_novels_than_maxime_l2_2215


namespace negative_solution_range_l2_2593

theorem negative_solution_range (m x : ℝ) (h : (2 * x + m) / (x - 1) = 1) (hx : x < 0) : m > -1 :=
  sorry

end negative_solution_range_l2_2593


namespace find_positive_integer_triples_l2_2207

-- Define the condition for the integer divisibility problem
def is_integer_division (t a b : ℕ) : Prop :=
  (t ^ (a + b) + 1) % (t ^ a + t ^ b + 1) = 0

-- Statement of the theorem
theorem find_positive_integer_triples :
  ∀ (t a b : ℕ), t > 0 → a > 0 → b > 0 → is_integer_division t a b → (t, a, b) = (2, 1, 1) :=
by
  intros t a b t_pos a_pos b_pos h
  sorry

end find_positive_integer_triples_l2_2207


namespace cosine_F_in_triangle_DEF_l2_2919

theorem cosine_F_in_triangle_DEF
  (D E F : ℝ)
  (h_triangle : D + E + F = π)
  (sin_D : Real.sin D = 4 / 5)
  (cos_E : Real.cos E = 12 / 13) :
  Real.cos F = - (16 / 65) := by
  sorry

end cosine_F_in_triangle_DEF_l2_2919


namespace student_solved_correctly_l2_2033

theorem student_solved_correctly (x : ℕ) :
  (x + 2 * x = 36) → x = 12 :=
by
  intro h
  sorry

end student_solved_correctly_l2_2033


namespace find_bottle_caps_l2_2330

variable (B : ℕ) -- Number of bottle caps Danny found at the park.

-- Conditions
variable (current_wrappers : ℕ := 67) -- Danny has 67 wrappers in his collection now.
variable (current_bottle_caps : ℕ := 35) -- Danny has 35 bottle caps in his collection now.
variable (found_wrappers : ℕ := 18) -- Danny found 18 wrappers at the park.
variable (more_wrappers_than_bottle_caps : ℕ := 32) -- Danny has 32 more wrappers than bottle caps.

-- Given the conditions, prove that Danny found 18 bottle caps at the park.
theorem find_bottle_caps (h1 : current_wrappers = current_bottle_caps + more_wrappers_than_bottle_caps)
                         (h2 : current_bottle_caps - B + found_wrappers = current_wrappers - more_wrappers_than_bottle_caps - B) :
  B = 18 :=
by
  sorry

end find_bottle_caps_l2_2330


namespace journey_distance_last_day_l2_2923

theorem journey_distance_last_day (S₆ : ℕ) (q : ℝ) (n : ℕ) (a₁ : ℝ) : 
  S₆ = 378 ∧ q = 1 / 2 ∧ n = 6 ∧ S₆ = a₁ * (1 - q^n) / (1 - q)
  → a₁ * q^(n - 1) = 6 :=
by
  intro h
  sorry

end journey_distance_last_day_l2_2923


namespace propositions_A_and_D_true_l2_2235

theorem propositions_A_and_D_true :
  (∀ x : ℝ, x^2 - 4*x + 5 > 0) ∧ (∃ x : ℤ, 3*x^2 - 2*x - 1 = 0) :=
by
  sorry

end propositions_A_and_D_true_l2_2235


namespace brother_age_l2_2181

variables (M B : ℕ)

theorem brother_age (h1 : M = B + 12) (h2 : M + 2 = 2 * (B + 2)) : B = 10 := by
  sorry

end brother_age_l2_2181


namespace power_complex_l2_2700

theorem power_complex (a b : ℝ) (i : ℂ) (h1 : i^2 = -1) (h2 : -64 = (-4)^3) (h3 : (a^b)^((3:ℝ) / 2) = a^(b * ((3:ℝ) / 2))) (h4 : (-4:ℂ)^(1/2) = 2 * i) :
  (↑(-64):ℂ) ^ (3/2) = 512 * i :=
by
  sorry

end power_complex_l2_2700


namespace Total_marbles_equal_231_l2_2403

def Connie_marbles : Nat := 39
def Juan_marbles : Nat := Connie_marbles + 25
def Maria_marbles : Nat := 2 * Juan_marbles
def Total_marbles : Nat := Connie_marbles + Juan_marbles + Maria_marbles

theorem Total_marbles_equal_231 : Total_marbles = 231 := sorry

end Total_marbles_equal_231_l2_2403


namespace tax_rate_as_percent_l2_2874

def TaxAmount (amount : ℝ) : Prop := amount = 82
def BaseAmount (amount : ℝ) : Prop := amount = 100

theorem tax_rate_as_percent {tax_amt base_amt : ℝ} 
  (h_tax : TaxAmount tax_amt) (h_base : BaseAmount base_amt) : 
  (tax_amt / base_amt) * 100 = 82 := 
by 
  sorry

end tax_rate_as_percent_l2_2874


namespace Raine_steps_to_school_l2_2232

-- Define Raine's conditions
variable (steps_total : ℕ) (days : ℕ) (round_trip_steps : ℕ)

-- Given conditions
def Raine_conditions := steps_total = 1500 ∧ days = 5 ∧ round_trip_steps = steps_total / days

-- Prove that the steps to school is 150 given Raine's conditions
theorem Raine_steps_to_school (h : Raine_conditions 1500 5 300) : (300 / 2) = 150 :=
by
  sorry

end Raine_steps_to_school_l2_2232


namespace find_legs_of_triangle_l2_2233

-- Definition of the problem conditions
def right_triangle (x y : ℝ) := x * y = 200 ∧ 4 * (y - 4) = 8 * (x - 8)

-- Theorem we want to prove
theorem find_legs_of_triangle : 
  ∃ (x y : ℝ), right_triangle x y ∧ ((x = 40 ∧ y = 5) ∨ (x = 10 ∧ y = 20)) :=
by
  sorry

end find_legs_of_triangle_l2_2233


namespace number_of_attempted_problems_l2_2120

-- Lean statement to define the problem setup
def student_assignment_problem (x y : ℕ) : Prop :=
  8 * x - 5 * y = 13 ∧ x + y ≤ 20

-- The Lean statement asserting the solution to the problem
theorem number_of_attempted_problems : ∃ x y : ℕ, student_assignment_problem x y ∧ x + y = 13 := 
by
  sorry

end number_of_attempted_problems_l2_2120


namespace solve_for_y_l2_2208

theorem solve_for_y (y : ℚ) : y - 1 / 2 = 1 / 6 - 2 / 3 + 1 / 4 → y = 1 / 4 := by
  intro h
  sorry

end solve_for_y_l2_2208


namespace upgraded_fraction_l2_2407

theorem upgraded_fraction (N U : ℕ) (h1 : ∀ (k : ℕ), k = 24)
  (h2 : ∀ (n : ℕ), N = n) (h3 : ∀ (u : ℕ), U = u)
  (h4 : N = U / 8) : U / (24 * N + U) = 1 / 4 := by
  sorry

end upgraded_fraction_l2_2407


namespace geometric_sequence_sum_l2_2786

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ)
  (h_geometric : ∀ n, a (n + 1) = r * a n)
  (h_sum1 : a 1 + a 2 = 40)
  (h_sum2 : a 3 + a 4 = 60) :
  a 5 + a 6 = 90 :=
sorry

end geometric_sequence_sum_l2_2786


namespace illuminated_area_correct_l2_2812

noncomputable def cube_illuminated_area (a ρ : ℝ) (h₁ : a = 1 / Real.sqrt 2) (h₂ : ρ = Real.sqrt (2 - Real.sqrt 3)) : ℝ :=
  (Real.sqrt 3 - 3 / 2) * (Real.pi + 3)

theorem illuminated_area_correct :
  cube_illuminated_area (1 / Real.sqrt 2) (Real.sqrt (2 - Real.sqrt 3)) (by norm_num) (by norm_num) = (Real.sqrt 3 - 3 / 2) * (Real.pi + 3) :=
sorry

end illuminated_area_correct_l2_2812


namespace minimum_value_expression_l2_2323

theorem minimum_value_expression :
  ∃ x y : ℝ, (∀ a b : ℝ, (a^2 + 4*a*b + 5*b^2 - 8*a - 6*b) ≥ -41) ∧ (x^2 + 4*x*y + 5*y^2 - 8*x - 6*y) = -41 := 
sorry

end minimum_value_expression_l2_2323


namespace positive_integer_fraction_l2_2027

theorem positive_integer_fraction (p : ℕ) (h1 : p > 0) (h2 : (3 * p + 25) / (2 * p - 5) > 0) :
  3 ≤ p ∧ p ≤ 35 :=
by
  sorry

end positive_integer_fraction_l2_2027


namespace remainder_is_one_l2_2567

theorem remainder_is_one (dividend divisor quotient remainder : ℕ) 
  (h1 : dividend = 222) 
  (h2 : divisor = 13)
  (h3 : quotient = 17)
  (h4 : dividend = divisor * quotient + remainder) : remainder = 1 :=
sorry

end remainder_is_one_l2_2567


namespace quadratic_has_real_root_l2_2075

theorem quadratic_has_real_root (a b : ℝ) : ¬ (∀ x : ℝ, x^2 + a * x + b ≠ 0) → ∃ x : ℝ, x^2 + a * x + b = 0 := 
by
  sorry

end quadratic_has_real_root_l2_2075


namespace solve_equation_l2_2837

theorem solve_equation (x : ℝ) (h : x ≠ 1) :
  (3 * x + 6) / (x^2 + 5 * x - 6) = (3 - x) / (x - 1) → x = -4 :=
by
  intro hyp
  sorry

end solve_equation_l2_2837


namespace intersection_complement_l2_2351

open Set

def U : Set ℤ := univ
def M : Set ℤ := {1, 2}
def P : Set ℤ := {-2, -1, 0, 1, 2}

theorem intersection_complement :
  P ∩ (U \ M) = {-2, -1, 0} :=
by
  sorry

end intersection_complement_l2_2351


namespace range_f_real_l2_2549

noncomputable def f (a : ℝ) (x : ℝ) :=
  if x > 1 then (a ^ x) else (4 - a / 2) * x + 2

theorem range_f_real (a : ℝ) :
  (∀ y, ∃ x, f a x = y) ↔ (1 < a ∧ a ≤ 4) :=
by
  sorry

end range_f_real_l2_2549


namespace sequence_problem_l2_2016

noncomputable def b_n (n : ℕ) : ℝ := 5 * (5/3)^(n-2)

theorem sequence_problem 
  (a_n : ℕ → ℝ) 
  (d : ℝ) 
  (h1 : ∀ n, a_n (n + 1) = a_n n + d)
  (h2 : d ≠ 0)
  (h3 : a_n 8 = a_n 5 + 3 * d)
  (h4 : a_n 13 = a_n 8 + 5 * d)
  (b_2 : ℝ)
  (hb2 : b_2 = 5)
  (h5 : ∀ n, b_n n = (match n with | 2 => b_2 | _ => sorry))
  (conseq_terms : ∀ (n : ℕ), (a_n 5 + 3 * d)^2 = a_n 5 * (a_n 5 + 8 * d)) 
  : ∀ n, b_n n = b_n 2 * (5/3)^(n-2) := 
by 
  sorry

end sequence_problem_l2_2016


namespace kathleen_remaining_money_l2_2291

-- Define the conditions
def saved_june := 21
def saved_july := 46
def saved_august := 45
def spent_school_supplies := 12
def spent_clothes := 54
def aunt_gift_threshold := 125
def aunt_gift := 25

-- Prove that Kathleen has the correct remaining amount of money
theorem kathleen_remaining_money : 
    (saved_june + saved_july + saved_august) - 
    (spent_school_supplies + spent_clothes) = 46 := 
by
  sorry

end kathleen_remaining_money_l2_2291


namespace sum_polynomial_coefficients_l2_2670

theorem sum_polynomial_coefficients :
  let a := 1
  let a_sum := -2
  (2009 * a + a_sum) = 2007 :=
by
  sorry

end sum_polynomial_coefficients_l2_2670


namespace problem_A_inter_B_empty_l2_2987

section

def set_A : Set ℝ := {x | |x| ≥ 2}
def set_B : Set ℝ := {x | -1 < x ∧ x < 2}

theorem problem_A_inter_B_empty : set_A ∩ set_B = ∅ := 
  sorry

end

end problem_A_inter_B_empty_l2_2987


namespace max_value_range_l2_2097

theorem max_value_range (a : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ)
  (h_deriv : ∀ x, f' x = a * (x - 1) * (x - a))
  (h_max : ∀ x, (x = a → (∀ y, f y ≤ f x))) : 0 < a ∧ a < 1 :=
sorry

end max_value_range_l2_2097


namespace mat_weavers_equiv_l2_2791

theorem mat_weavers_equiv {x : ℕ} 
  (h1 : 4 * 1 = 4) 
  (h2 : 16 * (64 / 16) = 64) 
  (h3 : 1 = 64 / (16 * x)) : x = 4 :=
by
  sorry

end mat_weavers_equiv_l2_2791


namespace dodecahedron_decagon_area_sum_l2_2527

theorem dodecahedron_decagon_area_sum {a b c : ℕ} (h1 : Nat.Coprime a c) (h2 : b ≠ 0) (h3 : ¬ ∃ p : ℕ, p.Prime ∧ p * p ∣ b) 
  (area_eq : (5 + 5 * Real.sqrt 5) / 4 = (a * Real.sqrt b) / c) : a + b + c = 14 :=
sorry

end dodecahedron_decagon_area_sum_l2_2527


namespace wednesday_tips_value_l2_2247

-- Definitions for the conditions
def hourly_wage : ℕ := 10
def monday_hours : ℕ := 7
def tuesday_hours : ℕ := 5
def wednesday_hours : ℕ := 7
def monday_tips : ℕ := 18
def tuesday_tips : ℕ := 12
def total_earnings : ℕ := 240

-- Hourly earnings
def monday_earnings := monday_hours * hourly_wage
def tuesday_earnings := tuesday_hours * hourly_wage
def wednesday_earnings := wednesday_hours * hourly_wage

-- Total wage earnings
def total_wage_earnings := monday_earnings + tuesday_earnings + wednesday_earnings

-- Total earnings with known tips
def known_earnings := total_wage_earnings + monday_tips + tuesday_tips

-- Prove that Wednesday tips is $20
theorem wednesday_tips_value : (total_earnings - known_earnings) = 20 := by
  sorry

end wednesday_tips_value_l2_2247


namespace count_solutions_l2_2429

theorem count_solutions :
  ∃ (n : ℕ), (∀ (x y z : ℕ), x * y * z + x * y + y * z + z * x + x + y + z = 2012 ↔ n = 27) :=
sorry

end count_solutions_l2_2429


namespace susan_remaining_spaces_l2_2314

def susan_first_turn_spaces : ℕ := 15
def susan_second_turn_spaces : ℕ := 7 - 5
def susan_third_turn_spaces : ℕ := 20
def susan_fourth_turn_spaces : ℕ := 0
def susan_fifth_turn_spaces : ℕ := 10 - 8
def susan_sixth_turn_spaces : ℕ := 0
def susan_seventh_turn_roll : ℕ := 6
def susan_seventh_turn_spaces : ℕ := susan_seventh_turn_roll * 2
def susan_total_moved_spaces : ℕ := susan_first_turn_spaces + susan_second_turn_spaces + susan_third_turn_spaces + susan_fourth_turn_spaces + susan_fifth_turn_spaces + susan_sixth_turn_spaces + susan_seventh_turn_spaces
def game_total_spaces : ℕ := 100

theorem susan_remaining_spaces : susan_total_moved_spaces = 51 ∧ (game_total_spaces - susan_total_moved_spaces) = 49 := by
  sorry

end susan_remaining_spaces_l2_2314


namespace passing_marks_l2_2677

-- Define the conditions and prove P = 160 given these conditions
theorem passing_marks (T P : ℝ) (h1 : 0.40 * T = P - 40) (h2 : 0.60 * T = P + 20) : P = 160 :=
by
  sorry

end passing_marks_l2_2677


namespace product_of_numbers_l2_2554

-- Definitions of the conditions
variables (x y : ℝ)

-- The conditions themselves
def cond1 : Prop := x + y = 20
def cond2 : Prop := x^2 + y^2 = 200

-- Statement of the proof problem
theorem product_of_numbers (h1 : cond1 x y) (h2 : cond2 x y) : x * y = 100 :=
sorry

end product_of_numbers_l2_2554


namespace syllogism_arrangement_l2_2715

theorem syllogism_arrangement : 
  (∀ n : ℕ, Odd n → ¬ (n % 2 = 0)) → 
  Odd 2013 → 
  (¬ (2013 % 2 = 0)) :=
by
  intros h1 h2
  exact h1 2013 h2

end syllogism_arrangement_l2_2715


namespace average_age_before_new_students_joined_l2_2064

/-
Problem: Given that the original strength of the class was 18, 
18 new students with an average age of 32 years joined the class, 
and the average age decreased by 4 years, prove that 
the average age of the class before the new students joined was 40 years.
-/

def original_strength := 18
def new_students := 18
def average_age_new_students := 32
def decrease_in_average_age := 4
def original_average_age := 40

theorem average_age_before_new_students_joined :
  (original_strength * original_average_age + new_students * average_age_new_students) / (original_strength + new_students) = original_average_age - decrease_in_average_age :=
by
  sorry

end average_age_before_new_students_joined_l2_2064


namespace greatest_percentage_increase_l2_2130

def pop1970_F := 30000
def pop1980_F := 45000
def pop1970_G := 60000
def pop1980_G := 75000
def pop1970_H := 40000
def pop1970_I := 20000
def pop1980_combined_H := 70000
def pop1970_J := 90000
def pop1980_J := 120000

def percentage_increase (pop1970 pop1980 : ℕ) : ℚ :=
  ((pop1980 - pop1970 : ℚ) / pop1970) * 100

theorem greatest_percentage_increase :
  ∀ (city : ℕ), (city = pop1970_F -> percentage_increase pop1970_F pop1980_F >= percentage_increase pop1970_G pop1980_G) ∧
                (city = pop1970_F -> percentage_increase pop1970_F pop1980_F >= percentage_increase (pop1970_H + pop1970_I) pop1980_combined_H) ∧
                (city = pop1970_F -> percentage_increase pop1970_F pop1980_F >= percentage_increase pop1970_J pop1980_J) := by 
  sorry

end greatest_percentage_increase_l2_2130


namespace percentage_difference_l2_2891

theorem percentage_difference:
  let x1 := 0.4 * 60
  let x2 := 0.8 * 25
  x1 - x2 = 4 :=
by
  sorry

end percentage_difference_l2_2891


namespace min_cost_to_form_closed_chain_l2_2536

/-- Definition for the cost model -/
def cost_separate_link : ℕ := 1
def cost_attach_link : ℕ := 2
def total_cost (n : ℕ) : ℕ := n * (cost_separate_link + cost_attach_link)

-- Number of pieces of gold chain and links in each chain
def num_pieces : ℕ := 13

/-- Minimum cost calculation proof statement -/
theorem min_cost_to_form_closed_chain : total_cost (num_pieces - 1) = 36 := 
by
  sorry

end min_cost_to_form_closed_chain_l2_2536


namespace ann_age_l2_2946

theorem ann_age {a b y : ℕ} (h1 : a + b = 44) (h2 : y = a - b) (h3 : b = a / 2 + 2 * (a - b)) : a = 24 :=
by
  sorry

end ann_age_l2_2946


namespace corresponding_angles_equal_l2_2521

-- Definition: Corresponding angles and their equality
def corresponding_angles (α β : ℝ) : Prop :=
  -- assuming definition of corresponding angles can be defined
  sorry

theorem corresponding_angles_equal {α β : ℝ} (h : corresponding_angles α β) : α = β :=
by
  -- the proof is provided in the problem statement
  sorry

end corresponding_angles_equal_l2_2521


namespace range_of_expression_l2_2433

theorem range_of_expression (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 ≤ β ∧ β ≤ π / 2) :
  -π / 6 < 2 * α - β / 3 ∧ 2 * α - β / 3 < π := by
  sorry

end range_of_expression_l2_2433


namespace profit_percentage_A_is_20_l2_2494

-- Definitions of conditions
def cost_price_A := 156 -- Cost price of the cricket bat for A
def selling_price_C := 234 -- Selling price of the cricket bat to C
def profit_percent_B := 25 / 100 -- Profit percentage for B

-- Calculations
def cost_price_B := selling_price_C / (1 + profit_percent_B) -- Cost price of the cricket bat for B
def selling_price_A := cost_price_B -- Selling price of the cricket bat for A

-- Profit and profit percentage calculations
def profit_A := selling_price_A - cost_price_A -- Profit for A
def profit_percent_A := profit_A / cost_price_A * 100 -- Profit percentage for A

-- Statement to prove
theorem profit_percentage_A_is_20 : profit_percent_A = 20 :=
by
  sorry

end profit_percentage_A_is_20_l2_2494


namespace trig_identity_proof_l2_2349

theorem trig_identity_proof :
  let sin := Real.sin
  let cos := Real.cos
  let deg_to_rad := fun θ : ℝ => θ * Real.pi / 180
  sin (deg_to_rad 30) * sin (deg_to_rad 75) - sin (deg_to_rad 60) * cos (deg_to_rad 105) = Real.sqrt 2 / 2 :=
by
  sorry

end trig_identity_proof_l2_2349


namespace total_number_of_coins_l2_2238

theorem total_number_of_coins (x : ℕ) :
  5 * x + 10 * x + 25 * x = 120 → 3 * x = 9 :=
by
  intro h
  sorry

end total_number_of_coins_l2_2238


namespace slope_of_line_determined_by_solutions_l2_2495

theorem slope_of_line_determined_by_solutions :
  (∀ (x₁ y₁ x₂ y₂ : ℝ), (4 / x₁ + 6 / y₁ = 0) ∧ (4 / x₂ + 6 / y₂ = 0) →
    (y₂ - y₁) / (x₂ - x₁) = -3 / 2) :=
sorry

end slope_of_line_determined_by_solutions_l2_2495


namespace basketball_team_wins_l2_2504

theorem basketball_team_wins (wins_first_60 : ℕ) (remaining_games : ℕ) (total_games : ℕ) (target_win_percentage : ℚ) (winning_games : ℕ) : 
  wins_first_60 = 45 → remaining_games = 40 → total_games = 100 → target_win_percentage = 0.75 → 
  winning_games = 30 := by
  intros h1 h2 h3 h4
  sorry

end basketball_team_wins_l2_2504


namespace extreme_points_inequality_l2_2943

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1 / 2) * x^2 + a * Real.log (1 - x)

theorem extreme_points_inequality (a x1 x2 : ℝ) (h_a : 0 < a ∧ a < 1 / 4) 
  (h_sum : x1 + x2 = 1) (h_prod : x1 * x2 = a) (h_order : x1 < x2) :
  f x2 a - x1 > -(3 + Real.log 4) / 8 := 
by
  -- proof needed
  sorry

end extreme_points_inequality_l2_2943


namespace arithmetic_geometric_sequence_l2_2390

theorem arithmetic_geometric_sequence (a b c : ℝ) 
  (a_ne_b : a ≠ b) (b_ne_c : b ≠ c) (a_ne_c : a ≠ c)
  (h1 : 2 * b = a + c)
  (h2 : (a * b)^2 = a * b * c^2)
  (h3 : a + b + c = 15) : a = 20 := 
by 
  sorry

end arithmetic_geometric_sequence_l2_2390


namespace area_of_shaded_region_l2_2178

-- Given conditions
def side_length := 8
def area_of_square := side_length * side_length
def area_of_triangle := area_of_square / 4

-- Lean 4 statement for the equivalence
theorem area_of_shaded_region : area_of_triangle = 16 :=
by
  sorry

end area_of_shaded_region_l2_2178


namespace original_number_solution_l2_2770

theorem original_number_solution (x : ℝ) (h : x^2 + 45 = 100) : x = Real.sqrt 55 ∨ x = -Real.sqrt 55 :=
by
  sorry

end original_number_solution_l2_2770


namespace range_a_for_increasing_f_l2_2218

theorem range_a_for_increasing_f :
  (∀ (x : ℝ), 1 ≤ x → (2 * x - 2 * a) ≥ 0) → a ≤ 1 := by
  intro h
  sorry

end range_a_for_increasing_f_l2_2218


namespace cone_volume_l2_2174

theorem cone_volume (lateral_area : ℝ) (angle : ℝ) 
  (h₀ : lateral_area = 20 * Real.pi)
  (h₁ : angle = Real.arccos (4/5)) : 
  (1/3) * Real.pi * (4^2) * 3 = 16 * Real.pi :=
by
  sorry

end cone_volume_l2_2174


namespace sin_cos_sum_eq_l2_2096

theorem sin_cos_sum_eq :
  (Real.sin (20 * Real.pi / 180) * Real.cos (10 * Real.pi / 180) +
   Real.sin (70 * Real.pi / 180) * Real.sin (10 * Real.pi / 180)) = 1 / 2 :=
by 
  sorry

end sin_cos_sum_eq_l2_2096


namespace remaining_apps_eq_files_plus_more_initial_apps_eq_16_l2_2598

-- Defining the initial number of files
def initial_files: ℕ := 9

-- Defining the remaining number of files and apps
def remaining_files: ℕ := 5
def remaining_apps: ℕ := 12

-- Given: Dave has 7 more apps than files left
def apps_more_than_files: ℕ := 7

-- Equating the given condition 12 = 5 + 7
theorem remaining_apps_eq_files_plus_more :
  remaining_apps = remaining_files + apps_more_than_files := by
  sorry -- This would trivially prove as 12 = 5+7

-- Proving the number of initial apps
theorem initial_apps_eq_16 (A: ℕ) (h1: initial_files = 9) (h2: remaining_files = 5) (h3: remaining_apps = 12) (h4: apps_more_than_files = 7):
  A - remaining_apps = initial_files - remaining_files → A = 16 := by
  sorry

end remaining_apps_eq_files_plus_more_initial_apps_eq_16_l2_2598


namespace problem_b_50_l2_2348

def seq (b : ℕ → ℕ) : Prop :=
  b 1 = 3 ∧ ∀ n ≥ 1, b (n + 1) = b n + 3 * n

theorem problem_b_50 (b : ℕ → ℕ) (h : seq b) : b 50 = 3678 := 
sorry

end problem_b_50_l2_2348


namespace collinear_points_sum_l2_2777

theorem collinear_points_sum (a b : ℝ) 
  (h_collin: ∃ k : ℝ, 
    (1 - a) / (a - a) = k * (a - b) / (b - b) ∧
    (a - a) / (2 - b) = k * (2 - 3) / (3 - 3) ∧
    (a - b) / (3 - 3) = k * (a - a) / (3 - b) ) : 
  a + b = 4 :=
by
  sorry

end collinear_points_sum_l2_2777


namespace proof_shortest_side_l2_2652

-- Definitions based on problem conditions
def side_divided (a b : ℕ) : Prop := a + b = 20

def radius (r : ℕ) : Prop := r = 5

noncomputable def shortest_side (a b c : ℕ) : ℕ :=
  if a ≤ b ∧ a ≤ c then a
  else if b ≤ a ∧ b ≤ c then b
  else c

-- Proof problem statement
theorem proof_shortest_side {a b c : ℕ} (h1 : side_divided 9 11) (h2 : radius 5) :
  shortest_side 15 (11 + 9) (2 * 6 + 9) = 14 :=
sorry

end proof_shortest_side_l2_2652


namespace total_hexagons_calculation_l2_2339

-- Define the conditions
-- Regular hexagon side length
def hexagon_side_length : ℕ := 3

-- Number of smaller triangles
def small_triangle_count : ℕ := 54

-- Small triangle side length
def small_triangle_side_length : ℕ := 1

-- Define the total number of hexagons calculated
def total_hexagons : ℕ := 36

-- Theorem stating that given the conditions, the total number of hexagons is 36
theorem total_hexagons_calculation :
    (hexagon_side_length = 3) →
    (small_triangle_count = 54) →
    (small_triangle_side_length = 1) →
    total_hexagons = 36 :=
    by
    intros
    sorry

end total_hexagons_calculation_l2_2339


namespace quadratic_has_real_roots_l2_2979

theorem quadratic_has_real_roots (k : ℝ) : (∃ x : ℝ, x^2 + 4 * x + k = 0) ↔ k ≤ 4 := by
  sorry

end quadratic_has_real_roots_l2_2979


namespace cone_sphere_ratio_l2_2458

noncomputable def volume_of_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

noncomputable def volume_of_cone (r : ℝ) (h : ℝ) : ℝ :=
  (1 / 3) * Real.pi * (2 * r)^2 * h

theorem cone_sphere_ratio (r h : ℝ) (V_cone V_sphere : ℝ) (h_sphere : V_sphere = volume_of_sphere r)
  (h_cone : V_cone = volume_of_cone r h) (h_relation : V_cone = (1/3) * V_sphere) :
  (h / (2 * r) = 1 / 6) :=
by
  sorry

end cone_sphere_ratio_l2_2458


namespace radius_of_circumscribed_sphere_eq_a_l2_2129

-- Assume a to be a real number representing the side length of the base and height of the hexagonal pyramid
variables (a : ℝ)

-- Representing the base as a regular hexagon and the pyramid as having equal side length and height
def regular_hexagonal_pyramid (a : ℝ) : Type := {b : ℝ // b = a}

-- The radius of the circumscribed sphere to a given regular hexagonal pyramid
def radius_of_circumscribed_sphere (a : ℝ) : ℝ := a

-- Theorem stating that the radius of the sphere circumscribed around a regular hexagonal pyramid 
-- with side length and height both equal to a is a
theorem radius_of_circumscribed_sphere_eq_a (a : ℝ) :
  radius_of_circumscribed_sphere a = a :=
by {
  sorry
}

end radius_of_circumscribed_sphere_eq_a_l2_2129


namespace problem_condition_necessary_and_sufficient_l2_2735

theorem problem_condition_necessary_and_sufficient (a b : ℝ) (h : a * b > 0) :
  (a > b) ↔ (1 / a < 1 / b) :=
sorry

end problem_condition_necessary_and_sufficient_l2_2735


namespace hives_needed_for_candles_l2_2672

theorem hives_needed_for_candles (h : (3 : ℕ) * c = 12) : (96 : ℕ) / c = 24 :=
by
  sorry

end hives_needed_for_candles_l2_2672


namespace overlap_percentage_l2_2011

noncomputable def square_side_length : ℝ := 10
noncomputable def rectangle_length : ℝ := 18
noncomputable def rectangle_width : ℝ := square_side_length
noncomputable def overlap_length : ℝ := 2
noncomputable def overlap_width : ℝ := rectangle_width

noncomputable def rectangle_area : ℝ :=
  rectangle_length * rectangle_width

noncomputable def overlap_area : ℝ :=
  overlap_length * overlap_width

noncomputable def percentage_shaded : ℝ :=
  (overlap_area / rectangle_area) * 100

theorem overlap_percentage :
  percentage_shaded = 100 * (1 / 9) :=
sorry

end overlap_percentage_l2_2011


namespace solve_ax_plus_b_l2_2563

theorem solve_ax_plus_b (a b : ℝ) : 
  (if a ≠ 0 then "unique solution, x = -b / a"
   else if b ≠ 0 then "no solution"
   else "infinitely many solutions") = "A conditional control structure should be adopted" :=
sorry

end solve_ax_plus_b_l2_2563


namespace uncounted_angle_measure_l2_2312

-- Define the given miscalculated sum
def miscalculated_sum : ℝ := 2240

-- Define the correct sum expression for an n-sided convex polygon
def correct_sum (n : ℕ) : ℝ := (n - 2) * 180

-- State the theorem: 
theorem uncounted_angle_measure (n : ℕ) (h1 : correct_sum n = 2340) (h2 : 2240 < correct_sum n) :
  correct_sum n - miscalculated_sum = 100 := 
by sorry

end uncounted_angle_measure_l2_2312


namespace mateen_garden_area_l2_2155

theorem mateen_garden_area :
  ∃ (L W : ℝ), (20 * L = 1000) ∧ (8 * (2 * L + 2 * W) = 1000) ∧ (L * W = 625) :=
by
  sorry

end mateen_garden_area_l2_2155


namespace largest_divisor_of_n_l2_2732

theorem largest_divisor_of_n (n : ℕ) (h_pos : 0 < n) (h_div : 72 ∣ n^2) : 12 ∣ n :=
by
  sorry

end largest_divisor_of_n_l2_2732


namespace probability_X_eq_2_l2_2961

namespace Hypergeometric

def combin (n k : ℕ) : ℕ := n.choose k

noncomputable def hypergeometric (N M n k : ℕ) : ℚ :=
  (combin M k * combin (N - M) (n - k)) / combin N n

theorem probability_X_eq_2 :
  hypergeometric 8 5 3 2 = 15 / 28 := by
  sorry

end Hypergeometric

end probability_X_eq_2_l2_2961


namespace painted_rooms_l2_2345

/-- Given that there are a total of 11 rooms to paint, each room takes 7 hours to paint,
and the painter has 63 hours of work left to paint the remaining rooms,
prove that the painter has already painted 2 rooms. -/
theorem painted_rooms (total_rooms : ℕ) (hours_per_room : ℕ) (hours_left : ℕ) 
  (h_total_rooms : total_rooms = 11) (h_hours_per_room : hours_per_room = 7) 
  (h_hours_left : hours_left = 63) : 
  (total_rooms - hours_left / hours_per_room) = 2 := 
by
  sorry

end painted_rooms_l2_2345


namespace leesburg_population_l2_2254

theorem leesburg_population (salem_population leesburg_population half_salem_population number_moved_out : ℕ)
  (h1 : half_salem_population * 2 = salem_population)
  (h2 : salem_population - number_moved_out = 754100)
  (h3 : salem_population = 15 * leesburg_population)
  (h4 : half_salem_population = 377050)
  (h5 : number_moved_out = 130000) :
  leesburg_population = 58940 :=
by
  sorry

end leesburg_population_l2_2254


namespace sqrt_prime_irrational_l2_2303

theorem sqrt_prime_irrational (p : ℕ) (hp : Nat.Prime p) : Irrational (Real.sqrt p) :=
by
  sorry

end sqrt_prime_irrational_l2_2303


namespace angle_B_in_parallelogram_l2_2079

theorem angle_B_in_parallelogram (ABCD : Parallelogram) (angle_A angle_C : ℝ) 
  (h : angle_A + angle_C = 100) : 
  angle_B = 130 :=
by
  -- Proof omitted
  sorry

end angle_B_in_parallelogram_l2_2079


namespace maximum_xy_l2_2853

theorem maximum_xy (x y : ℝ) (h : x^2 + 2 * y^2 - 2 * x * y = 4) : 
  xy ≤ 2 * (Float.sqrt 2) + 2 :=
sorry

end maximum_xy_l2_2853


namespace area_of_triangle_is_3_l2_2427

noncomputable def area_of_triangle_ABC (A B C : ℝ × ℝ) : ℝ :=
1 / 2 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_of_triangle_is_3 : 
  ∀ (A B C : ℝ × ℝ), 
  A = (-5, -2) → 
  B = (0, 0) → 
  C = (7, -4) →
  area_of_triangle_ABC A B C = 3 :=
by
  intros A B C hA hB hC
  rw [hA, hB, hC]
  sorry

end area_of_triangle_is_3_l2_2427


namespace calculation_result_l2_2482

theorem calculation_result :
  let a := 0.0088
  let b := 4.5
  let c := 0.05
  let d := 0.1
  let e := 0.008
  (a * b) / (c * d * e) = 990 :=
by
  sorry

end calculation_result_l2_2482


namespace water_added_l2_2523

theorem water_added (initial_volume : ℕ) (ratio_milk_water_initial : ℚ) 
  (ratio_milk_water_final : ℚ) (w : ℕ)
  (initial_volume_eq : initial_volume = 45)
  (ratio_milk_water_initial_eq : ratio_milk_water_initial = 4 / 1)
  (ratio_milk_water_final_eq : ratio_milk_water_final = 6 / 5)
  (final_ratio_eq : ratio_milk_water_final = 36 / (9 + w)) :
  w = 21 := 
sorry

end water_added_l2_2523


namespace both_true_of_neg_and_false_l2_2559

variable (P Q : Prop)

theorem both_true_of_neg_and_false (h : ¬ (P ∧ Q) = False) : P ∧ Q :=
by
  -- Proof goes here
  sorry

end both_true_of_neg_and_false_l2_2559


namespace tan_alpha_minus_pi_over_4_l2_2160

open Real

theorem tan_alpha_minus_pi_over_4
  (α : ℝ)
  (a b : ℝ × ℝ)
  (h1 : a = (cos α, -2))
  (h2 : b = (sin α, 1))
  (h3 : ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2) :
  tan (α - π / 4) = -3 := 
sorry

end tan_alpha_minus_pi_over_4_l2_2160


namespace N_is_composite_l2_2309

def N : ℕ := 7 * 9 * 13 + 2020 * 2018 * 2014

theorem N_is_composite : ¬ Prime N :=
by {
  sorry
}

end N_is_composite_l2_2309


namespace team_C_games_played_l2_2400

variable (x : ℕ)
variable (winC : ℕ := 5 * x / 7)
variable (loseC : ℕ := 2 * x / 7)
variable (winD : ℕ := 2 * x / 3)
variable (loseD : ℕ := x / 3)

theorem team_C_games_played :
  winD = winC - 5 →
  loseD = loseC - 5 →
  x = 105 := by
  sorry

end team_C_games_played_l2_2400


namespace kevin_final_cards_l2_2973

-- Define the initial conditions and problem
def initial_cards : ℕ := 20
def found_cards : ℕ := 47
def lost_cards_1 : ℕ := 7
def lost_cards_2 : ℕ := 12
def won_cards : ℕ := 15

-- Define the function to calculate the final count
def final_cards (initial found lost1 lost2 won : ℕ) : ℕ :=
  (initial + found - lost1 - lost2 + won)

-- Statement of the problem to be proven
theorem kevin_final_cards :
  final_cards initial_cards found_cards lost_cards_1 lost_cards_2 won_cards = 63 :=
by
  sorry

end kevin_final_cards_l2_2973


namespace cos2_plus_sin2_given_tan_l2_2864

noncomputable def problem_cos2_plus_sin2_given_tan : Prop :=
  ∀ (α : ℝ), Real.tan α = 2 → Real.cos α ^ 2 + Real.sin (2 * α) = 1

-- Proof is omitted
theorem cos2_plus_sin2_given_tan : problem_cos2_plus_sin2_given_tan := sorry

end cos2_plus_sin2_given_tan_l2_2864


namespace remainder_9053_div_98_l2_2807

theorem remainder_9053_div_98 : 9053 % 98 = 37 :=
by sorry

end remainder_9053_div_98_l2_2807


namespace tidy_up_time_l2_2906

theorem tidy_up_time (A B C : ℕ) (tidyA : A = 5 * 3600) (tidyB : B = 5 * 60) (tidyC : C = 5) :
  B < A ∧ B > C :=
by
  sorry

end tidy_up_time_l2_2906


namespace sum_of_reversed_base_digits_eq_zero_l2_2265

theorem sum_of_reversed_base_digits_eq_zero : ∃ n : ℕ, 
  (∀ a₁ a₀ : ℕ, n = 5 * a₁ + a₀ ∧ n = 12 * a₀ + a₁ ∧ 0 ≤ a₁ ∧ a₁ < 5 ∧ 0 ≤ a₀ ∧ a₀ < 12 
  ∧ n > 0 → n = 0)
:= sorry

end sum_of_reversed_base_digits_eq_zero_l2_2265


namespace sum_of_first_10_common_elements_eq_13981000_l2_2668

def arithmetic_prog (n : ℕ) : ℕ := 4 + 3 * n
def geometric_prog (k : ℕ) : ℕ := 20 * 2 ^ k

theorem sum_of_first_10_common_elements_eq_13981000 :
  let common_elements : List ℕ := 
    [40, 160, 640, 2560, 10240, 40960, 163840, 655360, 2621440, 10485760]
  let sum_common_elements : ℕ := common_elements.sum
  sum_common_elements = 13981000 := by
  sorry

end sum_of_first_10_common_elements_eq_13981000_l2_2668


namespace ratio_of_rectangle_sides_l2_2840

theorem ratio_of_rectangle_sides (x y : ℝ) (h : x < y) 
  (hs : x + y - Real.sqrt (x^2 + y^2) = (1 / 3) * y) : 
  x / y = 5 / 12 :=
by
  sorry

end ratio_of_rectangle_sides_l2_2840


namespace polyhedron_space_diagonals_l2_2454

theorem polyhedron_space_diagonals (V E F T P : ℕ) (total_pairs_of_vertices total_edges total_face_diagonals : ℕ)
  (hV : V = 30)
  (hE : E = 70)
  (hF : F = 40)
  (hT : T = 30)
  (hP : P = 10)
  (h_total_pairs_of_vertices : total_pairs_of_vertices = 30 * 29 / 2)
  (h_total_face_diagonals : total_face_diagonals = 5 * 10)
  :
  total_pairs_of_vertices - E - total_face_diagonals = 315 := 
by
  sorry

end polyhedron_space_diagonals_l2_2454


namespace relationship_cannot_be_determined_l2_2146

noncomputable def point_on_parabola (a b c x y : ℝ) : Prop :=
  y = a * x^2 + b * x + c

theorem relationship_cannot_be_determined
  (a b c x1 y1 x2 y2 x3 y3 x4 y4 : ℝ) (h1 : a ≠ 0) 
  (h2 : point_on_parabola a b c x1 y1) 
  (h3 : point_on_parabola a b c x2 y2) 
  (h4 : point_on_parabola a b c x3 y3) 
  (h5 : point_on_parabola a b c x4 y4)
  (h6 : x1 + x4 - x2 + x3 = 0) : 
  ¬( ∃ m n : ℝ, ((y4 - y1) / (x4 - x1) = m ∧ (y2 - y3) / (x2 - x3) = m) ∨ 
                     ((y4 - y1) / (x4 - x1) * (y2 - y3) / (x2 - x3) = -1) ∨ 
                     ((y4 - y1) / (x4 - x1) ≠ m ∧ (y2 - y3) / (x2 - x3) ≠ m ∧ 
                      (y4 - y1) / (x4 - x1) * (y2 - y3) / (x2 - x3) ≠ -1)) :=
sorry

end relationship_cannot_be_determined_l2_2146


namespace grade_assignment_ways_l2_2053

theorem grade_assignment_ways : (4 ^ 12) = 16777216 := by
  sorry

end grade_assignment_ways_l2_2053


namespace find_x_l2_2793

theorem find_x (x : ℚ) (h : (3 * x + 4) / 5 = 15) : x = 71 / 3 :=
by
  sorry

end find_x_l2_2793


namespace functional_relationship_l2_2926

-- Define the conditions and question for Scenario ①
def scenario1 (x y k : ℝ) (h1 : k ≠ 0) : Prop :=
  y = k / x

-- Define the conditions and question for Scenario ②
def scenario2 (n S k : ℝ) (h2 : k ≠ 0) : Prop :=
  S = k / n

-- Define the conditions and question for Scenario ③
def scenario3 (t s k : ℝ) (h3 : k ≠ 0) : Prop :=
  s = k * t

-- The main theorem
theorem functional_relationship (x y n S t s k : ℝ) (h1 : k ≠ 0) :
  (scenario1 x y k h1) ∧ (scenario2 n S k h1) ∧ ¬(scenario3 t s k h1) := 
sorry

end functional_relationship_l2_2926


namespace find_range_m_l2_2561

variables (m : ℝ)

def p (m : ℝ) : Prop :=
  (∀ x y : ℝ, (x^2 / (2 * m)) - (y^2 / (m - 1)) = 1) → false

def q (m : ℝ) : Prop :=
  (∀ e : ℝ, (1 < e ∧ e < 2) → (∀ x y : ℝ, (y^2 / 5) - (x^2 / m) = 1)) → false

noncomputable def range_m (m : ℝ) : Prop :=
  p m = false ∧ q m = false ∧ (p m ∨ q m) = true → (1/3 ≤ m ∧ m < 15)

theorem find_range_m : ∀ m : ℝ, range_m m :=
by
  intro m
  simp [range_m, p, q]
  sorry

end find_range_m_l2_2561


namespace find_m_same_foci_l2_2720

theorem find_m_same_foci (m : ℝ) 
(hyperbola_eq : ∃ x y : ℝ, x^2 - y^2 = m) 
(ellipse_eq : ∃ x y : ℝ, 2 * x^2 + 3 * y^2 = m + 1) 
(same_foci : ∀ a b : ℝ, (x^2 - y^2 = m) ∧ (2 * x^2 + 3 * y^2 = m + 1) → 
               let c_ellipse := (m + 1) / 6
               let c_hyperbola := 2 * m
               c_ellipse = c_hyperbola ) : 
m = 1 / 11 := 
sorry

end find_m_same_foci_l2_2720


namespace sum_2016_eq_1008_l2_2082

-- Define the arithmetic sequence {a_n} and the sum of the first n terms S_n
variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Given conditions
variable (h_arith_seq : ∀ n m, a (n+1) - a n = a (m+1) - a m)
variable (h_sum : ∀ n, S n = (n * (a 1 + a n)) / 2)

-- Additional conditions from the problem
variable (h_vector : a 4 + a 2013 = 1)

-- Goal: Prove that the sum of the first 2016 terms equals 1008
theorem sum_2016_eq_1008 : S 2016 = 1008 := by
  sorry

end sum_2016_eq_1008_l2_2082


namespace grace_can_reach_target_sum_l2_2031

theorem grace_can_reach_target_sum :
  ∃ (half_dollars dimes pennies : ℕ),
    half_dollars ≤ 5 ∧ dimes ≤ 20 ∧ pennies ≤ 25 ∧
    (5 * 50 + 13 * 10 + 5) = 385 :=
sorry

end grace_can_reach_target_sum_l2_2031


namespace first_motorcyclist_laps_per_hour_l2_2960

noncomputable def motorcyclist_laps (x y z : ℝ) (P1 : 0 < x - y) (P2 : 0 < x - z) (P3 : 0 < y - z) : Prop :=
  (4.5 / (x - y) = 4.5) ∧ (4.5 / (x - z) = 4.5 - 0.5) ∧ (3 / (y - z) = 3) → x = 3

theorem first_motorcyclist_laps_per_hour (x y z : ℝ) (P1: 0 < x - y) (P2: 0 < x - z) (P3: 0 < y - z) :
  motorcyclist_laps x y z P1 P2 P3 →
  x = 3 :=
sorry

end first_motorcyclist_laps_per_hour_l2_2960


namespace yoojeong_initial_correct_l2_2639

variable (yoojeong_initial yoojeong_after marbles_given : ℕ)

-- Given conditions
axiom marbles_given_cond : marbles_given = 8
axiom yoojeong_after_cond : yoojeong_after = 24

-- Equation relating initial, given marbles, and marbles left
theorem yoojeong_initial_correct : 
  yoojeong_initial = yoojeong_after + marbles_given := by
  -- Proof skipped
  sorry

end yoojeong_initial_correct_l2_2639


namespace problem1_equation_of_line_intersection_perpendicular_problem2_equation_of_line_point_equal_intercepts_l2_2222

/-- Lean statement for the math proof problem -/

/- First problem -/
theorem problem1_equation_of_line_intersection_perpendicular :
  ∃ k, 3 * k - 2 * ( - (5 - 3 * k) / 2) - 11 = 0 :=
sorry

/- Second problem -/
theorem problem2_equation_of_line_point_equal_intercepts :
  (∃ a, (1, 2) ∈ {(x, y) | x + y = a}) ∧ a = 3
  ∨ (∃ b, (1, 2) ∈ {(x, y) | y = b * x}) ∧ b = 2 :=
sorry

end problem1_equation_of_line_intersection_perpendicular_problem2_equation_of_line_point_equal_intercepts_l2_2222


namespace find_x_l2_2372

def integers_x_y (x y : ℤ) : Prop :=
  x > y ∧ y > 0 ∧ x + y + x * y = 110

theorem find_x (x y : ℤ) (h : integers_x_y x y) : x = 36 := sorry

end find_x_l2_2372


namespace triangle_is_isosceles_l2_2922

-- lean statement
theorem triangle_is_isosceles (a b c : ℝ) (C : ℝ) (h : a = 2 * b * Real.cos C) : 
  ∃ k : ℝ, a = k ∧ b = k := 
sorry

end triangle_is_isosceles_l2_2922


namespace probability_of_picking_red_ball_l2_2205

theorem probability_of_picking_red_ball (w r : ℕ) 
  (h1 : r > w) 
  (h2 : r < 2 * w) 
  (h3 : 2 * w + 3 * r = 60) : 
  r / (w + r) = 7 / 11 :=
sorry

end probability_of_picking_red_ball_l2_2205


namespace cracked_seashells_zero_l2_2971

/--
Tom found 15 seashells, and Fred found 43 seashells. After cleaning, it was discovered that Fred had 28 more seashells than Tom. Prove that the number of cracked seashells is 0.
-/
theorem cracked_seashells_zero
(Tom_seashells : ℕ)
(Fred_seashells : ℕ)
(cracked_seashells : ℕ)
(Tom_after_cleaning : ℕ := Tom_seashells - cracked_seashells)
(Fred_after_cleaning : ℕ := Fred_seashells - cracked_seashells)
(h1 : Tom_seashells = 15)
(h2 : Fred_seashells = 43)
(h3 : Fred_after_cleaning = Tom_after_cleaning + 28) :
  cracked_seashells = 0 :=
by
  -- Placeholder for the proof
  sorry

end cracked_seashells_zero_l2_2971


namespace election_total_valid_votes_l2_2131

theorem election_total_valid_votes (V B : ℝ) 
    (hA : 0.45 * V = B * V + 250) 
    (hB : 2.5 * B = 62.5) :
    V = 1250 :=
by
  sorry

end election_total_valid_votes_l2_2131


namespace expression_evaluates_to_one_l2_2571

noncomputable def a := Real.sqrt 2 + 0.8
noncomputable def b := Real.sqrt 2 - 0.2

theorem expression_evaluates_to_one : 
  ( (2 - b) / (b - 1) + 2 * (a - 1) / (a - 2) ) / ( b * (a - 1) / (b - 1) + a * (2 - b) / (a - 2) ) = 1 :=
by
  sorry

end expression_evaluates_to_one_l2_2571


namespace total_paths_from_X_to_Z_l2_2965

variable (X Y Z : Type)
variables (f : X → Y → Z)
variables (g : X → Z)

-- Conditions
def paths_X_to_Y : ℕ := 3
def paths_Y_to_Z : ℕ := 4
def direct_paths_X_to_Z : ℕ := 1

-- Proof problem statement
theorem total_paths_from_X_to_Z : paths_X_to_Y * paths_Y_to_Z + direct_paths_X_to_Z = 13 := sorry

end total_paths_from_X_to_Z_l2_2965


namespace perpendicular_lines_slope_l2_2453

theorem perpendicular_lines_slope {a : ℝ} :
  (∃ (a : ℝ), (∀ x y : ℝ, x + 2 * y - 1 = 0 → a * x - y - 1 = 0) ∧ (a * (-1 / 2)) = -1) → a = 2 :=
by sorry

end perpendicular_lines_slope_l2_2453


namespace other_root_and_m_l2_2353

-- Definitions for the conditions
def quadratic_eq (m : ℝ) := ∀ x : ℝ, x^2 + 2 * x + m = 0
def root (x : ℝ) (m : ℝ) := x^2 + 2 * x + m = 0

-- Theorem statement
theorem other_root_and_m (m : ℝ) (h : root 2 m) : ∃ t : ℝ, (2 + t = -2) ∧ (2 * t = m) ∧ t = -4 ∧ m = -8 := 
by {
  -- Placeholder for the actual proof
  sorry
}

end other_root_and_m_l2_2353


namespace area_trapezoid_def_l2_2838

noncomputable def area_trapezoid (a : ℝ) (h : a ≠ 0) : ℝ :=
  let b := 108 / a
  let DE := a / 2
  let FG := b / 3
  let height := b / 2
  (DE + FG) * height / 2

theorem area_trapezoid_def (a : ℝ) (h : a ≠ 0) :
  area_trapezoid a h = 18 + 18 / a :=
by
  sorry

end area_trapezoid_def_l2_2838


namespace min_sum_of_grid_numbers_l2_2464

-- Definition of the 2x2 grid and the problem conditions
variables (a b c d : ℕ)
variables (a_pos : a > 0) (b_pos : b > 0) (c_pos : c > 0) (d_pos : d > 0)

-- Lean statement for the minimum sum proof problem
theorem min_sum_of_grid_numbers :
  a + b + c + d + a * b + c * d + a * c + b * d = 2015 → a + b + c + d = 88 :=
by
  sorry

end min_sum_of_grid_numbers_l2_2464


namespace problem_solution_l2_2552

noncomputable def a_sequence : ℕ → ℕ := sorry
noncomputable def S_n : ℕ → ℕ := sorry
noncomputable def b_sequence : ℕ → ℕ := sorry
noncomputable def c_sequence : ℕ → ℕ := sorry
noncomputable def T_n : ℕ → ℕ := sorry

theorem problem_solution (n : ℕ) (a_condition : ∀ n : ℕ, 2 * S_n = (n + 1) ^ 2 * a_sequence n - n ^ 2 * a_sequence (n + 1))
                        (b_condition : ∀ n : ℕ, b_sequence 1 = a_sequence 1 ∧ (n ≠ 0 → n * b_sequence (n + 1) = a_sequence n * b_sequence n)) :
  (∀ n, a_sequence n = 2 * n) ∧
  (∀ n, b_sequence n = 2 ^ n) ∧
  (∀ n, T_n n = 2 ^ (n + 1) + n ^ 2 + n - 2) :=
sorry


end problem_solution_l2_2552


namespace find_d_l2_2271

theorem find_d (c : ℝ) (d : ℝ) (h1 : c = 7)
  (h2 : (2, 6) ∈ { p : ℝ × ℝ | ∃ d, (p = (2, 6) ∨ p = (5, c) ∨ p = (d, 0)) ∧
           ∃ m, m = (0 - 6) / (d - 2) ∧ m = (c - 6) / (5 - 2) }) : 
  d = -16 :=
by
  sorry

end find_d_l2_2271


namespace minute_hand_rotation_l2_2060

theorem minute_hand_rotation :
  (10 / 60) * (2 * Real.pi) = (- Real.pi / 3) :=
by
  sorry

end minute_hand_rotation_l2_2060


namespace total_tiles_needed_l2_2543

-- Definitions of the given conditions
def blue_tiles : Nat := 48
def red_tiles : Nat := 32
def additional_tiles_needed : Nat := 20

-- Statement to prove the total number of tiles needed to complete the pool
theorem total_tiles_needed : blue_tiles + red_tiles + additional_tiles_needed = 100 := by
  sorry

end total_tiles_needed_l2_2543


namespace measure_of_angle_l2_2145

theorem measure_of_angle (x : ℝ) (h1 : 90 - x = 3 * x - 10) : x = 25 :=
by
  sorry

end measure_of_angle_l2_2145


namespace max_black_cells_in_101x101_grid_l2_2825

theorem max_black_cells_in_101x101_grid :
  ∀ (k : ℕ), k ≤ 101 → 2 * k * (101 - k) ≤ 5100 :=
by
  sorry

end max_black_cells_in_101x101_grid_l2_2825


namespace vasya_max_earning_l2_2307

theorem vasya_max_earning (k : ℕ) (h₀: k ≤ 2013) (h₁: 2013 - 2*k % 11 = 0) : k % 11 = 0 → (k ≤ 5) := 
by
  sorry

end vasya_max_earning_l2_2307


namespace barney_no_clean_towels_days_l2_2196

theorem barney_no_clean_towels_days
  (wash_cycle_weeks : ℕ := 1)
  (total_towels : ℕ := 18)
  (towels_per_day : ℕ := 2)
  (days_per_week : ℕ := 7)
  (missed_laundry_weeks : ℕ := 1) :
  (days_per_week - (total_towels - (days_per_week * towels_per_day * missed_laundry_weeks)) / towels_per_day) = 5 :=
by
  sorry

end barney_no_clean_towels_days_l2_2196


namespace line_parabola_intersection_l2_2185

theorem line_parabola_intersection (k : ℝ) (M A B : ℝ × ℝ) (h1 : ¬ k = 0) 
  (h2 : M = (2, 0))
  (h3 : ∃ x y, (x = k * y + 2 ∧ (x, y) ∈ {p : ℝ × ℝ | p.2^2 = 4 * p.1} ∧ (p = A ∨ p = B))) 
  : 1 / |dist M A|^2 + 1 / |dist M B|^2 = 1 / 4 := 
by 
  sorry

end line_parabola_intersection_l2_2185


namespace payment_of_employee_B_l2_2669

-- Define the variables and conditions
variables (A B : ℝ) (total_payment : ℝ) (payment_ratio : ℝ)

-- Assume the given conditions
def conditions : Prop := 
  (A + B = total_payment) ∧ 
  (A = payment_ratio * B) ∧ 
  (total_payment = 550) ∧ 
  (payment_ratio = 1.5)

-- Prove the payment of employee B is 220 given the conditions
theorem payment_of_employee_B : conditions A B total_payment payment_ratio → B = 220 := 
by
  sorry

end payment_of_employee_B_l2_2669


namespace solve_equation_1_solve_equation_2_solve_equation_3_l2_2231

theorem solve_equation_1 (x : ℝ) : x^2 - 2 * x = 0 ↔ x = 0 ∨ x = 2 := 
sorry

theorem solve_equation_2 (x : ℝ) : (2 * x - 1)^2 = (3 - x)^2 ↔ x = -2 ∨ x = 4 / 3 := 
sorry

theorem solve_equation_3 (x : ℝ) : 3 * x * (x - 2) = x - 2 ↔ x = 2 ∨ x = 1 / 3 :=
sorry

end solve_equation_1_solve_equation_2_solve_equation_3_l2_2231


namespace a_gt_b_l2_2870

noncomputable def a (R : Type*) [OrderedRing R] := {x : R // 0 < x ∧ x ^ 3 = x + 1}
noncomputable def b (R : Type*) [OrderedRing R] (a : R) := {y : R // 0 < y ∧ y ^ 6 = y + 3 * a}

theorem a_gt_b (R : Type*) [OrderedRing R] (a_pos_real : a R) (b_pos_real : b R (a_pos_real.val)) : a_pos_real.val > b_pos_real.val :=
sorry

end a_gt_b_l2_2870


namespace n_squared_plus_3n_is_perfect_square_iff_l2_2272

theorem n_squared_plus_3n_is_perfect_square_iff (n : ℕ) : 
  ∃ k : ℕ, n^2 + 3 * n = k^2 ↔ n = 1 :=
by 
  sorry

end n_squared_plus_3n_is_perfect_square_iff_l2_2272


namespace largest_int_square_3_digits_base_7_l2_2244

theorem largest_int_square_3_digits_base_7 :
  ∃ (N : ℕ), (7^2 ≤ N^2) ∧ (N^2 < 7^3) ∧ 
  ∃ k : ℕ, N = k ∧ k^2 ≥ 7^2 ∧ k^2 < 7^3 ∧
  N = 45 := sorry

end largest_int_square_3_digits_base_7_l2_2244


namespace new_energy_vehicles_l2_2849

-- Given conditions
def conditions (a b : ℕ) : Prop :=
  3 * a + 2 * b = 95 ∧ 4 * a + 1 * b = 110

-- Given prices
def purchase_prices : Prop :=
  ∃ a b, conditions a b ∧ a = 25 ∧ b = 10

-- Total value condition for different purchasing plans
def purchase_plans (m n : ℕ) : Prop :=
  25 * m + 10 * n = 250 ∧ m > 0 ∧ n > 0

-- Number of different purchasing plans
def num_purchase_plans : Prop :=
  ∃ num_plans, num_plans = 4

-- Profit calculation for a given plan
def profit (m n : ℕ) : ℕ :=
  12 * m + 8 * n

-- Maximum profit condition
def max_profit : Prop :=
  ∃ max_profit, max_profit = 184 ∧ ∀ (m n : ℕ), purchase_plans m n → profit m n ≤ 184

-- Main theorem
theorem new_energy_vehicles : purchase_prices ∧ num_purchase_plans ∧ max_profit :=
  sorry

end new_energy_vehicles_l2_2849


namespace not_possible_to_fill_6x6_with_1x4_l2_2086

theorem not_possible_to_fill_6x6_with_1x4 :
  ¬ (∃ (a b : ℕ), a + 4 * b = 6 ∧ 4 * a + b = 6) :=
by
  -- Assuming a and b represent the number of 1x4 rectangles aligned horizontally and vertically respectively
  sorry

end not_possible_to_fill_6x6_with_1x4_l2_2086


namespace units_digit_of_M_is_1_l2_2620

def Q (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  if units = 0 then 0 else tens / units

def T (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

theorem units_digit_of_M_is_1 (a b : ℕ) (h₁ : 1 ≤ a) (h₂ : b ≤ 9) (h₃ : 10*a + b = Q (10*a + b) + T (10*a + b)) :
  b = 1 :=
by
  sorry

end units_digit_of_M_is_1_l2_2620


namespace find_angle_B_l2_2226

theorem find_angle_B (a b c : ℝ) (h : a^2 + c^2 - b^2 = a * c) : 
  ∃ B : ℝ, 0 < B ∧ B < 180 ∧ B = 60 :=
by 
  sorry

end find_angle_B_l2_2226


namespace sequence_general_term_correctness_l2_2214

def sequenceGeneralTerm (n : ℕ) : ℤ :=
  if n % 2 = 1 then
    0
  else
    (-1) ^ (n / 2 + 1)

theorem sequence_general_term_correctness (n : ℕ) :
  (∀ m, sequenceGeneralTerm m = 0 ↔ m % 2 = 1) ∧
  (∀ k, sequenceGeneralTerm k = (-1) ^ (k / 2 + 1) ↔ k % 2 = 0) :=
by
  sorry

end sequence_general_term_correctness_l2_2214


namespace mn_minus_n_values_l2_2251

theorem mn_minus_n_values (m n : ℝ) (h1 : |m| = 4) (h2 : |n| = 2.5) (h3 : m * n < 0) :
  m * n - n = -7.5 ∨ m * n - n = -12.5 :=
sorry

end mn_minus_n_values_l2_2251


namespace inequality_solution_l2_2583

theorem inequality_solution (x : ℝ) : 3 * x ^ 2 + x - 2 < 0 ↔ -1 < x ∧ x < 2 / 3 :=
by
  -- The proof should factor the quadratic expression and apply the rule for solving strict inequalities
  sorry

end inequality_solution_l2_2583


namespace number_of_children_l2_2800

-- Definitions for the conditions
def adult_ticket_cost : ℕ := 8
def child_ticket_cost : ℕ := 3
def total_amount : ℕ := 35

-- Theorem stating the proof problem
theorem number_of_children (A C T : ℕ) (hc: A = adult_ticket_cost) (ha: C = child_ticket_cost) (ht: T = total_amount) :
  (T - A) / C = 9 :=
by
  sorry

end number_of_children_l2_2800


namespace number_of_valid_groupings_l2_2296

-- Definitions based on conditions
def num_guides : ℕ := 2
def num_tourists : ℕ := 6
def total_groupings : ℕ := 2 ^ num_tourists
def invalid_groupings : ℕ := 2  -- All tourists go to one guide either a or b

-- The theorem to prove
theorem number_of_valid_groupings : total_groupings - invalid_groupings = 62 :=
by sorry

end number_of_valid_groupings_l2_2296


namespace rectangle_width_decrease_proof_l2_2868

def rectangle_width_decreased_percentage (L W : ℝ) (h : L * W = (1.4 * L) * (W / 1.4)) : ℝ := 
  28.57

theorem rectangle_width_decrease_proof (L W : ℝ) (h : L * W = (1.4 * L) * (W / 1.4)) : 
  rectangle_width_decreased_percentage L W h = 28.57 := 
by
  sorry

end rectangle_width_decrease_proof_l2_2868


namespace prob_two_segments_same_length_l2_2768

namespace hexagon_prob

noncomputable def prob_same_length : ℚ :=
  let total_elements : ℕ := 15
  let sides : ℕ := 6
  let diagonals : ℕ := 9
  (sides / total_elements) * ((sides - 1) / (total_elements - 1)) + (diagonals / total_elements) * ((diagonals - 1) / (total_elements - 1))

theorem prob_two_segments_same_length : prob_same_length = 17 / 35 :=
by
  sorry

end hexagon_prob

end prob_two_segments_same_length_l2_2768


namespace infinite_series_sum_l2_2666

theorem infinite_series_sum : (∑' n : ℕ, if n % 3 = 0 then 1 / (3 * 2^(((n - n % 3) / 3) + 1)) 
                                 else if n % 3 = 1 then -1 / (6 * 2^(((n - n % 3) / 3)))
                                 else -1 / (12 * 2^(((n - n % 3) / 3)))) = 1 / 72 :=
by
  sorry

end infinite_series_sum_l2_2666


namespace man_age_difference_l2_2862

theorem man_age_difference (S M : ℕ) (h1 : S = 22) (h2 : M + 2 = 2 * (S + 2)) :
  M - S = 24 :=
by sorry

end man_age_difference_l2_2862


namespace tiles_needed_to_cover_floor_l2_2519

-- Definitions of the conditions
def room_length : ℕ := 2
def room_width : ℕ := 12
def tile_area : ℕ := 4

-- The proof statement: calculate the number of tiles needed to cover the entire floor
theorem tiles_needed_to_cover_floor : 
  (room_length * room_width) / tile_area = 6 := 
by 
  sorry

end tiles_needed_to_cover_floor_l2_2519


namespace age_difference_l2_2148

-- Let D denote the daughter's age and M denote the mother's age
variable (D M : ℕ)

-- Conditions given in the problem
axiom h1 : M = 11 * D
axiom h2 : M + 13 = 2 * (D + 13)

-- The main proof statement to show the difference in their current ages
theorem age_difference : M - D = 40 :=
by
  sorry

end age_difference_l2_2148


namespace value_to_be_subtracted_l2_2816

theorem value_to_be_subtracted (N x : ℕ) (h1 : (N - x) / 7 = 7) (h2 : (N - 24) / 10 = 3) : x = 5 := by
  sorry

end value_to_be_subtracted_l2_2816


namespace pages_to_read_tomorrow_l2_2947

-- Define the problem setup
def total_pages : ℕ := 100
def pages_yesterday : ℕ := 35
def pages_today : ℕ := pages_yesterday - 5

-- Define the total pages read after two days
def pages_read_in_two_days : ℕ := pages_yesterday + pages_today

-- Define the number of pages left to read
def pages_left_to_read (total_pages read_so_far : ℕ) : ℕ := total_pages - read_so_far

-- Prove that the number of pages to read tomorrow is 35
theorem pages_to_read_tomorrow :
  pages_left_to_read total_pages pages_read_in_two_days = 35 :=
by
  -- Proof is omitted
  sorry

end pages_to_read_tomorrow_l2_2947


namespace sugar_left_in_grams_l2_2734

theorem sugar_left_in_grams 
  (initial_ounces : ℝ) (spilled_ounces : ℝ) (conversion_factor : ℝ)
  (h_initial : initial_ounces = 9.8) (h_spilled : spilled_ounces = 5.2)
  (h_conversion : conversion_factor = 28.35) :
  (initial_ounces - spilled_ounces) * conversion_factor = 130.41 := 
by
  sorry

end sugar_left_in_grams_l2_2734


namespace polyhedron_value_calculation_l2_2792

noncomputable def calculate_value (P T V : ℕ) : ℕ :=
  100 * P + 10 * T + V

theorem polyhedron_value_calculation :
  ∀ (P T V E F : ℕ),
    F = 36 ∧
    T + P = 36 ∧
    E = (3 * T + 5 * P) / 2 ∧
    V = E - F + 2 →
    calculate_value P T V = 2018 :=
by
  intros P T V E F h
  sorry

end polyhedron_value_calculation_l2_2792


namespace rationalize_simplify_l2_2134

theorem rationalize_simplify :
  3 / (Real.sqrt 75 + Real.sqrt 3) = Real.sqrt 3 / 6 :=
by
  sorry

end rationalize_simplify_l2_2134


namespace counterexample_to_prime_statement_l2_2707

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬ is_prime n

theorem counterexample_to_prime_statement 
  (n : ℕ) 
  (h_n_composite : is_composite n) 
  (h_n_minus_3_not_prime : ¬ is_prime (n - 3)) : 
  n = 18 ∨ n = 24 :=
by 
  sorry

end counterexample_to_prime_statement_l2_2707


namespace cos_alpha_neg_3_5_l2_2356

open Real

variables {α : ℝ} (h_alpha : sin α = 4 / 5) (h_quadrant : π / 2 < α ∧ α < π)

theorem cos_alpha_neg_3_5 : cos α = -3 / 5 :=
by
  -- Proof omitted
  sorry

end cos_alpha_neg_3_5_l2_2356


namespace g_f2_minus_f_g2_eq_zero_l2_2703

def f (x : ℝ) : ℝ := x^2 + 3 * x + 1

def g (x : ℝ) : ℝ := x^2 + 2 * x + 3

theorem g_f2_minus_f_g2_eq_zero : g (f 2) - f (g 2) = 0 := by
  sorry

end g_f2_minus_f_g2_eq_zero_l2_2703


namespace scientific_notation_of_274000000_l2_2240

theorem scientific_notation_of_274000000 :
  (274000000 : ℝ) = 2.74 * 10 ^ 8 :=
by
    sorry

end scientific_notation_of_274000000_l2_2240


namespace AngeliCandies_l2_2416

def CandyProblem : Prop :=
  ∃ (C B G : ℕ), 
    (1/3 : ℝ) * C = 3 * (B : ℝ) ∧
    (2/3 : ℝ) * C = 2 * (G : ℝ) ∧
    (B + G = 40) ∧ 
    C = 144

theorem AngeliCandies :
  CandyProblem :=
sorry

end AngeliCandies_l2_2416


namespace like_terms_exponents_l2_2575

theorem like_terms_exponents (m n : ℤ) 
  (h1 : 3 = m - 2) 
  (h2 : n + 1 = 2) : m - n = 4 := 
by
  sorry

end like_terms_exponents_l2_2575


namespace fishing_problem_l2_2962

theorem fishing_problem
  (everyday : ℕ)
  (every_other_day : ℕ)
  (every_three_days : ℕ)
  (yesterday_fishing : ℕ)
  (today_fishing : ℕ)
  (h_everyday : everyday = 7)
  (h_every_other_day : every_other_day = 8)
  (h_every_three_days : every_three_days = 3)
  (h_yesterday_fishing : yesterday_fishing = 12)
  (h_today_fishing : today_fishing = 10) :
  (every_three_days + everyday + (every_other_day - (yesterday_fishing - everyday))) = 15 := by
  sorry

end fishing_problem_l2_2962


namespace figure_perimeter_equals_26_l2_2881

noncomputable def rectangle_perimeter : ℕ := 26

def figure_arrangement (width height : ℕ) : Prop :=
width = 2 ∧ height = 1

theorem figure_perimeter_equals_26 {width height : ℕ} (h : figure_arrangement width height) :
  rectangle_perimeter = 26 :=
by
  sorry

end figure_perimeter_equals_26_l2_2881


namespace jade_savings_per_month_l2_2811

def jade_monthly_income : ℝ := 1600
def jade_living_expense_rate : ℝ := 0.75
def jade_insurance_rate : ℝ := 0.2

theorem jade_savings_per_month : 
  jade_monthly_income * (1 - jade_living_expense_rate - jade_insurance_rate) = 80 := by
  sorry

end jade_savings_per_month_l2_2811


namespace solve_fraction_eqn_l2_2799

def fraction_eqn_solution : Prop :=
  ∃ (x : ℝ), (x + 2) / (x - 1) = 0 ∧ x ≠ 1 ∧ x = -2

theorem solve_fraction_eqn : fraction_eqn_solution :=
sorry

end solve_fraction_eqn_l2_2799


namespace bob_total_questions_l2_2091

theorem bob_total_questions (q1 q2 q3 : ℕ) : 
  q1 = 13 ∧ q2 = 2 * q1 ∧ q3 = 2 * q2 → q1 + q2 + q3 = 91 :=
by
  intros
  sorry

end bob_total_questions_l2_2091


namespace tom_used_10_plates_l2_2333

theorem tom_used_10_plates
  (weight_per_plate : ℕ := 30)
  (felt_weight : ℕ := 360)
  (heavier_factor : ℚ := 1.20) :
  (felt_weight / heavier_factor / weight_per_plate : ℚ) = 10 := by
  sorry

end tom_used_10_plates_l2_2333


namespace largest_square_area_l2_2646

theorem largest_square_area (XY XZ YZ : ℝ)
  (h1 : XZ^2 = 2 * XY^2)
  (h2 : XY^2 + YZ^2 = XZ^2)
  (h3 : XY^2 + YZ^2 + XZ^2 = 450) :
  XZ^2 = 225 :=
by
  -- Proof skipped
  sorry

end largest_square_area_l2_2646


namespace negation_of_universal_sin_l2_2432

theorem negation_of_universal_sin (h : ∀ x : ℝ, Real.sin x > 0) : ∃ x : ℝ, Real.sin x ≤ 0 :=
sorry

end negation_of_universal_sin_l2_2432


namespace percentage_increase_l2_2774

theorem percentage_increase (lowest_price highest_price : ℝ) (h_low : lowest_price = 15) (h_high : highest_price = 25) :
  ((highest_price - lowest_price) / lowest_price) * 100 = 66.67 :=
by
  sorry

end percentage_increase_l2_2774


namespace right_triangle_distance_midpoint_l2_2087

noncomputable def distance_from_F_to_midpoint_DE
  (D E F : ℝ × ℝ)
  (right_triangle : ∃ A B C, A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
                    A.1^2 + A.2^2 = 1 ∧ B.1^2 + B.2^2 = 1 ∧ C.1^2 + C.2^2 = 1 ∧
                    D = A ∧ E = B ∧ F = C) 
  (DE : ℝ)
  (DF : ℝ)
  (EF : ℝ)
  : ℝ :=
  if hD : (D.1 - E.1)^2 + (D.2 - E.2)^2 = DE^2 then
    if hF : (D.1 - F.1)^2 + (D.2 - F.2)^2 = DF^2 then
      if hDE : DE = 15 then
        (15 / 2) --distance from F to midpoint of DE
      else
        0 -- This will never be executed since DE = 15 is a given condition
    else
      0 -- This will never be executed since DF = 9 is a given condition
  else
    0 -- This will never be executed since EF = 12 is a given condition

theorem right_triangle_distance_midpoint
  (D E F : ℝ × ℝ)
  (h_triangle : ∃ A B C, A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
                    A.1^2 + A.2^2 = 1 ∧ B.1^2 + B.2^2 = 1 ∧ C.1^2 + C.2^2 = 1 ∧
                    D = A ∧ E = B ∧ F = C)
  (hDE : (D.1 - E.1)^2 + (D.2 - E.2)^2 = 15^2)
  (hDF : (D.1 - F.1)^2 + (D.2 - F.2)^2 = 9^2)
  (hEF : (E.1 - F.1)^2 + (E.2 - F.2)^2 = 12^2) :
  distance_from_F_to_midpoint_DE D E F h_triangle 15 9 12 = 7.5 :=
by sorry

end right_triangle_distance_midpoint_l2_2087


namespace approximate_value_correct_l2_2431

noncomputable def P1 : ℝ := (47 / 100) * 1442
noncomputable def P2 : ℝ := (36 / 100) * 1412
noncomputable def result : ℝ := (P1 - P2) + 63

theorem approximate_value_correct : abs (result - 232.42) < 0.01 := 
by
  -- Proof to be completed
  sorry

end approximate_value_correct_l2_2431


namespace prob_two_white_balls_l2_2415

open Nat

def total_balls : ℕ := 8 + 10

def prob_first_white : ℚ := 8 / total_balls

def prob_second_white (total_balls_minus_one : ℕ) : ℚ := 7 / total_balls_minus_one

theorem prob_two_white_balls : 
  ∃ (total_balls_minus_one : ℕ) (p_first p_second : ℚ), 
    total_balls_minus_one = total_balls - 1 ∧
    p_first = prob_first_white ∧
    p_second = prob_second_white total_balls_minus_one ∧
    p_first * p_second = 28 / 153 := 
by
  sorry

end prob_two_white_balls_l2_2415


namespace triangle_is_isosceles_l2_2576

theorem triangle_is_isosceles (A B C : ℝ) (a b c : ℝ) 
  (h1 : c = 2 * a * Real.cos B) 
  (h2 : a = b) :
  ∃ (isIsosceles : Bool), isIsosceles := 
sorry

end triangle_is_isosceles_l2_2576


namespace gcd_three_numbers_l2_2806

theorem gcd_three_numbers :
  gcd (gcd 324 243) 135 = 27 :=
by
  sorry

end gcd_three_numbers_l2_2806


namespace vectors_parallel_x_squared_eq_two_l2_2147

theorem vectors_parallel_x_squared_eq_two (x : ℝ) 
  (a : ℝ × ℝ := (x+2, 1+x)) 
  (b : ℝ × ℝ := (x-2, 1-x)) 
  (parallel : (a.1 * b.2 - a.2 * b.1) = 0) : x^2 = 2 :=
sorry

end vectors_parallel_x_squared_eq_two_l2_2147


namespace solve_system_of_inequalities_l2_2077

theorem solve_system_of_inequalities 
  (x : ℝ) 
  (h1 : x - 3 * (x - 2) ≥ 4)
  (h2 : (1 + 2 * x) / 3 > x - 1) : 
  x ≤ 1 := 
sorry

end solve_system_of_inequalities_l2_2077


namespace denominator_of_fractions_l2_2997

theorem denominator_of_fractions (y a : ℝ) (hy : y > 0) 
  (h : (2 * y) / a + (3 * y) / a = 0.5 * y) : a = 10 :=
by
  sorry

end denominator_of_fractions_l2_2997


namespace absolute_difference_avg_median_l2_2709

theorem absolute_difference_avg_median (a b : ℝ) (h1 : 1 < a) (h2 : a < b) : 
  |((3 + 4 * a + 2 * b) / 4) - (a + b / 2 + 1)| = 1 / 4 :=
by
  sorry

end absolute_difference_avg_median_l2_2709


namespace power_of_two_minus_one_divisible_by_seven_power_of_two_plus_one_not_divisible_by_seven_l2_2275

theorem power_of_two_minus_one_divisible_by_seven (n : ℕ) (hn : 0 < n) : 
  (∃ k : ℕ, 0 < k ∧ n = k * 3) ↔ (7 ∣ 2^n - 1) :=
by sorry

theorem power_of_two_plus_one_not_divisible_by_seven (n : ℕ) (hn : 0 < n) :
  ¬(7 ∣ 2^n + 1) :=
by sorry

end power_of_two_minus_one_divisible_by_seven_power_of_two_plus_one_not_divisible_by_seven_l2_2275


namespace fraction_book_read_l2_2354

theorem fraction_book_read (read_pages : ℚ) (h : read_pages = 3/7) :
  (1 - read_pages = 4/7) ∧ (read_pages / (1 - read_pages) = 3/4) :=
by
  sorry

end fraction_book_read_l2_2354


namespace thomas_score_l2_2972

def average (scores : List ℕ) : ℚ := scores.sum / scores.length

variable (scores : List ℕ)

theorem thomas_score (h_length : scores.length = 19)
                     (h_avg_before : average scores = 78)
                     (h_avg_after : average ((98 :: scores)) = 79) :
  let thomas_score := 98
  thomas_score = 98 := sorry

end thomas_score_l2_2972


namespace compare_fractions_l2_2439

def a : ℚ := -3 / 4
def b : ℚ := -4 / 5

theorem compare_fractions : a > b :=
by {
  sorry
}

end compare_fractions_l2_2439


namespace net_effect_on_sale_value_l2_2264

theorem net_effect_on_sale_value (P Q : ℝ) (hP : P > 0) (hQ : Q > 0) :
  let original_sale_value := P * Q
  let new_price := 0.82 * P
  let new_quantity := 1.88 * Q
  let new_sale_value := new_price * new_quantity
  let net_effect := (new_sale_value / original_sale_value - 1) * 100
  net_effect = 54.16 :=
by
  sorry

end net_effect_on_sale_value_l2_2264


namespace distance_between_points_A_B_l2_2169

theorem distance_between_points_A_B :
  let A := (8, -5)
  let B := (0, 10)
  Real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2) = 17 :=
by
  let A := (8, -5)
  let B := (0, 10)
  sorry

end distance_between_points_A_B_l2_2169


namespace even_function_value_l2_2005

-- Define the function condition
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- Define the main problem with given conditions
theorem even_function_value (f : ℝ → ℝ) (h1 : is_even_function f) (h2 : ∀ x : ℝ, x < 0 → f x = x * (x + 1)) 
  (x : ℝ) (hx : x > 0) : f x = x * (x - 1) :=
  sorry

end even_function_value_l2_2005


namespace cost_of_3600_pens_l2_2395

-- Define the conditions
def cost_per_200_pens : ℕ := 50
def pens_bought : ℕ := 3600

-- Define a theorem to encapsulate our question and provide the necessary definitions
theorem cost_of_3600_pens : cost_per_200_pens / 200 * pens_bought = 900 := by sorry

end cost_of_3600_pens_l2_2395


namespace hannah_spent_65_l2_2083

-- Definitions based on the conditions
def sweatshirts_count : ℕ := 3
def t_shirts_count : ℕ := 2
def sweatshirt_cost : ℕ := 15
def t_shirt_cost : ℕ := 10

-- The total amount spent
def total_spent : ℕ := sweatshirts_count * sweatshirt_cost + t_shirts_count * t_shirt_cost

-- The theorem stating the problem
theorem hannah_spent_65 : total_spent = 65 :=
by
  sorry

end hannah_spent_65_l2_2083


namespace angle_between_hands_230_pm_l2_2159

def hour_hand_position (hour minute : ℕ) : ℕ := hour % 12 * 5 + minute / 12
def minute_hand_position (minute : ℕ) : ℕ := minute
def divisions_to_angle (divisions : ℕ) : ℕ := divisions * 30

theorem angle_between_hands_230_pm :
    hour_hand_position 2 30 = 2 * 5 + 30 / 12 ∧
    minute_hand_position 30 = 30 ∧
    divisions_to_angle (minute_hand_position 30 / 5 - hour_hand_position 2 30 / 5) = 105 :=
by {
    sorry
}

end angle_between_hands_230_pm_l2_2159


namespace problem_statement_l2_2012

-- Universal set U is the set of all real numbers
def U : Set ℝ := Set.univ

-- Definition of set M
def M : Set ℝ := { y | ∃ x : ℝ, y = 2 ^ (Real.sqrt (2 * x - x ^ 2 + 3)) }

-- Complement of M in U
def C_U_M : Set ℝ := { y | y < 1 ∨ y > 4 }

-- Definition of set N
def N : Set ℝ := { x | -3 < x ∧ x < 2 }

-- Theorem stating (C_U_M) ∩ N = (-3, 1)
theorem problem_statement : (C_U_M ∩ N) = { x | -3 < x ∧ x < 1 } :=
sorry

end problem_statement_l2_2012


namespace hyperbola_ratio_l2_2503

theorem hyperbola_ratio (a b c : ℝ)
  (h1 : a > 0) (h2 : b > 0)
  (h_eq : a^2 - b^2 = 1)
  (h_ecc : 2 = c / a)
  (h_focus : c = 1) :
  a / b = Real.sqrt 3 / 3 := by
  have ha : a = 1 / 2 := sorry
  have hc : c = 1 := h_focus
  have hb : b = Real.sqrt 3 / 2 := sorry
  exact sorry

end hyperbola_ratio_l2_2503


namespace a_2017_eq_2_l2_2990

variable (n : ℕ)
variable (S : ℕ → ℤ)

/-- Define the sequence sum Sn -/
def S_n (n : ℕ) : ℤ := 2 * n - 1

/-- Define the sequence term an -/
def a_n (n : ℕ) : ℤ := S_n n - S_n (n - 1)

theorem a_2017_eq_2 : a_n 2017 = 2 := 
by
  have hSn : ∀ n, S_n n = (2 * n - 1) := by intro; simp [S_n] 
  have ha : ∀ n, a_n n = (S_n n - S_n (n - 1)) := by intro; simp [a_n]
  simp only [ha, hSn] 
  sorry

end a_2017_eq_2_l2_2990


namespace candy_cost_correct_l2_2951

-- Given conditions:
def given_amount : ℝ := 1.00
def change_received : ℝ := 0.46

-- Define candy cost based on given conditions
def candy_cost : ℝ := given_amount - change_received

-- Statement to be proved
theorem candy_cost_correct : candy_cost = 0.54 := 
by
  sorry

end candy_cost_correct_l2_2951


namespace samantha_spends_36_dollars_l2_2100

def cost_per_toy : ℝ := 12.00
def discount_factor : ℝ := 0.5
def num_toys_bought : ℕ := 4

def total_spent (cost_per_toy : ℝ) (discount_factor : ℝ) (num_toys_bought : ℕ) : ℝ :=
  let pair_cost := cost_per_toy + (cost_per_toy * discount_factor)
  let num_pairs := num_toys_bought / 2
  num_pairs * pair_cost

theorem samantha_spends_36_dollars :
  total_spent cost_per_toy discount_factor num_toys_bought = 36.00 :=
sorry

end samantha_spends_36_dollars_l2_2100


namespace weight_of_b_l2_2560

variable (A B C : ℕ)

theorem weight_of_b 
  (h1 : A + B + C = 180) 
  (h2 : A + B = 140) 
  (h3 : B + C = 100) :
  B = 60 :=
sorry

end weight_of_b_l2_2560


namespace area_of_common_region_l2_2419

theorem area_of_common_region (β : ℝ) (h1 : 0 < β ∧ β < π / 2) (h2 : Real.cos β = 3 / 5) :
  ∃ (area : ℝ), area = 4 / 9 := 
by 
  sorry

end area_of_common_region_l2_2419


namespace geometric_vs_arithmetic_l2_2662

-- Definition of a positive geometric progression
def positive_geometric_progression (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = q * a n ∧ q > 0

-- Definition of an arithmetic progression
def arithmetic_progression (b : ℕ → ℝ) (d : ℝ) := ∀ n, b (n + 1) = b n + d

-- Theorem statement based on the problem and conditions
theorem geometric_vs_arithmetic
  (a : ℕ → ℝ) (b : ℕ → ℝ) (q : ℝ) (d : ℝ)
  (h1 : positive_geometric_progression a q)
  (h2 : arithmetic_progression b d)
  (h3 : a 6 = b 7) :
  a 3 + a 9 ≥ b 4 + b 10 := 
by 
  sorry

end geometric_vs_arithmetic_l2_2662


namespace triangle_isosceles_or_right_angled_l2_2898

theorem triangle_isosceles_or_right_angled
  (β γ : ℝ)
  (h : Real.tan β * Real.sin γ ^ 2 = Real.tan γ * Real.sin β ^ 2) :
  (β = γ ∨ β + γ = Real.pi / 2) :=
sorry

end triangle_isosceles_or_right_angled_l2_2898


namespace max_area_and_length_l2_2104

def material_cost (x y : ℝ) : ℝ :=
  900 * x + 400 * y + 200 * x * y

def area (x y : ℝ) : ℝ := x * y

theorem max_area_and_length (x y : ℝ) (h₁ : material_cost x y ≤ 32000) :
  ∃ (S : ℝ) (x : ℝ), S = 100 ∧ x = 20 / 3 :=
sorry

end max_area_and_length_l2_2104


namespace sarah_took_correct_amount_l2_2180

-- Definition of the conditions
def total_cookies : Nat := 150
def neighbors_count : Nat := 15
def correct_amount_per_neighbor : Nat := 10
def remaining_cookies : Nat := 8
def first_neighbors_count : Nat := 14
def last_neighbor : String := "Sarah"

-- Calculations based on conditions
def total_cookies_taken : Nat := total_cookies - remaining_cookies
def correct_cookies_taken : Nat := first_neighbors_count * correct_amount_per_neighbor
def extra_cookies_taken : Nat := total_cookies_taken - correct_cookies_taken
def sarah_cookies : Nat := correct_amount_per_neighbor + extra_cookies_taken

-- Proof statement: Sarah took 12 cookies
theorem sarah_took_correct_amount : sarah_cookies = 12 := by
  sorry

end sarah_took_correct_amount_l2_2180


namespace decomposition_of_5_to_4_eq_125_l2_2913

theorem decomposition_of_5_to_4_eq_125 :
  (∃ a b c : ℕ, (5^4 = a + b + c) ∧ 
                (a = 121) ∧ 
                (b = 123) ∧ 
                (c = 125)) := by 
sorry

end decomposition_of_5_to_4_eq_125_l2_2913


namespace product_of_integers_l2_2752

theorem product_of_integers (x y : ℤ) (h1 : Int.gcd x y = 5) (h2 : Int.lcm x y = 60) : x * y = 300 :=
by
  sorry

end product_of_integers_l2_2752


namespace number_of_members_l2_2186

-- Definitions based on conditions in the problem
def sock_cost : ℕ := 6
def tshirt_cost : ℕ := sock_cost + 7
def cap_cost : ℕ := tshirt_cost

def home_game_cost_per_member : ℕ := sock_cost + tshirt_cost
def away_game_cost_per_member : ℕ := sock_cost + tshirt_cost + cap_cost
def total_cost_per_member : ℕ := home_game_cost_per_member + away_game_cost_per_member

def total_league_cost : ℕ := 4324

-- Statement to be proved
theorem number_of_members (m : ℕ) (h : total_league_cost = m * total_cost_per_member) : m = 85 :=
sorry

end number_of_members_l2_2186


namespace value_of_g_at_3_l2_2467

theorem value_of_g_at_3 (g : ℕ → ℕ) (h : ∀ x, g (x + 2) = 2 * x + 3) : g 3 = 5 := by
  sorry

end value_of_g_at_3_l2_2467


namespace watch_loss_percentage_l2_2066

noncomputable def loss_percentage (CP SP_gain : ℝ) : ℝ :=
  100 * (CP - SP_gain) / CP

theorem watch_loss_percentage (CP : ℝ) (SP_gain : ℝ) :
  (SP_gain = CP + 0.04 * CP) →
  (CP = 700) →
  (CP - (SP_gain - 140) = CP * (16 / 100)) :=
by
  intros h_SP_gain h_CP
  rw [h_SP_gain, h_CP]
  simp
  sorry

end watch_loss_percentage_l2_2066


namespace length_of_train_l2_2751

def speed_kmh : ℝ := 162
def time_seconds : ℝ := 2.222044458665529
def speed_ms : ℝ := 45  -- from conversion: 162 * (1000 / 3600)

theorem length_of_train :
  (speed_kmh * (1000 / 3600)) * time_seconds = 100 := by
  -- Proof is left out
  sorry 

end length_of_train_l2_2751


namespace solve_floor_equation_l2_2044

noncomputable def x_solution_set : Set ℚ := 
  {x | x = 1 ∨ ∃ k : ℕ, 16 ≤ k ∧ k ≤ 22 ∧ x = (k : ℚ)/23 }

theorem solve_floor_equation (x : ℚ) (hx : x ∈ x_solution_set) : 
  (⌊20*x + 23⌋ : ℚ) = 20 + 23*x :=
sorry

end solve_floor_equation_l2_2044


namespace qin_jiushao_value_l2_2316

def polynomial (x : ℤ) : ℤ :=
  2 * x^5 + 5 * x^4 + 8 * x^3 + 7 * x^2 - 6 * x + 11

def step1 (x : ℤ) : ℤ := 2 * x + 5
def step2 (x : ℤ) (v : ℤ) : ℤ := v * x + 8
def step3 (x : ℤ) (v : ℤ) : ℤ := v * x + 7
def step_v3 (x : ℤ) (v : ℤ) : ℤ := v * x - 6

theorem qin_jiushao_value (x : ℤ) (v3 : ℤ) (h1 : x = 3) (h2 : v3 = 130) :
  step_v3 3 (step3 3 (step2 3 (step1 3))) = v3 :=
by {
  sorry
}

end qin_jiushao_value_l2_2316


namespace probability_of_multiples_of_4_l2_2263

def number_of_multiples_of_4 (n : ℕ) : ℕ :=
  n / 4

def number_not_multiples_of_4 (n : ℕ) (m : ℕ) : ℕ :=
  n - m

def probability_neither_multiples_of_4 (n : ℕ) (m : ℕ) : ℚ :=
  (m / n : ℚ) * (m / n)

def probability_at_least_one_multiple_of_4 (n : ℕ) (m : ℕ) : ℚ :=
  1 - probability_neither_multiples_of_4 n m

theorem probability_of_multiples_of_4 :
  probability_at_least_one_multiple_of_4 60 45 = 7 / 16 :=
by
  sorry

end probability_of_multiples_of_4_l2_2263


namespace marble_problem_l2_2848

theorem marble_problem (a : ℚ) (total : ℚ) 
  (h1 : total = a + 2 * a + 6 * a + 42 * a) :
  a = 42 / 17 :=
by 
  sorry

end marble_problem_l2_2848


namespace regular_polygon_sides_l2_2832

theorem regular_polygon_sides (n : ℕ) (h : ∀ (polygon : ℕ), (polygon = 160) → 2 < polygon ∧ (180 * (polygon - 2) / polygon) = 160) : n = 18 := 
sorry

end regular_polygon_sides_l2_2832


namespace solution_set_of_quadratic_inequality_l2_2636

theorem solution_set_of_quadratic_inequality (x : ℝ) :
  (x - 2) * (x + 2) < 5 ↔ -3 < x ∧ x < 3 :=
by 
  sorry

end solution_set_of_quadratic_inequality_l2_2636


namespace range_of_m_l2_2383

open Real

noncomputable def x (y : ℝ) : ℝ := 2 / (1 - 1 / y)

theorem range_of_m (y : ℝ) (m : ℝ) (h1 : y > 0) (h2 : 1 - 1 / y > 0) (h3 : -4 < m) (h4 : m < 2) : 
  x y + 2 * y > m^2 + 2 * m := 
by 
  have hx_pos : x y > 0 := sorry
  have hxy_eq : 2 / x y + 1 / y = 1 := sorry
  have hxy_ge : x y + 2 * y ≥ 8 := sorry
  have h_m_le : 8 > m^2 + 2 * m := sorry
  exact sorry

end range_of_m_l2_2383


namespace no_full_conspiracies_in_same_lab_l2_2154

theorem no_full_conspiracies_in_same_lab
(six_conspiracies : Finset (Finset (Fin 10)))
(h_conspiracies : ∀ c ∈ six_conspiracies, c.card = 3)
(h_total : six_conspiracies.card = 6) :
  ∃ (lab1 lab2 : Finset (Fin 10)), lab1 ∩ lab2 = ∅ ∧ lab1 ∪ lab2 = Finset.univ ∧ ∀ c ∈ six_conspiracies, ¬(c ⊆ lab1 ∨ c ⊆ lab2) :=
by
  sorry

end no_full_conspiracies_in_same_lab_l2_2154


namespace domain_shift_l2_2548

theorem domain_shift (f : ℝ → ℝ) (h : ∀ (x : ℝ), (-2 < x ∧ x < 2) → (f (x + 2) = f x)) :
  ∀ (y : ℝ), (3 < y ∧ y < 7) ↔ (y - 3 < 4 ∧ y - 3 > -2) :=
by
  sorry

end domain_shift_l2_2548


namespace perfect_squares_diff_consecutive_l2_2888

theorem perfect_squares_diff_consecutive (h1 : ∀ a : ℕ, a^2 < 1000000 → ∃ b : ℕ, a^2 = (b + 1)^2 - b^2) : 
  (∃ n : ℕ, n = 500) := 
by 
  sorry

end perfect_squares_diff_consecutive_l2_2888


namespace max_value_expression_l2_2859

theorem max_value_expression (p : ℝ) (q : ℝ) (h : q = p - 2) :
  ∃ M : ℝ, M = -70 + 96.66666666666667 ∧ (∀ p : ℝ, -3 * p^2 + 24 * p - 50 + 10 * q ≤ M) :=
sorry

end max_value_expression_l2_2859


namespace smallest_x_l2_2517

theorem smallest_x (a b x : ℤ) (h1 : x = 2 * a^5) (h2 : x = 5 * b^2) (pos_x : x > 0) : x = 200000 := sorry

end smallest_x_l2_2517


namespace simplify_triangle_expression_l2_2942

theorem simplify_triangle_expression (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  |a + b + c| - |a - b - c| - |a + b - c| = a - b + c :=
by
  sorry

end simplify_triangle_expression_l2_2942


namespace sum_of_consecutive_integers_l2_2164

theorem sum_of_consecutive_integers (x : ℤ) (h1 : x * (x + 1) + x + (x + 1) = 156) (h2 : x + 1 < 20) : x + (x + 1) = 23 :=
by
  sorry

end sum_of_consecutive_integers_l2_2164


namespace min_value_of_fraction_l2_2285

theorem min_value_of_fraction (a b : ℝ) (h_pos : a > 0 ∧ b > 0) (h_sum : a + 3 * b = 2) : 
  ∃ m, (∀ (a b : ℝ), a > 0 → b > 0 → a + 3 * b = 2 → 1 / a + 3 / b ≥ m) ∧ m = 8 := 
by
  sorry

end min_value_of_fraction_l2_2285


namespace right_isosceles_areas_no_relations_l2_2520

theorem right_isosceles_areas_no_relations :
  let W := 1 / 2 * 5 * 5
  let X := 1 / 2 * 12 * 12
  let Y := 1 / 2 * 13 * 13
  ¬ (X + Y = 2 * W + X ∨ W + X = Y ∨ 2 * X = W + Y ∨ X + W = W ∨ W + Y = 2 * X) :=
by
  sorry

end right_isosceles_areas_no_relations_l2_2520


namespace least_n_for_distance_l2_2411

theorem least_n_for_distance (n : ℕ) : n = 17 ↔ (100 ≤ n * (n + 1) / 3) := sorry

end least_n_for_distance_l2_2411


namespace nonnegative_diff_roots_eq_8sqrt2_l2_2826

noncomputable def roots_diff (a b c : ℝ) : ℝ :=
  if h : b^2 - 4*a*c ≥ 0 then 
    let root1 := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
    let root2 := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
    abs (root1 - root2)
  else 
    0

theorem nonnegative_diff_roots_eq_8sqrt2 : 
  roots_diff 1 42 409 = 8 * Real.sqrt 2 :=
sorry

end nonnegative_diff_roots_eq_8sqrt2_l2_2826


namespace find_g_5_l2_2597

theorem find_g_5 (g : ℝ → ℝ) (h : ∀ x : ℝ, g x + 3 * g (1 - x) = 2 * x ^ 2 + 1) : g 5 = 8 :=
sorry

end find_g_5_l2_2597


namespace olafs_dad_points_l2_2417

-- Let D be the number of points Olaf's dad scored.
def dad_points : ℕ := sorry

-- Olaf scored three times more points than his dad.
def olaf_points (dad_points : ℕ) : ℕ := 3 * dad_points

-- Total points scored is 28.
def total_points (dad_points olaf_points : ℕ) : Prop := dad_points + olaf_points = 28

theorem olafs_dad_points (D : ℕ) :
  (D + olaf_points D = 28) → (D = 7) :=
by
  sorry

end olafs_dad_points_l2_2417


namespace candy_bars_total_l2_2594

theorem candy_bars_total :
  let people : ℝ := 3.0;
  let candy_per_person : ℝ := 1.66666666699999;
  people * candy_per_person = 5.0 :=
by
  let people : ℝ := 3.0
  let candy_per_person : ℝ := 1.66666666699999
  show people * candy_per_person = 5.0
  sorry

end candy_bars_total_l2_2594


namespace fraction_of_constants_l2_2342

theorem fraction_of_constants :
  ∃ a b c : ℤ, (4 : ℤ) * a * (k + b)^2 + c = 4 * k^2 - 8 * k + 16 ∧
             4 * -1 * (k + (-1))^2 + 12 = 4 * k^2 - 8 * k + 16 ∧
             a = 4 ∧ b = -1 ∧ c = 12 ∧ c / b = -12 :=
by
  sorry

end fraction_of_constants_l2_2342


namespace shelves_needed_l2_2534

theorem shelves_needed (total_books : ℕ) (books_taken : ℕ) (books_per_shelf : ℕ) (remaining_books : ℕ) (shelves : ℕ) :
  total_books = 34 →
  books_taken = 7 →
  books_per_shelf = 3 →
  remaining_books = total_books - books_taken →
  shelves = remaining_books / books_per_shelf →
  shelves = 9 :=
by
  intros h_total h_taken h_per_shelf h_remaining h_shelves
  rw [h_total, h_taken, h_per_shelf] at *
  sorry

end shelves_needed_l2_2534


namespace ab_value_l2_2398

theorem ab_value (a b : ℝ) (h1 : |a| = 3) (h2 : |b - 2| = 9) (h3 : a + b > 0) :
  ab = 33 ∨ ab = -33 :=
by
  sorry

end ab_value_l2_2398


namespace foil_covered_prism_width_l2_2871

def inner_prism_dimensions (l w h : ℕ) : Prop :=
  w = 2 * l ∧ w = 2 * h ∧ l * w * h = 128

def outer_prism_width (l w h outer_width : ℕ) : Prop :=
  inner_prism_dimensions l w h ∧ outer_width = w + 2

theorem foil_covered_prism_width (l w h outer_width : ℕ) (h_inner_prism : inner_prism_dimensions l w h) :
  outer_prism_width l w h outer_width → outer_width = 10 :=
by
  intro h_outer_prism
  obtain ⟨h_w_eq, h_w_eq_2, h_volume_eq⟩ := h_inner_prism
  obtain ⟨_, h_outer_width_eq⟩ := h_outer_prism
  sorry

end foil_covered_prism_width_l2_2871


namespace initial_acidic_liquid_quantity_l2_2267

theorem initial_acidic_liquid_quantity
  (A : ℝ) -- initial quantity of the acidic liquid in liters
  (W : ℝ) -- quantity of water to be removed in liters
  (h1 : W = 6)
  (h2 : (0.40 * A) = 0.60 * (A - W)) : 
  A = 18 :=
by sorry

end initial_acidic_liquid_quantity_l2_2267


namespace moses_more_than_esther_l2_2810

theorem moses_more_than_esther (total_amount: ℝ) (moses_share: ℝ) (tony_esther_share: ℝ) :
  total_amount = 50 → moses_share = 0.40 * total_amount → 
  tony_esther_share = (total_amount - moses_share) / 2 → 
  moses_share - tony_esther_share = 5 :=
by
  intros h1 h2 h3
  sorry

end moses_more_than_esther_l2_2810


namespace number_of_balls_condition_l2_2378

theorem number_of_balls_condition (X : ℕ) (h1 : 25 - 20 = X - 25) : X = 30 :=
by
  sorry

end number_of_balls_condition_l2_2378


namespace min_value_of_sum_l2_2224

theorem min_value_of_sum (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : 1 / x + 1 / y + 1 / z = 1) :
  x + 4 * y + 9 * z ≥ 36 ∧ (x + 4 * y + 9 * z = 36 ↔ x = 6 ∧ y = 3 ∧ z = 2) := 
sorry

end min_value_of_sum_l2_2224


namespace f_3_1_plus_f_3_4_l2_2619

def f (a b : ℕ) : ℚ :=
  if a + b < 5 then (a * b - a + 4) / (2 * a)
  else (a * b - b - 5) / (-2 * b)

theorem f_3_1_plus_f_3_4 :
  f 3 1 + f 3 4 = 7 / 24 :=
by
  sorry

end f_3_1_plus_f_3_4_l2_2619


namespace perimeter_after_growth_operations_perimeter_after_four_growth_operations_l2_2912

theorem perimeter_after_growth_operations (initial_perimeter : ℝ) (growth_factor : ℝ) (growth_steps : ℕ):
  initial_perimeter = 27 ∧ growth_factor = 4/3 ∧ growth_steps = 2 → 
    initial_perimeter * growth_factor^growth_steps = 48 :=
by
  sorry

theorem perimeter_after_four_growth_operations (initial_perimeter : ℝ) (growth_factor : ℝ) (growth_steps : ℕ):
  initial_perimeter = 27 ∧ growth_factor = 4/3 ∧ growth_steps = 4 → 
    initial_perimeter * growth_factor^growth_steps = 256/3 :=
by
  sorry

end perimeter_after_growth_operations_perimeter_after_four_growth_operations_l2_2912


namespace perfect_square_sequence_l2_2603

theorem perfect_square_sequence (x : ℕ → ℤ) (h₀ : x 0 = 0) (h₁ : x 1 = 3) 
  (h₂ : ∀ n, x (n + 1) + x (n - 1) = 4 * x n) : 
  ∀ n, ∃ k : ℤ, x (n + 1) * x (n - 1) + 9 = k^2 :=
by 
  sorry

end perfect_square_sequence_l2_2603


namespace inequality_log_l2_2634

variable (a b c : ℝ)
variable (h1 : 1 < a)
variable (h2 : 1 < b)
variable (h3 : 1 < c)

theorem inequality_log (a b c : ℝ) (h1 : 1 < a) (h2 : 1 < b) (h3 : 1 < c) : 
  2 * ( (Real.log a / Real.log b) / (a + b) + (Real.log b / Real.log c) / (b + c) + (Real.log c / Real.log a) / (c + a) ) 
  ≥ 9 / (a + b + c) := 
sorry

end inequality_log_l2_2634


namespace alcohol_water_ratio_l2_2253

theorem alcohol_water_ratio (alcohol water : ℝ) (h_alcohol : alcohol = 3 / 5) (h_water : water = 2 / 5) :
  alcohol / water = 3 / 2 :=
by 
  sorry

end alcohol_water_ratio_l2_2253


namespace intersection_A_B_l2_2228

open Set

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {x | ∃ a ∈ A, x = 2 * a}

theorem intersection_A_B : A ∩ B = {0, 2} :=
by
  sorry

end intersection_A_B_l2_2228


namespace sum_of_cubes_is_24680_l2_2739

noncomputable def jake_age := 10
noncomputable def amy_age := 12
noncomputable def ryan_age := 28

theorem sum_of_cubes_is_24680 (j a r : ℕ) (h1 : 2 * j + 3 * a = 4 * r)
  (h2 : j^3 + a^3 = 1 / 2 * r^3) (h3 : j + a + r = 50) : j^3 + a^3 + r^3 = 24680 :=
by
  sorry

end sum_of_cubes_is_24680_l2_2739


namespace cost_price_l2_2762

theorem cost_price (SP : ℝ) (profit_percentage : ℝ) : SP = 600 ∧ profit_percentage = 60 → ∃ CP : ℝ, CP = 375 :=
by
  intro h
  sorry

end cost_price_l2_2762


namespace find_y_l2_2089

theorem find_y (n x y : ℝ)
  (h1 : (100 + 200 + n + x) / 4 = 250)
  (h2 : (n + 150 + 100 + x + y) / 5 = 200) :
  y = 50 :=
by
  sorry

end find_y_l2_2089


namespace marbles_count_l2_2197

theorem marbles_count (initial_marble: ℕ) (bought_marble: ℕ) (final_marble: ℕ) 
  (h1: initial_marble = 53) (h2: bought_marble = 134) : 
  final_marble = initial_marble + bought_marble -> final_marble = 187 :=
by
  intros h3
  rw [h1, h2] at h3
  exact h3

-- sorry is omitted as proof is given.

end marbles_count_l2_2197


namespace avg_score_first_4_l2_2487

-- Definitions based on conditions
def average_score_all_7 : ℝ := 56
def total_matches : ℕ := 7
def average_score_last_3 : ℝ := 69.33333333333333
def matches_first : ℕ := 4
def matches_last : ℕ := 3

-- Calculation of total runs from average scores.
def total_runs_all_7 : ℝ := average_score_all_7 * total_matches
def total_runs_last_3 : ℝ := average_score_last_3 * matches_last

-- Total runs for the first 4 matches
def total_runs_first_4 : ℝ := total_runs_all_7 - total_runs_last_3

-- Prove the average score for the first 4 matches.
theorem avg_score_first_4 :
  (total_runs_first_4 / matches_first) = 46 := 
sorry

end avg_score_first_4_l2_2487


namespace hank_donated_percentage_l2_2105

variable (A_c D_c A_b D_b A_l D_t D_l p : ℝ) (h1 : A_c = 100) (h2 : D_c = 0.90 * A_c)
variable (h3 : A_b = 80) (h4 : D_b = 0.75 * A_b) (h5 : A_l = 50) (h6 : D_t = 200)

theorem hank_donated_percentage :
  D_l = D_t - (D_c + D_b) → 
  p = (D_l / A_l) * 100 → 
  p = 100 :=
by
  sorry

end hank_donated_percentage_l2_2105


namespace solution_volume_l2_2860

theorem solution_volume (x : ℝ) (h1 : (0.16 * x) / (x + 13) = 0.0733333333333333) : x = 11 :=
by sorry

end solution_volume_l2_2860


namespace one_in_M_l2_2626

def N := { x : ℕ | true } -- Define the natural numbers ℕ

def M : Set ℕ := { x ∈ N | 1 / (x - 2) ≤ 0 }

theorem one_in_M : 1 ∈ M :=
  sorry

end one_in_M_l2_2626


namespace range_of_a_l2_2578

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  if x < 1 then (2 - a) * x - 3 * a + 3 
  else Real.log x / Real.log a

-- Main statement to prove
theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ (5 / 4 ≤ a ∧ a < 2) :=
sorry

end range_of_a_l2_2578


namespace spotted_and_fluffy_cats_l2_2624

theorem spotted_and_fluffy_cats (total_cats : ℕ) (total_cats_equiv : total_cats = 120) (one_third_spotted : ℕ → ℕ) (one_fourth_fluffy_spotted : ℕ → ℕ) :
  (one_third_spotted total_cats * one_fourth_fluffy_spotted (one_third_spotted total_cats) = 10) :=
by
  sorry

end spotted_and_fluffy_cats_l2_2624


namespace range_f_l2_2172

noncomputable def g (x : ℝ) : ℝ := 30 + 14 * Real.cos x - 7 * Real.cos (2 * x)

noncomputable def z (t : ℝ) : ℝ := 40.5 - 14 * (t - 0.5) ^ 2

noncomputable def u (z : ℝ) : ℝ := (Real.pi / 54) * z

noncomputable def f (x : ℝ) : ℝ := Real.sin (u (z (Real.cos x)))

theorem range_f : ∀ x : ℝ, 0.5 ≤ f x ∧ f x ≤ 1 :=
by
  intro x
  sorry

end range_f_l2_2172


namespace sum_of_squares_l2_2008

-- Define conditions
def condition1 (a b : ℝ) : Prop := a - b = 6
def condition2 (a b : ℝ) : Prop := a * b = 7

-- Define what we want to prove
def target (a b : ℝ) : Prop := a^2 + b^2 = 50

-- Main theorem stating the required proof
theorem sum_of_squares (a b : ℝ) (h1 : condition1 a b) (h2 : condition2 a b) : target a b :=
by sorry

end sum_of_squares_l2_2008


namespace min_xy_value_min_x_plus_y_value_l2_2212

variable {x y : ℝ}

theorem min_xy_value (hx : 0 < x) (hy : 0 < y) (h : (2 / y) + (8 / x) = 1) : xy ≥ 64 := 
sorry

theorem min_x_plus_y_value (hx : 0 < x) (hy : 0 < y) (h : (2 / y) + (8 / x) = 1) : x + y ≥ 18 :=
sorry

end min_xy_value_min_x_plus_y_value_l2_2212


namespace solve_quadratic_l2_2872

theorem solve_quadratic {x : ℝ} (h : 2 * (x - 1)^2 = x - 1) : x = 1 ∨ x = 3 / 2 :=
sorry

end solve_quadratic_l2_2872


namespace sqrt_18_mul_sqrt_6_sqrt_8_sub_sqrt_2_add_2_sqrt_half_sqrt_12_mul_sqrt_9_div_3_div_sqrt_3_div_3_sqrt_7_add_sqrt_5_mul_sqrt_7_sub_sqrt_5_l2_2765

-- Problem 1
theorem sqrt_18_mul_sqrt_6 : (Real.sqrt 18 * Real.sqrt 6 = 6 * Real.sqrt 3) :=
sorry

-- Problem 2
theorem sqrt_8_sub_sqrt_2_add_2_sqrt_half : (Real.sqrt 8 - Real.sqrt 2 + 2 * Real.sqrt (1 / 2) = 3 * Real.sqrt 2) :=
sorry

-- Problem 3
theorem sqrt_12_mul_sqrt_9_div_3_div_sqrt_3_div_3 : (Real.sqrt 12 * (Real.sqrt 9 / 3) / (Real.sqrt 3 / 3) = 6) :=
sorry

-- Problem 4
theorem sqrt_7_add_sqrt_5_mul_sqrt_7_sub_sqrt_5 : ((Real.sqrt 7 + Real.sqrt 5) * (Real.sqrt 7 - Real.sqrt 5) = 2) :=
sorry

end sqrt_18_mul_sqrt_6_sqrt_8_sub_sqrt_2_add_2_sqrt_half_sqrt_12_mul_sqrt_9_div_3_div_sqrt_3_div_3_sqrt_7_add_sqrt_5_mul_sqrt_7_sub_sqrt_5_l2_2765


namespace angle_in_second_quadrant_l2_2374

theorem angle_in_second_quadrant (n : ℤ) : (460 : ℝ) = 360 * n + 100 := by
  sorry

end angle_in_second_quadrant_l2_2374


namespace central_angle_of_regular_hexagon_l2_2111

theorem central_angle_of_regular_hexagon:
  ∀ (α : ℝ), 
  (∃ n : ℕ, n = 6 ∧ n * α = 360) →
  α = 60 :=
by
  sorry

end central_angle_of_regular_hexagon_l2_2111


namespace problem_f_2010_l2_2588

noncomputable def f : ℝ → ℝ := sorry

axiom f_1 : f 1 = 1 / 4
axiom f_eq : ∀ x y : ℝ, 4 * f x * f y = f (x + y) + f (x - y)

theorem problem_f_2010 : f 2010 = 1 / 2 :=
sorry

end problem_f_2010_l2_2588


namespace find_number_l2_2708

theorem find_number (x : ℕ) (h : x / 4 + 3 = 5) : x = 8 :=
by sorry

end find_number_l2_2708


namespace how_many_kids_joined_l2_2308

theorem how_many_kids_joined (original_kids : ℕ) (new_kids : ℕ) (h : original_kids = 14) (h1 : new_kids = 36) :
  new_kids - original_kids = 22 :=
by
  sorry

end how_many_kids_joined_l2_2308


namespace farmer_eggs_per_week_l2_2830

theorem farmer_eggs_per_week (E : ℝ) (chickens : ℝ) (price_per_dozen : ℝ) (total_revenue : ℝ) (num_weeks : ℝ) (total_chickens : ℝ) (dozen : ℝ) 
    (H1 : total_chickens = 46)
    (H2 : price_per_dozen = 3)
    (H3 : total_revenue = 552)
    (H4 : num_weeks = 8)
    (H5 : dozen = 12)
    (H6 : chickens = 46)
    : E = 6 :=
by
  sorry

end farmer_eggs_per_week_l2_2830


namespace magicStack_cardCount_l2_2828

-- Define the conditions and question based on a)
def isMagicStack (n : ℕ) : Prop :=
  let totalCards := 2 * n
  ∃ (A B : Finset ℕ), (A ∪ B = Finset.range totalCards) ∧
    (∀ x ∈ A, x < n) ∧ (∀ x ∈ B, x ≥ n) ∧
    (∀ i ∈ A, i % 2 = 1) ∧ (∀ j ∈ B, j % 2 = 0) ∧
    (151 ∈ A) ∧
    ∃ (newStack : Finset ℕ), (newStack = A ∪ B) ∧
    (∀ k ∈ newStack, k ∈ A ∨ k ∈ B) ∧
    (151 = 151)

-- The theorem that states the number of cards, when card 151 retains its position, is 452.
theorem magicStack_cardCount :
  isMagicStack 226 → 2 * 226 = 452 :=
by
  sorry

end magicStack_cardCount_l2_2828


namespace length_of_tunnel_l2_2343

theorem length_of_tunnel (time : ℝ) (speed : ℝ) (train_length : ℝ) (total_distance : ℝ) (tunnel_length : ℝ) 
  (h1 : time = 30) (h2 : speed = 100 / 3) (h3 : train_length = 400) (h4 : total_distance = speed * time) 
  (h5 : tunnel_length = total_distance - train_length) : 
  tunnel_length = 600 :=
by
  sorry

end length_of_tunnel_l2_2343


namespace largest_k_for_3_in_g_l2_2193

theorem largest_k_for_3_in_g (k : ℝ) :
  (∃ x : ℝ, 2*x^2 - 8*x + k = 3) ↔ k ≤ 11 :=
by
  sorry

end largest_k_for_3_in_g_l2_2193


namespace correct_operation_l2_2621

theorem correct_operation :
  (∀ {a : ℝ}, a^6 / a^3 = a^3) = false ∧
  (∀ {a b : ℝ}, (a + b) * (a - b) = a^2 - b^2) ∧
  (∀ {a : ℝ}, (-a^3)^3 = -a^9) = false ∧
  (∀ {a : ℝ}, 2 * a^2 + 3 * a^3 = 5 * a^5) = false :=
by
  sorry

end correct_operation_l2_2621


namespace least_integer_square_double_l2_2299

theorem least_integer_square_double (x : ℤ) : x^2 = 2 * x + 50 → x = -5 :=
by
  sorry

end least_integer_square_double_l2_2299


namespace technicians_count_l2_2686

theorem technicians_count 
    (total_workers : ℕ) (avg_salary_all : ℕ) (avg_salary_technicians : ℕ) (avg_salary_rest : ℕ)
    (h_workers : total_workers = 28) (h_avg_all : avg_salary_all = 8000) 
    (h_avg_tech : avg_salary_technicians = 14000) (h_avg_rest : avg_salary_rest = 6000) : 
    ∃ T R : ℕ, T + R = total_workers ∧ (avg_salary_technicians * T + avg_salary_rest * R = avg_salary_all * total_workers) ∧ T = 7 :=
by
  sorry

end technicians_count_l2_2686


namespace range_of_a_l2_2835

theorem range_of_a (a : ℝ) : (∀ x : ℕ, 4 * x + a ≤ 5 → x ≥ 1 → x ≤ 3) ↔ (-11 < a ∧ a ≤ -7) :=
by sorry

end range_of_a_l2_2835


namespace average_book_width_l2_2241

noncomputable def bookWidths : List ℝ := [5, 0.75, 1.5, 3, 12, 2, 7.5]

theorem average_book_width :
  (bookWidths.sum / bookWidths.length = 4.54) :=
by
  sorry

end average_book_width_l2_2241


namespace slices_with_both_onions_and_olives_l2_2836

noncomputable def slicesWithBothToppings (total_slices slices_with_onions slices_with_olives : Nat) : Nat :=
  slices_with_onions + slices_with_olives - total_slices

theorem slices_with_both_onions_and_olives 
  (total_slices : Nat) (slices_with_onions : Nat) (slices_with_olives : Nat) :
  total_slices = 18 ∧ slices_with_onions = 10 ∧ slices_with_olives = 10 →
  slicesWithBothToppings total_slices slices_with_onions slices_with_olives = 2 :=
by
  sorry

end slices_with_both_onions_and_olives_l2_2836


namespace mabel_age_l2_2430

theorem mabel_age (n : ℕ) (h : n * (n + 1) / 2 = 28) : n = 7 :=
sorry

end mabel_age_l2_2430


namespace roots_interlaced_l2_2335

variable {α : Type*} [LinearOrderedField α]
variables {f g : α → α}

theorem roots_interlaced
    (x1 x2 x3 x4 : α)
    (h1 : x1 < x2) (h2 : x3 < x4)
    (hfx1 : f x1 = 0) (hfx2 : f x2 = 0)
    (hfx_distinct : x1 ≠ x2)
    (hgx3 : g x3 = 0) (hgx4 : g x4 = 0)
    (hgx_distinct : x3 ≠ x4)
    (hgx1_ne_0 : g x1 ≠ 0) (hgx2_ne_0 : g x2 ≠ 0)
    (hgx1_gx2_lt_0 : g x1 * g x2 < 0) :
    (x1 < x3 ∧ x3 < x2 ∧ x2 < x4) ∨ (x3 < x1 ∧ x1 < x4 ∧ x4 < x2) :=
sorry

end roots_interlaced_l2_2335


namespace ratio_w_y_l2_2346

theorem ratio_w_y (w x y z : ℚ) 
  (h1 : w / x = 5 / 2) 
  (h2 : y / z = 5 / 3) 
  (h3 : z / x = 1 / 6) : 
  w / y = 9 := 
by 
  sorry

end ratio_w_y_l2_2346


namespace expression_value_l2_2167

theorem expression_value (x y z : ℕ) (hx : x = 3) (hy : y = 2) (hz : z = 4) :
  3 * x - 2 * y + 4 * z = 21 :=
by
  subst hx
  subst hy
  subst hz
  sorry

end expression_value_l2_2167


namespace library_books_l2_2098

/-- Last year, the school library purchased 50 new books. 
    This year, it purchased 3 times as many books. 
    If the library had 100 books before it purchased new books last year,
    prove that the library now has 300 books in total. -/
theorem library_books (initial_books : ℕ) (last_year_books : ℕ) (multiplier : ℕ)
  (h1 : initial_books = 100) (h2 : last_year_books = 50) (h3 : multiplier = 3) :
  initial_books + last_year_books + (multiplier * last_year_books) = 300 := 
sorry

end library_books_l2_2098


namespace smallest_value_n_l2_2085

theorem smallest_value_n :
  ∃ (n : ℕ), n * 25 = Nat.lcm (Nat.lcm 10 18) 20 ∧ (∀ m, m * 25 = Nat.lcm (Nat.lcm 10 18) 20 → n ≤ m) := 
sorry

end smallest_value_n_l2_2085


namespace sequence_of_arrows_l2_2394

theorem sequence_of_arrows (n : ℕ) (h : n % 5 = 0) : 
  (n < 570 ∧ n % 5 = 0) → 
  (n + 1 < 573 ∧ (n + 1) % 5 = 1) → 
  (n + 2 < 573 ∧ (n + 2) % 5 = 2) → 
  (n + 3 < 573 ∧ (n + 3) % 5 = 3) →
    true :=
by
  sorry

end sequence_of_arrows_l2_2394


namespace possible_N_l2_2796

/-- 
  Let N be an integer with N ≥ 3, and let a₀, a₁, ..., a_(N-1) be pairwise distinct reals such that 
  aᵢ ≥ a_(2i mod N) for all i. Prove that N must be a power of 2.
-/
theorem possible_N (N : ℕ) (hN : N ≥ 3) (a : Fin N → ℝ) (h_distinct: Function.Injective a) 
  (h_condition : ∀ i : Fin N, a i ≥ a (⟨(2 * i) % N, sorry⟩)) 
  : ∃ k : ℕ, N = 2^k := 
sorry

end possible_N_l2_2796


namespace probability_at_least_6_heads_in_8_flips_l2_2283

def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then (Nat.choose n k) else 0

theorem probability_at_least_6_heads_in_8_flips :
  let total_outcomes := 2^8
  let successful_outcomes := binomial 8 6 + binomial 8 7 + binomial 8 8
  let probability := (successful_outcomes : ℚ) / total_outcomes
  probability = 37 / 256 :=
by
  sorry

end probability_at_least_6_heads_in_8_flips_l2_2283


namespace prime_factorial_division_l2_2019

theorem prime_factorial_division (p k n : ℕ) (hp : Prime p) (h : p^k ∣ n!) : (p!)^k ∣ n! :=
sorry

end prime_factorial_division_l2_2019


namespace percentage_increase_is_200_l2_2573

noncomputable def total_cost : ℝ := 300
noncomputable def rate_per_sq_m : ℝ := 5
noncomputable def length : ℝ := 13.416407864998739
noncomputable def area : ℝ := total_cost / rate_per_sq_m
noncomputable def breadth : ℝ := area / length
noncomputable def percentage_increase : ℝ := (length - breadth) / breadth * 100

theorem percentage_increase_is_200 :
  percentage_increase = 200 :=
by
  sorry

end percentage_increase_is_200_l2_2573


namespace percentageReduction_l2_2248

variable (R P : ℝ)

def originalPrice (R : ℝ) (P : ℝ) : Prop :=
  2400 / R - 2400 / P = 8 ∧ R = 120

theorem percentageReduction : 
  originalPrice 120 P → ((P - 120) / P) * 100 = 40 := 
by
  sorry

end percentageReduction_l2_2248


namespace total_fish_l2_2640

-- Conditions
def initial_fish : ℕ := 22
def given_fish : ℕ := 47

-- Question: Total fish Mrs. Sheridan has now
theorem total_fish : initial_fish + given_fish = 69 := by
  sorry

end total_fish_l2_2640


namespace max_sum_ac_bc_l2_2784

noncomputable def triangle_ab_bc_sum_max (AB : ℝ) (C : ℝ) : ℝ :=
  if AB = Real.sqrt 6 - Real.sqrt 2 ∧ C = Real.pi / 6 then 4 else 0

theorem max_sum_ac_bc {A B C : ℝ} (h1 : AB = Real.sqrt 6 - Real.sqrt 2) (h2 : C = Real.pi / 6) :
  triangle_ab_bc_sum_max AB C = 4 :=
by {
  sorry
}

end max_sum_ac_bc_l2_2784


namespace rectangle_area_change_area_analysis_l2_2284

noncomputable def original_area (a b : ℝ) : ℝ := a * b

noncomputable def new_area (a b : ℝ) : ℝ := (a - 3) * (b + 3)

theorem rectangle_area_change (a b : ℝ) :
  let S := original_area a b
  let S₁ := new_area a b
  S₁ - S = 3 * (a - b - 3) :=
by
  sorry

theorem area_analysis (a b : ℝ) :
  if a - b - 3 = 0 then new_area a b = original_area a b
  else if a - b - 3 > 0 then new_area a b > original_area a b
  else new_area a b < original_area a b :=
by
  sorry

end rectangle_area_change_area_analysis_l2_2284


namespace cuboid_inequality_l2_2276

theorem cuboid_inequality 
  (a b c : ℝ) 
  (h : a^2 + b^2 + c^2 = 1) : 
  4*a + 4*b + 4*c + 4*a*b + 4*a*c + 4*b*c + 4*a*b*c < 12 := by
  sorry

end cuboid_inequality_l2_2276


namespace Seokjin_total_fish_l2_2547

-- Define the conditions
def fish_yesterday := 10
def cost_yesterday := 3000
def additional_cost := 6000
def price_per_fish := cost_yesterday / fish_yesterday
def total_cost_today := cost_yesterday + additional_cost
def fish_today := total_cost_today / price_per_fish

-- Define the goal
theorem Seokjin_total_fish (h1 : fish_yesterday = 10)
                           (h2 : cost_yesterday = 3000)
                           (h3 : additional_cost = 6000)
                           (h4 : price_per_fish = cost_yesterday / fish_yesterday)
                           (h5 : total_cost_today = cost_yesterday + additional_cost)
                           (h6 : fish_today = total_cost_today / price_per_fish) :
  fish_yesterday + fish_today = 40 :=
by
  sorry

end Seokjin_total_fish_l2_2547


namespace second_crane_height_l2_2932

noncomputable def height_of_second_crane : ℝ :=
  let crane1 := 228
  let building1 := 200
  let building2 := 100
  let crane3 := 147
  let building3 := 140
  let avg_building_height := (building1 + building2 + building3) / 3
  let avg_crane_height := avg_building_height * 1.13
  let h := (avg_crane_height * 3) - (crane1 - building1 + crane3 - building3) + building2
  h

theorem second_crane_height : height_of_second_crane = 122 := 
  sorry

end second_crane_height_l2_2932


namespace marshmallow_ratio_l2_2094

theorem marshmallow_ratio:
  (∀ h m b, 
    h = 8 ∧ 
    m = 3 * h ∧ 
    h + m + b = 44
  ) → (1 / 2 = b / m) :=
by
sorry

end marshmallow_ratio_l2_2094


namespace necessary_and_sufficient_condition_extremum_l2_2025

noncomputable def f (a x : ℝ) : ℝ := a * x^3 + 6 * x^2 + (a - 1) * x - 5

theorem necessary_and_sufficient_condition_extremum (a : ℝ) :
  (∃ x, f a x = 0) ↔ -3 < a ∧ a < 4 :=
sorry

end necessary_and_sufficient_condition_extremum_l2_2025


namespace largest_divisor_of_m_l2_2437

theorem largest_divisor_of_m (m : ℤ) (hm_pos : 0 < m) (h : 33 ∣ m^2) : 33 ∣ m :=
sorry

end largest_divisor_of_m_l2_2437


namespace joan_initial_balloons_l2_2659

-- Definitions using conditions from a)
def initial_balloons (lost : ℕ) (current : ℕ) : ℕ := lost + current

-- Statement of our equivalent math proof problem
theorem joan_initial_balloons : initial_balloons 2 7 = 9 := 
by
  -- Proof skipped using sorry
  sorry

end joan_initial_balloons_l2_2659


namespace no_solution_equation_l2_2237

theorem no_solution_equation (x : ℝ) : (x + 1) / (x - 1) + 4 / (1 - x^2) ≠ 1 :=
  sorry

end no_solution_equation_l2_2237


namespace minutes_in_hours_l2_2963

theorem minutes_in_hours (h : ℝ) (m : ℝ) (H : h = 3.5) (M : m = 60) : h * m = 210 := by
  sorry

end minutes_in_hours_l2_2963


namespace determine_sixth_face_l2_2290

-- Define a cube configuration and corresponding functions
inductive Color
| black
| white

structure Cube where
  faces : Fin 6 → Fin 9 → Color

noncomputable def sixth_face_color (cube : Cube) : Fin 9 → Color := sorry

-- The statement of the theorem proving the coloring of the sixth face
theorem determine_sixth_face (cube : Cube) : 
  (exists f : (Fin 9 → Color), f = sixth_face_color cube) := 
sorry

end determine_sixth_face_l2_2290


namespace problem1_solution_problem2_solution_l2_2857

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 4|
def g (x : ℝ) : ℝ := |2 * x + 1|

-- Problem 1
theorem problem1_solution :
  {x : ℝ | f x < g x} = {x : ℝ | x < -5 ∨ x > 1} :=
sorry

-- Problem 2
theorem problem2_solution :
  ∀ (a : ℝ), (∀ x : ℝ, 2 * f x + g x > a * x) ↔ -4 ≤ a ∧ a < 9 / 4 :=
sorry

end problem1_solution_problem2_solution_l2_2857


namespace expression_value_l2_2080

theorem expression_value : 4 * (8 - 2) ^ 2 - 6 = 138 :=
by
  sorry

end expression_value_l2_2080


namespace car_mileage_city_l2_2268

theorem car_mileage_city (h c t : ℕ) 
  (h_eq_tank_mileage : 462 = h * t) 
  (c_eq_tank_mileage : 336 = c * t) 
  (mileage_diff : c = h - 3) : 
  c = 8 := 
by
  sorry

end car_mileage_city_l2_2268


namespace find_amount_l2_2070

def total_amount (A : ℝ) : Prop :=
  A / 20 = A / 25 + 100

theorem find_amount 
  (A : ℝ) 
  (h : total_amount A) : 
  A = 10000 := 
  sorry

end find_amount_l2_2070


namespace dartboard_area_ratio_l2_2184

theorem dartboard_area_ratio
  (side_length : ℝ)
  (h_side_length : side_length = 2)
  (t : ℝ)
  (q : ℝ)
  (h_t : t = (1 / 2) * (1 / (Real.sqrt 2)) * (1 / (Real.sqrt 2)))
  (h_q : q = ((side_length * side_length) - (8 * t)) / 4) :
  q / t = 2 := by
  sorry

end dartboard_area_ratio_l2_2184


namespace product_of_roots_eq_neg25_l2_2992

theorem product_of_roots_eq_neg25 : 
  ∀ (x : ℝ), 24 * x^2 + 36 * x - 600 = 0 → x * (x - ((-36 - 24 * x)/24)) = -25 :=
by
  sorry

end product_of_roots_eq_neg25_l2_2992


namespace maddie_weekend_watch_time_l2_2794

-- Defining the conditions provided in the problem
def num_episodes : ℕ := 8
def duration_per_episode : ℕ := 44
def minutes_on_monday : ℕ := 138
def minutes_on_tuesday : ℕ := 0
def minutes_on_wednesday : ℕ := 0
def minutes_on_thursday : ℕ := 21
def episodes_on_friday : ℕ := 2

-- Define the total time watched from Monday to Friday
def total_minutes_week : ℕ := num_episodes * duration_per_episode
def total_minutes_mon_to_fri : ℕ := 
  minutes_on_monday + 
  minutes_on_tuesday + 
  minutes_on_wednesday + 
  minutes_on_thursday + 
  (episodes_on_friday * duration_per_episode)

-- Define the weekend watch time
def weekend_watch_time : ℕ := total_minutes_week - total_minutes_mon_to_fri

-- The theorem to prove the correct answer
theorem maddie_weekend_watch_time : weekend_watch_time = 105 := by
  sorry

end maddie_weekend_watch_time_l2_2794


namespace math_problem_l2_2902

variable (x y : ℚ)

theorem math_problem (h : 1.5 * x = 0.04 * y) : (y - x) / (y + x) = 73 / 77 := by
  sorry

end math_problem_l2_2902


namespace sum_of_three_numbers_l2_2574

def a : ℚ := 859 / 10
def b : ℚ := 531 / 100
def c : ℚ := 43 / 2

theorem sum_of_three_numbers : a + b + c = 11271 / 100 := by
  sorry

end sum_of_three_numbers_l2_2574


namespace repeating_decimal_sum_l2_2194

theorem repeating_decimal_sum :
  let x := (1 : ℚ) / 3
  let y := (7 : ℚ) / 33
  x + y = 6 / 11 :=
  by
  sorry

end repeating_decimal_sum_l2_2194


namespace company_production_l2_2480

theorem company_production (bottles_per_case number_of_cases total_bottles : ℕ)
  (h1 : bottles_per_case = 12)
  (h2 : number_of_cases = 10000)
  (h3 : total_bottles = number_of_cases * bottles_per_case) : 
  total_bottles = 120000 :=
by {
  -- Proof is omitted, add actual proof here
  sorry
}

end company_production_l2_2480


namespace number_of_recipes_l2_2136

-- Let's define the necessary conditions.
def cups_per_recipe : ℕ := 2
def total_cups_needed : ℕ := 46

-- Prove that the number of recipes required is 23.
theorem number_of_recipes : total_cups_needed / cups_per_recipe = 23 :=
by
  sorry

end number_of_recipes_l2_2136


namespace solve_system_of_equations_l2_2420

-- Conditions from the problem
variables (x y : ℚ)

-- Definitions (the original equations)
def equation1 := x + 2 * y = 3
def equation2 := 9 * x - 8 * y = 5

-- Correct answer
def solution_x := 17 / 13
def solution_y := 11 / 13

-- The final proof statement
theorem solve_system_of_equations (h1 : equation1 solution_x solution_y) (h2 : equation2 solution_x solution_y) :
  x = solution_x ∧ y = solution_y := sorry

end solve_system_of_equations_l2_2420


namespace exists_X_Y_sum_not_in_third_subset_l2_2834

open Nat Set

theorem exists_X_Y_sum_not_in_third_subset :
  ∀ (M_1 M_2 M_3 : Set ℕ), 
  Disjoint M_1 M_2 ∧ Disjoint M_2 M_3 ∧ Disjoint M_1 M_3 → 
  ∃ (X Y : ℕ), (X ∈ M_1 ∪ M_2 ∪ M_3) ∧ (Y ∈ M_1 ∪ M_2 ∪ M_3) ∧  
  (X ∈ M_1 → Y ∈ M_2 ∨ Y ∈ M_3) ∧
  (X ∈ M_2 → Y ∈ M_1 ∨ Y ∈ M_3) ∧
  (X ∈ M_3 → Y ∈ M_1 ∨ Y ∈ M_2) ∧
  (X + Y ∉ M_3) :=
by
  intros M_1 M_2 M_3 disj
  sorry

end exists_X_Y_sum_not_in_third_subset_l2_2834


namespace heart_beats_during_marathon_l2_2436

theorem heart_beats_during_marathon :
  (∃ h_per_min t1 t2 total_time,
    h_per_min = 140 ∧
    t1 = 15 * 6 ∧
    t2 = 15 * 5 ∧
    total_time = t1 + t2 ∧
    23100 = h_per_min * total_time) :=
  sorry

end heart_beats_during_marathon_l2_2436


namespace grandmother_cheapest_option_l2_2998

-- Conditions definition
def cost_of_transportation : Nat := 200
def berries_collected : Nat := 5
def market_price_berries : Nat := 150
def price_sugar : Nat := 54
def amount_jam_from_1kg_berries_sugar : ℚ := 1.5
def cost_ready_made_jam_per_kg : Nat := 220

-- Calculations
def cost_per_kg_berries : ℚ := cost_of_transportation / berries_collected
def cost_bought_berries : Nat := market_price_berries
def total_cost_1kg_self_picked : ℚ := cost_per_kg_berries + price_sugar
def total_cost_1kg_bought : Nat := cost_bought_berries + price_sugar
def total_cost_1_5kg_self_picked : ℚ := total_cost_1kg_self_picked
def total_cost_1_5kg_bought : ℚ := total_cost_1kg_bought
def total_cost_1_5kg_ready_made : ℚ := cost_ready_made_jam_per_kg * amount_jam_from_1kg_berries_sugar

theorem grandmother_cheapest_option :
  total_cost_1_5kg_self_picked ≤ total_cost_1_5kg_bought ∧ 
  total_cost_1_5kg_self_picked ≤ total_cost_1_5kg_ready_made :=
  by
    sorry

end grandmother_cheapest_option_l2_2998


namespace largest_integer_less_than_100_with_remainder_5_l2_2410

theorem largest_integer_less_than_100_with_remainder_5 (n : ℤ) (h₁ : n < 100) (h₂ : n % 8 = 5) : n ≤ 99 :=
sorry

end largest_integer_less_than_100_with_remainder_5_l2_2410


namespace number_of_groups_l2_2761

noncomputable def original_students : ℕ := 22 + 2

def students_per_group : ℕ := 8

theorem number_of_groups : original_students / students_per_group = 3 :=
by
  sorry

end number_of_groups_l2_2761


namespace seq_properties_l2_2629

noncomputable def f (x : ℝ) : ℝ := (1 / 3) ^ x

theorem seq_properties :
  (∀ n, a_n = -2 * (1 / 3) ^ n) ∧
  (∀ n, b_n = 2 * n - 1) ∧
  (∀ t m, (-1 ≤ m ∧ m ≤ 1) → (t^2 - 2 * m * t + 1/2 > T_n) ↔ (t < -2 ∨ t > 2)) ∧
  (∃ m n, 1 < m ∧ m < n ∧ T_1 * T_n = T_m^2 ∧ m = 2 ∧ n = 12) :=
sorry

end seq_properties_l2_2629


namespace max_value_expression_l2_2451

noncomputable def factorize_15000 := 2^3 * 3 * 5^4

theorem max_value_expression (x y : ℕ) (h1 : 6 * x^2 - 5 * x * y + y^2 = 0) (h2 : x ∣ factorize_15000) : 
  2 * x + 3 * y ≤ 60000 := sorry

end max_value_expression_l2_2451


namespace average_salary_8800_l2_2017

theorem average_salary_8800 
  (average_salary_start : ℝ)
  (salary_jan : ℝ)
  (salary_may : ℝ)
  (total_salary : ℝ)
  (avg_specific_months : ℝ)
  (jan_salary_rate : average_salary_start * 4 = total_salary)
  (may_salary_rate : total_salary - salary_jan = total_salary - 3300)
  (final_salary_rate : total_salary - salary_jan + salary_may = 35200)
  (specific_avg_calculation : 35200 / 4 = avg_specific_months)
  : avg_specific_months = 8800 :=
sorry -- Proof steps will be filled in later

end average_salary_8800_l2_2017


namespace bridge_length_l2_2331

theorem bridge_length 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (cross_time_seconds : ℝ)
  (train_length_eq : train_length = 150)
  (train_speed_kmph_eq : train_speed_kmph = 45)
  (cross_time_seconds_eq : cross_time_seconds = 30) : 
  ∃ (bridge_length : ℝ), bridge_length = 225 := 
  by
  sorry

end bridge_length_l2_2331


namespace bag_with_cracks_number_l2_2956

def marbles : List ℕ := [18, 19, 21, 23, 25, 34]

def total_marbles : ℕ := marbles.sum

def modulo_3 (n : ℕ) : ℕ := n % 3

theorem bag_with_cracks_number :
  ∃ (c : ℕ), c ∈ marbles ∧ 
    (total_marbles - c) % 3 = 0 ∧
    c = 23 :=
by 
  sorry

end bag_with_cracks_number_l2_2956


namespace simplify_and_evaluate_expression_l2_2126

theorem simplify_and_evaluate_expression (a b : ℝ) (h1 : a = -1) (h2 : a * b = 2) :
  3 * (2 * a^2 * b + a * b^2) - (3 * a * b^2 - a^2 * b) = -14 := by
  sorry

end simplify_and_evaluate_expression_l2_2126


namespace marly_needs_3_bags_l2_2475

-- Definitions based on the problem conditions
def milk : ℕ := 2
def chicken_stock : ℕ := 3 * milk
def vegetables : ℕ := 1
def total_soup : ℕ := milk + chicken_stock + vegetables
def bag_capacity : ℕ := 3

-- The theorem to prove the number of bags required
theorem marly_needs_3_bags : total_soup / bag_capacity = 3 := 
sorry

end marly_needs_3_bags_l2_2475


namespace exist_x_y_l2_2533

theorem exist_x_y (a b c : ℝ) (h₁ : abs a > 2) (h₂ : a^2 + b^2 + c^2 = a * b * c + 4) :
  ∃ x y : ℝ, a = x + 1/x ∧ b = y + 1/y ∧ c = x*y + 1/(x*y) :=
sorry

end exist_x_y_l2_2533


namespace fraction_of_field_planted_l2_2221

theorem fraction_of_field_planted (a b : ℕ) (d : ℝ) :
  a = 5 → b = 12 → d = 3 →
  let hypotenuse := Real.sqrt (a^2 + b^2)
  let side_square := (d * hypotenuse - d^2)/(a + b - 2 * d)
  let area_square := side_square^2
  let area_triangle : ℝ := 1/2 * a * b
  let planted_area := area_triangle - area_square
  let fraction_planted := planted_area / area_triangle
  fraction_planted = 9693/10140 := by
  sorry

end fraction_of_field_planted_l2_2221


namespace ratio_of_P_Q_l2_2327

theorem ratio_of_P_Q (P Q : ℤ)
  (h : ∀ x : ℝ, x ≠ -5 → x ≠ 0 → x ≠ 4 →
    P / (x + 5) + Q / (x^2 - 4 * x) = (x^2 + x + 15) / (x^3 + x^2 - 20 * x)) :
  Q / P = -45 / 2 :=
by
  sorry

end ratio_of_P_Q_l2_2327


namespace find_cans_lids_l2_2933

-- Define the given conditions
def total_lids (x : ℕ) : ℕ := 14 + 3 * x

-- Define the proof problem
theorem find_cans_lids (x : ℕ) (h : total_lids x = 53) : x = 13 :=
sorry

end find_cans_lids_l2_2933


namespace jane_doe_total_investment_mutual_funds_l2_2651

theorem jane_doe_total_investment_mutual_funds :
  ∀ (c m : ℝ) (total_investment : ℝ),
  total_investment = 250000 → m = 3 * c → c + m = total_investment → m = 187500 :=
by
  intros c m total_investment h_total h_relation h_sum
  sorry

end jane_doe_total_investment_mutual_funds_l2_2651


namespace monthly_price_reduction_rate_l2_2150

-- Let's define the given conditions
def initial_price_March : ℝ := 23000
def price_in_May : ℝ := 16000

-- Define the monthly average price reduction rate
variable (x : ℝ)

-- Define the statement to be proven
theorem monthly_price_reduction_rate :
  23 * (1 - x) ^ 2 = 16 :=
sorry

end monthly_price_reduction_rate_l2_2150


namespace number_before_star_is_five_l2_2809

theorem number_before_star_is_five (n : ℕ) (h1 : n % 72 = 0) (h2 : n % 10 = 0) (h3 : ∃ k, n = 400 + 10 * k) : (n / 10) % 10 = 5 :=
sorry

end number_before_star_is_five_l2_2809


namespace mean_greater_than_median_by_l2_2506

-- Define the data: number of students missing specific days
def studentsMissingDays := [3, 1, 4, 1, 1, 5] -- corresponding to 0, 1, 2, 3, 4, 5 days missed

-- Total number of students
def totalStudents := 15

-- Function to calculate the sum of missed days weighted by the number of students
def totalMissedDays := (0 * 3) + (1 * 1) + (2 * 4) + (3 * 1) + (4 * 1) + (5 * 5)

-- Calculate the mean number of missed days
def meanDaysMissed := totalMissedDays / totalStudents

-- Select the median number of missed days (8th student) from the ordered list
def medianDaysMissed := 2

-- Calculate the difference between the mean and median
def difference := meanDaysMissed - medianDaysMissed

-- Define the proof problem statement
theorem mean_greater_than_median_by : 
  difference = 11 / 15 :=
by
  -- This is where the actual proof would be written
  sorry

end mean_greater_than_median_by_l2_2506


namespace probability_exactly_one_win_l2_2671

theorem probability_exactly_one_win :
  let P_win_Jp := 2 / 3
  let P_win_Us := 2 / 5
  let P_exactly_one_win := P_win_Jp * (1 - P_win_Us) + (1 - P_win_Jp) * P_win_Us
  P_exactly_one_win = 8 / 15 :=
by
  let P_win_Jp := 2 / 3
  let P_win_Us := 2 / 5
  let P_exactly_one_win := P_win_Jp * (1 - P_win_Us) + (1 - P_win_Jp) * P_win_Us
  have h1 : P_exactly_one_win = 8 / 15 := sorry
  exact h1

end probability_exactly_one_win_l2_2671


namespace money_constraints_l2_2804

variable (a b : ℝ)

theorem money_constraints (h1 : 8 * a - b = 98) (h2 : 2 * a + b > 36) : a > 13.4 ∧ b > 9.2 :=
sorry

end money_constraints_l2_2804


namespace one_half_percent_as_decimal_l2_2729

def percent_to_decimal (x : ℚ) := x / 100

theorem one_half_percent_as_decimal : percent_to_decimal (1 / 2) = 0.005 := 
by
  sorry

end one_half_percent_as_decimal_l2_2729


namespace initial_population_l2_2692

theorem initial_population (P : ℝ) (h1 : ∀ n : ℕ, n = 2 → P * (0.7 ^ n) = 3920) : P = 8000 := by
  sorry

end initial_population_l2_2692


namespace subtract_and_convert_l2_2787

theorem subtract_and_convert : (3/4 - 1/16 : ℚ) = 0.6875 :=
by
  sorry

end subtract_and_convert_l2_2787


namespace lcm_and_sum_of_14_21_35_l2_2680

def lcm_of_numbers_and_sum (a b c : ℕ) : ℕ × ℕ :=
  (Nat.lcm (Nat.lcm a b) c, a + b + c)

theorem lcm_and_sum_of_14_21_35 :
  lcm_of_numbers_and_sum 14 21 35 = (210, 70) :=
  sorry

end lcm_and_sum_of_14_21_35_l2_2680


namespace percentage_of_mothers_l2_2551

open Real

-- Define the constants based on the conditions provided.
def P : ℝ := sorry -- Total number of parents surveyed
def M : ℝ := sorry -- Number of mothers
def F : ℝ := sorry -- Number of fathers

-- The equations derived from the conditions.
axiom condition1 : M + F = P
axiom condition2 : (1/8)*M + (1/4)*F = 17.5/100 * P

-- The proof goal: to show the percentage of mothers.
theorem percentage_of_mothers :
  M / P = 3 / 5 :=
by
  -- Proof goes here
  sorry

end percentage_of_mothers_l2_2551


namespace chenny_candies_l2_2489

def friends_count : ℕ := 7
def candies_per_friend : ℕ := 2
def candies_have : ℕ := 10

theorem chenny_candies : 
    (friends_count * candies_per_friend - candies_have) = 4 := by
    sorry

end chenny_candies_l2_2489


namespace carX_travel_distance_after_carY_started_l2_2355

-- Define the conditions
def carX_speed : ℝ := 35
def carY_speed : ℝ := 40
def delay_time : ℝ := 1.2

-- Define the problem to prove the question is equal to the correct answer given the conditions
theorem carX_travel_distance_after_carY_started : 
  ∃ t : ℝ, carY_speed * t = carX_speed * t + carX_speed * delay_time ∧ 
           carX_speed * t = 294 :=
by
  sorry

end carX_travel_distance_after_carY_started_l2_2355


namespace sin_75_is_sqrt_6_add_sqrt_2_div_4_l2_2591

noncomputable def sin_75_angle (a : Real) (b : Real) : Real :=
  Real.sin (75 * Real.pi / 180)

theorem sin_75_is_sqrt_6_add_sqrt_2_div_4 :
  sin_75_angle π (π / 6) = (Real.sqrt 6 + Real.sqrt 2) / 4 :=
by
  sorry

end sin_75_is_sqrt_6_add_sqrt_2_div_4_l2_2591


namespace functional_eq_uniq_l2_2473

noncomputable def f (x : ℝ) : ℝ := sorry

theorem functional_eq_uniq (f : ℝ → ℝ) (h : ∀ x y : ℝ, (f x * f y - f (x * y)) / 4 = x^2 + y^2 + 2) : 
  ∀ x : ℝ, f x = x^2 + 3 :=
by 
  sorry

end functional_eq_uniq_l2_2473


namespace science_books_have_9_copies_l2_2133

theorem science_books_have_9_copies :
  ∃ (A B C D : ℕ), A + B + C + D = 35 ∧ A + B = 17 ∧ B + C = 16 ∧ A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧ B = 9 :=
by
  sorry

end science_books_have_9_copies_l2_2133


namespace number_of_different_pairs_l2_2364

theorem number_of_different_pairs :
  let mystery := 4
  let fantasy := 4
  let science_fiction := 4
  (mystery * fantasy) + (mystery * science_fiction) + (fantasy * science_fiction) = 48 :=
by
  let mystery := 4
  let fantasy := 4
  let science_fiction := 4
  show (mystery * fantasy) + (mystery * science_fiction) + (fantasy * science_fiction) = 48
  sorry

end number_of_different_pairs_l2_2364


namespace work_rate_B_l2_2592

theorem work_rate_B :
  (∀ A B : ℝ, A = 30 → (1 / A + 1 / B = 1 / 19.411764705882355) → B = 55) := by 
    intro A B A_cond combined_rate
    have hA : A = 30 := A_cond
    rw [hA] at combined_rate
    sorry

end work_rate_B_l2_2592


namespace sequences_properties_l2_2938

-- Definitions for properties P and P'
def is_property_P (seq : List ℕ) : Prop := sorry
def is_property_P' (seq : List ℕ) : Prop := sorry

-- Define sequences
def sequence1 := [1, 2, 3, 1]
def sequence2 := [1, 234, 5]  -- Extend as needed

-- Conditions
def bn_is_permutation_of_an (a b : List ℕ) : Prop := sorry -- Placeholder for permutation check

-- Main Statement 
theorem sequences_properties :
  is_property_P sequence1 ∧
  is_property_P' sequence2 := 
by
  sorry

-- Additional theorem to check permutation if needed
-- theorem permutation_check :
--  bn_is_permutation_of_an sequence1 sequence2 :=
-- by
--  sorry

end sequences_properties_l2_2938


namespace circle_tangent_to_directrix_and_yaxis_on_parabola_l2_2535

noncomputable def circle1_eq (x y : ℝ) := (x - 1)^2 + (y - 1 / 2)^2 = 1
noncomputable def circle2_eq (x y : ℝ) := (x + 1)^2 + (y - 1 / 2)^2 = 1

theorem circle_tangent_to_directrix_and_yaxis_on_parabola :
  ∀ (x y : ℝ), (x^2 = 2 * y) → 
  ((y = -1 / 2 → circle1_eq x y) ∨ (y = -1 / 2 → circle2_eq x y)) :=
by
  intro x y h_parabola
  sorry

end circle_tangent_to_directrix_and_yaxis_on_parabola_l2_2535


namespace negation_proof_l2_2135

theorem negation_proof (a b : ℝ) : 
  (¬ (a > b → 2 * a > 2 * b - 1)) = (a ≤ b → 2 * a ≤ 2 * b - 1) :=
by
  sorry

end negation_proof_l2_2135


namespace solve_expression_l2_2179

theorem solve_expression (x y : ℝ) (h : (x + y - 2020) * (2023 - x - y) = 2) :
  (x + y - 2020)^2 * (2023 - x - y)^2 = 4 := by
  sorry

end solve_expression_l2_2179


namespace ratio_areas_of_circumscribed_circles_l2_2210

theorem ratio_areas_of_circumscribed_circles (P : ℝ) (A B : ℝ)
  (h1 : ∃ (x : ℝ), P = 8 * x)
  (h2 : ∃ (s : ℝ), s = P / 3)
  (hA : A = (5 * (P^2) * Real.pi) / 128)
  (hB : B = (P^2 * Real.pi) / 27) :
  A / B = 135 / 128 := by
  sorry

end ratio_areas_of_circumscribed_circles_l2_2210


namespace Catriona_goldfish_count_l2_2051

theorem Catriona_goldfish_count (G : ℕ) (A : ℕ) (U : ℕ) 
    (h1 : A = G + 4) 
    (h2 : U = 2 * A) 
    (h3 : G + A + U = 44) : G = 8 :=
by
  -- Proof goes here
  sorry

end Catriona_goldfish_count_l2_2051


namespace proof_m_n_sum_l2_2618

-- Definitions based on conditions
def m : ℕ := 2
def n : ℕ := 49

-- Problem statement as a Lean theorem
theorem proof_m_n_sum : m + n = 51 :=
by
  -- This is where the detailed proof would go. Using sorry to skip the proof.
  sorry

end proof_m_n_sum_l2_2618


namespace absolute_value_inequality_solution_l2_2227

theorem absolute_value_inequality_solution (x : ℝ) : abs (x - 3) < 2 ↔ 1 < x ∧ x < 5 :=
by
  sorry

end absolute_value_inequality_solution_l2_2227


namespace hyperbola_center_l2_2057

theorem hyperbola_center : ∃ c : ℝ × ℝ, c = (3, 5) ∧
  ∀ x y : ℝ, 9 * x ^ 2 - 54 * x - 36 * y ^ 2 + 360 * y - 891 = 0 → (c.1 = 3 ∧ c.2 = 5) :=
by
  use (3, 5)
  sorry

end hyperbola_center_l2_2057


namespace fill_tank_with_leak_l2_2617

theorem fill_tank_with_leak (P L T : ℝ) 
  (hP : P = 1 / 2)  -- Rate of the pump
  (hL : L = 1 / 6)  -- Rate of the leak
  (hT : T = 3)  -- Time taken to fill the tank with the leak
  : 1 / (P - L) = T := 
by
  sorry

end fill_tank_with_leak_l2_2617


namespace smallest_m_for_probability_l2_2538

-- Define the conditions in Lean
def nonWithInTwoUnits (x y z : ℝ) : Prop :=
  abs (x - y) ≥ 2 ∧ abs (y - z) ≥ 2 ∧ abs (z - x) ≥ 2

def probabilityCondition (m : ℝ) : Prop :=
  (m - 4)^3 / m^3 > 2/3

-- The theorem statement
theorem smallest_m_for_probability : ∃ m : ℕ, 0 < m ∧ (∀ x y z : ℝ, 0 ≤ x ∧ x ≤ m ∧ 0 ≤ y ∧ y ≤ m ∧ 0 ≤ z ∧ z ≤ m → nonWithInTwoUnits x y z) → probabilityCondition m ∧ m = 14 :=
by sorry

end smallest_m_for_probability_l2_2538


namespace tan_3theta_eq_9_13_l2_2045

open Real

noncomputable def tan3theta (θ : ℝ) (h : tan θ = 3) : Prop :=
  tan (3 * θ) = (9 / 13)

theorem tan_3theta_eq_9_13 (θ : ℝ) (h : tan θ = 3) : tan3theta θ h :=
by
  sorry

end tan_3theta_eq_9_13_l2_2045


namespace probability_sum_divisible_by_3_l2_2920

theorem probability_sum_divisible_by_3:
  ∀ (n a b c : ℕ), a + b + c = n →
  4 * (a^3 + b^3 + c^3 + 6 * a * b * c) ≥ (a + b + c)^3 :=
by 
  intros n a b c habc_eq_n
  sorry

end probability_sum_divisible_by_3_l2_2920


namespace percentage_passed_all_three_l2_2110

variable (F_H F_E F_M F_HE F_EM F_HM F_HEM : ℝ)

theorem percentage_passed_all_three :
  F_H = 0.46 →
  F_E = 0.54 →
  F_M = 0.32 →
  F_HE = 0.18 →
  F_EM = 0.12 →
  F_HM = 0.1 →
  F_HEM = 0.06 →
  (100 - (F_H + F_E + F_M - F_HE - F_EM - F_HM + F_HEM)) = 2 :=
by sorry

end percentage_passed_all_three_l2_2110


namespace angle_Z_is_120_l2_2156

-- Define angles and lines
variables {p q : Prop} {X Y Z : ℝ}
variables (h_parallel : p ∧ q)
variables (hX : X = 100)
variables (hY : Y = 140)

-- Proof statement: Given the angles X and Y, we prove that angle Z is 120 degrees.
theorem angle_Z_is_120 (h_parallel : p ∧ q) (hX : X = 100) (hY : Y = 140) : Z = 120 := by 
  -- Here we would add the proof steps
  sorry

end angle_Z_is_120_l2_2156


namespace tic_tac_toe_alex_wins_second_X_l2_2448

theorem tic_tac_toe_alex_wins_second_X :
  ∃ b : ℕ, b = 12 := 
sorry

end tic_tac_toe_alex_wins_second_X_l2_2448


namespace digit_relationship_l2_2818

theorem digit_relationship (d1 d2 : ℕ) (h1 : d1 * 10 + d2 = 16) (h2 : d1 + d2 = 7) : d2 = 6 * d1 :=
by
  sorry

end digit_relationship_l2_2818


namespace circumscribed_circle_radius_l2_2109

noncomputable def radius_of_circumscribed_circle (a b : ℝ) : ℝ :=
  (Real.sqrt (a^2 + b^2)) / 2

theorem circumscribed_circle_radius (a r l b R : ℝ)
  (h1 : r = 1)
  (h2 : a = 2 * Real.sqrt 3)
  (h3 : b = 3)
  (h4 : l = a)
  (h5 : R = radius_of_circumscribed_circle l b) :
  R = Real.sqrt 21 / 2 :=
by
  sorry

end circumscribed_circle_radius_l2_2109


namespace common_points_intervals_l2_2062

noncomputable def h (x : ℝ) : ℝ := (2 * Real.log x) / x

theorem common_points_intervals (a : ℝ) (h₀ : 1 < a) : 
  (∀ f g : ℝ → ℝ, (f x = a ^ x) → (g x = x ^ 2) → 
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ = g x₁ ∧ f x₂ = g x₂ ∧ f x₃ = g x₃) → 
  a < Real.exp (2 / Real.exp 1) :=
by
  sorry

end common_points_intervals_l2_2062


namespace find_real_triples_l2_2514

theorem find_real_triples :
  ∀ (a b c : ℝ), a^2 + a * b + c = 0 ∧ b^2 + b * c + a = 0 ∧ c^2 + c * a + b = 0
  ↔ (a = 0 ∧ b = 0 ∧ c = 0) ∨ (a = -1/2 ∧ b = -1/2 ∧ c = -1/2) :=
by
  sorry

end find_real_triples_l2_2514


namespace find_integer_a_l2_2727

theorem find_integer_a (x d e a : ℤ) :
  ((x - a)*(x - 8) - 3 = (x + d)*(x + e)) → (a = 6) :=
by
  sorry

end find_integer_a_l2_2727


namespace find_fraction_l2_2544

-- Let f be a real number representing the fraction
theorem find_fraction (f : ℝ) (h : f * 12 + 5 = 11) : f = 1 / 2 := 
by
  sorry

end find_fraction_l2_2544


namespace polynomial_expansion_l2_2153

theorem polynomial_expansion (t : ℝ) :
  (3 * t^3 + 2 * t^2 - 4 * t + 3) * (-2 * t^2 + 3 * t - 4) =
  -6 * t^5 + 5 * t^4 + 2 * t^3 - 26 * t^2 + 25 * t - 12 :=
by sorry

end polynomial_expansion_l2_2153


namespace investment_period_l2_2026

theorem investment_period (P A : ℝ) (r n t : ℝ)
  (hP : P = 4000)
  (hA : A = 4840.000000000001)
  (hr : r = 0.10)
  (hn : n = 1)
  (hC : A = P * (1 + r / n) ^ (n * t)) :
  t = 2 := by
-- Adding a sorry to skip the actual proof.
sorry

end investment_period_l2_2026


namespace fraction_irreducible_l2_2200

theorem fraction_irreducible (n : ℤ) : Nat.gcd (18 * n + 3).natAbs (12 * n + 1).natAbs = 1 := 
sorry

end fraction_irreducible_l2_2200


namespace proposition_R_is_converse_negation_of_P_l2_2187

variables (x y : ℝ)

def P : Prop := x + y = 0 → x = -y
def Q : Prop := ¬(x + y = 0) → x ≠ -y
def R : Prop := x ≠ -y → ¬(x + y = 0)

theorem proposition_R_is_converse_negation_of_P : R x y ↔ ¬P x y :=
by sorry

end proposition_R_is_converse_negation_of_P_l2_2187


namespace find_age_of_b_l2_2604

variable (a b : ℤ)

-- Conditions
axiom cond1 : a + 10 = 2 * (b - 10)
axiom cond2 : a = b + 9

-- Goal
theorem find_age_of_b : b = 39 :=
sorry

end find_age_of_b_l2_2604


namespace blue_to_red_ratio_l2_2363

-- Define the conditions as given in the problem
def initial_red_balls : ℕ := 16
def lost_red_balls : ℕ := 6
def bought_yellow_balls : ℕ := 32
def total_balls_after_events : ℕ := 74

-- Based on the conditions, we define the remaining red balls and the total balls equation
def remaining_red_balls := initial_red_balls - lost_red_balls

-- Suppose B is the number of blue balls
def blue_balls (B : ℕ) : Prop :=
  remaining_red_balls + B + bought_yellow_balls = total_balls_after_events

-- Now, state the theorem to prove the ratio of blue balls to red balls is 16:5
theorem blue_to_red_ratio (B : ℕ) (h : blue_balls B) : B = 32 → B / remaining_red_balls = 16 / 5 :=
by
  intro B_eq
  subst B_eq
  have h1 : remaining_red_balls = 10 := rfl
  have h2 : 32 / 10  = 16 / 5 := by rfl
  exact h2

-- Note: The proof itself is skipped, so the statement is left with sorry.

end blue_to_red_ratio_l2_2363


namespace smallest_m_l2_2262

theorem smallest_m (m : ℤ) :
  (∀ x : ℝ, (3 * x * (m * x - 5) - x^2 + 8) = 0) → (257 - 96 * m < 0) → (m = 3) :=
sorry

end smallest_m_l2_2262


namespace oxen_grazing_months_l2_2497

theorem oxen_grazing_months (a_oxen : ℕ) (a_months : ℕ) (b_oxen : ℕ) (c_oxen : ℕ) (c_months : ℕ) (total_rent : ℝ) (c_share_rent : ℝ) (x : ℕ) :
  a_oxen = 10 →
  a_months = 7 →
  b_oxen = 12 →
  c_oxen = 15 →
  c_months = 3 →
  total_rent = 245 →
  c_share_rent = 63 →
  (c_oxen * c_months) / ((a_oxen * a_months) + (b_oxen * x) + (c_oxen * c_months)) = c_share_rent / total_rent →
  x = 5 :=
sorry

end oxen_grazing_months_l2_2497


namespace modular_inverse_example_l2_2895

open Int

theorem modular_inverse_example :
  ∃ b : ℤ, 0 ≤ b ∧ b < 120 ∧ (7 * b) % 120 = 1 ∧ b = 103 :=
by
  sorry

end modular_inverse_example_l2_2895


namespace max_profit_l2_2405

-- Define the given conditions
def cost_price : ℝ := 80
def sales_relationship (x : ℝ) : ℝ := -0.5 * x + 160
def selling_price_range (x : ℝ) : Prop := 120 ≤ x ∧ x ≤ 180

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - cost_price) * sales_relationship x

-- The goal: prove the maximum profit and the selling price that achieves it
theorem max_profit : ∃ (x : ℝ), selling_price_range x ∧ profit x = 7000 := 
  sorry

end max_profit_l2_2405


namespace knights_count_l2_2758

theorem knights_count (T F : ℕ) (h1 : T + F = 65) (h2 : ∀ n < 21, ¬(T = F - 20)) 
  (h3 : ∀ n ≥ 21, if n % 2 = 1 then T = (n - 1) / 2 + 1 else T = (n - 1) / 2):
  T = 23 :=
by
      -- Here the specific steps of the proof will go
      sorry

end knights_count_l2_2758


namespace minimum_guests_at_banquet_l2_2041

theorem minimum_guests_at_banquet (total_food : ℝ) (max_food_per_guest : ℝ) (min_guests : ℕ) 
  (h1 : total_food = 411) (h2 : max_food_per_guest = 2.5) : min_guests = 165 :=
by
  -- Proof omitted
  sorry

end minimum_guests_at_banquet_l2_2041


namespace range_of_f_l2_2401

theorem range_of_f (x : ℝ) (h1 : 1 ≤ x) (h2 : x ≤ 2) : -3 ≤ (3^x - 6/x) ∧ (3^x - 6/x) ≤ 6 :=
by
  sorry

end range_of_f_l2_2401


namespace pentagon_rectangle_ratio_l2_2911

theorem pentagon_rectangle_ratio (p w : ℝ) 
    (pentagon_perimeter : 5 * p = 30) 
    (rectangle_perimeter : ∃ l, 2 * w + 2 * l = 30 ∧ l = 2 * w) : 
    p / w = 6 / 5 := 
by
  sorry

end pentagon_rectangle_ratio_l2_2911


namespace S₉_eq_81_l2_2795

variable (aₙ : ℕ → ℕ) (S : ℕ → ℕ)
variable (n : ℕ)
variable (a₁ d : ℕ)

-- Conditions
axiom S₃_eq_9 : S 3 = 9
axiom S₆_eq_36 : S 6 = 36
axiom S_n_def : ∀ n, S n = n * a₁ + n * (n - 1) / 2 * d

-- Proof obligation
theorem S₉_eq_81 : S 9 = 81 :=
by
  sorry

end S₉_eq_81_l2_2795


namespace percentage_of_number_l2_2243

theorem percentage_of_number (X P : ℝ) (h1 : 0.20 * X = 80) (h2 : (P / 100) * X = 160) : P = 40 := by
  sorry

end percentage_of_number_l2_2243


namespace tan_sub_eq_one_eight_tan_add_eq_neg_four_seven_l2_2325

variable (α β : ℝ)

theorem tan_sub_eq_one_eight (h1 : Real.tan α = 5) (h2 : Real.tan β = 3) : 
  Real.tan (α - β) = 1 / 8 := 
sorry

theorem tan_add_eq_neg_four_seven (h1 : Real.tan α = 5) (h2 : Real.tan β = 3) : 
  Real.tan (α + β) = -4 / 7 := 
sorry

end tan_sub_eq_one_eight_tan_add_eq_neg_four_seven_l2_2325


namespace find_x_l2_2532

def vector_a (x : ℝ) : ℝ × ℝ := (2, x)
def vector_b : ℝ × ℝ := (-3, 2)

def is_perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem find_x (x : ℝ) (h : is_perpendicular (vector_a x + vector_b) vector_b) : 
  x = -7 / 2 :=
  sorry

end find_x_l2_2532


namespace solve_for_x_l2_2820

theorem solve_for_x : ∃ x : ℝ, (x + 36) / 3 = (7 - 2 * x) / 6 ∧ x = -65 / 4 := by
  sorry

end solve_for_x_l2_2820


namespace sum_of_tripled_numbers_l2_2024

theorem sum_of_tripled_numbers (a b S : ℤ) (h : a + b = S) : 3 * (a + 5) + 3 * (b + 5) = 3 * S + 30 :=
by
  sorry

end sum_of_tripled_numbers_l2_2024


namespace sasha_made_50_muffins_l2_2869

/-- 
Sasha made some chocolate muffins for her school bake sale fundraiser. Melissa made 4 times as many 
muffins as Sasha, and Tiffany made half of Sasha and Melissa's total number of muffins. They 
contributed $900 to the fundraiser by selling muffins at $4 each. Prove that Sasha made 50 muffins.
-/
theorem sasha_made_50_muffins 
  (S : ℕ)
  (Melissa_made : ℕ := 4 * S)
  (Tiffany_made : ℕ := (1 / 2) * (S + Melissa_made))
  (Total_muffins : ℕ := S + Melissa_made + Tiffany_made)
  (total_income : ℕ := 900)
  (price_per_muffin : ℕ := 4)
  (muffins_sold : ℕ := total_income / price_per_muffin)
  (eq_muffins_sold : Total_muffins = muffins_sold) : 
  S = 50 := 
by sorry

end sasha_made_50_muffins_l2_2869


namespace savings_of_person_l2_2827

theorem savings_of_person (income expenditure : ℕ) (h_ratio : 3 * expenditure = 2 * income) (h_income : income = 21000) :
  income - expenditure = 7000 :=
by
  sorry

end savings_of_person_l2_2827


namespace no_nontrivial_integer_solutions_l2_2149

theorem no_nontrivial_integer_solutions (x y z : ℤ) : x^3 + 2*y^3 + 4*z^3 - 6*x*y*z = 0 -> x = 0 ∧ y = 0 ∧ z = 0 :=
by
  sorry

end no_nontrivial_integer_solutions_l2_2149


namespace maximum_tied_teams_round_robin_l2_2204

noncomputable def round_robin_tournament_max_tied_teams (n : ℕ) : ℕ := 
  sorry

theorem maximum_tied_teams_round_robin (h : n = 8) : round_robin_tournament_max_tied_teams n = 7 :=
sorry

end maximum_tied_teams_round_robin_l2_2204


namespace circle_radius_l2_2360

noncomputable def circle_problem (rD rE : ℝ) (m n : ℝ) :=
  rD = 2 * rE ∧
  rD = (Real.sqrt m) - n ∧
  m ≥ 0 ∧ n ≥ 0

theorem circle_radius (rE rD : ℝ) (m n : ℝ) (h : circle_problem rD rE m n) :
  m + n = 5.76 :=
by
  sorry

end circle_radius_l2_2360


namespace locker_number_problem_l2_2484

theorem locker_number_problem 
  (cost_per_digit : ℝ)
  (total_cost : ℝ)
  (one_digit_cost : ℝ)
  (two_digit_cost : ℝ)
  (three_digit_cost : ℝ) :
  cost_per_digit = 0.03 →
  one_digit_cost = 0.27 →
  two_digit_cost = 5.40 →
  three_digit_cost = 81.00 →
  total_cost = 206.91 →
  10 * cost_per_digit = six_cents →
  9 * cost_per_digit = three_cents →
  1 * 9 * cost_per_digit = one_digit_cost →
  2 * 45 * cost_per_digit = two_digit_cost →
  3 * 300 * cost_per_digit = three_digit_cost →
  (999 * 3 + x * 4 = 6880) →
  ∀ total_locker : ℕ, total_locker = 2001 := sorry

end locker_number_problem_l2_2484


namespace meatballs_left_l2_2737
open Nat

theorem meatballs_left (meatballs_per_plate sons : ℕ)
  (hp : meatballs_per_plate = 3) 
  (hs : sons = 3) 
  (fraction_eaten : ℚ)
  (hf : fraction_eaten = 2 / 3): 
  (meatballs_per_plate - meatballs_per_plate * fraction_eaten) * sons = 3 := by
  -- Placeholder proof; the details would be filled in by a full proof.
  sorry

end meatballs_left_l2_2737


namespace maximum_sin_C_in_triangle_l2_2880

theorem maximum_sin_C_in_triangle 
  (A B C : ℝ)
  (h1 : A + B + C = π) 
  (h2 : 1 / Real.tan A + 1 / Real.tan B = 6 / Real.tan C) : 
  Real.sin C = Real.sqrt 15 / 4 :=
sorry

end maximum_sin_C_in_triangle_l2_2880


namespace hexagon_area_correct_m_plus_n_l2_2069

noncomputable def hexagon_area (b : ℝ) : ℝ :=
  let A := (0, 0)
  let B := (b, 3)
  let F := (-3 * (3 + b) / 2, 9)  -- derived from complex numbers and angle conversion
  let hexagon_height := 12  -- height difference between the y-coordinates
  let hexagon_base := 3 * (b + 3) / 2  -- distance between parallel lines AB and DE
  36 / 2 * (b + 3) + 6 * (6 + b * Real.sqrt 3)

theorem hexagon_area_correct (b : ℝ) :
  hexagon_area b = 72 * Real.sqrt 3 :=
sorry

theorem m_plus_n : 72 + 3 = 75 := rfl

end hexagon_area_correct_m_plus_n_l2_2069


namespace prove_relationship_l2_2001

noncomputable def relationship_x_y_z (x y z : ℝ) (t : ℝ) : Prop :=
  (x / Real.sin t) = (y / Real.sin (2 * t)) ∧ (x / Real.sin t) = (z / Real.sin (3 * t))

theorem prove_relationship (x y z t : ℝ) (h : relationship_x_y_z x y z t) : x^2 - y^2 + x * z = 0 :=
by
  sorry

end prove_relationship_l2_2001


namespace water_park_admission_l2_2610

def adult_admission_charge : ℝ := 1
def child_admission_charge : ℝ := 0.75
def children_accompanied : ℕ := 3
def total_admission_charge (adults : ℝ) (children : ℝ) : ℝ := adults + children

theorem water_park_admission :
  let adult_charge := adult_admission_charge
  let children_charge := children_accompanied * child_admission_charge
  total_admission_charge adult_charge children_charge = 3.25 :=
by sorry

end water_park_admission_l2_2610


namespace time_for_B_alone_l2_2747

theorem time_for_B_alone (h1 : 4 * (1/15 + 1/x) = 7/15) : x = 20 :=
sorry

end time_for_B_alone_l2_2747


namespace magnitude_squared_complex_l2_2121

noncomputable def complex_number := Complex.mk 3 (-4)
noncomputable def squared_complex := complex_number * complex_number

theorem magnitude_squared_complex : Complex.abs squared_complex = 25 :=
by
  sorry

end magnitude_squared_complex_l2_2121


namespace range_of_m_l2_2959

noncomputable def A (x : ℝ) : Prop := |x - 2| ≤ 4
noncomputable def B (x : ℝ) (m : ℝ) : Prop := (x - 1 - m) * (x - 1 + m) ≤ 0 ∧ m > 0

theorem range_of_m (m : ℝ) :
  (∀ x, (¬A x) → (¬B x m)) ∧ (∃ x, (¬B x m) ∧ ¬(¬A x)) → m ≥ 5 :=
sorry

end range_of_m_l2_2959


namespace lily_pads_cover_half_l2_2239

theorem lily_pads_cover_half (P D : ℕ) (cover_entire : P * (2 ^ 25) = D) : P * (2 ^ 24) = D / 2 :=
by sorry

end lily_pads_cover_half_l2_2239


namespace staff_members_attended_meeting_l2_2191

theorem staff_members_attended_meeting
  (n_doughnuts_served : ℕ)
  (e_each_staff_member : ℕ)
  (n_doughnuts_left : ℕ)
  (h1 : n_doughnuts_served = 50)
  (h2 : e_each_staff_member = 2)
  (h3 : n_doughnuts_left = 12) :
  (n_doughnuts_served - n_doughnuts_left) / e_each_staff_member = 19 := 
by
  sorry

end staff_members_attended_meeting_l2_2191


namespace smallest_p_l2_2102

theorem smallest_p 
  (p q : ℕ) 
  (h1 : (5 : ℚ) / 8 < p / (q : ℚ) ∧ p / (q : ℚ) < 7 / 8)
  (h2 : p + q = 2005) : p = 772 :=
sorry

end smallest_p_l2_2102


namespace non_student_ticket_price_l2_2465

theorem non_student_ticket_price (x : ℕ) : 
  (∃ (n_student_ticket_price ticket_count total_revenue student_tickets : ℕ),
    n_student_ticket_price = 9 ∧
    ticket_count = 2000 ∧
    total_revenue = 20960 ∧
    student_tickets = 520 ∧
    (student_tickets * n_student_ticket_price + (ticket_count - student_tickets) * x = total_revenue)) -> 
  x = 11 := 
by
  -- placeholder for proof
  sorry

end non_student_ticket_price_l2_2465


namespace find_circle_center_l2_2371

-- Definition of the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 - 6*x + y^2 + 10*y - 7 = 0

-- The main statement to prove
theorem find_circle_center :
  (∃ center : ℝ × ℝ, center = (3, -5) ∧ ∀ x y : ℝ, circle_eq x y ↔ (x - 3)^2 + (y + 5)^2 = 41) :=
sorry

end find_circle_center_l2_2371


namespace intersection_M_N_is_valid_l2_2841

-- Define the conditions given in the problem
def M := {x : ℝ |  3 / 4 < x ∧ x ≤ 1}
def N := {y : ℝ | 0 ≤ y}

-- State the theorem that needs to be proved
theorem intersection_M_N_is_valid : M ∩ N = {x : ℝ | 3 / 4 < x ∧ x ≤ 1} :=
by 
  sorry

end intersection_M_N_is_valid_l2_2841


namespace max_value_fraction_l2_2606

theorem max_value_fraction (x : ℝ) : 
  ∃ (n : ℤ), n = 3 ∧ 
  ∃ (y : ℝ), y = (4 * x^2 + 8 * x + 19) / (4 * x^2 + 8 * x + 9) ∧ 
  y ≤ n := 
sorry

end max_value_fraction_l2_2606


namespace simplest_fraction_l2_2925

theorem simplest_fraction (x y : ℝ) (h1 : 2 * x ≠ 0) (h2 : x + y ≠ 0) :
  let A := (2 * x) / (4 * x^2)
  let B := (x^2 + y^2) / (x + y)
  let C := (x^2 + 2 * x + 1) / (x + 1)
  let D := (x^2 - 4) / (x + 2)
  B = (x^2 + y^2) / (x + y) ∧
  A ≠ (2 * x) / (4 * x^2) ∧
  C ≠ (x^2 + 2 * x + 1) / (x + 1) ∧
  D ≠ (x^2 - 4) / (x + 2) := sorry

end simplest_fraction_l2_2925


namespace robins_hair_length_l2_2900

-- Conditions:
-- Robin cut off 4 inches of his hair.
-- After cutting, his hair is now 13 inches long.
-- Question: How long was Robin's hair before he cut it? Answer: 17 inches

theorem robins_hair_length (current_length : ℕ) (cut_length : ℕ) (initial_length : ℕ) 
  (h_cut_length : cut_length = 4) 
  (h_current_length : current_length = 13) 
  (h_initial : initial_length = current_length + cut_length) :
  initial_length = 17 :=
sorry

end robins_hair_length_l2_2900


namespace factor_poly_l2_2875

theorem factor_poly (a b : ℤ) (h : 3*(y^2) - y - 24 = (3*y + a)*(y + b)) : a - b = 11 :=
sorry

end factor_poly_l2_2875


namespace integral_value_l2_2202

theorem integral_value (a : ℝ) (h : a = 2) : ∫ x in a..2*Real.exp 1, 1/x = 1 := by
  sorry

end integral_value_l2_2202


namespace physics_students_l2_2587

variable (B : Nat) (G : Nat) (Biology : Nat) (Physics : Nat)

axiom h1 : B = 25
axiom h2 : G = 3 * B
axiom h3 : Biology = B + G
axiom h4 : Physics = 2 * Biology

theorem physics_students : Physics = 200 :=
by
  sorry

end physics_students_l2_2587


namespace cat_finishes_food_on_sunday_l2_2471

-- Define the constants and parameters
def daily_morning_consumption : ℚ := 2 / 5
def daily_evening_consumption : ℚ := 1 / 5
def total_food : ℕ := 8
def days_in_week : ℕ := 7

-- Define the total daily consumption
def total_daily_consumption : ℚ := daily_morning_consumption + daily_evening_consumption

-- Define the sum of consumptions over each day until the day when all food is consumed
def food_remaining_after_days (days : ℕ) : ℚ := total_food - days * total_daily_consumption

-- Proposition that the food is finished on Sunday
theorem cat_finishes_food_on_sunday :
  ∃ days : ℕ, (food_remaining_after_days days ≤ 0) ∧ days ≡ 7 [MOD days_in_week] :=
sorry

end cat_finishes_food_on_sunday_l2_2471


namespace maximum_value_of_x_plus_2y_l2_2693

theorem maximum_value_of_x_plus_2y (x y : ℝ) (h : x^2 - 2 * x + 4 * y = 5) : ∃ m, m = x + 2 * y ∧ m ≤ 9/2 := by
  sorry

end maximum_value_of_x_plus_2y_l2_2693


namespace systematic_sampling_l2_2719

-- Define the conditions
def total_products : ℕ := 100
def selected_products (n : ℕ) : ℕ := 3 + 10 * n
def is_systematic (f : ℕ → ℕ) : Prop :=
  ∃ k b, ∀ n, f n = b + k * n

-- Theorem to prove that the selection method is systematic sampling
theorem systematic_sampling : is_systematic selected_products :=
  sorry

end systematic_sampling_l2_2719


namespace parabola_directrix_standard_eq_l2_2590

theorem parabola_directrix_standard_eq (p : ℝ) (h : p = 2) :
  ∀ y x : ℝ, (x = -1) → (y^2 = 4 * x) :=
by
  sorry

end parabola_directrix_standard_eq_l2_2590


namespace initial_oranges_l2_2078

theorem initial_oranges (left_oranges taken_oranges : ℕ) (h1 : left_oranges = 25) (h2 : taken_oranges = 35) : 
  left_oranges + taken_oranges = 60 := 
by 
  sorry

end initial_oranges_l2_2078


namespace find_angle_B_l2_2766

theorem find_angle_B (a b c A B C : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = π / 5)
  (h3 : 0 < A) (h4 : A < π)
  (h5 : 0 < B) (h6 : B < π)
  (h7 : 0 < C) (h8 : C < π)
  (h_triangle : A + B + C = π) :
  B = 3 * π / 10 :=
sorry

end find_angle_B_l2_2766


namespace length_of_plot_l2_2043

theorem length_of_plot (breadth length : ℕ) 
                       (h1 : length = breadth + 26)
                       (fencing_cost total_cost : ℝ)
                       (h2 : fencing_cost = 26.50)
                       (h3 : total_cost = 5300)
                       (perimeter : ℝ) 
                       (h4 : perimeter = 2 * (breadth + length)) 
                       (h5 : total_cost = perimeter * fencing_cost) :
                       length = 63 :=
by
  sorry

end length_of_plot_l2_2043


namespace bret_total_spend_l2_2369

/-- Bret and his team are working late along with another team of 4 co-workers.
He decides to order dinner for everyone. -/

def team_A : ℕ := 4 -- Bret’s team
def team_B : ℕ := 4 -- Other team

def main_meal_cost : ℕ := 12
def team_A_appetizers_cost : ℕ := 2 * 6  -- Two appetizers at $6 each
def team_B_appetizers_cost : ℕ := 3 * 8  -- Three appetizers at $8 each
def sharing_plates_cost : ℕ := 4 * 10    -- Four sharing plates at $10 each

def tip_percentage : ℝ := 0.20           -- Tip is 20%
def rush_order_fee : ℕ := 5              -- Rush order fee is $5
def sales_tax : ℝ := 0.07                -- Local sales tax is 7%

def total_cost_without_tip_and_tax : ℕ :=
  team_A * main_meal_cost + team_B * main_meal_cost + team_A_appetizers_cost +
  team_B_appetizers_cost + sharing_plates_cost

def total_cost_with_tip : ℝ :=
  total_cost_without_tip_and_tax + 
  (tip_percentage * total_cost_without_tip_and_tax)

def total_cost_before_tax : ℝ :=
  total_cost_with_tip + rush_order_fee

def final_total_cost : ℝ :=
  total_cost_before_tax + (sales_tax * total_cost_with_tip)


theorem bret_total_spend : final_total_cost = 225.85 := by
  sorry

end bret_total_spend_l2_2369


namespace selling_price_of_cricket_bat_l2_2387

variable (profit : ℝ) (profit_percentage : ℝ)
variable (selling_price : ℝ)

theorem selling_price_of_cricket_bat 
  (h1 : profit = 215)
  (h2 : profit_percentage = 33.85826771653544) : 
  selling_price = 849.70 :=
sorry

end selling_price_of_cricket_bat_l2_2387


namespace heidi_more_nail_polishes_l2_2773

theorem heidi_more_nail_polishes :
  ∀ (k h r : ℕ), 
    k = 12 ->
    r = k - 4 ->
    h + r = 25 ->
    h - k = 5 :=
by
  intros k h r hk hr hr_sum
  sorry

end heidi_more_nail_polishes_l2_2773


namespace value_of_y_at_3_l2_2674

-- Define the function
def f (x : ℕ) : ℕ := 2 * x^2 + 1

-- Prove that when x = 3, y = 19
theorem value_of_y_at_3 : f 3 = 19 :=
by
  -- Provide the definition and conditions
  let x := 3
  let y := f x
  have h : y = 2 * x^2 + 1 := rfl
  -- State the actual proof could go here
  sorry

end value_of_y_at_3_l2_2674


namespace two_pow_p_add_three_pow_p_eq_a_pow_n_imp_n_eq_one_l2_2294

theorem two_pow_p_add_three_pow_p_eq_a_pow_n_imp_n_eq_one
  (p a n : ℕ) (hp : Nat.Prime p) (ha : 0 < a) (hn : 0 < n) 
  (h : 2 ^ p + 3 ^ p = a ^ n) : n = 1 :=
sorry

end two_pow_p_add_three_pow_p_eq_a_pow_n_imp_n_eq_one_l2_2294


namespace sum_of_distinct_FGHJ_values_l2_2528

theorem sum_of_distinct_FGHJ_values (A B C D E F G H I J K : ℕ)
  (h1: 0 ≤ A ∧ A ≤ 9)
  (h2: 0 ≤ B ∧ B ≤ 9)
  (h3: 0 ≤ C ∧ C ≤ 9)
  (h4: 0 ≤ D ∧ D ≤ 9)
  (h5: 0 ≤ E ∧ E ≤ 9)
  (h6: 0 ≤ F ∧ F ≤ 9)
  (h7: 0 ≤ G ∧ G ≤ 9)
  (h8: 0 ≤ H ∧ H ≤ 9)
  (h9: 0 ≤ I ∧ I ≤ 9)
  (h10: 0 ≤ J ∧ J ≤ 9)
  (h11: 0 ≤ K ∧ K ≤ 9)
  (h_divisibility_16: ∃ x, GHJK = x ∧ x % 16 = 0)
  (h_divisibility_9: (1 + B + C + D + E + F + G + H + I + J + K) % 9 = 0) :
  (F * G * H * J = 12 ∨ F * G * H * J = 120 ∨ F * G * H * J = 448) →
  (12 + 120 + 448 = 580) := 
by sorry

end sum_of_distinct_FGHJ_values_l2_2528


namespace distance_to_other_focus_of_ellipse_l2_2556

noncomputable def ellipse_param (a b : ℝ) (x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1

def is_focus_distance (a distF1 distF2 : ℝ) : Prop :=
  ∀ P₁ P₂ : ℝ, distF1 + distF2 = 2 * a

theorem distance_to_other_focus_of_ellipse (x y : ℝ) (distF1 : ℝ) :
  ellipse_param 4 5 x y ∧ distF1 = 6 → is_focus_distance 5 distF1 4 :=
by
  simp [ellipse_param, is_focus_distance]
  sorry

end distance_to_other_focus_of_ellipse_l2_2556


namespace problem1_problem2_l2_2522

-- Problem (1)
theorem problem1 (a b : ℝ) (h : 2 * a^2 + 3 * b = 6) : a^2 + (3 / 2) * b - 5 = -2 := 
sorry

-- Problem (2)
theorem problem2 (x : ℝ) (h : 14 * x + 5 - 21 * x^2 = -2) : 6 * x^2 - 4 * x + 5 = 7 := 
sorry

end problem1_problem2_l2_2522
