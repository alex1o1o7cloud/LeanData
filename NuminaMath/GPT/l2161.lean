import Mathlib

namespace airplane_average_speed_l2161_216106

-- Define the conditions
def miles_to_kilometers (miles : ℕ) : ℝ :=
  miles * 1.60934

def distance_miles : ℕ := 1584
def time_hours : ℕ := 24

-- Define the problem to prove
theorem airplane_average_speed : 
  (miles_to_kilometers distance_miles) / (time_hours : ℝ) = 106.24 :=
by
  sorry

end airplane_average_speed_l2161_216106


namespace prove_product_reduced_difference_l2161_216135

-- We are given two numbers x and y such that:
variable (x y : ℚ)
-- 1. The sum of the numbers is 6
axiom sum_eq_six : x + y = 6
-- 2. The quotient of the larger number by the smaller number is 6
axiom quotient_eq_six : x / y = 6

-- We need to prove that the product of these two numbers reduced by their difference is 6/49
theorem prove_product_reduced_difference (x y : ℚ) 
  (sum_eq_six : x + y = 6) (quotient_eq_six : x / y = 6) : 
  (x * y) - (x - y) = 6 / 49 := 
by
  sorry

end prove_product_reduced_difference_l2161_216135


namespace difference_of_fractions_l2161_216190

theorem difference_of_fractions (a : ℝ) (b : ℝ) (h1 : a = 700) (h2 : b = 7) : a - b = 693 :=
by
  rw [h1, h2]
  norm_num

end difference_of_fractions_l2161_216190


namespace totalCandlesInHouse_l2161_216127

-- Definitions for the problem's conditions
def bedroomCandles : ℕ := 20
def livingRoomCandles : ℕ := bedroomCandles / 2
def donovanCandles : ℕ := 20

-- Problem to prove
theorem totalCandlesInHouse : bedroomCandles + livingRoomCandles + donovanCandles = 50 := by
  sorry

end totalCandlesInHouse_l2161_216127


namespace planes_touch_three_spheres_count_l2161_216109

-- Declare the conditions as definitions
def square_side_length : ℝ := 10
def radii : Fin 4 → ℝ
| 0 => 1
| 1 => 2
| 2 => 4
| 3 => 3

-- The proof problem statement
theorem planes_touch_three_spheres_count :
    ∃ (planes_that_touch_three_spheres : ℕ) (planes_that_intersect_fourth_sphere : ℕ),
    planes_that_touch_three_spheres = 26 ∧ planes_that_intersect_fourth_sphere = 8 := 
by
  -- sorry skips the proof
  sorry

end planes_touch_three_spheres_count_l2161_216109


namespace total_cost_of_phone_l2161_216123

theorem total_cost_of_phone (cost_per_phone : ℕ) (monthly_cost : ℕ) (months : ℕ) (phone_count : ℕ) :
  cost_per_phone = 2 → monthly_cost = 7 → months = 4 → phone_count = 1 →
  (cost_per_phone * phone_count + monthly_cost * months) = 30 :=
by
  intros h1 h2 h3 h4
  sorry

end total_cost_of_phone_l2161_216123


namespace car_average_speed_l2161_216154

noncomputable def average_speed (D : ℝ) : ℝ :=
  let t1 := (D / 3) / 80
  let t2 := (D / 3) / 24
  let t3 := (D / 3) / 30
  let total_time := t1 + t2 + t3
  D / total_time

theorem car_average_speed :
  average_speed D = 34.2857 := by
  sorry

end car_average_speed_l2161_216154


namespace min_value_f_l2161_216133

noncomputable def f (x : ℝ) : ℝ := x^3 + 9 * x + 81 / x^4

theorem min_value_f : ∃ x > 0, f x = 21 ∧ ∀ y > 0, f y ≥ 21 := by
  sorry

end min_value_f_l2161_216133


namespace circumradius_of_triangle_l2161_216108

theorem circumradius_of_triangle (a b c : ℕ) (h₁ : a = 8) (h₂ : b = 6) (h₃ : c = 10) 
  (h₄ : a^2 + b^2 = c^2) : 
  (c : ℝ) / 2 = 5 := 
by {
  -- proof goes here
  sorry
}

end circumradius_of_triangle_l2161_216108


namespace part_1_part_2_part_3_l2161_216141

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := (2 * Real.exp x) / (Real.exp x + 1) + k

theorem part_1 (k : ℝ) :
  (∀ x, f x k = -f (-x) k) → k = -1 :=
sorry

theorem part_2 (m : ℝ) :
  (∀ x > 0, (2 * Real.exp x - 1) / (Real.exp x + 1) ≤ m * (Real.exp x - 1) / (Real.exp x + 1)) → 2 ≤ m :=
sorry

noncomputable def g (x : ℝ) : ℝ := (f x (-1) + 1) / (1 - f x (-1))

theorem part_3 (n : ℝ) :
  (∀ a b c : ℝ, 0 < a ∧ a ≤ n → 0 < b ∧ b ≤ n → 0 < c ∧ c ≤ n → (a + b > c ∧ b + c > a ∧ c + a > b) →
   (g a + g b > g c ∧ g b + g c > g a ∧ g c + g a > g b)) → n = 2 * Real.log 2 :=
sorry

end part_1_part_2_part_3_l2161_216141


namespace solution_set_of_xf_gt_0_l2161_216160

noncomputable def f : ℝ → ℝ := sorry

axiom f_even : ∀ x : ℝ, f x = f (-x)
axiom f_ineq : ∀ x : ℝ, x > 0 → f x < x * (deriv f x)
axiom f_at_one : f 1 = 0

theorem solution_set_of_xf_gt_0 : {x : ℝ | x * f x > 0} = {x : ℝ | -1 < x ∧ x < 0} ∪ {x : ℝ | x > 1} :=
by
  sorry

end solution_set_of_xf_gt_0_l2161_216160


namespace probability_athlete_A_selected_number_of_males_selected_number_of_females_selected_l2161_216139

noncomputable def total_members := 42
noncomputable def boys := 28
noncomputable def girls := 14
noncomputable def selected := 6

theorem probability_athlete_A_selected :
  (selected : ℚ) / total_members = 1 / 7 :=
by sorry

theorem number_of_males_selected :
  (selected * (boys : ℚ)) / total_members = 4 :=
by sorry

theorem number_of_females_selected :
  (selected * (girls : ℚ)) / total_members = 2 :=
by sorry

end probability_athlete_A_selected_number_of_males_selected_number_of_females_selected_l2161_216139


namespace complete_square_transformation_l2161_216187

theorem complete_square_transformation : 
  ∀ (x : ℝ), (x^2 - 8 * x + 9 = 0) → ((x - 4)^2 = 7) :=
by
  intros x h
  sorry

end complete_square_transformation_l2161_216187


namespace find_r4_l2161_216189

variable (r : ℝ)

theorem find_r4 (h : (r + 1/r)^2 = 5) :
  r^4 + 1/r^4 = 7 :=
sorry

end find_r4_l2161_216189


namespace more_spent_on_keychains_bracelets_than_tshirts_l2161_216199

-- Define the conditions as variables
variable (spent_keychains_bracelets spent_total_spent : ℝ)
variable (spent_keychains_bracelets_eq : spent_keychains_bracelets = 347.00)
variable (spent_total_spent_eq : spent_total_spent = 548.00)

-- Using these conditions, define the problem to prove the desired result
theorem more_spent_on_keychains_bracelets_than_tshirts :
  spent_keychains_bracelets - (spent_total_spent - spent_keychains_bracelets) = 146.00 :=
by
  rw [spent_keychains_bracelets_eq, spent_total_spent_eq]
  sorry

end more_spent_on_keychains_bracelets_than_tshirts_l2161_216199


namespace num_ways_to_assign_grades_l2161_216161

-- Define the number of students
def num_students : ℕ := 12

-- Define the number of grades available to each student
def num_grades : ℕ := 4

-- The theorem stating that the total number of ways to assign grades is 4^12
theorem num_ways_to_assign_grades : num_grades ^ num_students = 16777216 := by
  sorry

end num_ways_to_assign_grades_l2161_216161


namespace geometric_diff_l2161_216196

-- Definitions based on conditions
def is_geometric (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 ∧ (d2 * d2 = d1 * d3)

-- Problem statement
theorem geometric_diff :
  let largest_geometric := 964
  let smallest_geometric := 124
  is_geometric largest_geometric ∧ is_geometric smallest_geometric ∧
  (largest_geometric - smallest_geometric = 840) :=
by
  sorry

end geometric_diff_l2161_216196


namespace diplomats_neither_french_nor_russian_l2161_216169

variable (total_diplomats : ℕ)
variable (speak_french : ℕ)
variable (not_speak_russian : ℕ)
variable (speak_both : ℕ)

theorem diplomats_neither_french_nor_russian {total_diplomats speak_french not_speak_russian speak_both : ℕ} 
  (h1 : total_diplomats = 100)
  (h2 : speak_french = 22)
  (h3 : not_speak_russian = 32)
  (h4 : speak_both = 10) :
  ((total_diplomats - (speak_french + (total_diplomats - not_speak_russian) - speak_both)) * 100) / total_diplomats = 20 := 
by
  sorry

end diplomats_neither_french_nor_russian_l2161_216169


namespace slower_speed_is_correct_l2161_216112

/-- 
A person walks at 14 km/hr instead of a slower speed, 
and as a result, he would have walked 20 km more. 
The actual distance travelled by him is 50 km. 
What is the slower speed he usually walks at?
-/
theorem slower_speed_is_correct :
    ∃ x : ℝ, (14 * (50 / 14) - (x * (30 / x))) = 20 ∧ x = 8.4 :=
by
  sorry

end slower_speed_is_correct_l2161_216112


namespace a_16_value_l2161_216117

-- Define the recurrence relation
def seq (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  match n with
  | 0       => 2
  | (n + 1) => (1 + a n) / (1 - a n)

-- State the theorem
theorem a_16_value :
  seq (a : ℕ → ℚ) 16 = -1/3 := 
sorry

end a_16_value_l2161_216117


namespace joy_valid_rod_count_l2161_216125

theorem joy_valid_rod_count : 
  let l := [4, 12, 21]
  let qs := [1, 2, 3, 5, 13, 20, 22, 40].filter (fun x => x != 4 ∧ x != 12 ∧ x != 21)
  (∀ d ∈ qs, 4 + 12 + 21 > d ∧ 4 + 12 + d > 21 ∧ 4 + 21 + d > 12 ∧ 12 + 21 + d > 4) → 
  ∃ n, n = 28 :=
by sorry

end joy_valid_rod_count_l2161_216125


namespace absolute_value_inequality_solution_l2161_216176

theorem absolute_value_inequality_solution (x : ℝ) : |2*x - 1| < 3 ↔ -1 < x ∧ x < 2 := 
sorry

end absolute_value_inequality_solution_l2161_216176


namespace intersection_complement_l2161_216147

open Set

variable (U : Type) [TopologicalSpace U]

def A : Set ℝ := { x | x ≥ 0 }

def B : Set ℝ := { y | y ≤ 0 }

theorem intersection_complement (U : Type) [TopologicalSpace U] : 
  A ∩ (compl B) = { x | x > 0 } :=
by
  sorry

end intersection_complement_l2161_216147


namespace min_value_function_l2161_216134

theorem min_value_function (x : ℝ) (h : x > 0) : 
  ∃ y, y = (x^2 + x + 25) / x ∧ y ≥ 11 :=
sorry

end min_value_function_l2161_216134


namespace solution_set_of_abs_inequality_l2161_216107

theorem solution_set_of_abs_inequality :
  { x : ℝ | |x^2 - 2| < 2 } = { x : ℝ | -2 < x ∧ x < 0 ∨ 0 < x ∧ x < 2 } :=
sorry

end solution_set_of_abs_inequality_l2161_216107


namespace smallest_gcd_six_l2161_216167

theorem smallest_gcd_six (x : ℕ) (hx1 : 70 ≤ x) (hx2 : x ≤ 90) (hx3 : Nat.gcd 24 x = 6) : x = 78 :=
by
  sorry

end smallest_gcd_six_l2161_216167


namespace range_of_k_l2161_216191

-- Given conditions
variables {k : ℝ} (h : ∃ (x y : ℝ), x^2 + k * y^2 = 2)

-- Theorem statement
theorem range_of_k : 0 < k ∧ k < 1 :=
by
  sorry

end range_of_k_l2161_216191


namespace evaluate_expression_l2161_216144

theorem evaluate_expression (m n : ℝ) (h : m - n = 2) :
  (2 * m^2 - 4 * m * n + 2 * n^2 - 1) = 7 := by
  sorry

end evaluate_expression_l2161_216144


namespace crayons_lost_or_given_away_l2161_216192

theorem crayons_lost_or_given_away (P E L : ℕ) (h1 : P = 479) (h2 : E = 134) (h3 : L = P - E) : L = 345 :=
by
  rw [h1, h2] at h3
  exact h3

end crayons_lost_or_given_away_l2161_216192


namespace sergio_has_6_more_correct_answers_l2161_216115

-- Define conditions
def total_questions : ℕ := 50
def incorrect_answers_sylvia : ℕ := total_questions / 5
def incorrect_answers_sergio : ℕ := 4

-- Calculate correct answers
def correct_answers_sylvia : ℕ := total_questions - incorrect_answers_sylvia
def correct_answers_sergio : ℕ := total_questions - incorrect_answers_sergio

-- The proof problem
theorem sergio_has_6_more_correct_answers :
  correct_answers_sergio - correct_answers_sylvia = 6 :=
by
  sorry

end sergio_has_6_more_correct_answers_l2161_216115


namespace minimum_group_members_round_table_l2161_216162

theorem minimum_group_members_round_table (n : ℕ) (h1 : ∀ (a : ℕ),  a < n) : 5 ≤ n :=
by
  sorry

end minimum_group_members_round_table_l2161_216162


namespace ladder_base_length_l2161_216163

theorem ladder_base_length {a b c : ℕ} (h1 : c = 13) (h2 : b = 12) (h3 : a^2 + b^2 = c^2) :
  a = 5 := 
by 
  sorry

end ladder_base_length_l2161_216163


namespace money_left_is_41_l2161_216155

-- Define the amounts saved by Tanner in each month
def savings_september : ℕ := 17
def savings_october : ℕ := 48
def savings_november : ℕ := 25

-- Define the amount spent by Tanner on the video game
def spent_video_game : ℕ := 49

-- Total savings after the three months
def total_savings : ℕ := savings_september + savings_october + savings_november

-- Calculate the money left after spending on the video game
def money_left : ℕ := total_savings - spent_video_game

-- The theorem we need to prove
theorem money_left_is_41 : money_left = 41 := by
  sorry

end money_left_is_41_l2161_216155


namespace trapezoid_sides_l2161_216111

theorem trapezoid_sides (r kl: ℝ) (h1 : r = 5) (h2 : kl = 8) :
  ∃ (ab cd bc_ad : ℝ), ab = 5 ∧ cd = 20 ∧ bc_ad = 12.5 :=
by
  sorry

end trapezoid_sides_l2161_216111


namespace inequality_I_inequality_II_inequality_III_l2161_216101

variable {a b c x y z : ℝ}

-- Assume the conditions
def conditions (a b c x y z : ℝ) : Prop :=
  x^2 < a ∧ y^2 < b ∧ z^2 < c

-- Prove the first inequality
theorem inequality_I (h : conditions a b c x y z) : x^2 * y^2 + y^2 * z^2 + z^2 * x^2 < a * b + b * c + c * a :=
sorry

-- Prove the second inequality
theorem inequality_II (h : conditions a b c x y z) : x^4 + y^4 + z^4 < a^2 + b^2 + c^2 :=
sorry

-- Prove the third inequality
theorem inequality_III (h : conditions a b c x y z) : x^2 * y^2 * z^2 < a * b * c :=
sorry

end inequality_I_inequality_II_inequality_III_l2161_216101


namespace area_in_sq_yds_l2161_216173

-- Definitions based on conditions
def side_length_ft : ℕ := 9
def sq_ft_per_sq_yd : ℕ := 9

-- Statement to prove
theorem area_in_sq_yds : (side_length_ft * side_length_ft) / sq_ft_per_sq_yd = 9 :=
by
  sorry

end area_in_sq_yds_l2161_216173


namespace perimeter_hypotenuse_ratios_l2161_216174

variable {x y : Real}
variable (h_pos_x : x > 0) (h_pos_y : y > 0)

theorem perimeter_hypotenuse_ratios
    (h_sides : (3 * x + 3 * y = (3 * x + 3 * y)) ∨ 
               (4 * x = (4 * x)) ∨
               (4 * y = (4 * y)))
    : 
    (∃ p : Real, p = 7 * (x + y) / (3 * (x + y)) ∨
                 p = 32 * y / (100 / 7 * y) ∨
                 p = 224 / 25 * y / 4 * y ∨ 
                 p = 7 / 3 ∨ 
                 p = 56 / 25) := by sorry

end perimeter_hypotenuse_ratios_l2161_216174


namespace chemist_solution_l2161_216114

theorem chemist_solution (x : ℝ) (h1 : ∃ x, 0 < x) 
  (h2 : x + 1 > 1) : 0.60 * x = 0.10 * (x + 1) → x = 0.2 := by
  sorry

end chemist_solution_l2161_216114


namespace average_glasses_is_15_l2161_216152

variable (S L : ℕ)

-- Conditions:
def box1 := 12 -- One box contains 12 glasses
def box2 := 16 -- Another box contains 16 glasses
def total_glasses := 480 -- Total number of glasses
def diff_L_S := 16 -- There are 16 more larger boxes

-- Equations derived from conditions:
def eq1 : Prop := (12 * S + 16 * L = total_glasses)
def eq2 : Prop := (L = S + diff_L_S)

-- We need to prove that the average number of glasses per box is 15:
def avg_glasses_per_box := total_glasses / (S + L)

-- The statement we need to prove:
theorem average_glasses_is_15 :
  (12 * S + 16 * L = total_glasses) ∧ (L = S + diff_L_S) → avg_glasses_per_box = 15 :=
by
  sorry

end average_glasses_is_15_l2161_216152


namespace nina_money_l2161_216157

theorem nina_money (W M : ℕ) (h1 : 6 * W = M) (h2 : 8 * (W - 2) = M) : M = 48 :=
by
  sorry

end nina_money_l2161_216157


namespace total_first_year_students_400_l2161_216140

theorem total_first_year_students_400 (N : ℕ) (A B C : ℕ) 
  (h1 : A = 80) 
  (h2 : B = 100) 
  (h3 : C = 20) 
  (h4 : A * B = C * N) : 
  N = 400 :=
sorry

end total_first_year_students_400_l2161_216140


namespace work_alone_days_l2161_216194

theorem work_alone_days (d : ℝ) (p q : ℝ) (h1 : q = 10) (h2 : 2 * (1/d + 1/q) = 0.3) : d = 20 :=
by
  sorry

end work_alone_days_l2161_216194


namespace wolves_total_games_l2161_216195

theorem wolves_total_games
  (x y : ℕ) -- Before district play, the Wolves had won x games out of y games.
  (hx : x = 40 * y / 100) -- The Wolves had won 40% of their basketball games before district play.
  (hx' : 5 * x = 2 * y)
  (hy : 60 * (y + 10) / 100 = x + 9) -- They finished the season having won 60% of their total games.
  : y + 10 = 25 := by
  sorry

end wolves_total_games_l2161_216195


namespace sample_size_correct_l2161_216132

-- Definitions derived from conditions in a)
def total_employees : ℕ := 120
def male_employees : ℕ := 90
def sampled_male_employees : ℕ := 18

-- Theorem stating the mathematically equivalent proof problem
theorem sample_size_correct : 
  ∃ (sample_size : ℕ), sample_size = (total_employees * (sampled_male_employees / male_employees)) :=
sorry

end sample_size_correct_l2161_216132


namespace find_y_l2161_216113

-- Definitions of the given conditions
def angle_ABC_is_straight_line := true  -- This is to ensure the angle is a straight line.
def angle_ABD_is_exterior_of_triangle_BCD := true -- This is to ensure ABD is an exterior angle.
def angle_ABD : ℝ := 118
def angle_BCD : ℝ := 82

-- Theorem to prove y = 36 given the conditions
theorem find_y (A B C D : Type) (y : ℝ) 
    (h1 : angle_ABC_is_straight_line)
    (h2 : angle_ABD_is_exterior_of_triangle_BCD)
    (h3 : angle_ABD = 118)
    (h4 : angle_BCD = 82) : 
            y = 36 :=
  by
  sorry

end find_y_l2161_216113


namespace lorenzo_cans_l2161_216185

theorem lorenzo_cans (c : ℕ) (tacks_per_can : ℕ) (total_tacks : ℕ) (boards_tested : ℕ) (remaining_tacks : ℕ) :
  boards_tested = 120 →
  remaining_tacks = 30 →
  total_tacks = 450 →
  tacks_per_can = (boards_tested + remaining_tacks) →
  c * tacks_per_can = total_tacks →
  c = 3 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end lorenzo_cans_l2161_216185


namespace sequence_general_formula_l2161_216150

theorem sequence_general_formula
  (S : ℕ → ℝ)
  (a : ℕ → ℝ)
  (hSn : ∀ n, S n = (3 / 2) * (a n) - 3) :
  ∀ n, a n = 3 * (2 : ℝ) ^ n :=
by sorry

end sequence_general_formula_l2161_216150


namespace construct_3x3x3_cube_l2161_216116

theorem construct_3x3x3_cube :
  ∃ (cubes_1x2x2 : Finset (Set (Fin 3 × Fin 3 × Fin 3))),
  ∃ (cubes_1x1x1 : Finset (Fin 3 × Fin 3 × Fin 3)),
  cubes_1x2x2.card = 6 ∧ 
  cubes_1x1x1.card = 3 ∧ 
  (∀ c ∈ cubes_1x2x2, ∃ a b : Fin 3, ∀ x, x = (a, b, 0) ∨ x = (a, b, 1) ∨ x = (a, b, 2)) ∧
  (∀ c ∈ cubes_1x1x1, ∃ a b c : Fin 3, ∀ x, x = (a, b, c)) :=
sorry

end construct_3x3x3_cube_l2161_216116


namespace min_ab_value_l2161_216100

theorem min_ab_value 
  (a b : ℝ) 
  (hab_pos : a * b > 0)
  (collinear_condition : 2 * a + 2 * b + a * b = 0) :
  a * b ≥ 16 := 
sorry

end min_ab_value_l2161_216100


namespace final_score_proof_l2161_216182

def final_score (bullseye_points : ℕ) (miss_points : ℕ) (half_bullseye_points : ℕ) : ℕ :=
  bullseye_points + miss_points + half_bullseye_points

theorem final_score_proof : final_score 50 0 25 = 75 :=
by
  -- Considering the given conditions
  -- bullseye_points = 50
  -- miss_points = 0
  -- half_bullseye_points = half of bullseye_points = 25
  -- Summing them up: 50 + 0 + 25 = 75
  sorry

end final_score_proof_l2161_216182


namespace division_remainder_false_l2161_216171

theorem division_remainder_false :
  ¬(1700 / 500 = 17 / 5 ∧ (1700 % 500 = 3 ∧ 17 % 5 = 2)) := by
  sorry

end division_remainder_false_l2161_216171


namespace set_equality_l2161_216184

-- Define the universe U
def U := ℝ

-- Define the set M
def M := {x : ℝ | (x + 1) * (x - 2) ≤ 0}

-- Define the set N
def N := {x : ℝ | x > 1}

-- Define the set we want to prove is equal to the intersection of M and N
def target_set := {x : ℝ | 1 < x ∧ x ≤ 2}

theorem set_equality : target_set = M ∩ N := 
by sorry

end set_equality_l2161_216184


namespace smallest_n_in_range_l2161_216172

theorem smallest_n_in_range (n : ℤ) (h1 : 4 ≤ n ∧ n ≤ 12) (h2 : n ≡ 2 [ZMOD 9]) : n = 11 :=
sorry

end smallest_n_in_range_l2161_216172


namespace solve_for_m_l2161_216119

namespace ProofProblem

def f (x m : ℝ) : ℝ := x^2 - 3*x + m
def g (x m : ℝ) : ℝ := x^2 - 3*x + 5*m

theorem solve_for_m (m : ℝ) : 3 * f 3 m = g 3 m → m = 0 := by
  sorry

end ProofProblem

end solve_for_m_l2161_216119


namespace sarahs_trip_length_l2161_216121

noncomputable def sarahsTrip (x : ℝ) : Prop :=
  x / 4 + 15 + x / 3 = x

theorem sarahs_trip_length : ∃ x : ℝ, sarahsTrip x ∧ x = 36 := by
  -- There should be a proof here, but it's omitted as per the task instructions
  sorry

end sarahs_trip_length_l2161_216121


namespace total_cost_mulch_l2161_216149

-- Define the conditions
def tons_to_pounds (tons : ℕ) : ℕ := tons * 2000

def price_per_pound : ℝ := 2.5

-- Define the statement to prove
theorem total_cost_mulch (mulch_in_tons : ℕ) (h₁ : mulch_in_tons = 3) : 
  tons_to_pounds mulch_in_tons * price_per_pound = 15000 :=
by
  -- The proof would normally go here.
  sorry

end total_cost_mulch_l2161_216149


namespace equivalent_problem_l2161_216122

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then x^2 else sorry

theorem equivalent_problem 
  (f_odd : ∀ x : ℝ, f (-x) = -f x)
  (f_periodic : ∀ x : ℝ, f (x + 2) = f x)
  (f_interval : ∀ x : ℝ, 0 < x ∧ x < 1 → f x = x^2)
  : f (-3/2) + f 1 = 3/4 :=
sorry

end equivalent_problem_l2161_216122


namespace tan_alpha_eq_l2161_216129

theorem tan_alpha_eq : ∀ (α : ℝ),
  (Real.tan (α - (5 * Real.pi / 4)) = 1 / 5) →
  Real.tan α = 3 / 2 :=
by
  intro α h
  sorry

end tan_alpha_eq_l2161_216129


namespace pages_read_on_Sunday_l2161_216166

def total_pages : ℕ := 93
def pages_read_on_Saturday : ℕ := 30
def pages_remaining_after_Sunday : ℕ := 43

theorem pages_read_on_Sunday : total_pages - pages_read_on_Saturday - pages_remaining_after_Sunday = 20 := by
  sorry

end pages_read_on_Sunday_l2161_216166


namespace clairaut_equation_solution_l2161_216102

open Real

noncomputable def clairaut_solution (f : ℝ → ℝ) (C : ℝ) : Prop :=
  (∀ x, f x = C * x + 1/(2 * C)) ∨ (∀ x, (f x)^2 = 2 * x)

theorem clairaut_equation_solution (y : ℝ → ℝ) :
  (∀ x, y x = x * (deriv y x) + 1/(2 * (deriv y x))) →
  ∃ C, clairaut_solution y C :=
sorry

end clairaut_equation_solution_l2161_216102


namespace grain_to_rice_system_l2161_216142

variable (x y : ℕ)

/-- Conversion rate of grain to rice is 3/5. -/
def conversion_rate : ℚ := 3 / 5

/-- Total bucket capacity is 10 dou. -/
def total_capacity : ℕ := 10

/-- Rice obtained after threshing is 7 dou. -/
def rice_obtained : ℕ := 7

/-- The system of equations representing the problem. -/
theorem grain_to_rice_system :
  (x + y = total_capacity) ∧ (conversion_rate * x + y = rice_obtained) := 
sorry

end grain_to_rice_system_l2161_216142


namespace f_analytical_expression_g_value_l2161_216138

noncomputable def f (ω x : ℝ) : ℝ := (1/2) * Real.sin (ω * x) - (Real.sqrt 3 / 2) * Real.sin (ω * x + Real.pi / 2)

noncomputable def g (ω x : ℝ) : ℝ := f ω (x + Real.pi / 4)

theorem f_analytical_expression (x : ℝ) (hω : ω = 2 ∧ ω > 0) : 
  f 2 x = Real.sin (2 * x - Real.pi / 3) :=
sorry

theorem g_value (α : ℝ) (hω : ω = 2 ∧ ω > 0) (h : g 2 (α / 2) = 4/5) : 
  g 2 (-α) = -7/25 :=
sorry

end f_analytical_expression_g_value_l2161_216138


namespace AM_GM_Inequality_l2161_216153

theorem AM_GM_Inequality 
  (a b c : ℝ) 
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (habc : a * b * c = 1) : 
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := by
  sorry

end AM_GM_Inequality_l2161_216153


namespace health_risk_probability_l2161_216158

theorem health_risk_probability :
  let p := 26
  let q := 57
  p + q = 83 :=
by {
  sorry
}

end health_risk_probability_l2161_216158


namespace find_ratio_MH_NH_OH_l2161_216175

-- Defining the main problem variables.
variable {A B C O H M N : Type} -- A, B, C are points, O is circumcenter, H is orthocenter, M and N are points on other segments
variables (angleA : ℝ) (AB AC : ℝ)
variables (angleBOC angleBHC : ℝ)
variables (BM CN MH NH OH : ℝ)

-- Conditions: Given constraints from the problem.
axiom angle_A_eq_60 : angleA = 60 -- ∠A = 60°
axiom AB_greater_AC : AB > AC -- AB > AC
axiom circumcenter_property : angleBOC = 120 -- ∠BOC = 120°
axiom orthocenter_property : angleBHC = 120 -- ∠BHC = 120°
axiom BM_eq_CN : BM = CN -- BM = CN

-- Statement of the mathematical proof we need to show.
theorem find_ratio_MH_NH_OH : (MH + NH) / OH = Real.sqrt 3 :=
by
  sorry

end find_ratio_MH_NH_OH_l2161_216175


namespace solve_mod_equation_l2161_216131

theorem solve_mod_equation (x : ℤ) (h : 10 * x + 3 ≡ 7 [ZMOD 18]) : x ≡ 4 [ZMOD 9] :=
sorry

end solve_mod_equation_l2161_216131


namespace part1_part2_l2161_216193

-- Define the absolute value function
def f (x : ℝ) (a : ℝ) : ℝ := abs (2 * x - a) + a

-- Given conditions
def condition1 : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 ↔ f x a ≤ 6

def condition2 (a : ℝ) : Prop :=
  ∃ t m : ℝ, f (t / 2) a ≤ m - f (-t) a

-- Statements to prove
theorem part1 : ∃ a : ℝ, condition1 ∧ a = 1 := by
  sorry

theorem part2 : ∀ {a : ℝ}, a = 1 → ∃ m : ℝ, m ≥ 3.5 ∧ condition2 a := by
  sorry

end part1_part2_l2161_216193


namespace A_eq_D_l2161_216179

def A := {θ : ℝ | 0 < θ ∧ θ < 90}
def D := {θ : ℝ | 0 < θ ∧ θ < 90}

theorem A_eq_D : A = D :=
by
  sorry

end A_eq_D_l2161_216179


namespace units_digit_of_factorial_sum_l2161_216146

theorem units_digit_of_factorial_sum : 
  (1 + 2 + 6 + 4) % 10 = 3 := sorry

end units_digit_of_factorial_sum_l2161_216146


namespace garden_length_increase_l2161_216130

variable (L W : ℝ)  -- Original length and width
variable (X : ℝ)    -- Percentage increase in length

theorem garden_length_increase :
  (1 + X / 100) * 0.8 = 1.1199999999999999 → X = 40 :=
by
  sorry

end garden_length_increase_l2161_216130


namespace ribbons_problem_l2161_216197

/-
    In a large box of ribbons, 1/3 are yellow, 1/4 are purple, 1/6 are orange, and the remaining 40 ribbons are black.
    Prove that the total number of orange ribbons is 27.
-/

theorem ribbons_problem :
  ∀ (total : ℕ), 
    (1 / 3 : ℚ) * total + (1 / 4 : ℚ) * total + (1 / 6 : ℚ) * total + 40 = total →
    (1 / 6 : ℚ) * total = 27 := sorry

end ribbons_problem_l2161_216197


namespace setB_is_correct_l2161_216128

def setA : Set ℤ := {1, 0, -1, 2}
def setB : Set ℤ := { y | ∃ x ∈ setA, y = Int.natAbs x }

theorem setB_is_correct : setB = {0, 1, 2} := by
  sorry

end setB_is_correct_l2161_216128


namespace monotonically_increasing_implies_non_negative_derivative_non_negative_derivative_not_implies_monotonically_increasing_l2161_216137

variables {f : ℝ → ℝ}

-- Definition that f is monotonically increasing
def monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 < x2 → f x1 ≤ f x2

-- Definition of the derivative being non-negative everywhere
def non_negative_derivative (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 0 ≤ (deriv f) x

theorem monotonically_increasing_implies_non_negative_derivative (f : ℝ → ℝ) :
  monotonically_increasing f → non_negative_derivative f :=
sorry

theorem non_negative_derivative_not_implies_monotonically_increasing (f : ℝ → ℝ) :
  non_negative_derivative f → ¬ monotonically_increasing f :=
sorry

end monotonically_increasing_implies_non_negative_derivative_non_negative_derivative_not_implies_monotonically_increasing_l2161_216137


namespace find_divisor_l2161_216165

theorem find_divisor (n : ℕ) (h_n : n = 36) : 
  ∃ D : ℕ, ((n + 10) * 2 / D) - 2 = 44 → D = 2 :=
by
  use 2
  intros h
  sorry

end find_divisor_l2161_216165


namespace sum_of_coefficients_l2161_216105

theorem sum_of_coefficients :
  ∃ (a b c d e : ℤ), (512 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) ∧ (a + b + c + d + e = 60) :=
by
  sorry

end sum_of_coefficients_l2161_216105


namespace value_of_y_l2161_216110

variables (x y : ℝ)

theorem value_of_y (h1 : x - y = 16) (h2 : x + y = 8) : y = -4 :=
by
  sorry

end value_of_y_l2161_216110


namespace find_m_value_l2161_216180

noncomputable def m_value (x : ℤ) (m : ℝ) : Prop :=
  3 * (x + 1) - 2 ≤ 4 * (x - 3) + 1 ∧
  (∃ x, x ≥ 12 ∧ (1 / 2 : ℝ) * x - m = 5)

theorem find_m_value : ∃ m : ℝ, ∀ x : ℤ, m_value x m → m = 1 :=
by
  sorry

end find_m_value_l2161_216180


namespace opposite_of_neg_11_l2161_216181

-- Define the opposite (negative) of a number
def opposite (a : ℤ) : ℤ := -a

-- Prove that the opposite of -11 is 11
theorem opposite_of_neg_11 : opposite (-11) = 11 := 
by
  -- Proof not required, so using sorry as placeholder
  sorry

end opposite_of_neg_11_l2161_216181


namespace obrien_hats_after_loss_l2161_216170

noncomputable def hats_simpson : ℕ := 15

noncomputable def initial_hats_obrien : ℕ := 2 * hats_simpson + 5

theorem obrien_hats_after_loss : initial_hats_obrien - 1 = 34 :=
by
  sorry

end obrien_hats_after_loss_l2161_216170


namespace container_volume_ratio_l2161_216156

theorem container_volume_ratio (V1 V2 : ℚ)
  (h1 : (3 / 5) * V1 = (2 / 3) * V2) :
  V1 / V2 = 10 / 9 :=
by sorry

end container_volume_ratio_l2161_216156


namespace ratio_of_pieces_l2161_216136

theorem ratio_of_pieces (total_length : ℝ) (shorter_piece : ℝ) : 
  total_length = 60 ∧ shorter_piece = 20 → shorter_piece / (total_length - shorter_piece) = 1 / 2 :=
by
  sorry

end ratio_of_pieces_l2161_216136


namespace prob_five_coins_heads_or_one_tail_l2161_216145

theorem prob_five_coins_heads_or_one_tail : 
  (∃ (H T : ℚ), H = 1/32 ∧ T = 31/32 ∧ H + T = 1) ↔ 1 = 1 :=
by sorry

end prob_five_coins_heads_or_one_tail_l2161_216145


namespace maximum_profit_l2161_216159

noncomputable def sales_volume (x : ℝ) : ℝ := -10 * x + 1000
noncomputable def profit (x : ℝ) : ℝ := -10 * x^2 + 1300 * x - 30000

theorem maximum_profit : ∀ x : ℝ, 44 ≤ x ∧ x ≤ 46 → profit x ≤ 8640 :=
by
  intro x hx
  sorry

end maximum_profit_l2161_216159


namespace problems_per_page_l2161_216164

theorem problems_per_page (total_problems finished_problems remaining_pages : Nat) (h1 : total_problems = 101) 
  (h2 : finished_problems = 47) (h3 : remaining_pages = 6) :
  (total_problems - finished_problems) / remaining_pages = 9 :=
by
  sorry

end problems_per_page_l2161_216164


namespace pipe_C_draining_rate_l2161_216186

noncomputable def pipe_rate := 25

def tank_capacity := 2000
def pipe_A_rate := 200
def pipe_B_rate := 50
def pipe_C_duration_per_cycle := 2
def pipe_A_duration := 1
def pipe_B_duration := 2
def cycle_duration := pipe_A_duration + pipe_B_duration + pipe_C_duration_per_cycle
def total_time := 40
def number_of_cycles := total_time / cycle_duration
def water_filled_per_cycle := (pipe_A_rate * pipe_A_duration) + (pipe_B_rate * pipe_B_duration)
def total_water_filled := number_of_cycles * water_filled_per_cycle
def excess_water := total_water_filled - tank_capacity 
def pipe_C_rate := excess_water / (pipe_C_duration_per_cycle * number_of_cycles)

theorem pipe_C_draining_rate :
  pipe_C_rate = pipe_rate := by
  sorry

end pipe_C_draining_rate_l2161_216186


namespace range_of_a_l2161_216118

variable {x a : ℝ}

def A : Set ℝ := {x | x ≤ -3 ∨ x > 2}
def B (a : ℝ) : Set ℝ := {x | x ≤ a - 1}

theorem range_of_a (A_union_B_R : A ∪ B a = Set.univ) : a ∈ Set.Ici 3 :=
  sorry

end range_of_a_l2161_216118


namespace smallest_part_proportional_division_l2161_216178

theorem smallest_part_proportional_division (a b c d total : ℕ) (h : a + b + c + d = total) (sum_equals_360 : 360 = total * 15):
  min (4 * 15) (min (5 * 15) (min (7 * 15) (8 * 15))) = 60 :=
by
  -- Defining the proportions and overall total
  let a := 5
  let b := 7
  let c := 4
  let d := 8
  let total_parts := a + b + c + d

  -- Given that the division is proportional
  let part_value := 360 / total_parts

  -- Assert that the smallest part is equal to the smallest proportion times the value of one part
  let smallest_part := c * part_value
  trivial

end smallest_part_proportional_division_l2161_216178


namespace number_of_non_empty_proper_subsets_of_A_range_of_m_for_A_superset_B_l2161_216177

-- Definitions for the sets A and B
def A : Set Int := {x | x^2 - 3 * x - 10 <= 0}
def B (m : Int) : Set Int := {x | m - 1 <= x ∧ x <= 2 * m + 1}

-- Proof for the number of non-empty proper subsets of A
theorem number_of_non_empty_proper_subsets_of_A (x : Int) (h : x ∈ A) : 2^(8 : Nat) - 2 = 254 := by
  sorry

-- Proof for the range of m such that A ⊇ B
theorem range_of_m_for_A_superset_B (m : Int) : (∀ x, x ∈ B m → x ∈ A) ↔ (m < -2 ∨ (-1 ≤ m ∧ m ≤ 2)) := by
  sorry

end number_of_non_empty_proper_subsets_of_A_range_of_m_for_A_superset_B_l2161_216177


namespace quadratic_minimization_l2161_216198

theorem quadratic_minimization : 
  ∃ x : ℝ, ∀ y : ℝ, (x^2 - 12 * x + 36 ≤ y^2 - 12 * y + 36) ∧ x^2 - 12 * x + 36 = 0 :=
by
  sorry

end quadratic_minimization_l2161_216198


namespace larger_exceeds_smaller_by_16_l2161_216151

-- Define the smaller number S and the larger number L in terms of the ratio 7:11
def S : ℕ := 28
def L : ℕ := (11 * S) / 7

-- State the theorem that the larger number exceeds the smaller number by 16
theorem larger_exceeds_smaller_by_16 : L - S = 16 :=
by
  -- Proof steps will go here
  sorry

end larger_exceeds_smaller_by_16_l2161_216151


namespace possible_values_of_b_l2161_216168

theorem possible_values_of_b 
        (b : ℤ)
        (h : ∃ x : ℤ, (x ^ 3 + 2 * x ^ 2 + b * x + 8 = 0)) :
        b = -81 ∨ b = -26 ∨ b = -12 ∨ b = -6 ∨ b = 4 ∨ b = 9 ∨ b = 47 :=
  sorry

end possible_values_of_b_l2161_216168


namespace two_trains_clearing_time_l2161_216143

noncomputable def length_train1 : ℝ := 100  -- Length of Train 1 in meters
noncomputable def length_train2 : ℝ := 160  -- Length of Train 2 in meters
noncomputable def speed_train1 : ℝ := 42 * 1000 / 3600  -- Speed of Train 1 in m/s
noncomputable def speed_train2 : ℝ := 30 * 1000 / 3600  -- Speed of Train 2 in m/s
noncomputable def total_distance : ℝ := length_train1 + length_train2  -- Total distance to be covered
noncomputable def relative_speed : ℝ := speed_train1 + speed_train2  -- Relative speed

theorem two_trains_clearing_time : total_distance / relative_speed = 13 := by
  sorry

end two_trains_clearing_time_l2161_216143


namespace intersection_A_B_l2161_216104

open Set

def A : Set ℤ := {x | ∃ n : ℤ, x = 3 * n - 1}
def B : Set ℤ := {x | 0 < x ∧ x < 6}

theorem intersection_A_B : A ∩ B = {2, 5} := by
  sorry

end intersection_A_B_l2161_216104


namespace trees_distance_l2161_216124

theorem trees_distance (num_trees : ℕ) (yard_length : ℕ) (trees_at_end : Prop) (tree_count : num_trees = 26) (yard_size : yard_length = 800) : 
  (yard_length / (num_trees - 1)) = 32 := 
by
  sorry

end trees_distance_l2161_216124


namespace tan_ratio_l2161_216188

theorem tan_ratio (p q : Real) (hpq1 : Real.sin (p + q) = 0.6) (hpq2 : Real.sin (p - q) = 0.3) : 
  Real.tan p / Real.tan q = 3 := 
by
  sorry

end tan_ratio_l2161_216188


namespace percentage_increase_is_50_l2161_216148

-- Defining the conditions
def new_wage : ℝ := 51
def original_wage : ℝ := 34
def increase : ℝ := new_wage - original_wage

-- Proving the required percentage increase is 50%
theorem percentage_increase_is_50 :
  (increase / original_wage) * 100 = 50 := by
  sorry

end percentage_increase_is_50_l2161_216148


namespace problem_statement_l2161_216120

theorem problem_statement (x : ℤ) (h₁ : (x - 5) / 7 = 7) : (x - 24) / 10 = 3 := 
sorry

end problem_statement_l2161_216120


namespace distinct_arrangements_balloon_l2161_216126

theorem distinct_arrangements_balloon : 
  let n := 7
  let r1 := 2
  let r2 := 2
  (Nat.factorial n) / ((Nat.factorial r1) * (Nat.factorial r2)) = 1260 :=
by
  sorry

end distinct_arrangements_balloon_l2161_216126


namespace area_of_trapezoid_EFGH_l2161_216183

-- Define the vertices of the trapezoid
structure Point where
  x : ℤ
  y : ℤ

def E : Point := ⟨-2, -3⟩
def F : Point := ⟨-2, 2⟩
def G : Point := ⟨4, 5⟩
def H : Point := ⟨4, 0⟩

-- Define the formula for the area of a trapezoid
def trapezoid_area (b1 b2 height : ℤ) : ℤ :=
  (b1 + b2) * height / 2

-- The proof statement
theorem area_of_trapezoid_EFGH : trapezoid_area (F.y - E.y) (G.y - H.y) (G.x - E.x) = 30 := by
  sorry -- proof not required

end area_of_trapezoid_EFGH_l2161_216183


namespace quadratic_second_root_l2161_216103

noncomputable def second_root (p q : ℝ) : ℝ :=
  -2 * p / (p - 2)

theorem quadratic_second_root (p q : ℝ) (h1 : (p + q) * 1^2 + (p - q) * 1 + p * q = 0) :
  ∃ r : ℝ, r = second_root p q :=
by 
  sorry

end quadratic_second_root_l2161_216103
