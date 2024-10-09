import Mathlib

namespace find_angle_A_l1003_100382

open Real

theorem find_angle_A (a b : ℝ) (B A : ℝ) 
  (ha : a = sqrt 2) 
  (hb : b = 2) 
  (hB : sin B + cos B = sqrt 2) :
  A = π / 6 := 
  sorry

end find_angle_A_l1003_100382


namespace fraction_incorrect_like_music_l1003_100396

-- Define the conditions as given in the problem
def total_students : ℕ := 100
def like_music_percentage : ℝ := 0.7
def dislike_music_percentage : ℝ := 1 - like_music_percentage

def correct_like_percentage : ℝ := 0.75
def incorrect_like_percentage : ℝ := 1 - correct_like_percentage

def correct_dislike_percentage : ℝ := 0.85
def incorrect_dislike_percentage : ℝ := 1 - correct_dislike_percentage

-- The number of students liking music
def like_music_students : ℝ := total_students * like_music_percentage
-- The number of students disliking music
def dislike_music_students : ℝ := total_students * dislike_music_percentage

-- The number of students who correctly say they like music
def correct_like_music_say : ℝ := like_music_students * correct_like_percentage
-- The number of students who incorrectly say they dislike music
def incorrect_dislike_music_say : ℝ := like_music_students * incorrect_like_percentage

-- The number of students who correctly say they dislike music
def correct_dislike_music_say : ℝ := dislike_music_students * correct_dislike_percentage
-- The number of students who incorrectly say they like music
def incorrect_like_music_say : ℝ := dislike_music_students * incorrect_dislike_percentage

-- The total number of students who say they like music
def total_say_like_music : ℝ := correct_like_music_say + incorrect_like_music_say

-- The final theorem we want to prove
theorem fraction_incorrect_like_music : ((incorrect_like_music_say : ℝ) / total_say_like_music) = (5 / 58) :=
by
  -- here we would provide the proof, but for now, we use sorry
  sorry

end fraction_incorrect_like_music_l1003_100396


namespace find_power_l1003_100326

theorem find_power (a b c d e : ℕ) (h1 : a = 105) (h2 : b = 21) (h3 : c = 25) (h4 : d = 45) (h5 : e = 49) 
(h6 : a ^ (3 : ℕ) = b * c * d * e) : 3 = 3 := by
  sorry

end find_power_l1003_100326


namespace years_before_marriage_l1003_100349

theorem years_before_marriage {wedding_anniversary : ℕ} 
  (current_year : ℕ) (met_year : ℕ) (years_before_dating : ℕ) :
  wedding_anniversary = 20 →
  current_year = 2025 →
  met_year = 2000 →
  years_before_dating = 2 →
  met_year + years_before_dating + (current_year - met_year - wedding_anniversary) = current_year - wedding_anniversary - years_before_dating + wedding_anniversary - current_year :=
by
  sorry

end years_before_marriage_l1003_100349


namespace students_helped_on_third_day_l1003_100318

theorem students_helped_on_third_day (books_total : ℕ) (books_per_student : ℕ) (students_day1 : ℕ) (students_day2 : ℕ) (students_day4 : ℕ) (books_day3 : ℕ) :
  books_total = 120 →
  books_per_student = 5 →
  students_day1 = 4 →
  students_day2 = 5 →
  students_day4 = 9 →
  books_day3 = books_total - ((students_day1 + students_day2 + students_day4) * books_per_student) →
  books_day3 / books_per_student = 6 :=
by
  sorry

end students_helped_on_third_day_l1003_100318


namespace count_quadruples_l1003_100314

open Real

theorem count_quadruples:
  ∃ qs : Finset (ℝ × ℝ × ℝ × ℝ),
  (∀ (a b c k : ℝ), (a, b, c, k) ∈ qs ↔ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 
    a^k = b * c ∧
    b^k = c * a ∧
    c^k = a * b
  ) ∧
  qs.card = 8 :=
sorry

end count_quadruples_l1003_100314


namespace cosine_inequality_l1003_100312

theorem cosine_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 0 < x^2 + y^2 ∧ x^2 + y^2 ≤ π) :
  1 + Real.cos (x * y) ≥ Real.cos x + Real.cos y :=
sorry

end cosine_inequality_l1003_100312


namespace steel_mill_production_2010_l1003_100320

noncomputable def steel_mill_production (P : ℕ → ℕ) : Prop :=
  (P 1990 = 400000) ∧ (P 2000 = 500000) ∧ ∀ n, (P n) = (P (n-1)) + (500000 - 400000) / 10

theorem steel_mill_production_2010 (P : ℕ → ℕ) (h : steel_mill_production P) : P 2010 = 630000 :=
by
  sorry -- proof omitted

end steel_mill_production_2010_l1003_100320


namespace relationship_between_a_b_c_d_l1003_100381

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.cos x)
noncomputable def g (x : ℝ) : ℝ := Real.cos (Real.sin x)

open Real

theorem relationship_between_a_b_c_d :
  ∀ (x : ℝ) (a b c d : ℝ),
  (0 ≤ x ∧ x ≤ π) →
  (∀ x, f x ≤ a ∧ b ≤ f x) →
  (∀ x, g x ≤ c ∧ d ≤ g x) →
  a = sin 1 →
  b = -sin 1 →
  c = 1 →
  d = cos 1 →
  b < d ∧ d < a ∧ a < c := by
  sorry

end relationship_between_a_b_c_d_l1003_100381


namespace dave_more_than_jerry_games_l1003_100321

variable (K D J : ℕ)  -- Declaring the variables for Ken, Dave, and Jerry respectively

-- Defining the conditions
def ken_more_games := K = D + 5
def dave_more_than_jerry := D > 7
def jerry_games := J = 7
def total_games := K + D + 7 = 32

-- Defining the proof problem
theorem dave_more_than_jerry_games (hK : ken_more_games K D) (hD : dave_more_than_jerry D) (hJ : jerry_games J) (hT : total_games K D) : D - 7 = 3 :=
by
  sorry

end dave_more_than_jerry_games_l1003_100321


namespace nominal_rate_of_interest_correct_l1003_100385

noncomputable def nominal_rate_of_interest (EAR : ℝ) (n : ℕ) : ℝ :=
  let i := by 
    sorry
  i

theorem nominal_rate_of_interest_correct :
  nominal_rate_of_interest 0.0609 2 = 0.0598 :=
by 
  sorry

end nominal_rate_of_interest_correct_l1003_100385


namespace shoe_length_size_15_l1003_100319

theorem shoe_length_size_15 : 
  ∀ (length : ℕ → ℝ), 
    (∀ n, 8 ≤ n ∧ n ≤ 17 → length (n + 1) = length n + 1 / 4) → 
    length 17 = (1 + 0.10) * length 8 →
    length 15 = 24.25 :=
by
  intro length h_increase h_largest
  sorry

end shoe_length_size_15_l1003_100319


namespace james_total_room_area_l1003_100307

theorem james_total_room_area :
  let original_length := 13
  let original_width := 18
  let increase := 2
  let new_length := original_length + increase
  let new_width := original_width + increase
  let area_of_one_room := new_length * new_width
  let number_of_rooms := 4
  let area_of_four_rooms := area_of_one_room * number_of_rooms
  let area_of_larger_room := area_of_one_room * 2
  let total_area := area_of_four_rooms + area_of_larger_room
  total_area = 1800
  := sorry

end james_total_room_area_l1003_100307


namespace measure_of_angle_A_possibilities_l1003_100315

theorem measure_of_angle_A_possibilities (A B : ℕ) (h1 : A + B = 180) (h2 : ∃ k : ℕ, k ≥ 1 ∧ A = k * B) : 
  ∃ n : ℕ, n = 17 :=
by
  -- the statement needs provable proof and equal 17
  -- skip the proof
  sorry

end measure_of_angle_A_possibilities_l1003_100315


namespace prob_a_prob_b_l1003_100378

-- Given conditions and question for Part a
def election_prob (p q : ℕ) (h : p > q) : ℚ :=
  (p - q) / (p + q)

theorem prob_a : election_prob 3 2 (by decide) = 1 / 5 :=
  sorry

-- Given conditions and question for Part b
theorem prob_b : election_prob 1010 1009 (by decide) = 1 / 2019 :=
  sorry

end prob_a_prob_b_l1003_100378


namespace dust_particles_calculation_l1003_100353

theorem dust_particles_calculation (D : ℕ) (swept : ℝ) (left_by_shoes : ℕ) (total_after_walk : ℕ)  
  (h_swept : swept = 9 / 10)
  (h_left_by_shoes : left_by_shoes = 223)
  (h_total_after_walk : total_after_walk = 331)
  (h_equation : (1 - swept) * D + left_by_shoes = total_after_walk) : 
  D = 1080 := 
by
  sorry

end dust_particles_calculation_l1003_100353


namespace num_positive_integers_m_l1003_100317

theorem num_positive_integers_m (h : ∀ m : ℕ, ∃ d : ℕ, 3087 = d ∧ m^2 = d + 3) :
  ∃! m : ℕ, 0 < m ∧ (3087 % (m^2 - 3) = 0) := by
  sorry

end num_positive_integers_m_l1003_100317


namespace determine_a_l1003_100375

theorem determine_a (x y a : ℝ) 
  (h1 : x + y = a)
  (h2 : x^3 + y^3 = a)
  (h3 : x^5 + y^5 = a) : 
  a = 0 := 
sorry

end determine_a_l1003_100375


namespace money_given_to_each_friend_l1003_100344

-- Define the conditions
def initial_amount : ℝ := 20.10
def money_spent_on_sweets : ℝ := 1.05
def amount_left : ℝ := 17.05
def number_of_friends : ℝ := 2.0

-- Theorem statement
theorem money_given_to_each_friend :
  (initial_amount - amount_left - money_spent_on_sweets) / number_of_friends = 1.00 :=
by
  sorry

end money_given_to_each_friend_l1003_100344


namespace solveAdultsMonday_l1003_100345

def numAdultsMonday (A : ℕ) : Prop :=
  let childrenMondayCost := 7 * 3
  let childrenTuesdayCost := 4 * 3
  let adultsTuesdayCost := 2 * 4
  let totalChildrenCost := childrenMondayCost + childrenTuesdayCost
  let totalAdultsCost := A * 4 + adultsTuesdayCost
  let totalRevenue := totalChildrenCost + totalAdultsCost
  totalRevenue = 61

theorem solveAdultsMonday : numAdultsMonday 5 := 
  by 
    -- Proof goes here
    sorry

end solveAdultsMonday_l1003_100345


namespace simplify_fraction_l1003_100343

theorem simplify_fraction (x : ℝ) (hx : x ≠ 1) : (x^2 / (x-1)) - (1 / (x-1)) = x + 1 :=
by 
  sorry

end simplify_fraction_l1003_100343


namespace compare_values_of_even_and_monotone_function_l1003_100399

variable (f : ℝ → ℝ)

def is_even_function := ∀ x : ℝ, f x = f (-x)
def is_monotone_increasing_on_nonneg := ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y

theorem compare_values_of_even_and_monotone_function
  (h_even : is_even_function f)
  (h_monotone : is_monotone_increasing_on_nonneg f) :
  f (-π) > f 3 ∧ f 3 > f (-2) :=
by
  sorry

end compare_values_of_even_and_monotone_function_l1003_100399


namespace modified_cube_surface_area_l1003_100377

noncomputable def total_surface_area_modified_cube : ℝ :=
  let side_length := 10
  let triangle_side := 7 * Real.sqrt 2
  let tunnel_wall_area := 3 * (Real.sqrt 3 / 4 * triangle_side^2)
  let original_surface_area := 6 * side_length^2
  original_surface_area + tunnel_wall_area

theorem modified_cube_surface_area : 
  total_surface_area_modified_cube = 600 + 73.5 * Real.sqrt 3 := 
  sorry

end modified_cube_surface_area_l1003_100377


namespace union_of_A_and_B_l1003_100351

open Set

def A : Set ℕ := {1, 3, 7, 8}
def B : Set ℕ := {1, 5, 8}

theorem union_of_A_and_B : A ∪ B = {1, 3, 5, 7, 8} := by
  sorry

end union_of_A_and_B_l1003_100351


namespace increase_in_area_correct_l1003_100376

-- Define the dimensions of the original rectangular garden
def length_rect := 60
def width_rect := 20

-- Define the perimeter of the rectangle
def perimeter_rect := 2 * (length_rect + width_rect)

-- Calculate the side length of the square garden using the same perimeter.
def side_square := perimeter_rect / 4

-- Define the area of the rectangular garden
def area_rect := length_rect * width_rect

-- Define the area of the square garden
def area_square := side_square * side_square

-- Define the increase in area after reshaping
def increase_in_area := area_square - area_rect

-- Prove that the increase in the area is 400 square feet
theorem increase_in_area_correct : increase_in_area = 400 := by
  -- The proof is omitted
  sorry

end increase_in_area_correct_l1003_100376


namespace solution_mod_5_l1003_100379

theorem solution_mod_5 (a : ℤ) : 
  (a^3 + 3 * a + 1) % 5 = 0 ↔ (a % 5 = 1 ∨ a % 5 = 2) := 
by
  sorry

end solution_mod_5_l1003_100379


namespace min_cells_marked_l1003_100340

/-- The minimum number of cells that need to be marked in a 50x50 grid so
each 1x6 vertical or horizontal strip has at least one marked cell is 416. -/
theorem min_cells_marked {n : ℕ} : n = 416 → 
  (∀ grid : Fin 50 × Fin 50, ∃ cells : Finset (Fin 50 × Fin 50), 
    (∀ (r c : Fin 50), (r = 6 * i + k ∨ c = 6 * i + k) →
      (∃ (cell : Fin 50 × Fin 50), cell ∈ cells)) →
    cells.card = n) := 
sorry

end min_cells_marked_l1003_100340


namespace intersection_of_A_and_B_l1003_100368

open Set Int

def A : Set ℝ := { x | x ^ 2 - 6 * x + 8 ≤ 0 }
def B : Set ℤ := { x | abs (x - 3) < 2 }

theorem intersection_of_A_and_B :
  (A ∩ (coe '' B) = { x : ℝ | x = 2 ∨ x = 3 ∨ x = 4 }) :=
by
  sorry

end intersection_of_A_and_B_l1003_100368


namespace solve_equation_1_solve_equation_2_l1003_100369

open Real

theorem solve_equation_1 (x : ℝ) (h_ne1 : x + 1 ≠ 0) (h_ne2 : x - 3 ≠ 0) : 
  (5 / (x + 1) = 1 / (x - 3)) → x = 4 :=
by
    intro h
    sorry

theorem solve_equation_2 (x : ℝ) (h_ne1 : x - 4 ≠ 0) (h_ne2 : 4 - x ≠ 0) :
    (3 - x) / (x - 4) = 1 / (4 - x) - 2 → False :=
by
    intro h
    sorry

end solve_equation_1_solve_equation_2_l1003_100369


namespace abs_condition_implies_l1003_100300

theorem abs_condition_implies (x : ℝ) 
  (h : |x - 1| < 2) : x < 3 := by
  sorry

end abs_condition_implies_l1003_100300


namespace divisor_of_2n_when_remainder_is_two_l1003_100372

theorem divisor_of_2n_when_remainder_is_two (n : ℤ) (k : ℤ) : 
  (n = 22 * k + 12) → ∃ d : ℤ, d = 22 ∧ (2 * n) % d = 2 :=
by
  sorry

end divisor_of_2n_when_remainder_is_two_l1003_100372


namespace max_x_values_l1003_100363

noncomputable def y (x : ℝ) : ℝ := (1/2) * (Real.cos x)^2 + (Real.sqrt 3 / 2) * (Real.sin x) * (Real.cos x) + 1

theorem max_x_values :
  {x : ℝ | ∃ k : ℤ, x = k * Real.pi + Real.pi / 6} = {x : ℝ | y x = y (x)} :=
sorry

end max_x_values_l1003_100363


namespace find_point_B_l1003_100380

-- Definition of Point
structure Point where
  x : ℝ
  y : ℝ

-- Definitions of conditions
def A : Point := ⟨1, 2⟩
def d : ℝ := 3
def AB_parallel_x (A B : Point) : Prop := A.y = B.y

theorem find_point_B (B : Point) (h_parallel : AB_parallel_x A B) (h_dist : abs (B.x - A.x) = d) :
  (B = ⟨4, 2⟩) ∨ (B = ⟨-2, 2⟩) :=
by
  sorry

end find_point_B_l1003_100380


namespace problem_statement_l1003_100303

theorem problem_statement (x y z : ℝ) (h : x^2 + y^2 + z^2 = 2) : x + y + z ≤ x * y * z + 2 := 
sorry

end problem_statement_l1003_100303


namespace compute_p2_q2_compute_p3_q3_l1003_100304

variables (p q : ℝ)

theorem compute_p2_q2 (h1 : p * q = 15) (h2 : p + q = 8) : p^2 + q^2 = 34 :=
sorry

theorem compute_p3_q3 (h1 : p * q = 15) (h2 : p + q = 8) : p^3 + q^3 = 152 :=
sorry

end compute_p2_q2_compute_p3_q3_l1003_100304


namespace avg_of_multiples_l1003_100392

theorem avg_of_multiples (n : ℝ) (h : (n + 2 * n + 3 * n + 4 * n + 5 * n + 6 * n + 7 * n + 8 * n + 9 * n + 10 * n) / 10 = 60.5) : n = 11 :=
by
  sorry

end avg_of_multiples_l1003_100392


namespace car_drive_highway_distance_l1003_100365

theorem car_drive_highway_distance
  (d_local : ℝ)
  (s_local : ℝ)
  (s_highway : ℝ)
  (s_avg : ℝ)
  (d_total := d_local + s_avg * (d_local / s_local + d_local / s_highway))
  (t_local := d_local / s_local)
  (t_highway : ℝ := (d_total - d_local) / s_highway)
  (t_total := t_local + t_highway)
  (avg_speed := (d_total) / t_total)
  : d_local = 60 → s_local = 20 → s_highway = 60 → s_avg = 36 → avg_speed = 36 → d_total - d_local = 120 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4]
  sorry

end car_drive_highway_distance_l1003_100365


namespace sample_size_120_l1003_100388

theorem sample_size_120
  (x y : ℕ)
  (h_ratio : x / 2 = y / 3 ∧ y / 3 = 60 / 5)
  (h_max : max x (max y 60) = 60) :
  x + y + 60 = 120 := by
  sorry

end sample_size_120_l1003_100388


namespace marie_keeps_lollipops_l1003_100347

def total_lollipops (raspberry mint blueberry coconut : ℕ) : ℕ :=
  raspberry + mint + blueberry + coconut

def lollipops_per_friend (total friends : ℕ) : ℕ :=
  total / friends

def lollipops_kept (total friends : ℕ) : ℕ :=
  total % friends

theorem marie_keeps_lollipops :
  lollipops_kept (total_lollipops 75 132 9 315) 13 = 11 :=
by
  sorry

end marie_keeps_lollipops_l1003_100347


namespace pages_written_on_wednesday_l1003_100356

variable (minutesMonday minutesTuesday rateMonday rateTuesday : ℕ)
variable (totalPages : ℕ)

def pagesOnMonday (minutesMonday rateMonday : ℕ) : ℕ :=
  minutesMonday / rateMonday

def pagesOnTuesday (minutesTuesday rateTuesday : ℕ) : ℕ :=
  minutesTuesday / rateTuesday

def totalPagesMondayAndTuesday (minutesMonday minutesTuesday rateMonday rateTuesday : ℕ) : ℕ :=
  pagesOnMonday minutesMonday rateMonday + pagesOnTuesday minutesTuesday rateTuesday

def pagesOnWednesday (minutesMonday minutesTuesday rateMonday rateTuesday totalPages : ℕ) : ℕ :=
  totalPages - totalPagesMondayAndTuesday minutesMonday minutesTuesday rateMonday rateTuesday

theorem pages_written_on_wednesday :
  pagesOnWednesday 60 45 30 15 10 = 5 := by
  sorry

end pages_written_on_wednesday_l1003_100356


namespace avg_score_all_matches_l1003_100329

-- Definitions from the conditions
variable (score1 score2 : ℕ → ℕ) 
variable (avg1 avg2 : ℕ)
variable (count1 count2 : ℕ)

-- Assumptions from the conditions
axiom avg_score1 : avg1 = 30
axiom avg_score2 : avg2 = 40
axiom count1_matches : count1 = 2
axiom count2_matches : count2 = 3

-- The proof statement
theorem avg_score_all_matches : 
  ((score1 0 + score1 1) + (score2 0 + score2 1 + score2 2)) / (count1 + count2) = 36 := 
  sorry

end avg_score_all_matches_l1003_100329


namespace polynomial_square_solution_l1003_100338

variable (a b : ℝ)

theorem polynomial_square_solution (h : 
  ∃ g : Polynomial ℝ, g^2 = Polynomial.C (1 : ℝ) * Polynomial.X^4 -
  Polynomial.C (1 : ℝ) * Polynomial.X^3 +
  Polynomial.C (1 : ℝ) * Polynomial.X^2 +
  Polynomial.C a * Polynomial.X +
  Polynomial.C b) : b = 9 / 64 :=
by sorry

end polynomial_square_solution_l1003_100338


namespace find_selling_price_l1003_100342

-- Define the basic parameters
def cost := 80
def s0 := 30
def profit0 := 50
def desired_profit := 2000

-- Additional shirts sold per price reduction
def add_shirts (p : ℕ) := 2 * p

-- Number of shirts sold given selling price x
def num_shirts (x : ℕ) := 290 - 2 * x

-- Profit equation
def profit_equation (x : ℕ) := (x - cost) * num_shirts x = desired_profit

theorem find_selling_price (x : ℕ) :
  (x = 105 ∨ x = 120) ↔ profit_equation x := by
  sorry

end find_selling_price_l1003_100342


namespace girls_came_in_classroom_l1003_100370

theorem girls_came_in_classroom (initial_boys initial_girls boys_left final_children girls_in_classroom : ℕ)
  (h1 : initial_boys = 5)
  (h2 : initial_girls = 4)
  (h3 : boys_left = 3)
  (h4 : final_children = 8)
  (h5 : girls_in_classroom = final_children - (initial_boys - boys_left)) :
  girls_in_classroom - initial_girls = 2 :=
by
  sorry

end girls_came_in_classroom_l1003_100370


namespace carol_weight_l1003_100364

variable (a c : ℝ)

theorem carol_weight (h1 : a + c = 240) (h2 : c - a = (2 / 3) * c) : c = 180 :=
by
  sorry

end carol_weight_l1003_100364


namespace find_abs_xyz_l1003_100373

noncomputable def conditions_and_question (x y z : ℝ) : Prop :=
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x ≠ y ∧ y ≠ z ∧ z ≠ x ∧
  (x + 1/y = y + 1/z ∧ y + 1/z = z + 1/x + 1)

theorem find_abs_xyz (x y z : ℝ) (h : conditions_and_question x y z) : |x * y * z| = 1 :=
  sorry

end find_abs_xyz_l1003_100373


namespace correct_option_l1003_100328

theorem correct_option :
  (∀ a : ℝ, a ≠ 0 → (a ^ 0 = 1)) ∧
  ¬(∀ a : ℝ, a ≠ 0 → (a^6 / a^3 = a^2)) ∧
  ¬(∀ a : ℝ, a ≠ 0 → ((a^2)^3 = a^5)) ∧
  ¬(∀ a b : ℝ, a ≠ 0 → b ≠ 0 → (a / (a + b)^2 + b / (a + b)^2 = a + b)) :=
by {
  sorry
}

end correct_option_l1003_100328


namespace problem1_solution_set_problem2_range_of_m_l1003_100335

open Real

noncomputable def f (x : ℝ) := abs (x + 1) - abs (x - 2)

theorem problem1_solution_set :
  {x : ℝ | f x ≥ 1} = {x : ℝ | x ≥ 1} :=
sorry

theorem problem2_range_of_m (m : ℝ) :
  (∃ x : ℝ, f x ≥ x^2 - x + m) → m ≤ 5 / 4 :=
sorry

end problem1_solution_set_problem2_range_of_m_l1003_100335


namespace games_given_to_neil_is_five_l1003_100393

variable (x : ℕ)

def initial_games_henry : ℕ := 33
def initial_games_neil : ℕ := 2
def games_given_to_neil : ℕ := x

theorem games_given_to_neil_is_five
  (H : initial_games_henry - games_given_to_neil = 4 * (initial_games_neil + games_given_to_neil)) :
  games_given_to_neil = 5 := by
  sorry

end games_given_to_neil_is_five_l1003_100393


namespace missing_fraction_l1003_100346

theorem missing_fraction (x : ℕ) (h1 : x > 0) :
  let lost := (1 / 3 : ℚ) * x
  let found := (2 / 3 : ℚ) * lost
  let remaining := x - lost + found
  (x - remaining) / x = 1 / 9 :=
by
  sorry

end missing_fraction_l1003_100346


namespace find_y_value_l1003_100334

-- Define the linear relationship
def linear_eq (k b x : ℝ) : ℝ := k * x + b

-- Given conditions
variables (k b : ℝ)
axiom h1 : linear_eq k b 0 = -1
axiom h2 : linear_eq k b (1/2) = 2

-- Prove that the value of y when x = -1/2 is -4
theorem find_y_value : linear_eq k b (-1/2) = -4 :=
by sorry

end find_y_value_l1003_100334


namespace nancy_pensils_total_l1003_100397

theorem nancy_pensils_total
  (initial: ℕ) 
  (mult_factor: ℕ) 
  (add_pencils: ℕ) 
  (final_total: ℕ) 
  (h1: initial = 27)
  (h2: mult_factor = 4)
  (h3: add_pencils = 45):
  final_total = initial * mult_factor + add_pencils := 
by
  sorry

end nancy_pensils_total_l1003_100397


namespace SWE4_l1003_100374

theorem SWE4 (a : ℕ → ℕ) (n : ℕ) :
  a 0 = 0 →
  (∀ n, a (n + 1) = 2 * a n + 2^n) →
  (∃ k : ℕ, n = 2^k) →
  ∃ m : ℕ, a n = 2^m :=
by
  intros h₀ h_recurrence h_power
  sorry

end SWE4_l1003_100374


namespace correct_conclusions_l1003_100361

variable (x1 x2 : ℝ) (hx1 : x1 > 0) (hx2 : x2 > 0)

def f (x : ℝ) : ℝ := x^2

theorem correct_conclusions (h_distinct : x1 ≠ x2) :
  (f x1 * x2 = f x1 * f x2) ∧
  ((f x1 - f x2) / (x1 - x2) > 0) ∧
  (f ((x1 + x2) / 2) < (f x1 + f x2) / 2) :=
by
  sorry

end correct_conclusions_l1003_100361


namespace new_solution_percentage_l1003_100387

theorem new_solution_percentage 
  (initial_weight : ℝ) (evaporated_water : ℝ) (added_solution_weight : ℝ) 
  (percentage_X : ℝ) (percentage_water : ℝ)
  (total_initial_X : ℝ := initial_weight * percentage_X)
  (initial_water : ℝ := initial_weight * percentage_water)
  (post_evaporation_weight : ℝ := initial_weight - evaporated_water)
  (post_evaporation_X : ℝ := total_initial_X)
  (post_evaporation_water : ℝ := post_evaporation_weight - total_initial_X)
  (added_X : ℝ := added_solution_weight * percentage_X)
  (added_water : ℝ := added_solution_weight * percentage_water)
  (total_X : ℝ := post_evaporation_X + added_X)
  (total_water : ℝ := post_evaporation_water + added_water)
  (new_total_weight : ℝ := post_evaporation_weight + added_solution_weight) :
  (total_X / new_total_weight) * 100 = 41.25 := 
by {
  sorry
}

end new_solution_percentage_l1003_100387


namespace complex_number_solution_l1003_100348

theorem complex_number_solution (a b : ℝ) (i : ℂ) (h₀ : Complex.I = i)
  (h₁ : (a - 2* (i^3)) / (b + i) = i) : a + b = 1 :=
by 
  sorry

end complex_number_solution_l1003_100348


namespace unpainted_cubes_eq_210_l1003_100310

-- Defining the structure of the 6x6x6 cube
def cube := Fin 6 × Fin 6 × Fin 6

-- Number of unit cubes in a 6x6x6 cube
def total_cubes : ℕ := 6 * 6 * 6

-- Number of unit squares painted by the plus pattern on each face
def squares_per_face := 13

-- Number of faces on the cube
def faces := 6

-- Initial total number of painted squares
def initial_painted_squares := squares_per_face * faces

-- Number of over-counted squares along edges
def edge_overcount := 12 * 2

-- Number of over-counted squares at corners
def corner_overcount := 8 * 1

-- Adjusted number of painted unit squares accounting for overcounts
noncomputable def adjusted_painted_squares := initial_painted_squares - edge_overcount - corner_overcount

-- Overlap adjustment: edge units and corner units
def edges_overlap := 24
def corners_overlap := 16

-- Final number of unique painted unit cubes
noncomputable def unique_painted_cubes := adjusted_painted_squares - edges_overlap - corners_overlap

-- Final unpainted unit cubes calculation
noncomputable def unpainted_cubes := total_cubes - unique_painted_cubes

-- Theorem to prove the number of unpainted unit cubes is 210
theorem unpainted_cubes_eq_210 : unpainted_cubes = 210 := by
  sorry

end unpainted_cubes_eq_210_l1003_100310


namespace find_certain_number_l1003_100309

theorem find_certain_number (x certain_number : ℤ) 
  (h1 : (28 + x + 42 + 78 + 104) / 5 = 62) 
  (h2 : (certain_number + 62 + 98 + 124 + x) / 5 = 78) : 
  certain_number = 106 := 
by 
  sorry

end find_certain_number_l1003_100309


namespace all_statements_imply_negation_l1003_100305

theorem all_statements_imply_negation :
  let s1 := (true ∧ true ∧ false)
  let s2 := (false ∧ true ∧ true)
  let s3 := (true ∧ false ∧ true)
  let s4 := (false ∧ false ∧ true)
  (s1 → ¬(true ∧ true ∧ true)) ∧
  (s2 → ¬(true ∧ true ∧ true)) ∧
  (s3 → ¬(true ∧ true ∧ true)) ∧
  (s4 → ¬(true ∧ true ∧ true)) :=
by sorry

end all_statements_imply_negation_l1003_100305


namespace find_k_l1003_100308

def f (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 5 else n - 2

theorem find_k :
  ∃ k : ℤ, k % 2 = 1 ∧ f (f (f k)) = 35 ∧ k = 29 := 
sorry

end find_k_l1003_100308


namespace partial_fraction_decomposition_l1003_100339

theorem partial_fraction_decomposition :
  ∃ (a b c : ℤ), (0 ≤ a ∧ a < 5) ∧ (0 ≤ b ∧ b < 13) ∧ (1 / 2015 = a / 5 + b / 13 + c / 31) ∧ (a + b = 14) :=
sorry

end partial_fraction_decomposition_l1003_100339


namespace inequality_proof_l1003_100322

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≥ 2) : |x^16 - y^16| + 4 * x^8 * y^8 ≥ 4 := 
sorry

end inequality_proof_l1003_100322


namespace find_divided_number_l1003_100383

theorem find_divided_number :
  ∃ (Number : ℕ), ∃ (q r d : ℕ), q = 8 ∧ r = 3 ∧ d = 21 ∧ Number = d * q + r ∧ Number = 171 :=
by
  sorry

end find_divided_number_l1003_100383


namespace car_speeds_midpoint_condition_l1003_100390

theorem car_speeds_midpoint_condition 
  (v k : ℝ) (h_k : k > 1) 
  (A B C D : ℝ) (AB AD CD : ℝ)
  (h_midpoint : AD = AB / 2) 
  (h_CD_AD : CD / AD = 1 / 2)
  (h_D_midpoint : D = (A + B) / 2) 
  (h_C_on_return : C = D - CD) 
  (h_speeds : (v > 0) ∧ (k * v > v)) 
  (h_AB_AD : AB = 2 * AD) :
  k = 2 :=
by
  sorry

end car_speeds_midpoint_condition_l1003_100390


namespace years_of_interest_l1003_100362

noncomputable def principal : ℝ := 2600
noncomputable def interest_difference : ℝ := 78

theorem years_of_interest (R : ℝ) (N : ℝ) (h : (principal * (R + 1) * N / 100) - (principal * R * N / 100) = interest_difference) : N = 3 :=
sorry

end years_of_interest_l1003_100362


namespace fraction_ratio_l1003_100366

variable (M Q P N R : ℝ)

theorem fraction_ratio (h1 : M = 0.40 * Q)
                       (h2 : Q = 0.25 * P)
                       (h3 : N = 0.40 * R)
                       (h4 : R = 0.75 * P) :
  M / N = 1 / 3 := 
by
  -- proof steps can be provided here
  sorry

end fraction_ratio_l1003_100366


namespace fraction_product_l1003_100386

theorem fraction_product : 
  (7 / 5) * (8 / 16) * (21 / 15) * (14 / 28) * (35 / 25) * (20 / 40) * (49 / 35) * (32 / 64) = 2401 / 10000 :=
by
  -- This line is to skip the proof
  sorry

end fraction_product_l1003_100386


namespace rain_on_Tuesday_correct_l1003_100395

-- Let the amount of rain on Monday be represented by m
def rain_on_Monday : ℝ := 0.9

-- Let the difference in rain between Monday and Tuesday be represented by d
def rain_difference : ℝ := 0.7

-- Define the calculated amount of rain on Tuesday
def rain_on_Tuesday : ℝ := rain_on_Monday - rain_difference

-- The statement we need to prove
theorem rain_on_Tuesday_correct : rain_on_Tuesday = 0.2 := 
by
  -- Proof omitted (to be provided)
  sorry

end rain_on_Tuesday_correct_l1003_100395


namespace solve_inequality_l1003_100323

-- Define the inequality as a function
def inequality_holds (x : ℝ) : Prop :=
  (2 * x + 3) / (x + 4) > (4 * x + 5) / (3 * x + 10)

-- Define the solution set as intervals excluding the points
def solution_set (x : ℝ) : Prop :=
  x < -5 / 2 ∨ x > -2

theorem solve_inequality (x : ℝ) : inequality_holds x ↔ solution_set x :=
by sorry

end solve_inequality_l1003_100323


namespace range_of_m_l1003_100360

-- Define the set A and condition
def A (m : ℝ) : Set ℝ := { x : ℝ | x^2 - 2 * x + m = 0 }

-- The theorem stating the range of m
theorem range_of_m (m : ℝ) : (A m = ∅) ↔ m > 1 :=
by
  sorry

end range_of_m_l1003_100360


namespace harmonic_mean_lcm_gcd_sum_l1003_100336

theorem harmonic_mean_lcm_gcd_sum {m n : ℕ} (h_lcm : Nat.lcm m n = 210) (h_gcd : Nat.gcd m n = 6) (h_sum : m + n = 72) :
  (1 / (m : ℚ) + 1 / (n : ℚ)) = 2 / 35 := 
sorry

end harmonic_mean_lcm_gcd_sum_l1003_100336


namespace abs_neg_2023_l1003_100302

theorem abs_neg_2023 : abs (-2023) = 2023 := by
  sorry

end abs_neg_2023_l1003_100302


namespace marcus_dropped_8_pies_l1003_100337

-- Step d): Rewrite as a Lean 4 statement
-- Define all conditions from the problem
def total_pies (pies_per_batch : ℕ) (batches : ℕ) : ℕ :=
  pies_per_batch * batches

def pies_dropped (total_pies : ℕ) (remaining_pies : ℕ) : ℕ :=
  total_pies - remaining_pies

-- Prove that Marcus dropped 8 pies
theorem marcus_dropped_8_pies : 
  total_pies 5 7 - 27 = 8 := by
  sorry

end marcus_dropped_8_pies_l1003_100337


namespace cost_of_five_dozens_l1003_100367

-- Define cost per dozen given the total cost for two dozen
noncomputable def cost_per_dozen : ℝ := 15.60 / 2

-- Define the number of dozen apples we want to calculate the cost for
def number_of_dozens := 5

-- Define the total cost for the given number of dozens
noncomputable def total_cost (n : ℕ) : ℝ := n * cost_per_dozen

-- State the theorem
theorem cost_of_five_dozens : total_cost number_of_dozens = 39 :=
by
  unfold total_cost cost_per_dozen
  sorry

end cost_of_five_dozens_l1003_100367


namespace workman_problem_l1003_100398

theorem workman_problem (x : ℝ) (h : (1 / x) + (1 / (2 * x)) = 1 / 32): x = 48 :=
sorry

end workman_problem_l1003_100398


namespace cos_seven_pi_over_four_l1003_100330

theorem cos_seven_pi_over_four : Real.cos (7 * Real.pi / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end cos_seven_pi_over_four_l1003_100330


namespace eq_of_div_eq_div_l1003_100324

theorem eq_of_div_eq_div {a b c : ℝ} (h : a / c = b / c) (hc : c ≠ 0) : a = b :=
by
  sorry

end eq_of_div_eq_div_l1003_100324


namespace negation_of_proposition_l1003_100331

theorem negation_of_proposition (p : Real → Prop) : 
  (∀ x : Real, p x) → ¬(∀ x : Real, x ≥ 1) ↔ (∃ x : Real, x < 1) := 
by sorry

end negation_of_proposition_l1003_100331


namespace geometric_sum_2015_2016_l1003_100357

theorem geometric_sum_2015_2016 (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) 
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_a1 : a 1 = 2)
  (h_a2_a5 : a 2 + a 5 = 0)
  (h_Sn : ∀ n, S n = (1 - (-1)^n)) :
  S 2015 + S 2016 = 2 :=
by sorry

end geometric_sum_2015_2016_l1003_100357


namespace solveEquation1_proof_solveEquation2_proof_l1003_100354

noncomputable def solveEquation1 : Set ℝ :=
  { x | 2 * x^2 - 5 * x = 0 }

theorem solveEquation1_proof :
  solveEquation1 = { 0, (5 / 2 : ℝ) } :=
by
  sorry

noncomputable def solveEquation2 : Set ℝ :=
  { x | x^2 + 3 * x - 3 = 0 }

theorem solveEquation2_proof :
  solveEquation2 = { ( (-3 + Real.sqrt 21) / 2 : ℝ ), ( (-3 - Real.sqrt 21) / 2 : ℝ ) } :=
by
  sorry

end solveEquation1_proof_solveEquation2_proof_l1003_100354


namespace perfect_square_condition_l1003_100394

theorem perfect_square_condition (m : ℤ) : 
  (∃ k : ℤ, (x - 1) * (x + 3) * (x - 4) * (x - 8) + m = k^2) ↔ m = 196 :=
by sorry

end perfect_square_condition_l1003_100394


namespace sum_of_integers_l1003_100384

theorem sum_of_integers:
  ∀ (m n p q : ℕ),
    m ≠ n → m ≠ p → m ≠ q → n ≠ p → n ≠ q → p ≠ q →
    (8 - m) * (8 - n) * (8 - p) * (8 - q) = 9 →
    m + n + p + q = 32 :=
by
  intros m n p q hmn hmp hmq hnp hnq hpq heq
  sorry

end sum_of_integers_l1003_100384


namespace horner_eval_hex_to_decimal_l1003_100333

-- Problem 1: Evaluate the polynomial using Horner's method
theorem horner_eval (x : ℤ) (f : ℤ → ℤ) (v3 : ℤ) :
  (f x = 3 * x^6 + 5 * x^5 + 6 * x^4 + 79 * x^3 - 8 * x^2 + 35 * x + 12) →
  x = -4 →
  v3 = (((((3 * x + 5) * x + 6) * x + 79) * x - 8) * x + 35) * x + 12 →
  v3 = -57 :=
by
  intros hf hx hv
  sorry

-- Problem 2: Convert hexadecimal base-6 to decimal
theorem hex_to_decimal (hex : ℕ) (dec : ℕ) :
  hex = 210 →
  dec = 0 * 6^0 + 1 * 6^1 + 2 * 6^2 →
  dec = 78 :=
by
  intros hhex hdec
  sorry

end horner_eval_hex_to_decimal_l1003_100333


namespace total_amount_division_l1003_100350

variables (w x y z : ℝ)

theorem total_amount_division (h_w : w = 2)
                              (h_x : x = 0.75)
                              (h_y : y = 1.25)
                              (h_z : z = 0.85)
                              (h_share_y : y * Rs48_50 = Rs48_50) :
                              total_amount = 4.85 * 38.80 := sorry

end total_amount_division_l1003_100350


namespace distinct_prime_factors_of_90_l1003_100325

theorem distinct_prime_factors_of_90 : 
  ∃ (s : Finset ℕ), s = {2, 3, 5} ∧ ∀ p ∈ s, Nat.Prime p ∧ 2 * 3 * 3 * 5 = 90 :=
by
  sorry

end distinct_prime_factors_of_90_l1003_100325


namespace point_N_coordinates_l1003_100316

/--
Given:
- point M with coordinates (5, -6)
- vector a = (1, -2)
- the vector NM equals 3 times vector a
Prove:
- the coordinates of point N are (2, 0)
-/

theorem point_N_coordinates (x y : ℝ) :
  let M := (5, -6)
  let a := (1, -2)
  let NM := (5 - x, -6 - y)
  3 * a = NM → 
  (x = 2 ∧ y = 0) :=
by 
  intros
  sorry

end point_N_coordinates_l1003_100316


namespace line_points_k_l1003_100301

noncomputable def k : ℝ := 8

theorem line_points_k (k : ℝ) : 
  (∀ k : ℝ, ∃ b : ℝ, b = (10 - k) / (5 - 5) ∧
  ∀ b, b = (-k) / (20 - 5) → k = 8) :=
  by
  sorry

end line_points_k_l1003_100301


namespace intersection_of_sets_l1003_100355

def setP : Set ℝ := { x | x ≤ 3 }
def setQ : Set ℝ := { x | x > 1 }

theorem intersection_of_sets : setP ∩ setQ = { x | 1 < x ∧ x ≤ 3 } :=
by
  sorry

end intersection_of_sets_l1003_100355


namespace pieces_after_10_cuts_l1003_100313

-- Define the number of cuts
def cuts : ℕ := 10

-- Define the function that calculates the number of pieces
def pieces (k : ℕ) : ℕ := k + 1

-- State the theorem to prove the number of pieces given 10 cuts
theorem pieces_after_10_cuts : pieces cuts = 11 :=
by
  -- Proof goes here
  sorry

end pieces_after_10_cuts_l1003_100313


namespace loan_amount_principal_l1003_100359

-- Definitions based on conditions
def rate_of_interest := 3
def time_period := 3
def simple_interest := 108

-- Question translated to Lean 4 statement
theorem loan_amount_principal : ∃ P, (simple_interest = (P * rate_of_interest * time_period) / 100) ∧ P = 1200 :=
sorry

end loan_amount_principal_l1003_100359


namespace Lavinia_daughter_age_difference_l1003_100306

-- Define the ages of the individuals involved
variables (Ld Ls Kd : ℕ)

-- Conditions given in the problem
variables (H1 : Kd = 12)
variables (H2 : Ls = 2 * Kd)
variables (H3 : Ls = Ld + 22)

-- Statement we need to prove
theorem Lavinia_daughter_age_difference(Ld Ls Kd : ℕ) (H1 : Kd = 12) (H2 : Ls = 2 * Kd) (H3 : Ls = Ld + 22) : 
  Kd - Ld = 10 :=
sorry

end Lavinia_daughter_age_difference_l1003_100306


namespace possible_values_of_a_l1003_100389

def setA := {x : ℝ | x ≥ 3}
def setB (a : ℝ) := {x : ℝ | 2 * a - x > 1}
def complementB (a : ℝ) := {x : ℝ | x ≥ (2 * a - 1)}

theorem possible_values_of_a (a : ℝ) :
  (∀ x, x ∈ setA → x ∈ complementB a) ↔ (a = -2 ∨ a = 0 ∨ a = 2) :=
by
  sorry

end possible_values_of_a_l1003_100389


namespace Ashok_took_six_subjects_l1003_100352

theorem Ashok_took_six_subjects
  (n : ℕ) -- number of subjects Ashok took
  (T : ℕ) -- total marks secured in those subjects
  (h_avg_n : T = n * 72) -- condition: average of marks in n subjects is 72
  (h_avg_5 : 5 * 74 = 370) -- condition: average of marks in 5 subjects is 74
  (h_6th_mark : 62 > 0) -- condition: the 6th subject's mark is 62
  (h_T : T = 370 + 62) -- condition: total marks including the 6th subject
  : n = 6 := 
sorry


end Ashok_took_six_subjects_l1003_100352


namespace least_number_four_digits_divisible_by_15_25_40_75_l1003_100341

noncomputable def least_four_digit_multiple : ℕ :=
  1200

theorem least_number_four_digits_divisible_by_15_25_40_75 :
  (∀ n, (n ∣ 15) ∧ (n ∣ 25) ∧ (n ∣ 40) ∧ (n ∣ 75)) → least_four_digit_multiple = 1200 :=
sorry

end least_number_four_digits_divisible_by_15_25_40_75_l1003_100341


namespace count_integers_in_solution_set_l1003_100332

-- Define the predicate for the condition given in the problem
def condition (x : ℝ) : Prop := abs (x - 3) ≤ 4.5

-- Define the list of integers within the range of the condition
def solution_set : List ℤ := [-1, 0, 1, 2, 3, 4, 5, 6, 7]

-- Prove that the number of integers satisfying the condition is 8
theorem count_integers_in_solution_set : solution_set.length = 8 :=
by
  sorry

end count_integers_in_solution_set_l1003_100332


namespace solve_for_x_l1003_100327

theorem solve_for_x (x : ℝ) (h : (x / 6) / 3 = (9 / (x / 3))^2) : x = 23.43 :=
by {
  sorry
}

end solve_for_x_l1003_100327


namespace find_m_given_root_exists_l1003_100391

theorem find_m_given_root_exists (x m : ℝ) (h : ∃ x, x ≠ 2 ∧ (x / (x - 2) - 2 = m / (x - 2))) : m = 2 :=
by
  sorry

end find_m_given_root_exists_l1003_100391


namespace max_value_expr_l1003_100358

theorem max_value_expr (x y z : ℝ)
  (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) 
  (hxyz : x + y + z = 3) :
  (x^2 - x * y + y^2) * (x^2 - x * z + z^2) * (y^2 - y * z + z^2) * (x - y + z) ≤ 2187 / 216 :=
sorry

end max_value_expr_l1003_100358


namespace min_distinct_values_l1003_100311

theorem min_distinct_values (n : ℕ) (mode_count : ℕ) (total_count : ℕ) 
  (h_mode : mode_count = 10) (h_total : total_count = 2018) 
  (h_distinct : ∀ k, k ≠ mode_count → k < 10) : 
  n ≥ 225 :=
by
  sorry

end min_distinct_values_l1003_100311


namespace tree_planting_campaign_l1003_100371

theorem tree_planting_campaign
  (P : ℝ)
  (h1 : 456 = P * (1 - 1/20))
  (h2 : P ≥ 0)
  : (P * (1 + 0.1)) = (456 / (1 - 1/20) * 1.1) :=
by
  sorry

end tree_planting_campaign_l1003_100371
