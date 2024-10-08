import Mathlib

namespace fraction_of_square_above_line_l182_182534

theorem fraction_of_square_above_line :
  let A := (2, 1)
  let B := (5, 1)
  let C := (5, 4)
  let D := (2, 4)
  let P := (2, 3)
  let Q := (5, 1)
  ∃ f : ℚ, f = 2 / 3 := 
by
  -- Placeholder for the proof
  sorry

end fraction_of_square_above_line_l182_182534


namespace sum_of_absolute_values_l182_182496

theorem sum_of_absolute_values (S : ℕ → ℤ) (a : ℕ → ℤ) :
  (∀ n, S n = n^2 - 4 * n + 2) →
  a 1 = -1 →
  (∀ n, 1 < n → a n = 2 * n - 5) →
  ((abs (a 1) + abs (a 2) + abs (a 3) + abs (a 4) + abs (a 5) +
    abs (a 6) + abs (a 7) + abs (a 8) + abs (a 9) + abs (a 10)) = 66) :=
by
  intros hS a1_eq ha_eq
  sorry

end sum_of_absolute_values_l182_182496


namespace sum_of_a_c_l182_182603

theorem sum_of_a_c (a b c d : ℝ) (h1 : -2 * abs (1 - a) + b = 7) (h2 : 2 * abs (1 - c) + d = 7)
    (h3 : -2 * abs (11 - a) + b = -1) (h4 : 2 * abs (11 - c) + d = -1) : a + c = 12 := by
  -- Definitions for conditions
  -- h1: intersection at (1, 7) for first graph
  -- h2: intersection at (1, 7) for second graph
  -- h3: intersection at (11, -1) for first graph
  -- h4: intersection at (11, -1) for second graph
  sorry

end sum_of_a_c_l182_182603


namespace sqrt_sqrt_of_81_eq_pm3_and_cube_root_self_l182_182991

theorem sqrt_sqrt_of_81_eq_pm3_and_cube_root_self (x : ℝ) : 
  (∃ y : ℝ, y^2 = 81 ∧ (x^2 = y → x = 3 ∨ x = -3)) ∧ (∀ z : ℝ, z^3 = z → (z = 1 ∨ z = -1 ∨ z = 0)) := by
  sorry

end sqrt_sqrt_of_81_eq_pm3_and_cube_root_self_l182_182991


namespace decimal_representation_of_fraction_l182_182260

theorem decimal_representation_of_fraction :
  (3 / 40 : ℝ) = 0.075 :=
sorry

end decimal_representation_of_fraction_l182_182260


namespace typists_retype_time_l182_182535

theorem typists_retype_time
  (x y : ℕ)
  (h1 : (x / 2) + (y / 2) = 25)
  (h2 : 1 / x + 1 / y = 1 / 12) :
  (x = 20 ∧ y = 30) ∨ (x = 30 ∧ y = 20) :=
by
  sorry

end typists_retype_time_l182_182535


namespace ramesh_paid_price_l182_182716

theorem ramesh_paid_price {P : ℝ} (h1 : P = 18880 / 1.18) : 
  (0.80 * P + 125 + 250) = 13175 :=
by sorry

end ramesh_paid_price_l182_182716


namespace complement_of_A_in_U_l182_182117

namespace SetTheory

def U : Set ℤ := {-1, 0, 1, 2}
def A : Set ℤ := {-1, 1, 2}

theorem complement_of_A_in_U :
  (U \ A) = {0} := by
  sorry

end SetTheory

end complement_of_A_in_U_l182_182117


namespace asymptotes_of_hyperbola_eq_m_l182_182163

theorem asymptotes_of_hyperbola_eq_m :
  ∀ (m : ℝ), (∀ (x y : ℝ), (x^2 / 16 - y^2 / 25 = 1) → (y = m * x ∨ y = -m * x)) → m = 5 / 4 :=
by 
  sorry

end asymptotes_of_hyperbola_eq_m_l182_182163


namespace proportion_terms_l182_182611

theorem proportion_terms (x v y z : ℤ) (a b c : ℤ)
  (h1 : x + v = y + z + a)
  (h2 : x^2 + v^2 = y^2 + z^2 + b)
  (h3 : x^4 + v^4 = y^4 + z^4 + c)
  (ha : a = 7) (hb : b = 21) (hc : c = 2625) :
  (x = -3 ∧ v = 8 ∧ y = -6 ∧ z = 4) :=
by
  sorry

end proportion_terms_l182_182611


namespace Youseff_time_difference_l182_182247

theorem Youseff_time_difference 
  (blocks : ℕ)
  (walk_time_per_block : ℕ) 
  (bike_time_per_block_sec : ℕ) 
  (sec_per_min : ℕ)
  (h_blocks : blocks = 12) 
  (h_walk_time_per_block : walk_time_per_block = 1) 
  (h_bike_time_per_block_sec : bike_time_per_block_sec = 20) 
  (h_sec_per_min : sec_per_min = 60) : 
  (blocks * walk_time_per_block) - ((blocks * bike_time_per_block_sec) / sec_per_min) = 8 :=
by 
  sorry

end Youseff_time_difference_l182_182247


namespace prove_ratio_l182_182728

noncomputable def box_dimensions : ℝ × ℝ × ℝ := (2, 3, 5)
noncomputable def d := (2 * 3 * 5 : ℝ)
noncomputable def a := ((4 * Real.pi) / 3 : ℝ)
noncomputable def b := (10 * Real.pi : ℝ)
noncomputable def c := (62 : ℝ)

theorem prove_ratio :
  (b * c) / (a * d) = (15.5 : ℝ) :=
by
  unfold a b c d
  sorry

end prove_ratio_l182_182728


namespace sum_of_squares_l182_182935

theorem sum_of_squares (a b c : ℝ) (h1 : a + b + c = 23) (h2 : a * b + b * c + a * c = 131) :
  a^2 + b^2 + c^2 = 267 :=
by
  sorry

end sum_of_squares_l182_182935


namespace price_per_piece_l182_182200

variable (y : ℝ)

theorem price_per_piece (h : (20 + y - 12) * (240 - 40 * y) = 1980) :
  20 + y = 21 ∨ 20 + y = 23 :=
sorry

end price_per_piece_l182_182200


namespace cyclist_wait_20_minutes_l182_182364

noncomputable def cyclist_wait_time 
  (hiker_speed : ℝ) (cyclist_speed : ℝ) (time_passed_minutes : ℝ) : ℝ :=
  let time_passed_hours := time_passed_minutes / 60
  let distance := cyclist_speed * time_passed_hours
  let hiker_catch_up_time := distance / hiker_speed
  hiker_catch_up_time * 60

theorem cyclist_wait_20_minutes :
  cyclist_wait_time 5 20 5 = 20 :=
by
  -- Definitions according to given conditions
  let hiker_speed := 5 -- miles per hour
  let cyclist_speed := 20 -- miles per hour
  let time_passed_minutes := 5
  -- Required result
  let result_needed := 20
  -- Using the cyclist_wait_time function
  show cyclist_wait_time hiker_speed cyclist_speed time_passed_minutes = result_needed
  sorry

end cyclist_wait_20_minutes_l182_182364


namespace find_eccentricity_l182_182844

-- Define the hyperbola structure
structure Hyperbola where
  a : ℝ
  b : ℝ
  (a_pos : 0 < a)
  (b_pos : 0 < b)

-- Define the point P and focus F₁ F₂ relationship
structure PointsRelation (C : Hyperbola) where
  P : ℝ × ℝ
  F1 : ℝ × ℝ
  F2 : ℝ × ℝ
  (distance_condition : dist P F1 = 3 * dist P F2)
  (dot_product_condition : (P.1 - F1.1) * (P.1 - F2.1) + (P.2 - F1.2) * (P.2 - F2.2) = C.a^2)

noncomputable def eccentricity (C : Hyperbola) (rel : PointsRelation C) : ℝ :=
  Real.sqrt (1 + (C.b ^ 2) / (C.a ^ 2))

theorem find_eccentricity (C : Hyperbola) (rel : PointsRelation C) : eccentricity C rel = Real.sqrt 2 := by
  sorry

end find_eccentricity_l182_182844


namespace weight_difference_l182_182190

theorem weight_difference (brown black white grey : ℕ) 
  (h_brown : brown = 4)
  (h_white : white = 2 * brown)
  (h_grey : grey = black - 2)
  (avg_weight : (brown + black + white + grey) / 4 = 5): 
  (black - brown) = 1 := by
  sorry

end weight_difference_l182_182190


namespace intersection_A_B_l182_182486

def A (y : ℝ) : Prop := ∃ x : ℝ, y = -x^2 + 2*x - 1
def B (y : ℝ) : Prop := ∃ x : ℝ, y = 2*x + 1

theorem intersection_A_B :
  {y : ℝ | A y} ∩ {y : ℝ | B y} = {y : ℝ | y ≤ 0} :=
sorry

end intersection_A_B_l182_182486


namespace sum_of_digits_l182_182325

def original_sum := 943587 + 329430
def provided_sum := 1412017
def correct_sum_after_change (d e : ℕ) : ℕ := 
  let new_first := if d = 3 then 944587 else 943587
  let new_second := if d = 3 then 429430 else 329430
  new_first + new_second

theorem sum_of_digits (d e : ℕ) : d = 3 ∧ e = 4 → d + e = 7 :=
by
  intros
  exact sorry

end sum_of_digits_l182_182325


namespace problem_1_problem_2_problem_3_problem_4_l182_182455

-- Problem 1
theorem problem_1 (x y : ℝ) : 
  -4 * x^2 * y * (x * y - 5 * y^2 - 1) = -4 * x^3 * y^2 + 20 * x^2 * y^3 + 4 * x^2 * y :=
by
  sorry

-- Problem 2
theorem problem_2 (a : ℝ) :
  (-3 * a)^2 - (2 * a + 1) * (a - 2) = 7 * a^2 + 3 * a + 2 :=
by
  sorry

-- Problem 3
theorem problem_3 (x y : ℝ) :
  (-2 * x - 3 * y) * (3 * y - 2 * x) - (2 * x - 3 * y)^2 = 12 * x * y - 18 * y^2 :=
by
  sorry

-- Problem 4
theorem problem_4 : 2010^2 - 2011 * 2009 = 1 :=
by
  sorry

end problem_1_problem_2_problem_3_problem_4_l182_182455


namespace full_batches_needed_l182_182048

def students : Nat := 150
def cookies_per_student : Nat := 3
def cookies_per_batch : Nat := 20
def attendance_rate : Rat := 0.70

theorem full_batches_needed : 
  let attendees := (students : Rat) * attendance_rate
  let total_cookies_needed := attendees * (cookies_per_student : Rat)
  let batches_needed := total_cookies_needed / (cookies_per_batch : Rat)
  batches_needed.ceil = 16 :=
by
  sorry

end full_batches_needed_l182_182048


namespace single_elimination_games_needed_l182_182637

theorem single_elimination_games_needed (n : ℕ) (n_pos : n > 0) :
  (number_of_games_needed : ℕ) = n - 1 :=
by
  sorry

end single_elimination_games_needed_l182_182637


namespace cookies_left_l182_182307

theorem cookies_left (total_cookies : ℕ) (fraction_given : ℚ) (given_cookies : ℕ) (remaining_cookies : ℕ) 
  (h1 : total_cookies = 20)
  (h2 : fraction_given = 2/5)
  (h3 : given_cookies = fraction_given * total_cookies)
  (h4 : remaining_cookies = total_cookies - given_cookies) :
  remaining_cookies = 12 :=
by
  sorry

end cookies_left_l182_182307


namespace cubes_with_all_three_faces_l182_182218

theorem cubes_with_all_three_faces (total_cubes red_cubes blue_cubes green_cubes: ℕ) 
  (h_total: total_cubes = 100)
  (h_red: red_cubes = 80)
  (h_blue: blue_cubes = 85)
  (h_green: green_cubes = 75) :
  40 ≤ total_cubes - ((total_cubes - red_cubes) + (total_cubes - blue_cubes) + (total_cubes - green_cubes)) ∧ (total_cubes - ((total_cubes - red_cubes) + (total_cubes - blue_cubes) + (total_cubes - green_cubes))) ≤ 75 :=
by {
  sorry
}

end cubes_with_all_three_faces_l182_182218


namespace simplify_sqrt_l182_182464

theorem simplify_sqrt (a b : ℝ) (hb : b > 0) : 
  Real.sqrt (20 * a^3 * b^2) = 2 * a * b * Real.sqrt (5 * a) :=
by
  sorry

end simplify_sqrt_l182_182464


namespace largest_unorderable_dumplings_l182_182227

theorem largest_unorderable_dumplings : 
  ∀ (a b c : ℕ), 43 ≠ 6 * a + 9 * b + 20 * c :=
by sorry

end largest_unorderable_dumplings_l182_182227


namespace avg_percentage_students_l182_182161

-- Define the function that calculates the average percentage of all students
def average_percent (n1 n2 : ℕ) (p1 p2 : ℕ) : ℕ :=
  (n1 * p1 + n2 * p2) / (n1 + n2)

-- Define the properties of the numbers of students and their respective percentages
def students_avg : Prop :=
  average_percent 15 10 70 90 = 78

-- The main theorem: Prove that given the conditions, the average percentage is 78%
theorem avg_percentage_students : students_avg :=
  by
    -- The proof will be provided here.
    sorry

end avg_percentage_students_l182_182161


namespace verify_ages_l182_182282

noncomputable def correct_ages (S M D W : ℝ) : Prop :=
  (M = S + 29) ∧
  (M + 2 = 2 * (S + 2)) ∧
  (D = S - 3.5) ∧
  (W = 1.5 * D) ∧
  (S = 27) ∧
  (M = 56) ∧
  (D = 23.5) ∧
  (W = 35.25)

theorem verify_ages : ∃ (S M D W : ℝ), correct_ages S M D W :=
by
  sorry

end verify_ages_l182_182282


namespace binomial_expression_value_l182_182360

theorem binomial_expression_value :
  (Nat.choose 1 2023 * 3^2023) / Nat.choose 4046 2023 = 0 := by
  sorry

end binomial_expression_value_l182_182360


namespace tickets_difference_l182_182589

def number_of_tickets_for_toys := 31
def number_of_tickets_for_clothes := 14

theorem tickets_difference : number_of_tickets_for_toys - number_of_tickets_for_clothes = 17 := by
  sorry

end tickets_difference_l182_182589


namespace general_form_of_line_l182_182209

theorem general_form_of_line (x y : ℝ) 
  (passes_through_A : ∃ y, 2 = y)          -- Condition 1: passes through A(-2, 2)
  (same_y_intercept : ∃ y, 6 = y)          -- Condition 2: same y-intercept as y = x + 6
  : 2 * x - y + 6 = 0 := 
sorry

end general_form_of_line_l182_182209


namespace typhoon_tree_survival_l182_182657

def planted_trees : Nat := 150
def died_trees : Nat := 92
def slightly_damaged_trees : Nat := 15

def total_trees_affected : Nat := died_trees + slightly_damaged_trees
def trees_survived_without_damages : Nat := planted_trees - total_trees_affected
def more_died_than_survived : Nat := died_trees - trees_survived_without_damages

theorem typhoon_tree_survival :
  more_died_than_survived = 49 :=
by
  -- Define the necessary computations and assertions
  let total_trees_affected := 92 + 15
  let trees_survived_without_damages := 150 - total_trees_affected
  let more_died_than_survived := 92 - trees_survived_without_damages
  -- Prove the statement
  have : total_trees_affected = 107 := rfl
  have : trees_survived_without_damages = 43 := rfl
  have : more_died_than_survived = 49 := rfl
  exact this

end typhoon_tree_survival_l182_182657


namespace n_mod_5_division_of_grid_l182_182215

theorem n_mod_5_division_of_grid (n : ℕ) :
  (∃ m : ℕ, n^2 = 4 + 5 * m) ↔ n % 5 = 2 :=
by
  sorry

end n_mod_5_division_of_grid_l182_182215


namespace angelina_speed_l182_182877

theorem angelina_speed (v : ℝ) (h₁ : ∀ t : ℝ, t = 100 / v) (h₂ : ∀ t : ℝ, t = 180 / (2 * v)) 
  (h₃ : ∀ d t : ℝ, 100 / v - 40 = 180 / (2 * v)) : 
  2 * v = 1 / 2 :=
by
  sorry

end angelina_speed_l182_182877


namespace range_of_a_circle_C_intersects_circle_D_l182_182064

/-- Definitions of circles C and D --/
def circle_C_eq (x y : ℝ) := (x - 1)^2 + (y - 1)^2 = 1
def circle_D_eq (x y m : ℝ) := x^2 + y^2 - 2 * m * x = 0

/-- Condition for the line intersecting Circle C --/
def line_intersects_circle_C (a : ℝ) := (∃ x y : ℝ, circle_C_eq x y ∧ (x + y = a))

/-- Proof of range for a --/
theorem range_of_a (a : ℝ) : line_intersects_circle_C a → (2 - Real.sqrt 2 ≤ a ∧ a ≤ 2 + Real.sqrt 2) :=
sorry

/-- Proposition for point A lying on circle C and satisfying the inequality --/
def point_A_on_circle_C_and_inequality (m : ℝ) (x y : ℝ) :=
  circle_C_eq x y ∧ x^2 + y^2 - (m + Real.sqrt 2 / 2) * x - (m + Real.sqrt 2 / 2) * y ≤ 0

/-- Proof that Circle C intersects Circle D --/
theorem circle_C_intersects_circle_D (m : ℝ) (a : ℝ) : 
  (∀ (x y : ℝ), point_A_on_circle_C_and_inequality m x y) →
  (1 ≤ m ∧
   ∃ (x y : ℝ), (circle_D_eq x y m ∧ (Real.sqrt ((m - 1)^2 + 1) < m + 1 ∧ Real.sqrt ((m - 1)^2 + 1) > m - 1))) :=
sorry

end range_of_a_circle_C_intersects_circle_D_l182_182064


namespace exists_four_numbers_product_fourth_power_l182_182974

theorem exists_four_numbers_product_fourth_power :
  ∃ (numbers : Fin 81 → ℕ),
    (∀ i, ∃ a b c : ℕ, numbers i = 2^a * 3^b * 5^c) ∧
    ∃ (i j k l : Fin 81), i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i ∧
    ∃ m : ℕ, m^4 = numbers i * numbers j * numbers k * numbers l :=
by
  sorry

end exists_four_numbers_product_fourth_power_l182_182974


namespace correct_calculation_result_l182_182417

theorem correct_calculation_result :
  ∃ x : ℕ, 6 * x = 42 ∧ 3 * x = 21 :=
by
  sorry

end correct_calculation_result_l182_182417


namespace Ponchik_week_day_l182_182025

theorem Ponchik_week_day (n s : ℕ) (h1 : s = 20) (h2 : s * (4 * n + 1) = 1360) : n = 4 :=
by
  sorry

end Ponchik_week_day_l182_182025


namespace number_of_cases_for_Ds_hearts_l182_182705

theorem number_of_cases_for_Ds_hearts (hA : 5 ≤ 13) (hB : 4 ≤ 13) (dist : 52 % 4 = 0) : 
  ∃ n, n = 5 ∧ 0 ≤ n ∧ n ≤ 13 := sorry

end number_of_cases_for_Ds_hearts_l182_182705


namespace m_divides_product_iff_composite_ne_4_l182_182845

theorem m_divides_product_iff_composite_ne_4 (m : ℕ) : 
  (m ∣ Nat.factorial (m - 1)) ↔ 
  (∃ a b : ℕ, a ≠ b ∧ 1 < a ∧ 1 < b ∧ m = a * b ∧ m ≠ 4) := 
sorry

end m_divides_product_iff_composite_ne_4_l182_182845


namespace math_problem_l182_182139

open Real -- Open the real number namespace

theorem math_problem (a b : ℝ) (n : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 1 / a + 1 / b = 1) : 
  (a + b)^n - a^n - b^n ≥ 2^(2 * n) - 2^(n + 1) :=
by
  sorry

end math_problem_l182_182139


namespace f_diff_l182_182238

def f (n : ℕ) : ℚ :=
  (Finset.range (3 * n)).sum (λ k => (1 : ℚ) / (k + 1))

theorem f_diff (n : ℕ) : f (n + 1) - f n = (1 / (3 * n) + 1 / (3 * n + 1) + 1 / (3 * n + 2)) :=
by
  sorry

end f_diff_l182_182238


namespace find_ratio_l182_182640
   
   -- Given Conditions
   variable (S T F : ℝ)
   variable (H1 : 30 + S + T + F = 450)
   variable (H2 : S > 30)
   variable (H3 : T > S)
   variable (H4 : F > T)
   
   -- The goal is to find the ratio S / 30
   theorem find_ratio :
     ∃ r : ℝ, r = S / 30 ↔ false :=
   by
     sorry
   
end find_ratio_l182_182640


namespace at_most_n_diameters_l182_182216

theorem at_most_n_diameters {n : ℕ} (h : n ≥ 3) (points : Fin n → ℝ × ℝ) (d : ℝ) 
  (hd : ∀ i j, dist (points i) (points j) ≤ d) :
  ∃ (diameters : Fin n → Fin n), 
    (∀ i, dist (points i) (points (diameters i)) = d) ∧
    (∀ i j, (dist (points i) (points j) = d) → 
      (∃ k, k = i ∨ k = j → diameters k = if k = i then j else i)) :=
sorry

end at_most_n_diameters_l182_182216


namespace number_of_boxes_l182_182575

def magazines : ℕ := 63
def magazines_per_box : ℕ := 9

theorem number_of_boxes : magazines / magazines_per_box = 7 :=
by 
  sorry

end number_of_boxes_l182_182575


namespace final_weight_is_correct_l182_182231

-- Define the various weights after each week
def initial_weight : ℝ := 180
def first_week_removed : ℝ := 0.28 * initial_weight
def first_week_remaining : ℝ := initial_weight - first_week_removed
def second_week_removed : ℝ := 0.18 * first_week_remaining
def second_week_remaining : ℝ := first_week_remaining - second_week_removed
def third_week_removed : ℝ := 0.20 * second_week_remaining
def final_weight : ℝ := second_week_remaining - third_week_removed

-- State the theorem to prove the final weight equals 85.0176 kg
theorem final_weight_is_correct : final_weight = 85.0176 := 
by 
  sorry

end final_weight_is_correct_l182_182231


namespace min_value_of_expression_ge_9_l182_182477

theorem min_value_of_expression_ge_9 
    (x : ℝ)
    (h1 : -2 < x ∧ x < -1)
    (m n : ℝ)
    (a b : ℝ)
    (ha : a = -2)
    (hb : b = -1)
    (h2 : mn > 0)
    (h3 : m * a + n * b + 1 = 0) :
    (2 / m) + (1 / n) ≥ 9 := by
  sorry

end min_value_of_expression_ge_9_l182_182477


namespace geometric_sequence_product_l182_182975

variable {a1 a2 a3 a4 a5 a6 : ℝ}
variable (r : ℝ)
variable (seq : ℕ → ℝ)

-- Conditions defining the terms of a geometric sequence
def is_geometric_sequence (seq : ℕ → ℝ) (a1 r : ℝ) : Prop :=
  ∀ n : ℕ, seq (n + 1) = seq n * r

-- Given condition: a_3 * a_4 = 5
def given_condition (seq : ℕ → ℝ) := (seq 2 * seq 3 = 5)

-- Proving the required question: a_1 * a_2 * a_5 * a_6 = 5
theorem geometric_sequence_product
  (h_geom : is_geometric_sequence seq a1 r)
  (h_given : given_condition seq) :
  seq 0 * seq 1 * seq 4 * seq 5 = 5 :=
sorry

end geometric_sequence_product_l182_182975


namespace sum_roots_l182_182645

theorem sum_roots :
  (∀ (x : ℂ), (3 * x^3 - 2 * x^2 + 4 * x - 15 = 0) → 
              x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
  (∀ (x : ℂ), (4 * x^3 - 16 * x^2 - 28 * x + 35 = 0) → 
              x = y₁ ∨ x = y₂ ∨ x = y₃) →
  (x₁ + x₂ + x₃ + y₁ + y₂ + y₃ = 14 / 3) :=
by
  sorry

end sum_roots_l182_182645


namespace permutation_6_4_l182_182020

theorem permutation_6_4 : (Nat.factorial 6) / (Nat.factorial (6 - 4)) = 360 := by
  sorry

end permutation_6_4_l182_182020


namespace maximize_volume_l182_182764

-- Define the problem-specific constants
def bar_length : ℝ := 0.18
def length_to_width_ratio : ℝ := 2

-- Function to define volume of the rectangle frame
def volume (length width height : ℝ) : ℝ := length * width * height

theorem maximize_volume :
  ∃ (length width height : ℝ), 
  (length / width = length_to_width_ratio) ∧ 
  (2 * (length + width) = bar_length) ∧ 
  ((length = 2) ∧ (height = 1.5)) :=
sorry

end maximize_volume_l182_182764


namespace circle_line_no_intersection_l182_182769

theorem circle_line_no_intersection (b : ℝ) :
  (∀ x y : ℝ, ¬ (x^2 + y^2 = 2 ∧ y = x + b)) ↔ (b > 2 ∨ b < -2) :=
by sorry

end circle_line_no_intersection_l182_182769


namespace total_pairs_is_11_l182_182544

-- Definitions for the conditions
def soft_lens_price : ℕ := 150
def hard_lens_price : ℕ := 85
def total_sales_last_week : ℕ := 1455

-- Variables
variables (H S : ℕ)

-- Condition that she sold 5 more pairs of soft lenses than hard lenses
def sold_more_soft : Prop := S = H + 5

-- Equation for total sales
def total_sales_eq : Prop := (hard_lens_price * H) + (soft_lens_price * S) = total_sales_last_week

-- Total number of pairs of contact lenses sold
def total_pairs_sold : ℕ := H + S

-- The theorem to prove
theorem total_pairs_is_11 (H S : ℕ) (h1 : sold_more_soft H S) (h2 : total_sales_eq H S) : total_pairs_sold H S = 11 :=
sorry

end total_pairs_is_11_l182_182544


namespace smallest_integer_solution_l182_182967

theorem smallest_integer_solution (y : ℤ) (h : 7 - 3 * y < 25) : y ≥ -5 :=
by {
  sorry
}

end smallest_integer_solution_l182_182967


namespace cos_pi_zero_l182_182354

theorem cos_pi_zero : ∃ f : ℝ → ℝ, (∀ x, f x = (Real.cos x) ^ 2 + Real.cos x) ∧ f Real.pi = 0 := by
  sorry

end cos_pi_zero_l182_182354


namespace max_units_of_material_A_l182_182531

theorem max_units_of_material_A (x y z : ℕ) 
    (h1 : 3 * x + 5 * y + 7 * z = 62)
    (h2 : 2 * x + 4 * y + 6 * z = 50) : x ≤ 5 :=
by
    sorry 

end max_units_of_material_A_l182_182531


namespace find_parabola_coeffs_l182_182610

def parabola_vertex_form (a b c : ℝ) : Prop :=
  ∃ k:ℝ, k = c - b^2 / (4*a) ∧ k = 3

def parabola_through_point (a b c : ℝ) : Prop :=
  ∃ x : ℝ, ∃ y : ℝ, x = 0 ∧ y = 1 ∧  y = a * x^2 + b * x + c

theorem find_parabola_coeffs :
  ∃ a b c : ℝ, parabola_vertex_form a b c ∧ parabola_through_point a b c ∧
  a = -1/2 ∧ b = 2 ∧ c = 1 :=
by
  sorry

end find_parabola_coeffs_l182_182610


namespace find_slope_of_q_l182_182448

theorem find_slope_of_q (j : ℝ) : 
  (∀ x y : ℝ, y = 2 * x + 3 → y = j * x + 1 → x = 1 → y = 5) → j = 4 := 
by
  intro h
  sorry

end find_slope_of_q_l182_182448


namespace smallest_h_divisible_by_primes_l182_182883

theorem smallest_h_divisible_by_primes :
  ∃ h k : ℕ, (∀ p q r : ℕ, Prime p ∧ Prime q ∧ Prime r ∧ p > 8 ∧ q > 11 ∧ r > 24 → (h + k) % (p * q * r) = 0 ∧ h = 1) :=
by
  sorry

end smallest_h_divisible_by_primes_l182_182883


namespace jed_change_l182_182058

theorem jed_change :
  ∀ (num_games : ℕ) (cost_per_game : ℕ) (payment : ℕ) (bill_value : ℕ),
  num_games = 6 →
  cost_per_game = 15 →
  payment = 100 →
  bill_value = 5 →
  (payment - num_games * cost_per_game) / bill_value = 2 :=
by
  intros num_games cost_per_game payment bill_value
  sorry

end jed_change_l182_182058


namespace annie_start_crayons_l182_182019

def start_crayons (end_crayons : ℕ) (added_crayons : ℕ) : ℕ := end_crayons - added_crayons

theorem annie_start_crayons (added_crayons end_crayons : ℕ) (h1 : added_crayons = 36) (h2 : end_crayons = 40) :
  start_crayons end_crayons added_crayons = 4 :=
by
  rw [h1, h2]
  exact Nat.sub_eq_of_eq_add sorry  -- skips the detailed proof

end annie_start_crayons_l182_182019


namespace max_height_of_table_l182_182732

theorem max_height_of_table (BC CA AB : ℕ) (h : ℝ) :
  BC = 24 →
  CA = 28 →
  AB = 32 →
  h ≤ (49 * Real.sqrt 60) / 19 :=
by
  intros
  sorry

end max_height_of_table_l182_182732


namespace remainder_approx_l182_182002

def x : ℝ := 74.99999999999716 * 96
def y : ℝ := 74.99999999999716
def quotient : ℝ := 96
def expected_remainder : ℝ := 0.4096

theorem remainder_approx (x y : ℝ) (quotient : ℝ) (h1 : y = 74.99999999999716)
  (h2 : quotient = 96) (h3 : x = y * quotient) :
  x - y * quotient = expected_remainder :=
by
  sorry

end remainder_approx_l182_182002


namespace total_treats_is_237_l182_182942

def num_children : ℕ := 3
def hours_out : ℕ := 4
def houses_visited (hour : ℕ) : ℕ :=
  match hour with
  | 1 => 4
  | 2 => 6
  | 3 => 5
  | 4 => 7
  | _ => 0

def treats_per_kid_per_house (hour : ℕ) : ℕ :=
  match hour with
  | 1 => 3
  | 3 => 3
  | 2 => 4
  | 4 => 4
  | _ => 0

def total_treats : ℕ :=
  (houses_visited 1 * treats_per_kid_per_house 1 * num_children) + 
  (houses_visited 2 * treats_per_kid_per_house 2 * num_children) +
  (houses_visited 3 * treats_per_kid_per_house 3 * num_children) +
  (houses_visited 4 * treats_per_kid_per_house 4 * num_children)

theorem total_treats_is_237 : total_treats = 237 :=
by
  -- Placeholder for the proof
  sorry

end total_treats_is_237_l182_182942


namespace sqrt_fraction_evaluation_l182_182756

theorem sqrt_fraction_evaluation :
  (Real.sqrt ((2 / 25) + (1 / 49) - (1 / 100)) = 3 / 10) :=
by sorry

end sqrt_fraction_evaluation_l182_182756


namespace adults_not_wearing_blue_is_10_l182_182793

section JohnsonFamilyReunion

-- Define the number of children
def children : ℕ := 45

-- Define the ratio between adults and children
def adults : ℕ := children / 3

-- Define the ratio of adults who wore blue
def adults_wearing_blue : ℕ := adults / 3

-- Define the number of adults who did not wear blue
def adults_not_wearing_blue : ℕ := adults - adults_wearing_blue

-- Theorem stating the number of adults who did not wear blue
theorem adults_not_wearing_blue_is_10 : adults_not_wearing_blue = 10 :=
by
  -- This is a placeholder for the actual proof
  sorry

end JohnsonFamilyReunion

end adults_not_wearing_blue_is_10_l182_182793


namespace min_quadratic_expression_l182_182702

theorem min_quadratic_expression:
  ∀ x : ℝ, x = 3 → (x^2 - 6 * x + 5 = -4) :=
by
  sorry

end min_quadratic_expression_l182_182702


namespace sequence_sum_l182_182278

theorem sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) 
  (H_n_def : H_n = (a 1 + (2:ℕ) * a 2 + (2:ℕ) ^ (n - 1) * a n) / n)
  (H_n_val : H_n = 2^n) :
  S n = n * (n + 3) / 2 :=
by
  sorry

end sequence_sum_l182_182278


namespace FDI_in_rural_AndhraPradesh_l182_182573

-- Definitions from conditions
def total_FDI : ℝ := 300 -- Total FDI calculated
def FDI_Gujarat : ℝ := 0.30 * total_FDI
def FDI_Gujarat_Urban : ℝ := 0.80 * FDI_Gujarat
def FDI_AndhraPradesh : ℝ := 0.20 * total_FDI
def FDI_AndhraPradesh_Rural : ℝ := 0.50 * FDI_AndhraPradesh 

-- Given the conditions, prove the size of FDI in rural Andhra Pradesh is 30 million
theorem FDI_in_rural_AndhraPradesh :
  FDI_Gujarat_Urban = 72 → FDI_AndhraPradesh_Rural = 30 :=
by
  sorry

end FDI_in_rural_AndhraPradesh_l182_182573


namespace boiling_point_of_water_l182_182780

theorem boiling_point_of_water :
  (boiling_point_F : ℝ) = 212 →
  (boiling_point_C : ℝ) = (5 / 9) * (boiling_point_F - 32) →
  boiling_point_C = 100 :=
by
  intro h1 h2
  sorry

end boiling_point_of_water_l182_182780


namespace simplify_and_evaluate_expr_l182_182616

theorem simplify_and_evaluate_expr (x y : ℚ) (h1 : x = -3/8) (h2 : y = 4) :
  (x - 2 * y) ^ 2 + (x - 2 * y) * (x + 2 * y) - 2 * x * (x - y) = 3 :=
by
  sorry

end simplify_and_evaluate_expr_l182_182616


namespace range_of_solutions_l182_182570

open Real

theorem range_of_solutions (b : ℝ) :
  (∀ x : ℝ, 
    (x = -3 → x^2 - b*x - 5 = 13)  ∧
    (x = -2 → x^2 - b*x - 5 = 5)   ∧
    (x = -1 → x^2 - b*x - 5 = -1)  ∧
    (x = 4 → x^2 - b*x - 5 = -1)   ∧
    (x = 5 → x^2 - b*x - 5 = 5)    ∧
    (x = 6 → x^2 - b*x - 5 = 13)) →
  (∀ x : ℝ,
    (x^2 - b*x - 5 = 0 → (-2 < x ∧ x < -1) ∨ (4 < x ∧ x < 5))) :=
by
  intros h x hx
  sorry

end range_of_solutions_l182_182570


namespace function_domain_l182_182541

open Set

noncomputable def domain_of_function : Set ℝ :=
  {x | x ≠ 2}

theorem function_domain :
  domain_of_function = {x : ℝ | x ≠ 2} :=
by sorry

end function_domain_l182_182541


namespace find_a_l182_182586

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem find_a (a : ℝ) (h : binomial_coefficient 4 2 + 4 * a = 10) : a = 1 :=
by
  sorry

end find_a_l182_182586


namespace math_problem_l182_182444

/-- Lean translation of the mathematical problem.
Given \(a, b \in \mathbb{R}\) such that \(a^2 + b^2 = a^2 b^2\) and 
\( |a| \neq 1 \) and \( |b| \neq 1 \), prove that 
\[
\frac{a^7}{(1 - a)^2} - \frac{a^7}{(1 + a)^2} = 
\frac{b^7}{(1 - b)^2} - \frac{b^7}{(1 + b)^2}.
\]
-/
theorem math_problem 
  (a b : ℝ) 
  (h1 : a^2 + b^2 = a^2 * b^2) 
  (h2 : |a| ≠ 1) 
  (h3 : |b| ≠ 1) : 
  (a^7 / (1 - a)^2 - a^7 / (1 + a)^2) = 
  (b^7 / (1 - b)^2 - b^7 / (1 + b)^2) := 
by 
  -- Proof is omitted for this exercise.
  sorry

end math_problem_l182_182444


namespace train_crossing_time_l182_182071

noncomputable def train_speed_kmph : ℕ := 72
noncomputable def platform_length_m : ℕ := 300
noncomputable def crossing_time_platform_s : ℕ := 33
noncomputable def train_speed_mps : ℕ := (train_speed_kmph * 5) / 18

theorem train_crossing_time (L : ℕ) (hL : L + platform_length_m = train_speed_mps * crossing_time_platform_s) :
  L / train_speed_mps = 18 :=
  by
    have : train_speed_mps = 20 := by
      sorry
    have : L = 360 := by
      sorry
    sorry

end train_crossing_time_l182_182071


namespace bella_earrings_l182_182450

theorem bella_earrings (B M R : ℝ) 
  (h1 : B = 0.25 * M) 
  (h2 : M = 2 * R) 
  (h3 : B + M + R = 70) : 
  B = 10 := by 
  sorry

end bella_earrings_l182_182450


namespace solve_equation_l182_182727

-- Definitions based on the conditions
def equation (a b c d : ℕ) : Prop :=
  2^a * 3^b - 5^c * 7^d = 1

def nonnegative_integers (a b c d : ℕ) : Prop := 
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0

-- Proof to show the exact solutions
theorem solve_equation :
  (∃ (a b c d : ℕ), nonnegative_integers a b c d ∧ equation a b c d) ↔ 
  ( (1, 0, 0, 0) = (1, 0, 0, 0) ∨ (3, 0, 0, 1) = (3, 0, 0, 1) ∨ 
    (1, 1, 1, 0) = (1, 1, 1, 0) ∨ (2, 2, 1, 1) = (2, 2, 1, 1) ) := by
  sorry

end solve_equation_l182_182727


namespace baby_polar_bear_playing_hours_l182_182773

-- Define the conditions
def total_hours_in_a_day : ℕ := 24
def total_central_angle : ℕ := 360
def angle_sleeping : ℕ := 130
def angle_eating : ℕ := 110

-- Main theorem statement
theorem baby_polar_bear_playing_hours :
  let angle_playing := total_central_angle - angle_sleeping - angle_eating
  let fraction_playing := angle_playing / total_central_angle
  let hours_playing := fraction_playing * total_hours_in_a_day
  hours_playing = 8 := by
  sorry

end baby_polar_bear_playing_hours_l182_182773


namespace simplify_expression_l182_182819

variables {x y : ℝ}
-- Ensure that x and y are not zero to avoid division by zero errors.
theorem simplify_expression (hx : x ≠ 0) (hy : y ≠ 0) : 
  (6 * x^2 * y - 2 * x * y^2) / (2 * x * y) = 3 * x - y :=
sorry

end simplify_expression_l182_182819


namespace simplify_expression_correct_l182_182074

def simplify_expression (x : ℝ) : Prop :=
  2 * x - 3 * (2 - x) + 4 * (2 + 3 * x) - 5 * (1 - 2 * x) = 27 * x - 3

theorem simplify_expression_correct (x : ℝ) : simplify_expression x :=
by
  sorry

end simplify_expression_correct_l182_182074


namespace initial_black_beads_l182_182765

theorem initial_black_beads (B : ℕ) : 
  let white_beads := 51
  let black_beads_removed := 1 / 6 * B
  let white_beads_removed := 1 / 3 * white_beads
  let total_beads_removed := 32
  white_beads_removed + black_beads_removed = total_beads_removed →
  B = 90 :=
by
  sorry

end initial_black_beads_l182_182765


namespace distance_AF_l182_182489

theorem distance_AF (A B C D E F : ℝ×ℝ)
  (h1 : A = (0, 0))
  (h2 : B = (5, 0))
  (h3 : C = (5, 5))
  (h4 : D = (0, 5))
  (h5 : E = (2.5, 5))
  (h6 : ∃ k : ℝ, F = (k, 2 * k) ∧ dist F C = 5) :
  dist A F = Real.sqrt 5 :=
by
  sorry

end distance_AF_l182_182489


namespace Todd_ate_5_cupcakes_l182_182817

theorem Todd_ate_5_cupcakes (original_cupcakes : ℕ) (packages : ℕ) (cupcakes_per_package : ℕ) (remaining_cupcakes : ℕ) :
  original_cupcakes = 50 ∧ packages = 9 ∧ cupcakes_per_package = 5 ∧ remaining_cupcakes = packages * cupcakes_per_package →
  original_cupcakes - remaining_cupcakes = 5 :=
by
  sorry

end Todd_ate_5_cupcakes_l182_182817


namespace sum_of_first_3m_terms_l182_182842

variable {a : ℕ → ℝ}   -- The arithmetic sequence
variable {S : ℕ → ℝ}   -- The sum of the first n terms of the sequence

def arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ) : Prop :=
  S m = 30 ∧ S (2 * m) = 100 ∧ S (3 * m) = 170

theorem sum_of_first_3m_terms (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ) :
  arithmetic_sequence_sum a S m :=
by
  sorry

end sum_of_first_3m_terms_l182_182842


namespace sum_of_non_domain_elements_l182_182867

theorem sum_of_non_domain_elements :
    let f (x : ℝ) : ℝ := 1 / (1 + 1 / (1 + 1 / (1 + 1 / x)))
    let is_not_in_domain (x : ℝ) := x = 0 ∨ x = -1 ∨ x = -1/2 ∨ x = -2/3
    (0 : ℝ) + (-1) + (-1/2) + (-2/3) = -19/6 :=
by 
  sorry

end sum_of_non_domain_elements_l182_182867


namespace Olivia_spent_25_dollars_l182_182653

theorem Olivia_spent_25_dollars
    (initial_amount : ℕ)
    (final_amount : ℕ)
    (spent_amount : ℕ)
    (h_initial : initial_amount = 54)
    (h_final : final_amount = 29)
    (h_spent : spent_amount = initial_amount - final_amount) :
    spent_amount = 25 := by
  sorry

end Olivia_spent_25_dollars_l182_182653


namespace clear_queue_with_three_windows_l182_182206

def time_to_clear_queue_one_window (a x y : ℕ) : Prop := a / (x - y) = 40

def time_to_clear_queue_two_windows (a x y : ℕ) : Prop := a / (2 * x - y) = 16

theorem clear_queue_with_three_windows (a x y : ℕ) 
  (h1 : time_to_clear_queue_one_window a x y) 
  (h2 : time_to_clear_queue_two_windows a x y) : 
  a / (3 * x - y) = 10 :=
by
  sorry

end clear_queue_with_three_windows_l182_182206


namespace min_value_symmetry_l182_182697

def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem min_value_symmetry (a b c : ℝ) (h_a : a > 0) (h_symm : ∀ x : ℝ, quadratic a b c (2 + x) = quadratic a b c (2 - x)) : 
  quadratic a b c 2 < quadratic a b c 1 ∧ quadratic a b c 1 < quadratic a b c 4 := 
sorry

end min_value_symmetry_l182_182697


namespace find_number_l182_182471

theorem find_number (N : ℝ) (h1 : (3 / 10) * N = 64.8) : N = 216 ∧ (1 / 3) * (1 / 4) * N = 18 := 
by 
  sorry

end find_number_l182_182471


namespace find_a_l182_182068

theorem find_a (a : ℝ) :
  (∃x y : ℝ, x^2 + y^2 + 2 * x - 2 * y + a = 0 ∧ x + y + 4 = 0) →
  ∃c : ℝ, c = 2 ∧ a = -7 :=
by
  -- proof to be filled in
  sorry

end find_a_l182_182068


namespace woman_alone_days_l182_182794

theorem woman_alone_days (M W : ℝ) (h1 : (10 * M + 15 * W) * 5 = 1) (h2 : M * 100 = 1) : W * 150 = 1 :=
by
  sorry

end woman_alone_days_l182_182794


namespace not_divisible_by_q_plus_one_l182_182176

theorem not_divisible_by_q_plus_one (q : ℕ) (hq_odd : q % 2 = 1) (hq_gt_two : q > 2) :
  ¬ (q + 1) ∣ ((q + 1) ^ ((q - 1) / 2) + 2) :=
by
  sorry

end not_divisible_by_q_plus_one_l182_182176


namespace remainder_of_7_pow_12_mod_100_l182_182955

theorem remainder_of_7_pow_12_mod_100 : (7 ^ 12) % 100 = 1 := 
by sorry

end remainder_of_7_pow_12_mod_100_l182_182955


namespace compute_expression_l182_182648

theorem compute_expression :
  18 * (216 / 3 + 36 / 6 + 4 / 9 + 2 + 1 / 18) = 1449 :=
by
  sorry

end compute_expression_l182_182648


namespace round_robin_points_change_l182_182876

theorem round_robin_points_change (n : ℕ) (athletes : Finset ℕ) (tournament1_scores tournament2_scores : ℕ → ℚ) :
  Finset.card athletes = 2 * n →
  (∀ a ∈ athletes, abs (tournament2_scores a - tournament1_scores a) ≥ n) →
  (∀ a ∈ athletes, abs (tournament2_scores a - tournament1_scores a) = n) :=
by
  sorry

end round_robin_points_change_l182_182876


namespace compare_costs_l182_182108

def cost_X (copies: ℕ) : ℝ :=
  if copies >= 40 then
    (copies * 1.25) * 0.95
  else
    copies * 1.25

def cost_Y (copies: ℕ) : ℝ :=
  if copies >= 100 then
    copies * 2.00
  else if copies >= 60 then
    copies * 2.25
  else
    copies * 2.75

def cost_Z (copies: ℕ) : ℝ :=
  if copies >= 50 then
    (copies * 3.00) * 0.90
  else
    copies * 3.00

def cost_W (copies: ℕ) : ℝ :=
  let bulk_groups := copies / 25
  let remainder := copies % 25
  (bulk_groups * 40) + (remainder * 2.00)

theorem compare_costs : 
  cost_X 60 < cost_Y 60 ∧ 
  cost_X 60 < cost_Z 60 ∧ 
  cost_X 60 < cost_W 60 ∧
  cost_Y 60 - cost_X 60 = 63.75 ∧
  cost_Z 60 - cost_X 60 = 90.75 ∧
  cost_W 60 - cost_X 60 = 28.75 :=
  sorry

end compare_costs_l182_182108


namespace find_original_denominator_l182_182460

theorem find_original_denominator (d : ℕ) (h : (3 + 7) / (d + 7) = 1 / 3) : d = 23 :=
sorry

end find_original_denominator_l182_182460


namespace cost_of_three_tshirts_l182_182119

-- Defining the conditions
def saving_per_tshirt : ℝ := 5.50
def full_price_per_tshirt : ℝ := 16.50
def number_of_tshirts : ℕ := 3
def number_of_paid_tshirts : ℕ := 2

-- Statement of the problem
theorem cost_of_three_tshirts :
  (number_of_paid_tshirts * full_price_per_tshirt) = 33 := 
by
  -- Proof steps go here (using sorry as a placeholder)
  sorry

end cost_of_three_tshirts_l182_182119


namespace evaluate_expression_l182_182647

theorem evaluate_expression : 
  3 + Real.sqrt 3 + (1 / (3 + Real.sqrt 3)) + (1 / (Real.sqrt 3 - 3)) = 3 + (2 * Real.sqrt 3 / 3) := 
by
  sorry

end evaluate_expression_l182_182647


namespace all_defective_is_impossible_l182_182142

def total_products : ℕ := 10
def defective_products : ℕ := 2
def selected_products : ℕ := 3

theorem all_defective_is_impossible :
  ∀ (products : Finset ℕ),
  products.card = selected_products →
  ∀ (product_ids : Finset ℕ),
  product_ids.card = defective_products →
  products ⊆ product_ids → False :=
by
  sorry

end all_defective_is_impossible_l182_182142


namespace homothety_transformation_l182_182760

variable {V : Type}
variable [AddCommGroup V] [Module ℝ V]

/-- Definition of a homothety transformation -/
def homothety (S A A' : V) (k : ℝ) : Prop :=
  A' = k • A + (1 - k) • S

theorem homothety_transformation (S A A' : V) (k : ℝ) :
  homothety S A A' k ↔ A' = k • A + (1 - k) • S := 
by
  sorry

end homothety_transformation_l182_182760


namespace math_problem_l182_182323

theorem math_problem : ((-7)^3 / 7^2 - 2^5 + 4^3 - 8) = 81 :=
by
  sorry

end math_problem_l182_182323


namespace add_fractions_l182_182235

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end add_fractions_l182_182235


namespace expression_positive_l182_182106

variable {a b c : ℝ}

theorem expression_positive (h₀ : 0 < a ∧ a < 2) (h₁ : -2 < b ∧ b < 0) : 0 < b + a^2 :=
by
  sorry

end expression_positive_l182_182106


namespace probability_nearest_odd_l182_182393

def is_odd_nearest (a b : ℝ) : Prop := ∃ k : ℤ, 2 * k + 1 = Int.floor ((a - b) / (a + b))

def is_valid (a b : ℝ) : Prop := 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1

noncomputable def probability_odd_nearest : ℝ :=
  let interval_area := 1 -- the area of the unit square [0, 1] x [0, 1]
  let odd_area := 1 / 3 -- as derived from the geometric interpretation in the problem's solution
  odd_area / interval_area

theorem probability_nearest_odd (a b : ℝ) (h : is_valid a b) :
  probability_odd_nearest = 1 / 3 := by
  sorry

end probability_nearest_odd_l182_182393


namespace inverse_of_f_at_2_l182_182056

noncomputable def f (x : ℝ) : ℝ := x^2 - 1

theorem inverse_of_f_at_2 : ∀ x, x ≥ 0 → f x = 2 → x = Real.sqrt 3 :=
by
  intro x hx heq
  sorry

end inverse_of_f_at_2_l182_182056


namespace filled_sandbag_weight_is_correct_l182_182245

-- Define the conditions
def sandbag_weight : ℝ := 250
def fill_percent : ℝ := 0.80
def heavier_factor : ℝ := 1.40

-- Define the intermediate weights
def sand_weight : ℝ := sandbag_weight * fill_percent
def extra_weight : ℝ := sand_weight * (heavier_factor - 1)
def filled_material_weight : ℝ := sand_weight + extra_weight

-- Define the total weight including the empty sandbag
def total_weight : ℝ := sandbag_weight + filled_material_weight

-- Prove the total weight is correct
theorem filled_sandbag_weight_is_correct : total_weight = 530 := 
by sorry

end filled_sandbag_weight_is_correct_l182_182245


namespace sum_of_coefficients_eq_zero_l182_182110

theorem sum_of_coefficients_eq_zero 
  (A B C D E F : ℝ) :
  (∀ x, (1 : ℝ) / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) 
  = A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5)) →
  A + B + C + D + E + F = 0 :=
by sorry

end sum_of_coefficients_eq_zero_l182_182110


namespace max_area_rectangle_l182_182861

/-- Given a rectangle with a perimeter of 40, the rectangle with the maximum area is a square
with sides of length 10. The maximum area is thus 100. -/
theorem max_area_rectangle (a b : ℝ) (h : a + b = 20) : a * b ≤ 100 :=
by
  sorry

end max_area_rectangle_l182_182861


namespace asian_countries_visited_l182_182483

theorem asian_countries_visited (total_countries europe_countries south_america_countries remaining_asian_countries : ℕ)
  (h1 : total_countries = 42)
  (h2 : europe_countries = 20)
  (h3 : south_america_countries = 10)
  (h4 : remaining_asian_countries = (total_countries - (europe_countries + south_america_countries)) / 2) :
  remaining_asian_countries = 6 :=
by sorry

end asian_countries_visited_l182_182483


namespace ratio_john_to_total_cost_l182_182502

noncomputable def cost_first_8_years := 8 * 10000
noncomputable def cost_next_10_years := 10 * 20000
noncomputable def university_tuition := 250000
noncomputable def cost_john_paid := 265000
noncomputable def total_cost := cost_first_8_years + cost_next_10_years + university_tuition

theorem ratio_john_to_total_cost : (cost_john_paid / total_cost : ℚ) = 1 / 2 := by
  sorry

end ratio_john_to_total_cost_l182_182502


namespace paint_per_color_equal_l182_182413

theorem paint_per_color_equal (total_paint : ℕ) (num_colors : ℕ) (paint_per_color : ℕ) : 
  total_paint = 15 ∧ num_colors = 3 → paint_per_color = 5 := by
  sorry

end paint_per_color_equal_l182_182413


namespace smallest_n_l182_182504

theorem smallest_n (n : ℕ) (hn1 : ∃ k, 5 * n = k^4) (hn2: ∃ m, 4 * n = m^3) : n = 2000 :=
sorry

end smallest_n_l182_182504


namespace solution_l182_182397

variable (a : ℕ → ℝ)

noncomputable def pos_sequence (n : ℕ) : Prop :=
  ∀ k : ℕ, k > 0 → a k > 0

noncomputable def recursive_relation (n : ℕ) : Prop :=
  ∀ n : ℕ, (n > 0) → (n+1) * a (n+1)^2 - n * a n^2 + a (n+1) * a n = 0

noncomputable def sequence_condition (n : ℕ) : Prop :=
  a 1 = 1 ∧ pos_sequence a n ∧ recursive_relation a n

theorem solution : ∀ n : ℕ, n > 0 → sequence_condition a n → a n = 1 / n :=
by
  intros n hn h
  sorry

end solution_l182_182397


namespace range_of_a_l182_182682

variable (a : ℝ)

def set_A (a : ℝ) : Set ℝ := {x | abs (x - 2) ≤ a}
def set_B : Set ℝ := {x | x ≤ 1 ∨ x ≥ 4}

lemma disjoint_sets (A B : Set ℝ) : A ∩ B = ∅ :=
  sorry

theorem range_of_a (h : set_A a ∩ set_B = ∅) : a < 1 :=
  by
  sorry

end range_of_a_l182_182682


namespace min_cubes_are_three_l182_182916

/-- 
  A toy construction set consists of cubes, each with one button on one side and socket holes on the other five sides.
  Prove that the minimum number of such cubes required to build a structure where all buttons are hidden, and only the sockets are visible is 3.
--/

def min_cubes_to_hide_buttons (num_cubes : ℕ) : Prop :=
  num_cubes = 3

theorem min_cubes_are_three : ∃ (n : ℕ), (∀ (num_buttons : ℕ), min_cubes_to_hide_buttons num_buttons) :=
by
  use 3
  sorry

end min_cubes_are_three_l182_182916


namespace aluminum_phosphate_molecular_weight_l182_182183

theorem aluminum_phosphate_molecular_weight :
  let Al := 26.98
  let P := 30.97
  let O := 16.00
  (Al + P + 4 * O) = 121.95 :=
by
  let Al := 26.98
  let P := 30.97
  let O := 16.00
  sorry

end aluminum_phosphate_molecular_weight_l182_182183


namespace sum_areas_of_circles_l182_182205

theorem sum_areas_of_circles (r s t : ℝ) 
  (h1 : r + s = 6)
  (h2 : r + t = 8)
  (h3 : s + t = 10) : 
  r^2 * Real.pi + s^2 * Real.pi + t^2 * Real.pi = 56 * Real.pi := 
by 
  sorry

end sum_areas_of_circles_l182_182205


namespace exponential_inequality_l182_182298

variables (x a b : ℝ)

theorem exponential_inequality (h1 : x > 0) (h2 : 1 < b^x) (h3 : b^x < a^x) : 1 < b ∧ b < a :=
by
   sorry

end exponential_inequality_l182_182298


namespace minimum_b_l182_182880

theorem minimum_b (k a b : ℝ) (h1 : 1 < k) (h2 : k < a) (h3 : a < b)
  (h4 : ¬(k + a > b)) (h5 : ¬(1/a + 1/b > 1/k)) :
  2 * k ≤ b :=
by
  sorry

end minimum_b_l182_182880


namespace total_cost_l182_182297

def cost_of_items (x y : ℝ) : Prop :=
  (6 * x + 5 * y = 6.10) ∧ (3 * x + 4 * y = 4.60)

theorem total_cost (x y : ℝ) (h : cost_of_items x y) : 12 * x + 8 * y = 10.16 :=
by
  sorry

end total_cost_l182_182297


namespace intersections_count_l182_182249

theorem intersections_count
  (c : ℕ)  -- crosswalks per intersection
  (l : ℕ)  -- lines per crosswalk
  (t : ℕ)  -- total lines
  (h_c : c = 4)
  (h_l : l = 20)
  (h_t : t = 400) :
  t / (c * l) = 5 :=
  by
    sorry

end intersections_count_l182_182249


namespace fertilizer_percentage_l182_182089

theorem fertilizer_percentage (total_volume : ℝ) (vol_74 : ℝ) (vol_53 : ℝ) (perc_74 : ℝ) (perc_53 : ℝ) (final_perc : ℝ) :
  total_volume = 42 ∧ vol_74 = 20 ∧ vol_53 = total_volume - vol_74 ∧ perc_74 = 0.74 ∧ perc_53 = 0.53 
  → final_perc = ((vol_74 * perc_74 + vol_53 * perc_53) / total_volume) * 100
  → final_perc = 63.0 :=
by
  intros
  sorry

end fertilizer_percentage_l182_182089


namespace no_real_k_for_distinct_roots_l182_182054

theorem no_real_k_for_distinct_roots (k : ℝ) : ¬ ( -8 * k^2 > 0 ) := 
by
  sorry

end no_real_k_for_distinct_roots_l182_182054


namespace fourth_equation_l182_182063

theorem fourth_equation :
  (5 * 6 * 7 * 8) = (2^4) * 1 * 3 * 5 * 7 :=
by
  sorry

end fourth_equation_l182_182063


namespace more_cats_than_dogs_l182_182337

-- Define the number of cats and dogs
def c : ℕ := 23
def d : ℕ := 9

-- The theorem we need to prove
theorem more_cats_than_dogs : c - d = 14 := by
  sorry

end more_cats_than_dogs_l182_182337


namespace smith_oldest_child_age_l182_182934

theorem smith_oldest_child_age
  (avg_age : ℕ)
  (youngest : ℕ)
  (middle : ℕ)
  (oldest : ℕ)
  (h1 : avg_age = 9)
  (h2 : youngest = 6)
  (h3 : middle = 8)
  (h4 : (youngest + middle + oldest) / 3 = avg_age) :
  oldest = 13 :=
by
  sorry

end smith_oldest_child_age_l182_182934


namespace floors_above_l182_182398

theorem floors_above (dennis_floor charlie_floor frank_floor : ℕ)
  (h1 : dennis_floor = 6)
  (h2 : frank_floor = 16)
  (h3 : charlie_floor = frank_floor / 4) :
  dennis_floor - charlie_floor = 2 :=
by
  sorry

end floors_above_l182_182398


namespace box_cookies_count_l182_182212

theorem box_cookies_count (cookies_per_bag : ℕ) (cookies_per_box : ℕ) :
  cookies_per_bag = 7 →
  8 * cookies_per_box = 9 * cookies_per_bag + 33 →
  cookies_per_box = 12 :=
by
  intros h1 h2
  sorry

end box_cookies_count_l182_182212


namespace blue_paint_needed_l182_182941

theorem blue_paint_needed (F B : ℝ) :
  (6/9 * F = 4/5 * (F * 1/3 + B) → B = 1/2 * F) :=
sorry

end blue_paint_needed_l182_182941


namespace problem1_problem2_l182_182250

theorem problem1 : (Real.sqrt 18 - Real.sqrt 8 - Real.sqrt 2) = 0 := 
by sorry

theorem problem2 : (6 * Real.sqrt 2 * Real.sqrt 3 + 3 * Real.sqrt 30 / Real.sqrt 5) = 9 * Real.sqrt 6 := 
by sorry

end problem1_problem2_l182_182250


namespace initial_tabs_count_l182_182818

theorem initial_tabs_count (T : ℕ) (h1 : T > 0)
  (h2 : (3 / 4 : ℚ) * T - (2 / 5 : ℚ) * ((3 / 4 : ℚ) * T) > 0)
  (h3 : (9 / 20 : ℚ) * T - (1 / 2 : ℚ) * ((9 / 20 : ℚ) * T) = 90) :
  T = 400 :=
sorry

end initial_tabs_count_l182_182818


namespace projectile_max_height_l182_182519

def h (t : ℝ) : ℝ := -9 * t^2 + 36 * t + 24

theorem projectile_max_height : ∃ t : ℝ, h t = 60 := 
sorry

end projectile_max_height_l182_182519


namespace car_distance_l182_182909

theorem car_distance (time_am_18 : ℕ) (time_car_48 : ℕ) (h : time_am_18 = time_car_48) : 
  let distance_am_18 := 18
  let distance_car_48 := 48
  let total_distance_am := 675
  let distance_ratio := (distance_am_18 : ℝ) / (distance_car_48 : ℝ)
  let distance_car := (total_distance_am : ℝ) * (distance_car_48 : ℝ) / (distance_am_18 : ℝ)
  distance_car = 1800 :=
by
  sorry

end car_distance_l182_182909


namespace probability_red_nonjoker_then_black_or_joker_l182_182306

theorem probability_red_nonjoker_then_black_or_joker :
  let total_cards := 60
  let red_non_joker := 26
  let black_or_joker := 40
  let total_ways_to_draw_two_cards := total_cards * (total_cards - 1)
  let probability := (red_non_joker * black_or_joker : ℚ) / total_ways_to_draw_two_cards
  probability = 5 / 17 :=
by
  -- Definitions for the conditions
  let total_cards := 60
  let red_non_joker := 26
  let black_or_joker := 40
  let total_ways_to_draw_two_cards := total_cards * (total_cards - 1)
  let probability := (red_non_joker * black_or_joker : ℚ) / total_ways_to_draw_two_cards
  -- Add sorry placeholder for proof
  sorry

end probability_red_nonjoker_then_black_or_joker_l182_182306


namespace alcohol_added_l182_182207

-- Definitions from conditions
def initial_volume : ℝ := 40
def initial_alcohol_concentration : ℝ := 0.05
def initial_alcohol_amount : ℝ := initial_volume * initial_alcohol_concentration
def added_water_volume : ℝ := 3.5
def final_alcohol_concentration : ℝ := 0.17

-- The problem to be proven
theorem alcohol_added :
  ∃ x : ℝ,
    x = (final_alcohol_concentration * (initial_volume + x + added_water_volume) - initial_alcohol_amount) :=
by
  sorry

end alcohol_added_l182_182207


namespace min_value_reciprocal_l182_182177

theorem min_value_reciprocal (m n : ℝ) (hmn_gt : 0 < m * n) (hmn_add : m + n = 2) :
  (∃ x : ℝ, x = (1/m + 1/n) ∧ x = 2) :=
by sorry

end min_value_reciprocal_l182_182177


namespace num_employees_is_143_l182_182555

def b := 143
def is_sol (b : ℕ) := 80 < b ∧ b < 150 ∧ b % 4 = 3 ∧ b % 5 = 3 ∧ b % 7 = 4

theorem num_employees_is_143 : is_sol b :=
by
  -- This is where the proof would be written
  sorry

end num_employees_is_143_l182_182555


namespace zero_in_interval_l182_182618

noncomputable def f (x : ℝ) : ℝ := Real.logb 3 x + 2 * x - 8

theorem zero_in_interval : (f 3 < 0) ∧ (f 4 > 0) → ∃ c, 3 < c ∧ c < 4 ∧ f c = 0 :=
by
  sorry

end zero_in_interval_l182_182618


namespace task1_task2_l182_182244

/-- Given conditions -/
def cost_A : Nat := 30
def cost_B : Nat := 40
def sell_A : Nat := 35
def sell_B : Nat := 50
def max_cost : Nat := 1550
def min_profit : Nat := 365
def total_cars : Nat := 40

/-- Task 1: Prove maximum B-type cars produced if 10 A-type cars are produced -/
theorem task1 (A: Nat) (B: Nat) (hA: A = 10) (hC: cost_A * A + cost_B * B ≤ max_cost) : B ≤ 31 :=
by sorry

/-- Task 2: Prove the possible production plans producing 40 cars meeting profit and cost constraints -/
theorem task2 (A: Nat) (B: Nat) (hTotal: A + B = total_cars)
(hCost: cost_A * A + cost_B * B ≤ max_cost) 
(hProfit: (sell_A - cost_A) * A + (sell_B - cost_B) * B ≥ min_profit) : 
  (A = 5 ∧ B = 35) ∨ (A = 6 ∧ B = 34) ∨ (A = 7 ∧ B = 33) 
∧ (375 ≤ (sell_A - cost_A) * 5 + (sell_B - cost_B) * 35 ∧ 375 ≥ (sell_A - cost_A) * 5 + (sell_B - cost_B) * 35) :=
by sorry

end task1_task2_l182_182244


namespace number_of_men_at_picnic_l182_182015

theorem number_of_men_at_picnic (total persons W M A C : ℕ) (h1 : total = 200) 
  (h2 : M = W + 20) (h3 : A = C + 20) (h4 : A = M + W) : M = 65 :=
by
  -- Proof can be filled in here
  sorry

end number_of_men_at_picnic_l182_182015


namespace value_of_f_f_f_2_l182_182828

def f (x : ℝ) : ℝ := x^3 - 3*x

theorem value_of_f_f_f_2 : f (f (f 2)) = 2 :=
by {
  sorry
}

end value_of_f_f_f_2_l182_182828


namespace total_rainfall_2019_to_2021_l182_182357

theorem total_rainfall_2019_to_2021 :
  let R2019 := 50
  let R2020 := R2019 + 5
  let R2021 := R2020 - 3
  12 * R2019 + 12 * R2020 + 12 * R2021 = 1884 :=
by
  sorry

end total_rainfall_2019_to_2021_l182_182357


namespace files_to_organize_in_afternoon_l182_182892

-- Defining the given conditions.
def initial_files : ℕ := 60
def files_organized_in_the_morning : ℕ := initial_files / 2
def missing_files_in_the_afternoon : ℕ := 15

-- The theorem to prove:
theorem files_to_organize_in_afternoon : 
  files_organized_in_the_morning + missing_files_in_the_afternoon = initial_files / 2 →
  ∃ afternoon_files : ℕ, 
    afternoon_files = (initial_files - files_organized_in_the_morning) - missing_files_in_the_afternoon :=
by
  -- Proof will go here, skipping with sorry for now.
  sorry

end files_to_organize_in_afternoon_l182_182892


namespace solve_quadratic_equation_l182_182852

theorem solve_quadratic_equation :
  ∀ x : ℝ, (10 - x) ^ 2 = 2 * x ^ 2 + 4 * x ↔ x = 3.62 ∨ x = -27.62 := by
  sorry

end solve_quadratic_equation_l182_182852


namespace four_numbers_sum_divisible_by_2016_l182_182169

theorem four_numbers_sum_divisible_by_2016 {x : Fin 65 → ℕ} (h_distinct: Function.Injective x) (h_range: ∀ i, x i ≤ 2016) :
  ∃ a b c d : Fin 65, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ (x a + x b - x c - x d) % 2016 = 0 :=
by
  -- Proof omitted
  sorry

end four_numbers_sum_divisible_by_2016_l182_182169


namespace area_of_gray_region_l182_182253

theorem area_of_gray_region (r : ℝ) (h1 : r * 3 - r = 3) : 
  π * (3 * r) ^ 2 - π * r ^ 2 = 18 * π :=
by
  sorry

end area_of_gray_region_l182_182253


namespace cauliflower_area_l182_182027

theorem cauliflower_area
  (s : ℕ) (a : ℕ) 
  (H1 : s * s / a = 40401)
  (H2 : s * s / a = 40000) :
  a = 1 :=
sorry

end cauliflower_area_l182_182027


namespace days_wages_l182_182454

theorem days_wages (S W_a W_b : ℝ) 
    (h1 : S = 28 * W_b) 
    (h2 : S = 12 * (W_a + W_b)) 
    (h3 : S = 21 * W_a) : 
    true := 
by sorry

end days_wages_l182_182454


namespace quarterback_passes_left_l182_182168

noncomputable def number_of_passes (L : ℕ) : Prop :=
  let R := 2 * L
  let C := L + 2
  L + R + C = 50

theorem quarterback_passes_left : ∃ L, number_of_passes L ∧ L = 12 := by
  sorry

end quarterback_passes_left_l182_182168


namespace negation_of_universal_l182_182699

variable (f : ℝ → ℝ) (m : ℝ)

theorem negation_of_universal :
  (∀ x : ℝ, f x ≥ m) → ¬ (∀ x : ℝ, f x ≥ m) → ∃ x : ℝ, f x < m :=
by
  sorry

end negation_of_universal_l182_182699


namespace largest_fraction_consecutive_primes_l182_182114

theorem largest_fraction_consecutive_primes (p q r s : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hs : Nat.Prime s)
  (h0 : 0 < p) (h1 : p < q) (h2 : q < r) (h3 : r < s)
  (hconsec : p + 2 = q ∧ q + 2 = r ∧ r + 2 = s) :
  (r + s) / (p + q) > max ((p + q) / (r + s)) (max ((p + s) / (q + r)) (max ((q + r) / (p + s)) ((q + s) / (p + r)))) :=
sorry

end largest_fraction_consecutive_primes_l182_182114


namespace absolute_value_expression_l182_182912

theorem absolute_value_expression {x : ℤ} (h : x = 2024) :
  abs (abs (abs x - x) - abs x) = 0 :=
by
  sorry

end absolute_value_expression_l182_182912


namespace find_three_numbers_l182_182998

theorem find_three_numbers :
  ∃ (x1 x2 x3 k1 k2 k3 : ℕ),
  x1 = 2500 * k1 / (3^k1 - 1) ∧
  x2 = 2500 * k2 / (3^k2 - 1) ∧
  x3 = 2500 * k3 / (3^k3 - 1) ∧
  k1 ≠ k2 ∧ k1 ≠ k3 ∧ k2 ≠ k3 ∧
  x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 :=
by
  sorry

end find_three_numbers_l182_182998


namespace negation_exists_to_forall_l182_182144

theorem negation_exists_to_forall (P : ℝ → Prop) (h : ∃ x : ℝ, x^2 + 3 * x + 2 < 0) :
  (¬ (∃ x : ℝ, x^2 + 3 * x + 2 < 0)) ↔ (∀ x : ℝ, x^2 + 3 * x + 2 ≥ 0) := by
sorry

end negation_exists_to_forall_l182_182144


namespace negative_solution_condition_l182_182999

-- Define the system of equations
def system_of_equations (a b c x y : ℝ) :=
  a * x + b * y = c ∧
  b * x + c * y = a ∧
  c * x + a * y = b

-- State the theorem
theorem negative_solution_condition (a b c : ℝ) :
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ system_of_equations a b c x y) ↔ a + b + c = 0 :=
by
  sorry

end negative_solution_condition_l182_182999


namespace max_mean_BC_l182_182626

theorem max_mean_BC (A_n B_n C_n A_total_weight B_total_weight C_total_weight : ℕ)
    (hA_mean : A_total_weight = 45 * A_n)
    (hB_mean : B_total_weight = 55 * B_n)
    (hAB_mean : (A_total_weight + B_total_weight) / (A_n + B_n) = 48)
    (hAC_mean : (A_total_weight + C_total_weight) / (A_n + C_n) = 50) :
    ∃ m : ℤ, m = 66 := by
  sorry

end max_mean_BC_l182_182626


namespace increasing_sequence_nec_but_not_suf_l182_182889

theorem increasing_sequence_nec_but_not_suf (a : ℕ → ℝ) :
  (∀ n, abs (a (n + 1)) > a n) → (∀ n, a (n + 1) > a n) ↔ 
  ∃ (n : ℕ), ¬ (abs (a (n + 1)) > a n) ∧ (a (n + 1) > a n) :=
sorry

end increasing_sequence_nec_but_not_suf_l182_182889


namespace slope_parallel_line_l182_182012

theorem slope_parallel_line (x y : ℝ) (a b c : ℝ) (h : 3 * x - 6 * y = 15) : 
  ∃ m : ℝ, m = 1 / 2 :=
by 
  sorry

end slope_parallel_line_l182_182012


namespace percent_equivalence_l182_182824

theorem percent_equivalence (x : ℝ) : (0.6 * 0.3 * x - 0.1 * x) / x * 100 = 8 := by
  sorry

end percent_equivalence_l182_182824


namespace bill_spots_39_l182_182772

theorem bill_spots_39 (P : ℕ) (h1 : P + (2 * P - 1) = 59) : 2 * P - 1 = 39 :=
by sorry

end bill_spots_39_l182_182772


namespace pepper_left_l182_182543

def initial_pepper : ℝ := 0.25
def used_pepper : ℝ := 0.16
def remaining_pepper : ℝ := 0.09

theorem pepper_left (h1 : initial_pepper = 0.25) (h2 : used_pepper = 0.16) :
  initial_pepper - used_pepper = remaining_pepper :=
by
  sorry

end pepper_left_l182_182543


namespace evaluate_expression_l182_182592

theorem evaluate_expression (a b c : ℝ)
  (h : a / (25 - a) + b / (65 - b) + c / (60 - c) = 7) :
  5 / (25 - a) + 13 / (65 - b) + 12 / (60 - c) = 2 := 
sorry

end evaluate_expression_l182_182592


namespace find_minimum_n_l182_182312

variable {a_1 d : ℝ}
variable {S : ℕ → ℝ}

def is_arithmetic_sequence (a_1 d : ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n / 2 : ℝ) * (2 * a_1 + (n - 1) * d)

def condition1 (a_1 : ℝ) : Prop := a_1 < 0

def condition2 (S : ℕ → ℝ) : Prop := S 7 = S 13

theorem find_minimum_n (a_1 d : ℝ) (S : ℕ → ℝ)
  (h_arith_seq : is_arithmetic_sequence a_1 d S)
  (h_a1_neg : condition1 a_1)
  (h_s7_eq_s13 : condition2 S) :
  ∃ n : ℕ, n = 10 ∧ ∀ m : ℕ, S n ≤ S m := 
sorry

end find_minimum_n_l182_182312


namespace find_value_of_p_l182_182172

theorem find_value_of_p (p q : ℚ) (h1 : p + q = 3 / 4)
    (h2 : 45 * p^8 * q^2 = 120 * p^7 * q^3) : p = 6 / 11 :=
by
    sorry

end find_value_of_p_l182_182172


namespace twenty_five_percent_of_2004_l182_182381

theorem twenty_five_percent_of_2004 : (1 / 4 : ℝ) * 2004 = 501 := by
  sorry

end twenty_five_percent_of_2004_l182_182381


namespace bird_needs_more_twigs_l182_182701

variable (base_twigs : ℕ := 12)
variable (additional_twigs_per_base : ℕ := 6)
variable (fraction_dropped : ℚ := 1/3)

theorem bird_needs_more_twigs (tree_dropped : ℕ) : 
  tree_dropped = (additional_twigs_per_base * base_twigs) * 1/3 →
  (base_twigs * additional_twigs_per_base - tree_dropped) = 48 :=
by
  sorry

end bird_needs_more_twigs_l182_182701


namespace numerator_in_second_fraction_l182_182060

theorem numerator_in_second_fraction (p q x: ℚ) (h1 : p / q = 4 / 5) (h2 : 11 / 7 + x / (2 * q + p) = 2) : x = 6 :=
sorry

end numerator_in_second_fraction_l182_182060


namespace marble_ratio_l182_182995

theorem marble_ratio
  (L_b : ℕ) (J_y : ℕ) (A : ℕ)
  (A_b : ℕ) (A_y : ℕ) (R : ℕ)
  (h1 : L_b = 4)
  (h2 : J_y = 22)
  (h3 : A = 19)
  (h4 : A_y = J_y / 2)
  (h5 : A = A_b + A_y)
  (h6 : A_b = L_b * R) :
  R = 2 := by
  sorry

end marble_ratio_l182_182995


namespace total_guitars_sold_l182_182470

theorem total_guitars_sold (total_revenue : ℕ) (price_electric : ℕ) (price_acoustic : ℕ)
  (num_electric_sold : ℕ) (num_acoustic_sold : ℕ) 
  (h1 : total_revenue = 3611) (h2 : price_electric = 479) 
  (h3 : price_acoustic = 339) (h4 : num_electric_sold = 4) 
  (h5 : num_acoustic_sold * price_acoustic + num_electric_sold * price_electric = total_revenue) :
  num_electric_sold + num_acoustic_sold = 9 :=
sorry

end total_guitars_sold_l182_182470


namespace vertex_of_quadratic_function_l182_182317

theorem vertex_of_quadratic_function :
  ∀ x: ℝ, (2 - (x + 1)^2) = 2 - (x + 1)^2 → (∃ h k : ℝ, (h, k) = (-1, 2) ∧ ∀ x: ℝ, (2 - (x + 1)^2) = k - (x - h)^2) :=
by
  sorry

end vertex_of_quadratic_function_l182_182317


namespace Andy_has_4_more_candies_than_Caleb_l182_182378

-- Define the initial candies each person has
def Billy_initial_candies : ℕ := 6
def Caleb_initial_candies : ℕ := 11
def Andy_initial_candies : ℕ := 9

-- Define the candies bought by the father and their distribution
def father_bought_candies : ℕ := 36
def Billy_received_from_father : ℕ := 8
def Caleb_received_from_father : ℕ := 11

-- Calculate the remaining candies for Andy after distribution
def Andy_received_from_father : ℕ := father_bought_candies - (Billy_received_from_father + Caleb_received_from_father)

-- Calculate the total candies each person has
def Billy_total_candies : ℕ := Billy_initial_candies + Billy_received_from_father
def Caleb_total_candies : ℕ := Caleb_initial_candies + Caleb_received_from_father
def Andy_total_candies : ℕ := Andy_initial_candies + Andy_received_from_father

-- Prove that Andy has 4 more candies than Caleb
theorem Andy_has_4_more_candies_than_Caleb :
  Andy_total_candies = Caleb_total_candies + 4 :=
by {
  -- Skipping the proof
  sorry
}

end Andy_has_4_more_candies_than_Caleb_l182_182378


namespace least_t_geometric_progression_exists_l182_182646

open Real

theorem least_t_geometric_progression_exists :
  ∃ (t : ℝ),
  (∃ (α : ℝ), 0 < α ∧ α < π / 3 ∧
             (arcsin (sin α) = α ∧
              arcsin (sin (3 * α)) = 3 * α ∧
              arcsin (sin (8 * α)) = 8 * α) ∧
              (arcsin (sin (t * α)) = (some_ratio) * (arcsin (sin (8 * α))) )) ∧ 
   0 < t := 
by 
  sorry

end least_t_geometric_progression_exists_l182_182646


namespace remainder_of_sum_of_first_150_numbers_l182_182395

def sum_of_first_n_natural_numbers (n : ℕ) : ℕ := n * (n + 1) / 2

theorem remainder_of_sum_of_first_150_numbers :
  (sum_of_first_n_natural_numbers 150) % 5000 = 1275 :=
by
  sorry

end remainder_of_sum_of_first_150_numbers_l182_182395


namespace problem_statement_l182_182617

open Set

variable (a : ℕ)
variable (A : Set ℕ := {2, 3, 4})
variable (B : Set ℕ := {a + 2, a})

theorem problem_statement (hB : B ⊆ A) : (A \ B) = {3} :=
sorry

end problem_statement_l182_182617


namespace min_AP_l182_182181

noncomputable def A : ℝ × ℝ := (2, 0)
noncomputable def B' : ℝ × ℝ := (8, 6)
def parabola (P' : ℝ × ℝ) : Prop := P'.2^2 = 8 * P'.1

theorem min_AP'_plus_BP' : 
  ∃ P' : ℝ × ℝ, parabola P' ∧ (dist A P' + dist B' P' = 12) := 
sorry

end min_AP_l182_182181


namespace square_cookie_cutters_count_l182_182577

def triangles_sides : ℕ := 6 * 3
def hexagons_sides : ℕ := 2 * 6
def total_sides : ℕ := 46
def sides_from_squares (S : ℕ) : ℕ := S * 4

theorem square_cookie_cutters_count (S : ℕ) :
  triangles_sides + hexagons_sides + sides_from_squares S = total_sides → S = 4 :=
by
  sorry

end square_cookie_cutters_count_l182_182577


namespace max_value_x2_plus_y2_l182_182299

theorem max_value_x2_plus_y2 (x y : ℝ) (h : 5 * x^2 + 4 * y^2 = 10 * x) : 
  x^2 + y^2 ≤ 4 :=
sorry

end max_value_x2_plus_y2_l182_182299


namespace unique_b_for_unique_solution_l182_182510

theorem unique_b_for_unique_solution (c : ℝ) (h₁ : c ≠ 0) :
  (∃ b : ℝ, b > 0 ∧ ∃! x : ℝ, x^2 + (b + (2 / b)) * x + c = 0) →
  c = 2 :=
by
  -- sorry will go here to indicate the proof is to be filled in
  sorry

end unique_b_for_unique_solution_l182_182510


namespace terminal_side_in_third_quadrant_l182_182196

-- Define the conditions
def sin_condition (α : Real) : Prop := Real.sin α < 0
def tan_condition (α : Real) : Prop := Real.tan α > 0

-- State the theorem
theorem terminal_side_in_third_quadrant (α : Real) (h1 : sin_condition α) (h2 : tan_condition α) : α ∈ Set.Ioo (π / 2) π :=
  sorry

end terminal_side_in_third_quadrant_l182_182196


namespace pairs_characterization_l182_182933

noncomputable def valid_pairs (A : ℝ) : Set (ℕ × ℕ) :=
  { p | ∃ x : ℝ, x > 0 ∧ (1 + x) ^ p.1 = (1 + A * x) ^ p.2 }

theorem pairs_characterization (A : ℝ) (hA : A > 1) :
  valid_pairs A = { p | p.2 < p.1 ∧ p.1 < A * p.2 } :=
by
  sorry

end pairs_characterization_l182_182933


namespace sum_first_six_terms_l182_182415

variable (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ)

-- Define the existence of a geometric sequence with given properties
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Given Condition: a_3 = 2a_4 = 2
def cond1 (a : ℕ → ℝ) : Prop :=
  a 3 = 2 ∧ a 4 = 1

-- Define the sum of the first n terms of the sequence
def geometric_sum (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = a 0 * (1 - q ^ n) / (1 - q)

-- We need to prove that under these conditions, S_6 = 63/4
theorem sum_first_six_terms 
  (hq : q = 1 / 2) 
  (ha : is_geometric_sequence a q) 
  (hcond1 : cond1 a) 
  (hS : geometric_sum a q S) : 
  S 6 = 63 / 4 := 
sorry

end sum_first_six_terms_l182_182415


namespace number_of_valid_pairs_l182_182342

-- Definition of the conditions according to step (a)
def perimeter (l w : ℕ) : Prop := 2 * (l + w) = 80
def integer_lengths (l w : ℕ) : Prop := true
def length_greater_than_width (l w : ℕ) : Prop := l > w

-- The mathematical proof problem according to step (c)
theorem number_of_valid_pairs : ∃ n : ℕ, 
  (∀ l w : ℕ, perimeter l w → integer_lengths l w → length_greater_than_width l w → ∃! pair : (ℕ × ℕ), pair = (l, w)) ∧
  n = 19 :=
by 
  sorry

end number_of_valid_pairs_l182_182342


namespace intersection_A_B_l182_182568

-- Define set A
def A : Set Int := { x | x^2 - x - 2 ≤ 0 }

-- Define set B
def B : Set Int := { x | x < 1 }

-- Define the intersection set
def intersection_AB : Set Int := { -1, 0 }

-- Formalize the proof statement
theorem intersection_A_B : (A ∩ B) = intersection_AB :=
by sorry

end intersection_A_B_l182_182568


namespace ab_value_l182_182505

theorem ab_value (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 30) (h4 : 2 * a * b + 12 * a = 3 * b + 270) :
  a * b = 216 := by
  sorry

end ab_value_l182_182505


namespace ratio_Q_P_l182_182118

theorem ratio_Q_P : 
  ∀ (P Q : ℚ), (∀ x : ℚ, x ≠ -3 → x ≠ 0 → x ≠ 5 → 
    (P / (x + 3) + Q / (x * (x - 5)) = (x^2 - 3*x + 12) / (x^3 + x^2 - 15*x))) →
    (Q / P) = 20 / 9 :=
by
  intros P Q h
  sorry

end ratio_Q_P_l182_182118


namespace convert_A03_to_decimal_l182_182660

theorem convert_A03_to_decimal :
  let A := 10
  let hex_value := A * 16^2 + 0 * 16^1 + 3 * 16^0
  hex_value = 2563 :=
by
  let A := 10
  let hex_value := A * 16^2 + 0 * 16^1 + 3 * 16^0
  have : hex_value = 2563 := sorry
  exact this

end convert_A03_to_decimal_l182_182660


namespace find_a9_for_geo_seq_l182_182715

noncomputable def geo_seq_a_3_a_13_positive_common_ratio_2 (a_3 a_9 a_13 : ℕ) : Prop :=
  (a_3 * a_13 = 16) ∧ (a_3 > 0) ∧ (a_9 > 0) ∧ (a_13 > 0) ∧ (forall (n₁ n₂ : ℕ), a_9 = a_3 * 2 ^ 6)

theorem find_a9_for_geo_seq (a_3 a_9 a_13 : ℕ) 
  (h : geo_seq_a_3_a_13_positive_common_ratio_2 a_3 a_9 a_13) :
  a_9 = 8 :=
  sorry

end find_a9_for_geo_seq_l182_182715


namespace equalize_expenses_l182_182259

def total_expenses := 130 + 160 + 150 + 180
def per_person_share := total_expenses / 4
def tom_owes := per_person_share - 130
def dorothy_owes := per_person_share - 160
def sammy_owes := per_person_share - 150
def alice_owes := per_person_share - 180
def t := tom_owes
def d := dorothy_owes

theorem equalize_expenses : t - dorothy_owes = 30 := by
  sorry

end equalize_expenses_l182_182259


namespace exponent_division_l182_182988

theorem exponent_division (m n : ℕ) (h : m - n = 1) : 5 ^ m / 5 ^ n = 5 :=
by {
  sorry
}

end exponent_division_l182_182988


namespace sqrt_sum_inequality_l182_182956

-- Define variables a and b as positive real numbers
variable {a b : ℝ}

-- State the theorem to be proved
theorem sqrt_sum_inequality (ha : 0 < a) (hb : 0 < b) : 
  (a.sqrt + b.sqrt)^8 ≥ 64 * a * b * (a + b)^2 :=
sorry

end sqrt_sum_inequality_l182_182956


namespace incorrect_statement_is_A_l182_182035

theorem incorrect_statement_is_A :
  (∀ (w h : ℝ), w * (2 * h) ≠ 3 * (w * h)) ∧
  (∀ (s : ℝ), (2 * s) ^ 2 = 4 * (s ^ 2)) ∧
  (∀ (s : ℝ), (2 * s) ^ 3 = 8 * (s ^ 3)) ∧
  (∀ (w h : ℝ), (w / 2) * (3 * h) = (3 / 2) * (w * h)) ∧
  (∀ (l w : ℝ), (2 * l) * (3 * w) = 6 * (l * w)) →
  ∃ (incorrect_statement : String), incorrect_statement = "A" := 
by 
  sorry

end incorrect_statement_is_A_l182_182035


namespace renata_donation_l182_182695

variable (D L : ℝ)

theorem renata_donation : ∃ D : ℝ, 
  (10 - D + 90 - L - 2 + 65 = 94) ↔ D = 4 :=
by
  sorry

end renata_donation_l182_182695


namespace sequence_sum_l182_182694

-- Define the arithmetic sequence {a_n}
def a_n (n : ℕ) : ℕ := n + 1

-- Define the geometric sequence {b_n}
def b_n (n : ℕ) : ℕ := 2^(n - 1)

-- State the theorem
theorem sequence_sum : (b_n (a_n 1) + b_n (a_n 2) + b_n (a_n 3) + b_n (a_n 4) + b_n (a_n 5) + b_n (a_n 6)) = 126 := by
  sorry

end sequence_sum_l182_182694


namespace real_roots_range_l182_182301

theorem real_roots_range (a : ℝ) : (∃ x : ℝ, a*x^2 + 2*x - 1 = 0) ↔ (a >= -1 ∧ a ≠ 0) :=
by 
  sorry

end real_roots_range_l182_182301


namespace evaluate_expression_l182_182484

theorem evaluate_expression :
    123 - (45 * (9 - 6) - 78) + (0 / 1994) = 66 :=
by
  sorry

end evaluate_expression_l182_182484


namespace triangle_angle_sum_depends_on_parallel_postulate_l182_182327

-- Definitions of conditions
def triangle_angle_sum_theorem (A B C : ℝ) : Prop :=
  A + B + C = 180

def parallel_postulate : Prop :=
  ∀ (l : ℝ) (P : ℝ), ∃! (m : ℝ), m ≠ l ∧ ∀ (Q : ℝ), Q ≠ P → (Q = l ∧ Q = m)

-- Theorem statement: proving the dependence of the triangle_angle_sum_theorem on the parallel_postulate
theorem triangle_angle_sum_depends_on_parallel_postulate: 
  ∀ (A B C : ℝ), parallel_postulate → triangle_angle_sum_theorem A B C :=
sorry

end triangle_angle_sum_depends_on_parallel_postulate_l182_182327


namespace ratio_A_to_B_l182_182437

theorem ratio_A_to_B (total_weight_X : ℕ) (weight_B : ℕ) (weight_A : ℕ) (h₁ : total_weight_X = 324) (h₂ : weight_B = 270) (h₃ : weight_A = total_weight_X - weight_B):
  weight_A / gcd weight_A weight_B = 1 ∧ weight_B / gcd weight_A weight_B = 5 :=
by
  sorry

end ratio_A_to_B_l182_182437


namespace systematic_sampling_interval_l182_182783

-- Define the total number of students and sample size
def N : ℕ := 1200
def n : ℕ := 40

-- Define the interval calculation for systematic sampling
def k : ℕ := N / n

-- Prove that the interval k is 30
theorem systematic_sampling_interval : k = 30 := by
sorry

end systematic_sampling_interval_l182_182783


namespace doodads_for_thingamabobs_l182_182232

-- Definitions for the conditions
def doodads_per_widgets : ℕ := 18
def widgets_per_thingamabobs : ℕ := 11
def widgets_count : ℕ := 5
def thingamabobs_count : ℕ := 4
def target_thingamabobs : ℕ := 80

-- Definition for the final proof statement
theorem doodads_for_thingamabobs : 
    doodads_per_widgets * (target_thingamabobs * widgets_per_thingamabobs / thingamabobs_count / widgets_count) = 792 := 
by
  sorry

end doodads_for_thingamabobs_l182_182232


namespace moles_of_HCl_formed_l182_182576

theorem moles_of_HCl_formed
  (C2H6_initial : Nat)
  (Cl2_initial : Nat)
  (HCl_expected : Nat)
  (balanced_reaction : C2H6_initial + Cl2_initial = C2H6_initial + HCl_expected):
  C2H6_initial = 2 → Cl2_initial = 2 → HCl_expected = 2 :=
by
  intros
  sorry

end moles_of_HCl_formed_l182_182576


namespace forty_percent_of_number_l182_182614

theorem forty_percent_of_number (N : ℝ) (h : (1/4) * (1/3) * (2/5) * N = 10) : 0.40 * N = 120 :=
sorry

end forty_percent_of_number_l182_182614


namespace radius_of_sphere_in_truncated_cone_l182_182762

-- Definitions based on conditions
def radius_top_base := 5
def radius_bottom_base := 24

-- Theorem statement (without proof)
theorem radius_of_sphere_in_truncated_cone :
    (∃ (R_s : ℝ),
      (R_s = Real.sqrt 180.5) ∧
      ∀ (h : ℝ),
      (h^2 + (radius_bottom_base - radius_top_base)^2 = (h + R_s)^2 - R_s^2)) :=
sorry

end radius_of_sphere_in_truncated_cone_l182_182762


namespace PQ_ratio_l182_182452

-- Definitions
def hexagon_area : ℕ := 7
def base_of_triangle : ℕ := 4

-- Conditions
def PQ_bisects_area (A : ℕ) : Prop :=
  A = hexagon_area / 2

def area_below_PQ (U T : ℚ) : Prop :=
  U + T = hexagon_area / 2 ∧ U = 1

def triangle_area (T b : ℚ) : ℚ :=
  1/2 * b * (5/4)

def XQ_QY_ratio (XQ QY : ℚ) : ℚ :=
  XQ / QY

-- Theorem Statement
theorem PQ_ratio (XQ QY : ℕ) (h1 : PQ_bisects_area (hexagon_area / 2))
  (h2 : area_below_PQ 1 (triangle_area (5/2) base_of_triangle))
  (h3 : XQ + QY = base_of_triangle) : XQ_QY_ratio XQ QY = 1 := sorry

end PQ_ratio_l182_182452


namespace base_is_16_l182_182400

noncomputable def base_y_eq : Prop := ∃ base : ℕ, base ^ 8 = 4 ^ 16

theorem base_is_16 (base : ℕ) (h₁ : base ^ 8 = 4 ^ 16) : base = 16 :=
by
  sorry  -- Proof goes here

end base_is_16_l182_182400


namespace min_cost_to_win_l182_182621

theorem min_cost_to_win (n : ℕ) : 
  (∀ m : ℕ, m = 0 →
  (∀ cents : ℕ, 
  (n = 5 * m ∨ n = m + 1) ∧ n > 2008 ∧ n % 100 = 42 → 
  cents = 35)) :=
sorry

end min_cost_to_win_l182_182621


namespace jellybean_ratio_l182_182644

-- Define the conditions
def Matilda_jellybeans := 420
def Steve_jellybeans := 84
def Matt_jellybeans := 10 * Steve_jellybeans

-- State the theorem to prove the ratio
theorem jellybean_ratio : (Matilda_jellybeans : Nat) / (Matt_jellybeans : Nat) = 1 / 2 :=
by
  sorry

end jellybean_ratio_l182_182644


namespace adam_books_l182_182669

theorem adam_books (before_books total_shelves books_per_shelf after_books leftover_books bought_books : ℕ)
  (h_before: before_books = 56)
  (h_shelves: total_shelves = 4)
  (h_books_per_shelf: books_per_shelf = 20)
  (h_leftover: leftover_books = 2)
  (h_after: after_books = (total_shelves * books_per_shelf) + leftover_books)
  (h_difference: bought_books = after_books - before_books) :
  bought_books = 26 :=
by
  sorry

end adam_books_l182_182669


namespace sin_double_angle_l182_182389

-- Lean code to define the conditions and represent the problem
variable (α : ℝ)
variable (x y : ℝ) 
variable (r : ℝ := Real.sqrt (x^2 + y^2))

-- Given conditions
def point_on_terminal_side (x y : ℝ) (h : x = 1 ∧ y = -2) : Prop :=
  ∃ α, (⟨1, -2⟩ : ℝ × ℝ) = ⟨Real.cos α * (Real.sqrt (1^2 + (-2)^2)), Real.sin α * (Real.sqrt (1^2 + (-2)^2))⟩

-- The theorem to prove
theorem sin_double_angle (h : point_on_terminal_side 1 (-2) ⟨rfl, rfl⟩) : 
  Real.sin (2 * α) = -4 / 5 := 
sorry

end sin_double_angle_l182_182389


namespace balls_distribution_ways_l182_182160

theorem balls_distribution_ways : 
  ∃ (ways : ℕ), ways = 15 := by
  sorry

end balls_distribution_ways_l182_182160


namespace find_a_l182_182199

theorem find_a (P : ℝ) (hP : P ≠ 0) (S : ℕ → ℝ) (a_n : ℕ → ℝ)
  (hSn : ∀ n, S n = 3^n + a)
  (ha_n : ∀ n, a_n (n + 1) = P * a_n n)
  (hS1 : S 1 = a_n 1)
  (hS2 : S 2 = S 1 + a_n 2 - a_n 1)
  (hS3 : S 3 = S 2 + a_n 3 - a_n 2) :
  a = -1 := sorry

end find_a_l182_182199


namespace prime_square_sum_l182_182814

theorem prime_square_sum (p q m : ℕ) (hp : Prime p) (hq : Prime q) (hne : p ≠ q)
  (hp_eq : p^2 - 2001 * p + m = 0) (hq_eq : q^2 - 2001 * q + m = 0) :
  p^2 + q^2 = 3996005 :=
sorry

end prime_square_sum_l182_182814


namespace podcast_length_l182_182677

theorem podcast_length (x : ℝ) (hx : x + 2 * x + 1.75 + 1 + 1 = 6) : x = 0.75 :=
by {
  -- We do not need the proof steps here
  sorry
}

end podcast_length_l182_182677


namespace angle_A_in_triangle_find_b_c_given_a_and_A_l182_182944

theorem angle_A_in_triangle (A B C : ℝ) (a b c : ℝ)
  (h1 : 2 * Real.cos (2 * A) + 4 * Real.cos (B + C) + 3 = 0) :
  A = π / 3 :=
by
  sorry

theorem find_b_c_given_a_and_A (b c : ℝ)
  (A : ℝ)
  (a : ℝ := Real.sqrt 3)
  (h1 : 2 * b * Real.cos A + Real.sqrt (0 - c^2 + 6 * c - 9) = a)
  (h2 : b + c = 3)
  (h3 : A = π / 3) :
  (b = 2 ∧ c = 1) ∨ (b = 1 ∧ c = 2) :=
by
  sorry

end angle_A_in_triangle_find_b_c_given_a_and_A_l182_182944


namespace daily_sales_volume_80_sales_volume_function_price_for_profit_l182_182052

-- Define all relevant conditions
def cost_price : ℝ := 70
def max_price : ℝ := 99
def initial_price : ℝ := 95
def initial_sales : ℕ := 50
def price_reduction_effect : ℕ := 2

-- Part 1: Proving daily sales volume at 80 yuan
theorem daily_sales_volume_80 : 
  (initial_price - 80) * price_reduction_effect + initial_sales = 80 := 
by sorry

-- Part 2: Proving functional relationship
theorem sales_volume_function (x : ℝ) (h₁ : 70 ≤ x) (h₂ : x ≤ 99) : 
  (initial_sales + price_reduction_effect * (initial_price - x) = -2 * x + 240) :=
by sorry

-- Part 3: Proving price for 1200 yuan daily profit
theorem price_for_profit (profit_target : ℝ) (h : profit_target = 1200) :
  ∃ x, (x - cost_price) * (initial_sales + price_reduction_effect * (initial_price - x)) = profit_target ∧ x ≤ max_price :=
by sorry

end daily_sales_volume_80_sales_volume_function_price_for_profit_l182_182052


namespace m_n_sum_l182_182580

theorem m_n_sum (m n : ℝ) (h : ∀ x : ℝ, x^2 + m * x + 6 = (x - 2) * (x - n)) : m + n = -2 :=
by
  sorry

end m_n_sum_l182_182580


namespace square_pizza_area_larger_by_27_percent_l182_182846

theorem square_pizza_area_larger_by_27_percent :
  let r := 5
  let A_circle := Real.pi * r^2
  let s := 2 * r
  let A_square := s^2
  let delta_A := A_square - A_circle
  let percent_increase := (delta_A / A_circle) * 100
  Int.floor (percent_increase + 0.5) = 27 :=
by
  sorry

end square_pizza_area_larger_by_27_percent_l182_182846


namespace volume_of_prism_l182_182914

theorem volume_of_prism 
  (a b c : ℝ) 
  (h₁ : a * b = 51) 
  (h₂ : b * c = 52) 
  (h₃ : a * c = 53) 
  : (a * b * c) = 374 :=
by sorry

end volume_of_prism_l182_182914


namespace multiply_base5_234_75_l182_182597

def to_base5 (n : ℕ) : ℕ := 
  let rec helper (n : ℕ) (acc : ℕ) (multiplier : ℕ) : ℕ := 
    if n = 0 then acc
    else
      let d := n % 5
      let q := n / 5
      helper q (acc + d * multiplier) (multiplier * 10)
  helper n 0 1

def base5_multiplication (a b : ℕ) : ℕ :=
  to_base5 ((a * b : ℕ))

theorem multiply_base5_234_75 : base5_multiplication 234 75 = 450620 := 
  sorry

end multiply_base5_234_75_l182_182597


namespace project_completion_time_l182_182077

theorem project_completion_time
  (A_time B_time : ℕ) 
  (hA : A_time = 20)
  (hB : B_time = 20)
  (A_quit_days : ℕ) 
  (hA_quit : A_quit_days = 10) :
  ∃ x : ℕ, (x - A_quit_days) * (1 / A_time : ℚ) + (x * (1 / B_time : ℚ)) = 1 ∧ x = 15 := by
  sorry

end project_completion_time_l182_182077


namespace problem1_part1_problem1_part2_l182_182070

theorem problem1_part1 (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : 
  (a + b + c) * (a^2 + b^2 + c^2) ≤ 3 * (a^3 + b^3 + c^3) := 
sorry

theorem problem1_part2 (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : 
  (a / (b + c) + b / (c + a) + c / (a + b)) ≥ 3 / 2 := 
sorry

end problem1_part1_problem1_part2_l182_182070


namespace sum_of_squares_l182_182753

theorem sum_of_squares (n : Nat) (h : n = 2005^2) : 
  ∃ a1 b1 a2 b2 a3 b3 a4 b4 : Int, 
    (n = a1^2 + b1^2 ∧ a1 ≠ 0 ∧ b1 ≠ 0) ∧ 
    (n = a2^2 + b2^2 ∧ a2 ≠ 0 ∧ b2 ≠ 0) ∧ 
    (n = a3^2 + b3^2 ∧ a3 ≠ 0 ∧ b3 ≠ 0) ∧ 
    (n = a4^2 + b4^2 ∧ a4 ≠ 0 ∧ b4 ≠ 0) ∧ 
    (a1, b1) ≠ (a2, b2) ∧ 
    (a1, b1) ≠ (a3, b3) ∧ 
    (a1, b1) ≠ (a4, b4) ∧ 
    (a2, b2) ≠ (a3, b3) ∧ 
    (a2, b2) ≠ (a4, b4) ∧ 
    (a3, b3) ≠ (a4, b4) :=
by
  sorry

end sum_of_squares_l182_182753


namespace three_digit_number_divisible_by_7_l182_182465

theorem three_digit_number_divisible_by_7 (t : ℕ) :
  (n : ℕ) = 600 + 10 * t + 5 →
  n ≥ 100 ∧ n < 1000 →
  n % 10 = 5 →
  (n / 100) % 10 = 6 →
  n % 7 = 0 →
  n = 665 :=
by
  sorry

end three_digit_number_divisible_by_7_l182_182465


namespace neg_p_l182_182051

-- Define the sets A and B
def is_odd (x : ℤ) : Prop := x % 2 = 1
def is_even (x : ℤ) : Prop := x % 2 = 0

-- Define the proposition p
def p : Prop := ∀ x : ℤ, is_odd x → is_even (2 * x)

-- State the negation of proposition p
theorem neg_p : ¬ p ↔ ∃ x : ℤ, is_odd x ∧ ¬ is_even (2 * x) := by sorry

end neg_p_l182_182051


namespace part1_part2_l182_182562

open Real

variables {a b c : ℝ}

theorem part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 1 / a + 1 / b + 1 / c = 1) :
    a + 4 * b + 9 * c ≥ 36 :=
sorry

theorem part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 1 / a + 1 / b + 1 / c = 1) :
    (b + c) / sqrt a + (a + c) / sqrt b + (a + b) / sqrt c ≥ 2 * sqrt (a * b * c) :=
sorry

end part1_part2_l182_182562


namespace evaluate_expression_l182_182263

theorem evaluate_expression : 
  (20 - 19 + 18 - 17 + 16 - 15 + 14 - 13 + 12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) / 
  (2 - 3 + 4 - 5 + 6 - 7 + 8 - 9 + 10 - 11 + 12 - 13 + 14 - 15 + 16 - 17 + 18 - 19 + 20)
  = 10 / 11 := 
by
  sorry

end evaluate_expression_l182_182263


namespace chandler_bike_purchase_l182_182147

theorem chandler_bike_purchase : 
  ∀ (x : ℕ), (120 + 20 * x = 640) → x = 26 := 
by
  sorry

end chandler_bike_purchase_l182_182147


namespace increase_number_correct_l182_182157

-- Definitions for the problem
def originalNumber : ℕ := 110
def increasePercent : ℝ := 0.5

-- Statement to be proved
theorem increase_number_correct : originalNumber + (originalNumber * increasePercent) = 165 := by
  sorry

end increase_number_correct_l182_182157


namespace original_proposition_false_converse_false_inverse_false_contrapositive_false_l182_182518

-- Define the original proposition
def original_proposition (a b : ℝ) : Prop := 
  (a * b ≤ 0) → (a ≤ 0 ∨ b ≤ 0)

-- Define the converse
def converse (a b : ℝ) : Prop := 
  (a ≤ 0 ∨ b ≤ 0) → (a * b ≤ 0)

-- Define the inverse
def inverse (a b : ℝ) : Prop := 
  (a * b > 0) → (a > 0 ∧ b > 0)

-- Define the contrapositive
def contrapositive (a b : ℝ) : Prop := 
  (a > 0 ∧ b > 0) → (a * b > 0)

-- Prove that the original proposition is false
theorem original_proposition_false : ∀ (a b : ℝ), ¬ original_proposition a b :=
by sorry

-- Prove that the converse is false
theorem converse_false : ∀ (a b : ℝ), ¬ converse a b :=
by sorry

-- Prove that the inverse is false
theorem inverse_false : ∀ (a b : ℝ), ¬ inverse a b :=
by sorry

-- Prove that the contrapositive is false
theorem contrapositive_false : ∀ (a b : ℝ), ¬ contrapositive a b :=
by sorry

end original_proposition_false_converse_false_inverse_false_contrapositive_false_l182_182518


namespace penny_nickel_dime_heads_probability_l182_182665

def num_successful_outcomes : Nat :=
1 * 1 * 1 * 2

def total_possible_outcomes : Nat :=
2 ^ 4

def probability_event : ℚ :=
num_successful_outcomes / total_possible_outcomes

theorem penny_nickel_dime_heads_probability :
  probability_event = 1 / 8 := 
by
  sorry

end penny_nickel_dime_heads_probability_l182_182665


namespace pyramid_base_side_length_l182_182654

theorem pyramid_base_side_length
  (area_lateral_face : ℝ)
  (slant_height : ℝ)
  (side_length : ℝ)
  (h1 : area_lateral_face = 144)
  (h2 : slant_height = 24)
  (h3 : 144 = 0.5 * side_length * 24) : 
  side_length = 12 :=
by
  sorry

end pyramid_base_side_length_l182_182654


namespace solve_for_m_l182_182638

def power_function_monotonic (m : ℝ) : Prop :=
  (m^2 - m - 1 = 1) ∧ (m^2 - 2 * m - 3 < 0)

theorem solve_for_m (m : ℝ) (h : power_function_monotonic m) : m = 2 :=
sorry

end solve_for_m_l182_182638


namespace problem1_problem2_l182_182887

section ProofProblems

-- Definitions for binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Problem 1: Prove that n! = binom(n, k) * k! * (n-k)!
theorem problem1 (n k : ℕ) : n.factorial = binom n k * k.factorial * (n - k).factorial :=
by sorry

-- Problem 2: Prove that binom(n, k) = binom(n-1, k) + binom(n-1, k-1)
theorem problem2 (n k : ℕ) : binom n k = binom (n-1) k + binom (n-1) (k-1) :=
by sorry

end ProofProblems

end problem1_problem2_l182_182887


namespace average_students_is_12_l182_182292

-- Definitions based on the problem's conditions
variables (a b c : Nat)

-- Given conditions
axiom condition1 : a + b + c = 30
axiom condition2 : a + c = 19
axiom condition3 : b + c = 9

-- Prove that the number of average students (c) is 12
theorem average_students_is_12 : c = 12 := by 
  sorry

end average_students_is_12_l182_182292


namespace teamA_worked_days_l182_182733

def teamA_days_to_complete := 10
def teamB_days_to_complete := 15
def teamC_days_to_complete := 20
def total_days := 6
def teamA_halfway_withdrew := true

theorem teamA_worked_days : 
  ∀ (T_A T_B T_C total: ℕ) (halfway_withdrawal: Bool),
    T_A = teamA_days_to_complete ->
    T_B = teamB_days_to_complete ->
    T_C = teamC_days_to_complete ->
    total = total_days ->
    halfway_withdrawal = teamA_halfway_withdrew ->
    (total / 2) * (1 / T_A + 1 / T_B + 1 / T_C) = 3 := 
by 
  sorry

end teamA_worked_days_l182_182733


namespace area_excluding_hole_correct_l182_182922

def large_rectangle_area (x: ℝ) : ℝ :=
  4 * (x + 7) * (x + 5)

def hole_area (x: ℝ) : ℝ :=
  9 * (2 * x - 3) * (x - 2)

def area_excluding_hole (x: ℝ) : ℝ :=
  large_rectangle_area x - hole_area x

theorem area_excluding_hole_correct (x: ℝ) :
  area_excluding_hole x = -14 * x^2 + 111 * x + 86 :=
by
  -- The proof is omitted
  sorry

end area_excluding_hole_correct_l182_182922


namespace day_after_7k_days_is_friday_day_before_7k_days_is_thursday_day_after_100_days_is_sunday_l182_182151

-- Definitions based on problem conditions and questions
def day_of_week_after (n : ℤ) (current_day : String) : String :=
  if n % 7 = 0 then current_day else
    if n % 7 = 1 then "Saturday" else
    if n % 7 = 2 then "Sunday" else
    if n % 7 = 3 then "Monday" else
    if n % 7 = 4 then "Tuesday" else
    if n % 7 = 5 then "Wednesday" else
    "Thursday"

def day_of_week_before (n : ℤ) (current_day : String) : String :=
  day_of_week_after (-n) current_day

-- Conditions
def today : String := "Friday"

-- Prove the following
theorem day_after_7k_days_is_friday (k : ℤ) : day_of_week_after (7 * k) today = "Friday" :=
by sorry

theorem day_before_7k_days_is_thursday (k : ℤ) : day_of_week_before (7 * k) today = "Thursday" :=
by sorry

theorem day_after_100_days_is_sunday : day_of_week_after 100 today = "Sunday" :=
by sorry

end day_after_7k_days_is_friday_day_before_7k_days_is_thursday_day_after_100_days_is_sunday_l182_182151


namespace clothing_store_profit_l182_182256

theorem clothing_store_profit 
  (cost_price selling_price : ℕ)
  (initial_items_per_day items_increment items_reduction : ℕ)
  (initial_profit_per_day : ℕ) :
  -- Conditions
  cost_price = 50 ∧
  selling_price = 90 ∧
  initial_items_per_day = 20 ∧
  items_increment = 2 ∧
  items_reduction = 1 ∧
  initial_profit_per_day = 1200 →
  -- Question
  exists x, 
  (selling_price - x - cost_price) * (initial_items_per_day + items_increment * x) = initial_profit_per_day ∧
  x = 20 := 
sorry

end clothing_store_profit_l182_182256


namespace find_polynomials_g_l182_182825

-- Define functions f and proof target is g
def f (x : ℝ) : ℝ := x ^ 2

-- g is defined as an unknown polynomial with some constraints
variable (g : ℝ → ℝ)

-- The proof problem stating that if f(g(x)) = 9x^2 + 12x + 4, 
-- then g(x) = 3x + 2 or g(x) = -3x - 2
theorem find_polynomials_g (h : ∀ x : ℝ, f (g x) = 9 * x ^ 2 + 12 * x + 4) :
  (∀ x : ℝ, g x = 3 * x + 2) ∨ (∀ x : ℝ, g x = -3 * x - 2) := 
by
  sorry

end find_polynomials_g_l182_182825


namespace elaine_rent_percentage_l182_182904

theorem elaine_rent_percentage (E : ℝ) (hE : E > 0) :
  let rent_last_year := 0.20 * E
  let earnings_this_year := 1.25 * E
  let rent_this_year := 0.30 * earnings_this_year
  (rent_this_year / rent_last_year) * 100 = 187.5 :=
by
  sorry

end elaine_rent_percentage_l182_182904


namespace temperature_difference_l182_182156

def lowest_temp : ℝ := -15
def highest_temp : ℝ := 3

theorem temperature_difference :
  highest_temp - lowest_temp = 18 :=
by
  sorry

end temperature_difference_l182_182156


namespace quadratic_inequality_solution_set_l182_182830

theorem quadratic_inequality_solution_set {x : ℝ} : 
  x^2 < x + 6 ↔ (-2 < x ∧ x < 3) :=
by
  sorry

end quadratic_inequality_solution_set_l182_182830


namespace possible_values_of_q_l182_182983

theorem possible_values_of_q {q : ℕ} (hq : q > 0) :
  (∃ k : ℕ, (5 * q + 35) = k * (3 * q - 7) ∧ k > 0) ↔
  q = 3 ∨ q = 4 ∨ q = 5 ∨ q = 7 ∨ q = 9 ∨ q = 15 ∨ q = 21 ∨ q = 31 :=
by
  sorry

end possible_values_of_q_l182_182983


namespace runners_adjacent_vertices_after_2013_l182_182219

def hexagon_run_probability (t : ℕ) : ℚ :=
  (2 / 3) + (1 / 3) * ((1 / 4) ^ t)

theorem runners_adjacent_vertices_after_2013 :
  hexagon_run_probability 2013 = (2 / 3) + (1 / 3) * ((1 / 4) ^ 2013) := 
by 
  sorry

end runners_adjacent_vertices_after_2013_l182_182219


namespace weight_of_each_bag_of_flour_l182_182685

-- Definitions based on the given conditions
def cookies_eaten_by_Jim : ℕ := 15
def cookies_left : ℕ := 105
def total_cookies : ℕ := cookies_eaten_by_Jim + cookies_left

def cookies_per_dozen : ℕ := 12
def pounds_per_dozen : ℕ := 2

def dozens_of_cookies := total_cookies / cookies_per_dozen
def total_pounds_of_flour := dozens_of_cookies * pounds_per_dozen

def bags_of_flour : ℕ := 4

-- Question to be proved
theorem weight_of_each_bag_of_flour : total_pounds_of_flour / bags_of_flour = 5 := by
  sorry

end weight_of_each_bag_of_flour_l182_182685


namespace multiply_binomials_l182_182021

theorem multiply_binomials (x : ℝ) : (4 * x + 3) * (2 * x - 7) = 8 * x^2 - 22 * x - 21 :=
by 
  -- Proof is to be filled here
  sorry

end multiply_binomials_l182_182021


namespace denomination_is_20_l182_182242

noncomputable def denomination_of_250_coins (x : ℕ) : Prop :=
  250 * x + 84 * 25 = 7100

theorem denomination_is_20 (x : ℕ) (h : denomination_of_250_coins x) : x = 20 :=
by
  sorry

end denomination_is_20_l182_182242


namespace evaluate_expression_at_3_l182_182479

theorem evaluate_expression_at_3 :
  ((3^(3^2))^(3^3)) = 3^(243) := 
by 
  sorry

end evaluate_expression_at_3_l182_182479


namespace max_triangle_perimeter_l182_182921

theorem max_triangle_perimeter (y : ℤ) (h1 : y < 16) (h2 : y > 2) : 7 + 9 + y ≤ 31 :=
by
  -- proof goes here
  sorry

end max_triangle_perimeter_l182_182921


namespace midpoint_on_hyperbola_l182_182918

noncomputable def point_on_hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_on_hyperbola
  (A B : ℝ × ℝ)
  (H_A : point_on_hyperbola A.1 A.2)
  (H_B : point_on_hyperbola B.1 B.2) :
  (∃ M : ℝ × ℝ, M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ M = (-1, -4)) :=
sorry

end midpoint_on_hyperbola_l182_182918


namespace david_on_sixth_platform_l182_182081

theorem david_on_sixth_platform 
  (h₁ : walter_initial_fall = 4)
  (h₂ : walter_additional_fall = 3 * walter_initial_fall)
  (h₃ : total_fall = walter_initial_fall + walter_additional_fall)
  (h₄ : total_platforms = 8)
  (h₅ : total_height = total_fall)
  (h₆ : platform_height = total_height / total_platforms)
  (h₇ : david_fall_distance = walter_initial_fall)
  : (total_height - david_fall_distance) / platform_height = 6 := 
  by sorry

end david_on_sixth_platform_l182_182081


namespace binom_identity_l182_182738

-- Definition: Combinatorial coefficient (binomial coefficient)
def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem binom_identity (n k : ℕ) (h : k ≤ n) :
  binom (n + 1) k = binom n k + binom n (k - 1) := by
  sorry

end binom_identity_l182_182738


namespace fred_red_marbles_l182_182252

variable (R G B : ℕ)
variable (total : ℕ := 63)
variable (B_val : ℕ := 6)
variable (G_def : G = (1 / 2) * R)
variable (eq1 : R + G + B = total)
variable (eq2 : B = B_val)

theorem fred_red_marbles : R = 38 := 
by
  sorry

end fred_red_marbles_l182_182252


namespace Clea_escalator_time_l182_182776

variable {s : ℝ} -- speed of the escalator at its normal operating speed
variable {c : ℝ} -- speed of Clea walking down the escalator
variable {d : ℝ} -- distance of the escalator

theorem Clea_escalator_time :
  (30 * (c + s) = 72 * c) →
  (s = (7 * c) / 5) →
  (t = (72 * c) / ((3 / 2) * s)) →
  t = 240 / 7 :=
by
  sorry

end Clea_escalator_time_l182_182776


namespace normal_pumping_rate_l182_182445

-- Define the conditions and the proof problem
def pond_capacity : ℕ := 200
def drought_factor : ℚ := 2/3
def fill_time : ℕ := 50

theorem normal_pumping_rate (R : ℚ) :
  (drought_factor * R) * (fill_time : ℚ) = pond_capacity → R = 6 :=
by
  sorry

end normal_pumping_rate_l182_182445


namespace gcd_of_factorials_l182_182959

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem gcd_of_factorials :
  Nat.gcd (factorial 8) ((factorial 6)^2) = 1440 := by
  sorry

end gcd_of_factorials_l182_182959


namespace tan_beta_is_six_over_seventeen_l182_182548
-- Import the Mathlib library

-- Define the problem in Lean
theorem tan_beta_is_six_over_seventeen
  (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h1 : Real.sin α = 4 / 5)
  (h2 : Real.tan (α - β) = 2 / 3) :
  Real.tan β = 6 / 17 := 
by
  sorry

end tan_beta_is_six_over_seventeen_l182_182548


namespace range_of_a_l182_182965

theorem range_of_a
    (a : ℝ)
    (h : ∀ x y : ℝ, (x - a) ^ 2 + (y - a) ^ 2 = 4 → x ^ 2 + y ^ 2 = 1) :
    a ∈ (Set.Ioo (-(3 * Real.sqrt 2 / 2)) (-(Real.sqrt 2 / 2)) ∪ Set.Ioo (Real.sqrt 2 / 2) (3 * Real.sqrt 2 / 2)) :=
by
  sorry

end range_of_a_l182_182965


namespace valentine_problem_l182_182045

def initial_valentines : ℕ := 30
def given_valentines : ℕ := 8
def remaining_valentines : ℕ := 22

theorem valentine_problem : initial_valentines - given_valentines = remaining_valentines := by
  sorry

end valentine_problem_l182_182045


namespace find_x2_times_x1_plus_x3_l182_182453

noncomputable def a := Real.sqrt 2023
noncomputable def x1 := -Real.sqrt 7
noncomputable def x2 := 1 / a
noncomputable def x3 := Real.sqrt 7

theorem find_x2_times_x1_plus_x3 :
  let x1 := -Real.sqrt 7
  let x2 := 1 / Real.sqrt 2023
  let x3 := Real.sqrt 7
  x2 * (x1 + x3) = 0 :=
by
  sorry

end find_x2_times_x1_plus_x3_l182_182453


namespace skating_minutes_needed_l182_182635

-- Define the conditions
def minutes_per_day (day: ℕ) : ℕ :=
  if day ≤ 4 then 80 else if day ≤ 6 then 100 else 0

-- Define total skating time up to 6 days
def total_time_six_days := (4 * 80) + (2 * 100)

-- Prove that Gage needs to skate 180 minutes on the seventh day
theorem skating_minutes_needed : 
  (total_time_six_days + x = 7 * 100) → x = 180 :=
by sorry

end skating_minutes_needed_l182_182635


namespace calculate_y_l182_182872

theorem calculate_y (x : ℤ) (y : ℤ) (h1 : x = 121) (h2 : 2 * x - y = 102) : y = 140 :=
by
  -- Placeholder proof
  sorry

end calculate_y_l182_182872


namespace empty_solution_set_implies_a_range_l182_182719

def f (a x: ℝ) := x^2 + (1 - a) * x - a

theorem empty_solution_set_implies_a_range (a : ℝ) :
  (∀ x : ℝ, ¬ (f a (f a x) < 0)) → -3 ≤ a ∧ a ≤ 2 * Real.sqrt 2 - 3 :=
by
  sorry

end empty_solution_set_implies_a_range_l182_182719


namespace complement_of_A_in_U_l182_182569

-- Given definitions from the problem
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 4, 5}

-- The theorem to be proven
theorem complement_of_A_in_U : U \ A = {1, 3, 6, 7} :=
by
  sorry

end complement_of_A_in_U_l182_182569


namespace smores_cost_calculation_l182_182741

variable (people : ℕ) (s'mores_per_person : ℕ) (s'mores_per_set : ℕ) (cost_per_set : ℕ)

theorem smores_cost_calculation
  (h1 : s'mores_per_person = 3)
  (h2 : people = 8)
  (h3 : s'mores_per_set = 4)
  (h4 : cost_per_set = 3):
  (people * s'mores_per_person / s'mores_per_set) * cost_per_set = 18 := 
by
  sorry

end smores_cost_calculation_l182_182741


namespace duration_of_period_l182_182305

/-- The duration of the period at which B gains Rs. 1125 by lending 
Rs. 25000 at rate of 11.5% per annum and borrowing the same 
amount at 10% per annum -/
theorem duration_of_period (principal : ℝ) (rate_borrow : ℝ) (rate_lend : ℝ) (gain : ℝ) : 
  ∃ (t : ℝ), principal = 25000 ∧ rate_borrow = 0.10 ∧ rate_lend = 0.115 ∧ gain = 1125 → 
  t = 3 :=
by
  sorry

end duration_of_period_l182_182305


namespace students_can_be_helped_on_fourth_day_l182_182457

theorem students_can_be_helped_on_fourth_day : 
  ∀ (total_books first_day_students second_day_students third_day_students books_per_student : ℕ),
  total_books = 120 →
  first_day_students = 4 →
  second_day_students = 5 →
  third_day_students = 6 →
  books_per_student = 5 →
  (total_books - (first_day_students * books_per_student + second_day_students * books_per_student + third_day_students * books_per_student)) / books_per_student = 9 :=
by
  intros total_books first_day_students second_day_students third_day_students books_per_student h_total h_first h_second h_third h_books_per_student
  sorry

end students_can_be_helped_on_fourth_day_l182_182457


namespace brenda_friends_l182_182423

def total_slices (pizzas : ℕ) (slices_per_pizza : ℕ) : ℕ := pizzas * slices_per_pizza
def total_people (total_slices : ℕ) (slices_per_person : ℕ) : ℕ := total_slices / slices_per_person
def friends (total_people : ℕ) : ℕ := total_people - 1

theorem brenda_friends (pizzas : ℕ) (slices_per_pizza : ℕ) 
  (slices_per_person : ℕ) (pizzas_ordered : pizzas = 5) 
  (slices_per_pizza_value : slices_per_pizza = 4) 
  (slices_per_person_value : slices_per_person = 2) :
  friends (total_people (total_slices pizzas slices_per_pizza) slices_per_person) = 9 :=
by
  rw [pizzas_ordered, slices_per_pizza_value, slices_per_person_value]
  sorry

end brenda_friends_l182_182423


namespace cubic_polynomial_evaluation_l182_182480

theorem cubic_polynomial_evaluation
  (f : ℚ → ℚ)
  (cubic_f : ∃ a b c d : ℚ, ∀ x, f x = a*x^3 + b*x^2 + c*x + d)
  (h1 : f (-2) = -4)
  (h2 : f 3 = -9)
  (h3 : f (-4) = -16) :
  f 1 = -23 :=
sorry

end cubic_polynomial_evaluation_l182_182480


namespace price_of_pants_l182_182585

theorem price_of_pants
  (P S H : ℝ)
  (h1 : P + S + H = 340)
  (h2 : S = (3 / 4) * P)
  (h3 : H = P + 10) :
  P = 120 :=
by
  sorry

end price_of_pants_l182_182585


namespace value_of_a_minus_b_l182_182255

theorem value_of_a_minus_b (a b c : ℝ) 
    (h1 : 2011 * a + 2015 * b + c = 2021)
    (h2 : 2013 * a + 2017 * b + c = 2023)
    (h3 : 2012 * a + 2016 * b + 2 * c = 2026) : 
    a - b = -2 := 
by
  sorry

end value_of_a_minus_b_l182_182255


namespace lines_intersect_at_common_point_iff_l182_182804

theorem lines_intersect_at_common_point_iff (a b : ℝ) :
  (∃ x y : ℝ, a * x + 2 * b * y + 3 * (a + b + 1) = 0 ∧ 
               b * x + 2 * (a + b + 1) * y + 3 * a = 0 ∧ 
               (a + b + 1) * x + 2 * a * y + 3 * b = 0) ↔ 
  a + b = -1/2 :=
by
  sorry

end lines_intersect_at_common_point_iff_l182_182804


namespace total_words_in_poem_l182_182149

theorem total_words_in_poem 
  (stanzas : ℕ) 
  (lines_per_stanza : ℕ) 
  (words_per_line : ℕ) 
  (h_stanzas : stanzas = 20) 
  (h_lines_per_stanza : lines_per_stanza = 10) 
  (h_words_per_line : words_per_line = 8) : 
  stanzas * lines_per_stanza * words_per_line = 1600 := 
sorry

end total_words_in_poem_l182_182149


namespace fraction_of_tomatoes_eaten_l182_182100

theorem fraction_of_tomatoes_eaten (original : ℕ) (remaining : ℕ) (birds_ate : ℕ) (h1 : original = 21) (h2 : remaining = 14) (h3 : birds_ate = original - remaining) :
  (birds_ate : ℚ) / original = 1 / 3 :=
by
  sorry

end fraction_of_tomatoes_eaten_l182_182100


namespace apples_remaining_l182_182220

-- Define the initial condition of the number of apples on the tree
def initial_apples : ℕ := 7

-- Define the number of apples picked by Rachel
def picked_apples : ℕ := 4

-- Proof goal: the number of apples remaining on the tree is 3
theorem apples_remaining : (initial_apples - picked_apples = 3) :=
sorry

end apples_remaining_l182_182220


namespace solve_for_x_l182_182863

theorem solve_for_x (x : ℝ) (h : 3 * x - 5 = -2 * x + 10) : x = 3 := 
sorry

end solve_for_x_l182_182863


namespace reflect_parallelogram_l182_182339

theorem reflect_parallelogram :
  let D : ℝ × ℝ := (4,1)
  let Dx : ℝ × ℝ := (D.1, -D.2) -- Reflect across x-axis
  let Dxy : ℝ × ℝ := (Dx.2 - 1, Dx.1 - 1) -- Translate point down by 1 unit and reflect across y=x
  let D'' : ℝ × ℝ := (Dxy.1 + 1, Dxy.2 + 1) -- Translate point back up by 1 unit
  D'' = (-2, 5) := by
  sorry

end reflect_parallelogram_l182_182339


namespace price_of_toy_organizers_is_78_l182_182350

variable (P : ℝ) -- Price per set of toy organizers

-- Conditions
def total_cost_of_toy_organizers (P : ℝ) : ℝ := 3 * P
def total_cost_of_gaming_chairs : ℝ := 2 * 83
def total_sales (P : ℝ) : ℝ := total_cost_of_toy_organizers P + total_cost_of_gaming_chairs
def delivery_fee (P : ℝ) : ℝ := 0.05 * total_sales P
def total_amount_paid (P : ℝ) : ℝ := total_sales P + delivery_fee P

-- Proof statement
theorem price_of_toy_organizers_is_78 (h : total_amount_paid P = 420) : P = 78 :=
by
  sorry

end price_of_toy_organizers_is_78_l182_182350


namespace eval_expr_l182_182031

-- Define the expression
def expr : ℚ := 2 + 3 / (2 + 1 / (2 + 1 / 2))

-- The theorem to prove the evaluation of the expression
theorem eval_expr : expr = 13 / 4 :=
by
  sorry

end eval_expr_l182_182031


namespace area_triangle_ACD_proof_area_trapezoid_ABCD_proof_l182_182366

noncomputable def area_of_triangle (b h : ℝ) : ℝ :=
  (1 / 2) * b * h

noncomputable def area_trapezoid (b1 b2 h : ℝ) : ℝ :=
  (1 / 2) * (b1 + b2) * h

theorem area_triangle_ACD_proof :
  ∀ (A B C D X Y : ℝ), 
  A = 24 → 
  C = 10 → 
  X = 6 → 
  Y = 8 → 
  B = 23 → 
  D = 27 →
  area_of_triangle C 20 = 100 :=
by
  intros A B C D X Y hAB hCD hAX hXY hXX1 hYY1
  sorry

theorem area_trapezoid_ABCD_proof :
  ∀ (A B C D X Y : ℝ), 
  A = 24 → 
  C = 10 → 
  X = 6 → 
  Y = 8 → 
  B = 23 → 
  D = 27 → 
  area_trapezoid 24 10 24 = 260 :=
by
  intros A B C D X Y hAB hCD hAX hXY hXX1 hYY1
  sorry

end area_triangle_ACD_proof_area_trapezoid_ABCD_proof_l182_182366


namespace projection_of_a_on_b_l182_182210

theorem projection_of_a_on_b (a b : ℝ) (θ : ℝ) 
  (ha : |a| = 2) 
  (hb : |b| = 1)
  (hθ : θ = 60) : 
  (|a| * Real.cos (θ * Real.pi / 180)) = 1 := 
sorry

end projection_of_a_on_b_l182_182210


namespace smallest_y_condition_l182_182692

theorem smallest_y_condition : ∃ y : ℕ, y % 6 = 5 ∧ y % 7 = 6 ∧ y % 8 = 7 ∧ y = 167 :=
by 
  sorry

end smallest_y_condition_l182_182692


namespace tangent_line_equation_even_derived_l182_182632

def f (x a : ℝ) : ℝ := x^3 + (a - 2) * x^2 + a * x - 1

def f' (x a : ℝ) : ℝ := 3 * x^2 + 2 * (a - 2) * x + a

theorem tangent_line_equation_even_derived (a : ℝ) (h : ∀ x : ℝ, f' x a = f' (-x) a) :
  5 * 1 - (f 1 a) - 3 = 0 :=
by
  sorry

end tangent_line_equation_even_derived_l182_182632


namespace sin_theta_minus_cos_theta_l182_182595

theorem sin_theta_minus_cos_theta (θ : ℝ) (b : ℝ) (hθ_acute : 0 < θ ∧ θ < π / 2) (h_cos2θ : Real.cos (2 * θ) = b) :
  ∃ x, x = Real.sin θ - Real.cos θ ∧ (x = Real.sqrt b ∨ x = -Real.sqrt b) := 
by
  sorry

end sin_theta_minus_cos_theta_l182_182595


namespace Matias_longest_bike_ride_l182_182371

-- Define conditions in Lean
def blocks : ℕ := 4
def block_side_length : ℕ := 100
def streets : ℕ := 12

def Matias_route : Prop :=
  ∀ (intersections_used : ℕ), 
    intersections_used ≤ 4 → (streets - intersections_used/2 * 2) = 10

def correct_maximum_path_length : ℕ := 1000

-- Objective: Prove that given the conditions the longest route is 1000 meters
theorem Matias_longest_bike_ride :
  (100 * (streets - 2)) = correct_maximum_path_length :=
by
  sorry

end Matias_longest_bike_ride_l182_182371


namespace evaluate_x3_minus_y3_l182_182224

theorem evaluate_x3_minus_y3 (x y : ℤ) (h1 : x + y = 12) (h2 : 3 * x + y = 20) : x^3 - y^3 = -448 :=
by
  sorry

end evaluate_x3_minus_y3_l182_182224


namespace Grace_minus_Lee_l182_182978

-- Definitions for the conditions
def Grace_calculation : ℤ := 12 - (3 * 4 - 2)
def Lee_calculation : ℤ := (12 - 3) * 4 - 2

-- Statement of the problem to prove
theorem Grace_minus_Lee : Grace_calculation - Lee_calculation = -32 := by
  sorry

end Grace_minus_Lee_l182_182978


namespace hyperbola_eccentricity_eq_two_l182_182689

theorem hyperbola_eccentricity_eq_two :
  (∀ x y : ℝ, ((x^2 / 2) - (y^2 / 6) = 1) → 
    let a_squared := 2
    let b_squared := 6
    let a := Real.sqrt a_squared
    let b := Real.sqrt b_squared
    let e := Real.sqrt (1 + b_squared / a_squared)
    e = 2) := 
sorry

end hyperbola_eccentricity_eq_two_l182_182689


namespace cheese_cost_l182_182928

theorem cheese_cost (bread_cost cheese_cost total_paid total_change coin_change nickels_value : ℝ) 
                    (quarter dime nickels_count : ℕ)
                    (h1 : bread_cost = 4.20)
                    (h2 : total_paid = 7.00)
                    (h3 : quarter = 1)
                    (h4 : dime = 1)
                    (h5 : nickels_count = 8)
                    (h6 : coin_change = (quarter * 0.25) + (dime * 0.10) + (nickels_count * 0.05))
                    (h7 : total_change = total_paid - bread_cost)
                    (h8 : cheese_cost = total_change - coin_change) :
                    cheese_cost = 2.05 :=
by {
    sorry
}

end cheese_cost_l182_182928


namespace sport_formulation_water_l182_182961

theorem sport_formulation_water
  (f : ℝ) (c : ℝ) (w : ℝ) 
  (f_s : ℝ) (c_s : ℝ) (w_s : ℝ)
  (standard_ratio : f / c = 1 / 12 ∧ f / w = 1 / 30)
  (sport_ratio_corn_syrup : f_s / c_s = 3 * (f / c))
  (sport_ratio_water : f_s / w_s = (1 / 2) * (f / w))
  (corn_syrup_amount : c_s = 3) :
  w_s = 45 :=
by
  sorry

end sport_formulation_water_l182_182961


namespace smallest_number_divisible_conditions_l182_182281

theorem smallest_number_divisible_conditions :
  ∃ n : ℕ, n % 8 = 6 ∧ n % 7 = 5 ∧ ∀ m : ℕ, m % 8 = 6 ∧ m % 7 = 5 → n ≤ m →
  n % 9 = 0 := by
  sorry

end smallest_number_divisible_conditions_l182_182281


namespace min_value_of_sum_inverse_l182_182279

theorem min_value_of_sum_inverse (m n : ℝ) 
  (H1 : ∃ (x y : ℝ), (x + y - 1 = 0 ∧ 3 * x - y - 7 = 0) ∧ (mx + y + n = 0))
  (H2 : mn > 0) : 
  ∃ k : ℝ, k = 8 ∧ ∀ (m n : ℝ), mn > 0 → (2 * m + n = 1) → 1 / m + 2 / n ≥ k :=
by
  sorry

end min_value_of_sum_inverse_l182_182279


namespace rajas_monthly_income_l182_182627

theorem rajas_monthly_income (I : ℝ) (h : 0.6 * I + 0.1 * I + 0.1 * I + 5000 = I) : I = 25000 :=
sorry

end rajas_monthly_income_l182_182627


namespace largest_8_11_double_l182_182084

def is_8_11_double (M : ℕ) : Prop :=
  let digits_8 := (Nat.digits 8 M)
  let M_11 := Nat.ofDigits 11 digits_8
  M_11 = 2 * M

theorem largest_8_11_double : ∃ (M : ℕ), is_8_11_double M ∧ ∀ (N : ℕ), is_8_11_double N → N ≤ M :=
sorry

end largest_8_11_double_l182_182084


namespace simplify_expression_l182_182399

theorem simplify_expression (x : ℝ) :
  x - 3 * (1 + x) + 4 * (1 - x)^2 - 5 * (1 + 3 * x) = 4 * x^2 - 25 * x - 4 := by
  sorry

end simplify_expression_l182_182399


namespace evaluate_statements_l182_182436

-- Defining what it means for angles to be vertical
def vertical_angles (α β : ℝ) : Prop := α = β

-- Defining what complementary angles are
def complementary (α β : ℝ) : Prop := α + β = 90

-- Defining what supplementary angles are
def supplementary (α β : ℝ) : Prop := α + β = 180

-- Define the geometric properties for perpendicular and parallel lines
def unique_perpendicular_through_point (l : ℝ → ℝ) (p : ℝ × ℝ): Prop :=
  ∃! m, ∀ x, m * x + p.2 = l x

def unique_parallel_through_point (l : ℝ → ℝ) (p : ℝ × ℝ): Prop :=
  ∃! m, ∀ x, (l x ≠ m * x + p.2) ∧ (∀ y, y ≠ p.2 → l y ≠ m * y)

theorem evaluate_statements :
  (¬ ∃ α β, α = β ∧ vertical_angles α β) ∧
  (¬ ∃ α β, supplementary α β ∧ complementary α β) ∧
  ∃ l p, unique_perpendicular_through_point l p ∧
  ∃ l p, unique_parallel_through_point l p →
  2 = 2
  :=
by
  sorry  -- Proof is omitted

end evaluate_statements_l182_182436


namespace peggy_dolls_after_all_events_l182_182269

def initial_dolls : Nat := 6
def grandmother_gift : Nat := 28
def birthday_gift : Nat := grandmother_gift / 2
def lost_dolls (total : Nat) : Nat := (10 * total + 9) / 100  -- using integer division for rounding 10% up
def easter_gift : Nat := (birthday_gift + 2) / 3  -- using integer division for rounding one-third up
def friend_exchange_gain : Int := -1  -- gaining 1 doll but losing 2
def christmas_gift (easter_dolls : Nat) : Nat := (20 * easter_dolls) / 100 + easter_dolls  -- 20% more dolls
def ruined_dolls : Nat := 3

theorem peggy_dolls_after_all_events : initial_dolls + grandmother_gift + birthday_gift - lost_dolls (initial_dolls + grandmother_gift + birthday_gift) + easter_gift + friend_exchange_gain.toNat + christmas_gift easter_gift - ruined_dolls = 50 :=
by
  sorry

end peggy_dolls_after_all_events_l182_182269


namespace good_apples_count_l182_182014

def total_apples : ℕ := 14
def unripe_apples : ℕ := 6

theorem good_apples_count : total_apples - unripe_apples = 8 :=
by
  unfold total_apples unripe_apples
  sorry

end good_apples_count_l182_182014


namespace sum_of_interior_angles_of_octagon_l182_182236

theorem sum_of_interior_angles_of_octagon (n : ℕ) (h : n = 8) : (n - 2) * 180 = 1080 := by
  sorry

end sum_of_interior_angles_of_octagon_l182_182236


namespace new_books_count_l182_182520

-- Defining the conditions
def num_adventure_books : ℕ := 13
def num_mystery_books : ℕ := 17
def num_used_books : ℕ := 15

-- Proving the number of new books Sam bought
theorem new_books_count : (num_adventure_books + num_mystery_books) - num_used_books = 15 :=
by
  sorry

end new_books_count_l182_182520


namespace walther_janous_inequality_equality_condition_l182_182574

theorem walther_janous_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxy : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 :=
sorry

theorem equality_condition (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxy : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y = 7 ↔ x = 2 * y ∧ x / y = 2 ∧ y = z :=
sorry

end walther_janous_inequality_equality_condition_l182_182574


namespace negation_proposition_l182_182283

theorem negation_proposition :
  (¬ (∀ x : ℝ, ∃ n : ℕ, n > 0 ∧ n ≥ x)) ↔ (∃ x : ℝ, ∀ n : ℕ, n > 0 → n < x^2) := 
by
  sorry

end negation_proposition_l182_182283


namespace expression_meaning_l182_182326

variable (m n : ℤ) -- Assuming m and n are integers for the context.

theorem expression_meaning : 2 * (m - n) = 2 * (m - n) := 
by
  -- It simply follows from the definition of the expression
  sorry

end expression_meaning_l182_182326


namespace seq_eq_a1_b1_l182_182069

theorem seq_eq_a1_b1 {a b : ℕ → ℝ} 
  (h1 : ∀ n, a (n + 1) = 2 * b n - a n)
  (h2 : ∀ n, b (n + 1) = 2 * a n - b n)
  (h3 : ∀ n, a n > 0) :
  a 1 = b 1 := 
sorry

end seq_eq_a1_b1_l182_182069


namespace find_numbers_l182_182536

theorem find_numbers 
  (a b c d : ℝ)
  (h1 : b / c = c / a)
  (h2 : a + b + c = 19)
  (h3 : b - c = c - d)
  (h4 : b + c + d = 12) :
  (a = 25 ∧ b = -10 ∧ c = 4 ∧ d = 18) ∨ (a = 9 ∧ b = 6 ∧ c = 4 ∧ d = 2) :=
sorry

end find_numbers_l182_182536


namespace largest_divisor_of_square_divisible_by_24_l182_182771

theorem largest_divisor_of_square_divisible_by_24 (n : ℕ) (h₁ : n > 0) (h₂ : 24 ∣ n^2) (h₃ : ∀ k : ℕ, k ∣ n → k ≤ 8) : n = 24 := 
sorry

end largest_divisor_of_square_divisible_by_24_l182_182771


namespace prod_72516_9999_l182_182258

theorem prod_72516_9999 : 72516 * 9999 = 724987484 :=
by
  sorry

end prod_72516_9999_l182_182258


namespace complex_coordinates_l182_182925

theorem complex_coordinates (i : ℂ) (z : ℂ) (h : i^2 = -1) (h_z : z = (1 + 2 * i^3) / (2 + i)) :
  z = -i := 
by {
  sorry
}

end complex_coordinates_l182_182925


namespace jasmine_spent_l182_182462

theorem jasmine_spent 
  (original_cost : ℝ)
  (discount : ℝ)
  (h_original : original_cost = 35)
  (h_discount : discount = 17) : 
  original_cost - discount = 18 := 
by
  sorry

end jasmine_spent_l182_182462


namespace find_focus_with_larger_x_l182_182424

def hyperbola_foci_coordinates : Prop :=
  let center := (5, 10)
  let a := 7
  let b := 3
  let c := Real.sqrt (a^2 + b^2)
  let focus1 := (5 + c, 10)
  let focus2 := (5 - c, 10)
  focus1 = (5 + Real.sqrt 58, 10)
  
theorem find_focus_with_larger_x : hyperbola_foci_coordinates := 
  by
    sorry

end find_focus_with_larger_x_l182_182424


namespace peaches_left_in_baskets_l182_182338

theorem peaches_left_in_baskets :
  let initial_baskets := 5
  let initial_peaches_per_basket := 20
  let new_baskets := 4
  let new_peaches_per_basket := 25
  let peaches_removed_per_basket := 10

  let total_initial_peaches := initial_baskets * initial_peaches_per_basket
  let total_new_peaches := new_baskets * new_peaches_per_basket
  let total_peaches_before_removal := total_initial_peaches + total_new_peaches

  let total_baskets := initial_baskets + new_baskets
  let total_peaches_removed := total_baskets * peaches_removed_per_basket
  let peaches_left := total_peaches_before_removal - total_peaches_removed

  peaches_left = 110 := by
  sorry

end peaches_left_in_baskets_l182_182338


namespace greatest_possible_value_l182_182966

theorem greatest_possible_value :
  ∃ (N P M : ℕ), (M < 10) ∧ (N < 10) ∧ (P < 10) ∧ (M * (111 * M) = N * 1000 + P * 100 + M * 10 + M)
                ∧ (N * 1000 + P * 100 + M * 10 + M = 3996) :=
by
  sorry

end greatest_possible_value_l182_182966


namespace calculate_F_5_f_6_l182_182869

def f (a : ℤ) : ℤ := a + 3

def F (a b : ℤ) : ℤ := b^3 - 2 * a

theorem calculate_F_5_f_6 : F 5 (f 6) = 719 := by
  sorry

end calculate_F_5_f_6_l182_182869


namespace negation_of_tan_one_l182_182740

theorem negation_of_tan_one :
  (∃ x : ℝ, Real.tan x = 1) ↔ ¬ (∀ x : ℝ, Real.tan x ≠ 1) :=
by
  sorry

end negation_of_tan_one_l182_182740


namespace original_decimal_number_l182_182240

theorem original_decimal_number (I : ℤ) (d : ℝ) (h1 : 0 ≤ d) (h2 : d < 1) (h3 : I + 4 * (I + d) = 21.2) : I + d = 4.3 :=
by
  sorry

end original_decimal_number_l182_182240


namespace exchange_ways_10_dollar_l182_182408

theorem exchange_ways_10_dollar (p q : ℕ) (H : 2 * p + 5 * q = 200) : 
  ∃ (n : ℕ), n = 20 :=
by {
  sorry
}

end exchange_ways_10_dollar_l182_182408


namespace part_time_job_pay_per_month_l182_182286

def tuition_fee : ℝ := 90
def scholarship_percent : ℝ := 0.30
def scholarship_amount := scholarship_percent * tuition_fee
def amount_after_scholarship := tuition_fee - scholarship_amount
def remaining_amount : ℝ := 18
def months_to_pay : ℝ := 3
def amount_paid_so_far := amount_after_scholarship - remaining_amount

theorem part_time_job_pay_per_month : amount_paid_so_far / months_to_pay = 15 := by
  sorry

end part_time_job_pay_per_month_l182_182286


namespace seating_arrangement_correct_l182_182874

noncomputable def seating_arrangements_around_table : Nat :=
  7

def B_G_next_to_C (A B C D E F G : Prop) (d : Nat) : Prop :=
  d = 48

theorem seating_arrangement_correct : ∃ d, d = 48 := sorry

end seating_arrangement_correct_l182_182874


namespace sum_of_positive_numbers_is_360_l182_182873

variable (x y : ℝ)
variable (h1 : x * y = 50 * (x + y))
variable (h2 : x * y = 75 * (x - y))

theorem sum_of_positive_numbers_is_360 (hx : 0 < x) (hy : 0 < y) : x + y = 360 :=
by sorry

end sum_of_positive_numbers_is_360_l182_182873


namespace avg_salary_of_Raj_and_Roshan_l182_182304

variable (R S : ℕ)

theorem avg_salary_of_Raj_and_Roshan (h1 : (R + S + 7000) / 3 = 5000) : (R + S) / 2 = 4000 := by
  sorry

end avg_salary_of_Raj_and_Roshan_l182_182304


namespace problem1_problem2_l182_182127

noncomputable def circle_ast (a b : ℕ) : ℕ := sorry

axiom circle_ast_self (a : ℕ) : circle_ast a a = a
axiom circle_ast_zero (a : ℕ) : circle_ast a 0 = 2 * a
axiom circle_ast_add (a b c d : ℕ) : circle_ast a b + circle_ast c d = circle_ast (a + c) (b + d)

theorem problem1 : circle_ast (2 + 3) (0 + 3) = 7 := sorry

theorem problem2 : circle_ast 1024 48 = 2000 := sorry

end problem1_problem2_l182_182127


namespace basketball_games_l182_182050

theorem basketball_games (N M : ℕ) (h1 : N > 3 * M) (h2 : M > 5) (h3 : 3 * N + 4 * M = 88) : 3 * N = 48 :=
by sorry

end basketball_games_l182_182050


namespace value_of_y_when_x_is_neg2_l182_182493

theorem value_of_y_when_x_is_neg2 :
  ∃ (k b : ℝ), (k + b = 2) ∧ (-k + b = -4) ∧ (∀ x, y = k * x + b) ∧ (x = -2) → (y = -7) := 
sorry

end value_of_y_when_x_is_neg2_l182_182493


namespace probability_is_correct_l182_182811

-- Define the ratios for the colors: red, yellow, blue, black
def red_ratio := 6
def yellow_ratio := 2
def blue_ratio := 1
def black_ratio := 4

-- Define the total ratio
def total_ratio := red_ratio + yellow_ratio + blue_ratio + black_ratio

-- Define the ratio of red or blue regions
def red_or_blue_ratio := red_ratio + blue_ratio

-- Define the probability of landing on a red or blue region
def probability_red_or_blue := red_or_blue_ratio / total_ratio

-- State the theorem to prove
theorem probability_is_correct : probability_red_or_blue = 7 / 13 := 
by 
  -- Proof will go here
  sorry

end probability_is_correct_l182_182811


namespace orchestra_admission_l182_182476

theorem orchestra_admission (x v c t: ℝ) 
  -- Conditions
  (h1 : v = 1.25 * 1.6 * x)
  (h2 : c = 0.8 * x)
  (h3 : t = 0.4 * x)
  (h4 : v + c + t = 32) :
  -- Conclusion
  v = 20 ∧ c = 8 ∧ t = 4 :=
sorry

end orchestra_admission_l182_182476


namespace distance_from_Martins_house_to_Lawrences_house_l182_182774

def speed : ℝ := 2 -- Martin's speed is 2 miles per hour
def time : ℝ := 6  -- Time taken is 6 hours
def distance : ℝ := speed * time -- Distance formula

theorem distance_from_Martins_house_to_Lawrences_house : distance = 12 := by
  sorry

end distance_from_Martins_house_to_Lawrences_house_l182_182774


namespace line_equation_l182_182997

theorem line_equation (a b : ℝ) (h1 : (1, 2) ∈ line) (h2 : ∃ a b : ℝ, b = 2 * a ∧ line = {p : ℝ × ℝ | p.1 / a + p.2 / b = 1}) :
  line = {p : ℝ × ℝ | 2 * p.1 - p.2 = 0} ∨ line = {p : ℝ × ℝ | 2 * p.1 + p.2 - 4 = 0} :=
sorry

end line_equation_l182_182997


namespace choir_girls_count_l182_182870

noncomputable def number_of_girls_in_choir (o b t c b_boys : ℕ) : ℕ :=
  c - b_boys

theorem choir_girls_count (o b t b_boys : ℕ) (h1 : o = 20) (h2 : b = 2 * o) (h3 : t = 88)
  (h4 : b_boys = 12) : number_of_girls_in_choir o b t (t - (o + b)) b_boys = 16 :=
by
  sorry

end choir_girls_count_l182_182870


namespace volume_ratio_l182_182134

theorem volume_ratio (a : ℕ) (b : ℕ) (ft_to_inch : ℕ) (h1 : a = 4) (h2 : b = 2 * ft_to_inch) (ft_to_inch_value : ft_to_inch = 12) :
  (a^3) / (b^3) = 1 / 216 :=
by
  sorry

end volume_ratio_l182_182134


namespace impossible_division_l182_182343

noncomputable def total_matches := 1230

theorem impossible_division :
  ∀ (x y z : ℕ), 
  (x + y + z = total_matches) → 
  (z = (1 / 2) * (x + y + z)) → 
  false :=
by
  sorry

end impossible_division_l182_182343


namespace quadratic_b_value_l182_182805

theorem quadratic_b_value (b : ℝ) : 
  (∀ x : ℝ, -x^2 + b * x - 12 < 0 ↔ x < 3 ∨ x > 7) → b = 10 :=
by 
  sorry

end quadratic_b_value_l182_182805


namespace chosen_number_l182_182777

theorem chosen_number (x : ℕ) (h : (x / 12) - 240 = 8) : x = 2976 :=
sorry

end chosen_number_l182_182777


namespace flour_needed_correct_l182_182557

-- Define the total flour required and the flour already added
def total_flour : ℕ := 8
def flour_already_added : ℕ := 2

-- Define the equation to determine the remaining flour needed
def flour_needed : ℕ := total_flour - flour_already_added

-- Prove that the flour needed to be added is 6 cups
theorem flour_needed_correct : flour_needed = 6 := by
  sorry

end flour_needed_correct_l182_182557


namespace Patel_family_theme_park_expenses_l182_182405

def regular_ticket_price : ℝ := 12.5
def senior_discount : ℝ := 0.8
def child_discount : ℝ := 0.6
def senior_ticket_price := senior_discount * regular_ticket_price
def child_ticket_price := child_discount * regular_ticket_price

theorem Patel_family_theme_park_expenses :
  (2 * senior_ticket_price + 2 * child_ticket_price + 4 * regular_ticket_price) = 85 := by
  sorry

end Patel_family_theme_park_expenses_l182_182405


namespace m_perpendicular_beta_l182_182820

variables {Plane : Type*} {Line : Type*}

-- Definitions of the perpendicularity and parallelism
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (p1 p2 : Plane) : Prop := sorry

-- Given variables
variables (α β : Plane) (m : Line)

-- Conditions
axiom M_perpendicular_Alpha : perpendicular m α
axiom Alpha_parallel_Beta : parallel α β

-- Proof goal
theorem m_perpendicular_beta 
  (h1 : perpendicular m α) 
  (h2 : parallel α β) : 
  perpendicular m β := 
  sorry

end m_perpendicular_beta_l182_182820


namespace probability_three_white_two_black_eq_eight_seventeen_l182_182295
-- Import Mathlib library to access combinatorics functions.

-- Define the total number of white and black balls.
def total_white := 8
def total_black := 7

-- The key function to calculate combinations.
noncomputable def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Define the problem conditions as constants.
def total_balls := total_white + total_black
def chosen_balls := 5
def white_balls_chosen := 3
def black_balls_chosen := 2

-- Calculate number of combinations.
noncomputable def total_combinations : ℕ := choose total_balls chosen_balls
noncomputable def white_combinations : ℕ := choose total_white white_balls_chosen
noncomputable def black_combinations : ℕ := choose total_black black_balls_chosen

-- Calculate the probability as a rational number.
noncomputable def probability_exact_three_white_two_black : ℚ :=
  (white_combinations * black_combinations : ℚ) / total_combinations

-- The theorem we want to prove
theorem probability_three_white_two_black_eq_eight_seventeen :
  probability_exact_three_white_two_black = 8 / 17 := by
  sorry

end probability_three_white_two_black_eq_eight_seventeen_l182_182295


namespace problem1_problem2_l182_182365

theorem problem1 : 
  -(3^3) * ((-1 : ℚ)/ 3)^2 - 24 * (3/4 - 1/6 + 3/8) = -26 := 
by 
  sorry

theorem problem2 : 
  -(1^100 : ℚ) - (3/4) / (((-2)^2) * ((-1 / 4) ^ 2) - 1 / 2) = 2 := 
by 
  sorry

end problem1_problem2_l182_182365


namespace original_price_l182_182656

theorem original_price (SP : ℝ) (rate_of_profit : ℝ) (CP : ℝ) 
  (h1 : SP = 60) 
  (h2 : rate_of_profit = 0.20) 
  (h3 : SP = CP * (1 + rate_of_profit)) : CP = 50 := by
  sorry

end original_price_l182_182656


namespace circle_radius_tangent_to_parabola_l182_182722

theorem circle_radius_tangent_to_parabola (a : ℝ) (b r : ℝ) :
  (∀ x : ℝ, y = 4 * x ^ 2) ∧ 
  (b = a ^ 2 / 4) ∧ 
  (∀ x : ℝ, x ^ 2 + (4 * x ^ 2 - b) ^ 2 = r ^ 2)  → 
  r = a ^ 2 / 4 := 
  sorry

end circle_radius_tangent_to_parabola_l182_182722


namespace sum_of_ages_is_60_l182_182129

theorem sum_of_ages_is_60 (A B : ℕ) (h1 : A = 2 * B) (h2 : (A + 3) + (B + 3) = 66) : A + B = 60 :=
by sorry

end sum_of_ages_is_60_l182_182129


namespace sin_sum_cos_product_tan_sum_tan_product_l182_182270

theorem sin_sum_cos_product
  (A B C : ℝ)
  (h : A + B + C = π) : 
  (Real.sin A + Real.sin B + Real.sin C = 4 * Real.cos (A / 2) * Real.cos (B / 2) * Real.cos (C / 2)) :=
sorry

theorem tan_sum_tan_product
  (A B C : ℝ)
  (h : A + B + C = π) :
  (Real.tan A + Real.tan B + Real.tan C = Real.tan A * Real.tan B * Real.tan C) := 
sorry

end sin_sum_cos_product_tan_sum_tan_product_l182_182270


namespace find_integer_n_l182_182221

theorem find_integer_n (n : ℕ) (hn1 : 0 ≤ n) (hn2 : n < 102) (hmod : 99 * n % 102 = 73) : n = 97 :=
  sorry

end find_integer_n_l182_182221


namespace even_function_and_inverse_property_l182_182841

noncomputable def f (x : ℝ) : ℝ := (1 + x^2) / (1 - x^2)

theorem even_function_and_inverse_property (x : ℝ) (hx : x ≠ 1 ∧ x ≠ -1) :
  f (-x) = f x ∧ f (1 / x) = -f x := by
  sorry

end even_function_and_inverse_property_l182_182841


namespace simplify_expression_l182_182383

theorem simplify_expression (x : ℝ) (h : x ≠ 1) :
  (x^2 + x) / (x^2 - 2*x + 1) / ((x + 1) / (x - 1)) = x / (x - 1) :=
by
  sorry

end simplify_expression_l182_182383


namespace plane_equation_l182_182422

theorem plane_equation (A B C D x y z : ℤ) (h1 : A = 15) (h2 : B = -3) (h3 : C = 2) (h4 : D = -238) 
  (h5 : gcd (abs A) (gcd (abs B) (gcd (abs C) (abs D))) = 1) (h6 : A > 0) :
  A * x + B * y + C * z + D = 0 ↔ 15 * x - 3 * y + 2 * z - 238 = 0 :=
by
  sorry

end plane_equation_l182_182422


namespace initial_matches_l182_182148

theorem initial_matches (x : ℕ) (h1 : (34 * x + 89) / (x + 1) = 39) : x = 10 := by
  sorry

end initial_matches_l182_182148


namespace complex_sum_abs_eq_1_or_3_l182_182230

open Complex

theorem complex_sum_abs_eq_1_or_3
  (a b c : ℂ)
  (ha : abs a = 1)
  (hb : abs b = 1)
  (hc : abs c = 1)
  (h : a^3 / (b * c) + b^3 / (a * c) + c^3 / (a * b) = 1) :
  ∃ r : ℝ, (r = 1 ∨ r = 3) ∧ abs (a + b + c) = r :=
by {
  -- Proof goes here
  sorry
}

end complex_sum_abs_eq_1_or_3_l182_182230


namespace max_marks_l182_182860

theorem max_marks (M : ℝ) (h1 : 0.25 * M = 185 + 25) : M = 840 :=
by
  sorry

end max_marks_l182_182860


namespace tan_sum_pi_over_12_l182_182800

theorem tan_sum_pi_over_12 : 
  (Real.tan (Real.pi / 12) + Real.tan (5 * Real.pi / 12)) = 4 := 
sorry

end tan_sum_pi_over_12_l182_182800


namespace fraction_product_simplified_l182_182895

theorem fraction_product_simplified:
  (2 / 3) * (4 / 7) * (9 / 11) = 24 / 77 := by
  sorry

end fraction_product_simplified_l182_182895


namespace length_of_goods_train_l182_182821

theorem length_of_goods_train 
  (speed_kmh : ℕ) 
  (platform_length_m : ℕ) 
  (cross_time_s : ℕ) :
  speed_kmh = 72 → platform_length_m = 280 → cross_time_s = 26 → 
  ∃ train_length_m : ℕ, train_length_m = 240 :=
by
  intros h1 h2 h3
  sorry

end length_of_goods_train_l182_182821


namespace probability_X_eq_4_l182_182710

-- Define the number of students and boys
def total_students := 15
def total_boys := 7
def selected_students := 10

-- Define the binomial coefficient function
def binomial_coeff (n k : ℕ) : ℕ := n.choose k

-- Calculate the probability
def P_X_eq_4 := (binomial_coeff total_boys 4 * binomial_coeff (total_students - total_boys) 6) / binomial_coeff total_students selected_students

-- The statement to be proven
theorem probability_X_eq_4 :
  P_X_eq_4 = (binomial_coeff total_boys 4 * binomial_coeff (total_students - total_boys) 6) / binomial_coeff total_students selected_students := by
  sorry

end probability_X_eq_4_l182_182710


namespace no_consecutive_squares_of_arithmetic_progression_l182_182277

theorem no_consecutive_squares_of_arithmetic_progression (d : ℕ):
  (d % 10000 = 2019) →
  (∀ a b c : ℕ, a < b ∧ b < c → b^2 - a^2 = d ∧ c^2 - b^2 = d →
  false) :=
sorry

end no_consecutive_squares_of_arithmetic_progression_l182_182277


namespace trig_identity_l182_182929

theorem trig_identity (α : ℝ) (h : Real.tan α = 2) :
  (2 * Real.sin α - Real.cos α) / (2 * Real.cos α + Real.sin α) = 3 / 4 :=
by
  sorry

end trig_identity_l182_182929


namespace part_one_part_two_l182_182370

theorem part_one (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) : 
  ab + bc + ca ≤ 1 / 3 := sorry

theorem part_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) : 
  a^2 / b + b^2 / c + c^2 / a ≥ 1 := sorry

end part_one_part_two_l182_182370


namespace find_x2_y2_l182_182390

theorem find_x2_y2 (x y : ℝ) (h₁ : (x + y)^2 = 9) (h₂ : x * y = -6) : x^2 + y^2 = 21 := 
by
  sorry

end find_x2_y2_l182_182390


namespace area_of_triangle_l182_182284

theorem area_of_triangle (a b c : ℝ) (C : ℝ) 
  (h1 : c^2 = (a - b)^2 + 6)
  (h2 : C = Real.pi / 3) : 
  1/2 * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 :=
by
  sorry

end area_of_triangle_l182_182284


namespace only_n_is_zero_l182_182154

theorem only_n_is_zero (n : ℕ) (h : (n^2 + 1) ∣ n) : n = 0 := 
by sorry

end only_n_is_zero_l182_182154


namespace courier_cost_formula_l182_182022

def cost (P : ℕ) : ℕ :=
if P = 0 then 0 else max 50 (30 + 7 * (P - 1))

theorem courier_cost_formula (P : ℕ) : cost P = 
  if P = 0 then 0 else max 50 (30 + 7 * (P - 1)) :=
by
  sorry

end courier_cost_formula_l182_182022


namespace medians_square_sum_l182_182601

theorem medians_square_sum (a b c : ℝ) (ha : a = 13) (hb : b = 13) (hc : c = 10) :
  let m_a := (1 / 2 * (2 * b^2 + 2 * c^2 - a^2))^(1/2)
  let m_b := (1 / 2 * (2 * c^2 + 2 * a^2 - b^2))^(1/2)
  let m_c := (1 / 2 * (2 * a^2 + 2 * b^2 - c^2))^(1/2)
  m_a^2 + m_b^2 + m_c^2 = 432 :=
by
  sorry

end medians_square_sum_l182_182601


namespace quadratic_condition_l182_182993

theorem quadratic_condition (p q : ℝ) (x1 x2 : ℝ) (hx : x1 + x2 = -p) (hq : x1 * x2 = q) :
  p + q = 0 := sorry

end quadratic_condition_l182_182993


namespace rectangle_area_constant_l182_182328

theorem rectangle_area_constant (d : ℝ) (length width : ℝ) (h_ratio : length / width = 5 / 2) (h_diag : d = Real.sqrt (length^2 + width^2)) :
  ∃ k : ℝ, (length * width) = k * d^2 ∧ k = 10 / 29 :=
by
  use 10 / 29
  sorry

end rectangle_area_constant_l182_182328


namespace ratio_sheep_to_horses_is_correct_l182_182348

-- Definitions of given conditions
def ounces_per_horse := 230
def total_ounces_per_day := 12880
def number_of_sheep := 16

-- Express the number of horses and the ratio of sheep to horses
def number_of_horses : ℕ := total_ounces_per_day / ounces_per_horse
def ratio_sheep_to_horses := number_of_sheep / number_of_horses

-- The main statement to be proved
theorem ratio_sheep_to_horses_is_correct : ratio_sheep_to_horses = 2 / 7 :=
by
  sorry

end ratio_sheep_to_horses_is_correct_l182_182348


namespace trigonometric_identity_l182_182143

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = -1 / 2) : 
  (1 + 2 * Real.sin α * Real.cos α) / (Real.sin α ^ 2 - Real.cos α ^ 2) = -1 / 3 := 
by 
  sorry

end trigonometric_identity_l182_182143


namespace area_calculation_l182_182833

variable (x : ℝ)

def area_large_rectangle : ℝ := (2 * x + 9) * (x + 6)
def area_rectangular_hole : ℝ := (x - 1) * (2 * x - 5)
def area_square : ℝ := (x + 3) ^ 2
def area_remaining : ℝ := area_large_rectangle x - area_rectangular_hole x - area_square x

theorem area_calculation : area_remaining x = -x^2 + 22 * x + 40 := by
  sorry

end area_calculation_l182_182833


namespace similar_right_triangles_l182_182132

open Real

theorem similar_right_triangles (x : ℝ) (h : ℝ)
  (h₁: 12^2 + 9^2 = (12^2 + 9^2))
  (similarity : (12 / x) = (9 / 6))
  (p : hypotenuse = 12*12) :
  x = 8 ∧ h = 10 := by
  sorry

end similar_right_triangles_l182_182132


namespace number_of_women_l182_182633

theorem number_of_women (w1 w2: ℕ) (m1 m2 d1 d2: ℕ)
    (h1: w2 = 5) (h2: m2 = 100) (h3: d2 = 1) 
    (h4: d1 = 3) (h5: m1 = 360)
    (h6: w1 * d1 = m1 * d2 / m2 * w2) : w1 = 6 :=
by
  sorry

end number_of_women_l182_182633


namespace waiter_tables_l182_182551

theorem waiter_tables (w m : ℝ) (avg_customers_per_table : ℝ) (total_customers : ℝ) (t : ℝ)
  (hw : w = 7.0)
  (hm : m = 3.0)
  (havg : avg_customers_per_table = 1.111111111)
  (htotal : total_customers = w + m)
  (ht : t = total_customers / avg_customers_per_table) :
  t = 90 :=
by
  -- Proof would be inserted here
  sorry

end waiter_tables_l182_182551


namespace probability_of_color_change_is_1_over_6_l182_182248

noncomputable def watchColorChangeProbability : ℚ :=
  let cycleDuration := 45 + 5 + 40
  let favorableDuration := 5 + 5 + 5
  favorableDuration / cycleDuration

theorem probability_of_color_change_is_1_over_6 :
  watchColorChangeProbability = 1 / 6 :=
by
  sorry

end probability_of_color_change_is_1_over_6_l182_182248


namespace calc_expression_l182_182994

theorem calc_expression : (3.242 * 14) / 100 = 0.45388 :=
by
  sorry

end calc_expression_l182_182994


namespace domain_real_iff_l182_182747

noncomputable def is_domain_ℝ (m : ℝ) : Prop :=
  ∀ x : ℝ, (m * x^2 + 4 * m * x + 3 ≠ 0)

theorem domain_real_iff (m : ℝ) :
  is_domain_ℝ m ↔ (0 ≤ m ∧ m < 3 / 4) :=
sorry

end domain_real_iff_l182_182747


namespace Roe_saved_15_per_month_aug_nov_l182_182751

-- Step 1: Define the given conditions
def savings_per_month_jan_jul : ℕ := 10
def months_jan_jul : ℕ := 7
def savings_dec : ℕ := 20
def total_savings_needed : ℕ := 150
def months_aug_nov : ℕ := 4

-- Step 2: Define the intermediary calculations based on the conditions
def total_saved_jan_jul := savings_per_month_jan_jul * months_jan_jul
def total_savings_aug_nov := total_savings_needed - total_saved_jan_jul - savings_dec

-- Step 3: Define what we need to prove
def savings_per_month_aug_nov : ℕ := total_savings_aug_nov / months_aug_nov

-- Step 4: State the proof goal
theorem Roe_saved_15_per_month_aug_nov :
  savings_per_month_aug_nov = 15 :=
by
  sorry

end Roe_saved_15_per_month_aug_nov_l182_182751


namespace height_of_pole_l182_182391

theorem height_of_pole (pole_shadow tree_shadow tree_height : ℝ) 
                       (ratio_equal : pole_shadow = 84 ∧ tree_shadow = 32 ∧ tree_height = 28) : 
                       round (tree_height * (pole_shadow / tree_shadow)) = 74 :=
by
  sorry

end height_of_pole_l182_182391


namespace max_value_a_l182_182347

theorem max_value_a (a : ℝ) : 
  (∀ x : ℝ, x > -1 → x + 1 > 0 → x + 1 + 1 / (x + 1) - 2 ≥ a) → a ≤ 0 :=
by
  -- Proof omitted
  sorry

end max_value_a_l182_182347


namespace symmetric_line_l182_182004

theorem symmetric_line (y : ℝ → ℝ) (h : ∀ x, y x = 2 * x + 1) :
  ∀ x, y (-x) = -2 * x + 1 :=
by
  -- Proof skipped
  sorry

end symmetric_line_l182_182004


namespace period_2_students_l182_182239

theorem period_2_students (x : ℕ) (h1 : 2 * x - 5 = 11) : x = 8 :=
by {
  sorry
}

end period_2_students_l182_182239


namespace quadratic_intersects_x_axis_l182_182778

theorem quadratic_intersects_x_axis (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 7 * x - 7 = 0) ↔ k ≥ -7/4 ∧ k ≠ 0 :=
by
  sorry

end quadratic_intersects_x_axis_l182_182778


namespace probability_white_given_black_drawn_l182_182612

-- Definitions based on the conditions
def num_white : ℕ := 3
def num_black : ℕ := 2
def total_balls : ℕ := num_white + num_black

def P (n : ℕ) : ℚ := n / total_balls

-- Event A: drawing a black ball on the first draw
def PA : ℚ := P num_black

-- Event B: drawing a white ball on the second draw
def PB_given_A : ℚ := num_white / (total_balls - 1)

-- Theorem statement
theorem probability_white_given_black_drawn :
  (PA * PB_given_A) / PA = 3 / 4 :=
by
  sorry

end probability_white_given_black_drawn_l182_182612


namespace number_of_neighborhoods_l182_182655

def street_lights_per_side : ℕ := 250
def roads_per_neighborhood : ℕ := 4
def total_street_lights : ℕ := 20000

theorem number_of_neighborhoods : 
  (total_street_lights / (2 * street_lights_per_side * roads_per_neighborhood)) = 10 :=
by
  -- proof to show that the number of neighborhoods is 10
  sorry

end number_of_neighborhoods_l182_182655


namespace amount_paid_is_correct_l182_182843

-- Define the conditions
def time_painting_house : ℕ := 8
def time_fixing_counter := 3 * time_painting_house
def time_mowing_lawn : ℕ := 6
def hourly_rate : ℕ := 15

-- Define the total time worked
def total_time_worked := time_painting_house + time_fixing_counter + time_mowing_lawn

-- Define the total amount paid
def total_amount_paid := total_time_worked * hourly_rate

-- Formalize the goal
theorem amount_paid_is_correct : total_amount_paid = 570 :=
by
  -- Proof steps to be filled in
  sorry  -- Placeholder for the proof

end amount_paid_is_correct_l182_182843


namespace sum_of_asymptotes_l182_182088

theorem sum_of_asymptotes :
  let c := -3/2
  let d := -1
  c + d = -5/2 :=
by
  -- Definitions corresponding to the problem conditions
  let c := -3/2
  let d := -1
  -- Statement of the theorem
  show c + d = -5/2
  sorry

end sum_of_asymptotes_l182_182088


namespace a_pow_11_b_pow_11_l182_182605

theorem a_pow_11_b_pow_11 (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^11 + b^11 = 199 :=
by
  sorry

end a_pow_11_b_pow_11_l182_182605


namespace amelia_jet_bars_l182_182571

theorem amelia_jet_bars
    (required : ℕ) (sold_monday : ℕ) (sold_tuesday_less : ℕ) (total_sold : ℕ) (remaining : ℕ) :
    required = 90 →
    sold_monday = 45 →
    sold_tuesday_less = 16 →
    total_sold = sold_monday + (sold_monday - sold_tuesday_less) →
    remaining = required - total_sold →
    remaining = 16 :=
by
  intros
  sorry

end amelia_jet_bars_l182_182571


namespace probability_all_same_color_l182_182784

def total_marbles := 15
def red_marbles := 4
def white_marbles := 5
def blue_marbles := 6

def prob_all_red := (red_marbles / total_marbles) * ((red_marbles - 1) / (total_marbles - 1)) * ((red_marbles - 2) / (total_marbles - 2))
def prob_all_white := (white_marbles / total_marbles) * ((white_marbles - 1) / (total_marbles - 1)) * ((white_marbles - 2) / (total_marbles - 2))
def prob_all_blue := (blue_marbles / total_marbles) * ((blue_marbles - 1) / (total_marbles - 1)) * ((blue_marbles - 2) / (total_marbles - 2))

def prob_all_same_color := prob_all_red + prob_all_white + prob_all_blue

theorem probability_all_same_color :
  prob_all_same_color = (34/455) :=
by sorry

end probability_all_same_color_l182_182784


namespace jar_and_beans_weight_is_60_percent_l182_182358

theorem jar_and_beans_weight_is_60_percent
  (J B : ℝ)
  (h1 : J = 0.10 * (J + B))
  (h2 : ∃ x : ℝ, x = 0.5555555555555556 ∧ (J + x * B = 0.60 * (J + B))) :
  J + 0.5555555555555556 * B = 0.60 * (J + B) :=
by
  sorry

end jar_and_beans_weight_is_60_percent_l182_182358


namespace range_of_a_l182_182816

open Set

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x → x ≤ 3 → x^2 - a * x - a + 1 ≥ 0) ↔ a ≤ 5 / 2 :=
sorry

end range_of_a_l182_182816


namespace average_age_is_25_l182_182879

theorem average_age_is_25 (A B C : ℝ) (h_avg_ac : (A + C) / 2 = 29) (h_b : B = 17) :
  (A + B + C) / 3 = 25 := 
  by
    sorry

end average_age_is_25_l182_182879


namespace a_3_def_a_4_def_a_r_recurrence_l182_182481

-- Define minimally the structure of the problem.
noncomputable def a_r (r : ℕ) : ℕ := -- Definition for minimum phone calls required.
by sorry

-- Assertions for the specific cases provided.
theorem a_3_def : a_r 3 = 3 :=
by
  -- Proof is omitted with sorry.
  sorry

theorem a_4_def : a_r 4 = 4 :=
by
  -- Proof is omitted with sorry.
  sorry

theorem a_r_recurrence (r : ℕ) (hr : r ≥ 3) : a_r r ≤ a_r (r - 1) + 2 :=
by
  -- Proof is omitted with sorry.
  sorry

end a_3_def_a_4_def_a_r_recurrence_l182_182481


namespace mo_hot_chocolate_l182_182854

noncomputable def cups_of_hot_chocolate (total_drinks: ℕ) (extra_tea: ℕ) (non_rainy_days: ℕ) (tea_per_day: ℕ) : ℕ :=
  let tea_drinks := non_rainy_days * tea_per_day 
  let chocolate_drinks := total_drinks - tea_drinks 
  (extra_tea - chocolate_drinks)

theorem mo_hot_chocolate :
  cups_of_hot_chocolate 36 14 5 5 = 11 :=
by
  sorry

end mo_hot_chocolate_l182_182854


namespace xy_sum_is_2_l182_182698

theorem xy_sum_is_2 (x y : ℝ) (h : 4 * x^2 + 4 * y^2 = 40 * x - 24 * y + 64) : x + y = 2 := 
by
  sorry

end xy_sum_is_2_l182_182698


namespace least_possible_value_of_c_l182_182884

theorem least_possible_value_of_c (a b c : ℕ) 
  (h1 : a + b + c = 60) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : b = a + 13) : c = 45 :=
sorry

end least_possible_value_of_c_l182_182884


namespace determine_a_plus_b_l182_182340

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b
noncomputable def f_inv (a b x : ℝ) : ℝ := b * x^2 + a

theorem determine_a_plus_b (a b : ℝ) (h: ∀ x : ℝ, f a b (f_inv a b x) = x) : a + b = 1 :=
sorry

end determine_a_plus_b_l182_182340


namespace problem_l182_182664

-- Given conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f (x)

variables (f : ℝ → ℝ)
variables (h_odd : is_odd_function f)
variables (h_f1 : f 1 = 5)
variables (h_period : ∀ x, f (x + 4) = -f x)

-- Prove that f(2012) + f(2015) = -5
theorem problem :
  f 2012 + f 2015 = -5 :=
sorry

end problem_l182_182664


namespace functional_equation_g_l182_182684

variable (g : ℝ → ℝ)
variable (f : ℝ)
variable (h : ℝ)

theorem functional_equation_g (H1 : ∀ x y : ℝ, g (x + y) = g x * g y)
                            (H2 : g 3 = 4) :
                            g 6 = 16 := 
by
  sorry

end functional_equation_g_l182_182684


namespace quadrilateral_area_inequality_l182_182011

theorem quadrilateral_area_inequality (a b c d : ℝ) :
  ∃ (S_ABCD : ℝ), S_ABCD ≤ (1 / 4) * (a + c) ^ 2 + b * d :=
sorry

end quadrilateral_area_inequality_l182_182011


namespace element_in_set_l182_182713

theorem element_in_set : 1 ∈ ({0, 1} : Set ℕ) := 
by 
  -- Proof goes here
  sorry

end element_in_set_l182_182713


namespace combined_experience_l182_182401

noncomputable def james_experience : ℕ := 20
noncomputable def john_experience_8_years_ago : ℕ := 2 * (james_experience - 8)
noncomputable def john_current_experience : ℕ := john_experience_8_years_ago + 8
noncomputable def mike_experience : ℕ := john_current_experience - 16

theorem combined_experience :
  james_experience + john_current_experience + mike_experience = 68 :=
by
  sorry

end combined_experience_l182_182401


namespace ways_to_select_at_least_one_defective_l182_182023

open Finset

-- Define basic combinatorial selection functions
def combination (n k : ℕ) := Nat.choose n k

-- Given conditions
def total_products : ℕ := 100
def defective_products : ℕ := 6
def selected_products : ℕ := 3
def non_defective_products : ℕ := total_products - defective_products

-- The question to prove: the number of ways to select at least one defective product
theorem ways_to_select_at_least_one_defective :
  (combination total_products selected_products) - (combination non_defective_products selected_products) =
  (combination 100 3) - (combination 94 3) := by
  sorry

end ways_to_select_at_least_one_defective_l182_182023


namespace cost_to_fill_pool_l182_182758

-- Definitions based on the conditions
def cubic_foot_to_liters: ℕ := 25
def pool_depth: ℕ := 10
def pool_width: ℕ := 6
def pool_length: ℕ := 20
def cost_per_liter: ℕ := 3

-- Statement to be proved
theorem cost_to_fill_pool : 
  (pool_depth * pool_width * pool_length * cubic_foot_to_liters * cost_per_liter) = 90000 := 
by 
  sorry

end cost_to_fill_pool_l182_182758


namespace probability_sum_greater_than_9_l182_182287

def num_faces := 6
def total_outcomes := num_faces * num_faces
def favorable_outcomes := 6
def probability := favorable_outcomes / total_outcomes

theorem probability_sum_greater_than_9 (h : total_outcomes = 36) :
  probability = 1 / 6 :=
by
  sorry

end probability_sum_greater_than_9_l182_182287


namespace sandy_total_money_l182_182868

-- Definitions based on conditions
def X_initial (X : ℝ) : Prop := 
  X - 0.30 * X = 210

def watch_cost : ℝ := 50

-- Question translated into a proof goal
theorem sandy_total_money (X : ℝ) (h : X_initial X) : 
  X + watch_cost = 350 := by
  sorry

end sandy_total_money_l182_182868


namespace product_of_numbers_l182_182300

theorem product_of_numbers :
  ∃ (x y z : ℚ), (x + y + z = 30) ∧ (x = 3 * (y + z)) ∧ (y = 5 * z) ∧ (x * y * z = 175.78125) :=
by
  sorry

end product_of_numbers_l182_182300


namespace circle_graph_to_bar_graph_correct_l182_182138

theorem circle_graph_to_bar_graph_correct :
  ∀ (white black gray blue : ℚ) (w_proportion b_proportion g_proportion blu_proportion : ℚ),
    white = 1/2 →
    black = 1/4 →
    gray = 1/8 →
    blue = 1/8 →
    w_proportion = 1/2 →
    b_proportion = 1/4 →
    g_proportion = 1/8 →
    blu_proportion = 1/8 →
    white = w_proportion ∧ black = b_proportion ∧ gray = g_proportion ∧ blue = blu_proportion :=
by
sorry

end circle_graph_to_bar_graph_correct_l182_182138


namespace river_width_l182_182923

theorem river_width (depth : ℝ) (flow_rate_kmph : ℝ) (volume_per_minute : ℝ) : depth = 5 → flow_rate_kmph = 2 → volume_per_minute = 5833.333333333333 → 
  (volume_per_minute / ((flow_rate_kmph * 1000 / 60) * depth) = 35) :=
by 
  intros h_depth h_flow_rate h_volume
  sorry

end river_width_l182_182923


namespace g_three_fifths_l182_182372

-- Given conditions
variable (g : ℝ → ℝ)
variable (h₀ : g 0 = 0)
variable (h₁ : ∀ ⦃x y : ℝ⦄, 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y)
variable (h₂ : ∀ ⦃x : ℝ⦄, 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x)
variable (h₃ : ∀ ⦃x : ℝ⦄, 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3)

-- Proof statement
theorem g_three_fifths : g (3 / 5) = 2 / 3 := by
  sorry

end g_three_fifths_l182_182372


namespace find_x_value_l182_182137

theorem find_x_value (x : ℝ) (a b c : ℝ × ℝ × ℝ) 
  (h_a : a = (1, 1, x)) 
  (h_b : b = (1, 2, 1)) 
  (h_c : c = (1, 1, 1)) 
  (h_cond : (c - a) • (2 • b) = -2) : 
  x = 2 := 
by 
  -- the proof goes here
  sorry

end find_x_value_l182_182137


namespace roses_in_centerpiece_l182_182065

variable (r : ℕ)

theorem roses_in_centerpiece (h : 6 * 15 * (3 * r + 6) = 2700) : r = 8 := 
  sorry

end roses_in_centerpiece_l182_182065


namespace greatest_integer_with_gcd_6_l182_182078

theorem greatest_integer_with_gcd_6 (n : ℕ) (h1 : n < 150) (h2 : Int.gcd n 18 = 6) : n = 138 :=
sorry

end greatest_integer_with_gcd_6_l182_182078


namespace teddy_bears_count_l182_182516

theorem teddy_bears_count (toys_count : ℕ) (toy_cost : ℕ) (total_money : ℕ) (teddy_bear_cost : ℕ) : 
  toys_count = 28 → 
  toy_cost = 10 → 
  total_money = 580 → 
  teddy_bear_cost = 15 →
  ((total_money - toys_count * toy_cost) / teddy_bear_cost) = 20 :=
by
  intros h1 h2 h3 h4
  sorry

end teddy_bears_count_l182_182516


namespace percent_birth_month_in_march_l182_182590

theorem percent_birth_month_in_march (total_people : ℕ) (march_births : ℕ) (h1 : total_people = 100) (h2 : march_births = 8) : (march_births * 100 / total_people) = 8 := by
  sorry

end percent_birth_month_in_march_l182_182590


namespace rotate_and_translate_line_l182_182724

theorem rotate_and_translate_line :
  let initial_line (x : ℝ) := 3 * x
  let rotated_line (x : ℝ) := - (1 / 3) * x
  let translated_line (x : ℝ) := - (1 / 3) * (x - 1)

  ∀ x : ℝ, translated_line x = - (1 / 3) * x + (1 / 3) := 
by
  intros
  simp
  sorry

end rotate_and_translate_line_l182_182724


namespace number_of_classes_l182_182837

theorem number_of_classes
  (p : ℕ) (s : ℕ) (t : ℕ) (c : ℕ)
  (hp : p = 2) (hs : s = 30) (ht : t = 360) :
  c = t / (p * s) :=
by
  simp [hp, hs, ht]
  sorry

end number_of_classes_l182_182837


namespace average_of_11_numbers_l182_182487

theorem average_of_11_numbers (a b c d e f g h i j k : ℝ)
  (h_first_6_avg : (a + b + c + d + e + f) / 6 = 98)
  (h_last_6_avg : (f + g + h + i + j + k) / 6 = 65)
  (h_6th_number : f = 318) :
  ((a + b + c + d + e + f + g + h + i + j + k) / 11) = 60 :=
by
  sorry

end average_of_11_numbers_l182_182487


namespace num_zeros_in_expansion_l182_182121

noncomputable def bigNum := (10^11 - 2) ^ 2

theorem num_zeros_in_expansion : ∀ n : ℕ, bigNum = n ↔ (n = 9999999999900000000004) := sorry

end num_zeros_in_expansion_l182_182121


namespace midpoint_of_points_l182_182076

theorem midpoint_of_points (x1 y1 x2 y2 : ℝ) (h1 : x1 = 6) (h2 : y1 = 10) (h3 : x2 = 8) (h4 : y2 = 4) :
  ((x1 + x2) / 2, (y1 + y2) / 2) = (7, 7) := 
by
  rw [h1, h2, h3, h4]
  norm_num

end midpoint_of_points_l182_182076


namespace arithmetic_sequence_common_difference_l182_182984

theorem arithmetic_sequence_common_difference (a : ℕ → ℤ) (d : ℤ) (h1 : a 3 = 7) (h2 : a 7 = -5)
  (h3 : ∀ n, a (n + 1) = a n + d) : 
  d = -3 :=
sorry

end arithmetic_sequence_common_difference_l182_182984


namespace sequence_bound_l182_182135

theorem sequence_bound
  (a : ℕ → ℕ)
  (h_base0 : a 0 < a 1)
  (h_base1 : 0 < a 0 ∧ 0 < a 1)
  (h_recur : ∀ n, 2 ≤ n → a n = 3 * a (n - 1) - 2 * a (n - 2)) :
  a 100 > 2^99 :=
by
  sorry

end sequence_bound_l182_182135


namespace term_number_l182_182410

theorem term_number (n : ℕ) : 
  (n ≥ 1) ∧ (5 * Real.sqrt 3 = Real.sqrt (3 + 4 * (n - 1))) → n = 19 :=
by
  intro h
  let h1 := h.1
  let h2 := h.2
  have h3 : (5 * Real.sqrt 3)^2 = (Real.sqrt (3 + 4 * (n - 1)))^2 := by sorry
  sorry

end term_number_l182_182410


namespace power_calc_l182_182977

theorem power_calc : (3^2)^4 = 6561 := 
by
  sorry

end power_calc_l182_182977


namespace asymptotes_of_hyperbola_l182_182233

theorem asymptotes_of_hyperbola (a : ℝ) :
  (∃ a : ℝ, 9 + a = 13) →
  (∀ x y : ℝ, (x^2 / 9 - y^2 / a = 1) → (a = 4)) →
  (forall (x y : ℝ), (x^2 / 9 - y^2 / 4 = 0) → 
    (y = (2/3) * x) ∨ (y = -(2/3) * x)) :=
by
  sorry

end asymptotes_of_hyperbola_l182_182233


namespace find_value_of_expression_l182_182313

theorem find_value_of_expression 
  (x y z w : ℤ)
  (hx : x = 3)
  (hy : y = 2)
  (hz : z = 4)
  (hw : w = -1) :
  x^2 * y - 2 * x * y + 3 * x * z - (x + y) * (y + z) * (z + w) = -48 :=
by
  sorry

end find_value_of_expression_l182_182313


namespace cost_of_one_shirt_l182_182789

theorem cost_of_one_shirt
  (cost_J : ℕ)  -- The cost of one pair of jeans
  (cost_S : ℕ)  -- The cost of one shirt
  (h1 : 3 * cost_J + 2 * cost_S = 69)
  (h2 : 2 * cost_J + 3 * cost_S = 81) :
  cost_S = 21 :=
by
  sorry

end cost_of_one_shirt_l182_182789


namespace more_tails_than_heads_l182_182992

def total_flips : ℕ := 211
def heads_flips : ℕ := 65
def tails_flips : ℕ := total_flips - heads_flips

theorem more_tails_than_heads : tails_flips - heads_flips = 81 := by
  -- proof is unnecessary according to the instructions
  sorry

end more_tails_than_heads_l182_182992


namespace f_prime_at_zero_l182_182431

-- Lean definition of the conditions.
def a (n : ℕ) : ℝ := 2 * (2 ^ (1/7)) ^ (n - 1)

-- The function f(x) based on the given conditions.
noncomputable def f (x : ℝ) : ℝ := 
  x * (x - a 1) * (x - a 2) * (x - a 3) * (x - a 4) * 
  (x - a 5) * (x - a 6) * (x - a 7) * (x - a 8)

-- The main goal to prove: f'(0) = 2^12
theorem f_prime_at_zero : deriv f 0 = 2^12 := by
  sorry

end f_prime_at_zero_l182_182431


namespace last_two_digits_A_pow_20_l182_182318

/-- 
Proof that for any even number A not divisible by 10, 
the last two digits of A^20 are 76.
--/
theorem last_two_digits_A_pow_20 (A : ℕ) (h_even : A % 2 = 0) (h_not_div_by_10 : A % 10 ≠ 0) : 
  (A ^ 20) % 100 = 76 :=
by
  sorry

end last_two_digits_A_pow_20_l182_182318


namespace range_of_b_over_a_l182_182203

noncomputable def f (a b x : ℝ) : ℝ := (x - a)^3 * (x - b)
noncomputable def g_k (a b k x : ℝ) : ℝ := (f a b x - f a b k) / (x - k)

theorem range_of_b_over_a (a b : ℝ) (h₀ : 0 < a) (h₁ : a < b) (h₂ : b < 1)
    (hk_inc : ∀ k : ℤ, ∀ x : ℝ, k < x → g_k a b k x ≥ g_k a b k (k + 1)) :
  1 < b / a ∧ b / a ≤ 3 :=
by
  sorry


end range_of_b_over_a_l182_182203


namespace tetrahedron_volume_eq_three_l182_182939

noncomputable def volume_of_tetrahedron : ℝ :=
  let PQ := 3
  let PR := 4
  let PS := 5
  let QR := 5
  let QS := Real.sqrt 34
  let RS := Real.sqrt 41
  have := (PQ = 3) ∧ (PR = 4) ∧ (PS = 5) ∧ (QR = 5) ∧ (QS = Real.sqrt 34) ∧ (RS = Real.sqrt 41)
  3

theorem tetrahedron_volume_eq_three : volume_of_tetrahedron = 3 := 
by { sorry }

end tetrahedron_volume_eq_three_l182_182939


namespace rate_per_sq_meter_l182_182377

theorem rate_per_sq_meter (length width : ℝ) (total_cost : ℝ) (h1 : length = 5.5) (h2 : width = 3.75) (h3 : total_cost = 16500) : 
  total_cost / (length * width) = 800 :=
by
  sorry

end rate_per_sq_meter_l182_182377


namespace trip_distance_l182_182905

theorem trip_distance (D : ℝ) (t1 t2 : ℝ) :
  (30 / 60 = t1) →
  (70 / 35 = t2) →
  (t1 + t2 = 2.5) →
  (40 = D / (t1 + t2)) →
  D = 100 :=
by
  intros h1 h2 h3 h4
  sorry

end trip_distance_l182_182905


namespace exists_visible_point_l182_182197

open Nat -- to use natural numbers and their operations

def is_visible (x y : ℤ) : Prop :=
  Int.gcd x y = 1

theorem exists_visible_point (n : ℕ) (hn : n > 0) :
  ∃ a b : ℤ, is_visible a b ∧
  ∀ (P : ℤ × ℤ), (P ≠ (a, b) → (Int.sqrt ((P.fst - a) * (P.fst - a) + (P.snd - b) * (P.snd - b)) > n)) :=
sorry

end exists_visible_point_l182_182197


namespace last_person_is_knight_l182_182539

def KnightLiarsGame1 (n : ℕ) : Prop :=
  let m := 10
  let p := 13
  (∀ i < n, (i % 2 = 0 → true) ∧ (i % 2 = 1 → true)) ∧ 
  (m % 2 ≠ p % 2 → (n - 1) % 2 = 1)

def KnightLiarsGame2 (n : ℕ) : Prop :=
  let m := 12
  let p := 9
  (∀ i < n, (i % 2 = 0 → true) ∧ (i % 2 = 1 → true)) ∧ 
  (m % 2 ≠ p % 2 → (n - 1) % 2 = 1)

theorem last_person_is_knight :
  ∃ n, KnightLiarsGame1 n ∧ KnightLiarsGame2 n :=
by 
  sorry

end last_person_is_knight_l182_182539


namespace find_ethanol_percentage_l182_182856

noncomputable def ethanol_percentage_in_fuel_A (P_A : ℝ) (V_A : ℝ) : Prop :=
  (P_A / 100) * V_A + 0.16 * (200 - V_A) = 18

theorem find_ethanol_percentage (P_A : ℝ) (V_A : ℝ) (h₀ : V_A ≤ 200) (h₁ : 0 ≤ V_A) :
  ethanol_percentage_in_fuel_A P_A V_A :=
by
  sorry

end find_ethanol_percentage_l182_182856


namespace clock_angle_7_35_l182_182105

noncomputable def hour_angle (hours : ℤ) (minutes : ℤ) : ℝ :=
  (hours * 30 + (minutes * 30) / 60 : ℝ)

noncomputable def minute_angle (minutes : ℤ) : ℝ :=
  (minutes * 360 / 60 : ℝ)

noncomputable def angle_between (angle1 angle2 : ℝ) : ℝ :=
  abs (angle1 - angle2)

theorem clock_angle_7_35 : angle_between (hour_angle 7 35) (minute_angle 35) = 17.5 :=
by
  sorry

end clock_angle_7_35_l182_182105


namespace imo1983_q6_l182_182333

theorem imo1983_q6 (a b c : ℝ) (h : a + b > c) (h2 : a + c > b) (h3 : b + c > a) : 
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 :=
by
  sorry

end imo1983_q6_l182_182333


namespace simplify_expr1_simplify_expr2_simplify_expr3_l182_182099

theorem simplify_expr1 : -2.48 + 4.33 + (-7.52) + (-4.33) = -10 := by
  sorry

theorem simplify_expr2 : (7/13) * (-9) + (7/13) * (-18) + (7/13) = -14 := by
  sorry

theorem simplify_expr3 : -((20 + 1/19) * 38) = -762 := by
  sorry

end simplify_expr1_simplify_expr2_simplify_expr3_l182_182099


namespace total_growing_space_correct_l182_182673

-- Define the dimensions of the garden beds
def length_bed1 : ℕ := 3
def width_bed1 : ℕ := 3
def num_bed1 : ℕ := 2

def length_bed2 : ℕ := 4
def width_bed2 : ℕ := 3
def num_bed2 : ℕ := 2

-- Define the areas of the individual beds and total growing space
def area_bed1 : ℕ := length_bed1 * width_bed1
def total_area_bed1 : ℕ := area_bed1 * num_bed1

def area_bed2 : ℕ := length_bed2 * width_bed2
def total_area_bed2 : ℕ := area_bed2 * num_bed2

def total_growing_space : ℕ := total_area_bed1 + total_area_bed2

-- The theorem proving the total growing space
theorem total_growing_space_correct : total_growing_space = 42 := by
  sorry

end total_growing_space_correct_l182_182673


namespace total_points_zach_ben_l182_182532

theorem total_points_zach_ben (zach_points ben_points : ℝ) (h1 : zach_points = 42.0) (h2 : ben_points = 21.0) : zach_points + ben_points = 63.0 :=
by
  sorry

end total_points_zach_ben_l182_182532


namespace hyperbola_min_focal_asymptote_eq_l182_182972

theorem hyperbola_min_focal_asymptote_eq {x y m : ℝ}
  (h1 : -2 ≤ m)
  (h2 : m < 0)
  (h_eq : x^2 / m^2 - y^2 / (2 * m + 6) = 1)
  (h_min_focal : m = -1) :
  y = 2 * x ∨ y = -2 * x :=
by
  sorry

end hyperbola_min_focal_asymptote_eq_l182_182972


namespace product_not_end_in_1999_l182_182958

theorem product_not_end_in_1999 (a b c d e : ℕ) (h : a + b + c + d + e = 200) : 
  ¬(a * b * c * d * e % 10000 = 1999) := 
by
  sorry

end product_not_end_in_1999_l182_182958


namespace sports_club_problem_l182_182951

theorem sports_club_problem (total_members : ℕ) (members_playing_badminton : ℕ) 
  (members_playing_tennis : ℕ) (members_not_playing_either : ℕ) 
  (h_total_members : total_members = 100) (h_badminton : members_playing_badminton = 60) 
  (h_tennis : members_playing_tennis = 70) (h_neither : members_not_playing_either = 10) : 
  (members_playing_badminton + members_playing_tennis - 
   (total_members - members_not_playing_either) = 40) :=
by {
  sorry
}

end sports_club_problem_l182_182951


namespace solve_for_x_l182_182615

theorem solve_for_x : ∃ x : ℝ, (9 - x) ^ 2 = x ^ 2 ∧ x = 4.5 :=
by
  sorry

end solve_for_x_l182_182615


namespace smallest_prime_angle_l182_182194

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem smallest_prime_angle :
  ∃ (x : ℕ), is_prime x ∧ is_prime (2 * x) ∧ x + 2 * x = 90 ∧ x = 29 :=
by sorry

end smallest_prime_angle_l182_182194


namespace solution_set_inequality_l182_182315

theorem solution_set_inequality (x : ℝ) :
  (x^2 - 4) * (x - 6)^2 ≤ 0 ↔ (-2 ≤ x ∧ x ≤ 2) ∨ (x = 6) := by
  sorry

end solution_set_inequality_l182_182315


namespace bridge_crossing_possible_l182_182651

/-- 
  There are four people A, B, C, and D. 
  The time it takes for each of them to cross the bridge is 2, 4, 6, and 8 minutes respectively.
  No more than two people can be on the bridge at the same time.
  Prove that it is possible for all four people to cross the bridge in 10 minutes.
--/
theorem bridge_crossing_possible : 
  ∃ (cross : ℕ → ℕ), 
  cross 1 = 2 ∧ cross 2 = 4 ∧ cross 3 = 6 ∧ cross 4 = 8 ∧
  (∀ (t : ℕ), t ≤ 2 → cross 1 + cross 2 + cross 3 + cross 4 = 10) :=
by
  sorry

end bridge_crossing_possible_l182_182651


namespace cylinder_projections_tangency_l182_182164

def plane1 : Type := sorry
def plane2 : Type := sorry
def projection_axis : Type := sorry
def is_tangent_to (cylinder : Type) (plane : Type) : Prop := sorry
def is_base_tangent_to (cylinder : Type) (axis : Type) : Prop := sorry
def cylinder : Type := sorry

theorem cylinder_projections_tangency (P1 P2 : Type) (axis : Type)
  (h1 : is_tangent_to cylinder P1) 
  (h2 : is_tangent_to cylinder P2) 
  (h3 : is_base_tangent_to cylinder axis) : 
  ∃ (solutions : ℕ), solutions = 4 :=
sorry

end cylinder_projections_tangency_l182_182164


namespace min_shift_for_even_function_l182_182029

theorem min_shift_for_even_function :
  ∃ (m : ℝ), (m > 0) ∧ (∀ x : ℝ, (Real.sin (x + m) + Real.cos (x + m)) = (Real.sin (-x + m) + Real.cos (-x + m))) ∧ m = π / 4 :=
by
  sorry

end min_shift_for_even_function_l182_182029


namespace add_water_to_solution_l182_182981

noncomputable def current_solution_volume : ℝ := 300
noncomputable def desired_water_percentage : ℝ := 0.70
noncomputable def current_water_volume : ℝ := 0.60 * current_solution_volume
noncomputable def current_acid_volume : ℝ := 0.40 * current_solution_volume

theorem add_water_to_solution (x : ℝ) : 
  (current_water_volume + x) / (current_solution_volume + x) = desired_water_percentage ↔ x = 100 :=
by
  sorry

end add_water_to_solution_l182_182981


namespace arc_length_TQ_l182_182166

-- Definitions from the conditions
def center (O : Type) : Prop := true
def inscribedAngle (T I Q : Type) (angle : ℝ) := angle = 45
def radius (T : Type) (len : ℝ) := len = 12

-- Theorem to be proved
theorem arc_length_TQ (O T I Q : Type) (r : ℝ) (angle : ℝ) 
  (h_center : center O) 
  (h_angle : inscribedAngle T I Q angle)
  (h_radius : radius T r) :
  ∃ l : ℝ, l = 6 * Real.pi := 
sorry

end arc_length_TQ_l182_182166


namespace remaining_area_is_correct_l182_182266

-- Define the large rectangle's side lengths
def large_rectangle_length1 (x : ℝ) := x + 7
def large_rectangle_length2 (x : ℝ) := x + 5

-- Define the hole's side lengths
def hole_length1 (x : ℝ) := x + 1
def hole_length2 (x : ℝ) := x + 4

-- Calculate the areas
def large_rectangle_area (x : ℝ) := large_rectangle_length1 x * large_rectangle_length2 x
def hole_area (x : ℝ) := hole_length1 x * hole_length2 x

-- Define the remaining area after subtracting the hole area from the large rectangle area
def remaining_area (x : ℝ) := large_rectangle_area x - hole_area x

-- Problem statement: prove that the remaining area is 7x + 31
theorem remaining_area_is_correct (x : ℝ) : remaining_area x = 7 * x + 31 :=
by 
  -- The proof should be provided here, but for now we use 'sorry' to omit it
  sorry

end remaining_area_is_correct_l182_182266


namespace largest_of_set_l182_182524

theorem largest_of_set : 
  let a := 1 / 2
  let b := -1
  let c := abs (-2)
  let d := -3
  c = 2 ∧ (d < b ∧ b < a ∧ a < c) := by
  let a := 1 / 2
  let b := -1
  let c := abs (-2)
  let d := -3
  sorry

end largest_of_set_l182_182524


namespace total_amount_shared_l182_182443

theorem total_amount_shared (z : ℝ) (hz : z = 150) (hy : y = 1.20 * z) (hx : x = 1.25 * y) : 
  x + y + z = 555 :=
by
  sorry

end total_amount_shared_l182_182443


namespace probability_sum_less_than_product_l182_182787

def set_of_even_integers : Set ℕ := {2, 4, 6, 8, 10}

def sum_less_than_product (a b : ℕ) : Prop :=
  a + b < a * b

theorem probability_sum_less_than_product :
  let total_combinations := 25
  let valid_combinations := 16
  (valid_combinations / total_combinations : ℚ) = 16 / 25 :=
by
  sorry

end probability_sum_less_than_product_l182_182787


namespace cost_of_socks_l182_182754

theorem cost_of_socks (S : ℝ) (players : ℕ) (jersey : ℝ) (shorts : ℝ) 
                      (total_cost : ℝ) 
                      (h1 : players = 16) 
                      (h2 : jersey = 25) 
                      (h3 : shorts = 15.20) 
                      (h4 : total_cost = 752) 
                      (h5 : total_cost = players * (jersey + shorts + S)) 
                      : S = 6.80 := 
by
  sorry

end cost_of_socks_l182_182754


namespace billy_total_tickets_l182_182930

theorem billy_total_tickets :
  let ferris_wheel_rides := 7
  let bumper_car_rides := 3
  let roller_coaster_rides := 4
  let teacups_rides := 5
  let ferris_wheel_cost := 5
  let bumper_car_cost := 6
  let roller_coaster_cost := 8
  let teacups_cost := 4
  let total_ferris_wheel := ferris_wheel_rides * ferris_wheel_cost
  let total_bumper_cars := bumper_car_rides * bumper_car_cost
  let total_roller_coaster := roller_coaster_rides * roller_coaster_cost
  let total_teacups := teacups_rides * teacups_cost
  let total_tickets := total_ferris_wheel + total_bumper_cars + total_roller_coaster + total_teacups
  total_tickets = 105 := 
sorry

end billy_total_tickets_l182_182930


namespace fraction_sum_equals_decimal_l182_182927

theorem fraction_sum_equals_decimal : 
  (3 / 30 + 9 / 300 + 27 / 3000 = 0.139) :=
by sorry

end fraction_sum_equals_decimal_l182_182927


namespace divisible_bc_ad_l182_182761

open Int

theorem divisible_bc_ad (a b c d m : ℤ) (hm : 0 < m)
  (h1 : m ∣ a * c)
  (h2 : m ∣ b * d)
  (h3 : m ∣ (b * c + a * d)) :
  m ∣ b * c ∧ m ∣ a * d :=
by
  sorry

end divisible_bc_ad_l182_182761


namespace coins_in_pockets_l182_182009

theorem coins_in_pockets : (Nat.choose (5 + 3 - 1) (3 - 1)) = 21 := by
  sorry

end coins_in_pockets_l182_182009


namespace max_and_min_sum_of_factors_of_2000_l182_182273

theorem max_and_min_sum_of_factors_of_2000 :
  ∃ (a b c d e : ℕ), 1 < a ∧ 1 < b ∧ 1 < c ∧ 1 < d ∧ 1 < e ∧ a * b * c * d * e = 2000
  ∧ (a + b + c + d + e = 133 ∨ a + b + c + d + e = 23) :=
by
  sorry

end max_and_min_sum_of_factors_of_2000_l182_182273


namespace smallest_number_l182_182798

theorem smallest_number (a b c d : ℤ) (h1 : a = 1) (h2 : b = 0) (h3 : c = -1) (h4 : d = -3) :
  d = -3 ∧ d < c ∧ d < b ∧ d < a :=
by
  sorry

end smallest_number_l182_182798


namespace group_d_forms_triangle_l182_182931

-- Definitions for the stick lengths in each group
def group_a := (1, 2, 6)
def group_b := (2, 2, 4)
def group_c := (1, 2, 3)
def group_d := (2, 3, 4)

-- Statement to prove that Group D can form a triangle
theorem group_d_forms_triangle (a b c : ℕ) : a = 2 → b = 3 → c = 4 → a + b > c ∧ a + c > b ∧ b + c > a := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  apply And.intro
  sorry
  apply And.intro
  sorry
  sorry

end group_d_forms_triangle_l182_182931


namespace find_a_l182_182341

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then (1/2)^x - 7 else x^2

theorem find_a (a : ℝ) (h : f a = 1) : a = -3 ∨ a = 1 := 
by
  sorry

end find_a_l182_182341


namespace pentagon_zero_impossible_l182_182271

theorem pentagon_zero_impossible
  (x : Fin 5 → ℝ)
  (h_sum : x 0 + x 1 + x 2 + x 3 + x 4 = 0)
  (operation : ∀ i : Fin 5, ∀ y : Fin 5 → ℝ,
    y i = (x i + x ((i + 1) % 5)) / 2 ∧ y ((i + 1) % 5) = (x i + x ((i + 1) % 5)) / 2) :
  ¬ ∃ (y : ℕ → (Fin 5 → ℝ)), ∃ N : ℕ, y N = 0 := 
sorry

end pentagon_zero_impossible_l182_182271


namespace part1_part2_l182_182560

-- Part (1)
theorem part1 (B : ℝ) (b : ℝ) (S : ℝ) (a c : ℝ) (B_eq : B = Real.pi / 3) 
  (b_eq : b = Real.sqrt 7) (S_eq : S = (3 * Real.sqrt 3) / 2) :
  a + c = 5 := 
sorry

-- Part (2)
theorem part2 (C : ℝ) (c : ℝ) (dot_BA_BC AB_AC : ℝ) 
  (C_cond : 2 * Real.cos C * (dot_BA_BC + AB_AC) = c^2) :
  C = Real.pi / 3 := 
sorry

end part1_part2_l182_182560


namespace PeytonManning_total_distance_l182_182596

noncomputable def PeytonManning_threw_distance : Prop :=
  let throw_distance_50 := 20
  let throw_times_sat := 20
  let throw_times_sun := 30
  let total_distance := 1600
  ∃ R : ℚ, 
    let throw_distance_80 := R * throw_distance_50
    let distance_sat := throw_distance_50 * throw_times_sat
    let distance_sun := throw_distance_80 * throw_times_sun
    distance_sat + distance_sun = total_distance

theorem PeytonManning_total_distance :
  PeytonManning_threw_distance := by
  sorry

end PeytonManning_total_distance_l182_182596


namespace sand_weight_proof_l182_182264

-- Definitions for the given conditions
def side_length : ℕ := 40
def bag_weight : ℕ := 30
def area_per_bag : ℕ := 80

-- Total area of the sandbox
def total_area := side_length * side_length

-- Number of bags needed
def number_of_bags := total_area / area_per_bag

-- Total weight of sand needed
def total_weight := number_of_bags * bag_weight

-- The proof statement
theorem sand_weight_proof :
  total_weight = 600 :=
by
  sorry

end sand_weight_proof_l182_182264


namespace obtuse_triangle_two_acute_angles_l182_182815

-- Define the angle type (could be Real between 0 and 180 in degrees).
def is_acute (θ : ℝ) : Prop := 0 < θ ∧ θ < 90

def is_obtuse (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

-- Define an obtuse triangle using three angles α, β, γ
structure obtuse_triangle :=
(angle1 angle2 angle3 : ℝ)
(sum_angles_eq : angle1 + angle2 + angle3 = 180)
(obtuse_condition : is_obtuse angle1 ∨ is_obtuse angle2 ∨ is_obtuse angle3)

-- The theorem to prove the number of acute angles in an obtuse triangle is 2.
theorem obtuse_triangle_two_acute_angles (T : obtuse_triangle) : 
  (is_acute T.angle1 ∧ is_acute T.angle2 ∧ ¬ is_acute T.angle3) ∨ 
  (is_acute T.angle1 ∧ ¬ is_acute T.angle2 ∧ is_acute T.angle3) ∨ 
  (¬ is_acute T.angle1 ∧ is_acute T.angle2 ∧ is_acute T.angle3) :=
by sorry

end obtuse_triangle_two_acute_angles_l182_182815


namespace gasoline_amount_added_l182_182711

noncomputable def initial_fill (capacity : ℝ) : ℝ := (3 / 4) * capacity
noncomputable def final_fill (capacity : ℝ) : ℝ := (9 / 10) * capacity
noncomputable def gasoline_added (capacity : ℝ) : ℝ := final_fill capacity - initial_fill capacity

theorem gasoline_amount_added :
  ∀ (capacity : ℝ), capacity = 24 → gasoline_added capacity = 3.6 :=
  by
    intros capacity h
    rw [h]
    have initial_fill_24 : initial_fill 24 = 18 := by norm_num [initial_fill]
    have final_fill_24 : final_fill 24 = 21.6 := by norm_num [final_fill]
    have gasoline_added_24 : gasoline_added 24 = 3.6 :=
      by rw [gasoline_added, initial_fill_24, final_fill_24]; norm_num
    exact gasoline_added_24

end gasoline_amount_added_l182_182711


namespace solve_log_equation_l182_182550

theorem solve_log_equation :
  ∀ x : ℝ, 
  5 * Real.logb x (x / 9) + Real.logb (x / 9) x^3 + 8 * Real.logb (9 * x^2) (x^2) = 2
  → (x = 3 ∨ x = Real.sqrt 3) := by
  sorry

end solve_log_equation_l182_182550


namespace cos_240_eq_neg_half_l182_182363

theorem cos_240_eq_neg_half (h1 : Real.cos (180 * Real.pi / 180) = -1)
                            (h2 : Real.sin (180 * Real.pi / 180) = 0)
                            (h3 : Real.cos (60 * Real.pi / 180) = 1 / 2) :
  Real.cos (240 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end cos_240_eq_neg_half_l182_182363


namespace associates_hired_l182_182968

variable (partners : ℕ) (associates initial_associates hired_associates : ℕ)
variable (initial_ratio : partners / initial_associates = 2 / 63)
variable (final_ratio : partners / (initial_associates + hired_associates) = 1 / 34)
variable (partners_count : partners = 18)

theorem associates_hired : hired_associates = 45 :=
by
  -- Insert solution steps here...
  sorry

end associates_hired_l182_182968


namespace segment_length_l182_182755

theorem segment_length (AB BC AC : ℝ) (hAB : AB = 4) (hBC : BC = 3) :
  AC = 7 ∨ AC = 1 :=
sorry

end segment_length_l182_182755


namespace geometric_sequence_l182_182178

open Nat

-- Define the sequence and conditions for the problem
variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}
variable {m p : ℕ}
variable (h1 : a 1 ≠ 0)
variable (h2 : ∀ n : ℕ, 2 * S (n + 1) - 3 * S n = 2 * a 1)
variable (h3 : S 0 = 0)
variable (h4 : ∀ n : ℕ, S (n + 1) = S n + a (n + 1))
variable (h5 : a 1 ≥ m^(p-1))
variable (h6 : a p ≤ (m+1)^(p-1))

-- The theorem that we need to prove
theorem geometric_sequence (n : ℕ) : 
  (exists r : ℕ → ℕ, ∀ k : ℕ, a (k + 1) = r (k + 1) * a k) ∧ 
  (∀ k : ℕ, a k = sorry) := sorry

end geometric_sequence_l182_182178


namespace true_false_questions_count_l182_182352

noncomputable def number_of_true_false_questions (T F M : ℕ) : Prop :=
  T + F + M = 45 ∧ M = 2 * F ∧ F = T + 7

theorem true_false_questions_count : ∃ T F M : ℕ, number_of_true_false_questions T F M ∧ T = 6 :=
by
  sorry

end true_false_questions_count_l182_182352


namespace find_speed_l182_182501

variables (x : ℝ) (V : ℝ)

def initial_speed (x : ℝ) (V : ℝ) : Prop := 
  let time_initial := x / V
  let time_second := (2 * x) / 20
  let total_distance := 3 * x
  let average_speed := 26.25
  average_speed = total_distance / (time_initial + time_second)

theorem find_speed (x : ℝ) (h : initial_speed x V) : V = 70 :=
by sorry

end find_speed_l182_182501


namespace triangle_is_equilateral_l182_182150

-- Define a triangle with angles A, B, and C
variables (A B C : ℝ)

-- The conditions of the problem
def log_sin_arithmetic_sequence : Prop :=
  Real.log (Real.sin A) + Real.log (Real.sin C) = 2 * Real.log (Real.sin B)

def angles_arithmetic_sequence : Prop :=
  2 * B = A + C

-- The theorem that the triangle is equilateral given these conditions
theorem triangle_is_equilateral :
  log_sin_arithmetic_sequence A B C → angles_arithmetic_sequence A B C → 
  A = 60 ∧ B = 60 ∧ C = 60 :=
by
  sorry

end triangle_is_equilateral_l182_182150


namespace new_tax_rate_l182_182903

-- Condition definitions
def previous_tax_rate : ℝ := 0.20
def initial_income : ℝ := 1000000
def new_income : ℝ := 1500000
def additional_taxes_paid : ℝ := 250000

-- Theorem statement
theorem new_tax_rate : 
  ∃ T : ℝ, 
    (new_income * T = initial_income * previous_tax_rate + additional_taxes_paid) ∧ 
    T = 0.30 :=
by sorry

end new_tax_rate_l182_182903


namespace problem_solution_set_l182_182678

-- Definitions and conditions according to the given problem
def odd_function_domain := {x : ℝ | x ≠ 0}
def function_condition1 (f : ℝ → ℝ) (x : ℝ) : Prop := x > 0 → deriv f x < (3 * f x) / x
def function_condition2 (f : ℝ → ℝ) : Prop := f 1 = 1 / 2
def function_condition3 (f : ℝ → ℝ) : Prop := ∀ x, f (2 * x) = 2 * f x

-- Main proof statement
theorem problem_solution_set (f : ℝ → ℝ)
  (odd_function : ∀ x, f (-x) = -f x)
  (dom : ∀ x, x ∈ odd_function_domain → f x ≠ 0)
  (cond1 : ∀ x, function_condition1 f x)
  (cond2 : function_condition2 f)
  (cond3 : function_condition3 f) :
  {x : ℝ | f x / (4 * x) < 2 * x^2} = {x : ℝ | x < -1 / 4} ∪ {x : ℝ | x > 1 / 4} :=
sorry

end problem_solution_set_l182_182678


namespace function_domain_l182_182028

noncomputable def domain_function (x : ℝ) : Prop :=
  x > 0 ∧ (Real.log x / Real.log 2)^2 - 1 > 0

theorem function_domain :
  { x : ℝ | domain_function x } = { x : ℝ | 0 < x ∧ x < 1/2 } ∪ { x : ℝ | x > 2 } :=
by
  sorry

end function_domain_l182_182028


namespace sufficient_but_not_necessary_condition_l182_182690

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x > 1 → |x| > 1) ∧ (|x| > 1 → (x > 1 ∨ x < -1)) ∧ ¬(|x| > 1 → x > 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l182_182690


namespace greatest_integer_radius_l182_182643

theorem greatest_integer_radius (r : ℝ) (h : π * r^2 < 75 * π) : r ≤ 8 :=
sorry

end greatest_integer_radius_l182_182643


namespace point_in_first_or_third_quadrant_l182_182059

-- Definitions based on conditions
variables {x y : ℝ}

-- The proof statement
theorem point_in_first_or_third_quadrant (h : x * y > 0) : 
  (0 < x ∧ 0 < y) ∨ (x < 0 ∧ y < 0) :=
  sorry

end point_in_first_or_third_quadrant_l182_182059


namespace paula_remaining_money_l182_182591

theorem paula_remaining_money 
  (M : Int) (C_s : Int) (N_s : Int) (C_p : Int) (N_p : Int)
  (h1 : M = 250) 
  (h2 : C_s = 15) 
  (h3 : N_s = 5) 
  (h4 : C_p = 25) 
  (h5 : N_p = 3) : 
  M - (C_s * N_s + C_p * N_p) = 100 := 
by
  sorry

end paula_remaining_money_l182_182591


namespace op_identity_l182_182226

-- Define the operation ⊕ as given by the table
def op (x y : ℕ) : ℕ :=
  match (x, y) with
  | (1, 1) => 4
  | (1, 2) => 1
  | (1, 3) => 2
  | (1, 4) => 3
  | (2, 1) => 1
  | (2, 2) => 3
  | (2, 3) => 4
  | (2, 4) => 2
  | (3, 1) => 2
  | (3, 2) => 4
  | (3, 3) => 1
  | (3, 4) => 3
  | (4, 1) => 3
  | (4, 2) => 2
  | (4, 3) => 3
  | (4, 4) => 4
  | _ => 0  -- default case for completeness

-- State the theorem
theorem op_identity : op (op 4 1) (op 2 3) = 3 := by
  sorry

end op_identity_l182_182226


namespace number_of_children_l182_182041

def cost_of_adult_ticket := 19
def cost_of_child_ticket := cost_of_adult_ticket - 6
def number_of_adults := 2
def total_cost := 77

theorem number_of_children : 
  ∃ (x : ℕ), cost_of_child_ticket * x + cost_of_adult_ticket * number_of_adults = total_cost ∧ x = 3 :=
by
  sorry

end number_of_children_l182_182041


namespace find_value_l182_182170

theorem find_value (
  a b c d e f : ℝ) 
  (h1 : a * b * c = 65) 
  (h2 : b * c * d = 65) 
  (h3 : c * d * e = 1000) 
  (h4 : (a * f) / (c * d) = 0.25) :
  d * e * f = 250 := 
sorry

end find_value_l182_182170


namespace inequalities_hold_l182_182522

theorem inequalities_hold (a b c x y z : ℝ) (hxa : x ≤ a) (hyb : y ≤ b) (hzc : z ≤ c) :
  (x * y + y * z + z * x ≤ a * b + b * c + c * a) ∧
  (x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2) ∧
  (x * y * z ≤ a * b * c) :=
by
  sorry

end inequalities_hold_l182_182522


namespace two_n_plus_m_value_l182_182875

theorem two_n_plus_m_value (n m : ℤ) :
  3 * n - m < 5 ∧ n + m > 26 ∧ 3 * m - 2 * n < 46 → 2 * n + m = 36 :=
sorry

end two_n_plus_m_value_l182_182875


namespace circle_center_radius_l182_182407

/-
Given:
- The endpoints of a diameter are (2, -3) and (-8, 7).

Prove:
- The center of the circle is (-3, 2).
- The radius of the circle is 5√2.
-/

noncomputable def center_and_radius (A B : ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let Cx := ((A.1 + B.1) / 2)
  let Cy := ((A.2 + B.2) / 2)
  let radius := Real.sqrt ((A.1 - Cx) * (A.1 - Cx) + (A.2 - Cy) * (A.2 - Cy))
  (Cx, Cy, radius)

theorem circle_center_radius :
  center_and_radius (2, -3) (-8, 7) = (-3, 2, 5 * Real.sqrt 2) :=
by
  sorry

end circle_center_radius_l182_182407


namespace unique_solution_is_2_or_minus_2_l182_182201

theorem unique_solution_is_2_or_minus_2 (a : ℝ) :
  (∃ x : ℝ, ∀ y : ℝ, (y^2 + a * y + 1 = 0 ↔ y = x)) → (a = 2 ∨ a = -2) :=
by sorry

end unique_solution_is_2_or_minus_2_l182_182201


namespace blue_cards_in_box_l182_182225

theorem blue_cards_in_box (x : ℕ) (h : 0.6 = (x : ℝ) / (x + 8)) : x = 12 :=
sorry

end blue_cards_in_box_l182_182225


namespace castle_lego_ratio_l182_182272

def total_legos : ℕ := 500
def legos_put_back : ℕ := 245
def legos_missing : ℕ := 5
def legos_used : ℕ := total_legos - legos_put_back - legos_missing
def ratio (a b : ℕ) : ℚ := a / b

theorem castle_lego_ratio : ratio legos_used total_legos = 1 / 2 :=
by
  unfold ratio legos_used total_legos legos_put_back legos_missing
  norm_num

end castle_lego_ratio_l182_182272


namespace probability_of_one_machine_maintenance_l182_182246

theorem probability_of_one_machine_maintenance :
  let pA := 0.1
  let pB := 0.2
  let pC := 0.4
  let qA := 1 - pA
  let qB := 1 - pB
  let qC := 1 - pC
  (pA * qB * qC) + (qA * pB * qC) + (qA * qB * pC) = 0.444 :=
by {
  let pA := 0.1
  let pB := 0.2
  let pC := 0.4
  let qA := 1 - pA
  let qB := 1 - pB
  let qC := 1 - pC
  show (pA * qB * qC) + (qA * pB * qC) + (qA * qB * pC) = 0.444
  sorry
}

end probability_of_one_machine_maintenance_l182_182246


namespace solve_equation_l182_182485

theorem solve_equation (x : ℝ) : x^4 + (4 - x)^4 = 272 ↔ x = 2 + Real.sqrt 6 ∨ x = 2 - Real.sqrt 6 := 
by sorry

end solve_equation_l182_182485


namespace Molly_swam_on_Saturday_l182_182171

variable (total_meters : ℕ) (sunday_meters : ℕ)

def saturday_meters := total_meters - sunday_meters

theorem Molly_swam_on_Saturday : 
  total_meters = 73 ∧ sunday_meters = 28 → saturday_meters total_meters sunday_meters = 45 := by
sorry

end Molly_swam_on_Saturday_l182_182171


namespace degree_of_d_l182_182926

noncomputable def f : Polynomial ℝ := sorry
noncomputable def d : Polynomial ℝ := sorry
noncomputable def q : Polynomial ℝ := sorry
noncomputable def r : Polynomial ℝ := 5 * Polynomial.X^2 + 3 * Polynomial.X - 8

axiom deg_f : f.degree = 15
axiom deg_q : q.degree = 7
axiom deg_r : r.degree = 2
axiom poly_div : f = d * q + r

theorem degree_of_d : d.degree = 8 :=
by
  sorry

end degree_of_d_l182_182926


namespace find_slope_l182_182588

theorem find_slope (k : ℝ) :
  (∀ x y : ℝ, y = -2 * x + 3 → y = k * x + 4 → (x, y) = (1, 1)) → k = -3 :=
by
  sorry

end find_slope_l182_182588


namespace weight_of_lightest_dwarf_l182_182775

noncomputable def weight_of_dwarf (n : ℕ) (x : ℝ) : ℝ := 5 - (n - 1) * x

theorem weight_of_lightest_dwarf :
  ∃ x : ℝ, 
    (∀ n : ℕ, n ≥ 1 ∧ n ≤ 101 → weight_of_dwarf 1 x = 5) ∧
    (weight_of_dwarf 76 x + weight_of_dwarf 77 x + weight_of_dwarf 78 x + weight_of_dwarf 79 x + weight_of_dwarf 80 x =
     weight_of_dwarf 96 x + weight_of_dwarf 97 x + weight_of_dwarf 98 x + weight_of_dwarf 99 x + weight_of_dwarf 100 x + weight_of_dwarf 101 x) →
    weight_of_dwarf 101 x = 2.5 :=
by
  sorry

end weight_of_lightest_dwarf_l182_182775


namespace scientific_notation_of_0_00000012_l182_182908

theorem scientific_notation_of_0_00000012 :
  0.00000012 = 1.2 * 10 ^ (-7) :=
by
  sorry

end scientific_notation_of_0_00000012_l182_182908


namespace classroom_books_l182_182661

theorem classroom_books (students_group1 students_group2 books_per_student_group1 books_per_student_group2 books_brought books_lost : ℕ)
  (h1 : students_group1 = 20)
  (h2 : books_per_student_group1 = 15)
  (h3 : students_group2 = 25)
  (h4 : books_per_student_group2 = 10)
  (h5 : books_brought = 30)
  (h6 : books_lost = 7) :
  (students_group1 * books_per_student_group1 + students_group2 * books_per_student_group2 + books_brought - books_lost) = 573 := by
  sorry

end classroom_books_l182_182661


namespace jack_mopping_time_l182_182960

-- Definitions for the conditions
def bathroom_area : ℝ := 24
def kitchen_area : ℝ := 80
def mopping_rate : ℝ := 8

-- The proof problem: Prove Jack will spend 13 minutes mopping
theorem jack_mopping_time : (bathroom_area + kitchen_area) / mopping_rate = 13 := by
  sorry

end jack_mopping_time_l182_182960


namespace smallest_perfect_cube_divisor_l182_182349

theorem smallest_perfect_cube_divisor 
  (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hpq : p ≠ q) 
  (hpr : p ≠ r) (hqr : q ≠ r) (s := 4) (hs : ¬ Nat.Prime s) 
  (hdiv : Nat.Prime 2) :
  ∃ n : ℕ, n = (p * q * r^2 * s)^3 ∧ ∀ m : ℕ, (∃ a b c d : ℕ, a = 3 ∧ b = 3 ∧ c = 6 ∧ d = 3 ∧ m = p^a * q^b * r^c * s^d) → m ≥ n :=
sorry

end smallest_perfect_cube_divisor_l182_182349


namespace max_shortest_part_duration_l182_182202

theorem max_shortest_part_duration (film_duration : ℕ) (part1 part2 part3 part4 : ℕ)
  (h_total : part1 + part2 + part3 + part4 = 192)
  (h_diff1 : part2 ≥ part1 + 6)
  (h_diff2 : part3 ≥ part2 + 6)
  (h_diff3 : part4 ≥ part3 + 6) :
  part1 ≤ 39 := 
sorry

end max_shortest_part_duration_l182_182202


namespace angle_C_is_30_degrees_l182_182823

theorem angle_C_is_30_degrees
  (A B C : ℝ)
  (h1 : 3 * Real.sin A + 4 * Real.cos B = 6)
  (h2 : 4 * Real.sin B + 3 * Real.cos A = 1)
  (A_rad: 0 ≤ A ∧ A ≤ Real.pi)
  (B_rad: 0 ≤ B ∧ B ≤ Real.pi)
  (C_rad : 0 ≤ C ∧ C ≤ Real.pi)
  (triangle_condition: A + B + C = Real.pi) :
  C = Real.pi / 6 :=
sorry

end angle_C_is_30_degrees_l182_182823


namespace vector_subtraction_proof_l182_182384

def v1 : ℝ × ℝ := (3, -8)
def v2 : ℝ × ℝ := (2, -6)
def a : ℝ := 5
def answer : ℝ × ℝ := (-7, 22)

theorem vector_subtraction_proof : (v1.1 - a * v2.1, v1.2 - a * v2.2) = answer := 
by
  sorry

end vector_subtraction_proof_l182_182384


namespace product_of_digits_of_nondivisible_by_5_number_is_30_l182_182049

-- Define the four-digit numbers
def numbers : List ℕ := [4825, 4835, 4845, 4855, 4865]

-- Define units and tens digit function
def units_digit (n : ℕ) := n % 10
def tens_digit (n : ℕ) := (n / 10) % 10

-- Assertion that 4865 is the number that is not divisible by 5
def not_divisible_by_5 (n : ℕ) : Prop := ¬ (units_digit n = 5 ∨ units_digit n = 0)

-- Lean 4 statement to prove the product of units and tens digit of the number not divisible by 5 is 30
theorem product_of_digits_of_nondivisible_by_5_number_is_30 :
  ∃ n ∈ numbers, not_divisible_by_5 n ∧ (units_digit n) * (tens_digit n) = 30 :=
by
  sorry

end product_of_digits_of_nondivisible_by_5_number_is_30_l182_182049


namespace base_6_conversion_l182_182006

-- Define the conditions given in the problem
def base_6_to_10 (a b c : ℕ) : ℕ := a * 6^2 + b * 6^1 + c * 6^0

-- given that 524_6 = 2cd_10 and c, d are base-10 digits, prove that (c * d) / 12 = 3/4
theorem base_6_conversion (c d : ℕ) (h1 : base_6_to_10 5 2 4 = 196) (h2 : 2 * 10 * c + d = 196) :
  (c * d) / 12 = 3 / 4 :=
sorry

end base_6_conversion_l182_182006


namespace problem_statement_l182_182387

theorem problem_statement :
  let a := (List.range (60 / 12)).card
  let b := (List.range (60 / Nat.lcm (Nat.lcm 2 3) 4)).card
  (a - b) ^ 3 = 0 :=
by
  sorry

end problem_statement_l182_182387


namespace triangular_pyramid_volume_l182_182447

theorem triangular_pyramid_volume (a b c : ℝ) 
  (h1 : 1 / 2 * a * b = 6) 
  (h2 : 1 / 2 * a * c = 4) 
  (h3 : 1 / 2 * b * c = 3) : 
  (1 / 3) * (1 / 2) * a * b * c = 4 := by 
  sorry

end triangular_pyramid_volume_l182_182447


namespace poodle_barks_proof_l182_182145

-- Definitions based on our conditions
def terrier_barks (hushes : Nat) : Nat := hushes * 2
def poodle_barks (terrier_barks : Nat) : Nat := terrier_barks * 2

-- Given that the terrier's owner says "hush" six times
def hushes : Nat := 6
def terrier_barks_total : Nat := terrier_barks hushes

-- The final statement that we need to prove
theorem poodle_barks_proof : 
    ∃ P, P = poodle_barks terrier_barks_total ∧ P = 24 := 
by
  -- The proof goes here
  sorry

end poodle_barks_proof_l182_182145


namespace jason_initial_cards_l182_182426

/-- Jason initially had some Pokemon cards, Alyssa bought him 224 more, 
and now Jason has 900 Pokemon cards in total.
Prove that initially Jason had 676 Pokemon cards. -/
theorem jason_initial_cards (a b c : ℕ) (h_a : a = 224) (h_b : b = 900) (h_cond : b = a + 676) : 676 = c :=
by 
  sorry

end jason_initial_cards_l182_182426


namespace cupboard_selling_percentage_l182_182369

theorem cupboard_selling_percentage (CP SP : ℝ) (h1 : CP = 6250) (h2 : SP + 1500 = 6250 * 1.12) :
  ((CP - SP) / CP) * 100 = 12 := by
sorry

end cupboard_selling_percentage_l182_182369


namespace min_distance_l182_182237

noncomputable def curve (x : ℝ) : ℝ := (1/4) * x^2 - (1/2) * Real.log x
noncomputable def line (x : ℝ) : ℝ := (3/4) * x - 1

theorem min_distance :
  ∀ P Q : ℝ × ℝ, 
  P.2 = curve P.1 → 
  Q.2 = line Q.1 → 
  ∃ min_dist : ℝ, 
  min_dist = (2 - 2 * Real.log 2) / 5 := 
sorry

end min_distance_l182_182237


namespace solution_points_satisfy_equation_l182_182080

theorem solution_points_satisfy_equation (x y : ℝ) :
  x^2 * (y + y^2) = y^3 + x^4 → (y = x ∨ y = -x ∨ y = x^2) := sorry

end solution_points_satisfy_equation_l182_182080


namespace find_length_of_PC_l182_182367

theorem find_length_of_PC (P A B C D : ℝ × ℝ) (h1 : (P.1 - A.1)^2 + (P.2 - A.2)^2 = 25)
                            (h2 : (P.1 - D.1)^2 + (P.2 - D.2)^2 = 36)
                            (h3 : (P.1 - B.1)^2 + (P.2 - B.2)^2 = 49)
                            (square_ABCD : (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2) :
  (P.1 - C.1)^2 + (P.2 - C.2)^2 = 38 :=
by
  sorry

end find_length_of_PC_l182_182367


namespace quadratic_function_condition_l182_182897

theorem quadratic_function_condition (m : ℝ) (h1 : |m| = 2) (h2 : m - 2 ≠ 0) : m = -2 :=
  sorry

end quadratic_function_condition_l182_182897


namespace minimum_bottles_needed_l182_182034

theorem minimum_bottles_needed (fl_oz_needed : ℝ) (bottle_size_ml : ℝ) (fl_oz_per_liter : ℝ) (ml_per_liter : ℝ)
  (h1 : fl_oz_needed = 60)
  (h2 : bottle_size_ml = 250)
  (h3 : fl_oz_per_liter = 33.8)
  (h4 : ml_per_liter = 1000) :
  ∃ n : ℕ, n = 8 ∧ fl_oz_needed * ml_per_liter / fl_oz_per_liter / bottle_size_ml ≤ n :=
by
  sorry

end minimum_bottles_needed_l182_182034


namespace map_length_25_cm_represents_125_km_l182_182792

-- Define the conditions
def map_scale (cm: ℝ) : ℝ := 5 * cm

-- Define the main statement to be proved
theorem map_length_25_cm_represents_125_km : map_scale 25 = 125 := by
  sorry

end map_length_25_cm_represents_125_km_l182_182792


namespace percentage_increase_l182_182351

variable (T : ℕ) (total_time : ℕ)

theorem percentage_increase (h1 : T = 4) (h2 : total_time = 10) : 
  ∃ P : ℕ, (T + P / 100 * T = total_time - T) → P = 50 := 
by 
  sorry

end percentage_increase_l182_182351


namespace total_tiles_144_l182_182061

-- Define the dimensions of the dining room
def diningRoomLength : ℕ := 15
def diningRoomWidth : ℕ := 20

-- Define the border width using 1x1 tiles
def borderWidth : ℕ := 2

-- Area of each 3x3 tile
def tileArea : ℕ := 9

-- Calculate the dimensions of the inner area after the border
def innerAreaLength : ℕ := diningRoomLength - 2 * borderWidth
def innerAreaWidth : ℕ := diningRoomWidth - 2 * borderWidth

-- Calculate the area of the inner region
def innerArea : ℕ := innerAreaLength * innerAreaWidth

-- Calculate the number of 3x3 tiles
def numThreeByThreeTiles : ℕ := (innerArea + tileArea - 1) / tileArea -- rounded up division

-- Calculate the number of 1x1 tiles for the border
def numOneByOneTiles : ℕ :=
  2 * (innerAreaLength + innerAreaWidth + 4 * borderWidth)

-- Total number of tiles
def totalTiles : ℕ := numOneByOneTiles + numThreeByThreeTiles

-- Prove that the total number of tiles is 144
theorem total_tiles_144 : totalTiles = 144 := by
  sorry

end total_tiles_144_l182_182061


namespace absolute_value_equation_sum_l182_182268

theorem absolute_value_equation_sum (x1 x2 : ℝ) (h1 : 3 * x1 - 12 = 6) (h2 : 3 * x2 - 12 = -6) : x1 + x2 = 8 := 
sorry

end absolute_value_equation_sum_l182_182268


namespace problem_l182_182952

variables (A B C D E : ℝ)

-- Conditions
def condition1 := A > C
def condition2 := E > B ∧ B > D
def condition3 := D > A
def condition4 := C > B

-- Proof goal: Dana (D) and Beth (B) have the same amount of money
theorem problem (h1 : condition1 A C) (h2 : condition2 E B D) (h3 : condition3 D A) (h4 : condition4 C B) : D = B :=
sorry

end problem_l182_182952


namespace ratio_proof_l182_182629

theorem ratio_proof (a b c : ℝ) (h1 : b / a = 4) (h2 : c / b = 5) : (a + 2 * b) / (3 * b + c) = 9 / 32 :=
by
  sorry

end ratio_proof_l182_182629


namespace find_ages_l182_182672

theorem find_ages (F S : ℕ) (h1 : F + 2 * S = 110) (h2 : 3 * F = 186) :
  F = 62 ∧ S = 24 := by
  sorry

end find_ages_l182_182672


namespace find_abcde_l182_182735

noncomputable def find_five_digit_number (a b c d e : ℕ) : ℕ :=
  10000 * a + 1000 * b + 100 * c + 10 * d + e

theorem find_abcde
  (a b c d e : ℕ)
  (h1 : 0 ≤ a ∧ a ≤ 9)
  (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : 0 ≤ c ∧ c ≤ 9)
  (h4 : 0 ≤ d ∧ d ≤ 9)
  (h5 : 0 ≤ e ∧ e ≤ 9)
  (h6 : a ≠ 0)
  (h7 : (10 * a + b + 10 * b + c) * (10 * b + c + 10 * c + d) * (10 * c + d + 10 * d + e) = 157605) :
  find_five_digit_number a b c d e = 12345 ∨ find_five_digit_number a b c d e = 21436 :=
sorry

end find_abcde_l182_182735


namespace negation_of_cos_proposition_l182_182836

variable (x : ℝ)

theorem negation_of_cos_proposition (h : ∀ x : ℝ, Real.cos x ≤ 1) : ∃ x₀ : ℝ, Real.cos x₀ > 1 :=
sorry

end negation_of_cos_proposition_l182_182836


namespace largest_number_in_L_shape_l182_182822

theorem largest_number_in_L_shape (x : ℤ) (sum : ℤ) (h : sum = 2015) : x = 676 :=
by
  sorry

end largest_number_in_L_shape_l182_182822


namespace sin_x1_sub_x2_l182_182394

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

theorem sin_x1_sub_x2 (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : x₁ < x₂) (h₃ : x₂ < Real.pi)
  (h₄ : f x₁ = 1 / 3) (h₅ : f x₂ = 1 / 3) : 
  Real.sin (x₁ - x₂) = - (2 * Real.sqrt 2) / 3 := 
sorry

end sin_x1_sub_x2_l182_182394


namespace solve_equation_l182_182866

theorem solve_equation (x : ℝ) (hx : 0 ≤ x) : 2021 * x^10 - 2020 * x - 1 = 0 ↔ x = 1 := 
by 
  sorry

end solve_equation_l182_182866


namespace total_potatoes_now_l182_182561

def initial_potatoes : ℕ := 8
def uneaten_new_potatoes : ℕ := 3

theorem total_potatoes_now : initial_potatoes + uneaten_new_potatoes = 11 := by
  sorry

end total_potatoes_now_l182_182561


namespace shifted_function_is_correct_l182_182829

-- Given conditions
def original_function (x : ℝ) : ℝ := -(x + 2) ^ 2 + 1

def shift_right (f : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ := f (x - a)

-- Resulting function after shifting 1 unit to the right
def shifted_function : ℝ → ℝ := shift_right original_function 1

-- Correct answer
def correct_function (x : ℝ) : ℝ := -(x + 1) ^ 2 + 1

-- Proof Statement
theorem shifted_function_is_correct :
  ∀ x : ℝ, shifted_function x = correct_function x := by
  sorry

end shifted_function_is_correct_l182_182829


namespace closest_clock_to_16_is_C_l182_182630

noncomputable def closestTo16InMirror (clock : Char) : Bool :=
  clock = 'C'

theorem closest_clock_to_16_is_C : 
  (closestTo16InMirror 'A' = False) ∧ 
  (closestTo16InMirror 'B' = False) ∧ 
  (closestTo16InMirror 'C' = True) ∧ 
  (closestTo16InMirror 'D' = False) := 
by
  sorry

end closest_clock_to_16_is_C_l182_182630


namespace boys_seen_l182_182093

theorem boys_seen (total_eyes : ℕ) (eyes_per_boy : ℕ) (h1 : total_eyes = 46) (h2 : eyes_per_boy = 2) : total_eyes / eyes_per_boy = 23 := 
by 
  sorry

end boys_seen_l182_182093


namespace value_of_a3_a6_a9_l182_182082

variable (a : ℕ → ℝ) (d : ℝ)

-- Condition: The common difference is 2
axiom common_difference : d = 2

-- Condition: a_1 + a_4 + a_7 = -50
axiom sum_a1_a4_a7 : a 1 + a 4 + a 7 = -50

-- The goal: a_3 + a_6 + a_9 = -38
theorem value_of_a3_a6_a9 : a 3 + a 6 + a 9 = -38 := 
by 
  sorry

end value_of_a3_a6_a9_l182_182082


namespace max_groups_eq_one_l182_182831

-- Defining the conditions 
def eggs : ℕ := 16
def marbles : ℕ := 3
def rubber_bands : ℕ := 5

-- The theorem statement
theorem max_groups_eq_one
  (h1 : eggs = 16)
  (h2 : marbles = 3)
  (h3 : rubber_bands = 5) :
  ∀ g : ℕ, (g ≤ eggs ∧ g ≤ marbles ∧ g ≤ rubber_bands) →
  (eggs % g = 0) ∧ (marbles % g = 0) ∧ (rubber_bands % g = 0) →
  g = 1 :=
by
  sorry

end max_groups_eq_one_l182_182831


namespace part_I_part_II_l182_182937

def sequence_def (x : ℕ → ℝ) (p : ℝ) : Prop :=
  x 1 = 1 ∧ ∀ n ∈ (Nat.succ <$> {n | n > 0}), x (n + 1) = 1 + x n / (p + x n)

theorem part_I (x : ℕ → ℝ) (p : ℝ) (h_seq : sequence_def x p) :
  p = 2 → ∀ n ∈ (Nat.succ <$> {n | n > 0}), x n < Real.sqrt 2 :=
sorry

theorem part_II (x : ℕ → ℝ) (p : ℝ) (h_seq : sequence_def x p) :
  (∀ n ∈ (Nat.succ <$> {n | n > 0}), x (n + 1) > x n) → ¬ ∃ M ∈ {n | n > 0}, ∀ n > 0, x M ≥ x n :=
sorry

end part_I_part_II_l182_182937


namespace namjoon_rank_l182_182003

theorem namjoon_rank (total_students : ℕ) (fewer_than_namjoon : ℕ) (rank_of_namjoon : ℕ) 
  (h1 : total_students = 13) (h2 : fewer_than_namjoon = 4) : rank_of_namjoon = 9 :=
sorry

end namjoon_rank_l182_182003


namespace solve_stamps_l182_182030

noncomputable def stamps_problem : Prop :=
  ∃ (A B C D : ℝ), 
    A + B + C + D = 251 ∧
    A = 2 * B + 2 ∧
    A = 3 * C + 6 ∧
    A = 4 * D - 16 ∧
    D = 32

theorem solve_stamps : stamps_problem :=
sorry

end solve_stamps_l182_182030


namespace avg_diff_noah_liam_l182_182513

-- Define the daily differences over 14 days
def daily_differences : List ℤ := [5, 0, 15, -5, 10, 10, -10, 5, 5, 10, -5, 15, 0, 5]

-- Define the function to calculate the average difference
def average_daily_difference (daily_diffs : List ℤ) : ℚ :=
  (daily_diffs.sum : ℚ) / daily_diffs.length

-- The proposition we want to prove
theorem avg_diff_noah_liam : average_daily_difference daily_differences = 60 / 14 := by
  sorry

end avg_diff_noah_liam_l182_182513


namespace min_living_allowance_inequality_l182_182786

variable (x : ℝ)

-- The regulation stipulates that the minimum living allowance should not be less than 300 yuan.
def min_living_allowance_regulation (x : ℝ) : Prop := x >= 300

theorem min_living_allowance_inequality (x : ℝ) :
  min_living_allowance_regulation x ↔ x ≥ 300 := by
  sorry

end min_living_allowance_inequality_l182_182786


namespace number_of_ordered_pairs_l182_182940

theorem number_of_ordered_pairs (p q : ℂ) (h1 : p^4 * q^3 = 1) (h2 : p^8 * q = 1) : (∃ n : ℕ, n = 40) :=
sorry

end number_of_ordered_pairs_l182_182940


namespace platform_length_l182_182418

theorem platform_length
  (train_length : ℝ := 360) -- The train is 360 meters long
  (train_speed_kmh : ℝ := 45) -- The train runs at a speed of 45 km/hr
  (time_to_pass_platform : ℝ := 60) -- It takes 60 seconds to pass the platform
  (platform_length : ℝ) : platform_length = 390 :=
by
  sorry

end platform_length_l182_182418


namespace max_k_constant_for_right_triangle_l182_182506

theorem max_k_constant_for_right_triangle (a b c : ℝ) (h : a^2 + b^2 = c^2) (h1 : a ≤ b) (h2 : b < c) :
  a^2 * (b + c) + b^2 * (c + a) + c^2 * (a + b) ≥ (2 + 3*Real.sqrt 2) * a * b * c :=
by 
  sorry

end max_k_constant_for_right_triangle_l182_182506


namespace kyle_and_miles_marbles_l182_182492

theorem kyle_and_miles_marbles (f k m : ℕ) 
  (h1 : f = 3 * k) 
  (h2 : f = 5 * m) 
  (h3 : f = 15) : 
  k + m = 8 := 
by 
  sorry

end kyle_and_miles_marbles_l182_182492


namespace grandma_olga_daughters_l182_182208

theorem grandma_olga_daughters :
  ∃ (D : ℕ), ∃ (S : ℕ),
  S = 3 ∧
  (∃ (total_grandchildren : ℕ), total_grandchildren = 33) ∧
  (∀ D', 6 * D' + 5 * S = 33 → D = D')
:=
sorry

end grandma_olga_daughters_l182_182208


namespace maximal_cardinality_set_l182_182162

theorem maximal_cardinality_set (n : ℕ) (h_n : n ≥ 2) :
  ∃ M : Finset (ℕ × ℕ), ∀ (j k : ℕ), (1 ≤ j ∧ j < k ∧ k ≤ n) → 
  ((j, k) ∈ M → ∀ m, (k, m) ∉ M) ∧ 
  M.card = ⌊(n * n / 4 : ℝ)⌋ :=
by
  sorry

end maximal_cardinality_set_l182_182162


namespace block_fraction_visible_above_water_l182_182198

-- Defining constants
def weight_of_block : ℝ := 30 -- N
def buoyant_force_submerged : ℝ := 50 -- N

-- Defining the proof problem
theorem block_fraction_visible_above_water (W Fb : ℝ) (hW : W = weight_of_block) (hFb : Fb = buoyant_force_submerged) :
  (1 - W / Fb) = 2 / 5 :=
by
  -- Proof is omitted
  sorry

end block_fraction_visible_above_water_l182_182198


namespace range_of_k_l182_182542

theorem range_of_k (k : ℝ) : (∀ y : ℝ, ∃ x : ℝ, y = Real.log (x^2 - 2*k*x + k)) ↔ (k ∈ Set.Iic 0 ∨ k ∈ Set.Ici 1) :=
by
  sorry

end range_of_k_l182_182542


namespace quadrilateral_side_length_eq_12_l182_182467

-- Definitions
def EF : ℝ := 7
def FG : ℝ := 15
def GH : ℝ := 7
def HE : ℝ := 12
def EH : ℝ := 12

-- Statement to prove that EH = 12 given the definition and conditions
theorem quadrilateral_side_length_eq_12
  (EF_eq : EF = 7)
  (FG_eq : FG = 15)
  (GH_eq : GH = 7)
  (HE_eq : HE = 12)
  (EH_eq : EH = 12) : 
  EH = 12 :=
sorry

end quadrilateral_side_length_eq_12_l182_182467


namespace scout_weekend_earnings_l182_182707

theorem scout_weekend_earnings
  (base_pay_per_hour : ℕ)
  (tip_per_delivery : ℕ)
  (hours_worked_saturday : ℕ)
  (deliveries_saturday : ℕ)
  (hours_worked_sunday : ℕ)
  (deliveries_sunday : ℕ)
  (total_earnings : ℕ)
  (h_base_pay : base_pay_per_hour = 10)
  (h_tip : tip_per_delivery = 5)
  (h_hours_sat : hours_worked_saturday = 4)
  (h_deliveries_sat : deliveries_saturday = 5)
  (h_hours_sun : hours_worked_sunday = 5)
  (h_deliveries_sun : deliveries_sunday = 8) :
  total_earnings = 155 :=
by
  sorry

end scout_weekend_earnings_l182_182707


namespace ferry_q_more_time_l182_182185

variables (speed_ferry_p speed_ferry_q distance_ferry_p distance_ferry_q time_ferry_p time_ferry_q : ℕ)
  -- Conditions given in the problem
  (h1 : speed_ferry_p = 8)
  (h2 : time_ferry_p = 2)
  (h3 : distance_ferry_p = speed_ferry_p * time_ferry_p)
  (h4 : distance_ferry_q = 3 * distance_ferry_p)
  (h5 : speed_ferry_q = speed_ferry_p + 4)
  (h6 : time_ferry_q = distance_ferry_q / speed_ferry_q)

theorem ferry_q_more_time : time_ferry_q - time_ferry_p = 2 :=
by
  sorry

end ferry_q_more_time_l182_182185


namespace simplify_polynomial_l182_182813

theorem simplify_polynomial :
  (3 * x ^ 4 - 2 * x ^ 3 + 5 * x ^ 2 - 8 * x + 10) + (7 * x ^ 5 - 3 * x ^ 4 + x ^ 3 - 7 * x ^ 2 + 2 * x - 2)
  = 7 * x ^ 5 - x ^ 3 - 2 * x ^ 2 - 6 * x + 8 :=
by sorry

end simplify_polynomial_l182_182813


namespace xy_system_sol_l182_182243

theorem xy_system_sol (x y : ℝ) (h1 : x * y = 8) (h2 : x^2 * y + x * y^2 + x + y = 80) :
  x^3 + y^3 = 416000 / 729 :=
by
  sorry

end xy_system_sol_l182_182243


namespace evaporation_period_length_l182_182274

theorem evaporation_period_length
  (initial_water : ℕ) (daily_evaporation : ℝ) (evaporated_percentage : ℝ) : 
  evaporated_percentage * (initial_water : ℝ) / 100 / daily_evaporation = 22 :=
by
  -- Conditions of the problem
  let initial_water := 12
  let daily_evaporation := 0.03
  let evaporated_percentage := 5.5
  -- Sorry proof placeholder
  sorry

end evaporation_period_length_l182_182274


namespace total_cost_after_discounts_l182_182013

-- Definition of the cost function with applicable discounts
def pencil_cost (price: ℝ) (count: ℕ) (discount_threshold: ℕ) (discount_rate: ℝ) :=
  let initial_cost := count * price
  if count > discount_threshold then
    initial_cost - (initial_cost * discount_rate)
  else initial_cost

def pen_cost (price: ℝ) (count: ℕ) (discount_threshold: ℕ) (discount_rate: ℝ) :=
  let initial_cost := count * price
  if count > discount_threshold then
    initial_cost - (initial_cost * discount_rate)
  else initial_cost

-- The statement to be proved
theorem total_cost_after_discounts :
  let pencil_price := 2.50
  let pen_price := 3.50
  let pencil_count := 38
  let pen_count := 56
  let pencil_discount_threshold := 30
  let pencil_discount_rate := 0.10
  let pen_discount_threshold := 50
  let pen_discount_rate := 0.15
  let total_cost := pencil_cost pencil_price pencil_count pencil_discount_threshold pencil_discount_rate
                   + pen_cost pen_price pen_count pen_discount_threshold pen_discount_rate
  total_cost = 252.10 := 
by 
  sorry

end total_cost_after_discounts_l182_182013


namespace field_area_restriction_l182_182344

theorem field_area_restriction (S : ℚ) (b : ℤ) (a : ℚ) (x y : ℚ) 
  (h1 : 10 * 300 * S ≤ 10000)
  (h2 : 2 * a = - b)
  (h3 : abs (6 * y) + 3 ≥ 3)
  (h4 : 2 * abs (2 * x) - abs b ≤ 9)
  (h5 : b ∈ [-4, -3, -2, -1, 0, 1, 2, 3, 4])
: S ≤ 10 / 3 := sorry

end field_area_restriction_l182_182344


namespace sin_double_angle_l182_182320

theorem sin_double_angle (α : ℝ) (h : Real.cos (π / 4 - α) = 3 / 5) : Real.sin (2 * α) = -7 / 25 := by
  sorry

end sin_double_angle_l182_182320


namespace income_on_fifth_day_l182_182584

-- Define the incomes for the first four days
def income_day1 := 600
def income_day2 := 250
def income_day3 := 450
def income_day4 := 400

-- Define the average income
def average_income := 500

-- Define the length of days
def days := 5

-- Define the total income for the 5 days
def total_income : ℕ := days * average_income

-- Define the total income for the first 4 days
def total_income_first4 := income_day1 + income_day2 + income_day3 + income_day4

-- Define the income on the fifth day
def income_day5 := total_income - total_income_first4

-- The theorem to prove the income of the fifth day is $800
theorem income_on_fifth_day : income_day5 = 800 := by
  -- proof is not required, so we leave the proof section with sorry
  sorry

end income_on_fifth_day_l182_182584


namespace reservoir_water_level_l182_182425

theorem reservoir_water_level (x : ℝ) (h : 0 ≤ x ∧ x ≤ 5) : 
  ∃ y : ℝ, y = 6 + 0.3 * x :=
by sorry

end reservoir_water_level_l182_182425


namespace sqrt_product_eq_l182_182882

theorem sqrt_product_eq :
  (Int.sqrt (2 ^ 2 * 3 ^ 4) : ℤ) = 18 :=
sorry

end sqrt_product_eq_l182_182882


namespace twenty_four_is_eighty_percent_of_what_number_l182_182649

theorem twenty_four_is_eighty_percent_of_what_number (x : ℝ) (hx : 24 = 0.8 * x) : x = 30 :=
  sorry

end twenty_four_is_eighty_percent_of_what_number_l182_182649


namespace remaining_volume_after_pours_l182_182791

-- Definitions based on the problem conditions
def initial_volume_liters : ℝ := 2
def initial_volume_milliliters : ℝ := initial_volume_liters * 1000
def pour_amount (x : ℝ) : ℝ := x

-- Statement of the problem as a theorem in Lean 4
theorem remaining_volume_after_pours (x : ℝ) : 
  ∃ remaining_volume : ℝ, remaining_volume = initial_volume_milliliters - 4 * pour_amount x :=
by
  -- To be filled with the proof
  sorry

end remaining_volume_after_pours_l182_182791


namespace new_quadratic_coeff_l182_182566

theorem new_quadratic_coeff (r s p q : ℚ) 
  (h1 : 3 * r^2 + 4 * r + 2 = 0)
  (h2 : 3 * s^2 + 4 * s + 2 = 0)
  (h3 : r + s = -4 / 3)
  (h4 : r * s = 2 / 3) 
  (h5 : r^3 + s^3 = - p) :
  p = 16 / 27 :=
by
  sorry

end new_quadratic_coeff_l182_182566


namespace arithmetic_sequence_general_term_l182_182957

theorem arithmetic_sequence_general_term (a : ℕ → ℝ)
  (h_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h_a3 : a 3 = 7)
  (h_a7 : a 7 = 3) :
  ∀ n, a n = -↑n + 10 :=
by
  sorry

end arithmetic_sequence_general_term_l182_182957


namespace soybeans_in_jar_l182_182609

theorem soybeans_in_jar
  (totalRedBeans : ℕ)
  (sampleSize : ℕ)
  (sampleRedBeans : ℕ)
  (totalBeans : ℕ)
  (proportion : sampleRedBeans / sampleSize = totalRedBeans / totalBeans)
  (h1 : totalRedBeans = 200)
  (h2 : sampleSize = 60)
  (h3 : sampleRedBeans = 5) :
  totalBeans = 2400 :=
by
  sorry

end soybeans_in_jar_l182_182609


namespace number_of_tiles_l182_182336

noncomputable def tile_count (room_length : ℝ) (room_width : ℝ) (tile_length : ℝ) (tile_width : ℝ) :=
  let room_area := room_length * room_width
  let tile_area := tile_length * tile_width
  room_area / tile_area

theorem number_of_tiles :
  tile_count 10 15 (1 / 4) (5 / 12) = 1440 := by
  sorry

end number_of_tiles_l182_182336


namespace sphere_shot_radius_l182_182280

theorem sphere_shot_radius (R : ℝ) (N : ℕ) (π : ℝ) (r : ℝ) 
  (h₀ : R = 4) (h₁ : N = 64) 
  (h₂ : (4 / 3) * π * (R ^ 3) / ((4 / 3) * π * (r ^ 3)) = N) : 
  r = 1 := 
by
  sorry

end sphere_shot_radius_l182_182280


namespace diagonal_of_square_l182_182055

theorem diagonal_of_square (length_rect width_rect : ℝ) (h1 : length_rect = 45) (h2 : width_rect = 40)
  (area_rect : ℝ) (h3 : area_rect = length_rect * width_rect) (area_square : ℝ) (h4 : area_square = area_rect)
  (side_square : ℝ) (h5 : side_square^2 = area_square) (diagonal_square : ℝ) (h6 : diagonal_square = side_square * Real.sqrt 2) :
  diagonal_square = 60 := by
  sorry

end diagonal_of_square_l182_182055


namespace sum_of_legs_of_similar_larger_triangle_l182_182628

-- Define the conditions for the problem
def smaller_triangle_area : ℝ := 10
def larger_triangle_area : ℝ := 400
def smaller_triangle_hypotenuse : ℝ := 10

-- Define the correct answer (sum of the lengths of the legs of the larger triangle)
def sum_of_legs_of_larger_triangle : ℝ := 88.55

-- State the Lean theorem
theorem sum_of_legs_of_similar_larger_triangle :
  (∀ (A B C a b c : ℝ), 
    a * b / 2 = smaller_triangle_area ∧ 
    c = smaller_triangle_hypotenuse ∧
    C * C / 4 = larger_triangle_area / smaller_triangle_area ∧
    A / a = B / b ∧ 
    A^2 + B^2 = C^2 → 
    A + B = sum_of_legs_of_larger_triangle) :=
  by sorry

end sum_of_legs_of_similar_larger_triangle_l182_182628


namespace moon_speed_conversion_correct_l182_182799

-- Define the conversions
def kilometers_per_second_to_miles_per_hour (kmps : ℝ) : ℝ :=
  kmps * 0.621371 * 3600

-- Condition: The moon's speed
def moon_speed_kmps : ℝ := 1.02

-- Correct answer in miles per hour
def expected_moon_speed_mph : ℝ := 2281.34

-- Theorem stating the equivalence of converted speed to expected speed
theorem moon_speed_conversion_correct :
  kilometers_per_second_to_miles_per_hour moon_speed_kmps = expected_moon_speed_mph :=
by 
  sorry

end moon_speed_conversion_correct_l182_182799


namespace sin_A_plus_B_lt_sin_A_add_sin_B_l182_182726

variable {A B : ℝ}
variable (A_pos : 0 < A)
variable (B_pos : 0 < B)
variable (AB_sum_pi : A + B < π)

theorem sin_A_plus_B_lt_sin_A_add_sin_B (a b : ℝ) (h1 : a = Real.sin (A + B)) (h2 : b = Real.sin A + Real.sin B) : 
  a < b := by
  sorry

end sin_A_plus_B_lt_sin_A_add_sin_B_l182_182726


namespace distance_between_points_l182_182911

theorem distance_between_points (a b c d m k : ℝ) 
  (h1 : b = 2 * m * a + k) (h2 : d = -m * c + k) : 
  (Real.sqrt ((c - a)^2 + (d - b)^2)) = Real.sqrt ((1 + m^2) * (c - a)^2) := 
by {
  sorry
}

end distance_between_points_l182_182911


namespace emma_correct_percentage_l182_182488

theorem emma_correct_percentage (t : ℕ) (lt : t > 0)
  (liam_correct_alone : ℝ := 0.70)
  (liam_overall_correct : ℝ := 0.82)
  (emma_correct_alone : ℝ := 0.85)
  (joint_error_rate : ℝ := 0.05)
  (liam_solved_together_correct : ℝ := liam_overall_correct * t - liam_correct_alone * (t / 2)) :
  (emma_correct_alone * (t / 2) + (1 - joint_error_rate) * liam_solved_together_correct) / t * 100 = 87.15 :=
by
  sorry

end emma_correct_percentage_l182_182488


namespace percentage_return_is_25_l182_182420

noncomputable def percentage_return_on_investment
  (dividend_rate : ℝ)
  (face_value : ℝ)
  (purchase_price : ℝ) : ℝ :=
  (dividend_rate / 100 * face_value / purchase_price) * 100

theorem percentage_return_is_25 :
  percentage_return_on_investment 18.5 50 37 = 25 := 
by
  sorry

end percentage_return_is_25_l182_182420


namespace probability_of_selecting_cooking_is_one_fourth_l182_182303

-- Define the set of available courses
def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

-- Define the probability of selecting a specific course from the list
def probability_of_selecting_cooking (course : String) (choices : List String) : ℚ :=
  if course ∈ choices then 1 / choices.length else 0

-- Prove that the probability of selecting "cooking" from the four courses is 1/4
theorem probability_of_selecting_cooking_is_one_fourth : 
  probability_of_selecting_cooking "cooking" courses = 1 / 4 := by
  sorry

end probability_of_selecting_cooking_is_one_fourth_l182_182303


namespace girls_count_l182_182421

-- Definition of the conditions
variables (B G : ℕ)

def college_conditions (B G : ℕ) : Prop :=
  (B + G = 416) ∧ (B = (8 * G) / 5)

-- Statement to prove
theorem girls_count (B G : ℕ) (h : college_conditions B G) : G = 160 :=
by
  sorry

end girls_count_l182_182421


namespace probability_of_two_red_balls_l182_182469

theorem probability_of_two_red_balls :
  let red_balls := 4
  let blue_balls := 4
  let green_balls := 2
  let total_balls := red_balls + blue_balls + green_balls
  let prob_red1 := (red_balls : ℚ) / total_balls
  let prob_red2 := ((red_balls - 1 : ℚ) / (total_balls - 1))
  (prob_red1 * prob_red2 = (2 : ℚ) / 15) :=
by
  sorry

end probability_of_two_red_balls_l182_182469


namespace complement_event_l182_182750

def total_students : ℕ := 4
def males : ℕ := 2
def females : ℕ := 2
def choose2 (n : ℕ) := n * (n - 1) / 2

noncomputable def eventA : ℕ := males * females
noncomputable def eventB : ℕ := choose2 males
noncomputable def eventC : ℕ := choose2 females

theorem complement_event {total_students males females : ℕ}
  (h_total : total_students = 4)
  (h_males : males = 2)
  (h_females : females = 2) :
  (total_students.choose 2 - (eventB + eventC)) / total_students.choose 2 = 1 / 3 :=
by
  sorry

end complement_event_l182_182750


namespace green_pens_l182_182917

theorem green_pens (blue_pens green_pens : ℕ) (ratio_blue_to_green : blue_pens / green_pens = 4 / 3) (total_blue : blue_pens = 16) : green_pens = 12 :=
by sorry

end green_pens_l182_182917


namespace alice_bob_sum_is_42_l182_182461

theorem alice_bob_sum_is_42 :
  ∃ (A B : ℕ), 
    (1 ≤ A ∧ A ≤ 60) ∧ 
    (1 ≤ B ∧ B ≤ 60) ∧ 
    Nat.Prime B ∧ B > 10 ∧ 
    (∀ n : ℕ, n < 5 → (A + B) % n ≠ 0) ∧ 
    ∃ k : ℕ, 150 * B + A = k * k ∧ 
    A + B = 42 :=
by 
  sorry

end alice_bob_sum_is_42_l182_182461


namespace arithmetic_sequence_n_terms_l182_182314

theorem arithmetic_sequence_n_terms:
  ∀ (a₁ d aₙ n: ℕ), 
  a₁ = 6 → d = 3 → aₙ = 300 → aₙ = a₁ + (n - 1) * d → n = 99 :=
by
  intros a₁ d aₙ n h1 h2 h3 h4
  sorry

end arithmetic_sequence_n_terms_l182_182314


namespace geom_seq_common_ratio_l182_182987

variable {a_n : ℕ → ℝ}
variable {q : ℝ}

def is_geom_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem geom_seq_common_ratio (h1 : a_n 0 + a_n 2 = 10)
                              (h2 : a_n 3 + a_n 5 = 5 / 4)
                              (h_geom : is_geom_seq a_n q) :
  q = 1 / 2 :=
by
  sorry

end geom_seq_common_ratio_l182_182987


namespace sum_of_integers_l182_182083

theorem sum_of_integers (m n p q : ℤ) 
(h1 : m ≠ n) (h2 : m ≠ p) 
(h3 : m ≠ q) (h4 : n ≠ p) 
(h5 : n ≠ q) (h6 : p ≠ q) 
(h7 : (5 - m) * (5 - n) * (5 - p) * (5 - q) = 9) : 
m + n + p + q = 20 :=
by
  sorry

end sum_of_integers_l182_182083


namespace ceil_inequality_range_x_solve_eq_l182_182362

-- Definition of the mathematical ceiling function to comply with the condition a).
def ceil (a : ℚ) : ℤ := ⌈a⌉

-- Condition 1: Relationship between m and ⌈m⌉.
theorem ceil_inequality (m : ℚ) : m ≤ ceil m ∧ ceil m < m + 1 :=
sorry

-- Part 2.1: Range of x given {3x + 2} = 8.
theorem range_x (x : ℚ) (h : ceil (3 * x + 2) = 8) : 5 / 3 < x ∧ x ≤ 2 :=
sorry

-- Part 2.2: Solving {3x - 2} = 2x + 1/2
theorem solve_eq (x : ℚ) (h : ceil (3 * x - 2) = 2 * x + 1 / 2) : x = 7 / 4 ∨ x = 9 / 4 :=
sorry

end ceil_inequality_range_x_solve_eq_l182_182362


namespace exponent_equivalence_l182_182801

theorem exponent_equivalence (a b : ℕ) (m n : ℕ) (m_pos : 0 < m) (n_pos : 0 < n) (h1 : 9 ^ m = a) (h2 : 3 ^ n = b) : 
  3 ^ (2 * m + 4 * n) = a * b ^ 4 := 
by 
  sorry

end exponent_equivalence_l182_182801


namespace problem_statement_l182_182898

def f (x : ℝ) : ℝ := x^2 - 3 * x + 6

def g (x : ℝ) : ℝ := x + 4

theorem problem_statement : f (g 3) - g (f 3) = 24 := by
  sorry

end problem_statement_l182_182898


namespace find_S_l182_182827

theorem find_S :
  (1/4 : ℝ) * (1/6 : ℝ) * S = (1/5 : ℝ) * (1/8 : ℝ) * 160 → S = 96 :=
by
  intro h
  -- Proof is omitted
  sorry 

end find_S_l182_182827


namespace find_solutions_l182_182131

noncomputable
def is_solution (a b c d : ℝ) : Prop :=
  a + b + c = d ∧ (1 / a + 1 / b + 1 / c = 1 / d)

theorem find_solutions (a b c d : ℝ) :
  is_solution a b c d ↔ (c = -a ∧ d = b) ∨ (c = -b ∧ d = a) :=
by
  sorry

end find_solutions_l182_182131


namespace pair_cannot_appear_l182_182451

theorem pair_cannot_appear :
  ¬ ∃ (sequence_of_pairs : List (ℤ × ℤ)), 
    (1, 2) ∈ sequence_of_pairs ∧ 
    (2022, 2023) ∈ sequence_of_pairs ∧ 
    ∀ (a b : ℤ) (seq : List (ℤ × ℤ)), 
      (a, b) ∈ seq → 
      ((-a, -b) ∈ seq ∨ (-b, a+b) ∈ seq ∨ 
      ∃ (c d : ℤ), ((a+c, b+d) ∈ seq ∧ (c, d) ∈ seq)) := 
sorry

end pair_cannot_appear_l182_182451


namespace train_speed_in_kmph_l182_182973

def train_length : ℕ := 125
def time_to_cross_pole : ℕ := 9
def conversion_factor : ℚ := 18 / 5

theorem train_speed_in_kmph
  (d : ℕ := train_length)
  (t : ℕ := time_to_cross_pole)
  (cf : ℚ := conversion_factor) :
  d / t * cf = 50 := 
sorry

end train_speed_in_kmph_l182_182973


namespace angle_PDO_45_degrees_l182_182010

-- Define the square configuration
variables (A B C D L P Q M N O : Type)
variables (a : ℝ) -- side length of the square ABCD

-- Conditions as hypothesized in the problem
def is_square (v₁ v₂ v₃ v₄ : Type) := true -- Placeholder for the square property
def on_diagonal_AC (L : Type) := true -- Placeholder for L being on diagonal AC
def common_vertex_L (sq1_v1 sq1_v2 sq1_v3 sq1_v4 sq2_v1 sq2_v2 sq2_v3 sq2_v4 : Type) := true -- Placeholder for common vertex L
def point_on_side (P AB_side: Type) := true -- Placeholder for P on side AB of ABCD
def square_center (center sq_v1 sq_v2 sq_v3 sq_v4 : Type) := true -- Placeholder for square's center

-- Prove the angle PDO is 45 degrees
theorem angle_PDO_45_degrees 
  (h₁ : is_square A B C D)
  (h₂ : on_diagonal_AC L)
  (h₃ : is_square A P L Q)
  (h₄ : is_square C M L N)
  (h₅ : common_vertex_L A P L Q C M L N)
  (h₆ : point_on_side P B)
  (h₇ : square_center O C M L N)
  : ∃ θ : ℝ, θ = 45 := 
  sorry

end angle_PDO_45_degrees_l182_182010


namespace arith_seq_geom_seq_l182_182634

theorem arith_seq_geom_seq (a : ℕ → ℝ) (d : ℝ) (h : d ≠ 0) 
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : (a 9)^2 = a 5 * a 15) :
  a 15 / a 9 = 3 / 2 := by
  sorry

end arith_seq_geom_seq_l182_182634


namespace smallest_integer_ratio_l182_182881

theorem smallest_integer_ratio (x y : ℕ) (hx : 10 ≤ x ∧ x ≤ 99) (hy : 10 ≤ y ∧ y ≤ 99) (h_sum : x + y = 120) (h_even : x % 2 = 0) : ∃ (k : ℕ), k = x / y ∧ k = 1 :=
by
  sorry

end smallest_integer_ratio_l182_182881


namespace find_constant_l182_182915

theorem find_constant (c : ℝ) (f : ℝ → ℝ)
  (h : f x = c * x^3 + 19 * x^2 - 4 * c * x + 20)
  (hx : f (-7) = 0) :
  c = 3 :=
sorry

end find_constant_l182_182915


namespace triangle_inequality_equality_iff_equilateral_l182_182946

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) : 
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 := 
sorry

theorem equality_iff_equilateral (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) : 
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c := 
sorry

end triangle_inequality_equality_iff_equilateral_l182_182946


namespace prob_sum_24_four_dice_l182_182291

-- The probability of each die landing on six
def prob_die_six : ℚ := 1 / 6

-- The probability of all four dice showing six
theorem prob_sum_24_four_dice : 
  prob_die_six ^ 4 = 1 / 1296 :=
by
  -- Equivalent Lean statement asserting the probability problem
  sorry

end prob_sum_24_four_dice_l182_182291


namespace watermelon_and_banana_weight_l182_182924

variables (w b : ℕ)
variables (h1 : 2 * w + b = 8100)
variables (h2 : 2 * w + 3 * b = 8300)

theorem watermelon_and_banana_weight (Hw : w = 4000) (Hb : b = 100) :
  2 * w + b = 8100 ∧ 2 * w + 3 * b = 8300 :=
by
  sorry

end watermelon_and_banana_weight_l182_182924


namespace pavan_travel_distance_l182_182593

theorem pavan_travel_distance (t : ℝ) (v1 v2 : ℝ) (D : ℝ) (h₁ : t = 15) (h₂ : v1 = 30) (h₃ : v2 = 25):
  (D / 2) / v1 + (D / 2) / v2 = t → D = 2250 / 11 :=
by
  intro h
  rw [h₁, h₂, h₃] at h
  sorry

end pavan_travel_distance_l182_182593


namespace contrapositive_of_proposition_l182_182038

theorem contrapositive_of_proposition :
  (∀ x : ℝ, x ≤ -3 → x < 0) ↔ (∀ x : ℝ, x ≥ 0 → x > -3) :=
by
  sorry

end contrapositive_of_proposition_l182_182038


namespace problem_l182_182086

-- Define the concept of reciprocal
def reciprocal (x : ℚ) : ℚ := 1 / x

-- Define the conditions in the problem
def condition1 : Prop := reciprocal 1.5 = 2/3
def condition2 : Prop := reciprocal 1 = 1

-- Theorem stating our goals
theorem problem : condition1 ∧ condition2 :=
by {
  sorry
}

end problem_l182_182086


namespace stopped_clock_more_accurate_l182_182910

theorem stopped_clock_more_accurate (slow_correct_time_frequency : ℕ)
  (stopped_correct_time_frequency : ℕ)
  (h1 : slow_correct_time_frequency = 720)
  (h2 : stopped_correct_time_frequency = 2) :
  stopped_correct_time_frequency > slow_correct_time_frequency / 720 :=
by
  sorry

end stopped_clock_more_accurate_l182_182910


namespace part1_max_area_part2_find_a_l182_182859

-- Part (1): Define the function and prove maximum area of the triangle
noncomputable def f (a x : ℝ) : ℝ := a^2 * Real.exp x - 3 * a * x + 2 * Real.sin x - 1

theorem part1_max_area (a : ℝ) (h : 0 < a ∧ a < 1) : 
  let f' := a^2 - 3 * a + 2
  ∃ h_a_max, h_a_max == 3 / 8 :=
  sorry

-- Part (2): Prove that the function reaches an extremum at x = 0 and determine the value of a.
theorem part2_find_a (a : ℝ) : (a^2 - 3 * a + 2 = 0) → (a = 1 ∨ a = 2) :=
  sorry

end part1_max_area_part2_find_a_l182_182859


namespace flower_bed_width_l182_182920

theorem flower_bed_width (length area : ℝ) (h_length : length = 4) (h_area : area = 143.2) :
  area / length = 35.8 :=
by
  sorry

end flower_bed_width_l182_182920


namespace connect_5_points_four_segments_l182_182517

theorem connect_5_points_four_segments (A B C D E : Type) (h : ∀ (P Q R : Type), P ≠ Q ∧ Q ≠ R ∧ R ≠ P)
: ∃ (n : ℕ), n = 135 := 
  sorry

end connect_5_points_four_segments_l182_182517


namespace problem_statement_l182_182302

theorem problem_statement (x : ℝ) (n : ℕ) (h1 : |x| < 1) (h2 : 2 ≤ n) : 
  (1 + x)^n + (1 - x)^n < 2^n :=
sorry

end problem_statement_l182_182302


namespace xiaodong_sister_age_correct_l182_182687

/-- Let's define the conditions as Lean definitions -/
def sister_age := 13
def xiaodong_age := sister_age - 8
def sister_age_in_3_years := sister_age + 3
def xiaodong_age_in_3_years := xiaodong_age + 3

/-- We need to prove that in 3 years, the sister's age will be twice Xiaodong's age -/
theorem xiaodong_sister_age_correct :
  (sister_age_in_3_years = 2 * xiaodong_age_in_3_years) → sister_age = 13 :=
by
  sorry

end xiaodong_sister_age_correct_l182_182687


namespace total_flowers_l182_182092

theorem total_flowers (R T L : ℕ) 
  (hR : R = 58)
  (hT : R = T + 15)
  (hL : R = L - 25) :
  R + T + L = 184 :=
by 
  sorry

end total_flowers_l182_182092


namespace show_linear_l182_182285

-- Define the conditions as given in the problem
variables (a b : ℤ)

-- The hypothesis that the equation is linear
def linear_equation_hypothesis : Prop :=
  (a + b = 1) ∧ (3 * a + 2 * b - 4 = 1)

-- Define the theorem we need to prove
theorem show_linear (h : linear_equation_hypothesis a b) : a + b = 1 := 
by
  sorry

end show_linear_l182_182285


namespace p_at_5_l182_182553

noncomputable def p (x : ℝ) : ℝ :=
  sorry

def p_cond (n : ℝ) : Prop :=
  (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4) → p n = 1 / n^3

theorem p_at_5 : (∀ n, p_cond n) → p 5 = -149 / 1500 :=
by
  intros
  sorry

end p_at_5_l182_182553


namespace profit_percentage_calculation_l182_182136

def selling_price : ℝ := 120
def cost_price : ℝ := 96

theorem profit_percentage_calculation (sp cp : ℝ) (hsp : sp = selling_price) (hcp : cp = cost_price) : 
  ((sp - cp) / cp) * 100 = 25 := 
 by
  sorry

end profit_percentage_calculation_l182_182136


namespace son_work_rate_l182_182184

noncomputable def man_work_rate := 1/10
noncomputable def combined_work_rate := 1/5

theorem son_work_rate :
  ∃ S : ℝ, man_work_rate + S = combined_work_rate ∧ S = 1/10 := sorry

end son_work_rate_l182_182184


namespace proof_of_problem_l182_182785

theorem proof_of_problem (a b : ℝ) (h1 : a > b) (h2 : a * b = a / b) : b = 1 ∧ 0 < a :=
by
  sorry

end proof_of_problem_l182_182785


namespace find_coordinates_of_A_l182_182565

theorem find_coordinates_of_A (x : ℝ) :
  let A := (x, 1, 2)
  let B := (2, 3, 4)
  (Real.sqrt ((x - 2)^2 + (1 - 3)^2 + (2 - 4)^2) = 2 * Real.sqrt 6) →
  (x = 6 ∨ x = -2) := 
by
  intros
  sorry

end find_coordinates_of_A_l182_182565


namespace find_larger_number_l182_182008

theorem find_larger_number (L S : ℕ) 
  (h1 : L - S = 1390)
  (h2 : L = 6 * S + 15) : 
  L = 1665 :=
sorry

end find_larger_number_l182_182008


namespace inequality_abc_distinct_positive_l182_182141

theorem inequality_abc_distinct_positive
  (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d) (hbc : b ≠ c) (hbd : b ≠ d) (hcd : c ≠ d) :
  (a^2 / b + b^2 / c + c^2 / d + d^2 / a > a + b + c + d) := 
by
  sorry

end inequality_abc_distinct_positive_l182_182141


namespace part1_part2_l182_182853

-- Define the conditions
def P_condition (a x : ℝ) : Prop := 1 - a / x < 0
def Q_condition (x : ℝ) : Prop := abs (x + 2) < 3

-- First part: Given a = 3, prove the solution set P
theorem part1 (x : ℝ) : P_condition 3 x ↔ 0 < x ∧ x < 3 := by 
  sorry

-- Second part: Prove the range of values for the positive number a
theorem part2 (a : ℝ) (ha : 0 < a) : 
  (∀ x, (P_condition a x → Q_condition x)) → 0 < a ∧ a ≤ 1 := by 
  sorry

end part1_part2_l182_182853


namespace convention_handshakes_l182_182625

-- Introducing the conditions
def companies : ℕ := 5
def reps_per_company : ℕ := 4
def total_reps : ℕ := companies * reps_per_company
def shakes_per_rep : ℕ := total_reps - 1 - (reps_per_company - 1)
def handshakes : ℕ := (total_reps * shakes_per_rep) / 2

-- Statement of the proof
theorem convention_handshakes : handshakes = 160 :=
by
  sorry  -- Proof is not required in this task.

end convention_handshakes_l182_182625


namespace problem1_problem2_l182_182402

-- First problem
theorem problem1 (x : ℝ) (h : x + 1/x = 3) : x^2 + 1/x^2 = 7 := 
by sorry

-- Second problem
theorem problem2 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hx : ∃ k, a^x = k ∧ b^y = k ∧ c^z = k) (h_sum : 1/x + 1/y + 1/z = 0) : a * b * c = 1 := 
by sorry

end problem1_problem2_l182_182402


namespace line_y_intercept_l182_182072

theorem line_y_intercept (x1 y1 x2 y2 : ℝ) (h1 : x1 = 2) (h2 : y1 = -3) (h3 : x2 = 6) (h4 : y2 = 9) :
  ∃ b : ℝ, b = -9 := 
by
  sorry

end line_y_intercept_l182_182072


namespace total_tickets_correct_l182_182509

-- Define the initial number of tickets Tate has
def initial_tickets_Tate : ℕ := 32

-- Define the additional tickets Tate buys
def additional_tickets_Tate : ℕ := 2

-- Calculate the total number of tickets Tate has
def total_tickets_Tate : ℕ := initial_tickets_Tate + additional_tickets_Tate

-- Define the number of tickets Peyton has (half of Tate's total tickets)
def tickets_Peyton : ℕ := total_tickets_Tate / 2

-- Calculate the total number of tickets Tate and Peyton have together
def total_tickets_together : ℕ := total_tickets_Tate + tickets_Peyton

-- Prove that the total number of tickets together equals 51
theorem total_tickets_correct : total_tickets_together = 51 := by
  sorry

end total_tickets_correct_l182_182509


namespace parallelepiped_volume_l182_182779

open Real

noncomputable def volume_parallelepiped
  (a b : ℝ) (angle : ℝ) (S : ℝ) (sin_30 : angle = π / 6) : ℝ :=
  let h := S / (2 * (a + b))
  let base_area := (a * b * sin (π / 6)) / 2
  base_area * h

theorem parallelepiped_volume 
  (a b : ℝ) (S : ℝ) (h : S ≠ 0 ∧ a > 0 ∧ b > 0) :
  volume_parallelepiped a b (π / 6) S (rfl) = (a * b * S) / (4 * (a + b)) :=
by
  sorry

end parallelepiped_volume_l182_182779


namespace range_of_a_l182_182434

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, abs (2 * x - 1) + abs (x + 2) ≥ a^2 + (1 / 2) * a + 2) →
  -1 ≤ a ∧ a ≤ (1 / 2) := by
sorry

end range_of_a_l182_182434


namespace true_weight_of_C_l182_182329

theorem true_weight_of_C (A1 B1 C1 A2 B2 : ℝ) (l1 l2 m1 m2 A B C : ℝ)
  (hA1 : (A + m1) * l1 = (A1 + m2) * l2)
  (hB1 : (B + m1) * l1 = (B1 + m2) * l2)
  (hC1 : (C + m1) * l1 = (C1 + m2) * l2)
  (hA2 : (A2 + m1) * l1 = (A + m2) * l2)
  (hB2 : (B2 + m1) * l1 = (B + m2) * l2) :
  C = (C1 - A1) * Real.sqrt ((A2 - B2) / (A1 - B1)) + 
      (A1 * Real.sqrt (A2 - B2) + A2 * Real.sqrt (A1 - B1)) / 
      (Real.sqrt (A1 - B1) + Real.sqrt (A2 - B2)) :=
sorry

end true_weight_of_C_l182_182329


namespace min_value_x2_y2_z2_l182_182546

theorem min_value_x2_y2_z2 (x y z : ℝ) (h : x^2 + y^2 + z^2 - 3 * x * y * z = 1) :
  x^2 + y^2 + z^2 ≥ 1 := 
sorry

end min_value_x2_y2_z2_l182_182546


namespace rectangle_width_is_14_l182_182766

noncomputable def rectangleWidth (areaOfCircle : ℝ) (length : ℝ) : ℝ :=
  let r := Real.sqrt (areaOfCircle / Real.pi)
  2 * r

theorem rectangle_width_is_14 :
  rectangleWidth 153.93804002589985 18 = 14 :=
by 
  sorry

end rectangle_width_is_14_l182_182766


namespace greatest_sum_consecutive_integers_l182_182808

theorem greatest_sum_consecutive_integers (n : ℤ) (h : n * (n + 1) < 360) : n + (n + 1) ≤ 37 := by
  sorry

end greatest_sum_consecutive_integers_l182_182808


namespace study_days_l182_182111

theorem study_days (chapters worksheets : ℕ) (chapter_hours worksheet_hours daily_study_hours hourly_break
                     snack_breaks_count snack_break time_lunch effective_hours : ℝ)
  (h1 : chapters = 2) 
  (h2 : worksheets = 4) 
  (h3 : chapter_hours = 3) 
  (h4 : worksheet_hours = 1.5) 
  (h5 : daily_study_hours = 4) 
  (h6 : hourly_break = 10 / 60) 
  (h7 : snack_breaks_count = 3) 
  (h8 : snack_break = 10 / 60) 
  (h9 : time_lunch = 30 / 60)
  (h10 : effective_hours = daily_study_hours - (hourly_break * (daily_study_hours - 1)) - (snack_breaks_count * snack_break) - time_lunch)
  : (chapters * chapter_hours + worksheets * worksheet_hours) / effective_hours = 4.8 :=
by
  sorry

end study_days_l182_182111


namespace speed_of_bus_l182_182950

def distance : ℝ := 500.04
def time : ℝ := 20.0
def conversion_factor : ℝ := 3.6

theorem speed_of_bus :
  (distance / time) * conversion_factor = 90.0072 := 
sorry

end speed_of_bus_l182_182950


namespace base_conversion_sum_l182_182355

def digit_C : ℕ := 12
def base_14_value : ℕ := 3 * 14^2 + 5 * 14^1 + 6 * 14^0
def base_13_value : ℕ := 4 * 13^2 + digit_C * 13^1 + 9 * 13^0

theorem base_conversion_sum :
  (base_14_value + base_13_value = 1505) :=
by sorry

end base_conversion_sum_l182_182355


namespace equation_has_three_solutions_l182_182385

theorem equation_has_three_solutions :
  ∃ (s : Finset ℝ), s.card = 3 ∧ ∀ x, x ∈ s ↔ x^2 * (x - 1) * (x - 2) = 0 := 
by
  sorry

end equation_has_three_solutions_l182_182385


namespace solve_linear_equation_l182_182631

theorem solve_linear_equation : ∀ x : ℝ, 4 * (2 * x - 1) = 1 - 3 * (x + 2) → x = -1 / 11 :=
by
  intro x h
  -- Proof to be filled in
  sorry

end solve_linear_equation_l182_182631


namespace range_of_a_l182_182533

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, (1 ≤ x) → ∀ y : ℝ, (1 ≤ y) → (x ≤ y) → (Real.exp (abs (x - a)) ≤ Real.exp (abs (y - a)))) : a ≤ 1 :=
sorry

end range_of_a_l182_182533


namespace card_subsets_l182_182739

theorem card_subsets (A : Finset ℕ) (hA_card : A.card = 3) : (A.powerset.card = 8) :=
sorry

end card_subsets_l182_182739


namespace inequality_holds_l182_182112

theorem inequality_holds (a b c d : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) (h_mul : a * b * c * d = 1) :
  (1 / a) + (1 / b) + (1 / c) + (1 / d) + (12 / (a + b + c + d)) ≥ 7 :=
by
  sorry

end inequality_holds_l182_182112


namespace find_equation_of_line_l182_182382

open Real

noncomputable def equation_of_line : Prop :=
  ∃ c : ℝ, (∀ (x y : ℝ), (3 * x + 5 * y - 4 = 0 ∧ 6 * x - y + 3 = 0 → 2 * x + 3 * y + c = 0)) ∧
  ∃ x y : ℝ, 3 * x + 5 * y - 4 = 0 ∧ 6 * x - y + 3 = 0 ∧
              (2 * x + 3 * y + c = 0 → 6 * x + 9 * y - 7 = 0)

theorem find_equation_of_line : equation_of_line :=
sorry

end find_equation_of_line_l182_182382


namespace find_m_plus_n_l182_182441

theorem find_m_plus_n (x : ℝ) (m n : ℕ) (h₁ : (1 + Real.sin x) / (Real.cos x) = 22 / 7) 
                      (h₂ : (1 + Real.cos x) / (Real.sin x) = m / n) :
                      m + n = 44 := by
  sorry

end find_m_plus_n_l182_182441


namespace apples_given_by_anita_l182_182192

variable (initial_apples current_apples needed_apples : ℕ)

theorem apples_given_by_anita (h1 : initial_apples = 4) 
                               (h2 : needed_apples = 10)
                               (h3 : needed_apples - current_apples = 1) : 
  current_apples - initial_apples = 5 := 
by
  sorry

end apples_given_by_anita_l182_182192


namespace calc_expression_l182_182581

theorem calc_expression : (2019 / 2018) - (2018 / 2019) = 4037 / 4036 := 
by sorry

end calc_expression_l182_182581


namespace polynomial_divisibility_by_6_l182_182788

theorem polynomial_divisibility_by_6 (a b c : ℤ) (h : (a + b + c) % 6 = 0) : (a^5 + b^3 + c) % 6 = 0 :=
sorry

end polynomial_divisibility_by_6_l182_182788


namespace winning_strategy_ping_pong_l182_182835

theorem winning_strategy_ping_pong:
  ∀ {n : ℕ}, n = 18 → (∀ a : ℕ, 1 ≤ a ∧ a ≤ 4 → (∀ k : ℕ, k = 3 * a → (∃ b : ℕ, 1 ≤ b ∧ b ≤ 4 ∧ n - k - b = 18 - (k + b))) → (∃ c : ℕ, c = 3)) :=
by
sorry

end winning_strategy_ping_pong_l182_182835


namespace a2b2_div_ab1_is_square_l182_182599

theorem a2b2_div_ab1_is_square (a b : ℕ) (h₁ : a > 0) (h₂ : b > 0) 
  (h₃ : (ab + 1) ∣ (a^2 + b^2)) : 
  ∃ k : ℕ, (a^2 + b^2) / (ab + 1) = k^2 :=
sorry

end a2b2_div_ab1_is_square_l182_182599


namespace calculation_of_nested_cuberoot_l182_182332

theorem calculation_of_nested_cuberoot (M : Real) (h : 1 < M) : (M^1 / 3 + 1 / 3)^(1 / 3 + 1 / 3)^(1 / 3 + 1 / 3)^(1 / 3 + 1 / 3)^(1 / 3 + 1 / 3) = M^(40 / 81) := 
by 
  sorry

end calculation_of_nested_cuberoot_l182_182332


namespace each_person_bids_five_times_l182_182257

noncomputable def auction_bidding : Prop :=
  let initial_price := 15
  let final_price := 65
  let price_increase_per_bid := 5
  let number_of_bidders := 2
  let total_increase := final_price - initial_price
  let total_bids := total_increase / price_increase_per_bid
  total_bids / number_of_bidders = 5

theorem each_person_bids_five_times : auction_bidding :=
by
  -- The proof will be filled in here.
  sorry

end each_person_bids_five_times_l182_182257


namespace intersection_of_sets_l182_182440

theorem intersection_of_sets (A B : Set ℕ) (hA : A = {1, 2, 3}) (hB : B = {3, 4}) :
  A ∩ B = {3} :=
by
  rw [hA, hB]
  exact sorry

end intersection_of_sets_l182_182440


namespace sunzi_system_l182_182104

variable (x y : ℝ)

theorem sunzi_system :
  (y - x = 4.5) ∧ (x - (1/2) * y = 1) :=
by
  sorry

end sunzi_system_l182_182104


namespace mark_birth_year_proof_l182_182499

-- Conditions
def current_year := 2021
def janice_age := 21
def graham_age := 2 * janice_age
def mark_age := graham_age + 3
def mark_birth_year (current_year : ℕ) (mark_age : ℕ) := current_year - mark_age

-- Statement to prove
theorem mark_birth_year_proof : 
  mark_birth_year current_year mark_age = 1976 := by
  sorry

end mark_birth_year_proof_l182_182499


namespace probability_four_or_more_same_value_l182_182494

theorem probability_four_or_more_same_value :
  let n := 5 -- number of dice
  let d := 10 -- number of sides on each die
  let event := "at least four of the five dice show the same value"
  let probability := (23 : ℚ) / 5000 -- given probability
  n = 5 ∧ d = 10 ∧ event = "at least four of the five dice show the same value" →
  (probability = 23 / 5000) := 
by
  intros
  sorry

end probability_four_or_more_same_value_l182_182494


namespace smallest_triangle_perimeter_l182_182515

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

noncomputable def smallest_possible_prime_perimeter : ℕ :=
  31

theorem smallest_triangle_perimeter :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
                  a > 5 ∧ b > 5 ∧ c > 5 ∧
                  is_prime a ∧ is_prime b ∧ is_prime c ∧
                  triangle_inequality a b c ∧
                  is_prime (a + b + c) ∧
                  a + b + c = smallest_possible_prime_perimeter :=
sorry

end smallest_triangle_perimeter_l182_182515


namespace move_symmetric_point_left_l182_182683

-- Define the original point and the operations
def original_point : ℝ × ℝ := (-2, 3)

def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

def move_left (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1 - d, p.2)

-- Prove the resulting point after the operations
theorem move_symmetric_point_left : move_left (symmetric_point original_point) 2 = (0, -3) :=
by
  sorry

end move_symmetric_point_left_l182_182683


namespace inclination_angle_of_line_3x_sqrt3y_minus1_l182_182165

noncomputable def inclination_angle_of_line (A B C : ℝ) (h : A ≠ 0 ∧ B ≠ 0) : ℝ :=
  let m := -A / B 
  if m = Real.tan (120 * Real.pi / 180) then 120
  else 0 -- This will return 0 if the slope m does not match, for simplifying purposes

theorem inclination_angle_of_line_3x_sqrt3y_minus1 :
  inclination_angle_of_line 3 (Real.sqrt 3) (-1) (by sorry) = 120 := 
sorry

end inclination_angle_of_line_3x_sqrt3y_minus1_l182_182165


namespace num_br_atoms_l182_182001

theorem num_br_atoms (num_br : ℕ) : 
  (1 * 1 + num_br * 80 + 3 * 16 = 129) → num_br = 1 :=
  by
    intro h
    sorry

end num_br_atoms_l182_182001


namespace bisector_length_is_correct_l182_182720

noncomputable def length_of_bisector_of_angle_C
    (BC AC : ℝ)
    (angleC : ℝ)
    (hBC : BC = 5)
    (hAC : AC = 7)
    (hAngleC: angleC = 80) : ℝ := 3.2

theorem bisector_length_is_correct
    (BC AC : ℝ)
    (angleC : ℝ)
    (hBC : BC = 5)
    (hAC : AC = 7)
    (hAngleC: angleC = 80) :
    length_of_bisector_of_angle_C BC AC angleC hBC hAC hAngleC = 3.2 := by
  sorry

end bisector_length_is_correct_l182_182720


namespace parallel_lines_slope_l182_182153

theorem parallel_lines_slope (a : ℝ) : 
  let m1 := - (a / 2)
  let m2 := 3
  ax + 2 * y + 2 = 0 ∧ 3 * x - y - 2 = 0 → m1 = m2 → a = -6 := 
by
  intros
  sorry

end parallel_lines_slope_l182_182153


namespace mean_exterior_angles_l182_182552

theorem mean_exterior_angles (a b c : ℝ) (ha : a = 45) (hb : b = 75) (hc : c = 60) :
  (180 - a + 180 - b + 180 - c) / 3 = 120 :=
by 
  sorry

end mean_exterior_angles_l182_182552


namespace S_sum_l182_182331

def S (n : ℕ) : ℤ :=
  if n % 2 = 0 then -(n / 2)
  else (n + 1) / 2

theorem S_sum :
  S 19 + S 37 + S 52 = 3 :=
by
  sorry

end S_sum_l182_182331


namespace number_of_adults_l182_182276

-- Given constants
def children : ℕ := 200
def price_child (price_adult : ℕ) : ℕ := price_adult / 2
def total_amount : ℕ := 16000

-- Based on the problem conditions
def price_adult := 32

-- The generated proof problem
theorem number_of_adults 
    (price_adult_gt_0 : price_adult > 0)
    (h_price_adult : price_adult = 32)
    (h_total_amount : total_amount = 16000) 
    (h_price_relation : ∀ price_adult, price_adult / 2 * 2 = price_adult) :
  ∃ A : ℕ, 32 * A + 16 * 200 = 16000 ∧ price_child price_adult = 16 := by
  sorry

end number_of_adults_l182_182276


namespace prism_distance_to_plane_l182_182526

theorem prism_distance_to_plane
  (side_length : ℝ)
  (volume : ℝ)
  (h : ℝ)
  (base_is_square : side_length = 6)
  (volume_formula : volume = (1 / 3) * h * (side_length ^ 2)) :
  h = 8 := 
  by sorry

end prism_distance_to_plane_l182_182526


namespace maritza_study_hours_l182_182261

noncomputable def time_to_study_for_citizenship_test (num_mc_questions num_fitb_questions time_mc time_fitb : ℕ) : ℕ :=
  (num_mc_questions * time_mc + num_fitb_questions * time_fitb) / 60

theorem maritza_study_hours :
  time_to_study_for_citizenship_test 30 30 15 25 = 20 :=
by
  sorry

end maritza_study_hours_l182_182261


namespace range_of_k_l182_182602

theorem range_of_k (k : ℝ) : ((∀ x : ℝ, k * x^2 - k * x - 1 < 0) ↔ (-4 < k ∧ k ≤ 0)) :=
sorry

end range_of_k_l182_182602


namespace value_of_a_l182_182545

theorem value_of_a (a : ℝ) (h : abs (2 * a + 1) = 3) :
  a = -2 ∨ a = 1 :=
sorry

end value_of_a_l182_182545


namespace distinct_solutions_diff_l182_182251

theorem distinct_solutions_diff (r s : ℝ) (hr : r > s) 
  (h : ∀ x, (x - 5) * (x + 5) = 25 * x - 125 ↔ x = r ∨ x = s) : r - s = 15 :=
sorry

end distinct_solutions_diff_l182_182251


namespace evaluate_expression_l182_182891

theorem evaluate_expression :
  let a := (1 : ℚ) / 5
  let b := (1 : ℚ) / 3
  let c := (3 : ℚ) / 7
  let d := (1 : ℚ) / 4
  (a + b) / (c - d) = 224 / 75 := by
sorry

end evaluate_expression_l182_182891


namespace percentage_cleared_land_l182_182087

theorem percentage_cleared_land (T C : ℝ) (hT : T = 6999.999999999999) (hC : 0.20 * C + 0.70 * C + 630 = C) :
  (C / T) * 100 = 90 :=
by {
  sorry
}

end percentage_cleared_land_l182_182087


namespace interest_rate_A_to_B_l182_182374

theorem interest_rate_A_to_B :
  ∀ (principal : ℝ) (rate_C : ℝ) (time : ℝ) (gain_B : ℝ) (interest_C : ℝ) (interest_A : ℝ),
    principal = 3500 →
    rate_C = 0.13 →
    time = 3 →
    gain_B = 315 →
    interest_C = principal * rate_C * time →
    gain_B = interest_C - interest_A →
    interest_A = principal * (R / 100) * time →
    R = 10 := by
  sorry

end interest_rate_A_to_B_l182_182374


namespace value_of_y_l182_182217

theorem value_of_y (y : ℚ) : |4 * y - 6| = 0 ↔ y = 3 / 2 :=
by
  sorry

end value_of_y_l182_182217


namespace simplify_abs_eq_l182_182113

variable {x : ℚ}

theorem simplify_abs_eq (hx : |1 - x| = 1 + |x|) : |x - 1| = 1 - x :=
by
  sorry

end simplify_abs_eq_l182_182113


namespace cube_surface_area_l182_182594

theorem cube_surface_area (side_length : ℝ) (h : side_length = 8) : 6 * side_length^2 = 384 :=
by
  rw [h]
  sorry

end cube_surface_area_l182_182594


namespace yarn_cut_parts_l182_182871

-- Define the given conditions
def total_length : ℕ := 10
def crocheted_parts : ℕ := 3
def crocheted_length : ℕ := 6

-- The main problem statement
theorem yarn_cut_parts (total_length crocheted_parts crocheted_length : ℕ) (h1 : total_length = 10) (h2 : crocheted_parts = 3) (h3 : crocheted_length = 6) :
  (total_length / (crocheted_length / crocheted_parts)) = 5 :=
by
  sorry

end yarn_cut_parts_l182_182871


namespace product_of_roots_l182_182409

theorem product_of_roots :
  ∀ (x : ℝ), (|x|^2 - 3 * |x| - 10 = 0) →
  (∃ a b : ℝ, a ≠ b ∧ (|a| = 5 ∧ |b| = 5) ∧ a * b = -25) :=
by {
  sorry
}

end product_of_roots_l182_182409


namespace proportion_equation_l182_182850

theorem proportion_equation (x y : ℝ) (h : 5 * y = 4 * x) : x / y = 5 / 4 :=
by 
  sorry

end proportion_equation_l182_182850


namespace polina_pizza_combinations_correct_l182_182449

def polina_pizza_combinations : Nat :=
  let total_toppings := 5
  let possible_combinations := total_toppings * (total_toppings - 1) / 2
  possible_combinations

theorem polina_pizza_combinations_correct :
  polina_pizza_combinations = 10 :=
by
  sorry

end polina_pizza_combinations_correct_l182_182449


namespace algebra_expression_value_l182_182414

theorem algebra_expression_value (x y : ℝ)
  (h1 : x + y = 3)
  (h2 : x * y = 1) :
  (5 * x + 3) - (2 * x * y - 5 * y) = 16 :=
by
  sorry

end algebra_expression_value_l182_182414


namespace total_people_seated_l182_182310

-- Define the setting
def seated_around_round_table (n : ℕ) : Prop :=
  ∀ a b, 1 ≤ a ∧ a ≤ n ∧ 1 ≤ b ∧ b ≤ n

-- Define the card assignment condition
def assigned_card_numbers (n : ℕ) : Prop :=
  ∀ k, 1 ≤ k ∧ k ≤ n → k = (k % n) + 1

-- Define the condition of equal distances
def equal_distance_condition (n : ℕ) (p1 p2 p3 : ℕ) : Prop :=
  p1 = 31 ∧ p2 = 7 ∧ p3 = 14 ∧
  ((p1 - p2 + n) % n = (p1 - p3 + n) % n ∨
   (p2 - p1 + n) % n = (p3 - p1 + n) % n)

-- Statement of the theorem
theorem total_people_seated (n : ℕ) :
  seated_around_round_table n →
  assigned_card_numbers n →
  equal_distance_condition n 31 7 14 →
  n = 41 :=
by
  sorry

end total_people_seated_l182_182310


namespace cards_choice_ways_l182_182688

theorem cards_choice_ways (S : List Char) (cards : Finset (Char × ℕ)) :
  (∀ c ∈ cards, c.1 ∈ S) ∧
  (∀ (c1 c2 : Char × ℕ), c1 ∈ cards → c2 ∈ cards → c1 ≠ c2 → c1.1 ≠ c2.1) ∧
  (∃ c ∈ cards, c.2 = 1 ∧ c.1 = 'H') →
  (∃ c ∈ cards, c.2 = 1) →
  ∃ (ways : ℕ), ways = 1014 := 
sorry

end cards_choice_ways_l182_182688


namespace subtract_from_40_squared_l182_182763

theorem subtract_from_40_squared : 39 * 39 = 40 * 40 - 79 := by
  sorry

end subtract_from_40_squared_l182_182763


namespace cistern_wet_surface_area_l182_182511

def cistern_length : ℝ := 4
def cistern_width : ℝ := 8
def water_depth : ℝ := 1.25

def area_bottom (l w : ℝ) : ℝ := l * w
def area_pair1 (l h : ℝ) : ℝ := 2 * (l * h)
def area_pair2 (w h : ℝ) : ℝ := 2 * (w * h)
def total_wet_surface_area (l w h : ℝ) : ℝ := area_bottom l w + area_pair1 l h + area_pair2 w h

theorem cistern_wet_surface_area : total_wet_surface_area cistern_length cistern_width water_depth = 62 := 
by 
  sorry

end cistern_wet_surface_area_l182_182511


namespace beaver_stores_60_carrots_l182_182322

theorem beaver_stores_60_carrots (b r : ℕ) (h1 : 4 * b = 5 * r) (h2 : b = r + 3) : 4 * b = 60 :=
by
  sorry

end beaver_stores_60_carrots_l182_182322


namespace hexagon_area_within_rectangle_of_5x4_l182_182709

-- Define the given conditions
def is_rectangle (length width : ℝ) := length > 0 ∧ width > 0

def vertices_touch_midpoints (length width : ℝ) (hexagon_area : ℝ) : Prop :=
  let rectangle_area := length * width
  let triangle_area := (1 / 2) * (length / 2) * (width / 2)
  let total_triangle_area := 4 * triangle_area
  rectangle_area - total_triangle_area = hexagon_area

-- Formulate the main statement to be proved
theorem hexagon_area_within_rectangle_of_5x4 : 
  vertices_touch_midpoints 5 4 10 := 
by
  -- Proof is omitted for this theorem
  sorry

end hexagon_area_within_rectangle_of_5x4_l182_182709


namespace x_squared_y_plus_xy_squared_l182_182067

-- Define the variables and their conditions
variables {x y : ℝ}

-- Define the theorem stating that if xy = 3 and x + y = 5, then x^2y + xy^2 = 15
theorem x_squared_y_plus_xy_squared (h1 : x * y = 3) (h2 : x + y = 5) : x^2 * y + x * y^2 = 15 :=
by {
  sorry
}

end x_squared_y_plus_xy_squared_l182_182067


namespace fraction_of_B_amount_equals_third_of_A_amount_l182_182620

variable (A B : ℝ)
variable (x : ℝ)

theorem fraction_of_B_amount_equals_third_of_A_amount
  (h1 : A + B = 1210)
  (h2 : B = 484)
  (h3 : (1 / 3) * A = x * B) : 
  x = 1 / 2 :=
sorry

end fraction_of_B_amount_equals_third_of_A_amount_l182_182620


namespace sufficient_but_not_necessary_l182_182066

theorem sufficient_but_not_necessary (x : ℝ) (h : 1 / x > 1) : x < 1 := by
  sorry

end sufficient_but_not_necessary_l182_182066


namespace find_a_plus_b_l182_182380

theorem find_a_plus_b (a b : ℤ) (h1 : 2 * a = 0) (h2 : a^2 - b = 25) : a + b = -25 :=
by 
  sorry

end find_a_plus_b_l182_182380


namespace regular_pentagon_cannot_cover_floor_completely_l182_182396

theorem regular_pentagon_cannot_cover_floor_completely
  (hexagon_interior_angle : ℝ)
  (pentagon_interior_angle : ℝ)
  (square_interior_angle : ℝ)
  (triangle_interior_angle : ℝ)
  (hexagon_condition : 360 / hexagon_interior_angle = 3)
  (square_condition : 360 / square_interior_angle = 4)
  (triangle_condition : 360 / triangle_interior_angle = 6)
  (pentagon_condition : 360 / pentagon_interior_angle ≠ 3)
  (pentagon_condition2 : 360 / pentagon_interior_angle ≠ 4)
  (pentagon_condition3 : 360 / pentagon_interior_angle ≠ 6) :
  pentagon_interior_angle = 108 := 
  sorry

end regular_pentagon_cannot_cover_floor_completely_l182_182396


namespace divisor_of_99_l182_182554

def reverse_digits (n : ℕ) : ℕ :=
  -- We assume a placeholder definition for reversing the digits of a number
  sorry

theorem divisor_of_99 (k : ℕ) (h : ∀ n : ℕ, k ∣ n → k ∣ reverse_digits n) : k ∣ 99 :=
  sorry

end divisor_of_99_l182_182554


namespace determine_g_l182_182703

theorem determine_g (t : ℝ) : ∃ (g : ℝ → ℝ), (∀ x y, y = 2 * x - 40 ∧ y = 20 * t - 14 → g t = 10 * t + 13) :=
by
  sorry

end determine_g_l182_182703


namespace ellipse_focus_value_l182_182334

theorem ellipse_focus_value (m : ℝ) (h1 : m > 0) :
  (∃ (x y : ℝ), (x, y) = (-4, 0) ∧ (25 - m^2 = 16)) → m = 3 :=
by
  sorry

end ellipse_focus_value_l182_182334


namespace rearrange_rooks_possible_l182_182018

theorem rearrange_rooks_possible (board : Fin 8 × Fin 8 → Prop) (rooks : Fin 8 → Fin 8 × Fin 8) (painted : Fin 8 × Fin 8 → Prop) :
  (∀ i j : Fin 8, i ≠ j → (rooks i).1 ≠ (rooks j).1 ∧ (rooks i).2 ≠ (rooks j).2) → -- no two rooks are in the same row or column
  (∃ (unpainted_count : ℕ), (unpainted_count = 64 - 27)) → -- 27 squares are painted red
  (∃ new_rooks : Fin 8 → Fin 8 × Fin 8,
    (∀ i : Fin 8, ¬painted (new_rooks i)) ∧ -- all rooks are on unpainted squares
    (∀ i j : Fin 8, i ≠ j → (new_rooks i).1 ≠ (new_rooks j).1 ∧ (new_rooks i).2 ≠ (new_rooks j).2) ∧ -- no two rooks are in the same row or column
    (∃ i : Fin 8, rooks i ≠ new_rooks i)) -- at least one rook has moved
:=
sorry

end rearrange_rooks_possible_l182_182018


namespace impossible_to_create_3_piles_l182_182433

-- Defining similar piles
def similar (x y : ℝ) : Prop :=
  x / y ≤ Real.sqrt 2 ∧ y / x ≤ Real.sqrt 2

-- Main theorem statement
theorem impossible_to_create_3_piles (initial_pile : ℝ) (h_initial : initial_pile > 0) :
  ∀ (x y z : ℝ), 
  x + y + z = initial_pile → 
  similar x y ∧ similar y z ∧ similar z x → 
  false := 
by 
  sorry

end impossible_to_create_3_piles_l182_182433


namespace max_elements_of_S_l182_182033

-- Define the relation on set S and the conditions given
variable {S : Type} (R : S → S → Prop)

-- Lean translation of the conditions
def condition_1 (a b : S) : Prop :=
  (R a b ∨ R b a) ∧ ¬ (R a b ∧ R b a)

def condition_2 (a b c : S) : Prop :=
  R a b ∧ R b c → R c a

-- Define the problem statement:
theorem max_elements_of_S (h1 : ∀ a b : S, condition_1 R a b)
                          (h2 : ∀ a b c : S, condition_2 R a b c) :
  ∃ (n : ℕ), (∀ T : Finset S, T.card ≤ n) ∧ (∃ T : Finset S, T.card = 3) :=
sorry

end max_elements_of_S_l182_182033


namespace positive_difference_balances_l182_182324

noncomputable def laura_balance (L_0 : ℝ) (L_r : ℝ) (L_n : ℕ) (t : ℕ) : ℝ :=
  L_0 * (1 + L_r / L_n) ^ (L_n * t)

noncomputable def mark_balance (M_0 : ℝ) (M_r : ℝ) (t : ℕ) : ℝ :=
  M_0 * (1 + M_r * t)

theorem positive_difference_balances :
  let L_0 := 10000
  let L_r := 0.04
  let L_n := 2
  let t := 20
  let M_0 := 10000
  let M_r := 0.06
  abs ((laura_balance L_0 L_r L_n t) - (mark_balance M_0 M_r t)) = 80.40 :=
by
  sorry

end positive_difference_balances_l182_182324


namespace prime_between_30_and_40_with_remainder_7_l182_182375

theorem prime_between_30_and_40_with_remainder_7 (n : ℕ) 
  (h1 : Nat.Prime n) 
  (h2 : 30 < n) 
  (h3 : n < 40) 
  (h4 : n % 12 = 7) : 
  n = 31 := 
sorry

end prime_between_30_and_40_with_remainder_7_l182_182375


namespace paintable_wall_area_l182_182107

/-- Given 4 bedrooms each with length 15 feet, width 11 feet, and height 9 feet,
and doorways and windows occupying 80 square feet in each bedroom,
prove that the total paintable wall area is 1552 square feet. -/
theorem paintable_wall_area
  (bedrooms : ℕ) (length width height doorway_window_area : ℕ) :
  bedrooms = 4 →
  length = 15 →
  width = 11 →
  height = 9 →
  doorway_window_area = 80 →
  4 * (2 * (length * height) + 2 * (width * height) - doorway_window_area) = 1552 :=
by
  intros bedrooms_eq length_eq width_eq height_eq doorway_window_area_eq
  -- Definition of the problem conditions
  have bedrooms_def : bedrooms = 4 := bedrooms_eq
  have length_def : length = 15 := length_eq
  have width_def : width = 11 := width_eq
  have height_def : height = 9 := height_eq
  have doorway_window_area_def : doorway_window_area = 80 := doorway_window_area_eq
  -- Assertion of the correct answer
  sorry

end paintable_wall_area_l182_182107


namespace x4_plus_inverse_x4_l182_182182

theorem x4_plus_inverse_x4 (x : ℝ) (hx : x ^ 2 + 1 / x ^ 2 = 2) : x ^ 4 + 1 / x ^ 4 = 2 := 
sorry

end x4_plus_inverse_x4_l182_182182


namespace base8_to_base10_362_eq_242_l182_182102

theorem base8_to_base10_362_eq_242 : 
  let digits := [3, 6, 2]
  let base := 8
  let base10_value := (digits[2] * base^0) + (digits[1] * base^1) + (digits[0] * base^2) 
  base10_value = 242 :=
by
  sorry

end base8_to_base10_362_eq_242_l182_182102


namespace largest_even_of_sum_140_l182_182180

theorem largest_even_of_sum_140 :
  ∃ (n : ℕ), 2 * n + 2 * (n + 1) + 2 * (n + 2) + 2 * (n + 3) = 140 ∧ 2 * (n + 3) = 38 :=
by
  sorry

end largest_even_of_sum_140_l182_182180


namespace equivalent_product_lists_l182_182133

-- Definitions of the value assigned to each letter.
def letter_value (c : Char) : ℕ :=
  match c with
  | 'A' => 1
  | 'B' => 2
  | 'C' => 3
  | 'D' => 4
  | 'E' => 5
  | 'F' => 6
  | 'G' => 7
  | 'H' => 8
  | 'I' => 9
  | 'J' => 10
  | 'K' => 11
  | 'L' => 12
  | 'M' => 13
  | 'N' => 14
  | 'O' => 15
  | 'P' => 16
  | 'Q' => 17
  | 'R' => 18
  | 'S' => 19
  | 'T' => 20
  | 'U' => 21
  | 'V' => 22
  | 'W' => 23
  | 'X' => 24
  | 'Y' => 25
  | 'Z' => 26
  | _ => 0  -- We only care about uppercase letters A-Z

def list_product (l : List Char) : ℕ :=
  l.foldl (λ acc c => acc * (letter_value c)) 1

-- Given the list MNOP with their products equals letter values.
def MNOP := ['M', 'N', 'O', 'P']
def BJUZ := ['B', 'J', 'U', 'Z']

-- Lean statement to assert the equivalence of the products.
theorem equivalent_product_lists :
  list_product MNOP = list_product BJUZ :=
by
  sorry

end equivalent_product_lists_l182_182133


namespace scientific_notation_819000_l182_182746

theorem scientific_notation_819000 :
  ∃ (a : ℝ) (n : ℤ), 819000 = a * 10 ^ n ∧ a = 8.19 ∧ n = 5 :=
by
  -- Proof goes here
  sorry

end scientific_notation_819000_l182_182746


namespace customer_paid_correct_amount_l182_182046

noncomputable def cost_price : ℝ := 5565.217391304348
noncomputable def markup_percentage : ℝ := 0.15
noncomputable def markup_amount (cost : ℝ) : ℝ := cost * markup_percentage
noncomputable def final_price (cost : ℝ) (markup : ℝ) : ℝ := cost + markup

theorem customer_paid_correct_amount :
  final_price cost_price (markup_amount cost_price) = 6400 := sorry

end customer_paid_correct_amount_l182_182046


namespace fraction_of_jam_eaten_for_dinner_l182_182032

-- Define the problem
theorem fraction_of_jam_eaten_for_dinner :
  ∃ (J : ℝ) (x : ℝ), 
  J > 0 ∧
  (1 / 3) * J + (x * (2 / 3) * J) + (4 / 7) * J = J ∧
  x = 1 / 7 :=
by
  sorry

end fraction_of_jam_eaten_for_dinner_l182_182032


namespace overall_percent_change_in_stock_l182_182007

noncomputable def stock_change (initial_value : ℝ) : ℝ :=
  let value_after_first_day := 0.85 * initial_value
  let value_after_second_day := 1.25 * value_after_first_day
  (value_after_second_day - initial_value) / initial_value * 100

theorem overall_percent_change_in_stock (x : ℝ) : stock_change x = 6.25 :=
by
  sorry

end overall_percent_change_in_stock_l182_182007


namespace soccer_balls_with_holes_l182_182155

-- Define the total number of soccer balls
def total_soccer_balls : ℕ := 40

-- Define the total number of basketballs
def total_basketballs : ℕ := 15

-- Define the number of basketballs with holes
def basketballs_with_holes : ℕ := 7

-- Define the total number of balls without holes
def total_balls_without_holes : ℕ := 18

-- Prove the number of soccer balls with holes given the conditions
theorem soccer_balls_with_holes : (total_soccer_balls - (total_balls_without_holes - (total_basketballs - basketballs_with_holes))) = 30 := by
  sorry

end soccer_balls_with_holes_l182_182155


namespace cups_of_flour_required_l182_182826

/-- Define the number of cups of sugar and salt required by the recipe. --/
def sugar := 14
def salt := 7
/-- Define the number of cups of flour already added. --/
def flour_added := 2
/-- Define the additional requirement of flour being 3 more cups than salt. --/
def additional_flour_requirement := 3

/-- Main theorem to prove the total amount of flour the recipe calls for. --/
theorem cups_of_flour_required : total_flour = 10 :=
by
  sorry

end cups_of_flour_required_l182_182826


namespace largest_possible_p_l182_182790

theorem largest_possible_p (m n p : ℕ) (h1 : m > 2) (h2 : n > 2) (h3 : p > 2) (h4 : gcd m n = 1) (h5 : gcd n p = 1) (h6 : gcd m p = 1)
  (h7 : (1/m : ℚ) + (1/n : ℚ) + (1/p : ℚ) = 1/2) : p ≤ 42 :=
by sorry

end largest_possible_p_l182_182790


namespace negation_of_p_converse_of_p_inverse_of_p_contrapositive_of_p_original_p_l182_182604

theorem negation_of_p (π : ℝ) (a b c d : ℚ) (h : a * π + b = c * π + d) : a ≠ c ∨ b ≠ d :=
  sorry

theorem converse_of_p (π : ℝ) (a b c d : ℚ) (h : a = c ∧ b = d) : a * π + b = c * π + d :=
  sorry

theorem inverse_of_p (π : ℝ) (a b c d : ℚ) (h : a * π + b ≠ c * π + d) : a ≠ c ∨ b ≠ d :=
  sorry

theorem contrapositive_of_p (π : ℝ) (a b c d : ℚ) (h : a ≠ c ∨ b ≠ d) : a * π + b ≠ c * π + d :=
  sorry

theorem original_p (π : ℝ) (a b c d : ℚ) (h : a * π + b = c * π + d) : a = c ∧ b = d :=
  sorry

end negation_of_p_converse_of_p_inverse_of_p_contrapositive_of_p_original_p_l182_182604


namespace sum_of_solutions_l182_182608

theorem sum_of_solutions (x : ℝ) (h1 : x^2 = 25) : ∃ S : ℝ, S = 0 ∧ (∀ x', x'^2 = 25 → x' = 5 ∨ x' = -5) := 
sorry

end sum_of_solutions_l182_182608


namespace ratio_area_triangle_circle_l182_182730

open Real

theorem ratio_area_triangle_circle
  (l r : ℝ)
  (h : ℝ := sqrt 2 * l)
  (h_eq_perimeter : 2 * l + h = 2 * π * r) :
  (1 / 2 * l^2) / (π * r^2) = (π * (3 - 2 * sqrt 2)) / 2 :=
by
  sorry

end ratio_area_triangle_circle_l182_182730


namespace prob_sum_geq_9_prob_tangent_line_prob_isosceles_triangle_l182_182528

-- Definitions
def fair_die : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Question 1: Probability that a + b >= 9
theorem prob_sum_geq_9 (a b : ℕ) (ha : a ∈ fair_die) (hb : b ∈ fair_die) :
  a + b ≥ 9 → (∃ (valid_outcomes : Finset (ℕ × ℕ)),
    valid_outcomes = {(3, 6), (4, 5), (4, 6), (5, 4), (5, 5), (5, 6), (6, 3), (6, 4), (6, 5), (6, 6)} ∧
    valid_outcomes.card = 10 ∧
    10 / 36 = 5 / 18) :=
sorry

-- Question 2: Probability that the line ax + by + 5 = 0 is tangent to the circle x^2 + y^2 = 1
theorem prob_tangent_line (a b : ℕ) (ha : a ∈ fair_die) (hb : b ∈ fair_die) :
  (∃ (tangent_outcomes : Finset (ℕ × ℕ)),
    tangent_outcomes = {(3, 4), (4, 3)} ∧
    a^2 + b^2 = 25 ∧
    tangent_outcomes.card = 2 ∧
    2 / 36 = 1 / 18) :=
sorry

-- Question 3: Probability that the lengths a, b, and 5 form an isosceles triangle
theorem prob_isosceles_triangle (a b : ℕ) (ha : a ∈ fair_die) (hb : b ∈ fair_die) :
  (∃ (isosceles_outcomes : Finset (ℕ × ℕ)),
    isosceles_outcomes = {(1, 5), (2, 5), (3, 3), (3, 5), (4, 4), (4, 5), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (6, 5), (6, 6)} ∧
    isosceles_outcomes.card = 14 ∧
    14 / 36 = 7 / 18) :=
sorry

end prob_sum_geq_9_prob_tangent_line_prob_isosceles_triangle_l182_182528


namespace solve_system_l182_182432

theorem solve_system : 
  ∃ x y : ℚ, (4 * x + 7 * y = -19) ∧ (4 * x - 5 * y = 17) ∧ x = 1/2 ∧ y = -3 :=
by
  sorry

end solve_system_l182_182432


namespace total_boys_l182_182900

theorem total_boys (T F : ℕ) 
  (avg_all : 37 * T = 39 * 110 + 15 * F) 
  (total_eq : T = 110 + F) : 
  T = 120 := 
sorry

end total_boys_l182_182900


namespace length_width_difference_l182_182704

noncomputable def width : ℝ := Real.sqrt (588 / 8)
noncomputable def length : ℝ := 4 * width
noncomputable def difference : ℝ := length - width

theorem length_width_difference : difference = 25.722 := by
  sorry

end length_width_difference_l182_182704


namespace group_value_21_le_a_lt_41_l182_182044

theorem group_value_21_le_a_lt_41 : 
  (∀ a: ℤ, 21 ≤ a ∧ a < 41 → (21 + 41) / 2 = 31) :=
by 
  sorry

end group_value_21_le_a_lt_41_l182_182044


namespace no_solution_exists_l182_182392

theorem no_solution_exists (x y : ℕ) (hx : 0 < x) (hy : 0 < y) : ¬ (x^y + 3 = y^x ∧ 3 * x^y = y^x + 8) :=
by
  intro h
  obtain ⟨eq1, eq2⟩ := h
  sorry

end no_solution_exists_l182_182392


namespace isosceles_triangle_perimeter_l182_182521

theorem isosceles_triangle_perimeter (a b : ℝ) (h₁ : a = 4) (h₂ : b = 9) (h_iso : ¬(4 + 4 > 9 ∧ 4 + 9 > 4 ∧ 9 + 4 > 4))
  (h_ineq : (9 + 9 > 4) ∧ (9 + 4 > 9) ∧ (4 + 9 > 9)) : 2 * b + a = 22 :=
by sorry

end isosceles_triangle_perimeter_l182_182521


namespace train_length_and_speed_l182_182482

theorem train_length_and_speed (L_bridge : ℕ) (t_cross : ℕ) (t_on_bridge : ℕ) (L_train : ℕ) (v_train : ℕ)
  (h_bridge : L_bridge = 1000)
  (h_t_cross : t_cross = 60)
  (h_t_on_bridge : t_on_bridge = 40)
  (h_crossing_eq : (L_bridge + L_train) / t_cross = v_train)
  (h_on_bridge_eq : L_bridge / t_on_bridge = v_train) : 
  L_train = 200 ∧ v_train = 20 := 
  by
  sorry

end train_length_and_speed_l182_182482


namespace savings_increase_l182_182458

variable (I : ℝ) -- Initial income
variable (E : ℝ) -- Initial expenditure
variable (S : ℝ) -- Initial savings
variable (I_new : ℝ) -- New income
variable (E_new : ℝ) -- New expenditure
variable (S_new : ℝ) -- New savings

theorem savings_increase (h1 : E = 0.75 * I) 
                         (h2 : I_new = 1.20 * I) 
                         (h3 : E_new = 1.10 * E) : 
                         (S_new - S) / S * 100 = 50 :=
by 
  have h4 : S = 0.25 * I := by sorry
  have h5 : E_new = 0.825 * I := by sorry
  have h6 : S_new = 0.375 * I := by sorry
  have increase : (S_new - S) / S * 100 = 50 := by sorry
  exact increase

end savings_increase_l182_182458


namespace find_square_l182_182120

-- Define the conditions as hypotheses
theorem find_square (p : ℕ) (sq : ℕ)
  (h1 : sq + p = 75)
  (h2 : (sq + p) + p = 142) :
  sq = 8 := by
  sorry

end find_square_l182_182120


namespace angle_relation_l182_182406

theorem angle_relation
  (x y z w : ℝ)
  (h_sum : x + y + z + (360 - w) = 360) :
  x = w - y - z :=
by
  sorry

end angle_relation_l182_182406


namespace beads_currently_have_l182_182475

-- Definitions of the conditions
def friends : Nat := 6
def beads_per_bracelet : Nat := 8
def additional_beads_needed : Nat := 12

-- Theorem statement
theorem beads_currently_have : (beads_per_bracelet * friends - additional_beads_needed) = 36 := by
  sorry

end beads_currently_have_l182_182475


namespace total_cost_accurate_l182_182040

def price_iphone: ℝ := 800
def price_iwatch: ℝ := 300
def price_ipad: ℝ := 500

def discount_iphone: ℝ := 0.15
def discount_iwatch: ℝ := 0.10
def discount_ipad: ℝ := 0.05

def tax_iphone: ℝ := 0.07
def tax_iwatch: ℝ := 0.05
def tax_ipad: ℝ := 0.06

def cashback: ℝ := 0.02

theorem total_cost_accurate:
  let discounted_auction (price: ℝ) (discount: ℝ) := price * (1 - discount)
  let taxed_auction (price: ℝ) (tax: ℝ) := price * (1 + tax)
  let total_cost :=
    let discount_iphone_cost := discounted_auction price_iphone discount_iphone
    let discount_iwatch_cost := discounted_auction price_iwatch discount_iwatch
    let discount_ipad_cost := discounted_auction price_ipad discount_ipad
    
    let tax_iphone_cost := taxed_auction discount_iphone_cost tax_iphone
    let tax_iwatch_cost := taxed_auction discount_iwatch_cost tax_iwatch
    let tax_ipad_cost := taxed_auction discount_ipad_cost tax_ipad
    
    let total_price := tax_iphone_cost + tax_iwatch_cost + tax_ipad_cost
    total_price * (1 - cashback)
  total_cost = 1484.31 := 
  by sorry

end total_cost_accurate_l182_182040


namespace scientific_notation_l182_182717

-- Given radius of a water molecule
def radius_of_water_molecule := 0.00000000192

-- Required scientific notation
theorem scientific_notation : radius_of_water_molecule = 1.92 * 10 ^ (-9) :=
by
  sorry

end scientific_notation_l182_182717


namespace height_at_age_10_is_around_146_l182_182079

noncomputable def predicted_height (x : ℝ) : ℝ :=
  7.2 * x + 74

theorem height_at_age_10_is_around_146 :
  abs (predicted_height 10 - 146) < ε :=
by
  let ε := 10
  sorry

end height_at_age_10_is_around_146_l182_182079


namespace fewest_printers_l182_182954

theorem fewest_printers (x y : ℕ) (h1 : 375 * x = 150 * y) : x + y = 7 :=
  sorry

end fewest_printers_l182_182954


namespace deepaks_age_l182_182403

theorem deepaks_age (R D : ℕ) (h1 : R / D = 5 / 2) (h2 : R + 6 = 26) : D = 8 := 
sorry

end deepaks_age_l182_182403


namespace radius_of_circumcircle_of_triangle_l182_182858

theorem radius_of_circumcircle_of_triangle (a b c : ℝ)
  (h : a^2 + b^2 + c^2 - 6*a - 8*b - 10*c + 50 = 0) :
  a = 3 ∧ b = 4 ∧ c = 5 ∧ (∃ (R : ℝ), R = 2.5) :=
by {
  sorry
}

end radius_of_circumcircle_of_triangle_l182_182858


namespace sum_of_distances_to_focus_is_ten_l182_182223

theorem sum_of_distances_to_focus_is_ten (P : ℝ × ℝ) (A B F : ℝ × ℝ)
  (hP : P = (2, 1))
  (hA : A.1^2 = 12 * A.2)
  (hB : B.1^2 = 12 * B.2)
  (hMidpoint : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (hFocus : F = (3, 0)) :
  |A.1 - F.1| + |B.1 - F.1| = 10 :=
by
  sorry

end sum_of_distances_to_focus_is_ten_l182_182223


namespace point_on_curve_l182_182195

-- Define the parametric curve equations
def onCurve (θ : ℝ) (x y : ℝ) : Prop :=
  x = Real.sin (2 * θ) ∧ y = Real.cos θ + Real.sin θ

-- Define the general form of the curve
def curveEquation (x y : ℝ) : Prop :=
  y^2 = 1 + x

-- The proof statement
theorem point_on_curve : 
  curveEquation (-3/4) (1/2) ∧ ∃ θ : ℝ, onCurve θ (-3/4) (1/2) :=
by
  sorry

end point_on_curve_l182_182195


namespace sum_of_reciprocal_of_roots_l182_182345

theorem sum_of_reciprocal_of_roots :
  ∀ x1 x2 : ℝ, (x1 * x2 = 2) → (x1 + x2 = 3) → (1 / x1 + 1 / x2 = 3 / 2) :=
by
  intros x1 x2 h_prod h_sum
  sorry

end sum_of_reciprocal_of_roots_l182_182345


namespace sum_of_integers_l182_182970

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 10) (h3 : x * y = 56) : x + y = 18 := 
sorry

end sum_of_integers_l182_182970


namespace initial_apples_l182_182290

theorem initial_apples (Initially_Apples : ℕ) (Added_Apples : ℕ) (Total_Apples : ℕ)
  (h1 : Added_Apples = 8) (h2 : Total_Apples = 17) : Initially_Apples = 9 :=
by
  have h3 : Added_Apples + Initially_Apples = Total_Apples := by
    sorry
  linarith

end initial_apples_l182_182290


namespace coloring_ways_l182_182512

-- Definitions for colors
inductive Color
| red
| green

open Color

-- Definition of the coloring function
def color (n : ℕ) : Color := sorry

-- Conditions:
-- 1. Each positive integer is colored either red or green
def condition1 (n : ℕ) : n > 0 → (color n = red ∨ color n = green) := sorry

-- 2. The sum of any two different red numbers is a red number
def condition2 (r1 r2 : ℕ) : r1 ≠ r2 → color r1 = red → color r2 = red → color (r1 + r2) = red := sorry

-- 3. The sum of any two different green numbers is a green number
def condition3 (g1 g2 : ℕ) : g1 ≠ g2 → color g1 = green → color g2 = green → color (g1 + g2) = green := sorry

-- The required theorem
theorem coloring_ways : ∃! (f : ℕ → Color), 
  (∀ n, n > 0 → (f n = red ∨ f n = green)) ∧ 
  (∀ r1 r2, r1 ≠ r2 → f r1 = red → f r2 = red → f (r1 + r2) = red) ∧
  (∀ g1 g2, g1 ≠ g2 → f g1 = green → f g2 = green → f (g1 + g2) = green) :=
sorry

end coloring_ways_l182_182512


namespace lecture_room_configuration_l182_182472

theorem lecture_room_configuration (m n : ℕ) (boys_per_row girls_per_column unoccupied_chairs : ℕ) :
    boys_per_row = 6 →
    girls_per_column = 8 →
    unoccupied_chairs = 15 →
    (m * n = boys_per_row * m + girls_per_column * n + unoccupied_chairs) →
    (m = 71 ∧ n = 7) ∨
    (m = 29 ∧ n = 9) ∨
    (m = 17 ∧ n = 13) ∨
    (m = 15 ∧ n = 15) ∨
    (m = 11 ∧ n = 27) ∨
    (m = 9 ∧ n = 69) :=
by
  intros h1 h2 h3 h4
  sorry

end lecture_room_configuration_l182_182472


namespace proof_problem_l182_182043

/-- 
  Given:
  - r, j, z are Ryan's, Jason's, and Zachary's earnings respectively.
  - Zachary sold 40 games at $5 each.
  - Jason received 30% more money than Zachary.
  - The total amount of money received by all three is $770.
  Prove:
  - Ryan received $50 more than Jason.
--/
def problem_statement : Prop :=
  ∃ (r j z : ℕ), 
    z = 40 * 5 ∧
    j = z + z * 30 / 100 ∧
    r + j + z = 770 ∧ 
    r - j = 50

theorem proof_problem : problem_statement :=
by 
  sorry

end proof_problem_l182_182043


namespace triangle_length_l182_182832

theorem triangle_length (DE DF : ℝ) (Median_to_EF : ℝ) (EF : ℝ) :
  DE = 2 ∧ DF = 3 ∧ Median_to_EF = EF → EF = (13:ℝ).sqrt / 5 := by
  sorry

end triangle_length_l182_182832


namespace victor_draw_order_count_l182_182803

-- Definitions based on the problem conditions
def num_piles : ℕ := 3
def num_cards_per_pile : ℕ := 3
def total_cards : ℕ := num_piles * num_cards_per_pile

-- The cardinality of the set of valid sequences where within each pile cards must be drawn in order
def valid_sequences_count : ℕ :=
  Nat.factorial total_cards / (Nat.factorial num_cards_per_pile ^ num_piles)

-- Now we state the problem: proving the valid sequences count is 1680
theorem victor_draw_order_count :
  valid_sequences_count = 1680 :=
by
  sorry

end victor_draw_order_count_l182_182803


namespace sum_zero_implies_product_terms_nonpositive_l182_182906

theorem sum_zero_implies_product_terms_nonpositive (a b c : ℝ) (h : a + b + c = 0) : 
  a * b + a * c + b * c ≤ 0 := 
by 
  sorry

end sum_zero_implies_product_terms_nonpositive_l182_182906


namespace circles_tangent_l182_182658

def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 16*y - 48 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 8*y - 44 = 0

theorem circles_tangent :
  ∃ (x y : ℝ), circle1 x y ∧ circle2 x y :=
sorry

end circles_tangent_l182_182658


namespace nonnegative_interval_l182_182878

theorem nonnegative_interval (x : ℝ) : 
  (x - 8 * x^2 + 16 * x^3) / (9 - x^3) ≥ 0 ↔ (x ≥ 0 ∧ x < 3) :=
by sorry

end nonnegative_interval_l182_182878


namespace people_in_room_l182_182606

theorem people_in_room (total_chairs seated_chairs total_people : ℕ) 
  (h1 : 3 * total_people = 5 * seated_chairs)
  (h2 : 4 * total_chairs = 5 * seated_chairs) 
  (h3 : total_chairs - seated_chairs = 8) : 
  total_people = 54 :=
by
  sorry

end people_in_room_l182_182606


namespace distance_between_foci_of_hyperbola_l182_182659

theorem distance_between_foci_of_hyperbola {x y : ℝ} (h : x ^ 2 - 4 * y ^ 2 = 4) :
  ∃ c : ℝ, 2 * c = 2 * Real.sqrt 5 :=
sorry

end distance_between_foci_of_hyperbola_l182_182659


namespace bulls_on_farm_l182_182947

theorem bulls_on_farm (C B : ℕ) (h1 : C / B = 10 / 27) (h2 : C + B = 555) : B = 405 :=
sorry

end bulls_on_farm_l182_182947


namespace charlie_metal_storage_l182_182913

theorem charlie_metal_storage (total_needed : ℕ) (amount_to_buy : ℕ) (storage : ℕ) 
    (h1 : total_needed = 635) 
    (h2 : amount_to_buy = 359) 
    (h3 : total_needed = storage + amount_to_buy) : 
    storage = 276 := 
sorry

end charlie_metal_storage_l182_182913


namespace A_beats_B_by_7_seconds_l182_182159

noncomputable def speed_A : ℝ := 200 / 33
noncomputable def distance_A : ℝ := 200
noncomputable def time_A : ℝ := 33

noncomputable def distance_B : ℝ := 200
noncomputable def distance_B_at_time_A : ℝ := 165

-- B's speed is calculated at the moment A finishes the race
noncomputable def speed_B : ℝ := distance_B_at_time_A / time_A
noncomputable def time_B : ℝ := distance_B / speed_B

-- Prove that A beats B by 7 seconds
theorem A_beats_B_by_7_seconds : time_B - time_A = 7 := 
by 
  -- Proof goes here, assume all definitions and variables are correct.
  sorry

end A_beats_B_by_7_seconds_l182_182159


namespace number_of_people_is_8_l182_182989

noncomputable def find_number_of_people (avg_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) (weight_diff : ℝ) (n : ℕ) :=
  avg_increase = weight_diff / n ∧ old_weight = 70 ∧ new_weight = 90 ∧ weight_diff = new_weight - old_weight → n = 8

theorem number_of_people_is_8 :
  ∃ n : ℕ, find_number_of_people 2.5 70 90 20 n :=
by
  use 8
  sorry

end number_of_people_is_8_l182_182989


namespace smallest_k_exists_l182_182428

theorem smallest_k_exists : ∃ (k : ℕ), k > 0 ∧ (∃ (n m : ℕ), n > 0 ∧ m > 0 ∧ k = 19^n - 5^m) ∧ k = 14 :=
by 
  sorry

end smallest_k_exists_l182_182428


namespace total_sandwiches_l182_182583

theorem total_sandwiches :
  let billy := 49
  let katelyn := billy + 47
  let chloe := katelyn / 4
  billy + katelyn + chloe = 169 :=
by
  sorry

end total_sandwiches_l182_182583


namespace integer_satisfies_inequality_l182_182725

theorem integer_satisfies_inequality (n : ℤ) : 
  (3 : ℚ) / 10 < n / 20 ∧ n / 20 < 2 / 5 → n = 7 :=
sorry

end integer_satisfies_inequality_l182_182725


namespace correct_relationship_in_triangle_l182_182204

theorem correct_relationship_in_triangle (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.sin (A + B) = Real.sin C :=
sorry

end correct_relationship_in_triangle_l182_182204


namespace line_eqn_with_given_conditions_l182_182885

theorem line_eqn_with_given_conditions : 
  ∃(m c : ℝ), (∀ x y : ℝ, y = m*x + c → x + y - 3 = 0) ↔ 
  ∀ x y, x + y = 3 :=
sorry

end line_eqn_with_given_conditions_l182_182885


namespace arithmetic_mean_difference_l182_182211

-- Definitions and conditions
variable (p q r : ℝ)
variable (h1 : (p + q) / 2 = 10)
variable (h2 : (q + r) / 2 = 26)

-- Theorem statement
theorem arithmetic_mean_difference : r - p = 32 := by
  -- Proof goes here
  sorry

end arithmetic_mean_difference_l182_182211


namespace problem_a_lt_c_lt_b_l182_182039

noncomputable def a : ℝ := Real.sin (14 * Real.pi / 180) + Real.cos (14 * Real.pi / 180)
noncomputable def b : ℝ := Real.sin (16 * Real.pi / 180) + Real.cos (16 * Real.pi / 180)
noncomputable def c : ℝ := Real.sqrt 6 / 2

theorem problem_a_lt_c_lt_b : a < c ∧ c < b := 
by {
  sorry
}

end problem_a_lt_c_lt_b_l182_182039


namespace find_angle_A_find_area_l182_182749

-- Definitions
variables {A B C : ℝ} {a b c : ℝ}

-- Given conditions
def law_c1 (A B C : ℝ) (a b c : ℝ) : Prop :=
  a * Real.cos C + c * Real.cos A = -2 * b * Real.cos A

def law_c2 (a : ℝ) : Prop := a = 2 * Real.sqrt 3
def law_c3 (b c : ℝ) : Prop := b + c = 4

-- Questions
theorem find_angle_A (A B C : ℝ) (a b c : ℝ) (h1 : law_c1 A B C a b c) (h2 : law_c2 a) (h3 : law_c3 b c) : 
  A = 2 * Real.pi / 3 :=
sorry

theorem find_area (A B C : ℝ) (a b c : ℝ) (h1 : law_c1 A B C a b c) (h2 : law_c2 a) (h3 : law_c3 b c)
  (hA : A = 2 * Real.pi / 3) : 
  (1/2) * b * c * Real.sin A = Real.sqrt 3 :=
sorry

end find_angle_A_find_area_l182_182749


namespace ladder_leaning_distance_l182_182125

variable (m f h : ℝ)
variable (f_pos : f > 0) (h_pos : h > 0)

def distance_to_wall_upper_bound : ℝ := 12.46
def distance_to_wall_lower_bound : ℝ := 8.35

theorem ladder_leaning_distance (m f h : ℝ) (f_pos : f > 0) (h_pos : h > 0) :
  ∃ x : ℝ, x = 12.46 ∨ x = 8.35 := 
sorry

end ladder_leaning_distance_l182_182125


namespace cost_difference_proof_l182_182098

noncomputable def sailboat_daily_rent : ℕ := 60
noncomputable def ski_boat_hourly_rent : ℕ := 80
noncomputable def sailboat_hourly_fuel_cost : ℕ := 10
noncomputable def ski_boat_hourly_fuel_cost : ℕ := 20
noncomputable def discount : ℕ := 10

noncomputable def rent_time : ℕ := 3
noncomputable def rent_days : ℕ := 2

noncomputable def ken_sailboat_rent_cost :=
  sailboat_daily_rent * rent_days - sailboat_daily_rent * discount / 100

noncomputable def ken_sailboat_fuel_cost :=
  sailboat_hourly_fuel_cost * rent_time * rent_days

noncomputable def ken_total_cost :=
  ken_sailboat_rent_cost + ken_sailboat_fuel_cost

noncomputable def aldrich_ski_boat_rent_cost :=
  ski_boat_hourly_rent * rent_time * rent_days - (ski_boat_hourly_rent * rent_time * discount / 100)

noncomputable def aldrich_ski_boat_fuel_cost :=
  ski_boat_hourly_fuel_cost * rent_time * rent_days

noncomputable def aldrich_total_cost :=
  aldrich_ski_boat_rent_cost + aldrich_ski_boat_fuel_cost

noncomputable def cost_difference :=
  aldrich_total_cost - ken_total_cost

theorem cost_difference_proof : cost_difference = 402 := by
  sorry

end cost_difference_proof_l182_182098


namespace laptop_selection_l182_182744

open Nat

theorem laptop_selection :
  ∃ (n : ℕ), n = (choose 4 2) * (choose 5 1) + (choose 4 1) * (choose 5 2) := 
sorry

end laptop_selection_l182_182744


namespace multiply_divide_repeating_decimals_l182_182600

theorem multiply_divide_repeating_decimals :
  (8 * (1 / 3) / 1) = 8 / 3 := by
  sorry

end multiply_divide_repeating_decimals_l182_182600


namespace intersection_A_B_l182_182090

def set_A (x : ℝ) : Prop := 2 * x + 1 > 0
def set_B (x : ℝ) : Prop := abs (x - 1) < 2

theorem intersection_A_B : 
  {x : ℝ | set_A x} ∩ {x : ℝ | set_B x} = {x : ℝ | -1/2 < x ∧ x < 3} :=
by
  sorry

end intersection_A_B_l182_182090


namespace students_same_group_in_all_lessons_l182_182459

theorem students_same_group_in_all_lessons (students : Fin 28 → Fin 3 × Fin 3 × Fin 3) :
  ∃ (i j : Fin 28), i ≠ j ∧ students i = students j :=
by
  sorry

end students_same_group_in_all_lessons_l182_182459


namespace quadrilateral_area_l182_182503

noncomputable def AreaOfQuadrilateral (AB AC AD : ℝ) : ℝ :=
  let BC := Real.sqrt (AC^2 - AB^2)
  let CD := Real.sqrt (AC^2 - AD^2)
  let AreaABC := (1 / 2) * AB * BC
  let AreaACD := (1 / 2) * AD * CD
  AreaABC + AreaACD

theorem quadrilateral_area :
  AreaOfQuadrilateral 5 13 12 = 60 :=
by
  sorry

end quadrilateral_area_l182_182503


namespace side_of_larger_square_l182_182146

theorem side_of_larger_square (s S : ℕ) (h₁ : s = 5) (h₂ : S^2 = 4 * s^2) : S = 10 := 
by sorry

end side_of_larger_square_l182_182146


namespace measure_of_angle_Q_l182_182598

theorem measure_of_angle_Q (a b c d e Q : ℝ)
  (ha : a = 138) (hb : b = 85) (hc : c = 130) (hd : d = 120) (he : e = 95)
  (h_hex : a + b + c + d + e + Q = 720) : 
  Q = 152 :=
by
  rw [ha, hb, hc, hd, he] at h_hex
  linarith

end measure_of_angle_Q_l182_182598


namespace calculate_crayons_lost_l182_182316

def initial_crayons := 440
def given_crayons := 111
def final_crayons := 223

def crayons_left_after_giving := initial_crayons - given_crayons
def crayons_lost := crayons_left_after_giving - final_crayons

theorem calculate_crayons_lost : crayons_lost = 106 :=
  by
    sorry

end calculate_crayons_lost_l182_182316


namespace basketball_team_wins_l182_182456

-- Define the known quantities
def games_won_initial : ℕ := 60
def games_total_initial : ℕ := 80
def games_left : ℕ := 50
def total_games : ℕ := games_total_initial + games_left
def desired_win_fraction : ℚ := 3 / 4

-- The main goal: Prove that the team must win 38 of the remaining 50 games to reach the desired win fraction
theorem basketball_team_wins :
  ∃ x : ℕ, x = 38 ∧ (games_won_initial + x : ℚ) / total_games = desired_win_fraction :=
by
  sorry

end basketball_team_wins_l182_182456


namespace money_saved_l182_182173

noncomputable def total_savings :=
  let fox_price := 15
  let pony_price := 18
  let num_fox_pairs := 3
  let num_pony_pairs := 2
  let total_discount_rate := 0.22
  let pony_discount_rate := 0.10999999999999996
  let fox_discount_rate := total_discount_rate - pony_discount_rate
  let fox_savings := fox_price * fox_discount_rate * num_fox_pairs
  let pony_savings := pony_price * pony_discount_rate * num_pony_pairs
  fox_savings + pony_savings

theorem money_saved :
  total_savings = 8.91 :=
by
  -- We assume the savings calculations are correct as per the problem statement
  sorry

end money_saved_l182_182173


namespace Miriam_gave_brother_60_marbles_l182_182795

def Miriam_current_marbles : ℕ := 30
def Miriam_initial_marbles : ℕ := 300
def brother_marbles (B : ℕ) : Prop := B = 60
def sister_marbles (B : ℕ) : ℕ := 2 * B
def friend_marbles : ℕ := 90
def total_given_away_marbles (B : ℕ) : ℕ := B + sister_marbles B + friend_marbles

theorem Miriam_gave_brother_60_marbles (B : ℕ) 
    (h1 : Miriam_current_marbles = 30) 
    (h2 : Miriam_initial_marbles = 300)
    (h3 : total_given_away_marbles B = Miriam_initial_marbles - Miriam_current_marbles) : 
    brother_marbles B :=
by 
    sorry

end Miriam_gave_brother_60_marbles_l182_182795


namespace gray_areas_trees_count_l182_182296

noncomputable def totalTreesInGrayAreas (T : ℕ) (white1 white2 white3 : ℕ) : ℕ :=
  let gray2 := T - white2
  let gray3 := T - white3
  gray2 + gray3

theorem gray_areas_trees_count (T : ℕ) :
  T = 100 → totalTreesInGrayAreas T 100 82 90 = 26 :=
by sorry

end gray_areas_trees_count_l182_182296


namespace seconds_in_minutes_l182_182708

-- Define the concepts of minutes and seconds
def minutes (m : ℝ) : ℝ := m

def seconds (s : ℝ) : ℝ := s

-- Define the given values
def conversion_factor : ℝ := 60 -- seconds in one minute

def given_minutes : ℝ := 12.5

-- State the theorem
theorem seconds_in_minutes : seconds (given_minutes * conversion_factor) = 750 := 
by
sorry

end seconds_in_minutes_l182_182708


namespace expected_waiting_time_correct_l182_182980

noncomputable def combined_average_bites_per_5_minutes := 6
def average_waiting_time_for_first_bite_in_seconds : ℝ := 50

theorem expected_waiting_time_correct :
  (1 / combined_average_bites_per_5_minutes) * 300 = average_waiting_time_for_first_bite_in_seconds :=
by
  sorry

end expected_waiting_time_correct_l182_182980


namespace fixed_point_of_line_l182_182527

theorem fixed_point_of_line (k : ℝ) : ∃ (p : ℝ × ℝ), p = (-3, 4) ∧ ∀ (x y : ℝ), (y - 4 = -k * (x + 3)) → (-3, 4) = (x, y) :=
by
  sorry

end fixed_point_of_line_l182_182527


namespace power_sum_evaluation_l182_182549

theorem power_sum_evaluation :
  (-1)^(4^3) + 2^(3^2) = 513 :=
by
  sorry

end power_sum_evaluation_l182_182549


namespace concave_side_probability_l182_182234

theorem concave_side_probability (tosses : ℕ) (frequency_convex : ℝ) (htosses : tosses = 1000) (hfrequency : frequency_convex = 0.44) :
  ∀ probability_concave : ℝ, probability_concave = 1 - frequency_convex → probability_concave = 0.56 :=
by
  intros probability_concave h
  rw [hfrequency] at h
  rw [h]
  norm_num
  done

end concave_side_probability_l182_182234


namespace neither_happy_nor_sad_boys_is_5_l182_182865

-- Define the total number of children
def total_children := 60

-- Define the number of happy children
def happy_children := 30

-- Define the number of sad children
def sad_children := 10

-- Define the number of neither happy nor sad children
def neither_happy_nor_sad_children := 20

-- Define the number of boys
def boys := 17

-- Define the number of girls
def girls := 43

-- Define the number of happy boys
def happy_boys := 6

-- Define the number of sad girls
def sad_girls := 4

-- Define the number of neither happy nor sad boys
def neither_happy_nor_sad_boys := boys - (happy_boys + (sad_children - sad_girls))

theorem neither_happy_nor_sad_boys_is_5 :
  neither_happy_nor_sad_boys = 5 :=
by
  -- This skips the proof
  sorry

end neither_happy_nor_sad_boys_is_5_l182_182865


namespace second_person_days_l182_182718

theorem second_person_days (h1 : 2 * (1 : ℝ) / 8 = 1) 
                           (h2 : 1 / 24 + x / 24 = 1 / 8) : x = 1 / 12 :=
sorry

end second_person_days_l182_182718


namespace geometric_sequence_sum_l182_182473

theorem geometric_sequence_sum 
  (a : ℕ → ℝ) -- a_n is a sequence of real numbers
  (q : ℝ) -- q is the common ratio
  (h1 : a 1 + a 2 = 20) -- first condition
  (h2 : a 3 + a 4 = 80) -- second condition
  (h_geom : ∀ n, a (n + 1) = a n * q) -- property of geometric sequence
  : a 5 + a 6 = 320 := 
sorry

end geometric_sequence_sum_l182_182473


namespace tan_increasing_interval_l182_182706

noncomputable def increasing_interval (k : ℤ) : Set ℝ := 
  {x | (k * Real.pi / 2 - 5 * Real.pi / 12 < x) ∧ (x < k * Real.pi / 2 + Real.pi / 12)}

theorem tan_increasing_interval (k : ℤ) : 
  ∀ x : ℝ, (k * Real.pi / 2 - 5 * Real.pi / 12 < x) ∧ (x < k * Real.pi / 2 + Real.pi / 12) ↔ 
    (∃ y, y = (2 * x + Real.pi / 3) ∧ Real.tan y > Real.tan (2 * x + Real.pi / 3 - 1e-6)) :=
sorry

end tan_increasing_interval_l182_182706


namespace find_number_l182_182359

-- Define the main problem statement
theorem find_number (x : ℝ) (h : 0.50 * x = 0.80 * 150 + 80) : x = 400 := by
  sorry

end find_number_l182_182359


namespace distance_between_ann_and_glenda_l182_182639

def ann_distance : ℝ := 
  let speed1 := 6
  let time1 := 1
  let speed2 := 8
  let time2 := 1
  let break1 := 0
  let speed3 := 4
  let time3 := 1
  speed1 * time1 + speed2 * time2 + break1 * 0 + speed3 * time3

def glenda_distance : ℝ := 
  let speed1 := 8
  let time1 := 1
  let speed2 := 5
  let time2 := 1
  let break1 := 0
  let speed3 := 9
  let back_time := 0.5
  let back_distance := speed3 * back_time
  let continue_time := 0.5
  let continue_distance := speed3 * continue_time
  speed1 * time1 + speed2 * time2 + break1 * 0 + (-back_distance) + continue_distance

theorem distance_between_ann_and_glenda : 
  ann_distance + glenda_distance = 35.5 := 
by 
  sorry

end distance_between_ann_and_glenda_l182_182639


namespace crayons_difference_l182_182419

theorem crayons_difference (total_crayons : ℕ) (given_crayons : ℕ) (lost_crayons : ℕ) (h1 : total_crayons = 589) (h2 : given_crayons = 571) (h3 : lost_crayons = 161) : (given_crayons - lost_crayons) = 410 := by
  sorry

end crayons_difference_l182_182419


namespace hexagonal_prism_surface_area_l182_182073

theorem hexagonal_prism_surface_area (h : ℝ) (a : ℝ) (H_h : h = 6) (H_a : a = 4) : 
  let base_area := 2 * (3 * a^2 * (Real.sqrt 3) / 2)
  let lateral_area := 6 * a * h
  let total_area := lateral_area + base_area
  total_area = 48 * (3 + Real.sqrt 3) :=
by
  -- let base_area := 2 * (3 * a^2 * (Real.sqrt 3) / 2)
  -- let lateral_area := 6 * a * h
  -- let total_area := lateral_area + base_area
  -- total_area = 48 * (3 + Real.sqrt 3)
  sorry

end hexagonal_prism_surface_area_l182_182073


namespace south_side_students_count_l182_182466

variables (N : ℕ)
def students_total := 41
def difference := 3

theorem south_side_students_count (N : ℕ) (h₁ : 2 * N + difference = students_total) : N + difference = 22 :=
sorry

end south_side_students_count_l182_182466


namespace relay_go_match_outcomes_l182_182213

theorem relay_go_match_outcomes : (Nat.choose 14 7) = 3432 := by
  sorry

end relay_go_match_outcomes_l182_182213


namespace pool_ratio_three_to_one_l182_182140

theorem pool_ratio_three_to_one (P : ℕ) (B B' : ℕ) (k : ℕ) :
  (P = 5 * B + 2) → (k * P = 5 * B' + 1) → k = 3 :=
by
  intros h1 h2
  sorry

end pool_ratio_three_to_one_l182_182140


namespace g_half_eq_neg_one_l182_182734

noncomputable def f (x : ℝ) : ℝ := 2^x
noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem g_half_eq_neg_one : g (1/2) = -1 := by 
  sorry

end g_half_eq_neg_one_l182_182734


namespace roja_alone_time_l182_182416

theorem roja_alone_time (W : ℝ) (R : ℝ) :
  (1 / 60 + 1 / R = 1 / 35) → (R = 210) :=
by
  intros
  -- Proof goes here
  sorry

end roja_alone_time_l182_182416


namespace total_cats_in_meow_and_paw_l182_182676

-- Define the conditions
def CatsInCatCafeCool : Nat := 5
def CatsInCatCafePaw : Nat := 2 * CatsInCatCafeCool
def CatsInCatCafeMeow : Nat := 3 * CatsInCatCafePaw

-- Define the total number of cats in Cat Cafe Meow and Cat Cafe Paw
def TotalCats : Nat := CatsInCatCafeMeow + CatsInCatCafePaw

-- The theorem stating the problem
theorem total_cats_in_meow_and_paw : TotalCats = 40 :=
by
  sorry

end total_cats_in_meow_and_paw_l182_182676


namespace abs_diff_31st_term_l182_182886

-- Define the sequences C and D
def C (n : ℕ) : ℤ := 40 + 20 * (n - 1)
def D (n : ℕ) : ℤ := 40 - 20 * (n - 1)

-- Question: What is the absolute value of the difference between the 31st term of C and D?
theorem abs_diff_31st_term : |C 31 - D 31| = 1200 := by
  sorry

end abs_diff_31st_term_l182_182886


namespace purely_imaginary_complex_l182_182540

theorem purely_imaginary_complex (a : ℝ) : (a - 2) = 0 → a = 2 :=
by
  intro h
  exact eq_of_sub_eq_zero h

end purely_imaginary_complex_l182_182540


namespace exists_common_element_l182_182888

variable (S : Fin 2011 → Set ℤ)
variable (h1 : ∀ i, (S i).Nonempty)
variable (h2 : ∀ i j, (S i ∩ S j).Nonempty)

theorem exists_common_element :
  ∃ a : ℤ, ∀ i, a ∈ S i :=
by {
  sorry
}

end exists_common_element_l182_182888


namespace exists_a_perfect_power_l182_182802

def is_perfect_power (n : ℕ) : Prop :=
  ∃ b k : ℕ, b > 0 ∧ k ≥ 2 ∧ n = b^k

theorem exists_a_perfect_power :
  ∃ a > 0, ∀ n, 2015 ≤ n ∧ n ≤ 2558 → is_perfect_power (n * a) :=
sorry

end exists_a_perfect_power_l182_182802


namespace sum_x1_x2_range_l182_182538

variable {x₁ x₂ : ℝ}

-- Definition of x₁ being the real root of the equation x * 2^x = 1
def is_root_1 (x : ℝ) : Prop :=
  x * 2^x = 1

-- Definition of x₂ being the real root of the equation x * log_2 x = 1
def is_root_2 (x : ℝ) : Prop :=
  x * Real.log x / Real.log 2 = 1

theorem sum_x1_x2_range (hx₁ : is_root_1 x₁) (hx₂ : is_root_2 x₂) :
  2 < x₁ + x₂ :=
sorry

end sum_x1_x2_range_l182_182538


namespace sum_prob_less_one_l182_182335

theorem sum_prob_less_one (x y z : ℝ) (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1) (hz : 0 < z ∧ z < 1) :
  x * (1 - y) * (1 - z) + (1 - x) * y * (1 - z) + (1 - x) * (1 - y) * z < 1 :=
by
  sorry

end sum_prob_less_one_l182_182335


namespace find_a_l182_182508

theorem find_a (a : ℝ) (x y : ℝ) :
  (x^2 - 4*x + y^2 = 0) →
  ((x - a)^2 + y^2 = 4*((x - 1)^2 + y^2)) →
  a = -2 :=
by
  intros h_circle h_distance
  sorry

end find_a_l182_182508


namespace blue_ball_higher_probability_l182_182442

noncomputable def probability_blue_ball_higher : ℝ :=
  let p (k : ℕ) : ℝ := 1 / (2^k : ℝ)
  let same_bin_prob := ∑' k : ℕ, (p (k + 1))^2
  let higher_prob := (1 - same_bin_prob) / 2
  higher_prob

theorem blue_ball_higher_probability :
  probability_blue_ball_higher = 1 / 3 :=
by
  sorry

end blue_ball_higher_probability_l182_182442


namespace sector_central_angle_l182_182996

-- Defining the problem as a theorem in Lean 4
theorem sector_central_angle (r θ : ℝ) (h1 : 2 * r + r * θ = 4) (h2 : (1 / 2) * r^2 * θ = 1) : θ = 2 :=
by
  sorry

end sector_central_angle_l182_182996


namespace james_total_toys_l182_182124

-- Definition for the number of toy cars
def numToyCars : ℕ := 20

-- Definition for the number of toy soldiers
def numToySoldiers : ℕ := 2 * numToyCars

-- The total number of toys is the sum of toy cars and toy soldiers
def totalToys : ℕ := numToyCars + numToySoldiers

-- Statement to prove: James buys a total of 60 toys
theorem james_total_toys : totalToys = 60 := by
  -- Insert proof here
  sorry

end james_total_toys_l182_182124


namespace find_a_l182_182085

noncomputable def log_a (a x : ℝ) := Real.log x / Real.log a

theorem find_a (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1)
  (h3 : (min (log_a a 2) (log_a a 4)) * (max (log_a a 2) (log_a a 4)) = 2) : 
  a = (1 / 2) ∨ a = 2 :=
sorry

end find_a_l182_182085


namespace union_A_B_inter_A_B_comp_int_B_l182_182097

open Set

variable (x : ℝ)

def A := {x : ℝ | 2 ≤ x ∧ x < 4}
def B := {x : ℝ | 3 ≤ x}

theorem union_A_B : A ∪ B = (Ici 2) :=
by
  sorry

theorem inter_A_B : A ∩ B = Ico 3 4 :=
by
  sorry

theorem comp_int_B : (univ \ A) ∩ B = Ici 4 :=
by
  sorry

end union_A_B_inter_A_B_comp_int_B_l182_182097


namespace find_k_l182_182742

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 + k^3 * x

theorem find_k (k : ℝ) (h : deriv (f k) 0 = 27) : k = 3 :=
by
  sorry

end find_k_l182_182742


namespace camel_cost_l182_182714

variables {C H O E G Z : ℕ} 

-- conditions
axiom h1 : 10 * C = 24 * H
axiom h2 : 16 * H = 4 * O
axiom h3 : 6 * O = 4 * E
axiom h4 : 3 * E = 15 * G
axiom h5 : 8 * G = 20 * Z
axiom h6 : 12 * E = 180000

-- goal
theorem camel_cost : C = 6000 :=
by sorry

end camel_cost_l182_182714


namespace total_profit_l182_182319

theorem total_profit (A B C : ℕ) (A_invest B_invest C_invest A_share : ℕ) (total_invest total_profit : ℕ)
  (h1 : A_invest = 6300)
  (h2 : B_invest = 4200)
  (h3 : C_invest = 10500)
  (h4 : A_share = 3630)
  (h5 : total_invest = A_invest + B_invest + C_invest)
  (h6 : total_profit * A_share = A_invest * total_invest) :
  total_profit = 12100 :=
by
  sorry

end total_profit_l182_182319


namespace boxes_used_l182_182373

-- Define the given conditions
def oranges_per_box : ℕ := 10
def total_oranges : ℕ := 2650

-- Define the proof statement
theorem boxes_used : total_oranges / oranges_per_box = 265 :=
by
  -- Proof goes here
  sorry

end boxes_used_l182_182373


namespace min_xy_min_x_add_y_l182_182289

open Real

theorem min_xy (x y : ℝ) (h1 : log x + log y = log (x + y + 3)) (hx : x > 0) (hy : y > 0) : xy ≥ 9 := sorry

theorem min_x_add_y (x y : ℝ) (h1 : log x + log y = log (x + y + 3)) (hx : x > 0) (hy : y > 0) : x + y ≥ 6 := sorry

end min_xy_min_x_add_y_l182_182289


namespace cos_half_angle_quadrant_l182_182901

theorem cos_half_angle_quadrant 
  (α : ℝ) 
  (h1 : 25 * Real.sin α ^ 2 + Real.sin α - 24 = 0) 
  (h2 : π / 2 < α ∧ α < π) 
  : Real.cos (α / 2) = 3 / 5 ∨ Real.cos (α / 2) = -3 / 5 :=
by
  sorry

end cos_half_angle_quadrant_l182_182901


namespace three_consecutive_multiples_sum_l182_182797

theorem three_consecutive_multiples_sum (h1 : Int) (h2 : h1 % 3 = 0) (h3 : Int) (h4 : h3 = h1 - 3) (h5 : Int) (h6 : h5 = h1 - 6) (h7: h1 = 27) : h1 + h3 + h5 = 72 := 
by 
  -- let numbers be n, n-3, n-6 and n = 27
  -- so n + n-3 + n-6 = 27 + 24 + 21 = 72
  sorry

end three_consecutive_multiples_sum_l182_182797


namespace quadratic_no_rational_solution_l182_182622

theorem quadratic_no_rational_solution 
  (a b c : ℤ) 
  (ha : a % 2 = 1) 
  (hb : b % 2 = 1) 
  (hc : c % 2 = 1) :
  ∀ (x : ℚ), ¬ (a * x^2 + b * x + c = 0) :=
by
  sorry

end quadratic_no_rational_solution_l182_182622


namespace number_of_white_balls_l182_182379

theorem number_of_white_balls (total_balls yellow_frequency : ℕ) (h1 : total_balls = 10) (h2 : yellow_frequency = 60) :
  (total_balls - (total_balls * yellow_frequency / 100) = 4) :=
by
  sorry

end number_of_white_balls_l182_182379


namespace probability_two_red_books_l182_182036

theorem probability_two_red_books (total_books red_books blue_books selected_books : ℕ)
  (h_total: total_books = 8)
  (h_red: red_books = 4)
  (h_blue: blue_books = 4)
  (h_selected: selected_books = 2) :
  (Nat.choose red_books selected_books : ℚ) / (Nat.choose total_books selected_books) = 3 / 14 := by
  sorry

end probability_two_red_books_l182_182036


namespace tangential_quadrilateral_difference_l182_182529

-- Definitions of the conditions given in the problem
def is_cyclic_quadrilateral (a b c d : ℝ) : Prop := sorry -- In real setting, it means the quadrilateral vertices lie on a circle
def is_tangential_quadrilateral (a b c d : ℝ) : Prop := sorry -- In real setting, it means the sides are tangent to a common incircle
def point_tangency (a b c : ℝ) : Prop := sorry

-- Main theorem
theorem tangential_quadrilateral_difference (AB BC CD DA : ℝ) (x y : ℝ) 
  (h1 : is_cyclic_quadrilateral AB BC CD DA)
  (h2 : is_tangential_quadrilateral AB BC CD DA)
  (h3 : AB = 80) (h4 : BC = 140) (h5 : CD = 120) (h6 : DA = 100)
  (h7 : point_tangency x y CD)
  (h8 : x + y = 120) :
  |x - y| = 80 := 
sorry

end tangential_quadrilateral_difference_l182_182529


namespace sum_coefficients_equals_l182_182062

theorem sum_coefficients_equals :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ), 
  (∀ x : ℤ, (2 * x + 1) ^ 5 = 
    a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) →
  a_0 = 1 →
  a_1 + a_2 + a_3 + a_4 + a_5 = 3^5 - 1 :=
by
  intros a_0 a_1 a_2 a_3 a_4 a_5 h h0
  sorry

end sum_coefficients_equals_l182_182062


namespace both_miss_probability_l182_182523

-- Define the probabilities of hitting the target for Persons A and B 
def prob_hit_A : ℝ := 0.85
def prob_hit_B : ℝ := 0.8

-- Calculate the probabilities of missing the target
def prob_miss_A : ℝ := 1 - prob_hit_A
def prob_miss_B : ℝ := 1 - prob_hit_B

-- Prove that the probability of both missing the target is 0.03
theorem both_miss_probability : prob_miss_A * prob_miss_B = 0.03 :=
by
  sorry

end both_miss_probability_l182_182523


namespace max_integer_solutions_l182_182663

noncomputable def semi_centered (p : ℕ → ℤ) :=
  ∃ k : ℕ, p k = k + 50 - 50 * 50

theorem max_integer_solutions (p : ℕ → ℤ) (h1 : semi_centered p) (h2 : ∀ x : ℕ, ∃ c : ℤ, p x = c * x^2) (h3 : p 50 = 50) :
  ∃ n ≤ 6, ∀ k : ℕ, (p k = k^2) → k ∈ Finset.range (n+1) :=
sorry

end max_integer_solutions_l182_182663


namespace solve_quadratic_equation_l182_182037

theorem solve_quadratic_equation (x : ℝ) (h : x^2 = x) : x = 0 ∨ x = 1 :=
sorry

end solve_quadratic_equation_l182_182037


namespace chocolates_bought_l182_182782

theorem chocolates_bought (C S : ℝ) (h1 : N * C = 45 * S) (h2 : 80 = ((S - C) / C) * 100) : 
  N = 81 :=
by
  sorry

end chocolates_bought_l182_182782


namespace base_of_hill_depth_l182_182757

theorem base_of_hill_depth : 
  ∀ (H : ℕ), 
  (H = 900) → 
  (1 / 4 * H = 225) :=
by
  intros H h
  sorry

end base_of_hill_depth_l182_182757


namespace determine_condition_l182_182254

theorem determine_condition (a b c : ℕ) (ha : 0 < a ∧ a < 12) (hb : 0 < b ∧ b < 12) (hc : 0 < c ∧ c < 12) 
    (h_eq : (12 * a + b) * (12 * a + c) = 144 * a * (a + 1) + b * c) : 
    b + c = 12 :=
by
  sorry

end determine_condition_l182_182254


namespace tangent_line_intersection_l182_182187

theorem tangent_line_intersection
    (circle1_center : ℝ × ℝ)
    (circle1_radius : ℝ)
    (circle2_center : ℝ × ℝ)
    (circle2_radius : ℝ)
    (tangent_intersection_x : ℝ)
    (h1 : circle1_center = (0, 0))
    (h2 : circle1_radius = 3)
    (h3 : circle2_center = (12, 0))
    (h4 : circle2_radius = 5)
    (h5 : tangent_intersection_x > 0) :
    tangent_intersection_x = 9 / 2 := by
  sorry

end tangent_line_intersection_l182_182187


namespace relationship_among_a_b_c_l182_182962

noncomputable def f (x : ℝ) : ℝ := sorry  -- The actual function definition is not necessary for this statement.

-- Lean statements for the given conditions
variables {f : ℝ → ℝ}

-- f is even
def even_function (f : ℝ → ℝ) := ∀ x, f (-x) = f x

-- f(x+1) = -f(x)
def periodic_property (f : ℝ → ℝ) := ∀ x, f (x + 1) = - f x

-- f is monotonically increasing on [-1, 0]
def monotonically_increasing_on (f : ℝ → ℝ) := ∀ x y, -1 ≤ x ∧ x ≤ y ∧ y ≤ 0 → f x ≤ f y

-- Define the relationship statement
theorem relationship_among_a_b_c (h1 : even_function f) (h2 : periodic_property f) 
  (h3 : monotonically_increasing_on f) :
  f 3 < f (Real.sqrt 2) ∧ f (Real.sqrt 2) < f 2 :=
sorry

end relationship_among_a_b_c_l182_182962


namespace remainder_492381_div_6_l182_182016

theorem remainder_492381_div_6 : 492381 % 6 = 3 := 
by
  sorry

end remainder_492381_div_6_l182_182016


namespace part_a_l182_182438

def system_of_equations (x y z a : ℝ) := 
  (x - a * y = y * z) ∧ (y - a * z = z * x) ∧ (z - a * x = x * y)

theorem part_a (x y z : ℝ) : 
  system_of_equations x y z 0 ↔ (x = 0 ∧ y = 0 ∧ z = 0) 
  ∨ (∃ x, y = x ∧ z = 1) 
  ∨ (∃ x, y = -x ∧ z = -1) := 
  sorry

end part_a_l182_182438


namespace sum_first_six_terms_arithmetic_seq_l182_182175

theorem sum_first_six_terms_arithmetic_seq :
  ∃ a_1 d : ℤ, (a_1 + 3 * d = 7) ∧ (a_1 + 4 * d = 12) ∧ (a_1 + 5 * d = 17) ∧ 
  (6 * (2 * a_1 + 5 * d) / 2 = 27) :=
by
  sorry

end sum_first_six_terms_arithmetic_seq_l182_182175


namespace part1_part2_l182_182047

def f (x : ℝ) (t : ℝ) : ℝ := x^2 + 2 * t * x + t - 1

theorem part1 (hf : ∀ x ∈ Set.Icc (-(3 : ℝ)) (1 : ℝ), f x 2 ≤ 6 ∧ f x 2 ≥ -3) : 
  ∀ x ∈ Set.Icc (-(3 : ℝ)) (1 : ℝ), f x 2 ≤ 6 ∧ f x 2 ≥ -3 :=
by 
  sorry
  
theorem part2 (ht : ∀ x ∈ Set.Icc (1 : ℝ) (2 : ℝ), f x t > 0) : 
  t ∈ Set.Ioi (0 : ℝ) :=
by 
  sorry

end part1_part2_l182_182047


namespace range_of_a_l182_182743

def is_in_third_quadrant (A : ℝ × ℝ) : Prop :=
  A.1 < 0 ∧ A.2 < 0

theorem range_of_a (a : ℝ) (h : is_in_third_quadrant (a, a - 1)) : a < 0 :=
by
  sorry

end range_of_a_l182_182743


namespace find_radius_of_tangent_circle_l182_182864

def tangent_circle_radius : Prop :=
  ∃ (r : ℝ), 
    (r > 0) ∧ 
    (∀ (θ : ℝ),
      (∃ (x y : ℝ),
        x = 1 + r * Real.cos θ ∧ 
        y = 1 + r * Real.sin θ ∧ 
        x + y - 1 = 0))
    → r = (Real.sqrt 2) / 2

theorem find_radius_of_tangent_circle : tangent_circle_radius :=
sorry

end find_radius_of_tangent_circle_l182_182864


namespace friends_courses_l182_182737

-- Define the notions of students and their properties
structure Student :=
  (first_name : String)
  (last_name : String)
  (year : ℕ)

-- Define the specific conditions from the problem
def students : List Student := [
  ⟨"Peter", "Krylov", 1⟩,
  ⟨"Nikolay", "Ivanov", 2⟩,
  ⟨"Boris", "Karpov", 3⟩,
  ⟨"Vasily", "Orlov", 4⟩
]

-- The main statement of the problem
theorem friends_courses :
  ∀ (s : Student), s ∈ students →
    (s.first_name = "Peter" → s.last_name = "Krylov" ∧ s.year = 1) ∧
    (s.first_name = "Nikolay" → s.last_name = "Ivanov" ∧ s.year = 2) ∧
    (s.first_name = "Boris" → s.last_name = "Karpov" ∧ s.year = 3) ∧
    (s.first_name = "Vasily" → s.last_name = "Orlov" ∧ s.year = 4) :=
by
  sorry

end friends_courses_l182_182737


namespace expand_expression_l182_182641

theorem expand_expression : ∀ (x : ℝ), (1 + x^3) * (1 - x^4 + x^5) = 1 + x^3 - x^4 + x^5 - x^7 + x^8 :=
by
  intro x
  sorry

end expand_expression_l182_182641


namespace max_dot_product_on_circle_l182_182311

theorem max_dot_product_on_circle :
  ∀ (P : ℝ × ℝ) (O : ℝ × ℝ) (A : ℝ × ℝ),
  O = (0, 0) →
  A = (-2, 0) →
  P.1 ^ 2 + P.2 ^ 2 = 1 →
  let AO := (2, 0)
  let AP := (P.1 + 2, P.2)
  ∃ α : ℝ, P = (Real.cos α, Real.sin α) ∧ 
  ∃ max_val : ℝ, max_val = 6 ∧ 
  (2 * (Real.cos α + 2) = max_val) :=
by
  intro P O A hO hA hP 
  let AO := (2, 0)
  let AP := (P.1 + 2, P.2)
  sorry

end max_dot_product_on_circle_l182_182311


namespace least_k_for_sum_divisible_l182_182652

theorem least_k_for_sum_divisible (n : ℕ) (hn : n > 0) : 
  (∃ k : ℕ, (∀ (xs : List ℕ), (xs.length = k) → (∃ ys : List ℕ, (ys.length % 2 = 0) ∧ (ys.sum % n = 0))) ∧ 
    (k = if n % 2 = 1 then 2 * n else n + 1)) :=
sorry

end least_k_for_sum_divisible_l182_182652


namespace total_number_of_animals_is_304_l182_182075

theorem total_number_of_animals_is_304
    (dogs frogs : ℕ) 
    (h1 : frogs = 160) 
    (h2 : frogs = 2 * dogs) 
    (cats : ℕ) 
    (h3 : cats = dogs - (dogs / 5)) :
  cats + dogs + frogs = 304 :=
by
  sorry

end total_number_of_animals_is_304_l182_182075


namespace math_problem_l182_182537

theorem math_problem 
  (a : ℤ) 
  (h_a : a = -1) 
  (b : ℚ) 
  (h_b : b = 0) 
  (c : ℕ) 
  (h_c : c = 1)
  : a^2024 + 2023 * b - c^2023 = 0 := by
  sorry

end math_problem_l182_182537


namespace value_of_a_l182_182680

def star (a b : ℝ) : ℝ := 3 * a - 2 * b ^ 2

theorem value_of_a (a : ℝ) (h : star a 3 = 15) : a = 11 := 
by
  sorry

end value_of_a_l182_182680


namespace initial_workers_l182_182670

theorem initial_workers (W : ℕ) (work1 : ℕ) (work2 : ℕ) :
  (work1 = W * 8 * 30) →
  (work2 = (W + 35) * 6 * 40) →
  (work1 / 30 = work2 / 40) →
  W = 105 :=
by
  intros hwork1 hwork2 hprop
  sorry

end initial_workers_l182_182670


namespace maxwell_walking_speed_l182_182693

open Real

theorem maxwell_walking_speed (v : ℝ) : 
  (∀ (v : ℝ), (4 * v + 6 * 3 = 34)) → v = 4 :=
by
  intros
  have h1 : 4 * v + 18 = 34 := by sorry
  have h2 : 4 * v = 16 := by sorry
  have h3 : v = 4 := by sorry
  exact h3

end maxwell_walking_speed_l182_182693


namespace not_p_and_not_p_and_q_implies_not_p_or_q_l182_182619

theorem not_p_and_not_p_and_q_implies_not_p_or_q (p q : Prop) (h1 : ¬p) (h2 : ¬(p ∧ q)) : ¬(p ∨ q) :=
sorry

end not_p_and_not_p_and_q_implies_not_p_or_q_l182_182619


namespace ratio_of_areas_l182_182768

noncomputable def circumferences_equal_arcs (C1 C2 : ℝ) (k1 k2 : ℕ) : Prop :=
  (k1 : ℝ) / 360 * C1 = (k2 : ℝ) / 360 * C2

theorem ratio_of_areas (C1 C2 : ℝ) (h : circumferences_equal_arcs C1 C2 60 30) :
  (π * (C1 / (2 * π))^2) / (π * (C2 / (2 * π))^2) = 1 / 4 :=
by
  sorry

end ratio_of_areas_l182_182768


namespace Namjoon_walk_extra_l182_182053

-- Define the usual distance Namjoon walks to school
def usual_distance := 1.2

-- Define the distance Namjoon walked to the intermediate point
def intermediate_distance := 0.3

-- Define the total distance Namjoon walked today
def total_distance_today := (intermediate_distance * 2) + usual_distance

-- Define the extra distance walked today compared to usual
def extra_distance := total_distance_today - usual_distance

-- State the theorem to prove that the extra distance walked today is 0.6 km
theorem Namjoon_walk_extra : extra_distance = 0.6 := 
by
  sorry

end Namjoon_walk_extra_l182_182053


namespace smallest_model_length_l182_182152

theorem smallest_model_length (full_size : ℕ) (mid_size_factor smallest_size_factor : ℚ) :
  full_size = 240 →
  mid_size_factor = 1 / 10 →
  smallest_size_factor = 1 / 2 →
  (full_size * mid_size_factor) * smallest_size_factor = 12 :=
by
  intros h_full_size h_mid_size_factor h_smallest_size_factor
  sorry

end smallest_model_length_l182_182152


namespace problem1_problem2_l182_182636

-- Problem 1
theorem problem1 : 3^2 * (-1 + 3) - (-16) / 8 = 20 :=
by decide  -- automatically prove simple arithmetic

-- Problem 2
variables {x : ℝ} (hx1 : x ≠ 1) (hx2 : x ≠ -1)

theorem problem2 : ((x^2 / (x + 1)) - (1 / (x + 1))) * (x + 1) / (x - 1) = x + 1 :=
by sorry  -- proof to be completed

end problem1_problem2_l182_182636


namespace number_of_students_l182_182122

theorem number_of_students (x : ℕ) (total_cards : ℕ) (h : x * (x - 1) = total_cards) (h_total : total_cards = 182) : x = 14 :=
by
  sorry

end number_of_students_l182_182122


namespace eight_digit_number_divisible_by_101_l182_182985

def repeat_twice (x : ℕ) : ℕ := 100 * x + x

theorem eight_digit_number_divisible_by_101 (ef gh ij kl : ℕ) 
  (hef : ef < 100) (hgh : gh < 100) (hij : ij < 100) (hkl : kl < 100) :
  (100010001 * repeat_twice ef + 1000010 * repeat_twice gh + 10010 * repeat_twice ij + 10 * repeat_twice kl) % 101 = 0 := sorry

end eight_digit_number_divisible_by_101_l182_182985


namespace egg_hunt_ratio_l182_182346

theorem egg_hunt_ratio :
  ∃ T : ℕ, (3 * T + 30 = 400 ∧ T = 123) ∧ (60 : ℚ) / (T - 20 : ℚ) = 60 / 103 :=
by
  sorry

end egg_hunt_ratio_l182_182346


namespace intersection_point_l182_182228

noncomputable def line1 (x : ℝ) : ℝ := 3 * x + 10

noncomputable def slope_perp : ℝ := -1/3

noncomputable def line_perp (x : ℝ) : ℝ := slope_perp * x + (2 - slope_perp * 3)

theorem intersection_point : 
  ∃ (x y : ℝ), y = line1 x ∧ y = line_perp x ∧ x = -21 / 10 ∧ y = 37 / 10 :=
by
  sorry

end intersection_point_l182_182228


namespace trig_identity_proof_l182_182478

variable (α : ℝ)

theorem trig_identity_proof : 
  16 * (Real.sin α)^5 - 20 * (Real.sin α)^3 + 5 * Real.sin α = Real.sin (5 * α) :=
  sorry

end trig_identity_proof_l182_182478


namespace perpendicular_line_through_point_l182_182186

theorem perpendicular_line_through_point (m t : ℝ) (h : 2 * m^2 + m + t = 0) :
  m = 1 → t = -3 → (∀ x y : ℝ, m^2 * x + m * y + t = 0 ↔ x + y - 3 = 0) :=
by
  intros hm ht
  subst hm
  subst ht
  sorry

end perpendicular_line_through_point_l182_182186


namespace relationship_between_A_B_C_l182_182729

-- Definitions based on the problem conditions
def A : Set ℝ := {θ | ∃ k : ℤ, 2 * k * Real.pi < θ ∧ θ < 2 * k * Real.pi + Real.pi / 2}
def B : Set ℝ := {θ | 0 < θ ∧ θ < Real.pi / 2}
def C : Set ℝ := {θ | θ < Real.pi / 2}

-- Proof statement: Prove the specified relationship
theorem relationship_between_A_B_C : B ∪ C = C := by
  sorry

end relationship_between_A_B_C_l182_182729


namespace house_cost_ratio_l182_182309

theorem house_cost_ratio {base_salary commission house_A_cost total_income : ℕ}
    (H_base_salary: base_salary = 3000)
    (H_commission: commission = 2)
    (H_house_A_cost: house_A_cost = 60000)
    (H_total_income: total_income = 8000)
    (H_total_sales_price: ℕ)
    (H_house_B_cost: ℕ)
    (H_house_C_cost: ℕ)
    (H_m: ℕ)
    (h1: total_income - base_salary = 5000)
    (h2: total_sales_price * commission / 100 = 5000)
    (h3: total_sales_price = 250000)
    (h4: house_B_cost = 3 * house_A_cost)
    (h5: total_sales_price = house_A_cost + house_B_cost + house_C_cost)
    (h6: house_C_cost = m * house_A_cost - 110000)
  : m = 2 :=
by
  sorry

end house_cost_ratio_l182_182309


namespace doughnut_machine_completion_l182_182430

noncomputable def completion_time (start_time : ℕ) (partial_duration : ℕ) : ℕ :=
  start_time + 4 * partial_duration

theorem doughnut_machine_completion :
  let start_time := 8 * 60  -- 8:00 AM in minutes
  let partial_completion_time := 11 * 60 + 40  -- 11:40 AM in minutes
  let one_fourth_duration := partial_completion_time - start_time
  completion_time start_time one_fourth_duration = (22 * 60 + 40) := -- 10:40 PM in minutes
by
  sorry

end doughnut_machine_completion_l182_182430


namespace cost_price_of_table_l182_182582

theorem cost_price_of_table (CP SP : ℝ) (h1 : SP = 1.20 * CP) (h2 : SP = 3000) : CP = 2500 := by
    sorry

end cost_price_of_table_l182_182582


namespace largest_common_divisor_414_345_l182_182666

theorem largest_common_divisor_414_345 : ∃ d, d ∣ 414 ∧ d ∣ 345 ∧ 
                                      (∀ e, e ∣ 414 ∧ e ∣ 345 → e ≤ d) ∧ d = 69 :=
by 
  sorry

end largest_common_divisor_414_345_l182_182666


namespace cricket_team_players_l182_182834

-- Define conditions 
def non_throwers (T P : ℕ) : ℕ := P - T
def left_handers (N : ℕ) : ℕ := N / 3
def right_handers_non_thrower (N : ℕ) : ℕ := 2 * N / 3
def total_right_handers (T R : ℕ) : Prop := R = T + right_handers_non_thrower (non_throwers T R)

-- Assume conditions are given
variables (P N R T : ℕ)
axiom hT : T = 37
axiom hR : R = 49
axiom hNonThrower : N = non_throwers T P
axiom hRightHanders : right_handers_non_thrower N = R - T

-- Prove the total number of players is 55
theorem cricket_team_players : P = 55 :=
by
  sorry

end cricket_team_players_l182_182834


namespace remainder_m_squared_plus_4m_plus_6_l182_182982

theorem remainder_m_squared_plus_4m_plus_6 (m : ℤ) (k : ℤ) (hk : m = 100 * k - 2) :
  (m ^ 2 + 4 * m + 6) % 100 = 2 := 
sorry

end remainder_m_squared_plus_4m_plus_6_l182_182982


namespace div_coeff_roots_l182_182497

theorem div_coeff_roots :
  ∀ (a b c d e : ℝ), (∀ x, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 → x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4)
  → (d / e = -25 / 12) :=
by
  intros a b c d e h
  sorry

end div_coeff_roots_l182_182497


namespace Alden_nephews_10_years_ago_l182_182229

noncomputable def nephews_Alden_now : ℕ := sorry
noncomputable def nephews_Alden_10_years_ago (N : ℕ) : ℕ := N / 2
noncomputable def nephews_Vihaan_now (N : ℕ) : ℕ := N + 60
noncomputable def total_nephews (N : ℕ) : ℕ := N + (nephews_Vihaan_now N)

theorem Alden_nephews_10_years_ago (N : ℕ) (h1 : total_nephews N = 260) : 
  nephews_Alden_10_years_ago N = 50 :=
by
  sorry

end Alden_nephews_10_years_ago_l182_182229


namespace total_revenue_l182_182026

def price_federal := 50
def price_state := 30
def price_quarterly := 80
def num_federal := 60
def num_state := 20
def num_quarterly := 10

theorem total_revenue : (num_federal * price_federal + num_state * price_state + num_quarterly * price_quarterly) = 4400 := by
  sorry

end total_revenue_l182_182026


namespace correct_system_l182_182675

def system_of_equations (x y : ℤ) : Prop :=
  (5 * x + 45 = y) ∧ (7 * x - 3 = y)

theorem correct_system : ∃ x y : ℤ, system_of_equations x y :=
sorry

end correct_system_l182_182675


namespace closed_polygon_inequality_l182_182115

noncomputable def length_eq (A B C D : ℝ × ℝ × ℝ) (l : ℝ) : Prop :=
  dist A B = l ∧ dist B C = l ∧ dist C D = l ∧ dist D A = l

theorem closed_polygon_inequality 
  (A B C D P : ℝ × ℝ × ℝ) (l : ℝ)
  (hABCD : length_eq A B C D l) :
  dist P A < dist P B + dist P C + dist P D :=
sorry

end closed_polygon_inequality_l182_182115


namespace intersection_A_B_l182_182848

-- Definitions based on conditions
variable (U : Set Int) (A B : Set Int)

#check Set

-- Given conditions
def U_def : Set Int := {-1, 3, 5, 7, 9}
def compl_U_A : Set Int := {-1, 9}
def B_def : Set Int := {3, 7, 9}

-- A is defined as the set difference of U and the complement of A in U
def A_def : Set Int := { x | x ∈ U_def ∧ ¬ (x ∈ compl_U_A) }

-- Theorem stating the intersection of A and B equals {3, 7}
theorem intersection_A_B : A_def ∩ B_def = {3, 7} :=
by
  -- Here would be the proof block, but we add 'sorry' to indicate it is unfinished.
  sorry

end intersection_A_B_l182_182848


namespace total_fish_purchased_l182_182095

/-- Definition of the conditions based on Roden's visits to the pet shop. -/
def first_visit_goldfish := 15
def first_visit_bluefish := 7
def second_visit_goldfish := 10
def second_visit_bluefish := 12
def second_visit_greenfish := 5
def third_visit_goldfish := 3
def third_visit_bluefish := 7
def third_visit_greenfish := 9

/-- Proof statement in Lean 4. -/
theorem total_fish_purchased :
  first_visit_goldfish + first_visit_bluefish +
  second_visit_goldfish + second_visit_bluefish + second_visit_greenfish +
  third_visit_goldfish + third_visit_bluefish + third_visit_greenfish = 68 :=
by
  sorry

end total_fish_purchased_l182_182095


namespace weight_of_steel_rod_l182_182000

theorem weight_of_steel_rod (length1 : ℝ) (weight1 : ℝ) (length2 : ℝ) (weight2 : ℝ) 
  (h1 : length1 = 9) (h2 : weight1 = 34.2) (h3 : length2 = 11.25) : 
  weight2 = (weight1 / length1) * length2 :=
by
  rw [h1, h2, h3]
  simp
  norm_num
  sorry

end weight_of_steel_rod_l182_182000


namespace scientific_notation_l182_182103

def given_number : ℝ := 632000

theorem scientific_notation : given_number = 6.32 * 10^5 :=
by sorry

end scientific_notation_l182_182103


namespace volleyball_problem_correct_l182_182427

noncomputable def volleyball_problem : Nat :=
  let total_players := 16
  let triplets : Finset String := {"Alicia", "Amanda", "Anna"}
  let twins : Finset String := {"Beth", "Brenda"}
  let remaining_players := total_players - triplets.card - twins.card
  let no_triplets_no_twins := Nat.choose remaining_players 6
  let one_triplet_no_twins := triplets.card * Nat.choose remaining_players 5
  let no_triplets_one_twin := twins.card * Nat.choose remaining_players 5
  no_triplets_no_twins + one_triplet_no_twins + no_triplets_one_twin

theorem volleyball_problem_correct : volleyball_problem = 2772 := by
  sorry

end volleyball_problem_correct_l182_182427


namespace responses_needed_l182_182807

noncomputable def Q : ℝ := 461.54
noncomputable def percentage : ℝ := 0.65
noncomputable def required_responses : ℝ := percentage * Q

theorem responses_needed : required_responses = 300 := by
  sorry

end responses_needed_l182_182807


namespace probability_third_attempt_success_l182_182907

noncomputable def P_xi_eq_3 : ℚ :=
  (4 / 5) * (3 / 4) * (1 / 3)

theorem probability_third_attempt_success :
  P_xi_eq_3 = 1 / 5 := by
  sorry

end probability_third_attempt_success_l182_182907


namespace min_value_of_squares_l182_182294

theorem min_value_of_squares (a b s t : ℝ) (h1 : a + b = t) (h2 : a - b = s) :
  a^2 + b^2 = (t^2 + s^2) / 2 :=
sorry

end min_value_of_squares_l182_182294


namespace am_gm_inequality_l182_182490

theorem am_gm_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) : (1 + x) * (1 + y) * (1 + z) ≥ 8 :=
sorry

end am_gm_inequality_l182_182490


namespace uncovered_area_l182_182525

theorem uncovered_area {s₁ s₂ : ℝ} (hs₁ : s₁ = 10) (hs₂ : s₂ = 4) : 
  (s₁^2 - 2 * s₂^2) = 68 := by
  sorry

end uncovered_area_l182_182525


namespace smallest_square_factor_2016_l182_182388

theorem smallest_square_factor_2016 : ∃ n : ℕ, (168 = n) ∧ (∃ k : ℕ, k^2 = n) ∧ (2016 ∣ k^2) :=
by
  sorry

end smallest_square_factor_2016_l182_182388


namespace windows_per_floor_is_3_l182_182745

-- Given conditions
variables (W : ℕ)
def windows_each_floor (W : ℕ) : Prop :=
  (3 * 2 * W) - 2 = 16

-- Correct answer
theorem windows_per_floor_is_3 : windows_each_floor 3 :=
by 
  sorry

end windows_per_floor_is_3_l182_182745


namespace intersection_eq_interval_l182_182091

def M : Set ℝ := {x | x > 1}
def N : Set ℝ := {x | x < 5}

theorem intersection_eq_interval : M ∩ N = {x | 1 < x ∧ x < 5} :=
sorry

end intersection_eq_interval_l182_182091


namespace fraction_of_tank_used_l182_182439

theorem fraction_of_tank_used (speed : ℝ) (fuel_efficiency : ℝ) (initial_fuel : ℝ) (time_traveled : ℝ)
  (h_speed : speed = 40) (h_fuel_eff : fuel_efficiency = 1 / 40) (h_initial_fuel : initial_fuel = 12) 
  (h_time : time_traveled = 5) : 
  (speed * time_traveled * fuel_efficiency) / initial_fuel = 5 / 12 :=
by
  -- Here the proof would go, but we add sorry to indicate it's incomplete.
  sorry

end fraction_of_tank_used_l182_182439


namespace cube_surface_area_l182_182667

theorem cube_surface_area (a : ℕ) (h : a = 2) : 6 * a^2 = 24 := 
by
  sorry

end cube_surface_area_l182_182667


namespace rectangle_area_l182_182507

theorem rectangle_area (x : ℝ) (h1 : (x^2 + (3*x)^2) = (15*Real.sqrt 2)^2) :
  (x * (3 * x)) = 135 := 
by
  sorry

end rectangle_area_l182_182507


namespace scientific_notation_suzhou_blood_donors_l182_182806

theorem scientific_notation_suzhou_blood_donors : ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 124000 = a * 10^n ∧ a = 1.24 ∧ n = 5 :=
by
  sorry

end scientific_notation_suzhou_blood_donors_l182_182806


namespace log_x3y2_value_l182_182579

open Real

noncomputable def log_identity (x y : ℝ) : Prop :=
  log (x * y^4) = 1 ∧ log (x^3 * y) = 1

theorem log_x3y2_value (x y : ℝ) (h : log_identity x y) : log (x^3 * y^2) = 13 / 11 :=
  by
  sorry

end log_x3y2_value_l182_182579


namespace sally_earnings_l182_182919

-- Definitions based on the conditions
def seashells_monday : ℕ := 30
def seashells_tuesday : ℕ := seashells_monday / 2
def total_seashells : ℕ := seashells_monday + seashells_tuesday
def price_per_seashell : ℝ := 1.20
def total_money : ℝ := total_seashells * price_per_seashell

-- Lean 4 statement to prove the total amount of money is $54
theorem sally_earnings : total_money = 54 := by
  -- Proof will go here
  sorry

end sally_earnings_l182_182919


namespace sum_due_is_363_l182_182275

/-
Conditions:
1. BD = 78
2. TD = 66
3. The formula: BD = TD + (TD^2 / PV)
This should imply that PV = 363 given the conditions.
-/

theorem sum_due_is_363 (BD TD PV : ℝ) (h1 : BD = 78) (h2 : TD = 66) (h3 : BD = TD + (TD^2 / PV)) : PV = 363 :=
by
  sorry

end sum_due_is_363_l182_182275


namespace range_of_a_exists_x_ax2_ax_1_lt_0_l182_182123

theorem range_of_a_exists_x_ax2_ax_1_lt_0 :
  {a : ℝ | ∃ x : ℝ, a * x^2 + a * x + 1 < 0} = {a : ℝ | a < 0 ∨ a > 4} :=
sorry

end range_of_a_exists_x_ax2_ax_1_lt_0_l182_182123


namespace lucy_initial_balance_l182_182862

theorem lucy_initial_balance (final_balance deposit withdrawal : Int) 
  (h_final : final_balance = 76)
  (h_deposit : deposit = 15)
  (h_withdrawal : withdrawal = 4) :
  let initial_balance := final_balance + withdrawal - deposit
  initial_balance = 65 := 
by
  sorry

end lucy_initial_balance_l182_182862


namespace determine_x_l182_182126

noncomputable def x_candidates := { x : ℝ | x = (3 + Real.sqrt 105) / 24 ∨ x = (3 - Real.sqrt 105) / 24 }

theorem determine_x (x y : ℝ) (h_y : y = 3 * x) 
  (h_eq : 4 * y ^ 2 + 2 * y + 7 = 3 * (8 * x ^ 2 + y + 3)) :
  x ∈ x_candidates :=
by
  sorry

end determine_x_l182_182126


namespace sum_of_fractions_l182_182130

-- Definition of the fractions
def frac1 : ℚ := 3/5
def frac2 : ℚ := 5/11
def frac3 : ℚ := 1/3

-- Main theorem stating that the sum of the fractions equals 229/165
theorem sum_of_fractions : frac1 + frac2 + frac3 = 229 / 165 := sorry

end sum_of_fractions_l182_182130


namespace avogadro_constant_problem_l182_182368

theorem avogadro_constant_problem 
  (N_A : ℝ) -- Avogadro's constant
  (mass1 : ℝ := 18) (molar_mass1 : ℝ := 20) (moles1 : ℝ := mass1 / molar_mass1) 
  (atoms_D2O_molecules : ℝ := 2) (atoms_D2O : ℝ := moles1 * atoms_D2O_molecules * N_A)
  (mass2 : ℝ := 14) (molar_mass_N2CO : ℝ := 28) (moles2 : ℝ := mass2 / molar_mass_N2CO)
  (electrons_per_molecule : ℝ := 14) (total_electrons_mixture : ℝ := moles2 * electrons_per_molecule * N_A)
  (volume3 : ℝ := 2.24) (temp_unk : Prop := true) -- unknown temperature
  (pressure_unk : Prop := true) -- unknown pressure
  (carbonate_molarity : ℝ := 0.1) (volume_solution : ℝ := 1) (moles_carbonate : ℝ := carbonate_molarity * volume_solution) 
  (anions_carbonate_solution : ℝ := moles_carbonate * N_A) :
  (atoms_D2O ≠ 2 * N_A) ∧ (anions_carbonate_solution > 0.1 * N_A) ∧ (total_electrons_mixture = 7 * N_A) -> 
  True := sorry

end avogadro_constant_problem_l182_182368


namespace evaluate_ninth_roots_of_unity_product_l182_182700

theorem evaluate_ninth_roots_of_unity_product : 
  (3 - Complex.exp (2 * Real.pi * Complex.I / 9)) *
  (3 - Complex.exp (4 * Real.pi * Complex.I / 9)) *
  (3 - Complex.exp (6 * Real.pi * Complex.I / 9)) *
  (3 - Complex.exp (8 * Real.pi * Complex.I / 9)) *
  (3 - Complex.exp (10 * Real.pi * Complex.I / 9)) *
  (3 - Complex.exp (12 * Real.pi * Complex.I / 9)) *
  (3 - Complex.exp (14 * Real.pi * Complex.I / 9)) *
  (3 - Complex.exp (16 * Real.pi * Complex.I / 9)) 
  = 9841 := 
by 
  sorry

end evaluate_ninth_roots_of_unity_product_l182_182700


namespace number_of_ways_to_select_starting_lineup_l182_182902

noncomputable def choose (n k : ℕ) : ℕ := 
if h : k ≤ n then Nat.choose n k else 0

theorem number_of_ways_to_select_starting_lineup (n k : ℕ) (h : n = 12) (h1 : k = 5) : 
  12 * choose 11 4 = 3960 := 
by sorry

end number_of_ways_to_select_starting_lineup_l182_182902


namespace smallest_number_of_eggs_l182_182890

theorem smallest_number_of_eggs (c : ℕ) (eggs_total : ℕ) :
  eggs_total = 15 * c - 3 ∧ eggs_total > 150 → eggs_total = 162 :=
by
  sorry

end smallest_number_of_eggs_l182_182890


namespace evaluate_expression_l182_182696

theorem evaluate_expression : 
  2 + (3 / (4 + (5 / (6 + (7 / 8))))) = (137 / 52) :=
by
  -- We need to evaluate from the innermost part to the outermost,
  -- as noted in the problem statement and solution steps.
  sorry

end evaluate_expression_l182_182696


namespace days_at_grandparents_l182_182057

theorem days_at_grandparents
  (total_vacation_days : ℕ)
  (travel_to_gp : ℕ)
  (travel_to_brother : ℕ)
  (days_at_brother : ℕ)
  (travel_to_sister : ℕ)
  (days_at_sister : ℕ)
  (travel_home : ℕ)
  (total_days : total_vacation_days = 21) :
  total_vacation_days - (travel_to_gp + travel_to_brother + days_at_brother + travel_to_sister + days_at_sister + travel_home) = 5 :=
by
  sorry -- proof to be constructed

end days_at_grandparents_l182_182057


namespace volume_of_prism_l182_182376

-- Given conditions
def length : ℕ := 12
def width : ℕ := 8
def depth : ℕ := 8

-- Proving the volume of the rectangular prism
theorem volume_of_prism : length * width * depth = 768 := by
  sorry

end volume_of_prism_l182_182376


namespace sum_of_powers_mod7_l182_182158

theorem sum_of_powers_mod7 (k : ℕ) : (2^k + 3^k) % 7 = 0 ↔ k % 6 = 3 := by
  sorry

end sum_of_powers_mod7_l182_182158


namespace population_sampling_precision_l182_182558

theorem population_sampling_precision (sample_size : ℕ → Prop) 
    (A : Prop) (B : Prop) (C : Prop) (D : Prop)
    (condition_A : A = (∀ n : ℕ, sample_size n → false))
    (condition_B : B = (∀ n : ℕ, sample_size n → n > 0 → true))
    (condition_C : C = (∀ n : ℕ, sample_size n → false))
    (condition_D : D = (∀ n : ℕ, sample_size n → false)) :
  B :=
by sorry

end population_sampling_precision_l182_182558


namespace circle_area_pi_div_2_l182_182024

open Real EuclideanGeometry

variable (x y : ℝ)

def circleEquation : Prop := 3 * x^2 + 3 * y^2 - 15 * x + 9 * y + 27 = 0

theorem circle_area_pi_div_2
  (h : circleEquation x y) : 
  ∃ (r : ℝ), r = sqrt 0.5 ∧ π * r * r = π / 2 :=
by
  sorry

end circle_area_pi_div_2_l182_182024


namespace angle_between_diagonals_l182_182781

variables (α β : ℝ) 

theorem angle_between_diagonals (α β : ℝ) :
  ∃ γ : ℝ, γ = Real.arccos (Real.sin α * Real.sin β) :=
by
  sorry

end angle_between_diagonals_l182_182781


namespace cricket_runs_product_l182_182262

theorem cricket_runs_product :
  let runs_first_10 := [11, 6, 7, 5, 12, 8, 3, 10, 9, 4]
  let total_runs_first_10 := runs_first_10.sum
  let total_runs := total_runs_first_10 + 2 + 7
  2 < 15 ∧ 7 < 15 ∧ (total_runs_first_10 + 2) % 11 = 0 ∧ (total_runs_first_10 + 2 + 7) % 12 = 0 →
  (2 * 7) = 14 :=
by
  intros h
  sorry

end cricket_runs_product_l182_182262


namespace vehicle_worth_l182_182567

-- Definitions from the conditions
def monthlyEarnings : ℕ := 4000
def savingFraction : ℝ := 0.5
def savingMonths : ℕ := 8

-- Theorem statement
theorem vehicle_worth : (monthlyEarnings * savingFraction * savingMonths : ℝ) = 16000 := 
by
  sorry

end vehicle_worth_l182_182567


namespace min_balls_for_color_15_l182_182288

theorem min_balls_for_color_15
  (red green yellow blue white black : ℕ)
  (h_red : red = 28)
  (h_green : green = 20)
  (h_yellow : yellow = 19)
  (h_blue : blue = 13)
  (h_white : white = 11)
  (h_black : black = 9) :
  ∃ n, n = 76 ∧ ∀ balls_drawn, balls_drawn = n →
  ∃ color, 
    (color = "red" ∧ balls_drawn ≥ 15 ∧ balls_drawn <= red) ∨
    (color = "green" ∧ balls_drawn ≥ 15 ∧ balls_drawn <= green) ∨
    (color = "yellow" ∧ balls_drawn ≥ 15 ∧ balls_drawn <= yellow) ∨
    (color = "blue" ∧ balls_drawn ≥ 15 ∧ balls_drawn <= blue) ∨
    (color = "white" ∧ balls_drawn >= 15 ∧ balls_drawn <= white) ∨
    (color = "black" ∧ balls_drawn ≥ 15 ∧ balls_drawn <= black) := 
sorry

end min_balls_for_color_15_l182_182288


namespace tangerines_left_l182_182964

def total_tangerines : ℕ := 27
def tangerines_eaten : ℕ := 18

theorem tangerines_left : total_tangerines - tangerines_eaten = 9 := by
  sorry

end tangerines_left_l182_182964


namespace geometric_seq_a5_a7_l182_182308

theorem geometric_seq_a5_a7 (a : ℕ → ℝ)
  (h1 : a 1 = 1)
  (h2 : ∀ n, a (n + 1) = a n * q)
  (h3 : a 3 + a 5 = 6)
  (q : ℝ) :
  (a 5 + a 7 = 12) :=
sorry

end geometric_seq_a5_a7_l182_182308


namespace cyclic_sum_inequality_l182_182691

theorem cyclic_sum_inequality
  (a b c d e : ℝ)
  (h1 : 0 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≤ e)
  (h6 : a + b + c + d + e = 1) :
  a * d + d * c + c * b + b * e + e * a ≤ 1 / 5 :=
by
  sorry

end cyclic_sum_inequality_l182_182691


namespace stream_speed_l182_182096

theorem stream_speed :
  ∀ (v : ℝ),
  (12 - v) / (12 + v) = 1 / 2 →
  v = 4 :=
by
  sorry

end stream_speed_l182_182096


namespace DE_plus_FG_equals_19_div_6_l182_182668

theorem DE_plus_FG_equals_19_div_6
    (AB AC : ℝ)
    (BC : ℝ)
    (h_isosceles : AB = 2 ∧ AC = 2 ∧ BC = 1.5)
    (D E G F : ℝ)
    (h_parallel_DE_BC : D = E)
    (h_parallel_FG_BC : F = G)
    (h_same_perimeter : 2 + D = 2 + F ∧ 2 + F = 5.5 - F) :
    D + F = 19 / 6 := by
  sorry

end DE_plus_FG_equals_19_div_6_l182_182668


namespace rectangle_area_l182_182679

theorem rectangle_area (b l : ℝ) (h1 : l = 3 * b) (h2 : 2 * (l + b) = 40) : l * b = 75 := by
  sorry

end rectangle_area_l182_182679


namespace evaluate_expression_l182_182607

theorem evaluate_expression :
  (-2)^3 + (-2)^2 + (-2)^1 + 2^1 + 2^2 + 2^3 = 8 :=
by
  sorry

end evaluate_expression_l182_182607


namespace carmen_rope_gcd_l182_182265

/-- Carmen has three ropes with lengths 48, 64, and 80 inches respectively.
    She needs to cut these ropes into pieces of equal length for a craft project,
    ensuring no rope is left unused.
    Prove that the greatest length in inches that each piece can have is 16. -/
theorem carmen_rope_gcd :
  Nat.gcd (Nat.gcd 48 64) 80 = 16 := by
  sorry

end carmen_rope_gcd_l182_182265


namespace smallest_a_satisfies_sin_condition_l182_182468

open Real

theorem smallest_a_satisfies_sin_condition :
  ∃ (a : ℝ), (∀ x : ℤ, sin (a * x + 0) = sin (45 * x)) ∧ 0 ≤ a ∧ ∀ b : ℝ, (∀ x : ℤ, sin (b * x + 0) = sin (45 * x)) ∧ 0 ≤ b → 45 ≤ b :=
by
  -- To be proved.
  sorry

end smallest_a_satisfies_sin_condition_l182_182468


namespace value_of_f_m_plus_one_l182_182840

variable (a m : ℝ)

def f (x : ℝ) : ℝ := x^2 - x + a

theorem value_of_f_m_plus_one 
  (h : f a (-m) < 0) : f a (m + 1) < 0 := by
  sorry

end value_of_f_m_plus_one_l182_182840


namespace problems_per_page_l182_182404

theorem problems_per_page (total_problems finished_problems remaining_pages problems_per_page : ℕ)
  (h1 : total_problems = 40)
  (h2 : finished_problems = 26)
  (h3 : remaining_pages = 2)
  (h4 : total_problems - finished_problems = 14)
  (h5 : 14 = remaining_pages * problems_per_page) :
  problems_per_page = 7 := 
by
  sorry

end problems_per_page_l182_182404


namespace problem_l182_182214

def is_irrational (x : ℝ) : Prop := ¬ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

theorem problem :
  let A := 3.14159265
  let B := Real.sqrt 36
  let C := Real.sqrt 7
  let D := 4.1
  is_irrational C := by
  sorry

end problem_l182_182214


namespace ivan_max_13_bars_a_ivan_max_13_bars_b_l182_182174

variable (n : ℕ) (ivan_max_bags : ℕ)

-- Condition 1: initial count of bars in the chest
def initial_bars := 13

-- Condition 2: function to check if transfers are possible
def can_transfer (bars_in_chest : ℕ) (bars_in_bag : ℕ) (last_transfer : ℕ) : Prop :=
  ∃ t₁ t₂, t₁ ≠ t₂ ∧ t₁ ≠ last_transfer ∧ t₂ ≠ last_transfer ∧
           t₁ + bars_in_bag ≤ initial_bars ∧ bars_in_chest - t₁ + t₂ = bars_in_chest

-- Proof Problem (a): Given initially 13 bars, prove Ivan can secure 13 bars
theorem ivan_max_13_bars_a 
  (initial_bars : ℕ := 13) 
  (target_bars : ℕ := 13)
  (can_transfer : ∀ (bars_in_chest bars_in_bag last_transfer : ℕ), can_transfer bars_in_chest bars_in_bag last_transfer) 
  (h_initial_bars : initial_bars = 13) :
  ivan_max_bags = target_bars :=
by
  sorry

-- Proof Problem (b): Given initially 14 bars, prove Ivan can secure 13 bars
theorem ivan_max_13_bars_b 
  (initial_bars : ℕ := 14)
  (target_bars : ℕ := 13)
  (can_transfer : ∀ (bars_in_chest bars_in_bag last_transfer : ℕ), can_transfer bars_in_chest bars_in_bag last_transfer) 
  (h_initial_bars : initial_bars = 14) :
  ivan_max_bags = target_bars :=
by
  sorry

end ivan_max_13_bars_a_ivan_max_13_bars_b_l182_182174


namespace find_quadruple_l182_182674

theorem find_quadruple :
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 
  a^3 + b^4 + c^5 = d^11 ∧ a * b * c < 10^5 :=
sorry

end find_quadruple_l182_182674


namespace repairs_cost_correct_l182_182990

variable (C : ℝ)

def cost_of_scooter : ℝ := C
def repair_cost (C : ℝ) : ℝ := 0.10 * C
def selling_price (C : ℝ) : ℝ := 1.20 * C
def profit (C : ℝ) : ℝ := 1100
def profit_percentage (C : ℝ) : ℝ := 0.20 

theorem repairs_cost_correct (C : ℝ) (h₁ : selling_price C - cost_of_scooter C = profit C) (h₂ : profit_percentage C = 0.20) : 
  repair_cost C = 550 := by
  sorry

end repairs_cost_correct_l182_182990


namespace tan_div_sin_cos_sin_mul_cos_l182_182949

theorem tan_div_sin_cos (α : ℝ) (h : Real.tan α = 7) :
  (Real.sin α + Real.cos α) / (2 * Real.sin α - Real.cos α) = 8 / 13 :=
by
  sorry

theorem sin_mul_cos (α : ℝ) (h : Real.tan α = 7) :
  Real.sin α * Real.cos α = 7 / 50 :=
by
  sorry

end tan_div_sin_cos_sin_mul_cos_l182_182949


namespace triangle_area_l182_182938

/-- The area of the triangle enclosed by a line with slope -1/2 passing through (2, -3) and the coordinate axes is 4. -/
theorem triangle_area {l : ℝ → ℝ} (h1 : ∀ x, l x = -1/2 * x + b)
  (h2 : l 2 = -3) : 
  ∃ (A : ℝ) (B : ℝ), 
  ((l 0 = B) ∧ (l A = 0) ∧ (A ≠ 0) ∧ (B ≠ 0)) ∧
  (1/2 * |A| * |B| = 4) := 
sorry

end triangle_area_l182_182938


namespace scientific_notation_correct_l182_182650

def original_number : ℕ := 31900

def scientific_notation_option_A : ℝ := 3.19 * 10^2
def scientific_notation_option_B : ℝ := 0.319 * 10^3
def scientific_notation_option_C : ℝ := 3.19 * 10^4
def scientific_notation_option_D : ℝ := 0.319 * 10^5

theorem scientific_notation_correct :
  original_number = 31900 ∧ scientific_notation_option_C = 3.19 * 10^4 ∧ (original_number : ℝ) = scientific_notation_option_C := 
by 
  sorry

end scientific_notation_correct_l182_182650


namespace inequality_solution_l182_182429

theorem inequality_solution (x : ℝ) (h_pos : 0 < x) :
  (3 / 8 + |x - 14 / 24| < 8 / 12) ↔ x ∈ Set.Ioo (7 / 24) (7 / 8) :=
by
  sorry

end inequality_solution_l182_182429


namespace base_prime_representation_450_l182_182731

-- Define prime factorization property for number 450
def prime_factorization_450 := (450 = 2^1 * 3^2 * 5^2)

-- Define base prime representation concept
def base_prime_representation (n : ℕ) : ℕ := 
  if n = 450 then 122 else 0

-- Prove that the base prime representation of 450 is 122
theorem base_prime_representation_450 : 
  prime_factorization_450 →
  base_prime_representation 450 = 122 :=
by
  intros
  sorry

end base_prime_representation_450_l182_182731


namespace tan_alpha_plus_beta_l182_182564

open Real

theorem tan_alpha_plus_beta (A alpha beta : ℝ) (h1 : sin alpha = A * sin (alpha + beta)) (h2 : abs A > 1) :
  tan (alpha + beta) = sin beta / (cos beta - A) :=
by
  sorry

end tan_alpha_plus_beta_l182_182564


namespace catriona_total_fish_eq_44_l182_182578

-- Definitions based on conditions
def goldfish : ℕ := 8
def angelfish : ℕ := goldfish + 4
def guppies : ℕ := 2 * angelfish
def total_fish : ℕ := goldfish + angelfish + guppies

-- The theorem we need to prove
theorem catriona_total_fish_eq_44 : total_fish = 44 :=
by
  -- We are skipping the proof steps with 'sorry' for now
  sorry

end catriona_total_fish_eq_44_l182_182578


namespace product_of_102_and_27_l182_182662

theorem product_of_102_and_27 : 102 * 27 = 2754 :=
by
  sorry

end product_of_102_and_27_l182_182662


namespace polynomial_identity_l182_182435

theorem polynomial_identity :
  (3 * x ^ 2 - 4 * y ^ 3) * (9 * x ^ 4 + 12 * x ^ 2 * y ^ 3 + 16 * y ^ 6) = 27 * x ^ 6 - 64 * y ^ 9 :=
by
  sorry

end polynomial_identity_l182_182435


namespace find_ab_l182_182838

theorem find_ab 
(a b : ℝ) 
(h1 : a + b = 2) 
(h2 : a * b = 1 ∨ a * b = -1) :
(a = 1 ∧ b = 1) ∨
(a = 1 + Real.sqrt 2 ∧ b = 1 - Real.sqrt 2) ∨
(a = 1 - Real.sqrt 2 ∧ b = 1 + Real.sqrt 2) :=
sorry

end find_ab_l182_182838


namespace sum_ab_equals_five_l182_182094

-- Definitions for conditions
variables {a b : ℝ}

-- Assumption that establishes the solution set for the quadratic inequality
axiom quadratic_solution_set : ∀ x : ℝ, -2 < x ∧ x < 3 ↔ x^2 + b * x - a < 0

-- Statement to be proved
theorem sum_ab_equals_five : a + b = 5 :=
sorry

end sum_ab_equals_five_l182_182094


namespace find_polynomial_h_l182_182587

theorem find_polynomial_h (f h : ℝ → ℝ) (hf : ∀ x, f x = x^2) (hh : ∀ x, f (h x) = 9 * x^2 + 6 * x + 1) : 
  (∀ x, h x = 3 * x + 1) ∨ (∀ x, h x = -3 * x - 1) :=
by
  sorry

end find_polynomial_h_l182_182587


namespace find_length_of_wood_l182_182361

-- Definitions based on given conditions
def Area := 24  -- square feet
def Width := 6  -- feet

-- The mathematical proof problem turned into Lean 4 statement
theorem find_length_of_wood (h : Area = 24) (hw : Width = 6) : (Length : ℕ) ∈ {l | l = Area / Width ∧ l = 4} :=
by {
  sorry
}

end find_length_of_wood_l182_182361


namespace rowing_time_ratio_l182_182412

theorem rowing_time_ratio
  (V_b : ℝ) (V_s : ℝ) (V_upstream : ℝ) (V_downstream : ℝ) (T_upstream T_downstream : ℝ)
  (h1 : V_b = 39) (h2 : V_s = 13)
  (h3 : V_upstream = V_b - V_s) (h4 : V_downstream = V_b + V_s)
  (h5 : T_upstream * V_upstream = T_downstream * V_downstream) :
  T_upstream / T_downstream = 2 := by
  sorry

end rowing_time_ratio_l182_182412


namespace area_of_inscribed_rectangle_l182_182241

theorem area_of_inscribed_rectangle (r l w : ℝ) (h1 : r = 8) (h2 : l / w = 3) (h3 : w = 2 * r) : l * w = 768 :=
by
  sorry

end area_of_inscribed_rectangle_l182_182241


namespace missing_fraction_l182_182624

-- Definitions for the given fractions
def a := 1 / 3
def b := 1 / 2
def c := 1 / 5
def d := 1 / 4
def e := -9 / 20
def f := -2 / 15
def target_sum := 2 / 15 -- because 0.13333333333333333 == 2 / 15

-- Main theorem statement for the problem
theorem missing_fraction : a + b + c + d + e + f + -17 / 30 = target_sum :=
by
  simp [a, b, c, d, e, f, target_sum]
  sorry

end missing_fraction_l182_182624


namespace ratio_KL_eq_3_over_5_l182_182474

theorem ratio_KL_eq_3_over_5
  (K L : ℤ)
  (h : ∀ (x : ℝ), x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 3 → 
    (K : ℝ) / (x + 3) + (L : ℝ) / (x^2 - 3 * x) = (x^2 - x + 5) / (x^3 + x^2 - 9 * x)):
  (K : ℝ) / (L : ℝ) = 3 / 5 :=
by
  sorry

end ratio_KL_eq_3_over_5_l182_182474


namespace roots_of_quadratic_l182_182556

theorem roots_of_quadratic (x : ℝ) : x^2 - 5 * x = 0 ↔ (x = 0 ∨ x = 5) := by 
  sorry

end roots_of_quadratic_l182_182556


namespace initial_boxes_l182_182330

theorem initial_boxes (x : ℕ) (h1 : 80 + 165 = 245) (h2 : 2000 * 245 = 490000) 
                      (h3 : 4 * 245 * x + 245 * x = 1225 * x) : x = 400 :=
by
  sorry

end initial_boxes_l182_182330


namespace max_value_squared_of_ratio_l182_182681

-- Definition of positive real numbers with given conditions
variables (a b x y : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 

-- Main statement
theorem max_value_squared_of_ratio 
  (h_ge : a ≥ b)
  (h_eq_1 : a ^ 2 + y ^ 2 = b ^ 2 + x ^ 2)
  (h_eq_2 : b ^ 2 + x ^ 2 = (a - x) ^ 2 + (b + y) ^ 2)
  (h_range_x : 0 ≤ x ∧ x < a)
  (h_range_y : 0 ≤ y ∧ y < b)
  (h_additional_x : x = a - 2 * b)
  (h_additional_y : y = b / 2) : 
  (a / b) ^ 2 = 4 / 9 := 
sorry

end max_value_squared_of_ratio_l182_182681


namespace edge_length_of_small_cube_l182_182491

-- Define the parameters
def volume_cube : ℕ := 1000
def num_small_cubes : ℕ := 8
def remaining_volume : ℕ := 488

-- Define the main theorem
theorem edge_length_of_small_cube (x : ℕ) :
  (volume_cube - num_small_cubes * x^3 = remaining_volume) → x = 4 := 
by 
  sorry

end edge_length_of_small_cube_l182_182491


namespace problem_solution_l182_182812

-- Given non-zero numbers x and y such that x = 1 / y,
-- prove that (2x - 1/x) * (y - 1/y) = -2x^2 + y^2.
theorem problem_solution (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x = 1 / y) :
  (2 * x - 1 / x) * (y - 1 / y) = -2 * x^2 + y^2 :=
by
  sorry

end problem_solution_l182_182812


namespace find_matrix_M_l182_182796

theorem find_matrix_M (M : Matrix (Fin 2) (Fin 2) ℝ) (h : M^3 - 3 • M^2 + 4 • M = ![![6, 12], ![3, 6]]) :
  M = ![![2, 4], ![1, 2]] :=
sorry

end find_matrix_M_l182_182796


namespace length_vector_eq_three_l182_182101

theorem length_vector_eq_three (A B : ℝ) (hA : A = -1) (hB : B = 2) : |B - A| = 3 :=
by
  sorry

end length_vector_eq_three_l182_182101


namespace cylinder_lateral_surface_area_l182_182293

theorem cylinder_lateral_surface_area
    (r h : ℝ) (hr : r = 3) (hh : h = 10) :
    2 * Real.pi * r * h = 60 * Real.pi :=
by
  sorry

end cylinder_lateral_surface_area_l182_182293


namespace stream_current_rate_l182_182128

theorem stream_current_rate (r w : ℝ) : (
  (18 / (r + w) + 6 = 18 / (r - w)) ∧ 
  (18 / (3 * r + w) + 2 = 18 / (3 * r - w))
) → w = 6 := 
by {
  sorry
}

end stream_current_rate_l182_182128


namespace area_of_triangle_AEC_l182_182893

theorem area_of_triangle_AEC (BE EC : ℝ) (h_ratio : BE / EC = 3 / 2) (area_abe : ℝ) (h_area_abe : area_abe = 27) : 
  ∃ area_aec, area_aec = 18 :=
by
  sorry

end area_of_triangle_AEC_l182_182893


namespace ratio_of_volumes_of_spheres_l182_182498

theorem ratio_of_volumes_of_spheres (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : a / b = 1 / 2 ∧ b / c = 2 / 3) : a^3 / b^3 = 1 / 8 ∧ b^3 / c^3 = 8 / 27 :=
by
  sorry

end ratio_of_volumes_of_spheres_l182_182498


namespace other_x_intercept_l182_182109

noncomputable def quadratic_function_vertex :=
  ∃ (a b c : ℝ), ∀ (x : ℝ), (a ≠ 0) →
  (5, -3) = ((-b) / (2 * a), a * ((-b) / (2 * a))^2 + b * ((-b) / (2 * a)) + c) ∧
  (x = 1) ∧ (a * x^2 + b * x + c = 0) →
  ∃ (x2 : ℝ), x2 = 9

theorem other_x_intercept :
  quadratic_function_vertex :=
sorry

end other_x_intercept_l182_182109


namespace fraction_identity_l182_182976

def at_op (a b : ℤ) : ℤ := a * b - 3 * b ^ 2
def hash_op (a b : ℤ) : ℤ := a + 2 * b - 2 * a * b ^ 2

theorem fraction_identity : (at_op 8 3) / (hash_op 8 3) = 3 / 130 := by
  sorry

end fraction_identity_l182_182976


namespace ratio_hexagon_octagon_l182_182721

noncomputable def ratio_of_areas (s : ℝ) :=
  let A1 := s / (2 * Real.tan (Real.pi / 6))
  let H1 := s / (2 * Real.sin (Real.pi / 6))
  let area1 := Real.pi * (H1^2 - A1^2)
  let A2 := s / (2 * Real.tan (Real.pi / 8))
  let H2 := s / (2 * Real.sin (Real.pi / 8))
  let area2 := Real.pi * (H2^2 - A2^2)
  area1 / area2

theorem ratio_hexagon_octagon (s : ℝ) (h : s = 3) : ratio_of_areas s = 49 / 25 :=
  sorry

end ratio_hexagon_octagon_l182_182721


namespace largest_number_of_hcf_lcm_l182_182042

theorem largest_number_of_hcf_lcm (a b c : ℕ) (h : Nat.gcd (Nat.gcd a b) c = 42)
  (factor1 : 10 ∣ Nat.lcm (Nat.lcm a b) c)
  (factor2 : 20 ∣ Nat.lcm (Nat.lcm a b) c)
  (factor3 : 25 ∣ Nat.lcm (Nat.lcm a b) c)
  (factor4 : 30 ∣ Nat.lcm (Nat.lcm a b) c) :
  max (max a b) c = 1260 := 
  sorry

end largest_number_of_hcf_lcm_l182_182042


namespace simplify_and_find_ab_ratio_l182_182179

-- Given conditions
def given_expression (k : ℤ) : ℤ := 10 * k + 15

-- Simplified form
def simplified_form (k : ℤ) : ℤ := 2 * k + 3

-- Proof problem statement
theorem simplify_and_find_ab_ratio
  (k : ℤ) :
  let a := 2
  let b := 3
  (10 * k + 15) / 5 = 2 * k + 3 → 
  (a:ℚ) / (b:ℚ) = 2 / 3 := sorry

end simplify_and_find_ab_ratio_l182_182179


namespace twentyfive_percent_in_usd_l182_182446

variable (X : ℝ)
variable (Y : ℝ) (hY : Y > 0)

theorem twentyfive_percent_in_usd : 0.25 * X * Y = (0.25 : ℝ) * X * Y := by
  sorry

end twentyfive_percent_in_usd_l182_182446


namespace ratio_of_lengths_l182_182723

variables (l1 l2 l3 : ℝ)

theorem ratio_of_lengths (h1 : l2 = (1/2) * (l1 + l3))
                         (h2 : l2^3 = (6/13) * (l1^3 + l3^3)) :
  (l1 / l3) = (7 / 5) :=
by
  sorry

end ratio_of_lengths_l182_182723


namespace find_t_l182_182193

variables (c o u n t s : ℕ)

theorem find_t (h1 : c + o = u) 
               (h2 : u + n = t)
               (h3 : t + c = s)
               (h4 : o + n + s = 18)
               (hz : c > 0) (ho : o > 0) (hu : u > 0) (hn : n > 0) (ht : t > 0) (hs : s > 0) : 
               t = 9 := 
by
  sorry

end find_t_l182_182193


namespace trip_time_maple_to_oak_l182_182514

noncomputable def total_trip_time (d1 d2 v1 v2 t_break : ℝ) : ℝ :=
  (d1 / v1) + t_break + (d2 / v2)

theorem trip_time_maple_to_oak : 
  total_trip_time 210 210 50 40 0.5 = 5.75 :=
by
  sorry

end trip_time_maple_to_oak_l182_182514


namespace volume_common_solid_hemisphere_cone_l182_182500

noncomputable def volume_common_solid (r : ℝ) : ℝ := 
  let V_1 := (2/3) * Real.pi * (r^3 - (3 * r / 5)^3)
  let V_2 := Real.pi * ((r / 5)^2) * (r - (r / 15))
  V_1 + V_2

theorem volume_common_solid_hemisphere_cone (r : ℝ) :
  volume_common_solid r = (14 * Real.pi * r^3) / 25 := 
by
  sorry

end volume_common_solid_hemisphere_cone_l182_182500


namespace find_m_l182_182671

-- Definitions for the sets
def setA (x : ℝ) : Prop := -2 < x ∧ x < 8
def setB (m : ℝ) (x : ℝ) : Prop := 2 * m - 1 < x ∧ x < m + 3

-- Condition on the intersection
def intersection (m : ℝ) (a b : ℝ) (x : ℝ) : Prop := 2 * m - 1 < x ∧ x < m + 3 ∧ -2 < x ∧ x < 8

-- Theorem statement
theorem find_m (m a b : ℝ) (h₀ : b - a = 3) (h₁ : ∀ x, intersection m a b x ↔ (a < x ∧ x < b)) : m = -2 ∨ m = 1 :=
sorry

end find_m_l182_182671


namespace inequality_holds_for_all_reals_l182_182896

theorem inequality_holds_for_all_reals (x y z : ℝ) :
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 :=
by
  sorry

end inequality_holds_for_all_reals_l182_182896


namespace debby_bottles_l182_182945

noncomputable def number_of_bottles_initial : ℕ := 301
noncomputable def number_of_bottles_drank : ℕ := 144
noncomputable def number_of_bottles_left : ℕ := 157

theorem debby_bottles:
  (number_of_bottles_initial - number_of_bottles_drank) = number_of_bottles_left :=
sorry

end debby_bottles_l182_182945


namespace solve_equation_1_solve_equation_2_l182_182847

theorem solve_equation_1 (x : ℝ) : x * (x - 2) + x - 2 = 0 ↔ (x = 2 ∨ x = -1) :=
by sorry

theorem solve_equation_2 (x : ℝ) : 2 * x^2 + 5 * x + 3 = 0 ↔ (x = -1 ∨ x = -3/2) :=
by sorry

end solve_equation_1_solve_equation_2_l182_182847


namespace inheritance_amount_l182_182932

-- Definitions based on conditions given
def inheritance (x : ℝ) : Prop :=
  let federal_tax := 0.25 * x
  let remaining_after_federal := x - federal_tax
  let state_tax := 0.15 * remaining_after_federal
  let total_tax := federal_tax + state_tax
  total_tax = 15000

-- The statement to be proven
theorem inheritance_amount (x : ℝ) (hx : inheritance x) : x = 41379 :=
by
  -- Proof goes here
  sorry

end inheritance_amount_l182_182932


namespace part1_part2_l182_182839

variables (A B C : ℝ)
variables (a b c : ℝ) -- sides of the triangle opposite to angles A, B, and C respectively

-- Part (I): Prove that c / a = 2 given b(cos A - 2 * cos C) = (2 * c - a) * cos B
theorem part1 (h1 : b * (Real.cos A - 2 * Real.cos C) = (2 * c - a) * Real.cos B) : c / a = 2 :=
sorry

-- Part (II): Prove that b = 2 given the results from part (I) and additional conditions
theorem part2 (h1 : c / a = 2) (h2 : Real.cos B = 1 / 4) (h3 : a + b + c = 5) : b = 2 :=
sorry

end part1_part2_l182_182839


namespace arithmetic_sequence_product_l182_182495

theorem arithmetic_sequence_product (b : ℕ → ℤ) (n : ℕ)
  (h1 : ∀ n, b (n + 1) > b n)
  (h2 : b 5 * b 6 = 21) :
  b 4 * b 7 = -779 ∨ b 4 * b 7 = -11 :=
sorry

end arithmetic_sequence_product_l182_182495


namespace alice_number_l182_182849

theorem alice_number (n : ℕ) 
  (h1 : 243 ∣ n) 
  (h2 : 36 ∣ n) 
  (h3 : 1000 < n) 
  (h4 : n < 3000) : 
  n = 1944 ∨ n = 2916 := 
sorry

end alice_number_l182_182849


namespace three_integers_same_parity_l182_182809

theorem three_integers_same_parity (a b c : ℤ) : 
  (∃ i j, i ≠ j ∧ (i = a ∨ i = b ∨ i = c) ∧ (j = a ∨ j = b ∨ j = c) ∧ (i % 2 = j % 2)) :=
by
  sorry

end three_integers_same_parity_l182_182809


namespace inequality_solution_set_l182_182953

theorem inequality_solution_set : 
  {x : ℝ | -x^2 + 4*x + 5 < 0} = {x : ℝ | x < -1 ∨ x > 5} := 
by
  sorry

end inequality_solution_set_l182_182953


namespace number_of_possible_values_for_b_l182_182116

theorem number_of_possible_values_for_b : 
  ∃ (n : ℕ), n = 2 ∧ ∀ b : ℕ, (b ≥ 2 ∧ b^3 ≤ 197 ∧ 197 < b^4) → b = 4 ∨ b = 5 :=
sorry

end number_of_possible_values_for_b_l182_182116


namespace paintable_wall_area_correct_l182_182963

noncomputable def paintable_wall_area : Nat :=
  let length := 15
  let width := 11
  let height := 9
  let closet_width := 3
  let closet_length := 4
  let unused_area := 70
  let room_wall_area :=
    2 * (length * height) +
    2 * (width * height)
  let closet_wall_area := 
    2 * (closet_width * height)
  let paintable_area_per_bedroom := 
    room_wall_area - (unused_area + closet_wall_area)
  4 * paintable_area_per_bedroom

theorem paintable_wall_area_correct : paintable_wall_area = 1376 := by
  sorry

end paintable_wall_area_correct_l182_182963


namespace staff_member_pays_l182_182748

noncomputable def calculate_final_price (d : ℝ) : ℝ :=
  let discounted_price := 0.55 * d
  let staff_discounted_price := 0.33 * d
  let final_price := staff_discounted_price + 0.08 * staff_discounted_price
  final_price

theorem staff_member_pays (d : ℝ) : calculate_final_price d = 0.3564 * d :=
by
  unfold calculate_final_price
  sorry

end staff_member_pays_l182_182748


namespace inequality_must_hold_l182_182353

theorem inequality_must_hold (m n : ℝ) (h : m > n) : 2 + m > 2 + n :=
sorry

end inequality_must_hold_l182_182353


namespace tan_sum_l182_182767

theorem tan_sum (A B : ℝ) (h₁ : A = 17) (h₂ : B = 28) :
  Real.tan (A) + Real.tan (B) + Real.tan (A) * Real.tan (B) = 1 := 
by
  sorry

end tan_sum_l182_182767


namespace boys_cannot_score_twice_l182_182530

-- Define the total number of points in the tournament
def total_points_in_tournament : ℕ := 15

-- Define the number of boys and girls
def number_of_boys : ℕ := 2
def number_of_girls : ℕ := 4

-- Define the points scored by boys and girls
axiom points_by_boys : ℕ
axiom points_by_girls : ℕ

-- The conditions
axiom total_points_condition : points_by_boys + points_by_girls = total_points_in_tournament
axiom boys_twice_girls_condition : points_by_boys = 2 * points_by_girls

-- The statement to prove
theorem boys_cannot_score_twice : False :=
  by {
    -- Note: provide a sketch to illustrate that under the given conditions the statement is false
    sorry
  }

end boys_cannot_score_twice_l182_182530


namespace runs_twice_l182_182547

-- Definitions of the conditions
def game_count : ℕ := 6
def runs_one : ℕ := 1
def runs_five : ℕ := 5
def average_runs : ℕ := 4

-- Assuming the number of runs scored twice is x
variable (x : ℕ)

-- Definition of total runs scored based on the conditions
def total_runs : ℕ := runs_one + 2 * x + 3 * runs_five

-- Statement to prove the number of runs scored twice
theorem runs_twice :
  (total_runs x) / game_count = average_runs → x = 4 :=
by
  sorry

end runs_twice_l182_182547


namespace perfect_square_trinomial_m_l182_182943

theorem perfect_square_trinomial_m (m : ℤ) (x : ℤ) : (∃ a : ℤ, x^2 - mx + 16 = (x - a)^2) ↔ (m = 8 ∨ m = -8) :=
by sorry

end perfect_square_trinomial_m_l182_182943


namespace number_of_sets_of_popcorn_l182_182386

theorem number_of_sets_of_popcorn (t p s : ℝ) (k : ℕ) 
  (h1 : t = 5)
  (h2 : p = 0.80 * t)
  (h3 : s = 0.50 * p)
  (h4 : 4 * t + 4 * s + k * p = 36) :
  k = 2 :=
by sorry

end number_of_sets_of_popcorn_l182_182386


namespace probability_point_between_lines_l182_182356

theorem probability_point_between_lines :
  let l (x : ℝ) := -2 * x + 8
  let m (x : ℝ) := -3 * x + 9
  let area_l := 1 / 2 * 4 * 8
  let area_m := 1 / 2 * 3 * 9
  let area_between := area_l - area_m
  let probability := area_between / area_l
  probability = 0.16 :=
by
  sorry

end probability_point_between_lines_l182_182356


namespace value_range_for_inequality_solution_set_l182_182948

-- Define the condition
def condition (a : ℝ) : Prop := a > 0

-- Define the inequality
def inequality (x a : ℝ) : Prop := |x - 4| + |x - 3| < a

-- State the theorem to be proven
theorem value_range_for_inequality_solution_set (a : ℝ) (h: condition a) : (a > 1) ↔ ∃ x : ℝ, inequality x a := 
sorry

end value_range_for_inequality_solution_set_l182_182948


namespace find_alpha_minus_beta_find_cos_2alpha_minus_beta_l182_182463

-- Definitions and assumptions
variables (α β : ℝ)
axiom sin_alpha : Real.sin α = (Real.sqrt 5) / 5
axiom sin_beta : Real.sin β = (3 * Real.sqrt 10) / 10
axiom alpha_acute : 0 < α ∧ α < Real.pi / 2
axiom beta_acute : 0 < β ∧ β < Real.pi / 2

-- Statement to prove α - β = -π/4
theorem find_alpha_minus_beta : α - β = -Real.pi / 4 :=
sorry

-- Given α - β = -π/4, statement to prove cos(2α - β) = 3√10 / 10
theorem find_cos_2alpha_minus_beta (h : α - β = -Real.pi / 4) : Real.cos (2 * α - β) = (3 * Real.sqrt 10) / 10 :=
sorry

end find_alpha_minus_beta_find_cos_2alpha_minus_beta_l182_182463


namespace PlayStation_cost_l182_182411

def birthday_money : ℝ := 200
def christmas_money : ℝ := 150
def price_per_game : ℝ := 7.5
def games_to_sell : ℕ := 20
def total_gift_money : ℝ := birthday_money + christmas_money
def total_games_money : ℝ := games_to_sell * price_per_game
def total_money : ℝ := total_gift_money + total_games_money

theorem PlayStation_cost : total_money = 500 := by
  sorry

end PlayStation_cost_l182_182411


namespace Anthony_vs_Jim_l182_182936

variable (Scott_pairs : ℕ)
variable (Anthony_pairs : ℕ)
variable (Jim_pairs : ℕ)

axiom Scott_value : Scott_pairs = 7
axiom Anthony_value : Anthony_pairs = 3 * Scott_pairs
axiom Jim_value : Jim_pairs = Anthony_pairs - 2

theorem Anthony_vs_Jim (Scott_pairs Anthony_pairs Jim_pairs : ℕ) 
  (Scott_value : Scott_pairs = 7) 
  (Anthony_value : Anthony_pairs = 3 * Scott_pairs) 
  (Jim_value : Jim_pairs = Anthony_pairs - 2) :
  Anthony_pairs - Jim_pairs = 2 := 
sorry

end Anthony_vs_Jim_l182_182936


namespace man_speed_down_l182_182167

variable (d : ℝ) (v : ℝ)

theorem man_speed_down (h1 : 32 > 0) (h2 : 38.4 > 0) (h3 : d > 0) (h4 : v > 0) 
  (avg_speed : 38.4 = (2 * d) / ((d / 32) + (d / v))) : v = 48 :=
sorry

end man_speed_down_l182_182167


namespace total_movies_in_series_l182_182855

def book_count := 4
def total_books_read := 19
def movies_watched := 7
def movies_to_watch := 10

theorem total_movies_in_series : movies_watched + movies_to_watch = 17 := by
  sorry

end total_movies_in_series_l182_182855


namespace minimum_prism_volume_l182_182857

theorem minimum_prism_volume (l m n : ℕ) (h1 : l > 0) (h2 : m > 0) (h3 : n > 0)
    (hidden_volume_condition : (l - 1) * (m - 1) * (n - 1) = 420) :
    ∃ N : ℕ, N = l * m * n ∧ N = 630 := by
  sorry

end minimum_prism_volume_l182_182857


namespace toy_cost_l182_182686

-- Definitions based on the conditions in part a)
def initial_amount : ℕ := 57
def spent_amount : ℕ := 49
def remaining_amount : ℕ := initial_amount - spent_amount
def number_of_toys : ℕ := 2

-- Statement to prove that each toy costs 4 dollars
theorem toy_cost :
  (remaining_amount / number_of_toys) = 4 :=
by
  sorry

end toy_cost_l182_182686


namespace total_cookies_needed_l182_182189

-- Define the conditions
def cookies_per_person : ℝ := 24.0
def number_of_people : ℝ := 6.0

-- Define the goal
theorem total_cookies_needed : cookies_per_person * number_of_people = 144.0 :=
by
  sorry

end total_cookies_needed_l182_182189


namespace length_of_bridge_is_correct_l182_182712

noncomputable def length_of_inclined_bridge (initial_speed : ℕ) (time : ℕ) (acceleration : ℕ) : ℚ :=
  (1 / 60) * (time * initial_speed + (time * (time - 1)) / 2)

theorem length_of_bridge_is_correct : 
  length_of_inclined_bridge 10 18 1 = 5.55 := 
by
  sorry

end length_of_bridge_is_correct_l182_182712


namespace largest_square_perimeter_is_28_l182_182969

-- Definitions and assumptions
def rect_length : ℝ := 10
def rect_width : ℝ := 7

-- Define the largest possible square
def largest_square_side := rect_width

-- Define the perimeter of a square
def perimeter_of_square (side : ℝ) : ℝ := 4 * side

-- Proving statement
theorem largest_square_perimeter_is_28 :
  perimeter_of_square largest_square_side = 28 := 
  by 
    -- sorry is used to skip the proof
    sorry

end largest_square_perimeter_is_28_l182_182969


namespace team_a_wins_3_2_prob_l182_182759

-- Definitions for the conditions in the problem
def prob_win_first_four : ℚ := 2 / 3
def prob_win_fifth : ℚ := 1 / 2

-- Definitions related to combinations
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Main statement: Proving the probability of winning 3:2
theorem team_a_wins_3_2_prob :
  (C 4 2 * (prob_win_first_four ^ 2) * ((1 - prob_win_first_four) ^ 2) * prob_win_fifth) = 4 / 27 := 
sorry

end team_a_wins_3_2_prob_l182_182759


namespace total_pens_l182_182321

theorem total_pens (r : ℕ) (r_gt_10 : r > 10) (r_div_357 : r ∣ 357) (r_div_441 : r ∣ 441) :
  357 / r + 441 / r = 38 := by
  sorry

end total_pens_l182_182321


namespace abs_inequality_solution_l182_182971

theorem abs_inequality_solution (x : ℝ) : (|x - 1| < 2) ↔ (x > -1 ∧ x < 3) := 
sorry

end abs_inequality_solution_l182_182971


namespace coeff_x3_in_binom_expansion_l182_182559

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the coefficient function for x^k in the binomial expansion of (x + 1)^n
def binom_coeff (n k : ℕ) : ℕ := binom n k

-- The theorem to prove that the coefficient of x^3 in the expansion of (x + 1)^36 is 7140
theorem coeff_x3_in_binom_expansion : binom_coeff 36 3 = 7140 :=
by
  sorry

end coeff_x3_in_binom_expansion_l182_182559


namespace problem_divisibility_l182_182979

theorem problem_divisibility (a : ℤ) (h1 : 0 ≤ a) (h2 : a ≤ 13) (h3 : (51^2012 + a) % 13 = 0) : a = 12 :=
by
  sorry

end problem_divisibility_l182_182979


namespace John_pays_amount_l182_182572

/-- Prove the amount John pays given the conditions -/
theorem John_pays_amount
  (total_candies : ℕ)
  (candies_paid_by_dave : ℕ)
  (cost_per_candy : ℚ)
  (candies_paid_by_john := total_candies - candies_paid_by_dave)
  (total_cost_paid_by_john := candies_paid_by_john * cost_per_candy) :
  total_candies = 20 →
  candies_paid_by_dave = 6 →
  cost_per_candy = 1.5 →
  total_cost_paid_by_john = 21 := 
by
  intros h1 h2 h3
  -- Proof is skipped
  sorry

end John_pays_amount_l182_182572


namespace largest_square_l182_182752

def sticks_side1 : List ℕ := [4, 4, 2, 3]
def sticks_side2 : List ℕ := [4, 4, 3, 1, 1]
def sticks_side3 : List ℕ := [4, 3, 3, 2, 1]
def sticks_side4 : List ℕ := [3, 3, 3, 2, 2]

def sum_of_sticks (sticks : List ℕ) : ℕ := sticks.foldl (· + ·) 0

theorem largest_square (h1 : sum_of_sticks sticks_side1 = 13)
                      (h2 : sum_of_sticks sticks_side2 = 13)
                      (h3 : sum_of_sticks sticks_side3 = 13)
                      (h4 : sum_of_sticks sticks_side4 = 13) :
  13 = 13 := by
  sorry

end largest_square_l182_182752


namespace max_gcd_is_one_l182_182191

-- Defining the sequence a_n
def a_n (n : ℕ) : ℕ := 101 + n^3

-- Defining the gcd function for a_n and a_(n+1)
def d_n (n : ℕ) : ℕ := Nat.gcd (a_n n) (a_n (n + 1))

-- The theorem stating the maximum value of d_n is 1
theorem max_gcd_is_one : ∀ n : ℕ, d_n n = 1 := by
  -- Proof is omitted as per instructions
  sorry

end max_gcd_is_one_l182_182191


namespace ancient_chinese_wine_problem_l182_182563

theorem ancient_chinese_wine_problem:
  ∃ x: ℝ, 10 * x + 3 * (5 - x) = 30 :=
by
  sorry

end ancient_chinese_wine_problem_l182_182563


namespace sum_of_three_numbers_is_520_l182_182736

noncomputable def sum_of_three_numbers (x y z : ℝ) : ℝ :=
  x + y + z

theorem sum_of_three_numbers_is_520 (x y z : ℝ) (h1 : z = (1848 / 1540) * x) (h2 : z = 0.4 * y) (h3 : x + y = 400) :
  sum_of_three_numbers x y z = 520 :=
sorry

end sum_of_three_numbers_is_520_l182_182736


namespace mobius_total_trip_time_l182_182005

theorem mobius_total_trip_time :
  ∀ (d1 d2 v1 v2 : ℝ) (n r : ℕ),
  d1 = 143 → d2 = 143 → 
  v1 = 11 → v2 = 13 → 
  n = 4 → r = (30:ℝ)/60 →
  d1 / v1 + d2 / v2 + n * r = 26 :=
by
  intros d1 d2 v1 v2 n r h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  norm_num

end mobius_total_trip_time_l182_182005


namespace max_discount_rate_l182_182986

theorem max_discount_rate (cp sp : ℝ) (min_profit_margin discount_rate : ℝ) 
  (h_cost : cp = 4) 
  (h_sell : sp = 5) 
  (h_profit : min_profit_margin = 0.1) :
  discount_rate ≤ 12 :=
by 
  sorry

end max_discount_rate_l182_182986


namespace minimum_xy_l182_182623

theorem minimum_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 1/y = 1/2) : x * y ≥ 16 :=
sorry

end minimum_xy_l182_182623


namespace problem_l182_182810

theorem problem (a b : ℕ)
  (ha : a = 2) 
  (hb : b = 121) 
  (h_minPrime : ∀ n, n < a → ¬ (∀ d, d ∣ n → d = 1 ∨ d = n))
  (h_threeDivisors : ∀ n, n < 150 → ∀ d, d ∣ n → d = 1 ∨ d = n → n = 121) :
  a + b = 123 := by
  sorry

end problem_l182_182810


namespace common_ratio_geometric_sequence_l182_182222

theorem common_ratio_geometric_sequence
  (a_1 : ℝ) (n : ℕ) (S : ℕ → ℝ)
  (geom_sum : ∀ n q, q ≠ 1 → S n = a_1 * (1 - q^n) / (1 - q))
  (h_arithmetic : 2 * S 4 = S 5 + S 6)
  : (∃ q : ℝ, ∀ n : ℕ, q ≠ 1 → S n = a_1 * (1 - q^n) / (1 - q)) → q = -2 :=
by
  sorry

end common_ratio_geometric_sequence_l182_182222


namespace sqrt_equality_l182_182770

theorem sqrt_equality :
  Real.sqrt ((18: ℝ) * (17: ℝ) * (16: ℝ) * (15: ℝ) + 1) = 271 :=
by
  sorry

end sqrt_equality_l182_182770


namespace mowed_times_in_spring_l182_182017

-- Definition of the problem conditions
def total_mowed_times : ℕ := 11
def summer_mowed_times : ℕ := 5

-- The theorem to prove
theorem mowed_times_in_spring : (total_mowed_times - summer_mowed_times = 6) :=
by
  sorry

end mowed_times_in_spring_l182_182017


namespace smallest_n_l182_182899

theorem smallest_n (n : ℕ) : 
  (n % 6 = 2) ∧ (n % 7 = 3) ∧ (n % 8 = 4) → n = 8 :=
  by sorry

end smallest_n_l182_182899


namespace sufficient_condition_of_implications_l182_182267

variables (P1 P2 θ : Prop)

theorem sufficient_condition_of_implications
  (h1 : P1 → θ)
  (h2 : P2 → P1) :
  P2 → θ :=
by sorry

end sufficient_condition_of_implications_l182_182267


namespace find_n_l182_182188

-- Define the variables d, Q, r, m, and n
variables (d Q r m n : ℝ)

-- Define the conditions Q = d / ((1 + r)^n - m) and m < (1 + r)^n
def conditions (d Q r m n : ℝ) : Prop :=
  Q = d / ((1 + r)^n - m) ∧ m < (1 + r)^n

theorem find_n (d Q r m : ℝ) (h : conditions d Q r m n) : 
  n = (Real.log (d / Q + m)) / (Real.log (1 + r)) :=
sorry

end find_n_l182_182188


namespace angle_A_is_30_degrees_l182_182851

theorem angle_A_is_30_degrees
    (a b : ℝ)
    (B A : ℝ)
    (a_eq_4 : a = 4)
    (b_eq_4_sqrt2 : b = 4 * Real.sqrt 2)
    (B_eq_45 : B = Real.pi / 4) : 
    A = Real.pi / 6 := 
by 
    sorry

end angle_A_is_30_degrees_l182_182851


namespace parabola_focus_to_directrix_distance_correct_l182_182613

def parabola_focus_to_directrix_distance (a : ℕ) (y x : ℝ) : Prop :=
  y^2 = 2 * x → a = 2 →  1 = 1

theorem parabola_focus_to_directrix_distance_correct :
  ∀ (a : ℕ) (y x : ℝ), parabola_focus_to_directrix_distance a y x :=
by
  unfold parabola_focus_to_directrix_distance
  intros
  sorry

end parabola_focus_to_directrix_distance_correct_l182_182613


namespace hunter_time_comparison_l182_182894

-- Definitions for time spent in swamp, forest, and highway
variables {a b c : ℝ}

-- Given conditions
-- 1. Total time equation
#check a + b + c = 4

-- 2. Total distance equation
#check 2 * a + 4 * b + 6 * c = 17

-- Prove that the hunter spent more time on the highway than in the swamp
theorem hunter_time_comparison (h1 : a + b + c = 4) (h2 : 2 * a + 4 * b + 6 * c = 17) : c > a :=
by sorry

end hunter_time_comparison_l182_182894


namespace minimum_value_abs_a_plus_2_abs_b_l182_182642

open Real

theorem minimum_value_abs_a_plus_2_abs_b 
  (a b : ℝ)
  (f : ℝ → ℝ)
  (x₁ x₂ x₃ : ℝ)
  (f_def : ∀ x, f x = x^3 + a*x^2 + b*x)
  (roots_cond : x₁ + 1 ≤ x₂ ∧ x₂ ≤ x₃ - 1)
  (equal_values : f x₁ = f x₂ ∧ f x₂ = f x₃) :
  ∃ minimum, minimum = (sqrt 3) ∧ (∀ (a b : ℝ), |a| + 2*|b| ≥ sqrt 3) :=
by
  sorry

end minimum_value_abs_a_plus_2_abs_b_l182_182642
