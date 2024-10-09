import Mathlib

namespace only_one_solution_l1131_113172

theorem only_one_solution (n : ℕ) (h : 0 < n ∧ ∃ a : ℕ, a * a = 5^n + 4) : n = 1 :=
sorry

end only_one_solution_l1131_113172


namespace inequality_solution_set_l1131_113165

theorem inequality_solution_set (a b : ℝ) (h1 : a > 0) (h2 : ∀ x : ℝ, ax^2 + bx - 1 < 0 ↔ -1/2 < x ∧ x < 1) :
  ∀ x : ℝ, (2 * x + 2) / (-x + 1) < 0 ↔ (x < -1 ∨ x > 1) :=
by sorry

end inequality_solution_set_l1131_113165


namespace area_increase_by_16_percent_l1131_113126

theorem area_increase_by_16_percent (L B : ℝ) :
  ((1.45 * L) * (0.80 * B)) / (L * B) = 1.16 :=
by
  sorry

end area_increase_by_16_percent_l1131_113126


namespace calculate_rolls_of_toilet_paper_l1131_113135

-- Definitions based on the problem conditions
def seconds_per_egg := 15
def minutes_per_roll := 30
def total_cleaning_minutes := 225
def number_of_eggs := 60
def time_per_minute := 60

-- Calculation of the time spent on eggs in minutes
def egg_cleaning_minutes := (number_of_eggs * seconds_per_egg) / time_per_minute

-- Total cleaning time minus time spent on eggs
def remaining_cleaning_minutes := total_cleaning_minutes - egg_cleaning_minutes

-- Verify the number of rolls of toilet paper cleaned up
def rolls_of_toilet_paper := remaining_cleaning_minutes / minutes_per_roll

-- Theorem statement to be proved
theorem calculate_rolls_of_toilet_paper : rolls_of_toilet_paper = 7 := by
  sorry

end calculate_rolls_of_toilet_paper_l1131_113135


namespace angle_covered_in_three_layers_l1131_113130

theorem angle_covered_in_three_layers 
  (total_coverage : ℝ) (sum_of_angles : ℝ) 
  (h1 : total_coverage = 90) (h2 : sum_of_angles = 290) : 
  ∃ x : ℝ, 3 * x + 2 * (90 - x) = 290 ∧ x = 20 :=
by
  sorry

end angle_covered_in_three_layers_l1131_113130


namespace ab_divides_a_squared_plus_b_squared_l1131_113145

theorem ab_divides_a_squared_plus_b_squared (a b : ℕ) (hab : a ≠ 1 ∨ b ≠ 1) (hpos : 0 < a ∧ 0 < b) (hdiv : (ab - 1) ∣ (a^2 + b^2)) :
  a^2 + b^2 = 5 * a * b - 5 := 
by
  sorry

end ab_divides_a_squared_plus_b_squared_l1131_113145


namespace exists_integers_u_v_l1131_113107

theorem exists_integers_u_v (A : ℕ) (a b s : ℤ)
  (hA: A = 1 ∨ A = 2 ∨ A = 3)
  (hab_rel_prime: Int.gcd a b = 1)
  (h_eq: a^2 + A * b^2 = s^3) :
  ∃ u v : ℤ, s = u^2 + A * v^2 ∧ a = u^3 - 3 * A * u * v^2 ∧ b = 3 * u^2 * v - A * v^3 := 
sorry

end exists_integers_u_v_l1131_113107


namespace max_books_borrowed_l1131_113163

theorem max_books_borrowed (total_students : ℕ) (students_no_books : ℕ) (students_1_book : ℕ)
  (students_2_books : ℕ) (avg_books_per_student : ℕ) (remaining_students_borrowed_at_least_3 :
  ∀ (s : ℕ), s ≥ 3) :
  total_students = 25 →
  students_no_books = 3 →
  students_1_book = 11 →
  students_2_books = 6 →
  avg_books_per_student = 2 →
  ∃ (max_books : ℕ), max_books = 15 :=
  by
  sorry

end max_books_borrowed_l1131_113163


namespace x_intercept_of_line_l1131_113164

theorem x_intercept_of_line (x1 y1 x2 y2 : ℝ) (h1 : (x1, y1) = (2, -4)) (h2 : (x2, y2) = (6, 8)) : 
  ∃ x0 : ℝ, (x0 = (10 / 3) ∧ ∃ m : ℝ, m = (y2 - y1) / (x2 - x1) ∧ ∀ y : ℝ, y = m * x0 + b) := 
sorry

end x_intercept_of_line_l1131_113164


namespace otimes_square_neq_l1131_113199

noncomputable def otimes (a b : ℝ) : ℝ :=
  if a > b then a else b

theorem otimes_square_neq (a b : ℝ) (h : a ≠ b) : (otimes a b) ^ 2 ≠ otimes (a ^ 2) (b ^ 2) := by
  sorry

end otimes_square_neq_l1131_113199


namespace river_width_l1131_113123

variable (depth : ℝ) (flow_rate_kmph : ℝ) (volume_per_minute : ℝ)

-- Define the given conditions:
def depth_of_river : ℝ := 4
def flow_rate : ℝ := 4
def volume_per_minute_water : ℝ := 10666.666666666666

-- The proposition to prove:
theorem river_width :
  let flow_rate_m_per_min := (flow_rate * 1000) / 60
  let width := volume_per_minute / (flow_rate_m_per_min * depth)
  width = 40 :=
by
  sorry

end river_width_l1131_113123


namespace find_constants_l1131_113151

theorem find_constants (A B C : ℚ) :
  (∀ x : ℚ, x ≠ 4 ∧ x ≠ 2 →
    (3 * x + 7) / ((x - 4) * (x - 2)^2) = A / (x - 4) + B / (x - 2) + C / (x - 2)^2) →
  A = 19 / 4 ∧ B = -19 / 4 ∧ C = -13 / 2 :=
by
  sorry

end find_constants_l1131_113151


namespace commute_days_l1131_113154

theorem commute_days (a b d e x : ℕ) 
  (h1 : b + e = 12)
  (h2 : a + d = 20)
  (h3 : a + b = 15)
  (h4 : x = a + b + d + e) :
  x = 32 :=
by {
  sorry
}

end commute_days_l1131_113154


namespace pencils_purchased_l1131_113193

theorem pencils_purchased 
  (total_cost : ℝ)
  (num_pens : ℕ)
  (price_per_pen : ℝ)
  (price_per_pencil : ℝ)
  (total_cost_condition : total_cost = 510)
  (num_pens_condition : num_pens = 30)
  (price_per_pen_condition : price_per_pen = 12)
  (price_per_pencil_condition : price_per_pencil = 2) :
  num_pens * price_per_pen + sorry = total_cost →
  150 / price_per_pencil = 75 :=
by
  sorry

end pencils_purchased_l1131_113193


namespace cone_height_l1131_113143

theorem cone_height (r l h : ℝ) (h_r : r = 1) (h_l : l = 4) : h = Real.sqrt 15 :=
by
  -- proof steps would go here
  sorry

end cone_height_l1131_113143


namespace mutually_exclusive_shots_proof_l1131_113182

/-- Definition of a mutually exclusive event to the event "at most one shot is successful". -/
def mutual_exclusive_at_most_one_shot_successful (both_shots_successful at_most_one_shot_successful : Prop) : Prop :=
  (at_most_one_shot_successful ↔ ¬both_shots_successful)

variable (both_shots_successful : Prop)
variable (at_most_one_shot_successful : Prop)

/-- Given two basketball shots, prove that "both shots are successful" is a mutually exclusive event to "at most one shot is successful". -/
theorem mutually_exclusive_shots_proof : mutual_exclusive_at_most_one_shot_successful both_shots_successful at_most_one_shot_successful :=
  sorry

end mutually_exclusive_shots_proof_l1131_113182


namespace peyton_total_yards_l1131_113120

def distance_on_Saturday (throws: Nat) (yards_per_throw: Nat) : Nat :=
  throws * yards_per_throw

def distance_on_Sunday (throws: Nat) (yards_per_throw: Nat) : Nat :=
  throws * yards_per_throw

def total_distance (distance_Saturday: Nat) (distance_Sunday: Nat) : Nat :=
  distance_Saturday + distance_Sunday

theorem peyton_total_yards :
  let throws_Saturday := 20
  let yards_per_throw_Saturday := 20
  let throws_Sunday := 30
  let yards_per_throw_Sunday := 40
  distance_on_Saturday throws_Saturday yards_per_throw_Saturday +
  distance_on_Sunday throws_Sunday yards_per_throw_Sunday = 1600 :=
by
  sorry

end peyton_total_yards_l1131_113120


namespace total_cement_used_l1131_113185

def cement_used_lexi : ℝ := 10
def cement_used_tess : ℝ := 5.1

theorem total_cement_used : cement_used_lexi + cement_used_tess = 15.1 :=
by sorry

end total_cement_used_l1131_113185


namespace decimal_to_base8_conversion_l1131_113101

theorem decimal_to_base8_conversion : (512 : ℕ) = 8^3 :=
by
  sorry

end decimal_to_base8_conversion_l1131_113101


namespace moles_CO2_required_l1131_113152

theorem moles_CO2_required
  (moles_MgO : ℕ) 
  (moles_MgCO3 : ℕ) 
  (balanced_equation : ∀ (MgO CO2 MgCO3 : ℕ), MgO + CO2 = MgCO3) 
  (reaction_produces : moles_MgO = 3 ∧ moles_MgCO3 = 3) :
  3 = 3 :=
by
  sorry

end moles_CO2_required_l1131_113152


namespace base_conversion_subtraction_l1131_113149

def base6_to_nat (d0 d1 d2 d3 d4 : ℕ) : ℕ :=
  d4 * 6^4 + d3 * 6^3 + d2 * 6^2 + d1 * 6^1 + d0 * 6^0

def base7_to_nat (d0 d1 d2 d3 : ℕ) : ℕ :=
  d3 * 7^3 + d2 * 7^2 + d1 * 7^1 + d0 * 7^0

theorem base_conversion_subtraction :
  base6_to_nat 1 2 3 5 4 - base7_to_nat 1 2 3 4 = 4851 := by
  sorry

end base_conversion_subtraction_l1131_113149


namespace sequence_decreasing_l1131_113128

noncomputable def x_n (a b : ℝ) (n : ℕ) : ℝ := 2 ^ n * (b ^ (1 / 2 ^ n) - a ^ (1 / 2 ^ n))

theorem sequence_decreasing (a b : ℝ) (h1 : 1 < a) (h2 : a < b) : ∀ n : ℕ, x_n a b n > x_n a b (n + 1) :=
by
  sorry

end sequence_decreasing_l1131_113128


namespace find_abc_l1131_113147

theorem find_abc (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a ≤ b ∧ b ≤ c) (h5 : a + b + c + a * b + b * c + c * a = a * b * c + 1) :
  (a = 2 ∧ b = 5 ∧ c = 8) ∨ (a = 3 ∧ b = 4 ∧ c = 13) :=
sorry

end find_abc_l1131_113147


namespace Megan_deleted_pictures_l1131_113156

/--
Megan took 15 pictures at the zoo and 18 at the museum. She still has 2 pictures from her vacation.
Prove that Megan deleted 31 pictures.
-/
theorem Megan_deleted_pictures :
  let zoo_pictures := 15
  let museum_pictures := 18
  let remaining_pictures := 2
  let total_pictures := zoo_pictures + museum_pictures
  let deleted_pictures := total_pictures - remaining_pictures
  deleted_pictures = 31 :=
by
  sorry

end Megan_deleted_pictures_l1131_113156


namespace probability_of_even_sum_is_two_thirds_l1131_113129

def first_twelve_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

noncomputable def choose_4_without_2 : ℕ := (Nat.factorial 11) / ((Nat.factorial 4) * (Nat.factorial 7))

noncomputable def choose_4_from_12 : ℕ := (Nat.factorial 12) / ((Nat.factorial 4) * (Nat.factorial 8))

noncomputable def probability_even_sum : ℚ := (choose_4_without_2 : ℚ) / (choose_4_from_12 : ℚ)

theorem probability_of_even_sum_is_two_thirds :
  probability_even_sum = (2 / 3 : ℚ) :=
sorry

end probability_of_even_sum_is_two_thirds_l1131_113129


namespace triangle_construction_feasible_l1131_113167

theorem triangle_construction_feasible (a b s : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (a - b) / 2 < s) (h4 : s < (a + b) / 2) :
  ∃ c, (a + b > c ∧ b + c > a ∧ c + a > b) :=
sorry

end triangle_construction_feasible_l1131_113167


namespace range_of_a_for_intersections_l1131_113180

theorem range_of_a_for_intersections (a : ℝ) : 
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    (x₁^3 - 3 * x₁ = a) ∧ (x₂^3 - 3 * x₂ = a) ∧ (x₃^3 - 3 * x₃ = a)) ↔ 
  (-2 < a ∧ a < 2) :=
by
  sorry

end range_of_a_for_intersections_l1131_113180


namespace election_votes_and_deposit_l1131_113157

theorem election_votes_and_deposit (V : ℕ) (A B C D E : ℕ) (hA : A = 40 * V / 100) 
  (hB : B = 28 * V / 100) (hC : C = 20 * V / 100) (hDE : D + E = 12 * V / 100)
  (win_margin : A - B = 500) :
  V = 4167 ∧ (15 * V / 100 ≤ A) ∧ (15 * V / 100 ≤ B) ∧ (15 * V / 100 ≤ C) ∧ 
  ¬ (15 * V / 100 ≤ D) ∧ ¬ (15 * V / 100 ≤ E) :=
by 
  sorry

end election_votes_and_deposit_l1131_113157


namespace value_of_f_sin_20_l1131_113118

theorem value_of_f_sin_20 (f : ℝ → ℝ) (h : ∀ x, f (Real.cos x) = Real.sin (3 * x)) :
  f (Real.sin (20 * Real.pi / 180)) = -1 / 2 :=
by sorry

end value_of_f_sin_20_l1131_113118


namespace num_diagonals_increase_by_n_l1131_113106

-- Definitions of the conditions
def num_diagonals (n : ℕ) : ℕ := sorry  -- Consider f(n) to be a function that calculates diagonals for n-sided polygon

-- Lean 4 proof problem statement
theorem num_diagonals_increase_by_n (n : ℕ) :
  num_diagonals (n + 1) = num_diagonals n + n :=
sorry

end num_diagonals_increase_by_n_l1131_113106


namespace no_three_perfect_squares_l1131_113103

theorem no_three_perfect_squares (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ¬(∃ k₁ k₂ k₃ : ℕ, k₁^2 = a^2 + b + c ∧ k₂^2 = b^2 + c + a ∧ k₃^2 = c^2 + a + b) :=
sorry

end no_three_perfect_squares_l1131_113103


namespace proof_f_f_pi_div_12_l1131_113124

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then 4 * x^2 - 1 else (Real.sin x)^2 - (Real.cos x)^2

theorem proof_f_f_pi_div_12 : f (f (Real.pi / 12)) = 2 := by
  sorry

end proof_f_f_pi_div_12_l1131_113124


namespace maximize_sqrt_expression_l1131_113158

theorem maximize_sqrt_expression :
  let a := Real.sqrt 8
  let b := Real.sqrt 2
  (a + b) > max (max (a - b) (a * b)) (a / b) := by
  sorry

end maximize_sqrt_expression_l1131_113158


namespace larger_integer_value_l1131_113111

-- Define the conditions as Lean definitions
def quotient_condition (a b : ℕ) : Prop := a / b = 5 / 2
def product_condition (a b : ℕ) : Prop := a * b = 160
def larger_integer (a b : ℕ) : ℕ := if a > b then a else b

-- State the theorem with conditions and expected outcome
theorem larger_integer_value (a b : ℕ) (h1 : quotient_condition a b) (h2 : product_condition a b) :
  larger_integer a b = 20 :=
sorry -- Proof to be provided

end larger_integer_value_l1131_113111


namespace smallest_positive_angle_l1131_113114

open Real

theorem smallest_positive_angle :
  ∃ x : ℝ, x > 0 ∧ x < 90 ∧ tan (4 * x * degree) = (cos (x * degree) - sin (x * degree)) / (cos (x * degree) + sin (x * degree)) ∧ x = 9 :=
sorry

end smallest_positive_angle_l1131_113114


namespace solve_equations_l1131_113134

theorem solve_equations :
  (∃ x1 x2 : ℝ, x1 = 2 + Real.sqrt 5 ∧ x2 = 2 - Real.sqrt 5 ∧ (x1^2 - 4 * x1 - 1 = 0) ∧ (x2^2 - 4 * x2 - 1 = 0)) ∧
  (∃ y1 y2 : ℝ, y1 = -4 ∧ y2 = 1 ∧ ((y1 + 4)^2 = 5 * (y1 + 4)) ∧ ((y2 + 4)^2 = 5 * (y2 + 4))) :=
by
  sorry

end solve_equations_l1131_113134


namespace n_divides_2n_plus_1_implies_multiple_of_3_l1131_113170

theorem n_divides_2n_plus_1_implies_multiple_of_3 {n : ℕ} (h₁ : n ≥ 2) (h₂ : n ∣ (2^n + 1)) : 3 ∣ n :=
sorry

end n_divides_2n_plus_1_implies_multiple_of_3_l1131_113170


namespace least_multiple_of_29_gt_500_l1131_113136

theorem least_multiple_of_29_gt_500 : ∃ n : ℕ, n > 0 ∧ 29 * n > 500 ∧ 29 * n = 522 :=
by
  use 18
  sorry

end least_multiple_of_29_gt_500_l1131_113136


namespace total_pastries_sum_l1131_113168

   theorem total_pastries_sum :
     let lola_mini_cupcakes := 13
     let lola_pop_tarts := 10
     let lola_blueberry_pies := 8
     let lola_chocolate_eclairs := 6

     let lulu_mini_cupcakes := 16
     let lulu_pop_tarts := 12
     let lulu_blueberry_pies := 14
     let lulu_chocolate_eclairs := 9

     let lila_mini_cupcakes := 22
     let lila_pop_tarts := 15
     let lila_blueberry_pies := 10
     let lila_chocolate_eclairs := 12

     lola_mini_cupcakes + lulu_mini_cupcakes + lila_mini_cupcakes +
     lola_pop_tarts + lulu_pop_tarts + lila_pop_tarts +
     lola_blueberry_pies + lulu_blueberry_pies + lila_blueberry_pies +
     lola_chocolate_eclairs + lulu_chocolate_eclairs + lila_chocolate_eclairs = 147 :=
   by
     sorry
   
end total_pastries_sum_l1131_113168


namespace total_days_2003_to_2006_l1131_113179

theorem total_days_2003_to_2006 : 
  let days_2003 := 365
  let days_2004 := 366
  let days_2005 := 365
  let days_2006 := 365
  days_2003 + days_2004 + days_2005 + days_2006 = 1461 :=
by {
  sorry
}

end total_days_2003_to_2006_l1131_113179


namespace whisker_ratio_l1131_113176

theorem whisker_ratio 
  (p : ℕ) (c : ℕ) (h1 : p = 14) (h2 : c = 22) (s := c + 6) :
  s / p = 2 := 
by
  sorry

end whisker_ratio_l1131_113176


namespace find_max_side_length_l1131_113109

noncomputable def max_side_length (a b c : ℕ) : ℕ :=
  if a + b + c = 24 ∧ a < b ∧ b < c ∧ a + b > c ∧ (a ≠ b ∧ b ≠ c ∧ a ≠ c) then c else 0

theorem find_max_side_length
  (a b c : ℕ)
  (h₁ : a ≠ b)
  (h₂ : b ≠ c)
  (h₃ : a ≠ c)
  (h₄ : a + b + c = 24)
  (h₅ : a < b)
  (h₆ : b < c)
  (h₇ : a + b > c) :
  max_side_length a b c = 10 :=
sorry

end find_max_side_length_l1131_113109


namespace polynomial_ascending_l1131_113190

theorem polynomial_ascending (x : ℝ) :
  (x^2 - 2 - 5*x^4 + 3*x^3) = (-2 + x^2 + 3*x^3 - 5*x^4) :=
by sorry

end polynomial_ascending_l1131_113190


namespace price_after_reductions_l1131_113140

theorem price_after_reductions (P : ℝ) : ((P * 0.85) * 0.90) = P * 0.765 :=
by sorry

end price_after_reductions_l1131_113140


namespace melissa_total_repair_time_l1131_113183

def time_flat_shoes := 3 + 8 + 9
def time_sandals :=  4 + 5
def time_high_heels := 6 + 12 + 10

def first_session_flat_shoes := 6 * time_flat_shoes
def first_session_sandals := 4 * time_sandals
def first_session_high_heels := 3 * time_high_heels

def second_session_flat_shoes := 4 * time_flat_shoes
def second_session_sandals := 7 * time_sandals
def second_session_high_heels := 5 * time_high_heels

def total_first_session := first_session_flat_shoes + first_session_sandals + first_session_high_heels
def total_second_session := second_session_flat_shoes + second_session_sandals + second_session_high_heels

def break_time := 15

def total_repair_time := total_first_session + total_second_session
def total_time_including_break := total_repair_time + break_time

theorem melissa_total_repair_time : total_time_including_break = 538 := by
  sorry

end melissa_total_repair_time_l1131_113183


namespace available_seats_l1131_113187

/-- Two-fifths of the seats in an auditorium that holds 500 people are currently taken. --/
def seats_taken : ℕ := (2 * 500) / 5

/-- One-tenth of the seats in an auditorium that holds 500 people are broken. --/
def seats_broken : ℕ := 500 / 10

/-- Total seats in the auditorium --/
def total_seats := 500

/-- There are 500 total seats in an auditorium. Two-fifths of the seats are taken and 
one-tenth are broken. Prove that the number of seats still available is 250. --/
theorem available_seats : (total_seats - seats_taken - seats_broken) = 250 :=
by 
  sorry

end available_seats_l1131_113187


namespace no_perfect_squares_l1131_113181

theorem no_perfect_squares (x y : ℕ) : ¬ (∃ a b : ℕ, x^2 + y = a^2 ∧ x + y^2 = b^2) :=
sorry

end no_perfect_squares_l1131_113181


namespace total_bike_cost_l1131_113131

def marions_bike_cost : ℕ := 356
def stephanies_bike_cost : ℕ := 2 * marions_bike_cost

theorem total_bike_cost : marions_bike_cost + stephanies_bike_cost = 1068 := by
  sorry

end total_bike_cost_l1131_113131


namespace no_tiling_possible_with_given_dimensions_l1131_113119

theorem no_tiling_possible_with_given_dimensions :
  ¬(∃ (n : ℕ), n * (2 * 2 * 1) = (3 * 4 * 5) ∧ 
   (∀ i j k : ℕ, i * 2 = 3 ∨ i * 2 = 4 ∨ i * 2 = 5) ∧
   (∀ i j k : ℕ, j * 2 = 3 ∨ j * 2 = 4 ∨ j * 2 = 5) ∧
   (∀ i j k : ℕ, k * 1 = 3 ∨ k * 1 = 4 ∨ k * 1 = 5)) :=
sorry

end no_tiling_possible_with_given_dimensions_l1131_113119


namespace Michelle_bought_14_chocolate_bars_l1131_113162

-- Definitions for conditions
def sugar_per_chocolate_bar : ℕ := 10
def sugar_in_lollipop : ℕ := 37
def total_sugar_in_candy : ℕ := 177

-- Theorem to prove
theorem Michelle_bought_14_chocolate_bars :
  (total_sugar_in_candy - sugar_in_lollipop) / sugar_per_chocolate_bar = 14 :=
by
  -- Proof steps will go here, but are omitted as per the requirements.
  sorry

end Michelle_bought_14_chocolate_bars_l1131_113162


namespace speed_of_train_in_km_per_hr_l1131_113171

-- Definitions for the condition
def length_of_train : ℝ := 180 -- in meters
def time_to_cross_pole : ℝ := 9 -- in seconds

-- Conversion factor
def meters_per_second_to_kilometers_per_hour (speed : ℝ) := speed * 3.6

-- Proof statement
theorem speed_of_train_in_km_per_hr : 
  meters_per_second_to_kilometers_per_hour (length_of_train / time_to_cross_pole) = 72 := 
by
  sorry

end speed_of_train_in_km_per_hr_l1131_113171


namespace escalator_time_l1131_113115

theorem escalator_time (speed_escalator: ℝ) (length_escalator: ℝ) (speed_person: ℝ) (combined_speed: ℝ)
  (h1: speed_escalator = 20) (h2: length_escalator = 250) (h3: speed_person = 5) (h4: combined_speed = speed_escalator + speed_person) :
  length_escalator / combined_speed = 10 := by
  sorry

end escalator_time_l1131_113115


namespace joshua_crates_l1131_113108

def joshua_packs (b : ℕ) (not_packed : ℕ) (b_per_crate : ℕ) : ℕ :=
  (b - not_packed) / b_per_crate

theorem joshua_crates : joshua_packs 130 10 12 = 10 := by
  sorry

end joshua_crates_l1131_113108


namespace least_positive_integer_is_4619_l1131_113160

noncomputable def least_positive_integer (N : ℕ) : Prop :=
  N % 4 = 3 ∧
  N % 5 = 4 ∧
  N % 6 = 5 ∧
  N % 7 = 6 ∧
  N % 11 = 10 ∧
  ∀ M : ℕ, (M % 4 = 3 ∧ M % 5 = 4 ∧ M % 6 = 5 ∧ M % 7 = 6 ∧ M % 11 = 10) → N ≤ M

theorem least_positive_integer_is_4619 : least_positive_integer 4619 :=
  sorry

end least_positive_integer_is_4619_l1131_113160


namespace part_a_l1131_113141

theorem part_a (a b : ℕ) (h : (3 * a + b) % 10 = (3 * b + a) % 10) : 
  (a % 10 = b % 10) := 
sorry

end part_a_l1131_113141


namespace find_abcd_l1131_113186

theorem find_abcd 
    (a b c d : ℕ) 
    (h : 5^a + 6^b + 7^c + 11^d = 1999) : 
    (a, b, c, d) = (4, 2, 1, 3) :=
by
    sorry

end find_abcd_l1131_113186


namespace train_distance_l1131_113178

def fuel_efficiency := 5 / 2 
def coal_remaining := 160
def expected_distance := 400

theorem train_distance : fuel_efficiency * coal_remaining = expected_distance := 
by
  sorry

end train_distance_l1131_113178


namespace distribute_balls_into_boxes_l1131_113117

/--
Given 6 distinguishable balls and 3 distinguishable boxes, 
there are 3^6 = 729 ways to distribute the balls into the boxes.
-/
theorem distribute_balls_into_boxes : (3 : ℕ)^6 = 729 := 
by
  sorry

end distribute_balls_into_boxes_l1131_113117


namespace polynomial_expansion_l1131_113177

theorem polynomial_expansion (x : ℝ) : 
  (1 - x^3) * (1 + x^4 - x^5) = 1 - x^3 + x^4 - x^5 - x^7 + x^8 :=
by
  sorry

end polynomial_expansion_l1131_113177


namespace escalator_rate_l1131_113138

theorem escalator_rate
  (length_escalator : ℕ) 
  (person_speed : ℕ) 
  (time_taken : ℕ) 
  (total_length : length_escalator = 112) 
  (person_speed_rate : person_speed = 4)
  (time_taken_rate : time_taken = 8) :
  ∃ v : ℕ, (person_speed + v) * time_taken = length_escalator ∧ v = 10 :=
by
  sorry

end escalator_rate_l1131_113138


namespace problem_1_problem_2_l1131_113195

noncomputable def f (a b x : ℝ) := |x + a| + |2 * x - b|

theorem problem_1 (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b)
(h_min : ∀ x, f a b x ≥ 1 ∧ (∃ x₀, f a b x₀ = 1)) :
2 * a + b = 2 :=
sorry

theorem problem_2 (a b t : ℝ) (h_a : 0 < a) (h_b : 0 < b) 
(h_tab : ∀ t > 0, a + 2 * b ≥ t * a * b)
(h_eq : 2 * a + b = 2) :
t ≤ 9 / 2 :=
sorry

end problem_1_problem_2_l1131_113195


namespace eight_pow_2012_mod_10_l1131_113132

theorem eight_pow_2012_mod_10 : (8 ^ 2012) % 10 = 2 :=
by {
  sorry
}

end eight_pow_2012_mod_10_l1131_113132


namespace lukas_games_played_l1131_113142

-- Define the given conditions
def average_points_per_game : ℕ := 12
def total_points_scored : ℕ := 60

-- Define Lukas' number of games
def number_of_games (total_points : ℕ) (average_points : ℕ) : ℕ :=
  total_points / average_points

-- Theorem and statement to prove
theorem lukas_games_played :
  number_of_games total_points_scored average_points_per_game = 5 :=
by
  sorry

end lukas_games_played_l1131_113142


namespace find_k_l1131_113174

-- Definitions for the conditions and the main theorem.
variables {x y k : ℝ}

-- The first equation of the system
def eq1 (x y k : ℝ) : Prop := 2 * x + 5 * y = k

-- The second equation of the system
def eq2 (x y : ℝ) : Prop := x - 4 * y = 15

-- Condition that x and y are opposites
def are_opposites (x y : ℝ) : Prop := x + y = 0

-- The theorem to prove
theorem find_k (hk : ∃ (x y : ℝ), eq1 x y k ∧ eq2 x y ∧ are_opposites x y) : k = -9 :=
sorry

end find_k_l1131_113174


namespace negation_of_exists_log3_nonnegative_l1131_113125

variable (x : ℝ)

theorem negation_of_exists_log3_nonnegative :
  (¬ (∃ x : ℝ, Real.logb 3 x ≥ 0)) ↔ (∀ x : ℝ, Real.logb 3 x < 0) :=
by
  sorry

end negation_of_exists_log3_nonnegative_l1131_113125


namespace point_on_coordinate_axes_l1131_113188

theorem point_on_coordinate_axes {x y : ℝ} 
  (h : x * y = 0) : (x = 0 ∨ y = 0) :=
by {
  sorry
}

end point_on_coordinate_axes_l1131_113188


namespace min_sum_of_m_n_l1131_113173

theorem min_sum_of_m_n (m n : ℕ) (h1 : m ≥ 1) (h2 : n ≥ 3) (h3 : 8 ∣ (180 * m * n - 360 * m)) : m + n = 5 :=
sorry

end min_sum_of_m_n_l1131_113173


namespace max_D_n_l1131_113127

-- Define the properties for each block
structure Block where
  shape : ℕ -- 1 for Square, 2 for Circular
  color : ℕ -- 1 for Red, 2 for Yellow
  city  : ℕ -- 1 for Nanchang, 2 for Beijing

-- The 8 blocks
def blocks : List Block := [
  { shape := 1, color := 1, city := 1 },
  { shape := 2, color := 1, city := 1 },
  { shape := 2, color := 2, city := 1 },
  { shape := 1, color := 2, city := 1 },
  { shape := 1, color := 1, city := 2 },
  { shape := 2, color := 1, city := 2 },
  { shape := 2, color := 2, city := 2 },
  { shape := 1, color := 2, city := 2 }
]

-- Define D_n counting function (to be implemented)
noncomputable def D_n (n : ℕ) : ℕ := sorry

-- Define the required proof
theorem max_D_n : 2 ≤ n → n ≤ 8 → ∃ k : ℕ, 2 ≤ k ∧ k ≤ 8 ∧ D_n k = 240 := sorry

end max_D_n_l1131_113127


namespace cat_count_after_10_days_l1131_113112

def initial_cats := 60 -- Shelter had 60 cats before the intake
def intake_cats := 30 -- Shelter took in 30 cats
def total_cats_at_start := initial_cats + intake_cats -- 90 cats after intake

def even_days_adoptions := 5 -- Cats adopted on even days
def odd_days_adoptions := 15 -- Cats adopted on odd days
def total_adoptions := even_days_adoptions + odd_days_adoptions -- Total adoptions over 10 days

def day4_births := 10 -- Kittens born on day 4
def day7_births := 5 -- Kittens born on day 7
def total_births := day4_births + day7_births -- Total births over 10 days

def claimed_pets := 2 -- Number of mothers claimed as missing pets

def final_cat_count := total_cats_at_start - total_adoptions + total_births - claimed_pets -- Final cat count

theorem cat_count_after_10_days : final_cat_count = 83 := by
  sorry

end cat_count_after_10_days_l1131_113112


namespace find_f_2_solve_inequality_l1131_113133

noncomputable def f : ℝ → ℝ :=
  sorry -- definition of f cannot be constructed without further info

axiom f_decreasing : ∀ x y : ℝ, 0 < x → 0 < y → (x ≤ y → f x ≥ f y)

axiom f_additive : ∀ x y : ℝ, 0 < x → 0 < y → f (x + y) = f x + f y - 1

axiom f_4 : f 4 = 5

theorem find_f_2 : f 2 = 3 :=
  sorry

theorem solve_inequality (m : ℝ) (h : f (m - 2) ≤ 3) : m ≥ 4 :=
  sorry

end find_f_2_solve_inequality_l1131_113133


namespace function_value_l1131_113197

theorem function_value (f : ℝ → ℝ) (h : ∀ x, x + 17 = 60 * f x) : f 3 = 1 / 3 :=
by
  sorry

end function_value_l1131_113197


namespace curve_symmetric_about_y_eq_x_l1131_113113

theorem curve_symmetric_about_y_eq_x (x y : ℝ) (h : x * y * (x + y) = 1) :
  (y * x * (y + x) = 1) :=
by
  sorry

end curve_symmetric_about_y_eq_x_l1131_113113


namespace find_number_l1131_113139

theorem find_number (x : ℕ) (h : x / 5 = 75 + x / 6) : x = 2250 := 
by
  sorry

end find_number_l1131_113139


namespace find_fake_coin_l1131_113191

theorem find_fake_coin (k : ℕ) :
  ∃ (weighings : ℕ), (weighings ≤ 3 * k + 1) :=
sorry

end find_fake_coin_l1131_113191


namespace max_three_digit_sum_l1131_113159

theorem max_three_digit_sum : ∃ (A B C : ℕ), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ (0 ≤ A ∧ A < 10) ∧ (0 ≤ B ∧ B < 10) ∧ (0 ≤ C ∧ C < 10) ∧ (111 * A + 10 * C + 2 * B = 976) := sorry

end max_three_digit_sum_l1131_113159


namespace tan_alpha_minus_2beta_l1131_113104

theorem tan_alpha_minus_2beta (α β : Real) 
  (h1 : Real.tan (α - β) = 2 / 5)
  (h2 : Real.tan β = 1 / 2) :
  Real.tan (α - 2 * β) = -1 / 12 := 
by 
  sorry

end tan_alpha_minus_2beta_l1131_113104


namespace solve_equation_l1131_113150

theorem solve_equation :
  (3 * x - 6 = abs (-21 + 8 - 3)) → x = 22 / 3 :=
by
  intro h
  sorry

end solve_equation_l1131_113150


namespace find_f_of_2_l1131_113100

noncomputable def f (x : ℕ) : ℕ := x^x + 2*x + 2

theorem find_f_of_2 : f 1 + 1 = 5 := 
by 
  sorry

end find_f_of_2_l1131_113100


namespace bill_earnings_per_ounce_l1131_113148

-- Given conditions
def ounces_sold : Nat := 8
def fine : Nat := 50
def money_left : Nat := 22
def total_money_earned : Nat := money_left + fine -- $72

-- The amount earned for every ounce of fool's gold
def price_per_ounce : Nat := total_money_earned / ounces_sold -- 72 / 8

-- The proof statement
theorem bill_earnings_per_ounce (h: price_per_ounce = 9) : True :=
by
  trivial

end bill_earnings_per_ounce_l1131_113148


namespace proportion_solution_l1131_113198

-- Define the given proportion condition as a hypothesis
variable (x : ℝ)

-- The definition is derived directly from the given problem
def proportion_condition : Prop := x / 5 = 1.2 / 8

-- State the theorem using the given proportion condition to prove x = 0.75
theorem proportion_solution (h : proportion_condition x) : x = 0.75 :=
  by
    sorry

end proportion_solution_l1131_113198


namespace quiz_points_minus_homework_points_l1131_113122

theorem quiz_points_minus_homework_points
  (total_points : ℕ)
  (quiz_points : ℕ)
  (test_points : ℕ)
  (homework_points : ℕ)
  (h1 : total_points = 265)
  (h2 : test_points = 4 * quiz_points)
  (h3 : homework_points = 40)
  (h4 : homework_points + quiz_points + test_points = total_points) :
  quiz_points - homework_points = 5 :=
by sorry

end quiz_points_minus_homework_points_l1131_113122


namespace worker_idle_days_l1131_113116

theorem worker_idle_days (W I : ℕ) 
  (h1 : 20 * W - 3 * I = 280)
  (h2 : W + I = 60) : 
  I = 40 :=
sorry

end worker_idle_days_l1131_113116


namespace quotient_korean_english_l1131_113161

theorem quotient_korean_english (K M E : ℝ) (h1 : K / M = 1.2) (h2 : M / E = 5 / 6) : K / E = 1 :=
sorry

end quotient_korean_english_l1131_113161


namespace solve_for_k_l1131_113192

theorem solve_for_k (t k : ℝ) (h1 : t = 5 / 9 * (k - 32)) (h2 : t = 105) : k = 221 :=
by
  sorry

end solve_for_k_l1131_113192


namespace min_value_expression_l1131_113144

open Classical

theorem min_value_expression (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (h : 1/x + 1/y = 1) :
  ∃ (m : ℝ), m = 25 ∧ ∀ x y : ℝ, 0 < x → 0 < y → 1/x + 1/y = 1 → (4*x/(x - 1) + 9*y/(y - 1)) ≥ m :=
by 
  sorry

end min_value_expression_l1131_113144


namespace mrs_hilt_read_chapters_l1131_113189

-- Define the problem conditions
def books : ℕ := 4
def chapters_per_book : ℕ := 17

-- State the proof problem
theorem mrs_hilt_read_chapters : (books * chapters_per_book) = 68 := 
by
  sorry

end mrs_hilt_read_chapters_l1131_113189


namespace day_of_week_after_2_power_50_days_l1131_113184

-- Conditions:
def today_is_monday : ℕ := 1  -- Monday corresponds to 1

def days_later (n : ℕ) : ℕ := (today_is_monday + n) % 7

theorem day_of_week_after_2_power_50_days :
  days_later (2^50) = 6 :=  -- Saturday corresponds to 6 (0 is Sunday)
by {
  -- Proof steps are skipped
  sorry
}

end day_of_week_after_2_power_50_days_l1131_113184


namespace new_total_energy_l1131_113105

-- Define the problem conditions
def identical_point_charges_positioned_at_vertices_of_equilateral_triangle (charges : ℕ) (initial_energy : ℝ) : Prop :=
  charges = 3 ∧ initial_energy = 18

def charge_moved_one_third_along_side (move_fraction : ℝ) : Prop :=
  move_fraction = 1/3

-- Define the theorem and proof goal
theorem new_total_energy (charges : ℕ) (initial_energy : ℝ) (move_fraction : ℝ) :
  identical_point_charges_positioned_at_vertices_of_equilateral_triangle charges initial_energy →
  charge_moved_one_third_along_side move_fraction →
  ∃ (new_energy : ℝ), new_energy = 21 :=
by
  intros h_triangle h_move
  sorry

end new_total_energy_l1131_113105


namespace calculation_result_l1131_113169

theorem calculation_result : (1000 * 7 / 10 * 17 * 5^2 = 297500) :=
by sorry

end calculation_result_l1131_113169


namespace initial_investment_l1131_113146

theorem initial_investment (A P : ℝ) (r : ℝ) (n t : ℕ) 
  (hA : A = 16537.5)
  (hr : r = 0.10)
  (hn : n = 2)
  (ht : t = 1)
  (hA_calc : A = P * (1 + r / n) ^ (n * t)) :
  P = 15000 :=
by {
  sorry
}

end initial_investment_l1131_113146


namespace factoring_correct_l1131_113155

-- Definitions corresponding to the problem conditions
def optionA (a : ℝ) : Prop := a^2 - 5*a - 6 = (a - 6) * (a + 1)
def optionB (a x b c : ℝ) : Prop := a*x + b*x + c = (a + b)*x + c
def optionC (a b : ℝ) : Prop := (a + b)^2 = a^2 + 2*a*b + b^2
def optionD (a b : ℝ) : Prop := (a + b)*(a - b) = a^2 - b^2

-- The main theorem that proves option A is the correct answer
theorem factoring_correct : optionA a := by
  sorry

end factoring_correct_l1131_113155


namespace probability_of_three_blue_marbles_l1131_113196

theorem probability_of_three_blue_marbles
  (red_marbles : ℕ) (blue_marbles : ℕ) (yellow_marbles : ℕ) (total_marbles : ℕ)
  (draws : ℕ) 
  (prob : ℚ) :
  red_marbles = 3 →
  blue_marbles = 4 →
  yellow_marbles = 13 →
  total_marbles = 20 →
  draws = 3 →
  prob = ((4 / 20) * (3 / 19) * (1 / 9)) →
  prob = 1 / 285 :=
by
  intros; 
  sorry

end probability_of_three_blue_marbles_l1131_113196


namespace part1_f_ge_0_part2_number_of_zeros_part2_number_of_zeros_case2_l1131_113121

-- Part 1: Prove f(x) ≥ 0 when a = 1
noncomputable def f (x : ℝ) : ℝ := Real.exp x - x - 1

theorem part1_f_ge_0 : ∀ x : ℝ, f x ≥ 0 := sorry

-- Part 2: Discuss the number of zeros of the function f(x)
noncomputable def g (a x : ℝ) : ℝ := Real.exp x - a * x - 1

theorem part2_number_of_zeros (a : ℝ) : 
  (a ≤ 0 ∨ a = 1) → ∃! x : ℝ, g a x = 0 := sorry

theorem part2_number_of_zeros_case2 (a : ℝ) : 
  (0 < a ∧ a < 1) ∨ (a > 1) → ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ g a x1 = 0 ∧ g a x2 = 0 := sorry

end part1_f_ge_0_part2_number_of_zeros_part2_number_of_zeros_case2_l1131_113121


namespace expression_evaluation_l1131_113153

theorem expression_evaluation : 5^3 - 3 * 5^2 + 3 * 5 - 1 = 64 :=
by
  sorry

end expression_evaluation_l1131_113153


namespace handshake_problem_l1131_113102

theorem handshake_problem (n : ℕ) (H : (n * (n - 1)) / 2 = 28) : n = 8 := 
sorry

end handshake_problem_l1131_113102


namespace find_m_l1131_113194

noncomputable def A (m : ℝ) : Set ℝ := {1, 3, 2 * m + 3}
noncomputable def B (m : ℝ) : Set ℝ := {3, m^2}

theorem find_m (m : ℝ) : B m ⊆ A m ↔ m = 1 ∨ m = 3 :=
by
  sorry

end find_m_l1131_113194


namespace polynomial_horner_value_l1131_113110

def f (x : ℤ) : ℤ :=
  7 * x^7 + 6 * x^6 + 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x

def horner (x : ℤ) : ℤ :=
  ((((((7 * x + 6) * x + 5) * x + 4) * x + 3) * x + 2) * x + 1)

theorem polynomial_horner_value :
  horner 3 = 262 := by
  sorry

end polynomial_horner_value_l1131_113110


namespace solve_quadratic_equation_l1131_113175

theorem solve_quadratic_equation (x : ℝ) :
  2 * x * (x + 1) = x + 1 ↔ (x = -1 ∨ x = 1 / 2) :=
by
  sorry

end solve_quadratic_equation_l1131_113175


namespace find_number_l1131_113137

theorem find_number (x : Real) (h1 : (2 / 5) * 300 = 120) (h2 : 120 - (3 / 5) * x = 45) : x = 125 :=
by
  sorry

end find_number_l1131_113137


namespace system_of_equations_solution_cases_l1131_113166

theorem system_of_equations_solution_cases
  (x y a b : ℝ) :
  (a = b → x + y = 2 * a) ∧
  (a = -b → ¬ (∃ (x y : ℝ), (x / (x - a)) + (y / (y - b)) = 2 ∧ a * x + b * y = 2 * a * b)) :=
by
  sorry

end system_of_equations_solution_cases_l1131_113166
