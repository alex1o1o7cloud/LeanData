import Mathlib

namespace train_length_l1433_143310

-- Define the conditions
def equal_length_trains (L : ℝ) : Prop :=
  ∃ (length : ℝ), length = L

def train_speeds : Prop :=
  ∃ v_fast v_slow : ℝ, v_fast = 46 ∧ v_slow = 36

def pass_time (t : ℝ) : Prop :=
  t = 36

-- The proof problem
theorem train_length (L : ℝ) 
  (h_equal_length : equal_length_trains L) 
  (h_speeds : train_speeds)
  (h_time : pass_time 36) : 
  L = 50 :=
sorry

end train_length_l1433_143310


namespace length_of_wall_l1433_143368

-- Define the dimensions of a brick
def brick_length : ℝ := 40
def brick_width : ℝ := 11.25
def brick_height : ℝ := 6

-- Define the dimensions of the wall
def wall_height : ℝ := 600
def wall_width : ℝ := 22.5

-- Define the required number of bricks
def required_bricks : ℝ := 4000

-- Calculate the volume of a single brick
def volume_brick : ℝ := brick_length * brick_width * brick_height

-- Calculate the volume of the wall
def volume_wall (length : ℝ) : ℝ := length * wall_height * wall_width

-- The theorem to prove
theorem length_of_wall : ∃ (L : ℝ), required_bricks * volume_brick = volume_wall L → L = 800 :=
sorry

end length_of_wall_l1433_143368


namespace distance_symmetric_reflection_l1433_143381

theorem distance_symmetric_reflection (x : ℝ) (y : ℝ) (B : (ℝ × ℝ)) 
  (hB : B = (-1, 4)) (A : (ℝ × ℝ)) (hA : A = (x, -y)) : 
  dist A B = 8 :=
by
  sorry

end distance_symmetric_reflection_l1433_143381


namespace total_volume_of_four_cubes_l1433_143392

theorem total_volume_of_four_cubes (s : ℝ) (h_s : s = 5) : 4 * s^3 = 500 :=
by
  sorry

end total_volume_of_four_cubes_l1433_143392


namespace tan_ratio_of_angles_l1433_143364

theorem tan_ratio_of_angles (a b : ℝ) (h1 : Real.sin (a + b) = 3/4) (h2 : Real.sin (a - b) = 1/2) :
    (Real.tan a / Real.tan b) = 5 := 
by 
  sorry

end tan_ratio_of_angles_l1433_143364


namespace pies_difference_l1433_143340

theorem pies_difference (time : ℕ) (alice_time : ℕ) (bob_time : ℕ) (charlie_time : ℕ)
    (h_time : time = 90) (h_alice : alice_time = 5) (h_bob : bob_time = 6) (h_charlie : charlie_time = 7) :
    (time / alice_time - time / bob_time) + (time / alice_time - time / charlie_time) = 9 := by
  sorry

end pies_difference_l1433_143340


namespace total_surface_area_l1433_143378

theorem total_surface_area (a b c : ℝ) 
  (h1 : a + b + c = 45) 
  (h2 : a^2 + b^2 + c^2 = 625) : 
  2 * (a * b + b * c + c * a) = 1400 :=
sorry

end total_surface_area_l1433_143378


namespace evaluation_at_2_l1433_143328

def f (x : ℚ) : ℚ := (2 * x^2 + 7 * x + 12) / (x^2 + 2 * x + 5)
def g (x : ℚ) : ℚ := x - 2

theorem evaluation_at_2 :
  f (g 2) + g (f 2) = 196 / 65 := by
  sorry

end evaluation_at_2_l1433_143328


namespace Tom_final_balance_l1433_143393

theorem Tom_final_balance :
  let initial_allowance := 12
  let week1_spending := initial_allowance / 3
  let balance_after_week1 := initial_allowance - week1_spending
  let week2_spending := balance_after_week1 / 4
  let balance_after_week2 := balance_after_week1 - week2_spending
  let additional_earning := 5
  let balance_after_earning := balance_after_week2 + additional_earning
  let week3_spending := balance_after_earning / 2
  let balance_after_week3 := balance_after_earning - week3_spending
  let penultimate_day_spending := 3
  let final_balance := balance_after_week3 - penultimate_day_spending
  final_balance = 2.50 :=
by
  sorry

end Tom_final_balance_l1433_143393


namespace compare_sqrts_l1433_143377

theorem compare_sqrts (a b c : ℝ) (h1 : a = 2 * Real.sqrt 7) (h2 : b = 3 * Real.sqrt 5) (h3 : c = 5 * Real.sqrt 2):
  c > b ∧ b > a :=
by
  sorry

end compare_sqrts_l1433_143377


namespace determinant_scaled_l1433_143325

variables (x y z w : ℝ)
variables (det : ℝ)

-- Given condition: determinant of the 2x2 matrix is 7.
axiom det_given : det = x * w - y * z
axiom det_value : det = 7

-- The target to be proven: the determinant of the scaled matrix is 63.
theorem determinant_scaled (x y z w : ℝ) (det : ℝ) (h_det : det = x * w - y * z) (det_value : det = 7) : 
  3 * 3 * (x * w - y * z) = 63 :=
by
  sorry

end determinant_scaled_l1433_143325


namespace escalator_time_l1433_143338

theorem escalator_time
    {d i s : ℝ}
    (h1 : d = 90 * i)
    (h2 : d = 30 * (i + s))
    (h3 : s = 2 * i):
    d / s = 45 := by
  sorry

end escalator_time_l1433_143338


namespace student_tickets_sold_l1433_143346

theorem student_tickets_sold (S NS : ℕ) (h1 : S + NS = 150) (h2 : 5 * S + 8 * NS = 930) : S = 90 :=
by
  sorry

end student_tickets_sold_l1433_143346


namespace parabola_incorrect_statement_B_l1433_143360

theorem parabola_incorrect_statement_B 
  (y₁ y₂ : ℝ → ℝ) 
  (h₁ : ∀ x, y₁ x = 2 * x^2) 
  (h₂ : ∀ x, y₂ x = -2 * x^2) : 
  ¬ (∀ x < 0, y₁ x < y₁ (x + 1)) ∧ (∀ x < 0, y₂ x < y₂ (x + 1)) := 
by 
  sorry

end parabola_incorrect_statement_B_l1433_143360


namespace arrangements_three_balls_four_boxes_l1433_143395

theorem arrangements_three_balls_four_boxes : 
  ∃ (f : Fin 4 → Fin 4), Function.Injective f :=
sorry

end arrangements_three_balls_four_boxes_l1433_143395


namespace roots_squared_sum_l1433_143334

theorem roots_squared_sum (a b : ℝ) (h : a^2 - 8 * a + 8 = 0 ∧ b^2 - 8 * b + 8 = 0) : a^2 + b^2 = 48 := 
sorry

end roots_squared_sum_l1433_143334


namespace real_solution_for_any_y_l1433_143366

theorem real_solution_for_any_y (x : ℝ) :
  (∀ y z : ℝ, x^2 + y^2 + z^2 + 2 * x * y * z = 1 → ∃ z : ℝ,  x^2 + y^2 + z^2 + 2 * x * y * z = 1) ↔ (x = 1 ∨ x = -1) :=
by sorry

end real_solution_for_any_y_l1433_143366


namespace students_in_class_l1433_143330

theorem students_in_class (y : ℕ) (H : 2 * y^2 + 6 * y + 9 = 490) : 
  y + (y + 3) = 31 := by
  sorry

end students_in_class_l1433_143330


namespace has_buried_correct_number_of_bones_l1433_143326

def bones_received_per_month : ℕ := 10
def number_of_months : ℕ := 5
def bones_available : ℕ := 8

def total_bones_received : ℕ := bones_received_per_month * number_of_months
def bones_buried : ℕ := total_bones_received - bones_available

theorem has_buried_correct_number_of_bones : bones_buried = 42 := by
  sorry

end has_buried_correct_number_of_bones_l1433_143326


namespace sequence_x_y_sum_l1433_143353

theorem sequence_x_y_sum (r : ℝ) (x y : ℝ)
  (h₁ : r = 1 / 4)
  (h₂ : x = 256 * r)
  (h₃ : y = x * r) :
  x + y = 80 :=
by
  sorry

end sequence_x_y_sum_l1433_143353


namespace geoff_tuesday_multiple_l1433_143391

variable (monday_spending : ℝ) (tuesday_multiple : ℝ) (total_spending : ℝ)

-- Given conditions
def geoff_conditions (monday_spending tuesday_multiple total_spending : ℝ) : Prop :=
  monday_spending = 60 ∧
  (tuesday_multiple * monday_spending) + (5 * monday_spending) + monday_spending = total_spending ∧
  total_spending = 600

-- Proof goal
theorem geoff_tuesday_multiple (monday_spending tuesday_multiple total_spending : ℝ)
  (h : geoff_conditions monday_spending tuesday_multiple total_spending) : 
  tuesday_multiple = 4 :=
by
  sorry

end geoff_tuesday_multiple_l1433_143391


namespace multiplication_is_247_l1433_143305

theorem multiplication_is_247 (a b : ℕ) (ha : 10 ≤ a ∧ a < 100) (hb : 10 ≤ b ∧ b < 100) (h : a * b = 247) : 
a = 13 ∧ b = 19 :=
by sorry

end multiplication_is_247_l1433_143305


namespace solve_for_a_l1433_143375

def i := Complex.I

theorem solve_for_a (a : ℝ) (h : (2 + i) / (1 + a * i) = i) : a = -2 := 
by 
  sorry

end solve_for_a_l1433_143375


namespace time_per_toy_is_3_l1433_143399

-- Define the conditions
variable (total_toys : ℕ) (total_hours : ℕ)

-- Define the given condition
def given_condition := (total_toys = 50 ∧ total_hours = 150)

-- Define the statement to be proved
theorem time_per_toy_is_3 (h : given_condition total_toys total_hours) :
  total_hours / total_toys = 3 := by
sorry

end time_per_toy_is_3_l1433_143399


namespace probability_at_least_one_prize_proof_l1433_143379

noncomputable def probability_at_least_one_wins_prize
  (total_tickets : ℕ) (prize_tickets : ℕ) (people : ℕ) :
  ℚ :=
1 - ((@Nat.choose (total_tickets - prize_tickets) people) /
      (@Nat.choose total_tickets people))

theorem probability_at_least_one_prize_proof :
  probability_at_least_one_wins_prize 10 3 5 = 11 / 12 :=
by
  sorry

end probability_at_least_one_prize_proof_l1433_143379


namespace binary_computation_l1433_143394

theorem binary_computation :
  (0b101101 * 0b10101 + 0b1010 / 0b10) = 0b110111100000 := by
  sorry

end binary_computation_l1433_143394


namespace rectangle_area_y_value_l1433_143314

theorem rectangle_area_y_value :
  ∀ (y : ℝ), 
  let P := (1, 1)
  let Q := (1, 4)
  let R := (y, 4)
  let S := (y, 1)
  (y > 1) → 
  (abs (R.1 - P.1) * abs (Q.2 - P.2) = 36) → 
  y = 13 :=
by
  intros y P Q R S hy harea
  let P := (1, 1)
  let Q := (1, 4)
  let R := (y, 4)
  let S := (y, 1)
  sorry

end rectangle_area_y_value_l1433_143314


namespace egyptian_fraction_l1433_143385

theorem egyptian_fraction (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) : 
  (2 : ℚ) / 7 = (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c :=
by
  sorry

end egyptian_fraction_l1433_143385


namespace fifth_term_arithmetic_sequence_l1433_143335

variable (a d : ℤ)

def arithmetic_sequence (n : ℤ) : ℤ :=
  a + (n - 1) * d

theorem fifth_term_arithmetic_sequence :
  arithmetic_sequence a d 20 = 12 →
  arithmetic_sequence a d 21 = 15 →
  arithmetic_sequence a d 5 = -33 :=
by
  intro h20 h21
  sorry

end fifth_term_arithmetic_sequence_l1433_143335


namespace books_sold_l1433_143349

-- Define the conditions
def initial_books : ℕ := 134
def books_given_away : ℕ := 39
def remaining_books : ℕ := 68

-- Define the intermediate calculation of books left after giving away
def books_after_giving_away : ℕ := initial_books - books_given_away

-- Prove the number of books sold
theorem books_sold (initial_books books_given_away remaining_books : ℕ) (h1 : books_after_giving_away = 95) (h2 : remaining_books = 68) :
  (books_after_giving_away - remaining_books) = 27 :=
by
  sorry

end books_sold_l1433_143349


namespace range_of_m_l1433_143345

variable (a b : ℝ)

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc 0 1, x^3 - m ≤ a * x + b ∧ a * x + b ≤ x^3 + m) ↔ m ∈ Set.Ici (Real.sqrt 3 / 9) :=
by
  sorry

end range_of_m_l1433_143345


namespace billy_avoids_swimming_n_eq_2022_billy_wins_for_odd_n_billy_wins_for_even_n_l1433_143344

theorem billy_avoids_swimming_n_eq_2022 :
  ∀ n : ℕ, n = 2022 → (∃ (strategy : ℕ → ℕ), ∀ k, strategy (2022 + 1 - k) ≠ strategy (k + 1)) :=
by
  sorry

theorem billy_wins_for_odd_n (n : ℕ) (h : n > 10 ∧ n % 2 = 1) :
  ∃ (strategy : ℕ → ℕ), (∀ k, strategy (n + 1 - k) ≠ strategy (k + 1)) :=
by
  sorry

theorem billy_wins_for_even_n (n : ℕ) (h : n > 10 ∧ n % 2 = 0) :
  ∃ (strategy : ℕ → ℕ), (∀ k, strategy (n + 1 - k) ≠ strategy (k + 1)) :=
by
  sorry

end billy_avoids_swimming_n_eq_2022_billy_wins_for_odd_n_billy_wins_for_even_n_l1433_143344


namespace caterpillar_count_l1433_143317

theorem caterpillar_count 
    (initial_count : ℕ)
    (hatched : ℕ)
    (left : ℕ)
    (h_initial : initial_count = 14)
    (h_hatched : hatched = 4)
    (h_left : left = 8) :
    initial_count + hatched - left = 10 :=
by
    sorry

end caterpillar_count_l1433_143317


namespace rectangular_prism_dimensions_l1433_143316

theorem rectangular_prism_dimensions (a b c : ℤ) (h1: c = (a * b) / 2) (h2: 2 * (a * b + b * c + c * a) = a * b * c) :
  (a = 3 ∧ b = 10 ∧ c = 15) ∨ (a = 4 ∧ b = 6 ∧ c = 12) :=
by {
  sorry
}

end rectangular_prism_dimensions_l1433_143316


namespace solve_for_k_l1433_143307

theorem solve_for_k : 
  ∃ k : ℤ, (k + 2) / 4 - (2 * k - 1) / 6 = 1 ∧ k = -4 := 
by
  use -4
  sorry

end solve_for_k_l1433_143307


namespace sin_double_angle_l1433_143341

open Real

theorem sin_double_angle (θ : ℝ) (h : cos (π / 4 - θ) = 1 / 2) : sin (2 * θ) = -1 / 2 := 
by 
  sorry

end sin_double_angle_l1433_143341


namespace total_books_read_l1433_143386

-- Definitions based on the conditions
def books_per_month : ℕ := 4
def months_per_year : ℕ := 12
def books_per_year_per_student : ℕ := books_per_month * months_per_year

variables (c s : ℕ)

-- Main theorem statement
theorem total_books_read (c s : ℕ) : 
  (books_per_year_per_student * c * s) = 48 * c * s :=
by
  sorry

end total_books_read_l1433_143386


namespace avg_diff_l1433_143332

theorem avg_diff (a x c : ℝ) (h1 : (a + x) / 2 = 40) (h2 : (x + c) / 2 = 60) :
  c - a = 40 :=
by
  sorry

end avg_diff_l1433_143332


namespace problem_1_problem_2_l1433_143390

open Real

def vec_parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

def vec_perpendicular (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

theorem problem_1 (k : ℝ) : 
  let a := (3, 4)
  let b := (2, k)
  (vec_parallel (a.1 + 2 * b.1, a.2 + 2 * b.2) (a.1 - b.1, a.2 - b.2)) →
  k = 8 / 3 := sorry

theorem problem_2 (k : ℝ) : 
  let a := (3, 4)
  let b := (2, k)
  (vec_perpendicular (a.1 + b.1, a.2 + b.2) (a.1 - b.1, a.2 - b.2)) →
  k = sqrt 21 ∨ k = - sqrt 21 := sorry

end problem_1_problem_2_l1433_143390


namespace missile_time_equation_l1433_143354

variable (x : ℝ)

def machToMetersPerSecond := 340
def missileSpeedInMach := 26
def secondsPerMinute := 60
def distanceToTargetInKilometers := 12000
def kilometersToMeters := 1000

theorem missile_time_equation :
  (missileSpeedInMach * machToMetersPerSecond * secondsPerMinute * x) / kilometersToMeters = distanceToTargetInKilometers :=
sorry

end missile_time_equation_l1433_143354


namespace derivative_at_one_l1433_143363

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem derivative_at_one : (deriv f 1) = 2 * Real.exp 1 := by
  sorry

end derivative_at_one_l1433_143363


namespace sachin_age_l1433_143397

variable {S R : ℕ}

theorem sachin_age
  (h1 : R = S + 7)
  (h2 : S * 3 = 2 * R) :
  S = 14 :=
sorry

end sachin_age_l1433_143397


namespace area_of_quadrilateral_l1433_143369

-- Definitions of the given conditions
def diagonal_length : ℝ := 40
def offset1 : ℝ := 11
def offset2 : ℝ := 9

-- The area of the quadrilateral
def quadrilateral_area : ℝ := 400

-- Proof statement
theorem area_of_quadrilateral :
  (1/2 * diagonal_length * offset1 + 1/2 * diagonal_length * offset2) = quadrilateral_area :=
by sorry

end area_of_quadrilateral_l1433_143369


namespace batsman_average_after_12th_innings_l1433_143339

theorem batsman_average_after_12th_innings (A : ℤ) :
  (∀ A : ℤ, (11 * A + 60 = 12 * (A + 2))) → (A = 36) → (A + 2 = 38) := 
by
  intro h_avg_increase h_init_avg
  sorry

end batsman_average_after_12th_innings_l1433_143339


namespace laila_scores_possible_values_l1433_143333

theorem laila_scores_possible_values :
  ∃ (num_y_values : ℕ), num_y_values = 4 ∧ 
  (∀ (x y : ℤ), 0 ≤ x ∧ x ≤ 100 ∧
                 0 ≤ y ∧ y ≤ 100 ∧
                 4 * x + y = 410 ∧
                 y > x → 
                 (y = 86 ∨ y = 90 ∨ y = 94 ∨ y = 98)
  ) :=
  ⟨4, by sorry⟩

end laila_scores_possible_values_l1433_143333


namespace group_selection_l1433_143308

theorem group_selection (m f : ℕ) (h1 : m + f = 8) (h2 : (m * (m - 1) / 2) * f = 30) : f = 3 :=
sorry

end group_selection_l1433_143308


namespace positive_difference_of_complementary_angles_in_ratio_five_to_four_l1433_143361

theorem positive_difference_of_complementary_angles_in_ratio_five_to_four
  (a b : ℝ)
  (h1 : a / b = 5 / 4)
  (h2 : a + b = 90) :
  |a - b| = 10 :=
sorry

end positive_difference_of_complementary_angles_in_ratio_five_to_four_l1433_143361


namespace primes_dividing_expression_l1433_143352

theorem primes_dividing_expression (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) : 
  6 * p * q ∣ p^3 + q^2 + 38 ↔ (p = 3 ∧ (q = 5 ∨ q = 13)) := 
sorry

end primes_dividing_expression_l1433_143352


namespace sequence_formula_l1433_143371

theorem sequence_formula (S : ℕ → ℤ) (a : ℕ → ℤ) (h : ∀ n : ℕ, n > 0 → S n = 2 * a n - 2^n + 1) : 
  ∀ n : ℕ, n > 0 → a n = n * 2^(n - 1) :=
by
  intro n hn
  sorry

end sequence_formula_l1433_143371


namespace break_even_shirts_needed_l1433_143303

-- Define the conditions
def initialInvestment : ℕ := 1500
def costPerShirt : ℕ := 3
def sellingPricePerShirt : ℕ := 20

-- Define the profit per T-shirt and the number of T-shirts to break even
def profitPerShirt (sellingPrice costPrice : ℕ) : ℕ := sellingPrice - costPrice

def shirtsToBreakEven (investment profit : ℕ) : ℕ :=
  (investment + profit - 1) / profit -- ceil division

-- The theorem to prove
theorem break_even_shirts_needed :
  shirtsToBreakEven initialInvestment (profitPerShirt sellingPricePerShirt costPerShirt) = 89 :=
by
  -- Calculation
  sorry

end break_even_shirts_needed_l1433_143303


namespace quotient_base4_l1433_143327

def base4_to_base10 (n : ℕ) : ℕ :=
  n % 10 + 4 * (n / 10 % 10) + 4^2 * (n / 100 % 10) + 4^3 * (n / 1000)

def base10_to_base4 (n : ℕ) : ℕ :=
  let rec convert (n acc : ℕ) : ℕ :=
    if n < 4 then n * acc
    else convert (n / 4) ((n % 4) * acc * 10 + acc)
  convert n 1

theorem quotient_base4 (a b : ℕ) (h1 : a = 2313) (h2 : b = 13) :
  base10_to_base4 ((base4_to_base10 a) / (base4_to_base10 b)) = 122 :=
by
  sorry

end quotient_base4_l1433_143327


namespace price_per_litre_of_second_oil_l1433_143309

-- Define the conditions given in the problem
def oil1_volume : ℝ := 10 -- 10 litres of first oil
def oil1_rate : ℝ := 50 -- Rs. 50 per litre

def oil2_volume : ℝ := 5 -- 5 litres of the second oil
def total_mixed_volume : ℝ := oil1_volume + oil2_volume -- Total volume of mixed oil

def mixed_rate : ℝ := 55.33 -- Rs. 55.33 per litre for the mixed oil

-- Define the target value to prove: price per litre of the second oil
def price_of_second_oil : ℝ := 65.99

-- Prove the statement
theorem price_per_litre_of_second_oil : 
  (oil1_volume * oil1_rate + oil2_volume * price_of_second_oil) = total_mixed_volume * mixed_rate :=
by 
  sorry -- actual proof to be provided

end price_per_litre_of_second_oil_l1433_143309


namespace abcd_sum_is_12_l1433_143365

theorem abcd_sum_is_12 (a b c d : ℤ) 
  (h1 : a + c = 2) 
  (h2 : a * c + b + d = -1) 
  (h3 : a * d + b * c = 18) 
  (h4 : b * d = 24) : 
  a + b + c + d = 12 :=
sorry

end abcd_sum_is_12_l1433_143365


namespace total_cost_correct_l1433_143343

noncomputable def total_cost (sandwiches: ℕ) (price_per_sandwich: ℝ) (sodas: ℕ) (price_per_soda: ℝ) (discount: ℝ) (tax: ℝ) : ℝ :=
  let total_sandwich_cost := sandwiches * price_per_sandwich
  let total_soda_cost := sodas * price_per_soda
  let discounted_sandwich_cost := total_sandwich_cost * (1 - discount)
  let total_before_tax := discounted_sandwich_cost + total_soda_cost
  let total_with_tax := total_before_tax * (1 + tax)
  total_with_tax

theorem total_cost_correct : 
  total_cost 2 3.49 4 0.87 0.10 0.05 = 10.25 :=
by
  sorry

end total_cost_correct_l1433_143343


namespace no_solution_exists_l1433_143315

theorem no_solution_exists (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : ¬(2 / a + 2 / b = 1 / (a + b)) :=
sorry

end no_solution_exists_l1433_143315


namespace inequality_solution_function_min_value_l1433_143359

theorem inequality_solution (a : ℕ) (h₁ : abs ((3 / 2 : ℚ) - 2) < a) (h₂ : abs ((1 / 2 : ℚ) - 2) ≥ a) : a = 1 := 
by
  -- proof omitted
  sorry

theorem function_min_value (a : ℕ) (h₁ : abs ((3 / 2 : ℚ) - 2) < a) (h₂ : abs ((1 / 2 : ℚ) - 2) ≥ a)
  (h₃ : a = 1) : ∃ x : ℝ, -1 ≤ x ∧ x ≤ 2 ∧ ∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → (abs (x + a) + abs (x - 2)) = 3 :=
by
  -- proof omitted
  use 0
  -- proof omitted
  sorry

end inequality_solution_function_min_value_l1433_143359


namespace range_of_t_l1433_143351

-- Define set A and set B as conditions
def setA := { x : ℝ | -3 < x ∧ x < 7 }
def setB (t : ℝ) := { x : ℝ | t + 1 < x ∧ x < 2 * t - 1 }

-- Lean statement to prove the range of t
theorem range_of_t (t : ℝ) : setB t ⊆ setA → t ≤ 4 :=
by
  -- sorry acts as a placeholder for the proof
  sorry

end range_of_t_l1433_143351


namespace ratio_of_saramago_readers_l1433_143387

theorem ratio_of_saramago_readers 
  (W : ℕ) (S K B N : ℕ)
  (h1 : W = 42)
  (h2 : K = W / 6)
  (h3 : B = 3)
  (h4 : N = (S - B) - 1)
  (h5 : W = (S - B) + (K - B) + B + N) :
  S / W = 1 / 2 :=
by
  sorry

end ratio_of_saramago_readers_l1433_143387


namespace factorize_polynomial_l1433_143318

noncomputable def polynomial_factorization : Prop :=
  ∀ x : ℤ, (x^12 + x^9 + 1) = (x^4 + x^3 + x^2 + x + 1) * (x^8 - x^7 + x^6 - x^5 + x^3 - x^2 + x - 1)

theorem factorize_polynomial : polynomial_factorization :=
by
  sorry

end factorize_polynomial_l1433_143318


namespace tape_recorder_cost_l1433_143358

-- Define the conditions
def conditions (x p : ℚ) : Prop :=
  170 < p ∧ p < 195 ∧
  2 * p = x * (x - 2) ∧
  1 * x = x - 2 + 2

-- Define the statement to be proved
theorem tape_recorder_cost (x : ℚ) (p : ℚ) : conditions x p → p = 180 := by
  sorry

end tape_recorder_cost_l1433_143358


namespace vector_arithmetic_l1433_143312

-- Define the vectors
def v1 : ℝ × ℝ := (3, -5)
def v2 : ℝ × ℝ := (2, -6)
def v3 : ℝ × ℝ := (-1, 4)

-- Define scalar multiplications
def scalar_mult1 : ℝ × ℝ := (12, -20)  -- 4 * v1
def scalar_mult2 : ℝ × ℝ := (6, -18)   -- 3 * v2

-- Define intermediate vector operations
def intermediate_vector1 : ℝ × ℝ := (6, -2)  -- (12, -20) - (6, -18)

-- Final operation
def final_vector : ℝ × ℝ := (5, 2)  -- (6, -2) + (-1, 4)

-- Prove the main statement
theorem vector_arithmetic : 
  (4 : ℝ) • v1 - (3 : ℝ) • v2 + v3 = final_vector := by
  sorry  -- proof placeholder

end vector_arithmetic_l1433_143312


namespace croissant_process_time_in_hours_l1433_143367

-- Conditions as definitions
def num_folds : ℕ := 4
def fold_time : ℕ := 5
def rest_time : ℕ := 75
def mix_time : ℕ := 10
def bake_time : ℕ := 30

-- The main theorem statement
theorem croissant_process_time_in_hours :
  (num_folds * (fold_time + rest_time) + mix_time + bake_time) / 60 = 6 := 
sorry

end croissant_process_time_in_hours_l1433_143367


namespace circle_radius_five_c_value_l1433_143370

theorem circle_radius_five_c_value {c : ℝ} :
  (∀ x y : ℝ, x^2 + 8 * x + y^2 + 2 * y + c = 0) → 
  (∃ x y : ℝ, (x + 4)^2 + (y + 1)^2 = 25) → 
  c = 42 :=
by
  sorry

end circle_radius_five_c_value_l1433_143370


namespace rebecca_has_22_eggs_l1433_143323

-- Define the conditions
def number_of_groups : ℕ := 11
def eggs_per_group : ℕ := 2

-- Define the total number of eggs calculated from the conditions.
def total_eggs : ℕ := number_of_groups * eggs_per_group

-- State the theorem and provide the proof outline.
theorem rebecca_has_22_eggs : total_eggs = 22 := by {
  -- Proof will go here, but for now we put sorry to indicate it is not yet provided.
  sorry
}

end rebecca_has_22_eggs_l1433_143323


namespace runner_time_difference_l1433_143383

theorem runner_time_difference (v : ℝ) (h1 : 0 < v) (h2 : 0 < 20 / v) (h3 : 8 = 40 / v) :
  8 - (20 / v) = 4 := by
  sorry

end runner_time_difference_l1433_143383


namespace special_collection_books_l1433_143374

theorem special_collection_books (initial_books loaned_books returned_percent: ℕ) (loaned_books_value: loaned_books = 55) (returned_percent_value: returned_percent = 80) (initial_books_value: initial_books = 75) :
  initial_books - (loaned_books - (returned_percent * loaned_books / 100)) = 64 := by
  sorry

end special_collection_books_l1433_143374


namespace selling_price_of_mixture_per_litre_l1433_143396

def cost_per_litre : ℝ := 3.60
def litres_of_pure_milk : ℝ := 25
def litres_of_water : ℝ := 5
def total_volume_of_mixture : ℝ := litres_of_pure_milk + litres_of_water
def total_cost_of_pure_milk : ℝ := cost_per_litre * litres_of_pure_milk

theorem selling_price_of_mixture_per_litre :
  total_cost_of_pure_milk / total_volume_of_mixture = 3 := by
  sorry

end selling_price_of_mixture_per_litre_l1433_143396


namespace simplify_divide_expression_l1433_143324

noncomputable def a : ℝ := Real.sqrt 2 + 1

theorem simplify_divide_expression : 
  (1 - (a / (a + 1))) / ((a^2 - 1) / (a^2 + 2 * a + 1)) = Real.sqrt 2 / 2 :=
by
  sorry

end simplify_divide_expression_l1433_143324


namespace solve_eq1_solve_eq2_l1433_143372

-- Define the theorem for the first equation
theorem solve_eq1 (x : ℝ) (h : 2 * x - 7 = 5 * x - 1) : x = -2 :=
sorry

-- Define the theorem for the second equation
theorem solve_eq2 (x : ℝ) (h : (x - 2) / 2 - (x - 1) / 6 = 1) : x = 11 / 2 :=
sorry

end solve_eq1_solve_eq2_l1433_143372


namespace parabola_properties_l1433_143347

theorem parabola_properties (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c < 0) :
  (∀ x, a * x^2 + b * x + c >= a * (x^2)) ∧
  (c < 0) ∧ 
  (-b / (2 * a) < 0) :=
by
  sorry

end parabola_properties_l1433_143347


namespace common_difference_arithmetic_sequence_l1433_143319

theorem common_difference_arithmetic_sequence
  (a : ℕ → ℝ)
  (h1 : ∃ a1 d, (∀ n, a n = a1 + (n - 1) * d))
  (h2 : a 7 - 2 * a 4 = -1)
  (h3 : a 3 = 0) :
  ∃ d, (∀ a1, (a1 + 2 * d = 0 ∧ -d = -1) → d = -1/2) :=
by
  sorry

end common_difference_arithmetic_sequence_l1433_143319


namespace original_number_l1433_143398

theorem original_number (x : ℝ) (h : 1.10 * x = 550) : x = 500 :=
by
  sorry

end original_number_l1433_143398


namespace range_of_a_l1433_143350

theorem range_of_a (a : ℝ) : 
  (∃ x1 x2 : ℝ, x1 < 1 ∧ x2 > 1 ∧ x1 * x1 + (a * a - 1) * x1 + a - 2 = 0 ∧ x2 * x2 + (a * a - 1) * x2 + a - 2 = 0) ↔ -2 < a ∧ a < 1 :=
sorry

end range_of_a_l1433_143350


namespace odd_function_at_zero_l1433_143356

-- Define the property of being an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f (x)

theorem odd_function_at_zero (f : ℝ → ℝ) (h : is_odd_function f) : f 0 = 0 :=
by
  -- assume the definitions but leave the proof steps and focus on the final conclusion
  sorry

end odd_function_at_zero_l1433_143356


namespace expression_equals_five_l1433_143388

theorem expression_equals_five (a : ℝ) (h : 2 * a^2 - 3 * a + 4 = 5) : 7 + 6 * a - 4 * a^2 = 5 :=
by
  sorry

end expression_equals_five_l1433_143388


namespace simplify_trig_l1433_143302

theorem simplify_trig (x : ℝ) :
  (1 + Real.sin x + Real.cos x + Real.sqrt 2 * Real.sin x * Real.cos x) / 
  (1 - Real.sin x + Real.cos x - Real.sqrt 2 * Real.sin x * Real.cos x) = 
  1 + (Real.sqrt 2 - 1) * Real.tan (x / 2) :=
by 
  sorry

end simplify_trig_l1433_143302


namespace students_passed_correct_l1433_143322

-- Define the number of students in ninth grade.
def students_total : ℕ := 180

-- Define the number of students who bombed their finals.
def students_bombed : ℕ := students_total / 4

-- Define the number of students remaining after removing those who bombed.
def students_remaining_after_bombed : ℕ := students_total - students_bombed

-- Define the number of students who didn't show up to take the test.
def students_didnt_show : ℕ := students_remaining_after_bombed / 3

-- Define the number of students remaining after removing those who didn't show up.
def students_remaining_after_no_show : ℕ := students_remaining_after_bombed - students_didnt_show

-- Define the number of students who got less than a D.
def students_less_than_d : ℕ := 20

-- Define the number of students who passed.
def students_passed : ℕ := students_remaining_after_no_show - students_less_than_d

-- Statement to prove the number of students who passed is 70.
theorem students_passed_correct : students_passed = 70 := by
  -- Proof will be inserted here.
  sorry

end students_passed_correct_l1433_143322


namespace constant_term_expansion_l1433_143331

theorem constant_term_expansion (a : ℝ) (h : (2 + a * x) * (1 + 1/x) ^ 5 = (2 + 5 * a)) : 2 + 5 * a = 12 → a = 2 :=
by
  intro h_eq
  have h_sum : 2 + 5 * a = 12 := h_eq
  sorry

end constant_term_expansion_l1433_143331


namespace total_marbles_l1433_143329

-- There are only red, blue, and yellow marbles
universe u
variable {α : Type u}

-- The ratio of red marbles to blue marbles to yellow marbles is \(2:3:4\)
variables {r b y T : ℕ}
variable (ratio_cond : 2 * y = 4 * r ∧ 3 * y = 4 * b)

-- There are 40 yellow marbles in the container
variable (yellow_cond : y = 40)

-- Prove the total number of marbles in the container is 90
theorem total_marbles (ratio_cond : 2 * y = 4 * r ∧ 3 * y = 4 * b) (yellow_cond : y = 40) :
  T = r + b + y → T = 90 :=
sorry

end total_marbles_l1433_143329


namespace problem_inequality_solution_l1433_143337

noncomputable def find_b_and_c (x : ℝ) (b c : ℝ) : Prop :=
  ∀ x, (x > 2 ∨ x < 1) ↔ x^2 + b*x + c > 0

theorem problem_inequality_solution (x : ℝ) :
  find_b_and_c x (-3) 2 ∧ (2*x^2 - 3*x + 1 ≤ 0 ↔ 1/2 ≤ x ∧ x ≤ 1) :=
by
  sorry

end problem_inequality_solution_l1433_143337


namespace polynomial_has_real_root_l1433_143304

theorem polynomial_has_real_root (b : ℝ) : ∃ x : ℝ, x^3 + b * x^2 - 4 * x + b = 0 := 
sorry

end polynomial_has_real_root_l1433_143304


namespace factorization_property_l1433_143389

theorem factorization_property (a b : ℤ) (h1 : 25 * x ^ 2 - 160 * x - 144 = (5 * x + a) * (5 * x + b)) 
    (h2 : a + b = -32) (h3 : a * b = -144) : 
    a + 2 * b = -68 := 
sorry

end factorization_property_l1433_143389


namespace john_annual_profit_l1433_143382

-- Definitions of monthly incomes
def TenantA_income : ℕ := 350
def TenantB_income : ℕ := 400
def TenantC_income : ℕ := 450

-- Total monthly income
def total_monthly_income : ℕ := TenantA_income + TenantB_income + TenantC_income

-- Definitions of monthly expenses
def rent_expense : ℕ := 900
def utilities_expense : ℕ := 100
def maintenance_fee : ℕ := 50

-- Total monthly expenses
def total_monthly_expense : ℕ := rent_expense + utilities_expense + maintenance_fee

-- Monthly profit
def monthly_profit : ℕ := total_monthly_income - total_monthly_expense

-- Annual profit
def annual_profit : ℕ := monthly_profit * 12

theorem john_annual_profit :
  annual_profit = 1800 := by
  -- The proof is omitted, but the statement asserts that John makes an annual profit of $1800.
  sorry

end john_annual_profit_l1433_143382


namespace solve_for_n_l1433_143380

theorem solve_for_n (n : ℝ) (h : 0.05 * n + 0.1 * (30 + n) - 0.02 * n = 15.5) : n = 96 := 
by 
  sorry

end solve_for_n_l1433_143380


namespace solve_system_of_equations_l1433_143306

theorem solve_system_of_equations :
  ∃ (x y : ℝ),
    (5 * x^2 - 14 * x * y + 10 * y^2 = 17) ∧ (4 * x^2 - 10 * x * y + 6 * y^2 = 8) ∧
    ((x = -1 ∧ y = -2) ∨ (x = 11 ∧ y = 7) ∨ (x = -11 ∧ y = -7) ∨ (x = 1 ∧ y = 2)) :=
by
  sorry

end solve_system_of_equations_l1433_143306


namespace max_expression_tends_to_infinity_l1433_143357

noncomputable def maximize_expression (x y z : ℝ) : ℝ :=
  1 / ((1 - x^2) * (1 - y^2) * (1 - z^2)) + 1 / ((1 + x^2) * (1 + y^2) * (1 + z^2))

theorem max_expression_tends_to_infinity : 
  ∀ (x y z : ℝ), -1 < x ∧ x < 1 ∧ -1 < y ∧ y < 1 ∧ -1 < z ∧ z < 1 → 
    ∃ M : ℝ, maximize_expression x y z > M :=
by
  intro x y z h
  sorry

end max_expression_tends_to_infinity_l1433_143357


namespace total_ticket_sales_is_48_l1433_143313

noncomputable def ticket_sales (total_revenue : ℕ) (price_per_ticket : ℕ) (discount_1 : ℕ) (discount_2 : ℕ) : ℕ :=
  let number_first_batch := 10
  let number_second_batch := 20
  let revenue_first_batch := number_first_batch * (price_per_ticket - (price_per_ticket * discount_1 / 100))
  let revenue_second_batch := number_second_batch * (price_per_ticket - (price_per_ticket * discount_2 / 100))
  let revenue_full_price := total_revenue - (revenue_first_batch + revenue_second_batch)
  let number_full_price_tickets := revenue_full_price / price_per_ticket
  number_first_batch + number_second_batch + number_full_price_tickets

theorem total_ticket_sales_is_48 : ticket_sales 820 20 40 15 = 48 :=
by
  sorry

end total_ticket_sales_is_48_l1433_143313


namespace cube_surface_area_l1433_143320

open Real

theorem cube_surface_area (V : ℝ) (a : ℝ) (S : ℝ)
  (h1 : V = a ^ 3)
  (h2 : a = 4)
  (h3 : V = 64) :
  S = 6 * a ^ 2 :=
by
  sorry

end cube_surface_area_l1433_143320


namespace percentage_difference_l1433_143384

-- Define the numbers
def n : ℕ := 1600
def m : ℕ := 650

-- Define the percentages calculated
def p₁ : ℕ := (20 * n) / 100
def p₂ : ℕ := (20 * m) / 100

-- The theorem to be proved: the difference between the two percentages is 190
theorem percentage_difference : p₁ - p₂ = 190 := by
  sorry

end percentage_difference_l1433_143384


namespace cubes_divisible_by_nine_l1433_143321

theorem cubes_divisible_by_nine (n : ℕ) (hn : n > 0) : 
    (n^3 + (n + 1)^3 + (n + 2)^3) % 9 = 0 := by
  sorry

end cubes_divisible_by_nine_l1433_143321


namespace find_angle_x_l1433_143355

theorem find_angle_x (A B C D : Type) 
  (angleACB angleBCD : ℝ) 
  (h1 : angleACB = 90)
  (h2 : angleBCD = 40) 
  (h3 : angleACB + angleBCD + x = 180) : 
  x = 50 :=
by
  sorry

end find_angle_x_l1433_143355


namespace distinct_gcd_numbers_l1433_143342

theorem distinct_gcd_numbers (nums : Fin 100 → ℕ) (h_distinct : Function.Injective nums) :
  ¬ ∃ a b c : Fin 100, 
    a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
    (nums a + Nat.gcd (nums b) (nums c) = nums b + Nat.gcd (nums a) (nums c)) ∧ 
    (nums b + Nat.gcd (nums a) (nums c) = nums c + Nat.gcd (nums a) (nums b)) := 
sorry

end distinct_gcd_numbers_l1433_143342


namespace pencils_lost_l1433_143336

theorem pencils_lost (bought_pencils remaining_pencils lost_pencils : ℕ)
                     (h1 : bought_pencils = 16)
                     (h2 : remaining_pencils = 8)
                     (h3 : lost_pencils = bought_pencils - remaining_pencils) :
                     lost_pencils = 8 :=
by {
  sorry
}

end pencils_lost_l1433_143336


namespace roots_of_polynomial_equation_l1433_143373

theorem roots_of_polynomial_equation (x : ℝ) :
  4 * x ^ 4 - 21 * x ^ 3 + 34 * x ^ 2 - 21 * x + 4 = 0 ↔ x = 4 ∨ x = 1 / 4 ∨ x = 1 :=
by
  sorry

end roots_of_polynomial_equation_l1433_143373


namespace problem_x_value_l1433_143301

theorem problem_x_value (x : ℝ) (h : 0.25 * x = 0.15 * 1500 - 15) : x = 840 :=
by
  sorry

end problem_x_value_l1433_143301


namespace line_through_parabola_vertex_unique_value_l1433_143348

theorem line_through_parabola_vertex_unique_value :
  ∃! a : ℝ, ∃ y : ℝ, y = x + a ∧ y = x^2 - 2*a*x + a^2 :=
sorry

end line_through_parabola_vertex_unique_value_l1433_143348


namespace evaluate_expression_l1433_143362

theorem evaluate_expression (x : ℝ) : (x+2)^2 + 2*(x+2)*(4-x) + (4-x)^2 = 36 :=
by sorry

end evaluate_expression_l1433_143362


namespace Fabian_total_cost_correct_l1433_143300

noncomputable def total_spent_by_Fabian (mouse_cost : ℝ) : ℝ :=
  let keyboard_cost := 2 * mouse_cost
  let headphones_cost := mouse_cost + 15
  let usb_hub_cost := 36 - mouse_cost
  let webcam_cost := keyboard_cost / 2
  let total_cost := mouse_cost + keyboard_cost + headphones_cost + usb_hub_cost + webcam_cost
  let discounted_total := total_cost * 0.90
  let final_total := discounted_total * 1.05
  final_total

theorem Fabian_total_cost_correct :
  total_spent_by_Fabian 20 = 123.80 :=
by
  sorry

end Fabian_total_cost_correct_l1433_143300


namespace compare_abc_l1433_143376

noncomputable def a : ℝ := 9 ^ (Real.log 4.1 / Real.log 2)
noncomputable def b : ℝ := 9 ^ (Real.log 2.7 / Real.log 2)
noncomputable def c : ℝ := (1 / 3 : ℝ) ^ (Real.log 0.1 / Real.log 2)

theorem compare_abc :
  a > c ∧ c > b := by
  sorry

end compare_abc_l1433_143376


namespace pencils_difference_l1433_143311

theorem pencils_difference
  (pencils_in_backpack : ℕ := 2)
  (pencils_at_home : ℕ := 15) :
  pencils_at_home - pencils_in_backpack = 13 := by
  sorry

end pencils_difference_l1433_143311
