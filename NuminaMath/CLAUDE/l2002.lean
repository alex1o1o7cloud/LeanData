import Mathlib

namespace count_distinct_digits_eq_2352_l2002_200282

/-- The count of integers between 2000 and 9999 with four distinct digits, none of which is 5 -/
def count_distinct_digits : ℕ :=
  let first_digit := 7  -- 2, 3, 4, 6, 7, 8, 9
  let second_digit := 8 -- 0, 1, 2, 3, 4, 6, 7, 8, 9 (excluding the first digit)
  let third_digit := 7  -- remaining digits excluding 5 and the first two chosen
  let fourth_digit := 6 -- remaining digits excluding 5 and the first three chosen
  first_digit * second_digit * third_digit * fourth_digit

theorem count_distinct_digits_eq_2352 : count_distinct_digits = 2352 := by
  sorry

end count_distinct_digits_eq_2352_l2002_200282


namespace card_value_decrease_l2002_200204

theorem card_value_decrease (initial_value : ℝ) (h : initial_value > 0) : 
  let first_year_value := initial_value * (1 - 0.1)
  let second_year_value := first_year_value * (1 - 0.1)
  let total_decrease := (initial_value - second_year_value) / initial_value
  total_decrease = 0.19 := by
sorry

end card_value_decrease_l2002_200204


namespace percentage_of_boys_with_dogs_l2002_200277

/-- Proves that 10% of boys have dogs at home given the conditions of the problem -/
theorem percentage_of_boys_with_dogs (total_students : ℕ) (girls_with_dogs : ℕ) (total_with_dogs : ℕ) :
  total_students = 100 →
  girls_with_dogs = (20 * (total_students / 2)) / 100 →
  total_with_dogs = 15 →
  (total_with_dogs - girls_with_dogs) * 100 / (total_students / 2) = 10 := by
sorry

end percentage_of_boys_with_dogs_l2002_200277


namespace min_values_ab_and_fraction_l2002_200211

theorem min_values_ab_and_fraction (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h : 1/a + 3/b = 1) : 
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 1/x + 3/y = 1 ∧ x*y ≤ a*b) ∧
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 1/x + 3/y = 1 ∧ x*y = 12) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → 1/x + 3/y = 1 → x*y ≥ 12) ∧
  (∃ (x y : ℝ), x > 1 ∧ y > 3 ∧ 1/x + 3/y = 1 ∧ 1/(x-1) + 3/(y-3) ≤ 1/(a-1) + 3/(b-3)) ∧
  (∃ (x y : ℝ), x > 1 ∧ y > 3 ∧ 1/x + 3/y = 1 ∧ 1/(x-1) + 3/(y-3) = 2) ∧
  (∀ (x y : ℝ), x > 1 → y > 3 → 1/x + 3/y = 1 → 1/(x-1) + 3/(y-3) ≥ 2) :=
by sorry

end min_values_ab_and_fraction_l2002_200211


namespace fruiting_plants_given_away_l2002_200260

/-- Represents the number of plants in Roxy's garden -/
structure GardenState where
  flowering : ℕ
  fruiting : ℕ

/-- Calculates the total number of plants -/
def GardenState.total (s : GardenState) : ℕ := s.flowering + s.fruiting

def initial_state : GardenState :=
  { flowering := 7,
    fruiting := 2 * 7 }

def after_buying : GardenState :=
  { flowering := initial_state.flowering + 3,
    fruiting := initial_state.fruiting + 2 }

def plants_remaining : ℕ := 21

def flowering_given_away : ℕ := 1

theorem fruiting_plants_given_away :
  ∃ (x : ℕ), 
    after_buying.fruiting - x = plants_remaining - (after_buying.flowering - flowering_given_away) ∧
    x = 4 := by
  sorry

end fruiting_plants_given_away_l2002_200260


namespace equilateral_triangle_perimeter_l2002_200288

/-- An equilateral triangle is a triangle with all sides of equal length -/
structure EquilateralTriangle where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- The perimeter of a triangle is the sum of its side lengths -/
def perimeter (triangle : EquilateralTriangle) : ℝ :=
  3 * triangle.side_length

/-- Theorem: The perimeter of an equilateral triangle with side length 'a' is 3a -/
theorem equilateral_triangle_perimeter (a : ℝ) (ha : a > 0) :
  let triangle : EquilateralTriangle := ⟨a, ha⟩
  perimeter triangle = 3 * a := by sorry

end equilateral_triangle_perimeter_l2002_200288


namespace cyclic_inequality_l2002_200202

theorem cyclic_inequality (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (sum_abc : a + b + c = 3) : 
  (a / (b + c) + b / (c + a) + c / (a + b)) + 
  Real.sqrt 2 * (Real.sqrt (a / (b + c)) + Real.sqrt (b / (c + a)) + Real.sqrt (c / (a + b))) 
  ≥ 9/2 := by
sorry

end cyclic_inequality_l2002_200202


namespace remaining_payment_remaining_payment_specific_l2002_200224

/-- Given a product with a deposit, sales tax, and discount, calculate the remaining amount to be paid. -/
theorem remaining_payment (deposit : ℝ) (deposit_percentage : ℝ) (sales_tax_rate : ℝ) (discount_rate : ℝ) : ℝ :=
  let full_price := deposit / deposit_percentage
  let discounted_price := full_price * (1 - discount_rate)
  let final_price := discounted_price * (1 + sales_tax_rate)
  final_price - deposit

/-- Prove that the remaining payment for a product with given conditions is $733.20 -/
theorem remaining_payment_specific : 
  remaining_payment 80 0.1 0.07 0.05 = 733.20 := by
  sorry

end remaining_payment_remaining_payment_specific_l2002_200224


namespace arcade_spend_example_l2002_200234

/-- Calculates the total amount spent at an arcade given the time spent and cost per interval. -/
def arcade_spend (hours : ℕ) (cost_per_interval : ℚ) (interval_minutes : ℕ) : ℚ :=
  let total_minutes : ℕ := hours * 60
  let num_intervals : ℕ := total_minutes / interval_minutes
  ↑num_intervals * cost_per_interval

/-- Proves that spending 3 hours at an arcade using $0.50 every 6 minutes results in a total spend of $15. -/
theorem arcade_spend_example : arcade_spend 3 (1/2) 6 = 15 := by
  sorry

end arcade_spend_example_l2002_200234


namespace belongs_to_32nd_group_l2002_200245

/-- The last number in the n-th group of odd numbers -/
def last_number_in_group (n : ℕ) : ℕ := 2 * n^2 - 1

/-- The first number in the n-th group of odd numbers -/
def first_number_in_group (n : ℕ) : ℕ := 2 * (n-1)^2 + 1

/-- Theorem stating that 1991 belongs to the 32nd group -/
theorem belongs_to_32nd_group : 
  first_number_in_group 32 ≤ 1991 ∧ 1991 ≤ last_number_in_group 32 :=
sorry

end belongs_to_32nd_group_l2002_200245


namespace statement_D_is_false_l2002_200280

-- Define the set A_k
def A (k : ℕ) : Set ℤ := {x : ℤ | ∃ n : ℤ, x = 4 * n + k}

-- State the theorem
theorem statement_D_is_false : ¬ (∀ a b : ℤ, (a + b) ∈ A 3 → a ∈ A 1 ∧ b ∈ A 2) := by
  sorry

end statement_D_is_false_l2002_200280


namespace dividend_divisor_quotient_problem_l2002_200249

theorem dividend_divisor_quotient_problem :
  ∀ (dividend divisor quotient : ℕ),
    dividend = 6 * divisor →
    divisor = 6 * quotient →
    dividend = divisor * quotient →
    dividend = 216 ∧ divisor = 36 ∧ quotient = 6 := by
  sorry

end dividend_divisor_quotient_problem_l2002_200249


namespace log7_2400_rounded_to_nearest_integer_l2002_200217

-- Define the logarithm base 7 function
noncomputable def log7 (x : ℝ) : ℝ := Real.log x / Real.log 7

-- Theorem statement
theorem log7_2400_rounded_to_nearest_integer :
  ⌊log7 2400 + 0.5⌋ = 4 := by sorry

end log7_2400_rounded_to_nearest_integer_l2002_200217


namespace cone_lateral_surface_area_l2002_200220

/-- The lateral surface area of a cone with base radius 3 and slant height 5 is 15π. -/
theorem cone_lateral_surface_area :
  let r : ℝ := 3  -- radius of the base
  let s : ℝ := 5  -- slant height
  let lateral_area := π * r * s  -- formula for lateral surface area of a cone
  lateral_area = 15 * π :=
by sorry

end cone_lateral_surface_area_l2002_200220


namespace geometric_progression_proof_l2002_200210

theorem geometric_progression_proof (b₁ q : ℝ) (h_decreasing : |q| < 1) :
  (b₁^3 / (1 - q^3)) / (b₁ / (1 - q)) = 48/7 →
  (b₁^4 / (1 - q^4)) / (b₁^2 / (1 - q^2)) = 144/17 →
  (b₁ = 3 ∨ b₁ = -3) ∧ q = 1/4 := by
  sorry

end geometric_progression_proof_l2002_200210


namespace power_product_simplification_l2002_200212

theorem power_product_simplification :
  (-3/2 : ℚ)^2023 * (-2/3 : ℚ)^2022 = -3/2 := by
  sorry

end power_product_simplification_l2002_200212


namespace star_problem_l2002_200297

def star (x y : ℕ) : ℕ := x^2 + y

theorem star_problem : (3^(star 5 7)) ^ 2 + 4^(star 4 6) = 3^64 + 4^22 := by
  sorry

end star_problem_l2002_200297


namespace probability_a_and_b_selected_l2002_200209

-- Define the total number of students
def total_students : ℕ := 5

-- Define the number of students to be selected
def selected_students : ℕ := 3

-- Define a function to calculate combinations
def combination (n : ℕ) (r : ℕ) : ℕ := 
  Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Theorem statement
theorem probability_a_and_b_selected :
  (combination (total_students - 2) (selected_students - 2)) / 
  (combination total_students selected_students) = 3 / 10 := by
sorry

end probability_a_and_b_selected_l2002_200209


namespace monic_quartic_with_specific_roots_l2002_200287

-- Define the polynomial
def p (x : ℝ) : ℝ := x^4 - 10*x^3 + 31*x^2 - 34*x - 7

-- Theorem statement
theorem monic_quartic_with_specific_roots :
  -- The polynomial is monic
  (∀ x, p x = x^4 + (-10)*x^3 + 31*x^2 + (-34)*x + (-7)) ∧
  -- The polynomial has rational coefficients
  (∃ a b c d e : ℚ, ∀ x, p x = a*x^4 + b*x^3 + c*x^2 + d*x + e) ∧
  -- 3+√2 is a root
  p (3 + Real.sqrt 2) = 0 ∧
  -- 2-√5 is a root
  p (2 - Real.sqrt 5) = 0 :=
sorry

end monic_quartic_with_specific_roots_l2002_200287


namespace max_points_for_top_teams_l2002_200295

/-- Represents a football tournament with the given rules --/
structure FootballTournament where
  num_teams : ℕ
  num_top_teams : ℕ
  points_for_win : ℕ
  points_for_draw : ℕ

/-- The maximum possible points that can be achieved by the top teams --/
def max_points (t : FootballTournament) : ℕ :=
  let internal_games := t.num_top_teams.choose 2
  let external_games := t.num_top_teams * (t.num_teams - t.num_top_teams)
  internal_games * t.points_for_win + external_games * t.points_for_win

/-- The theorem stating the maximum integer N for which at least 6 teams can score N or more points --/
theorem max_points_for_top_teams (t : FootballTournament) 
  (h1 : t.num_teams = 15)
  (h2 : t.num_top_teams = 6)
  (h3 : t.points_for_win = 3)
  (h4 : t.points_for_draw = 1) :
  ∃ (N : ℕ), N = 34 ∧ 
  (∀ (M : ℕ), (M : ℝ) * t.num_top_teams ≤ max_points t → M ≤ N) ∧
  (N : ℝ) * t.num_top_teams ≤ max_points t :=
by sorry

end max_points_for_top_teams_l2002_200295


namespace power_of_power_l2002_200247

theorem power_of_power (x : ℝ) : (x^3)^2 = x^6 := by sorry

end power_of_power_l2002_200247


namespace rth_term_of_sequence_l2002_200241

-- Define the sum function for the arithmetic progression
def S (n : ℕ) : ℕ := 5 * n + 4 * n^2

-- Define the rth term of the sequence
def a_r (r : ℕ) : ℕ := S r - S (r - 1)

-- Theorem statement
theorem rth_term_of_sequence (r : ℕ) (h : r > 0) : a_r r = 8 * r + 1 := by
  sorry

end rth_term_of_sequence_l2002_200241


namespace first_traveler_constant_speed_second_traveler_constant_speed_l2002_200262

-- Define the speeds and distances
def speed1 : ℝ := 4
def speed2 : ℝ := 6
def total_distance : ℝ := 24

-- Define the constant speeds to be proven
def constant_speed1 : ℝ := 4.8
def constant_speed2 : ℝ := 5

-- Theorem for the first traveler
theorem first_traveler_constant_speed :
  let time1 := (total_distance / 2) / speed1
  let time2 := (total_distance / 2) / speed2
  let total_time := time1 + time2
  total_distance / total_time = constant_speed1 := by sorry

-- Theorem for the second traveler
theorem second_traveler_constant_speed :
  let total_time : ℝ := 2 -- Arbitrary total time
  let distance1 := speed1 * (total_time / 2)
  let distance2 := speed2 * (total_time / 2)
  let total_distance := distance1 + distance2
  total_distance / total_time = constant_speed2 := by sorry

end first_traveler_constant_speed_second_traveler_constant_speed_l2002_200262


namespace power_of_three_mod_ten_l2002_200235

theorem power_of_three_mod_ten : 3^19 % 10 = 7 := by sorry

end power_of_three_mod_ten_l2002_200235


namespace inequality_solution_l2002_200238

theorem inequality_solution (x : ℝ) : 
  (x^(1/4) + 3 / (x^(1/4) + 4) ≤ 1) ↔ 
  (x < (((-3 - Real.sqrt 5) / 2)^4) ∨ x > (((-3 + Real.sqrt 5) / 2)^4)) :=
by sorry

end inequality_solution_l2002_200238


namespace quadratic_inequality_solution_set_l2002_200218

theorem quadratic_inequality_solution_set : 
  {x : ℝ | x^2 - 3*x - 18 ≤ 0} = {x : ℝ | -3 ≤ x ∧ x ≤ 6} := by sorry

end quadratic_inequality_solution_set_l2002_200218


namespace quadratic_real_root_condition_l2002_200223

/-- A quadratic equation x^2 + bx + 25 = 0 has at least one real root if and only if b ∈ (-∞, -10] ∪ [10, ∞) -/
theorem quadratic_real_root_condition (b : ℝ) : 
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by
sorry

end quadratic_real_root_condition_l2002_200223


namespace man_speed_against_current_l2002_200271

/-- Given a man's speed with the current and the speed of the current,
    calculate the man's speed against the current. -/
def speed_against_current (speed_with_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_with_current - 2 * current_speed

/-- Theorem stating that given the specific speeds, the man's speed against the current is 20 km/hr. -/
theorem man_speed_against_current :
  speed_against_current 25 2.5 = 20 := by
  sorry

end man_speed_against_current_l2002_200271


namespace power_function_values_l2002_200216

-- Define the power function f
def f (x : ℝ) : ℝ := x^2

-- Theorem statement
theorem power_function_values :
  (f 3 = 9) →
  (f 2 = 4) ∧ (∀ x, f (2*x + 1) = 4*x^2 + 4*x + 1) :=
by
  sorry

end power_function_values_l2002_200216


namespace removed_integer_problem_l2002_200227

theorem removed_integer_problem (n : ℕ) (x : ℕ) :
  x ≤ n →
  (n * (n + 1) / 2 - x) / (n - 1 : ℝ) = 163 / 4 →
  x = 61 :=
sorry

end removed_integer_problem_l2002_200227


namespace tetrahedron_volume_l2002_200201

/-- The volume of a tetrahedron with an inscribed sphere -/
theorem tetrahedron_volume (S₁ S₂ S₃ S₄ r : ℝ) (h₁ : S₁ > 0) (h₂ : S₂ > 0) (h₃ : S₃ > 0) (h₄ : S₄ > 0) (hr : r > 0) :
  ∃ V : ℝ, V = (1/3) * (S₁ + S₂ + S₃ + S₄) * r ∧ V > 0 := by
  sorry

end tetrahedron_volume_l2002_200201


namespace complement_union_theorem_l2002_200284

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3}
def B : Set Nat := {1, 2, 4}

theorem complement_union_theorem : (U \ B) ∪ A = {1, 3, 5} := by
  sorry

end complement_union_theorem_l2002_200284


namespace anna_bob_numbers_not_equal_l2002_200267

/-- Represents a number formed by concatenating consecutive positive integers -/
def ConsecutiveIntegerNumber (start : ℕ) (count : ℕ) : ℕ := sorry

/-- Anna's number is formed by 20 consecutive positive integers -/
def AnnaNumber (start : ℕ) : ℕ := ConsecutiveIntegerNumber start 20

/-- Bob's number is formed by 21 consecutive positive integers -/
def BobNumber (start : ℕ) : ℕ := ConsecutiveIntegerNumber start 21

/-- Theorem stating that Anna's and Bob's numbers cannot be equal -/
theorem anna_bob_numbers_not_equal :
  ∀ (a b : ℕ), AnnaNumber a ≠ BobNumber b :=
sorry

end anna_bob_numbers_not_equal_l2002_200267


namespace boat_distance_proof_l2002_200270

/-- The speed of the boat in still water (mph) -/
def boat_speed : ℝ := 15.6

/-- The time taken for the trip against the current (hours) -/
def time_against : ℝ := 8

/-- The time taken for the return trip with the current (hours) -/
def time_with : ℝ := 5

/-- The speed of the current (mph) -/
def current_speed : ℝ := 3.6

/-- The distance traveled by the boat (miles) -/
def distance : ℝ := 96

theorem boat_distance_proof :
  distance = (boat_speed - current_speed) * time_against ∧
  distance = (boat_speed + current_speed) * time_with :=
by sorry

end boat_distance_proof_l2002_200270


namespace sequence_term_exists_l2002_200248

theorem sequence_term_exists : ∃ n : ℕ, n * (n + 2) = 99 := by
  sorry

end sequence_term_exists_l2002_200248


namespace probability_both_truth_l2002_200269

theorem probability_both_truth (prob_A prob_B : ℝ) 
  (h_A : prob_A = 0.7) 
  (h_B : prob_B = 0.6) : 
  prob_A * prob_B = 0.42 := by
  sorry

end probability_both_truth_l2002_200269


namespace jake_peaches_count_l2002_200265

-- Define the variables
def steven_peaches : ℕ := 13
def steven_apples : ℕ := 52
def jake_apples : ℕ := steven_apples + 84

-- Define Jake's peaches in terms of Steven's
def jake_peaches : ℕ := steven_peaches - 10

-- Theorem to prove
theorem jake_peaches_count : jake_peaches = 3 := by
  sorry

end jake_peaches_count_l2002_200265


namespace rectangle_area_l2002_200298

theorem rectangle_area (d : ℝ) (w : ℝ) (h : w > 0) : 
  d^2 = 10 * w^2 → 3 * w^2 = 3 * d^2 / 10 :=
by
  sorry

#check rectangle_area

end rectangle_area_l2002_200298


namespace arithmetic_sequence_sum_l2002_200228

theorem arithmetic_sequence_sum (d : ℝ) (h : d ≠ 0) :
  let a : ℕ → ℝ := λ n => (n - 1 : ℝ) * d
  ∃ m : ℕ, a m = (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9) ∧ m = 37 :=
by sorry

end arithmetic_sequence_sum_l2002_200228


namespace course_size_is_400_l2002_200214

/-- Proves that the total number of students in a course is 400, given the distribution of grades --/
theorem course_size_is_400 (T : ℕ) 
  (grade_A : ℕ := T / 5)
  (grade_B : ℕ := T / 4)
  (grade_C : ℕ := T / 2)
  (grade_D : ℕ := 20)
  (total_sum : T = grade_A + grade_B + grade_C + grade_D) : T = 400 := by
  sorry

end course_size_is_400_l2002_200214


namespace fraction_sum_equality_l2002_200237

theorem fraction_sum_equality : (2 : ℚ) / 15 + 4 / 20 + 5 / 45 = 4 / 9 := by
  sorry

end fraction_sum_equality_l2002_200237


namespace real_part_of_i_times_i_minus_one_l2002_200258

theorem real_part_of_i_times_i_minus_one :
  Complex.re (Complex.I * (Complex.I - 1)) = -1 := by
  sorry

end real_part_of_i_times_i_minus_one_l2002_200258


namespace chess_piece_position_l2002_200229

theorem chess_piece_position : ∃! (x y : ℕ), x > 0 ∧ y > 0 ∧ x^2 + x*y - 2*y^2 = 13 ∧ x = 5 ∧ y = 4 := by
  sorry

end chess_piece_position_l2002_200229


namespace figure_area_theorem_l2002_200200

theorem figure_area_theorem (x : ℝ) : 
  let square1_area := (2 * x)^2
  let square2_area := (5 * x)^2
  let triangle_area := (1/2) * (2 * x) * (5 * x)
  square1_area + square2_area + triangle_area = 850 → x = 5 := by
sorry

end figure_area_theorem_l2002_200200


namespace probability_of_pair_letter_l2002_200291

def word : String := "PROBABILITY"
def target : String := "PAIR"

theorem probability_of_pair_letter : 
  (word.toList.filter (fun c => target.contains c)).length / word.length = 4 / 11 := by
  sorry

end probability_of_pair_letter_l2002_200291


namespace real_root_of_complex_quadratic_l2002_200268

theorem real_root_of_complex_quadratic (k : ℝ) (a : ℝ) :
  (∃ x : ℂ, x^2 + (k + 2*I)*x + (2 : ℂ) + k*I = 0) →
  (a^2 + (k + 2*I)*a + (2 : ℂ) + k*I = 0) →
  (a = Real.sqrt 2 ∨ a = -Real.sqrt 2) :=
by sorry

end real_root_of_complex_quadratic_l2002_200268


namespace intersection_M_N_l2002_200219

def M : Set ℕ := {0, 1, 2}

def N : Set ℕ := {x | ∃ a ∈ M, x = a^2}

theorem intersection_M_N : M ∩ N = {0, 1} := by
  sorry

end intersection_M_N_l2002_200219


namespace inequality_proof_l2002_200261

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*c*a)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end inequality_proof_l2002_200261


namespace harmonic_sets_theorem_l2002_200252

-- Define a circle
class Circle where
  -- Add any necessary properties for a circle

-- Define a point on a circle
class PointOnCircle (c : Circle) where
  -- Add any necessary properties for a point on a circle

-- Define a line
class Line where
  -- Add any necessary properties for a line

-- Define the property of lines intersecting at a single point
def intersectAtSinglePoint (l1 l2 l3 l4 : Line) : Prop :=
  sorry

-- Define a harmonic set of points
def isHarmonic {c : Circle} (A B C D : PointOnCircle c) : Prop :=
  sorry

-- Define the line connecting two points
def connectingLine {c : Circle} (P Q : PointOnCircle c) : Line :=
  sorry

theorem harmonic_sets_theorem
  {c : Circle}
  (A B C D A₁ B₁ C₁ D₁ : PointOnCircle c)
  (h_intersect : intersectAtSinglePoint
    (connectingLine A A₁)
    (connectingLine B B₁)
    (connectingLine C C₁)
    (connectingLine D D₁))
  (h_harmonic : isHarmonic A B C D ∨ isHarmonic A₁ B₁ C₁ D₁) :
  isHarmonic A B C D ∧ isHarmonic A₁ B₁ C₁ D₁ :=
sorry

end harmonic_sets_theorem_l2002_200252


namespace probability_at_least_one_vowel_l2002_200251

/-- The probability of picking at least one vowel from two sets of letters -/
theorem probability_at_least_one_vowel (set1 set2 : Finset Char) 
  (vowels1 vowels2 : Finset Char) : 
  set1.card = 6 →
  set2.card = 6 →
  vowels1 ⊆ set1 →
  vowels2 ⊆ set2 →
  vowels1.card = 2 →
  vowels2.card = 1 →
  (set1.card * set2.card : ℚ)⁻¹ * 
    ((vowels1.card * set2.card) + (set1.card - vowels1.card) * vowels2.card) = 1/2 := by
  sorry

#check probability_at_least_one_vowel

end probability_at_least_one_vowel_l2002_200251


namespace right_triangle_hypotenuse_l2002_200253

theorem right_triangle_hypotenuse (a b c : ℝ) :
  a = 8 → b = 15 → c^2 = a^2 + b^2 → c = 17 := by sorry

end right_triangle_hypotenuse_l2002_200253


namespace equation_solutions_l2002_200226

theorem equation_solutions :
  (∃ x : ℝ, 5 * x - 2 = 2 * (x + 2) ∧ x = 2) ∧
  (∃ x : ℝ, 2 * x + (x - 3) / 2 = (2 - x) / 3 - 5 ∧ x = -1) := by
  sorry

end equation_solutions_l2002_200226


namespace power_of_three_remainder_l2002_200278

theorem power_of_three_remainder (k : ℕ) : (3 ^ (4 * k + 3)) % 10 = 7 := by
  sorry

end power_of_three_remainder_l2002_200278


namespace x_minus_y_values_l2002_200244

theorem x_minus_y_values (x y : ℝ) (h1 : x^2 = 4) (h2 : |y| = 3) (h3 : x + y < 0) :
  x - y = 1 ∨ x - y = 5 := by
sorry

end x_minus_y_values_l2002_200244


namespace largest_multiple_of_seven_less_than_fifty_l2002_200246

theorem largest_multiple_of_seven_less_than_fifty :
  ∃ n : ℕ, n = 49 ∧ 7 ∣ n ∧ n < 50 ∧ ∀ m : ℕ, 7 ∣ m → m < 50 → m ≤ n :=
by sorry

end largest_multiple_of_seven_less_than_fifty_l2002_200246


namespace function_decreasing_iff_a_in_range_l2002_200213

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 3) * x + 5 else 3 * a / x

theorem function_decreasing_iff_a_in_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (x₁ - x₂) * (f a x₁ - f a x₂) < 0) ↔ 0 < a ∧ a ≤ 1 :=
sorry

end function_decreasing_iff_a_in_range_l2002_200213


namespace max_abs_f_value_l2002_200292

-- Define the band region type
def band_region (k l : ℝ) (y : ℝ) : Prop := k ≤ y ∧ y ≤ l

-- Define the quadratic function
variable (a b c : ℝ)
def f (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem max_abs_f_value :
  ∀ a b c : ℝ,
  (band_region 0 4 (f a b c (-2) + 2)) ∧
  (band_region 0 4 (f a b c 0 + 2)) ∧
  (band_region 0 4 (f a b c 2 + 2)) →
  (∀ t : ℝ, band_region (-1) 3 (t + 1) → |f a b c t| ≤ 5/2) ∧
  (∃ t : ℝ, band_region (-1) 3 (t + 1) ∧ |f a b c t| = 5/2) :=
by sorry

end max_abs_f_value_l2002_200292


namespace min_value_of_expression_l2002_200264

theorem min_value_of_expression (x y z : ℝ) (h : x + y + 3 * z = 6) :
  ∃ (m : ℝ), m = 0 ∧ ∀ (x' y' z' : ℝ), x' + y' + 3 * z' = 6 → x' * y' + 2 * x' * z' + 3 * y' * z' ≥ m :=
by sorry

end min_value_of_expression_l2002_200264


namespace cosine_sum_equality_l2002_200283

theorem cosine_sum_equality : 
  Real.cos (43 * π / 180) * Real.cos (13 * π / 180) + 
  Real.sin (43 * π / 180) * Real.sin (13 * π / 180) = 
  Real.sqrt 3 / 2 := by
sorry

end cosine_sum_equality_l2002_200283


namespace factorization_proof_l2002_200290

theorem factorization_proof (x : ℝ) : (x + 3)^2 - (x + 3) = (x + 3) * (x + 2) := by
  sorry

end factorization_proof_l2002_200290


namespace second_object_length_l2002_200259

def tape_length : ℕ := 5
def object1_length : ℕ := 100
def object2_length : ℕ := 780

theorem second_object_length :
  (tape_length ∣ object1_length) ∧ 
  (tape_length ∣ object2_length) ∧ 
  (∃ k : ℕ, k * tape_length = object2_length) →
  object2_length = 780 :=
by sorry

end second_object_length_l2002_200259


namespace range_of_expression_l2002_200276

theorem range_of_expression (x y : ℝ) (h : x^2 + y^2 = 1) :
  ∃ (z : ℝ), z = 5 * Real.arcsin x - 2 * Real.arccos y ∧ 
  -7/2 * Real.pi ≤ z ∧ z ≤ 3/2 * Real.pi ∧
  (∀ ε > 0, ∃ (x' y' : ℝ), x'^2 + y'^2 = 1 ∧
    (5 * Real.arcsin x' - 2 * Real.arccos y' < -7/2 * Real.pi + ε ∨
     5 * Real.arcsin x' - 2 * Real.arccos y' > 3/2 * Real.pi - ε)) :=
by sorry

end range_of_expression_l2002_200276


namespace tan_675_degrees_l2002_200230

theorem tan_675_degrees : Real.tan (675 * π / 180) = -1 := by
  sorry

end tan_675_degrees_l2002_200230


namespace intersection_and_union_subset_condition_l2002_200233

-- Define the sets A and B
def A : Set ℝ := {x | x < -4 ∨ x > 1}
def B : Set ℝ := {x | -3 ≤ x - 1 ∧ x - 1 ≤ 2}

-- Define set M
def M (a : ℝ) : Set ℝ := {x | 2*a ≤ x ∧ x < 2*a + 2}

-- Theorem for part 1
theorem intersection_and_union :
  (A ∩ B = {x | 1 < x ∧ x ≤ 3}) ∧
  ((Aᶜ ∪ Bᶜ) = {x | x ≤ 1 ∨ x > 3}) := by sorry

-- Theorem for part 2
theorem subset_condition (a : ℝ) :
  M a ⊆ A ↔ a ≤ -3 ∨ a > 1/2 := by sorry

end intersection_and_union_subset_condition_l2002_200233


namespace twelve_lines_theorem_l2002_200225

/-- A line in a plane. -/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- A point in a plane. -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A triangle in a plane. -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- The distance from a point to a line. -/
def distance_to_line (p : Point) (l : Line) : ℝ :=
  sorry

/-- Check if the distances from three points to a line are in one of the specified ratios. -/
def valid_ratio (A B C : Point) (l : Line) : Prop :=
  let dA := distance_to_line A l
  let dB := distance_to_line B l
  let dC := distance_to_line C l
  (dA = dB / 2 ∧ dA = dC / 2) ∨
  (dB = dA / 2 ∧ dB = dC / 2) ∨
  (dC = dA / 2 ∧ dC = dB / 2)

/-- The main theorem: there are exactly 12 lines satisfying the distance ratio condition for any triangle. -/
theorem twelve_lines_theorem (t : Triangle) : 
  ∃! (s : Finset Line), s.card = 12 ∧ ∀ l ∈ s, valid_ratio t.A t.B t.C l :=
sorry

end twelve_lines_theorem_l2002_200225


namespace smallest_n_divisible_by_18_and_640_l2002_200285

theorem smallest_n_divisible_by_18_and_640 : ∃! n : ℕ+, 
  (∀ m : ℕ+, m < n → (¬(18 ∣ m^2) ∨ ¬(640 ∣ m^3))) ∧ 
  (18 ∣ n^2) ∧ (640 ∣ n^3) :=
by
  use 120
  sorry

end smallest_n_divisible_by_18_and_640_l2002_200285


namespace fliers_remaining_l2002_200222

theorem fliers_remaining (total : ℕ) (morning_fraction : ℚ) (afternoon_fraction : ℚ) : 
  total = 2000 →
  morning_fraction = 1 / 10 →
  afternoon_fraction = 1 / 4 →
  (total - total * morning_fraction) * (1 - afternoon_fraction) = 1350 := by
sorry

end fliers_remaining_l2002_200222


namespace melies_money_left_l2002_200289

/-- Calculates the amount of money left after buying meat. -/
def money_left (meat_amount : ℝ) (cost_per_kg : ℝ) (initial_money : ℝ) : ℝ :=
  initial_money - meat_amount * cost_per_kg

/-- Proves that Méliès has $16 left after buying meat. -/
theorem melies_money_left :
  let meat_amount : ℝ := 2
  let cost_per_kg : ℝ := 82
  let initial_money : ℝ := 180
  money_left meat_amount cost_per_kg initial_money = 16 := by
  sorry

end melies_money_left_l2002_200289


namespace perimeter_ratio_triangles_l2002_200231

theorem perimeter_ratio_triangles :
  let small_triangle_sides : Fin 3 → ℝ := ![4, 8, 4 * Real.sqrt 3]
  let large_triangle_sides : Fin 3 → ℝ := ![8, 8, 8 * Real.sqrt 2]
  let small_perimeter := (Finset.univ.sum small_triangle_sides)
  let large_perimeter := (Finset.univ.sum large_triangle_sides)
  small_perimeter / large_perimeter = (4 + 8 + 4 * Real.sqrt 3) / (8 + 8 + 8 * Real.sqrt 2) := by
  sorry

end perimeter_ratio_triangles_l2002_200231


namespace largest_value_l2002_200254

def expr_a : ℤ := 2 * 0 * 2006
def expr_b : ℤ := 2 * 0 + 6
def expr_c : ℤ := 2 + 0 * 2006
def expr_d : ℤ := 2 * (0 + 6)
def expr_e : ℤ := 2006 * 0 + 0 * 6

theorem largest_value : 
  expr_d = max expr_a (max expr_b (max expr_c (max expr_d expr_e))) :=
by sorry

end largest_value_l2002_200254


namespace polynomial_identity_l2002_200266

theorem polynomial_identity (x : ℝ) (h : x + 1/x = 3) :
  x^12 - 7*x^6 + x^2 = 45363*x - 17327 := by
  sorry

end polynomial_identity_l2002_200266


namespace tangent_parallel_to_x_axis_l2002_200256

/-- A curve defined by y = kx + ln x has a tangent at the point (1, k) that is parallel to the x-axis if and only if k = -1 -/
theorem tangent_parallel_to_x_axis (k : ℝ) : 
  (∃ f : ℝ → ℝ, f x = k * x + Real.log x) →
  (∃ t : ℝ → ℝ, t x = k * x + Real.log 1) →
  (∀ x : ℝ, (k + 1 / x) = 0) →
  k = -1 := by
  sorry

end tangent_parallel_to_x_axis_l2002_200256


namespace sum_of_prime_factors_143_l2002_200203

def sum_of_prime_factors (n : ℕ) : ℕ := sorry

theorem sum_of_prime_factors_143 : sum_of_prime_factors 143 = 24 := by sorry

end sum_of_prime_factors_143_l2002_200203


namespace simultaneous_inequalities_solution_l2002_200236

theorem simultaneous_inequalities_solution (x : ℝ) :
  (x^2 - 8*x + 12 < 0 ∧ 2*x - 4 > 0) ↔ (x > 2 ∧ x < 6) :=
by sorry

end simultaneous_inequalities_solution_l2002_200236


namespace rational_sum_theorem_l2002_200243

theorem rational_sum_theorem (a b c : ℚ) 
  (h1 : a * b * c < 0) 
  (h2 : a + b + c = 0) : 
  (a - b - c) / abs a + (b - c - a) / abs b + (c - a - b) / abs c = 2 := by
  sorry

end rational_sum_theorem_l2002_200243


namespace product_increased_by_four_l2002_200207

theorem product_increased_by_four (x : ℝ) (h : x = 3) : 5 * x + 4 = 19 := by
  sorry

end product_increased_by_four_l2002_200207


namespace exponent_division_l2002_200239

theorem exponent_division (x : ℝ) : x^6 / x^2 = x^4 := by
  sorry

end exponent_division_l2002_200239


namespace unique_function_satisfying_conditions_l2002_200255

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + x + 1

-- State the theorem
theorem unique_function_satisfying_conditions :
  (∀ x y : ℝ, f (x^2) = (f x)^2 - 2*x*(f x)) ∧
  (∀ x : ℝ, f (-x) = f (x - 1)) ∧
  (∀ x y : ℝ, 1 < x → x < y → f x < f y) ∧
  (∀ x : ℝ, 0 < f x) ∧
  (∀ g : ℝ → ℝ, 
    ((∀ x y : ℝ, g (x^2) = (g x)^2 - 2*x*(g x)) ∧
     (∀ x : ℝ, g (-x) = g (x - 1)) ∧
     (∀ x y : ℝ, 1 < x → x < y → g x < g y) ∧
     (∀ x : ℝ, 0 < g x)) →
    (∀ x : ℝ, g x = f x)) :=
by sorry


end unique_function_satisfying_conditions_l2002_200255


namespace expression_value_l2002_200273

theorem expression_value : ∀ x : ℝ, x ≠ 5 →
  (x^2 - 3*x - 10) / (x - 5) = x + 2 ∧
  ((1^2 - 3*1 - 10) / (1 - 5) = 3) :=
by sorry

end expression_value_l2002_200273


namespace perpendicular_vectors_imply_t_equals_one_l2002_200286

/-- Given vectors a, b, and c in ℝ², prove that if a + 2b is perpendicular to c, then t = 1 -/
theorem perpendicular_vectors_imply_t_equals_one (a b c : ℝ × ℝ) :
  a = (Real.sqrt 3, 1) →
  b = (0, 1) →
  c = (-Real.sqrt 3, t) →
  (a.1 + 2 * b.1, a.2 + 2 * b.2) • c = 0 →
  t = 1 := by
  sorry

#check perpendicular_vectors_imply_t_equals_one

end perpendicular_vectors_imply_t_equals_one_l2002_200286


namespace polygon_sides_count_l2002_200299

theorem polygon_sides_count (n : ℕ) (k : ℕ) (r : ℚ) : 
  k = n * (n - 3) / 2 →
  k = r * n →
  r = 3 / 2 →
  n = 6 := by
sorry

end polygon_sides_count_l2002_200299


namespace symmetry_sum_l2002_200257

/-- Two points are symmetric about the x-axis if they have the same x-coordinate and opposite y-coordinates -/
def symmetric_about_x_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = p2.1 ∧ p1.2 = -p2.2

theorem symmetry_sum (x y : ℝ) :
  symmetric_about_x_axis (-2, y) (x, 3) → x + y = -5 := by
  sorry

end symmetry_sum_l2002_200257


namespace deepak_age_l2002_200250

theorem deepak_age (arun_age deepak_age : ℕ) : 
  (arun_age : ℚ) / deepak_age = 4 / 3 →
  arun_age + 6 = 26 →
  deepak_age = 15 := by
sorry

end deepak_age_l2002_200250


namespace prob_six_consecutive_heads_l2002_200293

/-- A fair coin is flipped 10 times. -/
def coin_flips : ℕ := 10

/-- The probability of getting heads on a single flip of a fair coin. -/
def prob_heads : ℚ := 1/2

/-- The set of all possible outcomes when flipping a coin 10 times. -/
def all_outcomes : Finset (Fin coin_flips → Bool) := sorry

/-- The set of outcomes with at least 6 consecutive heads. -/
def outcomes_with_six_consecutive_heads : Finset (Fin coin_flips → Bool) := sorry

/-- The probability of getting at least 6 consecutive heads in 10 flips of a fair coin. -/
theorem prob_six_consecutive_heads :
  (Finset.card outcomes_with_six_consecutive_heads : ℚ) / (Finset.card all_outcomes : ℚ) = 129/1024 :=
sorry

end prob_six_consecutive_heads_l2002_200293


namespace mary_potatoes_l2002_200242

theorem mary_potatoes (initial_potatoes : ℕ) : 
  initial_potatoes - 3 = 5 → initial_potatoes = 8 := by
  sorry

end mary_potatoes_l2002_200242


namespace solve_equation_l2002_200294

theorem solve_equation (b : ℚ) (h : b + b/4 = 10/4) : b = 2 := by
  sorry

end solve_equation_l2002_200294


namespace football_field_area_l2002_200275

theorem football_field_area (total_fertilizer : ℝ) (partial_fertilizer : ℝ) (partial_area : ℝ) :
  total_fertilizer = 1200 →
  partial_fertilizer = 400 →
  partial_area = 3600 →
  (total_fertilizer / (partial_fertilizer / partial_area)) = 10800 := by
  sorry

end football_field_area_l2002_200275


namespace certain_number_is_fourteen_l2002_200296

/-- A certain number multiplied by d is the square of an integer -/
def is_square_multiple (n : ℕ) (d : ℕ) : Prop :=
  ∃ m : ℕ, n * d = m^2

/-- d is the smallest positive integer satisfying the condition -/
def is_smallest_d (n : ℕ) (d : ℕ) : Prop :=
  is_square_multiple n d ∧ ∀ k < d, ¬(is_square_multiple n k)

theorem certain_number_is_fourteen (d : ℕ) (h1 : d = 14) 
  (h2 : ∃ n : ℕ, is_smallest_d n d) : 
  ∃ n : ℕ, is_smallest_d n d ∧ n = 14 :=
sorry

end certain_number_is_fourteen_l2002_200296


namespace ellipse_standard_equation_l2002_200263

/-- The standard equation of an ellipse with given minor axis length and eccentricity -/
theorem ellipse_standard_equation (b : ℝ) (e : ℝ) : 
  b = 4 ∧ e = 3/5 → 
  ∃ (a : ℝ), (a > b) ∧ 
  ((∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 ↔ (x^2/25 + y^2/16 = 1 ∨ x^2/16 + y^2/25 = 1)) ∧
   e^2 = 1 - b^2/a^2) :=
by sorry

end ellipse_standard_equation_l2002_200263


namespace mineral_age_arrangements_eq_60_l2002_200240

/-- The number of arrangements for a six-digit number using 2, 2, 4, 4, 7, 9, starting with an odd digit -/
def mineral_age_arrangements : ℕ :=
  let digits : List ℕ := [2, 2, 4, 4, 7, 9]
  let odd_digits : List ℕ := digits.filter (λ d => d % 2 = 1)
  let remaining_digits : ℕ := digits.length - 1
  let repeated_digits : List ℕ := [2, 4]
  odd_digits.length * (remaining_digits.factorial / (repeated_digits.map (λ d => (digits.count d).factorial)).prod)

/-- Theorem stating that the number of possible arrangements is 60 -/
theorem mineral_age_arrangements_eq_60 : mineral_age_arrangements = 60 := by
  sorry

end mineral_age_arrangements_eq_60_l2002_200240


namespace adjacent_repeat_percentage_is_16_l2002_200208

/-- The count of three-digit numbers -/
def three_digit_count : ℕ := 900

/-- The count of three-digit numbers with adjacent repeated digits -/
def adjacent_repeat_count : ℕ := 144

/-- The percentage of three-digit numbers with adjacent repeated digits -/
def adjacent_repeat_percentage : ℚ := adjacent_repeat_count / three_digit_count * 100

/-- Theorem stating that the percentage of three-digit numbers with adjacent repeated digits is 16.0% -/
theorem adjacent_repeat_percentage_is_16 :
  ⌊adjacent_repeat_percentage * 10⌋ / 10 = 16 :=
sorry

end adjacent_repeat_percentage_is_16_l2002_200208


namespace no_valid_tiling_l2002_200221

/-- Represents a tile on the grid -/
inductive Tile
  | OneByFour : Tile
  | TwoByTwo : Tile

/-- Represents a position on the 8x8 grid -/
structure Position :=
  (row : Fin 8)
  (col : Fin 8)

/-- Represents a placement of a tile on the grid -/
structure Placement :=
  (tile : Tile)
  (position : Position)

/-- Represents a tiling of the 8x8 grid -/
def Tiling := List Placement

/-- Checks if a tiling is valid (covers the entire grid without overlaps) -/
def isValidTiling (t : Tiling) : Prop := sorry

/-- Checks if a tiling uses exactly 15 1x4 tiles and 1 2x2 tile -/
def hasCorrectTileCount (t : Tiling) : Prop := sorry

/-- The main theorem stating that no valid tiling exists with the given constraints -/
theorem no_valid_tiling :
  ¬ ∃ (t : Tiling), isValidTiling t ∧ hasCorrectTileCount t := by
  sorry

end no_valid_tiling_l2002_200221


namespace order_of_abc_l2002_200206

theorem order_of_abc : 
  let a : ℝ := 1 / (6 * Real.sqrt 15)
  let b : ℝ := (3/4) * Real.sin (1/60)
  let c : ℝ := Real.log (61/60)
  b < c ∧ c < a := by
  sorry

end order_of_abc_l2002_200206


namespace tv_price_calculation_l2002_200272

/-- The actual selling price of a television set given its cost price,
    markup percentage, and discount percentage. -/
def actual_selling_price (cost_price : ℝ) (markup_percent : ℝ) (discount_percent : ℝ) : ℝ :=
  cost_price * (1 + markup_percent) * discount_percent

/-- Theorem stating that for a television with cost price 'a',
    25% markup, and 70% discount, the actual selling price is 70%(1+25%)a. -/
theorem tv_price_calculation (a : ℝ) :
  actual_selling_price a 0.25 0.7 = 0.7 * (1 + 0.25) * a := by
  sorry

#check tv_price_calculation

end tv_price_calculation_l2002_200272


namespace atticus_marbles_l2002_200279

theorem atticus_marbles (a j c : ℕ) : 
  3 * (a + j + c) = 60 →
  a = j / 2 →
  c = 8 →
  a = 4 := by
sorry

end atticus_marbles_l2002_200279


namespace sarah_trucks_l2002_200232

/-- The number of trucks Sarah had initially -/
def initial_trucks : ℕ := 51

/-- The number of trucks Sarah gave to Jeff -/
def trucks_given : ℕ := 13

/-- The number of trucks Sarah has now -/
def remaining_trucks : ℕ := initial_trucks - trucks_given

theorem sarah_trucks : remaining_trucks = 38 := by
  sorry

end sarah_trucks_l2002_200232


namespace f_range_on_domain_l2002_200205

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x

-- Define the domain
def domain : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem f_range_on_domain :
  ∃ (min max : ℝ), min = -1 ∧ max = 8 ∧
  (∀ x ∈ domain, min ≤ f x ∧ f x ≤ max) ∧
  (∃ x₁ ∈ domain, f x₁ = min) ∧
  (∃ x₂ ∈ domain, f x₂ = max) :=
sorry

end f_range_on_domain_l2002_200205


namespace hyperbola_equation_l2002_200281

/-- Given an ellipse and a hyperbola with specific properties, prove the equation of the hyperbola -/
theorem hyperbola_equation (a b c m n c' : ℝ) (e e' : ℝ) : 
  (∀ x y : ℝ, 2 * x^2 + y^2 = 2) →  -- Ellipse equation
  (a^2 = 2 ∧ b^2 = 1) →             -- Semi-major and semi-minor axes of ellipse
  (c = (a^2 - b^2).sqrt) →          -- Focal length of ellipse
  (e = c / a) →                     -- Eccentricity of ellipse
  (m = a) →                         -- Semi-major axis of hyperbola
  (e' * e = 1) →                    -- Product of eccentricities
  (c' = m * e') →                   -- Focal length of hyperbola
  (n^2 = c'^2 - m^2) →              -- Semi-minor axis of hyperbola
  (∀ x y : ℝ, y^2 / n^2 - x^2 / m^2 = 1) →  -- Standard form of hyperbola
  (∀ x y : ℝ, y^2 - x^2 = 2) :=     -- Desired hyperbola equation
by sorry

end hyperbola_equation_l2002_200281


namespace cycle_gain_percentage_l2002_200274

/-- Calculate the overall gain percentage for three cycles given their purchase and sale prices -/
theorem cycle_gain_percentage
  (purchase_a purchase_b purchase_c : ℕ)
  (sale_a sale_b sale_c : ℕ)
  (h_purchase_a : purchase_a = 1000)
  (h_purchase_b : purchase_b = 3000)
  (h_purchase_c : purchase_c = 6000)
  (h_sale_a : sale_a = 2000)
  (h_sale_b : sale_b = 4500)
  (h_sale_c : sale_c = 8000) :
  (((sale_a + sale_b + sale_c) - (purchase_a + purchase_b + purchase_c)) * 100) / (purchase_a + purchase_b + purchase_c) = 45 := by
  sorry


end cycle_gain_percentage_l2002_200274


namespace correct_algorithm_statement_l2002_200215

-- Define the concept of an algorithm
def Algorithm : Type := Unit

-- Define the property of being correct for an algorithm
def is_correct (a : Algorithm) : Prop := sorry

-- Define the property of yielding a definite result
def yields_definite_result (a : Algorithm) : Prop := sorry

-- Define the property of ending within a finite number of steps
def ends_in_finite_steps (a : Algorithm) : Prop := sorry

-- Define the property of having clear and unambiguous steps
def has_clear_steps (a : Algorithm) : Prop := sorry

-- Define the property of being unique for solving a certain type of problem
def is_unique_for_problem (a : Algorithm) : Prop := sorry

-- Theorem stating that the only correct statement is (2)
theorem correct_algorithm_statement :
  ∀ (a : Algorithm),
    is_correct a →
    (yields_definite_result a ∧
    ¬(¬(has_clear_steps a)) ∧
    ¬(is_unique_for_problem a) ∧
    ¬(ends_in_finite_steps a)) :=
by sorry

end correct_algorithm_statement_l2002_200215
