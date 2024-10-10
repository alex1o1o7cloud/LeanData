import Mathlib

namespace mixture_weight_l416_41619

/-- Given substances a and b mixed in a ratio of 9:11, prove that the total weight
    of the mixture is 58 kg when 26.1 kg of substance a is used. -/
theorem mixture_weight (a b : ℝ) (h1 : a / b = 9 / 11) (h2 : a = 26.1) :
  a + b = 58 := by
  sorry

end mixture_weight_l416_41619


namespace inequality_implication_l416_41681

theorem inequality_implication (a b : ℝ) (h : a < b) : a - b < 0 := by
  sorry

end inequality_implication_l416_41681


namespace candy_box_max_money_l416_41660

/-- Calculates the maximum amount of money that can be made by selling boxed candies. -/
def max_money (total_candies : ℕ) (candies_per_box : ℕ) (price_per_box : ℕ) : ℕ :=
  (total_candies / candies_per_box) * price_per_box

/-- Theorem stating the maximum amount of money for the given candy problem. -/
theorem candy_box_max_money :
  max_money 235 10 3000 = 69000 := by
  sorry

end candy_box_max_money_l416_41660


namespace triangle_properties_l416_41617

/-- Given a triangle ABC with the following properties:
    - Sides a, b, c are opposite to angles A, B, C respectively
    - Vector m = (2 * sin B, -√3)
    - Vector n = (cos(2B), 2 * cos²(B/2) - 1)
    - m is parallel to n
    - B is an acute angle
    - b = 2
    Prove that the measure of angle B is π/3 and the maximum area of the triangle is √3 -/
theorem triangle_properties (A B C : ℝ) (a b c : ℝ) 
  (m : ℝ × ℝ) (n : ℝ × ℝ) :
  m.1 = 2 * Real.sin B ∧ 
  m.2 = -Real.sqrt 3 ∧
  n.1 = Real.cos (2 * B) ∧ 
  n.2 = 2 * (Real.cos (B / 2))^2 - 1 ∧
  ∃ (k : ℝ), m = k • n ∧
  0 < B ∧ B < π / 2 ∧
  b = 2 →
  B = π / 3 ∧ 
  (∀ (S : ℝ), S = 1/2 * a * c * Real.sin B → S ≤ Real.sqrt 3) :=
by sorry

end triangle_properties_l416_41617


namespace final_class_size_l416_41602

theorem final_class_size (initial_size second_year_join final_year_leave : ℕ) :
  initial_size = 150 →
  second_year_join = 30 →
  final_year_leave = 15 →
  initial_size + second_year_join - final_year_leave = 165 := by
  sorry

end final_class_size_l416_41602


namespace bond_interest_rate_proof_l416_41691

/-- Proves that the interest rate of a bond is 5.75% given specific investment conditions -/
theorem bond_interest_rate_proof (total_investment : ℝ) (unknown_bond_investment : ℝ) 
  (known_bond_investment : ℝ) (known_interest_rate : ℝ) (desired_interest_income : ℝ) :
  total_investment = 32000 →
  unknown_bond_investment = 20000 →
  known_bond_investment = 12000 →
  known_interest_rate = 0.0625 →
  desired_interest_income = 1900 →
  ∃ unknown_interest_rate : ℝ,
    unknown_interest_rate = 0.0575 ∧
    desired_interest_income = unknown_bond_investment * unknown_interest_rate + 
                              known_bond_investment * known_interest_rate :=
by sorry

end bond_interest_rate_proof_l416_41691


namespace sequence_product_l416_41601

theorem sequence_product (a : ℕ → ℝ) (h1 : ∀ n, a (n - 1) = 2 * a n) (h2 : a 5 = 4) :
  a 4 * a 5 * a 6 = 64 := by
  sorry

end sequence_product_l416_41601


namespace six_ways_to_make_50_yuan_l416_41635

/-- The number of ways to make 50 yuan using 5 yuan and 10 yuan notes -/
def ways_to_make_50_yuan : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 5 * p.1 + 10 * p.2 = 50) (Finset.product (Finset.range 11) (Finset.range 6))).card

/-- Theorem stating that there are exactly 6 ways to make 50 yuan using 5 yuan and 10 yuan notes -/
theorem six_ways_to_make_50_yuan : ways_to_make_50_yuan = 6 := by
  sorry

end six_ways_to_make_50_yuan_l416_41635


namespace blue_balls_count_l416_41636

theorem blue_balls_count (total : ℕ) (green blue yellow white : ℕ) : 
  green = total / 4 →
  blue = total / 8 →
  yellow = total / 12 →
  white = 26 →
  total = green + blue + yellow + white →
  blue = 6 := by
sorry

end blue_balls_count_l416_41636


namespace wage_decrease_increase_l416_41624

theorem wage_decrease_increase (original : ℝ) (h : original > 0) :
  let decreased := original * 0.5
  let increased := decreased * 1.5
  increased = original * 0.75 :=
by sorry

end wage_decrease_increase_l416_41624


namespace boat_speed_in_still_water_l416_41682

/-- Proves that the speed of a boat in still water is 20 km/hr given the specified conditions -/
theorem boat_speed_in_still_water :
  ∀ (boat_speed : ℝ),
    (boat_speed + 5) * 0.4 = 10 →
    boat_speed = 20 := by
  sorry

end boat_speed_in_still_water_l416_41682


namespace geometric_sequence_tangent_l416_41653

open Real

theorem geometric_sequence_tangent (x : ℝ) : 
  (∃ (r : ℝ), (tan (π/12 - x) = tan (π/12) * r ∧ tan (π/12) = tan (π/12 + x) * r) ∨
               (tan (π/12 - x) = tan (π/12 + x) * r ∧ tan (π/12) = tan (π/12 - x) * r) ∨
               (tan (π/12) = tan (π/12 - x) * r ∧ tan (π/12 + x) = tan (π/12) * r)) ↔ 
  (∃ (ε : ℤ) (n : ℤ), ε ∈ ({-1, 0, 1} : Set ℤ) ∧ x = ε * (π/3) + n * π) :=
sorry

end geometric_sequence_tangent_l416_41653


namespace equilateral_triangle_area_decrease_l416_41684

/-- Prove that for an equilateral triangle with an area of 100√3 cm², 
    if each side is decreased by 6 cm, the decrease in area is 51√3 cm². -/
theorem equilateral_triangle_area_decrease 
  (original_area : ℝ) 
  (side_decrease : ℝ) :
  original_area = 100 * Real.sqrt 3 →
  side_decrease = 6 →
  let original_side := Real.sqrt ((4 * original_area) / Real.sqrt 3)
  let new_side := original_side - side_decrease
  let new_area := (new_side^2 * Real.sqrt 3) / 4
  original_area - new_area = 51 * Real.sqrt 3 := by
sorry


end equilateral_triangle_area_decrease_l416_41684


namespace function_inequality_implies_m_range_l416_41600

theorem function_inequality_implies_m_range (m : ℝ) : 
  (∀ x : ℝ, (1/2) * x^4 - 2 * x^3 + 3 * m + 6 ≥ 0) → m ≥ 5/2 := by
  sorry

end function_inequality_implies_m_range_l416_41600


namespace patio_perimeter_l416_41656

/-- A rectangular patio with length 40 feet and width equal to one-fourth of its length has a perimeter of 100 feet. -/
theorem patio_perimeter : 
  ∀ (length width : ℝ), 
  length = 40 → 
  width = length / 4 → 
  2 * length + 2 * width = 100 := by
sorry

end patio_perimeter_l416_41656


namespace min_value_expression_l416_41677

theorem min_value_expression (a b c k : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_eq : a = k ∧ b = k ∧ c = k) : 
  (a + b + c) * ((a + b)⁻¹ + (a + c)⁻¹ + (b + c)⁻¹) = 9 / 2 := by
  sorry

end min_value_expression_l416_41677


namespace characterize_satisfying_functions_l416_41611

/-- A function satisfying the given inequality -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) + y ≤ f (f (f x))

/-- The main theorem stating the form of functions satisfying the inequality -/
theorem characterize_satisfying_functions :
  ∀ f : ℝ → ℝ, SatisfiesInequality f →
  ∃ C : ℝ, ∀ x : ℝ, f x = -x + C :=
by sorry

end characterize_satisfying_functions_l416_41611


namespace biology_score_calculation_l416_41605

def math_score : ℕ := 76
def science_score : ℕ := 65
def social_studies_score : ℕ := 82
def english_score : ℕ := 67
def average_score : ℕ := 69
def total_subjects : ℕ := 5

theorem biology_score_calculation :
  let known_subjects_total := math_score + science_score + social_studies_score + english_score
  let all_subjects_total := average_score * total_subjects
  all_subjects_total - known_subjects_total = 55 := by
sorry

end biology_score_calculation_l416_41605


namespace functional_inequality_solution_l416_41613

open Real

-- Define the function type
def ContinuousRealFunction := {f : ℝ → ℝ // Continuous f}

-- State the theorem
theorem functional_inequality_solution 
  (f : ContinuousRealFunction) 
  (h1 : f.val 0 = 0) 
  (h2 : ∀ x y : ℝ, f.val ((x + y) / (1 + x * y)) ≥ f.val x + f.val y) :
  ∃ c : ℝ, ∀ x : ℝ, f.val x = (c / 2) * log (abs ((x + 1) / (x - 1))) :=
sorry

end functional_inequality_solution_l416_41613


namespace least_number_with_divisibility_property_l416_41685

theorem least_number_with_divisibility_property : ∃ m : ℕ, 
  (m > 0) ∧ 
  (∀ n : ℕ, n > 0 → n < m → ¬(∃ q r : ℕ, n = 5 * q ∧ n = 34 * (q - 8) + r ∧ r < 34)) ∧
  (∃ q r : ℕ, m = 5 * q ∧ m = 34 * (q - 8) + r ∧ r < 34) ∧
  m = 162 :=
by sorry

end least_number_with_divisibility_property_l416_41685


namespace connie_watch_savings_l416_41639

/-- The amount of money Connie needs to buy a watch -/
theorem connie_watch_savings (saved : ℕ) (watch_cost : ℕ) (h1 : saved = 39) (h2 : watch_cost = 55) :
  watch_cost - saved = 16 := by
  sorry

end connie_watch_savings_l416_41639


namespace baseball_season_games_l416_41609

/-- Calculates the total number of games played in a baseball season given the number of wins and a relationship between wins and losses. -/
theorem baseball_season_games (wins losses : ℕ) : 
  wins = 101 ∧ wins = 3 * losses + 14 → wins + losses = 130 :=
by sorry

end baseball_season_games_l416_41609


namespace range_of_a_l416_41638

-- Define sets A and B
def A : Set ℝ := {x | |x - 1| ≤ 2}
def B (a : ℝ) : Set ℝ := {x | x - a > 0}

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (A ∪ B a = B a) → a < -1 :=
by sorry

end range_of_a_l416_41638


namespace parallel_vectors_magnitude_l416_41622

/-- Given two parallel vectors a and b, prove that the magnitude of 3a + 2b is √5 -/
theorem parallel_vectors_magnitude (a b : ℝ × ℝ) : 
  a = (1, -2) → 
  b.1 = -2 → 
  ∃ y, b.2 = y → 
  (∃ k : ℝ, k ≠ 0 ∧ a = k • b) → 
  ‖3 • a + 2 • b‖ = Real.sqrt 5 := by
  sorry

end parallel_vectors_magnitude_l416_41622


namespace sum_squares_products_bound_l416_41678

theorem sum_squares_products_bound (a b c d : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : d ≥ 0) (h5 : a * b * c * d = 1) :
  a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d ≥ 10 := by
sorry

end sum_squares_products_bound_l416_41678


namespace percentage_of_filled_seats_l416_41628

/-- Given a hall with 600 seats and 240 vacant seats, prove that 60% of the seats were filled. -/
theorem percentage_of_filled_seats (total_seats : ℕ) (vacant_seats : ℕ) : 
  total_seats = 600 → vacant_seats = 240 → 
  (((total_seats - vacant_seats : ℚ) / total_seats) * 100 = 60) := by
  sorry

end percentage_of_filled_seats_l416_41628


namespace average_age_increase_l416_41603

theorem average_age_increase (num_students : ℕ) (student_avg_age : ℝ) (teacher_age : ℝ) :
  num_students = 23 →
  student_avg_age = 22 →
  teacher_age = 46 →
  ((num_students : ℝ) * student_avg_age + teacher_age) / ((num_students : ℝ) + 1) - student_avg_age = 1 :=
by sorry

end average_age_increase_l416_41603


namespace farmhand_work_hours_l416_41657

/-- Represents the number of golden delicious apples needed for one pint of cider -/
def golden_delicious_per_pint : ℕ := 20

/-- Represents the number of pink lady apples needed for one pint of cider -/
def pink_lady_per_pint : ℕ := 40

/-- Represents the number of farmhands -/
def num_farmhands : ℕ := 6

/-- Represents the number of apples each farmhand can pick per hour -/
def apples_per_hour_per_farmhand : ℕ := 240

/-- Represents the ratio of golden delicious to pink lady apples -/
def apple_ratio : Rat := 1 / 2

/-- Represents the number of pints of cider Haley can make with the gathered apples -/
def pints_of_cider : ℕ := 120

/-- Theorem stating that the farmhands will work for 5 hours -/
theorem farmhand_work_hours : 
  ∃ (hours : ℕ), 
    hours = 5 ∧ 
    hours * (num_farmhands * apples_per_hour_per_farmhand) = 
      pints_of_cider * (golden_delicious_per_pint + pink_lady_per_pint) ∧
    apple_ratio = (pints_of_cider * golden_delicious_per_pint : ℚ) / 
                  (pints_of_cider * pink_lady_per_pint : ℚ) := by
  sorry

end farmhand_work_hours_l416_41657


namespace cookie_consumption_l416_41690

theorem cookie_consumption (total cookies_left father_ate : ℕ) 
  (h1 : total = 30)
  (h2 : cookies_left = 8)
  (h3 : father_ate = 10) :
  let mother_ate := father_ate / 2
  let total_eaten := total - cookies_left
  let brother_ate := total_eaten - (father_ate + mother_ate)
  brother_ate - mother_ate = 2 := by sorry

end cookie_consumption_l416_41690


namespace sum_of_absolute_coefficients_l416_41689

theorem sum_of_absolute_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℤ) : 
  (∀ x : ℤ, (2 - x)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| = 665 := by
sorry

end sum_of_absolute_coefficients_l416_41689


namespace parabola_symmetry_problem_l416_41608

/-- Represents a point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y = 2 * x^2

/-- The problem statement -/
theorem parabola_symmetry_problem (A B : ParabolaPoint) (m : ℝ) 
  (h_symmetric : ∃ (t : ℝ), (A.x + B.x) / 2 = t ∧ (A.y + B.y) / 2 = t + m)
  (h_product : A.x * B.x = -1/2) :
  m = 3/2 := by
  sorry

end parabola_symmetry_problem_l416_41608


namespace g_value_at_negative_1001_l416_41695

/-- A function g satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g (x * y) + 2 * x = x * g y + g x

theorem g_value_at_negative_1001 (g : ℝ → ℝ) 
    (h1 : FunctionalEquation g) (h2 : g 1 = 3) : g (-1001) = 1005 := by
  sorry

end g_value_at_negative_1001_l416_41695


namespace megans_books_l416_41650

theorem megans_books (books_per_shelf : ℕ) (mystery_shelves : ℕ) (picture_shelves : ℕ) 
  (h1 : books_per_shelf = 7)
  (h2 : mystery_shelves = 8)
  (h3 : picture_shelves = 2) :
  books_per_shelf * (mystery_shelves + picture_shelves) = 70 :=
by sorry

end megans_books_l416_41650


namespace class_size_proof_l416_41655

/-- The number of students in a class with English and German courses -/
def class_size (english_only german_only both : ℕ) : ℕ :=
  english_only + german_only + both

theorem class_size_proof (english_only german_only both : ℕ) 
  (h1 : both = 12)
  (h2 : german_only + both = 22)
  (h3 : english_only = 30) :
  class_size english_only german_only both = 52 := by
  sorry

#check class_size_proof

end class_size_proof_l416_41655


namespace athletes_seating_arrangements_l416_41643

def number_of_arrangements (team_sizes : List Nat) : Nat :=
  (team_sizes.length.factorial) * (team_sizes.map Nat.factorial).prod

theorem athletes_seating_arrangements :
  number_of_arrangements [4, 3, 3] = 5184 := by
  sorry

end athletes_seating_arrangements_l416_41643


namespace sum_of_cubes_l416_41652

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 15) : x^3 + y^3 = 550 := by
  sorry

end sum_of_cubes_l416_41652


namespace modular_inverse_31_mod_35_l416_41697

theorem modular_inverse_31_mod_35 : ∃ x : ℕ, x ≤ 34 ∧ (31 * x) % 35 = 1 :=
by
  -- The proof goes here
  sorry

end modular_inverse_31_mod_35_l416_41697


namespace remaining_tickets_l416_41632

def tickets_from_whack_a_mole : ℕ := 32
def tickets_from_skee_ball : ℕ := 25
def tickets_spent_on_hat : ℕ := 7

theorem remaining_tickets :
  tickets_from_whack_a_mole + tickets_from_skee_ball - tickets_spent_on_hat = 50 := by
  sorry

end remaining_tickets_l416_41632


namespace binary_arithmetic_equality_l416_41654

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldr (λ (i, bit) acc => acc + if bit then 2^i else 0) 0

def decimal_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
    let rec to_binary_aux (m : ℕ) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: to_binary_aux (m / 2)
    (to_binary_aux n).reverse

theorem binary_arithmetic_equality :
  let a := binary_to_decimal [true, false, true, true]  -- 1101₂
  let b := binary_to_decimal [false, true, true]        -- 110₂
  let c := binary_to_decimal [false, true, true, true]  -- 1110₂
  let d := binary_to_decimal [true, true, true, true]   -- 1111₂
  let result := decimal_to_binary (a + b - c + d)
  result = [false, true, false, false, false, true]     -- 100010₂
:= by sorry

end binary_arithmetic_equality_l416_41654


namespace general_solution_zero_a_case_degenerate_case_l416_41615

-- Define the system of equations
def system (a b c x y z : ℝ) : Prop :=
  a * x + b * y - c * z = a * b ∧
  3 * a * x - b * y + 2 * c * z = a * (5 * c - b) ∧
  3 * y + 2 * z = 5 * a

-- Theorem for the general solution
theorem general_solution (a b c : ℝ) :
  ∃ x y z, system a b c x y z ∧ x = c ∧ y = a ∧ z = a :=
sorry

-- Theorem for the case when a = 0
theorem zero_a_case (b c : ℝ) :
  ∃ x y z, system 0 b c x y z ∧ y = 0 ∧ z = 0 :=
sorry

-- Theorem for the case when 8b + 15c = 0
theorem degenerate_case (a b : ℝ) :
  8 * b + 15 * (-8 * b / 15) = 0 →
  ∃ x y, ∀ z, system a b (-8 * b / 15) x y z :=
sorry

end general_solution_zero_a_case_degenerate_case_l416_41615


namespace negation_of_divisible_by_5_is_odd_l416_41623

theorem negation_of_divisible_by_5_is_odd :
  (¬ ∀ n : ℤ, n % 5 = 0 → Odd n) ↔ (∃ n : ℤ, n % 5 = 0 ∧ ¬ Odd n) := by
  sorry

end negation_of_divisible_by_5_is_odd_l416_41623


namespace distance_between_points_l416_41671

def point1 : ℝ × ℝ := (3, 7)
def point2 : ℝ × ℝ := (3, -4)

theorem distance_between_points : 
  |point1.2 - point2.2| = 11 := by
  sorry

end distance_between_points_l416_41671


namespace min_value_of_expression_limit_at_one_l416_41606

open Real

theorem min_value_of_expression (x : ℝ) (h1 : -3 < x) (h2 : x < 2) (h3 : x ≠ 1) :
  (x^2 - 4*x + 5) / (3*x - 3) ≥ 2/3 :=
sorry

theorem limit_at_one :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ →
    |(x^2 - 4*x + 5) / (3*x - 3) - 2/3| < ε :=
sorry

end min_value_of_expression_limit_at_one_l416_41606


namespace download_time_proof_l416_41648

def internet_speed : ℝ := 2
def file1_size : ℝ := 80
def file2_size : ℝ := 90
def file3_size : ℝ := 70
def minutes_per_hour : ℝ := 60

theorem download_time_proof :
  let total_size := file1_size + file2_size + file3_size
  let download_time_minutes := total_size / internet_speed
  let download_time_hours := download_time_minutes / minutes_per_hour
  download_time_hours = 2 := by
sorry

end download_time_proof_l416_41648


namespace stationery_problem_solution_l416_41686

/-- Represents a box of stationery --/
structure StationeryBox where
  sheets : ℕ
  envelopes : ℕ

/-- Defines the conditions of the stationery problem --/
def stationeryProblem (box : StationeryBox) : Prop :=
  -- Tom's condition: all envelopes used, 100 sheets left
  box.sheets - box.envelopes = 100 ∧
  -- Jerry's condition: all sheets used, 25 envelopes left
  box.envelopes + 25 = box.sheets / 3

/-- The theorem stating the solution to the stationery problem --/
theorem stationery_problem_solution :
  ∃ (box : StationeryBox), stationeryProblem box ∧ box.sheets = 120 :=
sorry

end stationery_problem_solution_l416_41686


namespace increasing_function_inequality_l416_41607

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the property of f being increasing on ℝ
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Theorem statement
theorem increasing_function_inequality (h_incr : IsIncreasing f) (m : ℝ) :
  f (2 * m) > f (-m + 9) → m > 3 := by
  sorry


end increasing_function_inequality_l416_41607


namespace ellas_food_calculation_l416_41641

/-- The amount of food Ella eats each day, in pounds -/
def ellas_daily_food : ℝ := 20

/-- The number of days considered -/
def days : ℕ := 10

/-- The total amount of food Ella and her dog eat in the given number of days, in pounds -/
def total_food : ℝ := 1000

/-- The ratio of food Ella's dog eats compared to Ella -/
def dog_food_ratio : ℝ := 4

theorem ellas_food_calculation :
  ellas_daily_food * (1 + dog_food_ratio) * days = total_food :=
by sorry

end ellas_food_calculation_l416_41641


namespace solutions_for_twenty_l416_41673

-- Define a function that counts the number of distinct integer solutions
def count_solutions (n : ℕ+) : ℕ := 4 * n

-- State the theorem
theorem solutions_for_twenty : count_solutions 20 = 80 := by
  sorry

end solutions_for_twenty_l416_41673


namespace tent_count_solution_l416_41649

def total_value : ℕ := 940000
def total_tents : ℕ := 600
def cost_A : ℕ := 1700
def cost_B : ℕ := 1300

theorem tent_count_solution :
  ∃ (x y : ℕ),
    x + y = total_tents ∧
    cost_A * x + cost_B * y = total_value ∧
    x = 400 ∧
    y = 200 := by
  sorry

end tent_count_solution_l416_41649


namespace geometric_sequence_sum_problem_l416_41668

/-- The sum of the first n terms of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The first term of the geometric sequence -/
def a : ℚ := 1/3

/-- The common ratio of the geometric sequence -/
def r : ℚ := 1/3

/-- The sum we're looking for -/
def target_sum : ℚ := 80/243

theorem geometric_sequence_sum_problem :
  ∃ n : ℕ, geometric_sum a r n = target_sum ∧ n = 5 := by
  sorry

end geometric_sequence_sum_problem_l416_41668


namespace square_division_exists_l416_41645

theorem square_division_exists : ∃ (n : ℕ) (a b c : ℝ), 
  n > 0 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≠ b ∧ c^2 = n * (a^2 + b^2) := by
  sorry

end square_division_exists_l416_41645


namespace not_p_and_not_q_is_false_l416_41633

-- Define proposition p
def p : Prop := ∃ x : ℝ, Real.tan x = 1

-- Define proposition q
def q : Prop := ∀ x : ℝ, x^2 > 0

-- Theorem statement
theorem not_p_and_not_q_is_false : ¬(¬p ∧ ¬q) := by
  sorry

end not_p_and_not_q_is_false_l416_41633


namespace multiply_divide_multiply_l416_41676

theorem multiply_divide_multiply : 8 * 7 / 8 * 7 = 49 := by
  sorry

end multiply_divide_multiply_l416_41676


namespace arithmetic_sequence_common_difference_l416_41642

/-- An arithmetic sequence with its sum function and common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  d : ℝ       -- Common difference
  sum_def : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2
  seq_def : ∀ n, a n = a 1 + (n - 1) * d

/-- Theorem: If 2S_3 = 3S_2 + 6 for an arithmetic sequence, then its common difference is 2 -/
theorem arithmetic_sequence_common_difference 
  (seq : ArithmeticSequence) 
  (h : 2 * seq.S 3 = 3 * seq.S 2 + 6) : 
  seq.d = 2 := by
sorry

end arithmetic_sequence_common_difference_l416_41642


namespace eighth_grade_students_l416_41631

/-- The number of students in eighth grade -/
theorem eighth_grade_students :
  let girls : ℕ := 28
  let boys : ℕ := 2 * girls - 16
  let total : ℕ := boys + girls
  total = 68 := by sorry

end eighth_grade_students_l416_41631


namespace partition_cases_num_partitions_formula_l416_41604

/-- The number of partitions of a set with n+1 elements into n subsets -/
def num_partitions (n : ℕ) : ℕ := (2^n - 1)^(n+1)

/-- Theorem stating the number of partitions for specific cases -/
theorem partition_cases :
  (num_partitions 2 = 3^3) ∧
  (num_partitions 3 = 7^4) ∧
  (num_partitions 4 = 15^5) := by sorry

/-- Main theorem: The number of partitions of a set with n+1 elements into n subsets is (2^n - 1)^(n+1) -/
theorem num_partitions_formula (n : ℕ) :
  num_partitions n = (2^n - 1)^(n+1) := by sorry

end partition_cases_num_partitions_formula_l416_41604


namespace smallest_norm_w_l416_41634

/-- Given a vector w such that ‖w + (4, 2)‖ = 10, 
    the smallest possible value of ‖w‖ is 10 - 2√5 -/
theorem smallest_norm_w (w : ℝ × ℝ) 
    (h : ‖w + (4, 2)‖ = 10) : 
    ∃ (w_min : ℝ × ℝ), ‖w_min‖ = 10 - 2 * Real.sqrt 5 ∧ 
    ∀ (v : ℝ × ℝ), ‖v + (4, 2)‖ = 10 → ‖w_min‖ ≤ ‖v‖ := by
  sorry

end smallest_norm_w_l416_41634


namespace valentino_farm_birds_l416_41694

/-- The number of birds on Mr. Valentino's farm -/
def total_birds (chickens ducks turkeys : ℕ) : ℕ :=
  chickens + ducks + turkeys

/-- Theorem stating the total number of birds on Mr. Valentino's farm -/
theorem valentino_farm_birds :
  ∀ (chickens ducks turkeys : ℕ),
    chickens = 200 →
    ducks = 2 * chickens →
    turkeys = 3 * ducks →
    total_birds chickens ducks turkeys = 1800 := by
  sorry

end valentino_farm_birds_l416_41694


namespace ounces_per_pound_l416_41646

def cat_food_bags : ℕ := 2
def cat_food_weight : ℕ := 3
def dog_food_bags : ℕ := 2
def dog_food_extra_weight : ℕ := 2
def total_ounces : ℕ := 256

theorem ounces_per_pound :
  ∃ (x : ℕ),
    x * (cat_food_bags * cat_food_weight + 
         dog_food_bags * (cat_food_weight + dog_food_extra_weight)) = total_ounces ∧
    x = 16 := by
  sorry

end ounces_per_pound_l416_41646


namespace base_for_888_l416_41640

theorem base_for_888 :
  ∃! b : ℕ,
    (b > 1) ∧
    (∃ a B : ℕ,
      a ≠ B ∧
      a < b ∧
      B < b ∧
      888 = a * b^3 + a * b^2 + B * b + B) ∧
    (b^3 ≤ 888) ∧
    (888 < b^4) :=
by sorry

end base_for_888_l416_41640


namespace students_without_A_l416_41687

theorem students_without_A (total : ℕ) (chemistry_A : ℕ) (physics_A : ℕ) (both_A : ℕ) 
  (h1 : total = 35)
  (h2 : chemistry_A = 9)
  (h3 : physics_A = 15)
  (h4 : both_A = 5) :
  total - (chemistry_A + physics_A - both_A) = 16 := by
  sorry

end students_without_A_l416_41687


namespace proportion_solution_l416_41674

theorem proportion_solution (x : ℝ) : (0.60 : ℝ) / x = (6 : ℝ) / 2 → x = 0.20 := by
  sorry

end proportion_solution_l416_41674


namespace remainder_problem_l416_41647

theorem remainder_problem (j : ℕ+) (h : 75 % (j^2 : ℕ) = 3) : 
  (130 % (j : ℕ) = 0) ∨ (130 % (j : ℕ) = 1) := by
sorry

end remainder_problem_l416_41647


namespace quadratic_max_value_l416_41665

/-- The quadratic function f(x) = -x^2 - 2x - 3 -/
def f (x : ℝ) : ℝ := -x^2 - 2*x - 3

theorem quadratic_max_value :
  (∀ x : ℝ, f x ≤ -2) ∧ (∃ x : ℝ, f x = -2) := by sorry

end quadratic_max_value_l416_41665


namespace child_height_calculation_l416_41688

/-- Given a child's previous height and growth, calculate the current height -/
def current_height (previous_height growth : ℝ) : ℝ :=
  previous_height + growth

/-- Theorem: The child's current height is 41.5 inches -/
theorem child_height_calculation : 
  current_height 38.5 3 = 41.5 := by
  sorry

end child_height_calculation_l416_41688


namespace smallest_with_twelve_odd_eighteen_even_divisors_l416_41663

def count_odd_divisors (n : ℕ) : ℕ := 
  (Finset.filter (λ d => d % 2 = 1) (Nat.divisors n)).card

def count_even_divisors (n : ℕ) : ℕ := 
  (Finset.filter (λ d => d % 2 = 0) (Nat.divisors n)).card

theorem smallest_with_twelve_odd_eighteen_even_divisors :
  ∀ n : ℕ, n > 0 → 
    (count_odd_divisors n = 12 ∧ count_even_divisors n = 18) → 
    n ≥ 900 :=
sorry

end smallest_with_twelve_odd_eighteen_even_divisors_l416_41663


namespace integer_pairs_sum_reciprocals_l416_41670

theorem integer_pairs_sum_reciprocals (x y : ℤ) : 
  x ≤ y ∧ (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 4 ↔ 
  (x = -4 ∧ y = 2) ∨ 
  (x = -12 ∧ y = 3) ∨ 
  (x = 5 ∧ y = 20) ∨ 
  (x = 6 ∧ y = 12) ∨ 
  (x = 8 ∧ y = 8) := by
sorry

end integer_pairs_sum_reciprocals_l416_41670


namespace unique_integer_fraction_l416_41661

theorem unique_integer_fraction (m n : ℕ) (h1 : m ≥ 3) (h2 : n ≥ 3) :
  (∃ S : Set ℕ, (Set.Infinite S ∧
    ∀ a ∈ S, ∃ k : ℤ, (a^m + a - 1 : ℤ) = k * (a^n + a^2 - 1)))
  ↔ m = 5 ∧ n = 3 := by
sorry

end unique_integer_fraction_l416_41661


namespace heather_remaining_blocks_l416_41651

/-- The number of blocks Heather starts with -/
def initial_blocks : ℕ := 86

/-- The number of blocks Heather shares with Jose -/
def shared_blocks : ℕ := 41

/-- The number of blocks Heather ends with -/
def remaining_blocks : ℕ := initial_blocks - shared_blocks

theorem heather_remaining_blocks : remaining_blocks = 45 := by
  sorry

end heather_remaining_blocks_l416_41651


namespace swimming_frequency_l416_41658

def runs_every : ℕ := 4
def cycles_every : ℕ := 16
def all_activities_every : ℕ := 48

theorem swimming_frequency :
  ∃ (swims_every : ℕ),
    swims_every > 0 ∧
    (Nat.lcm swims_every runs_every = Nat.lcm (Nat.lcm swims_every runs_every) cycles_every) ∧
    Nat.lcm (Nat.lcm swims_every runs_every) cycles_every = all_activities_every ∧
    swims_every = 3 := by
  sorry

end swimming_frequency_l416_41658


namespace polynomial_root_implies_coefficients_l416_41699

theorem polynomial_root_implies_coefficients :
  ∀ (a b : ℝ),
  (Complex.I : ℂ) ^ 4 + a * (Complex.I : ℂ) ^ 3 - (Complex.I : ℂ) ^ 2 + b * (Complex.I : ℂ) - 6 = 0 →
  (2 - Complex.I : ℂ) ^ 4 + a * (2 - Complex.I : ℂ) ^ 3 - (2 - Complex.I : ℂ) ^ 2 + b * (2 - Complex.I : ℂ) - 6 = 0 →
  a = -4 ∧ b = 0 := by
sorry

end polynomial_root_implies_coefficients_l416_41699


namespace staircase_expansion_l416_41693

/-- Calculates the number of toothpicks needed for a staircase of given steps -/
def toothpicks_for_steps (n : ℕ) : ℕ :=
  if n ≤ 1 then 4
  else if n = 2 then 10
  else 10 + 8 * (n - 2)

/-- The problem statement -/
theorem staircase_expansion :
  let initial_steps := 4
  let initial_toothpicks := 26
  let main_final_steps := 6
  let adjacent_steps := 3
  let additional_toothpicks := 
    (toothpicks_for_steps main_final_steps + toothpicks_for_steps adjacent_steps) - initial_toothpicks
  additional_toothpicks = 34 := by sorry

end staircase_expansion_l416_41693


namespace sum_of_even_coefficients_l416_41679

theorem sum_of_even_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ a₁₂ : ℝ) :
  (∀ x : ℝ, (x + 1)^4 * (x + 4)^8 = a + a₁*(x + 3) + a₂*(x + 3)^2 + a₃*(x + 3)^3 + 
    a₄*(x + 3)^4 + a₅*(x + 3)^5 + a₆*(x + 3)^6 + a₇*(x + 3)^7 + a₈*(x + 3)^8 + 
    a₉*(x + 3)^9 + a₁₀*(x + 3)^10 + a₁₁*(x + 3)^11 + a₁₂*(x + 3)^12) →
  a₂ + a₄ + a₆ + a₈ + a₁₀ + a₁₂ = 112 := by
sorry

end sum_of_even_coefficients_l416_41679


namespace toy_cost_price_l416_41614

/-- Given the sale of toys, prove the cost price of a single toy. -/
theorem toy_cost_price (num_sold : ℕ) (total_price : ℕ) (gain_equiv : ℕ) (cost_price : ℕ) :
  num_sold = 36 →
  total_price = 45000 →
  gain_equiv = 6 →
  total_price = num_sold * cost_price + gain_equiv * cost_price →
  cost_price = 500 := by
  sorry

end toy_cost_price_l416_41614


namespace largest_integer_with_remainder_l416_41644

theorem largest_integer_with_remainder : 
  ∃ n : ℕ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℕ, m < 100 → m % 7 = 4 → m ≤ n :=
by
  -- The proof would go here
  sorry

end largest_integer_with_remainder_l416_41644


namespace exchange_problem_l416_41627

/-- Calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Represents the exchange scenario -/
def exchangeScenario (d : ℕ) : Prop :=
  (8 : ℚ) / 5 * d - 80 = d

theorem exchange_problem :
  ∃ d : ℕ, exchangeScenario d ∧ sumOfDigits d = 9 := by sorry

end exchange_problem_l416_41627


namespace field_length_is_180_l416_41669

/-- Represents a rectangular field with a surrounding path -/
structure FieldWithPath where
  fieldLength : ℝ
  fieldWidth : ℝ
  pathWidth : ℝ

/-- Calculates the area of the path around the field -/
def pathArea (f : FieldWithPath) : ℝ :=
  (f.fieldLength + 2 * f.pathWidth) * (f.fieldWidth + 2 * f.pathWidth) - f.fieldLength * f.fieldWidth

/-- Theorem: If a rectangular field has width 55m, a surrounding path of 2.5m width, 
    and the path area is 1200 sq m, then the field length is 180m -/
theorem field_length_is_180 (f : FieldWithPath) 
    (h1 : f.fieldWidth = 55)
    (h2 : f.pathWidth = 2.5)
    (h3 : pathArea f = 1200) : 
  f.fieldLength = 180 := by
  sorry


end field_length_is_180_l416_41669


namespace trigonometric_identity_l416_41621

theorem trigonometric_identity (α : ℝ) : 
  3 + 4 * Real.sin (4 * α + 3 / 2 * Real.pi) + Real.sin (8 * α + 5 / 2 * Real.pi) = 8 * (Real.sin (2 * α))^4 := by
  sorry

end trigonometric_identity_l416_41621


namespace weeks_to_cover_all_combinations_l416_41616

/-- Represents a lottery ticket grid -/
structure LotteryGrid :=
  (rows : ℕ)
  (cols : ℕ)
  (row_constraint : rows ≥ 5)
  (col_constraint : cols ≥ 14)

/-- Represents the marking strategy -/
structure MarkingStrategy :=
  (square_size : ℕ)
  (extra_number : ℕ)
  (square_constraint : square_size = 2)
  (extra_constraint : extra_number = 1)

/-- Represents the weekly ticket filling strategy -/
def weekly_tickets : ℕ := 4

/-- Theorem stating the time required to cover all combinations -/
theorem weeks_to_cover_all_combinations 
  (grid : LotteryGrid) 
  (strategy : MarkingStrategy) : 
  (((grid.rows - 2) * (grid.cols - 2)) + weekly_tickets - 1) / weekly_tickets = 52 :=
sorry

end weeks_to_cover_all_combinations_l416_41616


namespace final_digit_is_nine_l416_41659

/-- Represents the sequence of digits formed by concatenating numbers from 1 to 1995 -/
def initial_sequence : List Nat := sorry

/-- Removes digits at even positions from a list of digits -/
def remove_even_positions (digits : List Nat) : List Nat := sorry

/-- Removes digits at odd positions from a list of digits -/
def remove_odd_positions (digits : List Nat) : List Nat := sorry

/-- Applies the alternating removal process until one digit remains -/
def process_sequence (digits : List Nat) : Nat := sorry

theorem final_digit_is_nine : 
  process_sequence initial_sequence = 9 := by sorry

end final_digit_is_nine_l416_41659


namespace mean_of_set_l416_41629

theorem mean_of_set (m : ℝ) : 
  (m + 8 = 16) → 
  (m + (m + 6) + (m + 8) + (m + 14) + (m + 21)) / 5 = 89 / 5 := by
sorry

end mean_of_set_l416_41629


namespace basketball_score_ratio_l416_41630

/-- Represents the points scored in basketball games -/
structure BasketballScores where
  first_away : ℕ
  second_away : ℕ
  third_away : ℕ
  last_home : ℕ
  next_game : ℕ

/-- Theorem stating the ratio of last home game points to first away game points -/
theorem basketball_score_ratio (scores : BasketballScores) : 
  scores.last_home = 62 →
  scores.second_away = scores.first_away + 18 →
  scores.third_away = scores.second_away + 2 →
  scores.next_game = 55 →
  scores.first_away + scores.second_away + scores.third_away + scores.last_home + scores.next_game = 4 * scores.last_home →
  (scores.last_home : ℚ) / scores.first_away = 2 := by
  sorry


end basketball_score_ratio_l416_41630


namespace gcd_8_factorial_6_factorial_squared_l416_41637

theorem gcd_8_factorial_6_factorial_squared : Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 5760 := by
  sorry

end gcd_8_factorial_6_factorial_squared_l416_41637


namespace craig_apple_count_l416_41625

/-- The number of apples Craig has initially -/
def craig_initial_apples : ℝ := 20.0

/-- The number of apples Craig receives from Eugene -/
def apples_from_eugene : ℝ := 7.0

/-- The total number of apples Craig will have -/
def craig_total_apples : ℝ := craig_initial_apples + apples_from_eugene

theorem craig_apple_count : craig_total_apples = 27.0 := by
  sorry

end craig_apple_count_l416_41625


namespace non_similar_1500_pointed_stars_l416_41610

/-- The number of non-similar regular n-pointed stars -/
def num_non_similar_stars (n : ℕ) : ℕ := 
  (Nat.totient n - 2) / 2

/-- Properties of regular n-pointed stars -/
axiom regular_star_properties (n : ℕ) : 
  ∃ (prop : ℕ → Prop), prop n ∧ prop 1000

theorem non_similar_1500_pointed_stars : 
  num_non_similar_stars 1500 = 199 := by
  sorry

end non_similar_1500_pointed_stars_l416_41610


namespace line_tangent_to_circle_l416_41664

/-- The line y - 1 = k(x - 1) is tangent to the circle x^2 + y^2 - 2y = 0 for any real k -/
theorem line_tangent_to_circle (k : ℝ) : 
  ∃! (x y : ℝ), (y - 1 = k * (x - 1)) ∧ (x^2 + y^2 - 2*y = 0) := by
  sorry

end line_tangent_to_circle_l416_41664


namespace smallest_muffin_boxes_l416_41692

theorem smallest_muffin_boxes : ∃ (n : ℕ), n > 0 ∧ (∀ (k : ℕ), 0 < k ∧ k < n → ¬(11 ∣ (17 * k - 1))) ∧ (11 ∣ (17 * n - 1)) := by
  sorry

end smallest_muffin_boxes_l416_41692


namespace mechanic_days_worked_l416_41618

/-- Calculates the number of days a mechanic worked on a car given the following conditions:
  * Hourly rate charged by the mechanic
  * Hours worked per day
  * Cost of parts used
  * Total amount paid by the car owner
-/
def days_worked (hourly_rate : ℚ) (hours_per_day : ℚ) (parts_cost : ℚ) (total_paid : ℚ) : ℚ :=
  (total_paid - parts_cost) / (hourly_rate * hours_per_day)

/-- Theorem stating that given the specific conditions in the problem,
    the number of days worked by the mechanic is 14 -/
theorem mechanic_days_worked :
  days_worked 60 8 2500 9220 = 14 := by
  sorry


end mechanic_days_worked_l416_41618


namespace exists_tangent_circle_l416_41675

-- Define a circle in 2D space
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the property of two circles intersecting
def intersects (c1 c2 : Circle) : Prop :=
  ∃ (p : ℝ × ℝ), (p.1 - c1.center.1)^2 + (p.2 - c1.center.2)^2 = c1.radius^2 ∧
                 (p.1 - c2.center.1)^2 + (p.2 - c2.center.2)^2 = c2.radius^2

-- Define the property of a point being on a circle
def onCircle (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define the property of a circle being tangent to another circle
def isTangent (c1 c2 : Circle) : Prop :=
  ∃ (p : ℝ × ℝ), onCircle p c1 ∧ onCircle p c2 ∧
    ∀ (q : ℝ × ℝ), q ≠ p → ¬(onCircle q c1 ∧ onCircle q c2)

-- Theorem statement
theorem exists_tangent_circle (S₁ S₂ S₃ : Circle) (O : ℝ × ℝ) :
  intersects S₁ S₂ ∧ intersects S₂ S₃ ∧ intersects S₃ S₁ ∧
  onCircle O S₁ ∧ onCircle O S₂ ∧ onCircle O S₃ →
  ∃ (S : Circle), isTangent S S₁ ∧ isTangent S S₂ ∧ isTangent S S₃ :=
sorry

end exists_tangent_circle_l416_41675


namespace same_solution_k_value_l416_41620

theorem same_solution_k_value (x k : ℝ) : 
  (2 * x + 4 = 4 * (x - 2) ∧ -x + k = 2 * x - 1) ↔ k = 17 := by
  sorry

end same_solution_k_value_l416_41620


namespace cricket_average_l416_41683

theorem cricket_average (current_innings : ℕ) (next_innings_runs : ℕ) (average_increase : ℕ) :
  current_innings = 10 →
  next_innings_runs = 80 →
  average_increase = 4 →
  (current_innings * x + next_innings_runs) / (current_innings + 1) = x + average_increase →
  x = 36 :=
by sorry

end cricket_average_l416_41683


namespace evolute_of_ellipse_l416_41612

/-- The equation of the evolute of an ellipse -/
theorem evolute_of_ellipse (a b x y : ℝ) (ha : a > 0) (hb : b > 0) (hx : x ≠ 0) (hy : y ≠ 0) :
  x^2 / a^2 + y^2 / b^2 = 1 →
  (a * x)^(2/3) + (b * y)^(2/3) = (a^2 - b^2)^(2/3) :=
by sorry

end evolute_of_ellipse_l416_41612


namespace even_function_sum_l416_41662

def f (a b x : ℝ) : ℝ := a * x^2 + (b - 3) * x + 3

theorem even_function_sum (a b : ℝ) :
  (∀ x ∈ Set.Icc (a - 2) a, f a b x = f a b ((a - 2) + a - x)) →
  a + b = 4 := by
sorry

end even_function_sum_l416_41662


namespace expression_equality_l416_41680

theorem expression_equality : (-1)^2 - |(-3)| + (-5) / (-5/3) = 1 := by
  sorry

end expression_equality_l416_41680


namespace power_of_512_l416_41626

theorem power_of_512 : (512 : ℝ) ^ (4/3) = 4096 := by sorry

end power_of_512_l416_41626


namespace larger_number_proof_l416_41667

theorem larger_number_proof (A B : ℕ+) : 
  (Nat.gcd A B = 20) → 
  (∃ (k : ℕ+), Nat.lcm A B = 20 * 21 * 23 * k) → 
  (max A B = 460) :=
by sorry

end larger_number_proof_l416_41667


namespace handshake_theorem_l416_41698

theorem handshake_theorem (n : ℕ) (h : n > 0) :
  ∃ (i j : Fin n) (k : ℕ), i ≠ j ∧
  (∃ (f : Fin n → ℕ), (∀ x, f x ≤ n - 1) ∧ f i = k ∧ f j = k) :=
sorry

end handshake_theorem_l416_41698


namespace aluminum_weight_l416_41672

-- Define the weights of the metal pieces
def iron_weight : ℝ := 11.17
def weight_difference : ℝ := 10.33

-- Theorem to prove
theorem aluminum_weight :
  iron_weight - weight_difference = 0.84 := by
  sorry

end aluminum_weight_l416_41672


namespace ages_sum_l416_41696

theorem ages_sum (a b c : ℕ) : 
  a = b + c + 20 → 
  a^2 = (b + c)^2 + 1800 → 
  a + b + c = 90 := by
sorry

end ages_sum_l416_41696


namespace center_of_given_hyperbola_l416_41666

/-- The equation of a hyperbola in the form (ay + b)^2/c^2 - (dx + e)^2/f^2 = 1 --/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ

/-- The center of a hyperbola --/
def center (h : Hyperbola) : ℝ × ℝ := sorry

/-- The given hyperbola --/
def given_hyperbola : Hyperbola :=
  { a := 4
    b := 8
    c := 7
    d := 5
    e := -5
    f := 3 }

/-- Theorem: The center of the given hyperbola is (1, -2) --/
theorem center_of_given_hyperbola :
  center given_hyperbola = (1, -2) := by sorry

end center_of_given_hyperbola_l416_41666
