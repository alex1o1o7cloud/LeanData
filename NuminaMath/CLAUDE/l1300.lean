import Mathlib

namespace centroid_trajectory_on_hyperbola_l1300_130077

/-- The trajectory of the centroid of a triangle formed by a point on a hyperbola and its foci -/
theorem centroid_trajectory_on_hyperbola (x y m n : ℝ) :
  let f₁ : ℝ × ℝ := (5, 0)
  let f₂ : ℝ × ℝ := (-5, 0)
  let p : ℝ × ℝ := (m, n)
  let g : ℝ × ℝ := (x, y)
  (m^2 / 16 - n^2 / 9 = 1) →  -- P is on the hyperbola
  (x = (m + 5 + (-5)) / 3 ∧ y = n / 3) →  -- G is the centroid of ΔF₁F₂P
  (y ≠ 0) →
  (x^2 / (16/9) - y^2 = 1) :=
by sorry

end centroid_trajectory_on_hyperbola_l1300_130077


namespace orange_removal_problem_l1300_130045

/-- Represents the number of oranges Mary must put back to achieve the desired average price -/
def oranges_to_remove : ℕ := sorry

/-- The price of an apple in cents -/
def apple_price : ℕ := 50

/-- The price of an orange in cents -/
def orange_price : ℕ := 60

/-- The total number of fruits initially selected -/
def total_fruits : ℕ := 10

/-- The initial average price of the fruits in cents -/
def initial_avg_price : ℕ := 56

/-- The desired average price after removing oranges in cents -/
def desired_avg_price : ℕ := 52

theorem orange_removal_problem :
  ∃ (apples oranges : ℕ),
    apples + oranges = total_fruits ∧
    (apple_price * apples + orange_price * oranges) / total_fruits = initial_avg_price ∧
    (apple_price * apples + orange_price * (oranges - oranges_to_remove)) / (total_fruits - oranges_to_remove) = desired_avg_price ∧
    oranges_to_remove = 5 := by sorry

end orange_removal_problem_l1300_130045


namespace abc_zero_necessary_not_sufficient_for_a_zero_l1300_130066

theorem abc_zero_necessary_not_sufficient_for_a_zero (a b c : ℝ) :
  (∀ a b c, a = 0 → a * b * c = 0) ∧
  (∃ a b c, a * b * c = 0 ∧ a ≠ 0) :=
by sorry

end abc_zero_necessary_not_sufficient_for_a_zero_l1300_130066


namespace arithmetic_sequence_sum_l1300_130032

theorem arithmetic_sequence_sum : 
  ∀ (a₁ aₙ d n : ℤ), 
  a₁ = -45 → 
  aₙ = -1 → 
  d = 2 → 
  n = (aₙ - a₁) / d + 1 → 
  n * (a₁ + aₙ) / 2 = -529 :=
by
  sorry

end arithmetic_sequence_sum_l1300_130032


namespace complex_modulus_equation_solution_l1300_130036

theorem complex_modulus_equation_solution :
  ∃ (x : ℝ), x > 0 ∧ Complex.abs (5 - 3 * Complex.I * x) = 7 ∧ x = Real.sqrt (8/3) := by
  sorry

end complex_modulus_equation_solution_l1300_130036


namespace angle_cde_is_eleven_degrees_l1300_130088

/-- Given a configuration in a rectangle where:
    - Angle ACB = 80°
    - Angle FEG = 64°
    - Angle DCE = 86°
    - Angle DEC = 83°
    Prove that angle CDE (θ) is equal to 11°. -/
theorem angle_cde_is_eleven_degrees 
  (angle_ACB : ℝ) (angle_FEG : ℝ) (angle_DCE : ℝ) (angle_DEC : ℝ)
  (h1 : angle_ACB = 80)
  (h2 : angle_FEG = 64)
  (h3 : angle_DCE = 86)
  (h4 : angle_DEC = 83) :
  180 - angle_DCE - angle_DEC = 11 := by
  sorry

end angle_cde_is_eleven_degrees_l1300_130088


namespace prime_sum_difference_l1300_130099

theorem prime_sum_difference (m n p : ℕ) 
  (hm : Nat.Prime m) (hn : Nat.Prime n) (hp : Nat.Prime p)
  (h_pos : 0 < p ∧ 0 < n ∧ 0 < m)
  (h_order : m > n ∧ n > p)
  (h_sum : m + n + p = 74)
  (h_diff : m - n - p = 44) :
  m = 59 ∧ n = 13 ∧ p = 2 := by
  sorry

end prime_sum_difference_l1300_130099


namespace second_grade_sample_size_l1300_130010

/-- Given a total sample size and ratios for three grades, calculate the number of students to be drawn from a specific grade -/
def stratified_sample (total_sample : ℕ) (ratio1 ratio2 ratio3 : ℕ) (grade : ℕ) : ℕ :=
  let total_ratio := ratio1 + ratio2 + ratio3
  let grade_ratio := match grade with
    | 1 => ratio1
    | 2 => ratio2
    | 3 => ratio3
    | _ => 0
  (grade_ratio * total_sample) / total_ratio

/-- Theorem stating that for a sample size of 50 and ratios 3:3:4, the second grade should have 15 students -/
theorem second_grade_sample_size :
  stratified_sample 50 3 3 4 2 = 15 := by
  sorry

end second_grade_sample_size_l1300_130010


namespace jim_bought_three_pictures_l1300_130072

def total_pictures : ℕ := 10
def probability_not_bought : ℚ := 21/45

theorem jim_bought_three_pictures :
  ∀ x : ℕ,
  x ≤ total_pictures →
  (total_pictures - x : ℚ) * (total_pictures - 1 - x) / (total_pictures * (total_pictures - 1)) = probability_not_bought →
  x = 3 := by
sorry

end jim_bought_three_pictures_l1300_130072


namespace triangle_tan_A_l1300_130011

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a/b = (b + √3c)/a and sin C = 2√3 sin B, then tan A = √3/3 -/
theorem triangle_tan_A (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  A > 0 → A < π → B > 0 → B < π → C > 0 → C < π →
  A + B + C = π →
  (a / b = (b + Real.sqrt 3 * c) / a) →
  (Real.sin C = 2 * Real.sqrt 3 * Real.sin B) →
  Real.tan A = Real.sqrt 3 / 3 := by
  sorry

end triangle_tan_A_l1300_130011


namespace lake_distance_difference_l1300_130017

/-- The difference between the circumference of a circle with diameter 2 miles
    and its diameter, given π = 3.14 -/
theorem lake_distance_difference : 
  let π : ℝ := 3.14
  let diameter : ℝ := 2
  let circumference := π * diameter
  circumference - diameter = 4.28 := by sorry

end lake_distance_difference_l1300_130017


namespace expression_factorization_l1300_130094

theorem expression_factorization (x y : ℝ) : 
  (x + y)^2 + 4*(x - y)^2 - 4*(x^2 - y^2) = (x - 3*y)^2 := by
  sorry

end expression_factorization_l1300_130094


namespace base_prime_rep_1170_l1300_130080

def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

def base_prime_representation (n : ℕ) : List ℕ := sorry

theorem base_prime_rep_1170 :
  base_prime_representation 1170 = [1, 2, 1, 0, 0, 1] :=
by
  sorry

end base_prime_rep_1170_l1300_130080


namespace log_equation_equivalence_l1300_130005

theorem log_equation_equivalence (x : ℝ) :
  x > 0 → ((Real.log x / Real.log 4) * (Real.log 5 / Real.log x) = Real.log 5 / Real.log 4 ↔ x ≠ 1) :=
by sorry

end log_equation_equivalence_l1300_130005


namespace penny_difference_l1300_130022

theorem penny_difference (kate_pennies john_pennies : ℕ) 
  (h1 : kate_pennies = 223) 
  (h2 : john_pennies = 388) : 
  john_pennies - kate_pennies = 165 := by
sorry

end penny_difference_l1300_130022


namespace arithmetic_sequence_problem_l1300_130029

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def geometric_sequence (b c d : ℝ) :=
  c * c = b * d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (d : ℝ) :
  d ≠ 0 →
  a 4 = 10 →
  arithmetic_sequence a d →
  geometric_sequence (a 3) (a 6) (a 10) →
  ∀ n, a n = n + 6 := by
  sorry

end arithmetic_sequence_problem_l1300_130029


namespace sports_club_tennis_players_l1300_130012

/-- Given a sports club with the following properties:
  * There are 80 total members
  * 48 members play badminton
  * 7 members play neither badminton nor tennis
  * 21 members play both badminton and tennis
  Prove that 46 members play tennis -/
theorem sports_club_tennis_players (total : ℕ) (badminton : ℕ) (neither : ℕ) (both : ℕ) 
  (h1 : total = 80)
  (h2 : badminton = 48)
  (h3 : neither = 7)
  (h4 : both = 21) :
  total - neither - (badminton - both) = 46 := by
  sorry

end sports_club_tennis_players_l1300_130012


namespace hyperbola_line_intersection_l1300_130079

/-- The hyperbola defined by (x^2/9) - y^2 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2/9 - y^2 = 1

/-- The line defined by y = (1/3)(x+1) -/
def line (x y : ℝ) : Prop := y = (1/3)*(x+1)

/-- The number of intersection points between the hyperbola and the line -/
def intersection_count : ℕ := 1

theorem hyperbola_line_intersection :
  ∃! p : ℝ × ℝ, hyperbola p.1 p.2 ∧ line p.1 p.2 :=
sorry

end hyperbola_line_intersection_l1300_130079


namespace book_difference_l1300_130019

def initial_books : ℕ := 28
def jungkook_bought : ℕ := 18
def seokjin_bought : ℕ := 11

theorem book_difference : 
  (initial_books + jungkook_bought) - (initial_books + seokjin_bought) = 7 := by
  sorry

end book_difference_l1300_130019


namespace infinite_solutions_exponential_equation_l1300_130054

theorem infinite_solutions_exponential_equation :
  ∀ x : ℝ, (2 : ℝ) ^ (6 * x + 3) * (4 : ℝ) ^ (3 * x + 6) = (8 : ℝ) ^ (4 * x + 5) := by
  sorry

end infinite_solutions_exponential_equation_l1300_130054


namespace distance_between_cities_l1300_130053

/-- The distance between two cities given two trains traveling towards each other -/
theorem distance_between_cities (t : ℝ) (v₁ v₂ : ℝ) (h₁ : t = 4) (h₂ : v₁ = 115) (h₃ : v₂ = 85) :
  (v₁ + v₂) * t = 800 := by
  sorry

end distance_between_cities_l1300_130053


namespace derivative_at_zero_implies_k_value_l1300_130083

def f (k : ℝ) (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 + k^3 * x

theorem derivative_at_zero_implies_k_value (k : ℝ) :
  (deriv (f k)) 0 = 27 → k = 3 := by
sorry

end derivative_at_zero_implies_k_value_l1300_130083


namespace complex_equation_solution_l1300_130031

theorem complex_equation_solution (a b : ℝ) (z : ℂ) :
  (∀ x : ℂ, x^2 + (4 + Complex.I) * x + 4 + a * Complex.I = 0 → x.im = 0) →
  z = a + b * Complex.I →
  (b : ℂ)^2 + (4 + Complex.I) * b + 4 + a * Complex.I = 0 →
  z = 2 - 2 * Complex.I :=
by sorry

end complex_equation_solution_l1300_130031


namespace milburg_adult_population_l1300_130039

theorem milburg_adult_population (total_population children : ℝ) 
  (h1 : total_population = 5256.0)
  (h2 : children = 2987.0) :
  total_population - children = 2269.0 := by
sorry

end milburg_adult_population_l1300_130039


namespace root_shift_polynomial_l1300_130091

theorem root_shift_polynomial (a b c : ℂ) : 
  (a^3 - 6*a^2 + 11*a - 6 = 0) ∧ 
  (b^3 - 6*b^2 + 11*b - 6 = 0) ∧ 
  (c^3 - 6*c^2 + 11*c - 6 = 0) →
  ((a - 3)^3 + 3*(a - 3)^2 + 2*(a - 3) = 0) ∧
  ((b - 3)^3 + 3*(b - 3)^2 + 2*(b - 3) = 0) ∧
  ((c - 3)^3 + 3*(c - 3)^2 + 2*(c - 3) = 0) :=
by sorry

end root_shift_polynomial_l1300_130091


namespace mean_temperature_is_80_point_2_l1300_130006

def temperatures : List ℝ := [75, 77, 76, 78, 80, 81, 83, 82, 84, 86]

theorem mean_temperature_is_80_point_2 :
  (temperatures.sum / temperatures.length : ℝ) = 80.2 := by
  sorry

end mean_temperature_is_80_point_2_l1300_130006


namespace real_part_of_z_l1300_130038

theorem real_part_of_z (z : ℂ) (h : Complex.I * (z + 1) = -3 + 2 * Complex.I) : 
  z.re = 1 := by sorry

end real_part_of_z_l1300_130038


namespace sophia_pie_consumption_l1300_130034

theorem sophia_pie_consumption (pie_weight : ℝ) (fridge_weight : ℝ) : 
  fridge_weight = (5/6) * pie_weight ∧ fridge_weight = 1200 → 
  pie_weight - fridge_weight = 240 := by
sorry

end sophia_pie_consumption_l1300_130034


namespace equation_solutions_inequality_system_solution_l1300_130081

-- Define the equation
def equation (x : ℝ) : Prop := x^2 - 2*x - 4 = 0

-- Define the inequality system
def inequality_system (x : ℝ) : Prop := 4*(x - 1) < x + 2 ∧ (x + 7) / 3 > x

-- Theorem for the equation solutions
theorem equation_solutions : 
  ∃ (x1 x2 : ℝ), x1 = 1 + Real.sqrt 5 ∧ x2 = 1 - Real.sqrt 5 ∧ 
  equation x1 ∧ equation x2 ∧ 
  ∀ (x : ℝ), equation x → x = x1 ∨ x = x2 := by sorry

-- Theorem for the inequality system solution
theorem inequality_system_solution :
  ∀ (x : ℝ), inequality_system x ↔ x < 2 := by sorry

end equation_solutions_inequality_system_solution_l1300_130081


namespace prob_from_third_farm_given_over_300kg_l1300_130043

/-- Represents the three farms supplying calves -/
inductive Farm : Type
  | first : Farm
  | second : Farm
  | third : Farm

/-- The proportion of calves from each farm -/
def farm_proportion : Farm → ℝ
  | Farm.first => 0.6
  | Farm.second => 0.3
  | Farm.third => 0.1

/-- The probability that a calf from a given farm weighs over 300 kg -/
def prob_over_300kg : Farm → ℝ
  | Farm.first => 0.15
  | Farm.second => 0.25
  | Farm.third => 0.35

/-- The probability that a randomly selected calf weighing over 300 kg came from the third farm -/
theorem prob_from_third_farm_given_over_300kg : 
  (farm_proportion Farm.third * prob_over_300kg Farm.third) / 
  (farm_proportion Farm.first * prob_over_300kg Farm.first + 
   farm_proportion Farm.second * prob_over_300kg Farm.second + 
   farm_proportion Farm.third * prob_over_300kg Farm.third) = 0.175 := by
  sorry

end prob_from_third_farm_given_over_300kg_l1300_130043


namespace books_selling_price_l1300_130007

/-- Calculates the total selling price of two books given their costs and profit/loss percentages -/
def total_selling_price (total_cost book1_cost loss_percent gain_percent : ℚ) : ℚ :=
  let book2_cost := total_cost - book1_cost
  let book1_sell := book1_cost * (1 - loss_percent / 100)
  let book2_sell := book2_cost * (1 + gain_percent / 100)
  book1_sell + book2_sell

/-- Theorem stating that the total selling price of two books is 297.50 Rs given the specified conditions -/
theorem books_selling_price :
  total_selling_price 300 175 15 19 = 297.50 := by
  sorry

end books_selling_price_l1300_130007


namespace no_solution_for_divisibility_l1300_130051

theorem no_solution_for_divisibility (n : ℕ) : n ≥ 1 → ¬(9 ∣ (7^n + n^3)) := by
  sorry

end no_solution_for_divisibility_l1300_130051


namespace vacant_seats_l1300_130049

theorem vacant_seats (total_seats : ℕ) (filled_percentage : ℚ) (vacant_seats : ℕ) : 
  total_seats = 700 →
  filled_percentage = 75 / 100 →
  vacant_seats = total_seats - (filled_percentage * total_seats).floor →
  vacant_seats = 175 := by
  sorry

end vacant_seats_l1300_130049


namespace total_shoes_l1300_130058

theorem total_shoes (brian_shoes : ℕ) (edward_shoes : ℕ) (jacob_shoes : ℕ) 
  (h1 : brian_shoes = 22)
  (h2 : edward_shoes = 3 * brian_shoes)
  (h3 : jacob_shoes = edward_shoes / 2) :
  brian_shoes + edward_shoes + jacob_shoes = 121 := by
  sorry

end total_shoes_l1300_130058


namespace discount_and_increase_l1300_130093

theorem discount_and_increase (original_price : ℝ) (h : original_price > 0) :
  let discounted_price := original_price * (1 - 0.2)
  let increased_price := discounted_price * (1 + 0.25)
  increased_price = original_price :=
by sorry

end discount_and_increase_l1300_130093


namespace boys_in_class_l1300_130062

theorem boys_in_class (total : ℕ) (ratio_boys : ℕ) (ratio_girls : ℕ) (h1 : total = 49) (h2 : ratio_boys = 4) (h3 : ratio_girls = 3) : 
  (ratio_boys * total) / (ratio_boys + ratio_girls) = 28 := by
  sorry

end boys_in_class_l1300_130062


namespace area_relation_implies_parallel_diagonals_l1300_130069

/-- A point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral :=
  (P Q R S : Point)

/-- Area of a quadrilateral -/
def area (q : Quadrilateral) : ℝ := sorry

/-- Check if two line segments are parallel -/
def parallel (A B C D : Point) : Prop := sorry

/-- Points A, B, C, D lie on the sides of quadrilateral PQRS -/
def pointsOnSides (PQRS : Quadrilateral) (A B C D : Point) : Prop := sorry

theorem area_relation_implies_parallel_diagonals 
  (PQRS : Quadrilateral) (A B C D : Point) :
  pointsOnSides PQRS A B C D →
  area PQRS = 2 * area ⟨A, B, C, D⟩ →
  parallel A C Q R ∨ parallel B D P Q := by
  sorry

end area_relation_implies_parallel_diagonals_l1300_130069


namespace perfect_square_condition_l1300_130014

/-- Given a quadratic expression of the form 16x^2 - bx + 9, 
    prove that it is a perfect square trinomial if and only if b = ±24 -/
theorem perfect_square_condition (b : ℝ) : 
  (∃ (k : ℝ), ∀ (x : ℝ), 16 * x^2 - b * x + 9 = (k * x + 3)^2) ↔ (b = 24 ∨ b = -24) := by
  sorry

end perfect_square_condition_l1300_130014


namespace probability_sphere_in_cube_l1300_130056

/-- The probability of a point (x, y, z) satisfying x^2 + y^2 + z^2 ≤ 4,
    given that -2 ≤ x ≤ 2, -2 ≤ y ≤ 2, and -2 ≤ z ≤ 2 -/
theorem probability_sphere_in_cube : 
  let cube_volume := (2 - (-2))^3
  let sphere_volume := (4/3) * Real.pi * 2^3
  sphere_volume / cube_volume = Real.pi / 6 := by
sorry

end probability_sphere_in_cube_l1300_130056


namespace unique_solution_quadratic_l1300_130095

/-- The quadratic equation 2qx^2 - 20x + 5 = 0 has only one solution when q = 10 -/
theorem unique_solution_quadratic :
  ∃! (q : ℝ), q ≠ 0 ∧ (∃! x, 2 * q * x^2 - 20 * x + 5 = 0) := by
sorry

end unique_solution_quadratic_l1300_130095


namespace dance_team_problem_l1300_130046

def student_heights : List ℝ := [161, 162, 162, 164, 165, 165, 165, 166, 166, 167, 168, 168, 170, 172, 172, 175]

def average_height : ℝ := 166.75

def group_A : List ℝ := [162, 165, 165, 166, 166]
def group_B : List ℝ := [161, 162, 164, 165, 175]

def preselected_heights : List ℝ := [168, 168, 172]

def median (l : List ℝ) : ℝ := sorry
def mode (l : List ℝ) : ℝ := sorry
def variance (l : List ℝ) : ℝ := sorry

theorem dance_team_problem :
  (median student_heights = 166) ∧
  (mode student_heights = 165) ∧
  (variance group_A < variance group_B) ∧
  (∃ (h1 h2 : ℝ), h1 ∈ student_heights ∧ h2 ∈ student_heights ∧
    h1 = 170 ∧ h2 = 172 ∧
    variance (h1 :: h2 :: preselected_heights) < 32/9 ∧
    ∀ (x y : ℝ), x ∈ student_heights → y ∈ student_heights →
      variance (x :: y :: preselected_heights) < 32/9 →
      (x + y) / 2 ≤ (h1 + h2) / 2) :=
by sorry

#check dance_team_problem

end dance_team_problem_l1300_130046


namespace routes_from_bristol_to_carlisle_l1300_130087

/-- The number of routes from Bristol to Birmingham -/
def bristol_to_birmingham : ℕ := 6

/-- The number of routes from Birmingham to Sheffield -/
def birmingham_to_sheffield : ℕ := 3

/-- The number of routes from Sheffield to Carlisle -/
def sheffield_to_carlisle : ℕ := 2

/-- The total number of routes from Bristol to Carlisle -/
def total_routes : ℕ := bristol_to_birmingham * birmingham_to_sheffield * sheffield_to_carlisle

theorem routes_from_bristol_to_carlisle : total_routes = 36 := by
  sorry

end routes_from_bristol_to_carlisle_l1300_130087


namespace triangle_area_l1300_130004

/-- Given a triangle ABC with angle A = 60°, side b = 4, and side a = 2√3, 
    prove that its area is 2√3 square units. -/
theorem triangle_area (A B C : ℝ) (a b c : ℝ) : 
  A = π / 3 →  -- 60° in radians
  b = 4 → 
  a = 2 * Real.sqrt 3 → 
  (1 / 2) * a * b * Real.sin C = 2 * Real.sqrt 3 := by
sorry

end triangle_area_l1300_130004


namespace cone_volume_arithmetic_progression_l1300_130063

/-- The volume of a right circular cone with radius, slant height, and height in arithmetic progression -/
theorem cone_volume_arithmetic_progression (r s h d : ℝ) (π : ℝ) : 
  (s = r + d) → (h = r + 2*d) → (0 < r) → (0 < d) → (0 < π) →
  (1/3 : ℝ) * π * r^2 * h = (1/3 : ℝ) * π * (r^3 + 2*d*r^2) :=
by sorry

end cone_volume_arithmetic_progression_l1300_130063


namespace quadratic_root_implies_d_value_l1300_130050

theorem quadratic_root_implies_d_value 
  (d : ℝ) 
  (h : ∀ x : ℝ, 2 * x^2 + 14 * x + d = 0 ↔ x = (-14 + Real.sqrt 20) / 4 ∨ x = (-14 - Real.sqrt 20) / 4) :
  d = 22 := by
sorry

end quadratic_root_implies_d_value_l1300_130050


namespace principal_is_2000_l1300_130097

/-- Given an interest rate, time period, and total interest, 
    calculates the principal amount borrowed. -/
def calculate_principal (rate : ℚ) (time : ℕ) (interest : ℚ) : ℚ :=
  (interest * 100) / (rate * time)

/-- Theorem stating that given the specific conditions, 
    the principal amount borrowed is 2000. -/
theorem principal_is_2000 : 
  let rate : ℚ := 5
  let time : ℕ := 13
  let interest : ℚ := 1300
  calculate_principal rate time interest = 2000 := by
  sorry

end principal_is_2000_l1300_130097


namespace arrow_sequence_equivalence_l1300_130024

/-- Represents a point in the cycle -/
def CyclePoint := ℕ

/-- The length of the cycle -/
def cycleLength : ℕ := 5

/-- Returns the equivalent point within the cycle -/
def cycleEquivalent (n : ℕ) : CyclePoint :=
  n % cycleLength

/-- Theorem: The sequence of arrows from point 630 to point 633 is equivalent
    to the sequence from point 0 to point 3 in a cycle of length 5 -/
theorem arrow_sequence_equivalence :
  (cycleEquivalent 630 = cycleEquivalent 0) ∧
  (cycleEquivalent 631 = cycleEquivalent 1) ∧
  (cycleEquivalent 632 = cycleEquivalent 2) ∧
  (cycleEquivalent 633 = cycleEquivalent 3) := by
  sorry


end arrow_sequence_equivalence_l1300_130024


namespace complex_expression_value_l1300_130074

theorem complex_expression_value : 
  (1 : ℝ) * (2 * 7 / 9) ^ (1 / 2 : ℝ) - (2 * Real.sqrt 3 - Real.pi) ^ (0 : ℝ) - 
  (2 * 10 / 27) ^ (-(2 / 3 : ℝ)) + (1 / 4 : ℝ) ^ (-(3 / 2 : ℝ)) = 389 / 48 := by
  sorry

end complex_expression_value_l1300_130074


namespace statements_correctness_l1300_130098

-- Define the statements
def statement_A (l : Set (ℝ × ℝ)) : Prop :=
  ∃ c : ℝ, l = {(x, y) | x + y = c} ∧ (-2, -3) ∈ l ∧ c = -5

def statement_B (m : ℝ) : Prop :=
  (1, 3) ∈ {(x, y) | 2 * (m + 1) * x + (m - 3) * y + 7 - 5 * m = 0}

def statement_C (θ : ℝ) : Prop :=
  ∀ x y : ℝ, y - 1 = Real.tan θ * (x - 1) ↔ (x, y) ∈ {(x, y) | y - 1 = Real.tan θ * (x - 1)}

def statement_D (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ∀ x y : ℝ, (x₂ - x₁) * (y - y₁) = (y₂ - y₁) * (x - x₁) ↔
    (x, y) ∈ {(x, y) | (x₂ - x₁) * (y - y₁) = (y₂ - y₁) * (x - x₁)}

-- Theorem stating which statements are correct and incorrect
theorem statements_correctness :
  (∃ l : Set (ℝ × ℝ), ¬statement_A l) ∧
  (∀ m : ℝ, statement_B m) ∧
  (∃ θ : ℝ, ¬statement_C θ) ∧
  (∀ x₁ y₁ x₂ y₂ : ℝ, statement_D x₁ y₁ x₂ y₂) := by
  sorry


end statements_correctness_l1300_130098


namespace female_officers_count_l1300_130084

theorem female_officers_count (total_on_duty : ℕ) (female_percentage : ℚ) : 
  total_on_duty = 152 →
  female_percentage = 19 / 100 →
  (total_on_duty / 2 : ℚ) = female_percentage * 400 := by
  sorry

end female_officers_count_l1300_130084


namespace cuboid_base_area_l1300_130018

/-- Theorem: For a cuboid with volume 28 cm³ and height 4 cm, the base area is 7 cm² -/
theorem cuboid_base_area (volume : ℝ) (height : ℝ) (base_area : ℝ) :
  volume = 28 →
  height = 4 →
  volume = base_area * height →
  base_area = 7 :=
by
  sorry

end cuboid_base_area_l1300_130018


namespace specific_device_works_prob_l1300_130042

/-- A device with two components, each having a probability of failure --/
structure Device where
  component_failure_prob : ℝ
  num_components : ℕ

/-- The probability that the device works --/
def device_works_prob (d : Device) : ℝ :=
  (1 - d.component_failure_prob) ^ d.num_components

/-- Theorem: The probability that a specific device works is 0.81 --/
theorem specific_device_works_prob :
  ∃ (d : Device), device_works_prob d = 0.81 := by
  sorry

end specific_device_works_prob_l1300_130042


namespace parallelogram_area_l1300_130064

/-- The area of a parallelogram with one angle of 100 degrees and two consecutive sides of lengths 10 inches and 20 inches is approximately 197.0 square inches. -/
theorem parallelogram_area (a b : ℝ) (θ : ℝ) : 
  a = 10 → b = 20 → θ = 100 * π / 180 → 
  abs (a * b * Real.sin (π - θ) - 197.0) < 0.1 := by
  sorry

end parallelogram_area_l1300_130064


namespace squirrel_cones_problem_l1300_130052

theorem squirrel_cones_problem :
  ∃ (x y : ℕ), 
    x + y < 25 ∧
    2 * x > y + 26 ∧
    2 * y > x - 4 ∧
    x = 17 ∧
    y = 7 := by
  sorry

end squirrel_cones_problem_l1300_130052


namespace sum_of_seven_odd_integers_remainder_l1300_130035

def consecutive_odd_integers (start : ℕ) (count : ℕ) : List ℕ :=
  List.range count |>.map (λ i => start + 2 * i)

theorem sum_of_seven_odd_integers_remainder (start : ℕ) (h : start = 12095) :
  (consecutive_odd_integers start 7).sum % 10 = 7 := by
  sorry

end sum_of_seven_odd_integers_remainder_l1300_130035


namespace output_is_three_l1300_130048

def program_output (a b : ℕ) : ℕ := a + b

theorem output_is_three : program_output 1 2 = 3 := by
  sorry

end output_is_three_l1300_130048


namespace inverse_of_A_cubed_l1300_130055

variable (A : Matrix (Fin 2) (Fin 2) ℝ)

theorem inverse_of_A_cubed 
  (h : A⁻¹ = ![![-3, 2], ![-1, 3]]) : 
  (A^3)⁻¹ = ![![-21, 14], ![-7, 21]] := by
  sorry

end inverse_of_A_cubed_l1300_130055


namespace isosceles_triangle_perimeter_l1300_130023

/-- Represents the side lengths of an isosceles triangle -/
structure IsoscelesTriangle where
  equalSide : ℝ
  baseSide : ℝ

/-- Checks if the triangle satisfies the given conditions -/
def satisfiesConditions (t : IsoscelesTriangle) : Prop :=
  t.equalSide = 20 ∧ t.baseSide = (2/5) * t.equalSide

/-- Calculates the perimeter of the triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ :=
  2 * t.equalSide + t.baseSide

/-- Theorem stating that the perimeter of the triangle is 48 cm -/
theorem isosceles_triangle_perimeter :
  ∀ t : IsoscelesTriangle, satisfiesConditions t → perimeter t = 48 :=
by sorry

end isosceles_triangle_perimeter_l1300_130023


namespace pandas_weekly_bamboo_consumption_l1300_130059

/-- The amount of bamboo eaten by pandas in a week -/
def bamboo_eaten_in_week (adult_daily : ℕ) (baby_daily : ℕ) : ℕ :=
  (adult_daily + baby_daily) * 7

/-- Theorem: The total amount of bamboo eaten by an adult panda and a baby panda in a week -/
theorem pandas_weekly_bamboo_consumption :
  bamboo_eaten_in_week 138 50 = 1316 := by
  sorry

end pandas_weekly_bamboo_consumption_l1300_130059


namespace man_rowing_speed_l1300_130021

/-- Given a man's speed in still water and downstream speed, calculate his upstream speed -/
theorem man_rowing_speed (v_still : ℝ) (v_downstream : ℝ) (v_upstream : ℝ) : 
  v_still = 31 → v_downstream = 37 → v_upstream = 25 := by
  sorry

#check man_rowing_speed

end man_rowing_speed_l1300_130021


namespace parabola_dot_product_zero_l1300_130075

/-- A point on the parabola y^2 = 4x -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 4*x

/-- The line passing through two points intersects (4,0) -/
def line_through_four (A B : ParabolaPoint) : Prop :=
  ∃ t : ℝ, A.x + t * (B.x - A.x) = 4 ∧ A.y + t * (B.y - A.y) = 0

/-- The dot product of vectors OA and OB -/
def dot_product (A B : ParabolaPoint) : ℝ :=
  A.x * B.x + A.y * B.y

theorem parabola_dot_product_zero (A B : ParabolaPoint) 
  (h : line_through_four A B) : dot_product A B = 0 := by
  sorry

end parabola_dot_product_zero_l1300_130075


namespace distance_AB_is_600_l1300_130086

-- Define the cities
structure City where
  name : String

-- Define the travelers
structure Traveler where
  name : String
  start : City
  destination : City
  travelTime : ℝ
  averageSpeed : ℝ

-- Define the problem setup
def cityA : City := ⟨"A"⟩
def cityB : City := ⟨"B"⟩
def cityC : City := ⟨"C"⟩

def eddy : Traveler := ⟨"Eddy", cityA, cityB, 3, 2⟩
def freddy : Traveler := ⟨"Freddy", cityA, cityC, 3, 1⟩

-- Define the distances
def distanceAC : ℝ := 300

-- Theorem statement
theorem distance_AB_is_600 :
  let distanceAB := eddy.averageSpeed * eddy.travelTime
  distanceAC = freddy.averageSpeed * freddy.travelTime →
  eddy.averageSpeed = 2 * freddy.averageSpeed →
  distanceAB = 600 := by
  sorry

end distance_AB_is_600_l1300_130086


namespace equation_solutions_l1300_130013

theorem equation_solutions :
  (∃ x₁ x₂, x₁ = -3/2 ∧ x₂ = 2 ∧ 2 * x₁^2 - x₁ - 6 = 0 ∧ 2 * x₂^2 - x₂ - 6 = 0) ∧
  (∃ y₁ y₂, y₁ = -1 ∧ y₂ = 1/2 ∧ (y₁ - 2)^2 = 9 * y₁^2 ∧ (y₂ - 2)^2 = 9 * y₂^2) :=
by sorry

end equation_solutions_l1300_130013


namespace geometric_sequence_sufficient_not_necessary_l1300_130033

/-- Defines a geometric sequence of three real numbers -/
def is_geometric_sequence (a b c : ℝ) : Prop :=
  b ≠ 0 ∧ c / b = b / a

/-- Proves that "a, b, c form a geometric sequence" is a sufficient but not necessary condition for "b^2 = ac" -/
theorem geometric_sequence_sufficient_not_necessary :
  (∀ a b c : ℝ, is_geometric_sequence a b c → b^2 = a*c) ∧
  (∃ a b c : ℝ, b^2 = a*c ∧ ¬is_geometric_sequence a b c) :=
by sorry

end geometric_sequence_sufficient_not_necessary_l1300_130033


namespace weight_equivalence_l1300_130096

/-- The weight ratio between small and large circles -/
def weight_ratio : ℚ := 2 / 5

/-- The number of small circles -/
def num_small_circles : ℕ := 15

/-- Theorem stating the equivalence in weight between small and large circles -/
theorem weight_equivalence :
  (num_small_circles : ℚ) * weight_ratio = 6 := by sorry

end weight_equivalence_l1300_130096


namespace contrapositive_equivalence_l1300_130026

theorem contrapositive_equivalence (m : ℝ) :
  (¬(∃ x : ℝ, x^2 + x - m = 0) → m ≤ 0) ↔
  (m > 0 → ∃ x : ℝ, x^2 + x - m = 0) :=
by sorry

end contrapositive_equivalence_l1300_130026


namespace sams_original_portion_l1300_130040

theorem sams_original_portion (s j r : ℝ) :
  s + j + r = 1200 →
  s - 200 + 3 * j + 3 * r = 1800 →
  s = 800 :=
by sorry

end sams_original_portion_l1300_130040


namespace part_one_part_two_l1300_130041

-- Define sets A and B
def A (a b : ℝ) : Set ℝ := {x | a - b < x ∧ x < a + b}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

-- Part 1
theorem part_one (a : ℝ) : 
  (A a 1 ∩ B = A a 1) → (a ≤ -2 ∨ a ≥ 6) := by sorry

-- Part 2
theorem part_two (b : ℝ) :
  (A 1 b ∩ B = ∅) → (b ≤ 2) := by sorry

end part_one_part_two_l1300_130041


namespace fishmonger_sales_l1300_130073

/-- Given a first week's sales and a multiplier for the second week's sales,
    calculate the total sales over two weeks. -/
def totalSales (firstWeekSales secondWeekMultiplier : ℕ) : ℕ :=
  firstWeekSales + firstWeekSales * secondWeekMultiplier

/-- Theorem stating that given the specific conditions of the problem,
    the total sales over two weeks is 200 kg. -/
theorem fishmonger_sales : totalSales 50 3 = 200 := by
  sorry

end fishmonger_sales_l1300_130073


namespace min_plates_for_seven_colors_l1300_130068

/-- The minimum number of plates needed to guarantee at least three matching pairs -/
def min_plates_for_three_pairs (num_colors : ℕ) : ℕ :=
  3 * num_colors + 3

/-- Theorem stating that given 7 different colors of plates, 
    the minimum number of plates needed to guarantee at least three matching pairs is 24 -/
theorem min_plates_for_seven_colors : 
  min_plates_for_three_pairs 7 = 24 := by
  sorry

#eval min_plates_for_three_pairs 7

end min_plates_for_seven_colors_l1300_130068


namespace smallest_resolvable_debt_l1300_130060

theorem smallest_resolvable_debt (pig_value goat_value : ℕ) 
  (h_pig : pig_value = 400) (h_goat : goat_value = 250) :
  ∃ (D : ℕ), D > 0 ∧ 
  (∃ (p g : ℤ), D = pig_value * p + goat_value * g) ∧
  (∀ (D' : ℕ), D' > 0 → 
    (∃ (p' g' : ℤ), D' = pig_value * p' + goat_value * g') → 
    D ≤ D') :=
by sorry

end smallest_resolvable_debt_l1300_130060


namespace notebook_distribution_l1300_130008

theorem notebook_distribution (total_notebooks : ℕ) (half_students_notebooks : ℕ) :
  total_notebooks = 512 →
  half_students_notebooks = 16 →
  ∃ (num_students : ℕ) (fraction : ℚ),
    num_students > 0 ∧
    fraction > 0 ∧
    fraction < 1 ∧
    (num_students / 2 : ℚ) * half_students_notebooks = total_notebooks ∧
    (num_students : ℚ) * (fraction * num_students) = total_notebooks ∧
    fraction = 1 / 8 :=
by sorry

end notebook_distribution_l1300_130008


namespace inequality_solution_sets_l1300_130015

theorem inequality_solution_sets (a : ℝ) :
  (∀ x, ax^2 + 5*x - 2 > 0 ↔ 1/2 < x ∧ x < 2) →
  (∀ x, ax^2 - 5*x + a^2 - 1 > 0 ↔ -3 < x ∧ x < 1/2) :=
by sorry

end inequality_solution_sets_l1300_130015


namespace consecutive_numbers_theorem_l1300_130057

theorem consecutive_numbers_theorem (a b c d e : ℕ) : 
  (a > b) ∧ (b > c) ∧ (c > d) ∧ (d > e) ∧  -- Descending order
  (a - b = 1) ∧ (b - c = 1) ∧ (c - d = 1) ∧ (d - e = 1) ∧  -- Consecutive numbers
  ((a + b + c) / 3 = 45) ∧  -- Average of first three
  ((c + d + e) / 3 = 43) →  -- Average of last three
  c = 44 :=
by sorry

end consecutive_numbers_theorem_l1300_130057


namespace total_fish_l1300_130028

theorem total_fish (lilly_fish rosy_fish tom_fish : ℕ) 
  (h1 : lilly_fish = 10)
  (h2 : rosy_fish = 14)
  (h3 : tom_fish = 8) :
  lilly_fish + rosy_fish + tom_fish = 32 := by
  sorry

end total_fish_l1300_130028


namespace tangent_line_at_origin_l1300_130092

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a - 2)*x

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + (a - 2)

-- State the theorem
theorem tangent_line_at_origin (a : ℝ) 
  (h : ∀ x, f' a x = f' a (-x)) : 
  ∃ m : ℝ, m = -2 ∧ ∀ x, f a x = m * x + f a 0 := by sorry

end tangent_line_at_origin_l1300_130092


namespace line_passes_through_fixed_point_l1300_130016

theorem line_passes_through_fixed_point (p q : ℝ) (h : p + 2*q - 1 = 0) :
  ∃ (x y : ℝ), x = 1/2 ∧ y = -1/6 ∧ p*x + 3*y + q = 0 := by sorry

end line_passes_through_fixed_point_l1300_130016


namespace solve_system_l1300_130025

theorem solve_system (a b : ℚ) 
  (eq1 : 3 * a + 2 * b = 25)
  (eq2 : 5 * a + b = 20) :
  3 * a + 3 * b = 240 / 7 := by
  sorry

end solve_system_l1300_130025


namespace sqrt_sum_reciprocals_equals_sqrt1111_over_112_l1300_130090

theorem sqrt_sum_reciprocals_equals_sqrt1111_over_112 :
  Real.sqrt (1 / 25 + 1 / 36 + 1 / 49) = Real.sqrt 1111 / 112 := by
  sorry

end sqrt_sum_reciprocals_equals_sqrt1111_over_112_l1300_130090


namespace kannon_fruit_consumption_l1300_130044

/-- Represents Kannon's fruit consumption over two days -/
structure FruitConsumption where
  apples_last_night : ℕ
  bananas_last_night : ℕ
  oranges_last_night : ℕ
  apples_increase : ℕ
  bananas_multiplier : ℕ

/-- Calculates the total number of fruits eaten over two days -/
def total_fruits (fc : FruitConsumption) : ℕ :=
  let apples_today := fc.apples_last_night + fc.apples_increase
  let bananas_today := fc.bananas_last_night * fc.bananas_multiplier
  let oranges_today := 2 * apples_today
  (fc.apples_last_night + apples_today) +
  (fc.bananas_last_night + bananas_today) +
  (fc.oranges_last_night + oranges_today)

/-- Theorem stating that Kannon's total fruit consumption is 39 -/
theorem kannon_fruit_consumption :
  ∃ (fc : FruitConsumption),
    fc.apples_last_night = 3 ∧
    fc.bananas_last_night = 1 ∧
    fc.oranges_last_night = 4 ∧
    fc.apples_increase = 4 ∧
    fc.bananas_multiplier = 10 ∧
    total_fruits fc = 39 := by
  sorry

end kannon_fruit_consumption_l1300_130044


namespace book_club_snack_fee_l1300_130076

theorem book_club_snack_fee (members : ℕ) (hardcover_price paperback_price : ℚ)
  (hardcover_count paperback_count : ℕ) (total_collected : ℚ) :
  members = 6 →
  hardcover_price = 30 →
  paperback_price = 12 →
  hardcover_count = 6 →
  paperback_count = 6 →
  total_collected = 2412 →
  (total_collected - members * (hardcover_price * hardcover_count + paperback_price * paperback_count)) / members = 150 := by
  sorry

#check book_club_snack_fee

end book_club_snack_fee_l1300_130076


namespace inequality_proof_l1300_130089

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : b + d < a + c := by
  sorry

end inequality_proof_l1300_130089


namespace scientific_notation_of_388800_l1300_130020

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_388800 :
  toScientificNotation 388800 = ScientificNotation.mk 3.888 5 sorry := by sorry

end scientific_notation_of_388800_l1300_130020


namespace x_value_l1300_130030

theorem x_value : ∃ x : ℚ, (2 / 5 * x) - (1 / 3 * x) = 110 ∧ x = 1650 := by
  sorry

end x_value_l1300_130030


namespace trivia_team_tryouts_l1300_130009

theorem trivia_team_tryouts (not_picked : ℕ) (groups : ℕ) (students_per_group : ℕ) : 
  not_picked = 10 → groups = 8 → students_per_group = 6 → 
  not_picked + groups * students_per_group = 58 := by
sorry

end trivia_team_tryouts_l1300_130009


namespace circle_radius_l1300_130070

theorem circle_radius (C : ℝ) (r : ℝ) (h : C = 72 * Real.pi) : C = 2 * Real.pi * r → r = 36 := by
  sorry

end circle_radius_l1300_130070


namespace robins_gum_problem_l1300_130061

theorem robins_gum_problem (initial_gum : ℕ) (total_gum : ℕ) (h1 : initial_gum = 18) (h2 : total_gum = 44) :
  total_gum - initial_gum = 26 := by
  sorry

end robins_gum_problem_l1300_130061


namespace a_4_equals_4_l1300_130047

/-- Given a sequence {aₙ} defined by aₙ = (-1)ⁿ n, prove that a₄ = 4 -/
theorem a_4_equals_4 (a : ℕ → ℤ) (h : ∀ n, a n = (-1)^n * n) : a 4 = 4 := by
  sorry

end a_4_equals_4_l1300_130047


namespace prob_even_sum_is_two_fifths_l1300_130067

/-- A card is represented by a natural number between 1 and 5 -/
def Card : Type := { n : ℕ // 1 ≤ n ∧ n ≤ 5 }

/-- The set of all cards -/
def allCards : Finset Card := sorry

/-- The function that determines if the sum of two cards is even -/
def isEvenSum (c1 c2 : Card) : Prop := Even (c1.val + c2.val)

/-- The set of all pairs of cards -/
def allPairs : Finset (Card × Card) := sorry

/-- The set of all pairs of cards with even sum -/
def evenSumPairs : Finset (Card × Card) := sorry

/-- The probability of drawing two cards with even sum -/
noncomputable def probEvenSum : ℚ := (Finset.card evenSumPairs : ℚ) / (Finset.card allPairs : ℚ)

theorem prob_even_sum_is_two_fifths : probEvenSum = 2 / 5 := by sorry

end prob_even_sum_is_two_fifths_l1300_130067


namespace sheila_hourly_rate_l1300_130085

/-- Represents Sheila's work schedule and earnings --/
structure WorkSchedule where
  monday_hours : ℕ
  tuesday_hours : ℕ
  wednesday_hours : ℕ
  thursday_hours : ℕ
  friday_hours : ℕ
  weekly_earnings : ℕ

/-- Calculates the total hours worked in a week --/
def total_hours (schedule : WorkSchedule) : ℕ :=
  schedule.monday_hours + schedule.tuesday_hours + schedule.wednesday_hours +
  schedule.thursday_hours + schedule.friday_hours

/-- Calculates the hourly rate given a work schedule --/
def hourly_rate (schedule : WorkSchedule) : ℚ :=
  schedule.weekly_earnings / (total_hours schedule)

/-- Sheila's work schedule --/
def sheila_schedule : WorkSchedule :=
  { monday_hours := 8
  , tuesday_hours := 6
  , wednesday_hours := 8
  , thursday_hours := 6
  , friday_hours := 8
  , weekly_earnings := 432 }

theorem sheila_hourly_rate :
  hourly_rate sheila_schedule = 12 := by
  sorry

end sheila_hourly_rate_l1300_130085


namespace quadratic_inequality_solution_condition_l1300_130065

theorem quadratic_inequality_solution_condition (d : ℝ) :
  d > 0 →
  (∃ x : ℝ, x^2 - 8*x + d < 0) ↔ d < 16 :=
by sorry

end quadratic_inequality_solution_condition_l1300_130065


namespace incorrect_number_calculation_l1300_130071

theorem incorrect_number_calculation (n : ℕ) (incorrect_avg correct_avg correct_num : ℝ) (X : ℝ) :
  n = 10 →
  incorrect_avg = 18 →
  correct_avg = 22 →
  correct_num = 66 →
  n * incorrect_avg = (n - 1) * correct_avg + X →
  n * correct_avg = (n - 1) * correct_avg + correct_num →
  X = 26 := by
    sorry

end incorrect_number_calculation_l1300_130071


namespace no_two_digit_sum_with_reverse_is_cube_l1300_130002

/-- Function to reverse the digits of a two-digit number -/
def reverseDigits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- Function to check if a number is a perfect cube -/
def isPerfectCube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

/-- Theorem: No two-digit positive integer N has the property that
    the sum of N and its digit-reversed number is a perfect cube -/
theorem no_two_digit_sum_with_reverse_is_cube :
  ¬∃ N : ℕ, 10 ≤ N ∧ N < 100 ∧ isPerfectCube (N + reverseDigits N) := by
  sorry

end no_two_digit_sum_with_reverse_is_cube_l1300_130002


namespace probability_white_ball_l1300_130078

/-- The probability of drawing a white ball from a bag with red, white, and black balls -/
theorem probability_white_ball (red white black : ℕ) (h : red = 5 ∧ white = 2 ∧ black = 3) :
  (white : ℚ) / (red + white + black : ℚ) = 1 / 5 := by
  sorry

end probability_white_ball_l1300_130078


namespace mia_has_110_dollars_l1300_130082

/-- The amount of money Darwin has -/
def darwins_money : ℕ := 45

/-- The amount of money Mia has -/
def mias_money : ℕ := 2 * darwins_money + 20

theorem mia_has_110_dollars : mias_money = 110 := by
  sorry

end mia_has_110_dollars_l1300_130082


namespace sin_300_degrees_l1300_130001

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by sorry

end sin_300_degrees_l1300_130001


namespace quadratic_equation_solution_l1300_130000

theorem quadratic_equation_solution (b : ℚ) : 
  ((-8 : ℚ)^2 + b * (-8) - 45 = 0) → b = 19/8 := by
  sorry

end quadratic_equation_solution_l1300_130000


namespace unique_integer_solution_l1300_130027

theorem unique_integer_solution (a : ℤ) : 
  (∃! x : ℤ, |a*x + a + 2| < 2) ↔ (a = 3 ∨ a = -3) :=
sorry

end unique_integer_solution_l1300_130027


namespace wynter_bicycle_count_l1300_130003

/-- The number of bicycles Wynter counted -/
def num_bicycles : ℕ := 50

/-- The number of tricycles Wynter counted -/
def num_tricycles : ℕ := 20

/-- The total number of wheels from all vehicles -/
def total_wheels : ℕ := 160

/-- The number of wheels on a bicycle -/
def wheels_per_bicycle : ℕ := 2

/-- The number of wheels on a tricycle -/
def wheels_per_tricycle : ℕ := 3

/-- Theorem stating that the number of bicycles Wynter counted is 50 -/
theorem wynter_bicycle_count :
  num_bicycles * wheels_per_bicycle + num_tricycles * wheels_per_tricycle = total_wheels :=
by sorry

end wynter_bicycle_count_l1300_130003


namespace brownie_pieces_count_l1300_130037

/-- The length of the pan in inches -/
def pan_length : ℕ := 24

/-- The width of the pan in inches -/
def pan_width : ℕ := 15

/-- The side length of a square brownie piece in inches -/
def piece_side : ℕ := 3

/-- The number of brownie pieces that can be cut from the pan -/
def num_pieces : ℕ := (pan_length * pan_width) / (piece_side * piece_side)

theorem brownie_pieces_count : num_pieces = 40 := by
  sorry

end brownie_pieces_count_l1300_130037
