import Mathlib

namespace square_of_prime_mod_six_l3350_335013

theorem square_of_prime_mod_six (p : ℕ) (h_prime : Nat.Prime p) (h_gt_five : p > 5) :
  p ^ 2 % 6 = 1 := by
  sorry

end square_of_prime_mod_six_l3350_335013


namespace secret_spread_theorem_l3350_335018

/-- The number of students who know the secret on day n -/
def students_knowing_secret (n : ℕ) : ℕ := (3^(n+1) - 1) / 2

/-- The day of the week given a number of days since Monday -/
def day_of_week (n : ℕ) : String :=
  match n % 7 with
  | 0 => "Sunday"
  | 1 => "Monday"
  | 2 => "Tuesday"
  | 3 => "Wednesday"
  | 4 => "Thursday"
  | 5 => "Friday"
  | _ => "Saturday"

theorem secret_spread_theorem : 
  ∃ n : ℕ, students_knowing_secret n = 2186 ∧ day_of_week n = "Sunday" :=
by sorry

end secret_spread_theorem_l3350_335018


namespace parallel_line_through_point_l3350_335015

/-- Given a point p and a line l, this function returns the equation of the line parallel to l that passes through p. -/
def parallel_line_equation (p : ℝ × ℝ) (l : ℝ → ℝ → ℝ → Prop) : ℝ → ℝ → ℝ → Prop :=
  sorry

theorem parallel_line_through_point :
  let p : ℝ × ℝ := (-1, 3)
  let l : ℝ → ℝ → ℝ → Prop := fun x y z ↦ x - 2*y + z = 0
  let result : ℝ → ℝ → ℝ → Prop := fun x y z ↦ x - 2*y + 7 = 0
  parallel_line_equation p l = result :=
sorry

end parallel_line_through_point_l3350_335015


namespace rectangle_perimeter_l3350_335058

/-- The perimeter of a rectangle with area 500 cm² and one side 25 cm is 90 cm. -/
theorem rectangle_perimeter (a b : ℝ) (h_area : a * b = 500) (h_side : a = 25) : 
  2 * (a + b) = 90 := by
sorry

end rectangle_perimeter_l3350_335058


namespace unique_denomination_l3350_335068

/-- Given unlimited supply of stamps of denominations 7, n, and n+2 cents,
    120 cents is the greatest postage that cannot be formed -/
def is_valid_denomination (n : ℕ) : Prop :=
  ∀ k : ℕ, k > 120 → ∃ (a b c : ℕ), k = 7 * a + n * b + (n + 2) * c

/-- 120 cents cannot be formed using stamps of denominations 7, n, and n+2 cents -/
def cannot_form_120 (n : ℕ) : Prop :=
  ¬∃ (a b c : ℕ), 120 = 7 * a + n * b + (n + 2) * c

theorem unique_denomination :
  ∃! n : ℕ, n > 0 ∧ is_valid_denomination n ∧ cannot_form_120 n :=
by sorry

end unique_denomination_l3350_335068


namespace solve_inequality_for_x_find_k_range_l3350_335099

-- Part 1
theorem solve_inequality_for_x (x : ℝ) :
  (|1 - x * 2| > |x - 2|) ↔ (x < -1 ∨ x > 1) := by sorry

-- Part 2
theorem find_k_range (k : ℝ) :
  (∀ x y : ℝ, |x| < 1 → |y| < 1 → |1 - k*x*y| > |k*x - y|) ↔ 
  (k ≥ -1 ∧ k ≤ 1) := by sorry

end solve_inequality_for_x_find_k_range_l3350_335099


namespace position_after_five_steps_l3350_335047

/-- A student's walk on a number line --/
structure StudentWalk where
  total_steps : ℕ
  total_distance : ℝ
  step_length : ℝ
  marking_distance : ℝ

/-- The position after a certain number of steps --/
def position_after_steps (walk : StudentWalk) (steps : ℕ) : ℝ :=
  walk.step_length * steps

/-- The theorem to prove --/
theorem position_after_five_steps (walk : StudentWalk) 
  (h1 : walk.total_steps = 8)
  (h2 : walk.total_distance = 48)
  (h3 : walk.marking_distance = 3)
  (h4 : walk.step_length = walk.total_distance / walk.total_steps) :
  position_after_steps walk 5 = 30 := by
  sorry

end position_after_five_steps_l3350_335047


namespace convex_pentagon_with_equal_diagonals_and_sides_l3350_335016

-- Define a pentagon as a set of 5 points in 2D space
def Pentagon := Fin 5 → ℝ × ℝ

-- Define a function to check if a pentagon is convex
def is_convex (p : Pentagon) : Prop := sorry

-- Define a function to calculate the length of a line segment
def length (a b : ℝ × ℝ) : ℝ := sorry

-- Define a function to check if a line segment is a diagonal of the pentagon
def is_diagonal (p : Pentagon) (i j : Fin 5) : Prop :=
  (i.val + 2) % 5 ≤ j.val ∨ (j.val + 2) % 5 ≤ i.val

-- Define a function to check if a line segment is a side of the pentagon
def is_side (p : Pentagon) (i j : Fin 5) : Prop :=
  (i.val + 1) % 5 = j.val ∨ (j.val + 1) % 5 = i.val

-- Theorem: There exists a convex pentagon where each diagonal is equal to some side
theorem convex_pentagon_with_equal_diagonals_and_sides :
  ∃ (p : Pentagon), is_convex p ∧
    ∀ (i j : Fin 5), is_diagonal p i j →
      ∃ (k l : Fin 5), is_side p k l ∧ length (p i) (p j) = length (p k) (p l) :=
sorry

end convex_pentagon_with_equal_diagonals_and_sides_l3350_335016


namespace polynomial_factorization_l3350_335040

theorem polynomial_factorization (p q : ℕ) (n : ℕ) (a : ℤ) :
  Prime p ∧ Prime q ∧ p ≠ q ∧ n ≥ 3 →
  (∃ (g h : Polynomial ℤ), (Polynomial.degree g ≥ 1) ∧ (Polynomial.degree h ≥ 1) ∧
    (Polynomial.X : Polynomial ℤ)^n + a * (Polynomial.X : Polynomial ℤ)^(n-1) + (p * q : ℤ) = g * h) ↔
  (a = (-1)^n * (p * q : ℤ) + 1 ∨ a = -(p * q : ℤ) - 1) :=
by sorry

end polynomial_factorization_l3350_335040


namespace no_three_digit_integers_with_five_units_divisible_by_ten_l3350_335005

theorem no_three_digit_integers_with_five_units_divisible_by_ten :
  ¬ ∃ n : ℕ, 
    100 ≤ n ∧ n < 1000 ∧  -- three-digit positive integer
    n % 10 = 5 ∧          -- 5 in the units place
    n % 10 = 0            -- divisible by 10
  := by sorry

end no_three_digit_integers_with_five_units_divisible_by_ten_l3350_335005


namespace divisibility_condition_l3350_335045

theorem divisibility_condition (a b : ℕ+) : 
  (∃ k : ℕ, (b.val ^ 2 + 3 * a.val) = a.val ^ 2 * b.val * k) ↔ 
  ((a, b) = (1, 1) ∨ (a, b) = (1, 3)) := by
sorry

end divisibility_condition_l3350_335045


namespace remainder_783245_div_7_l3350_335067

theorem remainder_783245_div_7 : 783245 % 7 = 1 := by
  sorry

end remainder_783245_div_7_l3350_335067


namespace line_slope_l3350_335086

/-- Given a line with equation y = -5x + 9, its slope is -5 -/
theorem line_slope (x y : ℝ) : y = -5 * x + 9 → (∃ m b : ℝ, y = m * x + b ∧ m = -5) := by
  sorry

end line_slope_l3350_335086


namespace annual_cost_difference_is_5525_l3350_335034

def annual_cost_difference : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ := 
  fun clarinet_rate clarinet_hours piano_rate piano_hours violin_rate violin_hours 
      singing_rate singing_hours weeks_per_year =>
    let weeks_with_lessons := weeks_per_year - 2
    let clarinet_cost := clarinet_rate * clarinet_hours * weeks_with_lessons
    let piano_cost := (piano_rate * piano_hours * weeks_with_lessons * 9) / 10
    let violin_cost := (violin_rate * violin_hours * weeks_with_lessons * 85) / 100
    let singing_cost := singing_rate * singing_hours * weeks_with_lessons
    piano_cost + violin_cost + singing_cost - clarinet_cost

theorem annual_cost_difference_is_5525 :
  annual_cost_difference 40 3 28 5 35 2 45 1 52 = 5525 := by
  sorry

end annual_cost_difference_is_5525_l3350_335034


namespace percentage_difference_l3350_335024

theorem percentage_difference (n : ℝ) (h : n = 140) : (4/5 * n) - (65/100 * n) = 21 := by
  sorry

end percentage_difference_l3350_335024


namespace employee_hire_year_l3350_335079

/-- Rule of 70 retirement provision -/
def rule_of_70 (age : ℕ) (years_employed : ℕ) : Prop :=
  age + years_employed ≥ 70

/-- The year the employee was hired -/
def hire_year : ℕ := sorry

/-- The employee's age when hired -/
def hire_age : ℕ := 32

/-- The year the employee became eligible for retirement -/
def retirement_year : ℕ := 2007

theorem employee_hire_year :
  rule_of_70 (hire_age + (retirement_year - hire_year)) (retirement_year - hire_year) ∧
  hire_year = 1969 := by
  sorry

end employee_hire_year_l3350_335079


namespace marble_problem_solution_l3350_335043

/-- Represents the number of marbles of each color in a box -/
structure MarbleBox where
  red : ℕ
  green : ℕ
  yellow : ℕ
  other : ℕ

/-- Calculates the total number of marbles in the box -/
def MarbleBox.total (box : MarbleBox) : ℕ :=
  box.red + box.green + box.yellow + box.other

/-- Represents the conditions of the marble problem -/
def marble_problem (box : MarbleBox) : Prop :=
  box.red = 20 ∧
  box.green = 3 * box.red ∧
  box.yellow = box.green / 5 ∧
  box.total = 4 * box.green

theorem marble_problem_solution (box : MarbleBox) :
  marble_problem box → box.other = 148 := by
  sorry

end marble_problem_solution_l3350_335043


namespace constant_term_in_expansion_l3350_335053

theorem constant_term_in_expansion :
  ∃ (k : ℕ), k > 0 ∧ k < 5 ∧ (2 * 5 = 5 * k) := by
  sorry

end constant_term_in_expansion_l3350_335053


namespace white_blue_line_difference_l3350_335025

/-- The length difference between two lines -/
def length_difference (white_line blue_line : ℝ) : ℝ :=
  white_line - blue_line

/-- Theorem stating the length difference between the white and blue lines -/
theorem white_blue_line_difference :
  let white_line : ℝ := 7.666666666666667
  let blue_line : ℝ := 3.3333333333333335
  length_difference white_line blue_line = 4.333333333333333 := by
  sorry

end white_blue_line_difference_l3350_335025


namespace pt_length_in_quadrilateral_l3350_335073

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (P Q R S : Point)

/-- Calculates the length between two points -/
def distance (A B : Point) : ℝ := sorry

/-- Calculates the area of a triangle given three points -/
def triangleArea (A B C : Point) : ℝ := sorry

/-- Theorem: In a convex quadrilateral PQRS, given specific side lengths and conditions, 
    the length of PT can be determined -/
theorem pt_length_in_quadrilateral 
  (PQRS : Quadrilateral)
  (T : Point)
  (convex : sorry) -- Assumption that PQRS is convex
  (pq_length : distance PQRS.P PQRS.Q = 8)
  (rs_length : distance PQRS.R PQRS.S = 14)
  (pr_length : distance PQRS.P PQRS.R = 18)
  (qs_length : distance PQRS.Q PQRS.S = 12)
  (T_on_PR : sorry) -- Assumption that T is on PR
  (T_on_QS : sorry) -- Assumption that T is on QS
  (equal_areas : triangleArea PQRS.P T PQRS.R = triangleArea PQRS.Q T PQRS.S) :
  distance PQRS.P T = 72 / 11 := by sorry

end pt_length_in_quadrilateral_l3350_335073


namespace fraction_equality_l3350_335060

theorem fraction_equality (x y : ℝ) (h : (1/x + 1/y) / (1/x + 2/y) = 4) :
  (x + y) / (x + 2*y) = 4/11 := by
  sorry

end fraction_equality_l3350_335060


namespace max_value_expression_l3350_335087

theorem max_value_expression : 
  (∀ x : ℝ, (3 * x^2 + 9 * x + 28) / (3 * x^2 + 9 * x + 7) ≤ 85) ∧ 
  (∀ ε > 0, ∃ x : ℝ, (3 * x^2 + 9 * x + 28) / (3 * x^2 + 9 * x + 7) > 85 - ε) :=
by sorry

end max_value_expression_l3350_335087


namespace arithmetic_sequence_sum_l3350_335021

def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmeticSequence a →
  (a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 420) →
  (a 2 + a 10 = 120) := by
  sorry

end arithmetic_sequence_sum_l3350_335021


namespace tan_alpha_two_implies_fraction_l3350_335027

theorem tan_alpha_two_implies_fraction (α : Real) (h : Real.tan α = 2) :
  (3 * Real.sin α - Real.cos α) / (2 * Real.sin α + 3 * Real.cos α) = 5 / 7 := by
  sorry

end tan_alpha_two_implies_fraction_l3350_335027


namespace expand_expression_l3350_335063

theorem expand_expression (x : ℝ) : 2 * (x + 3) * (x^2 - 2*x + 7) = 2*x^3 + 2*x^2 + 2*x + 42 := by
  sorry

end expand_expression_l3350_335063


namespace carnival_participants_l3350_335074

theorem carnival_participants (n : ℕ) (masks costumes both : ℕ) : 
  n ≥ 42 →
  masks = (3 * n) / 7 →
  costumes = (5 * n) / 6 →
  both = masks + costumes - n →
  both ≥ 11 := by
sorry

end carnival_participants_l3350_335074


namespace solve_system_for_w_l3350_335052

theorem solve_system_for_w (x y z w : ℝ) 
  (eq1 : 2*x + y + z + w = 1)
  (eq2 : x + 3*y + z + w = 2)
  (eq3 : x + y + 4*z + w = 3)
  (eq4 : x + y + z + 5*w = 25) : 
  w = 11/2 := by sorry

end solve_system_for_w_l3350_335052


namespace maggot_feeding_problem_l3350_335000

/-- The number of maggots attempted to be fed in the first feeding -/
def first_feeding : ℕ := 15

/-- The total number of maggots served -/
def total_maggots : ℕ := 20

/-- The number of maggots eaten in the first feeding -/
def eaten_first : ℕ := 1

/-- The number of maggots eaten in the second feeding -/
def eaten_second : ℕ := 3

theorem maggot_feeding_problem :
  first_feeding + eaten_first + eaten_second = total_maggots :=
by sorry

end maggot_feeding_problem_l3350_335000


namespace louise_boxes_l3350_335039

/-- The number of pencils each box can hold -/
def pencils_per_box : ℕ := 20

/-- The number of red pencils Louise has -/
def red_pencils : ℕ := 20

/-- The number of blue pencils Louise has -/
def blue_pencils : ℕ := 2 * red_pencils

/-- The number of yellow pencils Louise has -/
def yellow_pencils : ℕ := 40

/-- The number of green pencils Louise has -/
def green_pencils : ℕ := red_pencils + blue_pencils

/-- The total number of pencils Louise has -/
def total_pencils : ℕ := red_pencils + blue_pencils + yellow_pencils + green_pencils

/-- The number of boxes Louise needs -/
def boxes_needed : ℕ := total_pencils / pencils_per_box

theorem louise_boxes : boxes_needed = 8 := by
  sorry

end louise_boxes_l3350_335039


namespace y_can_take_any_real_value_l3350_335064

-- Define the equation
def equation (x y : ℝ) : Prop := 2 * x * abs x + y^2 = 1

-- Theorem statement
theorem y_can_take_any_real_value :
  ∀ y : ℝ, ∃ x : ℝ, equation x y :=
by
  sorry

end y_can_take_any_real_value_l3350_335064


namespace coin_to_sphere_weight_change_l3350_335004

theorem coin_to_sphere_weight_change 
  (R₁ R₂ R₃ : ℝ) 
  (h_positive : 0 < R₁ ∧ 0 < R₂ ∧ 0 < R₃) 
  (h_balance : R₁^2 + R₂^2 = R₃^2) : 
  R₁^3 + R₂^3 < R₃^3 := by
sorry

end coin_to_sphere_weight_change_l3350_335004


namespace michelle_sandwiches_l3350_335095

/-- The number of sandwiches Michelle has left to give to her other co-workers -/
def sandwiches_left (total : ℕ) (first : ℕ) (second : ℕ) : ℕ :=
  total - first - second - (2 * first) - (3 * second)

/-- Proof that Michelle has 26 sandwiches left -/
theorem michelle_sandwiches : sandwiches_left 50 4 3 = 26 := by
  sorry

end michelle_sandwiches_l3350_335095


namespace subtract_p_q_equals_five_twentyfourths_l3350_335054

theorem subtract_p_q_equals_five_twentyfourths 
  (p q : ℚ) 
  (hp : 3 / p = 8) 
  (hq : 3 / q = 18) : 
  p - q = 5 / 24 := by
  sorry

end subtract_p_q_equals_five_twentyfourths_l3350_335054


namespace quadratic_real_solutions_l3350_335009

theorem quadratic_real_solutions (m : ℝ) :
  (∃ x : ℝ, m * x^2 - 4 * x + 1 = 0) → m ≤ 4 := by
  sorry

end quadratic_real_solutions_l3350_335009


namespace mixture_problem_l3350_335006

/-- Proves that the percentage of the first solution is 30% given the conditions of the mixture problem. -/
theorem mixture_problem (total_volume : ℝ) (result_percentage : ℝ) (second_solution_percentage : ℝ)
  (first_solution_volume : ℝ) (second_solution_volume : ℝ)
  (h1 : total_volume = 40)
  (h2 : result_percentage = 45)
  (h3 : second_solution_percentage = 80)
  (h4 : first_solution_volume = 28)
  (h5 : second_solution_volume = 12)
  (h6 : total_volume = first_solution_volume + second_solution_volume)
  (h7 : result_percentage / 100 * total_volume =
        (first_solution_percentage / 100 * first_solution_volume) +
        (second_solution_percentage / 100 * second_solution_volume)) :
  first_solution_percentage = 30 :=
sorry

end mixture_problem_l3350_335006


namespace probability_ten_red_balls_in_twelve_draws_l3350_335041

theorem probability_ten_red_balls_in_twelve_draws 
  (total_balls : Nat) (white_balls : Nat) (red_balls : Nat)
  (h1 : total_balls = white_balls + red_balls)
  (h2 : white_balls = 5)
  (h3 : red_balls = 3) :
  let p_red := red_balls / total_balls
  let p_white := white_balls / total_balls
  let n := 11  -- number of draws before the last one
  let k := 9   -- number of red balls in the first 11 draws
  Nat.choose n k * p_red^k * p_white^(n-k) * p_red = 
    Nat.choose 11 9 * (3/8)^9 * (5/8)^2 * (3/8) :=
by sorry

end probability_ten_red_balls_in_twelve_draws_l3350_335041


namespace vans_needed_l3350_335036

theorem vans_needed (total_people : ℕ) (van_capacity : ℕ) (h1 : total_people = 35) (h2 : van_capacity = 4) :
  ↑⌈(total_people : ℚ) / van_capacity⌉ = 9 := by
  sorry

end vans_needed_l3350_335036


namespace max_value_of_d_l3350_335076

def a (n : ℕ) : ℕ := n^3 + 4

def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_value_of_d : ∃ (k : ℕ), d k = 433 ∧ ∀ (n : ℕ), d n ≤ 433 := by sorry

end max_value_of_d_l3350_335076


namespace equation_solutions_l3350_335033

def equation (x : ℝ) : Prop :=
  x ≠ 2/3 ∧ x ≠ -3 ∧ (8*x + 3) / (3*x^2 + 8*x - 6) = 3*x / (3*x - 2)

theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ (x = 1 ∨ x = -1) :=
sorry

end equation_solutions_l3350_335033


namespace exists_ratio_preserving_quadrilateral_l3350_335011

/-- A convex quadrilateral -/
structure ConvexQuadrilateral where
  sides : Fin 4 → ℝ
  angles : Fin 4 → ℝ
  sides_positive : ∀ i, sides i > 0
  angles_positive : ∀ i, angles i > 0
  sides_convex : ∀ i, sides i < sides ((i + 1) % 4) + sides ((i + 2) % 4) + sides ((i + 3) % 4)
  angles_convex : ∀ i, angles i < angles ((i + 1) % 4) + angles ((i + 2) % 4) + angles ((i + 3) % 4)
  angle_sum : angles 0 + angles 1 + angles 2 + angles 3 = 2 * Real.pi

/-- The existence of a quadrilateral with side-angle ratio preservation -/
theorem exists_ratio_preserving_quadrilateral (q : ConvexQuadrilateral) :
  ∃ q' : ConvexQuadrilateral,
    ∀ i : Fin 4, (q'.sides i) / (q'.sides ((i + 1) % 4)) = (q.angles i) / (q.angles ((i + 1) % 4)) :=
sorry

end exists_ratio_preserving_quadrilateral_l3350_335011


namespace sum_of_cubes_l3350_335072

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 8) (h2 : x * y = 15) :
  x^3 + y^3 = 152 := by
sorry

end sum_of_cubes_l3350_335072


namespace sector_perimeter_l3350_335031

/-- Given a circular sector with area 2 cm² and central angle 4 radians, its perimeter is 6 cm. -/
theorem sector_perimeter (r : ℝ) (θ : ℝ) : 
  (1/2 * r^2 * θ = 2) → θ = 4 → (r * θ + 2 * r = 6) := by
  sorry

end sector_perimeter_l3350_335031


namespace triangle_count_l3350_335080

/-- The total number of triangles in a specially divided rectangle -/
def total_triangles (small_right : ℕ) (isosceles_quarter_width : ℕ) (isosceles_third_length : ℕ) (larger_right : ℕ) (large_isosceles : ℕ) : ℕ :=
  small_right + isosceles_quarter_width + isosceles_third_length + larger_right + large_isosceles

/-- Theorem stating the total number of triangles in the specially divided rectangle -/
theorem triangle_count :
  total_triangles 24 8 12 16 4 = 64 := by sorry

end triangle_count_l3350_335080


namespace circle_area_ratio_l3350_335001

theorem circle_area_ratio (diameter_R diameter_S area_R area_S : ℝ) :
  diameter_R = 0.6 * diameter_S →
  area_R = π * (diameter_R / 2)^2 →
  area_S = π * (diameter_S / 2)^2 →
  area_R / area_S = 0.36 :=
by
  sorry

end circle_area_ratio_l3350_335001


namespace maria_water_bottles_l3350_335091

theorem maria_water_bottles (initial bottles_bought bottles_remaining : ℕ) 
  (h1 : initial = 14)
  (h2 : bottles_bought = 45)
  (h3 : bottles_remaining = 51) :
  initial - (bottles_remaining - bottles_bought) = 8 := by
  sorry

end maria_water_bottles_l3350_335091


namespace largest_prime_2015_digits_square_minus_one_div_15_l3350_335029

/-- The largest prime with 2015 digits -/
def p : ℕ := sorry

/-- p is prime -/
axiom p_prime : Nat.Prime p

/-- p has 2015 digits -/
axiom p_digits : 10^2014 ≤ p ∧ p < 10^2015

/-- p is the largest such prime -/
axiom p_largest : ∀ q : ℕ, Nat.Prime q → 10^2014 ≤ q ∧ q < 10^2015 → q ≤ p

theorem largest_prime_2015_digits_square_minus_one_div_15 : 15 ∣ (p^2 - 1) := by
  sorry

end largest_prime_2015_digits_square_minus_one_div_15_l3350_335029


namespace lawyer_fee_ratio_l3350_335012

/-- Lawyer fee calculation and payment ratio problem --/
theorem lawyer_fee_ratio :
  let upfront_fee : ℕ := 1000
  let hourly_rate : ℕ := 100
  let court_hours : ℕ := 50
  let prep_hours : ℕ := 2 * court_hours
  let total_fee : ℕ := upfront_fee + hourly_rate * (court_hours + prep_hours)
  let john_payment : ℕ := 8000
  let brother_payment : ℕ := total_fee - john_payment
  brother_payment * 2 = total_fee := by sorry

end lawyer_fee_ratio_l3350_335012


namespace ellipse_parabola_focus_l3350_335090

theorem ellipse_parabola_focus (m n : ℝ) (hm : m > 0) (hn : n > 0) : 
  (∀ x y : ℝ, x^2 / m^2 + y^2 / n^2 = 1) →
  (∃ k : ℝ, ∀ x y : ℝ, x^2 = 8*y → y = k) →
  (n^2 - m^2 = 4) →
  (Real.sqrt (n^2 - m^2) / n = 1/2) →
  m - n = 2 * Real.sqrt 3 - 4 := by
sorry

end ellipse_parabola_focus_l3350_335090


namespace order_powers_l3350_335050

theorem order_powers : 2^300 < 3^200 ∧ 3^200 < 10^100 := by
  sorry

end order_powers_l3350_335050


namespace monotonic_increase_interval_l3350_335085

theorem monotonic_increase_interval
  (f : ℝ → ℝ)
  (ω φ : ℝ)
  (h_ω : ω > 0)
  (h_φ : 0 < φ ∧ φ < π / 2)
  (h_f : ∀ x, f x = Real.sin (ω * x + φ))
  (x₁ x₂ : ℝ)
  (h_x₁ : f x₁ = 1)
  (h_x₂ : f x₂ = 0)
  (h_x_diff : |x₁ - x₂| = 1 / 2)
  (h_f_half : f (1 / 2) = 1 / 2) :
  ∃ k : ℤ, StrictMonoOn f (Set.Icc (- 5 / 6 + 2 * k) (1 / 6 + 2 * k)) :=
by sorry

end monotonic_increase_interval_l3350_335085


namespace find_b_value_l3350_335059

theorem find_b_value (b : ℚ) : 
  ((-8 : ℚ)^2 + b * (-8 : ℚ) - 15 = 0) → b = 49/8 := by
  sorry

end find_b_value_l3350_335059


namespace two_box_marble_problem_l3350_335098

/-- Represents a box containing marbles -/
structure MarbleBox where
  total : ℕ
  black : ℕ
  white : ℕ
  h_sum : total = black + white

/-- The probability of drawing a specific color marble from a box -/
def drawProbability (box : MarbleBox) (color : ℕ) : ℚ :=
  color / box.total

theorem two_box_marble_problem (box1 box2 : MarbleBox) : 
  box1.total + box2.total = 25 →
  drawProbability box1 box1.black * drawProbability box2 box2.black = 27/50 →
  drawProbability box1 box1.white * drawProbability box2 box2.white = 1/25 := by
sorry

end two_box_marble_problem_l3350_335098


namespace arrangement_count_l3350_335038

/-- The number of ways to arrange 3 male and 3 female students in a row with exactly two female students adjacent -/
def num_arrangements : ℕ := sorry

/-- The total number of students -/
def total_students : ℕ := 6

/-- The number of male students -/
def num_male : ℕ := 3

/-- The number of female students -/
def num_female : ℕ := 3

theorem arrangement_count :
  (total_students = num_male + num_female) →
  (num_arrangements = 432) := by sorry

end arrangement_count_l3350_335038


namespace range_of_a_l3350_335066

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≥ 0) → a ≤ 1 := by
  sorry

end range_of_a_l3350_335066


namespace coffee_lasts_12_days_l3350_335051

-- Define the constants
def coffee_lbs : ℕ := 3
def cups_per_lb : ℕ := 40
def weekday_consumption : ℕ := 3 + 2 + 4
def weekend_consumption : ℕ := 2 + 3 + 5
def days_in_week : ℕ := 7
def weekdays_per_week : ℕ := 5
def weekend_days_per_week : ℕ := 2

-- Define the theorem
theorem coffee_lasts_12_days :
  let total_cups := coffee_lbs * cups_per_lb
  let weekly_consumption := weekday_consumption * weekdays_per_week + weekend_consumption * weekend_days_per_week
  let days_coffee_lasts := (total_cups * days_in_week) / weekly_consumption
  days_coffee_lasts = 12 :=
by sorry

end coffee_lasts_12_days_l3350_335051


namespace sports_club_members_l3350_335071

/-- Represents a sports club with members playing badminton and tennis -/
structure SportsClub where
  badminton : ℕ  -- Number of members playing badminton
  tennis : ℕ     -- Number of members playing tennis
  neither : ℕ    -- Number of members playing neither badminton nor tennis
  both : ℕ       -- Number of members playing both badminton and tennis

/-- Calculates the total number of members in the sports club -/
def totalMembers (club : SportsClub) : ℕ :=
  club.badminton + club.tennis - club.both + club.neither

/-- Theorem stating that the total number of members in the given sports club is 30 -/
theorem sports_club_members :
  ∃ (club : SportsClub), 
    club.badminton = 16 ∧ 
    club.tennis = 19 ∧ 
    club.neither = 2 ∧ 
    club.both = 7 ∧ 
    totalMembers club = 30 := by
  sorry

end sports_club_members_l3350_335071


namespace problem_solution_l3350_335002

/-- The surface area of an open box formed by removing square corners from a rectangular sheet. -/
def boxSurfaceArea (length width corner_size : ℕ) : ℕ :=
  length * width - 4 * (corner_size * corner_size)

/-- Theorem stating that the surface area of the box described in the problem is 500 square units. -/
theorem problem_solution :
  boxSurfaceArea 30 20 5 = 500 := by
  sorry

end problem_solution_l3350_335002


namespace probability_of_negative_product_l3350_335028

def set_m : Finset Int := {-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4}
def set_t : Finset Int := {-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7}

def negative_product_pairs : Finset (Int × Int) :=
  (set_m.filter (λ x => x < 0) ×ˢ set_t.filter (λ y => y > 0)) ∪
  (set_m.filter (λ x => x > 0) ×ˢ set_t.filter (λ y => y < 0))

theorem probability_of_negative_product :
  (negative_product_pairs.card : ℚ) / ((set_m.card * set_t.card) : ℚ) = 65 / 144 := by
  sorry

end probability_of_negative_product_l3350_335028


namespace debate_team_groups_l3350_335035

theorem debate_team_groups (boys : ℕ) (girls : ℕ) (group_size : ℕ) : 
  boys = 31 → girls = 32 → group_size = 9 → 
  (boys + girls) / group_size = 7 ∧ (boys + girls) % group_size = 0 := by
  sorry

end debate_team_groups_l3350_335035


namespace final_number_is_100_l3350_335032

def board_numbers : List ℚ := List.map (λ i => 1 / i) (List.range 100)

def combine (a b : ℚ) : ℚ := a * b + a + b

theorem final_number_is_100 (numbers : List ℚ) (h : numbers = board_numbers) :
  (numbers.foldl combine 0 : ℚ) = 100 := by
  sorry

end final_number_is_100_l3350_335032


namespace max_vector_difference_l3350_335014

theorem max_vector_difference (x : ℝ) : 
  let m : ℝ × ℝ := (Real.cos (x / 2), Real.sin (x / 2))
  let n : ℝ × ℝ := (-Real.sqrt 3, 1)
  (∀ y : ℝ, ‖(m.1 - n.1, m.2 - n.2)‖ ≤ 3) ∧ 
  (∃ z : ℝ, ‖(Real.cos (z / 2) - (-Real.sqrt 3), Real.sin (z / 2) - 1)‖ = 3) := by
sorry

end max_vector_difference_l3350_335014


namespace society_member_numbers_l3350_335061

theorem society_member_numbers (n : ℕ) (k : ℕ) (members : Fin n → Fin k) :
  n = 1978 →
  k = 6 →
  (∀ i : Fin n, (members i).val + 1 = i.val) →
  ∃ i j l : Fin n,
    (members i = members j ∧ members i = members l ∧ i.val = j.val + l.val) ∨
    (members i = members j ∧ i.val = 2 * j.val) :=
by sorry

end society_member_numbers_l3350_335061


namespace intersection_of_A_and_B_l3350_335070

def A : Set ℤ := {0, 1, 2, 8}
def B : Set ℤ := {-1, 1, 6, 8}

theorem intersection_of_A_and_B : A ∩ B = {1, 8} := by sorry

end intersection_of_A_and_B_l3350_335070


namespace income_percentage_l3350_335077

/-- Given that Mart's income is 60% more than Tim's income, and Tim's income is 60% less than Juan's income, 
    prove that Mart's income is 64% of Juan's income. -/
theorem income_percentage (juan tim mart : ℝ) 
  (h1 : tim = 0.4 * juan)  -- Tim's income is 60% less than Juan's
  (h2 : mart = 1.6 * tim)  -- Mart's income is 60% more than Tim's
  : mart = 0.64 * juan := by
  sorry


end income_percentage_l3350_335077


namespace william_wins_l3350_335044

theorem william_wins (total_rounds : ℕ) (william_advantage : ℕ) (william_wins : ℕ) : 
  total_rounds = 15 → 
  william_advantage = 5 → 
  william_wins = total_rounds / 2 + william_advantage → 
  william_wins = 10 := by
sorry

end william_wins_l3350_335044


namespace min_positive_period_of_f_l3350_335089

open Real

noncomputable def f (x : ℝ) : ℝ := (sin x - Real.sqrt 3 * cos x) * (cos x - Real.sqrt 3 * sin x)

theorem min_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧
  (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧
  T = π :=
sorry

end min_positive_period_of_f_l3350_335089


namespace unique_cyclic_number_l3350_335037

def is_permutation (a b : Nat) : Prop := sorry

def has_distinct_digits (n : Nat) : Prop := sorry

theorem unique_cyclic_number : ∃! n : Nat, 
  100000 ≤ n ∧ n < 1000000 ∧ 
  has_distinct_digits n ∧
  is_permutation n (2*n) ∧
  is_permutation n (3*n) ∧
  is_permutation n (4*n) ∧
  is_permutation n (5*n) ∧
  is_permutation n (6*n) ∧
  n = 142857 := by sorry

end unique_cyclic_number_l3350_335037


namespace x_squared_y_not_less_than_x_cubed_plus_y_fifth_l3350_335023

theorem x_squared_y_not_less_than_x_cubed_plus_y_fifth 
  (x y : ℝ) (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1) : 
  x^2 * y ≥ x^3 + y^5 := by
  sorry

end x_squared_y_not_less_than_x_cubed_plus_y_fifth_l3350_335023


namespace polyhedron_20_faces_l3350_335046

/-- A polyhedron with triangular faces -/
structure Polyhedron where
  faces : Nat
  is_triangular : Bool

/-- The number of edges in a polyhedron -/
def num_edges (p : Polyhedron) : Nat :=
  3 * p.faces / 2

/-- The number of vertices in a polyhedron -/
def num_vertices (p : Polyhedron) : Nat :=
  p.faces + 2 - num_edges p

/-- Theorem: A polyhedron with 20 triangular faces has 30 edges and 12 vertices -/
theorem polyhedron_20_faces (p : Polyhedron) 
  (h1 : p.faces = 20) 
  (h2 : p.is_triangular = true) : 
  num_edges p = 30 ∧ num_vertices p = 12 := by
  sorry


end polyhedron_20_faces_l3350_335046


namespace no_arithmetic_sequence_with_arithmetic_digit_sum_l3350_335096

/-- An arithmetic sequence of positive integers. -/
def ArithmeticSequence (a : ℕ → ℕ) : Prop :=
  ∃ (a₀ d : ℕ), ∀ n, a n = a₀ + n * d

/-- The sum of digits of a natural number. -/
def sumOfDigits (n : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that no infinite arithmetic sequence of distinct positive integers
    exists such that the sum of digits of each term also forms an arithmetic sequence. -/
theorem no_arithmetic_sequence_with_arithmetic_digit_sum :
  ¬ ∃ (a : ℕ → ℕ),
    ArithmeticSequence a ∧
    (∀ n m, n ≠ m → a n ≠ a m) ∧
    ArithmeticSequence (λ n => sumOfDigits (a n)) :=
sorry

end no_arithmetic_sequence_with_arithmetic_digit_sum_l3350_335096


namespace exponential_function_fixed_point_l3350_335057

theorem exponential_function_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^x + 1
  f 0 = 2 := by
sorry

end exponential_function_fixed_point_l3350_335057


namespace altara_population_2040_l3350_335093

/-- Represents the population of Altara at a given year -/
def population (year : ℕ) : ℕ :=
  sorry

theorem altara_population_2040 :
  (population 2020 = 500) →
  (∀ y : ℕ, y ≥ 2020 → population (y + 10) = 2 * population y) →
  population 2040 = 2000 :=
by
  sorry

end altara_population_2040_l3350_335093


namespace factor_63x_minus_21_l3350_335092

theorem factor_63x_minus_21 : ∀ x : ℝ, 63 * x - 21 = 21 * (3 * x - 1) := by
  sorry

end factor_63x_minus_21_l3350_335092


namespace parabola_theorem_l3350_335049

-- Define a parabola type
structure Parabola where
  equation : ℝ → ℝ → Prop
  directrix : ℝ → ℝ → Prop

-- Define the conditions for the parabola
def parabola_conditions (p : Parabola) : Prop :=
  -- Vertex at origin
  p.equation 0 0 ∧
  -- Passes through (-3, 2)
  p.equation (-3) 2 ∧
  -- Axis of symmetry along coordinate axis (implied by the equation forms)
  (∃ (a : ℝ), ∀ (x y : ℝ), p.equation x y ↔ y^2 = a * x) ∨
  (∃ (b : ℝ), ∀ (x y : ℝ), p.equation x y ↔ x^2 = b * y)

-- Define the possible equations and directrices
def parabola1 : Parabola :=
  { equation := λ x y => y^2 = -4/3 * x
    directrix := λ x y => x = 1/3 }

def parabola2 : Parabola :=
  { equation := λ x y => x^2 = 9/2 * y
    directrix := λ x y => y = -9/8 }

-- Theorem statement
theorem parabola_theorem :
  ∀ (p : Parabola), parabola_conditions p →
    (p = parabola1 ∨ p = parabola2) :=
sorry

end parabola_theorem_l3350_335049


namespace lost_card_number_l3350_335008

theorem lost_card_number (n : ℕ) (h1 : n > 0) (h2 : (n * (n + 1)) / 2 - 101 ≤ n) : 
  (n * (n + 1)) / 2 - 101 = 4 := by
  sorry

#check lost_card_number

end lost_card_number_l3350_335008


namespace floor_abs_sum_equality_l3350_335003

theorem floor_abs_sum_equality : ⌊|(-3.7 : ℝ)|⌋ + |⌊(-3.7 : ℝ)⌋| = 7 := by
  sorry

end floor_abs_sum_equality_l3350_335003


namespace same_distinct_prime_factors_l3350_335082

-- Define the set of distinct prime factors
def distinct_prime_factors (n : ℕ) : Set ℕ :=
  {p : ℕ | Nat.Prime p ∧ p ∣ n}

-- State the theorem
theorem same_distinct_prime_factors (k : ℕ) (h : k > 1) :
  let A := 2^k - 2
  let B := 2^k * A
  (distinct_prime_factors A = distinct_prime_factors B) ∧
  (distinct_prime_factors (A + 1) = distinct_prime_factors (B + 1)) :=
by
  sorry


end same_distinct_prime_factors_l3350_335082


namespace carson_seed_amount_l3350_335083

-- Define variables
variable (seed : ℝ)
variable (fertilizer : ℝ)

-- Define the conditions
def seed_fertilizer_ratio : Prop := seed = 3 * fertilizer
def total_amount : Prop := seed + fertilizer = 60

-- Theorem statement
theorem carson_seed_amount 
  (h1 : seed_fertilizer_ratio seed fertilizer)
  (h2 : total_amount seed fertilizer) :
  seed = 45 := by
  sorry

end carson_seed_amount_l3350_335083


namespace product_first_three_terms_l3350_335056

/-- An arithmetic sequence with given properties -/
def ArithmeticSequence (a : ℕ → ℕ) : Prop :=
  (a 8 = 20) ∧ (∀ n : ℕ, a (n + 1) = a n + 2)

/-- Theorem stating the product of the first three terms -/
theorem product_first_three_terms (a : ℕ → ℕ) (h : ArithmeticSequence a) :
  a 1 * a 2 * a 3 = 480 := by
  sorry


end product_first_three_terms_l3350_335056


namespace lower_selling_price_l3350_335084

theorem lower_selling_price 
  (cost : ℕ) 
  (higher_price lower_price : ℕ) 
  (h1 : cost = 400)
  (h2 : higher_price = 600)
  (h3 : (higher_price - cost) = (lower_price - cost) + (cost * 5 / 100)) :
  lower_price = 580 := by
sorry

end lower_selling_price_l3350_335084


namespace rectangle_longer_side_l3350_335020

/-- Given a circle with radius 6 cm tangent to three sides of a rectangle, 
    and the rectangle's area being three times the area of the circle,
    the length of the longer side of the rectangle is 9π cm. -/
theorem rectangle_longer_side (circle_radius : ℝ) (rectangle_area : ℝ) 
  (h1 : circle_radius = 6)
  (h2 : rectangle_area = 3 * Real.pi * circle_radius^2) : 
  rectangle_area / (2 * circle_radius) = 9 * Real.pi := by
  sorry

end rectangle_longer_side_l3350_335020


namespace summer_salutations_l3350_335075

/-- The number of sun salutation yoga poses Summer performs on weekdays -/
def poses_per_day : ℕ := 5

/-- The number of weekdays in a week -/
def weekdays_per_week : ℕ := 5

/-- The number of weeks in a year -/
def weeks_per_year : ℕ := 52

/-- The total number of sun salutations Summer performs in a year -/
def total_salutations : ℕ := poses_per_day * weekdays_per_week * weeks_per_year

/-- Theorem stating that Summer performs 1300 sun salutations throughout an entire year -/
theorem summer_salutations : total_salutations = 1300 := by
  sorry

end summer_salutations_l3350_335075


namespace max_profit_at_25_yuan_manager_decision_suboptimal_l3350_335078

/-- Profit function based on price reduction -/
def profit (x : ℝ) : ℝ := (2 * x - 20) * (40 - x)

/-- Initial daily sales -/
def initial_sales : ℝ := 20

/-- Initial profit per piece -/
def initial_profit_per_piece : ℝ := 40

/-- Sales increase per yuan of price reduction -/
def sales_increase_rate : ℝ := 2

/-- Theorem stating the maximum profit and corresponding price reduction -/
theorem max_profit_at_25_yuan :
  ∃ (max_reduction : ℝ) (max_profit : ℝ),
    max_reduction = 25 ∧
    max_profit = 1250 ∧
    ∀ x, 0 ≤ x ∧ x ≤ 40 → profit x ≤ max_profit :=
sorry

/-- Theorem comparing manager's decision to optimal decision -/
theorem manager_decision_suboptimal (manager_reduction : ℝ) (h : manager_reduction = 15) :
  ∃ (optimal_reduction : ℝ) (optimal_profit : ℝ),
    optimal_reduction ≠ manager_reduction ∧
    optimal_profit > profit manager_reduction :=
sorry

end max_profit_at_25_yuan_manager_decision_suboptimal_l3350_335078


namespace answer_key_combinations_l3350_335042

/-- The number of answer choices for each multiple-choice question -/
def multiple_choice_options : ℕ := 4

/-- The number of true-false questions -/
def true_false_questions : ℕ := 5

/-- The number of multiple-choice questions -/
def multiple_choice_questions : ℕ := 2

/-- The total number of possible true-false answer combinations -/
def total_true_false_combinations : ℕ := 2^true_false_questions

/-- The number of true-false combinations where all answers are the same -/
def same_answer_combinations : ℕ := 2

/-- The number of valid true-false combinations (excluding all same answers) -/
def valid_true_false_combinations : ℕ := total_true_false_combinations - same_answer_combinations

/-- The total number of possible multiple-choice answer combinations -/
def multiple_choice_combinations : ℕ := multiple_choice_options^multiple_choice_questions

/-- The theorem stating the total number of ways to create the answer key -/
theorem answer_key_combinations : 
  valid_true_false_combinations * multiple_choice_combinations = 480 := by
  sorry

end answer_key_combinations_l3350_335042


namespace min_sum_absolute_values_l3350_335030

theorem min_sum_absolute_values :
  (∀ x : ℝ, |x + 3| + |x + 4| + |x + 6| ≥ 3) ∧
  (∃ x : ℝ, |x + 3| + |x + 4| + |x + 6| = 3) :=
by sorry

end min_sum_absolute_values_l3350_335030


namespace solution_set_for_negative_one_range_of_a_l3350_335019

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x + a| + |x - 2|

-- Part 1
theorem solution_set_for_negative_one (x : ℝ) :
  (f (-1) x ≥ 6) ↔ (x ≤ -1 ∨ x ≥ 3) := by sorry

-- Part 2
theorem range_of_a (a : ℝ) :
  (∀ x, f a x ≥ 3*a^2 - |2 - x|) → (-1 ≤ a ∧ a ≤ 4/3) := by sorry

end solution_set_for_negative_one_range_of_a_l3350_335019


namespace upgraded_sensor_fraction_l3350_335017

/-- Represents a satellite with modular units and sensors. -/
structure Satellite where
  units : ℕ
  non_upgraded_per_unit : ℕ
  total_upgraded : ℕ
  non_upgraded_ratio : non_upgraded_per_unit = total_upgraded / 4

/-- The fraction of upgraded sensors on the satellite is 1/7. -/
theorem upgraded_sensor_fraction (s : Satellite) (h : s.units = 24) :
  s.total_upgraded / (s.units * s.non_upgraded_per_unit + s.total_upgraded) = 1 / 7 := by
  sorry

end upgraded_sensor_fraction_l3350_335017


namespace min_cars_in_group_l3350_335048

theorem min_cars_in_group (total : ℕ) 
  (no_ac : ℕ) 
  (racing_stripes : ℕ) 
  (ac_no_stripes : ℕ) : 
  no_ac = 47 →
  racing_stripes ≥ 53 →
  ac_no_stripes ≤ 47 →
  total ≥ 100 :=
by
  sorry

end min_cars_in_group_l3350_335048


namespace min_shaded_triangles_theorem_l3350_335069

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  sideLength : ℕ

/-- Represents the division of a large equilateral triangle into smaller ones -/
structure TriangleDivision where
  largeSideLength : ℕ
  smallSideLength : ℕ

/-- Calculates the number of intersection points in a triangle division -/
def intersectionPoints (d : TriangleDivision) : ℕ :=
  let n : ℕ := d.largeSideLength / d.smallSideLength + 1
  n * (n + 1) / 2

/-- Calculates the minimum number of smaller triangles needed to be shaded -/
def minShadedTriangles (d : TriangleDivision) : ℕ :=
  (intersectionPoints d + 2) / 3

/-- The main theorem to prove -/
theorem min_shaded_triangles_theorem (t : EquilateralTriangle) (d : TriangleDivision) :
  t.sideLength = 8 →
  d.largeSideLength = 8 →
  d.smallSideLength = 1 →
  minShadedTriangles d = 15 := by
  sorry

end min_shaded_triangles_theorem_l3350_335069


namespace curve_scaling_transformation_l3350_335097

/-- Given a curve C with equation x² + y² = 1 and a scaling transformation,
    prove that the resulting curve C'' has the equation x² + y²/4 = 1 -/
theorem curve_scaling_transformation (x y x'' y'' : ℝ) :
  (x^2 + y^2 = 1) →    -- Equation of curve C
  (x'' = x) →          -- x-coordinate transformation
  (y'' = 2*y) →        -- y-coordinate transformation
  (x''^2 + y''^2/4 = 1) -- Equation of curve C''
:= by sorry

end curve_scaling_transformation_l3350_335097


namespace smallest_three_digit_multiple_of_17_l3350_335055

theorem smallest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n → 102 ≤ n :=
by
  sorry

end smallest_three_digit_multiple_of_17_l3350_335055


namespace simplify_product_l3350_335088

theorem simplify_product (b : R) [CommRing R] :
  (2 : R) * b * (3 : R) * b^2 * (4 : R) * b^3 * (5 : R) * b^4 * (6 : R) * b^5 = (720 : R) * b^15 := by
  sorry

end simplify_product_l3350_335088


namespace tangent_line_at_2_minus_6_l3350_335062

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x - 16

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Theorem statement
theorem tangent_line_at_2_minus_6 :
  let x₀ : ℝ := 2
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  let b : ℝ := y₀ - m * x₀
  (∀ x y, y = m * x + b ↔ y - y₀ = m * (x - x₀)) ∧ 
  y₀ = -6 ∧ 
  m = 13 ∧ 
  b = -32 := by sorry

end tangent_line_at_2_minus_6_l3350_335062


namespace sum_of_coefficients_cube_expansion_l3350_335065

theorem sum_of_coefficients_cube_expansion : 
  ∃ (a b c d e : ℚ), 
    (∀ x, 1000 * x^3 + 27 = (a*x + b) * (c*x^2 + d*x + e)) ∧
    a + b + c + d + e = 92 := by
  sorry

end sum_of_coefficients_cube_expansion_l3350_335065


namespace supplementary_angles_difference_l3350_335010

theorem supplementary_angles_difference (a b : ℝ) : 
  a + b = 180 →  -- supplementary angles
  a / b = 5 / 3 →  -- ratio of 5:3
  abs (a - b) = 45 :=  -- positive difference
by sorry

end supplementary_angles_difference_l3350_335010


namespace china_gdp_scientific_notation_l3350_335094

theorem china_gdp_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 
    1 ≤ |a| ∧ |a| < 10 ∧ 
    n = 5 ∧
    827000 = a * (10 : ℝ)^n ∧
    a = 8.27 := by
  sorry

end china_gdp_scientific_notation_l3350_335094


namespace diamond_six_three_l3350_335022

/-- Define the diamond operation for real numbers -/
noncomputable def diamond (x y : ℝ) : ℝ :=
  sorry

/-- Properties of the diamond operation -/
axiom diamond_zero (x : ℝ) : diamond x 0 = 2 * x
axiom diamond_comm (x y : ℝ) : diamond x y = diamond y x
axiom diamond_succ (x y : ℝ) : diamond (x + 1) y = diamond x y * (y + 2)

/-- Theorem: The value of 6 ◇ 3 is 93750 -/
theorem diamond_six_three : diamond 6 3 = 93750 := by
  sorry

end diamond_six_three_l3350_335022


namespace prob_2012_higher_than_2011_l3350_335081

/-- Probability of guessing the correct answer to each question -/
def p : ℝ := 0.25

/-- Probability of guessing the incorrect answer to each question -/
def q : ℝ := 1 - p

/-- Number of questions in the 2011 exam -/
def n_2011 : ℕ := 20

/-- Number of correct answers required to pass in 2011 -/
def k_2011 : ℕ := 3

/-- Number of questions in the 2012 exam -/
def n_2012 : ℕ := 40

/-- Number of correct answers required to pass in 2012 -/
def k_2012 : ℕ := 6

/-- Probability of passing the exam in 2011 -/
def prob_2011 : ℝ := 1 - (Finset.sum (Finset.range k_2011) (λ i => Nat.choose n_2011 i * p^i * q^(n_2011 - i)))

/-- Probability of passing the exam in 2012 -/
def prob_2012 : ℝ := 1 - (Finset.sum (Finset.range k_2012) (λ i => Nat.choose n_2012 i * p^i * q^(n_2012 - i)))

/-- Theorem stating that the probability of passing in 2012 is higher than in 2011 -/
theorem prob_2012_higher_than_2011 : prob_2012 > prob_2011 := by
  sorry

end prob_2012_higher_than_2011_l3350_335081


namespace sequence_properties_arithmetic_sequence_l3350_335007

def a_n (n a : ℕ+) : ℚ := n / (n + a)

theorem sequence_properties (a : ℕ+) :
  (∃ r : ℚ, a_n 1 a * r = a_n 3 a ∧ a_n 3 a * r = a_n 15 a) →
  a = 9 :=
sorry

theorem arithmetic_sequence (a k : ℕ+) :
  k ≥ 3 →
  (a_n 1 a + a_n k a = 2 * a_n 2 a) →
  ((a = 1 ∧ k = 5) ∨ (a = 2 ∧ k = 4)) :=
sorry

end sequence_properties_arithmetic_sequence_l3350_335007


namespace largest_whole_number_times_eight_less_than_150_l3350_335026

theorem largest_whole_number_times_eight_less_than_150 :
  ∃ y : ℕ, y = 18 ∧ 8 * y < 150 ∧ ∀ z : ℕ, z > y → 8 * z ≥ 150 :=
by sorry

end largest_whole_number_times_eight_less_than_150_l3350_335026
