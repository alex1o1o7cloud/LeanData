import Mathlib

namespace eldest_age_l921_92193

theorem eldest_age (x : ℝ) (h1 : 5*x - 7 + 7*x - 7 + 8*x - 7 = 59) : 8*x = 32 := by
  sorry

#check eldest_age

end eldest_age_l921_92193


namespace solution_set_of_f_positive_l921_92170

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 5*x + 6

-- State the theorem
theorem solution_set_of_f_positive :
  {x : ℝ | f x > 0} = {x : ℝ | x > 3 ∨ x < 2} := by sorry

end solution_set_of_f_positive_l921_92170


namespace average_weight_b_c_l921_92173

theorem average_weight_b_c (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 45)
  (h2 : (a + b) / 2 = 41)
  (h3 : b = 33) :
  (b + c) / 2 = 43 := by
sorry

end average_weight_b_c_l921_92173


namespace transformation_is_left_shift_l921_92105

/-- A function representing a horizontal shift transformation -/
def horizontalShift (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ :=
  λ x => f (x + shift)

/-- The original function composition -/
def originalFunc (f : ℝ → ℝ) : ℝ → ℝ :=
  λ x => f (2*x - 1)

/-- The transformed function composition -/
def transformedFunc (f : ℝ → ℝ) : ℝ → ℝ :=
  λ x => f (2*x + 1)

theorem transformation_is_left_shift (f : ℝ → ℝ) :
  transformedFunc f = horizontalShift (originalFunc f) 1 := by
  sorry

end transformation_is_left_shift_l921_92105


namespace range_of_a_l921_92199

-- Define the set P
def P : Set ℝ := {x : ℝ | x^2 ≤ 1}

-- Define the set M
def M (a : ℝ) : Set ℝ := {a}

-- Theorem statement
theorem range_of_a (a : ℝ) : P ∪ M a = P → a ∈ Set.Icc (-1) 1 := by
  sorry

end range_of_a_l921_92199


namespace continuous_function_inequality_l921_92157

theorem continuous_function_inequality (f : ℝ → ℝ) (hf : Continuous f) 
  (h : ∀ x : ℝ, (x - 1) * (deriv f x) < 0) : f 0 + f 2 < 2 * f 1 := by
  sorry

end continuous_function_inequality_l921_92157


namespace claire_male_pets_l921_92110

theorem claire_male_pets (total_pets : ℕ) (gerbils : ℕ) (hamsters : ℕ)
  (h_total : total_pets = 92)
  (h_only_gerbils_hamsters : total_pets = gerbils + hamsters)
  (h_gerbils : gerbils = 68)
  (h_male_gerbils : ℕ → ℕ := λ x => x / 4)
  (h_male_hamsters : ℕ → ℕ := λ x => x / 3)
  : h_male_gerbils gerbils + h_male_hamsters hamsters = 25 := by
  sorry

end claire_male_pets_l921_92110


namespace cannon_firing_time_l921_92126

/-- Represents a cannon with a specified firing rate and number of shots -/
structure Cannon where
  firing_rate : ℕ  -- shots per minute
  total_shots : ℕ

/-- Calculates the time taken to fire all shots for a given cannon -/
def time_to_fire (c : Cannon) : ℕ :=
  c.total_shots - 1

/-- The cannon from the problem -/
def test_cannon : Cannon :=
  { firing_rate := 1, total_shots := 60 }

/-- Theorem stating that the time to fire all shots is 59 minutes -/
theorem cannon_firing_time :
  time_to_fire test_cannon = 59 := by sorry

end cannon_firing_time_l921_92126


namespace sqrt_product_sqrt_l921_92136

theorem sqrt_product_sqrt : Real.sqrt (49 * Real.sqrt 25) = 5 * Real.sqrt 7 := by
  sorry

end sqrt_product_sqrt_l921_92136


namespace function_decreasing_interval_l921_92133

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^2 * (a*x + b)

-- Define the derivative of f(x)
def f_derivative (a b x : ℝ) : ℝ := 3*a*x^2 + 2*b*x

theorem function_decreasing_interval (a b : ℝ) :
  (∀ x, f_derivative a b x = 0 → x = 2) →  -- Extremum at x = 2
  (f_derivative a b 1 = -3) →              -- Tangent line parallel to 3x + y = 0
  (∀ x ∈ (Set.Ioo 0 2), f_derivative a b x < 0) ∧ 
  (∀ x ∉ (Set.Icc 0 2), f_derivative a b x ≥ 0) :=
by sorry

end function_decreasing_interval_l921_92133


namespace thick_line_segments_length_l921_92187

theorem thick_line_segments_length
  (perimeter_quadrilaterals : ℝ)
  (perimeter_triangles : ℝ)
  (perimeter_large_triangle : ℝ)
  (h1 : perimeter_quadrilaterals = 25)
  (h2 : perimeter_triangles = 20)
  (h3 : perimeter_large_triangle = 19) :
  (perimeter_quadrilaterals + perimeter_triangles - perimeter_large_triangle) / 2 = 13 :=
sorry

end thick_line_segments_length_l921_92187


namespace smallest_denominators_sum_l921_92120

theorem smallest_denominators_sum (p q : ℕ) (h : q > 0) :
  (∃ k : ℕ, Nat.div p q = k ∧ 
   ∃ r : ℕ, r < q ∧ 
   ∃ n : ℕ, Nat.div (r * 1000 + 171) q = n ∧
   ∃ s : ℕ, s < q ∧ Nat.div (s * 1000 + 171) q = n) →
  (∃ q1 q2 : ℕ, q1 < q2 ∧ 
   (∀ q' : ℕ, q' ≠ q1 ∧ q' ≠ q2 → q' > q2) ∧
   q1 + q2 = 99) :=
sorry

end smallest_denominators_sum_l921_92120


namespace max_value_constraint_l921_92198

theorem max_value_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3 * x + 4 * y < 72) :
  x * y * (72 - 3 * x - 4 * y) ≤ 1152 := by
sorry

end max_value_constraint_l921_92198


namespace p_false_and_q_true_l921_92104

-- Define proposition p
def p : Prop := ∀ x : ℝ, 2 * x^2 - 2 * x + 1 ≤ 0

-- Define proposition q
def q : Prop := ∃ x : ℝ, Real.sin x + Real.cos x = Real.sqrt 2

-- Theorem stating that p is false and q is true
theorem p_false_and_q_true : ¬p ∧ q := by sorry

end p_false_and_q_true_l921_92104


namespace complex_number_in_fourth_quadrant_l921_92134

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (1 - 2*I) * (2 + I)
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end complex_number_in_fourth_quadrant_l921_92134


namespace coach_team_division_l921_92138

/-- Given a total number of athletes and a maximum team size, 
    calculate the minimum number of teams needed. -/
def min_teams (total_athletes : ℕ) (max_team_size : ℕ) : ℕ :=
  ((total_athletes + max_team_size - 1) / max_team_size : ℕ)

theorem coach_team_division (total_athletes max_team_size : ℕ) 
  (h1 : total_athletes = 30) (h2 : max_team_size = 12) :
  min_teams total_athletes max_team_size = 3 := by
  sorry

#eval min_teams 30 12

end coach_team_division_l921_92138


namespace stratified_sampling_medium_supermarkets_l921_92180

theorem stratified_sampling_medium_supermarkets 
  (total_large : ℕ) (total_medium : ℕ) (total_small : ℕ) (sample_size : ℕ) :
  total_large = 200 →
  total_medium = 400 →
  total_small = 1400 →
  sample_size = 100 →
  (total_large + total_medium + total_small) * (sample_size / (total_large + total_medium + total_small)) = sample_size →
  total_medium * (sample_size / (total_large + total_medium + total_small)) = 20 :=
by
  sorry

end stratified_sampling_medium_supermarkets_l921_92180


namespace work_completion_time_l921_92144

/-- The number of days it takes for the original number of people to complete the work -/
def days_to_complete_work (original_people : ℕ) (total_work : ℝ) : ℕ :=
  16

theorem work_completion_time 
  (original_people : ℕ) 
  (total_work : ℝ) 
  (h : (2 * original_people : ℝ) * 4 = total_work / 2) : 
  days_to_complete_work original_people total_work = 16 := by
  sorry

end work_completion_time_l921_92144


namespace solve_equation_l921_92163

theorem solve_equation (m n : ℕ) (h1 : ((1^m) / (5^m)) * ((1^n) / (4^n)) = 1 / (2 * (10^31))) (h2 : m = 31) : n = 16 := by
  sorry

end solve_equation_l921_92163


namespace nebula_boys_count_total_students_correct_total_students_by_gender_correct_l921_92152

/-- Represents a school in the science camp. -/
inductive School
| Orion
| Nebula
| Galaxy

/-- Represents the gender of a student. -/
inductive Gender
| Boy
| Girl

/-- Represents the distribution of students in the science camp. -/
structure CampDistribution where
  total_students : ℕ
  total_boys : ℕ
  total_girls : ℕ
  students_by_school : School → ℕ
  boys_by_school : School → ℕ

/-- The actual distribution of students in the science camp. -/
def camp_data : CampDistribution :=
  { total_students := 150,
    total_boys := 84,
    total_girls := 66,
    students_by_school := fun s => match s with
      | School.Orion => 70
      | School.Nebula => 50
      | School.Galaxy => 30,
    boys_by_school := fun s => match s with
      | School.Orion => 30
      | _ => 0  -- We don't know these values yet
  }

/-- Theorem stating that the number of boys from Nebula Middle School is 32. -/
theorem nebula_boys_count (d : CampDistribution) (h : d = camp_data) :
  d.boys_by_school School.Nebula = 32 := by
  sorry

/-- Verify that the total number of students is correct. -/
theorem total_students_correct (d : CampDistribution) (h : d = camp_data) :
  d.total_students = d.students_by_school School.Orion +
                     d.students_by_school School.Nebula +
                     d.students_by_school School.Galaxy := by
  sorry

/-- Verify that the total number of students by gender is correct. -/
theorem total_students_by_gender_correct (d : CampDistribution) (h : d = camp_data) :
  d.total_students = d.total_boys + d.total_girls := by
  sorry

end nebula_boys_count_total_students_correct_total_students_by_gender_correct_l921_92152


namespace earnings_per_dog_l921_92117

def dogs_monday_wednesday_friday : ℕ := 7
def dogs_tuesday : ℕ := 12
def dogs_thursday : ℕ := 9
def weekly_earnings : ℕ := 210

def total_dogs : ℕ := dogs_monday_wednesday_friday * 3 + dogs_tuesday + dogs_thursday

theorem earnings_per_dog :
  weekly_earnings / total_dogs = 5 := by sorry

end earnings_per_dog_l921_92117


namespace rectangle_matchsticks_distribution_l921_92102

/-- Calculates the total number of matchsticks in the rectangle -/
def total_matchsticks (length width : ℕ) : ℕ :=
  (length + 1) * width + (width + 1) * length

/-- Checks if the number of matchsticks can be equally distributed among a given number of children -/
def is_valid_distribution (total_sticks children : ℕ) : Prop :=
  children > 100 ∧ total_sticks % children = 0

theorem rectangle_matchsticks_distribution :
  let total := total_matchsticks 60 10
  ∃ (n : ℕ), is_valid_distribution total n ∧
    ∀ (m : ℕ), m > n → ¬(is_valid_distribution total m) ∧
    n = 127 := by
  sorry

end rectangle_matchsticks_distribution_l921_92102


namespace monotone_decreasing_implies_g_nonnegative_l921_92111

/-- A function that is monotonically decreasing on an interval -/
def MonotonicallyDecreasing (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f y < f x

/-- The function f(x) = x^3 + ax^2 + bx + c -/
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

/-- The function g(x) = 3x^2 + 2ax + b -/
def g (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

/-- Theorem: If f(x) is monotonically decreasing on (0, 1), then g(0) * g(1) ≥ 0 -/
theorem monotone_decreasing_implies_g_nonnegative (a b c : ℝ) :
  MonotonicallyDecreasing (f a b c) 0 1 → g a b 0 * g a b 1 ≥ 0 := by
  sorry

end monotone_decreasing_implies_g_nonnegative_l921_92111


namespace january_salary_is_2900_l921_92154

/-- Calculates the salary for January given the average salaries and May's salary -/
def january_salary (avg_jan_to_apr avg_feb_to_may may_salary : ℚ) : ℚ :=
  4 * avg_jan_to_apr - (4 * avg_feb_to_may - may_salary)

/-- Theorem stating that the salary for January is 2900 given the provided conditions -/
theorem january_salary_is_2900 :
  january_salary 8000 8900 6500 = 2900 := by
  sorry

end january_salary_is_2900_l921_92154


namespace divisibility_by_seven_l921_92197

theorem divisibility_by_seven (n : ℕ) : 
  ∃ k : ℤ, ((-8)^(2019 : ℕ) + (-8)^(2018 : ℕ)) = 7 * k := by
  sorry

end divisibility_by_seven_l921_92197


namespace max_chesslike_subsquares_l921_92182

/-- Represents the color of a square on the board -/
inductive Color
| Red
| Green

/-- Represents a 6x6 board -/
def Board := Fin 6 → Fin 6 → Color

/-- Checks if four adjacent squares in a given direction are of the same color -/
def fourAdjacentSameColor (board : Board) : Bool := sorry

/-- Checks if a 2x2 subsquare is chesslike -/
def isChesslike (board : Board) (row col : Fin 5) : Bool := sorry

/-- Counts the number of chesslike 2x2 subsquares on the board -/
def countChesslike (board : Board) : Nat := sorry

/-- Theorem: The maximal number of chesslike 2x2 subsquares on a 6x6 board 
    with the given constraints is 25 -/
theorem max_chesslike_subsquares (board : Board) 
  (h : ¬fourAdjacentSameColor board) : 
  (∃ (b : Board), ¬fourAdjacentSameColor b ∧ countChesslike b = 25) ∧ 
  (∀ (b : Board), ¬fourAdjacentSameColor b → countChesslike b ≤ 25) := by
  sorry

end max_chesslike_subsquares_l921_92182


namespace triangle_inequality_third_side_bounds_l921_92103

theorem triangle_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ c : ℝ, 0 < c ∧ a - b < c ∧ c < a + b :=
by
  sorry

theorem third_side_bounds (side1 side2 : ℝ) 
  (h1 : side1 = 6) (h2 : side2 = 10) :
  ∃ side3 : ℝ, 4 < side3 ∧ side3 < 16 :=
by
  sorry

end triangle_inequality_third_side_bounds_l921_92103


namespace dj_oldies_ratio_l921_92178

/-- Represents the number of song requests for each genre and the total requests --/
structure SongRequests where
  total : Nat
  electropop : Nat
  dance : Nat
  rock : Nat
  oldies : Nat
  rap : Nat

/-- Calculates the number of DJ's choice songs --/
def djChoice (s : SongRequests) : Nat :=
  s.total - (s.electropop + s.rock + s.oldies + s.rap)

/-- Theorem stating the ratio of DJ's choice to oldies songs --/
theorem dj_oldies_ratio (s : SongRequests) : 
  s.total = 30 ∧ 
  s.electropop = s.total / 2 ∧ 
  s.dance = s.electropop / 3 ∧ 
  s.rock = 5 ∧ 
  s.oldies = s.rock - 3 ∧ 
  s.rap = 2 → 
  (djChoice s : Int) / s.oldies = 3 := by
  sorry

end dj_oldies_ratio_l921_92178


namespace hyperbola_max_y_coordinate_l921_92160

/-- Given a hyperbola with equation x²/4 - y²/b = 1 where b > 0,
    a point P(x,y) in the first quadrant satisfying |OP| = |F₁F₂|/2,
    and eccentricity e ∈ (1, 2], prove that the maximum y-coordinate is 3. -/
theorem hyperbola_max_y_coordinate (b : ℝ) (x y : ℝ) (e : ℝ) :
  b > 0 →
  x > 0 →
  y > 0 →
  x^2 / 4 - y^2 / b = 1 →
  x^2 + y^2 = 4 + b →
  1 < e ∧ e ≤ 2 →
  y ≤ 3 :=
sorry

end hyperbola_max_y_coordinate_l921_92160


namespace factorization_equality_l921_92148

theorem factorization_equality (m n : ℝ) : 2*m*n^2 - 12*m*n + 18*m = 2*m*(n-3)^2 := by
  sorry

end factorization_equality_l921_92148


namespace bear_hunting_problem_l921_92156

theorem bear_hunting_problem (bear_need : ℕ) (cub_need : ℕ) (num_cubs : ℕ) (animals_per_day : ℕ) : 
  bear_need = 210 →
  cub_need = 35 →
  num_cubs = 4 →
  animals_per_day = 10 →
  (bear_need + cub_need * num_cubs) / 7 / animals_per_day = 5 := by
sorry

end bear_hunting_problem_l921_92156


namespace tan_function_value_l921_92196

theorem tan_function_value (f : ℝ → ℝ) :
  (∀ x, f x = Real.tan (2 * x + π / 3)) →
  f (25 * π / 6) = -Real.sqrt 3 := by
sorry

end tan_function_value_l921_92196


namespace fixed_point_exponential_function_l921_92112

theorem fixed_point_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x + 2) + 1
  f (-2) = 2 := by
  sorry

end fixed_point_exponential_function_l921_92112


namespace T_is_three_rays_with_common_point_l921_92100

/-- The set T of points in the coordinate plane satisfying the given conditions -/
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (5 = x - 1 ∧ y + 3 ≤ 5) ∨
               (5 = y + 3 ∧ x - 1 ≤ 5) ∨
               (x - 1 = y + 3 ∧ 5 ≤ x - 1)}

/-- A ray in the coordinate plane -/
structure Ray where
  start : ℝ × ℝ
  direction : ℝ × ℝ

/-- The three rays that make up set T -/
def rays : List Ray :=
  [{ start := (6, 2), direction := (0, -1) },   -- Vertical ray downward
   { start := (6, 2), direction := (-1, 0) },   -- Horizontal ray leftward
   { start := (6, 2), direction := (1, 1) }]    -- Diagonal ray upward

/-- Theorem stating that T consists of three rays with a common point -/
theorem T_is_three_rays_with_common_point :
  ∃ (common_point : ℝ × ℝ) (rs : List Ray),
    common_point = (6, 2) ∧
    rs.length = 3 ∧
    (∀ r ∈ rs, r.start = common_point) ∧
    T = ⋃ r ∈ rs, {p | ∃ t ≥ 0, p = r.start + t • r.direction} :=
  sorry

end T_is_three_rays_with_common_point_l921_92100


namespace white_is_lightest_l921_92124

-- Define the puppy type
inductive Puppy
| White
| Black
| Yellowy
| Spotted

-- Define the "lighter than" relation
def lighterThan : Puppy → Puppy → Prop := sorry

-- State the theorem
theorem white_is_lightest :
  (lighterThan Puppy.White Puppy.Black) ∧
  (lighterThan Puppy.Black Puppy.Yellowy) ∧
  (lighterThan Puppy.Yellowy Puppy.Spotted) →
  ∀ p : Puppy, p ≠ Puppy.White → lighterThan Puppy.White p :=
sorry

end white_is_lightest_l921_92124


namespace least_multiple_of_35_over_500_l921_92161

theorem least_multiple_of_35_over_500 :
  (∀ k : ℕ, k * 35 > 500 → k * 35 ≥ 525) ∧ 525 > 500 ∧ ∃ n : ℕ, 525 = n * 35 :=
sorry

end least_multiple_of_35_over_500_l921_92161


namespace product_of_specific_integers_l921_92186

theorem product_of_specific_integers : 
  ∀ (a b : ℤ), a = 32 ∧ b = 32 ∧ a % 10 ≠ 0 ∧ b % 10 ≠ 0 → a * b = 1024 := by
  sorry

end product_of_specific_integers_l921_92186


namespace log_46328_between_consecutive_integers_l921_92125

theorem log_46328_between_consecutive_integers :
  ∃ (a b : ℤ), a + 1 = b ∧ (a : ℝ) < Real.log 46328 / Real.log 10 ∧ (Real.log 46328 / Real.log 10 < b) ∧ a + b = 9 := by
  sorry

end log_46328_between_consecutive_integers_l921_92125


namespace science_club_election_l921_92106

/-- The number of ways to select k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

theorem science_club_election (total : ℕ) (former : ℕ) (board_size : ℕ)
  (h1 : total = 18)
  (h2 : former = 8)
  (h3 : board_size = 6)
  (h4 : former ≤ total)
  (h5 : board_size ≤ total) :
  binomial total board_size - binomial (total - former) board_size = 18354 := by
  sorry

end science_club_election_l921_92106


namespace mean_problem_l921_92115

theorem mean_problem (x y : ℝ) : 
  (6 + 14 + x + 17 + 9 + y + 10) / 7 = 13 → x + y = 35 := by
  sorry

end mean_problem_l921_92115


namespace wallpaper_overlap_area_l921_92147

theorem wallpaper_overlap_area
  (total_area : ℝ)
  (double_layer_area : ℝ)
  (triple_layer_area : ℝ)
  (h1 : total_area = 300)
  (h2 : double_layer_area = 38)
  (h3 : triple_layer_area = 41) :
  total_area - 2 * double_layer_area - 3 * triple_layer_area = 101 :=
by
  sorry

end wallpaper_overlap_area_l921_92147


namespace figure_to_square_possible_l921_92121

/-- Represents a geometric figure on a grid paper -/
structure GridFigure where
  -- Add necessary fields to represent the figure

/-- Represents a part of the figure after cutting -/
structure FigurePart where
  -- Add necessary fields to represent a part

/-- Represents a square -/
structure Square where
  -- Add necessary fields to represent a square

/-- Function to check if a list of parts can be reassembled into a square -/
def can_form_square (parts : List FigurePart) : Bool :=
  sorry

/-- Function to check if all parts are triangles -/
def all_triangles (parts : List FigurePart) : Bool :=
  sorry

/-- Theorem stating that the figure can be cut and reassembled into a square under given conditions -/
theorem figure_to_square_possible (fig : GridFigure) : 
  (∃ (parts : List FigurePart), parts.length ≤ 4 ∧ can_form_square parts) ∧
  (∃ (parts : List FigurePart), parts.length ≤ 5 ∧ all_triangles parts ∧ can_form_square parts) :=
by
  sorry

end figure_to_square_possible_l921_92121


namespace largest_prime_divisor_check_l921_92116

theorem largest_prime_divisor_check (n : ℕ) (h1 : 1000 ≤ n) (h2 : n ≤ 1050) :
  ∀ p : ℕ, Prime p → p ∣ n → p ≤ 31 := by
  sorry

end largest_prime_divisor_check_l921_92116


namespace tonya_needs_22_hamburgers_l921_92141

/-- The number of hamburgers Tonya needs to eat to beat last year's winner -/
def hamburgers_to_beat_record (ounces_per_hamburger : ℕ) (last_year_winner_ounces : ℕ) : ℕ :=
  (last_year_winner_ounces / ounces_per_hamburger) + 1

/-- Theorem stating that Tonya needs to eat 22 hamburgers to beat last year's winner -/
theorem tonya_needs_22_hamburgers :
  hamburgers_to_beat_record 4 84 = 22 := by
  sorry

end tonya_needs_22_hamburgers_l921_92141


namespace billys_age_l921_92101

theorem billys_age (billy joe : ℕ) 
  (h1 : billy = 3 * joe) 
  (h2 : billy + joe = 52) : 
  billy = 39 := by
sorry

end billys_age_l921_92101


namespace power_equation_solution_l921_92119

theorem power_equation_solution : 5^3 - 7 = 6^2 + 82 := by sorry

end power_equation_solution_l921_92119


namespace martha_apples_l921_92174

theorem martha_apples (jane_apples james_apples martha_remaining martha_to_give : ℕ) :
  jane_apples = 5 →
  james_apples = jane_apples + 2 →
  martha_remaining = 4 →
  martha_to_give = 4 →
  jane_apples + james_apples + martha_remaining + martha_to_give = 20 :=
by sorry

end martha_apples_l921_92174


namespace parabola_coefficient_l921_92108

/-- Given a parabola y = ax^2 + bx + c with vertex at (h, 2h) and y-intercept at (0, -3h), where h ≠ 0, prove that b = 10 -/
theorem parabola_coefficient (a b c h : ℝ) : 
  h ≠ 0 → 
  (∀ x, a * x^2 + b * x + c = a * (x - h)^2 + 2 * h) → 
  a * 0^2 + b * 0 + c = -3 * h → 
  b = 10 := by sorry

end parabola_coefficient_l921_92108


namespace sufficient_not_necessary_condition_l921_92194

theorem sufficient_not_necessary_condition (x y z : ℝ) :
  (∀ z ≠ 0, x * z^2024 < y * z^2024 → x < y) ∧
  ¬(∀ x y : ℝ, x < y → ∀ z : ℝ, x * z^2024 < y * z^2024) :=
by sorry

end sufficient_not_necessary_condition_l921_92194


namespace square_area_from_diagonal_l921_92172

theorem square_area_from_diagonal (d : ℝ) (h : d = 12) : 
  (d^2 / 2 : ℝ) = 72 := by
  sorry

end square_area_from_diagonal_l921_92172


namespace max_product_under_constraint_l921_92143

theorem max_product_under_constraint (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3*a + 2*b = 1) :
  a * b ≤ 1/24 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 3*a₀ + 2*b₀ = 1 ∧ a₀ * b₀ = 1/24 :=
by sorry

end max_product_under_constraint_l921_92143


namespace even_q_l921_92139

theorem even_q (p q : ℕ) 
  (h1 : ∃ (n : ℕ), n^2 = 2*p - q) 
  (h2 : ∃ (m : ℕ), m^2 = 2*p + q) : 
  Even q := by
sorry

end even_q_l921_92139


namespace periodic_sequence_sum_l921_92135

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- A sequence is periodic with period T if a_{n+T} = a_n for all n -/
def IsPeriodic (a : Sequence) (T : ℕ) : Prop :=
  ∀ n, a (n + T) = a n

/-- The sum of the first n terms of a sequence -/
def SequenceSum (a : Sequence) (n : ℕ) : ℝ :=
  (Finset.range n).sum a

/-- Theorem: For a periodic sequence with period T, 
    the sum of m terms can be expressed in terms of T and r -/
theorem periodic_sequence_sum 
  (a : Sequence) (T m q r : ℕ) 
  (h_periodic : IsPeriodic a T) 
  (h_smallest : ∀ k, 0 < k → k < T → ¬IsPeriodic a k)
  (h_pos : 0 < T ∧ 0 < m ∧ 0 < q ∧ 0 < r)
  (h_decomp : m = q * T + r) :
  SequenceSum a m = q * SequenceSum a T + SequenceSum a r := by
  sorry

end periodic_sequence_sum_l921_92135


namespace shaded_circle_fraction_l921_92158

/-- Given a circle divided into equal regions, this theorem proves that
    if there are 4 regions and 1 is shaded, then the shaded fraction is 1/4. -/
theorem shaded_circle_fraction (total_regions shaded_regions : ℕ) :
  total_regions = 4 →
  shaded_regions = 1 →
  (shaded_regions : ℚ) / total_regions = 1 / 4 := by
  sorry

end shaded_circle_fraction_l921_92158


namespace cycle_alignment_l921_92169

def letter_cycle_length : ℕ := 5
def digit_cycle_length : ℕ := 4

theorem cycle_alignment :
  Nat.lcm letter_cycle_length digit_cycle_length = 20 := by
  sorry

end cycle_alignment_l921_92169


namespace sum_of_fifth_powers_l921_92184

theorem sum_of_fifth_powers (a b c d : ℝ) 
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) 
  (sum_condition : a + b + c + d = 3) 
  (sum_of_squares : a^2 + b^2 + c^2 + d^2 = 45) : 
  (a^5 / ((a-b)*(a-c)*(a-d))) + (b^5 / ((b-a)*(b-c)*(b-d))) + 
  (c^5 / ((c-a)*(c-b)*(c-d))) + (d^5 / ((d-a)*(d-b)*(d-c))) = -9 := by
sorry

end sum_of_fifth_powers_l921_92184


namespace mean_score_proof_l921_92137

theorem mean_score_proof (first_class_mean second_class_mean : ℝ)
                         (total_students : ℕ)
                         (class_ratio : ℚ) :
  first_class_mean = 90 →
  second_class_mean = 75 →
  total_students = 66 →
  class_ratio = 5 / 6 →
  ∃ (first_class_students second_class_students : ℕ),
    first_class_students + second_class_students = total_students ∧
    (first_class_students : ℚ) / (second_class_students : ℚ) = class_ratio ∧
    (first_class_mean * (first_class_students : ℝ) + 
     second_class_mean * (second_class_students : ℝ)) / (total_students : ℝ) = 82 :=
by sorry

end mean_score_proof_l921_92137


namespace axis_of_symmetry_shifted_cosine_l921_92109

open Real

theorem axis_of_symmetry_shifted_cosine :
  let f : ℝ → ℝ := λ x ↦ Real.sin (π / 2 - x)
  let g : ℝ → ℝ := λ x ↦ Real.cos (x + π / 6)
  ∀ x : ℝ, g (5 * π / 6 + x) = g (5 * π / 6 - x) := by
  sorry

end axis_of_symmetry_shifted_cosine_l921_92109


namespace coin_toss_probability_l921_92130

def binomial_coefficient (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (binomial_coefficient n k : ℚ) * p^k * (1 - p)^(n - k)

theorem coin_toss_probability :
  let n : ℕ := 10
  let k : ℕ := 3
  let p : ℚ := 1/2
  binomial_probability n k p = 15/128 := by
sorry

end coin_toss_probability_l921_92130


namespace triangle_agw_area_l921_92162

/-- Right triangle ABC with squares on legs and intersecting lines -/
structure RightTriangleWithSquares where
  -- Points
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  G : ℝ × ℝ
  W : ℝ × ℝ
  -- Conditions
  right_angle : (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0
  ac_length : (A.1 - C.1)^2 + (A.2 - C.2)^2 = 14^2
  bc_length : (B.1 - C.1)^2 + (B.2 - C.2)^2 = 28^2
  square_acde : D = (A.1 + C.2 - C.1, A.2 - C.2 + C.1) ∧ E = (C.1, A.2)
  square_cbfg : F = (C.1, B.2) ∧ G = (B.1, B.2 + B.1 - C.1)
  w_on_bc : ∃ t : ℝ, W = (t * B.1 + (1 - t) * C.1, t * B.2 + (1 - t) * C.2)
  w_on_af : ∃ s : ℝ, W = (s * A.1 + (1 - s) * F.1, s * A.2 + (1 - s) * F.2)

/-- The area of triangle AGW is 196 -/
theorem triangle_agw_area (t : RightTriangleWithSquares) : 
  abs ((t.A.1 * (t.G.2 - t.W.2) + t.G.1 * (t.W.2 - t.A.2) + t.W.1 * (t.A.2 - t.G.2)) / 2) = 196 := by
  sorry

end triangle_agw_area_l921_92162


namespace freezer_temperature_l921_92123

/-- Given a refrigerator with a refrigeration compartment and a freezer compartment,
    this theorem proves that if the refrigeration compartment is at 4°C and
    the freezer is 22°C colder, then the freezer temperature is -18°C. -/
theorem freezer_temperature
  (temp_refrigeration : ℝ)
  (temp_difference : ℝ)
  (h1 : temp_refrigeration = 4)
  (h2 : temp_difference = 22)
  : temp_refrigeration - temp_difference = -18 := by
  sorry

end freezer_temperature_l921_92123


namespace man_tshirt_count_l921_92175

/-- Given a man with pants and t-shirts, calculates the number of ways he can dress --/
def dressing_combinations (num_tshirts : ℕ) (num_pants : ℕ) : ℕ :=
  num_tshirts * num_pants

theorem man_tshirt_count :
  ∀ (num_pants : ℕ) (total_combinations : ℕ),
    num_pants = 9 →
    total_combinations = 72 →
    ∃ (num_tshirts : ℕ),
      dressing_combinations num_tshirts num_pants = total_combinations ∧
      num_tshirts = 8 :=
by
  sorry


end man_tshirt_count_l921_92175


namespace ajay_distance_theorem_l921_92150

/-- Calculates the distance traveled given speed and time -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

/-- Proves that given Ajay's speed of 50 km/hour and a travel time of 20 hours, 
    the distance traveled is 1000 km -/
theorem ajay_distance_theorem :
  let speed : ℝ := 50
  let time : ℝ := 20
  distance_traveled speed time = 1000 := by
sorry

end ajay_distance_theorem_l921_92150


namespace trig_identities_l921_92113

/-- Given an angle θ with vertex at the origin, initial side on positive x-axis,
    and terminal side on y = 1/2x (x ≤ 0), prove trigonometric identities. -/
theorem trig_identities (θ α : Real) : 
  (∃ x y : Real, x ≤ 0 ∧ y = (1/2) * x ∧ 
   Real.cos θ = x / Real.sqrt (x^2 + y^2) ∧ 
   Real.sin θ = y / Real.sqrt (x^2 + y^2)) →
  (Real.cos (π/2 + θ) = Real.sqrt 5 / 5) ∧
  (Real.cos (α + π/4) = Real.sin θ → 
   (Real.sin (2*α + π/4) = 7 * Real.sqrt 2 / 10 ∨ 
    Real.sin (2*α + π/4) = - Real.sqrt 2 / 10)) := by
  sorry

end trig_identities_l921_92113


namespace platform_length_l921_92168

/-- Given a train of length 300 meters, which takes 39 seconds to cross a platform
    and 16 seconds to cross a signal pole, prove that the length of the platform
    is 431.25 meters. -/
theorem platform_length
  (train_length : ℝ)
  (time_cross_platform : ℝ)
  (time_cross_pole : ℝ)
  (h1 : train_length = 300)
  (h2 : time_cross_platform = 39)
  (h3 : time_cross_pole = 16) :
  let speed := train_length / time_cross_pole
  let platform_length := speed * time_cross_platform - train_length
  platform_length = 431.25 := by
sorry

end platform_length_l921_92168


namespace average_of_c_and_d_l921_92191

theorem average_of_c_and_d (c d : ℝ) : 
  (4 + 6 + 8 + c + d) / 5 = 21 → (c + d) / 2 = 43.5 := by
  sorry

end average_of_c_and_d_l921_92191


namespace gcd_102_238_l921_92107

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end gcd_102_238_l921_92107


namespace equal_roots_quadratic_l921_92131

theorem equal_roots_quadratic (c : ℝ) :
  (∃ x : ℝ, x^2 + 6*x + c = 0 ∧ (∀ y : ℝ, y^2 + 6*y + c = 0 → y = x)) →
  c = 9 :=
by sorry

end equal_roots_quadratic_l921_92131


namespace april_flower_sale_l921_92114

/-- April's flower sale problem -/
theorem april_flower_sale 
  (price_per_rose : ℕ) 
  (initial_roses : ℕ) 
  (remaining_roses : ℕ) 
  (h1 : price_per_rose = 4)
  (h2 : initial_roses = 13)
  (h3 : remaining_roses = 4) :
  (initial_roses - remaining_roses) * price_per_rose = 36 := by
  sorry

end april_flower_sale_l921_92114


namespace lucy_integers_l921_92183

theorem lucy_integers (x y : ℤ) (h1 : 3 * x + 2 * y = 85) (h2 : x = 19 ∨ y = 19) : 
  (x = 19 ∧ y = 14) ∨ (y = 19 ∧ x = 14) :=
sorry

end lucy_integers_l921_92183


namespace simplify_expression_l921_92129

theorem simplify_expression (b : ℝ) : 3*b*(3*b^2 + 2*b) - 2*b^2 = 9*b^3 + 4*b^2 := by
  sorry

end simplify_expression_l921_92129


namespace f_derivative_at_zero_l921_92189

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.exp x

theorem f_derivative_at_zero : 
  (deriv f) 0 = 2 := by
  sorry

end f_derivative_at_zero_l921_92189


namespace number_of_girls_is_760_l921_92146

/-- Represents the number of students in a school survey --/
structure SchoolSurvey where
  total_students : ℕ
  sample_size : ℕ
  girls_sampled_difference : ℕ

/-- Calculates the number of girls in the school based on survey data --/
def number_of_girls (survey : SchoolSurvey) : ℕ :=
  survey.total_students / 2 - survey.girls_sampled_difference * (survey.total_students / survey.sample_size / 2)

/-- Theorem stating that given the survey conditions, the number of girls in the school is 760 --/
theorem number_of_girls_is_760 (survey : SchoolSurvey) 
    (h1 : survey.total_students = 1600)
    (h2 : survey.sample_size = 200)
    (h3 : survey.girls_sampled_difference = 10) : 
  number_of_girls survey = 760 := by
  sorry

#eval number_of_girls { total_students := 1600, sample_size := 200, girls_sampled_difference := 10 }

end number_of_girls_is_760_l921_92146


namespace bridgette_has_three_cats_l921_92171

/-- Represents the number of baths given to pets in a year -/
def total_baths : ℕ := 96

/-- Represents the number of dogs Bridgette has -/
def num_dogs : ℕ := 2

/-- Represents the number of birds Bridgette has -/
def num_birds : ℕ := 4

/-- Represents the number of baths given to a dog in a year -/
def dog_baths_per_year : ℕ := 24

/-- Represents the number of baths given to a bird in a year -/
def bird_baths_per_year : ℕ := 3

/-- Represents the number of baths given to a cat in a year -/
def cat_baths_per_year : ℕ := 12

/-- Theorem stating that Bridgette has 3 cats -/
theorem bridgette_has_three_cats :
  ∃ (num_cats : ℕ),
    num_cats * cat_baths_per_year = 
      total_baths - (num_dogs * dog_baths_per_year + num_birds * bird_baths_per_year) ∧
    num_cats = 3 :=
by sorry

end bridgette_has_three_cats_l921_92171


namespace range_of_x_when_a_is_one_range_of_a_for_necessary_not_sufficient_l921_92127

-- Define propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := (x - 3)*(2 - x) ≥ 0

-- Theorem 1
theorem range_of_x_when_a_is_one :
  ∀ x : ℝ, (p x 1 ∧ q x) ↔ (2 ≤ x ∧ x < 3) :=
sorry

-- Theorem 2
theorem range_of_a_for_necessary_not_sufficient :
  ∀ a : ℝ, (∀ x : ℝ, ¬(q x) → ¬(p x a)) ∧ (∃ x : ℝ, ¬(p x a) ∧ q x) ↔ (1 < a ∧ a < 2) :=
sorry

end range_of_x_when_a_is_one_range_of_a_for_necessary_not_sufficient_l921_92127


namespace tangent_point_on_parabola_l921_92165

theorem tangent_point_on_parabola : ∃ (x y : ℝ), 
  y = x^2 ∧ 
  (2 : ℝ) * x = Real.tan (π / 4) ∧ 
  x = 1 / 2 ∧ 
  y = 1 / 4 := by
  sorry

end tangent_point_on_parabola_l921_92165


namespace min_groups_and_people_is_16_l921_92176

/-- Represents the seating arrangement in a cafe -/
structure CafeSeating where
  tables : Nat
  counter_seats : Nat
  min_group_size : Nat
  max_group_size : Nat

/-- Represents the final seating state of the cafe -/
structure SeatingState where
  groups : Nat
  total_people : Nat

/-- The minimum possible value of groups + total people given the cafe seating conditions -/
def min_groups_and_people (cafe : CafeSeating) : Nat :=
  16

/-- Theorem stating that the minimum possible value of M + N is 16 -/
theorem min_groups_and_people_is_16 (cafe : CafeSeating) 
  (h1 : cafe.tables = 3)
  (h2 : cafe.counter_seats = 5)
  (h3 : cafe.min_group_size = 1)
  (h4 : cafe.max_group_size = 4)
  (state : SeatingState)
  (h5 : state.groups + state.total_people ≥ min_groups_and_people cafe) :
  min_groups_and_people cafe = 16 :=
by sorry

end min_groups_and_people_is_16_l921_92176


namespace missing_number_exists_l921_92192

theorem missing_number_exists : ∃ x : ℝ, (1 / ((1 / 0.03) + (1 / x))) = 0.02775 := by
  sorry

end missing_number_exists_l921_92192


namespace reciprocal_problem_l921_92164

theorem reciprocal_problem (x : ℝ) (h : 5 * x = 2) : 100 * (1 / x) = 250 := by
  sorry

end reciprocal_problem_l921_92164


namespace quadratic_root_difference_l921_92190

theorem quadratic_root_difference (p : ℝ) : 
  let a := 1
  let b := -(p + 1)
  let c := (p^2 + 2*p - 3) / 4
  let discriminant := b^2 - 4*a*c
  let larger_root := (-b + Real.sqrt discriminant) / (2*a)
  let smaller_root := (-b - Real.sqrt discriminant) / (2*a)
  larger_root - smaller_root = Real.sqrt (2*p + 1 - p^2) :=
by sorry

end quadratic_root_difference_l921_92190


namespace shelter_dogs_count_l921_92153

/-- Given an animal shelter with dogs and cats, prove the number of dogs. -/
theorem shelter_dogs_count (d c : ℕ) : 
  d * 7 = c * 15 →  -- Initial ratio of dogs to cats is 15:7
  d * 11 = (c + 16) * 15 →  -- Ratio after adding 16 cats is 15:11
  d = 60 :=  -- The number of dogs is 60
by sorry

end shelter_dogs_count_l921_92153


namespace kath_movie_cost_l921_92166

def movie_admission_cost (regular_price : ℚ) (discount_percent : ℚ) (before_6pm : Bool) (num_people : ℕ) : ℚ :=
  let discounted_price := if before_6pm then regular_price * (1 - discount_percent / 100) else regular_price
  discounted_price * num_people

theorem kath_movie_cost :
  let regular_price : ℚ := 8
  let discount_percent : ℚ := 25
  let before_6pm : Bool := true
  let num_people : ℕ := 6
  movie_admission_cost regular_price discount_percent before_6pm num_people = 36 := by
  sorry

end kath_movie_cost_l921_92166


namespace parabola_focus_directrix_distance_l921_92195

/-- Given a parabola with equation x^2 = 12y, the distance from its focus to its directrix is 6 -/
theorem parabola_focus_directrix_distance :
  ∀ (x y : ℝ), x^2 = 12*y →
  ∃ (focus_x focus_y directrix_y : ℝ),
    (focus_x = 0 ∧ focus_y = 3 ∧ directrix_y = -3) ∧
    (focus_y - directrix_y = 6) :=
by sorry

end parabola_focus_directrix_distance_l921_92195


namespace smallest_next_divisor_l921_92179

def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

theorem smallest_next_divisor (m : ℕ) (h1 : 1000 ≤ m ∧ m ≤ 9999) 
  (h2 : is_odd m) (h3 : m % 437 = 0) :
  ∃ d : ℕ, d > 437 ∧ m % d = 0 ∧ is_odd d ∧
  ∀ d' : ℕ, d' > 437 → m % d' = 0 → is_odd d' → d ≤ d' :=
sorry

end smallest_next_divisor_l921_92179


namespace square_root_fraction_simplification_l921_92128

theorem square_root_fraction_simplification :
  Real.sqrt (8^2 + 6^2) / Real.sqrt (25 + 16) = 10 / Real.sqrt 41 := by
  sorry

end square_root_fraction_simplification_l921_92128


namespace original_number_proof_l921_92159

theorem original_number_proof (x : ℝ) (h : 1 + 1/x = 5/2) : x = 2/3 := by
  sorry

end original_number_proof_l921_92159


namespace cube_three_times_cube_six_l921_92140

theorem cube_three_times_cube_six : 3^3 * 6^3 = 5832 := by
  sorry

end cube_three_times_cube_six_l921_92140


namespace apple_selection_probability_l921_92142

def total_apples : ℕ := 10
def red_apples : ℕ := 5
def green_apples : ℕ := 3
def yellow_apples : ℕ := 2
def selected_apples : ℕ := 3

theorem apple_selection_probability :
  (Nat.choose green_apples 2 * Nat.choose yellow_apples 1) / Nat.choose total_apples selected_apples = 1 / 20 := by
  sorry

end apple_selection_probability_l921_92142


namespace flag_arrangement_count_flag_arrangement_remainder_l921_92188

def M (b g : ℕ) : ℕ :=
  (b - 1) * Nat.choose (b + 2) g - 2 * Nat.choose (b + 1) g

theorem flag_arrangement_count :
  M 14 11 = 54054 :=
by sorry

theorem flag_arrangement_remainder :
  M 14 11 % 1000 = 54 :=
by sorry

end flag_arrangement_count_flag_arrangement_remainder_l921_92188


namespace square_sum_equals_fifteen_l921_92181

theorem square_sum_equals_fifteen (x y : ℝ) (h1 : x * y = 3) (h2 : (x - y)^2 = 9) :
  x^2 + y^2 = 15 := by
  sorry

end square_sum_equals_fifteen_l921_92181


namespace prob_same_team_is_one_third_l921_92155

/-- The number of teams -/
def num_teams : ℕ := 3

/-- The probability of two students choosing the same team -/
def prob_same_team : ℚ := 1 / 3

/-- Theorem: The probability of two students independently and randomly choosing the same team out of three teams is 1/3 -/
theorem prob_same_team_is_one_third :
  prob_same_team = 1 / 3 := by sorry

end prob_same_team_is_one_third_l921_92155


namespace common_chord_length_l921_92118

theorem common_chord_length (r : ℝ) (h : r = 15) :
  let chord_length := 2 * r * Real.sqrt 3
  chord_length = 15 * Real.sqrt 3 := by sorry

end common_chord_length_l921_92118


namespace num_factors_of_M_l921_92145

/-- The number of natural-number factors of M, where M = 2^5 · 3^2 · 7^3 · 11^1 -/
def num_factors (M : ℕ) : ℕ :=
  (5 + 1) * (2 + 1) * (3 + 1) * (1 + 1)

/-- M is defined as 2^5 · 3^2 · 7^3 · 11^1 -/
def M : ℕ := 2^5 * 3^2 * 7^3 * 11

theorem num_factors_of_M :
  num_factors M = 144 :=
sorry

end num_factors_of_M_l921_92145


namespace corys_initial_money_l921_92151

/-- The problem of determining Cory's initial amount of money -/
theorem corys_initial_money (cost_per_pack : ℝ) (additional_needed : ℝ) : 
  cost_per_pack = 49 → additional_needed = 78 → 
  2 * cost_per_pack - additional_needed = 20 := by
  sorry

end corys_initial_money_l921_92151


namespace amy_spent_32_pounds_l921_92177

/-- Represents the amount spent by Chloe in pounds -/
def chloe_spent : ℝ := 20

/-- Represents the amount spent by Becky as a fraction of Chloe's spending -/
def becky_spent_ratio : ℝ := 0.15

/-- Represents the amount spent by Amy as a fraction above Chloe's spending -/
def amy_spent_ratio : ℝ := 1.6

/-- The total amount spent by all three shoppers in pounds -/
def total_spent : ℝ := 55

theorem amy_spent_32_pounds :
  let becky_spent := becky_spent_ratio * chloe_spent
  let amy_spent := amy_spent_ratio * chloe_spent
  becky_spent + amy_spent + chloe_spent = total_spent ∧
  amy_spent = 32 := by sorry

end amy_spent_32_pounds_l921_92177


namespace right_triangle_area_l921_92122

/-- Right triangle XYZ with altitude foot W -/
structure RightTriangle where
  -- Point X
  X : ℝ × ℝ
  -- Point Y (right angle)
  Y : ℝ × ℝ
  -- Point Z
  Z : ℝ × ℝ
  -- Point W (foot of altitude from Y to XZ)
  W : ℝ × ℝ
  -- XW length
  xw_length : ℝ
  -- WZ length
  wz_length : ℝ
  -- Constraint: XYZ is a right triangle with right angle at Y
  right_angle_at_Y : (X.1 - Y.1) * (Z.1 - Y.1) + (X.2 - Y.2) * (Z.2 - Y.2) = 0
  -- Constraint: W is on XZ
  w_on_xz : ∃ t : ℝ, W = (t * X.1 + (1 - t) * Z.1, t * X.2 + (1 - t) * Z.2)
  -- Constraint: YW is perpendicular to XZ
  yw_perpendicular_xz : (Y.1 - W.1) * (X.1 - Z.1) + (Y.2 - W.2) * (X.2 - Z.2) = 0
  -- Constraint: XW length is 5
  xw_is_5 : xw_length = 5
  -- Constraint: WZ length is 7
  wz_is_7 : wz_length = 7

/-- The area of the right triangle XYZ -/
def triangleArea (t : RightTriangle) : ℝ := sorry

/-- Theorem: The area of the right triangle XYZ is 6√35 -/
theorem right_triangle_area (t : RightTriangle) : triangleArea t = 6 * Real.sqrt 35 := by
  sorry

end right_triangle_area_l921_92122


namespace heptagon_diagonals_l921_92167

/-- The number of diagonals in a polygon with n vertices -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon with one vertex removed is equivalent to a heptagon -/
def heptagon_vertices : ℕ := 8 - 1

theorem heptagon_diagonals : diagonals heptagon_vertices = 14 := by
  sorry

end heptagon_diagonals_l921_92167


namespace anands_income_is_2000_l921_92132

/-- Represents the financial data of a person --/
structure FinancialData where
  income : ℕ
  expenditure : ℕ
  savings : ℕ

/-- Proves that Anand's income is 2000 given the conditions --/
theorem anands_income_is_2000 
  (anand balu : FinancialData)
  (income_ratio : anand.income * 4 = balu.income * 5)
  (expenditure_ratio : anand.expenditure * 2 = balu.expenditure * 3)
  (anand_savings : anand.income - anand.expenditure = 800)
  (balu_savings : balu.income - balu.expenditure = 800) :
  anand.income = 2000 := by
  sorry

#check anands_income_is_2000

end anands_income_is_2000_l921_92132


namespace race_equation_l921_92149

theorem race_equation (x : ℝ) (h : x > 0) : 
  (1000 / x : ℝ) - (1000 / (1.25 * x)) = 30 :=
by sorry

end race_equation_l921_92149


namespace cube_volume_l921_92185

theorem cube_volume (s : ℝ) (h : s * s = 64) : s * s * s = 512 := by
  sorry

end cube_volume_l921_92185
