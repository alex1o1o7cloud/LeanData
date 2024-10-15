import Mathlib

namespace NUMINAMATH_CALUDE_inequality_solution_set_l3174_317455

theorem inequality_solution_set (x : ℝ) :
  (((x + 1) / (x - 2) + (x + 3) / (3 * x) ≥ 4) ↔ 
   (x ∈ Set.Ioc 0 (1/2) ∪ Set.Ioo (3/2) 2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3174_317455


namespace NUMINAMATH_CALUDE_squirrel_acorns_l3174_317452

theorem squirrel_acorns (total_acorns : ℕ) (num_months : ℕ) (acorns_per_month : ℕ) :
  total_acorns = 210 →
  num_months = 3 →
  acorns_per_month = 60 →
  total_acorns - num_months * acorns_per_month = 30 :=
by sorry

end NUMINAMATH_CALUDE_squirrel_acorns_l3174_317452


namespace NUMINAMATH_CALUDE_max_ratio_of_two_digit_integers_l3174_317465

theorem max_ratio_of_two_digit_integers (x y : ℕ) : 
  (10 ≤ x ∧ x ≤ 99) →  -- x is a two-digit positive integer
  (10 ≤ y ∧ y ≤ 99) →  -- y is a two-digit positive integer
  x > y →              -- x is greater than y
  (x + y) / 2 = 70 →   -- their mean is 70
  (∀ a b : ℕ, (10 ≤ a ∧ a ≤ 99) → (10 ≤ b ∧ b ≤ 99) → a > b → (a + b) / 2 = 70 → x / y ≥ a / b) →
  x / y = 99 / 41 :=
by sorry

end NUMINAMATH_CALUDE_max_ratio_of_two_digit_integers_l3174_317465


namespace NUMINAMATH_CALUDE_square_last_digits_l3174_317486

theorem square_last_digits (n : ℕ) :
  (n^2 % 10 % 2 = 1) → ((n^2 % 100) / 10 % 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_square_last_digits_l3174_317486


namespace NUMINAMATH_CALUDE_solve_for_a_l3174_317412

theorem solve_for_a (a x : ℝ) : 
  (3/10) * a + (2*x + 4)/2 = 4*(x - 1) ∧ x = 3 → a = 10 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l3174_317412


namespace NUMINAMATH_CALUDE_total_pencils_l3174_317419

/-- Given that each child has 2 pencils and there are 9 children, prove that the total number of pencils is 18. -/
theorem total_pencils (pencils_per_child : ℕ) (num_children : ℕ) (h1 : pencils_per_child = 2) (h2 : num_children = 9) :
  pencils_per_child * num_children = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_l3174_317419


namespace NUMINAMATH_CALUDE_cars_produced_in_europe_l3174_317403

def cars_north_america : ℕ := 3884
def total_cars : ℕ := 6755

theorem cars_produced_in_europe : 
  total_cars - cars_north_america = 2871 := by sorry

end NUMINAMATH_CALUDE_cars_produced_in_europe_l3174_317403


namespace NUMINAMATH_CALUDE_a_less_than_neg_one_sufficient_not_necessary_l3174_317446

theorem a_less_than_neg_one_sufficient_not_necessary (a : ℝ) :
  (∀ x : ℝ, x < -1 → x + 1/x < -2) ∧
  (∃ y : ℝ, y ≥ -1 ∧ y + 1/y < -2) :=
sorry

end NUMINAMATH_CALUDE_a_less_than_neg_one_sufficient_not_necessary_l3174_317446


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l3174_317408

theorem smallest_n_for_inequality : 
  (∃ n : ℕ, ∀ x y z : ℝ, (x^2 + y^2 + z^2)^2 ≤ n * (x^4 + y^4 + z^4)) ∧ 
  (∀ x y z : ℝ, (x^2 + y^2 + z^2)^2 ≤ 3 * (x^4 + y^4 + z^4)) ∧
  (∀ m : ℕ, m < 3 → ∃ x y z : ℝ, (x^2 + y^2 + z^2)^2 > m * (x^4 + y^4 + z^4)) :=
by sorry

#check smallest_n_for_inequality

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l3174_317408


namespace NUMINAMATH_CALUDE_existence_of_special_polynomial_l3174_317499

theorem existence_of_special_polynomial :
  ∃ (P : Polynomial ℝ), 
    (∃ (i : ℕ), (P.coeff i < 0)) ∧ 
    (∀ (n : ℕ), n > 1 → ∀ (j : ℕ), ((P^n).coeff j > 0)) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_polynomial_l3174_317499


namespace NUMINAMATH_CALUDE_normal_dist_prob_l3174_317400

-- Define a random variable following normal distribution
def normal_dist (μ σ : ℝ) : Type := ℝ

-- Define the probability function
noncomputable def P (X : normal_dist 1 σ) (event : Set ℝ) : ℝ := sorry

-- State the theorem
theorem normal_dist_prob (σ : ℝ) (ξ : normal_dist 1 σ) 
  (h : P ξ {x | x < 0} = 0.4) : 
  P ξ {x | x < 2} = 0.6 := by sorry

end NUMINAMATH_CALUDE_normal_dist_prob_l3174_317400


namespace NUMINAMATH_CALUDE_problem_solution_l3174_317436

theorem problem_solution (A B : ℝ) : 
  (A^2 = 0.012345678987654321 * (List.sum (List.range 9) + List.sum (List.reverse (List.range 9)))) →
  (B^2 = 0.012345679012345679) →
  9 * (10^9 : ℝ) * (1 - |A|) * B = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3174_317436


namespace NUMINAMATH_CALUDE_quadratic_function_property_l3174_317422

def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_property
  (f : ℝ → ℝ)
  (h_quad : is_quadratic f)
  (h_pos : ∃ a b c : ℝ, a > 0 ∧ ∀ x, f x = a * x^2 + b * x + c)
  (h_sym : ∀ x : ℝ, f x = f (4 - x))
  (h_ineq : ∀ a : ℝ, f (2 - a^2) < f (1 + a - a^2)) :
  ∀ a : ℝ, a < 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l3174_317422


namespace NUMINAMATH_CALUDE_inequality_proof_l3174_317498

theorem inequality_proof (x : ℝ) (h : 3 * x + 4 ≠ 0) :
  3 - 1 / (3 * x + 4) < 5 ↔ x < -4/3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3174_317498


namespace NUMINAMATH_CALUDE_brick_height_is_6cm_l3174_317407

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℝ := d.length * d.width * d.height

/-- The dimensions of the wall in centimeters -/
def wallDimensions : Dimensions :=
  { length := 800, width := 22.5, height := 600 }

/-- The known dimensions of a brick in centimeters (height is unknown) -/
def brickDimensions (h : ℝ) : Dimensions :=
  { length := 50, width := 11.25, height := h }

/-- The number of bricks needed to build the wall -/
def numberOfBricks : ℕ := 3200

/-- Theorem stating that the height of each brick is 6 cm -/
theorem brick_height_is_6cm :
  ∃ (h : ℝ), h = 6 ∧
    (volume wallDimensions = ↑numberOfBricks * volume (brickDimensions h)) := by
  sorry

end NUMINAMATH_CALUDE_brick_height_is_6cm_l3174_317407


namespace NUMINAMATH_CALUDE_max_x_plus_y_max_x_plus_y_achieved_l3174_317469

theorem max_x_plus_y (x y : ℝ) (h : 3 * (x^2 + y^2) = x - y) :
  x + y ≤ 1 / Real.sqrt 2 :=
by sorry

theorem max_x_plus_y_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ x y : ℝ, 3 * (x^2 + y^2) = x - y ∧ x + y > 1 / Real.sqrt 2 - ε :=
by sorry

end NUMINAMATH_CALUDE_max_x_plus_y_max_x_plus_y_achieved_l3174_317469


namespace NUMINAMATH_CALUDE_debby_total_texts_l3174_317432

def texts_before_noon : ℕ := 21
def initial_texts_after_noon : ℕ := 2
def hours_after_noon : ℕ := 12

def texts_after_noon (n : ℕ) : ℕ := initial_texts_after_noon * 2^n

def total_texts : ℕ := texts_before_noon + (Finset.sum (Finset.range hours_after_noon) texts_after_noon)

theorem debby_total_texts : total_texts = 8211 := by sorry

end NUMINAMATH_CALUDE_debby_total_texts_l3174_317432


namespace NUMINAMATH_CALUDE_B_power_97_l3174_317471

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![0, 1, 0],
    ![0, 0, 1],
    ![1, 0, 0]]

theorem B_power_97 : B^97 = B := by sorry

end NUMINAMATH_CALUDE_B_power_97_l3174_317471


namespace NUMINAMATH_CALUDE_second_point_y_coordinate_l3174_317405

/-- Given two points on a line, prove the y-coordinate of the second point -/
theorem second_point_y_coordinate
  (m n k : ℝ)
  (h1 : m = 2 * n + 3)  -- First point (m, n) satisfies line equation
  (h2 : m + 2 = 2 * (n + k) + 3)  -- Second point (m + 2, n + k) satisfies line equation
  (h3 : k = 1)  -- Given condition
  : n + k = n + 1 := by
  sorry

end NUMINAMATH_CALUDE_second_point_y_coordinate_l3174_317405


namespace NUMINAMATH_CALUDE_adolfo_tower_blocks_l3174_317440

-- Define the variables
def initial_blocks : ℕ := sorry
def added_blocks : ℝ := 65.0
def total_blocks : ℕ := 100

-- State the theorem
theorem adolfo_tower_blocks : initial_blocks = 35 := by
  sorry

end NUMINAMATH_CALUDE_adolfo_tower_blocks_l3174_317440


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3174_317489

theorem quadratic_equation_roots (a : ℤ) : 
  (∃ x y : ℤ, x ≠ y ∧ x < 0 ∧ y < 0 ∧ 
   (a^2 - 1) * x^2 - 2*(5*a + 1)*x + 24 = 0 ∧
   (a^2 - 1) * y^2 - 2*(5*a + 1)*y + 24 = 0) → 
  a = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3174_317489


namespace NUMINAMATH_CALUDE_gcd_problem_l3174_317430

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 17 * (2 * k + 1)) :
  Int.gcd (3 * b^2 + 65 * b + 143) (5 * b + 22) = 33 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l3174_317430


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3174_317470

theorem simplify_and_evaluate : 
  ∀ x : ℝ, x ≠ 2 → x ≠ -2 → x ≠ 0 → x = 1 →
  (3 * x / (x - 2) - x / (x + 2)) * (x^2 - 4) / x = 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3174_317470


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3174_317490

/-- A quadratic function f(x) with specific properties -/
def f (a c : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * x + c

/-- The theorem statement -/
theorem quadratic_function_properties (a c : ℝ) :
  (∀ x : ℝ, x ≠ 1/a → f a c x > 0) →
  (∃ min_a min_c : ℝ, 
    (∀ a' c' : ℝ, f a' c' 2 ≥ f min_a min_c 2) ∧
    f min_a min_c 2 = 0 ∧
    min_a = 1/2 ∧ min_c = 2) ∧
  (∀ m : ℝ, (∀ x : ℝ, x > 2 → f (1/2) 2 x + 4 ≥ m * (x - 2)) → m ≤ 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3174_317490


namespace NUMINAMATH_CALUDE_track_circumference_l3174_317413

/-- The circumference of a circular track given specific conditions -/
theorem track_circumference : 
  ∀ (circumference : ℝ) (distance_B_first_meet : ℝ) (distance_A_second_meet : ℝ),
  distance_B_first_meet = 100 →
  distance_A_second_meet = circumference - 60 →
  (circumference / 2 - distance_B_first_meet) / distance_B_first_meet = 
    distance_A_second_meet / (circumference + 60) →
  circumference = 480 := by
sorry

end NUMINAMATH_CALUDE_track_circumference_l3174_317413


namespace NUMINAMATH_CALUDE_andy_gave_five_to_brother_l3174_317473

/-- The number of cookies Andy had at the start -/
def initial_cookies : ℕ := 72

/-- The number of cookies Andy ate -/
def andy_ate : ℕ := 3

/-- The number of players in Andy's basketball team -/
def team_size : ℕ := 8

/-- The number of cookies taken by the i-th player -/
def player_cookies (i : ℕ) : ℕ := 2 * i - 1

/-- The sum of cookies taken by all team members -/
def team_total : ℕ := (team_size * (player_cookies 1 + player_cookies team_size)) / 2

/-- The number of cookies Andy gave to his little brother -/
def brother_cookies : ℕ := initial_cookies - andy_ate - team_total

theorem andy_gave_five_to_brother : brother_cookies = 5 := by
  sorry

end NUMINAMATH_CALUDE_andy_gave_five_to_brother_l3174_317473


namespace NUMINAMATH_CALUDE_video_votes_l3174_317423

theorem video_votes (net_score : ℚ) (like_percentage : ℚ) (dislike_percentage : ℚ) :
  net_score = 75 →
  like_percentage = 55 / 100 →
  dislike_percentage = 45 / 100 →
  like_percentage + dislike_percentage = 1 →
  ∃ (total_votes : ℚ),
    total_votes * (like_percentage - dislike_percentage) = net_score ∧
    total_votes = 750 :=
by sorry

end NUMINAMATH_CALUDE_video_votes_l3174_317423


namespace NUMINAMATH_CALUDE_angle_equality_l3174_317442

-- Define the problem statement
theorem angle_equality (θ : Real) (h1 : 0 < θ ∧ θ < π / 2) 
  (h2 : Real.sqrt 3 * Real.sin (5 * π / 180) = Real.cos θ - Real.sin θ) : 
  θ = 40 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_angle_equality_l3174_317442


namespace NUMINAMATH_CALUDE_sums_are_equal_l3174_317443

def S₁ : ℕ := 1 + 22 + 333 + 4444 + 55555 + 666666 + 7777777 + 88888888 + 999999999

def S₂ : ℕ := 9 + 98 + 987 + 9876 + 98765 + 987654 + 9876543 + 98765432 + 987654321

theorem sums_are_equal : S₁ = S₂ := by
  sorry

end NUMINAMATH_CALUDE_sums_are_equal_l3174_317443


namespace NUMINAMATH_CALUDE_school_population_l3174_317480

theorem school_population (total_students : ℕ) : 
  (5 : ℚ)/8 * total_students + (3 : ℚ)/8 * total_students = total_students →  -- Girls + Boys = Total
  ((3 : ℚ)/10 * (5 : ℚ)/8 * total_students : ℚ) + 
  ((3 : ℚ)/5 * (3 : ℚ)/8 * total_students : ℚ) = 330 →                       -- Middle schoolers
  total_students = 800 := by
sorry

end NUMINAMATH_CALUDE_school_population_l3174_317480


namespace NUMINAMATH_CALUDE_three_fifths_of_ten_x_minus_three_l3174_317433

theorem three_fifths_of_ten_x_minus_three (x : ℝ) : 
  (3 / 5) * (10 * x - 3) = 6 * x - 9 / 5 := by
  sorry

end NUMINAMATH_CALUDE_three_fifths_of_ten_x_minus_three_l3174_317433


namespace NUMINAMATH_CALUDE_f_two_zeros_implies_k_nonneg_l3174_317491

noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then x / (x - 2) + k * x^2 else Real.log x

theorem f_two_zeros_implies_k_nonneg (k : ℝ) :
  (∃ x y, x ≠ y ∧ f k x = 0 ∧ f k y = 0 ∧ ∀ z, f k z = 0 → z = x ∨ z = y) →
  k ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_f_two_zeros_implies_k_nonneg_l3174_317491


namespace NUMINAMATH_CALUDE_city_B_sand_amount_l3174_317492

def total_sand : ℝ := 95
def city_A_sand : ℝ := 16.5
def city_C_sand : ℝ := 24.5
def city_D_sand : ℝ := 28

theorem city_B_sand_amount : 
  total_sand - city_A_sand - city_C_sand - city_D_sand = 26 := by
  sorry

end NUMINAMATH_CALUDE_city_B_sand_amount_l3174_317492


namespace NUMINAMATH_CALUDE_brother_money_distribution_l3174_317464

theorem brother_money_distribution (older_initial younger_initial difference transfer : ℕ) :
  older_initial = 2800 →
  younger_initial = 1500 →
  difference = 360 →
  transfer = 470 →
  (older_initial - transfer) = (younger_initial + transfer + difference) :=
by
  sorry

end NUMINAMATH_CALUDE_brother_money_distribution_l3174_317464


namespace NUMINAMATH_CALUDE_employed_females_percentage_l3174_317466

theorem employed_females_percentage (total_population : ℝ) 
  (h1 : total_population > 0) 
  (employed_percentage : ℝ) 
  (h2 : employed_percentage = 60) 
  (employed_males_percentage : ℝ) 
  (h3 : employed_males_percentage = 45) : 
  (employed_percentage - employed_males_percentage) / employed_percentage * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_employed_females_percentage_l3174_317466


namespace NUMINAMATH_CALUDE_tan_sum_alpha_beta_l3174_317457

-- Define the line l
def line_l (x y : ℝ) (α β : ℝ) : Prop :=
  x * Real.tan α - y - 3 * Real.tan β = 0

-- Define the normal vector
def normal_vector : ℝ × ℝ := (2, -1)

-- Theorem statement
theorem tan_sum_alpha_beta (α β : ℝ) :
  line_l 0 1 α β ∧ 
  normal_vector = (2, -1) →
  Real.tan (α + β) = 1 :=
by sorry

end NUMINAMATH_CALUDE_tan_sum_alpha_beta_l3174_317457


namespace NUMINAMATH_CALUDE_elenas_earnings_l3174_317437

/-- Calculates the total earnings given an hourly wage and number of hours worked -/
def totalEarnings (hourlyWage : ℚ) (hoursWorked : ℚ) : ℚ :=
  hourlyWage * hoursWorked

/-- Proves that Elena's earnings for 4 hours at $13.25 per hour is $53.00 -/
theorem elenas_earnings :
  totalEarnings (13.25 : ℚ) (4 : ℚ) = (53 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_elenas_earnings_l3174_317437


namespace NUMINAMATH_CALUDE_minimize_reciprocal_sum_l3174_317424

theorem minimize_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 4 * a + b = 30) :
  (1 / a + 1 / b) ≥ 1 / 5 + 1 / 20 ∧
  (1 / a + 1 / b = 1 / 5 + 1 / 20 ↔ a = 5 ∧ b = 20) :=
by sorry

end NUMINAMATH_CALUDE_minimize_reciprocal_sum_l3174_317424


namespace NUMINAMATH_CALUDE_percentage_in_70_79_is_one_third_l3174_317479

/-- Represents the frequency distribution of test scores -/
def score_distribution : List (String × ℕ) :=
  [("90% - 100%", 3),
   ("80% - 89%", 5),
   ("70% - 79%", 8),
   ("60% - 69%", 4),
   ("50% - 59%", 1),
   ("Below 50%", 3)]

/-- Total number of students in the class -/
def total_students : ℕ := (score_distribution.map (λ x => x.2)).sum

/-- Number of students who scored in the 70%-79% range -/
def students_in_70_79 : ℕ := 
  (score_distribution.filter (λ x => x.1 = "70% - 79%")).map (λ x => x.2) |>.sum

/-- Theorem stating that the percentage of students who scored in the 70%-79% range is 1/3 of the class -/
theorem percentage_in_70_79_is_one_third :
  (students_in_70_79 : ℚ) / (total_students : ℚ) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_percentage_in_70_79_is_one_third_l3174_317479


namespace NUMINAMATH_CALUDE_julia_rental_cost_l3174_317487

/-- Calculates the total cost of a car rental --/
def calculateRentalCost (dailyRate : ℝ) (mileageRate : ℝ) (days : ℝ) (miles : ℝ) : ℝ :=
  dailyRate * days + mileageRate * miles

/-- Proves that Julia's car rental cost is $46.12 --/
theorem julia_rental_cost :
  let dailyRate : ℝ := 29
  let mileageRate : ℝ := 0.08
  let days : ℝ := 1
  let miles : ℝ := 214
  calculateRentalCost dailyRate mileageRate days miles = 46.12 := by
  sorry

end NUMINAMATH_CALUDE_julia_rental_cost_l3174_317487


namespace NUMINAMATH_CALUDE_satisfy_equation_l3174_317441

theorem satisfy_equation : ∀ (x y : ℝ), x = 1 ∧ y = 2 → 2 * x + 3 * y = 8 := by
  sorry

end NUMINAMATH_CALUDE_satisfy_equation_l3174_317441


namespace NUMINAMATH_CALUDE_intersection_with_complement_l3174_317458

open Set

def U : Finset ℕ := {0, 1, 2, 3, 4, 5}
def A : Finset ℕ := {0, 1, 3}
def B : Finset ℕ := {2, 3, 5}

theorem intersection_with_complement :
  A ∩ (U \ B) = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l3174_317458


namespace NUMINAMATH_CALUDE_min_distances_2019_points_l3174_317461

/-- The minimum number of distinct distances between pairs of points in a set of n points in a plane -/
noncomputable def min_distinct_distances (n : ℕ) : ℝ :=
  Real.sqrt (n - 3/4 : ℝ) - 1/2

/-- Theorem: For 2019 distinct points in a plane, the number of distinct distances between pairs of points is at least 44 -/
theorem min_distances_2019_points :
  ⌈min_distinct_distances 2019⌉ ≥ 44 := by sorry

end NUMINAMATH_CALUDE_min_distances_2019_points_l3174_317461


namespace NUMINAMATH_CALUDE_box_surface_area_l3174_317426

/-- Calculates the surface area of the interior of an open box formed by removing square corners from a rectangular sheet and folding up the sides. -/
def interior_surface_area (sheet_length : ℕ) (sheet_width : ℕ) (corner_size : ℕ) : ℕ :=
  let modified_area := sheet_length * sheet_width
  let corner_area := corner_size * corner_size
  let total_removed_area := 4 * corner_area
  modified_area - total_removed_area

/-- Theorem stating that the surface area of the interior of the box is 804 square units. -/
theorem box_surface_area :
  interior_surface_area 25 40 7 = 804 := by
  sorry

#eval interior_surface_area 25 40 7

end NUMINAMATH_CALUDE_box_surface_area_l3174_317426


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcd_four_exists_148_with_gcd_four_less_than_150_max_integer_with_gcd_four_l3174_317450

theorem greatest_integer_with_gcd_four (n : ℕ) : n < 150 ∧ Nat.gcd n 12 = 4 → n ≤ 148 :=
by sorry

theorem exists_148_with_gcd_four : Nat.gcd 148 12 = 4 :=
by sorry

theorem less_than_150 : 148 < 150 :=
by sorry

theorem max_integer_with_gcd_four :
  ∀ m : ℕ, m < 150 ∧ Nat.gcd m 12 = 4 → m ≤ 148 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcd_four_exists_148_with_gcd_four_less_than_150_max_integer_with_gcd_four_l3174_317450


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3174_317411

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℝ, (X^4 - 3*X^2 + 2 : Polynomial ℝ) = (X^2 - 3) * q + 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3174_317411


namespace NUMINAMATH_CALUDE_blueberry_count_l3174_317414

theorem blueberry_count (total : ℕ) (raspberries : ℕ) (blackberries : ℕ) (blueberries : ℕ)
  (h1 : total = 42)
  (h2 : raspberries = total / 2)
  (h3 : blackberries = total / 3)
  (h4 : total = raspberries + blackberries + blueberries) :
  blueberries = 7 := by
  sorry

end NUMINAMATH_CALUDE_blueberry_count_l3174_317414


namespace NUMINAMATH_CALUDE_area_of_EFGH_l3174_317404

-- Define the rectangle and squares
structure Rectangle :=
  (width : ℝ)
  (height : ℝ)

structure Square :=
  (side : ℝ)

-- Define the problem setup
def smallest_square : Square :=
  { side := 1 }

def rectangle_EFGH : Rectangle :=
  { width := 4, height := 3 }

-- Define the theorem
theorem area_of_EFGH :
  (rectangle_EFGH.width * rectangle_EFGH.height : ℝ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_area_of_EFGH_l3174_317404


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l3174_317460

theorem coefficient_x_squared_in_expansion : 
  let expansion := (X - 2 / X) ^ 4
  ∃ a b c d e : ℤ, 
    expansion = a * X^4 + b * X^3 + c * X^2 + d * X + e * X^0 ∧ 
    c = -8
  := by sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l3174_317460


namespace NUMINAMATH_CALUDE_matrix_power_four_l3174_317481

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]

theorem matrix_power_four :
  A ^ 4 = !![0, -9; 9, -9] := by sorry

end NUMINAMATH_CALUDE_matrix_power_four_l3174_317481


namespace NUMINAMATH_CALUDE_largest_of_three_consecutive_odds_l3174_317495

theorem largest_of_three_consecutive_odds (a b c : ℤ) : 
  (a + b + c = 75) →  -- sum is 75
  (c - a = 4) →       -- difference between largest and smallest is 4
  (Odd a ∧ Odd b ∧ Odd c) →  -- all numbers are odd
  (b = a + 2 ∧ c = b + 2) →  -- numbers are consecutive
  (c = 27) :=         -- largest number is 27
by sorry

end NUMINAMATH_CALUDE_largest_of_three_consecutive_odds_l3174_317495


namespace NUMINAMATH_CALUDE_cube_has_twelve_edges_l3174_317476

/-- A cube is a three-dimensional solid object with six square faces. -/
structure Cube where
  -- We don't need to specify any fields for this simple definition

/-- The number of edges in a cube. -/
def num_edges (c : Cube) : ℕ := 12

/-- Theorem: A cube has 12 edges. -/
theorem cube_has_twelve_edges (c : Cube) : num_edges c = 12 := by
  sorry

end NUMINAMATH_CALUDE_cube_has_twelve_edges_l3174_317476


namespace NUMINAMATH_CALUDE_sin_theta_plus_7pi_6_l3174_317416

theorem sin_theta_plus_7pi_6 (θ : ℝ) 
  (h : Real.cos (θ - π/6) + Real.sin θ = 4 * Real.sqrt 3 / 5) : 
  Real.sin (θ + 7*π/6) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_theta_plus_7pi_6_l3174_317416


namespace NUMINAMATH_CALUDE_sum_has_five_digits_l3174_317428

/-- A nonzero digit is a natural number between 1 and 9, inclusive. -/
def NonzeroDigit : Type := {n : ℕ // 1 ≤ n ∧ n ≤ 9}

/-- The function that constructs the second number (A75) from a nonzero digit A. -/
def secondNumber (A : NonzeroDigit) : ℕ := A.val * 100 + 75

/-- The function that constructs the third number (5B2) from a nonzero digit B. -/
def thirdNumber (B : NonzeroDigit) : ℕ := 500 + B.val * 10 + 2

/-- The theorem stating that the sum of the three numbers always has 5 digits. -/
theorem sum_has_five_digits (A B : NonzeroDigit) :
  ∃ n : ℕ, 10000 ≤ 9643 + secondNumber A + thirdNumber B ∧
           9643 + secondNumber A + thirdNumber B < 100000 := by
  sorry

end NUMINAMATH_CALUDE_sum_has_five_digits_l3174_317428


namespace NUMINAMATH_CALUDE_solve_equation_l3174_317410

theorem solve_equation : ∃ x : ℚ, 25 - (3 * 5) = (2 * x) + 1 ∧ x = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3174_317410


namespace NUMINAMATH_CALUDE_stuarts_initial_marbles_l3174_317484

/-- Stuart's initial marble count problem -/
theorem stuarts_initial_marbles (betty_marbles : ℕ) (stuart_final : ℕ) 
  (h1 : betty_marbles = 60)
  (h2 : stuart_final = 80) :
  ∃ (stuart_initial : ℕ), 
    stuart_initial + (betty_marbles * 2/5 : ℕ) = stuart_final ∧ 
    stuart_initial = 56 := by
  sorry

end NUMINAMATH_CALUDE_stuarts_initial_marbles_l3174_317484


namespace NUMINAMATH_CALUDE_baby_panda_eats_50_pounds_l3174_317406

/-- The amount of bamboo (in pounds) an adult panda eats per day -/
def adult_panda_daily : ℕ := 138

/-- The total amount of bamboo (in pounds) eaten by both adult and baby pandas in a week -/
def total_weekly : ℕ := 1316

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The amount of bamboo (in pounds) a baby panda eats per day -/
def baby_panda_daily : ℕ := (total_weekly - adult_panda_daily * days_per_week) / days_per_week

theorem baby_panda_eats_50_pounds : baby_panda_daily = 50 := by
  sorry

end NUMINAMATH_CALUDE_baby_panda_eats_50_pounds_l3174_317406


namespace NUMINAMATH_CALUDE_border_area_l3174_317472

/-- Given a rectangular photograph with a frame, calculate the area of the border. -/
theorem border_area (photo_height photo_width border_width : ℝ) 
  (h1 : photo_height = 8)
  (h2 : photo_width = 10)
  (h3 : border_width = 2) : 
  (photo_height + 2 * border_width) * (photo_width + 2 * border_width) - 
  photo_height * photo_width = 88 := by
  sorry

#check border_area

end NUMINAMATH_CALUDE_border_area_l3174_317472


namespace NUMINAMATH_CALUDE_unique_sums_count_l3174_317453

/-- Represents the set of available coins -/
def CoinSet : Finset ℕ := {1, 2, 5, 100, 100, 100, 100, 500, 500}

/-- Generates all possible sums using the given coin set -/
def PossibleSums (coins : Finset ℕ) : Finset ℕ :=
  sorry

/-- The number of unique sums that can be formed using the given coin set -/
theorem unique_sums_count : (PossibleSums CoinSet).card = 119 := by
  sorry

end NUMINAMATH_CALUDE_unique_sums_count_l3174_317453


namespace NUMINAMATH_CALUDE_range_of_alpha_plus_three_beta_l3174_317445

theorem range_of_alpha_plus_three_beta 
  (h1 : ∀ α β : ℝ, -1 ≤ α + β ∧ α + β ≤ 1 → 1 ≤ α + 2*β ∧ α + 2*β ≤ 3) :
  ∀ α β : ℝ, (-1 ≤ α + β ∧ α + β ≤ 1) → (1 ≤ α + 2*β ∧ α + 2*β ≤ 3) → 
  (1 ≤ α + 3*β ∧ α + 3*β ≤ 7) := by
sorry

end NUMINAMATH_CALUDE_range_of_alpha_plus_three_beta_l3174_317445


namespace NUMINAMATH_CALUDE_field_trip_theorem_l3174_317477

/-- Represents the number of students participating in the field trip -/
def num_students : ℕ := 245

/-- Represents the number of 35-seat buses needed to exactly fit all students -/
def num_35_seat_buses : ℕ := 7

/-- Represents the number of 45-seat buses needed to fit all students with one less bus -/
def num_45_seat_buses : ℕ := 6

/-- Represents the rental fee for a 35-seat bus in yuan -/
def fee_35_seat : ℕ := 320

/-- Represents the rental fee for a 45-seat bus in yuan -/
def fee_45_seat : ℕ := 380

/-- Represents the total number of buses to be rented -/
def total_buses : ℕ := 6

/-- Theorem stating the number of students and the most cost-effective rental plan -/
theorem field_trip_theorem : 
  (num_students = 35 * num_35_seat_buses) ∧ 
  (num_students = 45 * (num_45_seat_buses - 1) - 25) ∧
  (∀ a b : ℕ, a + b = total_buses → 
    35 * a + 45 * b ≥ num_students →
    fee_35_seat * a + fee_45_seat * b ≥ fee_35_seat * 2 + fee_45_seat * 4) :=
by sorry

end NUMINAMATH_CALUDE_field_trip_theorem_l3174_317477


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3174_317493

/-- Given that in the expansion of (2x + a/x^2)^5, the coefficient of x^(-4) is 320, prove that a = 2 -/
theorem binomial_expansion_coefficient (a : ℝ) : 
  (∃ (c : ℝ), c = (Nat.choose 5 3) * 2^2 * a^3 ∧ c = 320) → a = 2 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3174_317493


namespace NUMINAMATH_CALUDE_combined_sum_equals_3751_l3174_317431

/-- The first element of the nth set in the pattern -/
def first_element (n : ℕ) : ℕ := 1 + n * (n - 1) / 2

/-- The last element of the nth set in the pattern -/
def last_element (n : ℕ) : ℕ := first_element n + n - 1

/-- The sum of elements in the nth set -/
def set_sum (n : ℕ) : ℕ := n * (first_element n + last_element n) / 2

/-- The combined sum of elements in the 15th and 16th sets -/
def combined_sum : ℕ := set_sum 15 + set_sum 16

theorem combined_sum_equals_3751 : combined_sum = 3751 := by
  sorry

end NUMINAMATH_CALUDE_combined_sum_equals_3751_l3174_317431


namespace NUMINAMATH_CALUDE_diana_apollo_dice_probability_l3174_317417

def roll_die := Finset.range 6

def favorable_outcomes : Finset (ℕ × ℕ) :=
  Finset.filter (fun p => p.1 >= p.2) (roll_die.product roll_die)

theorem diana_apollo_dice_probability :
  (favorable_outcomes.card : ℚ) / (roll_die.card * roll_die.card) = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_diana_apollo_dice_probability_l3174_317417


namespace NUMINAMATH_CALUDE_jessica_money_difference_l3174_317459

/-- Proves that Jessica has 90 dollars more than Rodney given the stated conditions. -/
theorem jessica_money_difference (jessica_money : ℕ) (lily_money : ℕ) (ian_money : ℕ) (rodney_money : ℕ) :
  jessica_money = 150 ∧
  jessica_money = lily_money + 30 ∧
  lily_money = 3 * ian_money ∧
  ian_money + 20 = rodney_money →
  jessica_money - rodney_money = 90 :=
by sorry

end NUMINAMATH_CALUDE_jessica_money_difference_l3174_317459


namespace NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l3174_317425

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem sixth_term_of_geometric_sequence (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 4)^2 - 8*(a 4) + 9 = 0 →
  (a 8)^2 - 8*(a 8) + 9 = 0 →
  a 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l3174_317425


namespace NUMINAMATH_CALUDE_perimeter_APR_is_50_l3174_317497

/-- A circle with two tangents from an exterior point A touching at B and C,
    and a third tangent touching at Q and intersecting AB at P and AC at R. -/
structure TangentCircle where
  /-- The length of tangent AB -/
  AB : ℝ
  /-- The distance from A to Q along the tangent -/
  AQ : ℝ

/-- The perimeter of triangle APR in the TangentCircle configuration -/
def perimeterAPR (tc : TangentCircle) : ℝ :=
  tc.AB - tc.AQ + tc.AQ + tc.AQ

/-- Theorem stating that for a TangentCircle with AB = 25 and AQ = 12.5,
    the perimeter of triangle APR is 50 -/
theorem perimeter_APR_is_50 (tc : TangentCircle)
    (h1 : tc.AB = 25) (h2 : tc.AQ = 12.5) :
    perimeterAPR tc = 50 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_APR_is_50_l3174_317497


namespace NUMINAMATH_CALUDE_line_OF_equation_l3174_317447

/-- Given a triangle ABC with vertices A(0,a), B(b,0), C(c,0), and a point P(0,p) on line segment AO
    (not an endpoint), where a, b, c, and p are non-zero real numbers, prove that the equation of
    line OF is (1/c - 1/b)x + (1/p - 1/a)y = 0, where F is the intersection of lines CP and AB. -/
theorem line_OF_equation (a b c p : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hp : p ≠ 0)
    (hp_between : 0 < p ∧ p < a) : 
    ∃ (x y : ℝ), (1 / c - 1 / b) * x + (1 / p - 1 / a) * y = 0 ↔ 
    (∃ (t : ℝ), x = t * c ∧ y = t * p) ∧ (∃ (s : ℝ), x = s * b ∧ y = s * a) := by
  sorry

end NUMINAMATH_CALUDE_line_OF_equation_l3174_317447


namespace NUMINAMATH_CALUDE_direct_proportion_quadrants_l3174_317401

/-- A direct proportion function in a plane rectangular coordinate system -/
structure DirectProportionFunction where
  n : ℝ
  f : ℝ → ℝ
  h : ∀ x, f x = (n - 1) * x

/-- Predicate to check if a point (x, y) is in the first or third quadrant -/
def isInFirstOrThirdQuadrant (x y : ℝ) : Prop :=
  (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0)

/-- Predicate to check if the graph of a function passes through the first and third quadrants -/
def passesFirstAndThirdQuadrants (f : ℝ → ℝ) : Prop :=
  ∀ x, isInFirstOrThirdQuadrant x (f x)

/-- Theorem: If a direct proportion function's graph passes through the first and third quadrants,
    then n > 1 -/
theorem direct_proportion_quadrants (dpf : DirectProportionFunction)
    (h : passesFirstAndThirdQuadrants dpf.f) : dpf.n > 1 := by
  sorry

end NUMINAMATH_CALUDE_direct_proportion_quadrants_l3174_317401


namespace NUMINAMATH_CALUDE_inequality_range_l3174_317409

theorem inequality_range (a : ℝ) : 
  (∀ x ∈ Set.Ioo 0 (1/2), x^2 - Real.log x / Real.log a < 0) ↔ a ∈ Set.Ioo (1/16) 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l3174_317409


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3174_317485

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Theorem: For a geometric sequence, if a_4 * a_6 = 10, then a_2 * a_8 = 10 -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h : GeometricSequence a) 
    (h_prod : a 4 * a 6 = 10) : a 2 * a 8 = 10 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3174_317485


namespace NUMINAMATH_CALUDE_gcd_15893_35542_l3174_317454

theorem gcd_15893_35542 : Nat.gcd 15893 35542 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_15893_35542_l3174_317454


namespace NUMINAMATH_CALUDE_largest_number_with_sum_17_l3174_317420

/-- The largest number with all different digits whose sum is 17 -/
def largest_number : ℕ := 763210

/-- Function to get the digits of a natural number -/
def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
  aux n []

/-- Theorem stating that 763210 is the largest number with all different digits whose sum is 17 -/
theorem largest_number_with_sum_17 :
  (∀ n : ℕ, n ≤ largest_number ∨
    (digits n).sum ≠ 17 ∨
    (digits n).length ≠ (digits n).toFinset.card) ∧
  (digits largest_number).sum = 17 ∧
  (digits largest_number).length = (digits largest_number).toFinset.card :=
sorry

end NUMINAMATH_CALUDE_largest_number_with_sum_17_l3174_317420


namespace NUMINAMATH_CALUDE_negation_equivalence_l3174_317434

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x > 2 ∨ x ≤ -1) ↔ (∀ x : ℝ, -1 < x ∧ x ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3174_317434


namespace NUMINAMATH_CALUDE_opposite_sign_pair_l3174_317482

theorem opposite_sign_pair : ∃! (x : ℝ), (x > 0 ∧ x * x = 7) ∧ 
  (∀ a b : ℝ, (a = 131 ∧ b = 1 - 31) ∨ 
              (a = x ∧ b = -x) ∨ 
              (a = 1/3 ∧ b = Real.sqrt (1/9)) ∨ 
              (a = 5^2 ∧ b = (-5)^2) →
   (a + b = 0 ∧ a * b < 0) ↔ (a = x ∧ b = -x)) :=
by sorry

end NUMINAMATH_CALUDE_opposite_sign_pair_l3174_317482


namespace NUMINAMATH_CALUDE_line_angle_and_parallel_distance_l3174_317439

/-- Line in 2D plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Angle between two lines -/
def angle_between_lines (l1 l2 : Line) : ℝ := sorry

/-- Distance between two parallel lines -/
def distance_between_parallel_lines (l1 l2 : Line) : ℝ := sorry

/-- Check if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop := sorry

theorem line_angle_and_parallel_distance 
  (l : Line) 
  (l1 : Line) 
  (l2 : Line) 
  (h1 : l.a = 1 ∧ l.b = -2 ∧ l.c = 1) 
  (h2 : l1.a = 2 ∧ l1.b = 1 ∧ l1.c = 1) 
  (h3 : are_parallel l l2) 
  (h4 : distance_between_parallel_lines l l2 = 1) : 
  (angle_between_lines l l1 = π / 2) ∧ 
  ((l2.a = l.a ∧ l2.b = l.b ∧ (l2.c = l.c - Real.sqrt 5 ∨ l2.c = l.c + Real.sqrt 5))) := 
by sorry

end NUMINAMATH_CALUDE_line_angle_and_parallel_distance_l3174_317439


namespace NUMINAMATH_CALUDE_ellipse_ratio_squared_l3174_317468

/-- For an ellipse with semi-major axis a, semi-minor axis b, and distance from center to focus c,
    if b/a = a/c and c^2 = a^2 - b^2, then (b/a)^2 = 1/2 -/
theorem ellipse_ratio_squared (a b c : ℝ) (h1 : b / a = a / c) (h2 : c^2 = a^2 - b^2) :
  (b / a)^2 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_ratio_squared_l3174_317468


namespace NUMINAMATH_CALUDE_tomato_land_area_l3174_317418

/-- Represents the farm land allocation -/
structure FarmLand where
  total : ℝ
  cleared_percentage : ℝ
  barley_percentage : ℝ
  potato_percentage : ℝ

/-- Calculates the area of land planted with tomato -/
def tomato_area (farm : FarmLand) : ℝ :=
  let cleared_land := farm.total * farm.cleared_percentage
  let barley_land := cleared_land * farm.barley_percentage
  let potato_land := cleared_land * farm.potato_percentage
  cleared_land - (barley_land + potato_land)

/-- Theorem stating the area of land planted with tomato -/
theorem tomato_land_area : 
  let farm := FarmLand.mk 1000 0.9 0.8 0.1
  tomato_area farm = 90 := by
  sorry


end NUMINAMATH_CALUDE_tomato_land_area_l3174_317418


namespace NUMINAMATH_CALUDE_lifespan_survey_is_sample_l3174_317483

/-- Represents a collection of data from a survey --/
structure SurveyData where
  size : Nat
  provinces : Nat
  dataType : Type

/-- Defines what constitutes a sample in statistical terms --/
def IsSample (data : SurveyData) : Prop :=
  data.size < population_size ∧ data.size > 0
  where population_size : Nat := 1000000  -- Arbitrary large number for illustration

/-- The theorem to be proved --/
theorem lifespan_survey_is_sample :
  let survey : SurveyData := {
    size := 2500,
    provinces := 11,
    dataType := Nat  -- Assuming lifespan is measured in years
  }
  IsSample survey := by sorry


end NUMINAMATH_CALUDE_lifespan_survey_is_sample_l3174_317483


namespace NUMINAMATH_CALUDE_function_inequality_implies_range_l3174_317438

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) + Real.log (x + Real.sqrt (x^2 + 1))

theorem function_inequality_implies_range (a : ℝ) :
  (∀ x : ℝ, x ≥ 2 → f (x^2 + 2) + f (-2*a*x) ≥ 0) →
  a ≤ 3/2 :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_implies_range_l3174_317438


namespace NUMINAMATH_CALUDE_palace_rotation_l3174_317449

theorem palace_rotation (x : ℕ) : 
  (x % 30 = 15 ∧ x % 50 = 25 ∧ x % 70 = 35) → x ≥ 525 :=
by sorry

end NUMINAMATH_CALUDE_palace_rotation_l3174_317449


namespace NUMINAMATH_CALUDE_josh_bracelets_l3174_317421

-- Define the parameters
def cost_per_bracelet : ℚ := 1
def selling_price : ℚ := 1.5
def cookie_cost : ℚ := 3
def money_left : ℚ := 3

-- Define the function to calculate the number of bracelets
def num_bracelets : ℚ := (cookie_cost + money_left) / (selling_price - cost_per_bracelet)

-- Theorem statement
theorem josh_bracelets : num_bracelets = 12 := by
  sorry

end NUMINAMATH_CALUDE_josh_bracelets_l3174_317421


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l3174_317488

/-- Given two concentric circles where the radius of the outer circle is twice
    the radius of the inner circle, and the width of the gray region between
    them is 3 feet, prove that the area of the gray region is 21π square feet. -/
theorem area_between_concentric_circles (r : ℝ) : 
  r > 0 → -- Inner circle radius is positive
  2 * r - r = 3 → -- Width of gray region is 3
  π * (2 * r)^2 - π * r^2 = 21 * π := by
  sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_l3174_317488


namespace NUMINAMATH_CALUDE_max_value_sine_function_l3174_317496

theorem max_value_sine_function (ω : ℝ) (h1 : 0 < ω) (h2 : ω < 1) :
  (∀ x ∈ Set.Icc 0 (π/3), 2 * Real.sin (ω * x) ≤ Real.sqrt 2) ∧
  (∃ x ∈ Set.Icc 0 (π/3), 2 * Real.sin (ω * x) = Real.sqrt 2) →
  ω = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_sine_function_l3174_317496


namespace NUMINAMATH_CALUDE_carters_increased_baking_l3174_317463

def usual_cheesecakes : ℕ := 6
def usual_muffins : ℕ := 5
def usual_redvelvet : ℕ := 8
def tripling_factor : ℕ := 3

theorem carters_increased_baking :
  (usual_cheesecakes + usual_muffins + usual_redvelvet) * tripling_factor -
  (usual_cheesecakes + usual_muffins + usual_redvelvet) = 38 :=
by sorry

end NUMINAMATH_CALUDE_carters_increased_baking_l3174_317463


namespace NUMINAMATH_CALUDE_max_value_of_g_l3174_317435

-- Define the function g(x)
def g (x : ℝ) : ℝ := 4 * x - x^3

-- State the theorem
theorem max_value_of_g :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 2 ∧
  (∀ (y : ℝ), 0 ≤ y ∧ y ≤ 2 → g y ≤ g x) ∧
  g x = 16 * Real.sqrt 3 / 9 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_g_l3174_317435


namespace NUMINAMATH_CALUDE_students_not_taking_test_l3174_317448

theorem students_not_taking_test
  (total_students : ℕ)
  (correct_q1 : ℕ)
  (correct_q2 : ℕ)
  (h1 : total_students = 25)
  (h2 : correct_q1 = 22)
  (h3 : correct_q2 = 20)
  : total_students - max correct_q1 correct_q2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_students_not_taking_test_l3174_317448


namespace NUMINAMATH_CALUDE_largest_k_for_inequality_l3174_317494

theorem largest_k_for_inequality : 
  (∃ (k : ℝ), ∀ (x : ℝ), (1 + Real.sin x) / (2 + Real.cos x) ≥ k) ∧ 
  (∀ (k : ℝ), k > 4/3 → ¬(∃ (x : ℝ), (1 + Real.sin x) / (2 + Real.cos x) ≥ k)) :=
sorry

end NUMINAMATH_CALUDE_largest_k_for_inequality_l3174_317494


namespace NUMINAMATH_CALUDE_lottery_probability_l3174_317429

/-- The number of people participating in the lottery drawing event -/
def num_people : ℕ := 5

/-- The total number of tickets in the box -/
def total_tickets : ℕ := 5

/-- The number of winning tickets -/
def winning_tickets : ℕ := 3

/-- The probability of drawing exactly 2 winning tickets in the first 3 draws
    and the last winning ticket on the 4th draw -/
def event_probability : ℚ := 3 / 10

/-- Theorem stating that the probability of the event ending exactly after
    the 4th person has drawn is 3/10 -/
theorem lottery_probability :
  (num_people = 5) →
  (total_tickets = 5) →
  (winning_tickets = 3) →
  (event_probability = 3 / 10) :=
by
  sorry

end NUMINAMATH_CALUDE_lottery_probability_l3174_317429


namespace NUMINAMATH_CALUDE_distribute_four_balls_three_boxes_l3174_317475

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes,
    with each box containing at least one ball -/
def distribute_balls (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 3 ways to distribute 4 indistinguishable balls into 3 distinguishable boxes,
    with each box containing at least one ball -/
theorem distribute_four_balls_three_boxes :
  distribute_balls 4 3 = 3 := by sorry

end NUMINAMATH_CALUDE_distribute_four_balls_three_boxes_l3174_317475


namespace NUMINAMATH_CALUDE_fixed_point_sum_l3174_317474

theorem fixed_point_sum (a : ℝ) (m n : ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : n = a * (m - 1) + 2) : m + n = 4 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_sum_l3174_317474


namespace NUMINAMATH_CALUDE_rectangle_area_from_equilateral_triangle_l3174_317456

theorem rectangle_area_from_equilateral_triangle (triangle_area : ℝ) : 
  triangle_area = 9 * Real.sqrt 3 →
  ∃ (triangle_side : ℝ), 
    triangle_area = (Real.sqrt 3 / 4) * triangle_side^2 ∧
    ∃ (rect_width rect_length : ℝ),
      rect_width = triangle_side ∧
      rect_length = 3 * rect_width ∧
      rect_width * rect_length = 108 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_from_equilateral_triangle_l3174_317456


namespace NUMINAMATH_CALUDE_fourth_root_closest_to_6700_l3174_317415

def n : ℕ := 2001200120012001

def options : List ℕ := [2001, 6700, 21000, 12000, 2100]

theorem fourth_root_closest_to_6700 :
  ∃ (x : ℝ), x^4 = n ∧ 
  ∀ y ∈ options, |x - 6700| ≤ |x - y| :=
sorry

end NUMINAMATH_CALUDE_fourth_root_closest_to_6700_l3174_317415


namespace NUMINAMATH_CALUDE_smallest_fraction_between_l3174_317462

theorem smallest_fraction_between (r s : ℕ+) : 
  (7 : ℚ)/11 < r/s ∧ r/s < (5 : ℚ)/8 ∧ 
  (∀ r' s' : ℕ+, (7 : ℚ)/11 < r'/s' ∧ r'/s' < (5 : ℚ)/8 → s ≤ s') →
  s - r = 10 := by
sorry

end NUMINAMATH_CALUDE_smallest_fraction_between_l3174_317462


namespace NUMINAMATH_CALUDE_max_additional_plates_l3174_317402

def first_set : Finset Char := {'B', 'F', 'J', 'M', 'S'}
def second_set : Finset Char := {'E', 'U', 'Y'}
def third_set : Finset Char := {'G', 'K', 'R', 'Z'}

theorem max_additional_plates :
  ∃ (new_first : Char) (new_third : Char),
    new_first ∉ first_set ∧
    new_third ∉ third_set ∧
    (first_set.card + 1) * second_set.card * (third_set.card + 1) -
    first_set.card * second_set.card * third_set.card = 30 ∧
    ∀ (a : Char) (c : Char),
      a ∉ first_set →
      c ∉ third_set →
      (first_set.card + 1) * second_set.card * (third_set.card + 1) -
      first_set.card * second_set.card * third_set.card ≤ 30 :=
by sorry

end NUMINAMATH_CALUDE_max_additional_plates_l3174_317402


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3174_317444

-- Define set A
def A : Set Int := {x | (x + 2) * (x - 1) < 0}

-- Define set B
def B : Set Int := {-2, -1}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = {-2, -1, 0} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3174_317444


namespace NUMINAMATH_CALUDE_sum_of_roots_l3174_317427

theorem sum_of_roots (c d : ℝ) 
  (hc : c^3 - 21*c^2 + 28*c - 70 = 0) 
  (hd : 10*d^3 - 75*d^2 - 350*d + 3225 = 0) : 
  c + d = 21/2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3174_317427


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3174_317467

/-- Proves that given the conditions of simple and compound interest, the interest rate is 18.50% -/
theorem interest_rate_calculation (P : ℝ) (R : ℝ) : 
  P * R * 2 / 100 = 55 →
  P * ((1 + R / 100)^2 - 1) = 56.375 →
  R = 18.50 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l3174_317467


namespace NUMINAMATH_CALUDE_mean_of_combined_sets_l3174_317478

theorem mean_of_combined_sets (set1_count : Nat) (set1_mean : ℚ) (set2_count : Nat) (set2_mean : ℚ) 
  (h1 : set1_count = 7)
  (h2 : set1_mean = 15)
  (h3 : set2_count = 8)
  (h4 : set2_mean = 18) :
  let total_count := set1_count + set2_count
  let total_sum := set1_count * set1_mean + set2_count * set2_mean
  total_sum / total_count = 249 / 15 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_combined_sets_l3174_317478


namespace NUMINAMATH_CALUDE_fruit_shopping_cost_l3174_317451

/-- Calculates the price per unit of fruit given the number of fruits and their total price in cents. -/
def price_per_unit (num_fruits : ℕ) (total_price : ℕ) : ℚ :=
  total_price / num_fruits

/-- Determines the cheaper fruit given their prices per unit. -/
def cheaper_fruit (apple_price : ℚ) (orange_price : ℚ) : ℚ :=
  min apple_price orange_price

theorem fruit_shopping_cost :
  let apple_price := price_per_unit 10 200  -- 10 apples for $2 (200 cents)
  let orange_price := price_per_unit 5 150  -- 5 oranges for $1.50 (150 cents)
  let cheaper_price := cheaper_fruit apple_price orange_price
  (12 : ℕ) * (cheaper_price : ℚ) = 240
  := by sorry

end NUMINAMATH_CALUDE_fruit_shopping_cost_l3174_317451
