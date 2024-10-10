import Mathlib

namespace BE_length_l2268_226801

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  dist A B = 3 ∧ dist B C = 4 ∧ dist C A = 5

-- Define points D and E on ray AB
def points_on_ray (A B D E : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, t₁ > 1 ∧ t₂ > t₁ ∧ D = A + t₁ • (B - A) ∧ E = A + t₂ • (B - A)

-- Define point F as intersection of circumcircles
def point_F (A B C D E F : ℝ × ℝ) : Prop :=
  F ≠ C ∧
  ∃ r₁ r₂ : ℝ,
    dist A F = r₁ ∧ dist C F = r₁ ∧ dist D F = r₁ ∧
    dist E F = r₂ ∧ dist B F = r₂ ∧ dist C F = r₂

-- Main theorem
theorem BE_length (A B C D E F : ℝ × ℝ) :
  triangle_ABC A B C →
  points_on_ray A B D E →
  point_F A B C D E F →
  dist D F = 3 →
  dist E F = 8 →
  dist B E = 3 + Real.sqrt 34.6 :=
sorry

end BE_length_l2268_226801


namespace probability_A_hits_twice_B_hits_thrice_l2268_226863

def probability_A_hits : ℚ := 2/3
def probability_B_hits : ℚ := 3/4
def num_shots : ℕ := 4
def num_A_hits : ℕ := 2
def num_B_hits : ℕ := 3

theorem probability_A_hits_twice_B_hits_thrice : 
  (Nat.choose num_shots num_A_hits * probability_A_hits ^ num_A_hits * (1 - probability_A_hits) ^ (num_shots - num_A_hits)) *
  (Nat.choose num_shots num_B_hits * probability_B_hits ^ num_B_hits * (1 - probability_B_hits) ^ (num_shots - num_B_hits)) = 1/8 := by
  sorry

end probability_A_hits_twice_B_hits_thrice_l2268_226863


namespace post_office_mail_count_l2268_226844

/-- The number of pieces of mail handled by a post office in six months -/
def mail_in_six_months (letters_per_day : ℕ) (packages_per_day : ℕ) (days_per_month : ℕ) (num_months : ℕ) : ℕ :=
  (letters_per_day + packages_per_day) * (days_per_month * num_months)

/-- Theorem stating that the post office handles 14,400 pieces of mail in six months -/
theorem post_office_mail_count :
  mail_in_six_months 60 20 30 6 = 14400 := by
  sorry

end post_office_mail_count_l2268_226844


namespace first_runner_pace_correct_l2268_226866

/-- The average pace of the first runner in a race with the following conditions:
  * The race is 10 miles long.
  * The second runner's pace is 7 minutes per mile.
  * The second runner stops after 56 minutes.
  * The second runner could remain stopped for 8 minutes before the first runner catches up.
-/
def firstRunnerPace : ℝ :=
  let raceLength : ℝ := 10
  let secondRunnerPace : ℝ := 7
  let secondRunnerStopTime : ℝ := 56
  let catchUpTime : ℝ := 8
  
  4  -- The actual pace, to be proved

theorem first_runner_pace_correct :
  let raceLength : ℝ := 10
  let secondRunnerPace : ℝ := 7
  let secondRunnerStopTime : ℝ := 56
  let catchUpTime : ℝ := 8
  
  firstRunnerPace = 4 := by sorry

end first_runner_pace_correct_l2268_226866


namespace a_gt_2_sufficient_not_necessary_for_a_sq_gt_2a_l2268_226818

theorem a_gt_2_sufficient_not_necessary_for_a_sq_gt_2a :
  (∀ a : ℝ, a > 2 → a^2 > 2*a) ∧
  (∃ a : ℝ, a^2 > 2*a ∧ a ≤ 2) :=
by sorry

end a_gt_2_sufficient_not_necessary_for_a_sq_gt_2a_l2268_226818


namespace max_label_proof_l2268_226812

/-- Counts the number of '5' digits used to label boxes from 1 to n --/
def count_fives (n : ℕ) : ℕ := sorry

/-- The maximum number that can be labeled using 50 '5' digits --/
def max_label : ℕ := 235

theorem max_label_proof :
  count_fives max_label ≤ 50 ∧
  ∀ m : ℕ, m > max_label → count_fives m > 50 :=
sorry

end max_label_proof_l2268_226812


namespace value_of_a_l2268_226885

-- Define the operation *
def star (x y : ℝ) : ℝ := x + y - x * y

-- Define a
def a : ℝ := star 1 (star 0 1)

-- Theorem statement
theorem value_of_a : a = 1 := by
  sorry

end value_of_a_l2268_226885


namespace women_average_age_l2268_226810

/-- The average age of two women given the following conditions:
    1. There are initially 10 men.
    2. When two women replace two men (aged 10 and 12), the average age increases by 2 years.
    3. The number of people remains 10 after the replacement. -/
theorem women_average_age (T : ℕ) : 
  (T : ℝ) / 10 + 2 = (T - 10 - 12 + 42) / 10 → 21 = 42 / 2 := by
  sorry

end women_average_age_l2268_226810


namespace smallest_perfect_square_multiplier_l2268_226811

def y : ℕ := 3^(4^(5^(6^(7^(8^(9^10))))))

theorem smallest_perfect_square_multiplier :
  ∃ (k : ℕ), k > 0 ∧ 
  (∃ (n : ℕ), k * y = n^2) ∧
  (∀ (m : ℕ), m > 0 → m < k → ¬∃ (n : ℕ), m * y = n^2) ∧
  k = 75 := by
sorry

end smallest_perfect_square_multiplier_l2268_226811


namespace emmas_garden_area_l2268_226841

theorem emmas_garden_area :
  ∀ (short_posts long_posts : ℕ) (short_side long_side : ℝ),
  short_posts > 1 ∧
  long_posts > 1 ∧
  short_posts + long_posts = 12 ∧
  long_posts = 3 * short_posts ∧
  short_side = 6 * (short_posts - 1) ∧
  long_side = 6 * (long_posts - 1) →
  short_side * long_side = 576 := by
sorry

end emmas_garden_area_l2268_226841


namespace age_ratio_proof_l2268_226815

def sachin_age : ℚ := 24.5
def age_difference : ℕ := 7

theorem age_ratio_proof :
  let rahul_age : ℚ := sachin_age + age_difference
  (sachin_age / rahul_age) = 7 / 9 := by sorry

end age_ratio_proof_l2268_226815


namespace ball_count_theorem_l2268_226814

theorem ball_count_theorem (red_balls : ℕ) (white_balls : ℕ) (total_balls : ℕ) (prob_red : ℚ) : 
  red_balls = 4 →
  total_balls = red_balls + white_balls →
  prob_red = 1/4 →
  (red_balls : ℚ) / total_balls = prob_red →
  white_balls = 12 := by
sorry

end ball_count_theorem_l2268_226814


namespace love_betty_jane_l2268_226892

variable (A B : Prop)

theorem love_betty_jane : ((A → B) → A) → (A ∧ (B ∨ ¬B)) :=
  sorry

end love_betty_jane_l2268_226892


namespace sound_engineer_selection_probability_l2268_226833

theorem sound_engineer_selection_probability :
  let total_candidates : ℕ := 5
  let selected_engineers : ℕ := 3
  let specific_engineers : ℕ := 2

  let total_combinations := Nat.choose total_candidates selected_engineers
  let favorable_outcomes := 
    Nat.choose specific_engineers 1 * Nat.choose (total_candidates - specific_engineers) (selected_engineers - 1) +
    Nat.choose specific_engineers 2 * Nat.choose (total_candidates - specific_engineers) (selected_engineers - 2)

  (favorable_outcomes : ℚ) / total_combinations = 9 / 10 :=
by
  sorry

end sound_engineer_selection_probability_l2268_226833


namespace dianna_problem_l2268_226852

def correct_expression (f : ℤ) : ℤ := 1 - (2 - (3 - (4 + (5 - f))))

def misinterpreted_expression (f : ℤ) : ℤ := 1 - 2 - 3 - 4 + 5 - f

theorem dianna_problem : ∃ f : ℤ, correct_expression f = misinterpreted_expression f ∧ f = 2 := by
  sorry

end dianna_problem_l2268_226852


namespace parabola_and_line_intersection_l2268_226853

/-- Given a parabola E and a line intersecting it, prove properties about the parabola equation and slopes of lines connecting intersection points to a fixed point. -/
theorem parabola_and_line_intersection (p m : ℝ) (h_p : p > 0) : 
  let E := {(x, y) : ℝ × ℝ | y^2 = 2*p*x}
  let L := {(x, y) : ℝ × ℝ | x = m*y + 3}
  let A := (x₁, y₁)
  let B := (x₂, y₂)
  let C := (-3, 0)
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (A ∈ E ∧ A ∈ L) ∧ 
    (B ∈ E ∧ B ∈ L) ∧ 
    (x₁ * x₂ + y₁ * y₂ = 6) →
    (p = 1/2) ∧
    (let k₁ := (y₁ - 0) / (x₁ - (-3))
     let k₂ := (y₂ - 0) / (x₂ - (-3))
     1/k₁^2 + 1/k₂^2 - 2*m^2 = 24) := by
  sorry

end parabola_and_line_intersection_l2268_226853


namespace min_value_expression_l2268_226871

theorem min_value_expression (x : ℝ) (h : x > 1) :
  (x + 12) / Real.sqrt (x - 1) ≥ 2 * Real.sqrt 13 ∧
  ∃ y : ℝ, y > 1 ∧ (y + 12) / Real.sqrt (y - 1) = 2 * Real.sqrt 13 :=
by sorry

end min_value_expression_l2268_226871


namespace crickets_to_collect_l2268_226881

theorem crickets_to_collect (collected : ℕ) (target : ℕ) (additional : ℕ) : 
  collected = 7 → target = 11 → additional = target - collected :=
by
  sorry

end crickets_to_collect_l2268_226881


namespace negation_of_proposition_l2268_226868

theorem negation_of_proposition (p : Prop) :
  (¬ (∀ x : ℝ, x > 0 → 3 * x + 1 < 0)) ↔ (∃ x : ℝ, x > 0 ∧ 3 * x + 1 ≥ 0) := by
  sorry

end negation_of_proposition_l2268_226868


namespace complex_number_additive_inverse_l2268_226894

theorem complex_number_additive_inverse (b : ℝ) : 
  let z : ℂ := (2 - b * Complex.I) / (1 + 2 * Complex.I)
  (z.re = -z.im) → b = -2/3 := by
  sorry

end complex_number_additive_inverse_l2268_226894


namespace expression_value_l2268_226838

theorem expression_value (a b m n x : ℝ) 
  (h1 : a = -b) 
  (h2 : m * n = 1) 
  (h3 : |x| = 2) : 
  -2*m*n + 3*(a+b) - x = -4 ∨ -2*m*n + 3*(a+b) - x = 0 := by
  sorry

end expression_value_l2268_226838


namespace number_division_remainder_l2268_226862

theorem number_division_remainder (N : ℤ) (D : ℤ) : 
  N % 281 = 160 → N % D = 21 → D = 139 := by
  sorry

end number_division_remainder_l2268_226862


namespace rectangle_from_equal_bisecting_diagonals_parallelogram_from_bisecting_diagonals_square_from_rhombus_equal_diagonals_l2268_226891

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define properties of quadrilaterals
def has_equal_diagonals (q : Quadrilateral) : Prop := sorry
def diagonals_bisect_each_other (q : Quadrilateral) : Prop := sorry
def is_rectangle (q : Quadrilateral) : Prop := sorry
def is_parallelogram (q : Quadrilateral) : Prop := sorry
def is_rhombus (q : Quadrilateral) : Prop := sorry
def is_square (q : Quadrilateral) : Prop := sorry

-- Theorems to prove
theorem rectangle_from_equal_bisecting_diagonals (q : Quadrilateral) :
  has_equal_diagonals q → diagonals_bisect_each_other q → is_rectangle q := by sorry

theorem parallelogram_from_bisecting_diagonals (q : Quadrilateral) :
  diagonals_bisect_each_other q → is_parallelogram q := by sorry

theorem square_from_rhombus_equal_diagonals (q : Quadrilateral) :
  is_rhombus q → has_equal_diagonals q → is_square q := by sorry

end rectangle_from_equal_bisecting_diagonals_parallelogram_from_bisecting_diagonals_square_from_rhombus_equal_diagonals_l2268_226891


namespace tree_spacing_l2268_226865

/-- Given a road of length 151 feet where 11 trees can be planted, with each tree occupying 1 foot of space, 
    the distance between each tree is 14 feet. -/
theorem tree_spacing (road_length : ℕ) (num_trees : ℕ) (tree_space : ℕ) 
    (h1 : road_length = 151)
    (h2 : num_trees = 11)
    (h3 : tree_space = 1) : 
  (road_length - num_trees * tree_space) / (num_trees - 1) = 14 := by
  sorry


end tree_spacing_l2268_226865


namespace rectangle_max_area_l2268_226873

theorem rectangle_max_area (p : ℝ) (h1 : p = 40) : 
  ∃ (l w : ℝ), l > 0 ∧ w > 0 ∧ 2 * (l + w) = p ∧ w = 2 * l ∧ l * w = 800 / 9 := by
  sorry

end rectangle_max_area_l2268_226873


namespace equation_is_parabola_l2268_226837

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents the equation |y - 3| = √((x+4)² + (y-1)²) -/
def equation (p : Point2D) : Prop :=
  |p.y - 3| = Real.sqrt ((p.x + 4)^2 + (p.y - 1)^2)

/-- Represents a parabola in general form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point satisfies the parabola equation -/
def satisfies_parabola (p : Point2D) (para : Parabola) : Prop :=
  p.y = para.a * p.x^2 + para.b * p.x + para.c

/-- Theorem stating that the given equation represents a parabola -/
theorem equation_is_parabola :
  ∃ (para : Parabola), ∀ (p : Point2D), equation p → satisfies_parabola p para :=
sorry

end equation_is_parabola_l2268_226837


namespace arithmetic_geometric_mean_sum_squares_l2268_226835

theorem arithmetic_geometric_mean_sum_squares (x y : ℝ) :
  (x + y) / 2 = 20 →
  Real.sqrt (x * y) = Real.sqrt 110 →
  x^2 + y^2 = 1380 := by
  sorry

end arithmetic_geometric_mean_sum_squares_l2268_226835


namespace customers_in_us_l2268_226821

theorem customers_in_us (total : ℕ) (other_countries : ℕ) (h1 : total = 7422) (h2 : other_countries = 6699) :
  total - other_countries = 723 := by
  sorry

end customers_in_us_l2268_226821


namespace anne_cleaning_time_l2268_226874

variable (B A C : ℝ)

-- Define the conditions
def condition1 : Prop := B + A + C = 1/4
def condition2 : Prop := B + 2*A + 3*C = 1/3
def condition3 : Prop := B + C = 1/6

-- Theorem statement
theorem anne_cleaning_time 
  (h1 : condition1 B A C) 
  (h2 : condition2 B A C) 
  (h3 : condition3 B C) : 
  1/A = 12 := by
sorry

end anne_cleaning_time_l2268_226874


namespace fuchsia_survey_l2268_226897

/-- Given a survey about the color fuchsia with the following parameters:
  * total_surveyed: Total number of people surveyed
  * mostly_pink: Number of people who believe fuchsia is "mostly pink"
  * both: Number of people who believe fuchsia is both "mostly pink" and "mostly purple"
  * neither: Number of people who believe fuchsia is neither "mostly pink" nor "mostly purple"

  This theorem proves that the number of people who believe fuchsia is "mostly purple"
  is equal to total_surveyed - (mostly_pink - both) - neither.
-/
theorem fuchsia_survey (total_surveyed mostly_pink both neither : ℕ)
  (h1 : total_surveyed = 150)
  (h2 : mostly_pink = 80)
  (h3 : both = 40)
  (h4 : neither = 25) :
  total_surveyed - (mostly_pink - both) - neither = 85 := by
  sorry

#check fuchsia_survey

end fuchsia_survey_l2268_226897


namespace shortest_side_right_triangle_l2268_226857

theorem shortest_side_right_triangle (a b c : ℝ) (h_right_angle : a^2 + b^2 = c^2) 
  (h_a : a = 7) (h_b : b = 24) : 
  min a b = 7 := by sorry

end shortest_side_right_triangle_l2268_226857


namespace remainder_theorem_l2268_226847

theorem remainder_theorem (r : ℤ) : (r^17 + 1) % (r - 1) = 2 := by
  sorry

end remainder_theorem_l2268_226847


namespace sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l2268_226878

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁ + x₂ = -b / a := by sorry

theorem sum_of_roots_specific_quadratic :
  let a : ℝ := 3
  let b : ℝ := -12
  let c : ℝ := 9
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁ + x₂ = 4 := by sorry

end sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l2268_226878


namespace george_exchange_rate_l2268_226843

/-- The amount George will receive for each special bill he exchanges on his 25th birthday. -/
def exchange_rate (total_years : ℕ) (spent_percentage : ℚ) (total_exchange_amount : ℚ) : ℚ :=
  let total_bills := total_years
  let remaining_bills := total_bills - (spent_percentage * total_bills)
  total_exchange_amount / remaining_bills

/-- Theorem stating that George will receive $1.50 for each special bill he exchanges. -/
theorem george_exchange_rate :
  exchange_rate 10 (1/5) 12 = 3/2 := by
  sorry

end george_exchange_rate_l2268_226843


namespace function_representation_flexibility_l2268_226822

-- Define a function type
def Function (α : Type) (β : Type) := α → β

-- State the theorem
theorem function_representation_flexibility 
  {α β : Type} (f : Function α β) : 
  ¬ (∀ (formula : α → β), f = formula) :=
sorry

end function_representation_flexibility_l2268_226822


namespace power_two_greater_than_square_plus_one_l2268_226877

theorem power_two_greater_than_square_plus_one (n : ℕ) (h : n ≥ 5) :
  2^n > n^2 + 1 := by
  sorry

end power_two_greater_than_square_plus_one_l2268_226877


namespace rectangle_area_inequality_l2268_226851

theorem rectangle_area_inequality : ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 16 * 10 = 23 * 7 + ε := by
  sorry

end rectangle_area_inequality_l2268_226851


namespace boy_late_to_school_l2268_226869

/-- Proves that a boy traveling to school was 1 hour late on the first day given specific conditions -/
theorem boy_late_to_school (distance : ℝ) (speed_day1 speed_day2 : ℝ) (early_time : ℝ) : 
  distance = 60 ∧ 
  speed_day1 = 10 ∧ 
  speed_day2 = 20 ∧ 
  early_time = 1 ∧
  distance / speed_day2 + early_time = distance / speed_day1 - 1 →
  distance / speed_day1 - (distance / speed_day2 + early_time) = 1 :=
by
  sorry

#check boy_late_to_school

end boy_late_to_school_l2268_226869


namespace greatest_integer_with_conditions_l2268_226826

theorem greatest_integer_with_conditions : ∃ n : ℕ, 
  n < 150 ∧ 
  (∃ a b : ℕ, n + 2 = 9 * a ∧ n + 3 = 11 * b) ∧
  (∀ m : ℕ, m < 150 → (∃ c d : ℕ, m + 2 = 9 * c ∧ m + 3 = 11 * d) → m ≤ n) ∧
  n = 142 := by
  sorry

end greatest_integer_with_conditions_l2268_226826


namespace max_value_of_f_l2268_226823

/-- The function f(x) = x^3 - 3ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x + 2

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 3*a

theorem max_value_of_f (a : ℝ) :
  (∃ δ > 0, ∀ x, 0 < |x - 2| ∧ |x - 2| < δ → f a x ≥ f a 2) →
  (∃ x, f a x = 18 ∧ ∀ y, f a y ≤ f a x) :=
sorry

end max_value_of_f_l2268_226823


namespace quinn_reading_challenge_l2268_226805

/-- The number of books Quinn needs to read to get one free donut -/
def books_per_donut (books_per_week : ℕ) (weeks : ℕ) (total_donuts : ℕ) : ℕ :=
  (books_per_week * weeks) / total_donuts

/-- Proof that Quinn needs to read 5 books to get one free donut -/
theorem quinn_reading_challenge :
  books_per_donut 2 10 4 = 5 := by
  sorry

end quinn_reading_challenge_l2268_226805


namespace equation_transformation_l2268_226850

theorem equation_transformation (x y : ℝ) : x = y → -2 * x = -2 * y := by
  sorry

end equation_transformation_l2268_226850


namespace ten_streets_intersections_l2268_226802

/-- Represents a city with straight streets -/
structure City where
  num_streets : ℕ
  no_parallel_streets : Bool

/-- Calculates the maximum number of intersections in a city -/
def max_intersections (city : City) : ℕ :=
  if city.num_streets ≤ 1 then 0
  else (city.num_streets - 1) * (city.num_streets - 2) / 2

/-- Theorem: A city with 10 straight streets where no two are parallel has 45 intersections -/
theorem ten_streets_intersections :
  ∀ (c : City), c.num_streets = 10 → c.no_parallel_streets = true →
  max_intersections c = 45 := by
  sorry

end ten_streets_intersections_l2268_226802


namespace intersection_implies_sum_l2268_226855

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | x^2 + x - 6 < 0}

-- Define the set C with parameters a and b
def C (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b < 0}

-- Theorem statement
theorem intersection_implies_sum (a b : ℝ) :
  C a b = A ∩ B → a + b = -3 := by
  sorry

end intersection_implies_sum_l2268_226855


namespace towers_count_correct_l2268_226807

def number_of_towers (red green blue : ℕ) (height : ℕ) : ℕ :=
  let total := red + green + blue
  let leftout := total - height
  if leftout ≠ 1 then 0
  else
    (Nat.choose total height) *
    (Nat.factorial height / (Nat.factorial red * Nat.factorial (green - 1) * Nat.factorial blue) +
     Nat.factorial height / (Nat.factorial red * Nat.factorial green * Nat.factorial (blue - 1)) +
     Nat.factorial height / (Nat.factorial (red - 1) * Nat.factorial green * Nat.factorial blue))

theorem towers_count_correct :
  number_of_towers 3 4 4 10 = 26250 := by
  sorry

end towers_count_correct_l2268_226807


namespace volume_ratio_is_twenty_l2268_226808

-- Define the dimensions of the boxes
def sehee_side : ℝ := 1  -- 1 meter
def serin_width : ℝ := 0.5  -- 50 cm in meters
def serin_depth : ℝ := 0.5  -- 50 cm in meters
def serin_height : ℝ := 0.2  -- 20 cm in meters

-- Define the volumes of the boxes
def sehee_volume : ℝ := sehee_side ^ 3
def serin_volume : ℝ := serin_width * serin_depth * serin_height

-- State the theorem
theorem volume_ratio_is_twenty :
  sehee_volume / serin_volume = 20 := by sorry

end volume_ratio_is_twenty_l2268_226808


namespace regular_iff_all_face_angles_equal_exists_non_regular_with_five_equal_angles_l2268_226816

-- Define a tetrahedron
structure Tetrahedron where
  A : Point
  B : Point
  C : Point
  D : Point

-- Define a function to calculate the angle between two faces of a tetrahedron
def angleBetweenFaces (t : Tetrahedron) (face1 : Fin 4) (face2 : Fin 4) : ℝ := sorry

-- Define what it means for a tetrahedron to be regular
def isRegular (t : Tetrahedron) : Prop := sorry

-- Define what it means for all face angles to be equal
def allFaceAnglesEqual (t : Tetrahedron) : Prop :=
  ∀ (i j k l : Fin 4), i ≠ j ∧ k ≠ l → angleBetweenFaces t i j = angleBetweenFaces t k l

-- Define what it means for five out of six face angles to be equal
def fiveFaceAnglesEqual (t : Tetrahedron) : Prop :=
  ∃ (i j k l m n : Fin 4), i ≠ j ∧ k ≠ l ∧ m ≠ n ∧
    angleBetweenFaces t i j = angleBetweenFaces t k l ∧
    angleBetweenFaces t i j = angleBetweenFaces t m n ∧
    (∀ (a b : Fin 4), a ≠ b → 
      angleBetweenFaces t a b = angleBetweenFaces t i j ∨
      angleBetweenFaces t a b = angleBetweenFaces t k l ∨
      angleBetweenFaces t a b = angleBetweenFaces t m n)

-- Theorem 1: A tetrahedron is regular if and only if all face angles are equal
theorem regular_iff_all_face_angles_equal (t : Tetrahedron) :
  isRegular t ↔ allFaceAnglesEqual t := by sorry

-- Theorem 2: There exists a non-regular tetrahedron with five equal face angles
theorem exists_non_regular_with_five_equal_angles :
  ∃ (t : Tetrahedron), fiveFaceAnglesEqual t ∧ ¬isRegular t := by sorry

end regular_iff_all_face_angles_equal_exists_non_regular_with_five_equal_angles_l2268_226816


namespace intersection_M_N_l2268_226806

def M : Set ℕ := {1, 2, 3, 4}

def N : Set ℕ := {x | ∃ n ∈ M, x = n^2}

theorem intersection_M_N : M ∩ N = {1, 4} := by
  sorry

end intersection_M_N_l2268_226806


namespace equation_solution_l2268_226883

theorem equation_solution : ∃ x : ℚ, (2 / 7) * (1 / 4) * x = 12 ∧ x = 168 := by
  sorry

end equation_solution_l2268_226883


namespace semicircle_radius_l2268_226827

-- Define the triangle PQR
structure RightTriangle where
  PQ : ℝ
  QR : ℝ
  PR : ℝ
  right_angle : PQ^2 + QR^2 = PR^2

-- Define the theorem
theorem semicircle_radius (t : RightTriangle) 
  (h1 : (1/2) * π * (t.PQ/2)^2 = 18*π) 
  (h2 : π * (t.QR/2) = 10*π) : 
  t.PR/2 = 4*Real.sqrt 17 := by
  sorry

end semicircle_radius_l2268_226827


namespace unique_number_satisfying_conditions_l2268_226824

/-- Checks if three numbers form a geometric progression -/
def is_geometric_progression (a b c : ℕ) : Prop :=
  b * b = a * c

/-- Checks if three numbers form an arithmetic progression -/
def is_arithmetic_progression (a b c : ℕ) : Prop :=
  b - a = c - b

/-- Reverses a three-digit number -/
def reverse_number (n : ℕ) : ℕ :=
  (n % 10) * 100 + ((n / 10) % 10) * 10 + (n / 100)

theorem unique_number_satisfying_conditions : ∃! n : ℕ,
  100 ≤ n ∧ n < 1000 ∧
  is_geometric_progression (n / 100) ((n / 10) % 10) (n % 10) ∧
  n - 792 = reverse_number n ∧
  is_arithmetic_progression ((n / 100) - 4) ((n / 10) % 10) (n % 10) ∧
  n = 931 := by
  sorry

end unique_number_satisfying_conditions_l2268_226824


namespace impossible_constant_average_l2268_226899

theorem impossible_constant_average (n : ℕ) (initial_total_age : ℕ) : 
  initial_total_age = n * 19 →
  ¬ ∃ (new_total_age : ℕ), new_total_age = initial_total_age + 1 ∧ 
    new_total_age / (n + 1) = 19 :=
by sorry

end impossible_constant_average_l2268_226899


namespace probability_N18_mod7_equals_1_is_2_7_l2268_226839

/-- The probability that N^18 mod 7 = 1, given N is an odd integer randomly chosen from 1 to 2023 -/
def probability_N18_mod7_equals_1 : ℚ :=
  let N := Finset.filter (fun n => n % 2 = 1) (Finset.range 2023)
  let favorable := N.filter (fun n => (n^18) % 7 = 1)
  favorable.card / N.card

theorem probability_N18_mod7_equals_1_is_2_7 :
  probability_N18_mod7_equals_1 = 2/7 := by
  sorry

end probability_N18_mod7_equals_1_is_2_7_l2268_226839


namespace complex_magnitude_problem_l2268_226880

theorem complex_magnitude_problem (z : ℂ) (h : (1 - 2*I) * z = 5*I) : 
  Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_magnitude_problem_l2268_226880


namespace delta_value_l2268_226872

-- Define the simultaneous equations
def simultaneous_equations (x y z : ℝ) : Prop :=
  x - y - z = -1 ∧ y - x - z = -2 ∧ z - x - y = -4

-- Define β
def β : ℕ := 5

-- Define γ
def γ : ℕ := 2

-- Define the polynomial equation
def polynomial_equation (a b δ : ℝ) : Prop :=
  ∃ t : ℝ, t^4 + a*t^2 + b*t + δ = 0 ∧
           1^4 + a*1^2 + b*1 + δ = 0 ∧
           γ^4 + a*γ^2 + b*γ + δ = 0 ∧
           (γ^2)^4 + a*(γ^2)^2 + b*(γ^2) + δ = 0

-- Theorem statement
theorem delta_value :
  ∃ x y z a b : ℝ,
    simultaneous_equations x y z →
    polynomial_equation a b (-56) :=
sorry

end delta_value_l2268_226872


namespace quadratic_inequality_range_l2268_226893

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, 3 * x^2 + 2 * a * x + 1 ≥ 0) ↔ (-Real.sqrt 3 ≤ a ∧ a ≤ Real.sqrt 3) := by
  sorry

end quadratic_inequality_range_l2268_226893


namespace expression_equals_three_l2268_226886

theorem expression_equals_three : 
  (1/2)⁻¹ + 4 * Real.cos (45 * π / 180) - Real.sqrt 8 + (2023 - Real.pi)^0 = 3 := by
sorry

end expression_equals_three_l2268_226886


namespace inscribed_circle_radius_345_triangle_l2268_226804

/-- The radius of the inscribed circle of a triangle with sides 3, 4, and 5 is 1 -/
theorem inscribed_circle_radius_345_triangle : 
  ∀ (a b c : ℝ) (r : ℝ), 
    a = 3 ∧ b = 4 ∧ c = 5 →
    (a + b + c) / 2 = 6 →
    r = 6 / ((a + b + c) / 2) →
    r = 1 := by
  sorry

end inscribed_circle_radius_345_triangle_l2268_226804


namespace min_pumps_needed_l2268_226861

/-- Represents the water pumping scenario -/
structure WaterPumping where
  x : ℝ  -- Amount of water already gushed out before pumping
  a : ℝ  -- Amount of water gushing out per minute
  b : ℝ  -- Amount of water each pump can pump out per minute

/-- The conditions of the water pumping problem -/
def water_pumping_conditions (w : WaterPumping) : Prop :=
  w.x + 40 * w.a = 2 * 40 * w.b ∧
  w.x + 16 * w.a = 4 * 16 * w.b ∧
  w.a > 0 ∧ w.b > 0

/-- The theorem stating the minimum number of pumps needed -/
theorem min_pumps_needed (w : WaterPumping) 
  (h : water_pumping_conditions w) : 
  ∀ n : ℕ, (w.x + 10 * w.a ≤ 10 * n * w.b) → n ≥ 6 := by
  sorry

#check min_pumps_needed

end min_pumps_needed_l2268_226861


namespace distance_focus_to_asymptote_l2268_226834

-- Define the hyperbola C
def C (x y : ℝ) : Prop := x^2 - y^2 = 2

-- Define the focus of the hyperbola
def focus : ℝ × ℝ := (2, 0)

-- Define the asymptote of the hyperbola
def asymptote (x y : ℝ) : Prop := y + x = 0

-- Theorem stating the distance from focus to asymptote
theorem distance_focus_to_asymptote :
  ∃ (d : ℝ), d = Real.sqrt 2 ∧
  ∀ (x y : ℝ), C x y → asymptote x y →
  Real.sqrt ((x - focus.1)^2 + (y - focus.2)^2) ≥ d :=
sorry

end distance_focus_to_asymptote_l2268_226834


namespace charlies_laps_l2268_226848

/-- Given Charlie's steps per lap and total steps in a session, calculate the number of complete laps --/
theorem charlies_laps (steps_per_lap : ℕ) (total_steps : ℕ) : 
  steps_per_lap = 5350 → total_steps = 13375 → (total_steps / steps_per_lap : ℕ) = 2 :=
by
  sorry

#eval (13375 / 5350 : ℕ)

end charlies_laps_l2268_226848


namespace like_term_proof_l2268_226840

def is_like_term (t₁ t₂ : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ (x y : ℝ), t₁ x y = a * x^5 * y^3 ∧ t₂ x y = b * x^5 * y^3

theorem like_term_proof (a : ℝ) :
  is_like_term (λ x y => -5 * x^5 * y^3) (λ x y => a * x^5 * y^3) := by
  sorry

end like_term_proof_l2268_226840


namespace company_workforce_l2268_226858

theorem company_workforce (initial_total : ℕ) : 
  (initial_total * 60 = initial_total * 100 * 60 / 100) →
  ((initial_total + 24) * 55 = (initial_total * 60) * 100 / (initial_total + 24)) →
  (initial_total + 24 = 288) := by
sorry

end company_workforce_l2268_226858


namespace fraction_product_equals_one_over_23426_l2268_226859

def fraction_product : ℕ → ℚ
  | 0 => 1
  | n + 1 => (n + 1 : ℚ) / (n + 5 : ℚ) * fraction_product n

theorem fraction_product_equals_one_over_23426 :
  fraction_product 49 = 1 / 23426 := by
  sorry

end fraction_product_equals_one_over_23426_l2268_226859


namespace sum_every_third_odd_integer_l2268_226860

/-- The sum of every third odd integer between 200 and 500 (inclusive) is 17400 -/
theorem sum_every_third_odd_integer : 
  (Finset.range 50).sum (fun i => 201 + 6 * i) = 17400 := by
  sorry

end sum_every_third_odd_integer_l2268_226860


namespace intersection_M_N_l2268_226890

def M : Set ℤ := {0, 1, 2}
def N : Set ℤ := {x : ℤ | -1 ≤ x ∧ x ≤ 1}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end intersection_M_N_l2268_226890


namespace no_integer_square_root_l2268_226849

-- Define the polynomial Q
def Q (x : ℤ) : ℤ := x^4 + 5*x^3 + 10*x^2 + 5*x + 25

-- Theorem statement
theorem no_integer_square_root : ∀ x : ℤ, ¬∃ y : ℤ, Q x = y^2 := by
  sorry

end no_integer_square_root_l2268_226849


namespace sum_of_six_least_l2268_226888

/-- τ(n) denotes the number of positive integer divisors of n -/
def tau (n : ℕ) : ℕ := (Nat.divisors n).card

/-- The set of positive integers n that satisfy τ(n) + τ(n+1) = 8 -/
def S : Set ℕ := {n : ℕ | n > 0 ∧ tau n + tau (n + 1) = 8}

/-- The six least elements of S -/
def six_least : Finset ℕ := sorry

theorem sum_of_six_least : (six_least.sum id) = 800 := by sorry

end sum_of_six_least_l2268_226888


namespace line_parallel_perpendicular_implies_planes_perpendicular_l2268_226809

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism and perpendicularity relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_perpendicular_implies_planes_perpendicular
  (l : Line) (α β : Plane) :
  parallel l α → perpendicular l β → plane_perpendicular α β :=
sorry

end line_parallel_perpendicular_implies_planes_perpendicular_l2268_226809


namespace elective_schemes_count_l2268_226820

def total_courses : ℕ := 10
def courses_to_choose : ℕ := 3
def conflicting_courses : ℕ := 3

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem elective_schemes_count :
  (choose conflicting_courses 1 * choose (total_courses - conflicting_courses) (courses_to_choose - 1)) +
  (choose (total_courses - conflicting_courses) courses_to_choose) = 98 :=
by sorry

end elective_schemes_count_l2268_226820


namespace min_value_theorem_l2268_226889

theorem min_value_theorem (x : ℝ) (h : x > 10) :
  (x^2 + 100) / (x - 10) ≥ 20 + 20 * Real.sqrt 2 :=
by sorry

end min_value_theorem_l2268_226889


namespace quadratic_inequality_range_l2268_226803

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x - 1 ≥ m^2 - 3*m) → 
  m < 1 ∨ m > 2 := by
  sorry

end quadratic_inequality_range_l2268_226803


namespace quadratic_inequality_l2268_226896

theorem quadratic_inequality (x : ℝ) : x^2 + x - 20 < 0 ↔ -5 < x ∧ x < 4 := by
  sorry

end quadratic_inequality_l2268_226896


namespace read_book_in_seven_weeks_l2268_226895

/-- The number of weeks required to read a book given the total pages and pages read per week. -/
def weeks_to_read (total_pages : ℕ) (pages_per_week : ℕ) : ℕ :=
  (total_pages + pages_per_week - 1) / pages_per_week

/-- Theorem stating that it takes 7 weeks to read a 2100-page book at a rate of 300 pages per week. -/
theorem read_book_in_seven_weeks :
  let total_pages : ℕ := 2100
  let pages_per_day : ℕ := 100
  let days_per_week : ℕ := 3
  let pages_per_week : ℕ := pages_per_day * days_per_week
  weeks_to_read total_pages pages_per_week = 7 := by
  sorry

end read_book_in_seven_weeks_l2268_226895


namespace tom_initial_dimes_l2268_226825

/-- Represents the number of coins Tom has -/
structure TomCoins where
  initial_pennies : ℕ
  initial_dimes : ℕ
  dad_dimes : ℕ
  dad_nickels : ℕ
  final_dimes : ℕ

/-- The theorem states that given the conditions from the problem,
    Tom's initial number of dimes was 15 -/
theorem tom_initial_dimes (coins : TomCoins)
  (h1 : coins.initial_pennies = 27)
  (h2 : coins.dad_dimes = 33)
  (h3 : coins.dad_nickels = 49)
  (h4 : coins.final_dimes = 48)
  (h5 : coins.final_dimes = coins.initial_dimes + coins.dad_dimes) :
  coins.initial_dimes = 15 := by
  sorry


end tom_initial_dimes_l2268_226825


namespace angle_A_is_pi_over_3_area_is_3_sqrt_3_over_4_l2268_226884

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_condition1 (t : Triangle) : Prop :=
  t.a^2 = t.b^2 + t.c^2 - t.b * t.c

def satisfies_condition2 (t : Triangle) : Prop :=
  t.a = Real.sqrt 7

def satisfies_condition3 (t : Triangle) : Prop :=
  t.c - t.b = 2

-- Theorem 1
theorem angle_A_is_pi_over_3 (t : Triangle) (h : satisfies_condition1 t) :
  t.A = π / 3 := by sorry

-- Theorem 2
theorem area_is_3_sqrt_3_over_4 (t : Triangle) 
  (h1 : satisfies_condition1 t) 
  (h2 : satisfies_condition2 t) 
  (h3 : satisfies_condition3 t) :
  (1/2) * t.b * t.c * Real.sin t.A = (3 * Real.sqrt 3) / 4 := by sorry

end angle_A_is_pi_over_3_area_is_3_sqrt_3_over_4_l2268_226884


namespace monkey_peach_problem_l2268_226832

/-- The number of peaches the monkey's mother originally had -/
def mothers_original_peaches (little_monkey_initial : ℕ) (peaches_given : ℕ) (mother_ratio : ℕ) : ℕ :=
  (little_monkey_initial + peaches_given) * mother_ratio + peaches_given

theorem monkey_peach_problem :
  mothers_original_peaches 6 3 3 = 30 := by
  sorry

end monkey_peach_problem_l2268_226832


namespace president_vp_from_six_l2268_226854

/-- The number of ways to choose a President and Vice-President from a group of n people -/
def choose_president_and_vp (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: There are 30 ways to choose a President and Vice-President from 6 people -/
theorem president_vp_from_six : choose_president_and_vp 6 = 30 := by
  sorry

end president_vp_from_six_l2268_226854


namespace max_value_fraction_l2268_226828

theorem max_value_fraction (x y z : ℕ) : 
  (10 ≤ x ∧ x ≤ 99) → 
  (10 ≤ y ∧ y ≤ 99) → 
  (10 ≤ z ∧ z ≤ 99) → 
  ((x + y + z) / 3 = 60) → 
  ((x + y) / z : ℚ) ≤ 17 :=
by sorry

end max_value_fraction_l2268_226828


namespace parallelogram_intersection_l2268_226879

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a line in 2D space -/
structure Line :=
  (a : Point) (b : Point)

/-- Represents a parallelogram -/
structure Parallelogram :=
  (A : Point) (B : Point) (C : Point) (D : Point)

/-- Checks if a point is inside a parallelogram -/
def isInside (p : Point) (para : Parallelogram) : Prop := sorry

/-- Checks if two lines are parallel -/
def isParallel (l1 l2 : Line) : Prop := sorry

/-- Checks if a point lies on a line -/
def isOnLine (p : Point) (l : Line) : Prop := sorry

/-- Checks if three lines intersect at a single point -/
def intersectAtOnePoint (l1 l2 l3 : Line) : Prop := sorry

theorem parallelogram_intersection 
  (ABCD : Parallelogram) 
  (M : Point) 
  (P Q R S : Point)
  (PR QS BS PD MC : Line)
  (h1 : isInside M ABCD)
  (h2 : isParallel PR (Line.mk ABCD.B ABCD.C))
  (h3 : isParallel QS (Line.mk ABCD.A ABCD.B))
  (h4 : isOnLine P (Line.mk ABCD.A ABCD.B))
  (h5 : isOnLine Q (Line.mk ABCD.B ABCD.C))
  (h6 : isOnLine R (Line.mk ABCD.C ABCD.D))
  (h7 : isOnLine S (Line.mk ABCD.D ABCD.A))
  (h8 : PR = Line.mk P R)
  (h9 : QS = Line.mk Q S)
  (h10 : BS = Line.mk ABCD.B S)
  (h11 : PD = Line.mk P ABCD.D)
  (h12 : MC = Line.mk M ABCD.C)
  : intersectAtOnePoint BS PD MC := sorry

end parallelogram_intersection_l2268_226879


namespace snow_on_tuesday_l2268_226875

theorem snow_on_tuesday (monday_snow : ℝ) (total_snow : ℝ) (h1 : monday_snow = 0.32) (h2 : total_snow = 0.53) :
  total_snow - monday_snow = 0.21 := by
  sorry

end snow_on_tuesday_l2268_226875


namespace triangle_side_length_l2268_226864

/-- In a triangle ABC, given specific angle and side length conditions, prove the length of side b. -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  C = 4 * A ∧  -- Given angle condition
  a = 20 ∧  -- Given side length
  c = 40 ∧  -- Given side length
  a / Real.sin A = b / Real.sin B ∧  -- Law of Sines
  a / Real.sin A = c / Real.sin C  -- Law of Sines
  →
  b = 20 * (16 * (9 * Real.sqrt 3 / 16) - 20 * (3 * Real.sqrt 3 / 4) + 5 * Real.sqrt 3) :=
by sorry

end triangle_side_length_l2268_226864


namespace right_triangle_median_hypotenuse_l2268_226845

/-- 
A right triangle with hypotenuse length 6 has a median to the hypotenuse of length 3.
-/
theorem right_triangle_median_hypotenuse : 
  ∀ (a b c : ℝ), 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem for right triangle
  c = 6 →           -- Hypotenuse length is 6
  ∃ (m : ℝ),        -- There exists a median m
    m^2 = (a^2 + b^2) / 4 ∧  -- Median formula
    m = 3 :=        -- Median length is 3
by sorry

end right_triangle_median_hypotenuse_l2268_226845


namespace triangle_area_l2268_226842

/-- The area of a triangle with vertices at (3, -3), (3, 4), and (8, -3) is 17.5 square units -/
theorem triangle_area : 
  let v1 : ℝ × ℝ := (3, -3)
  let v2 : ℝ × ℝ := (3, 4)
  let v3 : ℝ × ℝ := (8, -3)
  let area := abs ((v1.1 * (v2.2 - v3.2) + v2.1 * (v3.2 - v1.2) + v3.1 * (v1.2 - v2.2)) / 2)
  area = 17.5 := by
sorry


end triangle_area_l2268_226842


namespace hyperbola_m_range_l2268_226829

theorem hyperbola_m_range (m : ℝ) : 
  (∀ x y : ℝ, x^2 / m - y^2 / (2*m - 1) = 1) → 
  (0 < m ∧ m < 1/2) :=
sorry

end hyperbola_m_range_l2268_226829


namespace vector_properties_l2268_226831

/-- Given two vectors a and b in ℝ³, prove properties about their components --/
theorem vector_properties (a b : ℝ × ℝ × ℝ) :
  let x := a.2.2
  let y := b.2.1
  (a = (2, 4, x) ∧ ‖a‖ = 6) →
  (x = 4 ∨ x = -4) ∧
  (a = (2, 4, x) ∧ b = (2, y, 2) ∧ ∃ (k : ℝ), a = k • b) →
  x + y = 6 := by sorry

end vector_properties_l2268_226831


namespace tree_height_problem_l2268_226870

theorem tree_height_problem (h₁ h₂ : ℝ) : 
  h₂ = h₁ + 20 →  -- One tree is 20 feet taller than the other
  h₁ / h₂ = 5 / 7 →  -- The heights are in the ratio 5:7
  h₁ = 50  -- The shorter tree is 50 feet tall
:= by sorry

end tree_height_problem_l2268_226870


namespace cliff_rock_collection_l2268_226887

theorem cliff_rock_collection :
  let total_rocks : ℕ := 180
  let sedimentary_rocks : ℕ := total_rocks * 2 / 3
  let igneous_rocks : ℕ := sedimentary_rocks / 2
  let shiny_igneous_ratio : ℚ := 2 / 3
  shiny_igneous_ratio * igneous_rocks = 40 := by
  sorry

end cliff_rock_collection_l2268_226887


namespace parallelogram_area_l2268_226830

theorem parallelogram_area (base : ℝ) (slant_height : ℝ) (angle : ℝ) :
  base = 10 →
  slant_height = 6 →
  angle = 30 * π / 180 →
  base * (slant_height * Real.sin angle) = 30 := by
  sorry

end parallelogram_area_l2268_226830


namespace exponential_function_fixed_point_l2268_226846

theorem exponential_function_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^x + 3
  f 0 = 4 := by sorry

end exponential_function_fixed_point_l2268_226846


namespace line_vector_coefficient_l2268_226882

/-- Given vectors a and b in a real vector space, if k*a + (2/5)*b lies on the line
    passing through a and b, then k = 3/5 -/
theorem line_vector_coefficient (V : Type*) [NormedAddCommGroup V] [NormedSpace ℝ V]
  (a b : V) (k : ℝ) :
  (∃ t : ℝ, k • a + (2/5) • b = a + t • (b - a)) →
  k = 3/5 := by
  sorry

end line_vector_coefficient_l2268_226882


namespace jenny_sleep_duration_l2268_226836

/-- Calculates the total minutes of sleep given the number of hours and minutes per hour. -/
def total_minutes_of_sleep (hours : ℕ) (minutes_per_hour : ℕ) : ℕ :=
  hours * minutes_per_hour

/-- Proves that 8 hours of sleep is equivalent to 480 minutes. -/
theorem jenny_sleep_duration :
  total_minutes_of_sleep 8 60 = 480 := by
  sorry

end jenny_sleep_duration_l2268_226836


namespace shortest_player_height_l2268_226817

theorem shortest_player_height (tallest_height : Float) (height_difference : Float) :
  tallest_height = 77.75 →
  height_difference = 9.5 →
  tallest_height - height_difference = 68.25 := by
  sorry

end shortest_player_height_l2268_226817


namespace BI_length_is_15_over_4_l2268_226813

/-- Two squares ABCD and EFGH with parallel sides -/
structure ParallelSquares :=
  (A B C D E F G H : ℝ × ℝ)

/-- Point where CG intersects BD -/
def I (squares : ParallelSquares) : ℝ × ℝ := sorry

/-- Length of BD -/
def BD_length (squares : ParallelSquares) : ℝ := 10

/-- Area of triangle BFC -/
def area_BFC (squares : ParallelSquares) : ℝ := 3

/-- Area of triangle CHD -/
def area_CHD (squares : ParallelSquares) : ℝ := 5

/-- Length of BI -/
def BI_length (squares : ParallelSquares) : ℝ := sorry

theorem BI_length_is_15_over_4 (squares : ParallelSquares) :
  BI_length squares = 15 / 4 := by sorry

end BI_length_is_15_over_4_l2268_226813


namespace min_value_quadratic_l2268_226819

theorem min_value_quadratic (x : ℝ) : 
  ∃ (y_min : ℝ), ∀ (y : ℝ), y = 5*x^2 + 20*x + 45 → y ≥ y_min ∧ y_min = 25 := by
  sorry

end min_value_quadratic_l2268_226819


namespace common_chord_of_circles_l2268_226867

-- Define the equations of the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 2 = 0

-- Define the equation of the common chord
def common_chord (x y : ℝ) : Prop := x + 2*y - 1 = 0

-- Theorem statement
theorem common_chord_of_circles :
  ∀ x y : ℝ, C₁ x y ∧ C₂ x y → common_chord x y :=
by sorry

end common_chord_of_circles_l2268_226867


namespace valid_paths_count_l2268_226898

/-- Represents the number of paths on a complete 9x3 grid -/
def total_paths : ℕ := 220

/-- Represents the number of paths through each forbidden segment -/
def forbidden_segment_paths : ℕ := 70

/-- Represents the number of forbidden segments -/
def num_forbidden_segments : ℕ := 2

/-- Theorem stating the number of valid paths on the grid with forbidden segments -/
theorem valid_paths_count : 
  total_paths - (forbidden_segment_paths * num_forbidden_segments) = 80 := by
  sorry

end valid_paths_count_l2268_226898


namespace quadratic_function_properties_l2268_226856

/-- Quadratic function -/
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

theorem quadratic_function_properties (b c : ℝ) :
  (∀ x, f b c x ≥ f b c 1) →  -- minimum at x = 1
  f b c 1 = 3 →              -- minimum value is 3
  f b c 2 = 4 →              -- f(2) = 4
  b = -2 ∧ c = 4 :=
by sorry

end quadratic_function_properties_l2268_226856


namespace polygon_area_l2268_226876

/-- A polygon on a unit grid with vertices at (0,0), (5,0), (5,5), (0,5), (5,10), (0,10), (0,0) -/
def polygon : List (ℤ × ℤ) := [(0,0), (5,0), (5,5), (0,5), (5,10), (0,10), (0,0)]

/-- The area enclosed by the polygon -/
def enclosed_area (p : List (ℤ × ℤ)) : ℚ := sorry

/-- Theorem stating that the area enclosed by the polygon is 37.5 square units -/
theorem polygon_area : enclosed_area polygon = 37.5 := by sorry

end polygon_area_l2268_226876


namespace ones_digit_of_6_power_52_l2268_226800

theorem ones_digit_of_6_power_52 : ∃ n : ℕ, 6^52 = 10 * n + 6 :=
sorry

end ones_digit_of_6_power_52_l2268_226800
