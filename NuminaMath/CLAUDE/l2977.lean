import Mathlib

namespace committee_probability_l2977_297709

/-- The probability of selecting exactly 2 boys in a 5-person committee
    chosen randomly from a group of 30 members (12 boys and 18 girls) -/
theorem committee_probability (total : Nat) (boys : Nat) (girls : Nat) (committee_size : Nat) :
  total = 30 →
  boys = 12 →
  girls = 18 →
  committee_size = 5 →
  (Nat.choose boys 2 * Nat.choose girls 3 : ℚ) / Nat.choose total committee_size = 26928 / 71253 := by
  sorry

end committee_probability_l2977_297709


namespace planes_intersect_necessary_not_sufficient_for_skew_lines_l2977_297784

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp : Line → Plane → Prop)

-- Define the intersection relation between two planes
variable (intersect : Plane → Plane → Prop)

-- Define the skew relation between two lines
variable (skew : Line → Line → Prop)

-- Define the parallel relation between two planes
variable (parallel : Plane → Plane → Prop)

theorem planes_intersect_necessary_not_sufficient_for_skew_lines
  (α β : Plane) (m n : Line)
  (h_distinct : α ≠ β)
  (h_perp_m : perp m α)
  (h_perp_n : perp n β) :
  (∀ α β m n, skew m n → intersect α β) ∧
  (∃ α β m n, intersect α β ∧ perp m α ∧ perp n β ∧ ¬skew m n) :=
sorry

end planes_intersect_necessary_not_sufficient_for_skew_lines_l2977_297784


namespace sum_of_squares_l2977_297747

/-- Given a matrix N with the specified structure, prove that if N^T N = I, then x^2 + y^2 + z^2 = 47/120 -/
theorem sum_of_squares (x y z : ℝ) : 
  let N : Matrix (Fin 3) (Fin 3) ℝ := !![0, 3*y, 2*z; 2*x, y, -z; 2*x, -y, z]
  (N.transpose * N = 1) → x^2 + y^2 + z^2 = 47/120 := by
  sorry

end sum_of_squares_l2977_297747


namespace score_difference_is_negative_1_75_l2977_297794

def score_distribution : List (Float × Float) := [
  (0.15, 80),
  (0.40, 90),
  (0.25, 95),
  (0.20, 100)
]

def median (dist : List (Float × Float)) : Float :=
  90  -- The median is 90 as per the problem description

def mean (dist : List (Float × Float)) : Float :=
  dist.foldr (λ (p, s) acc => acc + p * s) 0

theorem score_difference_is_negative_1_75 :
  median score_distribution - mean score_distribution = -1.75 := by
  sorry

#eval median score_distribution - mean score_distribution

end score_difference_is_negative_1_75_l2977_297794


namespace identity_proof_l2977_297746

theorem identity_proof (x : ℝ) : (2*x - 1)^3 = 5*x^3 + (3*x + 1)*(x^2 - x - 1) - 10*x^2 + 10*x := by
  sorry

end identity_proof_l2977_297746


namespace percentage_b_grades_l2977_297756

def scores : List Nat := [92, 81, 68, 88, 82, 63, 79, 70, 85, 99, 59, 67, 84, 90, 75, 61, 87, 65, 86]

def is_b_grade (score : Nat) : Bool :=
  80 ≤ score ∧ score ≤ 84

def count_b_grades (scores : List Nat) : Nat :=
  scores.filter is_b_grade |>.length

theorem percentage_b_grades : 
  (count_b_grades scores : Rat) / (scores.length : Rat) * 100 = 15 := by
  sorry

end percentage_b_grades_l2977_297756


namespace original_light_wattage_l2977_297786

/-- Given a new light with 25% higher wattage than the original light,
    proves that if the new light has 100 watts, then the original light had 80 watts. -/
theorem original_light_wattage (new_wattage : ℝ) (h1 : new_wattage = 100) :
  let original_wattage := new_wattage / 1.25
  original_wattage = 80 := by
sorry

end original_light_wattage_l2977_297786


namespace inscribed_circle_radius_right_triangle_l2977_297721

/-- The radius of the inscribed circle in a right-angled triangle -/
theorem inscribed_circle_radius_right_triangle (a b c r : ℝ) 
  (h_right_angle : a^2 + b^2 = c^2) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) : 
  r = (a * b) / (a + b + c) :=
sorry

end inscribed_circle_radius_right_triangle_l2977_297721


namespace quadratic_equation_root_zero_l2977_297707

theorem quadratic_equation_root_zero (m : ℝ) : 
  (m - 1 ≠ 0) →
  (∃ x : ℝ, (m - 1) * x^2 + 2 * x + m^2 - 1 = 0) →
  ((m - 1) * 0^2 + 2 * 0 + m^2 - 1 = 0) →
  m = -1 := by
sorry

end quadratic_equation_root_zero_l2977_297707


namespace product_of_numbers_with_given_sum_and_difference_l2977_297719

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 23 ∧ x - y = 7 → x * y = 120 := by
  sorry

end product_of_numbers_with_given_sum_and_difference_l2977_297719


namespace triangle_tangent_difference_l2977_297773

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a² = b² + 4bc sin A and tan A · tan B = 2, then tan B - tan A = -8 -/
theorem triangle_tangent_difference (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a^2 = b^2 + 4*b*c*(Real.sin A) →
  Real.tan A * Real.tan B = 2 →
  Real.tan B - Real.tan A = -8 := by
sorry

end triangle_tangent_difference_l2977_297773


namespace line_intersects_curve_l2977_297726

/-- Given real numbers a and b where ab ≠ 0, the line ax - y + b = 0 intersects
    the curve bx² + ay² = ab. -/
theorem line_intersects_curve (a b : ℝ) (h : a * b ≠ 0) :
  ∃ (x y : ℝ), (a * x - y + b = 0) ∧ (b * x^2 + a * y^2 = a * b) := by
  sorry

end line_intersects_curve_l2977_297726


namespace fifth_store_cars_l2977_297737

def store_count : Nat := 5
def car_counts : Vector Nat 4 := ⟨[30, 14, 14, 21], rfl⟩
def mean : Rat := 104/5

theorem fifth_store_cars : 
  ∃ x : Nat, (car_counts.toList.sum + x) / store_count = mean :=
by
  sorry

end fifth_store_cars_l2977_297737


namespace arithmetic_sequence_a12_l2977_297767

/-- An arithmetic sequence with specific conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  (a 7 + a 9 = 16) ∧
  (a 4 = 4)

/-- Theorem: For the given arithmetic sequence, a_12 = 12 -/
theorem arithmetic_sequence_a12 (a : ℕ → ℝ) (h : ArithmeticSequence a) : 
  a 12 = 12 := by
  sorry

end arithmetic_sequence_a12_l2977_297767


namespace x_value_l2977_297750

theorem x_value (x y : ℝ) (hx : x ≠ 0) (h1 : x/3 = y^2) (h2 : x/6 = 3*y) : x = 108 := by
  sorry

end x_value_l2977_297750


namespace city_mpg_calculation_l2977_297729

/-- The average miles per gallon (mpg) for an SUV in the city -/
def city_mpg : ℝ := 12.2

/-- The maximum distance in miles that the SUV can travel on 25 gallons of gasoline -/
def max_distance : ℝ := 305

/-- The amount of gasoline in gallons used for the maximum distance -/
def gasoline_amount : ℝ := 25

/-- Theorem stating that the average mpg in the city is 12.2 -/
theorem city_mpg_calculation : city_mpg = max_distance / gasoline_amount := by
  sorry

end city_mpg_calculation_l2977_297729


namespace balls_in_boxes_l2977_297700

theorem balls_in_boxes (x y z : ℕ) : 
  x + y + z = 320 →
  x > 0 ∧ y > 0 ∧ z > 0 →
  ∃ (a b c : ℕ), a ≤ x ∧ b ≤ y ∧ c ≤ z ∧ 6*a + 11*b + 15*c = 1001 :=
by sorry

end balls_in_boxes_l2977_297700


namespace rachels_homework_l2977_297799

/-- Rachel's homework problem -/
theorem rachels_homework (reading_pages : ℕ) (math_pages : ℕ) : 
  reading_pages = 2 → math_pages = reading_pages + 3 → math_pages = 5 :=
by sorry

end rachels_homework_l2977_297799


namespace divide_decimals_l2977_297706

theorem divide_decimals : (0.25 : ℚ) / (0.005 : ℚ) = 50 := by
  sorry

end divide_decimals_l2977_297706


namespace connie_marbles_problem_l2977_297780

/-- Proves that Connie started with 143 marbles given the conditions of the problem -/
theorem connie_marbles_problem :
  ∀ (initial : ℕ),
  initial - 73 = 70 →
  initial = 143 :=
by
  sorry

end connie_marbles_problem_l2977_297780


namespace average_of_remaining_numbers_l2977_297701

theorem average_of_remaining_numbers
  (total : ℕ)
  (avg_all : ℚ)
  (count1 : ℕ)
  (avg1 : ℚ)
  (count2 : ℕ)
  (avg2 : ℚ)
  (h_total : total = 6)
  (h_avg_all : avg_all = 3.95)
  (h_count1 : count1 = 2)
  (h_avg1 : avg1 = 4.4)
  (h_count2 : count2 = 2)
  (h_avg2 : avg2 = 3.85) :
  let sum_all := total * avg_all
  let sum1 := count1 * avg1
  let sum2 := count2 * avg2
  let remaining := total - count1 - count2
  let sum_remaining := sum_all - sum1 - sum2
  (sum_remaining / remaining : ℚ) = 3.6 := by
sorry

end average_of_remaining_numbers_l2977_297701


namespace quadratic_ratio_l2977_297781

/-- Given a quadratic function f(x) = x^2 + 1500x + 1500, 
    prove that when expressed as (x + b)^2 + c, 
    the ratio c/b equals -748 -/
theorem quadratic_ratio (f : ℝ → ℝ) (b c : ℝ) : 
  (∀ x, f x = x^2 + 1500*x + 1500) → 
  (∀ x, f x = (x + b)^2 + c) → 
  c / b = -748 := by
sorry

end quadratic_ratio_l2977_297781


namespace min_value_product_l2977_297779

theorem min_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  (3 * x + y) * (x + 3 * z) * (y + z + 1) ≥ 48 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 1 ∧
    (3 * x₀ + y₀) * (x₀ + 3 * z₀) * (y₀ + z₀ + 1) = 48 :=
by sorry

end min_value_product_l2977_297779


namespace kyler_won_zero_games_l2977_297757

/-- Represents a player in the chess tournament -/
inductive Player : Type
| Peter : Player
| Emma : Player
| Kyler : Player

/-- Represents the number of games won by each player -/
def games_won (p : Player) : ℕ :=
  match p with
  | Player.Peter => 5
  | Player.Emma => 4
  | Player.Kyler => 0  -- This is what we want to prove

/-- Represents the number of games lost by each player -/
def games_lost (p : Player) : ℕ :=
  match p with
  | Player.Peter => 3
  | Player.Emma => 4
  | Player.Kyler => 4

/-- The total number of games played in the tournament -/
def total_games : ℕ := (games_won Player.Peter + games_won Player.Emma + games_won Player.Kyler +
                        games_lost Player.Peter + games_lost Player.Emma + games_lost Player.Kyler) / 2

theorem kyler_won_zero_games :
  games_won Player.Kyler = 0 ∧
  2 * total_games = games_won Player.Peter + games_won Player.Emma + games_won Player.Kyler +
                    games_lost Player.Peter + games_lost Player.Emma + games_lost Player.Kyler :=
by sorry

end kyler_won_zero_games_l2977_297757


namespace mango_jelly_dishes_l2977_297789

theorem mango_jelly_dishes (total_dishes : ℕ) 
  (mango_salsa_dishes : ℕ) (fresh_mango_dishes : ℕ) 
  (oliver_pickout_dishes : ℕ) (oliver_left_dishes : ℕ) :
  total_dishes = 36 →
  mango_salsa_dishes = 3 →
  fresh_mango_dishes = total_dishes / 6 →
  oliver_pickout_dishes = 2 →
  oliver_left_dishes = 28 →
  total_dishes - oliver_left_dishes - (mango_salsa_dishes + (fresh_mango_dishes - oliver_pickout_dishes)) = 1 :=
by
  sorry

#check mango_jelly_dishes

end mango_jelly_dishes_l2977_297789


namespace partial_fraction_sum_l2977_297724

theorem partial_fraction_sum (p q r A B C : ℝ) : 
  p ≠ q ∧ q ≠ r ∧ p ≠ r →
  (∀ x : ℝ, x^3 - 20*x^2 + 125*x - 500 = (x - p)*(x - q)*(x - r)) →
  (∀ s : ℝ, s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 20*s^2 + 125*s - 500) = A / (s - p) + B / (s - q) + C / (s - r)) →
  1 / A + 1 / B + 1 / C = 720 := by
sorry

end partial_fraction_sum_l2977_297724


namespace function_decomposition_l2977_297704

/-- A function is α-periodic if f(x + α) = f(x) for all x -/
def Periodic (f : ℝ → ℝ) (α : ℝ) : Prop :=
  ∀ x, f (x + α) = f x

/-- A function is linear if f(x) = ax for some constant a -/
def Linear (f : ℝ → ℝ) : Prop :=
  ∃ a, ∀ x, f x = a * x

theorem function_decomposition (f : ℝ → ℝ) (α β : ℝ) (hα : α ≠ 0)
    (h : ∀ x, f (x + α) = f x + β) :
    ∃ (g h : ℝ → ℝ), Periodic g α ∧ Linear h ∧ ∀ x, f x = g x + h x := by
  sorry

end function_decomposition_l2977_297704


namespace inequality_proof_l2977_297714

theorem inequality_proof (a b c : ℝ) 
  (ha : -1 < a ∧ a < -2/3) 
  (hb : -1/3 < b ∧ b < 0) 
  (hc : c > 1) : 
  1/c < 1/(b-a) ∧ 1/(b-a) < 1/(a*b) := by
sorry

end inequality_proof_l2977_297714


namespace condition_relation_l2977_297758

theorem condition_relation (p q : Prop) 
  (h : (p → ¬q) ∧ ¬(¬q → p)) : 
  (q → ¬p) ∧ ¬(¬p → q) := by
  sorry

end condition_relation_l2977_297758


namespace expand_cube_105_plus_1_l2977_297791

theorem expand_cube_105_plus_1 : 105^3 + 3*(105^2) + 3*105 + 1 = 11856 := by
  sorry

end expand_cube_105_plus_1_l2977_297791


namespace line_circle_intersection_l2977_297720

/-- The circle equation x^2 + y^2 - 4x - 2y + 1 = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y + 1 = 0

/-- The line equation ax + y - 5 = 0 -/
def line_equation (a x y : ℝ) : Prop :=
  a*x + y - 5 = 0

/-- The chord length of the intersection is 4 -/
def chord_length_is_4 (a : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_equation x₁ y₁ ∧ circle_equation x₂ y₂ ∧
    line_equation a x₁ y₁ ∧ line_equation a x₂ y₂ ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = 4^2

theorem line_circle_intersection (a : ℝ) :
  chord_length_is_4 a → a = 2 := by
  sorry

end line_circle_intersection_l2977_297720


namespace only_frustum_has_two_parallel_surfaces_l2977_297764

-- Define the geometric bodies
inductive GeometricBody
| Pyramid
| Prism
| Frustum
| Cuboid

-- Define a function to count parallel surfaces
def parallelSurfaces : GeometricBody → ℕ
| GeometricBody.Pyramid => 0
| GeometricBody.Prism => 6
| GeometricBody.Frustum => 2
| GeometricBody.Cuboid => 6

-- Theorem: Only the frustum has exactly two parallel surfaces
theorem only_frustum_has_two_parallel_surfaces :
  ∀ b : GeometricBody, parallelSurfaces b = 2 ↔ b = GeometricBody.Frustum :=
by sorry

end only_frustum_has_two_parallel_surfaces_l2977_297764


namespace largest_number_with_property_l2977_297770

/-- A function that returns true if a natural number has all distinct digits --/
def has_distinct_digits (n : ℕ) : Prop := sorry

/-- A function that returns true if a number is not divisible by 11 --/
def not_divisible_by_11 (n : ℕ) : Prop := sorry

/-- A function that returns true if all subsequences of digits in a number are not divisible by 11 --/
def all_subsequences_not_divisible_by_11 (n : ℕ) : Prop := sorry

/-- The main theorem stating that 987654321 is the largest natural number 
    with all distinct digits and all subsequences not divisible by 11 --/
theorem largest_number_with_property : 
  ∀ n : ℕ, n > 987654321 → 
  ¬(has_distinct_digits n ∧ all_subsequences_not_divisible_by_11 n) :=
sorry

end largest_number_with_property_l2977_297770


namespace mini_marshmallows_count_l2977_297752

/-- Calculates the number of mini marshmallows used in a recipe --/
def mini_marshmallows_used (total_marshmallows : ℕ) (large_marshmallows : ℕ) : ℕ :=
  total_marshmallows - large_marshmallows

/-- Proves that the number of mini marshmallows used is correct --/
theorem mini_marshmallows_count 
  (total_marshmallows : ℕ) 
  (large_marshmallows : ℕ) 
  (h : large_marshmallows ≤ total_marshmallows) :
  mini_marshmallows_used total_marshmallows large_marshmallows = 
    total_marshmallows - large_marshmallows :=
by
  sorry

#eval mini_marshmallows_used 18 8  -- Should output 10

end mini_marshmallows_count_l2977_297752


namespace largest_square_area_l2977_297732

theorem largest_square_area (side_length : ℝ) (corner_size : ℝ) : 
  side_length = 5 → 
  corner_size = 1 → 
  (side_length - 2 * corner_size)^2 = 9 := by
  sorry

end largest_square_area_l2977_297732


namespace job_completion_time_specific_job_completion_time_l2977_297740

/-- The time taken to complete a job when three people work together, given their individual completion times. -/
theorem job_completion_time (t1 t2 t3 : ℝ) (h1 : t1 > 0) (h2 : t2 > 0) (h3 : t3 > 0) :
  1 / (1 / t1 + 1 / t2 + 1 / t3) = (t1 * t2 * t3) / (t2 * t3 + t1 * t3 + t1 * t2) :=
by sorry

/-- The specific case of the job completion time for the given problem. -/
theorem specific_job_completion_time :
  1 / (1 / 15 + 1 / 20 + 1 / 25 : ℝ) = 300 / 47 :=
by sorry

end job_completion_time_specific_job_completion_time_l2977_297740


namespace anime_watching_problem_l2977_297777

/-- The number of days from today to April 1, 2023 (exclusive) -/
def days_to_april_1 : ℕ := sorry

/-- The total number of episodes in the anime series -/
def total_episodes : ℕ := sorry

/-- Theorem stating the solution to the anime watching problem -/
theorem anime_watching_problem :
  (total_episodes - 2 * days_to_april_1 = 215) ∧
  (total_episodes - 5 * days_to_april_1 = 50) →
  (days_to_april_1 = 55 ∧ total_episodes = 325) :=
by sorry

end anime_watching_problem_l2977_297777


namespace papa_worms_correct_l2977_297722

/-- The number of worms Papa bird caught -/
def papa_worms (babies : ℕ) (worms_per_baby_per_day : ℕ) (days : ℕ) 
  (mama_caught : ℕ) (stolen : ℕ) (mama_needs : ℕ) : ℕ :=
  babies * worms_per_baby_per_day * days - ((mama_caught - stolen) + mama_needs)

theorem papa_worms_correct : 
  papa_worms 6 3 3 13 2 34 = 9 := by
  sorry

end papa_worms_correct_l2977_297722


namespace quadratic_completing_square_l2977_297793

theorem quadratic_completing_square (x : ℝ) : 
  (4 * x^2 - 24 * x - 96 = 0) → 
  ∃ q t : ℝ, ((x + q)^2 = t) ∧ (t = 33) := by
sorry

end quadratic_completing_square_l2977_297793


namespace sufficient_not_necessary_l2977_297763

theorem sufficient_not_necessary (x y : ℝ) :
  (∀ x y : ℝ, x > 2 ∧ y > 2 → x + y > 4) ∧
  (∃ x y : ℝ, x + y > 4 ∧ ¬(x > 2 ∧ y > 2)) :=
by sorry

end sufficient_not_necessary_l2977_297763


namespace total_jeans_purchased_l2977_297776

/-- Represents the number of pairs of Fox jeans purchased -/
def fox_jeans : ℕ := 3

/-- Represents the number of pairs of Pony jeans purchased -/
def pony_jeans : ℕ := 2

/-- Regular price of Fox jeans in dollars -/
def fox_price : ℚ := 15

/-- Regular price of Pony jeans in dollars -/
def pony_price : ℚ := 20

/-- Total discount in dollars -/
def total_discount : ℚ := 9

/-- Sum of discount rates as a percentage -/
def sum_discount_rates : ℚ := 22

/-- Discount rate on Pony jeans as a percentage -/
def pony_discount_rate : ℚ := 18.000000000000014

/-- Theorem stating the total number of jeans purchased -/
theorem total_jeans_purchased : fox_jeans + pony_jeans = 5 := by
  sorry

end total_jeans_purchased_l2977_297776


namespace orchid_bushes_total_l2977_297774

theorem orchid_bushes_total (current : ℕ) (today : ℕ) (tomorrow : ℕ) :
  current = 47 → today = 37 → tomorrow = 25 →
  current + today + tomorrow = 109 := by
  sorry

end orchid_bushes_total_l2977_297774


namespace circles_externally_tangent_l2977_297734

/-- Two circles are externally tangent if the distance between their centers
    is equal to the sum of their radii -/
def externally_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  Real.sqrt ((c1.1 - c2.1)^2 + (c1.2 - c2.2)^2) = r1 + r2

/-- The theorem stating that the two given circles are externally tangent -/
theorem circles_externally_tangent :
  let c1 : ℝ × ℝ := (0, 8)
  let c2 : ℝ × ℝ := (-6, 0)
  let r1 : ℝ := 6
  let r2 : ℝ := 2
  externally_tangent c1 c2 r1 r2 := by
  sorry

end circles_externally_tangent_l2977_297734


namespace broken_line_coverage_coin_covers_broken_line_l2977_297705

/-- A closed broken line in a 2D plane -/
structure ClosedBrokenLine where
  points : Set (ℝ × ℝ)
  is_closed : True  -- Placeholder for the closed property
  length : ℝ

/-- Theorem: Any closed broken line of length 5 can be covered by a circle of radius 1.25 -/
theorem broken_line_coverage (L : ClosedBrokenLine) (h : L.length = 5) :
  ∃ (center : ℝ × ℝ), ∀ (p : ℝ × ℝ), p ∈ L.points → dist center p ≤ 1.25 := by
  sorry

/-- Corollary: A coin with diameter > 2.5 can cover a 5 cm closed broken line -/
theorem coin_covers_broken_line (L : ClosedBrokenLine) (h : L.length = 5) 
  (coin_diameter : ℝ) (hd : coin_diameter > 2.5) :
  ∃ (center : ℝ × ℝ), ∀ (p : ℝ × ℝ), p ∈ L.points → dist center p ≤ coin_diameter / 2 := by
  sorry

end broken_line_coverage_coin_covers_broken_line_l2977_297705


namespace correct_result_largest_negative_integer_result_l2977_297738

/-- Given polynomial A -/
def A (x : ℝ) : ℝ := 3 * x^2 - x + 1

/-- Given polynomial B -/
def B (x : ℝ) : ℝ := -x^2 - 2*x - 3

/-- Theorem stating the correct result of A - B -/
theorem correct_result (x : ℝ) : A x - B x = 4 * x^2 + x + 4 := by sorry

/-- Theorem stating the value of A - B when x is the largest negative integer -/
theorem largest_negative_integer_result : A (-1) - B (-1) = 7 := by sorry

end correct_result_largest_negative_integer_result_l2977_297738


namespace division_problem_l2977_297753

theorem division_problem (dividend : Nat) (divisor : Nat) (remainder : Nat) (quotient : Nat) :
  dividend = 172 →
  divisor = 17 →
  remainder = 2 →
  dividend = divisor * quotient + remainder →
  quotient = 10 := by
  sorry

end division_problem_l2977_297753


namespace probability_ratio_is_twenty_l2977_297798

def total_balls : ℕ := 25
def num_bins : ℕ := 6

def distribution_A : List ℕ := [4, 4, 4, 5, 5, 2]
def distribution_B : List ℕ := [5, 5, 5, 5, 5, 0]

def probability_ratio : ℚ :=
  (Nat.choose num_bins 3 * Nat.choose 3 2 * Nat.choose 1 1 *
   (Nat.factorial total_balls / (Nat.factorial 4 * Nat.factorial 4 * Nat.factorial 4 * Nat.factorial 5 * Nat.factorial 5 * Nat.factorial 2))) /
  (Nat.choose num_bins 5 * Nat.choose 1 1 *
   (Nat.factorial total_balls / (Nat.factorial 5 * Nat.factorial 5 * Nat.factorial 5 * Nat.factorial 5 * Nat.factorial 5 * Nat.factorial 0)))

theorem probability_ratio_is_twenty :
  probability_ratio = 20 := by sorry

end probability_ratio_is_twenty_l2977_297798


namespace smallest_solution_floor_equation_l2977_297782

theorem smallest_solution_floor_equation :
  ∀ x : ℝ, (⌊x⌋ : ℝ) = 7 + 50 * (x - ⌊x⌋) → x ≥ 7 :=
by sorry

end smallest_solution_floor_equation_l2977_297782


namespace area_covered_by_strips_l2977_297783

/-- Represents a rectangular strip -/
structure Strip where
  length : ℝ
  width : ℝ

/-- Calculates the area of a strip -/
def stripArea (s : Strip) : ℝ := s.length * s.width

/-- Calculates the total area of strips without considering overlaps -/
def totalAreaNoOverlap (strips : List Strip) : ℝ :=
  (strips.map stripArea).sum

/-- Represents an overlap between two strips -/
structure Overlap where
  length : ℝ
  width : ℝ

/-- Calculates the area of an overlap -/
def overlapArea (o : Overlap) : ℝ := o.length * o.width

/-- Calculates the total area of overlaps -/
def totalOverlapArea (overlaps : List Overlap) : ℝ :=
  (overlaps.map overlapArea).sum

/-- Theorem: The area covered by five strips with given dimensions and overlaps is 58 -/
theorem area_covered_by_strips :
  let strips : List Strip := List.replicate 5 ⟨12, 1⟩
  let overlaps : List Overlap := List.replicate 4 ⟨0.5, 1⟩
  totalAreaNoOverlap strips - totalOverlapArea overlaps = 58 := by
  sorry

end area_covered_by_strips_l2977_297783


namespace change_in_cubic_expression_l2977_297796

theorem change_in_cubic_expression (x a : ℝ) (ha : a > 0) :
  abs ((x + a)^3 - 3*(x + a) - (x^3 - 3*x)) = 3*a*x^2 + 3*a^2*x + a^3 - 3*a ∧
  abs ((x - a)^3 - 3*(x - a) - (x^3 - 3*x)) = 3*a*x^2 + 3*a^2*x + a^3 - 3*a :=
by sorry

end change_in_cubic_expression_l2977_297796


namespace decimal_units_count_l2977_297769

theorem decimal_units_count :
  (∃ n : ℕ, n * (1 / 10 : ℚ) = (19 / 10 : ℚ) ∧ n = 19) ∧
  (∃ m : ℕ, m * (1 / 100 : ℚ) = (8 / 10 : ℚ) ∧ m = 80) :=
by sorry

end decimal_units_count_l2977_297769


namespace constant_term_binomial_expansion_l2977_297713

theorem constant_term_binomial_expansion :
  ∀ n : ℕ, n > 0 →
  ∃ k : ℕ, k > 0 ∧ k ≤ n + 1 ∧
  (∀ r : ℕ, r ≥ 0 ∧ r ≤ n →
    (Nat.choose n r * (1 : ℚ)) = 0 ∨ (2 * r = n → k = r + 1)) →
  k = 6 ∧ n = 10 :=
by sorry

end constant_term_binomial_expansion_l2977_297713


namespace complex_number_calculation_l2977_297790

/-- Given the complex number i where i^2 = -1, prove that (1+i)(1-i)+(-1+i) = 1+i -/
theorem complex_number_calculation : ∀ i : ℂ, i^2 = -1 → (1+i)*(1-i)+(-1+i) = 1+i := by
  sorry

end complex_number_calculation_l2977_297790


namespace baseball_cleats_price_l2977_297772

/-- Proves that the price of each pair of baseball cleats is $10 -/
theorem baseball_cleats_price :
  let cards_price : ℝ := 25
  let bat_price : ℝ := 10
  let glove_original_price : ℝ := 30
  let glove_discount_percentage : ℝ := 0.2
  let total_sales : ℝ := 79
  let num_cleats_pairs : ℕ := 2

  let glove_sale_price : ℝ := glove_original_price * (1 - glove_discount_percentage)
  let non_cleats_sales : ℝ := cards_price + bat_price + glove_sale_price
  let cleats_total_price : ℝ := total_sales - non_cleats_sales
  let cleats_pair_price : ℝ := cleats_total_price / num_cleats_pairs

  cleats_pair_price = 10 := by
    sorry

end baseball_cleats_price_l2977_297772


namespace quadratic_root_implies_k_l2977_297749

/-- If the quadratic equation x^2 - 3x + 2k = 0 has a root of 1, then k = 1 -/
theorem quadratic_root_implies_k (k : ℝ) : 
  (∃ x : ℝ, x^2 - 3*x + 2*k = 0) ∧ (1^2 - 3*1 + 2*k = 0) → k = 1 :=
by sorry

end quadratic_root_implies_k_l2977_297749


namespace choose_three_from_fifteen_l2977_297739

theorem choose_three_from_fifteen (n k : ℕ) : n = 15 ∧ k = 3 → Nat.choose n k = 455 := by
  sorry

end choose_three_from_fifteen_l2977_297739


namespace horner_method_innermost_polynomial_l2977_297716

def f (x : ℝ) : ℝ := x^5 + x^4 + x^3 + x^2 + x + 1

def horner_v1 (a : ℝ) : ℝ := 1 * a + 1

theorem horner_method_innermost_polynomial :
  horner_v1 3 = 4 :=
by sorry

end horner_method_innermost_polynomial_l2977_297716


namespace rice_yield_80kg_l2977_297771

/-- Linear regression equation for rice yield prediction -/
def rice_yield_prediction (x : ℝ) : ℝ := 5 * x + 250

/-- Theorem: The predicted rice yield for 80 kg of fertilizer is 650 kg -/
theorem rice_yield_80kg : rice_yield_prediction 80 = 650 := by
  sorry

end rice_yield_80kg_l2977_297771


namespace quadratic_equation_two_roots_l2977_297736

-- Define the geometric progression
def is_geometric_progression (a b c : ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ b = a * q ∧ c = a * q^2

-- Define the quadratic equation
def has_two_distinct_roots (a b c : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + 2 * Real.sqrt 2 * b * x₁ + c = 0 ∧
                        a * x₂^2 + 2 * Real.sqrt 2 * b * x₂ + c = 0

-- Theorem statement
theorem quadratic_equation_two_roots
  (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h_geom : is_geometric_progression a b c) :
  has_two_distinct_roots a b c :=
sorry

end quadratic_equation_two_roots_l2977_297736


namespace cubic_log_relationship_l2977_297742

theorem cubic_log_relationship (x : ℝ) :
  (x^3 < 27 → Real.log x / Real.log (1/3) > -1) ∧
  ¬(Real.log x / Real.log (1/3) > -1 → x^3 < 27) :=
sorry

end cubic_log_relationship_l2977_297742


namespace investment_problem_l2977_297775

/-- Calculates the final amount for a simple interest investment -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Given conditions and proof goal -/
theorem investment_problem (rate : ℝ) :
  simpleInterest 150 rate 6 = 210 →
  simpleInterest 200 rate 3 = 240 := by
  sorry

end investment_problem_l2977_297775


namespace existence_implies_lower_bound_l2977_297745

theorem existence_implies_lower_bound (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ 2^x * (x - a) < 1) → a > -1 := by
  sorry

end existence_implies_lower_bound_l2977_297745


namespace no_real_solutions_l2977_297759

theorem no_real_solutions :
  ¬∃ (x : ℝ), Real.sqrt (x + 16) - 8 / Real.sqrt (x + 16) + 1 = 7 := by
sorry

end no_real_solutions_l2977_297759


namespace complex_fraction_sum_l2977_297785

theorem complex_fraction_sum (x y : ℂ) 
  (h : (x + y) / (x - y) + (x - y) / (x + y) = 2) :
  (x^4 + y^4) / (x^4 - y^4) + (x^4 - y^4) / (x^4 + y^4) = 2 := by
  sorry

end complex_fraction_sum_l2977_297785


namespace sports_league_games_l2977_297723

/-- Calculates the number of games in a sports league with two divisions -/
theorem sports_league_games (n₁ n₂ : ℕ) (intra_games : ℕ) (inter_games : ℕ) :
  n₁ = 5 →
  n₂ = 6 →
  intra_games = 3 →
  inter_games = 2 →
  (n₁ * (n₁ - 1) * intra_games / 2) +
  (n₂ * (n₂ - 1) * intra_games / 2) +
  (n₁ * n₂ * inter_games) = 135 :=
by
  sorry

#check sports_league_games

end sports_league_games_l2977_297723


namespace cindy_marbles_l2977_297735

/-- Proves that Cindy initially had 500 marbles given the conditions -/
theorem cindy_marbles : 
  ∀ (initial_marbles : ℕ),
  (initial_marbles - 4 * 80 > 0) →
  (4 * (initial_marbles - 4 * 80) = 720) →
  initial_marbles = 500 :=
by
  sorry

end cindy_marbles_l2977_297735


namespace inequality_proof_l2977_297755

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^3 / (a^2 + a*b + b^2) + b^3 / (b^2 + b*c + c^2) + c^3 / (c^2 + c*a + a^2) ≥ (a + b + c) / 3 := by
  sorry

end inequality_proof_l2977_297755


namespace bookcase_weight_excess_l2977_297744

theorem bookcase_weight_excess :
  let bookcase_limit : ℝ := 80
  let hardcover_count : ℕ := 70
  let hardcover_weight : ℝ := 0.5
  let textbook_count : ℕ := 30
  let textbook_weight : ℝ := 2
  let knickknack_count : ℕ := 3
  let knickknack_weight : ℝ := 6
  let total_weight := hardcover_count * hardcover_weight +
                      textbook_count * textbook_weight +
                      knickknack_count * knickknack_weight
  total_weight - bookcase_limit = 33 := by
sorry

end bookcase_weight_excess_l2977_297744


namespace max_value_quadratic_l2977_297754

theorem max_value_quadratic (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 - 2*x*y + 3*y^2 = 9) : 
  x^2 + 2*x*y + 3*y^2 ≤ 18 + 6*Real.sqrt 3 := by
  sorry

end max_value_quadratic_l2977_297754


namespace brendan_weekly_taxes_l2977_297718

/-- Calculates Brendan's weekly taxes paid after deduction -/
def weekly_taxes_paid (wage1 wage2 wage3 : ℚ) 
                      (hours1 hours2 hours3 : ℚ)
                      (tips1 tips2 tips3 : ℚ)
                      (reported_tips1 reported_tips2 reported_tips3 : ℚ)
                      (tax_rate1 tax_rate2 tax_rate3 : ℚ)
                      (deduction : ℚ) : ℚ :=
  let income1 := wage1 * hours1 + reported_tips1 * tips1 * hours1
  let income2 := wage2 * hours2 + reported_tips2 * tips2 * hours2
  let income3 := wage3 * hours3 + reported_tips3 * tips3 * hours3
  let taxes1 := income1 * tax_rate1
  let taxes2 := income2 * tax_rate2
  let taxes3 := income3 * tax_rate3
  taxes1 + taxes2 + taxes3 - deduction

theorem brendan_weekly_taxes :
  weekly_taxes_paid 12 15 10    -- wages
                    12 8 10     -- hours
                    20 15 5     -- tips
                    (1/2) (1/4) (3/5)  -- reported tips percentages
                    (22/100) (18/100) (16/100)  -- tax rates
                    50  -- deduction
  = 5588 / 100 := by
  sorry

end brendan_weekly_taxes_l2977_297718


namespace square_difference_601_599_l2977_297743

theorem square_difference_601_599 : (601 : ℤ)^2 - (599 : ℤ)^2 = 2400 := by sorry

end square_difference_601_599_l2977_297743


namespace range_of_a_l2977_297710

open Set

def p (a : ℝ) : Prop := a ≤ -2 ∨ a ≥ 2
def q (a : ℝ) : Prop := a ≥ -10

theorem range_of_a (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ a ∈ Iio (-10) ∪ Ioo (-2) 2 :=
sorry

end range_of_a_l2977_297710


namespace managers_salary_l2977_297715

theorem managers_salary (num_employees : ℕ) (avg_salary : ℝ) (salary_increase : ℝ) :
  num_employees = 20 →
  avg_salary = 1500 →
  salary_increase = 600 →
  (num_employees * avg_salary + (avg_salary + salary_increase) * (num_employees + 1) - num_employees * avg_salary) = 14100 :=
by sorry

end managers_salary_l2977_297715


namespace symmetric_center_of_translated_cosine_l2977_297703

theorem symmetric_center_of_translated_cosine : 
  let f (x : ℝ) := Real.cos (2 * x + π / 4)
  let g (x : ℝ) := f (x - π / 4)
  ∃ (k : ℤ), g ((k : ℝ) * π / 2 + 3 * π / 8) = g (-(k : ℝ) * π / 2 - 3 * π / 8) :=
by sorry

end symmetric_center_of_translated_cosine_l2977_297703


namespace total_tips_proof_l2977_297748

/-- Calculates the total tips earned over three days given the tips per customer,
    customer counts for Friday and Sunday, and that Saturday's count is 3 times Friday's. -/
def total_tips (tips_per_customer : ℕ) (friday_customers : ℕ) (sunday_customers : ℕ) : ℕ :=
  let saturday_customers := 3 * friday_customers
  tips_per_customer * (friday_customers + saturday_customers + sunday_customers)

/-- Proves that the total tips earned over three days is $296 -/
theorem total_tips_proof : total_tips 2 28 36 = 296 := by
  sorry

end total_tips_proof_l2977_297748


namespace chessboard_zero_condition_l2977_297702

/-- Represents a chessboard with natural numbers -/
def Chessboard (m n : ℕ) := Fin m → Fin n → ℕ

/-- Sums the numbers on black squares of a chessboard -/
def sumBlack (board : Chessboard m n) : ℕ := sorry

/-- Sums the numbers on white squares of a chessboard -/
def sumWhite (board : Chessboard m n) : ℕ := sorry

/-- Represents an allowed move on the chessboard -/
def allowedMove (board : Chessboard m n) (i j : Fin m) (k l : Fin n) (value : ℤ) : Chessboard m n := sorry

/-- Predicate to check if all numbers on the board are zero -/
def allZero (board : Chessboard m n) : Prop := ∀ i j, board i j = 0

/-- Predicate to check if a board can be reduced to all zeros using allowed moves -/
def canReduceToZero (board : Chessboard m n) : Prop := sorry

theorem chessboard_zero_condition {m n : ℕ} (board : Chessboard m n) :
  canReduceToZero board ↔ sumBlack board = sumWhite board := by sorry

end chessboard_zero_condition_l2977_297702


namespace heart_shape_area_l2977_297761

/-- The area of a heart shape composed of specific geometric elements -/
theorem heart_shape_area : 
  let π : ℝ := 3.14
  let semicircle_diameter : ℝ := 10
  let sector_radius : ℝ := 10
  let sector_angle : ℝ := 45
  let square_side : ℝ := 10
  let semicircle_area : ℝ := 2 * (1/2 * π * (semicircle_diameter/2)^2)
  let sector_area : ℝ := 2 * ((sector_angle/360) * π * sector_radius^2)
  let square_area : ℝ := square_side^2
  semicircle_area + sector_area + square_area = 257 := by
  sorry

end heart_shape_area_l2977_297761


namespace angle_measure_proof_l2977_297733

theorem angle_measure_proof :
  ∃ (x : ℝ), x + (3 * x - 8) = 90 ∧ x = 24.5 := by
  sorry

end angle_measure_proof_l2977_297733


namespace swap_values_l2977_297727

theorem swap_values (a b : ℕ) (ha : a = 8) (hb : b = 17) :
  ∃ c : ℕ, (c = b) ∧ (b = a) ∧ (a = c) ∧ (a = 17 ∧ b = 8) := by
  sorry

end swap_values_l2977_297727


namespace integer_triplet_solution_l2977_297766

theorem integer_triplet_solution (x y z : ℤ) :
  x^2 + y^2 + z^2 - 2*x*y*z = 0 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end integer_triplet_solution_l2977_297766


namespace simplify_expression_l2977_297762

theorem simplify_expression : (1 / ((-5^4)^2)) * (-5)^9 = -5 := by
  sorry

end simplify_expression_l2977_297762


namespace fourth_term_of_geometric_progression_l2977_297787

def geometric_progression (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^(n - 1)

theorem fourth_term_of_geometric_progression 
  (a : ℝ) (r : ℝ) (h1 : a > 0) (h2 : r > 0) :
  geometric_progression a r 1 = 4 ∧ 
  geometric_progression a r 2 = Real.sqrt 4 ∧ 
  geometric_progression a r 3 = 4^(1/4) →
  geometric_progression a r 4 = 4^(1/8) := by
sorry

end fourth_term_of_geometric_progression_l2977_297787


namespace bug_position_after_3000_jumps_l2977_297751

/-- Represents the points on the circle -/
inductive Point : Type
| one : Point
| two : Point
| three : Point
| four : Point
| five : Point
| six : Point
| seven : Point

/-- Determines if a point is odd-numbered -/
def isOdd : Point → Bool
  | Point.one => true
  | Point.two => false
  | Point.three => true
  | Point.four => false
  | Point.five => true
  | Point.six => false
  | Point.seven => true

/-- Performs a single jump based on the current point -/
def jump : Point → Point
  | Point.one => Point.three
  | Point.two => Point.five
  | Point.three => Point.five
  | Point.four => Point.seven
  | Point.five => Point.seven
  | Point.six => Point.two
  | Point.seven => Point.two

/-- Performs multiple jumps -/
def multiJump (start : Point) (n : Nat) : Point :=
  match n with
  | 0 => start
  | n + 1 => jump (multiJump start n)

theorem bug_position_after_3000_jumps :
  multiJump Point.seven 3000 = Point.two :=
sorry

end bug_position_after_3000_jumps_l2977_297751


namespace lottery_solution_l2977_297711

def lottery_numbers (A B C D E : ℕ) : Prop :=
  -- Define the five numbers
  let AB := 10 * A + B
  let BC := 10 * B + C
  let CA := 10 * C + A
  let CB := 10 * C + B
  let CD := 10 * C + D
  -- Conditions
  (1 ≤ A) ∧ (A < B) ∧ (B < C) ∧ (C < 9) ∧ (B < D) ∧ (D ≤ 9) ∧
  (AB < BC) ∧ (BC < CA) ∧ (CA < CB) ∧ (CB < CD) ∧
  (AB + BC + CA + CB + CD = 100 * B + 10 * C + C) ∧
  (CA * BC = 1000 * B + 100 * B + 10 * E + C) ∧
  (CA * CD = 1000 * E + 100 * C + 10 * C + D)

theorem lottery_solution :
  ∃! (A B C D E : ℕ), lottery_numbers A B C D E ∧ A = 1 ∧ B = 2 ∧ C = 8 ∧ D = 5 ∧ E = 6 := by
  sorry

end lottery_solution_l2977_297711


namespace four_holes_when_unfolded_l2977_297760

/-- Represents a rectangular sheet of paper -/
structure Paper :=
  (width : ℝ)
  (height : ℝ)
  (holes : List (ℝ × ℝ))

/-- Represents the state of the paper after folding -/
inductive FoldState
  | Unfolded
  | DiagonalFold
  | HalfFold
  | FinalFold

/-- Represents a folding operation -/
def fold (p : Paper) (state : FoldState) : Paper :=
  sorry

/-- Represents the operation of punching a hole -/
def punchHole (p : Paper) (x : ℝ) (y : ℝ) : Paper :=
  sorry

/-- Represents the unfolding operation -/
def unfold (p : Paper) : Paper :=
  sorry

/-- The main theorem to prove -/
theorem four_holes_when_unfolded (p : Paper) :
  let p1 := fold p FoldState.DiagonalFold
  let p2 := fold p1 FoldState.HalfFold
  let p3 := fold p2 FoldState.FinalFold
  let p4 := punchHole p3 (p.width / 2) (p.height / 2)
  let final := unfold p4
  final.holes.length = 4 :=
sorry

end four_holes_when_unfolded_l2977_297760


namespace atlantic_charge_calculation_l2977_297792

/-- Represents the additional charge per minute for Atlantic Call -/
def atlantic_charge_per_minute : ℚ := 1/5

/-- United Telephone's base rate -/
def united_base_rate : ℚ := 11

/-- United Telephone's charge per minute -/
def united_charge_per_minute : ℚ := 1/4

/-- Atlantic Call's base rate -/
def atlantic_base_rate : ℚ := 12

/-- Number of minutes for which the bills are equal -/
def equal_bill_minutes : ℕ := 20

theorem atlantic_charge_calculation :
  united_base_rate + united_charge_per_minute * equal_bill_minutes =
  atlantic_base_rate + atlantic_charge_per_minute * equal_bill_minutes :=
sorry

end atlantic_charge_calculation_l2977_297792


namespace condition_p_sufficient_not_necessary_l2977_297717

-- Define a quadrilateral in a plane
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define vector equality
def vector_equal (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 = v2.1 ∧ v1.2 = v2.2

-- Define vector scaling
def vector_scale (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (k * v.1, k * v.2)

-- Define condition p
def condition_p (q : Quadrilateral) : Prop :=
  vector_equal (q.B.1 - q.A.1, q.B.2 - q.A.2) (vector_scale 2 (q.C.1 - q.D.1, q.C.2 - q.D.2))

-- Define a trapezoid
def is_trapezoid (q : Quadrilateral) : Prop :=
  (q.A.2 - q.B.2) / (q.A.1 - q.B.1) = (q.D.2 - q.C.2) / (q.D.1 - q.C.1) ∨
  (q.A.2 - q.D.2) / (q.A.1 - q.D.1) = (q.B.2 - q.C.2) / (q.B.1 - q.C.1)

-- Theorem statement
theorem condition_p_sufficient_not_necessary (q : Quadrilateral) :
  (condition_p q → is_trapezoid q) ∧ ¬(is_trapezoid q → condition_p q) :=
sorry

end condition_p_sufficient_not_necessary_l2977_297717


namespace daisy_toys_theorem_l2977_297731

/-- The number of dog toys Daisy's owner bought on Wednesday -/
def wednesday_toys (monday_toys tuesday_left tuesday_bought total_if_found : ℕ) : ℕ :=
  total_if_found - (tuesday_left + tuesday_bought)

theorem daisy_toys_theorem (monday_toys tuesday_left tuesday_bought total_if_found : ℕ) 
  (h1 : monday_toys = 5)
  (h2 : tuesday_left = 3)
  (h3 : tuesday_bought = 3)
  (h4 : total_if_found = 13) :
  wednesday_toys monday_toys tuesday_left tuesday_bought total_if_found = 7 := by
  sorry

#eval wednesday_toys 5 3 3 13

end daisy_toys_theorem_l2977_297731


namespace hyperbola_m_value_l2977_297797

/-- Represents a hyperbola with equation x²/m - y²/6 = 1 -/
structure Hyperbola where
  m : ℝ
  eq : ∀ x y : ℝ, x^2 / m - y^2 / 6 = 1

/-- The focal distance of a hyperbola -/
def focal_distance (h : Hyperbola) : ℝ := 6

theorem hyperbola_m_value (h : Hyperbola) (hf : focal_distance h = 6) : h.m = 3 := by
  sorry

end hyperbola_m_value_l2977_297797


namespace retail_price_increase_l2977_297788

theorem retail_price_increase (W R : ℝ) 
  (h : 0.80 * R = 1.44000000000000014 * W) : 
  (R - W) / W * 100 = 80.000000000000017 :=
by sorry

end retail_price_increase_l2977_297788


namespace express_y_in_terms_of_x_l2977_297795

theorem express_y_in_terms_of_x (x y : ℝ) (h : 2 * x + 3 * y = 4) :
  y = (4 - 2 * x) / 3 := by
sorry

end express_y_in_terms_of_x_l2977_297795


namespace may_largest_drop_l2977_297708

/-- Represents the months in the first half of the year -/
inductive Month
| January
| February
| March
| April
| May
| June

/-- The price change for each month -/
def price_change (m : Month) : ℝ :=
  match m with
  | .January  => -1.25
  | .February => 2.75
  | .March    => -0.75
  | .April    => 1.50
  | .May      => -3.00
  | .June     => -1.00

/-- Definition of a price drop -/
def is_price_drop (x : ℝ) : Prop := x < 0

/-- The month with the largest price drop -/
def largest_drop (m : Month) : Prop :=
  ∀ n : Month, is_price_drop (price_change n) →
    price_change n ≥ price_change m

theorem may_largest_drop :
  largest_drop Month.May :=
sorry

end may_largest_drop_l2977_297708


namespace instantaneous_velocity_at_2_l2977_297725

-- Define the distance function
def s (t : ℝ) : ℝ := 3 * t^2 + t

-- Define the velocity function as the derivative of s
def v (t : ℝ) : ℝ := 6 * t + 1

-- Theorem statement
theorem instantaneous_velocity_at_2 : v 2 = 13 := by
  sorry

end instantaneous_velocity_at_2_l2977_297725


namespace jessica_quarters_l2977_297765

/-- Calculates the number of quarters Jessica has after her sister borrows some. -/
def quarters_remaining (initial : ℕ) (borrowed : ℕ) : ℕ :=
  initial - borrowed

/-- Theorem stating that Jessica has 5 quarters remaining. -/
theorem jessica_quarters : quarters_remaining 8 3 = 5 := by
  sorry

end jessica_quarters_l2977_297765


namespace max_min_sum_of_quadratic_expression_l2977_297712

theorem max_min_sum_of_quadratic_expression (a b : ℝ) 
  (h : a^2 + a*b + b^2 = 3) : 
  let f := fun (x y : ℝ) => x^2 - x*y + y^2
  ∃ (M m : ℝ), (∀ x y, f x y ≤ M ∧ m ≤ f x y) ∧ M + m = 10 := by
sorry

end max_min_sum_of_quadratic_expression_l2977_297712


namespace factorization_ax_squared_minus_a_l2977_297728

theorem factorization_ax_squared_minus_a (a x : ℝ) : a * x^2 - a = a * (x + 1) * (x - 1) := by
  sorry

end factorization_ax_squared_minus_a_l2977_297728


namespace function_identity_l2977_297730

theorem function_identity (f : ℝ → ℝ) 
  (h_bounded : ∃ a b : ℝ, ∃ M : ℝ, ∀ x ∈ Set.Icc a b, |f x| ≤ M)
  (h_additive : ∀ x₁ x₂ : ℝ, f (x₁ + x₂) = f x₁ + f x₂)
  (h_one : f 1 = 1) :
  ∀ x : ℝ, f x = x := by
sorry

end function_identity_l2977_297730


namespace negation_equivalence_l2977_297778

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 1 > 3*x) ↔ (∀ x : ℝ, x^2 + 1 ≤ 3*x) := by sorry

end negation_equivalence_l2977_297778


namespace simplify_sqrt_144000_l2977_297741

theorem simplify_sqrt_144000 : Real.sqrt 144000 = 120 * Real.sqrt 10 := by
  sorry

end simplify_sqrt_144000_l2977_297741


namespace syllogism_cos_periodic_l2977_297768

-- Define the properties
def IsTrigonometric (f : ℝ → ℝ) : Prop := sorry
def IsPeriodic (f : ℝ → ℝ) : Prop := sorry

-- Define the cosine function
def cos : ℝ → ℝ := sorry

-- Theorem to prove
theorem syllogism_cos_periodic :
  (∀ f : ℝ → ℝ, IsTrigonometric f → IsPeriodic f) →
  (IsTrigonometric cos) →
  (IsPeriodic cos) := by sorry

end syllogism_cos_periodic_l2977_297768
