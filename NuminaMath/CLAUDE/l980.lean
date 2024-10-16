import Mathlib

namespace NUMINAMATH_CALUDE_complex_equation_solution_l980_98004

theorem complex_equation_solution (i : ℂ) (z : ℂ) 
  (h1 : i * i = -1) 
  (h2 : i * z = (1 - 2*i)^2) : 
  z = -4 + 3*i := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l980_98004


namespace NUMINAMATH_CALUDE_sum_of_integers_l980_98032

theorem sum_of_integers (x y : ℕ+) 
  (h1 : x.val^2 + y.val^2 = 130) 
  (h2 : x.val * y.val = 45) : 
  x.val + y.val = 2 * Real.sqrt 55 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l980_98032


namespace NUMINAMATH_CALUDE_sum_O_eq_321_l980_98021

/-- O(n) represents the sum of odd digits in number n -/
def O (n : ℕ) : ℕ := sorry

/-- The sum of O(n) from 1 to 75 -/
def sum_O : ℕ := (Finset.range 75).sum (λ n => O (n + 1))

/-- Theorem: The sum of O(n) from 1 to 75 equals 321 -/
theorem sum_O_eq_321 : sum_O = 321 := by sorry

end NUMINAMATH_CALUDE_sum_O_eq_321_l980_98021


namespace NUMINAMATH_CALUDE_water_speed_calculation_l980_98012

/-- Given a person who can swim in still water at 10 km/h and takes 2 hours to swim 12 km against
    the current, prove that the speed of the water is 4 km/h. -/
theorem water_speed_calculation (still_water_speed : ℝ) (distance : ℝ) (time : ℝ) (water_speed : ℝ) :
  still_water_speed = 10 →
  distance = 12 →
  time = 2 →
  distance = (still_water_speed - water_speed) * time →
  water_speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_water_speed_calculation_l980_98012


namespace NUMINAMATH_CALUDE_pole_height_pole_height_is_8_5_l980_98022

/-- The height of a pole given specific cable and person measurements -/
theorem pole_height (cable_length : ℝ) (cable_ground_distance : ℝ) 
  (person_height : ℝ) (person_distance : ℝ) : ℝ :=
  cable_length * person_height / (cable_ground_distance - person_distance)

/-- Proof that a pole is 8.5 meters tall given specific measurements -/
theorem pole_height_is_8_5 :
  pole_height 5 5 1.7 4 = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_pole_height_pole_height_is_8_5_l980_98022


namespace NUMINAMATH_CALUDE_smallest_B_for_divisibility_by_three_l980_98074

def seven_digit_number (B : Nat) : Nat :=
  4000000 + B * 100000 + 803942

theorem smallest_B_for_divisibility_by_three :
  ∃ (B : Nat), B < 10 ∧ 
    seven_digit_number B % 3 = 0 ∧
    ∀ (C : Nat), C < B → seven_digit_number C % 3 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_B_for_divisibility_by_three_l980_98074


namespace NUMINAMATH_CALUDE_problem_1_l980_98089

theorem problem_1 (a b : ℝ) (h1 : a - b = 2) (h2 : a * b = 3) :
  a^2 + b^2 = 10 := by sorry

end NUMINAMATH_CALUDE_problem_1_l980_98089


namespace NUMINAMATH_CALUDE_sample_size_representation_l980_98039

/-- Represents a population of students -/
structure Population where
  size : ℕ

/-- Represents a sample of students -/
structure Sample where
  size : ℕ
  population : Population
  h : size ≤ population.size

/-- Theorem: In a statistical analysis context, when 30 students are selected from a population of 500,
    the number 30 represents the sample size -/
theorem sample_size_representation (pop : Population) (s : Sample) :
  pop.size = 500 →
  s.size = 30 →
  s.population = pop →
  s.size = Sample.size s :=
by sorry

end NUMINAMATH_CALUDE_sample_size_representation_l980_98039


namespace NUMINAMATH_CALUDE_sin_150_minus_alpha_l980_98034

theorem sin_150_minus_alpha (α : Real) (h : α = 240 * Real.pi / 180) :
  Real.sin (150 * Real.pi / 180 - α) = -1 := by sorry

end NUMINAMATH_CALUDE_sin_150_minus_alpha_l980_98034


namespace NUMINAMATH_CALUDE_line_perp_two_planes_implies_planes_parallel_perp_lines_perp_planes_implies_planes_perpendicular_l980_98079

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the necessary relations
variable (belongs_to : Point → Line → Prop)
variable (lies_in : Point → Plane → Prop)
variable (perpendicular : Line → Line → Prop)
variable (line_perp_plane : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- Theorem 1
theorem line_perp_two_planes_implies_planes_parallel
  (m : Line) (α β : Plane) :
  line_perp_plane m α → line_perp_plane m β → plane_parallel α β :=
sorry

-- Theorem 2
theorem perp_lines_perp_planes_implies_planes_perpendicular
  (m n : Line) (α β : Plane) :
  line_perp_plane m α → line_perp_plane n β → perpendicular m n →
  plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_line_perp_two_planes_implies_planes_parallel_perp_lines_perp_planes_implies_planes_perpendicular_l980_98079


namespace NUMINAMATH_CALUDE_sum_of_four_integers_l980_98005

theorem sum_of_four_integers (a b c d : ℕ+) : 
  (a > 1) → (b > 1) → (c > 1) → (d > 1) →
  (a * b * c * d = 1000000) →
  (Nat.gcd a.val b.val = 1) → (Nat.gcd a.val c.val = 1) → (Nat.gcd a.val d.val = 1) →
  (Nat.gcd b.val c.val = 1) → (Nat.gcd b.val d.val = 1) →
  (Nat.gcd c.val d.val = 1) →
  (a + b + c + d = 15698) := by
sorry

end NUMINAMATH_CALUDE_sum_of_four_integers_l980_98005


namespace NUMINAMATH_CALUDE_log_relation_l980_98077

theorem log_relation (p q : ℝ) (hp : 0 < p) : 
  (Real.log 5 / Real.log 8 = p) → (Real.log 125 / Real.log 2 = q * p) → q = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_relation_l980_98077


namespace NUMINAMATH_CALUDE_male_students_in_sample_l980_98075

/-- Represents the stratified sampling scenario -/
structure StratifiedSample where
  total_population : ℕ
  sample_size : ℕ
  female_count : ℕ

/-- Calculates the number of male students to be drawn in a stratified sample -/
def male_students_drawn (s : StratifiedSample) : ℕ :=
  s.sample_size

/-- Theorem stating the number of male students to be drawn in the given scenario -/
theorem male_students_in_sample (s : StratifiedSample) 
  (h1 : s.total_population = 900)
  (h2 : s.sample_size = 45)
  (h3 : s.female_count = 0) :
  male_students_drawn s = 25 := by
    sorry

#eval male_students_drawn { total_population := 900, sample_size := 45, female_count := 0 }

end NUMINAMATH_CALUDE_male_students_in_sample_l980_98075


namespace NUMINAMATH_CALUDE_cubic_root_sum_l980_98054

theorem cubic_root_sum (p q r : ℝ) : 
  p^3 - 6*p^2 + 11*p - 6 = 0 →
  q^3 - 6*q^2 + 11*q - 6 = 0 →
  r^3 - 6*r^2 + 11*r - 6 = 0 →
  p * q / r + p * r / q + q * r / p = 49 / 6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l980_98054


namespace NUMINAMATH_CALUDE_x_plus_y_value_l980_98008

theorem x_plus_y_value (x y : ℤ) (hx : -x = 3) (hy : |y| = 5) : x + y = 2 ∨ x + y = -8 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l980_98008


namespace NUMINAMATH_CALUDE_mersenne_prime_definition_l980_98027

def is_mersenne_prime (n : ℕ) : Prop :=
  ∃ p : ℕ, n = 2^p - 1 ∧ Nat.Prime n

def largest_known_prime : ℕ := 2^82589933 - 1

axiom largest_known_prime_is_prime : Nat.Prime largest_known_prime

theorem mersenne_prime_definition :
  ∀ n : ℕ, is_mersenne_prime n → (∃ name : String, name = "Mersenne prime") :=
by sorry

end NUMINAMATH_CALUDE_mersenne_prime_definition_l980_98027


namespace NUMINAMATH_CALUDE_not_certain_rain_beijing_no_rain_shanghai_l980_98013

-- Define the probabilities of rainfall
def probability_rain_beijing : ℝ := 0.8
def probability_rain_shanghai : ℝ := 0.2

-- Theorem to prove
theorem not_certain_rain_beijing_no_rain_shanghai :
  ¬(probability_rain_beijing = 1 ∧ probability_rain_shanghai = 0) :=
sorry

end NUMINAMATH_CALUDE_not_certain_rain_beijing_no_rain_shanghai_l980_98013


namespace NUMINAMATH_CALUDE_square_perimeter_ratio_l980_98033

theorem square_perimeter_ratio (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∃ k : ℝ, b = k * a * Real.sqrt 2) → (4 * b) / (4 * a) = 5 → 
  b / (a * Real.sqrt 2) = 5 * Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_ratio_l980_98033


namespace NUMINAMATH_CALUDE_exp_decreasing_for_small_base_l980_98078

theorem exp_decreasing_for_small_base (a x y : ℝ) (ha : 0 < a) (ha' : a < 1) (hxy : x < y) :
  a^x > a^y := by
  sorry

end NUMINAMATH_CALUDE_exp_decreasing_for_small_base_l980_98078


namespace NUMINAMATH_CALUDE_certain_number_exists_and_unique_l980_98010

theorem certain_number_exists_and_unique : 
  ∃! x : ℝ, 22030 = (x + 445) * (2 * (x - 445)) + 30 := by
sorry

end NUMINAMATH_CALUDE_certain_number_exists_and_unique_l980_98010


namespace NUMINAMATH_CALUDE_first_part_value_l980_98073

theorem first_part_value (x y : ℝ) 
  (sum_constraint : x + y = 36)
  (weighted_sum_constraint : 8 * x + 3 * y = 203) :
  x = 19 := by
sorry

end NUMINAMATH_CALUDE_first_part_value_l980_98073


namespace NUMINAMATH_CALUDE_expression_evaluation_l980_98036

theorem expression_evaluation :
  (3 : ℚ)^3010 * 2^3008 / 6^3009 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l980_98036


namespace NUMINAMATH_CALUDE_infinite_coprime_pairs_l980_98040

theorem infinite_coprime_pairs (m : ℕ+) :
  ∃ (seq : ℕ → ℕ × ℕ), ∀ n : ℕ,
    let (x, y) := seq n
    Int.gcd x y = 1 ∧
    x > 0 ∧ y > 0 ∧
    (y^2 + m.val) % x = 0 ∧
    (x^2 + m.val) % y = 0 ∧
    (∀ k < n, seq k ≠ seq n) :=
sorry

end NUMINAMATH_CALUDE_infinite_coprime_pairs_l980_98040


namespace NUMINAMATH_CALUDE_cake_triangles_l980_98048

/-- The number of triangular pieces that can be cut from a rectangular cake -/
theorem cake_triangles (cake_length cake_width triangle_base triangle_height : ℝ) 
  (h1 : cake_length = 24)
  (h2 : cake_width = 20)
  (h3 : triangle_base = 2)
  (h4 : triangle_height = 2) :
  (cake_length * cake_width) / (1/2 * triangle_base * triangle_height) = 240 :=
by sorry

end NUMINAMATH_CALUDE_cake_triangles_l980_98048


namespace NUMINAMATH_CALUDE_roof_area_difference_l980_98093

/-- Proves the difference in area between two rectangular roofs -/
theorem roof_area_difference (w : ℝ) (h1 : w > 0) (h2 : 4 * w * w = 784) : 
  5 * w * w - 4 * w * w = 196 := by
  sorry

end NUMINAMATH_CALUDE_roof_area_difference_l980_98093


namespace NUMINAMATH_CALUDE_perpendicular_lines_l980_98047

theorem perpendicular_lines (x y : ℝ) : 
  let angle1 : ℝ := 50 + x - y
  let angle2 : ℝ := angle1 - (10 + 2*x - 2*y)
  angle1 + angle2 = 90 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l980_98047


namespace NUMINAMATH_CALUDE_one_pencil_one_pen_cost_l980_98056

def pencil_cost : ℝ → ℝ → Prop := λ p q ↦ 3 * p + 2 * q = 3.75
def pen_cost : ℝ → ℝ → Prop := λ p q ↦ 2 * p + 3 * q = 4.05

theorem one_pencil_one_pen_cost (p q : ℝ) 
  (h1 : pencil_cost p q) (h2 : pen_cost p q) : 
  p + q = 1.56 := by
  sorry

end NUMINAMATH_CALUDE_one_pencil_one_pen_cost_l980_98056


namespace NUMINAMATH_CALUDE_supermarket_spending_l980_98050

theorem supermarket_spending (total : ℚ) 
  (h1 : total = 120) 
  (h2 : ∃ (fruits meat bakery candy : ℚ), 
    fruits + meat + bakery + candy = total ∧
    fruits = (1/2) * total ∧
    meat = (1/3) * total ∧
    bakery = (1/10) * total) : 
  ∃ (candy : ℚ), candy = 8 := by
sorry

end NUMINAMATH_CALUDE_supermarket_spending_l980_98050


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_15_13_l980_98069

theorem half_abs_diff_squares_15_13 : 
  (1/2 : ℝ) * |15^2 - 13^2| = 28 := by sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_15_13_l980_98069


namespace NUMINAMATH_CALUDE_sum_mod_ten_l980_98059

theorem sum_mod_ten : (17145 + 17146 + 17147 + 17148 + 17149) % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_ten_l980_98059


namespace NUMINAMATH_CALUDE_basketball_conference_games_l980_98023

/-- The number of teams in the basketball conference -/
def num_teams : ℕ := 10

/-- The number of times each team plays every other team in the conference -/
def games_per_pair : ℕ := 3

/-- The number of non-conference games each team plays -/
def non_conference_games : ℕ := 5

/-- The total number of games in a season for the basketball conference -/
def total_games : ℕ := (num_teams.choose 2 * games_per_pair) + (num_teams * non_conference_games)

theorem basketball_conference_games :
  total_games = 185 := by sorry

end NUMINAMATH_CALUDE_basketball_conference_games_l980_98023


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l980_98090

-- Problem 1
theorem problem_1 : 0.108 / 1.2 + 0.7 = 0.79 := by sorry

-- Problem 2
theorem problem_2 : (9.8 - 3.75) / 25 / 0.4 = 0.605 := by sorry

-- Problem 3
theorem problem_3 : 6.3 * 15 + 1/3 * 75/100 = 94.75 := by sorry

-- Problem 4
theorem problem_4 : 8 * 0.56 + 5.4 * 0.8 - 80/100 = 8 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l980_98090


namespace NUMINAMATH_CALUDE_basketball_tryouts_l980_98045

theorem basketball_tryouts (girls : ℕ) (boys : ℕ) (called_back : ℕ) (not_selected : ℕ) : 
  boys = 14 →
  called_back = 2 →
  not_selected = 21 →
  girls + boys = called_back + not_selected →
  girls = 9 := by
sorry

end NUMINAMATH_CALUDE_basketball_tryouts_l980_98045


namespace NUMINAMATH_CALUDE_extra_grass_seed_coverage_l980_98095

/-- Calculates the extra coverage of grass seed after reseeding a lawn -/
theorem extra_grass_seed_coverage 
  (lawn_length : ℕ) 
  (lawn_width : ℕ) 
  (seed_bags : ℕ) 
  (coverage_per_bag : ℕ) : 
  lawn_length = 35 → 
  lawn_width = 48 → 
  seed_bags = 6 → 
  coverage_per_bag = 500 → 
  seed_bags * coverage_per_bag - lawn_length * lawn_width = 1320 :=
by
  sorry

#check extra_grass_seed_coverage

end NUMINAMATH_CALUDE_extra_grass_seed_coverage_l980_98095


namespace NUMINAMATH_CALUDE_correct_horses_b_l980_98026

/-- Represents the number of horses put in the pasture by party b -/
def horses_b : ℕ := 6

/-- Represents the total cost of the pasture -/
def total_cost : ℕ := 870

/-- Represents the amount b should pay -/
def b_payment : ℕ := 360

/-- Represents the number of horses put in by party a -/
def horses_a : ℕ := 12

/-- Represents the number of months horses from party a stayed -/
def months_a : ℕ := 8

/-- Represents the number of months horses from party b stayed -/
def months_b : ℕ := 9

/-- Represents the number of horses put in by party c -/
def horses_c : ℕ := 18

/-- Represents the number of months horses from party c stayed -/
def months_c : ℕ := 6

theorem correct_horses_b :
  (horses_b * months_b : ℚ) / (horses_a * months_a + horses_b * months_b + horses_c * months_c) * total_cost = b_payment :=
by sorry

end NUMINAMATH_CALUDE_correct_horses_b_l980_98026


namespace NUMINAMATH_CALUDE_sqrt_neg_three_l980_98018

theorem sqrt_neg_three (z : ℂ) : z * z = -3 ↔ z = Complex.I * Real.sqrt 3 ∨ z = -Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_neg_three_l980_98018


namespace NUMINAMATH_CALUDE_simplify_expression_l980_98096

theorem simplify_expression : 18 * (8 / 15) * (1 / 12)^2 = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l980_98096


namespace NUMINAMATH_CALUDE_octal_addition_sum_l980_98035

/-- Given an octal addition 3XY₈ + 52₈ = 4X3₈, prove that X + Y = 1 in base 10 -/
theorem octal_addition_sum (X Y : ℕ) : 
  (3 * 8^2 + X * 8 + Y) + (5 * 8 + 2) = 4 * 8^2 + X * 8 + 3 → X + Y = 1 := by
  sorry

end NUMINAMATH_CALUDE_octal_addition_sum_l980_98035


namespace NUMINAMATH_CALUDE_house_occupancy_l980_98081

/-- The number of people in the house given specific room occupancies. -/
def people_in_house (bedroom living_room kitchen garage patio : ℕ) : ℕ :=
  bedroom + living_room + kitchen + garage + patio

/-- The problem statement as a theorem. -/
theorem house_occupancy : ∃ (bedroom living_room kitchen garage patio : ℕ),
  bedroom = 7 ∧
  living_room = 8 ∧
  kitchen = living_room + 3 ∧
  garage * 2 = kitchen ∧
  patio = garage * 2 ∧
  people_in_house bedroom living_room kitchen garage patio = 41 := by
  sorry

end NUMINAMATH_CALUDE_house_occupancy_l980_98081


namespace NUMINAMATH_CALUDE_original_male_count_l980_98025

/-- Represents the number of students of each gender -/
structure StudentCount where
  male : ℕ
  female : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (s : StudentCount) : Prop :=
  (s.male : ℚ) / ((s.female : ℚ) - 15) = 2 ∧
  ((s.male : ℚ) - 45) / ((s.female : ℚ) - 15) = 1/5

/-- The theorem stating that the original number of male students is 50 -/
theorem original_male_count (s : StudentCount) :
  satisfiesConditions s → s.male = 50 := by
  sorry


end NUMINAMATH_CALUDE_original_male_count_l980_98025


namespace NUMINAMATH_CALUDE_binomial_fraction_integer_l980_98031

theorem binomial_fraction_integer (k n : ℕ) (h1 : 1 ≤ k) (h2 : k < n) :
  (∃ m : ℕ, n + 2 = m * (k + 2)) ↔ 
  ∃ z : ℤ, z = (2*n - 3*k - 2) * (n.choose k) / (k + 2) :=
sorry

end NUMINAMATH_CALUDE_binomial_fraction_integer_l980_98031


namespace NUMINAMATH_CALUDE_line_points_k_value_l980_98088

/-- Given a line with equation x - 5/2y + 1 = 0 and two points (m, n) and (m + 1/2, n + 1/k) on this line,
    prove that k = 3/5 -/
theorem line_points_k_value (m n k : ℝ) :
  (m - 5/2 * n + 1 = 0) →
  (m + 1/2 - 5/2 * (n + 1/k) + 1 = 0) →
  k = 3/5 := by
  sorry


end NUMINAMATH_CALUDE_line_points_k_value_l980_98088


namespace NUMINAMATH_CALUDE_x_value_proof_l980_98087

theorem x_value_proof (x : ℚ) (h : 2/3 - 1/4 = 4/x) : x = 48/5 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l980_98087


namespace NUMINAMATH_CALUDE_division_problem_l980_98038

theorem division_problem : ∃ x : ℝ, (3.242 * 15) / x = 0.04863 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l980_98038


namespace NUMINAMATH_CALUDE_evaluate_expression_l980_98064

theorem evaluate_expression : 4 * (8 - 3) - 6 = 14 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l980_98064


namespace NUMINAMATH_CALUDE_min_value_ab_l980_98061

theorem min_value_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b - a * b + 3 = 0) :
  9 ≤ a * b ∧ ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ a₀ + b₀ - a₀ * b₀ + 3 = 0 ∧ a₀ * b₀ = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_ab_l980_98061


namespace NUMINAMATH_CALUDE_intersection_distance_sum_l980_98065

noncomputable section

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 1 = 0

-- Define point P
def point_P : ℝ × ℝ := (0, 1)

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem intersection_distance_sum :
  ∃ (A B : ℝ × ℝ),
    curve_C A.1 A.2 ∧ curve_C B.1 B.2 ∧
    line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
    A ≠ B ∧
    distance point_P A + distance point_P B = 8 * Real.sqrt 2 / 5 :=
sorry

end

end NUMINAMATH_CALUDE_intersection_distance_sum_l980_98065


namespace NUMINAMATH_CALUDE_extra_workers_for_clay_soil_l980_98001

/-- Represents the digging problem with different soil types and worker requirements -/
structure DiggingProblem where
  sandy_workers : ℕ
  sandy_hours : ℕ
  clay_time_factor : ℕ
  new_hours : ℕ

/-- Calculates the number of extra workers needed for the clay soil digging task -/
def extra_workers_needed (p : DiggingProblem) : ℕ :=
  let sandy_man_hours := p.sandy_workers * p.sandy_hours
  let clay_man_hours := sandy_man_hours * p.clay_time_factor
  let total_workers_needed := clay_man_hours / p.new_hours
  total_workers_needed - p.sandy_workers

/-- Theorem stating that given the problem conditions, 75 extra workers are needed -/
theorem extra_workers_for_clay_soil : 
  let p : DiggingProblem := {
    sandy_workers := 45,
    sandy_hours := 8,
    clay_time_factor := 2,
    new_hours := 6
  }
  extra_workers_needed p = 75 := by sorry

end NUMINAMATH_CALUDE_extra_workers_for_clay_soil_l980_98001


namespace NUMINAMATH_CALUDE_pythagoras_field_planted_fraction_l980_98052

theorem pythagoras_field_planted_fraction :
  ∀ (a b c x : ℝ),
  a = 5 ∧ b = 12 ∧ c^2 = a^2 + b^2 →
  (a - x)^2 + (b - x)^2 = 4^2 →
  (a * b / 2 - x^2) / (a * b / 2) = 734 / 750 := by
sorry

end NUMINAMATH_CALUDE_pythagoras_field_planted_fraction_l980_98052


namespace NUMINAMATH_CALUDE_sqrt_2n_equals_64_l980_98055

theorem sqrt_2n_equals_64 (n : ℝ) : Real.sqrt (2 * n) = 64 → n = 2048 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2n_equals_64_l980_98055


namespace NUMINAMATH_CALUDE_benzoic_acid_molecular_weight_l980_98015

/-- The molecular weight of Benzoic acid -/
def molecular_weight_benzoic_acid : ℝ := 122

/-- The number of moles given in the problem -/
def moles_given : ℝ := 4

/-- The total molecular weight for the given number of moles -/
def total_molecular_weight : ℝ := 488

/-- Theorem stating that the molecular weight of Benzoic acid is correct -/
theorem benzoic_acid_molecular_weight :
  molecular_weight_benzoic_acid = total_molecular_weight / moles_given :=
sorry

end NUMINAMATH_CALUDE_benzoic_acid_molecular_weight_l980_98015


namespace NUMINAMATH_CALUDE_equation_one_solution_l980_98067

-- Define the equation
def equation (x p : ℝ) : Prop :=
  2 * |x - p| + |x - 2| = 1

-- Define the property of having exactly one solution
def has_exactly_one_solution (p : ℝ) : Prop :=
  ∃! x, equation x p

-- Theorem statement
theorem equation_one_solution :
  ∀ p : ℝ, has_exactly_one_solution p ↔ (p = 1 ∨ p = 3) :=
sorry

end NUMINAMATH_CALUDE_equation_one_solution_l980_98067


namespace NUMINAMATH_CALUDE_worker_y_defective_rate_l980_98043

/-- Calculates the defective rate of worker y given the conditions of the problem -/
theorem worker_y_defective_rate 
  (x_rate : Real) 
  (y_fraction : Real) 
  (total_rate : Real) 
  (hx : x_rate = 0.005) 
  (hy : y_fraction = 0.8) 
  (ht : total_rate = 0.0074) : 
  Real :=
by
  sorry

#check worker_y_defective_rate

end NUMINAMATH_CALUDE_worker_y_defective_rate_l980_98043


namespace NUMINAMATH_CALUDE_sqrt_sum_fraction_simplification_l980_98072

theorem sqrt_sum_fraction_simplification :
  Real.sqrt ((36 : ℝ) / 49 + 16 / 9 + 1 / 16) = 45 / 28 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_fraction_simplification_l980_98072


namespace NUMINAMATH_CALUDE_perpendicular_lines_slope_l980_98070

theorem perpendicular_lines_slope (a : ℝ) : 
  (∀ x y : ℝ, ax + 2*y + 2 = 0 → x - y - 2 = 0 → 
   ((-a/2) * 1 = -1)) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_slope_l980_98070


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l980_98007

theorem quadratic_inequality_solution_set (a : ℝ) :
  (∀ x : ℝ, x^2 - a*x + a > 0) ↔ (0 < a ∧ a < 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l980_98007


namespace NUMINAMATH_CALUDE_tangent_sum_l980_98011

theorem tangent_sum (x y : ℝ) 
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 2)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 4) :
  (Real.tan x / Real.tan y) + (Real.tan y / Real.tan x) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_l980_98011


namespace NUMINAMATH_CALUDE_tangent_line_equation_l980_98060

/-- The curve function -/
def f (x : ℝ) : ℝ := x^3 + 3*x^2 - 5

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x

/-- The point of tangency -/
def p : ℝ × ℝ := (-1, -3)

/-- Theorem: The equation of the tangent line to the curve y = x³ + 3x² - 5
    at the point (-1, -3) is 3x + y + 6 = 0 -/
theorem tangent_line_equation :
  ∀ (x y : ℝ), y = f' p.1 * (x - p.1) + p.2 ↔ 3*x + y + 6 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l980_98060


namespace NUMINAMATH_CALUDE_trajectory_equation_l980_98083

/-- The equation of the trajectory of point P in the xOy plane, given point A at (0,0,4) and |PA| = 5 -/
theorem trajectory_equation :
  ∀ (x y : ℝ),
  let P : ℝ × ℝ × ℝ := (x, y, 0)
  let A : ℝ × ℝ × ℝ := (0, 0, 4)
  (x^2 + y^2 + (0 - 4)^2 = 5^2) →
  (x^2 + y^2 = 9) :=
by
  sorry

end NUMINAMATH_CALUDE_trajectory_equation_l980_98083


namespace NUMINAMATH_CALUDE_other_asymptote_equation_l980_98080

/-- Represents a hyperbola -/
structure Hyperbola where
  /-- One of the asymptotes of the hyperbola -/
  asymptote1 : ℝ → ℝ
  /-- x-coordinate of the foci -/
  foci_x : ℝ

/-- The other asymptote of a hyperbola given one asymptote and foci x-coordinate -/
def other_asymptote (h : Hyperbola) : ℝ → ℝ := sorry

/-- Theorem stating the equation of the other asymptote -/
theorem other_asymptote_equation (h : Hyperbola) 
  (h1 : h.asymptote1 = fun x ↦ 2 * x) 
  (h2 : h.foci_x = 4) : 
  other_asymptote h = fun x ↦ -2 * x + 16 := by sorry

end NUMINAMATH_CALUDE_other_asymptote_equation_l980_98080


namespace NUMINAMATH_CALUDE_multiple_of_sum_and_smaller_l980_98041

theorem multiple_of_sum_and_smaller (s l : ℕ) : 
  s + l = 84 →  -- sum of two numbers is 84
  l = s * (l / s) →  -- one number is a multiple of the other
  s = 21 →  -- the smaller number is 21
  l / s = 3 :=  -- the multiple (ratio) is 3
by
  sorry

end NUMINAMATH_CALUDE_multiple_of_sum_and_smaller_l980_98041


namespace NUMINAMATH_CALUDE_olivias_cans_l980_98016

/-- The number of bags Olivia had -/
def num_bags : ℕ := 4

/-- The number of cans in each bag -/
def cans_per_bag : ℕ := 5

/-- The total number of cans Olivia had -/
def total_cans : ℕ := num_bags * cans_per_bag

theorem olivias_cans : total_cans = 20 := by
  sorry

end NUMINAMATH_CALUDE_olivias_cans_l980_98016


namespace NUMINAMATH_CALUDE_school_workbook_cost_l980_98099

/-- The total cost for purchasing workbooks -/
def total_cost (num_workbooks : ℕ) (cost_per_workbook : ℚ) : ℚ :=
  num_workbooks * cost_per_workbook

/-- Theorem: The total cost for the school to purchase 400 workbooks, each costing x yuan, is equal to 400x yuan -/
theorem school_workbook_cost (x : ℚ) : 
  total_cost 400 x = 400 * x := by
  sorry

end NUMINAMATH_CALUDE_school_workbook_cost_l980_98099


namespace NUMINAMATH_CALUDE_archer_weekly_expenditure_l980_98094

def archer_expenditure (shots_per_day : ℕ) (days_per_week : ℕ) (recovery_rate : ℚ) 
  (arrow_cost : ℚ) (team_contribution_rate : ℚ) : ℚ :=
  let total_shots := shots_per_day * days_per_week
  let recovered_arrows := total_shots * recovery_rate
  let net_arrows_used := total_shots - recovered_arrows
  let total_cost := net_arrows_used * arrow_cost
  let archer_cost := total_cost * (1 - team_contribution_rate)
  archer_cost

theorem archer_weekly_expenditure :
  archer_expenditure 200 4 (1/5) (11/2) (7/10) = 1056 := by
  sorry

end NUMINAMATH_CALUDE_archer_weekly_expenditure_l980_98094


namespace NUMINAMATH_CALUDE_students_taking_neither_proof_l980_98057

def students_taking_neither (total students_music students_art students_dance
                             students_music_art students_art_dance students_music_dance
                             students_all_three : ℕ) : ℕ :=
  total - (students_music + students_art + students_dance
           - students_music_art - students_art_dance - students_music_dance
           + students_all_three)

theorem students_taking_neither_proof :
  students_taking_neither 2500 200 150 100 75 50 40 25 = 2190 := by
  sorry

end NUMINAMATH_CALUDE_students_taking_neither_proof_l980_98057


namespace NUMINAMATH_CALUDE_number_puzzle_l980_98053

theorem number_puzzle : 
  ∃ x : ℝ, (x / 5 + 4 = x / 4 - 4) ∧ (x = 160) := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l980_98053


namespace NUMINAMATH_CALUDE_sum_of_squares_problem_l980_98037

theorem sum_of_squares_problem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c)
  (h4 : a^2 + b^2 + c^2 = 75) (h5 : a*b + b*c + c*a = 40) (h6 : c = 5) :
  a + b + c = 5 * Real.sqrt 62 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_problem_l980_98037


namespace NUMINAMATH_CALUDE_arithmetic_sequence_equality_l980_98068

theorem arithmetic_sequence_equality (N : ℕ) : 
  (3 + 4 + 5 + 6 + 7) / 5 = (1993 + 1994 + 1995 + 1996 + 1997) / N → N = 1995 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_equality_l980_98068


namespace NUMINAMATH_CALUDE_angle_range_for_positive_quadratic_l980_98086

open Real

theorem angle_range_for_positive_quadratic (θ : ℝ) : 
  (0 < θ) → (θ < π) →
  (∀ x : ℝ, (cos θ) * x^2 - 4 * (sin θ) * x + 6 > 0) →
  (0 < θ) ∧ (θ < π/3) := by
sorry

end NUMINAMATH_CALUDE_angle_range_for_positive_quadratic_l980_98086


namespace NUMINAMATH_CALUDE_clock_90_degree_times_l980_98009

/-- The angle between the hour hand and minute hand at time t minutes after 12:00 -/
def angle_between (t : ℝ) : ℝ :=
  |6 * t - 0.5 * t|

/-- The times when the hour hand and minute hand form a 90° angle after 12:00 -/
theorem clock_90_degree_times :
  ∃ (t₁ t₂ : ℝ), t₁ < t₂ ∧
  angle_between t₁ = 90 ∧
  angle_between t₂ = 90 ∧
  t₁ = 180 / 11 ∧
  t₂ = 540 / 11 :=
sorry

end NUMINAMATH_CALUDE_clock_90_degree_times_l980_98009


namespace NUMINAMATH_CALUDE_square_sum_equals_45_l980_98000

theorem square_sum_equals_45 (x y : ℝ) (h1 : x + 3*y = 3) (h2 : x*y = -6) :
  x^2 + 9*y^2 = 45 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_45_l980_98000


namespace NUMINAMATH_CALUDE_locus_of_P_B_l980_98014

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define points
def Point := ℝ × ℝ

-- Define the given circle and points
variable (c : Circle)
variable (A : Point)
variable (B : Point)

-- Define P_B as a function of B
def P_B (B : Point) : Point := sorry

-- Define the condition that A and B are on the circle
def on_circle (p : Point) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define the condition that B is not on line OA
def not_on_line_OA (B : Point) (c : Circle) (A : Point) : Prop := sorry

-- Define the condition that P_B is on the internal bisector of ∠AOB
def on_internal_bisector (P : Point) (O : Point) (A : Point) (B : Point) : Prop := sorry

-- State the theorem
theorem locus_of_P_B (c : Circle) (A B : Point) 
  (h1 : on_circle A c)
  (h2 : on_circle B c)
  (h3 : not_on_line_OA B c A)
  (h4 : on_internal_bisector (P_B B) c.center A B) :
  ∃ (r : ℝ), ∀ B, on_circle (P_B B) { center := c.center, radius := r } :=
sorry

end NUMINAMATH_CALUDE_locus_of_P_B_l980_98014


namespace NUMINAMATH_CALUDE_sqrt3_cos_minus_sin_eq_sqrt2_l980_98082

theorem sqrt3_cos_minus_sin_eq_sqrt2 :
  Real.sqrt 3 * Real.cos (π / 12) - Real.sin (π / 12) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt3_cos_minus_sin_eq_sqrt2_l980_98082


namespace NUMINAMATH_CALUDE_strongest_signal_l980_98049

def signal_strength (x : ℤ) : ℝ := |x|

def is_stronger (x y : ℤ) : Prop := signal_strength x < signal_strength y

theorem strongest_signal :
  let signals : List ℤ := [-50, -60, -70, -80]
  ∀ s ∈ signals, s ≠ -50 → is_stronger (-50) s :=
sorry

end NUMINAMATH_CALUDE_strongest_signal_l980_98049


namespace NUMINAMATH_CALUDE_smallest_n_digits_l980_98062

/-- Sum of digits function -/
def sum_of_digits (k : ℕ) : ℕ := sorry

/-- Theorem stating the number of digits in the smallest n satisfying the condition -/
theorem smallest_n_digits :
  ∃ n : ℕ,
    (∀ m : ℕ, m < n → sum_of_digits m - sum_of_digits (5 * m) ≠ 2013) ∧
    (sum_of_digits n - sum_of_digits (5 * n) = 2013) ∧
    (Nat.digits 10 n).length = 224 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_digits_l980_98062


namespace NUMINAMATH_CALUDE_proposition_truth_l980_98058

-- Define the propositions
def proposition1 (m : ℝ) : Prop := m > 0 ↔ ∃ (x y : ℝ), x^2 + m*y^2 = 1 ∧ ¬(x^2 + y^2 = 1)

def proposition2 (a : ℝ) : Prop := (a = 1 → ∃ (k : ℝ), ∀ (x y : ℝ), a*x + y - 1 = k*(x + a*y - 2)) ∧
                                   ¬(∀ (a : ℝ), a = 1 → ∃ (k : ℝ), ∀ (x y : ℝ), a*x + y - 1 = k*(x + a*y - 2))

def proposition3 (m : ℝ) : Prop := (∀ (x₁ x₂ : ℝ), x₁ < x₂ → x₁^3 + m*x₁ < x₂^3 + m*x₂) ↔ m > 0

def proposition4 (p q : Prop) : Prop := ((p ∨ q) → (p ∧ q)) ∧ ((p ∧ q) → (p ∨ q))

-- Theorem stating which propositions are true and which are false
theorem proposition_truth : 
  (∃ (m : ℝ), ¬proposition1 m) ∧ 
  (∀ (a : ℝ), proposition2 a) ∧
  (∃ (m : ℝ), ¬proposition3 m) ∧
  (∀ (p q : Prop), proposition4 p q) := by
  sorry

end NUMINAMATH_CALUDE_proposition_truth_l980_98058


namespace NUMINAMATH_CALUDE_sigma_phi_bounds_l980_98024

open Nat Real

/-- The sum of divisors function -/
noncomputable def sigma (n : ℕ) : ℕ := sorry

/-- Euler's totient function -/
noncomputable def phi (n : ℕ) : ℕ := sorry

theorem sigma_phi_bounds (n : ℕ) (h : n > 0) : 
  (sigma n * phi n : ℝ) < n^2 ∧ 
  ∃ c : ℝ, c > 0 ∧ ∀ m : ℕ, m > 0 → (sigma m * phi m : ℝ) ≥ c * m^2 := by
  sorry

end NUMINAMATH_CALUDE_sigma_phi_bounds_l980_98024


namespace NUMINAMATH_CALUDE_trig_identity_l980_98092

theorem trig_identity (α : Real) : 
  Real.sin α ^ 2 + Real.cos (π / 6 - α) ^ 2 - Real.sin α * Real.cos (π / 6 - α) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l980_98092


namespace NUMINAMATH_CALUDE_wages_problem_l980_98051

/-- Given a sum of money that can pay x's wages for 36 days and y's wages for 45 days,
    prove that it can pay both x and y's wages together for 20 days. -/
theorem wages_problem (S : ℝ) (x y : ℝ → ℝ) :
  (∃ (Wx Wy : ℝ), Wx > 0 ∧ Wy > 0 ∧ S = 36 * Wx ∧ S = 45 * Wy) →
  ∃ D : ℝ, D = 20 ∧ S = D * (x 1 + y 1) :=
by sorry

end NUMINAMATH_CALUDE_wages_problem_l980_98051


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l980_98046

theorem polynomial_division_quotient : 
  let dividend := fun x : ℚ => 10 * x^3 - 5 * x^2 + 8 * x - 9
  let divisor := fun x : ℚ => 3 * x - 4
  let quotient := fun x : ℚ => (10/3) * x^2 - (55/9) * x - 172/27
  ∀ x : ℚ, dividend x = divisor x * quotient x + (-971/27) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l980_98046


namespace NUMINAMATH_CALUDE_probability_at_least_two_correct_l980_98097

-- Define the number of questions and choices
def total_questions : ℕ := 30
def choices_per_question : ℕ := 6
def guessed_questions : ℕ := 5

-- Define the probability of a correct answer
def p_correct : ℚ := 1 / choices_per_question

-- Define the binomial probability function
def binomial_prob (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

-- State the theorem
theorem probability_at_least_two_correct :
  1 - binomial_prob guessed_questions 0 p_correct
    - binomial_prob guessed_questions 1 p_correct = 763 / 3888 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_two_correct_l980_98097


namespace NUMINAMATH_CALUDE_largest_base6_4digit_in_base10_l980_98002

def largest_base6_4digit : ℕ := 5 * 6^3 + 5 * 6^2 + 5 * 6^1 + 5 * 6^0

theorem largest_base6_4digit_in_base10 : 
  largest_base6_4digit = 1295 := by sorry

end NUMINAMATH_CALUDE_largest_base6_4digit_in_base10_l980_98002


namespace NUMINAMATH_CALUDE_percentage_of_b_l980_98028

theorem percentage_of_b (a b c : ℝ) (h1 : 12 = 0.04 * a) (h2 : ∃ p, p * b = 4) (h3 : c = b / a) :
  ∃ p, p * b = 4 ∧ p = 4 / (c * 300) := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_b_l980_98028


namespace NUMINAMATH_CALUDE_probability_at_least_one_male_l980_98084

theorem probability_at_least_one_male (male_count female_count : ℕ) 
  (h1 : male_count = 3) (h2 : female_count = 2) : 
  1 - (Nat.choose female_count 2 : ℚ) / (Nat.choose (male_count + female_count) 2 : ℚ) = 9/10 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_male_l980_98084


namespace NUMINAMATH_CALUDE_evaluate_expression_l980_98044

theorem evaluate_expression : (0.5 ^ 4) / (0.05 ^ 3) = 500 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l980_98044


namespace NUMINAMATH_CALUDE_percentage_of_50_to_125_l980_98030

theorem percentage_of_50_to_125 : 
  (50 : ℝ) / 125 * 100 = 40 :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_50_to_125_l980_98030


namespace NUMINAMATH_CALUDE_proposition_c_is_true_l980_98019

theorem proposition_c_is_true : ∀ x y : ℝ, x + y ≠ 3 → x ≠ 2 ∨ y ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_proposition_c_is_true_l980_98019


namespace NUMINAMATH_CALUDE_min_bottles_to_fill_l980_98020

theorem min_bottles_to_fill (small_capacity large_capacity : ℕ) 
  (h1 : small_capacity = 40)
  (h2 : large_capacity = 360) : 
  Nat.ceil (large_capacity / small_capacity) = 9 := by
  sorry

#check min_bottles_to_fill

end NUMINAMATH_CALUDE_min_bottles_to_fill_l980_98020


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l980_98042

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, -x^2 + 2*x + 3 ≥ 0 ↔ x ∈ Set.Icc (-1) 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l980_98042


namespace NUMINAMATH_CALUDE_median_squares_sum_l980_98003

/-- For a triangle with sides a, b, c, medians m_a, m_b, m_c, and circumcircle diameter D,
    the sum of squares of medians equals 3/4 of the sum of squares of sides plus 3/4 of the square of the diameter. -/
theorem median_squares_sum (a b c m_a m_b m_c D : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_m_a : 0 < m_a) (h_pos_m_b : 0 < m_b) (h_pos_m_c : 0 < m_c)
  (h_pos_D : 0 < D)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_median_a : 4 * m_a^2 + a^2 = 2 * b^2 + 2 * c^2)
  (h_median_b : 4 * m_b^2 + b^2 = 2 * c^2 + 2 * a^2)
  (h_median_c : 4 * m_c^2 + c^2 = 2 * a^2 + 2 * b^2)
  (h_D : D ≥ max a (max b c)) :
  m_a^2 + m_b^2 + m_c^2 = 3/4 * (a^2 + b^2 + c^2) + 3/4 * D^2 :=
sorry

end NUMINAMATH_CALUDE_median_squares_sum_l980_98003


namespace NUMINAMATH_CALUDE_P_intersect_Q_equals_target_l980_98091

-- Define the sets P and Q
def P : Set ℝ := {x | ∃ y, y = Real.log (2 - x)}
def Q : Set ℝ := {x | x^2 - 5*x + 4 ≤ 0}

-- State the theorem
theorem P_intersect_Q_equals_target : P ∩ Q = {x | 1 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_P_intersect_Q_equals_target_l980_98091


namespace NUMINAMATH_CALUDE_proportional_enlargement_l980_98017

/-- Proportional enlargement of a rectangle -/
theorem proportional_enlargement (original_width original_height new_width : ℝ) 
  (h1 : original_width > 0)
  (h2 : original_height > 0)
  (h3 : new_width > 0) :
  let scale_factor := new_width / original_width
  let new_height := original_height * scale_factor
  (original_width = 3 ∧ original_height = 2 ∧ new_width = 12) → new_height = 8 := by
sorry

end NUMINAMATH_CALUDE_proportional_enlargement_l980_98017


namespace NUMINAMATH_CALUDE_unique_solution_l980_98006

theorem unique_solution : ∃! (n : ℕ+), 
  Real.sin (π / (3 * n.val : ℝ)) + Real.cos (π / (3 * n.val : ℝ)) = Real.sqrt (2 * n.val : ℝ) / 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l980_98006


namespace NUMINAMATH_CALUDE_monkey_climb_l980_98098

/-- Proves that a monkey slips back 2 feet per hour when climbing a 17 ft tree in 15 hours, 
    climbing 3 ft and slipping back a constant distance each hour. -/
theorem monkey_climb (tree_height : ℝ) (total_hours : ℕ) (climb_rate : ℝ) (slip_back : ℝ) : 
  tree_height = 17 →
  total_hours = 15 →
  climb_rate = 3 →
  (total_hours - 1 : ℝ) * (climb_rate - slip_back) + climb_rate = tree_height →
  slip_back = 2 := by
  sorry

#check monkey_climb

end NUMINAMATH_CALUDE_monkey_climb_l980_98098


namespace NUMINAMATH_CALUDE_mirabel_candy_distribution_l980_98066

theorem mirabel_candy_distribution :
  ∃ (k : ℕ), k = 2 ∧ 
  (∀ (j : ℕ), j < k → ¬∃ (n : ℕ), 10 ≤ n ∧ n < 20 ∧ (47 - j) % n = 0) ∧
  (∃ (n : ℕ), 10 ≤ n ∧ n < 20 ∧ (47 - k) % n = 0) :=
by sorry

end NUMINAMATH_CALUDE_mirabel_candy_distribution_l980_98066


namespace NUMINAMATH_CALUDE_quadratic_one_solution_sum_l980_98085

theorem quadratic_one_solution_sum (b₁ b₂ : ℝ) : 
  (∀ x, 3 * x^2 + b₁ * x + 5 * x + 7 = 0 → (∀ y, 3 * y^2 + b₁ * y + 5 * y + 7 = 0 → x = y)) ∧
  (∀ x, 3 * x^2 + b₂ * x + 5 * x + 7 = 0 → (∀ y, 3 * y^2 + b₂ * y + 5 * y + 7 = 0 → x = y)) ∧
  (∀ b, (∀ x, 3 * x^2 + b * x + 5 * x + 7 = 0 → (∀ y, 3 * y^2 + b * y + 5 * y + 7 = 0 → x = y)) → b = b₁ ∨ b = b₂) →
  b₁ + b₂ = -10 :=
sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_sum_l980_98085


namespace NUMINAMATH_CALUDE_specific_ellipse_major_axis_l980_98071

/-- An ellipse with specific properties -/
structure Ellipse where
  -- The ellipse is tangent to both x-axis and y-axis
  tangent_to_axes : Bool
  -- The x-coordinate of both foci
  focus_x : ℝ
  -- The y-coordinates of the foci
  focus_y1 : ℝ
  focus_y2 : ℝ

/-- The length of the major axis of the ellipse -/
def major_axis_length (e : Ellipse) : ℝ :=
  sorry

/-- Theorem stating the length of the major axis for a specific ellipse -/
theorem specific_ellipse_major_axis :
  ∃ (e : Ellipse), 
    e.tangent_to_axes = true ∧
    e.focus_x = 3 ∧
    e.focus_y1 = -4 + 2 * Real.sqrt 2 ∧
    e.focus_y2 = -4 - 2 * Real.sqrt 2 ∧
    major_axis_length e = 8 :=
  sorry

end NUMINAMATH_CALUDE_specific_ellipse_major_axis_l980_98071


namespace NUMINAMATH_CALUDE_not_p_and_not_q_true_l980_98029

theorem not_p_and_not_q_true (p q : Prop) 
  (h1 : ¬(p ∧ q)) 
  (h2 : ¬(p ∨ q)) : 
  (¬p ∧ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_not_p_and_not_q_true_l980_98029


namespace NUMINAMATH_CALUDE_peter_distance_l980_98076

/-- The total distance Peter covers -/
def D : ℝ := sorry

/-- The time Peter takes to cover the distance in hours -/
def total_time : ℝ := 1.4

/-- The speed at which Peter covers two-thirds of the distance -/
def speed1 : ℝ := 4

/-- The speed at which Peter covers one-third of the distance -/
def speed2 : ℝ := 5

theorem peter_distance : 
  (2/3 * D) / speed1 + (1/3 * D) / speed2 = total_time ∧ D = 6 := by sorry

end NUMINAMATH_CALUDE_peter_distance_l980_98076


namespace NUMINAMATH_CALUDE_least_addend_for_divisibility_problem_solution_l980_98063

theorem least_addend_for_divisibility (n : Nat) (d : Nat) (h : d > 0) :
  ∃ (x : Nat), x < d ∧ (n + x) % d = 0 ∧ ∀ (y : Nat), y < x → (n + y) % d ≠ 0 :=
by sorry

theorem problem_solution :
  ∃ (x : Nat), x = 19 ∧ (1156 + x) % 25 = 0 ∧ ∀ (y : Nat), y < x → (1156 + y) % 25 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_addend_for_divisibility_problem_solution_l980_98063
