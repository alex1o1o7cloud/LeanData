import Mathlib

namespace NUMINAMATH_CALUDE_smaller_number_problem_l1946_194654

theorem smaller_number_problem (x y : ℤ) : 
  y = 2 * x - 3 →  -- One number is 3 less than twice another
  x + y = 39 →     -- The sum of the two numbers is 39
  x = 14           -- The smaller number is 14
  := by sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l1946_194654


namespace NUMINAMATH_CALUDE_largest_three_digit_perfect_square_diff_l1946_194670

/-- A function that returns the sum of digits of a natural number. -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number is a three-digit number. -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- The main theorem stating that 919 is the largest three-digit number
    such that the number minus the sum of its digits is a perfect square. -/
theorem largest_three_digit_perfect_square_diff :
  ∀ n : ℕ, is_three_digit n →
    (∃ k : ℕ, n - sum_of_digits n = k^2) →
    n ≤ 919 := by sorry

end NUMINAMATH_CALUDE_largest_three_digit_perfect_square_diff_l1946_194670


namespace NUMINAMATH_CALUDE_equation_solution_l1946_194639

theorem equation_solution : ∃ n : ℝ, 0.03 * n + 0.08 * (20 + n) = 12.6 ∧ n = 100 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1946_194639


namespace NUMINAMATH_CALUDE_fourth_cat_weight_proof_l1946_194617

/-- The weight of the fourth cat given the weights of three cats and the average weight of all four cats -/
def fourth_cat_weight (weight1 weight2 weight3 average_weight : ℝ) : ℝ :=
  4 * average_weight - (weight1 + weight2 + weight3)

/-- Theorem stating that given the specific weights of three cats and the average weight of all four cats, the weight of the fourth cat is 9.3 pounds -/
theorem fourth_cat_weight_proof :
  fourth_cat_weight 12 12 14.7 12 = 9.3 := by
  sorry

end NUMINAMATH_CALUDE_fourth_cat_weight_proof_l1946_194617


namespace NUMINAMATH_CALUDE_five_topping_pizzas_l1946_194631

theorem five_topping_pizzas (n : Nat) (k : Nat) (h1 : n = 8) (h2 : k = 5) :
  Nat.choose n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_five_topping_pizzas_l1946_194631


namespace NUMINAMATH_CALUDE_total_time_circling_island_l1946_194600

-- Define the problem parameters
def time_per_round : ℕ := 30
def saturday_rounds : ℕ := 11
def sunday_rounds : ℕ := 15

-- State the theorem
theorem total_time_circling_island : 
  (saturday_rounds + sunday_rounds) * time_per_round = 780 := by
  sorry

end NUMINAMATH_CALUDE_total_time_circling_island_l1946_194600


namespace NUMINAMATH_CALUDE_ring_arrangements_l1946_194686

theorem ring_arrangements (n k f : ℕ) (h1 : n = 10) (h2 : k = 7) (h3 : f = 5) :
  let m := (n.choose k) * k.factorial * ((k + f - 1).choose (f - 1))
  (m / 100000000 : ℕ) = 199 :=
by sorry

end NUMINAMATH_CALUDE_ring_arrangements_l1946_194686


namespace NUMINAMATH_CALUDE_intersection_shape_circumference_l1946_194635

/-- The circumference of the shape formed by intersecting quarter circles in a square -/
theorem intersection_shape_circumference (π : ℝ) (side_length : ℝ) : 
  π = 3.141 → side_length = 2 → (4 * π) / 3 = 4.188 := by sorry

end NUMINAMATH_CALUDE_intersection_shape_circumference_l1946_194635


namespace NUMINAMATH_CALUDE_solution_set_theorem_range_of_m_l1946_194698

-- Define the function f(x) = |x - 2|
def f (x : ℝ) : ℝ := |x - 2|

-- Theorem for the solution set of f(x) + f(2x + 1) ≥ 6
theorem solution_set_theorem (x : ℝ) :
  f x + f (2 * x + 1) ≥ 6 ↔ x ∈ Set.Iic (-1) ∪ Set.Ici 3 :=
sorry

-- Theorem for the range of m
theorem range_of_m (a b m : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ x : ℝ, f (x - m) - (-x) ≤ 4/a + 1/b) →
  -13 ≤ m ∧ m ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_solution_set_theorem_range_of_m_l1946_194698


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l1946_194607

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (contains : Plane → Line → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (m n : Line) (α β : Plane) :
  perpendicular m α → 
  parallel m n → 
  contains β n → 
  perpendicularPlanes α β :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l1946_194607


namespace NUMINAMATH_CALUDE_function_passes_through_point_l1946_194616

theorem function_passes_through_point (a : ℝ) (h : 0 < a ∧ a < 1) :
  let f : ℝ → ℝ := λ x ↦ 2 * a^(x - 1)
  f 1 = 2 := by sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l1946_194616


namespace NUMINAMATH_CALUDE_lowest_divisible_by_one_and_two_l1946_194684

theorem lowest_divisible_by_one_and_two : 
  ∃ n : ℕ+, (∀ k : ℕ, 1 ≤ k ∧ k ≤ 2 → k ∣ n) ∧ 
  (∀ m : ℕ+, (∀ k : ℕ, 1 ≤ k ∧ k ≤ 2 → k ∣ m) → n ≤ m) :=
by sorry

end NUMINAMATH_CALUDE_lowest_divisible_by_one_and_two_l1946_194684


namespace NUMINAMATH_CALUDE_one_and_half_times_product_of_digits_l1946_194669

/-- Function to calculate the product of digits of a natural number -/
def productOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem stating that 48 and 0 are the only natural numbers that are 1.5 times the product of their digits -/
theorem one_and_half_times_product_of_digits :
  ∀ (A : ℕ), A = (3 / 2 : ℚ) * (productOfDigits A) ↔ A = 48 ∨ A = 0 := by sorry

end NUMINAMATH_CALUDE_one_and_half_times_product_of_digits_l1946_194669


namespace NUMINAMATH_CALUDE_point_not_on_transformed_plane_l1946_194668

/-- The similarity transformation coefficient -/
def k : ℚ := 5/2

/-- The original plane equation: x + y - 2z + 2 = 0 -/
def plane_a (x y z : ℚ) : Prop := x + y - 2*z + 2 = 0

/-- The transformed plane equation: x + y - 2z + 5 = 0 -/
def plane_a_transformed (x y z : ℚ) : Prop := x + y - 2*z + 5 = 0

/-- Point A -/
def point_A : ℚ × ℚ × ℚ := (2, -3, 1)

/-- Theorem: Point A does not belong to the image of plane a after similarity transformation -/
theorem point_not_on_transformed_plane :
  ¬ plane_a_transformed point_A.1 point_A.2.1 point_A.2.2 :=
by sorry

end NUMINAMATH_CALUDE_point_not_on_transformed_plane_l1946_194668


namespace NUMINAMATH_CALUDE_problem_solution_l1946_194653

open Set

def U : Set ℝ := univ
def A : Set ℝ := {x | -1 < x ∧ x < 5}
def B : Set ℝ := {x | 2 < x ∧ x < 8}

theorem problem_solution (a b : ℝ) : 
  (∃ C : Set ℝ, C = {x | a + 1 ≤ x ∧ x ≤ 2*a - 2} ∧ 
   (Aᶜ ∩ C = {x | 6 ≤ x ∧ x ≤ b})) → a + b = 13 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1946_194653


namespace NUMINAMATH_CALUDE_three_digit_sum_property_l1946_194637

def digit_sum (n : ℕ) : ℕ := 
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem three_digit_sum_property (n : ℕ) :
  100 ≤ n ∧ n < 1000 ∧ 
  digit_sum n = 3 * digit_sum (n - 75) →
  n = 189 ∨ n = 675 := by
sorry

end NUMINAMATH_CALUDE_three_digit_sum_property_l1946_194637


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1946_194679

theorem inequality_solution_range (k : ℝ) : 
  (1 : ℝ)^2 * k^2 - 6 * k * (1 : ℝ) + 8 ≥ 0 → k ≤ 2 ∨ k ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1946_194679


namespace NUMINAMATH_CALUDE_equation_solution_l1946_194690

theorem equation_solution : ∃! x : ℝ, 3 * x + 1 = x - 3 :=
  by
    use -2
    constructor
    · -- Prove that -2 satisfies the equation
      sorry
    · -- Prove uniqueness
      sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l1946_194690


namespace NUMINAMATH_CALUDE_expression_evaluation_l1946_194628

theorem expression_evaluation : 7^2 - 4^2 + 2*5 - 3^3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1946_194628


namespace NUMINAMATH_CALUDE_divisibility_by_power_of_five_l1946_194691

theorem divisibility_by_power_of_five :
  ∀ k : ℕ, ∃ n : ℕ, (5^k : ℕ) ∣ (n^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_power_of_five_l1946_194691


namespace NUMINAMATH_CALUDE_range_of_a_l1946_194641

def p (x : ℝ) := |4*x - 3| ≤ 1

def q (x a : ℝ) := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

theorem range_of_a :
  (∀ x, q x a → p x) ∧
  (∃ x, p x ∧ ¬q x a) →
  a ∈ Set.Icc (0 : ℝ) (1/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1946_194641


namespace NUMINAMATH_CALUDE_perfect_cubes_between_powers_of_three_l1946_194612

theorem perfect_cubes_between_powers_of_three : 
  let lower_bound := 3^6 + 1
  let upper_bound := 3^12 + 1
  (Finset.filter (fun n => lower_bound ≤ n^3 ∧ n^3 ≤ upper_bound) 
    (Finset.range (upper_bound + 1))).card = 72 := by
  sorry

end NUMINAMATH_CALUDE_perfect_cubes_between_powers_of_three_l1946_194612


namespace NUMINAMATH_CALUDE_line_inclination_angle_l1946_194603

/-- The inclination angle of a line given by parametric equations -/
def inclinationAngle (x y : ℝ → ℝ) : ℝ := sorry

/-- Cosine of 20 degrees -/
def cos20 : ℝ := sorry

/-- Sine of 20 degrees -/
def sin20 : ℝ := sorry

theorem line_inclination_angle :
  let x : ℝ → ℝ := λ t => -t * cos20
  let y : ℝ → ℝ := λ t => 3 + t * sin20
  inclinationAngle x y = 160 * π / 180 := by sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l1946_194603


namespace NUMINAMATH_CALUDE_factorial_fraction_l1946_194678

theorem factorial_fraction (N : ℕ) (h : N > 2) :
  (Nat.factorial (N - 2) * (N - 1)) / Nat.factorial N = 1 / N := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_l1946_194678


namespace NUMINAMATH_CALUDE_sum_of_bases_is_fifteen_l1946_194695

/-- Represents a fraction in a given base --/
structure FractionInBase where
  numerator : ℕ
  denominator : ℕ
  base : ℕ

/-- Converts a repeating decimal to a fraction --/
def repeatingDecimalToFraction (digits : ℕ) (base : ℕ) : FractionInBase :=
  { numerator := digits,
    denominator := base^2 - 1,
    base := base }

theorem sum_of_bases_is_fifteen :
  let R₁ : ℕ := 9
  let R₂ : ℕ := 6
  let F₁_in_R₁ := repeatingDecimalToFraction 48 R₁
  let F₂_in_R₁ := repeatingDecimalToFraction 84 R₁
  let F₁_in_R₂ := repeatingDecimalToFraction 35 R₂
  let F₂_in_R₂ := repeatingDecimalToFraction 53 R₂
  R₁ + R₂ = 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_bases_is_fifteen_l1946_194695


namespace NUMINAMATH_CALUDE_pentagon_reassembly_l1946_194664

/-- Given a 10x15 rectangle cut into two congruent pentagons and reassembled into a larger rectangle,
    prove that one-third of the longer side of the new rectangle is 5√2. -/
theorem pentagon_reassembly (original_length original_width : ℝ) 
                            (new_length new_width : ℝ) (y : ℝ) : 
  original_length = 10 →
  original_width = 15 →
  new_length * new_width = original_length * original_width →
  y = new_length / 3 →
  y = 5 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_pentagon_reassembly_l1946_194664


namespace NUMINAMATH_CALUDE_line_through_two_points_l1946_194680

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The line equation derived from two points -/
def lineEquation (p₁ p₂ : Point) (p : Point) : Prop :=
  (p.x - p₁.x) * (p₂.y - p₁.y) = (p.y - p₁.y) * (p₂.x - p₁.x)

theorem line_through_two_points (p₁ p₂ : Point) (h : p₁ ≠ p₂) :
  ∃! l : Line, Point.onLine p₁ l ∧ Point.onLine p₂ l ∧
  ∀ p, Point.onLine p l ↔ lineEquation p₁ p₂ p :=
sorry

end NUMINAMATH_CALUDE_line_through_two_points_l1946_194680


namespace NUMINAMATH_CALUDE_system_solution_characterization_l1946_194633

/-- The system of equations has either a unique solution or infinitely many solutions when m ≠ -1 -/
theorem system_solution_characterization (m : ℝ) (hm : m ≠ -1) :
  (∃! x y : ℝ, m * x + y = m + 1 ∧ x + m * y = 2 * m) ∨
  (∃ f g : ℝ → ℝ, ∀ t : ℝ, m * (f t) + (g t) = m + 1 ∧ (f t) + m * (g t) = 2 * m) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_characterization_l1946_194633


namespace NUMINAMATH_CALUDE_doughnut_machine_completion_time_l1946_194674

-- Define the start time (6:00 AM) in minutes since midnight
def start_time : ℕ := 6 * 60

-- Define the time when one-fourth of the job is completed (9:00 AM) in minutes since midnight
def quarter_completion_time : ℕ := 9 * 60

-- Define the maintenance stop duration in minutes
def maintenance_duration : ℕ := 45

-- Define the completion time (6:45 PM) in minutes since midnight
def completion_time : ℕ := 18 * 60 + 45

-- Theorem statement
theorem doughnut_machine_completion_time :
  let working_duration := quarter_completion_time - start_time
  let total_duration := working_duration * 4 + maintenance_duration
  start_time + total_duration = completion_time :=
sorry

end NUMINAMATH_CALUDE_doughnut_machine_completion_time_l1946_194674


namespace NUMINAMATH_CALUDE_interior_angles_increase_l1946_194673

/-- The sum of interior angles of a convex polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

theorem interior_angles_increase (n : ℕ) :
  sum_interior_angles n = 1620 → sum_interior_angles (n + 3) = 2160 := by
  sorry

end NUMINAMATH_CALUDE_interior_angles_increase_l1946_194673


namespace NUMINAMATH_CALUDE_basketball_team_callbacks_l1946_194675

theorem basketball_team_callbacks (girls_tryout : ℕ) (boys_tryout : ℕ) (didnt_make_cut : ℕ) :
  girls_tryout = 9 →
  boys_tryout = 14 →
  didnt_make_cut = 21 →
  girls_tryout + boys_tryout - didnt_make_cut = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_team_callbacks_l1946_194675


namespace NUMINAMATH_CALUDE_football_throw_distance_l1946_194676

theorem football_throw_distance (parker_distance grant_distance kyle_distance : ℝ) :
  parker_distance = 16 ∧
  grant_distance = parker_distance * 1.25 ∧
  kyle_distance = grant_distance * 2 →
  kyle_distance - parker_distance = 24 := by
  sorry

end NUMINAMATH_CALUDE_football_throw_distance_l1946_194676


namespace NUMINAMATH_CALUDE_hockey_players_l1946_194618

theorem hockey_players (n : ℕ) : 
  n < 30 ∧ 
  2 ∣ n ∧ 
  4 ∣ n ∧ 
  7 ∣ n → 
  n / 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_hockey_players_l1946_194618


namespace NUMINAMATH_CALUDE_parabola_translation_l1946_194609

/-- Represents a parabola in the form y = ax^2 + bx + c --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically --/
def translate (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
  , b := 2 * p.a * h + p.b
  , c := p.a * h^2 + p.b * h + p.c + v }

theorem parabola_translation (x y : ℝ) :
  let p := Parabola.mk 1 4 (-4)
  let p_translated := translate p 2 (-3)
  y = x^2 + 4*x - 4 →
  y = (x + 4)^2 - 11 ↔
  y = p_translated.a * x^2 + p_translated.b * x + p_translated.c :=
by sorry

end NUMINAMATH_CALUDE_parabola_translation_l1946_194609


namespace NUMINAMATH_CALUDE_car_speed_problem_l1946_194644

/-- The speed of Car B in km/h -/
def speed_B : ℝ := 35

/-- The time it takes Car A to catch up with Car B when traveling at 50 km/h -/
def time_50 : ℝ := 6

/-- The time it takes Car A to catch up with Car B when traveling at 80 km/h -/
def time_80 : ℝ := 2

/-- The speed of Car A in the first scenario (km/h) -/
def speed_A1 : ℝ := 50

/-- The speed of Car A in the second scenario (km/h) -/
def speed_A2 : ℝ := 80

theorem car_speed_problem :
  (speed_A1 * time_50 - speed_B * time_50 = speed_A2 * time_80 - speed_B * time_80) ∧
  speed_B = 35 := by sorry

end NUMINAMATH_CALUDE_car_speed_problem_l1946_194644


namespace NUMINAMATH_CALUDE_total_groups_is_1026_l1946_194663

/-- The number of boys in the class -/
def num_boys : ℕ := 9

/-- The number of girls in the class -/
def num_girls : ℕ := 12

/-- The size of each group -/
def group_size : ℕ := 3

/-- Calculate the number of combinations of n items taken k at a time -/
def combinations (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.choose n k

/-- Calculate the number of groups with 2 boys and 1 girl -/
def groups_2boys1girl : ℕ :=
  combinations num_boys 2 * combinations num_girls 1

/-- Calculate the number of groups with 2 girls and 1 boy -/
def groups_2girls1boy : ℕ :=
  combinations num_girls 2 * combinations num_boys 1

/-- The total number of possible groups -/
def total_groups : ℕ :=
  groups_2boys1girl + groups_2girls1boy

/-- Theorem stating that the total number of possible groups is 1026 -/
theorem total_groups_is_1026 : total_groups = 1026 := by
  sorry

end NUMINAMATH_CALUDE_total_groups_is_1026_l1946_194663


namespace NUMINAMATH_CALUDE_sqrt3_div3_sufficient_sqrt3_div3_not_necessary_sqrt3_div3_sufficient_not_necessary_l1946_194659

/-- The condition for a line to be tangent to a circle --/
def is_tangent (k : ℝ) : Prop :=
  let line := fun x => k * (x + 2)
  let circle := fun x y => x^2 + y^2 = 1
  ∃ x y, circle x y ∧ y = line x ∧
  ∀ x' y', circle x' y' → (y' - line x')^2 ≥ 0

/-- k = √3/3 is sufficient for tangency --/
theorem sqrt3_div3_sufficient :
  is_tangent (Real.sqrt 3 / 3) := by sorry

/-- k = √3/3 is not necessary for tangency --/
theorem sqrt3_div3_not_necessary :
  ∃ k, k ≠ Real.sqrt 3 / 3 ∧ is_tangent k := by sorry

/-- k = √3/3 is a sufficient but not necessary condition for tangency --/
theorem sqrt3_div3_sufficient_not_necessary :
  (is_tangent (Real.sqrt 3 / 3)) ∧
  (∃ k, k ≠ Real.sqrt 3 / 3 ∧ is_tangent k) := by sorry

end NUMINAMATH_CALUDE_sqrt3_div3_sufficient_sqrt3_div3_not_necessary_sqrt3_div3_sufficient_not_necessary_l1946_194659


namespace NUMINAMATH_CALUDE_committee_selection_ways_l1946_194657

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem committee_selection_ways :
  let total_members : ℕ := 30
  let committee_size : ℕ := 5
  choose total_members committee_size = 142506 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_ways_l1946_194657


namespace NUMINAMATH_CALUDE_smallest_valid_distribution_l1946_194692

/-- Represents a distribution of candy pieces to children in a circle. -/
def CandyDistribution := List Nat

/-- Checks if all elements in the list are distinct. -/
def all_distinct (l : List Nat) : Prop :=
  l.Nodup

/-- Checks if all elements in the list are at least 1. -/
def all_at_least_one (l : List Nat) : Prop :=
  ∀ x ∈ l, x ≥ 1

/-- Checks if adjacent elements (including the first and last) have a common factor other than 1. -/
def adjacent_common_factor (l : List Nat) : Prop :=
  ∀ i, ∃ k > 1, k ∣ (l.get! i) ∧ k ∣ (l.get! ((i + 1) % l.length))

/-- Checks if there is no prime that divides all elements in the list. -/
def no_common_prime_divisor (l : List Nat) : Prop :=
  ¬∃ p, Nat.Prime p ∧ ∀ x ∈ l, p ∣ x

/-- Checks if a candy distribution satisfies all conditions. -/
def valid_distribution (d : CandyDistribution) : Prop :=
  d.length = 7 ∧
  all_distinct d ∧
  all_at_least_one d ∧
  adjacent_common_factor d ∧
  no_common_prime_divisor d

/-- The main theorem stating that 44 is the smallest number of candy pieces
    that satisfies all conditions for seven children. -/
theorem smallest_valid_distribution :
  (∃ d : CandyDistribution, valid_distribution d ∧ d.sum = 44) ∧
  (∀ d : CandyDistribution, valid_distribution d → d.sum ≥ 44) :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_distribution_l1946_194692


namespace NUMINAMATH_CALUDE_time_fraction_proof_l1946_194687

/-- Given a 24-hour day and the current time being 6, 
    prove that the fraction of time left to time already completed is 3. -/
theorem time_fraction_proof : 
  let hours_in_day : ℕ := 24
  let current_time : ℕ := 6
  let time_left : ℕ := hours_in_day - current_time
  let time_completed : ℕ := current_time
  (time_left : ℚ) / time_completed = 3 := by
  sorry

end NUMINAMATH_CALUDE_time_fraction_proof_l1946_194687


namespace NUMINAMATH_CALUDE_no_adjacent_standing_probability_l1946_194614

/-- Represents the number of valid arrangements where no two adjacent people are standing
    for n people around a circular table. -/
def validArrangements : ℕ → ℕ
| 0 => 1
| 1 => 2
| n + 2 => validArrangements (n + 1) + validArrangements n

/-- The probability of no two adjacent people standing when 10 people
    around a circular table flip fair coins. -/
theorem no_adjacent_standing_probability :
  (validArrangements 10 : ℚ) / 2^10 = 123 / 1024 := by sorry

end NUMINAMATH_CALUDE_no_adjacent_standing_probability_l1946_194614


namespace NUMINAMATH_CALUDE_anna_toy_production_l1946_194611

/-- Anna's toy production problem -/
theorem anna_toy_production (t : ℕ) : 
  let w : ℕ := 3 * t
  let monday_production : ℕ := w * t
  let tuesday_production : ℕ := (w + 5) * (t - 3)
  monday_production - tuesday_production = 4 * t + 15 := by
sorry

end NUMINAMATH_CALUDE_anna_toy_production_l1946_194611


namespace NUMINAMATH_CALUDE_exponent_equation_solution_l1946_194608

theorem exponent_equation_solution (a b : ℝ) (m n : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^m * b * b^n)^3 = a^6 * b^15 → m = 2 ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_exponent_equation_solution_l1946_194608


namespace NUMINAMATH_CALUDE_hall_of_mirrors_wall_length_l1946_194625

/-- Given three walls in a hall of mirrors, where two walls have the same unknown length and are 12 feet high,
    and the third wall is 20 feet by 12 feet, if the total glass needed is 960 square feet,
    then the length of each of the two unknown walls is 30 feet. -/
theorem hall_of_mirrors_wall_length :
  ∀ (L : ℝ),
  (2 * L * 12 + 20 * 12 = 960) →
  L = 30 := by
sorry

end NUMINAMATH_CALUDE_hall_of_mirrors_wall_length_l1946_194625


namespace NUMINAMATH_CALUDE_binomial_coefficient_ratio_l1946_194666

theorem binomial_coefficient_ratio (n k : ℕ) : 
  (Nat.choose n k : ℚ) / (Nat.choose n (k+1) : ℚ) = 1 / 3 ∧
  (Nat.choose n (k+1) : ℚ) / (Nat.choose n (k+2) : ℚ) = 3 / 5 →
  n + k = 19 :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_ratio_l1946_194666


namespace NUMINAMATH_CALUDE_canvas_bag_lower_carbon_l1946_194652

/-- The number of shopping trips required for a canvas bag to be the lower-carbon solution -/
def shopping_trips_for_lower_carbon (canvas_co2_pounds : ℕ) (plastic_co2_ounces : ℕ) (bags_per_trip : ℕ) : ℕ :=
  let canvas_co2_ounces : ℕ := canvas_co2_pounds * 16
  let plastic_co2_per_trip : ℕ := plastic_co2_ounces * bags_per_trip
  canvas_co2_ounces / plastic_co2_per_trip

/-- Theorem stating the number of shopping trips required for the canvas bag to be lower-carbon -/
theorem canvas_bag_lower_carbon :
  shopping_trips_for_lower_carbon 600 4 8 = 300 := by
  sorry

end NUMINAMATH_CALUDE_canvas_bag_lower_carbon_l1946_194652


namespace NUMINAMATH_CALUDE_inequality_solution_implication_l1946_194689

theorem inequality_solution_implication (a : ℝ) : 
  (∀ x : ℝ, (a + 1) * x > a + 1 ↔ x < 1) → a < -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_implication_l1946_194689


namespace NUMINAMATH_CALUDE_fishing_theorem_l1946_194681

def fishing_problem (jordan_catch perry_catch alex_catch bird_steal release_fraction : ℕ) : ℕ :=
  let total_catch := jordan_catch + perry_catch + alex_catch
  let after_bird := total_catch - bird_steal
  let to_release := (after_bird * release_fraction) / 3
  after_bird - to_release

theorem fishing_theorem :
  fishing_problem 4 8 36 2 1 = 31 :=
by sorry

end NUMINAMATH_CALUDE_fishing_theorem_l1946_194681


namespace NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_square_l1946_194667

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem smallest_multiplier_for_perfect_square : 
  (∀ k : ℕ, k > 0 ∧ k < 7 → ¬ is_perfect_square (1008 * k)) ∧ 
  is_perfect_square (1008 * 7) := by
  sorry

#check smallest_multiplier_for_perfect_square

end NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_square_l1946_194667


namespace NUMINAMATH_CALUDE_derivative_ln_2x_squared_plus_1_l1946_194610

open Real

theorem derivative_ln_2x_squared_plus_1 (x : ℝ) :
  deriv (λ x => Real.log (2 * x^2 + 1)) x = (4 * x) / (2 * x^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_derivative_ln_2x_squared_plus_1_l1946_194610


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_l1946_194671

theorem absolute_value_inequality_solution (a : ℝ) : 
  (∀ x : ℝ, |a * x - 2| < 3 ↔ -5/3 < x ∧ x < 1/3) → a = -3 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_l1946_194671


namespace NUMINAMATH_CALUDE_gertrude_has_ten_fleas_l1946_194655

/-- The number of fleas on Gertrude's chicken -/
def gertrude_fleas : ℕ := sorry

/-- The number of fleas on Maud's chicken -/
def maud_fleas : ℕ := sorry

/-- The number of fleas on Olive's chicken -/
def olive_fleas : ℕ := sorry

/-- Maud has 5 times the amount of fleas as Olive -/
axiom maud_olive_relation : maud_fleas = 5 * olive_fleas

/-- Olive has half the amount of fleas as Gertrude -/
axiom olive_gertrude_relation : olive_fleas * 2 = gertrude_fleas

/-- The total number of fleas is 40 -/
axiom total_fleas : gertrude_fleas + maud_fleas + olive_fleas = 40

/-- Theorem: Gertrude has 10 fleas -/
theorem gertrude_has_ten_fleas : gertrude_fleas = 10 := by sorry

end NUMINAMATH_CALUDE_gertrude_has_ten_fleas_l1946_194655


namespace NUMINAMATH_CALUDE_doll_count_sum_l1946_194602

/-- The number of dolls each person has -/
structure DollCounts where
  vera : ℕ
  lisa : ℕ
  sophie : ℕ
  aida : ℕ

/-- The conditions of the doll counting problem -/
def doll_problem (d : DollCounts) : Prop :=
  d.aida = 3 * d.sophie ∧
  d.sophie = 2 * d.vera ∧
  d.vera = d.lisa / 3 ∧
  d.lisa = d.vera + 10 ∧
  d.vera = 15

theorem doll_count_sum (d : DollCounts) : 
  doll_problem d → d.aida + d.sophie + d.vera + d.lisa = 160 := by
  sorry

end NUMINAMATH_CALUDE_doll_count_sum_l1946_194602


namespace NUMINAMATH_CALUDE_angle_CDE_value_l1946_194640

-- Define the points
variable (A B C D E : Point)

-- Define the angles
variable (angleA angleB angleC angleAEB angleBED angleAED angleADE angleCDE : Real)

-- State the given conditions
axiom right_angles : angleA = 90 ∧ angleB = 90 ∧ angleC = 90
axiom angle_AEB : angleAEB = 50
axiom angle_BED : angleBED = 45
axiom isosceles_ADE : angleAED = angleADE

-- State the theorem to be proved
theorem angle_CDE_value : angleCDE = 112.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_CDE_value_l1946_194640


namespace NUMINAMATH_CALUDE_quadratic_coefficients_l1946_194648

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 + 2 = 3*x

-- Define the standard form of the quadratic equation
def standard_form (a b c x : ℝ) : Prop := a*x^2 + b*x + c = 0

-- Theorem statement
theorem quadratic_coefficients :
  ∃ (c : ℝ), ∀ (x : ℝ),
    quadratic_equation x ↔ standard_form 1 (-3) c x :=
by sorry

end NUMINAMATH_CALUDE_quadratic_coefficients_l1946_194648


namespace NUMINAMATH_CALUDE_lava_lamp_probability_l1946_194630

def total_lamps : ℕ := 8
def red_lamps : ℕ := 4
def blue_lamps : ℕ := 4
def lamps_turned_on : ℕ := 4

theorem lava_lamp_probability :
  let total_arrangements := Nat.choose total_lamps red_lamps
  let color_condition := Nat.choose (total_lamps - 4) (red_lamps - 2)
  let on_off_condition := Nat.choose (total_lamps - 2) (lamps_turned_on - 2)
  (color_condition * on_off_condition : ℚ) / (total_arrangements * total_arrangements) = 225 / 4900 := by
  sorry

end NUMINAMATH_CALUDE_lava_lamp_probability_l1946_194630


namespace NUMINAMATH_CALUDE_right_triangle_sin_z_l1946_194651

theorem right_triangle_sin_z (X Y Z : ℝ) : 
  -- XYZ is a right triangle
  0 ≤ X ∧ X < π/2 ∧ 0 ≤ Y ∧ Y < π/2 ∧ 0 ≤ Z ∧ Z < π/2 ∧ X + Y + Z = π/2 →
  -- sin X = 3/5
  Real.sin X = 3/5 →
  -- cos Y = 0
  Real.cos Y = 0 →
  -- Then sin Z = 3/5
  Real.sin Z = 3/5 := by sorry

end NUMINAMATH_CALUDE_right_triangle_sin_z_l1946_194651


namespace NUMINAMATH_CALUDE_divisible_by_five_l1946_194662

theorem divisible_by_five (a b : ℕ) : 
  a > 0 → b > 0 → 5 ∣ (a * b) → ¬(¬(5 ∣ a) ∧ ¬(5 ∣ b)) :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_five_l1946_194662


namespace NUMINAMATH_CALUDE_min_value_expression_l1946_194645

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 1) :
  2 * x^2 + 8 * x * y + 6 * y^2 + 16 * y * z + 3 * z^2 ≥ 24 ∧
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 1 ∧
    2 * a^2 + 8 * a * b + 6 * b^2 + 16 * b * c + 3 * c^2 = 24 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1946_194645


namespace NUMINAMATH_CALUDE_angle_complement_half_supplement_l1946_194634

theorem angle_complement_half_supplement (x : ℝ) : 
  (90 - x) = (1/2) * (180 - x) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_angle_complement_half_supplement_l1946_194634


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1946_194650

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- A point on the ellipse -/
structure PointOnEllipse (E : Ellipse) where
  x : ℝ
  y : ℝ
  h : (x^2 / E.a^2) + (y^2 / E.b^2) = 1

/-- The equation of a line -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem: The equation of the tangent line to an ellipse at a point on the ellipse -/
theorem tangent_line_equation (E : Ellipse) (P : PointOnEllipse E) :
  ∃ (L : Line), L.a = P.x / E.a^2 ∧ L.b = P.y / E.b^2 ∧ L.c = -1 ∧
  (∀ (x y : ℝ), (x^2 / E.a^2) + (y^2 / E.b^2) ≤ 1 → L.a * x + L.b * y + L.c ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1946_194650


namespace NUMINAMATH_CALUDE_savings_difference_l1946_194615

def initial_order : ℝ := 15000

def scheme1_discounts : List ℝ := [0.25, 0.15, 0.10]
def scheme2_discounts : List ℝ := [0.30, 0.10, 0.05]

def apply_discounts (initial_value : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl (fun acc discount => acc * (1 - discount)) initial_value

theorem savings_difference :
  apply_discounts initial_order scheme2_discounts - apply_discounts initial_order scheme1_discounts = 371.25 := by
  sorry

end NUMINAMATH_CALUDE_savings_difference_l1946_194615


namespace NUMINAMATH_CALUDE_three_eighths_decimal_l1946_194649

theorem three_eighths_decimal : (3 : ℚ) / 8 = 0.375 := by
  sorry

end NUMINAMATH_CALUDE_three_eighths_decimal_l1946_194649


namespace NUMINAMATH_CALUDE_equation_solution_l1946_194646

theorem equation_solution : 
  ∃! x : ℚ, (x - 30) / 3 = (5 - 3 * x) / 4 + 2 :=
by
  use 15
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1946_194646


namespace NUMINAMATH_CALUDE_parallel_vectors_solution_perpendicular_vectors_solution_l1946_194619

-- Define the vectors
def a (x : ℝ) : ℝ × ℝ := (1, 2*x)
def b (x : ℝ) : ℝ × ℝ := (x, 3)
def c : ℝ × ℝ := (-2, 0)

-- Question 1: Parallel vectors condition
def parallel_condition (x : ℝ) : Prop :=
  (1 + 2*x) * 4*x = 4*(2*x + 6)

-- Question 2: Perpendicular vectors condition
def perpendicular_condition (x : ℝ) : Prop :=
  8*x^2 + 32*x + 4 = 0

-- Theorem for parallel vectors
theorem parallel_vectors_solution :
  ∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = -3/2 ∧ parallel_condition x₁ ∧ parallel_condition x₂ :=
sorry

-- Theorem for perpendicular vectors
theorem perpendicular_vectors_solution :
  ∃ x₁ x₂ : ℝ, x₁ = (-4 + Real.sqrt 14)/2 ∧ x₂ = (-4 - Real.sqrt 14)/2 ∧
  perpendicular_condition x₁ ∧ perpendicular_condition x₂ :=
sorry

end NUMINAMATH_CALUDE_parallel_vectors_solution_perpendicular_vectors_solution_l1946_194619


namespace NUMINAMATH_CALUDE_furniture_shop_cost_price_l1946_194677

theorem furniture_shop_cost_price (markup_percentage : ℝ) (selling_price : ℝ) (cost_price : ℝ) : 
  markup_percentage = 20 →
  selling_price = 3600 →
  selling_price = cost_price * (1 + markup_percentage / 100) →
  cost_price = 3000 := by
  sorry

end NUMINAMATH_CALUDE_furniture_shop_cost_price_l1946_194677


namespace NUMINAMATH_CALUDE_largest_root_of_f_cubed_l1946_194697

/-- The function f(x) = x^2 + 12x + 30 -/
def f (x : ℝ) : ℝ := x^2 + 12*x + 30

/-- The composition of f with itself three times -/
def f_cubed (x : ℝ) : ℝ := f (f (f x))

/-- The largest real root of f(f(f(x))) = 0 -/
noncomputable def largest_root : ℝ := -6 + (6 : ℝ)^(1/8)

theorem largest_root_of_f_cubed :
  (f_cubed largest_root = 0) ∧
  (∀ x : ℝ, f_cubed x = 0 → x ≤ largest_root) :=
by sorry

end NUMINAMATH_CALUDE_largest_root_of_f_cubed_l1946_194697


namespace NUMINAMATH_CALUDE_point_coordinates_in_third_quadrant_l1946_194647

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between a point and the x-axis -/
def distToXAxis (p : Point) : ℝ := |p.y|

/-- The distance between a point and the y-axis -/
def distToYAxis (p : Point) : ℝ := |p.x|

/-- Predicate for a point being in the third quadrant -/
def isInThirdQuadrant (p : Point) : Prop := p.x < 0 ∧ p.y < 0

theorem point_coordinates_in_third_quadrant 
  (M : Point) 
  (h1 : isInThirdQuadrant M) 
  (h2 : distToXAxis M = 1) 
  (h3 : distToYAxis M = 2) : 
  M.x = -2 ∧ M.y = -1 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_in_third_quadrant_l1946_194647


namespace NUMINAMATH_CALUDE_first_month_sale_is_3435_l1946_194624

/-- Calculates the sale in the first month given the sales for the next five months and the average sale --/
def calculate_first_month_sale (sales_2_to_5 : List ℕ) (sale_6 : ℕ) (average_sale : ℕ) : ℕ :=
  6 * average_sale - (sales_2_to_5.sum + sale_6)

/-- Theorem stating that the sale in the first month is 3435 given the specified conditions --/
theorem first_month_sale_is_3435 :
  let sales_2_to_5 := [3920, 3855, 4230, 3560]
  let sale_6 := 2000
  let average_sale := 3500
  calculate_first_month_sale sales_2_to_5 sale_6 average_sale = 3435 := by
  sorry

#eval calculate_first_month_sale [3920, 3855, 4230, 3560] 2000 3500

end NUMINAMATH_CALUDE_first_month_sale_is_3435_l1946_194624


namespace NUMINAMATH_CALUDE_sum_a_b_equals_nine_l1946_194601

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation
def equation (a b : ℝ) : Prop :=
  i * (a - i) = b - (2 * i) ^ 3

-- Theorem statement
theorem sum_a_b_equals_nine (a b : ℝ) (h : equation a b) : a + b = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_a_b_equals_nine_l1946_194601


namespace NUMINAMATH_CALUDE_opposite_expressions_theorem_l1946_194606

theorem opposite_expressions_theorem (a : ℚ) : 
  (3 * a + 1 = -(3 * (a - 1))) → a = 1/3 := by
sorry

end NUMINAMATH_CALUDE_opposite_expressions_theorem_l1946_194606


namespace NUMINAMATH_CALUDE_dividend_calculation_l1946_194660

theorem dividend_calculation (dividend quotient remainder divisor : ℕ) 
  (h1 : divisor = 28)
  (h2 : quotient = 7)
  (h3 : remainder = 11)
  (h4 : dividend = divisor * quotient + remainder) :
  dividend = 207 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1946_194660


namespace NUMINAMATH_CALUDE_height_comparison_l1946_194632

theorem height_comparison (a b : ℝ) (h : a = b * (1 - 0.25)) :
  b = a * (1 + 1/3) :=
by sorry

end NUMINAMATH_CALUDE_height_comparison_l1946_194632


namespace NUMINAMATH_CALUDE_emails_morning_evening_l1946_194643

def morning_emails : ℕ := 3
def evening_emails : ℕ := 8

theorem emails_morning_evening : 
  morning_emails + evening_emails = 11 :=
by sorry

end NUMINAMATH_CALUDE_emails_morning_evening_l1946_194643


namespace NUMINAMATH_CALUDE_subset_condition_l1946_194642

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ m + 1}

theorem subset_condition (m : ℝ) : B m ⊆ A → -1 ≤ m ∧ m ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_subset_condition_l1946_194642


namespace NUMINAMATH_CALUDE_chef_apples_l1946_194627

/-- Represents the number of apples used to make the pie -/
def apples_used : ℕ := 15

/-- Represents the number of apples left after making the pie -/
def apples_left : ℕ := 4

/-- Represents the total number of apples before making the pie -/
def total_apples : ℕ := apples_used + apples_left

/-- Theorem stating that the total number of apples before making the pie
    is equal to the sum of apples used and apples left -/
theorem chef_apples : total_apples = apples_used + apples_left := by
  sorry

end NUMINAMATH_CALUDE_chef_apples_l1946_194627


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l1946_194629

theorem quadratic_root_difference (a b c : ℝ) (h : a ≠ 0) :
  let eq := fun x => a * x^2 + b * x + c
  let r1 := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r2 := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  eq 1 + 40 + 300 = -64 →
  |r1 - r2| = 12 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l1946_194629


namespace NUMINAMATH_CALUDE_line_tangent_to_ellipse_l1946_194694

theorem line_tangent_to_ellipse (m : ℝ) : 
  (∀ x y : ℝ, y = m * x + 1 ∧ x^2 + 4 * y^2 = 1 → 
    ∀ x' y' : ℝ, y' = m * x' + 1 ∧ x'^2 + 4 * y'^2 = 1 → x = x' ∧ y = y') →
  m^2 = 3/4 := by
sorry

end NUMINAMATH_CALUDE_line_tangent_to_ellipse_l1946_194694


namespace NUMINAMATH_CALUDE_stock_price_change_l1946_194688

/-- The final stock price after two years of changes -/
def final_stock_price (initial_price : ℝ) (first_year_increase : ℝ) (second_year_decrease : ℝ) : ℝ :=
  initial_price * (1 + first_year_increase) * (1 - second_year_decrease)

/-- Theorem stating the final stock price after specific changes -/
theorem stock_price_change : final_stock_price 80 1.2 0.3 = 123.2 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_change_l1946_194688


namespace NUMINAMATH_CALUDE_circle_equation_correct_l1946_194613

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Check if a point lies on the circle -/
def pointOnCircle (c : Circle) (p : Point) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- The standard equation of the circle -/
def circleEquation (x y : ℝ) (c : Circle) : Prop :=
  (x - c.center.x)^2 + (y - c.center.y)^2 = c.radius^2

/-- Theorem: The given equation represents the circle with center (-3, 4) and radius 2 -/
theorem circle_equation_correct (x y : ℝ) : 
  let c := Circle.mk (Point.mk (-3) 4) 2
  circleEquation x y c ↔ (x + 3)^2 + (y - 4)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_correct_l1946_194613


namespace NUMINAMATH_CALUDE_largest_integral_x_l1946_194685

theorem largest_integral_x : ∃ x : ℤ, x = 4 ∧ 
  (∀ y : ℤ, (1/4 : ℚ) < (y : ℚ)/6 ∧ (y : ℚ)/6 < 7/9 → y ≤ x) ∧
  (1/4 : ℚ) < (x : ℚ)/6 ∧ (x : ℚ)/6 < 7/9 := by
  sorry

end NUMINAMATH_CALUDE_largest_integral_x_l1946_194685


namespace NUMINAMATH_CALUDE_tangent_line_length_l1946_194626

-- Define the circle C
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 6*y + 9 = 0

-- Define the point P
def P : ℝ × ℝ := (1, 0)

-- Theorem statement
theorem tangent_line_length :
  ∃ (t : ℝ × ℝ), 
    circle_equation t.1 t.2 ∧ 
    (t.1 - P.1)^2 + (t.2 - P.2)^2 = 8 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_length_l1946_194626


namespace NUMINAMATH_CALUDE_total_coins_last_month_l1946_194696

/-- The number of coins Mathilde had at the start of this month -/
def mathilde_this_month : ℕ := 100

/-- The number of coins Salah had at the start of this month -/
def salah_this_month : ℕ := 100

/-- The percentage increase in Mathilde's coins from last month to this month -/
def mathilde_increase_percent : ℚ := 25/100

/-- The percentage decrease in Salah's coins from last month to this month -/
def salah_decrease_percent : ℚ := 20/100

/-- The number of coins Mathilde had at the start of last month -/
def mathilde_last_month : ℚ := mathilde_this_month / (1 + mathilde_increase_percent)

/-- The number of coins Salah had at the start of last month -/
def salah_last_month : ℚ := salah_this_month / (1 - salah_decrease_percent)

theorem total_coins_last_month :
  mathilde_last_month + salah_last_month = 205 := by sorry

end NUMINAMATH_CALUDE_total_coins_last_month_l1946_194696


namespace NUMINAMATH_CALUDE_f_simplification_and_range_l1946_194672

noncomputable def f (x : ℝ) : ℝ := 
  (Real.sin x ^ 3 + 5 * Real.sin x ^ 2 + 2 * Real.sin x - 3 * Real.cos x ^ 2 - 9) / (Real.sin x - 2)

theorem f_simplification_and_range : 
  ∀ x : ℝ, Real.sin x ≠ 2 → 
    (f x = Real.sin x ^ 2 + 4 * Real.sin x + 6) ∧ 
    (∃ y : ℝ, f y = 1) ∧ 
    (∃ z : ℝ, f z = 13) ∧ 
    (∀ w : ℝ, Real.sin w ≠ 2 → 1 ≤ f w ∧ f w ≤ 13) :=
by sorry

end NUMINAMATH_CALUDE_f_simplification_and_range_l1946_194672


namespace NUMINAMATH_CALUDE_set_operations_l1946_194622

def U : Set ℝ := {x | 1 ≤ x ∧ x ≤ 7}
def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 5}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 7}

theorem set_operations :
  (A ∪ B = {x | 2 ≤ x ∧ x ≤ 7}) ∧
  ((U \ A) ∪ B = {x | (1 ≤ x ∧ x < 2) ∨ (3 ≤ x ∧ x ≤ 7)}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l1946_194622


namespace NUMINAMATH_CALUDE_goose_eggs_count_goose_eggs_solution_l1946_194605

theorem goose_eggs_count : ℕ → Prop :=
  fun total_eggs =>
    let hatched := (1 : ℚ) / 4 * total_eggs
    let survived_first_month := (4 : ℚ) / 5 * hatched
    let survived_six_months := (2 : ℚ) / 5 * survived_first_month
    let survived_first_year := (4 : ℚ) / 7 * survived_six_months
    survived_first_year = 120 ∧ total_eggs = 2625

/-- The number of goose eggs laid at the pond is 2625. -/
theorem goose_eggs_solution : goose_eggs_count 2625 := by
  sorry

end NUMINAMATH_CALUDE_goose_eggs_count_goose_eggs_solution_l1946_194605


namespace NUMINAMATH_CALUDE_triangle_area_unchanged_l1946_194656

theorem triangle_area_unchanged 
  (base height : ℝ) 
  (base_positive : base > 0) 
  (height_positive : height > 0) : 
  (1/2) * base * height = (1/2) * (base / 3) * (3 * height) := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_unchanged_l1946_194656


namespace NUMINAMATH_CALUDE_sarah_test_result_l1946_194699

/-- Represents a math test with a number of problems and a score percentage -/
structure MathTest where
  problems : ℕ
  score : ℚ
  score_valid : 0 ≤ score ∧ score ≤ 1

/-- Calculates the number of correctly answered problems in a test -/
def correctProblems (test : MathTest) : ℚ :=
  test.problems * test.score

/-- Calculates the overall percentage of correctly answered problems across multiple tests -/
def overallPercentage (tests : List MathTest) : ℚ :=
  let totalCorrect := (tests.map correctProblems).sum
  let totalProblems := (tests.map (·.problems)).sum
  totalCorrect / totalProblems

theorem sarah_test_result : 
  let test1 : MathTest := { problems := 30, score := 85/100, score_valid := by norm_num }
  let test2 : MathTest := { problems := 50, score := 75/100, score_valid := by norm_num }
  let test3 : MathTest := { problems := 20, score := 80/100, score_valid := by norm_num }
  let tests := [test1, test2, test3]
  overallPercentage tests = 78/100 := by
  sorry

end NUMINAMATH_CALUDE_sarah_test_result_l1946_194699


namespace NUMINAMATH_CALUDE_not_always_preservable_flight_relations_l1946_194604

/-- Represents a city in the country -/
structure City where
  id : Nat

/-- Represents the flight guide for the country -/
structure FlightGuide where
  cities : Finset City
  has_direct_flight : City → City → Bool

/-- Represents a permutation of city IDs -/
def CityPermutation := Nat → Nat

/-- Theorem stating that it's not always possible to maintain flight relations after swapping city numbers -/
theorem not_always_preservable_flight_relations :
  ∃ (fg : FlightGuide) (m n : City),
    m ∈ fg.cities → n ∈ fg.cities → m ≠ n →
    ¬∀ (p : CityPermutation),
      (∀ c : City, c ∈ fg.cities → p (c.id) ≠ c.id → (c = m ∨ c = n)) →
      (p m.id = n.id ∧ p n.id = m.id) →
      (∀ c1 c2 : City, c1 ∈ fg.cities → c2 ∈ fg.cities →
        fg.has_direct_flight c1 c2 = fg.has_direct_flight
          ⟨p c1.id⟩ ⟨p c2.id⟩) :=
sorry

end NUMINAMATH_CALUDE_not_always_preservable_flight_relations_l1946_194604


namespace NUMINAMATH_CALUDE_ant_position_2024_l1946_194638

-- Define the ant's movement pattern
def antMove (n : ℕ) : ℤ × ℤ :=
  sorry

-- Theorem statement
theorem ant_position_2024 : antMove 2024 = (13, 0) := by
  sorry

end NUMINAMATH_CALUDE_ant_position_2024_l1946_194638


namespace NUMINAMATH_CALUDE_walmart_ground_beef_sales_l1946_194658

theorem walmart_ground_beef_sales (thursday_sales : ℕ) (friday_sales : ℕ) (saturday_sales : ℕ) 
  (h1 : thursday_sales = 210)
  (h2 : friday_sales = 2 * thursday_sales)
  (h3 : (thursday_sales + friday_sales + saturday_sales) / 3 = 260) :
  saturday_sales = 150 := by
sorry

end NUMINAMATH_CALUDE_walmart_ground_beef_sales_l1946_194658


namespace NUMINAMATH_CALUDE_courtyard_length_l1946_194683

/-- The length of a rectangular courtyard given specific conditions -/
theorem courtyard_length (width : ℝ) (num_stones : ℕ) (stone_length stone_width : ℝ) : 
  width = 20 → 
  num_stones = 100 → 
  stone_length = 4 → 
  stone_width = 2 → 
  (width * (num_stones * stone_length * stone_width / width) : ℝ) = 40 := by
  sorry

end NUMINAMATH_CALUDE_courtyard_length_l1946_194683


namespace NUMINAMATH_CALUDE_expression_evaluation_l1946_194661

theorem expression_evaluation : (60 + 5 * 12) / (Real.sqrt 180 / 3) ^ 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1946_194661


namespace NUMINAMATH_CALUDE_blood_expiration_theorem_l1946_194665

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat

/-- Represents a date -/
inductive Date
  | jan1 : Date

/-- Represents the expiration of blood -/
def blood_expiration (donation_time : Time) (donation_date : Date) (expiration_seconds : Nat) : 
  (Time × Date) :=
  sorry

theorem blood_expiration_theorem :
  let donation_time : Time := ⟨18, 0⟩  -- 6 PM
  let donation_date : Date := Date.jan1
  let expiration_seconds : Nat := 7 * 6 * 5 * 4 * 3 * 2 * 1
  blood_expiration donation_time donation_date expiration_seconds = (⟨19, 24⟩, Date.jan1) :=
by
  sorry

end NUMINAMATH_CALUDE_blood_expiration_theorem_l1946_194665


namespace NUMINAMATH_CALUDE_cube_difference_equality_l1946_194620

theorem cube_difference_equality (x y : ℝ) (h : x - y = 1) :
  x^3 - 3*x*y - y^3 = 1 := by sorry

end NUMINAMATH_CALUDE_cube_difference_equality_l1946_194620


namespace NUMINAMATH_CALUDE_amelia_monday_distance_l1946_194621

theorem amelia_monday_distance (total_distance tuesday_distance remaining_distance : ℕ) 
  (h1 : total_distance = 8205)
  (h2 : tuesday_distance = 582)
  (h3 : remaining_distance = 6716) :
  total_distance = tuesday_distance + remaining_distance + 907 := by
  sorry

end NUMINAMATH_CALUDE_amelia_monday_distance_l1946_194621


namespace NUMINAMATH_CALUDE_sequence_product_l1946_194623

/-- Given that (-9, a, -1) is an arithmetic sequence and (-9, m, b, n, -1) is a geometric sequence,
    prove that ab = 5. -/
theorem sequence_product (a m b n : ℝ) : 
  ((-9 : ℝ) - a = a - (-1 : ℝ)) →  -- arithmetic sequence condition
  (m / (-9 : ℝ) = b / m) →         -- geometric sequence condition for first two terms
  (b / m = n / b) →                -- geometric sequence condition for middle terms
  (n / b = (-1 : ℝ) / n) →         -- geometric sequence condition for last two terms
  a * b = 5 := by
sorry

end NUMINAMATH_CALUDE_sequence_product_l1946_194623


namespace NUMINAMATH_CALUDE_student_count_l1946_194693

theorem student_count (n : ℕ) (rank_top rank_bottom : ℕ) 
  (h1 : rank_top = 75)
  (h2 : rank_bottom = 75)
  (h3 : n = rank_top + rank_bottom - 1) :
  n = 149 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l1946_194693


namespace NUMINAMATH_CALUDE_some_number_value_l1946_194636

theorem some_number_value : 
  ∀ some_number : ℝ, 
  (some_number * 3.6) / (0.04 * 0.1 * 0.007) = 990.0000000000001 → 
  some_number = 7.7 := by
sorry

end NUMINAMATH_CALUDE_some_number_value_l1946_194636


namespace NUMINAMATH_CALUDE_square_difference_evaluation_l1946_194682

theorem square_difference_evaluation (c d : ℕ) (h1 : c = 5) (h2 : d = 3) :
  (c^2 + d)^2 - (c^2 - d)^2 = 300 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_evaluation_l1946_194682
