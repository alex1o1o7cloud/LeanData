import Mathlib

namespace tennis_ball_ratio_l2427_242795

theorem tennis_ball_ratio (total_ordered : ℕ) (extra_yellow : ℕ) : 
  total_ordered = 288 →
  extra_yellow = 90 →
  let white := total_ordered / 2
  let yellow := total_ordered / 2 + extra_yellow
  (white : ℚ) / yellow = 8 / 13 := by
sorry

end tennis_ball_ratio_l2427_242795


namespace largest_n_with_conditions_l2427_242780

theorem largest_n_with_conditions : 
  ∃ (m : ℤ), 139^2 = m^3 - 1 ∧ 
  ∃ (a : ℤ), 2 * 139 + 83 = a^2 ∧
  ∀ (n : ℤ), n > 139 → 
    (∀ (m : ℤ), n^2 ≠ m^3 - 1 ∨ ¬∃ (a : ℤ), 2 * n + 83 = a^2) :=
by sorry

end largest_n_with_conditions_l2427_242780


namespace y_satisfies_differential_equation_l2427_242713

noncomputable def y (x : ℝ) : ℝ := x / (x - 1) + x^2

theorem y_satisfies_differential_equation (x : ℝ) :
  x * (x - 1) * (deriv y x) + y x = x^2 * (2 * x - 1) :=
by sorry

end y_satisfies_differential_equation_l2427_242713


namespace triangle_side_lengths_l2427_242729

theorem triangle_side_lengths 
  (a b c : ℚ) 
  (perimeter : a + b + c = 24)
  (relation : a + 2 * b = 2 * c)
  (ratio : a = (1 / 2) * b) :
  a = 16 / 3 ∧ b = 32 / 3 ∧ c = 8 := by
  sorry

end triangle_side_lengths_l2427_242729


namespace symmetry_of_points_l2427_242779

/-- The line of symmetry --/
def line_of_symmetry (x y : ℝ) : Prop := x - y - 1 = 0

/-- Check if two points are symmetric with respect to a line --/
def is_symmetric (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  let midpoint_x := (x₁ + x₂) / 2
  let midpoint_y := (y₁ + y₂) / 2
  line_of_symmetry midpoint_x midpoint_y ∧
  (y₂ - y₁) / (x₂ - x₁) = -1

theorem symmetry_of_points :
  is_symmetric 2 2 3 1 :=
sorry

end symmetry_of_points_l2427_242779


namespace unknown_number_proof_l2427_242753

theorem unknown_number_proof (x : ℝ) : 
  (10 + 30 + x) / 3 = (20 + 40 + 6) / 3 + 8 ↔ x = 50 := by
  sorry

end unknown_number_proof_l2427_242753


namespace cheyenne_pots_count_l2427_242742

/-- The number of clay pots Cheyenne made -/
def total_pots : ℕ := 80

/-- The fraction of pots that cracked -/
def cracked_fraction : ℚ := 2/5

/-- The revenue from selling the remaining pots -/
def revenue : ℕ := 1920

/-- The price of each clay pot -/
def price_per_pot : ℕ := 40

/-- Theorem stating that the number of clay pots Cheyenne made is 80 -/
theorem cheyenne_pots_count :
  total_pots = 80 ∧
  cracked_fraction = 2/5 ∧
  revenue = 1920 ∧
  price_per_pot = 40 ∧
  (1 - cracked_fraction) * total_pots * price_per_pot = revenue :=
by sorry

end cheyenne_pots_count_l2427_242742


namespace complement_A_inter_B_range_of_a_l2427_242744

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 3*x - 10 ≤ 0}
def B : Set ℝ := {x | 4 < x ∧ x < 6}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem for the complement of A ∩ B
theorem complement_A_inter_B :
  ∀ x : ℝ, x ∉ (A ∩ B) ↔ (x ≤ 4 ∨ x > 5) := by sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (A ∪ B) ⊆ C a → a ≥ 6 := by sorry

end complement_A_inter_B_range_of_a_l2427_242744


namespace dinner_lunch_ratio_l2427_242749

/-- Represents the amount of bread eaten at each meal in grams -/
structure BreadConsumption where
  breakfast : ℕ
  lunch : ℕ
  dinner : ℕ

/-- Proves that given the conditions, the ratio of dinner bread to lunch bread is 8:1 -/
theorem dinner_lunch_ratio (b : BreadConsumption) : 
  b.dinner = 240 ∧ 
  ∃ k : ℕ, b.dinner = k * b.lunch ∧ 
  b.dinner = 6 * b.breakfast ∧ 
  b.breakfast + b.lunch + b.dinner = 310 → 
  b.dinner / b.lunch = 8 := by
  sorry

end dinner_lunch_ratio_l2427_242749


namespace system_solution_l2427_242751

theorem system_solution (x y : ℚ) : 
  (3 * x - 2 * y = 8) ∧ (x + 3 * y = 7) → x = 38 / 11 := by
  sorry

end system_solution_l2427_242751


namespace quadratic_intersection_with_x_axis_l2427_242798

theorem quadratic_intersection_with_x_axis (a b c : ℝ) :
  ∃ x : ℝ, (x - a) * (x - b) - c^2 = 0 := by
  sorry

end quadratic_intersection_with_x_axis_l2427_242798


namespace solution_exists_in_interval_l2427_242784

-- Define the function f(x) = x^3 + x - 3
def f (x : ℝ) : ℝ := x^3 + x - 3

-- State the theorem
theorem solution_exists_in_interval :
  ∃! r : ℝ, r ∈ Set.Icc 1 2 ∧ f r = 0 := by
  sorry

end solution_exists_in_interval_l2427_242784


namespace derivative_not_critical_point_l2427_242796

-- Define the function g(x) as the derivative of f(x)
def g (a : ℝ) (x : ℝ) : ℝ := (x - 1) * (x^2 - 3*x + a)

-- State the theorem
theorem derivative_not_critical_point (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, (deriv f) x = g a x) →  -- The derivative of f is g
  (deriv f) 1 ≠ 0 →             -- 1 is not a critical point
  a = 2 := by
sorry

end derivative_not_critical_point_l2427_242796


namespace multiplier_problem_l2427_242731

/-- Given a = 5, b = 30, and 40 ab = 1800, prove that the multiplier m such that m * a = 30 is equal to 6. -/
theorem multiplier_problem (a b : ℝ) (h1 : a = 5) (h2 : b = 30) (h3 : 40 * a * b = 1800) :
  ∃ m : ℝ, m * a = 30 ∧ m = 6 := by
sorry

end multiplier_problem_l2427_242731


namespace system_solution_range_l2427_242792

theorem system_solution_range (x y k : ℝ) : 
  (2 * x + y = 2 * k - 1) → 
  (x + 2 * y = -4) → 
  (x + y > 1) → 
  (k > 4) := by
sorry

end system_solution_range_l2427_242792


namespace zacks_marbles_l2427_242764

theorem zacks_marbles (M : ℕ) : 
  (∃ k : ℕ, M = 3 * k + 5) → 
  (M - 60 = 5) → 
  M = 65 := by
sorry

end zacks_marbles_l2427_242764


namespace helen_cookies_l2427_242734

/-- The number of chocolate chip cookies Helen baked yesterday -/
def yesterday_chocolate : ℕ := 19

/-- The number of chocolate chip cookies Helen baked this morning -/
def today_chocolate : ℕ := 237

/-- The difference between the number of chocolate chip cookies and raisin cookies Helen baked -/
def chocolate_raisin_diff : ℕ := 25

/-- The number of raisin cookies Helen baked -/
def raisin_cookies : ℕ := 231

theorem helen_cookies :
  raisin_cookies = (yesterday_chocolate + today_chocolate) - chocolate_raisin_diff := by
  sorry

end helen_cookies_l2427_242734


namespace sum_of_xyz_is_718_l2427_242714

noncomputable def a : ℝ := -1 / Real.sqrt 3
noncomputable def b : ℝ := (3 + Real.sqrt 7) / 3

theorem sum_of_xyz_is_718 (ha : a^2 = 9/27) (hb : b^2 = (3 + Real.sqrt 7)^2 / 9)
  (ha_neg : a < 0) (hb_pos : b > 0)
  (h_expr : ∃ (x y z : ℕ+), (a + b)^3 = (x : ℝ) * Real.sqrt y / z) :
  ∃ (x y z : ℕ+), (a + b)^3 = (x : ℝ) * Real.sqrt y / z ∧ x + y + z = 718 :=
sorry

end sum_of_xyz_is_718_l2427_242714


namespace division_sum_theorem_l2427_242789

theorem division_sum_theorem (dividend : ℕ) (divisor : ℕ) (h1 : dividend = 54) (h2 : divisor = 9) :
  dividend / divisor + dividend + divisor = 69 := by
  sorry

end division_sum_theorem_l2427_242789


namespace circle_diameter_from_area_l2427_242722

/-- For a circle with area 4π, the diameter is 4 -/
theorem circle_diameter_from_area : 
  ∀ (r : ℝ), r > 0 → π * r^2 = 4 * π → 2 * r = 4 := by
  sorry

end circle_diameter_from_area_l2427_242722


namespace inequality_theorem_l2427_242730

theorem inequality_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / c) + (c / b) ≥ (4 * a) / (a + b) ∧
  ((a / c) + (c / b) = (4 * a) / (a + b) ↔ a = b ∧ b = c) :=
by sorry

end inequality_theorem_l2427_242730


namespace shaded_area_is_45_l2427_242763

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square -/
structure Square where
  bottomLeft : Point
  side : ℝ

/-- Represents a right triangle -/
structure RightTriangle where
  bottomRight : Point
  base : ℝ
  height : ℝ

/-- Calculates the area of the shaded region formed by the intersection of a square and a right triangle -/
def shadedArea (square : Square) (triangle : RightTriangle) : ℝ :=
  sorry

/-- Theorem stating that the shaded area is 45 square units given the specified conditions -/
theorem shaded_area_is_45 :
  ∀ (square : Square) (triangle : RightTriangle),
    square.bottomLeft = Point.mk 12 0 →
    square.side = 12 →
    triangle.bottomRight = Point.mk 12 0 →
    triangle.base = 12 →
    triangle.height = 9 →
    shadedArea square triangle = 45 :=
  sorry

end shaded_area_is_45_l2427_242763


namespace even_factors_count_l2427_242705

/-- The number of even natural-number factors of 2^2 * 3^1 * 7^2 -/
def num_even_factors : ℕ := 12

/-- The prime factorization of n -/
def n : ℕ := 2^2 * 3^1 * 7^2

/-- A function that counts the number of even natural-number factors of n -/
def count_even_factors (n : ℕ) : ℕ := sorry

theorem even_factors_count :
  count_even_factors n = num_even_factors := by sorry

end even_factors_count_l2427_242705


namespace stove_repair_ratio_l2427_242772

theorem stove_repair_ratio :
  let stove_cost : ℚ := 1200
  let total_cost : ℚ := 1400
  let wall_cost : ℚ := total_cost - stove_cost
  (wall_cost / stove_cost) = 1 / 6 := by
sorry

end stove_repair_ratio_l2427_242772


namespace line_through_point_forming_triangle_l2427_242774

theorem line_through_point_forming_triangle : ∃ (a b : ℝ), 
  (∀ x y : ℝ, (x / a + y / b = 1) → ((-2) / a + 2 / b = 1)) ∧ 
  (1/2 * |a * b| = 1) ∧
  ((a = -1 ∧ b = -2) ∨ (a = 2 ∧ b = 1)) := by sorry

end line_through_point_forming_triangle_l2427_242774


namespace first_perfect_square_all_remainders_l2427_242777

theorem first_perfect_square_all_remainders : 
  ∀ n : ℕ, n ≤ 20 → 
    (∃ k ≤ n, k^2 % 10 = 0) ∧ 
    (∃ k ≤ n, k^2 % 10 = 1) ∧ 
    (∃ k ≤ n, k^2 % 10 = 2) ∧ 
    (∃ k ≤ n, k^2 % 10 = 3) ∧ 
    (∃ k ≤ n, k^2 % 10 = 4) ∧ 
    (∃ k ≤ n, k^2 % 10 = 5) ∧ 
    (∃ k ≤ n, k^2 % 10 = 6) ∧ 
    (∃ k ≤ n, k^2 % 10 = 7) ∧ 
    (∃ k ≤ n, k^2 % 10 = 8) ∧ 
    (∃ k ≤ n, k^2 % 10 = 9) ↔ 
    n = 20 :=
by sorry

end first_perfect_square_all_remainders_l2427_242777


namespace ellipse_parameters_sum_l2427_242708

-- Define the foci
def F₁ : ℝ × ℝ := (1, 0)
def F₂ : ℝ × ℝ := (5, 0)

-- Define the constant sum of distances
def distance_sum : ℝ := 10

-- Define the ellipse
def is_on_ellipse (P : ℝ × ℝ) : Prop :=
  Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) +
  Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = distance_sum

-- Define the general form of the ellipse equation
def ellipse_equation (h k a b : ℝ) (P : ℝ × ℝ) : Prop :=
  (P.1 - h)^2 / a^2 + (P.2 - k)^2 / b^2 = 1

-- Theorem statement
theorem ellipse_parameters_sum :
  ∃ (h k a b : ℝ),
    (∀ P, is_on_ellipse P ↔ ellipse_equation h k a b P) ∧
    h + k + a + b = 8 + Real.sqrt 21 :=
sorry

end ellipse_parameters_sum_l2427_242708


namespace sum_of_three_numbers_l2427_242797

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) 
  (h2 : a^2 + b^2 + c^2 = 16) 
  (h3 : a*b + b*c + c*a = 9) 
  (h4 : a^2 + b^2 = 10) : 
  a + b + c = Real.sqrt 34 := by
  sorry

end sum_of_three_numbers_l2427_242797


namespace banana_count_l2427_242706

/-- The number of bananas in the fruit shop. -/
def bananas : ℕ := 30

/-- The number of apples in the fruit shop. -/
def apples : ℕ := 4 * bananas

/-- The number of persimmons in the fruit shop. -/
def persimmons : ℕ := 3 * bananas

/-- Theorem stating that the number of bananas is 30, given the conditions. -/
theorem banana_count : bananas = 30 := by
  have h1 : apples + persimmons = 210 := by sorry
  sorry

end banana_count_l2427_242706


namespace z_in_fourth_quadrant_z_purely_imaginary_iff_l2427_242778

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := m * (3 + Complex.I) - (2 + Complex.I)

-- Theorem 1: z is in the fourth quadrant when 2/3 < m < 1
theorem z_in_fourth_quadrant (m : ℝ) (h1 : 2/3 < m) (h2 : m < 1) :
  (z m).re > 0 ∧ (z m).im < 0 :=
sorry

-- Theorem 2: z is purely imaginary iff m = 2/3
theorem z_purely_imaginary_iff (m : ℝ) :
  (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = 2/3 :=
sorry

end z_in_fourth_quadrant_z_purely_imaginary_iff_l2427_242778


namespace females_together_count_females_apart_count_l2427_242746

/-- The number of male students -/
def num_male : ℕ := 4

/-- The number of female students -/
def num_female : ℕ := 3

/-- The total number of students -/
def total_students : ℕ := num_male + num_female

/-- Calculates the number of arrangements where female students must stand together -/
def arrangements_females_together : ℕ :=
  (Nat.factorial num_female) * (Nat.factorial (num_male + 1))

/-- Calculates the number of arrangements where no two female students can stand next to each other -/
def arrangements_females_apart : ℕ :=
  (Nat.factorial num_male) * (Nat.choose (num_male + 1) num_female)

/-- Theorem stating the number of arrangements where female students must stand together -/
theorem females_together_count : arrangements_females_together = 720 := by
  sorry

/-- Theorem stating the number of arrangements where no two female students can stand next to each other -/
theorem females_apart_count : arrangements_females_apart = 1440 := by
  sorry

end females_together_count_females_apart_count_l2427_242746


namespace cylinder_cross_section_area_l2427_242704

/-- Represents a cylinder with given dimensions and arc --/
structure Cylinder :=
  (radius : ℝ)
  (height : ℝ)
  (arc_angle : ℝ)

/-- Calculates the area of the cross-section of the cylinder --/
def cross_section_area (c : Cylinder) : ℝ := sorry

/-- Checks if a number is not divisible by the square of any prime --/
def not_divisible_by_square_prime (n : ℕ) : Prop := sorry

/-- Main theorem about the cross-section area of the specific cylinder --/
theorem cylinder_cross_section_area :
  let c := Cylinder.mk 7 10 (150 * π / 180)
  ∃ (d e : ℕ) (f : ℕ),
    cross_section_area c = d * π + e * Real.sqrt f ∧
    not_divisible_by_square_prime f ∧
    d = 60 ∧ e = 70 ∧ f = 3 := by sorry

end cylinder_cross_section_area_l2427_242704


namespace lucy_age_is_12_l2427_242785

def sisters_ages : List Nat := [2, 4, 6, 10, 12, 14]

def movie_pair (ages : List Nat) : Prop :=
  ∃ a b, a ∈ ages ∧ b ∈ ages ∧ a ≠ b ∧ a + b = 20

def basketball_pair (ages : List Nat) : Prop :=
  ∃ a b, a ∈ ages ∧ b ∈ ages ∧ a ≠ b ∧ a ≤ 10 ∧ b ≤ 10

def staying_home (lucy_age : Nat) (ages : List Nat) : Prop :=
  lucy_age ∈ ages ∧ ∃ a, a ∈ ages ∧ a ≠ lucy_age

theorem lucy_age_is_12 :
  movie_pair sisters_ages →
  basketball_pair sisters_ages →
  ∃ lucy_age, staying_home lucy_age sisters_ages ∧ lucy_age = 12 :=
by sorry

end lucy_age_is_12_l2427_242785


namespace solomon_collected_66_cans_l2427_242702

/-- The number of cans collected by Solomon, Juwan, and Levi -/
structure CanCollection where
  solomon : ℕ
  juwan : ℕ
  levi : ℕ

/-- The conditions of the can collection problem -/
def validCollection (c : CanCollection) : Prop :=
  c.solomon = 3 * c.juwan ∧
  c.levi = c.juwan / 2 ∧
  c.solomon + c.juwan + c.levi = 99

/-- Theorem stating that Solomon collected 66 cans -/
theorem solomon_collected_66_cans :
  ∃ (c : CanCollection), validCollection c ∧ c.solomon = 66 := by
  sorry

end solomon_collected_66_cans_l2427_242702


namespace union_of_A_and_B_l2427_242748

-- Define the sets A and B
def A : Set ℝ := {x | 0 < x ∧ x < 2}
def B : Set ℝ := {x | -1 < x ∧ x < 1}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end union_of_A_and_B_l2427_242748


namespace student_calculation_error_l2427_242712

theorem student_calculation_error (x : ℝ) : 
  (8/7) * x = (4/5) * x + 15.75 → x = 45.9375 := by
  sorry

end student_calculation_error_l2427_242712


namespace translation_motions_l2427_242790

/-- Represents a type of motion. -/
inductive Motion
  | Swing
  | VerticalElevator
  | PlanetMovement
  | ConveyorBelt

/-- Determines if a given motion is a translation. -/
def isTranslation (m : Motion) : Prop :=
  match m with
  | Motion.VerticalElevator => True
  | Motion.ConveyorBelt => True
  | _ => False

/-- The theorem stating which motions are translations. -/
theorem translation_motions :
  (∀ m : Motion, isTranslation m ↔ (m = Motion.VerticalElevator ∨ m = Motion.ConveyorBelt)) :=
by sorry

end translation_motions_l2427_242790


namespace set_operations_l2427_242700

-- Define the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < 3}
def B : Set ℝ := {x | -3 < x ∧ x ≤ 3}

-- State the theorem
theorem set_operations :
  (Aᶜ : Set ℝ) = {x | x ≥ 3 ∨ x ≤ -2} ∧
  (A ∩ B : Set ℝ) = {x | -2 < x ∧ x < 3} ∧
  ((A ∩ B)ᶜ : Set ℝ) = {x | x ≥ 3 ∨ x ≤ -2} ∧
  (Aᶜ ∩ B : Set ℝ) = {x | (-3 < x ∧ x ≤ -2) ∨ x = 3} := by
  sorry

end set_operations_l2427_242700


namespace sum_of_three_consecutive_cubes_divisible_by_nine_l2427_242776

theorem sum_of_three_consecutive_cubes_divisible_by_nine (n : ℕ) :
  ∃ k : ℤ, (n^3 + (n+1)^3 + (n+2)^3 : ℤ) = 9 * k := by
sorry

end sum_of_three_consecutive_cubes_divisible_by_nine_l2427_242776


namespace simplify_and_rationalize_l2427_242775

theorem simplify_and_rationalize (x : ℝ) : 
  1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5 := by sorry

end simplify_and_rationalize_l2427_242775


namespace activity_popularity_order_l2427_242733

-- Define the activities
inductive Activity
  | dodgeball
  | natureWalk
  | painting

-- Define the popularity fraction for each activity
def popularity (a : Activity) : Rat :=
  match a with
  | Activity.dodgeball => 13/40
  | Activity.natureWalk => 8/25
  | Activity.painting => 9/20

-- Define a function to compare two activities based on their popularity
def morePopular (a b : Activity) : Prop :=
  popularity a > popularity b

-- Theorem stating the correct order of activities
theorem activity_popularity_order :
  morePopular Activity.painting Activity.dodgeball ∧
  morePopular Activity.dodgeball Activity.natureWalk :=
by
  sorry

#check activity_popularity_order

end activity_popularity_order_l2427_242733


namespace train_length_l2427_242756

/-- The length of a train given its speed and time to pass a point --/
theorem train_length (speed : ℝ) (time : ℝ) (h1 : speed = 36) (h2 : time = 16) :
  speed * time * (5 / 18) = 160 := by
  sorry

#check train_length

end train_length_l2427_242756


namespace taxi_driver_theorem_l2427_242760

def taxi_trips : List Int := [5, -3, 6, -7, 6, -2, -5, 4, 6, -8]

theorem taxi_driver_theorem :
  (taxi_trips.take 7).sum = 0 ∧ taxi_trips.sum = 2 := by sorry

end taxi_driver_theorem_l2427_242760


namespace jose_profit_share_l2427_242743

def calculate_share (investment : ℕ) (duration : ℕ) (total_ratio : ℕ) (total_profit : ℕ) : ℕ :=
  (investment * duration * total_profit) / total_ratio

theorem jose_profit_share 
  (tom_investment : ℕ) (tom_duration : ℕ)
  (jose_investment : ℕ) (jose_duration : ℕ)
  (total_profit : ℕ) :
  tom_investment = 30000 →
  tom_duration = 12 →
  jose_investment = 45000 →
  jose_duration = 10 →
  total_profit = 45000 →
  calculate_share jose_investment jose_duration 
    (tom_investment * tom_duration + jose_investment * jose_duration) 
    total_profit = 25000 := by
  sorry

end jose_profit_share_l2427_242743


namespace election_votes_calculation_l2427_242718

theorem election_votes_calculation (total_votes : ℕ) : 
  (∃ (candidate_votes rival_votes : ℕ),
    candidate_votes = (34 * total_votes) / 100 ∧ 
    rival_votes = candidate_votes + 640 ∧
    candidate_votes + rival_votes = total_votes) →
  total_votes = 2000 := by
sorry

end election_votes_calculation_l2427_242718


namespace intersection_implies_sin_2α_l2427_242735

noncomputable section

-- Define the line l
def line_l (α : Real) (t : Real) : Real × Real :=
  (-1 + t * Real.cos α, -3 + t * Real.sin α)

-- Define the curve C
def curve_C (θ : Real) : Real × Real :=
  let ρ := 4 * Real.cos θ
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the distance between two points
def distance (p1 p2 : Real × Real) : Real :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem intersection_implies_sin_2α (α : Real) :
  ∃ (t1 t2 θ1 θ2 : Real),
    let A := line_l α t1
    let B := line_l α t2
    curve_C θ1 = A ∧
    curve_C θ2 = B ∧
    distance A B = 2 →
    Real.sin (2 * α) = 2/3 := by
  sorry

end

end intersection_implies_sin_2α_l2427_242735


namespace right_triangle_height_l2427_242719

theorem right_triangle_height (a b c h : ℝ) : 
  a = 25 → b = 20 → c^2 = a^2 - b^2 → h * a = 2 * (1/2 * b * c) → h = 12 := by
  sorry

end right_triangle_height_l2427_242719


namespace arithmetic_sum_equals_168_l2427_242759

/-- The sum of an arithmetic sequence with first term 3, common difference 2, and 12 terms -/
def arithmetic_sum : ℕ := 
  let a₁ : ℕ := 3  -- first term
  let d : ℕ := 2   -- common difference
  let n : ℕ := 12  -- number of terms
  n * (2 * a₁ + (n - 1) * d) / 2

/-- Theorem stating that the sum of the arithmetic sequence is 168 -/
theorem arithmetic_sum_equals_168 : arithmetic_sum = 168 := by
  sorry

end arithmetic_sum_equals_168_l2427_242759


namespace equation_equivalence_l2427_242701

theorem equation_equivalence (p q : ℝ) 
  (hp_nonzero : p ≠ 0) (hp_not_five : p ≠ 5) 
  (hq_nonzero : q ≠ 0) (hq_not_seven : q ≠ 7) :
  (3 / p + 5 / q = 1 / 3) ↔ (p = 9 * q / (q - 15)) :=
by sorry

end equation_equivalence_l2427_242701


namespace case_A_case_B_case_C_case_D_l2427_242767

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the number of solutions for a triangle
inductive TriangleSolutions
  | Unique
  | Two
  | None

-- Function to determine the number of solutions for a triangle
def triangleSolutions (t : Triangle) : TriangleSolutions := sorry

-- Theorem for case A
theorem case_A :
  let t : Triangle := { a := 5, b := 7, c := 8, A := 0, B := 0, C := 0 }
  triangleSolutions t = TriangleSolutions.Unique := by sorry

-- Theorem for case B
theorem case_B :
  let t : Triangle := { a := 0, b := 18, c := 20, A := 0, B := 60 * π / 180, C := 0 }
  triangleSolutions t = TriangleSolutions.None := by sorry

-- Theorem for case C
theorem case_C :
  let t : Triangle := { a := 8, b := 8 * Real.sqrt 2, c := 0, A := 0, B := 45 * π / 180, C := 0 }
  triangleSolutions t = TriangleSolutions.Two := by sorry

-- Theorem for case D
theorem case_D :
  let t : Triangle := { a := 30, b := 25, c := 0, A := 150 * π / 180, B := 0, C := 0 }
  triangleSolutions t = TriangleSolutions.Unique := by sorry

end case_A_case_B_case_C_case_D_l2427_242767


namespace last_digit_power_of_two_cycle_last_digit_2018th_power_of_two_l2427_242721

def last_digit (n : ℕ) : ℕ := n % 10

def power_of_two (n : ℕ) : ℕ := 2^n

theorem last_digit_power_of_two_cycle (n : ℕ) :
  last_digit (power_of_two n) = last_digit (power_of_two (n % 4)) :=
sorry

theorem last_digit_2018th_power_of_two :
  last_digit (power_of_two 2018) = 4 :=
sorry

end last_digit_power_of_two_cycle_last_digit_2018th_power_of_two_l2427_242721


namespace parallelogram_fourth_vertex_l2427_242750

def parallelogram (A B C D : ℂ) : Prop :=
  D - A = C - B

theorem parallelogram_fourth_vertex 
  (A B C D : ℂ) 
  (h1 : A = 1 + 3*I) 
  (h2 : B = -I) 
  (h3 : C = 2 + I) 
  (h4 : parallelogram A B C D) : 
  D = 3 + 5*I :=
sorry

end parallelogram_fourth_vertex_l2427_242750


namespace product_digits_sum_base7_l2427_242720

/-- Converts a base-7 number to base-10 --/
def toBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-7 --/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Computes the sum of digits of a number in base-7 --/
def sumOfDigitsBase7 (n : ℕ) : ℕ := sorry

/-- The main theorem --/
theorem product_digits_sum_base7 :
  let x := 35
  let y := 21
  sumOfDigitsBase7 (toBase7 (toBase10 x * toBase10 y)) = 15 := by sorry

end product_digits_sum_base7_l2427_242720


namespace candidate_percentage_l2427_242710

theorem candidate_percentage (passing_mark total_mark : ℕ) 
  (h1 : passing_mark = 160)
  (h2 : total_mark = 300)
  (h3 : (60 : ℕ) * total_mark / 100 = passing_mark + 20)
  (h4 : passing_mark - 40 > 0) : 
  (passing_mark - 40) * 100 / total_mark = 40 := by
  sorry

end candidate_percentage_l2427_242710


namespace max_value_xy_over_z_l2427_242773

theorem max_value_xy_over_z (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 4 * x^2 - 3 * x * y + y^2 - z = 0) :
  ∃ (M : ℝ), M = 1 ∧ ∀ (w : ℝ), w = x * y / z → w ≤ M :=
sorry

end max_value_xy_over_z_l2427_242773


namespace sequence_ratio_l2427_242726

/-- Given a sequence a with sum S of its first n terms satisfying 3S_n - 6 = 2a_n,
    prove that S_5 / a_5 = 11/16 -/
theorem sequence_ratio (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h : ∀ n, 3 * S n - 6 = 2 * a n) :
  S 5 / a 5 = 11 / 16 := by
  sorry

end sequence_ratio_l2427_242726


namespace y_properties_l2427_242788

/-- A function y(x) composed of two directly proportional components -/
def y (x : ℝ) (k₁ k₂ : ℝ) : ℝ := k₁ * (x - 3) + k₂ * (x^2 + 1)

/-- Theorem stating the properties of the function y(x) -/
theorem y_properties :
  ∃ (k₁ k₂ : ℝ),
    (y 0 k₁ k₂ = -2) ∧
    (y 1 k₁ k₂ = 4) ∧
    (∀ x, y x k₁ k₂ = 4*x^2 + 2*x - 2) ∧
    (y (-1) k₁ k₂ = 0) ∧
    (y (1/2) k₁ k₂ = 0) := by
  sorry

#check y_properties

end y_properties_l2427_242788


namespace manager_salary_is_220000_l2427_242799

/-- Represents the average salary of managers at Plutarch Enterprises -/
def manager_salary : ℝ := 220000

/-- Theorem stating that the average salary of managers at Plutarch Enterprises is $220,000 -/
theorem manager_salary_is_220000 
  (marketer_percent : ℝ) (engineer_percent : ℝ) (sales_percent : ℝ) (manager_percent : ℝ)
  (marketer_salary : ℝ) (engineer_salary : ℝ) (sales_salary : ℝ) (total_avg_salary : ℝ)
  (h1 : marketer_percent = 0.60)
  (h2 : engineer_percent = 0.20)
  (h3 : sales_percent = 0.10)
  (h4 : manager_percent = 0.10)
  (h5 : marketer_salary = 50000)
  (h6 : engineer_salary = 80000)
  (h7 : sales_salary = 70000)
  (h8 : total_avg_salary = 75000)
  (h9 : marketer_percent + engineer_percent + sales_percent + manager_percent = 1) :
  manager_salary = 220000 := by
  sorry

#check manager_salary_is_220000

end manager_salary_is_220000_l2427_242799


namespace correct_product_l2427_242752

theorem correct_product (a b c : ℚ) : 
  a = 0.125 → b = 3.2 → c = 4.0 → 
  (125 : ℚ) * 320 = 40000 → a * b = c := by
sorry

end correct_product_l2427_242752


namespace one_fifth_of_ten_x_plus_three_l2427_242739

theorem one_fifth_of_ten_x_plus_three (x : ℝ) : (1 / 5) * (10 * x + 3) = 2 * x + 3 / 5 := by
  sorry

end one_fifth_of_ten_x_plus_three_l2427_242739


namespace three_X_five_l2427_242782

/-- The operation X defined for real numbers -/
def X (a b : ℝ) : ℝ := b + 15 * a - 2 * a^2

/-- Theorem stating that 3X5 equals 32 -/
theorem three_X_five : X 3 5 = 32 := by
  sorry

end three_X_five_l2427_242782


namespace union_complement_equals_set_l2427_242727

def U : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {1, 4}
def N : Set Nat := {2, 5}

theorem union_complement_equals_set : N ∪ (U \ M) = {2, 3, 5} := by
  sorry

end union_complement_equals_set_l2427_242727


namespace imaginary_unit_power_l2427_242728

theorem imaginary_unit_power (i : ℂ) : i^2 = -1 → i^607 = -i := by
  sorry

end imaginary_unit_power_l2427_242728


namespace last_two_digits_of_product_l2427_242717

theorem last_two_digits_of_product (n : ℕ) : 
  (33 * 92025^1989) % 100 = 25 := by
  sorry

#eval (33 * 92025^1989) % 100

end last_two_digits_of_product_l2427_242717


namespace cars_sold_first_day_l2427_242740

theorem cars_sold_first_day (total : ℕ) (second_day : ℕ) (third_day : ℕ)
  (h1 : total = 57)
  (h2 : second_day = 16)
  (h3 : third_day = 27) :
  total - second_day - third_day = 14 := by
  sorry

end cars_sold_first_day_l2427_242740


namespace inverse_proportion_points_order_l2427_242757

/-- Given points A(x₁, -6), B(x₂, -2), C(x₃, 3) on the graph of y = -12/x,
    prove that x₃ < x₁ < x₂ -/
theorem inverse_proportion_points_order (x₁ x₂ x₃ : ℝ) : 
  (-6 : ℝ) = -12 / x₁ → 
  (-2 : ℝ) = -12 / x₂ → 
  (3 : ℝ) = -12 / x₃ → 
  x₃ < x₁ ∧ x₁ < x₂ := by
  sorry

#check inverse_proportion_points_order

end inverse_proportion_points_order_l2427_242757


namespace max_value_quadratic_l2427_242758

theorem max_value_quadratic (x : ℝ) (h : 0 < x ∧ x < 1.5) : 
  ∃ (y : ℝ), y = 4 * x * (3 - 2 * x) ∧ ∀ (z : ℝ), z = 4 * x * (3 - 2 * x) → z ≤ y :=
by
  sorry

end max_value_quadratic_l2427_242758


namespace shooting_probabilities_l2427_242703

-- Define the probabilities for each ring
def P_10 : ℝ := 0.24
def P_9 : ℝ := 0.28
def P_8 : ℝ := 0.19
def P_7 : ℝ := 0.16
def P_below_7 : ℝ := 0.13

-- Theorem for the three probability calculations
theorem shooting_probabilities :
  (P_10 + P_9 = 0.52) ∧
  (P_10 + P_9 + P_8 + P_7 = 0.87) ∧
  (P_7 + P_below_7 = 0.29) := by
  sorry

end shooting_probabilities_l2427_242703


namespace quadratic_roots_shift_l2427_242725

theorem quadratic_roots_shift (a b c : ℝ) (ha : a ≠ 0) :
  let f (x : ℝ) := a * x^2 + b * x + c
  let g (y : ℝ) := a * y^2 + (b - 2*a) * y + (a - b + c)
  ∀ (x y : ℝ), f x = 0 ∧ g y = 0 → y = x + 1 :=
by sorry

end quadratic_roots_shift_l2427_242725


namespace fraction_above_line_l2427_242761

/-- A square in the 2D plane --/
structure Square where
  bottomLeft : ℝ × ℝ
  topRight : ℝ × ℝ

/-- A line in the 2D plane defined by two points --/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Calculate the area of a square --/
def squareArea (s : Square) : ℝ :=
  let (x1, y1) := s.bottomLeft
  let (x2, y2) := s.topRight
  (x2 - x1) * (y2 - y1)

/-- Calculate the area of the part of the square above the line --/
def areaAboveLine (s : Square) (l : Line) : ℝ :=
  sorry  -- The actual calculation would go here

/-- The main theorem --/
theorem fraction_above_line (s : Square) (l : Line) : 
  s.bottomLeft = (4, 0) → 
  s.topRight = (9, 5) → 
  l.point1 = (4, 1) → 
  l.point2 = (9, 5) → 
  areaAboveLine s l / squareArea s = 9 / 10 := by
  sorry


end fraction_above_line_l2427_242761


namespace nabla_problem_l2427_242771

-- Define the ∇ operation
def nabla (a b : ℕ) : ℕ := 3 + a^b

-- Theorem to prove
theorem nabla_problem : nabla (nabla 2 1) 4 = 628 := by
  sorry

end nabla_problem_l2427_242771


namespace fraction_sum_bound_l2427_242786

theorem fraction_sum_bound (a b c : ℕ+) (h : (a : ℚ)⁻¹ + (b : ℚ)⁻¹ + (c : ℚ)⁻¹ < 1) :
  (a : ℚ)⁻¹ + (b : ℚ)⁻¹ + (c : ℚ)⁻¹ ≤ 41 / 42 := by
  sorry

end fraction_sum_bound_l2427_242786


namespace maci_pen_cost_l2427_242732

/-- The total cost of pens for Maci --/
def total_cost (blue_pens red_pens : ℕ) (blue_cost : ℚ) : ℚ :=
  (blue_pens : ℚ) * blue_cost + (red_pens : ℚ) * (2 * blue_cost)

/-- Theorem stating that Maci's total cost for pens is $4.00 --/
theorem maci_pen_cost :
  total_cost 10 15 (1/10) = 4 := by
  sorry

end maci_pen_cost_l2427_242732


namespace juan_friends_seating_l2427_242769

theorem juan_friends_seating (n : ℕ) : n = 5 :=
  by
    -- Define the conditions
    have juan_fixed : True := True.intro
    have jamal_next_to_juan : ℕ := 2
    have total_arrangements : ℕ := 48

    -- State the relationship between n and the conditions
    have seating_equation : jamal_next_to_juan * Nat.factorial (n - 1) = total_arrangements := by sorry

    -- Prove that n = 5 satisfies the equation
    sorry

end juan_friends_seating_l2427_242769


namespace correct_calculation_l2427_242738

theorem correct_calculation (a : ℝ) : (2*a)^2 / (4*a) = a := by
  sorry

end correct_calculation_l2427_242738


namespace bryans_deposit_l2427_242783

theorem bryans_deposit (mark_deposit : ℕ) (bryan_deposit : ℕ) 
  (h1 : mark_deposit = 88)
  (h2 : bryan_deposit < 5 * mark_deposit)
  (h3 : mark_deposit + bryan_deposit = 400) :
  bryan_deposit = 312 := by
sorry

end bryans_deposit_l2427_242783


namespace value_of_a_l2427_242768

theorem value_of_a (a b c d : ℤ) 
  (eq1 : a = b + 3)
  (eq2 : b = c + 6)
  (eq3 : c = d + 15)
  (eq4 : d = 50) : 
  a = 74 := by
  sorry

end value_of_a_l2427_242768


namespace geometric_sequence_property_l2427_242766

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property 
  (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_prod : a 7 * a 12 = 5) : 
  a 8 * a 9 * a 10 * a 11 = 25 := by
  sorry

end geometric_sequence_property_l2427_242766


namespace square_root_of_nine_l2427_242723

theorem square_root_of_nine : Real.sqrt 9 = 3 := by
  sorry

end square_root_of_nine_l2427_242723


namespace share_price_increase_l2427_242737

theorem share_price_increase (P : ℝ) (h1 : P > 0) : 
  let Q2 := 1.5 * P
  let Q1 := P * (1 + X / 100)
  X = 20 →
  Q2 = Q1 * 1.25 ∧ Q2 = 1.5 * P :=
by sorry

end share_price_increase_l2427_242737


namespace monster_family_eyes_total_l2427_242794

/-- The number of eyes in the extended monster family -/
def monster_family_eyes : ℕ :=
  let mom_eyes := 1
  let dad_eyes := 3
  let mom_dad_kids_eyes := 3 * 4
  let mom_previous_child_eyes := 5
  let dad_previous_children_eyes := 6 + 2
  let dad_ex_wife_eyes := 1
  let dad_ex_wife_partner_eyes := 7
  let dad_ex_wife_child_eyes := 8
  mom_eyes + dad_eyes + mom_dad_kids_eyes + mom_previous_child_eyes +
  dad_previous_children_eyes + dad_ex_wife_eyes + dad_ex_wife_partner_eyes +
  dad_ex_wife_child_eyes

/-- The total number of eyes in the extended monster family is 45 -/
theorem monster_family_eyes_total :
  monster_family_eyes = 45 := by sorry

end monster_family_eyes_total_l2427_242794


namespace inscribed_circle_rectangle_area_l2427_242781

/-- Given a rectangle with an inscribed circle of radius 7 and a length-to-width ratio of 3:1,
    prove that the area of the rectangle is 588. -/
theorem inscribed_circle_rectangle_area :
  ∀ (length width radius : ℝ),
    radius = 7 →
    length = 3 * width →
    width = 2 * radius →
    length * width = 588 :=
by
  sorry

end inscribed_circle_rectangle_area_l2427_242781


namespace junk_mail_distribution_l2427_242745

/-- Given a block with houses and junk mail to distribute, calculate the number of pieces per house -/
def junk_mail_per_house (num_houses : ℕ) (total_junk_mail : ℕ) : ℕ :=
  total_junk_mail / num_houses

/-- Theorem: In a block with 20 houses and 640 pieces of junk mail, each house receives 32 pieces -/
theorem junk_mail_distribution :
  junk_mail_per_house 20 640 = 32 := by
  sorry

end junk_mail_distribution_l2427_242745


namespace train_speed_l2427_242715

/-- Proves that a train with given parameters has a specific speed -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (man_speed : ℝ) : 
  train_length = 300 →
  crossing_time = 9 →
  man_speed = 3 →
  ∃ (train_speed : ℝ), train_speed = 117 ∧ 
    (train_speed * 1000 / 3600 + man_speed * 1000 / 3600) * crossing_time = train_length :=
by sorry


end train_speed_l2427_242715


namespace Q_proper_subset_of_P_l2427_242716

def P : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def Q : Set ℕ := {2, 3, 5, 6}

theorem Q_proper_subset_of_P : Q ⊂ P := by
  sorry

end Q_proper_subset_of_P_l2427_242716


namespace min_fraction_value_l2427_242793

theorem min_fraction_value (x y : ℝ) (hx : -5 ≤ x ∧ x ≤ -3) (hy : -3 ≤ y ∧ y ≤ 1) :
  (∀ x' y', -5 ≤ x' ∧ x' ≤ -3 → -3 ≤ y' ∧ y' ≤ 1 → (x' + y') / x' ≥ (x + y) / x) →
  (x + y) / x = 0.8 := by
sorry

end min_fraction_value_l2427_242793


namespace cube_sum_2001_l2427_242770

theorem cube_sum_2001 :
  ∀ a b c : ℕ+,
  a^3 + b^3 + c^3 = 2001 ↔ (a = 10 ∧ b = 10 ∧ c = 1) ∨ (a = 10 ∧ b = 1 ∧ c = 10) ∨ (a = 1 ∧ b = 10 ∧ c = 10) :=
by sorry

end cube_sum_2001_l2427_242770


namespace circle_condition_l2427_242762

/-- The equation of a potential circle with parameter m -/
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 2*y + 5*m = 0

/-- Theorem stating the necessary and sufficient condition for the equation to represent a circle -/
theorem circle_condition (m : ℝ) :
  (∃ (x₀ y₀ r : ℝ), ∀ (x y : ℝ), circle_equation x y m ↔ (x - x₀)^2 + (y - y₀)^2 = r^2) ↔ m < 1 :=
sorry

end circle_condition_l2427_242762


namespace length_of_cd_l2427_242711

/-- Given a line segment CD with points M and N on it, prove that CD has length 57.6 -/
theorem length_of_cd (C D M N : ℝ × ℝ) : 
  (∃ t : ℝ, M = (1 - t) • C + t • D ∧ 0 < t ∧ t < 1/2) →  -- M is on CD and same side of midpoint
  (∃ s : ℝ, N = (1 - s) • C + s • D ∧ 0 < s ∧ s < 1/2) →  -- N is on CD and same side of midpoint
  (dist C M) / (dist M D) = 3/5 →                         -- M divides CD in ratio 3:5
  (dist C N) / (dist N D) = 4/5 →                         -- N divides CD in ratio 4:5
  dist M N = 4 →                                          -- Length of MN is 4
  dist C D = 57.6 :=                                      -- Length of CD is 57.6
by sorry

end length_of_cd_l2427_242711


namespace plane_equation_correct_l2427_242791

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space defined by parametric equations -/
structure Line3D where
  x : ℝ → ℝ
  y : ℝ → ℝ
  z : ℝ → ℝ

/-- A plane in 3D space defined by the equation Ax + By + Cz + D = 0 -/
structure Plane where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- Check if a point lies on a plane -/
def pointOnPlane (plane : Plane) (point : Point3D) : Prop :=
  plane.A * point.x + plane.B * point.y + plane.C * point.z + plane.D = 0

/-- Check if a line is contained in a plane -/
def lineInPlane (plane : Plane) (line : Line3D) : Prop :=
  ∀ t, plane.A * line.x t + plane.B * line.y t + plane.C * line.z t + plane.D = 0

/-- The given point (1,2,-3) -/
def givenPoint : Point3D := ⟨1, 2, -3⟩

/-- The given line (x - 2)/4 = (y + 3)/(-6) = (z - 4)/2 -/
def givenLine : Line3D :=
  ⟨λ t => 4*t + 2, λ t => -6*t - 3, λ t => 2*t + 4⟩

/-- The plane we want to prove is correct -/
def resultPlane : Plane := ⟨3, 1, -3, 2⟩

theorem plane_equation_correct :
  pointOnPlane resultPlane givenPoint ∧
  lineInPlane resultPlane givenLine ∧
  resultPlane.A > 0 ∧
  Nat.gcd (Nat.gcd (Int.natAbs resultPlane.A) (Int.natAbs resultPlane.B))
          (Nat.gcd (Int.natAbs resultPlane.C) (Int.natAbs resultPlane.D)) = 1 :=
by sorry

end plane_equation_correct_l2427_242791


namespace matching_pair_guarantee_l2427_242755

/-- The number of different colors of plates -/
def num_colors : ℕ := 5

/-- The total number of plates to be pulled out -/
def total_plates : ℕ := 6

/-- The minimum number of plates needed to guarantee a matching pair -/
def min_matching_pair : ℕ := total_plates

theorem matching_pair_guarantee :
  min_matching_pair = total_plates :=
sorry

end matching_pair_guarantee_l2427_242755


namespace team_a_min_wins_l2427_242724

theorem team_a_min_wins (total_games : ℕ) (lost_games : ℕ) (min_points : ℕ) 
  (win_points draw_points lose_points : ℕ) :
  total_games = 5 →
  lost_games = 1 →
  min_points = 7 →
  win_points = 3 →
  draw_points = 1 →
  lose_points = 0 →
  ∃ (won_games : ℕ),
    won_games ≥ 2 ∧
    won_games + lost_games ≤ total_games ∧
    won_games * win_points + (total_games - won_games - lost_games) * draw_points > min_points :=
by sorry

end team_a_min_wins_l2427_242724


namespace tomatoes_sold_to_maxwell_l2427_242741

/-- Calculates the amount of tomatoes sold to Mrs. Maxwell -/
theorem tomatoes_sold_to_maxwell 
  (total_harvest : ℝ) 
  (sold_to_wilson : ℝ) 
  (not_sold : ℝ) 
  (h1 : total_harvest = 245.5)
  (h2 : sold_to_wilson = 78)
  (h3 : not_sold = 42) :
  total_harvest - sold_to_wilson - not_sold = 125.5 := by
sorry

end tomatoes_sold_to_maxwell_l2427_242741


namespace students_taking_both_languages_l2427_242707

theorem students_taking_both_languages (total : ℕ) (french : ℕ) (german : ℕ) (neither : ℕ) :
  total = 79 →
  french = 41 →
  german = 22 →
  neither = 25 →
  french + german - (total - neither) = 9 :=
by sorry

end students_taking_both_languages_l2427_242707


namespace cos_alpha_value_l2427_242709

theorem cos_alpha_value (α : Real) : 
  (∃ x y : Real, x ≤ 0 ∧ y = -4/3 * x ∧ 
   x = Real.cos α ∧ y = Real.sin α) → 
  Real.cos α = -3/5 := by
sorry

end cos_alpha_value_l2427_242709


namespace initial_piggy_bank_amount_l2427_242765

-- Define the variables
def weekly_allowance : ℕ := 10
def weeks : ℕ := 8
def final_amount : ℕ := 83

-- Define the function to calculate the amount added to the piggy bank
def amount_added (w : ℕ) : ℕ := w * (weekly_allowance / 2)

-- Theorem statement
theorem initial_piggy_bank_amount :
  ∃ (initial : ℕ), initial + amount_added weeks = final_amount :=
sorry

end initial_piggy_bank_amount_l2427_242765


namespace area_between_circles_and_x_axis_l2427_242736

/-- The area of the region bound by two circles and the x-axis -/
theorem area_between_circles_and_x_axis 
  (center_C : ℝ × ℝ) 
  (center_D : ℝ × ℝ) 
  (radius : ℝ) : 
  center_C = (3, 5) → 
  center_D = (13, 5) → 
  radius = 5 → 
  ∃ (area : ℝ), area = 50 - 25 * Real.pi := by
  sorry

end area_between_circles_and_x_axis_l2427_242736


namespace complex_number_quadrant_l2427_242754

theorem complex_number_quadrant (z : ℂ) (h : z / Complex.I = 2 - 3 * Complex.I) : 
  Complex.re z > 0 ∧ Complex.im z > 0 :=
sorry

end complex_number_quadrant_l2427_242754


namespace area_of_region_l2427_242787

-- Define the region
def region (x y : ℝ) : Prop :=
  |x - 2*y^2| + x + 2*y^2 ≤ 8 - 4*y

-- Define symmetry about Y-axis
def symmetricAboutYAxis (S : Set (ℝ × ℝ)) : Prop :=
  ∀ x y, (x, y) ∈ S ↔ (-x, y) ∈ S

-- Theorem statement
theorem area_of_region :
  ∃ S : Set (ℝ × ℝ),
    (∀ x y, (x, y) ∈ S ↔ region x y) ∧
    symmetricAboutYAxis S ∧
    MeasureTheory.volume S = 30 := by
  sorry

end area_of_region_l2427_242787


namespace equal_intercept_line_equation_l2427_242747

/-- A line passing through point (2, 1) with equal intercepts on the coordinate axes -/
structure EqualInterceptLine where
  /-- The slope of the line -/
  m : ℝ
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The line passes through point (2, 1) -/
  point_condition : m * 2 + b = 1
  /-- The line has equal intercepts on the coordinate axes -/
  equal_intercepts : b = 0 ∨ m = -1

/-- The equation of a line with equal intercepts passing through (2, 1) is either 2x - y = 0 or x + y - 3 = 0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.m = 1/2 ∧ l.b = 0) ∨ (l.m = -1 ∧ l.b = 3) :=
sorry

end equal_intercept_line_equation_l2427_242747
