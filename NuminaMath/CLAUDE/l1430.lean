import Mathlib

namespace club_officer_selection_l1430_143058

/-- Represents the number of ways to choose club officers under specific conditions -/
def choose_officers (total_members boys girls : ℕ) : ℕ :=
  2 * boys * girls * (boys - 1)

/-- Theorem stating the number of ways to choose club officers -/
theorem club_officer_selection :
  let total_members : ℕ := 30
  let boys : ℕ := 15
  let girls : ℕ := 15
  choose_officers total_members boys girls = 6300 := by
  sorry

#eval choose_officers 30 15 15

end club_officer_selection_l1430_143058


namespace cindys_calculation_l1430_143081

theorem cindys_calculation (x : ℚ) : 4 * (x / 2 - 6) = 24 → (2 * x - 4) / 6 = 22 / 3 := by
  sorry

end cindys_calculation_l1430_143081


namespace f_derivative_at_one_l1430_143061

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x + x

theorem f_derivative_at_one :
  deriv f 1 = 3 := by sorry

end f_derivative_at_one_l1430_143061


namespace cubic_roots_problem_l1430_143050

/-- Given a cubic polynomial x^3 + ax^2 + bx + c, returns the sum of its roots -/
def sumOfRoots (a b c : ℝ) : ℝ := -a

/-- Given a cubic polynomial x^3 + ax^2 + bx + c, returns the product of its roots -/
def productOfRoots (a b c : ℝ) : ℝ := -c

theorem cubic_roots_problem (p q r u v w : ℝ) :
  (∀ x, x^3 + 2*x^2 + 5*x - 8 = (x - p)*(x - q)*(x - r)) →
  (∀ x, x^3 + u*x^2 + v*x + w = (x - (p + q))*(x - (q + r))*(x - (r + p))) →
  w = 18 := by
  sorry

end cubic_roots_problem_l1430_143050


namespace purchase_cost_l1430_143082

theorem purchase_cost (pretzel_cost : ℝ) (chip_cost_percentage : ℝ) : 
  pretzel_cost = 4 →
  chip_cost_percentage = 175 →
  2 * pretzel_cost + 2 * (pretzel_cost * chip_cost_percentage / 100) = 22 := by
  sorry

end purchase_cost_l1430_143082


namespace eggs_leftover_l1430_143088

theorem eggs_leftover (daniel : Nat) (eliza : Nat) (fiona : Nat) (george : Nat)
  (h1 : daniel = 53)
  (h2 : eliza = 68)
  (h3 : fiona = 26)
  (h4 : george = 47) :
  (daniel + eliza + fiona + george) % 15 = 14 := by
  sorry

end eggs_leftover_l1430_143088


namespace f_max_value_l1430_143047

/-- The function f(x) = 9x - 4x^2 -/
def f (x : ℝ) : ℝ := 9*x - 4*x^2

/-- The maximum value of f(x) is 81/16 -/
theorem f_max_value : ∀ x : ℝ, f x ≤ 81/16 := by sorry

end f_max_value_l1430_143047


namespace triangle_ratio_l1430_143051

theorem triangle_ratio (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  a * Real.sin A * Real.sin B + b * (Real.cos A)^2 = Real.sqrt 3 * a →
  b / a = Real.sqrt 3 / 3 := by
sorry

end triangle_ratio_l1430_143051


namespace difference_exists_l1430_143021

def sequence_property (x : ℕ → ℕ) : Prop :=
  x 1 = 1 ∧ ∀ n, x n < x (n + 1) ∧ x (n + 1) ≤ 2 * n

theorem difference_exists (x : ℕ → ℕ) (h : sequence_property x) :
  ∀ k : ℕ, k > 0 → ∃ r s, x r - x s = k := by
  sorry

end difference_exists_l1430_143021


namespace zinc_copper_mixture_weight_l1430_143053

/-- Calculates the total weight of a zinc-copper mixture given the ratio and zinc weight -/
theorem zinc_copper_mixture_weight 
  (zinc_ratio : ℚ) 
  (copper_ratio : ℚ) 
  (zinc_weight : ℚ) 
  (h1 : zinc_ratio = 9) 
  (h2 : copper_ratio = 11) 
  (h3 : zinc_weight = 26.1) : 
  zinc_weight + (copper_ratio / zinc_ratio) * zinc_weight = 58 := by
sorry

end zinc_copper_mixture_weight_l1430_143053


namespace reciprocal_of_repeating_decimal_l1430_143024

/-- The reciprocal of the common fraction form of 0.353535... is 99/35 -/
theorem reciprocal_of_repeating_decimal :
  let x : ℚ := 35 / 99  -- Common fraction form of 0.353535...
  (1 : ℚ) / x = 99 / 35 := by
  sorry

end reciprocal_of_repeating_decimal_l1430_143024


namespace aeroplane_distance_l1430_143080

theorem aeroplane_distance (speed1 : ℝ) (time1 : ℝ) (speed2 : ℝ) (time2 : ℝ) 
  (h1 : speed1 = 590)
  (h2 : time1 = 8)
  (h3 : speed2 = 1716.3636363636363)
  (h4 : time2 = 2.75)
  (h5 : speed1 * time1 = speed2 * time2) : 
  speed1 * time1 = 4720 := by
  sorry

end aeroplane_distance_l1430_143080


namespace pyramid_height_l1430_143084

/-- Given a square pyramid whose lateral faces unfold into a square with side length 18,
    prove that the height of the pyramid is 6. -/
theorem pyramid_height (s : ℝ) (h : s > 0) : 
  s * s = 18 * 18 / 2 → (6 : ℝ) * s = 18 * 18 / 2 := by
  sorry

#check pyramid_height

end pyramid_height_l1430_143084


namespace inner_triangle_area_l1430_143068

/-- Given a triangle with area T, prove that the area of the triangle formed by
    joining the points that divide each side into three equal segments is T/9. -/
theorem inner_triangle_area (T : ℝ) (h : T > 0) :
  ∃ (M : ℝ), M = T / 9 ∧ M / T = 1 / 9 := by
  sorry

end inner_triangle_area_l1430_143068


namespace water_percentage_in_fresh_grapes_l1430_143004

/-- The percentage of water in fresh grapes -/
def water_percentage_fresh : ℝ := 90

/-- The percentage of water in dried grapes -/
def water_percentage_dried : ℝ := 20

/-- The weight of fresh grapes in kg -/
def fresh_weight : ℝ := 30

/-- The weight of dried grapes in kg -/
def dried_weight : ℝ := 3.75

/-- Theorem stating that the percentage of water in fresh grapes is 90% -/
theorem water_percentage_in_fresh_grapes :
  water_percentage_fresh = 90 :=
by sorry

end water_percentage_in_fresh_grapes_l1430_143004


namespace sampling_plans_correct_l1430_143077

/-- Represents a canning factory with its production rate and operating hours. -/
structure CanningFactory where
  production_rate : ℕ  -- cans per hour
  operating_hours : ℕ

/-- Represents a sampling plan with the number of cans sampled and the interval between samples. -/
structure SamplingPlan where
  cans_per_sample : ℕ
  sample_interval : ℕ  -- in minutes

/-- Calculates the total number of cans sampled in a day given a factory and a sampling plan. -/
def total_sampled_cans (factory : CanningFactory) (plan : SamplingPlan) : ℕ :=
  (factory.operating_hours * 60 / plan.sample_interval) * plan.cans_per_sample

/-- Theorem stating that the given sampling plans result in the required number of sampled cans. -/
theorem sampling_plans_correct (factory : CanningFactory) :
  factory.production_rate = 120000 ∧ factory.operating_hours = 12 →
  (∃ plan1200 : SamplingPlan, total_sampled_cans factory plan1200 = 1200 ∧ 
    plan1200.cans_per_sample = 10 ∧ plan1200.sample_interval = 6) ∧
  (∃ plan980 : SamplingPlan, total_sampled_cans factory plan980 = 980 ∧ 
    plan980.cans_per_sample = 49 ∧ plan980.sample_interval = 36) := by
  sorry

end sampling_plans_correct_l1430_143077


namespace number_theory_statements_l1430_143044

theorem number_theory_statements :
  (∃ n : ℕ, 20 = 4 * n) ∧
  (∃ n : ℕ, 209 = 19 * n) ∧ ¬(∃ n : ℕ, 63 = 19 * n) ∧
  ¬(∃ n : ℕ, 75 = 12 * n) ∧ ¬(∃ n : ℕ, 29 = 12 * n) ∧
  (∃ n : ℕ, 33 = 11 * n) ∧ ¬(∃ n : ℕ, 64 = 11 * n) ∧
  (∃ n : ℕ, 180 = 9 * n) := by
sorry

end number_theory_statements_l1430_143044


namespace arrangements_with_non_adjacent_students_l1430_143093

def number_of_students : ℕ := 5

-- Total number of permutations for n students
def total_permutations (n : ℕ) : ℕ := n.factorial

-- Number of permutations where A and B are adjacent
def adjacent_permutations (n : ℕ) : ℕ := 2 * (n - 1).factorial

theorem arrangements_with_non_adjacent_students :
  total_permutations number_of_students - adjacent_permutations number_of_students = 72 := by
  sorry

end arrangements_with_non_adjacent_students_l1430_143093


namespace quadratic_form_with_factor_l1430_143020

/-- A quadratic expression with (x + 3) as a factor and m = 2 -/
def quadratic_expression (c : ℝ) (x : ℝ) : ℝ :=
  2 * (x + 3) * (x + c)

/-- Theorem stating the form of the quadratic expression -/
theorem quadratic_form_with_factor (f : ℝ → ℝ) :
  (∃ (g : ℝ → ℝ), ∀ x, f x = (x + 3) * g x) →  -- (x + 3) is a factor
  (∃ c, ∀ x, f x = quadratic_expression c x) :=
by
  sorry

#check quadratic_form_with_factor

end quadratic_form_with_factor_l1430_143020


namespace inverse_exp_range_l1430_143017

noncomputable def f : ℝ → ℝ := Real.log

theorem inverse_exp_range (a b : ℝ) :
  (∀ x, f x = Real.log x) →
  (|f a| = |f b|) →
  (a ≠ b) →
  (∀ x > 2, ∃ a b : ℝ, a + b = x ∧ |f a| = |f b| ∧ a ≠ b) ∧
  (|f a| = |f b| ∧ a ≠ b → a + b > 2) :=
sorry

end inverse_exp_range_l1430_143017


namespace xfx_nonnegative_set_l1430_143036

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def monotone_decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f y ≤ f x

theorem xfx_nonnegative_set (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_monotone : monotone_decreasing_on f (Set.Iic 0))
  (h_f2 : f 2 = 0) :
  {x : ℝ | x * f x ≥ 0} = Set.Icc (-2) 0 ∪ Set.Ici 2 := by
  sorry

end xfx_nonnegative_set_l1430_143036


namespace largest_points_with_empty_square_fifteen_points_optimal_l1430_143031

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square in 2D space -/
structure Square where
  center : Point
  side_length : ℝ

/-- Checks if a point is inside a square -/
def is_point_inside_square (p : Point) (s : Square) : Prop :=
  abs (p.x - s.center.x) ≤ s.side_length / 2 ∧
  abs (p.y - s.center.y) ≤ s.side_length / 2

/-- The main theorem -/
theorem largest_points_with_empty_square :
  ∀ (points : List Point),
    (∀ p ∈ points, 0 < p.x ∧ p.x < 4 ∧ 0 < p.y ∧ p.y < 4) →
    points.length ≤ 15 →
    ∃ (s : Square),
      s.side_length = 1 ∧
      0 ≤ s.center.x ∧ s.center.x ≤ 3 ∧
      0 ≤ s.center.y ∧ s.center.y ≤ 3 ∧
      ∀ p ∈ points, ¬is_point_inside_square p s :=
by sorry

/-- The optimality of 15 -/
theorem fifteen_points_optimal :
  ∃ (points : List Point),
    points.length = 16 ∧
    (∀ p ∈ points, 0 < p.x ∧ p.x < 4 ∧ 0 < p.y ∧ p.y < 4) ∧
    ∀ (s : Square),
      s.side_length = 1 →
      0 ≤ s.center.x ∧ s.center.x ≤ 3 →
      0 ≤ s.center.y ∧ s.center.y ≤ 3 →
      ∃ p ∈ points, is_point_inside_square p s :=
by sorry

end largest_points_with_empty_square_fifteen_points_optimal_l1430_143031


namespace algebraic_expression_value_l1430_143045

theorem algebraic_expression_value (a : ℤ) (h : a = -2) : a + 1 = -1 := by
  sorry

end algebraic_expression_value_l1430_143045


namespace simple_interest_principal_l1430_143016

/-- Simple interest calculation -/
theorem simple_interest_principal (rate : ℝ) (time : ℝ) (interest : ℝ) (principal : ℝ) :
  rate = 0.05 →
  time = 1 →
  interest = 500 →
  principal * rate * time = interest →
  principal = 10000 := by
sorry

end simple_interest_principal_l1430_143016


namespace octahedron_triangles_l1430_143078

/-- The number of vertices in a regular octahedron -/
def octahedron_vertices : ℕ := 8

/-- The number of vertices required to form a triangle -/
def triangle_vertices : ℕ := 3

/-- The number of distinct triangles that can be constructed by connecting three different vertices of a regular octahedron -/
def distinct_triangles : ℕ := Nat.choose octahedron_vertices triangle_vertices

theorem octahedron_triangles : distinct_triangles = 56 := by sorry

end octahedron_triangles_l1430_143078


namespace triangle_rectangle_side_ratio_l1430_143071

/-- Given an equilateral triangle and a rectangle with the same perimeter and a specific length-width ratio for the rectangle, this theorem proves that the ratio of the triangle's side length to the rectangle's length is 1. -/
theorem triangle_rectangle_side_ratio (perimeter : ℝ) (length_width_ratio : ℝ) :
  perimeter > 0 →
  length_width_ratio = 2 →
  let triangle_side := perimeter / 3
  let rectangle_width := perimeter / (2 * (length_width_ratio + 1))
  let rectangle_length := length_width_ratio * rectangle_width
  (triangle_side / rectangle_length) = 1 := by
  sorry

#check triangle_rectangle_side_ratio

end triangle_rectangle_side_ratio_l1430_143071


namespace sum_using_splitting_terms_l1430_143019

/-- The sum of (-2017⅔) + 2016¾ + (-2015⅚) + 16½ using the method of splitting terms -/
theorem sum_using_splitting_terms :
  (-2017 - 2/3) + (2016 + 3/4) + (-2015 - 5/6) + (16 + 1/2) = -2000 - 1/4 := by
  sorry

end sum_using_splitting_terms_l1430_143019


namespace stating_holiday_lodge_assignments_l1430_143086

/-- Represents the number of rooms in the holiday lodge -/
def num_rooms : ℕ := 4

/-- Represents the number of friends staying at the lodge -/
def num_friends : ℕ := 6

/-- Represents the maximum number of friends allowed per room -/
def max_friends_per_room : ℕ := 2

/-- Represents the minimum number of empty rooms required -/
def min_empty_rooms : ℕ := 1

/-- 
Calculates the number of ways to assign friends to rooms 
given the constraints
-/
def num_assignments (n_rooms : ℕ) (n_friends : ℕ) 
  (max_per_room : ℕ) (min_empty : ℕ) : ℕ := 
  sorry

/-- 
Theorem stating that the number of assignments for the given problem is 1080
-/
theorem holiday_lodge_assignments : 
  num_assignments num_rooms num_friends max_friends_per_room min_empty_rooms = 1080 := by
  sorry

end stating_holiday_lodge_assignments_l1430_143086


namespace teacher_age_l1430_143069

/-- Proves that given a class of 50 students with an average age of 18 years, 
    if including the teacher's age changes the average to 19.5 years, 
    then the teacher's age is 94.5 years. -/
theorem teacher_age (num_students : ℕ) (student_avg_age : ℝ) (new_avg_age : ℝ) :
  num_students = 50 →
  student_avg_age = 18 →
  new_avg_age = 19.5 →
  (num_students * student_avg_age + (new_avg_age * (num_students + 1) - num_students * student_avg_age)) = 94.5 :=
by
  sorry

#check teacher_age

end teacher_age_l1430_143069


namespace complement_of_alpha_l1430_143029

-- Define a type for angles in degrees and minutes
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

-- Define the given angle
def alpha : Angle := ⟨75, 12⟩

-- Define the complement of an angle
def complement (a : Angle) : Angle :=
  let totalMinutes := 90 * 60 - (a.degrees * 60 + a.minutes)
  ⟨totalMinutes / 60, totalMinutes % 60⟩

-- Theorem statement
theorem complement_of_alpha :
  complement alpha = ⟨14, 48⟩ := by
  sorry

end complement_of_alpha_l1430_143029


namespace cost_of_scooter_l1430_143098

def scooter_cost (megan_money tara_money : ℕ) : Prop :=
  (tara_money = megan_money + 4) ∧
  (tara_money = 15) ∧
  (megan_money + tara_money = 26)

theorem cost_of_scooter :
  ∃ (megan_money tara_money : ℕ), scooter_cost megan_money tara_money :=
by
  sorry

end cost_of_scooter_l1430_143098


namespace bread_distribution_l1430_143003

theorem bread_distribution (a : ℚ) (d : ℚ) :
  (a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 100) →
  ((a + 3*d) + (a + 4*d) + (a + 2*d))/7 = a + (a + d) →
  a = 5/3 :=
by sorry

end bread_distribution_l1430_143003


namespace smallest_class_size_l1430_143018

theorem smallest_class_size (n : ℕ) : 
  (4*n + 2 > 40) ∧ 
  (∀ m : ℕ, m < n → 4*m + 2 ≤ 40) → 
  4*n + 2 = 42 :=
by sorry

end smallest_class_size_l1430_143018


namespace repair_shop_earnings_121_l1430_143062

/-- Represents the earnings for a repair shop for a week. -/
def repair_shop_earnings (phone_cost laptop_cost computer_cost : ℕ) 
  (phone_repairs laptop_repairs computer_repairs : ℕ) : ℕ :=
  phone_cost * phone_repairs + laptop_cost * laptop_repairs + computer_cost * computer_repairs

/-- Theorem stating that the repair shop's earnings for the week is $121. -/
theorem repair_shop_earnings_121 :
  repair_shop_earnings 11 15 18 5 2 2 = 121 := by
  sorry

end repair_shop_earnings_121_l1430_143062


namespace no_equilateral_from_splice_l1430_143030

/-- Represents a triangle with a 45° angle -/
structure Triangle45 where
  -- We only need to define two sides, as the third is determined by the right angle
  side1 : ℝ
  side2 : ℝ
  positive_sides : 0 < side1 ∧ 0 < side2

/-- Represents the result of splicing two Triangle45 objects -/
inductive SplicedShape
  | Equilateral
  | Other

/-- Function to splice two Triangle45 objects -/
def splice (t1 t2 : Triangle45) : SplicedShape :=
  sorry

/-- Theorem stating that splicing two Triangle45 objects cannot result in an equilateral triangle -/
theorem no_equilateral_from_splice (t1 t2 : Triangle45) :
  splice t1 t2 ≠ SplicedShape.Equilateral :=
sorry

end no_equilateral_from_splice_l1430_143030


namespace sandwich_combinations_l1430_143049

theorem sandwich_combinations (meat_types : ℕ) (cheese_types : ℕ) : 
  meat_types = 12 → cheese_types = 8 → 
  (meat_types.choose 2) * cheese_types = 528 :=
by
  sorry

end sandwich_combinations_l1430_143049


namespace extremum_at_one_min_value_one_l1430_143095

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x + 1) + (1 - x) / (1 + x)

-- Theorem 1: If f attains an extremum at x=1, then a = 1
theorem extremum_at_one (a : ℝ) (h : a > 0) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a x ≤ f a 1 ∨ f a x ≥ f a 1) →
  a = 1 :=
sorry

-- Theorem 2: If the minimum value of f is 1, then a ≥ 2
theorem min_value_one (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, f a x ≥ 1) →
  a ≥ 2 :=
sorry

end extremum_at_one_min_value_one_l1430_143095


namespace trigonometric_identities_l1430_143001

theorem trigonometric_identities (α : Real) 
  (h : (Real.tan α) / (Real.tan α - 1) = -1) : 
  (Real.sin α - 3 * Real.cos α) / (Real.sin α + 2 * Real.cos α) = -1 ∧ 
  (Real.sin (π - α) * Real.cos (π + α) * Real.cos (π/2 + α) * Real.cos (π/2 - α)) / 
  (Real.cos (π - α) * Real.sin (3*π - α) * Real.sin (-π - α) * Real.sin (π/2 + α)) = -1/2 := by
  sorry

end trigonometric_identities_l1430_143001


namespace moores_law_1985_to_1995_l1430_143065

/-- Moore's law doubling period in years -/
def moore_period : ℕ := 2

/-- Initial year for transistor count -/
def initial_year : ℕ := 1985

/-- Final year for transistor count -/
def final_year : ℕ := 1995

/-- Initial transistor count in 1985 -/
def initial_transistors : ℕ := 500000

/-- Calculate the number of transistors according to Moore's law -/
def transistor_count (start_year end_year start_count : ℕ) : ℕ :=
  start_count * 2 ^ ((end_year - start_year) / moore_period)

/-- Theorem stating that the transistor count in 1995 is 16,000,000 -/
theorem moores_law_1985_to_1995 :
  transistor_count initial_year final_year initial_transistors = 16000000 := by
  sorry

end moores_law_1985_to_1995_l1430_143065


namespace vector_sum_magnitude_l1430_143007

def vector_a : Fin 2 → ℝ := ![1, 2]
def vector_b : Fin 2 → ℝ := ![-2, 3]

theorem vector_sum_magnitude :
  ‖vector_a + vector_b‖ = Real.sqrt 26 := by
  sorry

end vector_sum_magnitude_l1430_143007


namespace unique_triple_solution_l1430_143008

theorem unique_triple_solution : 
  ∃! (p q n : ℕ), 
    p.Prime ∧ q.Prime ∧ 
    p > 0 ∧ q > 0 ∧ n > 0 ∧
    p * (p + 3) + q * (q + 3) = n * (n + 3) ∧
    p = 3 ∧ q = 2 ∧ n = 4 := by
  sorry

end unique_triple_solution_l1430_143008


namespace base_number_theorem_l1430_143055

theorem base_number_theorem (x w : ℝ) (h1 : x^(2*w) = 8^(w-4)) (h2 : w = 12) : x = 2 := by
  sorry

end base_number_theorem_l1430_143055


namespace system_solution_l1430_143096

/-- Given a system of equations, prove that t = 24 -/
theorem system_solution (p t j x y a b c : ℝ) 
  (h1 : j = 0.75 * p)
  (h2 : j = 0.8 * t)
  (h3 : t = p - (t / 100) * p)
  (h4 : x = 0.1 * t)
  (h5 : y = 0.5 * j)
  (h6 : x + y = 12)
  (h7 : a = x + y)
  (h8 : b = 0.15 * a)
  (h9 : c = 2 * b) :
  t = 24 := by sorry

end system_solution_l1430_143096


namespace abc_def_ratio_l1430_143041

theorem abc_def_ratio (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 2) :
  a * b * c / (d * e * f) = 1 / 12 := by
  sorry

end abc_def_ratio_l1430_143041


namespace polynomial_divisibility_l1430_143006

theorem polynomial_divisibility (n : ℕ) (hn : n > 2) :
  (∃ q : Polynomial ℚ, x^n + x^2 + 1 = (x^2 + x + 1) * q) ↔ 
  (∃ k : ℕ, n = 3 * k + 1) :=
sorry

end polynomial_divisibility_l1430_143006


namespace max_visible_cube_l1430_143091

/-- The size of the cube's edge -/
def n : ℕ := 13

/-- The number of unit cubes visible on one face -/
def face_visible : ℕ := n^2

/-- The number of unit cubes visible along one edge (excluding the corner) -/
def edge_visible : ℕ := n - 1

/-- The maximum number of unit cubes visible from a single point -/
def max_visible : ℕ := 3 * face_visible - 3 * edge_visible + 1

theorem max_visible_cube :
  max_visible = 472 :=
sorry

end max_visible_cube_l1430_143091


namespace polynomial_factorization_l1430_143048

theorem polynomial_factorization (x : ℝ) :
  x^4 + 16 = (x^2 + 2*x + 2) * (x^2 - 2*x + 2) := by
  sorry

end polynomial_factorization_l1430_143048


namespace sophomore_sample_size_l1430_143009

/-- Represents the number of students selected in a stratified sample -/
def stratifiedSample (totalPopulation : ℕ) (sampleSize : ℕ) (strataSize : ℕ) : ℕ :=
  (strataSize * sampleSize) / totalPopulation

/-- The problem statement -/
theorem sophomore_sample_size :
  let totalStudents : ℕ := 2800
  let sophomores : ℕ := 930
  let sampleSize : ℕ := 280
  stratifiedSample totalStudents sampleSize sophomores = 93 := by
  sorry

end sophomore_sample_size_l1430_143009


namespace box_cube_volume_l1430_143057

/-- Given a box with dimensions 12 cm × 16 cm × 6 cm built using 384 identical cubic cm cubes,
    prove that the volume of each cube is 3 cm³. -/
theorem box_cube_volume (length width height : ℝ) (num_cubes : ℕ) (cube_volume : ℝ) :
  length = 12 →
  width = 16 →
  height = 6 →
  num_cubes = 384 →
  length * width * height = num_cubes * cube_volume →
  cube_volume = 3 := by
  sorry

end box_cube_volume_l1430_143057


namespace elizabeth_stickers_l1430_143066

/-- The number of stickers Elizabeth uses on her water bottles -/
def total_stickers (initial_bottles : ℕ) (lost_bottles : ℕ) (stolen_bottles : ℕ) (stickers_per_bottle : ℕ) : ℕ :=
  (initial_bottles - lost_bottles - stolen_bottles) * stickers_per_bottle

/-- Theorem: Elizabeth uses 21 stickers in total -/
theorem elizabeth_stickers :
  total_stickers 10 2 1 3 = 21 := by
  sorry

end elizabeth_stickers_l1430_143066


namespace rhombus_perimeter_l1430_143089

/-- The perimeter of a rhombus with diagonals of 8 inches and 30 inches is 4√241 inches. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 8) (h2 : d2 = 30) : 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 4 * Real.sqrt 241 :=
by sorry

end rhombus_perimeter_l1430_143089


namespace abs_cubic_inequality_l1430_143063

theorem abs_cubic_inequality (x : ℝ) : 
  |x| ≤ 2 → |3*x - x^3| ≤ 2 := by sorry

end abs_cubic_inequality_l1430_143063


namespace quadratic_inequality_minimum_l1430_143060

theorem quadratic_inequality_minimum (a b : ℝ) (h1 : a > b) 
  (h2 : ∀ x, (a*x^2 + 2*x + b > 0) ↔ (x ≠ -1/a)) :
  ∃ m : ℝ, m = 6 ∧ ∀ x, x = (a^2 + b^2 + 7)/(a - b) → x ≥ m := by
sorry

end quadratic_inequality_minimum_l1430_143060


namespace val_coin_ratio_l1430_143002

theorem val_coin_ratio :
  -- Define the number of nickels Val has initially
  let initial_nickels : ℕ := 20
  -- Define the value of a nickel in cents
  let nickel_value : ℕ := 5
  -- Define the value of a dime in cents
  let dime_value : ℕ := 10
  -- Define the total value in cents after finding additional nickels
  let total_value_after : ℕ := 900
  -- Define the function to calculate the number of additional nickels
  let additional_nickels (n : ℕ) : ℕ := 2 * n
  -- Define the function to calculate the total number of nickels after finding additional ones
  let total_nickels (n : ℕ) : ℕ := n + additional_nickels n
  -- Define the function to calculate the value of nickels in cents
  let nickel_value_cents (n : ℕ) : ℕ := n * nickel_value
  -- Define the function to calculate the value of dimes in cents
  let dime_value_cents (d : ℕ) : ℕ := d * dime_value
  -- Define the function to calculate the number of dimes
  let num_dimes (n : ℕ) : ℕ := (total_value_after - nickel_value_cents (total_nickels n)) / dime_value
  -- The ratio of dimes to nickels is 3:1
  num_dimes initial_nickels / initial_nickels = 3 := by
  sorry

end val_coin_ratio_l1430_143002


namespace arctan_tan_difference_l1430_143094

open Real

theorem arctan_tan_difference (θ₁ θ₂ : ℝ) (h₁ : θ₁ = 70 * π / 180) (h₂ : θ₂ = 20 * π / 180) :
  arctan (tan θ₁ - 3 * tan θ₂) = 50 * π / 180 := by
  sorry

end arctan_tan_difference_l1430_143094


namespace no_integer_cubes_l1430_143087

theorem no_integer_cubes (a b : ℤ) : 
  a ≥ 1 → b ≥ 1 → 
  (∃ x : ℤ, a^5 * b + 3 = x^3) → 
  (∃ y : ℤ, a * b^5 + 3 = y^3) → 
  False :=
sorry

end no_integer_cubes_l1430_143087


namespace rectangle_area_l1430_143015

/-- Proves that a rectangular field with sides in ratio 3:4 and perimeter costing 8750 paise
    at 25 paise per metre has an area of 7500 square meters. -/
theorem rectangle_area (length width : ℝ) (h1 : length / width = 3 / 4)
    (h2 : 2 * (length + width) * 25 = 8750) : length * width = 7500 :=
by sorry

end rectangle_area_l1430_143015


namespace sandra_beignet_consumption_l1430_143090

/-- The number of beignets Sandra eats per day -/
def daily_beignets : ℕ := 3

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of weeks -/
def num_weeks : ℕ := 16

/-- The total number of beignets Sandra eats in the given period -/
def total_beignets : ℕ := daily_beignets * days_per_week * num_weeks

theorem sandra_beignet_consumption :
  total_beignets = 336 := by sorry

end sandra_beignet_consumption_l1430_143090


namespace employment_percentage_l1430_143075

theorem employment_percentage (population : ℝ) (employed : ℝ) 
  (h1 : employed > 0) 
  (h2 : population > 0) 
  (h3 : 0.42 * population = 0.7 * employed) : 
  employed / population = 0.6 := by
sorry

end employment_percentage_l1430_143075


namespace inequality_proof_l1430_143039

theorem inequality_proof (x : ℝ) (hx : x > 0) :
  Real.sqrt (x^2 - x + 1/2) ≥ 1 / (x + 1/x) := by
  sorry

end inequality_proof_l1430_143039


namespace quadratic_inequality_range_l1430_143059

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + (a - 1)*x + 1 < 0) ↔ (a < -1 ∨ a > 3) :=
by sorry

end quadratic_inequality_range_l1430_143059


namespace tangent_slope_angle_range_l1430_143072

theorem tangent_slope_angle_range :
  ∀ (x : ℝ),
  let y := x^3 - x + 2/3
  let slope := (3 * x^2 - 1 : ℝ)
  let α := Real.arctan slope
  α ∈ Set.union (Set.Ico 0 (π/2)) (Set.Icc (3*π/4) π) := by
sorry

end tangent_slope_angle_range_l1430_143072


namespace permit_increase_l1430_143012

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of digits (0-9) -/
def digit_count : ℕ := 10

/-- The number of letters in old permits -/
def old_permit_letters : ℕ := 2

/-- The number of digits in old permits -/
def old_permit_digits : ℕ := 3

/-- The number of letters in new permits -/
def new_permit_letters : ℕ := 4

/-- The number of digits in new permits -/
def new_permit_digits : ℕ := 4

/-- The ratio of new permits to old permits -/
def permit_ratio : ℕ := 67600

theorem permit_increase :
  (alphabet_size ^ new_permit_letters * digit_count ^ new_permit_digits) /
  (alphabet_size ^ old_permit_letters * digit_count ^ old_permit_digits) = permit_ratio :=
sorry

end permit_increase_l1430_143012


namespace fibonacci_sum_quadruples_l1430_143079

/-- The n-th Fibonacci number -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- A predicate that checks if a quadruple (a, b, c, d) satisfies the Fibonacci sum equation -/
def is_valid_quadruple (a b c d : ℕ) : Prop :=
  a ≥ 2 ∧ b ≥ 2 ∧ c ≥ 2 ∧ d ≥ 2 ∧ fib a + fib b = fib c + fib d

/-- The set of all valid quadruples -/
def valid_quadruples : Set (ℕ × ℕ × ℕ × ℕ) :=
  {q | ∃ a b c d, q = (a, b, c, d) ∧ is_valid_quadruple a b c d}

/-- The set of solution quadruples -/
def solution_quadruples : Set (ℕ × ℕ × ℕ × ℕ) :=
  {q | ∃ a b,
    (q = (a, b, a, b) ∨ q = (a, b, b, a) ∨
     q = (a, a-3, a-1, a-1) ∨ q = (a-3, a, a-1, a-1) ∨
     q = (a-1, a-1, a, a-3) ∨ q = (a-1, a-1, a-3, a)) ∧
    a ≥ 2 ∧ b ≥ 2}

/-- The main theorem stating that the valid quadruples are exactly the solution quadruples -/
theorem fibonacci_sum_quadruples : valid_quadruples = solution_quadruples := by
  sorry

end fibonacci_sum_quadruples_l1430_143079


namespace win_sector_area_l1430_143014

theorem win_sector_area (r : ℝ) (p : ℝ) (h1 : r = 12) (h2 : p = 1/3) :
  p * π * r^2 = 48 * π := by
  sorry

end win_sector_area_l1430_143014


namespace base_conversion_512_l1430_143067

/-- Converts a base-10 number to its base-6 representation -/
def toBase6 (n : ℕ) : List ℕ :=
  sorry

theorem base_conversion_512 :
  toBase6 512 = [2, 2, 1, 2] :=
sorry

end base_conversion_512_l1430_143067


namespace arithmetic_sequence_sum_l1430_143083

/-- A sequence of real numbers. -/
def Sequence := ℕ → ℝ

/-- Predicate for an arithmetic sequence. -/
def IsArithmetic (a : Sequence) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : Sequence) 
  (h1 : IsArithmetic a) 
  (h2 : a 1 + a 6 + a 11 = 3) : 
  a 3 + a 9 = 2 := by
sorry

end arithmetic_sequence_sum_l1430_143083


namespace simplify_fraction_ratio_l1430_143052

theorem simplify_fraction_ratio (k : ℤ) : 
  ∃ (a b : ℤ), (4*k + 8) / 4 = a*k + b ∧ a / b = 1 / 2 := by
  sorry

end simplify_fraction_ratio_l1430_143052


namespace adult_tickets_sold_l1430_143038

/-- Proves the number of adult tickets sold given ticket prices and total sales information -/
theorem adult_tickets_sold
  (adult_price : ℕ)
  (child_price : ℕ)
  (total_revenue : ℕ)
  (total_tickets : ℕ)
  (h1 : adult_price = 8)
  (h2 : child_price = 5)
  (h3 : total_revenue = 236)
  (h4 : total_tickets = 34)
  : ∃ (adult_tickets : ℕ) (child_tickets : ℕ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_tickets * adult_price + child_tickets * child_price = total_revenue ∧
    adult_tickets = 22 := by
  sorry


end adult_tickets_sold_l1430_143038


namespace quadratic_equation_roots_l1430_143032

theorem quadratic_equation_roots (p : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + p * x - 6 = 0 ∧ x = -2) → 
  (∃ y : ℝ, 3 * y^2 + p * y - 6 = 0 ∧ y = 1) :=
by sorry

end quadratic_equation_roots_l1430_143032


namespace count_four_digit_numbers_ending_25_is_90_l1430_143005

/-- A function that returns the count of four-digit numbers divisible by 5 with 25 as their last two digits -/
def count_four_digit_numbers_ending_25 : ℕ :=
  let first_number := 1025
  let last_number := 9925
  (last_number - first_number) / 100 + 1

/-- Theorem stating that the count of four-digit numbers divisible by 5 with 25 as their last two digits is 90 -/
theorem count_four_digit_numbers_ending_25_is_90 :
  count_four_digit_numbers_ending_25 = 90 := by
  sorry

end count_four_digit_numbers_ending_25_is_90_l1430_143005


namespace repair_cost_is_2400_l1430_143034

/-- The total cost of car repairs given labor rate, labor hours, and part cost. -/
def total_repair_cost (labor_rate : ℕ) (labor_hours : ℕ) (part_cost : ℕ) : ℕ :=
  labor_rate * labor_hours + part_cost

/-- Theorem stating that the total repair cost is $2400 given the specified conditions. -/
theorem repair_cost_is_2400 :
  total_repair_cost 75 16 1200 = 2400 := by
  sorry

end repair_cost_is_2400_l1430_143034


namespace last_digit_3_count_l1430_143040

/-- The last digit of 7^n -/
def last_digit (n : ℕ) : ℕ := (7^n) % 10

/-- Whether the last digit of 7^n is 3 -/
def is_last_digit_3 (n : ℕ) : Prop := last_digit n = 3

/-- The number of terms in the sequence 7^1, 7^2, ..., 7^n whose last digit is 3 -/
def count_last_digit_3 (n : ℕ) : ℕ := (n + 3) / 4

theorem last_digit_3_count :
  count_last_digit_3 2009 = 502 :=
sorry

end last_digit_3_count_l1430_143040


namespace sin_has_property_T_l1430_143042

-- Define property T
def has_property_T (f : ℝ → ℝ) : Prop :=
  ∃ x1 x2 : ℝ, (deriv f x1) * (deriv f x2) = -1

-- State the theorem
theorem sin_has_property_T : has_property_T Real.sin := by
  sorry


end sin_has_property_T_l1430_143042


namespace not_perfect_square_p_squared_plus_q_power_l1430_143037

theorem not_perfect_square_p_squared_plus_q_power (p q : ℕ) (hp : Prime p) (hq : Prime q) (h_perfect_square : ∃ a : ℕ, p + q^2 = a^2) :
  ∀ n : ℕ, ¬∃ b : ℕ, p^2 + q^n = b^2 :=
by sorry

end not_perfect_square_p_squared_plus_q_power_l1430_143037


namespace bus_passengers_l1430_143070

theorem bus_passengers (men women : ℕ) : 
  women = men / 2 → 
  men - 16 = women + 8 → 
  men + women = 72 :=
by sorry

end bus_passengers_l1430_143070


namespace cos_A_from_tan_A_l1430_143054

theorem cos_A_from_tan_A (A : Real) (h : Real.tan A = 2/3) : 
  Real.cos A = 3 * Real.sqrt 13 / 13 := by
  sorry

end cos_A_from_tan_A_l1430_143054


namespace black_tiles_to_total_l1430_143056

/-- Represents a square hall tiled with square tiles -/
structure SquareHall where
  side : ℕ

/-- Calculates the number of black tiles in the hall -/
def black_tiles (hall : SquareHall) : ℕ :=
  2 * hall.side

/-- Calculates the total number of tiles in the hall -/
def total_tiles (hall : SquareHall) : ℕ :=
  hall.side * hall.side

/-- Theorem stating the relationship between black tiles and total tiles -/
theorem black_tiles_to_total (hall : SquareHall) :
  black_tiles hall - 3 = 153 → total_tiles hall = 6084 := by
  sorry

end black_tiles_to_total_l1430_143056


namespace cookies_per_bag_l1430_143023

/-- Proves that the number of cookies in each bag is 20, given the conditions of the problem. -/
theorem cookies_per_bag (bags_per_box : ℕ) (total_calories : ℕ) (calories_per_cookie : ℕ)
  (h1 : bags_per_box = 4)
  (h2 : total_calories = 1600)
  (h3 : calories_per_cookie = 20) :
  total_calories / (bags_per_box * calories_per_cookie) = 20 :=
by sorry

end cookies_per_bag_l1430_143023


namespace inequality1_solution_inequality2_solution_l1430_143033

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := x^2 - 5*x - 6 < 0
def inequality2 (x : ℝ) : Prop := (x - 1) / (x + 2) ≤ 0

-- Define the solution sets
def solution_set1 : Set ℝ := {x | -1 < x ∧ x < 6}
def solution_set2 : Set ℝ := {x | -2 < x ∧ x ≤ 1}

-- Theorem statements
theorem inequality1_solution : 
  ∀ x : ℝ, inequality1 x ↔ x ∈ solution_set1 :=
sorry

theorem inequality2_solution : 
  ∀ x : ℝ, x ≠ -2 → (inequality2 x ↔ x ∈ solution_set2) :=
sorry

end inequality1_solution_inequality2_solution_l1430_143033


namespace path_result_l1430_143022

def move_north (x : ℚ) : ℚ := x + 7
def move_east (x : ℚ) : ℚ := x - 4
def move_south (x : ℚ) : ℚ := x / 2
def move_west (x : ℚ) : ℚ := x * 3

def path (x : ℚ) : ℚ :=
  move_north (move_east (move_south (move_west (move_west (move_south (move_east (move_north x)))))))

theorem path_result : path 21 = 57 := by
  sorry

end path_result_l1430_143022


namespace product_ab_equals_negative_one_l1430_143076

theorem product_ab_equals_negative_one (a b : ℝ) 
  (h : ∀ x : ℝ, x ≥ 0 → 0 ≤ x^4 - x^3 + a*x + b ∧ x^4 - x^3 + a*x + b ≤ (x^2 - 1)^2) : 
  a * b = -1 := by
sorry

end product_ab_equals_negative_one_l1430_143076


namespace final_book_count_l1430_143013

/-- Represents the number of books in the library system -/
structure LibraryState where
  books : ℕ

/-- Represents a transaction that changes the number of books -/
inductive Transaction
  | TakeOut (n : ℕ)
  | Return (n : ℕ)
  | Withdraw (n : ℕ)

/-- Applies a transaction to the library state -/
def applyTransaction (state : LibraryState) (t : Transaction) : LibraryState :=
  match t with
  | Transaction.TakeOut n => ⟨state.books - n⟩
  | Transaction.Return n => ⟨state.books + n⟩
  | Transaction.Withdraw n => ⟨state.books - n⟩

/-- Applies a list of transactions to the library state -/
def applyTransactions (state : LibraryState) (ts : List Transaction) : LibraryState :=
  ts.foldl applyTransaction state

/-- The initial state of the library -/
def initialState : LibraryState := ⟨250⟩

/-- The transactions that occur over the three weeks -/
def transactions : List Transaction := [
  Transaction.TakeOut 120,  -- Week 1 Tuesday
  Transaction.Return 35,    -- Week 1 Wednesday
  Transaction.Withdraw 15,  -- Week 1 Thursday
  Transaction.TakeOut 42,   -- Week 1 Friday
  Transaction.Return 72,    -- Week 2 Monday (60% of 120)
  Transaction.Return 34,    -- Week 2 Tuesday (80% of 42, rounded)
  Transaction.Withdraw 75,  -- Week 2 Wednesday
  Transaction.TakeOut 40,   -- Week 2 Thursday
  Transaction.Return 20,    -- Week 3 Monday (50% of 40)
  Transaction.TakeOut 20,   -- Week 3 Tuesday
  Transaction.Return 46,    -- Week 3 Wednesday (95% of 48, rounded)
  Transaction.Withdraw 10,  -- Week 3 Thursday
  Transaction.TakeOut 55    -- Week 3 Friday
]

/-- The theorem stating that after applying all transactions, the library has 80 books -/
theorem final_book_count :
  (applyTransactions initialState transactions).books = 80 := by
  sorry

end final_book_count_l1430_143013


namespace evaluate_expression_l1430_143085

theorem evaluate_expression : 5^2 + 2*(5 - 2) = 31 := by
  sorry

end evaluate_expression_l1430_143085


namespace diophantine_equation_solution_l1430_143097

theorem diophantine_equation_solution (k : ℕ+) : 
  (∃ (x y : ℕ+), x^2 + y^2 = k * x * y - 1) ↔ k = 3 := by
sorry

end diophantine_equation_solution_l1430_143097


namespace moon_speed_in_km_per_hour_l1430_143043

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- The moon's speed in kilometers per second -/
def moon_speed_km_per_sec : ℚ := 9/10

/-- Converts a speed from kilometers per second to kilometers per hour -/
def km_per_sec_to_km_per_hour (speed : ℚ) : ℚ :=
  speed * seconds_per_hour

theorem moon_speed_in_km_per_hour :
  km_per_sec_to_km_per_hour moon_speed_km_per_sec = 3240 := by
  sorry

end moon_speed_in_km_per_hour_l1430_143043


namespace blossom_room_area_l1430_143092

/-- Represents the length of a side of a square room in feet -/
def room_side_length : ℕ := 10

/-- Represents the number of inches in a foot -/
def inches_per_foot : ℕ := 12

/-- Calculates the area of a square room in square inches -/
def room_area_sq_inches (side_length : ℕ) (inches_per_foot : ℕ) : ℕ :=
  (side_length * inches_per_foot) ^ 2

/-- Theorem stating that the area of Blossom's room is 14400 square inches -/
theorem blossom_room_area :
  room_area_sq_inches room_side_length inches_per_foot = 14400 := by
  sorry

end blossom_room_area_l1430_143092


namespace intersection_implies_sum_l1430_143074

theorem intersection_implies_sum (a b : ℝ) : 
  let A : Set ℝ := {3, 2^a}
  let B : Set ℝ := {a, b}
  A ∩ B = {2} → a + b = 3 := by
sorry

end intersection_implies_sum_l1430_143074


namespace expression_decrease_value_decrease_l1430_143011

theorem expression_decrease (x y : ℝ) (h : x > 0 ∧ y > 0) : 
  (3/4 * x) * (3/4 * y)^2 = (27/64) * (x * y^2) := by
  sorry

theorem value_decrease (x y : ℝ) (h : x > 0 ∧ y > 0) : 
  1 - (3/4 * x) * (3/4 * y)^2 / (x * y^2) = 37/64 := by
  sorry

end expression_decrease_value_decrease_l1430_143011


namespace quadratic_equation_roots_l1430_143027

theorem quadratic_equation_roots : 
  let f : ℝ → ℝ := λ x => x^2 - 1 - 3
  ∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = -2 ∧ f x₁ = 0 ∧ f x₂ = 0 := by
  sorry

end quadratic_equation_roots_l1430_143027


namespace four_number_equation_solutions_l1430_143046

def is_solution (x₁ x₂ x₃ x₄ : ℝ) : Prop :=
  x₁ + x₂*x₃*x₄ = 2 ∧
  x₂ + x₁*x₃*x₄ = 2 ∧
  x₃ + x₁*x₂*x₄ = 2 ∧
  x₄ + x₁*x₂*x₃ = 2

theorem four_number_equation_solutions :
  ∀ x₁ x₂ x₃ x₄ : ℝ, is_solution x₁ x₂ x₃ x₄ ↔
    ((x₁, x₂, x₃, x₄) = (1, 1, 1, 1) ∨
     (x₁, x₂, x₃, x₄) = (-1, -1, -1, 3) ∨
     (x₁, x₂, x₃, x₄) = (-1, -1, 3, -1) ∨
     (x₁, x₂, x₃, x₄) = (-1, 3, -1, -1) ∨
     (x₁, x₂, x₃, x₄) = (3, -1, -1, -1)) :=
by sorry


end four_number_equation_solutions_l1430_143046


namespace magnitude_of_complex_fraction_l1430_143099

/-- The magnitude of the complex number (2+4i)/(1+i) is √10 -/
theorem magnitude_of_complex_fraction :
  let z : ℂ := (2 + 4 * Complex.I) / (1 + Complex.I)
  Complex.abs z = Real.sqrt 10 := by
  sorry

end magnitude_of_complex_fraction_l1430_143099


namespace smallest_number_l1430_143025

/-- Converts a number from base b to base 10 --/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- Represents the number 85 in base 9 --/
def num1 : List Nat := [8, 5]

/-- Represents the number 1000 in base 4 --/
def num2 : List Nat := [1, 0, 0, 0]

/-- Represents the number 111111 in base 2 --/
def num3 : List Nat := [1, 1, 1, 1, 1, 1]

theorem smallest_number :
  to_base_10 num3 2 ≤ to_base_10 num1 9 ∧
  to_base_10 num3 2 ≤ to_base_10 num2 4 := by
  sorry

end smallest_number_l1430_143025


namespace sum_of_roots_of_equation_l1430_143010

theorem sum_of_roots_of_equation (x : ℝ) : 
  (∃ r₁ r₂ : ℝ, (r₁ - 7)^2 = 16 ∧ (r₂ - 7)^2 = 16 ∧ r₁ + r₂ = 14) := by
  sorry

end sum_of_roots_of_equation_l1430_143010


namespace min_length_for_prob_threshold_l1430_143028

/-- The probability that a random sequence of length n using digits 0, 1, and 2 does not contain all three digits -/
def prob_not_all_digits (n : ℕ) : ℚ :=
  (2^n - 1) / 3^(n-1)

/-- The probability that a random sequence of length n using digits 0, 1, and 2 contains all three digits -/
def prob_all_digits (n : ℕ) : ℚ :=
  1 - prob_not_all_digits n

theorem min_length_for_prob_threshold :
  prob_all_digits 5 ≥ 61/100 ∧
  ∀ k < 5, prob_all_digits k < 61/100 :=
sorry

end min_length_for_prob_threshold_l1430_143028


namespace polynomial_divisibility_l1430_143035

theorem polynomial_divisibility : ∀ x : ℂ,
  (x^100 + x^75 + x^50 + x^25 + 1) % (x^9 + x^6 + x^3 + 1) = 0 := by
  sorry

end polynomial_divisibility_l1430_143035


namespace correct_sum_after_card_swap_l1430_143064

theorem correct_sum_after_card_swap : 
  ∃ (a b : ℕ), 
    (a + b = 81380) ∧ 
    (a ≠ 37541 ∨ b ≠ 43839) ∧
    (∃ (x y : ℕ), (x = 37541 ∧ y = 43839) ∧ (x + y = 80280)) :=
by sorry

end correct_sum_after_card_swap_l1430_143064


namespace locus_empty_near_origin_l1430_143026

/-- Represents a polynomial of degree 3 in two variables -/
structure Polynomial3 (α : Type*) [Ring α] where
  A : α
  B : α
  C : α
  D : α
  E : α
  F : α
  G : α

/-- Evaluates the polynomial at a given point (x, y) -/
def eval_poly (p : Polynomial3 ℝ) (x y : ℝ) : ℝ :=
  p.A * x^2 + p.B * x * y + p.C * y^2 + p.D * x^3 + p.E * x^2 * y + p.F * x * y^2 + p.G * y^3

theorem locus_empty_near_origin (p : Polynomial3 ℝ) (h : p.B^2 - 4 * p.A * p.C < 0) :
  ∃ δ : ℝ, δ > 0 ∧ ∀ x y : ℝ, 0 < x^2 + y^2 ∧ x^2 + y^2 < δ^2 → eval_poly p x y ≠ 0 := by
  sorry

end locus_empty_near_origin_l1430_143026


namespace product_of_large_integers_l1430_143073

theorem product_of_large_integers : ∃ (a b : ℤ), 
  a > 10^2009 ∧ b > 10^2009 ∧ a * b = 3^(4^5) + 4^(5^6) := by
  sorry

end product_of_large_integers_l1430_143073


namespace greatest_base_nine_digit_sum_l1430_143000

/-- The greatest possible sum of digits in base-nine representation of a positive integer less than 3000 -/
def max_base_nine_digit_sum : ℕ := 24

/-- Converts a natural number to its base-nine representation -/
def to_base_nine (n : ℕ) : List ℕ := sorry

/-- Calculates the sum of digits in a list -/
def digit_sum (digits : List ℕ) : ℕ := sorry

/-- Checks if a number is less than 3000 -/
def less_than_3000 (n : ℕ) : Prop := n < 3000

theorem greatest_base_nine_digit_sum :
  ∀ n : ℕ, less_than_3000 n → digit_sum (to_base_nine n) ≤ max_base_nine_digit_sum :=
by sorry

end greatest_base_nine_digit_sum_l1430_143000
