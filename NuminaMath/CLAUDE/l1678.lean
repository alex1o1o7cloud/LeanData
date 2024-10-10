import Mathlib

namespace mia_weight_l1678_167825

/-- 
Given two people, Anna and Mia, with the following conditions:
1. The sum of their weights is 220 pounds.
2. The difference between Mia's weight and Anna's weight is twice Anna's weight.
This theorem proves that Mia's weight is 165 pounds.
-/
theorem mia_weight (anna_weight mia_weight : ℝ) 
  (sum_condition : anna_weight + mia_weight = 220)
  (difference_condition : mia_weight - anna_weight = 2 * anna_weight) :
  mia_weight = 165 := by
sorry

end mia_weight_l1678_167825


namespace line_passes_through_point_line_has_equal_intercepts_line_equation_is_correct_l1678_167858

/-- A line passing through point P(1,3) with equal x and y intercepts -/
def line_with_equal_intercepts : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 = 4}

theorem line_passes_through_point :
  (1, 3) ∈ line_with_equal_intercepts := by sorry

theorem line_has_equal_intercepts :
  ∃ a : ℝ, a ≠ 0 ∧ (a, 0) ∈ line_with_equal_intercepts ∧ (0, a) ∈ line_with_equal_intercepts := by sorry

theorem line_equation_is_correct :
  line_with_equal_intercepts = {p : ℝ × ℝ | p.1 + p.2 = 4} := by sorry

end line_passes_through_point_line_has_equal_intercepts_line_equation_is_correct_l1678_167858


namespace E_parity_2021_2022_2023_l1678_167837

def E : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | n + 3 => E (n + 2) + E (n + 1) + E n

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem E_parity_2021_2022_2023 :
  is_even (E 2021) ∧ ¬is_even (E 2022) ∧ ¬is_even (E 2023) := by
  sorry

end E_parity_2021_2022_2023_l1678_167837


namespace count_distinct_prime_factors_30_factorial_l1678_167836

/-- The number of distinct prime factors of 30! -/
def distinct_prime_factors_30_factorial : ℕ := sorry

/-- Theorem stating that the number of distinct prime factors of 30! is 10 -/
theorem count_distinct_prime_factors_30_factorial :
  distinct_prime_factors_30_factorial = 10 := by sorry

end count_distinct_prime_factors_30_factorial_l1678_167836


namespace focus_coordinates_l1678_167854

/-- A parabola is defined by the equation x^2 = 4y -/
structure Parabola where
  equation : ∀ x y : ℝ, x^2 = 4*y

/-- The focus of a parabola is a point (h, k) on its axis of symmetry -/
structure Focus (p : Parabola) where
  h : ℝ
  k : ℝ

/-- Theorem: The focus of the parabola x^2 = 4y has coordinates (0, 1) -/
theorem focus_coordinates (p : Parabola) : 
  ∃ f : Focus p, f.h = 0 ∧ f.k = 1 := by
  sorry

end focus_coordinates_l1678_167854


namespace tin_in_new_alloy_tin_amount_is_correct_l1678_167824

/-- The amount of tin in a new alloy formed by mixing two alloys -/
theorem tin_in_new_alloy (alloy_a_mass : ℝ) (alloy_b_mass : ℝ) 
  (lead_tin_ratio_a : ℝ × ℝ) (tin_copper_ratio_b : ℝ × ℝ) : ℝ :=
  let tin_in_a := (lead_tin_ratio_a.2 / (lead_tin_ratio_a.1 + lead_tin_ratio_a.2)) * alloy_a_mass
  let tin_in_b := (tin_copper_ratio_b.1 / (tin_copper_ratio_b.1 + tin_copper_ratio_b.2)) * alloy_b_mass
  tin_in_a + tin_in_b

/-- The amount of tin in the new alloy is 139.5 kg -/
theorem tin_amount_is_correct : 
  tin_in_new_alloy 120 180 (2, 3) (3, 5) = 139.5 := by
  sorry

end tin_in_new_alloy_tin_amount_is_correct_l1678_167824


namespace fraction_addition_l1678_167888

theorem fraction_addition : (168 : ℚ) / 240 + 100 / 150 = 41 / 30 := by
  sorry

end fraction_addition_l1678_167888


namespace zeros_before_first_nonzero_of_fraction_l1678_167861

/-- The number of zeros between the decimal point and the first non-zero digit when 7/8000 is written as a decimal -/
def zeros_before_first_nonzero : ℕ :=
  3

/-- The fraction we're considering -/
def fraction : ℚ :=
  7 / 8000

theorem zeros_before_first_nonzero_of_fraction :
  zeros_before_first_nonzero = 3 ∧ fraction = 7 / 8000 := by
  sorry

end zeros_before_first_nonzero_of_fraction_l1678_167861


namespace consecutive_odd_integers_sum_l1678_167881

theorem consecutive_odd_integers_sum (x : ℤ) : 
  (x % 2 = 1) →  -- x is odd
  (x + (x + 2) + (x + 4) ≥ 51) →  -- sum is at least 51
  (x ≥ 15) ∧  -- x is at least 15
  (∀ y : ℤ, (y % 2 = 1) ∧ (y + (y + 2) + (y + 4) ≥ 51) → y ≥ x) -- x is the smallest such integer
  := by sorry

end consecutive_odd_integers_sum_l1678_167881


namespace thirteen_in_binary_l1678_167899

theorem thirteen_in_binary : 
  (13 : ℕ).digits 2 = [1, 0, 1, 1] :=
sorry

end thirteen_in_binary_l1678_167899


namespace frozen_yoghurt_cost_l1678_167807

theorem frozen_yoghurt_cost (ice_cream_quantity : ℕ) (frozen_yoghurt_quantity : ℕ) 
  (ice_cream_cost : ℕ) (ice_cream_total : ℕ) (price_difference : ℕ) :
  ice_cream_quantity = 10 →
  frozen_yoghurt_quantity = 4 →
  ice_cream_cost = 4 →
  ice_cream_total = ice_cream_quantity * ice_cream_cost →
  ice_cream_total = price_difference + (frozen_yoghurt_quantity * 1) →
  1 = (ice_cream_total - price_difference) / frozen_yoghurt_quantity :=
by
  sorry

end frozen_yoghurt_cost_l1678_167807


namespace orphanage_donation_l1678_167850

theorem orphanage_donation (total donation1 donation3 : ℚ) 
  (h1 : total = 650)
  (h2 : donation1 = 175)
  (h3 : donation3 = 250) :
  total - donation1 - donation3 = 225 := by
  sorry

end orphanage_donation_l1678_167850


namespace unique_solution_for_equation_l1678_167829

theorem unique_solution_for_equation (m p q : ℕ) : 
  m > 0 ∧ 
  Nat.Prime p ∧ 
  Nat.Prime q ∧ 
  2^m * p^2 + 1 = q^5 → 
  m = 1 ∧ p = 11 ∧ q = 3 :=
by sorry

end unique_solution_for_equation_l1678_167829


namespace log_equation_solution_l1678_167815

theorem log_equation_solution (x : ℝ) (hx : x > 0) :
  Real.log x / Real.log 3 + Real.log 3 / Real.log x - 2 * (Real.log x / Real.log 3) * (Real.log 3 / Real.log x) = 1/2 ↔ 
  x = Real.sqrt 3 ∨ x = 9 :=
by sorry

end log_equation_solution_l1678_167815


namespace max_sum_squared_distances_l1678_167880

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

def is_unit_vector (v : E) : Prop := ‖v‖ = 1

theorem max_sum_squared_distances (a b c : E) 
  (ha : is_unit_vector a) (hb : is_unit_vector b) (hc : is_unit_vector c) :
  ‖a - b‖^2 + ‖a - c‖^2 + ‖b - c‖^2 ≤ 9 ∧ 
  ∃ (a' b' c' : E), is_unit_vector a' ∧ is_unit_vector b' ∧ is_unit_vector c' ∧
    ‖a' - b'‖^2 + ‖a' - c'‖^2 + ‖b' - c'‖^2 = 9 :=
by sorry

end max_sum_squared_distances_l1678_167880


namespace vector_operation_l1678_167889

def vector_a : ℝ × ℝ × ℝ := (2, 0, -1)
def vector_b : ℝ × ℝ × ℝ := (0, 1, -2)

theorem vector_operation :
  (2 : ℝ) • vector_a - vector_b = (4, -1, 0) := by sorry

end vector_operation_l1678_167889


namespace balloon_distribution_l1678_167885

theorem balloon_distribution (total_balloons : ℕ) (num_friends : ℕ) 
  (h1 : total_balloons = 215) (h2 : num_friends = 9) :
  total_balloons % num_friends = 8 := by
sorry

end balloon_distribution_l1678_167885


namespace video_game_lives_l1678_167817

theorem video_game_lives (initial_lives lives_lost lives_gained : ℕ) :
  initial_lives - lives_lost + lives_gained = initial_lives + lives_gained - lives_lost :=
by sorry

#check video_game_lives 43 14 27

end video_game_lives_l1678_167817


namespace find_other_number_l1678_167805

theorem find_other_number (n m : ℕ+) 
  (h_lcm : Nat.lcm n m = 52)
  (h_gcd : Nat.gcd n m = 8)
  (h_n : n = 26) : 
  m = 16 := by
  sorry

end find_other_number_l1678_167805


namespace no_solutions_for_inequality_l1678_167891

theorem no_solutions_for_inequality : 
  ¬ ∃ (n : ℕ), n ≥ 1 ∧ n ≤ n! - 4^n ∧ n! - 4^n ≤ 4*n :=
by sorry

end no_solutions_for_inequality_l1678_167891


namespace bug_crawl_distance_l1678_167826

-- Define the bug's movement
def bugPath : List ℤ := [-3, -7, 0, 8]

-- Function to calculate distance between two points
def distance (a b : ℤ) : ℕ := (a - b).natAbs

-- Function to calculate total distance traveled
def totalDistance (path : List ℤ) : ℕ :=
  List.sum (List.zipWith distance path path.tail)

-- Theorem statement
theorem bug_crawl_distance :
  totalDistance bugPath = 19 := by sorry

end bug_crawl_distance_l1678_167826


namespace weight_of_b_l1678_167887

theorem weight_of_b (a b c : ℝ) : 
  (a + b + c) / 3 = 43 →
  (a + b) / 2 = 40 →
  (b + c) / 2 = 43 →
  b = 37 := by
sorry

end weight_of_b_l1678_167887


namespace min_words_to_learn_l1678_167882

/-- Represents the French vocabulary exam setup -/
structure FrenchExam where
  totalWords : ℕ
  guessSuccessRate : ℚ
  targetScore : ℚ

/-- Calculates the exam score based on the number of words learned -/
def examScore (exam : FrenchExam) (wordsLearned : ℕ) : ℚ :=
  let correctGuesses := exam.guessSuccessRate * (exam.totalWords - wordsLearned)
  (wordsLearned + correctGuesses) / exam.totalWords

/-- Theorem stating the minimum number of words to learn for the given exam conditions -/
theorem min_words_to_learn (exam : FrenchExam) 
    (h1 : exam.totalWords = 800)
    (h2 : exam.guessSuccessRate = 1/20)
    (h3 : exam.targetScore = 9/10) : 
    ∀ n : ℕ, (∀ m : ℕ, m < n → examScore exam m < exam.targetScore) ∧ 
              examScore exam n ≥ exam.targetScore ↔ n = 716 := by
  sorry

end min_words_to_learn_l1678_167882


namespace abc_product_one_l1678_167857

theorem abc_product_one (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h_eq : a^2 + 1/b^2 = b^2 + 1/c^2 ∧ b^2 + 1/c^2 = c^2 + 1/a^2) :
  |a*b*c| = 1 := by
  sorry

end abc_product_one_l1678_167857


namespace max_sum_squares_given_sum_cubes_l1678_167802

theorem max_sum_squares_given_sum_cubes 
  (a b c d : ℝ) 
  (h : a^3 + b^3 + c^3 + d^3 = 8) : 
  ∃ (m : ℝ), m = 4 ∧ ∀ (x y z w : ℝ), x^3 + y^3 + z^3 + w^3 = 8 → x^2 + y^2 + z^2 + w^2 ≤ m :=
sorry

end max_sum_squares_given_sum_cubes_l1678_167802


namespace unique_solution_for_equation_l1678_167859

theorem unique_solution_for_equation : ∃! (x y : ℕ), 1983 = 1982 * x - 1981 * y ∧ 1983 = 1982 * 31 * 5 - 1981 * (31 * 5 - 1) := by
  sorry

end unique_solution_for_equation_l1678_167859


namespace plane_through_line_and_point_l1678_167820

/-- A line in 3D space defined by symmetric equations -/
structure Line3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A plane in 3D space defined by its equation ax + by + cz + d = 0 -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point3D) (l : Line3D) : Prop :=
  (p.x - l.a) / l.b = (p.y - l.c) / l.d ∧ (p.y - l.c) / l.d = p.z / l.f

/-- Check if a point lies on a plane -/
def pointOnPlane (p : Point3D) (pl : Plane) : Prop :=
  pl.a * p.x + pl.b * p.y + pl.c * p.z + pl.d = 0

theorem plane_through_line_and_point 
  (l : Line3D) 
  (p : Point3D) : 
  ∃ (pl : Plane), 
    (∀ (q : Point3D), pointOnLine q l → pointOnPlane q pl) ∧ 
    pointOnPlane p pl ∧ 
    pl.a = 5 ∧ pl.b = -2 ∧ pl.c = 2 ∧ pl.d = 1 := by
  sorry

#check plane_through_line_and_point

end plane_through_line_and_point_l1678_167820


namespace expression_evaluation_l1678_167872

theorem expression_evaluation : 3^(0^(1^2)) + ((3^0)^2)^1 = 2 := by
  sorry

end expression_evaluation_l1678_167872


namespace complement_of_A_in_U_l1678_167864

def U : Finset ℕ := {1, 3, 5, 7, 9}
def A : Finset ℕ := {1, 5, 7}

theorem complement_of_A_in_U :
  U \ A = {3, 9} := by sorry

end complement_of_A_in_U_l1678_167864


namespace white_surface_fraction_l1678_167879

/-- Represents a cube with its properties -/
structure Cube where
  edge_length : ℕ
  total_subcubes : ℕ
  white_subcubes : ℕ
  black_subcubes : ℕ

/-- Calculates the surface area of a cube -/
def surface_area (c : Cube) : ℕ := 6 * c.edge_length * c.edge_length

/-- Calculates the number of exposed faces of subcubes at diagonal ends -/
def exposed_diagonal_faces (c : Cube) : ℕ := 3 * c.black_subcubes

/-- Theorem: The fraction of white surface area in the given cube configuration is 1/2 -/
theorem white_surface_fraction (c : Cube) 
  (h1 : c.edge_length = 4)
  (h2 : c.total_subcubes = 64)
  (h3 : c.white_subcubes = 48)
  (h4 : c.black_subcubes = 16)
  (h5 : exposed_diagonal_faces c = c.black_subcubes * 3) :
  (surface_area c - exposed_diagonal_faces c) / surface_area c = 1 / 2 := by
  sorry

end white_surface_fraction_l1678_167879


namespace distance_to_larger_section_l1678_167892

/-- Given a right hexagonal pyramid with two parallel cross sections -/
structure HexagonalPyramid where
  /-- Ratio of areas of two parallel cross sections -/
  area_ratio : ℝ
  /-- Distance between the two parallel cross sections -/
  distance_between_sections : ℝ

/-- Theorem stating the distance from apex to larger cross section -/
theorem distance_to_larger_section (pyramid : HexagonalPyramid)
  (h_area_ratio : pyramid.area_ratio = 4 / 9)
  (h_distance : pyramid.distance_between_sections = 12) :
  ∃ (d : ℝ), d = 36 ∧ d > 0 ∧ 
  d = (pyramid.distance_between_sections * 3) / (1 - (pyramid.area_ratio)^(1/2)) :=
sorry

end distance_to_larger_section_l1678_167892


namespace complex_number_in_third_quadrant_l1678_167896

/-- The complex number z = (2-i)/i corresponds to a point in the third quadrant -/
theorem complex_number_in_third_quadrant :
  let i : ℂ := Complex.I
  let z : ℂ := (2 - i) / i
  (z.re < 0) ∧ (z.im < 0) :=
by sorry

end complex_number_in_third_quadrant_l1678_167896


namespace smallest_valid_number_l1678_167852

def is_valid_number (n : Nat) : Prop :=
  (n ≥ 1000 ∧ n ≤ 9999) ∧
  ∃ (k : Nat), 
    (n / (n % 100) = k^2) ∧
    (k^2 = (n / 100 + 1)^2)

theorem smallest_valid_number : 
  is_valid_number 1805 ∧ 
  ∀ (m : Nat), is_valid_number m → m ≥ 1805 :=
by sorry

end smallest_valid_number_l1678_167852


namespace f_derivative_sum_l1678_167834

noncomputable def f (x : ℝ) : ℝ := Real.log 9 * (Real.log x / Real.log 3)

theorem f_derivative_sum : 
  (deriv (λ _ : ℝ => f 2)) 0 + (deriv f) 2 = 1 := by sorry

end f_derivative_sum_l1678_167834


namespace absolute_value_equation_extrema_l1678_167863

theorem absolute_value_equation_extrema :
  ∀ x : ℝ, |x - 3| = 10 → (∃ y : ℝ, |y - 3| = 10 ∧ y ≥ x) ∧ (∃ z : ℝ, |z - 3| = 10 ∧ z ≤ x) ∧
  (∀ w : ℝ, |w - 3| = 10 → w ≤ 13 ∧ w ≥ -7) :=
by sorry

end absolute_value_equation_extrema_l1678_167863


namespace number_times_three_equals_33_l1678_167830

theorem number_times_three_equals_33 : ∃ x : ℝ, 3 * x = 33 ∧ x = 11 := by sorry

end number_times_three_equals_33_l1678_167830


namespace school_workbooks_calculation_l1678_167827

/-- The number of workbooks a school should buy given the number of classes,
    workbooks per class, and spare workbooks. -/
def total_workbooks (num_classes : ℕ) (workbooks_per_class : ℕ) (spare_workbooks : ℕ) : ℕ :=
  num_classes * workbooks_per_class + spare_workbooks

/-- Theorem stating that the total number of workbooks the school should buy
    is equal to 25 * 144 + 80, given the specific conditions of the problem. -/
theorem school_workbooks_calculation :
  total_workbooks 25 144 80 = 25 * 144 + 80 := by
  sorry

end school_workbooks_calculation_l1678_167827


namespace total_cotton_needed_l1678_167846

/-- The amount of cotton needed for one tee-shirt in feet -/
def cotton_per_shirt : ℝ := 4

/-- The number of tee-shirts to be made -/
def num_shirts : ℕ := 15

/-- Theorem stating the total amount of cotton needed -/
theorem total_cotton_needed : 
  cotton_per_shirt * (num_shirts : ℝ) = 60 := by sorry

end total_cotton_needed_l1678_167846


namespace greatest_whole_number_inequality_l1678_167831

theorem greatest_whole_number_inequality :
  ∀ x : ℤ, (7 * x - 8 < 4 - 2 * x) → x ≤ 1 :=
by
  sorry

end greatest_whole_number_inequality_l1678_167831


namespace sin_alpha_plus_7pi_over_6_l1678_167809

theorem sin_alpha_plus_7pi_over_6 (α : ℝ) 
  (h : Real.cos (α - π / 6) + Real.sin α = (4 / 5) * Real.sqrt 3) : 
  Real.sin (α + 7 * π / 6) = -(4 / 5) := by
  sorry

end sin_alpha_plus_7pi_over_6_l1678_167809


namespace ice_cream_cones_sold_l1678_167865

theorem ice_cream_cones_sold (milkshakes : ℕ) (difference : ℕ) : 
  milkshakes = 82 → 
  milkshakes = ice_cream_cones + difference → 
  difference = 15 →
  ice_cream_cones = 67 :=
by
  sorry

end ice_cream_cones_sold_l1678_167865


namespace quadratic_inequality_solutions_l1678_167875

/-- The quadratic function f(x) = x^2 + ax + 6 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 6

theorem quadratic_inequality_solutions (a : ℝ) :
  (a = 5 → {x : ℝ | f 5 x < 0} = {x : ℝ | -3 < x ∧ x < -2}) ∧
  ({x : ℝ | f a x > 0} = Set.univ → a ∈ Set.Ioo (-2*Real.sqrt 6) (2*Real.sqrt 6)) :=
sorry

end quadratic_inequality_solutions_l1678_167875


namespace cos_180_deg_l1678_167867

-- Define cosine function for angles in degrees
noncomputable def cos_deg (θ : ℝ) : ℝ := 
  Real.cos (θ * Real.pi / 180)

-- Theorem statement
theorem cos_180_deg : cos_deg 180 = -1 := by
  sorry

end cos_180_deg_l1678_167867


namespace midpoint_trajectory_l1678_167886

/-- The trajectory of the midpoint of a line segment with one end fixed and the other on a circle -/
theorem midpoint_trajectory (m n x y : ℝ) : 
  (m + 1)^2 + n^2 = 4 →  -- B(m, n) is on the circle (x+1)^2 + y^2 = 4
  x = (m + 4) / 2 →      -- x-coordinate of midpoint M
  y = (n - 3) / 2 →      -- y-coordinate of midpoint M
  (x - 3/2)^2 + (y + 3/2)^2 = 1 := by
sorry


end midpoint_trajectory_l1678_167886


namespace games_given_to_neil_l1678_167828

theorem games_given_to_neil (henry_initial : ℕ) (neil_initial : ℕ) (games_given : ℕ) : 
  henry_initial = 33 →
  neil_initial = 2 →
  henry_initial - games_given = 4 * (neil_initial + games_given) →
  games_given = 5 := by
sorry

end games_given_to_neil_l1678_167828


namespace area_covered_by_strips_l1678_167883

/-- The area covered by overlapping rectangular strips -/
theorem area_covered_by_strips (n : ℕ) (length width overlap_length : ℝ) : 
  n = 5 → 
  length = 12 → 
  width = 1 → 
  overlap_length = 2 → 
  (n : ℝ) * length * width - (n.choose 2 : ℝ) * overlap_length * width = 40 := by
  sorry

#check area_covered_by_strips

end area_covered_by_strips_l1678_167883


namespace disjunction_implies_conjunction_false_l1678_167856

theorem disjunction_implies_conjunction_false :
  ¬(∀ (p q : Prop), (p ∨ q) → (p ∧ q)) := by
  sorry

end disjunction_implies_conjunction_false_l1678_167856


namespace snacks_at_dawn_l1678_167893

theorem snacks_at_dawn (S : ℕ) : 
  (3 * S / 5 : ℚ) = 180 → S = 300 := by
  sorry

end snacks_at_dawn_l1678_167893


namespace fruit_store_discount_l1678_167804

/-- 
Given a fruit store scenario with:
- Total weight of fruit: 1000kg
- Cost price: 7 yuan per kg
- Original selling price: 10 yuan per kg
- Half of the fruit is sold at original price
- Total profit must not be less than 2000 yuan

This theorem states that the minimum discount factor x for the remaining half of the fruit
satisfies: x ≤ 7/11
-/
theorem fruit_store_discount (total_weight : ℝ) (cost_price selling_price : ℝ) 
  (min_profit : ℝ) (x : ℝ) :
  total_weight = 1000 →
  cost_price = 7 →
  selling_price = 10 →
  min_profit = 2000 →
  (total_weight / 2 * (selling_price - cost_price) + 
   total_weight / 2 * (selling_price * (1 - x) - cost_price) ≥ min_profit) →
  x ≤ 7 / 11 := by
  sorry


end fruit_store_discount_l1678_167804


namespace g_value_l1678_167840

-- Define the polynomials f and g
variable (f g : ℝ → ℝ)

-- Define the conditions
axiom f_def : ∀ x, f x = x^4 - x^2 - 3
axiom sum_eq : ∀ x, f x + g x = 3 * x^2 - 1

-- State the theorem
theorem g_value : ∀ x, g x = -x^4 + 4 * x^2 + 2 := by sorry

end g_value_l1678_167840


namespace sugar_amount_in_new_recipe_l1678_167833

/-- Represents the ratio of ingredients in a recipe -/
structure RecipeRatio :=
  (flour : ℚ)
  (water : ℚ)
  (sugar : ℚ)

/-- The original recipe ratio -/
def originalRatio : RecipeRatio :=
  { flour := 11, water := 8, sugar := 1 }

/-- The new recipe ratio -/
def newRatio : RecipeRatio :=
  { flour := 22, water := 8, sugar := 1 }

/-- The amount of water in the new recipe -/
def newWaterAmount : ℚ := 4

/-- Theorem stating that the amount of sugar in the new recipe is 0.5 cups -/
theorem sugar_amount_in_new_recipe :
  (newWaterAmount * newRatio.sugar) / newRatio.water = 1/2 := by
  sorry


end sugar_amount_in_new_recipe_l1678_167833


namespace pythagorean_triple_value_l1678_167862

theorem pythagorean_triple_value (a : ℝ) : 
  (3 : ℝ)^2 + a^2 = 5^2 → a = 4 := by
sorry

end pythagorean_triple_value_l1678_167862


namespace division_remainder_proof_l1678_167870

theorem division_remainder_proof (dividend : Nat) (divisor : Nat) (quotient : Nat) (remainder : Nat) :
  dividend = 162 →
  divisor = 17 →
  quotient = 9 →
  dividend = divisor * quotient + remainder →
  remainder = 9 := by
sorry

end division_remainder_proof_l1678_167870


namespace perpendicular_vectors_tan_2x_l1678_167841

theorem perpendicular_vectors_tan_2x (x : ℝ) : 
  let a : ℝ × ℝ := (Real.cos x, Real.sin x)
  let b : ℝ × ℝ := (Real.sqrt 3, -1)
  (a.1 * b.1 + a.2 * b.2 = 0) → Real.tan (2 * x) = -Real.sqrt 3 := by
  sorry

end perpendicular_vectors_tan_2x_l1678_167841


namespace table_capacity_l1678_167839

theorem table_capacity (invited : ℕ) (no_shows : ℕ) (tables : ℕ) : 
  invited = 18 → no_shows = 12 → tables = 2 → 
  (invited - no_shows) / tables = 3 := by sorry

end table_capacity_l1678_167839


namespace acid_mixture_percentage_l1678_167818

theorem acid_mixture_percentage :
  ∀ (a w : ℝ),
  a > 0 ∧ w > 0 →
  a / (a + w + 2) = 0.3 →
  (a + 2) / (a + w + 4) = 0.4 →
  a / (a + w) = 0.36 := by
sorry

end acid_mixture_percentage_l1678_167818


namespace number_divided_by_2000_l1678_167890

theorem number_divided_by_2000 : ∃ x : ℝ, x / 2000 = 0.012625 ∧ x = 25.25 := by
  sorry

end number_divided_by_2000_l1678_167890


namespace wage_difference_l1678_167866

/-- Represents the hourly wages of employees at Joe's Steakhouse -/
structure SteakhouseWages where
  manager : ℝ
  dishwasher : ℝ
  chef : ℝ

/-- The conditions for wages at Joe's Steakhouse -/
def validSteakhouseWages (w : SteakhouseWages) : Prop :=
  w.manager = 7.5 ∧
  w.dishwasher = w.manager / 2 ∧
  w.chef = w.dishwasher * 1.2

/-- The theorem stating the difference between manager's and chef's wages -/
theorem wage_difference (w : SteakhouseWages) (h : validSteakhouseWages w) :
  w.manager - w.chef = 3 := by
  sorry

end wage_difference_l1678_167866


namespace clock_hand_overlaps_l1678_167853

/-- Represents the number of revolutions a clock hand makes in a day -/
structure ClockHand where
  revolutions : ℕ

/-- Calculates the number of overlaps between two clock hands in a day -/
def overlaps (hand1 hand2 : ClockHand) : ℕ :=
  hand2.revolutions - hand1.revolutions

theorem clock_hand_overlaps :
  let hour_hand : ClockHand := ⟨2⟩
  let minute_hand : ClockHand := ⟨24⟩
  let second_hand : ClockHand := ⟨1440⟩
  (overlaps hour_hand minute_hand = 22) ∧
  (overlaps minute_hand second_hand = 1416) :=
by sorry

end clock_hand_overlaps_l1678_167853


namespace frame_area_percentage_l1678_167803

theorem frame_area_percentage (square_side : ℝ) (frame_width : ℝ) : 
  square_side = 80 → frame_width = 4 → 
  (square_side^2 - (square_side - 2 * frame_width)^2) / square_side^2 * 100 = 19 := by
  sorry

end frame_area_percentage_l1678_167803


namespace inequality_proof_l1678_167877

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a * b + b * c + c * a = 1) :
  a / b + b / c + c / a ≥ a^2 + b^2 + c^2 + 2 := by
  sorry

end inequality_proof_l1678_167877


namespace triangle_perimeter_from_quadratic_roots_l1678_167845

theorem triangle_perimeter_from_quadratic_roots :
  ∀ a b c : ℝ,
  (a^2 - 7*a + 10 = 0) →
  (b^2 - 7*b + 10 = 0) →
  (c^2 - 7*c + 10 = 0) →
  (a + b > c) → (b + c > a) → (c + a > b) →
  (a + b + c = 12 ∨ a + b + c = 6 ∨ a + b + c = 15) :=
by sorry

end triangle_perimeter_from_quadratic_roots_l1678_167845


namespace group_arrangement_count_l1678_167897

theorem group_arrangement_count :
  let total_men : ℕ := 4
  let total_women : ℕ := 5
  let small_group_size : ℕ := 2
  let large_group_size : ℕ := 5
  let small_group_count : ℕ := 2
  let large_group_count : ℕ := 1
  let total_people : ℕ := total_men + total_women
  let total_groups : ℕ := small_group_count + large_group_count

  -- Condition: At least one man and one woman in each group
  ∀ (men_in_small_group men_in_large_group : ℕ),
    (men_in_small_group ≥ 1 ∧ men_in_small_group < small_group_size) →
    (men_in_large_group ≥ 1 ∧ men_in_large_group < large_group_size) →
    (men_in_small_group * small_group_count + men_in_large_group * large_group_count = total_men) →

  -- The number of ways to arrange the groups
  (Nat.choose total_men 2 * Nat.choose total_women 3 +
   Nat.choose total_men 3 * Nat.choose total_women 2) = 100 :=
by
  sorry

end group_arrangement_count_l1678_167897


namespace sqrt_three_times_sqrt_twelve_l1678_167835

theorem sqrt_three_times_sqrt_twelve : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_three_times_sqrt_twelve_l1678_167835


namespace decimal_to_fraction_l1678_167851

theorem decimal_to_fraction : (2.24 : ℚ) = 56 / 25 := by sorry

end decimal_to_fraction_l1678_167851


namespace consecutive_even_integers_sum_l1678_167855

theorem consecutive_even_integers_sum (n : ℤ) : 
  (n + (n + 4) = 156) → (n + (n + 2) + (n + 4) = 234) := by
  sorry

end consecutive_even_integers_sum_l1678_167855


namespace fence_pole_count_l1678_167868

/-- Calculates the number of fence poles required for a path with bridges -/
def fence_poles (total_length : ℕ) (pole_spacing : ℕ) (bridge_lengths : List ℕ) : ℕ :=
  let fenced_length := total_length - bridge_lengths.sum
  let poles_per_side := fenced_length / pole_spacing
  let total_poles := 2 * poles_per_side + 2
  total_poles

theorem fence_pole_count : 
  fence_poles 2300 8 [48, 58, 62] = 534 := by
  sorry

end fence_pole_count_l1678_167868


namespace complex_equation_solution_l1678_167878

theorem complex_equation_solution (a b : ℝ) : 
  (Complex.I * 2 + 1) * a + b = Complex.I * 2 → a = 1 ∧ b = -1 := by
  sorry

end complex_equation_solution_l1678_167878


namespace completing_square_equivalence_l1678_167814

theorem completing_square_equivalence (x : ℝ) :
  x^2 + 7*x - 5 = 0 ↔ (x + 7/2)^2 = 69/4 := by
  sorry

end completing_square_equivalence_l1678_167814


namespace multiples_properties_l1678_167813

theorem multiples_properties (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 3 * k) 
  (hb : ∃ m : ℤ, b = 6 * m) : 
  (∃ n : ℤ, b = 3 * n) ∧ 
  (∃ p : ℤ, a - b = 3 * p) := by
sorry

end multiples_properties_l1678_167813


namespace basketball_time_calc_l1678_167819

/-- Calculates the time spent playing basketball given total play time and football play time. -/
def basketball_time (total_time : Real) (football_time : Nat) : Real :=
  total_time * 60 - football_time

/-- Proves that given a total play time of 1.5 hours and 60 minutes of football,
    the time spent playing basketball is 30 minutes. -/
theorem basketball_time_calc :
  basketball_time 1.5 60 = 30 := by
  sorry

end basketball_time_calc_l1678_167819


namespace sum_of_coordinates_is_60_l1678_167810

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the conditions for the points
def satisfies_conditions (p : Point) : Prop :=
  (abs (p.y - 10) = 4) ∧
  ((p.x - 5)^2 + (p.y - 10)^2 = 12^2)

-- Theorem statement
theorem sum_of_coordinates_is_60 :
  ∀ (p1 p2 p3 p4 : Point),
    satisfies_conditions p1 →
    satisfies_conditions p2 →
    satisfies_conditions p3 →
    satisfies_conditions p4 →
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 →
    p1.x + p1.y + p2.x + p2.y + p3.x + p3.y + p4.x + p4.y = 60 :=
by sorry

end sum_of_coordinates_is_60_l1678_167810


namespace function_value_at_two_l1678_167871

theorem function_value_at_two (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f x + 2 * f (1 / x) = 2 * x + 1) : 
  f 2 = -1/3 := by
  sorry

end function_value_at_two_l1678_167871


namespace total_molecular_weight_eq_1284_07_l1678_167895

/-- Atomic weights in g/mol -/
def atomic_weight (element : String) : ℝ :=
  match element with
  | "Ca" => 40.08
  | "O"  => 16.00
  | "H"  => 1.01
  | "Al" => 26.98
  | "S"  => 32.07
  | "K"  => 39.10
  | "N"  => 14.01
  | _    => 0

/-- Molecular weight of Ca(OH)2 in g/mol -/
def mw_calcium_hydroxide : ℝ :=
  atomic_weight "Ca" + 2 * (atomic_weight "O" + atomic_weight "H")

/-- Molecular weight of Al2(SO4)3 in g/mol -/
def mw_aluminum_sulfate : ℝ :=
  2 * atomic_weight "Al" + 3 * (atomic_weight "S" + 4 * atomic_weight "O")

/-- Molecular weight of KNO3 in g/mol -/
def mw_potassium_nitrate : ℝ :=
  atomic_weight "K" + atomic_weight "N" + 3 * atomic_weight "O"

/-- Total molecular weight of the mixture in grams -/
def total_molecular_weight : ℝ :=
  4 * mw_calcium_hydroxide + 2 * mw_aluminum_sulfate + 3 * mw_potassium_nitrate

theorem total_molecular_weight_eq_1284_07 :
  total_molecular_weight = 1284.07 := by
  sorry


end total_molecular_weight_eq_1284_07_l1678_167895


namespace reciprocal_problem_l1678_167808

theorem reciprocal_problem (x : ℚ) : 8 * x = 6 → 60 * (1 / x) = 80 := by
  sorry

end reciprocal_problem_l1678_167808


namespace scientific_notation_of_goat_wool_fineness_l1678_167869

theorem scientific_notation_of_goat_wool_fineness :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 0.000015 = a * (10 : ℝ) ^ n :=
by sorry

end scientific_notation_of_goat_wool_fineness_l1678_167869


namespace derivative_zero_in_interval_l1678_167884

theorem derivative_zero_in_interval (n : ℕ) (f : ℝ → ℝ) 
  (h_diff : ContDiff ℝ (n + 1) f)
  (h_f_zero : f 1 = 0 ∧ f 0 = 0)
  (h_derivatives_zero : ∀ k : ℕ, k ≤ n → (deriv^[k] f) 0 = 0) :
  ∃ x : ℝ, x ∈ (Set.Ioo 0 1) ∧ (deriv^[n + 1] f) x = 0 := by
sorry

end derivative_zero_in_interval_l1678_167884


namespace average_race_time_l1678_167816

/-- Calculates the average time in seconds for two racers to complete a block,
    given the time for one racer to complete the full block and the time for
    the other racer to complete half the block. -/
theorem average_race_time (carlos_time : ℝ) (diego_half_time : ℝ) : 
  carlos_time = 3 →
  diego_half_time = 2.5 →
  (carlos_time + 2 * diego_half_time) / 2 * 60 = 240 := by
  sorry

#check average_race_time

end average_race_time_l1678_167816


namespace equation_identity_l1678_167832

theorem equation_identity (x : ℝ) : (2*x - 1)^3 = 5*x^3 + (3*x + 1)*(x^2 - x - 1) - 10*x^2 + 10*x := by
  sorry

end equation_identity_l1678_167832


namespace geometric_sequence_property_l1678_167823

/-- A sequence a : ℕ → ℝ is geometric if there exists a common ratio r such that
    a(n+1) = r * a(n) for all n -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) :
  IsGeometric a →
  (3 * (a 3)^2 - 11 * (a 3) + 9 = 0) →
  (3 * (a 7)^2 - 11 * (a 7) + 9 = 0) →
  a 5 = Real.sqrt 3 := by
  sorry

end geometric_sequence_property_l1678_167823


namespace x_plus_2y_equals_5_l1678_167847

theorem x_plus_2y_equals_5 (x y : ℝ) 
  (h1 : (x + y) / 3 = 1) 
  (h2 : 2 * x + y = 4) : 
  x + 2 * y = 5 := by
  sorry

end x_plus_2y_equals_5_l1678_167847


namespace angle_ABC_measure_l1678_167838

/-- A regular octagon with a square inscribed such that one side of the square
    coincides with one side of the octagon. -/
structure OctagonWithSquare where
  /-- The measure of an interior angle of the regular octagon -/
  octagon_interior_angle : ℝ
  /-- The measure of an interior angle of the square -/
  square_interior_angle : ℝ
  /-- A is a vertex of the octagon -/
  A : Point
  /-- B is the next vertex of the octagon after A -/
  B : Point
  /-- C is a vertex of the inscribed square on the line extended from side AB -/
  C : Point
  /-- The measure of angle ABC -/
  angle_ABC : ℝ
  /-- The octagon is regular -/
  octagon_regular : octagon_interior_angle = 135
  /-- The square has right angles -/
  square_right_angle : square_interior_angle = 90

/-- The measure of angle ABC in the described configuration is 67.5 degrees -/
theorem angle_ABC_measure (config : OctagonWithSquare) : config.angle_ABC = 67.5 := by
  sorry

end angle_ABC_measure_l1678_167838


namespace min_value_theorem_l1678_167898

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 1/x + 1/y = 1 → 1/(x-1) + 4/(y-1) ≥ 4 :=
by sorry

end min_value_theorem_l1678_167898


namespace five_line_intersections_l1678_167801

/-- Represents a configuration of lines on a plane -/
structure LineConfiguration where
  num_lines : ℕ
  num_intersections : ℕ
  no_three_point_intersection : Bool

/-- The maximum number of intersections for n lines -/
def max_intersections (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating the impossibility of 11 intersections and possibility of 9 intersections -/
theorem five_line_intersections (config : LineConfiguration) :
  config.num_lines = 5 ∧ config.no_three_point_intersection = true →
  (config.num_intersections ≠ 11 ∧ 
   ∃ (config' : LineConfiguration), 
     config'.num_lines = 5 ∧ 
     config'.no_three_point_intersection = true ∧ 
     config'.num_intersections = 9) := by
  sorry

end five_line_intersections_l1678_167801


namespace bob_bake_time_proof_l1678_167811

/-- The time it takes Alice to bake a pie, in minutes -/
def alice_bake_time : ℝ := 5

/-- The time it takes Bob to bake a pie, in minutes -/
def bob_bake_time : ℝ := 6

/-- The total time available for baking, in minutes -/
def total_time : ℝ := 60

/-- The number of additional pies Alice can bake compared to Bob in the total time -/
def additional_pies : ℕ := 2

theorem bob_bake_time_proof :
  alice_bake_time = 5 ∧
  (total_time / alice_bake_time - total_time / bob_bake_time = additional_pies) →
  bob_bake_time = 6 := by
sorry

end bob_bake_time_proof_l1678_167811


namespace system_a_solution_system_b_solutions_l1678_167812

-- Part (a)
theorem system_a_solution (x y : ℝ) : 
  x^2 - 3*x*y - 4*y^2 = 0 ∧ x^3 + y^3 = 65 → (x = 4 ∧ y = 1) :=
sorry

-- Part (b)
theorem system_b_solutions (x y : ℝ) :
  x^2 + 2*y^2 = 17 ∧ 2*x*y - x^2 = 3 →
  ((x = 3 ∧ y = 2) ∨ 
   (x = -3 ∧ y = -2) ∨ 
   (x = Real.sqrt 3 / 3 ∧ y = 5 * Real.sqrt 3 / 3) ∨ 
   (x = -Real.sqrt 3 / 3 ∧ y = -5 * Real.sqrt 3 / 3)) :=
sorry

end system_a_solution_system_b_solutions_l1678_167812


namespace triangle_angle_measure_l1678_167842

theorem triangle_angle_measure (D E F : ℝ) : 
  D = 75 → 
  E = 2 * F + 15 → 
  D + E + F = 180 → 
  F = 30 := by sorry

end triangle_angle_measure_l1678_167842


namespace point_distance_to_y_axis_l1678_167848

theorem point_distance_to_y_axis (a : ℝ) : 
  (a + 3 > 0) →  -- Point is in the first quadrant (x-coordinate is positive)
  (a > 0) →      -- Point is in the first quadrant (y-coordinate is positive)
  (a + 3 = 5) →  -- Distance to y-axis is 5
  a = 2 := by
sorry

end point_distance_to_y_axis_l1678_167848


namespace max_value_of_a_plus_2b_l1678_167849

/-- Two circles in a 2D plane -/
structure TwoCircles where
  a : ℝ
  b : ℝ
  circle1 : (x y : ℝ) → x^2 + y^2 + 2*a*x + a^2 - 4 = 0
  circle2 : (x y : ℝ) → x^2 + y^2 - 4*b*y - 1 + 4*b^2 = 0

/-- The property that two circles have exactly three common tangents -/
def have_three_common_tangents (c : TwoCircles) : Prop := sorry

/-- The theorem stating the maximum value of a+2b -/
theorem max_value_of_a_plus_2b (c : TwoCircles) 
  (h : have_three_common_tangents c) : 
  ∃ (max : ℝ), max = 3 * Real.sqrt 2 ∧ 
  ∀ (a b : ℝ), c.a = a → c.b = b → a + 2*b ≤ max := by
  sorry

end max_value_of_a_plus_2b_l1678_167849


namespace unique_a_value_l1678_167894

def A (a : ℝ) : Set ℝ := {a + 2, (a + 1)^2, a^2 + 3*a + 3}

theorem unique_a_value : ∃! a : ℝ, 1 ∈ A a ∧ a = 0 := by sorry

end unique_a_value_l1678_167894


namespace complement_A_intersect_B_l1678_167821

-- Define set A
def A : Set ℝ := {x | x * (x - 1) < 0}

-- Define set B
def B : Set ℝ := {x | Real.exp x > 1}

-- Define the closed interval [1,+∞)
def closed_interval : Set ℝ := {x | x ≥ 1}

-- Theorem statement
theorem complement_A_intersect_B : (Set.univ \ A) ∩ B = closed_interval := by sorry

end complement_A_intersect_B_l1678_167821


namespace school_c_sample_size_l1678_167843

/-- Represents the number of teachers in each school -/
structure SchoolPopulation where
  schoolA : ℕ
  schoolB : ℕ
  schoolC : ℕ

/-- Represents the sampling parameters -/
structure SamplingParams where
  totalSample : ℕ
  population : SchoolPopulation

/-- Calculates the stratified sample size for a given school -/
def stratifiedSampleSize (params : SamplingParams) (schoolSize : ℕ) : ℕ :=
  (schoolSize * params.totalSample) / (params.population.schoolA + params.population.schoolB + params.population.schoolC)

/-- Theorem stating that the stratified sample size for School C is 10 -/
theorem school_c_sample_size :
  let params : SamplingParams := {
    totalSample := 60,
    population := {
      schoolA := 180,
      schoolB := 270,
      schoolC := 90
    }
  }
  stratifiedSampleSize params params.population.schoolC = 10 := by
  sorry


end school_c_sample_size_l1678_167843


namespace greatest_fraction_l1678_167822

theorem greatest_fraction (a b : ℕ) (h1 : a + b = 101) (h2 : (a : ℚ) / b ≤ 1/3) :
  (a : ℚ) / b ≤ 25/76 ∧ ∃ (a' b' : ℕ), a' + b' = 101 ∧ (a' : ℚ) / b' = 25/76 ∧ (a' : ℚ) / b' ≤ 1/3 := by
  sorry

end greatest_fraction_l1678_167822


namespace triangle_value_l1678_167806

theorem triangle_value (Δ q : ℤ) 
  (h1 : 3 * Δ * q = 63) 
  (h2 : 7 * (Δ + q) = 161) : 
  Δ = 1 := by
sorry

end triangle_value_l1678_167806


namespace equation_D_is_linear_l1678_167800

/-- Definition of a linear equation in one variable -/
def is_linear_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The equation 2x - 3 = 5 -/
def equation_D (x : ℝ) : ℝ := 2 * x - 3

theorem equation_D_is_linear : is_linear_equation equation_D := by
  sorry

end equation_D_is_linear_l1678_167800


namespace polynomial_division_remainder_l1678_167844

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ, 
  3 * X^2 - 22 * X + 70 = (X - 7) * q + 63 := by
  sorry

end polynomial_division_remainder_l1678_167844


namespace arithmetic_sequence_cosine_l1678_167876

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_cosine (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 1 + a 5 + a 9 = 8 * Real.pi →
  Real.cos (a 3 + a 7) = -1/2 :=
by sorry

end arithmetic_sequence_cosine_l1678_167876


namespace right_triangle_hypotenuse_l1678_167860

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 15 → b = 36 → c^2 = a^2 + b^2 → c = 39 := by sorry

end right_triangle_hypotenuse_l1678_167860


namespace unique_representation_of_two_over_prime_l1678_167873

theorem unique_representation_of_two_over_prime (p : ℕ) (h_prime : Nat.Prime p) (h_gt_two : p > 2) :
  ∃! (x y : ℕ), x ≠ y ∧ x > 0 ∧ y > 0 ∧ (2 : ℚ) / p = 1 / x + 1 / y ∧ x = p * (p + 1) / 2 ∧ y = (p + 1) / 2 := by
  sorry

end unique_representation_of_two_over_prime_l1678_167873


namespace bella_steps_to_meet_ella_l1678_167874

/-- The distance between Bella's and Ella's houses in feet -/
def total_distance : ℕ := 15840

/-- The number of feet Bella covers in one step -/
def feet_per_step : ℕ := 3

/-- Ella's speed relative to Bella's -/
def speed_ratio : ℕ := 3

/-- The number of steps Bella takes before meeting Ella -/
def steps_taken : ℕ := 1320

theorem bella_steps_to_meet_ella :
  total_distance * speed_ratio = steps_taken * feet_per_step * (speed_ratio + 1) :=
sorry

end bella_steps_to_meet_ella_l1678_167874
