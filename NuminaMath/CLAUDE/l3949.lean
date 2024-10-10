import Mathlib

namespace expression_evaluation_l3949_394913

theorem expression_evaluation :
  let x : ℝ := 1
  let y : ℝ := -2
  3 * y^2 - x^2 + (2 * x - y) - (x^2 + 3 * y^2) = 2 := by
  sorry

end expression_evaluation_l3949_394913


namespace diophantine_equation_solutions_l3949_394978

theorem diophantine_equation_solutions :
  ∀ x y z : ℕ+,
    x > y ∧ y > z →
    (1 : ℚ) / x + 2 / y + 3 / z = 1 →
    ((x = 36 ∧ y = 9 ∧ z = 4) ∨
     (x = 20 ∧ y = 10 ∧ z = 4) ∨
     (x = 15 ∧ y = 6 ∧ z = 5)) :=
by sorry

end diophantine_equation_solutions_l3949_394978


namespace ellipse_intersection_fixed_point_l3949_394972

-- Define the ellipse Γ
def Γ (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

-- Define the point M
def M : ℝ × ℝ := (0, 1)

-- Define a line l that intersects Γ at two points
def l (k : ℝ) (x y : ℝ) : Prop := y = (k^2 - 1) / (4*k) * x - 1/2

-- Define the property that PQ is a diameter of the circumcircle of MPQ
def isPQDiameterOfCircumcircle (P Q : ℝ × ℝ) : Prop :=
  (P.1 - M.1) * (Q.1 - M.1) + (P.2 - M.2) * (Q.2 - M.2) = 0

theorem ellipse_intersection_fixed_point :
  ∀ (k : ℝ) (P Q : ℝ × ℝ),
    k ≠ 0 →
    Γ P.1 P.2 →
    Γ Q.1 Q.2 →
    l k P.1 P.2 →
    l k Q.1 Q.2 →
    isPQDiameterOfCircumcircle P Q →
    P ≠ M ∧ Q ≠ M →
    ∃ (x y : ℝ), l k x y ∧ x = 0 ∧ y = -1/2 :=
sorry

end ellipse_intersection_fixed_point_l3949_394972


namespace expression_simplification_l3949_394989

theorem expression_simplification (x y : ℝ) (hx : x = 1) (hy : y = 2) :
  ((x + y) * (x - y) - (x - y)^2 + 2 * y * (x - y)) / (4 * y) = -1 := by
  sorry

end expression_simplification_l3949_394989


namespace probability_abs_diff_greater_quarter_l3949_394949

-- Define the coin-flipping method
def coin_flip_method : Real → Prop := sorry

-- Define the probability measure for the coin-flipping method
def P : (Real → Prop) → ℝ := sorry

-- Define the event where |x-y| > 1/4
def event : Real → Real → Prop :=
  fun x y => |x - y| > 1/4

-- Theorem statement
theorem probability_abs_diff_greater_quarter :
  P (fun x => P (fun y => event x y) = 423/1024) = 1 := by sorry

end probability_abs_diff_greater_quarter_l3949_394949


namespace ladder_length_l3949_394942

/-- Proves that the length of a ladder is 9.2 meters, given specific conditions. -/
theorem ladder_length (angle : Real) (foot_distance : Real) (length : Real) : 
  angle = 60 * π / 180 →
  foot_distance = 4.6 →
  Real.cos angle = foot_distance / length →
  length = 9.2 := by
sorry

end ladder_length_l3949_394942


namespace average_multiplication_invariance_l3949_394919

theorem average_multiplication_invariance (S : Finset ℝ) (n : ℕ) (h : n > 0) :
  let avg := (S.sum id) / n
  let new_avg := (S.sum (fun x => 10 * x)) / n
  avg = 7 ∧ new_avg = 70 →
  ∃ (m : ℕ), m > 0 ∧ (S.sum id) / m = 7 ∧ (S.sum (fun x => 10 * x)) / m = 70 :=
by sorry

end average_multiplication_invariance_l3949_394919


namespace certain_number_is_80_l3949_394957

theorem certain_number_is_80 : 
  ∃ x : ℝ, (70 : ℝ) = 0.6 * x + 22 ∧ x = 80 := by
  sorry

end certain_number_is_80_l3949_394957


namespace water_breadth_in_cistern_l3949_394986

/-- Represents a rectangular cistern with water --/
structure WaterCistern where
  length : ℝ
  width : ℝ
  wetSurfaceArea : ℝ
  breadth : ℝ

/-- Theorem stating the correct breadth of water in the cistern --/
theorem water_breadth_in_cistern (c : WaterCistern)
  (h_length : c.length = 7)
  (h_width : c.width = 5)
  (h_wetArea : c.wetSurfaceArea = 68.6)
  (h_breadth_calc : c.breadth = (c.wetSurfaceArea - c.length * c.width) / (2 * (c.length + c.width))) :
  c.breadth = 1.4 := by
  sorry

end water_breadth_in_cistern_l3949_394986


namespace fraction_problem_l3949_394946

theorem fraction_problem (f : ℚ) : f * 10 + 6 = 11 → f = 1/2 := by
  sorry

end fraction_problem_l3949_394946


namespace largest_prime_divisor_factorial_sum_l3949_394943

theorem largest_prime_divisor_factorial_sum : 
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ (Nat.factorial 10 + Nat.factorial 11) ∧ 
  ∀ q : ℕ, Nat.Prime q → q ∣ (Nat.factorial 10 + Nat.factorial 11) → q ≤ p :=
by sorry

end largest_prime_divisor_factorial_sum_l3949_394943


namespace extreme_point_property_g_maximum_bound_l3949_394907

-- Define the function f
def f (a b x : ℝ) : ℝ := x^3 - a*x - b

-- Define the function g
def g (a b x : ℝ) : ℝ := |f a b x|

theorem extreme_point_property (a b x₀ x₁ : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (x₀ - ε) (x₀ + ε), f a b x₀ ≤ f a b x) →
  f a b x₁ = f a b x₀ →
  x₁ ≠ x₀ →
  x₁ + 2*x₀ = 0 := by sorry

theorem g_maximum_bound (a b : ℝ) (ha : a > 0) :
  ∃ x ∈ Set.Icc (-1 : ℝ) 1, ∀ y ∈ Set.Icc (-1 : ℝ) 1, g a b x ≥ (1/4 : ℝ) ∧ g a b x ≥ g a b y := by sorry

end extreme_point_property_g_maximum_bound_l3949_394907


namespace hyperbola_center_l3949_394937

/-- The center of a hyperbola given by the equation ((3y+3)^2)/(7^2) - ((4x-8)^2)/(6^2) = 1 -/
theorem hyperbola_center (x y : ℝ) : 
  (((3 * y + 3)^2) / 7^2) - (((4 * x - 8)^2) / 6^2) = 1 → 
  (∃ (h k : ℝ), h = 2 ∧ k = -1 ∧ 
    ((y - k)^2) / ((7/3)^2) - ((x - h)^2) / ((3/2)^2) = 1) :=
by sorry

end hyperbola_center_l3949_394937


namespace prob_two_green_marbles_l3949_394921

/-- The probability of drawing two green marbles without replacement from a bag containing 5 blue marbles and 7 green marbles is 7/22. -/
theorem prob_two_green_marbles (blue_marbles green_marbles : ℕ) 
  (h_blue : blue_marbles = 5) (h_green : green_marbles = 7) :
  let total_marbles := blue_marbles + green_marbles
  let prob_first_green := green_marbles / total_marbles
  let prob_second_green := (green_marbles - 1) / (total_marbles - 1)
  prob_first_green * prob_second_green = 7 / 22 := by
  sorry

end prob_two_green_marbles_l3949_394921


namespace cube_of_three_fifths_l3949_394980

theorem cube_of_three_fifths : (3 / 5 : ℚ) ^ 3 = 27 / 125 := by
  sorry

end cube_of_three_fifths_l3949_394980


namespace simple_interest_rate_problem_solution_l3949_394979

/-- Simple interest calculation -/
theorem simple_interest_rate (principal time interest : ℝ) (h : principal > 0) (h2 : time > 0) :
  interest = principal * (25 : ℝ) / 100 * time →
  25 = (interest * 100) / (principal * time) :=
by
  sorry

/-- Specific problem instance -/
theorem problem_solution :
  let principal : ℝ := 800
  let time : ℝ := 2
  let interest : ℝ := 400
  (interest * 100) / (principal * time) = 25 :=
by
  sorry

end simple_interest_rate_problem_solution_l3949_394979


namespace halfway_fraction_l3949_394994

theorem halfway_fraction (a b : ℚ) (ha : a = 1/4) (hb : b = 1/2) :
  (a + b) / 2 = 3/8 := by
  sorry

end halfway_fraction_l3949_394994


namespace barbecue_packages_l3949_394963

/-- Represents the number of items in each package type -/
structure PackageSizes where
  hotDogs : Nat
  buns : Nat
  soda : Nat

/-- Represents the number of packages for each item type -/
structure PackageCounts where
  hotDogs : Nat
  buns : Nat
  soda : Nat

/-- Given package sizes, check if the package counts result in equal number of items -/
def hasEqualItems (sizes : PackageSizes) (counts : PackageCounts) : Prop :=
  sizes.hotDogs * counts.hotDogs = sizes.buns * counts.buns ∧
  sizes.hotDogs * counts.hotDogs = sizes.soda * counts.soda

/-- Check if a given package count is the smallest possible -/
def isSmallestCount (sizes : PackageSizes) (counts : PackageCounts) : Prop :=
  ∀ (other : PackageCounts),
    hasEqualItems sizes other →
    counts.hotDogs ≤ other.hotDogs ∧
    counts.buns ≤ other.buns ∧
    counts.soda ≤ other.soda

theorem barbecue_packages :
  let sizes : PackageSizes := ⟨9, 12, 15⟩
  let counts : PackageCounts := ⟨20, 15, 12⟩
  hasEqualItems sizes counts ∧ isSmallestCount sizes counts :=
by sorry

end barbecue_packages_l3949_394963


namespace rest_time_calculation_l3949_394939

theorem rest_time_calculation (walking_rate : ℝ) (total_distance : ℝ) (total_time : ℝ) :
  walking_rate = 10 →
  total_distance = 50 →
  total_time = 328 →
  (∃ (rest_time : ℝ),
    rest_time * 4 = total_time - (total_distance / walking_rate * 60) ∧
    rest_time = 7) := by
  sorry

end rest_time_calculation_l3949_394939


namespace initial_cookies_count_l3949_394976

def cookies_eaten : ℕ := 15
def cookies_left : ℕ := 78

theorem initial_cookies_count : 
  cookies_eaten + cookies_left = 93 := by sorry

end initial_cookies_count_l3949_394976


namespace hunting_duration_is_three_weeks_l3949_394999

/-- Represents the hunting scenario in the forest -/
structure ForestHunt where
  initialWeasels : ℕ
  initialRabbits : ℕ
  foxes : ℕ
  weaselsPerFoxPerWeek : ℕ
  rabbitsPerFoxPerWeek : ℕ
  remainingRodents : ℕ

/-- Calculates the hunting duration in weeks -/
def huntingDuration (hunt : ForestHunt) : ℚ :=
  let initialRodents := hunt.initialWeasels + hunt.initialRabbits
  let rodentsCaughtPerWeek := hunt.foxes * (hunt.weaselsPerFoxPerWeek + hunt.rabbitsPerFoxPerWeek)
  let totalRodentsCaught := initialRodents - hunt.remainingRodents
  totalRodentsCaught / rodentsCaughtPerWeek

/-- Theorem stating that the hunting duration is 3 weeks for the given scenario -/
theorem hunting_duration_is_three_weeks (hunt : ForestHunt) 
    (h1 : hunt.initialWeasels = 100)
    (h2 : hunt.initialRabbits = 50)
    (h3 : hunt.foxes = 3)
    (h4 : hunt.weaselsPerFoxPerWeek = 4)
    (h5 : hunt.rabbitsPerFoxPerWeek = 2)
    (h6 : hunt.remainingRodents = 96) :
    huntingDuration hunt = 3 := by
  sorry


end hunting_duration_is_three_weeks_l3949_394999


namespace speed_of_current_l3949_394997

/-- 
Given a man's speed with and against a current, this theorem proves 
the speed of the current.
-/
theorem speed_of_current 
  (speed_with_current : ℝ) 
  (speed_against_current : ℝ) 
  (h1 : speed_with_current = 20) 
  (h2 : speed_against_current = 18) : 
  ∃ (current_speed : ℝ), current_speed = 1 := by
  sorry

end speed_of_current_l3949_394997


namespace max_non_managers_l3949_394906

theorem max_non_managers (managers : ℕ) (non_managers : ℕ) : 
  (managers : ℚ) / non_managers > 7 / 32 →
  non_managers ≤ 36 →
  ∃ (max_non_managers : ℕ), max_non_managers = 36 ∧ 
    ∀ n : ℕ, n > max_non_managers → (managers : ℚ) / n ≤ 7 / 32 := by
  sorry

end max_non_managers_l3949_394906


namespace range_of_x_l3949_394983

theorem range_of_x (x : ℝ) : 
  ((x + 2) * (x - 3) ≤ 0) ∧ (abs (x + 1) ≥ 2) → 
  x ∈ Set.Icc 1 3 := by
sorry

end range_of_x_l3949_394983


namespace gumballs_per_package_l3949_394927

/-- Given that Nathan ate 20 gumballs, finished 4 whole boxes, and no gumballs were left,
    prove that there are 5 gumballs in each package. -/
theorem gumballs_per_package :
  ∀ (total_gumballs : ℕ) (boxes_finished : ℕ) (gumballs_per_package : ℕ),
    total_gumballs = 20 →
    boxes_finished = 4 →
    total_gumballs = boxes_finished * gumballs_per_package →
    gumballs_per_package = 5 := by
  sorry

end gumballs_per_package_l3949_394927


namespace smallest_integer_solution_l3949_394975

theorem smallest_integer_solution (x : ℤ) : x^2 - x = 24 → x ≥ -4 := by
  sorry

end smallest_integer_solution_l3949_394975


namespace inequality_always_holds_l3949_394971

theorem inequality_always_holds (a b c : ℝ) (h : a < b ∧ b < c) : a - c < b - c := by
  sorry

end inequality_always_holds_l3949_394971


namespace gcd_problem_l3949_394926

/-- The GCD operation -/
def gcd_op (a b : ℕ) : ℕ := Nat.gcd a b

/-- The problem statement -/
theorem gcd_problem (n m k j : ℕ+) :
  gcd_op (gcd_op (16 * n) (20 * m)) (gcd_op (18 * k) (24 * j)) = 2 := by
  sorry

end gcd_problem_l3949_394926


namespace janes_calculation_l3949_394973

theorem janes_calculation (x y z : ℝ) 
  (h1 : x - (y - z) = 17) 
  (h2 : x - y - z = 5) : 
  x - y = 11 := by
sorry

end janes_calculation_l3949_394973


namespace output_for_five_l3949_394993

def program_output (x : ℤ) : ℤ :=
  if x < 3 then 2 * x
  else if x > 3 then x * x - 1
  else 2

theorem output_for_five :
  program_output 5 = 24 :=
by sorry

end output_for_five_l3949_394993


namespace unique_integer_solution_l3949_394954

theorem unique_integer_solution (a b c : ℤ) :
  a^2 + b^2 + c^2 + 3 < a*b + 3*b + 2*c ↔ a = 1 ∧ b = 2 ∧ c = 1 := by
  sorry

end unique_integer_solution_l3949_394954


namespace circle_symmetry_l3949_394968

/-- The equation of a circle -/
def circle_equation (x y a : ℝ) : Prop := x^2 + y^2 + 2*a*x - 2*a*y = 0

/-- The line of symmetry -/
def symmetry_line (x y : ℝ) : Prop := x + y = 0

/-- Theorem stating that the circle is symmetric with respect to the line x + y = 0 -/
theorem circle_symmetry (a : ℝ) (h : a ≠ 0) :
  ∃ (r : ℝ), r > 0 ∧
  ∀ (x y : ℝ), circle_equation x y a ↔
    ∃ (x₀ y₀ : ℝ), symmetry_line x₀ y₀ ∧ (x - x₀)^2 + (y - y₀)^2 = r^2 :=
sorry

end circle_symmetry_l3949_394968


namespace max_angle_C_l3949_394936

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  angleSum : A + B + C = Real.pi

-- Define the condition sin²A + sin²B = 2sin²C
def specialCondition (t : Triangle) : Prop :=
  Real.sin t.A ^ 2 + Real.sin t.B ^ 2 = 2 * Real.sin t.C ^ 2

-- Theorem statement
theorem max_angle_C (t : Triangle) (h : specialCondition t) : 
  t.C ≤ Real.pi / 3 :=
sorry

end max_angle_C_l3949_394936


namespace inequality_and_equality_condition_l3949_394959

theorem inequality_and_equality_condition (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  (1/2) * (a + b)^2 + (1/4) * (a + b) ≥ a * Real.sqrt b + b * Real.sqrt a ∧
  ((1/2) * (a + b)^2 + (1/4) * (a + b) = a * Real.sqrt b + b * Real.sqrt a ↔ 
    (a = 0 ∧ b = 0) ∨ (a = 1/4 ∧ b = 1/4)) :=
by sorry

end inequality_and_equality_condition_l3949_394959


namespace total_amount_in_euros_l3949_394947

/-- Proves that the total amount in euros is 172.55 given the specified conditions --/
theorem total_amount_in_euros : 
  ∀ (x y z w : ℝ),
  y = 0.8 * x →
  z = 0.7 * x →
  w = 0.6 * x →
  y = 42 →
  z = 49 →
  x + w = 120 →
  (x + y + z + w) * 0.85 = 172.55 := by
  sorry

end total_amount_in_euros_l3949_394947


namespace toms_age_problem_l3949_394977

/-- Tom's age problem -/
theorem toms_age_problem (T N : ℝ) : 
  T > 0 ∧ N > 0 ∧ 
  (∃ (a b c d : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ a + b + c + d = T) ∧
  T - N = 3 * (T - 4 * N) →
  T / N = 11 / 2 := by
sorry

end toms_age_problem_l3949_394977


namespace gcd_7920_14553_l3949_394964

theorem gcd_7920_14553 : Nat.gcd 7920 14553 = 11 := by
  sorry

end gcd_7920_14553_l3949_394964


namespace bug_total_distance_l3949_394945

def bug_crawl (start : ℝ) (p1 p2 p3 : ℝ) : ℝ :=
  |p1 - start| + |p2 - p1| + |p3 - p2|

theorem bug_total_distance : bug_crawl 3 (-4) 6 2 = 21 := by
  sorry

end bug_total_distance_l3949_394945


namespace circle_mapping_l3949_394982

-- Define the complex plane
variable (z : ℂ)

-- Define the transformation function
def w (z : ℂ) : ℂ := 3 * z + 2

-- Define the original circle
def original_circle (z : ℂ) : Prop := z.re^2 + z.im^2 = 4

-- Define the mapped circle
def mapped_circle (w : ℂ) : Prop := (w.re - 2)^2 + w.im^2 = 36

-- Theorem statement
theorem circle_mapping :
  ∀ z, original_circle z → mapped_circle (w z) :=
sorry

end circle_mapping_l3949_394982


namespace test_questions_count_l3949_394958

theorem test_questions_count (score : ℤ) (correct : ℤ) (incorrect : ℤ) :
  score = correct - 2 * incorrect →
  score = 61 →
  correct = 87 →
  correct + incorrect = 100 := by
sorry

end test_questions_count_l3949_394958


namespace sum_of_roots_quadratic_sum_of_roots_specific_equation_l3949_394912

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∃ s : ℝ, s = x + y ∧ f x = 0 ∧ f y = 0 → s = -b / a) :=
by sorry

theorem sum_of_roots_specific_equation :
  let f : ℝ → ℝ := λ x => x^2 + 1995 * x - 1996
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∃ s : ℝ, s = x + y ∧ f x = 0 ∧ f y = 0 → s = -1995) :=
by sorry

end sum_of_roots_quadratic_sum_of_roots_specific_equation_l3949_394912


namespace perimeter_of_triangle_l3949_394955

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 9

-- Define the foci
def left_focus : ℝ × ℝ := sorry
def right_focus : ℝ × ℝ := sorry

-- Define points P and Q on the left branch
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry

-- State that P and Q are on the hyperbola
axiom P_on_hyperbola : hyperbola P.1 P.2
axiom Q_on_hyperbola : hyperbola Q.1 Q.2

-- State that PQ passes through the left focus
axiom PQ_through_left_focus : sorry

-- Define the length of PQ
def PQ_length : ℝ := 7

-- Define the property of hyperbola for P and Q
axiom hyperbola_property_P : dist P right_focus - dist P left_focus = 6
axiom hyperbola_property_Q : dist Q right_focus - dist Q left_focus = 6

-- Theorem to prove
theorem perimeter_of_triangle : 
  dist P right_focus + dist Q right_focus + PQ_length = 26 := sorry

end perimeter_of_triangle_l3949_394955


namespace correct_equation_transformation_l3949_394938

theorem correct_equation_transformation (y : ℝ) :
  (5 * y = -4 * y + 2) ↔ (5 * y + 4 * y = 2) :=
by sorry

end correct_equation_transformation_l3949_394938


namespace basketball_team_size_l3949_394966

theorem basketball_team_size (total_points : ℕ) (points_per_person : ℕ) (h1 : total_points = 18) (h2 : points_per_person = 2) :
  total_points / points_per_person = 9 :=
by sorry

end basketball_team_size_l3949_394966


namespace specific_hexagon_area_l3949_394981

/-- Regular hexagon with vertices J and L -/
structure RegularHexagon where
  J : ℝ × ℝ
  L : ℝ × ℝ

/-- The area of a regular hexagon -/
def hexagon_area (h : RegularHexagon) : ℝ := sorry

/-- Theorem stating the area of the specific regular hexagon -/
theorem specific_hexagon_area :
  let h : RegularHexagon := { J := (0, 0), L := (10, 2) }
  hexagon_area h = 156 * Real.sqrt 3 := by sorry

end specific_hexagon_area_l3949_394981


namespace group_b_more_stable_l3949_394910

/-- Represents a group of data with its variance -/
structure DataGroup where
  variance : ℝ

/-- Defines stability comparison between two data groups -/
def more_stable (a b : DataGroup) : Prop := a.variance < b.variance

/-- Theorem stating that Group B is more stable than Group A given their variances -/
theorem group_b_more_stable (group_a group_b : DataGroup)
  (h1 : group_a.variance = 0.2)
  (h2 : group_b.variance = 0.03) :
  more_stable group_b group_a := by
  sorry

#check group_b_more_stable

end group_b_more_stable_l3949_394910


namespace paperback_ratio_l3949_394950

/-- Represents the number of books Thabo owns in each category. -/
structure BookCollection where
  paperbackFiction : ℕ
  paperbackNonfiction : ℕ
  hardcoverNonfiction : ℕ

/-- Thabo's book collection satisfies the given conditions. -/
def validCollection (books : BookCollection) : Prop :=
  books.paperbackFiction + books.paperbackNonfiction + books.hardcoverNonfiction = 200 ∧
  books.paperbackNonfiction = books.hardcoverNonfiction + 20 ∧
  books.hardcoverNonfiction = 35

/-- The theorem stating that for any valid book collection, 
    the ratio of paperback fiction to paperback nonfiction is 2:1. -/
theorem paperback_ratio (books : BookCollection) 
  (h : validCollection books) : 
  books.paperbackFiction * 1 = books.paperbackNonfiction * 2 := by
  sorry

end paperback_ratio_l3949_394950


namespace wrapping_paper_rolls_l3949_394923

/-- The number of rolls of wrapping paper Savannah bought -/
def rolls_bought : ℕ := 3

/-- The total number of gifts Savannah has to wrap -/
def total_gifts : ℕ := 12

/-- The number of gifts wrapped with the first roll -/
def gifts_first_roll : ℕ := 3

/-- The number of gifts wrapped with the second roll -/
def gifts_second_roll : ℕ := 5

/-- The number of gifts wrapped with the third roll -/
def gifts_third_roll : ℕ := 4

theorem wrapping_paper_rolls :
  rolls_bought = 3 ∧
  total_gifts = 12 ∧
  gifts_first_roll = 3 ∧
  gifts_second_roll = 5 ∧
  gifts_third_roll = 4 ∧
  total_gifts = gifts_first_roll + gifts_second_roll + gifts_third_roll :=
by sorry

end wrapping_paper_rolls_l3949_394923


namespace library_shelves_l3949_394900

theorem library_shelves (books_per_shelf : ℕ) (total_books : ℕ) (h1 : books_per_shelf = 8) (h2 : total_books = 113920) :
  total_books / books_per_shelf = 14240 := by
  sorry

end library_shelves_l3949_394900


namespace boxes_with_neither_l3949_394903

theorem boxes_with_neither (total : ℕ) (markers : ℕ) (crayons : ℕ) (both : ℕ) 
  (h1 : total = 15)
  (h2 : markers = 9)
  (h3 : crayons = 5)
  (h4 : both = 4) :
  total - (markers + crayons - both) = 5 := by
  sorry

end boxes_with_neither_l3949_394903


namespace integer_roots_of_cubic_l3949_394961

def f (x : ℤ) : ℤ := x^3 - 6*x^2 - 4*x + 24

theorem integer_roots_of_cubic :
  ∀ x : ℤ, f x = 0 ↔ x = 2 ∨ x = -2 := by sorry

end integer_roots_of_cubic_l3949_394961


namespace cube_volume_from_surface_area_l3949_394962

/-- The volume of a cube given its surface area -/
theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) :
  surface_area = 150 → volume = 125 := by
  sorry

end cube_volume_from_surface_area_l3949_394962


namespace strawberry_area_l3949_394925

/-- The area of strawberries in a circular garden -/
theorem strawberry_area (d : ℝ) (h1 : d = 16) : ∃ (A : ℝ), A = 8 * Real.pi ∧ A = (1/8) * Real.pi * d^2 := by
  sorry

end strawberry_area_l3949_394925


namespace trapezium_side_length_l3949_394953

/-- Theorem: In a trapezium with one parallel side of length 12 cm, a distance between parallel sides
    of 14 cm, and an area of 196 square centimeters, the length of the other parallel side is 16 cm. -/
theorem trapezium_side_length 
  (side1 : ℝ) 
  (height : ℝ) 
  (area : ℝ) 
  (h1 : side1 = 12) 
  (h2 : height = 14) 
  (h3 : area = 196) 
  (h4 : area = (side1 + side2) * height / 2) : 
  side2 = 16 := by
  sorry

end trapezium_side_length_l3949_394953


namespace nineteen_team_tournament_games_l3949_394941

/-- Represents a single-elimination tournament. -/
structure Tournament where
  num_teams : ℕ
  is_single_elimination : Bool
  no_ties : Bool

/-- Calculates the number of games needed to determine a winner in a tournament. -/
def games_to_determine_winner (t : Tournament) : ℕ :=
  t.num_teams - 1

/-- Theorem stating that a single-elimination tournament with 19 teams and no ties requires 18 games to determine a winner. -/
theorem nineteen_team_tournament_games (t : Tournament) 
  (h1 : t.num_teams = 19) 
  (h2 : t.is_single_elimination = true) 
  (h3 : t.no_ties = true) : 
  games_to_determine_winner t = 18 := by
  sorry


end nineteen_team_tournament_games_l3949_394941


namespace no_snow_probability_l3949_394944

theorem no_snow_probability (p : ℚ) (h : p = 4/5) :
  (1 - p)^5 = 1/3125 := by sorry

end no_snow_probability_l3949_394944


namespace absolute_value_sum_l3949_394940

theorem absolute_value_sum (x p : ℝ) : 
  (|x - 2| = p) → (x > 2) → (x + p = 2*p + 2) := by sorry

end absolute_value_sum_l3949_394940


namespace sqrt_product_exists_l3949_394998

theorem sqrt_product_exists (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : ∃ x : ℝ, x^2 = a * b := by
  sorry

end sqrt_product_exists_l3949_394998


namespace opposite_of_2023_l3949_394918

theorem opposite_of_2023 : -(2023 : ℤ) = -2023 := by sorry

end opposite_of_2023_l3949_394918


namespace number_puzzle_l3949_394931

theorem number_puzzle : ∃ x : ℚ, 45 + 3 * x = 60 ∧ x = 5 := by sorry

end number_puzzle_l3949_394931


namespace ant_movement_l3949_394987

-- Define the grid
structure Grid :=
  (has_black_pairs : Bool)
  (has_white_pairs : Bool)

-- Define the ant's position
inductive Position
  | Black
  | White

-- Define a single move
def move (p : Position) : Position :=
  match p with
  | Position.Black => Position.White
  | Position.White => Position.Black

-- Define the number of moves
def num_moves : Nat := 4

-- Define the function to count black finishing squares
def count_black_finish (g : Grid) : Nat :=
  sorry

-- Theorem statement
theorem ant_movement (g : Grid) :
  g.has_black_pairs = true →
  g.has_white_pairs = true →
  count_black_finish g = 6 :=
sorry

end ant_movement_l3949_394987


namespace retired_faculty_surveys_l3949_394909

/-- Given a total number of surveys and a ratio of surveys from different groups,
    calculate the number of surveys from the retired faculty. -/
theorem retired_faculty_surveys
  (total_surveys : ℕ)
  (retired_ratio : ℕ)
  (current_ratio : ℕ)
  (student_ratio : ℕ)
  (h1 : total_surveys = 300)
  (h2 : retired_ratio = 2)
  (h3 : current_ratio = 8)
  (h4 : student_ratio = 40) :
  (total_surveys * retired_ratio) / (retired_ratio + current_ratio + student_ratio) = 12 := by
  sorry

#check retired_faculty_surveys

end retired_faculty_surveys_l3949_394909


namespace starting_number_for_prime_factors_of_210_l3949_394969

def isPrime (n : Nat) : Prop := sorry

def isFactor (a b : Nat) : Prop := sorry

theorem starting_number_for_prime_factors_of_210 :
  ∃ (start : Nat),
    start ≤ 100 ∧
    (∀ p, isPrime p → p > start → p ≤ 100 → isFactor p 210 →
      ∃ (primes : Finset Nat),
        primes.card = 4 ∧
        (∀ q ∈ primes, isPrime q ∧ q > start ∧ q ≤ 100 ∧ isFactor q 210)) ∧
    (∀ start' > start,
      ¬(∃ (primes : Finset Nat),
        primes.card = 4 ∧
        (∀ q ∈ primes, isPrime q ∧ q > start' ∧ q ≤ 100 ∧ isFactor q 210))) ∧
    start = 1 :=
by sorry

end starting_number_for_prime_factors_of_210_l3949_394969


namespace arithmetic_sequence_fifth_term_l3949_394965

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 3 + a 8 = 22)
  (h_sixth : a 6 = 8) :
  a 5 = 14 := by
  sorry

end arithmetic_sequence_fifth_term_l3949_394965


namespace sum_of_squared_fractions_l3949_394908

theorem sum_of_squared_fractions (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a^2 / (b - c)^2 + b^2 / (c - a)^2 + c^2 / (a - b)^2 = 0 := by
  sorry

end sum_of_squared_fractions_l3949_394908


namespace gcd_1151_3079_l3949_394901

theorem gcd_1151_3079 : Nat.gcd 1151 3079 = 1 := by
  sorry

end gcd_1151_3079_l3949_394901


namespace parallel_line_y_intercept_l3949_394902

/-- A line in the xy-plane can be represented by its slope and a point it passes through. -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The y-intercept of a line is the y-coordinate of the point where the line intersects the y-axis. -/
def y_intercept (l : Line) : ℝ :=
  l.point.2 - l.slope * l.point.1

/-- Given that line b is parallel to y = 3x - 2 and passes through (3, 4), its y-intercept is -5. -/
theorem parallel_line_y_intercept :
  let reference_line : Line := { slope := 3, point := (0, -2) }
  let b : Line := { slope := reference_line.slope, point := (3, 4) }
  y_intercept b = -5 := by
  sorry

end parallel_line_y_intercept_l3949_394902


namespace dihedral_angle_BAC_ACD_is_120_degrees_l3949_394951

-- Define a unit cube
def UnitCube := Set (ℝ × ℝ × ℝ)

-- Define a function to calculate the dihedral angle between two faces of a cube
def dihedralAngle (cube : UnitCube) (face1 face2 : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

-- Define the specific faces for the B-A₁C-D dihedral angle
def faceBAC (cube : UnitCube) : Set (ℝ × ℝ × ℝ) := sorry
def faceACD (cube : UnitCube) : Set (ℝ × ℝ × ℝ) := sorry

-- State the theorem
theorem dihedral_angle_BAC_ACD_is_120_degrees (cube : UnitCube) : 
  dihedralAngle cube (faceBAC cube) (faceACD cube) = 120 * (π / 180) := by sorry

end dihedral_angle_BAC_ACD_is_120_degrees_l3949_394951


namespace flagstaff_height_l3949_394984

/-- Given a flagstaff and a building casting shadows under similar conditions, 
    this theorem proves the height of the flagstaff. -/
theorem flagstaff_height 
  (building_height : ℝ) 
  (building_shadow : ℝ) 
  (flagstaff_shadow : ℝ) 
  (h_building : building_height = 12.5)
  (h_building_shadow : building_shadow = 28.75)
  (h_flagstaff_shadow : flagstaff_shadow = 40.25) :
  (building_height * flagstaff_shadow) / building_shadow = 17.5 := by
  sorry

#check flagstaff_height

end flagstaff_height_l3949_394984


namespace petyas_coins_l3949_394920

/-- Represents the denominations of coins --/
inductive Coin
  | OneRuble
  | TwoRubles
  | Other

/-- Represents Petya's pocket of coins --/
structure Pocket where
  coins : List Coin

/-- Checks if a list of coins contains at least one 1 ruble coin --/
def hasOneRuble (coins : List Coin) : Prop :=
  Coin.OneRuble ∈ coins

/-- Checks if a list of coins contains at least one 2 rubles coin --/
def hasTwoRubles (coins : List Coin) : Prop :=
  Coin.TwoRubles ∈ coins

/-- The main theorem to prove --/
theorem petyas_coins (p : Pocket) :
  (∀ (subset : List Coin), subset ⊆ p.coins → subset.length = 3 → hasOneRuble subset) →
  (∀ (subset : List Coin), subset ⊆ p.coins → subset.length = 4 → hasTwoRubles subset) →
  p.coins.length = 5 →
  p.coins = [Coin.OneRuble, Coin.OneRuble, Coin.OneRuble, Coin.TwoRubles, Coin.TwoRubles] :=
by sorry

end petyas_coins_l3949_394920


namespace soda_crate_weight_l3949_394916

/-- Given the following conditions:
  - Bridge weight limit is 20,000 pounds
  - Empty truck weight is 12,000 pounds
  - There are 20 soda crates
  - There are 3 dryers
  - Each dryer weighs 3,000 pounds
  - Weight of produce is twice the weight of soda
  - Fully loaded truck weighs 24,000 pounds

  Prove that each soda crate weighs 50 pounds -/
theorem soda_crate_weight :
  ∀ (bridge_limit : ℕ) 
    (empty_truck_weight : ℕ) 
    (num_soda_crates : ℕ) 
    (num_dryers : ℕ) 
    (dryer_weight : ℕ) 
    (loaded_truck_weight : ℕ),
  bridge_limit = 20000 →
  empty_truck_weight = 12000 →
  num_soda_crates = 20 →
  num_dryers = 3 →
  dryer_weight = 3000 →
  loaded_truck_weight = 24000 →
  ∃ (soda_weight produce_weight : ℕ),
    produce_weight = 2 * soda_weight ∧
    loaded_truck_weight = empty_truck_weight + num_dryers * dryer_weight + soda_weight + produce_weight →
    soda_weight / num_soda_crates = 50 :=
by sorry

end soda_crate_weight_l3949_394916


namespace absolute_value_equation_solution_l3949_394915

theorem absolute_value_equation_solution :
  ∀ x : ℝ, |x + 5| = 3 * x - 1 ↔ x = 3 := by sorry

end absolute_value_equation_solution_l3949_394915


namespace max_value_inequality_l3949_394924

theorem max_value_inequality (x y : ℝ) (h1 : |x - 1| ≤ 1) (h2 : |y - 2| ≤ 1) :
  |x - 2*y + 1| ≤ 5 ∧ ∀ (z : ℝ), (∀ (a b : ℝ), |a - 1| ≤ 1 → |b - 2| ≤ 1 → |a - 2*b + 1| ≤ z) → 5 ≤ z :=
by sorry

end max_value_inequality_l3949_394924


namespace golden_ratio_in_line_segment_l3949_394948

theorem golden_ratio_in_line_segment (A B C : ℝ) (k : ℝ) 
  (h1 : B > A ∧ B < C)  -- B is between A and C
  (h2 : (C - B) / (B - A) = k)  -- BC/AB = k
  (h3 : (B - A) / (C - A) = k)  -- AB/AC = k
  : k = (Real.sqrt 5 - 1) / 2 := by
  sorry

end golden_ratio_in_line_segment_l3949_394948


namespace arithmetic_sequence_length_arithmetic_sequence_length_is_8_l3949_394932

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ → Prop :=
  λ aₙ ↦ aₙ = a₁ + (n - 1 : ℤ) * d

theorem arithmetic_sequence_length :
  ∃ n : ℕ, n > 0 ∧ arithmetic_sequence 20 (-3) n (-2) ∧ 
  ∀ m : ℕ, m > n → ¬arithmetic_sequence 20 (-3) m (-2) := by
  sorry

theorem arithmetic_sequence_length_is_8 :
  ∃! n : ℕ, n > 0 ∧ arithmetic_sequence 20 (-3) n (-2) ∧ 
  ∀ m : ℕ, m > n → ¬arithmetic_sequence 20 (-3) m (-2) ∧ n = 8 := by
  sorry

end arithmetic_sequence_length_arithmetic_sequence_length_is_8_l3949_394932


namespace money_difference_proof_l3949_394995

/-- The number of nickels in a quarter -/
def nickels_per_quarter : ℕ := 5

/-- Charles' quarters -/
def charles_quarters (q : ℕ) : ℕ := 7 * q + 3

/-- Richard's quarters -/
def richard_quarters (q : ℕ) : ℕ := 3 * q + 7

/-- The difference in money between Charles and Richard, expressed in nickels -/
def money_difference_in_nickels (q : ℕ) : ℕ := 
  nickels_per_quarter * (charles_quarters q - richard_quarters q)

theorem money_difference_proof (q : ℕ) : 
  money_difference_in_nickels q = 20 * (q - 1) := by
  sorry

end money_difference_proof_l3949_394995


namespace corrected_mean_l3949_394991

theorem corrected_mean (n : ℕ) (original_mean : ℝ) (wrong_value correct_value : ℝ) :
  n = 50 ∧ original_mean = 36 ∧ wrong_value = 23 ∧ correct_value = 45 →
  (n * original_mean + (correct_value - wrong_value)) / n = 36.44 := by
  sorry

end corrected_mean_l3949_394991


namespace sqrt_sum_reciprocal_l3949_394988

theorem sqrt_sum_reciprocal (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) :
  Real.sqrt x + 1 / Real.sqrt x = 2 * Real.sqrt 13 := by
  sorry

end sqrt_sum_reciprocal_l3949_394988


namespace calculation_proofs_l3949_394933

theorem calculation_proofs :
  (4800 / 125 = 38.4) ∧ (13 * 74 + 27 * 13 - 13 = 1300) := by
  sorry

end calculation_proofs_l3949_394933


namespace spade_calculation_l3949_394917

-- Define the ♠ operation
def spade (a b : ℝ) : ℝ := |a - b|

-- Theorem statement
theorem spade_calculation : (spade 5 (spade 3 10)) * (spade 2 4) = 4 := by sorry

end spade_calculation_l3949_394917


namespace rational_division_l3949_394911

theorem rational_division (x : ℚ) : (-2 : ℚ) / x = 8 → x = -1/4 := by
  sorry

end rational_division_l3949_394911


namespace polynomial_has_real_root_l3949_394928

/-- The polynomial function for which we want to prove the existence of a real root -/
def f (a x : ℝ) : ℝ := x^5 + a*x^4 - x^3 + a*x^2 - x + a

/-- Theorem stating that for all real 'a', the polynomial has at least one real root -/
theorem polynomial_has_real_root :
  ∀ a : ℝ, ∃ x : ℝ, f a x = 0 := by
  sorry

end polynomial_has_real_root_l3949_394928


namespace problem_solution_l3949_394935

theorem problem_solution (a b : ℝ) (h1 : a * b = 4) (h2 : 2 / a + 1 / b = 1.5) :
  a + 2 * b = 6 := by
sorry

end problem_solution_l3949_394935


namespace part_one_part_two_l3949_394974

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (x - 1) + Real.sqrt (2 - x)}
def B (a : ℝ) : Set ℝ := {y | ∃ x ≥ a, y = 2^x}

-- Part I
theorem part_one : 
  (Set.univ \ A) ∩ B 2 = Set.Ici 4 := by sorry

-- Part II
theorem part_two : 
  ∀ a : ℝ, (Set.univ \ A) ∪ B a = Set.univ ↔ a ≤ 0 := by sorry

end part_one_part_two_l3949_394974


namespace alarm_system_probability_l3949_394930

theorem alarm_system_probability (p : ℝ) (h1 : p = 0.4) : 
  1 - (1 - p)^2 = 0.64 := by
  sorry

end alarm_system_probability_l3949_394930


namespace perfect_square_prime_sum_l3949_394970

theorem perfect_square_prime_sum (x y : ℤ) : 
  (∃ k : ℤ, 2 * x * y = k^2) ∧ 
  (∃ p : ℕ, Nat.Prime p ∧ x^2 + y^2 = p) →
  ((x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2)) :=
by sorry

end perfect_square_prime_sum_l3949_394970


namespace same_color_probability_5_8_l3949_394992

/-- The probability of drawing two balls of the same color from a bag containing
    5 green balls and 8 white balls. -/
def same_color_probability (green : ℕ) (white : ℕ) : ℚ :=
  let total := green + white
  let prob_both_green := (green / total) * ((green - 1) / (total - 1))
  let prob_both_white := (white / total) * ((white - 1) / (total - 1))
  prob_both_green + prob_both_white

/-- Theorem stating that the probability of drawing two balls of the same color
    from a bag with 5 green balls and 8 white balls is 19/39. -/
theorem same_color_probability_5_8 :
  same_color_probability 5 8 = 19 / 39 := by
  sorry

end same_color_probability_5_8_l3949_394992


namespace roller_coaster_cost_proof_l3949_394996

/-- The cost of the Ferris wheel in tickets -/
def ferris_wheel_cost : ℕ := 6

/-- The cost of the log ride in tickets -/
def log_ride_cost : ℕ := 7

/-- The number of tickets Antonieta initially has -/
def initial_tickets : ℕ := 2

/-- The number of additional tickets Antonieta needs to buy -/
def additional_tickets : ℕ := 16

/-- The cost of the roller coaster in tickets -/
def roller_coaster_cost : ℕ := 5

theorem roller_coaster_cost_proof :
  roller_coaster_cost = 
    (initial_tickets + additional_tickets) - (ferris_wheel_cost + log_ride_cost) :=
by sorry

end roller_coaster_cost_proof_l3949_394996


namespace normal_binomial_properties_l3949_394985

/-- A random variable with normal distribution -/
structure NormalRV where
  μ : ℝ
  σ : ℝ

/-- A random variable with binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ

/-- The probability that X is less than or equal to x -/
noncomputable def P (X : NormalRV) (x : ℝ) : ℝ := sorry

/-- The expected value of a normal random variable -/
noncomputable def E_normal (X : NormalRV) : ℝ := X.μ

/-- The expected value of a binomial random variable -/
noncomputable def E_binomial (Y : BinomialRV) : ℝ := Y.n * Y.p

/-- The variance of a binomial random variable -/
noncomputable def D_binomial (Y : BinomialRV) : ℝ := Y.n * Y.p * (1 - Y.p)

/-- The main theorem -/
theorem normal_binomial_properties (X : NormalRV) (Y : BinomialRV) 
    (h1 : P X 2 = 0.5)
    (h2 : E_binomial Y = E_normal X)
    (h3 : Y.n = 3) :
  X.μ = 2 ∧ Y.p = 2/3 ∧ 9 * D_binomial Y = 6 := by
  sorry

end normal_binomial_properties_l3949_394985


namespace complex_product_real_condition_l3949_394967

theorem complex_product_real_condition (a b c d : ℝ) :
  (Complex.mk a b * Complex.mk c d).im = 0 ↔ a * d + b * c = 0 := by sorry

end complex_product_real_condition_l3949_394967


namespace calculate_expression_l3949_394960

theorem calculate_expression : 15 * 35 + 45 * 15 = 1200 := by
  sorry

end calculate_expression_l3949_394960


namespace eleventh_term_value_l3949_394922

/-- An arithmetic progression is a sequence where the difference between
    consecutive terms is constant. -/
def ArithmeticProgression (a : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ (n : ℕ), a (n + 1) = a n + d

/-- The theorem states that for an arithmetic progression satisfying
    certain conditions, the 11th term is 109. -/
theorem eleventh_term_value
    (a : ℕ → ℝ)
    (h_ap : ArithmeticProgression a)
    (h_sum1 : a 4 + a 7 + a 10 = 207)
    (h_sum2 : a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 = 553) :
    a 11 = 109 := by
  sorry


end eleventh_term_value_l3949_394922


namespace vasims_share_l3949_394956

/-- Proves that Vasim's share is 1500 given the specified conditions -/
theorem vasims_share (total : ℕ) (faruk vasim ranjith : ℕ) : 
  faruk + vasim + ranjith = total →
  3 * faruk = 3 * vasim →
  3 * faruk = 3 * vasim ∧ 3 * vasim = 7 * ranjith →
  ranjith - faruk = 2000 →
  vasim = 1500 := by
sorry

end vasims_share_l3949_394956


namespace roots_of_quadratic_l3949_394934

theorem roots_of_quadratic (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  (∀ x : ℝ, x^2 - (a + b + c) * x + (a * b + b * c + c * a) = 0 ↔ x = a ∨ x = b ∨ x = c) :=
sorry

end roots_of_quadratic_l3949_394934


namespace rectangle_area_proof_l3949_394914

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- The original rectangle --/
def original : Rectangle := { length := 5, width := 7 }

/-- The rectangle after shortening one side --/
def shortened : Rectangle := { length := 3, width := 7 }

theorem rectangle_area_proof :
  area original = 35 ∧
  area shortened = 21 →
  area { length := 5, width := 5 } = 25 := by
  sorry

end rectangle_area_proof_l3949_394914


namespace hypotenuse_squared_of_complex_zeros_l3949_394929

/-- Given complex numbers u, v, and w that are zeros of a cubic polynomial
    and form a right triangle in the complex plane, if the sum of their
    squared magnitudes is 400, then the square of the hypotenuse of the
    triangle is 720. -/
theorem hypotenuse_squared_of_complex_zeros (u v w : ℂ) (s t : ℂ) :
  (u^3 + s*u + t = 0) →
  (v^3 + s*v + t = 0) →
  (w^3 + s*w + t = 0) →
  (Complex.abs u)^2 + (Complex.abs v)^2 + (Complex.abs w)^2 = 400 →
  ∃ (a b : ℝ), a^2 + b^2 = (Complex.abs (u - v))^2 ∧
                a^2 + b^2 = (Complex.abs (v - w))^2 ∧
                a * b = (Complex.abs (u - w))^2 →
  (Complex.abs (u - w))^2 = 720 :=
by sorry

end hypotenuse_squared_of_complex_zeros_l3949_394929


namespace total_amount_theorem_l3949_394904

/-- Represents the types of books in the collection -/
inductive BookType
  | Novel
  | Biography
  | ScienceBook

/-- Calculates the total amount received from book sales -/
def calculateTotalAmount (totalBooks : ℕ) 
                         (soldPercentages : BookType → ℚ)
                         (prices : BookType → ℕ)
                         (remainingBooks : BookType → ℕ) : ℕ :=
  sorry

/-- The main theorem stating the total amount received from book sales -/
theorem total_amount_theorem (totalBooks : ℕ)
                             (soldPercentages : BookType → ℚ)
                             (prices : BookType → ℕ)
                             (remainingBooks : BookType → ℕ) : 
  totalBooks = 300 ∧
  soldPercentages BookType.Novel = 3/5 ∧
  soldPercentages BookType.Biography = 2/3 ∧
  soldPercentages BookType.ScienceBook = 7/10 ∧
  prices BookType.Novel = 4 ∧
  prices BookType.Biography = 7 ∧
  prices BookType.ScienceBook = 6 ∧
  remainingBooks BookType.Novel = 30 ∧
  remainingBooks BookType.Biography = 35 ∧
  remainingBooks BookType.ScienceBook = 25 →
  calculateTotalAmount totalBooks soldPercentages prices remainingBooks = 1018 :=
by
  sorry

end total_amount_theorem_l3949_394904


namespace problem_1_problem_2_l3949_394990

-- Problem 1
theorem problem_1 (x : ℝ) : (-2 * x^2)^3 + 4 * x^3 * x^3 = -4 * x^6 := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) : (3 * x^2 - x + 1) * (-4 * x) = -12 * x^3 + 4 * x^2 - 4 * x := by
  sorry

end problem_1_problem_2_l3949_394990


namespace caroling_boys_count_l3949_394952

/-- The number of boys who received 1 orange each -/
def boys_with_one_orange : ℕ := 2

/-- The number of boys who received 2 oranges each -/
def boys_with_two_oranges : ℕ := 4

/-- The number of boys who received 4 oranges -/
def boys_with_four_oranges : ℕ := 1

/-- The number of oranges received by boys with known names -/
def oranges_known_boys : ℕ := boys_with_one_orange + 2 * boys_with_two_oranges + 4 * boys_with_four_oranges

/-- The total number of oranges received by all boys -/
def total_oranges : ℕ := 23

/-- The number of oranges each of the other boys received -/
def oranges_per_other_boy : ℕ := 3

theorem caroling_boys_count : ∃ (n : ℕ), 
  n = boys_with_one_orange + boys_with_two_oranges + boys_with_four_oranges + 
      (total_oranges - oranges_known_boys) / oranges_per_other_boy ∧ 
  n = 10 := by
  sorry

end caroling_boys_count_l3949_394952


namespace cone_volume_l3949_394905

/-- Given a cone circumscribed by a sphere, proves that the volume of the cone is 3π -/
theorem cone_volume (r l h : ℝ) : 
  r > 0 → l > 0 → h > 0 →
  (π * r * l = 2 * π * r^2) →  -- lateral area is twice base area
  (4 * π * (2^2) = 16 * π) →   -- surface area of circumscribing sphere is 16π
  (1/3) * π * r^2 * h = 3 * π := by
  sorry

end cone_volume_l3949_394905
