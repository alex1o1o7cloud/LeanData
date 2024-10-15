import Mathlib

namespace NUMINAMATH_CALUDE_min_value_of_f_l1288_128886

def f (x m : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + m

theorem min_value_of_f (m : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f y m ≤ f x m) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ 3) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ f y m) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = -37) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1288_128886


namespace NUMINAMATH_CALUDE_two_prime_factors_phi_tau_equality_l1288_128809

/-- Euler's totient function -/
def phi (n : ℕ) : ℕ := sorry

/-- Number of positive divisors function -/
def tau (n : ℕ) : ℕ := sorry

/-- Predicate to check if a number has exactly two distinct prime factors -/
def has_two_distinct_prime_factors (n : ℕ) : Prop := sorry

/-- Main theorem -/
theorem two_prime_factors_phi_tau_equality (n : ℕ) :
  has_two_distinct_prime_factors n ∧ phi (tau n) = tau (phi n) ↔
  ∃ (t r : ℕ), r.Prime ∧ t > 0 ∧ n = 2^(t-1) * 3^(r-1) :=
sorry

end NUMINAMATH_CALUDE_two_prime_factors_phi_tau_equality_l1288_128809


namespace NUMINAMATH_CALUDE_linear_equation_values_l1288_128808

/-- Given that x^(a-2) - 2y^(a-b+5) = 1 is a linear equation in x and y, prove that a = 3 and b = 7 -/
theorem linear_equation_values (a b : ℤ) : 
  (∀ x y : ℝ, ∃ m n c : ℝ, x^(a-2) - 2*y^(a-b+5) = m*x + n*y + c) → 
  a = 3 ∧ b = 7 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_values_l1288_128808


namespace NUMINAMATH_CALUDE_quadratic_fit_coefficient_l1288_128816

/-- Given three points (1, y₁), (2, y₂), and (3, y₃), the coefficient 'a' of the 
    quadratic equation y = ax² + bx + c that best fits these points is equal to 
    (y₃ - 2y₂ + y₁) / 2. -/
theorem quadratic_fit_coefficient (y₁ y₂ y₃ : ℝ) : 
  ∃ (a b c : ℝ), 
    (a * 1^2 + b * 1 + c = y₁) ∧ 
    (a * 2^2 + b * 2 + c = y₂) ∧ 
    (a * 3^2 + b * 3 + c = y₃) ∧ 
    (a = (y₃ - 2 * y₂ + y₁) / 2) := by
  sorry


end NUMINAMATH_CALUDE_quadratic_fit_coefficient_l1288_128816


namespace NUMINAMATH_CALUDE_cherries_theorem_l1288_128896

def cherries_problem (initial : ℕ) (eaten : ℕ) : ℕ :=
  let remaining := initial - eaten
  let given_away := remaining / 2
  remaining - given_away

theorem cherries_theorem :
  cherries_problem 2450 1625 = 413 := by
  sorry

end NUMINAMATH_CALUDE_cherries_theorem_l1288_128896


namespace NUMINAMATH_CALUDE_total_food_consumption_l1288_128876

/-- The amount of food needed per soldier per day on the first side -/
def food_per_soldier_first : ℕ := 10

/-- The amount of food needed per soldier per day on the second side -/
def food_per_soldier_second : ℕ := food_per_soldier_first - 2

/-- The number of soldiers on the first side -/
def soldiers_first : ℕ := 4000

/-- The number of soldiers on the second side -/
def soldiers_second : ℕ := soldiers_first - 500

/-- The total amount of food consumed by both sides per day -/
def total_food : ℕ := soldiers_first * food_per_soldier_first + soldiers_second * food_per_soldier_second

theorem total_food_consumption :
  total_food = 68000 := by sorry

end NUMINAMATH_CALUDE_total_food_consumption_l1288_128876


namespace NUMINAMATH_CALUDE_binomial_expansion_special_case_l1288_128852

theorem binomial_expansion_special_case : 
  98^3 + 3*(98^2)*2 + 3*98*(2^2) + 2^3 = 1000000 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_special_case_l1288_128852


namespace NUMINAMATH_CALUDE_geometric_progression_solution_l1288_128827

theorem geometric_progression_solution :
  ∀ (a b c d : ℝ),
  (∃ (q : ℝ), b = a * q ∧ c = a * q^2 ∧ d = a * q^3) →
  a + d = -49 →
  b + c = 14 →
  ((a = 7 ∧ b = -14 ∧ c = 28 ∧ d = -56) ∨
   (a = -56 ∧ b = 28 ∧ c = -14 ∧ d = 7)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_solution_l1288_128827


namespace NUMINAMATH_CALUDE_smallest_number_l1288_128888

theorem smallest_number (a b c d : ℚ) (ha : a = 1) (hb : b = -2) (hc : c = 0) (hd : d = -5/2) :
  d < b ∧ b < c ∧ c < a := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l1288_128888


namespace NUMINAMATH_CALUDE_range_of_a_l1288_128842

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x : ℝ, x^2 + 2*a*x + (2-a) = 0) → 
  a ≤ -2 ∨ a = 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1288_128842


namespace NUMINAMATH_CALUDE_aquarium_trainers_l1288_128814

/-- The number of trainers required to equally split the total training hours for all dolphins -/
def number_of_trainers (num_dolphins : ℕ) (hours_per_dolphin : ℕ) (hours_per_trainer : ℕ) : ℕ :=
  (num_dolphins * hours_per_dolphin) / hours_per_trainer

theorem aquarium_trainers :
  number_of_trainers 4 3 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_aquarium_trainers_l1288_128814


namespace NUMINAMATH_CALUDE_total_turnips_l1288_128820

theorem total_turnips (melanie_turnips benny_turnips : ℕ) 
  (h1 : melanie_turnips = 139) 
  (h2 : benny_turnips = 113) : 
  melanie_turnips + benny_turnips = 252 := by
  sorry

end NUMINAMATH_CALUDE_total_turnips_l1288_128820


namespace NUMINAMATH_CALUDE_problem_statement_l1288_128806

theorem problem_statement (a : ℝ) (h : (a + 1/a)^2 = 5) : a^4 + 1/a^4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1288_128806


namespace NUMINAMATH_CALUDE_bertha_family_childless_count_l1288_128891

/-- Represents a family tree with two generations -/
structure FamilyTree where
  daughters : ℕ
  granddaughters : ℕ

/-- Bertha's family tree -/
def berthas_family : FamilyTree := { daughters := 10, granddaughters := 32 }

/-- The number of Bertha's daughters who have children -/
def daughters_with_children : ℕ := 8

/-- The number of daughters each child-bearing daughter has -/
def granddaughters_per_daughter : ℕ := 4

theorem bertha_family_childless_count :
  berthas_family.daughters + berthas_family.granddaughters - daughters_with_children = 34 :=
by sorry

end NUMINAMATH_CALUDE_bertha_family_childless_count_l1288_128891


namespace NUMINAMATH_CALUDE_equal_areas_of_equal_ratios_l1288_128898

noncomputable def curvilinearTrapezoidArea (a b : ℝ) : ℝ := ∫ x in a..b, (1 / x)

theorem equal_areas_of_equal_ratios (a₁ b₁ a₂ b₂ : ℝ) 
  (ha₁ : 0 < a₁) (hb₁ : a₁ < b₁)
  (ha₂ : 0 < a₂) (hb₂ : a₂ < b₂)
  (h_ratio : b₁ / a₁ = b₂ / a₂) :
  curvilinearTrapezoidArea a₁ b₁ = curvilinearTrapezoidArea a₂ b₂ := by
  sorry

end NUMINAMATH_CALUDE_equal_areas_of_equal_ratios_l1288_128898


namespace NUMINAMATH_CALUDE_lizard_wrinkle_eye_ratio_l1288_128873

theorem lizard_wrinkle_eye_ratio :
  ∀ (W : ℕ) (S : ℕ),
    S = 7 * W →
    3 = S + W - 69 →
    (W : ℚ) / 3 = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_lizard_wrinkle_eye_ratio_l1288_128873


namespace NUMINAMATH_CALUDE_complex_power_2018_l1288_128825

theorem complex_power_2018 : (((1 - Complex.I) / (1 + Complex.I)) ^ 2018 : ℂ) = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_2018_l1288_128825


namespace NUMINAMATH_CALUDE_sin_difference_l1288_128829

theorem sin_difference (A B : ℝ) : 
  Real.sin (A - B) = Real.sin A * Real.cos B - Real.cos A * Real.sin B := by
  sorry

end NUMINAMATH_CALUDE_sin_difference_l1288_128829


namespace NUMINAMATH_CALUDE_cylinder_height_equals_sphere_radius_l1288_128878

theorem cylinder_height_equals_sphere_radius 
  (r_sphere : ℝ) 
  (d_cylinder : ℝ) 
  (h_cylinder : ℝ) :
  r_sphere = 3 →
  d_cylinder = 6 →
  2 * π * (d_cylinder / 2) * h_cylinder = 4 * π * r_sphere^2 →
  h_cylinder = 6 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_height_equals_sphere_radius_l1288_128878


namespace NUMINAMATH_CALUDE_magnitude_of_mn_l1288_128813

/-- Given vectors and conditions, prove the magnitude of MN --/
theorem magnitude_of_mn (a b c : ℝ × ℝ) (x y : ℝ) : 
  a = (2, -1) →
  b = (x, -2) →
  c = (3, y) →
  ∃ (k : ℝ), a = k • b →  -- a is parallel to b
  (a + b) • (b - c) = 0 →  -- (a + b) is perpendicular to (b - c)
  ‖(y - x, x - y)‖ = 8 * Real.sqrt 2 := by
  sorry

#check magnitude_of_mn

end NUMINAMATH_CALUDE_magnitude_of_mn_l1288_128813


namespace NUMINAMATH_CALUDE_bella_steps_l1288_128859

/-- The distance between Bella's and Ella's houses in feet -/
def distance : ℝ := 10560

/-- Bella's step length in feet -/
def step_length : ℝ := 2.5

/-- The ratio of Ella's speed to Bella's speed -/
def speed_ratio : ℝ := 5

/-- Calculates the number of steps Bella takes before meeting Ella -/
def steps_taken : ℕ := sorry

/-- Theorem stating that Bella takes 704 steps before meeting Ella -/
theorem bella_steps : steps_taken = 704 := by sorry

end NUMINAMATH_CALUDE_bella_steps_l1288_128859


namespace NUMINAMATH_CALUDE_sandwich_cost_l1288_128895

theorem sandwich_cost (sandwich_cost juice_cost milk_cost : ℝ) :
  juice_cost = 2 * sandwich_cost →
  milk_cost = 0.75 * (sandwich_cost + juice_cost) →
  sandwich_cost + juice_cost + milk_cost = 21 →
  sandwich_cost = 4 := by
sorry

end NUMINAMATH_CALUDE_sandwich_cost_l1288_128895


namespace NUMINAMATH_CALUDE_range_of_x_l1288_128861

theorem range_of_x (a b c : ℝ) (h : a^2 + 2*b^2 + 3*c^2 = 6) :
  (∃ x : ℝ, ∀ y : ℝ, (∃ a' b' c' : ℝ, a'^2 + 2*b'^2 + 3*c'^2 = 6 ∧ a' + 2*b' + 3*c' > |y + 1|) ↔ -7 < y ∧ y < 5) :=
sorry

end NUMINAMATH_CALUDE_range_of_x_l1288_128861


namespace NUMINAMATH_CALUDE_no_x_squared_term_l1288_128883

theorem no_x_squared_term (a : ℚ) : 
  (∀ x, (-9 * x^3 + (-6*a - 4) * x^2 - 3*x) = (-9 * x^3 - 3*x)) ↔ a = -2/3 :=
by sorry

end NUMINAMATH_CALUDE_no_x_squared_term_l1288_128883


namespace NUMINAMATH_CALUDE_students_in_one_language_class_l1288_128818

theorem students_in_one_language_class 
  (french_class : ℕ) 
  (spanish_class : ℕ) 
  (both_classes : ℕ) 
  (h1 : french_class = 21) 
  (h2 : spanish_class = 21) 
  (h3 : both_classes = 6) :
  french_class + spanish_class - 2 * both_classes = 36 := by
  sorry

end NUMINAMATH_CALUDE_students_in_one_language_class_l1288_128818


namespace NUMINAMATH_CALUDE_alcohol_quantity_l1288_128843

/-- Proves that the quantity of alcohol is 16 liters given the initial and final ratios -/
theorem alcohol_quantity (initial_alcohol : ℚ) (initial_water : ℚ) (final_water : ℚ) :
  initial_alcohol / initial_water = 4 / 3 →
  initial_alcohol / (initial_water + 8) = 4 / 5 →
  initial_alcohol = 16 := by
sorry


end NUMINAMATH_CALUDE_alcohol_quantity_l1288_128843


namespace NUMINAMATH_CALUDE_paragraph_writing_time_l1288_128824

/-- Represents the time in minutes for various writing assignments -/
structure WritingTimes where
  short_answer : ℕ  -- Time for one short-answer question
  essay : ℕ         -- Time for one essay
  total : ℕ         -- Total homework time
  paragraph : ℕ     -- Time for one paragraph (to be proved)

/-- Represents the number of assignments -/
structure AssignmentCounts where
  essays : ℕ
  paragraphs : ℕ
  short_answers : ℕ

theorem paragraph_writing_time 
  (wt : WritingTimes) 
  (ac : AssignmentCounts) 
  (h1 : wt.short_answer = 3)
  (h2 : wt.essay = 60)
  (h3 : wt.total = 4 * 60)
  (h4 : ac.essays = 2)
  (h5 : ac.paragraphs = 5)
  (h6 : ac.short_answers = 15)
  (h7 : wt.total = ac.essays * wt.essay + ac.paragraphs * wt.paragraph + ac.short_answers * wt.short_answer) :
  wt.paragraph = 15 := by
  sorry

end NUMINAMATH_CALUDE_paragraph_writing_time_l1288_128824


namespace NUMINAMATH_CALUDE_smallest_earring_collection_l1288_128838

theorem smallest_earring_collection (M : ℕ) : 
  M > 2 ∧ 
  M % 7 = 2 ∧ 
  M % 11 = 2 ∧ 
  M % 13 = 2 → 
  M ≥ 1003 :=
by sorry

end NUMINAMATH_CALUDE_smallest_earring_collection_l1288_128838


namespace NUMINAMATH_CALUDE_equation_solutions_l1288_128897

theorem equation_solutions (a b c d : ℤ) (hab : a ≠ b) :
  let f : ℤ × ℤ → ℤ := λ (x, y) ↦ (x + a * y + c) * (x + b * y + d)
  (∃ S : Finset (ℤ × ℤ), (∀ p ∈ S, f p = 2) ∧ S.card ≤ 4) ∧
  ((|a - b| = 1 ∨ |a - b| = 2) → (c - d) % 2 ≠ 0 →
    ∃ S : Finset (ℤ × ℤ), (∀ p ∈ S, f p = 2) ∧ S.card = 4) := by
  sorry

#check equation_solutions

end NUMINAMATH_CALUDE_equation_solutions_l1288_128897


namespace NUMINAMATH_CALUDE_largest_angle_is_right_angle_l1288_128828

/-- Given a triangle with sides a, b, and c, if its area is (a+b+c)(a+b-c)/4, 
    then its largest angle is 90°. -/
theorem largest_angle_is_right_angle 
  (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_area : (a + b + c) * (a + b - c) / 4 = (a + b + c) * (a + b - c) / 4) :
  ∃ θ : ℝ, θ = Real.pi / 2 ∧ θ = max (Real.arccos ((b^2 + c^2 - a^2) / (2*b*c))) 
                                    (max (Real.arccos ((a^2 + c^2 - b^2) / (2*a*c)))
                                         (Real.arccos ((a^2 + b^2 - c^2) / (2*a*b)))) :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_is_right_angle_l1288_128828


namespace NUMINAMATH_CALUDE_rectangular_prism_cut_out_l1288_128894

theorem rectangular_prism_cut_out (x y : ℤ) : 
  (15 * 5 * 4 - y * 5 * x = 120) → 
  (0 < x) → 
  (x < 4) → 
  (0 < y) → 
  (y < 15) → 
  (x = 3 ∧ y = 12) := by
sorry

end NUMINAMATH_CALUDE_rectangular_prism_cut_out_l1288_128894


namespace NUMINAMATH_CALUDE_other_asymptote_equation_l1288_128858

/-- Represents a hyperbola -/
structure Hyperbola where
  /-- One of the asymptotes of the hyperbola -/
  asymptote1 : ℝ → ℝ
  /-- x-coordinate of the foci -/
  foci_x : ℝ

/-- Theorem: Given a hyperbola with one asymptote y = 2x and foci x-coordinate 4,
    the other asymptote has equation y = -0.5x + 10 -/
theorem other_asymptote_equation (h : Hyperbola) 
    (h1 : h.asymptote1 = fun x => 2 * x) 
    (h2 : h.foci_x = 4) : 
    ∃ asymptote2 : ℝ → ℝ, asymptote2 = fun x => -0.5 * x + 10 := by
  sorry

end NUMINAMATH_CALUDE_other_asymptote_equation_l1288_128858


namespace NUMINAMATH_CALUDE_joyce_apples_to_larry_l1288_128868

/-- The number of apples Joyce gave to Larry -/
def apples_given_to_larry (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

theorem joyce_apples_to_larry : 
  apples_given_to_larry 75 23 = 52 := by
  sorry

end NUMINAMATH_CALUDE_joyce_apples_to_larry_l1288_128868


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1288_128847

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 + 5*x₁ - 2 = 0) → (x₂^2 + 5*x₂ - 2 = 0) → (x₁ + x₂ = -5) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1288_128847


namespace NUMINAMATH_CALUDE_total_matches_is_120_l1288_128849

/-- Represents the number of factions in the game -/
def num_factions : ℕ := 3

/-- Represents the number of players in each team -/
def team_size : ℕ := 4

/-- Represents the total number of players -/
def total_players : ℕ := 8

/-- Calculates the number of ways to form a team of given size from available factions -/
def ways_to_form_team (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Calculates the total number of distinct matches possible -/
def total_distinct_matches : ℕ :=
  let ways_one_team := ways_to_form_team team_size num_factions
  ways_one_team + Nat.choose ways_one_team 2

/-- Theorem stating that the total number of distinct matches is 120 -/
theorem total_matches_is_120 : total_distinct_matches = 120 := by
  sorry

end NUMINAMATH_CALUDE_total_matches_is_120_l1288_128849


namespace NUMINAMATH_CALUDE_log_inequality_conditions_l1288_128800

/-- The set of positive real numbers excluding 1 -/
def S : Set ℝ := {x : ℝ | x > 0 ∧ x ≠ 1}

/-- The theorem stating the conditions for the logarithmic inequality -/
theorem log_inequality_conditions (a b : ℝ) :
  a > 0 → b > 0 → a ≠ 1 →
  (Real.log b / Real.log a < Real.log (b + 1) / Real.log (a + 1)) ↔
  ((b = 1 ∧ a ∈ S) ∨
   (a > b ∧ b > 1) ∨
   (b > 1 ∧ 1 > a) ∨
   (a < b ∧ b < 1) ∨
   (b < 1 ∧ 1 < a)) :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_conditions_l1288_128800


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_l1288_128892

theorem least_three_digit_multiple : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0) ∧
  (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ m % 3 = 0 ∧ m % 5 = 0 ∧ m % 7 = 0 → m ≥ n) ∧
  n = 105 :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_l1288_128892


namespace NUMINAMATH_CALUDE_square_points_sum_l1288_128885

/-- Square with side length 1000 -/
structure Square :=
  (side : ℝ)
  (is_1000 : side = 1000)

/-- Points on the side of the square -/
structure PointOnSide (S : Square) :=
  (pos : ℝ)
  (on_side : 0 ≤ pos ∧ pos ≤ S.side)

/-- Condition that E is between A and F -/
def between (A E F : ℝ) : Prop := A ≤ E ∧ E ≤ F

/-- Angle in degrees -/
def angle (θ : ℝ) := 0 ≤ θ ∧ θ < 360

/-- Distance between two points on a line -/
def distance (x y : ℝ) := |x - y|

/-- Representation of BF as p + q√r -/
structure IrrationalForm :=
  (p q r : ℕ)
  (r_not_square : ∀ (n : ℕ), n > 1 → r % (n^2) ≠ 0)

theorem square_points_sum (S : Square) 
  (E F : PointOnSide S)
  (AE_less_BF : E.pos < S.side - F.pos)
  (E_between_A_F : between 0 E.pos F.pos)
  (angle_EOF : angle 30)
  (EF_length : distance E.pos F.pos = 500)
  (BF_form : IrrationalForm)
  (BF_value : S.side - F.pos = BF_form.p + BF_form.q * Real.sqrt BF_form.r) :
  BF_form.p + BF_form.q + BF_form.r = 253 := by
  sorry

end NUMINAMATH_CALUDE_square_points_sum_l1288_128885


namespace NUMINAMATH_CALUDE_shaded_area_percentage_l1288_128802

/-- Given two congruent squares ABCD and EFGH with side length 20 units that overlap
    to form a 20 by 35 rectangle AEGD, prove that 14% of AEGD's area is shaded. -/
theorem shaded_area_percentage (side_length : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) :
  side_length = 20 →
  rectangle_width = 20 →
  rectangle_length = 35 →
  (((2 * side_length - rectangle_length) * side_length) / (rectangle_width * rectangle_length)) * 100 = 14 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_percentage_l1288_128802


namespace NUMINAMATH_CALUDE_cube_surface_area_l1288_128853

/-- The surface area of a cube with edge length 6 cm is 216 square centimeters. -/
theorem cube_surface_area : 
  let edge_length : ℝ := 6
  let surface_area := 6 * edge_length^2
  surface_area = 216 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l1288_128853


namespace NUMINAMATH_CALUDE_percentage_relation_l1288_128819

theorem percentage_relation (x a b : ℝ) (h1 : a = 0.06 * x) (h2 : b = 0.18 * x) :
  a / b * 100 = 100 / 3 :=
by sorry

end NUMINAMATH_CALUDE_percentage_relation_l1288_128819


namespace NUMINAMATH_CALUDE_complex_subtraction_l1288_128854

theorem complex_subtraction : (4 - 3*I) - (7 - 5*I) = -3 + 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_l1288_128854


namespace NUMINAMATH_CALUDE_givenCurve_is_parabola_l1288_128867

/-- A curve in 2D space represented by parametric equations -/
structure ParametricCurve where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Definition of a parabola in standard form -/
def IsParabola (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given parametric curve -/
def givenCurve : ParametricCurve where
  x := λ t => t
  y := λ t => t^2 + 1

/-- Theorem stating that the given curve is a parabola -/
theorem givenCurve_is_parabola :
  IsParabola (λ x => givenCurve.y (givenCurve.x⁻¹ x)) :=
sorry

end NUMINAMATH_CALUDE_givenCurve_is_parabola_l1288_128867


namespace NUMINAMATH_CALUDE_hypotenuse_segment_ratio_l1288_128835

/-- A right triangle with leg lengths in ratio 3:4 -/
structure RightTriangle where
  a : ℝ  -- length of first leg
  b : ℝ  -- length of second leg
  h : ℝ  -- ratio of legs is 3:4
  leg_ratio : b = (4/3) * a

/-- The segments of the hypotenuse created by the altitude -/
structure HypotenuseSegments where
  x : ℝ  -- length of first segment
  y : ℝ  -- length of second segment

/-- Theorem: The ratio of hypotenuse segments is 21:16 -/
theorem hypotenuse_segment_ratio (t : RightTriangle) (s : HypotenuseSegments) :
  s.y / s.x = 21 / 16 :=
sorry

end NUMINAMATH_CALUDE_hypotenuse_segment_ratio_l1288_128835


namespace NUMINAMATH_CALUDE_ellipse_focal_property_l1288_128833

/-- The equation of the ellipse -/
def ellipse_eq (x y : ℝ) : Prop := x^2 / 16 + y^2 / 9 = 1

/-- The foci of the ellipse -/
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

/-- Points A and B on the ellipse -/
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

/-- The distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem ellipse_focal_property :
  ellipse_eq A.1 A.2 ∧ 
  ellipse_eq B.1 B.2 ∧ 
  (∃ (t : ℝ), A = F₂ + t • (B - F₂) ∨ B = F₂ + t • (A - F₂)) ∧
  distance A B = 5 →
  distance A F₁ + distance B F₁ = 11 := by sorry

end NUMINAMATH_CALUDE_ellipse_focal_property_l1288_128833


namespace NUMINAMATH_CALUDE_opposite_absolute_values_l1288_128881

theorem opposite_absolute_values (a b : ℝ) :
  (|a - 1| + |b - 2| = 0) → (a + b = 3) := by
  sorry

end NUMINAMATH_CALUDE_opposite_absolute_values_l1288_128881


namespace NUMINAMATH_CALUDE_tangent_line_value_l1288_128855

/-- The value of a when the line 2x - y + 1 = 0 is tangent to the curve y = ae^x + x -/
theorem tangent_line_value (a : ℝ) : 
  (∃ x y : ℝ, 2*x - y + 1 = 0 ∧ y = a*(Real.exp x) + x ∧ 
    (∀ h : ℝ, h ≠ 0 → (a*(Real.exp (x + h)) + (x + h) - y) / h ≠ 2)) → 
  a = 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_value_l1288_128855


namespace NUMINAMATH_CALUDE_inequality_proof_l1288_128877

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_given : (3 : ℝ) / (a * b * c) ≥ a + b + c) :
  1 / a + 1 / b + 1 / c ≥ a + b + c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1288_128877


namespace NUMINAMATH_CALUDE_environmental_policy_survey_l1288_128839

theorem environmental_policy_survey (group_a_size : ℕ) (group_b_size : ℕ) 
  (group_a_favor_percent : ℚ) (group_b_favor_percent : ℚ) : 
  group_a_size = 200 →
  group_b_size = 800 →
  group_a_favor_percent = 70 / 100 →
  group_b_favor_percent = 75 / 100 →
  (group_a_size * group_a_favor_percent + group_b_size * group_b_favor_percent) / 
  (group_a_size + group_b_size) = 74 / 100 := by
  sorry

end NUMINAMATH_CALUDE_environmental_policy_survey_l1288_128839


namespace NUMINAMATH_CALUDE_ted_peeling_time_l1288_128857

/-- The time it takes Julie to peel potatoes individually (in hours) -/
def julie_time : ℝ := 10

/-- The time Julie and Ted work together (in hours) -/
def together_time : ℝ := 4

/-- The time it takes Julie to complete the task after Ted leaves (in hours) -/
def julie_remaining_time : ℝ := 0.9999999999999998

/-- The time it takes Ted to peel potatoes individually (in hours) -/
def ted_time : ℝ := 8

/-- Theorem stating that given the conditions, Ted's individual time to peel potatoes is 8 hours -/
theorem ted_peeling_time :
  (together_time * (1 / julie_time + 1 / ted_time)) + (julie_remaining_time * (1 / julie_time)) = 1 :=
sorry

end NUMINAMATH_CALUDE_ted_peeling_time_l1288_128857


namespace NUMINAMATH_CALUDE_pythagoras_students_l1288_128815

theorem pythagoras_students : ∃ n : ℕ, 
  n > 0 ∧ 
  (n / 2 : ℚ) + (n / 4 : ℚ) + (n / 7 : ℚ) + 3 = n ∧ 
  n = 28 := by
  sorry

end NUMINAMATH_CALUDE_pythagoras_students_l1288_128815


namespace NUMINAMATH_CALUDE_sqrt_3_expression_simplification_l1288_128810

theorem sqrt_3_expression_simplification :
  Real.sqrt 3 * (Real.sqrt 3 - 2) - Real.sqrt 12 / Real.sqrt 3 + |2 - Real.sqrt 3| = 3 - 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_expression_simplification_l1288_128810


namespace NUMINAMATH_CALUDE_degrees_to_radians_conversion_l1288_128865

theorem degrees_to_radians_conversion (deg : ℝ) (rad : ℝ) : 
  deg = 50 → rad = deg * (π / 180) → rad = 5 * π / 18 := by
  sorry

end NUMINAMATH_CALUDE_degrees_to_radians_conversion_l1288_128865


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1288_128874

theorem quadratic_inequality (x : ℝ) : x^2 - 36*x + 325 ≤ 9 ↔ 16 ≤ x ∧ x ≤ 20 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1288_128874


namespace NUMINAMATH_CALUDE_toad_frog_percentage_increase_l1288_128864

/-- Represents the number of bugs eaten by each animal -/
structure BugsEaten where
  gecko : ℕ
  lizard : ℕ
  frog : ℕ
  toad : ℕ

/-- Conditions from the problem -/
def garden_conditions (b : BugsEaten) : Prop :=
  b.gecko = 12 ∧
  b.lizard = b.gecko / 2 ∧
  b.frog = 3 * b.lizard ∧
  b.gecko + b.lizard + b.frog + b.toad = 63

/-- Calculate percentage increase -/
def percentage_increase (old_value new_value : ℕ) : ℚ :=
  (new_value - old_value : ℚ) / old_value * 100

/-- Theorem stating the percentage increase in bugs eaten by toad compared to frog -/
theorem toad_frog_percentage_increase (b : BugsEaten) 
  (h : garden_conditions b) : 
  percentage_increase b.frog b.toad = 50 := by
  sorry

end NUMINAMATH_CALUDE_toad_frog_percentage_increase_l1288_128864


namespace NUMINAMATH_CALUDE_smallest_number_l1288_128899

theorem smallest_number : ∀ (a b c d : ℚ), 
  a = -3 ∧ b = -2 ∧ c = 0 ∧ d = 1/3 → 
  a ≤ b ∧ a ≤ c ∧ a ≤ d := by sorry

end NUMINAMATH_CALUDE_smallest_number_l1288_128899


namespace NUMINAMATH_CALUDE_least_number_divisible_by_11_with_remainder_2_l1288_128804

def is_divisible_by_11 (n : ℕ) : Prop := ∃ k : ℕ, n = 11 * k

def leaves_remainder_2 (n : ℕ) (d : ℕ) : Prop := ∃ k : ℕ, n = d * k + 2

theorem least_number_divisible_by_11_with_remainder_2 : 
  (is_divisible_by_11 1262) ∧ 
  (∀ d : ℕ, 3 ≤ d → d ≤ 7 → leaves_remainder_2 1262 d) ∧
  (∀ m : ℕ, m < 1262 → 
    ¬(is_divisible_by_11 m ∧ 
      (∀ d : ℕ, 3 ≤ d → d ≤ 7 → leaves_remainder_2 m d))) :=
by sorry

end NUMINAMATH_CALUDE_least_number_divisible_by_11_with_remainder_2_l1288_128804


namespace NUMINAMATH_CALUDE_opposite_of_three_l1288_128841

-- Define the opposite function for real numbers
def opposite (x : ℝ) : ℝ := -x

-- State the theorem
theorem opposite_of_three : opposite 3 = -3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_three_l1288_128841


namespace NUMINAMATH_CALUDE_family_fruit_consumption_l1288_128801

/-- Represents the number of fruits in a box for each type of fruit -/
structure FruitBox where
  apples : ℕ := 14
  bananas : ℕ := 20
  oranges : ℕ := 12

/-- Represents the daily consumption of fruits for each family member -/
structure DailyConsumption where
  apples : ℕ := 2  -- Henry and his brother combined
  bananas : ℕ := 2 -- Henry's sister (on odd days)
  oranges : ℕ := 3 -- Father

/-- Represents the number of boxes for each type of fruit -/
structure FruitSupply where
  appleBoxes : ℕ := 3
  bananaBoxes : ℕ := 4
  orangeBoxes : ℕ := 5

/-- Calculates the maximum number of days the family can eat their preferred fruits together -/
def max_days_eating_fruits (box : FruitBox) (consumption : DailyConsumption) (supply : FruitSupply) : ℕ :=
  sorry

/-- Theorem stating that the family can eat their preferred fruits together for 20 days -/
theorem family_fruit_consumption 
  (box : FruitBox) 
  (consumption : DailyConsumption) 
  (supply : FruitSupply) 
  (orange_days : ℕ := 20) -- Oranges are only available for 20 days
  (h1 : box.apples = 14)
  (h2 : box.bananas = 20)
  (h3 : box.oranges = 12)
  (h4 : consumption.apples = 2)
  (h5 : consumption.bananas = 2)
  (h6 : consumption.oranges = 3)
  (h7 : supply.appleBoxes = 3)
  (h8 : supply.bananaBoxes = 4)
  (h9 : supply.orangeBoxes = 5) :
  max_days_eating_fruits box consumption supply = 20 :=
sorry

end NUMINAMATH_CALUDE_family_fruit_consumption_l1288_128801


namespace NUMINAMATH_CALUDE_line_y_axis_intersection_l1288_128870

/-- A line passing through two given points intersects the y-axis at a specific point -/
theorem line_y_axis_intersection (x₁ y₁ x₂ y₂ : ℝ) :
  x₁ = 4 ∧ y₁ = 20 ∧ x₂ = -6 ∧ y₂ = -2 →
  ∃ (y : ℝ), y = 11.2 ∧ 
    (y - y₁) / (0 - x₁) = (y₂ - y₁) / (x₂ - x₁) :=
by sorry

end NUMINAMATH_CALUDE_line_y_axis_intersection_l1288_128870


namespace NUMINAMATH_CALUDE_chemical_mixture_problem_l1288_128812

theorem chemical_mixture_problem (original_conc : ℝ) (final_conc : ℝ) (replaced_portion : ℝ) 
  (h1 : original_conc = 0.9)
  (h2 : final_conc = 0.4)
  (h3 : replaced_portion = 0.7142857142857143) :
  let replacement_conc := (final_conc - original_conc * (1 - replaced_portion)) / replaced_portion
  replacement_conc = 0.2 := by
sorry

end NUMINAMATH_CALUDE_chemical_mixture_problem_l1288_128812


namespace NUMINAMATH_CALUDE_loan_shark_fees_l1288_128882

/-- Calculates the total fees for a loan with a doubling weekly rate -/
def totalFees (loanAmount : ℝ) (initialRate : ℝ) (weeks : ℕ) : ℝ :=
  let weeklyFees := fun w => loanAmount * initialRate * (2 ^ w)
  (Finset.range weeks).sum weeklyFees

/-- Theorem stating that the total fees for a $100 loan at 5% initial rate for 2 weeks is $15 -/
theorem loan_shark_fees : totalFees 100 0.05 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_loan_shark_fees_l1288_128882


namespace NUMINAMATH_CALUDE_right_triangle_third_side_product_l1288_128884

theorem right_triangle_third_side_product (a b : ℝ) (ha : a = 5) (hb : b = 7) :
  (Real.sqrt (a^2 + b^2)) * (Real.sqrt (max a b)^2 - (min a b)^2) = Real.sqrt 1776 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_product_l1288_128884


namespace NUMINAMATH_CALUDE_prime_divisor_form_l1288_128821

theorem prime_divisor_form (n : ℕ) (q : ℕ) (h_prime : Nat.Prime q) (h_divides : q ∣ 2^(2^n) + 1) :
  ∃ x : ℤ, q = 2^(n + 1) * x + 1 :=
sorry

end NUMINAMATH_CALUDE_prime_divisor_form_l1288_128821


namespace NUMINAMATH_CALUDE_athlete_heartbeats_l1288_128860

/-- The number of heartbeats during a race -/
def heartbeats_during_race (heart_rate : ℕ) (pace : ℕ) (distance : ℕ) : ℕ :=
  heart_rate * pace * distance

/-- Proof that the athlete's heart beats 25200 times during the race -/
theorem athlete_heartbeats :
  heartbeats_during_race 140 6 30 = 25200 := by
  sorry

#eval heartbeats_during_race 140 6 30

end NUMINAMATH_CALUDE_athlete_heartbeats_l1288_128860


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l1288_128836

theorem sqrt_product_simplification (p : ℝ) (hp : p ≥ 0) :
  Real.sqrt (40 * p^2) * Real.sqrt (10 * p^3) * Real.sqrt (8 * p^2) = 40 * p^3 * Real.sqrt p :=
by sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l1288_128836


namespace NUMINAMATH_CALUDE_abs_sum_reciprocals_ge_two_l1288_128831

theorem abs_sum_reciprocals_ge_two (a b : ℝ) (h : a * b ≠ 0) :
  |a / b + b / a| ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_reciprocals_ge_two_l1288_128831


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l1288_128851

theorem tangent_line_to_circle (c : ℝ) : 
  (c > 0) → 
  (∀ x y : ℝ, x^2 + y^2 = 8 → (x + y = c → (x - 0)^2 + (y - 0)^2 = 8)) → 
  c = 4 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l1288_128851


namespace NUMINAMATH_CALUDE_nearest_integer_to_x_minus_y_l1288_128875

theorem nearest_integer_to_x_minus_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : |x| + y = 5) (h2 : |x| * y - x^3 = 0) : x - y = 5 := by
  sorry

end NUMINAMATH_CALUDE_nearest_integer_to_x_minus_y_l1288_128875


namespace NUMINAMATH_CALUDE_sadie_homework_problem_l1288_128846

/-- The total number of math homework problems Sadie has for the week. -/
def total_problems : ℕ := 140

/-- The number of solving linear equations problems Sadie has. -/
def linear_equations_problems : ℕ := 28

/-- Theorem stating that the total number of math homework problems is 140,
    given the conditions from the problem. -/
theorem sadie_homework_problem :
  (total_problems : ℝ) * 0.4 * 0.5 = linear_equations_problems :=
by sorry

end NUMINAMATH_CALUDE_sadie_homework_problem_l1288_128846


namespace NUMINAMATH_CALUDE_vacation_savings_proof_l1288_128817

/-- Calculates the amount to save per paycheck given a savings goal, time frame, and number of paychecks per month. -/
def amount_per_paycheck (savings_goal : ℚ) (months : ℕ) (paychecks_per_month : ℕ) : ℚ :=
  savings_goal / (months * paychecks_per_month)

/-- Proves that given a savings goal of $3,000.00 over 15 months with 2 paychecks per month, 
    the amount to save per paycheck is $100.00. -/
theorem vacation_savings_proof : 
  amount_per_paycheck 3000 15 2 = 100 := by
sorry

end NUMINAMATH_CALUDE_vacation_savings_proof_l1288_128817


namespace NUMINAMATH_CALUDE_country_club_members_l1288_128811

def initial_fee : ℕ := 4000
def monthly_cost : ℕ := 1000
def john_payment : ℕ := 32000

theorem country_club_members : 
  ∀ (F : ℕ), 
    (F + 1) * (initial_fee + 12 * monthly_cost) / 2 = john_payment → 
    F = 3 :=
by sorry

end NUMINAMATH_CALUDE_country_club_members_l1288_128811


namespace NUMINAMATH_CALUDE_special_triangle_b_value_l1288_128840

/-- Triangle ABC with sides a, b, c opposite angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Given conditions for the triangle -/
def SpecialTriangle (t : Triangle) : Prop :=
  t.a = 4 ∧ t.b + t.c = 6 ∧ t.b < t.c ∧ Real.cos t.A = 1/2

theorem special_triangle_b_value (t : Triangle) (h : SpecialTriangle t) : t.b = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_b_value_l1288_128840


namespace NUMINAMATH_CALUDE_airplane_passengers_l1288_128850

theorem airplane_passengers (total_passengers men : ℕ) 
  (h1 : total_passengers = 80)
  (h2 : men = 30)
  (h3 : ∃ women : ℕ, men = women ∧ men + women + (total_passengers - (men + women)) = total_passengers) :
  total_passengers - 2 * men = 20 := by
  sorry

end NUMINAMATH_CALUDE_airplane_passengers_l1288_128850


namespace NUMINAMATH_CALUDE_specific_project_time_l1288_128832

/-- A project requires workers to complete it. The number of workers and time can change during the project. -/
structure Project where
  initial_workers : ℕ
  initial_days : ℕ
  additional_workers : ℕ
  days_before_addition : ℕ

/-- Calculate the total time required to complete the project -/
def total_time (p : Project) : ℕ :=
  sorry

/-- The specific project described in the problem -/
def specific_project : Project :=
  { initial_workers := 10
  , initial_days := 15
  , additional_workers := 5
  , days_before_addition := 5 }

/-- Theorem stating that the total time for the specific project is 6 days -/
theorem specific_project_time : total_time specific_project = 6 :=
  sorry

end NUMINAMATH_CALUDE_specific_project_time_l1288_128832


namespace NUMINAMATH_CALUDE_log_of_negative_one_not_real_l1288_128826

/-- For b > 0 and b ≠ 1, log_b(-1) is not a real number -/
theorem log_of_negative_one_not_real (b : ℝ) (hb_pos : b > 0) (hb_ne_one : b ≠ 1) :
  ¬ ∃ (y : ℝ), b^y = -1 :=
sorry

end NUMINAMATH_CALUDE_log_of_negative_one_not_real_l1288_128826


namespace NUMINAMATH_CALUDE_problem_statement_l1288_128863

theorem problem_statement (x y : ℝ) (h : x^2 + y^2 - x*y = 1) : 
  (x + y ≥ -2) ∧ (x^2 + y^2 ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1288_128863


namespace NUMINAMATH_CALUDE_total_points_scored_l1288_128807

/-- Given a player who played 10.0 games and scored 12 points in each game,
    the total points scored is 120. -/
theorem total_points_scored (games : ℝ) (points_per_game : ℕ) : 
  games = 10.0 → points_per_game = 12 → games * (points_per_game : ℝ) = 120 := by
  sorry

end NUMINAMATH_CALUDE_total_points_scored_l1288_128807


namespace NUMINAMATH_CALUDE_rectangle_area_equals_perimeter_l1288_128845

theorem rectangle_area_equals_perimeter (b : ℝ) (h1 : b > 0) :
  let l := 3 * b
  let area := l * b
  let perimeter := 2 * (l + b)
  area = perimeter → b = 8/3 ∧ l = 8 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_equals_perimeter_l1288_128845


namespace NUMINAMATH_CALUDE_least_product_of_primes_above_30_l1288_128805

theorem least_product_of_primes_above_30 :
  ∃ (p q : ℕ), 
    p.Prime ∧ q.Prime ∧ 
    p > 30 ∧ q > 30 ∧ 
    p ≠ q ∧
    p * q = 1147 ∧
    ∀ (r s : ℕ), r.Prime → s.Prime → r > 30 → s > 30 → r ≠ s → r * s ≥ 1147 :=
by sorry

end NUMINAMATH_CALUDE_least_product_of_primes_above_30_l1288_128805


namespace NUMINAMATH_CALUDE_binary_multiplication_theorem_l1288_128866

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a decimal number to its binary representation as a list of bits -/
def decimal_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
    let rec to_bits (m : Nat) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: to_bits (m / 2)
    to_bits n

theorem binary_multiplication_theorem :
  let a := [true, false, true, true]  -- 1101₂
  let b := [true, true, true]         -- 111₂
  let c := [true, true, false, false, true, false, true]  -- 1010011₂
  binary_to_decimal a * binary_to_decimal b = binary_to_decimal c := by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_theorem_l1288_128866


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1288_128872

theorem negation_of_proposition (p : Prop) :
  (p ↔ ∃ x, x < -1 ∧ x^2 - x + 1 < 0) →
  (¬p ↔ ∀ x, x < -1 → x^2 - x + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1288_128872


namespace NUMINAMATH_CALUDE_emily_beads_count_l1288_128822

/-- Given that Emily can make 4 necklaces and each necklace requires 7 beads,
    prove that she has 28 beads in total. -/
theorem emily_beads_count :
  ∀ (necklaces : ℕ) (beads_per_necklace : ℕ),
    necklaces = 4 →
    beads_per_necklace = 7 →
    necklaces * beads_per_necklace = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_emily_beads_count_l1288_128822


namespace NUMINAMATH_CALUDE_acoustics_class_male_count_l1288_128887

/-- The number of male students in the acoustics class -/
def male_students : ℕ := 120

/-- The number of female students in the acoustics class -/
def female_students : ℕ := 100

/-- The percentage of male students who are engineering students -/
def male_eng_percent : ℚ := 25 / 100

/-- The percentage of female students who are engineering students -/
def female_eng_percent : ℚ := 20 / 100

/-- The percentage of male engineering students who passed the final exam -/
def male_pass_percent : ℚ := 20 / 100

/-- The percentage of female engineering students who passed the final exam -/
def female_pass_percent : ℚ := 25 / 100

/-- The percentage of all engineering students who passed the exam -/
def total_pass_percent : ℚ := 22 / 100

theorem acoustics_class_male_count :
  male_students = 120 ∧
  (male_eng_percent * male_students * male_pass_percent +
   female_eng_percent * female_students * female_pass_percent) =
  total_pass_percent * (male_eng_percent * male_students + female_eng_percent * female_students) :=
by sorry

end NUMINAMATH_CALUDE_acoustics_class_male_count_l1288_128887


namespace NUMINAMATH_CALUDE_train_passing_time_l1288_128856

/-- Proves that a train passing a platform in given time and speed will pass a stationary point in approximately 20 seconds -/
theorem train_passing_time (platform_length : ℝ) (platform_passing_time : ℝ) (train_speed_kmh : ℝ) 
  (h1 : platform_length = 360.0288)
  (h2 : platform_passing_time = 44)
  (h3 : train_speed_kmh = 54) : 
  ∃ (time : ℝ), abs (time - 20) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l1288_128856


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1288_128823

theorem rationalize_denominator : (45 : ℝ) / Real.sqrt 45 = 3 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1288_128823


namespace NUMINAMATH_CALUDE_candy_bars_purchased_l1288_128862

theorem candy_bars_purchased (initial_amount : ℕ) (candy_bar_cost : ℕ) (remaining_amount : ℕ) : 
  initial_amount = 20 →
  candy_bar_cost = 2 →
  remaining_amount = 12 →
  (initial_amount - remaining_amount) / candy_bar_cost = 4 := by
sorry

end NUMINAMATH_CALUDE_candy_bars_purchased_l1288_128862


namespace NUMINAMATH_CALUDE_unique_four_digit_number_l1288_128844

theorem unique_four_digit_number : ∃! n : ℕ,
  (1000 ≤ n) ∧ (n < 10000) ∧  -- 4-digit number
  (∃ a : ℕ, n = a^2) ∧  -- perfect square
  (∃ b : ℕ, n % 1000 = b^3) ∧  -- removing first digit results in a perfect cube
  (∃ c : ℕ, n % 100 = c^4) ∧  -- removing first two digits results in a fourth power
  n = 9216 :=
by sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_l1288_128844


namespace NUMINAMATH_CALUDE_problem_solution_l1288_128848

theorem problem_solution (x y z p q r : ℝ) 
  (h1 : x * y / (x + y) = p)
  (h2 : x * z / (x + z) = q)
  (h3 : y * z / (y + z) = r)
  (h4 : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0)
  (h5 : x ≠ -y ∧ x ≠ -z ∧ y ≠ -z)
  (h6 : p = 3 * q)
  (h7 : p = 2 * r) :
  x = 3 * p / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1288_128848


namespace NUMINAMATH_CALUDE_linear_function_properties_l1288_128893

/-- A linear function passing through two given points -/
def linear_function (k b : ℝ) (x : ℝ) : ℝ := k * x + b

theorem linear_function_properties :
  ∀ (k b : ℝ), k ≠ 0 →
  linear_function k b 2 = -3 →
  linear_function k b (-4) = 0 →
  (k = -1/2 ∧ b = -2) ∧
  (∀ (x m : ℝ), x > -2 → -x + m < linear_function k b x → m ≤ -3) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_properties_l1288_128893


namespace NUMINAMATH_CALUDE_orthocenter_of_specific_triangle_l1288_128879

/-- Triangle ABC in 3D space -/
structure Triangle3D where
  A : ℝ × ℝ × ℝ
  B : ℝ × ℝ × ℝ
  C : ℝ × ℝ × ℝ

/-- The orthocenter of a triangle in 3D space -/
def orthocenter (t : Triangle3D) : ℝ × ℝ × ℝ := sorry

/-- Theorem: The orthocenter of the given triangle is (5/2, 3, 7/2) -/
theorem orthocenter_of_specific_triangle :
  let t : Triangle3D := {
    A := (1, 2, 3),
    B := (5, 3, 1),
    C := (3, 4, 5)
  }
  orthocenter t = (5/2, 3, 7/2) := by sorry

end NUMINAMATH_CALUDE_orthocenter_of_specific_triangle_l1288_128879


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1288_128889

theorem quadratic_equation_solution (a : ℝ) : 
  (1 : ℝ)^2 + a*(1 : ℝ) + 1 = 0 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1288_128889


namespace NUMINAMATH_CALUDE_no_integer_solutions_l1288_128834

theorem no_integer_solutions : 
  ¬∃ (x y z : ℤ), 
    (x^2 - 3*x*y + 2*y^2 - z^2 = 31) ∧ 
    (-x^2 + 6*y*z + 2*z^2 = 44) ∧ 
    (x^2 + x*y + 8*z^2 = 100) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l1288_128834


namespace NUMINAMATH_CALUDE_teacher_li_flags_l1288_128880

theorem teacher_li_flags : ∃ (x : ℕ), x > 0 ∧ 4 * x + 20 = 44 ∧ 4 * x + 20 > 8 * (x - 1) ∧ 4 * x + 20 < 8 * x :=
by sorry

end NUMINAMATH_CALUDE_teacher_li_flags_l1288_128880


namespace NUMINAMATH_CALUDE_total_points_after_perfect_games_l1288_128869

/-- The number of points in a perfect score -/
def perfect_score : ℕ := 21

/-- The number of consecutive perfect games -/
def consecutive_games : ℕ := 3

/-- Theorem: The total points after 3 perfect games is 63 -/
theorem total_points_after_perfect_games :
  perfect_score * consecutive_games = 63 := by
  sorry

end NUMINAMATH_CALUDE_total_points_after_perfect_games_l1288_128869


namespace NUMINAMATH_CALUDE_minimum_beans_purchase_l1288_128890

theorem minimum_beans_purchase (r b : ℝ) : 
  (r ≥ 2 * b + 8 ∧ r ≤ 3 * b) → b ≥ 8 := by sorry

end NUMINAMATH_CALUDE_minimum_beans_purchase_l1288_128890


namespace NUMINAMATH_CALUDE_integer_solution_inequality_system_l1288_128871

theorem integer_solution_inequality_system : 
  ∃! x : ℤ, 2 * x ≤ 1 ∧ x + 2 > 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_integer_solution_inequality_system_l1288_128871


namespace NUMINAMATH_CALUDE_total_bankers_discount_l1288_128837

/-- Represents a bill with its amount, true discount, and interest rate -/
structure Bill where
  amount : ℝ
  trueDiscount : ℝ
  interestRate : ℝ

/-- Calculates the banker's discount for a given bill -/
def bankerDiscount (bill : Bill) : ℝ :=
  (bill.amount - bill.trueDiscount) * bill.interestRate

/-- The four bills given in the problem -/
def bills : List Bill := [
  { amount := 2260, trueDiscount := 360, interestRate := 0.08 },
  { amount := 3280, trueDiscount := 520, interestRate := 0.10 },
  { amount := 4510, trueDiscount := 710, interestRate := 0.12 },
  { amount := 6240, trueDiscount := 980, interestRate := 0.15 }
]

/-- Theorem: The total banker's discount for the given bills is 1673 -/
theorem total_bankers_discount :
  (bills.map bankerDiscount).sum = 1673 := by
  sorry

end NUMINAMATH_CALUDE_total_bankers_discount_l1288_128837


namespace NUMINAMATH_CALUDE_complex_arithmetic_equation_l1288_128803

theorem complex_arithmetic_equation : 
  (8 * 2.25 - 5 * 0.85) / 2.5 + (3/5 * 1.5 - 7/8 * 0.35) / 1.25 = 5.975 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equation_l1288_128803


namespace NUMINAMATH_CALUDE_three_a_equals_30_l1288_128830

theorem three_a_equals_30 
  (h1 : 3 * a - 2 * b - 2 * c = 30)
  (h2 : Real.sqrt (3 * a) - Real.sqrt (2 * b + 2 * c) = 4)
  (h3 : a + b + c = 10)
  : 3 * a = 30 := by
  sorry

end NUMINAMATH_CALUDE_three_a_equals_30_l1288_128830
