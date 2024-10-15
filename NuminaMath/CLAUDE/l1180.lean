import Mathlib

namespace NUMINAMATH_CALUDE_parallel_condition_l1180_118081

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := a * x + 2 * y - 1 = 0
def l₂ (a x y : ℝ) : Prop := x + (a + 1) * y + 4 = 0

-- Define the parallel relation between two lines
def parallel (a : ℝ) : Prop := ∀ x y : ℝ, l₁ a x y ↔ l₂ a x y

-- Theorem statement
theorem parallel_condition (a : ℝ) : parallel a ↔ a = 1 :=
sorry

end NUMINAMATH_CALUDE_parallel_condition_l1180_118081


namespace NUMINAMATH_CALUDE_walter_chores_l1180_118054

theorem walter_chores (total_days : ℕ) (normal_pay exceptional_pay : ℚ) 
  (total_earnings : ℚ) (min_exceptional_days : ℕ) :
  total_days = 15 →
  normal_pay = 4 →
  exceptional_pay = 6 →
  total_earnings = 70 →
  min_exceptional_days = 5 →
  ∃ (normal_days exceptional_days : ℕ),
    normal_days + exceptional_days = total_days ∧
    normal_days * normal_pay + exceptional_days * exceptional_pay = total_earnings ∧
    exceptional_days ≥ min_exceptional_days ∧
    exceptional_days = 5 :=
by sorry

end NUMINAMATH_CALUDE_walter_chores_l1180_118054


namespace NUMINAMATH_CALUDE_fathers_full_time_jobs_l1180_118091

theorem fathers_full_time_jobs (total_parents : ℝ) 
  (h1 : total_parents > 0) -- Ensure total_parents is positive
  (mothers_ratio : ℝ) 
  (h2 : mothers_ratio = 0.4) -- 40% of parents are mothers
  (mothers_full_time_ratio : ℝ) 
  (h3 : mothers_full_time_ratio = 3/4) -- 3/4 of mothers have full-time jobs
  (not_full_time_ratio : ℝ) 
  (h4 : not_full_time_ratio = 0.16) -- 16% of parents do not have full-time jobs
  : (total_parents * (1 - mothers_ratio) - 
     total_parents * (1 - not_full_time_ratio - mothers_ratio * mothers_full_time_ratio)) / 
    (total_parents * (1 - mothers_ratio)) = 9/10 := by
  sorry


end NUMINAMATH_CALUDE_fathers_full_time_jobs_l1180_118091


namespace NUMINAMATH_CALUDE_min_value_expression_l1180_118092

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 16) :
  x^2 + 4*x*y + 4*y^2 + z^3 ≥ 73 ∧ ∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x * y * z = 16 ∧ x^2 + 4*x*y + 4*y^2 + z^3 = 73 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1180_118092


namespace NUMINAMATH_CALUDE_sqrt_sum_diff_complex_expression_system_of_equations_l1180_118042

-- Problem 1
theorem sqrt_sum_diff (a b c : ℝ) (ha : a = 3) (hb : b = 27) (hc : c = 12) :
  Real.sqrt a + Real.sqrt b - Real.sqrt c = 2 * Real.sqrt 3 := by sorry

-- Problem 2
theorem complex_expression (a b c d e : ℝ) (ha : a = 3) (hb : b = 2) (hc : c = 20) (hd : d = 15) (he : e = 5) :
  (Real.sqrt a + Real.sqrt b) * (Real.sqrt a - Real.sqrt b) - (Real.sqrt c - Real.sqrt d) / Real.sqrt e = Real.sqrt 3 - 1 := by sorry

-- Problem 3
theorem system_of_equations (x y : ℝ) (h1 : 2 * (x + 1) - y = 6) (h2 : x = y - 1) :
  x = 5 ∧ y = 6 := by sorry

end NUMINAMATH_CALUDE_sqrt_sum_diff_complex_expression_system_of_equations_l1180_118042


namespace NUMINAMATH_CALUDE_score_54_recorded_as_negative_6_l1180_118039

/-- Calculates the recorded score based on the base score and actual score -/
def recordedScore (baseScore actualScore : Int) : Int :=
  actualScore - baseScore

/-- Theorem: A score of 54 points is recorded as -6 points when the base score is 60 -/
theorem score_54_recorded_as_negative_6 :
  recordedScore 60 54 = -6 := by
  sorry

end NUMINAMATH_CALUDE_score_54_recorded_as_negative_6_l1180_118039


namespace NUMINAMATH_CALUDE_equation_solution_l1180_118080

theorem equation_solution : ∃ (x₁ x₂ : ℝ), x₁ = -3/2 ∧ x₂ = 2 ∧
  (∀ x : ℝ, 2*x^2 - 4*x = 6 - 3*x ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1180_118080


namespace NUMINAMATH_CALUDE_mencius_view_contradicts_option_a_l1180_118014

-- Define the philosophical views
def MenciusView := "Human nature is inherently good"
def OptionAView := "Human nature is evil"

-- Define the passage content
def PassageContent := "Discussion on choices between fish, bear's paws, life, and righteousness"

-- Define Mencius's philosophy
def MenciusPhilosophy := "Advocate for inherent goodness of human nature"

-- Theorem to prove
theorem mencius_view_contradicts_option_a :
  (PassageContent = "Discussion on choices between fish, bear's paws, life, and righteousness") →
  (MenciusPhilosophy = "Advocate for inherent goodness of human nature") →
  (MenciusView ≠ OptionAView) :=
by
  sorry


end NUMINAMATH_CALUDE_mencius_view_contradicts_option_a_l1180_118014


namespace NUMINAMATH_CALUDE_complex_multiplication_l1180_118017

theorem complex_multiplication (z : ℂ) : 
  (z.re = 2 ∧ z.im = -1) → z * (2 + I) = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l1180_118017


namespace NUMINAMATH_CALUDE_candy_distribution_l1180_118064

/-- The number of children in the circle -/
def num_children : ℕ := 73

/-- The total number of candies distributed -/
def total_candies : ℕ := 2020

/-- The position of the n-th candy distribution -/
def candy_position (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of unique positions reached after distributing all candies -/
def unique_positions : ℕ := 37

theorem candy_distribution :
  num_children - unique_positions = 36 :=
sorry

end NUMINAMATH_CALUDE_candy_distribution_l1180_118064


namespace NUMINAMATH_CALUDE_solution_in_third_quadrant_implies_k_bound_l1180_118067

theorem solution_in_third_quadrant_implies_k_bound 
  (k : ℝ) 
  (h : ∃ x : ℝ, 
    π < x ∧ x < 3*π/2 ∧ 
    k * Real.cos x + Real.arccos (π/4) = 0) : 
  k > Real.arccos (π/4) := by
sorry

end NUMINAMATH_CALUDE_solution_in_third_quadrant_implies_k_bound_l1180_118067


namespace NUMINAMATH_CALUDE_area_between_parabola_and_line_l1180_118090

-- Define the parabola function
def parabola (x : ℝ) : ℝ := 2 * x^2

-- Define the line function
def line (x : ℝ) : ℝ := 2 * x

-- Theorem statement
theorem area_between_parabola_and_line :
  ∫ x in (0)..(1), (line x - parabola x) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_area_between_parabola_and_line_l1180_118090


namespace NUMINAMATH_CALUDE_max_value_of_d_l1180_118016

theorem max_value_of_d (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 10)
  (sum_prod_eq : a * b + a * c + a * d + b * c + b * d + c * d = 20) :
  d ≤ (5 + Real.sqrt 105) / 2 ∧ 
  ∃ (a' b' c' d' : ℝ), a' + b' + c' + d' = 10 ∧ 
    a' * b' + a' * c' + a' * d' + b' * c' + b' * d' + c' * d' = 20 ∧
    d' = (5 + Real.sqrt 105) / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_d_l1180_118016


namespace NUMINAMATH_CALUDE_simplify_expression_l1180_118036

theorem simplify_expression (a b m : ℝ) (h1 : a + b = m) (h2 : a * b = -4) :
  (a - 2) * (b - 2) = -2 * m := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1180_118036


namespace NUMINAMATH_CALUDE_scientific_notation_448000_l1180_118075

theorem scientific_notation_448000 : 448000 = 4.48 * (10 : ℝ) ^ 5 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_448000_l1180_118075


namespace NUMINAMATH_CALUDE_first_valid_year_is_2015_l1180_118023

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def is_valid_year (year : ℕ) : Prop :=
  year > 2010 ∧ sum_of_digits year = 8

theorem first_valid_year_is_2015 : 
  ∀ year : ℕ, is_valid_year year → year ≥ 2015 :=
sorry

end NUMINAMATH_CALUDE_first_valid_year_is_2015_l1180_118023


namespace NUMINAMATH_CALUDE_rectangular_plot_length_l1180_118053

theorem rectangular_plot_length (breadth : ℝ) (length : ℝ) (perimeter : ℝ) : 
  length = breadth + 28 →
  perimeter = 2 * length + 2 * breadth →
  perimeter = 5300 / 26.5 →
  length = 64 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_length_l1180_118053


namespace NUMINAMATH_CALUDE_complex_absolute_value_l1180_118006

theorem complex_absolute_value (ω : ℂ) (h : ω = 5 + 3*I) : 
  Complex.abs (ω^2 + 4*ω + 34) = Real.sqrt 6664 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l1180_118006


namespace NUMINAMATH_CALUDE_triangle_angle_sum_contradiction_l1180_118086

theorem triangle_angle_sum_contradiction :
  ∀ (left right top : ℝ),
  right = 60 →
  left = 2 * right →
  top = 70 →
  left + right + top ≠ 180 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_contradiction_l1180_118086


namespace NUMINAMATH_CALUDE_smallest_number_proof_l1180_118048

theorem smallest_number_proof (a b c : ℚ) : 
  b = 4 * a →
  c = 2 * b →
  (a + b + c) / 3 = 78 →
  a = 18 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l1180_118048


namespace NUMINAMATH_CALUDE_pan_division_l1180_118046

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Represents the pan of chocolate cake -/
def pan : Dimensions := ⟨24, 30⟩

/-- Represents a piece of the chocolate cake -/
def piece : Dimensions := ⟨3, 2⟩

/-- Theorem stating that the pan can be divided into exactly 120 pieces -/
theorem pan_division :
  (area pan) / (area piece) = 120 := by sorry

end NUMINAMATH_CALUDE_pan_division_l1180_118046


namespace NUMINAMATH_CALUDE_largest_three_digit_divisible_by_digits_l1180_118079

def is_divisible_by_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d ≠ 0 → n % d = 0

theorem largest_three_digit_divisible_by_digits :
  ∀ n : ℕ, 800 ≤ n → n < 900 → is_divisible_by_digits n → n ≤ 888 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_divisible_by_digits_l1180_118079


namespace NUMINAMATH_CALUDE_factorial_expression_equals_100_l1180_118031

-- Define factorial
def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

-- Theorem statement
theorem factorial_expression_equals_100 : 
  (factorial 11 - factorial 10) / factorial 9 = 100 := by
  sorry

end NUMINAMATH_CALUDE_factorial_expression_equals_100_l1180_118031


namespace NUMINAMATH_CALUDE_tangent_at_one_tangent_through_point_l1180_118022

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x^2 + x + 2

-- Theorem for the tangent line at x = 1
theorem tangent_at_one :
  ∃ (m b : ℝ), ∀ x y, y = m*x + b ↔ 
    (x = 1 ∧ y = f 1) ∨ 
    (y - f 1 = (3 * 1^2 - 2 * 1 + 1) * (x - 1)) :=
sorry

-- Theorem for the tangent lines passing through (1,3)
theorem tangent_through_point :
  ∃ (m₁ b₁ m₂ b₂ : ℝ), 
    (∀ x y, y = m₁*x + b₁ → (∃ t, f t = y ∧ 3*t^2 - 2*t + 1 = m₁ ∧ x = t)) ∧
    (∀ x y, y = m₂*x + b₂ → (∃ t, f t = y ∧ 3*t^2 - 2*t + 1 = m₂ ∧ x = t)) ∧
    m₁ = 1 ∧ b₁ = 2 ∧ m₂ = 2 ∧ b₂ = 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_at_one_tangent_through_point_l1180_118022


namespace NUMINAMATH_CALUDE_car_owners_without_motorcycles_l1180_118015

theorem car_owners_without_motorcycles 
  (total_adults : ℕ) 
  (car_owners : ℕ) 
  (motorcycle_owners : ℕ) 
  (h1 : total_adults = 400)
  (h2 : car_owners = 350)
  (h3 : motorcycle_owners = 60)
  (h4 : total_adults ≤ car_owners + motorcycle_owners) :
  car_owners - (car_owners + motorcycle_owners - total_adults) = 340 :=
by sorry

end NUMINAMATH_CALUDE_car_owners_without_motorcycles_l1180_118015


namespace NUMINAMATH_CALUDE_translation_exists_l1180_118068

-- Define the set of line segments
def LineSegments : Set (Set ℝ) := sorry

-- Define the property that the total length of line segments is less than 1
def TotalLengthLessThanOne (segments : Set (Set ℝ)) : Prop := sorry

-- Define a set of n points on the line
def Points (n : ℕ) : Set ℝ := sorry

-- Define a translation vector
def TranslationVector : ℝ := sorry

-- Define the property that the translation vector length does not exceed n/2
def TranslationLengthValid (v : ℝ) (n : ℕ) : Prop := 
  abs v ≤ n / 2

-- Define the translated points
def TranslatedPoints (points : Set ℝ) (v : ℝ) : Set ℝ := sorry

-- Define the property that no translated point intersects with any line segment
def NoIntersection (translatedPoints : Set ℝ) (segments : Set (Set ℝ)) : Prop := sorry

-- The main theorem
theorem translation_exists (n : ℕ) (segments : Set (Set ℝ)) (points : Set ℝ) 
  (h1 : TotalLengthLessThanOne segments) 
  (h2 : points = Points n) :
  ∃ v : ℝ, TranslationLengthValid v n ∧ 
    NoIntersection (TranslatedPoints points v) segments := by sorry

end NUMINAMATH_CALUDE_translation_exists_l1180_118068


namespace NUMINAMATH_CALUDE_jackson_souvenirs_l1180_118082

/-- Calculates the total number of souvenirs collected by Jackson -/
def total_souvenirs (hermit_crabs : ℕ) (shells_per_crab : ℕ) (starfish_per_shell : ℕ) : ℕ :=
  let spiral_shells := hermit_crabs * shells_per_crab
  let starfish := spiral_shells * starfish_per_shell
  hermit_crabs + spiral_shells + starfish

/-- Proves that Jackson collects 450 souvenirs in total -/
theorem jackson_souvenirs :
  total_souvenirs 45 3 2 = 450 := by
  sorry

#eval total_souvenirs 45 3 2

end NUMINAMATH_CALUDE_jackson_souvenirs_l1180_118082


namespace NUMINAMATH_CALUDE_daniels_animals_legs_l1180_118078

/-- The number of legs an animal has -/
def legs (animal : String) : ℕ :=
  match animal with
  | "horse" => 4
  | "dog" => 4
  | "cat" => 4
  | "turtle" => 4
  | "goat" => 4
  | _ => 0

/-- The number of animals Daniel has -/
def animal_count (animal : String) : ℕ :=
  match animal with
  | "horse" => 2
  | "dog" => 5
  | "cat" => 7
  | "turtle" => 3
  | "goat" => 1
  | _ => 0

/-- The total number of legs of all animals -/
def total_legs : ℕ :=
  (animal_count "horse" * legs "horse") +
  (animal_count "dog" * legs "dog") +
  (animal_count "cat" * legs "cat") +
  (animal_count "turtle" * legs "turtle") +
  (animal_count "goat" * legs "goat")

theorem daniels_animals_legs : total_legs = 72 := by
  sorry

end NUMINAMATH_CALUDE_daniels_animals_legs_l1180_118078


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l1180_118087

theorem cube_volume_ratio (edge1 edge2 : ℝ) (h : edge2 = 6 * edge1) :
  (edge1^3) / (edge2^3) = 1 / 216 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l1180_118087


namespace NUMINAMATH_CALUDE_least_four_digit_multiple_of_six_l1180_118096

theorem least_four_digit_multiple_of_six : ∃ n : ℕ, 
  (n ≥ 1000 ∧ n < 10000) ∧  -- four-digit number
  n % 6 = 0 ∧               -- multiple of 6
  (∀ m : ℕ, (m ≥ 1000 ∧ m < 10000 ∧ m % 6 = 0) → n ≤ m) ∧ -- least such number
  n = 1002 := by
sorry

end NUMINAMATH_CALUDE_least_four_digit_multiple_of_six_l1180_118096


namespace NUMINAMATH_CALUDE_sphere_cube_ratios_l1180_118077

theorem sphere_cube_ratios (R : ℝ) (a : ℝ) (h : a = 2 * R / Real.sqrt 3) :
  let sphere_surface := 4 * Real.pi * R^2
  let cube_surface := 6 * a^2
  let sphere_volume := 4 / 3 * Real.pi * R^3
  let cube_volume := a^3
  (sphere_surface / cube_surface = Real.pi / 2) ∧
  (sphere_volume / cube_volume = Real.pi * Real.sqrt 3 / 2) := by
sorry

end NUMINAMATH_CALUDE_sphere_cube_ratios_l1180_118077


namespace NUMINAMATH_CALUDE_special_rectangle_perimeter_l1180_118025

/-- A rectangle with the property that increasing both its length and width by 6
    results in an area increase of 114 -/
structure SpecialRectangle where
  length : ℝ
  width : ℝ
  area_increase : (length + 6) * (width + 6) - length * width = 114

theorem special_rectangle_perimeter (rect : SpecialRectangle) :
  2 * (rect.length + rect.width) = 26 :=
sorry

end NUMINAMATH_CALUDE_special_rectangle_perimeter_l1180_118025


namespace NUMINAMATH_CALUDE_games_per_season_l1180_118030

/-- Given the following conditions:
  - Louie scored 4 goals in the last match
  - Louie scored 40 goals in previous matches
  - Louie's brother scored twice as many goals as Louie in the last match
  - Louie's brother has played for 3 seasons
  - The total number of goals scored by both brothers is 1244
Prove that there are 50 games in each season -/
theorem games_per_season (louie_last_match : ℕ) (louie_previous : ℕ) 
  (brother_multiplier : ℕ) (brother_seasons : ℕ) (total_goals : ℕ) :
  louie_last_match = 4 →
  louie_previous = 40 →
  brother_multiplier = 2 →
  brother_seasons = 3 →
  total_goals = 1244 →
  ∃ (games_per_season : ℕ), 
    louie_last_match + louie_previous + 
    brother_multiplier * louie_last_match * games_per_season * brother_seasons = 
    total_goals ∧ games_per_season = 50 :=
by sorry

end NUMINAMATH_CALUDE_games_per_season_l1180_118030


namespace NUMINAMATH_CALUDE_larger_number_proof_l1180_118070

theorem larger_number_proof (a b : ℕ+) (h1 : Nat.gcd a b = 20) (h2 : Nat.lcm a b = 9660) :
  max a b = 460 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1180_118070


namespace NUMINAMATH_CALUDE_f_zero_f_expression_intersection_complement_l1180_118088

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Given conditions
axiom f_property : ∀ (x y : ℝ), f (x + y) - f y = x * (x + 2 * y + 1)
axiom f_one : f 1 = 0

-- Define sets A and B
def A : Set ℝ := {a | ∀ x ∈ Set.Ioo 0 (1/2), f x + 3 < 2 * x + a}
def B : Set ℝ := {a | ∀ x ∈ Set.Icc (-2) 2, Monotone (fun x ↦ f x - a * x)}

-- Theorem statements
theorem f_zero : f 0 = -2 := sorry

theorem f_expression : ∀ x : ℝ, f x = x^2 + x - 2 := sorry

theorem intersection_complement : A ∩ (Set.univ \ B) = Set.Icc 1 5 := sorry

end NUMINAMATH_CALUDE_f_zero_f_expression_intersection_complement_l1180_118088


namespace NUMINAMATH_CALUDE_sum_of_terms_l1180_118043

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem sum_of_terms (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 2 + a 8 = 10 →
  a 1 + a 3 + a 5 + a 7 + a 9 = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_terms_l1180_118043


namespace NUMINAMATH_CALUDE_sum_of_max_min_f_l1180_118073

noncomputable def f (x : ℝ) : ℝ := ((x + 1)^2 + Real.sin x) / (x^2 + 1)

theorem sum_of_max_min_f : 
  ∃ (M m : ℝ), (∀ x, f x ≤ M) ∧ (∀ x, m ≤ f x) ∧ (M + m = 2) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_max_min_f_l1180_118073


namespace NUMINAMATH_CALUDE_claire_pets_ratio_l1180_118095

theorem claire_pets_ratio : 
  ∀ (total_pets gerbils hamsters male_gerbils male_hamsters : ℕ),
    total_pets = 92 →
    gerbils + hamsters = total_pets →
    gerbils = 68 →
    male_hamsters = hamsters / 3 →
    male_gerbils + male_hamsters = 25 →
    male_gerbils * 4 = gerbils :=
by
  sorry

end NUMINAMATH_CALUDE_claire_pets_ratio_l1180_118095


namespace NUMINAMATH_CALUDE_eightieth_digit_of_one_seventh_l1180_118061

def decimal_representation_of_one_seventh : List Nat := [1, 4, 2, 8, 5, 7]

theorem eightieth_digit_of_one_seventh : 
  (decimal_representation_of_one_seventh[(80 - 1) % decimal_representation_of_one_seventh.length] = 4) := by
  sorry

end NUMINAMATH_CALUDE_eightieth_digit_of_one_seventh_l1180_118061


namespace NUMINAMATH_CALUDE_bob_cleaning_time_l1180_118009

/-- 
Given that Alice takes 32 minutes to clean her room and Bob takes 3/4 of Alice's time,
prove that Bob takes 24 minutes to clean his room.
-/
theorem bob_cleaning_time : 
  let alice_time : ℚ := 32
  let bob_fraction : ℚ := 3/4
  let bob_time : ℚ := alice_time * bob_fraction
  bob_time = 24 := by sorry

end NUMINAMATH_CALUDE_bob_cleaning_time_l1180_118009


namespace NUMINAMATH_CALUDE_intersection_M_N_l1180_118010

def M : Set ℝ := {0, 1, 2}
def N : Set ℝ := {x | x^2 - 3*x + 2 ≤ 0}

theorem intersection_M_N : M ∩ N = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1180_118010


namespace NUMINAMATH_CALUDE_subtract_negative_one_l1180_118029

theorem subtract_negative_one : 3 - (-1) = 4 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negative_one_l1180_118029


namespace NUMINAMATH_CALUDE_total_hotdogs_is_125_l1180_118004

/-- The total number of hotdogs brought by two neighbors, where one neighbor brings 75 hotdogs
    and the other brings 25 fewer hotdogs than the first. -/
def total_hotdogs : ℕ :=
  let first_neighbor := 75
  let second_neighbor := first_neighbor - 25
  first_neighbor + second_neighbor

/-- Theorem stating that the total number of hotdogs brought by the neighbors is 125. -/
theorem total_hotdogs_is_125 : total_hotdogs = 125 := by
  sorry

end NUMINAMATH_CALUDE_total_hotdogs_is_125_l1180_118004


namespace NUMINAMATH_CALUDE_quadratic_root_value_l1180_118008

theorem quadratic_root_value (c : ℝ) : 
  (∀ x : ℝ, 4 * x^2 + 20 * x + c = 0 ↔ x = (-20 + Real.sqrt 16) / 8 ∨ x = (-20 - Real.sqrt 16) / 8) 
  → c = 24 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l1180_118008


namespace NUMINAMATH_CALUDE_divisibility_by_396_l1180_118021

def is_divisible_by_396 (n : ℕ) : Prop :=
  n % 396 = 0

def last_two_digits (n : ℕ) : ℕ :=
  n % 100

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem divisibility_by_396 (n : ℕ) :
  (n ≥ 10000 ∧ n < 100000) →
  (is_divisible_by_396 n ↔ 
    (last_two_digits n % 4 = 0 ∧ 
    (digit_sum n = 18 ∨ digit_sum n = 27))) :=
by sorry

#check divisibility_by_396

end NUMINAMATH_CALUDE_divisibility_by_396_l1180_118021


namespace NUMINAMATH_CALUDE_savings_growth_l1180_118032

/-- The amount of money in a savings account after n years -/
def savings_amount (a : ℝ) (n : ℕ) : ℝ :=
  a * (1 + 0.02) ^ n

/-- Theorem: The amount of money in a savings account after n years,
    given an initial deposit of a rubles and a 2% annual interest rate,
    is equal to a × 1.02^n rubles. -/
theorem savings_growth (a : ℝ) (n : ℕ) :
  savings_amount a n = a * 1.02 ^ n :=
by sorry

end NUMINAMATH_CALUDE_savings_growth_l1180_118032


namespace NUMINAMATH_CALUDE_mountain_height_l1180_118072

/-- The relative height of a mountain given temperature conditions -/
theorem mountain_height (temp_decrease_rate : ℝ) (summit_temp : ℝ) (base_temp : ℝ) :
  temp_decrease_rate = 0.7 →
  summit_temp = 14.1 →
  base_temp = 26 →
  (base_temp - summit_temp) / temp_decrease_rate * 100 = 1700 := by
  sorry

#check mountain_height

end NUMINAMATH_CALUDE_mountain_height_l1180_118072


namespace NUMINAMATH_CALUDE_set_equivalences_l1180_118060

/-- The set of non-negative even numbers not greater than 10 -/
def nonNegEvenSet : Set ℕ :=
  {n : ℕ | n % 2 = 0 ∧ n ≤ 10}

/-- The set of prime numbers not greater than 10 -/
def primeSet : Set ℕ :=
  {n : ℕ | Nat.Prime n ∧ n ≤ 10}

/-- The equation x^2 + 2x - 15 = 0 -/
def equation (x : ℝ) : Prop :=
  x^2 + 2*x - 15 = 0

theorem set_equivalences :
  (nonNegEvenSet = {0, 2, 4, 6, 8, 10}) ∧
  (primeSet = {2, 3, 5, 7}) ∧
  ({x : ℝ | equation x} = {-5, 3}) := by
  sorry

end NUMINAMATH_CALUDE_set_equivalences_l1180_118060


namespace NUMINAMATH_CALUDE_catherine_wins_l1180_118013

/-- Represents a point on a circle -/
structure CirclePoint where
  -- Define necessary properties for a point on a circle

/-- Represents a triangle formed by three points on the circle -/
structure Triangle where
  vertex1 : CirclePoint
  vertex2 : CirclePoint
  vertex3 : CirclePoint

/-- Represents the state of the game -/
structure GameState where
  chosenTriangles : List Triangle
  currentPlayer : Bool  -- True for Peter, False for Catherine

/-- Checks if a set of triangles has a common interior point -/
def hasCommonInteriorPoint (triangles : List Triangle) : Bool :=
  sorry

/-- Checks if a triangle is valid to be chosen -/
def isValidTriangle (triangle : Triangle) (state : GameState) : Bool :=
  sorry

/-- Represents a move in the game -/
def makeMove (state : GameState) (triangle : Triangle) : Option GameState :=
  sorry

/-- Theorem stating Catherine has a winning strategy -/
theorem catherine_wins (points : List CirclePoint) 
  (h1 : points.length = 100)
  (h2 : points.Nodup) : 
  ∃ (strategy : GameState → Triangle), 
    ∀ (finalState : GameState), 
      (finalState.currentPlayer = false → 
        ∃ (move : Triangle), isValidTriangle move finalState) ∧
      (finalState.currentPlayer = true → 
        ¬∃ (move : Triangle), isValidTriangle move finalState) :=
  sorry

end NUMINAMATH_CALUDE_catherine_wins_l1180_118013


namespace NUMINAMATH_CALUDE_percentage_decrease_l1180_118011

theorem percentage_decrease (initial : ℝ) (increase : ℝ) (final : ℝ) :
  initial = 1500 →
  increase = 20 →
  final = 1080 →
  ∃ y : ℝ, y = 40 ∧ final = (initial * (1 + increase / 100)) * (1 - y / 100) :=
by sorry

end NUMINAMATH_CALUDE_percentage_decrease_l1180_118011


namespace NUMINAMATH_CALUDE_negation_of_implication_l1180_118099

/-- Two lines in a 3D space -/
structure Line3D where
  -- Define necessary properties for a 3D line
  -- This is a simplified representation
  dummy : Unit

/-- Predicate to check if two lines have a common point -/
def have_common_point (l1 l2 : Line3D) : Prop :=
  sorry -- Definition omitted for simplicity

/-- Predicate to check if two lines are skew -/
def are_skew (l1 l2 : Line3D) : Prop :=
  sorry -- Definition omitted for simplicity

theorem negation_of_implication (l1 l2 : Line3D) :
  (¬(¬(have_common_point l1 l2) → are_skew l1 l2)) ↔
  (have_common_point l1 l2 → ¬(are_skew l1 l2)) :=
by
  sorry

#check negation_of_implication

end NUMINAMATH_CALUDE_negation_of_implication_l1180_118099


namespace NUMINAMATH_CALUDE_exponent_equality_l1180_118045

theorem exponent_equality (y x : ℕ) (h1 : 16^y = 4^x) (h2 : y = 8) : x = 16 := by
  sorry

end NUMINAMATH_CALUDE_exponent_equality_l1180_118045


namespace NUMINAMATH_CALUDE_journey_time_reduction_l1180_118007

/-- Given a person's journey where increasing speed by 10% reduces time by x minutes, 
    prove the original journey time was 11x minutes. -/
theorem journey_time_reduction (d : ℝ) (s : ℝ) (x : ℝ) 
  (h1 : d > 0) (h2 : s > 0) (h3 : x > 0) 
  (h4 : d / s - d / (1.1 * s) = x) : 
  d / s = 11 * x := by
sorry

end NUMINAMATH_CALUDE_journey_time_reduction_l1180_118007


namespace NUMINAMATH_CALUDE_adjacent_same_face_exists_l1180_118083

/-- Represents a coin, which can be either heads or tails -/
inductive Coin
| Heads
| Tails

/-- Represents a circular arrangement of 11 coins -/
def CoinArrangement := Fin 11 → Coin

/-- Two positions in the circle are adjacent if they differ by 1 modulo 11 -/
def adjacent (i j : Fin 11) : Prop :=
  (i.val + 1) % 11 = j.val ∨ (j.val + 1) % 11 = i.val

/-- Main theorem: In any arrangement of 11 coins, there exists a pair of adjacent coins showing the same face -/
theorem adjacent_same_face_exists (arrangement : CoinArrangement) :
  ∃ (i j : Fin 11), adjacent i j ∧ arrangement i = arrangement j := by
  sorry

end NUMINAMATH_CALUDE_adjacent_same_face_exists_l1180_118083


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1180_118003

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (3 * x₁^2 + 6 * x₁ - 9 = 0) → 
  (3 * x₂^2 + 6 * x₂ - 9 = 0) → 
  (x₁ ≠ x₂) →
  (x₁^2 + x₂^2 = 10) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1180_118003


namespace NUMINAMATH_CALUDE_circumcircle_of_triangle_l1180_118040

-- Define the points A and B
def A : ℝ × ℝ := (3, 2)
def B : ℝ × ℝ := (-1, 5)

-- Define the line on which point C lies
def line_C (x y : ℝ) : Prop := 3 * x - y + 3 = 0

-- Define the area of the triangle
def triangle_area : ℝ := 10

-- Define the possible equations of the circumcircle
def circumcircle_eq1 (x y : ℝ) : Prop :=
  x^2 + y^2 - (1/2) * x - 5 * y - (3/2) = 0

def circumcircle_eq2 (x y : ℝ) : Prop :=
  x^2 + y^2 - (25/6) * x - (89/9) * y + (347/18) = 0

-- The theorem to be proved
theorem circumcircle_of_triangle :
  ∃ (C : ℝ × ℝ),
    line_C C.1 C.2 ∧
    (∀ (x y : ℝ), circumcircle_eq1 x y ∨ circumcircle_eq2 x y) :=
  sorry

end NUMINAMATH_CALUDE_circumcircle_of_triangle_l1180_118040


namespace NUMINAMATH_CALUDE_park_creatures_l1180_118098

/-- The number of dogs at the park -/
def num_dogs : ℕ := 60

/-- The number of people at the park -/
def num_people : ℕ := num_dogs / 2

/-- The number of snakes at the park -/
def num_snakes : ℕ := num_people / 2

/-- The total number of eyes and legs of all creatures at the park -/
def total_eyes_and_legs : ℕ := 510

theorem park_creatures :
  (num_dogs = 2 * num_people) ∧
  (num_people = 2 * num_snakes) ∧
  (4 * num_dogs + 4 * num_people + 2 * num_snakes = total_eyes_and_legs) :=
by sorry

#check park_creatures

end NUMINAMATH_CALUDE_park_creatures_l1180_118098


namespace NUMINAMATH_CALUDE_circle_trajectory_l1180_118052

/-- Circle 1 with center (a/2, -1) and radius sqrt((a/2)^2 + 2) -/
def circle1 (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a/2)^2 + (p.2 + 1)^2 = (a/2)^2 + 2}

/-- Circle 2 with center (0, 0) and radius 1 -/
def circle2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

/-- Line y = x - 1 -/
def symmetryLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1 - 1}

/-- Point C(-a, a) -/
def pointC (a : ℝ) : ℝ × ℝ := (-a, a)

/-- Circle P passing through point C(-a, a) and tangent to y-axis -/
def circleP (a : ℝ) (center : ℝ × ℝ) : Prop :=
  (center.1 + a)^2 + (center.2 - a)^2 = center.1^2

/-- Trajectory of the center of circle P -/
def trajectoryP : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 + 4*p.1 - 4*p.2 + 8 = 0}

theorem circle_trajectory :
  ∃ (a : ℝ), 
    (∀ (p : ℝ × ℝ), p ∈ symmetryLine → (p ∈ circle1 a ↔ p ∈ circle2)) ∧
    (a = 2) ∧
    (∀ (center : ℝ × ℝ), circleP a center → center ∈ trajectoryP) :=
  sorry

end NUMINAMATH_CALUDE_circle_trajectory_l1180_118052


namespace NUMINAMATH_CALUDE_find_k_value_l1180_118066

theorem find_k_value (x : ℝ) (k : ℝ) : 
  x = 2 → 
  k / (x - 3) - 1 / (3 - x) = 1 → 
  k = -2 := by
sorry

end NUMINAMATH_CALUDE_find_k_value_l1180_118066


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1180_118056

-- Problem 1
theorem problem_1 : (1) - 2 + 3 - 4 + 5 = 3 := by sorry

-- Problem 2
theorem problem_2 : (-4/7) / (8/49) = -7/2 := by sorry

-- Problem 3
theorem problem_3 : (1/2 - 3/5 + 2/3) * (-15) = -17/2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1180_118056


namespace NUMINAMATH_CALUDE_certain_number_problem_l1180_118057

theorem certain_number_problem (a b : ℝ) (h1 : a ≠ b) (h2 : a + b = 4) 
  (h3 : a * (a - 4) = b * (b - 4)) : a * (a - 4) = -3 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1180_118057


namespace NUMINAMATH_CALUDE_commercial_length_proof_l1180_118038

theorem commercial_length_proof (x : ℝ) : 
  (3 * x + 11 * 2 = 37) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_commercial_length_proof_l1180_118038


namespace NUMINAMATH_CALUDE_fraction_simplification_l1180_118089

theorem fraction_simplification :
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3 + 1 / (1/3)^4) = 1 / 120 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1180_118089


namespace NUMINAMATH_CALUDE_determine_coin_weight_in_two_weighings_l1180_118020

-- Define the set of possible coin weights
def coin_weights : Finset ℕ := {7, 8, 9, 10, 11, 12, 13}

-- Define a type for the balance scale comparison result
inductive ComparisonResult
| Equal : ComparisonResult
| LeftHeavier : ComparisonResult
| RightHeavier : ComparisonResult

-- Define a function to simulate a weighing
def weigh (left right : ℕ) : ComparisonResult :=
  if left = right then ComparisonResult.Equal
  else if left > right then ComparisonResult.LeftHeavier
  else ComparisonResult.RightHeavier

-- Define the theorem
theorem determine_coin_weight_in_two_weighings :
  ∀ (x : ℕ), x ∈ coin_weights →
    ∃ (w₁ w₂ : ℕ × ℕ),
      (weigh (70 * x) (w₁.1 * 70) = ComparisonResult.Equal ∨
       (weigh (70 * x) (w₁.1 * 70) = ComparisonResult.LeftHeavier ∧
        weigh (70 * x) (w₂.1 * 70) = ComparisonResult.Equal) ∨
       (weigh (70 * x) (w₁.1 * 70) = ComparisonResult.RightHeavier ∧
        weigh (70 * x) (w₂.2 * 70) = ComparisonResult.Equal)) :=
by sorry

end NUMINAMATH_CALUDE_determine_coin_weight_in_two_weighings_l1180_118020


namespace NUMINAMATH_CALUDE_logarithmic_equation_solution_l1180_118049

theorem logarithmic_equation_solution :
  ∀ x : ℝ, x > 0 → x ≠ 1 →
  (6 - (1 + 4 * 9^(4 - 2 * (Real.log 3 / Real.log (Real.sqrt 3)))) * (Real.log x / Real.log 7) = Real.log 7 / Real.log x) →
  (x = 7 ∨ x = Real.rpow 7 (1/5)) := by
  sorry

end NUMINAMATH_CALUDE_logarithmic_equation_solution_l1180_118049


namespace NUMINAMATH_CALUDE_computer_price_ratio_l1180_118074

theorem computer_price_ratio (c : ℝ) (h1 : c > 0) (h2 : c * 1.3 = 351) :
  (c + 351) / c = 2.3 := by
sorry

end NUMINAMATH_CALUDE_computer_price_ratio_l1180_118074


namespace NUMINAMATH_CALUDE_exchange_rate_l1180_118005

def goose_to_duck : ℕ := 2
def pigeon_to_duck : ℕ := 5

theorem exchange_rate (geese : ℕ) : 
  geese * (goose_to_duck * pigeon_to_duck) = 
  geese * 10 := by sorry

end NUMINAMATH_CALUDE_exchange_rate_l1180_118005


namespace NUMINAMATH_CALUDE_infinitely_many_common_divisors_l1180_118018

theorem infinitely_many_common_divisors :
  ∀ k : ℕ, ∃ n : ℕ, ∃ d : ℕ, d > 1 ∧ d ∣ (2 * n - 3) ∧ d ∣ (3 * n - 2) :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_common_divisors_l1180_118018


namespace NUMINAMATH_CALUDE_triangle_perimeter_bound_l1180_118000

theorem triangle_perimeter_bound : 
  ∀ (s : ℝ), s > 0 → 7 + s > 19 → 19 + s > 7 → s + 7 + 19 < 53 :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_bound_l1180_118000


namespace NUMINAMATH_CALUDE_balance_implies_20g_difference_l1180_118059

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k
def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

theorem balance_implies_20g_difference 
  (weights : Finset ℕ) 
  (h_weights : weights = Finset.range 40)
  (left_pan right_pan : Finset ℕ)
  (h_left : left_pan ⊆ weights ∧ left_pan.card = 10 ∧ ∀ n ∈ left_pan, is_even n)
  (h_right : right_pan ⊆ weights ∧ right_pan.card = 10 ∧ ∀ n ∈ right_pan, is_odd n)
  (h_balance : left_pan.sum id = right_pan.sum id) :
  ∃ (a b : ℕ), (a ∈ left_pan ∧ b ∈ left_pan ∧ a - b = 20) ∨ 
               (a ∈ right_pan ∧ b ∈ right_pan ∧ a - b = 20) :=
sorry

end NUMINAMATH_CALUDE_balance_implies_20g_difference_l1180_118059


namespace NUMINAMATH_CALUDE_range_of_f_l1180_118044

def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 3 * a + b

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def domain (a : ℝ) : Set ℝ := Set.Icc (a - 1) (2 * a)

theorem range_of_f (a b : ℝ) (h1 : is_even_function (f a b))
  (h2 : ∀ x ∈ domain a, f a b x ∈ Set.Icc 1 (31/27)) :
  Set.range (f a b) = Set.Icc 1 (31/27) := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l1180_118044


namespace NUMINAMATH_CALUDE_operations_sum_2345_l1180_118062

def apply_operations (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  (d1^2 * 1000) + (d2 * d3 * 100) + (d2 * d3 * 10) + (10 - d4)

theorem operations_sum_2345 :
  apply_operations 2345 = 5325 := by
  sorry

end NUMINAMATH_CALUDE_operations_sum_2345_l1180_118062


namespace NUMINAMATH_CALUDE_function_inequality_l1180_118024

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem function_inequality (h1 : Differentiable ℝ f) 
  (h2 : ∀ x, f x < deriv f x) : 
  f 2 > Real.exp 2 * f 0 ∧ f 2017 > Real.exp 2017 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1180_118024


namespace NUMINAMATH_CALUDE_carries_mountain_dew_oz_per_can_l1180_118084

/-- Represents the punch recipe and serving information -/
structure PunchRecipe where
  mountain_dew_cans : ℕ
  ice_oz : ℕ
  fruit_juice_oz : ℕ
  total_servings : ℕ
  oz_per_serving : ℕ

/-- Calculates the ounces of Mountain Dew per can -/
def mountain_dew_oz_per_can (recipe : PunchRecipe) : ℕ :=
  let total_oz := recipe.total_servings * recipe.oz_per_serving
  let non_mountain_dew_oz := recipe.ice_oz + recipe.fruit_juice_oz
  let total_mountain_dew_oz := total_oz - non_mountain_dew_oz
  total_mountain_dew_oz / recipe.mountain_dew_cans

/-- Carrie's punch recipe -/
def carries_recipe : PunchRecipe := {
  mountain_dew_cans := 6
  ice_oz := 28
  fruit_juice_oz := 40
  total_servings := 14
  oz_per_serving := 10
}

/-- Theorem stating that each can of Mountain Dew in Carrie's recipe contains 12 oz -/
theorem carries_mountain_dew_oz_per_can :
  mountain_dew_oz_per_can carries_recipe = 12 := by
  sorry

end NUMINAMATH_CALUDE_carries_mountain_dew_oz_per_can_l1180_118084


namespace NUMINAMATH_CALUDE_system_solution_l1180_118065

theorem system_solution :
  ∃ (x y : ℚ), (3 * x - 4 * y = 1) ∧ (5 * x + 2 * y = 6) ∧ (x = 1) ∧ (y = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1180_118065


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l1180_118034

theorem quadratic_roots_problem (a b : ℝ) (x₁ x₂ y₁ y₂ : ℝ) :
  x₁^2 + a^2*x₁ + b = 0 ∧
  x₂^2 + a^2*x₂ + b = 0 ∧
  y₁^2 + 5*a*y₁ + 7 = 0 ∧
  y₂^2 + 5*a*y₂ + 7 = 0 ∧
  x₁ - y₁ = 2 ∧
  x₂ - y₂ = 2 →
  a = 4 ∧ b = -29 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l1180_118034


namespace NUMINAMATH_CALUDE_line_intersection_canonical_equation_l1180_118063

/-- The canonical equation of the line of intersection of two planes -/
theorem line_intersection_canonical_equation 
  (plane1 : ℝ → ℝ → ℝ → Prop) 
  (plane2 : ℝ → ℝ → ℝ → Prop)
  (h1 : ∀ x y z, plane1 x y z ↔ x - y + z - 2 = 0)
  (h2 : ∀ x y z, plane2 x y z ↔ x - 2*y - z + 4 = 0) :
  ∃ t : ℝ, ∀ x y z, 
    (plane1 x y z ∧ plane2 x y z) ↔ 
    ((x - 8) / 3 = t ∧ (y - 6) / 2 = t ∧ z / (-1) = t) :=
sorry

end NUMINAMATH_CALUDE_line_intersection_canonical_equation_l1180_118063


namespace NUMINAMATH_CALUDE_right_triangle_with_consecutive_legs_and_hypotenuse_31_l1180_118041

theorem right_triangle_with_consecutive_legs_and_hypotenuse_31 :
  ∃ (a b : ℕ), 
    a + 1 = b ∧ 
    a^2 + b^2 = 31^2 ∧ 
    a + b = 43 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_with_consecutive_legs_and_hypotenuse_31_l1180_118041


namespace NUMINAMATH_CALUDE_stratified_sampling_medium_stores_stratified_sampling_medium_stores_correct_l1180_118019

theorem stratified_sampling_medium_stores 
  (total_stores medium_stores sample_size : ℕ) 
  (h1 : total_stores > 0) 
  (h2 : medium_stores ≤ total_stores) 
  (h3 : sample_size ≤ total_stores) : 
  ℕ :=
  let medium_stores_to_draw := (medium_stores * sample_size) / total_stores
  medium_stores_to_draw

#check stratified_sampling_medium_stores

theorem stratified_sampling_medium_stores_correct 
  (total_stores medium_stores sample_size : ℕ) 
  (h1 : total_stores > 0) 
  (h2 : medium_stores ≤ total_stores) 
  (h3 : sample_size ≤ total_stores) : 
  stratified_sampling_medium_stores total_stores medium_stores sample_size h1 h2 h3 = 
  (medium_stores * sample_size) / total_stores :=
by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_medium_stores_stratified_sampling_medium_stores_correct_l1180_118019


namespace NUMINAMATH_CALUDE_martha_family_women_without_daughters_l1180_118002

/-- Represents the family structure of Martha and her descendants -/
structure MarthaFamily where
  daughters : ℕ
  total_descendants : ℕ
  daughters_with_children : ℕ

/-- The number of women (daughters and granddaughters) who have no daughters -/
def women_without_daughters (f : MarthaFamily) : ℕ :=
  f.total_descendants - f.daughters_with_children

/-- Theorem stating the number of women without daughters in Martha's family -/
theorem martha_family_women_without_daughters :
  ∀ f : MarthaFamily,
  f.daughters = 8 →
  f.total_descendants = 40 →
  f.daughters_with_children * 8 = f.total_descendants - f.daughters →
  women_without_daughters f = 36 :=
by sorry

end NUMINAMATH_CALUDE_martha_family_women_without_daughters_l1180_118002


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_numbers_l1180_118035

def numbers : List ℝ := [12, 18, 25, 33, 40]

theorem arithmetic_mean_of_numbers :
  (numbers.sum / numbers.length : ℝ) = 25.6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_numbers_l1180_118035


namespace NUMINAMATH_CALUDE_prob_top_joker_modified_deck_l1180_118055

/-- A deck of cards with jokers -/
structure Deck :=
  (total_cards : ℕ)
  (standard_cards : ℕ)
  (jokers : ℕ)
  (h_total : total_cards = standard_cards + jokers)
  (h_jokers : jokers = 2)

/-- The probability of drawing a joker from the top of a shuffled deck -/
def prob_top_joker (d : Deck) : ℚ :=
  d.jokers / d.total_cards

/-- Theorem stating the probability of drawing a joker from a modified 54-card deck -/
theorem prob_top_joker_modified_deck :
  ∃ (d : Deck), d.total_cards = 54 ∧ d.standard_cards = 52 ∧ prob_top_joker d = 1 / 27 := by
  sorry


end NUMINAMATH_CALUDE_prob_top_joker_modified_deck_l1180_118055


namespace NUMINAMATH_CALUDE_students_liking_both_mountains_and_sea_l1180_118085

/-- Given a school with the following properties:
  * There are 500 total students
  * 289 students like mountains
  * 337 students like the sea
  * 56 students like neither mountains nor the sea
  Then the number of students who like both mountains and the sea is 182. -/
theorem students_liking_both_mountains_and_sea 
  (total : ℕ) 
  (like_mountains : ℕ) 
  (like_sea : ℕ) 
  (like_neither : ℕ) 
  (h1 : total = 500)
  (h2 : like_mountains = 289)
  (h3 : like_sea = 337)
  (h4 : like_neither = 56) :
  like_mountains + like_sea - (total - like_neither) = 182 := by
  sorry

end NUMINAMATH_CALUDE_students_liking_both_mountains_and_sea_l1180_118085


namespace NUMINAMATH_CALUDE_distribute_nine_balls_three_boxes_l1180_118033

/-- The number of ways to distribute n identical balls into k distinct boxes -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute n identical balls into k distinct boxes,
    where each box contains at least one ball -/
def distributeAtLeastOne (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute n identical balls into k distinct boxes,
    where each box contains at least one ball and the number of balls in each box is different -/
def distributeAtLeastOneDifferent (n k : ℕ) : ℕ := sorry

/-- Theorem: The number of ways to distribute 9 identical balls into 3 distinct boxes,
    where each box contains at least one ball and the number of balls in each box is different, is 18 -/
theorem distribute_nine_balls_three_boxes : distributeAtLeastOneDifferent 9 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_distribute_nine_balls_three_boxes_l1180_118033


namespace NUMINAMATH_CALUDE_range_of_a_l1180_118051

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 ≠ 0

def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (3 - 2*a)^x < (3 - 2*a)^y

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → a ∈ Set.Iic (-2) ∪ Set.Icc 1 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1180_118051


namespace NUMINAMATH_CALUDE_expression_evaluation_l1180_118097

theorem expression_evaluation (a b c : ℝ) 
  (h1 : a = 2)
  (h2 : b = a + 4)
  (h3 : c = b - 20)
  (h4 : a^2 + a ≠ 0)
  (h5 : b^2 - 6*b + 8 ≠ 0)
  (h6 : c^2 + 12*c + 36 ≠ 0) :
  (a^2 + 2*a) / (a^2 + a) * (b^2 - 4) / (b^2 - 6*b + 8) * (c^2 + 16*c + 64) / (c^2 + 12*c + 36) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1180_118097


namespace NUMINAMATH_CALUDE_water_consumption_equation_l1180_118093

theorem water_consumption_equation (x : ℝ) (h : x > 0) : 
  (80 / x) - (80 * (1 - 0.2) / x) = 5 ↔ 
  (80 / x) - (80 / (x / (1 - 0.2))) = 5 :=
sorry

end NUMINAMATH_CALUDE_water_consumption_equation_l1180_118093


namespace NUMINAMATH_CALUDE_commercial_fraction_l1180_118037

theorem commercial_fraction (num_programs : ℕ) (program_duration : ℕ) (commercial_time : ℕ) :
  num_programs = 6 →
  program_duration = 30 →
  commercial_time = 45 →
  (commercial_time : ℚ) / (num_programs * program_duration : ℚ) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_commercial_fraction_l1180_118037


namespace NUMINAMATH_CALUDE_polynomial_factor_l1180_118028

-- Define the polynomial
def P (a b d x : ℝ) : ℝ := a * x^4 + b * x^3 + 27 * x^2 + d * x + 10

-- Define the factor
def F (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 2

-- Theorem statement
theorem polynomial_factor (a b d : ℝ) :
  (∃ (e f : ℝ), ∀ x, P a b d x = F x * (e * x^2 + f * x + 5)) →
  a = 2 ∧ b = -13 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_l1180_118028


namespace NUMINAMATH_CALUDE_line_intersects_circle_r_range_l1180_118094

/-- The range of r for a line intersecting a circle -/
theorem line_intersects_circle_r_range (α : Real) (r : Real) :
  (∃ x y : Real, x * Real.cos α + y * Real.sin α = 1 ∧ x^2 + y^2 = r^2) →
  r > 0 →
  r > 1 := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_circle_r_range_l1180_118094


namespace NUMINAMATH_CALUDE_remainder_theorem_l1180_118027

theorem remainder_theorem (n : ℤ) (h : n % 9 = 3) : (5 * n - 12) % 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1180_118027


namespace NUMINAMATH_CALUDE_expected_socks_is_2n_l1180_118076

/-- The expected number of socks taken until a pair is found -/
def expected_socks (n : ℕ) : ℝ :=
  2 * n

/-- Theorem stating that the expected number of socks taken until a pair is found is 2n -/
theorem expected_socks_is_2n (n : ℕ) (h : n > 0) :
  expected_socks n = 2 * n := by
  sorry

end NUMINAMATH_CALUDE_expected_socks_is_2n_l1180_118076


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_length_l1180_118047

/-- An isosceles triangle with perimeter 53 and base 11 has equal sides of length 21 -/
theorem isosceles_triangle_side_length : 
  ∀ (x : ℝ), 
  x > 0 → -- Ensure positive side length
  x + x + 11 = 53 → -- Perimeter condition
  x = 21 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_length_l1180_118047


namespace NUMINAMATH_CALUDE_hundredthOddPositiveInteger_l1180_118012

/-- The nth odd positive integer -/
def nthOddPositiveInteger (n : ℕ) : ℕ := 2 * n - 1

/-- The 100th odd positive integer is 199 -/
theorem hundredthOddPositiveInteger : nthOddPositiveInteger 100 = 199 := by
  sorry

end NUMINAMATH_CALUDE_hundredthOddPositiveInteger_l1180_118012


namespace NUMINAMATH_CALUDE_candy_soda_price_before_increase_l1180_118026

/-- Proves that the total price of a candy box and a soda can before a price increase is 16 pounds, given their initial prices and percentage increases. -/
theorem candy_soda_price_before_increase 
  (candy_price : ℝ) 
  (soda_price : ℝ) 
  (candy_increase : ℝ) 
  (soda_increase : ℝ) 
  (h1 : candy_price = 10) 
  (h2 : soda_price = 6) 
  (h3 : candy_increase = 0.25) 
  (h4 : soda_increase = 0.50) : 
  candy_price + soda_price = 16 := by
  sorry

#check candy_soda_price_before_increase

end NUMINAMATH_CALUDE_candy_soda_price_before_increase_l1180_118026


namespace NUMINAMATH_CALUDE_green_square_area_percentage_l1180_118058

/-- Represents a square flag with a symmetrical cross -/
structure FlagWithCross where
  side : ℝ
  crossWidth : ℝ
  crossArea : ℝ
  greenSquareArea : ℝ

/-- Properties of the flag with cross -/
def FlagWithCross.properties (flag : FlagWithCross) : Prop :=
  flag.side > 0 ∧
  flag.crossWidth > 0 ∧
  flag.crossWidth < flag.side / 2 ∧
  flag.crossArea = 0.49 * flag.side * flag.side ∧
  flag.greenSquareArea = (flag.side - 2 * flag.crossWidth) * (flag.side - 2 * flag.crossWidth)

/-- Theorem: If the cross occupies 49% of the flag area, the green square occupies 25% -/
theorem green_square_area_percentage (flag : FlagWithCross) 
  (h : flag.properties) : flag.greenSquareArea = 0.25 * flag.side * flag.side := by
  sorry


end NUMINAMATH_CALUDE_green_square_area_percentage_l1180_118058


namespace NUMINAMATH_CALUDE_max_value_a4a8_l1180_118069

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∀ n, a n > 0 ∧ a (n + 1) = r * a n

theorem max_value_a4a8 (a : ℕ → ℝ) (h : GeometricSequence a) 
    (h_cond : a 2 * a 6 + a 5 * a 11 = 16) : 
    (∀ x, a 4 * a 8 ≤ x → x = 8) :=
  sorry

end NUMINAMATH_CALUDE_max_value_a4a8_l1180_118069


namespace NUMINAMATH_CALUDE_sum_of_polynomials_l1180_118050

/-- Given polynomials f, g, and h, prove their sum is equal to the specified polynomial -/
theorem sum_of_polynomials (x : ℝ) : 
  let f := fun (x : ℝ) => -3*x^2 + x - 4
  let g := fun (x : ℝ) => -5*x^2 + 3*x - 8
  let h := fun (x : ℝ) => 5*x^2 + 5*x + 1
  f x + g x + h x = -3*x^2 + 9*x - 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_polynomials_l1180_118050


namespace NUMINAMATH_CALUDE_equation_c_is_linear_l1180_118071

/-- Definition of a linear equation with one variable -/
def is_linear_one_var (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The equation 2x - 3 = 5 -/
def f (x : ℝ) : ℝ := 2 * x - 3

theorem equation_c_is_linear : is_linear_one_var f :=
sorry

end NUMINAMATH_CALUDE_equation_c_is_linear_l1180_118071


namespace NUMINAMATH_CALUDE_interest_rate_is_six_percent_l1180_118001

/-- Calculates the interest rate given the principal, time, and interest amount -/
def calculate_interest_rate (principal : ℚ) (time : ℚ) (interest : ℚ) : ℚ :=
  (interest * 100) / (principal * time)

theorem interest_rate_is_six_percent 
  (principal : ℚ) 
  (time : ℚ) 
  (interest : ℚ) 
  (h1 : principal = 1050)
  (h2 : time = 6)
  (h3 : interest = principal - 672) :
  calculate_interest_rate principal time interest = 6 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_is_six_percent_l1180_118001
