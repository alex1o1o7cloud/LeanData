import Mathlib

namespace log_inequality_inequality_with_roots_l1148_114862

-- Theorem 1
theorem log_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.log ((a + b) / 2) ≥ (Real.log a + Real.log b) / 2 := by
  sorry

-- Theorem 2
theorem inequality_with_roots :
  6 + Real.sqrt 10 > 2 * Real.sqrt 3 + 2 := by
  sorry

end log_inequality_inequality_with_roots_l1148_114862


namespace community_average_age_l1148_114849

/-- Given a community with a ratio of women to men of 13:10, where the average age of women
    is 36 years and the average age of men is 31 years, prove that the average age of the
    community is 33 19/23 years. -/
theorem community_average_age
  (ratio_women_men : ℚ)
  (avg_age_women : ℚ)
  (avg_age_men : ℚ)
  (h_ratio : ratio_women_men = 13 / 10)
  (h_women_age : avg_age_women = 36)
  (h_men_age : avg_age_men = 31) :
  (ratio_women_men * avg_age_women + avg_age_men) / (ratio_women_men + 1) = 33 + 19 / 23 :=
by sorry

end community_average_age_l1148_114849


namespace product_abcd_equals_1280_l1148_114865

theorem product_abcd_equals_1280 
  (a b c d : ℝ) 
  (eq1 : 2*a + 4*b + 6*c + 8*d = 48)
  (eq2 : 4*d + 2*c = 2*b)
  (eq3 : 4*b + 2*c = 2*a)
  (eq4 : c - 2 = d)
  (eq5 : d + b = 10) :
  a * b * c * d = 1280 := by
sorry

end product_abcd_equals_1280_l1148_114865


namespace linear_system_solution_l1148_114811

theorem linear_system_solution : 
  ∀ (x y : ℝ), 
    (2 * x + 3 * y = 4) → 
    (x = -y) → 
    (x = -4 ∧ y = 4) := by
  sorry

end linear_system_solution_l1148_114811


namespace moles_of_products_l1148_114801

-- Define the molar mass of Ammonium chloride
def molar_mass_NH4Cl : ℝ := 53.50

-- Define the mass of Ammonium chloride used
def mass_NH4Cl : ℝ := 53

-- Define the number of moles of Potassium hydroxide
def moles_KOH : ℝ := 1

-- Define the reaction ratio (1:1:1:1)
def reaction_ratio : ℝ := 1

-- Theorem stating the number of moles of products formed
theorem moles_of_products (ε : ℝ) (h_ε : ε > 0) :
  ∃ (moles_product : ℝ),
    moles_product > 0 ∧
    abs (moles_product - (mass_NH4Cl / molar_mass_NH4Cl)) < ε ∧
    moles_product * reaction_ratio = (mass_NH4Cl / molar_mass_NH4Cl) * reaction_ratio :=
by sorry

end moles_of_products_l1148_114801


namespace simplify_expression_1_simplify_expression_2_l1148_114834

-- First expression
theorem simplify_expression_1 (a b : ℝ) :
  -b * (2 * a - b) + (a + b)^2 = a^2 + 2 * b^2 := by sorry

-- Second expression
theorem simplify_expression_2 (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) :
  (1 - x / (2 + x)) / ((x^2 - 4) / (x^2 + 4*x + 4)) = 2 / (x - 2) := by sorry

end simplify_expression_1_simplify_expression_2_l1148_114834


namespace angle_between_vectors_l1148_114872

/-- The angle between two vectors in R² -/
def angle (v w : ℝ × ℝ) : ℝ := sorry

theorem angle_between_vectors (a b : ℝ × ℝ) (h1 : a = (1, 1)) (h2 : b = (-1, 2)) :
  angle (a + b) a = π / 4 := by sorry

end angle_between_vectors_l1148_114872


namespace inscribed_circles_distance_l1148_114866

-- Define the triangle
def Triangle (X Y Z : ℝ × ℝ) : Prop :=
  (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = 100^2 ∧
  (X.1 - Z.1)^2 + (X.2 - Z.2)^2 = 160^2 ∧
  (Y.1 - Z.1)^2 + (Y.2 - Z.2)^2 = 200^2

-- Define a right angle
def RightAngle (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

-- Define a line perpendicular to another line
def Perpendicular (A B C D : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (D.1 - C.1) + (B.2 - A.2) * (D.2 - C.2) = 0

-- Define the inscribed circle
def InscribedCircle (C : ℝ × ℝ) (r : ℝ) (A B D : ℝ × ℝ) : Prop :=
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = r^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = r^2 ∧
  (C.1 - D.1)^2 + (C.2 - D.2)^2 = r^2

theorem inscribed_circles_distance 
  (X Y Z M N O P : ℝ × ℝ) 
  (C₁ C₂ C₃ : ℝ × ℝ) 
  (r₁ r₂ r₃ : ℝ) :
  Triangle X Y Z →
  RightAngle X Y Z →
  Perpendicular X Z M N →
  Perpendicular X Y O P →
  InscribedCircle C₁ r₁ X Y Z →
  InscribedCircle C₂ r₂ Z M N →
  InscribedCircle C₃ r₃ Y O P →
  (C₂.1 - C₃.1)^2 + (C₂.2 - C₃.2)^2 = 26000 :=
sorry

end inscribed_circles_distance_l1148_114866


namespace largest_constant_inequality_l1148_114878

theorem largest_constant_inequality (x y z : ℝ) :
  ∃ (C : ℝ), (∀ (a b c : ℝ), a^2 + b^2 + c^2 + 1 ≥ C * (a + b + c)) ∧
  (C = 2 / Real.sqrt 3) ∧
  (∀ (D : ℝ), (∀ (a b c : ℝ), a^2 + b^2 + c^2 + 1 ≥ D * (a + b + c)) → D ≤ C) :=
by sorry

end largest_constant_inequality_l1148_114878


namespace floor_product_equals_49_l1148_114841

theorem floor_product_equals_49 (x : ℝ) :
  ⌊x * ⌊x⌋⌋ = 49 ↔ 7 ≤ x ∧ x < 50 / 7 := by
sorry

end floor_product_equals_49_l1148_114841


namespace mod_equiv_unique_solution_l1148_114882

theorem mod_equiv_unique_solution : 
  ∃! n : ℤ, 0 ≤ n ∧ n < 9 ∧ -2357 ≡ n [ZMOD 9] :=
by sorry

end mod_equiv_unique_solution_l1148_114882


namespace hyperbola_asymptote_l1148_114818

/-- Given a hyperbola C with equation x²/a² - y²/b² = 1 (a > 0, b > 0),
    a circle Ω with the real axis of C as its diameter,
    and P the intersection point of Ω and the asymptote of C in the first quadrant,
    if the slope of FP (where F is the right focus of C) is -b/a,
    then the equation of the asymptote of C is y = ±x -/
theorem hyperbola_asymptote (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  let F := (Real.sqrt (a^2 + b^2), 0)
  let Ω := {(x, y) : ℝ × ℝ | x^2 + y^2 = a^2}
  let P := (a / Real.sqrt 2, a / Real.sqrt 2)
  (P.2 - F.2) / (P.1 - F.1) = -b / a →
  ∀ (x y : ℝ), (x, y) ∈ {(x, y) : ℝ × ℝ | y = x ∨ y = -x} ↔
    ∃ (t : ℝ), t ≠ 0 ∧ x = a * t ∧ y = b * t :=
by sorry

end hyperbola_asymptote_l1148_114818


namespace sequence_equality_l1148_114860

theorem sequence_equality (a : Fin 100 → ℝ) 
  (h1 : ∀ n : Fin 98, a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0)
  (h2 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
  (h3 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
  (h4 : a 10 = 10) :
  a 22 = 10 := by sorry

end sequence_equality_l1148_114860


namespace bobby_cars_after_seven_years_l1148_114826

def initial_cars : ℕ := 30

def double (n : ℕ) : ℕ := 2 * n

def donate (n : ℕ) : ℕ := n - (n / 10)

def update_cars (year : ℕ) (cars : ℕ) : ℕ :=
  if year % 2 = 0 ∧ year ≠ 0 then
    donate (double cars)
  else
    double cars

def cars_after_years (years : ℕ) : ℕ :=
  match years with
  | 0 => initial_cars
  | n + 1 => update_cars n (cars_after_years n)

theorem bobby_cars_after_seven_years :
  cars_after_years 7 = 2792 := by sorry

end bobby_cars_after_seven_years_l1148_114826


namespace students_in_both_chorus_and_band_l1148_114829

theorem students_in_both_chorus_and_band :
  ∀ (total chorus band neither both : ℕ),
    total = 50 →
    chorus = 18 →
    band = 26 →
    neither = 8 →
    total = chorus + band - both + neither →
    both = 2 :=
by
  sorry

end students_in_both_chorus_and_band_l1148_114829


namespace Q_bounds_l1148_114844

/-- The equation of the given curve -/
def curve_equation (x y : ℝ) : Prop :=
  |5 * x + y| + |5 * x - y| = 20

/-- The expression we want to bound -/
def Q (x y : ℝ) : ℝ :=
  x^2 - x*y + y^2

/-- Theorem stating the bounds of Q for points on the curve -/
theorem Q_bounds :
  ∀ x y : ℝ, curve_equation x y → 3 ≤ Q x y ∧ Q x y ≤ 124 :=
by sorry

end Q_bounds_l1148_114844


namespace usual_time_to_school_l1148_114830

theorem usual_time_to_school (usual_rate : ℝ) (usual_time : ℝ) : 
  usual_rate > 0 →
  usual_time > 0 →
  (5/4 * usual_rate) * (usual_time - 4) = usual_rate * usual_time →
  usual_time = 20 := by
  sorry

end usual_time_to_school_l1148_114830


namespace principal_value_range_of_argument_l1148_114816

theorem principal_value_range_of_argument (z : ℂ) (h : Complex.abs z = 1) :
  ∃ (k : ℕ) (θ : ℝ), k ≤ 1 ∧ 
  Complex.arg z = θ ∧
  k * Real.pi - Real.arccos (-1/2) ≤ θ ∧ 
  θ ≤ k * Real.pi + Real.arccos (-1/2) :=
by sorry

end principal_value_range_of_argument_l1148_114816


namespace power_of_negative_cube_l1148_114883

theorem power_of_negative_cube (x : ℝ) : (-x^3)^2 = x^6 := by
  sorry

end power_of_negative_cube_l1148_114883


namespace angle_trigonometric_identity_l1148_114885

theorem angle_trigonometric_identity (α : Real) (m n : Real) : 
  -- Conditions
  α ∈ Set.Icc 0 π ∧ 
  m^2 + n^2 = 1 ∧ 
  n / m = -2 →
  -- Conclusion
  2 * Real.sin α * Real.cos α - Real.cos α ^ 2 = -1 := by
  sorry

end angle_trigonometric_identity_l1148_114885


namespace balloon_count_l1148_114804

/-- Represents the balloon shooting game with two levels. -/
structure BalloonGame where
  /-- The number of balloons missed in the first level -/
  missed_first_level : ℕ
  /-- The total number of balloons in each level -/
  total_balloons : ℕ

/-- The conditions of the balloon shooting game -/
def game_conditions (game : BalloonGame) : Prop :=
  let hit_first_level := 4 * game.missed_first_level + 2
  let hit_second_level := hit_first_level + 8
  hit_second_level = 6 * game.missed_first_level ∧
  game.total_balloons = hit_first_level + game.missed_first_level

/-- The theorem stating that the number of balloons in each level is 147 -/
theorem balloon_count (game : BalloonGame) 
  (h : game_conditions game) : game.total_balloons = 147 := by
  sorry

end balloon_count_l1148_114804


namespace kitchen_hours_theorem_l1148_114871

/-- The minimum number of hours required to produce a given number of large and small cakes -/
def min_hours_required (num_helpers : ℕ) (large_cakes_per_hour : ℕ) (small_cakes_per_hour : ℕ) (large_cakes_needed : ℕ) (small_cakes_needed : ℕ) : ℕ :=
  max 
    (large_cakes_needed / (num_helpers * large_cakes_per_hour))
    (small_cakes_needed / (num_helpers * small_cakes_per_hour))

theorem kitchen_hours_theorem :
  min_hours_required 10 2 35 20 700 = 2 := by
  sorry

end kitchen_hours_theorem_l1148_114871


namespace cos_squared_minus_three_sin_cos_angle_in_second_quadrant_l1148_114814

-- Problem 1
theorem cos_squared_minus_three_sin_cos (m : ℝ) (α : ℝ) (h : m ≠ 0) :
  let P : ℝ × ℝ := (m, 3 * m)
  (Real.cos α)^2 - 3 * (Real.sin α) * (Real.cos α) = -4/5 := by sorry

-- Problem 2
theorem angle_in_second_quadrant (θ : ℝ) (a : ℝ) 
  (h1 : Real.sin θ = (1 - a) / (1 + a))
  (h2 : Real.cos θ = (3 * a - 1) / (1 + a))
  (h3 : 0 < Real.sin θ ∧ Real.cos θ < 0) :
  a = 1/9 := by sorry

end cos_squared_minus_three_sin_cos_angle_in_second_quadrant_l1148_114814


namespace quadratic_equation_coefficient_l1148_114887

theorem quadratic_equation_coefficient :
  ∀ a b c : ℝ,
  (∀ x : ℝ, 2 * x^2 = 9 * x + 8) →
  (a * x^2 + b * x + c = 0 ↔ 2 * x^2 - 9 * x - 8 = 0) →
  a = 2 →
  b = -9 :=
by sorry

end quadratic_equation_coefficient_l1148_114887


namespace product_remainder_mod_five_l1148_114840

theorem product_remainder_mod_five : (1234 * 5678 * 9012) % 5 = 4 := by
  sorry

end product_remainder_mod_five_l1148_114840


namespace proposition_A_necessary_not_sufficient_l1148_114822

/-- Proposition A: The inequality x^2 + 2ax + 4 ≤ 0 has solutions -/
def proposition_A (a : ℝ) : Prop := a ≤ -2 ∨ a ≥ 2

/-- Proposition B: The function f(x) = log_a(x + a - 2) is always positive on the interval (1, +∞) -/
def proposition_B (a : ℝ) : Prop := a ≥ 2

theorem proposition_A_necessary_not_sufficient :
  (∀ a : ℝ, proposition_B a → proposition_A a) ∧
  ¬(∀ a : ℝ, proposition_A a → proposition_B a) := by
  sorry

end proposition_A_necessary_not_sufficient_l1148_114822


namespace hyperbola_from_circle_intersection_l1148_114803

/-- Circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 9 = 0

/-- Points A and B on y-axis -/
def points_on_y_axis (A B : ℝ × ℝ) : Prop :=
  A.1 = 0 ∧ B.1 = 0 ∧ circle_eq A.1 A.2 ∧ circle_eq B.1 B.2

/-- Points A and B trisect focal distance -/
def trisect_focal_distance (A B : ℝ × ℝ) (c : ℝ) : Prop :=
  abs (A.2 - B.2) = 2 * c / 3

/-- Standard hyperbola equation -/
def hyperbola_eq (x y : ℝ) : Prop := y^2/9 - x^2/72 = 1

/-- Main theorem -/
theorem hyperbola_from_circle_intersection :
  ∀ (A B : ℝ × ℝ) (c : ℝ),
  points_on_y_axis A B →
  trisect_focal_distance A B c →
  ∀ (x y : ℝ), hyperbola_eq x y :=
sorry

end hyperbola_from_circle_intersection_l1148_114803


namespace washington_high_ratio_l1148_114813

/-- The student-teacher ratio at Washington High School -/
def student_teacher_ratio (num_students : ℕ) (num_teachers : ℕ) : ℚ :=
  num_students / num_teachers

/-- Theorem: The student-teacher ratio at Washington High School is 27.5 to 1 -/
theorem washington_high_ratio :
  student_teacher_ratio 1155 42 = 27.5 := by
  sorry

end washington_high_ratio_l1148_114813


namespace max_receptivity_receptivity_comparison_no_continuous_high_receptivity_l1148_114856

-- Define the piecewise function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 10 then -0.1 * x^2 + 2.6 * x + 44
  else if 10 < x ∧ x ≤ 15 then 60
  else if 15 < x ∧ x ≤ 25 then -3 * x + 105
  else if 25 < x ∧ x ≤ 40 then 30
  else 0  -- Define a default value for x outside the given ranges

-- Theorem statements
theorem max_receptivity (x : ℝ) :
  (∀ x, f x ≤ 60) ∧
  (f 10 = 60) ∧
  (∀ x, 10 < x → x ≤ 15 → f x = 60) :=
sorry

theorem receptivity_comparison :
  f 5 > f 20 ∧ f 20 > f 35 :=
sorry

theorem no_continuous_high_receptivity :
  ¬ ∃ a b : ℝ, b - a = 12 ∧ ∀ x, a ≤ x ∧ x ≤ b → f x ≥ 56 :=
sorry

end max_receptivity_receptivity_comparison_no_continuous_high_receptivity_l1148_114856


namespace furniture_production_l1148_114831

theorem furniture_production (total_wood : ℕ) (table_wood : ℕ) (chair_wood : ℕ) (tables_made : ℕ) :
  total_wood = 672 →
  table_wood = 12 →
  chair_wood = 8 →
  tables_made = 24 →
  (total_wood - tables_made * table_wood) / chair_wood = 48 :=
by sorry

end furniture_production_l1148_114831


namespace solve_inequality_max_a_value_l1148_114851

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| + |x + 1|

-- Theorem for part I
theorem solve_inequality :
  ∀ x : ℝ, f x > 4 ↔ x < -1.5 ∨ x > 2.5 := by sorry

-- Theorem for part II
theorem max_a_value :
  ∃ a : ℝ, (∀ x : ℝ, f x ≥ a) ∧ (∀ b : ℝ, (∀ x : ℝ, f x ≥ b) → b ≤ a) ∧ a = 3 := by sorry

end solve_inequality_max_a_value_l1148_114851


namespace largest_prime_divisor_of_base_6_number_l1148_114825

/-- Represents the number 100111001 in base 6 -/
def base_6_number : ℕ := 6^8 + 6^5 + 6^4 + 6^3 + 6 + 1

/-- The largest prime divisor of base_6_number -/
def largest_prime_divisor : ℕ := 43

theorem largest_prime_divisor_of_base_6_number :
  (∀ p : ℕ, Prime p → p ∣ base_6_number → p ≤ largest_prime_divisor) ∧
  (Prime largest_prime_divisor ∧ largest_prime_divisor ∣ base_6_number) := by
  sorry

end largest_prime_divisor_of_base_6_number_l1148_114825


namespace beach_trip_result_l1148_114861

/-- Represents the number of seashells found during a beach trip -/
def beach_trip (days : ℕ) (shells_per_day : ℕ) : ℕ :=
  days * shells_per_day

/-- Proves that a 5-day beach trip with 7 shells found per day results in 35 total shells -/
theorem beach_trip_result : beach_trip 5 7 = 35 := by
  sorry

end beach_trip_result_l1148_114861


namespace license_plate_theorem_l1148_114874

def num_letters : Nat := 26
def num_letter_positions : Nat := 4
def num_digit_positions : Nat := 3

def license_plate_combinations : Nat :=
  Nat.choose num_letters 2 *
  Nat.choose num_letter_positions 2 *
  2 *
  (10 * 9 * 8)

theorem license_plate_theorem :
  license_plate_combinations = 2808000 := by
  sorry

end license_plate_theorem_l1148_114874


namespace geometric_series_sum_l1148_114899

/-- The sum of the infinite geometric series 4/3 - 1/2 + 3/32 - 9/256 + ... -/
theorem geometric_series_sum : 
  let a : ℚ := 4/3  -- first term
  let r : ℚ := -3/8 -- common ratio
  let S := Nat → ℚ  -- sequence type
  let series : S := fun n => a * r^n  -- geometric series
  ∑' n, series n = 32/33 := by
  sorry

end geometric_series_sum_l1148_114899


namespace power_of_two_equality_l1148_114812

theorem power_of_two_equality (a b : ℕ+) (h : 2^(a.val) * 2^(b.val) = 8) : 
  (2^(a.val))^(b.val) = 4 := by
  sorry

end power_of_two_equality_l1148_114812


namespace goldfish_disappeared_l1148_114806

theorem goldfish_disappeared (original : ℕ) (left : ℕ) (disappeared : ℕ) : 
  original = 15 → left = 4 → disappeared = original - left → disappeared = 11 := by
  sorry

end goldfish_disappeared_l1148_114806


namespace absolute_value_equation_solutions_l1148_114875

theorem absolute_value_equation_solutions (m n k : ℝ) : 
  (∀ x : ℝ, |2*x - 3| + m ≠ 0) →
  (∃! x : ℝ, |3*x - 4| + n = 0) →
  (∃ x y : ℝ, x ≠ y ∧ |4*x - 5| + k = 0 ∧ |4*y - 5| + k = 0) →
  m > n ∧ n > k :=
sorry

end absolute_value_equation_solutions_l1148_114875


namespace center_of_hyperbola_l1148_114809

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  9 * x^2 - 54 * x - 36 * y^2 + 360 * y - 891 = 0

-- Define the center of a hyperbola
def is_center (c : ℝ × ℝ) (eq : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b : ℝ), ∀ (x y : ℝ),
    eq x y ↔ (x - c.1)^2 / a^2 - (y - c.2)^2 / b^2 = 1

-- Theorem stating that (3, 5) is the center of the given hyperbola
theorem center_of_hyperbola :
  is_center (3, 5) hyperbola_eq :=
sorry

end center_of_hyperbola_l1148_114809


namespace tan_4305_degrees_l1148_114892

theorem tan_4305_degrees : Real.tan (4305 * π / 180) = -2 + Real.sqrt 3 := by
  sorry

end tan_4305_degrees_l1148_114892


namespace weekend_rain_probability_l1148_114823

def prob_rain_friday : ℝ := 0.30
def prob_rain_saturday : ℝ := 0.45
def prob_rain_sunday : ℝ := 0.55

theorem weekend_rain_probability : 
  let prob_no_rain_friday := 1 - prob_rain_friday
  let prob_no_rain_saturday := 1 - prob_rain_saturday
  let prob_no_rain_sunday := 1 - prob_rain_sunday
  let prob_no_rain_weekend := prob_no_rain_friday * prob_no_rain_saturday * prob_no_rain_sunday
  1 - prob_no_rain_weekend = 0.82675 := by
sorry

end weekend_rain_probability_l1148_114823


namespace tile_border_ratio_l1148_114857

theorem tile_border_ratio (n : ℕ) (s d : ℝ) (h1 : n = 24) 
  (h2 : (n^2 : ℝ) * s^2 * 0.64 = 576 * s^2) : d / s = 6 / 25 := by
  sorry

end tile_border_ratio_l1148_114857


namespace boat_speed_ratio_l1148_114850

theorem boat_speed_ratio (b r : ℝ) (h1 : b > 0) (h2 : r > 0) 
  (h3 : (b - r)⁻¹ = 2 * (b + r)⁻¹) 
  (s1 s2 : ℝ) (h4 : s1 > 0) (h5 : s2 > 0)
  (h6 : b * (1/4) + b * (3/4) = b) :
  b / (s1 + s2) = 3 / 1 := by
sorry

end boat_speed_ratio_l1148_114850


namespace base_eight_representation_l1148_114839

-- Define the representation function
def represent (base : ℕ) (n : ℕ) : ℕ := 
  3 * base^4 + 0 * base^3 + 4 * base^2 + 0 * base + 7

-- Define the theorem
theorem base_eight_representation : 
  ∃ (base : ℕ), base > 1 ∧ represent base 12551 = 30407 ∧ base = 8 := by
  sorry

end base_eight_representation_l1148_114839


namespace meetings_count_is_four_l1148_114886

/-- Represents the meeting problem between Michael and the garbage truck --/
structure MeetingProblem where
  michael_speed : ℝ
  pail_distance : ℝ
  truck_speed : ℝ
  truck_stop_time : ℝ

/-- Calculates the number of times Michael and the truck meet --/
def number_of_meetings (p : MeetingProblem) : ℕ :=
  sorry

/-- The main theorem stating that the number of meetings is 4 --/
theorem meetings_count_is_four :
  ∀ (p : MeetingProblem),
    p.michael_speed = 6 ∧
    p.pail_distance = 150 ∧
    p.truck_speed = 12 ∧
    p.truck_stop_time = 20 →
    number_of_meetings p = 4 :=
  sorry

end meetings_count_is_four_l1148_114886


namespace clock_right_angle_time_l1148_114855

/-- The time (in minutes) between two consecutive instances of the clock hands forming a right angle after 7 PM -/
def time_between_right_angles : ℚ := 360 / 11

/-- The angle (in degrees) that the minute hand moves relative to the hour hand between two consecutive right angle formations -/
def relative_angle_change : ℚ := 180

theorem clock_right_angle_time :
  time_between_right_angles = 360 / 11 :=
by sorry

end clock_right_angle_time_l1148_114855


namespace fidos_yard_l1148_114808

theorem fidos_yard (r : ℝ) (h : r > 0) : 
  let circle_area := π * r^2
  let hexagon_area := 3 * r^2 * Real.sqrt 3 / 2
  let ratio := circle_area / hexagon_area
  ratio = Real.sqrt 3 * π / 6 ∧ 3 * 6 = 18 := by sorry

end fidos_yard_l1148_114808


namespace unique_poly_pair_l1148_114835

/-- A polynomial of degree 3 -/
def Poly3 (R : Type*) [CommRing R] := R → R

/-- The evaluation of a polynomial at a point -/
def eval (p : Poly3 ℝ) (x : ℝ) : ℝ := p x

/-- The composition of two polynomials -/
def comp (p q : Poly3 ℝ) : Poly3 ℝ := λ x ↦ p (q x)

/-- The cube of a polynomial -/
def cube (p : Poly3 ℝ) : Poly3 ℝ := λ x ↦ (p x)^3

theorem unique_poly_pair (f g : Poly3 ℝ) 
  (h1 : f ≠ g)
  (h2 : ∀ x, eval (comp f f) x = eval (cube g) x)
  (h3 : ∀ x, eval (comp f g) x = eval (cube f) x)
  (h4 : eval f 0 = 1) :
  (∀ x, f x = (1 - x)^3) ∧ (∀ x, g x = (x - 1)^3 + 1) := by
  sorry


end unique_poly_pair_l1148_114835


namespace negative_x_count_l1148_114848

theorem negative_x_count : 
  ∃ (S : Finset ℤ), 
    (∀ x ∈ S, x < 0 ∧ ∃ n : ℕ+, (x + 196 : ℝ) = n^2) ∧ 
    (∀ x : ℤ, x < 0 → (∃ n : ℕ+, (x + 196 : ℝ) = n^2) → x ∈ S) ∧
    Finset.card S = 13 := by
  sorry

end negative_x_count_l1148_114848


namespace order_of_mnpq_l1148_114867

theorem order_of_mnpq (m n p q : ℝ) 
  (h1 : m < n) 
  (h2 : p < q) 
  (h3 : (p - m) * (p - n) < 0) 
  (h4 : (q - m) * (q - n) < 0) : 
  m < p ∧ p < q ∧ q < n := by
  sorry

end order_of_mnpq_l1148_114867


namespace flood_damage_conversion_l1148_114884

/-- Conversion of flood damage from Canadian to American dollars -/
theorem flood_damage_conversion (damage_cad : ℝ) (exchange_rate : ℝ) 
  (h1 : damage_cad = 50000000)
  (h2 : exchange_rate = 1.25)
  : damage_cad / exchange_rate = 40000000 := by
  sorry

end flood_damage_conversion_l1148_114884


namespace no_rain_time_l1148_114833

theorem no_rain_time (total_time rain_time : ℕ) (h1 : total_time = 8) (h2 : rain_time = 2) :
  total_time - rain_time = 6 := by
  sorry

end no_rain_time_l1148_114833


namespace all_multiples_contain_two_l1148_114889

def numbers : List ℕ := [418, 244, 816, 426, 24]

def containsTwo (n : ℕ) : Prop :=
  ∃ d, d ∈ n.digits 10 ∧ d = 2

theorem all_multiples_contain_two :
  ∀ n ∈ numbers, containsTwo (3 * n) :=
by sorry

end all_multiples_contain_two_l1148_114889


namespace solution_interval_l1148_114864

theorem solution_interval (f : ℝ → ℝ) (k : ℝ) : 
  (∃ x, f x = 0 ∧ k < x ∧ x < k + 1/2) →
  (∃ n : ℤ, k = n * 1/2) →
  (∀ x, f x = x^3 - 4 + x) →
  k = 1 := by
sorry

end solution_interval_l1148_114864


namespace perfect_square_condition_l1148_114891

/-- If 4x^2 + mxy + y^2 is a perfect square, then m = ±4 -/
theorem perfect_square_condition (x y m : ℝ) : 
  (∃ (k : ℝ), 4*x^2 + m*x*y + y^2 = k^2) → (m = 4 ∨ m = -4) := by
sorry

end perfect_square_condition_l1148_114891


namespace stratified_sampling_and_probability_l1148_114881

-- Define the total number of students
def total_students : ℕ := 350

-- Define the number of students excellent in Chinese
def excellent_chinese : ℕ := 200

-- Define the number of students excellent in English
def excellent_english : ℕ := 150

-- Define the probability of being excellent in both subjects
def prob_both_excellent : ℚ := 1 / 6

-- Define the number of students selected for the sample
def sample_size : ℕ := 6

-- Define the function to calculate the number of students in each category
def calculate_category_sizes : ℕ × ℕ × ℕ := sorry

-- Define the function to calculate the probability of selecting two students with excellent Chinese scores
def calculate_probability : ℚ := sorry

-- Theorem statement
theorem stratified_sampling_and_probability :
  let (a, b, c) := calculate_category_sizes
  (a = 3 ∧ b = 2 ∧ c = 1) ∧ calculate_probability = 2 / 5 := by sorry

end stratified_sampling_and_probability_l1148_114881


namespace quadratic_function_properties_l1148_114846

-- Define the quadratic function
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

-- State the theorem
theorem quadratic_function_properties :
  ∀ b c : ℝ,
  (∀ x ∈ Set.Ioo 2 3, f b c x ≤ 1) →  -- Maximum value of 1 in (2,3]
  (∃ x ∈ Set.Ioo 2 3, f b c x = 1) →  -- Maximum value of 1 is achieved in (2,3]
  (∀ x : ℝ, abs x > 2 → f b c x > 0) →  -- f(x) > 0 when |x| > 2
  (c = 4 → b = -4) ∧  -- Part 1: When c = 4, b = -4
  (Set.Icc (-34/7) (-15/4) = {x | ∃ b c : ℝ, b + 1/c = x}) -- Part 2: Range of b + 1/c
  := by sorry

end quadratic_function_properties_l1148_114846


namespace max_third_side_triangle_l1148_114807

theorem max_third_side_triangle (D E F : Real) (a b : Real) :
  -- Triangle DEF exists
  0 < D ∧ 0 < E ∧ 0 < F ∧ D + E + F = π →
  -- Two sides are 12 and 15
  a = 12 ∧ b = 15 →
  -- Angle condition
  Real.cos (2 * D) + Real.cos (2 * E) + Real.cos (2 * F) = 1 / 2 →
  -- Maximum length of third side
  ∃ c : Real, c ≤ Real.sqrt 549 ∧
    ∀ c' : Real, (c' = Real.sqrt (a^2 + b^2 - 2*a*b*(Real.cos E)) → c' ≤ c) :=
by sorry

end max_third_side_triangle_l1148_114807


namespace candy_ratio_l1148_114877

theorem candy_ratio (kitkat : ℕ) (nerds : ℕ) (lollipops : ℕ) (babyruths : ℕ) (reeses : ℕ) (remaining : ℕ) :
  kitkat = 5 →
  nerds = 8 →
  lollipops = 11 →
  babyruths = 10 →
  reeses = babyruths / 2 →
  remaining = 49 →
  ∃ (hershey : ℕ),
    hershey + kitkat + nerds + (lollipops - 5) + babyruths + reeses = remaining ∧
    hershey / kitkat = 3 :=
by
  sorry

end candy_ratio_l1148_114877


namespace economy_class_seats_count_l1148_114894

/-- Represents the seating configuration of an airplane -/
structure AirplaneSeating where
  first_class_seats : ℕ
  business_class_seats : ℕ
  economy_class_seats : ℕ
  first_class_occupied : ℕ
  business_class_occupied : ℕ

/-- Theorem stating the number of economy class seats in the given airplane configuration -/
theorem economy_class_seats_count (a : AirplaneSeating) 
  (h1 : a.first_class_seats = 10)
  (h2 : a.business_class_seats = 30)
  (h3 : a.first_class_occupied = 3)
  (h4 : a.business_class_occupied = 22)
  (h5 : a.first_class_occupied + a.business_class_occupied = a.economy_class_seats / 2)
  : a.economy_class_seats = 50 := by
  sorry

#check economy_class_seats_count

end economy_class_seats_count_l1148_114894


namespace perpendicular_lines_a_value_l1148_114868

theorem perpendicular_lines_a_value (a : ℝ) :
  (∀ x y : ℝ, (a * x + y - 1 = 0 ∧ x - y + 3 = 0) → 
   (a * 1 + (-1) * 1 = -1)) → a = 1 := by
  sorry

end perpendicular_lines_a_value_l1148_114868


namespace complex_number_modulus_l1148_114880

theorem complex_number_modulus (z : ℂ) (h : 1 + z = (1 - z) * Complex.I) : Complex.abs z = 1 := by
  sorry

end complex_number_modulus_l1148_114880


namespace weight_range_proof_l1148_114870

/-- Given the weights of Tracy, John, and Jake, prove that the range of their weights is 14 kg -/
theorem weight_range_proof (tracy_weight john_weight jake_weight : ℕ) 
  (h1 : tracy_weight + john_weight + jake_weight = 158)
  (h2 : tracy_weight = 52)
  (h3 : jake_weight = tracy_weight + 8) :
  (max tracy_weight (max john_weight jake_weight)) - 
  (min tracy_weight (min john_weight jake_weight)) = 14 := by
  sorry

#check weight_range_proof

end weight_range_proof_l1148_114870


namespace difference_numbers_between_500_and_600_l1148_114898

def is_difference_number (n : ℕ) : Prop :=
  n % 7 = 6 ∧ n % 5 = 4

theorem difference_numbers_between_500_and_600 :
  {n : ℕ | 500 < n ∧ n < 600 ∧ is_difference_number n} = {524, 559, 594} := by
  sorry

end difference_numbers_between_500_and_600_l1148_114898


namespace solution_to_system_of_equations_l1148_114805

theorem solution_to_system_of_equations :
  ∃ (x y : ℝ), 
    (1 / x + 1 / y = 1) ∧
    (2 / x + 3 / y = 4) ∧
    (x = -1) ∧
    (y = 1 / 2) := by
  sorry

end solution_to_system_of_equations_l1148_114805


namespace planes_parallel_l1148_114800

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (non_coincident : Line → Line → Prop)
variable (plane_non_coincident : Plane → Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel
  (l m : Line) (α β γ : Plane)
  (h1 : non_coincident l m)
  (h2 : plane_non_coincident α β γ)
  (h3 : perpendicular l α)
  (h4 : perpendicular m β)
  (h5 : parallel l m) :
  plane_parallel α β :=
sorry

end planes_parallel_l1148_114800


namespace target_hit_probability_l1148_114828

theorem target_hit_probability (p_a p_b p_c : ℚ) 
  (h_a : p_a = 1/2) 
  (h_b : p_b = 1/3) 
  (h_c : p_c = 1/4) : 
  1 - (1 - p_a) * (1 - p_b) * (1 - p_c) = 3/4 := by
  sorry

end target_hit_probability_l1148_114828


namespace total_fruit_punch_l1148_114896

def orange_punch : Real := 4.5
def cherry_punch : Real := 2 * orange_punch
def apple_juice : Real := cherry_punch - 1.5
def pineapple_juice : Real := 3
def grape_punch : Real := apple_juice + 0.5 * apple_juice

theorem total_fruit_punch :
  orange_punch + cherry_punch + apple_juice + pineapple_juice + grape_punch = 35.25 := by
  sorry

end total_fruit_punch_l1148_114896


namespace prime_product_sum_l1148_114810

theorem prime_product_sum (m n p : ℕ) : 
  Prime m ∧ Prime n ∧ Prime p ∧ m * n * p = 5 * (m + n + p) → m^2 + n^2 + p^2 = 78 :=
by sorry

end prime_product_sum_l1148_114810


namespace multiplication_exercise_l1148_114827

theorem multiplication_exercise (a b : ℕ+) 
  (h1 : (a + 6) * b = 255)  -- Units digit changed from 1 to 7
  (h2 : (a - 10) * b = 335) -- Tens digit changed from 6 to 5
  : a * b = 285 := by
  sorry

end multiplication_exercise_l1148_114827


namespace vector_expression_l1148_114863

def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (-1, 1)
def c : ℝ × ℝ := (4, 2)

theorem vector_expression : c = 3 • a - b := by sorry

end vector_expression_l1148_114863


namespace inequality_and_uniqueness_l1148_114824

theorem inequality_and_uniqueness 
  (a b c d : ℝ) 
  (pos_a : 0 < a) 
  (pos_b : 0 < b) 
  (pos_c : 0 < c) 
  (pos_d : 0 < d) 
  (sum_eq : a + b = 4) 
  (prod_eq : c * d = 4) : 
  (a * b ≤ c + d) ∧ 
  (a * b = c + d → 
    ∀ (a' b' c' d' : ℝ), 
      0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 0 < d' ∧ 
      a' + b' = 4 ∧ c' * d' = 4 ∧ 
      a' * b' = c' + d' → 
      a' = a ∧ b' = b ∧ c' = c ∧ d' = d) := by
sorry

end inequality_and_uniqueness_l1148_114824


namespace missing_bricks_count_l1148_114836

/-- Represents a brick wall -/
structure BrickWall where
  total_positions : ℕ
  filled_positions : ℕ
  h_filled_le_total : filled_positions ≤ total_positions

/-- The number of missing bricks in a wall -/
def missing_bricks (wall : BrickWall) : ℕ :=
  wall.total_positions - wall.filled_positions

/-- Theorem stating that the number of missing bricks in the given wall is 26 -/
theorem missing_bricks_count (wall : BrickWall) 
  (h_total : wall.total_positions = 60)
  (h_filled : wall.filled_positions = 34) : 
  missing_bricks wall = 26 := by
sorry


end missing_bricks_count_l1148_114836


namespace negation_of_existence_negation_of_quadratic_inequality_l1148_114817

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 + x - 1 < 0) ↔ (∀ x : ℝ, x^2 + x - 1 ≥ 0) := by sorry

end negation_of_existence_negation_of_quadratic_inequality_l1148_114817


namespace johns_remaining_money_l1148_114832

/-- The amount of money John has left after purchasing pizzas and drinks -/
def money_left (p : ℝ) : ℝ :=
  let drink_cost := p
  let medium_pizza_cost := 3 * p
  let large_pizza_cost := 4 * p
  let total_cost := 4 * drink_cost + medium_pizza_cost + 2 * large_pizza_cost
  50 - total_cost

/-- Theorem stating that John's remaining money is 50 - 15p dollars -/
theorem johns_remaining_money (p : ℝ) : money_left p = 50 - 15 * p := by
  sorry

end johns_remaining_money_l1148_114832


namespace f_range_l1148_114873

noncomputable def f (x : ℝ) : ℝ := (1/3) ^ (x^2 - 2*x)

theorem f_range :
  (∀ y ∈ Set.range f, 0 < y ∧ y ≤ 3) ∧
  (∀ ε > 0, ∃ x, |f x - 3| < ε ∧ f x > 0) :=
sorry

end f_range_l1148_114873


namespace floor_e_l1148_114838

theorem floor_e : ⌊Real.exp 1⌋ = 2 := by sorry

end floor_e_l1148_114838


namespace max_area_OAPF_l1148_114876

/-- The equation of ellipse C is (x^2/9) + (y^2/10) = 1 -/
def ellipse_equation (x y : ℝ) : Prop := x^2/9 + y^2/10 = 1

/-- F is the upper focus of ellipse C -/
def F : ℝ × ℝ := (0, 1)

/-- A is the right vertex of ellipse C -/
def A : ℝ × ℝ := (3, 0)

/-- P is a point on ellipse C located in the first quadrant -/
def P : ℝ × ℝ := sorry

/-- The area of quadrilateral OAPF -/
def area_OAPF (P : ℝ × ℝ) : ℝ := sorry

theorem max_area_OAPF :
  ∃ (P : ℝ × ℝ), ellipse_equation P.1 P.2 ∧ P.1 > 0 ∧ P.2 > 0 ∧
  ∀ (Q : ℝ × ℝ), ellipse_equation Q.1 Q.2 → Q.1 > 0 → Q.2 > 0 →
  area_OAPF P ≥ area_OAPF Q ∧
  area_OAPF P = (3 * Real.sqrt 11) / 2 := by
  sorry

end max_area_OAPF_l1148_114876


namespace sum_equals_140_l1148_114879

theorem sum_equals_140 
  (x y z w : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) (pos_w : 0 < w)
  (h1 : x^2 + y^2 = 2500) (h2 : z^2 + w^2 = 2500)
  (h3 : x * z = 1200) (h4 : y * w = 1200) : 
  x + y + z + w = 140 := by
sorry

end sum_equals_140_l1148_114879


namespace problem_statement_l1148_114815

theorem problem_statement (a b c : ℝ) 
  (h1 : a + b + c = 0) 
  (h2 : a * b + c + 1 = 0) 
  (h3 : a = 1) : 
  b^2 - 4*c ≥ 0 := by
  sorry

end problem_statement_l1148_114815


namespace height_inscribed_circle_inequality_l1148_114893

/-- For a right triangle, the height dropped to the hypotenuse is at most (1 + √2) times the radius of the inscribed circle. -/
theorem height_inscribed_circle_inequality (a b c h r : ℝ) : 
  a > 0 → b > 0 → c > 0 → h > 0 → r > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  h = (a * b) / c →  -- Definition of height
  r = (a + b - c) / 2 →  -- Definition of inscribed circle radius
  h ≤ r * (1 + Real.sqrt 2) := by
  sorry

end height_inscribed_circle_inequality_l1148_114893


namespace polynomial_division_degree_l1148_114852

/-- Given a polynomial division where:
    - p(x) is a polynomial of degree 17
    - g(x) is the divisor polynomial
    - The quotient polynomial has degree 9
    - The remainder polynomial has degree 5
    Then the degree of g(x) is 8. -/
theorem polynomial_division_degree (p g q r : Polynomial ℝ) : 
  Polynomial.degree p = 17 →
  p = g * q + r →
  Polynomial.degree q = 9 →
  Polynomial.degree r = 5 →
  Polynomial.degree g = 8 := by
  sorry

end polynomial_division_degree_l1148_114852


namespace vector_angle_condition_l1148_114821

-- Define the vectors a and b as functions of x
def a (x : ℝ) : Fin 2 → ℝ := ![2, x + 1]
def b (x : ℝ) : Fin 2 → ℝ := ![x + 2, 6]

-- Define the dot product of a and b
def dot_product (x : ℝ) : ℝ := (a x 0) * (b x 0) + (a x 1) * (b x 1)

-- Define the cross product of a and b
def cross_product (x : ℝ) : ℝ := (a x 0) * (b x 1) - (a x 1) * (b x 0)

-- Theorem statement
theorem vector_angle_condition (x : ℝ) :
  (dot_product x > 0 ∧ cross_product x ≠ 0) ↔ (x > -5/4 ∧ x ≠ 2) :=
sorry

end vector_angle_condition_l1148_114821


namespace arithmetic_sequence_properties_l1148_114820

def a (n : ℕ) : ℝ := 2 * n - 8

theorem arithmetic_sequence_properties :
  (∀ n : ℕ, a (n + 1) > a n) ∧
  (∀ n : ℕ, n > 0 → a (n + 1) / (n + 1) > a n / n) ∧
  (∃ n : ℕ, (n + 1) * a (n + 1) ≤ n * a n) ∧
  (∃ n : ℕ, a (n + 1)^2 ≤ a n^2) :=
by sorry

end arithmetic_sequence_properties_l1148_114820


namespace largest_integer_with_remainder_l1148_114859

theorem largest_integer_with_remainder : ∃ n : ℕ, 
  (n < 100) ∧ 
  (n % 6 = 4) ∧ 
  (∀ m : ℕ, m < 100 → m % 6 = 4 → m ≤ n) ∧
  (n = 94) :=
sorry

end largest_integer_with_remainder_l1148_114859


namespace spinach_not_music_lover_l1148_114845

-- Define the universe
variable (U : Type)

-- Define predicates
variable (S : U → Prop)  -- x likes spinach
variable (G : U → Prop)  -- x is a pearl diver
variable (Z : U → Prop)  -- x is a music lover

-- State the theorem
theorem spinach_not_music_lover 
  (h1 : ∃ x, S x ∧ ¬G x)
  (h2 : ∀ x, Z x → (G x ∨ ¬S x))
  (h3 : (∀ x, ¬G x → Z x) ∨ (∀ x, G x → ¬Z x))
  : ∀ x, S x → ¬Z x :=
by sorry

end spinach_not_music_lover_l1148_114845


namespace saltwater_aquariums_count_l1148_114869

/-- The number of saltwater aquariums Tyler has -/
def saltwater_aquariums : ℕ := 2184 / 39

/-- The number of freshwater aquariums Tyler has -/
def freshwater_aquariums : ℕ := 10

/-- The number of animals in each aquarium -/
def animals_per_aquarium : ℕ := 39

/-- The total number of saltwater animals Tyler has -/
def total_saltwater_animals : ℕ := 2184

theorem saltwater_aquariums_count : saltwater_aquariums = 56 := by
  sorry

end saltwater_aquariums_count_l1148_114869


namespace basin_capacity_l1148_114888

/-- The capacity of a basin given waterfall flow rate, leak rate, and fill time -/
theorem basin_capacity
  (waterfall_flow : ℝ)  -- Flow rate of the waterfall in gallons per second
  (leak_rate : ℝ)       -- Leak rate of the basin in gallons per second
  (fill_time : ℝ)       -- Time to fill the basin in seconds
  (h1 : waterfall_flow = 24)
  (h2 : leak_rate = 4)
  (h3 : fill_time = 13)
  : (waterfall_flow - leak_rate) * fill_time = 260 :=
by
  sorry

#check basin_capacity

end basin_capacity_l1148_114888


namespace intersected_cubes_count_l1148_114858

-- Define a cube structure
structure Cube where
  size : ℕ
  unit_cubes : ℕ

-- Define a plane that bisects the diagonal
structure BisectingPlane where
  perpendicular_to_diagonal : Bool
  bisects_diagonal : Bool

-- Define the function to count intersected cubes
def count_intersected_cubes (c : Cube) (p : BisectingPlane) : ℕ :=
  sorry

-- Theorem statement
theorem intersected_cubes_count 
  (c : Cube) 
  (p : BisectingPlane) 
  (h1 : c.size = 4) 
  (h2 : c.unit_cubes = 64) 
  (h3 : p.perpendicular_to_diagonal = true) 
  (h4 : p.bisects_diagonal = true) : 
  count_intersected_cubes c p = 24 :=
sorry

end intersected_cubes_count_l1148_114858


namespace cone_lateral_surface_angle_l1148_114843

/-- For a cone with an equilateral triangle as its axial section, 
    the angle of the sector formed by unfolding its lateral surface is π radians. -/
theorem cone_lateral_surface_angle (R r : ℝ) (α : ℝ) : 
  R > 0 ∧ r > 0 ∧ R = 2 * r → α = π := by sorry

end cone_lateral_surface_angle_l1148_114843


namespace invalid_diagonal_sets_l1148_114842

-- Define a function to check if a set of three numbers satisfies the condition
def isValidDiagonalSet (x y z : ℝ) : Prop :=
  x^2 + y^2 ≥ z^2 ∧ x^2 + z^2 ≥ y^2 ∧ y^2 + z^2 ≥ x^2

-- Theorem stating which sets are invalid for external diagonals of a right regular prism
theorem invalid_diagonal_sets :
  (¬ isValidDiagonalSet 3 4 6) ∧
  (¬ isValidDiagonalSet 5 5 8) ∧
  (¬ isValidDiagonalSet 7 8 12) ∧
  (isValidDiagonalSet 6 8 10) ∧
  (isValidDiagonalSet 3 4 5) :=
by sorry

end invalid_diagonal_sets_l1148_114842


namespace subadditive_sequence_inequality_l1148_114854

/-- A non-negative sequence satisfying the subadditivity property -/
def SubadditiveSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n ≥ 0) ∧ (∀ m n, a (m + n) ≤ a m + a n)

/-- The main theorem to be proved -/
theorem subadditive_sequence_inequality (a : ℕ → ℝ) (h : SubadditiveSequence a) :
    ∀ m n, m > 0 → n ≥ m → a n ≤ m * a 1 + (n / m - 1) * a m := by
  sorry

end subadditive_sequence_inequality_l1148_114854


namespace complex_real_part_theorem_l1148_114890

theorem complex_real_part_theorem (a : ℝ) : 
  (((a - Complex.I) / (3 + Complex.I)).re = 1/2) → a = 2 := by
  sorry

end complex_real_part_theorem_l1148_114890


namespace sofa_bench_arrangement_l1148_114895

/-- The number of ways to arrange n indistinguishable objects of one type
    and k indistinguishable objects of another type in a row -/
def arrangements (n k : ℕ) : ℕ := Nat.choose (n + k) n

/-- Theorem: There are 210 distinct ways to arrange 6 indistinguishable objects
    of one type and 4 indistinguishable objects of another type in a row -/
theorem sofa_bench_arrangement : arrangements 6 4 = 210 := by
  sorry

end sofa_bench_arrangement_l1148_114895


namespace min_value_expression_l1148_114837

theorem min_value_expression (x y z : ℝ) 
  (hx : -0.5 ≤ x ∧ x ≤ 1) 
  (hy : -0.5 ≤ y ∧ y ≤ 1) 
  (hz : -0.5 ≤ z ∧ z ≤ 1) : 
  3 / ((1 - x) * (1 - y) * (1 - z)) + 3 / ((1 + x) * (1 + y) * (1 + z)) ≥ 6 ∧
  (x = 0 ∧ y = 0 ∧ z = 0 → 3 / ((1 - x) * (1 - y) * (1 - z)) + 3 / ((1 + x) * (1 + y) * (1 + z)) = 6) :=
by sorry

end min_value_expression_l1148_114837


namespace floor_equation_solutions_l1148_114802

theorem floor_equation_solutions (n : ℤ) : 
  (⌊n^2 / 9⌋ : ℤ) - (⌊n / 3⌋ : ℤ)^2 = 3 ↔ n = 8 ∨ n = 10 :=
sorry

end floor_equation_solutions_l1148_114802


namespace arithmetic_geometric_ratio_l1148_114819

/-- Given an arithmetic sequence {a_n} with common difference d ≠ 0,
    if a_1, a_3, and a_9 form a geometric sequence, then a_3 / a_1 = 3 -/
theorem arithmetic_geometric_ratio (a : ℕ → ℝ) (d : ℝ) (h1 : d ≠ 0)
  (h2 : ∀ n, a (n + 1) = a n + d)  -- arithmetic sequence condition
  (h3 : (a 3 / a 1) = (a 9 / a 3)) -- geometric sequence condition
  : a 3 / a 1 = 3 := by
  sorry


end arithmetic_geometric_ratio_l1148_114819


namespace max_a_for_monotonic_cubic_l1148_114853

/-- Given a function f(x) = x^3 - ax that is monotonically increasing on [1, +∞),
    the maximum value of a is 3. -/
theorem max_a_for_monotonic_cubic (a : ℝ) : 
  (∀ x ≥ 1, ∀ y ≥ x, (x^3 - a*x) ≤ (y^3 - a*y)) → a ≤ 3 :=
sorry

end max_a_for_monotonic_cubic_l1148_114853


namespace smallest_prime_perimeter_scalene_triangle_l1148_114847

/-- A function that checks if a number is prime --/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if three numbers form a scalene triangle --/
def isScalene (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

/-- The main theorem --/
theorem smallest_prime_perimeter_scalene_triangle :
  ∀ a b c : ℕ,
    a ≥ 5 → b ≥ 5 → c ≥ 5 →
    isPrime a → isPrime b → isPrime c →
    isScalene a b c →
    isPrime (a + b + c) →
    a + b + c ≥ 23 :=
sorry

end smallest_prime_perimeter_scalene_triangle_l1148_114847


namespace solution_difference_l1148_114897

theorem solution_difference (r s : ℝ) : 
  (∀ x, (6 * x - 18) / (x^2 + 3 * x - 18) = x + 3 ↔ x = r ∨ x = s) →
  r ≠ s →
  r > s →
  r - s = 3 := by
sorry

end solution_difference_l1148_114897
