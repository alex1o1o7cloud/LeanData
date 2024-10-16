import Mathlib

namespace NUMINAMATH_CALUDE_leftover_money_l2668_266894

/-- Calculates the leftover money after reading books and buying candy -/
theorem leftover_money
  (payment_rate : ℚ)
  (pages_per_book : ℕ)
  (books_read : ℕ)
  (candy_cost : ℚ)
  (h1 : payment_rate = 1 / 100)  -- $0.01 per page
  (h2 : pages_per_book = 150)
  (h3 : books_read = 12)
  (h4 : candy_cost = 15) :
  payment_rate * (pages_per_book * books_read : ℚ) - candy_cost = 3 := by
sorry

end NUMINAMATH_CALUDE_leftover_money_l2668_266894


namespace NUMINAMATH_CALUDE_money_distribution_ratio_l2668_266860

def distribute_money (total : ℝ) (p q r s : ℝ) : Prop :=
  p + q + r + s = total ∧
  p = 2 * q ∧
  q = r ∧
  s - p = 250

theorem money_distribution_ratio :
  ∀ (total p q r s : ℝ),
    total = 1000 →
    distribute_money total p q r s →
    s / r = 4 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_ratio_l2668_266860


namespace NUMINAMATH_CALUDE_calculation_proof_l2668_266854

theorem calculation_proof : (3.242 * 15) / 100 = 0.4863 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2668_266854


namespace NUMINAMATH_CALUDE_cos_minus_sin_nine_pi_fourths_l2668_266843

theorem cos_minus_sin_nine_pi_fourths : 
  Real.cos (-9 * Real.pi / 4) - Real.sin (-9 * Real.pi / 4) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_minus_sin_nine_pi_fourths_l2668_266843


namespace NUMINAMATH_CALUDE_new_person_weight_l2668_266820

/-- Represents a group of people with their weights -/
structure WeightGroup where
  size : Nat
  total_weight : ℝ
  avg_weight : ℝ

/-- Represents the change in the group when a person is replaced -/
structure WeightChange where
  old_weight : ℝ
  new_weight : ℝ
  avg_increase : ℝ

/-- Theorem stating the weight of the new person -/
theorem new_person_weight 
  (group : WeightGroup)
  (change : WeightChange)
  (h1 : group.size = 8)
  (h2 : change.old_weight = 65)
  (h3 : change.avg_increase = 3.5)
  (h4 : ∀ w E, (w * (1 + E / 100) - w) ≤ change.avg_increase) :
  change.new_weight = 93 := by
sorry


end NUMINAMATH_CALUDE_new_person_weight_l2668_266820


namespace NUMINAMATH_CALUDE_circles_internally_tangent_l2668_266867

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y + 12 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 14*x - 2*y + 14 = 0

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (3, -2)
def center2 : ℝ × ℝ := (7, 1)
def radius1 : ℝ := 1
def radius2 : ℝ := 6

-- Theorem stating that the circles are internally tangent
theorem circles_internally_tangent :
  let d := Real.sqrt ((center2.1 - center1.1)^2 + (center2.2 - center1.2)^2)
  d = radius2 - radius1 :=
sorry

end NUMINAMATH_CALUDE_circles_internally_tangent_l2668_266867


namespace NUMINAMATH_CALUDE_solution_in_interval_l2668_266804

theorem solution_in_interval : ∃ x₀ : ℝ, (Real.exp x₀ + x₀ = 2) ∧ (0 < x₀ ∧ x₀ < 1) := by sorry

end NUMINAMATH_CALUDE_solution_in_interval_l2668_266804


namespace NUMINAMATH_CALUDE_minimum_students_in_class_l2668_266858

theorem minimum_students_in_class (boys girls : ℕ) : 
  boys > 0 → girls > 0 →
  2 * (boys / 2) = 3 * (girls / 3) →
  boys + girls ≥ 7 :=
by
  sorry

#check minimum_students_in_class

end NUMINAMATH_CALUDE_minimum_students_in_class_l2668_266858


namespace NUMINAMATH_CALUDE_two_zeros_iff_k_range_l2668_266893

/-- The function f(x) = xe^x - k has exactly two zeros if and only if -1/e < k < 0 -/
theorem two_zeros_iff_k_range (k : ℝ) :
  (∃! (a b : ℝ), a ≠ b ∧ a * Real.exp a - k = 0 ∧ b * Real.exp b - k = 0) ↔
  -1 / Real.exp 1 < k ∧ k < 0 := by sorry

end NUMINAMATH_CALUDE_two_zeros_iff_k_range_l2668_266893


namespace NUMINAMATH_CALUDE_factorization_sum_l2668_266889

theorem factorization_sum (a b c d e f g h j k : ℤ) :
  (∀ x y : ℝ, 8 * x^4 - 125 * y^4 = (a*x + b*y) * (c*x^2 + d*x*y + e*y^2) * (f*x + g*y) * (h*x^2 + j*x*y + k*y^2)) →
  a + b + c + d + e + f + g + h + j + k = 4 := by
  sorry

end NUMINAMATH_CALUDE_factorization_sum_l2668_266889


namespace NUMINAMATH_CALUDE_percentage_problem_l2668_266890

theorem percentage_problem : 
  ∃ x : ℝ, (120 / 100) * x = 1800 → (20 / 100) * x = 300 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l2668_266890


namespace NUMINAMATH_CALUDE_right_triangular_prism_volume_l2668_266873

theorem right_triangular_prism_volume 
  (a b h : ℝ) 
  (ha : a = Real.sqrt 2) 
  (hb : b = Real.sqrt 2) 
  (hh : h = 3) 
  (right_triangle : a * a + b * b = (a + b) * (a + b) / 2) :
  (1 / 2) * a * b * h = 3 := by sorry

end NUMINAMATH_CALUDE_right_triangular_prism_volume_l2668_266873


namespace NUMINAMATH_CALUDE_cos_330_degrees_l2668_266898

theorem cos_330_degrees : Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_330_degrees_l2668_266898


namespace NUMINAMATH_CALUDE_missing_fraction_proof_l2668_266895

theorem missing_fraction_proof (x : ℚ) : 
  1/2 + (-5/6) + 1/5 + 1/4 + (-9/20) + (-5/6) + x = 5/6 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_missing_fraction_proof_l2668_266895


namespace NUMINAMATH_CALUDE_cindy_calculation_l2668_266822

theorem cindy_calculation (x : ℝ) : (x - 7) / 5 = 15 → (x - 5) / 7 = 11 := by
  sorry

end NUMINAMATH_CALUDE_cindy_calculation_l2668_266822


namespace NUMINAMATH_CALUDE_atLeastOneWhite_oneRedOneBlack_mutually_exclusive_not_exhaustive_l2668_266882

-- Define the set of possible ball colors
inductive Color
| Red
| White
| Black

-- Define the bag contents
def bag : Multiset Color :=
  Multiset.replicate 3 Color.Red + Multiset.replicate 2 Color.White + Multiset.replicate 1 Color.Black

-- Define a draw as a pair of colors
def Draw := (Color × Color)

-- Define the event "At least one white ball"
def atLeastOneWhite (draw : Draw) : Prop :=
  draw.1 = Color.White ∨ draw.2 = Color.White

-- Define the event "one red ball and one black ball"
def oneRedOneBlack (draw : Draw) : Prop :=
  (draw.1 = Color.Red ∧ draw.2 = Color.Black) ∨ (draw.1 = Color.Black ∧ draw.2 = Color.Red)

-- Theorem stating that the events are mutually exclusive but not exhaustive
theorem atLeastOneWhite_oneRedOneBlack_mutually_exclusive_not_exhaustive :
  (∀ (draw : Draw), ¬(atLeastOneWhite draw ∧ oneRedOneBlack draw)) ∧
  (∃ (draw : Draw), ¬atLeastOneWhite draw ∧ ¬oneRedOneBlack draw) :=
sorry

end NUMINAMATH_CALUDE_atLeastOneWhite_oneRedOneBlack_mutually_exclusive_not_exhaustive_l2668_266882


namespace NUMINAMATH_CALUDE_trivia_team_tryouts_l2668_266809

theorem trivia_team_tryouts (not_picked : ℕ) (groups : ℕ) (students_per_group : ℕ) : 
  not_picked = 10 → groups = 8 → students_per_group = 6 → 
  not_picked + groups * students_per_group = 58 := by
sorry

end NUMINAMATH_CALUDE_trivia_team_tryouts_l2668_266809


namespace NUMINAMATH_CALUDE_smallest_19_factor_number_is_78732_l2668_266876

/-- A function that returns the number of factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- The smallest positive integer with exactly 19 factors -/
def smallest_19_factor_number : ℕ+ := sorry

/-- Theorem stating that the smallest positive integer with exactly 19 factors is 78732 -/
theorem smallest_19_factor_number_is_78732 : 
  smallest_19_factor_number = 78732 ∧ num_factors smallest_19_factor_number = 19 := by sorry

end NUMINAMATH_CALUDE_smallest_19_factor_number_is_78732_l2668_266876


namespace NUMINAMATH_CALUDE_price_increase_percentage_l2668_266830

theorem price_increase_percentage (initial_price new_price : ℝ) 
  (h1 : initial_price = 5)
  (h2 : new_price = 5.55) : 
  (new_price - initial_price) / initial_price * 100 = 11 := by
  sorry

end NUMINAMATH_CALUDE_price_increase_percentage_l2668_266830


namespace NUMINAMATH_CALUDE_tangent_slope_at_one_l2668_266815

-- Define the function representing the curve
def f (x : ℝ) : ℝ := x^3 - 2*x^2

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 3*x^2 - 4*x

-- Theorem statement
theorem tangent_slope_at_one :
  f' 1 = -1 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_one_l2668_266815


namespace NUMINAMATH_CALUDE_point_movement_l2668_266857

/-- Given point A(-1, 3), moving it 5 units down and 2 units to the left results in point B(-3, -2) -/
theorem point_movement (A B : ℝ × ℝ) : 
  A = (-1, 3) → 
  B.1 = A.1 - 2 → 
  B.2 = A.2 - 5 → 
  B = (-3, -2) := by
sorry

end NUMINAMATH_CALUDE_point_movement_l2668_266857


namespace NUMINAMATH_CALUDE_rotated_semicircle_area_l2668_266850

/-- The area of a shaded figure formed by rotating a semicircle -/
theorem rotated_semicircle_area (R : ℝ) (h : R > 0) :
  let α : ℝ := 20 * π / 180  -- Convert 20° to radians
  let semicircle_area : ℝ := π * R^2 / 2
  let sector_area : ℝ := 2 * R^2 * α / 2
  sector_area = 2 * π * R^2 / 9 :=
by sorry

end NUMINAMATH_CALUDE_rotated_semicircle_area_l2668_266850


namespace NUMINAMATH_CALUDE_number_problem_l2668_266875

theorem number_problem : ∃ x : ℚ, x^2 + 95 = (x - 15)^2 ∧ x = 13/3 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2668_266875


namespace NUMINAMATH_CALUDE_distance_origin_to_point_l2668_266821

/-- The distance between the origin (0, 0, 0) and the point (1, 2, 3) is √14 -/
theorem distance_origin_to_point :
  Real.sqrt ((1 : ℝ)^2 + 2^2 + 3^2) = Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_distance_origin_to_point_l2668_266821


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l2668_266826

theorem complex_number_quadrant : ∀ z : ℂ, 
  (3 - Complex.I) * z = 1 - 2 * Complex.I →
  0 < z.re ∧ z.im < 0 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l2668_266826


namespace NUMINAMATH_CALUDE_gcd_228_1995_l2668_266838

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by
  sorry

end NUMINAMATH_CALUDE_gcd_228_1995_l2668_266838


namespace NUMINAMATH_CALUDE_impossible_perpendicular_intersection_l2668_266840

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (intersect : Line → Line → Prop)
variable (coincident : Line → Line → Prop)

-- Define the planes and lines
variable (α : Plane)
variable (a b : Line)

-- State the theorem
theorem impossible_perpendicular_intersection 
  (h1 : ¬ coincident a b)
  (h2 : perpendicular a α)
  (h3 : intersect a b) :
  ¬ (perpendicular b α) :=
sorry

end NUMINAMATH_CALUDE_impossible_perpendicular_intersection_l2668_266840


namespace NUMINAMATH_CALUDE_quadratic_form_b_l2668_266864

/-- Given a quadratic of the form x^2 + bx + 54 where b is positive,
    if it can be rewritten as (x+m)^2 + 18, then b = 12 -/
theorem quadratic_form_b (b : ℝ) (m : ℝ) : 
  b > 0 → 
  (∀ x, x^2 + b*x + 54 = (x+m)^2 + 18) → 
  b = 12 := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_b_l2668_266864


namespace NUMINAMATH_CALUDE_value_of_T_l2668_266874

-- Define the variables
variable (M A T H E : ℤ)

-- Define the conditions
def condition_H : H = 8 := by sorry
def condition_MATH : M + A + T + H = 47 := by sorry
def condition_MEET : M + E + E + T = 62 := by sorry
def condition_TEAM : T + E + A + M = 58 := by sorry

-- Theorem to prove
theorem value_of_T : T = 9 := by sorry

end NUMINAMATH_CALUDE_value_of_T_l2668_266874


namespace NUMINAMATH_CALUDE_girls_to_boys_ratio_l2668_266834

theorem girls_to_boys_ratio (total : ℕ) (difference : ℕ) (girls boys : ℕ) : 
  total = 24 → 
  difference = 6 → 
  girls + boys = total → 
  girls = boys + difference → 
  (girls : ℚ) / (boys : ℚ) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_girls_to_boys_ratio_l2668_266834


namespace NUMINAMATH_CALUDE_complement_union_problem_l2668_266881

def U : Finset Nat := {0, 1, 2, 3, 4, 5}
def A : Finset Nat := {1, 2, 3, 5}
def B : Finset Nat := {2, 4}

theorem complement_union_problem :
  (U \ A) ∪ B = {0, 2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_problem_l2668_266881


namespace NUMINAMATH_CALUDE_dilation_problem_l2668_266896

/-- Dilation of a complex number -/
def dilation (z : ℂ) (center : ℂ) (scale : ℝ) : ℂ :=
  center + scale * (z - center)

/-- The problem statement -/
theorem dilation_problem : dilation (-2 + I) (1 - 3*I) 3 = -8 + 9*I := by
  sorry

end NUMINAMATH_CALUDE_dilation_problem_l2668_266896


namespace NUMINAMATH_CALUDE_parabola_vertex_correct_l2668_266891

-- Define the parabola equation
def parabola_equation (x y : ℝ) : Prop :=
  y = (x + 2)^2 + 1

-- Define the vertex of the parabola
def vertex : ℝ × ℝ := (-2, 1)

-- Theorem stating that the given equation represents a parabola with the specified vertex
theorem parabola_vertex_correct :
  ∀ x y : ℝ, parabola_equation x y → (x, y) = vertex ↔ x = -2 ∧ y = 1 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_correct_l2668_266891


namespace NUMINAMATH_CALUDE_phil_coin_collection_l2668_266884

def initial_coins : ℕ := 250
def years_tripling : ℕ := 3
def weeks_in_year : ℕ := 52
def days_in_year : ℕ := 365
def coins_per_week_4th_year : ℕ := 5
def coins_every_second_day_5th_year : ℕ := 2
def coins_per_day_6th_year : ℕ := 1
def loss_fraction : ℚ := 1/3

def coins_after_loss : ℕ := 1160

theorem phil_coin_collection :
  let coins_after_3_years := initial_coins * (2^years_tripling)
  let coins_4th_year := coins_after_3_years + coins_per_week_4th_year * weeks_in_year
  let coins_5th_year := coins_4th_year + coins_every_second_day_5th_year * (days_in_year / 2)
  let coins_6th_year := coins_5th_year + coins_per_day_6th_year * days_in_year
  let coins_before_loss := coins_6th_year
  coins_after_loss = coins_before_loss - ⌊coins_before_loss * loss_fraction⌋ :=
by sorry

end NUMINAMATH_CALUDE_phil_coin_collection_l2668_266884


namespace NUMINAMATH_CALUDE_complex_number_existence_l2668_266855

theorem complex_number_existence : ∃ z : ℂ, 
  (∃ r : ℝ, z + 5 / z = r) ∧ 
  (Complex.re (z + 3) = -Complex.im (z + 3)) ∧
  ((z = -1 - 2*Complex.I) ∨ (z = -2 - Complex.I)) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_existence_l2668_266855


namespace NUMINAMATH_CALUDE_binomial_congruence_l2668_266813

theorem binomial_congruence (p n : ℕ) (hp : Prime p) (hp_gt_3 : p > 3) (hn : n > 0) 
  (h_cong : n ≡ 1 [MOD p]) : 
  (Nat.choose (n * p) p) ≡ n [MOD p^4] := by sorry

end NUMINAMATH_CALUDE_binomial_congruence_l2668_266813


namespace NUMINAMATH_CALUDE_max_value_theorem_l2668_266812

theorem max_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 1) 
  (h_equal_roots : a^2 = 4*(b-1)) : 
  (∃ (x : ℝ), (3*a + 2*b) / (a + b) ≤ x) ∧ 
  (∀ (y : ℝ), (3*a + 2*b) / (a + b) ≤ y → y ≥ 5/2) :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2668_266812


namespace NUMINAMATH_CALUDE_equation_solutions_l2668_266803

theorem equation_solutions :
  (∃ x1 x2 : ℝ, x1 = (3 + Real.sqrt 15) / 3 ∧ x2 = (3 - Real.sqrt 15) / 3 ∧ 
    3 * x1^2 - 6 * x1 - 2 = 0 ∧ 3 * x2^2 - 6 * x2 - 2 = 0) ∧
  (∃ y1 y2 : ℝ, y1 = 3 ∧ y2 = 5 ∧ 
    (y1 - 3)^2 = 2 * y1 - 6 ∧ (y2 - 3)^2 = 2 * y2 - 6) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2668_266803


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2668_266853

theorem sufficient_not_necessary (a b x : ℝ) :
  (∀ x, x > a^2 + b^2 → x > 2*a*b) ∧
  (∃ a b x, x > 2*a*b ∧ x ≤ a^2 + b^2) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2668_266853


namespace NUMINAMATH_CALUDE_gig_song_ratio_l2668_266879

/-- Proves that the ratio of the length of the last song to the length of the first two songs is 3:1 --/
theorem gig_song_ratio :
  let days_in_two_weeks : ℕ := 14
  let gigs_in_two_weeks : ℕ := days_in_two_weeks / 2
  let songs_per_gig : ℕ := 3
  let length_of_first_two_songs : ℕ := 2 * 5
  let total_playing_time : ℕ := 280
  let total_length_first_two_songs : ℕ := gigs_in_two_weeks * length_of_first_two_songs
  let total_length_third_song : ℕ := total_playing_time - total_length_first_two_songs
  let length_third_song_per_gig : ℕ := total_length_third_song / gigs_in_two_weeks
  length_third_song_per_gig / length_of_first_two_songs = 3 := by
  sorry

end NUMINAMATH_CALUDE_gig_song_ratio_l2668_266879


namespace NUMINAMATH_CALUDE_twelfth_even_multiple_of_5_l2668_266819

/-- The nth positive integer that is both even and a multiple of 5 -/
def evenMultipleOf5 (n : ℕ) : ℕ := 10 * n

/-- Proof that the 12th positive integer that is both even and a multiple of 5 is 120 -/
theorem twelfth_even_multiple_of_5 : evenMultipleOf5 12 = 120 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_even_multiple_of_5_l2668_266819


namespace NUMINAMATH_CALUDE_gcd_f_x_l2668_266814

def f (x : ℤ) : ℤ := (5*x+3)*(11*x+2)*(14*x+7)*(3*x+8)

theorem gcd_f_x (x : ℤ) (h : ∃ k : ℤ, x = 3456 * k) : 
  Nat.gcd (Int.natAbs (f x)) (Int.natAbs x) = 48 := by
  sorry

end NUMINAMATH_CALUDE_gcd_f_x_l2668_266814


namespace NUMINAMATH_CALUDE_brittany_age_theorem_l2668_266872

/-- Brittany's age after returning from vacation -/
def brittany_age_after_vacation (rebecca_age : ℕ) (age_difference : ℕ) (vacation_duration : ℕ) : ℕ :=
  rebecca_age + age_difference + vacation_duration

theorem brittany_age_theorem (rebecca_age : ℕ) (age_difference : ℕ) (vacation_duration : ℕ) 
  (h1 : rebecca_age = 25)
  (h2 : age_difference = 3)
  (h3 : vacation_duration = 4) :
  brittany_age_after_vacation rebecca_age age_difference vacation_duration = 32 := by
  sorry

end NUMINAMATH_CALUDE_brittany_age_theorem_l2668_266872


namespace NUMINAMATH_CALUDE_assembly_line_production_rate_l2668_266817

/-- Assembly line production problem -/
theorem assembly_line_production_rate 
  (initial_rate : ℝ) 
  (initial_order : ℝ) 
  (second_order : ℝ) 
  (average_output : ℝ) 
  (h1 : initial_rate = 20)
  (h2 : initial_order = 60)
  (h3 : second_order = 60)
  (h4 : average_output = 30) :
  let total_cogs := initial_order + second_order
  let total_time := total_cogs / average_output
  let initial_time := initial_order / initial_rate
  let second_time := total_time - initial_time
  second_order / second_time = 60 := by
  sorry

end NUMINAMATH_CALUDE_assembly_line_production_rate_l2668_266817


namespace NUMINAMATH_CALUDE_area_inscribed_circle_l2668_266831

/-- The area of an inscribed circle in a triangle with given side lengths -/
theorem area_inscribed_circle (a b c : ℝ) (ha : a = 13) (hb : b = 14) (hc : c = 15) :
  let s := (a + b + c) / 2
  let area_triangle := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let r := area_triangle / s
  π * r^2 = (3136 / 81) * π := by sorry

end NUMINAMATH_CALUDE_area_inscribed_circle_l2668_266831


namespace NUMINAMATH_CALUDE_unique_two_digit_integer_l2668_266839

theorem unique_two_digit_integer (u : ℕ) : 
  (10 ≤ u ∧ u < 100) →
  (15 * u) % 100 = 45 →
  u % 17 = 7 →
  u = 43 :=
by sorry

end NUMINAMATH_CALUDE_unique_two_digit_integer_l2668_266839


namespace NUMINAMATH_CALUDE_extra_tip_amount_l2668_266828

/-- The amount of a bill in dollars -/
def bill_amount : ℚ := 26

/-- The percentage of a bad tip -/
def bad_tip_percentage : ℚ := 5 / 100

/-- The percentage of a good tip -/
def good_tip_percentage : ℚ := 20 / 100

/-- Converts dollars to cents -/
def dollars_to_cents (dollars : ℚ) : ℚ := dollars * 100

/-- Calculates the tip amount in cents -/
def tip_amount (bill : ℚ) (percentage : ℚ) : ℚ :=
  dollars_to_cents (bill * percentage)

theorem extra_tip_amount :
  tip_amount bill_amount good_tip_percentage - tip_amount bill_amount bad_tip_percentage = 390 := by
  sorry

end NUMINAMATH_CALUDE_extra_tip_amount_l2668_266828


namespace NUMINAMATH_CALUDE_square_root_of_four_l2668_266825

theorem square_root_of_four (x : ℝ) : x^2 = 4 ↔ x = 2 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_four_l2668_266825


namespace NUMINAMATH_CALUDE_largest_square_from_rectangle_l2668_266802

theorem largest_square_from_rectangle (width length : ℕ) 
  (h_width : width = 32) (h_length : length = 74) :
  ∃ (side : ℕ), side = Nat.gcd width length ∧ 
  side * (width / side) = width ∧ 
  side * (length / side) = length ∧
  ∀ (n : ℕ), n * (width / n) = width ∧ n * (length / n) = length → n ≤ side :=
sorry

end NUMINAMATH_CALUDE_largest_square_from_rectangle_l2668_266802


namespace NUMINAMATH_CALUDE_circumcircle_of_triangle_ABC_l2668_266869

/-- The circumcircle of a triangle is the circle that passes through all three vertices of the triangle. -/
def is_circumcircle (a b c : ℝ × ℝ) (f : ℝ → ℝ → ℝ) : Prop :=
  f a.1 a.2 = 0 ∧ f b.1 b.2 = 0 ∧ f c.1 c.2 = 0

/-- The equation of a circle in general form is x^2 + y^2 + Dx + Ey + F = 0 -/
def circle_equation (D E F : ℝ) (x y : ℝ) : ℝ :=
  x^2 + y^2 + D*x + E*y + F

theorem circumcircle_of_triangle_ABC :
  let A : ℝ × ℝ := (-1, 5)
  let B : ℝ × ℝ := (5, 5)
  let C : ℝ × ℝ := (6, -2)
  let f (x y : ℝ) := circle_equation (-4) (-2) (-20) x y
  is_circumcircle A B C f :=
sorry

end NUMINAMATH_CALUDE_circumcircle_of_triangle_ABC_l2668_266869


namespace NUMINAMATH_CALUDE_quadratic_inverse_unique_solution_l2668_266849

/-- A quadratic function with its inverse -/
structure QuadraticWithInverse where
  a : ℝ
  b : ℝ
  c : ℝ
  f : ℝ → ℝ
  f_inv : ℝ → ℝ
  h_f : ∀ x, f x = a * x^2 + b * x + c
  h_f_inv : ∀ x, f_inv x = c * x^2 + b * x + a
  h_inverse : (∀ x, f (f_inv x) = x) ∧ (∀ x, f_inv (f x) = x)

/-- Theorem stating the unique solution for a, b, and c -/
theorem quadratic_inverse_unique_solution (q : QuadraticWithInverse) :
  q.a = -1 ∧ q.b = 1 ∧ q.c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inverse_unique_solution_l2668_266849


namespace NUMINAMATH_CALUDE_projections_of_P_l2668_266805

def P : ℝ × ℝ × ℝ := (2, 3, 4)

def projection_planes : List (ℝ × ℝ × ℝ) := [(2, 3, 0), (0, 3, 4), (2, 0, 4)]
def projection_axes : List (ℝ × ℝ × ℝ) := [(2, 0, 0), (0, 3, 0), (0, 0, 4)]

theorem projections_of_P :
  (projection_planes = [(2, 3, 0), (0, 3, 4), (2, 0, 4)]) ∧
  (projection_axes = [(2, 0, 0), (0, 3, 0), (0, 0, 4)]) := by
  sorry

end NUMINAMATH_CALUDE_projections_of_P_l2668_266805


namespace NUMINAMATH_CALUDE_temperature_change_l2668_266829

/-- The temperature change problem -/
theorem temperature_change (initial temp_rise temp_drop : Int) : 
  initial = -5 → temp_rise = 5 → temp_drop = 8 → 
  initial + temp_rise - temp_drop = -8 := by
  sorry

end NUMINAMATH_CALUDE_temperature_change_l2668_266829


namespace NUMINAMATH_CALUDE_quadratic_coefficient_relation_l2668_266818

/-- Given two quadratic equations and their root relationships, prove the relation between their coefficients -/
theorem quadratic_coefficient_relation (a b c d : ℝ) (α β : ℝ) : 
  (∀ x, x^2 + a*x + b = 0 ↔ x = α ∨ x = β) →
  (∀ x, x^2 + c*x + d = 0 ↔ x = α^2 + 1 ∨ x = β^2 + 1) →
  c = -a^2 + 2*b - 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_relation_l2668_266818


namespace NUMINAMATH_CALUDE_space_filling_tetrahedrons_octahedrons_l2668_266846

/-- A regular tetrahedron -/
structure RegularTetrahedron :=
  (edge_length : ℝ)
  (edge_length_pos : edge_length > 0)

/-- A regular octahedron -/
structure RegularOctahedron :=
  (edge_length : ℝ)
  (edge_length_pos : edge_length > 0)

/-- A space-filling arrangement -/
structure SpaceFillingArrangement :=
  (tetrahedrons : Set RegularTetrahedron)
  (octahedrons : Set RegularOctahedron)

/-- No gaps or overlaps in the arrangement -/
def NoGapsOrOverlaps (arrangement : SpaceFillingArrangement) : Prop :=
  sorry

/-- All polyhedra in the arrangement are congruent and have equal edge lengths -/
def CongruentWithEqualEdges (arrangement : SpaceFillingArrangement) : Prop :=
  sorry

/-- The main theorem: There exists a space-filling arrangement of congruent regular tetrahedrons
    and regular octahedrons with equal edge lengths, without gaps or overlaps -/
theorem space_filling_tetrahedrons_octahedrons :
  ∃ (arrangement : SpaceFillingArrangement),
    CongruentWithEqualEdges arrangement ∧ NoGapsOrOverlaps arrangement :=
sorry

end NUMINAMATH_CALUDE_space_filling_tetrahedrons_octahedrons_l2668_266846


namespace NUMINAMATH_CALUDE_equation_solution_l2668_266856

theorem equation_solution : ∃! x : ℝ, x + Real.sqrt (3 * x - 2) = 6 ∧ x = (15 - Real.sqrt 73) / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2668_266856


namespace NUMINAMATH_CALUDE_largest_number_with_sum_16_l2668_266863

def is_valid_digit (d : ℕ) : Prop := d = 2 ∨ d = 3 ∨ d = 4

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def all_digits_valid (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, is_valid_digit d

theorem largest_number_with_sum_16 :
  ∀ n : ℕ,
    all_digits_valid n →
    digit_sum n = 16 →
    n ≤ 4432 :=
by sorry

end NUMINAMATH_CALUDE_largest_number_with_sum_16_l2668_266863


namespace NUMINAMATH_CALUDE_quad_pair_coeff_sum_l2668_266899

/-- Two distinct quadratic polynomials with specific properties -/
structure QuadraticPair where
  f : ℝ → ℝ
  g : ℝ → ℝ
  a : ℝ
  c : ℝ
  h_f_def : ∀ x, f x = x^2 + a*x + 3
  h_g_def : ∀ x, g x = x^2 + c*x + 6
  h_distinct : f ≠ g
  h_vertex_f_root_g : g (-a/2) = 0
  h_vertex_g_root_f : f (-c/2) = 0
  h_intersection : f 50 = -50 ∧ g 50 = -50
  h_min_f : ∀ x, f x ≥ 3
  h_min_g : ∀ x, g x ≥ 6

/-- The sum of coefficients a and c is approximately -102.18 -/
theorem quad_pair_coeff_sum (qp : QuadraticPair) : 
  ∃ ε > 0, |qp.a + qp.c + 102.18| < ε := by
  sorry

end NUMINAMATH_CALUDE_quad_pair_coeff_sum_l2668_266899


namespace NUMINAMATH_CALUDE_domain_of_f_l2668_266886

-- Define the function f
def f (x : ℝ) : ℝ := (x - 5) ^ (1/3) + (x - 7) ^ (1/4)

-- Define the domain of f
def domain_f : Set ℝ := {x : ℝ | x ≥ 7}

-- Theorem statement
theorem domain_of_f : 
  ∀ x : ℝ, x ∈ domain_f ↔ ∃ y : ℝ, f x = y :=
sorry

end NUMINAMATH_CALUDE_domain_of_f_l2668_266886


namespace NUMINAMATH_CALUDE_applicant_age_standard_deviation_l2668_266861

theorem applicant_age_standard_deviation
  (average_age : ℝ)
  (max_different_ages : ℕ)
  (h_average : average_age = 31)
  (h_max_ages : max_different_ages = 11) :
  let standard_deviation := (max_different_ages - 1) / 2
  standard_deviation = 5 := by
  sorry

end NUMINAMATH_CALUDE_applicant_age_standard_deviation_l2668_266861


namespace NUMINAMATH_CALUDE_halloween_candy_count_l2668_266883

/-- Represents the number of candies each person has -/
structure CandyCount where
  bob : Nat
  mary : Nat
  john : Nat
  sue : Nat
  sam : Nat

/-- The total number of candies for all friends -/
def totalCandies (cc : CandyCount) : Nat :=
  cc.bob + cc.mary + cc.john + cc.sue + cc.sam

/-- Theorem stating that the total number of candies is 50 -/
theorem halloween_candy_count :
  ∃ (cc : CandyCount),
    cc.bob = 10 ∧
    cc.mary = 5 ∧
    cc.john = 5 ∧
    cc.sue = 20 ∧
    cc.sam = 10 ∧
    totalCandies cc = 50 := by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_count_l2668_266883


namespace NUMINAMATH_CALUDE_original_equals_scientific_l2668_266811

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The original number -/
def original_number : ℕ := 346000000

/-- The scientific notation representation of the original number -/
def scientific_form : ScientificNotation := {
  coefficient := 3.46
  exponent := 8
  is_valid := by sorry
}

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific : (original_number : ℝ) = scientific_form.coefficient * (10 : ℝ) ^ scientific_form.exponent := by
  sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l2668_266811


namespace NUMINAMATH_CALUDE_ascendant_function_theorem_l2668_266845

/-- A function is ascendant if it is non-decreasing --/
def Ascendant (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

theorem ascendant_function_theorem (f : ℝ → ℝ) 
  (h1 : Ascendant (fun x => f x - 3 * x))
  (h2 : Ascendant (fun x => f x - x^3)) :
  Ascendant (fun x => f x - x^2 - x) :=
sorry

end NUMINAMATH_CALUDE_ascendant_function_theorem_l2668_266845


namespace NUMINAMATH_CALUDE_hair_growth_calculation_l2668_266851

theorem hair_growth_calculation (current_length desired_after_donation donation_length : ℕ) 
  (h1 : current_length = 14)
  (h2 : desired_after_donation = 12)
  (h3 : donation_length = 23) :
  donation_length + desired_after_donation - current_length = 21 := by
  sorry

end NUMINAMATH_CALUDE_hair_growth_calculation_l2668_266851


namespace NUMINAMATH_CALUDE_chocolate_bar_cost_proof_l2668_266836

/-- The original cost of one chocolate bar before discount -/
def chocolate_bar_cost : ℝ := 4.82

theorem chocolate_bar_cost_proof (
  gummy_bear_cost : ℝ)
  (chocolate_chip_cost : ℝ)
  (total_cost : ℝ)
  (gummy_bear_discount : ℝ)
  (chocolate_chip_discount : ℝ)
  (chocolate_bar_discount : ℝ)
  (h1 : gummy_bear_cost = 2)
  (h2 : chocolate_chip_cost = 5)
  (h3 : total_cost = 150)
  (h4 : gummy_bear_discount = 0.05)
  (h5 : chocolate_chip_discount = 0.10)
  (h6 : chocolate_bar_discount = 0.15)
  : chocolate_bar_cost = 4.82 := by
  sorry

#check chocolate_bar_cost_proof

end NUMINAMATH_CALUDE_chocolate_bar_cost_proof_l2668_266836


namespace NUMINAMATH_CALUDE_floor_sqrt_50_l2668_266866

theorem floor_sqrt_50 : ⌊Real.sqrt 50⌋ = 7 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_50_l2668_266866


namespace NUMINAMATH_CALUDE_no_real_solutions_l2668_266885

theorem no_real_solutions : ¬∃ (x : ℝ), -3 * x - 8 = 8 * x^2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2668_266885


namespace NUMINAMATH_CALUDE_parkway_elementary_girls_l2668_266833

theorem parkway_elementary_girls (
  total_students : ℕ) 
  (boys : ℕ)
  (soccer_players : ℕ)
  (drama_club : ℕ)
  (both_activities : ℕ)
  (h1 : total_students = 450)
  (h2 : boys = 312)
  (h3 : soccer_players = 250)
  (h4 : drama_club = 150)
  (h5 : both_activities = 75)
  (h6 : (soccer_players : ℚ) * (4 / 5) = boys)
  (h7 : (drama_club : ℚ) * (3 / 5) = total_students - boys)
  (h8 : (both_activities : ℚ) * (33 / 50) = total_students - boys) :
  total_students - boys - 
  ((soccer_players : ℚ) * (1 / 5)).floor - 
  ((drama_club : ℚ) * (3 / 5)).floor + 
  ((both_activities : ℚ) * (33 / 50)).floor = 48 := by
sorry

end NUMINAMATH_CALUDE_parkway_elementary_girls_l2668_266833


namespace NUMINAMATH_CALUDE_total_path_length_is_5pi_sqrt34_l2668_266806

/-- Rectangle ABCD with given dimensions -/
structure Rectangle where
  AB : ℝ
  BC : ℝ

/-- Rotation parameters -/
structure RotationParams where
  firstAngle : ℝ  -- in radians
  secondAngle : ℝ  -- in radians

/-- Calculate the total path length of point A during rotations -/
def totalPathLength (rect : Rectangle) (rotParams : RotationParams) : ℝ :=
  sorry

/-- Theorem: The total path length of point A is 5π × √34 -/
theorem total_path_length_is_5pi_sqrt34 (rect : Rectangle) (rotParams : RotationParams) :
  rect.AB = 3 → rect.BC = 5 → rotParams.firstAngle = π → rotParams.secondAngle = 3 * π / 2 →
  totalPathLength rect rotParams = 5 * π * Real.sqrt 34 :=
sorry

end NUMINAMATH_CALUDE_total_path_length_is_5pi_sqrt34_l2668_266806


namespace NUMINAMATH_CALUDE_area_of_polygon15_l2668_266823

/-- A 15-sided polygon on a 1 cm x 1 cm grid -/
def Polygon15 : List (ℤ × ℤ) :=
  [(1,3), (2,4), (2,5), (3,6), (4,6), (5,6), (6,5), (6,4), (5,3), (5,2), (4,1), (3,1), (2,2), (1,2), (1,3)]

/-- The area of a polygon given its vertices -/
def polygonArea (vertices : List (ℤ × ℤ)) : ℚ :=
  sorry

/-- Theorem stating that the area of the 15-sided polygon is 15 cm² -/
theorem area_of_polygon15 : polygonArea Polygon15 = 15 := by
  sorry

end NUMINAMATH_CALUDE_area_of_polygon15_l2668_266823


namespace NUMINAMATH_CALUDE_li_ming_weight_estimate_l2668_266816

-- Define the regression equation
def weight_estimate (height : ℝ) : ℝ :=
  0.7 * height - 52

-- State the theorem
theorem li_ming_weight_estimate :
  weight_estimate 180 = 74 := by
  sorry

end NUMINAMATH_CALUDE_li_ming_weight_estimate_l2668_266816


namespace NUMINAMATH_CALUDE_anthony_friend_house_distance_l2668_266880

/-- Given the distances between various locations, prove the distance to Anthony's friend's house -/
theorem anthony_friend_house_distance 
  (distance_to_work : ℝ) 
  (distance_to_gym : ℝ) 
  (distance_to_grocery : ℝ) 
  (distance_to_friend : ℝ) : 
  distance_to_work = 10 ∧ 
  distance_to_gym = (distance_to_work / 2) + 2 ∧
  distance_to_grocery = 4 ∧
  distance_to_grocery = 2 * distance_to_gym ∧
  distance_to_friend = 3 * (distance_to_gym + distance_to_grocery) →
  distance_to_friend = 63 := by
  sorry


end NUMINAMATH_CALUDE_anthony_friend_house_distance_l2668_266880


namespace NUMINAMATH_CALUDE_bike_distance_proof_l2668_266870

theorem bike_distance_proof (x t : ℝ) 
  (h1 : (x + 1) * (3 * t / 4) = x * t)
  (h2 : (x - 1) * (t + 3) = x * t) :
  x * t = 36 := by
  sorry

end NUMINAMATH_CALUDE_bike_distance_proof_l2668_266870


namespace NUMINAMATH_CALUDE_total_length_of_T_l2668_266888

-- Define the set T
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p; |(|(|x| - 3)| - 1)| + |(|(|y| - 3)| - 1)| = 2}

-- Define the total length function
def totalLength (S : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem total_length_of_T : totalLength T = 32 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_total_length_of_T_l2668_266888


namespace NUMINAMATH_CALUDE_simplify_negative_a_minus_a_l2668_266835

theorem simplify_negative_a_minus_a (a : ℝ) : -a - a = -2 * a := by
  sorry

end NUMINAMATH_CALUDE_simplify_negative_a_minus_a_l2668_266835


namespace NUMINAMATH_CALUDE_power_mod_theorem_l2668_266842

theorem power_mod_theorem : 2^2001 ≡ 64 [MOD (2^7 - 1)] := by sorry

end NUMINAMATH_CALUDE_power_mod_theorem_l2668_266842


namespace NUMINAMATH_CALUDE_four_corner_holes_l2668_266841

/-- Represents the state of a rectangular paper. -/
structure Paper where
  folded : Bool
  holes : List (Nat × Nat)

/-- Represents the folding operations. -/
inductive FoldOperation
  | BottomToTop
  | LeftToRight
  | TopToBottom

/-- Folds the paper according to the given operation. -/
def fold (p : Paper) (op : FoldOperation) : Paper :=
  { p with folded := true }

/-- Punches a hole in the top left corner of the folded paper. -/
def punchHole (p : Paper) : Paper :=
  { p with holes := (0, 0) :: p.holes }

/-- Unfolds the paper and calculates the final hole positions. -/
def unfold (p : Paper) : Paper :=
  { p with 
    folded := false,
    holes := [(0, 0), (0, 1), (1, 0), (1, 1)] }

/-- The main theorem stating that after folding, punching, and unfolding, 
    the paper will have four holes, one in each corner. -/
theorem four_corner_holes (p : Paper) :
  let p1 := fold p FoldOperation.BottomToTop
  let p2 := fold p1 FoldOperation.LeftToRight
  let p3 := fold p2 FoldOperation.TopToBottom
  let p4 := punchHole p3
  let final := unfold p4
  final.holes = [(0, 0), (0, 1), (1, 0), (1, 1)] :=
by sorry

end NUMINAMATH_CALUDE_four_corner_holes_l2668_266841


namespace NUMINAMATH_CALUDE_mindy_tax_rate_l2668_266852

theorem mindy_tax_rate 
  (mork_rate : ℝ) 
  (mindy_income_ratio : ℝ) 
  (combined_rate : ℝ)
  (h1 : mork_rate = 0.45)
  (h2 : mindy_income_ratio = 4)
  (h3 : combined_rate = 0.25) :
  ∃ mindy_rate : ℝ,
    mindy_rate * mindy_income_ratio * mork_rate + mork_rate = 
    combined_rate * (mindy_income_ratio + 1) ∧ 
    mindy_rate = 0.2 :=
by sorry

end NUMINAMATH_CALUDE_mindy_tax_rate_l2668_266852


namespace NUMINAMATH_CALUDE_circle_M_equation_l2668_266887

-- Define the line on which point M lies
def line (x y : ℝ) : Prop := 2 * x + y - 1 = 0

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 5

-- Define that points (3,0) and (0,1) lie on circle M
def points_on_circle : Prop := circle_M 3 0 ∧ circle_M 0 1

-- Theorem statement
theorem circle_M_equation : 
  ∃ (x y : ℝ), line x y ∧ points_on_circle → circle_M x y :=
sorry

end NUMINAMATH_CALUDE_circle_M_equation_l2668_266887


namespace NUMINAMATH_CALUDE_tank_capacity_is_21600_l2668_266824

/-- The capacity of a tank with specific inlet and outlet pipe properties -/
def tank_capacity : ℝ := by
  -- Define the time to empty the tank with only the outlet pipe open
  let outlet_time : ℝ := 10

  -- Define the inlet pipe rate in litres per minute
  let inlet_rate_per_minute : ℝ := 16

  -- Define the time to empty the tank with both pipes open
  let both_pipes_time : ℝ := 18

  -- Calculate the inlet rate in litres per hour
  let inlet_rate_per_hour : ℝ := inlet_rate_per_minute * 60

  -- The capacity of the tank
  exact 21600

/-- Theorem stating that the tank capacity is 21,600 litres -/
theorem tank_capacity_is_21600 : tank_capacity = 21600 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_is_21600_l2668_266824


namespace NUMINAMATH_CALUDE_max_triangle_area_l2668_266865

-- Define the curve C
def C (x y : ℝ) : Prop :=
  Real.sqrt ((x + 1)^2 + y^2) + Real.sqrt ((x - 1)^2 + y^2) + Real.sqrt (x^2 + (y - 1)^2) = 2 * Real.sqrt 2

-- Define the area of triangle F₁PF₂
def triangle_area (x y : ℝ) : ℝ :=
  abs (y) -- The base of the triangle is 2, so the area is |y|

-- Theorem statement
theorem max_triangle_area :
  ∀ x y : ℝ, C x y → triangle_area x y ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_max_triangle_area_l2668_266865


namespace NUMINAMATH_CALUDE_quadratic_rewrite_sum_l2668_266877

theorem quadratic_rewrite_sum (x : ℝ) : 
  ∃ (u v : ℝ), (9 * x^2 - 36 * x - 81 = 0 ↔ (x + u)^2 = v) ∧ u + v = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_sum_l2668_266877


namespace NUMINAMATH_CALUDE_absolute_difference_of_mn_l2668_266837

theorem absolute_difference_of_mn (m n : ℝ) 
  (h1 : m * n = 2) 
  (h2 : m + n = 6) : 
  |m - n| = 2 * Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_absolute_difference_of_mn_l2668_266837


namespace NUMINAMATH_CALUDE_crayons_left_l2668_266832

theorem crayons_left (initial : ℕ) (given_away : ℕ) (lost : ℕ) : 
  initial = 1453 → given_away = 563 → lost = 558 → 
  initial - given_away - lost = 332 := by
sorry

end NUMINAMATH_CALUDE_crayons_left_l2668_266832


namespace NUMINAMATH_CALUDE_inequality_implies_a_geq_4_l2668_266807

theorem inequality_implies_a_geq_4 (a : ℝ) (h_a_pos : a > 0) :
  (∀ x y : ℝ, x > 0 → y > 0 → (x + y) * (1 / x + a / y) ≥ 9) →
  a ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_inequality_implies_a_geq_4_l2668_266807


namespace NUMINAMATH_CALUDE_super_soup_expansion_l2668_266862

/-- The number of new stores opened by Super Soup in 2020 -/
def new_stores_2020 (initial_2018 : ℕ) (opened_2019 closed_2019 closed_2020 final_2020 : ℕ) : ℕ :=
  final_2020 - (initial_2018 + opened_2019 - closed_2019 - closed_2020)

/-- Theorem stating that Super Soup opened 10 new stores in 2020 -/
theorem super_soup_expansion :
  new_stores_2020 23 5 2 6 30 = 10 := by
  sorry

end NUMINAMATH_CALUDE_super_soup_expansion_l2668_266862


namespace NUMINAMATH_CALUDE_squares_below_specific_line_l2668_266827

/-- Represents a line in the coordinate plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Counts the number of unit squares below a line in the first quadrant -/
def countSquaresBelowLine (l : Line) : ℕ :=
  sorry

theorem squares_below_specific_line :
  let l : Line := { a := 5, b := 195, c := 975 }
  countSquaresBelowLine l = 388 := by sorry

end NUMINAMATH_CALUDE_squares_below_specific_line_l2668_266827


namespace NUMINAMATH_CALUDE_triangle_cosine_theorem_l2668_266844

theorem triangle_cosine_theorem (A : ℝ) (a m : ℝ) (θ : ℝ) : 
  A = 24 →
  a = 12 →
  m = 5 →
  A = (1/2) * a * m * Real.sin θ →
  Real.cos θ = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_theorem_l2668_266844


namespace NUMINAMATH_CALUDE_distance_2_5_distance_neg2_5_distance_x_3_solutions_abs_x_minus_1_int_solutions_sum_distances_min_value_sum_distances_l2668_266878

-- Define the distance function
def distance (a b : ℚ) : ℚ := |a - b|

-- Theorem 1: Distance between 2 and 5 is 3
theorem distance_2_5 : distance 2 5 = 3 := by sorry

-- Theorem 2: Distance between -2 and 5 is 7
theorem distance_neg2_5 : distance (-2) 5 = 7 := by sorry

-- Theorem 3: |x-3| represents the distance between x and 3
theorem distance_x_3 (x : ℚ) : |x - 3| = distance x 3 := by sorry

-- Theorem 4: Solutions of |x-1| = 3
theorem solutions_abs_x_minus_1 (x : ℚ) : |x - 1| = 3 ↔ x = 4 ∨ x = -2 := by sorry

-- Theorem 5: Integer solutions of |x-1| + |x+2| = 3
theorem int_solutions_sum_distances (x : ℤ) : 
  |x - 1| + |x + 2| = 3 ↔ x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1 := by sorry

-- Theorem 6: Minimum value of |x+8| + |x-3| + |x-6|
theorem min_value_sum_distances :
  ∃ (x : ℚ), ∀ (y : ℚ), |x + 8| + |x - 3| + |x - 6| ≤ |y + 8| + |y - 3| + |y - 6| ∧
  |x + 8| + |x - 3| + |x - 6| = 14 := by sorry

end NUMINAMATH_CALUDE_distance_2_5_distance_neg2_5_distance_x_3_solutions_abs_x_minus_1_int_solutions_sum_distances_min_value_sum_distances_l2668_266878


namespace NUMINAMATH_CALUDE_impossible_relationships_l2668_266810

theorem impossible_relationships (a b : ℝ) (h : 1 / a = 1 / b) :
  ¬(0 < a ∧ a < b) ∧ ¬(b < a ∧ a < 0) := by
  sorry

end NUMINAMATH_CALUDE_impossible_relationships_l2668_266810


namespace NUMINAMATH_CALUDE_parabola_distance_difference_l2668_266897

/-- Parabola type representing y² = 4x -/
structure Parabola where
  focus : ℝ × ℝ
  directrix : ℝ

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line type -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Function to check if a point is on the parabola -/
def on_parabola (p : Parabola) (pt : Point) : Prop :=
  pt.y^2 = 4 * pt.x

/-- Function to check if a point is on a line -/
def on_line (l : Line) (pt : Point) : Prop :=
  pt.y = l.slope * pt.x + l.intercept

/-- Function to check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

/-- Main theorem -/
theorem parabola_distance_difference 
  (p : Parabola)
  (F N A B : Point)
  (l : Line) :
  p.focus = (1, 0) →
  p.directrix = -1 →
  N.x = -1 ∧ N.y = 0 →
  on_parabola p A →
  on_parabola p B →
  on_line l A →
  on_line l B →
  on_line l F →
  perpendicular (Line.mk (B.y / (B.x - N.x)) 0) l →
  |A.x - F.x| - |B.x - F.x| = 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_distance_difference_l2668_266897


namespace NUMINAMATH_CALUDE_min_garden_cost_l2668_266848

-- Define the regions and their areas
def region1_area : ℕ := 10  -- 5x2
def region2_area : ℕ := 9   -- 3x3
def region3_area : ℕ := 20  -- 5x4
def region4_area : ℕ := 2   -- 2x1
def region5_area : ℕ := 7   -- 7x1

-- Define the flower costs
def aster_cost : ℚ := 1
def begonia_cost : ℚ := 2
def canna_cost : ℚ := 2
def dahlia_cost : ℚ := 3
def easter_lily_cost : ℚ := 2.5

-- Define the total garden area
def total_area : ℕ := region1_area + region2_area + region3_area + region4_area + region5_area

-- Theorem statement
theorem min_garden_cost : 
  ∃ (aster_count begonia_count canna_count dahlia_count easter_lily_count : ℕ),
    aster_count + begonia_count + canna_count + dahlia_count + easter_lily_count = total_area ∧
    aster_count * aster_cost + 
    begonia_count * begonia_cost + 
    canna_count * canna_cost + 
    dahlia_count * dahlia_cost + 
    easter_lily_count * easter_lily_cost = 81.5 ∧
    ∀ (a b c d e : ℕ),
      a + b + c + d + e = total_area →
      a * aster_cost + b * begonia_cost + c * canna_cost + d * dahlia_cost + e * easter_lily_cost ≥ 81.5 :=
by sorry

end NUMINAMATH_CALUDE_min_garden_cost_l2668_266848


namespace NUMINAMATH_CALUDE_modulo_eleven_residue_l2668_266801

theorem modulo_eleven_residue : (368 + 3 * 78 + 8 * 242 + 6 * 22) % 11 = 8 := by
  sorry

end NUMINAMATH_CALUDE_modulo_eleven_residue_l2668_266801


namespace NUMINAMATH_CALUDE_expression_simplification_l2668_266868

theorem expression_simplification (x : ℝ) : 
  ((((x + 2)^2 * (x^2 - 2*x + 2)^2) / (x^3 + 2)^2)^2 * 
   (((x - 2)^2 * (x^2 + 2*x + 2)^2) / (x^3 - 2)^2)^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2668_266868


namespace NUMINAMATH_CALUDE_solve_equation_l2668_266871

theorem solve_equation (x : ℝ) : 2*x - 3*x + 4*x = 150 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2668_266871


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2668_266800

theorem quadratic_equation_roots (m : ℝ) :
  ((-1 : ℝ)^2 + m * (-1) - 5 = 0) →
  (m = -4 ∧ ∃ x₂ : ℝ, x₂ = 5 ∧ x₂^2 + m * x₂ - 5 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2668_266800


namespace NUMINAMATH_CALUDE_jeans_cost_per_pair_l2668_266808

def leonard_cost : ℕ := 250
def michael_backpack_cost : ℕ := 100
def total_spent : ℕ := 450
def jeans_pairs : ℕ := 2

theorem jeans_cost_per_pair : 
  (total_spent - leonard_cost - michael_backpack_cost) / jeans_pairs = 50 :=
by sorry

end NUMINAMATH_CALUDE_jeans_cost_per_pair_l2668_266808


namespace NUMINAMATH_CALUDE_flag_distance_l2668_266847

theorem flag_distance (road_length : ℝ) (total_flags : ℕ) (h1 : road_length = 191.8) (h2 : total_flags = 58) :
  let intervals := total_flags / 2 - 1
  road_length / intervals = 6.85 := by
sorry

end NUMINAMATH_CALUDE_flag_distance_l2668_266847


namespace NUMINAMATH_CALUDE_alex_candles_left_l2668_266859

theorem alex_candles_left (initial_candles used_candles : ℕ) 
  (h1 : initial_candles = 44)
  (h2 : used_candles = 32) :
  initial_candles - used_candles = 12 := by
  sorry

end NUMINAMATH_CALUDE_alex_candles_left_l2668_266859


namespace NUMINAMATH_CALUDE_incorrect_derivation_l2668_266892

theorem incorrect_derivation : ¬ (∀ (a b c : ℝ), c > 0 → c / a > c / b → a < b) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_derivation_l2668_266892
