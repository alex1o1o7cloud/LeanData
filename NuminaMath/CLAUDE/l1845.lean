import Mathlib

namespace NUMINAMATH_CALUDE_line_slope_problem_l1845_184521

/-- Given a line passing through points (-1, -4) and (5, k) with slope k, prove that k = 4/5 -/
theorem line_slope_problem (k : ℚ) : 
  (k - (-4)) / (5 - (-1)) = k → k = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_problem_l1845_184521


namespace NUMINAMATH_CALUDE_original_number_proof_l1845_184516

theorem original_number_proof : ∃ x : ℝ, 16 * x = 3408 ∧ x = 213 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l1845_184516


namespace NUMINAMATH_CALUDE_sqrt_12_bounds_l1845_184518

theorem sqrt_12_bounds : 3 < Real.sqrt 12 ∧ Real.sqrt 12 < 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_12_bounds_l1845_184518


namespace NUMINAMATH_CALUDE_percentage_female_officers_on_duty_l1845_184527

def total_officers_on_duty : ℕ := 204
def female_ratio_on_duty : ℚ := 1/2
def total_female_officers : ℕ := 600

theorem percentage_female_officers_on_duty :
  (total_officers_on_duty * female_ratio_on_duty) / total_female_officers * 100 = 17 := by
  sorry

end NUMINAMATH_CALUDE_percentage_female_officers_on_duty_l1845_184527


namespace NUMINAMATH_CALUDE_largest_integral_x_l1845_184571

theorem largest_integral_x : ∃ (x : ℤ), 
  (1/4 : ℚ) < (x : ℚ)/7 ∧ (x : ℚ)/7 < 3/5 ∧ 
  ∀ (y : ℤ), ((1/4 : ℚ) < (y : ℚ)/7 ∧ (y : ℚ)/7 < 3/5) → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_largest_integral_x_l1845_184571


namespace NUMINAMATH_CALUDE_periodic_product_quotient_iff_commensurable_l1845_184512

theorem periodic_product_quotient_iff_commensurable 
  (f g : ℝ → ℝ) (T₁ T₂ : ℝ) 
  (hf : ∀ x, f (x + T₁) = f x) 
  (hg : ∀ x, g (x + T₂) = g x)
  (hpos_f : ∀ x, f x > 0)
  (hpos_g : ∀ x, g x > 0) :
  (∃ T, ∀ x, (f x * g x) = (f (x + T) * g (x + T)) ∧ 
            (f x / g x) = (f (x + T) / g (x + T))) ↔ 
  (∃ m n : ℤ, m ≠ 0 ∧ n ≠ 0 ∧ m * T₁ = n * T₂) :=
sorry

end NUMINAMATH_CALUDE_periodic_product_quotient_iff_commensurable_l1845_184512


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l1845_184575

theorem cubic_equation_solution : 
  ∀ x y : ℕ+, 
  (x : ℝ)^3 + (y : ℝ)^3 = 4 * ((x : ℝ)^2 * (y : ℝ) + (x : ℝ) * (y : ℝ)^2 - 5) → 
  ((x = 1 ∧ y = 3) ∨ (x = 3 ∧ y = 1)) := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l1845_184575


namespace NUMINAMATH_CALUDE_cubic_root_equation_solution_l1845_184517

theorem cubic_root_equation_solution :
  ∃ x : ℝ, (30 * x + (30 * x + 15) ^ (1/3)) ^ (1/3) = 15 ∧ x = 112 :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_root_equation_solution_l1845_184517


namespace NUMINAMATH_CALUDE_essay_introduction_length_l1845_184560

theorem essay_introduction_length 
  (total_words : ℕ) 
  (body_section_words : ℕ) 
  (body_section_count : ℕ) 
  (h1 : total_words = 5000)
  (h2 : body_section_words = 800)
  (h3 : body_section_count = 4) :
  ∃ (intro_words : ℕ),
    intro_words = 450 ∧ 
    total_words = intro_words + (body_section_count * body_section_words) + (3 * intro_words) :=
by sorry

end NUMINAMATH_CALUDE_essay_introduction_length_l1845_184560


namespace NUMINAMATH_CALUDE_no_solution_implies_m_leq_2_l1845_184546

theorem no_solution_implies_m_leq_2 (m : ℝ) :
  (∀ x : ℝ, ¬(x - 2 < 3*x - 6 ∧ x < m)) → m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_m_leq_2_l1845_184546


namespace NUMINAMATH_CALUDE_equation_solution_l1845_184544

theorem equation_solution :
  let S : Set ℂ := {x | (x - 1)^4 + (x - 1) = 0}
  S = {1, 0, Complex.mk 1 (Real.sqrt 3 / 2), Complex.mk 1 (-Real.sqrt 3 / 2)} := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1845_184544


namespace NUMINAMATH_CALUDE_ram_price_increase_l1845_184561

theorem ram_price_increase (original_price current_price : ℝ) 
  (h1 : original_price = 50)
  (h2 : current_price = 52)
  (h3 : current_price = 0.8 * (original_price * (1 + increase_percentage / 100))) :
  increase_percentage = 30 := by
  sorry

end NUMINAMATH_CALUDE_ram_price_increase_l1845_184561


namespace NUMINAMATH_CALUDE_pair_2017_is_1_64_l1845_184538

/-- Represents an integer pair -/
structure IntPair :=
  (first : ℕ)
  (second : ℕ)

/-- Calculates the total number of pairs in the first n groups -/
def totalPairsInGroups (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- Generates the first pair of the nth group -/
def firstPairOfGroup (n : ℕ) : IntPair :=
  ⟨1, n⟩

theorem pair_2017_is_1_64 :
  ∃ n : ℕ, totalPairsInGroups n = 2016 ∧ firstPairOfGroup (n + 1) = ⟨1, 64⟩ := by
  sorry

#check pair_2017_is_1_64

end NUMINAMATH_CALUDE_pair_2017_is_1_64_l1845_184538


namespace NUMINAMATH_CALUDE_right_triangle_ac_length_l1845_184587

/-- 
Given a right triangle ABC in the x-y plane where:
- ∠B = 90°
- The slope of line segment AC is 4/3
- The length of AB is 20

Prove that the length of AC is 25.
-/
theorem right_triangle_ac_length 
  (A B C : ℝ × ℝ) 
  (right_angle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0)
  (slope_ac : (C.2 - A.2) / (C.1 - A.1) = 4 / 3)
  (length_ab : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 20) :
  Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 25 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_ac_length_l1845_184587


namespace NUMINAMATH_CALUDE_problem_statement_l1845_184572

theorem problem_statement (a b : ℝ) (ha : a > 0) (heq : Real.exp a + Real.log b = 1) :
  (a + Real.log b < 0) ∧ (Real.exp a + b > 2) ∧ (a + b > 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1845_184572


namespace NUMINAMATH_CALUDE_certain_number_existence_and_value_l1845_184559

theorem certain_number_existence_and_value :
  ∃ n : ℝ, 8 * n - (0.6 * 10) / 1.2 = 31.000000000000004 ∧ 
  ∃ ε > 0, |n - 4.5| < ε := by
  sorry

end NUMINAMATH_CALUDE_certain_number_existence_and_value_l1845_184559


namespace NUMINAMATH_CALUDE_mikes_age_l1845_184574

theorem mikes_age (claire_age jessica_age mike_age : ℕ) : 
  jessica_age = claire_age + 6 →
  claire_age + 2 = 20 →
  mike_age = 2 * (jessica_age - 3) →
  mike_age = 42 := by
  sorry

end NUMINAMATH_CALUDE_mikes_age_l1845_184574


namespace NUMINAMATH_CALUDE_softball_team_size_l1845_184507

/-- Proves that a co-ed softball team with 5 more women than men and a men-to-women ratio of 0.5 has 15 total players -/
theorem softball_team_size (men women : ℕ) : 
  women = men + 5 →
  men / women = 1 / 2 →
  men + women = 15 := by
sorry

end NUMINAMATH_CALUDE_softball_team_size_l1845_184507


namespace NUMINAMATH_CALUDE_blue_candy_count_l1845_184503

theorem blue_candy_count (total : ℕ) (red : ℕ) (blue : ℕ) 
  (h1 : total = 3409)
  (h2 : red = 145)
  (h3 : blue = total - red) :
  blue = 3264 := by
  sorry

end NUMINAMATH_CALUDE_blue_candy_count_l1845_184503


namespace NUMINAMATH_CALUDE_problem_solution_l1845_184501

def A (x : ℝ) : Set ℝ := {x^2 + 2, -x, -x - 1}
def B (y : ℝ) : Set ℝ := {-y, -y/2, y + 1}

theorem problem_solution (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (A x = B y → x^2 + y^2 = 5) ∧
  (A x ∩ B y = {6} → A x ∪ B y = {-2, -3, -5, -5/2, 6}) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1845_184501


namespace NUMINAMATH_CALUDE_family_road_trip_l1845_184522

/-- A theorem about a family's road trip with constant speed -/
theorem family_road_trip 
  (total_time : ℝ) 
  (first_part_distance : ℝ) 
  (first_part_time : ℝ) 
  (h1 : total_time = 4) 
  (h2 : first_part_distance = 100) 
  (h3 : first_part_time = 1) :
  let speed := first_part_distance / first_part_time
  let remaining_time := total_time - first_part_time
  remaining_time * speed = 300 := by
  sorry

#check family_road_trip

end NUMINAMATH_CALUDE_family_road_trip_l1845_184522


namespace NUMINAMATH_CALUDE_data_set_mode_l1845_184530

def data_set : List ℕ := [9, 7, 10, 8, 10, 9, 10]

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem data_set_mode :
  mode data_set = 10 := by sorry

end NUMINAMATH_CALUDE_data_set_mode_l1845_184530


namespace NUMINAMATH_CALUDE_orthocenter_of_triangle_l1845_184539

/-- The orthocenter of a triangle in 3D space -/
def orthocenter (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem: The orthocenter of triangle ABC is at (-29/12, 77/12, 49/12) -/
theorem orthocenter_of_triangle :
  let A : ℝ × ℝ × ℝ := (2, 3, 1)
  let B : ℝ × ℝ × ℝ := (4, -1, 2)
  let C : ℝ × ℝ × ℝ := (1, 1, 4)
  orthocenter A B C = (-29/12, 77/12, 49/12) := by sorry

end NUMINAMATH_CALUDE_orthocenter_of_triangle_l1845_184539


namespace NUMINAMATH_CALUDE_equation_implication_l1845_184531

theorem equation_implication (x : ℝ) : 3 * x + 2 = 11 → 6 * x + 4 = 22 := by
  sorry

end NUMINAMATH_CALUDE_equation_implication_l1845_184531


namespace NUMINAMATH_CALUDE_zero_last_to_appear_l1845_184508

/-- Modified Fibonacci sequence -/
def modFib : ℕ → ℕ
  | 0 => 2
  | 1 => 1
  | (n + 2) => modFib (n + 1) + modFib n

/-- The set of digits that have appeared in the units position up to the nth term -/
def digitsAppeared (n : ℕ) : Finset ℕ :=
  Finset.filter (fun d => ∃ k ≤ n, modFib k % 10 = d) (Finset.range 10)

/-- The proposition that 0 is the last digit to appear in the units position -/
theorem zero_last_to_appear : ∃ N : ℕ, 
  (∀ n ≥ N, 0 ∈ digitsAppeared n) ∧ 
  (∀ d : ℕ, d < 10 → d ≠ 0 → ∃ n < N, d ∈ digitsAppeared n) :=
sorry

end NUMINAMATH_CALUDE_zero_last_to_appear_l1845_184508


namespace NUMINAMATH_CALUDE_sum_ages_in_5_years_l1845_184578

/-- Represents the ages of three siblings and their relationships -/
structure SiblingAges where
  will_age_3_years_ago : ℕ
  diane_age : ℕ
  will_age : ℕ
  janet_age : ℕ

/-- Calculates the sum of ages after a given number of years -/
def sum_ages_after (s : SiblingAges) (years : ℕ) : ℕ :=
  s.will_age + years + s.diane_age + years + s.janet_age + years

/-- Theorem stating the sum of ages in 5 years will be 53 -/
theorem sum_ages_in_5_years (s : SiblingAges) 
  (h1 : s.will_age_3_years_ago = 4)
  (h2 : s.will_age = s.will_age_3_years_ago + 3)
  (h3 : s.diane_age = 2 * s.will_age)
  (h4 : s.janet_age = s.diane_age + 3) :
  sum_ages_after s 5 = 53 := by
  sorry

end NUMINAMATH_CALUDE_sum_ages_in_5_years_l1845_184578


namespace NUMINAMATH_CALUDE_nara_height_l1845_184509

/-- Given the heights of Sangheon, Chiho, and Nara, prove Nara's height -/
theorem nara_height (sangheon_height : Real) (chiho_diff : Real) (nara_diff : Real)
  (h1 : sangheon_height = 1.56)
  (h2 : chiho_diff = 0.14)
  (h3 : nara_diff = 0.27) :
  sangheon_height - chiho_diff + nara_diff = 1.69 := by
  sorry


end NUMINAMATH_CALUDE_nara_height_l1845_184509


namespace NUMINAMATH_CALUDE_florist_roses_theorem_l1845_184502

/-- Represents the number of roses picked in the first picking -/
def first_picking : ℝ := 16.0

theorem florist_roses_theorem (initial : ℝ) (second_picking : ℝ) (final_total : ℝ) :
  initial = 37.0 →
  second_picking = 19.0 →
  final_total = 72 →
  initial + first_picking + second_picking = final_total :=
by
  sorry

#check florist_roses_theorem

end NUMINAMATH_CALUDE_florist_roses_theorem_l1845_184502


namespace NUMINAMATH_CALUDE_cabbage_sales_theorem_l1845_184520

/-- Calculates the total kilograms of cabbage sold given the price per kilogram and earnings from three days -/
def total_cabbage_sold (price_per_kg : ℚ) (day1_earnings day2_earnings day3_earnings : ℚ) : ℚ :=
  (day1_earnings + day2_earnings + day3_earnings) / price_per_kg

/-- Theorem stating that given the specific conditions, the total cabbage sold is 48 kg -/
theorem cabbage_sales_theorem :
  let price_per_kg : ℚ := 2
  let day1_earnings : ℚ := 30
  let day2_earnings : ℚ := 24
  let day3_earnings : ℚ := 42
  total_cabbage_sold price_per_kg day1_earnings day2_earnings day3_earnings = 48 := by
  sorry

end NUMINAMATH_CALUDE_cabbage_sales_theorem_l1845_184520


namespace NUMINAMATH_CALUDE_quadrilateral_area_l1845_184563

/-- The area of a quadrilateral given its perspective drawing properties -/
theorem quadrilateral_area (base_angle : ℝ) (leg_length : ℝ) (top_base_length : ℝ) : 
  base_angle = π / 4 →
  leg_length = 1 →
  top_base_length = 1 →
  (2 + Real.sqrt 2) = 
    (((1 + top_base_length + Real.sqrt 2) * (Real.sqrt 2 / 2)) / 2) * (2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l1845_184563


namespace NUMINAMATH_CALUDE_sum_of_powers_l1845_184550

theorem sum_of_powers (ω : ℂ) (h1 : ω^9 = 1) (h2 : ω ≠ 1) :
  ω^20 + ω^24 + ω^28 + ω^32 + ω^36 + ω^40 + ω^44 + ω^48 + ω^52 + ω^56 + ω^60 + ω^64 + ω^68 = (ω^2 - 1) / (ω^4 - 1) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l1845_184550


namespace NUMINAMATH_CALUDE_perpendicular_bisector_intersection_equidistant_l1845_184599

-- Define a triangle in a 2D plane
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a function to calculate the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define a function to find the intersection point of perpendicular bisectors
def intersectionOfPerpendicularBisectors (t : Triangle) : ℝ × ℝ := sorry

-- Theorem statement
theorem perpendicular_bisector_intersection_equidistant (t : Triangle) :
  let P := intersectionOfPerpendicularBisectors t
  distance P t.A = distance P t.B ∧ distance P t.B = distance P t.C := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_intersection_equidistant_l1845_184599


namespace NUMINAMATH_CALUDE_root_difference_absolute_value_specific_root_difference_l1845_184524

theorem root_difference_absolute_value (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * x^2 + b * x + c = 0 → |r₁ - r₂| = 1 :=
by
  sorry

-- Specific instance for the given problem
theorem specific_root_difference :
  let r₁ := (-(-7) + Real.sqrt ((-7)^2 - 4*1*12)) / (2*1)
  let r₂ := (-(-7) - Real.sqrt ((-7)^2 - 4*1*12)) / (2*1)
  x^2 - 7*x + 12 = 0 → |r₁ - r₂| = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_root_difference_absolute_value_specific_root_difference_l1845_184524


namespace NUMINAMATH_CALUDE_sum_of_squares_specific_numbers_l1845_184515

theorem sum_of_squares_specific_numbers : 52^2 + 81^2 + 111^2 = 21586 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_specific_numbers_l1845_184515


namespace NUMINAMATH_CALUDE_depth_of_iron_cone_in_mercury_l1845_184582

/-- The depth of submersion of an iron cone in mercury -/
noncomputable def depth_of_submersion (cone_volume : ℝ) (iron_density : ℝ) (mercury_density : ℝ) : ℝ :=
  let submerged_volume := (iron_density * cone_volume) / mercury_density
  (3 * submerged_volume / Real.pi) ^ (1/3)

/-- The theorem stating the depth of submersion for the given problem -/
theorem depth_of_iron_cone_in_mercury :
  let cone_volume : ℝ := 350
  let iron_density : ℝ := 7.2
  let mercury_density : ℝ := 13.6
  abs (depth_of_submersion cone_volume iron_density mercury_density - 5.6141) < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_depth_of_iron_cone_in_mercury_l1845_184582


namespace NUMINAMATH_CALUDE_identity_function_only_l1845_184523

def P (f : ℕ → ℕ) : ℕ → ℕ 
  | 0 => 1
  | n + 1 => (P f n) * (f (n + 1))

theorem identity_function_only (f : ℕ → ℕ) :
  (∀ a b : ℕ, (P f a + P f b) ∣ (Nat.factorial a + Nat.factorial b)) →
  (∀ n : ℕ, f n = n) := by
  sorry

end NUMINAMATH_CALUDE_identity_function_only_l1845_184523


namespace NUMINAMATH_CALUDE_container_volume_ratio_l1845_184564

theorem container_volume_ratio (volume_first volume_second : ℚ) : 
  volume_first > 0 →
  volume_second > 0 →
  (4 / 5 : ℚ) * volume_first = (2 / 3 : ℚ) * volume_second →
  volume_first / volume_second = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l1845_184564


namespace NUMINAMATH_CALUDE_log_equation_solution_l1845_184525

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 3 + Real.log x / Real.log 9 = 7 →
  x = 3 ^ (14 / 3) := by
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1845_184525


namespace NUMINAMATH_CALUDE_star_value_for_specific_conditions_l1845_184569

-- Define the operation * for non-zero integers
def star (a b : ℤ) : ℚ := (1 : ℚ) / a + (1 : ℚ) / b

-- Theorem statement
theorem star_value_for_specific_conditions 
  (a b : ℤ) 
  (h1 : a ≠ 0) 
  (h2 : b ≠ 0) 
  (h3 : a + b = 15) 
  (h4 : a * b = 36) : 
  star a b = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_star_value_for_specific_conditions_l1845_184569


namespace NUMINAMATH_CALUDE_four_digit_number_theorem_l1845_184589

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def reverse_digits (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  d4 * 1000 + d3 * 100 + d2 * 10 + d1

theorem four_digit_number_theorem (n : ℕ) : 
  is_four_digit n ∧
  (∃ k : ℕ, n + 1 = 15 * k) ∧
  (∃ m : ℕ, n - 3 = 38 * m) ∧
  (∃ l : ℕ, n + reverse_digits n = 10 * l) →
  n = 1409 ∨ n = 1979 :=
by sorry

end NUMINAMATH_CALUDE_four_digit_number_theorem_l1845_184589


namespace NUMINAMATH_CALUDE_smallest_factorizable_b_l1845_184541

theorem smallest_factorizable_b : ∃ (b : ℕ), 
  (∀ (x : ℤ), ∃ (p q : ℤ), x^2 + b*x + 2520 = (x + p) * (x + q)) ∧
  (∀ (b' : ℕ), b' < b → ¬∃ (p q : ℤ), ∀ (x : ℤ), x^2 + b'*x + 2520 = (x + p) * (x + q)) ∧
  b = 106 :=
sorry

end NUMINAMATH_CALUDE_smallest_factorizable_b_l1845_184541


namespace NUMINAMATH_CALUDE_circle_condition_intersection_condition_l1845_184542

-- Define the equation of the circle
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the equation of the line
def line_equation (x y : ℝ) : Prop :=
  4*x - 3*y + 7 = 0

-- Theorem 1: If the equation represents a circle, then m < 5
theorem circle_condition (m : ℝ) :
  (∃ x y : ℝ, circle_equation x y m) → m < 5 :=
by sorry

-- Theorem 2: If the circle intersects the line at two points M and N where |MN| = 2√3, then m = 1
theorem intersection_condition (m : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    circle_equation x₁ y₁ m ∧ 
    circle_equation x₂ y₂ m ∧
    line_equation x₁ y₁ ∧ 
    line_equation x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 12) →
  m = 1 :=
by sorry

end NUMINAMATH_CALUDE_circle_condition_intersection_condition_l1845_184542


namespace NUMINAMATH_CALUDE_combine_squares_simplify_expression_linear_combination_l1845_184532

-- Part 1
theorem combine_squares (a b : ℝ) :
  3 * (a - b)^2 - 6 * (a - b)^2 + 2 * (a - b)^2 = -(a - b)^2 := by sorry

-- Part 2
theorem simplify_expression (x y : ℝ) (h : x^2 - 2*y = 4) :
  3*x^2 - 6*y - 21 = -9 := by sorry

-- Part 3
theorem linear_combination (a b c d : ℝ) 
  (h1 : a - 5*b = 3) (h2 : 5*b - 3*c = -5) (h3 : 3*c - d = 10) :
  (a - 3*c) + (5*b - d) - (5*b - 3*c) = 8 := by sorry

end NUMINAMATH_CALUDE_combine_squares_simplify_expression_linear_combination_l1845_184532


namespace NUMINAMATH_CALUDE_no_real_roots_l1845_184584

theorem no_real_roots : ∀ x : ℝ, x^2 + |x| + 1 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l1845_184584


namespace NUMINAMATH_CALUDE_sons_age_l1845_184580

/-- Proves that the son's age is 7.5 years given the conditions of the problem -/
theorem sons_age (son_age man_age : ℝ) : 
  man_age = son_age + 25 →
  man_age + 5 = 3 * (son_age + 5) →
  son_age = 7.5 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l1845_184580


namespace NUMINAMATH_CALUDE_flag_pole_shadow_length_l1845_184562

/-- Given a tree and a flag pole, calculate the length of the flag pole's shadow -/
theorem flag_pole_shadow_length 
  (tree_height : ℝ) 
  (tree_shadow : ℝ) 
  (flag_pole_height : ℝ) 
  (h1 : tree_height = 12) 
  (h2 : tree_shadow = 8) 
  (h3 : flag_pole_height = 150) : 
  (flag_pole_height * tree_shadow) / tree_height = 100 := by
  sorry

#check flag_pole_shadow_length

end NUMINAMATH_CALUDE_flag_pole_shadow_length_l1845_184562


namespace NUMINAMATH_CALUDE_third_grade_boys_count_l1845_184500

theorem third_grade_boys_count (total : ℕ) (difference : ℕ) : total = 41 → difference = 3 → 2 * (total - difference) / 2 + difference = 22 := by
  sorry

end NUMINAMATH_CALUDE_third_grade_boys_count_l1845_184500


namespace NUMINAMATH_CALUDE_unique_solution_modular_equation_l1845_184593

theorem unique_solution_modular_equation :
  ∃! n : ℤ, 0 ≤ n ∧ n < 102 ∧ 99 * n % 102 = 73 % 102 ∧ n = 97 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_modular_equation_l1845_184593


namespace NUMINAMATH_CALUDE_point_P_coordinates_l1845_184506

/-- Given a point P with coordinates (3m+6, m-3), prove its coordinates under different conditions --/
theorem point_P_coordinates (m : ℝ) :
  let P : ℝ × ℝ := (3*m + 6, m - 3)
  -- Condition 1: P lies on the angle bisector in the first and third quadrants
  (P.1 = P.2 → P = (-7.5, -7.5)) ∧
  -- Condition 2: The ordinate of P is 5 greater than the abscissa
  (P.2 = P.1 + 5 → P = (-15, -10)) ∧
  -- Condition 3: P lies on the line passing through A(3, -2) and parallel to the y-axis
  (P.1 = 3 → P = (3, -4)) := by sorry

end NUMINAMATH_CALUDE_point_P_coordinates_l1845_184506


namespace NUMINAMATH_CALUDE_range_of_a_l1845_184592

-- Define set A
def A : Set ℝ := {x | x^2 - x < 0}

-- Define set B
def B (a : ℝ) : Set ℝ := Set.Ioo 0 a

-- Theorem statement
theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : A ⊆ B a) : a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1845_184592


namespace NUMINAMATH_CALUDE_remaining_integers_count_l1845_184597

def S : Finset Nat := Finset.range 51 \ {0}

theorem remaining_integers_count : 
  (S.filter (fun n => n % 2 ≠ 0 ∧ n % 3 ≠ 0)).card = 17 := by
  sorry

end NUMINAMATH_CALUDE_remaining_integers_count_l1845_184597


namespace NUMINAMATH_CALUDE_max_money_collectible_l1845_184553

-- Define the structure of the land plot
structure LandPlot where
  circles : Fin 36 → ℕ
  -- circles represents the amount of money in each of the 36 circles

-- Define the concept of a valid path
def ValidPath (plot : LandPlot) (path : List (Fin 36)) : Prop :=
  -- A path is valid if it doesn't pass twice along the same straight line
  -- The actual implementation of this condition is complex and omitted here
  sorry

-- Define the sum of money collected along a path
def PathSum (plot : LandPlot) (path : List (Fin 36)) : ℕ :=
  path.map plot.circles |> List.sum

-- The main theorem
theorem max_money_collectible (plot : LandPlot) : 
  (∃ (path : List (Fin 36)), ValidPath plot path ∧ PathSum plot path = 47) ∧
  (∀ (path : List (Fin 36)), ValidPath plot path → PathSum plot path ≤ 47) := by
  sorry

end NUMINAMATH_CALUDE_max_money_collectible_l1845_184553


namespace NUMINAMATH_CALUDE_B_is_midpoint_of_AC_l1845_184534

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define points A, B, C, and O
variable (O A B C : V)

-- Define the collinearity of points A, B, and C
def collinear (A B C : V) : Prop :=
  ∃ t : ℝ, B - A = t • (C - A)

-- Define the vector equation
def vector_equation (m : ℝ) : Prop :=
  m • (A - O) - 2 • (B - O) + (C - O) = 0

-- Theorem statement
theorem B_is_midpoint_of_AC 
  (h_collinear : collinear A B C)
  (h_equation : ∃ m : ℝ, vector_equation O A B C m) :
  B - O = (1/2) • ((A - O) + (C - O)) :=
sorry

end NUMINAMATH_CALUDE_B_is_midpoint_of_AC_l1845_184534


namespace NUMINAMATH_CALUDE_lily_remaining_milk_l1845_184513

theorem lily_remaining_milk (initial_milk : ℚ) (james_milk : ℚ) (maria_milk : ℚ) :
  initial_milk = 5 →
  james_milk = 15 / 4 →
  maria_milk = 3 / 4 →
  initial_milk - (james_milk + maria_milk) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_lily_remaining_milk_l1845_184513


namespace NUMINAMATH_CALUDE_second_sunday_on_13th_l1845_184519

/-- Represents a day of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a month with specific properties -/
structure Month where
  /-- The day of the week on which the month starts -/
  startDay : DayOfWeek
  /-- The number of days in the month -/
  numDays : Nat
  /-- Predicate that is true if three Wednesdays fall on even dates -/
  threeWednesdaysOnEvenDates : Prop

/-- Given a month and a day number, returns the day of the week -/
def dayOfWeek (m : Month) (day : Nat) : DayOfWeek :=
  sorry

/-- Predicate that is true if the given day is a Sunday -/
def isSunday (dow : DayOfWeek) : Prop :=
  sorry

/-- Returns the date of the nth occurrence of a specific day in the month -/
def nthOccurrence (m : Month) (dow : DayOfWeek) (n : Nat) : Nat :=
  sorry

/-- Theorem stating that in a month where three Wednesdays fall on even dates, 
    the second Sunday of that month falls on the 13th -/
theorem second_sunday_on_13th (m : Month) :
  m.threeWednesdaysOnEvenDates → nthOccurrence m DayOfWeek.Sunday 2 = 13 :=
sorry

end NUMINAMATH_CALUDE_second_sunday_on_13th_l1845_184519


namespace NUMINAMATH_CALUDE_distance_covered_l1845_184577

/-- Proves that the total distance covered is 16 km given the specified conditions -/
theorem distance_covered (walking_speed running_speed : ℝ) (total_time : ℝ) :
  walking_speed = 4 →
  running_speed = 8 →
  total_time = 3 →
  ∃ (distance : ℝ),
    distance / walking_speed / 2 + distance / running_speed / 2 = total_time ∧
    distance = 16 := by
  sorry

end NUMINAMATH_CALUDE_distance_covered_l1845_184577


namespace NUMINAMATH_CALUDE_geometric_series_sum_l1845_184579

theorem geometric_series_sum : 
  let a : ℚ := 2
  let r : ℚ := -2/5
  let series : ℕ → ℚ := λ n => a * r^n
  ∑' n, series n = 10/7 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l1845_184579


namespace NUMINAMATH_CALUDE_number_division_problem_l1845_184591

theorem number_division_problem :
  ∃ x : ℝ, (x / 5 = 60 + x / 6) ∧ (x = 1800) := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l1845_184591


namespace NUMINAMATH_CALUDE_xyz_value_l1845_184585

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 49)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 19) :
  x * y * z = 10 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l1845_184585


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l1845_184547

def arithmeticSequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ :=
  λ n => a₁ + d * (n - 1)

theorem arithmetic_sequence_general_term :
  let a₁ : ℝ := -1
  let d : ℝ := 4
  ∀ n : ℕ, arithmeticSequence a₁ d n = 4 * n - 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l1845_184547


namespace NUMINAMATH_CALUDE_pitcher_problem_l1845_184511

theorem pitcher_problem (pitcher_capacity : ℝ) (h_positive : pitcher_capacity > 0) :
  let juice_amount : ℝ := (2/3) * pitcher_capacity
  let num_cups : ℕ := 6
  let juice_per_cup : ℝ := juice_amount / num_cups
  (juice_per_cup / pitcher_capacity) * 100 = 11.1111111111 :=
by sorry

end NUMINAMATH_CALUDE_pitcher_problem_l1845_184511


namespace NUMINAMATH_CALUDE_roots_of_unity_sum_one_l1845_184595

theorem roots_of_unity_sum_one (n : ℕ) (h : Even n) (h_pos : n > 0) :
  ∃ (z₁ z₂ z₃ : ℂ), (z₁^n = 1) ∧ (z₂^n = 1) ∧ (z₃^n = 1) ∧ (z₁ + z₂ + z₃ = 1) :=
sorry

end NUMINAMATH_CALUDE_roots_of_unity_sum_one_l1845_184595


namespace NUMINAMATH_CALUDE_pauls_earnings_duration_l1845_184533

/-- Calculates how many weeks Paul's earnings will last given his weekly earnings and expenses. -/
def weeks_earnings_last (lawn_mowing : ℚ) (weed_eating : ℚ) (bush_trimming : ℚ) (fence_painting : ℚ)
                        (food_expense : ℚ) (transportation_expense : ℚ) (entertainment_expense : ℚ) : ℚ :=
  (lawn_mowing + weed_eating + bush_trimming + fence_painting) /
  (food_expense + transportation_expense + entertainment_expense)

/-- Theorem stating that Paul's earnings will last 2.5 weeks given his specific earnings and expenses. -/
theorem pauls_earnings_duration :
  weeks_earnings_last 12 8 5 20 10 5 3 = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_pauls_earnings_duration_l1845_184533


namespace NUMINAMATH_CALUDE_integral_x_squared_l1845_184558

theorem integral_x_squared : ∫ x in (0:ℝ)..(1:ℝ), x^2 = (1:ℝ)/3 := by sorry

end NUMINAMATH_CALUDE_integral_x_squared_l1845_184558


namespace NUMINAMATH_CALUDE_apples_in_basket_l1845_184540

/-- The number of apples left in a basket after removals --/
def applesLeft (initial : ℕ) (rickiRemoves : ℕ) : ℕ :=
  initial - rickiRemoves - (2 * rickiRemoves)

/-- Theorem stating the number of apples left in the basket --/
theorem apples_in_basket : applesLeft 74 14 = 32 := by
  sorry

end NUMINAMATH_CALUDE_apples_in_basket_l1845_184540


namespace NUMINAMATH_CALUDE_kimberly_skittles_l1845_184510

/-- Given that Kimberly buys 7 more Skittles and ends up with 12 Skittles in total,
    prove that she initially had 5 Skittles. -/
theorem kimberly_skittles (bought : ℕ) (total : ℕ) (initial : ℕ) : 
  bought = 7 → total = 12 → initial + bought = total → initial = 5 := by
  sorry

end NUMINAMATH_CALUDE_kimberly_skittles_l1845_184510


namespace NUMINAMATH_CALUDE_squirrel_acorns_l1845_184505

theorem squirrel_acorns (initial_acorns : ℕ) (winter_months : ℕ) (remaining_per_month : ℕ) : 
  initial_acorns = 210 →
  winter_months = 3 →
  remaining_per_month = 60 →
  initial_acorns - (winter_months * remaining_per_month) = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_squirrel_acorns_l1845_184505


namespace NUMINAMATH_CALUDE_switcheroo_period_l1845_184548

/-- Represents a word of length 2^n -/
def Word (n : ℕ) := Fin (2^n) → Char

/-- Performs a single switcheroo operation on a word -/
def switcheroo (n : ℕ) (w : Word n) : Word n :=
  sorry

/-- Returns true if two words are equal -/
def word_eq (n : ℕ) (w1 w2 : Word n) : Prop :=
  ∀ i, w1 i = w2 i

/-- Applies the switcheroo operation m times -/
def apply_switcheroo (n m : ℕ) (w : Word n) : Word n :=
  sorry

theorem switcheroo_period (n : ℕ) :
  ∀ w : Word n, word_eq n (apply_switcheroo n (2^n) w) w ∧
  ∀ m : ℕ, m < 2^n → ¬(word_eq n (apply_switcheroo n m w) w) :=
by sorry

end NUMINAMATH_CALUDE_switcheroo_period_l1845_184548


namespace NUMINAMATH_CALUDE_special_subset_count_l1845_184581

def subset_count (n : ℕ) : ℕ :=
  (Finset.range 11).sum (fun k => Nat.choose (n - k + 1) k)

theorem special_subset_count : subset_count 20 = 3164 := by
  sorry

end NUMINAMATH_CALUDE_special_subset_count_l1845_184581


namespace NUMINAMATH_CALUDE_three_digit_sum_property_l1845_184588

theorem three_digit_sum_property : ∃! n : ℕ, 
  100 ≤ n ∧ n < 1000 ∧
  (∃ x y z : ℕ, 
    n = 100 * x + 10 * y + z ∧
    x < 10 ∧ y < 10 ∧ z < 10 ∧
    n = y + x^2 + z^3) ∧
  n = 357 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_sum_property_l1845_184588


namespace NUMINAMATH_CALUDE_circle_center_sum_l1845_184526

/-- Given a circle with equation x^2 + y^2 - 6x + 8y + 9 = 0, 
    prove that the sum of the coordinates of its center is -1 -/
theorem circle_center_sum (h k : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 6*x + 8*y + 9 = 0 ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 9)) →
  h + k = -1 := by
sorry

end NUMINAMATH_CALUDE_circle_center_sum_l1845_184526


namespace NUMINAMATH_CALUDE_souvenir_sales_problem_l1845_184566

/-- Souvenir sales problem -/
theorem souvenir_sales_problem
  (cost_price : ℕ)
  (initial_price : ℕ)
  (initial_sales : ℕ)
  (price_change : ℕ → ℤ)
  (sales_change : ℕ → ℤ)
  (h1 : cost_price = 40)
  (h2 : initial_price = 44)
  (h3 : initial_sales = 300)
  (h4 : ∀ x : ℕ, price_change x = x)
  (h5 : ∀ x : ℕ, sales_change x = -10 * x)
  (h6 : ∀ x : ℕ, initial_price + price_change x ≥ 44)
  (h7 : ∀ x : ℕ, initial_price + price_change x ≤ 60) :
  (∃ x : ℕ, (initial_price + price_change x - cost_price) * (initial_sales + sales_change x) = 2640 ∧
             initial_price + price_change x = 52) ∧
  (∃ x : ℕ, ∀ y : ℕ, 
    (initial_price + price_change x - cost_price) * (initial_sales + sales_change x) ≥
    (initial_price + price_change y - cost_price) * (initial_sales + sales_change y) ∧
    initial_price + price_change x = 57) ∧
  (∃ max_profit : ℕ, 
    (∃ x : ℕ, (initial_price + price_change x - cost_price) * (initial_sales + sales_change x) = max_profit) ∧
    (∀ y : ℕ, (initial_price + price_change y - cost_price) * (initial_sales + sales_change y) ≤ max_profit) ∧
    max_profit = 2890) :=
by sorry

end NUMINAMATH_CALUDE_souvenir_sales_problem_l1845_184566


namespace NUMINAMATH_CALUDE_x_squared_plus_inverse_x_squared_l1845_184549

theorem x_squared_plus_inverse_x_squared (x : ℝ) (h : x - 1/x = 3) : x^2 + 1/x^2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_inverse_x_squared_l1845_184549


namespace NUMINAMATH_CALUDE_root_power_equality_l1845_184529

theorem root_power_equality (x : ℝ) (h : x > 0) :
  (x^((1:ℝ)/5)) / (x^((1:ℝ)/2)) = x^(-(3:ℝ)/10) := by sorry

end NUMINAMATH_CALUDE_root_power_equality_l1845_184529


namespace NUMINAMATH_CALUDE_certain_number_sum_l1845_184556

theorem certain_number_sum : ∃ x : ℝ, x = 5.46 - 3.97 ∧ x + 5.46 = 6.95 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_sum_l1845_184556


namespace NUMINAMATH_CALUDE_ratio_arithmetic_properties_l1845_184594

-- Define a ratio arithmetic sequence
def is_ratio_arithmetic_seq (p : ℕ → ℝ) (k : ℝ) :=
  ∀ n ≥ 2, p (n + 1) / p n - p n / p (n - 1) = k

-- Define a geometric sequence
def is_geometric_seq (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = q * a n

-- Define an arithmetic sequence
def is_arithmetic_seq (b : ℕ → ℝ) (d : ℝ) :=
  ∀ n, b (n + 1) = b n + d

-- Define the Fibonacci-like sequence
def fib_like (a : ℕ → ℝ) :=
  a 1 = 1 ∧ a 2 = 1 ∧ ∀ n ≥ 2, a (n + 1) = a n + a (n - 1)

theorem ratio_arithmetic_properties :
  (∀ a q, q ≠ 0 → is_geometric_seq a q → is_ratio_arithmetic_seq a 0) ∧
  (∃ b d, is_arithmetic_seq b d ∧ ∃ k, is_ratio_arithmetic_seq b k) ∧
  (∃ a b q d, is_arithmetic_seq a d ∧ is_geometric_seq b q ∧
    ¬∃ k, is_ratio_arithmetic_seq (fun n ↦ a n * b n) k) ∧
  (∀ a, fib_like a → ¬∃ k, is_ratio_arithmetic_seq a k) :=
sorry

end NUMINAMATH_CALUDE_ratio_arithmetic_properties_l1845_184594


namespace NUMINAMATH_CALUDE_emily_necklaces_l1845_184537

def necklaces (total_beads : ℕ) (beads_per_necklace : ℕ) : ℕ :=
  total_beads / beads_per_necklace

theorem emily_necklaces :
  necklaces 28 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_emily_necklaces_l1845_184537


namespace NUMINAMATH_CALUDE_product_sum_multiple_l1845_184555

theorem product_sum_multiple (a b m : ℤ) : 
  b = 9 → b - a = 5 → a * b = m * (a + b) + 10 → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_multiple_l1845_184555


namespace NUMINAMATH_CALUDE_thirty_factorial_trailing_zeros_l1845_184576

def trailing_zeros (n : ℕ) : ℕ := 
  (n / 5) + (n / 25)

theorem thirty_factorial_trailing_zeros : 
  trailing_zeros 30 = 7 := by
  sorry

end NUMINAMATH_CALUDE_thirty_factorial_trailing_zeros_l1845_184576


namespace NUMINAMATH_CALUDE_u_limit_and_bound_l1845_184535

def u : ℕ → ℚ
  | 0 => 1/3
  | n + 1 => 3 * u n - 3 * (u n)^2

theorem u_limit_and_bound : 
  (∀ k : ℕ, u k = 1/3) ∧ |u 0 - 1/3| ≤ 1/(2^1000) := by sorry

end NUMINAMATH_CALUDE_u_limit_and_bound_l1845_184535


namespace NUMINAMATH_CALUDE_apples_on_tree_l1845_184552

/-- Represents the number of apples in various states -/
structure AppleCount where
  onTree : ℕ
  onGround : ℕ
  eatenByDog : ℕ
  remaining : ℕ

/-- The theorem stating the number of apples on the tree -/
theorem apples_on_tree (a : AppleCount) 
  (h1 : a.onGround = 8)
  (h2 : a.eatenByDog = 3)
  (h3 : a.remaining = 10)
  (h4 : a.onGround = a.remaining + a.eatenByDog) :
  a.onTree = 5 := by
  sorry


end NUMINAMATH_CALUDE_apples_on_tree_l1845_184552


namespace NUMINAMATH_CALUDE_office_persons_count_l1845_184586

theorem office_persons_count :
  ∀ (N : ℕ) (avg_age : ℚ) (avg_age_5 : ℚ) (avg_age_9 : ℚ) (age_15th : ℕ),
  avg_age = 15 →
  avg_age_5 = 14 →
  avg_age_9 = 16 →
  age_15th = 26 →
  N * avg_age = 5 * avg_age_5 + 9 * avg_age_9 + age_15th →
  N = 16 := by
sorry

end NUMINAMATH_CALUDE_office_persons_count_l1845_184586


namespace NUMINAMATH_CALUDE_last_four_digits_of_5_pow_2017_l1845_184528

/-- The last four digits of a natural number -/
def lastFourDigits (n : ℕ) : ℕ := n % 10000

/-- The function that raises 5 to a power and takes the last four digits -/
def f (n : ℕ) : ℕ := lastFourDigits (5^n)

theorem last_four_digits_of_5_pow_2017 :
  f 5 = 3125 → f 6 = 5625 → f 7 = 8125 → f 2017 = 3125 := by
  sorry

end NUMINAMATH_CALUDE_last_four_digits_of_5_pow_2017_l1845_184528


namespace NUMINAMATH_CALUDE_lake_crossing_cost_l1845_184590

/-- The cost of crossing a lake back and forth -/
theorem lake_crossing_cost (crossing_time : ℕ) (assistant_cost : ℕ) : 
  crossing_time = 4 → assistant_cost = 10 → crossing_time * 2 * assistant_cost = 80 := by
  sorry

#check lake_crossing_cost

end NUMINAMATH_CALUDE_lake_crossing_cost_l1845_184590


namespace NUMINAMATH_CALUDE_chess_proficiency_multiple_chess_proficiency_multiple_proof_l1845_184543

theorem chess_proficiency_multiple : ℕ → Prop :=
  fun x =>
    let time_learn_rules : ℕ := 2
    let time_get_proficient : ℕ := time_learn_rules * x
    let time_become_master : ℕ := 100 * (time_learn_rules + time_get_proficient)
    let total_time : ℕ := 10100
    total_time = time_learn_rules + time_get_proficient + time_become_master →
    x = 49

theorem chess_proficiency_multiple_proof : chess_proficiency_multiple 49 := by
  sorry

end NUMINAMATH_CALUDE_chess_proficiency_multiple_chess_proficiency_multiple_proof_l1845_184543


namespace NUMINAMATH_CALUDE_power_function_inequality_l1845_184596

/-- A power function that passes through the point (2,√2) -/
def f (x : ℝ) : ℝ := x^(1/2)

/-- Theorem stating the inequality for any two points on the graph of f -/
theorem power_function_inequality (x₁ x₂ : ℝ) (h1 : 0 < x₁) (h2 : x₁ < x₂) :
  x₂ * f x₁ > x₁ * f x₂ := by
  sorry

end NUMINAMATH_CALUDE_power_function_inequality_l1845_184596


namespace NUMINAMATH_CALUDE_roger_actual_earnings_l1845_184567

/-- Calculates Roger's earnings from mowing lawns --/
def roger_earnings (small_price medium_price large_price : ℕ)
                   (total_small total_medium total_large : ℕ)
                   (forgot_small forgot_medium forgot_large : ℕ) : ℕ :=
  (small_price * (total_small - forgot_small)) +
  (medium_price * (total_medium - forgot_medium)) +
  (large_price * (total_large - forgot_large))

/-- Theorem: Roger's actual earnings are $69 --/
theorem roger_actual_earnings :
  roger_earnings 9 12 15 5 4 5 2 3 3 = 69 := by
  sorry

end NUMINAMATH_CALUDE_roger_actual_earnings_l1845_184567


namespace NUMINAMATH_CALUDE_sequence_a_correct_l1845_184557

def sequence_a (n : ℕ) : ℚ :=
  if n = 1 then 1
  else 1 / (2 * n - 1 : ℚ) - 1 / (2 * n - 3 : ℚ)

def sum_S (n : ℕ) : ℚ :=
  if n = 1 then 1
  else 1 / (2 * n - 1 : ℚ)

theorem sequence_a_correct :
  ∀ n : ℕ, n ≥ 1 →
    (n = 1 ∧ sequence_a n = 1) ∨
    (n ≥ 2 ∧ (sum_S n)^2 = sequence_a n * (sum_S n - 1/2)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_a_correct_l1845_184557


namespace NUMINAMATH_CALUDE_strawberry_harvest_l1845_184545

/-- Calculates the total number of strawberries harvested in a rectangular garden --/
theorem strawberry_harvest (length width : ℕ) (plants_per_sqft : ℕ) (strawberries_per_plant : ℕ) :
  length = 10 → width = 12 → plants_per_sqft = 5 → strawberries_per_plant = 8 →
  length * width * plants_per_sqft * strawberries_per_plant = 4800 := by
  sorry

#check strawberry_harvest

end NUMINAMATH_CALUDE_strawberry_harvest_l1845_184545


namespace NUMINAMATH_CALUDE_function_inequality_l1845_184504

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, deriv f x > f x) : f 2 > Real.exp 2 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1845_184504


namespace NUMINAMATH_CALUDE_remainder_negation_l1845_184536

theorem remainder_negation (a : ℤ) : 
  (a % 1999 = 1) → ((-a) % 1999 = 1998) := by
  sorry

end NUMINAMATH_CALUDE_remainder_negation_l1845_184536


namespace NUMINAMATH_CALUDE_pond_width_l1845_184554

/-- The width of a rectangular pond, given its length, depth, and volume -/
theorem pond_width (length : ℝ) (depth : ℝ) (volume : ℝ) : 
  length = 28 → depth = 5 → volume = 1400 → volume = length * depth * 10 := by
  sorry

end NUMINAMATH_CALUDE_pond_width_l1845_184554


namespace NUMINAMATH_CALUDE_close_interval_is_zero_one_l1845_184598

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 + x + 2
def g (x : ℝ) : ℝ := 2*x + 1

-- Define what it means for two functions to be "close"
def are_close (f g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → |f x - g x| ≤ 1

-- State the theorem
theorem close_interval_is_zero_one :
  are_close f g 0 1 ∧
  ∀ a b, a < 0 ∨ b > 1 → ¬(are_close f g a b) :=
sorry

end NUMINAMATH_CALUDE_close_interval_is_zero_one_l1845_184598


namespace NUMINAMATH_CALUDE_min_k_for_triangle_inequality_l1845_184514

theorem min_k_for_triangle_inequality : 
  ∃ (k : ℕ), k > 0 ∧ 
  (∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → 
    k * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) → 
    (a + b > c ∧ b + c > a ∧ c + a > b)) ∧
  (∀ (k' : ℕ), k' > 0 → k' < k → 
    ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    k' * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) ∧
    ¬(a + b > c ∧ b + c > a ∧ c + a > b)) ∧
  k = 6 :=
sorry

end NUMINAMATH_CALUDE_min_k_for_triangle_inequality_l1845_184514


namespace NUMINAMATH_CALUDE_inscribed_angles_sum_l1845_184568

/-- Given a circle divided into 18 equal arcs, if central angle x spans 3 arcs
    and central angle y spans 6 arcs, then the sum of the corresponding
    inscribed angles x and y is 90°. -/
theorem inscribed_angles_sum (x y : ℝ) : 
  (18 : ℝ) * x = 360 →  -- The circle is divided into 18 equal arcs
  3 * x = y →           -- Central angle y is twice central angle x
  2 * x = 60 →          -- Central angle x spans 3 arcs (3 * 20° = 60°)
  x / 2 + y / 2 = 90    -- Sum of inscribed angles x and y is 90°
  := by sorry

end NUMINAMATH_CALUDE_inscribed_angles_sum_l1845_184568


namespace NUMINAMATH_CALUDE_closest_to_target_l1845_184583

def numbers : List ℤ := [100260, 99830, 98900, 100320]

def target : ℤ := 100000

theorem closest_to_target :
  ∀ n ∈ numbers, |99830 - target| ≤ |n - target| :=
sorry

end NUMINAMATH_CALUDE_closest_to_target_l1845_184583


namespace NUMINAMATH_CALUDE_divisors_of_n_squared_less_than_n_not_dividing_n_l1845_184565

def n : ℕ := 2^31 * 3^19 * 5^7

-- Function to count divisors of a number given its prime factorization
def count_divisors (factorization : List (ℕ × ℕ)) : ℕ :=
  factorization.foldl (λ acc (_, exp) => acc * (exp + 1)) 1

-- Function to count divisors less than n
def count_divisors_less_than_n (total_divisors : ℕ) : ℕ :=
  (total_divisors - 1) / 2

theorem divisors_of_n_squared_less_than_n_not_dividing_n :
  let n_squared_factorization : List (ℕ × ℕ) := [(2, 62), (3, 38), (5, 14)]
  let n_factorization : List (ℕ × ℕ) := [(2, 31), (3, 19), (5, 7)]
  let total_divisors_n_squared := count_divisors n_squared_factorization
  let divisors_less_than_n := count_divisors_less_than_n total_divisors_n_squared
  let divisors_of_n := count_divisors n_factorization
  divisors_less_than_n - divisors_of_n = 13307 :=
by sorry

end NUMINAMATH_CALUDE_divisors_of_n_squared_less_than_n_not_dividing_n_l1845_184565


namespace NUMINAMATH_CALUDE_parallel_condition_l1845_184570

-- Define the lines l1 and l2
def l1 (a x y : ℝ) : Prop := a * x + y = 1
def l2 (a x y : ℝ) : Prop := 9 * x + a * y = 1

-- Define parallel lines
def parallel (a : ℝ) : Prop := ∀ x y, l1 a x y ↔ l2 a x y

-- Theorem statement
theorem parallel_condition (a : ℝ) :
  (a + 3 = 0 → parallel a) ∧ ¬(parallel a → a + 3 = 0) :=
sorry

end NUMINAMATH_CALUDE_parallel_condition_l1845_184570


namespace NUMINAMATH_CALUDE_number_of_players_is_five_l1845_184573

/-- Represents the number of chips each player receives -/
def chips_per_player (m : ℕ) (n : ℕ) : ℕ := n * m

/-- Represents the number of chips taken by the i-th player -/
def chips_taken (i : ℕ) (m : ℕ) (remaining : ℕ) : ℕ :=
  i * m + remaining / 6

/-- The main theorem stating that the number of players is 5 -/
theorem number_of_players_is_five (m : ℕ) (total_chips : ℕ) :
  ∃ (n : ℕ),
    n = 5 ∧
    (∀ i : ℕ, i ≤ n →
      chips_taken i m (total_chips - (chips_per_player m i)) =
      chips_per_player m n) :=
sorry

end NUMINAMATH_CALUDE_number_of_players_is_five_l1845_184573


namespace NUMINAMATH_CALUDE_optimal_coin_strategy_l1845_184551

/-- Probability of winning with n turns left and difference k between heads and tails -/
noncomputable def q (n : ℕ) (k : ℕ) : ℝ :=
  sorry

/-- The optimal strategy theorem -/
theorem optimal_coin_strategy (n : ℕ) (k : ℕ) : q n k ≥ q n (k + 2) := by
  sorry

end NUMINAMATH_CALUDE_optimal_coin_strategy_l1845_184551
