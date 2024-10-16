import Mathlib

namespace NUMINAMATH_CALUDE_onion_harvest_weight_l827_82795

-- Define the number of bags per trip
def bags_per_trip : ℕ := 10

-- Define the weight of each bag in kg
def weight_per_bag : ℕ := 50

-- Define the number of trips
def number_of_trips : ℕ := 20

-- Define the total weight of onions harvested
def total_weight : ℕ := bags_per_trip * weight_per_bag * number_of_trips

-- Theorem statement
theorem onion_harvest_weight :
  total_weight = 10000 := by sorry

end NUMINAMATH_CALUDE_onion_harvest_weight_l827_82795


namespace NUMINAMATH_CALUDE_horseshoe_cost_per_set_l827_82741

/-- Proves that the cost per set of horseshoes is $20.75 given the initial outlay,
    selling price, number of sets sold, and profit. -/
theorem horseshoe_cost_per_set 
  (initial_outlay : ℝ)
  (selling_price : ℝ)
  (sets_sold : ℕ)
  (profit : ℝ)
  (h1 : initial_outlay = 12450)
  (h2 : selling_price = 50)
  (h3 : sets_sold = 950)
  (h4 : profit = 15337.5)
  (h5 : profit = selling_price * sets_sold - (initial_outlay + cost_per_set * sets_sold)) :
  cost_per_set = 20.75 :=
by
  sorry

#check horseshoe_cost_per_set

end NUMINAMATH_CALUDE_horseshoe_cost_per_set_l827_82741


namespace NUMINAMATH_CALUDE_ellipse_vector_dot_product_range_l827_82707

/-- The ellipse equation -/
def on_ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Point M -/
def M : ℝ × ℝ := (1, 0)

/-- Vector dot product -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- Vector from M to a point -/
def vector_MA (A : ℝ × ℝ) : ℝ × ℝ :=
  (A.1 - M.1, A.2 - M.2)

theorem ellipse_vector_dot_product_range :
  ∀ A B : ℝ × ℝ,
  on_ellipse A.1 A.2 →
  on_ellipse B.1 B.2 →
  dot_product (vector_MA A) (vector_MA B) = 0 →
  ∃ x : ℝ, x = dot_product (vector_MA A) (A.1 - B.1, A.2 - B.2) ∧
           2/3 ≤ x ∧ x ≤ 9 :=
sorry

end NUMINAMATH_CALUDE_ellipse_vector_dot_product_range_l827_82707


namespace NUMINAMATH_CALUDE_octal_subtraction_example_l827_82742

/-- Represents a number in base 8 --/
def OctalNumber := ℕ

/-- Converts a natural number to its octal representation --/
def toOctal (n : ℕ) : OctalNumber :=
  sorry

/-- Performs subtraction in base 8 --/
def octalSubtract (a b : OctalNumber) : OctalNumber :=
  sorry

/-- Theorem: 325₈ - 237₈ = 66₈ in base 8 arithmetic --/
theorem octal_subtraction_example : octalSubtract (toOctal 325) (toOctal 237) = toOctal 66 := by
  sorry

end NUMINAMATH_CALUDE_octal_subtraction_example_l827_82742


namespace NUMINAMATH_CALUDE_equation_solution_l827_82763

theorem equation_solution (x : ℝ) : (x^2 - 1) / (x + 1) = 0 ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l827_82763


namespace NUMINAMATH_CALUDE_rudy_typing_speed_l827_82752

def team_size : ℕ := 5
def team_average : ℕ := 80
def joyce_speed : ℕ := 76
def gladys_speed : ℕ := 91
def lisa_speed : ℕ := 80
def mike_speed : ℕ := 89

theorem rudy_typing_speed :
  ∃ (rudy_speed : ℕ),
    rudy_speed = team_size * team_average - (joyce_speed + gladys_speed + lisa_speed + mike_speed) :=
by sorry

end NUMINAMATH_CALUDE_rudy_typing_speed_l827_82752


namespace NUMINAMATH_CALUDE_total_dogs_in_kennel_l827_82758

-- Define the sets and their sizes
def T : ℕ := 45  -- Number of dogs with tags
def C : ℕ := 40  -- Number of dogs with collars
def B : ℕ := 6   -- Number of dogs with both tags and collars
def N : ℕ := 1   -- Number of dogs with neither tags nor collars

-- Theorem statement
theorem total_dogs_in_kennel : T + C - B + N = 80 := by
  sorry

end NUMINAMATH_CALUDE_total_dogs_in_kennel_l827_82758


namespace NUMINAMATH_CALUDE_three_lines_theorem_l827_82790

/-- A line in 3D space -/
structure Line3D where
  -- Define a line using two points
  point1 : ℝ × ℝ × ℝ
  point2 : ℝ × ℝ × ℝ
  ne_points : point1 ≠ point2

/-- Three lines in 3D space -/
structure ThreeLines where
  line1 : Line3D
  line2 : Line3D
  line3 : Line3D

/-- Predicate to check if three lines are coplanar -/
def are_coplanar (lines : ThreeLines) : Prop :=
  sorry

/-- Predicate to check if two lines are skew -/
def are_skew (l1 l2 : Line3D) : Prop :=
  sorry

/-- Predicate to check if three lines intersect at a single point -/
def intersect_at_point (lines : ThreeLines) : Prop :=
  sorry

/-- Predicate to check if three lines are parallel -/
def are_parallel (lines : ThreeLines) : Prop :=
  sorry

/-- Theorem stating that three non-coplanar lines with no two being skew
    either intersect at a single point or are parallel -/
theorem three_lines_theorem (lines : ThreeLines) 
  (h1 : ¬ are_coplanar lines)
  (h2 : ¬ are_skew lines.line1 lines.line2)
  (h3 : ¬ are_skew lines.line1 lines.line3)
  (h4 : ¬ are_skew lines.line2 lines.line3) :
  intersect_at_point lines ∨ are_parallel lines :=
sorry

end NUMINAMATH_CALUDE_three_lines_theorem_l827_82790


namespace NUMINAMATH_CALUDE_intersection_A_B_l827_82751

-- Define set A
def A : Set ℝ := {x | x^2 - 3*x < 0}

-- Define set B
def B : Set ℝ := {x | x > 2}

-- Theorem statement
theorem intersection_A_B : A ∩ B = Set.Ioo 2 3 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l827_82751


namespace NUMINAMATH_CALUDE_count_four_digit_with_five_thousands_l827_82733

/-- A four-digit positive integer with thousands digit 5 -/
def FourDigitWithFiveThousands : Type := { n : ℕ // 5000 ≤ n ∧ n ≤ 5999 }

/-- The count of four-digit positive integers with thousands digit 5 -/
def CountFourDigitWithFiveThousands : ℕ := Finset.card (Finset.filter (λ n => 5000 ≤ n ∧ n ≤ 5999) (Finset.range 10000))

theorem count_four_digit_with_five_thousands :
  CountFourDigitWithFiveThousands = 1000 := by
  sorry

end NUMINAMATH_CALUDE_count_four_digit_with_five_thousands_l827_82733


namespace NUMINAMATH_CALUDE_common_root_of_three_equations_l827_82720

theorem common_root_of_three_equations (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h_ab : ∃ x : ℝ, a * x^11 + b * x^4 + c = 0 ∧ b * x^11 + c * x^4 + a = 0)
  (h_bc : ∃ x : ℝ, b * x^11 + c * x^4 + a = 0 ∧ c * x^11 + a * x^4 + b = 0)
  (h_ca : ∃ x : ℝ, c * x^11 + a * x^4 + b = 0 ∧ a * x^11 + b * x^4 + c = 0) :
  a * 1^11 + b * 1^4 + c = 0 ∧ b * 1^11 + c * 1^4 + a = 0 ∧ c * 1^11 + a * 1^4 + b = 0 :=
by sorry

end NUMINAMATH_CALUDE_common_root_of_three_equations_l827_82720


namespace NUMINAMATH_CALUDE_jury_deliberation_theorem_l827_82766

/-- Calculates the equivalent full days spent in jury deliberation --/
def jury_deliberation_days (total_days : ℕ) (selection_days : ℕ) (trial_multiplier : ℕ) 
  (deliberation_hours_per_day : ℕ) (hours_per_day : ℕ) : ℕ :=
  let trial_days := selection_days * trial_multiplier
  let deliberation_days := total_days - selection_days - trial_days
  let total_deliberation_hours := deliberation_days * deliberation_hours_per_day
  total_deliberation_hours / hours_per_day

theorem jury_deliberation_theorem :
  jury_deliberation_days 19 2 4 16 24 = 6 := by
  sorry

end NUMINAMATH_CALUDE_jury_deliberation_theorem_l827_82766


namespace NUMINAMATH_CALUDE_cos_pi_sixth_plus_alpha_l827_82717

theorem cos_pi_sixth_plus_alpha (α : Real) 
  (h : Real.sin (α - π/3) = 1/3) : 
  Real.cos (π/6 + α) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_sixth_plus_alpha_l827_82717


namespace NUMINAMATH_CALUDE_complex_magnitude_l827_82715

theorem complex_magnitude (z : ℂ) : z + 2 * Complex.I = (3 - Complex.I ^ 3) / (1 + Complex.I) → Complex.abs z = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l827_82715


namespace NUMINAMATH_CALUDE_mark_soup_donation_l827_82709

/-- The number of homeless shelters Mark donates to -/
def num_shelters : ℕ := 6

/-- The number of people serviced by each shelter -/
def people_per_shelter : ℕ := 30

/-- The number of cans of soup Mark buys per person -/
def cans_per_person : ℕ := 10

/-- The total number of cans of soup Mark donates -/
def total_cans : ℕ := num_shelters * people_per_shelter * cans_per_person

theorem mark_soup_donation : total_cans = 1800 := by
  sorry

end NUMINAMATH_CALUDE_mark_soup_donation_l827_82709


namespace NUMINAMATH_CALUDE_inequality_theorem_l827_82700

theorem inequality_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x * (y + z - x)^2 + y * (z + x - y)^2 + z * (x + y - z)^2 ≥ 
  2 * x * y * z * (x / (y + z) + y / (z + x) + z / (x + y)) ∧
  (x * (y + z - x)^2 + y * (z + x - y)^2 + z * (x + y - z)^2 = 
   2 * x * y * z * (x / (y + z) + y / (z + x) + z / (x + y)) ↔ x = y ∧ y = z) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l827_82700


namespace NUMINAMATH_CALUDE_four_solutions_to_g_composition_l827_82740

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 4*x

-- State the theorem
theorem four_solutions_to_g_composition :
  ∃! (s : Finset ℝ), (∀ c ∈ s, g (g (g (g c))) = 5) ∧ s.card = 4 :=
sorry

end NUMINAMATH_CALUDE_four_solutions_to_g_composition_l827_82740


namespace NUMINAMATH_CALUDE_square_diff_sum_eq_three_l827_82781

theorem square_diff_sum_eq_three (a b c : ℤ) 
  (ha : a = 2011) (hb : b = 2012) (hc : c = 2013) : 
  a^2 + b^2 + c^2 - a*b - b*c - a*c = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_sum_eq_three_l827_82781


namespace NUMINAMATH_CALUDE_line_segment_ratios_l827_82750

/-- Given points A, B, C on a straight line with AC : BC = m : n,
    prove the ratios AC : AB and BC : AB -/
theorem line_segment_ratios
  (A B C : ℝ) -- Points on a real line
  (m n : ℕ) -- Natural numbers for the ratio
  (h_line : (A ≤ B ∧ B ≤ C) ∨ (A ≤ C ∧ C ≤ B) ∨ (B ≤ A ∧ A ≤ C)) -- Points are on a line
  (h_ratio : |C - A| / |C - B| = m / n) : -- Given ratio
  (∃ (r₁ r₂ : ℚ),
    (r₁ = m / (m + n) ∧ r₂ = n / (m + n)) ∨
    (r₁ = m / (n - m) ∧ r₂ = n / (n - m)) ∨
    (m = n ∧ r₁ = 1 / 2 ∧ r₂ = 1 / 2)) ∧
  (|A - C| / |A - B| = r₁ ∧ |B - C| / |A - B| = r₂) :=
sorry

end NUMINAMATH_CALUDE_line_segment_ratios_l827_82750


namespace NUMINAMATH_CALUDE_solution_sets_equality_l827_82739

theorem solution_sets_equality (a b : ℝ) : 
  (∀ x : ℝ, |8*x + 9| < 7 ↔ a*x^2 + b*x > 2) → 
  (a = -4 ∧ b = -9) := by
sorry

end NUMINAMATH_CALUDE_solution_sets_equality_l827_82739


namespace NUMINAMATH_CALUDE_division_into_proportional_parts_l827_82719

theorem division_into_proportional_parts (total : ℚ) (a b c : ℚ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  total = 104 →
  a = 1 →
  b = 1/2 →
  c = 1/3 →
  let x := total * b / (a + b + c)
  x = 28 + 4/11 := by
  sorry

end NUMINAMATH_CALUDE_division_into_proportional_parts_l827_82719


namespace NUMINAMATH_CALUDE_daves_painted_area_l827_82769

theorem daves_painted_area 
  (total_area : ℝ) 
  (cathy_ratio : ℝ) 
  (dave_ratio : ℝ) 
  (h1 : total_area = 330) 
  (h2 : cathy_ratio = 4) 
  (h3 : dave_ratio = 7) : 
  dave_ratio / (cathy_ratio + dave_ratio) * total_area = 210 := by
sorry

end NUMINAMATH_CALUDE_daves_painted_area_l827_82769


namespace NUMINAMATH_CALUDE_number_square_equation_l827_82799

theorem number_square_equation : ∃ x : ℝ, x^2 + 100 = (x - 20)^2 ∧ x = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_number_square_equation_l827_82799


namespace NUMINAMATH_CALUDE_smallest_AAB_l827_82743

/-- Represents a two-digit number --/
def TwoDigitNumber (a b : Nat) : Prop :=
  a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9

/-- Represents a three-digit number --/
def ThreeDigitNumber (a b : Nat) : Prop :=
  a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9

/-- The value of a two-digit number AB --/
def ValueAB (a b : Nat) : Nat :=
  10 * a + b

/-- The value of a three-digit number AAB --/
def ValueAAB (a b : Nat) : Nat :=
  100 * a + 10 * a + b

theorem smallest_AAB :
  ∀ a b : Nat,
    TwoDigitNumber a b →
    ThreeDigitNumber a b →
    a ≠ b →
    8 * (ValueAB a b) = ValueAAB a b →
    ∀ x y : Nat,
      TwoDigitNumber x y →
      ThreeDigitNumber x y →
      x ≠ y →
      8 * (ValueAB x y) = ValueAAB x y →
      ValueAAB a b ≤ ValueAAB x y →
    ValueAAB a b = 224 :=
by sorry

end NUMINAMATH_CALUDE_smallest_AAB_l827_82743


namespace NUMINAMATH_CALUDE_V_upper_bound_l827_82784

/-- V(n; b) is the number of decompositions of n into a product of one or more positive integers greater than b -/
def V (n b : ℕ+) : ℕ := sorry

/-- For all positive integers n and b, V(n; b) < n/b -/
theorem V_upper_bound (n b : ℕ+) : V n b < (n : ℚ) / b := by sorry

end NUMINAMATH_CALUDE_V_upper_bound_l827_82784


namespace NUMINAMATH_CALUDE_parallelepiped_volume_l827_82747

/-- Given a rectangular parallelepiped with dimensions l, w, and h, if the shortest distances
    from an interior diagonal to the edges it does not meet are 2√5, 30/√13, and 15/√10,
    then the volume of the parallelepiped is 750. -/
theorem parallelepiped_volume (l w h : ℝ) (hl : l > 0) (hw : w > 0) (hh : h > 0) : 
  (l * w / Real.sqrt (l^2 + w^2) = 2 * Real.sqrt 5) →
  (h * w / Real.sqrt (h^2 + w^2) = 30 / Real.sqrt 13) →
  (h * l / Real.sqrt (h^2 + l^2) = 15 / Real.sqrt 10) →
  l * w * h = 750 := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_volume_l827_82747


namespace NUMINAMATH_CALUDE_hyperbola_curve_is_hyperbola_l827_82775

/-- A curve defined by x = cos^2 u and y = sin^4 u for real u -/
def HyperbolaCurve : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ u : ℝ, p.1 = Real.cos u ^ 2 ∧ p.2 = Real.sin u ^ 4}

/-- The curve defined by HyperbolaCurve is a hyperbola -/
theorem hyperbola_curve_is_hyperbola : 
  ∃ a b c d e f : ℝ, a ≠ 0 ∧ (a * b > 0 ∨ a * b < 0) ∧
  ∀ p : ℝ × ℝ, p ∈ HyperbolaCurve ↔ 
    a * p.1^2 + b * p.2^2 + c * p.1 * p.2 + d * p.1 + e * p.2 + f = 0 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_curve_is_hyperbola_l827_82775


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l827_82723

theorem inequality_system_solution_set :
  let S : Set ℝ := {x | x > -6 - 2*x ∧ x ≤ (3 + x) / 4}
  S = {x | -2 < x ∧ x ≤ 1} := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l827_82723


namespace NUMINAMATH_CALUDE_cube_of_102_l827_82765

theorem cube_of_102 : (100 + 2)^3 = 1061208 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_102_l827_82765


namespace NUMINAMATH_CALUDE_cubic_fraction_factorization_l827_82789

theorem cubic_fraction_factorization (a b c : ℝ) :
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3) = 
  (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by sorry

end NUMINAMATH_CALUDE_cubic_fraction_factorization_l827_82789


namespace NUMINAMATH_CALUDE_lukes_remaining_money_l827_82710

def octal_to_decimal (n : ℕ) : ℕ := 
  (n / 1000) * 512 + ((n / 100) % 10) * 64 + ((n / 10) % 10) * 8 + (n % 10)

theorem lukes_remaining_money :
  let savings := octal_to_decimal 5555
  let ticket_cost := 1200
  savings - ticket_cost = 1725 := by sorry

end NUMINAMATH_CALUDE_lukes_remaining_money_l827_82710


namespace NUMINAMATH_CALUDE_extra_domino_possible_l827_82792

/-- Represents a 6x6 chessboard -/
def Chessboard := Fin 6 → Fin 6 → Bool

/-- A domino is a pair of adjacent squares on the chessboard -/
def Domino := (Fin 6 × Fin 6) × (Fin 6 × Fin 6)

/-- Checks if two squares are adjacent -/
def adjacent (s1 s2 : Fin 6 × Fin 6) : Prop :=
  (s1.1 = s2.1 ∧ s1.2.succ = s2.2) ∨
  (s1.1 = s2.1 ∧ s1.2 = s2.2.succ) ∨
  (s1.1.succ = s2.1 ∧ s1.2 = s2.2) ∨
  (s1.1 = s2.1.succ ∧ s1.2 = s2.2)

/-- Checks if a domino is valid (covers two adjacent squares) -/
def validDomino (d : Domino) : Prop :=
  adjacent d.1 d.2

/-- Checks if two dominoes overlap -/
def overlap (d1 d2 : Domino) : Prop :=
  d1.1 = d2.1 ∨ d1.1 = d2.2 ∨ d1.2 = d2.1 ∨ d1.2 = d2.2

/-- Represents a configuration of 11 dominoes on the chessboard -/
def Configuration := Fin 11 → Domino

/-- Checks if a configuration is valid (no overlaps) -/
def validConfiguration (config : Configuration) : Prop :=
  ∀ i j : Fin 11, i ≠ j → ¬(overlap (config i) (config j))

/-- Theorem: Given a valid configuration of 11 dominoes on a 6x6 chessboard,
    there always exists at least two adjacent empty squares -/
theorem extra_domino_possible (config : Configuration) 
  (h_valid : validConfiguration config) :
  ∃ s1 s2 : Fin 6 × Fin 6, adjacent s1 s2 ∧
    (∀ i : Fin 11, s1 ≠ (config i).1 ∧ s1 ≠ (config i).2 ∧
                   s2 ≠ (config i).1 ∧ s2 ≠ (config i).2) :=
  sorry


end NUMINAMATH_CALUDE_extra_domino_possible_l827_82792


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l827_82724

theorem stratified_sampling_theorem :
  let total_employees : ℕ := 150
  let senior_titles : ℕ := 15
  let intermediate_titles : ℕ := 45
  let junior_titles : ℕ := 90
  let sample_size : ℕ := 30
  
  senior_titles + intermediate_titles + junior_titles = total_employees →
  
  let senior_sample := sample_size * senior_titles / total_employees
  let intermediate_sample := sample_size * intermediate_titles / total_employees
  let junior_sample := sample_size * junior_titles / total_employees
  
  (senior_sample = 3 ∧ intermediate_sample = 9 ∧ junior_sample = 18) :=
by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l827_82724


namespace NUMINAMATH_CALUDE_cosine_sine_inequality_l827_82745

theorem cosine_sine_inequality (x : Real) (h : 0 < x ∧ x < π/4) :
  (Real.cos x) ^ (Real.cos x)^2 > (Real.sin x) ^ (Real.sin x)^2 ∧
  (Real.cos x) ^ (Real.cos x)^4 < (Real.sin x) ^ (Real.sin x)^4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_inequality_l827_82745


namespace NUMINAMATH_CALUDE_tom_age_l827_82704

theorem tom_age (adam_age : ℕ) (future_years : ℕ) (future_combined_age : ℕ) :
  adam_age = 8 →
  future_years = 12 →
  future_combined_age = 44 →
  ∃ tom_age : ℕ, tom_age + adam_age + 2 * future_years = future_combined_age ∧ tom_age = 12 :=
by sorry

end NUMINAMATH_CALUDE_tom_age_l827_82704


namespace NUMINAMATH_CALUDE_translation_theorem_l827_82737

/-- The original function -/
def f (x : ℝ) : ℝ := -(x - 1)^2 + 3

/-- The target function -/
def g (x : ℝ) : ℝ := -x^2

/-- The translation function -/
def translate (x : ℝ) : ℝ := x + 1

theorem translation_theorem :
  ∀ x : ℝ, f (translate x) - 3 = g x := by sorry

end NUMINAMATH_CALUDE_translation_theorem_l827_82737


namespace NUMINAMATH_CALUDE_triangle_perimeter_l827_82796

/-- A triangle with specific area and angles has a specific perimeter -/
theorem triangle_perimeter (A B C : ℝ) (h_area : A = 3 - Real.sqrt 3)
    (h_angle1 : B = 45 * π / 180) (h_angle2 : C = 60 * π / 180) (h_angle3 : A = 75 * π / 180) :
  let perimeter := Real.sqrt 2 * (3 + 2 * Real.sqrt 3 - Real.sqrt 6)
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = perimeter ∧
    (1/2) * a * b * Real.sin C = 3 - Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l827_82796


namespace NUMINAMATH_CALUDE_same_solution_implies_k_equals_one_l827_82770

theorem same_solution_implies_k_equals_one :
  (∃ x : ℝ, x - 2 = 0 ∧ 1 - (x + k) / 3 = 0) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_implies_k_equals_one_l827_82770


namespace NUMINAMATH_CALUDE_triangle_inequality_l827_82714

theorem triangle_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c)
  (hab : a + b ≥ c) (hbc : b + c ≥ a) (hca : c + a ≥ b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l827_82714


namespace NUMINAMATH_CALUDE_oranges_from_second_tree_l827_82783

theorem oranges_from_second_tree :
  ∀ (first_tree second_tree third_tree total : ℕ),
  first_tree = 80 →
  third_tree = 120 →
  total = 260 →
  total = first_tree + second_tree + third_tree →
  second_tree = 60 := by
sorry

end NUMINAMATH_CALUDE_oranges_from_second_tree_l827_82783


namespace NUMINAMATH_CALUDE_dans_helmet_craters_l827_82718

theorem dans_helmet_craters (dans_craters daniel_craters rins_craters : ℕ) : 
  dans_craters = daniel_craters + 10 →
  rins_craters = dans_craters + daniel_craters + 15 →
  rins_craters = 75 →
  dans_craters = 35 := by
  sorry

end NUMINAMATH_CALUDE_dans_helmet_craters_l827_82718


namespace NUMINAMATH_CALUDE_f_values_f_inequality_range_l827_82759

noncomputable section

variable (f : ℝ → ℝ)

axiom domain : ∀ x, x > 0 → f x ≠ 0
axiom f_2 : f 2 = 1
axiom f_mult : ∀ x y, x > 0 → y > 0 → f (x * y) = f x + f y
axiom f_increasing : ∀ x₁ x₂, 0 < x₁ → x₁ < x₂ → f x₁ < f x₂

theorem f_values :
  f 1 = 0 ∧ f 4 = 2 ∧ f 8 = 3 :=
sorry

theorem f_inequality_range :
  ∀ x, (f x + f (x - 2) ≤ 3) ↔ (2 < x ∧ x ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_f_values_f_inequality_range_l827_82759


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l827_82736

-- Define the conditions p and q
def p (x : ℝ) : Prop := x - 3 > 0
def q (x : ℝ) : Prop := (x - 3) * (x - 4) < 0

-- State the theorem
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x, ¬(p x) → ¬(q x)) ∧ 
  ∃ x, ¬(q x) ∧ p x :=
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l827_82736


namespace NUMINAMATH_CALUDE_overlapping_squares_area_l827_82732

theorem overlapping_squares_area (side_length : ℝ) (rotation_angle : ℝ) : 
  side_length = 12 →
  rotation_angle = 30 * π / 180 →
  ∃ (common_area : ℝ), common_area = 48 * Real.sqrt 3 ∧
    common_area = 2 * (1/2 * side_length * (side_length / Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_overlapping_squares_area_l827_82732


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l827_82701

theorem partial_fraction_decomposition (a b c : ℤ) 
  (h1 : (1 : ℚ) / 2015 = a / 5 + b / 13 + c / 31)
  (h2 : 0 ≤ a ∧ a < 5)
  (h3 : 0 ≤ b ∧ b < 13) :
  a + b = 14 := by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l827_82701


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l827_82730

-- Define a geometric sequence
def isGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  isGeometricSequence a →
  (a 1 + a 2 = 40) →
  (a 3 + a 4 = 60) →
  (a 7 + a 8 = 135) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l827_82730


namespace NUMINAMATH_CALUDE_two_variable_data_representable_by_scatter_plot_l827_82760

/-- Represents statistical data for two variables -/
structure TwoVariableData where
  -- Define the structure of two-variable data
  -- (We don't need to specify the exact structure for this problem)

/-- Represents a scatter plot -/
structure ScatterPlot where
  -- Define the structure of a scatter plot
  -- (We don't need to specify the exact structure for this problem)

/-- Creates a scatter plot from two-variable data -/
def create_scatter_plot (data : TwoVariableData) : ScatterPlot :=
  sorry -- The actual implementation is not important for this statement

/-- Theorem: Any two-variable statistical data can be represented by a scatter plot -/
theorem two_variable_data_representable_by_scatter_plot (data : TwoVariableData) :
  ∃ (plot : ScatterPlot), plot = create_scatter_plot data :=
sorry

end NUMINAMATH_CALUDE_two_variable_data_representable_by_scatter_plot_l827_82760


namespace NUMINAMATH_CALUDE_concentric_circles_radius_change_l827_82735

theorem concentric_circles_radius_change (R_o R_i : ℝ) 
  (h1 : R_o = 6)
  (h2 : R_i = 4)
  (h3 : R_o > R_i)
  (h4 : 0 < R_i)
  (h5 : 0 < R_o) :
  let A_original := π * (R_o^2 - R_i^2)
  let R_i_new := R_i * 0.75
  let A_new := A_original * 3.6
  ∃ x : ℝ, 
    (π * ((R_o * (1 + x/100))^2 - R_i_new^2) = A_new) ∧
    x = 50 :=
sorry

end NUMINAMATH_CALUDE_concentric_circles_radius_change_l827_82735


namespace NUMINAMATH_CALUDE_chocolate_count_correct_l827_82794

/-- The number of small boxes in the large box -/
def total_small_boxes : ℕ := 17

/-- The number of small boxes containing medium boxes -/
def boxes_with_medium : ℕ := 10

/-- The number of medium boxes in each of the first 10 small boxes -/
def medium_boxes_per_small : ℕ := 4

/-- The number of chocolate bars in each medium box -/
def chocolates_per_medium : ℕ := 26

/-- The number of chocolate bars in each of the first two of the remaining small boxes -/
def chocolates_in_first_two : ℕ := 18

/-- The number of chocolate bars in each of the next three of the remaining small boxes -/
def chocolates_in_next_three : ℕ := 22

/-- The number of chocolate bars in each of the last two of the remaining small boxes -/
def chocolates_in_last_two : ℕ := 30

/-- The total number of chocolate bars in the large box -/
def total_chocolates : ℕ := 1202

theorem chocolate_count_correct : 
  (boxes_with_medium * medium_boxes_per_small * chocolates_per_medium) +
  (2 * chocolates_in_first_two) +
  (3 * chocolates_in_next_three) +
  (2 * chocolates_in_last_two) = total_chocolates :=
by sorry

end NUMINAMATH_CALUDE_chocolate_count_correct_l827_82794


namespace NUMINAMATH_CALUDE_decimal_sum_l827_82706

theorem decimal_sum : 0.5 + 0.035 + 0.0041 = 0.5391 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_l827_82706


namespace NUMINAMATH_CALUDE_family_members_count_l827_82746

/-- Represents the number of family members -/
def n : ℕ := sorry

/-- The average age of family members in years -/
def average_age : ℕ := 29

/-- The present age of the youngest member in years -/
def youngest_age : ℕ := 5

/-- The average age of the remaining members at the time of birth of the youngest member in years -/
def average_age_at_birth : ℕ := 28

/-- The sum of ages of all family members -/
def sum_of_ages : ℕ := n * average_age

/-- The sum of ages of the remaining members at present -/
def sum_of_remaining_ages : ℕ := (n - 1) * (average_age_at_birth + youngest_age)

theorem family_members_count :
  sum_of_ages = sum_of_remaining_ages + youngest_age → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_family_members_count_l827_82746


namespace NUMINAMATH_CALUDE_charles_discount_l827_82757

/-- The discount given to a customer, given the total cost before discount and the amount paid after discount. -/
def discount (total_cost : ℝ) (amount_paid : ℝ) : ℝ :=
  total_cost - amount_paid

/-- Theorem: The discount given to Charles is $2. -/
theorem charles_discount : discount 45 43 = 2 := by
  sorry

end NUMINAMATH_CALUDE_charles_discount_l827_82757


namespace NUMINAMATH_CALUDE_vector_sum_theorem_l827_82703

def vector_a : ℝ × ℝ × ℝ := (2, -3, 4)
def vector_b : ℝ × ℝ × ℝ := (-5, 1, 6)
def vector_c : ℝ × ℝ × ℝ := (3, 0, -2)

theorem vector_sum_theorem :
  vector_a.1 + vector_b.1 + vector_c.1 = 0 ∧
  vector_a.2.1 + vector_b.2.1 + vector_c.2.1 = -2 ∧
  vector_a.2.2 + vector_b.2.2 + vector_c.2.2 = 8 :=
by sorry

end NUMINAMATH_CALUDE_vector_sum_theorem_l827_82703


namespace NUMINAMATH_CALUDE_fourth_power_sum_l827_82729

theorem fourth_power_sum (x y z : ℝ) 
  (h1 : x + y + z = 3)
  (h2 : x^2 + y^2 + z^2 = 5)
  (h3 : x^3 + y^3 + z^3 = 7) :
  x^4 + y^4 + z^4 = 59/3 := by
sorry

end NUMINAMATH_CALUDE_fourth_power_sum_l827_82729


namespace NUMINAMATH_CALUDE_negative_sqrt_three_squared_equals_negative_three_l827_82779

theorem negative_sqrt_three_squared_equals_negative_three :
  -Real.sqrt (3^2) = -3 := by
  sorry

end NUMINAMATH_CALUDE_negative_sqrt_three_squared_equals_negative_three_l827_82779


namespace NUMINAMATH_CALUDE_calculation_proof_l827_82773

theorem calculation_proof : (π - 2019)^0 + |Real.sqrt 3 - 1| + (-1/2)⁻¹ - 2 * Real.tan (30 * π / 180) = -2 + Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l827_82773


namespace NUMINAMATH_CALUDE_line_passes_first_and_fourth_quadrants_l827_82798

/-- A line passes through the first quadrant if there exists a point (x, y) on the line where both x and y are positive. -/
def passes_through_first_quadrant (k b : ℝ) : Prop :=
  ∃ x > 0, k * x + b > 0

/-- A line passes through the fourth quadrant if there exists a point (x, y) on the line where x is positive and y is negative. -/
def passes_through_fourth_quadrant (k b : ℝ) : Prop :=
  ∃ x > 0, k * x + b < 0

/-- If bk < 0, then the line y = kx + b passes through both the first and fourth quadrants. -/
theorem line_passes_first_and_fourth_quadrants (k b : ℝ) (h : b * k < 0) :
  passes_through_first_quadrant k b ∧ passes_through_fourth_quadrant k b :=
sorry

end NUMINAMATH_CALUDE_line_passes_first_and_fourth_quadrants_l827_82798


namespace NUMINAMATH_CALUDE_equation_solution_in_interval_l827_82712

theorem equation_solution_in_interval :
  ∃ x₀ ∈ Set.Ioo 2 3, Real.log x₀ + x₀ - 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_in_interval_l827_82712


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l827_82780

theorem complex_fraction_evaluation :
  2 + (3 / (4 + (5 / 6))) = 76 / 29 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l827_82780


namespace NUMINAMATH_CALUDE_square_side_length_l827_82708

theorem square_side_length (x : ℝ) (h : x > 0) :
  x^2 = 2 * (4 * x) → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l827_82708


namespace NUMINAMATH_CALUDE_cubic_sum_divisible_by_nine_l827_82782

theorem cubic_sum_divisible_by_nine (n : ℕ) :
  9 ∣ (n^3 + (n + 1)^3 + (n + 2)^3) := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_divisible_by_nine_l827_82782


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l827_82711

def Rectangle (O A B : ℝ × ℝ) : Prop :=
  ∃ C : ℝ × ℝ, (O.1 - A.1) * (A.1 - C.1) + (O.2 - A.2) * (A.2 - C.2) = 0 ∧
              (O.1 - B.1) * (B.1 - C.1) + (O.2 - B.2) * (B.2 - C.2) = 0

theorem rectangle_diagonal (O A B : ℝ × ℝ) (h : Rectangle O A B) :
  let OA : ℝ × ℝ := (-3, 1)
  let OB : ℝ × ℝ := (-2, k)
  k = 4 :=
sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l827_82711


namespace NUMINAMATH_CALUDE_area_of_smaller_circle_l827_82748

/-- Two circles are externally tangent with common tangent lines -/
structure TangentCircles where
  center_small : ℝ × ℝ
  center_large : ℝ × ℝ
  radius_small : ℝ
  radius_large : ℝ
  tangent_point : ℝ × ℝ
  externally_tangent : (center_small.1 - center_large.1)^2 + (center_small.2 - center_large.2)^2 = (radius_small + radius_large)^2
  radius_ratio : radius_large = 3 * radius_small

/-- Common tangent line -/
structure CommonTangent where
  P : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  PA_length : ℝ
  AB_length : ℝ
  PA_eq_AB : PA_length = AB_length
  PA_eq_8 : PA_length = 8

/-- The main theorem -/
theorem area_of_smaller_circle (tc : TangentCircles) (ct : CommonTangent) : 
  π * tc.radius_small^2 = 16 * π :=
sorry

end NUMINAMATH_CALUDE_area_of_smaller_circle_l827_82748


namespace NUMINAMATH_CALUDE_least_number_to_add_or_subtract_l827_82761

def original_number : ℕ := 856324

def is_three_digit_prime (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ Nat.Prime n

def divisible_by_three_digit_prime (n : ℕ) : Prop :=
  ∃ p : ℕ, is_three_digit_prime p ∧ p ∣ n

theorem least_number_to_add_or_subtract :
  ∀ k : ℕ, k < 46 →
    ¬(divisible_by_three_digit_prime (original_number + k) ∨
      divisible_by_three_digit_prime (original_number - k)) ∧
    (divisible_by_three_digit_prime (original_number - 46)) :=
by sorry

end NUMINAMATH_CALUDE_least_number_to_add_or_subtract_l827_82761


namespace NUMINAMATH_CALUDE_seating_theorem_l827_82791

/-- The number of seats in the row -/
def n : ℕ := 7

/-- The number of people to be seated -/
def k : ℕ := 2

/-- The number of different seating arrangements for two people in n seats
    with at least one empty seat between them -/
def seating_arrangements (n : ℕ) (k : ℕ) : ℕ :=
  (n.factorial / ((n - k).factorial * k.factorial)) - ((n - 1).factorial / ((n - k - 1).factorial * k.factorial))

theorem seating_theorem : seating_arrangements n k = 30 := by
  sorry

end NUMINAMATH_CALUDE_seating_theorem_l827_82791


namespace NUMINAMATH_CALUDE_simple_interest_problem_l827_82705

theorem simple_interest_problem (P R : ℝ) (h1 : P > 0) (h2 : R > 0) :
  (P * (R + 1) * 3 / 100 = P * R * 3 / 100 + 75) → P = 2500 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l827_82705


namespace NUMINAMATH_CALUDE_finite_n_with_prime_factors_in_A_l827_82754

theorem finite_n_with_prime_factors_in_A (A : Finset Nat) (a : Nat) 
  (h_A : ∀ p ∈ A, Nat.Prime p) (h_a : a ≥ 2) :
  ∃ S : Finset Nat, ∀ n : Nat, (∀ p : Nat, p ∣ (a^n - 1) → p ∈ A) → n ∈ S :=
by sorry

end NUMINAMATH_CALUDE_finite_n_with_prime_factors_in_A_l827_82754


namespace NUMINAMATH_CALUDE_problem_1_l827_82755

theorem problem_1 : Real.sqrt 8 - 4 * Real.sin (45 * π / 180) + (1/3)^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l827_82755


namespace NUMINAMATH_CALUDE_total_remaining_value_l827_82788

/-- Represents the types of gift cards Jack has --/
inductive GiftCardType
  | BestBuy
  | Target
  | Walmart
  | Amazon

/-- Represents the initial number and value of each type of gift card --/
def initial_gift_cards : List (GiftCardType × Nat × Nat) :=
  [(GiftCardType.BestBuy, 5, 500),
   (GiftCardType.Target, 3, 250),
   (GiftCardType.Walmart, 7, 100),
   (GiftCardType.Amazon, 2, 1000)]

/-- Represents the number of gift cards Jack sent codes for --/
def sent_gift_cards : List (GiftCardType × Nat) :=
  [(GiftCardType.BestBuy, 1),
   (GiftCardType.Walmart, 2),
   (GiftCardType.Amazon, 1)]

/-- Calculates the total value of remaining gift cards --/
def remaining_value (initial : List (GiftCardType × Nat × Nat)) (sent : List (GiftCardType × Nat)) : Nat :=
  sorry

/-- Theorem stating that the total value of gift cards Jack can still return is $4250 --/
theorem total_remaining_value : 
  remaining_value initial_gift_cards sent_gift_cards = 4250 := by
  sorry

end NUMINAMATH_CALUDE_total_remaining_value_l827_82788


namespace NUMINAMATH_CALUDE_puzzle_solution_l827_82764

/-- Represents a chip in the puzzle -/
def Chip := Fin 25

/-- Represents the arrangement of chips -/
def Arrangement := Fin 25 → Chip

/-- The initial arrangement of chips -/
def initial_arrangement : Arrangement := sorry

/-- The target arrangement of chips (in order) -/
def target_arrangement : Arrangement := sorry

/-- Represents a swap of two chips -/
def Swap := Chip × Chip

/-- Applies a swap to an arrangement -/
def apply_swap (a : Arrangement) (s : Swap) : Arrangement := sorry

/-- A sequence of swaps -/
def SwapSequence := List Swap

/-- Applies a sequence of swaps to an arrangement -/
def apply_swap_sequence (a : Arrangement) (ss : SwapSequence) : Arrangement := sorry

/-- The optimal swap sequence to solve the puzzle -/
def optimal_swap_sequence : SwapSequence := sorry

theorem puzzle_solution :
  apply_swap_sequence initial_arrangement optimal_swap_sequence = target_arrangement ∧
  optimal_swap_sequence.length = 19 := by sorry

end NUMINAMATH_CALUDE_puzzle_solution_l827_82764


namespace NUMINAMATH_CALUDE_minimum_pages_per_day_l827_82797

theorem minimum_pages_per_day (total_pages : ℕ) (days : ℕ) (pages_per_day : ℕ) : 
  total_pages = 220 → days = 7 → 
  (pages_per_day * days ≥ total_pages ∧ 
   ∀ n : ℕ, n * days ≥ total_pages → n ≥ pages_per_day) →
  pages_per_day = 32 := by
sorry

end NUMINAMATH_CALUDE_minimum_pages_per_day_l827_82797


namespace NUMINAMATH_CALUDE_geometric_arithmetic_inequality_l827_82738

/-- A geometric sequence with positive integer terms -/
def geometric_sequence (a : ℕ → ℕ) : Prop :=
  ∃ (r : ℚ), r > 0 ∧ ∀ n, a (n + 1) = a n * ⌊r⌋

/-- An arithmetic sequence -/
def arithmetic_sequence (b : ℕ → ℤ) : Prop :=
  ∃ d, ∀ n, b (n + 1) = b n + d

/-- The main theorem -/
theorem geometric_arithmetic_inequality
  (a : ℕ → ℕ) (b : ℕ → ℤ)
  (h_geo : geometric_sequence a)
  (h_arith : arithmetic_sequence b)
  (h_eq : a 6 = b 7) :
  a 3 + a 9 ≥ b 4 + b 10 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_inequality_l827_82738


namespace NUMINAMATH_CALUDE_game_points_calculation_l827_82768

/-- Calculates the total points scored in a game given points per round and number of rounds played. -/
def totalPoints (pointsPerRound : ℕ) (numRounds : ℕ) : ℕ :=
  pointsPerRound * numRounds

/-- Theorem stating that for a game with 146 points per round and 157 rounds, the total points scored is 22822. -/
theorem game_points_calculation :
  totalPoints 146 157 = 22822 := by
  sorry

end NUMINAMATH_CALUDE_game_points_calculation_l827_82768


namespace NUMINAMATH_CALUDE_max_piles_theorem_l827_82762

/-- Represents the state of the stone piles -/
structure StonePiles where
  piles : List Nat
  sum_stones : Nat
  deriving Repr

/-- Check if the piles satisfy the size constraint -/
def valid_piles (sp : StonePiles) : Prop :=
  ∀ i j, i < sp.piles.length → j < sp.piles.length →
    2 * sp.piles[i]! > sp.piles[j]! ∧ 2 * sp.piles[j]! > sp.piles[i]!

/-- The initial state with 660 stones -/
def initial_state : StonePiles :=
  { piles := [660], sum_stones := 660 }

/-- A move splits one pile into two smaller piles -/
def move (sp : StonePiles) (index : Nat) (split : Nat) : Option StonePiles :=
  if index ≥ sp.piles.length ∨ split ≥ sp.piles[index]! then none
  else some {
    piles := sp.piles.set index (sp.piles[index]! - split) |>.insertNth index split,
    sum_stones := sp.sum_stones
  }

/-- The theorem to be proved -/
theorem max_piles_theorem (sp : StonePiles) :
  sp.sum_stones = 660 →
  valid_piles sp →
  sp.piles.length ≤ 30 :=
sorry

#eval initial_state

end NUMINAMATH_CALUDE_max_piles_theorem_l827_82762


namespace NUMINAMATH_CALUDE_simplify_expression_l827_82772

theorem simplify_expression : (5 * 10^10) / (2 * 10^4 * 10^2) = 25000 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l827_82772


namespace NUMINAMATH_CALUDE_cubic_roots_condition_l827_82726

theorem cubic_roots_condition (a b c : ℝ) (α β γ : ℝ) : 
  (∀ x : ℝ, x^3 + a*x^2 + b*x + c = (x - α)*(x - β)*(x - γ)) →
  (∀ x : ℝ, x^3 + a^3*x^2 + b^3*x + c^3 = (x - α^3)*(x - β^3)*(x - γ^3)) →
  c = a*b ∧ b ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_condition_l827_82726


namespace NUMINAMATH_CALUDE_percentage_of_270_l827_82777

theorem percentage_of_270 : (33 + 1/3 : ℚ) / 100 * 270 = 90 := by sorry

end NUMINAMATH_CALUDE_percentage_of_270_l827_82777


namespace NUMINAMATH_CALUDE_carnival_spending_theorem_l827_82785

def carnival_spending (total_budget food_cost : ℕ) : ℕ :=
  let ride_cost := 2 * food_cost
  let total_spent := food_cost + ride_cost
  total_budget - total_spent

theorem carnival_spending_theorem :
  carnival_spending 100 20 = 40 :=
by sorry

end NUMINAMATH_CALUDE_carnival_spending_theorem_l827_82785


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l827_82753

theorem quadratic_equation_roots (p q : ℝ) : 
  (∃ α β : ℝ, α ≠ β ∧ 
   α^2 + p*α + q = 0 ∧ 
   β^2 + p*β + q = 0 ∧ 
   ({α, β} : Set ℝ) ⊆ {1, 2, 3, 4} ∧ 
   ({α, β} : Set ℝ) ∩ {2, 4, 5, 6} = ∅) →
  p = -4 ∧ q = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l827_82753


namespace NUMINAMATH_CALUDE_range_of_a_l827_82786

-- Define the set M
def M (a : ℝ) : Set ℝ := {x : ℝ | (a * x - 5) / (x^2 - a) < 0}

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, (3 ∈ M a) ∧ (5 ∉ M a) ↔ (1 ≤ a ∧ a < 5/3) ∨ (9 < a ∧ a ≤ 25) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l827_82786


namespace NUMINAMATH_CALUDE_fraction_simplification_l827_82722

theorem fraction_simplification : (3 : ℚ) / 462 + (17 : ℚ) / 42 = (95 : ℚ) / 231 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l827_82722


namespace NUMINAMATH_CALUDE_polynomial_rational_difference_l827_82749

theorem polynomial_rational_difference (f : ℝ → ℝ) :
  (∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c) →
  (∀ (x y : ℝ), ∃ (q : ℚ), x - y = q → ∃ (r : ℚ), f x - f y = r) →
  ∃ (b : ℚ) (c : ℝ), ∀ x, f x = b * x + c :=
by sorry

end NUMINAMATH_CALUDE_polynomial_rational_difference_l827_82749


namespace NUMINAMATH_CALUDE_taxi_ride_distance_l827_82771

/-- Calculates the distance of a taxi ride given the fare structure and total fare -/
theorem taxi_ride_distance 
  (initial_fare : ℚ) 
  (initial_distance : ℚ) 
  (additional_fare : ℚ) 
  (additional_distance : ℚ) 
  (total_fare : ℚ) 
  (h1 : initial_fare = 2)
  (h2 : initial_distance = 1/5)
  (h3 : additional_fare = 3/5)
  (h4 : additional_distance = 1/5)
  (h5 : total_fare = 127/5) : 
  ∃ (distance : ℚ), distance = 8 ∧ 
    total_fare = initial_fare + (distance - initial_distance) / additional_distance * additional_fare :=
by sorry

end NUMINAMATH_CALUDE_taxi_ride_distance_l827_82771


namespace NUMINAMATH_CALUDE_constants_are_like_terms_different_variables_not_like_terms_different_exponents_not_like_terms_like_terms_classification_l827_82702

/-- Represents an algebraic term --/
inductive Term
  | Constant (n : ℕ)
  | Variable (name : String)
  | Product (terms : List Term)

/-- Defines when two terms are like terms --/
def areLikeTerms (t1 t2 : Term) : Prop :=
  match t1, t2 with
  | Term.Constant _, Term.Constant _ => True
  | Term.Variable x, Term.Variable y => x = y
  | Term.Product l1, Term.Product l2 => l1 = l2
  | _, _ => False

/-- Theorem stating that constants are like terms --/
theorem constants_are_like_terms (a b : ℕ) :
  areLikeTerms (Term.Constant a) (Term.Constant b) := by sorry

/-- Theorem stating that terms with different variables are not like terms --/
theorem different_variables_not_like_terms (x y : String) (h : x ≠ y) :
  ¬ areLikeTerms (Term.Variable x) (Term.Variable y) := by sorry

/-- Theorem stating that terms with different exponents are not like terms --/
theorem different_exponents_not_like_terms (x : String) (a b : ℕ) (h : a ≠ b) :
  ¬ areLikeTerms 
    (Term.Product [Term.Variable x, Term.Constant a]) 
    (Term.Product [Term.Variable x, Term.Constant b]) := by sorry

/-- Main theorem combining the results for the given problem --/
theorem like_terms_classification 
  (a b : ℕ) 
  (x y z : String) 
  (h1 : x ≠ y) 
  (h2 : y ≠ z) 
  (h3 : x ≠ z) :
  areLikeTerms (Term.Constant a) (Term.Constant b) ∧
  ¬ areLikeTerms 
    (Term.Product [Term.Variable x, Term.Variable x, Term.Variable y])
    (Term.Product [Term.Variable y, Term.Variable y, Term.Variable x]) ∧
  ¬ areLikeTerms 
    (Term.Product [Term.Variable x, Term.Variable y])
    (Term.Product [Term.Variable y, Term.Variable z]) ∧
  ¬ areLikeTerms 
    (Term.Product [Term.Variable x, Term.Variable y])
    (Term.Product [Term.Variable x, Term.Variable y, Term.Variable z]) := by sorry

end NUMINAMATH_CALUDE_constants_are_like_terms_different_variables_not_like_terms_different_exponents_not_like_terms_like_terms_classification_l827_82702


namespace NUMINAMATH_CALUDE_central_academy_olympiad_l827_82728

theorem central_academy_olympiad (j s : ℕ) (hj : j > 0) (hs : s > 0) : 
  (3 * j : ℚ) / 7 = (6 * s : ℚ) / 7 → j = 2 * s := by
  sorry

end NUMINAMATH_CALUDE_central_academy_olympiad_l827_82728


namespace NUMINAMATH_CALUDE_min_value_theorem_l827_82756

theorem min_value_theorem (x : ℝ) (h : x > 4) :
  (x + 10) / Real.sqrt (x - 4) ≥ 2 * Real.sqrt 14 ∧
  ((x + 10) / Real.sqrt (x - 4) = 2 * Real.sqrt 14 ↔ x = 22) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l827_82756


namespace NUMINAMATH_CALUDE_alloy_composition_l827_82731

/-- Proves that the amount of the first alloy used is 15 kg given the specified conditions -/
theorem alloy_composition (x : ℝ) : 
  (0.12 * x + 0.10 * 35 = 0.106 * (x + 35)) → x = 15 :=
by sorry

end NUMINAMATH_CALUDE_alloy_composition_l827_82731


namespace NUMINAMATH_CALUDE_least_subtrahend_l827_82716

def is_valid (x : ℕ) : Prop :=
  (997 - x) % 5 = 3 ∧ (997 - x) % 9 = 3 ∧ (997 - x) % 11 = 3

theorem least_subtrahend :
  ∃ (x : ℕ), is_valid x ∧ ∀ (y : ℕ), y < x → ¬is_valid y :=
by sorry

end NUMINAMATH_CALUDE_least_subtrahend_l827_82716


namespace NUMINAMATH_CALUDE_coefficient_x10_is_179_l827_82767

/-- The coefficient of x^10 in the expansion of (x+2)^10(x^2-1) -/
def coefficient_x10 : ℤ := 179

/-- The expansion of (x+2)^10(x^2-1) -/
def expansion (x : ℝ) : ℝ := (x + 2)^10 * (x^2 - 1)

/-- Theorem stating that the coefficient of x^10 in the expansion is equal to 179 -/
theorem coefficient_x10_is_179 : 
  (∃ f : ℝ → ℝ, ∀ x, expansion x = coefficient_x10 * x^10 + f x ∧ (∀ ε > 0, ∃ δ > 0, ∀ x, |x| < δ → |f x| < ε * |x|^10)) :=
sorry

end NUMINAMATH_CALUDE_coefficient_x10_is_179_l827_82767


namespace NUMINAMATH_CALUDE_translation_theorem_l827_82787

/-- Represents a point in the 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a translation in the 2D Cartesian coordinate system -/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- Applies a translation to a point -/
def applyTranslation (p : Point) (t : Translation) : Point :=
  { x := p.x + t.dx, y := p.y + t.dy }

/-- Theorem: Given points A, B, and C, where AB is translated to CD,
    prove that D has the correct coordinates -/
theorem translation_theorem (A B C : Point)
    (h1 : A = { x := -1, y := 0 })
    (h2 : B = { x := 1, y := 2 })
    (h3 : C = { x := 1, y := -2 }) :
  let t : Translation := { dx := C.x - A.x, dy := C.y - A.y }
  let D : Point := applyTranslation B t
  D = { x := 3, y := 0 } := by
  sorry


end NUMINAMATH_CALUDE_translation_theorem_l827_82787


namespace NUMINAMATH_CALUDE_sqrt_two_plus_one_squared_l827_82774

theorem sqrt_two_plus_one_squared : (Real.sqrt 2 + 1)^2 = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_plus_one_squared_l827_82774


namespace NUMINAMATH_CALUDE_zeros_not_adjacent_probability_l827_82776

/-- The number of ones in the arrangement -/
def num_ones : ℕ := 4

/-- The number of zeros in the arrangement -/
def num_zeros : ℕ := 2

/-- The total number of elements to be arranged -/
def total_elements : ℕ := num_ones + num_zeros

/-- The number of spaces where zeros can be placed without being adjacent -/
def num_spaces : ℕ := num_ones + 1

/-- The probability that the zeros are not adjacent when randomly arranged -/
theorem zeros_not_adjacent_probability :
  (Nat.choose num_spaces num_zeros : ℚ) / (Nat.choose total_elements num_zeros : ℚ) = 2/3 :=
sorry

end NUMINAMATH_CALUDE_zeros_not_adjacent_probability_l827_82776


namespace NUMINAMATH_CALUDE_prob_10_or_7_prob_below_7_l827_82793

/-- Probability of hitting the 10 ring -/
def P10 : ℝ := 0.21

/-- Probability of hitting the 9 ring -/
def P9 : ℝ := 0.23

/-- Probability of hitting the 8 ring -/
def P8 : ℝ := 0.25

/-- Probability of hitting the 7 ring -/
def P7 : ℝ := 0.28

/-- The probability of hitting either the 10 or 7 ring is 0.49 -/
theorem prob_10_or_7 : P10 + P7 = 0.49 := by sorry

/-- The probability of scoring below 7 rings is 0.03 -/
theorem prob_below_7 : 1 - (P10 + P9 + P8 + P7) = 0.03 := by sorry

end NUMINAMATH_CALUDE_prob_10_or_7_prob_below_7_l827_82793


namespace NUMINAMATH_CALUDE_shepherd_boys_sticks_l827_82721

theorem shepherd_boys_sticks (x : ℕ) : 6 * x + 14 = 8 * x - 2 := by
  sorry

end NUMINAMATH_CALUDE_shepherd_boys_sticks_l827_82721


namespace NUMINAMATH_CALUDE_mary_regular_hours_l827_82713

/-- Mary's work schedule and pay structure --/
structure MaryWork where
  max_hours : ℕ
  regular_rate : ℚ
  overtime_rate : ℚ
  total_earnings : ℚ

/-- Theorem stating Mary's regular work hours --/
theorem mary_regular_hours (w : MaryWork) 
  (h1 : w.max_hours = 60)
  (h2 : w.regular_rate = 8)
  (h3 : w.overtime_rate = w.regular_rate * (1 + 1/4))
  (h4 : w.total_earnings = 560) :
  ∃ (regular_hours overtime_hours : ℕ),
    regular_hours + overtime_hours = w.max_hours ∧
    regular_hours * w.regular_rate + overtime_hours * w.overtime_rate = w.total_earnings ∧
    regular_hours = 20 := by
  sorry

end NUMINAMATH_CALUDE_mary_regular_hours_l827_82713


namespace NUMINAMATH_CALUDE_certain_number_value_l827_82727

-- Define the operation #
def hash (a b : ℝ) : ℝ := a * b - b + b^2

-- Theorem statement
theorem certain_number_value :
  ∀ x : ℝ, hash x 6 = 48 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_value_l827_82727


namespace NUMINAMATH_CALUDE_paving_cost_calculation_l827_82725

/-- The cost of paving a rectangular floor -/
def paving_cost (length width rate : ℝ) : ℝ :=
  length * width * rate

/-- Theorem: The cost of paving a rectangular floor with given dimensions and rate -/
theorem paving_cost_calculation (length width rate : ℝ) 
  (h1 : length = 5)
  (h2 : width = 4.75)
  (h3 : rate = 900) :
  paving_cost length width rate = 21375 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_calculation_l827_82725


namespace NUMINAMATH_CALUDE_inequality_implication_l827_82778

theorem inequality_implication (x y z : ℝ) (h : x^2 + x*y + x*z < 0) : y^2 > 4*x*z := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l827_82778


namespace NUMINAMATH_CALUDE_organization_size_l827_82744

/-- The total number of employees in the organization -/
def total_employees : ℕ := sorry

/-- The number of employees earning below 10k $ -/
def below_10k : ℕ := 250

/-- The number of employees earning between 10k $ and 50k $ -/
def between_10k_50k : ℕ := 500

/-- The percentage of employees earning less than 50k $ -/
def percent_below_50k : ℚ := 75 / 100

theorem organization_size :
  total_employees = 1000 :=
by
  sorry

end NUMINAMATH_CALUDE_organization_size_l827_82744


namespace NUMINAMATH_CALUDE_hcf_problem_l827_82734

theorem hcf_problem (a b : ℕ) (h : ℕ) : 
  (max a b = 600) →
  (∃ (k : ℕ), lcm a b = h * 11 * 12) →
  gcd a b = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_hcf_problem_l827_82734
