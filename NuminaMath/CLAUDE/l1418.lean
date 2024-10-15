import Mathlib

namespace NUMINAMATH_CALUDE_digits_of_large_number_l1418_141886

/-- The number of digits in a positive integer -/
def num_digits (n : ℕ) : ℕ := sorry

/-- 8^22 * 5^19 expressed as a natural number -/
def large_number : ℕ := 8^22 * 5^19

theorem digits_of_large_number :
  num_digits large_number = 35 := by sorry

end NUMINAMATH_CALUDE_digits_of_large_number_l1418_141886


namespace NUMINAMATH_CALUDE_price_first_box_is_two_l1418_141894

/-- The price of each movie in the first box -/
def price_first_box : ℝ := 2

/-- The number of movies bought from the first box -/
def num_first_box : ℕ := 10

/-- The number of movies bought from the second box -/
def num_second_box : ℕ := 5

/-- The price of each movie in the second box -/
def price_second_box : ℝ := 5

/-- The average price of all DVDs bought -/
def average_price : ℝ := 3

/-- The total number of movies bought -/
def total_movies : ℕ := num_first_box + num_second_box

theorem price_first_box_is_two :
  price_first_box * num_first_box + price_second_box * num_second_box = average_price * total_movies :=
by sorry

end NUMINAMATH_CALUDE_price_first_box_is_two_l1418_141894


namespace NUMINAMATH_CALUDE_expression_evaluation_l1418_141879

theorem expression_evaluation : 
  let a : ℚ := -1/2
  (a - 2) * (a + 2) - (a + 1) * (a - 3) = -2 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1418_141879


namespace NUMINAMATH_CALUDE_smallest_number_l1418_141874

def number_set : Finset ℕ := {5, 9, 10, 2}

theorem smallest_number : 
  ∃ (x : ℕ), x ∈ number_set ∧ ∀ y ∈ number_set, x ≤ y ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l1418_141874


namespace NUMINAMATH_CALUDE_parabola_vertex_coordinates_l1418_141861

/-- The vertex coordinates of the parabola y = 2 - (2x + 1)^2 are (-1/2, 2) -/
theorem parabola_vertex_coordinates :
  let f (x : ℝ) := 2 - (2*x + 1)^2
  ∃ (a b : ℝ), (∀ x, f x ≤ f a) ∧ f a = b ∧ a = -1/2 ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_coordinates_l1418_141861


namespace NUMINAMATH_CALUDE_cubic_tangent_line_theorem_l1418_141812

/-- Given a cubic function f(x) = ax^3 + bx^2 + cx + d, 
    if the equation of the tangent line to its graph at x=0 is 24x + y - 12 = 0, 
    then c + 2d = 0 -/
theorem cubic_tangent_line_theorem (a b c d : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^3 + b * x^2 + c * x + d
  let f' : ℝ → ℝ := λ x => 3 * a * x^2 + 2 * b * x + c
  (∀ x y, y = f x → (24 * 0 + y - 12 = 0 ↔ 24 * x + y - 12 = 0)) →
  c + 2 * d = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_tangent_line_theorem_l1418_141812


namespace NUMINAMATH_CALUDE_klinker_double_age_l1418_141860

/-- Represents the current age of Mr. Klinker -/
def klinker_age : ℕ := 35

/-- Represents the current age of Mr. Klinker's daughter -/
def daughter_age : ℕ := 10

/-- Represents the number of years until Mr. Klinker is twice as old as his daughter -/
def years_until_double : ℕ := 15

/-- Proves that in 15 years, Mr. Klinker will be twice as old as his daughter -/
theorem klinker_double_age :
  klinker_age + years_until_double = 2 * (daughter_age + years_until_double) :=
by sorry

end NUMINAMATH_CALUDE_klinker_double_age_l1418_141860


namespace NUMINAMATH_CALUDE_f_properties_l1418_141832

noncomputable section

variables (a : ℝ) (x m : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := (a / (a - 1)) * (2^x - 2^(-x))

theorem f_properties (h1 : a > 0) (h2 : a ≠ 1) :
  -- f is an odd function
  (∀ x, f a (-x) = -f a x) ∧
  -- f is decreasing when 0 < a < 1
  ((0 < a ∧ a < 1) → (∀ x₁ x₂, x₁ < x₂ → f a x₁ > f a x₂)) ∧
  -- f is increasing when a > 1
  (a > 1 → (∀ x₁ x₂, x₁ < x₂ → f a x₁ < f a x₂)) ∧
  -- For x ∈ (-1, 1), if f(m-1) + f(m) < 0, then:
  (∀ m, -1 < m ∧ m < 1 → f a (m-1) + f a m < 0 →
    ((0 < a ∧ a < 1 → 1/2 < m ∧ m < 1) ∧
     (a > 1 → 0 < m ∧ m < 1/2))) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1418_141832


namespace NUMINAMATH_CALUDE_no_decagon_partition_l1418_141869

/-- A partition of a polygon into triangles -/
structure TrianglePartition (n : ℕ) where
  black_sides : ℕ
  white_sides : ℕ
  adjacent_diff_color : Prop
  decagon_sides_black : Prop

/-- The theorem stating that a decagon cannot be partitioned in the specified manner -/
theorem no_decagon_partition : ¬ ∃ (p : TrianglePartition 10),
  p.black_sides % 3 = 0 ∧ 
  p.white_sides % 3 = 0 ∧
  p.black_sides - p.white_sides = 10 :=
sorry

end NUMINAMATH_CALUDE_no_decagon_partition_l1418_141869


namespace NUMINAMATH_CALUDE_intersection_with_complement_l1418_141809

def U : Finset ℕ := {0, 1, 2, 3}
def A : Finset ℕ := {0, 1, 2}
def B : Finset ℕ := {0, 2, 3}

theorem intersection_with_complement :
  A ∩ (U \ B) = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l1418_141809


namespace NUMINAMATH_CALUDE_tanning_salon_pricing_l1418_141837

theorem tanning_salon_pricing (first_visit_charge : ℕ) (total_customers : ℕ) (second_visits : ℕ) (third_visits : ℕ) (total_revenue : ℕ) :
  first_visit_charge = 10 →
  total_customers = 100 →
  second_visits = 30 →
  third_visits = 10 →
  total_revenue = 1240 →
  ∃ (subsequent_visit_charge : ℕ),
    subsequent_visit_charge = 6 ∧
    total_revenue = first_visit_charge * total_customers + subsequent_visit_charge * (second_visits + third_visits) :=
by sorry

end NUMINAMATH_CALUDE_tanning_salon_pricing_l1418_141837


namespace NUMINAMATH_CALUDE_complex_sum_of_parts_l1418_141801

theorem complex_sum_of_parts (a b : ℝ) : 
  let z : ℂ := Complex.mk a b
  zi = Complex.mk 1 (-2) → a + b = -3 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_of_parts_l1418_141801


namespace NUMINAMATH_CALUDE_clock_hands_right_angle_count_l1418_141897

/-- The number of times clock hands form a right angle in a 12-hour period -/
def right_angles_12h : ℕ := 22

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- The number of days we're considering -/
def days : ℕ := 2

theorem clock_hands_right_angle_count :
  (right_angles_12h * hours_per_day * days) / 12 = 88 := by
  sorry

end NUMINAMATH_CALUDE_clock_hands_right_angle_count_l1418_141897


namespace NUMINAMATH_CALUDE_identical_pairs_x_value_l1418_141884

def star : (ℤ × ℤ) → (ℤ × ℤ) → (ℤ × ℤ) := λ (a, b) (c, d) ↦ (a - c, b + d)

theorem identical_pairs_x_value :
  ∀ x y : ℤ, star (2, 2) (4, 1) = star (x, y) (1, 4) → x = -1 :=
by sorry

end NUMINAMATH_CALUDE_identical_pairs_x_value_l1418_141884


namespace NUMINAMATH_CALUDE_complex_number_on_line_l1418_141893

def complex_number (a : ℝ) : ℂ := (a : ℂ) - Complex.I

theorem complex_number_on_line (a : ℝ) :
  let z := 1 / complex_number a
  (z.re : ℝ) + 2 * (z.im : ℝ) = 0 → a = -2 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_on_line_l1418_141893


namespace NUMINAMATH_CALUDE_equation_represents_two_intersecting_lines_l1418_141859

-- Define the equation
def equation (x y : ℝ) : Prop := (x - y)^2 = 3 * x^2 - y^2

-- Theorem statement
theorem equation_represents_two_intersecting_lines :
  ∃ (m₁ m₂ : ℝ), m₁ ≠ m₂ ∧ 
  (∀ (x y : ℝ), equation x y ↔ (y = m₁ * x ∨ y = m₂ * x)) :=
sorry

end NUMINAMATH_CALUDE_equation_represents_two_intersecting_lines_l1418_141859


namespace NUMINAMATH_CALUDE_shoes_price_proof_l1418_141855

theorem shoes_price_proof (total_cost jersey_count shoe_count : ℕ) 
  (jersey_price_ratio : ℚ) (h1 : total_cost = 560) (h2 : jersey_count = 4) 
  (h3 : shoe_count = 6) (h4 : jersey_price_ratio = 1/4) : 
  shoe_count * (total_cost / (shoe_count + jersey_count * jersey_price_ratio)) = 480 := by
  sorry

end NUMINAMATH_CALUDE_shoes_price_proof_l1418_141855


namespace NUMINAMATH_CALUDE_fast_food_cost_l1418_141850

/-- The total cost of buying fast food -/
def total_cost (a b : ℕ) : ℕ := 30 * a + 20 * b

/-- The price of one serving of type A fast food -/
def price_A : ℕ := 30

/-- The price of one serving of type B fast food -/
def price_B : ℕ := 20

/-- Theorem: The total cost of buying 'a' servings of type A fast food and 'b' servings of type B fast food is 30a + 20b yuan -/
theorem fast_food_cost (a b : ℕ) : total_cost a b = price_A * a + price_B * b := by
  sorry

end NUMINAMATH_CALUDE_fast_food_cost_l1418_141850


namespace NUMINAMATH_CALUDE_triangle_max_area_l1418_141826

theorem triangle_max_area (A B C : Real) (a b c : Real) :
  -- Triangle ABC with circumradius 1
  (a / Real.sin A = 2) ∧ (b / Real.sin B = 2) ∧ (c / Real.sin C = 2) →
  -- Given condition
  (Real.tan A) / (Real.tan B) = (2 * c - b) / b →
  -- Theorem: Maximum area is √3/2
  (∃ (S : Real), S = (1/2) * b * c * Real.sin A ∧
                S ≤ Real.sqrt 3 / 2 ∧
                (∃ (B' C' : Real), S = Real.sqrt 3 / 2)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l1418_141826


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1418_141865

universe u

def U : Set (Fin 5) := {1, 2, 3, 4, 5}
def M : Set (Fin 5) := {1, 2}
def N : Set (Fin 5) := {3, 5}

theorem complement_intersection_theorem :
  (U \ M) ∩ N = {3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1418_141865


namespace NUMINAMATH_CALUDE_students_taking_no_subjects_l1418_141827

theorem students_taking_no_subjects (total : ℕ) (music art dance : ℕ) 
  (music_and_art music_and_dance art_and_dance : ℕ) (all_three : ℕ) :
  total = 800 ∧ 
  music = 140 ∧ 
  art = 90 ∧ 
  dance = 75 ∧
  music_and_art = 50 ∧
  music_and_dance = 30 ∧
  art_and_dance = 25 ∧
  all_three = 20 →
  total - (music + art + dance - music_and_art - music_and_dance - art_and_dance + all_three) = 580 := by
  sorry

end NUMINAMATH_CALUDE_students_taking_no_subjects_l1418_141827


namespace NUMINAMATH_CALUDE_grape_juice_mixture_l1418_141883

theorem grape_juice_mixture (initial_volume : ℝ) (added_volume : ℝ) (final_percentage : ℝ) :
  initial_volume = 40 →
  added_volume = 10 →
  final_percentage = 0.36 →
  let final_volume := initial_volume + added_volume
  let initial_percentage := (final_percentage * final_volume - added_volume) / initial_volume
  initial_percentage = 0.2 := by
sorry

end NUMINAMATH_CALUDE_grape_juice_mixture_l1418_141883


namespace NUMINAMATH_CALUDE_fair_coin_probability_l1418_141858

-- Define a fair coin
def fair_coin := { p : ℝ | 0 ≤ p ∧ p ≤ 1 ∧ p = 1 - p }

-- Define the number of tosses
def num_tosses : ℕ := 20

-- Define the number of heads
def num_heads : ℕ := 8

-- Define the number of tails
def num_tails : ℕ := 12

-- Theorem statement
theorem fair_coin_probability : 
  ∀ (p : ℝ), p ∈ fair_coin → p = 1/2 :=
sorry

end NUMINAMATH_CALUDE_fair_coin_probability_l1418_141858


namespace NUMINAMATH_CALUDE_equidistant_locus_equation_l1418_141820

/-- The locus of points equidistant from the coordinate axes -/
def EquidistantLocus : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | abs p.1 = abs p.2}

/-- The equation |x| - |y| = 0 holds for points on the locus -/
theorem equidistant_locus_equation (p : ℝ × ℝ) :
  p ∈ EquidistantLocus ↔ abs p.1 - abs p.2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_locus_equation_l1418_141820


namespace NUMINAMATH_CALUDE_base7_product_l1418_141870

/-- Converts a base 7 number to base 10 -/
def toBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 7^i) 0

/-- Converts a base 10 number to base 7 -/
def toBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

/-- The problem statement -/
theorem base7_product :
  let a := [1, 2, 3]  -- 321 in base 7 (least significant digit first)
  let b := [3, 1]     -- 13 in base 7 (least significant digit first)
  toBase7 (toBase10 a * toBase10 b) = [3, 0, 5, 4] := by
  sorry

end NUMINAMATH_CALUDE_base7_product_l1418_141870


namespace NUMINAMATH_CALUDE_right_triangle_pq_length_l1418_141862

/-- Represents a right triangle PQR -/
structure RightTriangle where
  PQ : ℝ
  PR : ℝ
  QR : ℝ
  tanQ : ℝ

/-- Theorem: In a right triangle PQR where ∠R = 90°, tan Q = 3/4, and PR = 12, PQ = 9 -/
theorem right_triangle_pq_length 
  (t : RightTriangle) 
  (h1 : t.tanQ = 3 / 4) 
  (h2 : t.PR = 12) : 
  t.PQ = 9 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_pq_length_l1418_141862


namespace NUMINAMATH_CALUDE_fort_blocks_theorem_l1418_141808

/-- Represents the dimensions of a rectangular fort -/
structure FortDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of blocks needed for a fort with given dimensions and wall thickness -/
def blocksNeeded (dimensions : FortDimensions) (wallThickness : ℕ) : ℕ :=
  let totalVolume := dimensions.length * dimensions.width * dimensions.height
  let interiorLength := dimensions.length - 2 * wallThickness
  let interiorWidth := dimensions.width - 2 * wallThickness
  let interiorHeight := dimensions.height - wallThickness
  let interiorVolume := interiorLength * interiorWidth * interiorHeight
  totalVolume - interiorVolume

/-- Theorem stating that a fort with given dimensions and wall thickness requires 728 blocks -/
theorem fort_blocks_theorem :
  let dimensions : FortDimensions := ⟨15, 12, 6⟩
  let wallThickness : ℕ := 2
  blocksNeeded dimensions wallThickness = 728 := by
  sorry

end NUMINAMATH_CALUDE_fort_blocks_theorem_l1418_141808


namespace NUMINAMATH_CALUDE_game_show_probability_l1418_141878

theorem game_show_probability : 
  let num_tables : ℕ := 3
  let boxes_per_table : ℕ := 3
  let zonk_boxes_per_table : ℕ := 1
  let prob_no_zonk_per_table : ℚ := (boxes_per_table - zonk_boxes_per_table) / boxes_per_table
  (prob_no_zonk_per_table ^ num_tables : ℚ) = 8 / 27 := by sorry

end NUMINAMATH_CALUDE_game_show_probability_l1418_141878


namespace NUMINAMATH_CALUDE_fraction_not_simplifiable_l1418_141840

theorem fraction_not_simplifiable (n : ℕ) : 
  (∃ k : ℕ, n = 6 * k + 1 ∨ n = 6 * k + 3) ↔ 
  ¬∃ (a b : ℕ), (n^2 + 2 : ℚ) / (n * (n + 1)) = (a : ℚ) / b ∧ 
                gcd a b = 1 ∧ 
                b < n * (n + 1) :=
sorry

end NUMINAMATH_CALUDE_fraction_not_simplifiable_l1418_141840


namespace NUMINAMATH_CALUDE_diagonal_length_of_quadrilateral_l1418_141818

/-- The length of a diagonal in a quadrilateral with given area and offsets -/
theorem diagonal_length_of_quadrilateral (area : ℝ) (offset1 offset2 : ℝ) 
  (h_area : area = 140)
  (h_offset1 : offset1 = 8)
  (h_offset2 : offset2 = 2)
  (h_quad_area : area = (1/2) * (offset1 + offset2) * diagonal_length) :
  diagonal_length = 28 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_length_of_quadrilateral_l1418_141818


namespace NUMINAMATH_CALUDE_f_zero_eq_three_l1418_141846

noncomputable def f (x : ℝ) : ℝ :=
  if x = 1 then 0  -- handle the case where x = 1 (2x-1 = 1)
  else (1 - ((x + 1) / 2)^2) / ((x + 1) / 2)^2

theorem f_zero_eq_three :
  f 0 = 3 :=
by sorry

end NUMINAMATH_CALUDE_f_zero_eq_three_l1418_141846


namespace NUMINAMATH_CALUDE_sin_45_degrees_l1418_141864

theorem sin_45_degrees : Real.sin (π / 4) = 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_45_degrees_l1418_141864


namespace NUMINAMATH_CALUDE_max_daily_sales_revenue_l1418_141885

-- Define the domain of t
def T : Set ℕ := {t : ℕ | 1 ≤ t ∧ t ≤ 20}

-- Define the daily sales volume function
def f (t : ℕ) : ℝ := -t + 30

-- Define the daily sales price function
def g (t : ℕ) : ℝ :=
  if t ≤ 10 then 2 * t + 40 else 15

-- Define the daily sales revenue function
def S (t : ℕ) : ℝ := f t * g t

-- Theorem stating the maximum daily sales revenue
theorem max_daily_sales_revenue :
  ∃ (t_max : ℕ), t_max ∈ T ∧
    (∀ (t : ℕ), t ∈ T → S t ≤ S t_max) ∧
    t_max = 5 ∧ S t_max = 1250 := by
  sorry

end NUMINAMATH_CALUDE_max_daily_sales_revenue_l1418_141885


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l1418_141838

/-- Given a hyperbola C₁ and a parabola C₂, prove that the focus of C₂ has coordinates (0, 3/2) -/
theorem parabola_focus_coordinates 
  (a b p : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hp : p > 0) 
  (C₁ : ℝ → ℝ → Prop) 
  (C₂ : ℝ → ℝ → Prop) 
  (h_C₁ : ∀ x y, C₁ x y ↔ x^2 / a^2 - y^2 / b^2 = 1)
  (h_C₂ : ∀ x y, C₂ x y ↔ x^2 = 2 * p * y)
  (h_eccentricity : a^2 + b^2 = 2 * a^2)  -- Eccentricity of C₁ is √2
  (P : ℝ × ℝ) 
  (h_P_on_C₂ : C₂ P.1 P.2)
  (h_tangent_parallel : ∃ (m : ℝ), m = 1 ∨ m = -1 ∧ 
    ∀ x y, C₂ x y → (y - P.2) = m * (x - P.1))
  (F : ℝ × ℝ)
  (h_F_focus : F.1 = 0 ∧ F.2 = p / 2)
  (h_PF_distance : (P.1 - F.1)^2 + (P.2 - F.2)^2 = 9)  -- |PF| = 3
  : F = (0, 3/2) := by sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_l1418_141838


namespace NUMINAMATH_CALUDE_farm_animal_difference_l1418_141898

/-- Represents the number of horses and cows on a farm before and after a transaction --/
structure FarmAnimals where
  initial_horses : ℕ
  initial_cows : ℕ
  final_horses : ℕ
  final_cows : ℕ

/-- The conditions of the farm animal problem --/
def farm_conditions (farm : FarmAnimals) : Prop :=
  farm.initial_horses = 6 * farm.initial_cows ∧
  farm.final_horses = farm.initial_horses - 15 ∧
  farm.final_cows = farm.initial_cows + 15 ∧
  farm.final_horses = 3 * farm.final_cows

theorem farm_animal_difference (farm : FarmAnimals) 
  (h : farm_conditions farm) : farm.final_horses - farm.final_cows = 70 := by
  sorry

end NUMINAMATH_CALUDE_farm_animal_difference_l1418_141898


namespace NUMINAMATH_CALUDE_r_power_sum_l1418_141899

theorem r_power_sum (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_r_power_sum_l1418_141899


namespace NUMINAMATH_CALUDE_high_correlation_implies_r_near_one_l1418_141817

-- Define the correlation coefficient
def correlation_coefficient (x y : List ℝ) : ℝ := sorry

-- Define what it means for a correlation to be "very high"
def is_very_high_correlation (r : ℝ) : Prop := sorry

-- Theorem statement
theorem high_correlation_implies_r_near_one 
  (x y : List ℝ) (r : ℝ) 
  (h1 : r = correlation_coefficient x y) 
  (h2 : is_very_high_correlation r) : 
  ∀ ε > 0, |r| > 1 - ε := by
  sorry

end NUMINAMATH_CALUDE_high_correlation_implies_r_near_one_l1418_141817


namespace NUMINAMATH_CALUDE_calendar_sum_l1418_141802

/-- Given three consecutive numbers in a vertical column of a calendar where the top number is n,
    the sum of these three numbers is equal to 3n + 21. -/
theorem calendar_sum (n : ℕ) : n + (n + 7) + (n + 14) = 3 * n + 21 := by
  sorry

end NUMINAMATH_CALUDE_calendar_sum_l1418_141802


namespace NUMINAMATH_CALUDE_max_sum_of_cubes_max_sum_of_cubes_attained_l1418_141890

theorem max_sum_of_cubes (a b c d e : ℝ) 
  (h : a^2 + b^2 + c^2 + d^2 + e^2 = 9) : 
  a^3 + b^3 + c^3 + d^3 + e^3 ≤ 27 :=
by sorry

theorem max_sum_of_cubes_attained (ε : ℝ) (hε : ε > 0) : 
  ∃ a b c d e : ℝ, a^2 + b^2 + c^2 + d^2 + e^2 = 9 ∧ 
  a^3 + b^3 + c^3 + d^3 + e^3 > 27 - ε :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_cubes_max_sum_of_cubes_attained_l1418_141890


namespace NUMINAMATH_CALUDE_all_solutions_irrational_l1418_141828

/-- A number is rational if it can be expressed as a ratio of two integers -/
def IsRational (x : ℝ) : Prop := ∃ (m n : ℤ), n ≠ 0 ∧ x = m / n

/-- A number is irrational if it is not rational -/
def IsIrrational (x : ℝ) : Prop := ¬ IsRational x

/-- The equation in question -/
def SatisfiesEquation (x : ℝ) : Prop := 0.001 * x^3 + x^2 - 1 = 0

theorem all_solutions_irrational :
  ∀ x : ℝ, SatisfiesEquation x → IsIrrational x := by
  sorry

end NUMINAMATH_CALUDE_all_solutions_irrational_l1418_141828


namespace NUMINAMATH_CALUDE_robin_gum_count_l1418_141805

/-- Calculates the total number of gum pieces given the number of packages, pieces per package, and extra pieces. -/
def total_gum_pieces (packages : ℕ) (pieces_per_package : ℕ) (extra_pieces : ℕ) : ℕ :=
  packages * pieces_per_package + extra_pieces

/-- Proves that Robin has 997 pieces of gum given the specified conditions. -/
theorem robin_gum_count :
  total_gum_pieces 43 23 8 = 997 := by
  sorry

end NUMINAMATH_CALUDE_robin_gum_count_l1418_141805


namespace NUMINAMATH_CALUDE_intersection_point_existence_l1418_141845

theorem intersection_point_existence : ∃ x : ℝ, 1/2 < x ∧ x < 1 ∧ Real.exp x = -2*x + 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_existence_l1418_141845


namespace NUMINAMATH_CALUDE_hide_and_seek_l1418_141834

-- Define the participants
variable (Andrew Boris Vasya Gena Denis : Prop)

-- Define the conditions
variable (h1 : Andrew → (Boris ∧ ¬Vasya))
variable (h2 : Boris → (Gena ∨ Denis))
variable (h3 : ¬Vasya → (¬Boris ∧ ¬Denis))
variable (h4 : ¬Andrew → (Boris ∧ ¬Gena))

-- Theorem statement
theorem hide_and_seek :
  Boris ∧ Vasya ∧ Denis ∧ ¬Andrew ∧ ¬Gena := by sorry

end NUMINAMATH_CALUDE_hide_and_seek_l1418_141834


namespace NUMINAMATH_CALUDE_total_cantaloupes_l1418_141876

def keith_cantaloupes : ℝ := 29.5
def fred_cantaloupes : ℝ := 16.25
def jason_cantaloupes : ℝ := 20.75
def olivia_cantaloupes : ℝ := 12.5
def emily_cantaloupes : ℝ := 15.8

theorem total_cantaloupes : 
  keith_cantaloupes + fred_cantaloupes + jason_cantaloupes + olivia_cantaloupes + emily_cantaloupes = 94.8 := by
  sorry

end NUMINAMATH_CALUDE_total_cantaloupes_l1418_141876


namespace NUMINAMATH_CALUDE_dodecagon_diagonals_and_triangles_l1418_141892

def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

def number_of_triangles (n : ℕ) : ℕ := n.choose 3

theorem dodecagon_diagonals_and_triangles :
  let n : ℕ := 12
  number_of_diagonals n = 54 ∧ number_of_triangles n = 220 := by sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_and_triangles_l1418_141892


namespace NUMINAMATH_CALUDE_degenerate_iff_c_eq_52_l1418_141849

/-- A point in R^2 -/
structure Point where
  x : ℝ
  y : ℝ

/-- The equation of the graph -/
def equation (p : Point) (c : ℝ) : Prop :=
  3 * p.x^2 + p.y^2 + 6 * p.x - 14 * p.y + c = 0

/-- The graph is degenerate (represents a single point) -/
def is_degenerate (c : ℝ) : Prop :=
  ∃! p : Point, equation p c

theorem degenerate_iff_c_eq_52 :
  ∀ c : ℝ, is_degenerate c ↔ c = 52 := by sorry

end NUMINAMATH_CALUDE_degenerate_iff_c_eq_52_l1418_141849


namespace NUMINAMATH_CALUDE_cookies_problem_l1418_141873

theorem cookies_problem (mona jasmine rachel : ℕ) 
  (h1 : jasmine = mona - 5)
  (h2 : rachel = jasmine + 10)
  (h3 : mona + jasmine + rachel = 60) :
  mona = 20 := by
sorry

end NUMINAMATH_CALUDE_cookies_problem_l1418_141873


namespace NUMINAMATH_CALUDE_equation_solution_l1418_141815

theorem equation_solution (y : ℝ) (h : y ≠ 2) : 
  (y^2 - 10*y + 24) / (y - 2) + (4*y^2 + 8*y - 48) / (4*y - 8) = 0 ↔ y = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1418_141815


namespace NUMINAMATH_CALUDE_initial_mean_calculation_l1418_141848

/-- Given 20 observations with an initial mean, prove that correcting one observation
    from 40 to 25 results in a new mean of 34.9 if and only if the initial mean was 35.65 -/
theorem initial_mean_calculation (n : ℕ) (initial_mean corrected_mean : ℝ) :
  n = 20 ∧
  corrected_mean = 34.9 ∧
  (n : ℝ) * initial_mean - 15 = (n : ℝ) * corrected_mean →
  initial_mean = 35.65 := by
  sorry

end NUMINAMATH_CALUDE_initial_mean_calculation_l1418_141848


namespace NUMINAMATH_CALUDE_equal_intercept_line_correct_l1418_141836

/-- A line passing through point (1, 2) with equal x and y intercepts -/
def equal_intercept_line (x y : ℝ) : Prop :=
  x + y - 3 = 0

theorem equal_intercept_line_correct :
  (equal_intercept_line 1 2) ∧
  (∃ (a : ℝ), a ≠ 0 ∧ equal_intercept_line a 0 ∧ equal_intercept_line 0 a) :=
by sorry

end NUMINAMATH_CALUDE_equal_intercept_line_correct_l1418_141836


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1418_141857

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ, 
  3 * X^4 + 13 * X^3 + 5 * X^2 - 10 * X + 20 = 
  (X^2 + 5 * X + 1) * q + (-68 * X + 8) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1418_141857


namespace NUMINAMATH_CALUDE_no_divisors_between_30_and_40_l1418_141882

theorem no_divisors_between_30_and_40 : ∀ n : ℕ, 30 < n → n < 40 → ¬(2^28 - 1) % n = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_divisors_between_30_and_40_l1418_141882


namespace NUMINAMATH_CALUDE_resulting_polynomial_degree_is_eight_l1418_141824

/-- The degree of the polynomial resulting from the given operations -/
def resultingPolynomialDegree : ℕ :=
  let expr1 := fun x : ℝ => x^4
  let expr2 := fun x : ℝ => x^2 - 1/x^2
  let expr3 := fun x : ℝ => 1 - 3/x + 3/x^2
  let squaredExpr2 := fun x : ℝ => (expr2 x)^2
  let result := fun x : ℝ => (expr1 x) * (squaredExpr2 x) * (expr3 x)
  8

/-- Theorem stating that the degree of the resulting polynomial is 8 -/
theorem resulting_polynomial_degree_is_eight :
  resultingPolynomialDegree = 8 := by sorry

end NUMINAMATH_CALUDE_resulting_polynomial_degree_is_eight_l1418_141824


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l1418_141813

open Real

theorem trigonometric_inequality : 
  let a : ℝ := sin (46 * π / 180)
  let b : ℝ := cos (46 * π / 180)
  let c : ℝ := tan (46 * π / 180)
  c > a ∧ a > b :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l1418_141813


namespace NUMINAMATH_CALUDE_part_one_part_two_l1418_141891

-- Define the linear function f
def f (a b x : ℝ) : ℝ := a * x + b

-- Define the function g
def g (m x : ℝ) : ℝ := (x + m) * (4 * x + 1)

-- Theorem for part (I)
theorem part_one (a b : ℝ) :
  (∀ x y, x < y → f a b x < f a b y) →
  (∀ x, f a b (f a b x) = 16 * x + 5) →
  a = 4 ∧ b = 1 := by sorry

-- Theorem for part (II)
theorem part_two (m : ℝ) :
  (∀ x y, 1 ≤ x ∧ x < y → g m x < g m y) →
  m ≥ -9/4 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1418_141891


namespace NUMINAMATH_CALUDE_gcd_111_1850_l1418_141800

theorem gcd_111_1850 : Nat.gcd 111 1850 = 37 := by sorry

end NUMINAMATH_CALUDE_gcd_111_1850_l1418_141800


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1418_141866

theorem fixed_point_of_exponential_function 
  (a : ℝ) (ha : a > 0 ∧ a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ 7 + a^(x - 3)
  f 3 = 8 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1418_141866


namespace NUMINAMATH_CALUDE_zeros_before_first_nonzero_of_fraction_l1418_141871

/-- The number of zeros after the decimal point and before the first non-zero digit
    in the decimal representation of 1/(2^3 * 5^6) -/
def zeros_before_first_nonzero : ℕ :=
  6

/-- The fraction we're considering -/
def fraction : ℚ :=
  1 / (2^3 * 5^6)

theorem zeros_before_first_nonzero_of_fraction :
  zeros_before_first_nonzero = 
    (fraction.den.factors.count 2 + fraction.den.factors.count 5).min
      (fraction.den.factors.count 5) :=
by
  sorry

end NUMINAMATH_CALUDE_zeros_before_first_nonzero_of_fraction_l1418_141871


namespace NUMINAMATH_CALUDE_cube_volume_problem_l1418_141851

theorem cube_volume_problem (s : ℝ) : 
  s > 0 →
  s^3 - ((s + 2) * (s - 3) * s) = 8 →
  s^3 = 8 :=
by sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l1418_141851


namespace NUMINAMATH_CALUDE_grassy_plot_width_l1418_141839

/-- Proves that the width of a rectangular grassy plot is 60 meters given the specified conditions --/
theorem grassy_plot_width :
  ∀ (w : ℝ),
  let plot_length : ℝ := 100
  let path_width : ℝ := 2.5
  let gravel_cost_per_sqm : ℝ := 0.9  -- 90 paise = 0.9 rupees
  let total_gravel_cost : ℝ := 742.5
  let total_length : ℝ := plot_length + 2 * path_width
  let total_width : ℝ := w + 2 * path_width
  let path_area : ℝ := total_length * total_width - plot_length * w
  gravel_cost_per_sqm * path_area = total_gravel_cost →
  w = 60 := by
sorry

end NUMINAMATH_CALUDE_grassy_plot_width_l1418_141839


namespace NUMINAMATH_CALUDE_parabola_hyperbola_focus_l1418_141887

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the hyperbola
def hyperbola (x y : ℝ) (n : ℝ) : Prop := x^2/3 - y^2/n = 1

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (2, 0)

-- Define a predicate for a point being a focus of the hyperbola
def is_hyperbola_focus (x y : ℝ) (n : ℝ) : Prop :=
  hyperbola x y n ∧ x^2 - y^2 = 3 + n

-- State the theorem
theorem parabola_hyperbola_focus (n : ℝ) :
  (∃ x y, is_hyperbola_focus x y n ∧ (x, y) = parabola_focus) → n = 1 :=
sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_focus_l1418_141887


namespace NUMINAMATH_CALUDE_cone_central_angle_l1418_141830

/-- Given a circular piece of paper with radius 18 cm, when partially cut to form a cone
    with radius 8 cm and volume 128π cm³, the central angle of the sector used to create
    the cone is approximately 53 degrees. -/
theorem cone_central_angle (paper_radius : ℝ) (cone_radius : ℝ) (cone_volume : ℝ) :
  paper_radius = 18 →
  cone_radius = 8 →
  cone_volume = 128 * Real.pi →
  ∃ (central_angle : ℝ), 52 < central_angle ∧ central_angle < 54 := by
  sorry

end NUMINAMATH_CALUDE_cone_central_angle_l1418_141830


namespace NUMINAMATH_CALUDE_vanessa_scoring_record_l1418_141842

/-- Vanessa's new scoring record in a basketball game -/
theorem vanessa_scoring_record (total_score : ℕ) (other_players : ℕ) (average_score : ℕ) 
  (h1 : total_score = 55)
  (h2 : other_players = 7)
  (h3 : average_score = 4) : 
  total_score - (other_players * average_score) = 27 := by
  sorry

end NUMINAMATH_CALUDE_vanessa_scoring_record_l1418_141842


namespace NUMINAMATH_CALUDE_notebook_calculation_sara_sister_notebooks_l1418_141895

theorem notebook_calculation : ℕ → ℕ
  | initial =>
    let ordered := initial + (initial * 3 / 2)
    let after_loss := ordered - 2
    let after_sale := after_loss - (after_loss / 4)
    let final := after_sale - 3
    final

theorem sara_sister_notebooks :
  notebook_calculation 4 = 3 := by sorry

end NUMINAMATH_CALUDE_notebook_calculation_sara_sister_notebooks_l1418_141895


namespace NUMINAMATH_CALUDE_x_squared_in_set_l1418_141852

theorem x_squared_in_set (x : ℝ) : x^2 ∈ ({1, 0, x} : Set ℝ) → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_in_set_l1418_141852


namespace NUMINAMATH_CALUDE_product_remainder_mod_five_l1418_141872

theorem product_remainder_mod_five : (1483 * 1773 * 1827 * 2001) % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_five_l1418_141872


namespace NUMINAMATH_CALUDE_sum_of_roots_is_twelve_l1418_141807

/-- A function satisfying the symmetry property g(3+x) = g(3-x) -/
def SymmetricAboutThree (g : ℝ → ℝ) : Prop :=
  ∀ x, g (3 + x) = g (3 - x)

/-- The property that a function has exactly four distinct real roots -/
def HasFourDistinctRealRoots (g : ℝ → ℝ) : Prop :=
  ∃ (a b c d : ℝ), (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
    (g a = 0 ∧ g b = 0 ∧ g c = 0 ∧ g d = 0) ∧
    (∀ x, g x = 0 → x = a ∨ x = b ∨ x = c ∨ x = d)

/-- The main theorem statement -/
theorem sum_of_roots_is_twelve (g : ℝ → ℝ) 
  (h1 : SymmetricAboutThree g) (h2 : HasFourDistinctRealRoots g) :
  ∃ (a b c d : ℝ), (HasFourDistinctRealRoots g → g a = 0 ∧ g b = 0 ∧ g c = 0 ∧ g d = 0) ∧
    a + b + c + d = 12 :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_twelve_l1418_141807


namespace NUMINAMATH_CALUDE_cars_produced_in_north_america_l1418_141843

theorem cars_produced_in_north_america :
  ∀ (total_cars europe_cars north_america_cars : ℕ),
    total_cars = 6755 →
    europe_cars = 2871 →
    total_cars = europe_cars + north_america_cars →
    north_america_cars = 3884 :=
by
  sorry

end NUMINAMATH_CALUDE_cars_produced_in_north_america_l1418_141843


namespace NUMINAMATH_CALUDE_base2_to_base4_conversion_l1418_141814

/-- Converts a natural number from base 2 to base 4 -/
def base2ToBase4 (n : ℕ) : ℕ := sorry

/-- The base 2 representation of the number -/
def base2Number : ℕ := 1011101100

/-- The expected base 4 representation of the number -/
def expectedBase4Number : ℕ := 23230

theorem base2_to_base4_conversion :
  base2ToBase4 base2Number = expectedBase4Number := by sorry

end NUMINAMATH_CALUDE_base2_to_base4_conversion_l1418_141814


namespace NUMINAMATH_CALUDE_folder_cost_l1418_141853

/-- The cost of office supplies problem -/
theorem folder_cost (pencil_cost : ℚ) (pencil_count : ℕ) (folder_count : ℕ) (total_cost : ℚ) : 
  pencil_cost = 1/2 →
  pencil_count = 24 →
  folder_count = 20 →
  total_cost = 30 →
  (total_cost - pencil_cost * pencil_count) / folder_count = 9/10 := by
  sorry

end NUMINAMATH_CALUDE_folder_cost_l1418_141853


namespace NUMINAMATH_CALUDE_base_n_multiple_of_11_l1418_141875

theorem base_n_multiple_of_11 : 
  ∀ n : ℕ, 2 ≤ n ∧ n ≤ 100 → 
  ¬(11 ∣ (7 + 4*n + 6*n^2 + 3*n^3 + 4*n^4 + 3*n^5)) := by
sorry

end NUMINAMATH_CALUDE_base_n_multiple_of_11_l1418_141875


namespace NUMINAMATH_CALUDE_f_20_l1418_141889

/-- A linear function with specific properties -/
def f (x : ℝ) : ℝ := sorry

/-- The function f satisfies f(0) = 3 -/
axiom f_0 : f 0 = 3

/-- The function f increases by 10 when x increases by 4 -/
axiom f_increase (x : ℝ) : f (x + 4) - f x = 10

/-- Theorem: f(20) = 53 -/
theorem f_20 : f 20 = 53 := by sorry

end NUMINAMATH_CALUDE_f_20_l1418_141889


namespace NUMINAMATH_CALUDE_special_quadratic_a_range_l1418_141841

/-- A quadratic function with specific properties -/
structure SpecialQuadratic where
  f : ℝ → ℝ
  is_quadratic : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c
  symmetry : ∀ x, f (2 + x) = f (2 - x)
  inequality : ∃ a : ℝ, f a ≤ f 0 ∧ f 0 < f 1

/-- The range of 'a' for a special quadratic function -/
def range_of_a (q : SpecialQuadratic) : Set ℝ :=
  {x | x ≤ 0 ∨ x ≥ 4}

/-- Theorem stating the range of 'a' for a special quadratic function -/
theorem special_quadratic_a_range (q : SpecialQuadratic) :
  ∀ a : ℝ, q.f a ≤ q.f 0 → a ∈ range_of_a q := by
  sorry

end NUMINAMATH_CALUDE_special_quadratic_a_range_l1418_141841


namespace NUMINAMATH_CALUDE_distribute_five_items_three_bags_l1418_141816

/-- The number of ways to distribute n distinct items into k identical bags,
    allowing for empty bags and the possibility of leaving items out. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- Theorem stating that there are 106 ways to distribute 5 distinct items
    into 3 identical bags, allowing for empty bags and the possibility of
    leaving one item out. -/
theorem distribute_five_items_three_bags : distribute 5 3 = 106 := by sorry

end NUMINAMATH_CALUDE_distribute_five_items_three_bags_l1418_141816


namespace NUMINAMATH_CALUDE_simon_initial_stamps_l1418_141803

/-- The number of stamps Simon has after receiving stamps from friends -/
def total_stamps : ℕ := 61

/-- The number of stamps Simon received from friends -/
def received_stamps : ℕ := 27

/-- The number of stamps Simon initially had -/
def initial_stamps : ℕ := total_stamps - received_stamps

theorem simon_initial_stamps :
  initial_stamps = 34 := by sorry

end NUMINAMATH_CALUDE_simon_initial_stamps_l1418_141803


namespace NUMINAMATH_CALUDE_dvd_sales_l1418_141811

theorem dvd_sales (dvd cd : ℕ) : 
  dvd = (1.6 : ℝ) * cd →
  dvd + cd = 273 →
  dvd = 168 := by
sorry

end NUMINAMATH_CALUDE_dvd_sales_l1418_141811


namespace NUMINAMATH_CALUDE_trigonometric_equation_solutions_l1418_141831

theorem trigonometric_equation_solutions (x : ℝ) : 
  (1 + Real.sin x + Real.cos (3 * x) = Real.cos x + Real.sin (2 * x) + Real.cos (2 * x)) ↔ 
  (∃ k : ℤ, x = k * Real.pi ∨ 
            x = (-1)^(k+1) * Real.pi / 6 + k * Real.pi ∨ 
            x = Real.pi / 3 + 2 * k * Real.pi ∨ 
            x = -Real.pi / 3 + 2 * k * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solutions_l1418_141831


namespace NUMINAMATH_CALUDE_not_prime_sum_product_l1418_141881

theorem not_prime_sum_product (a b c d : ℕ) 
  (h_pos : 0 < d ∧ d < c ∧ c < b ∧ b < a) 
  (h_eq : a * c + b * d = (b + d + a - c) * (b + d - a + c)) : 
  ¬ Nat.Prime (a * b + c * d) := by
  sorry

end NUMINAMATH_CALUDE_not_prime_sum_product_l1418_141881


namespace NUMINAMATH_CALUDE_function_critical_points_and_inequality_l1418_141829

open Real

noncomputable def f (a x : ℝ) : ℝ := (x - 2) * exp x - a * x^2 + 2 * a * x - 2 * a

theorem function_critical_points_and_inequality (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧
    (∀ x : ℝ, f a x = 0 → x = x₁ ∨ x = x₂) ∧
    (∀ x : ℝ, 0 < x → x < x₂ → f a x < -2 * a)) →
  a = exp 1 / 4 ∨ a = 2 * exp 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_function_critical_points_and_inequality_l1418_141829


namespace NUMINAMATH_CALUDE_log_equation_root_range_l1418_141822

theorem log_equation_root_range (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 1 < x ∧ x < 3 ∧ 1 < y ∧ y < 3 ∧ 
   Real.log (x - 1) + Real.log (3 - x) = Real.log (a - x) ∧
   Real.log (y - 1) + Real.log (3 - y) = Real.log (a - y)) →
  (3 < a ∧ a < 13/4) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_root_range_l1418_141822


namespace NUMINAMATH_CALUDE_f_nonpositive_implies_k_geq_one_l1418_141804

open Real

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := log x - k * x + 1

theorem f_nonpositive_implies_k_geq_one (k : ℝ) :
  (∀ x > 0, f k x ≤ 0) → k ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_f_nonpositive_implies_k_geq_one_l1418_141804


namespace NUMINAMATH_CALUDE_trolley_problem_l1418_141844

/-- Trolley problem theorem -/
theorem trolley_problem (initial_passengers : ℕ) 
  (second_stop_off : ℕ) (second_stop_on_multiplier : ℕ)
  (third_stop_off : ℕ) (final_passengers : ℕ) :
  initial_passengers = 10 →
  second_stop_off = 3 →
  second_stop_on_multiplier = 2 →
  third_stop_off = 18 →
  final_passengers = 12 →
  (initial_passengers - second_stop_off + second_stop_on_multiplier * initial_passengers - third_stop_off) + 3 = final_passengers :=
by
  sorry

#check trolley_problem

end NUMINAMATH_CALUDE_trolley_problem_l1418_141844


namespace NUMINAMATH_CALUDE_books_left_over_l1418_141847

/-- Calculates the number of books left over after filling a bookcase -/
theorem books_left_over
  (initial_books : ℕ)
  (shelves : ℕ)
  (books_per_shelf : ℕ)
  (new_books : ℕ)
  (h1 : initial_books = 56)
  (h2 : shelves = 4)
  (h3 : books_per_shelf = 20)
  (h4 : new_books = 26) :
  initial_books + new_books - (shelves * books_per_shelf) = 2 :=
by
  sorry

#check books_left_over

end NUMINAMATH_CALUDE_books_left_over_l1418_141847


namespace NUMINAMATH_CALUDE_bowtie_equation_l1418_141821

-- Define the ⋈ operation
noncomputable def bowtie (a b : ℝ) : ℝ :=
  a + Real.sqrt (b + 3 + Real.sqrt (b + 3 + Real.sqrt (b + 3 + Real.sqrt (b + 3))))

-- State the theorem
theorem bowtie_equation (g : ℝ) : bowtie 8 g = 11 → g = 3 := by
  sorry

end NUMINAMATH_CALUDE_bowtie_equation_l1418_141821


namespace NUMINAMATH_CALUDE_product_price_interval_l1418_141888

theorem product_price_interval (price : ℝ) 
  (h1 : price < 2000)
  (h2 : price > 1000)
  (h3 : price < 1500)
  (h4 : price > 1250)
  (h5 : price > 1375) :
  price ∈ Set.Ioo 1375 1500 := by
sorry

end NUMINAMATH_CALUDE_product_price_interval_l1418_141888


namespace NUMINAMATH_CALUDE_chessboard_rearrangement_3x3_no_chessboard_rearrangement_8x8_l1418_141880

/-- Represents a chessboard of size N × N -/
structure Chessboard (N : ℕ) where
  size : ℕ
  size_eq : size = N

/-- Represents a position on the chessboard -/
structure Position (N : ℕ) where
  row : Fin N
  col : Fin N

/-- Knight's move distance between two positions -/
def knightDistance (N : ℕ) (p1 p2 : Position N) : ℕ :=
  sorry

/-- King's move distance between two positions -/
def kingDistance (N : ℕ) (p1 p2 : Position N) : ℕ :=
  sorry

/-- Represents a rearrangement of checkers on the board -/
def Rearrangement (N : ℕ) := Position N → Position N

/-- Checks if a rearrangement satisfies the problem condition -/
def isValidRearrangement (N : ℕ) (r : Rearrangement N) : Prop :=
  ∀ p1 p2 : Position N, knightDistance N p1 p2 = 1 → kingDistance N (r p1) (r p2) = 1

theorem chessboard_rearrangement_3x3 :
  ∃ (r : Rearrangement 3), isValidRearrangement 3 r :=
sorry

theorem no_chessboard_rearrangement_8x8 :
  ¬ ∃ (r : Rearrangement 8), isValidRearrangement 8 r :=
sorry

end NUMINAMATH_CALUDE_chessboard_rearrangement_3x3_no_chessboard_rearrangement_8x8_l1418_141880


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_2_l1418_141819

def motion_equation (t : ℝ) : ℝ := 2 * t^2 + 3

theorem instantaneous_velocity_at_2 :
  (deriv motion_equation) 2 = 8 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_2_l1418_141819


namespace NUMINAMATH_CALUDE_angle_A_is_120_degrees_l1418_141806

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the condition given in the problem
def satisfiesCondition (t : Triangle) : Prop :=
  2 * t.a * Real.sin t.A = (2 * t.b + t.c) * Real.sin t.B + (2 * t.c + t.b) * Real.sin t.C

-- Theorem statement
theorem angle_A_is_120_degrees (t : Triangle) 
  (h : satisfiesCondition t) : t.A = 2 * π / 3 := by
  sorry


end NUMINAMATH_CALUDE_angle_A_is_120_degrees_l1418_141806


namespace NUMINAMATH_CALUDE_sin_cos_sum_11_19_l1418_141863

theorem sin_cos_sum_11_19 : 
  Real.sin (11 * π / 180) * Real.cos (19 * π / 180) + 
  Real.cos (11 * π / 180) * Real.sin (19 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_11_19_l1418_141863


namespace NUMINAMATH_CALUDE_union_equality_implies_a_value_l1418_141810

def A (a : ℝ) : Set ℝ := {2*a, 3}
def B : Set ℝ := {2, 3}

theorem union_equality_implies_a_value (a : ℝ) :
  A a ∪ B = {2, 3, 4} → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_union_equality_implies_a_value_l1418_141810


namespace NUMINAMATH_CALUDE_b_completion_time_l1418_141833

/- Define the work completion time for A -/
def a_completion_time : ℝ := 9

/- Define B's efficiency relative to A -/
def b_efficiency_factor : ℝ := 1.5

/- Theorem statement -/
theorem b_completion_time :
  let a_rate := 1 / a_completion_time
  let b_rate := b_efficiency_factor * a_rate
  (1 / b_rate) = 6 := by sorry

end NUMINAMATH_CALUDE_b_completion_time_l1418_141833


namespace NUMINAMATH_CALUDE_car_speed_comparison_l1418_141835

/-- Proves that the average speed of Car A is less than or equal to the average speed of Car B -/
theorem car_speed_comparison
  (u v : ℝ) -- speeds in miles per hour
  (hu : u > 0) (hv : v > 0) -- speeds are positive
  (x : ℝ) -- average speed of Car A
  (hx : x = 3 / (1 / u + 2 / v)) -- definition of x
  (y : ℝ) -- average speed of Car B
  (hy : y = (u + 2 * v) / 3) -- definition of y
  : x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_car_speed_comparison_l1418_141835


namespace NUMINAMATH_CALUDE_camping_trip_percentage_l1418_141823

theorem camping_trip_percentage (total_students : ℕ) 
  (h1 : (16 : ℝ) / 100 * total_students = (25 : ℝ) / 100 * (64 : ℝ) / 100 * total_students)
  (h2 : (75 : ℝ) / 100 * (64 : ℝ) / 100 * total_students = (64 : ℝ) / 100 * total_students - (16 : ℝ) / 100 * total_students) :
  (64 : ℝ) / 100 * total_students = (64 : ℝ) / 100 * total_students :=
by sorry

end NUMINAMATH_CALUDE_camping_trip_percentage_l1418_141823


namespace NUMINAMATH_CALUDE_construct_75_degree_angle_l1418_141896

/-- Given an angle of 19°, it is possible to construct an angle of 75°. -/
theorem construct_75_degree_angle (angle : ℝ) (h : angle = 19) : 
  ∃ (constructed_angle : ℝ), constructed_angle = 75 := by
sorry

end NUMINAMATH_CALUDE_construct_75_degree_angle_l1418_141896


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l1418_141868

theorem quadratic_form_sum (x : ℝ) : ∃ (a h k : ℝ), 
  (6 * x^2 + 12 * x + 8 = a * (x - h)^2 + k) ∧ (a + h + k = 9) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l1418_141868


namespace NUMINAMATH_CALUDE_barkley_bones_theorem_l1418_141867

/-- Calculates the number of bones Barkley has available after a given number of months -/
def bones_available (bones_per_month : ℕ) (months : ℕ) (buried_bones : ℕ) : ℕ :=
  bones_per_month * months - buried_bones

/-- Theorem: Barkley has 8 bones available after 5 months -/
theorem barkley_bones_theorem :
  bones_available 10 5 42 = 8 := by
  sorry

end NUMINAMATH_CALUDE_barkley_bones_theorem_l1418_141867


namespace NUMINAMATH_CALUDE_f_919_equals_6_l1418_141854

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period_six (f : ℝ → ℝ) : Prop := ∀ x, f (x + 6) = f x

theorem f_919_equals_6 (f : ℝ → ℝ) 
  (h1 : is_even f)
  (h2 : ∀ x, f (x + 4) = f (x - 2))
  (h3 : ∀ x ∈ Set.Icc (-3) 0, f x = (6 : ℝ) ^ (-x)) :
  f 919 = 6 := by
  sorry

end NUMINAMATH_CALUDE_f_919_equals_6_l1418_141854


namespace NUMINAMATH_CALUDE_no_real_solutions_condition_l1418_141856

theorem no_real_solutions_condition (a : ℝ) : 
  (∀ x : ℝ, (a^2 + 2*a)*x^2 + 3*a*x + 1 ≠ 0) ↔ (0 < a ∧ a < 8/5) :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_condition_l1418_141856


namespace NUMINAMATH_CALUDE_remaining_area_in_square_configuration_l1418_141825

/-- The area of the remaining space in a square configuration -/
theorem remaining_area_in_square_configuration : 
  ∀ (s : ℝ) (small_square : ℝ) (rect1_length rect1_width : ℝ) (rect2_length rect2_width : ℝ),
  s = 4 →
  small_square = 1 →
  rect1_length = 2 ∧ rect1_width = 1 →
  rect2_length = 1 ∧ rect2_width = 2 →
  s^2 - (small_square^2 + rect1_length * rect1_width + rect2_length * rect2_width) = 11 :=
by sorry

end NUMINAMATH_CALUDE_remaining_area_in_square_configuration_l1418_141825


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1418_141877

-- Define the functions f and g
def f (k : ℝ) (x : ℝ) : ℝ := 8 * x^2 + 16 * x - k
def g (x : ℝ) : ℝ := 2 * x^3 + 5 * x^2 + 4 * x

-- Define the interval [-3, 3]
def I : Set ℝ := Set.Icc (-3) 3

-- Theorem statements
theorem problem_1 (k : ℝ) : 
  (∀ x ∈ I, f k x ≤ g x) ↔ k ≥ 45 :=
sorry

theorem problem_2 (k : ℝ) :
  (∃ x ∈ I, f k x ≤ g x) ↔ k ≥ -7 :=
sorry

theorem problem_3 (k : ℝ) :
  (∀ x1 ∈ I, ∀ x2 ∈ I, f k x1 ≤ g x2) ↔ k ≥ 141 :=
sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1418_141877
