import Mathlib

namespace NUMINAMATH_CALUDE_f_roots_and_monotonicity_imply_b_range_l2956_295658

/-- The function f(x) = -x^3 + bx -/
def f (b : ℝ) (x : ℝ) : ℝ := -x^3 + b*x

/-- Theorem: If all roots of f(x) = 0 are within [-2, 2] and f(x) is monotonically increasing in (0, 1), then 3 ≤ b ≤ 4 -/
theorem f_roots_and_monotonicity_imply_b_range (b : ℝ) :
  (∀ x, f b x = 0 → x ∈ Set.Icc (-2) 2) →
  (∀ x ∈ Set.Ioo 0 1, ∀ y ∈ Set.Ioo 0 1, x < y → f b x < f b y) →
  b ∈ Set.Icc 3 4 := by
  sorry

end NUMINAMATH_CALUDE_f_roots_and_monotonicity_imply_b_range_l2956_295658


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2956_295671

def A : Set ℕ := {2, 4, 6}
def B : Set ℕ := {1, 3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2956_295671


namespace NUMINAMATH_CALUDE_complex_product_magnitude_l2956_295649

theorem complex_product_magnitude (a b : ℂ) (t : ℝ) :
  Complex.abs a = 3 →
  Complex.abs b = 5 →
  a * b = t - 3 * Complex.I →
  t = 6 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_complex_product_magnitude_l2956_295649


namespace NUMINAMATH_CALUDE_sum_of_digits_M_l2956_295637

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Predicate for numbers using only specified digits -/
def uses_specified_digits (n : ℕ) : Prop := sorry

theorem sum_of_digits_M (M : ℕ) 
  (h_even : Even M)
  (h_digits : uses_specified_digits M)
  (h_double : sum_of_digits (2 * M) = 39)
  (h_half : sum_of_digits (M / 2) = 30) :
  sum_of_digits M = 33 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_M_l2956_295637


namespace NUMINAMATH_CALUDE_figure_214_is_triangle_l2956_295611

/-- The type representing figures in the sequence -/
inductive Figure
| triangle : Figure
| square : Figure
| circle : Figure

/-- The function that returns the figure at a given position in the sequence -/
def sequenceFigure (n : ℕ) : Figure :=
  match n % 5 with
  | 0 => Figure.circle
  | 1 => Figure.triangle
  | 2 => Figure.square
  | 3 => Figure.triangle
  | _ => Figure.circle

/-- Theorem stating that the 214th figure in the sequence is a triangle -/
theorem figure_214_is_triangle : sequenceFigure 214 = Figure.triangle := by
  sorry

end NUMINAMATH_CALUDE_figure_214_is_triangle_l2956_295611


namespace NUMINAMATH_CALUDE_bull_count_l2956_295602

theorem bull_count (total_cattle : ℕ) (cow_ratio bull_ratio : ℕ) 
  (h1 : total_cattle = 555)
  (h2 : cow_ratio = 10)
  (h3 : bull_ratio = 27) : 
  (bull_ratio : ℚ) / (cow_ratio + bull_ratio : ℚ) * total_cattle = 405 :=
by sorry

end NUMINAMATH_CALUDE_bull_count_l2956_295602


namespace NUMINAMATH_CALUDE_ellipse_minor_axis_length_l2956_295694

/-- An ellipse with a focus and distances to vertices -/
structure Ellipse :=
  (F : ℝ × ℝ)  -- Focus
  (d1 : ℝ)     -- Distance from focus to first vertex
  (d2 : ℝ)     -- Distance from focus to second vertex

/-- The length of the minor axis of the ellipse -/
def minorAxisLength (E : Ellipse) : ℝ := sorry

/-- Theorem: If the distances from the focus to the vertices are 1 and 9, 
    then the minor axis length is 6 -/
theorem ellipse_minor_axis_length 
  (E : Ellipse) 
  (h1 : E.d1 = 1) 
  (h2 : E.d2 = 9) : 
  minorAxisLength E = 6 := by sorry

end NUMINAMATH_CALUDE_ellipse_minor_axis_length_l2956_295694


namespace NUMINAMATH_CALUDE_second_camp_selection_l2956_295626

def systematic_sample (start : Nat) (step : Nat) (total : Nat) : List Nat :=
  List.range (total / step + 1) |> List.map (fun i => start + i * step)

def count_in_range (list : List Nat) (lower : Nat) (upper : Nat) : Nat :=
  list.filter (fun x => lower ≤ x ∧ x ≤ upper) |> List.length

theorem second_camp_selection :
  let sample := systematic_sample 3 5 100
  let second_camp_count := count_in_range sample 16 55
  second_camp_count = 8 := by
  sorry

end NUMINAMATH_CALUDE_second_camp_selection_l2956_295626


namespace NUMINAMATH_CALUDE_revenue_increase_80_percent_l2956_295603

/-- Represents the change in revenue given a price decrease and sales increase --/
def revenue_change (price_decrease : ℝ) (sales_increase_ratio : ℝ) : ℝ :=
  let new_price_factor := 1 - price_decrease
  let sales_increase := price_decrease * sales_increase_ratio
  let new_quantity_factor := 1 + sales_increase
  new_price_factor * new_quantity_factor - 1

/-- 
Theorem: Given a 10% price decrease and a sales increase ratio of 10,
the total revenue will increase by 80%
-/
theorem revenue_increase_80_percent :
  revenue_change 0.1 10 = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_revenue_increase_80_percent_l2956_295603


namespace NUMINAMATH_CALUDE_jerry_has_36_stickers_l2956_295627

/-- The number of stickers each person has -/
structure StickerCount where
  fred : ℕ
  george : ℕ
  jerry : ℕ

/-- The conditions of the problem -/
def sticker_problem (s : StickerCount) : Prop :=
  s.fred = 18 ∧
  s.george = s.fred - 6 ∧
  s.jerry = 3 * s.george

/-- The theorem stating Jerry has 36 stickers -/
theorem jerry_has_36_stickers (s : StickerCount) (h : sticker_problem s) : s.jerry = 36 := by
  sorry

end NUMINAMATH_CALUDE_jerry_has_36_stickers_l2956_295627


namespace NUMINAMATH_CALUDE_min_values_ab_and_fraction_l2956_295608

theorem min_values_ab_and_fraction (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h : 1/a + 3/b = 1) : 
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 1/x + 3/y = 1 ∧ x*y ≤ a*b) ∧
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 1/x + 3/y = 1 ∧ x*y = 12) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → 1/x + 3/y = 1 → x*y ≥ 12) ∧
  (∃ (x y : ℝ), x > 1 ∧ y > 3 ∧ 1/x + 3/y = 1 ∧ 1/(x-1) + 3/(y-3) ≤ 1/(a-1) + 3/(b-3)) ∧
  (∃ (x y : ℝ), x > 1 ∧ y > 3 ∧ 1/x + 3/y = 1 ∧ 1/(x-1) + 3/(y-3) = 2) ∧
  (∀ (x y : ℝ), x > 1 → y > 3 → 1/x + 3/y = 1 → 1/(x-1) + 3/(y-3) ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_min_values_ab_and_fraction_l2956_295608


namespace NUMINAMATH_CALUDE_graphs_intersect_once_l2956_295633

/-- The value of b for which the graphs of y = bx^2 + 5x + 2 and y = -2x - 3 intersect at exactly one point -/
def b : ℚ := 49 / 20

/-- The quadratic function representing the first graph -/
def f (x : ℝ) : ℝ := b * x^2 + 5 * x + 2

/-- The linear function representing the second graph -/
def g (x : ℝ) : ℝ := -2 * x - 3

/-- Theorem stating that the graphs intersect at exactly one point when b = 49/20 -/
theorem graphs_intersect_once :
  ∃! x : ℝ, f x = g x :=
sorry

end NUMINAMATH_CALUDE_graphs_intersect_once_l2956_295633


namespace NUMINAMATH_CALUDE_trays_needed_to_refill_l2956_295685

/-- The number of ice cubes Dylan used in his glass -/
def dylan_glass_cubes : ℕ := 8

/-- The number of ice cubes used per glass for lemonade -/
def lemonade_glass_cubes : ℕ := 2 * dylan_glass_cubes

/-- The total number of glasses served (including Dylan's) -/
def total_glasses : ℕ := 5 + 1

/-- The number of spaces in each ice cube tray -/
def tray_spaces : ℕ := 14

/-- The fraction of total ice cubes used -/
def fraction_used : ℚ := 4/5

/-- The total number of ice cubes used -/
def total_used : ℕ := dylan_glass_cubes + lemonade_glass_cubes * total_glasses

/-- The initial total number of ice cubes -/
def initial_total : ℚ := (total_used : ℚ) / fraction_used

theorem trays_needed_to_refill : 
  ⌈initial_total / tray_spaces⌉ = 10 := by sorry

end NUMINAMATH_CALUDE_trays_needed_to_refill_l2956_295685


namespace NUMINAMATH_CALUDE_count_distinct_tetrahedrons_is_423_l2956_295675

/-- Represents a regular tetrahedron with its vertices and edge midpoints -/
structure RegularTetrahedron :=
  (vertices : Finset (Fin 4))
  (edge_midpoints : Finset (Fin 6))

/-- Represents a new tetrahedron formed from points of a regular tetrahedron -/
def NewTetrahedron (t : RegularTetrahedron) := Finset (Fin 4)

/-- Counts the number of distinct new tetrahedrons that can be formed -/
def count_distinct_tetrahedrons (t : RegularTetrahedron) : ℕ :=
  sorry

/-- The main theorem stating that the number of distinct tetrahedrons is 423 -/
theorem count_distinct_tetrahedrons_is_423 (t : RegularTetrahedron) :
  count_distinct_tetrahedrons t = 423 :=
sorry

end NUMINAMATH_CALUDE_count_distinct_tetrahedrons_is_423_l2956_295675


namespace NUMINAMATH_CALUDE_trig_inequality_l2956_295647

theorem trig_inequality (θ : Real) (h1 : 0 < θ) (h2 : θ < Real.pi / 4) :
  Real.sin θ ^ 2 < Real.cos θ ^ 2 ∧ Real.cos θ ^ 2 < (Real.cos θ / Real.sin θ) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_inequality_l2956_295647


namespace NUMINAMATH_CALUDE_count_valid_numbers_l2956_295628

/-- The number of ways to choose 2 items from 3 -/
def choose_3_2 : ℕ := 3

/-- The number of ways to arrange 2 items -/
def arrange_2 : ℕ := 2

/-- The number of ways to insert 2 items into 3 gaps -/
def insert_2_into_3 : ℕ := 6

/-- The number of valid five-digit numbers -/
def valid_numbers : ℕ := choose_3_2 * arrange_2 * arrange_2 * insert_2_into_3

theorem count_valid_numbers : valid_numbers = 72 := by
  sorry

end NUMINAMATH_CALUDE_count_valid_numbers_l2956_295628


namespace NUMINAMATH_CALUDE_ninth_term_of_arithmetic_sequence_l2956_295682

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem states that for an arithmetic sequence with the given properties, the ninth term is 35. -/
theorem ninth_term_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_third_term : a 3 = 23)
  (h_sixth_term : a 6 = 29) :
  a 9 = 35 := by
  sorry

end NUMINAMATH_CALUDE_ninth_term_of_arithmetic_sequence_l2956_295682


namespace NUMINAMATH_CALUDE_apple_quantity_proof_l2956_295676

/-- Calculates the final quantity of apples given initial quantity, sold quantity, and purchased quantity. -/
def final_quantity (initial : ℕ) (sold : ℕ) (purchased : ℕ) : ℕ :=
  initial - sold + purchased

/-- Theorem stating that given the specific quantities in the problem, the final quantity is 293 kg. -/
theorem apple_quantity_proof :
  final_quantity 280 132 145 = 293 := by
  sorry

end NUMINAMATH_CALUDE_apple_quantity_proof_l2956_295676


namespace NUMINAMATH_CALUDE_max_value_of_a_l2956_295660

theorem max_value_of_a (x y : ℝ) (hx : x > 1/3) (hy : y > 1) :
  (∀ a : ℝ, (9 * x^2) / (a^2 * (y - 1)) + (y^2) / (a^2 * (3 * x - 1)) ≥ 1) →
  (∃ a_max : ℝ, a_max = 2 * Real.sqrt 2 ∧
    ∀ a : ℝ, (9 * x^2) / (a^2 * (y - 1)) + (y^2) / (a^2 * (3 * x - 1)) ≥ 1 → a ≤ a_max) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l2956_295660


namespace NUMINAMATH_CALUDE_exists_22_same_age_l2956_295666

/-- Represents a villager in Roche -/
structure Villager where
  age : ℕ

/-- The village of Roche -/
structure Village where
  inhabitants : Finset Villager
  total_count : inhabitants.card = 2020
  knows_same_age : ∀ v ∈ inhabitants, ∃ w ∈ inhabitants, v ≠ w ∧ v.age = w.age
  three_same_age_in_192 : ∀ (group : Finset Villager), group ⊆ inhabitants → group.card = 192 →
    ∃ (a : ℕ) (v₁ v₂ v₃ : Villager), v₁ ∈ group ∧ v₂ ∈ group ∧ v₃ ∈ group ∧
      v₁ ≠ v₂ ∧ v₁ ≠ v₃ ∧ v₂ ≠ v₃ ∧ v₁.age = a ∧ v₂.age = a ∧ v₃.age = a

/-- There exists a group of at least 22 villagers of the same age in Roche -/
theorem exists_22_same_age (roche : Village) : 
  ∃ (a : ℕ) (group : Finset Villager), group ⊆ roche.inhabitants ∧ group.card ≥ 22 ∧
    ∀ v ∈ group, v.age = a :=
sorry

end NUMINAMATH_CALUDE_exists_22_same_age_l2956_295666


namespace NUMINAMATH_CALUDE_dress_design_combinations_l2956_295696

theorem dress_design_combinations (colors patterns : ℕ) (h1 : colors = 5) (h2 : patterns = 6) : colors * patterns = 30 := by
  sorry

end NUMINAMATH_CALUDE_dress_design_combinations_l2956_295696


namespace NUMINAMATH_CALUDE_marathon_average_time_l2956_295604

/-- Calculates the average time per mile for a marathon --/
def average_time_per_mile (distance : ℕ) (hours : ℕ) (minutes : ℕ) : ℚ :=
  (hours * 60 + minutes : ℚ) / distance

/-- Theorem: The average time per mile for a 24-mile marathon completed in 3 hours and 36 minutes is 9 minutes --/
theorem marathon_average_time :
  average_time_per_mile 24 3 36 = 9 := by
  sorry

end NUMINAMATH_CALUDE_marathon_average_time_l2956_295604


namespace NUMINAMATH_CALUDE_symmetric_lines_coefficient_sum_l2956_295691

-- Define the two lines
def line1 (x y : ℝ) : Prop := x + 2*y - 3 = 0
def line2 (a b x y : ℝ) : Prop := a*x + 4*y + b = 0

-- Define the point A
def point_A : ℝ × ℝ := (1, 0)

-- Define symmetry with respect to a point
def symmetric_wrt (p : ℝ × ℝ) (l1 l2 : (ℝ → ℝ → Prop)) : Prop :=
  ∀ (x y : ℝ), l1 x y ↔ l2 (2*p.1 - x) (2*p.2 - y)

-- Theorem statement
theorem symmetric_lines_coefficient_sum (a b : ℝ) :
  symmetric_wrt point_A (line1) (line2 a b) →
  a + b = 0 :=
sorry

end NUMINAMATH_CALUDE_symmetric_lines_coefficient_sum_l2956_295691


namespace NUMINAMATH_CALUDE_car_trip_duration_l2956_295629

theorem car_trip_duration (initial_speed initial_time remaining_speed average_speed : ℝ) 
  (h1 : initial_speed = 70)
  (h2 : initial_time = 4)
  (h3 : remaining_speed = 60)
  (h4 : average_speed = 65) :
  ∃ (total_time : ℝ), 
    (initial_speed * initial_time + remaining_speed * (total_time - initial_time)) / total_time = average_speed ∧ 
    total_time = 8 := by
sorry

end NUMINAMATH_CALUDE_car_trip_duration_l2956_295629


namespace NUMINAMATH_CALUDE_calculate_matches_played_rahul_matches_played_l2956_295646

/-- 
Given a cricketer's current batting average and the change in average after scoring in an additional match,
this theorem calculates the number of matches played before the additional match.
-/
theorem calculate_matches_played (current_average : ℚ) (additional_runs : ℕ) (new_average : ℚ) : ℕ :=
  let m := (additional_runs - new_average) / (new_average - current_average)
  m.num.toNat

/--
Proves that Rahul has played 5 matches given his current batting average and the change after an additional match.
-/
theorem rahul_matches_played : calculate_matches_played 51 69 54 = 5 := by
  sorry

end NUMINAMATH_CALUDE_calculate_matches_played_rahul_matches_played_l2956_295646


namespace NUMINAMATH_CALUDE_function_F_property_l2956_295638

-- Define the function F
noncomputable def F : ℝ → ℝ := sorry

-- State the theorem
theorem function_F_property (x : ℝ) : 
  (F ((1 - x) / (1 + x)) = x) → 
  (F (-2 - x) = -2 - F x) := by sorry

end NUMINAMATH_CALUDE_function_F_property_l2956_295638


namespace NUMINAMATH_CALUDE_rod_triangle_impossibility_l2956_295659

theorem rod_triangle_impossibility (L : ℝ) (a b : ℝ) 
  (h1 : L > 0) 
  (h2 : a > 0) 
  (h3 : b > 0) 
  (h4 : a + b = L / 2) : 
  ¬(L / 2 + a > b ∧ L / 2 + b > a ∧ a + b > L / 2) := by
  sorry


end NUMINAMATH_CALUDE_rod_triangle_impossibility_l2956_295659


namespace NUMINAMATH_CALUDE_quadratic_decreasing_interval_l2956_295695

def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_decreasing_interval 
  (a b c : ℝ) 
  (h1 : a > 0) 
  (h2 : quadratic_function a b c (-5) = 0) 
  (h3 : quadratic_function a b c 3 = 0) :
  ∀ x ∈ Set.Iic (-1), 
    ∀ y ∈ Set.Iic (-1), 
      x < y → quadratic_function a b c x > quadratic_function a b c y :=
by sorry

end NUMINAMATH_CALUDE_quadratic_decreasing_interval_l2956_295695


namespace NUMINAMATH_CALUDE_log_equality_implies_y_value_l2956_295648

-- Define the logarithm function (base 10)
noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the theorem
theorem log_equality_implies_y_value 
  (a b c x : ℝ) 
  (p q r y : ℝ) 
  (h1 : log a / p = log b / q)
  (h2 : log b / q = log c / r)
  (h3 : log c / r = log x)
  (h4 : x ≠ 1)
  (h5 : b^3 / (a^2 * c) = x^y) :
  y = 3*q - 2*p - r := by
  sorry

#check log_equality_implies_y_value

end NUMINAMATH_CALUDE_log_equality_implies_y_value_l2956_295648


namespace NUMINAMATH_CALUDE_concentric_circles_ratio_l2956_295651

theorem concentric_circles_ratio (a b : ℝ) (h : a > 0) (k : b > 0) 
  (h_area : π * b^2 - π * a^2 = 4 * (π * a^2)) : 
  a / b = Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_concentric_circles_ratio_l2956_295651


namespace NUMINAMATH_CALUDE_product_difference_square_l2956_295606

theorem product_difference_square (n : ℤ) : (n - 1) * (n + 1) - n^2 = -1 :=
by sorry

end NUMINAMATH_CALUDE_product_difference_square_l2956_295606


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2956_295616

def original_proposition (x : ℝ) : Prop := x = 1 → x^2 - 3*x + 2 = 0

theorem contrapositive_equivalence :
  (∀ x : ℝ, x^2 - 3*x + 2 ≠ 0 → x ≠ 1) ↔ 
  (∀ x : ℝ, original_proposition x) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2956_295616


namespace NUMINAMATH_CALUDE_max_negative_coefficients_no_real_roots_l2956_295632

def polynomial_coefficients (p : ℝ → ℝ) : List ℤ :=
  sorry

def has_no_real_roots (p : ℝ → ℝ) : Prop :=
  sorry

def count_negative_coefficients (coeffs : List ℤ) : ℕ :=
  sorry

theorem max_negative_coefficients_no_real_roots 
  (p : ℝ → ℝ) 
  (h1 : ∃ (coeffs : List ℤ), polynomial_coefficients p = coeffs ∧ coeffs.length = 2011)
  (h2 : has_no_real_roots p) :
  count_negative_coefficients (polynomial_coefficients p) ≤ 1005 :=
sorry

end NUMINAMATH_CALUDE_max_negative_coefficients_no_real_roots_l2956_295632


namespace NUMINAMATH_CALUDE_special_function_properties_l2956_295686

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) = f x + f y) ∧ 
  (f (1/3) = 1) ∧
  (∀ x : ℝ, x > 0 → f x > 0)

theorem special_function_properties (f : ℝ → ℝ) (h : special_function f) :
  (f 0 = 0) ∧
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x : ℝ, x < -2/3 → f x + f (2 + x) < 2) :=
by sorry

end NUMINAMATH_CALUDE_special_function_properties_l2956_295686


namespace NUMINAMATH_CALUDE_inequality_proof_l2956_295663

theorem inequality_proof (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x^2 + y^2 + 1 = (x*y - 1)^2) : 
  (x + y ≥ 4) ∧ (x^2 + y^2 ≥ 8) ∧ (x + 4*y ≥ 9) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2956_295663


namespace NUMINAMATH_CALUDE_sequence_sum_problem_l2956_295636

theorem sequence_sum_problem (N : ℤ) : 
  (995 : ℤ) + 997 + 999 + 1001 + 1003 = 5005 - N → N = 5 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_problem_l2956_295636


namespace NUMINAMATH_CALUDE_arithmetic_contains_geometric_l2956_295630

/-- Given positive integers a and d, there exist positive integers b and q such that 
    the geometric progression b, bq, bq^2, ... is a subset of the arithmetic progression a, a+d, a+2d, ... -/
theorem arithmetic_contains_geometric (a d : ℕ+) : 
  ∃ (b q : ℕ+), ∀ (n : ℕ), ∃ (k : ℕ), b * q ^ n = a + k * d := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_contains_geometric_l2956_295630


namespace NUMINAMATH_CALUDE_equation_solution_l2956_295698

theorem equation_solution : 
  let f : ℝ → ℝ := λ x => 1/((x - 1)*(x - 2)) + 1/((x - 2)*(x - 3)) + 
                         1/((x - 3)*(x - 4)) + 1/((x - 4)*(x - 5))
  ∀ x : ℝ, f x = 1/10 ↔ x = 9 ∨ x = 5 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2956_295698


namespace NUMINAMATH_CALUDE_termite_ridden_collapsing_homes_l2956_295667

theorem termite_ridden_collapsing_homes 
  (total_homes : ℕ) 
  (termite_ridden : ℚ) 
  (termite_not_collapsing : ℚ) 
  (h1 : termite_ridden = 1/3) 
  (h2 : termite_not_collapsing = 1/7) : 
  (termite_ridden - termite_not_collapsing) / termite_ridden = 4/21 := by
sorry

end NUMINAMATH_CALUDE_termite_ridden_collapsing_homes_l2956_295667


namespace NUMINAMATH_CALUDE_beryllium_hydroxide_formation_l2956_295642

/-- Represents a chemical species in a reaction -/
structure ChemicalSpecies where
  formula : String
  moles : ℚ

/-- Represents a chemical reaction with reactants and products -/
structure ChemicalReaction where
  reactants : List ChemicalSpecies
  products : List ChemicalSpecies

/-- The balanced chemical equation for the reaction of beryllium carbide with water -/
def berylliumCarbideReaction : ChemicalReaction :=
  { reactants := [
      { formula := "Be2C", moles := 1 },
      { formula := "H2O", moles := 4 }
    ],
    products := [
      { formula := "Be(OH)2", moles := 2 },
      { formula := "CH4", moles := 1 }
    ]
  }

/-- Given 1 mole of Be2C and 4 moles of H2O, 2 moles of Be(OH)2 are formed -/
theorem beryllium_hydroxide_formation :
  ∀ (reaction : ChemicalReaction),
    reaction = berylliumCarbideReaction →
    ∃ (product : ChemicalSpecies),
      product ∈ reaction.products ∧
      product.formula = "Be(OH)2" ∧
      product.moles = 2 :=
by sorry

end NUMINAMATH_CALUDE_beryllium_hydroxide_formation_l2956_295642


namespace NUMINAMATH_CALUDE_janice_starting_sentences_janice_starting_sentences_proof_l2956_295645

/-- Proves the number of sentences Janice started with today -/
theorem janice_starting_sentences : ℕ :=
  let typing_speed : ℕ := 6  -- sentences per minute
  let first_session : ℕ := 20  -- minutes
  let second_session : ℕ := 15  -- minutes
  let third_session : ℕ := 18  -- minutes
  let erased_sentences : ℕ := 40
  let total_sentences : ℕ := 536

  let total_typed : ℕ := typing_speed * (first_session + second_session + third_session)
  let net_added : ℕ := total_typed - erased_sentences
  
  total_sentences - net_added

/-- The theorem statement -/
theorem janice_starting_sentences_proof : janice_starting_sentences = 258 := by
  sorry

end NUMINAMATH_CALUDE_janice_starting_sentences_janice_starting_sentences_proof_l2956_295645


namespace NUMINAMATH_CALUDE_investment_in_bank_a_l2956_295656

def total_investment : ℝ := 1500
def bank_a_rate : ℝ := 0.04
def bank_b_rate : ℝ := 0.06
def years : ℕ := 3
def final_amount : ℝ := 1740.54

theorem investment_in_bank_a (x : ℝ) :
  x * (1 + bank_a_rate) ^ years + (total_investment - x) * (1 + bank_b_rate) ^ years = final_amount →
  x = 695 := by
sorry

end NUMINAMATH_CALUDE_investment_in_bank_a_l2956_295656


namespace NUMINAMATH_CALUDE_no_intersection_points_l2956_295650

/-- Parabola 1 defined by y = 2x^2 + 3x - 4 -/
def parabola1 (x : ℝ) : ℝ := 2 * x^2 + 3 * x - 4

/-- Parabola 2 defined by y = 3x^2 + 12 -/
def parabola2 (x : ℝ) : ℝ := 3 * x^2 + 12

/-- Theorem stating that the two parabolas have no real intersection points -/
theorem no_intersection_points : ∀ x : ℝ, parabola1 x ≠ parabola2 x := by
  sorry

end NUMINAMATH_CALUDE_no_intersection_points_l2956_295650


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2956_295680

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ≥ 1) ↔ (∃ x : ℝ, x < 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2956_295680


namespace NUMINAMATH_CALUDE_symmetry_axis_l2956_295655

-- Define a function g with the given symmetry property
def g : ℝ → ℝ := sorry

-- State the symmetry property of g
axiom g_symmetry : ∀ x : ℝ, g x = g (3 - x)

-- Define what it means for a vertical line to be an axis of symmetry
def is_axis_of_symmetry (a : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (a + x) = f (a - x)

-- Theorem statement
theorem symmetry_axis :
  is_axis_of_symmetry (3/2) g := by sorry

end NUMINAMATH_CALUDE_symmetry_axis_l2956_295655


namespace NUMINAMATH_CALUDE_jakes_earnings_theorem_l2956_295643

/-- Calculates Jake's weekly earnings based on Jacob's hourly rates and Jake's work schedule. -/
def jakes_weekly_earnings (jacobs_weekday_rate : ℕ) (jacobs_weekend_rate : ℕ) 
  (jakes_weekday_hours : ℕ) (jakes_weekend_hours : ℕ) : ℕ :=
  let jakes_weekday_rate := 3 * jacobs_weekday_rate
  let jakes_weekend_rate := 3 * jacobs_weekend_rate
  let weekday_earnings := jakes_weekday_rate * jakes_weekday_hours * 5
  let weekend_earnings := jakes_weekend_rate * jakes_weekend_hours * 2
  weekday_earnings + weekend_earnings

/-- Theorem stating that Jake's weekly earnings are $960. -/
theorem jakes_earnings_theorem : 
  jakes_weekly_earnings 6 8 8 5 = 960 := by
  sorry

end NUMINAMATH_CALUDE_jakes_earnings_theorem_l2956_295643


namespace NUMINAMATH_CALUDE_ellipse_triangle_area_l2956_295692

/-- The ellipse with equation 4x^2/49 + y^2/6 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (4 * p.1^2) / 49 + p.2^2 / 6 = 1}

/-- The foci of the ellipse -/
def F1 : ℝ × ℝ := sorry
def F2 : ℝ × ℝ := sorry

/-- A point P on the ellipse satisfying the given ratio condition -/
def P : ℝ × ℝ := sorry

/-- The distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- The area of a triangle given its three vertices -/
def triangleArea (p q r : ℝ × ℝ) : ℝ := sorry

theorem ellipse_triangle_area :
  P ∈ Ellipse ∧ 
  distance P F1 / distance P F2 = 4/3 →
  triangleArea P F1 F2 = 6 := by sorry

end NUMINAMATH_CALUDE_ellipse_triangle_area_l2956_295692


namespace NUMINAMATH_CALUDE_area_circle_outside_square_l2956_295634

/-- The area inside a circle but outside a square with shared center -/
theorem area_circle_outside_square (r : ℝ) (d : ℝ) :
  r = 1 →  -- radius of circle is 1
  d = 2 →  -- diagonal of square is 2
  π - d^2 / 2 = π - 2 :=
by
  sorry

#check area_circle_outside_square

end NUMINAMATH_CALUDE_area_circle_outside_square_l2956_295634


namespace NUMINAMATH_CALUDE_additive_implies_odd_l2956_295624

-- Define the property of the function
def is_additive (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- State the theorem
theorem additive_implies_odd (f : ℝ → ℝ) (h : is_additive f) : is_odd f := by
  sorry

end NUMINAMATH_CALUDE_additive_implies_odd_l2956_295624


namespace NUMINAMATH_CALUDE_probability_three_black_balls_l2956_295657

-- Define the number of white and black balls
def white_balls : ℕ := 4
def black_balls : ℕ := 8

-- Define the total number of balls
def total_balls : ℕ := white_balls + black_balls

-- Define the number of balls drawn
def drawn_balls : ℕ := 3

-- Define the probability function
def probability_all_black : ℚ :=
  (Nat.choose black_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ)

-- State the theorem
theorem probability_three_black_balls :
  probability_all_black = 14 / 55 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_black_balls_l2956_295657


namespace NUMINAMATH_CALUDE_xiaoming_mother_height_l2956_295697

/-- Given Xiaoming's height, stool height, and the difference between Xiaoming on the stool and his mother's height, prove the height of Xiaoming's mother. -/
theorem xiaoming_mother_height 
  (xiaoming_height : ℝ) 
  (stool_height : ℝ) 
  (height_difference : ℝ) 
  (h1 : xiaoming_height = 1.30)
  (h2 : stool_height = 0.4)
  (h3 : height_difference = 0.08)
  (h4 : xiaoming_height + stool_height = height_difference + mother_height) :
  mother_height = 1.62 :=
by
  sorry

#check xiaoming_mother_height

end NUMINAMATH_CALUDE_xiaoming_mother_height_l2956_295697


namespace NUMINAMATH_CALUDE_romanian_sequence_swaps_l2956_295605

/-- A Romanian sequence is a sequence of 3n letters where I, M, and O each occur exactly n times. -/
def RomanianSequence (n : ℕ) := Vector (Fin 3) (3 * n)

/-- The number of swaps required to transform one sequence into another. -/
def swapsRequired (n : ℕ) (X Y : RomanianSequence n) : ℕ := sorry

/-- There exists a Romanian sequence Y for any Romanian sequence X such that
    at least 3n^2/2 swaps are required to transform X into Y. -/
theorem romanian_sequence_swaps (n : ℕ) :
  ∀ X : RomanianSequence n, ∃ Y : RomanianSequence n,
    swapsRequired n X Y ≥ (3 * n^2) / 2 := by sorry

end NUMINAMATH_CALUDE_romanian_sequence_swaps_l2956_295605


namespace NUMINAMATH_CALUDE_tangent_parallel_to_x_axis_l2956_295600

/-- A curve defined by y = kx + ln x has a tangent at the point (1, k) that is parallel to the x-axis if and only if k = -1 -/
theorem tangent_parallel_to_x_axis (k : ℝ) : 
  (∃ f : ℝ → ℝ, f x = k * x + Real.log x) →
  (∃ t : ℝ → ℝ, t x = k * x + Real.log 1) →
  (∀ x : ℝ, (k + 1 / x) = 0) →
  k = -1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_parallel_to_x_axis_l2956_295600


namespace NUMINAMATH_CALUDE_julio_twice_james_age_l2956_295609

/-- The age difference between Julio and James -/
def age_difference : ℕ := 36 - 11

/-- The number of years until Julio's age is twice James' age -/
def years_until_double : ℕ := 14

theorem julio_twice_james_age :
  36 + years_until_double = 2 * (11 + years_until_double) :=
sorry

end NUMINAMATH_CALUDE_julio_twice_james_age_l2956_295609


namespace NUMINAMATH_CALUDE_square_root_fraction_equality_l2956_295678

theorem square_root_fraction_equality :
  Real.sqrt (8^2 + 15^2) / Real.sqrt (49 + 36) = 17 * Real.sqrt 85 / 85 := by
  sorry

end NUMINAMATH_CALUDE_square_root_fraction_equality_l2956_295678


namespace NUMINAMATH_CALUDE_angle_equality_from_cofunctions_l2956_295615

-- Define a type for angles
variable {α : Type*} [AddCommGroup α]

-- Define a function for co-functions (abstract representation)
variable (cofunc : α → ℝ)

-- State the theorem
theorem angle_equality_from_cofunctions (θ₁ θ₂ : α) :
  (θ₁ = θ₂) ∨ (cofunc θ₁ = cofunc θ₂) → θ₁ = θ₂ := by
  sorry

end NUMINAMATH_CALUDE_angle_equality_from_cofunctions_l2956_295615


namespace NUMINAMATH_CALUDE_geometric_progression_proof_l2956_295607

theorem geometric_progression_proof (b₁ q : ℝ) (h_decreasing : |q| < 1) :
  (b₁^3 / (1 - q^3)) / (b₁ / (1 - q)) = 48/7 →
  (b₁^4 / (1 - q^4)) / (b₁^2 / (1 - q^2)) = 144/17 →
  (b₁ = 3 ∨ b₁ = -3) ∧ q = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_proof_l2956_295607


namespace NUMINAMATH_CALUDE_inequalities_for_positive_reals_l2956_295688

theorem inequalities_for_positive_reals (a b : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : a + b = 2) : 
  a * b ≤ 1 ∧ a^2 + b^2 ≥ 2 ∧ Real.sqrt a + Real.sqrt b ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequalities_for_positive_reals_l2956_295688


namespace NUMINAMATH_CALUDE_divisibility_of_fifth_power_differences_l2956_295619

theorem divisibility_of_fifth_power_differences (x y z : ℤ) 
  (h_distinct_xy : x ≠ y) (h_distinct_yz : y ≠ z) (h_distinct_zx : z ≠ x) :
  ∃ k : ℤ, (x - y)^5 + (y - z)^5 + (z - x)^5 = 5 * k * (x - y) * (y - z) * (z - x) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_fifth_power_differences_l2956_295619


namespace NUMINAMATH_CALUDE_pizza_fraction_eaten_l2956_295652

theorem pizza_fraction_eaten (total_slices : ℕ) (whole_slices_eaten : ℕ) (shared_slices : ℕ) :
  total_slices = 16 →
  whole_slices_eaten = 2 →
  shared_slices = 2 →
  (whole_slices_eaten : ℚ) / total_slices + (shared_slices : ℚ) / (2 * total_slices) = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_pizza_fraction_eaten_l2956_295652


namespace NUMINAMATH_CALUDE_transformation_result_l2956_295687

/-- Reflect a point about the line y = x -/
def reflect_about_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

/-- Rotate a point 180° counterclockwise around (2,3) -/
def rotate_180_around_2_3 (p : ℝ × ℝ) : ℝ × ℝ :=
  (4 - p.1, 6 - p.2)

/-- The final position after transformations -/
def final_position : ℝ × ℝ := (-2, -1)

theorem transformation_result (m n : ℝ) : 
  rotate_180_around_2_3 (reflect_about_y_eq_x (m, n)) = final_position → n - m = -1 := by
  sorry

#check transformation_result

end NUMINAMATH_CALUDE_transformation_result_l2956_295687


namespace NUMINAMATH_CALUDE_negation_of_proposition_is_true_l2956_295662

theorem negation_of_proposition_is_true : 
  ∃ a : ℝ, (a > 2 ∧ a^2 ≥ 4) := by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_is_true_l2956_295662


namespace NUMINAMATH_CALUDE_triangle_side_difference_l2956_295670

theorem triangle_side_difference (a b c : ℝ) : 
  b = 8 → c = 3 → 
  a > 0 → b > 0 → c > 0 →
  a + b > c → a + c > b → b + c > a →
  (∃ (a_min a_max : ℕ), 
    (∀ x : ℕ, (x : ℝ) = a → a_min ≤ x ∧ x ≤ a_max) ∧
    (∀ y : ℕ, y < a_min → (y : ℝ) ≠ a) ∧
    (∀ z : ℕ, z > a_max → (z : ℝ) ≠ a) ∧
    a_max - a_min = 4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_difference_l2956_295670


namespace NUMINAMATH_CALUDE_fence_cost_l2956_295664

/-- The cost of fencing a square plot -/
theorem fence_cost (area : ℝ) (price_per_foot : ℝ) (cost : ℝ) : 
  area = 49 → price_per_foot = 58 → cost = 1624 := by
  sorry

end NUMINAMATH_CALUDE_fence_cost_l2956_295664


namespace NUMINAMATH_CALUDE_revenue_decrease_percentage_l2956_295689

theorem revenue_decrease_percentage (old_revenue new_revenue : ℝ) 
  (h1 : old_revenue = 69.0)
  (h2 : new_revenue = 52.0) :
  ∃ (ε : ℝ), abs ((old_revenue - new_revenue) / old_revenue * 100 - 24.64) < ε ∧ ε > 0 :=
by sorry

end NUMINAMATH_CALUDE_revenue_decrease_percentage_l2956_295689


namespace NUMINAMATH_CALUDE_lunch_cost_with_tip_l2956_295683

theorem lunch_cost_with_tip (total_cost : ℝ) (tip_percentage : ℝ) (cost_before_tip : ℝ) : 
  total_cost = 60.24 →
  tip_percentage = 0.20 →
  total_cost = cost_before_tip * (1 + tip_percentage) →
  cost_before_tip = 50.20 := by
sorry

end NUMINAMATH_CALUDE_lunch_cost_with_tip_l2956_295683


namespace NUMINAMATH_CALUDE_factor_and_divisor_statements_l2956_295661

-- Define what it means for a number to be a factor of another
def is_factor (a b : ℤ) : Prop := ∃ k : ℤ, b = a * k

-- Define what it means for a number to be a divisor of another
def is_divisor (a b : ℤ) : Prop := ∃ k : ℤ, b = a * k

theorem factor_and_divisor_statements :
  (is_factor 4 100) ∧
  (is_divisor 19 133 ∧ ¬ is_divisor 19 51) ∧
  (is_divisor 30 90 ∨ is_divisor 30 53) ∧
  (is_divisor 7 21 ∧ is_divisor 7 49) ∧
  (is_factor 10 200) := by sorry

end NUMINAMATH_CALUDE_factor_and_divisor_statements_l2956_295661


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l2956_295622

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l2956_295622


namespace NUMINAMATH_CALUDE_oil_redistribution_l2956_295668

theorem oil_redistribution (trucks_type1 trucks_type2 boxes_per_truck1 boxes_per_truck2 containers_per_box final_trucks : ℕ) 
  (h1 : trucks_type1 = 7)
  (h2 : trucks_type2 = 5)
  (h3 : boxes_per_truck1 = 20)
  (h4 : boxes_per_truck2 = 12)
  (h5 : containers_per_box = 8)
  (h6 : final_trucks = 10) :
  (trucks_type1 * boxes_per_truck1 + trucks_type2 * boxes_per_truck2) * containers_per_box / final_trucks = 160 := by
  sorry

#check oil_redistribution

end NUMINAMATH_CALUDE_oil_redistribution_l2956_295668


namespace NUMINAMATH_CALUDE_complex_set_equality_l2956_295612

theorem complex_set_equality (z : ℂ) : 
  Complex.abs ((z - 1)^2) = Complex.abs (z - 1)^2 ↔ z.im = 0 :=
sorry

end NUMINAMATH_CALUDE_complex_set_equality_l2956_295612


namespace NUMINAMATH_CALUDE_rectangle_vertex_numbers_l2956_295614

theorem rectangle_vertex_numbers (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : 2 * a ≥ b + d)
  (h2 : 2 * b ≥ a + c)
  (h3 : 2 * c ≥ b + d)
  (h4 : 2 * d ≥ a + c) :
  a = b ∧ b = c ∧ c = d :=
sorry

end NUMINAMATH_CALUDE_rectangle_vertex_numbers_l2956_295614


namespace NUMINAMATH_CALUDE_valid_number_difference_l2956_295684

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 1000000000) ∧ (n < 10000000000) ∧ (n % 11 = 0) ∧
  (∀ d : ℕ, d < 10 → (∃! i : ℕ, i < 10 ∧ (n / 10^i) % 10 = d))

def largest_valid_number : ℕ := 9876524130

def smallest_valid_number : ℕ := 1024375869

theorem valid_number_difference :
  is_valid_number largest_valid_number ∧
  is_valid_number smallest_valid_number ∧
  (∀ n : ℕ, is_valid_number n → n ≤ largest_valid_number) ∧
  (∀ n : ℕ, is_valid_number n → n ≥ smallest_valid_number) ∧
  (largest_valid_number - smallest_valid_number = 8852148261) :=
sorry

end NUMINAMATH_CALUDE_valid_number_difference_l2956_295684


namespace NUMINAMATH_CALUDE_remaining_payment_l2956_295623

def part_payment : ℝ := 875
def payment_percentage : ℝ := 0.25

theorem remaining_payment :
  let total_cost := part_payment / payment_percentage
  let remaining := total_cost - part_payment
  remaining = 2625 := by sorry

end NUMINAMATH_CALUDE_remaining_payment_l2956_295623


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l2956_295653

theorem polynomial_coefficient_sum (a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, a₁ * (x - 1)^4 + a₂ * (x - 1)^3 + a₃ * (x - 1)^2 + a₄ * (x - 1) + a₅ = x^4) →
  a₂ - a₃ + a₄ = 2 := by
sorry


end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l2956_295653


namespace NUMINAMATH_CALUDE_unknown_number_value_l2956_295640

theorem unknown_number_value (a x : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * x * 35 * 63) : x = 25 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_value_l2956_295640


namespace NUMINAMATH_CALUDE_cindys_calculation_l2956_295625

theorem cindys_calculation (x : ℝ) (h : (x - 7) / 5 = 51) : (x - 5) / 7 = 36 := by
  sorry

end NUMINAMATH_CALUDE_cindys_calculation_l2956_295625


namespace NUMINAMATH_CALUDE_probability_of_flush_l2956_295644

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of suits in a standard deck -/
def NumSuits : ℕ := 4

/-- Represents the number of cards chosen -/
def CardsChosen : ℕ := 6

/-- Represents the number of cards in each suit -/
def CardsPerSuit : ℕ := StandardDeck / NumSuits

/-- Calculates the number of ways to choose n items from k items -/
def choose (n k : ℕ) : ℕ := sorry

/-- Probability of forming a flush when choosing 6 cards at random from a standard 52-card deck -/
theorem probability_of_flush : 
  (NumSuits * choose CardsPerSuit CardsChosen) / choose StandardDeck CardsChosen = 3432 / 10179260 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_flush_l2956_295644


namespace NUMINAMATH_CALUDE_even_function_symmetry_is_universal_l2956_295665

-- Define what a universal proposition is
def is_universal_proposition (p : Prop) : Prop :=
  ∃ (U : Type) (P : U → Prop), p = ∀ (x : U), P x

-- Define what an even function is
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Define symmetry about y-axis for a function's graph
def symmetric_about_y_axis (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- State the theorem
theorem even_function_symmetry_is_universal :
  is_universal_proposition (∀ f : ℝ → ℝ, is_even_function f → symmetric_about_y_axis f) :=
sorry

end NUMINAMATH_CALUDE_even_function_symmetry_is_universal_l2956_295665


namespace NUMINAMATH_CALUDE_circumscribed_circle_equation_l2956_295679

theorem circumscribed_circle_equation (A B C : ℝ × ℝ) :
  A = (4, 1) → B = (6, -3) → C = (-3, 0) →
  ∃ (center : ℝ × ℝ) (r : ℝ),
    (∀ (x y : ℝ), (x - center.1)^2 + (y - center.2)^2 = r^2 ↔
      (x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2) ∨ (x = C.1 ∧ y = C.2)) ∧
    center = (1, -3) ∧ r^2 = 25 := by sorry

end NUMINAMATH_CALUDE_circumscribed_circle_equation_l2956_295679


namespace NUMINAMATH_CALUDE_class_test_problem_l2956_295631

theorem class_test_problem (p_first : ℝ) (p_second : ℝ) (p_neither : ℝ)
  (h1 : p_first = 0.7)
  (h2 : p_second = 0.55)
  (h3 : p_neither = 0.2) :
  p_first + p_second - (1 - p_neither) = 0.45 := by
sorry

end NUMINAMATH_CALUDE_class_test_problem_l2956_295631


namespace NUMINAMATH_CALUDE_min_value_expression_l2956_295610

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (((x^2 + y^2) * (2 * x^2 + 4 * y^2)).sqrt) / (x * y) ≥ 4 + 6 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2956_295610


namespace NUMINAMATH_CALUDE_system_solution_exists_l2956_295618

theorem system_solution_exists (x y z : ℝ) : 
  (x * y = 8 - 3 * x - 2 * y) →
  (y * z = 8 - 2 * y - 3 * z) →
  (x * z = 35 - 5 * x - 3 * z) →
  ∃ (x : ℝ), x = 8 := by
sorry

end NUMINAMATH_CALUDE_system_solution_exists_l2956_295618


namespace NUMINAMATH_CALUDE_xyz_product_l2956_295635

theorem xyz_product (x y z : ℂ) 
  (eq1 : x * y + 6 * y = -24)
  (eq2 : y * z + 6 * z = -24)
  (eq3 : z * x + 6 * x = -24) :
  x * y * z = 144 := by
sorry

end NUMINAMATH_CALUDE_xyz_product_l2956_295635


namespace NUMINAMATH_CALUDE_expression_value_l2956_295681

theorem expression_value : 
  (150^2 - 13^2) / (90^2 - 17^2) * ((90-17)*(90+17)) / ((150-13)*(150+13)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2956_295681


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2956_295621

theorem arithmetic_sequence_sum (d : ℝ) (h : d ≠ 0) :
  let a : ℕ → ℝ := λ n => (n - 1 : ℝ) * d
  ∃ m : ℕ, a m = (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9) ∧ m = 37 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2956_295621


namespace NUMINAMATH_CALUDE_valbonne_group_separation_l2956_295699

-- Define a type for participants
variable {Participant : Type}

-- Define a friendship relation
variable (friends : Participant → Participant → Prop)

-- Define a property that each participant has at most three friends
variable (at_most_three_friends : ∀ p : Participant, ∃ (f₁ f₂ f₃ : Participant), 
  ∀ f : Participant, friends p f → (f = f₁ ∨ f = f₂ ∨ f = f₃))

-- Define a partition of participants into two groups
variable (group : Participant → Bool)

-- State the theorem
theorem valbonne_group_separation :
  ∃ group : Participant → Bool,
    ∀ p : Participant,
      (∃! f : Participant, friends p f ∧ group p = group f) ∨
      (∀ f : Participant, friends p f → group p ≠ group f) :=
sorry

end NUMINAMATH_CALUDE_valbonne_group_separation_l2956_295699


namespace NUMINAMATH_CALUDE_range_of_a_l2956_295620

-- Define the propositions P and Q
def P (a : ℝ) : Prop := ∀ x : ℝ, x^2 - (a + 1)*x + 1 > 0

def Q (a : ℝ) : Prop := ∀ x : ℝ, |x - 1| ≥ a + 2

-- State the theorem
theorem range_of_a (a : ℝ) : (¬(P a ∨ Q a)) → a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2956_295620


namespace NUMINAMATH_CALUDE_subset_implies_a_equals_one_l2956_295693

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a-2, 2*a-2}

theorem subset_implies_a_equals_one :
  ∀ a : ℝ, A a ⊆ B a → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_one_l2956_295693


namespace NUMINAMATH_CALUDE_trapezoid_base_midpoint_relation_shorter_base_length_l2956_295674

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  long_base : ℝ
  midpoint_segment : ℝ
  short_base : ℝ

/-- The theorem stating the relationship between the bases and midpoint segment in a trapezoid -/
theorem trapezoid_base_midpoint_relation (t : Trapezoid) 
  (h1 : t.long_base = 113)
  (h2 : t.midpoint_segment = 4) :
  t.short_base = 105 := by
  sorry

/-- The main theorem proving the length of the shorter base -/
theorem shorter_base_length :
  ∃ t : Trapezoid, t.long_base = 113 ∧ t.midpoint_segment = 4 ∧ t.short_base = 105 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_base_midpoint_relation_shorter_base_length_l2956_295674


namespace NUMINAMATH_CALUDE_quadratic_increasing_condition_l2956_295690

/-- Given a quadratic function f(x) = x^2 - 2ax + 1 that is increasing on [1, +∞),
    prove that a ≤ 1 -/
theorem quadratic_increasing_condition (a : ℝ) : 
  (∀ x ≥ 1, ∀ y ≥ x, (y^2 - 2*a*y + 1) ≥ (x^2 - 2*a*x + 1)) → a ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_increasing_condition_l2956_295690


namespace NUMINAMATH_CALUDE_souvenir_price_increase_l2956_295672

theorem souvenir_price_increase (original_price final_price : ℝ) 
  (h1 : original_price = 76.8)
  (h2 : final_price = 120)
  (h3 : ∃ x : ℝ, original_price * (1 + x)^2 = final_price) :
  ∃ x : ℝ, original_price * (1 + x)^2 = final_price ∧ x = 0.25 := by
sorry

end NUMINAMATH_CALUDE_souvenir_price_increase_l2956_295672


namespace NUMINAMATH_CALUDE_central_angle_from_arc_length_l2956_295677

/-- Given a circle with radius 12 mm and an arc length of 144 mm, 
    the central angle in radians is equal to 12. -/
theorem central_angle_from_arc_length (R L θ : ℝ) : 
  R = 12 → L = 144 → L = R * θ → θ = 12 := by
  sorry

end NUMINAMATH_CALUDE_central_angle_from_arc_length_l2956_295677


namespace NUMINAMATH_CALUDE_definite_integral_ln_squared_over_sqrt_l2956_295669

theorem definite_integral_ln_squared_over_sqrt (e : Real) :
  let f : Real → Real := fun x => (Real.log x)^2 / Real.sqrt x
  let a : Real := 1
  let b : Real := Real.exp 2
  e > 0 →
  ∫ x in a..b, f x = 24 * e - 32 := by
sorry

end NUMINAMATH_CALUDE_definite_integral_ln_squared_over_sqrt_l2956_295669


namespace NUMINAMATH_CALUDE_dorothy_interest_l2956_295613

/-- Calculates the interest earned on an investment with annual compound interest. -/
def interest_earned (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years - principal

/-- The interest earned on Dorothy's investment -/
theorem dorothy_interest : 
  let principal := 2000
  let rate := 0.02
  let years := 3
  ⌊interest_earned principal rate years⌋ = 122 := by
  sorry

end NUMINAMATH_CALUDE_dorothy_interest_l2956_295613


namespace NUMINAMATH_CALUDE_james_weight_plates_purchase_l2956_295654

/-- Represents the purchase of a weight vest and weight plates -/
structure WeightPurchase where
  vest_cost : ℝ
  plate_cost_per_pound : ℝ
  discounted_200lb_vest_cost : ℝ
  savings : ℝ

/-- Calculates the number of pounds of weight plates purchased -/
def weight_plates_purchased (purchase : WeightPurchase) : ℕ :=
  sorry

/-- Theorem stating that James purchased 291 pounds of weight plates -/
theorem james_weight_plates_purchase :
  let purchase : WeightPurchase := {
    vest_cost := 250,
    plate_cost_per_pound := 1.2,
    discounted_200lb_vest_cost := 700 - 100,
    savings := 110
  }
  weight_plates_purchased purchase = 291 := by
  sorry

end NUMINAMATH_CALUDE_james_weight_plates_purchase_l2956_295654


namespace NUMINAMATH_CALUDE_positive_X_value_l2956_295641

-- Define the # relation
def hash (X Y : ℝ) : ℝ := X^2 + Y^2

-- Theorem statement
theorem positive_X_value (X : ℝ) (h : hash X 7 = 250) : X = Real.sqrt 201 :=
by sorry

end NUMINAMATH_CALUDE_positive_X_value_l2956_295641


namespace NUMINAMATH_CALUDE_symmetry_sum_l2956_295601

/-- Two points are symmetric about the x-axis if they have the same x-coordinate and opposite y-coordinates -/
def symmetric_about_x_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = p2.1 ∧ p1.2 = -p2.2

theorem symmetry_sum (x y : ℝ) :
  symmetric_about_x_axis (-2, y) (x, 3) → x + y = -5 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_sum_l2956_295601


namespace NUMINAMATH_CALUDE_inequality_condition_l2956_295617

theorem inequality_condition (a x : ℝ) : x^3 + 13*a^2*x > 5*a*x^2 + 9*a^3 ↔ x > a := by
  sorry

end NUMINAMATH_CALUDE_inequality_condition_l2956_295617


namespace NUMINAMATH_CALUDE_inequality_solution_implies_a_bound_l2956_295639

theorem inequality_solution_implies_a_bound (a : ℝ) : 
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ 2*x^2 - 8*x - 4 - a > 0) → a < -4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_a_bound_l2956_295639


namespace NUMINAMATH_CALUDE_expected_left_handed_students_l2956_295673

theorem expected_left_handed_students
  (total_students : ℕ)
  (left_handed_proportion : ℚ)
  (h1 : total_students = 32)
  (h2 : left_handed_proportion = 3 / 8) :
  ↑total_students * left_handed_proportion = 12 :=
by sorry

end NUMINAMATH_CALUDE_expected_left_handed_students_l2956_295673
