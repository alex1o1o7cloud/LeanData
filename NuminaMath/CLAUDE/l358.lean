import Mathlib

namespace NUMINAMATH_CALUDE_asterisk_replacement_l358_35813

theorem asterisk_replacement : ∃! (x : ℝ), x > 0 ∧ (x / 21) * (x / 189) = 1 := by
  sorry

end NUMINAMATH_CALUDE_asterisk_replacement_l358_35813


namespace NUMINAMATH_CALUDE_order_of_products_and_square_l358_35871

theorem order_of_products_and_square (x a b : ℝ) 
  (h1 : x < a) (h2 : a < b) (h3 : b < 0) : 
  b * x > a * x ∧ a * x > a ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_order_of_products_and_square_l358_35871


namespace NUMINAMATH_CALUDE_ellipse_condition_range_l358_35839

theorem ellipse_condition_range (m a : ℝ) : 
  (a > 0) →
  (m^2 + 12*a^2 < 7*a*m) →
  (∀ x y : ℝ, x^2 / (m - 1) + y^2 / (2 - m) = 1 → 
    ∃ c : ℝ, c > 0 ∧ ∀ p : ℝ × ℝ, p.1 = 0 → 
      (p.2 - c)^2 + p.1^2 = (m - 1)^2 ∨ (p.2 + c)^2 + p.1^2 = (m - 1)^2) →
  (∀ m : ℝ, (m^2 + 12*a^2 < 7*a*m) → 
    (∃ x y : ℝ, x^2 / (m - 1) + y^2 / (2 - m) = 1 ∧
      ∃ c : ℝ, c > 0 ∧ ∀ p : ℝ × ℝ, p.1 = 0 → 
        (p.2 - c)^2 + p.1^2 = (m - 1)^2 ∨ (p.2 + c)^2 + p.1^2 = (m - 1)^2)) →
  (∃ m : ℝ, (m^2 + 12*a^2 < 7*a*m) ∧ 
    ¬(∃ x y : ℝ, x^2 / (m - 1) + y^2 / (2 - m) = 1 ∧
      ∃ c : ℝ, c > 0 ∧ ∀ p : ℝ × ℝ, p.1 = 0 → 
        (p.2 - c)^2 + p.1^2 = (m - 1)^2 ∨ (p.2 + c)^2 + p.1^2 = (m - 1)^2)) →
  a ∈ Set.Icc (1/3 : ℝ) (3/8 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_ellipse_condition_range_l358_35839


namespace NUMINAMATH_CALUDE_custard_combinations_l358_35853

theorem custard_combinations (flavors : ℕ) (toppings : ℕ) 
  (h1 : flavors = 5) (h2 : toppings = 7) :
  flavors * (toppings.choose 2) = 105 := by
  sorry

end NUMINAMATH_CALUDE_custard_combinations_l358_35853


namespace NUMINAMATH_CALUDE_doctors_lawyers_ratio_l358_35873

theorem doctors_lawyers_ratio (d l : ℕ) (h_group_avg : (40 * d + 55 * l) / (d + l) = 45) : d = 2 * l := by
  sorry

end NUMINAMATH_CALUDE_doctors_lawyers_ratio_l358_35873


namespace NUMINAMATH_CALUDE_total_bread_is_370_l358_35855

/-- The amount of bread Cara ate for dinner, in grams -/
def dinner_bread : ℕ := 240

/-- The amount of bread Cara ate for lunch, in grams -/
def lunch_bread : ℕ := dinner_bread / 8

/-- The amount of bread Cara ate for breakfast, in grams -/
def breakfast_bread : ℕ := dinner_bread / 6

/-- The amount of bread Cara ate for snack, in grams -/
def snack_bread : ℕ := dinner_bread / 4

/-- The total amount of bread Cara ate, in grams -/
def total_bread : ℕ := dinner_bread + lunch_bread + breakfast_bread + snack_bread

theorem total_bread_is_370 : total_bread = 370 := by
  sorry

end NUMINAMATH_CALUDE_total_bread_is_370_l358_35855


namespace NUMINAMATH_CALUDE_line_moved_down_l358_35806

/-- Given a line with equation y = 2x + 3, prove that moving it down by 5 units
    results in the equation y = 2x - 2. -/
theorem line_moved_down (x y : ℝ) :
  (y = 2 * x + 3) → (y - 5 = 2 * x - 2) := by
  sorry

end NUMINAMATH_CALUDE_line_moved_down_l358_35806


namespace NUMINAMATH_CALUDE_simplify_fraction_l358_35811

theorem simplify_fraction : (90 : ℚ) / 150 = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l358_35811


namespace NUMINAMATH_CALUDE_unique_prime_with_prime_neighbors_l358_35814

theorem unique_prime_with_prime_neighbors : 
  ∃! p : ℕ, Nat.Prime p ∧ Nat.Prime (p^2 - 6) ∧ Nat.Prime (p^2 + 6) :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_with_prime_neighbors_l358_35814


namespace NUMINAMATH_CALUDE_rotation_180_maps_points_l358_35874

def rotation_180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

theorem rotation_180_maps_points :
  let C : ℝ × ℝ := (3, -2)
  let D : ℝ × ℝ := (2, -5)
  let C' : ℝ × ℝ := (-3, 2)
  let D' : ℝ × ℝ := (-2, 5)
  rotation_180 C = C' ∧ rotation_180 D = D' :=
by sorry

end NUMINAMATH_CALUDE_rotation_180_maps_points_l358_35874


namespace NUMINAMATH_CALUDE_smallest_n_for_trig_inequality_l358_35882

theorem smallest_n_for_trig_inequality :
  ∃ (n : ℕ), n > 0 ∧
  (∀ (x : ℝ), (Real.sin x)^n + (Real.cos x)^n ≤ 2^(1 - n)) ∧
  (∀ (m : ℕ), m > 0 → m < n →
    ∃ (x : ℝ), (Real.sin x)^m + (Real.cos x)^m > 2^(1 - m)) ∧
  n = 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_trig_inequality_l358_35882


namespace NUMINAMATH_CALUDE_smallest_multiple_37_congruent_7_mod_76_l358_35863

theorem smallest_multiple_37_congruent_7_mod_76 : ∃ (n : ℕ), n > 0 ∧ 37 ∣ n ∧ n ≡ 7 [MOD 76] ∧ ∀ (m : ℕ), m > 0 ∧ 37 ∣ m ∧ m ≡ 7 [MOD 76] → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_37_congruent_7_mod_76_l358_35863


namespace NUMINAMATH_CALUDE_fifteen_percent_problem_l358_35848

theorem fifteen_percent_problem : ∃ x : ℝ, (15 / 100) * x = 90 ∧ x = 600 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_percent_problem_l358_35848


namespace NUMINAMATH_CALUDE_parabola_translation_l358_35861

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  f : ℝ → ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h v : ℝ) : Parabola :=
  { f := fun x => p.f (x - h) + v }

/-- The original parabola y = x^2 - 2 -/
def original_parabola : Parabola :=
  { f := fun x => x^2 - 2 }

/-- The translated parabola -/
def translated_parabola : Parabola :=
  translate original_parabola 1 3

theorem parabola_translation :
  translated_parabola.f = fun x => (x - 1)^2 + 1 := by
  sorry


end NUMINAMATH_CALUDE_parabola_translation_l358_35861


namespace NUMINAMATH_CALUDE_overlaid_triangles_result_l358_35809

/-- Represents a transparent sheet with shaded triangles -/
structure Sheet :=
  (total_triangles : Nat)
  (shaded_triangles : Nat)

/-- Calculates the number of visible shaded triangles when sheets are overlaid -/
def visible_shaded_triangles (sheets : List Sheet) : Nat :=
  sorry

/-- Theorem stating the result for the specific problem -/
theorem overlaid_triangles_result :
  let sheets := [
    { total_triangles := 49, shaded_triangles := 16 },
    { total_triangles := 49, shaded_triangles := 16 },
    { total_triangles := 49, shaded_triangles := 16 }
  ]
  visible_shaded_triangles sheets = 31 := by
  sorry

end NUMINAMATH_CALUDE_overlaid_triangles_result_l358_35809


namespace NUMINAMATH_CALUDE_typing_problem_l358_35802

/-- Represents the typing speed of a typist in pages per hour -/
structure TypingSpeed :=
  (speed : ℝ)

/-- Represents the length of a chapter in pages -/
structure ChapterLength :=
  (pages : ℝ)

/-- Represents the time taken to type a chapter in hours -/
structure TypingTime :=
  (hours : ℝ)

theorem typing_problem (x y : TypingSpeed) (c1 c2 c3 : ChapterLength) (t1 t2 : TypingTime) :
  -- First chapter is twice as short as the second
  c1.pages = c2.pages / 2 →
  -- First chapter is three times longer than the third
  c1.pages = 3 * c3.pages →
  -- Typists retyped first chapter together in 3 hours and 36 minutes
  t1.hours = 3.6 →
  c1.pages / (x.speed + y.speed) = t1.hours →
  -- Second chapter was retyped in 8 hours
  t2.hours = 8 →
  -- First typist worked alone for 2 hours on second chapter
  2 * x.speed + 6 * (x.speed + y.speed) = c2.pages →
  -- Time for second typist to retype third chapter alone
  c3.pages / y.speed = 3 := by
sorry

end NUMINAMATH_CALUDE_typing_problem_l358_35802


namespace NUMINAMATH_CALUDE_juvys_garden_rows_l358_35804

/-- Represents Juvy's garden -/
structure Garden where
  rows : ℕ
  plants_per_row : ℕ
  parsley_rows : ℕ
  rosemary_rows : ℕ
  chive_plants : ℕ

/-- Theorem: The number of rows in Juvy's garden is 20 -/
theorem juvys_garden_rows (g : Garden) 
  (h1 : g.plants_per_row = 10)
  (h2 : g.parsley_rows = 3)
  (h3 : g.rosemary_rows = 2)
  (h4 : g.chive_plants = 150)
  (h5 : g.chive_plants = g.plants_per_row * (g.rows - g.parsley_rows - g.rosemary_rows)) :
  g.rows = 20 := by
  sorry

end NUMINAMATH_CALUDE_juvys_garden_rows_l358_35804


namespace NUMINAMATH_CALUDE_min_coach_handshakes_l358_35896

/-- Represents the number of handshakes in a soccer tournament -/
def tournament_handshakes (n : ℕ) (k : ℕ) : ℕ :=
  n.choose 2 + k

/-- Theorem stating the minimum number of coach handshakes -/
theorem min_coach_handshakes :
  ∃ (n : ℕ) (k : ℕ), tournament_handshakes n k = 406 ∧ k = 0 ∧ 
  ∀ (m : ℕ) (j : ℕ), tournament_handshakes m j = 406 → j ≥ k :=
by sorry

end NUMINAMATH_CALUDE_min_coach_handshakes_l358_35896


namespace NUMINAMATH_CALUDE_intersection_M_N_l358_35876

def M : Set ℕ := {x | x > 0 ∧ x ≤ 2}
def N : Set ℕ := {2, 6}

theorem intersection_M_N : M ∩ N = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l358_35876


namespace NUMINAMATH_CALUDE_triangle_angle_C_l358_35867

noncomputable def f (x θ : Real) : Real :=
  2 * Real.sin x * Real.cos (θ / 2) ^ 2 + Real.cos x * Real.sin θ - Real.sin x

theorem triangle_angle_C (θ A B C : Real) (a b c : Real) :
  0 < θ ∧ θ < Real.pi →
  f A θ = Real.sqrt 3 / 2 →
  a = 1 →
  b = Real.sqrt 2 →
  A + B + C = Real.pi →
  Real.sin A / a = Real.sin B / b →
  Real.sin A / a = Real.sin C / c →
  (C = 7 * Real.pi / 12 ∨ C = Real.pi / 12) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_C_l358_35867


namespace NUMINAMATH_CALUDE_matrix_inverse_proof_l358_35858

def A : Matrix (Fin 2) (Fin 2) ℚ := !![5, 4; -2, 8]

def A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![1/6, -1/12; 1/24, 5/48]

theorem matrix_inverse_proof :
  A * A_inv = 1 ∧ A_inv * A = 1 :=
by sorry

end NUMINAMATH_CALUDE_matrix_inverse_proof_l358_35858


namespace NUMINAMATH_CALUDE_two_person_subcommittees_l358_35826

/-- The number of combinations of n items taken k at a time -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The size of the original committee -/
def committeeSize : ℕ := 8

/-- The size of the sub-committee -/
def subCommitteeSize : ℕ := 2

/-- The number of different two-person sub-committees that can be selected from a committee of eight people -/
theorem two_person_subcommittees : choose committeeSize subCommitteeSize = 28 := by
  sorry

end NUMINAMATH_CALUDE_two_person_subcommittees_l358_35826


namespace NUMINAMATH_CALUDE_student_grouping_l358_35849

/-- Calculates the minimum number of groups needed to split students -/
def minGroups (totalStudents : ℕ) (maxGroupSize : ℕ) : ℕ :=
  (totalStudents + maxGroupSize - 1) / maxGroupSize

theorem student_grouping (totalStudents : ℕ) (maxGroupSize : ℕ) 
  (h1 : totalStudents = 30) (h2 : maxGroupSize = 12) :
  minGroups totalStudents maxGroupSize = 3 := by
  sorry

#eval minGroups 30 12  -- Should output 3

end NUMINAMATH_CALUDE_student_grouping_l358_35849


namespace NUMINAMATH_CALUDE_enthalpy_combustion_10_moles_glucose_l358_35875

/-- The standard enthalpy of combustion for glucose (C6H12O6) in kJ/mol -/
def standard_enthalpy_combustion_glucose : ℝ := -2800

/-- The number of moles of glucose -/
def moles_glucose : ℝ := 10

/-- The enthalpy of combustion for a given number of moles of glucose -/
def enthalpy_combustion (moles : ℝ) : ℝ :=
  standard_enthalpy_combustion_glucose * moles

/-- Theorem: The enthalpy of combustion for 10 moles of C6H12O6 is -28000 kJ -/
theorem enthalpy_combustion_10_moles_glucose :
  enthalpy_combustion moles_glucose = -28000 := by
  sorry

end NUMINAMATH_CALUDE_enthalpy_combustion_10_moles_glucose_l358_35875


namespace NUMINAMATH_CALUDE_franks_candy_bags_l358_35816

/-- Given that Frank puts 11 pieces of candy in each bag and has 22 pieces of candy in total,
    prove that the number of bags Frank would have is equal to 2. -/
theorem franks_candy_bags (pieces_per_bag : ℕ) (total_pieces : ℕ) (h1 : pieces_per_bag = 11) (h2 : total_pieces = 22) :
  total_pieces / pieces_per_bag = 2 := by
  sorry

end NUMINAMATH_CALUDE_franks_candy_bags_l358_35816


namespace NUMINAMATH_CALUDE_perpendicular_bisector_b_value_l358_35840

/-- The line x + y = b is a perpendicular bisector of the line segment from (2, 5) to (8, 11) -/
def is_perpendicular_bisector (b : ℝ) : Prop :=
  let midpoint := ((2 + 8) / 2, (5 + 11) / 2)
  midpoint.1 + midpoint.2 = b

/-- The value of b for which x + y = b is a perpendicular bisector of the line segment from (2, 5) to (8, 11) is 13 -/
theorem perpendicular_bisector_b_value :
  ∃ b : ℝ, is_perpendicular_bisector b ∧ b = 13 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_b_value_l358_35840


namespace NUMINAMATH_CALUDE_mode_identifies_favorite_dish_l358_35884

/-- A statistical measure for a dataset -/
inductive StatisticalMeasure
  | Mean
  | Median
  | Mode
  | Variance

/-- A dataset representing student preferences for dishes at a food festival -/
structure FoodFestivalData where
  preferences : List String

/-- Definition: The mode of a dataset is the most frequently occurring value -/
def mode (data : FoodFestivalData) : String :=
  sorry

/-- The statistical measure that identifies the favorite dish at a food festival -/
def favoriteDishMeasure : StatisticalMeasure :=
  sorry

/-- Theorem: The mode is the appropriate measure for identifying the favorite dish -/
theorem mode_identifies_favorite_dish :
  favoriteDishMeasure = StatisticalMeasure.Mode :=
  sorry

end NUMINAMATH_CALUDE_mode_identifies_favorite_dish_l358_35884


namespace NUMINAMATH_CALUDE_percentage_equivalence_l358_35822

theorem percentage_equivalence : 
  (75 / 100) * 600 = (50 / 100) * 900 := by sorry

end NUMINAMATH_CALUDE_percentage_equivalence_l358_35822


namespace NUMINAMATH_CALUDE_evelyns_remaining_bottle_caps_l358_35808

/-- The number of bottle caps Evelyn has left after losing some -/
def bottle_caps_left (initial : ℝ) (lost : ℝ) : ℝ := initial - lost

/-- Theorem: Evelyn's remaining bottle caps -/
theorem evelyns_remaining_bottle_caps :
  bottle_caps_left 63.75 18.36 = 45.39 := by
  sorry

end NUMINAMATH_CALUDE_evelyns_remaining_bottle_caps_l358_35808


namespace NUMINAMATH_CALUDE_students_disliking_menu_l358_35823

theorem students_disliking_menu (total : ℕ) (liked : ℕ) (h1 : total = 400) (h2 : liked = 235) :
  total - liked = 165 := by
  sorry

end NUMINAMATH_CALUDE_students_disliking_menu_l358_35823


namespace NUMINAMATH_CALUDE_min_sum_squares_l358_35892

theorem min_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  ∃ (m : ℝ), m = 4 ∧ ∀ (a b c : ℝ), a^3 + b^3 + c^3 - 3*a*b*c = 8 → x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2 ∧ x^2 + y^2 + z^2 ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l358_35892


namespace NUMINAMATH_CALUDE_area_ratio_bounds_l358_35893

-- Define an equilateral triangle
structure EquilateralTriangle where
  side : ℝ
  side_positive : side > 0

-- Define a line passing through the centroid
structure CentroidLine where
  angle : ℝ  -- Angle of the line with respect to a reference

-- Define the two parts created by the line
structure TriangleParts where
  part1 : ℝ
  part2 : ℝ
  parts_positive : part1 > 0 ∧ part2 > 0
  parts_sum : part1 + part2 = 1  -- Normalized to total area 1

-- Main theorem
theorem area_ratio_bounds (t : EquilateralTriangle) (l : CentroidLine) 
  (p : TriangleParts) : 
  4/5 ≤ min (p.part1 / p.part2) (p.part2 / p.part1) ∧ 
  max (p.part1 / p.part2) (p.part2 / p.part1) ≤ 5/4 :=
sorry

end NUMINAMATH_CALUDE_area_ratio_bounds_l358_35893


namespace NUMINAMATH_CALUDE_sum_of_extrema_l358_35815

/-- A function f(x) = 2x³ - ax² + 1 with exactly one zero in (0, +∞) -/
def f (a : ℝ) : ℝ → ℝ := λ x ↦ 2 * x^3 - a * x^2 + 1

/-- The property that f has exactly one zero in (0, +∞) -/
def has_one_zero (a : ℝ) : Prop :=
  ∃! x : ℝ, x > 0 ∧ f a x = 0

/-- The theorem stating that if f has one zero in (0, +∞), then the sum of its max and min on [-1, 1] is -3 -/
theorem sum_of_extrema (a : ℝ) (h : has_one_zero a) :
  (⨆ x ∈ Set.Icc (-1) 1, f a x) + (⨅ x ∈ Set.Icc (-1) 1, f a x) = -3 :=
sorry

end NUMINAMATH_CALUDE_sum_of_extrema_l358_35815


namespace NUMINAMATH_CALUDE_original_fraction_is_two_thirds_l358_35872

theorem original_fraction_is_two_thirds :
  ∀ (a b : ℕ), 
    a ≠ 0 → b ≠ 0 →
    (a^3 : ℚ) / (b + 3 : ℚ) = 2 * (a : ℚ) / (b : ℚ) →
    (∀ d : ℕ, d ≠ 0 → d ∣ a ∧ d ∣ b → d = 1) →
    a = 2 ∧ b = 3 := by
  sorry

end NUMINAMATH_CALUDE_original_fraction_is_two_thirds_l358_35872


namespace NUMINAMATH_CALUDE_shooting_challenge_sequences_l358_35812

theorem shooting_challenge_sequences : ℕ := by
  -- Define the total number of targets
  let total_targets : ℕ := 10

  -- Define the number of targets in each column
  let targets_A : ℕ := 4
  let targets_B : ℕ := 4
  let targets_C : ℕ := 2

  -- Assert that the sum of targets in all columns equals the total targets
  have h1 : targets_A + targets_B + targets_C = total_targets := by sorry

  -- Define the number of different sequences
  let num_sequences : ℕ := (Nat.factorial total_targets) / 
    ((Nat.factorial targets_A) * (Nat.factorial targets_B) * (Nat.factorial targets_C))

  -- Prove that the number of sequences equals 3150
  have h2 : num_sequences = 3150 := by sorry

  -- Return the result
  exact 3150

end NUMINAMATH_CALUDE_shooting_challenge_sequences_l358_35812


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l358_35810

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_sum : a 3 + a 4 + a 5 + a 13 + a 14 + a 15 = 8) :
  5 * a 7 - 2 * a 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l358_35810


namespace NUMINAMATH_CALUDE_cookies_per_neighbor_l358_35881

/-- Proves the number of cookies each neighbor was supposed to take -/
theorem cookies_per_neighbor
  (total_cookies : ℕ)
  (num_neighbors : ℕ)
  (cookies_left : ℕ)
  (sarah_cookies : ℕ)
  (h1 : total_cookies = 150)
  (h2 : num_neighbors = 15)
  (h3 : cookies_left = 8)
  (h4 : sarah_cookies = 12)
  : total_cookies / num_neighbors = 10 := by
  sorry

#check cookies_per_neighbor

end NUMINAMATH_CALUDE_cookies_per_neighbor_l358_35881


namespace NUMINAMATH_CALUDE_elderly_arrangement_count_l358_35829

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem elderly_arrangement_count :
  let volunteers : ℕ := 5
  let elderly : ℕ := 2
  let total_units : ℕ := volunteers + 1  -- Treating elderly as one unit
  let total_arrangements : ℕ := factorial total_units * factorial elderly
  let end_arrangements : ℕ := 2 * factorial (total_units - 1) * factorial elderly
  total_arrangements - end_arrangements = 960 := by
  sorry

end NUMINAMATH_CALUDE_elderly_arrangement_count_l358_35829


namespace NUMINAMATH_CALUDE_all_cells_happy_l358_35894

def Board := Fin 10 → Fin 10 → Bool

def isBlue (board : Board) (i j : Fin 10) : Bool :=
  (i.val + j.val) % 2 = 0

def neighbors (i j : Fin 10) : List (Fin 10 × Fin 10) :=
  [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
    |> List.filter (fun (x, y) => x.val < 10 && y.val < 10)

def countBlueNeighbors (board : Board) (i j : Fin 10) : Nat :=
  (neighbors i j).filter (fun (x, y) => isBlue board x y) |>.length

theorem all_cells_happy (board : Board) :
  ∀ i j : Fin 10, countBlueNeighbors board i j = 2 := by
  sorry

#check all_cells_happy

end NUMINAMATH_CALUDE_all_cells_happy_l358_35894


namespace NUMINAMATH_CALUDE_product_properties_l358_35831

-- Define a function to represent the product of all combinations
def product_of_combinations (a : List ℕ) : ℝ :=
  sorry

-- Theorem statement
theorem product_properties (a : List ℕ) :
  (∃ m : ℤ, product_of_combinations a = m) ∧
  (∃ n : ℤ, product_of_combinations a = n^2) :=
sorry

end NUMINAMATH_CALUDE_product_properties_l358_35831


namespace NUMINAMATH_CALUDE_camp_wonka_ratio_l358_35889

theorem camp_wonka_ratio : 
  ∀ (total_campers : ℕ) (boys girls : ℕ) (marshmallows : ℕ),
    total_campers = 96 →
    girls = total_campers / 3 →
    boys = total_campers - girls →
    marshmallows = 56 →
    (boys : ℚ) * (1/2) + (girls : ℚ) * (3/4) = marshmallows →
    (boys : ℚ) / (total_campers : ℚ) = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_camp_wonka_ratio_l358_35889


namespace NUMINAMATH_CALUDE_x_eq_3_sufficient_not_necessary_l358_35885

/-- Two-dimensional vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Check if two 2D vectors are parallel -/
def areParallel (v w : Vector2D) : Prop :=
  v.x * w.y = v.y * w.x

/-- Definition of vector a -/
def a (x : ℝ) : Vector2D :=
  ⟨2, x - 1⟩

/-- Definition of vector b -/
def b (x : ℝ) : Vector2D :=
  ⟨x + 1, 4⟩

/-- Theorem stating that x = 3 is a sufficient but not necessary condition for a ∥ b -/
theorem x_eq_3_sufficient_not_necessary :
  (∃ (x : ℝ), x ≠ 3 ∧ areParallel (a x) (b x)) ∧
  (∀ (x : ℝ), x = 3 → areParallel (a x) (b x)) :=
sorry

end NUMINAMATH_CALUDE_x_eq_3_sufficient_not_necessary_l358_35885


namespace NUMINAMATH_CALUDE_ring_width_equals_disk_radius_l358_35887

/-- A flat ring formed by two concentric circles with seven equal touching disks inserted -/
structure FlatRing where
  R₁ : ℝ  -- Radius of the outer circle
  R₂ : ℝ  -- Radius of the inner circle
  r : ℝ   -- Radius of each disk
  h₁ : R₁ > R₂  -- Outer radius is greater than inner radius
  h₂ : R₂ = 3 * r  -- Inner radius is 3 times the disk radius
  h₃ : 7 * π * r^2 = π * (R₁^2 - R₂^2)  -- Area of ring equals sum of disk areas

/-- The width of the ring is equal to the radius of one disk -/
theorem ring_width_equals_disk_radius (ring : FlatRing) : ring.R₁ - ring.R₂ = ring.r := by
  sorry


end NUMINAMATH_CALUDE_ring_width_equals_disk_radius_l358_35887


namespace NUMINAMATH_CALUDE_valid_sequence_power_of_two_l358_35879

/-- A sequence of pairwise distinct reals satisfying the given condition -/
def ValidSequence (N : ℕ) (a : ℕ → ℝ) : Prop :=
  N ≥ 3 ∧
  (∀ i j, i < N → j < N → i ≠ j → a i ≠ a j) ∧
  (∀ i, i < N → a i ≥ a ((2 * i) % N))

/-- The theorem stating that N must be a power of 2 -/
theorem valid_sequence_power_of_two (N : ℕ) (a : ℕ → ℝ) :
  ValidSequence N a → ∃ k : ℕ, N = 2^k :=
sorry

end NUMINAMATH_CALUDE_valid_sequence_power_of_two_l358_35879


namespace NUMINAMATH_CALUDE_yellow_tint_percentage_l358_35828

/-- Calculates the percentage of yellow tint in an updated mixture -/
theorem yellow_tint_percentage 
  (original_volume : ℝ) 
  (original_yellow_percentage : ℝ) 
  (added_yellow : ℝ) : 
  original_volume = 20 →
  original_yellow_percentage = 0.5 →
  added_yellow = 6 →
  let original_yellow := original_volume * original_yellow_percentage
  let total_yellow := original_yellow + added_yellow
  let new_volume := original_volume + added_yellow
  (total_yellow / new_volume) * 100 = 61.5 := by
sorry

end NUMINAMATH_CALUDE_yellow_tint_percentage_l358_35828


namespace NUMINAMATH_CALUDE_scaled_arithmetic_sequence_l358_35817

/-- Given an arithmetic sequence and a non-zero constant, prove that scaling the sequence by the constant results in another arithmetic sequence with a scaled common difference. -/
theorem scaled_arithmetic_sequence
  (a : ℕ → ℝ) -- The original arithmetic sequence
  (d : ℝ) -- Common difference of the original sequence
  (c : ℝ) -- Scaling constant
  (h₁ : c ≠ 0) -- Assumption that c is non-zero
  (h₂ : ∀ n : ℕ, a (n + 1) - a n = d) -- Definition of arithmetic sequence
  : ∀ n : ℕ, (c * a (n + 1)) - (c * a n) = c * d := by
  sorry

end NUMINAMATH_CALUDE_scaled_arithmetic_sequence_l358_35817


namespace NUMINAMATH_CALUDE_fahrenheit_to_celsius_l358_35856

theorem fahrenheit_to_celsius (C F : ℚ) : 
  C = 35 → C = (7/12) * (F - 40) → F = 100 := by
  sorry

end NUMINAMATH_CALUDE_fahrenheit_to_celsius_l358_35856


namespace NUMINAMATH_CALUDE_derivative_even_implies_b_zero_l358_35868

/-- A cubic polynomial function -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + 2

/-- The derivative of f -/
def f' (a b c : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

/-- A function is even if f(x) = f(-x) for all x -/
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

/-- If the derivative of f is even, then b = 0 -/
theorem derivative_even_implies_b_zero (a b c : ℝ) :
  is_even (f' a b c) → b = 0 := by sorry

end NUMINAMATH_CALUDE_derivative_even_implies_b_zero_l358_35868


namespace NUMINAMATH_CALUDE_min_value_fraction_sum_l358_35854

theorem min_value_fraction_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 6) :
  (9 / a + 4 / b + 25 / c) ≥ 50 / 3 ∧ 
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ + b₀ + c₀ = 6 ∧ (9 / a₀ + 4 / b₀ + 25 / c₀ = 50 / 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_sum_l358_35854


namespace NUMINAMATH_CALUDE_triangle_inequality_fraction_l358_35886

theorem triangle_inequality_fraction (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b) : 
  (a + b) / (1 + a + b) > c / (1 + c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_fraction_l358_35886


namespace NUMINAMATH_CALUDE_remainder_problem_l358_35883

theorem remainder_problem :
  {x : ℕ | x < 100 ∧ x % 7 = 3 ∧ x % 9 = 4} = {31, 94} := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l358_35883


namespace NUMINAMATH_CALUDE_initial_stops_l358_35843

/-- Represents the number of stops made by a delivery driver -/
structure DeliveryStops where
  total : Nat
  after_initial : Nat
  initial : Nat

/-- Theorem stating the number of initial stops given the total and after-initial stops -/
theorem initial_stops (d : DeliveryStops) 
  (h1 : d.total = 7) 
  (h2 : d.after_initial = 4) 
  (h3 : d.total = d.initial + d.after_initial) : 
  d.initial = 3 := by
  sorry

end NUMINAMATH_CALUDE_initial_stops_l358_35843


namespace NUMINAMATH_CALUDE_quadratic_root_implies_s_value_l358_35860

theorem quadratic_root_implies_s_value (p s : ℝ) :
  (∃ (x : ℂ), 3 * x^2 + p * x + s = 0 ∧ x = 4 + 3*I) →
  s = 75 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_s_value_l358_35860


namespace NUMINAMATH_CALUDE_range_of_x_given_conditions_l358_35818

def is_monotone_increasing_on_nonpositive (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y ∧ y ≤ 0 → f x ≤ f y

def is_symmetric_about_y_axis (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem range_of_x_given_conditions (f : ℝ → ℝ) 
  (h1 : is_monotone_increasing_on_nonpositive f)
  (h2 : is_symmetric_about_y_axis f)
  (h3 : ∀ x, f (x - 2) > f 2) :
  ∀ x, (0 < x ∧ x < 4) ↔ (f (x - 2) > f 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_x_given_conditions_l358_35818


namespace NUMINAMATH_CALUDE_gas_cost_per_gallon_l358_35898

/-- Calculates the cost of gas per gallon given Carla's trip details --/
theorem gas_cost_per_gallon 
  (distance_to_grocery : ℝ) 
  (distance_to_school : ℝ) 
  (distance_to_soccer : ℝ) 
  (miles_per_gallon : ℝ) 
  (total_gas_cost : ℝ) 
  (h1 : distance_to_grocery = 8) 
  (h2 : distance_to_school = 6) 
  (h3 : distance_to_soccer = 12) 
  (h4 : miles_per_gallon = 25) 
  (h5 : total_gas_cost = 5) :
  (total_gas_cost / ((distance_to_grocery + distance_to_school + distance_to_soccer + 2 * distance_to_soccer) / miles_per_gallon)) = 2.5 := by
sorry

end NUMINAMATH_CALUDE_gas_cost_per_gallon_l358_35898


namespace NUMINAMATH_CALUDE_pascal_triangle_ratio_l358_35807

/-- Binomial coefficient -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The row number in Pascal's Triangle -/
def n : ℕ := 53

/-- The position of the first entry in the consecutive trio -/
def r : ℕ := 23

/-- Theorem stating that three consecutive entries in row 53 of Pascal's Triangle are in the ratio 4:5:6 -/
theorem pascal_triangle_ratio :
  ∃ (r : ℕ), r < n ∧ 
    (choose n r : ℚ) / (choose n (r + 1)) = 4 / 5 ∧
    (choose n (r + 1) : ℚ) / (choose n (r + 2)) = 5 / 6 := by
  sorry


end NUMINAMATH_CALUDE_pascal_triangle_ratio_l358_35807


namespace NUMINAMATH_CALUDE_baker_pastries_sold_l358_35805

/-- Given information about a baker's production and sales of cakes and pastries, 
    prove that the number of pastries sold equals the number of cakes made. -/
theorem baker_pastries_sold (cakes_made pastries_made : ℕ) 
    (h1 : cakes_made = 19)
    (h2 : pastries_made = 131)
    (h3 : pastries_made - cakes_made = 112) :
    pastries_made - (pastries_made - cakes_made) = cakes_made := by
  sorry

end NUMINAMATH_CALUDE_baker_pastries_sold_l358_35805


namespace NUMINAMATH_CALUDE_xiao_wei_wears_five_l358_35801

/-- Represents the five people in the line -/
inductive Person : Type
  | XiaoWang
  | XiaoZha
  | XiaoTian
  | XiaoYan
  | XiaoWei

/-- Represents the hat numbers -/
inductive HatNumber : Type
  | One
  | Two
  | Three
  | Four
  | Five

/-- Function that assigns a hat number to each person -/
def hatAssignment : Person → HatNumber := sorry

/-- Function that determines if a person can see another person's hat -/
def canSee : Person → Person → Prop := sorry

/-- The hat numbers are all different -/
axiom all_different : ∀ p q : Person, p ≠ q → hatAssignment p ≠ hatAssignment q

/-- Xiao Wang cannot see any hats -/
axiom xiao_wang_sees_none : ∀ p : Person, ¬(canSee Person.XiaoWang p)

/-- Xiao Zha can only see hat 4 -/
axiom xiao_zha_sees_four : ∃! p : Person, canSee Person.XiaoZha p ∧ hatAssignment p = HatNumber.Four

/-- Xiao Tian does not see hat 3, but can see hat 1 -/
axiom xiao_tian_condition : (∃ p : Person, canSee Person.XiaoTian p ∧ hatAssignment p = HatNumber.One) ∧
                            (∀ p : Person, canSee Person.XiaoTian p → hatAssignment p ≠ HatNumber.Three)

/-- Xiao Yan can see three hats, but not hat 3 -/
axiom xiao_yan_condition : (∃ p q r : Person, p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
                            canSee Person.XiaoYan p ∧ canSee Person.XiaoYan q ∧ canSee Person.XiaoYan r) ∧
                           (∀ p : Person, canSee Person.XiaoYan p → hatAssignment p ≠ HatNumber.Three)

/-- Xiao Wei can see hat 3 and hat 2 -/
axiom xiao_wei_condition : (∃ p : Person, canSee Person.XiaoWei p ∧ hatAssignment p = HatNumber.Three) ∧
                           (∃ q : Person, canSee Person.XiaoWei q ∧ hatAssignment q = HatNumber.Two)

/-- Theorem: Xiao Wei is wearing hat number 5 -/
theorem xiao_wei_wears_five : hatAssignment Person.XiaoWei = HatNumber.Five := by sorry

end NUMINAMATH_CALUDE_xiao_wei_wears_five_l358_35801


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_4_range_of_a_for_inequality_l358_35834

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 + 2
def g (a x : ℝ) : ℝ := |x - a| - |x - 1|

-- Theorem for part 1
theorem solution_set_when_a_is_4 :
  {x : ℝ | f x > g 4 x} = {x : ℝ | x > 1 ∨ x ≤ -1} := by sorry

-- Theorem for part 2
theorem range_of_a_for_inequality :
  (∀ x₁ x₂ : ℝ, f x₁ ≥ g a x₂) ↔ -1 ≤ a ∧ a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_4_range_of_a_for_inequality_l358_35834


namespace NUMINAMATH_CALUDE_max_pairs_correct_l358_35800

def max_pairs (n : ℕ) : ℕ :=
  let k := (8037 : ℕ) / 5
  k

theorem max_pairs_correct (n : ℕ) (h : n = 4019) :
  ∀ (k : ℕ) (pairs : List (ℕ × ℕ)),
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 < p.2 ∧ p.1 ∈ Finset.range n ∧ p.2 ∈ Finset.range n) →
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) →
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 + p.2 ≤ n) →
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) →
    pairs.length ≤ max_pairs n :=
by sorry

end NUMINAMATH_CALUDE_max_pairs_correct_l358_35800


namespace NUMINAMATH_CALUDE_sin_405_degrees_l358_35820

theorem sin_405_degrees : Real.sin (405 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_405_degrees_l358_35820


namespace NUMINAMATH_CALUDE_distribute_five_items_to_fifteen_recipients_l358_35850

/-- The number of ways to distribute distinct items to recipients -/
def distribute_items (num_items : ℕ) (num_recipients : ℕ) : ℕ :=
  num_recipients ^ num_items

/-- Theorem: Distributing 5 distinct items to 15 recipients results in 759,375 possible ways -/
theorem distribute_five_items_to_fifteen_recipients :
  distribute_items 5 15 = 759375 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_items_to_fifteen_recipients_l358_35850


namespace NUMINAMATH_CALUDE_total_points_scored_l358_35865

/-- Given a player who plays 13 games and scores 7 points in each game,
    the total number of points scored is equal to 91. -/
theorem total_points_scored (games : ℕ) (points_per_game : ℕ) : 
  games = 13 → points_per_game = 7 → games * points_per_game = 91 := by
  sorry

end NUMINAMATH_CALUDE_total_points_scored_l358_35865


namespace NUMINAMATH_CALUDE_square_equation_solution_l358_35819

theorem square_equation_solution : ∃ (M : ℕ), M > 0 ∧ 12^2 * 30^2 = 15^2 * M^2 ∧ M = 24 := by sorry

end NUMINAMATH_CALUDE_square_equation_solution_l358_35819


namespace NUMINAMATH_CALUDE_cube_roots_of_unity_l358_35842

theorem cube_roots_of_unity :
  let z₁ : ℂ := 1
  let z₂ : ℂ := -1/2 + Complex.I * Real.sqrt 3 / 2
  let z₃ : ℂ := -1/2 - Complex.I * Real.sqrt 3 / 2
  ∀ z : ℂ, z^3 = 1 ↔ z = z₁ ∨ z = z₂ ∨ z = z₃ := by
sorry

end NUMINAMATH_CALUDE_cube_roots_of_unity_l358_35842


namespace NUMINAMATH_CALUDE_planes_perpendicular_to_line_are_parallel_lines_perpendicular_to_plane_are_parallel_l358_35836

-- Define the basic geometric objects
variable (Point Line Plane : Type)

-- Define the geometric relationships
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane_line : Plane → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- Theorem 1: Two different planes perpendicular to the same line are parallel
theorem planes_perpendicular_to_line_are_parallel
  (l : Line) (p1 p2 : Plane) (h1 : p1 ≠ p2)
  (h2 : perpendicular_plane_line p1 l) (h3 : perpendicular_plane_line p2 l) :
  parallel_planes p1 p2 :=
sorry

-- Theorem 2: Two different lines perpendicular to the same plane are parallel
theorem lines_perpendicular_to_plane_are_parallel
  (p : Plane) (l1 l2 : Line) (h1 : l1 ≠ l2)
  (h2 : perpendicular_line_plane l1 p) (h3 : perpendicular_line_plane l2 p) :
  parallel_lines l1 l2 :=
sorry

end NUMINAMATH_CALUDE_planes_perpendicular_to_line_are_parallel_lines_perpendicular_to_plane_are_parallel_l358_35836


namespace NUMINAMATH_CALUDE_min_filtration_processes_correct_l358_35824

/-- The reduction rate of impurities for each filtration process -/
def reduction_rate : ℝ := 0.20

/-- The target percentage of impurities after filtration -/
def target_percentage : ℝ := 0.05

/-- Approximation of log₂ -/
def log2_approx : ℝ := 0.3010

/-- The minimum number of filtration processes required -/
def min_filtration_processes : ℕ := 14

/-- Theorem stating the minimum number of filtration processes required -/
theorem min_filtration_processes_correct :
  ∀ n : ℕ,
  (1 - reduction_rate) ^ n < target_percentage →
  n ≥ min_filtration_processes :=
sorry

end NUMINAMATH_CALUDE_min_filtration_processes_correct_l358_35824


namespace NUMINAMATH_CALUDE_smallest_area_special_square_l358_35870

/-- A square with two vertices on a line and two on a parabola -/
structure SpecialSquare where
  /-- The y-intercept of the line containing two vertices of the square -/
  k : ℝ
  /-- The side length of the square -/
  s : ℝ
  /-- Two vertices of the square lie on the line y = 3x - 5 -/
  line_constraint : ∃ (x₁ x₂ : ℝ), y = 3 * x₁ - 5 ∧ y = 3 * x₂ - 5
  /-- Two vertices of the square lie on the parabola y = x^2 + 4 -/
  parabola_constraint : ∃ (x₁ x₂ : ℝ), y = x₁^2 + 4 ∧ y = x₂^2 + 4
  /-- The square's sides are parallel/perpendicular to coordinate axes -/
  axis_aligned : True
  /-- The area of the square is s^2 -/
  area_eq : s^2 = 10 * (25 + 4 * k)

/-- The theorem stating the smallest possible area of the special square -/
theorem smallest_area_special_square :
  ∀ (sq : SpecialSquare), sq.s^2 ≥ 200 :=
sorry

end NUMINAMATH_CALUDE_smallest_area_special_square_l358_35870


namespace NUMINAMATH_CALUDE_sequence_product_theorem_l358_35841

def arithmetic_sequence (n : ℕ) : ℕ :=
  2 * n - 1

def geometric_sequence (n : ℕ) : ℕ :=
  2^(n - 1)

theorem sequence_product_theorem :
  let a := arithmetic_sequence
  let b := geometric_sequence
  b (a 1) * b (a 3) * b (a 5) = 4096 := by
  sorry

end NUMINAMATH_CALUDE_sequence_product_theorem_l358_35841


namespace NUMINAMATH_CALUDE_derivative_cos_times_exp_sin_l358_35862

/-- The derivative of f(x) = cos(x) * e^(sin(x)) -/
theorem derivative_cos_times_exp_sin (x : ℝ) :
  deriv (fun x => Real.cos x * Real.exp (Real.sin x)) x =
  (Real.cos x ^ 2 - Real.sin x) * Real.exp (Real.sin x) := by
sorry

end NUMINAMATH_CALUDE_derivative_cos_times_exp_sin_l358_35862


namespace NUMINAMATH_CALUDE_min_value_quadratic_l358_35821

theorem min_value_quadratic (s : ℝ) :
  -8 * s^2 + 64 * s + 20 ≥ 148 ∧ ∃ t : ℝ, -8 * t^2 + 64 * t + 20 = 148 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l358_35821


namespace NUMINAMATH_CALUDE_correct_statement_l358_35878

-- Define propositions P and Q
def P : Prop := Real.pi < 2
def Q : Prop := Real.pi > 3

-- Theorem statement
theorem correct_statement :
  (P ∨ Q) ∧ (¬P) := by sorry

end NUMINAMATH_CALUDE_correct_statement_l358_35878


namespace NUMINAMATH_CALUDE_range_of_trig_function_l358_35895

open Real

theorem range_of_trig_function :
  let f : ℝ → ℝ := λ x ↦ 2 * cos x + sin (2 * x)
  ∃ (a b : ℝ), a = -3 * Real.sqrt 3 / 2 ∧ b = 3 * Real.sqrt 3 / 2 ∧
    (∀ x, f x ∈ Set.Icc a b) ∧
    (∀ y ∈ Set.Icc a b, ∃ x, f x = y) :=
by sorry

end NUMINAMATH_CALUDE_range_of_trig_function_l358_35895


namespace NUMINAMATH_CALUDE_min_operation_result_l358_35846

def S : Finset Nat := {4, 6, 8, 12, 14, 18}

def operation (a b c : Nat) : Nat :=
  (a + b) * c - min a (min b c)

theorem min_operation_result :
  ∃ (result : Nat), result = 52 ∧
  ∀ (a b c : Nat), a ∈ S → b ∈ S → c ∈ S →
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  operation a b c ≥ result :=
sorry

end NUMINAMATH_CALUDE_min_operation_result_l358_35846


namespace NUMINAMATH_CALUDE_chord_equation_l358_35869

def Circle := {p : ℝ × ℝ | (p.1 - 3)^2 + p.2^2 = 9}

def P : ℝ × ℝ := (1, 1)

def is_midpoint (m : ℝ × ℝ) (p : ℝ × ℝ) (n : ℝ × ℝ) : Prop :=
  p.1 = (m.1 + n.1) / 2 ∧ p.2 = (m.2 + n.2) / 2

theorem chord_equation (M N : ℝ × ℝ) (h1 : M ∈ Circle) (h2 : N ∈ Circle)
  (h3 : is_midpoint M P N) :
  ∃ (a b c : ℝ), a * P.1 + b * P.2 + c = 0 ∧
                 ∀ (x y : ℝ), (x, y) ∈ Circle → 
                 ((x, y) = M ∨ (x, y) = N) → 
                 a * x + b * y + c = 0 ∧
                 (a, b, c) = (2, -1, -1) := by
  sorry

end NUMINAMATH_CALUDE_chord_equation_l358_35869


namespace NUMINAMATH_CALUDE_solution_xyz_l358_35837

theorem solution_xyz (x y z : ℝ) 
  (eq1 : 2*x + y = 4) 
  (eq2 : x + 2*y = 5) 
  (eq3 : 3*x - 1.5*y + z = 7) : 
  (x + y + z) / 3 = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_solution_xyz_l358_35837


namespace NUMINAMATH_CALUDE_max_value_of_expression_l358_35845

theorem max_value_of_expression (x : ℝ) :
  (x^6) / (x^10 + 3*x^8 - 6*x^6 + 12*x^4 + 32) ≤ 1/18 ∧
  (2^6) / (2^10 + 3*2^8 - 6*2^6 + 12*2^4 + 32) = 1/18 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l358_35845


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l358_35888

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2 - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = -31 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l358_35888


namespace NUMINAMATH_CALUDE_original_profit_percentage_l358_35852

theorem original_profit_percentage 
  (original_selling_price : ℝ) 
  (additional_profit : ℝ) :
  original_selling_price = 1100 →
  additional_profit = 70 →
  ∃ (original_purchase_price : ℝ),
    (1.3 * (0.9 * original_purchase_price) = original_selling_price + additional_profit) ∧
    ((original_selling_price - original_purchase_price) / original_purchase_price * 100 = 10) :=
by sorry

end NUMINAMATH_CALUDE_original_profit_percentage_l358_35852


namespace NUMINAMATH_CALUDE_simplest_fraction_sum_l358_35825

theorem simplest_fraction_sum (a b : ℕ) : 
  a > 0 ∧ b > 0 ∧ 
  (a : ℚ) / b = 0.84375 ∧
  ∀ (c d : ℕ), c > 0 → d > 0 → (c : ℚ) / d = 0.84375 → a ≤ c ∧ b ≤ d →
  a + b = 59 := by
sorry

end NUMINAMATH_CALUDE_simplest_fraction_sum_l358_35825


namespace NUMINAMATH_CALUDE_auto_finance_fraction_l358_35880

theorem auto_finance_fraction (total_credit auto_credit finance_credit : ℝ) 
  (h1 : total_credit = 291.6666666666667)
  (h2 : auto_credit = 0.36 * total_credit)
  (h3 : finance_credit = 35) :
  finance_credit / auto_credit = 1/3 := by
sorry

end NUMINAMATH_CALUDE_auto_finance_fraction_l358_35880


namespace NUMINAMATH_CALUDE_probability_failed_math_given_failed_chinese_l358_35835

theorem probability_failed_math_given_failed_chinese 
  (failed_math : ℝ) 
  (failed_chinese : ℝ) 
  (failed_both : ℝ) 
  (h1 : failed_math = 0.16)
  (h2 : failed_chinese = 0.07)
  (h3 : failed_both = 0.04) :
  failed_both / failed_chinese = 4 / 7 := by sorry

end NUMINAMATH_CALUDE_probability_failed_math_given_failed_chinese_l358_35835


namespace NUMINAMATH_CALUDE_parts_count_l358_35847

/-- Represents the number of parts in pile A -/
def pile_a : ℕ := sorry

/-- Represents the number of parts in pile B -/
def pile_b : ℕ := sorry

/-- The condition that transferring 15 parts from A to B makes them equal -/
axiom equal_after_a_to_b : pile_a - 15 = pile_b + 15

/-- The condition that transferring 15 parts from B to A makes A three times B -/
axiom triple_after_b_to_a : pile_a + 15 = 3 * (pile_b - 15)

/-- The theorem stating the original number in pile A and the total number of parts -/
theorem parts_count : pile_a = 75 ∧ pile_a + pile_b = 120 := by sorry

end NUMINAMATH_CALUDE_parts_count_l358_35847


namespace NUMINAMATH_CALUDE_tagalong_boxes_per_case_l358_35891

theorem tagalong_boxes_per_case 
  (total_boxes : ℕ) 
  (total_cases : ℕ) 
  (h1 : total_boxes = 36) 
  (h2 : total_cases = 3) 
  (h3 : total_cases > 0) : 
  total_boxes / total_cases = 12 := by
sorry

end NUMINAMATH_CALUDE_tagalong_boxes_per_case_l358_35891


namespace NUMINAMATH_CALUDE_labourer_monthly_income_labourer_monthly_income_proof_l358_35838

/-- Proves that the monthly income of a labourer is 78 given specific expenditure patterns --/
theorem labourer_monthly_income : ℝ → Prop :=
  fun monthly_income =>
    let first_period_months : ℕ := 6
    let second_period_months : ℕ := 4
    let first_period_expenditure : ℝ := 85
    let second_period_expenditure : ℝ := 60
    let savings : ℝ := 30
    
    -- First period: fell into debt
    (monthly_income * first_period_months < first_period_expenditure * first_period_months) ∧
    
    -- Second period: cleared debt and saved
    (monthly_income * second_period_months = 
      second_period_expenditure * second_period_months + 
      (first_period_expenditure * first_period_months - monthly_income * first_period_months) + 
      savings) →
    
    monthly_income = 78

theorem labourer_monthly_income_proof : labourer_monthly_income 78 := by
  sorry

end NUMINAMATH_CALUDE_labourer_monthly_income_labourer_monthly_income_proof_l358_35838


namespace NUMINAMATH_CALUDE_circle_line_intersection_l358_35832

-- Define the circle C
def circle_equation (x y : ℝ) : Prop :=
  (x - 2)^2 + (y + 3)^2 = 13

-- Define the line l
def line_equation (x y θ : ℝ) : Prop :=
  ∃ t, x = 4 + t * Real.cos θ ∧ y = t * Real.sin θ

-- Define the intersection condition
def intersects_at_two_points (C : ℝ → ℝ → Prop) (l : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∃ x₁ y₁ x₂ y₂ θ, 
    C x₁ y₁ ∧ C x₂ y₂ ∧ 
    l x₁ y₁ θ ∧ l x₂ y₂ θ ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

-- Define the distance condition
def distance_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = 16

-- Main theorem
theorem circle_line_intersection :
  intersects_at_two_points circle_equation line_equation →
  (∃ x₁ y₁ x₂ y₂ θ, 
    circle_equation x₁ y₁ ∧ circle_equation x₂ y₂ ∧
    line_equation x₁ y₁ θ ∧ line_equation x₂ y₂ θ ∧
    distance_condition x₁ y₁ x₂ y₂) →
  ∃ k, k = 0 ∨ k = -12/5 :=
sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l358_35832


namespace NUMINAMATH_CALUDE_k_range_l358_35859

/-- Piecewise function f(x) -/
noncomputable def f (k a x : ℝ) : ℝ :=
  if x ≥ 0 then k^2 * x + a^2 - k
  else x^2 + (a^2 + 4*a) * x + (2-a)^2

/-- Condition for the existence of a unique nonzero x₂ for any nonzero x₁ -/
def unique_nonzero_solution (k a : ℝ) : Prop :=
  ∀ x₁ : ℝ, x₁ ≠ 0 → ∃! x₂ : ℝ, x₂ ≠ 0 ∧ x₂ ≠ x₁ ∧ f k a x₂ = f k a x₁

theorem k_range (k a : ℝ) :
  unique_nonzero_solution k a → k ∈ Set.Icc (-20) (-4) :=
by sorry

end NUMINAMATH_CALUDE_k_range_l358_35859


namespace NUMINAMATH_CALUDE_function_divisibility_property_l358_35866

def PositiveInt := {n : ℕ // n > 0}

theorem function_divisibility_property 
  (f : PositiveInt → PositiveInt) 
  (h : ∀ (m n : PositiveInt), (m.val^2 + (f n).val) ∣ (m.val * (f m).val + n.val)) :
  ∀ (n : PositiveInt), (f n).val = n.val :=
sorry

end NUMINAMATH_CALUDE_function_divisibility_property_l358_35866


namespace NUMINAMATH_CALUDE_foreign_language_ratio_l358_35830

theorem foreign_language_ratio (M F : ℕ) (h1 : M > 0) (h2 : F > 0) : 
  (3 * M + 4 * F : ℚ) / (5 * M + 6 * F) = 19 / 30 → M = F :=
by sorry

end NUMINAMATH_CALUDE_foreign_language_ratio_l358_35830


namespace NUMINAMATH_CALUDE_tax_discount_commute_petes_equals_pollys_l358_35857

/-- Proves that the order of applying tax and discount doesn't affect the final price -/
theorem tax_discount_commute (price : ℝ) (tax_rate discount_rate : ℝ) 
  (h1 : 0 ≤ tax_rate) (h2 : 0 ≤ discount_rate) (h3 : discount_rate ≤ 1) :
  price * (1 + tax_rate) * (1 - discount_rate) = price * (1 - discount_rate) * (1 + tax_rate) :=
by sorry

/-- Calculates Pete's method: tax then discount -/
def petes_method (price : ℝ) (tax_rate discount_rate : ℝ) : ℝ :=
  price * (1 + tax_rate) * (1 - discount_rate)

/-- Calculates Polly's method: discount then tax -/
def pollys_method (price : ℝ) (tax_rate discount_rate : ℝ) : ℝ :=
  price * (1 - discount_rate) * (1 + tax_rate)

/-- Proves that Pete's and Polly's methods yield the same result -/
theorem petes_equals_pollys (price : ℝ) (tax_rate discount_rate : ℝ) 
  (h1 : 0 ≤ tax_rate) (h2 : 0 ≤ discount_rate) (h3 : discount_rate ≤ 1) :
  petes_method price tax_rate discount_rate = pollys_method price tax_rate discount_rate :=
by sorry

end NUMINAMATH_CALUDE_tax_discount_commute_petes_equals_pollys_l358_35857


namespace NUMINAMATH_CALUDE_probability_half_correct_l358_35897

/-- The probability of getting exactly k successes in n trials with probability p for each trial. -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The number of questions in the test -/
def num_questions : ℕ := 20

/-- The number of choices for each question -/
def num_choices : ℕ := 3

/-- The probability of guessing a question correctly -/
def prob_correct : ℚ := 1 / num_choices

/-- The number of questions to get correct -/
def target_correct : ℕ := num_questions / 2

theorem probability_half_correct :
  binomial_probability num_questions target_correct prob_correct = 189399040 / 3486784401 := by
  sorry

end NUMINAMATH_CALUDE_probability_half_correct_l358_35897


namespace NUMINAMATH_CALUDE_restaurant_customer_prediction_l358_35890

theorem restaurant_customer_prediction 
  (breakfast_customers : ℕ) 
  (lunch_customers : ℕ) 
  (dinner_customers : ℕ) 
  (h1 : breakfast_customers = 73)
  (h2 : lunch_customers = 127)
  (h3 : dinner_customers = 87) :
  2 * (breakfast_customers + lunch_customers + dinner_customers) = 574 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_customer_prediction_l358_35890


namespace NUMINAMATH_CALUDE_negation_of_existence_exponential_cube_inequality_l358_35833

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x : ℝ, x > 0 ∧ p x) ↔ (∀ x : ℝ, x > 0 → ¬ p x) := by sorry

theorem exponential_cube_inequality :
  (¬ ∃ x : ℝ, x > 0 ∧ 3^x < x^3) ↔ (∀ x : ℝ, x > 0 → 3^x ≥ x^3) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_exponential_cube_inequality_l358_35833


namespace NUMINAMATH_CALUDE_minimum_games_for_90_percent_win_rate_l358_35864

theorem minimum_games_for_90_percent_win_rate :
  let initial_games : ℕ := 5
  let lions_initial_wins : ℕ := 3
  let eagles_initial_wins : ℕ := 2
  let target_win_rate : ℚ := 9/10
  ∃ N : ℕ,
    (N = 25) ∧
    (∀ n : ℕ, n < N →
      (eagles_initial_wins + n : ℚ) / (initial_games + n) < target_win_rate) ∧
    (eagles_initial_wins + N : ℚ) / (initial_games + N) ≥ target_win_rate :=
by sorry

end NUMINAMATH_CALUDE_minimum_games_for_90_percent_win_rate_l358_35864


namespace NUMINAMATH_CALUDE_remaining_slices_eq_ten_l358_35851

/-- The number of slices in a large pizza -/
def large_pizza_slices : ℕ := 8

/-- The number of slices in an extra-large pizza -/
def extra_large_pizza_slices : ℕ := 12

/-- The number of slices Mary eats from the large pizza -/
def slices_eaten_from_large : ℕ := 7

/-- The number of slices Mary eats from the extra-large pizza -/
def slices_eaten_from_extra_large : ℕ := 3

/-- The total number of remaining slices after Mary eats from both pizzas -/
def total_remaining_slices : ℕ := 
  (large_pizza_slices - slices_eaten_from_large) + 
  (extra_large_pizza_slices - slices_eaten_from_extra_large)

theorem remaining_slices_eq_ten : total_remaining_slices = 10 := by
  sorry

end NUMINAMATH_CALUDE_remaining_slices_eq_ten_l358_35851


namespace NUMINAMATH_CALUDE_choir_members_count_total_choir_members_l358_35899

theorem choir_members_count : ℕ → ℕ → ℕ → ℕ
  | group1, group2, group3 =>
    group1 + group2 + group3

theorem total_choir_members : 
  choir_members_count 25 30 15 = 70 := by
  sorry

end NUMINAMATH_CALUDE_choir_members_count_total_choir_members_l358_35899


namespace NUMINAMATH_CALUDE_square_equality_solution_l358_35803

theorem square_equality_solution : ∃ (N : ℕ+), (36 ^ 2 * 72 ^ 2 : ℕ) = 12 ^ 2 * N ^ 2 ∧ N = 216 := by
  sorry

end NUMINAMATH_CALUDE_square_equality_solution_l358_35803


namespace NUMINAMATH_CALUDE_cosine_in_right_triangle_l358_35844

theorem cosine_in_right_triangle (D E F : Real) (h1 : 0 < D) (h2 : 0 < E) (h3 : 0 < F) : 
  D^2 + E^2 = F^2 → D = 8 → F = 17 → Real.cos (Real.arccos (D / F)) = 15 / 17 := by
sorry

end NUMINAMATH_CALUDE_cosine_in_right_triangle_l358_35844


namespace NUMINAMATH_CALUDE_student_square_substitution_l358_35877

theorem student_square_substitution (a b : ℕ) : 
  (a + 2 * b - 3)^2 = a^2 + 4 * b^2 - 9 → a = 3 ∧ ∀ n : ℕ, (3 + 2 * n - 3)^2 = 3^2 + 4 * n^2 - 9 :=
by sorry

end NUMINAMATH_CALUDE_student_square_substitution_l358_35877


namespace NUMINAMATH_CALUDE_triangle_probability_theorem_l358_35827

/-- The number of points in the plane -/
def num_points : ℕ := 10

/-- The total number of possible segments -/
def total_segments : ℕ := (num_points * (num_points - 1)) / 2

/-- The number of segments chosen -/
def chosen_segments : ℕ := 4

/-- The probability of choosing 4 segments that form a triangle -/
def triangle_probability : ℚ := 1680 / 49665

theorem triangle_probability_theorem :
  triangle_probability = (num_points.choose 3 * (total_segments - 3)) / total_segments.choose chosen_segments :=
by sorry

end NUMINAMATH_CALUDE_triangle_probability_theorem_l358_35827
