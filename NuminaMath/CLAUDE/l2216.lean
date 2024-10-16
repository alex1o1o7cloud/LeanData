import Mathlib

namespace NUMINAMATH_CALUDE_consecutive_sum_100_l2216_221667

theorem consecutive_sum_100 (n : ℕ) : 
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 100) → n = 18 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_sum_100_l2216_221667


namespace NUMINAMATH_CALUDE_tea_blend_cost_l2216_221686

theorem tea_blend_cost (blend_ratio : ℚ) (second_tea_cost : ℚ) (blend_sell_price : ℚ) (gain_percent : ℚ) :
  blend_ratio = 5 / 3 →
  second_tea_cost = 20 →
  blend_sell_price = 21 →
  gain_percent = 12 →
  ∃ first_tea_cost : ℚ,
    first_tea_cost = 18 ∧
    (1 + gain_percent / 100) * ((blend_ratio * first_tea_cost + second_tea_cost) / (blend_ratio + 1)) = blend_sell_price :=
by sorry

end NUMINAMATH_CALUDE_tea_blend_cost_l2216_221686


namespace NUMINAMATH_CALUDE_smallest_integer_y_smallest_integer_is_negative_six_l2216_221690

theorem smallest_integer_y (y : ℤ) : (10 + 3 * y ≤ -8) ↔ (y ≤ -6) :=
  sorry

theorem smallest_integer_is_negative_six : ∃ (y : ℤ), (10 + 3 * y ≤ -8) ∧ (∀ (z : ℤ), (10 + 3 * z ≤ -8) → z ≥ y) ∧ y = -6 :=
  sorry

end NUMINAMATH_CALUDE_smallest_integer_y_smallest_integer_is_negative_six_l2216_221690


namespace NUMINAMATH_CALUDE_sequence_length_l2216_221632

theorem sequence_length (n : ℕ) (b : ℕ → ℝ) : 
  n > 0 ∧
  b 0 = 41 ∧
  b 1 = 68 ∧
  b n = 0 ∧
  (∀ k : ℕ, 1 ≤ k ∧ k < n → b (k + 1) = b (k - 1) - 4 / b k) →
  n = 698 := by
sorry

end NUMINAMATH_CALUDE_sequence_length_l2216_221632


namespace NUMINAMATH_CALUDE_incircle_center_locus_is_mid_distance_strip_l2216_221645

/-- Represents a line in a plane -/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- Represents a point in a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (a : Point)
  (b : Point)
  (c : Point)

/-- Represents the incircle of a triangle -/
structure Incircle :=
  (center : Point)
  (radius : ℝ)

/-- Three parallel lines on a plane -/
def parallel_lines : List Line := sorry

/-- A triangle with vertices on the parallel lines -/
def triangle_on_lines (t : Triangle) : Prop := sorry

/-- The incircle of a triangle -/
def incircle_of_triangle (t : Triangle) : Incircle := sorry

/-- The strip between the mid-distances of outer and middle lines -/
def mid_distance_strip : Set Point := sorry

/-- The geometric locus of incircle centers -/
def incircle_center_locus : Set Point := sorry

/-- Theorem: The geometric locus of the centers of incircles of triangles with vertices on three parallel lines
    is the strip bound by lines parallel and in the mid-distance between the outer and mid lines -/
theorem incircle_center_locus_is_mid_distance_strip :
  incircle_center_locus = mid_distance_strip := by sorry

end NUMINAMATH_CALUDE_incircle_center_locus_is_mid_distance_strip_l2216_221645


namespace NUMINAMATH_CALUDE_denis_neighbors_l2216_221685

-- Define the set of children
inductive Child : Type
  | Anya : Child
  | Borya : Child
  | Vera : Child
  | Gena : Child
  | Denis : Child

-- Define the line as a function from position (1 to 5) to Child
def Line : Type := Fin 5 → Child

-- Define what it means for two children to be next to each other
def NextTo (line : Line) (c1 c2 : Child) : Prop :=
  ∃ i : Fin 4, (line i = c1 ∧ line (i.succ) = c2) ∨ (line i = c2 ∧ line (i.succ) = c1)

-- Define the conditions
def LineConditions (line : Line) : Prop :=
  (line 0 = Child.Borya) ∧ 
  (NextTo line Child.Vera Child.Anya) ∧
  (¬ NextTo line Child.Vera Child.Gena) ∧
  (¬ NextTo line Child.Anya Child.Borya) ∧
  (¬ NextTo line Child.Anya Child.Gena) ∧
  (¬ NextTo line Child.Borya Child.Gena)

-- Theorem statement
theorem denis_neighbors 
  (line : Line) 
  (h : LineConditions line) : 
  (NextTo line Child.Denis Child.Anya) ∧ (NextTo line Child.Denis Child.Gena) :=
sorry

end NUMINAMATH_CALUDE_denis_neighbors_l2216_221685


namespace NUMINAMATH_CALUDE_bug_crawl_distance_l2216_221636

-- Define the bug's movement
def bugPath : List ℤ := [-3, -7, 0, 8]

-- Function to calculate distance between two points
def distance (a b : ℤ) : ℕ := (a - b).natAbs

-- Function to calculate total distance traveled
def totalDistance (path : List ℤ) : ℕ :=
  List.sum (List.zipWith distance path path.tail)

-- Theorem statement
theorem bug_crawl_distance :
  totalDistance bugPath = 19 := by sorry

end NUMINAMATH_CALUDE_bug_crawl_distance_l2216_221636


namespace NUMINAMATH_CALUDE_abs_sum_fraction_inequality_l2216_221682

theorem abs_sum_fraction_inequality (a b : ℝ) :
  |a + b| / (1 + |a + b|) ≤ |a| / (1 + |a|) + |b| / (1 + |b|) := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_fraction_inequality_l2216_221682


namespace NUMINAMATH_CALUDE_equation_solutions_l2216_221604

theorem equation_solutions : 
  let f (x : ℝ) := (15*x - x^2) / (x + 2) * (x + (15 - x) / (x + 2))
  ∃ (s : Set ℝ), s = {12, -3, -3 + Real.sqrt 33, -3 - Real.sqrt 33} ∧ 
    ∀ x ∈ s, f x = 54 ∧ 
    ∀ y ∉ s, f y ≠ 54 := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l2216_221604


namespace NUMINAMATH_CALUDE_license_plate_count_l2216_221634

def even_digits : Nat := 5
def consonants : Nat := 20
def vowels : Nat := 6

def license_plate_combinations : Nat :=
  even_digits * consonants * vowels * consonants

theorem license_plate_count :
  license_plate_combinations = 12000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l2216_221634


namespace NUMINAMATH_CALUDE_conjunction_false_implication_l2216_221642

theorem conjunction_false_implication : ∃ (p q : Prop), (p ∧ q → False) ∧ ¬(p → False ∧ q → False) := by sorry

end NUMINAMATH_CALUDE_conjunction_false_implication_l2216_221642


namespace NUMINAMATH_CALUDE_min_sum_parallel_vectors_l2216_221628

theorem min_sum_parallel_vectors (x y : ℝ) : 
  x > 0 → y > 0 → 
  (∃ (k : ℝ), k ≠ 0 ∧ k • (1 - x, x) = (1, -y)) →
  (∀ a b : ℝ, a > 0 → b > 0 → (∃ (k : ℝ), k ≠ 0 ∧ k • (1 - a, a) = (1, -b)) → a + b ≥ 4) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (∃ (k : ℝ), k ≠ 0 ∧ k • (1 - a, a) = (1, -b)) ∧ a + b = 4) :=
by sorry


end NUMINAMATH_CALUDE_min_sum_parallel_vectors_l2216_221628


namespace NUMINAMATH_CALUDE_a_minus_b_value_l2216_221671

-- Define the functions
def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := -4 * x + 6
def h (a b : ℝ) (x : ℝ) : ℝ := f a b (g x)

-- State the theorem
theorem a_minus_b_value (a b : ℝ) :
  (∀ x, h a b x = x - 9) →
  a - b = 29/4 :=
by sorry

end NUMINAMATH_CALUDE_a_minus_b_value_l2216_221671


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2216_221664

/-- The line equation passing through a fixed point for all real m -/
def line_equation (m x y : ℝ) : Prop :=
  (m + 2) * x + (m - 1) * y - 3 = 0

/-- The theorem stating that the line passes through (1, -1) for all real m -/
theorem line_passes_through_fixed_point :
  ∀ m : ℝ, line_equation m 1 (-1) :=
by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2216_221664


namespace NUMINAMATH_CALUDE_die_roll_probabilities_l2216_221661

def DieFaces := Finset.range 6

def roll_twice : Finset (ℕ × ℕ) :=
  DieFaces.product DieFaces

theorem die_roll_probabilities :
  let total_outcomes := (roll_twice.card : ℚ)
  let sum_at_least_nine := (roll_twice.filter (fun (a, b) => a + b ≥ 9)).card
  let tangent_to_circle := (roll_twice.filter (fun (a, b) => a^2 + b^2 = 25)).card
  let isosceles_triangle := (roll_twice.filter (fun (a, b) => 
    a = b ∨ a = 5 ∨ b = 5)).card
  (sum_at_least_nine : ℚ) / total_outcomes = 5 / 18 ∧
  (tangent_to_circle : ℚ) / total_outcomes = 1 / 18 ∧
  (isosceles_triangle : ℚ) / total_outcomes = 7 / 18 :=
by sorry

end NUMINAMATH_CALUDE_die_roll_probabilities_l2216_221661


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_four_l2216_221612

theorem fraction_zero_implies_x_negative_four (x : ℝ) :
  (|x| - 4) / (4 - x) = 0 → x = -4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_four_l2216_221612


namespace NUMINAMATH_CALUDE_axis_of_symmetry_l2216_221617

noncomputable def f (ω φ x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem axis_of_symmetry 
  (ω φ : ℝ) 
  (h_ω : ω > 0) 
  (h_φ : 0 ≤ φ ∧ φ < Real.pi) 
  (h_even : ∀ x, f ω φ x = f ω φ (-x)) 
  (h_distance : ∃ (a b : ℝ), b - a = 4 * Real.sqrt 2 ∧ f ω φ b = f ω φ a) :
  ∃ (x : ℝ), x = 4 ∧ ∀ y, f ω φ (x + y) = f ω φ (x - y) :=
sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_l2216_221617


namespace NUMINAMATH_CALUDE_grapes_purchased_l2216_221602

/-- Proves that the number of kg of grapes purchased is 8 -/
theorem grapes_purchased (grape_price : ℕ) (mango_price : ℕ) (mango_kg : ℕ) (total_paid : ℕ) : 
  grape_price = 70 → 
  mango_price = 55 → 
  mango_kg = 9 → 
  total_paid = 1055 → 
  ∃ (grape_kg : ℕ), grape_kg * grape_price + mango_kg * mango_price = total_paid ∧ grape_kg = 8 :=
by sorry

end NUMINAMATH_CALUDE_grapes_purchased_l2216_221602


namespace NUMINAMATH_CALUDE_sets_equality_l2216_221611

-- Define the sets M, N, and P
def M : Set ℝ := {x | ∃ n : ℤ, x = n}
def N : Set ℝ := {x | ∃ n : ℤ, x = n / 2}
def P : Set ℝ := {x | ∃ n : ℤ, x = n + 1/2}

-- Theorem statement
theorem sets_equality : N = M ∪ P := by sorry

end NUMINAMATH_CALUDE_sets_equality_l2216_221611


namespace NUMINAMATH_CALUDE_transform_equation_l2216_221683

theorem transform_equation (x m : ℝ) : 
  (x^2 + 4*x = m) ∧ ((x + 2)^2 = 5) → m = 1 := by
sorry

end NUMINAMATH_CALUDE_transform_equation_l2216_221683


namespace NUMINAMATH_CALUDE_distance_AB_when_parallel_coordinates_C_when_perpendicular_l2216_221691

-- Define the points A, B, C in the Cartesian coordinate system
def A (a : ℝ) : ℝ × ℝ := (-2, a + 1)
def B (a : ℝ) : ℝ × ℝ := (a - 1, 4)
def C (b : ℝ) : ℝ × ℝ := (b - 2, b)

-- Define the condition that AB is parallel to x-axis
def AB_parallel_x (a : ℝ) : Prop := (A a).2 = (B a).2

-- Define the condition that CD is perpendicular to x-axis
def CD_perpendicular_x (b : ℝ) : Prop := (C b).1 = b - 2

-- Define the condition that CD = 1
def CD_length_1 (b : ℝ) : Prop := (C b).2 - 0 = 1 ∨ (C b).2 - 0 = -1

-- Theorem for part 1
theorem distance_AB_when_parallel (a : ℝ) :
  AB_parallel_x a → (B a).1 - (A a).1 = 4 :=
sorry

-- Theorem for part 2
theorem coordinates_C_when_perpendicular (b : ℝ) :
  CD_perpendicular_x b ∧ CD_length_1 b →
  C b = (-1, 1) ∨ C b = (-3, -1) :=
sorry

end NUMINAMATH_CALUDE_distance_AB_when_parallel_coordinates_C_when_perpendicular_l2216_221691


namespace NUMINAMATH_CALUDE_functional_equation_solutions_l2216_221669

theorem functional_equation_solutions (f : ℤ → ℤ) :
  (∀ a b c : ℤ, a + b + c = 0 →
    f a ^ 2 + f b ^ 2 + f c ^ 2 = 2 * f a * f b + 2 * f b * f c + 2 * f c * f a) →
  (∀ x, f x = 0) ∨
  (∃ k : ℤ, ∀ x, f x = if x % 2 = 0 then 0 else k) ∨
  (∃ k : ℤ, ∀ x, f x = if x % 4 = 0 then 0 else if x % 4 = 2 then k else k) ∨
  (∃ k : ℤ, ∀ x, f x = k * x ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solutions_l2216_221669


namespace NUMINAMATH_CALUDE_school_workbooks_calculation_l2216_221637

/-- The number of workbooks a school should buy given the number of classes,
    workbooks per class, and spare workbooks. -/
def total_workbooks (num_classes : ℕ) (workbooks_per_class : ℕ) (spare_workbooks : ℕ) : ℕ :=
  num_classes * workbooks_per_class + spare_workbooks

/-- Theorem stating that the total number of workbooks the school should buy
    is equal to 25 * 144 + 80, given the specific conditions of the problem. -/
theorem school_workbooks_calculation :
  total_workbooks 25 144 80 = 25 * 144 + 80 := by
  sorry

end NUMINAMATH_CALUDE_school_workbooks_calculation_l2216_221637


namespace NUMINAMATH_CALUDE_max_mice_caught_max_mice_achievable_l2216_221633

/-- Production Possibility Frontier for a male kitten -/
def male_ppf (k : ℝ) : ℝ := 80 - 4 * k

/-- Production Possibility Frontier for a female kitten -/
def female_ppf (k : ℝ) : ℝ := 16 - 0.25 * k

/-- The maximum number of mice that can be caught by any combination of two kittens -/
def max_mice : ℝ := 160

/-- Theorem stating that the maximum number of mice that can be caught by any combination
    of two kittens is 160 -/
theorem max_mice_caught :
  ∀ k₁ k₂ : ℝ, k₁ ≥ 0 → k₂ ≥ 0 →
  (male_ppf k₁ + male_ppf k₂ ≤ max_mice) ∧
  (male_ppf k₁ + female_ppf k₂ ≤ max_mice) ∧
  (female_ppf k₁ + female_ppf k₂ ≤ max_mice) :=
sorry

/-- Theorem stating that there exist values of k₁ and k₂ for which the maximum is achieved -/
theorem max_mice_achievable :
  ∃ k₁ k₂ : ℝ, k₁ ≥ 0 ∧ k₂ ≥ 0 ∧ male_ppf k₁ + male_ppf k₂ = max_mice :=
sorry

end NUMINAMATH_CALUDE_max_mice_caught_max_mice_achievable_l2216_221633


namespace NUMINAMATH_CALUDE_soup_ingredients_weights_l2216_221643

/-- Represents the ingredients of the soup --/
structure SoupIngredients where
  fat : ℝ
  onion : ℝ
  potatoes : ℝ
  grain : ℝ
  water : ℝ

/-- The conditions of the soup recipe --/
def SoupConditions (s : SoupIngredients) : Prop :=
  s.water = s.grain + s.potatoes + s.onion + s.fat ∧
  s.grain = s.potatoes + s.onion + s.fat ∧
  s.potatoes = s.onion + s.fat ∧
  s.fat = s.onion / 2 ∧
  s.water + s.grain + s.potatoes + s.onion + s.fat = 12

/-- The theorem stating the correct weights of the ingredients --/
theorem soup_ingredients_weights :
  ∃ (s : SoupIngredients),
    SoupConditions s ∧
    s.fat = 0.5 ∧
    s.onion = 1 ∧
    s.potatoes = 1.5 ∧
    s.grain = 3 ∧
    s.water = 6 :=
  sorry

end NUMINAMATH_CALUDE_soup_ingredients_weights_l2216_221643


namespace NUMINAMATH_CALUDE_binomial_congruence_l2216_221692

theorem binomial_congruence (p m n : ℕ) (hp : Prime p) (h_mn : m ≥ n) :
  (Nat.choose (p * m) (p * n)) ≡ (Nat.choose m n) [MOD p] := by
  sorry

end NUMINAMATH_CALUDE_binomial_congruence_l2216_221692


namespace NUMINAMATH_CALUDE_new_students_admitted_l2216_221639

theorem new_students_admitted (initial_students_per_section : ℕ) 
                               (new_sections : ℕ)
                               (final_total_sections : ℕ)
                               (final_students_per_section : ℕ) :
  initial_students_per_section = 23 →
  new_sections = 5 →
  final_total_sections = 20 →
  final_students_per_section = 19 →
  (final_total_sections * final_students_per_section) - 
  ((final_total_sections - new_sections) * initial_students_per_section) = 35 :=
by sorry

end NUMINAMATH_CALUDE_new_students_admitted_l2216_221639


namespace NUMINAMATH_CALUDE_magic_8_ball_probability_l2216_221626

theorem magic_8_ball_probability :
  let n : ℕ := 7  -- total number of questions
  let k : ℕ := 4  -- number of positive answers
  let p : ℚ := 3/7  -- probability of a positive answer
  Nat.choose n k * p^k * (1-p)^(n-k) = 181440/823543 :=
by sorry

end NUMINAMATH_CALUDE_magic_8_ball_probability_l2216_221626


namespace NUMINAMATH_CALUDE_intersection_condition_l2216_221614

-- Define the sets A and B
def A (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + a * p.1 - p.2 + 2 = 0}
def B : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2 * p.1 - p.2 + 1 = 0 ∧ p.1 > 0}

-- State the theorem
theorem intersection_condition (a : ℝ) :
  (A a ∩ B).Nonempty ↔ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_intersection_condition_l2216_221614


namespace NUMINAMATH_CALUDE_lucy_crayons_l2216_221659

/-- Given that Willy has 1400 crayons and 1110 more crayons than Lucy, 
    prove that Lucy has 290 crayons. -/
theorem lucy_crayons (willy_crayons : ℕ) (difference : ℕ) (lucy_crayons : ℕ) 
  (h1 : willy_crayons = 1400) 
  (h2 : difference = 1110) 
  (h3 : willy_crayons = lucy_crayons + difference) : 
  lucy_crayons = 290 := by
  sorry

end NUMINAMATH_CALUDE_lucy_crayons_l2216_221659


namespace NUMINAMATH_CALUDE_eleven_billion_scientific_notation_l2216_221605

def billion : ℕ := 10^9

theorem eleven_billion_scientific_notation : 
  11 * billion = 11 * 10^9 ∧ 11 * 10^9 = 1.1 * 10^10 :=
sorry

end NUMINAMATH_CALUDE_eleven_billion_scientific_notation_l2216_221605


namespace NUMINAMATH_CALUDE_arithmetic_sequence_theorem_l2216_221629

def is_arithmetic_sequence (a b c d : ℚ) : Prop :=
  b - a = c - b ∧ c - b = d - c

def sum_is_26 (a b c d : ℚ) : Prop :=
  a + b + c + d = 26

def middle_product_is_40 (b c : ℚ) : Prop :=
  b * c = 40

theorem arithmetic_sequence_theorem (a b c d : ℚ) :
  is_arithmetic_sequence a b c d →
  sum_is_26 a b c d →
  middle_product_is_40 b c →
  ((a = 2 ∧ b = 5 ∧ c = 8 ∧ d = 11) ∨ (a = 11 ∧ b = 8 ∧ c = 5 ∧ d = 2)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_theorem_l2216_221629


namespace NUMINAMATH_CALUDE_invisibility_elixir_combinations_l2216_221601

/-- The number of magical herbs available for the invisibility elixir. -/
def num_herbs : ℕ := 4

/-- The number of enchanted gems available for the invisibility elixir. -/
def num_gems : ℕ := 6

/-- The number of herb-gem combinations that cancel each other's magic. -/
def num_cancelling_combinations : ℕ := 3

/-- The number of successful combinations for the invisibility elixir. -/
def num_successful_combinations : ℕ := num_herbs * num_gems - num_cancelling_combinations

theorem invisibility_elixir_combinations :
  num_successful_combinations = 21 := by sorry

end NUMINAMATH_CALUDE_invisibility_elixir_combinations_l2216_221601


namespace NUMINAMATH_CALUDE_volunteers_selection_theorem_l2216_221695

theorem volunteers_selection_theorem :
  let n : ℕ := 5  -- Total number of volunteers
  let k : ℕ := 2  -- Number of people to be sent to each location
  let locations : ℕ := 2  -- Number of locations
  Nat.choose n k * Nat.choose (n - k) k = 30 := by
  sorry

end NUMINAMATH_CALUDE_volunteers_selection_theorem_l2216_221695


namespace NUMINAMATH_CALUDE_product_of_fractions_l2216_221609

theorem product_of_fractions (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a/b + b/c + c/a + b/a + c/b + a/c = 12) :
  (a/b + b/c + c/a) * (b/a + c/b + a/c) = 72 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l2216_221609


namespace NUMINAMATH_CALUDE_max_value_complex_expression_l2216_221619

theorem max_value_complex_expression (z : ℂ) (h : Complex.abs z = Real.sqrt 2) :
  Complex.abs ((z - 1)^2 * (z + 1)) ≤ 4 * Real.sqrt 2 ∧
  ∃ w : ℂ, Complex.abs w = Real.sqrt 2 ∧ Complex.abs ((w - 1)^2 * (w + 1)) = 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_complex_expression_l2216_221619


namespace NUMINAMATH_CALUDE_circular_board_area_l2216_221620

/-- The area of a circular wooden board that rolls forward for 10 revolutions
    and advances exactly 62.8 meters is π square meters. -/
theorem circular_board_area (revolutions : ℕ) (distance : ℝ) (area : ℝ) :
  revolutions = 10 →
  distance = 62.8 →
  area = π →
  area = (distance / (2 * revolutions : ℝ))^2 * π := by
  sorry

end NUMINAMATH_CALUDE_circular_board_area_l2216_221620


namespace NUMINAMATH_CALUDE_pants_to_shirts_ratio_l2216_221668

/-- Proves that the ratio of pants to shirts is 1/2 given the problem conditions --/
theorem pants_to_shirts_ratio :
  ∀ (num_pants : ℕ),
  (10 * 6 + num_pants * 8 = 100) →
  (num_pants : ℚ) / 10 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_pants_to_shirts_ratio_l2216_221668


namespace NUMINAMATH_CALUDE_frank_reading_speed_l2216_221616

/-- The number of days Frank took to finish all books -/
def total_days : ℕ := 492

/-- The total number of books Frank read -/
def total_books : ℕ := 41

/-- The number of days it took Frank to finish each book -/
def days_per_book : ℚ := total_days / total_books

/-- Theorem stating that Frank took 12 days to finish each book -/
theorem frank_reading_speed : days_per_book = 12 := by
  sorry

end NUMINAMATH_CALUDE_frank_reading_speed_l2216_221616


namespace NUMINAMATH_CALUDE_original_price_of_discounted_items_l2216_221625

theorem original_price_of_discounted_items 
  (num_items : ℕ) 
  (discount_rate : ℚ) 
  (total_paid : ℚ) 
  (h1 : num_items = 6)
  (h2 : discount_rate = 1/2)
  (h3 : total_paid = 60) :
  (total_paid / (1 - discount_rate)) / num_items = 20 := by
sorry

end NUMINAMATH_CALUDE_original_price_of_discounted_items_l2216_221625


namespace NUMINAMATH_CALUDE_infinitely_many_primes_of_year_2022_l2216_221680

/-- A prime p is a prime of the year 2022 if there exists a positive integer n 
    such that p^2022 divides n^2022 + 2022 -/
def IsPrimeOfYear2022 (p : Nat) : Prop :=
  Nat.Prime p ∧ ∃ n : Nat, n > 0 ∧ (p^2022 ∣ n^2022 + 2022)

/-- There are infinitely many primes of the year 2022 -/
theorem infinitely_many_primes_of_year_2022 :
  ∀ N : Nat, ∃ p : Nat, p > N ∧ IsPrimeOfYear2022 p := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_of_year_2022_l2216_221680


namespace NUMINAMATH_CALUDE_two_box_marble_problem_l2216_221676

/-- Represents a box containing marbles -/
structure MarbleBox where
  total : ℕ
  black : ℕ
  white : ℕ
  h_sum : total = black + white

/-- The probability of drawing a specific color marble from a box -/
def drawProbability (box : MarbleBox) (color : ℕ) : ℚ :=
  color / box.total

theorem two_box_marble_problem (box1 box2 : MarbleBox) : 
  box1.total + box2.total = 25 →
  drawProbability box1 box1.black * drawProbability box2 box2.black = 27/50 →
  drawProbability box1 box1.white * drawProbability box2 box2.white = 1/25 := by
sorry

end NUMINAMATH_CALUDE_two_box_marble_problem_l2216_221676


namespace NUMINAMATH_CALUDE_integral_p_equals_one_l2216_221693

noncomputable def p (α : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then 0 else α * Real.exp (-α * x)

theorem integral_p_equals_one (α : ℝ) (h : α > 0) :
  ∫ (x : ℝ), p α x = 1 := by sorry

end NUMINAMATH_CALUDE_integral_p_equals_one_l2216_221693


namespace NUMINAMATH_CALUDE_multiply_polynomial_equals_difference_of_powers_l2216_221627

theorem multiply_polynomial_equals_difference_of_powers (x : ℝ) :
  (x^4 + 25*x^2 + 625) * (x^2 - 25) = x^6 - 15625 := by
  sorry

end NUMINAMATH_CALUDE_multiply_polynomial_equals_difference_of_powers_l2216_221627


namespace NUMINAMATH_CALUDE_mary_fruits_left_l2216_221697

/-- Calculates the total number of fruits left after eating some. -/
def fruits_left (initial_apples initial_oranges initial_blueberries eaten : ℕ) : ℕ :=
  (initial_apples - eaten) + (initial_oranges - eaten) + (initial_blueberries - eaten)

/-- Proves that Mary has 26 fruits left after eating one of each. -/
theorem mary_fruits_left : fruits_left 14 9 6 1 = 26 := by
  sorry

end NUMINAMATH_CALUDE_mary_fruits_left_l2216_221697


namespace NUMINAMATH_CALUDE_x_value_l2216_221673

def A (x : ℝ) : Set ℝ := {2, x, x^2 - 30}

theorem x_value (x : ℝ) (h : -5 ∈ A x) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2216_221673


namespace NUMINAMATH_CALUDE_squareable_numbers_l2216_221613

-- Define what it means for a number to be squareable
def is_squareable (n : ℕ) : Prop :=
  ∃ (perm : Fin n → Fin n), Function.Bijective perm ∧
    ∀ i : Fin n, ∃ k : ℕ, (perm i).val + i.val + 1 = k^2

-- Theorem statement
theorem squareable_numbers :
  is_squareable 9 ∧ is_squareable 15 ∧ ¬is_squareable 7 ∧ ¬is_squareable 11 :=
sorry

end NUMINAMATH_CALUDE_squareable_numbers_l2216_221613


namespace NUMINAMATH_CALUDE_wood_length_proof_l2216_221665

/-- The initial length of the wood Tom cut. -/
def initial_length : ℝ := 143

/-- The length cut off from the initial piece of wood. -/
def cut_length : ℝ := 25

/-- The original length of other boards before cutting. -/
def other_boards_original : ℝ := 125

/-- The length cut off from other boards. -/
def other_boards_cut : ℝ := 7

theorem wood_length_proof :
  initial_length - cut_length > other_boards_original - other_boards_cut ∧
  initial_length = 143 := by
  sorry

#check wood_length_proof

end NUMINAMATH_CALUDE_wood_length_proof_l2216_221665


namespace NUMINAMATH_CALUDE_min_value_of_p_l2216_221662

-- Define the polynomial p
def p (a b : ℝ) : ℝ := a^2 + 2*b^2 + 2*a + 4*b + 2008

-- Theorem stating the minimum value of p
theorem min_value_of_p :
  ∃ (min : ℝ), min = 2005 ∧ ∀ (a b : ℝ), p a b ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_p_l2216_221662


namespace NUMINAMATH_CALUDE_roller_derby_teams_l2216_221623

/-- The number of teams competing in a roller derby --/
def number_of_teams (members_per_team : ℕ) (skates_per_member : ℕ) (laces_per_skate : ℕ) (total_laces : ℕ) : ℕ :=
  total_laces / (members_per_team * skates_per_member * laces_per_skate)

/-- Theorem stating that the number of teams competing is 4 --/
theorem roller_derby_teams : number_of_teams 10 2 3 240 = 4 := by
  sorry

end NUMINAMATH_CALUDE_roller_derby_teams_l2216_221623


namespace NUMINAMATH_CALUDE_intersection_area_theorem_l2216_221679

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with given side length -/
structure Cube where
  sideLength : ℝ

/-- Defines the position of points P, Q, R on the cube edges -/
structure PointsOnCube where
  cube : Cube
  P : Point3D
  Q : Point3D
  R : Point3D

/-- Calculates the area of the intersection polygon -/
def intersectionArea (c : Cube) (pts : PointsOnCube) : ℝ :=
  sorry

/-- Theorem stating the area of the intersection polygon -/
theorem intersection_area_theorem (c : Cube) (pts : PointsOnCube) :
  c.sideLength = 30 ∧
  pts.P.x = 10 ∧ pts.P.y = 0 ∧ pts.P.z = 0 ∧
  pts.Q.x = 30 ∧ pts.Q.y = 0 ∧ pts.Q.z = 20 ∧
  pts.R.x = 30 ∧ pts.R.y = 5 ∧ pts.R.z = 30 →
  intersectionArea c pts = 450 := by
  sorry

end NUMINAMATH_CALUDE_intersection_area_theorem_l2216_221679


namespace NUMINAMATH_CALUDE_problem_statement_l2216_221648

theorem problem_statement : 
  (-1)^2023 + (8 : ℝ)^(1/3) - 2 * (1/4 : ℝ)^(1/2) + |Real.sqrt 3 - 2| = 2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2216_221648


namespace NUMINAMATH_CALUDE_third_term_range_l2216_221641

/-- A sequence of positive real numbers satisfying certain conditions -/
def SpecialSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧
  (a 1 = 1) ∧
  (∀ n : ℕ, n ≥ 1 → a (n + 1)^2 + a n^2 < (5/2) * a (n + 1) * a n) ∧
  (a 2 = 3/2) ∧
  (a 4 = 4)

/-- The third term of the sequence is within the range (2, 3) -/
theorem third_term_range (a : ℕ → ℝ) (h : SpecialSequence a) :
  ∃ x, a 3 = x ∧ 2 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_third_term_range_l2216_221641


namespace NUMINAMATH_CALUDE_blue_to_red_ratio_is_four_to_one_l2216_221621

/-- Represents the number of pencils of each color and the total number of pencils. -/
structure PencilCounts where
  total : ℕ
  red : ℕ
  yellow : ℕ
  blue : ℕ
  green : ℕ

/-- Theorem stating that under given conditions, the ratio of blue to red pencils is 4:1. -/
theorem blue_to_red_ratio_is_four_to_one (p : PencilCounts)
    (h_total : p.total = 160)
    (h_red : p.red = 20)
    (h_yellow : p.yellow = 40)
    (h_green : p.green = p.red + p.blue) :
    p.blue / p.red = 4 := by
  sorry

end NUMINAMATH_CALUDE_blue_to_red_ratio_is_four_to_one_l2216_221621


namespace NUMINAMATH_CALUDE_edna_distance_fraction_l2216_221688

/-- Proves that Edna ran 2/3 of Mary's distance given the conditions of the problem -/
theorem edna_distance_fraction (total_distance : ℝ) (mary_fraction edna_fraction lucy_fraction : ℚ)
  (h1 : total_distance = 24)
  (h2 : mary_fraction = 3/8)
  (h3 : lucy_fraction = 5/6 * edna_fraction)
  (h4 : mary_fraction * total_distance = lucy_fraction * edna_fraction * total_distance + 4) :
  edna_fraction = 2/3 := by
sorry


end NUMINAMATH_CALUDE_edna_distance_fraction_l2216_221688


namespace NUMINAMATH_CALUDE_xy_value_l2216_221624

theorem xy_value (x y : ℝ) (h : Real.sqrt (2 * x - 4) + |y - 1| = 0) : x * y = 2 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2216_221624


namespace NUMINAMATH_CALUDE_equation_roots_l2216_221600

/-- Given an equation with two real roots, prove the range of m and a specific case. -/
theorem equation_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁^2 - 2*m*x₁ = -m^2 + 2*x₁ ∧ x₂^2 - 2*m*x₂ = -m^2 + 2*x₂ ∧ x₁ ≠ x₂) → 
  (m ≥ -1/2 ∧ 
   (∀ x₁ x₂ : ℝ, x₁^2 - 2*m*x₁ = -m^2 + 2*x₁ → x₂^2 - 2*m*x₂ = -m^2 + 2*x₂ → |x₁| = x₂ → m = -1/2)) :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_l2216_221600


namespace NUMINAMATH_CALUDE_minimum_additional_wins_l2216_221672

def puppy_cost : ℕ := 1000
def weekly_prize : ℕ := 100
def initial_wins : ℕ := 2

theorem minimum_additional_wins : 
  ∃ (n : ℕ), n = (puppy_cost - initial_wins * weekly_prize) / weekly_prize ∧ 
  n * weekly_prize + initial_wins * weekly_prize ≥ puppy_cost ∧
  ∀ m : ℕ, m < n → m * weekly_prize + initial_wins * weekly_prize < puppy_cost :=
by sorry

end NUMINAMATH_CALUDE_minimum_additional_wins_l2216_221672


namespace NUMINAMATH_CALUDE_continuous_of_strictly_increasing_and_continuous_compose_l2216_221607

/-- Given a strictly increasing function f: ℝ → ℝ where f ∘ f is continuous, f is continuous. -/
theorem continuous_of_strictly_increasing_and_continuous_compose (f : ℝ → ℝ)
  (h_increasing : StrictMono f) (h_continuous_compose : Continuous (f ∘ f)) :
  Continuous f := by
  sorry

end NUMINAMATH_CALUDE_continuous_of_strictly_increasing_and_continuous_compose_l2216_221607


namespace NUMINAMATH_CALUDE_chocolate_chip_count_l2216_221687

/-- Proves the number of chocolate chips in a batch of cookies given specific conditions --/
theorem chocolate_chip_count (total_cookies : ℕ) (avg_pieces_per_cookie : ℚ) 
  (h1 : total_cookies = 48)
  (h2 : avg_pieces_per_cookie = 3)
  (h3 : ∀ (chips m_and_ms : ℕ), m_and_ms = chips / 3 → 
    chips + m_and_ms = total_cookies * avg_pieces_per_cookie) :
  ∃ (chips : ℕ), chips = 108 ∧ 
    (∃ (m_and_ms : ℕ), m_and_ms = chips / 3 ∧ 
      chips + m_and_ms = total_cookies * avg_pieces_per_cookie) :=
by sorry

end NUMINAMATH_CALUDE_chocolate_chip_count_l2216_221687


namespace NUMINAMATH_CALUDE_less_than_implies_less_than_minus_one_l2216_221681

theorem less_than_implies_less_than_minus_one {a b : ℝ} (h : a < b) : a - 1 < b - 1 := by
  sorry

end NUMINAMATH_CALUDE_less_than_implies_less_than_minus_one_l2216_221681


namespace NUMINAMATH_CALUDE_last_three_average_l2216_221675

theorem last_three_average (numbers : List ℝ) : 
  numbers.length = 7 → 
  numbers.sum / 7 = 60 → 
  (numbers.take 4).sum / 4 = 55 → 
  (numbers.drop 4).sum / 3 = 200 / 3 := by
sorry

end NUMINAMATH_CALUDE_last_three_average_l2216_221675


namespace NUMINAMATH_CALUDE_circle_diameter_l2216_221677

/-- Given a circle with radius 7 cm, prove that its diameter is 14 cm -/
theorem circle_diameter (r : ℝ) (h : r = 7) : 2 * r = 14 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_l2216_221677


namespace NUMINAMATH_CALUDE_triangle_side_length_l2216_221638

theorem triangle_side_length (AB : ℝ) (cosA sinC : ℝ) (angleADB : ℝ) :
  AB = 30 →
  angleADB = 90 →
  cosA = 4/5 →
  sinC = 2/5 →
  ∃ (AD : ℝ), AD = 24 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2216_221638


namespace NUMINAMATH_CALUDE_triangle_tangent_product_l2216_221653

theorem triangle_tangent_product (A B C : ℝ) (a b c : ℝ) :
  (A + B + C = π) →  -- Sum of angles in a triangle
  (a > 0) → (b > 0) → (c > 0) →  -- Positive side lengths
  (a / (2 * Real.sin (A / 2)) = b / (2 * Real.sin (B / 2))) →  -- Sine law
  (b / (2 * Real.sin (B / 2)) = c / (2 * Real.sin (C / 2))) →  -- Sine law
  (a + c = 2 * b) →  -- Given condition
  (Real.tan (A / 2) * Real.tan (C / 2) = 1 / 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_tangent_product_l2216_221653


namespace NUMINAMATH_CALUDE_matrix_inverse_proof_l2216_221646

def A : Matrix (Fin 2) (Fin 2) ℤ := !![7, -5; -4, 3]
def A_inv : Matrix (Fin 2) (Fin 2) ℤ := !![3, 5; 4, 7]

theorem matrix_inverse_proof :
  IsUnit (Matrix.det A) ∧ A * A_inv = 1 ∧ A_inv * A = 1 := by
  sorry

end NUMINAMATH_CALUDE_matrix_inverse_proof_l2216_221646


namespace NUMINAMATH_CALUDE_cube_root_not_always_two_l2216_221610

theorem cube_root_not_always_two (x : ℝ) (h : x^2 = 64) : 
  ∃ y, y^3 = x ∧ y ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_not_always_two_l2216_221610


namespace NUMINAMATH_CALUDE_log_equation_solution_l2216_221650

theorem log_equation_solution : 
  ∃ y : ℝ, (Real.log y - 3 * Real.log 5 = -3) ∧ (y = 0.125) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2216_221650


namespace NUMINAMATH_CALUDE_matrix_inverse_proof_l2216_221698

theorem matrix_inverse_proof :
  let N : Matrix (Fin 3) (Fin 3) ℝ := !![4, 2.5, 0; 3, 2, 0; 0, 0, 1]
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![-4, 5, 0; 6, -8, 0; 0, 0, 1]
  N * A = (1 : Matrix (Fin 3) (Fin 3) ℝ) := by sorry

end NUMINAMATH_CALUDE_matrix_inverse_proof_l2216_221698


namespace NUMINAMATH_CALUDE_sum_a_b_equals_six_l2216_221658

theorem sum_a_b_equals_six (a b : ℝ) 
  (eq1 : 3 * a + 5 * b = 22) 
  (eq2 : 4 * a + 2 * b = 20) : 
  a + b = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_a_b_equals_six_l2216_221658


namespace NUMINAMATH_CALUDE_mary_warmth_duration_l2216_221654

/-- The number of sticks of wood produced by chopping up furniture -/
def sticksFromFurniture (chairs tables stools : ℕ) : ℕ :=
  6 * chairs + 9 * tables + 2 * stools

/-- The number of hours Mary can keep warm given a certain amount of wood -/
def hoursWarm (totalSticks burningRate : ℕ) : ℕ :=
  totalSticks / burningRate

/-- Theorem: Mary can keep warm for 34 hours with the wood from 18 chairs, 6 tables, and 4 stools -/
theorem mary_warmth_duration :
  let totalSticks := sticksFromFurniture 18 6 4
  let burningRate := 5
  hoursWarm totalSticks burningRate = 34 := by
  sorry

end NUMINAMATH_CALUDE_mary_warmth_duration_l2216_221654


namespace NUMINAMATH_CALUDE_base7_subtraction_l2216_221655

/-- Converts a base-7 number to decimal --/
def toDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 7^i) 0

/-- Converts a decimal number to base-7 --/
def toBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

theorem base7_subtraction :
  let a := [5, 5, 2, 1]  -- 1255 in base 7
  let b := [2, 3, 4]     -- 432 in base 7
  let c := [1, 2, 5]     -- 521 in base 7
  toBase7 (toDecimal a - toDecimal b) = c := by sorry

end NUMINAMATH_CALUDE_base7_subtraction_l2216_221655


namespace NUMINAMATH_CALUDE_program_result_l2216_221666

/-- The program's operation on input n -/
def program (n : ℝ) : ℝ := n^2 + 3*n - (2*n^2 - n)

/-- Theorem stating that the program's result equals -n^2 + 4n for any real n -/
theorem program_result (n : ℝ) : program n = -n^2 + 4*n := by
  sorry

end NUMINAMATH_CALUDE_program_result_l2216_221666


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2216_221644

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 3 * a 5 * a 7 * a 9 * a 11 = 243 →
  a 9^2 / a 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2216_221644


namespace NUMINAMATH_CALUDE_difference_of_extreme_valid_numbers_l2216_221678

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ 
  (n.digits 10).count 2 = 3 ∧ 
  (n.digits 10).count 0 = 1

def largest_valid_number : ℕ := 2220
def smallest_valid_number : ℕ := 2022

theorem difference_of_extreme_valid_numbers :
  largest_valid_number - smallest_valid_number = 198 ∧
  is_valid_number largest_valid_number ∧
  is_valid_number smallest_valid_number ∧
  (∀ n : ℕ, is_valid_number n → n ≤ largest_valid_number) ∧
  (∀ n : ℕ, is_valid_number n → smallest_valid_number ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_difference_of_extreme_valid_numbers_l2216_221678


namespace NUMINAMATH_CALUDE_odd_function_inequality_l2216_221603

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_inequality (f : ℝ → ℝ) (a b : ℝ) 
  (h_odd : IsOdd f) (h_ineq : f a > f b) : f (-a) < f (-b) := by
  sorry

end NUMINAMATH_CALUDE_odd_function_inequality_l2216_221603


namespace NUMINAMATH_CALUDE_mia_weight_l2216_221635

/-- 
Given two people, Anna and Mia, with the following conditions:
1. The sum of their weights is 220 pounds.
2. The difference between Mia's weight and Anna's weight is twice Anna's weight.
This theorem proves that Mia's weight is 165 pounds.
-/
theorem mia_weight (anna_weight mia_weight : ℝ) 
  (sum_condition : anna_weight + mia_weight = 220)
  (difference_condition : mia_weight - anna_weight = 2 * anna_weight) :
  mia_weight = 165 := by
sorry

end NUMINAMATH_CALUDE_mia_weight_l2216_221635


namespace NUMINAMATH_CALUDE_instances_in_one_hour_l2216_221696

/-- The number of seconds in one minute -/
def seconds_per_minute : ℕ := 60

/-- The number of minutes in one hour -/
def minutes_per_hour : ℕ := 60

/-- The interval in seconds at which the device records data -/
def recording_interval : ℕ := 5

/-- Proves that the number of 5-second intervals in one hour is equal to 720 -/
theorem instances_in_one_hour :
  (seconds_per_minute * minutes_per_hour) / recording_interval = 720 := by
  sorry

end NUMINAMATH_CALUDE_instances_in_one_hour_l2216_221696


namespace NUMINAMATH_CALUDE_joshua_crates_count_l2216_221670

def bottles_per_crate : ℕ := 12
def total_bottles : ℕ := 130
def unpacked_bottles : ℕ := 10

theorem joshua_crates_count :
  (total_bottles - unpacked_bottles) / bottles_per_crate = 10 := by
  sorry

end NUMINAMATH_CALUDE_joshua_crates_count_l2216_221670


namespace NUMINAMATH_CALUDE_total_balloons_sam_dan_l2216_221606

-- Define the initial quantities
def sam_initial : ℝ := 46.5
def fred_receive : ℝ := 10.2
def gaby_receive : ℝ := 3.3
def dan_balloons : ℝ := 16.4

-- Define Sam's remaining balloons after distribution
def sam_remaining : ℝ := sam_initial - fred_receive - gaby_receive

-- Theorem statement
theorem total_balloons_sam_dan : 
  sam_remaining + dan_balloons = 49.4 := by sorry

end NUMINAMATH_CALUDE_total_balloons_sam_dan_l2216_221606


namespace NUMINAMATH_CALUDE_min_stable_stories_l2216_221656

/-- Represents a domino placement on a rectangular grid --/
structure DominoPlacement :=
  (width : Nat) -- Width of the rectangle
  (height : Nat) -- Height of the rectangle
  (dominoes : Nat) -- Number of dominoes per story

/-- Represents a tower of domino placements --/
structure DominoTower :=
  (base : DominoPlacement)
  (stories : Nat)

/-- Defines when a domino tower is considered stable --/
def isStable (tower : DominoTower) : Prop :=
  ∀ (x y : ℚ), 0 ≤ x ∧ x < tower.base.width ∧ 0 ≤ y ∧ y < tower.base.height →
    ∃ (s : Nat), s < tower.stories ∧ 
      ∃ (dx dy : ℚ), (0 ≤ dx ∧ dx < 2 ∧ 0 ≤ dy ∧ dy < 1) ∧
        (⌊x⌋ ≤ x - dx ∧ x - dx < ⌊x⌋ + 1) ∧
        (⌊y⌋ ≤ y - dy ∧ y - dy < ⌊y⌋ + 1)

/-- The main theorem stating the minimum number of stories for a stable tower --/
theorem min_stable_stories (tower : DominoTower) 
  (h_width : tower.base.width = 10)
  (h_height : tower.base.height = 11)
  (h_dominoes : tower.base.dominoes = 55) :
  (isStable tower ↔ tower.stories ≥ 5) :=
sorry

end NUMINAMATH_CALUDE_min_stable_stories_l2216_221656


namespace NUMINAMATH_CALUDE_translation_theorem_l2216_221615

/-- Represents a point in 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a translation in 2D space -/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- Apply a translation to a point -/
def apply_translation (p : Point) (t : Translation) : Point :=
  { x := p.x + t.dx, y := p.y + t.dy }

theorem translation_theorem :
  let A : Point := { x := -1, y := 0 }
  let B : Point := { x := 1, y := 2 }
  let A1 : Point := { x := 2, y := -1 }
  let t : Translation := { dx := A1.x - A.x, dy := A1.y - A.y }
  let B1 : Point := apply_translation B t
  B1 = { x := 4, y := 1 } := by
  sorry

end NUMINAMATH_CALUDE_translation_theorem_l2216_221615


namespace NUMINAMATH_CALUDE_more_b_shoes_than_a_l2216_221689

/-- Given the conditions about shoe boxes, prove that there are 640 more pairs of (B) shoes than (A) shoes. -/
theorem more_b_shoes_than_a : 
  ∀ (pairs_per_box : ℕ) (num_a_boxes : ℕ) (num_b_boxes : ℕ),
  pairs_per_box = 20 →
  num_a_boxes = 8 →
  num_b_boxes = 5 * num_a_boxes →
  num_b_boxes * pairs_per_box - num_a_boxes * pairs_per_box = 640 :=
by
  sorry

#check more_b_shoes_than_a

end NUMINAMATH_CALUDE_more_b_shoes_than_a_l2216_221689


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2216_221674

/-- Given an arithmetic sequence {a_n} where a_2 = 3 and a_6 = 13, 
    prove that the common difference is 5/2 -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℚ) 
  (h1 : a 2 = 3) 
  (h2 : a 6 = 13) 
  (h_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) :
  ∃ d : ℚ, d = 5/2 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2216_221674


namespace NUMINAMATH_CALUDE_circle_properties_l2216_221651

-- Define the circle equations
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 5*x - 6*y + 5 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - x + 6*y - 6 = 0

-- Define points A and B
def pointA : ℝ × ℝ := (1, 0)
def pointB : ℝ × ℝ := (0, 1)

-- Define the chord length on x-axis
def chordLength : ℝ := 6

-- Theorem statement
theorem circle_properties :
  (∀ (x y : ℝ), circle1 x y → ((x = pointA.1 ∧ y = pointA.2) ∨ (x = pointB.1 ∧ y = pointB.2))) ∧
  (∀ (x y : ℝ), circle2 x y → ((x = pointA.1 ∧ y = pointA.2) ∨ (x = pointB.1 ∧ y = pointB.2))) ∧
  (∃ (x1 x2 : ℝ), x1 < x2 ∧ circle1 x1 0 ∧ circle1 x2 0 ∧ x2 - x1 = chordLength) ∧
  (∃ (x1 x2 : ℝ), x1 < x2 ∧ circle2 x1 0 ∧ circle2 x2 0 ∧ x2 - x1 = chordLength) :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l2216_221651


namespace NUMINAMATH_CALUDE_max_value_quadratic_l2216_221608

theorem max_value_quadratic (s t : ℝ) (h : t = 4) :
  ∃ (max : ℝ), max = 46 ∧ ∀ s, -2 * s^2 + 24 * s + 3 * t - 38 ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l2216_221608


namespace NUMINAMATH_CALUDE_part_one_part_two_l2216_221647

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a| + |2*x + 1|

-- Part I
theorem part_one :
  {x : ℝ | f 1 x ≤ 3} = {x : ℝ | -1 ≤ x ∧ x ≤ 1} :=
sorry

-- Part II
theorem part_two :
  {a : ℝ | ∃ x ≥ a, f a x ≤ 2*a + x} = {a : ℝ | a ≥ 1} :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2216_221647


namespace NUMINAMATH_CALUDE_triangle_roots_range_l2216_221618

theorem triangle_roots_range (m : ℝ) : 
  (∃ x y z : ℝ, (x - 1) * (x^2 - 2*x + m) = 0 ∧ 
                (y - 1) * (y^2 - 2*y + m) = 0 ∧ 
                (z - 1) * (z^2 - 2*z + m) = 0 ∧
                x + y > z ∧ y + z > x ∧ z + x > y) ↔ 
  (3/4 < m ∧ m ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_triangle_roots_range_l2216_221618


namespace NUMINAMATH_CALUDE_share_ratio_l2216_221657

theorem share_ratio (total amount : ℕ) (a_share : ℕ) : 
  total = 366 → a_share = 122 → a_share / (total - a_share) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_share_ratio_l2216_221657


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l2216_221640

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) : 
  let side := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  4 * side = 52 := by
sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l2216_221640


namespace NUMINAMATH_CALUDE_four_digit_number_expansion_l2216_221684

/-- Represents a four-digit number with digits a, b, c, and d -/
def four_digit_number (a b c d : ℕ) : ℕ := 1000 * a + 100 * b + 10 * c + d

/-- Theorem stating that a four-digit number with digits a, b, c, and d
    is equal to 1000a + 100b + 10c + d -/
theorem four_digit_number_expansion {a b c d : ℕ} (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) (h4 : d < 10) :
  four_digit_number a b c d = 1000 * a + 100 * b + 10 * c + d := by
  sorry

end NUMINAMATH_CALUDE_four_digit_number_expansion_l2216_221684


namespace NUMINAMATH_CALUDE_line_slope_l2216_221699

/-- Given a line with equation y = -5x + 9, its slope is -5 -/
theorem line_slope (x y : ℝ) : y = -5 * x + 9 → (∃ m b : ℝ, y = m * x + b ∧ m = -5) := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l2216_221699


namespace NUMINAMATH_CALUDE_financial_audit_equation_l2216_221652

theorem financial_audit_equation (p v : ℂ) : 
  (7 * p - v = 23000) → (v = 50 + 250 * Complex.I) → 
  (p = 3292.857 + 35.714 * Complex.I) := by
sorry

end NUMINAMATH_CALUDE_financial_audit_equation_l2216_221652


namespace NUMINAMATH_CALUDE_other_number_proof_l2216_221663

theorem other_number_proof (x y : ℕ+) 
  (h1 : Nat.lcm x y = 2640)
  (h2 : Nat.gcd x y = 24)
  (h3 : x = 240) :
  y = 264 := by
  sorry

end NUMINAMATH_CALUDE_other_number_proof_l2216_221663


namespace NUMINAMATH_CALUDE_not_valid_base_5_l2216_221631

/-- Given a base k and a sequence of digits, determines if it's a valid representation in that base -/
def is_valid_base_k_number (k : ℕ) (digits : List ℕ) : Prop :=
  ∀ d ∈ digits, d < k

/-- The theorem states that 32501 is not a valid base-5 number -/
theorem not_valid_base_5 :
  ¬ (is_valid_base_k_number 5 [3, 2, 5, 0, 1]) :=
sorry

end NUMINAMATH_CALUDE_not_valid_base_5_l2216_221631


namespace NUMINAMATH_CALUDE_point_on_line_trig_identity_l2216_221630

/-- 
Given a point P with coordinates (cos θ, sin θ) that lies on the line 2x + y = 0,
prove that cos 2θ + (1/2) sin 2θ = -1.
-/
theorem point_on_line_trig_identity (θ : Real) 
  (h : 2 * Real.cos θ + Real.sin θ = 0) : 
  Real.cos (2 * θ) + (1/2) * Real.sin (2 * θ) = -1 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_trig_identity_l2216_221630


namespace NUMINAMATH_CALUDE_xyz_equals_five_l2216_221622

theorem xyz_equals_five
  (a b c x y z : ℂ)
  (nonzero_a : a ≠ 0)
  (nonzero_b : b ≠ 0)
  (nonzero_c : c ≠ 0)
  (nonzero_x : x ≠ 0)
  (nonzero_y : y ≠ 0)
  (nonzero_z : z ≠ 0)
  (eq_a : a = (b + c) / (x - 2))
  (eq_b : b = (a + c) / (y - 2))
  (eq_c : c = (a + b) / (z - 2))
  (sum_xy_xz_yz : x * y + x * z + y * z = 5)
  (sum_x_y_z : x + y + z = 3) :
  x * y * z = 5 := by
sorry

end NUMINAMATH_CALUDE_xyz_equals_five_l2216_221622


namespace NUMINAMATH_CALUDE_lunks_for_apples_l2216_221660

/-- Exchange rate between lunks and kunks -/
def lunk_to_kunk_rate : ℚ := 6 / 4

/-- Exchange rate between kunks and apples -/
def kunk_to_apple_rate : ℚ := 5 / 3

/-- Number of apples to purchase -/
def apples_to_buy : ℕ := 20

/-- Calculate the number of lunks needed to buy a given number of apples -/
def lunks_needed (apples : ℕ) : ℚ :=
  (apples : ℚ) / kunk_to_apple_rate / lunk_to_kunk_rate

theorem lunks_for_apples :
  lunks_needed apples_to_buy = 8 := by
  sorry

end NUMINAMATH_CALUDE_lunks_for_apples_l2216_221660


namespace NUMINAMATH_CALUDE_dolls_in_small_box_l2216_221694

theorem dolls_in_small_box :
  let big_box_count : ℕ := 5
  let big_box_dolls : ℕ := 7
  let small_box_count : ℕ := 9
  let total_dolls : ℕ := 71
  let small_box_dolls : ℕ := (total_dolls - big_box_count * big_box_dolls) / small_box_count
  small_box_dolls = 4 := by sorry

end NUMINAMATH_CALUDE_dolls_in_small_box_l2216_221694


namespace NUMINAMATH_CALUDE_opposite_of_negative_fraction_l2216_221649

theorem opposite_of_negative_fraction :
  -(-(1 / 2023)) = 1 / 2023 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_fraction_l2216_221649
