import Mathlib

namespace NUMINAMATH_CALUDE_bob_pennies_bob_pennies_proof_l2087_208769

theorem bob_pennies : ℕ → ℕ → Prop :=
  fun a b =>
    (b + 1 = 4 * (a - 1)) ∧
    (b - 1 = 3 * (a + 1)) →
    b = 31

-- The proof is omitted
theorem bob_pennies_proof : bob_pennies 9 31 := by sorry

end NUMINAMATH_CALUDE_bob_pennies_bob_pennies_proof_l2087_208769


namespace NUMINAMATH_CALUDE_average_after_removing_two_numbers_l2087_208770

/-- Given a list of 50 numbers with an average of 62, prove that if we remove 45 and 55 from the list,
    the average of the remaining numbers is 62.5 -/
theorem average_after_removing_two_numbers
  (numbers : List ℝ)
  (h_count : numbers.length = 50)
  (h_avg : numbers.sum / numbers.length = 62)
  (h_contains_45 : 45 ∈ numbers)
  (h_contains_55 : 55 ∈ numbers) :
  let remaining := numbers.filter (λ x => x ≠ 45 ∧ x ≠ 55)
  remaining.sum / remaining.length = 62.5 := by
sorry


end NUMINAMATH_CALUDE_average_after_removing_two_numbers_l2087_208770


namespace NUMINAMATH_CALUDE_f_properties_l2087_208744

-- Define the function f
def f (x a b : ℝ) : ℝ := |x + a| - |x - b|

-- Main theorem
theorem f_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  -- Part I: Solution set for f(x) > 2 when a = 1 and b = 2
  (∀ x, f x 1 2 > 2 ↔ x > 3/2) ∧
  -- Part II: If max(f) = 3, then min(1/a + 2/b) = (3 + 2√2)/3
  (∃ x, ∀ y, f y a b ≤ f x a b) ∧ (∀ y, f y a b ≤ 3) →
    ∀ a' b', a' > 0 → b' > 0 → 1/a' + 2/b' ≥ (3 + 2*Real.sqrt 2)/3 ∧
    ∃ a'' b'', a'' > 0 ∧ b'' > 0 ∧ 1/a'' + 2/b'' = (3 + 2*Real.sqrt 2)/3 :=
by sorry


end NUMINAMATH_CALUDE_f_properties_l2087_208744


namespace NUMINAMATH_CALUDE_quadratic_trinomial_exists_l2087_208709

/-- A quadratic trinomial satisfying the given conditions -/
def f (a c : ℝ) (m : ℝ) : ℝ := a * m^2 - a * m + c

theorem quadratic_trinomial_exists :
  ∃ (a c : ℝ), a ≠ 0 ∧ f a c 4 = 13 :=
sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_exists_l2087_208709


namespace NUMINAMATH_CALUDE_sin_135_degrees_l2087_208785

theorem sin_135_degrees : Real.sin (135 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_135_degrees_l2087_208785


namespace NUMINAMATH_CALUDE_almond_croissant_price_l2087_208704

/-- The price of an almond croissant given Harrison's croissant buying habits -/
theorem almond_croissant_price :
  let regular_price : ℚ := 7/2  -- $3.50
  let weeks_in_year : ℕ := 52
  let total_spent : ℚ := 468
  let almond_price : ℚ := (total_spent - weeks_in_year * regular_price) / weeks_in_year
  almond_price = 11/2  -- $5.50
  := by sorry

end NUMINAMATH_CALUDE_almond_croissant_price_l2087_208704


namespace NUMINAMATH_CALUDE_cylinder_height_l2087_208763

/-- The height of a cylindrical tin given its diameter and volume -/
theorem cylinder_height (d V : ℝ) (h_d : d = 4) (h_V : V = 20) :
  let r := d / 2
  let h := V / (π * r^2)
  h = 5 / π :=
by sorry

end NUMINAMATH_CALUDE_cylinder_height_l2087_208763


namespace NUMINAMATH_CALUDE_extreme_value_theorem_l2087_208731

theorem extreme_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = 5 * x * y) :
  ∀ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 5 * a * b → 4 * x + 3 * y ≥ 4 * a + 3 * b ∧ 4 * x + 3 * y ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_theorem_l2087_208731


namespace NUMINAMATH_CALUDE_square_field_area_specific_field_area_l2087_208742

/-- The area of a square field given diagonal travel time and speed -/
theorem square_field_area (travel_time : Real) (speed : Real) : Real :=
  let diagonal_length : Real := speed * (travel_time / 60)
  let side_length : Real := (diagonal_length * 1000) / Real.sqrt 2
  side_length * side_length

/-- Proof of the specific field area -/
theorem specific_field_area : 
  square_field_area 2 3 = 5000 := by
  sorry

end NUMINAMATH_CALUDE_square_field_area_specific_field_area_l2087_208742


namespace NUMINAMATH_CALUDE_sequence_properties_l2087_208730

def sequence_a (n : ℕ) : ℝ := 3 * (2^n - 1)

def sequence_b (n : ℕ) : ℝ := sequence_a n + 3

def sum_S (n : ℕ) : ℝ := 2 * sequence_a n - 3 * n

theorem sequence_properties :
  (∀ n : ℕ, sum_S (n + 1) = 2 * sequence_a (n + 1) - 3 * (n + 1)) ∧
  sequence_a 1 = 3 ∧
  sequence_a 2 = 9 ∧
  sequence_a 3 = 21 ∧
  (∀ n : ℕ, sequence_b (n + 1) = 2 * sequence_b n) ∧
  (∀ n : ℕ, sequence_a n = 3 * (2^n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l2087_208730


namespace NUMINAMATH_CALUDE_tv_price_increase_l2087_208797

theorem tv_price_increase (x : ℝ) : 
  (1 + x / 100) * 1.2 = 1 + 56.00000000000001 / 100 → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_tv_price_increase_l2087_208797


namespace NUMINAMATH_CALUDE_a_equals_3_necessary_not_sufficient_l2087_208707

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Condition for two lines to be parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a * l2.c ≠ l1.c * l2.a

/-- The first line ax-2y-1=0 -/
def line1 (a : ℝ) : Line :=
  { a := a, b := -2, c := -1 }

/-- The second line 6x-4y+c=0 -/
def line2 (c : ℝ) : Line :=
  { a := 6, b := -4, c := c }

theorem a_equals_3_necessary_not_sufficient :
  (∀ c, parallel (line1 3) (line2 c)) ∧
  (∃ a c, a ≠ 3 ∧ parallel (line1 a) (line2 c)) :=
sorry

end NUMINAMATH_CALUDE_a_equals_3_necessary_not_sufficient_l2087_208707


namespace NUMINAMATH_CALUDE_function_equality_l2087_208722

theorem function_equality (f g h k : ℝ → ℝ) (a b : ℝ) 
  (h1 : ∀ x, f x = (x - 1) * g x + 3)
  (h2 : ∀ x, f x = (x + 1) * h x + 1)
  (h3 : ∀ x, f x = (x^2 - 1) * k x + a * x + b) :
  a = 1 ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_function_equality_l2087_208722


namespace NUMINAMATH_CALUDE_min_value_of_arithmetic_sequence_l2087_208794

/-- An arithmetic sequence with positive terms -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d ∧ a n > 0

theorem min_value_of_arithmetic_sequence (a : ℕ → ℝ) 
  (h_arith : ArithmeticSequence a) 
  (h_eq : 2 * a 4 + a 3 - 2 * a 2 - a 1 = 8) :
  ∃ m : ℝ, m = 12 * Real.sqrt 3 ∧ ∀ q : ℝ, 2 * a 5 + a 4 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_arithmetic_sequence_l2087_208794


namespace NUMINAMATH_CALUDE_friday_ice_cream_amount_l2087_208791

/-- The amount of ice cream eaten on Friday night, given the total amount eaten over two nights and the amount eaten on Saturday night. -/
theorem friday_ice_cream_amount (total : ℝ) (saturday : ℝ) (h1 : total = 3.5) (h2 : saturday = 0.25) :
  total - saturday = 3.25 := by
  sorry

end NUMINAMATH_CALUDE_friday_ice_cream_amount_l2087_208791


namespace NUMINAMATH_CALUDE_rostov_true_supporters_l2087_208723

structure Island where
  total_population : ℕ
  knights : ℕ
  liars : ℕ
  rostov_yes : ℕ
  zenit_yes : ℕ
  lokomotiv_yes : ℕ
  cska_yes : ℕ

def percentage (n : ℕ) (total : ℕ) : ℚ :=
  (n : ℚ) / (total : ℚ) * 100

theorem rostov_true_supporters (i : Island) :
  i.knights + i.liars = i.total_population →
  percentage i.rostov_yes i.total_population = 40 →
  percentage i.zenit_yes i.total_population = 30 →
  percentage i.lokomotiv_yes i.total_population = 50 →
  percentage i.cska_yes i.total_population = 0 →
  percentage i.liars i.total_population = 10 →
  percentage (i.rostov_yes - i.liars) i.total_population = 30 := by
  sorry

#check rostov_true_supporters

end NUMINAMATH_CALUDE_rostov_true_supporters_l2087_208723


namespace NUMINAMATH_CALUDE_not_all_altitudes_inside_l2087_208738

-- Define a triangle
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

-- Define an altitude of a triangle
def altitude (t : Triangle) (v : Fin 3) : Set (ℝ × ℝ) :=
  sorry

-- Define the property of being inside a triangle
def inside_triangle (t : Triangle) (p : ℝ × ℝ) : Prop :=
  sorry

-- Define different types of triangles
def is_acute_triangle (t : Triangle) : Prop :=
  sorry

def is_right_triangle (t : Triangle) : Prop :=
  sorry

def is_obtuse_triangle (t : Triangle) : Prop :=
  sorry

-- The theorem to be proven
theorem not_all_altitudes_inside : ¬ ∀ (t : Triangle), 
  (∀ (v : Fin 3), ∀ (p : ℝ × ℝ), p ∈ altitude t v → inside_triangle t p) :=
sorry

end NUMINAMATH_CALUDE_not_all_altitudes_inside_l2087_208738


namespace NUMINAMATH_CALUDE_lunch_cost_per_person_l2087_208760

theorem lunch_cost_per_person (total_price : ℝ) (num_people : ℕ) (gratuity_rate : ℝ) : 
  total_price = 207 ∧ num_people = 15 ∧ gratuity_rate = 0.15 →
  (total_price / (1 + gratuity_rate)) / num_people = 12 := by
sorry

end NUMINAMATH_CALUDE_lunch_cost_per_person_l2087_208760


namespace NUMINAMATH_CALUDE_mrsHiltFramePerimeter_l2087_208702

/-- An irregular octagon with specified side lengths -/
structure IrregularOctagon where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  side5 : ℝ
  side6 : ℝ
  side7 : ℝ
  side8 : ℝ

/-- Calculate the perimeter of an irregular octagon -/
def perimeter (o : IrregularOctagon) : ℝ :=
  o.side1 + o.side2 + o.side3 + o.side4 + o.side5 + o.side6 + o.side7 + o.side8

/-- Mrs. Hilt's irregular octagonal picture frame -/
def mrsHiltFrame : IrregularOctagon :=
  { side1 := 10
    side2 := 9
    side3 := 11
    side4 := 6
    side5 := 7
    side6 := 2
    side7 := 3
    side8 := 4 }

/-- Theorem: The perimeter of Mrs. Hilt's irregular octagonal picture frame is 52 inches -/
theorem mrsHiltFramePerimeter : perimeter mrsHiltFrame = 52 := by
  sorry

end NUMINAMATH_CALUDE_mrsHiltFramePerimeter_l2087_208702


namespace NUMINAMATH_CALUDE_simplify_algebraic_expression_l2087_208754

theorem simplify_algebraic_expression (a : ℝ) : 2*a - 7*a + 4*a = -a := by
  sorry

end NUMINAMATH_CALUDE_simplify_algebraic_expression_l2087_208754


namespace NUMINAMATH_CALUDE_odd_even_f_l2087_208780

def f (n : ℕ) : ℕ := (n * (Nat.totient n)) / 2

theorem odd_even_f (n : ℕ) (h : n > 1) :
  (Odd (f n) ∧ Even (f (2015 * n))) ↔ Odd n ∧ n > 1 := by sorry

end NUMINAMATH_CALUDE_odd_even_f_l2087_208780


namespace NUMINAMATH_CALUDE_cos_540_degrees_l2087_208719

theorem cos_540_degrees : Real.cos (540 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cos_540_degrees_l2087_208719


namespace NUMINAMATH_CALUDE_positive_reals_inequality_l2087_208768

theorem positive_reals_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (1 + a * b) / (1 + a) + (1 + b * c) / (1 + b) + (1 + c * a) / (1 + c) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_positive_reals_inequality_l2087_208768


namespace NUMINAMATH_CALUDE_max_a_is_correct_l2087_208775

/-- The quadratic function f(x) = -x^2 + 2x - 2 -/
def f (x : ℝ) : ℝ := -x^2 + 2*x - 2

/-- The maximum value of a for which f(x) is increasing when x ≤ a -/
def max_a : ℝ := 1

theorem max_a_is_correct :
  ∀ a : ℝ, (∀ x y : ℝ, x ≤ y → y ≤ a → f x ≤ f y) → a ≤ max_a :=
by sorry

end NUMINAMATH_CALUDE_max_a_is_correct_l2087_208775


namespace NUMINAMATH_CALUDE_fish_pond_population_l2087_208724

/-- Represents the total number of fish in a pond using the mark and recapture method. -/
def totalFishInPond (initialMarked : ℕ) (secondCatch : ℕ) (markedInSecondCatch : ℕ) : ℕ :=
  (initialMarked * secondCatch) / markedInSecondCatch

/-- Theorem stating that under the given conditions, the total number of fish in the pond is 2400. -/
theorem fish_pond_population :
  let initialMarked : ℕ := 80
  let secondCatch : ℕ := 150
  let markedInSecondCatch : ℕ := 5
  totalFishInPond initialMarked secondCatch markedInSecondCatch = 2400 :=
by sorry


end NUMINAMATH_CALUDE_fish_pond_population_l2087_208724


namespace NUMINAMATH_CALUDE_mary_story_characters_l2087_208766

theorem mary_story_characters (total : ℕ) (a b c g d e f h : ℕ) : 
  total = 360 →
  a = total / 3 →
  b = (total - a) / 4 →
  c = (total - a - b) / 5 →
  g = (total - a - b - c) / 6 →
  d + e + f + h = total - a - b - c - g →
  d = 3 * e →
  f = 2 * e →
  h = f →
  d = 45 :=
by sorry

end NUMINAMATH_CALUDE_mary_story_characters_l2087_208766


namespace NUMINAMATH_CALUDE_notebook_distribution_l2087_208729

theorem notebook_distribution (S : ℕ) 
  (h1 : S > 0)
  (h2 : S * (S / 8) = (S / 2) * 16) : 
  S * (S / 8) = 512 := by
sorry

end NUMINAMATH_CALUDE_notebook_distribution_l2087_208729


namespace NUMINAMATH_CALUDE_count_integer_lengths_specific_triangle_l2087_208715

/-- Represents a right triangle with integer side lengths -/
structure RightTriangle where
  de : ℕ
  ef : ℕ
  df : ℕ
  is_right : de^2 + ef^2 = df^2

/-- Counts the number of distinct integer lengths of line segments from E to DF -/
def count_integer_lengths (t : RightTriangle) : ℕ :=
  let max_length := max t.de t.ef
  let min_length := min t.de t.ef
  max_length - min_length + 1

/-- The main theorem -/
theorem count_integer_lengths_specific_triangle :
  ∃ (t : RightTriangle), t.de = 12 ∧ t.ef = 16 ∧ count_integer_lengths t = 5 :=
sorry

end NUMINAMATH_CALUDE_count_integer_lengths_specific_triangle_l2087_208715


namespace NUMINAMATH_CALUDE_fraction_sum_equals_thirteen_fourths_l2087_208720

theorem fraction_sum_equals_thirteen_fourths (a b : ℝ) (h1 : a = 3) (h2 : b = 1) :
  5 / (a + b) + 2 = 13 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_thirteen_fourths_l2087_208720


namespace NUMINAMATH_CALUDE_younger_person_age_l2087_208782

/-- Given two people with an age difference of 20 years, where 15 years ago the elder was twice as old as the younger, prove that the younger person's current age is 35 years. -/
theorem younger_person_age (younger elder : ℕ) : 
  elder - younger = 20 →
  elder - 15 = 2 * (younger - 15) →
  younger = 35 := by
  sorry

end NUMINAMATH_CALUDE_younger_person_age_l2087_208782


namespace NUMINAMATH_CALUDE_symmetric_point_xoy_plane_l2087_208734

/-- A point in 3D space --/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Symmetry with respect to the xoy plane --/
def symmetricXOY (p : Point3D) : Point3D :=
  { x := p.x, y := p.y, z := -p.z }

theorem symmetric_point_xoy_plane :
  let M : Point3D := { x := 2, y := 5, z := 8 }
  let N : Point3D := symmetricXOY M
  N = { x := 2, y := 5, z := -8 } := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_xoy_plane_l2087_208734


namespace NUMINAMATH_CALUDE_newspaper_pages_read_l2087_208725

theorem newspaper_pages_read (jairus_pages arniel_pages total_pages : ℕ) : 
  jairus_pages = 20 →
  arniel_pages = 2 * jairus_pages + 2 →
  total_pages = jairus_pages + arniel_pages →
  total_pages = 62 := by
sorry

end NUMINAMATH_CALUDE_newspaper_pages_read_l2087_208725


namespace NUMINAMATH_CALUDE_maple_leaf_high_basketball_score_l2087_208765

theorem maple_leaf_high_basketball_score :
  ∀ (x : ℚ) (y : ℕ),
    x > 0 →
    (1/3 : ℚ) * x + (3/8 : ℚ) * x + 18 + y = x →
    10 ≤ y →
    y ≤ 30 →
    y = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_maple_leaf_high_basketball_score_l2087_208765


namespace NUMINAMATH_CALUDE_distribute_four_to_three_l2087_208745

/-- The number of ways to distribute volunteers to venues -/
def distribute_volunteers (num_volunteers : ℕ) (num_venues : ℕ) : ℕ :=
  if num_venues > num_volunteers then 0
  else if num_venues = 1 then 1
  else (num_volunteers - 1).choose (num_venues - 1) * num_venues.factorial

/-- Theorem: Distributing 4 volunteers to 3 venues yields 36 schemes -/
theorem distribute_four_to_three :
  distribute_volunteers 4 3 = 36 := by
  sorry

#eval distribute_volunteers 4 3

end NUMINAMATH_CALUDE_distribute_four_to_three_l2087_208745


namespace NUMINAMATH_CALUDE_kindergarten_cats_count_l2087_208728

/-- Represents the number of children in each category in the kindergarten. -/
structure KindergartenPets where
  total : ℕ
  dogsOnly : ℕ
  bothPets : ℕ
  catsOnly : ℕ

/-- Calculates the total number of children with cats in the kindergarten. -/
def childrenWithCats (k : KindergartenPets) : ℕ :=
  k.catsOnly + k.bothPets

/-- Theorem stating the number of children with cats in the kindergarten. -/
theorem kindergarten_cats_count (k : KindergartenPets)
    (h1 : k.total = 30)
    (h2 : k.dogsOnly = 18)
    (h3 : k.bothPets = 6)
    (h4 : k.total = k.dogsOnly + k.catsOnly + k.bothPets) :
    childrenWithCats k = 12 := by
  sorry

#check kindergarten_cats_count

end NUMINAMATH_CALUDE_kindergarten_cats_count_l2087_208728


namespace NUMINAMATH_CALUDE_same_color_prob_is_eleven_thirty_sixths_l2087_208792

/-- A die with 12 sides and specific color distribution -/
structure TwelveSidedDie :=
  (red : ℕ)
  (blue : ℕ)
  (green : ℕ)
  (golden : ℕ)
  (total_sides : red + blue + green + golden = 12)

/-- The probability of two dice showing the same color -/
def same_color_probability (d : TwelveSidedDie) : ℚ :=
  (d.red^2 + d.blue^2 + d.green^2 + d.golden^2) / 144

/-- Theorem stating the probability of two 12-sided dice showing the same color -/
theorem same_color_prob_is_eleven_thirty_sixths :
  ∀ d : TwelveSidedDie,
  d.red = 3 → d.blue = 5 → d.green = 3 → d.golden = 1 →
  same_color_probability d = 11 / 36 :=
sorry

end NUMINAMATH_CALUDE_same_color_prob_is_eleven_thirty_sixths_l2087_208792


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l2087_208786

theorem quadratic_equation_properties (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 3 * x₁^2 - 9 * x₁ + c = 0 ∧ 3 * x₂^2 - 9 * x₂ + c = 0) →
  (c < 6.75 ∧ (x₁ + x₂) / 2 = 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l2087_208786


namespace NUMINAMATH_CALUDE_point_P_coordinates_l2087_208795

def M : ℝ × ℝ := (-2, 7)
def N : ℝ × ℝ := (10, -2)

def vector (A B : ℝ × ℝ) : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

theorem point_P_coordinates :
  ∃ P : ℝ × ℝ,
    (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (M.1 + t * (N.1 - M.1), M.2 + t * (N.2 - M.2))) ∧
    vector P N = (-2 : ℝ) • (vector P M) ∧
    P = (2, 4) := by
  sorry

end NUMINAMATH_CALUDE_point_P_coordinates_l2087_208795


namespace NUMINAMATH_CALUDE_x_fourth_plus_inverse_x_fourth_l2087_208757

theorem x_fourth_plus_inverse_x_fourth (x : ℝ) (h : x ≠ 0) :
  x^2 + 1/x^2 = 5 → x^4 + 1/x^4 = 23 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_plus_inverse_x_fourth_l2087_208757


namespace NUMINAMATH_CALUDE_fraction_simplification_l2087_208737

theorem fraction_simplification (a b c d : ℕ) (h1 : a = 2637) (h2 : b = 18459) (h3 : c = 5274) (h4 : d = 36918) :
  a / b = 1 / 7 → c / d = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2087_208737


namespace NUMINAMATH_CALUDE_max_value_of_ab_l2087_208718

theorem max_value_of_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 1) :
  ab ≤ 1/8 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 2 * a₀ + b₀ = 1 ∧ a₀ * b₀ = 1/8 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_ab_l2087_208718


namespace NUMINAMATH_CALUDE_total_jelly_beans_l2087_208796

theorem total_jelly_beans (vanilla : ℕ) (grape : ℕ) : 
  vanilla = 120 → 
  grape = 5 * vanilla + 50 → 
  vanilla + grape = 770 := by
sorry

end NUMINAMATH_CALUDE_total_jelly_beans_l2087_208796


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l2087_208756

theorem trigonometric_inequality (α β γ : Real) 
  (h1 : 0 ≤ α ∧ α ≤ Real.pi / 2)
  (h2 : 0 ≤ β ∧ β ≤ Real.pi / 2)
  (h3 : 0 ≤ γ ∧ γ ≤ Real.pi / 2)
  (h4 : Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 = 1) :
  2 ≤ (1 + Real.cos α ^ 2) ^ 2 * Real.sin α ^ 4 + 
      (1 + Real.cos β ^ 2) ^ 2 * Real.sin β ^ 4 + 
      (1 + Real.cos γ ^ 2) ^ 2 * Real.sin γ ^ 4 ∧
  (1 + Real.cos α ^ 2) ^ 2 * Real.sin α ^ 4 + 
  (1 + Real.cos β ^ 2) ^ 2 * Real.sin β ^ 4 + 
  (1 + Real.cos γ ^ 2) ^ 2 * Real.sin γ ^ 4 ≤ 
  (1 + Real.cos α ^ 2) * (1 + Real.cos β ^ 2) * (1 + Real.cos γ ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l2087_208756


namespace NUMINAMATH_CALUDE_coordinates_of_P_wrt_origin_l2087_208789

-- Define a point in 2D Cartesian coordinate system
def Point := ℝ × ℝ

-- Define point P
def P : Point := (-5, 3)

-- Theorem stating that the coordinates of P with respect to the origin are (-5, 3)
theorem coordinates_of_P_wrt_origin :
  P = (-5, 3) := by sorry

end NUMINAMATH_CALUDE_coordinates_of_P_wrt_origin_l2087_208789


namespace NUMINAMATH_CALUDE_brads_running_speed_l2087_208749

/-- Prove Brad's running speed given the conditions of the problem -/
theorem brads_running_speed 
  (total_distance : ℝ) 
  (maxwells_speed : ℝ) 
  (maxwells_distance : ℝ) 
  (h1 : total_distance = 40) 
  (h2 : maxwells_speed = 3) 
  (h3 : maxwells_distance = 15) : 
  ∃ (brads_speed : ℝ), brads_speed = 5 := by
  sorry


end NUMINAMATH_CALUDE_brads_running_speed_l2087_208749


namespace NUMINAMATH_CALUDE_correct_percentage_l2087_208758

theorem correct_percentage (y : ℕ) (y_pos : y > 0) : 
  let total := 7 * y
  let incorrect := 2 * y
  let correct := total - incorrect
  (correct : ℚ) / total * 100 = 500 / 7 := by
  sorry

end NUMINAMATH_CALUDE_correct_percentage_l2087_208758


namespace NUMINAMATH_CALUDE_rational_solution_cosine_equation_l2087_208767

theorem rational_solution_cosine_equation (q : ℚ) 
  (h1 : 0 < q) (h2 : q < 1) 
  (h3 : Real.cos (3 * Real.pi * q) + 2 * Real.cos (2 * Real.pi * q) = 0) : 
  q = 2/3 := by
sorry

end NUMINAMATH_CALUDE_rational_solution_cosine_equation_l2087_208767


namespace NUMINAMATH_CALUDE_xyz_value_l2087_208790

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 25)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 7) : 
  x * y * z = 6 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l2087_208790


namespace NUMINAMATH_CALUDE_cow_count_l2087_208764

theorem cow_count (ducks cows : ℕ) : 
  (2 * ducks + 4 * cows = 2 * (ducks + cows) + 16) → cows = 8 := by
sorry

end NUMINAMATH_CALUDE_cow_count_l2087_208764


namespace NUMINAMATH_CALUDE_lcm_problem_l2087_208777

theorem lcm_problem (n : ℕ+) (h1 : Nat.lcm 40 n = 200) (h2 : Nat.lcm n 45 = 180) : n = 100 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l2087_208777


namespace NUMINAMATH_CALUDE_problem_solution_l2087_208784

theorem problem_solution : ∃! x : ℝ, 0.8 * x + (0.2 * 0.4) = 0.56 ∧ x = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2087_208784


namespace NUMINAMATH_CALUDE_regular_tetrahedron_side_edge_length_l2087_208771

/-- A regular triangular pyramid (tetrahedron) with specific properties -/
structure RegularTetrahedron where
  /-- The length of the base edge -/
  base_edge : ℝ
  /-- The angle between side faces in radians -/
  side_face_angle : ℝ
  /-- The length of the side edges -/
  side_edge : ℝ
  /-- The base edge is 1 unit long -/
  base_edge_length : base_edge = 1
  /-- The side faces form an angle of 120° (2π/3 radians) with each other -/
  face_angle : side_face_angle = 2 * Real.pi / 3

/-- Theorem stating the length of side edges in a regular tetrahedron with given properties -/
theorem regular_tetrahedron_side_edge_length (t : RegularTetrahedron) : 
  t.side_edge = Real.sqrt 6 / 4 := by
  sorry

end NUMINAMATH_CALUDE_regular_tetrahedron_side_edge_length_l2087_208771


namespace NUMINAMATH_CALUDE_can_form_123_l2087_208716

-- Define a data type for arithmetic expressions
inductive Expr
  | num : Nat → Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr

-- Define a function to evaluate an expression
def eval : Expr → Int
  | Expr.num n => n
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.sub e1 e2 => eval e1 - eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2

-- Define a predicate to check if an expression uses all numbers exactly once
def usesAllNumbers (e : Expr) : Prop := sorry

-- Theorem stating that 123 can be formed
theorem can_form_123 : ∃ e : Expr, usesAllNumbers e ∧ eval e = 123 := by
  sorry

end NUMINAMATH_CALUDE_can_form_123_l2087_208716


namespace NUMINAMATH_CALUDE_root_properties_l2087_208761

-- Define the polynomial f
def f (x : ℝ) : ℝ := x^20 - 123*x^10 + 1

-- Define the polynomial g
def g (x : ℝ) : ℝ := x^4 + 3*x^3 + 4*x^2 + 2*x + 1

theorem root_properties (a β : ℝ) : 
  (f a = 0 → f (-a) = 0 ∧ f (1/a) = 0 ∧ f (-1/a) = 0) ∧
  (g β = 0 → g (-β) ≠ 0 ∧ g (1/β) ≠ 0 ∧ g (-1/β) ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_root_properties_l2087_208761


namespace NUMINAMATH_CALUDE_derivative_one_implies_x_is_one_l2087_208741

open Real

theorem derivative_one_implies_x_is_one (f : ℝ → ℝ) (x₀ : ℝ) :
  (f = λ x => x * log x) →
  (deriv f x₀ = 1) →
  x₀ = 1 := by
sorry

end NUMINAMATH_CALUDE_derivative_one_implies_x_is_one_l2087_208741


namespace NUMINAMATH_CALUDE_sara_wins_731_l2087_208779

/-- Represents the state of a wall in the brick removal game -/
def Wall := List Nat

/-- Calculates the nim-value of a single wall -/
def nimValue (wall : Nat) : Nat :=
  sorry

/-- Calculates the nim-sum (XOR) of a list of natural numbers -/
def nimSum (values : List Nat) : Nat :=
  sorry

/-- Determines if a given game state is a winning position for the second player -/
def isWinningForSecondPlayer (state : Wall) : Prop :=
  nimSum (state.map nimValue) = 0

/-- The main theorem stating that (7, 3, 1) is a winning position for the second player -/
theorem sara_wins_731 : isWinningForSecondPlayer [7, 3, 1] := by
  sorry

end NUMINAMATH_CALUDE_sara_wins_731_l2087_208779


namespace NUMINAMATH_CALUDE_F_is_odd_l2087_208701

-- Define the function f on the real numbers
variable (f : ℝ → ℝ)

-- Define F in terms of f
def F (f : ℝ → ℝ) (x : ℝ) : ℝ := f x - f (-x)

-- Theorem statement
theorem F_is_odd (f : ℝ → ℝ) : 
  ∀ x : ℝ, F f x = -(F f (-x)) := by
  sorry

end NUMINAMATH_CALUDE_F_is_odd_l2087_208701


namespace NUMINAMATH_CALUDE_cos_shift_symmetry_axis_l2087_208740

/-- The axis of symmetry for a cosine function shifted left by π/12 -/
theorem cos_shift_symmetry_axis (k : ℤ) :
  let f : ℝ → ℝ := λ x ↦ Real.cos (2 * (x + π / 12))
  ∀ x : ℝ, f (k * π / 2 - π / 12 - x) = f (k * π / 2 - π / 12 + x) := by
  sorry

end NUMINAMATH_CALUDE_cos_shift_symmetry_axis_l2087_208740


namespace NUMINAMATH_CALUDE_min_value_theorem_l2087_208774

theorem min_value_theorem (a : ℝ) :
  ∃ (min : ℝ), min = 8 ∧
  ∀ (x y : ℝ), x > 0 → y = -x^2 + 3 * Real.log x →
  (a - x)^2 + (a + 2 - y)^2 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2087_208774


namespace NUMINAMATH_CALUDE_teena_yoe_distance_l2087_208717

/-- Calculates the initial distance between two drivers given their speeds and future relative position --/
def initialDistance (teenaSpeed yoeSpeed : ℝ) (timeAhead : ℝ) (distanceAhead : ℝ) : ℝ :=
  (teenaSpeed - yoeSpeed) * timeAhead - distanceAhead

theorem teena_yoe_distance :
  let teenaSpeed : ℝ := 55
  let yoeSpeed : ℝ := 40
  let timeAhead : ℝ := 1.5  -- 90 minutes in hours
  let distanceAhead : ℝ := 15
  initialDistance teenaSpeed yoeSpeed timeAhead distanceAhead = 7.5 := by
  sorry

#eval initialDistance 55 40 1.5 15

end NUMINAMATH_CALUDE_teena_yoe_distance_l2087_208717


namespace NUMINAMATH_CALUDE_quadratic_zero_point_range_l2087_208739

/-- The quadratic function f(x) = x^2 - 2x + a has a zero point in the interval (-1,3) 
    if and only if a is in the range (-3,1]. -/
theorem quadratic_zero_point_range (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Ioo (-1) 3 ∧ x^2 - 2*x + a = 0) ↔ a ∈ Set.Ioc (-3) 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_zero_point_range_l2087_208739


namespace NUMINAMATH_CALUDE_distance_to_origin_l2087_208773

theorem distance_to_origin (x y : ℝ) (h1 : y = 15) 
  (h2 : x = 2 + 2 * Real.sqrt 30) 
  (h3 : Real.sqrt ((x - 2)^2 + (y - 8)^2) = 13) : 
  Real.sqrt (x^2 + y^2) = Real.sqrt (349 + 8 * Real.sqrt 30) := by
sorry

end NUMINAMATH_CALUDE_distance_to_origin_l2087_208773


namespace NUMINAMATH_CALUDE_dinner_cost_difference_l2087_208793

theorem dinner_cost_difference (initial_amount : ℝ) (first_course_cost : ℝ) (remaining_amount : ℝ) : 
  initial_amount = 60 →
  first_course_cost = 15 →
  remaining_amount = 20 →
  ∃ (second_course_cost : ℝ),
    initial_amount = first_course_cost + second_course_cost + (0.25 * second_course_cost) + remaining_amount ∧
    second_course_cost - first_course_cost = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_dinner_cost_difference_l2087_208793


namespace NUMINAMATH_CALUDE_sin_120_degrees_l2087_208788

theorem sin_120_degrees : Real.sin (120 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_120_degrees_l2087_208788


namespace NUMINAMATH_CALUDE_complex_expression_evaluation_l2087_208753

theorem complex_expression_evaluation :
  (7 - 3 * Complex.I) - 3 * (2 + 4 * Complex.I) = 1 - 15 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_evaluation_l2087_208753


namespace NUMINAMATH_CALUDE_tetrahedron_projection_ratio_l2087_208762

/-- Represents a tetrahedron with edge lengths a, b, c, d, e, f -/
structure Tetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  a_greatest : a ≥ max b (max c (max d (max e f)))

/-- The ratio of projection areas for a tetrahedron -/
noncomputable def projection_area_ratio (t : Tetrahedron) : ℝ := sorry

/-- Theorem: For every tetrahedron, there exist two planes such that 
    the ratio of projection areas on those planes is not less than √2 -/
theorem tetrahedron_projection_ratio (t : Tetrahedron) : 
  projection_area_ratio t ≥ Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_tetrahedron_projection_ratio_l2087_208762


namespace NUMINAMATH_CALUDE_rectangle_area_change_l2087_208759

theorem rectangle_area_change (L W : ℝ) (h1 : L * W = 600) : 
  (0.8 * L) * (1.3 * W) = 624 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l2087_208759


namespace NUMINAMATH_CALUDE_chord_length_of_concentric_circles_l2087_208747

/-- Given two concentric circles with radii R and r, where the area of the annulus
    between them is 12½π square inches, the length of the chord of the larger circle
    which is tangent to the smaller circle is 5√2 inches. -/
theorem chord_length_of_concentric_circles (R r : ℝ) :
  R > r →
  π * R^2 - π * r^2 = 25 / 2 * π →
  ∃ (c : ℝ), c^2 = R^2 - r^2 ∧ c = 5 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_of_concentric_circles_l2087_208747


namespace NUMINAMATH_CALUDE_doughnuts_per_box_l2087_208706

theorem doughnuts_per_box (total : ℕ) (boxes : ℕ) (h1 : total = 48) (h2 : boxes = 4) :
  total / boxes = 12 := by
  sorry

end NUMINAMATH_CALUDE_doughnuts_per_box_l2087_208706


namespace NUMINAMATH_CALUDE_pasta_calculation_l2087_208776

/-- Given a recipe that uses 2 pounds of pasta to serve 7 people,
    calculate the amount of pasta needed to serve 35 people. -/
theorem pasta_calculation (original_pasta : ℝ) (original_servings : ℕ) 
    (target_servings : ℕ) (h1 : original_pasta = 2) 
    (h2 : original_servings = 7) (h3 : target_servings = 35) : 
    (original_pasta * target_servings / original_servings : ℝ) = 10 := by
  sorry

#check pasta_calculation

end NUMINAMATH_CALUDE_pasta_calculation_l2087_208776


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2087_208708

theorem inequality_solution_set (x : ℝ) : 3 * x - 2 > x ↔ x > 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2087_208708


namespace NUMINAMATH_CALUDE_unique_base_solution_l2087_208712

/-- Convert a number from base b to decimal --/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.foldl (fun acc d => acc * b + d) 0

/-- Check if the equation holds for a given base --/
def equation_holds (b : Nat) : Prop :=
  to_decimal [2, 5, 1] b + to_decimal [1, 7, 4] b = to_decimal [4, 3, 5] b

theorem unique_base_solution :
  ∃! b : Nat, b > 1 ∧ equation_holds b :=
sorry

end NUMINAMATH_CALUDE_unique_base_solution_l2087_208712


namespace NUMINAMATH_CALUDE_square_plus_abs_zero_implies_both_zero_l2087_208711

theorem square_plus_abs_zero_implies_both_zero (a b : ℝ) : 
  a^2 + |b| = 0 → a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_abs_zero_implies_both_zero_l2087_208711


namespace NUMINAMATH_CALUDE_seven_consecutive_beautiful_numbers_odd_numbers_beautiful_divisible_by_four_beautiful_not_beautiful_mod_eight_six_l2087_208743

def is_beautiful (n : ℤ) : Prop := ∃ a b : ℤ, n = a^2 + b^2 ∨ n = a^2 - b^2

theorem seven_consecutive_beautiful_numbers (k : ℤ) (hk : k ≥ 0) :
  ∃ n : ℤ, n ≥ k ∧ 
    (∀ i : ℤ, 0 ≤ i ∧ i < 7 → is_beautiful (8*n + i - 1)) ∧
    ¬(∀ i : ℤ, 0 ≤ i ∧ i < 8 → is_beautiful (8*n + i - 1)) :=
sorry

theorem odd_numbers_beautiful (n : ℤ) :
  n % 2 = 1 → is_beautiful n :=
sorry

theorem divisible_by_four_beautiful (n : ℤ) :
  n % 4 = 0 → is_beautiful n :=
sorry

theorem not_beautiful_mod_eight_six (n : ℤ) :
  n % 8 = 6 → ¬is_beautiful n :=
sorry

end NUMINAMATH_CALUDE_seven_consecutive_beautiful_numbers_odd_numbers_beautiful_divisible_by_four_beautiful_not_beautiful_mod_eight_six_l2087_208743


namespace NUMINAMATH_CALUDE_monotonic_sequence_bound_l2087_208748

theorem monotonic_sequence_bound (b : ℝ) :
  (∀ n : ℕ, (n + 1)^2 + b*(n + 1) > n^2 + b*n) →
  b > -3 := by
sorry

end NUMINAMATH_CALUDE_monotonic_sequence_bound_l2087_208748


namespace NUMINAMATH_CALUDE_min_c_value_l2087_208703

theorem min_c_value (a b c : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  (a + b + c) / 3 = 20 →
  a ≤ b → b ≤ c →
  b = a + 13 →
  ∀ c' : ℕ, c' > 0 ∧ 
    (∃ a' b' : ℕ, a' > 0 ∧ b' > 0 ∧
      (a' + b' + c') / 3 = 20 ∧
      a' ≤ b' ∧ b' ≤ c' ∧
      b' = a' + 13) →
    c ≤ c' →
  c = 45 :=
sorry

end NUMINAMATH_CALUDE_min_c_value_l2087_208703


namespace NUMINAMATH_CALUDE_ball_probabilities_l2087_208772

theorem ball_probabilities (total_balls : ℕ) (p_red p_black p_yellow : ℚ) :
  total_balls = 12 →
  p_red + p_black + p_yellow = 1 →
  p_red = 1/3 →
  p_black - p_yellow = 1/6 →
  p_black = 5/12 ∧ p_yellow = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ball_probabilities_l2087_208772


namespace NUMINAMATH_CALUDE_additive_multiplicative_inverses_problem_l2087_208798

theorem additive_multiplicative_inverses_problem (a b c d m : ℝ) 
  (h1 : a + b = 0)  -- a and b are additive inverses
  (h2 : c * d = 1)  -- c and d are multiplicative inverses
  (h3 : abs m = 1)  -- absolute value of m is 1
  : (a + b) * c * d - 2009 * m = -2009 ∨ (a + b) * c * d - 2009 * m = 2009 := by
  sorry

end NUMINAMATH_CALUDE_additive_multiplicative_inverses_problem_l2087_208798


namespace NUMINAMATH_CALUDE_simplify_expression1_simplify_expression2_l2087_208752

-- Problem 1
theorem simplify_expression1 (m n : ℝ) :
  2 * m^2 * n - 3 * m * n + 8 - 3 * m^2 * n + 5 * m * n - 3 = -m^2 * n + 2 * m * n + 5 :=
by sorry

-- Problem 2
theorem simplify_expression2 (a b : ℝ) :
  2 * (2 * a - 3 * b) - 3 * (2 * b - 3 * a) = 13 * a - 12 * b :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression1_simplify_expression2_l2087_208752


namespace NUMINAMATH_CALUDE_duplicate_page_sum_l2087_208751

theorem duplicate_page_sum (n : ℕ) (p : ℕ) : 
  p ≤ n →
  n * (n + 1) / 2 + p = 3005 →
  p = 2 :=
sorry

end NUMINAMATH_CALUDE_duplicate_page_sum_l2087_208751


namespace NUMINAMATH_CALUDE_inequality_holds_l2087_208736

theorem inequality_holds (a b : ℝ) (h1 : 2 < a) (h2 : a < b) (h3 : b < 3) :
  b * 2^a < a * 2^b := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l2087_208736


namespace NUMINAMATH_CALUDE_computer_literate_female_employees_l2087_208733

theorem computer_literate_female_employees 
  (total_employees : ℕ) 
  (female_percentage : ℚ) 
  (male_literate_percentage : ℚ) 
  (total_literate_percentage : ℚ) 
  (h1 : total_employees = 1400)
  (h2 : female_percentage = 60 / 100)
  (h3 : male_literate_percentage = 50 / 100)
  (h4 : total_literate_percentage = 62 / 100) :
  ↑(total_employees : ℚ) * female_percentage * total_literate_percentage - 
  (↑total_employees * (1 - female_percentage) * male_literate_percentage) = 588 := by
  sorry

#check computer_literate_female_employees

end NUMINAMATH_CALUDE_computer_literate_female_employees_l2087_208733


namespace NUMINAMATH_CALUDE_tracy_dogs_food_consumption_l2087_208727

/-- Proves that Tracy's two dogs consume 4 pounds of food per day -/
theorem tracy_dogs_food_consumption :
  let num_dogs : ℕ := 2
  let cups_per_meal_per_dog : ℚ := 3/2
  let meals_per_day : ℕ := 3
  let cups_per_pound : ℚ := 9/4
  
  let total_cups_per_day : ℚ := num_dogs * cups_per_meal_per_dog * meals_per_day
  let total_pounds_per_day : ℚ := total_cups_per_day / cups_per_pound
  
  total_pounds_per_day = 4 := by sorry

end NUMINAMATH_CALUDE_tracy_dogs_food_consumption_l2087_208727


namespace NUMINAMATH_CALUDE_only_statements_1_and_2_correct_l2087_208700

-- Define the structure of a programming statement
inductive ProgrammingStatement
| Input : String → ProgrammingStatement
| Output : String → ProgrammingStatement
| Assignment : String → String → ProgrammingStatement

-- Define the property of being a correct statement
def is_correct (s : ProgrammingStatement) : Prop :=
  match s with
  | ProgrammingStatement.Input _ => true
  | ProgrammingStatement.Output _ => false
  | ProgrammingStatement.Assignment lhs rhs => lhs ≠ rhs

-- Define the four statements from the problem
def statement1 : ProgrammingStatement := ProgrammingStatement.Input "x=3"
def statement2 : ProgrammingStatement := ProgrammingStatement.Input "A, B, C"
def statement3 : ProgrammingStatement := ProgrammingStatement.Output "A+B=C"
def statement4 : ProgrammingStatement := ProgrammingStatement.Assignment "3" "A"

-- Theorem to prove
theorem only_statements_1_and_2_correct :
  is_correct statement1 ∧ 
  is_correct statement2 ∧ 
  ¬is_correct statement3 ∧ 
  ¬is_correct statement4 :=
sorry

end NUMINAMATH_CALUDE_only_statements_1_and_2_correct_l2087_208700


namespace NUMINAMATH_CALUDE_slipper_discount_percentage_l2087_208705

/-- Calculates the discount percentage on slippers given the original price, 
    embroidery cost per shoe, shipping cost, and final discounted price. -/
theorem slipper_discount_percentage 
  (original_price : ℝ) 
  (embroidery_cost_per_shoe : ℝ) 
  (shipping_cost : ℝ) 
  (final_price : ℝ) : 
  original_price = 50 ∧ 
  embroidery_cost_per_shoe = 5.5 ∧ 
  shipping_cost = 10 ∧ 
  final_price = 66 →
  (original_price - (final_price - shipping_cost - 2 * embroidery_cost_per_shoe)) / original_price * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_slipper_discount_percentage_l2087_208705


namespace NUMINAMATH_CALUDE_S_min_at_24_l2087_208783

/-- The sequence term a_n as a function of n -/
def a (n : ℕ) : ℤ := 2 * n - 49

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ) : ℤ := n * (2 * a 1 + (n - 1) * 2) / 2

/-- Theorem stating that S reaches its minimum value when n = 24 -/
theorem S_min_at_24 : ∀ k : ℕ, S 24 ≤ S k :=
sorry

end NUMINAMATH_CALUDE_S_min_at_24_l2087_208783


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l2087_208799

/-- The trajectory of the midpoint M of a line segment PP', where P is on a circle
    with center (0,0) and radius 2, and P' is the projection of P on the x-axis. -/
theorem midpoint_trajectory (x y : ℝ) : 
  (∃ x₀ y₀ : ℝ, 
    x₀^2 + y₀^2 = 4 ∧   -- P is on the circle
    x = x₀ ∧            -- M's x-coordinate is same as P's
    2 * y = y₀) →       -- M's y-coordinate is half of P's
  x^2 / 4 + y^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l2087_208799


namespace NUMINAMATH_CALUDE_quadratic_root_form_l2087_208781

theorem quadratic_root_form (m n p : ℕ+) (h_gcd : Nat.gcd m.val (Nat.gcd n.val p.val) = 1) :
  (∀ x : ℝ, 3 * x^2 - 8 * x + 2 = 0 ↔ x = (m.val + Real.sqrt n.val) / p.val ∨ x = (m.val - Real.sqrt n.val) / p.val) →
  n = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_form_l2087_208781


namespace NUMINAMATH_CALUDE_multiplicative_inverse_203_mod_301_l2087_208755

theorem multiplicative_inverse_203_mod_301 :
  ∃ x : ℕ, x < 301 ∧ (7236 : ℤ) ≡ x [ZMOD 301] ∧ (203 * x) ≡ 1 [ZMOD 301] := by
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_203_mod_301_l2087_208755


namespace NUMINAMATH_CALUDE_flea_meeting_configuration_l2087_208746

/-- Represents a small triangle on the infinite sheet of triangulated paper -/
structure SmallTriangle where
  x : ℤ
  y : ℤ

/-- Represents the equilateral triangle containing n^2 small triangles -/
def LargeTriangle (n : ℕ) : Set SmallTriangle :=
  { t : SmallTriangle | 0 ≤ t.x ∧ 0 ≤ t.y ∧ t.x + t.y < n }

/-- Represents the set of possible jumps a flea can make -/
def PossibleJumps : List (ℤ × ℤ) := [(1, 0), (-1, 1), (0, -1)]

/-- Defines a valid jump for a flea -/
def ValidJump (t1 t2 : SmallTriangle) : Prop :=
  (t2.x - t1.x, t2.y - t1.y) ∈ PossibleJumps

/-- Theorem: For which positive integers n does there exist an initial configuration
    such that after a finite number of jumps all the n fleas can meet in a single small triangle? -/
theorem flea_meeting_configuration (n : ℕ) :
  (∃ (initial_config : Fin n → SmallTriangle)
     (final_triangle : SmallTriangle)
     (num_jumps : ℕ),
   (∀ i j : Fin n, i ≠ j → initial_config i ≠ initial_config j) ∧
   (∀ i : Fin n, initial_config i ∈ LargeTriangle n) ∧
   (∃ (jump_sequence : Fin n → ℕ → SmallTriangle),
     (∀ i : Fin n, jump_sequence i 0 = initial_config i) ∧
     (∀ i : Fin n, ∀ k : ℕ, k < num_jumps →
       ValidJump (jump_sequence i k) (jump_sequence i (k+1))) ∧
     (∀ i : Fin n, jump_sequence i num_jumps = final_triangle))) ↔
  (n ≥ 1 ∧ n ≠ 2 ∧ n ≠ 4) :=
sorry

end NUMINAMATH_CALUDE_flea_meeting_configuration_l2087_208746


namespace NUMINAMATH_CALUDE_canal_construction_efficiency_l2087_208721

theorem canal_construction_efficiency (total_length : ℝ) (efficiency_multiplier : ℝ) (days_ahead : ℝ) 
  (original_daily_plan : ℝ) : 
  total_length = 3600 ∧ 
  efficiency_multiplier = 1.8 ∧ 
  days_ahead = 20 ∧
  (total_length / original_daily_plan - total_length / (efficiency_multiplier * original_daily_plan) = days_ahead) →
  original_daily_plan = 20 := by
sorry

end NUMINAMATH_CALUDE_canal_construction_efficiency_l2087_208721


namespace NUMINAMATH_CALUDE_opposite_of_negative_sqrt_seven_l2087_208735

theorem opposite_of_negative_sqrt_seven (x : ℝ) : 
  x = -Real.sqrt 7 → -x = Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_sqrt_seven_l2087_208735


namespace NUMINAMATH_CALUDE_values_equal_l2087_208710

/-- The value of the expression at point C -/
def value_at_C : ℝ := 5 * 5 + 6 * 8.73

/-- The value of the expression at point D -/
def value_at_D : ℝ := 105

/-- Theorem stating that the values at points C and D are equal -/
theorem values_equal : value_at_C = value_at_D := by
  sorry

#eval value_at_C
#eval value_at_D

end NUMINAMATH_CALUDE_values_equal_l2087_208710


namespace NUMINAMATH_CALUDE_return_speed_theorem_l2087_208750

theorem return_speed_theorem (v : ℕ) : 
  v > 50 ∧ 
  v ≤ 100 ∧ 
  (∃ k : ℕ, k = (100 * v) / (50 + v)) → 
  v = 75 := by
sorry

end NUMINAMATH_CALUDE_return_speed_theorem_l2087_208750


namespace NUMINAMATH_CALUDE_quadratic_roots_transformation_l2087_208714

theorem quadratic_roots_transformation (a b : ℝ) (r₁ r₂ : ℝ) : 
  (r₁^2 - a*r₁ + b = 0) → 
  (r₂^2 - a*r₂ + b = 0) → 
  ∃ (x : ℝ), x^2 - (a^2 + a - 2*b)*x + (a^3 - a*b) = 0 ∧ 
  (x = r₁^2 + r₂ ∨ x = r₁ + r₂^2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_transformation_l2087_208714


namespace NUMINAMATH_CALUDE_remainder_3_101_plus_5_mod_11_l2087_208726

theorem remainder_3_101_plus_5_mod_11 : (3^101 + 5) % 11 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_101_plus_5_mod_11_l2087_208726


namespace NUMINAMATH_CALUDE_artist_payment_multiple_l2087_208778

theorem artist_payment_multiple : ∃ (x : ℕ+) (D : ℕ+), 
  D + (x * D + 1000) = 50000 ∧ 
  ∀ (y : ℕ+), y > x → ¬∃ (E : ℕ+), E + (y * E + 1000) = 50000 := by
  sorry

end NUMINAMATH_CALUDE_artist_payment_multiple_l2087_208778


namespace NUMINAMATH_CALUDE_divisors_half_of_n_l2087_208732

theorem divisors_half_of_n (n : ℕ) : 
  (n > 0) → (Finset.card (Nat.divisors n) = n / 2) → (n = 8 ∨ n = 12) := by
  sorry

end NUMINAMATH_CALUDE_divisors_half_of_n_l2087_208732


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2087_208787

/-- Given a function f: ℝ → ℝ satisfying the conditions:
    1) f(x+5) = 4x^3 + 5x^2 + 9x + 6
    2) f(x) = ax^3 + bx^2 + cx + d
    Prove that a + b + c + d = -206 -/
theorem sum_of_coefficients (f : ℝ → ℝ) (a b c d : ℝ) :
  (∀ x, f (x + 5) = 4 * x^3 + 5 * x^2 + 9 * x + 6) →
  (∀ x, f x = a * x^3 + b * x^2 + c * x + d) →
  a + b + c + d = -206 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2087_208787


namespace NUMINAMATH_CALUDE_triangle_is_equilateral_l2087_208713

/-- A triangle with sides a, b, and c is equilateral if b^2 = ac and 2b = a + c -/
theorem triangle_is_equilateral (a b c : ℝ) 
  (h1 : b^2 = a * c) 
  (h2 : 2 * b = a + c) : 
  a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_triangle_is_equilateral_l2087_208713
