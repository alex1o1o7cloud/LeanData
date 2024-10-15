import Mathlib

namespace NUMINAMATH_CALUDE_two_distinct_cool_triples_for_odd_x_l755_75589

/-- A cool type triple (x, y, z) consists of positive integers with y ≥ 2 
    and satisfies the equation x^2 - 3y^2 = z^2 - 3 -/
def CoolTriple (x y z : ℕ) : Prop :=
  x > 0 ∧ y ≥ 2 ∧ z > 0 ∧ x^2 - 3*y^2 = z^2 - 3

/-- For every odd x ≥ 5, there exist at least two distinct cool triples -/
theorem two_distinct_cool_triples_for_odd_x (x : ℕ) (h_odd : Odd x) (h_ge_5 : x ≥ 5) :
  ∃ y1 z1 y2 z2 : ℕ, 
    CoolTriple x y1 z1 ∧ 
    CoolTriple x y2 z2 ∧ 
    (y1 ≠ y2 ∨ z1 ≠ z2) :=
by
  sorry

end NUMINAMATH_CALUDE_two_distinct_cool_triples_for_odd_x_l755_75589


namespace NUMINAMATH_CALUDE_parabola_vertex_l755_75553

/-- The vertex of the parabola y = -3x^2 + 6x + 1 is (1, 4) -/
theorem parabola_vertex (x y : ℝ) : 
  y = -3 * x^2 + 6 * x + 1 → 
  ∃ (vertex_x vertex_y : ℝ), 
    vertex_x = 1 ∧ 
    vertex_y = 4 ∧ 
    ∀ (x' : ℝ), -3 * x'^2 + 6 * x' + 1 ≤ vertex_y :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l755_75553


namespace NUMINAMATH_CALUDE_cafe_customers_l755_75572

/-- Prove that the number of customers in a group is 12, given the following conditions:
  * 3 offices ordered 10 sandwiches each
  * Half of the group ordered 4 sandwiches each
  * Total sandwiches made is 54
-/
theorem cafe_customers (num_offices : Nat) (sandwiches_per_office : Nat)
  (sandwiches_per_customer : Nat) (total_sandwiches : Nat) :
  num_offices = 3 →
  sandwiches_per_office = 10 →
  sandwiches_per_customer = 4 →
  total_sandwiches = 54 →
  ∃ (num_customers : Nat),
    num_customers = 12 ∧
    total_sandwiches = num_offices * sandwiches_per_office +
      (num_customers / 2) * sandwiches_per_customer :=
by
  sorry

end NUMINAMATH_CALUDE_cafe_customers_l755_75572


namespace NUMINAMATH_CALUDE_smallest_even_triangle_perimeter_l755_75584

/-- A triangle with consecutive even integer side lengths -/
structure EvenTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  h1 : b = a + 2
  h2 : c = b + 2
  h3 : Even a
  h4 : a + b > c
  h5 : a + c > b
  h6 : b + c > a

/-- The perimeter of an EvenTriangle -/
def perimeter (t : EvenTriangle) : ℕ := t.a + t.b + t.c

/-- The statement that 18 is the smallest possible perimeter of an EvenTriangle -/
theorem smallest_even_triangle_perimeter :
  ∀ t : EvenTriangle, perimeter t ≥ 18 ∧ ∃ t₀ : EvenTriangle, perimeter t₀ = 18 := by
  sorry

end NUMINAMATH_CALUDE_smallest_even_triangle_perimeter_l755_75584


namespace NUMINAMATH_CALUDE_largest_b_value_l755_75531

theorem largest_b_value (b : ℚ) (h : (3*b+4)*(b-2) = 9*b) : 
  ∃ (max_b : ℚ), max_b = 4 ∧ ∀ (x : ℚ), (3*x+4)*(x-2) = 9*x → x ≤ max_b :=
by sorry

end NUMINAMATH_CALUDE_largest_b_value_l755_75531


namespace NUMINAMATH_CALUDE_correct_termination_condition_l755_75594

/-- Represents the state of the program at each iteration --/
structure ProgramState :=
  (i : ℕ)
  (S : ℕ)

/-- Simulates one iteration of the loop --/
def iterate (state : ProgramState) : ProgramState :=
  { i := state.i - 1, S := state.S * state.i }

/-- Checks if the given condition terminates the loop correctly --/
def is_correct_termination (condition : ℕ → Bool) : Prop :=
  let final_state := iterate (iterate (iterate (iterate { i := 12, S := 1 })))
  final_state.S = 11880 ∧ condition final_state.i = true ∧ 
  ∀ n, n < 4 → condition ((iterate^[n] { i := 12, S := 1 }).i) = false

theorem correct_termination_condition :
  is_correct_termination (λ i => i = 8) := by sorry

end NUMINAMATH_CALUDE_correct_termination_condition_l755_75594


namespace NUMINAMATH_CALUDE_cube_sum_given_sum_and_product_l755_75525

theorem cube_sum_given_sum_and_product (x y : ℝ) 
  (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := by sorry

end NUMINAMATH_CALUDE_cube_sum_given_sum_and_product_l755_75525


namespace NUMINAMATH_CALUDE_three_n_equals_twenty_seven_l755_75528

theorem three_n_equals_twenty_seven (n : ℤ) : 3 * n = 9 + 9 + 9 → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_three_n_equals_twenty_seven_l755_75528


namespace NUMINAMATH_CALUDE_max_volume_right_prism_l755_75538

/-- Given a right prism with rectangular base (sides a and b) and height h, 
    where the sum of areas of two lateral faces and one base is 40,
    the maximum volume of the prism is 80√30/9 -/
theorem max_volume_right_prism (a b h : ℝ) 
  (h₁ : a > 0) (h₂ : b > 0) (h₃ : h > 0)
  (h₄ : a * h + b * h + a * b = 40) : 
  a * b * h ≤ 80 * Real.sqrt 30 / 9 := by
  sorry

#check max_volume_right_prism

end NUMINAMATH_CALUDE_max_volume_right_prism_l755_75538


namespace NUMINAMATH_CALUDE_unique_snuggly_number_l755_75548

def is_snuggly (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 10 * a + b ∧ a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 2 * a + b^2

theorem unique_snuggly_number : ∃! n : ℕ, is_snuggly n :=
  sorry

end NUMINAMATH_CALUDE_unique_snuggly_number_l755_75548


namespace NUMINAMATH_CALUDE_emmas_speed_last_segment_l755_75516

def total_distance : ℝ := 150
def total_time : ℝ := 2
def speed_segment1 : ℝ := 50
def speed_segment2 : ℝ := 75
def num_segments : ℕ := 3

theorem emmas_speed_last_segment (speed_segment3 : ℝ) : 
  (speed_segment1 + speed_segment2 + speed_segment3) / num_segments = total_distance / total_time →
  speed_segment3 = 100 := by
sorry

end NUMINAMATH_CALUDE_emmas_speed_last_segment_l755_75516


namespace NUMINAMATH_CALUDE_destination_distance_l755_75597

theorem destination_distance (d : ℝ) : 
  (¬ (d ≥ 8)) →  -- Alice's statement is false
  (¬ (d ≤ 7)) →  -- Bob's statement is false
  (d ≠ 6) →      -- Charlie's statement is false
  7 < d ∧ d < 8 := by
sorry

end NUMINAMATH_CALUDE_destination_distance_l755_75597


namespace NUMINAMATH_CALUDE_intermediate_value_theorem_l755_75500

theorem intermediate_value_theorem {f : ℝ → ℝ} {a b : ℝ} (h_cont : ContinuousOn f (Set.Icc a b)) 
  (h_ab : a ≤ b) (h_sign : f a * f b < 0) : 
  ∃ c ∈ Set.Icc a b, f c = 0 := by
  sorry

end NUMINAMATH_CALUDE_intermediate_value_theorem_l755_75500


namespace NUMINAMATH_CALUDE_f_properties_l755_75504

noncomputable section

variable (f : ℝ → ℝ)

-- f is an even function
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- f satisfies the given functional equation
def satisfies_equation (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = f x + f 1

-- f is monotonically increasing on [0, 1]
def monotone_increasing_on_unit_interval (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 1 → f x ≤ f y

-- The graph of f is symmetric about x = 1
def symmetric_about_one (f : ℝ → ℝ) : Prop := ∀ x, f (x + 1) = f (1 - x)

-- f is periodic
def periodic (f : ℝ → ℝ) : Prop := ∃ p > 0, ∀ x, f (x + p) = f x

-- f has local minima at even x-coordinates
def local_minima_at_even (f : ℝ → ℝ) : Prop :=
  ∀ x, ∃ ε > 0, ∀ y, |y - x| < ε → f x ≤ f y

theorem f_properties (heven : even_function f)
                     (heq : satisfies_equation f)
                     (hmon : monotone_increasing_on_unit_interval f) :
  symmetric_about_one f ∧ periodic f ∧ local_minima_at_even f := by sorry

end

end NUMINAMATH_CALUDE_f_properties_l755_75504


namespace NUMINAMATH_CALUDE_arithmetic_sequence_separable_special_sequence_a_value_complex_sequence_separable_values_l755_75579

/-- A sequence is m-th degree separable if there exists an n such that a_{m+n} = a_m + a_n -/
def IsNthDegreeSeparable (a : ℕ → ℝ) (m : ℕ) : Prop :=
  ∃ n : ℕ, a (m + n) = a m + a n

/-- An arithmetic sequence with first term 2 and common difference 2 -/
def ArithmeticSequence (n : ℕ) : ℝ := 2 * n

/-- A sequence with sum of first n terms S_n = 2^n - a where a > 0 -/
def SpecialSequence (a : ℝ) (n : ℕ) : ℝ := 2^n - a

/-- A sequence defined by a_n = 2^n + n^2 + 12 -/
def ComplexSequence (n : ℕ) : ℝ := 2^n + n^2 + 12

theorem arithmetic_sequence_separable :
  IsNthDegreeSeparable ArithmeticSequence 3 :=
sorry

theorem special_sequence_a_value (a : ℝ) (h : a > 0) :
  IsNthDegreeSeparable (SpecialSequence a) 1 → a = 1 :=
sorry

theorem complex_sequence_separable_values :
  (∃ m : ℕ, IsNthDegreeSeparable ComplexSequence m) ∧
  (∀ m : ℕ, IsNthDegreeSeparable ComplexSequence m → (m = 1 ∨ m = 3)) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_separable_special_sequence_a_value_complex_sequence_separable_values_l755_75579


namespace NUMINAMATH_CALUDE_geese_count_l755_75555

/-- The number of geese in a flock that land on n lakes -/
def geese (n : ℕ) : ℕ := 2^n - 1

/-- 
Theorem: The number of geese in a flock is 2^n - 1, where n is the number of lakes,
given the landing pattern described.
-/
theorem geese_count (n : ℕ) : 
  (∀ k < n, (geese k + 1) / 2 + (geese k) / 2 = geese (k + 1)) → 
  geese 0 = 0 → 
  geese n = 2^n - 1 := by
sorry

end NUMINAMATH_CALUDE_geese_count_l755_75555


namespace NUMINAMATH_CALUDE_x_value_l755_75598

theorem x_value (x y : ℝ) (h1 : x - y = 18) (h2 : x + y = 10) : x = 14 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l755_75598


namespace NUMINAMATH_CALUDE_smallest_number_l755_75577

/-- Convert a number from base b to decimal --/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b^i) 0

/-- The given numbers in their respective bases --/
def number_A : List Nat := [0, 2]
def number_B : List Nat := [0, 3]
def number_C : List Nat := [3, 2]
def number_D : List Nat := [1, 3]

/-- The bases of the given numbers --/
def base_A : Nat := 7
def base_B : Nat := 5
def base_C : Nat := 6
def base_D : Nat := 4

theorem smallest_number :
  to_decimal number_D base_D < to_decimal number_A base_A ∧
  to_decimal number_D base_D < to_decimal number_B base_B ∧
  to_decimal number_D base_D < to_decimal number_C base_C :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l755_75577


namespace NUMINAMATH_CALUDE_power_calculation_l755_75559

theorem power_calculation (a : ℝ) (h : a ≠ 0) : (a^2)^3 / a^2 = a^4 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l755_75559


namespace NUMINAMATH_CALUDE_min_tablets_extraction_l755_75512

/-- Represents the number of tablets of each medicine type in the box -/
structure TabletCounts where
  a : Nat
  b : Nat
  c : Nat

/-- Calculates the minimum number of tablets to extract to ensure at least 3 of each kind -/
def minTablets (counts : TabletCounts) : Nat :=
  16

/-- Theorem stating that for the given tablet counts, the minimum number of tablets to extract is 16 -/
theorem min_tablets_extraction (counts : TabletCounts) 
  (h1 : counts.a = 30) (h2 : counts.b = 24) (h3 : counts.c = 18) : 
  minTablets counts = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_tablets_extraction_l755_75512


namespace NUMINAMATH_CALUDE_unique_m_equals_three_l755_75513

/-- A graph is k-flowing-chromatic if it satisfies certain coloring and movement conditions -/
def is_k_flowing_chromatic (G : Graph) (k : ℕ) : Prop := sorry

/-- T(G) is the least k such that G is k-flowing-chromatic, or 0 if no such k exists -/
def T (G : Graph) : ℕ := sorry

/-- χ(G) is the chromatic number of graph G -/
def chromatic_number (G : Graph) : ℕ := sorry

/-- A graph has no small cycles if all its cycles have length at least 2017 -/
def no_small_cycles (G : Graph) : Prop := sorry

/-- Main theorem: m = 3 is the only positive integer satisfying the conditions -/
theorem unique_m_equals_three :
  ∀ m : ℕ, m > 0 →
  (∃ G : Graph, chromatic_number G ≤ m ∧ T G ≥ 2^m ∧ no_small_cycles G) ↔ m = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_m_equals_three_l755_75513


namespace NUMINAMATH_CALUDE_cost_of_dozen_pens_l755_75534

/-- Given the cost of some pens and 5 pencils is Rs. 200, and the cost ratio of one pen to one pencil
    is 5:1, prove that the cost of one dozen pens is Rs. 120. -/
theorem cost_of_dozen_pens (n : ℕ) (x : ℚ) : 
  5 * n * x + 5 * x = 200 →  -- Cost of n pens and 5 pencils is 200
  (5 * x) / x = 5 / 1 →      -- Cost ratio of pen to pencil is 5:1
  12 * (5 * x) = 120 :=      -- Cost of dozen pens is 120
by sorry

end NUMINAMATH_CALUDE_cost_of_dozen_pens_l755_75534


namespace NUMINAMATH_CALUDE_min_value_of_sum_l755_75561

theorem min_value_of_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum_squares : x^2 + y^2 + z^2 = 1) :
  x*y/z + y*z/x + z*x/y ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l755_75561


namespace NUMINAMATH_CALUDE_turner_syndrome_classification_l755_75590

-- Define the types of mutations
inductive MutationType
  | GeneMutation
  | ChromosomalNumberVariation
  | GeneRecombination
  | ChromosomalStructureVariation

-- Define a structure for chromosomes
structure Chromosome where
  isSexChromosome : Bool

-- Define a human genetic condition
structure GeneticCondition where
  name : String
  missingChromosome : Option Chromosome
  mutationType : MutationType

-- Define Turner syndrome
def TurnerSyndrome : GeneticCondition where
  name := "Turner syndrome"
  missingChromosome := some { isSexChromosome := true }
  mutationType := MutationType.ChromosomalNumberVariation

-- Theorem statement
theorem turner_syndrome_classification :
  TurnerSyndrome.mutationType = MutationType.ChromosomalNumberVariation :=
by
  sorry


end NUMINAMATH_CALUDE_turner_syndrome_classification_l755_75590


namespace NUMINAMATH_CALUDE_eggs_sold_count_l755_75587

/-- The number of eggs in each tray -/
def eggs_per_tray : ℕ := 30

/-- The initial number of trays to be collected -/
def initial_trays : ℕ := 10

/-- The number of trays dropped -/
def dropped_trays : ℕ := 2

/-- The number of additional trays added after the accident -/
def additional_trays : ℕ := 7

/-- Theorem stating the total number of eggs sold -/
theorem eggs_sold_count : 
  (initial_trays - dropped_trays + additional_trays) * eggs_per_tray = 450 := by
sorry

end NUMINAMATH_CALUDE_eggs_sold_count_l755_75587


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_but_not_opposite_l755_75543

-- Define the set of cards
inductive Card : Type
| Red : Card
| Black : Card
| Blue : Card
| White : Card

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

-- Define a distribution as a function from Person to Card
def Distribution := Person → Card

-- Define the event "A receives the red card"
def A_receives_red (d : Distribution) : Prop := d Person.A = Card.Red

-- Define the event "B receives the red card"
def B_receives_red (d : Distribution) : Prop := d Person.B = Card.Red

-- Define the property of a valid distribution
def valid_distribution (d : Distribution) : Prop :=
  ∀ (c : Card), ∃! (p : Person), d p = c

theorem events_mutually_exclusive_but_not_opposite :
  (∀ (d : Distribution), valid_distribution d →
    ¬(A_receives_red d ∧ B_receives_red d)) ∧
  (∃ (d : Distribution), valid_distribution d ∧
    ¬A_receives_red d ∧ ¬B_receives_red d) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_but_not_opposite_l755_75543


namespace NUMINAMATH_CALUDE_exponent_division_l755_75562

theorem exponent_division (a : ℝ) : a^7 / a^4 = a^3 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l755_75562


namespace NUMINAMATH_CALUDE_min_colors_for_triangle_free_edge_coloring_l755_75567

theorem min_colors_for_triangle_free_edge_coloring (n : Nat) (h : n = 2015) :
  ∃ (f : Fin n → Fin n → Fin n),
    (∀ (i j : Fin n), i ≠ j → f i j = f j i) ∧
    (∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k → f i j ≠ f j k ∨ f j k ≠ f i k ∨ f i k ≠ f i j) ∧
    (∀ (g : Fin n → Fin n → Fin (n - 1)),
      ∃ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ g i j = g j k ∧ g j k = g i k) :=
by sorry

#check min_colors_for_triangle_free_edge_coloring

end NUMINAMATH_CALUDE_min_colors_for_triangle_free_edge_coloring_l755_75567


namespace NUMINAMATH_CALUDE_circle_collinearity_l755_75574

-- Define the circle ω
def circle_ω (A B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P | dist P ((A + B) / 2) = dist A B / 2}

-- Define a point on the circle
def point_on_circle (O : ℝ × ℝ) (ω : Set (ℝ × ℝ)) : Prop :=
  O ∈ ω

-- Define orthogonal projection
def orthogonal_projection (O H : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  (H.1 - A.1) * (B.2 - A.2) = (H.2 - A.2) * (B.1 - A.1) ∧
  (O.1 - H.1) * (B.1 - A.1) + (O.2 - H.2) * (B.2 - A.2) = 0

-- Define the intersection of two circles
def circle_intersection (O H X Y : ℝ × ℝ) (ω : Set (ℝ × ℝ)) : Prop :=
  X ∈ ω ∧ Y ∈ ω ∧
  dist X O = dist O H ∧ dist Y O = dist O H

-- Define collinearity
def collinear (P Q R : ℝ × ℝ) : Prop :=
  (Q.2 - P.2) * (R.1 - P.1) = (R.2 - P.2) * (Q.1 - P.1)

-- The main theorem
theorem circle_collinearity 
  (A B O H X Y : ℝ × ℝ) (ω : Set (ℝ × ℝ)) :
  ω = circle_ω A B →
  point_on_circle O ω →
  orthogonal_projection O H A B →
  circle_intersection O H X Y ω →
  collinear X Y ((O + H) / 2) :=
sorry

end NUMINAMATH_CALUDE_circle_collinearity_l755_75574


namespace NUMINAMATH_CALUDE_nikki_movie_length_l755_75529

/-- The lengths of favorite movies for Joyce, Michael, Nikki, and Ryn satisfy certain conditions -/
structure MovieLengths where
  michael : ℝ
  joyce : ℝ
  nikki : ℝ
  ryn : ℝ
  joyce_longer : joyce = michael + 2
  nikki_triple : nikki = 3 * michael
  ryn_proportion : ryn = (4/5) * nikki
  total_length : michael + joyce + nikki + ryn = 76

/-- Given the conditions, Nikki's favorite movie is 30 hours long -/
theorem nikki_movie_length (m : MovieLengths) : m.nikki = 30 := by
  sorry

end NUMINAMATH_CALUDE_nikki_movie_length_l755_75529


namespace NUMINAMATH_CALUDE_student_number_factor_l755_75556

theorem student_number_factor : ∃ f : ℚ, 122 * f - 138 = 106 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_student_number_factor_l755_75556


namespace NUMINAMATH_CALUDE_franks_reading_rate_l755_75552

/-- Represents a book with its properties --/
structure Book where
  pages : ℕ
  chapters : ℕ
  days_to_read : ℕ

/-- Calculates the number of chapters read per day --/
def chapters_per_day (b : Book) : ℚ :=
  (b.chapters : ℚ) / (b.days_to_read : ℚ)

/-- Theorem stating the number of chapters read per day for Frank's book --/
theorem franks_reading_rate (b : Book) 
    (h1 : b.pages = 193)
    (h2 : b.chapters = 15)
    (h3 : b.days_to_read = 660) :
    chapters_per_day b = 15 / 660 := by
  sorry

end NUMINAMATH_CALUDE_franks_reading_rate_l755_75552


namespace NUMINAMATH_CALUDE_ball_radius_under_shadow_l755_75501

/-- The radius of a ball under specific shadow conditions -/
theorem ball_radius_under_shadow (ball_shadow_length : ℝ) (ruler_shadow_length : ℝ) 
  (h1 : ball_shadow_length = 10)
  (h2 : ruler_shadow_length = 2) : 
  ∃ (r : ℝ), r = 10 * Real.sqrt 5 - 20 ∧ r > 0 := by
  sorry

end NUMINAMATH_CALUDE_ball_radius_under_shadow_l755_75501


namespace NUMINAMATH_CALUDE_twenty_five_binary_l755_75521

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinaryAux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
  toBinaryAux n

/-- Converts a list of bits to its decimal representation -/
def fromBinary (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

theorem twenty_five_binary :
  toBinary 25 = [true, false, false, true, true] :=
by sorry

end NUMINAMATH_CALUDE_twenty_five_binary_l755_75521


namespace NUMINAMATH_CALUDE_pairwise_sum_difference_l755_75573

theorem pairwise_sum_difference (n : ℕ) (x : Fin n → ℝ) 
  (h_n : n ≥ 4) 
  (h_pos : ∀ i, x i > 0) : 
  ∃ (i j k l : Fin n), i ≠ j ∧ k ≠ l ∧ 
    (x i + x j) ≤ (x k + x l) * (2 : ℝ)^(1 / (n - 2 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_pairwise_sum_difference_l755_75573


namespace NUMINAMATH_CALUDE_savings_calculation_l755_75563

def income_expenditure_ratio (income expenditure : ℚ) : Prop :=
  income / expenditure = 5 / 4

theorem savings_calculation (income : ℚ) (h : income_expenditure_ratio income ((4/5) * income)) :
  income - ((4/5) * income) = 3200 :=
by
  sorry

#check savings_calculation (16000 : ℚ)

end NUMINAMATH_CALUDE_savings_calculation_l755_75563


namespace NUMINAMATH_CALUDE_range_of_a_l755_75599

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + x - 6 < 0}
def B (a : ℝ) : Set ℝ := {x | x > a}

-- Define the sufficient condition
def sufficient_condition (a : ℝ) : Prop := ∀ x, x ∈ A → x ∈ B a

-- Theorem statement
theorem range_of_a : 
  ∀ a : ℝ, sufficient_condition a ↔ a ≤ -3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l755_75599


namespace NUMINAMATH_CALUDE_volleyball_starters_count_l755_75527

def volleyball_team_size : ℕ := 16
def triplet_size : ℕ := 3
def starter_size : ℕ := 6

def choose_starters (team_size triplet_size starter_size : ℕ) : ℕ :=
  let non_triplet_size := team_size - triplet_size
  let with_one_triplet := triplet_size * Nat.choose non_triplet_size (starter_size - 1)
  let without_triplets := Nat.choose non_triplet_size starter_size
  with_one_triplet + without_triplets

theorem volleyball_starters_count :
  choose_starters volleyball_team_size triplet_size starter_size = 5577 :=
by sorry

end NUMINAMATH_CALUDE_volleyball_starters_count_l755_75527


namespace NUMINAMATH_CALUDE_greg_age_l755_75595

/-- Given the ages of five people with certain relationships, prove Greg's age --/
theorem greg_age (C D E F G : ℕ) : 
  D = E - 5 →
  E = 2 * C →
  F = C - 1 →
  G = 2 * F →
  D = 15 →
  G = 18 := by
  sorry

#check greg_age

end NUMINAMATH_CALUDE_greg_age_l755_75595


namespace NUMINAMATH_CALUDE_non_congruent_squares_on_6x6_grid_l755_75592

/-- A square on a lattice grid --/
structure LatticeSquare where
  side_length : ℕ
  is_diagonal : Bool

/-- The size of the grid --/
def grid_size : ℕ := 6

/-- Counts the number of squares with a given side length that fit on the grid --/
def count_squares (s : ℕ) : ℕ :=
  (grid_size - s + 1) ^ 2

/-- Counts all non-congruent squares on the grid --/
def total_non_congruent_squares : ℕ :=
  (List.range 5).map (λ i => count_squares (i + 1)) |> List.sum

/-- The main theorem stating the number of non-congruent squares on a 6x6 grid --/
theorem non_congruent_squares_on_6x6_grid :
  total_non_congruent_squares = 110 := by
  sorry


end NUMINAMATH_CALUDE_non_congruent_squares_on_6x6_grid_l755_75592


namespace NUMINAMATH_CALUDE_hyperbola_equation_l755_75541

/-- Given a hyperbola with eccentricity e = √6/2 and the area of rectangle OMPN equal to √2,
    which is also equal to (1/2)ab, prove that the equation of the hyperbola is x^2/4 - y^2/2 = 1. -/
theorem hyperbola_equation (e a b : ℝ) (h1 : e = Real.sqrt 6 / 2) 
    (h2 : (1/2) * a * b = Real.sqrt 2) : 
    ∀ (x y : ℝ), x^2/4 - y^2/2 = 1 ↔ x^2/a^2 - y^2/b^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l755_75541


namespace NUMINAMATH_CALUDE_circle_and_line_properties_l755_75591

-- Define the circle C
def circle_C (x y b : ℝ) : Prop := (x + 2)^2 + (y - b)^2 = 3

-- Define the line l
def line_l (x y m : ℝ) : Prop := y = x + m

-- Define the point that the circle passes through
def point_on_circle (b : ℝ) : Prop := circle_C (-2 + Real.sqrt 2) 0 b

-- Define the tangency condition
def is_tangent (m b : ℝ) : Prop :=
  (|(-2) - 1 + m| / Real.sqrt 2) = Real.sqrt 3

-- Define the perpendicular condition
def is_perpendicular (m : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    circle_C x₁ y₁ 1 ∧ circle_C x₂ y₂ 1 ∧
    line_l x₁ y₁ m ∧ line_l x₂ y₂ m ∧
    x₁ * x₂ + y₁ * y₂ = 0

theorem circle_and_line_properties :
  (∃ b : ℝ, b > 0 ∧ point_on_circle b) ∧
  (∃ m : ℝ, is_tangent m 1) ∧
  (∃ m : ℝ, is_perpendicular m) :=
sorry

end NUMINAMATH_CALUDE_circle_and_line_properties_l755_75591


namespace NUMINAMATH_CALUDE_paper_tearing_theorem_l755_75526

/-- Represents the number of pieces after n tearing operations -/
def pieces (n : ℕ) : ℕ := 1 + 4 * n

theorem paper_tearing_theorem :
  (¬ ∃ n : ℕ, pieces n = 1994) ∧ (∃ n : ℕ, pieces n = 1997) := by
  sorry

end NUMINAMATH_CALUDE_paper_tearing_theorem_l755_75526


namespace NUMINAMATH_CALUDE_defeat_monster_time_l755_75566

/-- The time required to defeat a monster given the attack rates of two Ultramen and the monster's durability. -/
theorem defeat_monster_time 
  (monster_durability : ℕ) 
  (ultraman1_rate : ℕ) 
  (ultraman2_rate : ℕ) 
  (h1 : monster_durability = 100)
  (h2 : ultraman1_rate = 12)
  (h3 : ultraman2_rate = 8) : 
  (monster_durability : ℚ) / (ultraman1_rate + ultraman2_rate : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_defeat_monster_time_l755_75566


namespace NUMINAMATH_CALUDE_expression_equality_equation_solutions_l755_75581

-- Problem 1
theorem expression_equality : 
  |Real.sqrt 3 - 1| - 2 * Real.cos (π / 3) + (Real.sqrt 3 - 2)^2 + Real.sqrt 12 = 5 - Real.sqrt 3 := by
  sorry

-- Problem 2
theorem equation_solutions (x : ℝ) : 
  2 * (x - 3)^2 = x^2 - 9 ↔ x = 3 ∨ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_equation_solutions_l755_75581


namespace NUMINAMATH_CALUDE_m_values_l755_75514

def A : Set ℝ := {x | x^2 + x - 2 = 0}
def B (m : ℝ) : Set ℝ := {x | m * x + 1 = 0}

theorem m_values (m : ℝ) : (A ∪ B m = A) → (m = 0 ∨ m = -1 ∨ m = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_m_values_l755_75514


namespace NUMINAMATH_CALUDE_boat_distance_l755_75518

/-- The distance covered by a boat given its speed in still water and the time taken to cover the same distance downstream and upstream. -/
theorem boat_distance (v : ℝ) (t_down t_up : ℝ) (h1 : v = 7) (h2 : t_down = 2) (h3 : t_up = 5) :
  ∃ (d : ℝ), d = 20 ∧ d = (v + (v * t_down - d) / t_down) * t_down ∧ d = (v - (v * t_up - d) / t_up) * t_up :=
sorry

end NUMINAMATH_CALUDE_boat_distance_l755_75518


namespace NUMINAMATH_CALUDE_window_purchase_savings_l755_75533

/-- Represents the store's window sale offer -/
structure WindowOffer where
  regularPrice : ℕ
  buyCount : ℕ
  freeCount : ℕ

/-- Calculates the cost of purchasing a given number of windows under the offer -/
def calculateCost (offer : WindowOffer) (windowsNeeded : ℕ) : ℕ :=
  let fullSets := windowsNeeded / (offer.buyCount + offer.freeCount)
  let remainder := windowsNeeded % (offer.buyCount + offer.freeCount)
  let windowsPaidFor := fullSets * offer.buyCount + min remainder offer.buyCount
  windowsPaidFor * offer.regularPrice

/-- The main theorem stating the savings when Dave and Doug purchase windows together -/
theorem window_purchase_savings : 
  let offer : WindowOffer := ⟨100, 3, 1⟩
  let davesWindows : ℕ := 9
  let dougsWindows : ℕ := 10
  let totalWindows : ℕ := davesWindows + dougsWindows
  let separateCost : ℕ := calculateCost offer davesWindows + calculateCost offer dougsWindows
  let combinedCost : ℕ := calculateCost offer totalWindows
  let savings : ℕ := separateCost - combinedCost
  savings = 600 := by sorry

end NUMINAMATH_CALUDE_window_purchase_savings_l755_75533


namespace NUMINAMATH_CALUDE_f_derivative_l755_75517

noncomputable def f (x : ℝ) : ℝ := Real.log x + Real.cos x

theorem f_derivative (x : ℝ) (h : x > 0) : 
  deriv f x = 1 / x - Real.sin x := by sorry

end NUMINAMATH_CALUDE_f_derivative_l755_75517


namespace NUMINAMATH_CALUDE_range_of_c_l755_75522

/-- Given c > 0, if the function y = c^x is decreasing on ℝ and the minimum value of f(x) = x^2 - c^2 
    is no greater than -1/16, then 1/4 ≤ c < 1 -/
theorem range_of_c (c : ℝ) (hc : c > 0) 
  (hp : ∀ (x y : ℝ), x < y → c^x > c^y) 
  (hq : ∃ (k : ℝ), ∀ (x : ℝ), x^2 - c^2 ≥ k ∧ k ≤ -1/16) : 
  1/4 ≤ c ∧ c < 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_c_l755_75522


namespace NUMINAMATH_CALUDE_youngbin_shopping_combinations_l755_75560

def n : ℕ := 3
def k : ℕ := 2

theorem youngbin_shopping_combinations : Nat.choose n k = 3 := by
  sorry

end NUMINAMATH_CALUDE_youngbin_shopping_combinations_l755_75560


namespace NUMINAMATH_CALUDE_video_recorder_wholesale_cost_l755_75596

theorem video_recorder_wholesale_cost :
  ∀ (wholesale_cost retail_price employee_price : ℝ),
    retail_price = 1.2 * wholesale_cost →
    employee_price = 0.7 * retail_price →
    employee_price = 168 →
    wholesale_cost = 200 := by
  sorry

end NUMINAMATH_CALUDE_video_recorder_wholesale_cost_l755_75596


namespace NUMINAMATH_CALUDE_frog_hop_probability_l755_75510

/-- Represents a position on a 4x4 grid -/
structure Position :=
  (x : Fin 4)
  (y : Fin 4)

/-- Represents the possible directions of movement -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Defines whether a position is on the edge of the grid -/
def isEdgePosition (p : Position) : Bool :=
  p.x = 0 || p.x = 3 || p.y = 0 || p.y = 3

/-- Defines the next position after a hop in a given direction -/
def nextPosition (p : Position) (d : Direction) : Position :=
  match d with
  | Direction.Up => ⟨p.x, (p.y + 1) % 4⟩
  | Direction.Down => ⟨p.x, (p.y + 3) % 4⟩
  | Direction.Left => ⟨(p.x + 3) % 4, p.y⟩
  | Direction.Right => ⟨(p.x + 1) % 4, p.y⟩

/-- The probability of reaching an edge position within n hops -/
def probReachEdge (n : Nat) (start : Position) : Rat :=
  sorry

/-- The main theorem to prove -/
theorem frog_hop_probability :
  probReachEdge 5 ⟨2, 2⟩ = 15/16 := by
  sorry

end NUMINAMATH_CALUDE_frog_hop_probability_l755_75510


namespace NUMINAMATH_CALUDE_sqrt_x_minus_5_meaningful_l755_75578

theorem sqrt_x_minus_5_meaningful (x : ℝ) : 
  ∃ y : ℝ, y ^ 2 = x - 5 ↔ x ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_5_meaningful_l755_75578


namespace NUMINAMATH_CALUDE_three_numbers_ratio_l755_75550

theorem three_numbers_ratio (F S T : ℚ) : 
  F + S + T = 550 → 
  S = 150 → 
  T = F / 3 → 
  F / S = 2 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_ratio_l755_75550


namespace NUMINAMATH_CALUDE_identity_proof_special_case_proof_l755_75546

-- Define the sequence f_n = a^n + b^n
def f (a b : ℝ) : ℕ → ℝ
  | 0 => 1
  | n + 1 => a^(n+1) + b^(n+1)

theorem identity_proof (a b : ℝ) (n : ℕ) :
  f a b (n + 1) = (a + b) * (f a b n) - a * b * (f a b (n - 1)) :=
by sorry

theorem special_case_proof (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = -1) :
  f a b 10 = 123 :=
by sorry

end NUMINAMATH_CALUDE_identity_proof_special_case_proof_l755_75546


namespace NUMINAMATH_CALUDE_arithmetic_geometric_inequality_two_vars_arithmetic_geometric_inequality_three_vars_l755_75536

theorem arithmetic_geometric_inequality_two_vars (a b : ℝ) (h : a ≤ b) :
  a^2 + b^2 ≥ 2 * a * b := by sorry

theorem arithmetic_geometric_inequality_three_vars (a b c : ℝ) (h1 : a ≤ b) (h2 : b ≤ c) :
  a^2 + b^2 + c^2 ≥ a * b + b * c + c * a := by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_inequality_two_vars_arithmetic_geometric_inequality_three_vars_l755_75536


namespace NUMINAMATH_CALUDE_todd_gum_problem_l755_75558

theorem todd_gum_problem (initial_gum : ℕ) (steve_gum : ℕ) (total_gum : ℕ) : 
  steve_gum = 16 → total_gum = 54 → total_gum = initial_gum + steve_gum → initial_gum = 38 := by
  sorry

end NUMINAMATH_CALUDE_todd_gum_problem_l755_75558


namespace NUMINAMATH_CALUDE_scallop_dinner_cost_l755_75576

/-- Represents the problem of calculating the cost of scallops for Nate's dinner. -/
theorem scallop_dinner_cost :
  let scallops_per_pound : ℕ := 8
  let cost_per_pound : ℚ := 24
  let scallops_per_person : ℕ := 2
  let number_of_people : ℕ := 8
  
  let total_scallops : ℕ := scallops_per_person * number_of_people
  let pounds_needed : ℚ := total_scallops / scallops_per_pound
  let total_cost : ℚ := pounds_needed * cost_per_pound
  
  total_cost = 48 :=
by
  sorry


end NUMINAMATH_CALUDE_scallop_dinner_cost_l755_75576


namespace NUMINAMATH_CALUDE_polar_to_cartesian_equivalence_l755_75580

def polar_equation (ρ θ : ℝ) : Prop :=
  ρ^2 * Real.sin θ - ρ * (Real.cos θ)^2 - Real.sin θ = 0

def cartesian_equation (x y : ℝ) : Prop :=
  x = 1 ∨ (x^2 + y^2 + y = 0 ∧ y ≠ 0)

theorem polar_to_cartesian_equivalence :
  ∀ x y ρ θ, θ ∈ Set.Ioo 0 Real.pi →
  x = ρ * Real.cos θ → y = ρ * Real.sin θ →
  (polar_equation ρ θ ↔ cartesian_equation x y) := by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_equivalence_l755_75580


namespace NUMINAMATH_CALUDE_sector_central_angle_l755_75539

/-- Given a sector with radius 10 cm and perimeter 45 cm, its central angle is 2.5 radians. -/
theorem sector_central_angle (r : ℝ) (p : ℝ) (α : ℝ) : 
  r = 10 → p = 45 → α = (p - 2 * r) / r → α = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l755_75539


namespace NUMINAMATH_CALUDE_magic_square_sum_l755_75520

/-- Represents a 3x3 magic square --/
structure MagicSquare :=
  (a b c d e : ℕ)
  (row1_sum : 30 + d + 24 = 32 + e + b)
  (row2_sum : 20 + e + b = 32 + e + b)
  (row3_sum : c + 32 + a = 32 + e + b)
  (col1_sum : 30 + 20 + c = 32 + e + b)
  (col2_sum : d + e + 32 = 32 + e + b)
  (col3_sum : 24 + b + a = 32 + e + b)
  (diag1_sum : 30 + e + a = 32 + e + b)
  (diag2_sum : 24 + e + c = 32 + e + b)

/-- The sum of d and e in the magic square is 54 --/
theorem magic_square_sum (ms : MagicSquare) : ms.d + ms.e = 54 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_sum_l755_75520


namespace NUMINAMATH_CALUDE_original_group_size_is_correct_l755_75540

/-- Represents the number of men in the original group -/
def original_group_size : ℕ := 22

/-- Represents the number of days the original group planned to work -/
def original_days : ℕ := 20

/-- Represents the number of men who became absent -/
def absent_men : ℕ := 2

/-- Represents the number of days the remaining group worked -/
def actual_days : ℕ := 22

/-- Theorem stating that the original group size is correct given the conditions -/
theorem original_group_size_is_correct :
  (original_group_size : ℚ) * (actual_days : ℚ) * ((original_group_size - absent_men) : ℚ) = 
  (original_group_size : ℚ) * (original_group_size : ℚ) * (original_days : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_original_group_size_is_correct_l755_75540


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_sum_of_squares_l755_75532

theorem largest_prime_divisor_of_sum_of_squares : ∃ p : ℕ, 
  Nat.Prime p ∧ 
  p ∣ (36^2 + 49^2) ∧ 
  ∀ q : ℕ, Nat.Prime q → q ∣ (36^2 + 49^2) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_sum_of_squares_l755_75532


namespace NUMINAMATH_CALUDE_simplify_sqrt_18_l755_75547

theorem simplify_sqrt_18 : Real.sqrt 18 = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_18_l755_75547


namespace NUMINAMATH_CALUDE_solution_set_f_less_than_3_range_of_a_when_f_plus_g_greater_than_1_l755_75568

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := |3 * x - a|
def g (x : ℝ) : ℝ := |x + 1|

-- Statement for part (I)
theorem solution_set_f_less_than_3 (x : ℝ) :
  |3 * x - 4| < 3 ↔ 1/3 < x ∧ x < 7/3 :=
sorry

-- Statement for part (II)
theorem range_of_a_when_f_plus_g_greater_than_1 (a : ℝ) :
  (∀ x : ℝ, f a x + g x > 1) ↔ a < -6 ∨ a > 0 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_less_than_3_range_of_a_when_f_plus_g_greater_than_1_l755_75568


namespace NUMINAMATH_CALUDE_magnitude_relationship_l755_75549

-- Define the equations for a, b, and c
def equation_a (x : ℝ) : Prop := 2^x + x = 1
def equation_b (x : ℝ) : Prop := 2^x + x = 2
def equation_c (x : ℝ) : Prop := 3^x + x = 2

-- State the theorem
theorem magnitude_relationship (a b c : ℝ) 
  (ha : equation_a a) (hb : equation_b b) (hc : equation_c c) : 
  a < c ∧ c < b := by
  sorry

end NUMINAMATH_CALUDE_magnitude_relationship_l755_75549


namespace NUMINAMATH_CALUDE_sum_of_decimals_l755_75564

theorem sum_of_decimals : 5.67 + (-3.92) = 1.75 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_decimals_l755_75564


namespace NUMINAMATH_CALUDE_weight_of_a_l755_75505

/-- Given the weights of 5 people A, B, C, D, and E, prove that A weighs 64 kg -/
theorem weight_of_a (a b c d e : ℝ) : 
  (a + b + c) / 3 = 84 →
  (a + b + c + d) / 4 = 80 →
  e = d + 6 →
  (b + c + d + e) / 4 = 79 →
  a = 64 := by
sorry

end NUMINAMATH_CALUDE_weight_of_a_l755_75505


namespace NUMINAMATH_CALUDE_star_five_three_l755_75506

def star (a b : ℝ) : ℝ := 4 * a + 6 * b

theorem star_five_three : star 5 3 = 38 := by
  sorry

end NUMINAMATH_CALUDE_star_five_three_l755_75506


namespace NUMINAMATH_CALUDE_alfreds_savings_l755_75551

/-- Alfred's savings problem -/
theorem alfreds_savings (goal : ℝ) (months : ℕ) (monthly_savings : ℝ) 
  (h1 : goal = 1000)
  (h2 : months = 12)
  (h3 : monthly_savings = 75) :
  goal - (monthly_savings * months) = 100 := by
  sorry

end NUMINAMATH_CALUDE_alfreds_savings_l755_75551


namespace NUMINAMATH_CALUDE_equality_condition_l755_75508

theorem equality_condition (x y z a b c : ℝ) :
  (Real.sqrt (x + a) + Real.sqrt (y + b) + Real.sqrt (z + c) =
   Real.sqrt (y + a) + Real.sqrt (z + b) + Real.sqrt (x + c)) ∧
  (Real.sqrt (y + a) + Real.sqrt (z + b) + Real.sqrt (x + c) =
   Real.sqrt (z + a) + Real.sqrt (x + b) + Real.sqrt (y + c)) →
  (x = y ∧ y = z) ∨ (a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_equality_condition_l755_75508


namespace NUMINAMATH_CALUDE_min_lines_for_37_segments_l755_75524

/-- Represents an open non-self-intersecting broken line -/
structure BrokenLine where
  segments : ℕ
  is_open : Bool
  is_non_self_intersecting : Bool

/-- Represents the minimum number of lines needed to cover all segments of a broken line -/
def minimum_lines (bl : BrokenLine) : ℕ := sorry

/-- The theorem stating the minimum number of lines for a 37-segment broken line -/
theorem min_lines_for_37_segments (bl : BrokenLine) : 
  bl.segments = 37 → bl.is_open = true → bl.is_non_self_intersecting = true →
  minimum_lines bl = 9 := by sorry

end NUMINAMATH_CALUDE_min_lines_for_37_segments_l755_75524


namespace NUMINAMATH_CALUDE_sphere_volume_from_inscribed_cube_l755_75565

theorem sphere_volume_from_inscribed_cube (s : ℝ) (h : s > 0) :
  let cube_surface_area := 6 * s^2
  let cube_diagonal := s * Real.sqrt 3
  let sphere_radius := cube_diagonal / 2
  let sphere_volume := (4 / 3) * Real.pi * sphere_radius^3
  cube_surface_area = 24 → sphere_volume = 4 * Real.sqrt 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_inscribed_cube_l755_75565


namespace NUMINAMATH_CALUDE_factorization_cubic_minus_linear_l755_75542

theorem factorization_cubic_minus_linear (x : ℝ) : x^3 - 4*x = x*(x + 2)*(x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_cubic_minus_linear_l755_75542


namespace NUMINAMATH_CALUDE_model_y_completion_time_l755_75503

/-- The time (in minutes) taken by a Model Y computer to complete the task -/
def model_y_time : ℝ := 36

/-- The time (in minutes) taken by a Model X computer to complete the task -/
def model_x_time : ℝ := 72

/-- The number of Model X computers used -/
def num_computers : ℕ := 24

theorem model_y_completion_time :
  (num_computers : ℝ) * (1 / model_x_time + 1 / model_y_time) = 1 →
  model_y_time = 36 := by
  sorry

end NUMINAMATH_CALUDE_model_y_completion_time_l755_75503


namespace NUMINAMATH_CALUDE_base_equality_proof_l755_75582

/-- Converts a base-6 number to decimal --/
def base6ToDecimal (n : Nat) : Nat :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * 36 + tens * 6 + ones

/-- Converts a number in base b to decimal --/
def baseToDecimal (n : Nat) (b : Nat) : Nat :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * b^2 + tens * b + ones

theorem base_equality_proof : 
  ∃! (b : Nat), b > 0 ∧ base6ToDecimal 142 = baseToDecimal 215 b :=
by
  sorry

end NUMINAMATH_CALUDE_base_equality_proof_l755_75582


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l755_75507

theorem triangle_angle_measure (A B C : ℝ) (a b : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  a > 0 ∧ b > 0 →
  a^2 + b^2 = 6 * a * b * Real.cos C →
  Real.sin C^2 = 2 * Real.sin A * Real.sin B →
  C = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l755_75507


namespace NUMINAMATH_CALUDE_girls_in_class_l755_75588

theorem girls_in_class (total : ℕ) (boy_ratio girl_ratio : ℕ) (h1 : total = 260) (h2 : boy_ratio = 5) (h3 : girl_ratio = 8) :
  (girl_ratio * total) / (boy_ratio + girl_ratio) = 160 := by
sorry

end NUMINAMATH_CALUDE_girls_in_class_l755_75588


namespace NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l755_75557

def digit_sum (n : ℕ) : ℕ := 
  if n < 10 then n else n % 10 + digit_sum (n / 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem smallest_prime_with_digit_sum_23 :
  ∃ (p : ℕ), is_prime p ∧ digit_sum p = 23 ∧
  ∀ (q : ℕ), is_prime q → digit_sum q = 23 → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l755_75557


namespace NUMINAMATH_CALUDE_ellipse_dot_product_bound_l755_75515

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2/16 + y^2/12 = 1

-- Define point M
def M : ℝ × ℝ := (0, 2)

-- Define the dot product of two 2D vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Define the function to be bounded
def f (P Q : ℝ × ℝ) : ℝ :=
  dot_product (P.1, P.2) (Q.1, Q.2) + 
  dot_product (P.1 - M.1, P.2 - M.2) (Q.1 - M.1, Q.2 - M.2)

-- Theorem statement
theorem ellipse_dot_product_bound :
  ∀ P Q : ℝ × ℝ, C P.1 P.2 → C Q.1 Q.2 →
  ∃ k : ℝ, P.2 - M.2 = k * (P.1 - M.1) ∧ Q.2 - M.2 = k * (Q.1 - M.1) →
  -20 ≤ f P Q ∧ f P Q ≤ -52/3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_dot_product_bound_l755_75515


namespace NUMINAMATH_CALUDE_milk_bottle_boxes_l755_75585

/-- Given a total number of milk bottles, bottles per bag, and bags per box,
    calculate the total number of boxes. -/
def calculate_boxes (total_bottles : ℕ) (bottles_per_bag : ℕ) (bags_per_box : ℕ) : ℕ :=
  total_bottles / (bottles_per_bag * bags_per_box)

/-- Theorem stating that given 8640 milk bottles, with 12 bottles per bag and 6 bags per box,
    the total number of boxes is equal to 120. -/
theorem milk_bottle_boxes :
  calculate_boxes 8640 12 6 = 120 := by
  sorry

end NUMINAMATH_CALUDE_milk_bottle_boxes_l755_75585


namespace NUMINAMATH_CALUDE_craig_total_distance_l755_75530

/-- The distance Craig walked from school to David's house -/
def distance_school_to_david : ℝ := 0.2

/-- The distance Craig walked from David's house to his own house -/
def distance_david_to_home : ℝ := 0.7

/-- The total distance Craig walked -/
def total_distance : ℝ := distance_school_to_david + distance_david_to_home

/-- Theorem stating that the total distance Craig walked is 0.9 miles -/
theorem craig_total_distance : total_distance = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_craig_total_distance_l755_75530


namespace NUMINAMATH_CALUDE_cosine_sine_expression_value_l755_75583

theorem cosine_sine_expression_value : 
  Real.cos (10 * π / 180) * Real.sin (70 * π / 180) - 
  Real.cos (80 * π / 180) * Real.sin (20 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_cosine_sine_expression_value_l755_75583


namespace NUMINAMATH_CALUDE_empty_jar_weight_l755_75575

/-- Represents the weight of a jar with water -/
structure JarWeight where
  empty : ℝ  -- Weight of the empty jar
  water : ℝ  -- Weight of water when fully filled

/-- The weight of the jar when partially filled -/
def partialWeight (j : JarWeight) (fraction : ℝ) : ℝ :=
  j.empty + fraction * j.water

theorem empty_jar_weight (j : JarWeight) :
  (partialWeight j (1/5) = 560) →
  (partialWeight j (4/5) = 740) →
  j.empty = 500 := by
  sorry

end NUMINAMATH_CALUDE_empty_jar_weight_l755_75575


namespace NUMINAMATH_CALUDE_geometric_sequence_a7_l755_75535

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_a7 (a : ℕ → ℝ) :
  geometric_sequence a → a 3 = 3 → a 11 = 27 → a 7 = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a7_l755_75535


namespace NUMINAMATH_CALUDE_blue_marble_percent_is_35_l755_75571

/-- Represents the composition of items in an urn -/
structure UrnComposition where
  button_percent : ℝ
  red_marble_percent : ℝ
  blue_marble_percent : ℝ

/-- The percentage of blue marbles in the urn -/
def blue_marble_percentage (urn : UrnComposition) : ℝ :=
  urn.blue_marble_percent

/-- Theorem stating the percentage of blue marbles in the urn -/
theorem blue_marble_percent_is_35 (urn : UrnComposition) 
  (h1 : urn.button_percent = 0.3)
  (h2 : urn.red_marble_percent = 0.5 * (1 - urn.button_percent)) :
  blue_marble_percentage urn = 0.35 := by
  sorry

#check blue_marble_percent_is_35

end NUMINAMATH_CALUDE_blue_marble_percent_is_35_l755_75571


namespace NUMINAMATH_CALUDE_fixed_point_of_linear_function_l755_75537

def linear_function (k b x : ℝ) : ℝ := k * x + b

theorem fixed_point_of_linear_function (k b : ℝ) 
  (h : 3 * k - b = 2) : 
  linear_function k b (-3) = -2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_linear_function_l755_75537


namespace NUMINAMATH_CALUDE_arithmetic_progression_11_arithmetic_progression_10000_no_infinite_arithmetic_progression_l755_75523

-- Define a function to calculate the sum of digits of a natural number
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Define what it means for a sequence to be an arithmetic progression
def isArithmeticProgression (seq : ℕ → ℕ) : Prop := sorry

-- Define what it means for a sequence to be increasing
def isIncreasing (seq : ℕ → ℕ) : Prop := sorry

-- Theorem for the case of 11 terms
theorem arithmetic_progression_11 : 
  ∃ (seq : Fin 11 → ℕ), 
    isArithmeticProgression (λ i => seq i) ∧ 
    isIncreasing (λ i => seq i) ∧
    isArithmeticProgression (λ i => sumOfDigits (seq i)) ∧
    isIncreasing (λ i => sumOfDigits (seq i)) := sorry

-- Theorem for the case of 10,000 terms
theorem arithmetic_progression_10000 : 
  ∃ (seq : Fin 10000 → ℕ), 
    isArithmeticProgression (λ i => seq i) ∧ 
    isIncreasing (λ i => seq i) ∧
    isArithmeticProgression (λ i => sumOfDigits (seq i)) ∧
    isIncreasing (λ i => sumOfDigits (seq i)) := sorry

-- Theorem for the case of infinite natural numbers
theorem no_infinite_arithmetic_progression :
  ¬∃ (seq : ℕ → ℕ), 
    isArithmeticProgression seq ∧ 
    isIncreasing seq ∧
    isArithmeticProgression (λ n => sumOfDigits (seq n)) ∧
    isIncreasing (λ n => sumOfDigits (seq n)) := sorry

end NUMINAMATH_CALUDE_arithmetic_progression_11_arithmetic_progression_10000_no_infinite_arithmetic_progression_l755_75523


namespace NUMINAMATH_CALUDE_dhoni_rent_percentage_dhoni_rent_percentage_proof_l755_75519

theorem dhoni_rent_percentage : ℝ → Prop :=
  fun rent_percentage =>
    let dishwasher_percentage := rent_percentage - 5
    let leftover_percentage := 61
    rent_percentage + dishwasher_percentage + leftover_percentage = 100 →
    rent_percentage = 22

-- The proof is omitted
theorem dhoni_rent_percentage_proof : dhoni_rent_percentage 22 := by sorry

end NUMINAMATH_CALUDE_dhoni_rent_percentage_dhoni_rent_percentage_proof_l755_75519


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l755_75544

theorem quadratic_inequality_solution_range (c : ℝ) : 
  (c > 0 ∧ ∃ x : ℝ, x^2 - 8*x + c > 0) ↔ (0 < c ∧ c < 16) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l755_75544


namespace NUMINAMATH_CALUDE_reciprocal_geometric_progression_sum_l755_75569

theorem reciprocal_geometric_progression_sum
  (n : ℕ)  -- number of terms divided by 2
  (r : ℝ)  -- half of the common ratio
  (S : ℝ)  -- sum of the original geometric progression
  (h1 : S = (1 - (2*r)^(2*n)) / (1 - 2*r))  -- definition of S
  : (1 - (1/(2*r))^(2*n)) / (1 - 1/(2*r)) = S / (2^n * r^(2*n-1)) :=
sorry

end NUMINAMATH_CALUDE_reciprocal_geometric_progression_sum_l755_75569


namespace NUMINAMATH_CALUDE_amys_net_earnings_result_l755_75509

/-- Calculates Amy's net earnings for a week given her daily work details and tax rate -/
def amys_net_earnings (day1_hours day1_rate day1_tips day1_bonus : ℝ)
                      (day2_hours day2_rate day2_tips : ℝ)
                      (day3_hours day3_rate day3_tips : ℝ)
                      (day4_hours day4_rate day4_tips day4_overtime : ℝ)
                      (day5_hours day5_rate day5_tips : ℝ)
                      (tax_rate : ℝ) : ℝ :=
  let day1_earnings := day1_hours * day1_rate + day1_tips + day1_bonus
  let day2_earnings := day2_hours * day2_rate + day2_tips
  let day3_earnings := day3_hours * day3_rate + day3_tips
  let day4_earnings := day4_hours * day4_rate + day4_tips + day4_overtime
  let day5_earnings := day5_hours * day5_rate + day5_tips
  let gross_earnings := day1_earnings + day2_earnings + day3_earnings + day4_earnings + day5_earnings
  let taxes := tax_rate * gross_earnings
  gross_earnings - taxes

/-- Theorem stating that Amy's net earnings for the week are $118.58 -/
theorem amys_net_earnings_result :
  amys_net_earnings 4 3 6 10 6 4 7 3 5 2 5 3.5 8 5 7 4 5 0.15 = 118.58 := by
  sorry

#eval amys_net_earnings 4 3 6 10 6 4 7 3 5 2 5 3.5 8 5 7 4 5 0.15

end NUMINAMATH_CALUDE_amys_net_earnings_result_l755_75509


namespace NUMINAMATH_CALUDE_total_trees_planted_l755_75570

theorem total_trees_planted (total_gardeners : ℕ) (street_a_gardeners : ℕ) (street_b_gardeners : ℕ) 
  (h1 : total_gardeners = street_a_gardeners + street_b_gardeners)
  (h2 : total_gardeners = 19)
  (h3 : street_a_gardeners = 4)
  (h4 : street_b_gardeners = 15)
  (h5 : ∃ x : ℕ, street_b_gardeners * x - 1 = 4 * (street_a_gardeners * x - 1)) :
  ∃ trees_per_gardener : ℕ, total_gardeners * trees_per_gardener = 57 :=
by sorry

end NUMINAMATH_CALUDE_total_trees_planted_l755_75570


namespace NUMINAMATH_CALUDE_computation_problem_value_l755_75593

theorem computation_problem_value (total_problems : Nat) (word_problem_value : Nat) 
  (total_points : Nat) (computation_problems : Nat) :
  total_problems = 30 →
  word_problem_value = 5 →
  total_points = 110 →
  computation_problems = 20 →
  ∃ (computation_value : Nat),
    computation_value = 3 ∧
    total_points = computation_problems * computation_value + 
      (total_problems - computation_problems) * word_problem_value :=
by sorry

end NUMINAMATH_CALUDE_computation_problem_value_l755_75593


namespace NUMINAMATH_CALUDE_consecutive_integers_average_l755_75502

theorem consecutive_integers_average (a b c d e : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 →
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 →
  (a + b + c + d + e) / 5 = 8 →
  e - a = 4 →
  (b + d) / 2 = 8 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_average_l755_75502


namespace NUMINAMATH_CALUDE_tensor_plus_relation_l755_75511

-- Define a structure for pairs of real numbers
structure Pair :=
  (x : ℝ)
  (y : ℝ)

-- Define equality for pairs
def pair_eq (a b : Pair) : Prop :=
  a.x = b.x ∧ a.y = b.y

-- Define the ⊗ operation
def tensor (a b : Pair) : Pair :=
  ⟨a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x⟩

-- Define the ⊕ operation
def plus (a b : Pair) : Pair :=
  ⟨a.x + b.x, a.y + b.y⟩

-- State the theorem
theorem tensor_plus_relation (p q : ℝ) :
  pair_eq (tensor ⟨1, 2⟩ ⟨p, q⟩) ⟨5, 0⟩ →
  pair_eq (plus ⟨1, 2⟩ ⟨p, q⟩) ⟨2, 0⟩ :=
by
  sorry

end NUMINAMATH_CALUDE_tensor_plus_relation_l755_75511


namespace NUMINAMATH_CALUDE_equation_solution_l755_75554

theorem equation_solution (x : ℝ) (h : x ≠ 3) : (x + 6) / (x - 3) = 4 ↔ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l755_75554


namespace NUMINAMATH_CALUDE_sum_of_radii_tangent_circles_l755_75586

/-- The sum of radii of a circle tangent to x and y axes and externally tangent to another circle -/
theorem sum_of_radii_tangent_circles : ∀ r : ℝ,
  (r > 0) →
  ((r - 5)^2 + r^2 = (r + 2)^2) →
  ∃ r₁ r₂ : ℝ, (r = r₁ ∨ r = r₂) ∧ (r₁ + r₂ = 14) := by
sorry

end NUMINAMATH_CALUDE_sum_of_radii_tangent_circles_l755_75586


namespace NUMINAMATH_CALUDE_scalar_mult_assoc_l755_75545

variable (V : Type*) [NormedAddCommGroup V] [NormedSpace ℝ V]

theorem scalar_mult_assoc (a : V) (h : a ≠ 0) :
  (-4 : ℝ) • (3 • a) = (-12 : ℝ) • a := by sorry

end NUMINAMATH_CALUDE_scalar_mult_assoc_l755_75545
