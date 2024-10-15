import Mathlib

namespace NUMINAMATH_CALUDE_largest_negative_integer_l3186_318622

theorem largest_negative_integer :
  ∃! n : ℤ, n < 0 ∧ ∀ m : ℤ, m < 0 → m ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_largest_negative_integer_l3186_318622


namespace NUMINAMATH_CALUDE_households_with_car_l3186_318633

theorem households_with_car (total : Nat) (without_car_or_bike : Nat) (with_both : Nat) (with_bike_only : Nat)
  (h1 : total = 90)
  (h2 : without_car_or_bike = 11)
  (h3 : with_both = 18)
  (h4 : with_bike_only = 35) :
  total - without_car_or_bike - with_bike_only + with_both = 62 := by
sorry

end NUMINAMATH_CALUDE_households_with_car_l3186_318633


namespace NUMINAMATH_CALUDE_leap_year_53_sundays_5_feb_sundays_probability_l3186_318654

/-- Represents a day of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a leap year -/
structure LeapYear where
  days : Fin 366
  sundays : Nat
  februarySundays : Nat

/-- The probability of a specific configuration of extra days in a leap year -/
def extraDaysProbability : ℚ := 1 / 7

/-- The probability of a leap year having 53 Sundays -/
def prob53Sundays : ℚ := 2 / 7

/-- The probability of February in a leap year having 5 Sundays -/
def probFeb5Sundays : ℚ := 1 / 7

/-- 
Theorem: The probability of a randomly selected leap year having 53 Sundays, 
with exactly 5 of those Sundays falling in February, is 2/49.
-/
theorem leap_year_53_sundays_5_feb_sundays_probability : 
  prob53Sundays * probFeb5Sundays = 2 / 49 := by
  sorry

end NUMINAMATH_CALUDE_leap_year_53_sundays_5_feb_sundays_probability_l3186_318654


namespace NUMINAMATH_CALUDE_rebecca_camping_items_l3186_318631

/-- The number of items Rebecca bought for her camping trip -/
def total_items (tent_stakes drink_mix water : ℕ) : ℕ :=
  tent_stakes + drink_mix + water

/-- Theorem stating the total number of items Rebecca bought -/
theorem rebecca_camping_items : ∃ (tent_stakes drink_mix water : ℕ),
  tent_stakes = 4 ∧
  drink_mix = 3 * tent_stakes ∧
  water = tent_stakes + 2 ∧
  total_items tent_stakes drink_mix water = 22 := by
  sorry

end NUMINAMATH_CALUDE_rebecca_camping_items_l3186_318631


namespace NUMINAMATH_CALUDE_rectangular_prism_surface_area_increase_l3186_318674

theorem rectangular_prism_surface_area_increase 
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  let original_surface_area := 2 * (a * b + b * c + a * c)
  let new_surface_area := 2 * ((1.8 * a) * (1.8 * b) + (1.8 * b) * (1.8 * c) + (1.8 * c) * (1.8 * a))
  (new_surface_area - original_surface_area) / original_surface_area = 2.24 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_surface_area_increase_l3186_318674


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3186_318605

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  -- Add necessary conditions for a valid triangle
  pos_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π
  -- Add cosine law
  cos_law_c : c^2 = a^2 + b^2 - 2*a*b*Real.cos C

-- Define what it means for a triangle to be isosceles
def isIsosceles (t : Triangle) : Prop := t.b = t.c ∨ t.a = t.c ∨ t.a = t.b

-- State the theorem
theorem sufficient_not_necessary_condition (t : Triangle) :
  (t.a = 2 * t.b * Real.cos t.C → isIsosceles t) ∧
  ∃ t' : Triangle, isIsosceles t' ∧ t'.a ≠ 2 * t'.b * Real.cos t'.C :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3186_318605


namespace NUMINAMATH_CALUDE_rearrangement_maintains_ratio_l3186_318638

/-- Represents a figure made of sticks -/
structure StickFigure where
  num_sticks : ℕ
  area : ℝ

/-- The total number of sticks -/
def total_sticks : ℕ := 20

/-- The number of fixed sticks -/
def fixed_sticks : ℕ := 12

/-- The initial figure with 14 sticks -/
def initial_figure_14 : StickFigure := ⟨14, 3⟩

/-- The initial figure with 6 sticks -/
def initial_figure_6 : StickFigure := ⟨6, 1⟩

/-- The rearranged figure with 7 sticks -/
def rearranged_figure_7 : StickFigure := ⟨7, 1⟩

/-- The rearranged figure with 13 sticks -/
def rearranged_figure_13 : StickFigure := ⟨13, 3⟩

/-- Theorem stating that the rearrangement maintains the area ratio -/
theorem rearrangement_maintains_ratio :
  (initial_figure_14.area / initial_figure_6.area = 
   rearranged_figure_13.area / rearranged_figure_7.area) ∧
  (total_sticks = initial_figure_14.num_sticks + initial_figure_6.num_sticks) ∧
  (total_sticks = rearranged_figure_13.num_sticks + rearranged_figure_7.num_sticks) ∧
  (fixed_sticks + rearranged_figure_13.num_sticks - rearranged_figure_7.num_sticks = initial_figure_14.num_sticks) :=
by sorry

end NUMINAMATH_CALUDE_rearrangement_maintains_ratio_l3186_318638


namespace NUMINAMATH_CALUDE_experiment_arrangements_l3186_318645

/-- Represents the number of procedures in the experiment -/
def num_procedures : ℕ := 6

/-- Represents whether procedure A is at the beginning or end -/
inductive A_position
| beginning
| end

/-- Calculates the number of arrangements for a given A position -/
def arrangements_for_A_position (pos : A_position) : ℕ := 
  (Nat.factorial (num_procedures - 3)) * 2

/-- Calculates the total number of possible arrangements -/
def total_arrangements : ℕ :=
  arrangements_for_A_position A_position.beginning + 
  arrangements_for_A_position A_position.end

/-- Theorem stating that the total number of arrangements is 96 -/
theorem experiment_arrangements :
  total_arrangements = 96 := by sorry

end NUMINAMATH_CALUDE_experiment_arrangements_l3186_318645


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_not_regular_polygon_l3186_318680

/-- An isosceles right triangle -/
structure IsoscelesRightTriangle where
  /-- The triangle has two equal angles -/
  has_two_equal_angles : Bool
  /-- The triangle has a right angle -/
  has_right_angle : Bool
  /-- All isosceles right triangles are similar -/
  always_similar : Bool
  /-- The triangle has two equal sides -/
  has_two_equal_sides : Bool

/-- A regular polygon -/
structure RegularPolygon where
  /-- All sides are equal -/
  equilateral : Bool
  /-- All angles are equal -/
  equiangular : Bool

/-- Theorem: Isosceles right triangles are not regular polygons -/
theorem isosceles_right_triangle_not_regular_polygon (t : IsoscelesRightTriangle) : 
  ¬∃(p : RegularPolygon), (t.has_two_equal_angles ∧ t.has_right_angle ∧ t.always_similar ∧ t.has_two_equal_sides) → 
  (p.equilateral ∧ p.equiangular) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_not_regular_polygon_l3186_318680


namespace NUMINAMATH_CALUDE_undominated_implies_favorite_toy_l3186_318687

/-- A type representing children -/
def Child : Type := Nat

/-- A type representing toys -/
def Toy : Type := Nat

/-- A type representing a preference ordering of toys for a child -/
def Preference := Toy → Toy → Prop

/-- A type representing a distribution of toys to children -/
def Distribution := Child → Toy

/-- Predicate indicating if a toy is preferred over another for a given child's preference -/
def IsPreferred (pref : Preference) (t1 t2 : Toy) : Prop := pref t1 t2 ∧ ¬pref t2 t1

/-- Predicate indicating if a distribution is dominated by another -/
def Dominates (prefs : Child → Preference) (d1 d2 : Distribution) : Prop :=
  ∀ c : Child, IsPreferred (prefs c) (d1 c) (d2 c) ∨ d1 c = d2 c

/-- Predicate indicating if a toy is the favorite for a child -/
def IsFavorite (pref : Preference) (t : Toy) : Prop :=
  ∀ t' : Toy, t ≠ t' → IsPreferred pref t t'

theorem undominated_implies_favorite_toy
  (n : Nat)
  (prefs : Child → Preference)
  (d : Distribution)
  (h_strict : ∀ c : Child, ∀ t1 t2 : Toy, t1 ≠ t2 → (IsPreferred (prefs c) t1 t2 ∨ IsPreferred (prefs c) t2 t1))
  (h_undominated : ∀ d' : Distribution, ¬Dominates prefs d' d) :
  ∃ c : Child, IsFavorite (prefs c) (d c) :=
sorry

end NUMINAMATH_CALUDE_undominated_implies_favorite_toy_l3186_318687


namespace NUMINAMATH_CALUDE_b_formula_T_formula_l3186_318682

/-- An arithmetic sequence with first term 1 and common difference 1 -/
def arithmetic_sequence (n : ℕ) : ℕ := n

/-- Sum of the first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sequence b_n defined as 1/S_n -/
def b (n : ℕ) : ℚ := 1 / (S n)

/-- Sum of the first n terms of the sequence b_n -/
def T (n : ℕ) : ℚ := sorry

theorem b_formula (n : ℕ) : b n = 2 / (n * (n + 1)) :=
  sorry

theorem T_formula (n : ℕ) : T n = 2 * n / (n + 1) :=
  sorry

end NUMINAMATH_CALUDE_b_formula_T_formula_l3186_318682


namespace NUMINAMATH_CALUDE_max_inscribed_circle_radius_l3186_318621

/-- The maximum radius of an inscribed circle centered at (0,0) in the curve |y| = 1 - a x^2 where |x| ≤ 1/√a -/
noncomputable def f (a : ℝ) : ℝ :=
  if a ≤ 1/2 then 1 else Real.sqrt (4*a - 1) / (2*a)

/-- The curve C defined by |y| = 1 - a x^2 where |x| ≤ 1/√a -/
def C (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |p.2| = 1 - a * p.1^2 ∧ |p.1| ≤ 1/Real.sqrt a}

theorem max_inscribed_circle_radius (a : ℝ) (ha : a > 0) :
  ∀ r : ℝ, r > 0 → (∀ p : ℝ × ℝ, p ∈ C a → (p.1^2 + p.2^2 ≥ r^2)) → r ≤ f a :=
sorry

end NUMINAMATH_CALUDE_max_inscribed_circle_radius_l3186_318621


namespace NUMINAMATH_CALUDE_remainder_problem_l3186_318643

theorem remainder_problem (N : ℕ) (D : ℕ) : 
  (N % 158 = 50) → (N % D = 13) → (D > 13) → (D < 158) → D = 37 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3186_318643


namespace NUMINAMATH_CALUDE_circle1_correct_circle2_correct_l3186_318624

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x - 8)^2 + (y + 3)^2 = 25
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y - 20 = 0

-- Define the points
def point_A1 : ℝ × ℝ := (5, 1)
def point_A2 : ℝ × ℝ := (-1, 5)
def point_B2 : ℝ × ℝ := (5, 5)
def point_C2 : ℝ × ℝ := (6, -2)

-- Theorem for circle 1
theorem circle1_correct :
  circle1 (point_A1.1) (point_A1.2) ∧
  ∀ (x y : ℝ), circle1 x y → (x - 8)^2 + (y + 3)^2 = 25 := by sorry

-- Theorem for circle 2
theorem circle2_correct :
  circle2 (point_A2.1) (point_A2.2) ∧
  circle2 (point_B2.1) (point_B2.2) ∧
  circle2 (point_C2.1) (point_C2.2) ∧
  ∀ (x y : ℝ), circle2 x y → x^2 + y^2 - 4*x - 2*y - 20 = 0 := by sorry

end NUMINAMATH_CALUDE_circle1_correct_circle2_correct_l3186_318624


namespace NUMINAMATH_CALUDE_train_length_train_length_problem_l3186_318656

/-- The length of a train given its speed, the speed of a person running in the opposite direction, and the time it takes for the train to pass the person. -/
theorem train_length (train_speed : ℝ) (person_speed : ℝ) (passing_time : ℝ) : ℝ :=
  let relative_speed := train_speed + person_speed
  let relative_speed_mps := relative_speed * 1000 / 3600
  relative_speed_mps * passing_time

/-- Proof that a train with speed 56 km/hr passing a man running at 6 km/hr in the opposite direction in 6.386585847325762 seconds has a length of approximately 110 meters. -/
theorem train_length_problem : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |train_length 56 6 6.386585847325762 - 110| < ε :=
sorry

end NUMINAMATH_CALUDE_train_length_train_length_problem_l3186_318656


namespace NUMINAMATH_CALUDE_arithmetic_sequence_n_is_27_l3186_318636

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  first : a 1 = 20
  last : ∃ n : ℕ, a n = 54
  sum : ∃ n : ℕ, (n : ℝ) / 2 * (a 1 + a n) = 999

/-- The number of terms in the arithmetic sequence is 27 -/
theorem arithmetic_sequence_n_is_27 (seq : ArithmeticSequence) : 
  ∃ n : ℕ, n = 27 ∧ seq.a n = 54 ∧ (n : ℝ) / 2 * (seq.a 1 + seq.a n) = 999 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_n_is_27_l3186_318636


namespace NUMINAMATH_CALUDE_football_campers_count_l3186_318616

theorem football_campers_count (total : ℕ) (basketball : ℕ) (soccer : ℕ) 
  (h1 : total = 88) 
  (h2 : basketball = 24) 
  (h3 : soccer = 32) : 
  total - soccer - basketball = 32 := by
sorry

end NUMINAMATH_CALUDE_football_campers_count_l3186_318616


namespace NUMINAMATH_CALUDE_different_suit_combinations_l3186_318696

/-- The number of cards in a standard deck -/
def standard_deck_size : ℕ := 52

/-- The number of suits in a standard deck -/
def number_of_suits : ℕ := 4

/-- The number of cards per suit in a standard deck -/
def cards_per_suit : ℕ := 13

/-- The number of cards to be chosen -/
def cards_to_choose : ℕ := 4

/-- Theorem stating the number of ways to choose 4 cards of different suits from a standard deck -/
theorem different_suit_combinations : 
  (number_of_suits.choose cards_to_choose) * (cards_per_suit ^ cards_to_choose) = 28561 := by
  sorry

end NUMINAMATH_CALUDE_different_suit_combinations_l3186_318696


namespace NUMINAMATH_CALUDE_central_cell_value_l3186_318667

/-- A 3x3 table of real numbers -/
structure Table :=
  (a b c d e f g h i : ℝ)

/-- The conditions for the 3x3 table -/
def satisfies_conditions (t : Table) : Prop :=
  t.a * t.b * t.c = 10 ∧
  t.d * t.e * t.f = 10 ∧
  t.g * t.h * t.i = 10 ∧
  t.a * t.d * t.g = 10 ∧
  t.b * t.e * t.h = 10 ∧
  t.c * t.f * t.i = 10 ∧
  t.a * t.b * t.d * t.e = 3 ∧
  t.b * t.c * t.e * t.f = 3 ∧
  t.d * t.e * t.g * t.h = 3 ∧
  t.e * t.f * t.h * t.i = 3

/-- The theorem stating that the central cell value is 0.00081 -/
theorem central_cell_value (t : Table) (h : satisfies_conditions t) : t.e = 0.00081 := by
  sorry

end NUMINAMATH_CALUDE_central_cell_value_l3186_318667


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3186_318681

theorem quadratic_inequality (x : ℝ) : x^2 + x - 12 > 0 ↔ x > 3 ∨ x < -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3186_318681


namespace NUMINAMATH_CALUDE_arrange_75510_eq_48_l3186_318671

/-- The number of ways to arrange the digits of 75,510 to form a 5-digit number not beginning with '0' -/
def arrange_75510 : ℕ :=
  let digits : List ℕ := [7, 5, 5, 1, 0]
  let total_digits := digits.length
  let non_zero_digits := digits.filter (· ≠ 0)
  let zero_count := total_digits - non_zero_digits.length
  let non_zero_permutations := Nat.factorial non_zero_digits.length / 
    (Nat.factorial 2 * Nat.factorial (non_zero_digits.length - 2))
  (total_digits - 1) * non_zero_permutations

theorem arrange_75510_eq_48 : arrange_75510 = 48 := by
  sorry

end NUMINAMATH_CALUDE_arrange_75510_eq_48_l3186_318671


namespace NUMINAMATH_CALUDE_largest_number_l3186_318620

theorem largest_number (a b c d e : ℝ) : 
  a = 15467 + 3 / 5791 → 
  b = 15467 - 3 / 5791 → 
  c = 15467 * 3 / 5791 → 
  d = 15467 / (3 / 5791) → 
  e = 15467.5791 → 
  d > a ∧ d > b ∧ d > c ∧ d > e :=
by sorry

end NUMINAMATH_CALUDE_largest_number_l3186_318620


namespace NUMINAMATH_CALUDE_liam_and_sisters_ages_l3186_318676

theorem liam_and_sisters_ages (a b : ℕ+) (h1 : a < b) (h2 : a * b * b = 72) : 
  a + b + b = 14 := by
sorry

end NUMINAMATH_CALUDE_liam_and_sisters_ages_l3186_318676


namespace NUMINAMATH_CALUDE_second_largest_is_five_l3186_318660

def number_set : Finset ℕ := {5, 8, 4, 3, 2}

theorem second_largest_is_five :
  ∃ (x : ℕ), x ∈ number_set ∧ 
  (∀ y ∈ number_set, y ≠ x → y ≤ x) ∧
  (∃ z ∈ number_set, z > x) ∧
  x = 5 := by
  sorry

end NUMINAMATH_CALUDE_second_largest_is_five_l3186_318660


namespace NUMINAMATH_CALUDE_smaller_mold_radius_prove_smaller_mold_radius_l3186_318698

/-- The radius of smaller hemisphere-shaped molds when jelly from a larger hemisphere
    is evenly distributed -/
theorem smaller_mold_radius (large_radius : ℝ) (num_small_molds : ℕ) : ℝ :=
  let large_volume := (2 / 3) * Real.pi * large_radius ^ 3
  let small_radius := (large_volume / (num_small_molds * ((2 / 3) * Real.pi))) ^ (1 / 3)
  small_radius

/-- Prove that the radius of each smaller mold is 1 / (2^(2/3)) feet -/
theorem prove_smaller_mold_radius :
  smaller_mold_radius 2 64 = 1 / (2 ^ (2 / 3)) := by
  sorry

end NUMINAMATH_CALUDE_smaller_mold_radius_prove_smaller_mold_radius_l3186_318698


namespace NUMINAMATH_CALUDE_first_division_meiosis_characteristics_l3186_318650

/-- Represents the behavior of chromosomes during cell division -/
inductive ChromosomeBehavior
  | separate
  | notSeparate

/-- Represents the behavior of centromeres during cell division -/
inductive CentromereBehavior
  | split
  | notSplit

/-- Represents the characteristics of a cell division -/
structure CellDivisionCharacteristics where
  chromosomeBehavior : ChromosomeBehavior
  centromereBehavior : CentromereBehavior

/-- Represents the first division of meiosis -/
def firstDivisionMeiosis : CellDivisionCharacteristics := sorry

/-- Theorem stating the characteristics of the first division of meiosis -/
theorem first_division_meiosis_characteristics :
  firstDivisionMeiosis.chromosomeBehavior = ChromosomeBehavior.separate ∧
  firstDivisionMeiosis.centromereBehavior = CentromereBehavior.notSplit :=
sorry

end NUMINAMATH_CALUDE_first_division_meiosis_characteristics_l3186_318650


namespace NUMINAMATH_CALUDE_inequality_proof_l3186_318669

theorem inequality_proof (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_four : a + b + c + d = 4) :
  a^2 * b * c + b^2 * d * a + c^2 * d * a + d^2 * b * c ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3186_318669


namespace NUMINAMATH_CALUDE_abc_product_l3186_318630

theorem abc_product (a b c : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h_sum : a + b + c = 30)
  (h_eq : (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c + 672 / (a * b * c) = 1) :
  a * b * c = 2808 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_l3186_318630


namespace NUMINAMATH_CALUDE_alcohol_concentration_after_dilution_l3186_318611

theorem alcohol_concentration_after_dilution
  (original_volume : ℝ)
  (original_concentration : ℝ)
  (added_water : ℝ)
  (h1 : original_volume = 24)
  (h2 : original_concentration = 0.9)
  (h3 : added_water = 16) :
  let alcohol_volume := original_volume * original_concentration
  let new_volume := original_volume + added_water
  let new_concentration := alcohol_volume / new_volume
  new_concentration = 0.54 := by
sorry

end NUMINAMATH_CALUDE_alcohol_concentration_after_dilution_l3186_318611


namespace NUMINAMATH_CALUDE_michael_and_brothers_ages_l3186_318695

/-- The ages of Michael and his brothers satisfy the given conditions and their combined age is 28. -/
theorem michael_and_brothers_ages :
  ∀ (michael_age older_brother_age younger_brother_age : ℕ),
    younger_brother_age = 5 →
    older_brother_age = 3 * younger_brother_age →
    older_brother_age = 1 + 2 * (michael_age - 1) →
    michael_age + older_brother_age + younger_brother_age = 28 :=
by
  sorry


end NUMINAMATH_CALUDE_michael_and_brothers_ages_l3186_318695


namespace NUMINAMATH_CALUDE_conic_section_properties_l3186_318615

/-- A conic section defined by the equation x^2 + x + 2y - 2 = 0 -/
def conic_section (x y : ℝ) : Prop := x^2 + x + 2*y - 2 = 0

/-- The first line: x - 2y + 3 = 0 -/
def line1 (x y : ℝ) : Prop := x - 2*y + 3 = 0

/-- The second line: 5x + 2y - 6 = 0 -/
def line2 (x y : ℝ) : Prop := 5*x + 2*y - 6 = 0

/-- Point P -/
def P : ℝ × ℝ := (-1, 1)

/-- Point Q -/
def Q : ℝ × ℝ := (2, -2)

/-- Point R -/
def R : ℝ × ℝ := (1, 0)

/-- The conic section is tangent to line1 at point P, tangent to line2 at point Q, and passes through point R -/
theorem conic_section_properties :
  (conic_section P.1 P.2 ∧ line1 P.1 P.2) ∧
  (conic_section Q.1 Q.2 ∧ line2 Q.1 Q.2) ∧
  conic_section R.1 R.2 :=
sorry

end NUMINAMATH_CALUDE_conic_section_properties_l3186_318615


namespace NUMINAMATH_CALUDE_sum_of_multiples_l3186_318625

theorem sum_of_multiples (p q : ℤ) : 
  (∃ m : ℤ, p = 5 * m) → (∃ n : ℤ, q = 10 * n) → (∃ k : ℤ, p + q = 5 * k) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_multiples_l3186_318625


namespace NUMINAMATH_CALUDE_vector_MN_l3186_318658

def M : ℝ × ℝ := (-3, 3)
def N : ℝ × ℝ := (-5, -1)

theorem vector_MN : N.1 - M.1 = -2 ∧ N.2 - M.2 = -4 := by sorry

end NUMINAMATH_CALUDE_vector_MN_l3186_318658


namespace NUMINAMATH_CALUDE_smallest_n_for_probability_threshold_l3186_318689

def P (n : ℕ) : ℚ := 3 / ((n + 1) * (n + 2) * (n + 3))

theorem smallest_n_for_probability_threshold : 
  ∀ k : ℕ, k ≥ 1 → (P k < 1 / 3015 ↔ k ≥ 19) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_probability_threshold_l3186_318689


namespace NUMINAMATH_CALUDE_solve_bucket_problem_l3186_318657

def bucket_problem (b1 b2 b3 b4 b5 : ℕ) : Prop :=
  b1 = 11 ∧ b2 = 13 ∧ b3 = 12 ∧ b4 = 16 ∧ b5 = 10 →
  (b5 + b2 = 23) →
  (b1 + b3 + b4 = 39)

theorem solve_bucket_problem :
  ∀ b1 b2 b3 b4 b5 : ℕ, bucket_problem b1 b2 b3 b4 b5 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_bucket_problem_l3186_318657


namespace NUMINAMATH_CALUDE_equation_solution_l3186_318662

theorem equation_solution : ∃ x : ℝ, 4*x + 4 - x - 2*x + 2 - 2 - x + 2 + 6 = 0 ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3186_318662


namespace NUMINAMATH_CALUDE_option_a_same_function_option_b_different_function_option_c_different_domain_option_d_same_function_l3186_318691

-- Option A
theorem option_a_same_function (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by sorry

-- Option B
theorem option_b_different_function : ∃ x : ℤ, 2*x + 1 ≠ 2*x - 1 := by sorry

-- Option C
def domain_f (x : ℝ) : Prop := x^2 ≥ 9
def domain_g (x : ℝ) : Prop := x ≥ 3

theorem option_c_different_domain : domain_f ≠ domain_g := by sorry

-- Option D
theorem option_d_same_function (x t : ℝ) (h : x = t) : x^2 - 2*x - 1 = t^2 - 2*t - 1 := by sorry

end NUMINAMATH_CALUDE_option_a_same_function_option_b_different_function_option_c_different_domain_option_d_same_function_l3186_318691


namespace NUMINAMATH_CALUDE_greatest_integer_sqrt_l3186_318659

theorem greatest_integer_sqrt (N : ℤ) : 
  (∀ m : ℤ, m ≤ Real.sqrt (2007^2 - 20070 + 31) → m ≤ N) ∧ 
  N ≤ Real.sqrt (2007^2 - 20070 + 31) → 
  N = 2002 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_sqrt_l3186_318659


namespace NUMINAMATH_CALUDE_iris_rose_ratio_l3186_318623

/-- Proves that given an initial ratio of irises to roses of 2:5, 
    with 25 roses initially and 20 roses added, 
    maintaining the same ratio results in a total of 18 irises. -/
theorem iris_rose_ratio (initial_roses : ℕ) (added_roses : ℕ) 
  (iris_ratio : ℕ) (rose_ratio : ℕ) : 
  initial_roses = 25 →
  added_roses = 20 →
  iris_ratio = 2 →
  rose_ratio = 5 →
  (iris_ratio : ℚ) / rose_ratio * (initial_roses + added_roses) = 18 := by
  sorry

#check iris_rose_ratio

end NUMINAMATH_CALUDE_iris_rose_ratio_l3186_318623


namespace NUMINAMATH_CALUDE_fathers_age_is_32_l3186_318601

/-- The son's current age -/
def sons_age : ℕ := 16

/-- The father's current age -/
def fathers_age : ℕ := 32

/-- Theorem stating that the father's age is 32 -/
theorem fathers_age_is_32 :
  (fathers_age - sons_age = sons_age) ∧ 
  (sons_age = 11 + 5) →
  fathers_age = 32 := by sorry

end NUMINAMATH_CALUDE_fathers_age_is_32_l3186_318601


namespace NUMINAMATH_CALUDE_no_root_greater_than_three_l3186_318647

theorem no_root_greater_than_three : 
  ¬∃ x : ℝ, (x > 3 ∧ 
    ((3 * x^2 - 2 = 25) ∨ 
     ((2*x-1)^2 = (x-1)^2) ∨ 
     (x^2 - 7 = x - 1 ∧ x ≥ 1))) := by
  sorry

end NUMINAMATH_CALUDE_no_root_greater_than_three_l3186_318647


namespace NUMINAMATH_CALUDE_inequality_proof_l3186_318640

theorem inequality_proof (a b : ℝ) : a^2 + b^2 + 7/4 ≥ a*b + 2*a + b/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3186_318640


namespace NUMINAMATH_CALUDE_g_negative_101_l3186_318617

/-- A function g satisfying the given functional equation -/
def g_function (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g (x * y) + x = x * g y + g x

theorem g_negative_101 (g : ℝ → ℝ) (h1 : g_function g) (h2 : g 1 = 7) : 
  g (-101) = -95 :=
sorry

end NUMINAMATH_CALUDE_g_negative_101_l3186_318617


namespace NUMINAMATH_CALUDE_quadratic_equation_two_distinct_roots_l3186_318683

theorem quadratic_equation_two_distinct_roots (c d : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
  (x₁ + c) * (x₁ + d) - (2 * x₁ + c + d) = 0 ∧
  (x₂ + c) * (x₂ + d) - (2 * x₂ + c + d) = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_two_distinct_roots_l3186_318683


namespace NUMINAMATH_CALUDE_hamburger_cost_is_correct_l3186_318672

/-- The cost of a hamburger given the conditions of Robert and Teddy's snack purchase --/
def hamburger_cost : ℚ :=
  let pizza_box_cost : ℚ := 10
  let soft_drink_cost : ℚ := 2
  let robert_pizza_boxes : ℕ := 5
  let robert_soft_drinks : ℕ := 10
  let teddy_hamburgers : ℕ := 6
  let teddy_soft_drinks : ℕ := 10
  let total_spent : ℚ := 106

  let robert_spent : ℚ := pizza_box_cost * robert_pizza_boxes + soft_drink_cost * robert_soft_drinks
  let teddy_spent : ℚ := total_spent - robert_spent
  let teddy_hamburgers_cost : ℚ := teddy_spent - soft_drink_cost * teddy_soft_drinks

  teddy_hamburgers_cost / teddy_hamburgers

theorem hamburger_cost_is_correct :
  hamburger_cost = 267/100 := by
  sorry

end NUMINAMATH_CALUDE_hamburger_cost_is_correct_l3186_318672


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3186_318614

/-- 
Given a quadratic equation 2kx^2 + (8k+1)x + 8k = 0 with real coefficient k,
the equation has two distinct real roots if and only if k > -1/16 and k ≠ 0.
-/
theorem quadratic_two_distinct_roots (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 2 * k * x₁^2 + (8 * k + 1) * x₁ + 8 * k = 0 ∧
                          2 * k * x₂^2 + (8 * k + 1) * x₂ + 8 * k = 0) ↔
  (k > -1/16 ∧ k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3186_318614


namespace NUMINAMATH_CALUDE_remaining_water_l3186_318602

-- Define the initial amount of water
def initial_water : ℚ := 3

-- Define the first usage
def first_usage : ℚ := 5/4

-- Define the second usage
def second_usage : ℚ := 1/3

-- Theorem to prove
theorem remaining_water :
  initial_water - first_usage - second_usage = 17/12 := by
  sorry

end NUMINAMATH_CALUDE_remaining_water_l3186_318602


namespace NUMINAMATH_CALUDE_least_possible_z_l3186_318665

theorem least_possible_z (x y z : ℤ) : 
  Even x → Odd y → Odd z → y - x > 5 → (∀ w, Odd w → w - x ≥ 9 → z ≤ w) → z = 11 :=
by sorry

end NUMINAMATH_CALUDE_least_possible_z_l3186_318665


namespace NUMINAMATH_CALUDE_perfect_square_identity_l3186_318639

theorem perfect_square_identity (x y : ℝ) : x^2 + 2*x*y + y^2 = (x + y)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_identity_l3186_318639


namespace NUMINAMATH_CALUDE_sin_alpha_for_point_neg_one_three_l3186_318613

/-- Given an angle α whose terminal side passes through the point (-1, 3),
    prove that sin α = (3 * √10) / 10 -/
theorem sin_alpha_for_point_neg_one_three (α : Real) :
  (∃ (t : Real), t > 0 ∧ t * Real.cos α = -1 ∧ t * Real.sin α = 3) →
  Real.sin α = (3 * Real.sqrt 10) / 10 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_for_point_neg_one_three_l3186_318613


namespace NUMINAMATH_CALUDE_triangle_BC_length_l3186_318646

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the conditions of the problem
def triangle_conditions (t : Triangle) : Prop :=
  t.A.1 = 1 ∧ t.A.2 = 1 ∧
  t.B.2 = parabola t.B.1 ∧
  t.C.2 = parabola t.C.1 ∧
  t.B.2 = t.C.2 ∧
  (1/2 * (t.C.1 - t.B.1) * (t.B.2 - t.A.2) = 32)

-- Theorem statement
theorem triangle_BC_length (t : Triangle) :
  triangle_conditions t → (t.C.1 - t.B.1 = 8) :=
by sorry

end NUMINAMATH_CALUDE_triangle_BC_length_l3186_318646


namespace NUMINAMATH_CALUDE_total_plums_picked_l3186_318635

theorem total_plums_picked (melanie_plums dan_plums sally_plums : ℕ) 
  (h1 : melanie_plums = 4)
  (h2 : dan_plums = 9)
  (h3 : sally_plums = 3) :
  melanie_plums + dan_plums + sally_plums = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_plums_picked_l3186_318635


namespace NUMINAMATH_CALUDE_expression_values_l3186_318668

theorem expression_values (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let expr := a / |a| + b / |b| + c / |c| + d / |d| + (a * b * c * d) / |a * b * c * d|
  expr = -5 ∨ expr = -1 ∨ expr = 1 ∨ expr = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_values_l3186_318668


namespace NUMINAMATH_CALUDE_remaining_squares_l3186_318604

/-- A chocolate bar with rectangular shape -/
structure ChocolateBar where
  length : ℕ
  width : ℕ
  total_squares : ℕ
  h_width : width = 6
  h_length : length ≥ 9
  h_total : total_squares = length * width

/-- The number of squares removed by Irena and Jack -/
def squares_removed : ℕ := 12 + 9

/-- The theorem stating the number of remaining squares -/
theorem remaining_squares (bar : ChocolateBar) : 
  bar.total_squares - squares_removed = 45 := by
  sorry

#check remaining_squares

end NUMINAMATH_CALUDE_remaining_squares_l3186_318604


namespace NUMINAMATH_CALUDE_f_monotonicity_and_range_l3186_318685

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + x^2 - x

theorem f_monotonicity_and_range (a : ℝ) :
  (a ≥ 1/8 → ∀ x > 0, StrictMono (f a)) ∧
  (∀ x ≥ 1, f a x ≥ 0 ↔ a ≥ -1) :=
sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_range_l3186_318685


namespace NUMINAMATH_CALUDE_total_rectangles_area_2_l3186_318697

-- Define the structure of the figure
structure Figure where
  patterns : List String
  small_square_side : ℕ

-- Define a rectangle in the figure
structure Rectangle where
  width : ℕ
  height : ℕ

-- Function to calculate the area of a rectangle
def rectangle_area (r : Rectangle) : ℕ :=
  r.width * r.height

-- Function to count rectangles with area 2 in a specific pattern
def count_rectangles_area_2 (pattern : String) : ℕ :=
  match pattern with
  | "2" => 10
  | "0" => 12
  | "1" => 4
  | "4" => 8
  | _ => 0

-- Theorem stating the total number of rectangles with area 2
theorem total_rectangles_area_2 (fig : Figure) 
  (h1 : fig.small_square_side = 1) 
  (h2 : fig.patterns = ["2", "0", "1", "4"]) : 
  (fig.patterns.map count_rectangles_area_2).sum = 34 := by
  sorry

end NUMINAMATH_CALUDE_total_rectangles_area_2_l3186_318697


namespace NUMINAMATH_CALUDE_contrapositive_square_inequality_l3186_318666

theorem contrapositive_square_inequality (x y : ℝ) :
  (¬(x > y) → ¬(x^2 > y^2)) ↔ (x ≤ y → x^2 ≤ y^2) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_square_inequality_l3186_318666


namespace NUMINAMATH_CALUDE_calculation_proof_l3186_318632

theorem calculation_proof : 
  ((0.8 + (1 / 5)) * 24 + 6.6) / (9 / 14) - 7.6 = 40 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3186_318632


namespace NUMINAMATH_CALUDE_sum_of_digits_greatest_prime_divisor_16385_l3186_318628

def greatest_prime_divisor (n : Nat) : Nat :=
  sorry

def sum_of_digits (n : Nat) : Nat :=
  sorry

theorem sum_of_digits_greatest_prime_divisor_16385 :
  sum_of_digits (greatest_prime_divisor 16385) = 19 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_greatest_prime_divisor_16385_l3186_318628


namespace NUMINAMATH_CALUDE_pythagorean_pattern_solution_for_eleven_l3186_318690

theorem pythagorean_pattern (n : ℕ) : 
  (2*n + 1)^2 + (2*n^2 + 2*n)^2 = (2*n^2 + 2*n + 1)^2 := by sorry

theorem solution_for_eleven : 
  let n : ℕ := 5
  (2*n^2 + 2*n + 1) = 61 := by sorry

end NUMINAMATH_CALUDE_pythagorean_pattern_solution_for_eleven_l3186_318690


namespace NUMINAMATH_CALUDE_parabola_vertex_on_x_axis_l3186_318688

/-- If the vertex of the parabola y = x^2 + 2x + c is on the x-axis, then c = 1 -/
theorem parabola_vertex_on_x_axis (c : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x + c = 0 ∧ ∀ t : ℝ, t^2 + 2*t + c ≥ x^2 + 2*x + c) → c = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_on_x_axis_l3186_318688


namespace NUMINAMATH_CALUDE_function_inequality_l3186_318670

-- Define the functions f and g
def f (x b : ℝ) : ℝ := |x + b^2| - |-x + 1|
def g (x a b c : ℝ) : ℝ := |x + a^2 + c^2| + |x - 2*b^2|

-- State the theorem
theorem function_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a*b + b*c + a*c = 1) :
  (∀ x : ℝ, f x 1 ≥ 1 ↔ x ∈ Set.Ici (1/2)) ∧
  (∀ x : ℝ, f x b ≤ g x a b c) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3186_318670


namespace NUMINAMATH_CALUDE_even_sum_of_even_sum_of_squares_l3186_318610

theorem even_sum_of_even_sum_of_squares (n m : ℤ) (h : Even (n^2 + m^2)) : Even (n + m) := by
  sorry

end NUMINAMATH_CALUDE_even_sum_of_even_sum_of_squares_l3186_318610


namespace NUMINAMATH_CALUDE_area_of_semicircle_with_inscribed_rectangle_l3186_318651

/-- A semicircle with an inscribed 1 × 3 rectangle -/
structure InscribedRectangleSemicircle where
  /-- The radius of the semicircle -/
  radius : ℝ
  /-- The width of the inscribed rectangle -/
  rect_width : ℝ
  /-- The length of the inscribed rectangle -/
  rect_length : ℝ
  /-- The width of the rectangle is 1 -/
  width_is_one : rect_width = 1
  /-- The length of the rectangle is 3 -/
  length_is_three : rect_length = 3
  /-- The rectangle is inscribed in the semicircle -/
  inscribed : radius^2 = (rect_width / 2)^2 + (rect_length / 2)^2

/-- The area of the semicircle with an inscribed 1 × 3 rectangle is 13π/8 -/
theorem area_of_semicircle_with_inscribed_rectangle 
  (s : InscribedRectangleSemicircle) : 
  π * s.radius^2 / 2 = 13 * π / 8 := by
  sorry

#check area_of_semicircle_with_inscribed_rectangle

end NUMINAMATH_CALUDE_area_of_semicircle_with_inscribed_rectangle_l3186_318651


namespace NUMINAMATH_CALUDE_mrs_thompson_chicken_cost_l3186_318607

/-- Given the total cost, number of chickens, and cost of potatoes, 
    calculate the cost of each chicken. -/
def chicken_cost (total : ℚ) (num_chickens : ℕ) (potato_cost : ℚ) : ℚ :=
  (total - potato_cost) / num_chickens

/-- Prove that each chicken costs $3 given the problem conditions -/
theorem mrs_thompson_chicken_cost :
  chicken_cost 15 3 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_mrs_thompson_chicken_cost_l3186_318607


namespace NUMINAMATH_CALUDE_five_digit_multiplication_reversal_l3186_318629

theorem five_digit_multiplication_reversal :
  ∃! (a b c d e : ℕ),
    0 ≤ a ∧ a ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    0 ≤ c ∧ c ≤ 9 ∧
    0 ≤ d ∧ d ≤ 9 ∧
    0 ≤ e ∧ e ≤ 9 ∧
    a ≠ 0 ∧
    (10000 * a + 1000 * b + 100 * c + 10 * d + e) * 9 =
    10000 * e + 1000 * d + 100 * c + 10 * b + a ∧
    a = 1 ∧ b = 0 ∧ c = 9 ∧ d = 8 ∧ e = 9 :=
by sorry

end NUMINAMATH_CALUDE_five_digit_multiplication_reversal_l3186_318629


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l3186_318648

/-- Proves that a train of given length and speed takes the calculated time to cross a bridge of given length -/
theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (bridge_length : ℝ) 
  (h1 : train_length = 120) 
  (h2 : train_speed_kmh = 45) 
  (h3 : bridge_length = 255) : 
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l3186_318648


namespace NUMINAMATH_CALUDE_total_sweaters_61_l3186_318644

def sweaters_fortnight (day1 day2 day3_4 day5 day6 day7 day8_10 day11 day12_13 day14 : ℕ) : Prop :=
  day1 = 8 ∧
  day2 = day1 + 2 ∧
  day3_4 = day2 - 4 ∧
  day5 = day3_4 ∧
  day6 = day1 / 2 ∧
  day7 = 0 ∧
  day8_10 = (day1 + day2 + day3_4 * 2 + day5 + day6) * 3 * 3 / (4 * 6) ∧
  day11 = day8_10 / 3 / 3 ∧
  day12_13 = day8_10 / 2 / 3 ∧
  day14 = 1

theorem total_sweaters_61 :
  ∀ day1 day2 day3_4 day5 day6 day7 day8_10 day11 day12_13 day14 : ℕ,
  sweaters_fortnight day1 day2 day3_4 day5 day6 day7 day8_10 day11 day12_13 day14 →
  day1 + day2 + day3_4 * 2 + day5 + day6 + day7 + day8_10 + day11 + day12_13 * 2 + day14 = 61 :=
by
  sorry

end NUMINAMATH_CALUDE_total_sweaters_61_l3186_318644


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l3186_318677

def complex_equation (z : ℂ) : Prop :=
  (z - 2) * (z^2 + z + 2) * (z^2 + 5*z + 8) = 0

def is_root (z : ℂ) : Prop :=
  complex_equation z

def ellipse_through_roots (e : ℝ) : Prop :=
  ∃ (a b : ℝ) (h : ℂ), 
    a > 0 ∧ b > 0 ∧
    ∀ (z : ℂ), is_root z → 
      (z.re - h.re)^2 / a^2 + (z.im - h.im)^2 / b^2 = 1 ∧
    e = Real.sqrt (1 - b^2 / a^2)

theorem ellipse_eccentricity : 
  ellipse_through_roots (Real.sqrt (1/5)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l3186_318677


namespace NUMINAMATH_CALUDE_log_inequality_may_not_hold_l3186_318655

theorem log_inequality_may_not_hold (a b : ℝ) (h : 1/a < 1/b ∧ 1/b < 0) :
  ¬ (∀ a b : ℝ, 1/a < 1/b ∧ 1/b < 0 → Real.log (-a) / Real.log (-b) ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_may_not_hold_l3186_318655


namespace NUMINAMATH_CALUDE_johns_journey_speed_l3186_318637

theorem johns_journey_speed (total_distance : ℝ) (first_duration : ℝ) (second_duration : ℝ) (second_speed : ℝ) (S : ℝ) :
  total_distance = 240 →
  first_duration = 2 →
  second_duration = 3 →
  second_speed = 50 →
  total_distance = first_duration * S + second_duration * second_speed →
  S = 45 := by
  sorry

end NUMINAMATH_CALUDE_johns_journey_speed_l3186_318637


namespace NUMINAMATH_CALUDE_circle_equation_l3186_318609

-- Define the circle C
def circle_C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 3)^2 + p.2^2 = 2}

-- Define the line x-y-1=0
def line : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - p.2 - 1 = 0}

-- Define points A and B
def point_A : ℝ × ℝ := (4, 1)
def point_B : ℝ × ℝ := (2, 1)

-- Theorem statement
theorem circle_equation :
  (point_A ∈ circle_C) ∧
  (point_B ∈ line) ∧
  (∃ (t : ℝ), ∀ (p : ℝ × ℝ), p ∈ circle_C → (p.1 - point_B.1) * 1 + (p.2 - point_B.2) * (-1) = t * ((p.1 - point_B.1)^2 + (p.2 - point_B.2)^2)) →
  ∀ (x y : ℝ), (x, y) ∈ circle_C ↔ (x - 3)^2 + y^2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l3186_318609


namespace NUMINAMATH_CALUDE_rhombus_properties_l3186_318692

/-- Properties of a rhombus with given area and one diagonal --/
theorem rhombus_properties (area : ℝ) (d1 : ℝ) (d2 : ℝ) (θ : ℝ) 
  (h1 : area = 432)
  (h2 : d1 = 36)
  (h3 : area = (d1 * d2) / 2)
  (h4 : θ = 2 * Real.arccos (2 / 3)) :
  d2 = 24 ∧ θ = 2 * Real.arccos (2 / 3) := by
  sorry


end NUMINAMATH_CALUDE_rhombus_properties_l3186_318692


namespace NUMINAMATH_CALUDE_odd_difference_of_even_and_odd_l3186_318652

theorem odd_difference_of_even_and_odd (a b : ℤ) 
  (ha : Even a) (hb : Odd b) : Odd (a - b) := by
  sorry

end NUMINAMATH_CALUDE_odd_difference_of_even_and_odd_l3186_318652


namespace NUMINAMATH_CALUDE_polygon_diagonals_l3186_318626

/-- A polygon with interior angle sum of 1800 degrees has 9 diagonals from one vertex -/
theorem polygon_diagonals (n : ℕ) : 
  (n - 2) * 180 = 1800 → n - 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_l3186_318626


namespace NUMINAMATH_CALUDE_division_of_hundred_by_quarter_l3186_318693

theorem division_of_hundred_by_quarter : (100 : ℝ) / 0.25 = 400 := by
  sorry

end NUMINAMATH_CALUDE_division_of_hundred_by_quarter_l3186_318693


namespace NUMINAMATH_CALUDE_volleyball_points_product_l3186_318606

def first_10_games : List ℕ := [5, 6, 4, 7, 5, 6, 2, 3, 4, 9]

def total_first_10 : ℕ := first_10_games.sum

theorem volleyball_points_product :
  ∀ (points_11 points_12 : ℕ),
    points_11 < 15 →
    points_12 < 15 →
    (total_first_10 + points_11) % 11 = 0 →
    (total_first_10 + points_11 + points_12) % 12 = 0 →
    points_11 * points_12 = 20 := by
sorry

end NUMINAMATH_CALUDE_volleyball_points_product_l3186_318606


namespace NUMINAMATH_CALUDE_bacon_only_count_l3186_318618

theorem bacon_only_count (total_bacon : ℕ) (both : ℕ) (h1 : total_bacon = 569) (h2 : both = 218) :
  total_bacon - both = 351 := by
  sorry

end NUMINAMATH_CALUDE_bacon_only_count_l3186_318618


namespace NUMINAMATH_CALUDE_base_height_calculation_l3186_318664

/-- Given a sculpture height and total height, calculate the base height -/
theorem base_height_calculation (sculpture_height_feet : ℚ) (sculpture_height_inches : ℚ) (total_height : ℚ) : 
  sculpture_height_feet = 2 ∧ 
  sculpture_height_inches = 10 ∧ 
  total_height = 3.6666666666666665 →
  total_height - (sculpture_height_feet + sculpture_height_inches / 12) = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_base_height_calculation_l3186_318664


namespace NUMINAMATH_CALUDE_minibus_children_count_l3186_318686

theorem minibus_children_count (total_seats : ℕ) (full_seats : ℕ) (children_per_full_seat : ℕ) (children_per_remaining_seat : ℕ) :
  total_seats = 7 →
  full_seats = 5 →
  children_per_full_seat = 3 →
  children_per_remaining_seat = 2 →
  full_seats * children_per_full_seat + (total_seats - full_seats) * children_per_remaining_seat = 19 :=
by sorry

end NUMINAMATH_CALUDE_minibus_children_count_l3186_318686


namespace NUMINAMATH_CALUDE_unbiased_scale_impossible_biased_scale_possible_l3186_318641

/-- Represents the result of a weighing -/
inductive WeighResult
  | LeftHeavier
  | RightHeavier
  | Equal

/-- Represents a weighing strategy -/
def WeighStrategy := List WeighResult → WeighResult

/-- Represents a set of weights -/
def Weights := List Nat

/-- Represents a balance scale -/
structure Balance where
  bias : Int  -- Positive means left pan is lighter

/-- Function to perform a weighing -/
def weigh (b : Balance) (left right : Weights) : WeighResult :=
  sorry

/-- Function to determine if a set of weights can be uniquely identified -/
def canIdentifyWeights (w : Weights) (b : Balance) (n : Nat) : Prop :=
  sorry

/-- The main theorem for the unbiased scale -/
theorem unbiased_scale_impossible 
  (w : Weights) 
  (h1 : w = [1000, 1002, 1004, 1005]) 
  (b : Balance) 
  (h2 : b.bias = 0) : 
  ¬ (canIdentifyWeights w b 4) :=
sorry

/-- The main theorem for the biased scale -/
theorem biased_scale_possible 
  (w : Weights) 
  (h1 : w = [1000, 1002, 1004, 1005]) 
  (b : Balance) 
  (h2 : b.bias = 1) : 
  canIdentifyWeights w b 4 :=
sorry

end NUMINAMATH_CALUDE_unbiased_scale_impossible_biased_scale_possible_l3186_318641


namespace NUMINAMATH_CALUDE_square_area_above_line_l3186_318678

/-- The fraction of a square's area above a line -/
def fractionAboveLine (p1 p2 v1 v2 v3 v4 : ℝ × ℝ) : ℚ :=
  sorry

/-- The main theorem -/
theorem square_area_above_line :
  let p1 : ℝ × ℝ := (2, 3)
  let p2 : ℝ × ℝ := (5, 1)
  let v1 : ℝ × ℝ := (2, 1)
  let v2 : ℝ × ℝ := (5, 1)
  let v3 : ℝ × ℝ := (5, 4)
  let v4 : ℝ × ℝ := (2, 4)
  fractionAboveLine p1 p2 v1 v2 v3 v4 = 2/3 :=
sorry

end NUMINAMATH_CALUDE_square_area_above_line_l3186_318678


namespace NUMINAMATH_CALUDE_obtuse_triangle_count_l3186_318694

/-- A triangle with sides 5, 12, and k is obtuse -/
def isObtuse (k : ℕ) : Prop :=
  (k > 5 ∧ k > 12 ∧ k^2 > 5^2 + 12^2) ∨
  (12 > 5 ∧ 12 > k ∧ 12^2 > 5^2 + k^2) ∨
  (5 > 12 ∧ 5 > k ∧ 5^2 > 12^2 + k^2)

/-- The triangle with sides 5, 12, and k is valid (satisfies triangle inequality) -/
def isValidTriangle (k : ℕ) : Prop :=
  k + 5 > 12 ∧ k + 12 > 5 ∧ 5 + 12 > k

theorem obtuse_triangle_count :
  ∃! (s : Finset ℕ), (∀ k ∈ s, k > 0 ∧ isValidTriangle k ∧ isObtuse k) ∧ s.card = 6 :=
sorry

end NUMINAMATH_CALUDE_obtuse_triangle_count_l3186_318694


namespace NUMINAMATH_CALUDE_debt_payment_average_l3186_318699

theorem debt_payment_average (n : ℕ) (first_payment second_payment : ℚ) : 
  n = 40 →
  first_payment = 410 →
  second_payment = first_payment + 65 →
  (20 * first_payment + 20 * second_payment) / n = 442.50 := by
  sorry

end NUMINAMATH_CALUDE_debt_payment_average_l3186_318699


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiplication_result_l3186_318612

/-- Represents a digit from 1 to 9 -/
def Digit := Fin 9

/-- Represents a four-digit number -/
structure FourDigitNumber where
  thousands : Digit
  hundreds : Digit
  tens : Digit
  ones : Digit

/-- Converts a FourDigitNumber to a natural number -/
def FourDigitNumber.toNat (n : FourDigitNumber) : Nat :=
  1000 * (n.thousands.val + 1) + 100 * (n.hundreds.val + 1) + 10 * (n.tens.val + 1) + (n.ones.val + 1)

/-- The theorem statement -/
theorem smallest_four_digit_multiplication_result :
  ∀ (a b c d e f g h : Digit),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
    f ≠ g ∧ f ≠ h ∧
    g ≠ h →
    ((a.val + 1) * 10 + (b.val + 1)) * ((c.val + 1) * 10 + (d.val + 1)) =
      (e.val + 1) * 1000 + (f.val + 1) * 100 + (g.val + 1) * 10 + (h.val + 1) →
    ∀ (n : FourDigitNumber),
      n.toNat ≥ 4396 :=
by sorry

#check smallest_four_digit_multiplication_result

end NUMINAMATH_CALUDE_smallest_four_digit_multiplication_result_l3186_318612


namespace NUMINAMATH_CALUDE_rug_area_theorem_l3186_318608

theorem rug_area_theorem (total_floor_area : ℝ) (two_layer_area : ℝ) (three_layer_area : ℝ) 
  (h1 : total_floor_area = 140)
  (h2 : two_layer_area = 22)
  (h3 : three_layer_area = 19) :
  total_floor_area + two_layer_area + 2 * three_layer_area = 200 :=
by sorry

end NUMINAMATH_CALUDE_rug_area_theorem_l3186_318608


namespace NUMINAMATH_CALUDE_sqrt_8_to_6th_power_l3186_318619

theorem sqrt_8_to_6th_power : (Real.sqrt 8) ^ 6 = 512 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_8_to_6th_power_l3186_318619


namespace NUMINAMATH_CALUDE_constant_value_l3186_318603

def f (x : ℝ) : ℝ := 3 * x - 5

theorem constant_value : ∃ c : ℝ, 2 * f 3 - c = f (3 - 2) ∧ c = 10 := by
  sorry

end NUMINAMATH_CALUDE_constant_value_l3186_318603


namespace NUMINAMATH_CALUDE_subtract_like_terms_l3186_318673

theorem subtract_like_terms (a : ℝ) : 7 * a - 3 * a = 4 * a := by
  sorry

end NUMINAMATH_CALUDE_subtract_like_terms_l3186_318673


namespace NUMINAMATH_CALUDE_book_reading_ratio_l3186_318684

theorem book_reading_ratio (total_pages : ℕ) (total_days : ℕ) (speed1 speed2 : ℕ) 
  (h1 : total_pages = 500)
  (h2 : total_days = 75)
  (h3 : speed1 = 10)
  (h4 : speed2 = 5)
  (h5 : ∃ x : ℕ, speed1 * x + speed2 * (total_days - x) = total_pages) :
  ∃ x : ℕ, (speed1 * x : ℚ) / total_pages = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_book_reading_ratio_l3186_318684


namespace NUMINAMATH_CALUDE_share_A_is_240_l3186_318642

/-- Calculates the share of profit for partner A in a business partnership --/
def calculate_share_A (initial_A initial_B : ℕ) (withdraw_A advance_B : ℕ) (months : ℕ) (total_profit : ℕ) : ℕ :=
  let investment_months_A := initial_A * months + (initial_A - withdraw_A) * (12 - months)
  let investment_months_B := initial_B * months + (initial_B + advance_B) * (12 - months)
  let total_investment_months := investment_months_A + investment_months_B
  (investment_months_A * total_profit) / total_investment_months

theorem share_A_is_240 :
  calculate_share_A 3000 4000 1000 1000 8 630 = 240 := by
  sorry

#eval calculate_share_A 3000 4000 1000 1000 8 630

end NUMINAMATH_CALUDE_share_A_is_240_l3186_318642


namespace NUMINAMATH_CALUDE_slope_range_ordinate_range_l3186_318663

-- Define the point A
def A : ℝ × ℝ := (0, 3)

-- Define the line l
def line_l (x : ℝ) : ℝ := 2 * x - 4

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the conditions
def conditions (C : Circle) (k : ℝ) : Prop :=
  C.radius = 1 ∧
  C.center.2 = line_l C.center.1 ∧
  C.center.2 = C.center.1 - 1 ∧
  ∃ (x y : ℝ), (x - C.center.1)^2 + (y - C.center.2)^2 = 1 ∧ y = k * x + 3

-- Define the theorems to be proved
theorem slope_range (C : Circle) :
  (∃ k, conditions C k) → ∃ k, -3/4 ≤ k ∧ k ≤ 0 :=
sorry

theorem ordinate_range (C : Circle) :
  (∃ M : ℝ × ℝ, (M.1 - C.center.1)^2 + (M.2 - C.center.2)^2 = 1 ∧
   (M.1 - A.1)^2 + (M.2 - A.2)^2 = 4 * ((M.1 - 0)^2 + (M.2 - 0)^2)) →
  -4 ≤ C.center.2 ∧ C.center.2 ≤ 4/5 :=
sorry

end NUMINAMATH_CALUDE_slope_range_ordinate_range_l3186_318663


namespace NUMINAMATH_CALUDE_largest_negative_integer_and_abs_property_l3186_318600

theorem largest_negative_integer_and_abs_property :
  (∀ n : ℤ, n < 0 → n ≤ -1) ∧
  (∀ x : ℝ, |x| = x → x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_largest_negative_integer_and_abs_property_l3186_318600


namespace NUMINAMATH_CALUDE_shortest_tree_height_l3186_318649

/-- Proves that the height of the shortest tree is 50 feet given the conditions of the problem. -/
theorem shortest_tree_height (tallest middle shortest : ℝ) : 
  tallest = 150 ∧ 
  middle = 2/3 * tallest ∧ 
  shortest = 1/2 * middle →
  shortest = 50 := by
  sorry

end NUMINAMATH_CALUDE_shortest_tree_height_l3186_318649


namespace NUMINAMATH_CALUDE_train_passengers_l3186_318675

theorem train_passengers (initial_passengers : ℕ) : 
  initial_passengers = 288 →
  let after_first := initial_passengers * 2 / 3 + 280
  let after_second := after_first / 2 + 12
  after_second = 248 := by
sorry

end NUMINAMATH_CALUDE_train_passengers_l3186_318675


namespace NUMINAMATH_CALUDE_prop_a_neither_sufficient_nor_necessary_l3186_318634

-- Define propositions A and B
def PropA (a b : ℝ) : Prop := a + b ≠ 4
def PropB (a b : ℝ) : Prop := a ≠ 1 ∧ b ≠ 3

-- Theorem stating that Prop A is neither sufficient nor necessary for Prop B
theorem prop_a_neither_sufficient_nor_necessary :
  (∃ a b : ℝ, PropA a b ∧ ¬PropB a b) ∧
  (∃ a b : ℝ, PropB a b ∧ ¬PropA a b) :=
sorry

end NUMINAMATH_CALUDE_prop_a_neither_sufficient_nor_necessary_l3186_318634


namespace NUMINAMATH_CALUDE_probability_at_least_one_man_l3186_318653

theorem probability_at_least_one_man (total : ℕ) (men : ℕ) (women : ℕ) (selected : ℕ) :
  total = men + women →
  men = 10 →
  women = 5 →
  selected = 5 →
  (1 : ℚ) - (Nat.choose women selected : ℚ) / (Nat.choose total selected : ℚ) = 3002 / 3003 :=
by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_man_l3186_318653


namespace NUMINAMATH_CALUDE_winner_for_10_winner_for_12_winner_for_15_winner_for_30_l3186_318679

/-- Represents the outcome of the game -/
inductive GameOutcome
  | FirstPlayerWins
  | SecondPlayerWins

/-- Represents the game state -/
structure GameState where
  n : Nat
  circled : List Nat

/-- Checks if two numbers are relatively prime -/
def isRelativelyPrime (a b : Nat) : Bool :=
  Nat.gcd a b = 1

/-- Checks if a number can be circled given the current game state -/
def canCircle (state : GameState) (num : Nat) : Bool :=
  num ≤ state.n &&
  num ∉ state.circled &&
  state.circled.all (isRelativelyPrime num)

/-- Determines the winner of the game given the initial value of N -/
def determineWinner (n : Nat) : GameOutcome :=
  sorry

/-- Theorem stating the game outcome for N = 10 -/
theorem winner_for_10 : determineWinner 10 = GameOutcome.FirstPlayerWins := by sorry

/-- Theorem stating the game outcome for N = 12 -/
theorem winner_for_12 : determineWinner 12 = GameOutcome.FirstPlayerWins := by sorry

/-- Theorem stating the game outcome for N = 15 -/
theorem winner_for_15 : determineWinner 15 = GameOutcome.SecondPlayerWins := by sorry

/-- Theorem stating the game outcome for N = 30 -/
theorem winner_for_30 : determineWinner 30 = GameOutcome.FirstPlayerWins := by sorry

end NUMINAMATH_CALUDE_winner_for_10_winner_for_12_winner_for_15_winner_for_30_l3186_318679


namespace NUMINAMATH_CALUDE_tribe_organization_ways_l3186_318627

/-- The number of members in the tribe -/
def tribeSize : ℕ := 13

/-- The number of supporting chiefs -/
def numSupportingChiefs : ℕ := 3

/-- The number of inferiors for each supporting chief -/
def numInferiors : ℕ := 2

/-- Calculate the number of ways to organize the tribe's leadership -/
def organizationWays : ℕ := 
  tribeSize * (tribeSize - 1) * (tribeSize - 2) * (tribeSize - 3) * 
  Nat.choose (tribeSize - 4) 2 * 
  Nat.choose (tribeSize - 6) 2 * 
  Nat.choose (tribeSize - 8) 2

/-- Theorem stating that the number of ways to organize the leadership is 12355200 -/
theorem tribe_organization_ways : organizationWays = 12355200 := by
  sorry

end NUMINAMATH_CALUDE_tribe_organization_ways_l3186_318627


namespace NUMINAMATH_CALUDE_complex_number_magnitude_squared_l3186_318661

theorem complex_number_magnitude_squared (z : ℂ) (h : z^2 + Complex.abs z^2 = 4 - 7*I) :
  Complex.abs z^2 = 65/8 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_magnitude_squared_l3186_318661
