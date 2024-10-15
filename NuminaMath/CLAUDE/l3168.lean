import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_integers_l3168_316852

theorem sum_of_integers (a b : ℕ+) (h1 : a - b = 14) (h2 : a * b = 120) : a + b = 26 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l3168_316852


namespace NUMINAMATH_CALUDE_exists_non_grid_aligned_right_triangle_l3168_316819

/-- A triangle represented by its three vertices -/
structure Triangle where
  a : ℤ × ℤ
  b : ℤ × ℤ
  c : ℤ × ℤ

/-- Check if a triangle is right-angled -/
def is_right_angled (t : Triangle) : Prop :=
  let ab := (t.b.1 - t.a.1, t.b.2 - t.a.2)
  let ac := (t.c.1 - t.a.1, t.c.2 - t.a.2)
  ab.1 * ac.1 + ab.2 * ac.2 = 0

/-- Check if a line segment is aligned with the grid -/
def is_grid_aligned (p1 p2 : ℤ × ℤ) : Prop :=
  p1.1 = p2.1 ∨ p1.2 = p2.2 ∨ (p2.2 - p1.2) * (p2.1 - p1.1) = 0

/-- The main theorem -/
theorem exists_non_grid_aligned_right_triangle :
  ∃ (t : Triangle),
    is_right_angled t ∧
    ¬is_grid_aligned t.a t.b ∧
    ¬is_grid_aligned t.b t.c ∧
    ¬is_grid_aligned t.c t.a :=
  sorry

end NUMINAMATH_CALUDE_exists_non_grid_aligned_right_triangle_l3168_316819


namespace NUMINAMATH_CALUDE_billys_age_l3168_316842

/-- Given the ages of Billy, Joe, and Mary, prove that Billy is 45 years old. -/
theorem billys_age (B J M : ℕ) 
  (h1 : B = 3 * J)           -- Billy's age is three times Joe's age
  (h2 : B + J = 60)          -- The sum of Billy's and Joe's ages is 60
  (h3 : B + M = 90)          -- The sum of Billy's and Mary's ages is 90
  : B = 45 := by
  sorry


end NUMINAMATH_CALUDE_billys_age_l3168_316842


namespace NUMINAMATH_CALUDE_investment_ratio_is_three_l3168_316847

/-- Represents the investment scenario of three partners A, B, and C --/
structure Investment where
  x : ℝ  -- A's initial investment
  m : ℝ  -- Ratio of C's investment to A's investment
  total_gain : ℝ  -- Total annual gain
  a_share : ℝ  -- A's share of the gain

/-- The ratio of C's investment to A's investment in the given scenario --/
def investment_ratio (inv : Investment) : ℝ :=
  let a_investment := inv.x * 12  -- A's investment for 12 months
  let b_investment := 2 * inv.x * 6  -- B's investment for 6 months
  let c_investment := inv.m * inv.x * 4  -- C's investment for 4 months
  let total_investment := a_investment + b_investment + c_investment
  inv.m

/-- Theorem stating that the investment ratio is 3 given the conditions --/
theorem investment_ratio_is_three (inv : Investment)
  (h1 : inv.total_gain = 15000)
  (h2 : inv.a_share = 5000)
  (h3 : inv.x > 0)
  : investment_ratio inv = 3 := by
  sorry

#check investment_ratio_is_three

end NUMINAMATH_CALUDE_investment_ratio_is_three_l3168_316847


namespace NUMINAMATH_CALUDE_total_students_correct_l3168_316818

/-- Represents the total number of high school students -/
def total_students : ℕ := 1800

/-- Represents the sample size -/
def sample_size : ℕ := 45

/-- Represents the number of second-year students -/
def second_year_students : ℕ := 600

/-- Represents the number of second-year students selected in the sample -/
def selected_second_year : ℕ := 15

/-- Theorem stating that the total number of students is correct given the sampling information -/
theorem total_students_correct :
  (total_students : ℚ) / sample_size = (second_year_students : ℚ) / selected_second_year :=
sorry

end NUMINAMATH_CALUDE_total_students_correct_l3168_316818


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l3168_316888

/-- The complex number z defined as (i+2)/i is located in the fourth quadrant of the complex plane. -/
theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (Complex.I + 2) / Complex.I
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l3168_316888


namespace NUMINAMATH_CALUDE_number_of_factors_60_l3168_316803

/-- The number of positive factors of 60 is 12 -/
theorem number_of_factors_60 : Finset.card (Nat.divisors 60) = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_of_factors_60_l3168_316803


namespace NUMINAMATH_CALUDE_inequality_problem_l3168_316864

theorem inequality_problem (x : ℝ) : 
  (x - 1) * |4 - x| < 12 ∧ x - 2 > 0 → 4 < x ∧ x < 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_problem_l3168_316864


namespace NUMINAMATH_CALUDE_no_periodic_difference_with_3_and_pi_periods_l3168_316886

-- Define a periodic function
def isPeriodic (f : ℝ → ℝ) :=
  (∃ x y, f x ≠ f y) ∧ (∃ p > 0, ∀ x, f (x + p) = f x)

-- Define the period of a function
def isPeriodOf (p : ℝ) (f : ℝ → ℝ) :=
  p > 0 ∧ ∀ x, f (x + p) = f x

-- Theorem statement
theorem no_periodic_difference_with_3_and_pi_periods :
  ¬ ∃ (g h : ℝ → ℝ),
    isPeriodic g ∧ isPeriodic h ∧
    isPeriodOf 3 g ∧ isPeriodOf π h ∧
    isPeriodic (g - h) :=
sorry

end NUMINAMATH_CALUDE_no_periodic_difference_with_3_and_pi_periods_l3168_316886


namespace NUMINAMATH_CALUDE_unpartnered_students_correct_l3168_316891

/-- Calculates the number of students unable to partner in square dancing --/
def unpartnered_students (class1_males class1_females class2_males class2_females class3_males class3_females : ℕ) : ℕ :=
  let total_males := class1_males + class2_males + class3_males
  let total_females := class1_females + class2_females + class3_females
  Int.natAbs (total_males - total_females)

/-- Theorem stating that the number of unpartnered students is correct --/
theorem unpartnered_students_correct 
  (class1_males class1_females class2_males class2_females class3_males class3_females : ℕ) :
  unpartnered_students class1_males class1_females class2_males class2_females class3_males class3_females =
  Int.natAbs ((class1_males + class2_males + class3_males) - (class1_females + class2_females + class3_females)) :=
by sorry

#eval unpartnered_students 17 13 14 18 15 17  -- Should evaluate to 2

end NUMINAMATH_CALUDE_unpartnered_students_correct_l3168_316891


namespace NUMINAMATH_CALUDE_bottle_cap_distribution_l3168_316835

/-- Given 18 bottle caps shared among 6 friends, prove that each friend receives 3 bottle caps. -/
theorem bottle_cap_distribution (total_caps : ℕ) (num_friends : ℕ) (caps_per_friend : ℕ) : 
  total_caps = 18 → num_friends = 6 → caps_per_friend = total_caps / num_friends → caps_per_friend = 3 := by
  sorry

end NUMINAMATH_CALUDE_bottle_cap_distribution_l3168_316835


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3168_316807

theorem complex_fraction_simplification (z : ℂ) (h : z = 1 - I) :
  (z^2 - 2*z) / (z - 1) = -2*I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3168_316807


namespace NUMINAMATH_CALUDE_sphere_box_height_l3168_316856

/-- A rectangular box with a large sphere and eight smaller spheres -/
structure SphereBox where
  length : ℝ
  width : ℝ
  height : ℝ
  large_sphere_radius : ℝ
  small_sphere_radius : ℝ
  small_sphere_count : ℕ

/-- Conditions for the sphere arrangement in the box -/
def valid_sphere_arrangement (box : SphereBox) : Prop :=
  box.length = 6 ∧
  box.width = 6 ∧
  box.large_sphere_radius = 3 ∧
  box.small_sphere_radius = 1 ∧
  box.small_sphere_count = 8 ∧
  ∀ (small_sphere : Fin box.small_sphere_count),
    (∃ (side1 side2 side3 : ℝ), side1 + side2 + side3 = box.length + box.width + box.height) ∧
    (box.large_sphere_radius + box.small_sphere_radius = 
     (box.length / 2)^2 + (box.width / 2)^2 + (box.height / 2 - box.small_sphere_radius)^2)

/-- Theorem stating that the height of the box is 8 -/
theorem sphere_box_height (box : SphereBox) 
  (h : valid_sphere_arrangement box) : box.height = 8 := by
  sorry

end NUMINAMATH_CALUDE_sphere_box_height_l3168_316856


namespace NUMINAMATH_CALUDE_original_deck_size_l3168_316860

/-- Represents a deck of cards with blue and yellow cards -/
structure Deck where
  blue : ℕ
  yellow : ℕ

/-- The probability of drawing a blue card from the deck -/
def blueProbability (d : Deck) : ℚ :=
  d.blue / (d.blue + d.yellow)

/-- Adds yellow cards to the deck -/
def addYellow (d : Deck) (n : ℕ) : Deck :=
  { blue := d.blue, yellow := d.yellow + n }

theorem original_deck_size (d : Deck) :
  blueProbability d = 2/5 ∧ 
  blueProbability (addYellow d 6) = 5/14 →
  d.blue + d.yellow = 50 := by
  sorry

end NUMINAMATH_CALUDE_original_deck_size_l3168_316860


namespace NUMINAMATH_CALUDE_factor_value_theorem_l3168_316858

theorem factor_value_theorem (m n : ℚ) : 
  (∀ x : ℚ, (x - 3) * (x + 1) ∣ (3 * x^4 - m * x^2 + n * x - 5)) → 
  |3 * m - 2 * n| = 302 / 3 := by
  sorry

end NUMINAMATH_CALUDE_factor_value_theorem_l3168_316858


namespace NUMINAMATH_CALUDE_ellipse_properties_l3168_316896

/-- Properties of a specific ellipse -/
structure EllipseC where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0
  h_c : 2 = Real.sqrt (a^2 - b^2)
  h_slope : (b - 0) / (0 - a) = -Real.sqrt 3 / 3

/-- Theorem about the standard equation and a geometric property of the ellipse -/
theorem ellipse_properties (e : EllipseC) :
  (∃ (x y : ℝ), x^2 / 6 + y^2 / 2 = 1) ∧
  (∃ (F P M N : ℝ × ℝ),
    F.1 = 2 ∧ F.2 = 0 ∧
    P.1 = 3 ∧
    (M.1^2 / 6 + M.2^2 / 2 = 1) ∧
    (N.1^2 / 6 + N.2^2 / 2 = 1) ∧
    (M.2 - N.2) * (P.1 - F.1) = (P.2 - F.2) * (M.1 - N.1) ∧
    Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) / Real.sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2) ≤ Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3168_316896


namespace NUMINAMATH_CALUDE_morning_afternoon_email_difference_l3168_316867

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 3

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 5

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := 16

/-- The theorem states that Jack received 2 more emails in the morning than in the afternoon -/
theorem morning_afternoon_email_difference : morning_emails - afternoon_emails = 2 := by
  sorry

end NUMINAMATH_CALUDE_morning_afternoon_email_difference_l3168_316867


namespace NUMINAMATH_CALUDE_phone_bill_increase_l3168_316884

theorem phone_bill_increase (original_monthly_bill : ℝ) (new_yearly_bill : ℝ) : 
  original_monthly_bill = 50 → 
  new_yearly_bill = 660 → 
  (new_yearly_bill / (12 * original_monthly_bill) - 1) * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_phone_bill_increase_l3168_316884


namespace NUMINAMATH_CALUDE_bullet_train_length_l3168_316893

/-- The length of a bullet train passing a man running in the opposite direction -/
theorem bullet_train_length (train_speed : ℝ) (man_speed : ℝ) (passing_time : ℝ) :
  train_speed = 59 →
  man_speed = 7 →
  passing_time = 12 →
  (train_speed + man_speed) * (1000 / 3600) * passing_time = 220 :=
by sorry

end NUMINAMATH_CALUDE_bullet_train_length_l3168_316893


namespace NUMINAMATH_CALUDE_factorial16_trailingZeroes_base8_l3168_316828

/-- The number of trailing zeroes in the base 8 representation of 16! -/
def trailingZeroesBase8Factorial16 : ℕ := 5

/-- Theorem stating that the number of trailing zeroes in the base 8 representation of 16! is 5 -/
theorem factorial16_trailingZeroes_base8 :
  trailingZeroesBase8Factorial16 = 5 := by sorry

end NUMINAMATH_CALUDE_factorial16_trailingZeroes_base8_l3168_316828


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l3168_316880

/-- A geometric sequence is a sequence where each term after the first is found by multiplying 
    the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ q : ℚ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sixth_term 
  (a : ℕ → ℚ) 
  (h_geo : is_geometric_sequence a) 
  (h_sum1 : a 1 + a 3 = 5/2) 
  (h_sum2 : a 2 + a 4 = 5/4) : 
  a 6 = 1/16 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l3168_316880


namespace NUMINAMATH_CALUDE_senior_mean_score_l3168_316878

theorem senior_mean_score (total_students : ℕ) (overall_mean : ℚ) 
  (senior_count : ℕ) (non_senior_count : ℕ) (senior_mean : ℚ) (non_senior_mean : ℚ) :
  total_students = 120 →
  overall_mean = 110 →
  non_senior_count = 2 * senior_count →
  senior_mean = (3/2) * non_senior_mean →
  senior_count + non_senior_count = total_students →
  (senior_count * senior_mean + non_senior_count * non_senior_mean) / total_students = overall_mean →
  senior_mean = 141.43 := by
sorry

#eval (141.43 : ℚ)

end NUMINAMATH_CALUDE_senior_mean_score_l3168_316878


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l3168_316866

theorem quadratic_equation_result (m : ℝ) (h : 2 * m^2 + m = -1) : 4 * m^2 + 2 * m + 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l3168_316866


namespace NUMINAMATH_CALUDE_cannot_equalize_sugar_l3168_316829

/-- Represents a jar with tea and sugar -/
structure Jar :=
  (volume : ℚ)
  (sugar : ℚ)

/-- Represents the state of all three jars -/
structure JarState :=
  (jar1 : Jar)
  (jar2 : Jar)
  (jar3 : Jar)

/-- Represents a single pouring operation -/
inductive PourOperation
  | pour12 : PourOperation  -- Pour from jar1 to jar2
  | pour13 : PourOperation  -- Pour from jar1 to jar3
  | pour21 : PourOperation  -- Pour from jar2 to jar1
  | pour23 : PourOperation  -- Pour from jar2 to jar3
  | pour31 : PourOperation  -- Pour from jar3 to jar1
  | pour32 : PourOperation  -- Pour from jar3 to jar2

def initialState : JarState :=
  { jar1 := { volume := 0, sugar := 0 },
    jar2 := { volume := 700/1000, sugar := 50 },
    jar3 := { volume := 800/1000, sugar := 60 } }

def measureCup : ℚ := 100/1000

/-- Applies a single pouring operation to the current state -/
def applyOperation (state : JarState) (op : PourOperation) : JarState :=
  sorry

/-- Checks if the sugar content is equal in jars 2 and 3, and jar 1 is empty -/
def isDesiredState (state : JarState) : Prop :=
  state.jar1.volume = 0 ∧ state.jar2.sugar = state.jar3.sugar

/-- The main theorem to prove -/
theorem cannot_equalize_sugar : ¬∃ (ops : List PourOperation),
  isDesiredState (ops.foldl applyOperation initialState) :=
sorry

end NUMINAMATH_CALUDE_cannot_equalize_sugar_l3168_316829


namespace NUMINAMATH_CALUDE_peter_pictures_l3168_316879

theorem peter_pictures (peter_pictures : ℕ) (quincy_pictures : ℕ) (randy_pictures : ℕ)
  (h1 : quincy_pictures = peter_pictures + 20)
  (h2 : randy_pictures + peter_pictures + quincy_pictures = 41)
  (h3 : randy_pictures = 5) :
  peter_pictures = 8 := by
sorry

end NUMINAMATH_CALUDE_peter_pictures_l3168_316879


namespace NUMINAMATH_CALUDE_integral_of_special_function_l3168_316833

theorem integral_of_special_function (f : ℝ → ℝ) 
  (h_diff : Differentiable ℝ f) 
  (h_def : ∀ x, f x = x^3 + x^2 * (deriv f 1)) : 
  ∫ x in (0:ℝ)..(2:ℝ), f x = -4 := by
  sorry

end NUMINAMATH_CALUDE_integral_of_special_function_l3168_316833


namespace NUMINAMATH_CALUDE_workshop_handshakes_l3168_316801

/-- Represents the workshop scenario -/
structure Workshop where
  total_people : Nat
  trainers : Nat
  participants : Nat
  knowledgeable_participants : Nat
  trainers_known_by_knowledgeable : Nat

/-- Calculate the number of handshakes in the workshop -/
def count_handshakes (w : Workshop) : Nat :=
  let unknown_participants := w.participants - w.knowledgeable_participants
  let handshakes_unknown := unknown_participants * (w.total_people - 1)
  let handshakes_knowledgeable := w.knowledgeable_participants * (w.total_people - w.trainers_known_by_knowledgeable - 1)
  handshakes_unknown + handshakes_knowledgeable

/-- The theorem to be proved -/
theorem workshop_handshakes :
  let w : Workshop := {
    total_people := 40,
    trainers := 25,
    participants := 15,
    knowledgeable_participants := 5,
    trainers_known_by_knowledgeable := 10
  }
  count_handshakes w = 540 := by sorry

end NUMINAMATH_CALUDE_workshop_handshakes_l3168_316801


namespace NUMINAMATH_CALUDE_NaCl_selectively_precipitates_Ag_other_reagents_do_not_selectively_precipitate_Ag_l3168_316877

/-- Represents the solubility of a compound in water -/
inductive Solubility
  | Soluble
  | SlightlySoluble
  | Insoluble

/-- Represents a metal ion -/
inductive MetalIon
  | Ag
  | Mg
  | Sr

/-- Represents a reagent -/
inductive Reagent
  | NaCl
  | NaOH
  | Na2SO4
  | Na3PO4

/-- Returns the solubility of the compound formed by a metal ion and a reagent -/
def solubility (ion : MetalIon) (reagent : Reagent) : Solubility :=
  match ion, reagent with
  | MetalIon.Ag, Reagent.NaCl => Solubility.Insoluble
  | MetalIon.Mg, Reagent.NaCl => Solubility.Soluble
  | MetalIon.Sr, Reagent.NaCl => Solubility.Soluble
  | MetalIon.Ag, Reagent.NaOH => Solubility.Insoluble
  | MetalIon.Mg, Reagent.NaOH => Solubility.Insoluble
  | MetalIon.Sr, Reagent.NaOH => Solubility.SlightlySoluble
  | MetalIon.Ag, Reagent.Na2SO4 => Solubility.SlightlySoluble
  | MetalIon.Mg, Reagent.Na2SO4 => Solubility.Soluble
  | MetalIon.Sr, Reagent.Na2SO4 => Solubility.SlightlySoluble
  | MetalIon.Ag, Reagent.Na3PO4 => Solubility.Insoluble
  | MetalIon.Mg, Reagent.Na3PO4 => Solubility.Insoluble
  | MetalIon.Sr, Reagent.Na3PO4 => Solubility.Insoluble

/-- Checks if a reagent selectively precipitates Ag+ -/
def selectivelyPrecipitatesAg (reagent : Reagent) : Prop :=
  solubility MetalIon.Ag reagent = Solubility.Insoluble ∧
  solubility MetalIon.Mg reagent = Solubility.Soluble ∧
  solubility MetalIon.Sr reagent = Solubility.Soluble

theorem NaCl_selectively_precipitates_Ag :
  selectivelyPrecipitatesAg Reagent.NaCl :=
by sorry

theorem other_reagents_do_not_selectively_precipitate_Ag :
  ∀ r : Reagent, r ≠ Reagent.NaCl → ¬selectivelyPrecipitatesAg r :=
by sorry

end NUMINAMATH_CALUDE_NaCl_selectively_precipitates_Ag_other_reagents_do_not_selectively_precipitate_Ag_l3168_316877


namespace NUMINAMATH_CALUDE_midpoint_of_intersection_l3168_316800

-- Define the line
def line (x y : ℝ) : Prop := x - y = 2

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line A.1 A.2 ∧ parabola A.1 A.2 ∧
  line B.1 B.2 ∧ parabola B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem midpoint_of_intersection :
  ∀ A B : ℝ × ℝ, intersection_points A B →
  (A.1 + B.1) / 2 = 4 ∧ (A.2 + B.2) / 2 = 2 := by sorry

end NUMINAMATH_CALUDE_midpoint_of_intersection_l3168_316800


namespace NUMINAMATH_CALUDE_smallest_divisible_by_9_11_13_l3168_316816

theorem smallest_divisible_by_9_11_13 : ∃ n : ℕ, n > 0 ∧ 
  9 ∣ n ∧ 11 ∣ n ∧ 13 ∣ n ∧ 
  ∀ m : ℕ, m > 0 → 9 ∣ m → 11 ∣ m → 13 ∣ m → n ≤ m :=
by
  use 1287
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_9_11_13_l3168_316816


namespace NUMINAMATH_CALUDE_no_positive_sequence_with_sum_property_l3168_316851

open Real
open Set
open Nat

theorem no_positive_sequence_with_sum_property :
  ¬ (∃ b : ℕ → ℝ, 
    (∀ i : ℕ, i > 0 → b i > 0) ∧ 
    (∀ m : ℕ, m > 0 → (∑' k : ℕ, b (m * k)) = 1 / m)) := by
  sorry

end NUMINAMATH_CALUDE_no_positive_sequence_with_sum_property_l3168_316851


namespace NUMINAMATH_CALUDE_pat_height_l3168_316806

/-- Represents the depth dug on each day in centimeters -/
def depth_day1 : ℝ := 40

/-- Represents the total depth after day 2 in centimeters -/
def depth_day2 : ℝ := 3 * depth_day1

/-- Represents the additional depth dug on day 3 in centimeters -/
def depth_day3 : ℝ := depth_day2 - depth_day1

/-- Represents the distance from the ground surface to Pat's head at the end in centimeters -/
def surface_to_head : ℝ := 50

/-- Theorem stating Pat's height in centimeters -/
theorem pat_height : 
  depth_day2 + depth_day3 - surface_to_head = 150 := by sorry

end NUMINAMATH_CALUDE_pat_height_l3168_316806


namespace NUMINAMATH_CALUDE_triangle_side_length_l3168_316824

/-- In a triangle ABC, given that angle C is four times angle A, 
    side a is 35, and side c is 64, prove that side b equals 140cos²A -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧          -- Sum of angles in a triangle
  C = 4 * A ∧              -- Angle C is four times angle A
  a = 35 ∧                 -- Side a is 35
  c = 64 →                 -- Side c is 64
  b = 140 * (Real.cos A)^2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3168_316824


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l3168_316814

theorem solve_exponential_equation :
  ∃ x : ℝ, (125 : ℝ) = 5 * (25 : ℝ)^(x - 2) → x = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l3168_316814


namespace NUMINAMATH_CALUDE_floor_negative_seven_fourths_l3168_316820

theorem floor_negative_seven_fourths : ⌊(-7 : ℚ) / 4⌋ = -2 := by
  sorry

end NUMINAMATH_CALUDE_floor_negative_seven_fourths_l3168_316820


namespace NUMINAMATH_CALUDE_largest_d_for_g_range_contains_one_l3168_316871

/-- The quadratic function g(x) defined as 2x^2 - 8x + d -/
def g (d : ℝ) (x : ℝ) : ℝ := 2 * x^2 - 8 * x + d

/-- Theorem stating that the largest value of d such that 1 is in the range of g(x) is 9 -/
theorem largest_d_for_g_range_contains_one :
  (∃ (d : ℝ), ∀ (d' : ℝ), (∃ (x : ℝ), g d' x = 1) → d' ≤ d) ∧
  (∃ (x : ℝ), g 9 x = 1) :=
sorry

end NUMINAMATH_CALUDE_largest_d_for_g_range_contains_one_l3168_316871


namespace NUMINAMATH_CALUDE_timothy_initial_amount_matches_purchases_l3168_316846

/-- The amount of money Timothy had initially -/
def initial_amount : ℕ := 50

/-- The cost of a single t-shirt -/
def tshirt_cost : ℕ := 8

/-- The cost of a single bag -/
def bag_cost : ℕ := 10

/-- The number of t-shirts Timothy bought -/
def tshirts_bought : ℕ := 2

/-- The number of bags Timothy bought -/
def bags_bought : ℕ := 2

/-- The cost of a set of 3 key chains -/
def keychain_set_cost : ℕ := 2

/-- The number of key chains in a set -/
def keychains_per_set : ℕ := 3

/-- The number of key chains Timothy bought -/
def keychains_bought : ℕ := 21

/-- Theorem stating that Timothy's initial amount matches his purchases -/
theorem timothy_initial_amount_matches_purchases :
  initial_amount = 
    tshirts_bought * tshirt_cost + 
    bags_bought * bag_cost + 
    (keychains_bought / keychains_per_set) * keychain_set_cost :=
by
  sorry


end NUMINAMATH_CALUDE_timothy_initial_amount_matches_purchases_l3168_316846


namespace NUMINAMATH_CALUDE_min_value_of_f_l3168_316899

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem min_value_of_f :
  ∃ (min_val : ℝ), min_val = -2 ∧
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 → f x ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3168_316899


namespace NUMINAMATH_CALUDE_second_polygon_sides_l3168_316855

/-- Given two regular polygons with the same perimeter, where one polygon has 50 sides
    and each of its sides is three times as long as each side of the other polygon,
    prove that the number of sides of the second polygon is 150. -/
theorem second_polygon_sides (s : ℝ) (n : ℕ) : s > 0 →
  50 * (3 * s) = n * s → n = 150 := by
  sorry

end NUMINAMATH_CALUDE_second_polygon_sides_l3168_316855


namespace NUMINAMATH_CALUDE_common_chord_of_circles_l3168_316881

/-- Given two circles in the xy-plane, prove that their common chord has a specific equation. -/
theorem common_chord_of_circles (x y : ℝ) :
  (x^2 + y^2 + 2*x = 0) ∧ (x^2 + y^2 - 4*y = 0) → (x + 2*y = 0) :=
by sorry

end NUMINAMATH_CALUDE_common_chord_of_circles_l3168_316881


namespace NUMINAMATH_CALUDE_lg_expression_equals_zero_l3168_316831

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem lg_expression_equals_zero :
  lg 5 * lg 2 + lg (2^2) - lg 2 = 0 :=
by
  -- Properties of logarithms
  have h1 : ∀ m n : ℝ, lg (m^n) = n * lg m := sorry
  have h2 : ∀ a b : ℝ, lg (a * b) = lg a + lg b := sorry
  have h3 : lg 1 = 0 := sorry
  have h4 : lg 2 > 0 := sorry
  have h5 : lg 5 > 0 := sorry
  
  sorry -- Proof omitted

end NUMINAMATH_CALUDE_lg_expression_equals_zero_l3168_316831


namespace NUMINAMATH_CALUDE_local_min_implies_b_range_l3168_316805

theorem local_min_implies_b_range (b : ℝ) : 
  (∃ x ∈ Set.Ioo 0 1, IsLocalMin (fun x : ℝ ↦ x^3 - 3*b*x + 3*b) x) → 
  0 < b ∧ b < 1 :=
sorry

end NUMINAMATH_CALUDE_local_min_implies_b_range_l3168_316805


namespace NUMINAMATH_CALUDE_candy_pebbles_l3168_316875

theorem candy_pebbles (candy : ℕ) (lance : ℕ) : 
  lance = 3 * candy ∧ lance = candy + 8 → candy = 4 :=
by sorry

end NUMINAMATH_CALUDE_candy_pebbles_l3168_316875


namespace NUMINAMATH_CALUDE_jerry_cans_time_l3168_316859

def throw_away_cans (total_cans : ℕ) (cans_per_trip : ℕ) (drain_time : ℕ) (walk_time : ℕ) : ℕ :=
  let trips := (total_cans + cans_per_trip - 1) / cans_per_trip
  let drain_total := trips * drain_time
  let walk_total := trips * (2 * walk_time)
  drain_total + walk_total

theorem jerry_cans_time :
  throw_away_cans 35 3 30 10 = 600 := by
  sorry

end NUMINAMATH_CALUDE_jerry_cans_time_l3168_316859


namespace NUMINAMATH_CALUDE_helen_laundry_time_l3168_316895

/-- Represents the time spent on each activity for each item type -/
structure ItemTime where
  wash : Nat
  dry : Nat
  fold : Nat
  iron : Nat

/-- Calculates the total time spent on an item -/
def totalItemTime (item : ItemTime) : Nat :=
  item.wash + item.dry + item.fold + item.iron

/-- Represents the time Helen spends on her delicate items -/
structure HelenLaundryTime where
  silkPillowcases : ItemTime
  woolBlankets : ItemTime
  cashmereScarves : ItemTime
  washingInterval : Nat
  leapYear : Nat
  regularYear : Nat
  numRegularYears : Nat

/-- Calculates the total time Helen spends on laundry over the given period -/
def totalLaundryTime (h : HelenLaundryTime) : Nat :=
  let totalTimePerSession := totalItemTime h.silkPillowcases + totalItemTime h.woolBlankets + totalItemTime h.cashmereScarves
  let totalDays := h.leapYear + h.regularYear * h.numRegularYears
  let totalSessions := totalDays / h.washingInterval
  totalTimePerSession * totalSessions

theorem helen_laundry_time : 
  ∀ h : HelenLaundryTime,
    h.silkPillowcases = { wash := 30, dry := 20, fold := 10, iron := 5 } →
    h.woolBlankets = { wash := 45, dry := 30, fold := 15, iron := 20 } →
    h.cashmereScarves = { wash := 15, dry := 10, fold := 5, iron := 10 } →
    h.washingInterval = 28 →
    h.leapYear = 366 →
    h.regularYear = 365 →
    h.numRegularYears = 3 →
    totalLaundryTime h = 11180 := by
  sorry

end NUMINAMATH_CALUDE_helen_laundry_time_l3168_316895


namespace NUMINAMATH_CALUDE_expected_worth_is_one_third_l3168_316834

/-- The probability of getting heads on a coin flip -/
def prob_heads : ℚ := 2/3

/-- The probability of getting tails on a coin flip -/
def prob_tails : ℚ := 1/3

/-- The amount gained on a heads flip -/
def gain_heads : ℚ := 5

/-- The amount lost on a tails flip -/
def loss_tails : ℚ := 9

/-- The expected worth of a coin flip -/
def expected_worth : ℚ := prob_heads * gain_heads - prob_tails * loss_tails

theorem expected_worth_is_one_third : expected_worth = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_expected_worth_is_one_third_l3168_316834


namespace NUMINAMATH_CALUDE_a_share_l3168_316821

/-- Represents the share of money for each person -/
structure Share where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ

/-- The theorem stating A's share given the conditions -/
theorem a_share (s : Share) 
  (h1 : s.a = 5 * s.d ∧ s.b = 2 * s.d ∧ s.c = 4 * s.d) 
  (h2 : s.c = s.d + 500) : 
  s.a = 2500 := by
  sorry

end NUMINAMATH_CALUDE_a_share_l3168_316821


namespace NUMINAMATH_CALUDE_female_employees_count_l3168_316898

/-- Given a company with the following properties:
  1. There are 280 female managers.
  2. 2/5 of all employees are managers.
  3. 2/5 of all male employees are managers.
  Prove that the total number of female employees is 700. -/
theorem female_employees_count (total_employees : ℕ) (male_employees : ℕ) :
  let female_managers : ℕ := 280
  let total_managers : ℕ := (2 * total_employees) / 5
  let male_managers : ℕ := (2 * male_employees) / 5
  total_managers = female_managers + male_managers →
  total_employees - male_employees = 700 := by
  sorry

end NUMINAMATH_CALUDE_female_employees_count_l3168_316898


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3168_316840

def U : Set ℕ := {x | x > 0 ∧ x^2 - 9*x + 8 ≤ 0}

def A : Set ℕ := {1, 2, 3}

def B : Set ℕ := {5, 6, 7}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {4, 8} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3168_316840


namespace NUMINAMATH_CALUDE_cross_section_distance_from_apex_l3168_316883

-- Define the structure of a right pentagonal pyramid
structure RightPentagonalPyramid where
  -- Add any necessary fields

-- Define a cross section of the pyramid
structure CrossSection where
  area : ℝ
  distanceFromApex : ℝ

-- Define the theorem
theorem cross_section_distance_from_apex 
  (pyramid : RightPentagonalPyramid)
  (section1 section2 : CrossSection)
  (h1 : section1.area = 125 * Real.sqrt 3)
  (h2 : section2.area = 500 * Real.sqrt 3)
  (h3 : section2.distanceFromApex - section1.distanceFromApex = 12)
  (h4 : section2.area > section1.area) :
  section2.distanceFromApex = 24 := by
sorry

end NUMINAMATH_CALUDE_cross_section_distance_from_apex_l3168_316883


namespace NUMINAMATH_CALUDE_adjacent_numbers_to_10000_l3168_316869

theorem adjacent_numbers_to_10000 :
  let adjacent_numbers (n : ℤ) := (n - 1, n + 1)
  adjacent_numbers 10000 = (9999, 10001) := by
  sorry

end NUMINAMATH_CALUDE_adjacent_numbers_to_10000_l3168_316869


namespace NUMINAMATH_CALUDE_tanya_accompanied_twice_l3168_316813

/-- Represents a girl in the group --/
inductive Girl
| Anya
| Tanya
| Olya
| Katya

/-- The number of songs sung by each girl --/
def songsSung (g : Girl) : ℕ :=
  match g with
  | Girl.Anya => 8
  | Girl.Tanya => 6
  | Girl.Olya => 3
  | Girl.Katya => 7

/-- The total number of girls --/
def totalGirls : ℕ := 4

/-- The number of singers per song --/
def singersPerSong : ℕ := 3

/-- The total number of songs played --/
def totalSongs : ℕ := (songsSung Girl.Anya + songsSung Girl.Tanya + songsSung Girl.Olya + songsSung Girl.Katya) / singersPerSong

/-- The number of times Tanya accompanied --/
def tanyaAccompanied : ℕ := totalSongs - songsSung Girl.Tanya

theorem tanya_accompanied_twice :
  tanyaAccompanied = 2 :=
sorry

end NUMINAMATH_CALUDE_tanya_accompanied_twice_l3168_316813


namespace NUMINAMATH_CALUDE_eighth_grade_girls_l3168_316837

/-- Given the number of boys and girls in eighth grade, proves the number of girls -/
theorem eighth_grade_girls (total : ℕ) (boys girls : ℕ) : 
  total = 68 → 
  boys = 2 * girls - 16 → 
  boys + girls = total → 
  girls = 28 := by
sorry

end NUMINAMATH_CALUDE_eighth_grade_girls_l3168_316837


namespace NUMINAMATH_CALUDE_factorization_proof_l3168_316861

theorem factorization_proof (x : ℝ) : 
  (2 * x^3 - 8 * x^2 = 2 * x^2 * (x - 4)) ∧ 
  (x^2 - 14 * x + 49 = (x - 7)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l3168_316861


namespace NUMINAMATH_CALUDE_sarah_savings_l3168_316873

/-- Represents Sarah's savings pattern over time -/
def savings_pattern : List (Nat × Nat) :=
  [(4, 5), (4, 10), (4, 20)]

/-- Calculates the total amount saved given a savings pattern -/
def total_saved (pattern : List (Nat × Nat)) : Nat :=
  pattern.foldl (fun acc (weeks, amount) => acc + weeks * amount) 0

/-- Calculates the total number of weeks in a savings pattern -/
def total_weeks (pattern : List (Nat × Nat)) : Nat :=
  pattern.foldl (fun acc (weeks, _) => acc + weeks) 0

/-- Theorem: Sarah saves $140 in 12 weeks -/
theorem sarah_savings : 
  total_saved savings_pattern = 140 ∧ total_weeks savings_pattern = 12 :=
sorry

end NUMINAMATH_CALUDE_sarah_savings_l3168_316873


namespace NUMINAMATH_CALUDE_cookie_eating_contest_l3168_316897

theorem cookie_eating_contest (first_student second_student : ℚ) : 
  first_student = 5/6 → second_student = 2/3 → first_student - second_student = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_cookie_eating_contest_l3168_316897


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l3168_316843

theorem sqrt_expression_equality : 
  2 * Real.sqrt 12 * (3 * Real.sqrt 48 - 4 * Real.sqrt (1/8) - 3 * Real.sqrt 27) = 36 - 4 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l3168_316843


namespace NUMINAMATH_CALUDE_plane_equation_l3168_316850

def point_on_plane (A B C D : ℤ) (x y z : ℝ) : Prop :=
  A * x + B * y + C * z + D = 0

def perpendicular_planes (A1 B1 C1 D1 A2 B2 C2 D2 : ℤ) : Prop :=
  A1 * A2 + B1 * B2 + C1 * C2 = 0

theorem plane_equation : ∃ (A B C D : ℤ),
  (A > 0) ∧
  (Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Nat.gcd (Int.natAbs C) (Int.natAbs D)) = 1) ∧
  point_on_plane A B C D 0 0 0 ∧
  point_on_plane A B C D 2 (-2) 2 ∧
  perpendicular_planes A B C D 2 (-1) 3 4 ∧
  A = 2 ∧ B = -1 ∧ C = 1 ∧ D = 0 :=
sorry

end NUMINAMATH_CALUDE_plane_equation_l3168_316850


namespace NUMINAMATH_CALUDE_sum_of_digits_divisibility_l3168_316862

/-- Sum of digits function -/
def sum_of_digits (a : ℕ) : ℕ := sorry

/-- Theorem: If the sum of digits of a equals the sum of digits of 2a, then a is divisible by 9 -/
theorem sum_of_digits_divisibility (a : ℕ) : sum_of_digits a = sum_of_digits (2 * a) → 9 ∣ a := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_divisibility_l3168_316862


namespace NUMINAMATH_CALUDE_jen_work_hours_l3168_316839

/-- 
Given that:
- Jen works 7 hours a week more than Ben
- Jen's work in 4 weeks equals Ben's work in 6 weeks
Prove that Jen works 21 hours per week
-/
theorem jen_work_hours (ben_hours : ℕ) 
  (h1 : ben_hours + 7 = 4 * ben_hours + 28) :
  ben_hours + 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_jen_work_hours_l3168_316839


namespace NUMINAMATH_CALUDE_nuts_ratio_l3168_316848

/-- Given the following conditions:
  - Sue has 48 nuts
  - Bill has 6 times as many nuts as Harry
  - Bill and Harry have combined 672 nuts
  Prove that the ratio of Harry's nuts to Sue's nuts is 2:1 -/
theorem nuts_ratio (sue_nuts : ℕ) (bill_harry_total : ℕ) :
  sue_nuts = 48 →
  bill_harry_total = 672 →
  ∃ (harry_nuts : ℕ),
    harry_nuts + 6 * harry_nuts = bill_harry_total ∧
    harry_nuts / sue_nuts = 2 :=
by sorry

end NUMINAMATH_CALUDE_nuts_ratio_l3168_316848


namespace NUMINAMATH_CALUDE_triangle_properties_l3168_316889

theorem triangle_properties (a b c A B C : ℝ) (r : ℝ) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  -- Condition: √3b = a(√3cosC - sinC)
  (Real.sqrt 3 * b = a * (Real.sqrt 3 * Real.cos C - Real.sin C)) →
  -- Condition: a = 8
  (a = 8) →
  -- Condition: Radius of incircle = √3
  (r = Real.sqrt 3) →
  -- Proof that angle A = 2π/3
  (A = 2 * Real.pi / 3) ∧
  -- Proof that perimeter = 18
  (a + b + c = 18) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3168_316889


namespace NUMINAMATH_CALUDE_division_problem_l3168_316865

theorem division_problem (A : ℕ) (h : 59 = 8 * A + 3) : A = 7 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3168_316865


namespace NUMINAMATH_CALUDE_children_playing_neither_sport_l3168_316841

theorem children_playing_neither_sport (total : ℕ) (tennis : ℕ) (squash : ℕ) (both : ℕ) 
  (h1 : total = 38)
  (h2 : tennis = 19)
  (h3 : squash = 21)
  (h4 : both = 12) :
  total - (tennis + squash - both) = 10 := by
  sorry

end NUMINAMATH_CALUDE_children_playing_neither_sport_l3168_316841


namespace NUMINAMATH_CALUDE_coin_problem_l3168_316804

theorem coin_problem :
  ∀ (nickels dimes quarters : ℕ),
    nickels + dimes + quarters = 100 →
    5 * nickels + 10 * dimes + 25 * quarters = 835 →
    ∃ (min_dimes max_dimes : ℕ),
      (∀ d : ℕ, 
        (∃ n q : ℕ, n + d + q = 100 ∧ 5 * n + 10 * d + 25 * q = 835) →
        min_dimes ≤ d ∧ d ≤ max_dimes) ∧
      max_dimes - min_dimes = 64 := by
  sorry

end NUMINAMATH_CALUDE_coin_problem_l3168_316804


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l3168_316830

theorem min_value_sum_squares (x y z : ℝ) (h : 2*x + 3*y + 4*z = 1) :
  ∃ (min : ℝ), min = (1 : ℝ) / 29 ∧ x^2 + y^2 + z^2 ≥ min ∧
  (x^2 + y^2 + z^2 = min ↔ x/2 = y/3 ∧ y/3 = z/4) :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l3168_316830


namespace NUMINAMATH_CALUDE_inscribed_cone_volume_l3168_316892

/-- The volume of an inscribed cone in a larger cone -/
theorem inscribed_cone_volume 
  (H : ℝ) -- Height of the outer cone
  (α : ℝ) -- Angle between slant height and altitude of outer cone
  (h_pos : H > 0) -- Assumption that height is positive
  (α_range : 0 < α ∧ α < π/2) -- Assumption that α is between 0 and π/2
  : ∃ (V : ℝ), 
    -- V represents the volume of the inscribed cone
    -- The inscribed cone's vertex coincides with the center of the base of the outer cone
    -- The slant heights of both cones are mutually perpendicular
    V = (1/12) * π * H^3 * (Real.sin α)^2 * (Real.sin (2*α))^2 :=
by
  sorry

end NUMINAMATH_CALUDE_inscribed_cone_volume_l3168_316892


namespace NUMINAMATH_CALUDE_men_to_women_percentage_l3168_316870

/-- If the population of women is 50% of the population of men,
    then the population of men is 200% of the population of women. -/
theorem men_to_women_percentage (men women : ℝ) (h : women = 0.5 * men) :
  men / women * 100 = 200 := by
  sorry

end NUMINAMATH_CALUDE_men_to_women_percentage_l3168_316870


namespace NUMINAMATH_CALUDE_painted_cubes_l3168_316808

theorem painted_cubes (n : ℕ) (interior_cubes : ℕ) : 
  n = 4 → 
  interior_cubes = 23 → 
  n^3 - interior_cubes = 41 :=
by sorry

end NUMINAMATH_CALUDE_painted_cubes_l3168_316808


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l3168_316836

def QuadraticFunction (a h k : ℝ) : ℝ → ℝ := fun x ↦ a * (x - h)^2 + k

theorem quadratic_function_theorem (a : ℝ) (h k : ℝ) :
  (∀ x, QuadraticFunction a h k x ≤ 2) ∧
  QuadraticFunction a h k 2 = 1 ∧
  QuadraticFunction a h k 4 = 1 →
  h = 3 ∧ k = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l3168_316836


namespace NUMINAMATH_CALUDE_w_magnitude_bounds_l3168_316827

theorem w_magnitude_bounds (z : ℂ) (h : Complex.abs z = 1) : 
  let w : ℂ := z^4 - z^3 - 3 * z^2 * Complex.I - z + 1
  3 ≤ Complex.abs w ∧ Complex.abs w ≤ 5 := by
sorry

end NUMINAMATH_CALUDE_w_magnitude_bounds_l3168_316827


namespace NUMINAMATH_CALUDE_largest_integer_in_range_l3168_316890

theorem largest_integer_in_range : 
  ∃ (x : ℤ), (1/4 : ℚ) < (x : ℚ)/7 ∧ (x : ℚ)/7 < 3/5 ∧ 
  ∀ (y : ℤ), (1/4 : ℚ) < (y : ℚ)/7 ∧ (y : ℚ)/7 < 3/5 → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_in_range_l3168_316890


namespace NUMINAMATH_CALUDE_max_value_product_l3168_316825

open Real

-- Define the function f(x) = ln(x+2) - x
noncomputable def f (x : ℝ) : ℝ := log (x + 2) - x

-- Define the derivative of f(x)
noncomputable def f' (x : ℝ) : ℝ := 1 / (x + 2) - 1

theorem max_value_product (a b : ℝ) : f' a = 0 → f a = b → a * b = -1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_product_l3168_316825


namespace NUMINAMATH_CALUDE_class_average_problem_l3168_316802

/-- Given a class where:
  - 20% of students average 80% on a test
  - 50% of students average X% on a test
  - 30% of students average 40% on a test
  - The overall class average is 58%
  Prove that X = 60 -/
theorem class_average_problem (X : ℝ) : 
  0.2 * 80 + 0.5 * X + 0.3 * 40 = 58 → X = 60 := by
  sorry

end NUMINAMATH_CALUDE_class_average_problem_l3168_316802


namespace NUMINAMATH_CALUDE_parallel_lines_parallelograms_l3168_316832

/-- The number of ways to choose 2 items from n items -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of parallelograms formed by intersecting sets of parallel lines -/
def parallelograms_count (set1 : ℕ) (set2 : ℕ) : ℕ :=
  (choose_two set1) * (choose_two set2)

theorem parallel_lines_parallelograms :
  parallelograms_count 3 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_parallelograms_l3168_316832


namespace NUMINAMATH_CALUDE_binomial_16_12_l3168_316849

theorem binomial_16_12 : Nat.choose 16 12 = 1820 := by
  sorry

end NUMINAMATH_CALUDE_binomial_16_12_l3168_316849


namespace NUMINAMATH_CALUDE_room_carpet_cost_l3168_316811

/-- Calculates the total cost of carpeting a rectangular room -/
def carpet_cost (length width cost_per_sq_yard : ℚ) : ℚ :=
  let length_yards := length / 3
  let width_yards := width / 3
  let area_sq_yards := length_yards * width_yards
  area_sq_yards * cost_per_sq_yard

/-- Theorem stating the total cost of carpeting the given room -/
theorem room_carpet_cost :
  carpet_cost 15 12 10 = 200 := by
  sorry

end NUMINAMATH_CALUDE_room_carpet_cost_l3168_316811


namespace NUMINAMATH_CALUDE_room_breadth_calculation_l3168_316853

/-- Given a room with specified dimensions and carpeting costs, calculate its breadth. -/
theorem room_breadth_calculation (room_length : ℝ) (carpet_width : ℝ) (carpet_cost_per_meter : ℝ) (total_cost : ℝ) :
  room_length = 15 →
  carpet_width = 0.75 →
  carpet_cost_per_meter = 0.3 →
  total_cost = 36 →
  (total_cost / carpet_cost_per_meter) * carpet_width / room_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_room_breadth_calculation_l3168_316853


namespace NUMINAMATH_CALUDE_jumping_contest_l3168_316872

/-- The jumping contest problem -/
theorem jumping_contest (grasshopper_jump : ℕ) (frog_extra : ℕ) (mouse_extra : ℕ) 
  (h1 : grasshopper_jump = 19)
  (h2 : frog_extra = 10)
  (h3 : mouse_extra = 20) :
  (grasshopper_jump + frog_extra + mouse_extra) - grasshopper_jump = 30 := by
  sorry


end NUMINAMATH_CALUDE_jumping_contest_l3168_316872


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l3168_316809

theorem pure_imaginary_condition (b : ℝ) (i : ℂ) : 
  i * i = -1 →  -- i is the imaginary unit
  (∃ (k : ℝ), i * (b * i + 1) = k * i) →  -- i(bi+1) is a pure imaginary number
  b = 0 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l3168_316809


namespace NUMINAMATH_CALUDE_twelve_balloons_floated_away_l3168_316812

/-- Calculates the number of balloons that floated away -/
def balloons_floated_away (initial_count : ℕ) (given_away : ℕ) (grabbed : ℕ) (final_count : ℕ) : ℕ :=
  initial_count - given_away + grabbed - final_count

/-- Proves that 12 balloons floated away given the problem conditions -/
theorem twelve_balloons_floated_away :
  balloons_floated_away 50 10 11 39 = 12 := by
  sorry

#eval balloons_floated_away 50 10 11 39

end NUMINAMATH_CALUDE_twelve_balloons_floated_away_l3168_316812


namespace NUMINAMATH_CALUDE_rectangle_area_l3168_316838

-- Define the rectangle ABCD
def rectangle (AB DE : ℝ) : Prop :=
  DE - AB = 9 ∧ DE > AB ∧ AB > 0

-- Define the relationship between areas of trapezoid ABCE and triangle ADE
def area_relation (AB DE : ℝ) : Prop :=
  (AB * DE) / 2 = 5 * ((DE - AB) * AB / 2)

-- Define the relationship between perimeters
def perimeter_relation (AB : ℝ) : Prop :=
  AB * 4/3 = 68

-- Main theorem
theorem rectangle_area (AB DE : ℝ) :
  rectangle AB DE →
  area_relation AB DE →
  perimeter_relation AB →
  AB * DE = 3060 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l3168_316838


namespace NUMINAMATH_CALUDE_m_squared_plus_inverse_squared_plus_three_l3168_316868

theorem m_squared_plus_inverse_squared_plus_three (m : ℝ) (h : m + 1/m = 6) :
  m^2 + 1/m^2 + 3 = 37 := by sorry

end NUMINAMATH_CALUDE_m_squared_plus_inverse_squared_plus_three_l3168_316868


namespace NUMINAMATH_CALUDE_probability_product_greater_than_five_l3168_316817

def S : Finset ℕ := {1, 2, 3, 4, 5}

def pairs : Finset (ℕ × ℕ) := S.product S |>.filter (λ (a, b) => a < b)

def valid_pairs : Finset (ℕ × ℕ) := pairs.filter (λ (a, b) => a * b > 5)

theorem probability_product_greater_than_five :
  (valid_pairs.card : ℚ) / pairs.card = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_product_greater_than_five_l3168_316817


namespace NUMINAMATH_CALUDE_camel_height_28_feet_l3168_316854

/-- The height of a camel in feet, given the height of a hare in inches and their relative heights -/
def camel_height_in_feet (hare_height_inches : ℕ) (camel_hare_ratio : ℕ) : ℚ :=
  (hare_height_inches * camel_hare_ratio : ℚ) / 12

/-- Theorem stating the height of a camel in feet given specific measurements -/
theorem camel_height_28_feet :
  camel_height_in_feet 14 24 = 28 := by
  sorry

end NUMINAMATH_CALUDE_camel_height_28_feet_l3168_316854


namespace NUMINAMATH_CALUDE_exponent_equality_l3168_316857

theorem exponent_equality 
  (a b c d : ℝ) 
  (x y q z : ℝ) 
  (h1 : a^(2*x) = c^(3*q)) 
  (h2 : a^(2*x) = b) 
  (h3 : c^(2*y) = a^(3*z)) 
  (h4 : c^(2*y) = d) 
  (h5 : a ≠ 0) 
  (h6 : c ≠ 0) : 
  2*x * 3*z = 3*q * 2*y := by
sorry

end NUMINAMATH_CALUDE_exponent_equality_l3168_316857


namespace NUMINAMATH_CALUDE_pen_cost_l3168_316823

theorem pen_cost (pen pencil : ℚ) 
  (h1 : 3 * pen + 4 * pencil = 264/100)
  (h2 : 4 * pen + 2 * pencil = 230/100) : 
  pen = 392/1000 := by
sorry

end NUMINAMATH_CALUDE_pen_cost_l3168_316823


namespace NUMINAMATH_CALUDE_inequality_proof_l3168_316887

def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

theorem inequality_proof (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∀ x : ℝ, f 1 x ≤ 1 ↔ 0 ≤ x ∧ x ≤ 2) →
  1/m + 1/(2*n) = 1 →
  m + 2*n ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3168_316887


namespace NUMINAMATH_CALUDE_arithmetic_sequences_ratio_l3168_316876

/-- Sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℚ) (d : ℚ) (aₙ : ℚ) : ℚ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_sequences_ratio : 
  let seq1_sum := arithmetic_sum 5 3 59
  let seq2_sum := arithmetic_sum 4 4 64
  seq1_sum / seq2_sum = 19 / 17 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequences_ratio_l3168_316876


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_l3168_316874

theorem quadratic_roots_difference (a b : ℝ) : 
  (∀ t : ℝ, t^2 - 12*t + 20 = 0 ↔ t = a ∨ t = b) →
  a > b →
  a - b = 8 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_l3168_316874


namespace NUMINAMATH_CALUDE_andrea_height_l3168_316810

/-- Given a tree's height and shadow length, and a person's shadow length,
    calculate the person's height assuming the same lighting conditions. -/
theorem andrea_height (tree_height shadow_tree shadow_andrea : ℝ) 
    (h_tree : tree_height = 70)
    (h_shadow_tree : shadow_tree = 14)
    (h_shadow_andrea : shadow_andrea = 3.5) :
  tree_height / shadow_tree * shadow_andrea = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_andrea_height_l3168_316810


namespace NUMINAMATH_CALUDE_max_students_distribution_l3168_316822

theorem max_students_distribution (pens pencils : ℕ) (h1 : pens = 2730) (h2 : pencils = 1890) :
  Nat.gcd pens pencils = 210 := by
  sorry

end NUMINAMATH_CALUDE_max_students_distribution_l3168_316822


namespace NUMINAMATH_CALUDE_impossible_visit_all_squares_l3168_316844

/-- Represents a position on the chessboard -/
structure Position :=
  (x : Fin 6)
  (y : Fin 6)

/-- Represents a jump on the chessboard -/
inductive Jump
  | One
  | Two

/-- Represents a sequence of jumps -/
def JumpSequence := List Jump

/-- Checks if a jump sequence is valid (alternating One and Two) -/
def isValidJumpSequence : JumpSequence → Bool
  | [] => true
  | [_] => true
  | Jump.One :: Jump.Two :: rest => isValidJumpSequence rest
  | Jump.Two :: Jump.One :: rest => isValidJumpSequence rest
  | _ => false

/-- Applies a jump to a position -/
def applyJump (pos : Position) (jump : Jump) (direction : Bool) : Position :=
  match jump, direction with
  | Jump.One, true => ⟨pos.x + 1, pos.y⟩
  | Jump.One, false => ⟨pos.x, pos.y + 1⟩
  | Jump.Two, true => ⟨pos.x + 2, pos.y⟩
  | Jump.Two, false => ⟨pos.x, pos.y + 2⟩

/-- Applies a sequence of jumps to a position -/
def applyJumpSequence (pos : Position) (jumps : JumpSequence) (directions : List Bool) : List Position :=
  match jumps, directions with
  | [], _ => [pos]
  | j :: js, d :: ds => pos :: applyJumpSequence (applyJump pos j d) js ds
  | _, _ => [pos]

/-- Theorem: It's impossible to visit all squares on a 6x6 chessboard
    with 35 jumps alternating between 1 and 2 squares -/
theorem impossible_visit_all_squares :
  ∀ (start : Position) (jumps : JumpSequence) (directions : List Bool),
    isValidJumpSequence jumps →
    jumps.length = 35 →
    directions.length = 35 →
    ¬(∀ (p : Position), p ∈ applyJumpSequence start jumps directions) :=
by
  sorry

end NUMINAMATH_CALUDE_impossible_visit_all_squares_l3168_316844


namespace NUMINAMATH_CALUDE_five_digit_divisible_by_nine_l3168_316894

theorem five_digit_divisible_by_nine (B : ℕ) : 
  B < 10 →
  (40000 + 10000 * B + 500 + 20 + B) % 9 = 0 →
  B = 8 :=
by sorry

end NUMINAMATH_CALUDE_five_digit_divisible_by_nine_l3168_316894


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l3168_316863

theorem quadratic_no_real_roots : ∀ x : ℝ, 3 * x^2 - 6 * x + 4 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l3168_316863


namespace NUMINAMATH_CALUDE_negation_equivalence_l3168_316885

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3168_316885


namespace NUMINAMATH_CALUDE_alpha_beta_equivalence_l3168_316882

theorem alpha_beta_equivalence (α β : ℝ) :
  (α + β > 0) ↔ (α + β > Real.cos α - Real.cos β) := by
  sorry

end NUMINAMATH_CALUDE_alpha_beta_equivalence_l3168_316882


namespace NUMINAMATH_CALUDE_sticker_remainder_l3168_316826

theorem sticker_remainder (nina_stickers : Nat) (oliver_stickers : Nat) (patty_stickers : Nat) 
  (package_size : Nat) (h1 : nina_stickers = 53) (h2 : oliver_stickers = 68) 
  (h3 : patty_stickers = 29) (h4 : package_size = 18) : 
  (nina_stickers + oliver_stickers + patty_stickers) % package_size = 6 := by
  sorry

end NUMINAMATH_CALUDE_sticker_remainder_l3168_316826


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3168_316815

-- Define the inequality function
def f (m x : ℝ) : ℝ := m * x^2 + (2*m - 1) * x - 2

-- Define the solution set
def solution_set (m : ℝ) : Set ℝ :=
  if m < -1/2 then Set.Ioo (-2) (1/m)
  else if m = -1/2 then ∅
  else if -1/2 < m ∧ m < 0 then Set.Ioo (1/m) (-2)
  else if m = 0 then Set.Ioi (-2)
  else Set.union (Set.Iio (-2)) (Set.Ioi (1/m))

-- Theorem statement
theorem inequality_solution_set (m : ℝ) :
  {x : ℝ | f m x > 0} = solution_set m :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3168_316815


namespace NUMINAMATH_CALUDE_number_puzzle_l3168_316845

theorem number_puzzle (N : ℝ) : 6 + (1/2) * (1/3) * (1/5) * N = (1/15) * N → N = 180 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l3168_316845
