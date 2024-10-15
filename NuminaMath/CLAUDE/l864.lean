import Mathlib

namespace NUMINAMATH_CALUDE_range_of_a_l864_86432

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) ↔ -2 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l864_86432


namespace NUMINAMATH_CALUDE_digit_150_of_one_thirteenth_l864_86436

def decimal_representation (n : ℕ) : ℚ → List ℕ := sorry

def nth_digit (n : ℕ) (l : List ℕ) : ℕ := sorry

theorem digit_150_of_one_thirteenth :
  let rep := decimal_representation 13 (1/13)
  nth_digit 150 rep = 3 := by sorry

end NUMINAMATH_CALUDE_digit_150_of_one_thirteenth_l864_86436


namespace NUMINAMATH_CALUDE_negation_of_existence_cubic_equation_negation_l864_86420

theorem negation_of_existence (f : ℝ → ℝ) :
  (¬ ∃ x, f x = 0) ↔ (∀ x, f x ≠ 0) := by sorry

theorem cubic_equation_negation :
  (¬ ∃ x : ℝ, x^3 - 2*x + 1 = 0) ↔ (∀ x : ℝ, x^3 - 2*x + 1 ≠ 0) := by
  apply negation_of_existence

end NUMINAMATH_CALUDE_negation_of_existence_cubic_equation_negation_l864_86420


namespace NUMINAMATH_CALUDE_sentence_reappears_l864_86422

-- Define the type for documents
def Document := List String

-- Define William's word assignment function
def wordAssignment : Char → String := sorry

-- Generate the nth document
def generateDocument : ℕ → Document
  | 0 => [wordAssignment 'A']
  | n + 1 => sorry -- Replace each letter in the previous document with its assigned word

-- The 40th document starts with this sentence
def startingSentence : List String :=
  ["Till", "whatsoever", "star", "that", "guides", "my", "moving"]

-- Main theorem
theorem sentence_reappears (d : Document) (h : d = generateDocument 40) :
  ∃ (i j : ℕ), i < j ∧ j < d.length ∧
  (List.take 7 (List.drop i d) = startingSentence) ∧
  (List.take 7 (List.drop j d) = startingSentence) :=
sorry

end NUMINAMATH_CALUDE_sentence_reappears_l864_86422


namespace NUMINAMATH_CALUDE_johns_price_calculation_l864_86405

/-- The price per sheet charged by John's Photo World -/
def johns_price_per_sheet : ℚ := 2.75

/-- The sitting fee charged by John's Photo World -/
def johns_sitting_fee : ℚ := 125

/-- The price per sheet charged by Sam's Picture Emporium -/
def sams_price_per_sheet : ℚ := 1.50

/-- The sitting fee charged by Sam's Picture Emporium -/
def sams_sitting_fee : ℚ := 140

/-- The number of sheets for which both companies charge the same amount -/
def num_sheets : ℕ := 12

theorem johns_price_calculation :
  johns_price_per_sheet * num_sheets + johns_sitting_fee =
  sams_price_per_sheet * num_sheets + sams_sitting_fee :=
by sorry

end NUMINAMATH_CALUDE_johns_price_calculation_l864_86405


namespace NUMINAMATH_CALUDE_cubic_equation_natural_roots_l864_86426

theorem cubic_equation_natural_roots (p : ℝ) : 
  (∃ (x y : ℕ) (z : ℝ), 
    x ≠ y ∧ 
    (5 * x^3 - 5*(p+1)*x^2 + (71*p-1)*x + 1 = 66*p) ∧
    (5 * y^3 - 5*(p+1)*y^2 + (71*p-1)*y + 1 = 66*p) ∧
    (5 * z^3 - 5*(p+1)*z^2 + (71*p-1)*z + 1 = 66*p)) ↔ 
  p = 76 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_natural_roots_l864_86426


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l864_86489

theorem arithmetic_mean_of_fractions :
  (5 : ℚ) / 6 = ((3 : ℚ) / 4 + (7 : ℚ) / 8) / 2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l864_86489


namespace NUMINAMATH_CALUDE_negative_majority_sequence_l864_86480

theorem negative_majority_sequence :
  ∃ (x : Fin 2004 → ℤ),
    (∀ k : Fin 2001, x (k + 3) = x (k + 2) + x k * x (k + 1)) ∧
    (∃ n : ℕ, 2 * n > 2004 ∧ (∃ S : Finset (Fin 2004), S.card = n ∧ ∀ i ∈ S, x i < 0)) := by
  sorry

end NUMINAMATH_CALUDE_negative_majority_sequence_l864_86480


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l864_86493

/-- The lateral surface area of a cone with base radius 3 and central angle of its lateral surface unfolded diagram 90° is 36π. -/
theorem cone_lateral_surface_area :
  let base_radius : ℝ := 3
  let central_angle : ℝ := 90
  let lateral_surface_area : ℝ := (1 / 2) * (2 * Real.pi * base_radius) * (4 * base_radius)
  lateral_surface_area = 36 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l864_86493


namespace NUMINAMATH_CALUDE_matts_assignment_problems_l864_86487

/-- The number of minutes it takes Matt to solve one problem with a calculator -/
def time_with_calculator : ℕ := 2

/-- The number of minutes it takes Matt to solve one problem without a calculator -/
def time_without_calculator : ℕ := 5

/-- The total number of minutes saved by using a calculator -/
def time_saved : ℕ := 60

/-- The number of problems in Matt's assignment -/
def number_of_problems : ℕ := 20

theorem matts_assignment_problems :
  (time_without_calculator - time_with_calculator) * number_of_problems = time_saved :=
by sorry

end NUMINAMATH_CALUDE_matts_assignment_problems_l864_86487


namespace NUMINAMATH_CALUDE_junior_percentage_is_22_l864_86494

/-- Represents the number of students in each grade --/
structure StudentCount where
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ
  seniors : ℕ

/-- The total number of students in the sample --/
def totalStudents : ℕ := 800

/-- The conditions of the problem --/
def sampleConditions (s : StudentCount) : Prop :=
  s.freshmen + s.sophomores + s.juniors + s.seniors = totalStudents ∧
  s.sophomores = totalStudents / 4 ∧
  s.seniors = 160 ∧
  s.freshmen = s.sophomores + 64

/-- The percentage of juniors in the sample --/
def juniorPercentage (s : StudentCount) : ℚ :=
  s.juniors * 100 / totalStudents

/-- Theorem stating that the percentage of juniors is 22% --/
theorem junior_percentage_is_22 (s : StudentCount) 
  (h : sampleConditions s) : juniorPercentage s = 22 := by
  sorry

end NUMINAMATH_CALUDE_junior_percentage_is_22_l864_86494


namespace NUMINAMATH_CALUDE_product_purchase_l864_86430

theorem product_purchase (misunderstood_total : ℕ) (actual_total : ℕ) 
  (h1 : misunderstood_total = 189)
  (h2 : actual_total = 147) :
  ∃ (price : ℕ) (quantity : ℕ),
    price * quantity = actual_total ∧
    (price + 6) * quantity = misunderstood_total ∧
    price = 21 ∧
    quantity = 7 := by
  sorry

end NUMINAMATH_CALUDE_product_purchase_l864_86430


namespace NUMINAMATH_CALUDE_bubble_sort_probability_correct_l864_86482

/-- The probability of the 25th element in a random permutation of 50 distinct real numbers
    ending up in the 40th position after one bubble pass -/
def bubble_sort_probability : ℚ :=
  1 / 1640

/-- The sequence length -/
def n : ℕ := 50

/-- The initial position of the element we're tracking -/
def initial_position : ℕ := 25

/-- The final position of the element we're tracking -/
def final_position : ℕ := 40

theorem bubble_sort_probability_correct :
  bubble_sort_probability = 1 / 1640 ∧ n = 50 ∧ initial_position = 25 ∧ final_position = 40 := by
  sorry

end NUMINAMATH_CALUDE_bubble_sort_probability_correct_l864_86482


namespace NUMINAMATH_CALUDE_tangent_circle_height_difference_l864_86491

/-- A circle tangent to the parabola y = x^2 + 1 at two points and inside the parabola -/
structure TangentCircle where
  /-- x-coordinate of one tangent point -/
  a : ℝ
  /-- y-coordinate of the circle's center -/
  b : ℝ
  /-- radius of the circle -/
  r : ℝ
  /-- The circle is tangent to the parabola at (a, a^2 + 1) and (-a, a^2 + 1) -/
  tangent_points : (a^2 + (a^2 + 1 - b)^2 = r^2) ∧ (a^2 + (a^2 + 1 - b)^2 = r^2)
  /-- The circle lies inside the parabola -/
  inside_parabola : ∀ x y, x^2 + (y - b)^2 = r^2 → y ≤ x^2 + 1

/-- The height difference between the center of the circle and the points of tangency is 1/2 -/
theorem tangent_circle_height_difference (c : TangentCircle) : 
  c.b - (c.a^2 + 1) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circle_height_difference_l864_86491


namespace NUMINAMATH_CALUDE_first_two_nonzero_digits_of_one_over_137_l864_86453

theorem first_two_nonzero_digits_of_one_over_137 :
  ∃ (a b : ℕ), a ≠ 0 ∧ b ≠ 0 ∧ (1 : ℚ) / 137 = (a * 10 + b : ℕ) / 1000 + r ∧ 0 ≤ r ∧ r < 1 / 100 ∧ a = 7 ∧ b = 6 := by
  sorry

end NUMINAMATH_CALUDE_first_two_nonzero_digits_of_one_over_137_l864_86453


namespace NUMINAMATH_CALUDE_at_most_one_perfect_square_l864_86446

def sequence_a : ℕ → ℤ
  | 0 => 0  -- arbitrary initial value
  | n + 1 => (sequence_a n)^3 + 103

theorem at_most_one_perfect_square :
  ∃ (n : ℕ), (∃ (k : ℤ), sequence_a n = k^2) →
    ∀ (m : ℕ), m ≠ n → ¬∃ (l : ℤ), sequence_a m = l^2 :=
by sorry

end NUMINAMATH_CALUDE_at_most_one_perfect_square_l864_86446


namespace NUMINAMATH_CALUDE_parabola_equation_with_sqrt3_distance_l864_86435

/-- Represents a parabola opening upwards -/
structure UprightParabola where
  /-- The distance from the focus to the directrix -/
  focus_directrix_distance : ℝ
  /-- Condition that the parabola opens upwards -/
  opens_upward : focus_directrix_distance > 0

/-- The standard equation of an upright parabola -/
def standard_equation (p : UprightParabola) : Prop :=
  ∀ x y : ℝ, x^2 = 2 * p.focus_directrix_distance * y

/-- Theorem stating the standard equation of a parabola with focus-directrix distance √3 -/
theorem parabola_equation_with_sqrt3_distance :
  ∀ (p : UprightParabola),
    p.focus_directrix_distance = Real.sqrt 3 →
    standard_equation p
    := by sorry

end NUMINAMATH_CALUDE_parabola_equation_with_sqrt3_distance_l864_86435


namespace NUMINAMATH_CALUDE_probability_blue_then_yellow_l864_86406

def blue_marbles : ℕ := 3
def yellow_marbles : ℕ := 4
def pink_marbles : ℕ := 9

def total_marbles : ℕ := blue_marbles + yellow_marbles + pink_marbles

theorem probability_blue_then_yellow :
  (blue_marbles : ℚ) / total_marbles * yellow_marbles / (total_marbles - 1) = 1 / 20 := by
  sorry

end NUMINAMATH_CALUDE_probability_blue_then_yellow_l864_86406


namespace NUMINAMATH_CALUDE_addilynns_broken_eggs_l864_86452

/-- The number of eggs in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of eggs Addilynn bought -/
def dozens_bought : ℕ := 6

/-- The number of eggs left on the shelf -/
def eggs_left : ℕ := 21

/-- The number of eggs Addilynn accidentally broke -/
def eggs_broken : ℕ := dozens_bought * dozen / 2 - eggs_left

theorem addilynns_broken_eggs :
  eggs_broken = 15 :=
sorry

end NUMINAMATH_CALUDE_addilynns_broken_eggs_l864_86452


namespace NUMINAMATH_CALUDE_prob_only_one_selected_l864_86431

/-- The probability of only one person being selected given individual and joint selection probabilities -/
theorem prob_only_one_selected
  (pH : ℚ) (pW : ℚ) (pHW : ℚ)
  (hpH : pH = 2 / 5)
  (hpW : pW = 3 / 7)
  (hpHW : pHW = 1 / 3) :
  pH * (1 - pW) + (1 - pH) * pW = 17 / 35 := by
sorry


end NUMINAMATH_CALUDE_prob_only_one_selected_l864_86431


namespace NUMINAMATH_CALUDE_cube_cutting_l864_86401

theorem cube_cutting (n : ℕ) : 
  (∃ s : ℕ, n > s ∧ n^3 - s^3 = 152) → n = 6 := by
sorry

end NUMINAMATH_CALUDE_cube_cutting_l864_86401


namespace NUMINAMATH_CALUDE_unique_negative_zero_implies_a_gt_two_l864_86433

/-- The function f(x) = ax³ - 3x² + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1

/-- The unique zero point of f(x) -/
noncomputable def x₀ (a : ℝ) : ℝ := sorry

theorem unique_negative_zero_implies_a_gt_two (a : ℝ) :
  (∃! x, f a x = 0) ∧ (x₀ a < 0) → a > 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_negative_zero_implies_a_gt_two_l864_86433


namespace NUMINAMATH_CALUDE_prime_relation_l864_86492

theorem prime_relation (P Q : ℕ) (hP : Nat.Prime P) (hQ : Nat.Prime Q)
  (h1 : P ∣ (Q^3 - 1)) (h2 : Q ∣ (P - 1)) : P = 1 + Q + Q^2 := by
  sorry

end NUMINAMATH_CALUDE_prime_relation_l864_86492


namespace NUMINAMATH_CALUDE_equality_condition_l864_86485

theorem equality_condition (a b c : ℝ) : 
  a = b + c + 2 → (a + b * c = (a + b) * (a + c) ↔ a = 0 ∨ a = 1) := by
  sorry

end NUMINAMATH_CALUDE_equality_condition_l864_86485


namespace NUMINAMATH_CALUDE_erica_safari_lions_erica_saw_three_lions_l864_86497

/-- Prove that Erica saw 3 lions on Saturday during her safari -/
theorem erica_safari_lions : ℕ → Prop := fun n =>
  let total_animals : ℕ := 20
  let saturday_elephants : ℕ := 2
  let sunday_animals : ℕ := 2 + 5  -- 2 buffaloes and 5 leopards
  let monday_animals : ℕ := 5 + 3  -- 5 rhinos and 3 warthogs
  n = total_animals - (saturday_elephants + sunday_animals + monday_animals)

/-- The number of lions Erica saw on Saturday is 3 -/
theorem erica_saw_three_lions : erica_safari_lions 3 := by
  sorry

end NUMINAMATH_CALUDE_erica_safari_lions_erica_saw_three_lions_l864_86497


namespace NUMINAMATH_CALUDE_complex_equation_real_part_l864_86421

theorem complex_equation_real_part (z : ℂ) (a b : ℝ) (h1 : z = a + b * Complex.I) 
  (h2 : b > 0) (h3 : z * (z + 2 * Complex.I) * (z - 2 * Complex.I) * (z + 5 * Complex.I) = 8000) :
  a^3 - 4*a = 8000 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_real_part_l864_86421


namespace NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l864_86447

theorem remainder_theorem (f : ℝ → ℝ) (a : ℝ) :
  (∃ q : ℝ → ℝ, f = λ x => (x - a) * q x + f a) :=
sorry

theorem polynomial_remainder (f : ℝ → ℝ) (h : f = λ x => x^8 + 3) :
  ∃ q : ℝ → ℝ, f = λ x => (x + 1) * q x + 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l864_86447


namespace NUMINAMATH_CALUDE_inverse_composition_l864_86402

/-- A function g with specific values --/
def g : Fin 5 → Fin 5
| 1 => 4
| 2 => 5
| 3 => 1
| 4 => 2
| 5 => 3

/-- The inverse of g --/
def g_inv : Fin 5 → Fin 5
| 1 => 3
| 2 => 4
| 3 => 5
| 4 => 1
| 5 => 2

/-- g is bijective --/
axiom g_bijective : Function.Bijective g

/-- g_inv is indeed the inverse of g --/
axiom g_inv_is_inverse : Function.LeftInverse g_inv g ∧ Function.RightInverse g_inv g

/-- The main theorem --/
theorem inverse_composition : g_inv (g_inv (g_inv 5)) = 1 := by sorry

end NUMINAMATH_CALUDE_inverse_composition_l864_86402


namespace NUMINAMATH_CALUDE_problem_statement_l864_86460

-- Define the set X
def X : Set ℕ := {n | 1 ≤ n ∧ n ≤ 2017}

-- Define the set S
def S : Set (ℕ × ℕ × ℕ) :=
  {t | t.1 ∈ X ∧ t.2.1 ∈ X ∧ t.2.2 ∈ X ∧
    ((t.1 < t.2.1 ∧ t.2.1 < t.2.2) ∨
     (t.2.1 < t.2.2 ∧ t.2.2 < t.1) ∨
     (t.2.2 < t.1 ∧ t.1 < t.2.1)) ∧
    ¬((t.1 < t.2.1 ∧ t.2.1 < t.2.2) ∧
      (t.2.1 < t.2.2 ∧ t.2.2 < t.1)) ∧
    ¬((t.1 < t.2.1 ∧ t.2.1 < t.2.2) ∧
      (t.2.2 < t.1 ∧ t.1 < t.2.1)) ∧
    ¬((t.2.1 < t.2.2 ∧ t.2.2 < t.1) ∧
      (t.2.2 < t.1 ∧ t.1 < t.2.1))}

-- Theorem statement
theorem problem_statement (x y z w : ℕ) 
  (h1 : (x, y, z) ∈ S) (h2 : (z, w, x) ∈ S) :
  (y, z, w) ∈ S ∧ (x, y, w) ∈ S := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l864_86460


namespace NUMINAMATH_CALUDE_inequality_proof_l864_86411

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (1 + a + b)) + (1 / (1 + b + c)) + (1 / (1 + c + a)) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l864_86411


namespace NUMINAMATH_CALUDE_point_movement_l864_86449

/-- A point in the 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Move a point up by a given number of units -/
def moveUp (p : Point2D) (units : ℝ) : Point2D :=
  { x := p.x, y := p.y + units }

/-- Move a point left by a given number of units -/
def moveLeft (p : Point2D) (units : ℝ) : Point2D :=
  { x := p.x - units, y := p.y }

theorem point_movement :
  let A : Point2D := { x := 1, y := -2 }
  let B : Point2D := moveLeft (moveUp A 3) 2
  B.x = -1 ∧ B.y = 1 := by sorry

end NUMINAMATH_CALUDE_point_movement_l864_86449


namespace NUMINAMATH_CALUDE_pyramid_layers_l864_86443

/-- Calculates the number of exterior golfballs for a given layer -/
def exteriorGolfballs (layer : ℕ) : ℕ :=
  if layer ≤ 2 then layer * layer else 4 * (layer - 1)

/-- Calculates the total number of exterior golfballs up to a given layer -/
def totalExteriorGolfballs (n : ℕ) : ℕ :=
  (List.range n).map exteriorGolfballs |>.sum

/-- Theorem stating that a pyramid with 145 exterior golfballs has 9 layers -/
theorem pyramid_layers (n : ℕ) : totalExteriorGolfballs n = 145 ↔ n = 9 := by
  sorry

#eval totalExteriorGolfballs 9  -- Should output 145

end NUMINAMATH_CALUDE_pyramid_layers_l864_86443


namespace NUMINAMATH_CALUDE_solution_satisfies_inequalities_l864_86418

theorem solution_satisfies_inequalities :
  ∀ (x y z : ℝ),
  x = 3 ∧ y = 4 ∧ z = 5 →
  x < y ∧ y < z ∧ z < 6 ∧
  1 / (y - x) + 1 / (z - y) ≤ 2 ∧
  1 / (6 - z) + 2 ≤ x :=
by sorry

end NUMINAMATH_CALUDE_solution_satisfies_inequalities_l864_86418


namespace NUMINAMATH_CALUDE_biking_distance_difference_l864_86437

/-- The distance biked by Daniela in four hours -/
def daniela_distance : ℤ := 75

/-- The distance biked by Carlos in four hours -/
def carlos_distance : ℤ := 60

/-- The distance biked by Emilio in four hours -/
def emilio_distance : ℤ := 45

/-- Theorem stating the difference between Daniela's distance and the sum of Carlos' and Emilio's distances -/
theorem biking_distance_difference : 
  daniela_distance - (carlos_distance + emilio_distance) = -30 := by
  sorry

end NUMINAMATH_CALUDE_biking_distance_difference_l864_86437


namespace NUMINAMATH_CALUDE_sara_golf_balls_l864_86455

/-- The number of items in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of golf balls Sara has -/
def sara_dozens : ℕ := 16

/-- The total number of golf balls Sara has -/
def sara_total : ℕ := sara_dozens * dozen

theorem sara_golf_balls : sara_total = 192 := by
  sorry

end NUMINAMATH_CALUDE_sara_golf_balls_l864_86455


namespace NUMINAMATH_CALUDE_cos_sin_inequality_solution_set_l864_86470

open Real

theorem cos_sin_inequality_solution_set (x : ℝ) : 
  (cos x)^4 - 2 * sin x * cos x - (sin x)^4 - 1 > 0 ↔ 
  ∃ k : ℤ, x ∈ Set.Ioo (k * π - π/4) (k * π) := by sorry

end NUMINAMATH_CALUDE_cos_sin_inequality_solution_set_l864_86470


namespace NUMINAMATH_CALUDE_alcohol_dilution_l864_86467

theorem alcohol_dilution (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hab : b < a) (hbc : c < b) :
  let initial_volume : ℝ := 1
  let first_dilution_volume : ℝ := a / b
  let second_dilution_volume : ℝ := a / (b + c)
  let total_water_used : ℝ := (first_dilution_volume - initial_volume) + (2 * second_dilution_volume - first_dilution_volume)
  total_water_used = 2 * a / (b + c) - 1 := by
sorry

end NUMINAMATH_CALUDE_alcohol_dilution_l864_86467


namespace NUMINAMATH_CALUDE_min_ratio_four_digit_number_l864_86479

/-- Represents a four-digit number -/
def FourDigitNumber := { n : ℕ // 1000 ≤ n ∧ n ≤ 9999 }

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- The theorem stating that 1099 minimizes x/y for four-digit numbers -/
theorem min_ratio_four_digit_number :
  ∀ (x : FourDigitNumber),
    (x.val : ℚ) / digit_sum x.val ≥ 1099 / digit_sum 1099 := by sorry

end NUMINAMATH_CALUDE_min_ratio_four_digit_number_l864_86479


namespace NUMINAMATH_CALUDE_three_from_nine_combination_l864_86481

theorem three_from_nine_combination : (Nat.choose 9 3) = 84 := by
  sorry

end NUMINAMATH_CALUDE_three_from_nine_combination_l864_86481


namespace NUMINAMATH_CALUDE_f_of_2_equals_5_l864_86499

def f (x : ℝ) : ℝ := x^2 + 2*x - 3

theorem f_of_2_equals_5 : f 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_f_of_2_equals_5_l864_86499


namespace NUMINAMATH_CALUDE_parallel_line_slope_l864_86408

/-- The slope of a line parallel to 3x - 6y = 12 is 1/2 -/
theorem parallel_line_slope (a b c : ℝ) : 
  (3 : ℝ) * a - (6 : ℝ) * b = (12 : ℝ) → 
  ∃ (m : ℝ), m = (1 : ℝ) / (2 : ℝ) ∧ 
  ∀ (x y : ℝ), (y = m * x + c) → 
  (∃ (k : ℝ), (3 : ℝ) * x - (6 : ℝ) * y = k) :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l864_86408


namespace NUMINAMATH_CALUDE_no_two_digit_divisible_by_reverse_l864_86490

theorem no_two_digit_divisible_by_reverse : ¬ ∃ (a b : ℕ), 
  1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ a ≠ b ∧ 
  ∃ (k : ℕ), k > 1 ∧ (10 * a + b) = k * (10 * b + a) :=
by sorry

end NUMINAMATH_CALUDE_no_two_digit_divisible_by_reverse_l864_86490


namespace NUMINAMATH_CALUDE_power_of_three_product_l864_86416

theorem power_of_three_product (x : ℕ) : 3^12 * 3^18 = x^6 → x = 243 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_product_l864_86416


namespace NUMINAMATH_CALUDE_patricia_books_l864_86454

def book_tournament (candice amanda kara patricia : ℕ) : Prop :=
  candice = 3 * amanda ∧
  kara = amanda / 2 ∧
  patricia = 7 * kara ∧
  candice = 18

theorem patricia_books :
  ∀ candice amanda kara patricia : ℕ,
    book_tournament candice amanda kara patricia →
    patricia = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_patricia_books_l864_86454


namespace NUMINAMATH_CALUDE_hawks_win_rate_theorem_l864_86466

/-- The minimum number of additional games needed for the Hawks to reach 90% win rate -/
def min_additional_games : ℕ := 25

/-- The initial number of games played -/
def initial_games : ℕ := 5

/-- The number of games initially won by the Hawks -/
def initial_hawks_wins : ℕ := 2

/-- The target win percentage as a fraction -/
def target_win_rate : ℚ := 9/10

theorem hawks_win_rate_theorem :
  ∀ n : ℕ, 
    (initial_hawks_wins + n : ℚ) / (initial_games + n) ≥ target_win_rate ↔ 
    n ≥ min_additional_games := by
  sorry

end NUMINAMATH_CALUDE_hawks_win_rate_theorem_l864_86466


namespace NUMINAMATH_CALUDE_selling_price_l864_86471

/-- Given an original price and a percentage increase, calculate the selling price -/
theorem selling_price (a : ℝ) : (a * (1 + 0.1)) = 1.1 * a := by sorry

end NUMINAMATH_CALUDE_selling_price_l864_86471


namespace NUMINAMATH_CALUDE_correct_sentence_structure_l864_86473

-- Define the possible options for filling the blanks
inductive BlankOption
  | indefiniteArticle : BlankOption
  | definiteArticle : BlankOption
  | empty : BlankOption

-- Define the sentence structure
structure Sentence where
  firstBlank : BlankOption
  secondBlank : BlankOption

-- Define the grammatical correctness of the sentence
def isGrammaticallyCorrect (s : Sentence) : Prop :=
  s.firstBlank = BlankOption.indefiniteArticle ∧ s.secondBlank = BlankOption.empty

-- Theorem: The sentence is grammatically correct when the first blank is "a" and the second is empty
theorem correct_sentence_structure :
  ∃ (s : Sentence), isGrammaticallyCorrect s :=
sorry

end NUMINAMATH_CALUDE_correct_sentence_structure_l864_86473


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l864_86459

theorem inscribed_circle_radius (d : ℝ) (h : d = Real.sqrt 12) : 
  let R := d / 2
  let side₁ := R * Real.sqrt 3
  let height := side₁ * (Real.sqrt 3 / 2)
  let side₂ := 2 * height / Real.sqrt 3
  let r := side₂ * Real.sqrt 3 / 6
  r = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l864_86459


namespace NUMINAMATH_CALUDE_b_work_time_l864_86484

-- Define the work completion time for A
def a_time : ℝ := 6

-- Define the total payment for A and B
def total_payment : ℝ := 3200

-- Define the time taken with C's help
def time_with_c : ℝ := 3

-- Define C's payment
def c_payment : ℝ := 400.0000000000002

-- Define B's work completion time (to be proved)
def b_time : ℝ := 8

-- Theorem statement
theorem b_work_time : 
  1 / a_time + 1 / b_time + (c_payment / total_payment) * (1 / time_with_c) = 1 / time_with_c :=
sorry

end NUMINAMATH_CALUDE_b_work_time_l864_86484


namespace NUMINAMATH_CALUDE_tight_sequence_x_range_l864_86434

/-- Definition of a tight sequence -/
def is_tight_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → (1/2 : ℝ) ≤ a (n+1) / a n ∧ a (n+1) / a n ≤ 2

/-- Theorem about the range of x in a specific tight sequence -/
theorem tight_sequence_x_range (a : ℕ → ℝ) (x : ℝ) 
  (h_tight : is_tight_sequence a)
  (h_a1 : a 1 = 1)
  (h_a2 : a 2 = 3/2)
  (h_a3 : a 3 = x)
  (h_a4 : a 4 = 4) :
  2 ≤ x ∧ x ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_tight_sequence_x_range_l864_86434


namespace NUMINAMATH_CALUDE_expression_equals_one_l864_86469

theorem expression_equals_one (a b c : ℝ) :
  ((a^2 - b^2)^2 + (b^2 - c^2)^2 + (c^2 - a^2)^2) / ((a - b)^2 + (b - c)^2 + (c - a)^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l864_86469


namespace NUMINAMATH_CALUDE_total_sample_volume_l864_86424

/-- The total sample volume -/
def M : ℝ := 50

/-- The frequency of the first group -/
def freq1 : ℝ := 10

/-- The frequency of the second group -/
def freq2 : ℝ := 0.35

/-- The frequency of the third group -/
def freq3 : ℝ := 0.45

/-- Theorem stating that M is the correct total sample volume given the frequencies -/
theorem total_sample_volume : M = freq1 + freq2 * M + freq3 * M := by
  sorry

end NUMINAMATH_CALUDE_total_sample_volume_l864_86424


namespace NUMINAMATH_CALUDE_parallelepiped_length_l864_86425

theorem parallelepiped_length : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (n - 2 > 0) ∧ 
  (n - 4 > 0) ∧
  ((n - 2) * (n - 4) * (n - 6) = (2 * n * (n - 2) * (n - 4)) / 3) ∧
  (n = 18) := by
sorry

end NUMINAMATH_CALUDE_parallelepiped_length_l864_86425


namespace NUMINAMATH_CALUDE_additional_cupcakes_count_l864_86427

-- Define the initial number of cupcakes
def initial_cupcakes : ℕ := 30

-- Define the number of cupcakes sold
def sold_cupcakes : ℕ := 9

-- Define the total number of cupcakes after making additional ones
def total_cupcakes : ℕ := 49

-- Theorem to prove
theorem additional_cupcakes_count :
  total_cupcakes - (initial_cupcakes - sold_cupcakes) = 28 :=
by sorry

end NUMINAMATH_CALUDE_additional_cupcakes_count_l864_86427


namespace NUMINAMATH_CALUDE_cube_diff_even_iff_sum_even_l864_86451

theorem cube_diff_even_iff_sum_even (p q : ℕ) :
  Even (p^3 - q^3) ↔ Even (p + q) := by sorry

end NUMINAMATH_CALUDE_cube_diff_even_iff_sum_even_l864_86451


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l864_86439

theorem smallest_integer_satisfying_inequality :
  ∀ x : ℤ, x < 3*x - 14 → x ≥ 8 ∧ 8 < 3*8 - 14 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l864_86439


namespace NUMINAMATH_CALUDE_janet_hires_four_warehouse_workers_l864_86465

/-- Represents the employment scenario for Janet's company --/
structure EmploymentScenario where
  total_employees : ℕ
  managers : ℕ
  warehouse_wage : ℝ
  manager_wage : ℝ
  fica_tax_rate : ℝ
  work_days : ℕ
  work_hours : ℕ
  total_cost : ℝ

/-- Calculates the number of warehouse workers in Janet's company --/
def calculate_warehouse_workers (scenario : EmploymentScenario) : ℕ :=
  scenario.total_employees - scenario.managers

/-- Theorem stating that Janet hires 4 warehouse workers --/
theorem janet_hires_four_warehouse_workers :
  let scenario : EmploymentScenario := {
    total_employees := 6,
    managers := 2,
    warehouse_wage := 15,
    manager_wage := 20,
    fica_tax_rate := 0.1,
    work_days := 25,
    work_hours := 8,
    total_cost := 22000
  }
  calculate_warehouse_workers scenario = 4 := by
  sorry


end NUMINAMATH_CALUDE_janet_hires_four_warehouse_workers_l864_86465


namespace NUMINAMATH_CALUDE_tan_sum_one_fortyfour_l864_86486

theorem tan_sum_one_fortyfour : (1 + Real.tan (1 * π / 180)) * (1 + Real.tan (44 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_one_fortyfour_l864_86486


namespace NUMINAMATH_CALUDE_tiles_needed_to_cover_floor_l864_86415

-- Define the dimensions of the floor and tiles
def floor_length : ℚ := 10
def floor_width : ℚ := 14
def tile_length : ℚ := 1/2  -- 6 inches in feet
def tile_width : ℚ := 2/3   -- 8 inches in feet

-- Theorem statement
theorem tiles_needed_to_cover_floor :
  (floor_length * floor_width) / (tile_length * tile_width) = 420 := by
  sorry

end NUMINAMATH_CALUDE_tiles_needed_to_cover_floor_l864_86415


namespace NUMINAMATH_CALUDE_truck_weight_l864_86457

/-- Given a truck and trailer with specified weight relationship, prove the truck's weight -/
theorem truck_weight (truck_weight trailer_weight : ℝ) : 
  truck_weight + trailer_weight = 7000 →
  trailer_weight = 0.5 * truck_weight - 200 →
  truck_weight = 4800 := by
sorry

end NUMINAMATH_CALUDE_truck_weight_l864_86457


namespace NUMINAMATH_CALUDE_gcd_multiple_relation_l864_86498

theorem gcd_multiple_relation (a b : ℕ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a = 7 * b) :
  Nat.gcd a b = b :=
by sorry

end NUMINAMATH_CALUDE_gcd_multiple_relation_l864_86498


namespace NUMINAMATH_CALUDE_two_digit_number_interchange_l864_86474

theorem two_digit_number_interchange (a b j : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : 10 * a + b = j * (a + b)) :
  10 * b + a = (10 * j - 9) * (a + b) :=
sorry

end NUMINAMATH_CALUDE_two_digit_number_interchange_l864_86474


namespace NUMINAMATH_CALUDE_sum_of_four_digit_numbers_l864_86400

/-- The set of digits used to form the numbers -/
def digits : Finset Nat := {1, 2, 3, 4, 5}

/-- A four-digit number formed from the given digits -/
structure FourDigitNumber where
  d₁ : Nat
  d₂ : Nat
  d₃ : Nat
  d₄ : Nat
  h₁ : d₁ ∈ digits
  h₂ : d₂ ∈ digits
  h₃ : d₃ ∈ digits
  h₄ : d₄ ∈ digits
  distinct : d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₁ ≠ d₄ ∧ d₂ ≠ d₃ ∧ d₂ ≠ d₄ ∧ d₃ ≠ d₄

/-- The value of a four-digit number -/
def FourDigitNumber.value (n : FourDigitNumber) : Nat :=
  1000 * n.d₁ + 100 * n.d₂ + 10 * n.d₃ + n.d₄

/-- The set of all valid four-digit numbers -/
def allFourDigitNumbers : Finset FourDigitNumber :=
  sorry

theorem sum_of_four_digit_numbers :
  (allFourDigitNumbers.sum FourDigitNumber.value) = 399960 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_digit_numbers_l864_86400


namespace NUMINAMATH_CALUDE_root_product_theorem_l864_86445

theorem root_product_theorem (c d m r s : ℝ) : 
  (c^2 - m*c + 3 = 0) →
  (d^2 - m*d + 3 = 0) →
  ((c + 1/d)^2 - r*(c + 1/d) + s = 0) →
  ((d + 1/c)^2 - r*(d + 1/c) + s = 0) →
  s = 16/3 := by
sorry

end NUMINAMATH_CALUDE_root_product_theorem_l864_86445


namespace NUMINAMATH_CALUDE_trains_meeting_time_l864_86477

/-- Two trains meeting problem -/
theorem trains_meeting_time (distance : ℝ) (express_speed : ℝ) (speed_difference : ℝ) : 
  distance = 390 →
  express_speed = 80 →
  speed_difference = 30 →
  (distance / (express_speed + (express_speed - speed_difference))) = 3 := by
  sorry

end NUMINAMATH_CALUDE_trains_meeting_time_l864_86477


namespace NUMINAMATH_CALUDE_curve_transformation_l864_86496

/-- The matrix A --/
def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, -2; 0, 1]

/-- The original curve C --/
def C (x y : ℝ) : Prop := (x - y)^2 + y^2 = 1

/-- The transformed curve C' --/
def C' (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Theorem stating that C' is the result of transforming C under A --/
theorem curve_transformation (x y : ℝ) : 
  C' x y ↔ ∃ x₀ y₀ : ℝ, C x₀ y₀ ∧ A.mulVec ![x₀, y₀] = ![x, y] :=
sorry

end NUMINAMATH_CALUDE_curve_transformation_l864_86496


namespace NUMINAMATH_CALUDE_triangle_area_l864_86483

/-- The area of a triangle with vertices at (2, 2), (7, 2), and (4, 9) is 17.5 square units. -/
theorem triangle_area : 
  let A : ℝ × ℝ := (2, 2)
  let B : ℝ × ℝ := (7, 2)
  let C : ℝ × ℝ := (4, 9)
  let base := |B.1 - A.1|
  let height := |C.2 - A.2|
  let area := (1/2) * base * height
  area = 17.5 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l864_86483


namespace NUMINAMATH_CALUDE_coefficients_of_given_equation_l864_86475

/-- Represents a quadratic equation in the form ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The given quadratic equation 2x^2 + x - 5 = 0 -/
def givenEquation : QuadraticEquation := ⟨2, 1, -5⟩

theorem coefficients_of_given_equation :
  givenEquation.a = 2 ∧ givenEquation.b = 1 ∧ givenEquation.c = -5 := by
  sorry

end NUMINAMATH_CALUDE_coefficients_of_given_equation_l864_86475


namespace NUMINAMATH_CALUDE_exists_multiple_of_E_l864_86495

def E (n : ℕ) : ℕ := Finset.prod (Finset.range n) (fun i => 2 * (i + 1))

def D (n : ℕ) : ℕ := Finset.prod (Finset.range n) (fun i => 2 * i + 1)

theorem exists_multiple_of_E (n : ℕ) : ∃ m : ℕ, ∃ k : ℕ, D n * 2^m = k * E n := by
  sorry

end NUMINAMATH_CALUDE_exists_multiple_of_E_l864_86495


namespace NUMINAMATH_CALUDE_derivative_at_one_l864_86409

def f (x : ℝ) : ℝ := (x - 2)^2

theorem derivative_at_one :
  deriv f 1 = -2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_one_l864_86409


namespace NUMINAMATH_CALUDE_a_upper_bound_l864_86404

theorem a_upper_bound (a : ℝ) : 
  (∀ x y : ℝ, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 1 2 → x * y = 2 → 
    2 - x ≥ a / (4 - y)) → 
  a ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_a_upper_bound_l864_86404


namespace NUMINAMATH_CALUDE_third_smallest_prime_squared_cubed_l864_86414

-- Define a function to get the nth smallest prime number
def nthSmallestPrime (n : ℕ) : ℕ := sorry

-- State the theorem
theorem third_smallest_prime_squared_cubed :
  (nthSmallestPrime 3) ^ 2 ^ 3 = 15625 := by sorry

end NUMINAMATH_CALUDE_third_smallest_prime_squared_cubed_l864_86414


namespace NUMINAMATH_CALUDE_johns_journey_distance_l864_86423

theorem johns_journey_distance :
  let total_distance : ℚ := 360 / 7
  let highway_distance : ℚ := total_distance / 4
  let city_distance : ℚ := 30
  let country_distance : ℚ := total_distance / 6
  highway_distance + city_distance + country_distance = total_distance := by
  sorry

end NUMINAMATH_CALUDE_johns_journey_distance_l864_86423


namespace NUMINAMATH_CALUDE_value_calculation_l864_86478

theorem value_calculation (N : ℝ) (h : 0.4 * N = 300) : (1/4) * (1/3) * (2/5) * N = 25 := by
  sorry

end NUMINAMATH_CALUDE_value_calculation_l864_86478


namespace NUMINAMATH_CALUDE_production_order_machines_correct_initial_machines_l864_86442

/-- The number of machines initially used to complete a production order -/
def initial_machines : ℕ := 3

/-- The time (in hours) to complete the order with the initial number of machines -/
def initial_time : ℕ := 44

/-- The time (in hours) to complete the order with one additional machine -/
def reduced_time : ℕ := 33

/-- The production rate of a single machine (assumed to be constant) -/
def machine_rate : ℚ := 1 / initial_machines / initial_time

theorem production_order_machines :
  (initial_machines * machine_rate * initial_time : ℚ) =
  ((initial_machines + 1) * machine_rate * reduced_time : ℚ) :=
sorry

theorem correct_initial_machines :
  initial_machines = 3 :=
sorry

end NUMINAMATH_CALUDE_production_order_machines_correct_initial_machines_l864_86442


namespace NUMINAMATH_CALUDE_angle_y_is_90_l864_86461

-- Define the angles
def angle_ABC : ℝ := 120
def angle_ABE : ℝ := 30

-- Define the theorem
theorem angle_y_is_90 :
  ∀ (angle_y angle_ABD : ℝ),
  -- Condition 3
  angle_ABD + angle_ABC = 180 →
  -- Condition 4
  angle_ABE + angle_y = 180 →
  -- Condition 5 (using angle_y instead of explicitly stating the third angle)
  angle_y + angle_ABD + angle_ABE = 180 →
  -- Conclusion
  angle_y = 90 := by
sorry

end NUMINAMATH_CALUDE_angle_y_is_90_l864_86461


namespace NUMINAMATH_CALUDE_coins_missing_fraction_l864_86412

theorem coins_missing_fraction (initial_coins : ℚ) : 
  initial_coins > 0 →
  let lost_coins := (1 / 3 : ℚ) * initial_coins
  let found_coins := (2 / 3 : ℚ) * lost_coins
  let remaining_coins := initial_coins - lost_coins + found_coins
  (initial_coins - remaining_coins) / initial_coins = (1 / 9 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_coins_missing_fraction_l864_86412


namespace NUMINAMATH_CALUDE_clock_hands_coincidence_coincidence_time_in_hours_and_minutes_l864_86450

/-- The time in minutes when the hour and minute hands of a clock coincide after midnight -/
def coincidence_time : ℚ :=
  720 / 11

theorem clock_hands_coincidence :
  let minute_speed : ℚ := 360 / 60  -- degrees per minute
  let hour_speed : ℚ := 360 / 720   -- degrees per minute
  ∀ t : ℚ,
    t > 0 →
    t < coincidence_time →
    minute_speed * t ≠ hour_speed * t + 360 * (t / 720).floor →
    minute_speed * coincidence_time = hour_speed * coincidence_time + 360 :=
by sorry

theorem coincidence_time_in_hours_and_minutes :
  (coincidence_time / 60).floor = 1 ∧
  (coincidence_time % 60 : ℚ) = 65 / 11 :=
by sorry

end NUMINAMATH_CALUDE_clock_hands_coincidence_coincidence_time_in_hours_and_minutes_l864_86450


namespace NUMINAMATH_CALUDE_quadratic_solution_existence_l864_86458

theorem quadratic_solution_existence (a b c : ℝ) (f : ℝ → ℝ) 
  (hf : f = fun x ↦ a * x^2 + b * x + c)
  (h1 : f 3.11 < 0)
  (h2 : f 3.12 > 0) :
  ∃ x : ℝ, f x = 0 ∧ 3.11 < x ∧ x < 3.12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_existence_l864_86458


namespace NUMINAMATH_CALUDE_profit_distribution_correct_l864_86476

def total_profit : ℕ := 280000

def shekhar_percentage : ℚ := 28 / 100
def rajeev_percentage : ℚ := 22 / 100
def jatin_percentage : ℚ := 20 / 100
def simran_percentage : ℚ := 18 / 100
def ramesh_percentage : ℚ := 12 / 100

def shekhar_share : ℕ := (shekhar_percentage * total_profit).num.toNat
def rajeev_share : ℕ := (rajeev_percentage * total_profit).num.toNat
def jatin_share : ℕ := (jatin_percentage * total_profit).num.toNat
def simran_share : ℕ := (simran_percentage * total_profit).num.toNat
def ramesh_share : ℕ := (ramesh_percentage * total_profit).num.toNat

theorem profit_distribution_correct :
  shekhar_share + rajeev_share + jatin_share + simran_share + ramesh_share = total_profit :=
by sorry

end NUMINAMATH_CALUDE_profit_distribution_correct_l864_86476


namespace NUMINAMATH_CALUDE_symmetric_point_x_axis_l864_86463

def point_symmetric_to_x_axis (x y : ℝ) : ℝ × ℝ := (x, -y)

theorem symmetric_point_x_axis :
  let M : ℝ × ℝ := (1, 2)
  point_symmetric_to_x_axis M.1 M.2 = (1, -2) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_x_axis_l864_86463


namespace NUMINAMATH_CALUDE_total_students_l864_86419

theorem total_students (middle_school : ℕ) (elementary_school : ℕ) : 
  middle_school = 50 →
  elementary_school = 4 * middle_school - 3 →
  middle_school + elementary_school = 247 := by
sorry

end NUMINAMATH_CALUDE_total_students_l864_86419


namespace NUMINAMATH_CALUDE_red_notes_per_row_l864_86413

theorem red_notes_per_row (total_rows : ℕ) (total_notes : ℕ) (extra_blue : ℕ) :
  total_rows = 5 →
  total_notes = 100 →
  extra_blue = 10 →
  ∃ (red_per_row : ℕ),
    red_per_row = 6 ∧
    total_notes = total_rows * red_per_row + 2 * (total_rows * red_per_row) + extra_blue :=
by sorry

end NUMINAMATH_CALUDE_red_notes_per_row_l864_86413


namespace NUMINAMATH_CALUDE_inequality_proof_l864_86429

theorem inequality_proof (x y z : ℝ) 
  (hpos : x > 0 ∧ y > 0 ∧ z > 0) 
  (hsum : x + y + z = 3) : 
  2 * Real.sqrt (x + Real.sqrt y) + 2 * Real.sqrt (y + Real.sqrt z) + 2 * Real.sqrt (z + Real.sqrt x) 
  ≤ Real.sqrt (8 + x - y) + Real.sqrt (8 + y - z) + Real.sqrt (8 + z - x) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l864_86429


namespace NUMINAMATH_CALUDE_stimulus_check_distribution_l864_86488

theorem stimulus_check_distribution (total amount_to_wife amount_to_first_son amount_to_second_son savings : ℚ) :
  total = 2000 ∧
  amount_to_wife = (2 / 5) * total ∧
  amount_to_first_son = (2 / 5) * (total - amount_to_wife) ∧
  savings = 432 ∧
  amount_to_second_son = total - amount_to_wife - amount_to_first_son - savings →
  amount_to_second_son / (total - amount_to_wife - amount_to_first_son) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_stimulus_check_distribution_l864_86488


namespace NUMINAMATH_CALUDE_pelican_migration_l864_86472

/-- Represents the number of Pelicans originally in Shark Bite Cove -/
def original_pelicans : ℕ := 30

/-- Represents the number of sharks in Pelican Bay -/
def sharks_in_pelican_bay : ℕ := 60

/-- Represents the number of Pelicans remaining in Shark Bite Cove -/
def remaining_pelicans : ℕ := 20

/-- The fraction of Pelicans that moved from Shark Bite Cove to Pelican Bay -/
def fraction_moved : ℚ := 1 / 3

theorem pelican_migration :
  (sharks_in_pelican_bay = 2 * original_pelicans) ∧
  (remaining_pelicans < original_pelicans) ∧
  (fraction_moved = (original_pelicans - remaining_pelicans : ℚ) / original_pelicans) :=
by sorry

end NUMINAMATH_CALUDE_pelican_migration_l864_86472


namespace NUMINAMATH_CALUDE_power_of_81_l864_86462

theorem power_of_81 : (81 : ℝ) ^ (5/2) = 59049 := by
  sorry

end NUMINAMATH_CALUDE_power_of_81_l864_86462


namespace NUMINAMATH_CALUDE_monster_family_eyes_l864_86441

/-- Represents the number of eyes for each family member -/
structure MonsterEyes where
  mom : Nat
  dad : Nat
  child : Nat
  num_children : Nat

/-- Calculates the total number of eyes in the monster family -/
def total_eyes (m : MonsterEyes) : Nat :=
  m.mom + m.dad + m.child * m.num_children

/-- Theorem stating that the total number of eyes in the given monster family is 16 -/
theorem monster_family_eyes :
  ∃ m : MonsterEyes, m.mom = 1 ∧ m.dad = 3 ∧ m.child = 4 ∧ m.num_children = 3 ∧ total_eyes m = 16 := by
  sorry

end NUMINAMATH_CALUDE_monster_family_eyes_l864_86441


namespace NUMINAMATH_CALUDE_same_sign_as_B_l864_86438

/-- A line in 2D space defined by the equation Ax + By + C = 0 -/
structure Line2D where
  A : ℝ
  B : ℝ
  C : ℝ
  A_nonzero : A ≠ 0
  B_nonzero : B ≠ 0

/-- Determines if a point (x, y) is above a given line -/
def IsAboveLine (l : Line2D) (x y : ℝ) : Prop :=
  l.A * x + l.B * y + l.C > 0

/-- The main theorem stating that for a point above the line, 
    Ax + By + C has the same sign as B -/
theorem same_sign_as_B (l : Line2D) (x y : ℝ) 
    (h : IsAboveLine l x y) : 
    (l.A * x + l.B * y + l.C) * l.B > 0 := by
  sorry

end NUMINAMATH_CALUDE_same_sign_as_B_l864_86438


namespace NUMINAMATH_CALUDE_common_divisors_9240_8820_l864_86403

theorem common_divisors_9240_8820 : 
  (Nat.divisors (Nat.gcd 9240 8820)).card = 24 := by sorry

end NUMINAMATH_CALUDE_common_divisors_9240_8820_l864_86403


namespace NUMINAMATH_CALUDE_num_tough_weeks_is_three_l864_86417

def tough_week_sales : ℕ := 800
def good_week_sales : ℕ := 2 * tough_week_sales
def num_good_weeks : ℕ := 5
def total_sales : ℕ := 10400

theorem num_tough_weeks_is_three :
  ∃ (num_tough_weeks : ℕ),
    num_tough_weeks * tough_week_sales + num_good_weeks * good_week_sales = total_sales ∧
    num_tough_weeks = 3 :=
by sorry

end NUMINAMATH_CALUDE_num_tough_weeks_is_three_l864_86417


namespace NUMINAMATH_CALUDE_zoe_coloring_books_l864_86407

/-- Given two coloring books with the same number of pictures and the number of pictures left to color,
    calculate the number of pictures colored. -/
def pictures_colored (pictures_per_book : ℕ) (books : ℕ) (pictures_left : ℕ) : ℕ :=
  pictures_per_book * books - pictures_left

/-- Theorem stating that given two coloring books with 44 pictures each and 68 pictures left to color,
    the number of pictures colored is 20. -/
theorem zoe_coloring_books : pictures_colored 44 2 68 = 20 := by
  sorry

end NUMINAMATH_CALUDE_zoe_coloring_books_l864_86407


namespace NUMINAMATH_CALUDE_shifted_quadratic_function_l864_86468

/-- A quadratic function -/
def f (x : ℝ) : ℝ := -x^2

/-- Horizontal shift of a function -/
def horizontalShift (f : ℝ → ℝ) (h : ℝ) : ℝ → ℝ := fun x ↦ f (x + h)

/-- Vertical shift of a function -/
def verticalShift (f : ℝ → ℝ) (v : ℝ) : ℝ → ℝ := fun x ↦ f x + v

/-- The shifted function -/
def g : ℝ → ℝ := verticalShift (horizontalShift f 1) 3

theorem shifted_quadratic_function :
  ∀ x : ℝ, g x = -(x + 1)^2 + 3 :=
by sorry

end NUMINAMATH_CALUDE_shifted_quadratic_function_l864_86468


namespace NUMINAMATH_CALUDE_even_sum_implies_one_even_l864_86456

theorem even_sum_implies_one_even (a b c : ℕ) :
  Even (a + b + c) →
  ¬((Odd a ∧ Odd b ∧ Odd c) ∨ 
    (Even a ∧ Even b) ∨ (Even a ∧ Even c) ∨ (Even b ∧ Even c)) :=
by sorry

end NUMINAMATH_CALUDE_even_sum_implies_one_even_l864_86456


namespace NUMINAMATH_CALUDE_complement_of_intersection_l864_86410

open Set

theorem complement_of_intersection (A B : Set ℝ) : 
  A = {x : ℝ | x ≤ 1} → 
  B = {x : ℝ | x < 2} → 
  (A ∩ B)ᶜ = {x : ℝ | x > 1} := by
sorry

end NUMINAMATH_CALUDE_complement_of_intersection_l864_86410


namespace NUMINAMATH_CALUDE_scientific_notation_425000_l864_86448

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_425000 :
  toScientificNotation 425000 = ScientificNotation.mk 4.25 5 sorry := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_425000_l864_86448


namespace NUMINAMATH_CALUDE_opposite_of_one_over_twentythree_l864_86464

theorem opposite_of_one_over_twentythree :
  ∀ x : ℚ, x = 1 / 23 → -x = -(1 / 23) := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_one_over_twentythree_l864_86464


namespace NUMINAMATH_CALUDE_inequality_not_always_true_l864_86428

theorem inequality_not_always_true (a b : ℝ) (h : a > b) :
  ¬ ∀ c : ℝ, a * c > b * c :=
by sorry

end NUMINAMATH_CALUDE_inequality_not_always_true_l864_86428


namespace NUMINAMATH_CALUDE_cube_difference_l864_86440

theorem cube_difference (x y : ℝ) (h1 : x - y = 3) (h2 : x^2 + y^2 = 27) :
  x^3 - y^3 = 108 := by sorry

end NUMINAMATH_CALUDE_cube_difference_l864_86440


namespace NUMINAMATH_CALUDE_space_diagonals_of_specific_polyhedron_l864_86444

/-- A convex polyhedron with specified properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  pentagonal_faces : ℕ

/-- Calculate the number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  sorry

/-- Theorem stating the number of space diagonals in the specific polyhedron Q -/
theorem space_diagonals_of_specific_polyhedron :
  ∃ Q : ConvexPolyhedron,
    Q.vertices = 30 ∧
    Q.edges = 70 ∧
    Q.faces = 40 ∧
    Q.triangular_faces = 30 ∧
    Q.pentagonal_faces = 10 ∧
    space_diagonals Q = 315 :=
  sorry

end NUMINAMATH_CALUDE_space_diagonals_of_specific_polyhedron_l864_86444
