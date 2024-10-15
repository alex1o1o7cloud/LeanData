import Mathlib

namespace NUMINAMATH_CALUDE_twenty_loaves_slices_thirty_loaves_not_enough_l3440_344078

-- Define the number of slices per loaf
def slices_per_loaf : ℕ := 12

-- Define the function to calculate total slices
def total_slices (loaves : ℕ) : ℕ := slices_per_loaf * loaves

-- Theorem 1
theorem twenty_loaves_slices : total_slices 20 = 240 := by sorry

-- Theorem 2
theorem thirty_loaves_not_enough (children : ℕ) (h : children = 385) : 
  total_slices 30 < children := by sorry

end NUMINAMATH_CALUDE_twenty_loaves_slices_thirty_loaves_not_enough_l3440_344078


namespace NUMINAMATH_CALUDE_simplify_expression_l3440_344006

theorem simplify_expression (a b c d x : ℝ) 
  (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d) 
  (hbc : b ≠ c) (hbd : b ≠ d) (hcd : c ≠ d) :
  ((x + a)^4) / ((a - b)*(a - c)*(a - d)) + 
  ((x + b)^4) / ((b - a)*(b - c)*(b - d)) + 
  ((x + c)^4) / ((c - a)*(c - b)*(c - d)) + 
  ((x + d)^4) / ((d - a)*(d - b)*(d - c)) = 
  a + b + c + d + 4*x := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l3440_344006


namespace NUMINAMATH_CALUDE_chosen_number_proof_l3440_344018

theorem chosen_number_proof (x : ℝ) : (x / 6) - 189 = 3 → x = 1152 := by
  sorry

end NUMINAMATH_CALUDE_chosen_number_proof_l3440_344018


namespace NUMINAMATH_CALUDE_optimal_strategy_l3440_344043

/-- Represents the expected score when answering question A first -/
def E_xi (P1 P2 a b : ℝ) : ℝ := a * P1 * (1 - P2) + (a + b) * P1 * P2

/-- Represents the expected score when answering question B first -/
def E_epsilon (P1 P2 a b : ℝ) : ℝ := b * P2 * (1 - P1) + (a + b) * P1 * P2

/-- The theorem states that given P1 = 2/5, a = 10, b = 20, 
    choosing to answer question A first is optimal when 0 ≤ P2 ≤ 1/4 -/
theorem optimal_strategy (P2 : ℝ) :
  0 ≤ P2 ∧ P2 ≤ 1/4 ↔ E_xi (2/5) P2 10 20 ≥ E_epsilon (2/5) P2 10 20 := by
  sorry

end NUMINAMATH_CALUDE_optimal_strategy_l3440_344043


namespace NUMINAMATH_CALUDE_distance_before_stop_correct_concert_drive_distance_l3440_344019

/-- Calculates the distance driven before stopping for gas --/
def distance_before_stop (total_distance : ℕ) (remaining_distance : ℕ) : ℕ :=
  total_distance - remaining_distance

/-- Theorem: The distance driven before stopping for gas is equal to 
    the total distance minus the remaining distance --/
theorem distance_before_stop_correct (total_distance : ℕ) (remaining_distance : ℕ) 
    (h : remaining_distance ≤ total_distance) :
  distance_before_stop total_distance remaining_distance = 
    total_distance - remaining_distance := by
  sorry

/-- Given the total distance and remaining distance, 
    prove that the distance driven before stopping is 32 miles --/
theorem concert_drive_distance :
  distance_before_stop 78 46 = 32 := by
  sorry

end NUMINAMATH_CALUDE_distance_before_stop_correct_concert_drive_distance_l3440_344019


namespace NUMINAMATH_CALUDE_games_for_champion_l3440_344038

/-- Represents a single-elimination tournament -/
structure Tournament where
  num_players : ℕ
  single_elimination : Bool

/-- The number of games required to determine the champion in a single-elimination tournament -/
def games_required (t : Tournament) : ℕ :=
  t.num_players - 1

theorem games_for_champion (t : Tournament) (h1 : t.single_elimination = true) (h2 : t.num_players = 512) :
  games_required t = 511 := by
  sorry

end NUMINAMATH_CALUDE_games_for_champion_l3440_344038


namespace NUMINAMATH_CALUDE_product_inequality_l3440_344039

theorem product_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a + b + c + a * b * c = 4) : 
  (1 + a / b + c * a) * (1 + b / c + a * b) * (1 + c / a + b * c) ≥ 27 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l3440_344039


namespace NUMINAMATH_CALUDE_plane_q_satisfies_conditions_l3440_344014

/-- Plane type representing ax + by + cz + d = 0 --/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Line type representing the intersection of two planes --/
structure Line where
  p1 : Plane
  p2 : Plane

/-- Point type in 3D space --/
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Function to check if a plane contains a line --/
def containsLine (p : Plane) (l : Line) : Prop := sorry

/-- Function to calculate the distance between a plane and a point --/
def distancePlanePoint (p : Plane) (pt : Point) : ℝ := sorry

/-- Given planes --/
def plane1 : Plane := ⟨1, 3, 2, -4⟩
def plane2 : Plane := ⟨2, -1, 3, -6⟩

/-- Line M --/
def lineM : Line := ⟨plane1, plane2⟩

/-- Given point --/
def givenPoint : Point := ⟨4, 2, -2⟩

/-- Plane Q --/
def planeQ : Plane := ⟨1, -9, 5, -2⟩

theorem plane_q_satisfies_conditions :
  containsLine planeQ lineM ∧
  planeQ ≠ plane1 ∧
  planeQ ≠ plane2 ∧
  distancePlanePoint planeQ givenPoint = 3 / Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_plane_q_satisfies_conditions_l3440_344014


namespace NUMINAMATH_CALUDE_range_of_negative_power_function_l3440_344032

open Set
open Function
open Real

theorem range_of_negative_power_function {m : ℝ} (hm : m < 0) :
  let g : ℝ → ℝ := fun x ↦ x ^ m
  range (g ∘ (fun x ↦ x) : Set.Ioo 0 1 → ℝ) = Set.Ioi 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_negative_power_function_l3440_344032


namespace NUMINAMATH_CALUDE_pencil_count_l3440_344067

/-- The total number of pencils in the drawer after Sarah's addition -/
def total_pencils (initial : ℕ) (mike_added : ℕ) (sarah_added : ℕ) : ℕ :=
  initial + mike_added + sarah_added

/-- Theorem stating the total number of pencils after all additions -/
theorem pencil_count (x : ℕ) :
  total_pencils 41 30 x = 71 + x := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l3440_344067


namespace NUMINAMATH_CALUDE_unique_prime_pair_l3440_344099

theorem unique_prime_pair : ∃! (p q : ℕ), 
  Prime p ∧ Prime q ∧ 
  ∃ r : ℕ, Prime r ∧ 
  (1 : ℚ) + (p^q - q^p : ℚ) / (p + q : ℚ) = r := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_pair_l3440_344099


namespace NUMINAMATH_CALUDE_value_of_S_l3440_344047

theorem value_of_S : ∀ S : ℕ, 
  S = 6 * 10000 + 5 * 1000 + 4 * 10 + 3 * 1 → S = 65043 := by
  sorry

end NUMINAMATH_CALUDE_value_of_S_l3440_344047


namespace NUMINAMATH_CALUDE_total_cantaloupes_is_65_l3440_344009

/-- The number of cantaloupes grown by Keith -/
def keith_cantaloupes : ℕ := 29

/-- The number of cantaloupes grown by Fred -/
def fred_cantaloupes : ℕ := 16

/-- The number of cantaloupes grown by Jason -/
def jason_cantaloupes : ℕ := 20

/-- The total number of cantaloupes grown by Keith, Fred, and Jason -/
def total_cantaloupes : ℕ := keith_cantaloupes + fred_cantaloupes + jason_cantaloupes

theorem total_cantaloupes_is_65 : total_cantaloupes = 65 := by
  sorry

end NUMINAMATH_CALUDE_total_cantaloupes_is_65_l3440_344009


namespace NUMINAMATH_CALUDE_bill_calculation_l3440_344020

def restaurant_bill (num_friends : ℕ) (extra_payment : ℕ) : Prop :=
  num_friends > 0 ∧ 
  ∃ (total_bill : ℕ), 
    total_bill = num_friends * (total_bill / num_friends + extra_payment * (num_friends - 1) / num_friends)

theorem bill_calculation :
  restaurant_bill 6 3 → ∃ (total_bill : ℕ), total_bill = 90 :=
by sorry

end NUMINAMATH_CALUDE_bill_calculation_l3440_344020


namespace NUMINAMATH_CALUDE_min_sum_of_fractions_l3440_344055

def Digits : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem min_sum_of_fractions (A B C D E : Nat) 
  (h1 : A ∈ Digits) (h2 : B ∈ Digits) (h3 : C ∈ Digits) (h4 : D ∈ Digits) (h5 : E ∈ Digits)
  (h6 : A ≠ B) (h7 : A ≠ C) (h8 : A ≠ D) (h9 : A ≠ E)
  (h10 : B ≠ C) (h11 : B ≠ D) (h12 : B ≠ E)
  (h13 : C ≠ D) (h14 : C ≠ E)
  (h15 : D ≠ E) :
  (A : ℚ) / B + (C : ℚ) / D + (E : ℚ) / 9 ≥ 125 / 168 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_fractions_l3440_344055


namespace NUMINAMATH_CALUDE_cubic_polynomial_problem_l3440_344007

/-- Given a cubic equation and conditions on a polynomial P,
    prove that P has a specific form. -/
theorem cubic_polynomial_problem (a b c : ℝ) (P : ℝ → ℝ) :
  (a^3 + 5*a^2 + 8*a + 13 = 0) →
  (b^3 + 5*b^2 + 8*b + 13 = 0) →
  (c^3 + 5*c^2 + 8*c + 13 = 0) →
  (∀ x, ∃ p q r s, P x = p*x^3 + q*x^2 + r*x + s) →
  (P a = b + c + 2) →
  (P b = a + c + 2) →
  (P c = a + b + 2) →
  (P (a + b + c) = -22) →
  (∀ x, P x = (19*x^3 + 95*x^2 + 152*x + 247) / 52 - x - 3) :=
by sorry


end NUMINAMATH_CALUDE_cubic_polynomial_problem_l3440_344007


namespace NUMINAMATH_CALUDE_dice_probability_l3440_344095

def num_dice : ℕ := 15
def num_ones : ℕ := 3
def prob_one : ℚ := 1/6
def prob_not_one : ℚ := 5/6

theorem dice_probability : 
  (Nat.choose num_dice num_ones : ℚ) * prob_one ^ num_ones * prob_not_one ^ (num_dice - num_ones) = 
  455 * (1/6)^3 * (5/6)^12 := by sorry

end NUMINAMATH_CALUDE_dice_probability_l3440_344095


namespace NUMINAMATH_CALUDE_sector_angle_l3440_344075

theorem sector_angle (area : Real) (radius : Real) (h1 : area = 3 * Real.pi / 16) (h2 : radius = 1) :
  (2 * area) / (radius ^ 2) = 3 * Real.pi / 8 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_l3440_344075


namespace NUMINAMATH_CALUDE_sphere_area_ratio_l3440_344069

/-- The area of a region on a sphere is proportional to the square of its radius -/
axiom area_proportional_to_radius_squared {r₁ r₂ A₁ A₂ : ℝ} (h : r₁ > 0 ∧ r₂ > 0) :
  A₂ / A₁ = (r₂ / r₁) ^ 2

/-- Given two concentric spheres with radii 4 cm and 6 cm, if a region on the smaller sphere
    has an area of 37 square cm, then the corresponding region on the larger sphere
    has an area of 83.25 square cm -/
theorem sphere_area_ratio (r₁ r₂ A₁ : ℝ) (hr₁ : r₁ = 4) (hr₂ : r₂ = 6) (hA₁ : A₁ = 37) :
  ∃ A₂ : ℝ, A₂ = 83.25 ∧ A₂ / A₁ = (r₂ / r₁) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_area_ratio_l3440_344069


namespace NUMINAMATH_CALUDE_unique_successful_arrangement_l3440_344024

/-- Represents a cell in the table -/
inductive Cell
| One
| NegOne

/-- Represents a square table -/
def Table (n : ℕ) := Fin (2^n - 1) → Fin (2^n - 1) → Cell

/-- Checks if two cells are neighbors -/
def is_neighbor (n : ℕ) (i j i' j' : Fin (2^n - 1)) : Prop :=
  (i = i' ∧ (j.val + 1 = j'.val ∨ j.val = j'.val + 1)) ∨
  (j = j' ∧ (i.val + 1 = i'.val ∨ i.val = i'.val + 1))

/-- Checks if a table is a successful arrangement -/
def is_successful (n : ℕ) (t : Table n) : Prop :=
  ∀ i j, t i j = Cell.One ↔ 
    ∀ i' j', is_neighbor n i j i' j' → t i' j' = Cell.One

/-- The main theorem -/
theorem unique_successful_arrangement (n : ℕ) :
  ∃! t : Table n, is_successful n t ∧ (∀ i j, t i j = Cell.One) :=
sorry

end NUMINAMATH_CALUDE_unique_successful_arrangement_l3440_344024


namespace NUMINAMATH_CALUDE_money_distribution_l3440_344046

theorem money_distribution (total : ℝ) (share_d : ℝ) :
  let proportion_sum := 5 + 2 + 4 + 3
  let proportion_d := 3
  let proportion_c := 4
  share_d = 1500 →
  share_d = (proportion_d / proportion_sum) * total →
  let share_c := (proportion_c / proportion_sum) * total
  share_c - share_d = 500 :=
by sorry

end NUMINAMATH_CALUDE_money_distribution_l3440_344046


namespace NUMINAMATH_CALUDE_units_digit_of_fraction_l3440_344008

def numerator : ℕ := 30 * 32 * 34 * 36 * 38 * 40
def denominator : ℕ := 2000

theorem units_digit_of_fraction (n d : ℕ) (h : d ≠ 0) :
  (n / d) % 10 = 2 :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_fraction_l3440_344008


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l3440_344028

/-- Proves that a train with given length and speed takes a specific time to cross a bridge of given length -/
theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (bridge_length : ℝ) : 
  train_length = 180 ∧ 
  train_speed_kmh = 72 ∧ 
  bridge_length = 270 → 
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l3440_344028


namespace NUMINAMATH_CALUDE_johns_class_boys_count_l3440_344063

theorem johns_class_boys_count :
  ∀ (g b : ℕ),
  g + b = 28 →
  g = (3 * b) / 4 →
  b = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_johns_class_boys_count_l3440_344063


namespace NUMINAMATH_CALUDE_melissa_games_l3440_344096

/-- The number of games Melissa played -/
def num_games : ℕ := 91 / 7

/-- The total points Melissa scored -/
def total_points : ℕ := 91

/-- The points Melissa scored per game -/
def points_per_game : ℕ := 7

theorem melissa_games : num_games = 13 := by
  sorry

end NUMINAMATH_CALUDE_melissa_games_l3440_344096


namespace NUMINAMATH_CALUDE_local_minimum_implies_a_equals_negative_three_l3440_344021

/-- The function f(x) defined as x(x-a)² --/
def f (a : ℝ) (x : ℝ) : ℝ := x * (x - a)^2

/-- The first derivative of f(x) --/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*a*x + a^2

/-- The second derivative of f(x) --/
def f_second_derivative (a : ℝ) (x : ℝ) : ℝ := 6*x - 4*a

theorem local_minimum_implies_a_equals_negative_three (a : ℝ) :
  (f_derivative a (-1) = 0) ∧ 
  (∀ ε > 0, ∃ δ > 0, ∀ x, |x - (-1)| < δ → f a x ≥ f a (-1)) →
  a = -3 :=
sorry

end NUMINAMATH_CALUDE_local_minimum_implies_a_equals_negative_three_l3440_344021


namespace NUMINAMATH_CALUDE_early_arrival_l3440_344049

theorem early_arrival (usual_time : ℝ) (rate_increase : ℝ) (early_time : ℝ) : 
  usual_time = 35 →
  rate_increase = 7/6 →
  early_time = usual_time - (usual_time / rate_increase) →
  early_time = 5 := by
sorry

end NUMINAMATH_CALUDE_early_arrival_l3440_344049


namespace NUMINAMATH_CALUDE_simplify_fraction_l3440_344040

theorem simplify_fraction : (54 : ℚ) / 486 = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3440_344040


namespace NUMINAMATH_CALUDE_rectangle_cutting_l3440_344023

theorem rectangle_cutting (a b : ℕ) (h_ab : a ≤ b) 
  (h_2 : a * (b - 1) + b * (a - 1) = 940)
  (h_3 : a * (b - 2) + b * (a - 2) = 894) :
  a * (b - 4) + b * (a - 4) = 802 :=
sorry

end NUMINAMATH_CALUDE_rectangle_cutting_l3440_344023


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3440_344098

theorem complex_modulus_problem (z₁ z₂ : ℂ) : 
  (z₁ - 2) * Complex.I = 1 + Complex.I →
  z₂.im = 2 →
  ∃ (r : ℝ), z₁ * z₂ = r →
  Complex.abs z₂ = 2 * Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3440_344098


namespace NUMINAMATH_CALUDE_max_sphere_in_intersecting_cones_l3440_344082

/-- 
Given two congruent right circular cones with base radius 5 and height 12,
whose axes of symmetry intersect at right angles at a point 4 units from
the base of each cone, prove that the maximum possible value of r^2 for a
sphere lying within both cones is 625/169.
-/
theorem max_sphere_in_intersecting_cones (r : ℝ) : 
  let base_radius : ℝ := 5
  let cone_height : ℝ := 12
  let intersection_distance : ℝ := 4
  let slant_height : ℝ := Real.sqrt (cone_height^2 + base_radius^2)
  let max_r_squared : ℝ := (base_radius * (slant_height - intersection_distance) / slant_height)^2
  max_r_squared = 625 / 169 :=
by sorry

end NUMINAMATH_CALUDE_max_sphere_in_intersecting_cones_l3440_344082


namespace NUMINAMATH_CALUDE_books_per_shelf_l3440_344029

theorem books_per_shelf (mystery_shelves : ℕ) (picture_shelves : ℕ) (total_books : ℕ) :
  mystery_shelves = 6 →
  picture_shelves = 2 →
  total_books = 72 →
  total_books / (mystery_shelves + picture_shelves) = 9 :=
by sorry

end NUMINAMATH_CALUDE_books_per_shelf_l3440_344029


namespace NUMINAMATH_CALUDE_files_remaining_l3440_344033

theorem files_remaining (music_files video_files deleted_files : ℕ) 
  (h1 : music_files = 16)
  (h2 : video_files = 48)
  (h3 : deleted_files = 30) :
  music_files + video_files - deleted_files = 34 :=
by sorry

end NUMINAMATH_CALUDE_files_remaining_l3440_344033


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l3440_344015

def M : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def N : Set ℝ := {x | x < 0}

theorem intersection_complement_theorem : 
  M ∩ (Set.univ \ N) = {x : ℝ | 0 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l3440_344015


namespace NUMINAMATH_CALUDE_smallest_perimeter_after_folding_l3440_344065

theorem smallest_perimeter_after_folding (l w : ℝ) (hl : l = 17 / 2) (hw : w = 11) : 
  let original_perimeter := 2 * l + 2 * w
  let folded_perimeter1 := 2 * l + 2 * (w / 4)
  let folded_perimeter2 := 2 * (l / 2) + 2 * (w / 2)
  min folded_perimeter1 folded_perimeter2 = 39 / 2 := by
sorry

end NUMINAMATH_CALUDE_smallest_perimeter_after_folding_l3440_344065


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3440_344070

-- Define sets A and B
def A : Set ℝ := Set.univ
def B : Set ℝ := {x : ℝ | x ≤ 1}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = B := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3440_344070


namespace NUMINAMATH_CALUDE_sum_base6_to_55_l3440_344080

/-- Converts a number from base 6 to base 10 -/
def base6ToBase10 (n : ℕ) : ℕ := sorry

/-- Sums numbers from 1 to n in base 6 -/
def sumBase6 (n : ℕ) : ℕ := sorry

theorem sum_base6_to_55 : base6ToBase10 (sumBase6 55) = 630 := by sorry

end NUMINAMATH_CALUDE_sum_base6_to_55_l3440_344080


namespace NUMINAMATH_CALUDE_parabola_intersection_circle_l3440_344002

/-- The equation of the circle passing through the intersections of the parabola
    y = x^2 - 2x - 3 with the coordinate axes -/
theorem parabola_intersection_circle : 
  ∃ (x y : ℝ), (y = x^2 - 2*x - 3) → 
  ((x = 0 ∧ y = -3) ∨ (y = 0 ∧ (x = -1 ∨ x = 3))) →
  (x - 1)^2 + (y + 1)^2 = 5 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_circle_l3440_344002


namespace NUMINAMATH_CALUDE_basketball_score_proof_l3440_344081

theorem basketball_score_proof (total_points : ℕ) : 
  (∃ (linda_points maria_points other_points : ℕ),
    linda_points = total_points / 5 ∧ 
    maria_points = total_points * 3 / 8 ∧
    other_points ≤ 16 ∧
    linda_points + maria_points + 18 + other_points = total_points ∧
    other_points ≤ 8 * 2) →
  (∃ (other_points : ℕ), 
    other_points = 16 ∧
    other_points ≤ 8 * 2 ∧
    ∃ (linda_points maria_points : ℕ),
      linda_points = total_points / 5 ∧ 
      maria_points = total_points * 3 / 8 ∧
      linda_points + maria_points + 18 + other_points = total_points) :=
by sorry

end NUMINAMATH_CALUDE_basketball_score_proof_l3440_344081


namespace NUMINAMATH_CALUDE_difference_sum_rational_product_irrational_l3440_344045

theorem difference_sum_rational_product_irrational : 
  let a : ℝ := 8
  let b : ℝ := 1
  let c : ℝ := Real.sqrt 3 - 1
  let d : ℝ := 3 * Real.sqrt 3
  (a + b) - (c * d) = 3 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_difference_sum_rational_product_irrational_l3440_344045


namespace NUMINAMATH_CALUDE_distance_between_foci_l3440_344034

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 4)^2 + (y - 3)^2) + Real.sqrt ((x + 6)^2 + (y - 7)^2) = 26

-- Define the foci
def focus1 : ℝ × ℝ := (4, 3)
def focus2 : ℝ × ℝ := (-6, 7)

-- Theorem statement
theorem distance_between_foci :
  let (x1, y1) := focus1
  let (x2, y2) := focus2
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 2 * Real.sqrt 29 := by sorry

end NUMINAMATH_CALUDE_distance_between_foci_l3440_344034


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l3440_344051

def polynomial (x : ℤ) : ℤ := x^3 + 2*x^2 - 3*x - 17

def is_root (x : ℤ) : Prop := polynomial x = 0

theorem integer_roots_of_polynomial :
  {x : ℤ | is_root x} = {-17, -1, 1, 17} := by sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l3440_344051


namespace NUMINAMATH_CALUDE_angle_measure_l3440_344003

theorem angle_measure : 
  ∀ x : ℝ, 
  (x + (4 * x + 7) = 90) →  -- Condition 2 (complementary angles)
  x = 83 / 5 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l3440_344003


namespace NUMINAMATH_CALUDE_ratio_equation_solution_l3440_344025

theorem ratio_equation_solution (c d : ℚ) 
  (h1 : c / d = 4)
  (h2 : c = 15 - 3 * d) : 
  d = 15 / 7 := by sorry

end NUMINAMATH_CALUDE_ratio_equation_solution_l3440_344025


namespace NUMINAMATH_CALUDE_triangle_side_sum_range_l3440_344073

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (a b c : ℝ)
  (acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2)
  (law_of_sines : a / Real.sin A = b / Real.sin B)
  (law_of_cosines : Real.cos A = (b^2 + c^2 - a^2) / (2*b*c))

-- State the theorem
theorem triangle_side_sum_range (t : Triangle) 
  (h1 : Real.cos t.A / t.a + Real.cos t.C / t.c = Real.sin t.B * Real.sin t.C / (3 * Real.sin t.A))
  (h2 : Real.sqrt 3 * Real.sin t.C + Real.cos t.C = 2) :
  6 < t.a + t.b ∧ t.a + t.b ≤ 4 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_sum_range_l3440_344073


namespace NUMINAMATH_CALUDE_diamond_is_conditional_l3440_344088

/-- Represents shapes in a flowchart --/
inductive FlowchartShape
  | Diamond
  | Rectangle
  | Oval

/-- Represents logical structures in an algorithm --/
inductive LogicalStructure
  | Conditional
  | Loop
  | Sequential

/-- A function that maps flowchart shapes to logical structures --/
def shapeToStructure : FlowchartShape → LogicalStructure
  | FlowchartShape.Diamond => LogicalStructure.Conditional
  | FlowchartShape.Rectangle => LogicalStructure.Sequential
  | FlowchartShape.Oval => LogicalStructure.Sequential

/-- Theorem stating that a diamond shape in a flowchart represents a conditional structure --/
theorem diamond_is_conditional :
  shapeToStructure FlowchartShape.Diamond = LogicalStructure.Conditional :=
by
  sorry

end NUMINAMATH_CALUDE_diamond_is_conditional_l3440_344088


namespace NUMINAMATH_CALUDE_smallest_M_for_Q_less_than_three_fourths_l3440_344087

def is_multiple_of_six (M : ℕ) : Prop := ∃ k : ℕ, M = 6 * k

def Q (M : ℕ) : ℚ := (⌈(2 / 3 : ℚ) * M + 1⌉ : ℚ) / (M + 1 : ℚ)

theorem smallest_M_for_Q_less_than_three_fourths :
  ∀ M : ℕ, is_multiple_of_six M → (Q M < 3 / 4 → M ≥ 6) ∧ (Q 6 < 3 / 4) := by sorry

end NUMINAMATH_CALUDE_smallest_M_for_Q_less_than_three_fourths_l3440_344087


namespace NUMINAMATH_CALUDE_managers_salary_l3440_344012

/-- Given an organization with 20 employees and a manager, prove the manager's salary
    based on the change in average salary. -/
theorem managers_salary (num_employees : ℕ) (avg_salary : ℝ) (avg_increase : ℝ) :
  num_employees = 20 →
  avg_salary = 1600 →
  avg_increase = 100 →
  (num_employees * avg_salary + (avg_salary + avg_increase) * (num_employees + 1)) -
    (num_employees * avg_salary) = 3700 :=
by sorry

end NUMINAMATH_CALUDE_managers_salary_l3440_344012


namespace NUMINAMATH_CALUDE_min_value_quadratic_sum_l3440_344066

theorem min_value_quadratic_sum (x y z : ℝ) (h : x + y + z = 1) :
  ∃ (m : ℝ), m = 6/11 ∧ ∀ (a b c : ℝ), a + b + c = 1 → 2*a^2 + b^2 + 3*c^2 ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_sum_l3440_344066


namespace NUMINAMATH_CALUDE_scientific_notation_3900000000_l3440_344071

theorem scientific_notation_3900000000 :
  3900000000 = 3.9 * (10 ^ 9) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_3900000000_l3440_344071


namespace NUMINAMATH_CALUDE_quadratic_solution_property_l3440_344037

theorem quadratic_solution_property (a b : ℝ) : 
  (1 : ℝ)^2 + a * (1 : ℝ) + 2 * b = 0 → 2 * a + 4 * b = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_property_l3440_344037


namespace NUMINAMATH_CALUDE_cyclist_distance_l3440_344022

/-- Represents a cyclist with a given speed -/
structure Cyclist where
  speed : ℝ
  speed_positive : speed > 0

/-- The problem setup -/
def cyclistProblem (c₁ c₂ c₃ : Cyclist) (total_time : ℝ) : Prop :=
  c₁.speed = 12 ∧ c₂.speed = 16 ∧ c₃.speed = 24 ∧
  total_time = 3 ∧
  ∃ (t₁ t₂ t₃ : ℝ),
    t₁ > 0 ∧ t₂ > 0 ∧ t₃ > 0 ∧
    t₁ + t₂ + t₃ = total_time ∧
    c₁.speed * t₁ = c₂.speed * t₂ ∧
    c₂.speed * t₂ = c₃.speed * t₃

theorem cyclist_distance (c₁ c₂ c₃ : Cyclist) (total_time : ℝ) :
  cyclistProblem c₁ c₂ c₃ total_time →
  ∃ (distance : ℝ), distance = 16 ∧
    c₁.speed * (total_time / (1 + c₁.speed / c₂.speed + c₁.speed / c₃.speed)) = distance :=
sorry

end NUMINAMATH_CALUDE_cyclist_distance_l3440_344022


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3440_344056

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0) → a ∈ Set.Icc (-1) 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3440_344056


namespace NUMINAMATH_CALUDE_ratio_problem_l3440_344052

theorem ratio_problem (first_part : ℝ) (percent : ℝ) (second_part : ℝ) : 
  first_part = 4 →
  percent = 20 →
  first_part / second_part = percent / 100 →
  second_part = 20 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l3440_344052


namespace NUMINAMATH_CALUDE_problem_solution_l3440_344064

theorem problem_solution : -1^6 + 8 / (-2)^2 - |(-4) * 3| = -9 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3440_344064


namespace NUMINAMATH_CALUDE_expression_equality_l3440_344044

theorem expression_equality : 
  Real.sqrt 12 + 2 * Real.tan (π / 4) - Real.sin (π / 3) - (1 / 2)⁻¹ = (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3440_344044


namespace NUMINAMATH_CALUDE_correct_subtraction_l3440_344076

theorem correct_subtraction (x : ℤ) (h : x - 32 = 25) : x - 23 = 34 := by
  sorry

end NUMINAMATH_CALUDE_correct_subtraction_l3440_344076


namespace NUMINAMATH_CALUDE_surface_area_of_T_l3440_344036

-- Define the cube
def cube_edge_length : ℝ := 10

-- Define points M, N, O
def point_M : ℝ × ℝ × ℝ := (3, 0, 0)
def point_N : ℝ × ℝ × ℝ := (0, 3, 0)
def point_O : ℝ × ℝ × ℝ := (0, 0, 3)

-- Define the distance from A to M, N, O
def distance_AM : ℝ := 3
def distance_AN : ℝ := 3
def distance_AO : ℝ := 3

-- Function to calculate the area of a triangle given three points in 3D space
def triangle_area (p1 p2 p3 : ℝ × ℝ × ℝ) : ℝ := sorry

-- Function to calculate the surface area of a cube given its edge length
def cube_surface_area (edge_length : ℝ) : ℝ := sorry

-- Theorem: The surface area of solid T is 600 - 27√2
theorem surface_area_of_T :
  let triangle_face_area := triangle_area point_M point_N point_O
  let cube_area := cube_surface_area cube_edge_length
  let removed_area := 3 * triangle_face_area
  cube_area - removed_area = 600 - 27 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_surface_area_of_T_l3440_344036


namespace NUMINAMATH_CALUDE_marble_probability_l3440_344026

theorem marble_probability (total : ℕ) (blue : ℕ) (red_white_prob : ℚ) : 
  total = 30 → blue = 5 → red_white_prob = 5/6 → (total - blue : ℚ) / total = red_white_prob :=
by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l3440_344026


namespace NUMINAMATH_CALUDE_team_a_games_l3440_344010

theorem team_a_games (a : ℕ) (h1 : 3 * a = 4 * (a - (a / 4)))
  (h2 : 2 * (a + 16) = 3 * ((a + 16) - ((a + 16) / 3)))
  (h3 : (a + 16) - ((a + 16) / 3) = a - (a / 4) + 8)
  (h4 : ((a + 16) / 3) = (a / 4) + 8) : a = 192 := by
  sorry

end NUMINAMATH_CALUDE_team_a_games_l3440_344010


namespace NUMINAMATH_CALUDE_emergency_kit_problem_l3440_344086

/-- Given the conditions of Veronica's emergency-preparedness kits problem, 
    prove that the number of food cans must be a multiple of 4 and at least 4. -/
theorem emergency_kit_problem (num_water_bottles : Nat) (num_food_cans : Nat) :
  num_water_bottles = 20 →
  num_water_bottles % 4 = 0 →
  num_food_cans % 4 = 0 →
  (num_water_bottles + num_food_cans) % 4 = 0 →
  num_food_cans ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_emergency_kit_problem_l3440_344086


namespace NUMINAMATH_CALUDE_continuous_piecewise_function_sum_l3440_344083

-- Define the piecewise function g
noncomputable def g (c d : ℝ) (x : ℝ) : ℝ :=
  if x > 3 then c * x + 4
  else if x ≥ -3 then x - 6
  else 3 * x - d

-- Theorem statement
theorem continuous_piecewise_function_sum (c d : ℝ) :
  (∀ x, ContinuousAt (g c d) x) → c + d = -7/3 := by sorry

end NUMINAMATH_CALUDE_continuous_piecewise_function_sum_l3440_344083


namespace NUMINAMATH_CALUDE_special_ellipse_eccentricity_l3440_344011

/-- An ellipse with the property that the lines connecting the two vertices 
    on the minor axis and one of its foci are perpendicular to each other. -/
structure SpecialEllipse where
  /-- Semi-major axis length -/
  a : ℝ
  /-- Semi-minor axis length -/
  b : ℝ
  /-- Distance from center to focus -/
  c : ℝ
  /-- The ellipse satisfies a² = b² + c² -/
  h1 : a^2 = b^2 + c^2
  /-- The lines connecting the vertices on the minor axis and a focus are perpendicular -/
  h2 : b = c

/-- The eccentricity of a SpecialEllipse is √2/2 -/
theorem special_ellipse_eccentricity (E : SpecialEllipse) : 
  E.c / E.a = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_special_ellipse_eccentricity_l3440_344011


namespace NUMINAMATH_CALUDE_mary_regular_hours_l3440_344031

/-- Represents Mary's work schedule and earnings --/
structure WorkSchedule where
  regularHours : ℝ
  overtimeHours : ℝ
  regularRate : ℝ
  overtimeRate : ℝ
  maxHours : ℝ
  maxEarnings : ℝ

/-- Calculates total earnings based on work schedule --/
def totalEarnings (w : WorkSchedule) : ℝ :=
  w.regularHours * w.regularRate + w.overtimeHours * w.overtimeRate

/-- Theorem stating that Mary works 20 hours at her regular rate --/
theorem mary_regular_hours (w : WorkSchedule) 
  (h1 : w.maxHours = 80)
  (h2 : w.regularRate = 8)
  (h3 : w.overtimeRate = w.regularRate * 1.25)
  (h4 : w.maxEarnings = 760)
  (h5 : w.regularHours + w.overtimeHours = w.maxHours)
  (h6 : totalEarnings w = w.maxEarnings) :
  w.regularHours = 20 := by
  sorry

#check mary_regular_hours

end NUMINAMATH_CALUDE_mary_regular_hours_l3440_344031


namespace NUMINAMATH_CALUDE_larger_number_proof_l3440_344000

theorem larger_number_proof (L S : ℕ) (h1 : L > S) (h2 : L - S = 1370) (h3 : L = 6 * S + 15) : L = 1641 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l3440_344000


namespace NUMINAMATH_CALUDE_a_completes_in_15_days_l3440_344050

/-- The number of days it takes for B to complete the work alone -/
def b_days : ℝ := 20

/-- The number of days A and B work together -/
def days_together : ℝ := 6

/-- The fraction of work left after A and B work together -/
def work_left : ℝ := 0.3

/-- The number of days it takes for A to complete the work alone -/
def a_days : ℝ := 15

/-- Proves that given the conditions, A takes 15 days to complete the work alone -/
theorem a_completes_in_15_days :
  days_together * (1 / a_days + 1 / b_days) = 1 - work_left := by
  sorry

end NUMINAMATH_CALUDE_a_completes_in_15_days_l3440_344050


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3440_344072

-- Define set A
def A : Set ℝ := {x | x^2 ≤ 4*x}

-- Define set B
def B : Set ℝ := {x | |x| ≥ 2}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 2 ≤ x ∧ x ≤ 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3440_344072


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3440_344093

theorem complex_modulus_problem (z : ℂ) (h : (1 + Complex.I) * z = (1 - Complex.I)^2) :
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3440_344093


namespace NUMINAMATH_CALUDE_max_digit_sum_for_valid_number_l3440_344092

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 2000 ∧ n < 3000 ∧ n % 13 = 0

def digit_sum (n : ℕ) : ℕ :=
  (n / 100 % 10) + (n / 10 % 10) + (n % 10)

theorem max_digit_sum_for_valid_number :
  ∃ (n : ℕ), is_valid_number n ∧
    ∀ (m : ℕ), is_valid_number m → digit_sum m ≤ digit_sum n ∧
    digit_sum n = 26 :=
sorry

end NUMINAMATH_CALUDE_max_digit_sum_for_valid_number_l3440_344092


namespace NUMINAMATH_CALUDE_range_of_a_l3440_344059

/-- Two circles intersect at exactly two points if and only if 
the distance between their centers is greater than the absolute difference 
of their radii and less than the sum of their radii. -/
axiom circle_intersection_condition (r₁ r₂ d : ℝ) : 
  (∃! (p₁ p₂ : ℝ × ℝ), p₁ ≠ p₂ ∧ 
    ((p₁.1 - d)^2 + p₁.2^2 = r₁^2) ∧ (p₁.1^2 + p₁.2^2 = r₂^2) ∧
    ((p₂.1 - d)^2 + p₂.2^2 = r₁^2) ∧ (p₂.1^2 + p₂.2^2 = r₂^2)) ↔ 
  (abs (r₁ - r₂) < d ∧ d < r₁ + r₂)

/-- The main theorem stating the range of a given the intersection condition. -/
theorem range_of_a : 
  ∀ a : ℝ, (∃! (p₁ p₂ : ℝ × ℝ), p₁ ≠ p₂ ∧ 
    ((p₁.1 - a)^2 + (p₁.2 - a)^2 = 4) ∧ (p₁.1^2 + p₁.2^2 = 4) ∧
    ((p₂.1 - a)^2 + (p₂.2 - a)^2 = 4) ∧ (p₂.1^2 + p₂.2^2 = 4)) → 
  (-2 * Real.sqrt 2 < a ∧ a < 2 * Real.sqrt 2 ∧ a ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3440_344059


namespace NUMINAMATH_CALUDE_hannah_running_difference_l3440_344058

/-- Hannah's running distances for different days of the week -/
structure RunningDistances where
  monday : ℕ     -- Distance in kilometers
  wednesday : ℕ  -- Distance in meters
  friday : ℕ     -- Distance in meters

/-- Calculates the difference in meters between Monday's run and the combined Wednesday and Friday runs -/
def run_difference (distances : RunningDistances) : ℕ :=
  distances.monday * 1000 - (distances.wednesday + distances.friday)

/-- Theorem stating the difference in Hannah's running distances -/
theorem hannah_running_difference : 
  let distances : RunningDistances := { monday := 9, wednesday := 4816, friday := 2095 }
  run_difference distances = 2089 := by
  sorry

end NUMINAMATH_CALUDE_hannah_running_difference_l3440_344058


namespace NUMINAMATH_CALUDE_elderly_employees_in_sample_l3440_344001

theorem elderly_employees_in_sample
  (total_employees : ℕ)
  (young_employees : ℕ)
  (sample_young : ℕ)
  (h1 : total_employees = 430)
  (h2 : young_employees = 160)
  (h3 : sample_young = 32)
  (h4 : ∃ n : ℕ, total_employees = young_employees + 2 * n + n) :
  ∃ m : ℕ, m = 18 ∧ (sample_young : ℚ) / young_employees = (m : ℚ) / ((total_employees - young_employees) / 3) :=
by sorry

end NUMINAMATH_CALUDE_elderly_employees_in_sample_l3440_344001


namespace NUMINAMATH_CALUDE_max_winner_number_l3440_344074

/-- Represents a player in the tournament -/
structure Player where
  number : Nat
  deriving Repr

/-- Represents the tournament -/
def Tournament :=
  {players : Finset Player // players.card = 1024 ∧ ∀ p ∈ players, p.number ≤ 1024}

/-- Predicate for whether a player wins against another player -/
def wins (p1 p2 : Player) : Prop :=
  p1.number < p2.number ∧ p2.number - p1.number > 2

/-- The winner of the tournament -/
def tournamentWinner (t : Tournament) : Player :=
  sorry

/-- The theorem stating the maximum qualification number of the winner -/
theorem max_winner_number (t : Tournament) :
  (tournamentWinner t).number ≤ 20 :=
sorry

end NUMINAMATH_CALUDE_max_winner_number_l3440_344074


namespace NUMINAMATH_CALUDE_people_eating_both_veg_and_nonveg_l3440_344084

theorem people_eating_both_veg_and_nonveg (veg_only : ℕ) (nonveg_only : ℕ) (total_veg : ℕ) 
  (h1 : veg_only = 15)
  (h2 : nonveg_only = 8)
  (h3 : total_veg = 26) :
  total_veg - veg_only = 11 := by
  sorry

#check people_eating_both_veg_and_nonveg

end NUMINAMATH_CALUDE_people_eating_both_veg_and_nonveg_l3440_344084


namespace NUMINAMATH_CALUDE_sin_negative_four_thirds_pi_l3440_344016

theorem sin_negative_four_thirds_pi : 
  Real.sin (-(4/3) * Real.pi) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_four_thirds_pi_l3440_344016


namespace NUMINAMATH_CALUDE_triangle_geometric_sequence_l3440_344091

/-- In a triangle ABC, if sides a, b, c form a geometric sequence and angle A is 60°,
    then (b * sin B) / c = √3/2 -/
theorem triangle_geometric_sequence (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  (b / c = a / b) →  -- Geometric sequence condition
  A = π / 3 →        -- 60° in radians
  A + B + C = π →    -- Sum of angles in a triangle
  a = b * Real.sin A / Real.sin B →  -- Sine rule
  b = c * Real.sin B / Real.sin C →  -- Sine rule
  c = a * Real.sin C / Real.sin A →  -- Sine rule
  (b * Real.sin B) / c = Real.sqrt 3 / 2 := by
sorry


end NUMINAMATH_CALUDE_triangle_geometric_sequence_l3440_344091


namespace NUMINAMATH_CALUDE_unique_solution_cube_equation_l3440_344041

theorem unique_solution_cube_equation :
  ∃! (x : ℝ), x ≠ 0 ∧ (3 * x)^5 = (9 * x)^4 ∧ x = 27 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_cube_equation_l3440_344041


namespace NUMINAMATH_CALUDE_line_through_two_points_l3440_344048

/-- A line in the rectangular coordinate system -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A point in the rectangular coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (l : Line) (p : Point) : Prop :=
  p.x = l.slope * p.y + l.intercept

theorem line_through_two_points 
  (l : Line)
  (p1 p2 : Point)
  (h1 : pointOnLine l p1)
  (h2 : pointOnLine l p2)
  (h3 : l.slope = 8)
  (h4 : l.intercept = 5)
  (h5 : p2.x = p1.x + 2)
  (h6 : p2.y = p1.y + p)
  : p = 1/4 := by
  sorry

#check line_through_two_points

end NUMINAMATH_CALUDE_line_through_two_points_l3440_344048


namespace NUMINAMATH_CALUDE_problem_statement_l3440_344090

-- Define the basic geometric shapes
def Quadrilateral : Type := Unit
def Square : Type := Unit
def Trapezoid : Type := Unit
def Parallelogram : Type := Unit

-- Define the properties
def has_equal_sides (q : Quadrilateral) : Prop := sorry
def is_square (q : Quadrilateral) : Prop := sorry
def is_trapezoid (q : Quadrilateral) : Prop := sorry
def is_parallelogram (q : Quadrilateral) : Prop := sorry

-- Define the propositions
def proposition_1 : Prop :=
  ∀ q : Quadrilateral, ¬(has_equal_sides q → is_square q)

def proposition_2 : Prop :=
  ∀ q : Quadrilateral, is_parallelogram q → ¬is_trapezoid q

def proposition_3 (a b c : ℝ) : Prop :=
  a > b → a * c^2 > b * c^2

theorem problem_statement :
  proposition_1 ∧
  proposition_2 ∧
  ¬(∀ a b c : ℝ, proposition_3 a b c) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_l3440_344090


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l3440_344027

theorem simplify_trig_expression : 
  (1 - Real.cos (30 * π / 180)) * (1 + Real.cos (30 * π / 180)) = 1/4 := by sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l3440_344027


namespace NUMINAMATH_CALUDE_max_k_value_l3440_344089

theorem max_k_value (m : ℝ) (h1 : 0 < m) (h2 : m < 1/2) :
  (∀ k : ℝ, (1/m + 2/(1-2*m) ≥ k) → k ≤ 8) ∧
  (∃ k : ℝ, k = 8 ∧ 1/m + 2/(1-2*m) ≥ k) :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l3440_344089


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3440_344054

theorem complex_equation_solution (z : ℂ) : (z + Complex.I) * (2 + Complex.I) = 5 → z = 2 - 2*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3440_344054


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3440_344004

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The right focus of a hyperbola -/
def right_focus (h : Hyperbola) : ℝ × ℝ := sorry

/-- The asymptotes of a hyperbola -/
def asymptotes (h : Hyperbola) : (ℝ → ℝ) × (ℝ → ℝ) := sorry

/-- A perpendicular line from a point to a line -/
def perpendicular_line (p : ℝ × ℝ) (l : ℝ → ℝ) : ℝ → ℝ := sorry

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- The intersection point of two lines -/
def intersection_point (l1 l2 : ℝ → ℝ) : ℝ × ℝ := sorry

theorem hyperbola_eccentricity (h : Hyperbola) :
  let f := right_focus h
  let (asym1, asym2) := asymptotes h
  let perp := perpendicular_line f asym1
  let a := intersection_point perp asym1
  let b := intersection_point perp asym2
  (b.1 - f.1)^2 + (b.2 - f.2)^2 = 4 * ((a.1 - f.1)^2 + (a.2 - f.2)^2) →
  eccentricity h = 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3440_344004


namespace NUMINAMATH_CALUDE_base_12_remainder_l3440_344085

-- Define the base-12 number 2625₁₂
def base_12_num : ℕ := 2 * 12^3 + 6 * 12^2 + 2 * 12 + 5

-- Theorem statement
theorem base_12_remainder :
  base_12_num % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_base_12_remainder_l3440_344085


namespace NUMINAMATH_CALUDE_cubic_expression_value_l3440_344030

theorem cubic_expression_value (x : ℝ) (h : x^2 + x - 3 = 0) :
  x^3 + 2*x^2 - 2*x + 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_value_l3440_344030


namespace NUMINAMATH_CALUDE_graph_not_simple_l3440_344053

-- Define the equation
def equation (x y : ℝ) : Prop := (x + y)^2 = x^2 + y^2 + 1

-- Define the set of points satisfying the equation
def graph : Set (ℝ × ℝ) := {p | equation p.1 p.2}

-- Theorem stating that the graph is not any of the given options
theorem graph_not_simple : 
  (graph ≠ ∅) ∧ 
  (∃ p q : ℝ × ℝ, p ∈ graph ∧ q ∈ graph ∧ p ≠ q) ∧ 
  (¬∃ a b : ℝ, graph = {p | p.2 = a * p.1 + b} ∪ {p | p.2 = a * p.1 + (b + 1)}) ∧
  (¬∃ c r : ℝ, graph = {p | (p.1 - c)^2 + (p.2 - c)^2 = r^2}) ∧
  (graph ≠ Set.univ) :=
sorry

end NUMINAMATH_CALUDE_graph_not_simple_l3440_344053


namespace NUMINAMATH_CALUDE_train_crossing_time_l3440_344042

/-- Proves that a train 400 meters long, traveling at 144 km/hr, will take 10 seconds to cross an electric pole. -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) (crossing_time : Real) :
  train_length = 400 ∧ train_speed_kmh = 144 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 10 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l3440_344042


namespace NUMINAMATH_CALUDE_f_is_convex_l3440_344062

/-- The function f(x) = x^4 - 2x^3 + 36x^2 - x + 7 -/
def f (x : ℝ) : ℝ := x^4 - 2*x^3 + 36*x^2 - x + 7

/-- The second derivative of f(x) -/
def f'' (x : ℝ) : ℝ := 12*x^2 - 12*x + 72

theorem f_is_convex : ConvexOn ℝ Set.univ f := by
  sorry

end NUMINAMATH_CALUDE_f_is_convex_l3440_344062


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l3440_344068

/-- An increasing arithmetic sequence with specific properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) ∧  -- Arithmetic sequence
  (∀ n, a (n + 1) > a n) ∧  -- Increasing
  (a 1 = 1) ∧  -- a_1 = 1
  (a 3 = (a 2)^2 - 4)  -- a_3 = a_2^2 - 4

/-- The theorem stating the general formula for the sequence -/
theorem arithmetic_sequence_formula (a : ℕ → ℝ) 
    (h : ArithmeticSequence a) : 
    ∀ n : ℕ, a n = 3 * n - 2 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l3440_344068


namespace NUMINAMATH_CALUDE_parking_lot_tires_l3440_344005

/-- Represents the number of tires for a vehicle type -/
structure VehicleTires where
  count : Nat
  wheels : Nat
  spares : Nat

/-- Calculates the total number of tires for a vehicle type -/
def totalTires (v : VehicleTires) : Nat :=
  v.count * (v.wheels + v.spares)

/-- Theorem: The total number of tires in the parking lot is 310 -/
theorem parking_lot_tires :
  let cars := VehicleTires.mk 30 4 1
  let motorcycles := VehicleTires.mk 20 2 2
  let trucks := VehicleTires.mk 10 6 1
  let bicycles := VehicleTires.mk 5 2 0
  totalTires cars + totalTires motorcycles + totalTires trucks + totalTires bicycles = 310 :=
by sorry

end NUMINAMATH_CALUDE_parking_lot_tires_l3440_344005


namespace NUMINAMATH_CALUDE_compare_negative_numbers_l3440_344079

theorem compare_negative_numbers : -4 < -2.1 := by
  sorry

end NUMINAMATH_CALUDE_compare_negative_numbers_l3440_344079


namespace NUMINAMATH_CALUDE_lines_intersect_on_ellipse_l3440_344013

/-- Two lines intersect and their intersection point lies on a specific ellipse -/
theorem lines_intersect_on_ellipse (k₁ k₂ : ℝ) (h : k₁ * k₂ + 2 = 0) :
  ∃ (x y : ℝ),
    (y = k₁ * x + 1 ∧ y = k₂ * x - 1) ∧  -- Lines intersect
    2 * x^2 + y^2 = 6 :=                 -- Intersection point on ellipse
by sorry

end NUMINAMATH_CALUDE_lines_intersect_on_ellipse_l3440_344013


namespace NUMINAMATH_CALUDE_triangle_perimeter_from_average_side_length_l3440_344060

/-- The perimeter of a triangle with average side length 12 is 36 -/
theorem triangle_perimeter_from_average_side_length :
  ∀ (a b c : ℝ), 
  (a + b + c) / 3 = 12 →
  a + b + c = 36 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_from_average_side_length_l3440_344060


namespace NUMINAMATH_CALUDE_rachel_winter_clothing_boxes_l3440_344094

theorem rachel_winter_clothing_boxes : 
  let scarves_per_box : ℕ := 3
  let mittens_per_box : ℕ := 4
  let total_pieces : ℕ := 49
  let pieces_per_box : ℕ := scarves_per_box + mittens_per_box
  let num_boxes : ℕ := total_pieces / pieces_per_box
  num_boxes = 7 := by
sorry

end NUMINAMATH_CALUDE_rachel_winter_clothing_boxes_l3440_344094


namespace NUMINAMATH_CALUDE_infinite_product_equals_sqrt_two_l3440_344017

/-- The nth term of the sequence in the exponent -/
def a (n : ℕ) : ℚ := (2^n - 1) / (3^n)

/-- The infinite product as a function -/
noncomputable def infiniteProduct : ℝ := Real.rpow 2 (∑' n, a n)

/-- The theorem stating that the infinite product equals √2 -/
theorem infinite_product_equals_sqrt_two : infiniteProduct = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_infinite_product_equals_sqrt_two_l3440_344017


namespace NUMINAMATH_CALUDE_cost_calculation_theorem_l3440_344057

/-- Represents the cost calculation for purchasing table tennis equipment --/
def cost_calculation (x : ℕ) : Prop :=
  let racket_price : ℕ := 80
  let ball_price : ℕ := 20
  let racket_quantity : ℕ := 20
  let option1_cost : ℕ := racket_price * racket_quantity
  let option2_cost : ℕ := (racket_price * racket_quantity + ball_price * x) * 9 / 10
  x > 20 → option1_cost = 1600 ∧ option2_cost = 1440 + 18 * x

/-- Theorem stating the cost calculation for purchasing table tennis equipment --/
theorem cost_calculation_theorem (x : ℕ) : cost_calculation x := by
  sorry

#check cost_calculation_theorem

end NUMINAMATH_CALUDE_cost_calculation_theorem_l3440_344057


namespace NUMINAMATH_CALUDE_binomial_variance_example_l3440_344077

/-- A random variable following a binomial distribution with n trials and probability p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- The variance of a binomial distribution -/
def variance (X : BinomialDistribution) : ℝ := X.n * X.p * (1 - X.p)

/-- Theorem: The variance of X ~ B(10, 0.4) is 2.4 -/
theorem binomial_variance_example :
  let X : BinomialDistribution := ⟨10, 0.4, by norm_num⟩
  variance X = 2.4 := by sorry

end NUMINAMATH_CALUDE_binomial_variance_example_l3440_344077


namespace NUMINAMATH_CALUDE_sin_two_theta_l3440_344097

theorem sin_two_theta (θ : ℝ) (h : Real.sin (π/4 + θ) = 1/3) : Real.sin (2*θ) = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_two_theta_l3440_344097


namespace NUMINAMATH_CALUDE_eighty_one_to_negative_two_to_negative_two_equals_three_l3440_344061

theorem eighty_one_to_negative_two_to_negative_two_equals_three :
  (81 : ℝ) ^ (-(2 : ℝ)^(-(2 : ℝ))) = 3 := by
  sorry

end NUMINAMATH_CALUDE_eighty_one_to_negative_two_to_negative_two_equals_three_l3440_344061


namespace NUMINAMATH_CALUDE_unique_two_digit_square_l3440_344035

theorem unique_two_digit_square : ∃! n : ℕ,
  10 ≤ n ∧ n < 100 ∧
  1000 ≤ n^2 ∧ n^2 < 10000 ∧
  (∃ a b : ℕ, n^2 = 1100 * a + 11 * b ∧ 0 < a ∧ a < 10 ∧ 0 ≤ b ∧ b < 10) ∧
  n = 88 := by
sorry

end NUMINAMATH_CALUDE_unique_two_digit_square_l3440_344035
