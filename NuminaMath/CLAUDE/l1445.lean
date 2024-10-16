import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l1445_144551

theorem sum_of_a_and_b (a b : ℝ) (h1 : |a| = 10) (h2 : |b| = 7) (h3 : a > b) :
  a + b = 17 ∨ a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l1445_144551


namespace NUMINAMATH_CALUDE_rearrangement_time_l1445_144544

/-- The time required to write all rearrangements of a name -/
theorem rearrangement_time (name_length : ℕ) (writing_speed : ℕ) (h1 : name_length = 8) (h2 : writing_speed = 15) :
  (name_length.factorial / writing_speed : ℚ) / 60 = 44.8 := by
sorry

end NUMINAMATH_CALUDE_rearrangement_time_l1445_144544


namespace NUMINAMATH_CALUDE_inequality_condition_l1445_144560

theorem inequality_condition (a b : ℝ) (h1 : a * b ≠ 0) :
  (a < b ∧ b < 0) → (1 / a^2 > 1 / b^2) ∧
  ¬(∀ a b : ℝ, a * b ≠ 0 → (1 / a^2 > 1 / b^2) → (a < b ∧ b < 0)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_condition_l1445_144560


namespace NUMINAMATH_CALUDE_prism_problem_l1445_144584

theorem prism_problem (x : ℝ) (d : ℝ) : 
  x > 0 → 
  let a := Real.log x / Real.log 5
  let b := Real.log x / Real.log 7
  let c := Real.log x / Real.log 9
  let surface_area := 2 * (a * b + b * c + c * a)
  let volume := a * b * c
  surface_area * (1/3 * volume) = 54 →
  d = Real.sqrt (a^2 + b^2 + c^2) →
  x = 216 ∧ d = 7 := by
  sorry

#check prism_problem

end NUMINAMATH_CALUDE_prism_problem_l1445_144584


namespace NUMINAMATH_CALUDE_base_5_of_156_l1445_144535

/-- Converts a natural number to its base 5 representation as a list of digits --/
def toBase5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: toBase5 (n / 5)

/-- Theorem: The base 5 representation of 156 (base 10) is [1, 1, 1, 1] --/
theorem base_5_of_156 : toBase5 156 = [1, 1, 1, 1] := by
  sorry

end NUMINAMATH_CALUDE_base_5_of_156_l1445_144535


namespace NUMINAMATH_CALUDE_complex_power_of_one_plus_i_l1445_144548

theorem complex_power_of_one_plus_i : (1 + Complex.I) ^ 6 = -8 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_power_of_one_plus_i_l1445_144548


namespace NUMINAMATH_CALUDE_plywood_area_l1445_144511

/-- The area of a rectangular piece of plywood with width 6 feet and length 4 feet is 24 square feet. -/
theorem plywood_area : 
  ∀ (area width length : ℝ), 
    width = 6 → 
    length = 4 → 
    area = width * length → 
    area = 24 :=
by sorry

end NUMINAMATH_CALUDE_plywood_area_l1445_144511


namespace NUMINAMATH_CALUDE_number_exists_l1445_144536

theorem number_exists : ∃ N : ℝ, 2.5 * N = 199.99999999999997 := by
  sorry

end NUMINAMATH_CALUDE_number_exists_l1445_144536


namespace NUMINAMATH_CALUDE_football_field_area_l1445_144590

theorem football_field_area (A : ℝ) 
  (h1 : 500 / 3500 = 1200 / A) : A = 8400 := by
  sorry

end NUMINAMATH_CALUDE_football_field_area_l1445_144590


namespace NUMINAMATH_CALUDE_reflections_on_circumcircle_l1445_144572

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))

-- Define the orthocenter H
variable (H : EuclideanSpace ℝ (Fin 2))

-- Define the circumcircle
variable (circumcircle : Sphere (EuclideanSpace ℝ (Fin 2)))

-- Assumptions
variable (h_acute : IsAcute A B C)
variable (h_orthocenter : IsOrthocenter H A B C)
variable (h_circumcircle : IsCircumcircle circumcircle A B C)

-- Define the reflections of H with respect to the sides
def reflect_H_BC : EuclideanSpace ℝ (Fin 2) := sorry
def reflect_H_CA : EuclideanSpace ℝ (Fin 2) := sorry
def reflect_H_AB : EuclideanSpace ℝ (Fin 2) := sorry

-- Theorem statement
theorem reflections_on_circumcircle :
  circumcircle.mem reflect_H_BC ∧
  circumcircle.mem reflect_H_CA ∧
  circumcircle.mem reflect_H_AB :=
sorry

end NUMINAMATH_CALUDE_reflections_on_circumcircle_l1445_144572


namespace NUMINAMATH_CALUDE_baseball_card_price_l1445_144552

/-- Given the following conditions:
  - 2 packs of basketball cards were bought at $3 each
  - 5 decks of baseball cards were bought
  - A $50 bill was used for payment
  - $24 was received in change
  Prove that the price of each baseball card deck is $4 -/
theorem baseball_card_price 
  (basketball_packs : ℕ)
  (basketball_price : ℕ)
  (baseball_decks : ℕ)
  (total_paid : ℕ)
  (change_received : ℕ)
  (h1 : basketball_packs = 2)
  (h2 : basketball_price = 3)
  (h3 : baseball_decks = 5)
  (h4 : total_paid = 50)
  (h5 : change_received = 24) :
  (total_paid - change_received - basketball_packs * basketball_price) / baseball_decks = 4 :=
by sorry

end NUMINAMATH_CALUDE_baseball_card_price_l1445_144552


namespace NUMINAMATH_CALUDE_trapezoid_ab_length_l1445_144510

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  -- The length of side AB
  ab : ℝ
  -- The length of side CD
  cd : ℝ
  -- The ratio of the area of triangle ABC to the area of triangle ADC
  area_ratio : ℝ
  -- The sum of AB and CD is 280
  sum_sides : ab + cd = 280
  -- The ratio of the areas is 5:2
  ratio_condition : area_ratio = 5 / 2

/-- Theorem: In a trapezoid ABCD, if the ratio of the area of triangle ABC to the area of triangle ADC
    is 5:2, and AB + CD = 280 cm, then AB = 200 cm -/
theorem trapezoid_ab_length (t : Trapezoid) : t.ab = 200 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_ab_length_l1445_144510


namespace NUMINAMATH_CALUDE_remainder_of_190_div_18_l1445_144591

theorem remainder_of_190_div_18 :
  let g := Nat.gcd 60 190
  g = 18 → 190 % 18 = 10 := by
sorry

end NUMINAMATH_CALUDE_remainder_of_190_div_18_l1445_144591


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1445_144561

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h_a1 : a 1 = 2) 
  (h_S3 : a 1 + a 2 + a 3 = 26) 
  : q = 3 ∨ q = -4 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1445_144561


namespace NUMINAMATH_CALUDE_scientific_notation_300_billion_l1445_144588

theorem scientific_notation_300_billion :
  ∃ (a : ℝ) (n : ℤ), 300000000000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 3 ∧ n = 11 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_300_billion_l1445_144588


namespace NUMINAMATH_CALUDE_cube_gt_iff_gt_l1445_144514

theorem cube_gt_iff_gt (a b : ℝ) : a^3 > b^3 ↔ a > b := by sorry

end NUMINAMATH_CALUDE_cube_gt_iff_gt_l1445_144514


namespace NUMINAMATH_CALUDE_min_value_expression_l1445_144532

theorem min_value_expression (a d b c : ℝ) 
  (ha : a ≥ 0) (hd : d ≥ 0) (hb : b > 0) (hc : c > 0) (h_sum : b + c ≥ a + d) :
  (b / (c + d) + c / (a + b)) ≥ Real.sqrt 2 - 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1445_144532


namespace NUMINAMATH_CALUDE_sphere_between_inclined_planes_l1445_144527

/-- The distance from the center of a sphere to the horizontal plane when placed between two inclined planes -/
theorem sphere_between_inclined_planes 
  (r : ℝ) 
  (angle1 : ℝ) 
  (angle2 : ℝ) 
  (h_r : r = 2) 
  (h_angle1 : angle1 = π / 3)  -- 60 degrees in radians
  (h_angle2 : angle2 = π / 6)  -- 30 degrees in radians
  : ∃ (d : ℝ), d = Real.sqrt 3 + 1 ∧ d = 
    r * Real.sin ((π / 2 - angle1 - angle2) / 2 + angle2) :=
by sorry

end NUMINAMATH_CALUDE_sphere_between_inclined_planes_l1445_144527


namespace NUMINAMATH_CALUDE_slope_inequality_l1445_144578

open Real

theorem slope_inequality (x₁ x₂ : ℝ) (hx : 0 < x₁ ∧ x₁ < x₂) :
  let f := λ x : ℝ => Real.log x
  let k := (f x₂ - f x₁) / (x₂ - x₁)
  1 / x₂ < k ∧ k < 1 / x₁ := by
  sorry

end NUMINAMATH_CALUDE_slope_inequality_l1445_144578


namespace NUMINAMATH_CALUDE_cube_arrangement_exists_l1445_144526

/-- Represents the arrangement of numbers on a cube's edges -/
def CubeArrangement := Fin 12 → Fin 12

/-- Checks if the given arrangement is valid (uses all numbers from 1 to 12 exactly once) -/
def is_valid_arrangement (arr : CubeArrangement) : Prop :=
  (∀ i : Fin 12, ∃ j : Fin 12, arr j = i) ∧ 
  (∀ i j : Fin 12, arr i = arr j → i = j)

/-- Returns the product of numbers on the top face -/
def top_face_product (arr : CubeArrangement) : ℕ :=
  (arr 0 + 1) * (arr 1 + 1) * (arr 2 + 1) * (arr 3 + 1)

/-- Returns the product of numbers on the bottom face -/
def bottom_face_product (arr : CubeArrangement) : ℕ :=
  (arr 4 + 1) * (arr 5 + 1) * (arr 6 + 1) * (arr 7 + 1)

/-- Theorem stating that there exists a valid arrangement with equal products on top and bottom faces -/
theorem cube_arrangement_exists : 
  ∃ (arr : CubeArrangement), 
    is_valid_arrangement arr ∧ 
    top_face_product arr = bottom_face_product arr :=
by sorry

end NUMINAMATH_CALUDE_cube_arrangement_exists_l1445_144526


namespace NUMINAMATH_CALUDE_v_closed_under_multiplication_l1445_144599

-- Define the set of cubes of positive integers
def v : Set ℕ := {n : ℕ | ∃ m : ℕ+, n = m^3}

-- State the theorem
theorem v_closed_under_multiplication :
  ∀ x y : ℕ, x ∈ v → y ∈ v → (x * y) ∈ v :=
by
  sorry

end NUMINAMATH_CALUDE_v_closed_under_multiplication_l1445_144599


namespace NUMINAMATH_CALUDE_line_perp_plane_implies_perp_line_l1445_144570

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the necessary relations
variable (subset : Line → Plane → Prop)
variable (perp : Line → Plane → Prop)
variable (perpLine : Line → Line → Prop)

-- State the theorem
theorem line_perp_plane_implies_perp_line 
  (α β : Plane) (m n : Line) 
  (h_diff_planes : α ≠ β)
  (h_diff_lines : m ≠ n)
  (h_m_subset_β : subset m β)
  (h_n_subset_β : subset n β)
  (h_m_subset_α : subset m α)
  (h_n_perp_α : perp n α) :
  perpLine n m :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_implies_perp_line_l1445_144570


namespace NUMINAMATH_CALUDE_parentheses_number_l1445_144554

theorem parentheses_number (x : ℤ) (h : x - (-2) = 3) : x = 1 := by
  sorry

end NUMINAMATH_CALUDE_parentheses_number_l1445_144554


namespace NUMINAMATH_CALUDE_remainder_theorem_l1445_144530

theorem remainder_theorem (n : ℤ) (k : ℤ) (h : n = 30 * k - 1) :
  (n^2 + 2*n + n^3 + 3) % 30 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1445_144530


namespace NUMINAMATH_CALUDE_min_sum_reciprocals_l1445_144581

theorem min_sum_reciprocals (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 3) :
  (1 / x + 1 / y + 1 / z) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_reciprocals_l1445_144581


namespace NUMINAMATH_CALUDE_odd_function_fourth_composition_even_l1445_144507

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Theorem statement
theorem odd_function_fourth_composition_even (f : ℝ → ℝ) (h : OddFunction f) : 
  EvenFunction (fun x ↦ f (f (f (f x)))) :=
sorry

end NUMINAMATH_CALUDE_odd_function_fourth_composition_even_l1445_144507


namespace NUMINAMATH_CALUDE_hexagon_perimeter_hexagon_perimeter_proof_l1445_144537

/-- The perimeter of a regular hexagon with side length 8 is 48. -/
theorem hexagon_perimeter : ℕ → ℕ
  | 6 => 48
  | _ => 0

#check hexagon_perimeter
-- hexagon_perimeter : ℕ → ℕ

theorem hexagon_perimeter_proof (n : ℕ) (h : n = 6) : 
  hexagon_perimeter n = 8 * n :=
by sorry

end NUMINAMATH_CALUDE_hexagon_perimeter_hexagon_perimeter_proof_l1445_144537


namespace NUMINAMATH_CALUDE_chapter_page_difference_l1445_144500

theorem chapter_page_difference (first_chapter_pages second_chapter_pages : ℕ) 
  (h1 : first_chapter_pages = 48) 
  (h2 : second_chapter_pages = 11) : 
  first_chapter_pages - second_chapter_pages = 37 := by
  sorry

end NUMINAMATH_CALUDE_chapter_page_difference_l1445_144500


namespace NUMINAMATH_CALUDE_expression_evaluation_l1445_144549

theorem expression_evaluation (x y : ℚ) (hx : x = 1/3) (hy : y = -1/2) :
  (2*x + 3*y)^2 - (2*x + y)*(2*x - y) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1445_144549


namespace NUMINAMATH_CALUDE_race_champion_is_C_l1445_144563

-- Define the participants
inductive Participant : Type
| A : Participant
| B : Participant
| C : Participant
| D : Participant

-- Define the opinions
def xiaozhangs_opinion (champion : Participant) : Prop :=
  champion = Participant.A ∨ champion = Participant.B

def xiaowangs_opinion (champion : Participant) : Prop :=
  champion ≠ Participant.C

def xiaolis_opinion (champion : Participant) : Prop :=
  champion ≠ Participant.A ∧ champion ≠ Participant.B

-- Theorem statement
theorem race_champion_is_C :
  ∀ (champion : Participant),
    (xiaozhangs_opinion champion ∨ xiaowangs_opinion champion ∨ xiaolis_opinion champion) ∧
    (¬(xiaozhangs_opinion champion ∧ xiaowangs_opinion champion) ∧
     ¬(xiaozhangs_opinion champion ∧ xiaolis_opinion champion) ∧
     ¬(xiaowangs_opinion champion ∧ xiaolis_opinion champion)) →
    champion = Participant.C :=
by sorry

end NUMINAMATH_CALUDE_race_champion_is_C_l1445_144563


namespace NUMINAMATH_CALUDE_least_integer_for_triangle_with_integer_area_l1445_144525

theorem least_integer_for_triangle_with_integer_area : 
  ∃ (a : ℕ), a > 14 ∧ 
  (∀ b : ℕ, b > 14 ∧ b < a → 
    ¬(∃ A : ℕ, A^2 = (3*b^2/4) * ((b^2/4) - 1))) ∧
  (∃ A : ℕ, A^2 = (3*a^2/4) * ((a^2/4) - 1)) ∧
  a = 52 := by
sorry

end NUMINAMATH_CALUDE_least_integer_for_triangle_with_integer_area_l1445_144525


namespace NUMINAMATH_CALUDE_two_digit_reverse_pythagoras_sum_l1445_144558

theorem two_digit_reverse_pythagoras_sum : ∃ (x y n : ℕ), 
  (10 ≤ x ∧ x < 100) ∧ 
  (10 ≤ y ∧ y < 100) ∧ 
  (∃ (a b : ℕ), x = 10 * a + b ∧ y = 10 * b + a ∧ a < 10 ∧ b < 10) ∧
  x^2 + y^2 = n^2 ∧
  x + y + n = 132 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_reverse_pythagoras_sum_l1445_144558


namespace NUMINAMATH_CALUDE_number_of_boys_l1445_144587

theorem number_of_boys (total : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 150 →
  boys + girls = total →
  girls = boys * total / 100 →
  boys = 60 := by
sorry

end NUMINAMATH_CALUDE_number_of_boys_l1445_144587


namespace NUMINAMATH_CALUDE_computer_arrangements_l1445_144550

theorem computer_arrangements : Nat.factorial 8 = 40320 := by
  sorry

end NUMINAMATH_CALUDE_computer_arrangements_l1445_144550


namespace NUMINAMATH_CALUDE_steak_knife_cost_l1445_144573

/-- The cost of each single steak knife, given the number of sets, knives per set, and cost per set. -/
theorem steak_knife_cost (num_sets : ℕ) (knives_per_set : ℕ) (cost_per_set : ℝ) :
  num_sets = 2 → knives_per_set = 4 → cost_per_set = 80 →
  (num_sets * cost_per_set) / (num_sets * knives_per_set) = 20 := by
  sorry

end NUMINAMATH_CALUDE_steak_knife_cost_l1445_144573


namespace NUMINAMATH_CALUDE_cos_36_minus_cos_72_eq_half_l1445_144541

theorem cos_36_minus_cos_72_eq_half :
  Real.cos (36 * π / 180) - Real.cos (72 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_36_minus_cos_72_eq_half_l1445_144541


namespace NUMINAMATH_CALUDE_parrots_per_cage_l1445_144531

theorem parrots_per_cage (num_cages : ℝ) (parakeets_per_cage : ℝ) (total_birds : ℕ) :
  num_cages = 6 →
  parakeets_per_cage = 2 →
  total_birds = 48 →
  ∃ parrots_per_cage : ℕ, 
    (parrots_per_cage : ℝ) * num_cages + parakeets_per_cage * num_cages = total_birds ∧
    parrots_per_cage = 6 := by
  sorry

end NUMINAMATH_CALUDE_parrots_per_cage_l1445_144531


namespace NUMINAMATH_CALUDE_mean_of_added_numbers_l1445_144515

theorem mean_of_added_numbers (original_numbers : List ℝ) (x y z : ℝ) :
  original_numbers.length = 7 →
  original_numbers.sum / original_numbers.length = 75 →
  (original_numbers.sum + x + y + z) / (original_numbers.length + 3) = 90 →
  (x + y + z) / 3 = 125 := by
sorry

end NUMINAMATH_CALUDE_mean_of_added_numbers_l1445_144515


namespace NUMINAMATH_CALUDE_alices_favorite_number_l1445_144597

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem alices_favorite_number :
  ∃! n : ℕ, 100 < n ∧ n < 150 ∧ 
  13 ∣ n ∧ ¬(2 ∣ n) ∧ 
  4 ∣ sum_of_digits n ∧
  n = 143 :=
sorry

end NUMINAMATH_CALUDE_alices_favorite_number_l1445_144597


namespace NUMINAMATH_CALUDE_existence_of_n_with_2000_prime_divisors_l1445_144509

theorem existence_of_n_with_2000_prime_divisors :
  ∃ n : ℕ+, (n.val.factors.length = 2000) ∧ (n.val ∣ 2^n.val + 1) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_n_with_2000_prime_divisors_l1445_144509


namespace NUMINAMATH_CALUDE_twelve_tone_equal_temperament_l1445_144529

theorem twelve_tone_equal_temperament (a : ℕ → ℝ) :
  (∀ n, 1 ≤ n → n < 13 → a (n + 1) / a n = a 2 / a 1) →  -- Equal ratio between adjacent terms
  a 13 = 2 * a 1 →                                      -- Last term is twice the first term
  a 8 / a 2 = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_twelve_tone_equal_temperament_l1445_144529


namespace NUMINAMATH_CALUDE_octagon_circles_theorem_l1445_144534

theorem octagon_circles_theorem (r : ℝ) (a b : ℤ) : 
  (∃ (s : ℝ), s = 2 ∧ s = r * Real.sqrt (2 - Real.sqrt 2)) →
  r^2 = a + b * Real.sqrt 2 →
  (a : ℝ) + b = 6 := by
sorry

end NUMINAMATH_CALUDE_octagon_circles_theorem_l1445_144534


namespace NUMINAMATH_CALUDE_exactly_two_absent_probability_l1445_144501

-- Define the probability of a student being absent
def prob_absent : ℚ := 1 / 20

-- Define the probability of a student being present
def prob_present : ℚ := 1 - prob_absent

-- Define the number of students we're considering
def num_students : ℕ := 3

-- Define the number of absent students we're looking for
def num_absent : ℕ := 2

-- Theorem statement
theorem exactly_two_absent_probability :
  (prob_absent ^ num_absent * prob_present ^ (num_students - num_absent)) * (num_students.choose num_absent) = 7125 / 1000000 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_absent_probability_l1445_144501


namespace NUMINAMATH_CALUDE_probability_3_1_is_5_over_10_2_l1445_144524

def blue_balls : ℕ := 10
def red_balls : ℕ := 8
def total_balls : ℕ := blue_balls + red_balls
def drawn_balls : ℕ := 4

def probability_3_1 : ℚ :=
  let total_ways := Nat.choose total_balls drawn_balls
  let ways_3blue_1red := Nat.choose blue_balls 3 * Nat.choose red_balls 1
  let ways_1blue_3red := Nat.choose blue_balls 1 * Nat.choose red_balls 3
  (ways_3blue_1red + ways_1blue_3red : ℚ) / total_ways

theorem probability_3_1_is_5_over_10_2 :
  probability_3_1 = 5 / 10.2 := by sorry

end NUMINAMATH_CALUDE_probability_3_1_is_5_over_10_2_l1445_144524


namespace NUMINAMATH_CALUDE_fifth_bowler_score_l1445_144542

/-- A bowling team with 5 members and their scores -/
structure BowlingTeam where
  total_points : ℕ
  p1 : ℕ
  p2 : ℕ
  p3 : ℕ
  p4 : ℕ
  p5 : ℕ

/-- The conditions of the bowling team's scores -/
def validBowlingTeam (team : BowlingTeam) : Prop :=
  team.total_points = 2000 ∧
  team.p1 = team.p2 / 4 ∧
  team.p2 = team.p3 * 5 / 3 ∧
  team.p3 ≤ 500 ∧
  team.p3 = team.p4 * 3 / 5 ∧
  team.p4 = team.p5 * 9 / 10 ∧
  team.p1 + team.p2 + team.p3 + team.p4 + team.p5 = team.total_points

theorem fifth_bowler_score (team : BowlingTeam) :
  validBowlingTeam team → team.p5 = 561 := by
  sorry

end NUMINAMATH_CALUDE_fifth_bowler_score_l1445_144542


namespace NUMINAMATH_CALUDE_track_completion_time_l1445_144559

/-- Represents a circular running track --/
structure Track where
  circumference : ℝ
  circumference_positive : circumference > 0

/-- Represents a runner on the track --/
structure Runner where
  speed : ℝ
  speed_positive : speed > 0

/-- Represents an event where two runners meet --/
structure MeetingEvent where
  time : ℝ
  time_nonnegative : time ≥ 0

/-- The main theorem to prove --/
theorem track_completion_time
  (track : Track)
  (runner1 runner2 runner3 : Runner)
  (meeting12 : MeetingEvent)
  (meeting23 : MeetingEvent)
  (meeting31 : MeetingEvent)
  (h1 : meeting23.time - meeting12.time = 15)
  (h2 : meeting31.time - meeting23.time = 25) :
  track.circumference / runner1.speed = 80 :=
sorry

end NUMINAMATH_CALUDE_track_completion_time_l1445_144559


namespace NUMINAMATH_CALUDE_scientific_notation_of_2720000_l1445_144574

theorem scientific_notation_of_2720000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 2720000 = a * (10 : ℝ) ^ n ∧ a = 2.72 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_2720000_l1445_144574


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l1445_144582

/-- The eccentricity of an ellipse with equation x²/a² + y² = 1, where a > 1 and the major axis length is 4 -/
theorem ellipse_eccentricity (a : ℝ) (h1 : a > 1) (h2 : 2 * a = 4) :
  let c := Real.sqrt (a^2 - 1)
  (c / a) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l1445_144582


namespace NUMINAMATH_CALUDE_multiplication_simplification_l1445_144565

theorem multiplication_simplification : 11 * (1 / 17) * 34 = 22 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_simplification_l1445_144565


namespace NUMINAMATH_CALUDE_sue_grocery_spending_l1445_144585

/-- Calculates Sue's spending on a grocery shopping trip with specific conditions --/
theorem sue_grocery_spending : 
  let apple_price : ℚ := 2
  let apple_quantity : ℕ := 4
  let juice_price : ℚ := 6
  let juice_quantity : ℕ := 2
  let bread_price : ℚ := 3
  let bread_quantity : ℕ := 3
  let cheese_price : ℚ := 4
  let cheese_quantity : ℕ := 2
  let cereal_price : ℚ := 8
  let cereal_quantity : ℕ := 1
  let cheese_discount : ℚ := 0.25
  let order_discount_threshold : ℚ := 40
  let order_discount_rate : ℚ := 0.1

  let discounted_cheese_price : ℚ := cheese_price * (1 - cheese_discount)
  let subtotal : ℚ := 
    apple_price * apple_quantity +
    juice_price * juice_quantity +
    bread_price * bread_quantity +
    discounted_cheese_price * cheese_quantity +
    cereal_price * cereal_quantity

  let final_total : ℚ := 
    if subtotal ≥ order_discount_threshold
    then subtotal * (1 - order_discount_rate)
    else subtotal

  final_total = 387/10 := by sorry

end NUMINAMATH_CALUDE_sue_grocery_spending_l1445_144585


namespace NUMINAMATH_CALUDE_diamond_three_four_l1445_144555

def diamond (a b : ℝ) : ℝ := a^2 * b^2 - b + 2

theorem diamond_three_four : diamond 3 4 = 142 := by
  sorry

end NUMINAMATH_CALUDE_diamond_three_four_l1445_144555


namespace NUMINAMATH_CALUDE_min_value_and_monotonicity_l1445_144596

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x ^ 2 - a * x

theorem min_value_and_monotonicity (a : ℝ) :
  (∃ (x : ℝ), x > 0 ∧ f (-2 * Real.exp 1) x = 3 ∧ 
    ∀ (y : ℝ), y > 0 → f (-2 * Real.exp 1) y ≥ f (-2 * Real.exp 1) x) ∧
  (∀ (x y : ℝ), 0 < x ∧ x < y → (f a x ≥ f a y ↔ a ≥ 2 / Real.exp 1)) :=
sorry

end NUMINAMATH_CALUDE_min_value_and_monotonicity_l1445_144596


namespace NUMINAMATH_CALUDE_triangle_side_ratio_max_l1445_144512

/-- For any triangle with sides a, b, c, where c is the largest side and θ is the angle opposite to c,
    the maximum value of (a + b) / c is √2, given that θ ≤ 90°. -/
theorem triangle_side_ratio_max (a b c : ℝ) (θ : Real) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
    (h4 : c ≥ a ∧ c ≥ b) (h5 : θ ≤ Real.pi / 2) :
    (a + b) / c ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_max_l1445_144512


namespace NUMINAMATH_CALUDE_script_lines_proof_l1445_144567

theorem script_lines_proof :
  ∀ (lines1 lines2 lines3 : ℕ),
  lines1 = 20 →
  lines1 = lines2 + 8 →
  lines2 = 3 * lines3 + 6 →
  lines3 = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_script_lines_proof_l1445_144567


namespace NUMINAMATH_CALUDE_integer_solution_existence_l1445_144586

theorem integer_solution_existence (a : ℤ) : 
  (∃ k : ℤ, 2 * a^2 = 7 * k + 2) ↔ (∃ ℓ : ℤ, a = 7 * ℓ + 1 ∨ a = 7 * ℓ - 1) :=
by sorry

end NUMINAMATH_CALUDE_integer_solution_existence_l1445_144586


namespace NUMINAMATH_CALUDE_smallest_n_for_integer_T_l1445_144557

/-- Sum of reciprocals of prime digits -/
def P : ℚ := 1/2 + 1/3 + 1/5 + 1/7

/-- T_n is the sum of the reciprocals of the prime digits of integers from 1 to 5^n inclusive -/
def T (n : ℕ) : ℚ := n * (5^(n-1) : ℚ) * P

/-- 42 is the smallest positive integer n for which T_n is an integer -/
theorem smallest_n_for_integer_T : ∀ k : ℕ, k > 0 → (∃ m : ℤ, T k = m) → k ≥ 42 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_integer_T_l1445_144557


namespace NUMINAMATH_CALUDE_christines_dog_weight_l1445_144598

/-- Theorem: Christine's dog weight calculation -/
theorem christines_dog_weight (cat1_weight cat2_weight : ℕ) (dog_weight : ℕ) : 
  cat1_weight = 7 →
  cat2_weight = 10 →
  dog_weight = 2 * (cat1_weight + cat2_weight) →
  dog_weight = 34 := by
  sorry

end NUMINAMATH_CALUDE_christines_dog_weight_l1445_144598


namespace NUMINAMATH_CALUDE_lars_baking_hours_l1445_144589

/-- The number of loaves of bread Lars can bake per hour -/
def loaves_per_hour : ℕ := 10

/-- The number of baguettes Lars can bake per hour -/
def baguettes_per_hour : ℕ := 15

/-- The total number of breads Lars makes -/
def total_breads : ℕ := 150

/-- The number of hours Lars bakes each day -/
def baking_hours : ℕ := 6

theorem lars_baking_hours :
  loaves_per_hour * baking_hours + baguettes_per_hour * baking_hours = total_breads :=
sorry

end NUMINAMATH_CALUDE_lars_baking_hours_l1445_144589


namespace NUMINAMATH_CALUDE_fly_path_total_distance_l1445_144503

theorem fly_path_total_distance (radius : ℝ) (leg : ℝ) (h1 : radius = 75) (h2 : leg = 70) :
  let diameter : ℝ := 2 * radius
  let other_leg : ℝ := Real.sqrt (diameter^2 - leg^2)
  diameter + leg + other_leg = 352.6 := by
sorry

end NUMINAMATH_CALUDE_fly_path_total_distance_l1445_144503


namespace NUMINAMATH_CALUDE_red_peaches_per_basket_l1445_144576

/-- Given 6 baskets of peaches with a total of 96 red peaches,
    prove that each basket contains 16 red peaches. -/
theorem red_peaches_per_basket :
  let total_baskets : ℕ := 6
  let total_red_peaches : ℕ := 96
  let green_peaches_per_basket : ℕ := 18
  (total_red_peaches / total_baskets : ℚ) = 16 := by
  sorry

end NUMINAMATH_CALUDE_red_peaches_per_basket_l1445_144576


namespace NUMINAMATH_CALUDE_science_club_neither_subject_l1445_144545

theorem science_club_neither_subject (total : ℕ) (chemistry : ℕ) (biology : ℕ) (both : ℕ) 
  (h1 : total = 75)
  (h2 : chemistry = 42)
  (h3 : biology = 33)
  (h4 : both = 18) :
  total - (chemistry + biology - both) = 18 := by
  sorry

end NUMINAMATH_CALUDE_science_club_neither_subject_l1445_144545


namespace NUMINAMATH_CALUDE_min_four_dollar_frisbees_l1445_144569

theorem min_four_dollar_frisbees (total_frisbees : ℕ) (total_receipts : ℕ) : 
  total_frisbees = 64 →
  total_receipts = 196 →
  ∃ (three_dollar : ℕ) (four_dollar : ℕ),
    three_dollar + four_dollar = total_frisbees ∧
    3 * three_dollar + 4 * four_dollar = total_receipts ∧
    ∀ (other_four_dollar : ℕ),
      (∃ (other_three_dollar : ℕ),
        other_three_dollar + other_four_dollar = total_frisbees ∧
        3 * other_three_dollar + 4 * other_four_dollar = total_receipts) →
      four_dollar ≤ other_four_dollar ∧
      four_dollar = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_four_dollar_frisbees_l1445_144569


namespace NUMINAMATH_CALUDE_square_to_obtuse_triangle_l1445_144543

/-- Represents a part of a square -/
structure SquarePart where
  -- Add necessary fields to represent a part of a square
  -- This is a placeholder and should be defined more precisely based on the problem requirements

/-- Represents a triangle -/
structure Triangle where
  -- Add necessary fields to represent a triangle
  -- This is a placeholder and should be defined more precisely based on the problem requirements

/-- Determines if a triangle is obtuse -/
def is_obtuse (t : Triangle) : Prop :=
  -- Add the condition for a triangle to be obtuse
  -- This is a placeholder and should be defined more precisely based on the problem requirements
  sorry

/-- Determines if parts can form a triangle -/
def can_form_triangle (parts : List SquarePart) : Prop :=
  -- Add the condition for parts to be able to form a triangle
  -- This is a placeholder and should be defined more precisely based on the problem requirements
  sorry

/-- Theorem stating that a square can be cut into 3 parts that form an obtuse triangle -/
theorem square_to_obtuse_triangle :
  ∃ (parts : List SquarePart), parts.length = 3 ∧
    ∃ (t : Triangle), can_form_triangle parts ∧ is_obtuse t :=
sorry

end NUMINAMATH_CALUDE_square_to_obtuse_triangle_l1445_144543


namespace NUMINAMATH_CALUDE_y_equals_seven_l1445_144580

/-- A shape composed entirely of right angles with specific side lengths -/
structure RightAngledShape where
  /-- Length of one side -/
  side1 : ℝ
  /-- Length of another side -/
  side2 : ℝ
  /-- Length of another side -/
  side3 : ℝ
  /-- Length of another side -/
  side4 : ℝ
  /-- Unknown length to be calculated -/
  Y : ℝ
  /-- The total horizontal lengths on the top and bottom sides are equal -/
  total_length_eq : side1 + side3 + Y + side2 = side4 + side2 + side3 + 5

/-- The theorem stating that Y equals 7 for the given shape -/
theorem y_equals_seven (shape : RightAngledShape) 
  (h1 : shape.side1 = 2) 
  (h2 : shape.side2 = 3) 
  (h3 : shape.side3 = 1) 
  (h4 : shape.side4 = 4) : 
  shape.Y = 7 := by
  sorry

end NUMINAMATH_CALUDE_y_equals_seven_l1445_144580


namespace NUMINAMATH_CALUDE_triple_overlap_is_six_l1445_144502

/-- Represents a rectangular carpet with width and height -/
structure Carpet where
  width : ℝ
  height : ℝ

/-- Represents the hall and the arrangement of carpets -/
structure CarpetArrangement where
  hallWidth : ℝ
  hallHeight : ℝ
  carpet1 : Carpet
  carpet2 : Carpet
  carpet3 : Carpet

/-- Calculates the area of triple overlap in the carpet arrangement -/
def tripleOverlapArea (arrangement : CarpetArrangement) : ℝ :=
  sorry

/-- Theorem stating that the triple overlap area is 6 square meters -/
theorem triple_overlap_is_six (arrangement : CarpetArrangement) 
  (h1 : arrangement.hallWidth = 10 ∧ arrangement.hallHeight = 10)
  (h2 : arrangement.carpet1 = ⟨6, 8⟩)
  (h3 : arrangement.carpet2 = ⟨6, 6⟩)
  (h4 : arrangement.carpet3 = ⟨5, 7⟩) :
  tripleOverlapArea arrangement = 6 := by
  sorry

end NUMINAMATH_CALUDE_triple_overlap_is_six_l1445_144502


namespace NUMINAMATH_CALUDE_bookstore_problem_l1445_144593

theorem bookstore_problem (total_books : ℕ) (unsold_books : ℕ) (customers : ℕ) :
  total_books = 40 →
  unsold_books = 4 →
  customers = 4 →
  (total_books - unsold_books) % customers = 0 →
  (total_books - unsold_books) / customers = 9 :=
by sorry

end NUMINAMATH_CALUDE_bookstore_problem_l1445_144593


namespace NUMINAMATH_CALUDE_xy_is_zero_l1445_144517

theorem xy_is_zero (x y : ℝ) (h1 : x + y = 5) (h2 : x^3 + y^3 = 125) : x * y = 0 := by
  sorry

end NUMINAMATH_CALUDE_xy_is_zero_l1445_144517


namespace NUMINAMATH_CALUDE_shaded_area_is_75_l1445_144568

-- Define the side lengths of the squares
def larger_side : ℝ := 10
def smaller_side : ℝ := 5

-- Define the areas of the squares
def larger_area : ℝ := larger_side ^ 2
def smaller_area : ℝ := smaller_side ^ 2

-- Define the shaded area
def shaded_area : ℝ := larger_area - smaller_area

-- Theorem to prove
theorem shaded_area_is_75 : shaded_area = 75 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_75_l1445_144568


namespace NUMINAMATH_CALUDE_sum_of_squares_representation_l1445_144516

theorem sum_of_squares_representation (x y z : ℕ) (h : x * y = z^2 + 1) :
  ∃ (a b c d : ℤ), (x : ℤ) = a^2 + b^2 ∧ (y : ℤ) = c^2 + d^2 ∧ (z : ℤ) = a * c + b * d := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_representation_l1445_144516


namespace NUMINAMATH_CALUDE_lineup_combinations_l1445_144540

def team_size : ℕ := 12
def strong_players : ℕ := 4
def positions_to_fill : ℕ := 5

theorem lineup_combinations : 
  (strong_players * (strong_players - 1) * 
   (team_size - 2) * (team_size - 3) * (team_size - 4)) = 8640 := by
  sorry

end NUMINAMATH_CALUDE_lineup_combinations_l1445_144540


namespace NUMINAMATH_CALUDE_quadratic_root_l1445_144575

theorem quadratic_root (a b k : ℝ) : 
  (∃ x : ℝ, x^2 - (a+b)*x + a*b*(1-k) = 0 ∧ x = 1) →
  (∃ y : ℝ, y^2 - (a+b)*y + a*b*(1-k) = 0 ∧ y = a + b - 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_l1445_144575


namespace NUMINAMATH_CALUDE_lucky_number_property_l1445_144579

/-- A number is lucky if the sum of its digits is 7 -/
def IsLucky (n : ℕ) : Prop :=
  (n.digits 10).sum = 7

/-- The sequence of lucky numbers in ascending order -/
def LuckySequence : ℕ → ℕ :=
  sorry

theorem lucky_number_property (n : ℕ) :
  LuckySequence n = 2005 → LuckySequence (5 * n) = 30301 :=
by
  sorry

end NUMINAMATH_CALUDE_lucky_number_property_l1445_144579


namespace NUMINAMATH_CALUDE_sum_of_factorials_1_to_10_l1445_144577

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def sum_of_factorials (n : ℕ) : ℕ :=
  match n with
  | 0 => factorial 0
  | n + 1 => sum_of_factorials n + factorial (n + 1)

theorem sum_of_factorials_1_to_10 : sum_of_factorials 10 = 4037913 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_factorials_1_to_10_l1445_144577


namespace NUMINAMATH_CALUDE_least_m_for_no_real_roots_l1445_144562

theorem least_m_for_no_real_roots : 
  ∃ (m : ℤ), (∀ (x : ℝ), 3 * x * (m * x + 6) - 2 * x^2 + 8 ≠ 0) ∧
  (∀ (k : ℤ), k < m → ∃ (x : ℝ), 3 * x * (k * x + 6) - 2 * x^2 + 8 = 0) ∧
  m = 4 := by
  sorry

end NUMINAMATH_CALUDE_least_m_for_no_real_roots_l1445_144562


namespace NUMINAMATH_CALUDE_real_estate_investment_l1445_144523

theorem real_estate_investment 
  (total_investment : ℝ) 
  (real_estate_ratio : ℝ) 
  (h1 : total_investment = 200000)
  (h2 : real_estate_ratio = 7) : 
  let mutual_funds := total_investment / (real_estate_ratio + 1)
  let real_estate := real_estate_ratio * mutual_funds
  real_estate = 175000 := by
sorry

end NUMINAMATH_CALUDE_real_estate_investment_l1445_144523


namespace NUMINAMATH_CALUDE_inscribed_sphere_surface_area_l1445_144539

/-- Given a cube with edge length 4, the surface area of its inscribed sphere is 16π. -/
theorem inscribed_sphere_surface_area (edge_length : ℝ) (h : edge_length = 4) :
  let radius : ℝ := edge_length / 2
  4 * π * radius^2 = 16 * π :=
by sorry

end NUMINAMATH_CALUDE_inscribed_sphere_surface_area_l1445_144539


namespace NUMINAMATH_CALUDE_min_sum_squares_l1445_144533

theorem min_sum_squares (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) 
  (heq : a^2 - 2015*a = b^2 - 2015*b) : 
  ∃ (m : ℝ), (∀ (x y : ℝ), x > 0 → y > 0 → x ≠ y → x^2 - 2015*x = y^2 - 2015*y → 
    x^2 + y^2 ≥ m) ∧ m = (2015^2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1445_144533


namespace NUMINAMATH_CALUDE_total_share_l1445_144513

/-- Given that z's share is 400, y's share is 20% more than z's, and x's share is 25% more than y's,
    the total amount shared between x, y, and z is 1480. -/
theorem total_share (z : ℕ) (h1 : z = 400) : 
  let y := z + z / 5
  let x := y + y / 4
  x + y + z = 1480 := by
  sorry

end NUMINAMATH_CALUDE_total_share_l1445_144513


namespace NUMINAMATH_CALUDE_absolute_difference_equation_l1445_144505

theorem absolute_difference_equation : 
  ∃! x : ℝ, |16 - x| - |x - 12| = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_absolute_difference_equation_l1445_144505


namespace NUMINAMATH_CALUDE_fred_gave_25_seashells_l1445_144522

/-- The number of seashells Fred initially had -/
def initial_seashells : ℕ := 47

/-- The number of seashells Fred has now -/
def remaining_seashells : ℕ := 22

/-- The number of seashells Fred gave to Jessica -/
def seashells_given : ℕ := initial_seashells - remaining_seashells

theorem fred_gave_25_seashells : seashells_given = 25 := by
  sorry

end NUMINAMATH_CALUDE_fred_gave_25_seashells_l1445_144522


namespace NUMINAMATH_CALUDE_sum_of_integers_l1445_144504

theorem sum_of_integers (m n : ℕ+) 
  (h1 : m^2 + n^2 = 3789)
  (h2 : Nat.gcd m.val n.val + Nat.lcm m.val n.val = 633) : 
  m + n = 87 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1445_144504


namespace NUMINAMATH_CALUDE_grapefruit_orchards_count_l1445_144508

/-- Calculates the number of grapefruit orchards in a citrus grove. -/
def grapefruit_orchards (total : ℕ) (lemon : ℕ) : ℕ :=
  let orange := lemon / 2
  let remaining := total - (lemon + orange)
  remaining / 2

/-- Proves that the number of grapefruit orchards is 2 given the specified conditions. -/
theorem grapefruit_orchards_count :
  grapefruit_orchards 16 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_grapefruit_orchards_count_l1445_144508


namespace NUMINAMATH_CALUDE_pencils_purchased_correct_l1445_144520

/-- The number of pencils purchased given the conditions of the problem -/
def num_pencils : ℕ := 75

/-- The number of pens purchased -/
def num_pens : ℕ := 30

/-- The price of each pencil -/
def pencil_price : ℚ := 2

/-- The price of each pen -/
def pen_price : ℚ := 10

/-- The total cost of the purchase -/
def total_cost : ℚ := 450

/-- Theorem stating that the number of pencils purchased is correct given the conditions -/
theorem pencils_purchased_correct :
  (num_pens : ℚ) * pen_price + (num_pencils : ℚ) * pencil_price = total_cost :=
sorry

end NUMINAMATH_CALUDE_pencils_purchased_correct_l1445_144520


namespace NUMINAMATH_CALUDE_parallelogram_area_l1445_144546

/-- The area of a parallelogram with base 24 cm and height 16 cm is 384 square centimeters. -/
theorem parallelogram_area : 
  ∀ (base height area : ℝ), 
  base = 24 → height = 16 → area = base * height → area = 384 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_area_l1445_144546


namespace NUMINAMATH_CALUDE_college_student_count_l1445_144553

theorem college_student_count (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : girls = 200) :
  boys + girls = 520 := by
  sorry

end NUMINAMATH_CALUDE_college_student_count_l1445_144553


namespace NUMINAMATH_CALUDE_replaced_sailor_weight_l1445_144595

/-- Given 8 sailors, if replacing one sailor with a new 64 kg sailor increases
    the average weight by 1 kg, then the replaced sailor's weight was 56 kg. -/
theorem replaced_sailor_weight
  (num_sailors : ℕ)
  (new_sailor_weight : ℕ)
  (avg_weight_increase : ℚ)
  (h1 : num_sailors = 8)
  (h2 : new_sailor_weight = 64)
  (h3 : avg_weight_increase = 1)
  : ℕ :=
by
  sorry

#check replaced_sailor_weight

end NUMINAMATH_CALUDE_replaced_sailor_weight_l1445_144595


namespace NUMINAMATH_CALUDE_any_nonzero_to_zero_power_l1445_144528

theorem any_nonzero_to_zero_power (x : ℚ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_any_nonzero_to_zero_power_l1445_144528


namespace NUMINAMATH_CALUDE_intersection_points_existence_and_variability_l1445_144506

/-- The parabola equation -/
def parabola (A : ℝ) (x y : ℝ) : Prop := y = A * x^2

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := y^2 + 2 = x^2 + 6 * y

/-- The line equation -/
def line (x y : ℝ) : Prop := y = 2 * x - 1

/-- The intersection point satisfies all three equations -/
def is_intersection_point (A : ℝ) (x y : ℝ) : Prop :=
  parabola A x y ∧ hyperbola x y ∧ line x y

/-- The theorem stating that there is at least one intersection point and the number can vary -/
theorem intersection_points_existence_and_variability :
  ∀ A : ℝ, A > 0 →
  (∃ x y : ℝ, is_intersection_point A x y) ∧
  (∃ A₁ A₂ : ℝ, A₁ > 0 ∧ A₂ > 0 ∧
    (∃ x₁ y₁ x₂ y₂ x₃ y₃ : ℝ,
      is_intersection_point A₁ x₁ y₁ ∧
      is_intersection_point A₁ x₂ y₂ ∧
      is_intersection_point A₂ x₃ y₃ ∧
      (x₁ ≠ x₂ ∨ y₁ ≠ y₂))) :=
sorry

end NUMINAMATH_CALUDE_intersection_points_existence_and_variability_l1445_144506


namespace NUMINAMATH_CALUDE_valid_coloring_exists_l1445_144519

/-- Represents a 9x9 grid where each cell can be colored or uncolored -/
def Grid := Fin 9 → Fin 9 → Bool

/-- Check if two cells are adjacent (by side or corner) -/
def adjacent (x1 y1 x2 y2 : Fin 9) : Bool :=
  (x1 = x2 ∧ y1.val = y2.val + 1) ∨
  (x1 = x2 ∧ y1.val + 1 = y2.val) ∨
  (x1.val = x2.val + 1 ∧ y1 = y2) ∨
  (x1.val + 1 = x2.val ∧ y1 = y2) ∨
  (x1.val = x2.val + 1 ∧ y1.val = y2.val + 1) ∨
  (x1.val + 1 = x2.val ∧ y1.val + 1 = y2.val) ∨
  (x1.val = x2.val + 1 ∧ y1.val + 1 = y2.val) ∨
  (x1.val + 1 = x2.val ∧ y1.val = y2.val + 1)

/-- Check if a grid coloring is valid -/
def valid_coloring (g : Grid) : Prop :=
  -- Center is not colored
  ¬g 4 4 ∧
  -- No adjacent cells are colored
  (∀ x1 y1 x2 y2, adjacent x1 y1 x2 y2 → ¬(g x1 y1 ∧ g x2 y2)) ∧
  -- Any ray from center intersects a colored cell
  (∀ dx dy, dx ≠ 0 ∨ dy ≠ 0 →
    ∃ t : ℚ, t > 0 ∧ g ⌊4 + t * dx⌋ ⌊4 + t * dy⌋)

/-- Theorem: There exists a valid coloring of the 9x9 grid -/
theorem valid_coloring_exists : ∃ g : Grid, valid_coloring g :=
sorry

end NUMINAMATH_CALUDE_valid_coloring_exists_l1445_144519


namespace NUMINAMATH_CALUDE_hexagon_enclosure_l1445_144571

theorem hexagon_enclosure (m n : ℕ) (h1 : m = 6) (h2 : m + 1 = 7) : 
  (3 * (360 / n) = 2 * (180 - (m - 2) * 180 / m)) → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_enclosure_l1445_144571


namespace NUMINAMATH_CALUDE_first_chapter_pages_l1445_144547

/-- A book with two chapters -/
structure Book where
  total_pages : ℕ
  chapter2_pages : ℕ

/-- The number of pages in the first chapter of a book -/
def pages_in_chapter1 (b : Book) : ℕ := b.total_pages - b.chapter2_pages

/-- Theorem: For a book with 81 total pages and 68 pages in the second chapter,
    the first chapter has 13 pages -/
theorem first_chapter_pages :
  ∀ (b : Book), b.total_pages = 81 → b.chapter2_pages = 68 →
  pages_in_chapter1 b = 13 := by
  sorry

end NUMINAMATH_CALUDE_first_chapter_pages_l1445_144547


namespace NUMINAMATH_CALUDE_boat_speed_ratio_l1445_144556

/-- Proves that the ratio of boat speed in still water to stream speed is 6:1 -/
theorem boat_speed_ratio (still_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  still_speed = 24 →
  downstream_distance = 112 →
  downstream_time = 4 →
  (still_speed / (downstream_distance / downstream_time - still_speed) = 6) :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_ratio_l1445_144556


namespace NUMINAMATH_CALUDE_ball_drawing_probabilities_l1445_144518

/-- The number of white balls in the bag -/
def white_balls : ℕ := 5

/-- The number of black balls in the bag -/
def black_balls : ℕ := 3

/-- The total number of balls in the bag -/
def total_balls : ℕ := white_balls + black_balls

/-- The number of balls drawn -/
def drawn_balls : ℕ := 4

/-- Probability of drawing either 2 or 3 white balls -/
def prob_two_or_three_white : ℚ := 6/7

/-- Probability of drawing at least one black ball -/
def prob_at_least_one_black : ℚ := 13/14

/-- Theorem stating the probabilities of drawing specific combinations of balls -/
theorem ball_drawing_probabilities :
  (prob_two_or_three_white = 6/7) ∧ (prob_at_least_one_black = 13/14) :=
by sorry

end NUMINAMATH_CALUDE_ball_drawing_probabilities_l1445_144518


namespace NUMINAMATH_CALUDE_average_increase_l1445_144592

theorem average_increase (numbers : Finset ℕ) (sum : ℕ) (added_value : ℕ) :
  numbers.card = 15 →
  sum = numbers.sum id →
  sum / numbers.card = 40 →
  added_value = 10 →
  (sum + numbers.card * added_value) / numbers.card = 50 := by
sorry

end NUMINAMATH_CALUDE_average_increase_l1445_144592


namespace NUMINAMATH_CALUDE_line_slope_slope_value_l1445_144583

theorem line_slope (x y : ℝ) :
  x + Real.sqrt 3 * y + 1 = 0 → (y = -(Real.sqrt 3 / 3) * x - (1 / Real.sqrt 3)) := by
  sorry

theorem slope_value :
  let m := -(Real.sqrt 3 / 3)
  ∀ x y : ℝ, x + Real.sqrt 3 * y + 1 = 0 → y = m * x - (1 / Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_line_slope_slope_value_l1445_144583


namespace NUMINAMATH_CALUDE_investment_rate_proof_l1445_144566

theorem investment_rate_proof (initial_investment : ℝ) (initial_rate : ℝ) 
  (additional_investment : ℝ) (additional_rate : ℝ) :
  initial_investment = 8000 →
  initial_rate = 0.05 →
  additional_investment = 4000 →
  additional_rate = 0.08 →
  let total_interest := initial_investment * initial_rate + additional_investment * additional_rate
  let total_investment := initial_investment + additional_investment
  (total_interest / total_investment) = 0.06 := by
  sorry

end NUMINAMATH_CALUDE_investment_rate_proof_l1445_144566


namespace NUMINAMATH_CALUDE_defective_pens_l1445_144538

/-- The number of defective pens in a box of 12 pens, given the probability of selecting two non-defective pens. -/
theorem defective_pens (total : ℕ) (prob : ℚ) (h_total : total = 12) (h_prob : prob = 22727272727272727 / 100000000000000000) :
  ∃ (defective : ℕ), defective = 6 ∧ 
    (prob = (↑(total - defective) / ↑total) * (↑(total - defective - 1) / ↑(total - 1))) :=
by sorry

end NUMINAMATH_CALUDE_defective_pens_l1445_144538


namespace NUMINAMATH_CALUDE_first_discount_percentage_l1445_144521

/-- Proves that the first discount percentage is 10% given the initial price, 
    second discount percentage, and final price after both discounts. -/
theorem first_discount_percentage (initial_price : ℝ) (second_discount : ℝ) (final_price : ℝ) :
  initial_price = 200 →
  second_discount = 5 →
  final_price = 171 →
  ∃ (x : ℝ), 
    (initial_price * (1 - x / 100) * (1 - second_discount / 100) = final_price) ∧
    x = 10 := by
  sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l1445_144521


namespace NUMINAMATH_CALUDE_special_rectangle_sides_l1445_144594

/-- A rectangle with special properties -/
structure SpecialRectangle where
  -- The length of the rectangle
  l : ℝ
  -- The width of the rectangle
  w : ℝ
  -- The perimeter of the rectangle is 24
  perimeter : l + w = 12
  -- M is the midpoint of BC
  midpoint : w / 2 = w / 2
  -- MA is perpendicular to MD
  perpendicular : l ^ 2 + (w / 2) ^ 2 = l ^ 2 + (w / 2) ^ 2

/-- The sides of a special rectangle are 4 and 8 -/
theorem special_rectangle_sides (r : SpecialRectangle) : r.l = 4 ∧ r.w = 8 := by
  sorry

#check special_rectangle_sides

end NUMINAMATH_CALUDE_special_rectangle_sides_l1445_144594


namespace NUMINAMATH_CALUDE_triangle_area_l1445_144564

theorem triangle_area (a b : ℝ) (h1 : a = 8) (h2 : b = 7) : (1/2) * a * b = 28 := by
  sorry

#check triangle_area

end NUMINAMATH_CALUDE_triangle_area_l1445_144564
