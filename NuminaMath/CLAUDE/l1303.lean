import Mathlib

namespace NUMINAMATH_CALUDE_left_placement_equals_100a_plus_b_l1303_130360

/-- A single-digit number is a natural number from 0 to 9 -/
def SingleDigit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

/-- A two-digit number is a natural number from 10 to 99 -/
def TwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The three-digit number formed by placing a to the left of b -/
def LeftPlacement (a b : ℕ) : ℕ := 100 * a + b

theorem left_placement_equals_100a_plus_b (a b : ℕ) 
  (ha : SingleDigit a) (hb : TwoDigit b) : 
  LeftPlacement a b = 100 * a + b := by
  sorry

end NUMINAMATH_CALUDE_left_placement_equals_100a_plus_b_l1303_130360


namespace NUMINAMATH_CALUDE_sum_unit_digit_not_two_l1303_130369

theorem sum_unit_digit_not_two (n : ℕ) : (n * (n + 1) / 2) % 10 ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_unit_digit_not_two_l1303_130369


namespace NUMINAMATH_CALUDE_green_blue_difference_after_border_l1303_130328

/-- Represents the number of tiles in a hexagonal figure --/
structure HexFigure where
  blue : ℕ
  green : ℕ

/-- Adds a double-layer border of green tiles to a hexagonal figure --/
def addDoubleBorder (fig : HexFigure) : HexFigure :=
  { blue := fig.blue,
    green := fig.green + 12 + 18 }

/-- The initial hexagonal figure --/
def initialFigure : HexFigure :=
  { blue := 20, green := 10 }

/-- Theorem stating the difference between green and blue tiles after adding a double border --/
theorem green_blue_difference_after_border :
  let newFigure := addDoubleBorder initialFigure
  (newFigure.green - newFigure.blue) = 20 := by
  sorry

end NUMINAMATH_CALUDE_green_blue_difference_after_border_l1303_130328


namespace NUMINAMATH_CALUDE_euler_conjecture_counterexample_l1303_130355

theorem euler_conjecture_counterexample : 133^5 + 110^5 + 84^5 + 27^5 = 144^5 := by
  sorry

end NUMINAMATH_CALUDE_euler_conjecture_counterexample_l1303_130355


namespace NUMINAMATH_CALUDE_all_statements_imply_not_all_true_l1303_130316

theorem all_statements_imply_not_all_true (p q r : Prop) :
  -- Statement 1
  ((p ∧ q ∧ ¬r) → ¬(p ∧ q ∧ r)) ∧
  -- Statement 2
  ((p ∧ ¬q ∧ r) → ¬(p ∧ q ∧ r)) ∧
  -- Statement 3
  ((¬p ∧ q ∧ ¬r) → ¬(p ∧ q ∧ r)) ∧
  -- Statement 4
  ((¬p ∧ ¬q ∧ ¬r) → ¬(p ∧ q ∧ r)) :=
by sorry


end NUMINAMATH_CALUDE_all_statements_imply_not_all_true_l1303_130316


namespace NUMINAMATH_CALUDE_h2o_mass_formed_l1303_130303

-- Define the chemical reaction
structure Reaction where
  hcl : ℝ
  caco3 : ℝ
  h2o : ℝ

-- Define the molar masses
def molar_mass_h : ℝ := 1.008
def molar_mass_o : ℝ := 15.999

-- Define the reaction stoichiometry
def reaction_stoichiometry (r : Reaction) : Prop :=
  r.hcl = 2 * r.caco3 ∧ r.h2o = r.caco3

-- Calculate the molar mass of H2O
def molar_mass_h2o : ℝ := 2 * molar_mass_h + molar_mass_o

-- Main theorem
theorem h2o_mass_formed (r : Reaction) : 
  reaction_stoichiometry r → r.hcl = 2 → r.caco3 = 1 → r.h2o * molar_mass_h2o = 18.015 :=
sorry

end NUMINAMATH_CALUDE_h2o_mass_formed_l1303_130303


namespace NUMINAMATH_CALUDE_burger_share_inches_l1303_130341

/-- The length of a foot in inches -/
def foot_in_inches : ℝ := 12

/-- The length of the burger in feet -/
def burger_length_feet : ℝ := 1

/-- The number of people sharing the burger -/
def num_people : ℕ := 2

/-- Theorem: Each person's share of a foot-long burger is 6 inches when shared equally between two people -/
theorem burger_share_inches : 
  (burger_length_feet * foot_in_inches) / num_people = 6 := by sorry

end NUMINAMATH_CALUDE_burger_share_inches_l1303_130341


namespace NUMINAMATH_CALUDE_sum_of_non_common_roots_is_zero_l1303_130331

/-- Given two quadratic equations with one common root, prove that the sum of the non-common roots is 0 -/
theorem sum_of_non_common_roots_is_zero (m : ℝ) :
  (∃ x : ℝ, x^2 + (m + 1) * x - 3 = 0 ∧ x^2 - 4 * x - m = 0) →
  (∃ α β γ : ℝ, 
    (α^2 + (m + 1) * α - 3 = 0 ∧ β^2 + (m + 1) * β - 3 = 0 ∧ α ≠ β) ∧
    (α^2 - 4 * α - m = 0 ∧ γ^2 - 4 * γ - m = 0 ∧ α ≠ γ) ∧
    β + γ = 0) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_non_common_roots_is_zero_l1303_130331


namespace NUMINAMATH_CALUDE_village_population_growth_l1303_130332

theorem village_population_growth (
  adult_percentage : Real)
  (child_percentage : Real)
  (employed_adult_percentage : Real)
  (unemployed_adult_percentage : Real)
  (employed_adult_population : ℕ)
  (adult_growth_rate : Real)
  (h1 : adult_percentage = 0.6)
  (h2 : child_percentage = 0.4)
  (h3 : employed_adult_percentage = 0.7)
  (h4 : unemployed_adult_percentage = 0.3)
  (h5 : employed_adult_population = 18000)
  (h6 : adult_growth_rate = 0.05)
  (h7 : adult_percentage + child_percentage = 1)
  (h8 : employed_adult_percentage + unemployed_adult_percentage = 1) :
  ∃ (new_total_population : ℕ), new_total_population = 45000 := by
  sorry

#check village_population_growth

end NUMINAMATH_CALUDE_village_population_growth_l1303_130332


namespace NUMINAMATH_CALUDE_fold_angle_is_36_degrees_l1303_130327

/-- The angle of fold that creates a regular decagon when a piece of paper is folded and cut,
    given that all vertices except one lie on a circle centered at that vertex,
    and the angle between adjacent vertices at the center is 144°. -/
def fold_angle_for_decagon : ℝ := sorry

/-- The internal angle of a regular decagon. -/
def decagon_internal_angle : ℝ := sorry

/-- Theorem stating that the fold angle for creating a regular decagon
    under the given conditions is 36°. -/
theorem fold_angle_is_36_degrees :
  fold_angle_for_decagon = 36 * (π / 180) ∧
  0 < fold_angle_for_decagon ∧
  fold_angle_for_decagon < π / 2 ∧
  decagon_internal_angle = 144 * (π / 180) :=
sorry

end NUMINAMATH_CALUDE_fold_angle_is_36_degrees_l1303_130327


namespace NUMINAMATH_CALUDE_snack_store_spending_l1303_130330

/-- The amount Ben spent at the snack store -/
def ben_spent : ℝ := 60

/-- The amount David spent at the snack store -/
def david_spent : ℝ := 45

/-- For every dollar Ben spent, David spent 25 cents less -/
axiom david_spent_less : david_spent = ben_spent - 0.25 * ben_spent

/-- Ben paid $15 more than David -/
axiom ben_paid_more : ben_spent = david_spent + 15

/-- The total amount spent by Ben and David -/
def total_spent : ℝ := ben_spent + david_spent

theorem snack_store_spending : total_spent = 105 := by
  sorry

end NUMINAMATH_CALUDE_snack_store_spending_l1303_130330


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_of_13_l1303_130380

theorem least_three_digit_multiple_of_13 : ∃ (n : ℕ), 
  n = 104 ∧ 
  13 ∣ n ∧ 
  100 ≤ n ∧ 
  n < 1000 ∧
  ∀ m : ℕ, (13 ∣ m ∧ 100 ≤ m ∧ m < 1000) → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_of_13_l1303_130380


namespace NUMINAMATH_CALUDE_remainder_theorem_l1303_130377

theorem remainder_theorem (y : ℤ) : 
  ∃ (P : ℤ → ℤ), y^50 = (y^2 - 5*y + 6) * P y + (2^50*(y-3) - 3^50*(y-2)) := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1303_130377


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1303_130311

theorem inequality_solution_set (a b c : ℝ) (h1 : a > c) (h2 : b + c > 0) :
  {x : ℝ | (x - c) * (x + b) / (x - a) > 0} = {x : ℝ | -b < x ∧ x < c ∨ x > a} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1303_130311


namespace NUMINAMATH_CALUDE_cross_figure_sum_l1303_130319

-- Define the set of digits
def Digits : Set Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the structure
structure CrossFigure where
  vertical : Fin 3 → Nat
  horizontal1 : Fin 3 → Nat
  horizontal2 : Fin 3 → Nat
  all_digits : List Nat
  h_vertical_sum : vertical 0 + vertical 1 + vertical 2 = 17
  h_horizontal1_sum : horizontal1 0 + horizontal1 1 + horizontal1 2 = 18
  h_horizontal2_sum : horizontal2 0 + horizontal2 1 + horizontal2 2 = 13
  h_intersection1 : vertical 0 = horizontal1 0
  h_intersection2 : vertical 2 = horizontal2 0
  h_all_digits : all_digits.length = 7
  h_all_digits_unique : all_digits.Nodup
  h_all_digits_in_set : ∀ d ∈ all_digits, d ∈ Digits
  h_all_digits_cover : (vertical 0 :: vertical 1 :: vertical 2 :: 
                        horizontal1 1 :: horizontal1 2 :: 
                        horizontal2 1 :: horizontal2 2 :: []).toFinset = all_digits.toFinset

theorem cross_figure_sum (cf : CrossFigure) : 
  cf.all_digits.sum = 34 := by
  sorry

end NUMINAMATH_CALUDE_cross_figure_sum_l1303_130319


namespace NUMINAMATH_CALUDE_winning_candidate_percentage_l1303_130374

/-- Given an election with a total of 5200 votes where the winning candidate
    has a majority of 1040 votes, prove that the winning candidate received 60% of the votes. -/
theorem winning_candidate_percentage (total_votes : ℕ) (majority : ℕ) 
  (h_total : total_votes = 5200)
  (h_majority : majority = 1040) :
  (majority : ℚ) / total_votes * 100 + 50 = 60 := by
  sorry

end NUMINAMATH_CALUDE_winning_candidate_percentage_l1303_130374


namespace NUMINAMATH_CALUDE_investment_interest_calculation_l1303_130381

theorem investment_interest_calculation (total_investment : ℝ) (first_investment : ℝ) 
  (first_rate : ℝ) (second_rate : ℝ) (h1 : total_investment = 10000) 
  (h2 : first_investment = 6000) (h3 : first_rate = 0.09) (h4 : second_rate = 0.11) : 
  first_investment * first_rate + (total_investment - first_investment) * second_rate = 980 :=
by
  sorry

end NUMINAMATH_CALUDE_investment_interest_calculation_l1303_130381


namespace NUMINAMATH_CALUDE_manuscript_pages_count_l1303_130361

/-- The cost structure and revision information for a manuscript typing service. -/
structure ManuscriptTyping where
  first_time_cost : ℕ
  revision_cost : ℕ
  pages_revised_once : ℕ
  pages_revised_twice : ℕ
  total_cost : ℕ

/-- Calculates the total number of pages in a manuscript given the typing costs and revision information. -/
def total_pages (m : ManuscriptTyping) : ℕ :=
  (m.total_cost - (m.pages_revised_once * (m.first_time_cost + m.revision_cost) + 
   m.pages_revised_twice * (m.first_time_cost + 2 * m.revision_cost))) / m.first_time_cost + 
   m.pages_revised_once + m.pages_revised_twice

/-- Theorem stating that for the given manuscript typing scenario, the total number of pages is 100. -/
theorem manuscript_pages_count (m : ManuscriptTyping) 
  (h1 : m.first_time_cost = 6)
  (h2 : m.revision_cost = 4)
  (h3 : m.pages_revised_once = 35)
  (h4 : m.pages_revised_twice = 15)
  (h5 : m.total_cost = 860) :
  total_pages m = 100 := by
  sorry

end NUMINAMATH_CALUDE_manuscript_pages_count_l1303_130361


namespace NUMINAMATH_CALUDE_quadratic_non_real_roots_l1303_130375

/-- A quadratic equation x^2 + bx + 16 has two non-real roots if and only if b is in the open interval (-8, 8) -/
theorem quadratic_non_real_roots (b : ℝ) : 
  (∀ x : ℂ, x^2 + b*x + 16 = 0 → x.im ≠ 0) ↔ -8 < b ∧ b < 8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_non_real_roots_l1303_130375


namespace NUMINAMATH_CALUDE_smallest_satisfying_congruences_l1303_130334

theorem smallest_satisfying_congruences : 
  ∃ (x : ℕ), x > 0 ∧ 
    x % 3 = 2 ∧ 
    x % 5 = 3 ∧ 
    x % 7 = 2 ∧
    (∀ (y : ℕ), y > 0 ∧ y % 3 = 2 ∧ y % 5 = 3 ∧ y % 7 = 2 → x ≤ y) ∧
    x = 23 :=
by sorry

end NUMINAMATH_CALUDE_smallest_satisfying_congruences_l1303_130334


namespace NUMINAMATH_CALUDE_unique_fixed_point_l1303_130354

-- Define the type for points in the plane
variable (Point : Type)

-- Define the type for lines in the plane
variable (Line : Type)

-- Define the set of all lines in the plane
variable (L : Set Line)

-- Define the function f that assigns a point to each line
variable (f : Line → Point)

-- Define a predicate to check if a point is on a line
variable (on_line : Point → Line → Prop)

-- Define a predicate to check if points are on a circle
variable (on_circle : Point → Point → Point → Point → Prop)

-- Axiom: f(l) is on l for all lines l
axiom f_on_line : ∀ l : Line, on_line (f l) l

-- Axiom: For any point X and any three lines l1, l2, l3 passing through X,
--        the points f(l1), f(l2), f(l3), and X lie on a circle
axiom circle_property : 
  ∀ (X : Point) (l1 l2 l3 : Line),
  on_line X l1 → on_line X l2 → on_line X l3 →
  on_circle X (f l1) (f l2) (f l3)

-- Theorem: There exists a unique point P such that f(l) = P for any line l passing through P
theorem unique_fixed_point :
  ∃! P : Point, ∀ l : Line, on_line P l → f l = P :=
sorry

end NUMINAMATH_CALUDE_unique_fixed_point_l1303_130354


namespace NUMINAMATH_CALUDE_parabola_transformation_l1303_130300

/-- Represents a parabola of the form y = (x + a)^2 + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- Applies a horizontal shift to a parabola -/
def horizontal_shift (p : Parabola) (shift : ℝ) : Parabola :=
  { a := p.a - shift, b := p.b }

/-- Applies a vertical shift to a parabola -/
def vertical_shift (p : Parabola) (shift : ℝ) : Parabola :=
  { a := p.a, b := p.b + shift }

/-- The initial parabola y = (x + 2)^2 + 3 -/
def initial_parabola : Parabola := { a := 2, b := 3 }

/-- The final parabola after transformations -/
def final_parabola : Parabola := { a := -1, b := 1 }

theorem parabola_transformation :
  (vertical_shift (horizontal_shift initial_parabola 3) (-2)) = final_parabola := by
  sorry

end NUMINAMATH_CALUDE_parabola_transformation_l1303_130300


namespace NUMINAMATH_CALUDE_no_four_digit_number_equals_46_10X_plus_Y_l1303_130393

theorem no_four_digit_number_equals_46_10X_plus_Y :
  ¬ ∃ (X Y : ℕ) (a b c d : ℕ),
    (a = 4 ∨ a = 6 ∨ a = X ∨ a = Y) ∧
    (b = 4 ∨ b = 6 ∨ b = X ∨ b = Y) ∧
    (c = 4 ∨ c = 6 ∨ c = X ∨ c = Y) ∧
    (d = 4 ∨ d = 6 ∨ d = X ∨ d = Y) ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    1000 ≤ 1000 * a + 100 * b + 10 * c + d ∧
    1000 * a + 100 * b + 10 * c + d < 10000 ∧
    1000 * a + 100 * b + 10 * c + d = 46 * (10 * X + Y) :=
by sorry

end NUMINAMATH_CALUDE_no_four_digit_number_equals_46_10X_plus_Y_l1303_130393


namespace NUMINAMATH_CALUDE_solutions_for_20_l1303_130382

/-- The number of distinct integer solutions (x, y) for |x| + |y| = n -/
def num_solutions (n : ℕ) : ℕ := sorry

/-- The theorem stating the number of solutions for n = 20 -/
theorem solutions_for_20 :
  num_solutions 1 = 4 →
  num_solutions 2 = 8 →
  num_solutions 3 = 12 →
  num_solutions 20 = 80 := by sorry

end NUMINAMATH_CALUDE_solutions_for_20_l1303_130382


namespace NUMINAMATH_CALUDE_sum_plus_difference_l1303_130368

theorem sum_plus_difference (a b c : ℝ) (h : c = a + b + 5.1) : c = 48.9 :=
  by sorry

#check sum_plus_difference 20.2 33.8 48.9

end NUMINAMATH_CALUDE_sum_plus_difference_l1303_130368


namespace NUMINAMATH_CALUDE_smallest_difference_l1303_130336

/-- Vovochka's sum method for three-digit numbers -/
def vovochka_sum (a b c d e f : ℕ) : ℕ :=
  1000 * (a + d) + 100 * (b + e) + (c + f)

/-- Correct sum method for three-digit numbers -/
def correct_sum (a b c d e f : ℕ) : ℕ :=
  100 * (a + d) + 10 * (b + e) + (c + f)

/-- The difference between Vovochka's sum and the correct sum -/
def sum_difference (a b c d e f : ℕ) : ℕ :=
  vovochka_sum a b c d e f - correct_sum a b c d e f

theorem smallest_difference :
  ∀ a b c d e f : ℕ,
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 →
    sum_difference a b c d e f > 0 →
    sum_difference a b c d e f ≥ 1800 :=
sorry

end NUMINAMATH_CALUDE_smallest_difference_l1303_130336


namespace NUMINAMATH_CALUDE_parabola_circle_intersection_l1303_130308

/-- Given a parabola y^2 = 2px (p > 0) and a point A(t, 0) (t > 0), 
    a line through A intersects the parabola at B and C. 
    Lines OB and OC intersect the line x = -t at M and N respectively. 
    This theorem states that the circle with diameter MN intersects 
    the x-axis at two fixed points. -/
theorem parabola_circle_intersection 
  (p t : ℝ) 
  (hp : p > 0) 
  (ht : t > 0) : 
  ∃ (x₁ x₂ : ℝ), 
    x₁ = -t + Real.sqrt (2 * p * t) ∧ 
    x₂ = -t - Real.sqrt (2 * p * t) ∧ 
    (∀ (x y : ℝ), 
      (x + t)^2 + y^2 = (x₁ + t)^2 → 
      y = 0 → x = x₁ ∨ x = x₂) :=
by sorry


end NUMINAMATH_CALUDE_parabola_circle_intersection_l1303_130308


namespace NUMINAMATH_CALUDE_worker_travel_time_l1303_130309

theorem worker_travel_time (normal_speed : ℝ) (normal_time : ℝ) 
  (h1 : normal_speed > 0) (h2 : normal_time > 0) : 
  (3/4 * normal_speed) * (normal_time + 8) = normal_speed * normal_time → 
  normal_time = 24 := by
  sorry

end NUMINAMATH_CALUDE_worker_travel_time_l1303_130309


namespace NUMINAMATH_CALUDE_not_in_second_quadrant_l1303_130333

-- Define the linear function
def f (x : ℝ) : ℝ := x - 1

-- Define the second quadrant
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- Theorem statement
theorem not_in_second_quadrant :
  ∀ x : ℝ, ¬(second_quadrant x (f x)) := by
  sorry

end NUMINAMATH_CALUDE_not_in_second_quadrant_l1303_130333


namespace NUMINAMATH_CALUDE_enrollment_ways_count_l1303_130398

/-- The number of elective courses -/
def num_courses : ℕ := 4

/-- The number of students -/
def num_students : ℕ := 3

/-- The number of courses each student must choose -/
def courses_per_student : ℕ := 2

/-- The number of different ways each course can have students enrolled -/
def num_enrollment_ways : ℕ := 114

theorem enrollment_ways_count :
  (num_courses = 4) →
  (num_students = 3) →
  (courses_per_student = 2) →
  (num_enrollment_ways = 114) := by
  sorry

end NUMINAMATH_CALUDE_enrollment_ways_count_l1303_130398


namespace NUMINAMATH_CALUDE_rationalize_denominator_seven_sqrt_147_l1303_130337

theorem rationalize_denominator_seven_sqrt_147 :
  ∃ (a b : ℝ) (h : b ≠ 0), (7 / Real.sqrt 147) = (a * Real.sqrt b) / b :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_seven_sqrt_147_l1303_130337


namespace NUMINAMATH_CALUDE_polynomial_sum_equals_256_l1303_130339

theorem polynomial_sum_equals_256 
  (a a₁ a₂ a₃ a₄ : ℝ) 
  (h : ∀ x, (3 - x)^4 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) :
  a - a₁ + a₂ - a₃ + a₄ = 256 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_equals_256_l1303_130339


namespace NUMINAMATH_CALUDE_roots_product_l1303_130340

/-- Given that x₁ = ∛(17 - (27/4)√6) and x₂ = ∛(17 + (27/4)√6) are roots of x² - ax + b = 0, prove that ab = 10 -/
theorem roots_product (a b : ℝ) : 
  let x₁ : ℝ := (17 - (27/4) * Real.sqrt 6) ^ (1/3)
  let x₂ : ℝ := (17 + (27/4) * Real.sqrt 6) ^ (1/3)
  (x₁ ^ 2 - a * x₁ + b = 0) ∧ (x₂ ^ 2 - a * x₂ + b = 0) → a * b = 10 := by
  sorry


end NUMINAMATH_CALUDE_roots_product_l1303_130340


namespace NUMINAMATH_CALUDE_five_heads_before_two_tails_l1303_130343

/-- The probability of getting 5 heads before 2 consecutive tails when repeatedly flipping a fair coin -/
def probability_5H_before_2T : ℚ :=
  3 / 34

/-- A fair coin has equal probability of heads and tails -/
def fair_coin (p : ℚ → Prop) : Prop :=
  p (1/2) ∧ p (1/2)

theorem five_heads_before_two_tails (p : ℚ → Prop) (h : fair_coin p) :
  probability_5H_before_2T = 3 / 34 :=
sorry

end NUMINAMATH_CALUDE_five_heads_before_two_tails_l1303_130343


namespace NUMINAMATH_CALUDE_modular_congruence_solution_l1303_130322

theorem modular_congruence_solution : ∃ n : ℤ, 0 ≤ n ∧ n < 23 ∧ -250 ≡ n [ZMOD 23] ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_solution_l1303_130322


namespace NUMINAMATH_CALUDE_smallest_dual_palindrome_l1303_130326

/-- Checks if a number is a palindrome in a given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Bool := sorry

/-- Converts a number from base 10 to another base -/
def toBase (n : ℕ) (base : ℕ) : List ℕ := sorry

theorem smallest_dual_palindrome : 
  ∀ n : ℕ, n > 6 → (isPalindrome n 3 ∧ isPalindrome n 5) → n ≥ 26 := by
  sorry

#check smallest_dual_palindrome

end NUMINAMATH_CALUDE_smallest_dual_palindrome_l1303_130326


namespace NUMINAMATH_CALUDE_winner_percentage_approx_62_l1303_130370

/-- Represents an election with two candidates -/
structure Election :=
  (total_votes : ℕ)
  (winner_votes : ℕ)
  (margin : ℕ)

/-- Calculates the percentage of votes for the winner -/
def winner_percentage (e : Election) : ℚ :=
  (e.winner_votes : ℚ) / (e.total_votes : ℚ) * 100

/-- Theorem stating the winner's percentage in the given election -/
theorem winner_percentage_approx_62 (e : Election) 
  (h1 : e.winner_votes = 837)
  (h2 : e.margin = 324)
  (h3 : e.total_votes = e.winner_votes + (e.winner_votes - e.margin)) :
  ∃ (p : ℚ), abs (winner_percentage e - p) < 1 ∧ p = 62 := by
  sorry

#eval winner_percentage { total_votes := 1350, winner_votes := 837, margin := 324 }

end NUMINAMATH_CALUDE_winner_percentage_approx_62_l1303_130370


namespace NUMINAMATH_CALUDE_total_books_l1303_130362

theorem total_books (joan_books tom_books sarah_books alex_books : ℕ) 
  (h1 : joan_books = 10)
  (h2 : tom_books = 38)
  (h3 : sarah_books = 25)
  (h4 : alex_books = 45) :
  joan_books + tom_books + sarah_books + alex_books = 118 := by
  sorry

end NUMINAMATH_CALUDE_total_books_l1303_130362


namespace NUMINAMATH_CALUDE_average_speed_calculation_l1303_130392

/-- Calculates the average speed of a trip given the conditions specified in the problem -/
theorem average_speed_calculation (total_distance : ℝ) (first_half_speed : ℝ) (second_half_time_factor : ℝ) :
  total_distance = 640 →
  first_half_speed = 80 →
  second_half_time_factor = 3 →
  (total_distance / (total_distance / (2 * first_half_speed) * (1 + second_half_time_factor))) = 40 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l1303_130392


namespace NUMINAMATH_CALUDE_product_of_numbers_l1303_130346

theorem product_of_numbers (x y : ℝ) (h1 : x - y = 6) (h2 : x^2 + y^2 = 170) : x * y = -67 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l1303_130346


namespace NUMINAMATH_CALUDE_shopping_expenditure_l1303_130347

/-- Represents the percentage spent on clothing -/
def clothing_percentage : ℝ := sorry

/-- Represents the percentage spent on food -/
def food_percentage : ℝ := 20

/-- Represents the percentage spent on other items -/
def other_percentage : ℝ := 30

/-- Represents the tax rate on clothing -/
def clothing_tax_rate : ℝ := 4

/-- Represents the tax rate on other items -/
def other_tax_rate : ℝ := 8

/-- Represents the total tax rate as a percentage of pre-tax spending -/
def total_tax_rate : ℝ := 4.4

theorem shopping_expenditure :
  clothing_percentage + food_percentage + other_percentage = 100 ∧
  clothing_percentage * clothing_tax_rate / 100 + other_percentage * other_tax_rate / 100 = total_tax_rate ∧
  clothing_percentage = 50 := by sorry

end NUMINAMATH_CALUDE_shopping_expenditure_l1303_130347


namespace NUMINAMATH_CALUDE_solution_set_subset_interval_l1303_130399

def solution_set (a : ℝ) : Set ℝ :=
  {x | x^2 - 2*a*x + a + 2 ≤ 0}

theorem solution_set_subset_interval (a : ℝ) :
  solution_set a ⊆ Set.Icc 1 4 ↔ a ∈ Set.Ioo (-1) (18/7) :=
sorry

end NUMINAMATH_CALUDE_solution_set_subset_interval_l1303_130399


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_l1303_130356

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (6 * x) % 31 = 19 % 31 ∧ 
  ∀ (y : ℕ), y > 0 → (6 * y) % 31 = 19 % 31 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_l1303_130356


namespace NUMINAMATH_CALUDE_parallel_not_sufficient_nor_necessary_l1303_130323

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relationships between lines and planes
def parallel (m : Line) (α : Plane) : Prop := sorry
def perpendicular (m : Line) (β : Plane) : Prop := sorry
def planes_perpendicular (α β : Plane) : Prop := sorry

-- Theorem statement
theorem parallel_not_sufficient_nor_necessary 
  (m : Line) (α β : Plane) 
  (h_perp : planes_perpendicular α β) :
  ¬(∀ m α β, parallel m α → perpendicular m β) ∧ 
  ¬(∀ m α β, perpendicular m β → parallel m α) := by
  sorry

end NUMINAMATH_CALUDE_parallel_not_sufficient_nor_necessary_l1303_130323


namespace NUMINAMATH_CALUDE_five_percent_difference_l1303_130306

theorem five_percent_difference (x y : ℝ) 
  (hx : 5 = 0.25 * x) 
  (hy : 5 = 0.50 * y) : 
  x - y = 10 := by
sorry

end NUMINAMATH_CALUDE_five_percent_difference_l1303_130306


namespace NUMINAMATH_CALUDE_all_statements_false_l1303_130302

theorem all_statements_false :
  (∀ x : ℝ, x^2 = 4 → x = 2 ∨ x = -2) ∧
  (∀ x : ℝ, x^2 = 9 → x = 3 ∨ x = -3) ∧
  (∀ x : ℝ, x^3 = -125 → x = -5) ∧
  (∀ x : ℝ, x^2 = 16 → x = 4 ∨ x = -4) :=
by sorry

end NUMINAMATH_CALUDE_all_statements_false_l1303_130302


namespace NUMINAMATH_CALUDE_intersection_M_N_l1303_130310

-- Define set M
def M : Set ℝ := {x | x^2 - 2*x < 0}

-- Define set N
def N : Set ℝ := {x | |x| < 1}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1303_130310


namespace NUMINAMATH_CALUDE_bird_watching_percentage_difference_l1303_130315

-- Define the number of birds seen by each person
def gabrielle_robins : ℕ := 5
def gabrielle_cardinals : ℕ := 4
def gabrielle_blue_jays : ℕ := 3

def chase_robins : ℕ := 2
def chase_cardinals : ℕ := 5
def chase_blue_jays : ℕ := 3

-- Calculate total birds seen by each person
def gabrielle_total : ℕ := gabrielle_robins + gabrielle_cardinals + gabrielle_blue_jays
def chase_total : ℕ := chase_robins + chase_cardinals + chase_blue_jays

-- Define the percentage difference
def percentage_difference : ℚ := (gabrielle_total - chase_total : ℚ) / chase_total * 100

-- Theorem statement
theorem bird_watching_percentage_difference :
  percentage_difference = 20 :=
sorry

end NUMINAMATH_CALUDE_bird_watching_percentage_difference_l1303_130315


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1303_130344

def A : Set ℝ := {x | |x| < 1}
def B : Set ℝ := {x | -2 < x ∧ x < 0}

theorem intersection_of_A_and_B : A ∩ B = Set.Ioo (-1) 0 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1303_130344


namespace NUMINAMATH_CALUDE_functional_equation_characterization_l1303_130359

theorem functional_equation_characterization
  (a : ℤ) (ha : a ≠ 0)
  (f g : ℚ → ℚ)
  (h : ∀ x y : ℚ, f (x + g y) = g x + f y + a * y) :
  (∃ n : ℤ, n ≠ 0 ∧ n ≠ 1 ∧ a = n^2 - n) ∧
  (∃ n : ℤ, ∃ v : ℚ, (n ≠ 0 ∧ n ≠ 1) ∧
    ((∀ x : ℚ, f x = n * x + v ∧ g x = n * x) ∨
     (∀ x : ℚ, f x = (1 - n) * x + v ∧ g x = (1 - n) * x))) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_characterization_l1303_130359


namespace NUMINAMATH_CALUDE_spears_from_log_l1303_130397

/-- The number of spears Marcy can make from a sapling -/
def spears_from_sapling : ℕ := 3

/-- The total number of spears Marcy can make from 6 saplings and a log -/
def total_spears : ℕ := 27

/-- The number of saplings used -/
def num_saplings : ℕ := 6

/-- Theorem: Marcy can make 9 spears from a single log -/
theorem spears_from_log : 
  ∃ (L : ℕ), L = total_spears - (num_saplings * spears_from_sapling) ∧ L = 9 :=
by sorry

end NUMINAMATH_CALUDE_spears_from_log_l1303_130397


namespace NUMINAMATH_CALUDE_equal_area_rectangles_length_l1303_130304

/-- Given two rectangles of equal area, where one rectangle has dimensions 2 inches by 60 inches,
    and the other has a width of 24 inches, prove that the length of the second rectangle is 5 inches. -/
theorem equal_area_rectangles_length (l : ℝ) :
  (2 : ℝ) * 60 = l * 24 → l = 5 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_length_l1303_130304


namespace NUMINAMATH_CALUDE_sqrt_two_irrational_l1303_130394

-- Define what it means for a real number to be rational
def IsRational (x : ℝ) : Prop :=
  ∃ (n d : ℤ), d ≠ 0 ∧ x = n / d

-- Define what it means for a real number to be irrational
def IsIrrational (x : ℝ) : Prop := ¬(IsRational x)

-- Theorem statement
theorem sqrt_two_irrational : IsIrrational (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_irrational_l1303_130394


namespace NUMINAMATH_CALUDE_pet_store_gerbils_l1303_130349

/-- The number of gerbils left in a pet store after some are sold -/
def gerbils_left (initial : ℕ) (sold : ℕ) : ℕ := initial - sold

/-- Theorem: Given 85 initial gerbils and 69 sold, 16 gerbils are left -/
theorem pet_store_gerbils : gerbils_left 85 69 = 16 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_gerbils_l1303_130349


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l1303_130312

theorem rectangular_box_volume (l w h : ℝ) 
  (face1 : l * w = 30)
  (face2 : w * h = 20)
  (face3 : l * h = 12) :
  l * w * h = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l1303_130312


namespace NUMINAMATH_CALUDE_parabola_directrix_l1303_130386

/-- The directrix of a parabola x^2 + 12y = 0 -/
theorem parabola_directrix (x y : ℝ) : 
  (∀ x y, x^2 + 12*y = 0) → (∃ k : ℝ, k = 3 ∧ y = k) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1303_130386


namespace NUMINAMATH_CALUDE_divisible_by_nine_l1303_130364

/-- The eight-digit number in the form 973m2158 -/
def eight_digit_number (m : ℕ) : ℕ := 973000000 + m * 10000 + 2158

/-- A number is divisible by 9 if and only if the sum of its digits is divisible by 9 -/
def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100000000) + ((n / 10000000) % 10) + ((n / 1000000) % 10) + 
  ((n / 100000) % 10) + ((n / 10000) % 10) + ((n / 1000) % 10) + 
  ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem divisible_by_nine (m : ℕ) : 
  (eight_digit_number m) % 9 = 0 ↔ m = 1 :=
sorry

end NUMINAMATH_CALUDE_divisible_by_nine_l1303_130364


namespace NUMINAMATH_CALUDE_ravish_exam_marks_l1303_130329

theorem ravish_exam_marks (pass_percentage : ℚ) (max_marks : ℕ) (fail_margin : ℕ) : 
  pass_percentage = 40 / 100 →
  max_marks = 200 →
  fail_margin = 40 →
  (pass_percentage * max_marks : ℚ) - fail_margin = 40 := by
  sorry

end NUMINAMATH_CALUDE_ravish_exam_marks_l1303_130329


namespace NUMINAMATH_CALUDE_danny_bottle_caps_l1303_130353

theorem danny_bottle_caps (initial : ℕ) (found : ℕ) (current : ℕ) (thrown_away : ℕ) : 
  initial = 69 → found = 58 → current = 67 → 
  thrown_away = initial + found - current →
  thrown_away = 60 := by
sorry

end NUMINAMATH_CALUDE_danny_bottle_caps_l1303_130353


namespace NUMINAMATH_CALUDE_four_point_segment_ratio_l1303_130348

/-- Given four distinct points on a plane with segment lengths a, a, a, a, 2a, and b,
    prove that b = a√3 -/
theorem four_point_segment_ratio (a b : ℝ) :
  ∃ (A B C D : ℝ × ℝ),
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    ({dist A B, dist A C, dist A D, dist B C, dist B D, dist C D} : Finset ℝ) =
      {a, a, a, a, 2*a, b} →
    b = a * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_four_point_segment_ratio_l1303_130348


namespace NUMINAMATH_CALUDE_amaya_total_marks_l1303_130363

/-- Represents the marks scored in different subjects -/
structure Marks where
  music : ℕ
  socialStudies : ℕ
  arts : ℕ
  maths : ℕ

/-- Calculates the total marks across all subjects -/
def totalMarks (m : Marks) : ℕ := m.music + m.socialStudies + m.arts + m.maths

/-- Theorem stating the total marks Amaya scored given the conditions -/
theorem amaya_total_marks :
  ∀ (m : Marks),
    m.music = 70 →
    m.socialStudies = m.music + 10 →
    m.maths = m.arts - 20 →
    m.maths = (9 : ℕ) * m.arts / 10 →
    totalMarks m = 530 := by
  sorry

#check amaya_total_marks

end NUMINAMATH_CALUDE_amaya_total_marks_l1303_130363


namespace NUMINAMATH_CALUDE_box_area_product_equals_volume_squared_l1303_130388

/-- Given a rectangular box with dimensions x, y, and z, 
    prove that the product of the areas of its three pairs of opposite faces 
    is equal to the square of its volume. -/
theorem box_area_product_equals_volume_squared 
  (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y) * (y * z) * (z * x) = (x * y * z)^2 := by
  sorry

end NUMINAMATH_CALUDE_box_area_product_equals_volume_squared_l1303_130388


namespace NUMINAMATH_CALUDE_fraction_meaningful_implies_x_not_one_l1303_130325

-- Define a function that represents the meaningfulness of the fraction 1/(x-1)
def is_meaningful (x : ℝ) : Prop := x ≠ 1

-- Theorem statement
theorem fraction_meaningful_implies_x_not_one :
  ∀ x : ℝ, is_meaningful x → x ≠ 1 :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_meaningful_implies_x_not_one_l1303_130325


namespace NUMINAMATH_CALUDE_lines_intersect_implies_planes_intersect_l1303_130379

-- Define the space
variable (S : Type*) [NormedAddCommGroup S] [InnerProductSpace ℝ S] [CompleteSpace S]

-- Define lines and planes
def Line (S : Type*) [NormedAddCommGroup S] := Set S
def Plane (S : Type*) [NormedAddCommGroup S] := Set S

-- Define the subset relation
def IsSubset {S : Type*} (A B : Set S) := A ⊆ B

-- Define intersection for lines and planes
def Intersect {S : Type*} (A B : Set S) := ∃ x, x ∈ A ∧ x ∈ B

-- Theorem statement
theorem lines_intersect_implies_planes_intersect
  (m n : Line S) (α β : Plane S)
  (hm : m ≠ n) (hα : α ≠ β)
  (hmα : IsSubset m α) (hnβ : IsSubset n β)
  (hmn : Intersect m n) :
  Intersect α β :=
sorry

end NUMINAMATH_CALUDE_lines_intersect_implies_planes_intersect_l1303_130379


namespace NUMINAMATH_CALUDE_regular_octagon_diagonal_ratio_regular_octagon_diagonal_ratio_proof_l1303_130376

/-- The ratio of the shorter diagonal to the longer diagonal in a regular octagon -/
theorem regular_octagon_diagonal_ratio : ℝ :=
  1 / Real.sqrt 2

/-- Proof that the ratio of the shorter diagonal to the longer diagonal in a regular octagon is 1 / √2 -/
theorem regular_octagon_diagonal_ratio_proof :
  regular_octagon_diagonal_ratio = 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_diagonal_ratio_regular_octagon_diagonal_ratio_proof_l1303_130376


namespace NUMINAMATH_CALUDE_cricketer_average_score_l1303_130367

theorem cricketer_average_score (avg1 avg2 overall_avg : ℚ) (n1 n2 : ℕ) : 
  avg1 = 30 → avg2 = 40 → overall_avg = 36 → n1 = 2 → n2 = 3 →
  (n1 * avg1 + n2 * avg2) / (n1 + n2 : ℚ) = overall_avg →
  n1 + n2 = 5 := by sorry

end NUMINAMATH_CALUDE_cricketer_average_score_l1303_130367


namespace NUMINAMATH_CALUDE_ellipse_m_range_collinearity_AGN_l1303_130365

-- Define the curve C
def C (m : ℝ) (x y : ℝ) : Prop := (5 - m) * x^2 + (m - 2) * y^2 = 8

-- Define the condition for C to be an ellipse with foci on x-axis
def is_ellipse_x_foci (m : ℝ) : Prop :=
  (8 / (5 - m) > 8 / (m - 2)) ∧ (8 / (5 - m) > 0) ∧ (8 / (m - 2) > 0)

-- Define the line y = kx + 4
def line_k (k : ℝ) (x y : ℝ) : Prop := y = k * x + 4

-- Define the line y = 1
def line_one (x y : ℝ) : Prop := y = 1

-- Theorem for part 1
theorem ellipse_m_range (m : ℝ) :
  is_ellipse_x_foci m → (7/2 < m) ∧ (m < 5) := by sorry

-- Theorem for part 2
theorem collinearity_AGN (k : ℝ) (xA yA xB yB xM yM xN yN xG : ℝ) :
  C 4 0 yA ∧ C 4 0 yB ∧ yA > yB ∧
  C 4 xM yM ∧ C 4 xN yN ∧
  line_k k xM yM ∧ line_k k xN yN ∧
  line_one xG 1 ∧
  (yM - yB) / (xM - xB) = (1 - yB) / (xG - xB) →
  ∃ (t : ℝ), xG = t * xA + (1 - t) * xN ∧ 1 = t * yA + (1 - t) * yN := by sorry

end NUMINAMATH_CALUDE_ellipse_m_range_collinearity_AGN_l1303_130365


namespace NUMINAMATH_CALUDE_only_rainbow_statement_correct_l1303_130357

/-- Represents the conditions for seeing a rainbow --/
structure RainbowConditions :=
  (sunlight : Bool)
  (rain : Bool)
  (observer_position : ℝ × ℝ × ℝ)

/-- Represents the outcome of a coin flip --/
inductive CoinFlip
  | Heads
  | Tails

/-- Represents the precipitation data for a city --/
structure PrecipitationData :=
  (average : ℝ)
  (variance : ℝ)

/-- The set of statements about random events and statistical concepts --/
inductive Statement
  | RainbowRandomEvent
  | AircraftRandomSampling
  | CoinFlipDeterministic
  | PrecipitationStability

/-- Function to determine if seeing a rainbow is random given the conditions --/
def is_rainbow_random (conditions : RainbowConditions) : Prop :=
  ∃ (c1 c2 : RainbowConditions), c1 ≠ c2 ∧ 
    (c1.sunlight ∧ c1.rain) ∧ (c2.sunlight ∧ c2.rain) ∧ 
    c1.observer_position ≠ c2.observer_position

/-- Function to determine if a statement is correct --/
def is_correct_statement (s : Statement) : Prop :=
  match s with
  | Statement.RainbowRandomEvent => ∀ c, is_rainbow_random c
  | _ => False

/-- Theorem stating that only the rainbow statement is correct --/
theorem only_rainbow_statement_correct :
  ∀ s, is_correct_statement s ↔ s = Statement.RainbowRandomEvent :=
sorry

end NUMINAMATH_CALUDE_only_rainbow_statement_correct_l1303_130357


namespace NUMINAMATH_CALUDE_min_value_of_sum_fractions_l1303_130378

theorem min_value_of_sum_fractions (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b) / c + (a + c) / b + (b + c) / a ≥ 6 ∧
  ((a + b) / c + (a + c) / b + (b + c) / a = 6 ↔ a = b ∧ b = c) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_sum_fractions_l1303_130378


namespace NUMINAMATH_CALUDE_rectangle_ratio_l1303_130324

theorem rectangle_ratio (s : ℝ) (x y : ℝ) (h1 : s > 0) (h2 : x > 0) (h3 : y > 0) : 
  (s + 2*y = 3*s) → (x + s = 3*s) → (x/y = 2) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l1303_130324


namespace NUMINAMATH_CALUDE_right_triangle_circle_theorem_l1303_130373

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle with vertices P, Q, and R -/
structure Triangle :=
  (P : Point)
  (Q : Point)
  (R : Point)

/-- Checks if a triangle is right-angled at Q -/
def isRightTriangle (t : Triangle) : Prop :=
  -- Definition of right triangle at Q
  sorry

/-- Checks if a point S lies on the circle with diameter QR -/
def isOnCircle (t : Triangle) (S : Point) : Prop :=
  -- Definition of S being on the circle with diameter QR
  sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ :=
  -- Definition of distance between two points
  sorry

/-- Main theorem -/
theorem right_triangle_circle_theorem (t : Triangle) (S : Point) :
  isRightTriangle t →
  isOnCircle t S →
  distance t.P S = 3 →
  distance t.Q S = 9 →
  distance t.R S = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_circle_theorem_l1303_130373


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l1303_130320

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

/-- The factorial of a natural number n, denoted as n!, is the product of all positive integers less than or equal to n. -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem geometric_sequence_first_term
  (a : ℕ → ℝ)
  (h_geom : IsGeometricSequence a)
  (h_sixth : a 6 = factorial 9)
  (h_ninth : a 9 = factorial 10) :
  a 1 = (factorial 9 : ℝ) / (10 ^ (5/3)) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l1303_130320


namespace NUMINAMATH_CALUDE_scientific_notation_35000_l1303_130352

theorem scientific_notation_35000 :
  35000 = 3.5 * (10 ^ 4) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_35000_l1303_130352


namespace NUMINAMATH_CALUDE_angle_with_parallel_sides_l1303_130358

-- Define the concept of parallel angles
def parallel_angles (A B : Real) : Prop := sorry

-- Theorem statement
theorem angle_with_parallel_sides (A B : Real) :
  parallel_angles A B → A = 45 → (B = 45 ∨ B = 135) := by
  sorry

end NUMINAMATH_CALUDE_angle_with_parallel_sides_l1303_130358


namespace NUMINAMATH_CALUDE_polynomial_property_l1303_130301

-- Define the polynomial Q(x)
def Q (x a b c : ℝ) : ℝ := 3 * x^3 + a * x^2 + b * x + c

-- State the theorem
theorem polynomial_property (a b c : ℝ) :
  -- The y-intercept is 6
  Q 0 a b c = 6 →
  -- The mean of zeros, product of zeros, and sum of coefficients are equal
  (∃ m : ℝ, 
    -- Mean of zeros
    (-(a / 3) / 3 = m) ∧ 
    -- Product of zeros
    (-c / 3 = m) ∧ 
    -- Sum of coefficients
    (3 + a + b + c = m)) →
  -- Conclusion: b = -29
  b = -29 := by sorry

end NUMINAMATH_CALUDE_polynomial_property_l1303_130301


namespace NUMINAMATH_CALUDE_whale_population_growth_l1303_130350

/-- Proves that given the conditions of whale population growth, 
    the initial number of whales was 4000 -/
theorem whale_population_growth (w : ℕ) 
  (h1 : 2 * w = w + w)  -- The number of whales doubles each year
  (h2 : 2 * (2 * w) + 800 = 8800)  -- Prediction for third year
  : w = 4000 := by
  sorry

end NUMINAMATH_CALUDE_whale_population_growth_l1303_130350


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l1303_130396

theorem cubic_equation_roots : ∃ (x₁ x₂ x₃ : ℝ),
  (x₁ = 3 ∧ x₂ = -3 ∧ x₃ = 5) ∧
  (∀ x : ℝ, x^3 - 5*x^2 - 9*x + 45 = 0 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃)) ∧
  (x₁ = -x₂) := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l1303_130396


namespace NUMINAMATH_CALUDE_subset_intersection_iff_bounds_l1303_130351

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 2*a + 1 ≤ x ∧ x ≤ 3*a - 5}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 22}

-- State the theorem
theorem subset_intersection_iff_bounds (a : ℝ) :
  (A a).Nonempty → (A a ⊆ A a ∩ B ↔ 6 ≤ a ∧ a ≤ 9) := by
  sorry

#check subset_intersection_iff_bounds

end NUMINAMATH_CALUDE_subset_intersection_iff_bounds_l1303_130351


namespace NUMINAMATH_CALUDE_charlie_widget_production_l1303_130395

/-- Charlie's widget production problem -/
theorem charlie_widget_production 
  (w t : ℕ) -- w: widgets per hour, t: hours worked on Thursday
  (h1 : w = 3 * t) -- Condition: w = 3t
  : w * t - (w + 6) * (t - 3) = 3 * t + 18 := by
  sorry


end NUMINAMATH_CALUDE_charlie_widget_production_l1303_130395


namespace NUMINAMATH_CALUDE_triangle_inequality_l1303_130338

theorem triangle_inequality (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  a + b > c → b + c > a → c + a > b →
  a + b + c = 4 → 
  a^2 + b^2 + c^2 + a*b*c < 8 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1303_130338


namespace NUMINAMATH_CALUDE_eight_ninths_position_l1303_130387

/-- Represents a fraction as a pair of natural numbers -/
def Fraction := ℕ × ℕ

/-- The sequence of fractions as described in the problem -/
def fraction_sequence : ℕ → Fraction := sorry

/-- The sum of numerator and denominator of a fraction -/
def sum_of_parts (f : Fraction) : ℕ := f.1 + f.2

/-- The position of a fraction in the sequence -/
def position_in_sequence (f : Fraction) : ℕ := sorry

/-- The main theorem: 8/9 is at position 128 in the sequence -/
theorem eight_ninths_position :
  position_in_sequence (8, 9) = 128 := by sorry

end NUMINAMATH_CALUDE_eight_ninths_position_l1303_130387


namespace NUMINAMATH_CALUDE_ice_cream_melt_l1303_130307

theorem ice_cream_melt (r_sphere r_cylinder : ℝ) (h_sphere : r_sphere = 3) (h_cylinder : r_cylinder = 10) :
  (4 / 3 * Real.pi * r_sphere ^ 3) / (Real.pi * r_cylinder ^ 2) = 9 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_melt_l1303_130307


namespace NUMINAMATH_CALUDE_one_ton_equals_2000_pounds_l1303_130385

/-- The weight of a blue whale's tongue in pounds -/
def tongue_weight_pounds : ℕ := 6000

/-- The weight of a blue whale's tongue in tons -/
def tongue_weight_tons : ℕ := 3

/-- The number of pounds in one ton -/
def pounds_per_ton : ℕ := tongue_weight_pounds / tongue_weight_tons

theorem one_ton_equals_2000_pounds : pounds_per_ton = 2000 := by sorry

end NUMINAMATH_CALUDE_one_ton_equals_2000_pounds_l1303_130385


namespace NUMINAMATH_CALUDE_power_mod_23_l1303_130321

theorem power_mod_23 : 17^1499 % 23 = 11 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_23_l1303_130321


namespace NUMINAMATH_CALUDE_trapezium_other_side_length_l1303_130366

theorem trapezium_other_side_length (a b h : ℝ) (area : ℝ) : 
  a = 20 → h = 15 → area = 285 → area = (a + b) * h / 2 → b = 18 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_other_side_length_l1303_130366


namespace NUMINAMATH_CALUDE_homework_ratio_l1303_130314

theorem homework_ratio (total : ℕ) (algebra_percent : ℚ) (linear_eq : ℕ) : 
  total = 140 →
  algebra_percent = 40/100 →
  linear_eq = 28 →
  (linear_eq : ℚ) / (algebra_percent * total) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_homework_ratio_l1303_130314


namespace NUMINAMATH_CALUDE_mean_height_is_correct_l1303_130318

/-- Represents the heights of players on a basketball team -/
def heights : List Nat := [57, 62, 64, 64, 65, 67, 68, 70, 71, 72, 72, 73, 74, 75, 75]

/-- The number of players on the team -/
def num_players : Nat := heights.length

/-- The sum of all player heights -/
def total_height : Nat := heights.sum

/-- Calculates the mean height of the players -/
def mean_height : Rat := total_height / num_players

theorem mean_height_is_correct : mean_height = 1029 / 15 := by sorry

end NUMINAMATH_CALUDE_mean_height_is_correct_l1303_130318


namespace NUMINAMATH_CALUDE_congruence_solutions_count_l1303_130384

theorem congruence_solutions_count : 
  ∃! (s : Finset ℤ), 
    (∀ x ∈ s, (x^3 + 3*x^2 + x + 3) % 25 = 0) ∧ 
    (∀ x, (x^3 + 3*x^2 + x + 3) % 25 = 0 → x % 25 ∈ s) ∧ 
    s.card = 6 :=
by sorry

end NUMINAMATH_CALUDE_congruence_solutions_count_l1303_130384


namespace NUMINAMATH_CALUDE_rachel_pool_fill_time_l1303_130390

/-- Represents the time (in hours) required to fill a pool -/
def fill_time (pool_capacity : ℕ) (num_hoses : ℕ) (flow_rate : ℕ) : ℕ :=
  let total_flow_per_hour := num_hoses * flow_rate * 60
  (pool_capacity + total_flow_per_hour - 1) / total_flow_per_hour

/-- Proves that it takes 33 hours to fill Rachel's pool -/
theorem rachel_pool_fill_time :
  fill_time 30000 5 3 = 33 := by
  sorry

end NUMINAMATH_CALUDE_rachel_pool_fill_time_l1303_130390


namespace NUMINAMATH_CALUDE_remainder_problem_l1303_130317

theorem remainder_problem (N : ℕ) : 
  N % 68 = 0 ∧ N / 68 = 269 → N % 67 = 1 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l1303_130317


namespace NUMINAMATH_CALUDE_unique_solution_f_two_equals_four_l1303_130342

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + y) = f (x^2 - y) + 2 * (f x) * y + y^2

/-- The theorem stating that x^2 is the only function satisfying the equation -/
theorem unique_solution (f : ℝ → ℝ) (h : FunctionalEquation f) : 
  ∀ x : ℝ, f x = x^2 :=
sorry

/-- The value of f(2) is 4 -/
theorem f_two_equals_four (f : ℝ → ℝ) (h : FunctionalEquation f) : 
  f 2 = 4 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_f_two_equals_four_l1303_130342


namespace NUMINAMATH_CALUDE_train_length_l1303_130305

/-- Calculates the length of a train given its speed and time to pass a fixed point. -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 72 → time = 14 → speed * time * (1000 / 3600) = 280 :=
by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l1303_130305


namespace NUMINAMATH_CALUDE_fixed_point_on_curve_l1303_130335

/-- The curve equation for any real m and n -/
def curve_equation (x y m n : ℝ) : Prop :=
  x^2 + y^2 - 2*m*x - 2*n*y + 4*(m - n - 2) = 0

/-- Theorem stating that the point (2, -2) lies on the curve for all real m and n -/
theorem fixed_point_on_curve :
  ∀ (m n : ℝ), curve_equation 2 (-2) m n := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_curve_l1303_130335


namespace NUMINAMATH_CALUDE_eliminate_denominators_l1303_130389

theorem eliminate_denominators (x : ℝ) : 
  (x / 2 - 1 = (x - 1) / 3) ↔ (3 * x - 6 = 2 * (x - 1)) := by
  sorry

end NUMINAMATH_CALUDE_eliminate_denominators_l1303_130389


namespace NUMINAMATH_CALUDE_wallet_cost_l1303_130371

theorem wallet_cost (wallet_cost purse_cost : ℝ) : 
  purse_cost = 4 * wallet_cost - 3 →
  wallet_cost + purse_cost = 107 →
  wallet_cost = 22 := by
sorry

end NUMINAMATH_CALUDE_wallet_cost_l1303_130371


namespace NUMINAMATH_CALUDE_arithmetic_progression_squares_products_l1303_130345

/-- If a, b, and c form an arithmetic progression, then a^2 + ab + b^2, a^2 + ac + c^2, and b^2 + bc + c^2 form an arithmetic progression. -/
theorem arithmetic_progression_squares_products (a b c : ℝ) :
  (∃ d : ℝ, b = a + d ∧ c = a + 2*d) →
  ∃ q : ℝ, (a^2 + a*c + c^2) - (a^2 + a*b + b^2) = q ∧
           (b^2 + b*c + c^2) - (a^2 + a*c + c^2) = q :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_squares_products_l1303_130345


namespace NUMINAMATH_CALUDE_inequality_proof_l1303_130372

theorem inequality_proof (a b : ℝ) (h : 4 * b + a = 1) : a^2 + 4 * b^2 ≥ 1/5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1303_130372


namespace NUMINAMATH_CALUDE_modified_triathlon_speed_l1303_130391

theorem modified_triathlon_speed (total_time : ℝ) (swim_distance swim_speed : ℝ)
  (run_distance run_speed : ℝ) (kayak_distance kayak_speed : ℝ)
  (bike_distance : ℝ) :
  total_time = 3 ∧
  swim_distance = 1/2 ∧ swim_speed = 2 ∧
  run_distance = 5 ∧ run_speed = 10 ∧
  kayak_distance = 1 ∧ kayak_speed = 3 ∧
  bike_distance = 20 →
  (bike_distance / (total_time - (swim_distance / swim_speed + run_distance / run_speed + kayak_distance / kayak_speed))) = 240/23 :=
by sorry

end NUMINAMATH_CALUDE_modified_triathlon_speed_l1303_130391


namespace NUMINAMATH_CALUDE_percentage_not_sold_approx_l1303_130313

def initial_stock : ℕ := 1200

def monday_sold : ℕ := 75
def monday_returned : ℕ := 6

def tuesday_sold : ℕ := 50

def wednesday_sold : ℕ := 64
def wednesday_returned : ℕ := 8

def thursday_sold : ℕ := 78

def friday_sold : ℕ := 135
def friday_returned : ℕ := 5

def total_sold : ℕ := 
  (monday_sold - monday_returned) + 
  tuesday_sold + 
  (wednesday_sold - wednesday_returned) + 
  thursday_sold + 
  (friday_sold - friday_returned)

def books_not_sold : ℕ := initial_stock - total_sold

def percentage_not_sold : ℚ := (books_not_sold : ℚ) / (initial_stock : ℚ) * 100

theorem percentage_not_sold_approx :
  abs (percentage_not_sold - 68.08) < 0.01 := by sorry

end NUMINAMATH_CALUDE_percentage_not_sold_approx_l1303_130313


namespace NUMINAMATH_CALUDE_basketball_substitutions_l1303_130383

/-- Represents the number of ways to make exactly n substitutions -/
def b (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => 5 * (11 - n) * b n

/-- The total number of ways to make substitutions -/
def m : ℕ := (b 0) + (b 1) + (b 2) + (b 3) + (b 4) + (b 5)

theorem basketball_substitutions :
  m % 1000 = 301 := by sorry

end NUMINAMATH_CALUDE_basketball_substitutions_l1303_130383
