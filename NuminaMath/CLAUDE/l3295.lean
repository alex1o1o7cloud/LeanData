import Mathlib

namespace NUMINAMATH_CALUDE_range_of_a_l3295_329520

theorem range_of_a (p q : ℝ → Prop) (a : ℝ) :
  (∀ x, p x ↔ (x ≤ 1/2 ∨ x ≥ 1)) →
  (∀ x, q x ↔ (x - a) * (x - a - 1) ≤ 0) →
  (∀ x, ¬(q x) → p x) →
  (∃ x, ¬(q x) ∧ ¬(p x)) →
  0 ≤ a ∧ a ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3295_329520


namespace NUMINAMATH_CALUDE_simon_lego_count_l3295_329582

theorem simon_lego_count (kent_legos : ℕ) (bruce_extra : ℕ) (simon_percentage : ℚ) :
  kent_legos = 40 →
  bruce_extra = 20 →
  simon_percentage = 1/5 →
  (kent_legos + bruce_extra) * (1 + simon_percentage) = 72 :=
by sorry

end NUMINAMATH_CALUDE_simon_lego_count_l3295_329582


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l3295_329579

/-- Given an arithmetic sequence {a_n} with sum of first n terms S_n and common difference d -/
def arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, S n = n * (2 * (a 1) + (n - 1) * d) / 2

/-- The theorem to be proved -/
theorem arithmetic_sequence_difference (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a S d →
  (S 2017 / 2017) - (S 17 / 17) = 100 →
  d = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l3295_329579


namespace NUMINAMATH_CALUDE_distance_from_origin_l3295_329512

/-- Given a point (x,y) in the first quadrant satisfying certain conditions,
    prove that its distance from the origin is √(233 + 12√7). -/
theorem distance_from_origin (x y : ℝ) (h1 : y = 14) (h2 : (x - 3)^2 + (y - 8)^2 = 64) (h3 : x > 3) :
  Real.sqrt (x^2 + y^2) = Real.sqrt (233 + 12 * Real.sqrt 7) := by
  sorry

end NUMINAMATH_CALUDE_distance_from_origin_l3295_329512


namespace NUMINAMATH_CALUDE_queen_placement_probability_l3295_329568

/-- The number of squares on a chessboard -/
def chessboardSize : ℕ := 64

/-- The number of trials in the experiment -/
def numberOfTrials : ℕ := 3

/-- The probability that two randomly placed queens can attack each other -/
def attackingProbability : ℚ := 13 / 36

/-- The probability of at least one non-attacking configuration in 3 trials -/
def nonAttackingProbability : ℚ := 1 - attackingProbability ^ numberOfTrials

theorem queen_placement_probability :
  nonAttackingProbability = 1 - (13 / 36) ^ 3 :=
by sorry

end NUMINAMATH_CALUDE_queen_placement_probability_l3295_329568


namespace NUMINAMATH_CALUDE_ellipse_intersection_properties_l3295_329511

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the line
def line (k m x y : ℝ) : Prop := y = k*x + m

-- Define the intersection points
def intersection (k m : ℝ) (A B : ℝ × ℝ) : Prop :=
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ line k m A.1 A.2 ∧ line k m B.1 B.2

-- Define the condition |OA| = |AC|
def equal_distances (A : ℝ × ℝ) : Prop :=
  A.1^2 + A.2^2 = A.1^2 + (A.2 - 3)^2

-- Define the condition S_△AOC = S_△AOB
def equal_areas (A B : ℝ × ℝ) : Prop :=
  A.1 * 3 = (A.1 + B.1) * A.2

-- Define the condition k_OA · k_OB = -3/4
def slope_product (A B : ℝ × ℝ) : Prop :=
  (A.2 / A.1) * (B.2 / B.1) = -3/4

theorem ellipse_intersection_properties (k m : ℝ) (A B : ℝ × ℝ) :
  intersection k m A B →
  line k m 0 3 →
  equal_distances A →
  equal_areas A B →
  slope_product A B →
  ((A = (1, 3/2) ∨ A = (-1, 3/2)) ∧
   (k = 3/2 ∧ m = 3 ∨ k = -3/2 ∧ m = 3) ∧
   (A.1 - B.1)*(A.2 + B.2)/2 = Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_intersection_properties_l3295_329511


namespace NUMINAMATH_CALUDE_pet_shop_guinea_pigs_l3295_329559

/-- Given a ratio of rabbits to guinea pigs and the number of rabbits, 
    calculate the number of guinea pigs -/
def calculate_guinea_pigs (rabbit_ratio : ℕ) (guinea_pig_ratio : ℕ) (num_rabbits : ℕ) : ℕ :=
  (num_rabbits * guinea_pig_ratio) / rabbit_ratio

/-- Theorem: In a pet shop with a 5:4 ratio of rabbits to guinea pigs and 25 rabbits, 
    there are 20 guinea pigs -/
theorem pet_shop_guinea_pigs : 
  let rabbit_ratio : ℕ := 5
  let guinea_pig_ratio : ℕ := 4
  let num_rabbits : ℕ := 25
  calculate_guinea_pigs rabbit_ratio guinea_pig_ratio num_rabbits = 20 := by
  sorry

end NUMINAMATH_CALUDE_pet_shop_guinea_pigs_l3295_329559


namespace NUMINAMATH_CALUDE_prob_select_one_from_two_out_of_four_prob_select_one_from_two_out_of_four_proof_l3295_329502

/-- The probability of selecting exactly one person from a group of two when randomly choosing two people from a group of four -/
theorem prob_select_one_from_two_out_of_four : ℚ :=
  2 / 3

/-- The total number of ways to select two people from four -/
def total_selections : ℕ := 6

/-- The number of ways to select exactly one person from a specific group of two when choosing two from four -/
def favorable_selections : ℕ := 4

/-- The probability is equal to the number of favorable outcomes divided by the total number of possible outcomes -/
theorem prob_select_one_from_two_out_of_four_proof :
  prob_select_one_from_two_out_of_four = favorable_selections / total_selections :=
sorry

end NUMINAMATH_CALUDE_prob_select_one_from_two_out_of_four_prob_select_one_from_two_out_of_four_proof_l3295_329502


namespace NUMINAMATH_CALUDE_club_president_vicepresident_selection_l3295_329523

theorem club_president_vicepresident_selection (total_members : ℕ) (boys : ℕ) (girls : ℕ) 
  (h1 : total_members = 30)
  (h2 : boys = 18)
  (h3 : girls = 12)
  (h4 : total_members = boys + girls) :
  (boys * (total_members - 1)) = 522 := by
  sorry

end NUMINAMATH_CALUDE_club_president_vicepresident_selection_l3295_329523


namespace NUMINAMATH_CALUDE_abs_diff_positive_iff_not_equal_l3295_329552

theorem abs_diff_positive_iff_not_equal (x : ℝ) : x ≠ 3 ↔ |x - 3| > 0 := by sorry

end NUMINAMATH_CALUDE_abs_diff_positive_iff_not_equal_l3295_329552


namespace NUMINAMATH_CALUDE_max_integers_less_than_negative_five_l3295_329508

theorem max_integers_less_than_negative_five (a b c d e : ℤ) 
  (sum_condition : a + b + c + d + e = 20) : 
  (∃ (count : ℕ), count ≤ 4 ∧ 
    (count = (if a < -5 then 1 else 0) + 
             (if b < -5 then 1 else 0) + 
             (if c < -5 then 1 else 0) + 
             (if d < -5 then 1 else 0) + 
             (if e < -5 then 1 else 0)) ∧
    ∀ (other_count : ℕ), 
      (other_count = (if a < -5 then 1 else 0) + 
                     (if b < -5 then 1 else 0) + 
                     (if c < -5 then 1 else 0) + 
                     (if d < -5 then 1 else 0) + 
                     (if e < -5 then 1 else 0)) → 
      other_count ≤ count) ∧
  ¬(∃ (impossible_count : ℕ), impossible_count = 5 ∧
    impossible_count = (if a < -5 then 1 else 0) + 
                       (if b < -5 then 1 else 0) + 
                       (if c < -5 then 1 else 0) + 
                       (if d < -5 then 1 else 0) + 
                       (if e < -5 then 1 else 0)) :=
by sorry

end NUMINAMATH_CALUDE_max_integers_less_than_negative_five_l3295_329508


namespace NUMINAMATH_CALUDE_line_slope_l3295_329562

/-- Given a line described by the equation 3y + 9 = -6x - 15, its slope is -2. -/
theorem line_slope (x y : ℝ) : 3 * y + 9 = -6 * x - 15 → (y - (y + 1)) / (x - (x + 1)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l3295_329562


namespace NUMINAMATH_CALUDE_min_draws_for_twelve_balls_l3295_329532

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  black : Nat

/-- Represents the minimum number of balls needed to guarantee at least n balls of a single color -/
def minDrawsForColor (counts : BallCounts) (n : Nat) : Nat :=
  sorry

/-- The main theorem stating the minimum number of draws required -/
theorem min_draws_for_twelve_balls (counts : BallCounts) 
  (h_red : counts.red = 30)
  (h_green : counts.green = 22)
  (h_yellow : counts.yellow = 18)
  (h_blue : counts.blue = 15)
  (h_black : counts.black = 10) :
  minDrawsForColor counts 12 = 55 := by
  sorry

end NUMINAMATH_CALUDE_min_draws_for_twelve_balls_l3295_329532


namespace NUMINAMATH_CALUDE_perimeter_of_cut_square_perimeter_of_specific_cut_square_l3295_329584

/-- The perimeter of a figure formed by cutting a square into two equal rectangles and placing them next to each other -/
theorem perimeter_of_cut_square (square_side : ℝ) : 
  square_side > 0 → 
  (3 * square_side + 4 * (square_side / 2)) = 5 * square_side := by
  sorry

/-- The perimeter of a figure formed by cutting a square with side length 100 into two equal rectangles and placing them next to each other is 500 -/
theorem perimeter_of_specific_cut_square : 
  (3 * 100 + 4 * (100 / 2)) = 500 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_cut_square_perimeter_of_specific_cut_square_l3295_329584


namespace NUMINAMATH_CALUDE_max_knights_and_courtiers_l3295_329574

def king_table_size : ℕ := 7
def min_courtiers : ℕ := 12
def max_courtiers : ℕ := 18
def min_knights : ℕ := 10
def max_knights : ℕ := 20

def is_valid_solution (courtiers knights : ℕ) : Prop :=
  min_courtiers ≤ courtiers ∧ courtiers ≤ max_courtiers ∧
  min_knights ≤ knights ∧ knights ≤ max_knights ∧
  (1 : ℚ) / courtiers + (1 : ℚ) / knights = (1 : ℚ) / king_table_size

theorem max_knights_and_courtiers :
  ∃ (max_knights courtiers : ℕ),
    is_valid_solution courtiers max_knights ∧
    ∀ (k : ℕ), is_valid_solution courtiers k → k ≤ max_knights ∧
    max_knights = 14 ∧ courtiers = 14 := by
  sorry

end NUMINAMATH_CALUDE_max_knights_and_courtiers_l3295_329574


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l3295_329597

theorem smallest_prime_divisor_of_sum : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (3^19 + 11^13) ∧ ∀ q, Nat.Prime q → q ∣ (3^19 + 11^13) → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l3295_329597


namespace NUMINAMATH_CALUDE_fraction_relation_l3295_329581

theorem fraction_relation (q r s t : ℚ) 
  (h1 : q / r = 12)
  (h2 : s / r = 8)
  (h3 : s / t = 1 / 4) :
  t / q = 8 / 3 := by sorry

end NUMINAMATH_CALUDE_fraction_relation_l3295_329581


namespace NUMINAMATH_CALUDE_parallelogram_base_given_triangle_l3295_329555

/-- Given a triangle and a parallelogram with equal areas and the same height,
    if the base of the triangle is 24 inches, then the base of the parallelogram is 12 inches. -/
theorem parallelogram_base_given_triangle (h : ℝ) (b_p : ℝ) : 
  (1/2 * 24 * h = b_p * h) → b_p = 12 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_given_triangle_l3295_329555


namespace NUMINAMATH_CALUDE_compute_alpha_l3295_329564

variable (α β : ℂ)

theorem compute_alpha (h1 : (α + β).re > 0)
                       (h2 : (Complex.I * (α - 3 * β)).re > 0)
                       (h3 : β = 4 + 3 * Complex.I) :
  α = 12 - 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_compute_alpha_l3295_329564


namespace NUMINAMATH_CALUDE_parabola_theorem_l3295_329599

/-- Parabola with parameter p and a tangent line -/
structure Parabola where
  p : ℝ
  tangent_x_intercept : ℝ
  tangent_y_intercept : ℝ

/-- Properties of the parabola -/
def parabola_properties (para : Parabola) : Prop :=
  -- Tangent line equation matches the given form
  para.tangent_x_intercept = -75 ∧ para.tangent_y_intercept = 15 ∧
  -- Parameter p is 6
  para.p = 6 ∧
  -- Focus coordinates are (3, 0)
  (3 : ℝ) = para.p / 2 ∧
  -- Directrix equation is x = -3
  (-3 : ℝ) = -para.p / 2

/-- Theorem stating the properties of the parabola -/
theorem parabola_theorem (para : Parabola) :
  parabola_properties para :=
sorry

end NUMINAMATH_CALUDE_parabola_theorem_l3295_329599


namespace NUMINAMATH_CALUDE_harmonic_geometric_log_ratio_l3295_329557

/-- Given distinct positive real numbers a, b, c forming a harmonic sequence,
    and their logarithms forming a geometric sequence, 
    prove that the common ratio of the geometric sequence is a non-1 cube root of unity. -/
theorem harmonic_geometric_log_ratio 
  (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (hharmseq : (1 / b) = (1 / (2 : ℝ)) * ((1 / a) + (1 / c)))
  (hgeomseq : ∃ r : ℂ, (Complex.log b / Complex.log a) = r ∧ 
                       (Complex.log c / Complex.log b) = r ∧ 
                       (Complex.log a / Complex.log c) = r) :
  ∃ r : ℂ, r^3 = 1 ∧ r ≠ 1 ∧ 
    ((Complex.log b / Complex.log a) = r ∧ 
     (Complex.log c / Complex.log b) = r ∧ 
     (Complex.log a / Complex.log c) = r) := by
  sorry

end NUMINAMATH_CALUDE_harmonic_geometric_log_ratio_l3295_329557


namespace NUMINAMATH_CALUDE_championship_outcomes_l3295_329549

/-- The number of possible outcomes for awarding n championship titles to m students. -/
def numberOfOutcomes (m n : ℕ) : ℕ := m^n

/-- Theorem: Given 8 students competing for 3 championship titles, 
    the number of possible outcomes for the champions is equal to 8^3. -/
theorem championship_outcomes : numberOfOutcomes 8 3 = 512 := by
  sorry

end NUMINAMATH_CALUDE_championship_outcomes_l3295_329549


namespace NUMINAMATH_CALUDE_prob_other_side_red_is_two_thirds_l3295_329592

/-- Represents a card with two sides --/
structure Card where
  side1 : Bool  -- True for red, False for black
  side2 : Bool

/-- The set of all cards in the box --/
def box : Finset Card := sorry

/-- The total number of cards in the box --/
def total_cards : Nat := 8

/-- The number of cards that are black on both sides --/
def black_both_sides : Nat := 4

/-- The number of cards that are black on one side and red on the other --/
def black_red : Nat := 2

/-- The number of cards that are red on both sides --/
def red_both_sides : Nat := 2

/-- Axiom: The box contains the correct number of each type of card --/
axiom box_composition :
  (box.filter (fun c => !c.side1 ∧ !c.side2)).card = black_both_sides ∧
  (box.filter (fun c => (c.side1 ∧ !c.side2) ∨ (!c.side1 ∧ c.side2))).card = black_red ∧
  (box.filter (fun c => c.side1 ∧ c.side2)).card = red_both_sides

/-- Axiom: The total number of cards is correct --/
axiom total_cards_correct : box.card = total_cards

/-- The probability of selecting a card with a red side, given that one side is observed to be red --/
def prob_other_side_red (observed_red : Bool) : ℚ := sorry

/-- Theorem: The probability that the other side is red, given that the observed side is red, is 2/3 --/
theorem prob_other_side_red_is_two_thirds (observed_red : Bool) :
  observed_red → prob_other_side_red observed_red = 2/3 := by sorry

end NUMINAMATH_CALUDE_prob_other_side_red_is_two_thirds_l3295_329592


namespace NUMINAMATH_CALUDE_color_copies_comparison_l3295_329548

/-- The cost per color copy at print shop X -/
def cost_x : ℚ := 120 / 100

/-- The cost per color copy at print shop Y -/
def cost_y : ℚ := 170 / 100

/-- The difference in total cost between print shops Y and X -/
def cost_difference : ℚ := 35

/-- The number of color copies being compared -/
def n : ℚ := 70

theorem color_copies_comparison :
  cost_y * n = cost_x * n + cost_difference :=
by sorry

end NUMINAMATH_CALUDE_color_copies_comparison_l3295_329548


namespace NUMINAMATH_CALUDE_root_value_theorem_l3295_329522

theorem root_value_theorem (a : ℝ) (h : a^2 + 2*a - 1 = 0) : -a^2 - 2*a + 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_root_value_theorem_l3295_329522


namespace NUMINAMATH_CALUDE_fayes_age_l3295_329590

/-- Represents the ages of Chad, Diana, Eduardo, and Faye --/
structure Ages where
  chad : ℕ
  diana : ℕ
  eduardo : ℕ
  faye : ℕ

/-- The age relationships between Chad, Diana, Eduardo, and Faye --/
def age_relationships (ages : Ages) : Prop :=
  ages.diana = ages.eduardo - 4 ∧
  ages.eduardo = ages.chad + 5 ∧
  ages.faye = ages.chad + 2 ∧
  ages.diana = 16

/-- Theorem stating that given the age relationships, Faye's age is 17 --/
theorem fayes_age (ages : Ages) : age_relationships ages → ages.faye = 17 := by
  sorry

end NUMINAMATH_CALUDE_fayes_age_l3295_329590


namespace NUMINAMATH_CALUDE_biology_marks_proof_l3295_329530

def english_marks : ℕ := 86
def math_marks : ℕ := 85
def physics_marks : ℕ := 92
def chemistry_marks : ℕ := 87
def average_marks : ℕ := 89
def total_subjects : ℕ := 5

def calculate_biology_marks (eng : ℕ) (math : ℕ) (phys : ℕ) (chem : ℕ) (avg : ℕ) (total : ℕ) : ℕ :=
  avg * total - (eng + math + phys + chem)

theorem biology_marks_proof :
  calculate_biology_marks english_marks math_marks physics_marks chemistry_marks average_marks total_subjects = 95 := by
  sorry

end NUMINAMATH_CALUDE_biology_marks_proof_l3295_329530


namespace NUMINAMATH_CALUDE_largest_c_value_l3295_329553

/-- The function f(x) = x^2 - 6x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - 6*x + c

/-- 2 is in the range of f -/
def two_in_range (c : ℝ) : Prop := ∃ x, f c x = 2

theorem largest_c_value :
  (∃ c_max : ℝ, two_in_range c_max ∧ ∀ c > c_max, ¬(two_in_range c)) ∧
  (∀ c_max : ℝ, (two_in_range c_max ∧ ∀ c > c_max, ¬(two_in_range c)) → c_max = 11) :=
sorry

end NUMINAMATH_CALUDE_largest_c_value_l3295_329553


namespace NUMINAMATH_CALUDE_height_increase_calculation_l3295_329531

/-- Represents the increase in height per decade for a specific plant species -/
def height_increase_per_decade : ℝ := sorry

/-- The number of decades in 4 centuries -/
def decades_in_four_centuries : ℕ := 40

/-- The total increase in height over 4 centuries in meters -/
def total_height_increase : ℝ := 3000

theorem height_increase_calculation :
  height_increase_per_decade * (decades_in_four_centuries : ℝ) = total_height_increase ∧
  height_increase_per_decade = 75 := by sorry

end NUMINAMATH_CALUDE_height_increase_calculation_l3295_329531


namespace NUMINAMATH_CALUDE_chess_club_members_count_l3295_329509

theorem chess_club_members_count : ∃! n : ℕ, 
  300 ≤ n ∧ n ≤ 400 ∧ 
  (n + 4) % 10 = 0 ∧ 
  (n + 5) % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_chess_club_members_count_l3295_329509


namespace NUMINAMATH_CALUDE_linear_function_increases_iff_positive_slope_increasing_linear_function_k_equals_four_l3295_329529

/-- A linear function y = mx + b increases if and only if its slope m is positive -/
theorem linear_function_increases_iff_positive_slope {m b : ℝ} :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → m * x₁ + b < m * x₂ + b) ↔ m > 0 := by sorry

/-- For the function y = (k - 3)x + 2, if y increases as x increases, then k = 4 -/
theorem increasing_linear_function_k_equals_four (k : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → (k - 3) * x₁ + 2 < (k - 3) * x₂ + 2) → k = 4 := by sorry

end NUMINAMATH_CALUDE_linear_function_increases_iff_positive_slope_increasing_linear_function_k_equals_four_l3295_329529


namespace NUMINAMATH_CALUDE_saturday_sales_77_l3295_329526

/-- Represents the number of boxes sold on each day --/
structure DailySales where
  saturday : ℕ
  sunday : ℕ
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ

/-- Calculates the total sales over 5 days --/
def totalSales (sales : DailySales) : ℕ :=
  sales.saturday + sales.sunday + sales.monday + sales.tuesday + sales.wednesday

/-- Checks if the sales follow the given percentage increases --/
def followsPercentageIncreases (sales : DailySales) : Prop :=
  sales.sunday = (sales.saturday * 3) / 2 ∧
  sales.monday = (sales.sunday * 13) / 10 ∧
  sales.tuesday = (sales.monday * 6) / 5 ∧
  sales.wednesday = (sales.tuesday * 11) / 10

theorem saturday_sales_77 (sales : DailySales) :
  followsPercentageIncreases sales →
  totalSales sales = 720 →
  sales.saturday = 77 := by
  sorry


end NUMINAMATH_CALUDE_saturday_sales_77_l3295_329526


namespace NUMINAMATH_CALUDE_triangle_height_proof_l3295_329598

theorem triangle_height_proof (area : ℝ) (base : ℝ) (height : ℝ) : 
  area = 576 → base = 32 → area = (base * height) / 2 → height = 36 := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_proof_l3295_329598


namespace NUMINAMATH_CALUDE_debby_museum_pictures_l3295_329514

/-- The number of pictures Debby took at the zoo -/
def zoo_pictures : ℕ := 24

/-- The number of pictures Debby deleted -/
def deleted_pictures : ℕ := 14

/-- The number of pictures Debby had remaining after deletion -/
def remaining_pictures : ℕ := 22

/-- The number of pictures Debby took at the museum -/
def museum_pictures : ℕ := zoo_pictures + deleted_pictures - remaining_pictures

theorem debby_museum_pictures : museum_pictures = 12 := by
  sorry

end NUMINAMATH_CALUDE_debby_museum_pictures_l3295_329514


namespace NUMINAMATH_CALUDE_otimes_nested_equality_l3295_329575

/-- Custom binary operation ⊗ -/
def otimes (x y : ℝ) : ℝ := x^3 + 5*x*y - y

theorem otimes_nested_equality (a : ℝ) : otimes a (otimes a a) = 5*a^4 + 24*a^3 - 10*a^2 + a := by
  sorry

end NUMINAMATH_CALUDE_otimes_nested_equality_l3295_329575


namespace NUMINAMATH_CALUDE_charlotte_dan_mean_score_l3295_329594

def test_scores : List ℝ := [82, 84, 86, 88, 90, 92, 95, 97]

def total_score : ℝ := test_scores.sum

def ava_ben_mean : ℝ := 90

def num_tests : ℕ := 4

theorem charlotte_dan_mean_score :
  let ava_ben_total : ℝ := ava_ben_mean * num_tests
  let charlotte_dan_total : ℝ := total_score - ava_ben_total
  charlotte_dan_total / num_tests = 88.5 := by sorry

end NUMINAMATH_CALUDE_charlotte_dan_mean_score_l3295_329594


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3295_329583

theorem quadratic_equation_solution : 
  ∀ x : ℝ, x^2 = -7*x ↔ x = 0 ∨ x = -7 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3295_329583


namespace NUMINAMATH_CALUDE_shanmukham_purchase_l3295_329540

/-- Calculates the final amount to pay for goods given the original price, rebate percentage, and sales tax percentage. -/
def finalAmount (originalPrice rebatePercentage salesTaxPercentage : ℚ) : ℚ :=
  let priceAfterRebate := originalPrice * (1 - rebatePercentage / 100)
  let salesTax := priceAfterRebate * (salesTaxPercentage / 100)
  priceAfterRebate + salesTax

/-- Theorem stating that given the specific conditions, the final amount to pay is 6876.10 -/
theorem shanmukham_purchase :
  finalAmount 6650 6 10 = 6876.1 := by
  sorry

end NUMINAMATH_CALUDE_shanmukham_purchase_l3295_329540


namespace NUMINAMATH_CALUDE_min_m_plus_n_l3295_329500

/-- The sum of interior angles of a regular n-gon -/
def interior_angle_sum (n : ℕ) : ℕ := 180 * (n - 2)

/-- The sum of interior angles of m regular n-gons -/
def total_interior_angle_sum (m n : ℕ) : ℕ := m * interior_angle_sum n

/-- Predicate to check if the sum of interior angles is divisible by 27 -/
def is_divisible_by_27 (m n : ℕ) : Prop :=
  (total_interior_angle_sum m n) % 27 = 0

/-- The main theorem stating the minimum value of m + n -/
theorem min_m_plus_n :
  ∃ (m n : ℕ), m ≥ 1 ∧ n ≥ 3 ∧ is_divisible_by_27 m n ∧
  (∀ (m' n' : ℕ), m' ≥ 1 → n' ≥ 3 → is_divisible_by_27 m' n' → m + n ≤ m' + n') ∧
  m + n = 6 :=
sorry

end NUMINAMATH_CALUDE_min_m_plus_n_l3295_329500


namespace NUMINAMATH_CALUDE_ratio_DO_OP_l3295_329571

/-- Parallelogram ABCD with points P on AB and Q on BC -/
structure Parallelogram (V : Type*) [AddCommGroup V] [Module ℝ V] :=
  (A B C D P Q O : V)
  (is_parallelogram : (C - B) = (D - A))
  (P_on_AB : ∃ t : ℝ, P = A + t • (B - A) ∧ 0 ≤ t ∧ t ≤ 1)
  (Q_on_BC : ∃ s : ℝ, Q = B + s • (C - B) ∧ 0 ≤ s ∧ s ≤ 1)
  (prop_AB_BP : 3 • (B - A) = 7 • (P - B))
  (prop_BC_BQ : 3 • (C - B) = 4 • (Q - B))
  (O_intersect : ∃ r t : ℝ, O = A + r • (Q - A) ∧ O = D + t • (P - D))

/-- The ratio DO : OP is 7 : 3 -/
theorem ratio_DO_OP (V : Type*) [AddCommGroup V] [Module ℝ V] (para : Parallelogram V) :
  ∃ k : ℝ, para.D - para.O = (7 * k) • (para.O - para.P) ∧ k ≠ 0 := by sorry

end NUMINAMATH_CALUDE_ratio_DO_OP_l3295_329571


namespace NUMINAMATH_CALUDE_blue_button_probability_l3295_329542

/-- Represents a jar containing buttons of different colors. -/
structure Jar where
  red : ℕ
  blue : ℕ

/-- The probability of selecting a blue button from a jar. -/
def blueProb (j : Jar) : ℚ :=
  j.blue / (j.red + j.blue)

/-- The initial state of Jar C. -/
def jarC : Jar := { red := 6, blue := 10 }

/-- The number of buttons transferred from Jar C to Jar D. -/
def transferred : ℕ := 4

/-- Jar C after the transfer. -/
def jarCAfter : Jar := { red := jarC.red - transferred / 2, blue := jarC.blue - transferred / 2 }

/-- Jar D after the transfer. -/
def jarD : Jar := { red := transferred / 2, blue := transferred / 2 }

/-- Theorem stating the probability of selecting blue buttons from both jars. -/
theorem blue_button_probability : 
  blueProb jarCAfter * blueProb jarD = 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_blue_button_probability_l3295_329542


namespace NUMINAMATH_CALUDE_function_inequality_l3295_329577

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem function_inequality (h1 : Differentiable ℝ f) 
  (h2 : ∀ x, f x < deriv f x) : 
  f 2 > Real.exp 2 * f 0 ∧ f 2017 > Real.exp 2017 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3295_329577


namespace NUMINAMATH_CALUDE_complex_fourth_power_l3295_329525

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_fourth_power : (1 - i) ^ 4 = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_fourth_power_l3295_329525


namespace NUMINAMATH_CALUDE_total_cost_is_43_l3295_329521

-- Define the prices
def sandwich_price : ℚ := 4
def soda_price : ℚ := 3

-- Define the discount threshold and rate
def discount_threshold : ℚ := 50
def discount_rate : ℚ := 0.1

-- Define the quantities
def num_sandwiches : ℕ := 7
def num_sodas : ℕ := 5

-- Calculate the total cost before discount
def total_cost : ℚ := sandwich_price * num_sandwiches + soda_price * num_sodas

-- Function to apply discount if applicable
def apply_discount (cost : ℚ) : ℚ :=
  if cost > discount_threshold then cost * (1 - discount_rate) else cost

-- Theorem to prove
theorem total_cost_is_43 : apply_discount total_cost = 43 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_43_l3295_329521


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l3295_329501

theorem isosceles_triangle_base_angle (α β γ : ℝ) : 
  -- The triangle is isosceles
  (α = β ∨ β = γ ∨ γ = α) →
  -- The sum of angles in a triangle is 180°
  α + β + γ = 180 →
  -- One angle is 80°
  (α = 80 ∨ β = 80 ∨ γ = 80) →
  -- The base angle is either 50° or 80°
  (α = 50 ∨ α = 80 ∨ β = 50 ∨ β = 80 ∨ γ = 50 ∨ γ = 80) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l3295_329501


namespace NUMINAMATH_CALUDE_special_rectangle_perimeter_l3295_329578

/-- A rectangle with the property that increasing both its length and width by 6
    results in an area increase of 114 -/
structure SpecialRectangle where
  length : ℝ
  width : ℝ
  area_increase : (length + 6) * (width + 6) - length * width = 114

theorem special_rectangle_perimeter (rect : SpecialRectangle) :
  2 * (rect.length + rect.width) = 26 :=
sorry

end NUMINAMATH_CALUDE_special_rectangle_perimeter_l3295_329578


namespace NUMINAMATH_CALUDE_sum_of_rectangle_areas_l3295_329527

def first_six_odd_numbers : List ℕ := [1, 3, 5, 7, 9, 11]

def rectangle_areas (base_width : ℕ) (lengths : List ℕ) : List ℕ :=
  lengths.map (λ l => base_width * l^2)

theorem sum_of_rectangle_areas :
  let base_width := 2
  let areas := rectangle_areas base_width first_six_odd_numbers
  List.sum areas = 572 := by sorry

end NUMINAMATH_CALUDE_sum_of_rectangle_areas_l3295_329527


namespace NUMINAMATH_CALUDE_kelly_initial_bracelets_l3295_329538

/-- Proves that Kelly initially had 16 bracelets given the problem conditions -/
theorem kelly_initial_bracelets :
  ∀ (k : ℕ), -- k represents Kelly's initial number of bracelets
  let b_initial : ℕ := 5 -- Bingley's initial number of bracelets
  let b_after_kelly : ℕ := b_initial + k / 4 -- Bingley's bracelets after receiving from Kelly
  let b_final : ℕ := b_after_kelly * 2 / 3 -- Bingley's final number of bracelets
  b_final = 6 → k = 16 := by
  sorry

end NUMINAMATH_CALUDE_kelly_initial_bracelets_l3295_329538


namespace NUMINAMATH_CALUDE_inequality_solution_count_l3295_329543

theorem inequality_solution_count : 
  (∃ (S : Finset Int), 
    (∀ n : Int, n ∈ S ↔ Real.sqrt (2 * n) ≤ Real.sqrt (5 * n - 8) ∧ 
                        Real.sqrt (5 * n - 8) < Real.sqrt (3 * n + 7)) ∧
    S.card = 5) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_count_l3295_329543


namespace NUMINAMATH_CALUDE_sqrt_three_plus_one_over_two_lt_sqrt_two_l3295_329585

theorem sqrt_three_plus_one_over_two_lt_sqrt_two :
  (Real.sqrt 3 + 1) / 2 < Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_plus_one_over_two_lt_sqrt_two_l3295_329585


namespace NUMINAMATH_CALUDE_percentage_calculation_l3295_329528

theorem percentage_calculation (whole : ℝ) (part : ℝ) (percentage : ℝ) 
  (h1 : whole = 800)
  (h2 : part = 200)
  (h3 : percentage = (part / whole) * 100) :
  percentage = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l3295_329528


namespace NUMINAMATH_CALUDE_angle_inclination_range_l3295_329534

theorem angle_inclination_range (a : ℝ) (h1 : a ≠ 0) 
  (h2 : ((-a - 2 + 1) * (Real.sqrt 3 / 3 * a + 1) > 0)) :
  ∃ α : ℝ, (2 * Real.pi / 3 < α) ∧ (α < 3 * Real.pi / 4) ∧ 
  (a = Real.tan α) :=
sorry

end NUMINAMATH_CALUDE_angle_inclination_range_l3295_329534


namespace NUMINAMATH_CALUDE_square_sum_is_seven_l3295_329593

theorem square_sum_is_seven (x y : ℝ) (h : (x^2 + 1) * (y^2 + 1) + 9 = 6 * (x + y)) : 
  x^2 + y^2 = 7 := by sorry

end NUMINAMATH_CALUDE_square_sum_is_seven_l3295_329593


namespace NUMINAMATH_CALUDE_triangle_side_calculation_l3295_329537

theorem triangle_side_calculation (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively
  -- Given conditions
  (a = 2) →
  (A = 30 * π / 180) →  -- Convert degrees to radians
  (B = 45 * π / 180) →  -- Convert degrees to radians
  -- Law of Sines
  (a / Real.sin A = b / Real.sin B) →
  -- Conclusion
  b = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_calculation_l3295_329537


namespace NUMINAMATH_CALUDE_aunt_age_proof_l3295_329560

def cori_age_today : ℕ := 3
def years_until_comparison : ℕ := 5

def aunt_age_today : ℕ := 19

theorem aunt_age_proof :
  (cori_age_today + years_until_comparison) * 3 = aunt_age_today + years_until_comparison :=
by sorry

end NUMINAMATH_CALUDE_aunt_age_proof_l3295_329560


namespace NUMINAMATH_CALUDE_urn_problem_l3295_329570

theorem urn_problem (total : ℕ) (red_percent : ℚ) (new_red_percent : ℚ) 
  (h1 : total = 120)
  (h2 : red_percent = 2/5)
  (h3 : new_red_percent = 4/5) :
  ∃ (removed : ℕ), 
    (red_percent * total : ℚ) / (total - removed : ℚ) = new_red_percent ∧ 
    removed = 60 := by
  sorry

end NUMINAMATH_CALUDE_urn_problem_l3295_329570


namespace NUMINAMATH_CALUDE_franks_savings_l3295_329547

/-- The amount of money Frank had saved initially -/
def initial_savings : ℕ := sorry

/-- The cost of one toy -/
def toy_cost : ℕ := 8

/-- The number of toys Frank could buy -/
def num_toys : ℕ := 5

/-- The additional allowance Frank received -/
def additional_allowance : ℕ := 37

/-- Theorem stating that Frank's initial savings is $3 -/
theorem franks_savings : 
  (initial_savings + additional_allowance = num_toys * toy_cost) → 
  initial_savings = 3 := by sorry

end NUMINAMATH_CALUDE_franks_savings_l3295_329547


namespace NUMINAMATH_CALUDE_product_expansion_sum_l3295_329524

theorem product_expansion_sum (a b c d : ℝ) : 
  (∀ x, (2 * x^2 - 3 * x + 5) * (5 - x) = a * x^3 + b * x^2 + c * x + d) →
  27 * a + 9 * b + 3 * c + d = 28 := by
sorry

end NUMINAMATH_CALUDE_product_expansion_sum_l3295_329524


namespace NUMINAMATH_CALUDE_prob_kings_or_aces_l3295_329573

/-- A standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of kings in a standard deck -/
def KingsInDeck : ℕ := 4

/-- Number of aces in a standard deck -/
def AcesInDeck : ℕ := 4

/-- Number of cards drawn -/
def CardsDrawn : ℕ := 3

/-- Probability of drawing three kings -/
def probThreeKings : ℚ :=
  (KingsInDeck / StandardDeck) * ((KingsInDeck - 1) / (StandardDeck - 1)) * ((KingsInDeck - 2) / (StandardDeck - 2))

/-- Probability of drawing exactly two aces -/
def probTwoAces : ℚ :=
  3 * (AcesInDeck / StandardDeck) * ((AcesInDeck - 1) / (StandardDeck - 1)) * ((StandardDeck - AcesInDeck) / (StandardDeck - 2))

/-- Probability of drawing three aces -/
def probThreeAces : ℚ :=
  (AcesInDeck / StandardDeck) * ((AcesInDeck - 1) / (StandardDeck - 1)) * ((AcesInDeck - 2) / (StandardDeck - 2))

/-- The probability of drawing either three kings or at least 2 aces when selecting 3 cards from a standard 52-card deck -/
theorem prob_kings_or_aces : probThreeKings + probTwoAces + probThreeAces = 43 / 33150 := by
  sorry

end NUMINAMATH_CALUDE_prob_kings_or_aces_l3295_329573


namespace NUMINAMATH_CALUDE_no_valid_arrangement_l3295_329506

/-- Represents a seating arrangement of delegates around a circular table. -/
def SeatingArrangement := Fin 54 → Fin 27

/-- Checks if two positions in the circular table are separated by exactly 9 other positions. -/
def isSeparatedByNine (a b : Fin 54) : Prop :=
  (b - a) % 54 = 10 ∨ (a - b) % 54 = 10

/-- Represents a valid seating arrangement where each country's delegates are separated by 9 others. -/
def isValidArrangement (arrangement : SeatingArrangement) : Prop :=
  ∀ country : Fin 27, ∃ a b : Fin 54,
    a ≠ b ∧
    arrangement a = country ∧
    arrangement b = country ∧
    isSeparatedByNine a b

/-- Theorem stating that a valid seating arrangement is impossible. -/
theorem no_valid_arrangement : ¬∃ arrangement : SeatingArrangement, isValidArrangement arrangement := by
  sorry

end NUMINAMATH_CALUDE_no_valid_arrangement_l3295_329506


namespace NUMINAMATH_CALUDE_geometric_progression_sum_change_l3295_329519

/-- Given a geometric progression with 3000 terms, all positive, prove that
    if increasing every third term by 50 times increases the sum by 10 times,
    then doubling every even term increases the sum by 11/8 times. -/
theorem geometric_progression_sum_change (b₁ : ℝ) (q : ℝ) (S : ℝ) : 
  b₁ > 0 ∧ q > 0 ∧ S > 0 →
  S = b₁ * (1 - q^3000) / (1 - q) →
  S + 49 * b₁ * q^2 * (1 - q^3000) / ((1 - q) * (1 + q + q^2)) = 10 * S →
  S + 2 * b₁ * q * (1 - q^3000) / (1 - q^2) = 11 * S / 8 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_sum_change_l3295_329519


namespace NUMINAMATH_CALUDE_product_of_differences_equals_seven_l3295_329576

theorem product_of_differences_equals_seven
  (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ)
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2016)
  (h₂ : y₁^3 - 3*x₁^2*y₁ = 2000)
  (h₃ : x₂^3 - 3*x₂*y₂^2 = 2016)
  (h₄ : y₂^3 - 3*x₂^2*y₂ = 2000)
  (h₅ : x₃^3 - 3*x₃*y₃^2 = 2016)
  (h₆ : y₃^3 - 3*x₃^2*y₃ = 2000)
  (h₇ : y₁ ≠ 0)
  (h₈ : y₂ ≠ 0)
  (h₉ : y₃ ≠ 0) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = 7 := by
sorry

end NUMINAMATH_CALUDE_product_of_differences_equals_seven_l3295_329576


namespace NUMINAMATH_CALUDE_intersection_A_B_l3295_329556

def U : Set Nat := {0, 1, 3, 7, 9}
def C_UA : Set Nat := {0, 5, 9}
def B : Set Nat := {3, 5, 7}
def A : Set Nat := U \ C_UA

theorem intersection_A_B : A ∩ B = {3, 7} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3295_329556


namespace NUMINAMATH_CALUDE_shopping_expenditure_l3295_329516

theorem shopping_expenditure (x : ℝ) 
  (h1 : x + 10 + 40 = 100) 
  (h2 : 0.04 * x + 0.08 * 40 = 5.2) : x = 50 := by
  sorry

end NUMINAMATH_CALUDE_shopping_expenditure_l3295_329516


namespace NUMINAMATH_CALUDE_brent_nerds_count_l3295_329554

/-- Represents the candy inventory of Brent --/
structure CandyInventory where
  kitKat : ℕ
  hersheyKisses : ℕ
  nerds : ℕ
  lollipops : ℕ
  babyRuths : ℕ
  reeseCups : ℕ

/-- Calculates the total number of candy pieces --/
def totalCandy (inventory : CandyInventory) : ℕ :=
  inventory.kitKat + inventory.hersheyKisses + inventory.nerds + 
  inventory.lollipops + inventory.babyRuths + inventory.reeseCups

/-- Theorem stating that Brent received 8 boxes of Nerds --/
theorem brent_nerds_count : ∃ (inventory : CandyInventory),
  inventory.kitKat = 5 ∧
  inventory.hersheyKisses = 3 * inventory.kitKat ∧
  inventory.lollipops = 11 ∧
  inventory.babyRuths = 10 ∧
  inventory.reeseCups = inventory.babyRuths / 2 ∧
  totalCandy inventory - 5 = 49 ∧
  inventory.nerds = 8 := by
  sorry

end NUMINAMATH_CALUDE_brent_nerds_count_l3295_329554


namespace NUMINAMATH_CALUDE_zoo_feeding_sequences_l3295_329588

def number_of_animal_pairs : ℕ := 5

def alternating_feeding_sequences (n : ℕ) : ℕ :=
  (Nat.factorial n) * (Nat.factorial n)

theorem zoo_feeding_sequences :
  alternating_feeding_sequences number_of_animal_pairs = 14400 :=
by sorry

end NUMINAMATH_CALUDE_zoo_feeding_sequences_l3295_329588


namespace NUMINAMATH_CALUDE_audiobook_length_l3295_329589

/-- Proves that if a person listens to audiobooks for a certain amount of time each day
    and completes a certain number of audiobooks in a given number of days,
    then each audiobook has a specific length. -/
theorem audiobook_length
  (daily_listening_hours : ℝ)
  (total_days : ℕ)
  (num_audiobooks : ℕ)
  (h1 : daily_listening_hours = 2)
  (h2 : total_days = 90)
  (h3 : num_audiobooks = 6)
  : (daily_listening_hours * total_days) / num_audiobooks = 30 := by
  sorry

end NUMINAMATH_CALUDE_audiobook_length_l3295_329589


namespace NUMINAMATH_CALUDE_class_size_l3295_329504

theorem class_size (initial_avg : ℝ) (misread_weight : ℝ) (correct_weight : ℝ) (final_avg : ℝ) :
  initial_avg = 58.4 →
  misread_weight = 56 →
  correct_weight = 66 →
  final_avg = 58.9 →
  ∃ n : ℕ, n > 0 ∧ n * initial_avg + (correct_weight - misread_weight) = n * final_avg ∧ n = 20 :=
by sorry

end NUMINAMATH_CALUDE_class_size_l3295_329504


namespace NUMINAMATH_CALUDE_max_contribution_scenario_l3295_329513

/-- Represents the maximum possible contribution by a single person given the total contribution and number of people. -/
def max_contribution (total : ℝ) (num_people : ℕ) (min_contribution : ℝ) : ℝ :=
  total - (min_contribution * (num_people - 1 : ℝ))

/-- Theorem stating the maximum possible contribution in the given scenario. -/
theorem max_contribution_scenario :
  max_contribution 20 10 1 = 11 := by
  sorry

end NUMINAMATH_CALUDE_max_contribution_scenario_l3295_329513


namespace NUMINAMATH_CALUDE_prob_spade_or_king_l3295_329546

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of spades in a standard deck -/
def num_spades : ℕ := 13

/-- The number of kings in a standard deck -/
def num_kings : ℕ := 4

/-- The number of cards that are both spades and kings -/
def overlap : ℕ := 1

/-- The probability of drawing a spade or a king from a standard 52-card deck -/
theorem prob_spade_or_king : 
  (num_spades + num_kings - overlap : ℚ) / deck_size = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_prob_spade_or_king_l3295_329546


namespace NUMINAMATH_CALUDE_prime_sum_squares_l3295_329545

theorem prime_sum_squares (a b c d : ℕ) : 
  Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c ∧ Nat.Prime d →
  a > 3 →
  b > 6 →
  c > 12 →
  a^2 - b^2 + c^2 - d^2 = 1749 →
  a^2 + b^2 + c^2 + d^2 = 1999 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_squares_l3295_329545


namespace NUMINAMATH_CALUDE_quadratic_roots_and_k_value_l3295_329569

-- Define the quadratic equation
def quadratic (k x : ℝ) : ℝ := x^2 + (2*k + 1)*x + k^2 + 1

-- Theorem statement
theorem quadratic_roots_and_k_value (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic k x₁ = 0 ∧ quadratic k x₂ = 0) ↔ k > 3/4 ∧
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic k x₁ = 0 ∧ quadratic k x₂ = 0 ∧ x₁ * x₂ = 5) → k = 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_and_k_value_l3295_329569


namespace NUMINAMATH_CALUDE_hex_numeric_count_and_sum_l3295_329596

/-- Represents a hexadecimal digit --/
inductive HexDigit
| Numeric (n : Fin 10)
| Alpha (a : Fin 6)

/-- Converts a natural number to its hexadecimal representation --/
def toHex (n : ℕ) : List HexDigit := sorry

/-- Checks if a hexadecimal representation contains only numeric digits --/
def onlyNumeric (hex : List HexDigit) : Bool := sorry

/-- Counts the number of positive integers up to n whose hexadecimal 
    representation contains only numeric digits --/
def countNumericHex (n : ℕ) : ℕ := sorry

/-- Calculates the sum of digits of a natural number --/
def sumOfDigits (n : ℕ) : ℕ := sorry

theorem hex_numeric_count_and_sum : 
  countNumericHex 2000 = 1999 ∧ sumOfDigits 1999 = 28 := by sorry

end NUMINAMATH_CALUDE_hex_numeric_count_and_sum_l3295_329596


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l3295_329533

/-- Two-dimensional vector type -/
def Vec2 := ℝ × ℝ

/-- Dot product of two 2D vectors -/
def dot_product (v w : Vec2) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- Perpendicularity of two 2D vectors -/
def perpendicular (v w : Vec2) : Prop :=
  dot_product v w = 0

theorem perpendicular_vectors_m_value (m : ℝ) :
  let a : Vec2 := (1, m)
  let b : Vec2 := (2, -m)
  perpendicular a b → m = -Real.sqrt 2 ∨ m = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l3295_329533


namespace NUMINAMATH_CALUDE_sin_315_degrees_l3295_329550

theorem sin_315_degrees : 
  Real.sin (315 * π / 180) = -Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_315_degrees_l3295_329550


namespace NUMINAMATH_CALUDE_linear_equation_solution_l3295_329536

theorem linear_equation_solution : 
  ∃ x : ℚ, (x - 75) / 4 = (5 - 3 * x) / 7 ∧ x = 545 / 19 := by sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l3295_329536


namespace NUMINAMATH_CALUDE_fuse_length_safety_l3295_329587

theorem fuse_length_safety (safe_distance : ℝ) (fuse_speed : ℝ) (operator_speed : ℝ) 
  (h1 : safe_distance = 400)
  (h2 : fuse_speed = 1.2)
  (h3 : operator_speed = 5) :
  ∃ (min_length : ℝ), min_length > 96 ∧ 
  ∀ (fuse_length : ℝ), fuse_length > min_length → 
  (fuse_length / fuse_speed) > (safe_distance / operator_speed) := by
  sorry

end NUMINAMATH_CALUDE_fuse_length_safety_l3295_329587


namespace NUMINAMATH_CALUDE_number_of_placements_is_36_l3295_329563

/-- The number of ways to place 3 men and 4 women into groups -/
def number_of_placements : ℕ :=
  let num_men : ℕ := 3
  let num_women : ℕ := 4
  let num_groups_of_two : ℕ := 2
  let num_groups_of_three : ℕ := 1
  let ways_to_choose_man_for_three : ℕ := Nat.choose num_men 1
  let ways_to_choose_women_for_three : ℕ := Nat.choose num_women 2
  let ways_to_pair_remaining : ℕ := 2
  ways_to_choose_man_for_three * ways_to_choose_women_for_three * ways_to_pair_remaining

/-- Theorem stating that the number of placements is 36 -/
theorem number_of_placements_is_36 : number_of_placements = 36 := by
  sorry

end NUMINAMATH_CALUDE_number_of_placements_is_36_l3295_329563


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3295_329544

theorem least_subtraction_for_divisibility : ∃! k : ℕ, 
  k ≤ 16 ∧ (762429836 - k) % 17 = 0 ∧ 
  ∀ m : ℕ, m < k → (762429836 - m) % 17 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3295_329544


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_l3295_329539

theorem cube_sum_reciprocal (x : ℝ) (h : x + 1/x = 4) : x^3 + 1/x^3 = 52 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_l3295_329539


namespace NUMINAMATH_CALUDE_triangle_angle_and_perimeter_l3295_329551

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

theorem triangle_angle_and_perimeter (t : Triangle) (h : t.b^2 + t.c^2 - t.a^2 = t.b * t.c) :
  t.A = π / 3 ∧ 
  (t.a = Real.sqrt 3 → ∃ y : ℝ, y > 2 * Real.sqrt 3 ∧ y ≤ 3 * Real.sqrt 3 ∧ y = t.a + t.b + t.c) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_and_perimeter_l3295_329551


namespace NUMINAMATH_CALUDE_paula_four_hops_l3295_329517

def hop_distance (goal : ℚ) (remaining : ℚ) : ℚ :=
  (1 / 4) * remaining

def remaining_distance (goal : ℚ) (hopped : ℚ) : ℚ :=
  goal - hopped

theorem paula_four_hops :
  let goal : ℚ := 2
  let hop1 := hop_distance goal goal
  let hop2 := hop_distance goal (remaining_distance goal hop1)
  let hop3 := hop_distance goal (remaining_distance goal (hop1 + hop2))
  let hop4 := hop_distance goal (remaining_distance goal (hop1 + hop2 + hop3))
  hop1 + hop2 + hop3 + hop4 = 175 / 128 := by
  sorry

end NUMINAMATH_CALUDE_paula_four_hops_l3295_329517


namespace NUMINAMATH_CALUDE_parabola_through_point_l3295_329535

/-- A parabola passing through the point (4, -2) has either the equation y² = x or x² = -8y -/
theorem parabola_through_point (P : ℝ × ℝ) (h : P = (4, -2)) :
  (∃ (x y : ℝ), y^2 = x ∧ P = (x, y)) ∨ (∃ (x y : ℝ), x^2 = -8*y ∧ P = (x, y)) := by
  sorry

end NUMINAMATH_CALUDE_parabola_through_point_l3295_329535


namespace NUMINAMATH_CALUDE_total_crayons_l3295_329572

theorem total_crayons (billy jane mike sue : ℕ) 
  (h1 : billy = 62) 
  (h2 : jane = 52) 
  (h3 : mike = 78) 
  (h4 : sue = 97) : 
  billy + jane + mike + sue = 289 := by
  sorry

end NUMINAMATH_CALUDE_total_crayons_l3295_329572


namespace NUMINAMATH_CALUDE_bankers_gain_calculation_l3295_329591

/-- Banker's gain calculation -/
theorem bankers_gain_calculation (P TD : ℚ) (h1 : P = 576) (h2 : TD = 96) :
  TD^2 / P = 16 := by sorry

end NUMINAMATH_CALUDE_bankers_gain_calculation_l3295_329591


namespace NUMINAMATH_CALUDE_smallest_x_y_sum_l3295_329586

def is_fourth_power (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^4

def is_sixth_power (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^6

theorem smallest_x_y_sum :
  ∃ (x y : ℕ),
    (∀ x' : ℕ, is_fourth_power (180 * x') → x ≤ x') ∧
    (∀ y' : ℕ, is_sixth_power (180 * y') → y ≤ y') ∧
    is_fourth_power (180 * x) ∧
    is_sixth_power (180 * y) ∧
    x + y = 4054500 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_y_sum_l3295_329586


namespace NUMINAMATH_CALUDE_pants_bought_l3295_329503

def total_cost : ℕ := 1500
def tshirt_cost : ℕ := 100
def pants_cost : ℕ := 250
def num_tshirts : ℕ := 5

theorem pants_bought :
  (total_cost - num_tshirts * tshirt_cost) / pants_cost = 4 := by
sorry

end NUMINAMATH_CALUDE_pants_bought_l3295_329503


namespace NUMINAMATH_CALUDE_hari_contribution_is_2160_l3295_329510

/-- Represents the investment details and profit sharing ratio --/
structure InvestmentDetails where
  praveen_investment : ℕ
  praveen_months : ℕ
  hari_months : ℕ
  total_months : ℕ
  profit_ratio_praveen : ℕ
  profit_ratio_hari : ℕ

/-- Calculates Hari's contribution given the investment details --/
def calculate_hari_contribution (details : InvestmentDetails) : ℕ :=
  (details.praveen_investment * details.praveen_months * details.profit_ratio_hari) /
  (details.profit_ratio_praveen * details.hari_months)

/-- Theorem stating that Hari's contribution is 2160 given the problem conditions --/
theorem hari_contribution_is_2160 :
  let details : InvestmentDetails := {
    praveen_investment := 3360,
    praveen_months := 12,
    hari_months := 7,
    total_months := 12,
    profit_ratio_praveen := 2,
    profit_ratio_hari := 3
  }
  calculate_hari_contribution details = 2160 := by
  sorry

#eval calculate_hari_contribution {
  praveen_investment := 3360,
  praveen_months := 12,
  hari_months := 7,
  total_months := 12,
  profit_ratio_praveen := 2,
  profit_ratio_hari := 3
}

end NUMINAMATH_CALUDE_hari_contribution_is_2160_l3295_329510


namespace NUMINAMATH_CALUDE_four_digit_sum_l3295_329507

/-- The number of four-digit even numbers -/
def C : ℕ := 4500

/-- The number of four-digit numbers that are multiples of 7 -/
def D : ℕ := 1285

/-- Theorem stating that the sum of four-digit even numbers and four-digit multiples of 7 is 5785 -/
theorem four_digit_sum : C + D = 5785 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_sum_l3295_329507


namespace NUMINAMATH_CALUDE_find_a_l3295_329515

-- Define the sets A and B
def A (a : ℤ) : Set ℤ := {1, 3, a}
def B (a : ℤ) : Set ℤ := {1, a^2}

-- State the theorem
theorem find_a : 
  ∀ a : ℤ, (A a ∪ B a = {1, 3, a}) → (a = 0 ∨ a = 1 ∨ a = -1) :=
by sorry

end NUMINAMATH_CALUDE_find_a_l3295_329515


namespace NUMINAMATH_CALUDE_binary_21_l3295_329566

/-- The binary representation of a natural number. -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinaryAux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
  toBinaryAux n

/-- Proposition: The binary representation of 21 is [true, false, true, false, true] -/
theorem binary_21 : toBinary 21 = [true, false, true, false, true] := by
  sorry

end NUMINAMATH_CALUDE_binary_21_l3295_329566


namespace NUMINAMATH_CALUDE_article_price_fraction_l3295_329541

/-- Proves that selling an article at 2/3 of its original price results in a 10% loss,
    given that the original price has a 35% markup from the cost price. -/
theorem article_price_fraction (original_price cost_price : ℝ) :
  original_price = cost_price * (1 + 35 / 100) →
  original_price * (2 / 3) = cost_price * (1 - 10 / 100) := by
  sorry

end NUMINAMATH_CALUDE_article_price_fraction_l3295_329541


namespace NUMINAMATH_CALUDE_pentagon_diagonals_l3295_329567

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A pentagon has 5 sides -/
def pentagon_sides : ℕ := 5

/-- Theorem: The number of diagonals in a pentagon is 5 -/
theorem pentagon_diagonals : num_diagonals pentagon_sides = 5 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_diagonals_l3295_329567


namespace NUMINAMATH_CALUDE_soccer_score_theorem_l3295_329558

/-- Represents the scores of a soccer player -/
structure SoccerScores where
  game6 : ℕ
  game7 : ℕ
  game8 : ℕ
  game9 : ℕ
  first6GamesTotal : ℕ
  game10 : ℕ

/-- The minimum number of points scored in the 10th game -/
def minGame10Score (s : SoccerScores) : Prop :=
  s.game10 = 13

/-- The given conditions of the problem -/
def problemConditions (s : SoccerScores) : Prop :=
  s.game6 = 18 ∧
  s.game7 = 25 ∧
  s.game8 = 15 ∧
  s.game9 = 22 ∧
  (s.first6GamesTotal + s.game6 + s.game7 + s.game8 + s.game9 + s.game10) / 10 >
    (s.first6GamesTotal + s.game6) / 6 ∧
  (s.first6GamesTotal + s.game6 + s.game7 + s.game8 + s.game9 + s.game10) / 10 > 20

theorem soccer_score_theorem (s : SoccerScores) :
  problemConditions s → minGame10Score s := by
  sorry

end NUMINAMATH_CALUDE_soccer_score_theorem_l3295_329558


namespace NUMINAMATH_CALUDE_three_planes_theorem_l3295_329561

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- Three non-coplanar lines through a point -/
structure ThreeLines where
  point : Point3D
  line1 : Line3D
  line2 : Line3D
  line3 : Line3D
  non_coplanar : Line3D → Line3D → Line3D → Prop

/-- A plane in 3D space -/
structure Plane3D where
  normal : Point3D
  point : Point3D

/-- The number of planes determined by three lines -/
def planes_determined_by_lines (lines : ThreeLines) : ℕ :=
  3

theorem three_planes_theorem (lines : ThreeLines) :
  planes_determined_by_lines lines = 3 :=
sorry

end NUMINAMATH_CALUDE_three_planes_theorem_l3295_329561


namespace NUMINAMATH_CALUDE_canada_animal_population_l3295_329518

/-- Represents the population of different species in Canada -/
structure CanadaPopulation where
  humans : ℚ
  moose : ℚ
  beavers : ℚ
  caribou : ℚ
  wolves : ℚ
  grizzly_bears : ℚ

/-- The relationships between species in Canada -/
def population_relationships (p : CanadaPopulation) : Prop :=
  p.beavers = 2 * p.moose ∧
  p.humans = 19 * p.beavers ∧
  3 * p.caribou = 2 * p.moose ∧
  p.wolves = 4 * p.caribou ∧
  3 * p.grizzly_bears = p.wolves

/-- The theorem stating the combined population of animals given the human population -/
theorem canada_animal_population 
  (p : CanadaPopulation) 
  (h : population_relationships p) 
  (humans_pop : p.humans = 38) : 
  p.moose + p.beavers + p.caribou + p.wolves + p.grizzly_bears = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_canada_animal_population_l3295_329518


namespace NUMINAMATH_CALUDE_multiple_algorithms_exist_l3295_329505

/-- Represents a problem type -/
structure ProblemType where
  description : String

/-- Represents an algorithm -/
structure Algorithm where
  steps : List String

/-- Predicate to check if an algorithm solves a problem type -/
def solves (a : Algorithm) (p : ProblemType) : Prop :=
  sorry  -- Definition of what it means for an algorithm to solve a problem

/-- Theorem: There can exist multiple valid algorithms for a given problem type -/
theorem multiple_algorithms_exist (p : ProblemType) : 
  ∃ (a1 a2 : Algorithm), a1 ≠ a2 ∧ solves a1 p ∧ solves a2 p := by
  sorry

#check multiple_algorithms_exist

end NUMINAMATH_CALUDE_multiple_algorithms_exist_l3295_329505


namespace NUMINAMATH_CALUDE_double_box_11_l3295_329565

def box_sum (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem double_box_11 : box_sum (box_sum 11) = 28 := by
  sorry

end NUMINAMATH_CALUDE_double_box_11_l3295_329565


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3295_329580

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ 2 + a^(x - 1)
  f 1 = 3 :=
sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3295_329580


namespace NUMINAMATH_CALUDE_prime_difference_values_l3295_329595

theorem prime_difference_values (p q : ℕ) (n : ℕ+) 
  (h_p : Nat.Prime p) (h_q : Nat.Prime q) 
  (h_eq : (p : ℚ) / (p + 1) + (q + 1 : ℚ) / q = (2 * n) / (n + 2)) :
  q - p ∈ ({2, 3, 5} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_prime_difference_values_l3295_329595
