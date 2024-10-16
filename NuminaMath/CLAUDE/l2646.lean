import Mathlib

namespace NUMINAMATH_CALUDE_difference_between_numbers_l2646_264620

theorem difference_between_numbers (a b : ℕ) (h1 : a + b = 84) (h2 : a = 36) (h3 : b = 48) :
  b - a = 12 := by
sorry

end NUMINAMATH_CALUDE_difference_between_numbers_l2646_264620


namespace NUMINAMATH_CALUDE_smallest_q_in_geometric_sequence_l2646_264669

def is_geometric_sequence (p q r : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ q = p * k ∧ r = q * k

theorem smallest_q_in_geometric_sequence (p q r : ℝ) :
  p > 0 → q > 0 → r > 0 →
  is_geometric_sequence p q r →
  p * q * r = 216 →
  q ≥ 6 ∧ ∃ p' q' r' : ℝ, p' > 0 ∧ q' > 0 ∧ r' > 0 ∧
    is_geometric_sequence p' q' r' ∧ p' * q' * r' = 216 ∧ q' = 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_q_in_geometric_sequence_l2646_264669


namespace NUMINAMATH_CALUDE_left_handed_jazz_lovers_l2646_264652

/-- Represents a club with members and their music preferences -/
structure Club where
  total_members : ℕ
  left_handed : ℕ
  jazz_lovers : ℕ
  right_handed_dislike_both : ℕ

/-- Theorem stating the number of left-handed jazz lovers in the club -/
theorem left_handed_jazz_lovers (c : Club)
  (h1 : c.total_members = 25)
  (h2 : c.left_handed = 10)
  (h3 : c.jazz_lovers = 18)
  (h4 : c.right_handed_dislike_both = 3)
  (h5 : c.left_handed + (c.total_members - c.left_handed) = c.total_members)
  (h6 : ∀ (m : ℕ), m < c.total_members → 
    (m ∈ (Finset.range c.jazz_lovers) ∨ 
     m ∈ (Finset.range (c.total_members - c.jazz_lovers - c.right_handed_dislike_both))))
  : {x : ℕ // x = 10 ∧ x ≤ c.left_handed ∧ x ≤ c.jazz_lovers} :=
by
  sorry

#check left_handed_jazz_lovers

end NUMINAMATH_CALUDE_left_handed_jazz_lovers_l2646_264652


namespace NUMINAMATH_CALUDE_reading_pages_in_seven_days_l2646_264673

theorem reading_pages_in_seven_days 
  (total_hours : ℕ) 
  (days : ℕ) 
  (pages_per_hour : ℕ) 
  (h1 : total_hours = 10)
  (h2 : days = 5)
  (h3 : pages_per_hour = 50) : 
  (total_hours / days) * pages_per_hour * 7 = 700 := by
  sorry

end NUMINAMATH_CALUDE_reading_pages_in_seven_days_l2646_264673


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2646_264690

open Set Real

noncomputable def A : Set ℝ := {x | x^2 < 1}
noncomputable def B : Set ℝ := {x | x^2 - 2*x > 0}

theorem intersection_A_complement_B :
  A ∩ (𝒰 \ B) = Icc 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2646_264690


namespace NUMINAMATH_CALUDE_homework_problem_count_l2646_264606

theorem homework_problem_count (p t : ℕ) : 
  p > 0 → 
  t > 0 → 
  p ≥ 15 → 
  p * t = (2 * p - 10) * (t - 1) → 
  p * t = 60 := by
  sorry

end NUMINAMATH_CALUDE_homework_problem_count_l2646_264606


namespace NUMINAMATH_CALUDE_fraction_sum_l2646_264637

theorem fraction_sum (m n : ℕ) (hcoprime : Nat.Coprime m n) 
  (heq : (2013 * 2013) / (2014 * 2014 + 2012) = n / m) : m + n = 1343 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l2646_264637


namespace NUMINAMATH_CALUDE_drive_time_between_towns_l2646_264666

/-- Proves that the time to drive between two towns is 4 hours given the map distance, scale, and average speed. -/
theorem drive_time_between_towns
  (map_distance : ℝ)
  (scale_distance : ℝ)
  (scale_miles : ℝ)
  (average_speed : ℝ)
  (h1 : map_distance = 12)
  (h2 : scale_distance = 0.5)
  (h3 : scale_miles = 10)
  (h4 : average_speed = 60)
  : (map_distance * scale_miles / scale_distance) / average_speed = 4 :=
by
  sorry

#check drive_time_between_towns

end NUMINAMATH_CALUDE_drive_time_between_towns_l2646_264666


namespace NUMINAMATH_CALUDE_sqrt_combinable_with_sqrt_two_l2646_264647

theorem sqrt_combinable_with_sqrt_two : ∃! x : ℝ, 
  (x = Real.sqrt 10 ∨ x = Real.sqrt 12 ∨ x = Real.sqrt (1/2) ∨ x = 1 / Real.sqrt 6) ∧
  ∃ (a : ℝ), x = a * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_combinable_with_sqrt_two_l2646_264647


namespace NUMINAMATH_CALUDE_modulus_complex_power_eight_l2646_264614

theorem modulus_complex_power_eight :
  Complex.abs ((2 : ℂ) + Complex.I) ^ 8 = 625 := by
  sorry

end NUMINAMATH_CALUDE_modulus_complex_power_eight_l2646_264614


namespace NUMINAMATH_CALUDE_area_AXYD_area_AXYD_is_72_l2646_264686

/-- Rectangle ABCD with given dimensions and point E -/
structure Rectangle :=
  (A B C D E : ℝ × ℝ)
  (AB : ℝ)
  (BC : ℝ)

/-- Point Z on the extension of BC -/
def Z (rect : Rectangle) : ℝ × ℝ := (rect.C.1, rect.C.2 + 18)

/-- Conditions for the rectangle and point E -/
def validRectangle (rect : Rectangle) : Prop :=
  rect.AB = 20 ∧
  rect.BC = 12 ∧
  rect.A = (0, 0) ∧
  rect.B = (20, 0) ∧
  rect.C = (20, 12) ∧
  rect.D = (0, 12) ∧
  rect.E = (6, 6)

/-- Theorem: Area of quadrilateral AXYD is 72 -/
theorem area_AXYD (rect : Rectangle) (h : validRectangle rect) : ℝ :=
  72

/-- Main theorem: If the rectangle satisfies the conditions, then the area of AXYD is 72 -/
theorem area_AXYD_is_72 (rect : Rectangle) (h : validRectangle rect) : 
  area_AXYD rect h = 72 := by
  sorry

end NUMINAMATH_CALUDE_area_AXYD_area_AXYD_is_72_l2646_264686


namespace NUMINAMATH_CALUDE_unique_three_config_score_l2646_264653

/-- Represents a quiz score configuration -/
structure QuizScore where
  correct : ℕ
  unanswered : ℕ
  incorrect : ℕ

/-- The scoring system for the quiz -/
def score (qs : QuizScore) : ℚ :=
  5 * qs.correct + 1.5 * qs.unanswered

/-- Predicate to check if a QuizScore is valid -/
def is_valid_score (qs : QuizScore) : Prop :=
  qs.correct + qs.unanswered + qs.incorrect = 20

/-- Predicate to check if a rational number is a possible quiz score -/
def is_possible_score (s : ℚ) : Prop :=
  ∃ qs : QuizScore, is_valid_score qs ∧ score qs = s

/-- Predicate to check if a rational number has exactly three distinct valid quiz configurations -/
def has_three_configurations (s : ℚ) : Prop :=
  ∃ qs1 qs2 qs3 : QuizScore,
    is_valid_score qs1 ∧ is_valid_score qs2 ∧ is_valid_score qs3 ∧
    score qs1 = s ∧ score qs2 = s ∧ score qs3 = s ∧
    qs1 ≠ qs2 ∧ qs1 ≠ qs3 ∧ qs2 ≠ qs3 ∧
    ∀ qs : QuizScore, is_valid_score qs → score qs = s → (qs = qs1 ∨ qs = qs2 ∨ qs = qs3)

theorem unique_three_config_score :
  ∀ s : ℚ, 0 ≤ s ∧ s ≤ 100 → has_three_configurations s → s = 75 :=
sorry

end NUMINAMATH_CALUDE_unique_three_config_score_l2646_264653


namespace NUMINAMATH_CALUDE_negation_of_proposition_is_true_l2646_264654

theorem negation_of_proposition_is_true :
  let p := (∀ x y : ℝ, x + y = 5 → x = 2 ∧ y = 3)
  ¬p = True :=
by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_is_true_l2646_264654


namespace NUMINAMATH_CALUDE_position_of_2005_2004_l2646_264663

/-- The sum of numerator and denominator for the fraction 2005/2004 -/
def target_sum : ℕ := 2005 + 2004

/-- The position of a fraction in the sequence -/
def position (n d : ℕ) : ℕ :=
  let s := n + d
  (s - 1) * (s - 2) / 2 + (s - n)

/-- The theorem stating the position of 2005/2004 in the sequence -/
theorem position_of_2005_2004 : position 2005 2004 = 8028032 := by
  sorry


end NUMINAMATH_CALUDE_position_of_2005_2004_l2646_264663


namespace NUMINAMATH_CALUDE_complement_union_theorem_l2646_264600

-- Define the universal set U
def U : Finset Nat := {0, 1, 2, 3}

-- Define set A
def A : Finset Nat := {1, 2}

-- Define set B
def B : Finset Nat := {2, 3}

-- Theorem statement
theorem complement_union_theorem :
  (U \ A) ∪ B = {0, 2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l2646_264600


namespace NUMINAMATH_CALUDE_min_value_f_l2646_264660

/-- A function f(x) = x^2 + x + a with a maximum value of 2 on [-1, 1] -/
def f (a : ℝ) : ℝ → ℝ := λ x ↦ x^2 + x + a

/-- The maximum value of f on [-1, 1] is 2 -/
axiom max_value (a : ℝ) : ∃ x ∈ Set.Icc (-1 : ℝ) 1, ∀ y ∈ Set.Icc (-1 : ℝ) 1, f a y ≤ f a x ∧ f a x = 2

/-- The minimum value of f on [-1, 1] is -1/4 -/
theorem min_value_f (a : ℝ) : ∃ x ∈ Set.Icc (-1 : ℝ) 1, ∀ y ∈ Set.Icc (-1 : ℝ) 1, f a x ≤ f a y ∧ f a x = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_f_l2646_264660


namespace NUMINAMATH_CALUDE_remainder_problem_l2646_264639

theorem remainder_problem (N : ℤ) (h : N % 296 = 75) : N % 37 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2646_264639


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2646_264671

/-- Given a parabola and a hyperbola with coinciding foci, 
    prove that the eccentricity of the hyperbola is 2√3/3 -/
theorem hyperbola_eccentricity 
  (parabola : ℝ → ℝ) 
  (hyperbola : ℝ → ℝ → ℝ) 
  (a : ℝ) 
  (h1 : ∀ x y, parabola y = (1/8) * x^2)
  (h2 : ∀ x y, hyperbola x y = y^2/a - x^2 - 1)
  (h3 : ∃ x y, parabola y = (1/8) * x^2 ∧ hyperbola x y = 0 ∧ 
              x^2 + (y - a/2)^2 = (a/2)^2) : 
  ∃ e : ℝ, e = 2 * Real.sqrt 3 / 3 ∧ 
    ∀ x y, hyperbola x y = 0 → x^2/(a/e^2) + y^2/a = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2646_264671


namespace NUMINAMATH_CALUDE_cos_alpha_plus_5pi_12_l2646_264644

theorem cos_alpha_plus_5pi_12 (α : Real) (h : Real.sin (α - π/12) = 1/3) :
  Real.cos (α + 5*π/12) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_5pi_12_l2646_264644


namespace NUMINAMATH_CALUDE_f_sum_2009_2010_l2646_264676

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem f_sum_2009_2010 (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_period : has_period (fun x ↦ f (3*x + 1)) 2)
  (h_f_1 : f 1 = 2010) :
  f 2009 + f 2010 = -2010 := by sorry

end NUMINAMATH_CALUDE_f_sum_2009_2010_l2646_264676


namespace NUMINAMATH_CALUDE_sum_P_Q_equals_52_l2646_264668

theorem sum_P_Q_equals_52 (P Q : ℝ) :
  (∀ x : ℝ, x ≠ 3 → P / (x - 3) + Q * (x - 2) = (-4 * x^2 + 20 * x + 32) / (x - 3)) →
  P + Q = 52 := by
sorry

end NUMINAMATH_CALUDE_sum_P_Q_equals_52_l2646_264668


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2646_264645

/-- An arithmetic sequence with the given properties -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℚ) 
  (h_seq : ArithmeticSequence a) 
  (h_prod : a 7 * a 11 = 6) 
  (h_sum : a 4 + a 14 = 5) : 
  ∃ d : ℚ, (d = 1/4 ∨ d = -1/4) ∧ ∀ n : ℕ, a (n + 1) = a n + d := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2646_264645


namespace NUMINAMATH_CALUDE_tiffany_lives_lost_l2646_264609

theorem tiffany_lives_lost (initial_lives gained_lives final_lives : ℕ) 
  (h1 : initial_lives = 43)
  (h2 : gained_lives = 27)
  (h3 : final_lives = 56) :
  initial_lives - (final_lives - gained_lives) = 14 :=
by sorry

end NUMINAMATH_CALUDE_tiffany_lives_lost_l2646_264609


namespace NUMINAMATH_CALUDE_sum_squares_ge_sum_products_l2646_264657

theorem sum_squares_ge_sum_products (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^2 + b^2 + c^2 ≥ a*b + b*c + c*a := by
  sorry

end NUMINAMATH_CALUDE_sum_squares_ge_sum_products_l2646_264657


namespace NUMINAMATH_CALUDE_proper_subset_condition_l2646_264667

def A (a : ℝ) : Set ℝ := {1, 4, a}
def B (a : ℝ) : Set ℝ := {1, a^2}

def valid_a : Set ℝ := {-2, -1, 0, 1, 2}

theorem proper_subset_condition (a : ℝ) : 
  (B a ⊂ A a) ↔ a ∈ valid_a := by sorry

end NUMINAMATH_CALUDE_proper_subset_condition_l2646_264667


namespace NUMINAMATH_CALUDE_complex_in_second_quadrant_l2646_264681

/-- A complex number z is in the second quadrant if its real part is negative and its imaginary part is positive -/
def in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

/-- Given that (2a+2i)/(1+i) is purely imaginary for some real a, 
    prove that 2a+2i is in the second quadrant -/
theorem complex_in_second_quadrant (a : ℝ) 
    (h : (Complex.I : ℂ).re * ((2*a + 2*Complex.I) / (1 + Complex.I)).im = 
         (Complex.I : ℂ).im * ((2*a + 2*Complex.I) / (1 + Complex.I)).re) : 
    in_second_quadrant (2*a + 2*Complex.I) := by
  sorry


end NUMINAMATH_CALUDE_complex_in_second_quadrant_l2646_264681


namespace NUMINAMATH_CALUDE_train_length_l2646_264692

/-- The length of a train given its speed, platform length, and crossing time --/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  train_speed = 72 * (5 / 18) →
  platform_length = 230 →
  crossing_time = 26 →
  train_speed * crossing_time - platform_length = 290 := by
sorry

end NUMINAMATH_CALUDE_train_length_l2646_264692


namespace NUMINAMATH_CALUDE_coefficient_of_x_fifth_l2646_264662

theorem coefficient_of_x_fifth (a : ℝ) : 
  (Nat.choose 8 5) * a^5 = 56 → a = 1 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_fifth_l2646_264662


namespace NUMINAMATH_CALUDE_jims_pantry_flour_l2646_264672

/-- The amount of flour Jim has in the pantry -/
def flour_in_pantry : ℕ := sorry

/-- The total amount of flour Jim has in the cupboard and on the kitchen counter -/
def flour_elsewhere : ℕ := 300

/-- The amount of flour required for one loaf of bread -/
def flour_per_loaf : ℕ := 200

/-- The number of loaves Jim can bake -/
def loaves_baked : ℕ := 2

theorem jims_pantry_flour :
  flour_in_pantry = 100 :=
by sorry

end NUMINAMATH_CALUDE_jims_pantry_flour_l2646_264672


namespace NUMINAMATH_CALUDE_hcd_7560_180_minus_12_l2646_264646

theorem hcd_7560_180_minus_12 : Nat.gcd 7560 180 - 12 = 168 := by sorry

end NUMINAMATH_CALUDE_hcd_7560_180_minus_12_l2646_264646


namespace NUMINAMATH_CALUDE_marks_father_gave_85_l2646_264610

/-- The amount of money Mark's father gave him. -/
def fathers_money (num_books : ℕ) (book_price : ℕ) (money_left : ℕ) : ℕ :=
  num_books * book_price + money_left

/-- Theorem stating that Mark's father gave him $85. -/
theorem marks_father_gave_85 :
  fathers_money 10 5 35 = 85 := by
  sorry

end NUMINAMATH_CALUDE_marks_father_gave_85_l2646_264610


namespace NUMINAMATH_CALUDE_f_derivative_positive_implies_a_bound_l2646_264677

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * exp x - 2 * x^2

theorem f_derivative_positive_implies_a_bound (a : ℝ) :
  (∀ x₀ ∈ Set.Ioo 0 1, ∀ ε > 0, ∃ δ > 0, ∀ x ∈ Set.Ioo (x₀ - δ) (x₀ + δ),
    x ≠ x₀ → (f a x - f a x₀ - x + x₀) / (x - x₀) > 0) →
  a > 4 / exp (3/4) :=
sorry

end NUMINAMATH_CALUDE_f_derivative_positive_implies_a_bound_l2646_264677


namespace NUMINAMATH_CALUDE_handshake_count_l2646_264618

theorem handshake_count (n : ℕ) (h : n = 25) : 
  (n * (n - 1) / 2) * 3 = 900 := by
  sorry

end NUMINAMATH_CALUDE_handshake_count_l2646_264618


namespace NUMINAMATH_CALUDE_greatest_x_with_lcm_l2646_264658

theorem greatest_x_with_lcm (x : ℕ) : 
  (∃ m : ℕ, Nat.lcm x (Nat.lcm 15 21) = 105) → 
  x ≤ 105 ∧ 
  ∃ y : ℕ, y > 105 → ¬(∃ m : ℕ, Nat.lcm y (Nat.lcm 15 21) = 105) :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_with_lcm_l2646_264658


namespace NUMINAMATH_CALUDE_average_difference_l2646_264616

def even_integers_16_to_44 : List Int :=
  List.range 15 |> List.map (fun i => 16 + 2 * i)

def even_integers_14_to_56 : List Int :=
  List.range 22 |> List.map (fun i => 14 + 2 * i)

def average (l : List Int) : ℚ :=
  (l.sum : ℚ) / l.length

theorem average_difference :
  average even_integers_16_to_44 + 5 = average even_integers_14_to_56 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_l2646_264616


namespace NUMINAMATH_CALUDE_six_arts_arrangement_l2646_264611

def number_of_arrangements (n : ℕ) (k : ℕ) (m : ℕ) : ℕ :=
  -- n: total number of lectures
  -- k: position limit for the specific lecture (Mathematics)
  -- m: number of lectures that must be adjacent (Archery and Charioteering)
  sorry

theorem six_arts_arrangement : number_of_arrangements 6 3 2 = 120 := by
  sorry

end NUMINAMATH_CALUDE_six_arts_arrangement_l2646_264611


namespace NUMINAMATH_CALUDE_inequality_problem_l2646_264638

theorem inequality_problem (s r p q : ℝ) 
  (hs : s > 0) 
  (hr : r > 0) 
  (hpq : p * q ≠ 0) 
  (hineq : s * (p * r) > s * (q * r)) : 
  ¬(-p > -q) ∧ ¬(-p > q) ∧ ¬(1 > -q/p) ∧ ¬(1 < q/p) := by
  sorry

end NUMINAMATH_CALUDE_inequality_problem_l2646_264638


namespace NUMINAMATH_CALUDE_equation_solution_l2646_264693

theorem equation_solution (x y : ℝ) : 
  y = 3 * x → (5 * y^2 + y + 10 = 2 * (9 * x^2 + y + 6)) ↔ (x = 1/3 ∨ x = -2/9) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2646_264693


namespace NUMINAMATH_CALUDE_zane_picked_up_62_pounds_l2646_264687

/-- The amount of garbage picked up by Daliah in pounds -/
def daliah_garbage : ℝ := 17.5

/-- The amount of garbage picked up by Dewei in pounds -/
def dewei_garbage : ℝ := daliah_garbage - 2

/-- The amount of garbage picked up by Zane in pounds -/
def zane_garbage : ℝ := 4 * dewei_garbage

/-- Theorem stating that Zane picked up 62 pounds of garbage -/
theorem zane_picked_up_62_pounds : zane_garbage = 62 := by
  sorry

end NUMINAMATH_CALUDE_zane_picked_up_62_pounds_l2646_264687


namespace NUMINAMATH_CALUDE_cost_dozen_pens_l2646_264680

-- Define the cost of one pen and one pencil
def cost_pen : ℝ := sorry
def cost_pencil : ℝ := sorry

-- Define the conditions
axiom total_cost : 3 * cost_pen + 5 * cost_pencil = 100
axiom cost_ratio : cost_pen = 5 * cost_pencil

-- Theorem to prove
theorem cost_dozen_pens : 12 * cost_pen = 300 := by
  sorry

end NUMINAMATH_CALUDE_cost_dozen_pens_l2646_264680


namespace NUMINAMATH_CALUDE_chessboard_tiling_impossible_l2646_264691

/-- Represents a chessboard tile -/
inductive Tile
| L
| T

/-- Represents the color distribution of a tile placement -/
structure ColorDistribution :=
  (black : ℕ)
  (white : ℕ)

/-- The color distribution of an L-tile -/
def l_tile_distribution : ColorDistribution :=
  ⟨2, 2⟩

/-- The possible color distributions of a T-tile -/
def t_tile_distributions : List ColorDistribution :=
  [⟨3, 1⟩, ⟨1, 3⟩]

/-- The number of squares on the chessboard -/
def chessboard_squares : ℕ := 64

/-- The number of black squares on the chessboard -/
def chessboard_black_squares : ℕ := 32

/-- The number of white squares on the chessboard -/
def chessboard_white_squares : ℕ := 32

/-- The number of L-tiles -/
def num_l_tiles : ℕ := 15

/-- The number of T-tiles -/
def num_t_tiles : ℕ := 1

theorem chessboard_tiling_impossible :
  ∀ (t_dist : ColorDistribution),
    t_dist ∈ t_tile_distributions →
    (num_l_tiles * l_tile_distribution.black + t_dist.black ≠ chessboard_black_squares ∨
     num_l_tiles * l_tile_distribution.white + t_dist.white ≠ chessboard_white_squares) :=
by sorry

end NUMINAMATH_CALUDE_chessboard_tiling_impossible_l2646_264691


namespace NUMINAMATH_CALUDE_elliptical_machine_payment_l2646_264665

/-- Proves that the daily minimum payment for an elliptical machine is $6 given the specified conditions --/
theorem elliptical_machine_payment 
  (total_cost : ℝ) 
  (down_payment_ratio : ℝ) 
  (payment_days : ℕ) 
  (h1 : total_cost = 120) 
  (h2 : down_payment_ratio = 1/2) 
  (h3 : payment_days = 10) : 
  (total_cost * (1 - down_payment_ratio)) / payment_days = 6 := by
sorry

end NUMINAMATH_CALUDE_elliptical_machine_payment_l2646_264665


namespace NUMINAMATH_CALUDE_sum_of_diagonals_is_190_l2646_264607

/-- A hexagon inscribed in a circle -/
structure InscribedHexagon where
  -- Sides of the hexagon
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  side5 : ℝ
  side6 : ℝ
  -- Conditions on the sides
  h1 : side1 = 20
  h2 : side3 = 30
  h3 : side2 = 50
  h4 : side4 = 50
  h5 : side5 = 50
  h6 : side6 = 50

/-- The sum of diagonals from one vertex in the inscribed hexagon -/
def sumOfDiagonals (h : InscribedHexagon) : ℝ := sorry

/-- Theorem: The sum of diagonals from one vertex in the specified hexagon is 190 -/
theorem sum_of_diagonals_is_190 (h : InscribedHexagon) : sumOfDiagonals h = 190 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_diagonals_is_190_l2646_264607


namespace NUMINAMATH_CALUDE_smallest_solution_quadratic_l2646_264697

theorem smallest_solution_quadratic (x : ℝ) :
  (6 * x^2 - 37 * x + 48 = 0) → (x ≥ 13/6) :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_quadratic_l2646_264697


namespace NUMINAMATH_CALUDE_binary_110010_equals_50_l2646_264655

-- Define the binary number as a list of digits
def binary_number : List Nat := [1, 1, 0, 0, 1, 0]

-- Function to convert binary to decimal
def binary_to_decimal (binary : List Nat) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

-- Theorem statement
theorem binary_110010_equals_50 :
  binary_to_decimal binary_number = 50 := by
  sorry

end NUMINAMATH_CALUDE_binary_110010_equals_50_l2646_264655


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2646_264684

theorem inequality_solution_set (a : ℕ) : 
  (∀ x, (a - 2) * x > a - 2 ↔ x < 1) → (a = 0 ∨ a = 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2646_264684


namespace NUMINAMATH_CALUDE_group_5_frequency_l2646_264659

/-- The frequency of Group 5 in a class of 50 students divided into 5 groups -/
theorem group_5_frequency (total_students : ℕ) (group_1_freq : ℚ) (group_2_3_freq_sum : ℚ) (group_4_freq : ℚ) :
  total_students = 50 →
  group_1_freq = 7 / 50 →
  group_2_3_freq_sum = 46 / 100 →
  group_4_freq = 1 / 5 →
  1 - (group_1_freq + group_2_3_freq_sum + group_4_freq) = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_group_5_frequency_l2646_264659


namespace NUMINAMATH_CALUDE_correct_num_cats_l2646_264689

/-- Represents the number of cats on the ship -/
def num_cats : ℕ := 5

/-- Represents the number of sailors on the ship -/
def num_sailors : ℕ := 14 - num_cats

/-- The total number of heads on the ship -/
def total_heads : ℕ := 16

/-- The total number of legs on the ship -/
def total_legs : ℕ := 41

/-- Theorem stating that the number of cats is correct given the conditions -/
theorem correct_num_cats : 
  num_cats + num_sailors + 2 = total_heads ∧ 
  4 * num_cats + 2 * num_sailors + 3 = total_legs :=
by sorry

end NUMINAMATH_CALUDE_correct_num_cats_l2646_264689


namespace NUMINAMATH_CALUDE_triangle_tan_C_l2646_264635

/-- Given a triangle ABC with sides a, b, and c satisfying 3a² + 3b² - 3c² + 2ab = 0,
    prove that tan C = -2√2 -/
theorem triangle_tan_C (a b c : ℝ) (h : 3 * a^2 + 3 * b^2 - 3 * c^2 + 2 * a * b = 0) :
  let C := Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))
  Real.tan C = -2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_tan_C_l2646_264635


namespace NUMINAMATH_CALUDE_simplify_absolute_value_l2646_264670

theorem simplify_absolute_value : |(-5^2 + 6)| = 19 := by
  sorry

end NUMINAMATH_CALUDE_simplify_absolute_value_l2646_264670


namespace NUMINAMATH_CALUDE_last_digit_of_large_prime_l2646_264619

theorem last_digit_of_large_prime (n : ℕ) (h : n = 859433) :
  (2^n - 1) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_large_prime_l2646_264619


namespace NUMINAMATH_CALUDE_percentage_problem_l2646_264617

theorem percentage_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h1 : 8 = (6 / 100) * a) 
  (h2 : ∃ x, (x / 100) * b = 6) (h3 : b / a = 9 / 2) : 
  ∃ x, (x / 100) * b = 6 ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2646_264617


namespace NUMINAMATH_CALUDE_tournament_score_sum_l2646_264685

/-- A round-robin tournament with three players -/
structure Tournament :=
  (players : Fin 3 → ℕ)

/-- The scoring system for the tournament -/
def score (result : ℕ) : ℕ :=
  match result with
  | 0 => 2  -- win
  | 1 => 1  -- draw
  | _ => 0  -- loss

/-- The theorem stating that the sum of all players' scores is always 6 -/
theorem tournament_score_sum (t : Tournament) : 
  (t.players 0) + (t.players 1) + (t.players 2) = 6 :=
sorry

end NUMINAMATH_CALUDE_tournament_score_sum_l2646_264685


namespace NUMINAMATH_CALUDE_hat_color_game_l2646_264649

/-- Represents the maximum number of correct guesses in the hat color game -/
def max_correct_guesses (n k : ℕ) : ℕ :=
  n - k - 1

/-- Theorem stating the maximum number of guaranteed correct guesses in the hat color game -/
theorem hat_color_game (n k : ℕ) (h1 : k < n) :
  max_correct_guesses n k = n - k - 1 :=
by sorry

end NUMINAMATH_CALUDE_hat_color_game_l2646_264649


namespace NUMINAMATH_CALUDE_solutions_for_20_l2646_264696

/-- The number of distinct integer solutions (x,y) for |x| + |y| = n -/
def num_solutions (n : ℕ) : ℕ := 4 * n

theorem solutions_for_20 : num_solutions 20 = 80 := by
  sorry

end NUMINAMATH_CALUDE_solutions_for_20_l2646_264696


namespace NUMINAMATH_CALUDE_platform_length_l2646_264601

/-- Given a train of length 600 meters that crosses a platform in 39 seconds
    and a signal pole in 18 seconds, prove that the platform length is 700 meters. -/
theorem platform_length (train_length : ℝ) (platform_cross_time : ℝ) (pole_cross_time : ℝ) :
  train_length = 600 ∧ 
  platform_cross_time = 39 ∧ 
  pole_cross_time = 18 →
  (train_length + (train_length / pole_cross_time * platform_cross_time - train_length)) = 700 :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l2646_264601


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2646_264634

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function
  seq_def : ∀ n, a (n + 1) = a n + d
  sum_def : ∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * d) / 2

/-- If 2S_3 = 3S_2 + 6 for an arithmetic sequence, then its common difference is 2 -/
theorem arithmetic_sequence_common_difference 
  (seq : ArithmeticSequence) 
  (h : 2 * seq.S 3 = 3 * seq.S 2 + 6) : 
  seq.d = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2646_264634


namespace NUMINAMATH_CALUDE_f_one_third_bounds_l2646_264613

def f_conditions (f : ℝ → ℝ) : Prop :=
  (∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f x ∧ f x ≤ 1) ∧
  f 0 = 0 ∧
  f 1 = 1 ∧
  ∀ x y z, 0 ≤ x ∧ x < y ∧ y < z ∧ z ≤ 1 ∧ z - y = y - x →
    (1/2 : ℝ) ≤ (f z - f y) / (f y - f x) ∧ (f z - f y) / (f y - f x) ≤ 2

theorem f_one_third_bounds (f : ℝ → ℝ) (h : f_conditions f) :
  (1/7 : ℝ) ≤ f (1/3) ∧ f (1/3) ≤ 4/7 := by
  sorry

end NUMINAMATH_CALUDE_f_one_third_bounds_l2646_264613


namespace NUMINAMATH_CALUDE_y_sqrt_x_plus_one_l2646_264679

theorem y_sqrt_x_plus_one (x y k : ℝ) : 
  (y * (Real.sqrt x + 1) = k) →
  (x = 1 ∧ y = 5 → k = 10) ∧
  (y = 2 → x = 16) := by
sorry

end NUMINAMATH_CALUDE_y_sqrt_x_plus_one_l2646_264679


namespace NUMINAMATH_CALUDE_john_bought_36_rolls_l2646_264661

/-- The number of rolls John bought given the cost per dozen and the amount spent -/
def rolls_bought (cost_per_dozen : ℚ) (amount_spent : ℚ) : ℚ :=
  (amount_spent / cost_per_dozen) * 12

/-- Theorem stating that John bought 36 rolls -/
theorem john_bought_36_rolls :
  let cost_per_dozen : ℚ := 5
  let amount_spent : ℚ := 15
  rolls_bought cost_per_dozen amount_spent = 36 := by
  sorry

end NUMINAMATH_CALUDE_john_bought_36_rolls_l2646_264661


namespace NUMINAMATH_CALUDE_problem_statement_l2646_264694

theorem problem_statement (a b c : ℝ) (h : b^2 = a*c) :
  (a^2 * b^2 * c^2) / (a^3 + b^3 + c^3) * (1/a^3 + 1/b^3 + 1/c^3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2646_264694


namespace NUMINAMATH_CALUDE_penny_nickel_dime_heads_probability_l2646_264626

def coin_flip_probability : ℚ :=
  let total_outcomes : ℕ := 2^5
  let successful_outcomes : ℕ := 2^2
  (successful_outcomes : ℚ) / total_outcomes

theorem penny_nickel_dime_heads_probability :
  coin_flip_probability = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_penny_nickel_dime_heads_probability_l2646_264626


namespace NUMINAMATH_CALUDE_no_such_sequence_l2646_264688

theorem no_such_sequence : ¬∃ (a : ℕ → ℕ),
  (∀ n > 1, a n > a (n - 1)) ∧
  (∀ m n : ℕ, a (m * n) = a m + a n) :=
sorry

end NUMINAMATH_CALUDE_no_such_sequence_l2646_264688


namespace NUMINAMATH_CALUDE_max_good_sequences_theorem_l2646_264636

/-- A necklace with blue, red, and green beads. -/
structure Necklace where
  blue : ℕ
  red : ℕ
  green : ℕ

/-- A sequence of four consecutive beads in the necklace. -/
structure Sequence where
  blue : ℕ
  red : ℕ
  green : ℕ

/-- A "good" sequence contains exactly 2 blue beads, 1 red bead, and 1 green bead. -/
def is_good (s : Sequence) : Prop :=
  s.blue = 2 ∧ s.red = 1 ∧ s.green = 1

/-- The maximum number of good sequences in a necklace. -/
def max_good_sequences (n : Necklace) : ℕ := sorry

/-- Theorem: The maximum number of good sequences in a necklace with 50 blue, 100 red, and 100 green beads is 99. -/
theorem max_good_sequences_theorem (n : Necklace) (h : n.blue = 50 ∧ n.red = 100 ∧ n.green = 100) :
  max_good_sequences n = 99 := by sorry

end NUMINAMATH_CALUDE_max_good_sequences_theorem_l2646_264636


namespace NUMINAMATH_CALUDE_smartphone_price_problem_l2646_264682

theorem smartphone_price_problem (store_a_price : ℝ) (store_a_discount : ℝ) (store_b_discount : ℝ) :
  store_a_price = 125 →
  store_a_discount = 0.08 →
  store_b_discount = 0.10 →
  store_a_price * (1 - store_a_discount) = store_b_price * (1 - store_b_discount) - 2 →
  store_b_price = 130 :=
by
  sorry

#check smartphone_price_problem

end NUMINAMATH_CALUDE_smartphone_price_problem_l2646_264682


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2646_264699

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, x^2 + 2*x - 3 < 0 ↔ -3 < x ∧ x < 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2646_264699


namespace NUMINAMATH_CALUDE_circle_center_on_line_ab_range_l2646_264615

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y + 1 = 0

-- Define the line equation
def line_eq (a b x y : ℝ) : Prop :=
  a*x - b*y + 1 = 0

-- Define the center of the circle
def center (x y : ℝ) : Prop :=
  x = -1 ∧ y = 2

-- Theorem statement
theorem circle_center_on_line_ab_range :
  ∀ (a b : ℝ),
  (∃ (x y : ℝ), circle_eq x y ∧ center x y ∧ line_eq a b x y) →
  0 < a * b ∧ a * b ≤ 2 - Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_circle_center_on_line_ab_range_l2646_264615


namespace NUMINAMATH_CALUDE_frogs_eaten_by_fish_l2646_264651

/-- The number of flies eaten by each frog per day -/
def flies_per_frog : ℕ := 30

/-- The number of fish eaten by each gharial per day -/
def fish_per_gharial : ℕ := 15

/-- The number of gharials in the swamp -/
def num_gharials : ℕ := 9

/-- The total number of flies eaten per day -/
def total_flies_eaten : ℕ := 32400

/-- The number of frogs each fish needs to eat per day -/
def frogs_per_fish : ℕ := 8

theorem frogs_eaten_by_fish :
  frogs_per_fish = 
    total_flies_eaten / (flies_per_frog * (num_gharials * fish_per_gharial)) :=
by sorry

end NUMINAMATH_CALUDE_frogs_eaten_by_fish_l2646_264651


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2646_264656

theorem polynomial_simplification (x : ℝ) :
  (2 * x^5 - 3 * x^4 + x^3 + 5 * x^2 - 2 * x + 8) + 
  (x^4 - 2 * x^3 + 3 * x^2 + 4 * x - 16) = 
  2 * x^5 - 2 * x^4 - x^3 + 8 * x^2 + 2 * x - 8 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2646_264656


namespace NUMINAMATH_CALUDE_cucumbers_for_apples_l2646_264674

-- Define the cost relationships
def apple_banana_ratio : ℚ := 12 / 6
def banana_cucumber_ratio : ℚ := 3 / 4

-- Define the number of apples we're interested in
def apples_of_interest : ℕ := 24

-- Theorem to prove
theorem cucumbers_for_apples :
  (apples_of_interest : ℚ) / apple_banana_ratio * banana_cucumber_ratio = 16 := by
  sorry

end NUMINAMATH_CALUDE_cucumbers_for_apples_l2646_264674


namespace NUMINAMATH_CALUDE_A_intersect_B_eq_zero_one_two_l2646_264608

def A : Set ℕ := {x : ℕ | 5 + 4 * x - x^2 > 0}

def B : Set ℕ := {x : ℕ | x < 3}

theorem A_intersect_B_eq_zero_one_two : A ∩ B = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_eq_zero_one_two_l2646_264608


namespace NUMINAMATH_CALUDE_sample_size_calculation_l2646_264648

/-- Represents the staff composition in a company -/
structure StaffComposition where
  sales : ℕ
  management : ℕ
  logistics : ℕ

/-- Represents a stratified sample from the company -/
structure StratifiedSample where
  total_size : ℕ
  sales_size : ℕ

/-- The theorem stating the relationship between the company's staff composition,
    the number of sales staff in the sample, and the total sample size -/
theorem sample_size_calculation 
  (company : StaffComposition)
  (sample : StratifiedSample)
  (h1 : company.sales = 15)
  (h2 : company.management = 3)
  (h3 : company.logistics = 2)
  (h4 : sample.sales_size = 30) :
  sample.total_size = 40 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_calculation_l2646_264648


namespace NUMINAMATH_CALUDE_expression_equality_l2646_264625

theorem expression_equality (w : ℝ) : 
  (Real.sqrt 1.21 / Real.sqrt 0.81 + Real.sqrt 1.00 / Real.sqrt w = 2.650793650793651) → 
  w = 0.49 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2646_264625


namespace NUMINAMATH_CALUDE_power_of_5000_times_2_l2646_264603

theorem power_of_5000_times_2 : ∃ n : ℕ, 2 * (5000 ^ 150) = 10 ^ n ∧ n = 600 := by
  sorry

end NUMINAMATH_CALUDE_power_of_5000_times_2_l2646_264603


namespace NUMINAMATH_CALUDE_point_on_x_axis_l2646_264602

/-- If a point M(a+3, a+1) lies on the x-axis, then its coordinates are (2,0) -/
theorem point_on_x_axis (a : ℝ) :
  (a + 1 = 0) →  -- Condition for M to be on x-axis
  ((a + 3, a + 1) : ℝ × ℝ) = (2, 0) := by
sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l2646_264602


namespace NUMINAMATH_CALUDE_last_two_digits_sum_factorials_l2646_264631

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem last_two_digits_sum_factorials :
  let sum := (List.range 50).map (λ i => factorial (i + 1)) |> List.foldl (· + ·) 0
  last_two_digits sum = 13 := by sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_factorials_l2646_264631


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l2646_264698

theorem square_area_from_diagonal (d : ℝ) (h : d = 12 * Real.sqrt 2) :
  let s := d / Real.sqrt 2
  s ^ 2 = 144 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l2646_264698


namespace NUMINAMATH_CALUDE_range_of_x2_plus_y2_l2646_264664

theorem range_of_x2_plus_y2 (x y : ℝ) 
  (h1 : x - 2*y + 1 ≥ 0) 
  (h2 : y ≥ x) 
  (h3 : x ≥ 0) : 
  0 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 2 := by
  sorry

#check range_of_x2_plus_y2

end NUMINAMATH_CALUDE_range_of_x2_plus_y2_l2646_264664


namespace NUMINAMATH_CALUDE_mean_correction_l2646_264623

def correct_mean (n : ℕ) (original_mean : ℚ) (wrong_value : ℚ) (correct_value : ℚ) : ℚ :=
  (n : ℚ) * original_mean - wrong_value + correct_value

theorem mean_correction (n : ℕ) (original_mean : ℚ) (wrong_value : ℚ) (correct_value : ℚ) 
  (h1 : n = 50)
  (h2 : original_mean = 36)
  (h3 : wrong_value = 23)
  (h4 : correct_value = 45) :
  (correct_mean n original_mean wrong_value correct_value) / n = 36.44 := by
  sorry

end NUMINAMATH_CALUDE_mean_correction_l2646_264623


namespace NUMINAMATH_CALUDE_farmer_tomatoes_l2646_264678

/-- A farmer picks tomatoes from his garden. -/
theorem farmer_tomatoes (initial : ℕ) (remaining : ℕ) (picked : ℕ)
    (h1 : initial = 97)
    (h2 : remaining = 14)
    (h3 : picked = initial - remaining) :
  picked = 83 := by
  sorry

end NUMINAMATH_CALUDE_farmer_tomatoes_l2646_264678


namespace NUMINAMATH_CALUDE_five_numbers_sequence_exists_l2646_264695

theorem five_numbers_sequence_exists : 
  ∃ (a₁ a₂ a₃ a₄ a₅ : ℝ), 
    (a₁ + a₂ < 0) ∧ 
    (a₂ + a₃ < 0) ∧ 
    (a₃ + a₄ < 0) ∧ 
    (a₄ + a₅ < 0) ∧ 
    (a₁ + a₂ + a₃ + a₄ + a₅ > 0) := by
  sorry

end NUMINAMATH_CALUDE_five_numbers_sequence_exists_l2646_264695


namespace NUMINAMATH_CALUDE_min_sum_squares_l2646_264624

theorem min_sum_squares (x y : ℝ) (h : (x - 1)^2 + y^2 = 16) :
  ∃ (m : ℝ), (∀ (a b : ℝ), (a - 1)^2 + b^2 = 16 → x^2 + y^2 ≤ a^2 + b^2) ∧ m = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l2646_264624


namespace NUMINAMATH_CALUDE_foreign_stamps_count_l2646_264643

/-- Represents a stamp collection with various properties -/
structure StampCollection where
  total : ℕ
  old : ℕ
  foreignAndOld : ℕ
  neitherForeignNorOld : ℕ

/-- Calculates the number of foreign stamps in the collection -/
def foreignStamps (sc : StampCollection) : ℕ :=
  sc.total - sc.neitherForeignNorOld - (sc.old - sc.foreignAndOld)

/-- Theorem stating the number of foreign stamps in the given collection -/
theorem foreign_stamps_count (sc : StampCollection) 
    (h1 : sc.total = 200)
    (h2 : sc.old = 70)
    (h3 : sc.foreignAndOld = 20)
    (h4 : sc.neitherForeignNorOld = 60) :
    foreignStamps sc = 90 := by
  sorry

#eval foreignStamps { total := 200, old := 70, foreignAndOld := 20, neitherForeignNorOld := 60 }

end NUMINAMATH_CALUDE_foreign_stamps_count_l2646_264643


namespace NUMINAMATH_CALUDE_power_sum_equality_l2646_264628

theorem power_sum_equality : 2^567 + 8^5 / 8^3 = 2^567 + 64 := by sorry

end NUMINAMATH_CALUDE_power_sum_equality_l2646_264628


namespace NUMINAMATH_CALUDE_unique_five_digit_number_l2646_264604

/-- A five-digit number is a natural number between 10000 and 99999 inclusive. -/
def FiveDigitNumber (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

/-- The property that defines our target number. -/
def SatisfiesCondition (x : ℕ) : Prop :=
  FiveDigitNumber x ∧ 7 * 10^5 + x = 5 * (10 * x + 7)

theorem unique_five_digit_number : 
  ∃! x : ℕ, SatisfiesCondition x ∧ x = 14285 :=
sorry

end NUMINAMATH_CALUDE_unique_five_digit_number_l2646_264604


namespace NUMINAMATH_CALUDE_cylinder_volume_relation_l2646_264642

/-- Proves that the volume of cylinder C is (1/9) π h³ given the specified conditions --/
theorem cylinder_volume_relation (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  h = 3 * r →  -- Height of C is three times radius of D
  r = h →      -- Radius of D is equal to height of C
  π * r^2 * h = 3 * (π * h^2 * r) →  -- Volume of C is three times volume of D
  π * r^2 * h = (1/9) * π * h^3 :=
by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_relation_l2646_264642


namespace NUMINAMATH_CALUDE_no_real_solution_l2646_264675

theorem no_real_solution :
  ¬∃ (r s : ℝ), (r - 50) / 3 = (s - 2 * r) / 4 ∧ r^2 + 3 * s = 50 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_l2646_264675


namespace NUMINAMATH_CALUDE_line_touches_ellipse_l2646_264622

theorem line_touches_ellipse (a b : ℝ) (m : ℝ) (h1 : a = 3) (h2 : b = 1) :
  (∃! p : ℝ × ℝ, p.1^2 / a^2 + p.2^2 / b^2 = 1 ∧ p.2 = m * p.1 + 2) ↔ m^2 = 1/3 :=
sorry

end NUMINAMATH_CALUDE_line_touches_ellipse_l2646_264622


namespace NUMINAMATH_CALUDE_skt_lineup_count_l2646_264612

/-- The total number of StarCraft programmers -/
def total_programmers : ℕ := 111

/-- The number of programmers in SKT's initial team -/
def initial_team_size : ℕ := 11

/-- The number of programmers needed for the lineup -/
def lineup_size : ℕ := 5

/-- The number of different lineups for SKT's second season opening match -/
def number_of_lineups : ℕ := 
  initial_team_size * (total_programmers - initial_team_size + 1) * 
  (Nat.choose initial_team_size lineup_size) * (Nat.factorial lineup_size)

theorem skt_lineup_count : number_of_lineups = 61593840 := by
  sorry

end NUMINAMATH_CALUDE_skt_lineup_count_l2646_264612


namespace NUMINAMATH_CALUDE_polygon_chain_sides_l2646_264683

/-- Represents a regular polygon with a given number of sides. -/
structure RegularPolygon where
  sides : ℕ
  sides_positive : sides > 0

/-- Represents a chain of connected regular polygons. -/
structure PolygonChain where
  polygons : List RegularPolygon
  connected : polygons.length > 1

/-- Calculates the number of exposed sides in a chain of connected polygons. -/
def exposedSides (chain : PolygonChain) : ℕ :=
  let n := chain.polygons.length
  let total_sides := (chain.polygons.map RegularPolygon.sides).sum
  let shared_sides := 2 * (n - 1) - 2
  total_sides - shared_sides

/-- The theorem to be proved. -/
theorem polygon_chain_sides :
  ∀ (chain : PolygonChain),
    chain.polygons.map RegularPolygon.sides = [3, 4, 5, 6, 7, 8, 9] →
    exposedSides chain = 30 := by
  sorry

end NUMINAMATH_CALUDE_polygon_chain_sides_l2646_264683


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_range_l2646_264627

/-- The range of m for which the line y = kx + 1 and the ellipse x²/5 + y²/m = 1 always intersect -/
theorem line_ellipse_intersection_range (k : ℝ) (m : ℝ) : 
  (∀ x y : ℝ, y = k * x + 1 ∧ x^2 / 5 + y^2 / m = 1 → m ≥ 1 ∧ m ≠ 5) ∧
  (m ≥ 1 ∧ m ≠ 5 → ∃ x y : ℝ, y = k * x + 1 ∧ x^2 / 5 + y^2 / m = 1) :=
sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_range_l2646_264627


namespace NUMINAMATH_CALUDE_floor_width_is_eight_meters_l2646_264641

/-- Proves that a rectangular floor with given dimensions has a width of 8 meters -/
theorem floor_width_is_eight_meters
  (floor_length : ℝ)
  (rug_area : ℝ)
  (strip_width : ℝ)
  (h1 : floor_length = 10)
  (h2 : rug_area = 24)
  (h3 : strip_width = 2)
  : ∃ (floor_width : ℝ),
    floor_width = 8 ∧
    rug_area = (floor_length - 2 * strip_width) * (floor_width - 2 * strip_width) :=
by
  sorry


end NUMINAMATH_CALUDE_floor_width_is_eight_meters_l2646_264641


namespace NUMINAMATH_CALUDE_complex_multiplication_l2646_264650

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (1 - 2*i) = 2 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l2646_264650


namespace NUMINAMATH_CALUDE_tanya_erasers_l2646_264621

/-- Given the number of erasers for Hanna, Rachel, and Tanya, prove that Tanya has 20 erasers -/
theorem tanya_erasers (h r t : ℕ) (tr : ℕ) : 
  h = 2 * r →  -- Hanna has twice as many erasers as Rachel
  r = tr / 2 - 3 →  -- Rachel has three less than one-half as many erasers as Tanya has red erasers
  tr = t / 2 →  -- Half of Tanya's erasers are red
  h = 4 →  -- Hanna has 4 erasers
  t = 20 := by sorry

end NUMINAMATH_CALUDE_tanya_erasers_l2646_264621


namespace NUMINAMATH_CALUDE_percentage_not_covering_politics_l2646_264605

/-- Represents the percentage of reporters covering local politics in country X -/
def local_politics_coverage : ℝ := 10

/-- Represents the percentage of political reporters not covering local politics in country X -/
def non_local_political_coverage : ℝ := 30

/-- Represents the total number of reporters (assumed for calculation purposes) -/
def total_reporters : ℝ := 100

/-- Theorem stating that 86% of reporters do not cover politics -/
theorem percentage_not_covering_politics :
  (total_reporters - (local_politics_coverage / (100 - non_local_political_coverage) * 100)) / total_reporters * 100 = 86 := by
  sorry

end NUMINAMATH_CALUDE_percentage_not_covering_politics_l2646_264605


namespace NUMINAMATH_CALUDE_jennys_money_l2646_264632

theorem jennys_money (original : ℚ) : 
  (original - (3/7 * original + 2/5 * original) = 24) → 
  (1/2 * original = 70) := by
sorry

end NUMINAMATH_CALUDE_jennys_money_l2646_264632


namespace NUMINAMATH_CALUDE_coefficient_a3b3_value_l2646_264640

/-- The coefficient of a^3b^3 in (a+b)^6(c+1/c)^8 -/
def coefficient_a3b3 (a b c : ℝ) : ℕ :=
  (Nat.choose 6 3) * (Nat.choose 8 4)

theorem coefficient_a3b3_value :
  ∀ a b c : ℝ, coefficient_a3b3 a b c = 1400 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_a3b3_value_l2646_264640


namespace NUMINAMATH_CALUDE_equation_one_solution_l2646_264629

theorem equation_one_solution (k : ℝ) : 
  (∃! x : ℝ, (3*x + 6)*(x - 4) = -40 + k*x) ↔ 
  (k = -6 + 8*Real.sqrt 3 ∨ k = -6 - 8*Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_equation_one_solution_l2646_264629


namespace NUMINAMATH_CALUDE_house_pets_problem_l2646_264633

theorem house_pets_problem (total : Nat) (dogs cats turtles : Nat)
  (h_total : total = 2017)
  (h_dogs : dogs = 1820)
  (h_cats : cats = 1651)
  (h_turtles : turtles = 1182)
  (h_dogs_le : dogs ≤ total)
  (h_cats_le : cats ≤ total)
  (h_turtles_le : turtles ≤ total) :
  ∃ (max min : Nat),
    (max ≤ turtles) ∧
    (min ≥ dogs + cats + turtles - 2 * total) ∧
    (max - min = 563) := by
  sorry

end NUMINAMATH_CALUDE_house_pets_problem_l2646_264633


namespace NUMINAMATH_CALUDE_path_area_calculation_l2646_264630

/-- Calculates the area of a path surrounding a rectangular field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

theorem path_area_calculation (field_length field_width path_width : ℝ) 
  (h1 : field_length = 75)
  (h2 : field_width = 55)
  (h3 : path_width = 2.5) :
  path_area field_length field_width path_width = 675 := by
  sorry

#eval path_area 75 55 2.5

end NUMINAMATH_CALUDE_path_area_calculation_l2646_264630
