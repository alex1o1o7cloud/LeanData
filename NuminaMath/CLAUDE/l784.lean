import Mathlib

namespace NUMINAMATH_CALUDE_inequality_and_quadratic_solution_l784_78414

theorem inequality_and_quadratic_solution :
  ∃ (m : ℤ), 1 < m ∧ m < 4 ∧
  ∀ (x : ℝ), 1 < x ∧ x < 4 →
  (x^2 - 2*x - m = 0 ↔ (x = 1 + Real.sqrt 3 ∨ x = 1 - Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_quadratic_solution_l784_78414


namespace NUMINAMATH_CALUDE_square_area_ratio_l784_78484

theorem square_area_ratio (d : ℝ) (h : d > 0) :
  let small_square_side := d / Real.sqrt 2
  let big_square_side := d
  let small_square_area := small_square_side ^ 2
  let big_square_area := big_square_side ^ 2
  big_square_area / small_square_area = 2 := by
sorry

end NUMINAMATH_CALUDE_square_area_ratio_l784_78484


namespace NUMINAMATH_CALUDE_arithmetic_expressions_l784_78466

-- Define the number of arithmetic expressions
def f (n : ℕ) : ℚ :=
  (7/10) * 12^n - (1/5) * (-3)^n

-- State the theorem
theorem arithmetic_expressions (n : ℕ) :
  f n = (7/10) * 12^n - (1/5) * (-3)^n ∧
  f 1 = 9 ∧
  f 2 = 99 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_expressions_l784_78466


namespace NUMINAMATH_CALUDE_sea_glass_collection_l784_78434

/-- Sea glass collection problem -/
theorem sea_glass_collection (blanche_green blanche_red rose_red rose_blue : ℕ) 
  (h1 : blanche_green = 12)
  (h2 : blanche_red = 3)
  (h3 : rose_red = 9)
  (h4 : rose_blue = 11)
  : 2 * (blanche_red + rose_red) + 3 * rose_blue = 57 := by
  sorry

end NUMINAMATH_CALUDE_sea_glass_collection_l784_78434


namespace NUMINAMATH_CALUDE_f_lower_bound_l784_78475

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * Real.log x

-- State the theorem
theorem f_lower_bound (a : ℝ) (h_a : a > 0) :
  ∀ x > 0, f a x ≥ a * (2 - Real.log a) := by
  sorry

end NUMINAMATH_CALUDE_f_lower_bound_l784_78475


namespace NUMINAMATH_CALUDE_ken_payment_l784_78424

/-- The price of steak per pound -/
def price_per_pound : ℕ := 7

/-- The number of pounds of steak Ken bought -/
def pounds_bought : ℕ := 2

/-- The amount of change Ken received -/
def change_received : ℕ := 6

/-- The amount Ken paid -/
def amount_paid : ℕ := 20

/-- Proof that Ken paid $20 given the conditions -/
theorem ken_payment :
  amount_paid = price_per_pound * pounds_bought + change_received :=
by sorry

end NUMINAMATH_CALUDE_ken_payment_l784_78424


namespace NUMINAMATH_CALUDE_triangle_one_two_two_l784_78482

/-- Triangle inequality theorem for three sides --/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Check if three lengths can form a triangle --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

theorem triangle_one_two_two :
  can_form_triangle 1 2 2 :=
sorry

end NUMINAMATH_CALUDE_triangle_one_two_two_l784_78482


namespace NUMINAMATH_CALUDE_t_range_l784_78453

-- Define the propositions p and q as functions of t
def p (t : ℝ) : Prop := ∀ x, x^2 + 2*x + 2*t - 4 ≠ 0

def q (t : ℝ) : Prop := 2 < t ∧ t < 3

-- Define the main theorem
theorem t_range (t : ℝ) : 
  (p t ∨ q t) ∧ ¬(p t ∧ q t) → (2 < t ∧ t ≤ 5/2) ∨ t ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_t_range_l784_78453


namespace NUMINAMATH_CALUDE_special_ellipse_equation_l784_78447

/-- An ellipse with specific properties -/
structure SpecialEllipse where
  /-- The foci of the ellipse are on the x-axis -/
  foci_on_x_axis : Bool
  /-- The center of the ellipse is at the origin -/
  center_at_origin : Bool
  /-- The four vertices of a square with side length 2 are on the minor axis and coincide with the foci -/
  square_vertices_on_minor_axis : Bool
  /-- The side length of the square is 2 -/
  square_side_length : ℝ

/-- The standard equation of an ellipse -/
def standard_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Theorem stating the standard equation of the special ellipse -/
theorem special_ellipse_equation (E : SpecialEllipse) (x y : ℝ) :
  E.foci_on_x_axis ∧
  E.center_at_origin ∧
  E.square_vertices_on_minor_axis ∧
  E.square_side_length = 2 →
  standard_equation 4 2 x y :=
sorry

end NUMINAMATH_CALUDE_special_ellipse_equation_l784_78447


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l784_78463

theorem solve_exponential_equation (x y z : ℕ) :
  (3 : ℝ)^x * (4 : ℝ)^y / (2 : ℝ)^z = 59049 ∧ x - y + 2*z = 10 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l784_78463


namespace NUMINAMATH_CALUDE_square_difference_l784_78488

theorem square_difference (a b : ℝ) :
  ∃ A : ℝ, (5*a + 3*b)^2 = (5*a - 3*b)^2 + A ∧ A = 60*a*b := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l784_78488


namespace NUMINAMATH_CALUDE_passing_percentage_is_40_l784_78409

/-- The maximum marks possible in the exam -/
def max_marks : ℕ := 550

/-- The marks obtained by the student -/
def obtained_marks : ℕ := 200

/-- The number of marks by which the student failed -/
def fail_margin : ℕ := 20

/-- The passing percentage for the exam -/
def passing_percentage : ℚ :=
  (obtained_marks + fail_margin : ℚ) / max_marks * 100

theorem passing_percentage_is_40 :
  passing_percentage = 40 := by sorry

end NUMINAMATH_CALUDE_passing_percentage_is_40_l784_78409


namespace NUMINAMATH_CALUDE_infinitely_many_a_for_positive_integer_l784_78429

theorem infinitely_many_a_for_positive_integer (n : ℕ) :
  ∃ (f : ℕ → ℤ), Function.Injective f ∧
  ∀ (k : ℕ), (n^6 + 3 * (f k) : ℤ) > 0 :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_a_for_positive_integer_l784_78429


namespace NUMINAMATH_CALUDE_train_speed_calculation_l784_78425

/-- Proves that under given conditions, the train's speed is 45 km/hr -/
theorem train_speed_calculation (jogger_speed : ℝ) (initial_distance : ℝ) 
  (train_length : ℝ) (passing_time : ℝ) :
  jogger_speed = 9 →
  initial_distance = 200 →
  train_length = 210 →
  passing_time = 41 →
  ∃ (train_speed : ℝ), train_speed = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l784_78425


namespace NUMINAMATH_CALUDE_swimming_class_problem_l784_78420

theorem swimming_class_problem (N : ℕ) (non_swimmers : ℕ) (signed_up : ℕ) (not_signed_up : ℕ) :
  N / 4 = non_swimmers →
  non_swimmers / 2 = signed_up →
  not_signed_up = 4 →
  signed_up + not_signed_up = non_swimmers →
  N = 32 ∧ N - non_swimmers = 24 := by
  sorry

end NUMINAMATH_CALUDE_swimming_class_problem_l784_78420


namespace NUMINAMATH_CALUDE_apple_distribution_l784_78431

theorem apple_distribution (total_apples : Nat) (total_bags : Nat) (x : Nat) : 
  total_apples = 109 →
  total_bags = 20 →
  (∃ (a b : Nat), a + b = total_bags ∧ a * x + b * 3 = total_apples) →
  (x = 10 ∨ x = 52) := by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l784_78431


namespace NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l784_78473

/-- The number of ways to distribute n indistinguishable balls into k indistinguishable boxes -/
def distribute_balls (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 7 ways to distribute 6 indistinguishable balls into 3 indistinguishable boxes -/
theorem distribute_six_balls_three_boxes : distribute_balls 6 3 = 7 := by sorry

end NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l784_78473


namespace NUMINAMATH_CALUDE_hcl_formed_l784_78451

-- Define the chemical reaction
structure Reaction where
  ch4 : ℕ
  cl2 : ℕ
  ccl4 : ℕ
  hcl : ℕ

-- Define the balanced equation
def balanced_equation (r : Reaction) : Prop :=
  r.ch4 = 1 ∧ r.cl2 = 4 ∧ r.ccl4 = 1 ∧ r.hcl = 4

-- Define the given amounts of reactants
def given_reactants (r : Reaction) : Prop :=
  r.ch4 = 1 ∧ r.cl2 = 4

-- Theorem: Given the reactants and balanced equation, prove that 4 moles of HCl are formed
theorem hcl_formed (r : Reaction) 
  (h1 : balanced_equation r) 
  (h2 : given_reactants r) : 
  r.hcl = 4 := by
  sorry


end NUMINAMATH_CALUDE_hcl_formed_l784_78451


namespace NUMINAMATH_CALUDE_smallest_positive_angle_l784_78493

/-- Given a point P on the unit circle with coordinates (sin(2π/3), cos(2π/3)) 
    on the terminal side of angle α, prove that the smallest positive value of α is 11π/6 -/
theorem smallest_positive_angle (P : ℝ × ℝ) (α : ℝ) : 
  P.1 = Real.sin (2 * Real.pi / 3) →
  P.2 = Real.cos (2 * Real.pi / 3) →
  P ∈ {Q : ℝ × ℝ | ∃ t : ℝ, Q.1 = Real.cos t ∧ Q.2 = Real.sin t ∧ t ≥ 0 ∧ t < 2 * Real.pi} →
  (∃ k : ℤ, α = 11 * Real.pi / 6 + 2 * Real.pi * k) ∧ 
  (∀ β : ℝ, (∃ k : ℤ, β = α + 2 * Real.pi * k) → β ≥ 11 * Real.pi / 6) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_l784_78493


namespace NUMINAMATH_CALUDE_jess_walks_to_gallery_l784_78479

/-- The number of blocks Jess walks to work -/
def total_blocks : ℕ := 25

/-- The number of blocks Jess walks to the store -/
def blocks_to_store : ℕ := 11

/-- The number of blocks Jess walks from the gallery to work -/
def blocks_gallery_to_work : ℕ := 8

/-- The number of blocks Jess walks to the gallery -/
def blocks_to_gallery : ℕ := total_blocks - blocks_to_store - blocks_gallery_to_work

theorem jess_walks_to_gallery : blocks_to_gallery = 6 := by
  sorry

end NUMINAMATH_CALUDE_jess_walks_to_gallery_l784_78479


namespace NUMINAMATH_CALUDE_wolf_catches_hare_in_problem_l784_78436

/-- Represents the chase scenario between a wolf and a hare -/
structure ChaseScenario where
  initial_distance : ℝ
  hiding_spot_distance : ℝ
  wolf_speed : ℝ
  hare_speed : ℝ

/-- Determines if the wolf catches the hare in the given chase scenario -/
def wolf_catches_hare (scenario : ChaseScenario) : Prop :=
  let relative_speed := scenario.wolf_speed - scenario.hare_speed
  let chase_distance := scenario.hiding_spot_distance - scenario.initial_distance
  let chase_time := chase_distance / relative_speed
  scenario.hare_speed * chase_time ≤ scenario.hiding_spot_distance

/-- The specific chase scenario from the problem -/
def problem_scenario : ChaseScenario :=
  { initial_distance := 30
    hiding_spot_distance := 333
    wolf_speed := 600
    hare_speed := 550 }

/-- Theorem stating that the wolf catches the hare in the problem scenario -/
theorem wolf_catches_hare_in_problem : wolf_catches_hare problem_scenario := by
  sorry

end NUMINAMATH_CALUDE_wolf_catches_hare_in_problem_l784_78436


namespace NUMINAMATH_CALUDE_six_possible_values_for_A_l784_78408

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the sum operation in the problem -/
def SumOperation (A B X Y : Digit) : Prop :=
  (A.val * 1000000 + B.val * 1000 + A.val) + 
  (B.val * 1000000 + A.val * 1000 + B.val) = 
  (X.val * 10000000 + X.val * 1000000 + X.val * 10000 + Y.val * 1000 + X.val * 100 + X.val)

/-- The main theorem stating that there are exactly 6 possible values for A -/
theorem six_possible_values_for_A :
  ∃! (s : Finset Digit), 
    (∀ A ∈ s, ∃ (B X Y : Digit), A ≠ B ∧ A ≠ X ∧ A ≠ Y ∧ B ≠ X ∧ B ≠ Y ∧ X ≠ Y ∧ SumOperation A B X Y) ∧
    s.card = 6 := by
  sorry

end NUMINAMATH_CALUDE_six_possible_values_for_A_l784_78408


namespace NUMINAMATH_CALUDE_class_3_1_fairy_tales_l784_78417

theorem class_3_1_fairy_tales (andersen : ℕ) (grimm : ℕ) (both : ℕ) (total : ℕ) :
  andersen = 20 →
  grimm = 27 →
  both = 8 →
  total = 55 →
  andersen + grimm - both ≠ total :=
by
  sorry

end NUMINAMATH_CALUDE_class_3_1_fairy_tales_l784_78417


namespace NUMINAMATH_CALUDE_new_average_weight_l784_78433

theorem new_average_weight 
  (A B C D E : ℝ)
  (h1 : (A + B + C) / 3 = 70)
  (h2 : (A + B + C + D) / 4 = 70)
  (h3 : E = D + 3)
  (h4 : A = 81) :
  (B + C + D + E) / 4 = 68 := by
sorry

end NUMINAMATH_CALUDE_new_average_weight_l784_78433


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l784_78446

theorem simplify_and_evaluate : 
  ∀ x : ℝ, (3*x^2 + 8*x - 6) - (2*x^2 + 4*x - 15) = x^2 + 4*x + 9 ∧ 
  (let x : ℝ := 3; (3*x^2 + 8*x - 6) - (2*x^2 + 4*x - 15) = 30) :=
by
  sorry

#check simplify_and_evaluate

end NUMINAMATH_CALUDE_simplify_and_evaluate_l784_78446


namespace NUMINAMATH_CALUDE_isosceles_triangle_rectangle_equal_area_l784_78438

/-- Given an isosceles triangle and a rectangle with equal areas,
    where the rectangle's length is twice its width and
    the triangle's base equals the rectangle's width,
    prove that the triangle's height is four times the rectangle's width. -/
theorem isosceles_triangle_rectangle_equal_area
  (w h : ℝ) -- w: width of rectangle, h: height of triangle
  (hw : w > 0) -- assume width is positive
  (triangle_area : ℝ → ℝ → ℝ) -- area function for triangle
  (rectangle_area : ℝ → ℝ → ℝ) -- area function for rectangle
  (h_triangle_area : triangle_area w h = 1/2 * w * h) -- definition of triangle area
  (h_rectangle_area : rectangle_area w (2*w) = 2 * w^2) -- definition of rectangle area
  (h_equal_area : triangle_area w h = rectangle_area w (2*w)) -- areas are equal
  : h = 4 * w :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_rectangle_equal_area_l784_78438


namespace NUMINAMATH_CALUDE_remainder_98_102_div_11_l784_78492

theorem remainder_98_102_div_11 : (98 * 102) % 11 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_98_102_div_11_l784_78492


namespace NUMINAMATH_CALUDE_today_is_wednesday_l784_78415

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def prevDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Saturday
  | DayOfWeek.Monday => DayOfWeek.Sunday
  | DayOfWeek.Tuesday => DayOfWeek.Monday
  | DayOfWeek.Wednesday => DayOfWeek.Tuesday
  | DayOfWeek.Thursday => DayOfWeek.Wednesday
  | DayOfWeek.Friday => DayOfWeek.Thursday
  | DayOfWeek.Saturday => DayOfWeek.Friday

def dayAfterTomorrow (d : DayOfWeek) : DayOfWeek := nextDay (nextDay d)

def distanceToSunday (d : DayOfWeek) : Nat :=
  match d with
  | DayOfWeek.Sunday => 0
  | DayOfWeek.Monday => 6
  | DayOfWeek.Tuesday => 5
  | DayOfWeek.Wednesday => 4
  | DayOfWeek.Thursday => 3
  | DayOfWeek.Friday => 2
  | DayOfWeek.Saturday => 1

theorem today_is_wednesday :
  ∃ (today : DayOfWeek),
    (dayAfterTomorrow today = prevDay today) ∧
    (distanceToSunday today = distanceToSunday (prevDay (nextDay today))) ∧
    (today = DayOfWeek.Wednesday) := by
  sorry


end NUMINAMATH_CALUDE_today_is_wednesday_l784_78415


namespace NUMINAMATH_CALUDE_greatest_good_set_size_l784_78430

/-- A set S of positive integers is "good" if there exists a coloring of positive integers
    with k colors such that no element from S can be written as the sum of two distinct
    positive integers having the same color. -/
def IsGood (S : Set ℕ) (k : ℕ) : Prop :=
  ∃ (c : ℕ → Fin k), ∀ s ∈ S, ∀ x y : ℕ, x < y → x + y = s → c x ≠ c y

/-- The set S defined as {a+1, a+2, ..., a+t} for some positive integer a -/
def S (a t : ℕ) : Set ℕ := {n : ℕ | a + 1 ≤ n ∧ n ≤ a + t}

theorem greatest_good_set_size (k : ℕ) (h : k > 1) :
  (∃ t : ℕ, ∀ a : ℕ, a > 0 → IsGood (S a t) k ∧
    ∀ t' : ℕ, t' > t → ∃ a : ℕ, a > 0 ∧ ¬IsGood (S a t') k) ∧
  (∀ t : ℕ, (∀ a : ℕ, a > 0 → IsGood (S a t) k) → t ≤ 2 * k - 2) :=
sorry

end NUMINAMATH_CALUDE_greatest_good_set_size_l784_78430


namespace NUMINAMATH_CALUDE_sixth_term_is_twelve_l784_78465

/-- An arithmetic sequence with its first term and sum of first three terms specified -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  a₁ : a 1 = 2
  S₃ : (a 1) + (a 2) + (a 3) = 12
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

/-- The 6th term of the arithmetic sequence is 12 -/
theorem sixth_term_is_twelve (seq : ArithmeticSequence) : seq.a 6 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_is_twelve_l784_78465


namespace NUMINAMATH_CALUDE_count_six_digit_numbers_with_seven_l784_78404

def digits : Finset ℕ := Finset.range 10

def multiset_count (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) k

def six_digit_numbers_with_seven : ℕ :=
  (multiset_count 9 5) +
  (multiset_count 9 4) +
  (multiset_count 9 3) +
  (multiset_count 9 2) +
  (multiset_count 9 1) +
  (multiset_count 9 0)

theorem count_six_digit_numbers_with_seven :
  six_digit_numbers_with_seven = 2002 := by sorry

end NUMINAMATH_CALUDE_count_six_digit_numbers_with_seven_l784_78404


namespace NUMINAMATH_CALUDE_geese_percentage_among_non_swans_l784_78422

theorem geese_percentage_among_non_swans 
  (total : ℝ) 
  (geese_percent : ℝ) 
  (swan_percent : ℝ) 
  (heron_percent : ℝ) 
  (duck_percent : ℝ) 
  (h1 : geese_percent = 30)
  (h2 : swan_percent = 25)
  (h3 : heron_percent = 10)
  (h4 : duck_percent = 35)
  (h5 : geese_percent + swan_percent + heron_percent + duck_percent = 100) :
  (geese_percent / (100 - swan_percent)) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_geese_percentage_among_non_swans_l784_78422


namespace NUMINAMATH_CALUDE_multiply_b_equals_five_l784_78406

theorem multiply_b_equals_five (a b x : ℝ) 
  (h1 : 4 * a = x * b) 
  (h2 : a * b ≠ 0) 
  (h3 : (a / 5) / (b / 4) = 1) : 
  x = 5 := by sorry

end NUMINAMATH_CALUDE_multiply_b_equals_five_l784_78406


namespace NUMINAMATH_CALUDE_complex_second_quadrant_l784_78455

theorem complex_second_quadrant (a : ℝ) : 
  let z : ℂ := (a + 3*Complex.I)/Complex.I + a
  (z.re < 0 ∧ z.im > 0) → a = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_second_quadrant_l784_78455


namespace NUMINAMATH_CALUDE_min_distance_to_line_l784_78467

theorem min_distance_to_line (x y : ℝ) (h : 2 * x + y + 5 = 0) :
  ∃ (min_val : ℝ), min_val = Real.sqrt 5 ∧
  ∀ (a b : ℝ), 2 * a + b + 5 = 0 → Real.sqrt (a^2 + b^2) ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l784_78467


namespace NUMINAMATH_CALUDE_unique_geometric_sequence_l784_78435

/-- A geometric sequence with the given properties has a unique first term of 1/3 -/
theorem unique_geometric_sequence (a : ℝ) (a_n : ℕ → ℝ) : 
  a > 0 ∧ 
  (∀ n, a_n (n + 1) = a_n n * (a_n 2 / a_n 1)) ∧ 
  a_n 1 = a ∧
  (∃ q : ℝ, (a_n 1 + 1) * q = a_n 2 + 2 ∧ (a_n 2 + 2) * q = a_n 3 + 3) ∧
  (∃! q : ℝ, q ≠ 0 ∧ a * q^2 - 4 * a * q + 3 * a - 1 = 0) →
  a = 1/3 := by
sorry

end NUMINAMATH_CALUDE_unique_geometric_sequence_l784_78435


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l784_78477

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ b : ℝ, (Complex.I : ℂ) * b = (a + Complex.I) / (1 + 2 * Complex.I)) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l784_78477


namespace NUMINAMATH_CALUDE_football_cards_per_box_l784_78481

theorem football_cards_per_box (basketball_boxes : ℕ) (basketball_cards_per_box : ℕ) (total_cards : ℕ) :
  basketball_boxes = 9 →
  basketball_cards_per_box = 15 →
  total_cards = 255 →
  let football_boxes := basketball_boxes - 3
  let basketball_cards := basketball_boxes * basketball_cards_per_box
  let football_cards := total_cards - basketball_cards
  football_cards / football_boxes = 20 := by
sorry

end NUMINAMATH_CALUDE_football_cards_per_box_l784_78481


namespace NUMINAMATH_CALUDE_double_average_l784_78470

theorem double_average (n : ℕ) (original_avg : ℝ) (h1 : n = 10) (h2 : original_avg = 80) :
  let new_avg := 2 * original_avg
  new_avg = 160 := by sorry

end NUMINAMATH_CALUDE_double_average_l784_78470


namespace NUMINAMATH_CALUDE_family_age_difference_l784_78410

/-- Represents a family with changing composition over time -/
structure Family where
  initialSize : ℕ
  initialAvgAge : ℕ
  timePassed : ℕ
  currentSize : ℕ
  currentAvgAge : ℕ
  youngestChildAge : ℕ

/-- The age difference between the two youngest children in the family -/
def ageDifference (f : Family) : ℕ := sorry

theorem family_age_difference (f : Family)
  (h1 : f.initialSize = 4)
  (h2 : f.initialAvgAge = 24)
  (h3 : f.timePassed = 10)
  (h4 : f.currentSize = 6)
  (h5 : f.currentAvgAge = 24)
  (h6 : f.youngestChildAge = 3) :
  ageDifference f = 2 := by sorry

end NUMINAMATH_CALUDE_family_age_difference_l784_78410


namespace NUMINAMATH_CALUDE_min_value_a_l784_78457

theorem min_value_a (a : ℝ) : 
  (∀ x y : ℝ, |y + 4| - |y| ≤ 2^x + a / (2^x)) → 
  a ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_l784_78457


namespace NUMINAMATH_CALUDE_power_calculation_l784_78469

theorem power_calculation : 4^2011 * (-0.25)^2010 - 1 = 3 := by sorry

end NUMINAMATH_CALUDE_power_calculation_l784_78469


namespace NUMINAMATH_CALUDE_find_m_l784_78448

-- Define the function f
def f (x : ℝ) : ℝ := 2 * (2 * x + 2) + 3

-- State the theorem
theorem find_m : ∃ (m : ℝ), f (1/2 * (2 * m + 2) - 1) = 2 * (2 * m + 2) + 3 ∧ f m = 6 ∧ m = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l784_78448


namespace NUMINAMATH_CALUDE_uncle_height_l784_78440

/-- Represents the heights of James and his uncle before and after James' growth spurt -/
structure HeightScenario where
  james_initial : ℝ
  uncle : ℝ
  james_growth : ℝ
  height_diff_after : ℝ

/-- The conditions of the problem -/
def problem_conditions (h : HeightScenario) : Prop :=
  h.james_initial = (2/3) * h.uncle ∧
  h.james_growth = 10 ∧
  h.height_diff_after = 14 ∧
  h.uncle = (h.james_initial + h.james_growth + h.height_diff_after)

/-- The theorem stating that given the problem conditions, the uncle's height is 72 inches -/
theorem uncle_height (h : HeightScenario) : 
  problem_conditions h → h.uncle = 72 := by sorry

end NUMINAMATH_CALUDE_uncle_height_l784_78440


namespace NUMINAMATH_CALUDE_unique_prime_in_special_form_l784_78495

def special_form (n : ℕ) : ℚ :=
  (1 / 11) * ((10^(2*n) - 1) / 9)

theorem unique_prime_in_special_form :
  ∃! p : ℕ, Prime p ∧ ∃ n : ℕ, (special_form n : ℚ) = p ∧ p = 101 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_prime_in_special_form_l784_78495


namespace NUMINAMATH_CALUDE_journey_length_l784_78428

theorem journey_length :
  ∀ (total : ℝ),
  (total / 4 : ℝ) + 30 + (total / 3 : ℝ) = total →
  total = 72 := by
sorry

end NUMINAMATH_CALUDE_journey_length_l784_78428


namespace NUMINAMATH_CALUDE_nancy_metal_bead_sets_l784_78485

/-- The number of metal bead sets Nancy bought -/
def metal_bead_sets : ℕ := 2

/-- The cost of one set of crystal beads in dollars -/
def crystal_bead_cost : ℕ := 9

/-- The cost of one set of metal beads in dollars -/
def metal_bead_cost : ℕ := 10

/-- The total amount Nancy spent in dollars -/
def total_spent : ℕ := 29

/-- Proof that Nancy bought 2 sets of metal beads -/
theorem nancy_metal_bead_sets :
  crystal_bead_cost + metal_bead_cost * metal_bead_sets = total_spent :=
by sorry

end NUMINAMATH_CALUDE_nancy_metal_bead_sets_l784_78485


namespace NUMINAMATH_CALUDE_employee_survey_40_50_l784_78432

/-- Represents the number of employees to be selected in a stratified sampling -/
def stratified_sample (total : ℕ) (group : ℕ) (sample_size : ℕ) : ℚ :=
  (group : ℚ) / (total : ℚ) * (sample_size : ℚ)

/-- Proves that the number of employees aged 40-50 to be selected is 12 -/
theorem employee_survey_40_50 :
  let total_employees : ℕ := 350
  let over_50 : ℕ := 70
  let under_40 : ℕ := 175
  let survey_size : ℕ := 40
  let employees_40_50 : ℕ := total_employees - over_50 - under_40
  stratified_sample total_employees employees_40_50 survey_size = 12 := by
  sorry

end NUMINAMATH_CALUDE_employee_survey_40_50_l784_78432


namespace NUMINAMATH_CALUDE_expression_equality_l784_78497

theorem expression_equality (a b c d : ℕ) : 
  a = 12 → b = 13 → c = 16 → d = 11 → 3 * a^2 - 3 * b + 2 * c * d^2 = 4265 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l784_78497


namespace NUMINAMATH_CALUDE_chives_count_l784_78499

/-- Given a garden with the following properties:
  - The garden has 20 rows with 10 plants in each row.
  - Parsley is planted in the first 3 rows.
  - Rosemary is planted in the last 2 rows.
  - The remaining rows are planted with chives.
  This theorem proves that the number of chives planted is 150. -/
theorem chives_count (total_rows : ℕ) (plants_per_row : ℕ) 
  (parsley_rows : ℕ) (rosemary_rows : ℕ) : 
  total_rows = 20 → 
  plants_per_row = 10 → 
  parsley_rows = 3 → 
  rosemary_rows = 2 → 
  (total_rows - (parsley_rows + rosemary_rows)) * plants_per_row = 150 := by
  sorry

end NUMINAMATH_CALUDE_chives_count_l784_78499


namespace NUMINAMATH_CALUDE_tan_double_angle_l784_78491

theorem tan_double_angle (α : Real) 
  (h : (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 1/2) : 
  Real.tan (2 * α) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_l784_78491


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_with_8_0_digits_l784_78439

/-- A function that checks if all digits of a natural number are either 8 or 0 -/
def all_digits_eight_or_zero (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 8 ∨ d = 0

/-- The theorem statement -/
theorem largest_multiple_of_15_with_8_0_digits :
  ∃ n : ℕ, n > 0 ∧ 15 ∣ n ∧ all_digits_eight_or_zero n ∧
  (∀ m : ℕ, m > n → ¬(15 ∣ m ∧ all_digits_eight_or_zero m)) ∧
  n / 15 = 592 := by
sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_with_8_0_digits_l784_78439


namespace NUMINAMATH_CALUDE_problem_l784_78407

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem problem (a b : ℝ) (h : f (a * b) = 1) : f (a^2) + f (b^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_l784_78407


namespace NUMINAMATH_CALUDE_cuboid_4x3x3_two_sided_cubes_l784_78423

/-- Represents a cuboid with given dimensions -/
structure Cuboid where
  width : ℕ
  length : ℕ
  height : ℕ

/-- Represents a cube with given side length -/
structure Cube where
  side : ℕ

/-- Function to calculate the number of cubes with paint on exactly two sides -/
def cubesWithTwoPaintedSides (c : Cuboid) (numCubes : ℕ) : ℕ :=
  sorry

/-- Theorem stating that a 4x3x3 cuboid cut into 36 equal-sized cubes has 16 cubes with paint on exactly two sides -/
theorem cuboid_4x3x3_two_sided_cubes :
  let c : Cuboid := { width := 4, length := 3, height := 3 }
  cubesWithTwoPaintedSides c 36 = 16 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_4x3x3_two_sided_cubes_l784_78423


namespace NUMINAMATH_CALUDE_smallest_positive_integer_satisfying_congruences_l784_78412

theorem smallest_positive_integer_satisfying_congruences :
  ∃! x : ℕ+, 
    (45 * x.val + 9) % 25 = 3 ∧
    (2 * x.val) % 5 = 3 ∧
    ∀ y : ℕ+, 
      ((45 * y.val + 9) % 25 = 3 ∧ (2 * y.val) % 5 = 3) → x ≤ y :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_satisfying_congruences_l784_78412


namespace NUMINAMATH_CALUDE_tower_height_range_l784_78458

theorem tower_height_range (h : ℝ) : 
  (¬(h ≥ 200)) ∧ (¬(h ≤ 150)) ∧ (¬(h ≤ 180)) → h ∈ Set.Ioo 180 200 := by
  sorry

end NUMINAMATH_CALUDE_tower_height_range_l784_78458


namespace NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l784_78452

-- Define the sets A and S
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 7}
def S (k : ℝ) : Set ℝ := {x : ℝ | k + 1 ≤ x ∧ x ≤ 2*k - 1}

-- Theorem for condition (1)
theorem subset_condition (k : ℝ) : A ⊇ S k ↔ k ≤ 4 := by sorry

-- Theorem for condition (2)
theorem disjoint_condition (k : ℝ) : A ∩ S k = ∅ ↔ k < 2 ∨ k > 6 := by sorry

end NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l784_78452


namespace NUMINAMATH_CALUDE_coTerminalAngle_equiv_neg525_l784_78476

/-- The angle that shares the same terminal side as -525° -/
def coTerminalAngle (k : ℤ) : ℝ := 195 + k * 360

/-- Proves that coTerminalAngle shares the same terminal side as -525° -/
theorem coTerminalAngle_equiv_neg525 :
  ∀ k : ℤ, ∃ n : ℤ, coTerminalAngle k = -525 + n * 360 := by sorry

end NUMINAMATH_CALUDE_coTerminalAngle_equiv_neg525_l784_78476


namespace NUMINAMATH_CALUDE_four_by_seven_same_color_corners_l784_78480

/-- A coloring of a rectangular board. -/
def Coloring (m n : ℕ) := Fin m → Fin n → Bool

/-- A rectangle on the board, represented by its corners. -/
structure Rectangle (m n : ℕ) where
  top_left : Fin m × Fin n
  bottom_right : Fin m × Fin n
  h_valid : top_left.1 < bottom_right.1
  w_valid : top_left.2 < bottom_right.2

/-- Check if all corners of a rectangle have the same color. -/
def sameColorCorners (c : Coloring m n) (r : Rectangle m n) : Prop :=
  let (i₁, j₁) := r.top_left
  let (i₂, j₂) := r.bottom_right
  c i₁ j₁ = c i₁ j₂ ∧ c i₁ j₁ = c i₂ j₁ ∧ c i₁ j₁ = c i₂ j₂

/-- The main theorem: any 4x7 coloring has a rectangle with same-colored corners. -/
theorem four_by_seven_same_color_corners :
  ∀ (c : Coloring 4 7), ∃ (r : Rectangle 4 7), sameColorCorners c r :=
sorry


end NUMINAMATH_CALUDE_four_by_seven_same_color_corners_l784_78480


namespace NUMINAMATH_CALUDE_three_factors_for_cash_preference_l784_78464

/-- Represents an economic factor influencing payment preference --/
structure EconomicFactor where
  name : String
  description : String

/-- Represents a large retail chain --/
structure RetailChain where
  name : String
  prefersCash : Bool

/-- Determines if an economic factor contributes to cash preference --/
def contributesToCashPreference (factor : EconomicFactor) (chain : RetailChain) : Prop :=
  factor.description ≠ "" ∧ chain.prefersCash

/-- The main theorem stating that there are at least three distinct economic factors
    contributing to cash preference for large retail chains --/
theorem three_factors_for_cash_preference :
  ∃ (f1 f2 f3 : EconomicFactor) (chain : RetailChain),
    f1 ≠ f2 ∧ f1 ≠ f3 ∧ f2 ≠ f3 ∧
    contributesToCashPreference f1 chain ∧
    contributesToCashPreference f2 chain ∧
    contributesToCashPreference f3 chain :=
  sorry

end NUMINAMATH_CALUDE_three_factors_for_cash_preference_l784_78464


namespace NUMINAMATH_CALUDE_square_perimeter_from_quadratic_root_l784_78471

theorem square_perimeter_from_quadratic_root : ∃ (x₁ x₂ : ℝ), 
  (x₁ - 1) * (x₁ - 10) = 0 ∧ 
  (x₂ - 1) * (x₂ - 10) = 0 ∧ 
  x₁ ≠ x₂ ∧
  (max x₁ x₂)^2 = 100 ∧
  4 * (max x₁ x₂) = 40 :=
by sorry


end NUMINAMATH_CALUDE_square_perimeter_from_quadratic_root_l784_78471


namespace NUMINAMATH_CALUDE_sector_radius_gt_two_l784_78413

theorem sector_radius_gt_two (R : ℝ) (l : ℝ) (h : R > 0) (h_l : l > 0) :
  (1/2 * l * R = 2 * R + l) → R > 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_radius_gt_two_l784_78413


namespace NUMINAMATH_CALUDE_total_cost_usd_l784_78462

/-- Calculate the total cost of items with discounts and tax -/
def calculate_total_cost (shirt_price : ℚ) (shoe_price_diff : ℚ) (dress_price : ℚ)
  (shoe_discount : ℚ) (dress_discount : ℚ) (sales_tax : ℚ) (exchange_rate : ℚ) : ℚ :=
  let shoe_price := shirt_price + shoe_price_diff
  let discounted_shoe_price := shoe_price * (1 - shoe_discount)
  let discounted_dress_price := dress_price * (1 - dress_discount)
  let subtotal := 2 * shirt_price + discounted_shoe_price + discounted_dress_price
  let bag_price := subtotal / 2
  let total_before_tax := subtotal + bag_price
  let tax_amount := total_before_tax * sales_tax
  let total_with_tax := total_before_tax + tax_amount
  total_with_tax * exchange_rate

/-- Theorem stating the total cost in USD -/
theorem total_cost_usd :
  calculate_total_cost 12 5 25 (1/10) (1/20) (7/100) (118/100) = 11942/100 :=
by sorry

end NUMINAMATH_CALUDE_total_cost_usd_l784_78462


namespace NUMINAMATH_CALUDE_joan_total_cents_l784_78421

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- The number of quarters Joan has -/
def num_quarters : ℕ := 12

/-- The number of dimes Joan has -/
def num_dimes : ℕ := 8

/-- The number of nickels Joan has -/
def num_nickels : ℕ := 15

/-- The number of pennies Joan has -/
def num_pennies : ℕ := 25

/-- The total value of Joan's coins in cents -/
theorem joan_total_cents : 
  num_quarters * quarter_value + 
  num_dimes * dime_value + 
  num_nickels * nickel_value + 
  num_pennies * penny_value = 480 := by
  sorry

end NUMINAMATH_CALUDE_joan_total_cents_l784_78421


namespace NUMINAMATH_CALUDE_largest_five_digit_divisible_by_97_l784_78442

theorem largest_five_digit_divisible_by_97 : 
  ∀ n : ℕ, n ≤ 99999 ∧ n ≥ 10000 ∧ n % 97 = 0 → n ≤ 99930 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_five_digit_divisible_by_97_l784_78442


namespace NUMINAMATH_CALUDE_exists_lt_function_sum_product_l784_78419

/-- A function f: ℝ → ℝ is non-constant if there exist two distinct real numbers that map to different values. -/
def NonConstant (f : ℝ → ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ f x ≠ f y

/-- For any non-constant function f: ℝ → ℝ, there exist real numbers x and y such that f(x+y) < f(xy). -/
theorem exists_lt_function_sum_product (f : ℝ → ℝ) (h : NonConstant f) :
  ∃ x y : ℝ, f (x + y) < f (x * y) := by
  sorry

end NUMINAMATH_CALUDE_exists_lt_function_sum_product_l784_78419


namespace NUMINAMATH_CALUDE_lemon_profit_problem_l784_78478

theorem lemon_profit_problem (buy_lemons : ℕ) (buy_cost : ℚ) (sell_lemons : ℕ) (sell_price : ℚ) (target_profit : ℚ) : 
  buy_lemons = 4 →
  buy_cost = 15 →
  sell_lemons = 6 →
  sell_price = 25 →
  target_profit = 120 →
  ∃ (n : ℕ), n = 286 ∧ 
    n * (sell_price / sell_lemons - buy_cost / buy_lemons) ≥ target_profit ∧
    (n - 1) * (sell_price / sell_lemons - buy_cost / buy_lemons) < target_profit :=
by sorry

end NUMINAMATH_CALUDE_lemon_profit_problem_l784_78478


namespace NUMINAMATH_CALUDE_cube_cut_possible_4_5_cube_cut_impossible_4_7_l784_78411

/-- Represents a cut of a cube using four planes -/
structure CubeCut where
  planes : Fin 4 → Plane

/-- The maximum distance between any two points in a part resulting from a cube cut -/
def max_distance (cut : CubeCut) : ℝ := sorry

theorem cube_cut_possible_4_5 :
  ∃ (cut : CubeCut), max_distance cut < 4/5 := by sorry

theorem cube_cut_impossible_4_7 :
  ¬ ∃ (cut : CubeCut), max_distance cut < 4/7 := by sorry

end NUMINAMATH_CALUDE_cube_cut_possible_4_5_cube_cut_impossible_4_7_l784_78411


namespace NUMINAMATH_CALUDE_x_value_l784_78459

theorem x_value (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 8) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l784_78459


namespace NUMINAMATH_CALUDE_expected_distinct_values_formula_l784_78496

/-- The number of elements in our set -/
def n : ℕ := 2013

/-- The probability of choosing any specific value -/
def p : ℚ := 1 / n

/-- The probability of not choosing a specific value -/
def q : ℚ := 1 - p

/-- The expected number of distinct values in a set of n elements,
    each chosen independently and randomly from {1, ..., n} -/
def expected_distinct_values : ℚ := n * (1 - q^n)

/-- Theorem stating that the expected number of distinct values
    is equal to the formula derived in the solution -/
theorem expected_distinct_values_formula :
  expected_distinct_values = n * (1 - (n - 1 : ℚ)^n / n^n) :=
sorry

end NUMINAMATH_CALUDE_expected_distinct_values_formula_l784_78496


namespace NUMINAMATH_CALUDE_inequality_proof_l784_78416

theorem inequality_proof (a : ℝ) (h : a ≠ 2) :
  1 / (a^2 - 4*a + 4) > 2 / (a^3 - 8) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l784_78416


namespace NUMINAMATH_CALUDE_max_marks_calculation_l784_78489

theorem max_marks_calculation (passing_percentage : ℝ) (scored_marks : ℕ) (short_marks : ℕ) :
  passing_percentage = 0.40 →
  scored_marks = 212 →
  short_marks = 44 →
  ∃ max_marks : ℕ, max_marks = 640 ∧ 
    (scored_marks + short_marks : ℝ) / max_marks = passing_percentage :=
by sorry

end NUMINAMATH_CALUDE_max_marks_calculation_l784_78489


namespace NUMINAMATH_CALUDE_ace_king_probability_l784_78483

/-- The probability of drawing an Ace first and a King second from a modified deck -/
theorem ace_king_probability (total_cards : ℕ) (num_aces : ℕ) (num_kings : ℕ) 
  (h1 : total_cards = 54) 
  (h2 : num_aces = 5) 
  (h3 : num_kings = 4) : 
  (num_aces : ℚ) / total_cards * num_kings / (total_cards - 1) = 10 / 1426 := by
  sorry

end NUMINAMATH_CALUDE_ace_king_probability_l784_78483


namespace NUMINAMATH_CALUDE_jogger_speed_l784_78441

theorem jogger_speed (usual_distance : ℝ) (faster_speed : ℝ) (extra_distance : ℝ) :
  usual_distance = 30 →
  faster_speed = 16 →
  extra_distance = 10 →
  ∃ (usual_speed : ℝ),
    usual_speed * (usual_distance + extra_distance) / faster_speed = usual_distance ∧
    usual_speed = 12 := by
  sorry

end NUMINAMATH_CALUDE_jogger_speed_l784_78441


namespace NUMINAMATH_CALUDE_three_lines_not_necessarily_coplanar_l784_78437

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a line in 3D space
structure Line3D where
  point : Point3D
  direction : Point3D

-- Define a plane in 3D space
structure Plane3D where
  point : Point3D
  normal : Point3D

-- Function to check if a line lies on a plane
def lineOnPlane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

-- Function to check if a point is on a line
def pointOnLine (pt : Point3D) (l : Line3D) : Prop :=
  sorry

-- Theorem statement
theorem three_lines_not_necessarily_coplanar :
  ∃ (p : Point3D) (l1 l2 l3 : Line3D),
    pointOnLine p l1 ∧ pointOnLine p l2 ∧ pointOnLine p l3 ∧
    ¬∃ (plane : Plane3D), lineOnPlane l1 plane ∧ lineOnPlane l2 plane ∧ lineOnPlane l3 plane :=
sorry

end NUMINAMATH_CALUDE_three_lines_not_necessarily_coplanar_l784_78437


namespace NUMINAMATH_CALUDE_cost_of_one_juice_and_sandwich_janice_purchase_l784_78402

/-- Given the cost of multiple juices and sandwiches, calculate the cost of one juice and one sandwich. -/
theorem cost_of_one_juice_and_sandwich 
  (total_juice_cost : ℝ) 
  (juice_quantity : ℕ) 
  (total_sandwich_cost : ℝ) 
  (sandwich_quantity : ℕ) : 
  total_juice_cost / juice_quantity + total_sandwich_cost / sandwich_quantity = 5 :=
by
  sorry

/-- Specific instance of the theorem with given values -/
theorem janice_purchase : 
  (10 : ℝ) / 5 + (6 : ℝ) / 2 = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_cost_of_one_juice_and_sandwich_janice_purchase_l784_78402


namespace NUMINAMATH_CALUDE_window_offer_savings_l784_78443

/-- Represents the store's window offer -/
structure WindowOffer where
  regularPrice : ℕ
  buyQuantity : ℕ
  freeQuantity : ℕ

/-- Calculates the cost for a given number of windows under the offer -/
def costUnderOffer (offer : WindowOffer) (windowsNeeded : ℕ) : ℕ :=
  let fullSets := windowsNeeded / (offer.buyQuantity + offer.freeQuantity)
  let remainder := windowsNeeded % (offer.buyQuantity + offer.freeQuantity)
  (fullSets * offer.buyQuantity + min remainder offer.buyQuantity) * offer.regularPrice

/-- Calculates the savings for a given number of windows under the offer -/
def savingsUnderOffer (offer : WindowOffer) (windowsNeeded : ℕ) : ℕ :=
  windowsNeeded * offer.regularPrice - costUnderOffer offer windowsNeeded

/-- The main theorem to prove -/
theorem window_offer_savings (offer : WindowOffer) 
  (h1 : offer.regularPrice = 100)
  (h2 : offer.buyQuantity = 8)
  (h3 : offer.freeQuantity = 2)
  (dave_windows : ℕ) (doug_windows : ℕ)
  (h4 : dave_windows = 9)
  (h5 : doug_windows = 10) :
  savingsUnderOffer offer (dave_windows + doug_windows) - 
  (savingsUnderOffer offer dave_windows + savingsUnderOffer offer doug_windows) = 100 := by
  sorry

end NUMINAMATH_CALUDE_window_offer_savings_l784_78443


namespace NUMINAMATH_CALUDE_quadratic_root_range_l784_78456

theorem quadratic_root_range (m : ℝ) : 
  (∃ x y : ℝ, x < -1 ∧ y > 1 ∧ 
   x^2 + (m-1)*x + m^2 - 2 = 0 ∧
   y^2 + (m-1)*y + m^2 - 2 = 0) →
  0 < m ∧ m < 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l784_78456


namespace NUMINAMATH_CALUDE_complex_functional_equation_l784_78461

theorem complex_functional_equation 
  (f : ℂ → ℂ) 
  (h : ∀ z : ℂ, f z + z * f (1 - z) = 1 + z) : 
  ∀ w : ℂ, f w = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_functional_equation_l784_78461


namespace NUMINAMATH_CALUDE_rectangle_toothpicks_l784_78426

/-- The number of toothpicks needed to form a rectangle --/
def toothpicks_in_rectangle (length width : ℕ) : ℕ :=
  2 * (length + width)

/-- Theorem: A rectangle with length 20 and width 10 requires 430 toothpicks --/
theorem rectangle_toothpicks : toothpicks_in_rectangle 20 10 = 430 := by
  sorry

#eval toothpicks_in_rectangle 20 10

end NUMINAMATH_CALUDE_rectangle_toothpicks_l784_78426


namespace NUMINAMATH_CALUDE_limit_ratio_recursive_sequences_l784_78486

/-- Two sequences satisfying given recursive relations -/
def RecursiveSequences (a b : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ b 1 = 7 ∧
  ∀ n, a (n + 1) = b n - 2 * a n ∧ b (n + 1) = 3 * b n - 4 * a n

/-- The limit of the ratio of two sequences -/
def LimitRatio (a b : ℕ → ℝ) (l : ℝ) : Prop :=
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |a n / b n - l| < ε

/-- The main theorem stating the limit of the ratio of the sequences -/
theorem limit_ratio_recursive_sequences (a b : ℕ → ℝ) (h : RecursiveSequences a b) :
  LimitRatio a b (1/4) := by
  sorry

end NUMINAMATH_CALUDE_limit_ratio_recursive_sequences_l784_78486


namespace NUMINAMATH_CALUDE_triangle_problem_l784_78450

theorem triangle_problem (A B C : Real) (a b c : Real) (D : Real × Real) :
  -- Given conditions
  0 < A ∧ A < π/2 ∧
  0 < B ∧ B < π/2 ∧
  0 < C ∧ C < π/2 ∧
  A + B + C = π ∧
  a = 3 ∧
  b = Real.sqrt 13 ∧
  a * Real.sin (2 * B) = b * Real.sin A ∧
  -- Definition of point D
  D = ((1/3) * (Real.cos A, Real.sin A) + (2/3) * (Real.cos C, Real.sin C)) →
  -- Conclusions
  B = π/3 ∧
  Real.sqrt ((D.1 - Real.cos B)^2 + (D.2 - Real.sin B)^2) = (2 * Real.sqrt 19) / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l784_78450


namespace NUMINAMATH_CALUDE_correct_alarm_time_l784_78487

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60 := by sorry

/-- Calculates the difference between two times in minutes -/
def timeDifference (t1 t2 : Time) : ℕ := by sorry

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : ℕ) : Time := by sorry

theorem correct_alarm_time :
  let alarmSetTime : Time := ⟨7, 0, by sorry⟩
  let museumArrivalTime : Time := ⟨8, 50, by sorry⟩
  let museumVisitDuration : ℕ := 90 -- in minutes
  let returnHomeTime : Time := ⟨11, 50, by sorry⟩
  
  let totalTripTime := timeDifference alarmSetTime returnHomeTime
  let walkingTime := totalTripTime - museumVisitDuration
  let oneWayWalkingTime := walkingTime / 2
  
  let museumDepartureTime := addMinutes museumArrivalTime museumVisitDuration
  let actualReturnTime := addMinutes museumDepartureTime oneWayWalkingTime
  
  let correctTime := addMinutes actualReturnTime 30

  correctTime = ⟨12, 0, by sorry⟩ := by sorry

end NUMINAMATH_CALUDE_correct_alarm_time_l784_78487


namespace NUMINAMATH_CALUDE_ordered_pairs_satisfying_equations_l784_78403

theorem ordered_pairs_satisfying_equations :
  ∀ (x y : ℝ), x^2 * y = 3 ∧ x + x*y = 4 ↔ (x = 1 ∧ y = 3) ∨ (x = 3 ∧ y = 1/3) := by
  sorry

end NUMINAMATH_CALUDE_ordered_pairs_satisfying_equations_l784_78403


namespace NUMINAMATH_CALUDE_inequalities_hold_l784_78472

theorem inequalities_hold (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a * b ≤ 1 / 4) ∧ (Real.sqrt a + Real.sqrt b ≤ Real.sqrt 2) ∧ (a^2 + b^2 ≥ 1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_hold_l784_78472


namespace NUMINAMATH_CALUDE_concatenatedDecimal_irrational_l784_78444

/-- The infinite decimal formed by concatenating all natural numbers after the decimal point -/
noncomputable def concatenatedDecimal : ℝ :=
  sorry

/-- Function that generates the n-th digit of the concatenatedDecimal -/
def nthDigit (n : ℕ) : ℕ :=
  sorry

theorem concatenatedDecimal_irrational : Irrational concatenatedDecimal :=
  sorry

end NUMINAMATH_CALUDE_concatenatedDecimal_irrational_l784_78444


namespace NUMINAMATH_CALUDE_swim_meet_transportation_l784_78401

/-- Represents the swim meet transportation problem -/
theorem swim_meet_transportation (num_cars : ℕ) (people_per_car : ℕ) (people_per_van : ℕ)
  (max_car_capacity : ℕ) (max_van_capacity : ℕ) (additional_capacity : ℕ) :
  num_cars = 2 →
  people_per_car = 5 →
  people_per_van = 3 →
  max_car_capacity = 6 →
  max_van_capacity = 8 →
  additional_capacity = 17 →
  ∃ (num_vans : ℕ), 
    num_vans = 3 ∧
    (num_cars * max_car_capacity + num_vans * max_van_capacity) - 
    (num_cars * people_per_car + num_vans * people_per_van) = additional_capacity :=
by sorry

end NUMINAMATH_CALUDE_swim_meet_transportation_l784_78401


namespace NUMINAMATH_CALUDE_jack_deer_hunting_l784_78449

/-- The number of times Jack goes hunting per month -/
def hunts_per_month : ℕ := 6

/-- The duration of the hunting season in months -/
def hunting_season_months : ℕ := 3

/-- The number of deer Jack catches per hunting trip -/
def deer_per_hunt : ℕ := 2

/-- The weight of each deer in pounds -/
def deer_weight : ℕ := 600

/-- The fraction of deer weight Jack keeps -/
def kept_fraction : ℚ := 1 / 2

/-- The total amount of deer Jack keeps in pounds -/
def deer_kept : ℕ := 10800

theorem jack_deer_hunting :
  hunts_per_month * hunting_season_months * deer_per_hunt * deer_weight * kept_fraction = deer_kept := by
  sorry

end NUMINAMATH_CALUDE_jack_deer_hunting_l784_78449


namespace NUMINAMATH_CALUDE_total_money_calculation_l784_78454

-- Define the proportions
def prop1 : ℚ := 1/2
def prop2 : ℚ := 1/3
def prop3 : ℚ := 3/4

-- Define the value of the second part
def second_part : ℝ := 164.6315789473684

-- Theorem statement
theorem total_money_calculation (total : ℝ) :
  (total * (prop2 / (prop1 + prop2 + prop3)) = second_part) →
  total = 65.1578947368421 := by
sorry

end NUMINAMATH_CALUDE_total_money_calculation_l784_78454


namespace NUMINAMATH_CALUDE_train_platform_crossing_time_l784_78494

theorem train_platform_crossing_time
  (train_length : ℝ)
  (tree_crossing_time : ℝ)
  (platform_length : ℝ)
  (h1 : train_length = 1200)
  (h2 : tree_crossing_time = 120)
  (h3 : platform_length = 800) :
  let train_speed := train_length / tree_crossing_time
  let total_distance := train_length + platform_length
  total_distance / train_speed = 200 :=
by sorry

end NUMINAMATH_CALUDE_train_platform_crossing_time_l784_78494


namespace NUMINAMATH_CALUDE_square_area_ratio_l784_78490

theorem square_area_ratio (side_c side_d : ℝ) (hc : side_c = 24) (hd : side_d = 54) :
  (side_c^2) / (side_d^2) = 16 / 81 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l784_78490


namespace NUMINAMATH_CALUDE_combined_discount_rate_l784_78460

/-- Calculate the combined rate of discount for three items -/
theorem combined_discount_rate
  (bag_marked : ℝ) (shoes_marked : ℝ) (hat_marked : ℝ)
  (bag_discounted : ℝ) (shoes_discounted : ℝ) (hat_discounted : ℝ)
  (h_bag : bag_marked = 150 ∧ bag_discounted = 120)
  (h_shoes : shoes_marked = 100 ∧ shoes_discounted = 80)
  (h_hat : hat_marked = 50 ∧ hat_discounted = 40) :
  let total_marked := bag_marked + shoes_marked + hat_marked
  let total_discounted := bag_discounted + shoes_discounted + hat_discounted
  let discount_rate := (total_marked - total_discounted) / total_marked
  discount_rate = 0.2 := by
sorry

end NUMINAMATH_CALUDE_combined_discount_rate_l784_78460


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l784_78474

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 - 36*x + 320 ≤ 16} = Set.Icc 16 19 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l784_78474


namespace NUMINAMATH_CALUDE_task_completion_probability_l784_78468

theorem task_completion_probability (p1 p2 : ℚ) (h1 : p1 = 5/8) (h2 : p2 = 3/5) :
  p1 * (1 - p2) = 1/4 := by sorry

end NUMINAMATH_CALUDE_task_completion_probability_l784_78468


namespace NUMINAMATH_CALUDE_wedding_attendance_percentage_l784_78418

def expected_attendees : ℕ := 220
def actual_attendees : ℕ := 209

theorem wedding_attendance_percentage :
  (expected_attendees - actual_attendees : ℚ) / expected_attendees * 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_wedding_attendance_percentage_l784_78418


namespace NUMINAMATH_CALUDE_num_employees_correct_l784_78405

/-- The number of employees in an organization, excluding the manager -/
def num_employees : ℕ := 15

/-- The average monthly salary of employees, excluding the manager -/
def avg_salary : ℕ := 1800

/-- The increase in average salary when the manager's salary is added -/
def avg_increase : ℕ := 150

/-- The manager's monthly salary -/
def manager_salary : ℕ := 4200

/-- Theorem stating that the number of employees is correct given the conditions -/
theorem num_employees_correct :
  (avg_salary * num_employees + manager_salary) / (num_employees + 1) = avg_salary + avg_increase :=
by sorry

end NUMINAMATH_CALUDE_num_employees_correct_l784_78405


namespace NUMINAMATH_CALUDE_no_solutions_exist_l784_78445

theorem no_solutions_exist : ¬∃ (x y z : ℝ), (x + y = 3) ∧ (x * y - z^2 = 4) := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_exist_l784_78445


namespace NUMINAMATH_CALUDE_like_terms_exponent_value_l784_78400

theorem like_terms_exponent_value (a b : ℝ) (n m : ℕ) :
  (∃ k : ℝ, k * a^(n+1) * b^n = -3 * a^(2*m) * b^3) →
  n^m = 9 := by
sorry

end NUMINAMATH_CALUDE_like_terms_exponent_value_l784_78400


namespace NUMINAMATH_CALUDE_campaign_funds_proof_l784_78498

/-- The total campaign funds raised by the 40th president -/
def total_funds : ℝ := 10000

/-- The amount raised by friends -/
def friends_contribution (total : ℝ) : ℝ := 0.4 * total

/-- The amount raised by family -/
def family_contribution (total : ℝ) : ℝ := 0.3 * (total - friends_contribution total)

/-- The amount contributed by the president himself -/
def president_contribution : ℝ := 4200

theorem campaign_funds_proof :
  friends_contribution total_funds +
  family_contribution total_funds +
  president_contribution = total_funds :=
by sorry

end NUMINAMATH_CALUDE_campaign_funds_proof_l784_78498


namespace NUMINAMATH_CALUDE_cakes_per_friend_l784_78427

def total_cakes : ℕ := 8
def num_friends : ℕ := 4

theorem cakes_per_friend :
  total_cakes / num_friends = 2 :=
by sorry

end NUMINAMATH_CALUDE_cakes_per_friend_l784_78427
