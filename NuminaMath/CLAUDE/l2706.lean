import Mathlib

namespace B_alone_time_l2706_270614

/-- The time it takes for A and B together to complete the job -/
def time_AB : ℝ := 3

/-- The time it takes for B and C together to complete the job -/
def time_BC : ℝ := 6

/-- The time it takes for A and C together to complete the job -/
def time_AC : ℝ := 3.6

/-- The rate at which A completes the job -/
def rate_A : ℝ := sorry

/-- The rate at which B completes the job -/
def rate_B : ℝ := sorry

/-- The rate at which C completes the job -/
def rate_C : ℝ := sorry

theorem B_alone_time : 
  rate_A + rate_B = 1 / time_AB ∧ 
  rate_B + rate_C = 1 / time_BC ∧ 
  rate_A + rate_C = 1 / time_AC → 
  1 / rate_B = 9 := by sorry

end B_alone_time_l2706_270614


namespace mirror_frame_areas_l2706_270667

/-- Represents the dimensions and properties of a rectangular mirror frame -/
structure MirrorFrame where
  outer_width : ℝ
  outer_length : ℝ
  frame_width : ℝ

/-- Calculates the area of the frame alone -/
def frame_area (frame : MirrorFrame) : ℝ :=
  frame.outer_width * frame.outer_length - (frame.outer_width - 2 * frame.frame_width) * (frame.outer_length - 2 * frame.frame_width)

/-- Calculates the area of the mirror inside the frame -/
def mirror_area (frame : MirrorFrame) : ℝ :=
  (frame.outer_width - 2 * frame.frame_width) * (frame.outer_length - 2 * frame.frame_width)

theorem mirror_frame_areas (frame : MirrorFrame) 
  (h1 : frame.outer_width = 100)
  (h2 : frame.outer_length = 120)
  (h3 : frame.frame_width = 15) :
  frame_area frame = 5700 ∧ mirror_area frame = 6300 := by
  sorry

end mirror_frame_areas_l2706_270667


namespace largest_even_digit_multiple_of_5_l2706_270684

/-- A function that checks if all digits of a natural number are even -/
def allDigitsEven (n : ℕ) : Prop := sorry

/-- A function that returns the largest positive integer less than 10000 
    with all even digits that is a multiple of 5 -/
noncomputable def largestEvenDigitMultipleOf5 : ℕ := sorry

/-- Theorem stating that 8860 is the largest positive integer less than 10000 
    with all even digits that is a multiple of 5 -/
theorem largest_even_digit_multiple_of_5 : 
  largestEvenDigitMultipleOf5 = 8860 ∧ 
  allDigitsEven 8860 ∧ 
  8860 < 10000 ∧ 
  8860 % 5 = 0 :=
by sorry

end largest_even_digit_multiple_of_5_l2706_270684


namespace fourth_root_equivalence_l2706_270685

theorem fourth_root_equivalence (x : ℝ) (hx : x > 0) : 
  (x^3 * (x^2)^(1/2))^(1/4) = x := by
  sorry

end fourth_root_equivalence_l2706_270685


namespace probability_blue_between_red_and_triple_red_l2706_270679

-- Define the probability space
def Ω : Type := ℝ × ℝ

-- Define the probability measure
noncomputable def P : Set Ω → ℝ := sorry

-- Define the event where the blue point is greater than the red point but less than three times the red point
def E : Set Ω := {ω : Ω | let (x, y) := ω; 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ x < y ∧ y < 3*x}

-- State the theorem
theorem probability_blue_between_red_and_triple_red : P E = 5/6 := sorry

end probability_blue_between_red_and_triple_red_l2706_270679


namespace quadratic_fixed_point_l2706_270681

-- Define the quadratic function
def f (p q : ℝ) (x : ℝ) : ℝ := x^2 + p*x + q

-- Define the theorem
theorem quadratic_fixed_point 
  (p q : ℝ) 
  (h1 : ∀ x ∈ Set.Icc 3 5, |f p q x| ≤ 1/2)
  (h2 : f p q ((7 + Real.sqrt 15) / 2) = 0) :
  (f p q)^[2017] ((7 + Real.sqrt 15) / 2) = (7 - Real.sqrt 15) / 2 :=
sorry

end quadratic_fixed_point_l2706_270681


namespace area_ratio_theorem_l2706_270682

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define point W on side XZ
def W (t : Triangle) : ℝ × ℝ := sorry

-- Define the conditions
axiom XW_length : ∀ t : Triangle, dist (t.X) (W t) = 9
axiom WZ_length : ∀ t : Triangle, dist (W t) (t.Z) = 15

-- Define the areas of triangles XYW and WYZ
def area_XYW (t : Triangle) : ℝ := sorry
def area_WYZ (t : Triangle) : ℝ := sorry

-- State the theorem
theorem area_ratio_theorem (t : Triangle) :
  (area_XYW t) / (area_WYZ t) = 3 / 5 :=
sorry

end area_ratio_theorem_l2706_270682


namespace product_calculation_l2706_270616

theorem product_calculation : (1/2 : ℚ) * 8 * (1/8 : ℚ) * 32 * (1/32 : ℚ) * 128 * (1/128 : ℚ) * 512 * (1/512 : ℚ) * 2048 = 1024 := by
  sorry

end product_calculation_l2706_270616


namespace quadratic_root_value_l2706_270651

theorem quadratic_root_value (p q : ℝ) : 
  3 * p^2 - 5 * p - 8 = 0 →
  3 * q^2 - 5 * q - 8 = 0 →
  p ≠ q →
  (9 * p^4 - 9 * q^4) / (p - q) = 365 := by
sorry

end quadratic_root_value_l2706_270651


namespace class_mean_score_l2706_270607

/-- Proves that the overall mean score of a class is 76.17% given the specified conditions -/
theorem class_mean_score (total_students : ℕ) (group1_students : ℕ) (group2_students : ℕ)
  (group1_avg : ℚ) (group2_avg : ℚ) :
  total_students = 48 →
  group1_students = 40 →
  group2_students = 8 →
  group1_avg = 75 / 100 →
  group2_avg = 82 / 100 →
  let overall_avg := (group1_students * group1_avg + group2_students * group2_avg) / total_students
  overall_avg = 7617 / 10000 := by
  sorry

end class_mean_score_l2706_270607


namespace sms_is_fraudulent_l2706_270650

/-- Represents an SMS message -/
structure SMS where
  claims_prize : Bool
  requests_payment : Bool
  recipient_participated : Bool

/-- Represents characteristics of a legitimate contest -/
structure LegitimateContest where
  requires_payment : Bool

/-- Determines if an SMS is fraudulent based on given conditions -/
def is_fraudulent (sms : SMS) (contest : LegitimateContest) : Prop :=
  sms.claims_prize ∧ 
  sms.requests_payment ∧ 
  ¬sms.recipient_participated ∧
  ¬contest.requires_payment

/-- Theorem stating that an SMS with specific characteristics is fraudulent -/
theorem sms_is_fraudulent (sms : SMS) (contest : LegitimateContest) :
  sms.claims_prize = true →
  sms.requests_payment = true →
  sms.recipient_participated = false →
  contest.requires_payment = false →
  is_fraudulent sms contest := by
  sorry

#check sms_is_fraudulent

end sms_is_fraudulent_l2706_270650


namespace base7_to_base10_equality_l2706_270676

/-- Conversion from base 7 to base 10 -/
def base7to10 (n : ℕ) : ℕ := 
  7 * 7 * (n / 100) + 7 * ((n / 10) % 10) + (n % 10)

theorem base7_to_base10_equality (c d e : ℕ) : 
  (c < 10 ∧ d < 10 ∧ e < 10) → 
  (base7to10 761 = 100 * c + 10 * d + e) → 
  (d * e : ℚ) / 15 = 48 / 15 := by
sorry

end base7_to_base10_equality_l2706_270676


namespace students_AD_combined_prove_students_AD_combined_l2706_270633

/-- The number of students in classes A and B combined -/
def students_AB : ℕ := 83

/-- The number of students in classes B and C combined -/
def students_BC : ℕ := 86

/-- The number of students in classes C and D combined -/
def students_CD : ℕ := 88

/-- Theorem stating that the number of students in classes A and D combined is 85 -/
theorem students_AD_combined : ℕ := 85

/-- Proof of the theorem -/
theorem prove_students_AD_combined : students_AD_combined = 85 := by
  sorry

end students_AD_combined_prove_students_AD_combined_l2706_270633


namespace chord_intersection_ratio_l2706_270603

-- Define a circle
variable (circle : Set ℝ × ℝ)

-- Define points E, F, G, H, Q
variable (E F G H Q : ℝ × ℝ)

-- Define that EF and GH are chords of the circle
variable (chord_EF : Set (ℝ × ℝ))
variable (chord_GH : Set (ℝ × ℝ))

-- Define that Q is the intersection point of EF and GH
variable (intersect_Q : Q ∈ chord_EF ∩ chord_GH)

-- Define lengths
def EQ : ℝ := sorry
def FQ : ℝ := sorry
def GQ : ℝ := sorry
def HQ : ℝ := sorry

-- State the theorem
theorem chord_intersection_ratio 
  (h1 : EQ = 4) 
  (h2 : GQ = 10) : 
  FQ / HQ = 5 / 2 := by sorry

end chord_intersection_ratio_l2706_270603


namespace school_run_speed_l2706_270661

theorem school_run_speed (v : ℝ) (h : v > 0) : 
  (v + 2) / v = 2.5 → (v + 4) / v = 4 := by
  sorry

end school_run_speed_l2706_270661


namespace message_pairs_l2706_270664

theorem message_pairs (n m : ℕ) (hn : n = 100) (hm : m = 50) :
  let total_messages := n * m
  let max_unique_pairs := n * (n - 1) / 2
  total_messages - max_unique_pairs = 50 :=
by sorry

end message_pairs_l2706_270664


namespace original_number_calculation_l2706_270680

theorem original_number_calculation (r : ℝ) : 
  (r + 0.15 * r) - (r - 0.30 * r) = 40 → r = 40 / 0.45 := by
sorry

end original_number_calculation_l2706_270680


namespace company_kw_price_percentage_l2706_270612

/-- The price of company KW as a percentage of the combined assets of companies A and B -/
theorem company_kw_price_percentage (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) : 
  let p := 1.2 * a  -- Price of company KW
  let combined_assets := a + b
  p / combined_assets = 0.75 := by sorry

end company_kw_price_percentage_l2706_270612


namespace wall_width_l2706_270648

/-- Theorem: Width of a wall with specific proportions and volume --/
theorem wall_width (w h l : ℝ) (volume : ℝ) : 
  h = 6 * w →
  l = 7 * h →
  volume = w * h * l →
  volume = 129024 →
  w = 8 := by
  sorry

end wall_width_l2706_270648


namespace valid_configuration_exists_l2706_270683

/-- Represents a point in a 2D grid --/
structure Point where
  x : Int
  y : Int

/-- Represents a line containing 4 points --/
structure Line where
  points : Fin 4 → Point

/-- The configuration of ships --/
structure ShipConfiguration where
  ships : Fin 10 → Point
  lines : Fin 5 → Line

/-- Checks if a line contains 4 distinct points from the given set of points --/
def Line.isValidLine (l : Line) (points : Fin 10 → Point) : Prop :=
  ∃ (indices : Fin 4 → Fin 10), (∀ i j, i ≠ j → indices i ≠ indices j) ∧
    (∀ i, l.points i = points (indices i))

/-- Checks if a configuration is valid --/
def ShipConfiguration.isValid (config : ShipConfiguration) : Prop :=
  ∀ l, config.lines l |>.isValidLine config.ships

/-- The theorem stating that a valid configuration exists --/
theorem valid_configuration_exists : ∃ (config : ShipConfiguration), config.isValid := by
  sorry


end valid_configuration_exists_l2706_270683


namespace dina_machine_l2706_270600

def f (x : ℚ) : ℚ := 2 * x - 3

theorem dina_machine (x : ℚ) : f (f x) = -35 → x = -13/2 := by
  sorry

end dina_machine_l2706_270600


namespace expression_evaluation_l2706_270644

theorem expression_evaluation :
  let a : ℚ := 2
  let b : ℚ := 1/2
  2 * (a^2 - 2*a*b) - 3 * (a^2 - a*b - 4*b^2) = -2 :=
by sorry

end expression_evaluation_l2706_270644


namespace set_operations_and_range_l2706_270609

-- Define the sets A, B, and C
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | 1 < x ∧ x < 4}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- State the theorem
theorem set_operations_and_range :
  (∀ a : ℝ,
    (A ∩ B = {x | 1 < x ∧ x ≤ 3}) ∧
    (A ∪ B = {x | -1 ≤ x ∧ x < 4}) ∧
    ((Aᶜ ∩ Bᶜ) = {x | x < -1 ∨ 4 ≤ x}) ∧
    ((B ∩ C a = B) → a ≥ 4)) :=
by sorry

end set_operations_and_range_l2706_270609


namespace class_size_correct_l2706_270675

/-- The number of students in class A -/
def class_size : ℕ := 30

/-- The number of students who like social studies -/
def social_studies_fans : ℕ := 25

/-- The number of students who like music -/
def music_fans : ℕ := 32

/-- The number of students who like both social studies and music -/
def both_fans : ℕ := 27

/-- Theorem stating that the class size is correct given the conditions -/
theorem class_size_correct :
  class_size = social_studies_fans + music_fans - both_fans ∧
  class_size = social_studies_fans + music_fans - both_fans :=
by sorry

end class_size_correct_l2706_270675


namespace only_2222_cannot_form_24_l2706_270631

/-- A hand is a list of four natural numbers representing card values. -/
def Hand := List Nat

/-- Possible arithmetic operations -/
inductive Operation
| Add
| Sub
| Mul
| Div

/-- Apply an operation to two natural numbers -/
def applyOp (op : Operation) (a b : Nat) : Option Nat :=
  match op with
  | Operation.Add => some (a + b)
  | Operation.Sub => if a ≥ b then some (a - b) else none
  | Operation.Mul => some (a * b)
  | Operation.Div => if b ≠ 0 && a % b = 0 then some (a / b) else none

/-- Check if a hand can form 24 using the given operations and rules -/
def canForm24 (hand : Hand) : Prop :=
  ∃ (op1 op2 op3 : Operation) (perm : List Nat),
    perm.length = 4 ∧
    perm.toFinset = hand.toFinset ∧
    (∃ (x y z : Nat),
      applyOp op1 perm[0]! perm[1]! = some x ∧
      applyOp op2 x perm[2]! = some y ∧
      applyOp op3 y perm[3]! = some 24)

theorem only_2222_cannot_form_24 :
  canForm24 [1, 2, 3, 3] ∧
  canForm24 [1, 5, 5, 5] ∧
  canForm24 [3, 3, 3, 3] ∧
  ¬canForm24 [2, 2, 2, 2] := by
  sorry

end only_2222_cannot_form_24_l2706_270631


namespace range_of_a_l2706_270660

theorem range_of_a (a : ℝ) :
  (((∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≥ 0) ∨ 
    (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0)) ∧
   ¬((∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≥ 0) ∧ 
     (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0))) →
  (a > 1 ∨ (-2 < a ∧ a < 1)) := by
  sorry

end range_of_a_l2706_270660


namespace sequence_closed_form_l2706_270619

theorem sequence_closed_form (a : ℕ → ℤ) :
  a 1 = 0 ∧
  (∀ n : ℕ, n ≥ 2 → a n - 2 * a (n - 1) = n^2 - 3) →
  ∀ n : ℕ, n ≥ 1 → a n = 2^(n + 2) - n^2 - 4*n - 3 :=
by sorry

end sequence_closed_form_l2706_270619


namespace cubic_equation_property_l2706_270695

/-- A cubic equation with coefficients a, b, c, and three non-zero real roots forming a geometric progression -/
structure CubicEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  root1 : ℝ
  root2 : ℝ
  root3 : ℝ
  nonzero_roots : root1 ≠ 0 ∧ root2 ≠ 0 ∧ root3 ≠ 0
  is_root1 : root1^3 + a*root1^2 + b*root1 + c = 0
  is_root2 : root2^3 + a*root2^2 + b*root2 + c = 0
  is_root3 : root3^3 + a*root3^2 + b*root3 + c = 0
  geometric_progression : ∃ (q : ℝ), q ≠ 0 ∧ q ≠ 1 ∧ (root2 = q * root1) ∧ (root3 = q * root2)

/-- The theorem stating that a^3c - b^3 = 0 for a cubic equation with three non-zero real roots in geometric progression -/
theorem cubic_equation_property (eq : CubicEquation) : eq.a^3 * eq.c - eq.b^3 = 0 := by
  sorry

end cubic_equation_property_l2706_270695


namespace double_fraction_value_l2706_270657

theorem double_fraction_value (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (2*x * 2*y) / (2*x + 2*y) = 2 * (x*y / (x + y)) :=
by sorry

end double_fraction_value_l2706_270657


namespace function_properties_l2706_270637

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^3 + x^2 + b * x

-- Define the function g
def g (a b x : ℝ) : ℝ := f a b x + (3 * a * x^2 + 2 * x + b)

-- Main theorem
theorem function_properties (a b : ℝ) :
  (∀ x, g a b x = -g a b (-x)) →
  (∃ f_simplified : ℝ → ℝ, 
    (∀ x, f a b x = f_simplified x) ∧
    (∀ x, f_simplified x = x^2 - x) ∧
    (∀ x ∈ Set.Icc 1 2, HasDerivAt (g a b) ((2 : ℝ) * x + 1) x) ∧
    (g a b 1 = 1) ∧
    (g a b 2 = 5) ∧
    (∀ x ∈ Set.Icc 1 2, g a b x ≥ 1 ∧ g a b x ≤ 5)) :=
by sorry

end function_properties_l2706_270637


namespace speed_difference_l2706_270694

/-- The difference in average speeds between no traffic and heavy traffic conditions -/
theorem speed_difference (distance : ℝ) (time_heavy : ℝ) (time_no : ℝ)
  (h_distance : distance = 200)
  (h_time_heavy : time_heavy = 5)
  (h_time_no : time_no = 4) :
  distance / time_no - distance / time_heavy = 10 := by
  sorry

end speed_difference_l2706_270694


namespace fourth_root_equation_solutions_l2706_270659

theorem fourth_root_equation_solutions :
  ∀ x : ℝ, (x^(1/4) = 12 / (7 - x^(1/4))) ↔ (x = 81 ∨ x = 256) :=
by sorry

end fourth_root_equation_solutions_l2706_270659


namespace least_integer_divisible_by_four_primes_l2706_270626

theorem least_integer_divisible_by_four_primes : 
  ∃ n : ℕ, (n > 0) ∧ 
  (∃ p q r s : ℕ, Prime p ∧ Prime q ∧ Prime r ∧ Prime s ∧ 
   p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧
   n % p = 0 ∧ n % q = 0 ∧ n % r = 0 ∧ n % s = 0) ∧
  (∀ m : ℕ, m > 0 → 
    (∃ p q r s : ℕ, Prime p ∧ Prime q ∧ Prime r ∧ Prime s ∧ 
     p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧
     m % p = 0 ∧ m % q = 0 ∧ m % r = 0 ∧ m % s = 0) → 
    m ≥ 210) ∧
  n = 210 :=
by sorry

end least_integer_divisible_by_four_primes_l2706_270626


namespace eighth_term_value_l2706_270654

/-- Given a sequence {aₙ} where the sum of the first n terms Sₙ = n², 
    prove that the 8th term a₈ = 15 -/
theorem eighth_term_value (a : ℕ → ℕ) (S : ℕ → ℕ) 
    (h : ∀ n, S n = n^2) : 
    a 8 = 15 := by
  sorry

end eighth_term_value_l2706_270654


namespace equal_cell_squares_count_l2706_270699

/-- Represents a cell in the grid -/
inductive Cell
| White
| Black

/-- Represents the 5x5 grid configuration -/
def Grid : Type := Fin 5 → Fin 5 → Cell

/-- The specific grid configuration given in the problem -/
def problem_grid : Grid := sorry

/-- A square in the grid -/
structure Square where
  top_left : Fin 5 × Fin 5
  size : Nat

/-- Checks if a square has equal number of black and white cells -/
def has_equal_cells (g : Grid) (s : Square) : Bool := sorry

/-- Counts the number of squares with equal black and white cells -/
def count_equal_squares (g : Grid) : Nat := sorry

/-- The main theorem -/
theorem equal_cell_squares_count :
  count_equal_squares problem_grid = 16 := by sorry

end equal_cell_squares_count_l2706_270699


namespace grains_in_cup_is_480_l2706_270605

/-- Represents the number of grains of rice in one cup -/
def grains_in_cup : ℕ :=
  let half_cup_tablespoons : ℕ := 8
  let teaspoons_per_tablespoon : ℕ := 3
  let grains_per_teaspoon : ℕ := 10
  2 * (half_cup_tablespoons * teaspoons_per_tablespoon * grains_per_teaspoon)

/-- Theorem stating that there are 480 grains of rice in one cup -/
theorem grains_in_cup_is_480 : grains_in_cup = 480 := by
  sorry

end grains_in_cup_is_480_l2706_270605


namespace pages_copied_for_fifteen_dollars_l2706_270627

/-- Given that 4 pages cost 6 cents, prove that $15 (1500 cents) will allow copying 1000 pages. -/
theorem pages_copied_for_fifteen_dollars :
  let pages_per_six_cents : ℚ := 4
  let cents_per_four_pages : ℚ := 6
  let total_cents : ℚ := 1500
  (total_cents * pages_per_six_cents) / cents_per_four_pages = 1000 := by
  sorry

end pages_copied_for_fifteen_dollars_l2706_270627


namespace first_liquid_volume_l2706_270640

theorem first_liquid_volume (x : ℝ) : 
  (0.75 * x + 63) / (x + 90) = 0.7263157894736842 → x = 100 := by
  sorry

end first_liquid_volume_l2706_270640


namespace product_of_difference_and_sum_of_squares_l2706_270665

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 5) 
  (h2 : a^2 + b^2 = 13) : 
  a * b = -6 := by sorry

end product_of_difference_and_sum_of_squares_l2706_270665


namespace inverse_square_inequality_l2706_270634

theorem inverse_square_inequality {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : x ≤ y) :
  1 / y ^ 2 ≤ 1 / x ^ 2 := by
  sorry

end inverse_square_inequality_l2706_270634


namespace min_value_on_negative_interval_l2706_270610

/-- Given positive real numbers a and b, and a function f with maximum value 4 on [0,1],
    prove that the minimum value of f on [-1,0] is -3/2 -/
theorem min_value_on_negative_interval
  (a b : ℝ) (f : ℝ → ℝ)
  (a_pos : 0 < a) (b_pos : 0 < b)
  (f_def : ∀ x, f x = a * x^3 + b * x + 2^x)
  (max_value : ∀ x ∈ Set.Icc 0 1, f x ≤ 4)
  (max_achieved : ∃ x ∈ Set.Icc 0 1, f x = 4) :
  ∀ x ∈ Set.Icc (-1) 0, f x ≥ -3/2 ∧ ∃ y ∈ Set.Icc (-1) 0, f y = -3/2 :=
by sorry

end min_value_on_negative_interval_l2706_270610


namespace painted_face_probability_for_specific_prism_l2706_270697

/-- Represents a rectangular prism -/
structure RectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the total number of cubes in the prism -/
def total_cubes (p : RectangularPrism) : ℕ :=
  p.length * p.width * p.height

/-- Calculates the number of corner cubes -/
def corner_cubes : ℕ := 8

/-- Calculates the number of edge cubes -/
def edge_cubes (p : RectangularPrism) : ℕ :=
  4 * (p.length - 2) + 8 * (p.height - 2)

/-- Calculates the number of face cubes -/
def face_cubes (p : RectangularPrism) : ℕ :=
  2 * (p.length * p.height) - edge_cubes p - corner_cubes

/-- Calculates the probability of a randomly chosen cube showing a painted face when rolled -/
def painted_face_probability (p : RectangularPrism) : ℚ :=
  (3 * corner_cubes + 2 * edge_cubes p + face_cubes p) / (6 * total_cubes p)

theorem painted_face_probability_for_specific_prism :
  let p : RectangularPrism := ⟨20, 1, 7⟩
  painted_face_probability p = 9 / 35 := by
  sorry

end painted_face_probability_for_specific_prism_l2706_270697


namespace total_CDs_is_448_l2706_270601

/-- The number of shelves in Store A -/
def store_A_shelves : ℕ := 5

/-- The number of CD racks per shelf in Store A -/
def store_A_racks_per_shelf : ℕ := 7

/-- The number of CDs per rack in Store A -/
def store_A_CDs_per_rack : ℕ := 8

/-- The number of shelves in Store B -/
def store_B_shelves : ℕ := 4

/-- The number of CD racks per shelf in Store B -/
def store_B_racks_per_shelf : ℕ := 6

/-- The number of CDs per rack in Store B -/
def store_B_CDs_per_rack : ℕ := 7

/-- The total number of CDs that can be held in Store A and Store B together -/
def total_CDs : ℕ := 
  (store_A_shelves * store_A_racks_per_shelf * store_A_CDs_per_rack) +
  (store_B_shelves * store_B_racks_per_shelf * store_B_CDs_per_rack)

/-- Theorem stating that the total number of CDs that can be held in Store A and Store B together is 448 -/
theorem total_CDs_is_448 : total_CDs = 448 := by
  sorry

end total_CDs_is_448_l2706_270601


namespace negation_existence_quadratic_l2706_270639

theorem negation_existence_quadratic (m : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2*x + m ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + m > 0) := by
  sorry

end negation_existence_quadratic_l2706_270639


namespace function_f_property_l2706_270653

/-- A function satisfying the given properties -/
def FunctionF (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = 2 - f x) ∧
  (∀ x y, x ≤ y ∧ y ≤ 0 → f x ≤ f y)

/-- The theorem statement -/
theorem function_f_property (f : ℝ → ℝ) (a : ℝ) 
  (hf : FunctionF f) 
  (h : ∀ x ∈ Set.Icc 1 2, f (a * x + 2) + f 1 ≤ 2) : 
  a ∈ Set.Iic (-3) :=
sorry

end function_f_property_l2706_270653


namespace constant_t_equality_l2706_270672

theorem constant_t_equality (x : ℝ) : 
  (5*x^2 - 6*x + 7) * (4*x^2 + (-6)*x + 10) = 20*x^4 - 54*x^3 + 114*x^2 - 102*x + 70 := by
  sorry


end constant_t_equality_l2706_270672


namespace average_of_numbers_divisible_by_4_l2706_270621

theorem average_of_numbers_divisible_by_4 :
  let numbers := (Finset.range 25).filter (fun n => 6 < n + 6 ∧ n + 6 ≤ 30 ∧ (n + 6) % 4 = 0)
  (numbers.sum id) / numbers.card = 18 := by
  sorry

end average_of_numbers_divisible_by_4_l2706_270621


namespace rectangle_area_increase_rectangle_area_percentage_increase_l2706_270617

theorem rectangle_area_increase (l w : ℝ) (hl : l > 0) (hw : w > 0) : 
  (1.3 * l) * (1.2 * w) = 1.56 * (l * w) := by
  sorry

theorem rectangle_area_percentage_increase : 
  (1.56 - 1) * 100 = 56 := by
  sorry

end rectangle_area_increase_rectangle_area_percentage_increase_l2706_270617


namespace tenths_vs_thousandths_l2706_270630

def number : ℚ := 85247.2048

theorem tenths_vs_thousandths :
  (number - number.floor) * 10 % 1 * 10 = 
  100 * ((number - number.floor) * 1000 % 10) / 10 := by
  sorry

end tenths_vs_thousandths_l2706_270630


namespace problem_1_problem_2_problem_3_problem_4_l2706_270656

-- Problem 1
theorem problem_1 : (-7) - (-8) + (-9) - 14 = -22 := by sorry

-- Problem 2
theorem problem_2 : (-4) * (-3)^2 - 14 / (-7) = -34 := by sorry

-- Problem 3
theorem problem_3 : (3/10 - 1/4 + 4/5) * (-20) = -17 := by sorry

-- Problem 4
theorem problem_4 : (-2)^2 / |1-3| + 3 * (1/2 - 1) = 1/2 := by sorry

end problem_1_problem_2_problem_3_problem_4_l2706_270656


namespace largest_c_for_range_l2706_270693

theorem largest_c_for_range (f : ℝ → ℝ) (c : ℝ) : 
  (∀ x, f x = x^2 - 7*x + c) →
  (∃ x, f x = 3) →
  c ≤ 61/4 ∧ ∀ d > 61/4, ¬∃ x, x^2 - 7*x + d = 3 :=
by sorry

end largest_c_for_range_l2706_270693


namespace remainder_theorem_l2706_270615

def q (x : ℝ) : ℝ := 2*x^6 - 3*x^4 + 5*x^2 + 3

theorem remainder_theorem (q : ℝ → ℝ) (a : ℝ) :
  q a = (q 2) → q (-2) = 103 := by sorry

end remainder_theorem_l2706_270615


namespace sphere_surface_area_l2706_270655

theorem sphere_surface_area (R : ℝ) (r₁ r₂ d : ℝ) : 
  r₁ = 24 → r₂ = 15 → d = 27 → 
  R^2 = r₁^2 + x^2 → 
  R^2 = r₂^2 + (d - x)^2 → 
  4 * π * R^2 = 2500 * π :=
by sorry

end sphere_surface_area_l2706_270655


namespace point_distance_product_l2706_270662

theorem point_distance_product (y₁ y₂ : ℝ) : 
  ((-4 - 3)^2 + (y₁ - (-1))^2 = 13^2) →
  ((-4 - 3)^2 + (y₂ - (-1))^2 = 13^2) →
  y₁ ≠ y₂ →
  y₁ * y₂ = -119 := by
sorry

end point_distance_product_l2706_270662


namespace seven_digit_number_count_l2706_270604

def SevenDigitNumber := Fin 7 → Fin 7

def IsAscending (n : SevenDigitNumber) (start fin : Fin 7) : Prop :=
  ∀ i j, start ≤ i ∧ i < j ∧ j ≤ fin → n i < n j

def IsDescending (n : SevenDigitNumber) (start fin : Fin 7) : Prop :=
  ∀ i j, start ≤ i ∧ i < j ∧ j ≤ fin → n i > n j

def IsValidNumber (n : SevenDigitNumber) : Prop :=
  ∀ i j : Fin 7, i ≠ j → n i ≠ n j

theorem seven_digit_number_count :
  (∃ (S : Finset SevenDigitNumber),
    (∀ n ∈ S, IsValidNumber n ∧ IsAscending n 0 5 ∧ IsDescending n 5 6) ∧
    S.card = 6) ∧
  (∃ (T : Finset SevenDigitNumber),
    (∀ n ∈ T, IsValidNumber n ∧ IsAscending n 0 4 ∧ IsDescending n 4 6) ∧
    T.card = 15) := by sorry

end seven_digit_number_count_l2706_270604


namespace cos_300_deg_l2706_270623

theorem cos_300_deg : Real.cos (300 * π / 180) = 1 / 2 := by
  sorry

end cos_300_deg_l2706_270623


namespace square_area_with_inscribed_circle_l2706_270622

theorem square_area_with_inscribed_circle (r : ℝ) (h1 : r > 0) 
  (h2 : (r - 1)^2 + (r - 2)^2 = r^2) : (2*r)^2 = 100 := by
  sorry

end square_area_with_inscribed_circle_l2706_270622


namespace max_value_theorem_l2706_270698

theorem max_value_theorem (a b c : ℝ) (h1 : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) 
  (h2 : a + b + c = 3) (h3 : a = 1) :
  (∀ x y z : ℝ, 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z → x + y + z = 3 → x = 1 →
    (a*b)/(a+b) + (a*c)/(a+c) + (b*c)/(b+c) ≥ (x*y)/(x+y) + (x*z)/(x+z) + (y*z)/(y+z)) ∧
  (∃ b' c' : ℝ, 0 ≤ b' ∧ 0 ≤ c' ∧ a + b' + c' = 3 ∧
    (a*b')/(a+b') + (a*c')/(a+c') + (b'*c')/(b'+c') = 3/2) :=
by sorry

end max_value_theorem_l2706_270698


namespace expected_value_3X_plus_2_l2706_270663

/-- Probability distribution for random variable X -/
def prob_dist : List (ℝ × ℝ) :=
  [(1, 0.1), (2, 0.3), (3, 0.4), (4, 0.1), (5, 0.1)]

/-- Expected value of X -/
def E (X : List (ℝ × ℝ)) : ℝ :=
  (X.map (fun (x, p) => x * p)).sum

/-- Theorem: Expected value of 3X+2 is 10.4 -/
theorem expected_value_3X_plus_2 :
  E (prob_dist.map (fun (x, p) => (3 * x + 2, p))) = 10.4 := by
  sorry

end expected_value_3X_plus_2_l2706_270663


namespace sum_of_x_and_y_l2706_270678

theorem sum_of_x_and_y (x y m : ℝ) 
  (eq1 : x + m = 4) 
  (eq2 : y - 3 = m) : 
  x + y = 7 := by
sorry

end sum_of_x_and_y_l2706_270678


namespace max_t_is_one_l2706_270670

/-- The function f(x) = x^2 - ax + a - 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + a - 1

/-- The theorem stating that the maximum value of t is 1 -/
theorem max_t_is_one (t : ℝ) :
  (∀ a ∈ Set.Ioo 0 4, ∃ x₀ ∈ Set.Icc 0 2, t ≤ |f a x₀|) →
  t ≤ 1 := by
  sorry

end max_t_is_one_l2706_270670


namespace photo_arrangements_l2706_270646

/-- The number of ways to arrange n distinct objects taken r at a time -/
def A (n : ℕ) (r : ℕ) : ℕ := sorry

/-- The number of students -/
def total_students : ℕ := 7

/-- The number of students in the front row -/
def front_row : ℕ := 3

/-- The number of students in the back row -/
def back_row : ℕ := 4

/-- The number of boys -/
def num_boys : ℕ := 4

/-- The number of girls -/
def num_girls : ℕ := 3

/-- The number of spaces between boys -/
def spaces_between_boys : ℕ := 5

theorem photo_arrangements :
  (A total_students front_row * A back_row back_row = 5040) ∧
  (A front_row 1 * A back_row 1 * A (total_students - 2) (total_students - 2) = 1440) ∧
  (A (total_students - 2) (total_students - 2) * A 3 3 = 720) ∧
  (A num_boys num_boys * A spaces_between_boys num_girls = 1440) :=
sorry

end photo_arrangements_l2706_270646


namespace cookies_in_bags_l2706_270643

def total_cookies : ℕ := 75
def cookies_per_bag : ℕ := 3

theorem cookies_in_bags : total_cookies / cookies_per_bag = 25 := by
  sorry

end cookies_in_bags_l2706_270643


namespace ideal_complex_condition_l2706_270687

def is_ideal_complex (z : ℂ) : Prop :=
  z.re = -z.im

theorem ideal_complex_condition (a b : ℝ) :
  let z : ℂ := (a / (1 - 2*I)) + b*I
  is_ideal_complex z → 3*a + 5*b = 0 := by
  sorry

end ideal_complex_condition_l2706_270687


namespace intersection_condition_coincidence_condition_l2706_270649

/-- Two lines in the xy-plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : ∀ x y : ℝ, a * x + b * y + c = 0

/-- Definition of line l₁ -/
def l₁ (m : ℝ) : Line where
  a := 1
  b := m
  c := 6
  eq := by sorry

/-- Definition of line l₂ -/
def l₂ (m : ℝ) : Line where
  a := m - 2
  b := 3
  c := 2 * m
  eq := by sorry

/-- Two lines intersect if they are not parallel -/
def intersect (l₁ l₂ : Line) : Prop :=
  l₁.a * l₂.b ≠ l₁.b * l₂.a

/-- Two lines coincide if they are equivalent -/
def coincide (l₁ l₂ : Line) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ l₁.a = k * l₂.a ∧ l₁.b = k * l₂.b ∧ l₁.c = k * l₂.c

/-- Main theorem for intersection -/
theorem intersection_condition (m : ℝ) :
  intersect (l₁ m) (l₂ m) ↔ m ≠ -1 ∧ m ≠ 3 := by sorry

/-- Main theorem for coincidence -/
theorem coincidence_condition (m : ℝ) :
  coincide (l₁ m) (l₂ m) ↔ m = 3 := by sorry

end intersection_condition_coincidence_condition_l2706_270649


namespace exponent_division_rule_l2706_270645

theorem exponent_division_rule (a b : ℝ) (m : ℤ) 
  (ha : a > 0) (hb : b ≠ 0) : 
  (b / a) ^ m = a ^ (-m) * b ^ m := by sorry

end exponent_division_rule_l2706_270645


namespace cranberry_juice_can_ounces_l2706_270606

/-- Given a can of cranberry juice that sells for 84 cents with a unit cost of 7.0 cents per ounce,
    prove that the can contains 12 ounces of juice. -/
theorem cranberry_juice_can_ounces :
  ∀ (total_cost unit_cost : ℚ),
    total_cost = 84 →
    unit_cost = 7 →
    total_cost / unit_cost = 12 := by
sorry

end cranberry_juice_can_ounces_l2706_270606


namespace potion_kit_cost_is_18_silver_l2706_270611

/-- Represents the cost of items in Harry's purchase --/
structure PurchaseCost where
  spellbookCost : ℕ
  owlCost : ℕ
  totalSilver : ℕ
  silverToGold : ℕ

/-- Calculates the cost of each potion kit in silver --/
def potionKitCost (p : PurchaseCost) : ℕ :=
  let totalGold := p.totalSilver / p.silverToGold
  let spellbooksTotalCost := 5 * p.spellbookCost
  let remainingGold := totalGold - spellbooksTotalCost - p.owlCost
  let potionKitGold := remainingGold / 3
  potionKitGold * p.silverToGold

/-- Theorem stating that each potion kit costs 18 silvers --/
theorem potion_kit_cost_is_18_silver (p : PurchaseCost) 
  (h1 : p.spellbookCost = 5)
  (h2 : p.owlCost = 28)
  (h3 : p.totalSilver = 537)
  (h4 : p.silverToGold = 9) : 
  potionKitCost p = 18 := by
  sorry


end potion_kit_cost_is_18_silver_l2706_270611


namespace coin_distribution_l2706_270692

theorem coin_distribution (a b c d e : ℚ) : 
  a + b + c + d + e = 5 →  -- Total 5 coins
  a + b = c + d + e →  -- Sum condition
  b - a = c - b ∧ c - b = d - c ∧ d - c = e - d →  -- Arithmetic sequence
  e = 2/3 := by sorry

end coin_distribution_l2706_270692


namespace train_length_l2706_270691

/-- Given a train that crosses a signal post in 40 seconds and takes 600 seconds
    to cross a 9000-meter long bridge at a constant speed, prove that the length
    of the train is 642.857142857... meters. -/
theorem train_length (signal_time : ℝ) (bridge_time : ℝ) (bridge_length : ℝ) :
  signal_time = 40 →
  bridge_time = 600 →
  bridge_length = 9000 →
  ∃ (train_length : ℝ), train_length = 360000 / 560 :=
by sorry

end train_length_l2706_270691


namespace circle_properties_l2706_270671

/-- A circle passing through two points with its center on a line -/
def circle_equation (x y : ℝ) : Prop :=
  (x - 1/2)^2 + (y - 1)^2 = 5/4

theorem circle_properties :
  (circle_equation 1 0) ∧
  (circle_equation 0 2) ∧
  (∃ (a : ℝ), circle_equation a (2*a)) :=
sorry

end circle_properties_l2706_270671


namespace calculate_3Y5_l2706_270632

-- Define the operation Y
def Y (a b : ℤ) : ℤ := b + 12 * a - a^3

-- Theorem to prove
theorem calculate_3Y5 : Y 3 5 = 14 := by
  sorry

end calculate_3Y5_l2706_270632


namespace pencils_per_group_l2706_270696

theorem pencils_per_group (total_pencils : ℕ) (num_groups : ℕ) 
  (h1 : total_pencils = 154) (h2 : num_groups = 14) :
  total_pencils / num_groups = 11 := by
sorry

end pencils_per_group_l2706_270696


namespace cube_volume_doubling_l2706_270624

theorem cube_volume_doubling (original_volume : ℝ) (new_volume : ℝ) : 
  original_volume = 216 →
  new_volume = (2 * original_volume^(1/3))^3 →
  new_volume = 1728 :=
by sorry

end cube_volume_doubling_l2706_270624


namespace total_weight_calculation_l2706_270689

theorem total_weight_calculation (a b c d : ℝ) 
  (h1 : a + b = 156)
  (h2 : c + d = 195)
  (h3 : a + c = 174)
  (h4 : b + d = 186) :
  a + b + c + d = 355.5 := by
sorry

end total_weight_calculation_l2706_270689


namespace polynomial_consecutive_integers_l2706_270641

/-- A polynomial P(n) = (n^5 + a) / b takes integer values for three consecutive integers
    if and only if (a, b) = (k, 1) or (11k ± 1, 11) for some integer k. -/
theorem polynomial_consecutive_integers (a b : ℕ+) :
  (∃ n : ℤ, ∀ i ∈ ({0, 1, 2} : Set ℤ), ∃ k : ℤ, (n + i)^5 + a = b * k) ↔
  (∃ k : ℤ, (a = k ∧ b = 1) ∨ (a = 11 * k + 1 ∧ b = 11) ∨ (a = 11 * k - 1 ∧ b = 11)) :=
sorry

end polynomial_consecutive_integers_l2706_270641


namespace arithmetic_sequence_common_difference_l2706_270669

/-- Given an arithmetic sequence {a_n} with sum of first n terms S_n,
    prove that if S_6 = 3S_2 + 24, then the common difference d = 2 -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) -- a_n is the nth term of the arithmetic sequence
  (S : ℕ → ℝ) -- S_n is the sum of the first n terms
  (h_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1) -- condition for arithmetic sequence
  (h_sum : ∀ n, S n = n * (a 1 + a n) / 2) -- formula for sum of arithmetic sequence
  (h_given : S 6 = 3 * S 2 + 24) -- given condition
  : a 2 - a 1 = 2 := by sorry

end arithmetic_sequence_common_difference_l2706_270669


namespace sin_cos_identity_l2706_270652

theorem sin_cos_identity : 
  Real.sin (21 * π / 180) * Real.cos (81 * π / 180) - 
  Real.sin (69 * π / 180) * Real.cos (9 * π / 180) = 
  -Real.sqrt 3 / 2 := by
  sorry

end sin_cos_identity_l2706_270652


namespace solve_system_l2706_270647

theorem solve_system (x y : ℝ) 
  (eq1 : 3 * x - 2 * y = 18) 
  (eq2 : x + y = 20) : 
  y = 8.4 := by sorry

end solve_system_l2706_270647


namespace fractional_equation_range_l2706_270620

theorem fractional_equation_range (x m : ℝ) : 
  (x / (x - 1) = m / (2 * x - 2) + 3) →
  (x ≥ 0) →
  (m ≤ 6 ∧ m ≠ 2) :=
by sorry

end fractional_equation_range_l2706_270620


namespace exists_m_eq_power_plus_n_l2706_270642

/-- n(m) denotes the number of factors of 2 in m! -/
def n (m : ℕ+) : ℕ := sorry

/-- Theorem: There exists a natural number m > 2006^2006 such that m = 3^2006 + n(m) -/
theorem exists_m_eq_power_plus_n : ∃ m : ℕ+, 
  (m : ℕ) > 2006^2006 ∧ (m : ℕ) = 3^2006 + n m := by
  sorry

end exists_m_eq_power_plus_n_l2706_270642


namespace binary_10001000_to_octal_l2706_270636

def binary_to_octal (b : ℕ) : ℕ := sorry

theorem binary_10001000_to_octal :
  binary_to_octal 0b10001000 = 0o210 := by sorry

end binary_10001000_to_octal_l2706_270636


namespace jamie_ball_collection_l2706_270625

/-- Calculates the total number of balls Jamie has after all transactions --/
def total_balls (initial_red : ℕ) (blue_multiplier : ℕ) (lost_red : ℕ) (yellow_multiplier : ℕ) : ℕ :=
  let initial_blue := initial_red * blue_multiplier
  let remaining_red := initial_red - lost_red
  let bought_yellow := lost_red * yellow_multiplier
  remaining_red + initial_blue + bought_yellow

theorem jamie_ball_collection :
  total_balls 16 2 6 3 = 60 := by
  sorry

end jamie_ball_collection_l2706_270625


namespace coin_and_die_probability_l2706_270658

theorem coin_and_die_probability :
  let coin_outcomes : ℕ := 2  -- Fair coin has 2 possible outcomes
  let die_outcomes : ℕ := 8   -- Eight-sided die has 8 possible outcomes
  let total_outcomes : ℕ := coin_outcomes * die_outcomes
  let successful_outcomes : ℕ := 1  -- Only one successful outcome (Tails and 5)
  
  (successful_outcomes : ℚ) / total_outcomes = 1 / 16 :=
by
  sorry


end coin_and_die_probability_l2706_270658


namespace right_triangle_ratio_l2706_270618

theorem right_triangle_ratio (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (right_triangle : a^2 + b^2 = c^2) (leg_relation : a = 2*b) :
  (a + b) / c = 3 * Real.sqrt 5 / 5 := by
  sorry

end right_triangle_ratio_l2706_270618


namespace arithmetic_expression_equality_l2706_270677

theorem arithmetic_expression_equality : 61 + 5 * 12 / (180 / 3) = 62 := by
  sorry

end arithmetic_expression_equality_l2706_270677


namespace sum_of_fractions_l2706_270688

theorem sum_of_fractions (p q r : ℝ) 
  (h1 : p + q + r = 5) 
  (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) : 
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end sum_of_fractions_l2706_270688


namespace first_hole_depth_l2706_270673

/-- Represents the depth of a hole dug by workers -/
structure HoleDigging where
  workers : ℕ
  hours : ℕ
  depth : ℝ

theorem first_hole_depth 
  (hole1 : HoleDigging)
  (hole2 : HoleDigging)
  (h1 : hole1.workers = 45)
  (h2 : hole1.hours = 8)
  (h3 : hole2.workers = 110)
  (h4 : hole2.hours = 6)
  (h5 : hole2.depth = 55)
  (h6 : hole1.workers * hole1.hours * hole2.depth = hole2.workers * hole2.hours * hole1.depth) :
  hole1.depth = 30 := by
sorry


end first_hole_depth_l2706_270673


namespace parabola_line_intersection_ratio_l2706_270628

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Represents a line passing through a point with a given angle -/
structure Line where
  point : Point
  angle : ℝ

/-- Theorem: For a parabola y^2 = 2px and a line passing through its focus
    with an inclination angle of 60°, intersecting the parabola at points A and B
    in the first and fourth quadrants respectively, the ratio |AF| / |BF| = 3 -/
theorem parabola_line_intersection_ratio 
  (para : Parabola) 
  (l : Line) 
  (A B : Point) 
  (h1 : l.point = Point.mk (para.p / 2) 0)  -- Focus of the parabola
  (h2 : l.angle = π / 3)  -- 60° in radians
  (h3 : A.x > 0 ∧ A.y > 0)  -- A in first quadrant
  (h4 : B.x > 0 ∧ B.y < 0)  -- B in fourth quadrant
  (h5 : A.y^2 = 2 * para.p * A.x)  -- A on parabola
  (h6 : B.y^2 = 2 * para.p * B.x)  -- B on parabola
  : abs (A.x - para.p / 2) / abs (B.x - para.p / 2) = 3 := by
  sorry

end parabola_line_intersection_ratio_l2706_270628


namespace five_dice_not_same_probability_l2706_270635

theorem five_dice_not_same_probability :
  let n_faces : ℕ := 6
  let n_dice : ℕ := 5
  let total_outcomes : ℕ := n_faces ^ n_dice
  let same_number_outcomes : ℕ := n_faces
  (1 : ℚ) - (same_number_outcomes : ℚ) / total_outcomes = 1295 / 1296 :=
by sorry

end five_dice_not_same_probability_l2706_270635


namespace small_paintings_completed_l2706_270666

/-- Represents the number of ounces of paint used for a large canvas --/
def paint_per_large_canvas : ℕ := 3

/-- Represents the number of ounces of paint used for a small canvas --/
def paint_per_small_canvas : ℕ := 2

/-- Represents the number of large paintings completed --/
def large_paintings_completed : ℕ := 3

/-- Represents the total amount of paint used in ounces --/
def total_paint_used : ℕ := 17

/-- Proves that the number of small paintings completed is 4 --/
theorem small_paintings_completed :
  (total_paint_used - large_paintings_completed * paint_per_large_canvas) / paint_per_small_canvas = 4 :=
by sorry

end small_paintings_completed_l2706_270666


namespace jameson_medal_count_l2706_270613

/-- Represents the number of medals Jameson has in each category -/
structure MedalCount where
  track : Nat
  swimming : Nat
  badminton : Nat

/-- Calculates the total number of medals -/
def totalMedals (medals : MedalCount) : Nat :=
  medals.track + medals.swimming + medals.badminton

/-- Theorem: Jameson's total medal count is 20 -/
theorem jameson_medal_count :
  ∀ (medals : MedalCount),
    medals.track = 5 →
    medals.swimming = 2 * medals.track →
    medals.badminton = 5 →
    totalMedals medals = 20 := by
  sorry


end jameson_medal_count_l2706_270613


namespace tangent_line_at_P_l2706_270602

-- Define the curve
def f (x : ℝ) : ℝ := x^2 - 2*x

-- Define the point of tangency
def P : ℝ × ℝ := (2, 0)

-- Define the proposed tangent line
def tangentLine (x : ℝ) : ℝ := 2*x - 4

theorem tangent_line_at_P :
  (∀ x : ℝ, HasDerivAt f (tangentLine P.1) P.1) ∧
  f P.1 = tangentLine P.1 :=
sorry

end tangent_line_at_P_l2706_270602


namespace cabbage_price_calculation_l2706_270668

/-- Represents the price of the cabbage in Janet's grocery purchase. -/
def cabbage_price : ℝ := sorry

/-- Represents Janet's total grocery budget. -/
def total_budget : ℝ := sorry

theorem cabbage_price_calculation :
  let broccoli_cost : ℝ := 3 * 4
  let oranges_cost : ℝ := 3 * 0.75
  let bacon_cost : ℝ := 1 * 3
  let chicken_cost : ℝ := 2 * 3
  let meat_cost : ℝ := bacon_cost + chicken_cost
  let known_items_cost : ℝ := broccoli_cost + oranges_cost + bacon_cost + chicken_cost
  meat_cost = 0.33 * total_budget ∧
  cabbage_price = total_budget - known_items_cost →
  cabbage_price = 4.02 := by sorry

end cabbage_price_calculation_l2706_270668


namespace vasyas_numbers_l2706_270690

theorem vasyas_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = (1 : ℝ) / 2 ∧ y = -(1 : ℝ) := by
  sorry

end vasyas_numbers_l2706_270690


namespace optimal_advertising_plan_l2706_270629

/-- Represents the advertising plan for a company --/
structure AdvertisingPlan where
  timeA : ℝ  -- Time allocated to TV station A in minutes
  timeB : ℝ  -- Time allocated to TV station B in minutes

/-- Calculates the total advertising time for a given plan --/
def totalTime (plan : AdvertisingPlan) : ℝ :=
  plan.timeA + plan.timeB

/-- Calculates the total advertising cost for a given plan --/
def totalCost (plan : AdvertisingPlan) : ℝ :=
  500 * plan.timeA + 200 * plan.timeB

/-- Calculates the total revenue for a given plan --/
def totalRevenue (plan : AdvertisingPlan) : ℝ :=
  0.3 * plan.timeA + 0.2 * plan.timeB

/-- Theorem stating the optimal advertising plan and maximum revenue --/
theorem optimal_advertising_plan :
  ∃ (plan : AdvertisingPlan),
    totalTime plan ≤ 300 ∧
    totalCost plan ≤ 90000 ∧
    plan.timeA = 100 ∧
    plan.timeB = 200 ∧
    totalRevenue plan = 70 ∧
    ∀ (other : AdvertisingPlan),
      totalTime other ≤ 300 →
      totalCost other ≤ 90000 →
      totalRevenue other ≤ totalRevenue plan :=
by
  sorry


end optimal_advertising_plan_l2706_270629


namespace inverse_proportion_ratio_l2706_270608

theorem inverse_proportion_ratio {x₁ x₂ y₁ y₂ : ℝ} (hx : x₁ ≠ 0 ∧ x₂ ≠ 0) (hy : y₁ ≠ 0 ∧ y₂ ≠ 0)
  (h_inv_prop : ∃ k : ℝ, k ≠ 0 ∧ x₁ * y₁ = k ∧ x₂ * y₂ = k)
  (h_x_ratio : x₁ / x₂ = 3 / 5) :
  y₁ / y₂ = 5 / 3 := by
  sorry

end inverse_proportion_ratio_l2706_270608


namespace bird_migration_difference_l2706_270674

/-- The number of bird families that flew to Africa -/
def africa_birds : ℕ := 42

/-- The number of bird families that flew to Asia -/
def asia_birds : ℕ := 31

/-- The number of bird families living near the mountain -/
def mountain_birds : ℕ := 8

/-- Theorem stating the difference between bird families that flew to Africa and Asia -/
theorem bird_migration_difference : africa_birds - asia_birds = 11 := by
  sorry

end bird_migration_difference_l2706_270674


namespace shortest_altitude_right_triangle_l2706_270686

theorem shortest_altitude_right_triangle :
  ∀ (a b c h : ℝ),
  a = 8 ∧ b = 15 ∧ c = 17 →
  a^2 + b^2 = c^2 →
  h = (2 * (a * b) / 2) / c →
  h = 120 / 17 ∧ 
  (∀ h' : ℝ, (h' = a ∨ h' = b ∨ h' = h) → h ≤ h') :=
by sorry

end shortest_altitude_right_triangle_l2706_270686


namespace binary_multiplication_example_l2706_270638

/-- Represents a binary number as a list of booleans, where true represents 1 and false represents 0 -/
def BinaryNumber := List Bool

/-- Converts a binary number to its decimal representation -/
def binary_to_decimal (b : BinaryNumber) : Nat :=
  b.foldl (fun acc digit => 2 * acc + if digit then 1 else 0) 0

/-- Multiplies two binary numbers -/
def binary_multiply (a b : BinaryNumber) : BinaryNumber :=
  sorry

theorem binary_multiplication_example :
  let a : BinaryNumber := [true, true, false, true]  -- 1101₂
  let b : BinaryNumber := [true, true, true]         -- 111₂
  let result : BinaryNumber := [true, false, false, true, true, true, true]  -- 1001111₂
  binary_multiply a b = result :=
sorry

end binary_multiplication_example_l2706_270638
