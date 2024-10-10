import Mathlib

namespace arithmetic_sequence_properties_l476_47662

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum of first n terms
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * d) / 2

/-- Properties of the specific arithmetic sequence -/
def SpecificSequence (seq : ArithmeticSequence) : Prop :=
  seq.S 5 < seq.S 6 ∧ seq.S 6 = seq.S 7 ∧ seq.S 7 > seq.S 8

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) 
  (h : SpecificSequence seq) : 
  seq.d < 0 ∧ 
  seq.S 9 < seq.S 5 ∧ 
  seq.a 7 = 0 ∧ 
  (∀ n, seq.S n ≤ seq.S 6 ∧ seq.S n ≤ seq.S 7) :=
by sorry

end arithmetic_sequence_properties_l476_47662


namespace square_pentagon_side_ratio_l476_47649

theorem square_pentagon_side_ratio :
  ∀ (s_s s_p : ℝ),
  s_s > 0 → s_p > 0 →
  s_s^2 = (5 * s_p^2 * (Real.sqrt 5 + 1)) / 8 →
  s_p / s_s = Real.sqrt (8 / (5 * (Real.sqrt 5 + 1))) :=
by sorry

end square_pentagon_side_ratio_l476_47649


namespace bisection_method_result_l476_47694

def f (x : ℝ) := x^3 - 3*x + 1

theorem bisection_method_result :
  ∃ x₀ : ℝ, f x₀ = 0 ∧ 0 < x₀ ∧ x₀ < 1 →
  ∃ a b : ℝ, 1/4 < a ∧ a < x₀ ∧ x₀ < b ∧ b < 1/2 ∧
    f a * f b < 0 ∧
    ∀ c ∈ Set.Ioo (0 : ℝ) 1, f c * f (1/2) ≤ 0 → c ≤ 1/2 ∧
    ∀ d ∈ Set.Ioo (0 : ℝ) (1/2), f d * f (1/4) ≤ 0 → 1/4 ≤ d :=
by sorry

end bisection_method_result_l476_47694


namespace integer_condition_l476_47657

theorem integer_condition (x : ℝ) : 
  (∀ x : ℤ, ∃ y : ℤ, 2 * (x : ℝ) + 1 = y) ∧ 
  (∃ x : ℝ, ∃ y : ℤ, 2 * x + 1 = y ∧ ¬∃ z : ℤ, x = z) :=
sorry

end integer_condition_l476_47657


namespace soccer_league_games_l476_47601

/-- Calculate the number of games in a soccer league --/
theorem soccer_league_games (n : ℕ) (h : n = 11) : n * (n - 1) / 2 = 55 := by
  sorry

#check soccer_league_games

end soccer_league_games_l476_47601


namespace arithmetic_sequence_sum_l476_47673

/-- An arithmetic sequence with given first two terms -/
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  a 1 = -1 ∧ a 2 = 2 ∧ ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (h : arithmetic_sequence a) :
  a 4 + a 5 = 19 := by
  sorry

end arithmetic_sequence_sum_l476_47673


namespace arithmetic_sequence_sum_l476_47615

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 2 = 3 →
  a 1 + a 6 = 12 →
  a 7 + a 8 + a 9 = 45 :=
by
  sorry

end arithmetic_sequence_sum_l476_47615


namespace benzene_formation_enthalpy_l476_47648

-- Define the substances
def C : Type := Unit
def H₂ : Type := Unit
def C₂H₂ : Type := Unit
def C₆H₆ : Type := Unit

-- Define the states
inductive State
| Gas
| Liquid
| Graphite

-- Define a reaction
structure Reaction :=
  (reactants : List (Type × State × ℕ))
  (products : List (Type × State × ℕ))
  (heat_effect : ℝ)

-- Given reactions
def reaction1 : Reaction :=
  ⟨[(C₂H₂, State.Gas, 1)], [(C, State.Graphite, 2), (H₂, State.Gas, 1)], 226.7⟩

def reaction2 : Reaction :=
  ⟨[(C₂H₂, State.Gas, 3)], [(C₆H₆, State.Liquid, 1)], 631.1⟩

def reaction3 : Reaction :=
  ⟨[(C₆H₆, State.Liquid, 1)], [(C₆H₆, State.Liquid, 1)], -33.9⟩

-- Standard enthalpy of formation
def standard_enthalpy_of_formation (substance : Type) (state : State) : ℝ := sorry

-- Theorem statement
theorem benzene_formation_enthalpy :
  standard_enthalpy_of_formation C₆H₆ State.Liquid = -82.9 :=
sorry

end benzene_formation_enthalpy_l476_47648


namespace davids_crunches_l476_47640

theorem davids_crunches (zachary_crunches : ℕ) (david_less_crunches : ℕ) 
  (h1 : zachary_crunches = 17)
  (h2 : david_less_crunches = 13) :
  zachary_crunches - david_less_crunches = 4 := by
  sorry

end davids_crunches_l476_47640


namespace social_practice_arrangements_l476_47632

def number_of_teachers : Nat := 2
def number_of_students : Nat := 6
def teachers_per_group : Nat := 1
def students_per_group : Nat := 3
def number_of_groups : Nat := 2

theorem social_practice_arrangements :
  (number_of_teachers.choose teachers_per_group) *
  (number_of_students.choose students_per_group) = 40 := by
  sorry

end social_practice_arrangements_l476_47632


namespace geometric_distribution_sum_to_one_l476_47636

/-- The probability mass function for a geometric distribution -/
def geometric_pmf (p : ℝ) (m : ℕ) : ℝ := (1 - p) ^ (m - 1) * p

/-- Theorem: The sum of probabilities for a geometric distribution equals 1 -/
theorem geometric_distribution_sum_to_one (p : ℝ) (hp : 0 < p) (hp' : p < 1) :
  ∑' m : ℕ, geometric_pmf p m = 1 := by
  sorry

end geometric_distribution_sum_to_one_l476_47636


namespace student_count_l476_47616

theorem student_count (rank_right rank_left : ℕ) 
  (h1 : rank_right = 13) 
  (h2 : rank_left = 8) : 
  rank_right + rank_left - 1 = 20 := by
  sorry

end student_count_l476_47616


namespace equal_charges_at_300_minutes_l476_47666

/-- Represents a mobile phone plan -/
structure PhonePlan where
  monthly_fee : ℝ
  call_rate : ℝ

/-- Calculates the monthly bill for a given plan and call duration -/
def monthly_bill (plan : PhonePlan) (duration : ℝ) : ℝ :=
  plan.monthly_fee + plan.call_rate * duration

/-- The Unicom company's phone plans -/
def plan_a : PhonePlan := { monthly_fee := 15, call_rate := 0.1 }
def plan_b : PhonePlan := { monthly_fee := 0, call_rate := 0.15 }

theorem equal_charges_at_300_minutes : 
  ∃ (duration : ℝ), duration = 300 ∧ 
    monthly_bill plan_a duration = monthly_bill plan_b duration := by
  sorry

end equal_charges_at_300_minutes_l476_47666


namespace intersection_point_on_both_lines_unique_intersection_point_l476_47623

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (-3/10, -9/10)

/-- The first line equation -/
def line1 (x y : ℚ) : Prop := y = 3 * x

/-- The second line equation -/
def line2 (x y : ℚ) : Prop := y + 3 = -7 * x

/-- Theorem stating that the intersection_point satisfies both line equations -/
theorem intersection_point_on_both_lines :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y := by sorry

/-- Theorem stating that the intersection_point is the unique solution -/
theorem unique_intersection_point :
  ∀ (x y : ℚ), line1 x y ∧ line2 x y → (x, y) = intersection_point := by sorry

end intersection_point_on_both_lines_unique_intersection_point_l476_47623


namespace polynomial_roots_product_l476_47653

theorem polynomial_roots_product (d e : ℤ) : 
  (∀ r : ℝ, r^2 = r + 1 → r^6 = d*r + e) → d*e = 40 := by
  sorry

end polynomial_roots_product_l476_47653


namespace product_of_brackets_l476_47678

def bracket_a (a : ℕ) : ℕ := a^2 + 3

def bracket_b (b : ℕ) : ℕ := 2*b - 4

theorem product_of_brackets (p q : ℕ) (h1 : p = 7) (h2 : q = 10) :
  bracket_a p * bracket_b q = 832 := by
  sorry

end product_of_brackets_l476_47678


namespace max_abs_sum_on_circle_l476_47614

theorem max_abs_sum_on_circle (x y : ℝ) (h : x^2 + y^2 = 16) :
  ∃ (max : ℝ), (∀ a b : ℝ, a^2 + b^2 = 16 → |a| + |b| ≤ max) ∧ max = 4 * Real.sqrt 2 := by
  sorry

end max_abs_sum_on_circle_l476_47614


namespace probability_three_dice_divisible_by_10_l476_47655

-- Define a die as having 6 faces
def die_faces : ℕ := 6

-- Define a function to check if a number is divisible by 2
def divisible_by_2 (n : ℕ) : Prop := n % 2 = 0

-- Define a function to check if a number is divisible by 5
def divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

-- Define a function to check if a product is divisible by 10
def product_divisible_by_10 (a b c : ℕ) : Prop :=
  divisible_by_2 (a * b * c) ∧ divisible_by_5 (a * b * c)

-- Define the probability of the event
def probability_divisible_by_10 : ℚ :=
  (144 : ℚ) / (die_faces ^ 3 : ℚ)

-- State the theorem
theorem probability_three_dice_divisible_by_10 :
  probability_divisible_by_10 = 2 / 3 := by sorry

end probability_three_dice_divisible_by_10_l476_47655


namespace smallest_three_digit_multiple_of_17_l476_47687

theorem smallest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 17 ∣ n → n ≥ 102 :=
by sorry

end smallest_three_digit_multiple_of_17_l476_47687


namespace min_value_expression_l476_47689

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x = y) :
  ∃ (min : ℝ), min = 0 ∧ ∀ (a b : ℝ), a > 0 → b > 0 → a = b →
    (a + 1/b) * (a + 1/b - 2) + (b + 1/a) * (b + 1/a - 2) ≥ min :=
by sorry

end min_value_expression_l476_47689


namespace dilation_rotation_composition_l476_47664

def dilation_matrix (k : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![k, 0; 0, k]

def rotation_90_ccw : Matrix (Fin 2) (Fin 2) ℝ :=
  !![0, -1; 1, 0]

def combined_transformation : Matrix (Fin 2) (Fin 2) ℝ :=
  !![0, -4; 4, 0]

theorem dilation_rotation_composition :
  combined_transformation = rotation_90_ccw * dilation_matrix 4 := by
  sorry

end dilation_rotation_composition_l476_47664


namespace problem_solution_l476_47635

def f (x a : ℝ) := |x - a| * x + |x - 2| * (x - a)

theorem problem_solution :
  (∀ x, f x 1 < 0 ↔ x ∈ Set.Iio 1) ∧
  (∀ a, (∀ x, x ∈ Set.Iio 1 → f x a < 0) ↔ a ∈ Set.Ici 1) := by
  sorry

end problem_solution_l476_47635


namespace divisibility_property_l476_47671

theorem divisibility_property (q : ℕ) (h1 : Nat.Prime q) (h2 : q % 2 = 1) :
  ∃ k : ℤ, (q + 1 : ℤ)^(q - 1) - 1 = k * q :=
sorry

end divisibility_property_l476_47671


namespace horner_third_intermediate_value_l476_47680

def horner_polynomial (a : List ℚ) (x : ℚ) : ℚ :=
  a.foldl (fun acc coeff => acc * x + coeff) 0

def horner_intermediate (a : List ℚ) (x : ℚ) (n : ℕ) : ℚ :=
  (a.take (n + 1)).foldl (fun acc coeff => acc * x + coeff) 0

theorem horner_third_intermediate_value :
  let f (x : ℚ) := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64
  let coeffs := [1, -12, 60, -160, 240, -192, 64]
  let x := 2
  horner_intermediate coeffs x 3 = -80 := by sorry

end horner_third_intermediate_value_l476_47680


namespace exists_divisible_by_sum_of_digits_remainder_at_most_15_exists_number_for_remainder_l476_47676

/-- Represents a two-digit number -/
def TwoDigitNumber := { n : ℕ // 10 ≤ n ∧ n ≤ 99 }

/-- Sum of digits of a two-digit number -/
def sumOfDigits (n : TwoDigitNumber) : ℕ :=
  n.val / 10 + n.val % 10

/-- Theorem 1: There exists a two-digit number divisible by the sum of its digits -/
theorem exists_divisible_by_sum_of_digits :
  ∃ n : TwoDigitNumber, n.val % (sumOfDigits n) = 0 :=
sorry

/-- Theorem 2: The remainder when a two-digit number is divided by the sum of its digits is at most 15 -/
theorem remainder_at_most_15 (n : TwoDigitNumber) :
  n.val % (sumOfDigits n) ≤ 15 :=
sorry

/-- Theorem 3: For any remainder r ≤ 12, there exists a two-digit number that produces that remainder -/
theorem exists_number_for_remainder (r : ℕ) (h : r ≤ 12) :
  ∃ n : TwoDigitNumber, n.val % (sumOfDigits n) = r :=
sorry

end exists_divisible_by_sum_of_digits_remainder_at_most_15_exists_number_for_remainder_l476_47676


namespace right_triangle_perimeter_l476_47637

theorem right_triangle_perimeter (area : ℝ) (leg1 : ℝ) (leg2 : ℝ) (hypotenuse : ℝ) :
  area = (1/2) * leg1 * leg2 →
  leg1 = 30 →
  area = 150 →
  leg2^2 + leg1^2 = hypotenuse^2 →
  leg1 + leg2 + hypotenuse = 40 + 10 * Real.sqrt 10 := by
  sorry

end right_triangle_perimeter_l476_47637


namespace triangle_centroid_property_l476_47674

open Real

variable (A B C Q G' : ℝ × ℝ)

def is_inside_triangle (P A B C : ℝ × ℝ) : Prop := sorry

def distance_squared (P Q : ℝ × ℝ) : ℝ := 
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

theorem triangle_centroid_property :
  is_inside_triangle G' A B C →
  G' = ((1/4 : ℝ) • A + (1/4 : ℝ) • B + (1/2 : ℝ) • C) →
  distance_squared Q A + distance_squared Q B + distance_squared Q C = 
  4 * distance_squared Q G' + distance_squared G' A + distance_squared G' B + distance_squared G' C :=
by sorry

end triangle_centroid_property_l476_47674


namespace sequence_sum_l476_47688

theorem sequence_sum (n x y : ℝ) : 
  (3 + 16 + 33 + (n + 1) + x + y) / 6 = 25 → n + x + y = 97 := by
  sorry

end sequence_sum_l476_47688


namespace total_berries_l476_47631

/-- The number of berries each person has -/
structure Berries where
  stacy : ℕ
  steve : ℕ
  skylar : ℕ

/-- The conditions of the berry distribution -/
def berry_conditions (b : Berries) : Prop :=
  b.stacy = 4 * b.steve ∧ 
  b.steve = 2 * b.skylar ∧ 
  b.stacy = 800

/-- The theorem stating the total number of berries -/
theorem total_berries (b : Berries) (h : berry_conditions b) : 
  b.stacy + b.steve + b.skylar = 1100 := by
  sorry

end total_berries_l476_47631


namespace rhombus_diagonal_sum_l476_47620

/-- A rhombus with specific properties -/
structure Rhombus where
  longer_diagonal : ℝ
  shorter_diagonal : ℝ
  area : ℝ
  diagonal_diff : longer_diagonal - shorter_diagonal = 4
  area_eq : area = 6
  positive_diagonals : longer_diagonal > 0 ∧ shorter_diagonal > 0

/-- The sum of diagonals in a rhombus with given properties is 8 -/
theorem rhombus_diagonal_sum (r : Rhombus) : r.longer_diagonal + r.shorter_diagonal = 8 := by
  sorry

end rhombus_diagonal_sum_l476_47620


namespace independent_x_implies_result_l476_47610

theorem independent_x_implies_result (m n : ℝ) :
  (∀ x y : ℝ, ∃ k : ℝ, (m*x^2 + 3*x - y) - (4*x^2 - (2*n + 3)*x + 3*y - 2) = k) →
  (m - n) + |m*n| = 19 := by
sorry

end independent_x_implies_result_l476_47610


namespace quadratic_root_sum_l476_47644

theorem quadratic_root_sum (p r : ℝ) (hp : p > 0) (hr : r > 0) :
  (∀ x : ℝ, x^2 - p*x + r = 0 → ∃ y : ℝ, y^2 - p*y + r = 0 ∧ x + y = 8) →
  r = 8 := by
sorry

end quadratic_root_sum_l476_47644


namespace matthew_crackers_l476_47641

/-- The number of crackers Matthew had initially -/
def initial_crackers : ℕ := 55

/-- The number of cakes Matthew had -/
def cakes : ℕ := 34

/-- The number of friends Matthew gave crackers and cakes to -/
def friends : ℕ := 11

/-- The number of crackers each person ate -/
def crackers_eaten_per_person : ℕ := 2

theorem matthew_crackers :
  (cakes / friends = initial_crackers / friends) ∧
  (friends * crackers_eaten_per_person + friends * (cakes / friends) = initial_crackers) :=
by sorry

end matthew_crackers_l476_47641


namespace sampling_theorem_l476_47625

/-- Staff distribution in departments A and B -/
structure StaffDistribution where
  maleA : ℕ
  femaleA : ℕ
  maleB : ℕ
  femaleB : ℕ

/-- Sampling method for selecting staff members -/
inductive SamplingMethod
  | Stratified : SamplingMethod

/-- Result of the sampling process -/
structure SamplingResult where
  fromA : ℕ
  fromB : ℕ
  totalSelected : ℕ

/-- Theorem stating the probability of selecting at least one female from A
    and the expectation of the number of males selected -/
theorem sampling_theorem (sd : StaffDistribution) (sm : SamplingMethod) (sr : SamplingResult) :
  sd.maleA = 6 ∧ sd.femaleA = 4 ∧ sd.maleB = 3 ∧ sd.femaleB = 2 ∧
  sm = SamplingMethod.Stratified ∧
  sr.fromA = 2 ∧ sr.fromB = 1 ∧ sr.totalSelected = 3 →
  (ProbabilityAtLeastOneFemaleFromA = 2/3) ∧
  (ExpectationOfMalesSelected = 9/5) := by
  sorry

end sampling_theorem_l476_47625


namespace sequence_property_l476_47658

theorem sequence_property (a : ℕ → ℝ) (h_nonzero : ∀ n, a n ≠ 0) 
  (h_arith1 : a 2 - a 1 = a 3 - a 2)
  (h_geom : a 3 / a 2 = a 4 / a 3)
  (h_arith2 : 1 / a 4 - 1 / a 3 = 1 / a 5 - 1 / a 4) :
  a 3 ^ 2 = a 1 * a 5 := by
  sorry

end sequence_property_l476_47658


namespace divide_number_with_percentage_condition_l476_47629

theorem divide_number_with_percentage_condition : 
  ∃ (x : ℝ), 
    x + (80 - x) = 80 ∧ 
    0.3 * x = 0.2 * (80 - x) + 10 ∧ 
    min x (80 - x) = 28 := by
  sorry

end divide_number_with_percentage_condition_l476_47629


namespace basil_daytime_cookies_l476_47683

/-- Represents the number of cookies Basil gets per day -/
structure BasilCookies where
  morning : ℚ
  evening : ℚ
  daytime : ℕ

/-- Represents the cookie box information -/
structure CookieBox where
  cookies_per_box : ℕ
  boxes_needed : ℕ
  days_lasting : ℕ

theorem basil_daytime_cookies 
  (basil_cookies : BasilCookies)
  (cookie_box : CookieBox)
  (h1 : basil_cookies.morning = 1/2)
  (h2 : basil_cookies.evening = 1/2)
  (h3 : cookie_box.cookies_per_box = 45)
  (h4 : cookie_box.boxes_needed = 2)
  (h5 : cookie_box.days_lasting = 30) :
  basil_cookies.daytime = 2 :=
sorry

end basil_daytime_cookies_l476_47683


namespace B_equals_roster_l476_47669

def A : Set Int := {-2, 2, 3, 4}

def B : Set Int := {x | ∃ t ∈ A, x = t^2}

theorem B_equals_roster : B = {4, 9, 16} := by sorry

end B_equals_roster_l476_47669


namespace trapezoid_properties_l476_47696

/-- Represents a trapezoid ABCD with AB and CD as parallel bases (AB < CD) -/
structure Trapezoid where
  AD : ℝ  -- Length of larger base
  BC : ℝ  -- Length of smaller base
  AB : ℝ  -- Length of shorter leg
  midline : ℝ  -- Length of midline
  midpoint_segment : ℝ  -- Length of segment connecting midpoints of bases
  angle1 : ℝ  -- Angle at one end of larger base (in degrees)
  angle2 : ℝ  -- Angle at other end of larger base (in degrees)

/-- Theorem stating the properties of the specific trapezoid in the problem -/
theorem trapezoid_properties (T : Trapezoid) 
  (h1 : T.midline = 5)
  (h2 : T.midpoint_segment = 3)
  (h3 : T.angle1 = 30)
  (h4 : T.angle2 = 60) :
  T.AD = 8 ∧ T.BC = 2 ∧ T.AB = 3 := by
  sorry

end trapezoid_properties_l476_47696


namespace chocolates_per_student_l476_47622

theorem chocolates_per_student (n : ℕ) :
  (∀ (students : ℕ), students * n < 288 → students ≤ 9) ∧
  (∀ (students : ℕ), students * n > 300 → students ≥ 10) →
  n = 31 := by
  sorry

end chocolates_per_student_l476_47622


namespace min_value_of_ab_l476_47656

theorem min_value_of_ab (a b : ℝ) (ha : a > 1) (hb : b > 1) 
  (h_seq : (1/4 * Real.log a) * (Real.log b) = (1/4)^2) : 
  (∀ x y : ℝ, x > 1 → y > 1 → (1/4 * Real.log x) * (Real.log y) = (1/4)^2 → a * b ≤ x * y) ∧ 
  a * b = Real.exp 1 := by
sorry

end min_value_of_ab_l476_47656


namespace lansing_elementary_students_l476_47619

/-- The number of elementary schools in Lansing -/
def num_schools : ℕ := 25

/-- The number of students in each elementary school in Lansing -/
def students_per_school : ℕ := 247

/-- The total number of elementary students in Lansing -/
def total_students : ℕ := num_schools * students_per_school

theorem lansing_elementary_students :
  total_students = 6175 :=
sorry

end lansing_elementary_students_l476_47619


namespace no_integer_root_l476_47643

theorem no_integer_root (P : ℤ → ℤ) (h_poly : ∀ x y : ℤ, (x - y) ∣ (P x - P y)) 
  (h1 : P 1 = 10) (h_neg1 : P (-1) = 22) (h0 : P 0 = 4) :
  ∀ r : ℤ, P r ≠ 0 :=
sorry

end no_integer_root_l476_47643


namespace triangle_side_length_l476_47692

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  b = 7 →
  c = 6 →
  Real.cos (B - C) = 15/16 →
  a = 5 * Real.sqrt 3 :=
by
  sorry

end triangle_side_length_l476_47692


namespace function_is_constant_one_l476_47642

/-- A function satisfying the given conditions is constant and equal to 1 -/
theorem function_is_constant_one (f : ℝ → ℝ) 
  (h1 : ∀ x, f (x + 2) = 2 - f x) 
  (h2 : ∀ x, f (x + 3) ≥ f x) : 
  ∀ x, f x = 1 := by sorry

end function_is_constant_one_l476_47642


namespace big_bottles_sold_percentage_l476_47663

theorem big_bottles_sold_percentage
  (small_initial : Nat)
  (big_initial : Nat)
  (small_sold_percent : Rat)
  (total_remaining : Nat)
  (h1 : small_initial = 6000)
  (h2 : big_initial = 15000)
  (h3 : small_sold_percent = 12 / 100)
  (h4 : total_remaining = 18180)
  : (big_initial - (total_remaining - (small_initial - small_initial * small_sold_percent))) / big_initial = 14 / 100 := by
  sorry

end big_bottles_sold_percentage_l476_47663


namespace abc_subtraction_problem_l476_47659

theorem abc_subtraction_problem (a b c : ℕ) : 
  a < 10 ∧ b < 10 ∧ c < 10 ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (100 * b + 10 * c + a) - (100 * a + 10 * b + c) = 682 →
  a = 3 ∧ b = 7 ∧ c = 5 := by
sorry

end abc_subtraction_problem_l476_47659


namespace rhombus_area_l476_47654

/-- The area of a rhombus with side length 5 cm and an interior angle of 60 degrees is 12.5√3 square centimeters. -/
theorem rhombus_area (s : ℝ) (θ : ℝ) (h1 : s = 5) (h2 : θ = π / 3) :
  s * s * Real.sin θ = 25 * Real.sqrt 3 / 2 := by
  sorry

end rhombus_area_l476_47654


namespace matrix_equality_zero_l476_47621

open Matrix

theorem matrix_equality_zero (n : ℕ) (A B : Matrix (Fin n) (Fin n) ℝ) 
  (h1 : A * B = B) (h2 : det (A - 1) ≠ 0) : B = 0 := by
  sorry

end matrix_equality_zero_l476_47621


namespace inverse_f_at_142_l476_47691

def f (x : ℝ) : ℝ := 5 * x^3 + 7

theorem inverse_f_at_142 : f⁻¹ 142 = 3 := by
  sorry

end inverse_f_at_142_l476_47691


namespace a_eq_one_sufficient_not_necessary_l476_47693

-- Define the complex number z(a)
def z (a : ℝ) : ℂ := (a - 1) * (a + 2) + (a + 3) * Complex.I

-- Define what it means for a complex number to be purely imaginary
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0

-- State the theorem
theorem a_eq_one_sufficient_not_necessary :
  (∀ a : ℝ, a = 1 → isPurelyImaginary (z a)) ∧
  ¬(∀ a : ℝ, isPurelyImaginary (z a) → a = 1) :=
by sorry

end a_eq_one_sufficient_not_necessary_l476_47693


namespace stamps_needed_tara_stamps_problem_l476_47627

theorem stamps_needed (current_stamps : ℕ) (stamps_per_sheet : ℕ) : ℕ :=
  stamps_per_sheet - (current_stamps % stamps_per_sheet)

theorem tara_stamps_problem : stamps_needed 38 9 = 7 := by
  sorry

end stamps_needed_tara_stamps_problem_l476_47627


namespace multiplication_puzzle_l476_47646

/-- Represents a digit in the set {1, 2, 3, 4, 5, 6} -/
def Digit := Fin 6

/-- Represents the multiplication problem AB × C = DEF -/
def IsValidMultiplication (a b c d e f : Digit) : Prop :=
  (a.val + 1) * 10 + (b.val + 1) = (d.val + 1) * 100 + (e.val + 1) * 10 + (f.val + 1)

/-- All digits are distinct -/
def AreDistinct (a b c d e f : Digit) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f

theorem multiplication_puzzle :
  ∀ (a b c d e f : Digit),
    IsValidMultiplication a b c d e f →
    AreDistinct a b c d e f →
    c.val = 2 :=
by sorry

end multiplication_puzzle_l476_47646


namespace arithmetic_sequence_fourth_term_l476_47618

/-- Given an arithmetic sequence where the sum of the third and fifth terms is 12,
    prove that the fourth term is 6. -/
theorem arithmetic_sequence_fourth_term (a d : ℝ) 
  (h : (a + 2*d) + (a + 4*d) = 12) : a + 3*d = 6 := by
  sorry

end arithmetic_sequence_fourth_term_l476_47618


namespace marble_probability_l476_47670

theorem marble_probability (a b c : ℕ) : 
  a + b + c = 97 →
  (a * (a - 1) + b * (b - 1) + c * (c - 1)) / (97 * 96) = 5 / 12 →
  (a^2 + b^2 + c^2) / 97^2 = 41 / 97 := by
sorry

end marble_probability_l476_47670


namespace alice_weekly_distance_l476_47605

/-- The distance Alice walks to school each day -/
def distance_to_school : ℕ := 10

/-- The distance Alice walks back home each day -/
def distance_from_school : ℕ := 12

/-- The number of days Alice walks to and from school in a week -/
def days_per_week : ℕ := 5

/-- Theorem: Alice's total walking distance for the week is 110 miles -/
theorem alice_weekly_distance :
  (distance_to_school + distance_from_school) * days_per_week = 110 := by
  sorry

end alice_weekly_distance_l476_47605


namespace angle_sets_relation_l476_47639

-- Define the sets A, B, and C
def A : Set ℝ := {θ | ∃ k : ℤ, 2 * k * Real.pi < θ ∧ θ < 2 * k * Real.pi + Real.pi / 2}
def B : Set ℝ := {θ | 0 < θ ∧ θ < Real.pi / 2}
def C : Set ℝ := {θ | θ < Real.pi / 2}

-- State the theorem
theorem angle_sets_relation : B ∪ C = C := by
  sorry

end angle_sets_relation_l476_47639


namespace line_plane_perpendicularity_l476_47612

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (m n : Line) (β : Plane) 
  (h1 : m ≠ n) 
  (h2 : parallel m n) 
  (h3 : perpendicular m β) : 
  perpendicular n β :=
sorry

end line_plane_perpendicularity_l476_47612


namespace largest_ball_radius_is_four_l476_47638

/-- Represents a torus in 3D space --/
structure Torus where
  center : ℝ × ℝ × ℝ
  radius : ℝ
  inner_radius : ℝ
  outer_radius : ℝ

/-- Represents a spherical ball in 3D space --/
structure Ball where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- The torus described in the problem --/
def problem_torus : Torus :=
  { center := (4, 0, 1)
    radius := 1
    inner_radius := 3
    outer_radius := 5 }

/-- 
  Given a torus sitting on the xy-plane, returns the radius of the largest
  spherical ball that can be placed on top of the center of the torus and
  still touch the horizontal plane
--/
def largest_ball_radius (t : Torus) : ℝ :=
  sorry

theorem largest_ball_radius_is_four :
  largest_ball_radius problem_torus = 4 := by
  sorry

end largest_ball_radius_is_four_l476_47638


namespace simplify_fraction_product_l476_47679

theorem simplify_fraction_product : (240 / 12) * (5 / 150) * (12 / 3) = 8 / 3 := by
  sorry

end simplify_fraction_product_l476_47679


namespace equilateral_triangle_area_ratio_l476_47628

theorem equilateral_triangle_area_ratio :
  ∀ s : ℝ,
  s > 0 →
  let small_triangle_area := (s^2 * Real.sqrt 3) / 4
  let large_triangle_side := 3 * s
  let large_triangle_area := (large_triangle_side^2 * Real.sqrt 3) / 4
  (3 * small_triangle_area) / large_triangle_area = 1 / 3 :=
by
  sorry

end equilateral_triangle_area_ratio_l476_47628


namespace part1_part2_l476_47661

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 5

-- Part 1
theorem part1 (a : ℝ) (h1 : a > 1) 
  (h2 : ∀ x, x ∈ Set.Icc 1 a ↔ f a x ∈ Set.Icc 1 a) : 
  a = 2 := by sorry

-- Part 2
theorem part2 (a : ℝ) (h1 : a > 1) 
  (h2 : ∀ x₁ x₂, x₁ ∈ Set.Icc 1 (a+1) → x₂ ∈ Set.Icc 1 (a+1) → 
    |f a x₁ - f a x₂| ≤ 4) : 
  1 < a ∧ a ≤ 3 := by sorry

end part1_part2_l476_47661


namespace valid_selling_price_l476_47686

/-- Represents the business model for Oleg's water heater production --/
structure WaterHeaterBusiness where
  units_sold : ℕ
  variable_cost : ℕ
  fixed_cost : ℕ
  desired_profit : ℕ
  selling_price : ℕ

/-- Calculates the total revenue given the number of units sold and the selling price --/
def total_revenue (b : WaterHeaterBusiness) : ℕ :=
  b.units_sold * b.selling_price

/-- Calculates the total cost given the number of units sold, variable cost, and fixed cost --/
def total_cost (b : WaterHeaterBusiness) : ℕ :=
  b.units_sold * b.variable_cost + b.fixed_cost

/-- Checks if the selling price satisfies the business requirements --/
def is_valid_price (b : WaterHeaterBusiness) : Prop :=
  total_revenue b ≥ total_cost b + b.desired_profit

/-- Theorem stating that the calculated selling price satisfies the business requirements --/
theorem valid_selling_price :
  let b : WaterHeaterBusiness := {
    units_sold := 5000,
    variable_cost := 800,
    fixed_cost := 1000000,
    desired_profit := 1500000,
    selling_price := 1300
  }
  is_valid_price b ∧ b.selling_price ≥ 0 :=
by sorry


end valid_selling_price_l476_47686


namespace daffodil_cost_is_65_cents_l476_47617

/-- Represents the cost of bulbs and garden space --/
structure BulbGarden where
  totalSpace : ℕ
  crocusCost : ℚ
  totalBudget : ℚ
  crocusCount : ℕ

/-- Calculates the cost of each daffodil bulb --/
def daffodilCost (g : BulbGarden) : ℚ :=
  let crocusTotalCost := g.crocusCost * g.crocusCount
  let remainingBudget := g.totalBudget - crocusTotalCost
  let daffodilCount := g.totalSpace - g.crocusCount
  remainingBudget / daffodilCount

/-- Theorem stating the cost of each daffodil bulb --/
theorem daffodil_cost_is_65_cents (g : BulbGarden)
  (h1 : g.totalSpace = 55)
  (h2 : g.crocusCost = 35/100)
  (h3 : g.totalBudget = 2915/100)
  (h4 : g.crocusCount = 22) :
  daffodilCost g = 65/100 := by
  sorry

#eval daffodilCost { totalSpace := 55, crocusCost := 35/100, totalBudget := 2915/100, crocusCount := 22 }

end daffodil_cost_is_65_cents_l476_47617


namespace correct_mean_calculation_l476_47695

theorem correct_mean_calculation (n : ℕ) (incorrect_mean : ℚ) (incorrect_value : ℚ) (correct_value : ℚ) :
  n = 30 ∧ 
  incorrect_mean = 150 ∧ 
  incorrect_value = 135 ∧ 
  correct_value = 165 →
  (n * incorrect_mean - incorrect_value + correct_value) / n = 151 := by sorry

end correct_mean_calculation_l476_47695


namespace avg_f_value_l476_47633

/-- A function that counts the number of multiples of p in the partial sums of a permutation -/
def f (p : ℕ) (π : Fin p → Fin p) : ℕ := sorry

/-- The average value of f over all permutations -/
def avg_f (p : ℕ) : ℚ := sorry

theorem avg_f_value (p : ℕ) (h : p.Prime) (h2 : p > 2) :
  avg_f p = 2 - 1 / p := by sorry

end avg_f_value_l476_47633


namespace matrix_from_eigenvectors_l476_47682

theorem matrix_from_eigenvectors (A : Matrix (Fin 2) (Fin 2) ℝ) :
  (A.mulVec (![1, -3]) = ![-1, 3]) →
  (A.mulVec (![1, 1]) = ![3, 3]) →
  A = !![2, 1; 3, 0] := by
sorry

end matrix_from_eigenvectors_l476_47682


namespace cubic_root_increasing_l476_47606

theorem cubic_root_increasing : 
  ∀ (x y : ℝ), x < y → (x ^ (1/3 : ℝ)) < (y ^ (1/3 : ℝ)) := by
  sorry

end cubic_root_increasing_l476_47606


namespace logan_desired_amount_left_l476_47645

/-- Represents Logan's financial situation and goal --/
structure LoganFinances where
  current_income : ℕ
  rent_expense : ℕ
  groceries_expense : ℕ
  gas_expense : ℕ
  income_increase : ℕ

/-- Calculates the desired amount left each year for Logan --/
def desired_amount_left (f : LoganFinances) : ℕ :=
  (f.current_income + f.income_increase) - (f.rent_expense + f.groceries_expense + f.gas_expense)

/-- Theorem stating the desired amount left each year for Logan --/
theorem logan_desired_amount_left :
  let f : LoganFinances := {
    current_income := 65000,
    rent_expense := 20000,
    groceries_expense := 5000,
    gas_expense := 8000,
    income_increase := 10000
  }
  desired_amount_left f = 42000 := by
  sorry


end logan_desired_amount_left_l476_47645


namespace total_net_buried_bones_l476_47626

/-- Represents the types of bones Barkley receives --/
inductive BoneType
  | A
  | B
  | C

/-- Represents Barkley's bone statistics over 5 months --/
structure BoneStats where
  received : Nat
  buried : Nat
  eaten : Nat

/-- Calculates the net buried bones for a given BoneStats --/
def netBuried (stats : BoneStats) : Nat :=
  stats.buried - stats.eaten

/-- Defines Barkley's bone statistics for each type over 5 months --/
def barkleyStats : BoneType → BoneStats
  | BoneType.A => { received := 50, buried := 30, eaten := 3 }
  | BoneType.B => { received := 30, buried := 16, eaten := 2 }
  | BoneType.C => { received := 20, buried := 10, eaten := 2 }

/-- Theorem: The total net number of buried bones after 5 months is 49 --/
theorem total_net_buried_bones :
  (netBuried (barkleyStats BoneType.A) +
   netBuried (barkleyStats BoneType.B) +
   netBuried (barkleyStats BoneType.C)) = 49 := by
  sorry


end total_net_buried_bones_l476_47626


namespace triangle_area_l476_47603

/-- The area of a triangle ABC with given side lengths and angle -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  b = 2 →
  c = 2 * Real.sqrt 2 →
  C = π / 4 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 + 1 :=
by sorry

end triangle_area_l476_47603


namespace treasure_chest_rubies_l476_47604

theorem treasure_chest_rubies (total_gems diamonds : ℕ) 
  (h1 : total_gems = 5155)
  (h2 : diamonds = 45) :
  total_gems - diamonds = 5110 := by
  sorry

end treasure_chest_rubies_l476_47604


namespace prime_divisor_of_mersenne_number_l476_47608

theorem prime_divisor_of_mersenne_number (p q : ℕ) : 
  Prime p → Prime q → q ∣ (2^p - 1) → p ∣ (q - 1) := by sorry

end prime_divisor_of_mersenne_number_l476_47608


namespace christmas_to_birthday_ratio_l476_47698

def total_presents : ℕ := 90
def christmas_presents : ℕ := 60

theorem christmas_to_birthday_ratio :
  (christmas_presents : ℚ) / (total_presents - christmas_presents : ℚ) = 2 := by
  sorry

end christmas_to_birthday_ratio_l476_47698


namespace cube_root_equation_solution_l476_47634

theorem cube_root_equation_solution :
  ∃ y : ℝ, (5 - 2 / y)^(1/3 : ℝ) = -3 ↔ y = 1/16 :=
by sorry

end cube_root_equation_solution_l476_47634


namespace c_share_is_40_l476_47607

/-- Represents the share distribution among three parties -/
structure ShareDistribution where
  total : ℝ
  b_share : ℝ
  c_share : ℝ
  d_share : ℝ

/-- The condition for the share distribution -/
def valid_distribution (s : ShareDistribution) : Prop :=
  s.total = 80 ∧
  s.c_share = 1.5 * s.b_share ∧
  s.d_share = 0.5 * s.b_share ∧
  s.total = s.b_share + s.c_share + s.d_share

/-- Theorem stating that under the given conditions, c's share is 40 rupees -/
theorem c_share_is_40 (s : ShareDistribution) (h : valid_distribution s) : s.c_share = 40 := by
  sorry

end c_share_is_40_l476_47607


namespace specific_hyperbola_real_axis_length_l476_47675

/-- A hyperbola with given asymptotes and passing through a specific point -/
structure Hyperbola where
  -- The hyperbola passes through this point
  point : ℝ × ℝ
  -- The equations of the asymptotes
  asymptote1 : ℝ → ℝ → ℝ
  asymptote2 : ℝ → ℝ → ℝ

/-- The length of the real axis of a hyperbola -/
def realAxisLength (h : Hyperbola) : ℝ :=
  sorry

/-- Theorem stating the length of the real axis of the specific hyperbola -/
theorem specific_hyperbola_real_axis_length :
  ∃ (h : Hyperbola),
    h.point = (5, -2) ∧
    h.asymptote1 = (λ x y => x - 2*y) ∧
    h.asymptote2 = (λ x y => x + 2*y) ∧
    realAxisLength h = 6 :=
  sorry

end specific_hyperbola_real_axis_length_l476_47675


namespace minsu_age_proof_l476_47677

/-- Minsu's current age in years -/
def minsu_current_age : ℕ := 8

/-- Years in the future when Minsu's age will be four times his current age -/
def years_in_future : ℕ := 24

/-- Theorem stating that Minsu's current age is 8, given the condition -/
theorem minsu_age_proof :
  minsu_current_age = 8 ∧
  minsu_current_age + years_in_future = 4 * minsu_current_age :=
by sorry

end minsu_age_proof_l476_47677


namespace f_5_equals_207_l476_47609

def f (n : ℕ) : ℕ := n^3 + 2*n^2 + 3*n + 17

theorem f_5_equals_207 : f 5 = 207 := by
  sorry

end f_5_equals_207_l476_47609


namespace parabola_through_point_l476_47652

-- Define a parabola
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  -- ax² + by² = c

-- Define the point (1, -2)
def point : ℝ × ℝ := (1, -2)

-- Theorem statement
theorem parabola_through_point :
  ∃ (p1 p2 : Parabola),
    (p1.a = 0 ∧ p1.b = 1 ∧ p1.c = 4 ∧ p1.a * point.1^2 + p1.b * point.2^2 = p1.c) ∨
    (p2.a = 1 ∧ p2.b = -1/2 ∧ p2.c = 0 ∧ p2.a * point.1^2 + p2.b * point.2 = p2.c) :=
by sorry

end parabola_through_point_l476_47652


namespace monic_quartic_problem_l476_47672

-- Define a monic quartic polynomial
def monicQuartic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, f x = x^4 + a*x^3 + b*x^2 + c*x + f 0

-- State the theorem
theorem monic_quartic_problem (f : ℝ → ℝ) 
  (h_monic : monicQuartic f)
  (h_neg2 : f (-2) = 0)
  (h_3 : f 3 = -9)
  (h_neg4 : f (-4) = -16)
  (h_5 : f 5 = -25) :
  f 0 = 0 := by sorry

end monic_quartic_problem_l476_47672


namespace cheaper_feed_cost_l476_47613

/-- Proves that the cost of the cheaper feed is $0.18 per pound given the problem conditions --/
theorem cheaper_feed_cost (total_mix : ℝ) (mix_price : ℝ) (expensive_price : ℝ) (cheaper_amount : ℝ) 
  (h1 : total_mix = 35)
  (h2 : mix_price = 0.36)
  (h3 : expensive_price = 0.53)
  (h4 : cheaper_amount = 17) :
  ∃ (cheaper_price : ℝ), 
    cheaper_price * cheaper_amount + expensive_price * (total_mix - cheaper_amount) = mix_price * total_mix ∧ 
    cheaper_price = 0.18 := by
  sorry

end cheaper_feed_cost_l476_47613


namespace divisibility_of_polynomial_l476_47630

theorem divisibility_of_polynomial (x : ℕ) (h_prime : Nat.Prime x) (h_gt3 : x > 3) :
  (∃ n : ℤ, x = 3 * n + 1 ∧ (x^6 - x^3 - x^2 + x) % 12 = 0) ∨
  (∃ n : ℤ, x = 3 * n - 1 ∧ (x^6 - x^3 - x^2 + x) % 36 = 0) :=
by sorry

end divisibility_of_polynomial_l476_47630


namespace exists_common_element_l476_47699

/-- A collection of 1978 sets, each containing 40 elements -/
def SetCollection := Fin 1978 → Finset (Fin (1978 * 40))

/-- The property that any two sets in the collection have exactly one common element -/
def OneCommonElement (collection : SetCollection) : Prop :=
  ∀ i j, i ≠ j → (collection i ∩ collection j).card = 1

/-- The theorem stating that there exists an element in all sets of the collection -/
theorem exists_common_element (collection : SetCollection)
  (h1 : ∀ i, (collection i).card = 40)
  (h2 : OneCommonElement collection) :
  ∃ x, ∀ i, x ∈ collection i :=
sorry

end exists_common_element_l476_47699


namespace ellipse_focal_length_l476_47667

/-- Given an ellipse with equation x^2/23 + y^2/32 = 1, its focal length is 6. -/
theorem ellipse_focal_length : ∀ (x y : ℝ), x^2/23 + y^2/32 = 1 → ∃ (c : ℝ), c = 3 ∧ 2*c = 6 :=
by
  sorry

end ellipse_focal_length_l476_47667


namespace square_sum_of_xy_l476_47660

theorem square_sum_of_xy (x y : ℕ+) 
  (h1 : x * y + x + y = 83)
  (h2 : x^2 * y + x * y^2 = 1056) : 
  x^2 + y^2 = 458 := by
  sorry

end square_sum_of_xy_l476_47660


namespace downstream_speed_l476_47611

theorem downstream_speed (upstream_speed : ℝ) (average_speed : ℝ) (downstream_speed : ℝ) :
  upstream_speed = 6 →
  average_speed = 60 / 11 →
  (1 / upstream_speed + 1 / downstream_speed) / 2 = 1 / average_speed →
  downstream_speed = 5 := by
sorry

end downstream_speed_l476_47611


namespace M_is_hypersquared_l476_47624

def n : ℕ := 1000

def is_perfect_square (x : ℕ) : Prop := ∃ y : ℕ, x = y^2

def first_n_digits (x : ℕ) (n : ℕ) : ℕ := x / 10^n

def last_n_digits (x : ℕ) (n : ℕ) : ℕ := x % 10^n

def is_hypersquared (x : ℕ) : Prop :=
  ∃ n : ℕ,
    (x ≥ 10^(2*n - 1)) ∧
    (x < 10^(2*n)) ∧
    is_perfect_square x ∧
    is_perfect_square (first_n_digits x n) ∧
    is_perfect_square (last_n_digits x n) ∧
    (last_n_digits x n ≥ 10^(n-1))

def M : ℕ := ((5 * 10^(n-1) - 1) * 10^n + (10^n - 1))^2

theorem M_is_hypersquared : is_hypersquared M := by
  sorry

end M_is_hypersquared_l476_47624


namespace new_average_height_l476_47681

/-- Calculates the new average height of a class after some students leave and others join. -/
theorem new_average_height
  (initial_size : ℕ)
  (initial_avg : ℝ)
  (left_size : ℕ)
  (left_avg : ℝ)
  (joined_size : ℕ)
  (joined_avg : ℝ)
  (h_initial_size : initial_size = 35)
  (h_initial_avg : initial_avg = 180)
  (h_left_size : left_size = 7)
  (h_left_avg : left_avg = 120)
  (h_joined_size : joined_size = 7)
  (h_joined_avg : joined_avg = 140)
  : (initial_size * initial_avg - left_size * left_avg + joined_size * joined_avg) / initial_size = 184 := by
  sorry

end new_average_height_l476_47681


namespace divisibility_condition_l476_47647

theorem divisibility_condition (n : ℕ) : 
  (∃ k : ℤ, (7 * n + 5 : ℤ) = k * (3 * n - 1)) ↔ n = 1 ∨ n = 4 := by
  sorry

end divisibility_condition_l476_47647


namespace triangle_shape_not_unique_l476_47650

/-- A triangle with sides a, b, c and angles A, B, C --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The shape of a triangle is not uniquely determined by the product of two sides and the angle between them --/
theorem triangle_shape_not_unique (p : ℝ) (γ : ℝ) :
  ∃ (t1 t2 : Triangle), t1 ≠ t2 ∧ t1.a * t1.b = p ∧ t1.C = γ ∧ t2.a * t2.b = p ∧ t2.C = γ :=
sorry

end triangle_shape_not_unique_l476_47650


namespace line_equation_forms_l476_47600

/-- Given a line with equation (3x-2)/4 - (2y-1)/2 = 1, prove its various forms -/
theorem line_equation_forms (x y : ℝ) :
  (3*x - 2)/4 - (2*y - 1)/2 = 1 →
  (3*x - 8*y - 2 = 0) ∧
  (y = (3/8)*x - 1/4) ∧
  (x/(2/3) + y/(-1/4) = 1) ∧
  ((3/Real.sqrt 73)*x - (8/Real.sqrt 73)*y - (2/Real.sqrt 73) = 0) := by
  sorry

end line_equation_forms_l476_47600


namespace stapler_equation_l476_47651

theorem stapler_equation (sheets : ℕ) (time_first time_combined : ℝ) (time_second : ℝ) :
  sheets > 0 ∧ time_first > 0 ∧ time_combined > 0 ∧ time_second > 0 →
  (sheets / time_first + sheets / time_second = sheets / time_combined) ↔
  (1 / time_first + 1 / time_second = 1 / time_combined) :=
by sorry

end stapler_equation_l476_47651


namespace original_profit_percentage_l476_47685

/-- Calculates the profit percentage given the cost price and selling price -/
def profitPercentage (costPrice sellingPrice : ℚ) : ℚ :=
  (sellingPrice - costPrice) / costPrice * 100

theorem original_profit_percentage
  (costPrice : ℚ)
  (sellingPrice : ℚ)
  (h1 : costPrice = 80)
  (h2 : profitPercentage (costPrice * (1 - 0.2)) (sellingPrice - 16.8) = 30) :
  profitPercentage costPrice sellingPrice = 25 := by
  sorry

end original_profit_percentage_l476_47685


namespace initial_pennies_equation_l476_47690

/-- Given that Sam spent some pennies and has some left, prove that his initial number of pennies
    is equal to the sum of pennies spent and pennies left. -/
theorem initial_pennies_equation (initial spent left : ℕ) : 
  spent = 93 → left = 5 → initial = spent + left := by sorry

end initial_pennies_equation_l476_47690


namespace simplify_expression_l476_47668

theorem simplify_expression (a : ℝ) : 2*a + 1 - (1 - a) = 3*a := by
  sorry

end simplify_expression_l476_47668


namespace farm_animals_percentage_l476_47684

theorem farm_animals_percentage (cows ducks pigs : ℕ) : 
  cows = 20 →
  pigs = (ducks + cows) / 5 →
  cows + ducks + pigs = 60 →
  (ducks - cows : ℚ) / cows * 100 = 50 := by
sorry

end farm_animals_percentage_l476_47684


namespace treasure_chest_age_conversion_l476_47697

/-- Converts a number from base 8 to base 10 --/
def base8ToBase10 (hundreds : Nat) (tens : Nat) (ones : Nat) : Nat :=
  hundreds * 8^2 + tens * 8^1 + ones * 8^0

/-- The age of the treasure chest in base 8 --/
def treasureChestAgeBase8 : Nat × Nat × Nat := (3, 4, 7)

theorem treasure_chest_age_conversion :
  let (h, t, o) := treasureChestAgeBase8
  base8ToBase10 h t o = 231 := by
  sorry

end treasure_chest_age_conversion_l476_47697


namespace probability_three_digit_l476_47665

def set_start : ℕ := 60
def set_end : ℕ := 1000

def three_digit_start : ℕ := 100
def three_digit_end : ℕ := 999

def total_numbers : ℕ := set_end - set_start + 1
def three_digit_numbers : ℕ := three_digit_end - (three_digit_start - 1)

theorem probability_three_digit :
  (three_digit_numbers : ℚ) / total_numbers = 901 / 941 := by sorry

end probability_three_digit_l476_47665


namespace equidistant_function_property_l476_47602

open Complex

theorem equidistant_function_property (a b : ℝ) :
  (∀ z : ℂ, abs ((a + b * I) * z - z) = abs ((a + b * I) * z - I)) →
  abs (a + b * I) = 10 →
  b^2 = (1/4 : ℝ) := by
sorry

end equidistant_function_property_l476_47602
