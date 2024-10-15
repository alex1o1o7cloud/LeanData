import Mathlib

namespace NUMINAMATH_CALUDE_correct_sampling_methods_l2555_255566

-- Define the type for sampling methods
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic
  | DrawingLots
  | RandomNumber

-- Define the set of correct sampling methods
def correctSamplingMethods : Set SamplingMethod :=
  {SamplingMethod.SimpleRandom, SamplingMethod.Stratified, SamplingMethod.Systematic}

-- Define the property of being a valid sampling method
def isValidSamplingMethod (method : SamplingMethod) : Prop :=
  method ∈ correctSamplingMethods

-- State the conditions
axiom simple_random_valid : isValidSamplingMethod SamplingMethod.SimpleRandom
axiom stratified_valid : isValidSamplingMethod SamplingMethod.Stratified
axiom systematic_valid : isValidSamplingMethod SamplingMethod.Systematic
axiom drawing_lots_is_simple_random : SamplingMethod.DrawingLots = SamplingMethod.SimpleRandom
axiom random_number_is_simple_random : SamplingMethod.RandomNumber = SamplingMethod.SimpleRandom

-- State the theorem
theorem correct_sampling_methods :
  correctSamplingMethods = {SamplingMethod.SimpleRandom, SamplingMethod.Stratified, SamplingMethod.Systematic} :=
by sorry

end NUMINAMATH_CALUDE_correct_sampling_methods_l2555_255566


namespace NUMINAMATH_CALUDE_f_negative_when_x_greater_half_l2555_255527

/-- The linear function f(x) = -2x + 1 -/
def f (x : ℝ) : ℝ := -2 * x + 1

/-- When x > 1/2, f(x) < 0 -/
theorem f_negative_when_x_greater_half : ∀ x : ℝ, x > (1/2) → f x < 0 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_when_x_greater_half_l2555_255527


namespace NUMINAMATH_CALUDE_smallest_integer_result_l2555_255593

def expression : List ℕ := [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

def is_valid_bracketing (b : List (List ℕ)) : Prop :=
  b.join = expression ∧ ∀ l ∈ b, l.length > 0

def evaluate_bracketing (b : List (List ℕ)) : ℚ :=
  b.foldl (λ acc l => acc / l.foldl (λ x y => x / y) 1) 1

def is_integer_result (b : List (List ℕ)) : Prop :=
  ∃ n : ℤ, (evaluate_bracketing b).num = n * (evaluate_bracketing b).den

theorem smallest_integer_result :
  ∃ b : List (List ℕ),
    is_valid_bracketing b ∧
    is_integer_result b ∧
    evaluate_bracketing b = 7 ∧
    ∀ b' : List (List ℕ),
      is_valid_bracketing b' →
      is_integer_result b' →
      evaluate_bracketing b' ≥ 7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_result_l2555_255593


namespace NUMINAMATH_CALUDE_power_three_mod_five_l2555_255557

theorem power_three_mod_five : 3^244 % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_three_mod_five_l2555_255557


namespace NUMINAMATH_CALUDE_max_abs_sum_on_circle_l2555_255551

theorem max_abs_sum_on_circle (x y : ℝ) (h : x^2 + y^2 = 4) :
  ∃ (max : ℝ), (∀ a b : ℝ, a^2 + b^2 = 4 → |a| + |b| ≤ max) ∧ (|x| + |y| = max) :=
by sorry

end NUMINAMATH_CALUDE_max_abs_sum_on_circle_l2555_255551


namespace NUMINAMATH_CALUDE_range_of_a_for_always_positive_quadratic_l2555_255524

theorem range_of_a_for_always_positive_quadratic :
  (∀ a : ℝ, (∀ x : ℝ, x^2 + 2*a*x + a > 0) ↔ (0 < a ∧ a < 1)) := by sorry

end NUMINAMATH_CALUDE_range_of_a_for_always_positive_quadratic_l2555_255524


namespace NUMINAMATH_CALUDE_eight_pow_zero_minus_log_100_l2555_255503

theorem eight_pow_zero_minus_log_100 : 8^0 - Real.log 100 / Real.log 10 = -1 := by
  sorry

end NUMINAMATH_CALUDE_eight_pow_zero_minus_log_100_l2555_255503


namespace NUMINAMATH_CALUDE_negative_sqrt_four_equals_negative_two_l2555_255532

theorem negative_sqrt_four_equals_negative_two : -Real.sqrt 4 = -2 := by
  sorry

end NUMINAMATH_CALUDE_negative_sqrt_four_equals_negative_two_l2555_255532


namespace NUMINAMATH_CALUDE_excircle_lengths_sum_gt_semiperimeter_l2555_255561

/-- Given a triangle with sides a, b, and c, and semi-perimeter p,
    BB' and CC' are specific lengths related to the excircles of the triangle. -/
def triangle_excircle_lengths (a b c : ℝ) (p : ℝ) (BB' CC' : ℝ) : Prop :=
  p = (a + b + c) / 2 ∧ BB' = p - a ∧ CC' = p - b

/-- The sum of BB' and CC' is greater than the semi-perimeter p for any triangle. -/
theorem excircle_lengths_sum_gt_semiperimeter 
  {a b c p BB' CC' : ℝ} 
  (h : triangle_excircle_lengths a b c p BB' CC') :
  BB' + CC' > p :=
sorry

end NUMINAMATH_CALUDE_excircle_lengths_sum_gt_semiperimeter_l2555_255561


namespace NUMINAMATH_CALUDE_not_passed_implies_not_all_correct_l2555_255517

-- Define the universe of discourse
variable (Student : Type)

-- Define predicates
variable (passed : Student → Prop)
variable (answered_all_correctly : Student → Prop)

-- State the given condition
variable (h : ∀ s : Student, answered_all_correctly s → passed s)

-- Theorem to prove
theorem not_passed_implies_not_all_correct (s : Student) :
  ¬(passed s) → ¬(answered_all_correctly s) :=
by
  sorry


end NUMINAMATH_CALUDE_not_passed_implies_not_all_correct_l2555_255517


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2555_255537

def geometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometricSequence a → a 5 = 2 → a 1 * a 2 * a 3 * a 7 * a 8 * a 9 = 64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2555_255537


namespace NUMINAMATH_CALUDE_dist_to_left_focus_is_ten_l2555_255506

/-- The hyperbola type -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  (pos_a : a > 0)
  (pos_b : b > 0)

/-- A point on the hyperbola -/
structure PointOnHyperbola (h : Hyperbola) where
  x : ℝ
  y : ℝ
  on_hyperbola : x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- The distance from a point to the right focus of the hyperbola -/
def distToRightFocus (h : Hyperbola) (p : PointOnHyperbola h) : ℝ :=
  |p.x - h.a|

/-- The distance from a point to the left focus of the hyperbola -/
def distToLeftFocus (h : Hyperbola) (p : PointOnHyperbola h) : ℝ :=
  |p.x + h.a|

/-- The main theorem -/
theorem dist_to_left_focus_is_ten
  (h : Hyperbola)
  (p : PointOnHyperbola h)
  (right_focus_dist : distToRightFocus h p = 4)
  (h_eq : h.a = 3 ∧ h.b = 4) :
  distToLeftFocus h p = 10 := by
  sorry

end NUMINAMATH_CALUDE_dist_to_left_focus_is_ten_l2555_255506


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2555_255553

theorem arithmetic_sequence_sum (a₁ aₙ : ℤ) (n : ℕ) (h : n > 0) :
  let S := n * (a₁ + aₙ) / 2
  a₁ = -3 ∧ aₙ = 48 ∧ n = 12 → S = 270 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2555_255553


namespace NUMINAMATH_CALUDE_solution_to_equation_l2555_255545

theorem solution_to_equation (x : ℝ) : 2 * x - 8 = 0 ↔ x = 4 := by sorry

end NUMINAMATH_CALUDE_solution_to_equation_l2555_255545


namespace NUMINAMATH_CALUDE_standard_deviation_of_numbers_l2555_255511

def numbers : List ℝ := [9.8, 9.8, 9.9, 9.9, 10.0, 10.0, 10.1, 10.5]

theorem standard_deviation_of_numbers :
  let mean : ℝ := 10
  let count_within_one_std : ℕ := 7
  let n : ℕ := numbers.length
  ∀ σ : ℝ,
    (mean = (numbers.sum / n)) →
    (count_within_one_std = (numbers.filter (λ x => |x - mean| ≤ σ)).length) →
    (count_within_one_std = (n * 875 / 1000)) →
    σ = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_standard_deviation_of_numbers_l2555_255511


namespace NUMINAMATH_CALUDE_product_equals_seven_l2555_255528

theorem product_equals_seven : 
  (1 + 1/1) * (1 + 1/2) * (1 + 1/3) * (1 + 1/4) * (1 + 1/5) * (1 + 1/6) = 7 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_seven_l2555_255528


namespace NUMINAMATH_CALUDE_families_left_near_mountain_l2555_255563

/-- The number of bird families initially living near the mountain. -/
def initial_families : ℕ := 41

/-- The number of bird families that flew away for the winter. -/
def families_flew_away : ℕ := 27

/-- Theorem: The number of bird families left near the mountain is 14. -/
theorem families_left_near_mountain :
  initial_families - families_flew_away = 14 := by
  sorry

end NUMINAMATH_CALUDE_families_left_near_mountain_l2555_255563


namespace NUMINAMATH_CALUDE_difference_of_squares_153_147_l2555_255568

theorem difference_of_squares_153_147 : 153^2 - 147^2 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_153_147_l2555_255568


namespace NUMINAMATH_CALUDE_richards_walking_ratio_l2555_255579

/-- Proves that the ratio of Richard's second day walking distance to his first day walking distance is 1/5 --/
theorem richards_walking_ratio :
  let total_distance : ℝ := 70
  let first_day : ℝ := 20
  let third_day : ℝ := 10
  let remaining : ℝ := 36
  let second_day := total_distance - remaining - first_day - third_day
  second_day / first_day = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_richards_walking_ratio_l2555_255579


namespace NUMINAMATH_CALUDE_inverse_proportion_y_relationship_l2555_255548

/-- Proves the relationship between y-coordinates of points on an inverse proportion function -/
theorem inverse_proportion_y_relationship :
  ∀ (y₁ y₂ y₃ : ℝ),
  y₁ = -6 / (-2) →
  y₂ = -6 / (-1) →
  y₃ = -6 / 3 →
  y₂ > y₁ ∧ y₁ > y₃ :=
by
  sorry

#check inverse_proportion_y_relationship

end NUMINAMATH_CALUDE_inverse_proportion_y_relationship_l2555_255548


namespace NUMINAMATH_CALUDE_odd_m_triple_g_36_l2555_255569

def g (n : ℤ) : ℤ := 
  if n % 2 = 1 then 2 * n + 3
  else if n % 3 = 0 then n / 3
  else n - 1

theorem odd_m_triple_g_36 (m : ℤ) (h_odd : m % 2 = 1) :
  g (g (g m)) = 36 → m = 54 :=
by
  sorry

end NUMINAMATH_CALUDE_odd_m_triple_g_36_l2555_255569


namespace NUMINAMATH_CALUDE_ordering_abc_l2555_255521

theorem ordering_abc (a b c : ℝ) (ha : a = 1.01^(1/2 : ℝ)) (hb : b = 1.01^(3/5 : ℝ)) (hc : c = 0.6^(1/2 : ℝ)) : b > a ∧ a > c := by
  sorry

end NUMINAMATH_CALUDE_ordering_abc_l2555_255521


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_l2555_255512

def numbers : List Nat := [18, 24, 36]

theorem gcf_lcm_sum (C D : Nat) (hC : C = Nat.gcd 18 (Nat.gcd 24 36)) 
  (hD : D = Nat.lcm 18 (Nat.lcm 24 36)) : C + D = 78 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_l2555_255512


namespace NUMINAMATH_CALUDE_no_common_real_solution_l2555_255547

theorem no_common_real_solution :
  ¬ ∃ (x y : ℝ), (x^2 + y^2 + 16 = 0) ∧ (x^2 - 3*y + 12 = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_common_real_solution_l2555_255547


namespace NUMINAMATH_CALUDE_hawk_percentage_is_65_percent_l2555_255573

/-- Represents the percentage of birds that are hawks in the nature reserve -/
def hawk_percentage : ℝ := sorry

/-- Represents the percentage of non-hawks that are paddyfield-warblers -/
def paddyfield_warbler_ratio : ℝ := 0.4

/-- Represents the ratio of kingfishers to paddyfield-warblers -/
def kingfisher_to_warbler_ratio : ℝ := 0.25

/-- Represents the percentage of birds that are not hawks, paddyfield-warblers, or kingfishers -/
def other_birds_percentage : ℝ := 0.35

theorem hawk_percentage_is_65_percent :
  hawk_percentage = 0.65 ∧
  paddyfield_warbler_ratio * (1 - hawk_percentage) +
  kingfisher_to_warbler_ratio * paddyfield_warbler_ratio * (1 - hawk_percentage) +
  hawk_percentage +
  other_birds_percentage = 1 :=
sorry

end NUMINAMATH_CALUDE_hawk_percentage_is_65_percent_l2555_255573


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l2555_255560

/-- Given a quadratic function y = ax² + bx + c, if the points (2, y₁) and (-2, y₂) lie on the curve
    and y₁ - y₂ = 4, then b = 1 -/
theorem quadratic_coefficient (a b c y₁ y₂ : ℝ) : 
  y₁ = a * 4 + b * 2 + c →
  y₂ = a * 4 - b * 2 + c →
  y₁ - y₂ = 4 →
  b = 1 := by
  sorry


end NUMINAMATH_CALUDE_quadratic_coefficient_l2555_255560


namespace NUMINAMATH_CALUDE_goat_grazing_area_l2555_255591

/-- The side length of a square plot given a goat tied to one corner -/
theorem goat_grazing_area (rope_length : ℝ) (graze_area : ℝ) (side_length : ℝ) : 
  rope_length = 7 →
  graze_area = 38.48451000647496 →
  side_length = 7 →
  (1 / 4) * Real.pi * rope_length ^ 2 = graze_area →
  side_length = rope_length :=
by sorry

end NUMINAMATH_CALUDE_goat_grazing_area_l2555_255591


namespace NUMINAMATH_CALUDE_predicted_y_value_l2555_255584

-- Define the linear regression equation
def linear_regression (x : ℝ) (a : ℝ) : ℝ := -0.7 * x + a

-- Define the mean values
def x_mean : ℝ := 1
def y_mean : ℝ := 0.3

-- Theorem statement
theorem predicted_y_value :
  ∃ (a : ℝ), 
    (linear_regression x_mean a = y_mean) ∧ 
    (linear_regression 2 a = -0.4) := by
  sorry

end NUMINAMATH_CALUDE_predicted_y_value_l2555_255584


namespace NUMINAMATH_CALUDE_minimum_packs_for_90_cans_l2555_255541

/-- Represents the available pack sizes for soda cans -/
def PackSizes : List Nat := [6, 12, 24]

/-- The total number of cans we need to buy -/
def TotalCans : Nat := 90

/-- A function that calculates the minimum number of packs needed -/
def MinimumPacks (packSizes : List Nat) (totalCans : Nat) : Nat :=
  sorry -- Proof implementation goes here

/-- Theorem stating that the minimum number of packs needed is 5 -/
theorem minimum_packs_for_90_cans : 
  MinimumPacks PackSizes TotalCans = 5 := by
  sorry -- Proof goes here

end NUMINAMATH_CALUDE_minimum_packs_for_90_cans_l2555_255541


namespace NUMINAMATH_CALUDE_calculation_proof_l2555_255534

theorem calculation_proof : 0.2 * 63 + 1.9 * 126 + 196 * 9 = 2016 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2555_255534


namespace NUMINAMATH_CALUDE_tricia_age_l2555_255599

theorem tricia_age (vincent_age : ℕ)
  (h1 : vincent_age = 22)
  (rupert_age : ℕ)
  (h2 : rupert_age = vincent_age - 2)
  (khloe_age : ℕ)
  (h3 : rupert_age = khloe_age + 10)
  (eugene_age : ℕ)
  (h4 : khloe_age = eugene_age / 3)
  (yorick_age : ℕ)
  (h5 : yorick_age = 2 * eugene_age)
  (amilia_age : ℕ)
  (h6 : amilia_age = yorick_age / 4)
  (tricia_age : ℕ)
  (h7 : tricia_age = amilia_age / 3) :
  tricia_age = 5 := by
sorry

end NUMINAMATH_CALUDE_tricia_age_l2555_255599


namespace NUMINAMATH_CALUDE_f_divisibility_by_3_smallest_n_for_2017_l2555_255581

def f : ℕ → ℤ
  | 0 => 0  -- base case
  | n + 1 => if n.succ % 2 = 0 then -f (n.succ / 2) else f n + 1

theorem f_divisibility_by_3 (n : ℕ) : 3 ∣ f n ↔ 3 ∣ n := by sorry

def geometric_sum (n : ℕ) : ℕ := (4^(n+1) - 1) / 3

theorem smallest_n_for_2017 : 
  f (geometric_sum 1008) = 2017 ∧ 
  ∀ m : ℕ, m < geometric_sum 1008 → f m ≠ 2017 := by sorry

end NUMINAMATH_CALUDE_f_divisibility_by_3_smallest_n_for_2017_l2555_255581


namespace NUMINAMATH_CALUDE_average_donation_l2555_255588

def donations : List ℝ := [10, 12, 13.5, 40.8, 19.3, 20.8, 25, 16, 30, 30]

theorem average_donation : (donations.sum / donations.length) = 21.74 := by
  sorry

end NUMINAMATH_CALUDE_average_donation_l2555_255588


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2555_255505

theorem arithmetic_sequence_problem :
  ∀ (a d : ℝ),
  (a - d) + a + (a + d) = 6 ∧
  (a - d) * a * (a + d) = -10 →
  ((a - d = 5 ∧ a = 2 ∧ a + d = -1) ∨
   (a - d = -1 ∧ a = 2 ∧ a + d = 5)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2555_255505


namespace NUMINAMATH_CALUDE_green_ball_probability_l2555_255530

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- The probability of selecting a green ball from a given container -/
def greenProbability (c : Container) : ℚ :=
  c.green / (c.red + c.green)

/-- The probability of selecting a green ball given the problem conditions -/
theorem green_ball_probability :
  let containerX : Container := ⟨5, 5⟩
  let containerY : Container := ⟨8, 2⟩
  let containerZ : Container := ⟨3, 7⟩
  let totalContainers : ℕ := 3
  (1 : ℚ) / totalContainers * (greenProbability containerX +
                               greenProbability containerY +
                               greenProbability containerZ) = 7 / 15 := by
  sorry


end NUMINAMATH_CALUDE_green_ball_probability_l2555_255530


namespace NUMINAMATH_CALUDE_fraction_simplification_l2555_255576

theorem fraction_simplification : (3/8 + 5/6) / (5/12 + 1/4) = 29/16 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2555_255576


namespace NUMINAMATH_CALUDE_triangle_equation_l2555_255587

theorem triangle_equation (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  let angleA : ℝ := 60 * π / 180
  (a^2 = b^2 + c^2 - 2*b*c*(angleA.cos)) →
  (3 / (a + b + c) = 1 / (a + b) + 1 / (a + c)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_equation_l2555_255587


namespace NUMINAMATH_CALUDE_S_and_S_l2555_255519

-- Define the systems S and S'
def S (x y : ℝ) : Prop :=
  y * (x^4 - y^2 + x^2) = x ∧ x * (x^4 - y^2 + x^2) = 1

def S' (x y : ℝ) : Prop :=
  y * (x^4 - y^2 + x^2) = x ∧ y = x^2

-- Theorem stating that S and S' do not have the same set of solutions
theorem S_and_S'_different_solutions :
  ¬(∀ x y : ℝ, S x y ↔ S' x y) :=
sorry

end NUMINAMATH_CALUDE_S_and_S_l2555_255519


namespace NUMINAMATH_CALUDE_expression_simplification_l2555_255540

theorem expression_simplification (x : ℝ) : 
  ((3 * x - 1) - 5 * x) / 3 = -2/3 * x - 1/3 := by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2555_255540


namespace NUMINAMATH_CALUDE_circle_inside_polygon_l2555_255552

/-- A convex polygon in a 2D plane -/
structure ConvexPolygon where
  vertices : List (ℝ × ℝ)
  is_convex : Bool  -- We assume this is true for a convex polygon

/-- The area of a convex polygon -/
def area (p : ConvexPolygon) : ℝ := sorry

/-- The perimeter of a convex polygon -/
def perimeter (p : ConvexPolygon) : ℝ := sorry

/-- The distance from a point to a line segment -/
def distance_to_side (point : ℝ × ℝ) (side : (ℝ × ℝ) × (ℝ × ℝ)) : ℝ := sorry

/-- Theorem: In any convex polygon, there exists a point that is at least A/P distance away from all sides -/
theorem circle_inside_polygon (p : ConvexPolygon) :
  ∃ (center : ℝ × ℝ), 
    (∀ (side : (ℝ × ℝ) × (ℝ × ℝ)), 
      side.1 ∈ p.vertices ∧ side.2 ∈ p.vertices →
      distance_to_side center side ≥ area p / perimeter p) :=
sorry

end NUMINAMATH_CALUDE_circle_inside_polygon_l2555_255552


namespace NUMINAMATH_CALUDE_tan_4125_degrees_l2555_255549

theorem tan_4125_degrees : Real.tan (4125 * π / 180) = -(2 - Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_tan_4125_degrees_l2555_255549


namespace NUMINAMATH_CALUDE_students_taking_algebra_or_drafting_but_not_both_l2555_255546

-- Define the sets of students
def algebra : Finset ℕ := sorry
def drafting : Finset ℕ := sorry
def geometry : Finset ℕ := sorry

-- State the theorem
theorem students_taking_algebra_or_drafting_but_not_both : 
  (algebra.card + drafting.card - (algebra ∩ drafting).card) - ((geometry ∩ drafting).card - (algebra ∩ geometry ∩ drafting).card) = 42 :=
by
  -- Given conditions
  have h1 : (algebra ∩ drafting).card = 15 := sorry
  have h2 : algebra.card = 30 := sorry
  have h3 : (drafting \ algebra).card = 14 := sorry
  have h4 : (geometry \ (algebra ∪ drafting)).card = 8 := sorry
  have h5 : ((geometry ∩ drafting) \ algebra).card = 5 := sorry
  
  sorry -- Proof goes here

end NUMINAMATH_CALUDE_students_taking_algebra_or_drafting_but_not_both_l2555_255546


namespace NUMINAMATH_CALUDE_water_flow_solution_l2555_255516

/-- Represents the water flow problem --/
def water_flow_problem (t : ℝ) : Prop :=
  let initial_rate : ℝ := 2 / 10  -- 2 cups per 10 minutes
  let final_rate : ℝ := 4 / 10    -- 4 cups per 10 minutes
  let initial_duration : ℝ := 2 * t  -- flows for t minutes twice
  let final_duration : ℝ := 60    -- flows for 60 minutes at final rate
  let total_water : ℝ := initial_rate * initial_duration + final_rate * final_duration
  let remaining_water : ℝ := total_water / 2
  remaining_water = 18 ∧ t = 30

/-- Theorem stating the solution to the water flow problem --/
theorem water_flow_solution :
  ∃ t : ℝ, water_flow_problem t :=
sorry

end NUMINAMATH_CALUDE_water_flow_solution_l2555_255516


namespace NUMINAMATH_CALUDE_prob_one_pascal_20_l2555_255592

/-- The number of elements in the first n rows of Pascal's Triangle -/
def pascal_triangle_elements (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of ones in the first n rows of Pascal's Triangle -/
def pascal_triangle_ones (n : ℕ) : ℕ := 2 * n - 1

/-- The probability of selecting a 1 from the first n rows of Pascal's Triangle -/
def prob_one_pascal (n : ℕ) : ℚ :=
  (pascal_triangle_ones n) / (pascal_triangle_elements n)

theorem prob_one_pascal_20 :
  prob_one_pascal 20 = 13 / 70 := by
  sorry

end NUMINAMATH_CALUDE_prob_one_pascal_20_l2555_255592


namespace NUMINAMATH_CALUDE_one_correct_statement_l2555_255514

theorem one_correct_statement :
  (∃! n : Nat, n = 1 ∧
    (∀ a b : ℝ, a + b = 0 → a = -b) ∧
    (3^2 = 6) ∧
    (∀ a : ℚ, a > -a) ∧
    (∀ a b : ℝ, |a| = |b| → a = b)) :=
sorry

end NUMINAMATH_CALUDE_one_correct_statement_l2555_255514


namespace NUMINAMATH_CALUDE_complex_multiplication_l2555_255502

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) :
  (1 + i) * (1 - 2*i) = 3 - i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l2555_255502


namespace NUMINAMATH_CALUDE_complex_sum_example_l2555_255523

theorem complex_sum_example : 
  let z₁ : ℂ := 1 + 7*I
  let z₂ : ℂ := -2 - 4*I
  z₁ + z₂ = -1 + 3*I :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_example_l2555_255523


namespace NUMINAMATH_CALUDE_person_A_savings_l2555_255583

/-- The amount of money saved by person A -/
def savings_A : ℕ := sorry

/-- The amount of money saved by person B -/
def savings_B : ℕ := sorry

/-- The amount of money saved by person C -/
def savings_C : ℕ := sorry

/-- Person A and B together have saved 640 yuan -/
axiom AB_savings : savings_A + savings_B = 640

/-- Person B and C together have saved 600 yuan -/
axiom BC_savings : savings_B + savings_C = 600

/-- Person A and C together have saved 440 yuan -/
axiom AC_savings : savings_A + savings_C = 440

/-- Theorem: Given the conditions, person A has saved 240 yuan -/
theorem person_A_savings : savings_A = 240 :=
  sorry

end NUMINAMATH_CALUDE_person_A_savings_l2555_255583


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l2555_255572

theorem square_area_from_diagonal (d : ℝ) (h : d = 40) :
  let s := d / Real.sqrt 2
  s * s = 800 := by sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l2555_255572


namespace NUMINAMATH_CALUDE_ratio_common_value_l2555_255554

theorem ratio_common_value (x y z : ℝ) (h : (x + y) / z = (x + z) / y ∧ (x + z) / y = (y + z) / x) :
  (x + y) / z = -1 ∨ (x + y) / z = 2 :=
sorry

end NUMINAMATH_CALUDE_ratio_common_value_l2555_255554


namespace NUMINAMATH_CALUDE_total_art_pieces_l2555_255544

theorem total_art_pieces (asian : Nat) (egyptian : Nat) (european : Nat)
  (h1 : asian = 465)
  (h2 : egyptian = 527)
  (h3 : european = 320) :
  asian + egyptian + european = 1312 := by
  sorry

end NUMINAMATH_CALUDE_total_art_pieces_l2555_255544


namespace NUMINAMATH_CALUDE_ab_plus_cd_value_l2555_255577

theorem ab_plus_cd_value (a b c d : ℝ) 
  (eq1 : a + b + c = 5)
  (eq2 : a + b + d = -3)
  (eq3 : a + c + d = 10)
  (eq4 : b + c + d = -1) :
  a * b + c * d = -346 / 9 := by
sorry

end NUMINAMATH_CALUDE_ab_plus_cd_value_l2555_255577


namespace NUMINAMATH_CALUDE_brick_width_calculation_l2555_255526

/-- The width of a brick in centimeters -/
def brick_width : ℝ := 11.25

/-- The length of a brick in centimeters -/
def brick_length : ℝ := 25

/-- The height of a brick in centimeters -/
def brick_height : ℝ := 6

/-- The length of the wall in centimeters -/
def wall_length : ℝ := 900

/-- The width of the wall in centimeters -/
def wall_width : ℝ := 600

/-- The height of the wall in centimeters -/
def wall_height : ℝ := 22.5

/-- The number of bricks needed -/
def num_bricks : ℕ := 7200

/-- The volume of the wall in cubic centimeters -/
def wall_volume : ℝ := wall_length * wall_width * wall_height

/-- The volume of a single brick in cubic centimeters -/
def brick_volume : ℝ := brick_length * brick_width * brick_height

theorem brick_width_calculation :
  brick_width = (wall_volume / (num_bricks : ℝ)) / (brick_length * brick_height) :=
by sorry

end NUMINAMATH_CALUDE_brick_width_calculation_l2555_255526


namespace NUMINAMATH_CALUDE_student_class_sizes_l2555_255555

/-- Represents a configuration of students in classes -/
structure StudentConfig where
  total_students : ℕ
  classes : List ℕ
  classes_sum_eq_total : classes.sum = total_students

/-- Checks if any group of n students contains at least k from the same class -/
def satisfies_group_condition (config : StudentConfig) (n k : ℕ) : Prop :=
  ∀ (subset : List ℕ), subset.sum ≤ n → (∃ (c : ℕ), c ∈ config.classes ∧ c ≥ k)

/-- The main theorem to be proved -/
theorem student_class_sizes 
  (config : StudentConfig)
  (h_total : config.total_students = 60)
  (h_condition : satisfies_group_condition config 10 3) :
  (∃ (c : ℕ), c ∈ config.classes ∧ c ≥ 15) ∧
  ¬(∀ (config : StudentConfig), 
    config.total_students = 60 → 
    satisfies_group_condition config 10 3 → 
    ∃ (c : ℕ), c ∈ config.classes ∧ c ≥ 16) :=
by sorry

end NUMINAMATH_CALUDE_student_class_sizes_l2555_255555


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_condition_l2555_255522

/-- For a quadratic equation (k+2)x^2 + 4x + 1 = 0 to have two distinct real roots, 
    k must satisfy: k < 2 and k ≠ -2 -/
theorem quadratic_distinct_roots_condition (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
   (k + 2) * x^2 + 4 * x + 1 = 0 ∧ 
   (k + 2) * y^2 + 4 * y + 1 = 0) ↔ 
  (k < 2 ∧ k ≠ -2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_condition_l2555_255522


namespace NUMINAMATH_CALUDE_textbook_transfer_l2555_255595

theorem textbook_transfer (initial_a initial_b transfer : ℕ) 
  (h1 : initial_a = 200)
  (h2 : initial_b = 200)
  (h3 : transfer = 40) :
  (initial_b + transfer) = (initial_a - transfer) * 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_textbook_transfer_l2555_255595


namespace NUMINAMATH_CALUDE_iron_bar_height_l2555_255536

/-- Proves that the height of an iron bar is 6 cm given specific conditions --/
theorem iron_bar_height : 
  ∀ (length width height : ℝ) (num_bars num_balls ball_volume : ℕ),
  length = 12 →
  width = 8 →
  num_bars = 10 →
  num_balls = 720 →
  ball_volume = 8 →
  (num_bars : ℝ) * length * width * height = (num_balls : ℝ) * (ball_volume : ℝ) →
  height = 6 := by
sorry

end NUMINAMATH_CALUDE_iron_bar_height_l2555_255536


namespace NUMINAMATH_CALUDE_shoes_theorem_l2555_255565

def shoes_problem (bonny becky bobby cherry diane : ℚ) : Prop :=
  -- Conditions
  bonny = 13 ∧
  bonny = 2 * becky - 5 ∧
  bobby = 3.5 * becky ∧
  cherry = bonny + becky + 4.5 ∧
  diane = 3 * cherry - 2 - 3 ∧
  -- Conclusion
  ⌊bonny + becky + bobby + cherry + diane⌋ = 154

theorem shoes_theorem : ∃ bonny becky bobby cherry diane : ℚ, 
  shoes_problem bonny becky bobby cherry diane := by
  sorry

end NUMINAMATH_CALUDE_shoes_theorem_l2555_255565


namespace NUMINAMATH_CALUDE_baba_yaga_journey_l2555_255542

/-- The problem of Baba Yaga's journey to Bald Mountain -/
theorem baba_yaga_journey 
  (arrival_time : ℕ) 
  (slow_speed : ℕ) 
  (fast_speed : ℕ) 
  (late_hours : ℕ) 
  (early_hours : ℕ) 
  (h : arrival_time = 24) -- Midnight is represented as 24
  (h_slow : slow_speed = 50)
  (h_fast : fast_speed = 150)
  (h_late : late_hours = 2)
  (h_early : early_hours = 2)
  : ∃ (departure_time speed : ℕ),
    departure_time = 20 ∧ 
    speed = 75 ∧
    (arrival_time - departure_time) * speed = 
      (arrival_time - departure_time + late_hours) * slow_speed ∧
    (arrival_time - departure_time) * speed = 
      (arrival_time - departure_time - early_hours) * fast_speed :=
sorry

end NUMINAMATH_CALUDE_baba_yaga_journey_l2555_255542


namespace NUMINAMATH_CALUDE_factorial_plus_24_equals_square_l2555_255571

theorem factorial_plus_24_equals_square (n m : ℕ) : n.factorial + 24 = m ^ 2 ↔ (n = 1 ∧ m = 5) ∨ (n = 5 ∧ m = 12) := by
  sorry

end NUMINAMATH_CALUDE_factorial_plus_24_equals_square_l2555_255571


namespace NUMINAMATH_CALUDE_flu_outbreak_l2555_255518

theorem flu_outbreak (initial_infected : ℕ) (infected_after_two_rounds : ℕ) :
  initial_infected = 1 →
  infected_after_two_rounds = 81 →
  ∃ (avg_infected_per_round : ℕ),
    avg_infected_per_round = 8 ∧
    initial_infected + avg_infected_per_round + avg_infected_per_round * (avg_infected_per_round + 1) = infected_after_two_rounds ∧
    infected_after_two_rounds * avg_infected_per_round + infected_after_two_rounds = 729 :=
by sorry

end NUMINAMATH_CALUDE_flu_outbreak_l2555_255518


namespace NUMINAMATH_CALUDE_zoo_animal_ratio_l2555_255562

theorem zoo_animal_ratio (parrots : ℕ) (snakes : ℕ) (elephants : ℕ) (zebras : ℕ) (monkeys : ℕ) :
  parrots = 8 →
  snakes = 3 * parrots →
  elephants = (parrots + snakes) / 2 →
  zebras = elephants - 3 →
  monkeys - zebras = 35 →
  monkeys / snakes = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_zoo_animal_ratio_l2555_255562


namespace NUMINAMATH_CALUDE_composite_function_equation_solution_l2555_255578

theorem composite_function_equation_solution :
  let δ : ℝ → ℝ := λ x ↦ 4 * x + 9
  let φ : ℝ → ℝ := λ x ↦ 9 * x + 8
  ∃! x : ℝ, (δ ∘ φ) x = 10 ∧ x = -31 / 36 :=
by
  sorry

end NUMINAMATH_CALUDE_composite_function_equation_solution_l2555_255578


namespace NUMINAMATH_CALUDE_function_composition_problem_l2555_255589

theorem function_composition_problem (f g : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = x / 3 + 2) →
  (∀ x, g x = 5 - 2 * x) →
  f (g a) = 6 →
  a = -7/2 := by
sorry

end NUMINAMATH_CALUDE_function_composition_problem_l2555_255589


namespace NUMINAMATH_CALUDE_vector_relation_in_triangle_l2555_255500

/-- Given a triangle ABC and a point D, if AB = 4DB, then CD = (1/4)CA + (3/4)CB -/
theorem vector_relation_in_triangle (A B C D : EuclideanSpace ℝ (Fin 3)) :
  (B - A) = 4 • (B - D) →
  (D - C) = (1/4) • (A - C) + (3/4) • (B - C) := by
  sorry

end NUMINAMATH_CALUDE_vector_relation_in_triangle_l2555_255500


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l2555_255570

theorem circle_area_from_circumference (C : ℝ) (r : ℝ) (h : C = 36 * Real.pi) :
  r * r * Real.pi = 324 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l2555_255570


namespace NUMINAMATH_CALUDE_boxer_weight_theorem_l2555_255586

/-- Represents a diet with a specific weight loss per month -/
structure Diet where
  weightLossPerMonth : ℝ
  
/-- Calculates the weight after a given number of months on a diet -/
def weightAfterMonths (initialWeight : ℝ) (diet : Diet) (months : ℝ) : ℝ :=
  initialWeight - diet.weightLossPerMonth * months

/-- Theorem about boxer's weight and diets -/
theorem boxer_weight_theorem (x : ℝ) :
  let dietA : Diet := ⟨2⟩
  let dietB : Diet := ⟨3⟩
  let dietC : Diet := ⟨4⟩
  let monthsToFight : ℝ := 4
  
  (weightAfterMonths x dietB monthsToFight = 97) →
  (x = 109) ∧
  (weightAfterMonths x dietA monthsToFight = 101) ∧
  (weightAfterMonths x dietB monthsToFight = 97) ∧
  (weightAfterMonths x dietC monthsToFight = 93) := by
  sorry


end NUMINAMATH_CALUDE_boxer_weight_theorem_l2555_255586


namespace NUMINAMATH_CALUDE_intersection_symmetry_l2555_255567

/-- The line y = ax + 1 intersects the curve x^2 + y^2 + bx - y = 1 at two points
    which are symmetric about the line x + y = 0. -/
theorem intersection_symmetry (a b : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    -- Line equation
    (y₁ = a * x₁ + 1) ∧ (y₂ = a * x₂ + 1) ∧
    -- Curve equation
    (x₁^2 + y₁^2 + b * x₁ - y₁ = 1) ∧ (x₂^2 + y₂^2 + b * x₂ - y₂ = 1) ∧
    -- Symmetry condition
    (x₁ + y₁ = -(x₂ + y₂)) ∧
    -- Distinct points
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂)) →
  a + b = 2 := by
sorry

end NUMINAMATH_CALUDE_intersection_symmetry_l2555_255567


namespace NUMINAMATH_CALUDE_smallest_number_l2555_255580

theorem smallest_number (S : Set ℤ) (h : S = {-2, 0, 1, 2}) : 
  ∃ m ∈ S, ∀ n ∈ S, m ≤ n ∧ m = -2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l2555_255580


namespace NUMINAMATH_CALUDE_geometric_series_r_value_l2555_255501

theorem geometric_series_r_value (b r : ℝ) (h1 : r ≠ 1) (h2 : r ≠ -1) : 
  (b / (1 - r) = 18) → 
  (b * r^2 / (1 - r^2) = 6) → 
  r = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_r_value_l2555_255501


namespace NUMINAMATH_CALUDE_quadratic_roots_real_and_unequal_l2555_255525

theorem quadratic_roots_real_and_unequal :
  let a : ℝ := 1
  let b : ℝ := -6
  let c : ℝ := 8
  let discriminant := b^2 - 4*a*c
  discriminant > 0 ∧ ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 :=
by
  sorry

#check quadratic_roots_real_and_unequal

end NUMINAMATH_CALUDE_quadratic_roots_real_and_unequal_l2555_255525


namespace NUMINAMATH_CALUDE_percent_relationship_l2555_255598

theorem percent_relationship (x y : ℝ) (h : 0.25 * (x - y) = 0.15 * (x + y)) :
  y / x = 0.25 := by
sorry

end NUMINAMATH_CALUDE_percent_relationship_l2555_255598


namespace NUMINAMATH_CALUDE_sqrt_fraction_equality_l2555_255564

theorem sqrt_fraction_equality : Real.sqrt (25 / 121) = 5 / 11 := by sorry

end NUMINAMATH_CALUDE_sqrt_fraction_equality_l2555_255564


namespace NUMINAMATH_CALUDE_office_officers_count_l2555_255535

/-- Represents the salary and employee data for an office --/
structure OfficeSalaryData where
  avgSalaryAll : ℚ
  avgSalaryOfficers : ℚ
  avgSalaryNonOfficers : ℚ
  numNonOfficers : ℕ

/-- Calculates the number of officers given the office salary data --/
def calculateOfficers (data : OfficeSalaryData) : ℕ :=
  sorry

/-- Theorem stating that the number of officers is 15 given the specific salary data --/
theorem office_officers_count (data : OfficeSalaryData) 
  (h1 : data.avgSalaryAll = 120)
  (h2 : data.avgSalaryOfficers = 450)
  (h3 : data.avgSalaryNonOfficers = 110)
  (h4 : data.numNonOfficers = 495) :
  calculateOfficers data = 15 := by
  sorry

end NUMINAMATH_CALUDE_office_officers_count_l2555_255535


namespace NUMINAMATH_CALUDE_complement_of_A_l2555_255529

def A : Set ℝ := {x : ℝ | (x - 2) / (x - 1) ≥ 0}

theorem complement_of_A : 
  (Set.univ \ A : Set ℝ) = {x : ℝ | 1 ≤ x ∧ x < 2} :=
by sorry

end NUMINAMATH_CALUDE_complement_of_A_l2555_255529


namespace NUMINAMATH_CALUDE_inequality_proof_l2555_255538

theorem inequality_proof (a : ℝ) (h1 : 0 < a) (h2 : a < 1/2) :
  a^(Real.sqrt a) > a^(a^a) ∧ a^(a^a) > a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2555_255538


namespace NUMINAMATH_CALUDE_quadratic_properties_l2555_255509

/-- Represents a quadratic function f(x) = ax² + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The quadratic function satisfying the given conditions -/
def f : QuadraticFunction := {
  a := -1,
  b := 2,
  c := 3,
  a_nonzero := by norm_num
}

/-- Evaluation of the quadratic function -/
def eval (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

theorem quadratic_properties (t : ℝ) :
  let f := f
  (∃ x y, x > 0 ∧ y > 0 ∧ eval f x = y ∧ ∀ x', eval f x' ≤ eval f x) ∧ 
  (∀ m n, 0 < m → m < 4 → eval f m = n → -5 < n ∧ n ≤ 4) ∧
  (eval f (-2) = t ∧ eval f 4 = t) ∧
  (∀ p, (∀ x, eval f x < 2*x + p) → p > 3) := by
  sorry


end NUMINAMATH_CALUDE_quadratic_properties_l2555_255509


namespace NUMINAMATH_CALUDE_book_reading_time_l2555_255559

theorem book_reading_time (total_pages : ℕ) (rate1 rate2 : ℕ) (days1 days2 : ℕ) : 
  total_pages = 525 →
  rate1 = 25 →
  rate2 = 21 →
  days1 * rate1 = total_pages →
  days2 * rate2 = total_pages →
  (days1 = 21 ∧ days2 = 25) := by
  sorry

end NUMINAMATH_CALUDE_book_reading_time_l2555_255559


namespace NUMINAMATH_CALUDE_no_prime_pair_sum_65_l2555_255575

theorem no_prime_pair_sum_65 : ¬∃ (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ p + q = 65 ∧ ∃ (k : ℕ), p * q = k := by
  sorry

end NUMINAMATH_CALUDE_no_prime_pair_sum_65_l2555_255575


namespace NUMINAMATH_CALUDE_cricketer_average_score_l2555_255550

theorem cricketer_average_score (score1 score2 : ℕ) (matches1 matches2 : ℕ) 
  (h1 : matches1 = 2)
  (h2 : matches2 = 3)
  (h3 : score1 = 60)
  (h4 : score2 = 50) :
  (matches1 * score1 + matches2 * score2) / (matches1 + matches2) = 54 := by
  sorry

end NUMINAMATH_CALUDE_cricketer_average_score_l2555_255550


namespace NUMINAMATH_CALUDE_pen_package_size_l2555_255582

def is_proper_factor (n m : ℕ) : Prop := n ∣ m ∧ n ≠ 1 ∧ n ≠ m

theorem pen_package_size (pen_package_size : ℕ) 
  (h1 : pen_package_size > 0)
  (h2 : ∃ (num_packages : ℕ), num_packages * pen_package_size = 60) :
  is_proper_factor pen_package_size 60 := by
sorry

end NUMINAMATH_CALUDE_pen_package_size_l2555_255582


namespace NUMINAMATH_CALUDE_division_simplification_l2555_255507

theorem division_simplification (a : ℝ) (h : a ≠ 0) : 6 * a / (2 * a) = 3 := by
  sorry

end NUMINAMATH_CALUDE_division_simplification_l2555_255507


namespace NUMINAMATH_CALUDE_inscribed_rectangle_circle_circumference_l2555_255510

theorem inscribed_rectangle_circle_circumference :
  ∀ (rectangle_width rectangle_height : ℝ) (circle_circumference : ℝ),
    rectangle_width = 9 →
    rectangle_height = 12 →
    circle_circumference = π * (rectangle_width^2 + rectangle_height^2).sqrt →
    circle_circumference = 15 * π := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_circle_circumference_l2555_255510


namespace NUMINAMATH_CALUDE_wechat_payment_balance_l2555_255574

/-- Represents a transaction with a description and an amount -/
structure Transaction where
  description : String
  amount : Int

/-- Calculates the balance from a list of transactions -/
def calculate_balance (transactions : List Transaction) : Int :=
  transactions.foldl (fun acc t => acc + t.amount) 0

/-- Theorem stating that the WeChat change payment balance for the day is an expenditure of $32 -/
theorem wechat_payment_balance : 
  let transactions : List Transaction := [
    { description := "Transfer from LZT", amount := 48 },
    { description := "Blue Wisteria Culture", amount := -30 },
    { description := "Scan QR code payment", amount := -50 }
  ]
  calculate_balance transactions = -32 := by sorry

end NUMINAMATH_CALUDE_wechat_payment_balance_l2555_255574


namespace NUMINAMATH_CALUDE_number_puzzle_l2555_255520

theorem number_puzzle : ∃! x : ℝ, (x / 12) * 24 = x + 36 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l2555_255520


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_for_special_case_l2555_255558

theorem gcd_lcm_sum_for_special_case (a b : ℕ) (h : a = 1999 * b) :
  Nat.gcd a b + Nat.lcm a b = 2000 * b := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_for_special_case_l2555_255558


namespace NUMINAMATH_CALUDE_max_y_value_l2555_255531

theorem max_y_value (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) 
  (h_eq : x * y = (x - y) / (x + 3 * y)) : 
  y ≤ 1 / 3 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ = 1 / 3 ∧ x₀ * y₀ = (x₀ - y₀) / (x₀ + 3 * y₀) :=
by sorry

end NUMINAMATH_CALUDE_max_y_value_l2555_255531


namespace NUMINAMATH_CALUDE_pumpkin_total_weight_l2555_255590

/-- The total weight of two pumpkins is 12.7 pounds, given their individual weights -/
theorem pumpkin_total_weight (weight1 weight2 : ℝ) 
  (h1 : weight1 = 4) 
  (h2 : weight2 = 8.7) : 
  weight1 + weight2 = 12.7 := by
  sorry

end NUMINAMATH_CALUDE_pumpkin_total_weight_l2555_255590


namespace NUMINAMATH_CALUDE_cost_doubling_cost_percentage_increase_l2555_255556

theorem cost_doubling (t b : ℝ) (t_pos : t > 0) (b_pos : b > 0) : 
  t * (2 * b)^4 = 16 * (t * b^4) := by
  sorry

theorem cost_percentage_increase (t b : ℝ) (t_pos : t > 0) (b_pos : b > 0) :
  (t * (2 * b)^4) / (t * b^4) * 100 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_cost_doubling_cost_percentage_increase_l2555_255556


namespace NUMINAMATH_CALUDE_race_distance_l2555_255597

theorem race_distance (a_finish_time : ℝ) (time_diff : ℝ) (distance_diff : ℝ) :
  a_finish_time = 3 →
  time_diff = 7 →
  distance_diff = 56 →
  ∃ (total_distance : ℝ),
    total_distance = 136 ∧
    (total_distance / a_finish_time) * time_diff = distance_diff :=
by sorry

end NUMINAMATH_CALUDE_race_distance_l2555_255597


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l2555_255594

theorem diophantine_equation_solution :
  ∀ x y z : ℕ+,
    (2 * (x + y + z + 2 * x * y * z)^2 = (2 * x * y + 2 * y * z + 2 * z * x + 1)^2 + 2023) ↔
    ((x = 3 ∧ y = 3 ∧ z = 2) ∨
     (x = 3 ∧ y = 2 ∧ z = 3) ∨
     (x = 2 ∧ y = 3 ∧ z = 3)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l2555_255594


namespace NUMINAMATH_CALUDE_point_value_theorem_l2555_255533

/-- Represents a point on a number line -/
structure Point where
  value : ℝ

/-- Represents the number line -/
structure NumberLine where
  origin : Point
  pointA : Point
  pointB : Point
  pointC : Point

def NumberLine.sameSide (nl : NumberLine) : Prop :=
  (nl.pointA.value - nl.origin.value) * (nl.pointB.value - nl.origin.value) > 0

theorem point_value_theorem (nl : NumberLine) 
  (h1 : nl.sameSide)
  (h2 : nl.pointB.value = 1)
  (h3 : nl.pointC.value = nl.pointA.value - 3)
  (h4 : nl.pointC.value = -nl.pointB.value) :
  nl.pointA.value = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_value_theorem_l2555_255533


namespace NUMINAMATH_CALUDE_calculation_result_l2555_255539

theorem calculation_result : (0.0088 * 4.5) / (0.05 * 0.1 * 0.008) = 990 := by
  sorry

end NUMINAMATH_CALUDE_calculation_result_l2555_255539


namespace NUMINAMATH_CALUDE_arithmetic_progression_with_small_prime_factors_l2555_255543

/-- The greatest prime factor of a positive integer n > 1 -/
noncomputable def greatestPrimeFactor (n : ℕ) : ℕ := sorry

/-- Check if three numbers form an arithmetic progression -/
def isArithmeticProgression (x y z : ℕ) : Prop :=
  y - x = z - y

/-- Main theorem -/
theorem arithmetic_progression_with_small_prime_factors
  (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hdistinct : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (hap : isArithmeticProgression x y z)
  (hprime : greatestPrimeFactor (x * y * z) ≤ 3) :
  ∃ (l a b : ℕ), (a ≥ 0 ∧ b ≥ 0) ∧ l = 2^a * 3^b ∧
    ((x, y, z) = (l, 2*l, 3*l) ∨
     (x, y, z) = (2*l, 3*l, 4*l) ∨
     (x, y, z) = (2*l, 9*l, 16*l)) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_with_small_prime_factors_l2555_255543


namespace NUMINAMATH_CALUDE_definite_integral_exp_minus_2x_l2555_255513

theorem definite_integral_exp_minus_2x : 
  ∫ x in (0: ℝ)..1, (Real.exp x - 2 * x) = Real.exp 1 - 2 := by sorry

end NUMINAMATH_CALUDE_definite_integral_exp_minus_2x_l2555_255513


namespace NUMINAMATH_CALUDE_binomial_20_4_l2555_255504

theorem binomial_20_4 : Nat.choose 20 4 = 4845 := by sorry

end NUMINAMATH_CALUDE_binomial_20_4_l2555_255504


namespace NUMINAMATH_CALUDE_no_digit_move_multiplier_l2555_255585

theorem no_digit_move_multiplier : ¬∃ (N : ℕ), 
  ∃ (d : ℕ) (M : ℕ) (k : ℕ),
    (N = d * 10^k + M) ∧ 
    (d ≥ 1) ∧ (d ≤ 9) ∧ 
    (10 * M + d = 5 * N ∨ 10 * M + d = 6 * N ∨ 10 * M + d = 8 * N) := by
  sorry

end NUMINAMATH_CALUDE_no_digit_move_multiplier_l2555_255585


namespace NUMINAMATH_CALUDE_unique_solution_divisor_system_l2555_255508

theorem unique_solution_divisor_system :
  ∀ a b : ℕ+,
  (∃ (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℕ+),
    a₁ < a₂ ∧ a₂ < a₃ ∧ a₃ < a₄ ∧ a₄ < a₅ ∧ a₅ < a₆ ∧ a₆ < a₇ ∧ a₇ < a₈ ∧ a₈ < a₉ ∧ a₉ < a₁₀ ∧ a₁₀ < a₁₁ ∧
    a₁ ∣ a ∧ a₂ ∣ a ∧ a₃ ∣ a ∧ a₄ ∣ a ∧ a₅ ∣ a ∧ a₆ ∣ a ∧ a₇ ∣ a ∧ a₈ ∣ a ∧ a₉ ∣ a ∧ a₁₀ ∣ a ∧ a₁₁ ∣ a) →
  (∃ (b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ b₁₀ b₁₁ : ℕ+),
    b₁ < b₂ ∧ b₂ < b₃ ∧ b₃ < b₄ ∧ b₄ < b₅ ∧ b₅ < b₆ ∧ b₆ < b₇ ∧ b₇ < b₈ ∧ b₈ < b₉ ∧ b₉ < b₁₀ ∧ b₁₀ < b₁₁ ∧
    b₁ ∣ b ∧ b₂ ∣ b ∧ b₃ ∣ b ∧ b₄ ∣ b ∧ b₅ ∣ b ∧ b₆ ∣ b ∧ b₇ ∣ b ∧ b₈ ∣ b ∧ b₉ ∣ b ∧ b₁₀ ∣ b ∧ b₁₁ ∣ b) →
  a₁₀ + b₁₀ = a →
  a₁₁ + b₁₁ = b →
  a = 1024 ∧ b = 2048 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_divisor_system_l2555_255508


namespace NUMINAMATH_CALUDE_correct_divisor_l2555_255596

theorem correct_divisor (incorrect_result : ℝ) (dividend : ℝ) (h1 : incorrect_result = 204) (h2 : dividend = 30.6) :
  ∃ (correct_divisor : ℝ), 
    dividend / (correct_divisor * 10) = incorrect_result ∧
    correct_divisor = (dividend / incorrect_result) / 10 := by
  sorry

end NUMINAMATH_CALUDE_correct_divisor_l2555_255596


namespace NUMINAMATH_CALUDE_final_distance_to_catch_up_l2555_255515

/-- Represents the state of the race at any given point --/
structure RaceState where
  alex_lead : Int
  distance_covered : Nat

/-- Calculates the new race state after a terrain change --/
def update_race_state (current_state : RaceState) (alex_gain : Int) : RaceState :=
  { alex_lead := current_state.alex_lead + alex_gain,
    distance_covered := current_state.distance_covered }

def race_length : Nat := 5000

theorem final_distance_to_catch_up :
  let initial_state : RaceState := { alex_lead := 0, distance_covered := 200 }
  let after_uphill := update_race_state initial_state 300
  let after_downhill := update_race_state after_uphill (-170)
  let final_state := update_race_state after_downhill 440
  final_state.alex_lead = 570 := by sorry

end NUMINAMATH_CALUDE_final_distance_to_catch_up_l2555_255515
