import Mathlib

namespace simplify_and_evaluate_1_simplify_and_evaluate_2_l1072_107287

-- Part 1
theorem simplify_and_evaluate_1 (x : ℝ) (h : x = 3) :
  3 * x^2 - (5 * x - (6 * x - 4) - 2 * x^2) = 44 := by sorry

-- Part 2
theorem simplify_and_evaluate_2 (m n : ℝ) (h1 : m = -1) (h2 : n = 2) :
  (8 * m * n - 3 * m^2) - 5 * m * n - 2 * (3 * m * n - 2 * m^2) = 7 := by sorry

end simplify_and_evaluate_1_simplify_and_evaluate_2_l1072_107287


namespace sum_equality_l1072_107248

theorem sum_equality (a b c d : ℝ) 
  (hab : a + b = 4)
  (hbc : b + c = 5)
  (had : a + d = 2) :
  c + d = 3 := by
sorry

end sum_equality_l1072_107248


namespace fish_population_calculation_l1072_107296

/-- Calculates the number of fish in a lake on May 1 based on sampling data --/
theorem fish_population_calculation (tagged_may : ℕ) (caught_sept : ℕ) (tagged_sept : ℕ) 
  (death_rate : ℚ) (new_fish_rate : ℚ) :
  tagged_may = 60 →
  caught_sept = 70 →
  tagged_sept = 3 →
  death_rate = 1/4 →
  new_fish_rate = 2/5 →
  (1 - death_rate) * tagged_may * caught_sept / tagged_sept * (1 - new_fish_rate) = 630 := by
  sorry

end fish_population_calculation_l1072_107296


namespace physics_class_grades_l1072_107211

theorem physics_class_grades (total_students : ℕ) (prob_A prob_B prob_C : ℚ) :
  total_students = 42 →
  prob_A = 2 * prob_B →
  prob_C = 1.2 * prob_B →
  prob_A + prob_B + prob_C = 1 →
  (prob_B * total_students : ℚ) = 10 :=
by sorry

end physics_class_grades_l1072_107211


namespace cone_frustum_sphere_ratio_l1072_107245

/-- A cone frustum with given height and base radius -/
structure ConeFrustum where
  height : ℝ
  baseRadius : ℝ

/-- The radius of the inscribed sphere of a cone frustum -/
def inscribedSphereRadius (cf : ConeFrustum) : ℝ := sorry

/-- The radius of the circumscribed sphere of a cone frustum -/
def circumscribedSphereRadius (cf : ConeFrustum) : ℝ := sorry

theorem cone_frustum_sphere_ratio :
  let cf : ConeFrustum := { height := 12, baseRadius := 5 }
  (inscribedSphereRadius cf) / (circumscribedSphereRadius cf) = 80 / 169 := by
  sorry

end cone_frustum_sphere_ratio_l1072_107245


namespace winning_numbers_are_correct_l1072_107280

def winning_numbers : Set Nat :=
  {n : Nat | n ≥ 1 ∧ n ≤ 999 ∧ n % 100 = 88}

theorem winning_numbers_are_correct :
  winning_numbers = {88, 188, 288, 388, 488, 588, 688, 788, 888, 988} := by
  sorry

end winning_numbers_are_correct_l1072_107280


namespace jose_bottle_caps_l1072_107256

def initial_bottle_caps : ℕ := 26
def additional_bottle_caps : ℕ := 13

theorem jose_bottle_caps : 
  initial_bottle_caps + additional_bottle_caps = 39 := by
  sorry

end jose_bottle_caps_l1072_107256


namespace integer_solution_inequalities_l1072_107217

theorem integer_solution_inequalities :
  ∀ x : ℤ, (x + 7 > 5 ∧ -3*x > -9) ↔ x ∈ ({-1, 0, 1, 2} : Set ℤ) := by
  sorry

end integer_solution_inequalities_l1072_107217


namespace fraction_equality_sum_l1072_107255

theorem fraction_equality_sum (p q : ℚ) : p / q = 2 / 7 → 2 * p + q = 11 * p / 2 := by
  sorry

end fraction_equality_sum_l1072_107255


namespace no_integer_solution_l1072_107276

theorem no_integer_solution : ¬ ∃ (n : ℤ), (n + 15 > 18) ∧ (-3*n > -9) := by
  sorry

end no_integer_solution_l1072_107276


namespace overlapping_squares_area_l1072_107250

/-- Given two squares with side length a that overlap such that one pair of vertices coincide,
    and the overlapping part forms a right triangle with an angle of 30°,
    the area of the non-overlapping part is 2(1 - √3/3)a². -/
theorem overlapping_squares_area (a : ℝ) (h : a > 0) :
  let overlap_angle : ℝ := 30 * π / 180
  let overlap_area : ℝ := a^2 * (Real.sin overlap_angle * Real.cos overlap_angle)
  let non_overlap_area : ℝ := 2 * (a^2 - overlap_area)
  non_overlap_area = 2 * (1 - Real.sqrt 3 / 3) * a^2 := by
  sorry

end overlapping_squares_area_l1072_107250


namespace triangle_theorem_l1072_107242

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a * Real.sin t.C = Real.sqrt (3 * t.c) * Real.cos t.A) :
  t.A = π / 3 ∧ 
  (t.c = 4 ∧ t.a = 5 * Real.sqrt 3 → 
    Real.cos (2 * t.C - t.A) = (17 + 12 * Real.sqrt 7) / 50) := by
  sorry


end triangle_theorem_l1072_107242


namespace garrison_reinforcement_reinforcement_size_l1072_107284

theorem garrison_reinforcement (initial_garrison : ℕ) (initial_days : ℕ) 
  (days_before_reinforcement : ℕ) (days_after_reinforcement : ℕ) : ℕ :=
  let total_provisions := initial_garrison * initial_days
  let remaining_provisions := total_provisions - (initial_garrison * days_before_reinforcement)
  let reinforcement := (remaining_provisions / days_after_reinforcement) - initial_garrison
  reinforcement

theorem reinforcement_size :
  garrison_reinforcement 1000 60 15 20 = 1250 := by sorry

end garrison_reinforcement_reinforcement_size_l1072_107284


namespace complex_fraction_simplification_l1072_107246

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (1 + 3*i) / (1 + i) = 2 + i := by sorry

end complex_fraction_simplification_l1072_107246


namespace problem_solution_l1072_107205

theorem problem_solution (a b c : ℚ) : 
  8 = (2 / 100) * a → 
  2 = (8 / 100) * b → 
  c = b / a → 
  c = 1 / 16 := by sorry

end problem_solution_l1072_107205


namespace subset_ratio_theorem_l1072_107285

theorem subset_ratio_theorem (n k : ℕ) (h1 : n ≥ 2*k) (h2 : 2*k > 3) :
  (Nat.choose n k = (2*n - k) * Nat.choose n 2) ↔ (n = 27 ∧ k = 4) := by
  sorry

end subset_ratio_theorem_l1072_107285


namespace bernardo_wins_l1072_107222

def game_winner (N : ℕ) : Prop :=
  N ≤ 999 ∧
  2 * N < 1000 ∧
  2 * N + 100 < 1000 ∧
  4 * N + 200 < 1000 ∧
  4 * N + 300 < 1000 ∧
  8 * N + 600 < 1000 ∧
  8 * N + 700 ≥ 1000

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem bernardo_wins :
  ∃ N, game_winner N ∧ 
    (∀ M, M < N → ¬game_winner M) ∧
    N = 38 ∧
    sum_of_digits N = 11 :=
  sorry

end bernardo_wins_l1072_107222


namespace percentage_problem_l1072_107298

theorem percentage_problem (x : ℝ) :
  (0.15 * 0.30 * (x / 100) * 5200 = 117) → x = 50 := by
  sorry

end percentage_problem_l1072_107298


namespace greatest_n_for_perfect_square_T_l1072_107266

def g (x : ℕ) : ℕ := 
  if x % 2 = 1 then 1 else 0

def T (n : ℕ) : ℕ := 2^n

def is_perfect_square (x : ℕ) : Prop :=
  ∃ y : ℕ, y * y = x

theorem greatest_n_for_perfect_square_T : 
  (∃ n : ℕ, n < 500 ∧ is_perfect_square (T n) ∧ 
    ∀ m : ℕ, m < 500 → is_perfect_square (T m) → m ≤ n) →
  (∃ n : ℕ, n = 498 ∧ n < 500 ∧ is_perfect_square (T n) ∧ 
    ∀ m : ℕ, m < 500 → is_perfect_square (T m) → m ≤ n) :=
by sorry

end greatest_n_for_perfect_square_T_l1072_107266


namespace floor_sqrt_5_minus_3_l1072_107292

theorem floor_sqrt_5_minus_3 : ⌊Real.sqrt 5 - 3⌋ = -1 := by sorry

end floor_sqrt_5_minus_3_l1072_107292


namespace subset_implies_c_equals_two_l1072_107232

theorem subset_implies_c_equals_two :
  {p : ℝ × ℝ | p.1 + p.2 - 2 = 0 ∧ p.1 - 2*p.2 + 4 = 0} ⊆ {p : ℝ × ℝ | p.2 = 3*p.1 + c} →
  c = 2 :=
by sorry

end subset_implies_c_equals_two_l1072_107232


namespace nonreal_roots_product_l1072_107202

theorem nonreal_roots_product (x : ℂ) : 
  (x^4 - 4*x^3 + 6*x^2 - 4*x = 2047) → 
  (∃ a b : ℂ, a ≠ b ∧ a.im ≠ 0 ∧ b.im ≠ 0 ∧ 
   (x = a ∨ x = b) ∧ (x^4 - 4*x^3 + 6*x^2 - 4*x = 2047) ∧
   (a * b = 257)) := by
sorry

end nonreal_roots_product_l1072_107202


namespace train_meeting_probability_l1072_107218

-- Define the time intervals
def train_arrival_interval : ℝ := 60  -- 60 minutes between 1:00 and 2:00
def alex_arrival_interval : ℝ := 75   -- 75 minutes between 1:00 and 2:15
def train_wait_time : ℝ := 15         -- 15 minutes wait time

-- Define the probability calculation function
def calculate_probability (train_interval : ℝ) (alex_interval : ℝ) (wait_time : ℝ) : ℚ :=
  -- The actual calculation is not implemented, just the type signature
  0

-- Theorem statement
theorem train_meeting_probability :
  calculate_probability train_arrival_interval alex_arrival_interval train_wait_time = 7/40 := by
  sorry

end train_meeting_probability_l1072_107218


namespace smallest_multiple_of_2_3_5_l1072_107251

theorem smallest_multiple_of_2_3_5 : ∀ n : ℕ, n > 0 ∧ 2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n → n ≥ 30 := by
  sorry

end smallest_multiple_of_2_3_5_l1072_107251


namespace perimeter_of_hundred_rectangles_l1072_107209

/-- The perimeter of a shape formed by arranging rectangles edge-to-edge -/
def perimeter_of_arranged_rectangles (n : ℕ) (length width : ℝ) : ℝ :=
  let single_rectangle_perimeter := 2 * (length + width)
  let total_perimeter_without_overlap := n * single_rectangle_perimeter
  let number_of_joins := n - 1
  let overlap_per_join := 2 * width
  total_perimeter_without_overlap - (number_of_joins * overlap_per_join)

/-- Theorem stating that the perimeter of a shape formed by 100 rectangles 
    (each 3 cm by 1 cm) arranged edge-to-edge is 602 cm -/
theorem perimeter_of_hundred_rectangles : 
  perimeter_of_arranged_rectangles 100 3 1 = 602 := by
  sorry

end perimeter_of_hundred_rectangles_l1072_107209


namespace airplane_seats_l1072_107212

theorem airplane_seats : 
  ∀ (total : ℕ),
  (24 : ℕ) + (total / 4 : ℕ) + (2 * total / 3 : ℕ) = total →
  total = 288 := by
sorry

end airplane_seats_l1072_107212


namespace gcd_84_36_l1072_107269

theorem gcd_84_36 : Nat.gcd 84 36 = 12 := by
  sorry

end gcd_84_36_l1072_107269


namespace max_flight_time_l1072_107261

/-- The maximum flight time for a projectile launched under specific conditions -/
theorem max_flight_time (V₀ g : ℝ) (h₁ : V₀ > 0) (h₂ : g > 0) : 
  ∃ (τ : ℝ), τ = (2 * V₀ / g) * (2 / Real.sqrt 12.5) ∧ 
  ∀ (α : ℝ), 0 ≤ α ∧ α ≤ π / 2 ∧ Real.sin (2 * α) ≥ 0.96 → 
  (2 * V₀ * Real.sin α) / g ≤ τ :=
sorry

end max_flight_time_l1072_107261


namespace locus_of_Y_l1072_107253

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a line in 2D space -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Represents a trapezoid -/
structure Trapezoid :=
  (A B C D : Point)

/-- Defines a perpendicular line to the bases of a trapezoid -/
def perpendicularLine (t : Trapezoid) : Line := sorry

/-- Defines a point on a given line -/
def pointOnLine (l : Line) : Point := sorry

/-- Constructs perpendiculars from points to lines -/
def perpendicular (p : Point) (l : Line) : Line := sorry

/-- Finds the intersection of two lines -/
def lineIntersection (l1 l2 : Line) : Point := sorry

/-- Checks if a point is on a line -/
def isPointOnLine (p : Point) (l : Line) : Prop := sorry

/-- Checks if a line divides a segment in a given ratio -/
def dividesSameRatio (l1 l2 : Line) (seg1 seg2 : Point × Point) : Prop := sorry

/-- Main theorem: The locus of point Y is a line perpendicular to the bases -/
theorem locus_of_Y (t : Trapezoid) (l : Line) : 
  ∃ l' : Line, 
    (∀ X : Point, isPointOnLine X l → 
      let BX := Line.mk sorry sorry sorry
      let CX := Line.mk sorry sorry sorry
      let perp1 := perpendicular t.A BX
      let perp2 := perpendicular t.D CX
      let Y := lineIntersection perp1 perp2
      isPointOnLine Y l') ∧ 
    dividesSameRatio l' l (t.A, t.D) (t.B, t.C) := by
  sorry

end locus_of_Y_l1072_107253


namespace power_equation_solution_l1072_107290

theorem power_equation_solution : ∃ x : ℝ, (5^5 * 9^3 : ℝ) = 3 * 15^x ∧ x = 5 := by
  sorry

end power_equation_solution_l1072_107290


namespace inequality_proof_l1072_107231

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a^3 + b^3 + a + b ≥ 4*a*b := by
  sorry

end inequality_proof_l1072_107231


namespace ratio_to_percent_l1072_107286

theorem ratio_to_percent (a b : ℚ) (h : a / b = 2 / 10) : (a / b) * 100 = 20 := by
  sorry

end ratio_to_percent_l1072_107286


namespace kevins_watermelons_l1072_107289

/-- The weight of the first watermelon in pounds -/
def first_watermelon : ℝ := 9.91

/-- The weight of the second watermelon in pounds -/
def second_watermelon : ℝ := 4.11

/-- The total weight of watermelons Kevin bought -/
def total_weight : ℝ := first_watermelon + second_watermelon

/-- Theorem stating that the total weight of watermelons Kevin bought is 14.02 pounds -/
theorem kevins_watermelons : total_weight = 14.02 := by sorry

end kevins_watermelons_l1072_107289


namespace infinite_series_sum_l1072_107240

theorem infinite_series_sum : 
  (∑' n : ℕ, 1 / (n * (n + 3))) = 11 / 18 := by sorry

end infinite_series_sum_l1072_107240


namespace mutual_fund_yield_range_l1072_107258

/-- The range of annual yields for mutual funds increased by 15% --/
def yield_increase_rate : ℝ := 0.15

/-- The number of mutual funds --/
def num_funds : ℕ := 100

/-- The new range of annual yields after the increase --/
def new_range : ℝ := 11500

/-- The original range of annual yields --/
def original_range : ℝ := 10000

theorem mutual_fund_yield_range : 
  (1 + yield_increase_rate) * original_range = new_range := by
  sorry

end mutual_fund_yield_range_l1072_107258


namespace alex_escalator_time_l1072_107229

/-- The time it takes Alex to walk down the non-moving escalator -/
def time_not_moving : ℝ := 75

/-- The time it takes Alex to walk down the moving escalator -/
def time_moving : ℝ := 30

/-- The time it takes Alex to ride the escalator without walking -/
def time_riding : ℝ := 50

theorem alex_escalator_time :
  (time_not_moving * time_moving) / (time_not_moving - time_moving) = time_riding := by
  sorry

end alex_escalator_time_l1072_107229


namespace james_payment_l1072_107297

/-- Calculates James's payment at a restaurant given meal prices and tip percentage. -/
theorem james_payment (james_meal : ℝ) (friend_meal : ℝ) (tip_percentage : ℝ) : 
  james_meal = 16 →
  friend_meal = 14 →
  tip_percentage = 0.2 →
  james_meal + 0.5 * friend_meal + 0.5 * tip_percentage * (james_meal + friend_meal) = 19 :=
by sorry

end james_payment_l1072_107297


namespace greatest_b_value_l1072_107237

theorem greatest_b_value (b : ℝ) : 
  (∀ x : ℝ, -x^2 + 8*x - 15 ≥ 0 → x ≤ 5) ∧ 
  (-5^2 + 8*5 - 15 ≥ 0) := by
  sorry

end greatest_b_value_l1072_107237


namespace intersection_A_B_complement_union_A_B_l1072_107259

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -4 < x ∧ x < 4}
def B : Set ℝ := {x : ℝ | x ≤ 1 ∨ x ≥ 3}

-- Theorem for part (1)
theorem intersection_A_B : A ∩ B = {x : ℝ | (-4 < x ∧ x ≤ 1) ∨ (3 ≤ x ∧ x < 4)} := by
  sorry

-- Theorem for part (2)
theorem complement_union_A_B : (A ∪ B)ᶜ = ∅ := by
  sorry

end intersection_A_B_complement_union_A_B_l1072_107259


namespace yoga_studio_average_weight_l1072_107295

theorem yoga_studio_average_weight 
  (num_men : ℕ) 
  (num_women : ℕ) 
  (avg_weight_men : ℝ) 
  (avg_weight_women : ℝ) 
  (h1 : num_men = 8) 
  (h2 : num_women = 6) 
  (h3 : avg_weight_men = 190) 
  (h4 : avg_weight_women = 120) :
  let total_people := num_men + num_women
  let total_weight := num_men * avg_weight_men + num_women * avg_weight_women
  total_weight / total_people = 160 := by
sorry

end yoga_studio_average_weight_l1072_107295


namespace least_sum_of_equal_multiples_l1072_107239

theorem least_sum_of_equal_multiples (x y z : ℕ+) (h : (4 : ℕ) * x.val = (5 : ℕ) * y.val ∧ (5 : ℕ) * y.val = (6 : ℕ) * z.val) :
  ∃ (a b c : ℕ+), (4 : ℕ) * a.val = (5 : ℕ) * b.val ∧ (5 : ℕ) * b.val = (6 : ℕ) * c.val ∧
    a.val + b.val + c.val ≤ x.val + y.val + z.val ∧ a.val + b.val + c.val = 37 :=
by
  sorry

end least_sum_of_equal_multiples_l1072_107239


namespace inheritance_calculation_l1072_107291

theorem inheritance_calculation (x : ℝ) 
  (h1 : 0.25 * x + 0.1 * x = 15000) : 
  x = 42857 := by
  sorry

end inheritance_calculation_l1072_107291


namespace no_prime_solution_to_equation_l1072_107200

theorem no_prime_solution_to_equation :
  ∀ p q r s t : ℕ, 
    Prime p → Prime q → Prime r → Prime s → Prime t →
    p^2 + q^2 ≠ r^2 + s^2 + t^2 := by
  sorry

end no_prime_solution_to_equation_l1072_107200


namespace marts_income_percentage_l1072_107236

theorem marts_income_percentage (juan tim mart : ℝ) 
  (h1 : mart = 1.3 * tim) 
  (h2 : tim = 0.6 * juan) : 
  mart = 0.78 * juan := by
sorry

end marts_income_percentage_l1072_107236


namespace flower_arrangement_theorem_l1072_107264

/-- The number of ways to arrange flowers of three different hues -/
def flower_arrangements (X : ℕ+) : ℕ :=
  30

/-- Theorem stating that the number of valid flower arrangements is always 30 -/
theorem flower_arrangement_theorem (X : ℕ+) :
  flower_arrangements X = 30 :=
by
  sorry

end flower_arrangement_theorem_l1072_107264


namespace question_mark_solution_l1072_107208

theorem question_mark_solution : ∃! x : ℤ, x + 3699 + 1985 - 2047 = 31111 :=
  sorry

end question_mark_solution_l1072_107208


namespace rational_square_property_l1072_107263

theorem rational_square_property (x y : ℚ) (h : x^5 + y^5 = 2*x^2*y^2) :
  ∃ (z : ℚ), 1 - x*y = z^2 := by
  sorry

end rational_square_property_l1072_107263


namespace quadrilateral_vector_sum_l1072_107249

/-- Given a quadrilateral ABCD in a real inner product space, with M as the intersection of its diagonals,
    prove that for any point O not equal to M, the sum of vectors from O to each vertex
    is equal to four times the vector from O to M. -/
theorem quadrilateral_vector_sum (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (A B C D M O : V) : 
  M ≠ O →  -- O is not equal to M
  (A - C) = (D - B) →  -- M is the midpoint of AC
  (B - D) = (C - A) →  -- M is the midpoint of BD
  (O - A) + (O - B) + (O - C) + (O - D) = 4 • (O - M) := by sorry

end quadrilateral_vector_sum_l1072_107249


namespace twenty_sided_polygon_selection_l1072_107243

/-- A convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  sides : Fin n → Set (Fin n × Fin n)
  convex : sorry -- Additional property to ensure convexity

/-- The condition that two sides have at least k sides between them -/
def HasKSidesBetween (n : ℕ) (k : ℕ) (s₁ s₂ : Fin n × Fin n) : Prop :=
  sorry

/-- The number of ways to choose m sides from an n-sided polygon with k sides between each pair -/
def CountValidSelections (n m k : ℕ) : ℕ :=
  sorry

theorem twenty_sided_polygon_selection :
  CountValidSelections 20 3 2 = 520 :=
sorry

end twenty_sided_polygon_selection_l1072_107243


namespace right_to_left_grouping_l1072_107279

/-- A function that represents the right-to-left grouping evaluation of expressions -/
noncomputable def rightToLeftEval (a b c d : ℝ) : ℝ := a * (b + (c - d))

/-- The theorem stating that the right-to-left grouping evaluation is correct -/
theorem right_to_left_grouping (a b c d : ℝ) :
  rightToLeftEval a b c d = a * (b + c - d) := by sorry

end right_to_left_grouping_l1072_107279


namespace base_8_to_10_367_l1072_107201

-- Define the base-8 number as a list of digits
def base_8_number : List Nat := [3, 6, 7]

-- Define the function to convert base-8 to base-10
def base_8_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

-- Theorem statement
theorem base_8_to_10_367 :
  base_8_to_10 base_8_number = 247 := by sorry

end base_8_to_10_367_l1072_107201


namespace custom_op_nested_result_l1072_107221

/-- Custom binary operation ⊗ -/
def custom_op (x y : ℝ) : ℝ := x^3 - 3*x*y

/-- Theorem stating the result of h ⊗ (h ⊗ h) -/
theorem custom_op_nested_result (h : ℝ) : custom_op h (custom_op h h) = h^3 * (10 - 3*h) := by
  sorry

end custom_op_nested_result_l1072_107221


namespace equation_roots_l1072_107241

theorem equation_roots : ∀ x : ℝ, 3 * (x - 3) = (x - 3)^2 ↔ x = 3 ∨ x = 6 := by sorry

end equation_roots_l1072_107241


namespace max_value_properties_l1072_107254

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / (1 + x) - Real.log x

theorem max_value_properties (x₀ : ℝ) 
  (h₁ : ∀ x > 0, f x ≤ f x₀) 
  (h₂ : x₀ > 0) :
  f x₀ = x₀ ∧ f x₀ < 1/2 := by
  sorry

end max_value_properties_l1072_107254


namespace point_on_y_axis_has_x_zero_l1072_107278

/-- A point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a point lying on the y-axis -/
def lies_on_y_axis (p : Point) : Prop := p.x = 0

/-- Theorem: If a point lies on the y-axis, its x-coordinate is 0 -/
theorem point_on_y_axis_has_x_zero (M : Point) (h : lies_on_y_axis M) : M.x = 0 := by
  sorry

end point_on_y_axis_has_x_zero_l1072_107278


namespace number_of_roses_l1072_107215

theorem number_of_roses (total : ℕ) (rose_lily_diff : ℕ) (tulip_rose_diff : ℕ)
  (h1 : total = 100)
  (h2 : rose_lily_diff = 22)
  (h3 : tulip_rose_diff = 20) :
  ∃ (roses lilies tulips : ℕ),
    roses + lilies + tulips = total ∧
    roses = lilies + rose_lily_diff ∧
    tulips = roses + tulip_rose_diff ∧
    roses = 34 := by
  sorry

end number_of_roses_l1072_107215


namespace passing_mark_is_40_l1072_107213

/-- Represents the exam results of a class -/
structure ExamResults where
  total_students : ℕ
  absent_percentage : ℚ
  failed_percentage : ℚ
  just_passed_percentage : ℚ
  remaining_average : ℚ
  class_average : ℚ
  fail_margin : ℕ

/-- Calculates the passing mark for the exam given the exam results -/
def calculate_passing_mark (results : ExamResults) : ℚ :=
  let absent := results.total_students * results.absent_percentage
  let failed := results.total_students * results.failed_percentage
  let just_passed := results.total_students * results.just_passed_percentage
  let remaining := results.total_students - (absent + failed + just_passed)
  let total_marks := results.class_average * results.total_students
  let remaining_marks := remaining * results.remaining_average
  (total_marks - remaining_marks) / (failed + just_passed) + results.fail_margin

/-- Theorem stating that given the exam results, the passing mark is 40 -/
theorem passing_mark_is_40 (results : ExamResults) 
  (h1 : results.total_students = 100)
  (h2 : results.absent_percentage = 1/5)
  (h3 : results.failed_percentage = 3/10)
  (h4 : results.just_passed_percentage = 1/10)
  (h5 : results.remaining_average = 65)
  (h6 : results.class_average = 36)
  (h7 : results.fail_margin = 20) :
  calculate_passing_mark results = 40 := by
  sorry

end passing_mark_is_40_l1072_107213


namespace angle_sum_is_ninety_degrees_l1072_107216

theorem angle_sum_is_ninety_degrees (α β : Real) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) 
  (h3 : Real.sin α ^ 2 + Real.sin β ^ 2 = Real.sin (α + β)) : 
  α + β = π/2 := by
  sorry

end angle_sum_is_ninety_degrees_l1072_107216


namespace coffee_machine_price_l1072_107267

/-- The original price of a coffee machine given certain conditions -/
theorem coffee_machine_price (discount : ℕ) (payback_days : ℕ) (old_daily_cost new_daily_cost : ℕ) : 
  discount = 20 →
  payback_days = 36 →
  old_daily_cost = 8 →
  new_daily_cost = 3 →
  (payback_days * (old_daily_cost - new_daily_cost)) + discount = 200 :=
by sorry

end coffee_machine_price_l1072_107267


namespace geometric_sequence_fifth_term_l1072_107230

/-- Given a geometric sequence of positive integers with first term 3 and fourth term 240,
    prove that the fifth term is 768. -/
theorem geometric_sequence_fifth_term :
  ∀ (a : ℕ → ℕ),
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 3 →                            -- first term is 3
  a 4 = 240 →                          -- fourth term is 240
  a 5 = 768 :=                         -- conclusion: fifth term is 768
by sorry

end geometric_sequence_fifth_term_l1072_107230


namespace geometric_series_problem_l1072_107224

/-- Given two infinite geometric series with the specified conditions, prove that m = 8 -/
theorem geometric_series_problem (m : ℝ) : 
  let a₁ : ℝ := 18
  let b₁ : ℝ := 6
  let a₂ : ℝ := 18
  let b₂ : ℝ := 6 + m
  let r₁ := b₁ / a₁
  let r₂ := b₂ / a₂
  let S₁ := a₁ / (1 - r₁)
  let S₂ := a₂ / (1 - r₂)
  S₂ = 3 * S₁ → m = 8 := by
sorry


end geometric_series_problem_l1072_107224


namespace scientific_notation_of_0_0000064_l1072_107226

theorem scientific_notation_of_0_0000064 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 0.0000064 = a * (10 : ℝ) ^ n ∧ a = 6.4 ∧ n = -6 := by
  sorry

end scientific_notation_of_0_0000064_l1072_107226


namespace sara_sold_oranges_l1072_107294

/-- Represents the number of oranges Joan picked initially -/
def initial_oranges : ℕ := 37

/-- Represents the number of oranges Joan is left with -/
def remaining_oranges : ℕ := 27

/-- Represents the number of oranges Sara sold -/
def sold_oranges : ℕ := initial_oranges - remaining_oranges

theorem sara_sold_oranges : sold_oranges = 10 := by
  sorry

end sara_sold_oranges_l1072_107294


namespace hem_length_is_three_feet_l1072_107299

/-- The length of a stitch in inches -/
def stitch_length : ℚ := 1/4

/-- The number of stitches Jenna makes per minute -/
def stitches_per_minute : ℕ := 24

/-- The time it takes Jenna to hem her dress in minutes -/
def hemming_time : ℕ := 6

/-- The number of inches in a foot -/
def inches_per_foot : ℕ := 12

/-- The length of the dress's hem in feet -/
def hem_length : ℚ := (stitches_per_minute * hemming_time * stitch_length) / inches_per_foot

theorem hem_length_is_three_feet : hem_length = 3 := by
  sorry

end hem_length_is_three_feet_l1072_107299


namespace circle_equation_l1072_107270

/-- A circle with center (1, -2) and radius 3 -/
structure Circle where
  center : ℝ × ℝ := (1, -2)
  radius : ℝ := 3

/-- A point on the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point lies on the circle -/
def onCircle (c : Circle) (p : Point) : Prop :=
  (p.x - c.center.1)^2 + (p.y - c.center.2)^2 = c.radius^2

theorem circle_equation (c : Circle) (p : Point) :
  onCircle c p ↔ (p.x - 1)^2 + (p.y + 2)^2 = 9 := by
  sorry

end circle_equation_l1072_107270


namespace two_year_increase_l1072_107227

/-- Calculates the final amount after a given number of years with a fixed annual increase rate. -/
def finalAmount (initialValue : ℝ) (increaseRate : ℝ) (years : ℕ) : ℝ :=
  initialValue * (1 + increaseRate) ^ years

/-- Theorem stating that an initial amount of 59,000, increasing by 1/8 of itself annually, 
    will result in 74,671.875 after 2 years. -/
theorem two_year_increase : 
  let initialValue : ℝ := 59000
  let increaseRate : ℝ := 1/8
  let years : ℕ := 2
  finalAmount initialValue increaseRate years = 74671.875 := by
sorry

end two_year_increase_l1072_107227


namespace right_triangle_cosine_sine_l1072_107203

-- Define the right triangle XYZ
def RightTriangleXYZ (X Y Z : ℝ) : Prop :=
  X^2 + Y^2 = Z^2 ∧ X = 8 ∧ Z = 17

-- Theorem statement
theorem right_triangle_cosine_sine 
  (X Y Z : ℝ) (h : RightTriangleXYZ X Y Z) : 
  Real.cos (Real.arccos (X / Z)) = 15 / 17 ∧ Real.sin (Real.arcsin 1) = 1 := by
  sorry

end right_triangle_cosine_sine_l1072_107203


namespace prime_abs_nsquared_minus_6n_minus_27_l1072_107219

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem prime_abs_nsquared_minus_6n_minus_27 (n : ℤ) :
  is_prime (Int.natAbs (n^2 - 6*n - 27)) ↔ n = -4 ∨ n = -2 ∨ n = 8 ∨ n = 10 := by
  sorry

end prime_abs_nsquared_minus_6n_minus_27_l1072_107219


namespace third_person_investment_range_l1072_107252

theorem third_person_investment_range (total : ℝ) (ratio_high_low : ℝ) :
  total = 143 ∧ ratio_high_low = 5 / 3 →
  ∃ (max min : ℝ),
    max = 55 ∧ min = 39 ∧
    ∀ (third : ℝ),
      (∃ (high low : ℝ),
        high + low + third = total ∧
        high / low = ratio_high_low ∧
        high ≥ third ∧ third ≥ low) →
      third ≤ max ∧ third ≥ min :=
by sorry

end third_person_investment_range_l1072_107252


namespace place_left_representation_l1072_107288

/-- Represents a three-digit number -/
def ThreeDigitNumber (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- Represents a two-digit number -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Represents the operation of placing a three-digit number to the left of a two-digit number -/
def PlaceLeft (x y : ℕ) : ℕ := 100 * x + y

theorem place_left_representation (x y : ℕ) 
  (hx : ThreeDigitNumber x) (hy : TwoDigitNumber y) :
  PlaceLeft x y = 100 * x + y :=
by sorry

end place_left_representation_l1072_107288


namespace fraction_zero_implies_x_negative_one_l1072_107271

theorem fraction_zero_implies_x_negative_one (x : ℝ) :
  (x ^ 2 - 1) / (x - 1) = 0 → x = -1 := by
  sorry

end fraction_zero_implies_x_negative_one_l1072_107271


namespace min_value_inequality_l1072_107238

theorem min_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y) * (1/x + 4/y) ≥ 9 ∧
  ((x + y) * (1/x + 4/y) = 9 ↔ y/x = 4*x/y) :=
sorry

end min_value_inequality_l1072_107238


namespace triangle_square_perimeter_l1072_107268

theorem triangle_square_perimeter (a b c : ℝ) (s : ℝ) : 
  a = 5 → b = 12 → c = 13 → 
  (1/2) * a * b = s^2 → 
  4 * s = 4 * Real.sqrt 30 :=
by sorry

end triangle_square_perimeter_l1072_107268


namespace correct_change_amount_and_composition_l1072_107281

def initial_money : ℚ := 20.40
def avocado_prices : List ℚ := [1.50, 2.25, 3.00]
def water_price : ℚ := 1.75
def water_quantity : ℕ := 2
def apple_price : ℚ := 0.75
def apple_quantity : ℕ := 4

def total_cost : ℚ := (List.sum avocado_prices) + (water_price * water_quantity) + (apple_price * apple_quantity)

def change : ℚ := initial_money - total_cost

theorem correct_change_amount_and_composition :
  change = 7.15 ∧
  ∃ (five_dollar : ℕ) (one_dollar : ℕ) (dime : ℕ) (nickel : ℕ),
    five_dollar = 1 ∧
    one_dollar = 2 ∧
    dime = 1 ∧
    nickel = 1 ∧
    5 * five_dollar + one_dollar + 0.1 * dime + 0.05 * nickel = change :=
by sorry

end correct_change_amount_and_composition_l1072_107281


namespace largest_number_in_ratio_l1072_107225

theorem largest_number_in_ratio (a b c d : ℕ) (h_ratio : a * 3 = b * 2 ∧ b * 4 = c * 3 ∧ c * 5 = d * 4) 
  (h_sum : a + b + c + d = 1344) : d = 480 :=
by sorry

end largest_number_in_ratio_l1072_107225


namespace prob_two_aces_full_deck_prob_two_aces_after_two_kings_l1072_107275

/-- Represents a deck of cards with Aces, Kings, and Queens -/
structure Deck :=
  (num_aces : ℕ)
  (num_kings : ℕ)
  (num_queens : ℕ)

/-- Calculates the probability of drawing two Aces from a given deck -/
def prob_two_aces (d : Deck) : ℚ :=
  (d.num_aces.choose 2 : ℚ) / (d.num_aces + d.num_kings + d.num_queens).choose 2

/-- The full deck with 4 each of Aces, Kings, and Queens -/
def full_deck : Deck := ⟨4, 4, 4⟩

/-- The deck after two Kings have been drawn -/
def deck_after_two_kings : Deck := ⟨4, 2, 4⟩

theorem prob_two_aces_full_deck :
  prob_two_aces full_deck = 1 / 11 :=
sorry

theorem prob_two_aces_after_two_kings :
  prob_two_aces deck_after_two_kings = 2 / 15 :=
sorry

end prob_two_aces_full_deck_prob_two_aces_after_two_kings_l1072_107275


namespace days_took_capsules_l1072_107273

/-- The number of days in July -/
def days_in_july : ℕ := 31

/-- The number of days Isla forgot to take capsules -/
def days_forgot : ℕ := 2

/-- Theorem: The number of days Isla took capsules in July is 29 -/
theorem days_took_capsules : days_in_july - days_forgot = 29 := by
  sorry

end days_took_capsules_l1072_107273


namespace correct_operation_l1072_107214

theorem correct_operation (a : ℝ) : 4 * a - a = 3 * a := by
  sorry

end correct_operation_l1072_107214


namespace cyclic_sum_inequality_l1072_107272

theorem cyclic_sum_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_sum : 1 / (a * b) + 1 / (b * c) + 1 / (c * d) + 1 / (d * a) = 1) :
  a * b * c * d + 16 ≥ 8 * Real.sqrt ((a + c) * (1 / a + 1 / c)) +
    8 * Real.sqrt ((b + d) * (1 / b + 1 / d)) := by
  sorry

end cyclic_sum_inequality_l1072_107272


namespace min_value_x_plus_2y_l1072_107233

theorem min_value_x_plus_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x * y^2 = 4) :
  x + 2*y ≥ 3 * (4 : ℝ)^(1/3) ∧ 
  ∃ (x₀ y₀ : ℝ), 0 < x₀ ∧ 0 < y₀ ∧ x₀ * y₀^2 = 4 ∧ x₀ + 2*y₀ = 3 * (4 : ℝ)^(1/3) := by
  sorry

end min_value_x_plus_2y_l1072_107233


namespace min_e_value_l1072_107260

-- Define the circle and points
def Circle : Type := Unit
def Point : Type := Unit

-- Define the diameter of the circle
def diameter : ℝ := 4

-- Define the points
def P : Point := sorry
def Q : Point := sorry
def X : Point := sorry
def Y : Point := sorry
def Z : Point := sorry

-- Define the properties of the points
def is_diameter (p q : Point) : Prop := sorry
def on_semicircle (p : Point) : Prop := sorry
def is_midpoint (x : Point) : Prop := sorry
def distance (p q : Point) : ℝ := sorry
def symmetric_to (z p q : Point) : Prop := sorry

-- Define the intersection points
def A : Point := sorry
def B : Point := sorry

-- Define the length of AB
def e : ℝ := sorry

-- State the theorem
theorem min_e_value (c : Circle) :
  is_diameter P Q →
  on_semicircle X →
  on_semicircle Y →
  is_midpoint X →
  distance P Y = 5 / 4 →
  symmetric_to Z P Q →
  ∃ (min_e : ℝ), (∀ e', e' ≥ min_e) ∧ min_e = 6 - 5 * Real.sqrt 3 :=
sorry

end min_e_value_l1072_107260


namespace exists_point_sum_distances_gt_perimeter_l1072_107274

/-- A convex n-gon in a 2D plane -/
structure ConvexNGon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  is_convex : sorry -- Axiom for convexity

/-- The perimeter of a convex n-gon -/
def perimeter (polygon : ConvexNGon n) : ℝ := sorry

/-- The sum of distances from a point to all vertices of a convex n-gon -/
def sum_distances (polygon : ConvexNGon n) (point : ℝ × ℝ) : ℝ := sorry

/-- For any convex n-gon with n ≥ 7, there exists a point inside the n-gon
    such that the sum of distances from this point to all vertices
    is greater than the perimeter of the n-gon -/
theorem exists_point_sum_distances_gt_perimeter (n : ℕ) (h : n ≥ 7) (polygon : ConvexNGon n) :
  ∃ (point : ℝ × ℝ), sum_distances polygon point > perimeter polygon := by
  sorry

end exists_point_sum_distances_gt_perimeter_l1072_107274


namespace third_degree_polynomial_property_l1072_107207

/-- A third-degree polynomial with real coefficients -/
def ThirdDegreePolynomial : Type := ℝ → ℝ

/-- The property that the absolute value of g at specified points is 15 -/
def has_abs_value_15 (g : ThirdDegreePolynomial) : Prop :=
  |g 1| = 15 ∧ |g 3| = 15 ∧ |g 4| = 15 ∧ |g 5| = 15 ∧ |g 6| = 15 ∧ |g 7| = 15

theorem third_degree_polynomial_property (g : ThirdDegreePolynomial) 
  (h : has_abs_value_15 g) : |g 0| = 645/8 := by
  sorry


end third_degree_polynomial_property_l1072_107207


namespace fraction_value_l1072_107204

theorem fraction_value (a b c d : ℝ) 
  (h1 : a = 4 * b) 
  (h2 : b = 3 * c) 
  (h3 : c = 5 * d) : 
  a * c / (b * d) = 20 := by
  sorry

end fraction_value_l1072_107204


namespace tully_kate_age_ratio_l1072_107220

/-- Represents a person's age --/
structure Person where
  name : String
  current_age : ℕ

/-- Calculates the ratio of two numbers --/
def ratio (a b : ℕ) : ℚ := a / b

theorem tully_kate_age_ratio :
  let tully : Person := { name := "Tully", current_age := 61 }
  let kate : Person := { name := "Kate", current_age := 29 }
  let tully_future_age := tully.current_age + 3
  let kate_future_age := kate.current_age + 3
  ratio tully_future_age kate_future_age = 2 := by
  sorry

end tully_kate_age_ratio_l1072_107220


namespace product_equals_eight_l1072_107206

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, b (n + 1) = r * b n

/-- The main theorem -/
theorem product_equals_eight
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_nonzero : ∀ n, a n ≠ 0)
  (h_condition : a 3 - 2 * (a 6)^2 + 3 * a 7 = 0)
  (h_equal : b 6 = a 6) :
  b 1 * b 7 * b 10 = 8 := by
  sorry

end product_equals_eight_l1072_107206


namespace total_skips_is_33_l1072_107277

/-- Represents the number of skips for each throw -/
structure ThrowSkips :=
  (first : ℕ)
  (second : ℕ)
  (third : ℕ)
  (fourth : ℕ)
  (fifth : ℕ)

/-- Conditions for the stone skipping problem -/
def SkipConditions (t : ThrowSkips) : Prop :=
  t.second = t.first + 2 ∧
  t.third = 2 * t.second ∧
  t.fourth = t.third - 3 ∧
  t.fifth = t.fourth + 1 ∧
  t.fifth = 8

/-- The total number of skips across all throws -/
def TotalSkips (t : ThrowSkips) : ℕ :=
  t.first + t.second + t.third + t.fourth + t.fifth

/-- Theorem stating that the total number of skips is 33 -/
theorem total_skips_is_33 (t : ThrowSkips) (h : SkipConditions t) :
  TotalSkips t = 33 := by
  sorry

end total_skips_is_33_l1072_107277


namespace travel_time_with_reduced_speed_l1072_107244

/-- Proves that the time taken to travel a given distance with reduced speed is as expected -/
theorem travel_time_with_reduced_speed 
  (distance : ℝ) 
  (no_traffic_time : ℝ) 
  (speed_reduction : ℝ) 
  (heavy_traffic_time : ℝ) 
  (h1 : distance = 200) 
  (h2 : no_traffic_time = 4) 
  (h3 : speed_reduction = 10) 
  (h4 : heavy_traffic_time = 5) : 
  heavy_traffic_time = distance / (distance / no_traffic_time - speed_reduction) := by
  sorry

#check travel_time_with_reduced_speed

end travel_time_with_reduced_speed_l1072_107244


namespace willie_stickers_l1072_107247

theorem willie_stickers (initial : ℕ) (given_away : ℕ) (final : ℕ) : 
  initial = 36 → given_away = 7 → final = initial - given_away → final = 29 := by
  sorry

end willie_stickers_l1072_107247


namespace jordan_read_more_than_maxime_l1072_107228

-- Define the number of novels read by each person
def jordan_french : ℕ := 130
def jordan_spanish : ℕ := 20
def alexandre_french : ℕ := jordan_french / 10
def alexandre_spanish : ℕ := 3 * jordan_spanish
def camille_french : ℕ := 2 * alexandre_french
def camille_spanish : ℕ := jordan_spanish / 2

-- Define the total number of French novels read by Jordan, Alexandre, and Camille
def total_french : ℕ := jordan_french + alexandre_french + camille_french

-- Define Maxime's French and Spanish novels
def maxime_french : ℕ := total_french / 2 - 5
def maxime_spanish : ℕ := 2 * camille_spanish

-- Define the total novels read by Jordan and Maxime
def jordan_total : ℕ := jordan_french + jordan_spanish
def maxime_total : ℕ := maxime_french + maxime_spanish

-- Theorem statement
theorem jordan_read_more_than_maxime :
  jordan_total = maxime_total + 51 := by sorry

end jordan_read_more_than_maxime_l1072_107228


namespace polynomial_simplification_l1072_107257

theorem polynomial_simplification (r : ℝ) : 
  (2 * r^3 + r^2 + 4*r - 3) - (r^3 + r^2 + 6*r - 8) = r^3 - 2*r + 5 := by
  sorry

end polynomial_simplification_l1072_107257


namespace unique_solution_l1072_107223

-- Define the function g
def g (x : ℝ) : ℝ := (x - 1)^5 + (x - 1) - 34

-- State the theorem
theorem unique_solution :
  ∃! x : ℝ, g x = 0 ∧ x = 3 :=
by sorry

end unique_solution_l1072_107223


namespace fourth_term_of_geometric_sequence_l1072_107210

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem fourth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_diff : a 2 - a 1 = 2)
  (h_arithmetic : 2 * a 2 = (3 * a 1 + a 3) / 2) :
  a 4 = 27 := by
  sorry

end fourth_term_of_geometric_sequence_l1072_107210


namespace no_integer_solutions_for_equation_l1072_107282

theorem no_integer_solutions_for_equation : 
  ¬ ∃ (x y z : ℤ), 4 * x^2 + 77 * y^2 = 487 * z^2 := by
  sorry

end no_integer_solutions_for_equation_l1072_107282


namespace absolute_value_simplification_l1072_107234

theorem absolute_value_simplification (a b : ℝ) (h1 : a < 0) (h2 : a * b < 0) :
  |b - a + 1| - |a - b - 5| = -4 := by
  sorry

end absolute_value_simplification_l1072_107234


namespace unique_three_digit_number_with_three_divisors_l1072_107293

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def starts_with_three (n : ℕ) : Prop := ∃ k, n = 300 + k ∧ 0 ≤ k ∧ k < 100

def has_exactly_three_divisors (n : ℕ) : Prop := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = 3

theorem unique_three_digit_number_with_three_divisors :
  ∃! n : ℕ, is_three_digit n ∧ starts_with_three n ∧ has_exactly_three_divisors n ∧ n = 361 :=
sorry

end unique_three_digit_number_with_three_divisors_l1072_107293


namespace magnitude_AC_l1072_107283

def vector_AB : Fin 2 → ℝ := ![1, 2]
def vector_BC : Fin 2 → ℝ := ![3, 4]

theorem magnitude_AC : 
  let vector_AC := (vector_BC 0 - (-vector_AB 0), vector_BC 1 - (-vector_AB 1))
  Real.sqrt ((vector_AC.1)^2 + (vector_AC.2)^2) = 2 * Real.sqrt 13 := by
  sorry

end magnitude_AC_l1072_107283


namespace product_remainder_l1072_107265

theorem product_remainder (a b c d : ℕ) (h1 : a = 1729) (h2 : b = 1865) (h3 : c = 1912) (h4 : d = 2023) :
  (a * b * c * d) % 7 = 6 := by
  sorry

end product_remainder_l1072_107265


namespace hyperbola_k_squared_l1072_107262

/-- A hyperbola centered at the origin, opening vertically -/
structure VerticalHyperbola where
  a : ℝ
  b : ℝ

/-- Check if a point (x, y) lies on the hyperbola -/
def VerticalHyperbola.contains (h : VerticalHyperbola) (x y : ℝ) : Prop :=
  y^2 / h.a^2 - x^2 / h.b^2 = 1

/-- The main theorem -/
theorem hyperbola_k_squared (h : VerticalHyperbola) 
  (h1 : h.contains 4 3)
  (h2 : h.contains 0 2)
  (h3 : h.contains 2 k) : k^2 = 17/4 := by
  sorry


end hyperbola_k_squared_l1072_107262


namespace unique_number_exists_l1072_107235

theorem unique_number_exists : ∃! x : ℝ, x / 2 + x + 2 = 62 := by
  sorry

end unique_number_exists_l1072_107235
