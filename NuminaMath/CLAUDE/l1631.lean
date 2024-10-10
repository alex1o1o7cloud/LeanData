import Mathlib

namespace facebook_employee_bonus_l1631_163158

/-- Represents the Facebook employee bonus problem -/
theorem facebook_employee_bonus (
  total_employees : ℕ
  ) (annual_earnings : ℕ) (bonus_percentage : ℚ) (bonus_per_mother : ℕ) :
  total_employees = 3300 →
  annual_earnings = 5000000 →
  bonus_percentage = 1/4 →
  bonus_per_mother = 1250 →
  ∃ (non_mother_employees : ℕ),
    non_mother_employees = 1200 ∧
    non_mother_employees = 
      (2/3 : ℚ) * total_employees - 
      (bonus_percentage * annual_earnings) / bonus_per_mother :=
by sorry


end facebook_employee_bonus_l1631_163158


namespace remainder_theorem_l1631_163141

theorem remainder_theorem (n : ℤ) (h : n % 28 = 15) : (2 * n) % 14 = 2 := by
  sorry

end remainder_theorem_l1631_163141


namespace floor_fraction_difference_l1631_163106

theorem floor_fraction_difference (n : ℕ) (hn : n = 2009) : 
  ⌊((n + 1)^2 : ℝ) / ((n - 1) * n) - (n - 1)^2 / (n * (n + 1))⌋ = 6 := by
  sorry

end floor_fraction_difference_l1631_163106


namespace equation_solutions_l1631_163166

theorem equation_solutions :
  ∀ x y : ℤ, y ≥ 0 → (24 * y + 1 = (4 * y^2 - x^2)^2) →
    ((x = 1 ∨ x = -1) ∧ y = 0) ∨
    ((x = 3 ∨ x = -3) ∧ y = 1) ∨
    ((x = 3 ∨ x = -3) ∧ y = 2) :=
by sorry

end equation_solutions_l1631_163166


namespace parabola_equation_fixed_point_property_l1631_163132

-- Define the ellipse E
def E : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 / 9 + p.2^2 / 8 = 1}

-- Define the parabola C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 4 * p.1}

-- Define the right focus of the ellipse E
def right_focus_E : ℝ × ℝ := (1, 0)

-- Define the directrix of parabola C
def directrix_C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = -1}

-- Theorem for part (I)
theorem parabola_equation : 
  C = {p : ℝ × ℝ | p.2^2 = 4 * p.1} := by sorry

-- Theorem for part (II)
theorem fixed_point_property (P Q : ℝ × ℝ) 
  (hP : P ∈ C) (hQ : Q ∈ C) (hO : P ≠ (0, 0) ∧ Q ≠ (0, 0)) 
  (hPerp : (P.1 * Q.1 + P.2 * Q.2 = 0)) :
  ∃ (m n : ℝ), m * P.2 = P.1 + n ∧ m * Q.2 = Q.1 + n ∧ n = -4 := by sorry

end parabola_equation_fixed_point_property_l1631_163132


namespace grid_coverage_possible_specific_case_101x101_l1631_163160

/-- Represents a square stamp with black cells -/
structure Stamp :=
  (size : ℕ)
  (black_cells : ℕ)

/-- Represents a square grid -/
structure Grid :=
  (size : ℕ)

/-- Predicate to check if a grid can be covered by a stamp, leaving one corner uncovered -/
def can_cover (s : Stamp) (g : Grid) (num_stamps : ℕ) : Prop :=
  ∃ (N : ℕ), 
    g.size = 2*N + 1 ∧ 
    s.size = 2*N ∧ 
    s.black_cells = 4*N + 2 ∧ 
    num_stamps = 4*N

/-- Theorem stating that it's possible to cover a (2N+1) x (2N+1) grid with a 2N x 2N stamp -/
theorem grid_coverage_possible :
  ∀ (N : ℕ), N > 0 → 
    let s : Stamp := ⟨2*N, 4*N + 2⟩
    let g : Grid := ⟨2*N + 1⟩
    can_cover s g (4*N) :=
by
  sorry

/-- The specific case for the 101 x 101 grid with 102 black cells on the stamp -/
theorem specific_case_101x101 :
  let s : Stamp := ⟨100, 102⟩
  let g : Grid := ⟨101⟩
  can_cover s g 100 :=
by
  sorry

end grid_coverage_possible_specific_case_101x101_l1631_163160


namespace sarahs_bowling_score_l1631_163177

theorem sarahs_bowling_score (g s : ℕ) : 
  s = g + 50 → (s + g) / 2 = 95 → s = 120 := by
sorry

end sarahs_bowling_score_l1631_163177


namespace natalie_bushes_needed_l1631_163122

/-- Represents the number of containers of blueberries per bush -/
def containers_per_bush : ℕ := 10

/-- Represents the number of containers needed to trade for one cabbage -/
def containers_per_cabbage : ℕ := 4

/-- Represents the number of cabbages Natalie wants to obtain -/
def target_cabbages : ℕ := 20

/-- Calculates the number of bushes needed to obtain a given number of cabbages -/
def bushes_needed (cabbages : ℕ) : ℕ :=
  (cabbages * containers_per_cabbage) / containers_per_bush

theorem natalie_bushes_needed : bushes_needed target_cabbages = 8 := by
  sorry

end natalie_bushes_needed_l1631_163122


namespace probability_six_heads_twelve_flips_l1631_163120

/-- The probability of getting exactly 6 heads when flipping a fair coin 12 times -/
theorem probability_six_heads_twelve_flips : 
  (Nat.choose 12 6 : ℚ) / 2^12 = 231 / 1024 := by sorry

end probability_six_heads_twelve_flips_l1631_163120


namespace triangle_side_inequality_l1631_163119

theorem triangle_side_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (a^2 + c^2) / b^2 ≥ 1 ∧ ∃ (a' b' c' : ℝ), 0 < a' ∧ 0 < b' ∧ 0 < c' ∧ a' + b' > c' ∧ b' + c' > a' ∧ c' + a' > b' ∧ (a'^2 + c'^2) / b'^2 = 1 := by
  sorry

end triangle_side_inequality_l1631_163119


namespace expression_zero_iff_x_one_or_three_l1631_163103

theorem expression_zero_iff_x_one_or_three (x : ℝ) :
  x ≠ 0 →
  (x^2 - 4*x + 3) / (5*x) = 0 ↔ x = 1 ∨ x = 3 := by
  sorry

end expression_zero_iff_x_one_or_three_l1631_163103


namespace no_valid_two_digit_number_l1631_163181

theorem no_valid_two_digit_number : ¬ ∃ (N : ℕ), 
  (10 ≤ N ∧ N < 100) ∧ 
  (∃ (x : ℕ), 
    x > 3 ∧ 
    N - (10 * (N % 10) + N / 10) = x^3) :=
sorry

end no_valid_two_digit_number_l1631_163181


namespace complementary_implies_mutually_exclusive_exists_mutually_exclusive_not_complementary_l1631_163175

variable {Ω : Type} [MeasurableSpace Ω]
variable (A₁ A₂ : Set Ω)

-- Define mutually exclusive events
def mutually_exclusive (A₁ A₂ : Set Ω) : Prop := A₁ ∩ A₂ = ∅

-- Define complementary events
def complementary (A₁ A₂ : Set Ω) : Prop := A₁ ∪ A₂ = Ω ∧ A₁ ∩ A₂ = ∅

-- Theorem 1: Complementary events are mutually exclusive
theorem complementary_implies_mutually_exclusive :
  complementary A₁ A₂ → mutually_exclusive A₁ A₂ := by sorry

-- Theorem 2: Existence of mutually exclusive events that are not complementary
theorem exists_mutually_exclusive_not_complementary :
  ∃ A₁ A₂ : Set Ω, mutually_exclusive A₁ A₂ ∧ ¬complementary A₁ A₂ := by sorry

end complementary_implies_mutually_exclusive_exists_mutually_exclusive_not_complementary_l1631_163175


namespace product_of_roots_l1631_163105

theorem product_of_roots (x : ℝ) : (x - 1) * (x + 4) = 22 → ∃ y : ℝ, (x - 1) * (x + 4) = 22 ∧ (y - 1) * (y + 4) = 22 ∧ x * y = -26 := by
  sorry

end product_of_roots_l1631_163105


namespace horner_v3_equals_16_l1631_163125

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 7x^7 + 5x^5 + 4x^4 + 2x^2 + x + 2 -/
def f : List ℝ := [7, 0, 5, 4, 2, 1, 2]

/-- v_3 is the fourth intermediate value in Horner's method -/
def v_3 (coeffs : List ℝ) (x : ℝ) : ℝ :=
  (coeffs.take 4).foldl (fun acc a => acc * x + a) 0

theorem horner_v3_equals_16 :
  v_3 f 1 = 16 := by sorry

end horner_v3_equals_16_l1631_163125


namespace sqrt_square_eq_x_l1631_163169

theorem sqrt_square_eq_x (x : ℝ) (h : x ≥ 0) : (Real.sqrt x)^2 = x := by sorry

end sqrt_square_eq_x_l1631_163169


namespace remainder_yards_value_l1631_163185

/-- The number of half-marathons Jacob has run -/
def num_half_marathons : ℕ := 15

/-- The length of a half-marathon in miles -/
def half_marathon_miles : ℕ := 13

/-- The additional length of a half-marathon in yards -/
def half_marathon_extra_yards : ℕ := 193

/-- The number of yards in a mile -/
def yards_per_mile : ℕ := 1760

/-- The total distance Jacob has run in yards -/
def total_distance_yards : ℕ :=
  num_half_marathons * (half_marathon_miles * yards_per_mile + half_marathon_extra_yards)

/-- The remainder y in yards when the total distance is expressed as m miles and y yards -/
def remainder_yards : ℕ := total_distance_yards % yards_per_mile

theorem remainder_yards_value : remainder_yards = 1135 := by
  sorry

end remainder_yards_value_l1631_163185


namespace construct_triangle_from_excenters_l1631_163107

-- Define the basic types and structures
structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the concept of an excenter
def is_excenter (P : Point) (T : Triangle) : Prop :=
  sorry -- Definition of excenter

-- Define the concept of altitude foot
def is_altitude_foot (P : Point) (T : Triangle) : Prop :=
  sorry -- Definition of altitude foot

-- Theorem statement
theorem construct_triangle_from_excenters 
  (A₁ B₁ C₁ : Point) 
  (h_excenters : is_excenter A₁ T ∧ is_excenter B₁ T ∧ is_excenter C₁ T) :
  ∃ (T : Triangle),
    is_altitude_foot T.A (Triangle.mk A₁ B₁ C₁) ∧
    is_altitude_foot T.B (Triangle.mk A₁ B₁ C₁) ∧
    is_altitude_foot T.C (Triangle.mk A₁ B₁ C₁) :=
by
  sorry

end construct_triangle_from_excenters_l1631_163107


namespace triangle_equation_l1631_163188

/-- A non-isosceles triangle with side lengths a, b, c opposite to angles A, B, C respectively,
    where A, B, C form an arithmetic sequence. -/
structure NonIsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ
  nonIsosceles : a ≠ b ∧ b ≠ c ∧ a ≠ c
  oppositeAngles : angleA.cos = (b^2 + c^2 - a^2) / (2*b*c) ∧
                   angleB.cos = (a^2 + c^2 - b^2) / (2*a*c) ∧
                   angleC.cos = (a^2 + b^2 - c^2) / (2*a*b)
  arithmeticSequence : ∃ (d : ℝ), angleB = angleA + d ∧ angleC = angleB + d

/-- The main theorem stating the equation holds for non-isosceles triangles with angles
    in arithmetic sequence. -/
theorem triangle_equation (t : NonIsoscelesTriangle) :
  1 / (t.a - t.b) + 1 / (t.c - t.b) = 3 / (t.a - t.b + t.c) := by
  sorry

end triangle_equation_l1631_163188


namespace exactly_one_double_digit_sum_two_l1631_163173

/-- Sum of digits function -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Predicate for two-digit numbers -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The main theorem -/
theorem exactly_one_double_digit_sum_two :
  ∃! x : ℕ, is_two_digit x ∧ digit_sum (digit_sum x) = 2 := by sorry

end exactly_one_double_digit_sum_two_l1631_163173


namespace absolute_value_inequality_solution_set_l1631_163142

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2 - x| < 1} = Set.Ioo 1 3 := by sorry

end absolute_value_inequality_solution_set_l1631_163142


namespace badminton_tournament_matches_l1631_163101

theorem badminton_tournament_matches (n : ℕ) (h : n = 6) : 
  (n * (n - 1)) / 2 = 15 := by
  sorry

end badminton_tournament_matches_l1631_163101


namespace min_a6_geometric_sequence_l1631_163194

theorem min_a6_geometric_sequence (a : ℕ → ℕ) (q : ℚ) :
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 6 → a n > 0) →
  (∀ n : ℕ, 1 ≤ n ∧ n < 6 → a (n + 1) = (a n : ℚ) * q) →
  1 < q ∧ q < 2 →
  243 ≤ a 6 :=
by sorry

end min_a6_geometric_sequence_l1631_163194


namespace fraction_power_product_l1631_163190

theorem fraction_power_product :
  (1 / 3 : ℚ) ^ 4 * (1 / 5 : ℚ) = 1 / 405 := by
  sorry

end fraction_power_product_l1631_163190


namespace runner_problem_l1631_163157

theorem runner_problem (v : ℝ) (h1 : v > 0) :
  (40 / v = 20 / v + 4) →
  (40 / (v / 2) = 8) :=
by sorry

end runner_problem_l1631_163157


namespace rectangular_prism_volume_l1631_163140

/-- 
Given a rectangular prism with edges in the ratio 2:1:1.5 and a total edge length of 72 cm,
prove that its volume is 192 cubic centimeters.
-/
theorem rectangular_prism_volume (x : ℝ) 
  (h1 : x > 0)
  (h2 : 4 * (2*x) + 4 * x + 4 * (1.5*x) = 72) : 
  (2*x) * x * (1.5*x) = 192 := by
  sorry

end rectangular_prism_volume_l1631_163140


namespace smallest_dual_base_representation_l1631_163165

theorem smallest_dual_base_representation : ∃ (c d : ℕ), 
  c > 3 ∧ d > 3 ∧ 
  3 * c + 4 = 19 ∧ 
  4 * d + 3 = 19 ∧
  (∀ (x c' d' : ℕ), c' > 3 → d' > 3 → 3 * c' + 4 = x → 4 * d' + 3 = x → x ≥ 19) := by
  sorry

end smallest_dual_base_representation_l1631_163165


namespace simplify_expression_l1631_163128

theorem simplify_expression (x : ℝ) 
  (h1 : x ≠ 1) 
  (h2 : x ≠ -1) 
  (h3 : x ≠ (-1 + Real.sqrt 5) / 2) 
  (h4 : x ≠ (-1 - Real.sqrt 5) / 2) : 
  1 - (1 / (1 + x / (x^2 - 1))) = x / (x^2 + x - 1) := by
  sorry

end simplify_expression_l1631_163128


namespace max_students_for_given_supplies_l1631_163187

/-- The maximum number of students among whom pens and pencils can be distributed equally -/
def max_students (pens : ℕ) (pencils : ℕ) : ℕ :=
  Nat.gcd pens pencils

/-- Theorem stating that the GCD of 1048 and 828 is the maximum number of students -/
theorem max_students_for_given_supplies : 
  max_students 1048 828 = 4 := by sorry

end max_students_for_given_supplies_l1631_163187


namespace project_duration_proof_l1631_163186

/-- The original duration of the project in months -/
def original_duration : ℝ := 30

/-- The reduction in project duration when efficiency is increased -/
def duration_reduction : ℝ := 6

/-- The factor by which efficiency is increased -/
def efficiency_increase : ℝ := 1.25

theorem project_duration_proof :
  (original_duration - duration_reduction) / original_duration = 1 / efficiency_increase :=
by sorry

#check project_duration_proof

end project_duration_proof_l1631_163186


namespace cube_sum_from_sum_and_square_sum_l1631_163167

theorem cube_sum_from_sum_and_square_sum (x y : ℝ) 
  (h1 : x + y = 5) 
  (h2 : x^2 + y^2 = 13) : 
  x^3 + y^3 = 35 := by
sorry

end cube_sum_from_sum_and_square_sum_l1631_163167


namespace relay_race_assignments_l1631_163147

theorem relay_race_assignments (n : ℕ) (k : ℕ) (h1 : n = 15) (h2 : k = 4) :
  (n.factorial / (n - k).factorial : ℕ) = 32760 := by
  sorry

end relay_race_assignments_l1631_163147


namespace range_of_f_l1631_163163

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then |x| - 1 else Real.sin x ^ 2

theorem range_of_f :
  Set.range f = Set.Ioi (-1) := by sorry

end range_of_f_l1631_163163


namespace apple_eating_contest_difference_l1631_163126

/-- Represents the result of an apple eating contest -/
structure ContestResult where
  numStudents : Nat
  applesCounts : List Nat
  maxEater : Nat
  minEater : Nat

/-- Theorem stating the difference between the maximum and minimum number of apples eaten -/
theorem apple_eating_contest_difference (result : ContestResult)
  (h1 : result.numStudents = 8)
  (h2 : result.applesCounts.length = result.numStudents)
  (h3 : result.maxEater ∈ result.applesCounts)
  (h4 : result.minEater ∈ result.applesCounts)
  (h5 : ∀ x ∈ result.applesCounts, x ≤ result.maxEater ∧ x ≥ result.minEater) :
  result.maxEater - result.minEater = 8 :=
by sorry

end apple_eating_contest_difference_l1631_163126


namespace power_of_eight_mod_hundred_l1631_163182

theorem power_of_eight_mod_hundred : 8^2050 % 100 = 24 := by sorry

end power_of_eight_mod_hundred_l1631_163182


namespace identity_not_T_function_exponential_T_function_cosine_T_function_iff_l1631_163195

-- Definition of a "T function"
def is_T_function (f : ℝ → ℝ) : Prop :=
  ∃ T : ℝ, T ≠ 0 ∧ ∀ x : ℝ, f (x + T) = T * f x

-- Statement 1
theorem identity_not_T_function :
  ¬ is_T_function (λ x : ℝ => x) := by sorry

-- Statement 2
theorem exponential_T_function (a : ℝ) (ha : a > 0 ∧ a ≠ 1) :
  (∃ x : ℝ, a^x = x) → is_T_function (λ x : ℝ => a^x) := by sorry

-- Statement 3
theorem cosine_T_function_iff (m : ℝ) :
  is_T_function (λ x : ℝ => Real.cos (m * x)) ↔ ∃ k : ℤ, m = k * Real.pi := by sorry

end identity_not_T_function_exponential_T_function_cosine_T_function_iff_l1631_163195


namespace set_operations_l1631_163180

-- Define the sets
def U : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}
def M : Set ℝ := {x | -1 < x ∧ x < 1}
def C_U_N : Set ℝ := {x | 0 < x ∧ x < 2}

-- Define N (this is what we need to prove)
def N : Set ℝ := {x | (-3 ≤ x ∧ x ≤ 0) ∨ (2 ≤ x ∧ x ≤ 3)}

-- State the theorem
theorem set_operations :
  (N = {x : ℝ | (-3 ≤ x ∧ x ≤ 0) ∨ (2 ≤ x ∧ x ≤ 3)}) ∧
  (M ∩ C_U_N = {x : ℝ | 0 < x ∧ x < 1}) ∧
  (M ∪ N = {x : ℝ | (-3 ≤ x ∧ x < 1) ∨ (2 ≤ x ∧ x ≤ 3)}) :=
by sorry

end set_operations_l1631_163180


namespace high_school_total_students_l1631_163155

/-- Represents a high school with three grades in its senior section. -/
structure HighSchool where
  first_grade : ℕ
  second_grade : ℕ
  third_grade : ℕ

/-- Represents a stratified sample from the high school. -/
structure Sample where
  total : ℕ
  first_grade : ℕ
  second_grade : ℕ
  third_grade : ℕ

/-- The total number of students in the high school section. -/
def total_students (hs : HighSchool) : ℕ :=
  hs.first_grade + hs.second_grade + hs.third_grade

/-- The theorem stating the total number of students in the high school section. -/
theorem high_school_total_students 
  (hs : HighSchool)
  (sample : Sample)
  (h1 : hs.first_grade = 400)
  (h2 : sample.total = 45)
  (h3 : sample.second_grade = 15)
  (h4 : sample.third_grade = 10)
  (h5 : sample.first_grade = sample.total - sample.second_grade - sample.third_grade)
  (h6 : sample.first_grade * hs.first_grade = sample.total * 20) :
  total_students hs = 900 := by
  sorry


end high_school_total_students_l1631_163155


namespace paving_cost_calculation_l1631_163154

-- Define the room dimensions and paving rate
def room_length : ℝ := 5.5
def room_width : ℝ := 4
def paving_rate : ℝ := 750

-- Define the function to calculate the cost of paving
def paving_cost (length width rate : ℝ) : ℝ :=
  length * width * rate

-- State the theorem
theorem paving_cost_calculation :
  paving_cost room_length room_width paving_rate = 16500 := by
  sorry

end paving_cost_calculation_l1631_163154


namespace sales_tax_difference_l1631_163135

-- Define the item price
def item_price : ℝ := 20

-- Define the two tax rates
def tax_rate_1 : ℝ := 0.065
def tax_rate_2 : ℝ := 0.06

-- State the theorem
theorem sales_tax_difference : 
  (tax_rate_1 - tax_rate_2) * item_price = 0.1 := by
  sorry

end sales_tax_difference_l1631_163135


namespace special_polygon_area_l1631_163115

/-- A polygon with 32 congruent sides, where each side is perpendicular to its adjacent sides -/
structure SpecialPolygon where
  sides : ℕ
  side_length : ℝ
  perimeter : ℝ
  sides_eq : sides = 32
  perimeter_eq : perimeter = 64
  perimeter_calc : perimeter = sides * side_length

/-- The area of the special polygon -/
def polygon_area (p : SpecialPolygon) : ℝ :=
  36 * p.side_length ^ 2

theorem special_polygon_area (p : SpecialPolygon) : polygon_area p = 144 := by
  sorry

end special_polygon_area_l1631_163115


namespace parabola_line_theorem_l1631_163134

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 2px -/
structure Parabola where
  p : ℝ

/-- Checks if a point lies on a given parabola -/
def isOnParabola (point : Point) (parabola : Parabola) : Prop :=
  point.y^2 = 2 * parabola.p * point.x

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point is the centroid of a triangle -/
def isCentroid (centroid : Point) (p1 p2 p3 : Point) : Prop :=
  centroid.x = (p1.x + p2.x + p3.x) / 3 ∧
  centroid.y = (p1.y + p2.y + p3.y) / 3

theorem parabola_line_theorem (parabola : Parabola) 
    (A B C F : Point) : 
    isOnParabola A parabola → 
    isOnParabola B parabola → 
    isOnParabola C parabola → 
    A.x = 1 → 
    A.y = 2 → 
    F.x = parabola.p → 
    F.y = 0 → 
    isCentroid F A B C → 
    ∃ (line : Line), 
      line.a = 2 ∧ 
      line.b = 1 ∧ 
      line.c = -1 := by
  sorry

end parabola_line_theorem_l1631_163134


namespace min_square_edge_for_circle_l1631_163112

-- Define the circumference of the circle
def circle_circumference : ℝ := 31.4

-- Define π as an approximation
def π : ℝ := 3.14

-- Define the theorem
theorem min_square_edge_for_circle :
  ∃ (edge_length : ℝ), 
    edge_length = circle_circumference / π ∧ 
    edge_length = 10 := by sorry

end min_square_edge_for_circle_l1631_163112


namespace map_scale_l1631_163139

/-- Given a map scale where 15 cm represents 90 km, 
    prove that 20 cm on the map represents 120 km in reality. -/
theorem map_scale (scale : ℝ → ℝ) 
  (h1 : scale 15 = 90) -- 15 cm on map represents 90 km in reality
  (h2 : ∀ x : ℝ, scale x = (x / 15) * 90) -- scale is linear
  : scale 20 = 120 := by
  sorry

end map_scale_l1631_163139


namespace solution_set_quadratic_inequality_l1631_163151

theorem solution_set_quadratic_inequality (a : ℝ) :
  {x : ℝ | x^2 - 2*a + a^2 - 1 < 0} = {x : ℝ | a - 1 < x ∧ x < a + 1} := by
  sorry

end solution_set_quadratic_inequality_l1631_163151


namespace t_shape_perimeter_l1631_163149

/-- Calculates the perimeter of a T-shaped figure formed by two rectangles -/
def t_perimeter (rect1_width rect1_height rect2_width rect2_height overlap : ℕ) : ℕ :=
  2 * (rect1_width + rect1_height) + 2 * (rect2_width + rect2_height) - 2 * overlap

/-- The perimeter of the T-shaped figure is 26 inches -/
theorem t_shape_perimeter : t_perimeter 3 5 2 5 2 = 26 := by
  sorry

end t_shape_perimeter_l1631_163149


namespace tangent_line_to_x_ln_x_l1631_163109

theorem tangent_line_to_x_ln_x (m : ℝ) : 
  (∃ x₀ : ℝ, x₀ > 0 ∧ 
    (∀ x : ℝ, 2 * x + m = x₀ * Real.log x₀ + (Real.log x₀ + 1) * (x - x₀)) ∧
    (∀ x : ℝ, x > 0 → 2 * x + m ≥ x * Real.log x)) →
  m = -Real.exp 1 := by
sorry

end tangent_line_to_x_ln_x_l1631_163109


namespace mother_daughter_age_difference_l1631_163168

theorem mother_daughter_age_difference :
  ∀ (mother_age daughter_age : ℕ),
    mother_age = 55 →
    mother_age - 1 = 2 * (daughter_age - 1) →
    mother_age - daughter_age = 27 :=
by
  sorry

end mother_daughter_age_difference_l1631_163168


namespace condition_necessary_not_sufficient_l1631_163118

-- Define the equation
def is_ellipse (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m - 2) + y^2 / (6 - m) = 1 ∧ 
  m - 2 > 0 ∧ 6 - m > 0 ∧ m - 2 ≠ 6 - m

-- Define the condition
def condition (m : ℝ) : Prop :=
  2 < m ∧ m < 6

-- Theorem stating that the condition is necessary but not sufficient
theorem condition_necessary_not_sufficient :
  (∀ m : ℝ, is_ellipse m → condition m) ∧
  ¬(∀ m : ℝ, condition m → is_ellipse m) :=
sorry

end condition_necessary_not_sufficient_l1631_163118


namespace import_tax_calculation_l1631_163179

theorem import_tax_calculation (tax_rate : ℝ) (tax_threshold : ℝ) (tax_paid : ℝ) (total_value : ℝ) : 
  tax_rate = 0.07 →
  tax_threshold = 1000 →
  tax_paid = 87.50 →
  tax_rate * (total_value - tax_threshold) = tax_paid →
  total_value = 2250 := by
sorry

end import_tax_calculation_l1631_163179


namespace average_theorem_l1631_163113

theorem average_theorem (a b c : ℝ) :
  (a + b + c) / 3 = 12 →
  ((2*a + 1) + (2*b + 2) + (2*c + 3) + 2) / 4 = 20 := by
sorry

end average_theorem_l1631_163113


namespace unique_solution_iff_a_equals_one_l1631_163176

-- Define the equation
def equation (a x : ℝ) : Prop :=
  3^(x^2 - 2*a*x + a^2) = a*x^2 - 2*a^2*x + a^3 + a^2 - 4*a + 4

-- Define the property of having exactly one solution
def has_exactly_one_solution (a : ℝ) : Prop :=
  ∃! x, equation a x

-- Theorem statement
theorem unique_solution_iff_a_equals_one :
  ∀ a : ℝ, has_exactly_one_solution a ↔ a = 1 := by sorry

end unique_solution_iff_a_equals_one_l1631_163176


namespace opposite_of_2023_l1631_163156

theorem opposite_of_2023 : 
  ∃ x : ℤ, (2023 + x = 0) ∧ (x = -2023) := by
  sorry

end opposite_of_2023_l1631_163156


namespace nell_baseball_cards_l1631_163111

/-- Nell's baseball card collection problem -/
theorem nell_baseball_cards :
  ∀ (initial_cards given_cards remaining_cards : ℕ),
  given_cards = 28 →
  remaining_cards = 276 →
  initial_cards = given_cards + remaining_cards →
  initial_cards = 304 :=
by
  sorry

end nell_baseball_cards_l1631_163111


namespace root_sum_squares_l1631_163121

theorem root_sum_squares (p q r : ℝ) : 
  (p + q + r = 15) → (p * q + q * r + r * p = 25) → 
  (p + q)^2 + (q + r)^2 + (r + p)^2 = 400 := by
sorry

end root_sum_squares_l1631_163121


namespace smallest_factorial_divisible_by_7875_l1631_163152

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem smallest_factorial_divisible_by_7875 :
  ∃ (n : ℕ), (n > 0) ∧ (is_factor 7875 (Nat.factorial n)) ∧
  (∀ (m : ℕ), m > 0 → m < n → ¬(is_factor 7875 (Nat.factorial m))) ∧
  n = 15 := by
sorry

end smallest_factorial_divisible_by_7875_l1631_163152


namespace product_zero_l1631_163189

theorem product_zero (r : ℂ) (h1 : r^4 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) = 0 := by
  sorry

end product_zero_l1631_163189


namespace parabola_axis_of_symmetry_specific_parabola_axis_of_symmetry_l1631_163146

/-- The axis of symmetry of a parabola y = (x - h)^2 + k is the line x = h -/
theorem parabola_axis_of_symmetry (h k : ℝ) :
  let f : ℝ → ℝ := λ x => (x - h)^2 + k
  ∀ x, f (h + x) = f (h - x) :=
by sorry

/-- The axis of symmetry of the parabola y = (x - 1)^2 + 3 is the line x = 1 -/
theorem specific_parabola_axis_of_symmetry :
  let f : ℝ → ℝ := λ x => (x - 1)^2 + 3
  ∀ x, f (1 + x) = f (1 - x) :=
by sorry

end parabola_axis_of_symmetry_specific_parabola_axis_of_symmetry_l1631_163146


namespace parallel_lines_coincident_lines_perpendicular_lines_l1631_163159

-- Define the lines l₁ and l₂
def l₁ (m : ℚ) : ℚ → ℚ → Prop := λ x y => (m + 3) * x + 4 * y = 5 - 3 * m
def l₂ (m : ℚ) : ℚ → ℚ → Prop := λ x y => 2 * x + (m + 5) * y = 8

-- Define parallel lines
def parallel (l₁ l₂ : ℚ → ℚ → Prop) : Prop :=
  ∃ k : ℚ, ∀ x y, l₁ x y ↔ l₂ (k * x) (k * y)

-- Define coincident lines
def coincident (l₁ l₂ : ℚ → ℚ → Prop) : Prop :=
  ∀ x y, l₁ x y ↔ l₂ x y

-- Define perpendicular lines
def perpendicular (l₁ l₂ : ℚ → ℚ → Prop) : Prop :=
  ∃ k : ℚ, ∀ x y, l₁ x y → l₂ y (-x)

-- Theorem statements
theorem parallel_lines : parallel (l₁ (-7)) (l₂ (-7)) := sorry

theorem coincident_lines : coincident (l₁ (-1)) (l₂ (-1)) := sorry

theorem perpendicular_lines : perpendicular (l₁ (-13/3)) (l₂ (-13/3)) := sorry

end parallel_lines_coincident_lines_perpendicular_lines_l1631_163159


namespace problem_solution_l1631_163174

theorem problem_solution (x y : ℤ) : 
  y = 3 * x^2 ∧ 
  (2 * x : ℚ) / 5 = 1 / (1 - 2 / (3 + 1 / (4 - 5 / (6 - x)))) → 
  y = 147 := by
  sorry

end problem_solution_l1631_163174


namespace line_intersects_ellipse_l1631_163114

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 5 + y^2 / 4 = 1

-- Define the line
def line (m x y : ℝ) : Prop := m * x + y + m - 1 = 0

-- Theorem statement
theorem line_intersects_ellipse (m : ℝ) : 
  ∃ (x y : ℝ), ellipse x y ∧ line m x y :=
sorry

end line_intersects_ellipse_l1631_163114


namespace eulers_conjecture_counterexample_l1631_163123

theorem eulers_conjecture_counterexample : 133^5 + 110^5 + 84^5 + 27^5 = 144^5 := by
  sorry

end eulers_conjecture_counterexample_l1631_163123


namespace product_zero_l1631_163183

theorem product_zero (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 125) : a * b = 0 := by
  sorry

end product_zero_l1631_163183


namespace division_problem_l1631_163199

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) :
  dividend = 12401 →
  divisor = 163 →
  remainder = 13 →
  dividend = divisor * quotient + remainder →
  quotient = 76 := by
sorry

end division_problem_l1631_163199


namespace hyperbola_midpoint_existence_l1631_163127

theorem hyperbola_midpoint_existence :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁^2 - y₁^2/9 = 1) ∧
    (x₂^2 - y₂^2/9 = 1) ∧
    ((x₁ + x₂)/2 = -1) ∧
    ((y₁ + y₂)/2 = -4) :=
by sorry

end hyperbola_midpoint_existence_l1631_163127


namespace morning_snowfall_l1631_163129

theorem morning_snowfall (total : ℝ) (afternoon : ℝ) (h1 : total = 0.63) (h2 : afternoon = 0.5) :
  total - afternoon = 0.13 := by
  sorry

end morning_snowfall_l1631_163129


namespace first_native_is_liar_and_path_is_incorrect_l1631_163148

-- Define the types of natives
inductive NativeType
| Truthful
| Liar

-- Define a native
structure Native where
  type : NativeType

-- Define the claim about being a Liar
def claimToBeLiar (n : Native) : Prop :=
  match n.type with
  | NativeType.Truthful => False
  | NativeType.Liar => False

-- Define the first native's report about the second native's claim
def firstNativeReport (first : Native) (second : Native) : Prop :=
  claimToBeLiar second

-- Define the correctness of the path indication
def correctPathIndication (n : Native) : Prop :=
  match n.type with
  | NativeType.Truthful => True
  | NativeType.Liar => False

-- Theorem statement
theorem first_native_is_liar_and_path_is_incorrect 
  (first : Native) (second : Native) :
  firstNativeReport first second →
  first.type = NativeType.Liar ∧ ¬(correctPathIndication first) := by
  sorry

end first_native_is_liar_and_path_is_incorrect_l1631_163148


namespace complex_fraction_calculation_l1631_163197

theorem complex_fraction_calculation : 
  let initial := 104 + 2 / 5
  let step1 := (initial / (3 / 8))
  let step2 := step1 / 2
  let step3 := step2 + (14 + 1 / 2)
  let step4 := step3 * (4 / 7)
  let final := step4 - (2 + 3 / 28)
  final = 86 := by sorry

end complex_fraction_calculation_l1631_163197


namespace quilt_shaded_fraction_l1631_163136

/-- Represents a square quilt -/
structure Quilt :=
  (size : ℕ)
  (shaded_diagonal_squares : ℕ)
  (shaded_full_squares : ℕ)

/-- Calculates the fraction of the quilt that is shaded -/
def shaded_fraction (q : Quilt) : ℚ :=
  (q.shaded_diagonal_squares / 2 + q.shaded_full_squares : ℚ) / (q.size * q.size)

/-- Theorem stating the fraction of the quilt that is shaded -/
theorem quilt_shaded_fraction :
  ∀ q : Quilt, 
    q.size = 4 → 
    q.shaded_diagonal_squares = 4 → 
    q.shaded_full_squares = 1 → 
    shaded_fraction q = 3/16 := by
  sorry

end quilt_shaded_fraction_l1631_163136


namespace ticket_sales_total_l1631_163100

/-- Calculates the total sales from ticket sales given the number of tickets sold and prices. -/
theorem ticket_sales_total (total_tickets : ℕ) (child_tickets : ℕ) (adult_price : ℕ) (child_price : ℕ)
  (h1 : total_tickets = 42)
  (h2 : child_tickets = 16)
  (h3 : adult_price = 5)
  (h4 : child_price = 3) :
  (total_tickets - child_tickets) * adult_price + child_tickets * child_price = 178 := by
  sorry

end ticket_sales_total_l1631_163100


namespace barney_weight_difference_l1631_163196

/-- The weight difference between Barney and five regular dinosaurs -/
def weight_difference : ℕ → ℕ → ℕ → ℕ
  | regular_weight, total_weight, num_regular =>
    total_weight - num_regular * regular_weight

theorem barney_weight_difference :
  weight_difference 800 9500 5 = 1500 := by
  sorry

end barney_weight_difference_l1631_163196


namespace pencil_price_l1631_163138

theorem pencil_price (total_pencils : ℕ) (total_cost : ℚ) (h1 : total_pencils = 10) (h2 : total_cost = 2) :
  total_cost / total_pencils = 1/5 := by
sorry

end pencil_price_l1631_163138


namespace cube_skew_lines_theorem_l1631_163104

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A₁ : Point3D
  B₁ : Point3D
  C₁ : Point3D
  D₁ : Point3D
  edge_length : ℝ

/-- Represents a line in 3D space -/
structure Line3D where
  point1 : Point3D
  point2 : Point3D

/-- Calculate the distance between two skew lines in 3D space -/
def distance_between_skew_lines (l1 l2 : Line3D) : ℝ := sorry

/-- Check if a line segment is perpendicular to two other lines -/
def is_perpendicular_to_lines (segment : Line3D) (l1 l2 : Line3D) : Prop := sorry

/-- Main theorem about the distance between skew lines in a cube and their common perpendicular -/
theorem cube_skew_lines_theorem (cube : Cube) : 
  let A₁D := Line3D.mk cube.A₁ cube.D
  let D₁C := Line3D.mk cube.D₁ cube.C
  let X := Point3D.mk ((2 * cube.D.x + cube.A₁.x) / 3) ((2 * cube.D.y + cube.A₁.y) / 3) ((2 * cube.D.z + cube.A₁.z) / 3)
  let Y := Point3D.mk ((2 * cube.D₁.x + cube.C.x) / 3) ((2 * cube.D₁.y + cube.C.y) / 3) ((2 * cube.D₁.z + cube.C.z) / 3)
  let XY := Line3D.mk X Y
  distance_between_skew_lines A₁D D₁C = cube.edge_length * Real.sqrt 3 / 3 ∧
  is_perpendicular_to_lines XY A₁D D₁C := by
  sorry

end cube_skew_lines_theorem_l1631_163104


namespace picture_placement_l1631_163116

theorem picture_placement (wall_width picture_width : ℝ) 
  (hw : wall_width = 19) 
  (hp : picture_width = 3) : 
  (wall_width - picture_width) / 2 = 8 := by
  sorry

end picture_placement_l1631_163116


namespace integer_roots_fifth_degree_polynomial_l1631_163110

-- Define the set of possible values for m
def PossibleM : Set ℕ := {0, 1, 2, 3, 5}

-- Define a fifth-degree polynomial with integer coefficients
def FifthDegreePolynomial (a b c d e : ℤ) (x : ℤ) : ℤ :=
  x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + e

-- Define the number of integer roots (counting multiplicity)
def NumberOfIntegerRoots (p : ℤ → ℤ) : ℕ :=
  -- This is a placeholder definition. In reality, this would be more complex.
  0

-- The main theorem
theorem integer_roots_fifth_degree_polynomial 
  (a b c d e : ℤ) : 
  NumberOfIntegerRoots (FifthDegreePolynomial a b c d e) ∈ PossibleM :=
sorry

end integer_roots_fifth_degree_polynomial_l1631_163110


namespace cos_pi_minus_2alpha_l1631_163133

theorem cos_pi_minus_2alpha (α : ℝ) (h : Real.cos (π / 2 - α) = Real.sqrt 2 / 3) :
  Real.cos (π - 2 * α) = -5 / 9 := by
  sorry

end cos_pi_minus_2alpha_l1631_163133


namespace fraction_simplification_l1631_163145

theorem fraction_simplification :
  (3/7 + 5/8 + 2/9) / (5/12 + 1/4) = 643/336 := by
  sorry

end fraction_simplification_l1631_163145


namespace tv_show_cost_l1631_163144

/-- Calculates the total cost of a TV show season -/
def season_cost (total_episodes : ℕ) (first_half_cost : ℝ) (second_half_increase : ℝ) : ℝ :=
  let half_episodes := total_episodes / 2
  let first_half_total := first_half_cost * half_episodes
  let second_half_cost := first_half_cost * (1 + second_half_increase)
  let second_half_total := second_half_cost * half_episodes
  first_half_total + second_half_total

/-- Theorem stating the total cost of the TV show season -/
theorem tv_show_cost : 
  season_cost 22 1000 1.2 = 35200 := by
  sorry

#eval season_cost 22 1000 1.2

end tv_show_cost_l1631_163144


namespace ellipse_eccentricity_l1631_163191

/-- The eccentricity of an ellipse with specific properties -/
theorem ellipse_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0) : 
  let e := c / a
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 → 
    ((x + c)^2 + y^2 = (a + e * x)^2) ∧
    (0 - b)^2 / ((-c) - 0)^2 + (b - 0)^2 / (0 - a)^2 = 1) →
  e = (Real.sqrt 5 - 1) / 2 := by
  sorry

end ellipse_eccentricity_l1631_163191


namespace line_perpendicular_to_plane_and_line_in_plane_l1631_163117

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships
variable (perpendicular : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorem
theorem line_perpendicular_to_plane_and_line_in_plane
  (m n : Line) (α : Plane)
  (h1 : perpendicular m α)
  (h2 : contains α n) :
  perpendicular_lines m n :=
sorry

end line_perpendicular_to_plane_and_line_in_plane_l1631_163117


namespace max_min_values_of_f_l1631_163170

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 - 4*x + 6

-- Define the closed interval [1, 5]
def I : Set ℝ := Set.Icc 1 5

-- Theorem statement
theorem max_min_values_of_f :
  ∃ (a b : ℝ), a ∈ I ∧ b ∈ I ∧
  (∀ x ∈ I, f x ≤ f a) ∧
  (∀ x ∈ I, f x ≥ f b) ∧
  f a = 11 ∧ f b = 2 :=
sorry

end max_min_values_of_f_l1631_163170


namespace sphere_volume_from_surface_area_l1631_163150

theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 
    (4 * π * r^2 = 256 * π) →
    ((4 / 3) * π * r^3 = (2048 / 3) * π) :=
by
  sorry

end sphere_volume_from_surface_area_l1631_163150


namespace two_and_half_dozens_eq_30_l1631_163162

/-- The number of items in a dozen -/
def dozen : ℕ := 12

/-- The number of pens in two and one-half dozens -/
def two_and_half_dozens : ℕ := 2 * dozen + dozen / 2

/-- Theorem stating that two and one-half dozens of pens is equal to 30 pens -/
theorem two_and_half_dozens_eq_30 : two_and_half_dozens = 30 := by
  sorry

end two_and_half_dozens_eq_30_l1631_163162


namespace rectangle_to_square_l1631_163172

theorem rectangle_to_square (k : ℕ) (n : ℕ) : 
  k > 7 →
  k * (k - 7) = n^2 →
  n < k →
  n = 24 :=
by sorry

end rectangle_to_square_l1631_163172


namespace gcd_factorial_problem_l1631_163192

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem gcd_factorial_problem : Nat.gcd (factorial 7) ((factorial 12) / (factorial 5)) = 5040 := by
  sorry

end gcd_factorial_problem_l1631_163192


namespace auto_dealer_sales_l1631_163184

theorem auto_dealer_sales (trucks : ℕ) (cars : ℕ) : 
  trucks = 21 →
  cars = trucks + 27 →
  cars + trucks = 69 := by
sorry

end auto_dealer_sales_l1631_163184


namespace fiftieth_ring_squares_l1631_163102

/-- The number of squares in the nth ring around a 2x3 rectangular center block -/
def ring_squares (n : ℕ) : ℕ := 8 * n + 6

/-- The 50th ring contains 406 squares -/
theorem fiftieth_ring_squares : ring_squares 50 = 406 := by
  sorry

end fiftieth_ring_squares_l1631_163102


namespace no_integer_solutions_l1631_163164

theorem no_integer_solutions (n : ℤ) : ¬ ∃ x : ℤ, x^2 - 16*n*x + 7^5 = 0 := by
  sorry

end no_integer_solutions_l1631_163164


namespace sqrt_equation_solution_l1631_163161

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 16) = 5 → x = 9 := by
  sorry

end sqrt_equation_solution_l1631_163161


namespace no_x_squared_term_l1631_163178

theorem no_x_squared_term (a : ℝ) : 
  (∀ x, (x + 1) * (x^2 - 5*a*x + a) = x^3 + (-4*a)*x + a) → a = 1/5 := by
  sorry

end no_x_squared_term_l1631_163178


namespace inverse_proportion_point_ordering_l1631_163124

/-- Given an inverse proportion function and three points on its graph, 
    prove the ordering of their x-coordinates. -/
theorem inverse_proportion_point_ordering (k : ℝ) (a b c : ℝ) : 
  (∃ (k : ℝ), -3 = -((k^2 + 1) / a) ∧ 
               -2 = -((k^2 + 1) / b) ∧ 
                1 = -((k^2 + 1) / c)) →
  c < a ∧ a < b := by
  sorry

end inverse_proportion_point_ordering_l1631_163124


namespace john_videos_per_day_l1631_163153

/-- Represents the number of videos and their durations for a video creator --/
structure VideoCreator where
  short_videos_per_day : ℕ
  long_videos_per_day : ℕ
  short_video_duration : ℕ
  long_video_duration : ℕ
  days_per_week : ℕ
  total_weekly_minutes : ℕ

/-- Calculates the total number of videos released per day --/
def total_videos_per_day (vc : VideoCreator) : ℕ :=
  vc.short_videos_per_day + vc.long_videos_per_day

/-- Calculates the total minutes of video released per day --/
def total_minutes_per_day (vc : VideoCreator) : ℕ :=
  vc.short_videos_per_day * vc.short_video_duration +
  vc.long_videos_per_day * vc.long_video_duration

/-- Theorem stating that given the conditions, the total number of videos released per day is 3 --/
theorem john_videos_per_day :
  ∀ (vc : VideoCreator),
  vc.short_videos_per_day = 2 →
  vc.long_videos_per_day = 1 →
  vc.short_video_duration = 2 →
  vc.long_video_duration = 6 * vc.short_video_duration →
  vc.days_per_week = 7 →
  vc.total_weekly_minutes = 112 →
  vc.total_weekly_minutes = vc.days_per_week * (total_minutes_per_day vc) →
  total_videos_per_day vc = 3 := by
  sorry

end john_videos_per_day_l1631_163153


namespace power_difference_value_l1631_163171

theorem power_difference_value (a x y : ℝ) (ha : a > 0) (hx : a^x = 2) (hy : a^y = 3) :
  a^(x - y) = 2/3 := by sorry

end power_difference_value_l1631_163171


namespace problem_solution_l1631_163137

/-- Represents the number of students in different groups -/
structure StudentGroups where
  total : ℕ
  chinese : ℕ
  math : ℕ
  both : ℕ

/-- Calculates the number of students in neither group -/
def studentsInNeither (g : StudentGroups) : ℕ :=
  g.total - (g.chinese + g.math - g.both)

/-- Theorem statement for the given problem -/
theorem problem_solution (g : StudentGroups) 
  (h1 : g.total = 50)
  (h2 : g.chinese = 15)
  (h3 : g.math = 20)
  (h4 : g.both = 8) :
  studentsInNeither g = 23 := by
  sorry

/-- Example usage of the theorem -/
example : studentsInNeither ⟨50, 15, 20, 8⟩ = 23 := by
  apply problem_solution ⟨50, 15, 20, 8⟩
  repeat' rfl


end problem_solution_l1631_163137


namespace modulo_17_intercepts_l1631_163130

/-- Prove the x-intercept, y-intercept, and their sum for the equation 5x ≡ 3y - 1 (mod 17) -/
theorem modulo_17_intercepts :
  ∃ (x₀ y₀ : ℕ), 
    x₀ < 17 ∧ 
    y₀ < 17 ∧
    (5 * x₀) % 17 = 16 ∧ 
    (3 * y₀) % 17 = 1 ∧
    x₀ = 1 ∧ 
    y₀ = 6 ∧ 
    x₀ + y₀ = 7 :=
by sorry

end modulo_17_intercepts_l1631_163130


namespace ratio_of_repeating_decimals_l1631_163193

/-- Represents a repeating decimal with an integer part and a repeating fractional part. -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def toRational (d : RepeatingDecimal) : ℚ :=
  sorry

/-- The repeating decimal 0.75̅ -/
def zeroPointSevenFive : RepeatingDecimal :=
  { integerPart := 0, repeatingPart := 75 }

/-- The repeating decimal 2.25̅ -/
def twoPointTwoFive : RepeatingDecimal :=
  { integerPart := 2, repeatingPart := 25 }

/-- Theorem stating that the ratio of 0.75̅ to 2.25̅ is equal to 2475/7329 -/
theorem ratio_of_repeating_decimals :
  (toRational zeroPointSevenFive) / (toRational twoPointTwoFive) = 2475 / 7329 := by
  sorry

end ratio_of_repeating_decimals_l1631_163193


namespace divisibility_and_infinite_pairs_l1631_163143

theorem divisibility_and_infinite_pairs (c d : ℤ) :
  (∃ f : ℕ → ℤ × ℤ, Function.Injective f ∧
    ∀ n, (f n).1 ∣ (c * (f n).2 + d) ∧ (f n).2 ∣ (c * (f n).1 + d)) ↔
  c ∣ d :=
by sorry

end divisibility_and_infinite_pairs_l1631_163143


namespace pen_cost_theorem_l1631_163131

/-- The average cost per pen in cents, rounded to the nearest whole number,
    given the number of pens, cost of pens, and shipping cost. -/
def average_cost_per_pen (num_pens : ℕ) (pen_cost shipping_cost : ℚ) : ℕ :=
  let total_cost_cents := (pen_cost + shipping_cost) * 100
  let average_cost := total_cost_cents / num_pens
  (average_cost + 1/2).floor.toNat

/-- Theorem stating that for 300 pens costing $29.85 with $8.10 shipping,
    the average cost per pen is 13 cents when rounded to the nearest whole number. -/
theorem pen_cost_theorem :
  average_cost_per_pen 300 (29.85) (8.10) = 13 := by
  sorry

end pen_cost_theorem_l1631_163131


namespace compound_interest_problem_l1631_163198

/-- Given a principal amount where the simple interest for 3 years at 10% per annum is 900,
    prove that the compound interest for the same period and rate is 993. -/
theorem compound_interest_problem (P : ℝ) : 
  P * 0.10 * 3 = 900 → 
  P * (1 + 0.10)^3 - P = 993 := by
  sorry

end compound_interest_problem_l1631_163198


namespace autumn_pencils_l1631_163108

theorem autumn_pencils (initial : ℕ) 
  (misplaced : ℕ) (broken : ℕ) (found : ℕ) (bought : ℕ) (final : ℕ) : 
  misplaced = 7 → broken = 3 → found = 4 → bought = 2 → final = 16 →
  initial - misplaced - broken + found + bought = final →
  initial = 22 := by
sorry

end autumn_pencils_l1631_163108
