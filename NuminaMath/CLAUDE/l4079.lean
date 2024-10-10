import Mathlib

namespace find_A_l4079_407959

theorem find_A : ∀ A : ℕ, (A / 7 = 5) ∧ (A % 7 = 3) → A = 38 := by
  sorry

end find_A_l4079_407959


namespace two_numbers_problem_l4079_407939

theorem two_numbers_problem (x y : ℝ) : 
  x^2 + y^2 = 45/4 ∧ x - y = x * y → 
  (x = -3 ∧ y = 3/2) ∨ (x = -3/2 ∧ y = 3) := by
sorry

end two_numbers_problem_l4079_407939


namespace expression_factorization_l4079_407979

theorem expression_factorization (a b c : ℝ) :
  (((a^2 + 1) - (b^2 + 1))^3 + ((b^2 + 1) - (c^2 + 1))^3 + ((c^2 + 1) - (a^2 + 1))^3) /
  ((a - b)^3 + (b - c)^3 + (c - a)^3) = (a + b) * (b + c) * (c + a) :=
by sorry

end expression_factorization_l4079_407979


namespace largest_change_first_digit_l4079_407977

def original_number : ℚ := 0.05123

def change_digit (n : ℚ) (pos : ℕ) (new_digit : ℕ) : ℚ :=
  sorry

theorem largest_change_first_digit :
  ∀ pos : ℕ, pos > 0 → pos ≤ 5 →
    change_digit original_number 1 8 > change_digit original_number pos 8 :=
  sorry

end largest_change_first_digit_l4079_407977


namespace ratio_of_bases_l4079_407903

/-- An isosceles trapezoid with a point inside dividing it into four triangles -/
structure IsoscelesTrapezoidWithPoint where
  /-- Length of the larger base AB -/
  AB : ℝ
  /-- Length of the smaller base CD -/
  CD : ℝ
  /-- Area of triangle with base CD -/
  area_CD : ℝ
  /-- Area of triangle adjacent to CD (clockwise) -/
  area_adj_CD : ℝ
  /-- Area of triangle with base AB -/
  area_AB : ℝ
  /-- Area of triangle adjacent to AB (counter-clockwise) -/
  area_adj_AB : ℝ
  /-- AB is longer than CD -/
  h_AB_gt_CD : AB > CD
  /-- The trapezoid is isosceles -/
  h_isosceles : True  -- We don't need to specify this condition explicitly for the proof
  /-- The bases are parallel -/
  h_parallel : True   -- We don't need to specify this condition explicitly for the proof
  /-- Areas of triangles -/
  h_areas : area_CD = 5 ∧ area_adj_CD = 7 ∧ area_AB = 9 ∧ area_adj_AB = 3

/-- The ratio of bases in the isosceles trapezoid with given triangle areas -/
theorem ratio_of_bases (t : IsoscelesTrapezoidWithPoint) : t.AB / t.CD = 1 + 2 * Real.sqrt 2 := by
  sorry

end ratio_of_bases_l4079_407903


namespace total_selling_price_proof_l4079_407925

def calculate_selling_price (cost : ℕ) (profit_percentage : ℕ) : ℕ :=
  cost + cost * profit_percentage / 100

theorem total_selling_price_proof (cost_A cost_B cost_C : ℕ)
  (profit_percentage_A profit_percentage_B profit_percentage_C : ℕ)
  (h1 : cost_A = 400)
  (h2 : cost_B = 600)
  (h3 : cost_C = 800)
  (h4 : profit_percentage_A = 40)
  (h5 : profit_percentage_B = 35)
  (h6 : profit_percentage_C = 25) :
  calculate_selling_price cost_A profit_percentage_A +
  calculate_selling_price cost_B profit_percentage_B +
  calculate_selling_price cost_C profit_percentage_C = 2370 :=
by
  sorry

end total_selling_price_proof_l4079_407925


namespace factor_expression_l4079_407900

theorem factor_expression (x : ℝ) : 75 * x^11 + 135 * x^22 = 15 * x^11 * (5 + 9 * x^11) := by
  sorry

end factor_expression_l4079_407900


namespace x_eq_one_sufficient_not_necessary_for_cube_eq_x_l4079_407933

theorem x_eq_one_sufficient_not_necessary_for_cube_eq_x (x : ℝ) :
  (x = 1 → x^3 = x) ∧ ¬(x^3 = x → x = 1) := by
  sorry

end x_eq_one_sufficient_not_necessary_for_cube_eq_x_l4079_407933


namespace intersection_of_A_and_B_l4079_407966

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {x | x > 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 1} := by
  sorry

end intersection_of_A_and_B_l4079_407966


namespace fraction_zero_l4079_407954

theorem fraction_zero (x : ℝ) (h : x ≠ 1) : (x^2 - 1) / (x - 1) = 0 ↔ x = -1 := by
  sorry

end fraction_zero_l4079_407954


namespace students_watching_l4079_407996

theorem students_watching (total : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 33 → 
  total = boys + girls → 
  (2 * boys + 2 * girls) / 3 = 22 :=
by
  sorry

end students_watching_l4079_407996


namespace complement_of_A_in_U_l4079_407956

def U : Set Int := {-1, 0, 1, 2, 3}
def A : Set Int := {-1, 0, 2}

theorem complement_of_A_in_U :
  (U \ A) = {1, 3} := by sorry

end complement_of_A_in_U_l4079_407956


namespace celine_book_days_l4079_407906

/-- The number of days in May -/
def days_in_may : ℕ := 31

/-- The daily charge for borrowing a book (in dollars) -/
def daily_charge : ℚ := 1/2

/-- The total amount Celine paid (in dollars) -/
def total_paid : ℚ := 41

/-- The number of books Celine borrowed -/
def num_books : ℕ := 3

theorem celine_book_days :
  ∃ (x : ℕ), 
    daily_charge * x + daily_charge * (num_books - 1) * days_in_may = total_paid ∧
    x = 20 := by
  sorry

end celine_book_days_l4079_407906


namespace prob_same_color_is_139_435_l4079_407901

-- Define the number of socks for each color
def blue_socks : ℕ := 12
def gray_socks : ℕ := 10
def white_socks : ℕ := 8

-- Define the total number of socks
def total_socks : ℕ := blue_socks + gray_socks + white_socks

-- Define the probability of picking two socks of the same color
def prob_same_color : ℚ :=
  (Nat.choose blue_socks 2 + Nat.choose gray_socks 2 + Nat.choose white_socks 2) /
  Nat.choose total_socks 2

-- Theorem statement
theorem prob_same_color_is_139_435 : prob_same_color = 139 / 435 := by
  sorry

end prob_same_color_is_139_435_l4079_407901


namespace average_salary_raj_roshan_l4079_407938

theorem average_salary_raj_roshan (raj_salary roshan_salary : ℕ) : 
  (raj_salary + roshan_salary + 7000) / 3 = 5000 →
  (raj_salary + roshan_salary) / 2 = 4000 := by
  sorry

end average_salary_raj_roshan_l4079_407938


namespace peters_newspaper_delivery_l4079_407943

/-- Peter's newspaper delivery problem -/
theorem peters_newspaper_delivery :
  let total_weekend := 110
  let saturday := 45
  let sunday := 65
  sunday > saturday →
  sunday - saturday = 20 :=
by
  sorry

end peters_newspaper_delivery_l4079_407943


namespace cubic_two_intersections_l4079_407918

/-- A cubic function that intersects the x-axis at exactly two points -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x + 1

/-- The derivative of f with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 1

theorem cubic_two_intersections :
  ∃! a : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) ∧
  (∀ x₃ : ℝ, f a x₃ = 0 → x₃ = x₁ ∨ x₃ = x₂) ∧
  a = -4/27 :=
sorry

end cubic_two_intersections_l4079_407918


namespace intersection_and_lines_l4079_407989

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 2 * x - y + 7 = 0
def l₂ (x y : ℝ) : Prop := x + y - 1 = 0

-- Define the intersection point A(m, n)
def m : ℝ := -2
def n : ℝ := 3

-- Define line l
def l (x y : ℝ) : Prop := 2 * x - 3 * y - 1 = 0

-- Theorem statement
theorem intersection_and_lines :
  (l₁ m n ∧ l₂ m n) ∧  -- A is the intersection of l₁ and l₂
  (∀ x y : ℝ, x + 2 * y - 4 = 0 ↔ (x - m) * 2 + (y - n) * 1 = 0) ∧  -- l₃ equation
  (∀ x y : ℝ, 2 * x - 3 * y + 13 = 0 ↔ (y - n) = (2 / 3) * (x - m)) :=  -- l₄ equation
by sorry

end intersection_and_lines_l4079_407989


namespace expression_value_l4079_407945

theorem expression_value (x y : ℚ) (hx : x = 3) (hy : y = 4) :
  (x^5 + 3*y^2 + 7) / (x + 4) = 298 / 7 := by
  sorry

end expression_value_l4079_407945


namespace randys_trip_length_l4079_407910

theorem randys_trip_length :
  ∀ (total : ℚ),
  (total / 3 : ℚ) + 20 + (total / 5 : ℚ) = total →
  total = 300 / 7 := by
sorry

end randys_trip_length_l4079_407910


namespace complex_number_in_second_quadrant_l4079_407993

def second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem complex_number_in_second_quadrant :
  let z : ℂ := -1 + 2*I
  second_quadrant z := by
  sorry

end complex_number_in_second_quadrant_l4079_407993


namespace multiplication_subtraction_equality_l4079_407978

theorem multiplication_subtraction_equality : 75 * 3030 - 35 * 3030 = 121200 := by
  sorry

end multiplication_subtraction_equality_l4079_407978


namespace min_students_with_glasses_and_scarf_l4079_407963

theorem min_students_with_glasses_and_scarf (n : ℕ) 
  (h1 : n > 0)
  (h2 : ∃ k : ℕ, n * 3 = k * 7)
  (h3 : ∃ m : ℕ, n * 5 = m * 6)
  (h4 : ∀ p : ℕ, p > 0 → (∃ q : ℕ, p * 3 = q * 7) → (∃ r : ℕ, p * 5 = r * 6) → p ≥ n) :
  ∃ x : ℕ, x = 11 ∧ 
    x = n * 3 / 7 + n * 5 / 6 - n :=
by sorry

end min_students_with_glasses_and_scarf_l4079_407963


namespace subset_implies_m_equals_one_l4079_407948

def A (m : ℝ) : Set ℝ := {-1, 3, 2*m-1}
def B (m : ℝ) : Set ℝ := {3, m^2}

theorem subset_implies_m_equals_one (m : ℝ) : B m ⊆ A m → m = 1 := by
  sorry

end subset_implies_m_equals_one_l4079_407948


namespace count_twelve_digit_numbers_with_three_ones_l4079_407987

/-- Recursively defines the count of n-digit numbers with digits 1 or 2 without three consecutive 1's -/
def G : ℕ → ℕ
| 0 => 1  -- Base case for 0 digits (empty string)
| 1 => 2  -- Base case for 1 digit
| 2 => 3  -- Base case for 2 digits
| n + 3 => G (n + 2) + G (n + 1) + G n

/-- The count of 12-digit numbers with all digits 1 or 2 and at least three consecutive 1's -/
def count_with_three_ones : ℕ := 2^12 - G 12

theorem count_twelve_digit_numbers_with_three_ones : 
  count_with_three_ones = 3656 :=
sorry

end count_twelve_digit_numbers_with_three_ones_l4079_407987


namespace absolute_difference_sequence_l4079_407942

/-- Given three non-negative real numbers x, y, z, where z = 1, and after n steps of taking pairwise
    absolute differences, the sequence stabilizes with x_n = x, y_n = y, z_n = z, 
    then (x, y) = (0, 1) or (1, 0). -/
theorem absolute_difference_sequence (x y z : ℝ) (n : ℕ) :
  x ≥ 0 ∧ y ≥ 0 ∧ z = 1 →
  (∃ (x_seq y_seq z_seq : ℕ → ℝ),
    (∀ k, k < n → 
      x_seq (k+1) = |x_seq k - y_seq k| ∧
      y_seq (k+1) = |y_seq k - z_seq k| ∧
      z_seq (k+1) = |z_seq k - x_seq k|) ∧
    x_seq 0 = x ∧ y_seq 0 = y ∧ z_seq 0 = z ∧
    x_seq n = x ∧ y_seq n = y ∧ z_seq n = z) →
  (x = 0 ∧ y = 1) ∨ (x = 1 ∧ y = 0) :=
by sorry

end absolute_difference_sequence_l4079_407942


namespace product_difference_sum_problem_l4079_407904

theorem product_difference_sum_problem : 
  ∃ (a b : ℕ+), (a * b = 18) ∧ (max a b - min a b = 3) → (a + b = 9) :=
by sorry

end product_difference_sum_problem_l4079_407904


namespace triangle_formation_l4079_407983

/-- Triangle inequality theorem check for three sides --/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Check if three lengths can form a triangle --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

theorem triangle_formation :
  can_form_triangle 2 3 4 ∧
  ¬can_form_triangle 5 1 3 ∧
  ¬can_form_triangle 2 4 2 ∧
  ¬can_form_triangle 3 3 7 :=
by sorry

end triangle_formation_l4079_407983


namespace systematic_sampling_result_l4079_407922

/-- Represents the systematic sampling of students for a dental health check. -/
def systematicSampling (totalStudents : ℕ) (sampleSize : ℕ) (interval : ℕ) (start : ℕ) : List ℕ :=
  List.range sampleSize |>.map (fun i => start + i * interval)

/-- Theorem stating that the systematic sampling with given parameters results in the expected list of student numbers. -/
theorem systematic_sampling_result :
  systematicSampling 50 5 10 6 = [6, 16, 26, 36, 46] := by
  sorry

end systematic_sampling_result_l4079_407922


namespace intersection_A_B_l4079_407909

-- Define set A
def A : Set ℝ := {x | (x - 2) * (2 * x + 1) ≤ 0}

-- Define set B
def B : Set ℝ := {x | x < 1}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x : ℝ | -1/2 ≤ x ∧ x < 1} := by
  sorry

end intersection_A_B_l4079_407909


namespace ball_probability_l4079_407969

theorem ball_probability (total : ℕ) (white green yellow red purple : ℕ) 
  (h_total : total = 100)
  (h_white : white = 50)
  (h_green : green = 20)
  (h_yellow : yellow = 10)
  (h_red : red = 17)
  (h_purple : purple = 3)
  (h_sum : white + green + yellow + red + purple = total) :
  (white + green + yellow : ℚ) / total = 0.8 := by
sorry

end ball_probability_l4079_407969


namespace divided_triangle_angles_l4079_407902

/-- A triangle that can be divided into several smaller triangles -/
structure DividedTriangle where
  -- The number of triangles the original triangle is divided into
  num_divisions : ℕ
  -- Assertion that there are at least two divisions
  h_at_least_two : num_divisions ≥ 2

/-- Represents the properties of the divided triangles -/
structure DivisionProperties (T : DividedTriangle) where
  -- The number of equilateral triangles in the division
  num_equilateral : ℕ
  -- The number of isosceles (non-equilateral) triangles in the division
  num_isosceles : ℕ
  -- Assertion that there is exactly one isosceles triangle
  h_one_isosceles : num_isosceles = 1
  -- Assertion that all other triangles are equilateral
  h_rest_equilateral : num_equilateral + num_isosceles = T.num_divisions

/-- The theorem stating the angles of the original triangle -/
theorem divided_triangle_angles (T : DividedTriangle) (P : DivisionProperties T) :
  ∃ (a b c : ℝ), a = 30 ∧ b = 60 ∧ c = 90 ∧ a + b + c = 180 :=
sorry

end divided_triangle_angles_l4079_407902


namespace paper_I_maximum_marks_l4079_407915

theorem paper_I_maximum_marks :
  ∀ (max_marks passing_mark secured_marks deficit : ℕ),
    passing_mark = (max_marks * 40) / 100 →
    secured_marks = 40 →
    deficit = 20 →
    passing_mark = secured_marks + deficit →
    max_marks = 150 := by
  sorry

end paper_I_maximum_marks_l4079_407915


namespace pizza_toppings_combinations_l4079_407965

theorem pizza_toppings_combinations : Nat.choose 8 5 = 56 := by
  sorry

end pizza_toppings_combinations_l4079_407965


namespace parabola_point_ordinate_l4079_407999

/-- The y-coordinate of a point on the parabola y = 4x^2 that is at a distance of 1 from the focus -/
theorem parabola_point_ordinate : ∀ (x y : ℝ),
  y = 4 * x^2 →  -- Point is on the parabola
  (x - 0)^2 + (y - 1/16)^2 = 1 →  -- Distance from focus is 1
  y = 15/16 := by
  sorry

end parabola_point_ordinate_l4079_407999


namespace distinct_triangles_in_cube_l4079_407914

/-- The number of vertices in a cube -/
def cube_vertices : ℕ := 8

/-- The number of vertices needed to form a triangle -/
def triangle_vertices : ℕ := 3

/-- The number of distinct triangles that can be formed by connecting three different vertices of a cube -/
def distinct_triangles : ℕ := Nat.choose cube_vertices triangle_vertices

theorem distinct_triangles_in_cube :
  distinct_triangles = 56 :=
sorry

end distinct_triangles_in_cube_l4079_407914


namespace bucket_capacity_problem_l4079_407905

theorem bucket_capacity_problem (capacity : ℝ) : 
  (24 * capacity = 36 * 9) → capacity = 13.5 := by
  sorry

end bucket_capacity_problem_l4079_407905


namespace exists_abs_leq_zero_l4079_407991

theorem exists_abs_leq_zero : ∃ x : ℝ, |x| ≤ 0 := by
  sorry

end exists_abs_leq_zero_l4079_407991


namespace hyperbola_point_comparison_l4079_407932

theorem hyperbola_point_comparison 
  (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : y₁ = 2023 / x₁) 
  (h2 : y₂ = 2023 / x₂) 
  (h3 : y₁ > y₂) 
  (h4 : y₂ > 0) : 
  x₁ < x₂ := by
sorry

end hyperbola_point_comparison_l4079_407932


namespace probability_at_least_one_multiple_of_four_l4079_407944

theorem probability_at_least_one_multiple_of_four :
  let range := Finset.range 100
  let multiples_of_four := range.filter (λ n => n % 4 = 0)
  let prob_not_multiple := (range.card - multiples_of_four.card : ℚ) / range.card
  1 - prob_not_multiple ^ 2 = 7 / 16 := by
sorry

end probability_at_least_one_multiple_of_four_l4079_407944


namespace complex_number_in_fourth_quadrant_l4079_407994

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := 1 / (1 + 3 * Complex.I)
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end complex_number_in_fourth_quadrant_l4079_407994


namespace largest_sum_of_digits_l4079_407974

/-- Represents a digital time display in 24-hour format -/
structure TimeDisplay where
  hours : Nat
  minutes : Nat
  seconds : Nat
  hours_valid : hours < 24
  minutes_valid : minutes < 60
  seconds_valid : seconds < 60

/-- Calculates the sum of digits for a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  n.repr.foldl (fun sum c => sum + c.toNat - '0'.toNat) 0

/-- Calculates the sum of digits for a time display -/
def sumOfTimeDigits (t : TimeDisplay) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes + sumOfDigits t.seconds

/-- The largest possible sum of digits in a 24-hour format digital watch display is 38 -/
theorem largest_sum_of_digits : ∀ t : TimeDisplay, sumOfTimeDigits t ≤ 38 := by
  sorry

end largest_sum_of_digits_l4079_407974


namespace sum_of_fractions_equals_two_ninths_l4079_407973

theorem sum_of_fractions_equals_two_ninths :
  (1 / (3 * 4 : ℚ)) + (1 / (4 * 5 : ℚ)) + (1 / (5 * 6 : ℚ)) +
  (1 / (6 * 7 : ℚ)) + (1 / (7 * 8 : ℚ)) + (1 / (8 * 9 : ℚ)) = 2 / 9 := by
  sorry

end sum_of_fractions_equals_two_ninths_l4079_407973


namespace intersection_of_M_and_N_l4079_407986

def M : Set ℝ := {x | x^2 ≥ 4}
def N : Set ℝ := {-3, 0, 1, 3, 4}

theorem intersection_of_M_and_N :
  M ∩ N = {-3, 3, 4} := by sorry

end intersection_of_M_and_N_l4079_407986


namespace hitAtMostOnce_mutually_exclusive_hitBothTimes_l4079_407970

/-- Represents the outcome of a single shot -/
inductive ShotOutcome
| Hit
| Miss

/-- Represents the outcome of two shots -/
def TwoShotOutcome := (ShotOutcome × ShotOutcome)

/-- The event of hitting the target at most once -/
def hitAtMostOnce (outcome : TwoShotOutcome) : Prop :=
  match outcome with
  | (ShotOutcome.Miss, ShotOutcome.Miss) => True
  | (ShotOutcome.Hit, ShotOutcome.Miss) => True
  | (ShotOutcome.Miss, ShotOutcome.Hit) => True
  | (ShotOutcome.Hit, ShotOutcome.Hit) => False

/-- The event of hitting the target both times -/
def hitBothTimes (outcome : TwoShotOutcome) : Prop :=
  match outcome with
  | (ShotOutcome.Hit, ShotOutcome.Hit) => True
  | _ => False

theorem hitAtMostOnce_mutually_exclusive_hitBothTimes :
  ∀ (outcome : TwoShotOutcome), ¬(hitAtMostOnce outcome ∧ hitBothTimes outcome) :=
by sorry

end hitAtMostOnce_mutually_exclusive_hitBothTimes_l4079_407970


namespace compound_oxygen_atoms_l4079_407967

/-- Represents the atomic weights of elements in atomic mass units (amu) -/
structure AtomicWeight where
  Cu : ℝ
  C : ℝ
  O : ℝ

/-- Represents a compound with Cu, C, and O atoms -/
structure Compound where
  Cu : ℕ
  C : ℕ
  O : ℕ

/-- Calculates the molecular weight of a compound given atomic weights -/
def molecularWeight (c : Compound) (w : AtomicWeight) : ℝ :=
  c.Cu * w.Cu + c.C * w.C + c.O * w.O

/-- Theorem stating that a compound with 1 Cu, 1 C, and n O atoms
    with a molecular weight of 124 amu has 3 O atoms -/
theorem compound_oxygen_atoms
  (w : AtomicWeight)
  (h1 : w.Cu = 63.55)
  (h2 : w.C = 12.01)
  (h3 : w.O = 16.00)
  (c : Compound)
  (h4 : c.Cu = 1)
  (h5 : c.C = 1)
  (h6 : molecularWeight c w = 124) :
  c.O = 3 := by
  sorry

end compound_oxygen_atoms_l4079_407967


namespace clock_angle_at_8_30_clock_angle_at_8_30_is_75_l4079_407960

/-- The angle between clock hands at 8:30 -/
theorem clock_angle_at_8_30 : ℝ :=
  let degrees_per_hour : ℝ := 360 / 12
  let degrees_per_minute : ℝ := 360 / 60
  let hours : ℝ := 8.5
  let minutes : ℝ := 30
  let hour_hand_angle : ℝ := hours * degrees_per_hour
  let minute_hand_angle : ℝ := minutes * degrees_per_minute
  |hour_hand_angle - minute_hand_angle|

theorem clock_angle_at_8_30_is_75 : clock_angle_at_8_30 = 75 := by
  sorry

end clock_angle_at_8_30_clock_angle_at_8_30_is_75_l4079_407960


namespace min_z_shapes_cover_min_z_shapes_necessary_l4079_407913

/-- Represents a cell on the table -/
structure Cell where
  row : Nat
  col : Nat

/-- Represents a Z shape on the table -/
structure ZShape where
  base : Cell
  rotation : Nat

/-- The size of the table -/
def tableSize : Nat := 8

/-- Checks if a cell is within the table bounds -/
def isValidCell (c : Cell) : Prop :=
  c.row ≥ 1 ∧ c.row ≤ tableSize ∧ c.col ≥ 1 ∧ c.col ≤ tableSize

/-- Checks if a Z shape covers a given cell -/
def coversCell (z : ZShape) (c : Cell) : Prop :=
  match z.rotation % 4 with
  | 0 => c = z.base ∨ c = ⟨z.base.row, z.base.col + 1⟩ ∨ 
         c = ⟨z.base.row + 1, z.base.col + 1⟩ ∨ 
         c = ⟨z.base.row + 2, z.base.col + 1⟩ ∨ 
         c = ⟨z.base.row + 2, z.base.col + 2⟩
  | 1 => c = z.base ∨ c = ⟨z.base.row + 1, z.base.col⟩ ∨ 
         c = ⟨z.base.row + 1, z.base.col + 1⟩ ∨ 
         c = ⟨z.base.row + 1, z.base.col + 2⟩ ∨ 
         c = ⟨z.base.row + 2, z.base.col + 2⟩
  | 2 => c = z.base ∨ c = ⟨z.base.row, z.base.col + 1⟩ ∨ 
         c = ⟨z.base.row - 1, z.base.col + 1⟩ ∨ 
         c = ⟨z.base.row - 2, z.base.col + 1⟩ ∨ 
         c = ⟨z.base.row - 2, z.base.col + 2⟩
  | _ => c = z.base ∨ c = ⟨z.base.row - 1, z.base.col⟩ ∨ 
         c = ⟨z.base.row - 1, z.base.col - 1⟩ ∨ 
         c = ⟨z.base.row - 1, z.base.col - 2⟩ ∨ 
         c = ⟨z.base.row - 2, z.base.col - 2⟩

/-- The main theorem stating that 12 Z shapes are sufficient to cover the table -/
theorem min_z_shapes_cover (shapes : List ZShape) : 
  (∀ c : Cell, isValidCell c → ∃ z ∈ shapes, coversCell z c) → 
  shapes.length ≥ 12 :=
sorry

/-- The main theorem stating that 12 Z shapes are necessary to cover the table -/
theorem min_z_shapes_necessary : 
  ∃ shapes : List ZShape, shapes.length = 12 ∧ 
  ∀ c : Cell, isValidCell c → ∃ z ∈ shapes, coversCell z c :=
sorry

end min_z_shapes_cover_min_z_shapes_necessary_l4079_407913


namespace complement_equivalence_l4079_407998

-- Define the sample space
def Ω : Type := Bool × Bool

-- Define the event "at least one item is defective"
def at_least_one_defective : Set Ω :=
  {ω | ω.1 = true ∨ ω.2 = true}

-- Define the event "neither of the items is defective"
def neither_defective : Set Ω :=
  {ω | ω.1 = false ∧ ω.2 = false}

-- Theorem: The complement of "at least one item is defective" 
-- is equivalent to "neither of the items is defective"
theorem complement_equivalence :
  at_least_one_defective.compl = neither_defective :=
sorry

end complement_equivalence_l4079_407998


namespace divisibility_condition_implies_prime_relation_l4079_407984

theorem divisibility_condition_implies_prime_relation (m n : ℕ) : 
  m ≥ 2 → n ≥ 2 → 
  (∀ a : ℕ, a ∈ Finset.range n → (a^n - 1) % m = 0) →
  Nat.Prime m ∧ n = m - 1 := by
sorry

end divisibility_condition_implies_prime_relation_l4079_407984


namespace max_value_fraction_l4079_407988

theorem max_value_fraction (x : ℝ) : 
  (4 * x^2 + 12 * x + 19) / (4 * x^2 + 12 * x + 9) ≤ 11 ∧ 
  ∀ ε > 0, ∃ y : ℝ, (4 * y^2 + 12 * y + 19) / (4 * y^2 + 12 * y + 9) > 11 - ε :=
by sorry

end max_value_fraction_l4079_407988


namespace arithmetic_geometric_mean_inequality_l4079_407975

theorem arithmetic_geometric_mean_inequality 
  (a b k : ℝ) 
  (h1 : b = k * a) 
  (h2 : k > 0) 
  (h3 : 1 ≤ k) 
  (h4 : k ≤ 3) : 
  ((a + b) / 2)^2 ≥ (Real.sqrt (a * b))^2 := by
sorry

end arithmetic_geometric_mean_inequality_l4079_407975


namespace rectangle_sides_when_perimeter_equals_area_l4079_407971

theorem rectangle_sides_when_perimeter_equals_area :
  ∀ w l : ℝ,
  w > 0 →
  l = 3 * w →
  2 * (w + l) = w * l →
  w = 8 / 3 ∧ l = 8 := by
sorry

end rectangle_sides_when_perimeter_equals_area_l4079_407971


namespace max_sum_squares_l4079_407953

theorem max_sum_squares (m n : ℕ) : 
  m ∈ Finset.range 101 ∧ 
  n ∈ Finset.range 101 ∧ 
  (n^2 - m*n - m^2)^2 = 1 →
  m^2 + n^2 ≤ 10946 :=
by sorry

end max_sum_squares_l4079_407953


namespace quadratic_roots_condition_l4079_407940

theorem quadratic_roots_condition (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 2*x₁ + m = 0 ∧ x₂^2 - 2*x₂ + m = 0) → m < 1 := by
  sorry

end quadratic_roots_condition_l4079_407940


namespace six_digit_repeat_gcd_l4079_407976

theorem six_digit_repeat_gcd : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ x : ℕ, 100 ≤ x ∧ x < 1000 → 
    n = Nat.gcd (1000 * x + x) (1000 * (x + 1) + (x + 1))) ∧ 
  n = 1001 := by
  sorry

end six_digit_repeat_gcd_l4079_407976


namespace perpendicular_bisector_equation_l4079_407947

/-- The perpendicular bisector of a line segment with endpoints (x₁, y₁) and (x₂, y₂) -/
def perpendicular_bisector (x₁ y₁ x₂ y₂ : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - x₁)^2 + (p.2 - y₁)^2 = (p.1 - x₂)^2 + (p.2 - y₂)^2}

theorem perpendicular_bisector_equation :
  perpendicular_bisector 1 3 5 (-1) = {p : ℝ × ℝ | p.1 - p.2 - 2 = 0} := by
  sorry

end perpendicular_bisector_equation_l4079_407947


namespace jasper_kite_raising_time_l4079_407912

/-- Given Omar's kite-raising rate and Jasper's relative speed, prove Jasper's time to raise his kite -/
theorem jasper_kite_raising_time
  (omar_height : ℝ)
  (omar_time : ℝ)
  (jasper_speed_ratio : ℝ)
  (jasper_height : ℝ)
  (h1 : omar_height = 240)
  (h2 : omar_time = 12)
  (h3 : jasper_speed_ratio = 3)
  (h4 : jasper_height = 600) :
  (jasper_height / (jasper_speed_ratio * (omar_height / omar_time))) = 10 :=
by sorry

end jasper_kite_raising_time_l4079_407912


namespace single_digit_equation_l4079_407972

theorem single_digit_equation (a b : ℕ) : 
  (0 < a ∧ a < 10) →
  (0 < b ∧ b < 10) →
  82 * 10 * a + 7 + 6 * b = 190 →
  a + 2 * b = 6 →
  a = 6 := by
sorry

end single_digit_equation_l4079_407972


namespace solution_set_inequality_inequality_for_negative_a_l4079_407929

-- Define the function f(x) = |x - 5|
def f (x : ℝ) : ℝ := |x - 5|

-- Theorem 1: Solution set of f(x) + f(x + 2) ≤ 3
theorem solution_set_inequality (x : ℝ) :
  (f x + f (x + 2) ≤ 3) ↔ (5/2 ≤ x ∧ x ≤ 11/2) := by sorry

-- Theorem 2: f(ax) - f(5a) ≥ af(x) for a < 0
theorem inequality_for_negative_a (a x : ℝ) (h : a < 0) :
  f (a * x) - f (5 * a) ≥ a * f x := by sorry

end solution_set_inequality_inequality_for_negative_a_l4079_407929


namespace third_book_words_l4079_407957

-- Define the given constants
def days : ℕ := 10
def books : ℕ := 3
def reading_speed : ℕ := 100 -- words per hour
def first_book_words : ℕ := 200
def second_book_words : ℕ := 400
def average_reading_time : ℕ := 54 -- minutes per day

-- Define the theorem
theorem third_book_words :
  let total_reading_time : ℕ := days * average_reading_time
  let total_reading_hours : ℕ := total_reading_time / 60
  let total_words : ℕ := total_reading_hours * reading_speed
  let first_two_books_words : ℕ := first_book_words + second_book_words
  total_words - first_two_books_words = 300 := by
  sorry

end third_book_words_l4079_407957


namespace factorial_calculation_l4079_407997

-- Define the factorial function
def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- State the theorem
theorem factorial_calculation :
  (5 * factorial 6 + 30 * factorial 5) / factorial 7 = 30 / 7 := by
  sorry

end factorial_calculation_l4079_407997


namespace parabola_hyperbola_focus_coincidence_l4079_407958

/-- The value of 'a' for which the focus of the parabola y = ax^2 (a > 0) 
    coincides with one of the foci of the hyperbola y^2 - x^2 = 2 -/
theorem parabola_hyperbola_focus_coincidence (a : ℝ) : 
  a > 0 → 
  (∃ (x y : ℝ), y = a * x^2 ∧ y^2 - x^2 = 2 ∧ 
    ((x = 0 ∧ y = 1 / (4 * a)) ∨ (x = 0 ∧ y = 2) ∨ (x = 0 ∧ y = -2))) → 
  a = 1/8 := by
sorry


end parabola_hyperbola_focus_coincidence_l4079_407958


namespace red_marbles_count_l4079_407955

theorem red_marbles_count :
  ∀ (total blue red yellow : ℕ),
    total = 85 →
    blue = 3 * red →
    yellow = 29 →
    total = red + blue + yellow →
    red = 14 := by
  sorry

end red_marbles_count_l4079_407955


namespace largest_colorable_3subsets_correct_l4079_407941

/-- The largest number of 3-subsets that can be chosen from a set of n elements
    such that there always exists a 2-coloring with no monochromatic chosen 3-subset -/
def largest_colorable_3subsets (n : ℕ) : ℕ :=
  if n = 3 then 1
  else if n = 4 then 4
  else if n = 5 ∨ n = 6 then 9
  else if n ≥ 7 then 6
  else 0

/-- The theorem stating the correct values for the largest number of colorable 3-subsets -/
theorem largest_colorable_3subsets_correct (n : ℕ) (h : n ≥ 3) :
  largest_colorable_3subsets n =
    if n = 3 then 1
    else if n = 4 then 4
    else if n = 5 ∨ n = 6 then 9
    else 6 := by
  sorry

end largest_colorable_3subsets_correct_l4079_407941


namespace fifth_day_income_correct_l4079_407921

/-- Calculates the cab driver's income on the fifth day given the income for the first four days and the average income for five days. -/
def fifth_day_income (day1 day2 day3 day4 avg : ℚ) : ℚ :=
  5 * avg - (day1 + day2 + day3 + day4)

/-- Proves that the calculated fifth day income is correct given the income for the first four days and the average income for five days. -/
theorem fifth_day_income_correct (day1 day2 day3 day4 avg : ℚ) :
  let fifth_day := fifth_day_income day1 day2 day3 day4 avg
  (day1 + day2 + day3 + day4 + fifth_day) / 5 = avg :=
by sorry

#eval fifth_day_income 400 250 650 400 440

end fifth_day_income_correct_l4079_407921


namespace rectangular_solid_diagonal_l4079_407950

/-- Given a rectangular solid with side lengths x, y, and z, 
    if the surface area is 11 and the sum of the lengths of all edges is 24, 
    then the length of one of its diagonals is 5. -/
theorem rectangular_solid_diagonal 
  (x y z : ℝ) 
  (h1 : 2*x*y + 2*y*z + 2*x*z = 11) 
  (h2 : 4*(x + y + z) = 24) : 
  Real.sqrt (x^2 + y^2 + z^2) = 5 := by
  sorry

end rectangular_solid_diagonal_l4079_407950


namespace fruit_candy_cost_difference_l4079_407995

/-- The cost difference between two schools purchasing fruit candy --/
theorem fruit_candy_cost_difference : 
  let school_a_quantity : ℝ := 56
  let school_a_price_per_kg : ℝ := 8.06
  let price_reduction : ℝ := 0.56
  let free_candy_percentage : ℝ := 0.05
  
  let school_b_price_per_kg : ℝ := school_a_price_per_kg - price_reduction
  let school_b_quantity : ℝ := school_a_quantity / (1 + free_candy_percentage)
  
  let school_a_total_cost : ℝ := school_a_quantity * school_a_price_per_kg
  let school_b_total_cost : ℝ := school_b_quantity * school_b_price_per_kg
  
  school_a_total_cost - school_b_total_cost = 51.36 := by
  sorry

end fruit_candy_cost_difference_l4079_407995


namespace paths_count_l4079_407937

/-- The number of paths between two points given the number of right and down steps -/
def numPaths (right down : ℕ) : ℕ := sorry

/-- The total number of paths from A to D via B and C -/
def totalPaths : ℕ :=
  let pathsAB := numPaths 2 2  -- B is 2 right and 2 down from A
  let pathsBC := numPaths 1 3  -- C is 1 right and 3 down from B
  let pathsCD := numPaths 3 1  -- D is 3 right and 1 down from C
  pathsAB * pathsBC * pathsCD

theorem paths_count : totalPaths = 96 := by sorry

end paths_count_l4079_407937


namespace tan_alpha_minus_pi_over_four_l4079_407907

theorem tan_alpha_minus_pi_over_four (α β : Real) 
  (h1 : Real.tan (α + β) = 2/5)
  (h2 : Real.tan β = 1/3) :
  Real.tan (α - π/4) = -8/9 := by
  sorry

end tan_alpha_minus_pi_over_four_l4079_407907


namespace intersection_complement_equality_l4079_407935

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 3, 4, 5}
def B : Set ℕ := {2, 3, 6, 7}

theorem intersection_complement_equality : B ∩ (U \ A) = {6, 7} := by
  sorry

end intersection_complement_equality_l4079_407935


namespace simplify_expression_l4079_407962

theorem simplify_expression : (5 + 7 + 3 - 2) / 3 - 1 / 3 = 4 := by
  sorry

end simplify_expression_l4079_407962


namespace sum_exponents_of_sqrt_largest_perfect_square_12_factorial_l4079_407916

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def largest_perfect_square_divisor (n : ℕ) : ℕ := sorry

def prime_factor_exponents (n : ℕ) : List ℕ := sorry

theorem sum_exponents_of_sqrt_largest_perfect_square_12_factorial : 
  (prime_factor_exponents (largest_perfect_square_divisor (factorial 12)).sqrt).sum = 8 := by sorry

end sum_exponents_of_sqrt_largest_perfect_square_12_factorial_l4079_407916


namespace rattlesnake_count_l4079_407981

theorem rattlesnake_count (total_snakes : ℕ) (boa_constrictors : ℕ) : 
  total_snakes = 200 →
  boa_constrictors = 40 →
  total_snakes = boa_constrictors + 3 * boa_constrictors + (total_snakes - (boa_constrictors + 3 * boa_constrictors)) →
  total_snakes - (boa_constrictors + 3 * boa_constrictors) = 40 :=
by sorry

end rattlesnake_count_l4079_407981


namespace infinite_primes_l4079_407985

theorem infinite_primes : ∀ (S : Finset Nat), (∀ p ∈ S, Nat.Prime p) → ∃ q, Nat.Prime q ∧ q ∉ S := by
  sorry

end infinite_primes_l4079_407985


namespace unique_solution_exponential_equation_l4079_407964

theorem unique_solution_exponential_equation :
  ∀ x : ℝ, (4 : ℝ)^((9 : ℝ)^x) = (9 : ℝ)^((4 : ℝ)^x) ↔ x = 0 := by
sorry

end unique_solution_exponential_equation_l4079_407964


namespace convex_quadrilaterals_count_l4079_407911

/-- The number of ways to choose 4 points from 10 distinct points on a circle's circumference to form convex quadrilaterals -/
def convex_quadrilaterals_from_circle_points (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

/-- Theorem stating that the number of convex quadrilaterals formed from 10 points on a circle is 210 -/
theorem convex_quadrilaterals_count :
  convex_quadrilaterals_from_circle_points 10 4 = 210 := by
  sorry

#eval convex_quadrilaterals_from_circle_points 10 4

end convex_quadrilaterals_count_l4079_407911


namespace plane_distance_l4079_407930

/-- Proves that a plane flying east at 300 km/h and west at 400 km/h for a total of 7 hours travels 1200 km from the airport -/
theorem plane_distance (speed_east speed_west total_time : ℝ) 
  (h_speed_east : speed_east = 300)
  (h_speed_west : speed_west = 400)
  (h_total_time : total_time = 7) :
  ∃ (time_east time_west distance : ℝ),
    time_east + time_west = total_time ∧
    speed_east * time_east = distance ∧
    speed_west * time_west = distance ∧
    distance = 1200 := by
  sorry

end plane_distance_l4079_407930


namespace marbles_left_l4079_407946

theorem marbles_left (total_marbles : ℕ) (num_bags : ℕ) (removed_bags : ℕ) : 
  total_marbles = 28 → 
  num_bags = 4 → 
  removed_bags = 1 → 
  total_marbles % num_bags = 0 → 
  total_marbles - (total_marbles / num_bags * removed_bags) = 21 := by
sorry

end marbles_left_l4079_407946


namespace reciprocal_sum_theorem_l4079_407931

theorem reciprocal_sum_theorem (a b c : ℕ+) : 
  a < b ∧ b < c → 
  (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c = 1 → 
  (a : ℕ) + b + c = 11 := by
  sorry

end reciprocal_sum_theorem_l4079_407931


namespace probability_point_not_in_inner_square_l4079_407917

/-- The probability that a random point in a larger square is not in a smaller square inside it. -/
theorem probability_point_not_in_inner_square
  (area_A : ℝ) (perimeter_B : ℝ)
  (h_area_A : area_A = 65)
  (h_perimeter_B : perimeter_B = 16)
  (h_positive_A : area_A > 0)
  (h_positive_B : perimeter_B > 0) :
  let side_B := perimeter_B / 4
  let area_B := side_B ^ 2
  (area_A - area_B) / area_A = (65 - 16) / 65 := by
  sorry


end probability_point_not_in_inner_square_l4079_407917


namespace isosceles_right_triangle_angle_l4079_407990

theorem isosceles_right_triangle_angle (a h : ℝ) (θ : Real) : 
  a > 0 → -- leg length is positive
  h > 0 → -- hypotenuse length is positive
  h = a * Real.sqrt 2 → -- Pythagorean theorem for isosceles right triangle
  h^2 = 4 * a * Real.cos θ → -- given condition
  0 < θ ∧ θ < Real.pi / 2 → -- θ is an acute angle
  θ = Real.pi / 3 := by
sorry

end isosceles_right_triangle_angle_l4079_407990


namespace square_diff_roots_l4079_407951

theorem square_diff_roots : 
  (Real.sqrt (625681 + 1000) - Real.sqrt 1000)^2 = 
    626681 - 2 * Real.sqrt 626681 * 31.622776601683793 + 1000 := by
  sorry

end square_diff_roots_l4079_407951


namespace exists_integer_square_one_l4079_407919

theorem exists_integer_square_one : ∃ x : ℤ, x^2 = 1 := by sorry

end exists_integer_square_one_l4079_407919


namespace eleventh_term_is_320_l4079_407961

/-- A geometric sequence with given 5th and 8th terms -/
structure GeometricSequence where
  a : ℕ → ℝ
  fifth_term : a 5 = 5
  eighth_term : a 8 = 40

/-- The 11th term of the geometric sequence is 320 -/
theorem eleventh_term_is_320 (seq : GeometricSequence) : seq.a 11 = 320 := by
  sorry

end eleventh_term_is_320_l4079_407961


namespace sum_reciprocal_products_l4079_407980

theorem sum_reciprocal_products (x y z : ℝ) 
  (sum_eq : x + y + z = 6)
  (sum_products_eq : x*y + y*z + z*x = 11)
  (product_eq : x*y*z = 6) :
  x/(y*z) + y/(z*x) + z/(x*y) = 7/3 := by
  sorry

end sum_reciprocal_products_l4079_407980


namespace f_composition_negative_three_l4079_407949

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 1 / (5 - x) else Real.log x / Real.log 4

theorem f_composition_negative_three (f : ℝ → ℝ) :
  (∀ x ≤ 0, f x = 1 / (5 - x)) →
  (∀ x > 0, f x = Real.log x / Real.log 4) →
  f (f (-3)) = -3/2 := by
  sorry

end f_composition_negative_three_l4079_407949


namespace arithmetic_sequence_general_term_l4079_407936

/-- An arithmetic sequence with specific properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
  (a 2 + a 6) / 2 = 5 ∧
  (a 3 + a 7) / 2 = 7

/-- The general term of the arithmetic sequence -/
def GeneralTerm (n : ℕ) : ℝ := 2 * n - 3

theorem arithmetic_sequence_general_term (a : ℕ → ℝ) :
  ArithmeticSequence a → ∀ n : ℕ, a n = GeneralTerm n := by
  sorry

end arithmetic_sequence_general_term_l4079_407936


namespace move_right_2_units_l4079_407928

/-- Moving a point to the right in a 2D coordinate system -/
def move_right (x y dx : ℝ) : ℝ × ℝ :=
  (x + dx, y)

theorem move_right_2_units :
  let A : ℝ × ℝ := (1, 2)
  let A' : ℝ × ℝ := move_right A.1 A.2 2
  A' = (3, 2) := by
  sorry

end move_right_2_units_l4079_407928


namespace range_of_a_for_intersecting_circles_l4079_407992

/-- Two circles intersect if and only if the distance between their centers is greater than
    the absolute difference of their radii and less than the sum of their radii -/
axiom circles_intersect_iff_distance_between_centers (x₁ y₁ x₂ y₂ r₁ r₂ : ℝ) :
  ((x₁ - x₂)^2 + (y₁ - y₂)^2 > (r₁ - r₂)^2) ∧
  ((x₁ - x₂)^2 + (y₁ - y₂)^2 < (r₁ + r₂)^2)

/-- The range of a for intersecting circles -/
theorem range_of_a_for_intersecting_circles (a : ℝ) :
  (∃ x y : ℝ, (x + 2)^2 + (y - a)^2 = 1 ∧ (x - a)^2 + (y - 5)^2 = 16) ↔
  1 < a ∧ a < 2 :=
sorry

end range_of_a_for_intersecting_circles_l4079_407992


namespace rals_current_age_l4079_407923

/-- Given that Ral is three times as old as Suri, and in 6 years Suri's age will be 25,
    prove that Ral's current age is 57 years. -/
theorem rals_current_age (suri_age suri_future_age ral_age : ℕ) 
    (h1 : ral_age = 3 * suri_age)
    (h2 : suri_future_age = suri_age + 6)
    (h3 : suri_future_age = 25) : 
  ral_age = 57 := by
  sorry

end rals_current_age_l4079_407923


namespace rectangle_perimeter_l4079_407934

/-- A rectangle with length thrice its breadth and area 108 square meters has a perimeter of 48 meters -/
theorem rectangle_perimeter (b : ℝ) (h1 : b > 0) (h2 : 3 * b * b = 108) : 2 * (3 * b + b) = 48 := by
  sorry

end rectangle_perimeter_l4079_407934


namespace six_stairs_ways_l4079_407926

/-- The number of ways to climb n stairs, taking 1, 2, or 3 stairs at a time -/
def stairClimbWays (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | n + 3 => stairClimbWays (n + 2) + stairClimbWays (n + 1) + stairClimbWays n

theorem six_stairs_ways :
  stairClimbWays 6 = 24 := by
  sorry

end six_stairs_ways_l4079_407926


namespace unsprinkled_bricks_count_l4079_407927

/-- Represents a rectangular solid pile of bricks -/
structure BrickPile where
  length : Nat
  width : Nat
  height : Nat

/-- Calculates the number of bricks not sprinkled with lime water -/
def unsprinkledBricks (pile : BrickPile) : Nat :=
  pile.length * pile.width * pile.height - 
  (pile.length - 2) * (pile.width - 2) * (pile.height - 2)

/-- Theorem stating that the number of unsprinkled bricks in a 30x20x10 pile is 4032 -/
theorem unsprinkled_bricks_count :
  let pile : BrickPile := { length := 30, width := 20, height := 10 }
  unsprinkledBricks pile = 4032 := by
  sorry

end unsprinkled_bricks_count_l4079_407927


namespace range_of_m_l4079_407952

-- Define a decreasing function on [-1, 1]
def IsDecreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ∈ [-1, 1] → y ∈ [-1, 1] → x < y → f x > f y

-- Main theorem
theorem range_of_m (f : ℝ → ℝ) (m : ℝ) 
  (h_decreasing : IsDecreasingOn f)
  (h_inequality : f (m - 1) > f (2 * m - 1)) :
  m ∈ Set.Ioo 0 1 := by
  sorry

#check range_of_m

end range_of_m_l4079_407952


namespace sphere_volume_area_ratio_l4079_407982

theorem sphere_volume_area_ratio (r R : ℝ) (h : r > 0) (H : R > 0) :
  (4 / 3 * Real.pi * r^3) / (4 / 3 * Real.pi * R^3) = 1 / 8 →
  (4 * Real.pi * r^2) / (4 * Real.pi * R^2) = 1 / 4 := by
sorry

end sphere_volume_area_ratio_l4079_407982


namespace water_consumption_l4079_407968

theorem water_consumption (W : ℝ) : 
  W > 0 →
  (W - 0.2 * W - 0.35 * (W - 0.2 * W) = 130) →
  W = 250 := by
sorry

end water_consumption_l4079_407968


namespace andrew_mango_purchase_l4079_407920

-- Define the given constants
def grape_quantity : ℕ := 6
def grape_price : ℕ := 74
def mango_price : ℕ := 59
def total_paid : ℕ := 975

-- Define the function to calculate the mango quantity
def mango_quantity : ℕ := (total_paid - grape_quantity * grape_price) / mango_price

-- Theorem statement
theorem andrew_mango_purchase :
  mango_quantity = 9 := by
  sorry

end andrew_mango_purchase_l4079_407920


namespace max_sides_cube_cross_section_l4079_407908

/-- A cube is a three-dimensional solid object with six square faces. -/
structure Cube where

/-- A plane is a flat, two-dimensional surface that extends infinitely far. -/
structure Plane where

/-- A cross-section is the intersection of a plane with a three-dimensional object. -/
def CrossSection (c : Cube) (p : Plane) : Set (ℝ × ℝ × ℝ) := sorry

/-- The number of sides in a polygon. -/
def NumberOfSides (polygon : Set (ℝ × ℝ × ℝ)) : ℕ := sorry

/-- The maximum number of sides in any cross-section of a cube is 6. -/
theorem max_sides_cube_cross_section (c : Cube) : 
  ∀ p : Plane, NumberOfSides (CrossSection c p) ≤ 6 ∧ 
  ∃ p : Plane, NumberOfSides (CrossSection c p) = 6 :=
sorry

end max_sides_cube_cross_section_l4079_407908


namespace gcd_factorial_problem_l4079_407924

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 7) ((Nat.factorial 9) / (Nat.factorial 4)) = 5040 := by
  sorry

end gcd_factorial_problem_l4079_407924
