import Mathlib

namespace NUMINAMATH_CALUDE_vanessa_points_l1136_113630

/-- Calculates the points scored by a player given the total team points,
    number of other players, and average points of other players. -/
def player_points (total_points : ℕ) (other_players : ℕ) (avg_other_points : ℚ) : ℚ :=
  total_points - other_players * avg_other_points

/-- Proves that Vanessa scored 27 points given the problem conditions. -/
theorem vanessa_points :
  let total_points : ℕ := 48
  let other_players : ℕ := 6
  let avg_other_points : ℚ := 7/2
  player_points total_points other_players avg_other_points = 27 := by
sorry

#eval player_points 48 6 (7/2)

end NUMINAMATH_CALUDE_vanessa_points_l1136_113630


namespace NUMINAMATH_CALUDE_rotation_theorem_l1136_113664

/-- The original line -/
def original_line (x y : ℝ) : Prop := 2 * x - y - 2 = 0

/-- The point where the original line intersects the y-axis -/
def intersection_point : ℝ × ℝ := (0, -2)

/-- The rotated line -/
def rotated_line (x y : ℝ) : Prop := x + 2 * y + 4 = 0

/-- Theorem stating that rotating the original line 90° counterclockwise around the intersection point results in the rotated line -/
theorem rotation_theorem :
  ∀ (x y : ℝ),
  original_line x y →
  ∃ (x' y' : ℝ),
  (x' - intersection_point.1) ^ 2 + (y' - intersection_point.2) ^ 2 = (x - intersection_point.1) ^ 2 + (y - intersection_point.2) ^ 2 ∧
  (x' - intersection_point.1) * (x - intersection_point.1) + (y' - intersection_point.2) * (y - intersection_point.2) = 0 ∧
  rotated_line x' y' :=
sorry

end NUMINAMATH_CALUDE_rotation_theorem_l1136_113664


namespace NUMINAMATH_CALUDE_concert_duration_13h25m_l1136_113654

/-- Calculates the total duration in minutes of a concert given its length in hours and minutes. -/
def concert_duration (hours : ℕ) (minutes : ℕ) : ℕ :=
  hours * 60 + minutes

/-- Theorem stating that a concert lasting 13 hours and 25 minutes has a total duration of 805 minutes. -/
theorem concert_duration_13h25m :
  concert_duration 13 25 = 805 := by
  sorry

end NUMINAMATH_CALUDE_concert_duration_13h25m_l1136_113654


namespace NUMINAMATH_CALUDE_black_pens_per_student_l1136_113675

/-- Represents the problem of calculating the number of black pens each student received. -/
theorem black_pens_per_student (num_students : ℕ) (red_pens_per_student : ℕ) 
  (pens_taken_first_month : ℕ) (pens_taken_second_month : ℕ) (remaining_pens_per_student : ℕ) :
  num_students = 3 →
  red_pens_per_student = 62 →
  pens_taken_first_month = 37 →
  pens_taken_second_month = 41 →
  remaining_pens_per_student = 79 →
  (num_students * (red_pens_per_student + 43) - pens_taken_first_month - pens_taken_second_month) / num_students = remaining_pens_per_student :=
by sorry

#check black_pens_per_student

end NUMINAMATH_CALUDE_black_pens_per_student_l1136_113675


namespace NUMINAMATH_CALUDE_solution_count_l1136_113672

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem solution_count :
  let gcd_value := factorial 20
  let lcm_value := factorial 30
  (Finset.filter (fun p : ℕ × ℕ => 
    Nat.gcd p.1 p.2 = gcd_value ∧ 
    Nat.lcm p.1 p.2 = lcm_value
  ) (Finset.product (Finset.range (lcm_value + 1)) (Finset.range (lcm_value + 1)))).card = 1024 := by
  sorry

end NUMINAMATH_CALUDE_solution_count_l1136_113672


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l1136_113611

/-- A pyramid with a square base and equilateral triangle lateral faces -/
structure Pyramid :=
  (base_side : ℝ)
  (base_is_square : base_side > 0)
  (lateral_faces_equilateral : True)

/-- A cube inscribed in a pyramid -/
structure InscribedCube (p : Pyramid) :=
  (side_length : ℝ)
  (base_on_pyramid_base : True)
  (top_face_on_lateral_faces : True)

/-- The volume of the inscribed cube -/
def cube_volume (p : Pyramid) (c : InscribedCube p) : ℝ :=
  c.side_length ^ 3

/-- The theorem stating the volume of the inscribed cube -/
theorem inscribed_cube_volume (p : Pyramid) (c : InscribedCube p)
  (h : p.base_side = 1) :
  cube_volume p c = 5 * Real.sqrt 2 - 7 :=
sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l1136_113611


namespace NUMINAMATH_CALUDE_first_five_average_l1136_113640

theorem first_five_average (total_average : ℝ) (last_seven_average : ℝ) (fifth_result : ℝ) :
  total_average = 42 →
  last_seven_average = 52 →
  fifth_result = 147 →
  (5 * ((11 * total_average - (7 * last_seven_average - fifth_result)) / 5) = 245) ∧
  ((11 * total_average - (7 * last_seven_average - fifth_result)) / 5 = 49) :=
by
  sorry

end NUMINAMATH_CALUDE_first_five_average_l1136_113640


namespace NUMINAMATH_CALUDE_largest_n_for_integer_differences_l1136_113628

theorem largest_n_for_integer_differences : ∃ (x₁ x₂ x₃ x₄ y₁ y₂ y₃ y₄ : ℤ),
  (∀ k : ℕ, k ≤ 9 → 
    (∃ (i j : Fin 4), i < j ∧ (k : ℤ) = |x₁ - x₂| ∨ k = |x₁ - x₃| ∨ k = |x₁ - x₄| ∨ 
                               k = |x₂ - x₃| ∨ k = |x₂ - x₄| ∨ k = |x₃ - x₄| ∨
                               k = |y₁ - y₂| ∨ k = |y₁ - y₃| ∨ k = |y₁ - y₄| ∨ 
                               k = |y₂ - y₃| ∨ k = |y₂ - y₄| ∨ k = |y₃ - y₄|)) ∧
  (∀ n : ℕ, n > 9 → 
    ¬∃ (a₁ a₂ a₃ a₄ b₁ b₂ b₃ b₄ : ℤ),
      ∀ k : ℕ, k ≤ n → 
        (∃ (i j : Fin 4), i < j ∧ (k : ℤ) = |a₁ - a₂| ∨ k = |a₁ - a₃| ∨ k = |a₁ - a₄| ∨ 
                                   k = |a₂ - a₃| ∨ k = |a₂ - a₄| ∨ k = |a₃ - a₄| ∨
                                   k = |b₁ - b₂| ∨ k = |b₁ - b₃| ∨ k = |b₁ - b₄| ∨ 
                                   k = |b₂ - b₃| ∨ k = |b₂ - b₄| ∨ k = |b₃ - b₄|)) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_integer_differences_l1136_113628


namespace NUMINAMATH_CALUDE_largest_quotient_is_15_l1136_113623

def S : Set Int := {-30, -4, -2, 2, 4, 10}

def is_even (n : Int) : Prop := ∃ k : Int, n = 2 * k

theorem largest_quotient_is_15 :
  ∀ a b : Int,
    a ∈ S → b ∈ S →
    is_even a → is_even b →
    a < 0 → b > 0 →
    (-a : ℚ) / b ≤ 15 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_quotient_is_15_l1136_113623


namespace NUMINAMATH_CALUDE_quadratic_prime_roots_l1136_113673

theorem quadratic_prime_roots (k : ℕ) : 
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p + q = 63 ∧ p * q = k ∧ ∀ x : ℝ, x^2 - 63*x + k = 0 ↔ (x = p ∨ x = q)) → 
  k = 122 :=
sorry

end NUMINAMATH_CALUDE_quadratic_prime_roots_l1136_113673


namespace NUMINAMATH_CALUDE_greatest_four_digit_multiple_of_17_l1136_113653

theorem greatest_four_digit_multiple_of_17 : ∃ n : ℕ, 
  n ≤ 9999 ∧ 
  n > 999 ∧
  n % 17 = 0 ∧
  ∀ m : ℕ, m ≤ 9999 ∧ m > 999 ∧ m % 17 = 0 → m ≤ n :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_greatest_four_digit_multiple_of_17_l1136_113653


namespace NUMINAMATH_CALUDE_equal_tuesdays_thursdays_30_days_l1136_113649

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- A function that determines if a given day is a valid starting day for a 30-day month with equal Tuesdays and Thursdays -/
def isValidStartDay (d : DayOfWeek) : Prop := sorry

/-- The number of valid starting days for a 30-day month with equal Tuesdays and Thursdays -/
def numValidStartDays : ℕ := sorry

theorem equal_tuesdays_thursdays_30_days :
  numValidStartDays = 5 := by sorry

end NUMINAMATH_CALUDE_equal_tuesdays_thursdays_30_days_l1136_113649


namespace NUMINAMATH_CALUDE_similarity_condition_l1136_113618

theorem similarity_condition (a b : ℝ) :
  (∃ h : ℝ → ℝ, 
    (∀ y : ℝ, ∃ x : ℝ, h x = y) ∧ 
    (∀ x₁ x₂ : ℝ, h x₁ = h x₂ → x₁ = x₂) ∧
    (∀ x : ℝ, h (x^2 + a*x + b) = (h x)^2)) →
  b = a*(a + 2)/4 := by
sorry

end NUMINAMATH_CALUDE_similarity_condition_l1136_113618


namespace NUMINAMATH_CALUDE_discount_calculation_l1136_113641

/-- Calculates the discounted price of an item given the original price and discount rate. -/
def discounted_price (original_price : ℝ) (discount_rate : ℝ) : ℝ :=
  original_price * (1 - discount_rate)

/-- Proves that a 20% discount on a $120 item results in a price of $96. -/
theorem discount_calculation :
  let original_price : ℝ := 120
  let discount_rate : ℝ := 0.2
  discounted_price original_price discount_rate = 96 := by
  sorry

end NUMINAMATH_CALUDE_discount_calculation_l1136_113641


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1136_113668

theorem p_necessary_not_sufficient_for_q :
  (∀ x : ℝ, (5 * x - 6 > x^2) → (|x + 1| > 2)) ∧
  (∃ x : ℝ, (|x + 1| > 2) ∧ ¬(5 * x - 6 > x^2)) := by
  sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1136_113668


namespace NUMINAMATH_CALUDE_quadratic_completing_square_l1136_113691

theorem quadratic_completing_square :
  ∀ x : ℝ, (x^2 - 4*x - 6 = 0) ↔ ((x - 2)^2 = 10) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_completing_square_l1136_113691


namespace NUMINAMATH_CALUDE_lucas_february_bill_l1136_113666

/-- Calculates the total cost of a cell phone plan based on given parameters. -/
def calculate_phone_bill (base_cost : ℚ) (text_cost : ℚ) (extra_cost_30_31 : ℚ) 
  (extra_cost_beyond_31 : ℚ) (num_texts : ℕ) (talk_time : ℚ) : ℚ :=
  let text_total := num_texts * text_cost
  let extra_time := max (talk_time - 30) 0
  let extra_cost := 
    if extra_time ≤ 1 then
      extra_time * 60 * extra_cost_30_31
    else
      60 * extra_cost_30_31 + (extra_time - 1) * 60 * extra_cost_beyond_31
  base_cost + text_total + extra_cost

/-- Theorem stating that Lucas's phone bill for February is $55.00 -/
theorem lucas_february_bill : 
  calculate_phone_bill 25 0.1 0.15 0.2 150 31.5 = 55 := by
  sorry


end NUMINAMATH_CALUDE_lucas_february_bill_l1136_113666


namespace NUMINAMATH_CALUDE_fraction_problem_l1136_113694

theorem fraction_problem (N : ℝ) (F : ℝ) : 
  (3/10 : ℝ) * N = 64.8 →
  F * ((1/4 : ℝ) * N) = 18 →
  F = 1/3 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l1136_113694


namespace NUMINAMATH_CALUDE_points_on_line_implies_a_equals_two_l1136_113642

-- Define the points
def A : ℝ × ℝ := (2, 2)
def B : ℝ × ℝ := (5, 1)
def C (a : ℝ) : ℝ × ℝ := (-4, 2*a)

-- Define the condition for points being on the same line
def collinear (p q r : ℝ × ℝ) : Prop :=
  (q.2 - p.2) * (r.1 - p.1) = (r.2 - p.2) * (q.1 - p.1)

-- Theorem statement
theorem points_on_line_implies_a_equals_two :
  collinear A B (C a) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_points_on_line_implies_a_equals_two_l1136_113642


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1136_113631

theorem polynomial_remainder (x : ℝ) : 
  let p := fun x => 5*x^4 - 9*x^3 + 3*x^2 - 7*x - 30
  let d := fun x => 3*x - 9
  p 3 = 138 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1136_113631


namespace NUMINAMATH_CALUDE_bc_length_l1136_113652

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the properties of the triangle
def isObtuseTriangle (t : Triangle) : Prop := sorry

def triangleArea (t : Triangle) : ℝ := sorry

def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem bc_length (ABC : Triangle) 
  (h1 : isObtuseTriangle ABC)
  (h2 : triangleArea ABC = 10 * Real.sqrt 3)
  (h3 : distance ABC.A ABC.B = 5)
  (h4 : distance ABC.A ABC.C = 8) :
  distance ABC.B ABC.C = Real.sqrt 129 := by sorry

end NUMINAMATH_CALUDE_bc_length_l1136_113652


namespace NUMINAMATH_CALUDE_scientific_notation_conversion_l1136_113659

theorem scientific_notation_conversion :
  (1.8 : ℝ) * (10 ^ 8) = 180000000 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_conversion_l1136_113659


namespace NUMINAMATH_CALUDE_unique_solution_is_six_l1136_113690

theorem unique_solution_is_six :
  ∃! n : ℤ, ⌊(n^2 : ℚ) / 3⌋ - ⌊(n : ℚ) / 2⌋^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_is_six_l1136_113690


namespace NUMINAMATH_CALUDE_range_of_a_l1136_113651

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 2) * Real.exp x + a * (Real.log x - x + 1)

theorem range_of_a (a : ℝ) :
  (∀ x > 0, f a x ≥ -Real.exp 1) →
  (∃ x > 0, f a x = -Real.exp 1) →
  a ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1136_113651


namespace NUMINAMATH_CALUDE_rectangle_square_ratio_l1136_113634

/-- Represents a right triangle with a rectangle and square inscribed as described in the problem -/
structure TriangleWithInscriptions where
  /-- Side lengths of the right triangle -/
  a : ℝ
  b : ℝ
  c : ℝ
  /-- Side lengths of the inscribed rectangle -/
  rect_side1 : ℝ
  rect_side2 : ℝ
  /-- Side length of the inscribed square -/
  square_side : ℝ
  /-- Conditions for the triangle -/
  triangle_right : a^2 + b^2 = c^2
  triangle_sides : a = 5 ∧ b = 12 ∧ c = 13
  /-- Conditions for the rectangle -/
  rectangle_sides : rect_side1 = 5 ∧ rect_side2 = 12
  /-- Condition for the square -/
  square_formula : square_side = (a * b) / c

/-- The main theorem stating the ratio of the longer rectangle side to the square side -/
theorem rectangle_square_ratio (t : TriangleWithInscriptions) :
  t.rect_side2 / t.square_side = 13 / 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_square_ratio_l1136_113634


namespace NUMINAMATH_CALUDE_smallest_pascal_family_pascal_family_with_five_children_l1136_113674

/-- Represents a family with children -/
structure Family :=
  (boys : ℕ)
  (girls : ℕ)

/-- Defines the conditions for the Pascal family -/
def isPascalFamily (f : Family) : Prop :=
  f.boys ≥ 3 ∧ f.girls ≥ 2

/-- The total number of children in a family -/
def totalChildren (f : Family) : ℕ := f.boys + f.girls

/-- Theorem: The smallest possible number of children in a Pascal family is 5 -/
theorem smallest_pascal_family :
  ∀ f : Family, isPascalFamily f → totalChildren f ≥ 5 :=
by
  sorry

/-- Theorem: There exists a Pascal family with exactly 5 children -/
theorem pascal_family_with_five_children :
  ∃ f : Family, isPascalFamily f ∧ totalChildren f = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_pascal_family_pascal_family_with_five_children_l1136_113674


namespace NUMINAMATH_CALUDE_imo_1996_p5_l1136_113677

theorem imo_1996_p5 (n p q : ℕ+) (x : ℕ → ℤ)
  (h_npq : n > p + q)
  (h_x0 : x 0 = 0)
  (h_xn : x n = 0)
  (h_diff : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → (x i - x (i-1) = p ∨ x i - x (i-1) = -q)) :
  ∃ i j : ℕ, 0 ≤ i ∧ i < j ∧ j ≤ n ∧ (i, j) ≠ (0, n) ∧ x i = x j :=
sorry

end NUMINAMATH_CALUDE_imo_1996_p5_l1136_113677


namespace NUMINAMATH_CALUDE_trip_time_difference_l1136_113660

-- Define the given conditions
def speed : ℝ := 60
def distance1 : ℝ := 510
def distance2 : ℝ := 540

-- Define the theorem
theorem trip_time_difference : 
  (distance2 - distance1) / speed * 60 = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_trip_time_difference_l1136_113660


namespace NUMINAMATH_CALUDE_max_primes_arithmetic_seq_diff_12_l1136_113606

theorem max_primes_arithmetic_seq_diff_12 :
  ∀ (seq : ℕ → ℕ) (n : ℕ),
    (∀ i < n, seq i.succ = seq i + 12) →
    (∀ i < n, Nat.Prime (seq i)) →
    n ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_max_primes_arithmetic_seq_diff_12_l1136_113606


namespace NUMINAMATH_CALUDE_manuscript_completion_time_l1136_113602

/-- The time needed to complete the manuscript when two people work together after one has worked alone for some time. -/
theorem manuscript_completion_time
  (time_A : ℝ) -- Time for person A to complete the manuscript alone
  (time_B : ℝ) -- Time for person B to complete the manuscript alone
  (solo_work : ℝ) -- Time person A works alone before B joins
  (h_A_positive : time_A > 0)
  (h_B_positive : time_B > 0)
  (h_solo_work : 0 ≤ solo_work ∧ solo_work < time_A) :
  let remaining_time := (time_A * time_B - solo_work * time_B) / (time_A + time_B)
  remaining_time = 24 / 13 :=
by sorry

end NUMINAMATH_CALUDE_manuscript_completion_time_l1136_113602


namespace NUMINAMATH_CALUDE_peculiar_quadratic_minimum_l1136_113638

/-- A quadratic polynomial q(x) = x^2 + bx + c is peculiar if q(q(x)) = 0 has exactly four real roots, including a triple root. -/
def IsPeculiar (q : ℝ → ℝ) : Prop :=
  ∃ b c : ℝ, (∀ x, q x = x^2 + b*x + c) ∧
  (∃ r₁ r₂ r₃ r₄ : ℝ, (∀ x, q (q x) = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄) ∧
  (r₁ = r₂ ∧ r₂ = r₃ ∧ r₃ ≠ r₄))

theorem peculiar_quadratic_minimum :
  ∃! q : ℝ → ℝ, IsPeculiar q ∧
  (∀ p : ℝ → ℝ, IsPeculiar p → q 0 ≤ p 0) ∧
  (∀ x, q x = x^2 - 1/2) ∧
  q 0 = -1/2 := by sorry

end NUMINAMATH_CALUDE_peculiar_quadratic_minimum_l1136_113638


namespace NUMINAMATH_CALUDE_wilson_fraction_problem_l1136_113620

theorem wilson_fraction_problem (N F : ℚ) : 
  N = 8 → N - F * N = 16/3 → F = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_wilson_fraction_problem_l1136_113620


namespace NUMINAMATH_CALUDE_uniqueness_not_algorithm_characteristic_l1136_113663

/-- Represents characteristics of an algorithm -/
inductive AlgorithmCharacteristic
  | Abstraction
  | Precision
  | Finiteness
  | Uniqueness

/-- Predicate to check if a given characteristic is a valid algorithm characteristic -/
def isValidAlgorithmCharacteristic (c : AlgorithmCharacteristic) : Prop :=
  match c with
  | AlgorithmCharacteristic.Abstraction => True
  | AlgorithmCharacteristic.Precision => True
  | AlgorithmCharacteristic.Finiteness => True
  | AlgorithmCharacteristic.Uniqueness => False

theorem uniqueness_not_algorithm_characteristic :
  ¬(isValidAlgorithmCharacteristic AlgorithmCharacteristic.Uniqueness) :=
by sorry

end NUMINAMATH_CALUDE_uniqueness_not_algorithm_characteristic_l1136_113663


namespace NUMINAMATH_CALUDE_french_not_english_speakers_l1136_113688

/-- The number of students who speak French but not English in a survey -/
theorem french_not_english_speakers (total : ℕ) (french_speakers : ℕ) (both_speakers : ℕ) 
  (h1 : total = 200)
  (h2 : french_speakers = total / 4)
  (h3 : both_speakers = 10) :
  french_speakers - both_speakers = 40 := by
  sorry

end NUMINAMATH_CALUDE_french_not_english_speakers_l1136_113688


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_problem_l1136_113643

theorem arithmetic_geometric_mean_problem (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 20)
  (h_geometric : Real.sqrt (x * y) = Real.sqrt 96) :
  x^2 + y^2 = 1408 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_problem_l1136_113643


namespace NUMINAMATH_CALUDE_bee_count_l1136_113680

theorem bee_count (initial_bees : ℕ) (h1 : initial_bees = 144) : 
  ⌊(3 * initial_bees : ℚ) * (1 - 0.2)⌋ = 346 := by
  sorry

end NUMINAMATH_CALUDE_bee_count_l1136_113680


namespace NUMINAMATH_CALUDE_coins_can_be_all_heads_l1136_113615

/-- Represents the state of a coin (heads or tails) -/
inductive CoinState
| Heads
| Tails

/-- Represents a sequence of 100 coins -/
def CoinSequence := Fin 100 → CoinState

/-- Represents an operation that flips 7 coins at equal intervals -/
structure FlipOperation where
  start : Fin 100  -- Starting position of the flip
  interval : Nat   -- Interval between flipped coins
  valid : start.val + 6 * interval < 100  -- Ensure operation is within bounds

/-- Applies a flip operation to a coin sequence -/
def applyFlip (seq : CoinSequence) (op : FlipOperation) : CoinSequence :=
  λ i => if ∃ k : Fin 7, i.val = op.start.val + k.val * op.interval
         then match seq i with
              | CoinState.Heads => CoinState.Tails
              | CoinState.Tails => CoinState.Heads
         else seq i

/-- Checks if all coins in the sequence are heads -/
def allHeads (seq : CoinSequence) : Prop :=
  ∀ i : Fin 100, seq i = CoinState.Heads

/-- The main theorem: it's possible to make all coins heads -/
theorem coins_can_be_all_heads :
  ∀ (initial : CoinSequence),
  ∃ (ops : List FlipOperation),
  allHeads (ops.foldl applyFlip initial) :=
sorry

end NUMINAMATH_CALUDE_coins_can_be_all_heads_l1136_113615


namespace NUMINAMATH_CALUDE_x_fourth_plus_inverse_fourth_l1136_113637

theorem x_fourth_plus_inverse_fourth (x : ℝ) (h : x + 1/x = 3) :
  x^4 + 1/x^4 = 47 := by sorry

end NUMINAMATH_CALUDE_x_fourth_plus_inverse_fourth_l1136_113637


namespace NUMINAMATH_CALUDE_possible_values_of_a_l1136_113617

def P : Set ℝ := {x | x^2 + x - 6 = 0}
def S (a : ℝ) : Set ℝ := {x | a * x + 1 = 0}

theorem possible_values_of_a :
  ∀ a : ℝ, (S a ⊆ P) ↔ (a = 0 ∨ a = 1/3 ∨ a = -1/2) :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l1136_113617


namespace NUMINAMATH_CALUDE_parabola_hyperbola_focus_coincide_l1136_113667

/-- The value of p for which the focus of the parabola y^2 = 2px coincides with 
    the right focus of the hyperbola x^2/3 - y^2/1 = 1 -/
theorem parabola_hyperbola_focus_coincide : ∃ p : ℝ, 
  (∀ x y : ℝ, y^2 = 2*p*x → x^2/3 - y^2 = 1 → 
   ∃ f : ℝ × ℝ, f = (p, 0) ∧ f = (2, 0)) → 
  p = 2 := by sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_focus_coincide_l1136_113667


namespace NUMINAMATH_CALUDE_match_box_dozens_l1136_113669

theorem match_box_dozens (total_matches : ℕ) (matches_per_box : ℕ) (boxes_per_dozen : ℕ) : 
  total_matches = 1200 →
  matches_per_box = 20 →
  boxes_per_dozen = 12 →
  (total_matches / matches_per_box) / boxes_per_dozen = 5 :=
by sorry

end NUMINAMATH_CALUDE_match_box_dozens_l1136_113669


namespace NUMINAMATH_CALUDE_no_solution_exists_l1136_113608

theorem no_solution_exists : ¬∃ (n k : ℕ), 
  n ≠ 0 ∧ k ≠ 0 ∧ (n ∣ k^n - 1) ∧ Nat.gcd n (k - 1) = 1 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1136_113608


namespace NUMINAMATH_CALUDE_medication_price_reduction_l1136_113650

theorem medication_price_reduction (a : ℝ) :
  let new_price := a
  let reduction_rate := 0.4
  let original_price := (5 / 3) * a
  (1 - reduction_rate) * original_price = new_price :=
by sorry

end NUMINAMATH_CALUDE_medication_price_reduction_l1136_113650


namespace NUMINAMATH_CALUDE_line_vector_proof_l1136_113655

def line_vector (t : ℝ) : ℝ × ℝ × ℝ := sorry

theorem line_vector_proof :
  (line_vector 1 = (2, -3, 5) ∧ line_vector 4 = (-2, 9, -11)) →
  line_vector 5 = (-10/3, 13, -49/3) := by
  sorry

end NUMINAMATH_CALUDE_line_vector_proof_l1136_113655


namespace NUMINAMATH_CALUDE_max_distance_complex_l1136_113670

theorem max_distance_complex (z : ℂ) (h : Complex.abs z = 1) :
  Complex.abs ((1 + 2*Complex.I)*z^3 - z^6) ≤ Real.sqrt 5 + 1 ∧
  ∃ z₀ : ℂ, Complex.abs z₀ = 1 ∧ Complex.abs ((1 + 2*Complex.I)*z₀^3 - z₀^6) = Real.sqrt 5 + 1 :=
sorry

end NUMINAMATH_CALUDE_max_distance_complex_l1136_113670


namespace NUMINAMATH_CALUDE_exp_inequality_l1136_113683

/-- The function f(x) = (x-3)³ + 2x - 6 -/
def f (x : ℝ) : ℝ := (x - 3)^3 + 2*x - 6

/-- Theorem stating that if f(2a-b) + f(6-b) > 0, then e^a > e^b -/
theorem exp_inequality (a b : ℝ) (h : f (2*a - b) + f (6 - b) > 0) : Real.exp a > Real.exp b := by
  sorry

end NUMINAMATH_CALUDE_exp_inequality_l1136_113683


namespace NUMINAMATH_CALUDE_samantha_bus_time_l1136_113693

/-- Represents Samantha's daily schedule --/
structure Schedule where
  wakeUpTime : Nat
  busTime : Nat
  classCount : Nat
  classDuration : Nat
  lunchDuration : Nat
  chessClubDuration : Nat
  arrivalTime : Nat

/-- Calculates the total time spent on the bus given a schedule --/
def busTimeDuration (s : Schedule) : Nat :=
  let totalAwayTime := s.arrivalTime - s.busTime
  let totalSchoolTime := s.classCount * s.classDuration + s.lunchDuration + s.chessClubDuration
  totalAwayTime - totalSchoolTime

/-- Samantha's actual schedule --/
def samanthaSchedule : Schedule :=
  { wakeUpTime := 7 * 60
    busTime := 8 * 60
    classCount := 7
    classDuration := 45
    lunchDuration := 45
    chessClubDuration := 90
    arrivalTime := 17 * 60 + 30 }

/-- Theorem stating that Samantha spends 120 minutes on the bus --/
theorem samantha_bus_time :
  busTimeDuration samanthaSchedule = 120 := by
  sorry

end NUMINAMATH_CALUDE_samantha_bus_time_l1136_113693


namespace NUMINAMATH_CALUDE_no_prime_sum_10003_l1136_113657

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

theorem no_prime_sum_10003 : ¬∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 10003 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_sum_10003_l1136_113657


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_two_l1136_113656

theorem sqrt_sum_equals_two : 
  Real.sqrt (4 + 2 * Real.sqrt 3) + Real.sqrt (4 - 2 * Real.sqrt 3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_two_l1136_113656


namespace NUMINAMATH_CALUDE_original_savings_proof_l1136_113629

def lindas_savings : ℝ := 880
def tv_cost : ℝ := 220

theorem original_savings_proof :
  (1 / 4 : ℝ) * lindas_savings = tv_cost →
  lindas_savings = 880 := by
sorry

end NUMINAMATH_CALUDE_original_savings_proof_l1136_113629


namespace NUMINAMATH_CALUDE_mod_inverse_89_mod_90_l1136_113647

theorem mod_inverse_89_mod_90 : ∃ x : ℕ, 0 ≤ x ∧ x < 90 ∧ (89 * x) % 90 = 1 :=
by
  use 89
  sorry

end NUMINAMATH_CALUDE_mod_inverse_89_mod_90_l1136_113647


namespace NUMINAMATH_CALUDE_sin_4theta_from_exp_itheta_l1136_113681

theorem sin_4theta_from_exp_itheta (θ : ℝ) :
  Complex.exp (θ * Complex.I) = (2 + Complex.I * Real.sqrt 5) / 3 →
  Real.sin (4 * θ) = -8 * Real.sqrt 5 / 81 := by
  sorry

end NUMINAMATH_CALUDE_sin_4theta_from_exp_itheta_l1136_113681


namespace NUMINAMATH_CALUDE_integer_solutions_cubic_equation_l1136_113636

theorem integer_solutions_cubic_equation :
  ∀ x y : ℤ, y^2 = x^3 + (x + 1)^2 ↔ (x = 0 ∧ y = 1) ∨ (x = 0 ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_cubic_equation_l1136_113636


namespace NUMINAMATH_CALUDE_tangent_line_of_g_l1136_113605

/-- Given a function f with a tangent line y = 2x - 1 at (2, f(2)),
    the tangent line to g(x) = x^2 + f(x) at (2, g(2)) is 6x - y - 5 = 0 -/
theorem tangent_line_of_g (f : ℝ → ℝ) (h : HasDerivAt f 2 2) :
  let g := λ x => x^2 + f x
  ∃ L : ℝ → ℝ, HasDerivAt g 6 2 ∧ L x = 6*x - 5 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_of_g_l1136_113605


namespace NUMINAMATH_CALUDE_multiplication_puzzle_l1136_113603

theorem multiplication_puzzle (c d : ℕ) : 
  c < 10 → d < 10 →
  (∃ n : ℕ, n < 1000 ∧ n % 100 = 8 ∧ 3 * c * 10 + c = n / d / 10) →
  c * 4 % 10 = 2 →
  (∃ x : ℕ, x < 10 ∧ 34 * d + 12 ≥ 10 * x * 10 + 60 ∧ 34 * d + 12 < 10 * x * 10 + 70) →
  c + d = 5 := by
sorry

end NUMINAMATH_CALUDE_multiplication_puzzle_l1136_113603


namespace NUMINAMATH_CALUDE_max_value_of_sum_products_l1136_113627

theorem max_value_of_sum_products (a b c d : ℝ) 
  (nonneg_a : 0 ≤ a) (nonneg_b : 0 ≤ b) (nonneg_c : 0 ≤ c) (nonneg_d : 0 ≤ d)
  (sum_constraint : a + b + c + d = 200) :
  ab + bc + cd ≤ 10000 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_sum_products_l1136_113627


namespace NUMINAMATH_CALUDE_square_plot_area_l1136_113607

/-- Given a square plot with a fence around it, if the price per foot of the fence is 58 and
    the total cost is 2088, then the area of the square plot is 81 square feet. -/
theorem square_plot_area (price_per_foot : ℝ) (total_cost : ℝ) (side_length : ℝ) :
  price_per_foot = 58 →
  total_cost = 2088 →
  total_cost = 4 * side_length * price_per_foot →
  side_length ^ 2 = 81 :=
by sorry

end NUMINAMATH_CALUDE_square_plot_area_l1136_113607


namespace NUMINAMATH_CALUDE_determinant_of_specific_matrix_l1136_113614

theorem determinant_of_specific_matrix :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![7, -2; -3, 6]
  Matrix.det A = 36 := by
  sorry

end NUMINAMATH_CALUDE_determinant_of_specific_matrix_l1136_113614


namespace NUMINAMATH_CALUDE_simple_annual_interest_rate_l1136_113679

/-- Calculate the simple annual interest rate given monthly interest payment and principal amount -/
theorem simple_annual_interest_rate 
  (monthly_interest : ℝ) 
  (principal : ℝ) 
  (h1 : monthly_interest = 234)
  (h2 : principal = 31200) :
  (monthly_interest * 12 / principal) * 100 = 8.99 := by
sorry

end NUMINAMATH_CALUDE_simple_annual_interest_rate_l1136_113679


namespace NUMINAMATH_CALUDE_min_value_theorem_l1136_113699

theorem min_value_theorem (x : ℝ) (h : x > 1) : x + 4 / (x - 1) ≥ 5 ∧ ∃ y > 1, y + 4 / (y - 1) = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1136_113699


namespace NUMINAMATH_CALUDE_complex_equation_implies_sum_l1136_113610

theorem complex_equation_implies_sum (a t : ℝ) :
  (a + Complex.I) / (1 + 2 * Complex.I) = t * Complex.I →
  t + a = -1 := by sorry

end NUMINAMATH_CALUDE_complex_equation_implies_sum_l1136_113610


namespace NUMINAMATH_CALUDE_exponential_properties_l1136_113635

theorem exponential_properties (a : ℝ) (x y : ℝ) 
  (hx : a^x = 2) (hy : a^y = 3) : 
  a^(x + y) = 6 ∧ a^(2*x - 3*y) = 4/27 := by
  sorry

end NUMINAMATH_CALUDE_exponential_properties_l1136_113635


namespace NUMINAMATH_CALUDE_fruit_basket_count_l1136_113626

/-- The number of ways to choose items from a set of n items -/
def choiceCount (n : ℕ) : ℕ := n + 1

/-- The total number of fruit baskets including the empty basket -/
def totalBaskets (appleCount orangeCount : ℕ) : ℕ :=
  choiceCount appleCount * choiceCount orangeCount

/-- The number of valid fruit baskets (excluding the empty basket) -/
def validBaskets (appleCount orangeCount : ℕ) : ℕ :=
  totalBaskets appleCount orangeCount - 1

theorem fruit_basket_count :
  validBaskets 7 12 = 103 := by
  sorry

end NUMINAMATH_CALUDE_fruit_basket_count_l1136_113626


namespace NUMINAMATH_CALUDE_mud_weight_after_evaporation_l1136_113633

/-- 
Given a train car with mud, prove that the final weight after water evaporation
is 4000 pounds, given the initial conditions and final water percentage.
-/
theorem mud_weight_after_evaporation 
  (initial_weight : ℝ) 
  (initial_water_percent : ℝ)
  (final_water_percent : ℝ)
  (hw : initial_weight = 6000)
  (hiw : initial_water_percent = 88)
  (hfw : final_water_percent = 82) :
  (initial_weight * (100 - initial_water_percent) / 100) / ((100 - final_water_percent) / 100) = 4000 :=
by sorry

end NUMINAMATH_CALUDE_mud_weight_after_evaporation_l1136_113633


namespace NUMINAMATH_CALUDE_dave_candy_bars_l1136_113648

/-- Proves that Dave paid for 6 candy bars given the problem conditions -/
theorem dave_candy_bars (total_bars : ℕ) (cost_per_bar : ℚ) (john_paid : ℚ) : 
  total_bars = 20 →
  cost_per_bar = 3/2 →
  john_paid = 21 →
  (total_bars : ℚ) * cost_per_bar - john_paid = 6 * cost_per_bar :=
by sorry

end NUMINAMATH_CALUDE_dave_candy_bars_l1136_113648


namespace NUMINAMATH_CALUDE_escalator_travel_time_l1136_113622

/-- Proves that a person walking on a moving escalator takes 10 seconds to cover its length -/
theorem escalator_travel_time (escalator_speed : ℝ) (person_speed : ℝ) (escalator_length : ℝ) 
  (h1 : escalator_speed = 20)
  (h2 : person_speed = 5)
  (h3 : escalator_length = 250) :
  escalator_length / (escalator_speed + person_speed) = 10 := by
  sorry

end NUMINAMATH_CALUDE_escalator_travel_time_l1136_113622


namespace NUMINAMATH_CALUDE_exists_fib_divisible_by_2014_l1136_113696

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Theorem: There exists a positive integer n such that F_n is divisible by 2014 -/
theorem exists_fib_divisible_by_2014 : ∃ n : ℕ, n > 0 ∧ 2014 ∣ fib n := by
  sorry

end NUMINAMATH_CALUDE_exists_fib_divisible_by_2014_l1136_113696


namespace NUMINAMATH_CALUDE_bill_with_late_charges_l1136_113684

/-- Calculates the final amount owed after applying three consecutive 2% increases to an original bill. -/
def final_amount (original_bill : ℝ) : ℝ :=
  original_bill * (1 + 0.02)^3

/-- Theorem stating that given an original bill of $500 and three consecutive 2% increases, 
    the final amount owed is $530.604 (rounded to 3 decimal places) -/
theorem bill_with_late_charges :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0005 ∧ |final_amount 500 - 530.604| < ε :=
sorry

end NUMINAMATH_CALUDE_bill_with_late_charges_l1136_113684


namespace NUMINAMATH_CALUDE_first_group_size_l1136_113662

/-- The amount of work done by one person in one day -/
def work_unit : ℝ := 1

/-- The number of days -/
def days : ℕ := 3

/-- The number of people in the second group -/
def people_second_group : ℕ := 8

/-- The amount of work done by the first group -/
def work_first_group : ℝ := 3

/-- The amount of work done by the second group -/
def work_second_group : ℝ := 8

/-- The number of people in the first group -/
def people_first_group : ℕ := 3

theorem first_group_size :
  (people_first_group : ℝ) * days * work_unit = work_first_group ∧
  (people_second_group : ℝ) * days * work_unit = work_second_group →
  people_first_group = 3 := by
  sorry

end NUMINAMATH_CALUDE_first_group_size_l1136_113662


namespace NUMINAMATH_CALUDE_amount_distribution_l1136_113609

/-- The total amount distributed among boys -/
def total_amount : ℕ := 5040

/-- The number of boys in the first distribution -/
def boys_first : ℕ := 14

/-- The number of boys in the second distribution -/
def boys_second : ℕ := 18

/-- The difference in amount received by each boy between the two distributions -/
def difference : ℕ := 80

theorem amount_distribution :
  total_amount / boys_first = total_amount / boys_second + difference :=
sorry

end NUMINAMATH_CALUDE_amount_distribution_l1136_113609


namespace NUMINAMATH_CALUDE_cube_lines_properties_l1136_113621

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  a : ℝ
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A₁ : Point3D
  B₁ : Point3D
  C₁ : Point3D
  D₁ : Point3D

/-- Defines a cube with given edge length and correct vertex positions -/
def makeCube (a : ℝ) : Cube := {
  a := a,
  A := ⟨0, 0, 0⟩,
  B := ⟨a, 0, 0⟩,
  C := ⟨a, a, 0⟩,
  D := ⟨0, a, 0⟩,
  A₁ := ⟨0, 0, a⟩,
  B₁ := ⟨a, 0, a⟩,
  C₁ := ⟨a, a, a⟩,
  D₁ := ⟨0, a, a⟩
}

/-- Calculates the angle between two lines in the cube -/
def angleBetweenLines (cube : Cube) : ℝ := sorry

/-- Calculates the distance between two lines in the cube -/
def distanceBetweenLines (cube : Cube) : ℝ := sorry

theorem cube_lines_properties (a : ℝ) (h : a > 0) :
  let cube := makeCube a
  angleBetweenLines cube = 90 ∧ 
  distanceBetweenLines cube = a * Real.sqrt 6 / 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_lines_properties_l1136_113621


namespace NUMINAMATH_CALUDE_parallel_lines_k_values_l1136_113616

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The first line l₁: (k-3)x+(4-k)y+1=0 -/
def l1 (k : ℝ) : Line :=
  { a := k - 3, b := 4 - k, c := 1 }

/-- The second line l₂: 2(k-3)-2y+3=0, rewritten as 2(k-3)x-2y+3=0 -/
def l2 (k : ℝ) : Line :=
  { a := 2 * (k - 3), b := -2, c := 3 }

/-- Theorem stating that if l₁ and l₂ are parallel, then k is either 3 or 5 -/
theorem parallel_lines_k_values :
  ∀ k, parallel (l1 k) (l2 k) → k = 3 ∨ k = 5 := by
  sorry

#check parallel_lines_k_values

end NUMINAMATH_CALUDE_parallel_lines_k_values_l1136_113616


namespace NUMINAMATH_CALUDE_problem_statement_l1136_113639

theorem problem_statement (x y : ℝ) (hx : x = Real.sqrt 2 + 1) (hy : y = Real.sqrt 2 - 1) :
  (x + y) * (x - y) = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1136_113639


namespace NUMINAMATH_CALUDE_system_solution_l1136_113689

theorem system_solution (k j : ℝ) (h1 : 64 / k = 8) (h2 : k * j = 128) : k = 8 ∧ j = 16 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1136_113689


namespace NUMINAMATH_CALUDE_next_two_numbers_l1136_113682

def arithmetic_sequence (n : ℕ) : ℕ := n + 1

theorem next_two_numbers (n : ℕ) (h : n ≥ 6) :
  arithmetic_sequence n = n + 1 ∧
  arithmetic_sequence (n + 1) = n + 2 :=
by sorry

end NUMINAMATH_CALUDE_next_two_numbers_l1136_113682


namespace NUMINAMATH_CALUDE_first_prize_winners_l1136_113697

theorem first_prize_winners (n : ℕ) : 
  (30 ≤ n ∧ n ≤ 55) ∧ 
  (n % 3 = 2) ∧ 
  (n % 5 = 4) ∧ 
  (n % 7 = 2) → 
  n = 44 := by
sorry

end NUMINAMATH_CALUDE_first_prize_winners_l1136_113697


namespace NUMINAMATH_CALUDE_cubic_fifth_power_roots_l1136_113692

/-- The roots of a cubic polynomial x^3 + ax^2 + bx + c = 0 are the fifth powers of the roots of x^3 - 3x + 1 = 0 if and only if a = 15, b = -198, and c = 1 -/
theorem cubic_fifth_power_roots (a b c : ℝ) : 
  (∀ x : ℂ, x^3 - 3*x + 1 = 0 → ∃ y : ℂ, y^3 + a*y^2 + b*y + c = 0 ∧ y = x^5) ↔ 
  (a = 15 ∧ b = -198 ∧ c = 1) :=
by sorry

end NUMINAMATH_CALUDE_cubic_fifth_power_roots_l1136_113692


namespace NUMINAMATH_CALUDE_trailing_zeros_25_factorial_l1136_113604

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- Theorem: The number of trailing zeros in 25! is 6 -/
theorem trailing_zeros_25_factorial :
  trailingZeros 25 = 6 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_25_factorial_l1136_113604


namespace NUMINAMATH_CALUDE_second_month_sale_l1136_113665

theorem second_month_sale 
  (first_month : ℕ) 
  (third_month : ℕ) 
  (fourth_month : ℕ) 
  (fifth_month : ℕ) 
  (sixth_month : ℕ) 
  (average_sale : ℕ) 
  (h1 : first_month = 5435)
  (h2 : third_month = 5855)
  (h3 : fourth_month = 6230)
  (h4 : fifth_month = 5562)
  (h5 : sixth_month = 3991)
  (h6 : average_sale = 5500) :
  ∃ (second_month : ℕ), 
    (first_month + second_month + third_month + fourth_month + fifth_month + sixth_month) / 6 = average_sale ∧ 
    second_month = 5927 :=
by sorry

end NUMINAMATH_CALUDE_second_month_sale_l1136_113665


namespace NUMINAMATH_CALUDE_existence_of_small_difference_l1136_113658

theorem existence_of_small_difference (a : Fin 101 → ℝ)
  (h_increasing : ∀ i j, i < j → a i < a j)
  (h_bound : a 100 - a 0 ≤ 1000) :
  ∃ i j, i < j ∧ 0 < a j - a i ∧ a j - a i ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_existence_of_small_difference_l1136_113658


namespace NUMINAMATH_CALUDE_problem_statements_l1136_113601

theorem problem_statements :
  (∀ x : ℝ, (Real.sqrt (x + 1) * (2 * x - 1) ≥ 0) ↔ (x ≥ 1/2)) ∧
  (∀ x y : ℝ, (x > 1 ∧ y > 2) → (x + y > 3)) ∧
  (∃ x y : ℝ, (x + y > 3) ∧ ¬(x > 1 ∧ y > 2)) ∧
  (∀ x : ℝ, Real.sqrt (x^2 + 2) + 1 / Real.sqrt (x^2 + 2) > 2) ∧
  (¬(∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0)) :=
by sorry

end NUMINAMATH_CALUDE_problem_statements_l1136_113601


namespace NUMINAMATH_CALUDE_floor_negative_seven_fourths_l1136_113624

theorem floor_negative_seven_fourths : ⌊(-7 : ℤ) / 4⌋ = -2 := by
  sorry

end NUMINAMATH_CALUDE_floor_negative_seven_fourths_l1136_113624


namespace NUMINAMATH_CALUDE_median_mean_difference_l1136_113619

theorem median_mean_difference (x : ℤ) (a : ℤ) : 
  x > 0 → x + a > 0 → x + 4 > 0 → x + 7 > 0 → x + 37 > 0 →
  (x + (x + a) + (x + 4) + (x + 7) + (x + 37)) / 5 = (x + 4) + 6 →
  a = 2 := by sorry

end NUMINAMATH_CALUDE_median_mean_difference_l1136_113619


namespace NUMINAMATH_CALUDE_lemniscate_polar_to_rect_l1136_113671

/-- The lemniscate equation in polar coordinates -/
def lemniscate_polar (r φ a : ℝ) : Prop :=
  r^2 = 2 * a^2 * Real.cos (2 * φ)

/-- The lemniscate equation in rectangular coordinates -/
def lemniscate_rect (x y a : ℝ) : Prop :=
  (x^2 + y^2)^2 = 2 * a^2 * (x^2 - y^2)

/-- Theorem stating that the rectangular equation represents the lemniscate -/
theorem lemniscate_polar_to_rect (a : ℝ) :
  ∀ (x y r φ : ℝ), 
    x = r * Real.cos φ →
    y = r * Real.sin φ →
    lemniscate_polar r φ a →
    lemniscate_rect x y a :=
by
  sorry

end NUMINAMATH_CALUDE_lemniscate_polar_to_rect_l1136_113671


namespace NUMINAMATH_CALUDE_binomial_coefficient_divisibility_l1136_113678

theorem binomial_coefficient_divisibility (p n : ℕ) (hp : p.Prime) (hn : n ≥ p) :
  ∃ k : ℤ, (Nat.choose n p : ℤ) - (n / p : ℤ) = k * p := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_divisibility_l1136_113678


namespace NUMINAMATH_CALUDE_simplify_fraction_sum_l1136_113625

theorem simplify_fraction_sum (n d : Nat) : 
  n = 75 → d = 100 → ∃ (a b : Nat), (a.gcd b = 1) ∧ (n * b = d * a) ∧ (a + b = 7) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_sum_l1136_113625


namespace NUMINAMATH_CALUDE_sin_30_degrees_l1136_113687

/-- Sine of 30 degrees is equal to 1/2 -/
theorem sin_30_degrees : Real.sin (π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_30_degrees_l1136_113687


namespace NUMINAMATH_CALUDE_correct_mark_l1136_113612

theorem correct_mark (wrong_mark : ℝ) (class_size : ℕ) (average_increase : ℝ) : 
  wrong_mark = 85 → 
  class_size = 80 → 
  average_increase = 0.5 →
  (wrong_mark - (wrong_mark - class_size * average_increase)) = 45 := by
  sorry

end NUMINAMATH_CALUDE_correct_mark_l1136_113612


namespace NUMINAMATH_CALUDE_subtraction_puzzle_l1136_113613

theorem subtraction_puzzle (a b c : ℕ) (h1 : a ≤ 9) (h2 : b ≤ 9) (h3 : c ≤ 9)
  (h4 : (100 * a + 10 * b + c) - (100 * c + 10 * b + a) % 10 = 2)
  (h5 : b = c - 1)
  (h6 : (100 * a + 10 * b + c) - (100 * c + 10 * b + a) / 100 = 8) :
  a = 0 ∧ b = 1 ∧ c = 2 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_puzzle_l1136_113613


namespace NUMINAMATH_CALUDE_anne_found_five_bottle_caps_l1136_113676

/-- The number of bottle caps Anne found -/
def bottle_caps_found (initial final : ℕ) : ℕ := final - initial

/-- Proof that Anne found 5 bottle caps -/
theorem anne_found_five_bottle_caps :
  bottle_caps_found 10 15 = 5 := by
  sorry

end NUMINAMATH_CALUDE_anne_found_five_bottle_caps_l1136_113676


namespace NUMINAMATH_CALUDE_fraction_sum_integer_implies_fractions_integer_l1136_113686

theorem fraction_sum_integer_implies_fractions_integer (x y : ℕ) :
  ∃ k : ℤ, (x^2 - 1 : ℚ) / (y + 1) + (y^2 - 1 : ℚ) / (x + 1) = k →
  ∃ m n : ℤ, (x^2 - 1 : ℚ) / (y + 1) = m ∧ (y^2 - 1 : ℚ) / (x + 1) = n :=
by sorry

end NUMINAMATH_CALUDE_fraction_sum_integer_implies_fractions_integer_l1136_113686


namespace NUMINAMATH_CALUDE_order_of_rationals_l1136_113685

theorem order_of_rationals (a b : ℚ) (h : b < 0) : a + b < a ∧ a < a - b := by
  sorry

end NUMINAMATH_CALUDE_order_of_rationals_l1136_113685


namespace NUMINAMATH_CALUDE_right_triangle_with_53_hypotenuse_l1136_113644

theorem right_triangle_with_53_hypotenuse (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 53 →           -- Hypotenuse is 53
  b = a + 1 →        -- Legs are consecutive integers
  a + b = 75 :=      -- Sum of legs is 75
by sorry

end NUMINAMATH_CALUDE_right_triangle_with_53_hypotenuse_l1136_113644


namespace NUMINAMATH_CALUDE_lcm_14_21_35_l1136_113600

theorem lcm_14_21_35 : Nat.lcm 14 (Nat.lcm 21 35) = 210 := by
  sorry

end NUMINAMATH_CALUDE_lcm_14_21_35_l1136_113600


namespace NUMINAMATH_CALUDE_find_m_l1136_113695

theorem find_m : ∃ m : ℝ, 
  (∃ y : ℝ, 2 - 3 * (1 - y) = 2 * y) ∧ 
  (∃ x : ℝ, m * (x - 3) - 2 = -8) ∧ 
  (∀ y x : ℝ, 2 - 3 * (1 - y) = 2 * y ↔ m * (x - 3) - 2 = -8) → 
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_find_m_l1136_113695


namespace NUMINAMATH_CALUDE_certain_amount_proof_l1136_113645

theorem certain_amount_proof : 
  let x : ℝ := 900
  let A : ℝ := 0.15 * 1600 - 0.25 * x
  A = 15 := by sorry

end NUMINAMATH_CALUDE_certain_amount_proof_l1136_113645


namespace NUMINAMATH_CALUDE_color_drawing_cost_theorem_l1136_113698

/-- The cost of a color drawing given the cost of a black and white drawing and the additional percentage for color. -/
def color_drawing_cost (bw_cost : ℝ) (color_percentage : ℝ) : ℝ :=
  bw_cost * (1 + color_percentage)

/-- Theorem: The cost of a color drawing is $240 when a black and white drawing costs $160 and color is 50% more expensive. -/
theorem color_drawing_cost_theorem :
  color_drawing_cost 160 0.5 = 240 := by
  sorry

end NUMINAMATH_CALUDE_color_drawing_cost_theorem_l1136_113698


namespace NUMINAMATH_CALUDE_amusement_park_tickets_l1136_113646

theorem amusement_park_tickets (total_cost : ℕ) (adult_price child_price : ℕ) (adult_child_diff : ℕ) : 
  total_cost = 720 →
  adult_price = 15 →
  child_price = 8 →
  adult_child_diff = 25 →
  ∃ (num_children : ℕ), 
    num_children * child_price + (num_children + adult_child_diff) * adult_price = total_cost ∧ 
    num_children = 15 :=
by sorry

end NUMINAMATH_CALUDE_amusement_park_tickets_l1136_113646


namespace NUMINAMATH_CALUDE_max_sum_of_products_l1136_113632

/-- Represents the assignment of numbers to cube faces -/
def CubeAssignment := Fin 6 → Fin 6

/-- Computes the sum of products at cube vertices given a face assignment -/
def sumOfProducts (assignment : CubeAssignment) : ℕ :=
  sorry

/-- The set of all possible cube assignments -/
def allAssignments : Set CubeAssignment :=
  sorry

theorem max_sum_of_products :
  ∃ (assignment : CubeAssignment),
    assignment ∈ allAssignments ∧
    sumOfProducts assignment = 343 ∧
    ∀ (other : CubeAssignment),
      other ∈ allAssignments →
      sumOfProducts other ≤ 343 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_products_l1136_113632


namespace NUMINAMATH_CALUDE_james_tv_watching_time_l1136_113661

/-- The duration of a Jeopardy episode in minutes -/
def jeopardy_duration : ℕ := 20

/-- The number of Jeopardy episodes watched -/
def jeopardy_episodes : ℕ := 2

/-- The duration of a Wheel of Fortune episode in minutes -/
def wheel_of_fortune_duration : ℕ := 2 * jeopardy_duration

/-- The number of Wheel of Fortune episodes watched -/
def wheel_of_fortune_episodes : ℕ := 2

/-- The total time spent watching TV in minutes -/
def total_time_minutes : ℕ := 
  jeopardy_duration * jeopardy_episodes + 
  wheel_of_fortune_duration * wheel_of_fortune_episodes

/-- Conversion factor from minutes to hours -/
def minutes_per_hour : ℕ := 60

theorem james_tv_watching_time : 
  total_time_minutes / minutes_per_hour = 2 := by sorry

end NUMINAMATH_CALUDE_james_tv_watching_time_l1136_113661
