import Mathlib

namespace distance_from_rate_and_time_l535_53523

/-- Proves that given a constant walking rate and time, the distance covered is equal to the product of rate and time. -/
theorem distance_from_rate_and_time 
  (rate : ℝ) 
  (time : ℝ) 
  (h_rate : rate = 4) 
  (h_time : time = 2) : 
  rate * time = 8 := by
  sorry

#check distance_from_rate_and_time

end distance_from_rate_and_time_l535_53523


namespace sin_n_equals_cos_678_l535_53528

theorem sin_n_equals_cos_678 (n : ℤ) (h1 : -120 ≤ n) (h2 : n ≤ 120) :
  Real.sin (n * π / 180) = Real.cos (678 * π / 180) → n = 48 := by
  sorry

end sin_n_equals_cos_678_l535_53528


namespace denise_age_l535_53567

theorem denise_age (amanda beth carlos denise : ℕ) 
  (h1 : amanda = carlos - 4)
  (h2 : carlos = beth + 5)
  (h3 : denise = beth + 2)
  (h4 : amanda = 16) : 
  denise = 17 := by
  sorry

end denise_age_l535_53567


namespace correct_num_bedrooms_l535_53516

/-- The number of bedrooms to clean -/
def num_bedrooms : ℕ := sorry

/-- Time in minutes to clean one bedroom -/
def bedroom_time : ℕ := 20

/-- Time in minutes to clean the living room -/
def living_room_time : ℕ := num_bedrooms * bedroom_time

/-- Time in minutes to clean one bathroom -/
def bathroom_time : ℕ := 2 * living_room_time

/-- Time in minutes to clean the house (bedrooms, living room, and bathrooms) -/
def house_time : ℕ := num_bedrooms * bedroom_time + living_room_time + 2 * bathroom_time

/-- Time in minutes to clean outside -/
def outside_time : ℕ := 2 * house_time

/-- Total time in minutes for all three siblings to work -/
def total_work_time : ℕ := 3 * 4 * 60

theorem correct_num_bedrooms : num_bedrooms = 3 := by sorry

end correct_num_bedrooms_l535_53516


namespace f_minus_three_equals_six_l535_53556

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem f_minus_three_equals_six 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_period : has_period f 4) 
  (h_sum : f 1 + f 2 + f 3 + f 4 + f 5 = 6) : 
  f (-3) = 6 := by
sorry

end f_minus_three_equals_six_l535_53556


namespace arithmetic_sequence_sum_l535_53561

def arithmetic_sequence : List ℕ := [71, 75, 79, 83, 87, 91]

theorem arithmetic_sequence_sum : 
  3 * (arithmetic_sequence.sum) = 1458 := by
  sorry

end arithmetic_sequence_sum_l535_53561


namespace sum_geq_five_x_squared_l535_53593

theorem sum_geq_five_x_squared (x : ℝ) (hx : x > 0) :
  1 + x + x^2 + x^3 + x^4 ≥ 5 * x^2 := by
  sorry

end sum_geq_five_x_squared_l535_53593


namespace music_student_count_l535_53562

/-- Represents the number of students in different categories -/
structure StudentCounts where
  total : ℕ
  art : ℕ
  both : ℕ
  neither : ℕ

/-- Calculates the number of students taking music -/
def musicStudents (counts : StudentCounts) : ℕ :=
  counts.total - counts.neither - (counts.art - counts.both)

/-- Theorem stating the number of students taking music -/
theorem music_student_count (counts : StudentCounts)
    (h_total : counts.total = 500)
    (h_art : counts.art = 10)
    (h_both : counts.both = 10)
    (h_neither : counts.neither = 470) :
    musicStudents counts = 30 := by
  sorry

#eval musicStudents { total := 500, art := 10, both := 10, neither := 470 }

end music_student_count_l535_53562


namespace a_12_upper_bound_a_12_no_lower_bound_l535_53525

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

/-- The upper bound of a_12 in an arithmetic sequence satisfying given conditions -/
theorem a_12_upper_bound
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a8 : a 8 ≥ 15)
  (h_a9 : a 9 ≤ 13) :
  a 12 ≤ 7 :=
sorry

/-- The non-existence of a lower bound for a_12 in an arithmetic sequence satisfying given conditions -/
theorem a_12_no_lower_bound
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a8 : a 8 ≥ 15)
  (h_a9 : a 9 ≤ 13) :
  ∀ x : ℝ, ∃ y : ℝ, y < x ∧ ∃ (a' : ℕ → ℝ), arithmetic_sequence a' ∧ a' 8 ≥ 15 ∧ a' 9 ≤ 13 ∧ a' 12 = y :=
sorry

end a_12_upper_bound_a_12_no_lower_bound_l535_53525


namespace ratio_problem_l535_53584

theorem ratio_problem (second_part : ℝ) (ratio_percent : ℝ) : 
  second_part = 4 → ratio_percent = 125 → (ratio_percent / 100) * second_part = 5 := by
  sorry

end ratio_problem_l535_53584


namespace tangent_line_circle_range_l535_53514

theorem tangent_line_circle_range (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_tangent : ∀ x y : ℝ, x + y + a = 0 → (x - b)^2 + (y - 1)^2 ≥ 2) 
  (h_touch : ∃ x y : ℝ, x + y + a = 0 ∧ (x - b)^2 + (y - 1)^2 = 2) :
  ∀ z : ℝ, z > 0 → ∃ c : ℝ, c > 0 ∧ c < 1 ∧ z = c^2 / (1 - c) :=
sorry

end tangent_line_circle_range_l535_53514


namespace percentage_problem_l535_53535

theorem percentage_problem (N : ℝ) (P : ℝ) : 
  N = 3200 →
  0.1 * N = (P / 100) * 650 + 190 →
  P = 20 :=
by sorry

end percentage_problem_l535_53535


namespace r_fourth_plus_reciprocal_l535_53565

theorem r_fourth_plus_reciprocal (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 := by
  sorry

end r_fourth_plus_reciprocal_l535_53565


namespace rationalize_denominator_l535_53569

theorem rationalize_denominator : (35 : ℝ) / Real.sqrt 35 = Real.sqrt 35 := by
  sorry

end rationalize_denominator_l535_53569


namespace fraction_equality_l535_53573

theorem fraction_equality (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : 0 / b = b / c) (h2 : b / c = 1 / a) :
  (a + b - c) / (a - b + c) = 1 := by sorry

end fraction_equality_l535_53573


namespace points_collinear_l535_53541

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a triangular pyramid -/
structure TriangularPyramid where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Check if four points are collinear -/
def are_collinear (p1 p2 p3 p4 : Point3D) : Prop :=
  ∃ (t1 t2 t3 : ℝ), p2 = Point3D.mk (p1.x + t1 * (p4.x - p1.x)) (p1.y + t1 * (p4.y - p1.y)) (p1.z + t1 * (p4.z - p1.z)) ∧
                     p3 = Point3D.mk (p1.x + t2 * (p4.x - p1.x)) (p1.y + t2 * (p4.y - p1.y)) (p1.z + t2 * (p4.z - p1.z)) ∧
                     p4 = Point3D.mk (p1.x + t3 * (p4.x - p1.x)) (p1.y + t3 * (p4.y - p1.y)) (p1.z + t3 * (p4.z - p1.z))

/-- Main theorem -/
theorem points_collinear (pyramid : TriangularPyramid) 
  (M K P H E F Q T : Point3D)
  (h1 : (pyramid.A.x - M.x)^2 + (pyramid.A.y - M.y)^2 + (pyramid.A.z - M.z)^2 = 
        (M.x - K.x)^2 + (M.y - K.y)^2 + (M.z - K.z)^2)
  (h2 : (M.x - K.x)^2 + (M.y - K.y)^2 + (M.z - K.z)^2 = 
        (K.x - pyramid.D.x)^2 + (K.y - pyramid.D.y)^2 + (K.z - pyramid.D.z)^2)
  (h3 : (pyramid.B.x - P.x)^2 + (pyramid.B.y - P.y)^2 + (pyramid.B.z - P.z)^2 = 
        (P.x - H.x)^2 + (P.y - H.y)^2 + (P.z - H.z)^2)
  (h4 : (P.x - H.x)^2 + (P.y - H.y)^2 + (P.z - H.z)^2 = 
        (H.x - pyramid.C.x)^2 + (H.y - pyramid.C.y)^2 + (H.z - pyramid.C.z)^2)
  (h5 : (pyramid.A.x - E.x)^2 + (pyramid.A.y - E.y)^2 + (pyramid.A.z - E.z)^2 = 
        0.25 * ((pyramid.A.x - pyramid.B.x)^2 + (pyramid.A.y - pyramid.B.y)^2 + (pyramid.A.z - pyramid.B.z)^2))
  (h6 : (M.x - F.x)^2 + (M.y - F.y)^2 + (M.z - F.z)^2 = 
        0.25 * ((M.x - P.x)^2 + (M.y - P.y)^2 + (M.z - P.z)^2))
  (h7 : (K.x - Q.x)^2 + (K.y - Q.y)^2 + (K.z - Q.z)^2 = 
        0.25 * ((K.x - H.x)^2 + (K.y - H.y)^2 + (K.z - H.z)^2))
  (h8 : (pyramid.D.x - T.x)^2 + (pyramid.D.y - T.y)^2 + (pyramid.D.z - T.z)^2 = 
        0.25 * ((pyramid.D.x - pyramid.C.x)^2 + (pyramid.D.y - pyramid.C.y)^2 + (pyramid.D.z - pyramid.C.z)^2))
  : are_collinear E F Q T :=
sorry

end points_collinear_l535_53541


namespace geometry_propositions_l535_53566

-- Define the propositions
variable (p₁ p₂ p₃ p₄ : Prop)

-- Define the truth values of the propositions
axiom h₁ : p₁
axiom h₂ : ¬p₂
axiom h₃ : ¬p₃
axiom h₄ : p₄

-- Theorem to prove
theorem geometry_propositions :
  (p₁ ∧ p₄) ∧ (¬p₂ ∨ p₃) ∧ (¬p₃ ∨ ¬p₄) := by
  sorry

end geometry_propositions_l535_53566


namespace fourth_root_over_sixth_root_of_seven_l535_53542

theorem fourth_root_over_sixth_root_of_seven (x : ℝ) :
  (7 ^ (1/4)) / (7 ^ (1/6)) = 7 ^ (1/12) :=
by sorry

end fourth_root_over_sixth_root_of_seven_l535_53542


namespace R_value_at_7_l535_53512

/-- The function that defines R in terms of S and h -/
def R (S h : ℝ) : ℝ := h * S + 2 * S - 6

/-- The theorem stating that if R = 28 when S = 5, then R = 41 when S = 7 -/
theorem R_value_at_7 (h : ℝ) (h_condition : R 5 h = 28) : R 7 h = 41 := by
  sorry

end R_value_at_7_l535_53512


namespace fraction_equality_l535_53529

theorem fraction_equality (x y : ℚ) (a b : ℤ) (h1 : y = 40) (h2 : x + 35 = 4 * y) (h3 : 1/5 * x = a/b * y) (h4 : b ≠ 0) : a/b = 5/8 := by
  sorry

end fraction_equality_l535_53529


namespace xyz_equals_ten_l535_53580

theorem xyz_equals_ten (a b c x y z : ℂ) 
  (nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (eq1 : a = (b + c) / (x - 3))
  (eq2 : b = (a + c) / (y - 3))
  (eq3 : c = (a + b) / (z - 3))
  (sum_prod : x*y + x*z + y*z = 10)
  (sum : x + y + z = 6) :
  x * y * z = 10 := by
sorry

end xyz_equals_ten_l535_53580


namespace base_sum_equals_55_base_7_l535_53507

/-- Represents a number in a given base --/
def BaseNumber (base : ℕ) := ℕ

/-- Converts a base number to its decimal representation --/
def to_decimal (base : ℕ) (n : BaseNumber base) : ℕ := sorry

/-- Converts a decimal number to its representation in a given base --/
def from_decimal (base : ℕ) (n : ℕ) : BaseNumber base := sorry

/-- Multiplies two numbers in a given base --/
def base_mul (base : ℕ) (a b : BaseNumber base) : BaseNumber base := sorry

/-- Adds two numbers in a given base --/
def base_add (base : ℕ) (a b : BaseNumber base) : BaseNumber base := sorry

theorem base_sum_equals_55_base_7 (c : ℕ) 
  (h : base_mul c (base_mul c (from_decimal c 14) (from_decimal c 18)) (from_decimal c 17) = from_decimal c 4185) :
  base_add c (base_add c (from_decimal c 14) (from_decimal c 18)) (from_decimal c 17) = from_decimal 7 55 := 
sorry

end base_sum_equals_55_base_7_l535_53507


namespace fish_value_in_rice_fish_value_in_rice_mixed_l535_53579

-- Define the trading rates
def fish_to_bread_rate : ℚ := 3 / 5
def bread_to_rice_rate : ℕ := 7

-- Theorem statement
theorem fish_value_in_rice : 
  fish_to_bread_rate * bread_to_rice_rate = 21 / 5 := by
  sorry

-- Converting the result to a mixed number
theorem fish_value_in_rice_mixed : 
  ∃ (whole : ℕ) (frac : ℚ), 
    fish_to_bread_rate * bread_to_rice_rate = whole + frac ∧ 
    whole = 4 ∧ 
    frac = 1 / 5 := by
  sorry

end fish_value_in_rice_fish_value_in_rice_mixed_l535_53579


namespace solution_implies_k_value_l535_53517

theorem solution_implies_k_value (x k : ℚ) : 
  (x = 5 ∧ 4 * x + 2 * k = 8) → k = -6 := by
  sorry

end solution_implies_k_value_l535_53517


namespace orthocenter_of_triangle_l535_53544

/-- The orthocenter of a triangle in 3D space. -/
def orthocenter (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- The vertices of the triangle -/
def A : ℝ × ℝ × ℝ := (2, 3, 4)
def B : ℝ × ℝ × ℝ := (6, 4, 2)
def C : ℝ × ℝ × ℝ := (4, 6, 6)

/-- Theorem: The orthocenter of triangle ABC is (10/7, 51/7, 12/7) -/
theorem orthocenter_of_triangle :
  orthocenter A B C = (10/7, 51/7, 12/7) := by sorry

end orthocenter_of_triangle_l535_53544


namespace next_challenge_digits_estimate_l535_53582

/-- The number of decimal digits in RSA-640 -/
def rsa640_digits : ℕ := 193

/-- The prize amount for RSA-640 in dollars -/
def rsa640_prize : ℕ := 20000

/-- The prize amount for the next challenge in dollars -/
def next_challenge_prize : ℕ := 30000

/-- A reasonable upper bound for the number of digits in the next challenge -/
def reasonable_upper_bound : ℕ := 220

/-- Theorem stating that a reasonable estimate for the number of digits
    in the next challenge is greater than RSA-640's digits and at most 220 -/
theorem next_challenge_digits_estimate :
  ∃ (N : ℕ), N > rsa640_digits ∧ N ≤ reasonable_upper_bound ∧
  next_challenge_prize > rsa640_prize :=
sorry

end next_challenge_digits_estimate_l535_53582


namespace cos_135_degrees_l535_53515

theorem cos_135_degrees : Real.cos (135 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end cos_135_degrees_l535_53515


namespace quadratic_roots_product_l535_53537

theorem quadratic_roots_product (p q : ℝ) : 
  (3 * p^2 + 9 * p - 21 = 0) →
  (3 * q^2 + 9 * q - 21 = 0) →
  (3*p - 4) * (6*q - 8) = 122 := by
  sorry

end quadratic_roots_product_l535_53537


namespace complex_equality_proof_l535_53549

theorem complex_equality_proof (a : ℂ) : (1 + a * Complex.I) / (2 + Complex.I) = 1 + 2 * Complex.I → a = 5 + Complex.I := by
  sorry

end complex_equality_proof_l535_53549


namespace polynomial_properties_l535_53505

/-- A polynomial of the form f(x) = ax^5 + bx^3 + 4x + c -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^5 + b * x^3 + 4 * x + c

theorem polynomial_properties :
  ∀ (a b c : ℝ),
    (f a b c 0 = 6 → c = 6) ∧
    (f a b c 0 = -2 ∧ f a b c 1 = 5 → f a b c (-1) = -9) ∧
    (f a b c 5 + f a b c (-5) = 6 ∧ f a b c 2 = 8 → f a b c (-2) = -2) :=
by sorry

end polynomial_properties_l535_53505


namespace ellen_hits_nine_l535_53557

-- Define the set of possible scores
def ScoreSet : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 15}

-- Define a type for the players
inductive Player : Type
| Alice | Ben | Cindy | Dave | Ellen | Frank

-- Define a function that returns the total score for each player
def playerScore (p : Player) : ℕ :=
  match p with
  | Player.Alice => 27
  | Player.Ben => 14
  | Player.Cindy => 20
  | Player.Dave => 22
  | Player.Ellen => 24
  | Player.Frank => 30

-- Define a predicate that checks if a list of scores is valid for a player
def validScores (scores : List ℕ) (p : Player) : Prop :=
  scores.length = 3 ∧
  scores.toFinset.card = 3 ∧
  (∀ s ∈ scores, s ∈ ScoreSet) ∧
  scores.sum = playerScore p

theorem ellen_hits_nine :
  ∃ (scores : List ℕ), validScores scores Player.Ellen ∧ 9 ∈ scores ∧
  (∀ (p : Player), p ≠ Player.Ellen → ∀ (s : List ℕ), validScores s p → 9 ∉ s) :=
sorry

end ellen_hits_nine_l535_53557


namespace test_scores_order_l535_53555

-- Define the scores as natural numbers
variable (J E N L : ℕ)

-- Define the theorem
theorem test_scores_order :
  -- Conditions
  (E = J) →  -- Elina's score is the same as Jasper's
  (N ≤ J) →  -- Norah's score is not higher than Jasper's
  (L > J) →  -- Liam's score is higher than Jasper's
  -- Conclusion: The order of scores from lowest to highest is N, E, L
  (N ≤ E ∧ E < L) := by
sorry

end test_scores_order_l535_53555


namespace equation_has_one_solution_l535_53526

/-- The equation (3x^3 - 15x^2) / (x^2 - 5x) = 2x - 6 has exactly one solution -/
theorem equation_has_one_solution :
  ∃! x : ℝ, x ≠ 0 ∧ x ≠ 5 ∧ (3 * x^3 - 15 * x^2) / (x^2 - 5 * x) = 2 * x - 6 := by
  sorry

end equation_has_one_solution_l535_53526


namespace distance_for_specific_cube_l535_53591

/-- Represents a cube suspended above a plane -/
structure SuspendedCube where
  side_length : ℝ
  adjacent_heights : Fin 3 → ℝ

/-- The distance from the closest vertex to the plane for a suspended cube -/
def distance_to_plane (cube : SuspendedCube) : ℝ :=
  sorry

/-- Theorem stating the distance for the given cube configuration -/
theorem distance_for_specific_cube :
  let cube : SuspendedCube :=
    { side_length := 8
      adjacent_heights := ![8, 10, 9] }
  distance_to_plane cube = 5 := by
  sorry

end distance_for_specific_cube_l535_53591


namespace factorial_sum_remainder_l535_53585

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem factorial_sum_remainder (n : ℕ) (h : n ≥ 100) :
  sum_factorials n % 30 = sum_factorials 4 % 30 := by
  sorry

#eval sum_factorials 4 % 30  -- Should output 3

end factorial_sum_remainder_l535_53585


namespace complement_of_35_degree_angle_l535_53511

theorem complement_of_35_degree_angle (A : Real) : 
  A = 35 → 90 - A = 55 := by
  sorry

end complement_of_35_degree_angle_l535_53511


namespace arithmetic_sequence_sum_l535_53554

def isArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_sum (a : ℕ → ℚ) 
  (h_arith : isArithmeticSequence a) 
  (h_sum : a 3 + a 4 + a 6 + a 7 = 25) : 
  a 2 + a 8 = 25 / 2 := by
  sorry

end arithmetic_sequence_sum_l535_53554


namespace min_value_of_f_l535_53560

/-- The function f(x) = 2x³ - 6x² + m, where m is a constant -/
def f (x : ℝ) (m : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + m

/-- Theorem: Given f(x) = 2x³ - 6x² + m, where m is a constant,
    and f(x) reaches a maximum value of 2 within the interval [-2, 2],
    the minimum value of f(x) within [-2, 2] is -6. -/
theorem min_value_of_f (m : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f y m ≤ f x m) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = 2) →
  ∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = -6 ∧ ∀ y ∈ Set.Icc (-2 : ℝ) 2, -6 ≤ f y m :=
by sorry


end min_value_of_f_l535_53560


namespace original_cost_equals_new_cost_l535_53540

/-- Proves that the original manufacturing cost was equal to the new manufacturing cost
    when the profit percentage remains constant at 50% of the selling price. -/
theorem original_cost_equals_new_cost
  (selling_price : ℝ)
  (new_cost : ℝ)
  (h_profit_percentage : selling_price / 2 = selling_price - new_cost)
  (h_new_cost : new_cost = 50)
  : selling_price - (selling_price / 2) = new_cost :=
by sorry

end original_cost_equals_new_cost_l535_53540


namespace lastDigitOf2Power2023_l535_53501

-- Define the pattern of last digits for powers of 2
def lastDigitPattern : Fin 4 → Nat
  | 0 => 2
  | 1 => 4
  | 2 => 8
  | 3 => 6

-- Define the function to get the last digit of 2^n
def lastDigitOfPowerOf2 (n : Nat) : Nat :=
  lastDigitPattern ((n - 1) % 4)

-- Theorem statement
theorem lastDigitOf2Power2023 : lastDigitOfPowerOf2 2023 = 8 := by
  sorry

end lastDigitOf2Power2023_l535_53501


namespace sin_30_plus_cos_60_l535_53574

theorem sin_30_plus_cos_60 : Real.sin (30 * π / 180) + Real.cos (60 * π / 180) = 1 := by
  sorry

end sin_30_plus_cos_60_l535_53574


namespace reciprocal_of_2022_l535_53533

theorem reciprocal_of_2022 : (2022⁻¹ : ℚ) = 1 / 2022 := by
  sorry

end reciprocal_of_2022_l535_53533


namespace min_sum_squares_consecutive_integers_l535_53538

theorem min_sum_squares_consecutive_integers (y : ℤ) : 
  (∃ x : ℤ, y^2 = (x-5)^2 + (x-4)^2 + (x-3)^2 + (x-2)^2 + (x-1)^2 + x^2 + 
              (x+1)^2 + (x+2)^2 + (x+3)^2 + (x+4)^2 + (x+5)^2) →
  y^2 ≥ 121 :=
by sorry

end min_sum_squares_consecutive_integers_l535_53538


namespace distinct_digit_sums_count_l535_53524

/-- Calculate the digit sum of a natural number. -/
def digitSum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digitSum (n / 10)

/-- The set of all digit sums for numbers from 1 to 2021. -/
def digitSumSet : Finset ℕ :=
  Finset.image digitSum (Finset.range 2021)

/-- Theorem: The number of distinct digit sums for integers from 1 to 2021 is 28. -/
theorem distinct_digit_sums_count : digitSumSet.card = 28 := by
  sorry

end distinct_digit_sums_count_l535_53524


namespace basketball_score_proof_l535_53513

theorem basketball_score_proof (total : ℕ) 
  (hA : total / 4 = total / 4)  -- Player A scored 1/4 of total
  (hB : (total * 2) / 7 = (total * 2) / 7)  -- Player B scored 2/7 of total
  (hC : 15 ≤ total)  -- Player C scored 15 points
  (hRemaining : ∀ i : Fin 7, (total - (total / 4 + (total * 2) / 7 + 15)) / 7 ≤ 2)  -- Remaining players scored no more than 2 points each
  : total - (total / 4 + (total * 2) / 7 + 15) = 13 :=
by sorry

end basketball_score_proof_l535_53513


namespace tangent_line_min_slope_l535_53558

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 6*x^2 - x + 6

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 12*x - 1

-- Theorem statement
theorem tangent_line_min_slope :
  ∃ (x₀ y₀ : ℝ),
    f x₀ = y₀ ∧
    (∀ x : ℝ, f' x₀ ≤ f' x) ∧
    (13 * x₀ + y₀ - 14 = 0) :=
sorry

end tangent_line_min_slope_l535_53558


namespace final_position_l535_53547

def move_on_number_line (start : ℤ) (right : ℤ) (left : ℤ) : ℤ :=
  start + right - left

theorem final_position :
  move_on_number_line (-2) 3 5 = -4 := by
  sorry

end final_position_l535_53547


namespace berry_difference_l535_53568

theorem berry_difference (stacy_initial : ℕ) (steve_initial : ℕ) (taken : ℕ) : 
  stacy_initial = 32 → 
  steve_initial = 21 → 
  taken = 4 → 
  stacy_initial - (steve_initial + taken) = 7 := by
sorry

end berry_difference_l535_53568


namespace sin_two_alpha_value_l535_53510

theorem sin_two_alpha_value (α : Real) (h : Real.sin α - Real.cos α = 4/3) :
  Real.sin (2 * α) = -7/9 := by
  sorry

end sin_two_alpha_value_l535_53510


namespace monday_sales_is_five_l535_53571

/-- Represents the number of crates of eggs sold on each day of the week --/
structure EggSales where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ

/-- Defines the conditions for Gabrielle's egg sales --/
def validEggSales (sales : EggSales) : Prop :=
  sales.tuesday = 2 * sales.monday ∧
  sales.wednesday = sales.tuesday - 2 ∧
  sales.thursday = sales.tuesday / 2 ∧
  sales.monday + sales.tuesday + sales.wednesday + sales.thursday = 28

/-- Theorem stating that if the egg sales satisfy the given conditions,
    then the number of crates sold on Monday is 5 --/
theorem monday_sales_is_five (sales : EggSales) 
  (h : validEggSales sales) : sales.monday = 5 := by
  sorry

end monday_sales_is_five_l535_53571


namespace work_days_calculation_l535_53539

/-- Represents the number of days worked by each person -/
structure WorkDays where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents the daily wages of each person -/
structure DailyWages where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents the total earnings of all workers -/
def totalEarnings (days : WorkDays) (wages : DailyWages) : ℕ :=
  days.a * wages.a + days.b * wages.b + days.c * wages.c

/-- The main theorem stating the problem conditions and the result to be proved -/
theorem work_days_calculation (days : WorkDays) (wages : DailyWages) :
  days.a = 6 ∧
  days.c = 4 ∧
  wages.a * 4 = wages.b * 3 ∧
  wages.b * 5 = wages.c * 4 ∧
  wages.c = 125 ∧
  totalEarnings days wages = 1850 →
  days.b = 9 := by
  sorry


end work_days_calculation_l535_53539


namespace apartment_number_l535_53530

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

def swap_digits (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  ones * 100 + tens * 10 + hundreds

theorem apartment_number : 
  ∃! n : ℕ, is_three_digit n ∧ is_perfect_cube n ∧ Nat.Prime (swap_digits n) ∧ n = 125 := by
  sorry

end apartment_number_l535_53530


namespace plant_supplier_money_left_l535_53570

/-- Represents the plant supplier's business --/
structure PlantSupplier where
  orchids : ℕ
  orchidPrice : ℕ
  moneyPlants : ℕ
  moneyPlantPrice : ℕ
  bonsai : ℕ
  bonsaiPrice : ℕ
  cacti : ℕ
  cactiPrice : ℕ
  airPlants : ℕ
  airPlantPrice : ℕ
  fullTimeWorkers : ℕ
  fullTimeWage : ℕ
  partTimeWorkers : ℕ
  partTimeWage : ℕ
  ceramicPotsCost : ℕ
  plasticPotsCost : ℕ
  fertilizersCost : ℕ
  toolsCost : ℕ
  utilityBill : ℕ
  tax : ℕ

/-- Calculates the total earnings of the plant supplier --/
def totalEarnings (s : PlantSupplier) : ℕ :=
  s.orchids * s.orchidPrice +
  s.moneyPlants * s.moneyPlantPrice +
  s.bonsai * s.bonsaiPrice +
  s.cacti * s.cactiPrice +
  s.airPlants * s.airPlantPrice

/-- Calculates the total expenses of the plant supplier --/
def totalExpenses (s : PlantSupplier) : ℕ :=
  s.fullTimeWorkers * s.fullTimeWage +
  s.partTimeWorkers * s.partTimeWage +
  s.ceramicPotsCost +
  s.plasticPotsCost +
  s.fertilizersCost +
  s.toolsCost +
  s.utilityBill +
  s.tax

/-- Calculates the money left from the plant supplier's earnings --/
def moneyLeft (s : PlantSupplier) : ℕ :=
  totalEarnings s - totalExpenses s

/-- Theorem stating that the money left is $3755 given the specified conditions --/
theorem plant_supplier_money_left :
  ∃ (s : PlantSupplier),
    s.orchids = 35 ∧ s.orchidPrice = 52 ∧
    s.moneyPlants = 30 ∧ s.moneyPlantPrice = 32 ∧
    s.bonsai = 20 ∧ s.bonsaiPrice = 77 ∧
    s.cacti = 25 ∧ s.cactiPrice = 22 ∧
    s.airPlants = 40 ∧ s.airPlantPrice = 15 ∧
    s.fullTimeWorkers = 3 ∧ s.fullTimeWage = 65 ∧
    s.partTimeWorkers = 2 ∧ s.partTimeWage = 45 ∧
    s.ceramicPotsCost = 280 ∧
    s.plasticPotsCost = 150 ∧
    s.fertilizersCost = 100 ∧
    s.toolsCost = 125 ∧
    s.utilityBill = 225 ∧
    s.tax = 550 ∧
    moneyLeft s = 3755 := by
  sorry

end plant_supplier_money_left_l535_53570


namespace smallest_w_l535_53576

theorem smallest_w (w : ℕ+) : 
  (∃ k : ℕ, 936 * w.val = k * 2^5) ∧ 
  (∃ k : ℕ, 936 * w.val = k * 3^3) ∧ 
  (∃ k : ℕ, 936 * w.val = k * 14^2) → 
  w ≥ 1764 :=
by sorry

end smallest_w_l535_53576


namespace f_minimum_value_l535_53534

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.exp x / x + x - Real.log x

-- State the theorem
theorem f_minimum_value :
  ∃ (min_value : ℝ), min_value = Real.exp 1 + 1 ∧
  ∀ (x : ℝ), x > 0 → f x ≥ min_value :=
sorry

end f_minimum_value_l535_53534


namespace disjoint_subsets_remainder_l535_53589

def S : Finset Nat := Finset.range 12

def count_disjoint_subsets (S : Finset Nat) : Nat :=
  (3^S.card - 2 * 2^S.card + 1) / 2

theorem disjoint_subsets_remainder (S : Finset Nat) (h : S = Finset.range 12) :
  count_disjoint_subsets S % 1000 = 625 := by
  sorry

end disjoint_subsets_remainder_l535_53589


namespace circle_area_from_circumference_l535_53587

/-- Given a circle with circumference 36 cm, its area is 324/π square centimeters. -/
theorem circle_area_from_circumference : 
  ∀ (r : ℝ), 2 * π * r = 36 → π * r^2 = 324 / π := by
  sorry

end circle_area_from_circumference_l535_53587


namespace surjective_function_theorem_l535_53594

def is_surjective (f : ℕ → ℕ) : Prop :=
  ∀ y : ℕ, ∃ x : ℕ, f x = y

theorem surjective_function_theorem (f : ℕ → ℕ) 
  (h_surj : is_surjective f)
  (h_div : ∀ (m n : ℕ) (p : ℕ), Nat.Prime p → (p ∣ f (m + n) ↔ p ∣ f m + f n)) :
  ∀ n : ℕ, f n = n := by sorry

end surjective_function_theorem_l535_53594


namespace weaver_productivity_l535_53597

/-- Given that 16 weavers can weave 64 mats in 16 days at a constant rate,
    prove that 4 weavers can weave 16 mats in 4 days at the same rate. -/
theorem weaver_productivity 
  (rate : ℝ) -- The constant rate of weaving (mats per weaver per day)
  (h1 : 16 * rate * 16 = 64) -- 16 weavers can weave 64 mats in 16 days
  : 4 * rate * 4 = 16 := by
  sorry

end weaver_productivity_l535_53597


namespace stone_122_is_9_l535_53553

/-- Represents the number of stones in the line -/
def n : ℕ := 17

/-- The target count we're looking for -/
def target : ℕ := 122

/-- Function to determine the original stone number given a count in the sequence -/
def originalStone (count : ℕ) : ℕ :=
  let modulo := count % (2 * (n - 1))
  if modulo ≤ n then
    modulo
  else
    2 * n - modulo

/-- Theorem stating that the stone counted as 122 is originally stone number 9 -/
theorem stone_122_is_9 : originalStone target = 9 := by
  sorry


end stone_122_is_9_l535_53553


namespace sin_390_degrees_l535_53500

theorem sin_390_degrees : Real.sin (390 * π / 180) = 1 / 2 := by
  sorry

end sin_390_degrees_l535_53500


namespace total_feed_amount_l535_53595

/-- Proves that the total amount of feed is 27 pounds given the specified conditions --/
theorem total_feed_amount (cheap_cost expensive_cost mix_cost cheap_amount : ℝ) 
  (h1 : cheap_cost = 0.17)
  (h2 : expensive_cost = 0.36)
  (h3 : mix_cost = 0.26)
  (h4 : cheap_amount = 14.2105263158)
  (h5 : cheap_cost * cheap_amount + expensive_cost * (total - cheap_amount) = mix_cost * total)
  : total = 27 :=
by sorry

#check total_feed_amount

end total_feed_amount_l535_53595


namespace bouncy_balls_per_package_l535_53504

theorem bouncy_balls_per_package (total_packages : Nat) (total_balls : Nat) :
  total_packages = 16 →
  total_balls = 160 →
  ∃ (balls_per_package : Nat), balls_per_package * total_packages = total_balls ∧ balls_per_package = 10 := by
  sorry

end bouncy_balls_per_package_l535_53504


namespace whole_number_between_36_and_40_l535_53531

theorem whole_number_between_36_and_40 (M : ℕ) : 
  (9 < (M : ℚ) / 4) ∧ ((M : ℚ) / 4 < 10) → M = 37 ∨ M = 38 ∨ M = 39 := by
  sorry

end whole_number_between_36_and_40_l535_53531


namespace quadratic_always_positive_implies_k_range_l535_53506

theorem quadratic_always_positive_implies_k_range (k : ℝ) :
  (∀ x : ℝ, x^2 + 2*k*x - (k - 2) > 0) → k ∈ Set.Ioo (-2 : ℝ) 1 :=
by
  sorry

end quadratic_always_positive_implies_k_range_l535_53506


namespace kite_altitude_l535_53545

theorem kite_altitude (C D K : ℝ × ℝ) (h1 : D.1 - C.1 = 15) (h2 : C.2 = D.2)
  (h3 : K.1 = C.1) (h4 : Real.tan (45 * π / 180) = (K.2 - C.2) / (K.1 - C.1))
  (h5 : Real.tan (30 * π / 180) = (K.2 - D.2) / (D.1 - K.1)) :
  K.2 - C.2 = 15 * (Real.sqrt 3 - 1) / 2 := by
  sorry

end kite_altitude_l535_53545


namespace min_roads_theorem_l535_53559

/-- A graph representing cities and roads -/
structure CityGraph where
  num_cities : ℕ
  num_roads : ℕ
  is_connected : Bool

/-- Check if a given number of roads is sufficient for connectivity -/
def is_sufficient (g : CityGraph) : Prop :=
  g.is_connected = true

/-- The minimum number of roads needed for connectivity -/
def min_roads_for_connectivity (num_cities : ℕ) : ℕ :=
  191

/-- Theorem stating that 191 roads are sufficient and necessary for connectivity -/
theorem min_roads_theorem (g : CityGraph) :
  g.num_cities = 21 → 
  (g.num_roads ≥ 191 → is_sufficient g) ∧
  (is_sufficient g → g.num_roads ≥ 191) :=
sorry

#check min_roads_theorem

end min_roads_theorem_l535_53559


namespace product_and_sum_relations_l535_53536

/-- Given positive integers p, q, r satisfying the specified conditions, prove that p - r = -430 --/
theorem product_and_sum_relations (p q r : ℕ+) 
  (h_product : p * q * r = Nat.factorial 10)
  (h_sum1 : p * q + p + q = 2450)
  (h_sum2 : q * r + q + r = 1012)
  (h_sum3 : r * p + r + p = 2020) :
  (p : ℤ) - (r : ℤ) = -430 := by
  sorry

end product_and_sum_relations_l535_53536


namespace correct_addition_l535_53599

theorem correct_addition (x : ℤ) (h : x + 21 = 52) : x + 40 = 71 := by
  sorry

end correct_addition_l535_53599


namespace soap_box_theorem_l535_53532

/-- The number of bars of soap in each box of bars -/
def bars_per_box : ℕ := 5

/-- The smallest number of each type of soap sold -/
def min_sold : ℕ := 95

/-- The number of bottles of soap in each box of bottles -/
def bottles_per_box : ℕ := 19

theorem soap_box_theorem :
  ∃ (bar_boxes bottle_boxes : ℕ),
    bar_boxes * bars_per_box = bottle_boxes * bottles_per_box ∧
    bar_boxes * bars_per_box = min_sold ∧
    bottle_boxes * bottles_per_box = min_sold ∧
    bottles_per_box > 1 ∧
    bottles_per_box < min_sold :=
by sorry

end soap_box_theorem_l535_53532


namespace regular_polygon_120_degrees_l535_53546

/-- A regular polygon with interior angles of 120° has 6 sides -/
theorem regular_polygon_120_degrees (n : ℕ) : 
  (n ≥ 3) → 
  ((n - 2) * 180 / n = 120) → 
  n = 6 := by
sorry

end regular_polygon_120_degrees_l535_53546


namespace S_bounds_l535_53596

-- Define the function S
def S (x y z : ℝ) : ℝ := 2*x^2*y^2 + 2*x^2*z^2 + 2*y^2*z^2 - x^4 - y^4 - z^4

-- State the theorem
theorem S_bounds :
  ∀ x y z : ℝ,
  (0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) →
  (5 ≤ x ∧ x ≤ 8) →
  (5 ≤ y ∧ y ≤ 8) →
  (5 ≤ z ∧ z ≤ 8) →
  1875 ≤ S x y z ∧ S x y z ≤ 31488 :=
by sorry

end S_bounds_l535_53596


namespace absolute_value_of_seven_minus_sqrt_53_l535_53522

theorem absolute_value_of_seven_minus_sqrt_53 :
  |7 - Real.sqrt 53| = Real.sqrt 53 - 7 := by sorry

end absolute_value_of_seven_minus_sqrt_53_l535_53522


namespace sqrt_equation_solution_l535_53583

theorem sqrt_equation_solution :
  ∃! z : ℝ, Real.sqrt (-3 + 3 * z) = 9 :=
by
  -- The proof goes here
  sorry

end sqrt_equation_solution_l535_53583


namespace completed_square_form_l535_53543

theorem completed_square_form (x : ℝ) :
  x^2 - 6*x - 5 = 0 ↔ (x - 3)^2 = 14 := by
  sorry

end completed_square_form_l535_53543


namespace no_integer_solution_quadratic_prime_l535_53592

theorem no_integer_solution_quadratic_prime : 
  ¬ ∃ (x : ℤ), Nat.Prime (Int.natAbs (4 * x^2 - 39 * x + 35)) := by
sorry

end no_integer_solution_quadratic_prime_l535_53592


namespace function_inequality_l535_53521

theorem function_inequality (f : ℝ → ℝ) (h : Differentiable ℝ f) 
  (h1 : ∀ x, (x - 1) * deriv f x ≤ 0) : 
  f 0 + f 2 ≤ 2 * f 1 := by
  sorry

end function_inequality_l535_53521


namespace fraction_relation_l535_53518

theorem fraction_relation (n d : ℚ) (k : ℚ) : 
  d = k * (2 * n) →
  (n + 1) / (d + 1) = 3 / 5 →
  n / d = 5 / 9 →
  k = 9 / 10 := by
sorry

end fraction_relation_l535_53518


namespace geometric_sequence_common_ratio_l535_53578

/-- Given a geometric sequence {a_n} with a₂ = 8 and a₅ = 64, prove that the common ratio q = 2 -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- Geometric sequence definition
  a 2 = 8 →                    -- Given condition
  a 5 = 64 →                   -- Given condition
  q = 2 := by                  -- Conclusion to prove
sorry


end geometric_sequence_common_ratio_l535_53578


namespace triangle_side_calculation_l535_53551

theorem triangle_side_calculation (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a = 10 →
  A = π / 4 →
  B = π / 6 →
  a / Real.sin A = b / Real.sin B →
  b = 5 * Real.sqrt 2 := by
sorry

end triangle_side_calculation_l535_53551


namespace collectors_edition_dolls_combined_l535_53563

theorem collectors_edition_dolls_combined (dina_dolls : ℕ) (ivy_dolls : ℕ) (luna_dolls : ℕ) :
  dina_dolls = 60 →
  dina_dolls = 2 * ivy_dolls →
  ivy_dolls = luna_dolls + 10 →
  (2 : ℕ) * (ivy_dolls * 2) = 3 * ivy_dolls →
  2 * luna_dolls = luna_dolls →
  (2 : ℕ) * (ivy_dolls * 2) / 3 + luna_dolls / 2 = 30 := by
sorry

end collectors_edition_dolls_combined_l535_53563


namespace only_one_divides_power_minus_one_l535_53598

theorem only_one_divides_power_minus_one :
  ∀ n : ℕ+, n ∣ (2^n.val - 1) → n = 1 := by
  sorry

end only_one_divides_power_minus_one_l535_53598


namespace power_of_power_three_l535_53548

theorem power_of_power_three : (3^2)^4 = 6561 := by
  sorry

end power_of_power_three_l535_53548


namespace quadratic_roots_and_integer_k_l535_53572

/-- Represents a quadratic equation of the form kx^2 + (k-2)x - 2 = 0 --/
def QuadraticEquation (k : ℝ) : ℝ → Prop :=
  fun x => k * x^2 + (k - 2) * x - 2 = 0

theorem quadratic_roots_and_integer_k :
  ∀ k : ℝ, k ≠ 0 →
    (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ QuadraticEquation k x₁ ∧ QuadraticEquation k x₂) ∧
    (∃ k' : ℤ, k' ∈ ({-2, -1, 1, 2} : Set ℤ) ∧
      ∃ x₁ x₂ : ℤ, QuadraticEquation (k' : ℝ) x₁ ∧ QuadraticEquation (k' : ℝ) x₂) :=
by sorry

#check quadratic_roots_and_integer_k

end quadratic_roots_and_integer_k_l535_53572


namespace transportation_charges_l535_53520

theorem transportation_charges 
  (purchase_price : ℕ) 
  (repair_cost : ℕ) 
  (profit_percentage : ℚ) 
  (selling_price : ℕ) 
  (h1 : purchase_price = 13000)
  (h2 : repair_cost = 5000)
  (h3 : profit_percentage = 1/2)
  (h4 : selling_price = 28500) :
  ∃ (transportation_charges : ℕ),
    selling_price = (purchase_price + repair_cost + transportation_charges) * (1 + profit_percentage) ∧
    transportation_charges = 1000 :=
by sorry

end transportation_charges_l535_53520


namespace derivative_cos_squared_at_pi_eighth_l535_53577

/-- Given a function f(x) = cos²(2x), its derivative at π/8 is -2. -/
theorem derivative_cos_squared_at_pi_eighth (f : ℝ → ℝ) (h : ∀ x, f x = Real.cos (2 * x) ^ 2) :
  deriv f (π / 8) = -2 := by
  sorry

end derivative_cos_squared_at_pi_eighth_l535_53577


namespace ab_nonpositive_l535_53527

theorem ab_nonpositive (a b : ℚ) (ha : |a| = a) (hb : |b| = -b) : a * b ≤ 0 := by
  sorry

end ab_nonpositive_l535_53527


namespace collinear_probability_l535_53590

/-- The number of dots in each row or column of the grid -/
def gridSize : ℕ := 5

/-- The total number of dots in the grid -/
def totalDots : ℕ := gridSize * gridSize

/-- The number of dots we are selecting -/
def selectedDots : ℕ := 5

/-- The number of possible collinear sets of 5 dots in the grid -/
def collinearSets : ℕ := 2 * gridSize + 2

/-- The probability of selecting 5 collinear dots from a 5x5 grid -/
theorem collinear_probability : 
  (collinearSets : ℚ) / Nat.choose totalDots selectedDots = 2 / 8855 := by sorry

end collinear_probability_l535_53590


namespace smallest_four_digit_congruent_to_one_mod_23_l535_53564

theorem smallest_four_digit_congruent_to_one_mod_23 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n ≡ 1 [MOD 23] → 1013 ≤ n :=
by sorry

end smallest_four_digit_congruent_to_one_mod_23_l535_53564


namespace remainder_theorem_remainder_is_29_l535_53586

/-- The polynomial p(x) = x^4 + 2x^2 + 5 -/
def p (x : ℝ) : ℝ := x^4 + 2*x^2 + 5

/-- The remainder theorem: For a polynomial p(x) and a real number a,
    the remainder when p(x) is divided by (x - a) is equal to p(a) -/
theorem remainder_theorem (p : ℝ → ℝ) (a : ℝ) :
  ∃ q : ℝ → ℝ, ∀ x, p x = (x - a) * q x + p a :=
sorry

theorem remainder_is_29 :
  ∃ q : ℝ → ℝ, ∀ x, p x = (x - 2) * q x + 29 :=
sorry

end remainder_theorem_remainder_is_29_l535_53586


namespace largest_seventh_term_coefficient_l535_53509

/-- 
Given that in the expansion of (x + y)^n the coefficient of the seventh term is the largest,
this theorem states that n must be either 11, 12, or 13.
-/
theorem largest_seventh_term_coefficient (n : ℕ) : 
  (∀ k : ℕ, k ≠ 6 → (n.choose k) ≤ (n.choose 6)) → 
  n = 11 ∨ n = 12 ∨ n = 13 := by
  sorry

end largest_seventh_term_coefficient_l535_53509


namespace arnel_pencil_boxes_l535_53503

/-- The number of boxes of pencils Arnel had -/
def number_of_boxes : ℕ := sorry

/-- The number of pencils in each box -/
def pencils_per_box : ℕ := 5

/-- The number of pencils Arnel kept for himself -/
def pencils_kept : ℕ := 10

/-- The number of Arnel's friends -/
def number_of_friends : ℕ := 5

/-- The number of pencils each friend received -/
def pencils_per_friend : ℕ := 8

theorem arnel_pencil_boxes :
  number_of_boxes = 10 ∧
  number_of_boxes * pencils_per_box = 
    pencils_kept + number_of_friends * pencils_per_friend :=
by sorry

end arnel_pencil_boxes_l535_53503


namespace series_divergent_l535_53519

open Complex

/-- The series ∑_{n=1}^{∞} (e^(iπ/n))/n is divergent -/
theorem series_divergent : 
  ¬ Summable (fun n : ℕ => (exp (I * π / n : ℂ)) / n) :=
sorry

end series_divergent_l535_53519


namespace smallest_five_digit_divisible_by_53_and_3_l535_53575

theorem smallest_five_digit_divisible_by_53_and_3 : ∃ n : ℕ,
  (n ≥ 10000 ∧ n < 100000) ∧  -- five-digit positive integer
  n % 53 = 0 ∧                -- divisible by 53
  n % 3 = 0 ∧                 -- divisible by 3
  n = 10062 ∧                 -- the number is 10062
  ∀ m : ℕ, (m ≥ 10000 ∧ m < 100000 ∧ m % 53 = 0 ∧ m % 3 = 0) → m ≥ n :=
by sorry

end smallest_five_digit_divisible_by_53_and_3_l535_53575


namespace fractional_equation_one_l535_53502

theorem fractional_equation_one (x : ℝ) : 
  x ≠ 0 ∧ x ≠ -1 → (2 / x = 3 / (x + 1) ↔ x = 2) := by sorry

end fractional_equation_one_l535_53502


namespace remainder_cube_l535_53508

theorem remainder_cube (n : ℤ) : n % 13 = 5 → n^3 % 17 = 6 := by
  sorry

end remainder_cube_l535_53508


namespace conic_is_hyperbola_l535_53581

/-- The equation of a conic section -/
def conic_equation (x y : ℝ) : Prop :=
  x^2 - 64*y^2 + 16*x - 32 = 0

/-- Definition of a hyperbola -/
def is_hyperbola (eq : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (a b c d e : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ 
  (∀ x y, eq x y ↔ ((x - c)^2 / a^2) - ((y - d)^2 / b^2) = 1) ∨
  (∀ x y, eq x y ↔ ((y - d)^2 / a^2) - ((x - c)^2 / b^2) = 1)

/-- Theorem: The given equation represents a hyperbola -/
theorem conic_is_hyperbola : is_hyperbola conic_equation :=
sorry

end conic_is_hyperbola_l535_53581


namespace max_a_value_l535_53588

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 4

-- Define the property that f(x) ≤ 6 for all x in (0,2]
def property (a : ℝ) : Prop :=
  ∀ x : ℝ, 0 < x ∧ x ≤ 2 → f a x ≤ 6

-- State the theorem
theorem max_a_value :
  (∃ a : ℝ, property a) →
  (∃ a_max : ℝ, property a_max ∧ ∀ a : ℝ, property a → a ≤ a_max) →
  (∀ a_max : ℝ, (property a_max ∧ ∀ a : ℝ, property a → a ≤ a_max) → a_max = -1) :=
by sorry

end max_a_value_l535_53588


namespace larger_number_proof_l535_53550

theorem larger_number_proof (x y : ℝ) (h1 : x - y = 1670) (h2 : 0.075 * x = 0.125 * y) (h3 : x > 0) (h4 : y > 0) : max x y = 4175 := by
  sorry

end larger_number_proof_l535_53550


namespace linda_income_l535_53552

/-- Represents the tax structure and Linda's income --/
structure TaxInfo where
  p : ℝ  -- base tax rate in decimal form
  income : ℝ  -- Linda's annual income

/-- Calculates the total tax based on the given tax structure --/
def calculateTax (info : TaxInfo) : ℝ :=
  let baseTax := info.p * 35000
  let excessTax := (info.p + 0.03) * (info.income - 35000)
  baseTax + excessTax

/-- Theorem stating that Linda's income is $42000 given the tax conditions --/
theorem linda_income (info : TaxInfo) :
  (calculateTax info = (info.p + 0.005) * info.income) →
  info.income = 42000 := by
  sorry

#check linda_income

end linda_income_l535_53552
