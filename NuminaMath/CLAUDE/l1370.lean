import Mathlib

namespace trapezoid_cd_length_l1370_137089

structure Trapezoid (A B C D : ℝ × ℝ) :=
  (parallel : (A.2 - D.2) / (A.1 - D.1) = (B.2 - C.2) / (B.1 - C.1))
  (bd_length : Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) = 2)
  (angle_dbc : Real.arccos ((B.1 - D.1) * (C.1 - B.1) + (B.2 - D.2) * (C.2 - B.2)) / 
    (Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) * Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)) = 36 * π / 180)
  (angle_bda : Real.arccos ((B.1 - D.1) * (A.1 - D.1) + (B.2 - D.2) * (A.2 - D.2)) / 
    (Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) * Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2)) = 72 * π / 180)
  (ratio : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) / Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) = 5/3)

theorem trapezoid_cd_length (A B C D : ℝ × ℝ) (t : Trapezoid A B C D) :
  Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 4/3 := by
  sorry

end trapezoid_cd_length_l1370_137089


namespace union_eq_right_iff_complement_subset_l1370_137098

variable {U : Type*} -- Universal set
variable (A B : Set U) -- Sets A and B

theorem union_eq_right_iff_complement_subset :
  A ∪ B = B ↔ (Bᶜ : Set U) ⊆ (Aᶜ : Set U) := by sorry

end union_eq_right_iff_complement_subset_l1370_137098


namespace difference_divisible_by_99_l1370_137053

/-- Represents a three-digit number formed by digits a, b, and c -/
def three_digit_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

/-- The largest three-digit number formed by digits a, b, and c where a > b > c -/
def largest_number (a b c : ℕ) : ℕ := three_digit_number a b c

/-- The smallest three-digit number formed by digits a, b, and c where a > b > c -/
def smallest_number (a b c : ℕ) : ℕ := three_digit_number c b a

theorem difference_divisible_by_99 (a b c : ℕ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > 0) (h4 : a < 10) (h5 : b < 10) (h6 : c < 10) :
  ∃ k : ℕ, largest_number a b c - smallest_number a b c = 99 * k :=
sorry

end difference_divisible_by_99_l1370_137053


namespace regular_polygon_sides_l1370_137084

theorem regular_polygon_sides (n : ℕ) (h : n > 2) :
  (180 * (n - 2) : ℝ) / n = 156 → n = 15 := by
  sorry

end regular_polygon_sides_l1370_137084


namespace cameron_questions_total_l1370_137022

/-- Represents a tour group with a number of people and an optional inquisitive person. -/
structure TourGroup where
  people : Nat
  inquisitivePerson : Option Nat

/-- Calculates the number of questions answered for a given tour group. -/
def questionsAnswered (group : TourGroup) (questionsPerPerson : Nat) : Nat :=
  match group.inquisitivePerson with
  | none => group.people * questionsPerPerson
  | some n => (group.people - 1) * questionsPerPerson + n * questionsPerPerson

/-- Represents Cameron's tour day. -/
def cameronsTourDay : List TourGroup := [
  { people := 6, inquisitivePerson := none },
  { people := 11, inquisitivePerson := none },
  { people := 8, inquisitivePerson := some 3 },
  { people := 7, inquisitivePerson := none }
]

/-- The theorem stating the total number of questions Cameron answered. -/
theorem cameron_questions_total :
  (cameronsTourDay.map (questionsAnswered · 2)).sum = 68 := by
  sorry

end cameron_questions_total_l1370_137022


namespace inequality_proof_l1370_137096

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  a^(1/3) * b^(1/3) + c^(1/3) * d^(1/3) ≤ (a+b+c)^(1/3) * (a+c+d)^(1/3) ∧
  (a^(1/3) * b^(1/3) + c^(1/3) * d^(1/3) = (a+b+c)^(1/3) * (a+c+d)^(1/3) ↔
   b = (a/c)*(a+c) ∧ d = (c/a)*(a+c)) :=
by sorry

end inequality_proof_l1370_137096


namespace roots_and_coefficients_l1370_137019

theorem roots_and_coefficients (θ : Real) (m : Real) :
  0 < θ ∧ θ < 2 * Real.pi →
  (2 * Real.sin θ ^ 2 - (Real.sqrt 3 + 1) * Real.sin θ + m = 0) ∧
  (2 * Real.cos θ ^ 2 - (Real.sqrt 3 + 1) * Real.cos θ + m = 0) →
  (Real.sin θ ^ 2 / (Real.sin θ - Real.cos θ) + Real.cos θ ^ 2 / (Real.cos θ - Real.sin θ) = (Real.sqrt 3 + 1) / 2) ∧
  (m = Real.sqrt 3 / 2) := by
sorry

end roots_and_coefficients_l1370_137019


namespace distribute_books_count_l1370_137014

/-- The number of ways to distribute books among students -/
def distribute_books : ℕ :=
  let num_students : ℕ := 4
  let num_novels : ℕ := 4
  let num_anthologies : ℕ := 1
  -- Category 1: Each student gets 1 novel, anthology to any student
  let category1 : ℕ := num_students
  -- Category 2: Anthology to one student, novels distributed to others
  let category2 : ℕ := num_students * (num_students - 1)
  category1 + category2

/-- Theorem stating that the number of distribution methods is 16 -/
theorem distribute_books_count : distribute_books = 16 := by
  sorry

end distribute_books_count_l1370_137014


namespace contrapositive_sum_irrational_l1370_137057

/-- The contrapositive of "If a + b is irrational, then at least one of a or b is irrational" -/
theorem contrapositive_sum_irrational (a b : ℝ) :
  (¬(∃ q : ℚ, (a : ℝ) = q) ∨ ¬(∃ q : ℚ, (b : ℝ) = q) → ¬(∃ q : ℚ, (a + b : ℝ) = q)) ↔
  ((∃ q : ℚ, (a : ℝ) = q) ∧ (∃ q : ℚ, (b : ℝ) = q) → (∃ q : ℚ, (a + b : ℝ) = q)) :=
by sorry

end contrapositive_sum_irrational_l1370_137057


namespace line_slope_l1370_137036

theorem line_slope (x y : ℝ) (h : x / 4 + y / 3 = 1) : 
  ∃ m b : ℝ, y = m * x + b ∧ m = -3/4 := by
sorry

end line_slope_l1370_137036


namespace average_headcount_proof_l1370_137088

def fall_headcount_03_04 : ℕ := 11500
def fall_headcount_04_05 : ℕ := 11600
def fall_headcount_05_06 : ℕ := 11300

def average_headcount : ℕ := 
  (fall_headcount_03_04 + fall_headcount_04_05 + fall_headcount_05_06 + 1) / 3

theorem average_headcount_proof :
  average_headcount = 11467 := by sorry

end average_headcount_proof_l1370_137088


namespace average_first_five_primes_gt_50_l1370_137037

def first_five_primes_gt_50 : List Nat := [53, 59, 61, 67, 71]

def average (lst : List Nat) : ℚ :=
  (lst.sum : ℚ) / lst.length

theorem average_first_five_primes_gt_50 :
  average first_five_primes_gt_50 = 62.2 := by
  sorry

end average_first_five_primes_gt_50_l1370_137037


namespace article_cost_l1370_137071

/-- The cost of an article, given selling conditions -/
theorem article_cost : ∃ (cost : ℝ), 
  (580 - cost) = 1.08 * (520 - cost) ∧ 
  cost = 230 := by
  sorry

end article_cost_l1370_137071


namespace divisor_problem_l1370_137026

theorem divisor_problem (n d : ℕ) (h1 : n % d = 3) (h2 : (n^2) % d = 3) : d = 3 := by
  sorry

end divisor_problem_l1370_137026


namespace square_triangulation_100_points_l1370_137049

/-- Represents a triangulation of a square with additional points -/
structure SquareTriangulation where
  n : ℕ  -- number of additional points inside the square
  triangles : ℕ  -- number of triangles in the triangulation

/-- Theorem: A square triangulation with 100 additional points has 202 triangles -/
theorem square_triangulation_100_points :
  ∀ (st : SquareTriangulation), st.n = 100 → st.triangles = 202 := by
  sorry

end square_triangulation_100_points_l1370_137049


namespace inequality_abc_at_least_one_positive_l1370_137062

-- Problem 1
theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b + c ≥ Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) := by
  sorry

-- Problem 2
theorem at_least_one_positive (x y z : ℝ) :
  let a := x^2 - 2*y + π/2
  let b := y^2 - 2*z + π/3
  let c := z^2 - 2*x + π/6
  0 < a ∨ 0 < b ∨ 0 < c := by
  sorry

end inequality_abc_at_least_one_positive_l1370_137062


namespace shane_current_age_l1370_137028

/-- Given that twenty years ago Shane was 2 times older than Garret is now,
    and Garret is currently 12 years old, prove that Shane is 44 years old now. -/
theorem shane_current_age :
  (∀ (shane_age_now garret_age_now : ℕ),
    garret_age_now = 12 →
    shane_age_now - 20 = 2 * garret_age_now →
    shane_age_now = 44) :=
by sorry

end shane_current_age_l1370_137028


namespace prime_odd_sum_l1370_137015

theorem prime_odd_sum (a b : ℕ) : 
  Prime a → Odd b → a^2 + b = 2001 → a + b = 1999 := by sorry

end prime_odd_sum_l1370_137015


namespace complex_power_eight_l1370_137016

theorem complex_power_eight (a b : ℝ) (h : (a : ℂ) + Complex.I = 1 - b * Complex.I) : 
  (a + b * Complex.I : ℂ) ^ 8 = 16 := by sorry

end complex_power_eight_l1370_137016


namespace race_pace_cristina_pace_l1370_137033

/-- The race between Nicky and Cristina -/
theorem race_pace (nicky_pace : ℝ) (race_time : ℝ) (head_start : ℝ) : ℝ :=
  let nicky_distance := nicky_pace * race_time
  let cristina_distance := nicky_distance + head_start
  cristina_distance / race_time

/-- Cristina's pace in the race -/
theorem cristina_pace : race_pace 3 36 36 = 4 := by
  sorry

end race_pace_cristina_pace_l1370_137033


namespace fraction_simplification_l1370_137085

theorem fraction_simplification :
  (1 / 2 + 1 / 5) / (3 / 7 - 1 / 14) = 49 / 25 := by
  sorry

end fraction_simplification_l1370_137085


namespace angle_complement_l1370_137017

/-- 
Given an angle x and its complement y, prove that y equals 90° minus x.
-/
theorem angle_complement (x y : ℝ) (h : x + y = 90) : y = 90 - x := by
  sorry

end angle_complement_l1370_137017


namespace power_mod_five_l1370_137010

theorem power_mod_five : 2^345 % 5 = 2 := by
  sorry

end power_mod_five_l1370_137010


namespace x_squared_plus_reciprocal_squared_l1370_137023

theorem x_squared_plus_reciprocal_squared (x : ℝ) (h : x + 1/x = 8) : x^2 + 1/x^2 = 62 := by
  sorry

end x_squared_plus_reciprocal_squared_l1370_137023


namespace max_walk_distance_l1370_137077

/-- Represents the board system with a person walking on it. -/
structure BoardSystem where
  l : ℝ  -- Length of the board
  m : ℝ  -- Mass of the board
  x : ℝ  -- Distance the person walks from the stone

/-- The conditions for the board system to be in equilibrium. -/
def is_equilibrium (bs : BoardSystem) : Prop :=
  bs.l = 20 ∧  -- Board length is 20 meters
  bs.x ≤ bs.l ∧  -- Person cannot walk beyond the board length
  2 * bs.m * (bs.l / 4) = bs.m * (3 * bs.l / 8) + (bs.m / 2) * (bs.x - bs.l / 4)

/-- The theorem stating the maximum distance a person can walk. -/
theorem max_walk_distance (bs : BoardSystem) :
  is_equilibrium bs → bs.x = bs.l / 2 := by
  sorry

#check max_walk_distance

end max_walk_distance_l1370_137077


namespace f_min_value_l1370_137011

def f (x : ℝ) : ℝ := |2*x - 1| + |3*x - 2| + |4*x - 3| + |5*x - 4|

theorem f_min_value : ∀ x : ℝ, f x ≥ 1 := by
  sorry

end f_min_value_l1370_137011


namespace grocery_store_soda_l1370_137092

theorem grocery_store_soda (diet_soda apples : ℕ) 
  (h1 : diet_soda = 32)
  (h2 : apples = 78)
  (h3 : ∃ regular_soda : ℕ, regular_soda + diet_soda = apples + 26) :
  ∃ regular_soda : ℕ, regular_soda = 72 := by
sorry

end grocery_store_soda_l1370_137092


namespace f_properties_l1370_137001

noncomputable def f (x : ℝ) : ℝ := 1/2 - 1/(2^x + 1)

theorem f_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) ∧
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → f x ∈ Set.Icc (1/6) (3/10)) ∧
  f 1 = 1/6 ∧ f 2 = 3/10 :=
by sorry

end f_properties_l1370_137001


namespace allocation_methods_3_6_3_l1370_137086

/-- The number of ways to allocate doctors and nurses to schools -/
def allocation_methods (num_doctors num_nurses num_schools : ℕ) : ℕ :=
  (num_doctors.choose 1 * num_nurses.choose 2) *
  ((num_doctors - 1).choose 1 * (num_nurses - 2).choose 2) *
  ((num_doctors - 2).choose 1 * (num_nurses - 4).choose 2)

/-- Theorem stating that the number of allocation methods for 3 doctors and 6 nurses to 3 schools is 540 -/
theorem allocation_methods_3_6_3 :
  allocation_methods 3 6 3 = 540 := by
  sorry

end allocation_methods_3_6_3_l1370_137086


namespace parallel_lines_count_parallel_lines_problem_l1370_137070

/-- Given two sets of intersecting parallel lines, the number of parallelograms formed is the product of the spaces between the lines in each set. -/
def parallelogram_count (lines_set1 lines_set2 : ℕ) : ℕ := (lines_set1 - 1) * (lines_set2 - 1)

/-- The problem statement -/
theorem parallel_lines_count (lines_set1 : ℕ) (parallelograms : ℕ) : ℕ :=
  let lines_set2 := (parallelograms / (lines_set1 - 1)) + 1
  lines_set2

/-- The main theorem to prove -/
theorem parallel_lines_problem :
  parallel_lines_count 6 420 = 85 := by
  sorry

end parallel_lines_count_parallel_lines_problem_l1370_137070


namespace line_not_in_third_quadrant_line_through_two_points_l1370_137072

-- Define a line with coefficients A, B, and C
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

-- Define a point in 2D plane
structure Point where
  x : ℝ
  y : ℝ

-- Theorem 1: Line does not pass through third quadrant
theorem line_not_in_third_quadrant (l : Line) 
  (h1 : l.A * l.B < 0) (h2 : l.B * l.C < 0) : 
  ∀ (p : Point), p.x < 0 ∧ p.y < 0 → l.A * p.x - l.B * p.y - l.C ≠ 0 :=
sorry

-- Theorem 2: Line equation through two distinct points
theorem line_through_two_points (p1 p2 : Point) (h : p1 ≠ p2) :
  ∀ (p : Point), (p2.x - p1.x) * (p.y - p1.y) = (p2.y - p1.y) * (p.x - p1.x) ↔
  ∃ (t : ℝ), p.x = p1.x + t * (p2.x - p1.x) ∧ p.y = p1.y + t * (p2.y - p1.y) :=
sorry

end line_not_in_third_quadrant_line_through_two_points_l1370_137072


namespace total_spent_is_122_80_l1370_137002

-- Define the cost per deck
def cost_per_deck : ℚ := 8

-- Define the number of decks bought by each person
def victor_decks : ℕ := 6
def friend_a_decks : ℕ := 4
def friend_b_decks : ℕ := 5
def friend_c_decks : ℕ := 3

-- Define the discount rates
def discount_rate (n : ℕ) : ℚ :=
  if n ≥ 6 then 0.20
  else if n = 5 then 0.15
  else if n ≥ 3 then 0.10
  else 0

-- Define the function to calculate the total cost for a person
def total_cost (decks : ℕ) : ℚ :=
  let base_cost := cost_per_deck * decks
  base_cost - (base_cost * discount_rate decks)

-- Theorem statement
theorem total_spent_is_122_80 :
  total_cost victor_decks +
  total_cost friend_a_decks +
  total_cost friend_b_decks +
  total_cost friend_c_decks = 122.80 :=
by sorry

end total_spent_is_122_80_l1370_137002


namespace x_squared_minus_y_squared_l1370_137020

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 2/5) 
  (h2 : x - y = 1/10) : 
  x^2 - y^2 = 1/25 := by
sorry

end x_squared_minus_y_squared_l1370_137020


namespace range_of_f_l1370_137087

def f (x : ℤ) : ℤ := (x - 1)^2 - 1

def domain : Set ℤ := {-1, 0, 1, 2, 3}

theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {-1, 0, 3} := by sorry

end range_of_f_l1370_137087


namespace negation_equivalence_l1370_137000

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  sum_180 : angles 0 + angles 1 + angles 2 = 180
  all_positive : ∀ i, angles i > 0

-- Define what it means for an angle to be obtuse
def is_obtuse (angle : ℝ) : Prop := angle > 90

-- Define the proposition "A triangle has at most one obtuse angle"
def at_most_one_obtuse (t : Triangle) : Prop :=
  (is_obtuse (t.angles 0) → ¬is_obtuse (t.angles 1) ∧ ¬is_obtuse (t.angles 2)) ∧
  (is_obtuse (t.angles 1) → ¬is_obtuse (t.angles 0) ∧ ¬is_obtuse (t.angles 2)) ∧
  (is_obtuse (t.angles 2) → ¬is_obtuse (t.angles 0) ∧ ¬is_obtuse (t.angles 1))

-- Define the negation of the proposition
def negation_at_most_one_obtuse (t : Triangle) : Prop :=
  ¬(at_most_one_obtuse t)

-- Define the condition "There are at least two obtuse angles in the triangle"
def at_least_two_obtuse (t : Triangle) : Prop :=
  (is_obtuse (t.angles 0) ∧ is_obtuse (t.angles 1)) ∨
  (is_obtuse (t.angles 1) ∧ is_obtuse (t.angles 2)) ∨
  (is_obtuse (t.angles 0) ∧ is_obtuse (t.angles 2))

-- Theorem: The negation of "at most one obtuse angle" is equivalent to "at least two obtuse angles"
theorem negation_equivalence (t : Triangle) :
  negation_at_most_one_obtuse t ↔ at_least_two_obtuse t :=
sorry

end negation_equivalence_l1370_137000


namespace fraction_simplification_l1370_137045

theorem fraction_simplification :
  (15 : ℚ) / 35 * 28 / 45 * 75 / 28 = 5 / 7 := by
  sorry

end fraction_simplification_l1370_137045


namespace original_denominator_problem_l1370_137061

theorem original_denominator_problem (d : ℚ) : 
  (3 : ℚ) / d ≠ 0 →
  (3 + 7 : ℚ) / (d + 7) = 1 / 3 →
  d = 23 := by
sorry

end original_denominator_problem_l1370_137061


namespace gcd_of_three_numbers_l1370_137068

theorem gcd_of_three_numbers : Nat.gcd 8650 (Nat.gcd 11570 28980) = 10 := by
  sorry

end gcd_of_three_numbers_l1370_137068


namespace caiden_roofing_problem_l1370_137054

/-- Calculates the number of feet of free metal roofing given the total required roofing,
    cost per foot, and amount paid for the remaining roofing. -/
def free_roofing (total_required : ℕ) (cost_per_foot : ℕ) (amount_paid : ℕ) : ℕ :=
  total_required - (amount_paid / cost_per_foot)

/-- Theorem stating that given the specific conditions of Mr. Caiden's roofing problem,
    the amount of free roofing is 250 feet. -/
theorem caiden_roofing_problem :
  free_roofing 300 8 400 = 250 := by
  sorry

end caiden_roofing_problem_l1370_137054


namespace triangle_side_length_l1370_137018

/-- Theorem: In a triangle ABC, if c + b = 12, A = 60°, and B = 30°, then c = 8 -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  c + b = 12 → A = 60 → B = 30 → c = 8 := by
  sorry

end triangle_side_length_l1370_137018


namespace degree_to_radian_conversion_l1370_137004

theorem degree_to_radian_conversion (π : Real) (h : π * 1 = 180) : 
  60 * (π / 180) = π / 3 := by sorry

end degree_to_radian_conversion_l1370_137004


namespace binary_110101_equals_53_l1370_137006

/-- Converts a binary number represented as a list of bits (least significant bit first) to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 110101 -/
def binary_110101 : List Bool := [true, false, true, false, true, true]

theorem binary_110101_equals_53 :
  binary_to_decimal binary_110101 = 53 := by
  sorry

end binary_110101_equals_53_l1370_137006


namespace no_solutions_equation_l1370_137051

theorem no_solutions_equation (x y : ℕ+) : x * (x + 1) ≠ 4 * y * (y + 1) := by
  sorry

end no_solutions_equation_l1370_137051


namespace problem_solving_probability_l1370_137076

theorem problem_solving_probability : 
  let p_arthur : ℚ := 1/4
  let p_bella : ℚ := 3/10
  let p_xavier : ℚ := 1/6
  let p_yvonne : ℚ := 1/2
  let p_zelda : ℚ := 5/8
  let p_not_zelda : ℚ := 1 - p_zelda
  let p_four_solve : ℚ := p_arthur * p_yvonne * p_bella * p_xavier * p_not_zelda
  p_four_solve = 9/3840 := by sorry

end problem_solving_probability_l1370_137076


namespace no_valid_numbers_l1370_137035

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  first : Nat
  middle : Nat
  last : Nat
  first_digit : first < 10
  middle_digit : middle < 10
  last_digit : last < 10
  three_digits : first ≠ 0

/-- Checks if a number is not divisible by 3 -/
def notDivisibleByThree (n : ThreeDigitNumber) : Prop :=
  (100 * n.first + 10 * n.middle + n.last) % 3 ≠ 0

/-- Checks if the sum of digits is less than 22 -/
def sumLessThan22 (n : ThreeDigitNumber) : Prop :=
  n.first + n.middle + n.last < 22

/-- Checks if the middle digit is twice the first digit -/
def middleTwiceFirst (n : ThreeDigitNumber) : Prop :=
  n.middle = 2 * n.first

theorem no_valid_numbers :
  ¬ ∃ (n : ThreeDigitNumber),
    notDivisibleByThree n ∧
    sumLessThan22 n ∧
    middleTwiceFirst n :=
sorry

end no_valid_numbers_l1370_137035


namespace geometric_sequence_sum_l1370_137065

/-- Given a geometric sequence {a_n} with a_1 = 3 and a_1 + a_3 + a_5 = 21,
    prove that a_3 + a_5 + a_7 = 42 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) : 
  (∀ n, a (n + 1) = a n * q) →  -- Geometric sequence definition
  a 1 = 3 →                    -- First condition
  a 1 + a 3 + a 5 = 21 →       -- Second condition
  a 3 + a 5 + a 7 = 42 :=      -- Conclusion to prove
by sorry

end geometric_sequence_sum_l1370_137065


namespace profit_margin_in_terms_of_selling_price_l1370_137013

/-- Given a selling price S, cost C, and profit margin M, prove that
    if S = 3C and M = (1/2n)C + (1/3n)S, then M = S/(2n) -/
theorem profit_margin_in_terms_of_selling_price
  (S C : ℝ) (n : ℝ) (hn : n ≠ 0) (M : ℝ) 
  (h_selling_price : S = 3 * C)
  (h_profit_margin : M = (1 / (2 * n)) * C + (1 / (3 * n)) * S) :
  M = S / (2 * n) := by
sorry

end profit_margin_in_terms_of_selling_price_l1370_137013


namespace smallest_number_l1370_137032

theorem smallest_number (a b c d : ℝ) (ha : a = -2) (hb : b = 0) (hc : c = 1/2) (hd : d = 2) :
  a ≤ b ∧ a ≤ c ∧ a ≤ d :=
by sorry

end smallest_number_l1370_137032


namespace smith_family_laundry_l1370_137050

/-- The number of bath towels that can fit in one load of laundry for the Smith family. -/
def towels_per_load (kylie_towels : ℕ) (daughters_towels : ℕ) (husband_towels : ℕ) (total_loads : ℕ) : ℕ :=
  (kylie_towels + daughters_towels + husband_towels) / total_loads

/-- Theorem stating that the washing machine can fit 4 bath towels in one load of laundry. -/
theorem smith_family_laundry :
  towels_per_load 3 6 3 3 = 4 :=
by
  sorry

end smith_family_laundry_l1370_137050


namespace olivias_wallet_l1370_137044

/-- Calculates the remaining money in Olivia's wallet after shopping --/
def remaining_money (initial : ℕ) (supermarket : ℕ) (showroom : ℕ) : ℕ :=
  initial - supermarket - showroom

/-- Theorem stating that Olivia has 26 dollars left after shopping --/
theorem olivias_wallet : remaining_money 106 31 49 = 26 := by
  sorry

end olivias_wallet_l1370_137044


namespace train_speed_calculation_l1370_137048

/-- Given a train of length 140 meters passing a platform of length 260 meters in 23.998080153587715 seconds,
    prove that the speed of the train is 60.0048 kilometers per hour. -/
theorem train_speed_calculation (train_length platform_length time_to_pass : ℝ)
    (h1 : train_length = 140)
    (h2 : platform_length = 260)
    (h3 : time_to_pass = 23.998080153587715) :
    (train_length + platform_length) / time_to_pass * 3.6 = 60.0048 := by
  sorry

end train_speed_calculation_l1370_137048


namespace parabola_circle_intersection_l1370_137003

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  focus : ℝ × ℝ
  h_p_pos : p > 0

/-- Point on a parabola -/
structure PointOnParabola (C : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y^2 = 2 * C.p * x

/-- Circle structure -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Theorem: For a parabola y^2 = 2px (p > 0) with focus F, and a point M(x_0, 2√2) on the parabola,
    if a circle with center M is tangent to the y-axis and intersects MF at A such that |MA| / |AF| = 2,
    then p = 2 -/
theorem parabola_circle_intersection (C : Parabola) (M : PointOnParabola C)
  (circ : Circle) (A : ℝ × ℝ) :
  M.y = 2 * Real.sqrt 2 →
  circ.center = (M.x, M.y) →
  circ.radius = M.x →
  A.1 = M.x - C.p →
  (M.x - A.1) / A.1 = 2 →
  C.p = 2 := by sorry

end parabola_circle_intersection_l1370_137003


namespace base_4_of_185_l1370_137094

/-- Converts a base 4 number to base 10 --/
def base4ToBase10 (a b c d : ℕ) : ℕ :=
  a * (4^3) + b * (4^2) + c * (4^1) + d * (4^0)

/-- The base 4 representation of 185 (base 10) is 2321 --/
theorem base_4_of_185 : base4ToBase10 2 3 2 1 = 185 := by
  sorry

end base_4_of_185_l1370_137094


namespace log_inequality_l1370_137069

theorem log_inequality (a b : ℝ) (ha : a = Real.log 2 / Real.log 3) (hb : b = Real.log 3 / Real.log 2) :
  Real.log a < (1/2) ^ b := by
  sorry

end log_inequality_l1370_137069


namespace books_returned_count_l1370_137063

/-- Represents the number of books Mary has at different stages --/
structure BookCount where
  initial : Nat
  after_first_return : Nat
  after_second_checkout : Nat
  final : Nat

/-- Represents Mary's library transactions --/
def library_transactions (x : Nat) : BookCount :=
  { initial := 5,
    after_first_return := 5 - x + 5,
    after_second_checkout := 5 - x + 5 - 2 + 7,
    final := 12 }

/-- Theorem stating the number of books Mary returned --/
theorem books_returned_count : ∃ x : Nat, 
  (library_transactions x).final = 12 ∧ x = 3 := by
  sorry

end books_returned_count_l1370_137063


namespace cylinder_height_l1370_137067

theorem cylinder_height (r h : ℝ) : 
  r = 4 →
  2 * Real.pi * r^2 + 2 * Real.pi * r * h = 40 * Real.pi →
  h = 1 := by
sorry

end cylinder_height_l1370_137067


namespace fraction_simplification_l1370_137043

theorem fraction_simplification (a b : ℝ) (h : a ≠ b) : (1/2 * a) / (1/2 * b) = a / b := by
  sorry

end fraction_simplification_l1370_137043


namespace circle_A_tangent_to_x_axis_l1370_137025

def circle_A_center : ℝ × ℝ := (-4, -3)
def circle_A_radius : ℝ := 3

theorem circle_A_tangent_to_x_axis :
  let (x, y) := circle_A_center
  abs y = circle_A_radius := by sorry

end circle_A_tangent_to_x_axis_l1370_137025


namespace village_population_l1370_137007

theorem village_population (initial_population : ℕ) 
  (death_rate : ℚ) (leaving_rate : ℚ) (final_population : ℕ) : 
  initial_population = 4400 →
  death_rate = 5 / 100 →
  leaving_rate = 15 / 100 →
  final_population = 
    (initial_population - 
      (initial_population * death_rate).floor - 
      ((initial_population - (initial_population * death_rate).floor) * leaving_rate).floor) →
  final_population = 3553 := by
sorry

end village_population_l1370_137007


namespace jade_savings_l1370_137009

/-- Calculates Jade's monthly savings based on her income and expenses --/
def calculate_savings (
  monthly_income : ℝ)
  (contribution_401k_rate : ℝ)
  (tax_deduction_rate : ℝ)
  (living_expenses_rate : ℝ)
  (insurance_rate : ℝ)
  (transportation_rate : ℝ)
  (utilities_rate : ℝ) : ℝ :=
  let contribution_401k := monthly_income * contribution_401k_rate
  let tax_deduction := monthly_income * tax_deduction_rate
  let post_deduction_income := monthly_income - contribution_401k - tax_deduction
  let total_expenses := post_deduction_income * (living_expenses_rate + insurance_rate + transportation_rate + utilities_rate)
  post_deduction_income - total_expenses

/-- Theorem stating Jade's monthly savings --/
theorem jade_savings :
  calculate_savings 2800 0.08 0.10 0.55 0.20 0.12 0.08 = 114.80 := by
  sorry


end jade_savings_l1370_137009


namespace negative_sqrt_six_squared_equals_six_l1370_137008

theorem negative_sqrt_six_squared_equals_six : (-Real.sqrt 6)^2 = 6 := by
  sorry

end negative_sqrt_six_squared_equals_six_l1370_137008


namespace triangle_properties_l1370_137052

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The perimeter of the triangle is √2 + 1 -/
def perimeter_condition (t : Triangle) : Prop :=
  t.a + t.b + t.c = Real.sqrt 2 + 1

/-- The sum of sines condition -/
def sine_sum_condition (t : Triangle) : Prop :=
  Real.sin t.A + Real.sin t.B = Real.sqrt 2 * Real.sin t.C

/-- The area of the triangle is (1/6) * sin C -/
def area_condition (t : Triangle) : Prop :=
  (1/2) * t.a * t.b * Real.sin t.C = (1/6) * Real.sin t.C

theorem triangle_properties (t : Triangle) 
  (h_perimeter : perimeter_condition t)
  (h_sine_sum : sine_sum_condition t)
  (h_area : area_condition t) :
  t.c = 1 ∧ t.C = π/3 := by
  sorry

end triangle_properties_l1370_137052


namespace specialNumberCount_is_70_l1370_137029

/-- The count of numbers between 200 and 899 (inclusive) with three different digits 
    that can be arranged in either strictly increasing or strictly decreasing order -/
def specialNumberCount : ℕ :=
  let lowerBound := 200
  let upperBound := 899
  let digitSet := {2, 3, 4, 5, 6, 7, 8}
  2 * (Finset.card digitSet).choose 3

theorem specialNumberCount_is_70 : specialNumberCount = 70 := by
  sorry

end specialNumberCount_is_70_l1370_137029


namespace parabola_properties_l1370_137046

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 - 2*x - 3

-- Define the conditions
theorem parabola_properties :
  (parabola (-1) = 0) ∧
  (parabola 3 = 0) ∧
  (parabola 0 = -3) ∧
  (∃ (a b c : ℝ), ∀ x, parabola x = a * x^2 + b * x + c) ∧
  (let vertex := (1, -4);
   parabola vertex.1 = vertex.2 ∧
   ∀ x, parabola x ≥ parabola vertex.1) ∧
  (∀ x₁ x₂ y₁ y₂, 
    x₁ < x₂ → x₂ < 1 → 
    parabola x₁ = y₁ → parabola x₂ = y₂ → 
    y₁ < y₂) :=
by sorry

end parabola_properties_l1370_137046


namespace max_sum_and_reciprocal_l1370_137047

theorem max_sum_and_reciprocal (nums : Finset ℝ) (x : ℝ) :
  (Finset.card nums = 2023) →
  (∀ y ∈ nums, y > 0) →
  (x ∈ nums) →
  (Finset.sum nums id = 2024) →
  (Finset.sum nums (λ y => 1 / y) = 2024) →
  (x + 1 / x ≤ 4096094 / 2024) :=
by sorry

end max_sum_and_reciprocal_l1370_137047


namespace salary_solution_l1370_137081

def salary_problem (salary : ℝ) : Prop :=
  let food_expense := (1 / 5 : ℝ) * salary
  let rent_expense := (1 / 10 : ℝ) * salary
  let clothes_expense := (3 / 5 : ℝ) * salary
  let remaining := salary - (food_expense + rent_expense + clothes_expense)
  remaining = 19000

theorem salary_solution :
  ∃ (salary : ℝ), salary_problem salary ∧ salary = 190000 := by
  sorry

end salary_solution_l1370_137081


namespace binary_1101_to_base5_l1370_137055

-- Define a function to convert binary to decimal
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

-- Define a function to convert decimal to base-5
def decimal_to_base5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) :=
    if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
  aux n []

-- Theorem statement
theorem binary_1101_to_base5 :
  decimal_to_base5 (binary_to_decimal [true, false, true, true]) = [2, 3] := by
  sorry

end binary_1101_to_base5_l1370_137055


namespace election_majority_l1370_137082

theorem election_majority (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 800 →
  winning_percentage = 70 / 100 →
  (winning_percentage * total_votes : ℚ) - ((1 - winning_percentage) * total_votes : ℚ) = 320 := by
  sorry

end election_majority_l1370_137082


namespace probability_one_defective_is_half_l1370_137056

/-- Represents the total number of items -/
def total_items : Nat := 4

/-- Represents the number of genuine items -/
def genuine_items : Nat := 3

/-- Represents the number of defective items -/
def defective_items : Nat := 1

/-- Represents the number of items to be selected -/
def items_to_select : Nat := 2

/-- Calculates the number of ways to select k items from n items -/
def combinations (n k : Nat) : Nat := sorry

/-- Calculates the probability of selecting exactly one defective item -/
def probability_one_defective : Rat :=
  (combinations defective_items 1 * combinations genuine_items (items_to_select - 1)) /
  (combinations total_items items_to_select)

/-- Theorem stating that the probability of selecting exactly one defective item is 1/2 -/
theorem probability_one_defective_is_half :
  probability_one_defective = 1 / 2 := by sorry

end probability_one_defective_is_half_l1370_137056


namespace sam_read_100_pages_l1370_137091

def minimum_assigned : ℕ := 25

def harrison_pages (minimum : ℕ) : ℕ := minimum + 10

def pam_pages (harrison : ℕ) : ℕ := harrison + 15

def sam_pages (pam : ℕ) : ℕ := 2 * pam

theorem sam_read_100_pages :
  sam_pages (pam_pages (harrison_pages minimum_assigned)) = 100 := by
  sorry

end sam_read_100_pages_l1370_137091


namespace problem1_problem2_l1370_137058

-- Problem 1
def M : ℝ × ℝ := (3, 2)
def N : ℝ × ℝ := (4, -1)

def is_right_angle (A B C : ℝ × ℝ) : Prop :=
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C.1 - A.1, C.2 - A.2)
  AB.1 * AC.1 + AB.2 * AC.2 = 0

def on_x_axis (P : ℝ × ℝ) : Prop := P.2 = 0

theorem problem1 (P : ℝ × ℝ) :
  on_x_axis P ∧ is_right_angle M P N → P = (2, 0) ∨ P = (5, 0) :=
sorry

-- Problem 2
def A : ℝ × ℝ := (7, -4)
def B : ℝ × ℝ := (-5, 6)

def perpendicular_bisector (A B : ℝ × ℝ) (x y : ℝ) : Prop :=
  6 * x - 5 * y - 1 = 0

theorem problem2 :
  perpendicular_bisector A B = λ x y => 6 * x - 5 * y - 1 = 0 :=
sorry

end problem1_problem2_l1370_137058


namespace honor_students_count_l1370_137075

theorem honor_students_count 
  (total_students : ℕ) 
  (girls : ℕ) 
  (boys : ℕ) 
  (honor_girls : ℕ) 
  (honor_boys : ℕ) :
  total_students < 30 →
  total_students = girls + boys →
  (honor_girls : ℚ) / girls = 3 / 13 →
  (honor_boys : ℚ) / boys = 4 / 11 →
  honor_girls + honor_boys = 7 :=
by sorry

end honor_students_count_l1370_137075


namespace hamlet_47_impossible_hamlet_41_possible_hamlet_59_possible_hamlet_61_possible_hamlet_66_possible_l1370_137024

/-- Represents the total number of animals and people in Hamlet -/
def hamlet_total (h c : ℕ) : ℕ := 13 * h + 5 * c

/-- Theorem stating that 47 cannot be expressed as a hamlet total -/
theorem hamlet_47_impossible : ¬ ∃ (h c : ℕ), hamlet_total h c = 47 := by sorry

/-- Theorem stating that 41 can be expressed as a hamlet total -/
theorem hamlet_41_possible : ∃ (h c : ℕ), hamlet_total h c = 41 := by sorry

/-- Theorem stating that 59 can be expressed as a hamlet total -/
theorem hamlet_59_possible : ∃ (h c : ℕ), hamlet_total h c = 59 := by sorry

/-- Theorem stating that 61 can be expressed as a hamlet total -/
theorem hamlet_61_possible : ∃ (h c : ℕ), hamlet_total h c = 61 := by sorry

/-- Theorem stating that 66 can be expressed as a hamlet total -/
theorem hamlet_66_possible : ∃ (h c : ℕ), hamlet_total h c = 66 := by sorry

end hamlet_47_impossible_hamlet_41_possible_hamlet_59_possible_hamlet_61_possible_hamlet_66_possible_l1370_137024


namespace initial_speed_is_three_l1370_137080

/-- Represents the scenario of two pedestrians walking towards each other --/
structure PedestrianScenario where
  totalDistance : ℝ
  delayDistance : ℝ
  delayTime : ℝ
  meetingDistanceAfterDelay : ℝ
  speedIncrease : ℝ

/-- Calculates the initial speed of the pedestrians --/
def initialSpeed (scenario : PedestrianScenario) : ℝ :=
  sorry

/-- Theorem stating that the initial speed is 3 km/h for the given scenario --/
theorem initial_speed_is_three 
  (scenario : PedestrianScenario) 
  (h1 : scenario.totalDistance = 28)
  (h2 : scenario.delayDistance = 9)
  (h3 : scenario.delayTime = 1)
  (h4 : scenario.meetingDistanceAfterDelay = 4)
  (h5 : scenario.speedIncrease = 1) :
  initialSpeed scenario = 3 := by
  sorry

end initial_speed_is_three_l1370_137080


namespace other_root_of_quadratic_l1370_137059

theorem other_root_of_quadratic (m : ℝ) : 
  (1 : ℝ) ^ 2 - 3 * (1 : ℝ) + m = 0 → 
  ∃ (x : ℝ), x ≠ 1 ∧ x ^ 2 - 3 * x + m = 0 ∧ x = 2 := by
sorry

end other_root_of_quadratic_l1370_137059


namespace ella_sold_200_apples_l1370_137066

/-- The number of apples Ella sold -/
def apples_sold (bags_of_20 bags_of_25 apples_per_bag_20 apples_per_bag_25 apples_left : ℕ) : ℕ :=
  bags_of_20 * apples_per_bag_20 + bags_of_25 * apples_per_bag_25 - apples_left

/-- Theorem stating that Ella sold 200 apples -/
theorem ella_sold_200_apples :
  apples_sold 4 6 20 25 30 = 200 := by
  sorry

end ella_sold_200_apples_l1370_137066


namespace max_value_inequality_l1370_137031

theorem max_value_inequality (x y : ℝ) (h : x * y > 0) :
  (x / (x + y)) + (2 * y / (x + 2 * y)) ≤ 4 - 2 * Real.sqrt 2 := by
  sorry

end max_value_inequality_l1370_137031


namespace triangle_ratio_theorem_l1370_137093

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_ratio_theorem (t : Triangle) 
  (h1 : t.A = 30 * π / 180)  -- A = 30°
  (h2 : t.a = Real.sqrt 3)   -- a = √3
  (h3 : t.A + t.B + t.C = π) -- Sum of angles in a triangle is π
  (h4 : t.a / Real.sin t.A = t.b / Real.sin t.B) -- Law of Sines
  (h5 : t.b / Real.sin t.B = t.c / Real.sin t.C) -- Law of Sines
  : (t.a + t.b + t.c) / (Real.sin t.A + Real.sin t.B + Real.sin t.C) = 2 * Real.sqrt 3 := by
  sorry


end triangle_ratio_theorem_l1370_137093


namespace line_point_a_value_l1370_137090

theorem line_point_a_value (k : ℝ) (a : ℝ) :
  k = 0.75 →
  5 = k * a + 1 →
  a = 16/3 := by sorry

end line_point_a_value_l1370_137090


namespace work_rate_ab_together_days_for_ab_together_l1370_137064

-- Define the work rates for workers a, b, and c
variable (A B C : ℝ)

-- Define the conditions
variable (h1 : A + B + C = 1 / 5)  -- a, b, and c together finish in 5 days
variable (h2 : C = 1 / 7.5)        -- c alone finishes in 7.5 days

-- Theorem to prove
theorem work_rate_ab_together : A + B = 1 / 15 := by
  sorry

-- Theorem to prove the final result
theorem days_for_ab_together : 1 / (A + B) = 15 := by
  sorry

end work_rate_ab_together_days_for_ab_together_l1370_137064


namespace restaurant_bill_tax_calculation_l1370_137039

/-- Calculates the tax amount for a restaurant bill given specific conditions. -/
theorem restaurant_bill_tax_calculation
  (cheeseburger_price : ℚ)
  (milkshake_price : ℚ)
  (coke_price : ℚ)
  (fries_price : ℚ)
  (cookie_price : ℚ)
  (toby_initial_amount : ℚ)
  (toby_change : ℚ)
  (h1 : cheeseburger_price = 365/100)
  (h2 : milkshake_price = 2)
  (h3 : coke_price = 1)
  (h4 : fries_price = 4)
  (h5 : cookie_price = 1/2)
  (h6 : toby_initial_amount = 15)
  (h7 : toby_change = 7) :
  let subtotal := 2 * cheeseburger_price + milkshake_price + coke_price + fries_price + 3 * cookie_price
  let toby_spent := toby_initial_amount - toby_change
  let total_paid := 2 * toby_spent
  let tax := total_paid - subtotal
  tax = 1/5 :=
by sorry

end restaurant_bill_tax_calculation_l1370_137039


namespace initial_average_height_calculation_l1370_137079

theorem initial_average_height_calculation (n : ℕ) (error : ℝ) (actual_avg : ℝ) :
  n = 35 ∧ error = 60 ∧ actual_avg = 178 →
  (n * actual_avg + error) / n = 179.71 := by
  sorry

end initial_average_height_calculation_l1370_137079


namespace power_of_two_triples_l1370_137078

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

def valid_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  is_power_of_two (a * b - c) ∧
  is_power_of_two (b * c - a) ∧
  is_power_of_two (c * a - b)

theorem power_of_two_triples :
  ∀ a b c : ℕ, valid_triple a b c ↔
    (a = 2 ∧ b = 2 ∧ c = 2) ∨
    (a = 2 ∧ b = 2 ∧ c = 3) ∨
    (a = 3 ∧ b = 5 ∧ c = 7) ∨
    (a = 2 ∧ b = 6 ∧ c = 11) ∨
    (a = 2 ∧ b = 3 ∧ c = 2) ∨
    (a = 2 ∧ b = 11 ∧ c = 6) ∨
    (a = 3 ∧ b = 7 ∧ c = 5) ∨
    (a = 5 ∧ b = 7 ∧ c = 3) ∨
    (a = 5 ∧ b = 3 ∧ c = 7) ∨
    (a = 6 ∧ b = 11 ∧ c = 2) ∨
    (a = 7 ∧ b = 3 ∧ c = 5) ∨
    (a = 7 ∧ b = 5 ∧ c = 3) ∨
    (a = 11 ∧ b = 2 ∧ c = 6) :=
by sorry

end power_of_two_triples_l1370_137078


namespace sum_always_six_digits_l1370_137060

def first_number : Nat := 98765

def second_number (C : Nat) : Nat := C * 1000 + 433

def third_number (D : Nat) : Nat := D * 100 + 22

def is_nonzero_digit (n : Nat) : Prop := 1 ≤ n ∧ n ≤ 9

theorem sum_always_six_digits (C D : Nat) 
  (hC : is_nonzero_digit C) (hD : is_nonzero_digit D) : 
  ∃ (n : Nat), 100000 ≤ first_number + second_number C + third_number D ∧ 
               first_number + second_number C + third_number D < 1000000 :=
sorry

end sum_always_six_digits_l1370_137060


namespace absolute_value_equation_solutions_l1370_137030

theorem absolute_value_equation_solutions :
  ∃! (s : Finset ℝ), s.card = 2 ∧ 
  (∀ x : ℝ, x ∈ s ↔ |x - 1| = |x - 2| + |x - 3| + |x - 4|) ∧
  (3 ∈ s ∧ 4 ∈ s) := by
  sorry

end absolute_value_equation_solutions_l1370_137030


namespace problem_statement_l1370_137038

noncomputable def f (x : ℝ) : ℝ := (1 - 2^x) / (1 + 2^x)

noncomputable def g (x : ℝ) : ℝ := Real.log (Real.sqrt (x^2 + 1) - x)

noncomputable def F (x : ℝ) : ℝ := f x + g x

theorem problem_statement :
  (∀ x, f (-x) = -f x) ∧
  (∀ x, g (-x) = -g x) ∧
  (∃ M m, (∀ x ∈ Set.Icc (-1) 1, F x ≤ M ∧ m ≤ F x) ∧ M + m = 0) ∧
  (Set.Ioi 1 = {a | F (2*a) + F (-1-a) < 0}) :=
sorry

end problem_statement_l1370_137038


namespace remainder_97_power_51_mod_100_l1370_137083

theorem remainder_97_power_51_mod_100 : 97^51 % 100 = 39 := by
  sorry

end remainder_97_power_51_mod_100_l1370_137083


namespace road_trip_total_distance_l1370_137095

/-- Represents the road trip with given conditions -/
def RoadTrip (x : ℝ) : Prop :=
  let first_leg := x
  let second_leg := 2 * x
  let third_leg := 40
  let final_leg := 2 * (first_leg + second_leg + third_leg)
  (third_leg = x / 2) ∧
  (first_leg + second_leg + third_leg + final_leg = 840)

/-- Theorem stating the total distance of the road trip -/
theorem road_trip_total_distance : ∃ x : ℝ, RoadTrip x :=
  sorry

end road_trip_total_distance_l1370_137095


namespace yellow_flags_in_200_l1370_137027

/-- Represents the number of flags in one complete pattern -/
def pattern_length : ℕ := 9

/-- Represents the number of yellow flags in one complete pattern -/
def yellow_per_pattern : ℕ := 3

/-- Represents the total number of flags we're considering -/
def total_flags : ℕ := 200

/-- Calculates the number of yellow flags in the given sequence -/
def yellow_flags (n : ℕ) : ℕ :=
  (n / pattern_length) * yellow_per_pattern + min yellow_per_pattern (n % pattern_length)

theorem yellow_flags_in_200 : yellow_flags total_flags = 67 := by
  sorry

end yellow_flags_in_200_l1370_137027


namespace right_side_exponent_l1370_137005

theorem right_side_exponent (s : ℝ) : 
  (2^16 : ℝ) * (25^s) = 5 * (10^16) → 16 = 16 := by sorry

end right_side_exponent_l1370_137005


namespace absolute_value_sum_zero_l1370_137041

theorem absolute_value_sum_zero (a b : ℝ) :
  |a - 2| + |b + 3| = 0 → b^a = 9 := by sorry

end absolute_value_sum_zero_l1370_137041


namespace shoe_price_calculation_l1370_137012

theorem shoe_price_calculation (initial_price : ℝ) 
  (price_increase_percent : ℝ) (discount_percent : ℝ) (tax_percent : ℝ) : 
  initial_price = 50 ∧ 
  price_increase_percent = 20 ∧ 
  discount_percent = 15 ∧ 
  tax_percent = 5 → 
  initial_price * (1 + price_increase_percent / 100) * 
  (1 - discount_percent / 100) * (1 + tax_percent / 100) = 53.55 := by
sorry

end shoe_price_calculation_l1370_137012


namespace max_value_theorem_l1370_137097

theorem max_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (heq : a * (a + b + c) = b * c) : 
  a / (b + c) ≤ (Real.sqrt 2 - 1) / 2 ∧ 
  ∃ a₀ b₀ c₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ 
    a₀ * (a₀ + b₀ + c₀) = b₀ * c₀ ∧ 
    a₀ / (b₀ + c₀) = (Real.sqrt 2 - 1) / 2 := by
  sorry

end max_value_theorem_l1370_137097


namespace friday_work_proof_l1370_137099

/-- The time Mr. Willson worked on Friday in minutes -/
def friday_work_minutes : ℚ := 75

/-- Converts hours to minutes -/
def hours_to_minutes (hours : ℚ) : ℚ := hours * 60

theorem friday_work_proof (monday : ℚ) (tuesday : ℚ) (wednesday : ℚ) (thursday : ℚ) 
  (h_monday : monday = 3/4)
  (h_tuesday : tuesday = 1/2)
  (h_wednesday : wednesday = 2/3)
  (h_thursday : thursday = 5/6)
  (h_total : monday + tuesday + wednesday + thursday + friday_work_minutes / 60 = 4) :
  friday_work_minutes = 75 := by
  sorry


end friday_work_proof_l1370_137099


namespace triangle_perimeter_l1370_137073

theorem triangle_perimeter (a b c : ℝ) (h_right_angle : a^2 + b^2 = c^2) 
  (h_hypotenuse : c = 15) (h_side_ratio : a = b / 2) : 
  a + b + c = 15 + 9 * Real.sqrt 5 := by
  sorry

end triangle_perimeter_l1370_137073


namespace quadratic_equations_solutions_l1370_137042

theorem quadratic_equations_solutions :
  (∃ x1 x2 : ℝ, (2 * (x1 - 1)^2 = 18 ∧ x1 = 4) ∧ (2 * (x2 - 1)^2 = 18 ∧ x2 = -2)) ∧
  (∃ y1 y2 : ℝ, (y1^2 - 4*y1 - 3 = 0 ∧ y1 = 2 + Real.sqrt 7) ∧ (y2^2 - 4*y2 - 3 = 0 ∧ y2 = 2 - Real.sqrt 7)) :=
by sorry

end quadratic_equations_solutions_l1370_137042


namespace functional_equation_properties_l1370_137074

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * f y

theorem functional_equation_properties (f : ℝ → ℝ) 
  (h_eq : FunctionalEquation f) (h_nonzero : f 0 ≠ 0) : 
  (f 0 = 1) ∧ (∀ x : ℝ, f (-x) = f x) := by
  sorry

end functional_equation_properties_l1370_137074


namespace solve_equation_l1370_137021

theorem solve_equation (y : ℝ) (x : ℝ) (h1 : 9^y = x^12) (h2 : y = 6) : x = 3 := by
  sorry

end solve_equation_l1370_137021


namespace fraction_states_1800_1809_l1370_137040

/-- The number of states that joined the union from 1800 to 1809 -/
def states_1800_1809 : ℕ := 5

/-- The total number of states considered (first 30 states) -/
def total_states : ℕ := 30

/-- The fraction of states that joined from 1800 to 1809 -/
def fraction_1800_1809 : ℚ := states_1800_1809 / total_states

theorem fraction_states_1800_1809 : fraction_1800_1809 = 1 / 6 := by
  sorry

end fraction_states_1800_1809_l1370_137040


namespace simplify_fraction_division_l1370_137034

theorem simplify_fraction_division (x : ℝ) 
  (h1 : x ≠ 4) (h2 : x ≠ 2) (h3 : x ≠ 5) (h4 : x ≠ 3) (h5 : x ≠ 1) : 
  (x^2 - 4*x + 3) / (x^2 - 6*x + 8) / ((x^2 - 6*x + 5) / (x^2 - 8*x + 15)) = 1 / ((x - 4) * (x - 2)) :=
by
  sorry

#check simplify_fraction_division

end simplify_fraction_division_l1370_137034
