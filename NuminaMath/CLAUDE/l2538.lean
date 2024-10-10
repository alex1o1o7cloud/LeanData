import Mathlib

namespace triangle_angle_bounds_l2538_253832

/-- Given a positive real number R, if R, 1, and R+1/2 form a triangle with θ as the angle between R and R+1/2, then 1 < 2Rθ < π. -/
theorem triangle_angle_bounds (R : ℝ) (θ : ℝ) (h_pos : R > 0) 
  (h_triangle : R + 1 > R + 1/2 ∧ R + (R + 1/2) > 1 ∧ 1 + (R + 1/2) > R) 
  (h_angle : θ = Real.arccos ((R^2 + (R + 1/2)^2 - 1) / (2 * R * (R + 1/2)))) :
  1 < 2 * R * θ ∧ 2 * R * θ < π :=
sorry

end triangle_angle_bounds_l2538_253832


namespace exists_divisible_by_1988_l2538_253830

def f (x : ℤ) : ℤ := 3 * x + 2

def f_iter (k : ℕ) : ℤ → ℤ :=
  match k with
  | 0 => id
  | n + 1 => f ∘ (f_iter n)

theorem exists_divisible_by_1988 :
  ∃ m : ℕ+, (1988 : ℤ) ∣ (f_iter 100 m.val) := by
  sorry

end exists_divisible_by_1988_l2538_253830


namespace parabola_equation_l2538_253854

-- Define the parabola
structure Parabola where
  equation : ℝ → ℝ → Prop

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the problem
theorem parabola_equation (p : Parabola) (t : Triangle) (F : Point) :
  -- The vertex of the parabola is at the origin
  p.equation 0 0 ∧
  -- The focus of the parabola is on the x-axis
  F.y = 0 ∧
  -- The three vertices of triangle ABC lie on the parabola
  p.equation t.A.x t.A.y ∧ p.equation t.B.x t.B.y ∧ p.equation t.C.x t.C.y ∧
  -- The centroid of triangle ABC is the focus F of the parabola
  F.x = (t.A.x + t.B.x + t.C.x) / 3 ∧ F.y = (t.A.y + t.B.y + t.C.y) / 3 ∧
  -- The equation of the line where side BC lies is 4x + y - 20 = 0
  4 * t.B.x + t.B.y = 20 ∧ 4 * t.C.x + t.C.y = 20 →
  -- The equation of the parabola is y² = 16x
  ∀ x y, p.equation x y ↔ y^2 = 16*x :=
by sorry


end parabola_equation_l2538_253854


namespace final_sum_after_operations_l2538_253840

theorem final_sum_after_operations (a b S : ℝ) (h : a + b = S) :
  3 * (a + 5) + 3 * (b + 5) = 3 * S + 30 := by sorry

end final_sum_after_operations_l2538_253840


namespace factorization_proof_l2538_253880

theorem factorization_proof (x y a b : ℝ) : 
  (x * y^2 - 2 * x * y = x * y * (y - 2)) ∧ 
  (6 * a * (x + y) - 5 * b * (x + y) = (x + y) * (6 * a - 5 * b)) :=
by sorry

end factorization_proof_l2538_253880


namespace triangle_c_range_and_perimeter_l2538_253855

-- Define the triangle sides and conditions
def triangle_sides (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_c_range_and_perimeter
  (a b c : ℝ)
  (h_sides : triangle_sides a b c)
  (h_sum : a + b = 3 * c - 2)
  (h_diff : a - b = 2 * c - 6) :
  (1 < c ∧ c < 6) ∧
  (a + b + c = 18 → c = 5) := by
sorry


end triangle_c_range_and_perimeter_l2538_253855


namespace quadratic_equation_roots_l2538_253859

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ a / b = 3 ∧ 
   (∀ x : ℝ, x^2 - 4*x + m = 0 ↔ (x = a ∨ x = b))) → 
  m = 3 := by
sorry

end quadratic_equation_roots_l2538_253859


namespace angle_measure_proof_l2538_253862

theorem angle_measure_proof (x : ℝ) : 
  (x + (3 * x + 10) = 90) → x = 20 := by
  sorry

end angle_measure_proof_l2538_253862


namespace change5_descent_l2538_253863

/-- Proof of Chang'e-5 lunar probe descent calculations -/
theorem change5_descent (initial_distance initial_speed final_speed time : ℝ) 
  (h1 : initial_distance = 1800)
  (h2 : initial_speed = 1800)
  (h3 : final_speed = 0)
  (h4 : time = 12 * 60) :
  let v := (0 - initial_distance) / time
  let a := (final_speed - initial_speed) / time
  v = -5/2 ∧ a = -5/2 := by sorry

end change5_descent_l2538_253863


namespace equation_solution_l2538_253801

theorem equation_solution : ∃ x : ℝ, (2 / (3 * x) = 1 / (x + 2)) ∧ x = 4 := by
  sorry

end equation_solution_l2538_253801


namespace sqrt_equation_solution_l2538_253816

theorem sqrt_equation_solution :
  ∃ x : ℝ, (Real.sqrt (2 * x + 12) = 8) ∧ (x = 26) := by
  sorry

end sqrt_equation_solution_l2538_253816


namespace initial_amount_of_liquid_A_l2538_253876

/-- Given a can with a mixture of liquids A and B, this theorem proves the initial amount of liquid A
    based on the given conditions and ratios. -/
theorem initial_amount_of_liquid_A (x : ℝ) : 
  -- Initial ratio of A to B is 7:5
  7 * x / (5 * x) = 7 / 5 →
  -- After removing 9 liters and adding B to make new ratio 7:9
  (7 * x - 9 * (7 / 12)) / (5 * x - 9 * (5 / 12) + 9) = 7 / 9 →
  -- The initial amount of liquid A was 21 liters
  7 * x = 21 := by
  sorry

end initial_amount_of_liquid_A_l2538_253876


namespace allocation_count_l2538_253867

/-- The number of factories available for allocation --/
def num_factories : ℕ := 4

/-- The number of students to be allocated --/
def num_students : ℕ := 3

/-- The total number of possible allocations without restrictions --/
def total_allocations : ℕ := num_factories ^ num_students

/-- The number of allocations where no student goes to Factory A --/
def allocations_without_A : ℕ := (num_factories - 1) ^ num_students

/-- The number of valid allocations where at least one student goes to Factory A --/
def valid_allocations : ℕ := total_allocations - allocations_without_A

theorem allocation_count : valid_allocations = 37 := by
  sorry

end allocation_count_l2538_253867


namespace jessie_dimes_l2538_253841

/-- Represents the contents of Jessie's piggy bank -/
structure PiggyBank where
  dimes : ℕ
  quarters : ℕ
  total_cents : ℕ
  dime_quarter_difference : ℕ

/-- The piggy bank satisfies the given conditions -/
def valid_piggy_bank (pb : PiggyBank) : Prop :=
  pb.dimes = pb.quarters + pb.dime_quarter_difference ∧
  pb.total_cents = 10 * pb.dimes + 25 * pb.quarters ∧
  pb.dime_quarter_difference = 10 ∧
  pb.total_cents = 580

/-- The theorem stating that Jessie has 23 dimes -/
theorem jessie_dimes (pb : PiggyBank) (h : valid_piggy_bank pb) : pb.dimes = 23 := by
  sorry

end jessie_dimes_l2538_253841


namespace bicycle_problem_l2538_253812

/-- Represents the position of a person at a given time -/
structure Position where
  location : ℝ
  time : ℝ

/-- Represents a person traveling at a constant speed -/
structure Traveler where
  speed : ℝ
  startPosition : ℝ

def position (t : Traveler) (time : ℝ) : Position :=
  { location := t.startPosition + t.speed * time, time := time }

theorem bicycle_problem 
  (misha sasha vanya : Traveler)
  (h1 : misha.startPosition = 0 ∧ sasha.startPosition = 0 ∧ vanya.startPosition > 0)
  (h2 : misha.speed > 0 ∧ sasha.speed > 0 ∧ vanya.speed < 0)
  (h3 : (position sasha 1).location = ((position misha 1).location + (position vanya 1).location) / 2)
  (h4 : (position vanya 1.5).location = ((position misha 1.5).location + (position sasha 1.5).location) / 2) :
  (position misha 3).location = ((position sasha 3).location + (position vanya 3).location) / 2 := by
  sorry


end bicycle_problem_l2538_253812


namespace arc_length_calculation_l2538_253861

theorem arc_length_calculation (r : ℝ) (θ : ℝ) (h1 : r = 3) (h2 : θ = π / 7) :
  r * θ = 3 * π / 7 := by
  sorry

end arc_length_calculation_l2538_253861


namespace product_ab_is_zero_l2538_253873

theorem product_ab_is_zero (a b : ℝ) 
  (sum_eq : a + b = 4) 
  (sum_cubes_eq : a^3 + b^3 = 64) : 
  a * b = 0 := by
sorry

end product_ab_is_zero_l2538_253873


namespace triangle_perimeter_example_l2538_253839

/-- A triangle with sides a, b, and c is valid if the sum of any two sides is greater than the third side. -/
def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The perimeter of a triangle is the sum of its sides. -/
def triangle_perimeter (a b c : ℝ) : ℝ :=
  a + b + c

/-- Theorem: The perimeter of a triangle with sides 12, 15, and 9 is 36. -/
theorem triangle_perimeter_example : 
  is_valid_triangle 12 15 9 → triangle_perimeter 12 15 9 = 36 := by
  sorry

end triangle_perimeter_example_l2538_253839


namespace expand_expression_l2538_253809

theorem expand_expression (y : ℝ) : 5 * (3 * y^3 + 4 * y^2 - 7 * y + 2) = 15 * y^3 + 20 * y^2 - 35 * y + 10 := by
  sorry

end expand_expression_l2538_253809


namespace count_distinct_walls_l2538_253872

/-- The number of distinct walls that can be built with n identical cubes -/
def distinct_walls (n : ℕ+) : ℕ :=
  2^(n.val - 1)

/-- Theorem stating that the number of distinct walls with n cubes is 2^(n-1) -/
theorem count_distinct_walls (n : ℕ+) :
  distinct_walls n = 2^(n.val - 1) := by
  sorry

end count_distinct_walls_l2538_253872


namespace count_numbers_equals_fifteen_l2538_253878

/-- The set of digits available on the cards -/
def digits : Finset Nat := {1, 2, 3}

/-- The function to count valid numbers formed from the digits -/
def count_numbers (digits : Finset Nat) : Nat :=
  (digits.card) +  -- one-digit numbers
  (digits.card.choose 2 * 2) +  -- two-digit numbers
  (digits.card.factorial)  -- three-digit numbers

/-- Theorem stating that the total number of different natural numbers
    that can be formed using the digits 1, 2, and 3 is equal to 15 -/
theorem count_numbers_equals_fifteen :
  count_numbers digits = 15 := by
  sorry


end count_numbers_equals_fifteen_l2538_253878


namespace trigonometric_identity_l2538_253836

theorem trigonometric_identity (x : ℝ) : 
  4 * Real.sin (5 * x) * Real.cos (5 * x) * (Real.cos x ^ 4 - Real.sin x ^ 4) = Real.sin (4 * x) := by
  sorry

end trigonometric_identity_l2538_253836


namespace sarah_picked_five_times_as_many_l2538_253888

def sarah_apples : ℝ := 45.0
def brother_apples : ℝ := 9.0

theorem sarah_picked_five_times_as_many :
  sarah_apples / brother_apples = 5 := by
  sorry

end sarah_picked_five_times_as_many_l2538_253888


namespace difference_2020th_2010th_term_l2538_253803

-- Define the arithmetic sequence
def arithmeticSequence (n : ℕ) : ℤ :=
  -10 + (n - 1) * 9

-- State the theorem
theorem difference_2020th_2010th_term :
  (arithmeticSequence 2020 - arithmeticSequence 2010).natAbs = 90 := by
  sorry

end difference_2020th_2010th_term_l2538_253803


namespace sqrt_six_div_sqrt_two_eq_sqrt_three_l2538_253887

theorem sqrt_six_div_sqrt_two_eq_sqrt_three : 
  Real.sqrt 6 / Real.sqrt 2 = Real.sqrt 3 := by
  sorry

end sqrt_six_div_sqrt_two_eq_sqrt_three_l2538_253887


namespace angle_measure_proof_l2538_253808

theorem angle_measure_proof (x : ℝ) : 
  (x + (3 * x - 2) = 180) → x = 45.5 := by
  sorry

end angle_measure_proof_l2538_253808


namespace intersection_implies_a_value_l2538_253864

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | 2*x + a ≤ 0}

-- State the theorem
theorem intersection_implies_a_value :
  ∃ a : ℝ, (A ∩ B a = {x | -2 ≤ x ∧ x ≤ 1}) → a = -2 := by
  sorry

end intersection_implies_a_value_l2538_253864


namespace one_number_is_zero_l2538_253893

/-- Represents a card with a number -/
structure Card where
  value : ℤ

/-- Represents the deck of 30 cards -/
def Deck : Type := Fin 30 → Card

/-- The property that for any 5 cards, there exist another 5 cards such that their sum is zero -/
def has_zero_sum_property (deck : Deck) : Prop :=
  ∀ (s : Finset (Fin 30)) (hs : s.card = 5),
    ∃ (t : Finset (Fin 30)) (ht : t.card = 5) (hd : Disjoint s t),
      (s.sum (λ i => (deck i).value) + t.sum (λ i => (deck i).value) = 0)

/-- The theorem to be proved -/
theorem one_number_is_zero
  (deck : Deck)
  (ha : ∃ a : ℤ, (Finset.filter (λ i => (deck i).value = a) (Finset.univ : Finset (Fin 30))).card = 10)
  (hb : ∃ b : ℤ, (Finset.filter (λ i => (deck i).value = b) (Finset.univ : Finset (Fin 30))).card = 10)
  (hc : ∃ c : ℤ, (Finset.filter (λ i => (deck i).value = c) (Finset.univ : Finset (Fin 30))).card = 10)
  (hdiff : ∀ x y, x ≠ y → (Finset.filter (λ i => (deck i).value = x) (Finset.univ : Finset (Fin 30))).card = 10 →
                         (Finset.filter (λ i => (deck i).value = y) (Finset.univ : Finset (Fin 30))).card = 10 → x ≠ y)
  (hzero_sum : has_zero_sum_property deck) :
  ∃ x : ℤ, x = 0 ∧ (Finset.filter (λ i => (deck i).value = x) (Finset.univ : Finset (Fin 30))).card = 10 :=
by
  sorry

end one_number_is_zero_l2538_253893


namespace trigonometric_identities_l2538_253834

theorem trigonometric_identities :
  (2 * Real.cos (30 * π / 180) - Real.tan (60 * π / 180) + Real.sin (45 * π / 180) * Real.cos (45 * π / 180) = 1/2) ∧
  ((-1)^2023 + 2 * Real.sin (45 * π / 180) - Real.cos (30 * π / 180) + Real.sin (60 * π / 180) + Real.tan (60 * π / 180)^2 = 2 + Real.sqrt 2) :=
by sorry

end trigonometric_identities_l2538_253834


namespace power_inequality_l2538_253851

theorem power_inequality : 2^(1/5) > 0.4^(1/5) ∧ 0.4^(1/5) > 0.4^(3/5) := by
  sorry

end power_inequality_l2538_253851


namespace line_segment_solution_l2538_253817

def line_segment_start : ℝ × ℝ := (2, 5)
def line_segment_end (x : ℝ) : ℝ × ℝ := (x, 10)

theorem line_segment_solution (x : ℝ) 
  (h1 : Real.sqrt ((x - 2)^2 + (10 - 5)^2) = 13)
  (h2 : x > 0) : 
  x = 14 := by sorry

end line_segment_solution_l2538_253817


namespace intersection_complement_subset_condition_l2538_253889

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - x - 12 < 0}
def B : Set ℝ := {x | x^2 + 2*x - 8 > 0}
def C (a : ℝ) : Set ℝ := {x | x^2 - 4*a*x + 3*a^2 < 0}

-- Part I: Prove A ∩ (C \ B) = (-3, 2]
theorem intersection_complement (a : ℝ) (h : a > 0) :
  A ∩ (Set.diff (C a) B) = Set.Ioc (-3) 2 :=
sorry

-- Part II: Prove the range of a for which C ⊇ (A ∩ B)
theorem subset_condition (a : ℝ) (h : a > 0) :
  C a ⊇ (A ∩ B) ↔ 4/3 ≤ a ∧ a ≤ 2 :=
sorry

end intersection_complement_subset_condition_l2538_253889


namespace rearrangement_sum_not_all_nines_rearrangement_sum_power_ten_divisible_l2538_253871

/-- Represents a rearrangement of digits of a natural number -/
def rearrangement (n : ℕ) : ℕ → Prop :=
  sorry

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  sorry

theorem rearrangement_sum_not_all_nines (n : ℕ) :
  ∀ m : ℕ, rearrangement n m → n + m ≠ 10^1967 - 1 :=
sorry

theorem rearrangement_sum_power_ten_divisible (n : ℕ) :
  (∃ m : ℕ, rearrangement n m ∧ n + m = 10^10) → n % 10 = 0 :=
sorry

end rearrangement_sum_not_all_nines_rearrangement_sum_power_ten_divisible_l2538_253871


namespace triangle_count_is_53_l2538_253877

/-- Represents a rectangle divided into smaller sections -/
structure DividedRectangle where
  width : ℕ
  height : ℕ
  horizontal_divisions : ℕ
  vertical_divisions : ℕ

/-- Counts the number of triangles in a divided rectangle -/
def count_triangles (r : DividedRectangle) : ℕ :=
  let smallest_triangles := 24
  let isosceles_triangles := 6
  let rectangular_half_area_triangles := 12
  let larger_right_triangles := 8
  let large_isosceles_triangles := 3
  smallest_triangles + isosceles_triangles + rectangular_half_area_triangles + 
  larger_right_triangles + large_isosceles_triangles

/-- The total number of triangles in the given divided rectangle is 53 -/
theorem triangle_count_is_53 (r : DividedRectangle) : 
  count_triangles r = 53 := by sorry

end triangle_count_is_53_l2538_253877


namespace cookie_distribution_l2538_253814

theorem cookie_distribution (bags : ℕ) (cookies_per_bag : ℕ) (damaged_cookies : ℕ) (people : ℕ) :
  bags = 295 →
  cookies_per_bag = 738 →
  damaged_cookies = 13 →
  people = 125 →
  (bags * cookies_per_bag - bags * damaged_cookies) / people = 1711 :=
by sorry

end cookie_distribution_l2538_253814


namespace target_hit_probability_l2538_253860

/-- The probability of hitting a target in one shot. -/
def p : ℝ := 0.6

/-- The number of shots taken. -/
def n : ℕ := 3

/-- The probability of hitting the target at least twice in n shots. -/
def prob_at_least_two_hits : ℝ := 
  (n.choose 2) * p^2 * (1 - p) + (n.choose 3) * p^3

theorem target_hit_probability : prob_at_least_two_hits = 81/125 := by
  sorry

end target_hit_probability_l2538_253860


namespace original_bottle_size_l2538_253847

/-- The amount of wax needed for Kellan's car -/
def car_wax : ℕ := 3

/-- The amount of wax needed for Kellan's SUV -/
def suv_wax : ℕ := 4

/-- The amount of wax spilled before use -/
def spilled_wax : ℕ := 2

/-- The amount of wax left after detailing both vehicles -/
def leftover_wax : ℕ := 2

/-- Theorem stating the original bottle size -/
theorem original_bottle_size : 
  car_wax + suv_wax + spilled_wax + leftover_wax = 11 := by
  sorry

end original_bottle_size_l2538_253847


namespace equilateral_max_altitude_sum_l2538_253890

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)

-- Define the total length of bisectors
def totalBisectorLength (t : Triangle) : ℝ := sorry

-- Define the total length of altitudes
def totalAltitudeLength (t : Triangle) : ℝ := sorry

-- Define an equilateral triangle
def isEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

-- Theorem statement
theorem equilateral_max_altitude_sum 
  (t : Triangle) 
  (fixed_bisector_sum : ℝ) 
  (h_bisector_sum : totalBisectorLength t = fixed_bisector_sum) :
  ∀ t' : Triangle, 
    totalBisectorLength t' = fixed_bisector_sum → 
    totalAltitudeLength t' ≤ totalAltitudeLength t ↔ 
    isEquilateral t :=
sorry

end equilateral_max_altitude_sum_l2538_253890


namespace divisor_property_characterization_l2538_253897

/-- A positive integer n > 1 satisfies the divisor property if all its positive divisors
    greater than 1 are of the form a^r + 1, where a and r are positive integers and r > 1 -/
def satisfies_divisor_property (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d ∣ n → ∃ a r : ℕ, a > 0 ∧ r > 1 ∧ d = a^r + 1

/-- The main theorem stating that if n satisfies the divisor property,
    then n = 10 or n is a prime of the form a^2 + 1 -/
theorem divisor_property_characterization (n : ℕ) :
  satisfies_divisor_property n →
  (n = 10 ∨ (Nat.Prime n ∧ ∃ a : ℕ, n = a^2 + 1)) :=
by sorry


end divisor_property_characterization_l2538_253897


namespace special_number_theorem_l2538_253831

def is_smallest_nontrivial_divisor (a n : ℕ) : Prop :=
  a ≠ 1 ∧ a ∣ n ∧ ∀ d, 1 < d → d < a → ¬(d ∣ n)

theorem special_number_theorem :
  ∀ n : ℕ, n ≥ 2 →
  (∃ a b : ℕ, is_smallest_nontrivial_divisor a n ∧ b ∣ n ∧ n = a^2 + b^2) →
  (n = 8 ∨ n = 20) :=
sorry

end special_number_theorem_l2538_253831


namespace parabola_intersection_range_l2538_253804

/-- A parabola defined by y = ax^2 -/
def Parabola (a : ℝ) : ℝ → ℝ := λ x ↦ a * x^2

/-- A line passing through (1, -2) with slope k -/
def Line (k : ℝ) : ℝ → ℝ := λ x ↦ k * (x - 1) - 2

/-- Checks if a parabola intersects a line -/
def intersects (p : ℝ → ℝ) (l : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, p x = l x

theorem parabola_intersection_range (a : ℝ) :
  (∃ k : ℝ, k ≠ 0 ∧ (intersects (Parabola a) (Line k) ∨ intersects (Parabola a) (Line (-1/k)))) →
  a < 0 ∨ (0 < a ∧ a ≤ 1/8) := by
  sorry

end parabola_intersection_range_l2538_253804


namespace expression_equality_l2538_253821

theorem expression_equality (x y z : ℝ) :
  (-5 * x^3 * y^2 * z^3)^2 / (5 * x * y^2) * (6 * x^4 * y)^0 = 5 * x^5 * y^2 * z^6 := by
  sorry

end expression_equality_l2538_253821


namespace min_value_expression_l2538_253824

theorem min_value_expression (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
  (h4 : x + y + z = 1) (h5 : x = 2 * y) :
  ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 1 → a = 2 * b →
    (x + 2 * y) / (x * y * z) ≤ (a + 2 * b) / (a * b * c) :=
by sorry

end min_value_expression_l2538_253824


namespace area_of_encompassing_rectangle_l2538_253827

/-- Given two identical rectangles with intersecting extended sides, prove the area of the encompassing rectangle. -/
theorem area_of_encompassing_rectangle 
  (area_BNHM : ℝ) 
  (area_MBCK : ℝ) 
  (area_MLGH : ℝ) 
  (h1 : area_BNHM = 12)
  (h2 : area_MBCK = 63)
  (h3 : area_MLGH = 28) : 
  ∃ (area_IFJD : ℝ), area_IFJD = 418 := by
sorry

end area_of_encompassing_rectangle_l2538_253827


namespace first_terrific_tuesday_after_school_start_l2538_253879

/-- Represents a date in October --/
structure OctoberDate :=
  (day : Nat)
  (h : day ≥ 1 ∧ day ≤ 31)

/-- Represents a day of the week --/
inductive DayOfWeek
  | Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

/-- Function to determine the day of the week for a given October date --/
def dayOfWeek (date : OctoberDate) : DayOfWeek :=
  sorry

/-- Predicate to check if a date is a Tuesday --/
def isTuesday (date : OctoberDate) : Prop :=
  dayOfWeek date = DayOfWeek.Tuesday

/-- Function to count the number of Tuesdays before a given date in October --/
def tuesdaysBeforeDate (date : OctoberDate) : Nat :=
  sorry

/-- Predicate to check if a date is a Terrific Tuesday --/
def isTerrificTuesday (date : OctoberDate) : Prop :=
  isTuesday date ∧ tuesdaysBeforeDate date = 4

/-- The school start date --/
def schoolStartDate : OctoberDate :=
  ⟨5, sorry⟩

/-- Theorem: The first Terrific Tuesday after school starts is October 31 --/
theorem first_terrific_tuesday_after_school_start :
  ∃ (date : OctoberDate),
    date.day = 31 ∧
    isTerrificTuesday date ∧
    (∀ (earlier_date : OctoberDate),
      earlier_date.day > schoolStartDate.day ∧
      earlier_date.day < date.day →
      ¬isTerrificTuesday earlier_date) :=
  sorry

end first_terrific_tuesday_after_school_start_l2538_253879


namespace power_sum_equality_l2538_253813

theorem power_sum_equality : 2^300 + (-2^301) = -2^300 := by sorry

end power_sum_equality_l2538_253813


namespace persons_age_l2538_253806

theorem persons_age : ∃ (age : ℕ), 
  (6 * (age + 6) - 6 * (age - 6) = age) ∧ (age = 72) := by
  sorry

end persons_age_l2538_253806


namespace sqrt_real_iff_geq_two_l2538_253849

theorem sqrt_real_iff_geq_two (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 := by
  sorry

end sqrt_real_iff_geq_two_l2538_253849


namespace sixth_term_of_geometric_sequence_l2538_253846

/-- Given a geometric sequence with first term 3 and second term -1/6,
    prove that its sixth term is -1/629856 -/
theorem sixth_term_of_geometric_sequence (a₁ a₂ : ℚ) (h₁ : a₁ = 3) (h₂ : a₂ = -1/6) :
  let r := a₂ / a₁
  a₁ * r^5 = -1/629856 := by
sorry

end sixth_term_of_geometric_sequence_l2538_253846


namespace division_problem_l2538_253820

theorem division_problem (a b c d : ℚ) : 
  a + b + c + d = 5440 ∧
  a / b = 2 / 3 ∧
  b / c = 3 / 5 ∧
  c / d = 5 / 6 →
  a = 680 ∧ b = 1020 ∧ c = 1700 ∧ d = 2040 := by
sorry

end division_problem_l2538_253820


namespace david_squats_l2538_253874

/-- Fitness competition between David and Zachary -/
theorem david_squats (zachary_pushups zachary_crunches zachary_squats : ℕ) 
  (h1 : zachary_pushups = 68)
  (h2 : zachary_crunches = 130)
  (h3 : zachary_squats = 58) :
  ∃ (x : ℕ),
    x = zachary_squats ∧
    2 * zachary_pushups = zachary_pushups + x ∧
    zachary_crunches = (zachary_crunches - x / 2) + x / 2 ∧
    3 * x = 174 :=
by sorry

end david_squats_l2538_253874


namespace error_percentage_limit_l2538_253844

theorem error_percentage_limit :
  ∀ ε > 0, ∃ N : ℝ, ∀ x ≥ N,
    x > 0 → |((7 * x + 8) / (8 * x)) * 100 - 87.5| < ε := by
  sorry

end error_percentage_limit_l2538_253844


namespace rogers_books_l2538_253865

/-- Given that Roger reads a certain number of books per week and takes a specific number of weeks to finish a series, calculate the total number of books in the series. -/
theorem rogers_books (books_per_week : ℕ) (weeks_to_finish : ℕ) : books_per_week = 6 → weeks_to_finish = 5 → books_per_week * weeks_to_finish = 30 := by
  sorry

#check rogers_books

end rogers_books_l2538_253865


namespace inequality_proof_l2538_253894

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 + 3 / (a * b + b * c + c * a) ≥ 6 / (a + b + c) := by
  sorry

end inequality_proof_l2538_253894


namespace smallest_prime_digit_sum_18_l2538_253857

/-- The sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Primality test for natural numbers -/
def is_prime (n : ℕ) : Prop := sorry

/-- Theorem: 819 is the smallest prime number whose digits sum to 18 -/
theorem smallest_prime_digit_sum_18 : 
  (is_prime 819 ∧ digit_sum 819 = 18) ∧ 
  ∀ n : ℕ, n < 819 → ¬(is_prime n ∧ digit_sum n = 18) :=
sorry

end smallest_prime_digit_sum_18_l2538_253857


namespace complex_equation_solution_l2538_253882

theorem complex_equation_solution :
  ∀ (x y : ℝ), (Complex.mk (2 * x - 1) 1 = Complex.mk y (y - 3)) → (x = 2.5 ∧ y = 4) :=
by
  sorry

end complex_equation_solution_l2538_253882


namespace repeating_decimal_47_l2538_253899

theorem repeating_decimal_47 :
  ∃ (x : ℚ), (∀ (n : ℕ), (x * 10^(2*n+2) - ⌊x * 10^(2*n+2)⌋ = 47 / 100)) ∧ x = 47 / 99 :=
sorry

end repeating_decimal_47_l2538_253899


namespace parabola_vertex_l2538_253842

/-- Given a quadratic function f(x) = -2x^2 + cx + d where the solution to f(x) ≤ 0 is [-7/2, ∞),
    the vertex of the parabola defined by f(x) is (-7/2, 0). -/
theorem parabola_vertex (c d : ℝ) :
  let f : ℝ → ℝ := λ x => -2 * x^2 + c * x + d
  (∀ x, f x ≤ 0 ↔ x ∈ Set.Ici (-7/2)) →
  ∃! v : ℝ × ℝ, v.1 = -7/2 ∧ v.2 = 0 ∧ ∀ x, f x ≤ f v.1 :=
by sorry

end parabola_vertex_l2538_253842


namespace greatest_valid_number_l2538_253853

def is_valid_number (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length > 2 ∧
  ∀ i, 1 < i ∧ i < digits.length - 1 →
    digits[i]! < (digits[i-1]! + digits[i+1]!) / 2

theorem greatest_valid_number :
  ∀ n : ℕ, is_valid_number n → n ≤ 986421 :=
sorry

end greatest_valid_number_l2538_253853


namespace inequality_solution_set_l2538_253896

theorem inequality_solution_set : 
  {x : ℝ | 2*x + 1 > x + 2} = {x : ℝ | x > 1} := by sorry

end inequality_solution_set_l2538_253896


namespace inequality_proof_l2538_253819

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a * b + b * c + c * a = 1) : 
  (a + b + c ≥ Real.sqrt 3) ∧ 
  (Real.sqrt (a / (b * c)) + Real.sqrt (b / (c * a)) + Real.sqrt (c / (a * b)) ≥ 
   Real.sqrt 3 * (Real.sqrt a + Real.sqrt b + Real.sqrt c)) := by
  sorry

end inequality_proof_l2538_253819


namespace integer_pair_conditions_l2538_253810

theorem integer_pair_conditions (a b : ℤ) : 
  (a - b - 1) ∣ (a^2 + b^2) ∧ 
  (a^2 + b^2) / (2*a*b - 1) = 20/19 ↔ 
  ((a = 22 ∧ b = 16) ∨ 
   (a = -16 ∧ b = -22) ∨ 
   (a = 8 ∧ b = 6) ∨ 
   (a = -6 ∧ b = -8)) := by
sorry

end integer_pair_conditions_l2538_253810


namespace complex_fraction_simplification_l2538_253802

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (2 - 2*i) / (3 + 4*i) = -2/25 - 14/25*i :=
by
  sorry

end complex_fraction_simplification_l2538_253802


namespace second_train_speed_l2538_253838

/-- Given two trains leaving a station simultaneously, prove that if one train
    travels 200 miles at 50 MPH and the other travels 240 miles, and their
    average travel time is 4 hours, then the speed of the second train is 60 MPH. -/
theorem second_train_speed (distance1 distance2 speed1 avg_time : ℝ) :
  distance1 = 200 →
  distance2 = 240 →
  speed1 = 50 →
  avg_time = 4 →
  (distance1 / speed1 + distance2 / (distance2 / avg_time)) / 2 = avg_time →
  distance2 / avg_time = 60 := by
  sorry

#check second_train_speed

end second_train_speed_l2538_253838


namespace root_in_interval_l2538_253891

theorem root_in_interval : ∃ x : ℝ, x ∈ Set.Ioo (-4 : ℝ) (-3 : ℝ) ∧ x^3 + 3*x^2 - x + 1 = 0 := by
  sorry

end root_in_interval_l2538_253891


namespace inequality_proof_l2538_253825

theorem inequality_proof (a b c : ℝ) (h : a > b) : a * c^2 ≥ b * c^2 := by
  sorry

end inequality_proof_l2538_253825


namespace neg_one_pow_2022_eq_one_neg_one_pow_2022_and_one_are_opposite_l2538_253883

/-- Two real numbers are opposite if their sum is zero -/
def are_opposite (a b : ℝ) : Prop := a + b = 0

/-- -1^2022 equals 1 -/
theorem neg_one_pow_2022_eq_one : (-1 : ℝ)^2022 = 1 := by sorry

theorem neg_one_pow_2022_and_one_are_opposite :
  are_opposite ((-1 : ℝ)^2022) 1 := by sorry

end neg_one_pow_2022_eq_one_neg_one_pow_2022_and_one_are_opposite_l2538_253883


namespace competition_tables_l2538_253881

theorem competition_tables (total_legs : ℕ) : total_legs = 816 →
  ∃ (num_tables : ℕ),
    num_tables * (3 * 8 + 6 * 2 + 4) = total_legs ∧
    num_tables = 20 := by
  sorry

end competition_tables_l2538_253881


namespace quarterback_passes_l2538_253892

theorem quarterback_passes (total : ℕ) (left : ℕ) (right : ℕ) (center : ℕ) 
  (h1 : total = 50)
  (h2 : right = 2 * left)
  (h3 : center = left + 2)
  (h4 : total = left + right + center) :
  left = 12 := by
sorry

end quarterback_passes_l2538_253892


namespace x_one_minus_f_equals_seven_to_500_l2538_253884

theorem x_one_minus_f_equals_seven_to_500 :
  let x : ℝ := (3 + Real.sqrt 2) ^ 500
  let n : ℤ := ⌊x⌋
  let f : ℝ := x - n
  x * (1 - f) = 7 ^ 500 := by
sorry

end x_one_minus_f_equals_seven_to_500_l2538_253884


namespace at_least_one_white_certain_l2538_253870

-- Define the number of balls
def total_balls : ℕ := 6
def black_balls : ℕ := 2
def white_balls : ℕ := 4
def drawn_balls : ℕ := 3

-- Define the event of drawing at least one white ball
def at_least_one_white (drawn : Finset ℕ) : Prop :=
  ∃ b ∈ drawn, b > black_balls

-- Theorem statement
theorem at_least_one_white_certain :
  ∀ (drawn : Finset ℕ), drawn.card = drawn_balls → at_least_one_white drawn :=
sorry

end at_least_one_white_certain_l2538_253870


namespace polynomial_difference_divisibility_l2538_253845

/-- Given a polynomial P with integer coefficients, 
    (a-b) divides (P(a)-P(b)) for all integers a and b -/
theorem polynomial_difference_divisibility 
  (P : Polynomial ℤ) (a b : ℤ) : 
  (a - b) ∣ (P.eval a - P.eval b) := by
  sorry

end polynomial_difference_divisibility_l2538_253845


namespace cat_toy_cost_l2538_253850

/-- The cost of a cat toy given the total amount paid, the cost of a cage, and the change received. -/
theorem cat_toy_cost (total_paid : ℚ) (cage_cost : ℚ) (change : ℚ) :
  total_paid = 20 →
  cage_cost = 10.97 →
  change = 0.26 →
  total_paid - change - cage_cost = 8.77 := by
sorry

end cat_toy_cost_l2538_253850


namespace propositions_true_l2538_253826

-- Define reciprocals
def reciprocals (x y : ℝ) : Prop := x * y = 1

-- Define the quadratic equation
def has_real_roots (b : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - 2*b*x + b^2 + b = 0

theorem propositions_true :
  (∀ x y : ℝ, reciprocals x y → x * y = 1) ∧
  (∀ b : ℝ, ¬(has_real_roots b) → b > -1) :=
sorry

end propositions_true_l2538_253826


namespace ice_cream_combinations_l2538_253837

theorem ice_cream_combinations : Nat.choose 8 2 = 28 := by
  sorry

end ice_cream_combinations_l2538_253837


namespace max_tangent_circles_is_three_l2538_253823

/-- An annulus with inner radius 1 and outer radius 9 -/
structure Annulus where
  inner_radius : ℝ := 1
  outer_radius : ℝ := 9

/-- A circle tangent to both the inner and outer circles of the annulus -/
structure TangentCircle (A : Annulus) where
  radius : ℝ
  center_distance : ℝ
  tangent_inner : center_distance = A.inner_radius + radius
  tangent_outer : center_distance = A.outer_radius - radius

/-- The maximum number of non-overlapping tangent circles in the annulus -/
def max_tangent_circles (A : Annulus) : ℕ :=
  sorry

/-- The theorem stating that the maximum number of non-overlapping tangent circles is 3 -/
theorem max_tangent_circles_is_three (A : Annulus) : max_tangent_circles A = 3 := by
  sorry

end max_tangent_circles_is_three_l2538_253823


namespace speed_limit_correct_l2538_253898

/-- The fine rate for speeding in dollars per mph over the limit -/
def fine_rate : ℕ := 16

/-- The total fine Jed received in dollars -/
def total_fine : ℕ := 256

/-- Jed's speed in mph -/
def jed_speed : ℕ := 66

/-- The posted speed limit on the road -/
def speed_limit : ℕ := 50

/-- Theorem stating that the given speed limit satisfies the conditions of the problem -/
theorem speed_limit_correct : 
  fine_rate * (jed_speed - speed_limit) = total_fine :=
by
  sorry


end speed_limit_correct_l2538_253898


namespace f_symmetric_l2538_253852

/-- The number of integer sequences of length n with sum of absolute values not exceeding m -/
def f (n m : ℕ) : ℕ := sorry

/-- Theorem stating that f(a, b) = f(b, a) for positive integers a and b -/
theorem f_symmetric {a b : ℕ} (ha : 0 < a) (hb : 0 < b) : f a b = f b a := by sorry

end f_symmetric_l2538_253852


namespace water_requirement_proof_l2538_253875

/-- The water requirement per household per month in a village -/
def water_per_household (total_water : ℕ) (num_households : ℕ) : ℕ :=
  total_water / num_households

/-- Theorem: The water requirement per household per month is 200 litres -/
theorem water_requirement_proof :
  water_per_household 2000 10 = 200 := by
  sorry

end water_requirement_proof_l2538_253875


namespace unique_solution_cube_equation_l2538_253807

theorem unique_solution_cube_equation :
  ∃! (x : ℝ), x ≠ 0 ∧ (3 * x)^5 = (9 * x)^4 ∧ x = 27 := by
sorry

end unique_solution_cube_equation_l2538_253807


namespace infinitely_many_S_3n_geq_S_3n_plus_1_l2538_253885

-- Define the sum of digits function
def S (m : ℕ) : ℕ := sorry

-- Theorem statement
theorem infinitely_many_S_3n_geq_S_3n_plus_1 :
  ∀ N : ℕ, ∃ n : ℕ, n ≥ N ∧ S (3^n) ≥ S (3^(n+1)) :=
sorry

end infinitely_many_S_3n_geq_S_3n_plus_1_l2538_253885


namespace perimeter_implies_equilateral_l2538_253856

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the perimeter of a triangle
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

-- Define an equilateral triangle
def is_equilateral (t : Triangle) : Prop := t.a = t.b ∧ t.b = t.c

-- Theorem statement
theorem perimeter_implies_equilateral (t : Triangle) :
  perimeter t = 3 + 2 * Real.sqrt 3 → is_equilateral t := by
  sorry

end perimeter_implies_equilateral_l2538_253856


namespace triangle_isosceles_l2538_253822

/-- If in triangle ABC, a = 2b cos C, then triangle ABC is isosceles -/
theorem triangle_isosceles (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Sides are positive
  a = 2 * b * Real.cos C →  -- Given condition
  b = c  -- Definition of isosceles triangle
  := by sorry

end triangle_isosceles_l2538_253822


namespace jellybean_removal_l2538_253843

theorem jellybean_removal (initial : ℕ) (added_back : ℕ) (removed_after : ℕ) (final : ℕ) :
  initial = 37 ∧ added_back = 5 ∧ removed_after = 4 ∧ final = 23 →
  ∃ (removed : ℕ), initial - removed + added_back - removed_after = final ∧ removed = 15 :=
by sorry

end jellybean_removal_l2538_253843


namespace y_derivative_l2538_253868

noncomputable def y (x : ℝ) : ℝ := (Real.sin x) / x + Real.sqrt x + 2

theorem y_derivative (x : ℝ) (h : x ≠ 0) :
  deriv y x = (x * Real.cos x - Real.sin x) / x^2 + 1 / (2 * Real.sqrt x) :=
by sorry

end y_derivative_l2538_253868


namespace hyperbola_equation_l2538_253815

theorem hyperbola_equation (a b c : ℝ) (h1 : 2 * a = 2) (h2 : c / a = Real.sqrt 2) :
  (∀ x y, x^2 - y^2 = 1 ∨ y^2 - x^2 = 1) :=
sorry

end hyperbola_equation_l2538_253815


namespace root_properties_l2538_253866

theorem root_properties (m n : ℤ) (x₁ x₂ : ℝ) : 
  Odd m → Odd n → x₁^2 + m*x₁ + n = 0 → x₂^2 + m*x₂ + n = 0 → x₁ ≠ x₂ →
  ¬(∃ k : ℤ, x₁ = k) ∧ ¬(∃ k : ℤ, x₂ = k) := by
sorry

end root_properties_l2538_253866


namespace employee_wage_calculation_l2538_253848

theorem employee_wage_calculation (revenue : ℝ) (num_employees : ℕ) 
  (tax_rate : ℝ) (marketing_rate : ℝ) (operational_rate : ℝ) (wage_rate : ℝ) :
  revenue = 400000 →
  num_employees = 10 →
  tax_rate = 0.1 →
  marketing_rate = 0.05 →
  operational_rate = 0.2 →
  wage_rate = 0.15 →
  let after_tax := revenue * (1 - tax_rate)
  let after_marketing := after_tax * (1 - marketing_rate)
  let after_operational := after_marketing * (1 - operational_rate)
  let total_wages := after_operational * wage_rate
  let wage_per_employee := total_wages / num_employees
  wage_per_employee = 4104 :=
by sorry

end employee_wage_calculation_l2538_253848


namespace sets_theorem_l2538_253835

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}

-- Define set B (parameterized by a)
def B (a : ℝ) : Set ℝ := {x : ℝ | a ≤ x ∧ x ≤ 3 - 2*a}

theorem sets_theorem :
  (∀ a : ℝ, (Aᶜ ∪ B a = U) ↔ a ≤ 0) ∧
  (∀ a : ℝ, (A ∩ B a ≠ B a) ↔ a < 1/2) :=
sorry

end sets_theorem_l2538_253835


namespace arithmetic_sequence_formula_l2538_253886

/-- An arithmetic sequence with first three terms x-1, x+1, and 2x+3 has the general formula a_n = 2n - 3 -/
theorem arithmetic_sequence_formula (x : ℝ) (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 1 = x - 1 →                                         -- first term
  a 2 = x + 1 →                                         -- second term
  a 3 = 2 * x + 3 →                                     -- third term
  ∀ n : ℕ, a n = 2 * n - 3 :=                           -- general formula
by sorry

end arithmetic_sequence_formula_l2538_253886


namespace shooting_mode_l2538_253869

def binomial_mode (n : ℕ) (p : ℝ) : Set ℕ :=
  {k : ℕ | ∀ i : ℕ, i ≤ n → (n.choose k) * p^k * (1-p)^(n-k) ≥ (n.choose i) * p^i * (1-p)^(n-i)}

theorem shooting_mode :
  binomial_mode 19 0.8 = {15, 16} := by
  sorry

end shooting_mode_l2538_253869


namespace general_term_formula_l2538_253895

def S (n : ℕ) : ℤ := 1 - 2^n

def a (n : ℕ) : ℤ := -2^(n-1)

theorem general_term_formula (n : ℕ) (h : n ≥ 2) : 
  a n = S n - S (n-1) := by
  sorry

end general_term_formula_l2538_253895


namespace no_solution_exists_l2538_253833

theorem no_solution_exists : ¬ ∃ x : ℝ, 2 < 2 * x ∧ 2 * x < 3 ∧ 1 < 4 * x ∧ 4 * x < 4 := by
  sorry

end no_solution_exists_l2538_253833


namespace units_digit_of_j_squared_plus_three_to_j_l2538_253858

/-- Given j = 2017^3 + 3^2017 - 1, prove that j^2 + 3^j ≡ 8 (mod 10) -/
theorem units_digit_of_j_squared_plus_three_to_j (j : ℕ) : 
  j = 2017^3 + 3^2017 - 1 → (j^2 + 3^j) % 10 = 8 := by
  sorry

end units_digit_of_j_squared_plus_three_to_j_l2538_253858


namespace tenth_grader_max_points_l2538_253829

/-- Represents the grade of a student --/
inductive Grade
  | tenth
  | eleventh

/-- Represents the result of a chess game --/
inductive GameResult
  | win
  | draw
  | loss

/-- Calculate points for a game result --/
def pointsForResult (result : GameResult) : Real :=
  match result with
  | GameResult.win => 1
  | GameResult.draw => 0.5
  | GameResult.loss => 0

/-- Structure representing a chess tournament --/
structure ChessTournament where
  tenthGraders : Nat
  eleventhGraders : Nat
  tenthGraderPoints : Real
  eleventhGraderPoints : Real

/-- Theorem stating the maximum points a 10th grader can score --/
theorem tenth_grader_max_points (tournament : ChessTournament) 
  (h1 : tournament.eleventhGraders = 10 * tournament.tenthGraders)
  (h2 : tournament.eleventhGraderPoints = 4.5 * tournament.tenthGraderPoints)
  (h3 : tournament.tenthGraders > 0) :
  ∃ (maxPoints : Real), 
    maxPoints = 10 ∧ 
    ∀ (points : Real), 
      (∃ (player : Nat), player ≤ tournament.tenthGraders ∧ points = tournament.tenthGraderPoints / tournament.tenthGraders) →
      points ≤ maxPoints :=
by sorry

end tenth_grader_max_points_l2538_253829


namespace building_heights_properties_l2538_253800

/-- Heights of the buildings in meters -/
def burj_khalifa : ℕ := 828
def shanghai_tower : ℕ := 632
def one_world_trade_center : ℕ := 541
def willis_tower : ℕ := 527

/-- List of building heights -/
def building_heights : List ℕ := [burj_khalifa, shanghai_tower, one_world_trade_center, willis_tower]

/-- Theorem stating the total height and average height difference -/
theorem building_heights_properties :
  (building_heights.sum = 2528) ∧
  (((building_heights.map (λ h => h - willis_tower)).sum : ℚ) / 4 = 105) := by
  sorry

end building_heights_properties_l2538_253800


namespace line_through_point_l2538_253828

theorem line_through_point (b : ℝ) : 
  (∀ x y : ℝ, b*x + (b-1)*y = b^2 - 1 → (x = 2 ∧ y = -5)) → 
  (b = (-3 + Real.sqrt 33) / 2 ∨ b = (-3 - Real.sqrt 33) / 2) :=
sorry

end line_through_point_l2538_253828


namespace base_of_second_exponent_l2538_253811

theorem base_of_second_exponent (a b : ℕ+) (some_number : ℕ) 
  (h1 : (18 ^ a.val) * 9 ^ (3 * a.val - 1) = (2 ^ 6) * (some_number ^ b.val))
  (h2 : a = 6) : 
  some_number = 3 := by
  sorry

end base_of_second_exponent_l2538_253811


namespace cookie_problem_l2538_253818

/-- Calculates the number of chocolate chips per cookie -/
def chocolate_chips_per_cookie (cookies_per_batch : ℕ) (family_members : ℕ) (batches : ℕ) (chips_per_person : ℕ) : ℕ :=
  let total_cookies := cookies_per_batch * batches
  let cookies_per_person := total_cookies / family_members
  chips_per_person / cookies_per_person

/-- Proves that the number of chocolate chips per cookie is 2 under given conditions -/
theorem cookie_problem : 
  chocolate_chips_per_cookie 12 4 3 18 = 2 := by
  sorry

end cookie_problem_l2538_253818


namespace three_digit_subtraction_l2538_253805

/-- Given a three-digit number of the form 4c1 and another of the form 3d5,
    prove that if 786 - 4c1 = 3d5 and 3d5 is divisible by 7, then c + d = 8 -/
theorem three_digit_subtraction (c d : ℕ) : 
  (786 - (400 + c * 10 + 1) = 300 + d * 10 + 5) →
  (300 + d * 10 + 5) % 7 = 0 →
  c + d = 8 := by
sorry

end three_digit_subtraction_l2538_253805
