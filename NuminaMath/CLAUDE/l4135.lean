import Mathlib

namespace pure_imaginary_complex_number_l4135_413508

theorem pure_imaginary_complex_number (a : ℝ) :
  let z : ℂ := (1 + a * Complex.I) / (1 - Complex.I)
  (∃ (b : ℝ), z = b * Complex.I) → a = 1 := by
  sorry

end pure_imaginary_complex_number_l4135_413508


namespace reciprocal_difference_product_relation_l4135_413505

theorem reciprocal_difference_product_relation :
  ∃ (a b : ℕ), a > b ∧ (1 : ℚ) / (a - b) = 3 * (1 : ℚ) / (a * b) :=
by
  use 6, 2
  sorry

end reciprocal_difference_product_relation_l4135_413505


namespace largest_digit_divisible_by_6_l4135_413521

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

def is_single_digit (n : ℕ) : Prop := n < 10

theorem largest_digit_divisible_by_6 :
  ∀ M : ℕ, is_single_digit M →
    (is_divisible_by_6 (45670 + M) → M ≤ 8) ∧
    (is_divisible_by_6 (45678)) := by
  sorry

end largest_digit_divisible_by_6_l4135_413521


namespace triangle_angle_c_is_sixty_degrees_l4135_413541

theorem triangle_angle_c_is_sixty_degrees 
  (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
  (h_sides : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_sin : Real.sin A = 2 * Real.sin B)
  (h_sum : a + b = Real.sqrt 3 * c) :
  C = π / 3 := by
sorry

end triangle_angle_c_is_sixty_degrees_l4135_413541


namespace parallel_line_correct_perpendicular_line_correct_l4135_413559

-- Define the given line
def given_line (x y : ℝ) : Prop := 2 * x - y - 1 = 0

-- Define the point (1,0)
def point : ℝ × ℝ := (1, 0)

-- Define parallel line
def parallel_line (x y : ℝ) : Prop := 2 * x - y - 2 = 0

-- Define perpendicular line
def perpendicular_line (x y : ℝ) : Prop := x + 2 * y - 1 = 0

-- Theorem for parallel line
theorem parallel_line_correct :
  (∀ x y : ℝ, parallel_line x y ↔ (given_line x y ∧ parallel_line x y)) ∧
  parallel_line point.1 point.2 :=
sorry

-- Theorem for perpendicular line
theorem perpendicular_line_correct :
  (∀ x y : ℝ, perpendicular_line x y ↔ (given_line x y ∧ perpendicular_line x y)) ∧
  perpendicular_line point.1 point.2 :=
sorry

end parallel_line_correct_perpendicular_line_correct_l4135_413559


namespace largest_prime_divisor_factorial_sum_l4135_413595

theorem largest_prime_divisor_factorial_sum : ∃ p : ℕ, 
  Nat.Prime p ∧ 
  p ∣ (Nat.factorial 13 + Nat.factorial 14) ∧
  ∀ q : ℕ, Nat.Prime q → q ∣ (Nat.factorial 13 + Nat.factorial 14) → q ≤ p :=
by sorry

end largest_prime_divisor_factorial_sum_l4135_413595


namespace min_product_of_three_l4135_413581

def S : Set Int := {-10, -7, -3, 0, 4, 6, 9}

theorem min_product_of_three (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  ∃ (x y z : Int), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  x * y * z = -540 ∧ 
  ∀ (p q r : Int), p ∈ S → q ∈ S → r ∈ S → p ≠ q → q ≠ r → p ≠ r → 
  p * q * r ≥ -540 :=
sorry

end min_product_of_three_l4135_413581


namespace set_B_equals_l4135_413543

def U : Set Nat := {1, 3, 5, 7, 9}

theorem set_B_equals (A B : Set Nat) 
  (h1 : A ⊆ U)
  (h2 : B ⊆ U)
  (h3 : A ∩ B = {1, 3})
  (h4 : (U \ A) ∩ B = {5}) :
  B = {1, 3, 5} := by
  sorry

end set_B_equals_l4135_413543


namespace color_one_third_square_l4135_413570

theorem color_one_third_square (n : ℕ) (k : ℕ) : n = 18 ∧ k = 6 → Nat.choose n k = 18564 := by
  sorry

end color_one_third_square_l4135_413570


namespace stockholm_uppsala_distance_l4135_413520

/-- Calculates the actual distance between Stockholm and Uppsala based on map measurements and scales. -/
def actual_distance (map_distance : ℝ) (first_part : ℝ) (scale1 : ℝ) (scale2 : ℝ) : ℝ :=
  first_part * scale1 + (map_distance - first_part) * scale2

/-- Theorem stating that the actual distance between Stockholm and Uppsala is 375 km. -/
theorem stockholm_uppsala_distance :
  let map_distance : ℝ := 45
  let first_part : ℝ := 15
  let scale1 : ℝ := 5
  let scale2 : ℝ := 10
  actual_distance map_distance first_part scale1 scale2 = 375 := by
  sorry


end stockholm_uppsala_distance_l4135_413520


namespace unique_five_digit_number_l4135_413542

def is_valid_number (n : ℕ) : Prop :=
  ∃ (x y : ℕ),
    n = 10 * x + y ∧
    0 ≤ y ∧ y ≤ 9 ∧
    10000 ≤ n ∧ n ≤ 99999 ∧
    1000 ≤ x ∧ x ≤ 9999 ∧
    n - x = 54321

theorem unique_five_digit_number : 
  ∃! (n : ℕ), is_valid_number n ∧ n = 60356 :=
sorry

end unique_five_digit_number_l4135_413542


namespace new_person_weight_l4135_413501

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 6 →
  replaced_weight = 45 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 93 :=
by sorry

end new_person_weight_l4135_413501


namespace binomial_60_3_l4135_413531

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_l4135_413531


namespace infinitely_many_divisible_by_15_l4135_413576

def v : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 8 * v (n + 1) - v n

theorem infinitely_many_divisible_by_15 :
  ∀ k : ℕ, ∃ n : ℕ, n > k ∧ 15 ∣ v n :=
sorry

end infinitely_many_divisible_by_15_l4135_413576


namespace total_tabs_is_sixty_l4135_413509

/-- Calculates the total number of tabs opened across all browsers -/
def totalTabs (numBrowsers : ℕ) (windowsPerBrowser : ℕ) (tabsPerWindow : ℕ) : ℕ :=
  numBrowsers * windowsPerBrowser * tabsPerWindow

/-- Theorem: Given the specified conditions, the total number of tabs is 60 -/
theorem total_tabs_is_sixty :
  totalTabs 2 3 10 = 60 := by
  sorry

end total_tabs_is_sixty_l4135_413509


namespace cube_difference_square_root_l4135_413583

theorem cube_difference_square_root : ∃ (n : ℕ), n > 0 ∧ n^2 = 105^3 - 104^3 :=
by
  -- The proof goes here
  sorry

end cube_difference_square_root_l4135_413583


namespace triangle_inequalities_l4135_413529

theorem triangle_inequalities (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a < b + c) (hbc : b < a + c) (hca : c < a + b) :
  (a + b + c = 2 → a^2 + b^2 + c^2 + 2*a*b*c < 2) ∧
  (a + b + c = 1 → a^2 + b^2 + c^2 + 4*a*b*c < 1/2) ∧
  (a + b + c = 1 → 5*(a^2 + b^2 + c^2) + 18*a*b*c > 7/3) :=
by sorry

end triangle_inequalities_l4135_413529


namespace circular_film_radius_l4135_413567

/-- Given a cylindrical canister filled with a liquid that forms a circular film on water,
    this theorem proves that the radius of the resulting circular film is 25√2 cm. -/
theorem circular_film_radius
  (canister_radius : ℝ)
  (canister_height : ℝ)
  (film_thickness : ℝ)
  (h_canister_radius : canister_radius = 5)
  (h_canister_height : canister_height = 10)
  (h_film_thickness : film_thickness = 0.2) :
  let canister_volume := π * canister_radius^2 * canister_height
  let film_radius := Real.sqrt (canister_volume / (π * film_thickness))
  film_radius = 25 * Real.sqrt 2 := by
sorry

end circular_film_radius_l4135_413567


namespace soccer_teams_count_l4135_413590

theorem soccer_teams_count (n : ℕ) (k : ℕ) (h : n = 12 ∧ k = 6) :
  (Nat.choose n k : ℕ) = (Nat.choose n (k - 1) : ℕ) / k :=
by sorry

#check soccer_teams_count

end soccer_teams_count_l4135_413590


namespace square_number_placement_l4135_413516

theorem square_number_placement :
  ∃ (a b c d e : ℕ),
    (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0) ∧
    (Nat.gcd a b > 1 ∧ Nat.gcd b c > 1 ∧ Nat.gcd c d > 1 ∧ Nat.gcd d a > 1) ∧
    (Nat.gcd a e > 1 ∧ Nat.gcd b e > 1 ∧ Nat.gcd c e > 1 ∧ Nat.gcd d e > 1) ∧
    (Nat.gcd a c = 1 ∧ Nat.gcd b d = 1) :=
by sorry

end square_number_placement_l4135_413516


namespace average_problem_l4135_413591

theorem average_problem (y : ℝ) : (15 + 30 + 45 + y) / 4 = 35 → y = 50 := by
  sorry

end average_problem_l4135_413591


namespace middle_number_is_eleven_l4135_413596

theorem middle_number_is_eleven (x y z : ℕ) 
  (sum_xy : x + y = 18) 
  (sum_xz : x + z = 23) 
  (sum_yz : y + z = 27) : 
  y = 11 := by
sorry

end middle_number_is_eleven_l4135_413596


namespace candy_distribution_l4135_413522

theorem candy_distribution (total_candies : ℕ) (num_friends : ℕ) (candies_per_friend : ℕ) :
  total_candies = 36 →
  num_friends = 9 →
  candies_per_friend = 4 →
  total_candies = num_friends * candies_per_friend :=
by
  sorry

#check candy_distribution

end candy_distribution_l4135_413522


namespace f_neg_l4135_413565

-- Define an odd function f on the real numbers
def f : ℝ → ℝ := sorry

-- Define the property of f being odd
axiom f_odd : ∀ x : ℝ, f (-x) = -f x

-- Define f for positive x
axiom f_pos : ∀ x : ℝ, x > 0 → f x = x^2 + 1

-- Theorem to prove
theorem f_neg : ∀ x : ℝ, x < 0 → f x = -x^2 - 1 := by sorry

end f_neg_l4135_413565


namespace men_who_left_l4135_413530

/-- Given a hostel with provisions for a certain number of men and days,
    calculate the number of men who left if the provisions last longer. -/
theorem men_who_left (initial_men : ℕ) (initial_days : ℕ) (new_days : ℕ) :
  initial_men = 250 →
  initial_days = 32 →
  new_days = 40 →
  ∃ (men_left : ℕ),
    men_left = 50 ∧
    initial_men * initial_days = (initial_men - men_left) * new_days :=
by sorry

end men_who_left_l4135_413530


namespace roots_of_polynomial_l4135_413539

theorem roots_of_polynomial (a b c : ℝ) : 
  (∀ x : ℝ, x^5 + 2*x^4 + a*x^2 + b*x = c ↔ x = -1 ∨ x = 1) →
  a = -6 ∧ b = -1 ∧ c = -4 := by
  sorry

end roots_of_polynomial_l4135_413539


namespace sqrt_square_abs_l4135_413554

theorem sqrt_square_abs (x : ℝ) : Real.sqrt (x ^ 2) = |x| := by
  sorry

end sqrt_square_abs_l4135_413554


namespace restaurant_bill_division_l4135_413536

theorem restaurant_bill_division (total_bill : ℝ) (num_people : ℕ) (individual_share : ℝ) :
  total_bill = 135 →
  num_people = 3 →
  individual_share = total_bill / num_people →
  individual_share = 45 := by
  sorry

end restaurant_bill_division_l4135_413536


namespace second_largest_number_l4135_413547

theorem second_largest_number (A B C D : ℕ) : 
  A = 3 * 3 →
  C = 4 * A →
  B = C - 15 →
  D = A + 19 →
  (C > D ∧ D > B ∧ B > A) :=
by sorry

end second_largest_number_l4135_413547


namespace basketball_lineup_count_l4135_413597

/-- The number of ways to choose a lineup from a basketball team --/
def choose_lineup (team_size : ℕ) (lineup_size : ℕ) : ℕ :=
  (team_size - lineup_size + 1).factorial / (team_size - lineup_size).factorial

/-- Theorem: The number of ways to choose a lineup of 6 players from a team of 15 is 3,603,600 --/
theorem basketball_lineup_count :
  choose_lineup 15 6 = 3603600 := by
  sorry

end basketball_lineup_count_l4135_413597


namespace perpendicular_lines_from_perpendicular_planes_l4135_413571

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the relations
variable (distinct : Line → Line → Prop)
variable (distinct_plane : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane : Plane → Plane → Prop)
variable (perpendicular_line : Line → Line → Prop)

-- Theorem statement
theorem perpendicular_lines_from_perpendicular_planes
  (m n : Line) (α β : Plane)
  (h1 : distinct m n)
  (h2 : distinct_plane α β)
  (h3 : perpendicular_plane α β)
  (h4 : perpendicular_line_plane m α)
  (h5 : perpendicular_line_plane n β) :
  perpendicular_line m n :=
sorry

end perpendicular_lines_from_perpendicular_planes_l4135_413571


namespace sum_nine_is_negative_fiftyfour_l4135_413587

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  first_term : a 1 = 2
  fifth_term : a 5 = 3 * a 3
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * (2 * seq.a 1 + (n - 1) * (seq.a 2 - seq.a 1)) / 2

/-- Theorem: The sum of the first 9 terms of the given arithmetic sequence is -54 -/
theorem sum_nine_is_negative_fiftyfour (seq : ArithmeticSequence) : sum_n seq 9 = -54 := by
  sorry

end sum_nine_is_negative_fiftyfour_l4135_413587


namespace smallest_power_congruence_l4135_413544

theorem smallest_power_congruence (h : 2015 = 5 * 13 * 31) :
  (∃ n : ℕ, n > 0 ∧ 2^n ≡ 1 [ZMOD 2015]) ∧
  (∀ m : ℕ, m > 0 ∧ 2^m ≡ 1 [ZMOD 2015] → m ≥ 60) ∧
  2^60 ≡ 1 [ZMOD 2015] := by
  sorry

end smallest_power_congruence_l4135_413544


namespace triangle_ratio_l4135_413510

/-- Given a triangle ABC with angle A = 60°, side b = 1, and area = √3,
    prove that (a+b+c)/(sin A + sin B + sin C) = 2√39/3 -/
theorem triangle_ratio (a b c A B C : ℝ) : 
  A = π/3 → 
  b = 1 → 
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 2 * Real.sqrt 39 / 3 := by
  sorry

end triangle_ratio_l4135_413510


namespace modulus_z_l4135_413568

theorem modulus_z (z : ℂ) (h : z * (1 + 2*I) = 4 + 3*I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end modulus_z_l4135_413568


namespace range_of_m_l4135_413506

theorem range_of_m (x m : ℝ) : 
  (2 * x - m ≤ 3 ∧ -5 < x ∧ x < 4) ↔ m ≥ 5 :=
sorry

end range_of_m_l4135_413506


namespace sum_of_x_and_y_is_four_l4135_413514

theorem sum_of_x_and_y_is_four (x y : ℝ) 
  (eq1 : 4 * x - y = 3) 
  (eq2 : x + 6 * y = 17) : 
  x + y = 4 := by
sorry

end sum_of_x_and_y_is_four_l4135_413514


namespace water_left_for_fourth_neighborhood_l4135_413512

-- Define the total capacity of the water tower
def total_capacity : ℕ := 1200

-- Define the water usage of the first neighborhood
def first_neighborhood_usage : ℕ := 150

-- Define the water usage of the second neighborhood
def second_neighborhood_usage : ℕ := 2 * first_neighborhood_usage

-- Define the water usage of the third neighborhood
def third_neighborhood_usage : ℕ := second_neighborhood_usage + 100

-- Define the total usage of the first three neighborhoods
def total_usage : ℕ := first_neighborhood_usage + second_neighborhood_usage + third_neighborhood_usage

-- Theorem to prove
theorem water_left_for_fourth_neighborhood :
  total_capacity - total_usage = 350 := by sorry

end water_left_for_fourth_neighborhood_l4135_413512


namespace consecutive_numbers_sum_l4135_413557

theorem consecutive_numbers_sum (n : ℕ) : 
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 105) → 
  ((n + 5) - n = 5) := by
  sorry

end consecutive_numbers_sum_l4135_413557


namespace calculation_proof_l4135_413594

theorem calculation_proof :
  ((-56 * (-3/8)) / (-1 - 2/5) = -15) ∧
  ((-12) / (-4) * (1/4) = 3/4) := by
sorry

end calculation_proof_l4135_413594


namespace power_sum_fifth_l4135_413599

theorem power_sum_fifth (a b x y : ℝ) 
  (h1 : a*x + b*y = 1)
  (h2 : a*x^2 + b*y^2 = 9)
  (h3 : a*x^3 + b*y^3 = 28)
  (h4 : a*x^4 + b*y^4 = 96) :
  a*x^5 + b*y^5 = 28616 := by
  sorry

end power_sum_fifth_l4135_413599


namespace average_marks_l4135_413523

theorem average_marks (total_subjects : ℕ) (subjects_avg : ℕ) (last_subject_mark : ℕ) :
  total_subjects = 6 →
  subjects_avg = 74 →
  last_subject_mark = 110 →
  (subjects_avg * (total_subjects - 1) + last_subject_mark) / total_subjects = 80 :=
by sorry

end average_marks_l4135_413523


namespace min_distance_tan_intersection_l4135_413540

theorem min_distance_tan_intersection (a : ℝ) : 
  let f (x : ℝ) := Real.tan (2 * x - π / 3)
  let g (x : ℝ) := -a
  ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧ 
    f x₁ = g x₁ ∧ 
    f x₂ = g x₂ ∧
    ∀ (y : ℝ), x₁ < y ∧ y < x₂ → f y ≠ g y ∧
    x₂ - x₁ = π / 2 ∧
    ∀ (z₁ z₂ : ℝ), (f z₁ = g z₁ ∧ f z₂ = g z₂ ∧ z₁ < z₂) → z₂ - z₁ ≥ π / 2 :=
by sorry

end min_distance_tan_intersection_l4135_413540


namespace hyperbola_focal_property_l4135_413518

/-- The hyperbola with equation x^2 - y^2/9 = 1 -/
def Hyperbola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 - p.2^2/9 = 1}

/-- The left focus of the hyperbola -/
def F₁ : ℝ × ℝ := sorry

/-- The right focus of the hyperbola -/
def F₂ : ℝ × ℝ := sorry

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem hyperbola_focal_property (P : ℝ × ℝ) (h_P : P ∈ Hyperbola) 
    (h_dist : distance P F₁ = 5) : 
  distance P F₂ = 3 ∨ distance P F₂ = 7 := by
  sorry

end hyperbola_focal_property_l4135_413518


namespace coupon1_best_l4135_413532

def coupon1_discount (x : ℝ) : ℝ := 0.15 * x

def coupon2_discount (x : ℝ) : ℝ := 30

def coupon3_discount (x : ℝ) : ℝ := 0.25 * (x - 150)

theorem coupon1_best (x : ℝ) (h1 : x > 100) : 
  (coupon1_discount x > coupon2_discount x ∧ coupon1_discount x > coupon3_discount x) ↔ 
  (200 < x ∧ x < 375) := by sorry

end coupon1_best_l4135_413532


namespace chord_length_limit_l4135_413584

theorem chord_length_limit (r : ℝ) (chord_length : ℝ) :
  r = 6 →
  chord_length ≤ 2 * r →
  chord_length ≠ 14 :=
by sorry

end chord_length_limit_l4135_413584


namespace existence_of_symmetric_axis_l4135_413549

/-- Represents the color of a stone -/
inductive Color
| Black
| White

/-- Represents a regular 13-gon with colored stones at each vertex -/
def Regular13Gon := Fin 13 → Color

/-- Counts the number of symmetric pairs with the same color for a given axis -/
def symmetricPairsCount (polygon : Regular13Gon) (axis : Fin 13) : ℕ :=
  sorry

/-- Main theorem: There exists an axis with at least 4 symmetric pairs of the same color -/
theorem existence_of_symmetric_axis (polygon : Regular13Gon) :
  ∃ axis : Fin 13, symmetricPairsCount polygon axis ≥ 4 := by
  sorry

end existence_of_symmetric_axis_l4135_413549


namespace particular_propositions_count_l4135_413556

-- Define a proposition type
inductive Proposition
| ExistsDivisorImpossible
| PrismIsPolyhedron
| AllEquationsHaveRealSolutions
| SomeTrianglesAreAcute

-- Define a function to check if a proposition is particular
def isParticular (p : Proposition) : Bool :=
  match p with
  | Proposition.ExistsDivisorImpossible => true
  | Proposition.PrismIsPolyhedron => false
  | Proposition.AllEquationsHaveRealSolutions => false
  | Proposition.SomeTrianglesAreAcute => true

-- Define the list of all propositions
def allPropositions : List Proposition :=
  [Proposition.ExistsDivisorImpossible, Proposition.PrismIsPolyhedron,
   Proposition.AllEquationsHaveRealSolutions, Proposition.SomeTrianglesAreAcute]

-- Theorem: The number of particular propositions is 2
theorem particular_propositions_count :
  (allPropositions.filter isParticular).length = 2 := by
  sorry

end particular_propositions_count_l4135_413556


namespace town_population_distribution_l4135_413551

theorem town_population_distribution (total_population : ℕ) 
  (h1 : total_population = 600) 
  (h2 : ∃ (males females children : ℕ), 
    males + females + children = total_population ∧ 
    children = 2 * males ∧ 
    males + females + children = 4 * males) : 
  ∃ (males : ℕ), males = 150 := by
sorry

end town_population_distribution_l4135_413551


namespace f_min_value_f_min_at_9_2_l4135_413555

/-- The function f(x, y) defined in the problem -/
def f (x y : ℝ) : ℝ := x^2 + 6*y^2 - 2*x*y - 14*x - 6*y + 72

/-- Theorem stating that f(x, y) has a minimum value of 3 -/
theorem f_min_value (x y : ℝ) : f x y ≥ 3 := by sorry

/-- Theorem stating that f(9, 2) achieves the minimum value -/
theorem f_min_at_9_2 : f 9 2 = 3 := by sorry

end f_min_value_f_min_at_9_2_l4135_413555


namespace min_value_theorem_l4135_413578

/-- Given that the solution set of (x+2)/(x+1) < 0 is {x | a < x < b},
    and point A(a,b) lies on the line mx + ny + 1 = 0 where mn > 0,
    prove that the minimum value of 2/m + 1/n is 9. -/
theorem min_value_theorem (a b m n : ℝ) : 
  (∀ x, (x + 2) / (x + 1) < 0 ↔ a < x ∧ x < b) →
  m * a + n * b + 1 = 0 →
  m * n > 0 →
  (∀ m' n', m' * n' > 0 → 2 / m' + 1 / n' ≥ 2 / m + 1 / n) →
  2 / m + 1 / n = 9 :=
by sorry

end min_value_theorem_l4135_413578


namespace twelve_divisor_number_is_1989_l4135_413562

/-- The type of natural numbers with exactly 12 positive divisors. -/
def TwelveDivisorNumber (N : ℕ) : Prop :=
  (∃ (d : Fin 12 → ℕ), 
    (∀ i j, i < j → d i < d j) ∧
    (∀ i, d i ∣ N) ∧
    (∀ m, m ∣ N → ∃ i, d i = m) ∧
    (d 0 = 1) ∧
    (d 11 = N))

/-- The property that the divisor with index d₄ - 1 is equal to (d₁ + d₂ + d₄) · d₈ -/
def SpecialDivisorProperty (N : ℕ) (d : Fin 12 → ℕ) : Prop :=
  d ((d 3 : ℕ) - 1) = (d 0 + d 1 + d 3) * d 7

theorem twelve_divisor_number_is_1989 :
  ∃ N : ℕ, TwelveDivisorNumber N ∧ 
    (∃ d : Fin 12 → ℕ, SpecialDivisorProperty N d) ∧
    N = 1989 := by
  sorry

end twelve_divisor_number_is_1989_l4135_413562


namespace range_of_a_l4135_413593

/-- The range of values for a given the conditions -/
theorem range_of_a (f g : ℝ → ℝ) (a : ℝ) : 
  (∀ x > 0, f x = x * Real.log x) →
  (∀ x, g x = x^3 + a*x - x + 2) →
  (∀ x > 0, 2 * f x ≤ deriv g x + 2) →
  a ≥ -2 ∧ ∀ b ≥ -2, ∃ x > 0, 2 * f x ≤ deriv g x + 2 :=
by sorry


end range_of_a_l4135_413593


namespace hyperbola_asymptote_l4135_413560

/-- Given a hyperbola with equation x²/a² - y²/9 = 1 where a > 0,
    if its asymptotes are given by 2x ± 3y = 0, then a = 3 -/
theorem hyperbola_asymptote (a : ℝ) (h1 : a > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / 9 = 1 ↔ (2*x + 3*y = 0 ∨ 2*x - 3*y = 0)) →
  a = 3 := by
sorry

end hyperbola_asymptote_l4135_413560


namespace money_sharing_l4135_413592

theorem money_sharing (ken_share tony_share total : ℕ) : 
  ken_share = 1750 →
  tony_share = 2 * ken_share →
  total = ken_share + tony_share →
  total = 5250 := by
sorry

end money_sharing_l4135_413592


namespace perpendicular_line_equation_equal_intercepts_line_equation_l4135_413575

-- Define the types for points and lines
def Point := ℝ × ℝ
def Line := ℝ → ℝ → Prop

-- Define the intersection point of two lines
def intersection (l1 l2 : Line) : Point :=
  sorry

-- Define perpendicularity of two lines
def perpendicular (l1 l2 : Line) : Prop :=
  sorry

-- Define a line passing through a point
def passes_through (l : Line) (p : Point) : Prop :=
  sorry

-- Define a line having equal intercepts on coordinate axes
def equal_intercepts (l : Line) : Prop :=
  sorry

-- Define the lines given in the problem
def line1 : Line := λ x y ↦ 2*x + 3*y - 9 = 0
def line2 : Line := λ x y ↦ 3*x - y - 8 = 0
def line3 : Line := λ x y ↦ 3*x + 4*y - 1 = 0

-- Part 1
theorem perpendicular_line_equation :
  ∀ l : Line,
  passes_through l (intersection line1 line2) →
  perpendicular l line3 →
  l = λ x y ↦ y = (4/3)*x - 3 :=
sorry

-- Part 2
theorem equal_intercepts_line_equation :
  ∀ l : Line,
  passes_through l (intersection line1 line2) →
  equal_intercepts l →
  (l = λ x y ↦ y = -x + 4) ∨ (l = λ x y ↦ y = (1/3)*x) :=
sorry

end perpendicular_line_equation_equal_intercepts_line_equation_l4135_413575


namespace match_end_probability_l4135_413585

/-- The probability of player A winning a single game -/
def prob_A_win : ℝ := 0.6

/-- The probability of player B winning a single game -/
def prob_B_win : ℝ := 0.4

/-- The probability that the match ends after two more games -/
def prob_match_ends : ℝ := prob_A_win * prob_A_win + prob_B_win * prob_B_win

/-- Theorem stating that the probability of the match ending after two more games is 0.52 -/
theorem match_end_probability : prob_match_ends = 0.52 := by
  sorry

end match_end_probability_l4135_413585


namespace james_total_earnings_l4135_413548

def january_earnings : ℕ := 4000

def february_earnings (jan : ℕ) : ℕ := 2 * jan

def march_earnings (feb : ℕ) : ℕ := feb - 2000

def total_earnings (jan feb mar : ℕ) : ℕ := jan + feb + mar

theorem james_total_earnings :
  total_earnings january_earnings (february_earnings january_earnings) (march_earnings (february_earnings january_earnings)) = 18000 := by
  sorry

end james_total_earnings_l4135_413548


namespace sum_negative_implies_one_negative_l4135_413577

theorem sum_negative_implies_one_negative (a b : ℚ) : a + b < 0 → a < 0 ∨ b < 0 := by
  sorry

end sum_negative_implies_one_negative_l4135_413577


namespace family_tickets_count_l4135_413589

theorem family_tickets_count :
  let adult_ticket_cost : ℕ := 19
  let child_ticket_cost : ℕ := 13
  let adult_count : ℕ := 2
  let child_count : ℕ := 3
  let total_cost : ℕ := 77
  adult_ticket_cost = child_ticket_cost + 6 ∧
  total_cost = adult_count * adult_ticket_cost + child_count * child_ticket_cost →
  adult_count + child_count = 5 :=
by
  sorry

end family_tickets_count_l4135_413589


namespace largest_square_with_four_lattice_points_l4135_413538

/-- A point (x, y) is a lattice point if both x and y are integers. -/
def isLatticePoint (p : ℝ × ℝ) : Prop :=
  Int.floor p.1 = p.1 ∧ Int.floor p.2 = p.2

/-- A square contains exactly four lattice points in its interior. -/
def squareContainsFourLatticePoints (s : Set (ℝ × ℝ)) : Prop :=
  ∃ (p₁ p₂ p₃ p₄ : ℝ × ℝ), p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
  isLatticePoint p₁ ∧ isLatticePoint p₂ ∧ isLatticePoint p₃ ∧ isLatticePoint p₄ ∧
  (∀ p ∈ s, isLatticePoint p → p = p₁ ∨ p = p₂ ∨ p = p₃ ∨ p = p₄)

/-- The theorem statement -/
theorem largest_square_with_four_lattice_points :
  ∃ (s : Set (ℝ × ℝ)), squareContainsFourLatticePoints s ∧
  (∀ (t : Set (ℝ × ℝ)), squareContainsFourLatticePoints t → MeasureTheory.volume s ≥ MeasureTheory.volume t) ∧
  MeasureTheory.volume s = 8 :=
sorry

end largest_square_with_four_lattice_points_l4135_413538


namespace led_messages_count_l4135_413566

/-- Represents the number of LEDs in the row -/
def n : ℕ := 7

/-- Represents the number of LEDs that are lit -/
def k : ℕ := 3

/-- Represents the number of color options for each lit LED -/
def colors : ℕ := 2

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- Calculates the number of ways to arrange k non-adjacent items in n+1 slots -/
def nonAdjacentArrangements (n k : ℕ) : ℕ := choose (n + 1 - k) k

/-- Calculates the total number of different messages -/
def totalMessages : ℕ := nonAdjacentArrangements n k * colors^k

theorem led_messages_count : totalMessages = 80 := by
  sorry

end led_messages_count_l4135_413566


namespace polynomial_composition_pairs_l4135_413561

theorem polynomial_composition_pairs :
  ∀ (a b : ℝ),
    (∃ (P : ℝ → ℝ),
      (∀ x, P (P x) = x^4 - 8*x^3 + a*x^2 + b*x + 40) ∧
      (∃ (c d : ℝ), ∀ x, P x = x^2 + c*x + d)) ↔
    ((a = 28 ∧ b = -48) ∨ (a = 2 ∧ b = 56)) :=
by sorry

end polynomial_composition_pairs_l4135_413561


namespace odd_k_triple_f_35_l4135_413550

def f (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 5 else n - 2

theorem odd_k_triple_f_35 (k : ℤ) (h1 : k % 2 = 1) (h2 : f (f (f k)) = 35) : k = 29 := by
  sorry

end odd_k_triple_f_35_l4135_413550


namespace stating_escalator_steps_l4135_413537

/-- Represents the total number of steps on an escalator -/
def total_steps : ℕ := 40

/-- Represents the number of steps I ascend on the moving escalator -/
def my_steps : ℕ := 20

/-- Represents the time I take to ascend the escalator in seconds -/
def my_time : ℕ := 60

/-- Represents the number of steps my wife ascends on the moving escalator -/
def wife_steps : ℕ := 16

/-- Represents the time my wife takes to ascend the escalator in seconds -/
def wife_time : ℕ := 72

/-- 
Theorem stating that the total number of steps on the escalator is 40,
given the conditions about my ascent and my wife's ascent.
-/
theorem escalator_steps : 
  (total_steps - my_steps) / my_time = (total_steps - wife_steps) / wife_time :=
sorry

end stating_escalator_steps_l4135_413537


namespace factorial_prime_factorization_l4135_413507

theorem factorial_prime_factorization :
  ∃ (i k m p : ℕ+),
    (8 : ℕ).factorial = 2^(i.val) * 3^(k.val) * 5^(m.val) * 7^(p.val) ∧
    i.val + k.val + m.val + p.val = 11 := by
  sorry

end factorial_prime_factorization_l4135_413507


namespace intersection_M_N_l4135_413588

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = x^2 + 1}
def N : Set ℝ := {y | ∃ x, y = x + 1}

-- State the theorem
theorem intersection_M_N : M ∩ N = {y | y ≥ 1} := by sorry

end intersection_M_N_l4135_413588


namespace axis_triangle_line_equation_l4135_413511

/-- A line passing through a point and forming a triangle with the axes --/
structure AxisTriangleLine where
  /-- The slope of the line --/
  k : ℝ
  /-- The line passes through the point (1, 2) --/
  passes_through : k * (1 - 0) = 2 - 0
  /-- The slope is negative --/
  negative_slope : k < 0
  /-- The area of the triangle formed with the axes is 4 --/
  triangle_area : (1/2) * (2 - k) * (1 - 2/k) = 4

/-- The equation of the line is 2x + y - 4 = 0 --/
theorem axis_triangle_line_equation (l : AxisTriangleLine) : 
  ∃ (a b c : ℝ), a * 1 + b * 2 + c = 0 ∧ 
                  ∀ x y, a * x + b * y + c = 0 ↔ y - 2 = l.k * (x - 1) :=
sorry

end axis_triangle_line_equation_l4135_413511


namespace probability_x_less_than_y_l4135_413503

-- Define the rectangle
def rectangle : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 4 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define the condition x < y
def condition (p : ℝ × ℝ) : Prop := p.1 < p.2

-- Define the probability measure on the rectangle
noncomputable def prob : MeasureTheory.ProbabilityMeasure (ℝ × ℝ) :=
  sorry

-- State the theorem
theorem probability_x_less_than_y :
  prob {p ∈ rectangle | condition p} = 1/8 := by sorry

end probability_x_less_than_y_l4135_413503


namespace sqrt_2_irrational_l4135_413534

theorem sqrt_2_irrational : Irrational (Real.sqrt 2) := by
  sorry

end sqrt_2_irrational_l4135_413534


namespace circle_center_from_axis_intersections_l4135_413580

/-- Given a circle that intersects the x-axis at (a, 0) and (b, 0),
    and the y-axis at (0, c) and (0, d), its center is at ((a+b)/2, (c+d)/2) -/
theorem circle_center_from_axis_intersections 
  (a b c d : ℝ) : 
  ∃ (center : ℝ × ℝ),
    (∃ (circle : Set (ℝ × ℝ)), 
      (a, 0) ∈ circle ∧ 
      (b, 0) ∈ circle ∧ 
      (0, c) ∈ circle ∧ 
      (0, d) ∈ circle ∧
      center = ((a + b) / 2, (c + d) / 2) ∧
      ∀ p ∈ circle, (p.1 - center.1)^2 + (p.2 - center.2)^2 = 
        (a - center.1)^2 + (0 - center.2)^2) :=
by
  sorry

end circle_center_from_axis_intersections_l4135_413580


namespace janet_paper_clips_used_l4135_413582

/-- Calculates the number of paper clips Janet used during the day -/
def paperClipsUsed (initial : ℝ) (found : ℝ) (givenPerFriend : ℝ) (numFriends : ℕ) (final : ℝ) : ℝ :=
  initial + found - givenPerFriend * (numFriends : ℝ) - final

/-- Theorem stating that Janet used 62.5 paper clips during the day -/
theorem janet_paper_clips_used :
  paperClipsUsed 85 17.5 3.5 4 26 = 62.5 := by
  sorry

#eval paperClipsUsed 85 17.5 3.5 4 26

end janet_paper_clips_used_l4135_413582


namespace mike_marbles_l4135_413552

theorem mike_marbles (given_away : ℕ) (remaining : ℕ) : 
  given_away = 4 → remaining = 4 → given_away + remaining = 8 := by
  sorry

end mike_marbles_l4135_413552


namespace min_value_2a_plus_b_l4135_413558

theorem min_value_2a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : ∃ x : ℝ, x^2 + 2*a*x + 3*b = 0)
  (h2 : ∃ x : ℝ, x^2 + 3*b*x + 2*a = 0) :
  2*a + b ≥ 2 * Real.sqrt (3 * Real.rpow (8/3) (1/3)) + Real.rpow (8/3) (1/3) :=
sorry

end min_value_2a_plus_b_l4135_413558


namespace planes_parallel_l4135_413513

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (perp : Plane → Plane → Prop)
variable (line_parallel : Line → Line → Prop)
variable (line_perp : Line → Plane → Prop)

-- Define the objects
variable (α β γ : Plane) (a b : Line)

-- State the theorem
theorem planes_parallel 
  (h1 : parallel α γ)
  (h2 : parallel β γ)
  (h3 : line_perp a α)
  (h4 : line_perp b β)
  (h5 : line_parallel a b) :
  parallel α β :=
sorry

end planes_parallel_l4135_413513


namespace shirley_eggs_l4135_413517

theorem shirley_eggs (initial_eggs : ℕ) (bought_eggs : ℕ) : 
  initial_eggs = 98 → bought_eggs = 8 → initial_eggs + bought_eggs = 106 := by
  sorry

end shirley_eggs_l4135_413517


namespace sum_of_even_integers_l4135_413553

theorem sum_of_even_integers (n : ℕ) (sum_first_n : ℕ) (first : ℕ) (last : ℕ) :
  n = 50 →
  sum_first_n = 2550 →
  first = 102 →
  last = 200 →
  (n : ℕ) * (2 + 2 * n) = 2 * sum_first_n →
  (last - first) / 2 + 1 = n →
  n / 2 * (first + last) = 7550 :=
by sorry

end sum_of_even_integers_l4135_413553


namespace piper_wing_count_l4135_413528

/-- The number of commercial planes in the air exhibition -/
def num_planes : ℕ := 45

/-- The number of wings on each commercial plane -/
def wings_per_plane : ℕ := 2

/-- The total number of wings counted by Piper -/
def total_wings : ℕ := num_planes * wings_per_plane

theorem piper_wing_count : total_wings = 90 := by
  sorry

end piper_wing_count_l4135_413528


namespace negative_of_negative_two_equals_two_l4135_413572

theorem negative_of_negative_two_equals_two : -(-2) = 2 := by sorry

end negative_of_negative_two_equals_two_l4135_413572


namespace line_slope_is_one_l4135_413546

/-- The slope of a line in the xy-plane with y-intercept -2 and passing through 
    the midpoint of the line segment with endpoints (2, 8) and (8, -2) is 1. -/
theorem line_slope_is_one : 
  ∀ (m : Set (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ m → y = x - 2) →  -- y-intercept is -2
    ((5 : ℝ), 3) ∈ m →  -- passes through midpoint (5, 3)
    (∃ (k : ℝ), ∀ (x y : ℝ), (x, y) ∈ m → y = k * x - 2) →  -- line equation
    (∀ (x y : ℝ), (x, y) ∈ m → y = x - 2) :=  -- slope is 1
by sorry

end line_slope_is_one_l4135_413546


namespace line_point_theorem_l4135_413598

/-- The line equation y = -2/3x + 10 -/
def line_equation (x y : ℝ) : Prop := y = -2/3 * x + 10

/-- Point P is where the line crosses the x-axis -/
def point_P : ℝ × ℝ := (15, 0)

/-- Point Q is where the line crosses the y-axis -/
def point_Q : ℝ × ℝ := (0, 10)

/-- Point T is on the line segment PQ -/
def point_T_on_PQ (r s : ℝ) : Prop :=
  ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ 
  r = t * point_P.1 + (1 - t) * point_Q.1 ∧
  s = t * point_P.2 + (1 - t) * point_Q.2

/-- The area of triangle POQ is four times the area of triangle TOP -/
def area_condition (r s : ℝ) : Prop :=
  abs (point_P.1 * point_Q.2 - point_Q.1 * point_P.2) / 2 = 
  4 * abs (r * point_P.2 - point_P.1 * s) / 2

/-- Main theorem -/
theorem line_point_theorem (r s : ℝ) :
  line_equation r s →
  point_T_on_PQ r s →
  area_condition r s →
  r + s = 13.75 := by sorry

end line_point_theorem_l4135_413598


namespace quadratic_roots_property_l4135_413574

theorem quadratic_roots_property (d e : ℝ) : 
  (3 * d^2 + 5 * d - 7 = 0) → 
  (3 * e^2 + 5 * e - 7 = 0) → 
  (d - 2) * (e - 2) = 5 := by
sorry

end quadratic_roots_property_l4135_413574


namespace pulley_centers_distance_l4135_413500

theorem pulley_centers_distance (r₁ r₂ contact_distance : ℝ) 
  (h₁ : r₁ = 10)
  (h₂ : r₂ = 6)
  (h₃ : contact_distance = 30) :
  Real.sqrt ((contact_distance ^ 2) + ((r₁ - r₂) ^ 2)) = 2 * Real.sqrt 229 := by
  sorry

end pulley_centers_distance_l4135_413500


namespace unique_solution_l4135_413515

/-- Represents a five-digit number -/
def FiveDigitNumber := { n : ℕ // 10000 ≤ n ∧ n < 100000 }

/-- Represents a four-digit number -/
def FourDigitNumber := { n : ℕ // 1000 ≤ n ∧ n < 10000 }

/-- Given a five-digit number, returns all possible four-digit numbers
    that can be formed by removing one digit -/
def removeSingleDigit (n : FiveDigitNumber) : Set FourDigitNumber :=
  sorry

/-- The property that defines our solution -/
def isSolution (n : FiveDigitNumber) : Prop :=
  ∃ (m : FourDigitNumber), m ∈ removeSingleDigit n ∧ n.val + m.val = 54321

/-- Theorem stating that 49383 is the unique solution -/
theorem unique_solution :
  ∃! (n : FiveDigitNumber), isSolution n ∧ n.val = 49383 :=
sorry

end unique_solution_l4135_413515


namespace appetizers_needed_l4135_413526

/-- Represents the number of appetizers per guest -/
def appetizers_per_guest : ℕ := 6

/-- Represents the number of guests -/
def number_of_guests : ℕ := 30

/-- Represents the number of dozens of deviled eggs prepared -/
def dozens_deviled_eggs : ℕ := 3

/-- Represents the number of dozens of pigs in a blanket prepared -/
def dozens_pigs_in_blanket : ℕ := 2

/-- Represents the number of dozens of kebabs prepared -/
def dozens_kebabs : ℕ := 2

/-- Represents the number of items in a dozen -/
def items_per_dozen : ℕ := 12

/-- Theorem stating that Patsy needs to make 8 more dozen appetizers -/
theorem appetizers_needed : 
  (appetizers_per_guest * number_of_guests - 
   (dozens_deviled_eggs + dozens_pigs_in_blanket + dozens_kebabs) * items_per_dozen) / 
  items_per_dozen = 8 := by
  sorry

end appetizers_needed_l4135_413526


namespace geometry_relations_l4135_413533

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Line → Prop)
variable (perp_line_plane : Line → Plane → Prop)
variable (perp_plane : Plane → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (parallel_line : Line → Line → Prop)

-- Define the lines and planes
variable (l m : Line) (α β γ : Plane)

-- State the theorem
theorem geometry_relations :
  (perp l m ∧ perp_line_plane l α ∧ perp_line_plane m β → perp_plane α β) ∧
  (parallel α β ∧ parallel β γ → parallel α γ) ∧
  (perp_line_plane l α ∧ parallel α β → perp_line_plane l β) ∧
  (perp_line_plane l α ∧ perp_line_plane m α → parallel_line l m) :=
by sorry

end geometry_relations_l4135_413533


namespace relationship_proof_l4135_413545

open Real

noncomputable def f (x : ℝ) := Real.exp x + x - 2
noncomputable def g (x : ℝ) := Real.log x + x^2 - 3

theorem relationship_proof (a b : ℝ) (ha : f a = 0) (hb : g b = 0) :
  g a < 0 ∧ 0 < f b := by sorry

end relationship_proof_l4135_413545


namespace triangle_exists_l4135_413525

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_exists : can_form_triangle 8 6 4 := by
  sorry

end triangle_exists_l4135_413525


namespace salt_solution_dilution_l4135_413573

/-- Proves that the initial volume of a 20% salt solution is 90 liters,
    given that adding 30 liters of water dilutes it to a 15% salt solution. -/
theorem salt_solution_dilution (initial_volume : ℝ) : 
  (0.20 * initial_volume = 0.15 * (initial_volume + 30)) → 
  initial_volume = 90 := by
  sorry

end salt_solution_dilution_l4135_413573


namespace solve_commencement_addresses_l4135_413527

def commencement_addresses_problem (sandoval hawkins sloan : ℕ) : Prop :=
  sandoval = 12 ∧
  hawkins = sandoval / 2 ∧
  sloan > sandoval ∧
  sandoval + hawkins + sloan = 40 ∧
  sloan - sandoval = 10

theorem solve_commencement_addresses :
  ∃ (sandoval hawkins sloan : ℕ), commencement_addresses_problem sandoval hawkins sloan :=
by
  sorry

end solve_commencement_addresses_l4135_413527


namespace tino_jellybean_count_l4135_413569

/-- The number of jellybeans each person has -/
structure JellybeanCount where
  tino : ℕ
  lee : ℕ
  arnold : ℕ

/-- The conditions of the jellybean problem -/
def jellybean_conditions (j : JellybeanCount) : Prop :=
  j.tino = j.lee + 24 ∧
  j.arnold = j.lee / 2 ∧
  j.arnold = 5

/-- Theorem stating that under the given conditions, Tino has 34 jellybeans -/
theorem tino_jellybean_count (j : JellybeanCount) 
  (h : jellybean_conditions j) : j.tino = 34 := by
  sorry

end tino_jellybean_count_l4135_413569


namespace not_necessarily_divisible_by_48_l4135_413504

theorem not_necessarily_divisible_by_48 (k : ℤ) :
  let n := k * (k + 1) * (k + 2) * (k + 3)
  ∃ (n : ℤ), (8 ∣ n) ∧ ¬(48 ∣ n) := by
  sorry

end not_necessarily_divisible_by_48_l4135_413504


namespace circle_center_sum_l4135_413579

/-- Given a circle with equation x^2 + y^2 - 6x + 8y - 24 = 0, 
    prove that the sum of the coordinates of its center is -1 -/
theorem circle_center_sum (h k : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 6*x + 8*y - 24 = 0 ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 24 : ℝ)) →
  h + k = -1 := by
sorry

end circle_center_sum_l4135_413579


namespace boat_speed_in_still_water_l4135_413535

/-- A boat traveling downstream with the help of a stream. -/
structure BoatTrip where
  boat_speed : ℝ      -- Speed of the boat in still water (km/hr)
  stream_speed : ℝ    -- Speed of the stream (km/hr)
  time : ℝ            -- Time taken for the trip (hours)
  distance : ℝ        -- Distance traveled (km)

/-- The theorem stating the boat's speed in still water given the conditions. -/
theorem boat_speed_in_still_water (trip : BoatTrip)
  (h1 : trip.stream_speed = 5)
  (h2 : trip.time = 5)
  (h3 : trip.distance = 135) :
  trip.boat_speed = 22 := by
  sorry

end boat_speed_in_still_water_l4135_413535


namespace no_solutions_abs_equation_l4135_413524

theorem no_solutions_abs_equation : ¬∃ y : ℝ, |y - 2| = |y - 1| + |y - 4| := by
  sorry

end no_solutions_abs_equation_l4135_413524


namespace arithmetic_sequence_60th_term_l4135_413564

/-- Given an arithmetic sequence with first term 7 and 21st term 47, prove that the 60th term is 125 -/
theorem arithmetic_sequence_60th_term : 
  ∀ (a : ℕ → ℝ), 
    (∀ n : ℕ, a (n + 1) - a n = a 1 - a 0) →  -- arithmetic sequence condition
    a 0 = 7 →                                -- first term
    a 20 = 47 →                              -- 21st term (index starts at 0)
    a 59 = 125 :=                            -- 60th term (index starts at 0)
by
  sorry

end arithmetic_sequence_60th_term_l4135_413564


namespace second_number_value_l4135_413586

theorem second_number_value (x y z : ℝ) : 
  x + y + z = 660 ∧ 
  x = 2 * y ∧ 
  z = (1 / 3) * x → 
  y = 180 := by
  sorry

end second_number_value_l4135_413586


namespace inequality_proof_l4135_413519

theorem inequality_proof (x y z a b : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x * y * z = 1) :
  ((a = 0 ∧ b = 1) ∨ (a = 1 ∧ b = 0) ∨ (a + b = 1 ∧ a > 0 ∧ b > 0)) →
  (1 / (x * (a * y + b)) + 1 / (y * (a * z + b)) + 1 / (z * (a * x + b)) ≥ 3) ∧
  (x = 1 ∧ y = 1 ∧ z = 1 → 1 / (x * (a * y + b)) + 1 / (y * (a * z + b)) + 1 / (z * (a * x + b)) = 3) :=
by sorry

end inequality_proof_l4135_413519


namespace largest_square_size_l4135_413563

theorem largest_square_size (board_length board_width : ℕ) 
  (h1 : board_length = 77) (h2 : board_width = 93) :
  Nat.gcd board_length board_width = 1 := by
  sorry

end largest_square_size_l4135_413563


namespace same_gender_probability_l4135_413502

/-- The probability of selecting two students of the same gender -/
theorem same_gender_probability (n_male n_female : ℕ) (h_male : n_male = 2) (h_female : n_female = 8) :
  let total := n_male + n_female
  let same_gender_ways := Nat.choose n_male 2 + Nat.choose n_female 2
  let total_ways := Nat.choose total 2
  (same_gender_ways : ℚ) / total_ways = 29 / 45 := by sorry

end same_gender_probability_l4135_413502
