import Mathlib

namespace number_of_teachers_l3906_390662

/-- Represents the number of students at King Middle School -/
def total_students : ℕ := 1200

/-- Represents the number of classes each student takes per day -/
def classes_per_student : ℕ := 5

/-- Represents the number of classes each teacher teaches -/
def classes_per_teacher : ℕ := 4

/-- Represents the number of students in each class -/
def students_per_class : ℕ := 30

/-- Represents the number of teachers in each class -/
def teachers_per_class : ℕ := 1

/-- Theorem stating that the number of teachers at King Middle School is 50 -/
theorem number_of_teachers : 
  (total_students * classes_per_student) / students_per_class / classes_per_teacher = 50 := by
  sorry

end number_of_teachers_l3906_390662


namespace pies_difference_l3906_390602

/-- The number of pies sold by Smith's Bakery -/
def smiths_pies : ℕ := 70

/-- The number of pies sold by Mcgee's Bakery -/
def mcgees_pies : ℕ := 16

/-- Theorem stating the difference between Smith's pies and four times Mcgee's pies -/
theorem pies_difference : smiths_pies - 4 * mcgees_pies = 6 := by
  sorry

end pies_difference_l3906_390602


namespace triangle_altitude_and_median_l3906_390646

/-- Triangle with vertices A(0,1), B(-2,0), and C(2,0) -/
structure Triangle where
  A : ℝ × ℝ := (0, 1)
  B : ℝ × ℝ := (-2, 0)
  C : ℝ × ℝ := (2, 0)

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Altitude from A to AC -/
def altitude (t : Triangle) : LineEquation :=
  { a := 2, b := -1, c := 1 }

/-- Median from A to BC -/
def median (t : Triangle) : LineEquation :=
  { a := 1, b := 0, c := 0 }

theorem triangle_altitude_and_median (t : Triangle) :
  (altitude t = { a := 2, b := -1, c := 1 }) ∧
  (median t = { a := 1, b := 0, c := 0 }) := by
  sorry

end triangle_altitude_and_median_l3906_390646


namespace peters_horses_l3906_390667

/-- The number of horses Peter has -/
def num_horses : ℕ := 4

/-- The amount of oats each horse eats per feeding -/
def oats_per_feeding : ℕ := 4

/-- The number of oat feedings per day -/
def oat_feedings_per_day : ℕ := 2

/-- The amount of grain each horse eats per day -/
def grain_per_day : ℕ := 3

/-- The number of days Peter feeds his horses -/
def feeding_days : ℕ := 3

/-- The total amount of food Peter needs for all his horses for the given days -/
def total_food : ℕ := 132

theorem peters_horses :
  num_horses * (oats_per_feeding * oat_feedings_per_day + grain_per_day) * feeding_days = total_food :=
by sorry

end peters_horses_l3906_390667


namespace cos_four_arccos_two_fifths_l3906_390609

theorem cos_four_arccos_two_fifths :
  Real.cos (4 * Real.arccos (2/5)) = -47/625 := by
  sorry

end cos_four_arccos_two_fifths_l3906_390609


namespace lg_24_in_terms_of_a_b_l3906_390649

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem statement
theorem lg_24_in_terms_of_a_b (a b : ℝ) (h1 : lg 6 = a) (h2 : lg 12 = b) :
  lg 24 = 2 * b - a := by
  sorry

end lg_24_in_terms_of_a_b_l3906_390649


namespace oil_storage_solution_l3906_390601

/-- Represents the oil storage problem with given constraints --/
def oil_storage_problem (total_oil : ℕ) (large_barrel_capacity : ℕ) (small_barrels_used : ℕ) : Prop :=
  ∃ (small_barrel_capacity : ℕ) (large_barrels_used : ℕ),
    total_oil = large_barrels_used * large_barrel_capacity + small_barrels_used * small_barrel_capacity ∧
    small_barrels_used > 0 ∧
    small_barrel_capacity > 0 ∧
    small_barrel_capacity < large_barrel_capacity ∧
    ∀ (other_large : ℕ) (other_small : ℕ),
      total_oil = other_large * large_barrel_capacity + other_small * small_barrel_capacity →
      other_small ≥ small_barrels_used →
      other_large + other_small ≥ large_barrels_used + small_barrels_used

/-- The solution to the oil storage problem --/
theorem oil_storage_solution :
  oil_storage_problem 95 6 1 →
  ∃ (small_barrel_capacity : ℕ), small_barrel_capacity = 5 := by
  sorry

end oil_storage_solution_l3906_390601


namespace inequalities_satisfaction_l3906_390611

theorem inequalities_satisfaction (a b c x y z : ℝ) 
  (hx : |x| < |a|) (hy : |y| < |b|) (hz : |z| < |c|) : 
  (|x*y| + |y*z| + |z*x| < |a*b| + |b*c| + |c*a|) ∧ 
  (x^2 + z^2 < a^2 + c^2) := by
  sorry

end inequalities_satisfaction_l3906_390611


namespace mean_of_other_two_l3906_390669

def numbers : List ℤ := [2179, 2231, 2307, 2375, 2419, 2433]

def sum_of_all : ℤ := numbers.sum

def mean_of_four : ℤ := 2323

def sum_of_four : ℤ := 4 * mean_of_four

theorem mean_of_other_two (h : sum_of_four = 4 * mean_of_four) :
  (sum_of_all - sum_of_four) / 2 = 2321 := by
  sorry

end mean_of_other_two_l3906_390669


namespace pet_store_combinations_l3906_390617

/-- The number of puppies available in the pet store -/
def num_puppies : ℕ := 10

/-- The number of kittens available in the pet store -/
def num_kittens : ℕ := 6

/-- The number of hamsters available in the pet store -/
def num_hamsters : ℕ := 8

/-- The total number of ways Alice, Bob, and Charlie can buy pets and leave the store satisfied -/
def total_ways : ℕ := 960

/-- Theorem stating that the number of ways Alice, Bob, and Charlie can buy pets
    and leave the store satisfied is equal to total_ways -/
theorem pet_store_combinations :
  (num_puppies * num_kittens * num_hamsters) +
  (num_kittens * num_puppies * num_hamsters) = total_ways :=
by sorry

end pet_store_combinations_l3906_390617


namespace inequality_proof_l3906_390697

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z > 0) :
  (x^2 * y) / (y + z) + (y^2 * z) / (z + x) + (z^2 * x) / (x + y) ≥ (1/2) * (x^2 + y^2 + z^2) := by
  sorry

end inequality_proof_l3906_390697


namespace D_l3906_390660

def D' : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | 2 => 0
  | n + 3 => D' (n + 2) + D' (n + 1) + D' n

theorem D'_parity_2024_2025_2026 :
  Even (D' 2024) ∧ Odd (D' 2025) ∧ Odd (D' 2026) :=
by
  sorry

end D_l3906_390660


namespace hyperbola_eccentricity_l3906_390621

/-- Given a hyperbola with equation x²/a² - y²/(4a-2) = 1 and eccentricity √3, prove that a = 1 -/
theorem hyperbola_eccentricity (a : ℝ) :
  (∃ x y : ℝ, x^2 / a^2 - y^2 / (4*a - 2) = 1) →
  (∃ b : ℝ, b^2 = 4*a - 2 ∧ b^2 / a^2 = 2) →
  a = 1 :=
by sorry

end hyperbola_eccentricity_l3906_390621


namespace mother_hubbard_children_l3906_390600

theorem mother_hubbard_children (total_bar : ℚ) (children : ℕ) : 
  total_bar = 1 →
  (total_bar - total_bar / 3) = (children * (total_bar / 12)) →
  children = 8 := by
  sorry

end mother_hubbard_children_l3906_390600


namespace point_y_coordinate_l3906_390657

-- Define the parabola
def is_on_parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define the distance from a point to the focus
def distance_to_focus (x y : ℝ) : ℝ := 4

-- Define the y-coordinate of the directrix
def directrix_y : ℝ := -1

-- Theorem statement
theorem point_y_coordinate (x y : ℝ) :
  is_on_parabola x y →
  distance_to_focus x y = 4 →
  y = 3 := by sorry

end point_y_coordinate_l3906_390657


namespace linear_equation_root_conditions_l3906_390636

/-- Conditions for roots of a linear equation -/
theorem linear_equation_root_conditions (a b : ℝ) :
  let x := -b / a
  (x > 0 ↔ a * b < 0) ∧
  (x < 0 ↔ a * b > 0) ∧
  (x = 0 ↔ b = 0 ∧ a ≠ 0) :=
by sorry

end linear_equation_root_conditions_l3906_390636


namespace fraction_sum_l3906_390618

theorem fraction_sum (m n : ℚ) (h : n / m = 3 / 7) : (m + n) / m = 10 / 7 := by
  sorry

end fraction_sum_l3906_390618


namespace flight_time_estimate_l3906_390668

/-- The radius of the circular path in miles -/
def radius : ℝ := 3950

/-- The speed of the object in miles per hour -/
def speed : ℝ := 550

/-- The approximate value of π -/
def π_approx : ℝ := 3.14

/-- The theorem stating that the time taken to complete one revolution is approximately 45 hours -/
theorem flight_time_estimate :
  let circumference := 2 * π_approx * radius
  let exact_time := circumference / speed
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |exact_time - 45| < ε :=
sorry

end flight_time_estimate_l3906_390668


namespace missing_number_proof_l3906_390692

theorem missing_number_proof (x : ℝ) : 11 + Real.sqrt (-4 + 6 * x / 3) = 13 → x = 3 := by
  sorry

end missing_number_proof_l3906_390692


namespace shelbys_drive_l3906_390684

/-- Shelby's driving problem -/
theorem shelbys_drive (sunny_speed rainy_speed foggy_speed : ℚ)
  (total_distance total_time : ℚ) (sunny_time rainy_time foggy_time : ℚ) :
  sunny_speed = 35 →
  rainy_speed = 25 →
  foggy_speed = 15 →
  total_distance = 20 →
  total_time = 60 →
  sunny_time + rainy_time + foggy_time = total_time →
  sunny_speed * sunny_time / 60 + rainy_speed * rainy_time / 60 + foggy_speed * foggy_time / 60 = total_distance →
  foggy_time = 45 := by
  sorry

#check shelbys_drive

end shelbys_drive_l3906_390684


namespace B_power_15_minus_3_times_14_l3906_390639

def B : Matrix (Fin 3) (Fin 3) ℝ := !![3, 1, 2; 0, 4, 1; 0, 0, 2]

theorem B_power_15_minus_3_times_14 :
  B^15 - 3 • (B^14) = !![0, 3, 1; 0, 4, 1; 0, 0, -2] := by
  sorry

end B_power_15_minus_3_times_14_l3906_390639


namespace relationship_xyz_l3906_390686

theorem relationship_xyz (x y z : ℝ) 
  (hx : x = Real.log π) 
  (hy : y = Real.log 2 / Real.log 5)
  (hz : z = Real.exp (-1/2)) :
  y < z ∧ z < x := by sorry

end relationship_xyz_l3906_390686


namespace tangent_slope_parabola_l3906_390653

/-- The slope of the tangent line to y = (1/5)x^2 at (2, 4/5) is 4/5 -/
theorem tangent_slope_parabola :
  let f (x : ℝ) := (1/5) * x^2
  let a : ℝ := 2
  let slope := (deriv f) a
  slope = 4/5 := by sorry

end tangent_slope_parabola_l3906_390653


namespace platform_length_l3906_390658

/-- Given a train and platform with specific properties, prove the platform length --/
theorem platform_length
  (train_length : ℝ)
  (time_cross_platform : ℝ)
  (time_cross_pole : ℝ)
  (h1 : train_length = 300)
  (h2 : time_cross_platform = 39)
  (h3 : time_cross_pole = 36) :
  ∃ platform_length : ℝ,
    platform_length = 25 ∧
    (train_length + platform_length) / time_cross_platform = train_length / time_cross_pole :=
by sorry

end platform_length_l3906_390658


namespace strawberry_area_l3906_390685

/-- Given a garden with the following properties:
  * The total area is 64 square feet
  * Half of the garden is for fruits
  * A quarter of the fruit section is for strawberries
  Prove that the area for strawberries is 8 square feet. -/
theorem strawberry_area (garden_area : ℝ) (fruit_ratio : ℝ) (strawberry_ratio : ℝ) : 
  garden_area = 64 → 
  fruit_ratio = 1/2 → 
  strawberry_ratio = 1/4 → 
  garden_area * fruit_ratio * strawberry_ratio = 8 := by
  sorry

end strawberry_area_l3906_390685


namespace divisibility_pairs_l3906_390673

theorem divisibility_pairs : 
  {p : ℕ × ℕ | p.1 ∣ (2^(Nat.totient p.2) + 1) ∧ p.2 ∣ (2^(Nat.totient p.1) + 1)} = 
  {(1, 1), (1, 3), (3, 1)} := by
sorry

end divisibility_pairs_l3906_390673


namespace commercial_reduction_l3906_390631

def original_length : ℝ := 30
def reduction_percentage : ℝ := 0.30

theorem commercial_reduction :
  original_length * (1 - reduction_percentage) = 21 := by
  sorry

end commercial_reduction_l3906_390631


namespace pam_has_1200_apples_l3906_390672

/-- The number of apples in each of Gerald's bags -/
def geralds_bag_count : ℕ := 40

/-- The number of Gerald's bags equivalent to one of Pam's bags -/
def gerald_to_pam_ratio : ℕ := 3

/-- The number of bags Pam has -/
def pams_bag_count : ℕ := 10

/-- The total number of apples Pam has -/
def pams_total_apples : ℕ := pams_bag_count * (gerald_to_pam_ratio * geralds_bag_count)

theorem pam_has_1200_apples : pams_total_apples = 1200 := by
  sorry

end pam_has_1200_apples_l3906_390672


namespace encryption_game_team_sizes_l3906_390613

theorem encryption_game_team_sizes :
  ∀ (num_two num_three num_four num_five : ℕ),
    -- Total number of players
    168 = 2 * num_two + 3 * num_three + 4 * num_four + 5 * num_five →
    -- Total number of teams
    50 = num_two + num_three + num_four + num_five →
    -- Number of three-player teams
    num_three = 20 →
    -- At least one five-player team
    num_five > 0 →
    -- Four is the most common team size
    num_four ≥ num_two ∧ num_four > num_three ∧ num_four > num_five →
    -- Conclusion
    num_two = 7 ∧ num_four = 21 ∧ num_five = 2 := by
  sorry

end encryption_game_team_sizes_l3906_390613


namespace subset_implies_a_values_l3906_390681

def M : Set ℝ := {x | x^2 = 2}
def N (a : ℝ) : Set ℝ := {x | a * x = 1}

theorem subset_implies_a_values (a : ℝ) :
  N a ⊆ M → a = 0 ∨ a = -Real.sqrt 2 / 2 ∨ a = Real.sqrt 2 / 2 := by
  sorry

end subset_implies_a_values_l3906_390681


namespace product_a4b4_l3906_390614

theorem product_a4b4 (a₁ a₂ a₃ a₄ b₁ b₂ b₃ b₄ : ℝ) 
  (eq1 : a₁ * b₁ + a₂ * b₃ = 1)
  (eq2 : a₁ * b₂ + a₂ * b₄ = 0)
  (eq3 : a₃ * b₁ + a₄ * b₃ = 0)
  (eq4 : a₃ * b₂ + a₄ * b₄ = 1)
  (eq5 : a₂ * b₃ = 7) :
  a₄ * b₄ = -6 := by sorry

end product_a4b4_l3906_390614


namespace number_of_students_l3906_390656

theorem number_of_students (student_avg : ℝ) (teacher_age : ℝ) (new_avg : ℝ) :
  student_avg = 26 →
  teacher_age = 52 →
  new_avg = 27 →
  ∃ n : ℕ, (n : ℝ) * student_avg + teacher_age = (n + 1) * new_avg ∧ n = 25 :=
by
  sorry

end number_of_students_l3906_390656


namespace least_common_multiple_addition_l3906_390620

theorem least_common_multiple_addition (a b c d : ℕ) (n m : ℕ) : 
  (∀ k : ℕ, k < m → ¬(a ∣ (n + k) ∧ b ∣ (n + k) ∧ c ∣ (n + k) ∧ d ∣ (n + k))) →
  (a ∣ (n + m) ∧ b ∣ (n + m) ∧ c ∣ (n + m) ∧ d ∣ (n + m)) →
  m = 7 ∧ n = 857 ∧ a = 24 ∧ b = 32 ∧ c = 36 ∧ d = 54 :=
by sorry

end least_common_multiple_addition_l3906_390620


namespace point_in_third_quadrant_m_range_l3906_390688

theorem point_in_third_quadrant_m_range (m : ℝ) : 
  (m - 4 < 0 ∧ 1 - 2*m < 0) → (1/2 < m ∧ m < 4) :=
by sorry

end point_in_third_quadrant_m_range_l3906_390688


namespace median_length_l3906_390670

/-- Triangle ABC with given side lengths and median BM --/
structure Triangle where
  AB : ℝ
  BC : ℝ
  AC : ℝ
  BM : ℝ
  h_AB : AB = 5
  h_BC : BC = 12
  h_AC : AC = 13
  h_BM : ∃ m : ℝ, BM = m * Real.sqrt 2

/-- The value of m in the equation BM = m√2 is 13/2 --/
theorem median_length (t : Triangle) : ∃ m : ℝ, t.BM = m * Real.sqrt 2 ∧ m = 13 / 2 := by
  sorry

end median_length_l3906_390670


namespace smallest_divisible_by_1_to_10_l3906_390612

theorem smallest_divisible_by_1_to_10 : ∃ n : ℕ+, (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ n) ∧
  (∀ m : ℕ+, (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ m) → n ≤ m) ∧ n = 2520 :=
by sorry

end smallest_divisible_by_1_to_10_l3906_390612


namespace triangle_inequality_l3906_390634

theorem triangle_inequality (a b c n : ℝ) (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) (h_n : 1 ≤ n) :
  let s := (a + b + c) / 2
  (a^n / (b + c) + b^n / (c + a) + c^n / (a + b)) ≥ (2/3)^(n-2) * s^(n-1) := by
sorry

end triangle_inequality_l3906_390634


namespace area_equality_l3906_390637

/-- Given a function g defined on {a, b, c}, prove that the area of the triangle
    formed by y = 3g(3x) is equal to the area of the triangle formed by y = g(x) -/
theorem area_equality (g : ℝ → ℝ) (a b c : ℝ) (area : ℝ) 
    (h1 : Set.range g = {g a, g b, g c})
    (h2 : area = 50)
    (h3 : area = abs ((b - a) * (g c - g a) - (c - a) * (g b - g a)) / 2) :
  abs ((b/3 - a/3) * (3 * g c - 3 * g a) - (c/3 - a/3) * (3 * g b - 3 * g a)) / 2 = area := by
  sorry

end area_equality_l3906_390637


namespace smallest_k_for_64_power_gt_4_power_22_l3906_390638

theorem smallest_k_for_64_power_gt_4_power_22 : 
  ∃ k : ℕ, (∀ m : ℕ, 64^m > 4^22 → k ≤ m) ∧ 64^k > 4^22 :=
by
  -- The proof goes here
  sorry

end smallest_k_for_64_power_gt_4_power_22_l3906_390638


namespace sin_cos_product_l3906_390603

theorem sin_cos_product (α : Real) (h : Real.sin α + Real.cos α = Real.sqrt 2) : 
  Real.sin α * Real.cos α = 1/2 := by
  sorry

end sin_cos_product_l3906_390603


namespace one_quarter_between_thirds_l3906_390690

theorem one_quarter_between_thirds (x : ℚ) : 
  (x = 1/3 + 1/4 * (2/3 - 1/3)) → x = 5/12 := by
sorry

end one_quarter_between_thirds_l3906_390690


namespace greatest_prime_factor_of_factorial_sum_l3906_390625

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m > 0 → m < p → p % m ≠ 0

theorem greatest_prime_factor_of_factorial_sum :
  ∃ (p : ℕ), is_prime p ∧ 
    p ∣ (factorial 15 + factorial 18) ∧ 
    ∀ (q : ℕ), is_prime q → q ∣ (factorial 15 + factorial 18) → q ≤ p :=
  sorry

end greatest_prime_factor_of_factorial_sum_l3906_390625


namespace beavers_swimming_l3906_390610

theorem beavers_swimming (initial_beavers final_beavers : ℕ) : 
  initial_beavers ≥ final_beavers → 
  initial_beavers - final_beavers = initial_beavers - final_beavers :=
by
  sorry

#check beavers_swimming 2 1

end beavers_swimming_l3906_390610


namespace isosceles_triangle_property_l3906_390604

/-- Represents a triangle with vertices A, B, C and incentre I -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  I : ℝ × ℝ

/-- The distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- The squared distance between two points -/
def distanceSquared (p q : ℝ × ℝ) : ℝ := sorry

/-- Check if a triangle is isosceles with AB = AC -/
def isIsosceles (t : Triangle) : Prop :=
  distance t.A t.B = distance t.A t.C

/-- The distance from a point to a line defined by two points -/
def distanceToLine (p : ℝ × ℝ) (q r : ℝ × ℝ) : ℝ := sorry

/-- Theorem: In an isosceles triangle ABC with incentre I, 
    if AB = AC, AI = 3, and the distance from I to BC is 2, then BC² = 80 -/
theorem isosceles_triangle_property (t : Triangle) :
  isIsosceles t →
  distance t.A t.I = 3 →
  distanceToLine t.I t.B t.C = 2 →
  distanceSquared t.B t.C = 80 := by sorry

end isosceles_triangle_property_l3906_390604


namespace max_good_quadratics_less_than_500_l3906_390675

/-- A good quadratic trinomial has distinct coefficients and two distinct real roots -/
def is_good_quadratic (a b c : ℕ+) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (b.val : ℝ)^2 > 4 * (a.val : ℝ) * (c.val : ℝ)

/-- The set of 10 positive integers from which coefficients are chosen -/
def coefficient_set : Finset ℕ+ :=
  sorry

/-- The set of all good quadratic trinomials formed from the coefficient set -/
def good_quadratics : Finset (ℕ+ × ℕ+ × ℕ+) :=
  sorry

theorem max_good_quadratics_less_than_500 :
  Finset.card good_quadratics < 500 :=
sorry

end max_good_quadratics_less_than_500_l3906_390675


namespace black_midwest_percentage_is_31_l3906_390682

/-- Represents the population data for different ethnic groups in different regions --/
structure PopulationData :=
  (ne_white : ℕ) (mw_white : ℕ) (south_white : ℕ) (west_white : ℕ)
  (ne_black : ℕ) (mw_black : ℕ) (south_black : ℕ) (west_black : ℕ)
  (ne_asian : ℕ) (mw_asian : ℕ) (south_asian : ℕ) (west_asian : ℕ)
  (ne_hispanic : ℕ) (mw_hispanic : ℕ) (south_hispanic : ℕ) (west_hispanic : ℕ)

/-- Calculates the percentage of Black population in the Midwest --/
def black_midwest_percentage (data : PopulationData) : ℚ :=
  let total_black := data.ne_black + data.mw_black + data.south_black + data.west_black
  (data.mw_black : ℚ) / total_black * 100

/-- Rounds a rational number to the nearest integer --/
def round_to_nearest (q : ℚ) : ℤ :=
  ⌊q + 1/2⌋

/-- The main theorem stating that the rounded percentage of Black population in the Midwest is 31% --/
theorem black_midwest_percentage_is_31 (data : PopulationData) 
  (h : data = { ne_white := 45, mw_white := 55, south_white := 60, west_white := 40,
                ne_black := 6, mw_black := 12, south_black := 18, west_black := 3,
                ne_asian := 2, mw_asian := 2, south_asian := 2, west_asian := 5,
                ne_hispanic := 2, mw_hispanic := 3, south_hispanic := 4, west_hispanic := 6 }) :
  round_to_nearest (black_midwest_percentage data) = 31 := by
  sorry

end black_midwest_percentage_is_31_l3906_390682


namespace area_of_quadrilateral_l3906_390678

-- Define the lines
def line1 (x : ℝ) : ℝ := 3 * x - 3
def line2 (x : ℝ) : ℝ := -2 * x + 14
def line3 : ℝ := 0
def line4 : ℝ := 5

-- Define the vertices of the quadrilateral
def vertex1 : ℝ × ℝ := (0, line1 0)
def vertex2 : ℝ × ℝ := (0, line2 0)
def vertex3 : ℝ × ℝ := (line4, line1 line4)
def vertex4 : ℝ × ℝ := (line4, line2 line4)

-- Define the area of the quadrilateral
def quadrilateralArea : ℝ := 80

-- Theorem statement
theorem area_of_quadrilateral :
  let vertices := [vertex1, vertex2, vertex3, vertex4]
  quadrilateralArea = 80 := by sorry

end area_of_quadrilateral_l3906_390678


namespace danny_found_seven_caps_l3906_390607

/-- The number of bottle caps Danny found at the park -/
def bottleCapsFound (initialCaps currentCaps : ℕ) : ℕ :=
  currentCaps - initialCaps

/-- Proof that Danny found 7 bottle caps at the park -/
theorem danny_found_seven_caps : bottleCapsFound 25 32 = 7 := by
  sorry

end danny_found_seven_caps_l3906_390607


namespace quadrilateral_cyclic_l3906_390654

-- Define the points
variable (A B C D P O B' D' X : EuclideanPlane)

-- Define the conditions
def is_quadrilateral (A B C D : EuclideanPlane) : Prop := sorry

def is_intersection (P : EuclideanPlane) (AB CD : Set EuclideanPlane) : Prop := sorry

def is_perpendicular_bisector_intersection (O : EuclideanPlane) (AB CD : Set EuclideanPlane) : Prop := sorry

def not_on_line (O : EuclideanPlane) (AB : Set EuclideanPlane) : Prop := sorry

def is_reflection (B' : EuclideanPlane) (B : EuclideanPlane) (OP : Set EuclideanPlane) : Prop := sorry

def meet_on_line (AB' CD' OP : Set EuclideanPlane) : Prop := sorry

def is_cyclic (A B C D : EuclideanPlane) : Prop := sorry

-- State the theorem
theorem quadrilateral_cyclic 
  (h1 : is_quadrilateral A B C D)
  (h2 : is_intersection P {A, B} {C, D})
  (h3 : is_perpendicular_bisector_intersection O {A, B} {C, D})
  (h4 : not_on_line O {A, B})
  (h5 : not_on_line O {C, D})
  (h6 : is_reflection B' B {O, P})
  (h7 : is_reflection D' D {O, P})
  (h8 : meet_on_line {A, B'} {C, D'} {O, P}) :
  is_cyclic A B C D :=
sorry

end quadrilateral_cyclic_l3906_390654


namespace polyhedron_inequalities_l3906_390643

/-- A simply connected polyhedron -/
structure SimplyConnectedPolyhedron where
  B : ℕ  -- number of vertices
  P : ℕ  -- number of edges
  G : ℕ  -- number of faces
  euler : B - P + G = 2  -- Euler's formula
  edge_face : P ≥ 3 * G / 2  -- each face has at least 3 edges, each edge is shared by 2 faces
  edge_vertex : P ≥ 3 * B / 2  -- each vertex is connected to at least 3 edges

/-- Theorem stating the inequalities for a simply connected polyhedron -/
theorem polyhedron_inequalities (poly : SimplyConnectedPolyhedron) :
  (3 / 2 : ℝ) ≤ (poly.P : ℝ) / poly.B ∧ (poly.P : ℝ) / poly.B < 3 ∧
  (3 / 2 : ℝ) ≤ (poly.P : ℝ) / poly.G ∧ (poly.P : ℝ) / poly.G < 3 :=
by sorry

end polyhedron_inequalities_l3906_390643


namespace water_price_solution_l3906_390671

/-- Represents the problem of calculating water price per gallon -/
def water_price_problem (gallons_per_inch : ℝ) (monday_rain : ℝ) (tuesday_rain : ℝ) (total_revenue : ℝ) : Prop :=
  let total_gallons := gallons_per_inch * (monday_rain + tuesday_rain)
  let price_per_gallon := total_revenue / total_gallons
  price_per_gallon = 1.20

/-- The main theorem stating the solution to the water pricing problem -/
theorem water_price_solution :
  water_price_problem 15 4 3 126 := by
  sorry

#check water_price_solution

end water_price_solution_l3906_390671


namespace complex_real_condition_l3906_390652

theorem complex_real_condition (z m : ℂ) : z = (1 + Complex.I) * (1 + m * Complex.I) ∧ z.im = 0 → m = -1 := by
  sorry

end complex_real_condition_l3906_390652


namespace cell_population_after_10_days_l3906_390683

/-- The number of cells in a colony after a given number of days, 
    where the initial population is 5 cells and the population triples every 3 days. -/
def cell_population (days : ℕ) : ℕ :=
  5 * 3^(days / 3)

/-- Theorem stating that the cell population after 10 days is 135 cells. -/
theorem cell_population_after_10_days : cell_population 10 = 135 := by
  sorry

end cell_population_after_10_days_l3906_390683


namespace fixed_points_bound_l3906_390650

/-- A polynomial with integer coefficients -/
def IntPolynomial (n : ℕ) := Fin n → ℤ

/-- The degree of a polynomial -/
def degree (p : IntPolynomial n) : ℕ := n

/-- Evaluate a polynomial at a point -/
def evalPoly (p : IntPolynomial n) (x : ℤ) : ℤ := sorry

/-- Compose a polynomial with itself k times -/
def composeK (p : IntPolynomial n) (k : ℕ) : IntPolynomial n := sorry

/-- The number of integer fixed points of a polynomial -/
def numIntFixedPoints (p : IntPolynomial n) : ℕ := sorry

/-- Main theorem: The number of integer fixed points of Q is at most n -/
theorem fixed_points_bound (n k : ℕ) (p : IntPolynomial n) 
  (h1 : n > 1) (h2 : k > 0) : 
  numIntFixedPoints (composeK p k) ≤ n := by sorry

end fixed_points_bound_l3906_390650


namespace melanie_dimes_count_l3906_390627

/-- Calculates the final number of dimes Melanie has -/
def final_dimes (initial : ℕ) (given_away : ℕ) (received : ℕ) : ℕ :=
  initial - given_away + received

/-- Theorem: The final number of dimes is correct given the problem conditions -/
theorem melanie_dimes_count : final_dimes 8 7 4 = 5 := by
  sorry

end melanie_dimes_count_l3906_390627


namespace cos_sin_18_equality_l3906_390632

theorem cos_sin_18_equality :
  let cos_18 : ℝ := (Real.sqrt 5 + 1) / 4
  let sin_18 : ℝ := (Real.sqrt 5 - 1) / 4
  4 * cos_18^2 - 1 = 1 / (4 * sin_18^2) :=
by sorry

end cos_sin_18_equality_l3906_390632


namespace complex_fraction_evaluation_l3906_390626

theorem complex_fraction_evaluation : 
  (2 + 2)^2 / 2^2 * (3 + 3 + 3 + 3)^3 / (3 + 3 + 3)^3 * (6 + 6 + 6 + 6 + 6 + 6)^6 / (6 + 6 + 6 + 6)^6 = 108 := by
  sorry

end complex_fraction_evaluation_l3906_390626


namespace inverse_proportion_change_l3906_390689

theorem inverse_proportion_change (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = c) :
  let a' := 1.2 * a
  let b' := 80
  a' * b' = c →
  b = 96 := by
sorry

end inverse_proportion_change_l3906_390689


namespace simplify_expression_l3906_390606

theorem simplify_expression : 
  (((81 : ℝ) ^ (1/4 : ℝ)) + (Real.sqrt (8 + 3/4)))^2 = (71 + 12 * Real.sqrt 35) / 4 := by
  sorry

end simplify_expression_l3906_390606


namespace cubic_inequality_l3906_390695

theorem cubic_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^3 + b^3 + c^3 + 3*a*b*c > a*b*(a+b) + b*c*(b+c) + a*c*(a+c) := by
  sorry

end cubic_inequality_l3906_390695


namespace cards_distribution_l3906_390635

theorem cards_distribution (total_cards : Nat) (num_people : Nat) 
  (h1 : total_cards = 60) 
  (h2 : num_people = 9) : 
  (num_people - (total_cards % num_people)) = 3 := by
  sorry

end cards_distribution_l3906_390635


namespace complex_roots_circle_l3906_390696

theorem complex_roots_circle (z : ℂ) : 
  (z + 1)^6 = 243 * z^6 → Complex.abs (z - Complex.ofReal (1/8)) = 1/8 := by
  sorry

end complex_roots_circle_l3906_390696


namespace food_shelf_life_l3906_390665

/-- The shelf life function for a food product -/
noncomputable def shelf_life (k b : ℝ) (x : ℝ) : ℝ := Real.exp (k * x + b)

/-- Theorem stating the shelf life at 30°C and the maximum temperature for 80 hours shelf life -/
theorem food_shelf_life (k b : ℝ) :
  (shelf_life k b 0 = 160) →
  (shelf_life k b 20 = 40) →
  (shelf_life k b 30 = 20) ∧
  (∀ x : ℝ, shelf_life k b x ≥ 80 ↔ x ≤ 10) := by
  sorry


end food_shelf_life_l3906_390665


namespace cube_edge_sum_l3906_390644

/-- Given a cube with surface area 486 square centimeters, 
    prove that the sum of the lengths of all its edges is 108 centimeters. -/
theorem cube_edge_sum (surface_area : ℝ) (h : surface_area = 486) : 
  ∃ (edge_length : ℝ), 
    surface_area = 6 * edge_length^2 ∧ 
    12 * edge_length = 108 :=
by sorry

end cube_edge_sum_l3906_390644


namespace largest_angle_in_special_triangle_l3906_390616

theorem largest_angle_in_special_triangle : ∀ (a b c : ℝ),
  -- Two angles sum to 7/5 of a right angle
  a + b = 7/5 * 90 →
  -- One angle is 20° larger than the other
  b = a + 20 →
  -- All angles are non-negative
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 →
  -- Sum of all angles in a triangle is 180°
  a + b + c = 180 →
  -- The largest angle is 73°
  max a (max b c) = 73 :=
by
  sorry

end largest_angle_in_special_triangle_l3906_390616


namespace sqrt_12_minus_sqrt_3_equals_sqrt_3_l3906_390623

theorem sqrt_12_minus_sqrt_3_equals_sqrt_3 : 
  Real.sqrt 12 - Real.sqrt 3 = Real.sqrt 3 := by
  sorry

end sqrt_12_minus_sqrt_3_equals_sqrt_3_l3906_390623


namespace centroid_trajectory_l3906_390622

/-- The trajectory of the centroid of a triangle ABC, where A and B are fixed points
    and C moves on a hyperbola. -/
theorem centroid_trajectory
  (A B C : ℝ × ℝ)  -- Vertices of the triangle
  (x y : ℝ)        -- Coordinates of the centroid
  (h1 : A = (0, 0))
  (h2 : B = (6, 0))
  (h3 : (C.1^2 / 16) - (C.2^2 / 9) = 1)  -- C moves on the hyperbola
  (h4 : x = (A.1 + B.1 + C.1) / 3)       -- Centroid x-coordinate
  (h5 : y = (A.2 + B.2 + C.2) / 3)       -- Centroid y-coordinate
  (h6 : y ≠ 0) :
  9 * (x - 2)^2 / 16 - y^2 = 1 :=
sorry

end centroid_trajectory_l3906_390622


namespace charcoal_drawings_l3906_390694

theorem charcoal_drawings (total : Nat) (colored_pencil : Nat) (blending_marker : Nat) :
  total = 60 → colored_pencil = 24 → blending_marker = 19 →
  total - colored_pencil - blending_marker = 17 := by
  sorry

end charcoal_drawings_l3906_390694


namespace target_hit_probability_l3906_390619

theorem target_hit_probability (p1 p2 : ℝ) (h1 : p1 = 0.5) (h2 : p2 = 0.7) :
  1 - (1 - p1) * (1 - p2) = 0.85 := by
  sorry

end target_hit_probability_l3906_390619


namespace outfit_combinations_l3906_390624

/-- The number of possible outfits given the number of shirts, ties, and pants -/
def number_of_outfits (shirts ties pants : ℕ) : ℕ := shirts * ties * pants

/-- Theorem: Given 8 shirts, 6 ties, and 4 pairs of pants, the number of possible outfits is 192 -/
theorem outfit_combinations : number_of_outfits 8 6 4 = 192 := by
  sorry

end outfit_combinations_l3906_390624


namespace sufficient_not_necessary_condition_l3906_390661

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b : ℝ, a > b ∧ b > 0 → 1 / a < 1 / b) ∧
  (∃ a b : ℝ, 1 / a < 1 / b ∧ ¬(a > b ∧ b > 0)) :=
by sorry

end sufficient_not_necessary_condition_l3906_390661


namespace class_weighted_average_l3906_390666

/-- Calculates the weighted average score for a class with three groups of students -/
theorem class_weighted_average (total_students : ℕ) 
  (group1_count : ℕ) (group1_avg : ℚ)
  (group2_count : ℕ) (group2_avg : ℚ)
  (group3_count : ℕ) (group3_avg : ℚ)
  (h1 : total_students = group1_count + group2_count + group3_count)
  (h2 : total_students = 30)
  (h3 : group1_count = 12)
  (h4 : group2_count = 10)
  (h5 : group3_count = 8)
  (h6 : group1_avg = 72 / 100)
  (h7 : group2_avg = 85 / 100)
  (h8 : group3_avg = 92 / 100) :
  (group1_count * group1_avg + 2 * group2_count * group2_avg + group3_count * group3_avg) / 
  (group1_count + 2 * group2_count + group3_count) = 825 / 1000 := by
  sorry


end class_weighted_average_l3906_390666


namespace abs_sqrt_mul_eq_three_l3906_390687

theorem abs_sqrt_mul_eq_three : |(-3 : ℤ)| + Real.sqrt 4 + (-2 : ℤ) * (1 : ℤ) = 3 := by
  sorry

end abs_sqrt_mul_eq_three_l3906_390687


namespace power_equality_l3906_390629

theorem power_equality (x : ℝ) : (1/8 : ℝ) * 2^50 = 4^x → x = 23.5 := by
  sorry

end power_equality_l3906_390629


namespace cone_section_area_l3906_390615

-- Define the cone structure
structure Cone where
  -- Axial section is an isosceles right triangle
  axial_section_isosceles_right : Bool
  -- Hypotenuse of axial section
  hypotenuse : ℝ
  -- Angle between section and base
  α : ℝ

-- Define the theorem
theorem cone_section_area (c : Cone) 
  (h1 : c.axial_section_isosceles_right = true) 
  (h2 : c.hypotenuse = 2) 
  (h3 : 0 < c.α ∧ c.α < π / 2) : 
  ∃ (area : ℝ), area = (Real.sqrt 2 / 2) * (1 / (Real.cos c.α)^2) :=
sorry

end cone_section_area_l3906_390615


namespace sum_of_coefficients_is_negative_23_l3906_390608

-- Define the polynomial
def p (x : ℝ) : ℝ := 4 * (2 * x^8 + 5 * x^5 - 6) + 9 * (x^6 - 8 * x^3 + 4)

-- Theorem statement
theorem sum_of_coefficients_is_negative_23 :
  p 1 = -23 := by sorry

end sum_of_coefficients_is_negative_23_l3906_390608


namespace eighteenth_replacement_in_december_l3906_390674

/-- Represents months as integers from 1 to 12 -/
def Month := Fin 12

/-- Convert a number of months to a Month value -/
def monthsToMonth (n : ℕ) : Month :=
  ⟨(n - 1) % 12 + 1, by sorry⟩

/-- January represented as a Month -/
def january : Month := ⟨1, by sorry⟩

/-- December represented as a Month -/
def december : Month := ⟨12, by sorry⟩

/-- The number of months between replacements -/
def replacementInterval : ℕ := 7

/-- The number of the replacement we're interested in -/
def targetReplacement : ℕ := 18

theorem eighteenth_replacement_in_december :
  monthsToMonth (replacementInterval * (targetReplacement - 1) + 1) = december := by
  sorry

end eighteenth_replacement_in_december_l3906_390674


namespace purely_imaginary_complex_number_l3906_390691

theorem purely_imaginary_complex_number (i : ℂ) (a : ℝ) : 
  i * i = -1 →
  (∃ (b : ℝ), (1 + a * i) / (2 - i) = b * i) →
  a = 2 := by
sorry

end purely_imaginary_complex_number_l3906_390691


namespace inequality_proof_l3906_390693

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) : 
  (a * b + b * c + a * c ≤ 1 / 3) ∧ 
  (a^2 / b + b^2 / c + c^2 / a ≥ 1) :=
by sorry

end inequality_proof_l3906_390693


namespace unique_intersection_point_l3906_390655

-- Define the three lines
def line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 4
def line2 (x y : ℝ) : Prop := x + 4 * y = 1
def line3 (x y : ℝ) : Prop := 6 * x - 4 * y = 8

-- Define a point that lies on at least two of the lines
def intersection_point (p : ℝ × ℝ) : Prop :=
  (line1 p.1 p.2 ∧ line2 p.1 p.2) ∨
  (line1 p.1 p.2 ∧ line3 p.1 p.2) ∨
  (line2 p.1 p.2 ∧ line3 p.1 p.2)

-- Theorem stating that there is exactly one intersection point
theorem unique_intersection_point :
  ∃! p : ℝ × ℝ, intersection_point p :=
sorry

end unique_intersection_point_l3906_390655


namespace hyperbola_n_range_l3906_390676

-- Define the hyperbola equation
def hyperbola_equation (x y m n : ℝ) : Prop :=
  x^2 / (m^2 + n) - y^2 / (3 * m^2 - n) = 1

-- Define the distance between foci
def foci_distance : ℝ := 4

-- Theorem statement
theorem hyperbola_n_range (x y m n : ℝ) :
  hyperbola_equation x y m n ∧ 
  (∃ (a b : ℝ), (a - b)^2 = foci_distance^2) →
  -1 < n ∧ n < 3 :=
sorry

end hyperbola_n_range_l3906_390676


namespace cookie_calorie_consumption_l3906_390663

/-- Represents the number of calories in a single cookie of each type -/
structure CookieCalories where
  caramel : ℕ
  chocolate_chip : ℕ
  peanut_butter : ℕ

/-- Represents the number of cookies selected of each type -/
structure SelectedCookies where
  caramel : ℕ
  chocolate_chip : ℕ
  peanut_butter : ℕ

/-- Calculates the total calories consumed based on the number of cookies selected and their calorie content -/
def totalCalories (calories : CookieCalories) (selected : SelectedCookies) : ℕ :=
  calories.caramel * selected.caramel +
  calories.chocolate_chip * selected.chocolate_chip +
  calories.peanut_butter * selected.peanut_butter

/-- Proves that selecting 5 caramel, 3 chocolate chip, and 2 peanut butter cookies results in consuming 204 calories -/
theorem cookie_calorie_consumption :
  let calories := CookieCalories.mk 18 22 24
  let selected := SelectedCookies.mk 5 3 2
  totalCalories calories selected = 204 := by
  sorry

end cookie_calorie_consumption_l3906_390663


namespace sqrt_equation_solution_l3906_390641

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (1 - 4 * x) = 5 → x = -6 := by
  sorry

end sqrt_equation_solution_l3906_390641


namespace intersection_points_l3906_390642

theorem intersection_points (a : ℝ) : 
  (∃! p : ℝ × ℝ, (p.2 = a * p.1 + a ∧ p.2 = p.1 ∧ p.2 = 2 - 2 * a * p.1)) ↔ 
  (a = 1/2 ∨ a = -2) := by
  sorry

end intersection_points_l3906_390642


namespace intersection_of_A_and_B_l3906_390633

-- Define the sets A and B
def A : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2}
def B : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt (1 - x^2)}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Icc 0 1 := by
  sorry

end intersection_of_A_and_B_l3906_390633


namespace sum_of_multiples_is_even_l3906_390677

theorem sum_of_multiples_is_even (a b : ℤ) (ha : 4 ∣ a) (hb : 6 ∣ b) : Even (a + b) := by
  sorry

end sum_of_multiples_is_even_l3906_390677


namespace count_possible_sums_l3906_390648

/-- The set of integers from 1 to 150 -/
def S : Finset ℕ := Finset.range 150

/-- The size of subset C -/
def k : ℕ := 80

/-- The minimum possible sum of k elements from S -/
def min_sum : ℕ := k * (k + 1) / 2

/-- The maximum possible sum of k elements from S -/
def max_sum : ℕ := (Finset.sum S id - (150 - k) * (150 - k + 1) / 2)

/-- The number of possible values for the sum of k elements from S -/
def num_possible_sums : ℕ := max_sum - min_sum + 1

theorem count_possible_sums :
  num_possible_sums = 6844 := by sorry

end count_possible_sums_l3906_390648


namespace problem_1_problem_2_l3906_390645

-- Problem 1
theorem problem_1 (x : ℝ) : 4 * (x + 1)^2 - (2 * x + 5) * (2 * x - 5) = 8 * x + 29 := by
  sorry

-- Problem 2
theorem problem_2 (a b c : ℝ) : 
  (4 * a * b)^2 * (-1/4 * a^4 * b^3 * c^2) / (-4 * a^3 * b^2 * c^2) = a^3 * b^3 := by
  sorry

end problem_1_problem_2_l3906_390645


namespace paul_prediction_accuracy_l3906_390605

/-- Represents a team in the FIFA World Cup -/
inductive Team
| Ghana
| Bolivia
| Argentina
| France

/-- The probability of a team winning the tournament -/
def winProbability (t : Team) : ℚ :=
  match t with
  | Team.Ghana => 1/2
  | Team.Bolivia => 1/6
  | Team.Argentina => 1/6
  | Team.France => 1/6

/-- The probability of Paul correctly predicting the winner -/
def paulCorrectProbability : ℚ :=
  (winProbability Team.Ghana)^2 +
  (winProbability Team.Bolivia)^2 +
  (winProbability Team.Argentina)^2 +
  (winProbability Team.France)^2

theorem paul_prediction_accuracy :
  paulCorrectProbability = 1/3 := by
  sorry

end paul_prediction_accuracy_l3906_390605


namespace correct_stratified_sample_l3906_390651

/-- Represents the number of employees in each job category -/
structure EmployeeCount where
  total : ℕ
  senior : ℕ
  midLevel : ℕ
  junior : ℕ

/-- Represents the number of sampled employees in each job category -/
structure SampleCount where
  senior : ℕ
  midLevel : ℕ
  junior : ℕ

/-- Checks if the sample counts are correct for stratified sampling -/
def isCorrectStratifiedSample (ec : EmployeeCount) (sc : SampleCount) (sampleSize : ℕ) : Prop :=
  sc.senior = ec.senior * sampleSize / ec.total ∧
  sc.midLevel = ec.midLevel * sampleSize / ec.total ∧
  sc.junior = ec.junior * sampleSize / ec.total

theorem correct_stratified_sample :
  let ec : EmployeeCount := ⟨450, 45, 135, 270⟩
  let sc : SampleCount := ⟨3, 9, 18⟩
  isCorrectStratifiedSample ec sc 30 := by sorry

end correct_stratified_sample_l3906_390651


namespace subtraction_of_fractions_l3906_390699

theorem subtraction_of_fractions : (5 : ℚ) / 6 - (1 : ℚ) / 3 = (1 : ℚ) / 2 := by
  sorry

end subtraction_of_fractions_l3906_390699


namespace solution_p_proportion_l3906_390647

/-- Represents a solution mixture with lemonade and carbonated water -/
structure Solution where
  lemonade : ℝ
  carbonated_water : ℝ
  sum_to_one : lemonade + carbonated_water = 1

/-- The final mixture of solutions P and Q -/
structure Mixture where
  p : ℝ
  q : ℝ
  sum_to_one : p + q = 1

/-- Given two solutions and their mixture, prove that the proportion of Solution P is 0.4 -/
theorem solution_p_proportion
  (P : Solution)
  (Q : Solution)
  (M : Mixture)
  (h_P : P.carbonated_water = 0.8)
  (h_Q : Q.carbonated_water = 0.55)
  (h_M : P.carbonated_water * M.p + Q.carbonated_water * M.q = 0.65) :
  M.p = 0.4 := by
sorry

end solution_p_proportion_l3906_390647


namespace negation_of_existence_rational_sqrt_two_l3906_390679

theorem negation_of_existence_rational_sqrt_two :
  (¬ ∃ (x : ℚ), x^2 - 2 = 0) ↔ (∀ (x : ℚ), x^2 - 2 ≠ 0) := by sorry

end negation_of_existence_rational_sqrt_two_l3906_390679


namespace monomial_division_equality_l3906_390628

theorem monomial_division_equality (x y : ℝ) (m n : ℤ) :
  (x^m * y^n) / ((1/4) * x^3 * y) = 4 * x^2 ↔ m = 5 ∧ n = 1 := by
  sorry

end monomial_division_equality_l3906_390628


namespace triplet_satisfies_conditions_l3906_390630

/-- Checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- Checks if three numbers form a geometric sequence -/
def isGeometricSequence (a b c : ℕ) : Prop :=
  ∃ r : ℚ, r > 0 ∧ b = a * r ∧ c = b * r

theorem triplet_satisfies_conditions : 
  isPrime 17 ∧ isPrime 23 ∧ isPrime 31 ∧
  17 < 23 ∧ 23 < 31 ∧ 31 < 100 ∧
  isGeometricSequence 18 24 32 :=
by sorry

end triplet_satisfies_conditions_l3906_390630


namespace blood_pressure_analysis_l3906_390659

def systolic_pressure : List ℝ := [151, 148, 140, 139, 140, 136, 140]
def diastolic_pressure : List ℝ := [90, 92, 88, 88, 90, 80, 88]

def median (l : List ℝ) : ℝ := sorry
def mode (l : List ℝ) : ℝ := sorry
def mean (l : List ℝ) : ℝ := sorry
def variance (l : List ℝ) : ℝ := sorry

theorem blood_pressure_analysis :
  (median systolic_pressure = 140) ∧
  (mode diastolic_pressure = 88) ∧
  (mean systolic_pressure = 142) ∧
  (variance diastolic_pressure = 88 / 7) :=
by sorry

end blood_pressure_analysis_l3906_390659


namespace vector_equation_solution_l3906_390664

theorem vector_equation_solution :
  let a : Fin 2 → ℝ := ![2, 1]
  let b : Fin 2 → ℝ := ![1, -2]
  ∀ m n : ℝ, (m • a + n • b = ![9, -8]) → (m - n = -3) := by
sorry

end vector_equation_solution_l3906_390664


namespace optimal_hospital_location_l3906_390680

/-- Given three points A, B, and C in a plane, with AB = AC = 13 and BC = 10,
    prove that the point P(0, 4) on the perpendicular bisector of BC
    minimizes the sum of squares of distances PA^2 + PB^2 + PC^2 -/
theorem optimal_hospital_location (A B C P : ℝ × ℝ) :
  A = (0, 12) →
  B = (-5, 0) →
  C = (5, 0) →
  P.1 = 0 →
  (∀ y : ℝ, (0, y).1^2 + (0, y).2^2 + (-5 - 0)^2 + (0 - y)^2 + (5 - 0)^2 + (0 - y)^2 ≥
             (0, 4).1^2 + (0, 4).2^2 + (-5 - 0)^2 + (0 - 4)^2 + (5 - 0)^2 + (0 - 4)^2) →
  P = (0, 4) :=
by sorry

end optimal_hospital_location_l3906_390680


namespace sandbox_area_l3906_390698

/-- The area of a rectangle with length 312 cm and width 146 cm is 45552 square centimeters. -/
theorem sandbox_area :
  let length : ℕ := 312
  let width : ℕ := 146
  length * width = 45552 := by
  sorry

end sandbox_area_l3906_390698


namespace smallest_three_digit_number_l3906_390640

def digits : Finset Nat := {3, 0, 2, 5, 7}

def is_valid_number (n : Nat) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  ∃ (a b c : Nat), a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  n = 100 * a + 10 * b + c

theorem smallest_three_digit_number :
  ∀ n, is_valid_number n → n ≥ 203 :=
by sorry

end smallest_three_digit_number_l3906_390640
